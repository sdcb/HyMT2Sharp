using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Backends.Metal;

/// <summary>
/// Metal decode backend (M2): one full decode step encoded as a flat sequence
/// of compute dispatches into a single command buffer per token. Weights live
/// in shared-storage MTLBuffers uploaded once at load; activation/KV buffers
/// persist on device. Only the final logits row comes back to the host.
/// Supported weight types: Q4_K + F32 (Q4_K_M model); flat bf16 KV capped at
/// ContextLength (paged KV is M4).
/// </summary>
public sealed unsafe class MetalBackend : IComputeBackend
{
    private readonly MtlDevice _dev;
    private ModelConfig _cfg = null!;
    private readonly Dictionary<string, IntPtr> _w = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgmlTensorType> _wtype = new(StringComparer.Ordinal);
    private IntPtr _psoGemv, _psoGemvQ6, _psoEmbedQ4, _psoEmbedQ6, _psoRms, _psoRope, _psoKvAppend, _psoAttn, _psoSilu, _psoAdd;
    private IntPtr[] _kvK = null!, _kvV = null!;
    private int _kvCap;
    private int _kvStride;
    private IntPtr _embd, _outNorm;
    private IntPtr _h, _n1, _n2, _q, _k, _v, _ao, _attnOut, _gate, _up, _down, _normed, _logitsBuf, _tok;
    private float[] _logitsHost = null!;
    private int _vocab;
    private readonly bool _timing = Environment.GetEnvironmentVariable("HYMT_METAL_TIMING") == "1";
    private readonly Stopwatch _sw = new();
    private double _encMs, _gpuMs;
    private int _n;

    public string Name => "metal";

    public MetalBackend()
    {
        _dev = MtlDevice.Create();
    }

    private IntPtr W(string name) => _w.TryGetValue(name, out IntPtr b) ? b : 0;

    private static void* Contents(IntPtr buf) => (void*)ObjC.Send0(buf, ObjC.Sel("contents"));

    private IntPtr NewPso(IntPtr lib, string name) =>
        _dev.NewPso(_dev.NewFunction(lib, name));

    public void LoadModel(GgufFile gguf, ModelConfig cfg)
    {
        _cfg = cfg;
        IntPtr lib1 = _dev.NewLibraryFromSource(MslKernels.Source);
        IntPtr lib2 = _dev.NewLibraryFromSource(MslDecodeKernels.Source);
        _psoGemv = NewPso(lib1, "q4k_gemv_fast4");
        _psoGemvQ6 = NewPso(lib2, "q6k_gemv_fast4");
        _psoEmbedQ4 = NewPso(lib2, "q4k_embed_row");
        _psoEmbedQ6 = NewPso(lib2, "q6k_embed_row");
        _psoRms = NewPso(lib2, "rmsnorm_rows");
        _psoRope = NewPso(lib2, "rope_neox");
        _psoKvAppend = NewPso(lib2, "kv_append_bf16");
        _psoAttn = NewPso(lib2, "attn_decode");
        _psoSilu = NewPso(lib2, "silu_mul");
        _psoAdd = NewPso(lib2, "add_inplace");

        foreach ((string name, GgufTensorInfo info) in gguf.Tensors)
        {
            ulong bytes = info.Type switch
            {
                GgmlTensorType.F32 => info.NumElements * sizeof(float),
                GgmlTensorType.Q4_K => info.NumElements / 256 * (ulong)144,
                GgmlTensorType.Q6_K => info.NumElements / 256 * (ulong)210,
                _ => throw new NotSupportedException(
                    $"Metal backend (M2) supports F32/Q4_K/Q6_K only; {name} is {info.Type} — use --backend cpu"),
            };
            byte* data = gguf.DataBase + (long)info.Offset;
            _w[name] = _dev.NewBufferBytes(data, (nuint)bytes);
            _wtype[name] = info.Type;
        }

        _embd = W("token_embd.weight");
        if (_embd == 0) throw new KeyNotFoundException("token_embd.weight");
        _outNorm = W("output_norm.weight");
        if (W("output.weight") == 0)
        {
            _w["output.weight"] = _embd;  // tied lm_head
            _wtype["output.weight"] = _wtype["token_embd.weight"];
        }
        if (_outNorm == 0) throw new KeyNotFoundException("output_norm.weight");

        // Buffers: flat bf16 KV (cap = context length, clamped to threadgroup
        // score capacity of attn_decode: kvLen <= 4096)
        _kvStride = cfg.NumKvHeads * cfg.HeadDim;
        _kvCap = Math.Min(cfg.ContextLength, 4096);
        _kvK = new IntPtr[cfg.NumLayers];
        _kvV = new IntPtr[cfg.NumLayers];
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            _kvK[l] = _dev.NewBuffer((nuint)(_kvCap * _kvStride * sizeof(ushort)));
            _kvV[l] = _dev.NewBuffer((nuint)(_kvCap * _kvStride * sizeof(ushort)));
        }

        int hidden = cfg.HiddenSize, ffn = cfg.FfnSize;
        int qDim = cfg.NumHeads * cfg.HeadDim, kDim = _kvStride;
        _vocab = cfg.VocabSize;
        _h = _dev.NewBuffer((nuint)(hidden * 4));
        _n1 = _dev.NewBuffer((nuint)(hidden * 4));
        _n2 = _dev.NewBuffer((nuint)(hidden * 4));
        _q = _dev.NewBuffer((nuint)(qDim * 4));
        _k = _dev.NewBuffer((nuint)(kDim * 4));
        _v = _dev.NewBuffer((nuint)(kDim * 4));
        _ao = _dev.NewBuffer((nuint)(qDim * 4));
        _attnOut = _dev.NewBuffer((nuint)(hidden * 4));
        _gate = _dev.NewBuffer((nuint)(ffn * 4));
        _up = _dev.NewBuffer((nuint)(ffn * 4));
        _down = _dev.NewBuffer((nuint)(hidden * 4));
        _normed = _dev.NewBuffer((nuint)(hidden * 4));
        _logitsBuf = _dev.NewBuffer((nuint)(_vocab * 4));
        _tok = _dev.NewBuffer(4);
        _logitsHost = new float[_vocab];
    }

    public void UploadKv(int layer, int pos, int len, ushort* k, ushort* v)
    {
        if (pos + len > _kvCap)
            throw new NotSupportedException($"Metal KV capacity {_kvCap} exceeded at {pos + len} (M2 flat cap)");
        nuint off = (nuint)(pos * _kvStride * sizeof(ushort));
        nuint bytes = (nuint)(len * _kvStride * sizeof(ushort));
        Buffer.MemoryCopy(k + pos * _kvStride, (byte*)Contents(_kvK[layer]) + off, bytes, bytes);
        Buffer.MemoryCopy(v + pos * _kvStride, (byte*)Contents(_kvV[layer]) + off, bytes, bytes);
        MtlDevice.DidModifyRange(_kvK[layer], off, bytes);
        MtlDevice.DidModifyRange(_kvV[layer], off, bytes);
    }

    public float[] DecodeStep(int tokenId, int pos)
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, heads = c.NumHeads, kvHeads = c.NumKvHeads;
        int dim = c.HeadDim, qDim = heads * dim, kDim = kvHeads * dim, kvLen = pos + 1;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"Metal KV capacity {_kvCap} exceeded at {kvLen} (M2 flat cap)");
        float scale = 1f / MathF.Sqrt(dim);

        using AutoReleasePool pool = AutoReleasePool.Create();
        _sw.Restart();
        *(int*)Contents(_tok) = tokenId;
        MtlDevice.DidModifyRange(_tok, 0, 4);

        var cc = CmdCtx.Begin(_dev.Queue);
        // embed: h = dequant(embd[token])
        cc.SetPso(_wtype["token_embd.weight"] == GgmlTensorType.Q6_K ? _psoEmbedQ6 : _psoEmbedQ4);
        cc.SetBuffer(_embd, 0, 0); cc.SetBuffer(_tok, 0, 1); cc.SetBuffer(_h, 0, 2);
        cc.SetInt(3, hidden);
        cc.Dispatch((nuint)((hidden + 255) / 256), 1, 1, 256, 1, 1);

        for (int l = 0; l < c.NumLayers; l++)
        {
            // n1 = rmsnorm(h, attn_norm)
            Rms(cc, _h, W($"blk.{l}.attn_norm.weight"), _n1, 1, hidden);
            Gemv(cc, $"blk.{l}.attn_q.weight", _n1, _q, hidden, qDim);
            Gemv(cc, $"blk.{l}.attn_k.weight", _n1, _k, hidden, kDim);
            Gemv(cc, $"blk.{l}.attn_v.weight", _n1, _v, hidden, kDim);
            Rope(cc, _q, heads, dim, pos);
            Rope(cc, _k, kvHeads, dim, pos);
            Rms(cc, _q, W($"blk.{l}.attn_q_norm.weight"), _q, heads, dim);
            Rms(cc, _k, W($"blk.{l}.attn_k_norm.weight"), _k, kvHeads, dim);
            // append K/V at pos
            cc.SetPso(_psoKvAppend);
            cc.SetBuffer(_k, 0, 0); cc.SetBuffer(_v, 0, 1);
            cc.SetBuffer(_kvK[l], 0, 2); cc.SetBuffer(_kvV[l], 0, 3);
            cc.SetInt(4, kDim); cc.SetInt(5, _kvStride); cc.SetInt(6, pos);
            cc.Dispatch((nuint)((kDim + 255) / 256), 1, 1, 256, 1, 1);
            // attention -> ao
            cc.SetPso(_psoAttn);
            cc.SetBuffer(_q, 0, 0); cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(_kvV[l], 0, 2);
            cc.SetBuffer(_ao, 0, 3);
            cc.SetInt(4, heads); cc.SetInt(5, kvHeads); cc.SetInt(6, dim);
            cc.SetInt(7, _kvStride); cc.SetInt(8, kvLen); cc.SetFloat(9, scale);
            cc.Dispatch((nuint)heads, 1, 1, 128, 1, 1);
            Gemv(cc, $"blk.{l}.attn_output.weight", _ao, _attnOut, qDim, hidden);
            Add(cc, _h, _attnOut, hidden);

            // FFN
            Rms(cc, _h, W($"blk.{l}.ffn_norm.weight"), _n2, 1, hidden);
            Gemv(cc, $"blk.{l}.ffn_gate.weight", _n2, _gate, hidden, c.FfnSize);
            Gemv(cc, $"blk.{l}.ffn_up.weight", _n2, _up, hidden, c.FfnSize);
            cc.SetPso(_psoSilu);
            cc.SetBuffer(_gate, 0, 0); cc.SetBuffer(_up, 0, 1);
            cc.SetInt(2, c.FfnSize);
            cc.Dispatch((nuint)((c.FfnSize + 255) / 256), 1, 1, 256, 1, 1);
            Gemv(cc, $"blk.{l}.ffn_down.weight", _gate, _down, c.FfnSize, hidden);
            Add(cc, _h, _down, hidden);
        }

        Rms(cc, _h, _outNorm, _normed, 1, hidden);
        Gemv(cc, "output.weight", _normed, _logitsBuf, hidden, _vocab);

        cc.EndEnc();
        cc.Commit();
        double encMs = _timing ? _sw.Elapsed.TotalMilliseconds : 0;
        cc.Wait();
        if (_timing)
        {
            _encMs += encMs; _gpuMs += _sw.Elapsed.TotalMilliseconds - encMs; _n++;
            if (_n % 64 == 0)
            {
                Console.WriteLine($"[metal] enc={_encMs / _n:F2}ms gpu={_gpuMs / _n:F2}ms tok");
                _encMs = _gpuMs = 0; _n = 0;
            }
        }
        cc.Drain();

        Marshal.Copy((IntPtr)Contents(_logitsBuf), _logitsHost, 0, _vocab);
        return _logitsHost;
    }

    private void Gemv(CmdCtx c, string name, IntPtr x, IntPtr y, int inDim, int outDim)
    {
        IntPtr w = W(name);
        bool q6 = _wtype[name] == GgmlTensorType.Q6_K;
        // fast4 kernels stage scales in threadgroup sf[]:
        // q4k -> in_dim <= 6144 (sf[4*192] groups of 32), q6k -> sf[4*384] groups of 16.
        if (inDim > 6144)
            throw new NotSupportedException($"gemv {name}: in_dim {inDim} exceeds fast4 limit 6144");
        c.SetPso(q6 ? _psoGemvQ6 : _psoGemv);
        c.SetBuffer(w, 0, 0); c.SetBuffer(x, 0, 1); c.SetBuffer(y, 0, 2);
        c.SetInt(3, inDim); c.SetInt(4, outDim);
        c.Dispatch((nuint)((outDim + 3) / 4), 1, 1, 256, 1, 1);
    }

    private void Rms(CmdCtx c, IntPtr x, IntPtr w, IntPtr y, int rows, int dim)
    {
        c.SetPso(_psoRms);
        c.SetBuffer(x, 0, 0); c.SetBuffer(w, 0, 1); c.SetBuffer(y, 0, 2);
        c.SetInt(3, dim); c.SetFloat(4, _cfg.Eps);
        c.Dispatch((nuint)rows, 1, 1, 256, 1, 1);
    }

    private void Rope(CmdCtx c, IntPtr x, int heads, int headDim, int pos)
    {
        int half = _cfg.RopeDim / 2;
        c.SetPso(_psoRope);
        c.SetBuffer(x, 0, 0);
        c.SetInt(1, headDim); c.SetInt(2, _cfg.RopeDim); c.SetInt(3, pos); c.SetFloat(4, _cfg.RopeBase);
        c.Dispatch((nuint)((heads * half + 255) / 256), 1, 1, 256, 1, 1);
    }

    private void Add(CmdCtx c, IntPtr a, IntPtr b, int n)
    {
        c.SetPso(_psoAdd);
        c.SetBuffer(a, 0, 0); c.SetBuffer(b, 0, 1);
        c.SetInt(2, n);
        c.Dispatch((nuint)((n + 255) / 256), 1, 1, 256, 1, 1);
    }

    public void Dispose() { /* TODO(M2): release device/queue/buffers on the real backend lifecycle */ }
}
