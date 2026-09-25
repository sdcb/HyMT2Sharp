using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Backends.Cuda.Interop;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Backends.Cuda;

/// <summary>
/// CUDA backend: full forward on device via the vendored TensorSharp PTX
/// kernels (dp4a/MMQ quantized matmul, dequant+cuBLAS F16 GEMM for prefill
/// rows, GQA attention, rmsnorm, NeoX rope, silu-mul). Weights are uploaded
/// once at load; KV is a per-layer head-first f32 cache
/// [kvHead][cap][headDim] capped at 4096 positions (shared-memory bound of
/// the attention kernels). Supported weight types: the ggml-standard set in
/// <see cref="CudaQuantMatmul.SupportsMatmulType"/> — the HyMT2-specific
/// STQ1_0/Q2_0C formats have no CUDA kernels and fall back to
/// <c>--backend cpu</c>. Only the final logits row comes back to the host.
/// </summary>
public sealed unsafe class CudaBackend : IComputeBackend
{
    private readonly CudaEnv _env;
    private ModelConfig _cfg = null!;
    private readonly Dictionary<string, IntPtr> _w = new(StringComparer.Ordinal);
    private readonly Dictionary<string, int> _wtype = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _wbytes = new(StringComparer.Ordinal);
    private IntPtr[] _kvK = null!, _kvV = null!;
    private int _kvCap;
    private int _kvStride;          // kvHeads * headDim (flat row width)
    private int _dim, _heads, _kvHeads, _ropeHalf;
    private IntPtr _ropeCos, _ropeSin;   // [cap][ropeHalf] f32
    private IntPtr _embd, _outNorm;
    private IntPtr _h, _n1, _n2, _qkv, _gu, _ao, _attnOut, _down, _normed, _logitsBuf, _tok, _xq;
    private IntPtr _xqSrc;
    private int _xqSrcDim;
    private float[] _logitsHost = null!;
    private int _vocab;

    // CUDA-graph decode replay: the whole decode step is captured once (after a
    // first scalar step has warmed every lazy scratch allocation) and replayed
    // with one cuGraphLaunch per token. Per-step values travel through pinned
    // host buffers whose contents captured HtoD memcpy nodes re-read at replay:
    // _tokHost (the token id) and _dynHost (the TS dyn convention — slot 0
    // attend_len, 1 kv_write_pos, 3 rope_pos). Kernels read them from the
    // device copies _tok/_dyn; the rope table for the step is rebuilt on-device
    // by ts_fill_neox_rope_tables_dyn_f32 into _ropeCos1/_ropeSin1.
    private bool _useGraph =
        Environment.GetEnvironmentVariable("HYMT2SHARP_CUDA_GRAPH") is not ("0" or "false" or "no");
    private IntPtr _graphExec;
    private int _decodeSteps;
    private IntPtr _dyn;                 // int[4] device dyn buffer
    private IntPtr _ropeFreqs;           // f32[ropeHalf] device frequencies
    private IntPtr _ropeCos1, _ropeSin1; // f32[ropeHalf] single-position tables
    private int* _tokHost;               // pinned token id
    private int* _dynHost;               // pinned dyn values, int[4]
    private GCHandle _logitsPin;         // pinned _logitsHost for captured DtoH

    // HYMT2SHARP_CUDA_TIME=1 diagnostic: GPU-side step time via events.
    // =2 also prints a per-enqueue-op breakdown of one step and exits.
    private readonly int _timeGpu =
        Environment.GetEnvironmentVariable("HYMT2SHARP_CUDA_TIME") is "1" or "2"
            ? int.Parse(Environment.GetEnvironmentVariable("HYMT2SHARP_CUDA_TIME")!)
            : 0;
    private IntPtr _evt0, _evt1;
    private float _gpuAcc; private int _gpuCnt;
    private readonly List<(string label, IntPtr evt)> _marks = new();

    // Prefill scratch grows lazily to the largest T seen (model chunks at 1024).
    private int _pfCap;
    private IntPtr _pfToks, _pfH, _pfN1, _pfN2, _pfQ, _pfQH, _pfK, _pfV,
        _pfAo, _pfAttnOut, _pfGate, _pfUp, _pfDown, _pfNormed;
    // bf16 staging for UploadKv (CPU-prefill written rows).
    private IntPtr _kvStage;
    private long _kvStageBytes;

    public string Name => "cuda";
    public bool SupportsPrefill => true;

    private CudaBackend(CudaEnv env) => _env = env;

    /// <summary>Creates a backend on device 0, or null when CUDA is
    /// unavailable (driver/libraries/PTX missing) so callers fall back to
    /// CPU.</summary>
    public static CudaBackend? TryCreate(int deviceId = 0)
    {
        CudaEnv? env = CudaEnv.TryCreate(deviceId);
        return env is null ? null : new CudaBackend(env);
    }

    private IntPtr W(string name) => _w.TryGetValue(name, out IntPtr b) ? b : 0;
    private int Wt(string name) => _wtype.TryGetValue(name, out int t) ? t : -1;
    private CudaKernels K => _env.Kernels!;
    private IntPtr S => _env.Stream.Handle;

    public void LoadModel(GgufFile gguf, ModelConfig cfg)
    {
        _cfg = cfg;
        _dim = cfg.HeadDim;
        _heads = cfg.NumHeads;
        _kvHeads = cfg.NumKvHeads;
        _ropeHalf = cfg.RopeDim / 2;
        _kvStride = _kvHeads * _dim;
        _kvCap = Math.Min(cfg.ContextLength, 4096);
        _vocab = cfg.VocabSize;

        foreach ((string name, GgufTensorInfo info) in gguf.Tensors)
        {
            int t = (int)info.Type;
            if (!CudaQuantMatmul.SupportsMatmulType(t))
                throw new NotSupportedException(
                    $"CUDA backend does not support weight type {info.Type} ({name}) — use --backend cpu");
            long bytes = RawBytes(info);
            _w[name] = _env.AllocUpload(gguf.DataBase + (long)info.Offset, bytes);
            _wtype[name] = t;
            _wbytes[name] = bytes;
        }

        _embd = W("token_embd.weight");
        if (_embd == 0) throw new KeyNotFoundException("token_embd.weight");
        _outNorm = W("output_norm.weight");
        if (_outNorm == 0) throw new KeyNotFoundException("output_norm.weight");
        if (W("output.weight") == 0)
        {
            _w["output.weight"] = _embd;   // tied lm_head
            _wtype["output.weight"] = _wtype["token_embd.weight"];
        }

        // F32/F16 are dense cuBLAS operands, not quant_get_rows types: the
        // embedding gather supports quantized rows and plain F32 index select.
        int embdType = Wt("token_embd.weight");
        if (embdType != 0 && !CudaQuantMatmul.SupportsQuantizedType(embdType))
            throw new NotSupportedException(
                $"token_embd.weight type {(GgmlTensorType)embdType} not supported on CUDA — use --backend cpu");

        // Per-layer head-first f32 KV: [kvHeads][kvCap][dim].
        long kvBytes = (long)_kvHeads * _kvCap * _dim * sizeof(float);
        _kvK = new IntPtr[cfg.NumLayers];
        _kvV = new IntPtr[cfg.NumLayers];
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            _kvK[l] = _env.Alloc(kvBytes);
            _kvV[l] = _env.Alloc(kvBytes);
        }

        // RoPE cos/sin tables for every position up to cap: row-major
        // [pos][ropeHalf], freqs[i] = base^(-i/half) matching Ops.cs.
        float[] freqs = new float[_ropeHalf];
        for (int i = 0; i < _ropeHalf; i++)
            freqs[i] = 1f / MathF.Pow(cfg.RopeBase, (float)i / _ropeHalf);
        float[] cos = new float[(long)_kvCap * _ropeHalf];
        float[] sin = new float[(long)_kvCap * _ropeHalf];
        for (int pos = 0; pos < _kvCap; pos++)
            for (int i = 0; i < _ropeHalf; i++)
            {
                float angle = pos * freqs[i];
                cos[pos * _ropeHalf + i] = MathF.Cos(angle);
                sin[pos * _ropeHalf + i] = MathF.Sin(angle);
            }
        fixed (float* cp = cos) _ropeCos = _env.AllocUpload(cp, cos.Length * 4);
        fixed (float* sp = sin) _ropeSin = _env.AllocUpload(sp, sin.Length * 4);
        fixed (float* fp = freqs) _ropeFreqs = _env.AllocUpload(fp, freqs.Length * 4);
        _ropeCos1 = _env.Alloc(_ropeHalf * 4);
        _ropeSin1 = _env.Alloc(_ropeHalf * 4);

        int hidden = cfg.HiddenSize, ffn = cfg.FfnSize;
        int qDim = _heads * _dim, kDim = _kvStride;
        _h = _env.Alloc(hidden * 4);
        _n1 = _env.Alloc(hidden * 4);
        _n2 = _env.Alloc(hidden * 4);
        _qkv = _env.Alloc((qDim + 2L * kDim) * 4);
        _gu = _env.Alloc(2L * ffn * 4);
        // q8_1 activation scratch for the warp-per-row decode matvecs:
        // (inDim/32) * 36B per row; ffn is the widest activation input.
        _xq = _env.Alloc(Math.Max((ffn / 32), (hidden / 32)) * 36);
        _ao = _env.Alloc(qDim * 4);
        _attnOut = _env.Alloc(hidden * 4);
        _down = _env.Alloc(hidden * 4);
        _normed = _env.Alloc(hidden * 4);
        _logitsBuf = _env.Alloc((long)_vocab * 4);
        _tok = _env.Alloc(4);
        _dyn = _env.Alloc(4 * sizeof(int));
        _logitsHost = new float[_vocab];
        _logitsPin = GCHandle.Alloc(_logitsHost, GCHandleType.Pinned);
        _tokHost = (int*)NativeMemory.Alloc(4);
        _dynHost = (int*)NativeMemory.Alloc(4 * sizeof(int));
        _kvStage = _env.Alloc((long)_kvCap * _kvStride * 2);
        _kvStageBytes = (long)_kvCap * _kvStride * 2;

        // Decode-graph fused weights: concat same-type row groups so one
        // matmul produces q|k|v (or gate|up) — fewer graph nodes per layer.
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            TryFuseWeights(l, "attn_q", "attn_k", "attn_v", "qkv");
            // Mixed-type models (q,k = Q4_K but v = Q6_K) still fuse the
            // same-type pairs so the qkv projection is 2 matmuls, not 3.
            TryFuseWeights(l, "attn_q", "attn_k", "qk");
            TryFuseWeights(l, "ffn_gate", "ffn_up", "gu");
        }
    }

    /// <summary>
    /// If blk.{l}.{a,b[,c]}.weight all exist with the same quant type,
    /// concatenates their rows into one device buffer registered as
    /// blk.{l}.{fused}.weight so a single matmul emits the concatenated
    /// outputs. Device-to-device copies keep host round-trips out of load.
    /// </summary>
    private void TryFuseWeights(int l, string a, string b, string fused) =>
        TryFuseWeights(l, new[] { a, b }, fused);

    private void TryFuseWeights(int l, string a, string b, string c, string fused) =>
        TryFuseWeights(l, new[] { a, b, c }, fused);

    private void TryFuseWeights(int l, string[] parts, string fused)
    {
        IntPtr[] srcs = new IntPtr[parts.Length];
        long[] sizes = new long[parts.Length];
        int type = -1;
        for (int i = 0; i < parts.Length; i++)
        {
            string n = $"blk.{l}.{parts[i]}.weight";
            if (!_w.TryGetValue(n, out srcs[i]) || !_wtype.TryGetValue(n, out int t) || (type != -1 && t != type))
                return;
            sizes[i] = RawBytesOf(n);
            type = t;
        }
        long total = 0;
        foreach (long s in sizes) total += s;
        IntPtr dst = _env.Alloc(total);
        long off = 0;
        for (int i = 0; i < parts.Length; i++)
        {
            CudaDriverApi.cuMemcpyDtoD(new IntPtr(dst.ToInt64() + off), srcs[i], new UIntPtr((ulong)sizes[i])).ThrowOnError();
            off += sizes[i];
        }
        _w[$"blk.{l}.{fused}.weight"] = dst;
        _wtype[$"blk.{l}.{fused}.weight"] = type;
    }

    private long RawBytesOf(string name)
    {
        // Row widths share the tensor's element count / outDim; recover the
        // byte length from the element count recorded at upload time.
        return _wbytes[name];
    }

    private static long RawBytes(GgufTensorInfo info)
    {
        long ne = (long)info.NumElements;
        int t = (int)info.Type;
        return t switch
        {
            0 => ne * 4,                     // F32
            1 or 30 => ne * 2,               // F16 / BF16
            2 => ne / 32 * 18,               // Q4_0
            3 => ne / 32 * 20,               // Q4_1
            6 => ne / 32 * 22,               // Q5_0
            7 => ne / 32 * 24,               // Q5_1
            8 => ne / 32 * 34,               // Q8_0
            9 => ne / 32 * 36,               // Q8_1
            10 => ne / 256 * 84,             // Q2_K
            11 => ne / 256 * 110,            // Q3_K
            12 => ne / 256 * 144,            // Q4_K
            13 => ne / 256 * 176,            // Q5_K
            14 => ne / 256 * 210,            // Q6_K
            16 => ne / 256 * 66,             // IQ2_XXS
            18 => ne / 256 * 98,             // IQ3_XXS
            21 => ne / 256 * 110,            // IQ3_S
            22 => ne / 256 * 82,             // IQ2_S
            20 => ne / 32 * 18,              // IQ4_NL
            23 => ne / 256 * 136,            // IQ4_XS
            39 => ne / 32 * 17,              // MXFP4
            _ => throw new NotSupportedException(
                $"CUDA backend does not support weight type {info.Type} — use --backend cpu"),
        };
    }

    public void UploadKv(int layer, int pos, int len, ushort* k, ushort* v)
    {
        if (pos + len > _kvCap)
            throw new NotSupportedException($"CUDA KV capacity {_kvCap} exceeded at {pos + len}");
        long bytes = (long)len * _kvStride * sizeof(ushort);
        if (bytes > _kvStageBytes)
            throw new InvalidOperationException($"UploadKv chunk {bytes} exceeds staging {_kvStageBytes}");
        _env.Context.MakeCurrent();
        CudaDriverApi.cuMemcpyHtoDAsync(_kvStage, (IntPtr)(k + (long)pos * _kvStride),
            new UIntPtr((ulong)bytes), S).ThrowOnError();
        _env.HymtKernels!.LaunchRowsToHeadFirstBf16(_kvStage, _kvK[layer], _kvHeads, len, _dim, pos, _kvCap, S);
        CudaDriverApi.cuMemcpyHtoDAsync(_kvStage, (IntPtr)(v + (long)pos * _kvStride),
            new UIntPtr((ulong)bytes), S).ThrowOnError();
        _env.HymtKernels!.LaunchRowsToHeadFirstBf16(_kvStage, _kvV[layer], _kvHeads, len, _dim, pos, _kvCap, S);
    }

    public float[] ForwardStep(ReadOnlySpan<int> tokens, int pos)
    {
        return tokens.Length == 1 ? DecodeStep(tokens[0], pos) : PrefillStep(tokens, pos);
    }

    private float[] DecodeStep(int tokenId, int pos)
    {
        int kvLen = pos + 1;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"CUDA KV capacity {_kvCap} exceeded at {kvLen}");
        _env.Context.MakeCurrent();

        if (_useGraph)
        {
            if (_graphExec != IntPtr.Zero)
            {
                DecodeReplay(tokenId, pos);
                return _logitsHost;
            }
            // The first decode step runs the enqueue path below to warm every
            // lazy scratch allocation (device allocs are illegal mid-capture);
            // the second captures the graph and replays it for this step.
            if (_decodeSteps > 0 && TryDecodeGraphCapture())
            {
                DecodeReplay(tokenId, pos);
                return _logitsHost;
            }
        }
        _decodeSteps++;
        EnsureEvents();
        if (_timeGpu != 0)
        {
            CudaDriverApi.cuEventRecord(_evt0, S);
            EnqueueDecodeStep(tokenId, pos);
            CudaDriverApi.cuEventRecord(_evt1, S);
            _env.Synchronize();
            PrintGpuTime();
            if (_timeGpu == 2) DumpMarks();
        }
        else
        {
            EnqueueDecodeStep(tokenId, pos);
            _env.Synchronize();
        }
        return _logitsHost;
    }

    /// <summary>Records a labelled GPU event between enqueue ops for the
    /// =2 breakdown mode; no-op unless _timeGpu is set.</summary>
    private void Mark(string label)
    {
        if (_timeGpu == 0) return;
        CudaDriverApi.cuEventCreate(out IntPtr e, 0);
        CudaDriverApi.cuEventRecord(e, S);
        _marks.Add((label, e));
    }

    private void DumpMarks()
    {
        if (_marks.Count < 2) return;
        Console.Error.WriteLine("--- gpu op breakdown (last step) ---");
        var rows = new List<(string label, float ms)>();
        for (int i = 1; i < _marks.Count; i++)
        {
            CudaDriverApi.cuEventElapsedTime(out float ms, _marks[i - 1].evt, _marks[i].evt);
            rows.Add((_marks[i].label, ms));
        }
        // Aggregate identical labels (per-layer loop bodies).
        var agg = new Dictionary<string, (float sum, int n, float max)>();
        foreach ((string label, float ms) in rows)
        {
            if (!agg.TryGetValue(label, out var a)) a = (0, 0, 0);
            agg[label] = (a.sum + ms, a.n + 1, Math.Max(a.max, ms));
        }
        foreach (var kv in agg.OrderByDescending(kv => kv.Value.sum).Take(16))
            Console.Error.WriteLine($"  {kv.Value.sum,8:F3} ms  n={kv.Value.n,4}  max={kv.Value.max:F3}  {kv.Key}");
        foreach ((_, IntPtr e) in _marks) CudaDriverApi.cuEventDestroy(e);
        _marks.Clear();
    }

    private void EnsureEvents()
    {
        if (_timeGpu == 0 || _evt0 != IntPtr.Zero) return;
        CudaDriverApi.cuEventCreate(out _evt0, 0);
        CudaDriverApi.cuEventCreate(out _evt1, 0);
    }

    private void PrintGpuTime()
    {
        CudaDriverApi.cuEventElapsedTime(out float ms, _evt0, _evt1);
        _gpuAcc += ms; _gpuCnt++;
        if (_gpuCnt % 32 == 0)
            Console.Error.WriteLine($"cuda decode gpu={_gpuAcc / _gpuCnt:F3} ms/tok (n={_gpuCnt})");
    }

    /// <summary>
    /// Enqueues the whole decode step. Position/attend-length always flow
    /// through the device dyn buffer (the same sequence is captured verbatim
    /// into the decode graph), and node count is kept minimal: fused qkv/gu
    /// matmuls, one fused attention-prep kernel per layer, residual adds fused
    /// into the following RMSNorm.
    /// </summary>
    private void EnqueueDecodeStep(int tokenId, int pos)
    {
        _marks.Clear();
        _xqSrc = IntPtr.Zero;
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, ffn = c.FfnSize;
        int qDim = _heads * _dim, kDim = _kvStride;
        float scale = 1f / MathF.Sqrt(_dim);

        _tokHost[0] = tokenId;
        _dynHost[0] = pos + 1;   // ATTEND_LEN
        _dynHost[1] = pos;       // KV_WRITE_POS
        _dynHost[2] = 0;
        _dynHost[3] = pos;       // ROPE_POS
        CudaDriverApi.cuMemcpyHtoDAsync(_tok, (IntPtr)_tokHost, 4, S).ThrowOnError();
        CudaDriverApi.cuMemcpyHtoDAsync(_dyn, (IntPtr)_dynHost, 4 * sizeof(int), S).ThrowOnError();
        // Rebuild the single-position rope table on device.
        K.LaunchFillNeoXRopeTablesDynamicF32(
            _ropeCos1, _ropeSin1, _ropeFreqs, _ropeHalf,
            IntPtr.Zero, IntPtr.Zero, IntPtr.Zero, 0, _dyn, S);
        Embed(_tok, _h, 1);
        Mark("prelude");

        for (int l = 0; l < c.NumLayers; l++)
        {
            // Residual + norm fused: h += previous block output; n = rms(h)*w.
            // Layer 0 has no previous FFN output — plain rms on the embedding.
            if (l == 0)
                Rms(_h, W($"blk.{l}.attn_norm.weight"), _n1, 1, hidden);
            else
                _env.HymtKernels!.LaunchAddRmsNormF32(
                    _h, _down, W($"blk.{l}.attn_norm.weight"), _n1, hidden, c.Eps, S);

            Mark("rms");
            IntPtr qkv = W($"blk.{l}.qkv.weight");
            if (qkv != 0)
            {
                Matmul($"blk.{l}.qkv.weight", _n1, _qkv, hidden, qDim + 2 * kDim, 1);
            }
            else if (W($"blk.{l}.qk.weight") != 0)
            {
                // q,k fused (same quant type); v has a different type.
                Matmul($"blk.{l}.qk.weight", _n1, _qkv, hidden, qDim + kDim, 1);
                Matmul($"blk.{l}.attn_v.weight", _n1, _qkv + (qDim + kDim) * 4, hidden, kDim, 1);
            }
            else
            {
                Matmul($"blk.{l}.attn_q.weight", _n1, _qkv, hidden, qDim, 1);
                Matmul($"blk.{l}.attn_k.weight", _n1, _qkv + qDim * 4, hidden, kDim, 1);
                Matmul($"blk.{l}.attn_v.weight", _n1, _qkv + (qDim + kDim) * 4, hidden, kDim, 1);
            }
            Mark("qkv");

            _env.HymtKernels!.LaunchAttnPrepF32(
                _qkv, _qkv + qDim * 4, _qkv + (qDim + kDim) * 4,
                W($"blk.{l}.attn_q_norm.weight"), W($"blk.{l}.attn_k_norm.weight"),
                _ropeCos1, _ropeSin1, _kvK[l], _kvV[l],
                _heads, _kvHeads, _dim, _ropeHalf, _kvCap, c.Eps, _dyn, S);
            Mark("prep");
            K.LaunchGqaDecodeAttentionF32(_qkv, _kvK[l], _kvV[l], _ao,
                _heads, _kvHeads, _dim, 0, _kvCap, _kvCap, 0, scale, S,
                dynParams: _dyn, smemTokens: _kvCap);
            Mark("attn");
            Matmul($"blk.{l}.attn_output.weight", _ao, _attnOut, qDim, hidden, 1);
            Mark("outm");

            _env.HymtKernels!.LaunchAddRmsNormF32(
                _h, _attnOut, W($"blk.{l}.ffn_norm.weight"), _n2, hidden, c.Eps, S);
            if (W($"blk.{l}.gu.weight") != 0)
            {
                Matmul($"blk.{l}.gu.weight", _n2, _gu, hidden, 2 * ffn, 1);
            }
            else
            {
                Matmul($"blk.{l}.ffn_gate.weight", _n2, _gu, hidden, ffn, 1);
                Matmul($"blk.{l}.ffn_up.weight", _n2, _gu + ffn * 4, hidden, ffn, 1);
            }
            Mark("gu");
            K.LaunchSiluMulF32(_gu, _gu, _gu + ffn * 4, ffn, S);
            Mark("silu");
            Matmul($"blk.{l}.ffn_down.weight", _gu, _down, ffn, hidden, 1);
            Mark("down");
        }

        _env.HymtKernels!.LaunchAddRmsNormF32(_h, _down, _outNorm, _normed, hidden, c.Eps, S);
        Matmul("output.weight", _normed, _logitsBuf, hidden, _vocab, 1);
        Mark("logits");

        CudaDriverApi.cuMemcpyDtoHAsync(_logitsPin.AddrOfPinnedObject(), _logitsBuf,
            new UIntPtr((ulong)(_vocab * 4)), S).ThrowOnError();
        Mark("dtoh");
    }

    /// <summary>
    /// Captures <see cref="EnqueueDecodeStep"/> into a CUDA graph and
    /// instantiates it. Returns false (and permanently disables the graph
    /// path) if capture or instantiation fails — the scalar step still runs.
    /// </summary>
    private bool TryDecodeGraphCapture()
    {
        _env.Context.MakeCurrent();
        if (CudaDriverApi.cuStreamBeginCapture(S, 0) != 0)
        {
            _useGraph = false;
            return false;
        }
        IntPtr graph;
        try
        {
            EnqueueDecodeStep(0, 0);
            CudaDriverApi.cuStreamEndCapture(S, out graph).ThrowOnError();
        }
        catch
        {
            _useGraph = false;
            return false;
        }
        int rc = CudaDriverApi.cuGraphInstantiateWithFlags(out _graphExec, graph, 0);
        CudaDriverApi.cuGraphDestroy(graph);
        if (rc != 0)
        {
            _useGraph = false;
            return false;
        }
        return true;
    }

    private void DecodeReplay(int tokenId, int pos)
    {
        _tokHost[0] = tokenId;
        _dynHost[0] = pos + 1;
        _dynHost[1] = pos;
        _dynHost[2] = 0;
        _dynHost[3] = pos;
        if (_timeGpu != 0)
        {
            EnsureEvents();
            CudaDriverApi.cuEventRecord(_evt0, S);
            CudaDriverApi.cuGraphLaunch(_graphExec, S).ThrowOnError();
            CudaDriverApi.cuEventRecord(_evt1, S);
            _env.Synchronize();
            PrintGpuTime();
        }
        else
        {
            CudaDriverApi.cuGraphLaunch(_graphExec, S).ThrowOnError();
            _env.Synchronize();
        }
    }

    private void EnsurePrefillBuffers(int T)
    {
        if (_pfCap >= T) return;
        int hidden = _cfg.HiddenSize, ffn = _cfg.FfnSize;
        int qDim = _heads * _dim, kDim = _kvStride;
        _pfToks = _env.Alloc(T * 4);
        _pfH = _env.Alloc((long)T * hidden * 4);
        _pfN1 = _env.Alloc((long)T * hidden * 4);
        _pfN2 = _env.Alloc((long)T * hidden * 4);
        _pfQ = _env.Alloc((long)T * qDim * 4);
        _pfQH = _env.Alloc((long)T * qDim * 4);
        _pfK = _env.Alloc((long)T * kDim * 4);
        _pfV = _env.Alloc((long)T * kDim * 4);
        _pfAo = _env.Alloc((long)T * qDim * 4);
        _pfAttnOut = _env.Alloc((long)T * hidden * 4);
        _pfGate = _env.Alloc((long)T * ffn * 4);
        _pfUp = _env.Alloc((long)T * ffn * 4);
        _pfDown = _env.Alloc((long)T * hidden * 4);
        _pfNormed = _env.Alloc((long)T * hidden * 4);
        _pfCap = T;
    }

    private float[] PrefillStep(ReadOnlySpan<int> tokens, int start)
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, ffn = c.FfnSize;
        int qDim = _heads * _dim, kDim = _kvStride;
        int T = tokens.Length, kvLen = start + T;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"CUDA KV capacity {_kvCap} exceeded at {kvLen}");
        float scale = 1f / MathF.Sqrt(_dim);
        _env.Context.MakeCurrent();
        EnsurePrefillBuffers(T);

        fixed (int* tp = tokens)
            CudaDriverApi.cuMemcpyHtoDAsync(_pfToks, (IntPtr)tp, new UIntPtr((ulong)(T * 4)), S).ThrowOnError();
        Embed(_pfToks, _pfH, T);

        for (int l = 0; l < c.NumLayers; l++)
        {
            Rms(_pfH, W($"blk.{l}.attn_norm.weight"), _pfN1, T, hidden);
            Matmul($"blk.{l}.attn_q.weight", _pfN1, _pfQ, hidden, qDim, T);
            Matmul($"blk.{l}.attn_k.weight", _pfN1, _pfK, hidden, kDim, T);
            Matmul($"blk.{l}.attn_v.weight", _pfN1, _pfV, hidden, kDim, T);
            Rope(_pfQ, _heads, T, start);
            Rope(_pfK, _kvHeads, T, start);
            Rms(_pfQ, W($"blk.{l}.attn_q_norm.weight"), _pfQ, T * _heads, _dim);
            Rms(_pfK, W($"blk.{l}.attn_k_norm.weight"), _pfK, T * _kvHeads, _dim);
            _env.HymtKernels!.LaunchRowsToHeadFirstF32(_pfK, _kvK[l], _kvHeads, T, _dim, start, _kvCap, S);
            _env.HymtKernels!.LaunchRowsToHeadFirstF32(_pfV, _kvV[l], _kvHeads, T, _dim, start, _kvCap, S);
            // Head-first Q temp for the attention kernel.
            _env.HymtKernels!.LaunchRowsToHeadFirstF32(_pfQ, _pfQH, _heads, T, _dim, 0, T, S);
            K.LaunchGqaPrefillAttentionF32(_pfQH, _kvK[l], _kvV[l], _pfAo,
                _heads, _kvHeads, T, kvLen, _dim, start, 0, scale, _kvCap, S);
            Matmul($"blk.{l}.attn_output.weight", _pfAo, _pfAttnOut, qDim, hidden, T);
            Add(_pfH, _pfAttnOut, T * hidden);

            Rms(_pfH, W($"blk.{l}.ffn_norm.weight"), _pfN2, T, hidden);
            Matmul($"blk.{l}.ffn_gate.weight", _pfN2, _pfGate, hidden, ffn, T);
            Matmul($"blk.{l}.ffn_up.weight", _pfN2, _pfUp, hidden, ffn, T);
            K.LaunchSiluMulF32(_pfGate, _pfGate, _pfUp, (long)T * ffn, S);
            Matmul($"blk.{l}.ffn_down.weight", _pfGate, _pfDown, ffn, hidden, T);
            Add(_pfH, _pfDown, T * hidden);
        }

        Rms(_pfH, _outNorm, _pfNormed, T, hidden);
        // Logits for the last row only.
        IntPtr lastRow = new IntPtr(_pfNormed.ToInt64() + (long)(T - 1) * hidden * 4);
        Matmul("output.weight", lastRow, _logitsBuf, hidden, _vocab, 1);

        fixed (float* lp = _logitsHost)
            CudaDriverApi.cuMemcpyDtoHAsync((IntPtr)lp, _logitsBuf,
                new UIntPtr((ulong)(_vocab * 4)), S).ThrowOnError();
        _env.Synchronize();
        return _logitsHost;
    }

    private void Embed(IntPtr tokBuf, IntPtr dst, int T)
    {
        int t = Wt("token_embd.weight");
        if (t == 0)
            K.LaunchIndexSelectF32(_embd, tokBuf, dst, T, _cfg.HiddenSize, _vocab, 1, 0, S);
        else
            K.LaunchQuantGetRowsF32(_embd, tokBuf, dst, t, _cfg.HiddenSize, T, 1, S);
    }

    // HYMT2SHARP_CUDA_MATVEC=0 falls back to the vendored vec kernels.
    private static readonly bool UseHymtMatvec =
        !string.Equals(Environment.GetEnvironmentVariable("HYMT2SHARP_CUDA_MATVEC"), "0", StringComparison.Ordinal);

    private void Matmul(string name, IntPtr x, IntPtr y, int inDim, int outDim, int rows)
    {
        IntPtr w = W(name);
        if (w == 0) throw new KeyNotFoundException(name);
        int t = Wt(name);
        if (UseHymtMatvec && rows == 1 && (t == 12 || t == 14)
            && (inDim & 255) == 0 && _env.HymtKernels != null)
        {
            // Consecutive matmuls share the activation row (q|k|v all read the
            // same _n1) — quantize it once. The flag is baked at capture time,
            // so reset it per step in EnqueueDecodeStep.
            if (x != _xqSrc || inDim != _xqSrcDim)
            {
                K.LaunchQuantizeQ81Rows(x, _xq, inDim, 1, S, warpCooperative: true);
                _xqSrc = x;
                _xqSrcDim = inDim;
            }
            _env.HymtKernels.LaunchMatvecQ81F32(t, w, _xq, y, inDim, outDim, S);
            return;
        }
        CudaQuantMatmul.RunResidentMatmul(_env, K, w, t, x, y, inDim, outDim, rows);
    }

    private void Rms(IntPtr x, IntPtr w, IntPtr y, int rows, int cols)
        => K.LaunchRMSNormF32(x, w, IntPtr.Zero, y, rows, cols, _cfg.Eps, S);

    private void Rope(IntPtr x, int heads, int T, int startPos)
    {
        // Table rows are indexed by sequence position; offset to startPos.
        IntPtr cosRow = new IntPtr(_ropeCos.ToInt64() + (long)startPos * _ropeHalf * 4);
        IntPtr sinRow = new IntPtr(_ropeSin.ToInt64() + (long)startPos * _ropeHalf * 4);
        K.LaunchNeoXRoPEFlatF32(x, cosRow, sinRow, heads, T, _dim, _ropeHalf, S);
    }

    private void Add(IntPtr a, IntPtr b, long n)
        => K.LaunchBinaryF32(a, b, a, checked((int)n), 0, S);

    public void Dispose() => _env.Dispose();
}
