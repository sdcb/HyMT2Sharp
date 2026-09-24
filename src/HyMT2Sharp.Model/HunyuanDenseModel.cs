using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Model;

public sealed unsafe partial class HunyuanDenseModel : IDisposable
{
    private readonly GgufFile _gguf;
    private readonly CpuThreadPool _pool;
    private readonly ScratchArena _gemmScratch = new();
    private readonly Dictionary<string, Weight> _weights = new(StringComparer.Ordinal);
    private readonly float*[] _cacheK;
    private readonly float*[] _cacheV;
    // Keep ownership of the currently active cache allocations separately so
    // that an expansion can release the previous generation immediately.
    private readonly NativeBuffer[] _cacheKBuffers;
    private readonly NativeBuffer[] _cacheVBuffers;
    private readonly List<NativeBuffer> _buffers = [];
    private NativeBuffer? _scratch;
    private int _scratchUsed;
    private readonly List<int> _cacheTokens = [];
    private int _cacheLen;
    private int _cacheCap;
    private float[]? _logits;

    public ModelConfig Config { get; private set; }
    public BpeTokenizer Tokenizer { get; }
    public int ThreadCount => _pool.ThreadCount;
    public string ThreadAutoHint => CpuTopology.AutoHint;
    public static bool ProfileEnabled;
    public static long TicksQ4;
    public static long TicksQ2;
    public static long TicksSTQ;
    public static long TicksQ6;
    public static long TicksQ8;
    public static long TicksAttnScore;
    public static long TicksSoftmax;
    public static long TicksAttnCombine;
    public static long TicksRms;
    public static long TicksRope;
    public static long TicksSilu;
    public static long TicksQuant;
    public static long TicksEmbed;

    public static void ResetProfile()
    {
        TicksQ4 = 0;
        TicksQ2 = 0;
        TicksSTQ = 0;
        TicksQ6 = 0;
        TicksQ8 = 0;
        TicksAttnScore = 0;
        TicksSoftmax = 0;
        TicksAttnCombine = 0;
        TicksRms = 0;
        TicksRope = 0;
        TicksSilu = 0;
        TicksQuant = 0;
        TicksEmbed = 0;
    }

    public HunyuanDenseModel(string ggufPath, int threads = 0)
    {
        _gguf = new GgufFile(ggufPath);
        Config = ModelConfig.FromGguf(_gguf);
        if (Config.VocabSize == 0)
        {
            // Filled after weights load from token_embd rows.
        }
        if (!string.Equals(Config.Architecture, "hunyuan-dense", StringComparison.Ordinal))
            throw new NotSupportedException($"Expected hunyuan-dense, got {Config.Architecture}");
        Tokenizer = new BpeTokenizer(_gguf);
        _pool = new CpuThreadPool(threads);
        LoadWeights();
        if (Config.VocabSize == 0 && _weights.TryGetValue("token_embd.weight", out Weight emb) && emb.NOut > 0)
            Config.VocabSize = emb.NOut;
        _cacheCap = Math.Min(Config.ContextLength, 4096);
        _cacheK = new float*[Config.NumLayers];
        _cacheV = new float*[Config.NumLayers];
        _cacheKBuffers = new NativeBuffer[Config.NumLayers];
        _cacheVBuffers = new NativeBuffer[Config.NumLayers];
        int kvStride = Config.NumKvHeads * Config.HeadDim;
        for (int l = 0; l < Config.NumLayers; l++)
        {
            NativeBuffer k = Rent((nuint)((long)_cacheCap * kvStride * sizeof(float)));
            NativeBuffer v = Rent((nuint)((long)_cacheCap * kvStride * sizeof(float)));
            _cacheKBuffers[l] = k;
            _cacheVBuffers[l] = v;
            _cacheK[l] = (float*)k.Pointer;
            _cacheV[l] = (float*)v.Pointer;
        }
    }

    public int CacheLength => _cacheLen;

    public void ResetCache() => TruncateCache(0);

    public void TruncateCache(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (length > _cacheLen)
            throw new ArgumentOutOfRangeException(nameof(length), length, $"Cannot extend cache from {_cacheLen}.");
        _cacheLen = length;
        if (_cacheTokens.Count > length)
            _cacheTokens.RemoveRange(length, _cacheTokens.Count - length);
    }

    public PromptAlignment AlignPrompt(ReadOnlySpan<int> promptIds)
    {
        PromptReuse plan = KvCacheAlign.Plan(CollectionsMarshal.AsSpan(_cacheTokens), promptIds);
        TruncateCache(plan.TruncateTo);
        return new PromptAlignment(plan.TruncateTo, promptIds[plan.SuffixStart..].ToArray());
    }

    public float[] Forward(int[] tokens)
    {
        _scratchUsed = 0;
        int seq = tokens.Length;
        int start = _cacheLen;
        EnsureCache(start + seq);
        int hidden = Config.HiddenSize;
        float* h = (float*)Bump((nuint)((long)seq * hidden * sizeof(float)));
        Embed(tokens, h);
        int layerMark = _scratchUsed;
        if (seq == 1)
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _scratchUsed = layerMark;
                DecodeBlock(h, layer, start);
            }
        }
        else
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                _scratchUsed = layerMark;
                PrefillBlock(h, layer, seq, start);
            }
        }

        float* normed = (float*)Bump((nuint)((long)seq * hidden * sizeof(float)));
        Rms("output_norm.weight", h, normed, seq, hidden);
        float* last = normed + (seq - 1) * hidden;

        int vocab = Config.VocabSize > 0 ? Config.VocabSize : Tokenizer.VocabSize;
        float[] logits = _logits ??= new float[vocab];
        fixed (float* lp = logits)
            Linear(last, "output.weight", lp, hidden, vocab, 1, fallback: "token_embd.weight");

        _cacheLen += seq;
        _cacheTokens.AddRange(tokens);
        Debug.Assert(_cacheTokens.Count == _cacheLen);
        return logits;
    }

    private void Linear(float* input, string name, float* output, int nIn, int nOut, int tokens, string? fallback = null)
    {
        if (!_weights.TryGetValue(name, out Weight w) && fallback != null)
            _weights.TryGetValue(fallback, out w);
        if (w.Type == 0 && w.Q4 == null && w.Q5 == null && w.Q6 == null)
            throw new KeyNotFoundException(name);
        switch (w.Type)
        {
            case GgmlTensorType.Q4_K:
            {
                long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
                MulMatQ4K.Gemm(w.Packed, w.Q4, input, output, nIn, nOut, tokens, _pool, _gemmScratch, w.Meta);
                if (ProfileEnabled)
                    TicksQ4 += Stopwatch.GetTimestamp() - t0;
                break;
            }
            case GgmlTensorType.Q2_0C:
            {
                long t2 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
                MulMatQ2.Gemm(w.Packed2, w.Q2, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                if (ProfileEnabled)
                    TicksQ2 += Stopwatch.GetTimestamp() - t2;
                break;
            }
            case GgmlTensorType.STQ1_0:
            {
                long t = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
                MulMatSTQ.Gemm(w.PackedSTQ, w.STQ, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                if (ProfileEnabled)
                    TicksSTQ += Stopwatch.GetTimestamp() - t;
                break;
            }
            case GgmlTensorType.Q5_K:
                Q5K.Gemm(w.Q5, input, output, nIn, nOut, tokens, _pool);
                break;
            case GgmlTensorType.Q6_K:
            {
                long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
                MulMatQ6K.Gemm(w.Packed6, w.Q6, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                if (ProfileEnabled)
                    TicksQ6 += Stopwatch.GetTimestamp() - t0;
                break;
            }
            case GgmlTensorType.Q8_0:
            {
                long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
                MulMatQ8_0.Gemm(w.Packed8, w.Q8, input, output, nIn, nOut, tokens, _pool, _gemmScratch);
                if (ProfileEnabled)
                    TicksQ8 += Stopwatch.GetTimestamp() - t0;
                break;
            }
            default:
                throw new NotSupportedException($"{name} type {w.Type} cannot be used as a linear weight.");
        }
    }

    private void Embed(int[] tokens, float* dest)
    {
        Weight w = _weights["token_embd.weight"];
        int hidden = Config.HiddenSize;
        if (w.Type == GgmlTensorType.Q8_0)
        {
            if (hidden % Qk.Q8_0Block != 0)
                throw new NotSupportedException("token embedding inner dim must be a multiple of 32.");
        }
        else if (hidden % Qk.SuperBlock != 0)
            throw new NotSupportedException("token embedding inner dim must be a multiple of 256.");
        int nb = hidden / Qk.SuperBlock;
        int n = tokens.Length;
        long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        void Row(int t)
        {
            float* row = dest + t * hidden;
            switch (w.Type)
            {
                case GgmlTensorType.Q4_K:
                    Q4K.DequantizeRow(w.Q4 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q5_K:
                    Q5K.DequantizeRow(w.Q5 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q6_K:
                    Q6K.DequantizeRow(w.Q6 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.Q8_0:
                    Q8_0.DequantizeRow(w.Q8 + tokens[t] * (hidden / Qk.Q8_0Block), row, hidden);
                    break;
                case GgmlTensorType.Q2_0C:
                    Q2_0C.DequantizeRow(w.Q2 + tokens[t] * nb, row, hidden);
                    break;
                case GgmlTensorType.STQ1_0:
                    STQ1_0.DequantizeRow(w.STQ + tokens[t] * (hidden / STQ1_0.BlockLength), row, hidden);
                    break;
                case GgmlTensorType.F32:
                    Buffer.MemoryCopy(w.F32 + tokens[t] * hidden, row, hidden * sizeof(float), hidden * sizeof(float));
                    break;
                default:
                    throw new NotSupportedException($"token_embd type {w.Type}");
            }
        }

        // llama.cpp get_rows splits token rows across threads. Decode (1 token) stays serial.
        if (n <= 8)
        {
            for (int t = 0; t < n; t++)
                Row(t);
        }
        else
        {
            _pool.For(n, (int worker, int workers) =>
            {
                int begin = n * worker / workers;
                int end = n * (worker + 1) / workers;
                for (int t = begin; t < end; t++)
                    Row(t);
            });
        }

        if (ProfileEnabled)
            TicksEmbed += Stopwatch.GetTimestamp() - t0;
    }

    private void Rms(string name, float* x, float* y, int rows, int dim)
    {
        Weight w = _weights[name];
        long t0 = ProfileEnabled ? Stopwatch.GetTimestamp() : 0;
        Ops.RmsNorm(x, w.F32, y, rows, dim, Config.Eps, _pool);
        if (ProfileEnabled)
            TicksRms += Stopwatch.GetTimestamp() - t0;
    }

    private void LoadWeights()
    {
        foreach ((string name, GgufTensorInfo info) in _gguf.Tensors)
        {
            byte* data = _gguf.DataBase + (long)info.Offset;
            if (info.Type == GgmlTensorType.F32)
            {
                int n = checked((int)info.NumElements);
                NativeBuffer buf = Rent((nuint)((long)n * sizeof(float)));
                Buffer.MemoryCopy(data, buf.Pointer, buf.Bytes, buf.Bytes);
                _weights[name] = new Weight { Type = GgmlTensorType.F32, F32 = (float*)buf.Pointer, Count = n };
                continue;
            }

            int nIn = (int)info.Shape[0];
            int nOut = info.Shape.Length > 1 ? (int)info.Shape[1] : 1;
            int align = info.Type == GgmlTensorType.Q8_0 ? Qk.Q8_0Block : Qk.SuperBlock;
            if (nIn % align != 0)
                throw new NotSupportedException($"{name}: nIn={nIn} is not a multiple of {align}");
            int nb = info.Type switch
            {
                GgmlTensorType.Q2_0C => nIn / Q2_0C.BlockLength,
                GgmlTensorType.STQ1_0 => nIn / STQ1_0.BlockLength,
                GgmlTensorType.Q8_0 => nIn / Qk.Q8_0Block,
                _ => nIn / Qk.SuperBlock,
            };
            int blockBytes = info.Type switch
            {
                GgmlTensorType.Q4_K => Qk.Q4KSize,
                GgmlTensorType.Q5_K => Qk.Q5KSize,
                GgmlTensorType.Q6_K => Qk.Q6KSize,
                GgmlTensorType.Q8_0 => Qk.Q8_0Size,
                GgmlTensorType.Q2_0C => Qk.Q2_0CSize,
                GgmlTensorType.STQ1_0 => Qk.STQ1_0Size,
                _ => throw new NotSupportedException($"Unsupported quantized tensor type {info.Type} for {name}"),
            };
            NativeBuffer quant = Rent((nuint)((long)nOut * nb * blockBytes));
            Buffer.MemoryCopy(data, quant.Pointer, quant.Bytes, quant.Bytes);
            Weight w = new()
            {
                Type = info.Type,
                NIn = nIn,
                NOut = nOut,
            };
            if (info.Type == GgmlTensorType.Q4_K)
            {
                w.Q4 = (BlockQ4K*)quant.Pointer;
                // Panels only feed the AVX2 kernels; portable hosts keep raw rows.
                if (nOut >= 8 && Simd.UsePanels)
                {
                    NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8Size));
                    RepackQ4K.Rows((BlockQ4K*)quant.Pointer, (BlockQ4Kx8*)packed.Pointer, nIn, nOut & ~7);
                    NativeBuffer meta = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8MetaSize));
                    RepackQ4K.BuildMeta((BlockQ4K*)quant.Pointer, (BlockQ4Kx8Meta*)meta.Pointer, nIn, nOut & ~7);
                    w.Meta = (BlockQ4Kx8Meta*)meta.Pointer;
                    w.Packed = (BlockQ4Kx8*)packed.Pointer;
                }
            }
            else if (info.Type == GgmlTensorType.Q5_K)
            {
                w.Q5 = (BlockQ5K*)quant.Pointer;
            }
            else
            {
                if (info.Type == GgmlTensorType.Q2_0C)
                {
                    w.Q2 = (BlockQ2_0C*)quant.Pointer;
                    if (nOut >= 8 && Simd.UsePanels)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q2x8Size));
                        RepackQ2.Rows(w.Q2, (BlockQ2x8*)packed.Pointer, nIn, nOut & ~7);
                        w.Packed2 = (BlockQ2x8*)packed.Pointer;
                    }
                }
                else if (info.Type == GgmlTensorType.STQ1_0)
                {
                    w.STQ = (BlockSTQ1_0*)quant.Pointer;
                    // Token lookup is a gather/dequant operation and never
                    // enters a matrix kernel; keep it in the compact source
                    // layout instead of paying for an unused panel copy.
                    if (nOut >= 8 && Simd.UsePanels && !string.Equals(name, "token_embd.weight", StringComparison.Ordinal))
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.STQ1_0x8Size));
                        RepackSTQ.Rows(w.STQ, (BlockSTQ1_0x8*)packed.Pointer, nIn, nOut & ~7);
                        w.PackedSTQ = (BlockSTQ1_0x8*)packed.Pointer;
                    }
                }
                else if (info.Type == GgmlTensorType.Q6_K)
                {
                    w.Q6 = (BlockQ6K*)quant.Pointer;
                    // Only per-layer Q6 weights take the prefill panel; the tied lm_head stays a GEMV.
                    if (name.StartsWith("blk.", StringComparison.Ordinal) && nOut >= 8 && Simd.UsePanels)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q6Kx8Size));
                        if (Simd.UseDp)
                            RepackQ6K.RowsNeon(w.Q6, (BlockQ6Kx8*)packed.Pointer, nIn, nOut & ~7);
                        else
                            RepackQ6K.Rows(w.Q6, (BlockQ6Kx8*)packed.Pointer, nIn, nOut & ~7);
                        w.Packed6 = (BlockQ6Kx8*)packed.Pointer;
                    }
                }
                else if (info.Type == GgmlTensorType.Q8_0)
                {
                    w.Q8 = (BlockQ8_0*)quant.Pointer;
                    // token_embd is also packed: lm_head runs one panel GEMV per
                    // token, which reads weights as a single sequential stream.
                    if (nOut >= 8 && Simd.UsePanels)
                    {
                        NativeBuffer packed = Rent((nuint)((long)(nOut / 8) * nb * Qk.Q8_0x8Size));
                        RepackQ8_0.Rows(w.Q8, (BlockQ8_0x8*)packed.Pointer, nIn, nOut & ~7);
                        w.Packed8 = (BlockQ8_0x8*)packed.Pointer;
                    }
                }
                else
                    throw new NotSupportedException($"Unsupported quantized tensor type {info.Type} for {name}");
            }

            _weights[name] = w;
        }

    }

    private void EnsureCache(int needed)
    {
        if (needed <= _cacheCap)
            return;
        int next = Math.Max(needed, _cacheCap * 2);
        next = Math.Min(next, Config.ContextLength);
        int kvStride = Config.NumKvHeads * Config.HeadDim;
        NativeBuffer?[] nextK = new NativeBuffer?[Config.NumLayers];
        NativeBuffer?[] nextV = new NativeBuffer?[Config.NumLayers];
        try
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                NativeBuffer k = Rent((nuint)((long)next * kvStride * sizeof(float)));
                NativeBuffer v = Rent((nuint)((long)next * kvStride * sizeof(float)));
                nextK[l] = k;
                nextV[l] = v;
                long copyBytes = (long)_cacheLen * kvStride * sizeof(float);
                Buffer.MemoryCopy(_cacheK[l], k.Pointer, copyBytes, copyBytes);
                Buffer.MemoryCopy(_cacheV[l], v.Pointer, copyBytes, copyBytes);
            }
        }
        catch
        {
            // Rent() tracks these buffers, so remove and dispose any partial
            // allocation before leaving the old cache active.
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (nextK[l] is NativeBuffer k)
                {
                    _buffers.Remove(k);
                    k.Dispose();
                }
                if (nextV[l] is NativeBuffer v)
                {
                    _buffers.Remove(v);
                    v.Dispose();
                }
            }
            throw;
        }

        // All copies succeeded. Swap the complete cache generation before
        // releasing the old buffers, so an exception cannot leave K/V layers
        // pointing at different capacities.
        for (int l = 0; l < Config.NumLayers; l++)
        {
            NativeBuffer oldK = _cacheKBuffers[l];
            NativeBuffer oldV = _cacheVBuffers[l];
            NativeBuffer k = nextK[l]!;
            NativeBuffer v = nextV[l]!;
            _cacheKBuffers[l] = k;
            _cacheVBuffers[l] = v;
            _cacheK[l] = (float*)k.Pointer;
            _cacheV[l] = (float*)v.Pointer;
            _buffers.Remove(oldK);
            _buffers.Remove(oldV);
            oldK.Dispose();
            oldV.Dispose();
        }

        _cacheCap = next;
    }

    private NativeBuffer Rent(nuint bytes)
    {
        NativeBuffer buf = new(bytes);
        _buffers.Add(buf);
        return buf;
    }

    private void* Bump(nuint bytes)
    {
        const int scratchBytes = 128 * 1024 * 1024;
        if (_scratch == null)
            _scratch = Rent(scratchBytes);
        int aligned = (int)((bytes + 63) & ~(nuint)63);
        if (_scratchUsed + aligned > (int)_scratch.Bytes)
        {
            _scratch = Rent((nuint)Math.Max(scratchBytes, aligned));
            _scratchUsed = 0;
        }

        byte* ptr = (byte*)_scratch.Pointer + _scratchUsed;
        _scratchUsed += aligned;
        return ptr;
    }

    public void Dispose()
    {
        foreach (NativeBuffer buf in _buffers)
            buf.Dispose();
        _gemmScratch.Dispose();
        _pool.Dispose();
        _gguf.Dispose();
    }

    private struct Weight
    {
        public GgmlTensorType Type;
        public BlockQ4K* Q4;
        public BlockQ4Kx8* Packed;
        public BlockQ4Kx8Meta* Meta;
        public BlockQ5K* Q5;
        public BlockQ6K* Q6;
        public BlockQ2_0C* Q2;
        public BlockQ2x8* Packed2;
        public BlockSTQ1_0* STQ;
        public BlockSTQ1_0x8* PackedSTQ;
        public BlockQ6Kx8* Packed6;
        public BlockQ8_0* Q8;
        public BlockQ8_0x8* Packed8;
        public float* F32;
        public int NIn;
        public int NOut;
        public int Count;
    }
}
