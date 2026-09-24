using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Kernels;
using Sdcb.HyMT2Sharp.Model;

string modelPath = Args.Get(args, "--model")
    ?? @"D:\_\model\Hy-MT2-1.8B-Q4_K_M.gguf";
int threads = Args.GetInt(args, "--threads", CpuThreadPool.PreferPCoreCount());
int benchPrefill = Args.GetInt(args, "--bench-prefill", 0);
int benchDecode = Args.GetInt(args, "--bench-decode", 0);
int colTileKb = Args.GetInt(args, "--col-tile-kb", 0);
if (colTileKb > 0)
    GemmQ4K.ColTileBytes = colTileKb * 1024;
int q2ColTileKb = Args.GetInt(args, "--q2-col-tile-kb", 0);
if (q2ColTileKb > 0)
    MulMatQ2.ColTileBytes = q2ColTileKb * 1024;
bool bench = args.Contains("--bench") || benchPrefill > 0 || benchDecode > 0;
if (args.Contains("--dump-q4-gemm"))
{
    DumpQ4Gemm();
    return;
}

if (args.Contains("--micro-q4"))
{
    MicroQ4(Args.GetInt(args, "--micro-in", 2048), Args.GetInt(args, "--micro-out", 64), Args.GetInt(args, "--micro-tokens", 32), Args.GetInt(args, "--micro-reps", 500));
    return;
}

if (args.Contains("--micro-vec-q4"))
{
    MicroVecQ4(Args.GetInt(args, "--micro-in", 2048), Args.GetInt(args, "--micro-out", 6144), Args.GetInt(args, "--micro-tokens", 512), Args.GetInt(args, "--micro-reps", 5), threads);
    return;
}

if (args.Contains("--micro-q8"))
{
    if (args.Contains("--micro-model"))
        MicroQ8Model(Args.GetInt(args, "--micro-reps", 8), threads, args.Contains("--micro-arena"));
    else
        MicroQ8(Args.GetInt(args, "--micro-in", 2048), Args.GetInt(args, "--micro-out", 6144), Args.GetInt(args, "--micro-reps", 200), threads);
    return;
}

if (args.Contains("--dump-silu"))
{
    DumpSilu();
    return;
}

if (args.Contains("--dump"))
{
    using GgufFile dump = new(modelPath);
    Console.WriteLine($"arch={dump.GetString("general.architecture")} pre={dump.GetString("tokenizer.ggml.pre")} vocab={dump.GetStringArray("tokenizer.ggml.tokens").Length}");
    Dictionary<string, int> typeCounts = [];
    foreach ((string name, GgufTensorInfo info) in dump.Tensors)
    {
        string t = info.Type.ToString();
        typeCounts[t] = typeCounts.TryGetValue(t, out int c) ? c + 1 : 1;
        if (info.Type is not GgmlTensorType.Q4_K and not GgmlTensorType.F32)
            Console.WriteLine($"{info.Type,-8} {name} [{string.Join(",", info.Shape)}]");
    }

    Console.WriteLine("type counts:");
    foreach ((string t, int c) in typeCounts)
        Console.WriteLine($"  {t}={c}");
    return;
}

if (!bench && !args.Contains("--verify-prefill"))
{
    Console.WriteLine("HyMT2Sharp.Benchmark");
    Console.WriteLine("  --model PATH");
    Console.WriteLine("  --threads N");
    Console.WriteLine("  --bench");
    Console.WriteLine("  --bench-prefill N");
    Console.WriteLine("  --bench-decode N");
    Console.WriteLine("  --col-tile-kb N");
    Console.WriteLine("  --q2-col-tile-kb N");
    Console.WriteLine("  --profile");
    Console.WriteLine("  --verify-prefill N");
    Console.WriteLine("  --dump");
    Console.WriteLine("  --dump-q4-gemm");
    Console.WriteLine("  --dump-silu");
    Console.WriteLine("  --micro-q4 [--micro-in N --micro-out N --micro-tokens N --micro-reps N]");
    return;
}

Console.WriteLine($"HyMT2Sharp  model={modelPath}");
Console.WriteLine($"threads={threads}  avx2={System.Runtime.Intrinsics.X86.Avx2.IsSupported}  vnni={System.Runtime.Intrinsics.X86.AvxVnni.IsSupported}");

using HunyuanDenseModel model = new(modelPath, threads);
Console.WriteLine($"arch={model.Config.Architecture} layers={model.Config.NumLayers} hidden={model.Config.HiddenSize} heads={model.Config.NumHeads}/{model.Config.NumKvHeads} vocab={model.Config.VocabSize}");

if (args.Contains("--verify-prefill"))
{
    int count = Math.Max(1, Args.GetInt(args, "--verify-prefill", 16));
    int[] ids = new int[count];
    for (int i = 0; i < ids.Length; i++) ids[i] = 1 + i % Math.Max(1, model.Config.VocabSize - 1);
    model.ResetCache();
    float[] batched = model.Forward(ids);
    model.ResetCache();
    float[] serial = [];
    for (int i = 0; i < ids.Length; i++)
        serial = model.Forward([ids[i]]);
    float maxAbs = 0;
    double sumSq = 0;
    for (int i = 0; i < batched.Length; i++)
    {
        float d = batched[i] - serial[i];
        maxAbs = MathF.Max(maxAbs, MathF.Abs(d));
        sumSq += d * d;
    }
    Console.WriteLine($"verify-prefill tokens={count} max-abs={maxAbs:E3} rms={Math.Sqrt(sumSq / batched.Length):E3} batched-top={ArgMax(batched)} serial-top={ArgMax(serial)}");
    return;
}

if (bench && (benchPrefill > 0 || benchDecode > 0 || args.Contains("--bench")))
{
    int pp = benchPrefill > 0 ? benchPrefill : 512;
    int tg = benchDecode > 0 ? benchDecode : 128;
    int[] ids = new int[pp];
    int vocab = Math.Max(2, model.Config.VocabSize);
    for (int i = 0; i < pp; i++)
        ids[i] = 1 + (i % (vocab - 1));

    bool profile = args.Contains("--profile");
    // llama-bench warms the prompt and one generated token before the timed reps.
    model.ResetCache();
    model.Forward(ids);
    model.ResetCache();
    model.Forward([ids[0]]);
    model.ResetCache();

    HunyuanDenseModel.ProfileEnabled = profile;
    const int reps = 3;
    double[] prefillMs = new double[reps];
    float[] logits = [];
    Stopwatch sw = new();
    for (int r = 0; r < reps; r++)
    {
        model.ResetCache();
        HunyuanDenseModel.ResetProfile();
        sw.Restart();
        logits = model.Forward(ids);
        sw.Stop();
        prefillMs[r] = sw.Elapsed.TotalMilliseconds;
        if (profile)
            PrintProfile($"prefill[{r}]");
    }

    int next = ArgMax(logits);
    HunyuanDenseModel.ProfileEnabled = false;
    model.ResetCache();
    logits = model.Forward(ids);
    next = ArgMax(logits);
    HunyuanDenseModel.ProfileEnabled = profile;
    HunyuanDenseModel.ResetProfile();
    sw.Restart();
    for (int i = 0; i < tg; i++)
    {
        logits = model.Forward([next]);
        next = ArgMax(logits);
    }

    double decodeMs = sw.Elapsed.TotalMilliseconds;
    double ppMean = 0;
    for (int r = 0; r < reps; r++)
        ppMean += prefillMs[r];
    ppMean /= reps;
    Console.WriteLine($"prefill tokens={pp}  {ppMean:F1} ms  {pp / (ppMean / 1000.0):F2} tok/s  reps={pp / (prefillMs[0] / 1000.0):F1}/{pp / (prefillMs[1] / 1000.0):F1}/{pp / (prefillMs[2] / 1000.0):F1}");
    Console.WriteLine($"decode  tokens={tg}  {decodeMs:F1} ms  {tg / (decodeMs / 1000.0):F2} tok/s");
    if (profile)
        PrintProfile("decode");
    Console.WriteLine($"Compare llama-bench -m <same.gguf> -p {pp} -n {tg} -t {threads} -ngl 0");
    return;
}

static void PrintProfile(string phase)
{
    double freq = Stopwatch.Frequency;
    double q4 = HunyuanDenseModel.TicksQ4 / freq * 1000.0;
    double q2 = HunyuanDenseModel.TicksQ2 / freq * 1000.0;
    double stq = HunyuanDenseModel.TicksSTQ / freq * 1000.0;
    double q6 = HunyuanDenseModel.TicksQ6 / freq * 1000.0;
    double q8 = HunyuanDenseModel.TicksQ8 / freq * 1000.0;
    double score = HunyuanDenseModel.TicksAttnScore / freq * 1000.0;
    double soft = HunyuanDenseModel.TicksSoftmax / freq * 1000.0;
    double comb = HunyuanDenseModel.TicksAttnCombine / freq * 1000.0;
    double rms = HunyuanDenseModel.TicksRms / freq * 1000.0;
    double rope = HunyuanDenseModel.TicksRope / freq * 1000.0;
    double silu = HunyuanDenseModel.TicksSilu / freq * 1000.0;
    double quant = HunyuanDenseModel.TicksQuant / freq * 1000.0;
    double embed = HunyuanDenseModel.TicksEmbed / freq * 1000.0;
    double accounted = q4 + q2 + stq + q6 + q8 + score + soft + comb + rms + rope + silu + quant + embed;
    Console.WriteLine($"profile {phase}: q4={q4:F0}ms q2={q2:F0}ms stq={stq:F0}ms q6={q6:F0}ms q8={q8:F0}ms attnQK={score:F0}ms softmax={soft:F0}ms attnAV={comb:F0}ms rms={rms:F0}ms rope={rope:F0}ms silu={silu:F0}ms embed={embed:F0}ms accounted={accounted:F0}ms");
}

static unsafe void MicroQ4(int nIn, int nOut, int tokens, int reps)
{
    int nb = nIn / Qk.SuperBlock;
    using NativeBuffer q4 = new((nuint)((long)nOut * nb * Qk.Q4KSize));
    using NativeBuffer q4x8 = new((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8Size));
    using NativeBuffer q8x4 = new((nuint)((long)(tokens / 4) * nb * Qk.Q8Kx4Size));
    using NativeBuffer dst = new((nuint)((long)tokens * nOut * sizeof(float)));
    using NativeBuffer src = new((nuint)((long)tokens * nIn * sizeof(float)));
    using NativeBuffer scales = new((nuint)((long)(nOut / 8) * nb * Qk.Q4Kx8MetaSize));
    float* input = (float*)src.Pointer;
    for (int i = 0; i < tokens * nIn; i++)
        input[i] = MathF.Sin(i * 0.37f);
    BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
    for (int r = 0; r < nOut; r++)
        Q4K.PackSimple(input + (r % tokens) * nIn, rows + r * nb, nIn);
    RepackQ4K.Rows(rows, (BlockQ4Kx8*)q4x8.Pointer, nIn, nOut);
    RepackQ4K.BuildMeta(rows, (BlockQ4Kx8Meta*)scales.Pointer, nIn, nOut);
    for (int g = 0; g < tokens / 4; g++)
        QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, (BlockQ8Kx4*)q8x4.Pointer + g * nb, nIn);
    for (int i = 0; i < 20; i++)
        GemmQ4K.GemmAvx2(nIn, (float*)dst.Pointer, nOut, (BlockQ4Kx8*)q4x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut, (BlockQ4Kx8Meta*)scales.Pointer);
    Stopwatch sw = Stopwatch.StartNew();
    for (int i = 0; i < reps; i++)
        GemmQ4K.GemmAvx2(nIn, (float*)dst.Pointer, nOut, (BlockQ4Kx8*)q4x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut, (BlockQ4Kx8Meta*)scales.Pointer);
    sw.Stop();
    double macs = (double)nIn * nOut * tokens * reps;
    double maddubs = macs / 32;
    Console.WriteLine($"micro-q4 in={nIn} out={nOut} tokens={tokens} weights={(nOut / 8) * nb * Qk.Q4Kx8Size / 1024}KB act={(tokens / 4) * nb * Qk.Q8Kx4Size / 1024}KB  {sw.Elapsed.TotalMilliseconds:F1} ms  {macs / sw.Elapsed.TotalSeconds / 1e9:F1} GMAC/s  {maddubs / sw.Elapsed.TotalSeconds / 1e9:F2} G-maddubs/s (peak ~9 at 4.5GHz)");
}

static unsafe void MicroVecQ4(int nIn, int nOut, int tokens, int reps, int threads)
{
    int nb = nIn / Qk.SuperBlock;
    using CpuThreadPool pool = new(threads);
    using NativeBuffer q4 = new((nuint)((long)nOut * nb * Qk.Q4KSize));
    using NativeBuffer dst = new((nuint)((long)tokens * nOut * sizeof(float)));
    using NativeBuffer src = new((nuint)((long)tokens * nIn * sizeof(float)));
    using ScratchArena scratch = new();
    float* input = (float*)src.Pointer;
    for (long i = 0; i < (long)tokens * nIn; i++)
        input[i] = MathF.Sin(i * 0.37f);
    BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
    for (int r = 0; r < nOut; r++)
        Q4K.PackSimple(input + (r % tokens) * nIn, rows + r * nb, nIn);

    Console.WriteLine($"micro-vec-q4 in={nIn} out={nOut} tokens={tokens} threads={threads} V={System.Numerics.Vector<float>.Count} portable={Simd.ForcePortable}");
    MulMatQ4K.Gemm(null, rows, input, (float*)dst.Pointer, nIn, nOut, tokens, pool, scratch);
    Stopwatch sw = Stopwatch.StartNew();
    for (int i = 0; i < reps; i++)
        MulMatQ4K.Gemm(null, rows, input, (float*)dst.Pointer, nIn, nOut, tokens, pool, scratch);
    sw.Stop();
    double flops = 2.0 * nIn * nOut * tokens * reps;
    Console.WriteLine($"  gemm  {sw.Elapsed.TotalMilliseconds / reps:F2} ms/call  {flops / sw.Elapsed.TotalSeconds / 1e9:F1} GFLOP/s");

    int gemvReps = reps * 200;
    for (int i = 0; i < 50; i++)
        MulMatQ4K.Gemv(rows, input, (float*)dst.Pointer, nIn, nOut, pool, scratch);
    sw.Restart();
    for (int i = 0; i < gemvReps; i++)
        MulMatQ4K.Gemv(rows, input, (float*)dst.Pointer, nIn, nOut, pool, scratch);
    sw.Stop();
    double bytes = (double)nOut * nb * Qk.Q4KSize * gemvReps;
    Console.WriteLine($"  gemv  {sw.Elapsed.TotalMilliseconds * 1000 / gemvReps:F1} us/call  {bytes / sw.Elapsed.TotalSeconds / 1e9:F1} GB/s");
}

static unsafe void MicroQ8(int nIn, int nOut, int reps, int threads)
{
    int nb = nIn / Qk.Q8_0Block;
    using NativeBuffer q8 = new((nuint)((long)nOut * nb * Qk.Q8_0Size));
    using NativeBuffer q8x8 = new((nuint)((long)(nOut / 8) * nb * Qk.Q8_0x8Size));
    using NativeBuffer act = new((nuint)(nb * Qk.Q8_0ActSize));
    using NativeBuffer dst = new((nuint)(nOut * sizeof(float)));
    using NativeBuffer src = new((nuint)(nIn * sizeof(float)));
    float* input = (float*)src.Pointer;
    for (int i = 0; i < nIn; i++)
        input[i] = MathF.Sin(i * 0.37f);
    BlockQ8_0* rows = (BlockQ8_0*)q8.Pointer;
    for (int r = 0; r < nOut * nb; r++)
    {
        rows[r].D = Sdcb.HyMT2Sharp.Kernels.HalfBits.FromSingle(0.5f);
        for (int k = 0; k < Qk.Q8_0Block; k++)
            rows[r].Qs[k] = (sbyte)((r * 31 + k * 7) & 0x7F);
    }

    RepackQ8_0.Rows(rows, (BlockQ8_0x8*)q8x8.Pointer, nIn, nOut);
    Q8_0.QuantizeActs(input, (BlockQ8_0Act*)act.Pointer, nIn);
    using CpuThreadPool pool = new(threads);
    for (int i = 0; i < 20; i++)
        Q8_0.GemvPrequant(rows, (BlockQ8_0Act*)act.Pointer, (float*)dst.Pointer, nIn, nOut, pool);
    Stopwatch sw = Stopwatch.StartNew();
    for (int i = 0; i < reps; i++)
        Q8_0.GemvPrequant(rows, (BlockQ8_0Act*)act.Pointer, (float*)dst.Pointer, nIn, nOut, pool);
    sw.Stop();
    double bytes = (double)nOut * nb * Qk.Q8_0Size * reps;
    Console.WriteLine($"micro-q8 gemv-rows in={nIn} out={nOut} {sw.Elapsed.TotalMilliseconds:F1} ms  {bytes / sw.Elapsed.TotalSeconds / 1e9:F1} GB/s");

    for (int i = 0; i < 20; i++)
        Q8_0.GemvPacked((BlockQ8_0x8*)q8x8.Pointer, rows, (BlockQ8_0Act*)act.Pointer, (float*)dst.Pointer, nIn, nOut, pool);
    sw.Restart();
    for (int i = 0; i < reps; i++)
        Q8_0.GemvPacked((BlockQ8_0x8*)q8x8.Pointer, rows, (BlockQ8_0Act*)act.Pointer, (float*)dst.Pointer, nIn, nOut, pool);
    sw.Stop();
    bytes = (double)(nOut / 8) * nb * Qk.Q8_0x8Size * reps;
    Console.WriteLine($"micro-q8 gemv-packed in={nIn} out={nOut} {sw.Elapsed.TotalMilliseconds:F1} ms  {bytes / sw.Elapsed.TotalSeconds / 1e9:F1} GB/s");
}

/// <summary>Replay one decode token's worth of Q8_0 GEMVs at model shapes.</summary>
static unsafe void MicroQ8Model(int reps, int threads, bool useArena)
{
    const int hidden = 2048;
    const int ffn = 6144;
    const int qDim = 2048;
    const int kDim = 512;
    const int layers = 32;
    const int vocab = 120816;
    (int nIn, int nOut)[] shapes =
    [
        (hidden, qDim), (hidden, kDim), (hidden, kDim), (qDim, hidden),
        (hidden, ffn), (hidden, ffn), (ffn, hidden),
    ];
    nuint[] wsBytes = new nuint[shapes.Length];
    for (int i = 0; i < shapes.Length; i++)
        wsBytes[i] = (nuint)((long)(shapes[i].nOut / 8) * (shapes[i].nIn / Qk.Q8_0Block) * Qk.Q8_0x8Size);
    using NativeBuffer lm = new((nuint)((long)(vocab / 8) * (hidden / Qk.Q8_0Block) * Qk.Q8_0x8Size));
    NativeBuffer[] ws = new NativeBuffer[shapes.Length * layers];
    NativeBuffer[] dsts = new NativeBuffer[shapes.Length];
    nuint[] wptr = new nuint[shapes.Length * layers];
    NativeBuffer? arena = null;
    try
    {
        if (useArena)
        {
            nuint total = 0;
            for (int i = 0; i < shapes.Length; i++)
                total += wsBytes[i] * (nuint)layers;
            arena = new NativeBuffer(total + (nuint)(shapes.Length * layers) * 4096);
            nuint at = 0;
            for (int i = 0; i < shapes.Length; i++)
            {
                for (int l = 0; l < layers; l++)
                {
                    wptr[i * layers + l] = (nuint)arena.Pointer + at;
                    at += wsBytes[i];
                    at = (at + 4095) & ~(nuint)4095;
                }

                dsts[i] = new NativeBuffer((nuint)(shapes[i].nOut * sizeof(float)));
            }
        }
        else
        {
            for (int i = 0; i < shapes.Length; i++)
            {
                for (int l = 0; l < layers; l++)
                {
                    NativeBuffer b = ws[i * layers + l] = new(wsBytes[i]);
                    NativeMemory.Clear(b.Pointer, Math.Min(b.Bytes, (nuint)4096));
                    wptr[i * layers + l] = (nuint)b.Pointer;
                }

                dsts[i] = new NativeBuffer((nuint)(shapes[i].nOut * sizeof(float)));
            }
        }

        using NativeBuffer actB = new((nuint)((ffn / Qk.Q8_0Block) * Qk.Q8_0ActSize));
        using NativeBuffer lmDst = new((nuint)(vocab * sizeof(float)));
        using CpuThreadPool pool = new(threads);
        void LayerToken(int l)
        {
            BlockQ8_0x8* wq = (BlockQ8_0x8*)wptr[0 * layers + l];
            BlockQ8_0x8* wk = (BlockQ8_0x8*)wptr[1 * layers + l];
            BlockQ8_0x8* wv = (BlockQ8_0x8*)wptr[2 * layers + l];
            BlockQ8_0Act* act = (BlockQ8_0Act*)actB.Pointer;
            Q8_0.GemvPackedMulti(act, hidden, pool,
                new Q8GemvTarget(wq, null, (float*)dsts[0].Pointer, qDim),
                new Q8GemvTarget(wk, null, (float*)dsts[1].Pointer, kDim),
                new Q8GemvTarget(wv, null, (float*)dsts[2].Pointer, kDim));
            Q8_0.GemvPacked((BlockQ8_0x8*)wptr[3 * layers + l], null, act, (float*)dsts[3].Pointer, qDim, hidden, pool);
            Q8_0.GemvPackedMulti(act, hidden, pool,
                new Q8GemvTarget((BlockQ8_0x8*)wptr[4 * layers + l], null, (float*)dsts[4].Pointer, ffn),
                new Q8GemvTarget((BlockQ8_0x8*)wptr[5 * layers + l], null, (float*)dsts[5].Pointer, ffn));
            Q8_0.GemvPacked((BlockQ8_0x8*)wptr[6 * layers + l], null, act, (float*)dsts[6].Pointer, ffn, hidden, pool);
        }

        for (int i = 0; i < 2; i++)
        {
            for (int l = 0; l < layers; l++)
                LayerToken(l);
            Q8_0.GemvPacked((BlockQ8_0x8*)lm.Pointer, null, (BlockQ8_0Act*)actB.Pointer, (float*)lmDst.Pointer, hidden, vocab, pool);
        }

        Stopwatch sw = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++)
        {
            for (int l = 0; l < layers; l++)
                LayerToken(l);
            Q8_0.GemvPacked((BlockQ8_0x8*)lm.Pointer, null, (BlockQ8_0Act*)actB.Pointer, (float*)lmDst.Pointer, hidden, vocab, pool);
        }

        sw.Stop();
        long perToken = 0;
        foreach (nuint b in wsBytes)
            perToken += (long)b;
        perToken *= layers;
        perToken += (long)lm.Bytes;
        double gemvMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"micro-q8 model-seq {gemvMs:F1} ms  {perToken * (double)reps / gemvMs / 1e6:F1} GB/s  {reps / (gemvMs / 1000.0):F2} tok/s-equiv");

        // Pure sequential read ceiling: same buffers, load-only kernel.
        for (int i = 0; i < 2; i++)
        {
            for (int l = 0; l < layers; l++)
                for (int s = 0; s < shapes.Length; s++)
                    ReadOnly.Run((void*)wptr[s * layers + l], wsBytes[s], pool);
            ReadOnly.Run(lm.Pointer, lm.Bytes, pool);
        }

        sw.Restart();
        for (int r = 0; r < reps; r++)
        {
            for (int l = 0; l < layers; l++)
                for (int s = 0; s < shapes.Length; s++)
                    ReadOnly.Run((void*)wptr[s * layers + l], wsBytes[s], pool);
            ReadOnly.Run(lm.Pointer, lm.Bytes, pool);
        }

        sw.Stop();
        Console.WriteLine($"micro-q8 read-only  {sw.Elapsed.TotalMilliseconds:F1} ms  {perToken * (double)reps / sw.Elapsed.TotalSeconds / 1e9:F1} GB/s");
    }
    finally
    {
        foreach (NativeBuffer b in ws)
            b?.Dispose();
        foreach (NativeBuffer d in dsts)
            d?.Dispose();
        arena?.Dispose();
    }
}

static unsafe void DumpQ4Gemm()
{
    const int nIn = 256;
    const int nOut = 16;
    const int tokens = 4;
    using NativeBuffer q4 = new((nuint)((long)nOut * (nIn / Qk.SuperBlock) * Qk.Q4KSize));
    using NativeBuffer q4x8 = new((nuint)((long)(nOut / 8) * (nIn / Qk.SuperBlock) * Qk.Q4Kx8Size));
    using NativeBuffer q8x4 = new((nuint)((nIn / Qk.SuperBlock) * Qk.Q8Kx4Size));
    using NativeBuffer meta = new((nuint)((long)(nOut / 8) * (nIn / Qk.SuperBlock) * Qk.Q4Kx8MetaSize));
    using NativeBuffer dst = new((nuint)((long)tokens * nOut * sizeof(float)));
    using NativeBuffer src = new((nuint)((long)tokens * nIn * sizeof(float)));
    float* input = (float*)src.Pointer;
    for (int i = 0; i < tokens * nIn; i++)
        input[i] = (i % 17) * 0.01f;
    BlockQ4K* rows = (BlockQ4K*)q4.Pointer;
    int nb = nIn / Qk.SuperBlock;
    for (int r = 0; r < nOut; r++)
        Q4K.PackSimple(input, rows + r * nb, nIn);
    RepackQ4K.Rows(rows, (BlockQ4Kx8*)q4x8.Pointer, nIn, nOut);
    RepackQ4K.BuildMeta(rows, (BlockQ4Kx8Meta*)meta.Pointer, nIn, nOut);
    QuantizeQ8Kx4.Quantize4x8(input, (BlockQ8Kx4*)q8x4.Pointer, nIn);
    GemmQ4K.GemmAvx2(nIn, (float*)dst.Pointer, nOut, (BlockQ4Kx8*)q4x8.Pointer, (BlockQ8Kx4*)q8x4.Pointer, tokens, nOut, (BlockQ4Kx8Meta*)meta.Pointer);
    Console.WriteLine($"dump-q4-gemm dst0={*((float*)dst.Pointer)}");
}

static unsafe void DumpSilu()
{
    const int n = 64;
    using NativeBuffer gate = new((nuint)(n * sizeof(float)));
    using NativeBuffer up = new((nuint)(n * sizeof(float)));
    float* g = (float*)gate.Pointer;
    float* u = (float*)up.Pointer;
    for (int i = 0; i < n; i++)
    {
        g[i] = (i - 32) * 0.1f;
        u[i] = 1.25f;
    }

    Ops.SiLUMulRange(g, u, 0, n);
    Console.WriteLine($"dump-silu g0={g[0]}");
}

static unsafe int ArgMax(float[] logits)
{
    int n = logits.Length;
    int best = 0;
    float max;
    fixed (float* p = logits)
    {
        if (Avx.IsSupported)
        {
            var vmax = Avx.LoadVector256(p);
            var vidx = Vector256.Create(0, 1, 2, 3, 4, 5, 6, 7);
            var vbest = vidx;
            var step = Vector256.Create(8);
            int i = 8;
            for (; i + 8 <= n; i += 8)
            {
                var v = Avx.LoadVector256(p + i);
                vidx = Avx2.Add(vidx, step);
                var gt = Avx.Compare(v, vmax, FloatComparisonMode.OrderedGreaterThanNonSignaling);
                vmax = Avx.BlendVariable(vmax, v, gt);
                vbest = Avx2.BlendVariable(vbest, vidx, gt.AsInt32());
            }

            best = 0;
            max = vmax.GetElement(0);
            for (int l = 1; l < 8; l++)
            {
                if (vmax.GetElement(l) > max)
                {
                    max = vmax.GetElement(l);
                    best = vbest.GetElement(l);
                }
            }

            for (; i < n; i++)
            {
                if (p[i] > max)
                {
                    max = p[i];
                    best = i;
                }
            }

            return best;
        }

        max = p[0];
        for (int i = 1; i < n; i++)
        {
            if (p[i] > max)
            {
                max = p[i];
                best = i;
            }
        }
    }

    return best;
}

static unsafe class ReadOnly
{
    private static Vector256<byte> _sink;

    public static void Run(void* ptr, nuint bytes, CpuThreadPool pool)
    {
        long n = (long)bytes / 32;
        pool.For(1024, (int worker, int workers) =>
        {
            long begin = n * worker / workers;
            long end = n * (worker + 1) / workers;
            byte* p = (byte*)ptr + begin * 32;
            Vector256<byte> acc = Vector256<byte>.Zero;
            for (long i = begin; i < end; i++)
            {
                acc = Avx2.Xor(acc, Avx.LoadVector256(p));
                p += 32;
            }

            _sink = acc;
        });
    }
}

static class Args
{
    public static string? Get(string[] args, string name)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : null;
    }

    public static int GetInt(string[] args, string name, int fallback)
        => int.TryParse(Get(args, name), out int value) ? value : fallback;
}
