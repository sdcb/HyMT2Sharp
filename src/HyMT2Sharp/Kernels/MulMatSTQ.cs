using System.Threading;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public readonly unsafe struct STQPanelWeight(BlockSTQ1_0x8* packed, float* dst, int nOut)
{
    public readonly BlockSTQ1_0x8* Packed = packed;
    public readonly float* Dst = dst;
    public readonly int NOut = nOut;
}

/// <summary>
/// STQ1_0 matrix multiplication.  Decode retains a raw-row fallback for
/// tails and non-AVX2 hosts; aligned eight-row tiles use the packed panel
/// kernel and share one Q8_K quantization across all projections.
/// </summary>
public static unsafe class MulMatSTQ
{
    public static void Gemv(BlockSTQ1_0x8* packed, BlockSTQ1_0* rows, float* input, float* output,
        int nIn, int nOut, CpuThreadPool? pool, ScratchArena scratch)
    {
        BlockQ8K* x = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, x, nIn);
        GemvPrequant(packed, rows, x, output, nIn, nOut, pool);
    }

    public static void GemvPrequant(BlockSTQ1_0x8* packed, BlockSTQ1_0* rows, BlockQ8K* x,
        float* output, int nIn, int nOut, CpuThreadPool? pool)
    {
        if (packed == null || !Simd.UsePanels)
        {
            GemvPrequant(rows, x, output, nIn, nOut, pool);
            return;
        }
        int groups = nOut / 8;
        int tail = nOut - groups * 8;
        int nb = nIn / STQ1_0.BlockLength;
        int gCursor = 0;
        int tCursor = 0;
        void Run(int worker, int workers)
        {
            while (true)
            {
                int g = Interlocked.Increment(ref gCursor) - 1;
                if (g >= groups)
                    break;
                STQPanel.Gemv(packed + (long)g * nb, x, output + g * 8, nIn);
            }
            while (true)
            {
                int r = Interlocked.Increment(ref tCursor) - 1;
                if (r >= tail)
                    break;
                output[groups * 8 + r] = STQ1_0.Dot(rows + (long)(groups * 8 + r) * nb, x, nIn);
            }
        }
        if (pool == null) Run(0, 1); else pool.For(Math.Max(groups, tail), Run);
    }

    public static void Gemm(BlockSTQ1_0x8* packed, BlockSTQ1_0* rows, float* input, float* output,
        int nIn, int nOut, int tokens, CpuThreadPool? pool, ScratchArena scratch)
    {
        if (packed != null && Simd.UsePanels && tokens == 1)
        {
            Gemv(packed, rows, input, output, nIn, nOut, pool, scratch);
            return;
        }
        if (packed == null || !Simd.UsePanels || (tokens & 3) != 0 || (nOut & 7) != 0)
        {
            Gemm(rows, input, output, nIn, nOut, tokens, pool, scratch);
            return;
        }
        BlockQ8Kx4* q8 = (BlockQ8Kx4*)scratch.E((nuint)((long)(tokens / 4) * (nIn / STQ1_0.BlockLength) * Qk.Q8Kx4Size));
        QuantizeAndGemm(input, q8, nIn, tokens, pool, new STQPanelWeight(packed, output, nOut));
    }

    /// <summary>Quantize four-token groups once and run the packed STQ panels.</summary>
    public static void QuantizeAndGemm(float* input, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, STQPanelWeight w0, STQPanelWeight w1 = default, STQPanelWeight w2 = default)
        => RunQuantized(input, null, q8, nIn, tokens, pool, w0, w1, w2);

    /// <summary>Fused SiLU(gate)*up quantization followed by packed STQ panels.</summary>
    public static void SiluQuantizeAndGemm(float* gate, float* up, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, STQPanelWeight w)
        => RunQuantized(gate, up, q8, nIn, tokens, pool, w, default, default);

    public static void Gemv(BlockSTQ1_0* rows, float* input, float* output,
        int nIn, int nOut, CpuThreadPool? pool, ScratchArena scratch)
    {
        BlockQ8K* x = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, x, nIn);
        GemvPrequant(rows, x, output, nIn, nOut, pool);
    }

    public static void GemvPrequant(BlockSTQ1_0* rows, BlockQ8K* x, float* output,
        int nIn, int nOut, CpuThreadPool? pool)
    {
        if (nIn <= 0 || nIn % STQ1_0.BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows require a positive multiple of 256 inputs.", nameof(nIn));

        int nb = nIn / STQ1_0.BlockLength;
        void Run(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            for (int r = begin; r < end; r++)
                output[r] = STQ1_0.Dot(rows + r * nb, x, nIn);
        }

        if (pool == null)
            Run(0, 1);
        else
            pool.For(nOut, Run);
    }

    public static void Gemm(BlockSTQ1_0* rows, float* input, float* output,
        int nIn, int nOut, int tokens, CpuThreadPool? pool, ScratchArena scratch)
    {
        if (nIn <= 0 || nIn % STQ1_0.BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows require a positive multiple of 256 inputs.", nameof(nIn));
        if (nOut <= 0 || tokens <= 0)
            return;
        if (tokens == 1)
        {
            Gemv(rows, input, output, nIn, nOut, pool, scratch);
            return;
        }

        int nb = nIn / STQ1_0.BlockLength;
        VecGemmF.Gemm((byte*)rows, nb * sizeof(BlockSTQ1_0), sizeof(BlockSTQ1_0), STQ1_0.BlockLength,
            &DequantBlock, input, output, nIn, nOut, tokens, pool);
    }

    private static void DequantBlock(byte* p, float* dst) =>
        STQ1_0.DequantizeRow((BlockSTQ1_0*)p, dst, STQ1_0.BlockLength);

    private static void RunQuantized(float* input, float* up, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, STQPanelWeight w0, STQPanelWeight w1, STQPanelWeight w2)
    {
        if (!Simd.UsePanels)
            throw new PlatformNotSupportedException("STQ panels require AVX2 or AdvSimd+SDOT.");
        if (nIn <= 0 || nIn % STQ1_0.BlockLength != 0 || tokens <= 0 || (tokens & 3) != 0)
            throw new ArgumentException("STQ panels require a positive multiple of 256 inputs and four tokens.");
        ValidatePanel(w0);
        ValidatePanel(w1);
        ValidatePanel(w2);

        int nb = nIn / STQ1_0.BlockLength;
        int quantGroups = tokens / 4;
        int quantCursor = 0;
        int c0 = 0, c1 = 0, c2 = 0;
        int count = Math.Max(quantGroups, Math.Max(w0.NOut / 8, Math.Max(w1.NOut / 8, w2.NOut / 8)));
        void Run(int worker, int workers)
        {
            while (true)
            {
                int g = Interlocked.Increment(ref quantCursor) - 1;
                if (g >= quantGroups)
                    break;
                if (up == null)
                    QuantizeQ8Kx4.Quantize4x8(input + (long)g * 4 * nIn, q8 + (long)g * nb, nIn);
                else
                    QuantizeQ8Kx4.Quantize4x8Silu(input + (long)g * 4 * nIn, up + (long)g * 4 * nIn, q8 + (long)g * nb, nIn);
            }
            if (workers > 1)
                pool!.Barrier();
            RunRange(w0, q8, nIn, tokens, ref c0);
            RunRange(w1, q8, nIn, tokens, ref c1);
            RunRange(w2, q8, nIn, tokens, ref c2);
        }

        if (pool == null)
            Run(0, 1);
        else
            pool.For(count, Run);
    }

    private static void ValidatePanel(STQPanelWeight w)
    {
        if (w.NOut != 0 && (w.NOut < 0 || (w.NOut & 7) != 0 || w.Packed == null || w.Dst == null))
            throw new ArgumentException("STQ panels require valid buffers and an output count divisible by eight.");
    }

    /// <summary>Workers claim column groups until the weight is exhausted (dynamic, not static).</summary>
    private static void RunRange(STQPanelWeight w, BlockQ8Kx4* q8, int nIn, int tokens, ref int cursor)
    {
        if (w.NOut == 0)
            return;
        int nb = nIn / STQ1_0.BlockLength;
        int groups = w.NOut / 8;
        while (true)
        {
            int g = Interlocked.Increment(ref cursor) - 1;
            if (g >= groups)
                break;
            STQPanel.Gemm(w.Packed + (long)g * nb, q8, w.Dst + g * 8, nIn, w.NOut, tokens, 1);
        }
    }
}
