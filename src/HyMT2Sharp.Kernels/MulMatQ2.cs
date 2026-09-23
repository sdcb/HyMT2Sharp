using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public readonly unsafe struct Q2PanelWeight(BlockQ2x8* packed, float* dst, int nOut)
{
    public readonly BlockQ2x8* Packed = packed;
    public readonly float* Dst = dst;
    public readonly int NOut = nOut;
}

/// <summary>Q2_0C matmul using compact 2-bit panels and Q8_K activations.</summary>
public static unsafe class MulMatQ2
{
    // Keep a column tile in the per-core L2 while streaming all token groups.
    public static int ColTileBytes = 64 * 1024;

    public static void Gemv(BlockQ2x8* packed, BlockQ2_0C* rows, float* input, float* output,
        int nIn, int nOut, CpuThreadPool? pool, ScratchArena scratch)
    {
        BlockQ8K* q8 = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, q8, nIn);
        GemvPrequant(packed, rows, q8, output, nIn, nOut, pool);
    }

    public static void GemvPrequant(BlockQ2x8* packed, BlockQ2_0C* rows, BlockQ8K* x,
        float* y, int nIn, int nOut, CpuThreadPool? pool)
    {
        int nb = nIn / Q2_0C.BlockLength;
        int groups = packed != null && Simd.UseAvx2 ? nOut / 8 : 0;
        int tail = nOut - groups * 8;
        void Run(int worker, int workers)
        {
            int begin = groups * worker / workers;
            int end = groups * (worker + 1) / workers;
            for (int g = begin; g < end; g++)
                Q2Panel.Gemv(packed + g * nb, x, y + g * 8, nIn);
            for (int r = groups * 8 + tail * worker / workers; r < groups * 8 + tail * (worker + 1) / workers; r++)
                y[r] = Q2_0C.Dot(rows + r * nb, x, nIn);
        }
        if (pool == null) Run(0, 1); else pool.For(Math.Max(groups, tail), Run);
    }

    public static void Gemm(BlockQ2x8* packed, BlockQ2_0C* rows, float* input, float* output,
        int nIn, int nOut, int tokens, CpuThreadPool? pool, ScratchArena scratch)
    {
        if (nIn <= 0 || nIn % Q2_0C.BlockLength != 0)
            throw new ArgumentException("Q2 rows must contain a positive multiple of 512 values.", nameof(nIn));
        if (tokens <= 0 || nOut <= 0)
            return;
        if (tokens == 1)
        {
            Gemv(packed, rows, input, output, nIn, nOut, pool, scratch);
            return;
        }
        if (!Simd.UseAvx2 || packed == null)
        {
            int nbv = nIn / Q2_0C.BlockLength;
            VecGemmF.Gemm((byte*)rows, nbv * sizeof(BlockQ2_0C), sizeof(BlockQ2_0C), Q2_0C.BlockLength,
                &DequantBlock, input, output, nIn, nOut, tokens, pool);
            return;
        }

        int nb = nIn / Q2_0C.BlockLength;
        int groups = nOut / 8;
        int padded = (tokens + 3) & ~3;
        int q8Blocks = nIn / Qk.SuperBlock;
        BlockQ8Kx4* q8 = (BlockQ8Kx4*)scratch.E((nuint)((long)(padded / 4) * q8Blocks * Qk.Q8Kx4Size));
        float* src = input;
        float* dst = output;
        if (padded != tokens)
        {
            src = (float*)scratch.B((nuint)((long)padded * nIn * sizeof(float)));
            dst = (float*)scratch.C((nuint)((long)padded * nOut * sizeof(float)));
            Buffer.MemoryCopy(input, src, (long)padded * nIn * sizeof(float), (long)tokens * nIn * sizeof(float));
            NativeMemory.Clear(src + tokens * nIn, (nuint)((long)(padded - tokens) * nIn * sizeof(float)));
        }
        int quantGroups = padded / 4;
        void Run(int worker, int workers)
        {
            for (int g = quantGroups * worker / workers; g < quantGroups * (worker + 1) / workers; g++)
                QuantizeQ8Kx4.Quantize4x8(src + g * 4 * nIn, q8 + g * q8Blocks, nIn);
            if (workers > 1) pool!.Barrier();
            RunRange(packed, dst, groups, q8, nIn, nOut, padded, nb, worker, workers);
        }
        if (pool == null) Run(0, 1); else pool.For(Math.Max(quantGroups, groups), Run);

        if (nOut % 8 != 0)
        {
            BlockQ8K* rowQ8 = (BlockQ8K*)scratch.D((nuint)Q8K.RowBytes(nIn));
            for (int t = 0; t < tokens; t++)
            {
                Q8K.QuantizeRow(input + t * nIn, rowQ8, nIn);
                for (int r = groups * 8; r < nOut; r++)
                    dst[t * nOut + r] = Q2_0C.Dot(rows + r * nb, rowQ8, nIn);
            }
        }
        if (padded != tokens)
            Buffer.MemoryCopy(dst, output, (long)tokens * nOut * sizeof(float), (long)tokens * nOut * sizeof(float));
    }

    /// <summary>One Q8 conversion and parallel region shared by up to three Q2 matrices.</summary>
    public static void QuantizeAndGemm(float* input, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, Q2PanelWeight w0, Q2PanelWeight w1 = default, Q2PanelWeight w2 = default) =>
        RunQuantized(input, null, q8, nIn, tokens, pool, w0, w1, w2);

    /// <summary>SiLU(gate) * up, Q8 conversion, barrier, and down projection.</summary>
    public static void SiluQuantizeAndGemm(float* gate, float* up, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, Q2PanelWeight w) =>
        RunQuantized(gate, up, q8, nIn, tokens, pool, w, default, default);

    private static void RunQuantized(float* input, float* up, BlockQ8Kx4* q8, int nIn, int tokens,
        CpuThreadPool? pool, Q2PanelWeight w0, Q2PanelWeight w1, Q2PanelWeight w2)
    {
        if (!Simd.UseAvx2)
            throw new PlatformNotSupportedException("Q2 panels require AVX2.");
        if (nIn <= 0 || nIn % Q2_0C.BlockLength != 0 || tokens <= 0 || (tokens & 3) != 0)
            throw new ArgumentException("Q2 panels require a positive multiple of 512 inputs and four tokens.");
        ValidatePanel(w0);
        ValidatePanel(w1);
        ValidatePanel(w2);
        int nb = nIn / Q2_0C.BlockLength;
        int q8Blocks = nIn / Qk.SuperBlock;
        int quantGroups = tokens / 4;
        int count = Math.Max(quantGroups, Math.Max(w0.NOut / 8, Math.Max(w1.NOut / 8, w2.NOut / 8)));
        void Run(int worker, int workers)
        {
            int begin = quantGroups * worker / workers;
            int end = quantGroups * (worker + 1) / workers;
            if (up == null)
                for (int g = begin; g < end; g++)
                    QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, q8 + g * q8Blocks, nIn);
            else
                for (int g = begin; g < end; g++)
                    QuantizeQ8Kx4.Quantize4x8Silu(input + g * 4 * nIn, up + g * 4 * nIn, q8 + g * q8Blocks, nIn);
            if (workers > 1) pool!.Barrier();
            RunRange(w0.Packed, w0.Dst, w0.NOut / 8, q8, nIn, w0.NOut, tokens, nb, worker, workers);
            RunRange(w1.Packed, w1.Dst, w1.NOut / 8, q8, nIn, w1.NOut, tokens, nb, worker, workers);
            RunRange(w2.Packed, w2.Dst, w2.NOut / 8, q8, nIn, w2.NOut, tokens, nb, worker, workers);
        }
        if (pool == null) Run(0, 1); else pool.For(count, Run);
    }

    private static void ValidatePanel(Q2PanelWeight w)
    {
        if (w.NOut != 0 && (w.NOut < 0 || (w.NOut & 7) != 0 || w.Packed == null || w.Dst == null))
            throw new ArgumentException("Q2 panels require valid buffers and an output count divisible by eight.");
    }

    private static void DequantBlock(byte* p, float* dst) =>
        Q2_0C.DequantizeRow((BlockQ2_0C*)p, dst, Q2_0C.BlockLength);

    private static void RunRange(BlockQ2x8* packed, float* output, int groups, BlockQ8Kx4* q8,
        int nIn, int nOut, int tokens, int nb, int worker, int workers)
    {
        if (packed == null || groups == 0)
            return;
        int begin = groups * worker / workers;
        int end = groups * (worker + 1) / workers;
        int tile = Math.Max(1, ColTileBytes / (nb * sizeof(BlockQ2x8)));
        for (int g = begin; g < end; g += tile)
            Q2Panel.Gemm(packed + g * nb, q8, output + g * 8, nIn, nOut, tokens, Math.Min(tile, end - g));
    }
}
