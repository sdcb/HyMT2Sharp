using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Q4_K × F32 matmul. Decode (1 token) is AVX2 GEMV. Prefill is Q8_K 4×8
/// panel GEMM against q4_Kx8 weights.
/// </summary>
public static unsafe class MulMatQ4K
{
    public static void Gemv(BlockQ4K* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        nuint q8Bytes = (nuint)Q8K.RowBytes(nIn);
        NativeBuffer? owned = scratch == null ? new NativeBuffer(q8Bytes) : null;
        BlockQ8K* y = (BlockQ8K*)(scratch != null ? scratch.D(q8Bytes) : owned!.Pointer);
        try
        {
            Q8K.QuantizeRow(input, y, nIn);
            GemvPrequant(weights, y, output, nIn, nOut, pool);
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void GemvPrequant(BlockQ4K* weights, BlockQ8K* y, float* output, int nIn, int nOut, CpuThreadPool? pool = null)
    {
        int nb = nIn / Qk.SuperBlock;
        void Body(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            for (int row = begin; row < end; row++)
                output[row] = VecDotQ4K.Dot(weights + row * nb, y, nIn);
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(nOut, Body);
    }

    public static void Gemm(BlockQ4Kx8* packed, BlockQ4K* rows, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null, BlockQ4Kx8Meta* meta = null)
    {
        if (tokens == 1)
        {
            Gemv(rows, input, output, nIn, nOut, pool, scratch);
            return;
        }

        if (packed == null || !Simd.UseAvx2)
        {
            GemmRows(rows, input, output, nIn, nOut, tokens, pool, scratch);
            return;
        }

        int paddedTokens = (tokens + 3) & ~3;
        int packedCols = nOut & ~7;
        int nb = nIn / Qk.SuperBlock;
        bool aligned = paddedTokens == tokens;
        nuint actBytes = (nuint)((paddedTokens / 4) * nb * Qk.Q8Kx4Size);
        nuint stagingBytes = aligned ? 0 : (nuint)((long)paddedTokens * nIn * sizeof(float));
        nuint outBytes = aligned ? 0 : (nuint)((long)paddedTokens * nOut * sizeof(float));
        NativeBuffer? actOwn = null;
        NativeBuffer? stagingOwn = null;
        NativeBuffer? scratchOutOwn = null;
        BlockQ8Kx4* q8;
        float* src = input;
        float* dst = output;
        if (scratch != null)
        {
            q8 = (BlockQ8Kx4*)scratch.A(actBytes);
            if (!aligned)
            {
                src = (float*)scratch.B(stagingBytes);
                dst = (float*)scratch.C(outBytes);
            }
        }
        else
        {
            actOwn = new NativeBuffer(actBytes);
            q8 = (BlockQ8Kx4*)actOwn.Pointer;
            if (!aligned)
            {
                stagingOwn = new NativeBuffer(stagingBytes);
                scratchOutOwn = new NativeBuffer(outBytes);
                src = (float*)stagingOwn.Pointer;
                dst = (float*)scratchOutOwn.Pointer;
            }
        }

        try
        {
            if (!aligned)
            {
                Buffer.MemoryCopy(input, src, (long)tokens * nIn * sizeof(float), (long)tokens * nIn * sizeof(float));
                NativeMemory.Clear(src + tokens * nIn, (nuint)((long)(paddedTokens - tokens) * nIn * sizeof(float)));
            }

            QuantizeAndGemmPrequant(src, q8, packed, dst, nIn, nOut, paddedTokens, pool, meta);

            if (packedCols < nOut)
            {
                nuint q8Bytes = (nuint)Q8K.RowBytes(nIn);
                NativeBuffer? q8rowOwn = scratch == null ? new NativeBuffer(q8Bytes) : null;
                BlockQ8K* q8row = (BlockQ8K*)(scratch != null ? scratch.D(q8Bytes) : q8rowOwn!.Pointer);
                try
                {
                    for (int t = 0; t < tokens; t++)
                    {
                        Q8K.QuantizeRow(input + t * nIn, q8row, nIn);
                        for (int row = packedCols; row < nOut; row++)
                            dst[t * nOut + row] = VecDotQ4K.Dot(rows + row * nb, q8row, nIn);
                    }
                }
                finally
                {
                    q8rowOwn?.Dispose();
                }
            }

            if (!aligned)
                Buffer.MemoryCopy(dst, output, (long)tokens * nOut * sizeof(float), (long)tokens * nOut * sizeof(float));
        }
        finally
        {
            actOwn?.Dispose();
            stagingOwn?.Dispose();
            scratchOutOwn?.Dispose();
        }
    }

    /// <summary>
    /// Portable prefill fallback: dequantize-then-FMA GEMM over
    /// <see cref="Vector{T}"/>. Weight decode amortizes over tokens and row
    /// tiles share activation reads; no Q8_K activation quantization needed.
    /// </summary>
    private static void GemmRows(BlockQ4K* rows, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool, ScratchArena? scratch)
    {
        int nb = nIn / Qk.SuperBlock;
        VecGemmF.Gemm((byte*)rows, nb * sizeof(BlockQ4K), sizeof(BlockQ4K), Qk.SuperBlock,
            &DequantBlock, input, output, nIn, nOut, tokens, pool);
    }

    private static void DequantBlock(byte* p, float* dst) =>
        Q4K.DequantizeRow((BlockQ4K*)p, dst, Qk.SuperBlock);

    public static void QuantizeAligned(float* input, BlockQ8Kx4* q8, int nIn, int tokens, CpuThreadPool? pool)
    {
        int nb = nIn / Qk.SuperBlock;
        int quantGroups = tokens / 4;
        void QuantBody(int worker, int workers)
        {
            int begin = quantGroups * worker / workers;
            int end = quantGroups * (worker + 1) / workers;
            for (int g = begin; g < end; g++)
                QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, q8 + g * nb, nIn);
        }

        if (pool == null)
            QuantBody(0, 1);
        else
            pool.For(quantGroups, QuantBody);
    }

    /// <summary>
    /// llama.cpp mul_mat: all threads quantize src1, spin-barrier, then GEMM tiles.
    /// </summary>
    public static void QuantizeAndGemmPrequant(
        float* input,
        BlockQ8Kx4* q8,
        BlockQ4Kx8* packed,
        float* dst,
        int nIn,
        int nOut,
        int tokens,
        CpuThreadPool? pool,
        BlockQ4Kx8Meta* meta)
    {
        if (pool == null)
        {
            QuantizeAligned(input, q8, nIn, tokens, null);
            GemmPrequant(packed, q8, dst, nIn, nOut, tokens, null, meta);
            return;
        }

        int nb = nIn / Qk.SuperBlock;
        int quantGroups = tokens / 4;
        int gemmGroups = (nOut & ~7) / 8;
        int jobs = Math.Max(quantGroups, Math.Max(1, gemmGroups));
        pool.For(jobs, (int worker, int workers) =>
        {
            int q0 = quantGroups * worker / workers;
            int q1 = quantGroups * (worker + 1) / workers;
            for (int g = q0; g < q1; g++)
                QuantizeQ8Kx4.Quantize4x8(input + g * 4 * nIn, q8 + g * nb, nIn);
            pool.Barrier();
            if (gemmGroups == 0)
                return;
            int g0 = gemmGroups * worker / workers;
            int g1 = gemmGroups * (worker + 1) / workers;
            if (g0 >= g1)
                return;
            GemmQ4K.Gemm8x8(nIn, dst + g0 * 8, nOut, packed + g0 * nb, q8, tokens, (g1 - g0) * 8, meta == null ? null : meta + g0 * nb);
        });
    }

    public static void GemmPrequant(BlockQ4Kx8* packed, BlockQ8Kx4* q8, float* dst, int nIn, int nOut, int tokens, CpuThreadPool? pool, BlockQ4Kx8Meta* meta)
    {
        int packedCols = nOut & ~7;
        int nb = nIn / Qk.SuperBlock;
        void Body(int worker, int workers)
        {
            int groups = packedCols / 8;
            if (groups == 0)
                return;
            int begin = groups * worker / workers;
            int end = groups * (worker + 1) / workers;
            if (begin >= end)
                return;
            GemmQ4K.Gemm8x8(nIn, dst + begin * 8, nOut, packed + begin * nb, q8, tokens, (end - begin) * 8, meta == null ? null : meta + begin * nb);
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(Math.Max(1, packedCols / 8), Body);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PackedGroups(int nOut) => nOut / 8;
}
