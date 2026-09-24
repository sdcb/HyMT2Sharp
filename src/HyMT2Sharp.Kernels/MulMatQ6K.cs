using System.Threading;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Q6_K × F32 matmul. Decode is AVX2 GEMV. Prefill uses q6_Kx8 × q8_Kx4 panel GEMM.
/// </summary>
public static unsafe class MulMatQ6K
{
    public static void Gemm(BlockQ6Kx8* packed, BlockQ6K* rows, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        if (tokens == 1)
        {
            Q6K.Gemm(rows, input, output, nIn, nOut, tokens, pool, scratch);
            return;
        }
        if (packed == null || !Simd.UsePanels)
        {
            int nbv = nIn / Qk.SuperBlock;
            VecGemmF.Gemm((byte*)rows, nbv * sizeof(BlockQ6K), sizeof(BlockQ6K), Qk.SuperBlock,
                &DequantBlock, input, output, nIn, nOut, tokens, pool);
            return;
        }

        int paddedTokens = (tokens + 3) & ~3;
        int packedCols = nOut & ~7;
        int nb = nIn / Qk.SuperBlock;
        nuint actBytes = (nuint)((paddedTokens / 4) * nb * Qk.Q8Kx4Size);
        nuint stagingBytes = (nuint)((long)paddedTokens * nIn * sizeof(float));
        nuint outBytes = (nuint)((long)paddedTokens * nOut * sizeof(float));
        nuint q8Bytes = (nuint)Q8K.RowBytes(nIn);
        NativeBuffer? actOwn = null;
        NativeBuffer? stagingOwn = null;
        NativeBuffer? scratchOutOwn = null;
        NativeBuffer? q8rowOwn = null;
        BlockQ8Kx4* q8;
        float* paddedIn;
        float* dst;
        BlockQ8K* q8row;
        if (scratch != null)
        {
            q8 = (BlockQ8Kx4*)scratch.A(actBytes);
            paddedIn = (float*)scratch.B(stagingBytes);
            dst = (float*)scratch.C(outBytes);
            q8row = (BlockQ8K*)scratch.D(q8Bytes);
        }
        else
        {
            actOwn = new NativeBuffer(actBytes);
            stagingOwn = new NativeBuffer(stagingBytes);
            scratchOutOwn = new NativeBuffer(outBytes);
            q8rowOwn = new NativeBuffer(q8Bytes);
            q8 = (BlockQ8Kx4*)actOwn.Pointer;
            paddedIn = (float*)stagingOwn.Pointer;
            dst = (float*)scratchOutOwn.Pointer;
            q8row = (BlockQ8K*)q8rowOwn.Pointer;
        }

        try
        {
            Buffer.MemoryCopy(input, paddedIn, (long)tokens * nIn * sizeof(float), (long)tokens * nIn * sizeof(float));
            if (paddedTokens > tokens)
                NativeMemory.Clear(paddedIn + tokens * nIn, (nuint)((long)(paddedTokens - tokens) * nIn * sizeof(float)));

            int quantGroups = paddedTokens / 4;
            int quantCursor = 0;
            void QuantBody(int worker, int workers)
            {
                while (true)
                {
                    int g = Interlocked.Increment(ref quantCursor) - 1;
                    if (g >= quantGroups)
                        break;
                    QuantizeQ8Kx4.Quantize4x8(paddedIn + g * 4 * nIn, q8 + g * nb, nIn);
                }
            }

            if (pool == null)
                QuantBody(0, 1);
            else
                pool.For(quantGroups, QuantBody);

            int gemmCursor = 0;
            void Body(int worker, int workers)
            {
                int groups = packedCols / 8;
                if (groups == 0)
                    return;
                while (true)
                {
                    int g = Interlocked.Increment(ref gemmCursor) - 1;
                    if (g >= groups)
                        break;
                    GemmQ6K.Gemm8x8(nIn, dst + g * 8, nOut, packed + g * nb, q8, paddedTokens, 8);
                }
            }

            if (pool == null)
                Body(0, 1);
            else
                pool.For(Math.Max(1, packedCols / 8), Body);

            for (int t = 0; t < tokens; t++)
            {
                Q8K.QuantizeRow(input + t * nIn, q8row, nIn);
                for (int row = packedCols; row < nOut; row++)
                    dst[t * nOut + row] = Q6K.Dot(rows + row * nb, q8row, nIn);
            }

            Buffer.MemoryCopy(dst, output, (long)tokens * nOut * sizeof(float), (long)tokens * nOut * sizeof(float));
        }
        finally
        {
            actOwn?.Dispose();
            stagingOwn?.Dispose();
            scratchOutOwn?.Dispose();
            q8rowOwn?.Dispose();
        }
    }

    private static void DequantBlock(byte* p, float* dst) =>
        Q6K.DequantizeBlockVec((BlockQ6K*)p, dst);
}
