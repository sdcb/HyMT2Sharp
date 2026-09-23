using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>One weight + destination pair for a fused multi-matrix decode GEMV.</summary>
public unsafe struct Q8GemvTarget
{
    public BlockQ8_0x8* Packed;
    public BlockQ8_0* Rows;
    public float* Dst;
    public int NOut;

    public Q8GemvTarget(BlockQ8_0x8* packed, BlockQ8_0* rows, float* dst, int nOut)
    {
        Packed = packed;
        Rows = rows;
        Dst = dst;
        NOut = nOut;
    }
}

public unsafe struct Q8Panel
{
    public BlockQ8_0x8* Packed;
    public float* Dst;
    public int NOut;

    public readonly bool IsEmpty => NOut == 0;

    public Q8Panel(BlockQ8_0x8* packed, float* dst, int nOut)
    {
        Packed = packed;
        Dst = dst;
        NOut = nOut;
    }
}

/// <summary>
/// Q8_0 × F32 matmul. Decode is GEMV. Prefill quantizes four rows to <see cref="BlockQ8_0x4"/>
/// and runs the 8-column panel. One quantized activation is shared across Q, K and V.
/// </summary>
public static unsafe class MulMatQ8_0
{
    public static void Gemm(BlockQ8_0x8* packed, BlockQ8_0* rows, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        if (tokens == 1)
        {
            Gemv(packed, rows, input, output, nIn, nOut, pool, scratch);
            return;
        }

        if (packed == null)
        {
            Q8_0.Gemm(rows, input, output, nIn, nOut, tokens, pool, scratch);
            return;
        }

        int paddedTokens = (tokens + 3) & ~3;
        int packedCols = nOut & ~7;
        int nb = nIn / Qk.Q8_0Block;
        nuint actBytes = (nuint)((paddedTokens / 4) * nb * Qk.Q8_0x4Size);
        nuint stagingBytes = (nuint)((long)paddedTokens * nIn * sizeof(float));
        nuint outBytes = (nuint)((long)paddedTokens * nOut * sizeof(float));
        NativeBuffer? actOwn = null;
        NativeBuffer? stagingOwn = null;
        NativeBuffer? scratchOutOwn = null;
        BlockQ8_0x4* q8;
        float* paddedIn;
        float* dst;
        if (scratch != null)
        {
            q8 = (BlockQ8_0x4*)scratch.A(actBytes);
            paddedIn = (float*)scratch.B(stagingBytes);
            dst = (float*)scratch.C(outBytes);
        }
        else
        {
            actOwn = new NativeBuffer(actBytes);
            stagingOwn = new NativeBuffer(stagingBytes);
            scratchOutOwn = new NativeBuffer(outBytes);
            q8 = (BlockQ8_0x4*)actOwn.Pointer;
            paddedIn = (float*)stagingOwn.Pointer;
            dst = (float*)scratchOutOwn.Pointer;
        }

        try
        {
            Buffer.MemoryCopy(input, paddedIn, (long)tokens * nIn * sizeof(float), (long)tokens * nIn * sizeof(float));
            if (paddedTokens > tokens)
                NativeMemory.Clear(paddedIn + tokens * nIn, (nuint)((long)(paddedTokens - tokens) * nIn * sizeof(float)));

            QuantizeAndGemm(paddedIn, q8, nIn, paddedTokens, pool, new Q8Panel(packed, dst, nOut));
            if (packedCols < nOut)
            {
                int actNb = nIn / Qk.Q8_0Block;
                nuint tailBytes = (nuint)((long)tokens * actNb * Qk.Q8_0ActSize);
                NativeBuffer? tailOwn = scratch == null ? new NativeBuffer(tailBytes) : null;
                BlockQ8_0Act* tail = (BlockQ8_0Act*)(scratch != null ? scratch.D(tailBytes) : tailOwn!.Pointer);
                try
                {
                    for (int t = 0; t < tokens; t++)
                    {
                        Q8_0.QuantizeActs(input + t * nIn, tail, nIn);
                        for (int row = packedCols; row < nOut; row++)
                            dst[t * nOut + row] = Q8_0.DotScalar(rows + row * actNb, tail, nIn);
                    }
                }
                finally
                {
                    tailOwn?.Dispose();
                }
            }

            Buffer.MemoryCopy(dst, output, (long)tokens * nOut * sizeof(float), (long)tokens * nOut * sizeof(float));
        }
        finally
        {
            actOwn?.Dispose();
            stagingOwn?.Dispose();
            scratchOutOwn?.Dispose();
        }
    }

    /// <summary>Decode GEMV. Uses the 8-column panel when present so weights stream sequentially.</summary>
    public static void Gemv(BlockQ8_0x8* packed, BlockQ8_0* rows, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool, ScratchArena? scratch)
    {
        int nb = nIn / Qk.Q8_0Block;
        nuint bytes = (nuint)nb * (nuint)Qk.Q8_0ActSize;
        NativeBuffer? owned = scratch == null ? new NativeBuffer(bytes) : null;
        BlockQ8_0Act* act = (BlockQ8_0Act*)(scratch != null ? scratch.D(bytes) : owned!.Pointer);
        try
        {
            Q8_0.QuantizeActs(input, act, nIn);
            GemvPrequant(packed, rows, act, output, nIn, nOut, pool);
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void GemvPrequant(BlockQ8_0x8* packed, BlockQ8_0* rows, BlockQ8_0Act* act, float* output, int nIn, int nOut, CpuThreadPool? pool)
    {
        if (packed != null)
            Q8_0.GemvPacked(packed, rows, act, output, nIn, nOut, pool);
        else
            Q8_0.GemvPrequant(rows, act, output, nIn, nOut, pool);
    }

    public static void QuantizeAndGemm(float* input, BlockQ8_0x4* q8, int nIn, int tokens, CpuThreadPool? pool, Q8Panel w0, Q8Panel w1 = default, Q8Panel w2 = default)
    {
        int nb = nIn / Qk.Q8_0Block;
        int quantGroups = tokens / 4;
        if (pool == null)
        {
            for (int g = 0; g < quantGroups; g++)
                Q8_0.Quantize4(input + g * 4 * nIn, q8 + g * nb, nIn);
            GemmRange(w0, q8, nIn, tokens, 0, 1, nb);
            GemmRange(w1, q8, nIn, tokens, 0, 1, nb);
            GemmRange(w2, q8, nIn, tokens, 0, 1, nb);
            return;
        }

        int jobs = Math.Max(quantGroups, Math.Max(1, Math.Max(w0.NOut / 8, Math.Max(w1.NOut / 8, w2.NOut / 8))));
        pool.For(jobs, (int worker, int workers) =>
        {
            int g0 = quantGroups * worker / workers;
            int g1 = quantGroups * (worker + 1) / workers;
            for (int g = g0; g < g1; g++)
                Q8_0.Quantize4(input + g * 4 * nIn, q8 + g * nb, nIn);
            pool.Barrier();
            GemmRange(w0, q8, nIn, tokens, worker, workers, nb);
            GemmRange(w1, q8, nIn, tokens, worker, workers, nb);
            GemmRange(w2, q8, nIn, tokens, worker, workers, nb);
        });
    }

    public static void SiluQuantizeAndGemm(float* gate, float* up, BlockQ8_0x4* q8, int nIn, int tokens, CpuThreadPool? pool, Q8Panel w)
    {
        int nb = nIn / Qk.Q8_0Block;
        int quantGroups = tokens / 4;
        if (pool == null)
        {
            for (int g = 0; g < quantGroups; g++)
                Q8_0.Quantize4Silu(gate + g * 4 * nIn, up + g * 4 * nIn, q8 + g * nb, nIn);
            GemmRange(w, q8, nIn, tokens, 0, 1, nb);
            return;
        }

        int jobs = Math.Max(quantGroups, Math.Max(1, w.NOut / 8));
        pool.For(jobs, (int worker, int workers) =>
        {
            int g0 = quantGroups * worker / workers;
            int g1 = quantGroups * (worker + 1) / workers;
            for (int g = g0; g < g1; g++)
                Q8_0.Quantize4Silu(gate + g * 4 * nIn, up + g * 4 * nIn, q8 + g * nb, nIn);
            pool.Barrier();
            GemmRange(w, q8, nIn, tokens, worker, workers, nb);
        });
    }

    private static void GemmRange(Q8Panel w, BlockQ8_0x4* q8, int nIn, int tokens, int worker, int workers, int nb)
    {
        int groups = w.NOut / 8;
        if (groups == 0 || w.Packed == null)
            return;
        int begin = groups * worker / workers;
        int end = groups * (worker + 1) / workers;
        if (begin >= end)
            return;
        GemmQ8_0.Gemm8x8(nIn, w.Dst + begin * 8, w.NOut, w.Packed + begin * nb, q8, tokens, (end - begin) * 8);
    }
}
