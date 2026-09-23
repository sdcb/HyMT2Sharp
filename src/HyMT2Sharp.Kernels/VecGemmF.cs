using System.Numerics;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Portable prefill GEMM core for the non-AVX2 tier. Weights are dequantized to
/// f32 once per (row tile, block) and applied to every token, so decode cost
/// amortizes over the token count. Row tiles share each activation vector load
/// across <see cref="RT"/> register accumulators.
/// </summary>
internal static unsafe class VecGemmF
{
    private const int RT = 8;

    /// <summary>
    /// output[t][row] = Σ_i dequant(w[row][i]) · input[t][i].
    /// <paramref name="dequantBlock"/> writes <paramref name="blockLen"/> floats
    /// for one block; <paramref name="rowStride"/>/<paramref name="blockBytes"/>
    /// are in bytes. blockLen must be a multiple of Vector&lt;float&gt;.Count.
    /// Blocks shorter than 256 are grouped into 256-wide strips so per-strip
    /// accumulator traffic stays amortized.
    /// </summary>
    public static void Gemm(
        byte* rows,
        int rowStride,
        int blockBytes,
        int blockLen,
        delegate*<byte*, float*, void> dequantBlock,
        float* input,
        float* output,
        int nIn,
        int nOut,
        int tokens,
        CpuThreadPool? pool)
    {
        int stripLen = blockLen >= 256 ? blockLen : 256;
        int blocksPerStrip = stripLen / blockLen;
        int nb = nIn / stripLen;
        int V = Vector<float>.Count;
        void Compute(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            float* wf = stackalloc float[RT * stripLen];
            // Lane-wise accumulators per (row, token); reduced once per output.
            nuint accBytes = (nuint)(RT * tokens * V * sizeof(float));
            NativeBuffer accBuf = new(accBytes);
            Vector<float>* acc = (Vector<float>*)accBuf.Pointer;
            try
            {
                for (int rt = begin; rt < end; rt += RT)
                {
                    int R = Math.Min(RT, end - rt);
                    NativeMemory.Clear(acc, accBytes);
                    for (int i = 0; i < nb; i++)
                    {
                        for (int r = 0; r < R; r++)
                        {
                            byte* wr = rows + (long)(rt + r) * rowStride + (long)i * blocksPerStrip * blockBytes;
                            float* wfRow = wf + r * stripLen;
                            for (int b = 0; b < blocksPerStrip; b++)
                                dequantBlock(wr + (long)b * blockBytes, wfRow + b * blockLen);
                        }

                        if (R == RT)
                        {
                            for (int t = 0; t < tokens; t++)
                            {
                                float* xv = input + (long)t * nIn + (long)i * stripLen;
                                Vector<float> a0 = Vector<float>.Zero;
                                Vector<float> a1 = Vector<float>.Zero;
                                Vector<float> a2 = Vector<float>.Zero;
                                Vector<float> a3 = Vector<float>.Zero;
                                Vector<float> a4 = Vector<float>.Zero;
                                Vector<float> a5 = Vector<float>.Zero;
                                Vector<float> a6 = Vector<float>.Zero;
                                Vector<float> a7 = Vector<float>.Zero;
                                for (int l = 0; l < stripLen; l += V)
                                {
                                    Vector<float> x = ReadV(xv + l);
                                    a0 += ReadV(wf + l) * x;
                                    a1 += ReadV(wf + stripLen + l) * x;
                                    a2 += ReadV(wf + 2 * stripLen + l) * x;
                                    a3 += ReadV(wf + 3 * stripLen + l) * x;
                                    a4 += ReadV(wf + 4 * stripLen + l) * x;
                                    a5 += ReadV(wf + 5 * stripLen + l) * x;
                                    a6 += ReadV(wf + 6 * stripLen + l) * x;
                                    a7 += ReadV(wf + 7 * stripLen + l) * x;
                                }

                                acc[t] += a0;
                                acc[tokens + t] += a1;
                                acc[2 * tokens + t] += a2;
                                acc[3 * tokens + t] += a3;
                                acc[4 * tokens + t] += a4;
                                acc[5 * tokens + t] += a5;
                                acc[6 * tokens + t] += a6;
                                acc[7 * tokens + t] += a7;
                            }
                        }
                        else
                        {
                            for (int t = 0; t < tokens; t++)
                            {
                                float* xv = input + (long)t * nIn + (long)i * stripLen;
                                for (int r = 0; r < R; r++)
                                    acc[r * tokens + t] += DotV(wf + r * stripLen, xv, stripLen);
                            }
                        }
                    }

                    for (int t = 0; t < tokens; t++)
                        for (int r = 0; r < R; r++)
                            output[(long)t * nOut + rt + r] = Vector.Sum(acc[r * tokens + t]);
                }
            }
            finally
            {
                accBuf.Dispose();
            }
        }

        if (pool == null)
            Compute(0, 1);
        else
            pool.For(nOut, Compute);
    }

    /// <summary>Lane-wise w·x accumulator (no horizontal reduction).</summary>
    private static Vector<float> DotV(float* w, float* x, int n)
    {
        int V = Vector<float>.Count;
        Vector<float> acc0 = Vector<float>.Zero;
        Vector<float> acc1 = Vector<float>.Zero;
        int i = 0;
        for (; i + 2 * V <= n; i += 2 * V)
        {
            acc0 += ReadV(w + i) * ReadV(x + i);
            acc1 += ReadV(w + i + V) * ReadV(x + i + V);
        }

        Vector<float> acc = acc0 + acc1;
        for (; i + V <= n; i += V)
            acc += ReadV(w + i) * ReadV(x + i);
        return acc;
    }

    private static Vector<float> ReadV(float* p) =>
        System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Vector<float>>(p);
}
