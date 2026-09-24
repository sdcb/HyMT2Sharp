using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Portable prefill GEMM core for the non-AVX2 tier. Outer-product (BLIS-style)
/// micro-kernel: weights are dequantized per strip into an [k][NR] panel whose
/// vectors span NR consecutive output rows, activations are packed once per call
/// into [token tile][k][MR] so each k step broadcasts MR consecutive floats.
/// The MR×NR accumulators stay in registers for a whole strip, so there is no
/// horizontal reduction and the result maps straight onto the [token][row] output.
/// </summary>
internal static unsafe class VecGemmF
{
    private const int MR = 6;

    /// <summary>
    /// Panels sharing one pass over the packed activations: each x tile is reused from L1
    /// by PG kernels. 2 measured best on Zen 3; 4 pushes the w panels out of L1.
    /// </summary>
    private const int PG = 2;

    private static int NR => 2 * Vector<float>.Count;

    /// <summary>
    /// output[t][row] = Σ_i dequant(w[row][i]) · input[t][i].
    /// <paramref name="dequantBlock"/> writes <paramref name="blockLen"/> floats
    /// for one block; <paramref name="rowStride"/>/<paramref name="blockBytes"/>
    /// are in bytes. Blocks shorter than 256 are grouped into 256-wide strips.
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
        int nr = NR;
        int tiles = (tokens + MR - 1) / MR;
        int panels = (nOut + nr - 1) / nr;
        nuint packedBytes = (nuint)((long)tiles * MR * nIn * sizeof(float));
        float* xp = (float*)Rent(ref t_pack, packedBytes + (nuint)(nIn * sizeof(float)));
        float* zero = xp + (long)tiles * MR * nIn;
        if (tiles * MR != tokens)
            NativeMemory.Clear(zero, (nuint)(nIn * sizeof(float)));

        void Compute(int worker, int workers)
        {
            int t0 = tiles * worker / workers;
            int t1 = tiles * (worker + 1) / workers;
            for (int tile = t0; tile < t1; tile++)
                PackTile(input, xp + (long)tile * MR * nIn, nIn, tokens, tile * MR, zero);
            if (workers > 1)
                pool!.Barrier();

            int p0 = panels * worker / workers;
            int p1 = panels * (worker + 1) / workers;
            if (p0 >= p1)
                return;

            nuint panelBytes = (nuint)(stripLen * nr * sizeof(float));
            nuint cBytes = (nuint)((long)tiles * MR * nr * sizeof(float));
            byte* scratch = (byte*)Rent(ref t_work, (PG + 1) * panelBytes + PG * cBytes);
            float* wrow = (float*)scratch;
            float* wpan = (float*)(scratch + panelBytes);
            float* c = (float*)(scratch + (PG + 1) * panelBytes);
            int cStride = tiles * MR * nr;
            for (int g0 = p0; g0 < p1; g0 += PG)
            {
                int G = Math.Min(PG, p1 - g0);
                NativeMemory.Clear(c, (nuint)G * cBytes);
                for (int i = 0; i < nb; i++)
                {
                    for (int g = 0; g < G; g++)
                    {
                        int row0 = (g0 + g) * nr;
                        int R = Math.Min(nr, nOut - row0);
                        for (int r = 0; r < R; r++)
                        {
                            byte* wr = rows + (long)(row0 + r) * rowStride + (long)i * blocksPerStrip * blockBytes;
                            float* dst = wrow + r * stripLen;
                            for (int b = 0; b < blocksPerStrip; b++)
                                dequantBlock(wr + (long)b * blockBytes, dst + b * blockLen);
                        }

                        if (R < nr)
                            NativeMemory.Clear(wrow + R * stripLen, (nuint)((nr - R) * stripLen * sizeof(float)));
                        Transpose(wrow, wpan + g * stripLen * nr, stripLen, nr);
                    }

                    float* xs = xp + (long)i * stripLen * MR;
                    for (int tile = 0; tile < tiles; tile++)
                    {
                        float* xt = xs + (long)tile * MR * nIn;
                        float* ct = c + tile * MR * nr;
                        for (int g = 0; g < G; g++)
                            Kernel(stripLen, wpan + g * stripLen * nr, xt, ct + g * cStride);
                    }
                }

                for (int g = 0; g < G; g++)
                {
                    int row0 = (g0 + g) * nr;
                    int R = Math.Min(nr, nOut - row0);
                    float* cg = c + g * cStride;
                    for (int t = 0; t < tokens; t++)
                        Buffer.MemoryCopy(cg + t * nr, output + (long)t * nOut + row0, R * sizeof(float), R * sizeof(float));
                }
            }
        }

        if (pool == null)
            Compute(0, 1);
        else
            pool.For(Math.Max(panels, tiles), Compute);
    }

    // Grow-only per-thread scratch: the caller's packed activations and each pool
    // worker's panel/accumulator tiles. Pool threads are long-lived, so a prefill
    // pays the allocation (and first-touch page faults) once instead of per matmul.
    [ThreadStatic] private static NativeBuffer? t_pack;
    [ThreadStatic] private static NativeBuffer? t_work;

    private static void* Rent(ref NativeBuffer? buf, nuint bytes)
    {
        if (buf == null || buf.Bytes < bytes)
        {
            buf?.Dispose();
            buf = new NativeBuffer(bytes);
        }

        return buf.Pointer;
    }

    /// <summary>[nr][len] row-major → [len][nr]; contiguous writes, 4 rows per pass.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Transpose(float* src, float* dst, int len, int nr)
    {
        for (int r = 0; r < nr; r += 4)
        {
            float* s0 = src + r * len, s1 = s0 + len, s2 = s1 + len, s3 = s2 + len;
            float* d = dst + r;
            for (int k = 0; k < len; k++, d += nr)
            {
                d[0] = s0[k];
                d[1] = s1[k];
                d[2] = s2[k];
                d[3] = s3[k];
            }
        }
    }

    /// <summary>Tile layout [k][MR]; missing tokens (tail tile) are zero.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PackTile(float* input, float* dst, int nIn, int tokens, int tBase, float* zero)
    {
        float* s0 = Src(0), s1 = Src(1), s2 = Src(2), s3 = Src(3), s4 = Src(4), s5 = Src(5);
        for (int k = 0; k < nIn; k++, dst += MR)
        {
            dst[0] = s0[k];
            dst[1] = s1[k];
            dst[2] = s2[k];
            dst[3] = s3[k];
            dst[4] = s4[k];
            dst[5] = s5[k];
        }

        float* Src(int m) => tBase + m < tokens ? input + (long)(tBase + m) * nIn : zero;
    }

    /// <summary>c[MR][NR] += x[k][MR] ⊗ w[k][NR] over kc steps.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel(int kc, float* w, float* x, float* c)
    {
        int V = Vector<float>.Count;
        int nr = 2 * V;
        Vector<float> c00 = Load(c), c01 = Load(c + V);
        Vector<float> c10 = Load(c + nr), c11 = Load(c + nr + V);
        Vector<float> c20 = Load(c + 2 * nr), c21 = Load(c + 2 * nr + V);
        Vector<float> c30 = Load(c + 3 * nr), c31 = Load(c + 3 * nr + V);
        Vector<float> c40 = Load(c + 4 * nr), c41 = Load(c + 4 * nr + V);
        Vector<float> c50 = Load(c + 5 * nr), c51 = Load(c + 5 * nr + V);
        for (int k = 0; k < kc; k++)
        {
            Vector<float> w0 = Load(w);
            Vector<float> w1 = Load(w + V);
            Vector<float> b = new(x[0]);
            c00 = Vector.MultiplyAddEstimate(w0, b, c00);
            c01 = Vector.MultiplyAddEstimate(w1, b, c01);
            b = new(x[1]);
            c10 = Vector.MultiplyAddEstimate(w0, b, c10);
            c11 = Vector.MultiplyAddEstimate(w1, b, c11);
            b = new(x[2]);
            c20 = Vector.MultiplyAddEstimate(w0, b, c20);
            c21 = Vector.MultiplyAddEstimate(w1, b, c21);
            b = new(x[3]);
            c30 = Vector.MultiplyAddEstimate(w0, b, c30);
            c31 = Vector.MultiplyAddEstimate(w1, b, c31);
            b = new(x[4]);
            c40 = Vector.MultiplyAddEstimate(w0, b, c40);
            c41 = Vector.MultiplyAddEstimate(w1, b, c41);
            b = new(x[5]);
            c50 = Vector.MultiplyAddEstimate(w0, b, c50);
            c51 = Vector.MultiplyAddEstimate(w1, b, c51);
            w += nr;
            x += MR;
        }

        Store(c, c00); Store(c + V, c01);
        Store(c + nr, c10); Store(c + nr + V, c11);
        Store(c + 2 * nr, c20); Store(c + 2 * nr + V, c21);
        Store(c + 3 * nr, c30); Store(c + 3 * nr + V, c31);
        Store(c + 4 * nr, c40); Store(c + 4 * nr + V, c41);
        Store(c + 5 * nr, c50); Store(c + 5 * nr + V, c51);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> Load(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(float* p, Vector<float> v) => Unsafe.WriteUnaligned(p, v);
}
