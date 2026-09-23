using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class QuantizeQ8Kx4
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Quantize4x8(float* x, BlockQ8Kx4* y, int k)
    {
        if (Simd.UseAvx2)
            Quantize4x8Avx2(x, y, k);
        else
            Quantize4x8FromRows(x, y, k);
    }

    /// <summary>
    /// Prefill ffn_down: SiLU(gate)*up into a 4×256 scratch block, then the
    /// same q8_Kx4 pack. Decode never calls this — it stays GEMV + serial SiLU.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Quantize4x8Silu(float* gate, float* up, BlockQ8Kx4* y, int k)
    {
        float* tmp = stackalloc float[4 * Qk.SuperBlock];
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            for (int row = 0; row < 4; row++)
                SiluMulBlock(gate + row * k + i * Qk.SuperBlock, up + row * k + i * Qk.SuperBlock, tmp + row * Qk.SuperBlock);
            Quantize4x8(tmp, y + i, Qk.SuperBlock);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static void SiluMulBlock(float* gate, float* up, float* dest)
    {
        int j = 0;
        if (Simd.UseAvx2 && Simd.UseFma)
        {
            for (; j <= Qk.SuperBlock - 16; j += 16)
            {
                Vector256<float> g0 = Avx.LoadVector256(gate + j);
                Vector256<float> g1 = Avx.LoadVector256(gate + j + 8);
                Avx.Store(dest + j, Avx.Multiply(FastExp.SiluAvx2(g0), Avx.LoadVector256(up + j)));
                Avx.Store(dest + j + 8, Avx.Multiply(FastExp.SiluAvx2(g1), Avx.LoadVector256(up + j + 8)));
            }
        }

        for (; j + Vector<float>.Count <= Qk.SuperBlock; j += Vector<float>.Count)
            VecF.Store(dest + j, FastExp.SiluVec(VecF.Load(gate + j)) * VecF.Load(up + j));

        for (; j < Qk.SuperBlock; j++)
        {
            float g = gate[j];
            dest[j] = g / (1f + MathF.Exp(-g)) * up[j];
        }
    }

    /// <summary>
    /// Reference: quantize each row then interleave 8-byte chunks.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Quantize4x8FromRows(float* x, BlockQ8Kx4* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        BlockQ8K* tmp = stackalloc BlockQ8K[4 * nb];
        Q8K.QuantizeRow(x, tmp, k);
        Q8K.QuantizeRow(x + k, tmp + nb, k);
        Q8K.QuantizeRow(x + 2 * k, tmp + 2 * nb, k);
        Q8K.QuantizeRow(x + 3 * k, tmp + 3 * nb, k);

        for (int i = 0; i < nb; i++)
        {
            BlockQ8K* r0 = tmp + i;
            BlockQ8K* r1 = tmp + nb + i;
            BlockQ8K* r2 = tmp + 2 * nb + i;
            BlockQ8K* r3 = tmp + 3 * nb + i;
            y[i].D[0] = r0->D;
            y[i].D[1] = r1->D;
            y[i].D[2] = r2->D;
            y[i].D[3] = r3->D;
            for (int chunk = 0; chunk < 32; chunk++)
            {
                int dst = chunk * 32;
                int src = chunk * 8;
                Unsafe.WriteUnaligned(y[i].Qs + dst, Unsafe.ReadUnaligned<long>(r0->Qs + src));
                Unsafe.WriteUnaligned(y[i].Qs + dst + 8, Unsafe.ReadUnaligned<long>(r1->Qs + src));
                Unsafe.WriteUnaligned(y[i].Qs + dst + 16, Unsafe.ReadUnaligned<long>(r2->Qs + src));
                Unsafe.WriteUnaligned(y[i].Qs + dst + 24, Unsafe.ReadUnaligned<long>(r3->Qs + src));
            }

            for (int t = 0; t < Qk.SuperBlock / 16; t++)
            {
                int dest = ((t >> 2) << 4) + (t & 3);
                y[i].Bsums[dest] = r0->Bsums[t];
                y[i].Bsums[dest + 4] = r1->Bsums[t];
                y[i].Bsums[dest + 8] = r2->Bsums[t];
                y[i].Bsums[dest + 12] = r3->Bsums[t];
            }
        }
    }

    /// <summary>
    /// llama.cpp <c>ggml_quantize_mat_q8_K_4x8</c> (x86/repack.cpp ~290):
    /// scale 4 rows, pack 8-byte interleaved qs, then bsums from that layout.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Quantize4x8Avx2(float* x, BlockQ8Kx4* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        Vector256<int> perm = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
        Vector256<int> qmin = Vector256.Create(-127);
        Vector256<int> qmax = Vector256.Create(127);
        for (int i = 0; i < nb; i++)
        {
            float is0 = Q8K.ScaleBlock(x + 0 * k + i * Qk.SuperBlock, out float d0);
            float is1 = Q8K.ScaleBlock(x + 1 * k + i * Qk.SuperBlock, out float d1);
            float is2 = Q8K.ScaleBlock(x + 2 * k + i * Qk.SuperBlock, out float d2);
            float is3 = Q8K.ScaleBlock(x + 3 * k + i * Qk.SuperBlock, out float d3);
            y[i].D[0] = d0;
            y[i].D[1] = d1;
            y[i].D[2] = d2;
            y[i].D[3] = d3;
            Vector256<float> isv0 = Vector256.Create(is0);
            Vector256<float> isv1 = Vector256.Create(is1);
            Vector256<float> isv2 = Vector256.Create(is2);
            Vector256<float> isv3 = Vector256.Create(is3);

            for (int j = 0; j < 32; j++)
            {
                Vector256<int> i0 = Quant8(x + 0 * k + i * Qk.SuperBlock + j * 8, isv0, qmin, qmax);
                Vector256<int> i1 = Quant8(x + 1 * k + i * Qk.SuperBlock + j * 8, isv1, qmin, qmax);
                Vector256<int> i2 = Quant8(x + 2 * k + i * Qk.SuperBlock + j * 8, isv2, qmin, qmax);
                Vector256<int> i3 = Quant8(x + 3 * k + i * Qk.SuperBlock + j * 8, isv3, qmin, qmax);
                Vector256<short> p01 = Avx2.PackSignedSaturate(i0, i1);
                Vector256<short> p23 = Avx2.PackSignedSaturate(i2, i3);
                Vector256<sbyte> packed = Avx2.PackSignedSaturate(p01, p23);
                packed = Avx2.PermuteVar8x32(packed.AsInt32(), perm).AsSByte();
                Avx.Store((byte*)(y[i].Qs + 32 * j), packed.AsByte());
            }

            FillBsums(y[i].Qs, y[i].Bsums);
        }
    }

    public static void Quantize4x8Scalar(float* x, BlockQ8Kx4* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        const int interleave = 8;
        float* srcv = stackalloc float[4 * Qk.SuperBlock];
        float* iscale = stackalloc float[4];

        for (int i = 0; i < nb; i++)
        {
            for (int row = 0; row < 4; row++)
            {
                float* src = srcv + row * Qk.SuperBlock;
                float invScale = Q8K.ScaleBlock(x + row * k + i * Qk.SuperBlock, out float d);
                iscale[row] = invScale;
                y[i].D[row] = d;
                for (int j = 0; j < Qk.SuperBlock; j++)
                    src[j] = x[row * k + i * Qk.SuperBlock + j];
            }

            for (int j = 0; j < Qk.SuperBlock / 4; j++)
                y[i].Bsums[j] = 0;

            for (int j = 0; j < Qk.SuperBlock * 4; j++)
            {
                int srcOffset = (j / (4 * interleave)) * interleave;
                int srcId = (j % (4 * interleave)) / interleave;
                srcOffset += j % interleave;
                int index = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);
                int q = (int)MathF.Round(srcv[srcId * Qk.SuperBlock + srcOffset] * iscale[srcId]);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                y[i].Qs[j] = (sbyte)q;
                y[i].Bsums[index] += (short)q;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Quant8(float* src, Vector256<float> iscale, Vector256<int> qmin, Vector256<int> qmax)
    {
        Vector256<float> rounded = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(src), iscale));
        return Avx2.Min(Avx2.Max(Avx.ConvertToVector256Int32(rounded), qmin), qmax);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FillBsums(sbyte* qs, short* bsums)
    {
        for (int t = 0; t < Qk.SuperBlock / 16; t++)
        {
            int dest = ((t >> 2) << 4) + (t & 3);
            int c0 = (t * 2) * 32;
            int c1 = c0 + 32;
            bsums[dest] = (short)(Sum8(qs + c0) + Sum8(qs + c1));
            bsums[dest + 4] = (short)(Sum8(qs + c0 + 8) + Sum8(qs + c1 + 8));
            bsums[dest + 8] = (short)(Sum8(qs + c0 + 16) + Sum8(qs + c1 + 16));
            bsums[dest + 12] = (short)(Sum8(qs + c0 + 24) + Sum8(qs + c1 + 24));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static short Sum8(sbyte* p)
    {
        if (Sse41.IsSupported)
        {
            Vector128<short> w = Sse41.ConvertToVector128Int16(Vector128.CreateScalarUnsafe(Unsafe.ReadUnaligned<long>(p)).AsSByte());
            w = Ssse3.HorizontalAdd(w, w);
            w = Ssse3.HorizontalAdd(w, w);
            w = Ssse3.HorizontalAdd(w, w);
            return w.ToScalar();
        }

        int s = 0;
        for (int i = 0; i < 8; i++)
            s += p[i];
        return (short)s;
    }
}
