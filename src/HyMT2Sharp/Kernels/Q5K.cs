using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class Q5K
{
    public static void DequantizeRow(BlockQ5K* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            byte* ql = x[i].Qs;
            byte* qh = x[i].Qh;
            float d = HalfBits.ToSingle(x[i].D);
            float min = HalfBits.ToSingle(x[i].Dmin);
            int iscale = 0;
            byte u1 = 1;
            byte u2 = 2;
            for (int j = 0; j < Qk.SuperBlock; j += 64)
            {
                Q4K.GetScaleMin(iscale, x[i].Scales, out byte sc0, out byte min0);
                Q4K.GetScaleMin(iscale + 1, x[i].Scales, out byte sc1, out byte min1);
                float d1 = d * sc0;
                float m1 = min * min0;
                float d2 = d * sc1;
                float m2 = min * min1;
                for (int l = 0; l < 32; l++)
                    *y++ = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - m1;
                for (int l = 0; l < 32; l++)
                    *y++ = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - m2;
                ql += 32;
                iscale += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }

    public static float Dot(BlockQ5K* x, BlockQ8K* y, int n)
    {
        return DotVec(x, y, n);
    }

    /// <summary>Portable widening fallback; vectorized 5-bit decode then per-32 scaled dots.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotVec(BlockQ5K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        sbyte* aux = stackalloc sbyte[Qk.SuperBlock];
        Vector<byte> m15 = new Vector<byte>(15);
        Vector<byte> one = Vector<byte>.One;
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q5 = x[i].Qs;
            byte* hm = x[i].Qh;
            uint u1 = 1;
            for (int j = 0; j < Qk.SuperBlock / 64; j++)
            {
                int l = 0;
                for (; l + Vector<byte>.Count <= 32; l += Vector<byte>.Count)
                {
                    Vector<byte> lo = VecI8.LoadU8(q5 + 32 * j + l);
                    Vector<byte> hb = VecI8.LoadU8(hm + l);
                    Vector<byte> h1 = Vector.Min(hb & new Vector<byte>((byte)u1), one);
                    Vector<byte> h2 = Vector.Min(hb & new Vector<byte>((byte)(u1 * 2)), one);
                    Unsafe.WriteUnaligned(aux + 64 * j + l, Vector.AsVectorSByte((lo & m15) | (h1 << 4)));
                    Unsafe.WriteUnaligned(aux + 64 * j + 32 + l, Vector.AsVectorSByte((lo >> 4) | (h2 << 4)));
                }

                for (; l < 32; l++)
                {
                    aux[64 * j + l] = (sbyte)((q5[32 * j + l] & 0xF) + ((hm[l] & u1) != 0 ? 16 : 0));
                    aux[64 * j + 32 + l] = (sbyte)((q5[32 * j + l] >> 4) + ((hm[l] & (u1 * 2)) != 0 ? 16 : 0));
                }

                u1 *= 4;
            }

            Q4K.UnpackScales(x[i].Scales, utmp);
            byte* scales = (byte*)utmp;
            byte* mins = (byte*)(utmp + 2);

            int sumi = 0;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                sumi += y[i].Bsums[j] * mins[j / 2];

            sbyte* a = aux;
            sbyte* q8 = y[i].Qs;
            int acc = 0;
            for (int j = 0; j < Qk.SuperBlock / 32; j++)
            {
                acc += scales[j] * VecI8.DotI8(a, q8, 32);
                a += 32;
                q8 += 32;
            }

            float d = HalfBits.ToSingle(x[i].D) * y[i].D;
            float dmin = HalfBits.ToSingle(x[i].Dmin) * y[i].D;
            sumf += d * acc - dmin * sumi;
        }

        return sumf;
    }

    public static float DotScalar(BlockQ5K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        sbyte* aux = stackalloc sbyte[Qk.SuperBlock];
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q4 = x[i].Qs;
            byte* hm = x[i].Qh;
            sbyte* a = aux;
            byte m = 1;
            for (int j = 0; j < Qk.SuperBlock / 64; j++)
            {
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)((q4[l] & 0xF) + ((hm[l] & m) != 0 ? 16 : 0));
                a += 32;
                m <<= 1;
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)((q4[l] >> 4) + ((hm[l] & m) != 0 ? 16 : 0));
                a += 32;
                m <<= 1;
                q4 += 32;
            }

            Q4K.UnpackScales(x[i].Scales, utmp);
            byte* scales = (byte*)utmp;
            byte* mins = (byte*)(utmp + 2);
            int sumi = 0;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                sumi += y[i].Bsums[j] * mins[j / 2];

            int acc = 0;
            a = aux;
            sbyte* q8 = y[i].Qs;
            for (int j = 0; j < Qk.SuperBlock / 32; j++)
            {
                int scale = scales[j];
                for (int l = 0; l < 32; l++)
                    acc += scale * q8[l] * a[l];
                q8 += 32;
                a += 32;
            }

            float d = HalfBits.ToSingle(x[i].D) * y[i].D;
            float dmin = HalfBits.ToSingle(x[i].Dmin) * y[i].D;
            sumf += d * acc - dmin * sumi;
        }

        return sumf;
    }

    public static void Gemv(BlockQ5K* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null)
    {
        int nb = nIn / Qk.SuperBlock;
        using NativeBuffer q8 = new((nuint)Q8K.RowBytes(nIn));
        Q8K.QuantizeRow(input, (BlockQ8K*)q8.Pointer, nIn);
        BlockQ8K* y = (BlockQ8K*)q8.Pointer;
        void Body(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            for (int row = begin; row < end; row++)
                output[row] = Dot(weights + row * nb, y, nIn);
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(nOut, Body);
    }

    public static void Gemm(BlockQ5K* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null)
    {
        if (tokens == 1)
        {
            Gemv(weights, input, output, nIn, nOut, pool);
            return;
        }

        int nb = nIn / Qk.SuperBlock;
        VecGemmF.Gemm((byte*)weights, nb * sizeof(BlockQ5K), sizeof(BlockQ5K), Qk.SuperBlock,
            &DequantBlock, input, output, nIn, nOut, tokens, pool);
    }

    private static void DequantBlock(byte* p, float* dst) =>
        DequantizeRow((BlockQ5K*)p, dst, Qk.SuperBlock);
}
