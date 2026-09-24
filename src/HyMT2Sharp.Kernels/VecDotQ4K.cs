using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class VecDotQ4K
{
    public static float Dot(BlockQ4K* x, BlockQ8K* y, int n)
    {
        if (Simd.UseAvx2)
            return DotAvx2(x, y, n);
        return DotVec(x, y, n);
    }

    /// <summary>Portable single-row entry: builds the <see cref="BlockQ8KAct"/> view, then <see cref="DotAct"/>.</summary>
    public static float DotVec(BlockQ4K* x, BlockQ8K* y, int n)
    {
        if (!VecI8.PairLayoutSupported)
            return DotScalar(x, y, n);
        int nb = n / Qk.SuperBlock;
        BlockQ8KAct* act = stackalloc BlockQ8KAct[nb];
        Q8K.ToVecAct(y, act, nb);
        return DotAct(x, act, nb);
    }

    /// <summary>
    /// Portable Q4_K row dot. Packed bytes load as u16 lanes, so the four nibbles of a
    /// lane come out with and/shift and are already i16 — no byte widening. Pairwise i16
    /// products (≤ 2·15·128) fold to i32 with in-lane shifts (lane order is irrelevant for
    /// a sum), get scaled in-vector, and reduce horizontally once per row.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAct(BlockQ4K* x, BlockQ8KAct* a, int nb)
    {
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;
        int U = Vector<ushort>.Count;
        Vector<ushort> m4 = new(0x000F);
        Vector<float> acc = Vector<float>.Zero;
        float minAcc = 0;
        for (int i = 0; i < nb; i++, x++, a++)
        {
            uint u0 = Unsafe.ReadUnaligned<uint>(x->Scales);
            uint u1 = Unsafe.ReadUnaligned<uint>(x->Scales + 4);
            uint u2 = Unsafe.ReadUnaligned<uint>(x->Scales + 8);
            uint sc03 = u0 & kmask1;
            uint sc47 = (u2 & kmask2) | (((u0 >> 6) & kmask3) << 4);
            uint mn03 = u1 & kmask1;
            uint mn47 = ((u2 >> 4) & kmask2) | (((u1 >> 6) & kmask3) << 4);

            int* bs = a->Bs;
            int mins = (int)(mn03 & 0xFF) * bs[0] + (int)((mn03 >> 8) & 0xFF) * bs[1]
                + (int)((mn03 >> 16) & 0xFF) * bs[2] + (int)(mn03 >> 24) * bs[3]
                + (int)(mn47 & 0xFF) * bs[4] + (int)((mn47 >> 8) & 0xFF) * bs[5]
                + (int)((mn47 >> 16) & 0xFF) * bs[6] + (int)(mn47 >> 24) * bs[7];

            Vector<int> sumi = Vector<int>.Zero;
            byte* q = x->Qs;
            short* A = a->A;
            for (int j = 0; j < 4; j++, q += 32, A += 64)
            {
                uint scw = j < 2 ? sc03 : sc47;
                int sh = (j & 1) * 16;
                Vector<int> sl = new((int)((scw >> sh) & 0xFF));
                Vector<int> shv = new((int)((scw >> (sh + 8)) & 0xFF));
                for (int m = 0; m < 16; m += U)
                {
                    Vector<ushort> v = Unsafe.ReadUnaligned<Vector<ushort>>(q + 2 * m);
                    Vector<short> n0 = Vector.AsVectorInt16(v & m4);
                    Vector<short> n1 = Vector.AsVectorInt16((v >> 4) & m4);
                    Vector<short> n2 = Vector.AsVectorInt16((v >> 8) & m4);
                    Vector<short> n3 = Vector.AsVectorInt16(v >> 12);
                    Vector<short> pl = n0 * VecI8.LoadS16(A + m) + n2 * VecI8.LoadS16(A + 16 + m);
                    Vector<short> ph = n1 * VecI8.LoadS16(A + 32 + m) + n3 * VecI8.LoadS16(A + 48 + m);
                    sumi += VecI8.Fold(pl) * sl + VecI8.Fold(ph) * shv;
                }
            }

            float yd = a->D;
            acc = Vector.MultiplyAddEstimate(Vector.ConvertToSingle(sumi), new Vector<float>(HalfBits.ToSingle(x->D) * yd), acc);
            minAcc += HalfBits.ToSingle(x->Dmin) * yd * mins;
        }

        return Vector.Sum(acc) - minAcc;
    }

    public static float DotScalar(BlockQ4K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        sbyte* aux8 = stackalloc sbyte[Qk.SuperBlock];
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q4 = x[i].Qs;
            sbyte* q8 = y[i].Qs;
            sbyte* a = aux8;
            for (int j = 0; j < Qk.SuperBlock / 64; j++)
            {
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)(q4[l] & 0xF);
                a += 32;
                for (int l = 0; l < 32; l++)
                    a[l] = (sbyte)(q4[l] >> 4);
                a += 32;
                q4 += 32;
            }

            Q4K.UnpackScales(x[i].Scales, utmp);
            byte* scales = (byte*)utmp;
            byte* mins = (byte*)(utmp + 2);

            int sumi = 0;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
                sumi += y[i].Bsums[j] * mins[j / 2];

            a = aux8;
            q8 = y[i].Qs;
            int iscale = 0;
            int acc = 0;
            for (int j = 0; j < Qk.SuperBlock / 32; j++)
            {
                int scale = scales[iscale++];
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAvx2(BlockQ4K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        Vector256<byte> m4 = Vector256.Create((byte)0x0F);
        Vector256<float> acc = Vector256<float>.Zero;
        Vector128<float> accMin = Vector128<float>.Zero;

        for (int i = 0; i < nb; i++)
        {
            float d = y[i].D * HalfBits.ToSingle(x[i].D);
            float dmin = -y[i].D * HalfBits.ToSingle(x[i].Dmin);
            Q4K.UnpackScales(x[i].Scales, utmp);

            byte* q4 = x[i].Qs;
            sbyte* q8 = y[i].Qs;

            Vector128<byte> packed = Vector128.Create(utmp[0], utmp[1], utmp[2], utmp[3]).AsByte();
            Vector256<short> minsAndScales = Avx2.ConvertToVector256Int16(packed);
            Vector256<short> q8sums = Avx.LoadVector256((short*)y[i].Bsums);
            Vector128<short> q8s = Ssse3.HorizontalAdd(q8sums.GetLower(), q8sums.GetUpper());
            Vector128<int> prod = Sse2.MultiplyAddAdjacent(minsAndScales.GetUpper(), q8s);
            accMin = Sse.Add(accMin, Sse.Multiply(Vector128.Create(dmin), Sse2.ConvertToVector128Single(prod)));

            Vector128<short> sc128 = minsAndScales.GetLower();
            Vector256<short> scales = Vector256.Create(sc128, sc128);
            Vector256<int> sumi = Vector256<int>.Zero;

            for (int j = 0; j < Qk.SuperBlock / 64; j++)
            {
                Vector256<byte> scaleL = Avx2.Shuffle(scales.AsByte(), ScaleShuffle(2 * j + 0));
                Vector256<byte> scaleH = Avx2.Shuffle(scales.AsByte(), ScaleShuffle(2 * j + 1));

                Vector256<byte> q4bits = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4l = Avx2.And(q4bits, m4);
                Vector256<byte> q4h = Avx2.And(Avx2.ShiftRightLogical(q4bits.AsUInt16(), 4).AsByte(), m4);

                Vector256<sbyte> q8l = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<short> p16l = Avx2.MultiplyAddAdjacent(q4l, q8l);
                Vector256<int> p32l = Avx2.MultiplyAddAdjacent(p16l, scaleL.AsInt16());

                Vector256<sbyte> q8h = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<short> p16h = Avx2.MultiplyAddAdjacent(q4h, q8h);
                Vector256<int> p32h = Avx2.MultiplyAddAdjacent(p16h, scaleH.AsInt16());
                sumi = Avx2.Add(sumi, Avx2.Add(p32l, p32h));
            }

            acc = Simd.UseFma
                ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(sumi), Vector256.Create(d), acc)
                : Avx.Add(acc, Avx.Multiply(Avx.ConvertToVector256Single(sumi), Vector256.Create(d)));
        }

        accMin = Sse.Add(accMin, Sse.MoveHighToLow(accMin, accMin));
        accMin = Sse.AddScalar(accMin, Sse.Shuffle(accMin, accMin, 0x55));
        return HorizontalSum(acc) + accMin.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> ScaleShuffle(int i)
    {
        ReadOnlySpan<byte> shuffle = [
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
            4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
            6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
            8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
            10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
            12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13,
            14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        ];
        return Vector256.Create(shuffle.Slice(i * 32));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float HorizontalSum(Vector256<float> value)
    {
        Vector128<float> sum = Sse.Add(value.GetLower(), value.GetUpper());
        sum = Sse.Add(sum, Sse.MoveHighToLow(sum, sum));
        sum = Sse.AddScalar(sum, Sse.Shuffle(sum, sum, 0x55));
        return sum.ToScalar();
    }
}
