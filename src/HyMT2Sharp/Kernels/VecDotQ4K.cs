using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class VecDotQ4K
{
    public static float Dot(BlockQ4K* x, BlockQ8K* y, int n)
    {
        if (Simd.UseAvx512)
            return DotAvx512(x, y, n);
        if (Simd.UseAvx2)
            return DotAvx2(x, y, n);
        if (Simd.UseDp)
            return DotNeon(x, y, n);
        return DotVec(x, y, n);
    }

    /// <summary>
    /// ARM64 NEON row dot: nibble extraction stays in bytes (no u16 widening),
    /// each 16-value group runs one SDOT, and the eight group dots get scaled by
    /// a pairwise-pair tree so the whole loop stays in four i32 lanes.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotNeon(BlockQ4K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        Vector128<byte> m4 = Vector128.Create((byte)0x0F);
        float sumf = 0;
        float minAcc = 0;
        for (int i = 0; i < nb; i++, x++, y++)
        {
            float d = y->D * HalfBits.ToSingle(x->D);
            float dmin = y->D * HalfBits.ToSingle(x->Dmin);

            // Decode 12 packed scale bytes -> [sc0..sc7][mn0..mn7] via the utmp trick.
            uint u0 = Unsafe.ReadUnaligned<uint>(x->Scales);
            uint u1 = Unsafe.ReadUnaligned<uint>(x->Scales + 4);
            uint u2 = Unsafe.ReadUnaligned<uint>(x->Scales + 8);
            uint sc03 = u0 & 0x3f3f3f3f;
            uint sc47 = (u2 & 0x0f0f0f0f) | (((u0 >> 6) & 0x03030303) << 4);
            uint mn03 = u1 & 0x3f3f3f3f;
            uint mn47 = ((u2 >> 4) & 0x0f0f0f0f) | (((u1 >> 6) & 0x03030303) << 4);

            // mins: Σ_j bsums[2s]+bsums[2s+1] pairs · mins[s]  (s = 32-value group)
            Vector128<short> bs0 = Unsafe.ReadUnaligned<Vector128<short>>(y->Bsums);
            Vector128<short> bs1 = Unsafe.ReadUnaligned<Vector128<short>>(y->Bsums + 8);
            Vector128<short> pairSums = AdvSimd.Arm64.AddPairwise(bs0, bs1); // 8 shorts
            Vector128<short> mn = AdvSimd.ZeroExtendWideningLower(Vector128.Create(mn03, mn47, 0u, 0u).AsByte().GetLower()).AsInt16();
            Vector128<int> prodLo = AdvSimd.MultiplyWideningLower(pairSums.GetLower(), mn.GetLower());
            Vector128<int> prodHi = AdvSimd.MultiplyWideningUpper(pairSums, mn);
            minAcc += dmin * AdvSimd.Arm64.AddAcross(AdvSimd.Add(prodLo, prodHi)).ToScalar();

            // scale vectors: [sc0,sc2,sc4,sc6] for lo nibbles, [sc1,sc3,sc5,sc7] for hi
            Vector128<byte> scAll = Vector128.Create(sc03, sc47, 0u, 0u).AsByte();
            Vector128<int> scEven = Vector128.Create((int)scAll[0], (int)scAll[2], (int)scAll[4], (int)scAll[6]);
            Vector128<int> scOdd = Vector128.Create((int)scAll[1], (int)scAll[3], (int)scAll[5], (int)scAll[7]);

            byte* q = x->Qs;
            sbyte* a = y->Qs;
            Vector128<int> accLo0 = Vector128<int>.Zero, accLo1 = Vector128<int>.Zero;
            Vector128<int> accLo2 = Vector128<int>.Zero, accLo3 = Vector128<int>.Zero;
            Vector128<int> accHi0 = Vector128<int>.Zero, accHi1 = Vector128<int>.Zero;
            Vector128<int> accHi2 = Vector128<int>.Zero, accHi3 = Vector128<int>.Zero;
            for (int j = 0; j < 4; j++, q += 32, a += 64)
            {
                Vector128<byte> v0 = Neon.LoadU16(q);
                Vector128<byte> v1 = Neon.LoadU16(q + 16);
                Vector128<sbyte> lo0 = AdvSimd.And(v0, m4).AsSByte();
                Vector128<sbyte> lo1 = AdvSimd.And(v1, m4).AsSByte();
                Vector128<sbyte> hi0 = AdvSimd.And(AdvSimd.ShiftRightLogical(v0.AsUInt16(), 4).AsByte(), m4).AsSByte();
                Vector128<sbyte> hi1 = AdvSimd.And(AdvSimd.ShiftRightLogical(v1.AsUInt16(), 4).AsByte(), m4).AsSByte();
                switch (j)
                {
                    case 0:
                        accLo0 = Neon.Sdot(Neon.Sdot(accLo0, lo0, Neon.Load16(a)), lo1, Neon.Load16(a + 16));
                        accHi0 = Neon.Sdot(Neon.Sdot(accHi0, hi0, Neon.Load16(a + 32)), hi1, Neon.Load16(a + 48));
                        break;
                    case 1:
                        accLo1 = Neon.Sdot(Neon.Sdot(accLo1, lo0, Neon.Load16(a)), lo1, Neon.Load16(a + 16));
                        accHi1 = Neon.Sdot(Neon.Sdot(accHi1, hi0, Neon.Load16(a + 32)), hi1, Neon.Load16(a + 48));
                        break;
                    case 2:
                        accLo2 = Neon.Sdot(Neon.Sdot(accLo2, lo0, Neon.Load16(a)), lo1, Neon.Load16(a + 16));
                        accHi2 = Neon.Sdot(Neon.Sdot(accHi2, hi0, Neon.Load16(a + 32)), hi1, Neon.Load16(a + 48));
                        break;
                    default:
                        accLo3 = Neon.Sdot(Neon.Sdot(accLo3, lo0, Neon.Load16(a)), lo1, Neon.Load16(a + 16));
                        accHi3 = Neon.Sdot(Neon.Sdot(accHi3, hi0, Neon.Load16(a + 32)), hi1, Neon.Load16(a + 48));
                        break;
                }
            }

            // Pairwise trees -> [sub0 dot, sub2 dot, sub4 dot, sub6 dot] and odd subs.
            Vector128<int> dotsLo = Neon.PairAdd(Neon.PairAdd(accLo0, accLo1), Neon.PairAdd(accLo2, accLo3));
            Vector128<int> dotsHi = Neon.PairAdd(Neon.PairAdd(accHi0, accHi1), Neon.PairAdd(accHi2, accHi3));
            Vector128<int> sumi = AdvSimd.Add(
                AdvSimd.Multiply(dotsLo, scEven),
                AdvSimd.Multiply(dotsHi, scOdd));
            sumf += d * Neon.Reduce(sumi);
        }

        return sumf - minAcc;
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

    /// <summary>
    /// 512-bit row dot: same math as <see cref="DotAvx2"/> but a 64-byte weight load
    /// covers two 32-byte groups per step — the lo-nibble zmm holds sub-blocks
    /// (4m, 4m+2) and the hi-nibble zmm (4m+1, 4m+3), so the activation zmm is two
    /// unaligned ymm loads and the per-sub scales come from one vpermw each.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAvx512(BlockQ4K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        uint* utmp = stackalloc uint[4];
        Vector512<byte> m4 = Vector512.Create((byte)0x0F);
        Vector512<float> acc = Vector512<float>.Zero;
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
            Vector256<short> scDup = Vector256.Create(sc128, sc128);
            Vector512<short> scv = Vector512.Create(scDup, scDup);
            Vector512<int> sumi = Vector512<int>.Zero;

            for (int m = 0; m < Qk.SuperBlock / 128; m++)
            {
                Vector512<short> scaleL = Avx512BW.PermuteVar32x16(scv, ScaleIdx512(4 * m + 0, 4 * m + 2));
                Vector512<short> scaleH = Avx512BW.PermuteVar32x16(scv, ScaleIdx512(4 * m + 1, 4 * m + 3));

                Vector512<byte> q4bits = Avx512F.LoadVector512(q4);
                q4 += 64;
                Vector512<byte> q4l = Avx512BW.And(q4bits, m4);
                Vector512<byte> q4h = Avx512BW.And(Avx512BW.ShiftRightLogical(q4bits.AsUInt16(), 4).AsByte(), m4);

                sbyte* a = q8 + m * 128;
                Vector512<sbyte> q8l = Vector512.Create(Avx.LoadVector256(a), Avx.LoadVector256(a + 64));
                Vector512<short> p16l = Avx512BW.MultiplyAddAdjacent(q4l, q8l);
                Vector512<int> p32l = Avx512BW.MultiplyAddAdjacent(p16l, scaleL);

                Vector512<sbyte> q8h = Vector512.Create(Avx.LoadVector256(a + 32), Avx.LoadVector256(a + 96));
                Vector512<short> p16h = Avx512BW.MultiplyAddAdjacent(q4h, q8h);
                Vector512<int> p32h = Avx512BW.MultiplyAddAdjacent(p16h, scaleH);
                sumi = Avx512F.Add(sumi, Avx512F.Add(p32l, p32h));
            }

            acc = Simd.UseFma
                ? Avx512F.FusedMultiplyAdd(Avx512F.ConvertToVector512Single(sumi), Vector512.Create(d), acc)
                : Avx512F.Add(acc, Avx512F.Multiply(Avx512F.ConvertToVector512Single(sumi), Vector512.Create(d)));
        }

        accMin = Sse.Add(accMin, Sse.MoveHighToLow(accMin, accMin));
        accMin = Sse.AddScalar(accMin, Sse.Shuffle(accMin, accMin, 0x55));
        return HorizontalSum(acc.GetLower() + acc.GetUpper()) + accMin.ToScalar();
    }

    /// <summary>vpermw indices that replicate sub-block scales s0 (lo half) and s1 (hi half).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<short> ScaleIdx512(int s0, int s1) =>
        Vector512.Create(Vector256.Create((short)s0), Vector256.Create((short)s1));

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
