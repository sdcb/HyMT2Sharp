using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class Q6K
{
    /// <summary>One superblock → 256 floats: vector decode to int8, then widen × d·scale per 16.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DequantizeBlockVec(BlockQ6K* x, float* y)
    {
        if (Vector<sbyte>.Count > 32)
        {
            DequantizeRow(x, y, Qk.SuperBlock);
            return;
        }

        sbyte* a = stackalloc sbyte[Qk.SuperBlock];
        DecodeBlockVec(x, a);
        float d = HalfBits.ToSingle(x->D);
        int n = Vector<int>.Count;
        for (int i = 0; i < Qk.SuperBlock; i += Vector<sbyte>.Count)
        {
            Vector.Widen(VecI8.LoadS8(a + i), out Vector<short> lo, out Vector<short> hi);
            Vector.Widen(lo, out Vector<int> v0, out Vector<int> v1);
            Vector.Widen(hi, out Vector<int> v2, out Vector<int> v3);
            VecF.Store(y + i, Vector.ConvertToSingle(v0) * new Vector<float>(d * x->Scales[i / 16]));
            VecF.Store(y + i + n, Vector.ConvertToSingle(v1) * new Vector<float>(d * x->Scales[(i + n) / 16]));
            VecF.Store(y + i + 2 * n, Vector.ConvertToSingle(v2) * new Vector<float>(d * x->Scales[(i + 2 * n) / 16]));
            VecF.Store(y + i + 3 * n, Vector.ConvertToSingle(v3) * new Vector<float>(d * x->Scales[(i + 3 * n) / 16]));
        }
    }

    public static void DequantizeRow(BlockQ6K* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            float d = HalfBits.ToSingle(x[i].D);
            byte* ql = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* sc = x[i].Scales;
            for (int n = 0; n < Qk.SuperBlock; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int iscale = l / 16;
                    int q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
                    int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                    int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                    int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                    y[l] = d * sc[iscale] * (q1 - 32);
                    y[l + 32] = d * sc[iscale + 2] * (q2 - 32);
                    y[l + 64] = d * sc[iscale + 4] * (q3 - 32);
                    y[l + 96] = d * sc[iscale + 6] * (q4 - 32);
                }

                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    public static float Dot(BlockQ6K* x, BlockQ8K* y, int n)
        => Simd.UseAvx2 ? DotAvx2(x, y, n) : Simd.UseDp ? DotNeon(x, y, n) : DotVec(x, y, n);

    /// <summary>
    /// ARM64 NEON row dot: raw (unsigned) 6-bit weights go straight into SDOT,
    /// so each 16-value sub-group costs one instruction; per-sub scales are
    /// applied after a pairwise-pair tree, and the -32 bias lands once per
    /// superblock via the activation bsums correction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotNeon(BlockQ6K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        Vector128<byte> m15 = Vector128.Create((byte)15);
        Vector128<byte> m3 = Vector128.Create((byte)3);
        Vector128<byte> m48 = Vector128.Create((byte)0x30);
        float sumf = 0;
        for (int i = 0; i < nb; i++, x++, y++)
        {
            float d = y->D * HalfBits.ToSingle(x->D);
            byte* ql = x->Ql;
            byte* qh = x->Qh;
            sbyte* q8 = y->Qs;
            sbyte* sc = x->Scales;

            Vector128<int> sumi = Vector128<int>.Zero;
            for (int j = 0; j < 2; j++, ql += 64, qh += 32, q8 += 128, sc += 8)
            {
                Vector128<byte> ql0 = Neon.LoadU16(ql);
                Vector128<byte> ql1 = Neon.LoadU16(ql + 32);
                Vector128<byte> h0 = Neon.LoadU16(qh);
                Vector128<byte> h1 = Neon.LoadU16(qh + 16);

                // group k covers vals 128j+32k, i.e. scales 8j+2k and 8j+2k+1:
                //   k0 = ql0 lo | (qh&3)<<4      k1 = ql1 lo | (qh&12)<<2
                //   k2 = ql0 hi | (qh&48)        k3 = ql1 hi | (qh&C0)>>2
                Vector128<byte> hi0 = AdvSimd.ShiftLeftLogical(AdvSimd.And(h0, m3), 4);
                Vector128<byte> hi1 = AdvSimd.ShiftLeftLogical(AdvSimd.And(h1, m3), 4);
                Vector128<byte> hi2 = AdvSimd.And(AdvSimd.ShiftLeftLogical(h0, 2), m48);
                Vector128<byte> hi3 = AdvSimd.And(AdvSimd.ShiftLeftLogical(h1, 2), m48);
                Vector128<byte> hi4 = AdvSimd.And(h0, m48);
                Vector128<byte> hi5 = AdvSimd.And(h1, m48);
                Vector128<byte> hi6 = AdvSimd.ShiftRightLogical(AdvSimd.And(h0, Vector128.Create((byte)0xC0)), 2);
                Vector128<byte> hi7 = AdvSimd.ShiftRightLogical(AdvSimd.And(h1, Vector128.Create((byte)0xC0)), 2);

                Vector128<sbyte> g0a = AdvSimd.Or(AdvSimd.And(ql0, m15), hi0).AsSByte();
                Vector128<sbyte> g0b = AdvSimd.Or(AdvSimd.And(Neon.LoadU16(ql + 16), m15), hi1).AsSByte();
                Vector128<sbyte> g1a = AdvSimd.Or(AdvSimd.And(Neon.LoadU16(ql + 32), m15), hi2).AsSByte();
                Vector128<sbyte> g1b = AdvSimd.Or(AdvSimd.And(Neon.LoadU16(ql + 48), m15), hi3).AsSByte();
                Vector128<byte> ql0h = AdvSimd.ShiftRightLogical(ql0, 4);
                Vector128<byte> ql1h = AdvSimd.ShiftRightLogical(Neon.LoadU16(ql + 16), 4);
                Vector128<sbyte> g2a = AdvSimd.Or(AdvSimd.And(ql0h, m15), hi4).AsSByte();
                Vector128<sbyte> g2b = AdvSimd.Or(AdvSimd.And(ql1h, m15), hi5).AsSByte();
                Vector128<sbyte> g3a = AdvSimd.Or(AdvSimd.And(AdvSimd.ShiftRightLogical(Neon.LoadU16(ql + 32), 4), m15), hi6).AsSByte();
                Vector128<sbyte> g3b = AdvSimd.Or(AdvSimd.And(AdvSimd.ShiftRightLogical(Neon.LoadU16(ql + 48), 4), m15), hi7).AsSByte();

                Vector128<int> d0 = Neon.Sdot(Vector128<int>.Zero, g0a, Neon.Load16(q8));
                Vector128<int> d1 = Neon.Sdot(Vector128<int>.Zero, g0b, Neon.Load16(q8 + 16));
                Vector128<int> d2 = Neon.Sdot(Vector128<int>.Zero, g1a, Neon.Load16(q8 + 32));
                Vector128<int> d3 = Neon.Sdot(Vector128<int>.Zero, g1b, Neon.Load16(q8 + 48));
                Vector128<int> d4 = Neon.Sdot(Vector128<int>.Zero, g2a, Neon.Load16(q8 + 64));
                Vector128<int> d5 = Neon.Sdot(Vector128<int>.Zero, g2b, Neon.Load16(q8 + 80));
                Vector128<int> d6 = Neon.Sdot(Vector128<int>.Zero, g3a, Neon.Load16(q8 + 96));
                Vector128<int> d7 = Neon.Sdot(Vector128<int>.Zero, g3b, Neon.Load16(q8 + 112));

                // PairAdd(a,b)=[a01,a23,b01,b23]: two rounds give [s0..s3] and [s4..s7].
                Vector128<sbyte> scb = Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(sc)).AsSByte();
                Vector128<short> scw = AdvSimd.SignExtendWideningLower(scb.GetLower());
                Vector128<int> scL = AdvSimd.SignExtendWideningLower(scw.GetLower());
                Vector128<int> scH = AdvSimd.SignExtendWideningLower(scw.GetUpper());
                sumi = AdvSimd.Add(sumi,
                    AdvSimd.Multiply(Neon.PairAdd(Neon.PairAdd(d0, d1), Neon.PairAdd(d2, d3)), scL));
                sumi = AdvSimd.Add(sumi,
                    AdvSimd.Multiply(Neon.PairAdd(Neon.PairAdd(d4, d5), Neon.PairAdd(d6, d7)), scH));
            }

            // -32 bias correction: subtract 32*Σ sc_i·bsum_i.
            Vector128<sbyte> scb2 = Neon.Load16(x->Scales);
            Vector128<short> scv = AdvSimd.SignExtendWideningLower(scb2.GetLower());
            Vector128<short> scvH = AdvSimd.SignExtendWideningUpper(scb2);
            Vector128<short> bs0 = Unsafe.ReadUnaligned<Vector128<short>>(y->Bsums);
            Vector128<short> bs1 = Unsafe.ReadUnaligned<Vector128<short>>(y->Bsums + 8);
            Vector128<int> corr = AdvSimd.Add(
                AdvSimd.Add(AdvSimd.MultiplyWideningLower(scv.GetLower(), bs0.GetLower()),
                            AdvSimd.MultiplyWideningUpper(scv, bs0)),
                AdvSimd.Add(AdvSimd.MultiplyWideningLower(scvH.GetLower(), bs1.GetLower()),
                            AdvSimd.MultiplyWideningUpper(scvH, bs1)));
            int sub = Neon.Reduce(corr) << 5;

            sumf += d * (Neon.Reduce(sumi) - sub);
        }

        return sumf;
    }

    /// <summary>Portable single-row entry: builds the <see cref="BlockQ8KAct"/> view, then <see cref="DotAct"/>.</summary>
    public static float DotVec(BlockQ6K* x, BlockQ8K* y, int n)
    {
        if (!VecI8.PairLayoutSupported)
            return DotScalar(x, y, n);
        int nb = n / Qk.SuperBlock;
        BlockQ8KAct* act = stackalloc BlockQ8KAct[nb];
        Q8K.ToVecAct(y, act, nb);
        return DotAct(x, act, nb);
    }

    /// <summary>
    /// Portable Q6_K row dot on the even/odd <see cref="BlockQ8KAct"/> layout. Ql/Qh load as
    /// u16 lanes; each lane yields the even and odd 6-bit values of all four 32-value groups
    /// with and/shift/or only. (q−32)·a pairs stay within i16, fold to i32 in-lane and take
    /// their 16-value scale from a lane mask, so the only horizontal sum is per row.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAct(BlockQ6K* x, BlockQ8KAct* a, int nb)
    {
        int U = Vector<ushort>.Count;
        Vector<ushort> m4 = new(0x000F);
        Vector<ushort> m30 = new(0x0030);
        Vector<ushort> m3 = new(0x0003);
        Vector<short> bias = new(32);
        // i32 lane k of the u16 chunk starting at lane m covers group values 2m+4k..2m+4k+3.
        Vector<int> idx4 = Vector<int>.Indices * 4;
        Vector<int> sixteen = new(16);
        Vector<int> mask0 = Vector.LessThan(idx4, sixteen);
        Vector<int> mask1 = Vector.LessThan(idx4 + new Vector<int>(2 * U), sixteen);
        Vector<float> acc = Vector<float>.Zero;
        for (int i = 0; i < nb; i++, x++, a++)
        {
            Vector<int> sumi = Vector<int>.Zero;
            for (int j = 0; j < 2; j++)
            {
                byte* ql = x->Ql + 64 * j;
                byte* qh = x->Qh + 32 * j;
                sbyte* sc = x->Scales + 8 * j;
                short* A = a->A + 128 * j;
                Vector<int> s0 = new(sc[0]), s1 = new(sc[1]), s2 = new(sc[2]), s3 = new(sc[3]);
                Vector<int> s4 = new(sc[4]), s5 = new(sc[5]), s6 = new(sc[6]), s7 = new(sc[7]);
                for (int m = 0; m < 16; m += U)
                {
                    Vector<int> mask = m == 0 ? mask0 : mask1;
                    Vector<ushort> va = Unsafe.ReadUnaligned<Vector<ushort>>(ql + 2 * m);
                    Vector<ushort> vb = Unsafe.ReadUnaligned<Vector<ushort>>(ql + 32 + 2 * m);
                    Vector<ushort> h = Unsafe.ReadUnaligned<Vector<ushort>>(qh + 2 * m);

                    Vector<ushort> ev = (va & m4) | ((h & m3) << 4);
                    Vector<ushort> od = ((va >> 8) & m4) | ((h >> 4) & m30);
                    sumi += Group(ev, od, A + m, bias) * Vector.ConditionalSelect(mask, s0, s1);

                    ev = (vb & m4) | ((h << 2) & m30);
                    od = ((vb >> 8) & m4) | ((h >> 6) & m30);
                    sumi += Group(ev, od, A + 32 + m, bias) * Vector.ConditionalSelect(mask, s2, s3);

                    ev = ((va >> 4) & m4) | (h & m30);
                    od = (va >> 12) | ((h >> 8) & m30);
                    sumi += Group(ev, od, A + 64 + m, bias) * Vector.ConditionalSelect(mask, s4, s5);

                    ev = ((vb >> 4) & m4) | ((h >> 2) & m30);
                    od = (vb >> 12) | ((h >> 10) & m30);
                    sumi += Group(ev, od, A + 96 + m, bias) * Vector.ConditionalSelect(mask, s6, s7);
                }
            }

            acc = Vector.MultiplyAddEstimate(Vector.ConvertToSingle(sumi), new Vector<float>(HalfBits.ToSingle(x->D) * a->D), acc);
        }

        return Vector.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Group(Vector<ushort> ev, Vector<ushort> od, short* A, Vector<short> bias)
    {
        Vector<short> p = (Vector.AsVectorInt16(ev) - bias) * VecI8.LoadS16(A)
            + (Vector.AsVectorInt16(od) - bias) * VecI8.LoadS16(A + 16);
        return VecI8.Fold(p);
    }

    /// <summary>Decode one Q6_K superblock to signed values (-32..31) like <see cref="DotScalar"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DecodeBlockVec(BlockQ6K* x, sbyte* a)
    {
        byte* ql = x->Ql;
        byte* qh = x->Qh;
        Vector<byte> m15 = new Vector<byte>(15);
        Vector<byte> m3 = new Vector<byte>(3);
        Vector<sbyte> bias = new Vector<sbyte>(-32);
        for (int j = 0; j < Qk.SuperBlock / 128; j++)
        {
            int l = 0;
            for (; l + Vector<byte>.Count <= 32; l += Vector<byte>.Count)
            {
                Vector<byte> lo1 = VecI8.LoadU8(ql + l);
                Vector<byte> lo2 = VecI8.LoadU8(ql + 32 + l);
                Vector<byte> hb = VecI8.LoadU8(qh + l);
                Unsafe.WriteUnaligned(a + l, Vector.AsVectorSByte((lo1 & m15) | ((hb & m3) << 4)) + bias);
                Unsafe.WriteUnaligned(a + l + 32, Vector.AsVectorSByte((lo2 & m15) | (((hb >> 2) & m3) << 4)) + bias);
                Unsafe.WriteUnaligned(a + l + 64, Vector.AsVectorSByte((lo1 >> 4) | (((hb >> 4) & m3) << 4)) + bias);
                Unsafe.WriteUnaligned(a + l + 96, Vector.AsVectorSByte((lo2 >> 4) | ((hb >> 6) << 4)) + bias);
            }

            for (; l < 32; l++)
            {
                a[l] = (sbyte)(((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32);
                a[l + 32] = (sbyte)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = (sbyte)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = (sbyte)(((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }

            a += 128;
            ql += 64;
            qh += 32;
        }
    }

    public static float DotScalar(BlockQ6K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        sbyte* aux = stackalloc sbyte[Qk.SuperBlock];
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* a = aux;
            for (int j = 0; j < Qk.SuperBlock; j += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    a[l] = (sbyte)(((q4[l] & 0xF) | ((qh[l] & 3) << 4)) - 32);
                    a[l + 32] = (sbyte)(((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    a[l + 64] = (sbyte)(((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    a[l + 96] = (sbyte)(((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }

                a += 128;
                q4 += 64;
                qh += 32;
            }

            int acc = 0;
            a = aux;
            sbyte* q8 = y[i].Qs;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
            {
                int scale = x[i].Scales[j];
                for (int l = 0; l < 16; l++)
                    acc += scale * q8[l] * a[l];
                q8 += 16;
                a += 16;
            }

            sumf += HalfBits.ToSingle(x[i].D) * y[i].D * acc;
        }

        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAvx2(BlockQ6K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        Vector256<byte> m3 = Vector256.Create((byte)3);
        Vector256<byte> m15 = Vector256.Create((byte)15);
        Vector256<byte> m12 = Vector256.Create((byte)12);
        Vector256<byte> m48 = Vector256.Create((byte)48);
        Vector256<byte> mC0 = Vector256.Create((byte)0xC0);
        Vector256<float> acc = Vector256<float>.Zero;

        for (int i = 0; i < nb; i++)
        {
            float d = y[i].D * HalfBits.ToSingle(x[i].D);
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* q8 = y[i].Qs;

            Vector256<short> q8sums = Avx.LoadVector256((short*)y[i].Bsums);
            Vector128<sbyte> scales = Avx.LoadVector128(x[i].Scales);
            Vector256<short> scales16 = Avx2.ConvertToVector256Int16(scales);
            Vector256<int> q8sclsub = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);
            Vector256<int> sumi = Vector256<int>.Zero;
            int iscale = 0;

            for (int j = 0; j < Qk.SuperBlock / 128; j++)
            {
                Vector256<byte> q4bits1 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bits2 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bitsH = Avx.LoadVector256(qh);
                qh += 32;

                Vector256<byte> q4h0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsUInt16(), 4).AsByte();
                Vector256<byte> q4h1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsUInt16(), 2).AsByte();
                Vector256<byte> q4h2 = Avx2.And(q4bitsH, m48);
                Vector256<byte> q4h3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, mC0).AsUInt16(), 2).AsByte();

                Vector256<byte> q40 = Avx2.Or(Avx2.And(q4bits1, m15), q4h0);
                Vector256<byte> q41 = Avx2.Or(Avx2.And(q4bits2, m15), q4h1);
                Vector256<byte> q42 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsUInt16(), 4).AsByte(), m15), q4h2);
                Vector256<byte> q43 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsUInt16(), 4).AsByte(), m15), q4h3);

                Vector256<sbyte> q80 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q81 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q82 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q83 = Avx.LoadVector256(q8);
                q8 += 32;

                Vector256<short> sc0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 0)));
                Vector256<short> sc1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 1)));
                Vector256<short> sc2 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 2)));
                Vector256<short> sc3 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 3)));
                iscale += 4;

                Vector256<int> i0 = ScaleQ6(q40, q80, sc0);
                Vector256<int> i1 = ScaleQ6(q41, q81, sc1);
                Vector256<int> i2 = ScaleQ6(q42, q82, sc2);
                Vector256<int> i3 = ScaleQ6(q43, q83, sc3);
                sumi = Avx2.Add(sumi, Avx2.Add(i0, i1));
                sumi = Avx2.Add(sumi, Avx2.Add(i2, i3));
            }

            sumi = Avx2.Subtract(sumi, q8sclsub);
            Vector256<float> dv = Vector256.Create(d);
            acc = Simd.UseFma
                ? Fma.MultiplyAdd(dv, Avx.ConvertToVector256Single(sumi), acc)
                : Avx.Add(acc, Avx.Multiply(dv, Avx.ConvertToVector256Single(sumi)));
        }

        return VecDotQ4K.HorizontalSum(acc);
    }

    /// <summary>
    /// Decode each Q6_K superblock once and dot it against four Q8_K rows.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DotAvx2x4(BlockQ6K* x, BlockQ8K* y0, BlockQ8K* y1, BlockQ8K* y2, BlockQ8K* y3, int n, float* dst)
    {
        int nb = n / Qk.SuperBlock;
        Vector256<byte> m3 = Vector256.Create((byte)3);
        Vector256<byte> m15 = Vector256.Create((byte)15);
        Vector256<byte> m12 = Vector256.Create((byte)12);
        Vector256<byte> m48 = Vector256.Create((byte)48);
        Vector256<byte> mC0 = Vector256.Create((byte)0xC0);
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;

        for (int i = 0; i < nb; i++)
        {
            float xd = HalfBits.ToSingle(x[i].D);
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            Vector128<sbyte> scales = Avx.LoadVector128(x[i].Scales);
            Vector256<short> scales16 = Avx2.ConvertToVector256Int16(scales);
            Vector256<int> sum0 = Vector256<int>.Zero;
            Vector256<int> sum1 = Vector256<int>.Zero;
            Vector256<int> sum2 = Vector256<int>.Zero;
            Vector256<int> sum3 = Vector256<int>.Zero;
            int iscale = 0;

            for (int j = 0; j < Qk.SuperBlock / 128; j++)
            {
                Vector256<byte> q4bits1 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bits2 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bitsH = Avx.LoadVector256(qh);
                qh += 32;

                Vector256<byte> q4h0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsUInt16(), 4).AsByte();
                Vector256<byte> q4h1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsUInt16(), 2).AsByte();
                Vector256<byte> q4h2 = Avx2.And(q4bitsH, m48);
                Vector256<byte> q4h3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, mC0).AsUInt16(), 2).AsByte();

                Vector256<byte> q40 = Avx2.Or(Avx2.And(q4bits1, m15), q4h0);
                Vector256<byte> q41 = Avx2.Or(Avx2.And(q4bits2, m15), q4h1);
                Vector256<byte> q42 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsUInt16(), 4).AsByte(), m15), q4h2);
                Vector256<byte> q43 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsUInt16(), 4).AsByte(), m15), q4h3);

                Vector256<short> sc0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 0)));
                Vector256<short> sc1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 1)));
                Vector256<short> sc2 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 2)));
                Vector256<short> sc3 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 3)));
                iscale += 4;

                AccumulateQ6Lane(ref sum0, q40, q41, q42, q43, y0[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum1, q40, q41, q42, q43, y1[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum2, q40, q41, q42, q43, y2[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum3, q40, q41, q42, q43, y3[i].Qs + j * 128, sc0, sc1, sc2, sc3);
            }

            Vector256<int> sub0 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y0[i].Bsums), scales16), 5);
            Vector256<int> sub1 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y1[i].Bsums), scales16), 5);
            Vector256<int> sub2 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y2[i].Bsums), scales16), 5);
            Vector256<int> sub3 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y3[i].Bsums), scales16), 5);
            sum0 = Avx2.Subtract(sum0, sub0);
            sum1 = Avx2.Subtract(sum1, sub1);
            sum2 = Avx2.Subtract(sum2, sub2);
            sum3 = Avx2.Subtract(sum3, sub3);
            acc0 = MaddScale(acc0, xd * y0[i].D, sum0);
            acc1 = MaddScale(acc1, xd * y1[i].D, sum1);
            acc2 = MaddScale(acc2, xd * y2[i].D, sum2);
            acc3 = MaddScale(acc3, xd * y3[i].D, sum3);
        }

        dst[0] = VecDotQ4K.HorizontalSum(acc0);
        dst[1] = VecDotQ4K.HorizontalSum(acc1);
        dst[2] = VecDotQ4K.HorizontalSum(acc2);
        dst[3] = VecDotQ4K.HorizontalSum(acc3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateQ6Lane(
        ref Vector256<int> sum,
        Vector256<byte> q40,
        Vector256<byte> q41,
        Vector256<byte> q42,
        Vector256<byte> q43,
        sbyte* q8,
        Vector256<short> sc0,
        Vector256<short> sc1,
        Vector256<short> sc2,
        Vector256<short> sc3)
    {
        Vector256<int> i0 = ScaleQ6(q40, Avx.LoadVector256(q8), sc0);
        Vector256<int> i1 = ScaleQ6(q41, Avx.LoadVector256(q8 + 32), sc1);
        Vector256<int> i2 = ScaleQ6(q42, Avx.LoadVector256(q8 + 64), sc2);
        Vector256<int> i3 = ScaleQ6(q43, Avx.LoadVector256(q8 + 96), sc3);
        sum = Avx2.Add(sum, Avx2.Add(Avx2.Add(i0, i1), Avx2.Add(i2, i3)));
    }

    /// <summary>
    /// 32 unsigned 6-bit values × signed activations, then the shuffled Q6 scales.
    /// <paramref name="sc"/> is 8 copies of s0 followed by 8 copies of s1.
    /// vpmaddubsw pairs beat vpdpbusd+vpmulld here: the VNNI variant scales each
    /// dot lane separately, which is slower on Raptor Lake measured 2026-04.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> ScaleQ6(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc) =>
        ScaleQ6Avx2(q, a, sc);

    public static Vector256<int> ScaleQ6Avx2(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc) =>
        Avx2.MultiplyAddAdjacent(sc, Avx2.MultiplyAddAdjacent(q, a));

    public static Vector256<int> ScaleQ6Vnni(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc)
    {
        Vector256<int> dots = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, q, a);
        int s0 = sc.GetElement(0);
        int s1 = sc.GetElement(8);
        return Avx2.MultiplyLow(dots, Vector256.Create(s0, s0, s0, s0, s1, s1, s1, s1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> MaddScale(Vector256<float> acc, float d, Vector256<int> sumi)
    {
        Vector256<float> dv = Vector256.Create(d);
        Vector256<float> v = Avx.ConvertToVector256Single(sumi);
        return Simd.UseFma ? Fma.MultiplyAdd(dv, v, acc) : Avx.Add(acc, Avx.Multiply(dv, v));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<sbyte> ScaleShuffle(int i)
    {
        ReadOnlySpan<byte> shuffle = ScaleShuffleBytes;
        return Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(shuffle), (nuint)(i * 16)).AsSByte();
    }

    private static ReadOnlySpan<byte> ScaleShuffleBytes =>
    [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];

    public static void Gemv(BlockQ6K* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null, ScratchArena? scratch = null)
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

    public static void Gemm(BlockQ6K* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        if (tokens == 1)
        {
            Gemv(weights, input, output, nIn, nOut, pool, scratch);
            return;
        }

        nuint q8Bytes = (nuint)((long)tokens * Q8K.RowBytes(nIn));
        NativeBuffer? owned = scratch == null ? new NativeBuffer(q8Bytes) : null;
        BlockQ8K* y = (BlockQ8K*)(scratch != null ? scratch.A(q8Bytes) : owned!.Pointer);
        try
        {
            if (pool == null)
            {
                Quantize(input, y, nIn, tokens, null);
                GemmFromQ8(weights, y, output, nIn, nOut, tokens, null);
            }
            else
            {
                int nb = nIn / Qk.SuperBlock;
                pool.For(Math.Max(tokens, nOut), (int worker, int workers) =>
                {
                    int t0 = tokens * worker / workers;
                    int t1 = tokens * (worker + 1) / workers;
                    for (int t = t0; t < t1; t++)
                        Q8K.QuantizeRow(input + t * nIn, y + t * nb, nIn);
                    pool.Barrier();
                    GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, worker, workers);
                });
            }
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void Quantize(float* input, BlockQ8K* y, int nIn, int tokens, CpuThreadPool? pool)
    {
        int nb = nIn / Qk.SuperBlock;
        void QuantBody(int worker, int workers)
        {
            int begin = tokens * worker / workers;
            int end = tokens * (worker + 1) / workers;
            for (int t = begin; t < end; t++)
                Q8K.QuantizeRow(input + t * nIn, y + t * nb, nIn);
        }

        if (pool == null)
            QuantBody(0, 1);
        else
            pool.For(tokens, QuantBody);
    }

    public static void GemmFromQ8(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool)
    {
        if (pool == null)
            GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, 0, 1);
        else
            pool.For(nOut, (int worker, int workers) => GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, worker, workers));
    }

    [SkipLocalsInit]
    private static void GemmFromQ8Range(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, int tokens, int worker, int workers)
    {
        int nb = nIn / Qk.SuperBlock;
        int begin = nOut * worker / workers;
        int end = nOut * (worker + 1) / workers;
        float* tmp = stackalloc float[4];
        int t = 0;
        if (Simd.UseAvx2)
        {
            for (; t + 3 < tokens; t += 4)
            {
                BlockQ8K* y0 = y + t * nb;
                BlockQ8K* y1 = y + (t + 1) * nb;
                BlockQ8K* y2 = y + (t + 2) * nb;
                BlockQ8K* y3 = y + (t + 3) * nb;
                for (int row = begin; row < end; row++)
                {
                    DotAvx2x4(weights + row * nb, y0, y1, y2, y3, nIn, tmp);
                    output[(t + 0) * nOut + row] = tmp[0];
                    output[(t + 1) * nOut + row] = tmp[1];
                    output[(t + 2) * nOut + row] = tmp[2];
                    output[(t + 3) * nOut + row] = tmp[3];
                }
            }
        }

        for (; t < tokens; t++)
        {
            BlockQ8K* yt = y + t * nb;
            for (int row = begin; row < end; row++)
                output[t * nOut + row] = Dot(weights + row * nb, yt, nIn);
        }
    }

    public static void GemvPrequant(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, CpuThreadPool? pool)
        => GemvPrequantMulti(y, nIn, pool, weights, output, nOut, null, null, 0);

    /// <summary>Portable GEMV uses the <see cref="BlockQ8KAct"/> view; null means the AVX2/scalar Dot.</summary>
    private static bool UseVecAct => !Simd.UseAvx2 && !Simd.UseDp && VecI8.PairLayoutSupported;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Row(BlockQ6K* w, BlockQ8K* y, BlockQ8KAct* act, int nIn, int nb)
        => act != null ? DotAct(w, act, nb) : Dot(w, y, nIn);

    /// <summary>
    /// One parallel region over several weights sharing the same Q8_K activation:
    /// decode pays one dispatch for Q,K,V or gate,up instead of one per weight.
    /// </summary>
    public static void GemvPrequantMulti(BlockQ8K* y, int nIn, CpuThreadPool? pool, BlockQ6K* w0, float* d0, int n0, BlockQ6K* w1, float* d1, int n1, BlockQ6K* w2 = null, float* d2 = null, int n2 = 0)
    {
        int nb = nIn / Qk.SuperBlock;
        int total = n0 + n1 + n2;
        int next = 0;
        NativeBuffer? actOwned = UseVecAct && nb > 64 ? new NativeBuffer((nuint)(nb * sizeof(BlockQ8KAct))) : null;
        BlockQ8KAct* actStack = stackalloc BlockQ8KAct[UseVecAct && actOwned == null ? nb : 0];
        BlockQ8KAct* act = actOwned != null ? (BlockQ8KAct*)actOwned.Pointer : UseVecAct ? actStack : null;
        if (act != null)
            Q8K.ToVecAct(y, act, nb);

        // Dynamic chunks: a worker on a slow memory patch takes fewer rows
        // instead of gating the whole GEMV (same trick as ggml mul_mat).
        // ~16 claims per worker keeps each claim a long contiguous stream.
        void Body(int worker, int workers)
        {
            int chunk = Math.Max(8, total / (workers * 16));
            int begin;
            while ((begin = Interlocked.Add(ref next, chunk) - chunk) < total)
            {
                int end = Math.Min(begin + chunk, total);
                int r0 = Math.Clamp(begin, 0, n0);
                int r1 = Math.Clamp(end, 0, n0);
                for (int row = r0; row < r1; row++)
                    d0[row] = Row(w0 + row * nb, y, act, nIn, nb);
                r0 = Math.Clamp(begin - n0, 0, n1);
                r1 = Math.Clamp(end - n0, 0, n1);
                for (int row = r0; row < r1; row++)
                    d1[row] = Row(w1 + row * nb, y, act, nIn, nb);
                r0 = Math.Clamp(begin - n0 - n1, 0, n2);
                r1 = Math.Clamp(end - n0 - n1, 0, n2);
                for (int row = r0; row < r1; row++)
                    d2[row] = Row(w2 + row * nb, y, act, nIn, nb);
            }
        }

        try
        {
            if (pool == null || total == 0)
                Body(0, 1);
            else
                pool.For(total, Body);
        }
        finally
        {
            actOwned?.Dispose();
        }
    }
}
