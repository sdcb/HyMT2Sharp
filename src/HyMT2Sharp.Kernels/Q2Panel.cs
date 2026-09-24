using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class Q2Panel
{
    // vphaddd of the two four-column halves produces this order.
    private static Vector256<int> ColumnOrder => Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Gemv(BlockQ2x8* w, BlockQ8K* x, float* dst, int n)
    {
        if (Simd.UseDp && !Simd.UseAvx2)
        {
            GemvNeon(w, x, dst, n);
            return;
        }

        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<byte> mask = Vector256.Create((byte)3);
        for (int b = 0; b < n / 256; b++)
        {
            byte* q = w[b / 2].Qs + (b & 1) * 512;
            long* a = (long*)x[b].Qs;
            Vector256<short> i0 = Vector256<short>.Zero;
            Vector256<short> i1 = Vector256<short>.Zero;
            for (int k = 0; k < 8; k++)
            {
                Vector256<byte> p0 = Avx.LoadVector256(q);
                Vector256<byte> p1 = Avx.LoadVector256(q + 32);
                for (int plane = 0; plane < 4; plane++)
                {
                    Vector256<sbyte> l = Avx2.BroadcastScalarToVector256(a++).AsSByte();
                    i0 = Avx2.Add(i0, Avx2.MultiplyAddAdjacent(Avx2.And(p0, mask), l));
                    i1 = Avx2.Add(i1, Avx2.MultiplyAddAdjacent(Avx2.And(p1, mask), l));
                    p0 = Avx2.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                    p1 = Avx2.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                }
                q += 64;
            }
            Vector256<int> pairs = Avx2.MultiplyAddAdjacent(Avx.LoadVector256(x[b].Bsums), Vector256.Create((short)1));
            Vector128<int> sums = Sse2.Add(pairs.GetLower(), pairs.GetUpper());
            sums = Ssse3.HorizontalAdd(sums, sums);
            int sum = Ssse3.HorizontalAdd(sums, sums).ToScalar();
            Vector256<int> dot = RowSum(i0, i1);
            dot = Avx2.Subtract(Avx2.Add(dot, dot), Vector256.Create(3 * sum));
            Vector256<float> scale = Avx.Multiply(
                Avx2.PermuteVar8x32(Avx.LoadVector256(w[b / 2].D), ColumnOrder), Vector256.Create(x[b].D));
            acc = Fmadd(Avx.ConvertToVector256Single(dot), scale, acc);
        }
        Avx.Store(dst, Avx2.PermuteVar8x32(acc, ColumnOrder));
    }

    /// <summary>SDOT GEMV: 2-bit planes decoded per 64-byte tile, one correction per block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvNeon(BlockQ2x8* w, BlockQ8K* x, float* dst, int n)
    {
        Vector128<byte> mask = Vector128.Create((byte)3);
        Vector128<float> accL = Vector128<float>.Zero;
        Vector128<float> accH = Vector128<float>.Zero;
        for (int b = 0; b < n / 256; b++)
        {
            byte* q = w[b / 2].Qs + (b & 1) * 512;
            sbyte* a = x[b].Qs;
            Vector128<int> i0 = Vector128<int>.Zero;
            Vector128<int> i1 = Vector128<int>.Zero;
            Vector128<int> i2 = Vector128<int>.Zero;
            Vector128<int> i3 = Vector128<int>.Zero;
            for (int k = 0; k < 8; k++)
            {
                Vector128<byte> p0 = Neon.LoadU16(q);
                Vector128<byte> p1 = Neon.LoadU16(q + 16);
                Vector128<byte> p2 = Neon.LoadU16(q + 32);
                Vector128<byte> p3 = Neon.LoadU16(q + 48);
                for (int plane = 0; plane < 4; plane++)
                {
                    Vector128<sbyte> act = Neon.Dup8(a + (k * 4 + plane) * 8);
                    i0 = Neon.Sdot(i0, (p0 & mask).AsSByte(), act);
                    i1 = Neon.Sdot(i1, (p1 & mask).AsSByte(), act);
                    i2 = Neon.Sdot(i2, (p2 & mask).AsSByte(), act);
                    i3 = Neon.Sdot(i3, (p3 & mask).AsSByte(), act);
                    p0 = AdvSimd.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                    p1 = AdvSimd.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                    p2 = AdvSimd.ShiftRightLogical(p2.AsUInt16(), 2).AsByte();
                    p3 = AdvSimd.ShiftRightLogical(p3.AsUInt16(), 2).AsByte();
                }
                q += 64;
            }
            Vector128<int> corr = Vector128.Create(3 * Neon.BsumAll(x[b].Bsums));
            Vector128<int> dotL = AdvSimd.Subtract(AdvSimd.Add(Neon.PairAdd(i0, i1), Neon.PairAdd(i0, i1)), corr);
            Vector128<int> dotH = AdvSimd.Subtract(AdvSimd.Add(Neon.PairAdd(i2, i3), Neon.PairAdd(i2, i3)), corr);
            Vector128<float> xd = Vector128.Create(x[b].D);
            accL = AdvSimd.FusedMultiplyAdd(accL, AdvSimd.ConvertToSingle(dotL), AdvSimd.Multiply(Unsafe.ReadUnaligned<Vector128<float>>(w[b / 2].D), xd));
            accH = AdvSimd.FusedMultiplyAdd(accH, AdvSimd.ConvertToSingle(dotH), AdvSimd.Multiply(Unsafe.ReadUnaligned<Vector128<float>>(w[b / 2].D + 4), xd));
        }
        Unsafe.WriteUnaligned(dst, accL);
        Unsafe.WriteUnaligned(dst + 4, accH);
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Gemm(BlockQ2x8* weights, BlockQ8Kx4* x, float* dst, int n, int ldc, int tokens, int groups)
    {
        if (Simd.UseDp && !Simd.UseAvx2)
        {
            GemmNeon(weights, x, dst, n, ldc, tokens, groups);
            return;
        }

        int nb = n / 512;
        Vector256<byte> mask = Vector256.Create((byte)3);
        Vector256<int>* corrections = stackalloc Vector256<int>[nb * 2];
        Vector256<float>* activationScales = stackalloc Vector256<float>[nb * 2 * 4];
        for (int t = 0; t < tokens / 4; t++)
        {
            BlockQ8Kx4* rows = x + t * nb * 2;
            for (int b = 0; b < nb * 2; b++)
            {
                corrections[b] = TotalSums(rows + b);
                activationScales[b * 4 + 0] = Vector256.Create(rows[b].D[0]);
                activationScales[b * 4 + 1] = Vector256.Create(rows[b].D[1]);
                activationScales[b * 4 + 2] = Vector256.Create(rows[b].D[2]);
                activationScales[b * 4 + 3] = Vector256.Create(rows[b].D[3]);
            }
            for (int c = 0; c < groups; c++)
            {
                BlockQ2x8* w = weights + c * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                Vector256<float> weightScale = Vector256<float>.Zero;
                for (int b = 0; b < nb * 2; b++)
                {
                    byte* q = w[b / 2].Qs + (b & 1) * 512;
                    long* a = (long*)rows[b].Qs;
                    Vector256<short> i0a = Vector256<short>.Zero, i0b = Vector256<short>.Zero;
                    Vector256<short> i1a = Vector256<short>.Zero, i1b = Vector256<short>.Zero;
                    Vector256<short> i2a = Vector256<short>.Zero, i2b = Vector256<short>.Zero;
                    Vector256<short> i3a = Vector256<short>.Zero, i3b = Vector256<short>.Zero;
                    // Seed the accumulators from the first 8-value group. This
                    // removes eight vector adds from every 512-value block.
                    Vector256<byte> p0 = Avx.LoadVector256(q);
                    Vector256<byte> p1 = Avx.LoadVector256(q + 32);
                    Vector256<byte> r0 = Avx2.And(p0, mask);
                    Vector256<byte> r1 = Avx2.And(p1, mask);
                    Vector256<sbyte> l = Avx2.BroadcastScalarToVector256(a).AsSByte();
                    i0a = Avx2.MultiplyAddAdjacent(r0, l);
                    i0b = Avx2.MultiplyAddAdjacent(r1, l);
                    l = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                    i1a = Avx2.MultiplyAddAdjacent(r0, l);
                    i1b = Avx2.MultiplyAddAdjacent(r1, l);
                    l = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                    i2a = Avx2.MultiplyAddAdjacent(r0, l);
                    i2b = Avx2.MultiplyAddAdjacent(r1, l);
                    l = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                    i3a = Avx2.MultiplyAddAdjacent(r0, l);
                    i3b = Avx2.MultiplyAddAdjacent(r1, l);
                    p0 = Avx2.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                    p1 = Avx2.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                    a += 4;
                    for (int plane = 1; plane < 4; plane++)
                    {
                        r0 = Avx2.And(p0, mask);
                        r1 = Avx2.And(p1, mask);
                        l = Avx2.BroadcastScalarToVector256(a).AsSByte();
                        i0a = Avx2.Add(i0a, Avx2.MultiplyAddAdjacent(r0, l));
                        i0b = Avx2.Add(i0b, Avx2.MultiplyAddAdjacent(r1, l));
                        l = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                        i1a = Avx2.Add(i1a, Avx2.MultiplyAddAdjacent(r0, l));
                        i1b = Avx2.Add(i1b, Avx2.MultiplyAddAdjacent(r1, l));
                        l = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                        i2a = Avx2.Add(i2a, Avx2.MultiplyAddAdjacent(r0, l));
                        i2b = Avx2.Add(i2b, Avx2.MultiplyAddAdjacent(r1, l));
                        l = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                        i3a = Avx2.Add(i3a, Avx2.MultiplyAddAdjacent(r0, l));
                        i3b = Avx2.Add(i3b, Avx2.MultiplyAddAdjacent(r1, l));
                        p0 = Avx2.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                        p1 = Avx2.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                        a += 4;
                    }
                    q += 64;
                    for (int k = 1; k < 8; k++)
                    {
                        p0 = Avx.LoadVector256(q);
                        p1 = Avx.LoadVector256(q + 32);
                        for (int plane = 0; plane < 4; plane++)
                        {
                            r0 = Avx2.And(p0, mask);
                            r1 = Avx2.And(p1, mask);
                            l = Avx2.BroadcastScalarToVector256(a).AsSByte();
                            i0a = Avx2.Add(i0a, Avx2.MultiplyAddAdjacent(r0, l));
                            i0b = Avx2.Add(i0b, Avx2.MultiplyAddAdjacent(r1, l));
                            l = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                            i1a = Avx2.Add(i1a, Avx2.MultiplyAddAdjacent(r0, l));
                            i1b = Avx2.Add(i1b, Avx2.MultiplyAddAdjacent(r1, l));
                            l = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                            i2a = Avx2.Add(i2a, Avx2.MultiplyAddAdjacent(r0, l));
                            i2b = Avx2.Add(i2b, Avx2.MultiplyAddAdjacent(r1, l));
                            l = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                            i3a = Avx2.Add(i3a, Avx2.MultiplyAddAdjacent(r0, l));
                            i3b = Avx2.Add(i3b, Avx2.MultiplyAddAdjacent(r1, l));
                            p0 = Avx2.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                            p1 = Avx2.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                            a += 4;
                        }
                        q += 64;
                    }
                    // Each short holds 64 products: |sum| <= 64*3*128 = 24576.
                    // Widen BEFORE reducing across pairs; an int16 horizontal
                    // add here would overflow for uniform extreme inputs.
                    Vector256<int> correction = corrections[b];
                    // Both 256-value halves share one Q2 block scale.
                    if ((b & 1) == 0)
                        weightScale = Avx2.PermuteVar8x32(Avx.LoadVector256(w[b / 2].D), ColumnOrder);
                    acc0 = Fmadd(Correct(RowSum(i0a, i0b), Avx2.Shuffle(correction, 0x00)), Avx.Multiply(weightScale, activationScales[b * 4 + 0]), acc0);
                    acc1 = Fmadd(Correct(RowSum(i1a, i1b), Avx2.Shuffle(correction, 0x55)), Avx.Multiply(weightScale, activationScales[b * 4 + 1]), acc1);
                    acc2 = Fmadd(Correct(RowSum(i2a, i2b), Avx2.Shuffle(correction, 0xAA)), Avx.Multiply(weightScale, activationScales[b * 4 + 2]), acc2);
                    acc3 = Fmadd(Correct(RowSum(i3a, i3b), Avx2.Shuffle(correction, 0xFF)), Avx.Multiply(weightScale, activationScales[b * 4 + 3]), acc3);
                }
                Avx.Store(dst + (t * 4 + 0) * ldc + c * 8, Avx2.PermuteVar8x32(acc0, ColumnOrder));
                Avx.Store(dst + (t * 4 + 1) * ldc + c * 8, Avx2.PermuteVar8x32(acc1, ColumnOrder));
                Avx.Store(dst + (t * 4 + 2) * ldc + c * 8, Avx2.PermuteVar8x32(acc2, ColumnOrder));
                Avx.Store(dst + (t * 4 + 3) * ldc + c * 8, Avx2.PermuteVar8x32(acc3, ColumnOrder));
            }
        }
    }

    /// <summary>SDOT 4×8 GEMM over the 2-bit tile layout; correction folds once per block.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmNeon(BlockQ2x8* weights, BlockQ8Kx4* x, float* dst, int n, int ldc, int tokens, int groups)
    {
        int nb = n / 512;
        Vector128<byte> mask = Vector128.Create((byte)3);
        int* corrections = stackalloc int[nb * 2 * 4];
        for (int t = 0; t < tokens / 4; t++)
        {
            BlockQ8Kx4* rows = x + t * nb * 2;
            for (int b = 0; b < nb * 2; b++)
            {
                short* bs = rows[b].Bsums;
                for (int r = 0; r < 4; r++)
                {
                    int s = 0;
                    for (int q4 = 0; q4 < 4; q4++)
                        s += bs[q4 * 16 + r * 4] + bs[q4 * 16 + r * 4 + 1] + bs[q4 * 16 + r * 4 + 2] + bs[q4 * 16 + r * 4 + 3];
                    corrections[b * 4 + r] = 3 * s;
                }
            }
            for (int c = 0; c < groups; c++)
            {
                BlockQ2x8* w = weights + c * nb;
                Vector128<float> acc0L = Vector128<float>.Zero, acc0H = Vector128<float>.Zero;
                Vector128<float> acc1L = Vector128<float>.Zero, acc1H = Vector128<float>.Zero;
                Vector128<float> acc2L = Vector128<float>.Zero, acc2H = Vector128<float>.Zero;
                Vector128<float> acc3L = Vector128<float>.Zero, acc3H = Vector128<float>.Zero;
                for (int b = 0; b < nb * 2; b++)
                {
                    byte* q = w[b / 2].Qs + (b & 1) * 512;
                    sbyte* a = rows[b].Qs;
                    Vector128<int> i00 = Vector128<int>.Zero, i01 = Vector128<int>.Zero, i02 = Vector128<int>.Zero, i03 = Vector128<int>.Zero;
                    Vector128<int> i10 = Vector128<int>.Zero, i11 = Vector128<int>.Zero, i12 = Vector128<int>.Zero, i13 = Vector128<int>.Zero;
                    Vector128<int> i20 = Vector128<int>.Zero, i21 = Vector128<int>.Zero, i22 = Vector128<int>.Zero, i23 = Vector128<int>.Zero;
                    Vector128<int> i30 = Vector128<int>.Zero, i31 = Vector128<int>.Zero, i32 = Vector128<int>.Zero, i33 = Vector128<int>.Zero;
                    for (int k = 0; k < 8; k++)
                    {
                        Vector128<byte> p0 = Neon.LoadU16(q);
                        Vector128<byte> p1 = Neon.LoadU16(q + 16);
                        Vector128<byte> p2 = Neon.LoadU16(q + 32);
                        Vector128<byte> p3 = Neon.LoadU16(q + 48);
                        for (int plane = 0; plane < 4; plane++)
                        {
                            Vector128<sbyte> w0 = (p0 & mask).AsSByte();
                            Vector128<sbyte> w1 = (p1 & mask).AsSByte();
                            Vector128<sbyte> w2 = (p2 & mask).AsSByte();
                            Vector128<sbyte> w3 = (p3 & mask).AsSByte();
                            sbyte* ab = a + (k * 4 + plane) * 32;
                            Vector128<sbyte> act = Neon.Dup8(ab);
                            i00 = Neon.Sdot(i00, w0, act);
                            i01 = Neon.Sdot(i01, w1, act);
                            i02 = Neon.Sdot(i02, w2, act);
                            i03 = Neon.Sdot(i03, w3, act);
                            act = Neon.Dup8(ab + 8);
                            i10 = Neon.Sdot(i10, w0, act);
                            i11 = Neon.Sdot(i11, w1, act);
                            i12 = Neon.Sdot(i12, w2, act);
                            i13 = Neon.Sdot(i13, w3, act);
                            act = Neon.Dup8(ab + 16);
                            i20 = Neon.Sdot(i20, w0, act);
                            i21 = Neon.Sdot(i21, w1, act);
                            i22 = Neon.Sdot(i22, w2, act);
                            i23 = Neon.Sdot(i23, w3, act);
                            act = Neon.Dup8(ab + 24);
                            i30 = Neon.Sdot(i30, w0, act);
                            i31 = Neon.Sdot(i31, w1, act);
                            i32 = Neon.Sdot(i32, w2, act);
                            i33 = Neon.Sdot(i33, w3, act);
                            p0 = AdvSimd.ShiftRightLogical(p0.AsUInt16(), 2).AsByte();
                            p1 = AdvSimd.ShiftRightLogical(p1.AsUInt16(), 2).AsByte();
                            p2 = AdvSimd.ShiftRightLogical(p2.AsUInt16(), 2).AsByte();
                            p3 = AdvSimd.ShiftRightLogical(p3.AsUInt16(), 2).AsByte();
                        }
                        q += 64;
                    }
                    Vector128<float> dL = Unsafe.ReadUnaligned<Vector128<float>>(w[b / 2].D);
                    Vector128<float> dH = Unsafe.ReadUnaligned<Vector128<float>>(w[b / 2].D + 4);
                    FinishQ2Neon(ref acc0L, ref acc0H, i00, i01, i02, i03, rows[b].D[0], corrections[b * 4 + 0], dL, dH);
                    FinishQ2Neon(ref acc1L, ref acc1H, i10, i11, i12, i13, rows[b].D[1], corrections[b * 4 + 1], dL, dH);
                    FinishQ2Neon(ref acc2L, ref acc2H, i20, i21, i22, i23, rows[b].D[2], corrections[b * 4 + 2], dL, dH);
                    FinishQ2Neon(ref acc3L, ref acc3H, i30, i31, i32, i33, rows[b].D[3], corrections[b * 4 + 3], dL, dH);
                }
                float* row = dst + t * 4 * ldc + c * 8;
                Unsafe.WriteUnaligned(row, acc0L);
                Unsafe.WriteUnaligned(row + 4, acc0H);
                row += ldc;
                Unsafe.WriteUnaligned(row, acc1L);
                Unsafe.WriteUnaligned(row + 4, acc1H);
                row += ldc;
                Unsafe.WriteUnaligned(row, acc2L);
                Unsafe.WriteUnaligned(row + 4, acc2H);
                row += ldc;
                Unsafe.WriteUnaligned(row, acc3L);
                Unsafe.WriteUnaligned(row + 4, acc3H);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FinishQ2Neon(ref Vector128<float> accL, ref Vector128<float> accH,
        Vector128<int> i0, Vector128<int> i1, Vector128<int> i2, Vector128<int> i3,
        float ad, int corr, Vector128<float> dL, Vector128<float> dH)
    {
        Vector128<int> corrV = Vector128.Create(corr);
        Vector128<int> dotL = AdvSimd.Subtract(AdvSimd.Add(Neon.PairAdd(i0, i1), Neon.PairAdd(i0, i1)), corrV);
        Vector128<int> dotH = AdvSimd.Subtract(AdvSimd.Add(Neon.PairAdd(i2, i3), Neon.PairAdd(i2, i3)), corrV);
        Vector128<float> adV = Vector128.Create(ad);
        accL = AdvSimd.FusedMultiplyAdd(accL, AdvSimd.ConvertToSingle(dotL), AdvSimd.Multiply(dL, adV));
        accH = AdvSimd.FusedMultiplyAdd(accH, AdvSimd.ConvertToSingle(dotH), AdvSimd.Multiply(dH, adV));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> RowSum(Vector256<short> a, Vector256<short> b) =>
        Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(a, Vector256.Create((short)1)),
                          Avx2.MultiplyAddAdjacent(b, Vector256.Create((short)1)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> TotalSums(BlockQ8Kx4* x)
    {
        short* bs = x->Bsums;
        // Each lane sums 64 signed activations, so this int16 addition is
        // safe even for -128. Widen before summing the four lanes per row.
        Vector256<short> quarters = Avx2.Add(
            Avx2.Add(Avx.LoadVector256(bs), Avx.LoadVector256(bs + 16)),
            Avx2.Add(Avx.LoadVector256(bs + 32), Avx.LoadVector256(bs + 48)));
        Vector256<int> pairs = Avx2.MultiplyAddAdjacent(quarters, Vector256.Create((short)1));
        Vector128<int> sums = Ssse3.HorizontalAdd(pairs.GetLower(), pairs.GetUpper());
        sums = Sse2.Add(sums, Sse2.Add(sums, sums));
        return Vector256.Create(sums, sums);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Correct(Vector256<int> dot, Vector256<int> correction) =>
        Avx.ConvertToVector256Single(Avx2.Subtract(Avx2.Add(dot, dot), correction));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Fmadd(Vector256<float> x, Vector256<float> y, Vector256<float> z) =>
        Simd.UseFma ? Fma.MultiplyAdd(x, y, z) : Avx.Add(Avx.Multiply(x, y), z);
}
