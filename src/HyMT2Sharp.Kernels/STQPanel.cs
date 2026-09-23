using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class STQPanel
{
    // The tile construction is identical to Q2x8.  RowSum emits the natural
    // AVX2 lane order, which is restored with this permutation on store.
    private static Vector256<int> ColumnOrder => Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Gemv(BlockSTQ1_0x8* w, BlockQ8K* x, float* dst, int n)
    {
        int nb = n / STQ1_0.BlockLength;
        Vector256<byte> mask = Vector256.Create((byte)3);
        Vector256<float> acc = Vector256<float>.Zero;

        for (int b = 0; b < nb; b++)
        {
            byte* q = w[b].Qs;
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

            Vector256<int> dot = RowSum(i0, i1);
            dot = Avx2.Subtract(dot, Vector256.Create(TotalSum(x + b)));
            Vector256<float> scale = Avx.Multiply(
                Avx2.PermuteVar8x32(Avx.LoadVector256(w[b].D), ColumnOrder),
                Vector256.Create(x[b].D));
            acc = Fmadd(Avx.ConvertToVector256Single(dot), scale, acc);
        }

        Avx.Store(dst, Avx2.PermuteVar8x32(acc, ColumnOrder));
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void Gemm(BlockSTQ1_0x8* weights, BlockQ8Kx4* x, float* dst,
        int n, int ldc, int tokens, int groups)
    {
        int nb = n / STQ1_0.BlockLength;
        Vector256<byte> mask = Vector256.Create((byte)3);
        Vector256<int>* corrections = stackalloc Vector256<int>[nb];
        Vector256<float>* activationScales = stackalloc Vector256<float>[nb * 4];

        for (int t = 0; t < tokens / 4; t++)
        {
            BlockQ8Kx4* rows = x + (long)t * nb;
            for (int b = 0; b < nb; b++)
            {
                corrections[b] = TotalSums(rows + b);
                activationScales[b * 4 + 0] = Vector256.Create(rows[b].D[0]);
                activationScales[b * 4 + 1] = Vector256.Create(rows[b].D[1]);
                activationScales[b * 4 + 2] = Vector256.Create(rows[b].D[2]);
                activationScales[b * 4 + 3] = Vector256.Create(rows[b].D[3]);
            }

            for (int c = 0; c < groups; c++)
            {
                BlockSTQ1_0x8* w = weights + (long)c * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                Vector256<float> weightScale = Vector256<float>.Zero;

                for (int b = 0; b < nb; b++)
                {
                    byte* q = w[b].Qs;
                    long* a = (long*)rows[b].Qs;
                    Vector256<short> i0a = Vector256<short>.Zero, i0b = Vector256<short>.Zero;
                    Vector256<short> i1a = Vector256<short>.Zero, i1b = Vector256<short>.Zero;
                    Vector256<short> i2a = Vector256<short>.Zero, i2b = Vector256<short>.Zero;
                    Vector256<short> i3a = Vector256<short>.Zero, i3b = Vector256<short>.Zero;

                    // Seed from the first 32-value tile, then accumulate the
                    // remaining tiles.  This keeps the eight output rows in
                    // registers while four activation rows share each load.
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

                    // STQ has one independent fp16 scale for every 256-value
                    // block (unlike Q2_0C's shared 512-value scale).
                    weightScale = Avx2.PermuteVar8x32(Avx.LoadVector256(w[b].D), ColumnOrder);
                    Vector256<int> correction = corrections[b];
                    acc0 = Fmadd(Avx.ConvertToVector256Single(Avx2.Subtract(RowSum(i0a, i0b), Avx2.Shuffle(correction, 0x00))),
                        Avx.Multiply(weightScale, activationScales[b * 4 + 0]), acc0);
                    acc1 = Fmadd(Avx.ConvertToVector256Single(Avx2.Subtract(RowSum(i1a, i1b), Avx2.Shuffle(correction, 0x55))),
                        Avx.Multiply(weightScale, activationScales[b * 4 + 1]), acc1);
                    acc2 = Fmadd(Avx.ConvertToVector256Single(Avx2.Subtract(RowSum(i2a, i2b), Avx2.Shuffle(correction, 0xAA))),
                        Avx.Multiply(weightScale, activationScales[b * 4 + 2]), acc2);
                    acc3 = Fmadd(Avx.ConvertToVector256Single(Avx2.Subtract(RowSum(i3a, i3b), Avx2.Shuffle(correction, 0xFF))),
                        Avx.Multiply(weightScale, activationScales[b * 4 + 3]), acc3);
                }

                Avx.Store(dst + c * 8 + (t * 4 + 0) * ldc, Avx2.PermuteVar8x32(acc0, ColumnOrder));
                Avx.Store(dst + c * 8 + (t * 4 + 1) * ldc, Avx2.PermuteVar8x32(acc1, ColumnOrder));
                Avx.Store(dst + c * 8 + (t * 4 + 2) * ldc, Avx2.PermuteVar8x32(acc2, ColumnOrder));
                Avx.Store(dst + c * 8 + (t * 4 + 3) * ldc, Avx2.PermuteVar8x32(acc3, ColumnOrder));
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> RowSum(Vector256<short> a, Vector256<short> b) =>
        Avx2.HorizontalAdd(Avx2.MultiplyAddAdjacent(a, Vector256.Create((short)1)),
                          Avx2.MultiplyAddAdjacent(b, Vector256.Create((short)1)));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int TotalSum(BlockQ8K* x)
    {
        int sum = 0;
        for (int i = 0; i < Qk.SuperBlock / 16; i++)
            sum += x->Bsums[i];
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> TotalSums(BlockQ8Kx4* x)
    {
        short* bs = x->Bsums;
        Vector256<short> quarters = Avx2.Add(
            Avx2.Add(Avx.LoadVector256(bs), Avx.LoadVector256(bs + 16)),
            Avx2.Add(Avx.LoadVector256(bs + 32), Avx.LoadVector256(bs + 48)));
        Vector256<int> pairs = Avx2.MultiplyAddAdjacent(quarters, Vector256.Create((short)1));
        Vector128<int> sums = Ssse3.HorizontalAdd(pairs.GetLower(), pairs.GetUpper());
        return Vector256.Create(sums, sums);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Fmadd(Vector256<float> x, Vector256<float> y, Vector256<float> z) =>
        Simd.UseFma ? Fma.MultiplyAdd(x, y, z) : Avx.Add(Avx.Multiply(x, y), z);
}
