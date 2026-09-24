using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe partial class GemmQ4K
{
    /// <summary>
    /// SDOT variant of the 4×8 panel GEMM. Requires <paramref name="meta"/> in the
    /// natural-column NEON layout (<c>Scales[s * 8 + c]</c>, mins at <c>+64</c>).
    /// Four activation rows × eight columns; the int8 dots accumulate per
    /// (row, column-pair) and scales apply once per 32-value sub-block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmNeon(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* meta)
    {
        int nb = n / Qk.SuperBlock;
        int tileGroups = Math.Max(1, ColTileBytes / (nb * Qk.Q4Kx8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            GemmTileNeon(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8, meta + g0 * nb);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmTileNeon(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* meta)
    {
        int nb = n / Qk.SuperBlock;
        Vector128<byte> nibble = Vector128.Create((byte)0x0F);
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ4Kx8* bPtr = weights + x * nb;
                BlockQ4Kx8Meta* mPtr = meta + x * nb;
                Vector128<float> acc0L = Vector128<float>.Zero;
                Vector128<float> acc0H = Vector128<float>.Zero;
                Vector128<float> acc1L = Vector128<float>.Zero;
                Vector128<float> acc1H = Vector128<float>.Zero;
                Vector128<float> acc2L = Vector128<float>.Zero;
                Vector128<float> acc2H = Vector128<float>.Zero;
                Vector128<float> acc3L = Vector128<float>.Zero;
                Vector128<float> acc3H = Vector128<float>.Zero;
                for (int b = 0; b < nb; b++)
                {
                    BlockQ4Kx8* wb = bPtr + b;
                    BlockQ8Kx4* ab = aPtr + b;
                    BlockQ4Kx8Meta* mb = mPtr + b;
                    Vector128<float> dL = Unsafe.ReadUnaligned<Vector128<float>>(mb->D);
                    Vector128<float> dH = Unsafe.ReadUnaligned<Vector128<float>>(mb->D + 4);
                    Vector128<float> dmL = Unsafe.ReadUnaligned<Vector128<float>>(mb->Dmin);
                    Vector128<float> dmH = Unsafe.ReadUnaligned<Vector128<float>>(mb->Dmin + 4);
                    Vector128<float> ad = Unsafe.ReadUnaligned<Vector128<float>>(ab->D);
                    Vector128<float> am0 = Vector128.Create(ad.GetElement(0));
                    Vector128<float> am1 = Vector128.Create(ad.GetElement(1));
                    Vector128<float> am2 = Vector128.Create(ad.GetElement(2));
                    Vector128<float> am3 = Vector128.Create(ad.GetElement(3));
                    byte* qs = wb->Qs;
                    sbyte* aq = ab->Qs;
                    for (int sb = 0; sb < 4; sb++)
                    {
                        Vector128<int> i00 = Vector128<int>.Zero;
                        Vector128<int> i01 = Vector128<int>.Zero;
                        Vector128<int> i02 = Vector128<int>.Zero;
                        Vector128<int> i03 = Vector128<int>.Zero;
                        Vector128<int> i10 = Vector128<int>.Zero;
                        Vector128<int> i11 = Vector128<int>.Zero;
                        Vector128<int> i12 = Vector128<int>.Zero;
                        Vector128<int> i13 = Vector128<int>.Zero;
                        Vector128<int> i20 = Vector128<int>.Zero;
                        Vector128<int> i21 = Vector128<int>.Zero;
                        Vector128<int> i22 = Vector128<int>.Zero;
                        Vector128<int> i23 = Vector128<int>.Zero;
                        Vector128<int> i30 = Vector128<int>.Zero;
                        Vector128<int> i31 = Vector128<int>.Zero;
                        Vector128<int> i32 = Vector128<int>.Zero;
                        Vector128<int> i33 = Vector128<int>.Zero;
                        for (int s = 0; s < 4; s++)
                        {
                            byte* q = qs + sb * 256 + s * 64;
                            Vector128<sbyte> w0 = (Neon.LoadU16(q) & nibble).AsSByte();
                            Vector128<sbyte> w1 = (Neon.LoadU16(q + 16) & nibble).AsSByte();
                            Vector128<sbyte> w2 = (Neon.LoadU16(q + 32) & nibble).AsSByte();
                            Vector128<sbyte> w3 = (Neon.LoadU16(q + 48) & nibble).AsSByte();
                            sbyte* a = aq + sb * 256 + s * 32;
                            Vector128<sbyte> av = Neon.Dup8(a);
                            i00 = Neon.Sdot(i00, w0, av);
                            i01 = Neon.Sdot(i01, w1, av);
                            i02 = Neon.Sdot(i02, w2, av);
                            i03 = Neon.Sdot(i03, w3, av);
                            av = Neon.Dup8(a + 8);
                            i10 = Neon.Sdot(i10, w0, av);
                            i11 = Neon.Sdot(i11, w1, av);
                            i12 = Neon.Sdot(i12, w2, av);
                            i13 = Neon.Sdot(i13, w3, av);
                            av = Neon.Dup8(a + 16);
                            i20 = Neon.Sdot(i20, w0, av);
                            i21 = Neon.Sdot(i21, w1, av);
                            i22 = Neon.Sdot(i22, w2, av);
                            i23 = Neon.Sdot(i23, w3, av);
                            av = Neon.Dup8(a + 24);
                            i30 = Neon.Sdot(i30, w0, av);
                            i31 = Neon.Sdot(i31, w1, av);
                            i32 = Neon.Sdot(i32, w2, av);
                            i33 = Neon.Sdot(i33, w3, av);
                        }

                        Vector128<short> scv = Unsafe.ReadUnaligned<Vector128<short>>(mb->Scales + sb * 16);
                        Vector128<int> scLo = AdvSimd.SignExtendWideningLower(scv.GetLower());
                        Vector128<int> scHi = AdvSimd.SignExtendWideningLower(scv.GetUpper());
                        acc0L = AdvSimd.FusedMultiplyAdd(acc0L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i00, i01), scLo)), AdvSimd.Multiply(dL, am0));
                        acc0H = AdvSimd.FusedMultiplyAdd(acc0H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i02, i03), scHi)), AdvSimd.Multiply(dH, am0));
                        acc1L = AdvSimd.FusedMultiplyAdd(acc1L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i10, i11), scLo)), AdvSimd.Multiply(dL, am1));
                        acc1H = AdvSimd.FusedMultiplyAdd(acc1H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i12, i13), scHi)), AdvSimd.Multiply(dH, am1));
                        acc2L = AdvSimd.FusedMultiplyAdd(acc2L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i20, i21), scLo)), AdvSimd.Multiply(dL, am2));
                        acc2H = AdvSimd.FusedMultiplyAdd(acc2H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i22, i23), scHi)), AdvSimd.Multiply(dH, am2));
                        acc3L = AdvSimd.FusedMultiplyAdd(acc3L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i30, i31), scLo)), AdvSimd.Multiply(dL, am3));
                        acc3H = AdvSimd.FusedMultiplyAdd(acc3H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i32, i33), scHi)), AdvSimd.Multiply(dH, am3));

                        i00 = Vector128<int>.Zero;
                        i01 = Vector128<int>.Zero;
                        i02 = Vector128<int>.Zero;
                        i03 = Vector128<int>.Zero;
                        i10 = Vector128<int>.Zero;
                        i11 = Vector128<int>.Zero;
                        i12 = Vector128<int>.Zero;
                        i13 = Vector128<int>.Zero;
                        i20 = Vector128<int>.Zero;
                        i21 = Vector128<int>.Zero;
                        i22 = Vector128<int>.Zero;
                        i23 = Vector128<int>.Zero;
                        i30 = Vector128<int>.Zero;
                        i31 = Vector128<int>.Zero;
                        i32 = Vector128<int>.Zero;
                        i33 = Vector128<int>.Zero;
                        for (int s = 0; s < 4; s++)
                        {
                            byte* q = qs + sb * 256 + s * 64;
                            Vector128<sbyte> w0 = (AdvSimd.ShiftRightLogical(Neon.LoadU16(q), 4) & nibble).AsSByte();
                            Vector128<sbyte> w1 = (AdvSimd.ShiftRightLogical(Neon.LoadU16(q + 16), 4) & nibble).AsSByte();
                            Vector128<sbyte> w2 = (AdvSimd.ShiftRightLogical(Neon.LoadU16(q + 32), 4) & nibble).AsSByte();
                            Vector128<sbyte> w3 = (AdvSimd.ShiftRightLogical(Neon.LoadU16(q + 48), 4) & nibble).AsSByte();
                            sbyte* a = aq + sb * 256 + 128 + s * 32;
                            Vector128<sbyte> av = Neon.Dup8(a);
                            i00 = Neon.Sdot(i00, w0, av);
                            i01 = Neon.Sdot(i01, w1, av);
                            i02 = Neon.Sdot(i02, w2, av);
                            i03 = Neon.Sdot(i03, w3, av);
                            av = Neon.Dup8(a + 8);
                            i10 = Neon.Sdot(i10, w0, av);
                            i11 = Neon.Sdot(i11, w1, av);
                            i12 = Neon.Sdot(i12, w2, av);
                            i13 = Neon.Sdot(i13, w3, av);
                            av = Neon.Dup8(a + 16);
                            i20 = Neon.Sdot(i20, w0, av);
                            i21 = Neon.Sdot(i21, w1, av);
                            i22 = Neon.Sdot(i22, w2, av);
                            i23 = Neon.Sdot(i23, w3, av);
                            av = Neon.Dup8(a + 24);
                            i30 = Neon.Sdot(i30, w0, av);
                            i31 = Neon.Sdot(i31, w1, av);
                            i32 = Neon.Sdot(i32, w2, av);
                            i33 = Neon.Sdot(i33, w3, av);
                        }

                        scv = Unsafe.ReadUnaligned<Vector128<short>>(mb->Scales + sb * 16 + 8);
                        scLo = AdvSimd.SignExtendWideningLower(scv.GetLower());
                        scHi = AdvSimd.SignExtendWideningLower(scv.GetUpper());
                        acc0L = AdvSimd.FusedMultiplyAdd(acc0L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i00, i01), scLo)), AdvSimd.Multiply(dL, am0));
                        acc0H = AdvSimd.FusedMultiplyAdd(acc0H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i02, i03), scHi)), AdvSimd.Multiply(dH, am0));
                        acc1L = AdvSimd.FusedMultiplyAdd(acc1L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i10, i11), scLo)), AdvSimd.Multiply(dL, am1));
                        acc1H = AdvSimd.FusedMultiplyAdd(acc1H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i12, i13), scHi)), AdvSimd.Multiply(dH, am1));
                        acc2L = AdvSimd.FusedMultiplyAdd(acc2L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i20, i21), scLo)), AdvSimd.Multiply(dL, am2));
                        acc2H = AdvSimd.FusedMultiplyAdd(acc2H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i22, i23), scHi)), AdvSimd.Multiply(dH, am2));
                        acc3L = AdvSimd.FusedMultiplyAdd(acc3L, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i30, i31), scLo)), AdvSimd.Multiply(dL, am3));
                        acc3H = AdvSimd.FusedMultiplyAdd(acc3H, AdvSimd.ConvertToSingle(AdvSimd.Multiply(Neon.PairAdd(i32, i33), scHi)), AdvSimd.Multiply(dH, am3));
                    }

                    AccumulateMinsNeon(ref acc0L, ref acc0H, ab, mb, 0, dmL, dmH, am0);
                    AccumulateMinsNeon(ref acc1L, ref acc1H, ab, mb, 1, dmL, dmH, am1);
                    AccumulateMinsNeon(ref acc2L, ref acc2H, ab, mb, 2, dmL, dmH, am2);
                    AccumulateMinsNeon(ref acc3L, ref acc3H, ab, mb, 3, dmL, dmH, am3);
                }

                float* row = dst + y * 4 * ldc + x * 8;
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

    /// <summary>
    /// min correction for one activation row: Σ_s bsum_s[row] · min_s[col] · dmin[col] · d[row],
    /// subtracted straight out of the float accumulators. Bsums pair offsets follow
    /// <c>Bsums[s * 8 + m * 4 − (s &amp; 1) * 6]</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateMinsNeon(ref Vector128<float> accL, ref Vector128<float> accH, BlockQ8Kx4* ab, BlockQ4Kx8Meta* mb, int m, Vector128<float> dmL, Vector128<float> dmH, Vector128<float> am)
    {
        Vector128<int> minL = Vector128<int>.Zero;
        Vector128<int> minH = Vector128<int>.Zero;
        short* bs = ab->Bsums + m * 4;
        for (int s = 0; s < 8; s++)
        {
            Vector128<short> mv = Unsafe.ReadUnaligned<Vector128<short>>(mb->Scales + 64 + s * 8);
            Vector64<short> pair = Vector64.Create((short)(bs[s * 8 - (s & 1) * 6] + bs[s * 8 - (s & 1) * 6 + 1]));
            minL = AdvSimd.MultiplyWideningLowerAndAdd(minL, pair, mv.GetLower());
            minH = AdvSimd.MultiplyWideningLowerAndAdd(minH, pair, mv.GetUpper());
        }

        accL = AdvSimd.Subtract(accL, AdvSimd.Multiply(AdvSimd.ConvertToSingle(minL), AdvSimd.Multiply(dmL, am)));
        accH = AdvSimd.Subtract(accH, AdvSimd.Multiply(AdvSimd.ConvertToSingle(minH), AdvSimd.Multiply(dmH, am)));
    }
}
