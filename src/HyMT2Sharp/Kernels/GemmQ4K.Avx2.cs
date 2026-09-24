using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe partial class GemmQ4K
{
    /// <summary>
    /// Column tile (bytes of q4_Kx8 panel) processed against all row groups before moving on,
    /// so the weight slice stays L2-resident while q8 rows stream through.
    /// </summary>
    public static int ColTileBytes = 128 * 1024;

    /// <summary>
    /// 4×8 Q4_K panel GEMM on the llama.cpp <c>q4_Kx8</c> / <c>q8_Kx4</c> layouts.
    /// Unlike <c>ggml_gemm_q4_K_8x8_q8_K</c>, the inner loop has no shuffle-port work:
    /// each activation row is <c>vpbroadcastq</c>'d (a pure load) against the raw
    /// column-interleaved nibbles, so per 8 <c>vpmaddubsw</c> there are only 8 adds,
    /// 2 ands and 6 loads. Scales come pre-decoded from <see cref="BlockQ4Kx8Meta"/>.
    /// Column order comes out as [0 1 4 5 2 3 6 7] and is fixed once per tile on store.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmAvx2(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* meta)
    {
        int nb = n / Qk.SuperBlock;
        int tileGroups = Math.Max(1, ColTileBytes / (nb * Qk.Q4Kx8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            GemmAvx2Tile(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8, meta + g0 * nb);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmAvx2Tile(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* metaBase)
    {
        int nb = n / Qk.SuperBlock;
        Vector256<byte> m4b = Vector256.Create((byte)0x0F);
        // vphaddw(cols0123, cols4567) yields [0 1 4 5 | 2 3 6 7]; the permutation is its own inverse.
        Vector256<int> colPerm = Vector256.Create(0, 1, 4, 5, 2, 3, 6, 7);

        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ4Kx8* bPtr = weights + x * nb;
                BlockQ4Kx8Meta* meta = metaBase + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;
                Vector256<float> min0 = Vector256<float>.Zero;
                Vector256<float> min1 = Vector256<float>.Zero;
                Vector256<float> min2 = Vector256<float>.Zero;
                Vector256<float> min3 = Vector256<float>.Zero;

                for (int b = 0; b < nb; b++)
                {
                    Vector256<float> colScale = Avx2.PermuteVar8x32(Avx.LoadVector256(meta[b].D), colPerm);
                    Vector128<float> rowScale4 = Avx.LoadVector128(aPtr[b].D);
                    Vector256<float> rowScale = Vector256.Create(rowScale4, rowScale4);
                    Vector256<float> cs0 = Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 0));
                    Vector256<float> cs1 = Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 85));
                    Vector256<float> cs2 = Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 170));
                    Vector256<float> cs3 = Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 255));

                    byte* qs = bPtr[b].Qs;
                    sbyte* aqs = aPtr[b].Qs;
                    short* sc = meta[b].Scales;
                    for (int sb = 0; sb < Qk.SuperBlock / 64; sb++)
                    {
                        // Low nibble = sub-block 2*sb. First step seeds the accumulators (no zero+add).
                        byte* q = qs;
                        long* a = (long*)aqs;
                        Vector256<byte> r0123 = Avx2.And(Avx.LoadVector256(q), m4b);
                        Vector256<byte> r4567 = Avx2.And(Avx.LoadVector256(q + 32), m4b);
                        Vector256<sbyte> l0 = Avx2.BroadcastScalarToVector256(a).AsSByte();
                        Vector256<short> i0a = Avx2.MultiplyAddAdjacent(r0123, l0);
                        Vector256<short> i0b = Avx2.MultiplyAddAdjacent(r4567, l0);
                        Vector256<sbyte> l1 = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                        Vector256<short> i1a = Avx2.MultiplyAddAdjacent(r0123, l1);
                        Vector256<short> i1b = Avx2.MultiplyAddAdjacent(r4567, l1);
                        Vector256<sbyte> l2 = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                        Vector256<short> i2a = Avx2.MultiplyAddAdjacent(r0123, l2);
                        Vector256<short> i2b = Avx2.MultiplyAddAdjacent(r4567, l2);
                        Vector256<sbyte> l3 = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                        Vector256<short> i3a = Avx2.MultiplyAddAdjacent(r0123, l3);
                        Vector256<short> i3b = Avx2.MultiplyAddAdjacent(r4567, l3);
                        q += 64;
                        a += 4;
                        for (int s = 1; s < 4; s++)
                        {
                            r0123 = Avx2.And(Avx.LoadVector256(q), m4b);
                            r4567 = Avx2.And(Avx.LoadVector256(q + 32), m4b);
                            l0 = Avx2.BroadcastScalarToVector256(a).AsSByte();
                            i0a = Avx2.Add(i0a, Avx2.MultiplyAddAdjacent(r0123, l0));
                            i0b = Avx2.Add(i0b, Avx2.MultiplyAddAdjacent(r4567, l0));
                            l1 = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                            i1a = Avx2.Add(i1a, Avx2.MultiplyAddAdjacent(r0123, l1));
                            i1b = Avx2.Add(i1b, Avx2.MultiplyAddAdjacent(r4567, l1));
                            l2 = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                            i2a = Avx2.Add(i2a, Avx2.MultiplyAddAdjacent(r0123, l2));
                            i2b = Avx2.Add(i2b, Avx2.MultiplyAddAdjacent(r4567, l2));
                            l3 = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                            i3a = Avx2.Add(i3a, Avx2.MultiplyAddAdjacent(r0123, l3));
                            i3b = Avx2.Add(i3b, Avx2.MultiplyAddAdjacent(r4567, l3));
                            q += 64;
                            a += 4;
                        }

                        Vector256<short> scLo = Avx.LoadVector256(sc);
                        acc0 = Fmadd(acc0, RowSum(i0a, i0b, scLo), cs0);
                        acc1 = Fmadd(acc1, RowSum(i1a, i1b, scLo), cs1);
                        acc2 = Fmadd(acc2, RowSum(i2a, i2b, scLo), cs2);
                        acc3 = Fmadd(acc3, RowSum(i3a, i3b, scLo), cs3);

                        // High nibble = sub-block 2*sb+1.
                        q = qs;
                        r0123 = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(q).AsUInt16(), 4).AsByte(), m4b);
                        r4567 = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(q + 32).AsUInt16(), 4).AsByte(), m4b);
                        l0 = Avx2.BroadcastScalarToVector256(a).AsSByte();
                        i0a = Avx2.MultiplyAddAdjacent(r0123, l0);
                        i0b = Avx2.MultiplyAddAdjacent(r4567, l0);
                        l1 = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                        i1a = Avx2.MultiplyAddAdjacent(r0123, l1);
                        i1b = Avx2.MultiplyAddAdjacent(r4567, l1);
                        l2 = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                        i2a = Avx2.MultiplyAddAdjacent(r0123, l2);
                        i2b = Avx2.MultiplyAddAdjacent(r4567, l2);
                        l3 = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                        i3a = Avx2.MultiplyAddAdjacent(r0123, l3);
                        i3b = Avx2.MultiplyAddAdjacent(r4567, l3);
                        q += 64;
                        a += 4;
                        for (int s = 1; s < 4; s++)
                        {
                            r0123 = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(q).AsUInt16(), 4).AsByte(), m4b);
                            r4567 = Avx2.And(Avx2.ShiftRightLogical(Avx.LoadVector256(q + 32).AsUInt16(), 4).AsByte(), m4b);
                            l0 = Avx2.BroadcastScalarToVector256(a).AsSByte();
                            i0a = Avx2.Add(i0a, Avx2.MultiplyAddAdjacent(r0123, l0));
                            i0b = Avx2.Add(i0b, Avx2.MultiplyAddAdjacent(r4567, l0));
                            l1 = Avx2.BroadcastScalarToVector256(a + 1).AsSByte();
                            i1a = Avx2.Add(i1a, Avx2.MultiplyAddAdjacent(r0123, l1));
                            i1b = Avx2.Add(i1b, Avx2.MultiplyAddAdjacent(r4567, l1));
                            l2 = Avx2.BroadcastScalarToVector256(a + 2).AsSByte();
                            i2a = Avx2.Add(i2a, Avx2.MultiplyAddAdjacent(r0123, l2));
                            i2b = Avx2.Add(i2b, Avx2.MultiplyAddAdjacent(r4567, l2));
                            l3 = Avx2.BroadcastScalarToVector256(a + 3).AsSByte();
                            i3a = Avx2.Add(i3a, Avx2.MultiplyAddAdjacent(r0123, l3));
                            i3b = Avx2.Add(i3b, Avx2.MultiplyAddAdjacent(r4567, l3));
                            q += 64;
                            a += 4;
                        }

                        Vector256<short> scHi = Avx.LoadVector256(sc + 16);
                        acc0 = Fmadd(acc0, RowSum(i0a, i0b, scHi), cs0);
                        acc1 = Fmadd(acc1, RowSum(i1a, i1b, scHi), cs1);
                        acc2 = Fmadd(acc2, RowSum(i2a, i2b, scHi), cs2);
                        acc3 = Fmadd(acc3, RowSum(i3a, i3b, scHi), cs3);

                        qs += 256;
                        aqs += 256;
                        sc += 48;
                    }

                    // dmin compensation once per super-block: Σ_pairs bsums(row, pair) · mins(col), natural column order.
                    Vector256<int> m0 = Vector256<int>.Zero;
                    Vector256<int> m1 = Vector256<int>.Zero;
                    Vector256<int> m2 = Vector256<int>.Zero;
                    Vector256<int> m3 = Vector256<int>.Zero;
                    short* bs = aPtr[b].Bsums;
                    short* mins = meta[b].Scales + 32;
                    for (int p = 0; p < 4; p++)
                    {
                        Vector256<short> bsums = Avx.LoadVector256(bs);
                        Vector128<short> bsumH = Ssse3.HorizontalAdd(bsums.GetLower(), bsums.GetUpper());
                        Vector256<short> bsumDup = Vector256.Create(bsumH, bsumH);
                        Vector256<short> mins01 = Avx.LoadVector256(mins);
                        m0 = Avx2.Add(m0, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(bsumDup.AsInt32(), 0).AsInt16(), mins01));
                        m1 = Avx2.Add(m1, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(bsumDup.AsInt32(), 85).AsInt16(), mins01));
                        m2 = Avx2.Add(m2, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(bsumDup.AsInt32(), 170).AsInt16(), mins01));
                        m3 = Avx2.Add(m3, Avx2.MultiplyAddAdjacent(Avx2.Shuffle(bsumDup.AsInt32(), 255).AsInt16(), mins01));
                        bs += 16;
                        mins += 48;
                    }

                    Vector256<float> colDmin = Avx.LoadVector256(meta[b].Dmin);
                    min0 = Fmadd(min0, Avx.ConvertToVector256Single(m0), Avx.Multiply(colDmin, Avx.Shuffle(rowScale, rowScale, 0)));
                    min1 = Fmadd(min1, Avx.ConvertToVector256Single(m1), Avx.Multiply(colDmin, Avx.Shuffle(rowScale, rowScale, 85)));
                    min2 = Fmadd(min2, Avx.ConvertToVector256Single(m2), Avx.Multiply(colDmin, Avx.Shuffle(rowScale, rowScale, 170)));
                    min3 = Fmadd(min3, Avx.ConvertToVector256Single(m3), Avx.Multiply(colDmin, Avx.Shuffle(rowScale, rowScale, 255)));
                }

                Avx.Store(dst + (y * 4 + 0) * ldc + x * 8, Avx.Subtract(Avx2.PermuteVar8x32(acc0, colPerm), min0));
                Avx.Store(dst + (y * 4 + 1) * ldc + x * 8, Avx.Subtract(Avx2.PermuteVar8x32(acc1, colPerm), min1));
                Avx.Store(dst + (y * 4 + 2) * ldc + x * 8, Avx.Subtract(Avx2.PermuteVar8x32(acc2, colPerm), min2));
                Avx.Store(dst + (y * 4 + 3) * ldc + x * 8, Avx.Subtract(Avx2.PermuteVar8x32(acc3, colPerm), min3));
            }
        }
    }

    /// <summary>
    /// Fold the two half-row short accumulators into one f32 vector of per-column dots,
    /// [c0 c1 c4 c5 | c2 c3 c6 c7]. Each short is ≤ 8·15·127, so the pair sum fits int16.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> RowSum(Vector256<short> a0123, Vector256<short> a4567, Vector256<short> scales) =>
        Avx.ConvertToVector256Single(Avx2.MultiplyAddAdjacent(Avx2.HorizontalAdd(a0123, a4567), scales));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Fmadd(Vector256<float> acc, Vector256<float> a, Vector256<float> b) =>
        Simd.UseFma ? Fma.MultiplyAdd(a, b, acc) : Avx.Add(acc, Avx.Multiply(a, b));
}
