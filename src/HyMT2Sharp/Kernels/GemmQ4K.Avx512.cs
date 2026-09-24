using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe partial class GemmQ4K
{
    /// <summary>
    /// AVX-512 variant on the same <see cref="BlockQ4Kx8"/> panel and <see cref="BlockQ4Kx8Meta"/>
    /// sidecar as <see cref="GemmAvx2"/> — no repack fork. One zmm load covers the r0123+r4567
    /// pair, so one vpmaddubsw per row replaces two and activations broadcast straight to zmm
    /// (vpbroadcastq zmm). Meta scales stay in hadd order; a single vpermw expands each into
    /// the [s0×4, s1×4, s2×4, s3×4 | s4×4, …] lane order the 64-byte chunk needs. The doubled
    /// i32 accumulator folds with even/odd vpermd to natural [c0..c7], so the colPerm fixup
    /// disappears on both the scale and the store. Min correction is unchanged (ymm code,
    /// already natural order).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmAvx512(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* meta)
    {
        int nb = n / Qk.SuperBlock;
        int tileGroups = Math.Max(1, ColTileBytes / (nb * Qk.Q4Kx8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            GemmAvx512Tile(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8, meta + g0 * nb);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmAvx512Tile(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* metaBase)
    {
        int nb = n / Qk.SuperBlock;
        Vector512<byte> m4b = Vector512.Create((byte)0x0F);
        Vector512<short> scIdx = Q4MetaScaleIdx();
        Vector512<int> evenIdx = Q8_0.EvenLanes512();
        Vector512<int> oddIdx = Q8_0.OddLanes512();

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
                    Vector256<float> colScale = Avx.LoadVector256(meta[b].D);
                    float* ad = aPtr[b].D;
                    Vector256<float> cs0 = Avx.Multiply(colScale, Avx2.BroadcastScalarToVector256(ad + 0));
                    Vector256<float> cs1 = Avx.Multiply(colScale, Avx2.BroadcastScalarToVector256(ad + 1));
                    Vector256<float> cs2 = Avx.Multiply(colScale, Avx2.BroadcastScalarToVector256(ad + 2));
                    Vector256<float> cs3 = Avx.Multiply(colScale, Avx2.BroadcastScalarToVector256(ad + 3));

                    Vector512<int> j0 = Vector512<int>.Zero;
                    Vector512<int> j1 = Vector512<int>.Zero;
                    Vector512<int> j2 = Vector512<int>.Zero;
                    Vector512<int> j3 = Vector512<int>.Zero;
                    byte* qs = bPtr[b].Qs;
                    sbyte* aqs = aPtr[b].Qs;
                    short* sc = meta[b].Scales;
                    for (int sb = 0; sb < Qk.SuperBlock / 64; sb++)
                    {
                        byte* q = qs;
                        long* a = (long*)aqs;
                        Vector512<short> i0 = Vector512<short>.Zero;
                        Vector512<short> i1 = Vector512<short>.Zero;
                        Vector512<short> i2 = Vector512<short>.Zero;
                        Vector512<short> i3 = Vector512<short>.Zero;
                        for (int s = 0; s < 4; s++)
                        {
                            Vector512<byte> r = Avx512BW.And(Avx512F.LoadVector512(q), m4b);
                            i0 = Avx512BW.Add(i0, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[0]).AsSByte()));
                            i1 = Avx512BW.Add(i1, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[1]).AsSByte()));
                            i2 = Avx512BW.Add(i2, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[2]).AsSByte()));
                            i3 = Avx512BW.Add(i3, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[3]).AsSByte()));
                            q += 64;
                            a += 4;
                        }

                        Vector256<short> scLo = Avx.LoadVector256(sc);
                        Vector512<short> scLo512 = Avx512BW.PermuteVar32x16(Vector512.Create(scLo, scLo), scIdx);
                        j0 = Avx512F.Add(j0, Avx512BW.MultiplyAddAdjacent(i0, scLo512));
                        j1 = Avx512F.Add(j1, Avx512BW.MultiplyAddAdjacent(i1, scLo512));
                        j2 = Avx512F.Add(j2, Avx512BW.MultiplyAddAdjacent(i2, scLo512));
                        j3 = Avx512F.Add(j3, Avx512BW.MultiplyAddAdjacent(i3, scLo512));

                        q = qs;
                        i0 = Vector512<short>.Zero;
                        i1 = Vector512<short>.Zero;
                        i2 = Vector512<short>.Zero;
                        i3 = Vector512<short>.Zero;
                        for (int s = 0; s < 4; s++)
                        {
                            Vector512<byte> r = Avx512BW.And(
                                Avx512BW.ShiftRightLogical(Avx512F.LoadVector512(q).AsUInt16(), 4).AsByte(), m4b);
                            i0 = Avx512BW.Add(i0, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[0]).AsSByte()));
                            i1 = Avx512BW.Add(i1, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[1]).AsSByte()));
                            i2 = Avx512BW.Add(i2, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[2]).AsSByte()));
                            i3 = Avx512BW.Add(i3, Avx512BW.MultiplyAddAdjacent(r, Vector512.Create(a[3]).AsSByte()));
                            q += 64;
                            a += 4;
                        }

                        Vector256<short> scHi = Avx.LoadVector256(sc + 16);
                        Vector512<short> scHi512 = Avx512BW.PermuteVar32x16(Vector512.Create(scHi, scHi), scIdx);
                        j0 = Avx512F.Add(j0, Avx512BW.MultiplyAddAdjacent(i0, scHi512));
                        j1 = Avx512F.Add(j1, Avx512BW.MultiplyAddAdjacent(i1, scHi512));
                        j2 = Avx512F.Add(j2, Avx512BW.MultiplyAddAdjacent(i2, scHi512));
                        j3 = Avx512F.Add(j3, Avx512BW.MultiplyAddAdjacent(i3, scHi512));

                        qs += 256;
                        aqs += 256;
                        sc += 48;
                    }

                    acc0 = Fmadd(acc0, Avx.ConvertToVector256Single(FoldDoubled(j0, evenIdx, oddIdx)), cs0);
                    acc1 = Fmadd(acc1, Avx.ConvertToVector256Single(FoldDoubled(j1, evenIdx, oddIdx)), cs1);
                    acc2 = Fmadd(acc2, Avx.ConvertToVector256Single(FoldDoubled(j2, evenIdx, oddIdx)), cs2);
                    acc3 = Fmadd(acc3, Avx.ConvertToVector256Single(FoldDoubled(j3, evenIdx, oddIdx)), cs3);

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
                    min0 = Fmadd(min0, Avx.ConvertToVector256Single(m0), Avx.Multiply(colDmin, Avx2.BroadcastScalarToVector256(ad + 0)));
                    min1 = Fmadd(min1, Avx.ConvertToVector256Single(m1), Avx.Multiply(colDmin, Avx2.BroadcastScalarToVector256(ad + 1)));
                    min2 = Fmadd(min2, Avx.ConvertToVector256Single(m2), Avx.Multiply(colDmin, Avx2.BroadcastScalarToVector256(ad + 2)));
                    min3 = Fmadd(min3, Avx.ConvertToVector256Single(m3), Avx.Multiply(colDmin, Avx2.BroadcastScalarToVector256(ad + 3)));
                }

                Avx.Store(dst + (y * 4 + 0) * ldc + x * 8, Avx.Subtract(acc0, min0));
                Avx.Store(dst + (y * 4 + 1) * ldc + x * 8, Avx.Subtract(acc1, min1));
                Avx.Store(dst + (y * 4 + 2) * ldc + x * 8, Avx.Subtract(acc2, min2));
                Avx.Store(dst + (y * 4 + 3) * ldc + x * 8, Avx.Subtract(acc3, min3));
            }
        }
    }

    /// <summary>Fold doubled-column dwords [c0,c0,c1,c1,…,c7,c7] to natural [c0..c7].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> FoldDoubled(Vector512<int> j, Vector512<int> evenIdx, Vector512<int> oddIdx) =>
        Avx2.Add(Avx512F.PermuteVar16x32(j, evenIdx).GetLower(), Avx512F.PermuteVar16x32(j, oddIdx).GetLower());

    /// <summary>
    /// vpermw indices turning meta's hadd order [s0 s0 s1 s1 s4 s4 s5 s5 | s2 s2 s3 s3 s6 s6 s7 s7]
    /// into the [s0×4, s1×4, s2×4, s3×4 | s4×4, …] order the 64-byte chunk's s16 lanes take.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<short> Q4MetaScaleIdx()
    {
        ReadOnlySpan<short> idx =
        [
            0, 0, 0, 0, 2, 2, 2, 2, 8, 8, 8, 8, 10, 10, 10, 10,
            4, 4, 4, 4, 6, 6, 6, 6, 12, 12, 12, 12, 14, 14, 14, 14,
        ];
        return Vector512.Create(idx);
    }
}
