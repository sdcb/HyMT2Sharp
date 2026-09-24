using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// 4×8 Q6_K panel GEMM on <see cref="BlockQ6Kx8"/> (u8 values, blocklen 4) × <see cref="BlockQ8Kx4"/>.
/// Per 8 K values: 2 weight loads, 8 <c>vpbroadcastd</c>, 8 <c>vpmaddubsw</c>, 4 adds,
/// 4 <c>vpmaddwd</c> (16-value scales) into 4 int32 row accumulators; the −32 offset is
/// folded in once per super-block from the q8 bsums.
/// </summary>
public static unsafe class GemmQ6K
{
    /// <summary>
    /// Packed-panel GEMM for the active dispatch tier. <paramref name="weights"/> must be
    /// produced by the matching repacker: <see cref="RepackQ6K.RowsNeon"/> when
    /// <see cref="Simd.UseDp"/> (ARM64, signed NEON layout) or <see cref="RepackQ6K.Rows"/>
    /// otherwise (canonical layout). Feeding a canonical <c>Rows</c> panel on ARM64 reads
    /// unsigned codes as signed and yields wrong results.
    /// </summary>
    public static void Gemm8x8(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        if ((rows & 3) != 0)
            throw new ArgumentException("rows must be a multiple of 4.", nameof(rows));
        if ((cols & 7) != 0)
            throw new ArgumentException("cols must be a multiple of 8.", nameof(cols));
        if (Simd.UseAvx2)
            GemmAvx2(n, dst, ldc, weights, activations, rows, cols);
        else if (Simd.UseDp)
            GemmNeon(n, dst, ldc, weights, activations, rows, cols);
        else
            GemmScalar(n, dst, ldc, weights, activations, rows, cols);
    }

    public static void GemmScalar(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ6Kx8* bPtr = weights + x * nb;
                for (int r = 0; r < 4; r++)
                {
                    for (int c = 0; c < 8; c++)
                    {
                        float sum = 0;
                        for (int b = 0; b < nb; b++)
                        {
                            int acc = 0;
                            for (int i = 0; i < 16; i++)
                            {
                                int sc = bPtr[b].Scales[i * 16 + c * 2];
                                int part = 0;
                                for (int t = 0; t < 16; t++)
                                {
                                    int k = i * 16 + t;
                                    int w = bPtr[b].Qs[(k >> 2) * 32 + c * 4 + (k & 3)] - 32;
                                    int a = aPtr[b].Qs[(k >> 3) * 32 + r * 8 + (k & 7)];
                                    part += w * a;
                                }

                                acc += sc * part;
                            }

                            sum += bPtr[b].D[c] * aPtr[b].D[r] * acc;
                        }

                        dst[(y * 4 + r) * ldc + x * 8 + c] = sum;
                    }
                }
            }
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmAvx2(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        // Same L2 column tiling as the Q4 panel; a Q6 group is ~2.5× larger.
        int tileGroups = Math.Max(1, 2 * GemmQ4K.ColTileBytes / (nb * Qk.Q6Kx8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            GemmAvx2Tile(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmAvx2Tile(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ6Kx8* bPtr = weights + x * nb;
                Vector256<float> acc0 = Vector256<float>.Zero;
                Vector256<float> acc1 = Vector256<float>.Zero;
                Vector256<float> acc2 = Vector256<float>.Zero;
                Vector256<float> acc3 = Vector256<float>.Zero;

                for (int b = 0; b < nb; b++)
                {
                    Vector256<int> j0 = Vector256<int>.Zero;
                    Vector256<int> j1 = Vector256<int>.Zero;
                    Vector256<int> j2 = Vector256<int>.Zero;
                    Vector256<int> j3 = Vector256<int>.Zero;
                    byte* q = bPtr[b].Qs;
                    int* a = (int*)aPtr[b].Qs;
                    short* sc = bPtr[b].Scales;
                    for (int i = 0; i < Qk.SuperBlock / 16; i++)
                    {
                        Vector256<short> scv = Avx.LoadVector256(sc);
                        // Two 8-value steps share one 16-value scale.
                        Vector256<byte> wa = Avx.LoadVector256(q);
                        Vector256<byte> wb = Avx.LoadVector256(q + 32);
                        j0 = Avx2.Add(j0, Accumulate8(wa, wb, a, scv));
                        j1 = Avx2.Add(j1, Accumulate8(wa, wb, a + 2, scv));
                        j2 = Avx2.Add(j2, Accumulate8(wa, wb, a + 4, scv));
                        j3 = Avx2.Add(j3, Accumulate8(wa, wb, a + 6, scv));
                        wa = Avx.LoadVector256(q + 64);
                        wb = Avx.LoadVector256(q + 96);
                        j0 = Avx2.Add(j0, Accumulate8(wa, wb, a + 8, scv));
                        j1 = Avx2.Add(j1, Accumulate8(wa, wb, a + 10, scv));
                        j2 = Avx2.Add(j2, Accumulate8(wa, wb, a + 12, scv));
                        j3 = Avx2.Add(j3, Accumulate8(wa, wb, a + 14, scv));
                        q += 128;
                        a += 16;
                        sc += 16;
                    }

                    // −32 · Σ_i sc_i · bsum_i, two sub-blocks per vpmaddwd.
                    int* bs = (int*)aPtr[b].Bsums;
                    short* pairs = bPtr[b].ScalePairs;
                    Vector256<int> c0 = Vector256<int>.Zero;
                    Vector256<int> c1 = Vector256<int>.Zero;
                    Vector256<int> c2 = Vector256<int>.Zero;
                    Vector256<int> c3 = Vector256<int>.Zero;
                    for (int p = 0; p < 4; p++)
                    {
                        // Bsums quarter p: [r0: b0 b1 b2 b3][r1: …][r2: …][r3: …] as shorts → 2 dwords per row.
                        Vector256<short> pairA = Avx.LoadVector256(pairs);
                        Vector256<short> pairB = Avx.LoadVector256(pairs + 16);
                        c0 = Avx2.Add(c0, Avx2.Add(Avx2.MultiplyAddAdjacent(pairA, Avx2.BroadcastScalarToVector256(bs + 0).AsInt16()), Avx2.MultiplyAddAdjacent(pairB, Avx2.BroadcastScalarToVector256(bs + 1).AsInt16())));
                        c1 = Avx2.Add(c1, Avx2.Add(Avx2.MultiplyAddAdjacent(pairA, Avx2.BroadcastScalarToVector256(bs + 2).AsInt16()), Avx2.MultiplyAddAdjacent(pairB, Avx2.BroadcastScalarToVector256(bs + 3).AsInt16())));
                        c2 = Avx2.Add(c2, Avx2.Add(Avx2.MultiplyAddAdjacent(pairA, Avx2.BroadcastScalarToVector256(bs + 4).AsInt16()), Avx2.MultiplyAddAdjacent(pairB, Avx2.BroadcastScalarToVector256(bs + 5).AsInt16())));
                        c3 = Avx2.Add(c3, Avx2.Add(Avx2.MultiplyAddAdjacent(pairA, Avx2.BroadcastScalarToVector256(bs + 6).AsInt16()), Avx2.MultiplyAddAdjacent(pairB, Avx2.BroadcastScalarToVector256(bs + 7).AsInt16())));
                        bs += 8;
                        pairs += 32;
                    }

                    j0 = Avx2.Subtract(j0, Avx2.ShiftLeftLogical(c0, 5));
                    j1 = Avx2.Subtract(j1, Avx2.ShiftLeftLogical(c1, 5));
                    j2 = Avx2.Subtract(j2, Avx2.ShiftLeftLogical(c2, 5));
                    j3 = Avx2.Subtract(j3, Avx2.ShiftLeftLogical(c3, 5));

                    Vector256<float> colScale = Avx.LoadVector256(bPtr[b].D);
                    Vector128<float> rowScale4 = Avx.LoadVector128(aPtr[b].D);
                    Vector256<float> rowScale = Vector256.Create(rowScale4, rowScale4);
                    acc0 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j0), Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 0)), acc0);
                    acc1 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j1), Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 85)), acc1);
                    acc2 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j2), Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 170)), acc2);
                    acc3 = Fma.MultiplyAdd(Avx.ConvertToVector256Single(j3), Avx.Multiply(colScale, Avx.Shuffle(rowScale, rowScale, 255)), acc3);
                }

                Avx.Store(dst + (y * 4 + 0) * ldc + x * 8, acc0);
                Avx.Store(dst + (y * 4 + 1) * ldc + x * 8, acc1);
                Avx.Store(dst + (y * 4 + 2) * ldc + x * 8, acc2);
                Avx.Store(dst + (y * 4 + 3) * ldc + x * 8, acc3);
            }
        }
    }

    /// <summary>
    /// 8 K values × 8 columns. Scales are duplicated shorts [s0 s0 s1 s1 …].
    /// vpmaddubsw beats vpdpbusd here: the VNNI variant pays an extra
    /// vpmulld+shuffle to scale lanes separately (slower on Raptor Lake).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Accumulate8(Vector256<byte> wa, Vector256<byte> wb, int* a, Vector256<short> scv) =>
        Dot8ScaledAvx2(wa, wb, a, scv);

    /// <summary>AVX2 reference: shorts [c0 c0 c1 c1 …], each ≤ 4·63·127.</summary>
    public static Vector256<int> Dot8ScaledAvx2(Vector256<byte> wa, Vector256<byte> wb, int* a, Vector256<short> scv) =>
        Avx2.MultiplyAddAdjacent(Dot8(wa, wb, a), scv);

    public static Vector256<int> Dot8ScaledVnni(Vector256<byte> wa, Vector256<byte> wb, int* a, Vector256<short> scv)
    {
        Vector256<int> dots = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, wa, Avx2.BroadcastScalarToVector256(a).AsSByte());
        dots = AvxVnni.MultiplyWideningAndAdd(dots, wb, Avx2.BroadcastScalarToVector256(a + 1).AsSByte());
        return Avx2.MultiplyLow(dots, EvenScales(scv));
    }

    /// <summary>
    /// SDOT variant for <see cref="RepackQ6K.RowsNeon"/> panels (signed weights, natural
    /// scale order). Sixteen <c>sdot</c> per (row, 16-value group); the −32 offset is
    /// already folded into the weight bytes, so no bsum correction runs at all.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void GemmNeon(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        int tileGroups = Math.Max(1, 2 * GemmQ4K.ColTileBytes / (nb * Qk.Q6Kx8Size));
        int groups = cols / 8;
        for (int g0 = 0; g0 < groups; g0 += tileGroups)
        {
            int g1 = Math.Min(groups, g0 + tileGroups);
            GemmNeonTile(n, dst + g0 * 8, ldc, weights + g0 * nb, activations, rows, (g1 - g0) * 8);
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmNeonTile(int n, float* dst, int ldc, BlockQ6Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ6Kx8* bPtr = weights + x * nb;
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
                    Vector128<int> j0L = Vector128<int>.Zero;
                    Vector128<int> j0H = Vector128<int>.Zero;
                    Vector128<int> j1L = Vector128<int>.Zero;
                    Vector128<int> j1H = Vector128<int>.Zero;
                    Vector128<int> j2L = Vector128<int>.Zero;
                    Vector128<int> j2H = Vector128<int>.Zero;
                    Vector128<int> j3L = Vector128<int>.Zero;
                    Vector128<int> j3H = Vector128<int>.Zero;
                    sbyte* q = (sbyte*)bPtr[b].Qs;
                    sbyte* a = aPtr[b].Qs;
                    short* sc = bPtr[b].Scales;
                    Vector128<float> dL = Unsafe.ReadUnaligned<Vector128<float>>(bPtr[b].D);
                    Vector128<float> dH = Unsafe.ReadUnaligned<Vector128<float>>(bPtr[b].D + 4);
                    Vector128<float> ad = Unsafe.ReadUnaligned<Vector128<float>>(aPtr[b].D);
                    for (int i = 0; i < Qk.SuperBlock / 16; i++)
                    {
                        Vector128<short> scv = Unsafe.ReadUnaligned<Vector128<short>>(sc + i * 8);
                        Vector128<int> scL = AdvSimd.SignExtendWideningLower(scv.GetLower());
                        Vector128<int> scH = AdvSimd.SignExtendWideningLower(scv.GetUpper());
                        Vector128<sbyte> w00 = Neon.Load16(q);
                        Vector128<sbyte> w01 = Neon.Load16(q + 16);
                        Vector128<sbyte> w10 = Neon.Load16(q + 32);
                        Vector128<sbyte> w11 = Neon.Load16(q + 48);
                        Vector128<sbyte> w20 = Neon.Load16(q + 64);
                        Vector128<sbyte> w21 = Neon.Load16(q + 80);
                        Vector128<sbyte> w30 = Neon.Load16(q + 96);
                        Vector128<sbyte> w31 = Neon.Load16(q + 112);
                        sbyte* ar = a + i * 64;
                        AccumulateQ6Neon(ref j0L, ref j0H, w00, w01, w10, w11, w20, w21, w30, w31, ar, scL, scH);
                        AccumulateQ6Neon(ref j1L, ref j1H, w00, w01, w10, w11, w20, w21, w30, w31, ar + 8, scL, scH);
                        AccumulateQ6Neon(ref j2L, ref j2H, w00, w01, w10, w11, w20, w21, w30, w31, ar + 16, scL, scH);
                        AccumulateQ6Neon(ref j3L, ref j3H, w00, w01, w10, w11, w20, w21, w30, w31, ar + 24, scL, scH);
                        q += 128;
                    }

                    acc0L = AdvSimd.FusedMultiplyAdd(acc0L, AdvSimd.ConvertToSingle(j0L), AdvSimd.Multiply(dL, Vector128.Create(ad.GetElement(0))));
                    acc0H = AdvSimd.FusedMultiplyAdd(acc0H, AdvSimd.ConvertToSingle(j0H), AdvSimd.Multiply(dH, Vector128.Create(ad.GetElement(0))));
                    acc1L = AdvSimd.FusedMultiplyAdd(acc1L, AdvSimd.ConvertToSingle(j1L), AdvSimd.Multiply(dL, Vector128.Create(ad.GetElement(1))));
                    acc1H = AdvSimd.FusedMultiplyAdd(acc1H, AdvSimd.ConvertToSingle(j1H), AdvSimd.Multiply(dH, Vector128.Create(ad.GetElement(1))));
                    acc2L = AdvSimd.FusedMultiplyAdd(acc2L, AdvSimd.ConvertToSingle(j2L), AdvSimd.Multiply(dL, Vector128.Create(ad.GetElement(2))));
                    acc2H = AdvSimd.FusedMultiplyAdd(acc2H, AdvSimd.ConvertToSingle(j2H), AdvSimd.Multiply(dH, Vector128.Create(ad.GetElement(2))));
                    acc3L = AdvSimd.FusedMultiplyAdd(acc3L, AdvSimd.ConvertToSingle(j3L), AdvSimd.Multiply(dL, Vector128.Create(ad.GetElement(3))));
                    acc3H = AdvSimd.FusedMultiplyAdd(acc3H, AdvSimd.ConvertToSingle(j3H), AdvSimd.Multiply(dH, Vector128.Create(ad.GetElement(3))));
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

    /// <summary>16 K values × 8 columns for one activation row: 8 <c>sdot</c>, 2 widening muls.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateQ6Neon(ref Vector128<int> jL, ref Vector128<int> jH,
        Vector128<sbyte> w00, Vector128<sbyte> w01, Vector128<sbyte> w10, Vector128<sbyte> w11,
        Vector128<sbyte> w20, Vector128<sbyte> w21, Vector128<sbyte> w30, Vector128<sbyte> w31,
        sbyte* ar, Vector128<int> scL, Vector128<int> scH)
    {
        Vector128<sbyte> a0 = Neon.Dup4(ar);
        Vector128<sbyte> a1 = Neon.Dup4(ar + 4);
        Vector128<sbyte> a2 = Neon.Dup4(ar + 32);
        Vector128<sbyte> a3 = Neon.Dup4(ar + 36);
        Vector128<int> dL = Neon.Sdot(Vector128<int>.Zero, w00, a0);
        dL = Neon.Sdot(dL, w10, a1);
        dL = Neon.Sdot(dL, w20, a2);
        dL = Neon.Sdot(dL, w30, a3);
        Vector128<int> dH = Neon.Sdot(Vector128<int>.Zero, w01, a0);
        dH = Neon.Sdot(dH, w11, a1);
        dH = Neon.Sdot(dH, w21, a2);
        dH = Neon.Sdot(dH, w31, a3);
        jL = AdvSimd.Add(jL, AdvSimd.Multiply(dL, scL));
        jH = AdvSimd.Add(jH, AdvSimd.Multiply(dH, scH));
    }

    /// <summary>
    /// 8 K values × 8 columns for one activation row: shorts [c0 c0 c1 c1 … c7 c7],
    /// each ≤ 4·63·127 so the pair sum fits int16.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<short> Dot8(Vector256<byte> wa, Vector256<byte> wb, int* a) =>
        Avx2.Add(
            Avx2.MultiplyAddAdjacent(wa, Avx2.BroadcastScalarToVector256(a).AsSByte()),
            Avx2.MultiplyAddAdjacent(wb, Avx2.BroadcastScalarToVector256(a + 1).AsSByte()));

    /// <summary>Duplicated shorts [s0 s0 s1 s1 … s7 s7] to int32 [s0 s1 … s7].</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> EvenScales(Vector256<short> scv)
    {
        Vector256<byte> shuf = Vector256.Create(
            (byte)0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0,
            (byte)0, 1, 4, 5, 8, 9, 12, 13, 0, 0, 0, 0, 0, 0, 0, 0);
        Vector256<short> packed = Avx2.Shuffle(scv.AsByte(), shuf).AsInt16();
        return Vector256.Create(
            Sse41.ConvertToVector128Int32(packed.GetLower()),
            Sse41.ConvertToVector128Int32(packed.GetUpper()));
    }
}
