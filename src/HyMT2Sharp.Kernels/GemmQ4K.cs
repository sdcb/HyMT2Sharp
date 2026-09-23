using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// 4×8 Q4_K panel GEMM on <see cref="BlockQ4Kx8"/> × <see cref="BlockQ8Kx4"/>.
/// Scalar path matches llama.cpp <c>ggml_gemm_q4_K_8x8_q8_K_generic</c>.
/// AVX2 path ports the 4-row shuffle kernel and keeps named Vector256 accumulators.
/// </summary>
public static unsafe partial class GemmQ4K
{
    public static void Gemm8x8(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols, BlockQ4Kx8Meta* meta)
    {
        if ((rows & 3) != 0)
            throw new ArgumentException("rows must be a multiple of 4.", nameof(rows));
        if ((cols & 7) != 0)
            throw new ArgumentException("cols must be a multiple of 8.", nameof(cols));
        if (Simd.UseAvx2 && meta != null)
            GemmAvx2(n, dst, ldc, weights, activations, rows, cols, meta);
        else
            GemmScalar(n, dst, ldc, weights, activations, rows, cols);
    }

    public static void GemmScalar(int n, float* dst, int ldc, BlockQ4Kx8* weights, BlockQ8Kx4* activations, int rows, int cols)
    {
        int nb = n / Qk.SuperBlock;
        const int blocklen = 8;
        const uint kmask1 = 0x3f3f3f3f;
        const uint kmask2 = 0x0f0f0f0f;
        const uint kmask3 = 0x03030303;
        uint* utmp = stackalloc uint[32];

        for (int y = 0; y < rows / 4; y++)
        {
            BlockQ8Kx4* aPtr = activations + y * nb;
            for (int x = 0; x < cols / 8; x++)
            {
                BlockQ4Kx8* bPtr = weights + x * nb;
                float s00 = 0, s01 = 0, s02 = 0, s03 = 0, s04 = 0, s05 = 0, s06 = 0, s07 = 0;
                float s10 = 0, s11 = 0, s12 = 0, s13 = 0, s14 = 0, s15 = 0, s16 = 0, s17 = 0;
                float s20 = 0, s21 = 0, s22 = 0, s23 = 0, s24 = 0, s25 = 0, s26 = 0, s27 = 0;
                float s30 = 0, s31 = 0, s32 = 0, s33 = 0, s34 = 0, s35 = 0, s36 = 0, s37 = 0;
                float m00 = 0, m01 = 0, m02 = 0, m03 = 0, m04 = 0, m05 = 0, m06 = 0, m07 = 0;
                float m10 = 0, m11 = 0, m12 = 0, m13 = 0, m14 = 0, m15 = 0, m16 = 0, m17 = 0;
                float m20 = 0, m21 = 0, m22 = 0, m23 = 0, m24 = 0, m25 = 0, m26 = 0, m27 = 0;
                float m30 = 0, m31 = 0, m32 = 0, m33 = 0, m34 = 0, m35 = 0, m36 = 0, m37 = 0;

                for (int l = 0; l < nb; l++)
                {
                    for (int sb = 0; sb < 8; sb++)
                    {
                        byte* scaleSrc = bPtr[l].Scales + sb * 12;
                        utmp[sb * 4 + 0] = Unsafe.ReadUnaligned<uint>(scaleSrc);
                        utmp[sb * 4 + 1] = Unsafe.ReadUnaligned<uint>(scaleSrc + 4);
                        utmp[sb * 4 + 2] = Unsafe.ReadUnaligned<uint>(scaleSrc + 8);
                        utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                        uint uaux = utmp[sb * 4 + 1] & kmask1;
                        utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                        utmp[sb * 4 + 2] = uaux;
                        utmp[sb * 4 + 0] &= kmask1;
                    }

                    for (int k = 0; k < Qk.SuperBlock / (2 * blocklen); k++)
                    {
                        byte* scales0 = (byte*)utmp + (k / 4) * 32;
                        byte* scales1 = scales0 + 16;
                        AccumulateScalarRow(ref s00, ref s01, ref s02, ref s03, ref s04, ref s05, ref s06, ref s07,
                            &bPtr[l], &aPtr[l], k, 0, scales0, scales1);
                        AccumulateScalarRow(ref s10, ref s11, ref s12, ref s13, ref s14, ref s15, ref s16, ref s17,
                            &bPtr[l], &aPtr[l], k, 1, scales0, scales1);
                        AccumulateScalarRow(ref s20, ref s21, ref s22, ref s23, ref s24, ref s25, ref s26, ref s27,
                            &bPtr[l], &aPtr[l], k, 2, scales0, scales1);
                        AccumulateScalarRow(ref s30, ref s31, ref s32, ref s33, ref s34, ref s35, ref s36, ref s37,
                            &bPtr[l], &aPtr[l], k, 3, scales0, scales1);
                    }

                    for (int sb = 0; sb < 8; sb++)
                    {
                        byte* mins = (byte*)utmp + 8 + sb * 16;
                        AccumulateMins(ref m00, ref m01, ref m02, ref m03, ref m04, ref m05, ref m06, ref m07,
                            &bPtr[l], &aPtr[l], sb, 0, mins);
                        AccumulateMins(ref m10, ref m11, ref m12, ref m13, ref m14, ref m15, ref m16, ref m17,
                            &bPtr[l], &aPtr[l], sb, 1, mins);
                        AccumulateMins(ref m20, ref m21, ref m22, ref m23, ref m24, ref m25, ref m26, ref m27,
                            &bPtr[l], &aPtr[l], sb, 2, mins);
                        AccumulateMins(ref m30, ref m31, ref m32, ref m33, ref m34, ref m35, ref m36, ref m37,
                            &bPtr[l], &aPtr[l], sb, 3, mins);
                    }
                }

                float* row0 = dst + (y * 4 + 0) * ldc + x * 8;
                float* row1 = dst + (y * 4 + 1) * ldc + x * 8;
                float* row2 = dst + (y * 4 + 2) * ldc + x * 8;
                float* row3 = dst + (y * 4 + 3) * ldc + x * 8;
                row0[0] = s00 - m00; row0[1] = s01 - m01; row0[2] = s02 - m02; row0[3] = s03 - m03;
                row0[4] = s04 - m04; row0[5] = s05 - m05; row0[6] = s06 - m06; row0[7] = s07 - m07;
                row1[0] = s10 - m10; row1[1] = s11 - m11; row1[2] = s12 - m12; row1[3] = s13 - m13;
                row1[4] = s14 - m14; row1[5] = s15 - m15; row1[6] = s16 - m16; row1[7] = s17 - m17;
                row2[0] = s20 - m20; row2[1] = s21 - m21; row2[2] = s22 - m22; row2[3] = s23 - m23;
                row2[4] = s24 - m24; row2[5] = s25 - m25; row2[6] = s26 - m26; row2[7] = s27 - m27;
                row3[0] = s30 - m30; row3[1] = s31 - m31; row3[2] = s32 - m32; row3[3] = s33 - m33;
                row3[4] = s34 - m34; row3[5] = s35 - m35; row3[6] = s36 - m36; row3[7] = s37 - m37;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateScalarRow(
        ref float c0, ref float c1, ref float c2, ref float c3,
        ref float c4, ref float c5, ref float c6, ref float c7,
        BlockQ4Kx8* b, BlockQ8Kx4* a, int k, int m, byte* scales0, byte* scales1)
    {
        float ad = a->D[m];
        AddCol(ref c0, b, a, k, m, 0, scales0, scales1, ad);
        AddCol(ref c1, b, a, k, m, 1, scales0, scales1, ad);
        AddCol(ref c2, b, a, k, m, 2, scales0, scales1, ad);
        AddCol(ref c3, b, a, k, m, 3, scales0, scales1, ad);
        AddCol(ref c4, b, a, k, m, 4, scales0, scales1, ad);
        AddCol(ref c5, b, a, k, m, 5, scales0, scales1, ad);
        AddCol(ref c6, b, a, k, m, 6, scales0, scales1, ad);
        AddCol(ref c7, b, a, k, m, 7, scales0, scales1, ad);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AddCol(ref float dest, BlockQ4Kx8* b, BlockQ8Kx4* a, int k, int m, int j, byte* scales0, byte* scales1, float ad)
    {
        int sumi = 0;
        int aBase0 = (k >> 2) * 256 + (k % 4) * 32 + m * 8;
        int aBase1 = aBase0 + 128;
        int bBase = k * 64 + j * 8;
        for (int i = 0; i < 8; i++)
        {
            int v0 = b->Qs[bBase + i] & 0xF;
            int v1 = b->Qs[bBase + i] >> 4;
            sumi += v0 * a->Qs[aBase0 + i] * scales0[j];
            sumi += v1 * a->Qs[aBase1 + i] * scales1[j];
        }

        dest += sumi * HalfBits.ToSingle(b->D[j]) * ad;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateMins(
        ref float c0, ref float c1, ref float c2, ref float c3,
        ref float c4, ref float c5, ref float c6, ref float c7,
        BlockQ4Kx8* b, BlockQ8Kx4* a, int sb, int m, byte* mins)
    {
        short* bsums = &a->Bsums[(sb * 8) + (m * 4) - ((sb % 2) * 6)];
        int pair = bsums[0] + bsums[1];
        float ad = a->D[m];
        c0 += mins[0] * pair * HalfBits.ToSingle(b->Dmin[0]) * ad;
        c1 += mins[1] * pair * HalfBits.ToSingle(b->Dmin[1]) * ad;
        c2 += mins[2] * pair * HalfBits.ToSingle(b->Dmin[2]) * ad;
        c3 += mins[3] * pair * HalfBits.ToSingle(b->Dmin[3]) * ad;
        c4 += mins[4] * pair * HalfBits.ToSingle(b->Dmin[4]) * ad;
        c5 += mins[5] * pair * HalfBits.ToSingle(b->Dmin[5]) * ad;
        c6 += mins[6] * pair * HalfBits.ToSingle(b->Dmin[6]) * ad;
        c7 += mins[7] * pair * HalfBits.ToSingle(b->Dmin[7]) * ad;
    }
}
