using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static class Qk
{
    public const int SuperBlock = 256;
    public const int ScaleBytes = 12;
    public const int Q4KSize = 144;
    public const int Q8KSize = 292;
    public const int Q4Kx8Size = 1152;
    public const int Q4Kx8MetaSize = 448;
    public const int Q8Kx4Size = 1168;
    public const int Q5KSize = 176;
    public const int Q6KSize = 210;
    public const int Q6Kx8Size = 2848;
    public const int Q8_0Block = 32;
    public const int Q8_0Size = 34;
    // Eight Q8_0 columns, one 32-value block. Qs holds signed bytes grouped as
    // column c's K values 4j..4j+3 so one activation dword broadcast feeds all 8 columns.
    public const int Q8_0x8Size = 8 * sizeof(float) + Q8_0Block * 8;
    public const int Q8_0x4Size = 4 * sizeof(float) + 4 * sizeof(int) + Q8_0Block * 4;
    public const int Q8_0ActSize = 8 + Q8_0Block;
    public const int Q2_0CSize = 130;
    public const int Q2x8Size = 1056;
    public const int STQ1_0BlockLength = 256;
    public const int STQ1_0Size = 42;
    // Eight STQ rows for one 256-value block.  Qs is repacked to the same
    // 8-column/32-value tile shape used by the Q2 panel kernels; D keeps one
    // fp32 scale per row (the source STQ scale is fp16).
    public const int STQ1_0x8Size = 8 * sizeof(float) + 512;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ4K
{
    public ushort D;
    public ushort Dmin;
    public fixed byte Scales[Qk.ScaleBytes];
    public fixed byte Qs[Qk.SuperBlock / 2];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8K
{
    public float D;
    public fixed sbyte Qs[Qk.SuperBlock];
    public fixed short Bsums[Qk.SuperBlock / 16];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ5K
{
    public ushort D;
    public ushort Dmin;
    public fixed byte Scales[Qk.ScaleBytes];
    public fixed byte Qh[Qk.SuperBlock / 8];
    public fixed byte Qs[Qk.SuperBlock / 2];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ6K
{
    public fixed byte Ql[Qk.SuperBlock / 2];
    public fixed byte Qh[Qk.SuperBlock / 4];
    public fixed sbyte Scales[Qk.SuperBlock / 16];
    public ushort D;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ2_0C
{
    public ushort D;
    public fixed byte Qs[Qk.SuperBlock * 2 / 4];
}

/// <summary>
/// Sherry/STQ1_0 sparse ternary block.  A block contains 256 weights:
/// 64 stride-16 groups of four ternary lanes, represented by a 4-bit
/// codebook slot and a 1-bit global sign per group, followed by an fp16 scale.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockSTQ1_0
{
    public fixed byte Qs[Qk.STQ1_0BlockLength / 8];
    public fixed byte Sign[Qk.STQ1_0BlockLength / 32];
    public ushort D;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockSTQ1_0x8
{
    public fixed float D[8];
    public fixed byte Qs[512];
}

/// <summary>
/// Eight Q2 columns, still 2 bits per weight. Each 64-byte chunk contains
/// columns 0..7 x eight bytes; the four 2-bit planes hold consecutive
/// groups of eight K values. A broadcast of eight activations feeds four columns.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ2x8
{
    public fixed float D[8];
    public fixed byte Qs[1024];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ4Kx8
{
    public fixed ushort D[8];
    public fixed ushort Dmin[8];
    public fixed byte Scales[96];
    public fixed byte Qs[1024];
}

/// <summary>
/// Pre-decoded per-block metadata for the AVX2 panel GEMM so the hot loop never
/// unpacks 6-bit scales. <see cref="Scales"/> holds, per 64-value pair of sub-blocks,
/// three int16 vectors: lo scales and hi scales in the <c>vphaddw</c> output order
/// [c0 c0 c1 c1 c4 c4 c5 c5 | c2 c2 c3 c3 c6 c6 c7 c7], then mins interleaved
/// [m0lo m0hi m1lo m1hi … m7lo m7hi].
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ4Kx8Meta
{
    public fixed float D[8];
    public fixed float Dmin[8];
    public fixed short Scales[4 * 48];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8Kx4
{
    public fixed float D[4];
    public fixed sbyte Qs[Qk.SuperBlock * 4];
    public fixed short Bsums[Qk.SuperBlock / 4];
}

/// <summary>
/// Prefill panel for 8 Q6_K columns. Values are expanded to u8 (0..63) so the kernel
/// needs no bit surgery; <see cref="Qs"/> chunk j (32 bytes) holds columns 0..7 × K values
/// 4j..4j+3 so one <c>vpbroadcastd</c> of the activation feeds all 8 columns.
/// <see cref="Scales"/>[i] is [sc_i(c0) sc_i(c0) … sc_i(c7) sc_i(c7)] for 16-value sub-block i;
/// <see cref="ScalePairs"/>[p] is [sc_2p(c0) sc_2p+1(c0) …] for the −32 bsum correction.
/// </summary>
/// <summary>GGUF Q8_0 block: fp16 scale plus 32 signed int8 values. 34 bytes, tightly packed.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8_0
{
    public ushort D;
    public fixed sbyte Qs[Qk.Q8_0Block];
}

/// <summary>One activation row of Q8_0 for GEMV. <see cref="Sum"/> is the sum of the 32 signed quants.</summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8_0Act
{
    public float D;
    public int Sum;
    public fixed sbyte Qs[Qk.Q8_0Block];
}

/// <summary>
/// Four activation rows, 32 K values. <c>Qs[r * 32 + k]</c> is row r.
/// <c>Bias[r]</c> is 128 times the sum of row r's 32 signed quants, for the VNNI +128 weight bias.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8_0x4
{
    public fixed float D[4];
    public fixed int Bias[4];
    public fixed sbyte Qs[Qk.Q8_0Block * 4];
}

/// <summary>
/// Eight Q8_0 columns. <see cref="Qs"/> chunk j (32 bytes) is columns 0..7 × K values
/// 4j..4j+3, stored as signed int8 bit patterns.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ8_0x8
{
    public fixed float D[8];
    public fixed byte Qs[Qk.Q8_0Block * 8];
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public unsafe struct BlockQ6Kx8
{
    public fixed float D[8];
    public fixed short Scales[16 * 16];
    public fixed short ScalePairs[8 * 16];
    public fixed byte Qs[Qk.SuperBlock * 8];
}

public static class HalfBits
{
    public static float ToSingle(ushort bits) => (float)BitConverter.UInt16BitsToHalf(bits);

    public static ushort FromSingle(float value) => BitConverter.HalfToUInt16Bits((Half)value);

    /// <summary>
    /// AVX2 software cvtph_ps for finite normals and signed zeros (GGUF d/dmin).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static unsafe Vector256<float> Load8(ushort* src)
    {
        if (Simd.UseAvx2)
        {
            Vector128<ushort> h = Avx.LoadVector128(src);
            Vector256<int> bits = Avx2.ConvertToVector256Int32(h);
            Vector256<int> sign = Avx2.ShiftLeftLogical(Avx2.And(bits, Vector256.Create(0x8000)), 16);
            Vector256<int> mag = Avx2.And(bits, Vector256.Create(0x7FFF));
            Vector256<int> f = Avx2.Add(Avx2.ShiftLeftLogical(mag, 13), Vector256.Create(0x38000000));
            Vector256<int> isZero = Avx2.CompareEqual(mag, Vector256<int>.Zero);
            f = Avx2.AndNot(isZero, f);
            return Avx2.Or(f, sign).AsSingle();
        }

        return Vector256.Create(
            ToSingle(src[0]), ToSingle(src[1]), ToSingle(src[2]), ToSingle(src[3]),
            ToSingle(src[4]), ToSingle(src[5]), ToSingle(src[6]), ToSingle(src[7]));
    }
}
