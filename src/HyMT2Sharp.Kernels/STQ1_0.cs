using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Sherry/STQ1_0 sparse ternary dot products (canonical GGUF type 43;
/// legacy HyMT2 files are normalized from raw type 42 by GgufFile).
/// Each 256-weight block stores 64 stride-16 groups.  Every group has one
/// zero and three values of equal magnitude, encoded by a 32-entry codebook.
/// </summary>
public static unsafe class STQ1_0
{
    public const int BlockLength = Qk.STQ1_0BlockLength;

    private static readonly Vector128<byte> CodebookLo = Vector128.Create(
        (byte)0xA9, (byte)0x89, (byte)0x29, (byte)0x09,
        (byte)0xA6, (byte)0x86, (byte)0x26, (byte)0x06,
        (byte)0x9A, (byte)0x92, (byte)0x1A, (byte)0x12,
        (byte)0x6A, (byte)0x62, (byte)0x4A, (byte)0x42);
    private static readonly Vector128<byte> CodebookHi = Vector128.Create(
        (byte)0x01, (byte)0x21, (byte)0x81, (byte)0xA1,
        (byte)0x04, (byte)0x24, (byte)0x84, (byte)0xA4,
        (byte)0x10, (byte)0x18, (byte)0x90, (byte)0x98,
        (byte)0x40, (byte)0x48, (byte)0x60, (byte)0x68);
    private static readonly Vector128<byte> NibbleMask = Vector128.Create((byte)0x0F);
    // Interleave 16 lo-lanes with 16 hi-lanes: [0,16,1,17,…,15,31].
    private static readonly Vector256<byte> InterleaveSlots = Vector256.Create(
        (byte)0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23,
        8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    private static readonly Vector128<byte> LaneMask = Vector128.Create((byte)0x03);
    private static readonly Vector128<short> Ones = Vector128.Create((short)1);
    // Eight 0/0xff sign selectors per byte.  Keeping the expansion in a
    // vector LUT removes the stackalloc and bit loop from every 16-group
    // chunk in the AVX2 fallback/tail path.
    private static readonly Vector128<byte>[] SignMaskLut = BuildSignMaskLut();

    private static Vector128<byte>[] BuildSignMaskLut()
    {
        Vector128<byte>[] table = new Vector128<byte>[256];
        for (int value = 0; value < table.Length; value++)
        {
            ulong bits = 0;
            for (int i = 0; i < 8; i++)
                if (((value >> i) & 1) != 0)
                    bits |= 0xFFUL << (i * 8);
            table[value] = Vector128.CreateScalar(bits).AsByte();
        }
        return table;
    }

    // Packed four-lane ternary patterns.  A lane uses 2 bits: 0=-1, 1=0,
    // 2=+1.  The index is (sign << 4) | slot.
    private static ReadOnlySpan<byte> Codebook =>
    [
        0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06,
        0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
        0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4,
        0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
    ];

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte QPack(BlockSTQ1_0* x, int group)
    {
        int slot = (x->Qs[group >> 1] >> ((group & 1) * 4)) & 0x0F;
        int sign = (x->Sign[group >> 3] >> (group & 7)) & 1;
        return Codebook[(sign << 4) | slot];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int Lane(byte qpack, int lane) => ((qpack >> (lane * 2)) & 3) - 1;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static byte Code(BlockSTQ1_0* x, int index)
    {
        int chunk = index >> 6;
        int gloc = index & 15;
        int lane = (index >> 4) & 3;
        return (byte)((QPack(x, (chunk << 4) + gloc) >> (lane * 2)) & 3);
    }

    public static void DequantizeRow(BlockSTQ1_0* x, float* dst, int n)
    {
        if (n <= 0 || n % BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows must contain a positive multiple of 256 values.", nameof(n));

        for (int b = 0; b < n / BlockLength; b++)
        {
            float d = HalfBits.ToSingle(x[b].D);
            for (int g = 0; g < BlockLength / 4; g++)
            {
                byte qpack = QPack(x + b, g);
                int chunk = g >> 4;
                int gloc = g & 15;
                for (int lane = 0; lane < 4; lane++)
                    dst[b * BlockLength + chunk * 64 + gloc + lane * 16] = Lane(qpack, lane) * d;
            }
        }
    }

    public static float Dot(BlockSTQ1_0* x, BlockQ8K* y, int n) =>
        Simd.UseAvx2 ? DotAvx2(x, y, n) : DotVec(x, y, n);

    /// <summary>
    /// Portable fallback: vector codebook gather via <see cref="Vector128.Shuffle"/>
    /// (pshufb/tbl semantics with a software fallback), then lane extraction and
    /// widening dot. Encoded 0/1/2 sums are corrected by the activation sum
    /// exactly like <see cref="DotAvx2"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotVec(BlockSTQ1_0* x, BlockQ8K* y, int n)
    {
        if (n <= 0 || n % BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows must contain a positive multiple of 256 values.", nameof(n));

        Vector128<byte> laneMask = Vector128.Create((byte)3);
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            Vector256<int> acc = Vector256<int>.Zero;
            for (int chunk = 0; chunk < 4; chunk++)
            {
                byte* qs = x[b].Qs + chunk * 8;
                byte* signs = x[b].Sign + chunk * 2;
                sbyte* act = y[b].Qs + chunk * 64;

                // Portable version of DotChunk16's gather: eight bytes hold
                // sixteen 4-bit slots; a 256-wide shuffle interleaves the
                // lo/hi nibble vectors into one slot per byte.
                Vector128<byte> qbytes = Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(qs)).AsByte();
                Vector128<byte> lo = qbytes & NibbleMask;
                Vector128<byte> hi = (qbytes.AsUInt16() >> 4).AsByte() & NibbleMask;
                Vector128<byte> slots = Vector256.Shuffle(Vector256.Create(lo, hi), InterleaveSlots).GetLower();

                Vector128<byte> signMask = Vector128.Create(
                    SignMaskLut[signs[0]].AsUInt64().ToScalar(),
                    SignMaskLut[signs[1]].AsUInt64().ToScalar()).AsByte();
                Vector128<byte> q0 = Vector128.Shuffle(CodebookLo, slots);
                Vector128<byte> q1 = Vector128.Shuffle(CodebookHi, slots);
                Vector128<byte> qp128 = (q0 & ~signMask) | (q1 & signMask);

                for (int s = 0; s < 4; s++)
                {
                    Vector128<byte> codes = (qp128.AsUInt16() >> (2 * s)).AsByte() & laneMask;
                    (Vector128<short> cwL, Vector128<short> cwH) = Vector128.Widen(codes.AsSByte());
                    (Vector128<short> awL, Vector128<short> awH) =
                        Vector128.Widen(Unsafe.ReadUnaligned<Vector128<sbyte>>(act + s * 16));
                    (Vector128<int> p0, Vector128<int> p1) = Vector128.Widen(cwL * awL);
                    (Vector128<int> p2, Vector128<int> p3) = Vector128.Widen(cwH * awH);
                    acc += Vector256.Create(p0, p1) + Vector256.Create(p2, p3);
                }
            }

            int actSum = 0;
            for (int t = 0; t < Qk.SuperBlock / 16; t++)
                actSum += y[b].Bsums[t];
            int encoded = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
            sum += HalfBits.ToSingle(x[b].D) * y[b].D * (encoded - actSum);
        }

        return sum;
    }

    public static float DotScalar(BlockSTQ1_0* x, BlockQ8K* y, int n)
    {
        if (n <= 0 || n % BlockLength != 0)
            throw new ArgumentException("STQ1_0 rows must contain a positive multiple of 256 values.", nameof(n));

        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int si = 0;
            for (int g = 0; g < BlockLength / 4; g++)
            {
                byte qpack = QPack(x + b, g);
                int chunk = g >> 4;
                int gloc = g & 15;
                for (int lane = 0; lane < 4; lane++)
                    si += Lane(qpack, lane) * y[b].Qs[chunk * 64 + gloc + lane * 16];
            }
            sum += HalfBits.ToSingle(x[b].D) * y[b].D * si;
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotAvx2(BlockSTQ1_0* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int encoded = 0;
            int activationSum = 0;
            for (int chunk = 0; chunk < 4; chunk++)
            {
                encoded += DotChunk16(x + b, y + b, chunk);
                for (int i = 0; i < 4; i++)
                    activationSum += y[b].Bsums[chunk * 4 + i];
            }

            sum += HalfBits.ToSingle(x[b].D) * y[b].D * (encoded - activationSum);
        }
        return sum;
    }

    /// <summary>
    /// Computes the encoded 0/1/2 dot for one 64-value stride-16 chunk (16
    /// groups).  The caller subtracts the Q8 activation sum to turn q=0/1/2
    /// into the signed ternary values -1/0/+1.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int DotChunk16(BlockSTQ1_0* x, BlockQ8K* y, int chunk)
    {
        byte* qs = x->Qs + chunk * 8;
        byte* signs = x->Sign + chunk * 2;
        sbyte* activations = y->Qs + chunk * 64;

        // Eight bytes contain sixteen 4-bit codebook slots.  Interleaving the
        // low and high nibbles gives one slot per byte for vpshufb.
        ulong packed = Unsafe.ReadUnaligned<ulong>(qs);
        Vector128<byte> qbytes = Vector128.CreateScalar(packed).AsByte();
        Vector128<byte> lo = Sse2.And(qbytes, NibbleMask);
        Vector128<byte> hi = Sse2.And(Sse2.ShiftRightLogical(qbytes.AsUInt16(), 4).AsByte(), NibbleMask);
        Vector128<byte> slots = Sse2.UnpackLow(lo, hi);

        Vector128<byte> signMask = Sse2.Or(
            SignMaskLut[signs[0]],
            Sse2.ShiftLeftLogical128BitLane(SignMaskLut[signs[1]], 8));

        Vector128<byte> q0 = Ssse3.Shuffle(CodebookLo, slots);
        Vector128<byte> q1 = Ssse3.Shuffle(CodebookHi, slots);
        Vector128<byte> qpack = Sse2.Or(Sse2.AndNot(signMask, q0), Sse2.And(signMask, q1));

        Vector128<int> acc = Vector128<int>.Zero;
        Vector128<byte> q = Sse2.And(qpack, LaneMask);
        Vector128<sbyte> a = Sse2.LoadVector128(activations);
        Vector128<short> pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 2).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 16);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 4).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 32);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));
        q = Sse2.And(Sse2.ShiftRightLogical(qpack.AsUInt16(), 6).AsByte(), LaneMask);
        a = Sse2.LoadVector128(activations + 48);
        pair = Ssse3.MultiplyAddAdjacent(q, a);
        acc = Sse2.Add(acc, Sse2.MultiplyAddAdjacent(pair, Ones));

        acc = Ssse3.HorizontalAdd(acc, acc);
        acc = Ssse3.HorizontalAdd(acc, acc);
        return acc.ToScalar();
    }
}
