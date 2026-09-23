using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>512 weights per half scale, four consecutive 2-bit codes per byte.</summary>
public static unsafe class Q2_0C
{
    public const int BlockLength = 512;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Value(byte b, int shift) => ((b >> shift) & 3) * 2 - 3;

    public static void DequantizeRow(BlockQ2_0C* x, float* dst, int n)
    {
        for (int b = 0; b < n / BlockLength; b++)
        {
            float d = HalfBits.ToSingle(x[b].D);
            for (int j = 0; j < BlockLength; j++)
                dst[b * BlockLength + j] = Value(x[b].Qs[j / 4], (j & 3) * 2) * d;
        }
    }

    public static void Expand(BlockQ2_0C* src, sbyte* dst, int n)
    {
        for (int b = 0; b < n / BlockLength; b++)
            for (int j = 0; j < BlockLength; j++)
                dst[b * BlockLength + j] = (sbyte)Value(src[b].Qs[j / 4], (j & 3) * 2);
    }

    public static float Dot(BlockQ2_0C* x, BlockQ8K* y, int n) =>
        Simd.UseAvx2 ? DotAvx2(x, y, n) : DotVec(x, y, n);

    /// <summary>
    /// Portable fallback: same dword bit-spread as <see cref="DotAvx2"/> via
    /// <see cref="Vector{T}"/> widening, codes dot'd against activations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotVec(BlockQ2_0C* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int s0 = Dot256Vec(x[b].Qs, y[b * 2].Qs, y[b * 2].Bsums);
            int s1 = Dot256Vec(x[b].Qs + 64, y[b * 2 + 1].Qs, y[b * 2 + 1].Bsums);
            sum += HalfBits.ToSingle(x[b].D) * (s0 * y[b * 2].D + s1 * y[b * 2 + 1].D);
        }
        return sum;
    }

    /// <summary>Sum of (2c-3)*a over 256 values = 2*Σ(c*a) - 3*Σa; Σa comes from Bsums.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot256Vec(byte* q, sbyte* y, short* bsums)
    {
        Vector<int> acc = Vector<int>.Zero;
        sbyte* a = y;
        int j = 0;
        for (; j + Vector<byte>.Count <= 64; j += Vector<byte>.Count)
        {
            Vector.Widen(VecI8.LoadU8(q + j), out Vector<ushort> lo, out Vector<ushort> hi);
            Vector.Widen(lo, out Vector<uint> q0, out Vector<uint> q1);
            Vector.Widen(hi, out Vector<uint> q2, out Vector<uint> q3);
            SpreadDot(q0, a, ref acc);
            SpreadDot(q1, a + Vector<int>.Count * 4, ref acc);
            SpreadDot(q2, a + Vector<int>.Count * 8, ref acc);
            SpreadDot(q3, a + Vector<int>.Count * 12, ref acc);
            a += Vector<int>.Count * 16;
        }

        int tail = 0;
        for (; j < 64; j++, a += 4)
        {
            byte b = q[j];
            tail += (b & 3) * a[0] + ((b >> 2) & 3) * a[1] + ((b >> 4) & 3) * a[2] + ((b >> 6) & 3) * a[3];
        }

        int actSum = 0;
        for (int t = 0; t < Qk.SuperBlock / 16; t++)
            actSum += bsums[t];
        return 2 * (Vector.Sum(acc) + tail) - 3 * actSum;
    }

    /// <summary>Spread one packed byte per u32 lane into four 2-bit code bytes, then dot.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SpreadDot(Vector<uint> v, sbyte* a, ref Vector<int> acc)
    {
        v = (v | (v << 12)) & new Vector<uint>(0x000F000F);
        v = (v | (v << 6)) & new Vector<uint>(0x03030303);
        VecI8.AccDotI8(Vector.AsVectorSByte(v), VecI8.LoadS8(a), ref acc);
    }

    public static float DotScalarForTest(BlockQ2_0C* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int s0 = 0, s1 = 0;
            for (int j = 0; j < 256; j++)
            {
                s0 += Value(x[b].Qs[j / 4], (j & 3) * 2) * y[b * 2].Qs[j];
                s1 += Value(x[b].Qs[64 + j / 4], (j & 3) * 2) * y[b * 2 + 1].Qs[j];
            }
            sum += HalfBits.ToSingle(x[b].D) * (s0 * y[b * 2].D + s1 * y[b * 2 + 1].D);
        }
        return sum;
    }

    // Retained as a reference for callers with expanded signed weights. The model
    // uses the compressed x8 panel, so it never allocates a byte-per-weight copy.
    public static float DotExpanded(BlockQ2_0C* raw, sbyte* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int s0 = 0, s1 = 0;
            for (int j = 0; j < 256; j++)
            {
                s0 += x[b * 512 + j] * y[b * 2].Qs[j];
                s1 += x[b * 512 + 256 + j] * y[b * 2 + 1].Qs[j];
            }
            sum += HalfBits.ToSingle(raw[b].D) * (s0 * y[b * 2].D + s1 * y[b * 2 + 1].D);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAvx2(BlockQ2_0C* x, BlockQ8K* y, int n)
    {
        float sum = 0;
        for (int b = 0; b < n / BlockLength; b++)
        {
            int s0 = Dot256(x[b].Qs, y[b * 2].Qs);
            int s1 = Dot256(x[b].Qs + 64, y[b * 2 + 1].Qs);
            sum += HalfBits.ToSingle(x[b].D) * (s0 * y[b * 2].D + s1 * y[b * 2 + 1].D);
        }
        return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot256(byte* q, sbyte* y)
    {
        Vector256<short> acc = Vector256<short>.Zero;
        for (int j = 0; j < 64; j += 8)
        {
            // Eight packed bytes -> eight dwords, then spread each code into
            // its own byte. No scalar unpack or temporary weight stores.
            Vector256<int> v = Avx2.ConvertToVector256Int32(Vector128.CreateScalar(Unsafe.ReadUnaligned<ulong>(q + j)).AsByte());
            v = Avx2.And(Avx2.Or(v, Avx2.ShiftLeftLogical(v, 12)), Vector256.Create(0x000F000F));
            v = Avx2.And(Avx2.Or(v, Avx2.ShiftLeftLogical(v, 6)), Vector256.Create(0x03030303));
            Vector256<sbyte> a = Avx.LoadVector256(y + j * 4);
            Vector256<short> p = Avx2.MultiplyAddAdjacent(v.AsByte(), a);
            p = Avx2.Subtract(Avx2.Add(p, p), Avx2.MultiplyAddAdjacent(Vector256.Create((byte)3), a));
            acc = Avx2.Add(acc, p);
        }
        Vector256<int> ints = Avx2.MultiplyAddAdjacent(acc, Vector256.Create((short)1));
        Vector128<int> total = Sse2.Add(ints.GetLower(), ints.GetUpper());
        total = Ssse3.HorizontalAdd(total, total);
        return Ssse3.HorizontalAdd(total, total).ToScalar();
    }
}
