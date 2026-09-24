using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Portable <see cref="Vector{T}"/> int8 primitives for the non-AVX2 fallback tier.
/// <see cref="Vector.Dot{T}(Vector{T}, Vector{T})"/> is intentionally avoided for int8
/// data: it returns T and wraps modulo 256 for sbyte. These helpers widen to
/// i16/i32 explicitly. Pairwise i16 sums are safe because activations are clamped
/// to ±127: |w·a| ≤ 16384 and each i16 lane holds w_lo·a_lo + w_hi·a_hi ≤ 32512 &lt; 32767,
/// even when a weight is -128.
/// </summary>
public static unsafe class VecI8
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<sbyte> LoadS8(sbyte* p) => Unsafe.ReadUnaligned<Vector<sbyte>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<byte> LoadU8(byte* p) => Unsafe.ReadUnaligned<Vector<byte>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<short> LoadS16(short* p) => Unsafe.ReadUnaligned<Vector<short>>(p);

    /// <summary>
    /// The <see cref="BlockQ8KAct"/> kernels walk 16 u16 lanes per 32-value group, so
    /// they need Vector&lt;ushort&gt; of at most 16 lanes (128/256-bit Vector&lt;T&gt;).
    /// </summary>
    public static bool PairLayoutSupported => Vector<ushort>.Count <= 16;

    /// <summary>Pairwise i16 → i32 sum via in-lane shifts; lane order is irrelevant for a dot.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<int> Fold(Vector<short> p)
    {
        Vector<int> v = Vector.AsVectorInt32(p);
        return Vector.ShiftRightArithmetic(Vector.ShiftLeft(v, 16), 16) + Vector.ShiftRightArithmetic(v, 16);
    }

    /// <summary>acc += Σ w[k]·a[k]; each i16 pair stays in range under the ±127 activation bound.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void AccDotI8(Vector<sbyte> w, Vector<sbyte> a, ref Vector<int> acc)
    {
        Vector.Widen(w, out Vector<short> wl, out Vector<short> wh);
        Vector.Widen(a, out Vector<short> al, out Vector<short> ah);
        Vector.Widen(wl * al + wh * ah, out Vector<int> p0, out Vector<int> p1);
        acc += p0 + p1;
    }

    /// <summary>Σ w[i]·a[i] over n signed bytes, exact int32.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int DotI8(sbyte* w, sbyte* a, int n)
    {
        Vector<int> acc = Vector<int>.Zero;
        int i = 0;
        for (; i + Vector<sbyte>.Count <= n; i += Vector<sbyte>.Count)
            AccDotI8(LoadS8(w + i), LoadS8(a + i), ref acc);
        int sum = Vector.Sum(acc);
        for (; i < n; i++)
            sum += w[i] * a[i];
        return sum;
    }

    /// <summary>Σ a[i] over n signed bytes (Bsums/activation sums).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int SumI8(sbyte* a, int n)
    {
        Vector<short> acc = Vector<short>.Zero;
        int i = 0;
        for (; i + Vector<sbyte>.Count <= n; i += Vector<sbyte>.Count)
        {
            Vector.Widen(LoadS8(a + i), out Vector<short> lo, out Vector<short> hi);
            acc += lo + hi;
        }
        int sum = Vector.Sum(acc);
        for (; i < n; i++)
            sum += a[i];
        return sum;
    }

    /// <summary>
    /// Two independent 16-value dots on contiguous input: s0 = w[0..16]·a[0..16],
    /// s1 = w[16..32]·a[16..32] (Q6_K sub-block granularity at any vector width).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Dot16x2(sbyte* w, sbyte* a, out int s0, out int s1)
    {
        if (Vector<sbyte>.Count == 32)
        {
            Vector.Widen(LoadS8(w), out Vector<short> wl, out Vector<short> wh);
            Vector.Widen(LoadS8(a), out Vector<short> al, out Vector<short> ah);
            Vector.Widen(wl * al, out Vector<int> p0, out Vector<int> p1);
            Vector.Widen(wh * ah, out Vector<int> p2, out Vector<int> p3);
            s0 = Vector.Sum(p0 + p1);
            s1 = Vector.Sum(p2 + p3);
            return;
        }

        if (Vector<sbyte>.Count == 16)
        {
            Vector<int> acc = Vector<int>.Zero;
            AccDotI8(LoadS8(w), LoadS8(a), ref acc);
            s0 = Vector.Sum(acc);
            acc = Vector<int>.Zero;
            AccDotI8(LoadS8(w + 16), LoadS8(a + 16), ref acc);
            s1 = Vector.Sum(acc);
            return;
        }

        s0 = 0;
        s1 = 0;
        for (int i = 0; i < 16; i++)
        {
            s0 += w[i] * a[i];
            s1 += w[16 + i] * a[16 + i];
        }
    }

    /// <summary>
    /// Per-16 activation sums (Q8_K Bsums). Handles 16- and 32-byte vector widths;
    /// wider exotic widths take the scalar loop.
    /// </summary>
    public static void Bsums16(sbyte* qs, short* dst, int n)
    {
        int i = 0;
        if (Vector<sbyte>.Count == 32)
        {
            for (; i + 32 <= n; i += 32, dst += 2)
            {
                Vector.Widen(LoadS8(qs + i), out Vector<short> lo, out Vector<short> hi);
                dst[0] = (short)Vector.Sum(lo);
                dst[1] = (short)Vector.Sum(hi);
            }
        }
        else if (Vector<sbyte>.Count == 16)
        {
            for (; i + 16 <= n; i += 16)
            {
                Vector.Widen(LoadS8(qs + i), out Vector<short> lo, out Vector<short> hi);
                *dst++ = (short)(Vector.Sum(lo) + Vector.Sum(hi));
            }
        }

        for (; i + 16 <= n; i += 16)
        {
            int s = 0;
            for (int l = 0; l < 16; l++)
                s += qs[i + l];
            *dst++ = (short)s;
        }
    }

    /// <summary>
    /// 32 packed nibbles against two 32-value activation spans: lo nibbles → aLo,
    /// hi nibbles → aHi (the Q4_K sub-block layout).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void DotNibblesI8(byte* packed, sbyte* aLo, sbyte* aHi, out int lo, out int hi)
    {
        Vector<int> accLo = Vector<int>.Zero;
        Vector<int> accHi = Vector<int>.Zero;
        Vector<byte> m4 = new Vector<byte>(0x0F);
        int l = 0;
        for (; l + Vector<byte>.Count <= 32; l += Vector<byte>.Count)
        {
            Vector<byte> pk = LoadU8(packed + l);
            AccDotI8(Vector.AsVectorSByte(pk & m4), LoadS8(aLo + l), ref accLo);
            AccDotI8(Vector.AsVectorSByte(pk >> 4), LoadS8(aHi + l), ref accHi);
        }

        lo = Vector.Sum(accLo);
        hi = Vector.Sum(accHi);
        for (; l < 32; l++)
        {
            lo += (packed[l] & 0xF) * aLo[l];
            hi += (packed[l] >> 4) * aHi[l];
        }
    }

}
