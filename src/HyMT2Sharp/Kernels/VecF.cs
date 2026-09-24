using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Portable <see cref="Vector{T}"/> float helpers for the non-AVX2 fallback tier.
/// All loops handle a scalar tail so any length works on both 128- and 256-bit widths.
/// </summary>
public static unsafe class VecF
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector<float> Load(float* p) => Unsafe.ReadUnaligned<Vector<float>>(p);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Store(float* p, Vector<float> v) => Unsafe.WriteUnaligned(p, v);

    /// <summary>max |x[i]| over n floats.</summary>
    public static float AbsMax(float* src, int n)
    {
        Vector<float> acc = Vector<float>.Zero;
        int i = 0;
        for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
            acc = Vector.Max(acc, Vector.Abs(Load(src + i)));
        float amax = 0;
        for (int l = 0; l < Vector<float>.Count; l++)
            amax = MathF.Max(amax, acc[l]);
        for (; i < n; i++)
            amax = MathF.Max(amax, MathF.Abs(src[i]));
        return amax;
    }

    /// <summary>
    /// round(src·id) clamped to [-127,127] stored as sbyte. Banker's rounding via
    /// <see cref="Vector.Round"/> matches MathF.Round and Avx.RoundToNearestInteger.
    /// </summary>
    public static void QuantizeStore(float* src, float id, sbyte* qs, int n)
    {
        Vector<float> idv = new Vector<float>(id);
        Vector<int> lo = new Vector<int>(-127);
        Vector<int> hi = new Vector<int>(127);
        int i = 0;
        for (; i + 4 * Vector<int>.Count <= n; i += 4 * Vector<int>.Count)
        {
            Vector<int> q0 = Quantize(src + i + 0 * Vector<int>.Count, idv, lo, hi);
            Vector<int> q1 = Quantize(src + i + 1 * Vector<int>.Count, idv, lo, hi);
            Vector<int> q2 = Quantize(src + i + 2 * Vector<int>.Count, idv, lo, hi);
            Vector<int> q3 = Quantize(src + i + 3 * Vector<int>.Count, idv, lo, hi);
            Vector<sbyte> packed = Vector.Narrow(Vector.Narrow(q0, q1), Vector.Narrow(q2, q3));
            Unsafe.WriteUnaligned(qs + i, packed);
        }

        for (; i < n; i++)
        {
            int v = (int)MathF.Round(id * src[i]);
            qs[i] = (sbyte)Math.Clamp(v, -127, 127);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<int> Quantize(float* src, Vector<float> idv, Vector<int> lo, Vector<int> hi)
    {
        Vector<float> rounded = Vector.Round(Load(src) * idv);
        return Vector.Min(Vector.Max(Vector.ConvertToInt32(rounded), lo), hi);
    }

    /// <summary>dst[i] = d * qs[i] over n signed bytes.</summary>
    public static void DequantI8(sbyte* qs, float d, float* dst, int n)
    {
        Vector<float> dv = new Vector<float>(d);
        int i = 0;
        for (; i + Vector<sbyte>.Count <= n; i += Vector<sbyte>.Count)
        {
            Vector.Widen(VecI8.LoadS8(qs + i), out Vector<short> lo, out Vector<short> hi);
            Vector.Widen(lo, out Vector<int> l0, out Vector<int> l1);
            Vector.Widen(hi, out Vector<int> h0, out Vector<int> h1);
            Store(dst + i + 0 * Vector<int>.Count, Vector.ConvertToSingle(l0) * dv);
            Store(dst + i + 1 * Vector<int>.Count, Vector.ConvertToSingle(l1) * dv);
            Store(dst + i + 2 * Vector<int>.Count, Vector.ConvertToSingle(h0) * dv);
            Store(dst + i + 3 * Vector<int>.Count, Vector.ConvertToSingle(h1) * dv);
        }

        for (; i < n; i++)
            dst[i] = d * qs[i];
    }
}
