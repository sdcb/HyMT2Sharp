using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// llama.cpp <c>ggml_v_expf</c> / <c>ggml_v_silu</c> (AVX2+FMA, vec.h).
/// Max error ~1.45 ulp; values above 88.38 flush to +inf, below -103.97 to 0.
/// </summary>
public static class FastExp
{
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector256<float> ExpAvx2(Vector256<float> x)
    {
        Vector256<float> r = Vector256.Create(12582912f);
        Vector256<float> z = Fma.MultiplyAdd(x, Vector256.Create(1.4426950216293335f), r);
        Vector256<float> n = Avx.Subtract(z, r);
        Vector256<float> b = Fma.MultiplyAddNegated(n, Vector256.Create(1.4286067655086517e-6f),
            Fma.MultiplyAddNegated(n, Vector256.Create(0.693145751953125f), x));
        Vector256<int> e = Avx2.ShiftLeftLogical(z.AsInt32(), 23);
        Vector256<float> k = Avx2.Add(e, Vector256.Create(1f).AsInt32()).AsSingle();
        Vector256<float> nAbs = Avx.AndNot(Vector256.Create(-0f), n);
        Vector256<float> overflow = Avx.Compare(nAbs, Vector256.Create(126f), FloatComparisonMode.OrderedGreaterThanNonSignaling);
        Vector256<float> u = Avx.Multiply(b, b);
        Vector256<float> j = Fma.MultiplyAdd(
            Fma.MultiplyAdd(
                Fma.MultiplyAdd(Vector256.Create(0.008301135152578354f), b, Vector256.Create(0.04190601035952568f)),
                u,
                Fma.MultiplyAdd(Vector256.Create(0.166665181517601f), b, Vector256.Create(0.4999998211860657f))),
            u,
            Avx.Multiply(Vector256.Create(0.9999998211860657f), b));
        if (Avx.MoveMask(overflow) == 0)
            return Fma.MultiplyAdd(j, k, k);

        Vector256<float> nLe0 = Avx.Compare(n, Vector256<float>.Zero, FloatComparisonMode.OrderedLessThanOrEqualNonSignaling);
        Vector256<int> g = Avx2.And(nLe0.AsInt32(), Vector256.Create(unchecked((int)0x82000000)));
        Vector256<float> s1 = Avx2.Add(g, Vector256.Create(0x7F000000)).AsSingle();
        Vector256<float> s2 = Avx2.Subtract(e, g).AsSingle();
        Vector256<float> huge = Avx.Compare(nAbs, Vector256.Create(192f), FloatComparisonMode.OrderedGreaterThanNonSignaling);
        Vector256<float> hi = Avx.Multiply(s1, s1);
        Vector256<float> mid = Avx.Multiply(Fma.MultiplyAdd(s2, j, s2), s1);
        Vector256<float> lo = Fma.MultiplyAdd(k, j, k);
        return Avx.Or(Avx.And(huge, hi), Avx.AndNot(huge, Avx.Or(Avx.And(overflow, mid), Avx.AndNot(overflow, lo))));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector256<float> SiluAvx2(Vector256<float> x)
    {
        Vector256<float> expNeg = ExpAvx2(Avx.Subtract(Vector256<float>.Zero, x));
        return Avx.Divide(x, Avx.Add(Vector256<float>.One, expNeg));
    }

    /// <summary>
    /// Portable <see cref="Vector{T}"/> port of <see cref="ExpAvx2"/>: same polynomial
    /// and overflow handling, without fused multiply-add.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector<float> ExpVec(Vector<float> x)
    {
        Vector<float> r = new Vector<float>(12582912f);
        Vector<float> z = x * new Vector<float>(1.4426950216293335f) + r;
        Vector<float> n = z - r;
        Vector<float> b = x - n * new Vector<float>(0.693145751953125f) - n * new Vector<float>(1.4286067655086517e-6f);
        Vector<int> e = Vector.AsVectorInt32(z) << 23;
        Vector<float> k = Vector.AsVectorSingle(e + new Vector<int>(0x3F800000));
        Vector<float> nAbs = Vector.Abs(n);
        Vector<int> overflow = Vector.GreaterThan(nAbs, new Vector<float>(126f));
        Vector<float> u = b * b;
        Vector<float> j =
            (((new Vector<float>(0.008301135152578354f) * b + new Vector<float>(0.04190601035952568f)) * u +
              new Vector<float>(0.166665181517601f) * b + new Vector<float>(0.4999998211860657f)) * u) +
            new Vector<float>(0.9999998211860657f) * b;
        if (Vector.Sum(overflow) == 0)
            return j * k + k;

        Vector<int> nLe0 = Vector.LessThanOrEqual(n, Vector<float>.Zero);
        Vector<int> g = nLe0 & new Vector<int>(unchecked((int)0x82000000));
        Vector<float> s1 = Vector.AsVectorSingle(g + new Vector<int>(0x7F000000));
        Vector<float> s2 = Vector.AsVectorSingle(e - g);
        Vector<int> huge = Vector.GreaterThan(nAbs, new Vector<float>(192f));
        Vector<float> hi = s1 * s1;
        Vector<float> mid = (s2 * j + s2) * s1;
        Vector<float> lo = k * j + k;
        return Vector.ConditionalSelect(huge, hi, Vector.ConditionalSelect(overflow, mid, lo));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Vector<float> SiluVec(Vector<float> x) =>
        x / (Vector<float>.One + ExpVec(Vector<float>.Zero - x));
}
