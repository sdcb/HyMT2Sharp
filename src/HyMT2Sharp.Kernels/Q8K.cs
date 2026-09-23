using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class Q8K
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void QuantizeRow(float* x, BlockQ8K* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            float* src = x + i * Qk.SuperBlock;
            float iscale = ScaleBlock(src, out float delta);
            if (iscale == 0)
            {
                y[i].D = 0;
                NativeMemory.Clear(y[i].Qs, (nuint)Qk.SuperBlock);
                NativeMemory.Clear(y[i].Bsums, (nuint)((Qk.SuperBlock / 16) * sizeof(short)));
                continue;
            }

            y[i].D = delta;
            sbyte* qs = y[i].Qs;
            if (Simd.UseAvx2)
            {
                Vector256<float> isv = Vector256.Create(iscale);
                Vector256<int> lo = Vector256.Create(-127);
                Vector256<int> hi = Vector256.Create(127);
                for (int j = 0; j < Qk.SuperBlock; j += 8)
                {
                    Vector256<float> rounded = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(src + j), isv));
                    Vector256<int> qi = Avx2.Min(Avx2.Max(Avx.ConvertToVector256Int32(rounded), lo), hi);
                    Vector128<short> p16 = Sse2.PackSignedSaturate(qi.GetLower(), qi.GetUpper());
                    Vector128<sbyte> p8 = Sse2.PackSignedSaturate(p16, Vector128<short>.Zero);
                    Unsafe.WriteUnaligned(qs + j, p8.AsInt64().ToScalar());
                }
            }
            else
            {
                VecF.QuantizeStore(src, iscale, qs, Qk.SuperBlock);
            }

            if (Simd.UseAvx2)
            {
                for (int j = 0; j < Qk.SuperBlock / 16; j++)
                {
                    Vector256<short> w = Avx2.ConvertToVector256Int16(Avx.LoadVector128(qs + j * 16));
                    Vector128<short> s = Ssse3.HorizontalAdd(w.GetLower(), w.GetUpper());
                    s = Ssse3.HorizontalAdd(s, s);
                    s = Ssse3.HorizontalAdd(s, s);
                    s = Ssse3.HorizontalAdd(s, s);
                    y[i].Bsums[j] = s.ToScalar();
                }
            }
            else
            {
                VecI8.Bsums16(qs, y[i].Bsums, Qk.SuperBlock);
            }
        }
    }

    /// <summary>
    /// llama.cpp Q8_K scale: absmax, then the first signed extremum (x86/repack.cpp q8_K_4x8).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    public static float ScaleBlock(float* src, out float delta)
    {
        float amax;
        float max;
        if (Simd.UseAvx)
        {
            Vector256<float> vacc = Vector256<float>.Zero;
            Vector256<float> sign = Vector256.Create(-0.0f);
            for (int j = 0; j < Qk.SuperBlock; j += 8)
                vacc = Avx.Max(vacc, Avx.AndNot(sign, Avx.LoadVector256(src + j)));
            Vector128<float> lane = Sse.Max(vacc.GetLower(), vacc.GetUpper());
            lane = Sse.Max(lane, Sse.MoveHighToLow(lane, lane));
            lane = Sse.Max(lane, Sse.Shuffle(lane, lane, 0x55));
            amax = lane.ToScalar();
            if (amax == 0)
            {
                delta = 0;
                return 0;
            }

            Vector256<float> pos = Vector256.Create(amax);
            Vector256<float> neg = Vector256.Create(-amax);
            int firstPos = Qk.SuperBlock;
            int firstNeg = Qk.SuperBlock;
            for (int j = 0; j < Qk.SuperBlock; j += 8)
            {
                Vector256<float> v = Avx.LoadVector256(src + j);
                int mp = Avx.MoveMask(Avx.CompareEqual(v, pos));
                int mn = Avx.MoveMask(Avx.CompareEqual(v, neg));
                if (mp != 0)
                    firstPos = Math.Min(firstPos, j + BitOperations.TrailingZeroCount(mp));
                if (mn != 0)
                    firstNeg = Math.Min(firstNeg, j + BitOperations.TrailingZeroCount(mn));
                if (Math.Min(firstPos, firstNeg) < j + 8)
                    break;
            }

            max = firstNeg < firstPos ? -amax : amax;
        }
        else
        {
            amax = VecF.AbsMax(src, Qk.SuperBlock);
            if (amax == 0)
            {
                delta = 0;
                return 0;
            }

            // First element attaining the absmax wins, matching the scalar loop.
            max = 0;
            for (int j = 0; j < Qk.SuperBlock; j++)
            {
                float ax = MathF.Abs(src[j]);
                if (ax == amax)
                {
                    max = src[j];
                    break;
                }
            }
        }

        float iscale = -127f / max;
        delta = 1f / iscale;
        return iscale;
    }

    public static int RowBytes(int k) => (k / Qk.SuperBlock) * Qk.Q8KSize;
}
