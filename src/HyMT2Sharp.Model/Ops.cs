using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Sdcb.HyMT2Sharp.Kernels;

namespace Sdcb.HyMT2Sharp.Model;

public static unsafe class Ops
{
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void RmsNorm(float* x, float* weight, float* y, int rows, int dim, float eps, CpuThreadPool? pool = null)
    {
        if (pool == null || rows <= 1)
        {
            RmsNormRange(x, weight, y, 0, rows, dim, eps);
            return;
        }

        pool.For(rows, (int worker, int workers) =>
        {
            int begin = rows * worker / workers;
            int end = rows * (worker + 1) / workers;
            RmsNormRange(x, weight, y, begin, end, dim, eps);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RmsNormRange(float* x, float* weight, float* y, int begin, int end, int dim, float eps)
    {
        for (int r = begin; r < end; r++)
        {
            float* src = x + r * dim;
            float* dst = y + r * dim;
            float sum = 0;
            int i = 0;
            if (Simd.UseAvx)
            {
                Vector256<float> acc = Vector256<float>.Zero;
                for (; i <= dim - 8; i += 8)
                {
                    Vector256<float> v = Avx.LoadVector256(src + i);
                    acc = Simd.UseFma ? Fma.MultiplyAdd(v, v, acc) : Avx.Add(acc, Avx.Multiply(v, v));
                }

                sum = VecDotQ4K.HorizontalSum(acc);
            }
            else
            {
                Vector<float> acc = Vector<float>.Zero;
                for (; i + Vector<float>.Count <= dim; i += Vector<float>.Count)
                {
                    Vector<float> v = VecF.Load(src + i);
                    acc += v * v;
                }

                sum = Vector.Sum(acc);
            }

            for (; i < dim; i++)
                sum += src[i] * src[i];
            float scale = 1f / MathF.Sqrt(sum / dim + eps);
            i = 0;
            if (Simd.UseAvx)
            {
                Vector256<float> s = Vector256.Create(scale);
                for (; i <= dim - 8; i += 8)
                {
                    Vector256<float> v = Avx.Multiply(Avx.LoadVector256(src + i), s);
                    if (weight != null)
                        v = Avx.Multiply(v, Avx.LoadVector256(weight + i));
                    Avx.Store(dst + i, v);
                }
            }
            else
            {
                Vector<float> s = new Vector<float>(scale);
                for (; i + Vector<float>.Count <= dim; i += Vector<float>.Count)
                {
                    Vector<float> v = VecF.Load(src + i) * s;
                    if (weight != null)
                        v *= VecF.Load(weight + i);
                    VecF.Store(dst + i, v);
                }
            }

            for (; i < dim; i++)
                dst[i] = src[i] * scale * (weight == null ? 1f : weight[i]);
        }
    }

    private static float[]? RopeCos;
    private static float[]? RopeSin;
    private static int RopeCap;
    private static int RopeHalf;
    private static float RopeBase;

    public static void NeoXRoPE(float* x, int tokens, int heads, int headDim, int ropeDim, int startPos, float baseFreq, CpuThreadPool? pool = null)
    {
        int half = ropeDim / 2;
        EnsureRopeTable(startPos + tokens, half, baseFreq);
        int rows = tokens * heads;
        float[] cos = RopeCos!;
        float[] sin = RopeSin!;
        if (pool == null || rows <= 16)
        {
            RopeRange(x, cos, sin, 0, rows, heads, headDim, half, startPos);
            return;
        }

        pool.For(rows, (int worker, int workers) =>
        {
            int begin = rows * worker / workers;
            int end = rows * (worker + 1) / workers;
            RopeRange(x, cos, sin, begin, end, heads, headDim, half, startPos);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void RopeRange(float* x, float[] cos, float[] sin, int begin, int end, int heads, int headDim, int half, int startPos)
    {
        fixed (float* cp = cos, sp = sin)
        {
        for (int r = begin; r < end; r++)
        {
            int t = r / heads;
            int h = r - t * heads;
            float* cRow = cp + (startPos + t) * half;
            float* sRow = sp + (startPos + t) * half;
            float* row = x + (t * heads + h) * headDim;
            int i = 0;
            if (Simd.UseAvx)
            {
                for (; i <= half - 8; i += 8)
                {
                    Vector256<float> x0 = Avx.LoadVector256(row + i);
                    Vector256<float> x1 = Avx.LoadVector256(row + i + half);
                    Vector256<float> cv = Avx.LoadVector256(cRow + i);
                    Vector256<float> sv = Avx.LoadVector256(sRow + i);
                    Avx.Store(row + i, Avx.Subtract(Avx.Multiply(x0, cv), Avx.Multiply(x1, sv)));
                    Avx.Store(row + i + half, Avx.Add(Avx.Multiply(x0, sv), Avx.Multiply(x1, cv)));
                }
            }
            else
            {
                for (; i + Vector<float>.Count <= half; i += Vector<float>.Count)
                {
                    Vector<float> x0 = VecF.Load(row + i);
                    Vector<float> x1 = VecF.Load(row + i + half);
                    Vector<float> cv = VecF.Load(cRow + i);
                    Vector<float> sv = VecF.Load(sRow + i);
                    VecF.Store(row + i, x0 * cv - x1 * sv);
                    VecF.Store(row + i + half, x0 * sv + x1 * cv);
                }
            }

            for (; i < half; i++)
            {
                float c = cRow[i];
                float s = sRow[i];
                float x0 = row[i];
                float x1 = row[i + half];
                row[i] = x0 * c - x1 * s;
                row[i + half] = x0 * s + x1 * c;
            }
        }
        }
    }

    private static void EnsureRopeTable(int needed, int half, float baseFreq)
    {
        if (RopeCos != null && RopeHalf == half && RopeBase == baseFreq && RopeCap >= needed)
            return;
        int cap = Math.Max(needed, 4096);
        float[] freqs = new float[half];
        for (int i = 0; i < half; i++)
            freqs[i] = 1f / MathF.Pow(baseFreq, (float)i / half);
        float[] cos = new float[cap * half];
        float[] sin = new float[cap * half];
        for (int pos = 0; pos < cap; pos++)
        {
            for (int i = 0; i < half; i++)
            {
                float angle = pos * freqs[i];
                cos[pos * half + i] = MathF.Cos(angle);
                sin[pos * half + i] = MathF.Sin(angle);
            }
        }

        RopeCos = cos;
        RopeSin = sin;
        RopeCap = cap;
        RopeHalf = half;
        RopeBase = baseFreq;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SoftmaxCausal(float* scores, int heads, int qLen, int kvLen, int startPos, CpuThreadPool? pool = null)
    {
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                for (int q = 0; q < qLen; q++)
                {
                    float* row = scores + (h * qLen + q) * kvLen;
                    // Only the visible prefix is normalised; the masked tail becomes exact zeros.
                    int allowed = Math.Min(kvLen, startPos + q + 1);
                    SoftmaxRow(row, allowed);
                    FillZero(row + allowed, kvLen - allowed);
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SoftmaxRow(float* row, int n)
    {
        int i = 0;
        float max;
        float sum;
        if (Simd.UseAvx && n >= 8)
        {
            Vector256<float> vmax = Vector256.Create(-80f);
            for (; i <= n - 8; i += 8)
                vmax = Avx.Max(vmax, Avx.LoadVector256(row + i));
            max = HorizontalMax(vmax);
            for (; i < n; i++)
                if (row[i] > max)
                    max = row[i];

            Vector256<float> vsum = Vector256<float>.Zero;
            Vector256<float> vmaxb = Vector256.Create(max);
            Vector256<float> floor = Vector256.Create(-80f);
            i = 0;
            for (; i <= n - 8; i += 8)
            {
                Vector256<float> shifted = Avx.Max(Avx.Subtract(Avx.LoadVector256(row + i), vmaxb), floor);
                Vector256<float> e = Simd.UseAvx2 && Simd.UseFma
                    ? FastExp.ExpAvx2(shifted)
                    : Vector256.Exp(shifted);
                Avx.Store(row + i, e);
                vsum = Avx.Add(vsum, e);
            }

            sum = VecDotQ4K.HorizontalSum(vsum);
            for (; i < n; i++)
            {
                float e = MathF.Exp(MathF.Max(row[i] - max, -80f));
                row[i] = e;
                sum += e;
            }
        }
        else if (n >= Vector<float>.Count)
        {
            Vector<float> vmax = new Vector<float>(-80f);
            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                vmax = Vector.Max(vmax, VecF.Load(row + i));
            max = vmax[0];
            for (int l = 1; l < Vector<float>.Count; l++)
                if (vmax[l] > max)
                    max = vmax[l];
            for (; i < n; i++)
                if (row[i] > max)
                    max = row[i];

            Vector<float> vsum = Vector<float>.Zero;
            Vector<float> vmaxb = new Vector<float>(max);
            Vector<float> floor = new Vector<float>(-80f);
            i = 0;
            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
            {
                Vector<float> e = FastExp.ExpVec(Vector.Max(VecF.Load(row + i) - vmaxb, floor));
                VecF.Store(row + i, e);
                vsum += e;
            }

            sum = Vector.Sum(vsum);
            for (; i < n; i++)
            {
                float e = MathF.Exp(MathF.Max(row[i] - max, -80f));
                row[i] = e;
                sum += e;
            }
        }
        else
        {
            max = -80f;
            for (i = 0; i < n; i++)
                if (row[i] > max)
                    max = row[i];
            sum = 0;
            for (i = 0; i < n; i++)
            {
                float e = MathF.Exp(MathF.Max(row[i] - max, -80f));
                row[i] = e;
                sum += e;
            }
        }

        float inv = sum > 0 ? 1f / sum : 0;
        i = 0;
        if (Simd.UseAvx)
        {
            Vector256<float> vinv = Vector256.Create(inv);
            for (; i <= n - 8; i += 8)
                Avx.Store(row + i, Avx.Multiply(Avx.LoadVector256(row + i), vinv));
        }
        else
        {
            Vector<float> vinv = new Vector<float>(inv);
            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                VecF.Store(row + i, VecF.Load(row + i) * vinv);
        }

        for (; i < n; i++)
            row[i] *= inv;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void FillZero(float* dst, int n)
    {
        int i = 0;
        if (Simd.UseAvx)
        {
            for (; i <= n - 8; i += 8)
                Avx.Store(dst + i, Vector256<float>.Zero);
        }
        else
        {
            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                VecF.Store(dst + i, Vector<float>.Zero);
        }

        for (; i < n; i++)
            dst[i] = 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HorizontalMax(Vector256<float> v)
    {
        Vector128<float> x = Sse.Max(v.GetLower(), v.GetUpper());
        x = Sse.Max(x, Sse.MoveHighToLow(x, x));
        x = Sse.Max(x, Sse.Shuffle(x, x, 0x55));
        return x.ToScalar();
    }

    /// <summary>
    /// scores[h, qt, kt] = scale · q[qt,h] · k[kt,kvh] for kt &lt; startPos + qt + 1 only;
    /// the masked tail is left untouched (pass startPos ≥ kvLen for a full matrix).
    /// </summary>
    public static void AttentionScores(
        float* q,
        ushort* cacheK,
        float* scores,
        int heads,
        int kvHeads,
        int headDim,
        int qLen,
        int kvLen,
        int qDim,
        int kvStride,
        float scale,
        int startPos,
        CpuThreadPool? pool = null)
    {
        int group = heads / kvHeads;
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                int kvh = h / group;
                ushort* kBase = cacheK + kvh * headDim;
                int qt = 0;
                if (Simd.UseAvx2 && Simd.UseFma && (headDim & 7) == 0)
                {
                    // Two q rows share each K load; row qt's extra (masked) score lands in the
                    // tail that SoftmaxCausal zeroes anyway.
                    for (; qt + 1 < qLen; qt += 2)
                    {
                        float* q0 = q + qt * qDim + h * headDim;
                        float* q1 = q0 + qDim;
                        float* row0 = scores + (h * qLen + qt) * kvLen;
                        float* row1 = row0 + kvLen;
                        int allowed = Math.Min(kvLen, startPos + qt + 2);
                        int kt = 0;
                        for (; kt + 3 < allowed; kt += 4)
                            Dot2x4Bf16(q0, q1, kBase + kt * kvStride, kvStride, headDim, row0 + kt, row1 + kt, scale);
                        for (; kt < allowed; kt++)
                        {
                            ushort* kh = kBase + kt * kvStride;
                            row0[kt] = DotBf16(q0, kh, headDim) * scale;
                            row1[kt] = DotBf16(q1, kh, headDim) * scale;
                        }
                    }
                }
                else if (!Simd.UseAvx && headDim % Vector<float>.Count == 0)
                {
                    for (; qt + 1 < qLen; qt += 2)
                    {
                        float* q0 = q + qt * qDim + h * headDim;
                        float* q1 = q0 + qDim;
                        float* row0 = scores + (h * qLen + qt) * kvLen;
                        float* row1 = row0 + kvLen;
                        int allowed = Math.Min(kvLen, startPos + qt + 2);
                        int kt = 0;
                        for (; kt + 3 < allowed; kt += 4)
                            Dot2x4Bf16Vec(q0, q1, kBase + kt * kvStride, kvStride, headDim, row0 + kt, row1 + kt, scale);
                        for (; kt < allowed; kt++)
                        {
                            ushort* kh = kBase + kt * kvStride;
                            row0[kt] = DotBf16(q0, kh, headDim) * scale;
                            row1[kt] = DotBf16(q1, kh, headDim) * scale;
                        }
                    }
                }

                for (; qt < qLen; qt++)
                {
                    float* qh = q + qt * qDim + h * headDim;
                    float* row = scores + (h * qLen + qt) * kvLen;
                    int allowed = Math.Min(kvLen, startPos + qt + 1);
                    int kt = 0;
                    for (; kt + 3 < allowed; kt += 4)
                    {
                        ushort* k0 = kBase + (kt + 0) * kvStride;
                        ushort* k1 = kBase + (kt + 1) * kvStride;
                        ushort* k2 = kBase + (kt + 2) * kvStride;
                        ushort* k3 = kBase + (kt + 3) * kvStride;
                        DotBf16x4(qh, k0, k1, k2, k3, headDim, row + kt, scale);
                    }

                    for (; kt < allowed; kt++)
                    {
                        ushort* kh = kBase + kt * kvStride;
                        row[kt] = DotBf16(qh, kh, headDim) * scale;
                    }
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }

    /// <summary>
    /// Single-token attention: scores → softmax → combine per head inside one
    /// parallel region, so decode pays one dispatch per layer and the score row
    /// stays cache-hot between phases.
    /// </summary>
    public static void AttentionDecode(
        float* q,
        ushort* cacheK,
        ushort* cacheV,
        float* scores,
        float* output,
        int heads,
        int kvHeads,
        int headDim,
        int kvLen,
        int kvStride,
        float scale,
        int startPos,
        CpuThreadPool? pool = null)
    {
        int group = heads / kvHeads;
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                int kvh = h / group;
                float* qh = q + h * headDim;
                ushort* kBase = cacheK + kvh * headDim;
                ushort* vBase = cacheV + kvh * headDim;
                float* row = scores + h * kvLen;
                int allowed = Math.Min(kvLen, startPos + 1);
                int kt = 0;
                for (; kt + 3 < allowed; kt += 4)
                {
                    ushort* k0 = kBase + (kt + 0) * kvStride;
                    ushort* k1 = kBase + (kt + 1) * kvStride;
                    ushort* k2 = kBase + (kt + 2) * kvStride;
                    ushort* k3 = kBase + (kt + 3) * kvStride;
                    DotBf16x4(qh, k0, k1, k2, k3, headDim, row + kt, scale);
                }

                for (; kt < allowed; kt++)
                    row[kt] = DotBf16(qh, kBase + kt * kvStride, headDim) * scale;

                SoftmaxRow(row, allowed);

                float* outH = output + h * headDim;
                int d = 0;
                if (Simd.UseAvx2 && Simd.UseFma)
                {
                    for (; d + 63 < headDim; d += 64)
                        Axpy64Bf16(row, vBase + d, kvStride, allowed, outH + d);
                }
                else if (!Simd.UseAvx)
                {
                    for (; d + 8 * Vector<float>.Count <= headDim; d += 8 * Vector<float>.Count)
                        AxpyRegVecBf16(row, vBase + d, kvStride, allowed, outH + d);
                }

                if (d < headDim)
                {
                    for (int i = d; i < headDim; i++)
                        outH[i] = 0;
                    for (int k = 0; k < allowed; k++)
                        AxpyBf16(outH + d, vBase + k * kvStride + d, row[k], headDim - d);
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }

    /// <summary>output[qt,h] = Σ_{kt &lt; startPos+qt+1} scores[h,qt,kt] · v[kt,kvh].</summary>
    public static void AttentionCombine(
        ushort* cacheV,
        float* scores,
        float* output,
        int heads,
        int kvHeads,
        int headDim,
        int qLen,
        int kvLen,
        int qDim,
        int kvStride,
        int startPos,
        CpuThreadPool? pool = null)
    {
        int group = heads / kvHeads;
        void Body(int worker, int workers)
        {
            int begin = heads * worker / workers;
            int end = heads * (worker + 1) / workers;
            for (int h = begin; h < end; h++)
            {
                int kvh = h / group;
                ushort* vBase = cacheV + kvh * headDim;
                for (int qt = 0; qt < qLen; qt++)
                {
                    float* row = scores + (h * qLen + qt) * kvLen;
                    float* outH = output + qt * qDim + h * headDim;
                    int allowed = Math.Min(kvLen, startPos + qt + 1);
                    int d = 0;
                    if (Simd.UseAvx2 && Simd.UseFma)
                    {
                        // 64 output dims live in 8 accumulators; V streams through once per half.
                        for (; d + 63 < headDim; d += 64)
                            Axpy64Bf16(row, vBase + d, kvStride, allowed, outH + d);
                    }
                    else if (!Simd.UseAvx)
                    {
                        for (; d + 8 * Vector<float>.Count <= headDim; d += 8 * Vector<float>.Count)
                            AxpyRegVecBf16(row, vBase + d, kvStride, allowed, outH + d);
                    }

                    if (d < headDim)
                    {
                        for (int i = d; i < headDim; i++)
                            outH[i] = 0;
                        for (int kt = 0; kt < allowed; kt++)
                        {
                            float w = row[kt];
                            ushort* vh = vBase + kt * kvStride;
                            AxpyBf16(outH + d, vh + d, w, headDim - d);
                        }
                    }
                }
            }
        }

        if (pool == null || heads <= 1)
            Body(0, 1);
        else
            pool.For(heads, Body);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotF32(float* a, float* b, int n)
    {
        int i = 0;
        float sum = 0;
        if (Simd.UseAvx)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            for (; i <= n - 16; i += 16)
            {
                acc0 = Simd.UseFma
                    ? Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0)
                    : Avx.Add(acc0, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
                acc1 = Simd.UseFma
                    ? Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8), acc1)
                    : Avx.Add(acc1, Avx.Multiply(Avx.LoadVector256(a + i + 8), Avx.LoadVector256(b + i + 8)));
            }

            for (; i <= n - 8; i += 8)
            {
                acc0 = Simd.UseFma
                    ? Fma.MultiplyAdd(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i), acc0)
                    : Avx.Add(acc0, Avx.Multiply(Avx.LoadVector256(a + i), Avx.LoadVector256(b + i)));
            }

            sum = VecDotQ4K.HorizontalSum(Avx.Add(acc0, acc1));
        }
        else
        {
            Vector<float> v0 = Vector<float>.Zero;
            Vector<float> v1 = Vector<float>.Zero;
            for (; i + 2 * Vector<float>.Count <= n; i += 2 * Vector<float>.Count)
            {
                v0 = Vector.MultiplyAddEstimate(VecF.Load(a + i), VecF.Load(b + i), v0);
                v1 = Vector.MultiplyAddEstimate(VecF.Load(a + i + Vector<float>.Count), VecF.Load(b + i + Vector<float>.Count), v1);
            }

            for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                v0 = Vector.MultiplyAddEstimate(VecF.Load(a + i), VecF.Load(b + i), v0);

            sum = Vector.Sum(v0 + v1);
        }

        for (; i < n; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>[Σa, Σb, Σc, Σd] with two hadd stages instead of four scalar reductions.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> HorizontalSum4(Vector256<float> a, Vector256<float> b, Vector256<float> c, Vector256<float> d)
    {
        Vector256<float> ab = Avx.HorizontalAdd(a, b);
        Vector256<float> cd = Avx.HorizontalAdd(c, d);
        Vector256<float> abcd = Avx.HorizontalAdd(ab, cd);
        return Sse.Add(abcd.GetLower(), abcd.GetUpper());
    }


    /// <summary>fp32→bf16 for K/V writes into the cache (round-to-nearest-even).</summary>
    public static void ConvertToBf16(float* src, ushort* dst, int n, CpuThreadPool? pool = null)
    {
        if (pool == null || n < 65536)
        {
            for (int i = 0; i < n; i++)
            {
                uint bits = Unsafe.ReadUnaligned<uint>(src + i);
                dst[i] = (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
            }
            return;
        }

        pool.For(n, (int worker, int workers) =>
        {
            int begin = n * worker / workers;
            int end = n * (worker + 1) / workers;
            for (int i = begin; i < end; i++)
            {
                uint bits = Unsafe.ReadUnaligned<uint>(src + i);
                dst[i] = (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1)) >> 16);
            }
        });
    }

    // bf16 = top 16 bits of fp32: decode is a u16→u32 widen + <<16, encode is
    // RNE truncation. bf16 keeps fp32's exponent range (no overflow possible)
    // at 8 mantissa bits — enough for K/V payloads. .NET exposes no vectorized
    // Half conversion intrinsics at all, so fp16 would need ~4× more ALU ops
    // here; bf16 decode is nearly free on every ISA.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float Bf16ToF32(ushort h) => BitConverter.UInt32BitsToSingle((uint)h << 16);

    /// <summary>Load 2V bf16 lanes as two fp32 vectors.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static (Vector<float> Lo, Vector<float> Hi) LoadBf16x2(ushort* p)
    {
        Vector<ushort> u = Unsafe.ReadUnaligned<Vector<ushort>>(p);
        Vector.Widen(u, out Vector<uint> lo, out Vector<uint> hi);
        return (Unsafe.BitCast<Vector<uint>, Vector<float>>(Vector.ShiftLeft(lo, 16)),
                Unsafe.BitCast<Vector<uint>, Vector<float>>(Vector.ShiftLeft(hi, 16)));
    }

    /// <summary>Load 8 bf16 lanes, widen to fp32 (AVX2: vpmovzxwd + vpslld).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> LoadBf16x8(ushort* p)
        => Vector256.ShiftLeft(Avx2.ConvertToVector256Int32(Unsafe.ReadUnaligned<Vector128<ushort>>(p)), 16).AsSingle();

    /// <summary>bf16-cache dot: q fp32 × one K row, bit-decoded on load.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotBf16(float* a, ushort* b, int n)
    {
        int i = 0;
        float sum = 0;
        if (Simd.UseAvx2 && Simd.UseFma)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            for (; i <= n - 16; i += 16)
            {
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), LoadBf16x8(b + i), acc0);
                acc1 = Fma.MultiplyAdd(Avx.LoadVector256(a + i + 8), LoadBf16x8(b + i + 8), acc1);
            }

            for (; i <= n - 8; i += 8)
                acc0 = Fma.MultiplyAdd(Avx.LoadVector256(a + i), LoadBf16x8(b + i), acc0);

            sum = VecDotQ4K.HorizontalSum(Avx.Add(acc0, acc1));
        }
        else
        {
            int V = Vector<float>.Count;
            Vector<float> v0 = Vector<float>.Zero;
            Vector<float> v1 = Vector<float>.Zero;
            for (; i + 2 * V <= n; i += 2 * V)
            {
                (Vector<float> lo, Vector<float> hi) = LoadBf16x2(b + i);
                v0 = Vector.MultiplyAddEstimate(VecF.Load(a + i), lo, v0);
                v1 = Vector.MultiplyAddEstimate(VecF.Load(a + i + V), hi, v1);
            }

            sum = Vector.Sum(v0 + v1);
        }

        for (; i < n; i++)
            sum += a[i] * Bf16ToF32(b[i]);
        return sum;
    }

    /// <summary>bf16-cache counterpart of the old fp32 4-row dot: q fp32 × four K rows widened on load.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void DotBf16x4(float* q, ushort* k0, ushort* k1, ushort* k2, ushort* k3, int n, float* dst, float scale)
    {
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int i = 0;
        if (Simd.UseAvx2 && Simd.UseFma)
        {
            Vector256<float> acc0 = Vector256<float>.Zero;
            Vector256<float> acc1 = Vector256<float>.Zero;
            Vector256<float> acc2 = Vector256<float>.Zero;
            Vector256<float> acc3 = Vector256<float>.Zero;
            for (; i <= n - 8; i += 8)
            {
                Vector256<float> qv = Avx.LoadVector256(q + i);
                acc0 = Fma.MultiplyAdd(qv, LoadBf16x8(k0 + i), acc0);
                acc1 = Fma.MultiplyAdd(qv, LoadBf16x8(k1 + i), acc1);
                acc2 = Fma.MultiplyAdd(qv, LoadBf16x8(k2 + i), acc2);
                acc3 = Fma.MultiplyAdd(qv, LoadBf16x8(k3 + i), acc3);
            }

            s0 = VecDotQ4K.HorizontalSum(acc0);
            s1 = VecDotQ4K.HorizontalSum(acc1);
            s2 = VecDotQ4K.HorizontalSum(acc2);
            s3 = VecDotQ4K.HorizontalSum(acc3);
        }
        else
        {
            Vector<float> acc0 = Vector<float>.Zero;
            Vector<float> acc1 = Vector<float>.Zero;
            Vector<float> acc2 = Vector<float>.Zero;
            Vector<float> acc3 = Vector<float>.Zero;
            int V = Vector<float>.Count;
            for (; i + 2 * V <= n; i += 2 * V)
            {
                Vector<float> q0 = VecF.Load(q + i);
                Vector<float> q1 = VecF.Load(q + i + V);
                (Vector<float> lo, Vector<float> hi) = LoadBf16x2(k0 + i);
                acc0 = Vector.MultiplyAddEstimate(q0, lo, acc0);
                acc0 = Vector.MultiplyAddEstimate(q1, hi, acc0);
                (lo, hi) = LoadBf16x2(k1 + i);
                acc1 = Vector.MultiplyAddEstimate(q0, lo, acc1);
                acc1 = Vector.MultiplyAddEstimate(q1, hi, acc1);
                (lo, hi) = LoadBf16x2(k2 + i);
                acc2 = Vector.MultiplyAddEstimate(q0, lo, acc2);
                acc2 = Vector.MultiplyAddEstimate(q1, hi, acc2);
                (lo, hi) = LoadBf16x2(k3 + i);
                acc3 = Vector.MultiplyAddEstimate(q0, lo, acc3);
                acc3 = Vector.MultiplyAddEstimate(q1, hi, acc3);
            }

            s0 = Vector.Sum(acc0);
            s1 = Vector.Sum(acc1);
            s2 = Vector.Sum(acc2);
            s3 = Vector.Sum(acc3);
        }

        for (; i < n; i++)
        {
            float qv = q[i];
            s0 += qv * Bf16ToF32(k0[i]);
            s1 += qv * Bf16ToF32(k1[i]);
            s2 += qv * Bf16ToF32(k2[i]);
            s3 += qv * Bf16ToF32(k3[i]);
        }

        dst[0] = s0 * scale;
        dst[1] = s1 * scale;
        dst[2] = s2 * scale;
        dst[3] = s3 * scale;
    }

    /// <summary>bf16-cache counterpart of the old fp32 2x4 dot (AVX2): two q rows share each decoded K load.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot2x4Bf16(float* q0, float* q1, ushort* k, int kvStride, int n, float* dst0, float* dst1, float scale)
    {
        ushort* k1 = k + kvStride;
        ushort* k2 = k1 + kvStride;
        ushort* k3 = k2 + kvStride;
        Vector256<float> a00 = Vector256<float>.Zero;
        Vector256<float> a01 = Vector256<float>.Zero;
        Vector256<float> a02 = Vector256<float>.Zero;
        Vector256<float> a03 = Vector256<float>.Zero;
        Vector256<float> a10 = Vector256<float>.Zero;
        Vector256<float> a11 = Vector256<float>.Zero;
        Vector256<float> a12 = Vector256<float>.Zero;
        Vector256<float> a13 = Vector256<float>.Zero;
        for (int i = 0; i < n; i += 8)
        {
            Vector256<float> v0 = LoadBf16x8(k + i);
            Vector256<float> v1 = LoadBf16x8(k1 + i);
            Vector256<float> v2 = LoadBf16x8(k2 + i);
            Vector256<float> v3 = LoadBf16x8(k3 + i);
            Vector256<float> x0 = Avx.LoadVector256(q0 + i);
            a00 = Fma.MultiplyAdd(x0, v0, a00);
            a01 = Fma.MultiplyAdd(x0, v1, a01);
            a02 = Fma.MultiplyAdd(x0, v2, a02);
            a03 = Fma.MultiplyAdd(x0, v3, a03);
            Vector256<float> x1 = Avx.LoadVector256(q1 + i);
            a10 = Fma.MultiplyAdd(x1, v0, a10);
            a11 = Fma.MultiplyAdd(x1, v1, a11);
            a12 = Fma.MultiplyAdd(x1, v2, a12);
            a13 = Fma.MultiplyAdd(x1, v3, a13);
        }

        Vector128<float> s = Vector128.Create(scale);
        Sse.Store(dst0, Sse.Multiply(HorizontalSum4(a00, a01, a02, a03), s));
        Sse.Store(dst1, Sse.Multiply(HorizontalSum4(a10, a11, a12, a13), s));
    }

    /// <summary>Portable counterpart of <see cref="Dot2x4Bf16"/> on <see cref="Vector{T}"/> integer decode.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot2x4Bf16Vec(float* q0, float* q1, ushort* k, int kvStride, int n, float* dst0, float* dst1, float scale)
    {
        ushort* k1 = k + kvStride;
        ushort* k2 = k1 + kvStride;
        ushort* k3 = k2 + kvStride;
        Vector<float> a00 = Vector<float>.Zero, a01 = Vector<float>.Zero, a02 = Vector<float>.Zero, a03 = Vector<float>.Zero;
        Vector<float> a10 = Vector<float>.Zero, a11 = Vector<float>.Zero, a12 = Vector<float>.Zero, a13 = Vector<float>.Zero;
        int V = Vector<float>.Count;
        for (int i = 0; i + 2 * V <= n; i += 2 * V)
        {
            (Vector<float> k0l, Vector<float> k0h) = LoadBf16x2(k + i);
            (Vector<float> k1l, Vector<float> k1h) = LoadBf16x2(k1 + i);
            (Vector<float> k2l, Vector<float> k2h) = LoadBf16x2(k2 + i);
            (Vector<float> k3l, Vector<float> k3h) = LoadBf16x2(k3 + i);
            Vector<float> x0 = VecF.Load(q0 + i);
            Vector<float> x1 = VecF.Load(q0 + i + V);
            a00 = Vector.MultiplyAddEstimate(x0, k0l, a00);
            a00 = Vector.MultiplyAddEstimate(x1, k0h, a00);
            a01 = Vector.MultiplyAddEstimate(x0, k1l, a01);
            a01 = Vector.MultiplyAddEstimate(x1, k1h, a01);
            a02 = Vector.MultiplyAddEstimate(x0, k2l, a02);
            a02 = Vector.MultiplyAddEstimate(x1, k2h, a02);
            a03 = Vector.MultiplyAddEstimate(x0, k3l, a03);
            a03 = Vector.MultiplyAddEstimate(x1, k3h, a03);
            Vector<float> y0 = VecF.Load(q1 + i);
            Vector<float> y1 = VecF.Load(q1 + i + V);
            a10 = Vector.MultiplyAddEstimate(y0, k0l, a10);
            a10 = Vector.MultiplyAddEstimate(y1, k0h, a10);
            a11 = Vector.MultiplyAddEstimate(y0, k1l, a11);
            a11 = Vector.MultiplyAddEstimate(y1, k1h, a11);
            a12 = Vector.MultiplyAddEstimate(y0, k2l, a12);
            a12 = Vector.MultiplyAddEstimate(y1, k2h, a12);
            a13 = Vector.MultiplyAddEstimate(y0, k3l, a13);
            a13 = Vector.MultiplyAddEstimate(y1, k3h, a13);
        }

        dst0[0] = Vector.Sum(a00) * scale;
        dst0[1] = Vector.Sum(a01) * scale;
        dst0[2] = Vector.Sum(a02) * scale;
        dst0[3] = Vector.Sum(a03) * scale;
        dst1[0] = Vector.Sum(a10) * scale;
        dst1[1] = Vector.Sum(a11) * scale;
        dst1[2] = Vector.Sum(a12) * scale;
        dst1[3] = Vector.Sum(a13) * scale;
    }

    /// <summary>bf16-cache counterpart of the old fp32 64-dim axpy (AVX2): 64 output dims, 8 accumulators.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Axpy64Bf16(float* w, ushort* v, int kvStride, int count, float* dst)
    {
        Vector256<float> a0 = Vector256<float>.Zero;
        Vector256<float> a1 = Vector256<float>.Zero;
        Vector256<float> a2 = Vector256<float>.Zero;
        Vector256<float> a3 = Vector256<float>.Zero;
        Vector256<float> a4 = Vector256<float>.Zero;
        Vector256<float> a5 = Vector256<float>.Zero;
        Vector256<float> a6 = Vector256<float>.Zero;
        Vector256<float> a7 = Vector256<float>.Zero;
        for (int kt = 0; kt < count; kt++)
        {
            Vector256<float> wv = Avx.BroadcastScalarToVector256(w + kt);
            ushort* row = v + kt * kvStride;
            a0 = Fma.MultiplyAdd(wv, LoadBf16x8(row), a0);
            a1 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 8), a1);
            a2 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 16), a2);
            a3 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 24), a3);
            a4 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 32), a4);
            a5 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 40), a5);
            a6 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 48), a6);
            a7 = Fma.MultiplyAdd(wv, LoadBf16x8(row + 56), a7);
        }

        Avx.Store(dst, a0);
        Avx.Store(dst + 8, a1);
        Avx.Store(dst + 16, a2);
        Avx.Store(dst + 24, a3);
        Avx.Store(dst + 32, a4);
        Avx.Store(dst + 40, a5);
        Avx.Store(dst + 48, a6);
        Avx.Store(dst + 56, a7);
    }

    /// <summary>Portable counterpart of <see cref="Axpy64Bf16"/> on <see cref="Vector{T}"/> integer decode.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AxpyRegVecBf16(float* w, ushort* v, int kvStride, int count, float* dst)
    {
        int V = Vector<float>.Count;
        Vector<float> a0 = Vector<float>.Zero, a1 = Vector<float>.Zero, a2 = Vector<float>.Zero, a3 = Vector<float>.Zero;
        Vector<float> a4 = Vector<float>.Zero, a5 = Vector<float>.Zero, a6 = Vector<float>.Zero, a7 = Vector<float>.Zero;
        for (int kt = 0; kt < count; kt++)
        {
            Vector<float> wv = new(w[kt]);
            ushort* row = v + kt * kvStride;
            (Vector<float> lo, Vector<float> hi) = LoadBf16x2(row);
            a0 = Vector.MultiplyAddEstimate(wv, lo, a0);
            a1 = Vector.MultiplyAddEstimate(wv, hi, a1);
            (lo, hi) = LoadBf16x2(row + 2 * V);
            a2 = Vector.MultiplyAddEstimate(wv, lo, a2);
            a3 = Vector.MultiplyAddEstimate(wv, hi, a3);
            (lo, hi) = LoadBf16x2(row + 4 * V);
            a4 = Vector.MultiplyAddEstimate(wv, lo, a4);
            a5 = Vector.MultiplyAddEstimate(wv, hi, a5);
            (lo, hi) = LoadBf16x2(row + 6 * V);
            a6 = Vector.MultiplyAddEstimate(wv, lo, a6);
            a7 = Vector.MultiplyAddEstimate(wv, hi, a7);
        }

        VecF.Store(dst, a0);
        VecF.Store(dst + V, a1);
        VecF.Store(dst + 2 * V, a2);
        VecF.Store(dst + 3 * V, a3);
        VecF.Store(dst + 4 * V, a4);
        VecF.Store(dst + 5 * V, a5);
        VecF.Store(dst + 6 * V, a6);
        VecF.Store(dst + 7 * V, a7);
    }

    /// <summary>Scalar tail of the bf16 axpy: y[i] += a·x[i], x decoded per element.</summary>
    private static void AxpyBf16(float* y, ushort* x, float a, int n)
    {
        for (int i = 0; i < n; i++)
            y[i] += a * Bf16ToF32(x[i]);
    }

    public static void SiLUMul(float* gate, float* up, int n, CpuThreadPool? pool = null)
    {
        if (pool == null || n < 65536)
        {
            SiLUMulRange(gate, up, 0, n);
            return;
        }

        pool.For(n, (int worker, int workers) =>
        {
            int begin = n * worker / workers;
            int end = n * (worker + 1) / workers;
            SiLUMulRange(gate, up, begin, end);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void SiLUMulRange(float* gate, float* up, int begin, int end)
    {
        int i = begin;
        if (Simd.UseAvx2 && Simd.UseFma)
        {
            for (; i <= end - 16; i += 16)
            {
                Vector256<float> g0 = Avx.LoadVector256(gate + i);
                Vector256<float> g1 = Avx.LoadVector256(gate + i + 8);
                Avx.Store(gate + i, Avx.Multiply(FastExp.SiluAvx2(g0), Avx.LoadVector256(up + i)));
                Avx.Store(gate + i + 8, Avx.Multiply(FastExp.SiluAvx2(g1), Avx.LoadVector256(up + i + 8)));
            }

            for (; i <= end - 8; i += 8)
                Avx.Store(gate + i, Avx.Multiply(FastExp.SiluAvx2(Avx.LoadVector256(gate + i)), Avx.LoadVector256(up + i)));
        }
        else if (Simd.UseAvx)
        {
            Vector256<float> one = Vector256.Create(1f);
            Vector256<float> sign = Vector256.Create(-0.0f);
            for (; i <= end - 8; i += 8)
            {
                Vector256<float> g = Avx.LoadVector256(gate + i);
                Vector256<float> u = Avx.LoadVector256(up + i);
                Vector256<float> silu = Avx.Divide(g, Avx.Add(one, Vector256.Exp(Avx.Xor(g, sign))));
                Avx.Store(gate + i, Avx.Multiply(silu, u));
            }
        }
        else
        {
            for (; i + Vector<float>.Count <= end; i += Vector<float>.Count)
                VecF.Store(gate + i, FastExp.SiluVec(VecF.Load(gate + i)) * VecF.Load(up + i));
        }

        for (; i < end; i++)
        {
            float g = gate[i];
            gate[i] = g / (1f + MathF.Exp(-g)) * up[i];
        }
    }

    public static void AddInPlace(float* dest, float* src, int n, CpuThreadPool? pool = null)
    {
        if (pool == null || n < 65536)
        {
            AddInPlaceRange(dest, src, 0, n);
            return;
        }

        pool.For(n, (int worker, int workers) =>
        {
            int begin = n * worker / workers;
            int end = n * (worker + 1) / workers;
            AddInPlaceRange(dest, src, begin, end);
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void AddInPlaceRange(float* dest, float* src, int begin, int end)
    {
        int i = begin;
        if (Simd.UseAvx)
        {
            for (; i <= end - 16; i += 16)
            {
                Avx.Store(dest + i, Avx.Add(Avx.LoadVector256(dest + i), Avx.LoadVector256(src + i)));
                Avx.Store(dest + i + 8, Avx.Add(Avx.LoadVector256(dest + i + 8), Avx.LoadVector256(src + i + 8)));
            }

            for (; i <= end - 8; i += 8)
                Avx.Store(dest + i, Avx.Add(Avx.LoadVector256(dest + i), Avx.LoadVector256(src + i)));
        }
        else
        {
            for (; i + Vector<float>.Count <= end; i += Vector<float>.Count)
                VecF.Store(dest + i, VecF.Load(dest + i) + VecF.Load(src + i));
        }

        for (; i < end; i++)
            dest[i] += src[i];
    }
}
