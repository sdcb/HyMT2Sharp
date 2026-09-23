using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Q8_0 dequant, activation quant, and GEMV. Prefill GEMM lives in <see cref="GemmQ8_0"/>.
/// Dots use <c>vpdpbusd</c> when <see cref="AvxVnni.IsSupported"/>, otherwise AVX2 <c>vpmaddubsw</c>.
/// </summary>
public static unsafe class Q8_0
{
    public static void DequantizeRow(BlockQ8_0* x, float* y, int k)
    {
        int nb = k / Qk.Q8_0Block;
        for (int i = 0; i < nb; i++)
        {
            float d = HalfBits.ToSingle(x[i].D);
            sbyte* qs = x[i].Qs;
            float* dst = y + i * Qk.Q8_0Block;
            if (Avx2.IsSupported && Sse41.IsSupported)
            {
                for (int j = 0; j < Qk.Q8_0Block; j += 8)
                    Store8(qs + j, dst + j, d);
            }
            else
            {
                for (int j = 0; j < Qk.Q8_0Block; j++)
                    dst[j] = d * qs[j];
            }
        }
    }

    public static void QuantizeActs(float* x, BlockQ8_0Act* y, int k)
    {
        int nb = k / Qk.Q8_0Block;
        for (int i = 0; i < nb; i++)
        {
            Quantize32(x + i * Qk.Q8_0Block, y[i].Qs, out float d);
            y[i].D = d;
            int sum = 0;
            for (int j = 0; j < Qk.Q8_0Block; j++)
                sum += y[i].Qs[j];
            y[i].Sum = sum;
        }
    }

    public static void Quantize4(float* src, BlockQ8_0x4* y, int k)
    {
        int nb = k / Qk.Q8_0Block;
        for (int i = 0; i < nb; i++)
        {
            for (int r = 0; r < 4; r++)
            {
                sbyte* qs = y[i].Qs + r * Qk.Q8_0Block;
                Quantize32(src + r * k + i * Qk.Q8_0Block, qs, out float d);
                y[i].D[r] = d;
                int sum = 0;
                for (int j = 0; j < Qk.Q8_0Block; j++)
                    sum += qs[j];
                y[i].Bias[r] = sum << 7;
            }
        }
    }

    public static void Quantize4Silu(float* gate, float* up, BlockQ8_0x4* y, int k)
    {
        float* tmp = stackalloc float[4 * Qk.Q8_0Block];
        int nb = k / Qk.Q8_0Block;
        for (int i = 0; i < nb; i++)
        {
            for (int row = 0; row < 4; row++)
                Silu32(gate + row * k + i * Qk.Q8_0Block, up + row * k + i * Qk.Q8_0Block, tmp + row * Qk.Q8_0Block);
            Quantize4(tmp, y + i, Qk.Q8_0Block);
        }
    }

    public static float DotScalar(BlockQ8_0* weight, BlockQ8_0Act* act, int n)
    {
        int nb = n / Qk.Q8_0Block;
        float sum = 0;
        for (int i = 0; i < nb; i++)
        {
            int acc = 0;
            for (int k = 0; k < Qk.Q8_0Block; k++)
                acc += weight[i].Qs[k] * act[i].Qs[k];
            sum += HalfBits.ToSingle(weight[i].D) * act[i].D * acc;
        }

        return sum;
    }

    public static void Gemv(BlockQ8_0* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        int nb = nIn / Qk.Q8_0Block;
        nuint bytes = (nuint)nb * (nuint)Qk.Q8_0ActSize;
        NativeBuffer? owned = scratch == null ? new NativeBuffer(bytes) : null;
        BlockQ8_0Act* act = (BlockQ8_0Act*)(scratch != null ? scratch.D(bytes) : owned!.Pointer);
        try
        {
            QuantizeActs(input, act, nIn);
            GemvPrequant(weights, act, output, nIn, nOut, pool);
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void GemvPrequant(BlockQ8_0* weights, BlockQ8_0Act* act, float* output, int nIn, int nOut, CpuThreadPool? pool)
    {
        void Body(int worker, int workers)
        {
            int begin = nOut * worker / workers;
            int end = nOut * (worker + 1) / workers;
            int row = begin;
            if (Avx2.IsSupported)
            {
                for (; row + 3 < end; row += 4)
                    Dot4Rows(weights, act, output, nIn, row);
            }

            int nb = nIn / Qk.Q8_0Block;
            for (; row < end; row++)
                output[row] = DotScalar(weights + row * nb, act, nIn);
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(nOut, Body);
    }

    /// <summary>
    /// GEMV over the 8-column panel. Weight bytes are read as one sequential stream;
    /// tail columns that never made the panel fall back to <see cref="DotScalar"/> rows.
    /// </summary>
    public static void GemvPacked(BlockQ8_0x8* packed, BlockQ8_0* rows, BlockQ8_0Act* act, float* output, int nIn, int nOut, CpuThreadPool? pool)
    {
        int nb = nIn / Qk.Q8_0Block;
        int groups = nOut / 8;
        int next = 0;
        void Body(int worker, int workers)
        {
            // Dynamic chunks: a worker stuck on a slow memory patch takes fewer
            // panels instead of gating the whole GEMV (ggml does the same).
            // Aim for ~16 claims per worker so each claim is a long contiguous
            // stream (DRAM row locality) while tails still rebalance.
            int chunk = Math.Max(4, groups / (workers * 16));
            int g;
            while ((g = Interlocked.Add(ref next, chunk) - chunk) < groups)
            {
                int gEnd = Math.Min(g + chunk, groups);
                for (; g < gEnd; g++)
                {
                    BlockQ8_0x8* w = packed + g * nb;
                    if (AvxVnni.IsSupported)
                        PackedGroupVnni(w, act, output + g * 8, nb);
                    else if (Avx2.IsSupported)
                        PackedGroupAvx2(w, act, output + g * 8, nb);
                    else
                        PackedGroupScalar(w, act, output + g * 8, nb);
                }
            }

            if (worker == workers - 1)
            {
                for (int row = groups * 8; row < nOut; row++)
                    output[row] = DotScalar(rows + row * nb, act, nIn);
            }
        }

        if (pool == null || groups == 0)
            Body(0, 1);
        else
            pool.For(groups, Body);
    }

    /// <summary>
    /// One parallel region covering several weights that share the same quantized
    /// activation row. Work items are 8-column groups across all weights, so a
    /// decode layer pays one dispatch instead of one per weight.
    /// </summary>
    public static void GemvPackedMulti(BlockQ8_0Act* act, int nIn, CpuThreadPool? pool, Q8GemvTarget t0, Q8GemvTarget t1) =>
        GemvPackedMulti(act, nIn, pool, t0, t1, default);

    public static void GemvPackedMulti(BlockQ8_0Act* act, int nIn, CpuThreadPool? pool, Q8GemvTarget t0, Q8GemvTarget t1, Q8GemvTarget t2)
    {
        int nb = nIn / Qk.Q8_0Block;
        int g0t = t0.NOut / 8;
        int g1t = t1.NOut / 8;
        int g2t = t2.Dst == null ? 0 : t2.NOut / 8;
        int total = g0t + g1t + g2t;

        void Group(Q8GemvTarget t, int g)
        {
            if (t.Packed == null)
            {
                for (int row = g * 8; row < g * 8 + 8; row++)
                    t.Dst[row] = DotScalar(t.Rows + row * nb, act, nIn);
                return;
            }

            BlockQ8_0x8* w = t.Packed + g * nb;
            if (AvxVnni.IsSupported)
                PackedGroupVnni(w, act, t.Dst + g * 8, nb);
            else if (Avx2.IsSupported)
                PackedGroupAvx2(w, act, t.Dst + g * 8, nb);
            else
                PackedGroupScalar(w, act, t.Dst + g * 8, nb);
        }

        int next = 0;
        void Body(int worker, int workers)
        {
            // Dynamic chunks smooth out per-worker memory speed differences;
            // ~16 claims per worker keeps each claim a long contiguous stream.
            int chunk = Math.Max(4, total / (workers * 16));
            int begin;
            while ((begin = Interlocked.Add(ref next, chunk) - chunk) < total)
            {
                int end = Math.Min(begin + chunk, total);
                int g0 = Math.Clamp(begin, 0, g0t);
                int g1 = Math.Clamp(end, 0, g0t);
                for (int g = g0; g < g1; g++)
                    Group(t0, g);
                g0 = Math.Clamp(begin - g0t, 0, g1t);
                g1 = Math.Clamp(end - g0t, 0, g1t);
                for (int g = g0; g < g1; g++)
                    Group(t1, g);
                g0 = Math.Clamp(begin - g0t - g1t, 0, g2t);
                g1 = Math.Clamp(end - g0t - g1t, 0, g2t);
                for (int g = g0; g < g1; g++)
                    Group(t2, g);
            }

            if (worker == workers - 1)
            {
                Tail(t0);
                Tail(t1);
                if (t2.Dst != null)
                    Tail(t2);
            }
        }

        void Tail(Q8GemvTarget t)
        {
            for (int row = t.NOut & ~7; row < t.NOut; row++)
                t.Dst[row] = DotScalar(t.Rows + row * nb, act, nIn);
        }

        if (pool == null || total == 0)
            Body(0, 1);
        else
            pool.For(total, Body);
    }

    private static void PackedGroupVnni(BlockQ8_0x8* w, BlockQ8_0Act* act, float* dst, int nb)
    {
        Vector256<byte> bias = Vector256.Create((byte)0x80);
        Vector256<float> acc = Vector256<float>.Zero;
        for (int b = 0; b < nb; b++)
        {
            BlockQ8_0x8* wb = w + b;
            Vector256<int> j = Vector256<int>.Zero;
            sbyte* aq = act[b].Qs;
            byte* q = wb->Qs;
            for (int step = 0; step < 8; step++)
            {
                Vector256<byte> u = Avx2.Xor(Avx.LoadVector256(q + step * 32), bias);
                j = AvxVnni.MultiplyWideningAndAdd(j, u, Avx2.BroadcastScalarToVector256((int*)(aq + step * 4)).AsSByte());
            }

            j = Avx2.Subtract(j, Vector256.Create(act[b].Sum << 7));
            Vector256<float> d = Avx.Multiply(Avx.LoadVector256(wb->D), Vector256.Create(act[b].D));
            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(j), d, acc)
                : Avx.Add(acc, Avx.Multiply(Avx.ConvertToVector256Single(j), d));
        }

        Avx.Store(dst, acc);
    }

    private static void PackedGroupAvx2(BlockQ8_0x8* w, BlockQ8_0Act* act, float* dst, int nb)
    {
        Vector256<short> ones = Vector256<short>.One;
        Vector256<float> acc = Vector256<float>.Zero;
        for (int b = 0; b < nb; b++)
        {
            BlockQ8_0x8* wb = w + b;
            Vector256<int> j = Vector256<int>.Zero;
            sbyte* aq = act[b].Qs;
            byte* q = wb->Qs;
            for (int step = 0; step < 8; step++)
            {
                Vector256<sbyte> ws = Avx.LoadVector256(q + step * 32).AsSByte();
                Vector256<sbyte> a = Avx2.BroadcastScalarToVector256((int*)(aq + step * 4)).AsSByte();
                j = Avx2.Add(j, Avx2.MultiplyAddAdjacent(
                    Avx2.MultiplyAddAdjacent(Avx2.Sign(ws, ws).AsByte(), Avx2.Sign(a, ws)), ones));
            }

            Vector256<float> d = Avx.Multiply(Avx.LoadVector256(wb->D), Vector256.Create(act[b].D));
            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(Avx.ConvertToVector256Single(j), d, acc)
                : Avx.Add(acc, Avx.Multiply(Avx.ConvertToVector256Single(j), d));
        }

        Avx.Store(dst, acc);
    }

    private static void PackedGroupScalar(BlockQ8_0x8* w, BlockQ8_0Act* act, float* dst, int nb)
    {
        for (int c = 0; c < 8; c++)
        {
            float sum = 0;
            for (int b = 0; b < nb; b++)
            {
                int acc = 0;
                for (int k = 0; k < Qk.Q8_0Block; k++)
                    acc += (sbyte)w[b].Qs[(k >> 2) * 32 + c * 4 + (k & 3)] * act[b].Qs[k];
                sum += w[b].D[c] * act[b].D * acc;
            }

            dst[c] = sum;
        }
    }

    public static void Gemm(BlockQ8_0* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        if (tokens == 1)
        {
            Gemv(weights, input, output, nIn, nOut, pool, scratch);
            return;
        }

        int nb = nIn / Qk.Q8_0Block;
        nuint actBytes = (nuint)((long)tokens * nb * Qk.Q8_0ActSize);
        NativeBuffer? owned = scratch == null ? new NativeBuffer(actBytes) : null;
        BlockQ8_0Act* act = (BlockQ8_0Act*)(scratch != null ? scratch.A(actBytes) : owned!.Pointer);
        try
        {
            void QuantBody(int worker, int workers)
            {
                int t0 = tokens * worker / workers;
                int t1 = tokens * (worker + 1) / workers;
                for (int t = t0; t < t1; t++)
                    QuantizeActs(input + t * nIn, act + t * nb, nIn);
            }

            if (pool == null)
                QuantBody(0, 1);
            else
                pool.For(tokens, QuantBody);

            void Body(int worker, int workers)
            {
                int begin = nOut * worker / workers;
                int end = nOut * (worker + 1) / workers;
                for (int row = begin; row < end; row++)
                {
                    for (int t = 0; t < tokens; t++)
                        output[t * nOut + row] = DotScalar(weights + row * nb, act + t * nb, nIn);
                }
            }

            if (pool == null)
                Body(0, 1);
            else
                pool.For(nOut, Body);
        }
        finally
        {
            owned?.Dispose();
        }
    }

    /// <summary>32 signed weights × 32 signed activations. AVX2 sign trick, no int16 saturation for |q|≤128.</summary>
    public static int Dot32Avx2(sbyte* weight, sbyte* act)
    {
        Vector256<sbyte> w = Avx.LoadVector256(weight);
        Vector256<sbyte> a = Avx.LoadVector256(act);
        Vector256<int> lo = Avx2.MultiplyAddAdjacent(
            Avx2.ConvertToVector256Int16(w.GetLower()),
            Avx2.ConvertToVector256Int16(a.GetLower()));
        Vector256<int> hi = Avx2.MultiplyAddAdjacent(
            Avx2.ConvertToVector256Int16(w.GetUpper()),
            Avx2.ConvertToVector256Int16(a.GetUpper()));
        return HorizontalSum(Avx2.Add(lo, hi));
    }

    public static int Dot32Vnni(sbyte* weight, sbyte* act, int actSum)
    {
        Vector256<byte> u = Avx2.Xor(Avx.LoadVector256((byte*)weight), Vector256.Create((byte)0x80));
        Vector256<int> dots = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, u, Avx.LoadVector256(act));
        return HorizontalSum(dots) - (actSum << 7);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int HorizontalSum(Vector256<int> v)
    {
        Vector128<int> s = Sse2.Add(v.GetLower(), v.GetUpper());
        s = Sse2.Add(s, Sse2.Shuffle(s, 0x4E));
        s = Sse2.Add(s, Sse2.Shuffle(s, 0xB1));
        return s.ToScalar();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot32(sbyte* weight, sbyte* act, int actSum) =>
        AvxVnni.IsSupported ? Dot32Vnni(weight, act, actSum) : Dot32Avx2(weight, act);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot4Rows(BlockQ8_0* weights, BlockQ8_0Act* act, float* output, int nIn, int row)
    {
        int nb = nIn / Qk.Q8_0Block;
        BlockQ8_0* r0 = weights + row * nb;
        BlockQ8_0* r1 = r0 + nb;
        BlockQ8_0* r2 = r1 + nb;
        BlockQ8_0* r3 = r2 + nb;
        float a0 = 0;
        float a1 = 0;
        float a2 = 0;
        float a3 = 0;
        for (int b = 0; b < nb; b++)
        {
            sbyte* aq = act[b].Qs;
            int sum = act[b].Sum;
            float da = act[b].D;
            a0 += da * HalfBits.ToSingle(r0[b].D) * Dot32(r0[b].Qs, aq, sum);
            a1 += da * HalfBits.ToSingle(r1[b].D) * Dot32(r1[b].Qs, aq, sum);
            a2 += da * HalfBits.ToSingle(r2[b].D) * Dot32(r2[b].Qs, aq, sum);
            a3 += da * HalfBits.ToSingle(r3[b].D) * Dot32(r3[b].Qs, aq, sum);
        }

        output[row] = a0;
        output[row + 1] = a1;
        output[row + 2] = a2;
        output[row + 3] = a3;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Quantize32(float* src, sbyte* qs, out float delta)
    {
        float amax = AbsMax32(src);
        if (amax == 0)
        {
            delta = 0;
            NativeMemory.Clear(qs, Qk.Q8_0Block);
            return;
        }

        float id = 127f / amax;
        delta = amax / 127f;
        if (Avx2.IsSupported)
        {
            Vector256<float> idv = Vector256.Create(id);
            Vector256<int> lo = Vector256.Create(-127);
            Vector256<int> hi = Vector256.Create(127);
            for (int j = 0; j < Qk.Q8_0Block; j += 8)
            {
                Vector256<float> rounded = Avx.RoundToNearestInteger(Avx.Multiply(Avx.LoadVector256(src + j), idv));
                Vector256<int> qi = Avx2.Min(Avx2.Max(Avx.ConvertToVector256Int32(rounded), lo), hi);
                Vector128<short> p16 = Sse2.PackSignedSaturate(qi.GetLower(), qi.GetUpper());
                Vector128<sbyte> p8 = Sse2.PackSignedSaturate(p16, Vector128<short>.Zero);
                Unsafe.WriteUnaligned(qs + j, p8.AsInt64().ToScalar());
            }
        }
        else
        {
            for (int j = 0; j < Qk.Q8_0Block; j++)
            {
                int v = (int)MathF.Round(id * src[j]);
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                qs[j] = (sbyte)v;
            }
        }
    }

    private static float AbsMax32(float* src)
    {
        if (Avx.IsSupported)
        {
            Vector256<float> sign = Vector256.Create(-0.0f);
            Vector256<float> vacc = Avx.AndNot(sign, Avx.LoadVector256(src));
            vacc = Avx.Max(vacc, Avx.AndNot(sign, Avx.LoadVector256(src + 8)));
            vacc = Avx.Max(vacc, Avx.AndNot(sign, Avx.LoadVector256(src + 16)));
            vacc = Avx.Max(vacc, Avx.AndNot(sign, Avx.LoadVector256(src + 24)));
            Vector128<float> lane = Sse.Max(vacc.GetLower(), vacc.GetUpper());
            lane = Sse.Max(lane, Sse.MoveHighToLow(lane, lane));
            lane = Sse.Max(lane, Sse.Shuffle(lane, lane, 0x55));
            return lane.ToScalar();
        }

        float amax = 0;
        for (int j = 0; j < Qk.Q8_0Block; j++)
            amax = MathF.Max(amax, MathF.Abs(src[j]));
        return amax;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store8(sbyte* qs, float* dst, float d)
    {
        Vector128<sbyte> q = Vector128.CreateScalar(Unsafe.ReadUnaligned<long>(qs)).AsSByte();
        Vector128<int> lo = Sse41.ConvertToVector128Int32(q);
        Vector128<byte> shuf = Vector128.Create((byte)4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        Vector128<int> hi = Sse41.ConvertToVector128Int32(Ssse3.Shuffle(q.AsByte(), shuf).AsSByte());
        Vector256<float> f = Avx.Multiply(
            Vector256.Create(Sse2.ConvertToVector128Single(lo), Sse2.ConvertToVector128Single(hi)),
            Vector256.Create(d));
        Avx.Store(dst, f);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Silu32(float* gate, float* up, float* dest)
    {
        if (Avx2.IsSupported && Fma.IsSupported)
        {
            for (int j = 0; j < Qk.Q8_0Block; j += 8)
                Avx.Store(dest + j, Avx.Multiply(FastExp.SiluAvx2(Avx.LoadVector256(gate + j)), Avx.LoadVector256(up + j)));
            return;
        }

        for (int j = 0; j < Qk.Q8_0Block; j++)
        {
            float g = gate[j];
            float s = g / (1f + MathF.Exp(-g));
            dest[j] = s * up[j];
        }
    }
}
