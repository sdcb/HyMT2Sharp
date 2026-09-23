using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Sdcb.HyMT2Sharp.Kernels;

public static unsafe class Q6K
{
    public static void DequantizeRow(BlockQ6K* x, float* y, int k)
    {
        int nb = k / Qk.SuperBlock;
        for (int i = 0; i < nb; i++)
        {
            float d = HalfBits.ToSingle(x[i].D);
            byte* ql = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* sc = x[i].Scales;
            for (int n = 0; n < Qk.SuperBlock; n += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    int iscale = l / 16;
                    int q1 = (ql[l] & 0xF) | ((qh[l] & 3) << 4);
                    int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                    int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                    int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                    y[l] = d * sc[iscale] * (q1 - 32);
                    y[l + 32] = d * sc[iscale + 2] * (q2 - 32);
                    y[l + 64] = d * sc[iscale + 4] * (q3 - 32);
                    y[l + 96] = d * sc[iscale + 6] * (q4 - 32);
                }

                y += 128;
                ql += 64;
                qh += 32;
                sc += 8;
            }
        }
    }

    public static float Dot(BlockQ6K* x, BlockQ8K* y, int n)
        => Avx2.IsSupported ? DotAvx2(x, y, n) : DotScalar(x, y, n);

    public static float DotScalar(BlockQ6K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        sbyte* aux = stackalloc sbyte[Qk.SuperBlock];
        float sumf = 0;
        for (int i = 0; i < nb; i++)
        {
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* a = aux;
            for (int j = 0; j < Qk.SuperBlock; j += 128)
            {
                for (int l = 0; l < 32; l++)
                {
                    a[l] = (sbyte)(((q4[l] & 0xF) | ((qh[l] & 3) << 4)) - 32);
                    a[l + 32] = (sbyte)(((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                    a[l + 64] = (sbyte)(((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                    a[l + 96] = (sbyte)(((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                }

                a += 128;
                q4 += 64;
                qh += 32;
            }

            int acc = 0;
            a = aux;
            sbyte* q8 = y[i].Qs;
            for (int j = 0; j < Qk.SuperBlock / 16; j++)
            {
                int scale = x[i].Scales[j];
                for (int l = 0; l < 16; l++)
                    acc += scale * q8[l] * a[l];
                q8 += 16;
                a += 16;
            }

            sumf += HalfBits.ToSingle(x[i].D) * y[i].D * acc;
        }

        return sumf;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotAvx2(BlockQ6K* x, BlockQ8K* y, int n)
    {
        int nb = n / Qk.SuperBlock;
        Vector256<byte> m3 = Vector256.Create((byte)3);
        Vector256<byte> m15 = Vector256.Create((byte)15);
        Vector256<byte> m12 = Vector256.Create((byte)12);
        Vector256<byte> m48 = Vector256.Create((byte)48);
        Vector256<byte> mC0 = Vector256.Create((byte)0xC0);
        Vector256<float> acc = Vector256<float>.Zero;

        for (int i = 0; i < nb; i++)
        {
            float d = y[i].D * HalfBits.ToSingle(x[i].D);
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            sbyte* q8 = y[i].Qs;

            Vector256<short> q8sums = Avx.LoadVector256((short*)y[i].Bsums);
            Vector128<sbyte> scales = Avx.LoadVector128(x[i].Scales);
            Vector256<short> scales16 = Avx2.ConvertToVector256Int16(scales);
            Vector256<int> q8sclsub = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(q8sums, scales16), 5);
            Vector256<int> sumi = Vector256<int>.Zero;
            int iscale = 0;

            for (int j = 0; j < Qk.SuperBlock / 128; j++)
            {
                Vector256<byte> q4bits1 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bits2 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bitsH = Avx.LoadVector256(qh);
                qh += 32;

                Vector256<byte> q4h0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsUInt16(), 4).AsByte();
                Vector256<byte> q4h1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsUInt16(), 2).AsByte();
                Vector256<byte> q4h2 = Avx2.And(q4bitsH, m48);
                Vector256<byte> q4h3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, mC0).AsUInt16(), 2).AsByte();

                Vector256<byte> q40 = Avx2.Or(Avx2.And(q4bits1, m15), q4h0);
                Vector256<byte> q41 = Avx2.Or(Avx2.And(q4bits2, m15), q4h1);
                Vector256<byte> q42 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsUInt16(), 4).AsByte(), m15), q4h2);
                Vector256<byte> q43 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsUInt16(), 4).AsByte(), m15), q4h3);

                Vector256<sbyte> q80 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q81 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q82 = Avx.LoadVector256(q8);
                q8 += 32;
                Vector256<sbyte> q83 = Avx.LoadVector256(q8);
                q8 += 32;

                Vector256<short> sc0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 0)));
                Vector256<short> sc1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 1)));
                Vector256<short> sc2 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 2)));
                Vector256<short> sc3 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 3)));
                iscale += 4;

                Vector256<int> i0 = ScaleQ6(q40, q80, sc0);
                Vector256<int> i1 = ScaleQ6(q41, q81, sc1);
                Vector256<int> i2 = ScaleQ6(q42, q82, sc2);
                Vector256<int> i3 = ScaleQ6(q43, q83, sc3);
                sumi = Avx2.Add(sumi, Avx2.Add(i0, i1));
                sumi = Avx2.Add(sumi, Avx2.Add(i2, i3));
            }

            sumi = Avx2.Subtract(sumi, q8sclsub);
            Vector256<float> dv = Vector256.Create(d);
            acc = Fma.IsSupported
                ? Fma.MultiplyAdd(dv, Avx.ConvertToVector256Single(sumi), acc)
                : Avx.Add(acc, Avx.Multiply(dv, Avx.ConvertToVector256Single(sumi)));
        }

        return VecDotQ4K.HorizontalSum(acc);
    }

    /// <summary>
    /// Decode each Q6_K superblock once and dot it against four Q8_K rows.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void DotAvx2x4(BlockQ6K* x, BlockQ8K* y0, BlockQ8K* y1, BlockQ8K* y2, BlockQ8K* y3, int n, float* dst)
    {
        int nb = n / Qk.SuperBlock;
        Vector256<byte> m3 = Vector256.Create((byte)3);
        Vector256<byte> m15 = Vector256.Create((byte)15);
        Vector256<byte> m12 = Vector256.Create((byte)12);
        Vector256<byte> m48 = Vector256.Create((byte)48);
        Vector256<byte> mC0 = Vector256.Create((byte)0xC0);
        Vector256<float> acc0 = Vector256<float>.Zero;
        Vector256<float> acc1 = Vector256<float>.Zero;
        Vector256<float> acc2 = Vector256<float>.Zero;
        Vector256<float> acc3 = Vector256<float>.Zero;

        for (int i = 0; i < nb; i++)
        {
            float xd = HalfBits.ToSingle(x[i].D);
            byte* q4 = x[i].Ql;
            byte* qh = x[i].Qh;
            Vector128<sbyte> scales = Avx.LoadVector128(x[i].Scales);
            Vector256<short> scales16 = Avx2.ConvertToVector256Int16(scales);
            Vector256<int> sum0 = Vector256<int>.Zero;
            Vector256<int> sum1 = Vector256<int>.Zero;
            Vector256<int> sum2 = Vector256<int>.Zero;
            Vector256<int> sum3 = Vector256<int>.Zero;
            int iscale = 0;

            for (int j = 0; j < Qk.SuperBlock / 128; j++)
            {
                Vector256<byte> q4bits1 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bits2 = Avx.LoadVector256(q4);
                q4 += 32;
                Vector256<byte> q4bitsH = Avx.LoadVector256(qh);
                qh += 32;

                Vector256<byte> q4h0 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m3).AsUInt16(), 4).AsByte();
                Vector256<byte> q4h1 = Avx2.ShiftLeftLogical(Avx2.And(q4bitsH, m12).AsUInt16(), 2).AsByte();
                Vector256<byte> q4h2 = Avx2.And(q4bitsH, m48);
                Vector256<byte> q4h3 = Avx2.ShiftRightLogical(Avx2.And(q4bitsH, mC0).AsUInt16(), 2).AsByte();

                Vector256<byte> q40 = Avx2.Or(Avx2.And(q4bits1, m15), q4h0);
                Vector256<byte> q41 = Avx2.Or(Avx2.And(q4bits2, m15), q4h1);
                Vector256<byte> q42 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits1.AsUInt16(), 4).AsByte(), m15), q4h2);
                Vector256<byte> q43 = Avx2.Or(Avx2.And(Avx2.ShiftRightLogical(q4bits2.AsUInt16(), 4).AsByte(), m15), q4h3);

                Vector256<short> sc0 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 0)));
                Vector256<short> sc1 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 1)));
                Vector256<short> sc2 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 2)));
                Vector256<short> sc3 = Avx2.ConvertToVector256Int16(Ssse3.Shuffle(scales, ScaleShuffle(iscale + 3)));
                iscale += 4;

                AccumulateQ6Lane(ref sum0, q40, q41, q42, q43, y0[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum1, q40, q41, q42, q43, y1[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum2, q40, q41, q42, q43, y2[i].Qs + j * 128, sc0, sc1, sc2, sc3);
                AccumulateQ6Lane(ref sum3, q40, q41, q42, q43, y3[i].Qs + j * 128, sc0, sc1, sc2, sc3);
            }

            Vector256<int> sub0 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y0[i].Bsums), scales16), 5);
            Vector256<int> sub1 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y1[i].Bsums), scales16), 5);
            Vector256<int> sub2 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y2[i].Bsums), scales16), 5);
            Vector256<int> sub3 = Avx2.ShiftLeftLogical(Avx2.MultiplyAddAdjacent(Avx.LoadVector256((short*)y3[i].Bsums), scales16), 5);
            sum0 = Avx2.Subtract(sum0, sub0);
            sum1 = Avx2.Subtract(sum1, sub1);
            sum2 = Avx2.Subtract(sum2, sub2);
            sum3 = Avx2.Subtract(sum3, sub3);
            acc0 = MaddScale(acc0, xd * y0[i].D, sum0);
            acc1 = MaddScale(acc1, xd * y1[i].D, sum1);
            acc2 = MaddScale(acc2, xd * y2[i].D, sum2);
            acc3 = MaddScale(acc3, xd * y3[i].D, sum3);
        }

        dst[0] = VecDotQ4K.HorizontalSum(acc0);
        dst[1] = VecDotQ4K.HorizontalSum(acc1);
        dst[2] = VecDotQ4K.HorizontalSum(acc2);
        dst[3] = VecDotQ4K.HorizontalSum(acc3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void AccumulateQ6Lane(
        ref Vector256<int> sum,
        Vector256<byte> q40,
        Vector256<byte> q41,
        Vector256<byte> q42,
        Vector256<byte> q43,
        sbyte* q8,
        Vector256<short> sc0,
        Vector256<short> sc1,
        Vector256<short> sc2,
        Vector256<short> sc3)
    {
        Vector256<int> i0 = ScaleQ6(q40, Avx.LoadVector256(q8), sc0);
        Vector256<int> i1 = ScaleQ6(q41, Avx.LoadVector256(q8 + 32), sc1);
        Vector256<int> i2 = ScaleQ6(q42, Avx.LoadVector256(q8 + 64), sc2);
        Vector256<int> i3 = ScaleQ6(q43, Avx.LoadVector256(q8 + 96), sc3);
        sum = Avx2.Add(sum, Avx2.Add(Avx2.Add(i0, i1), Avx2.Add(i2, i3)));
    }

    /// <summary>
    /// 32 unsigned 6-bit values × signed activations, then the shuffled Q6 scales.
    /// <paramref name="sc"/> is 8 copies of s0 followed by 8 copies of s1.
    /// vpmaddubsw pairs beat vpdpbusd+vpmulld here: the VNNI variant scales each
    /// dot lane separately, which is slower on Raptor Lake measured 2026-04.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> ScaleQ6(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc) =>
        ScaleQ6Avx2(q, a, sc);

    public static Vector256<int> ScaleQ6Avx2(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc) =>
        Avx2.MultiplyAddAdjacent(sc, Avx2.MultiplyAddAdjacent(q, a));

    public static Vector256<int> ScaleQ6Vnni(Vector256<byte> q, Vector256<sbyte> a, Vector256<short> sc)
    {
        Vector256<int> dots = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, q, a);
        int s0 = sc.GetElement(0);
        int s1 = sc.GetElement(8);
        return Avx2.MultiplyLow(dots, Vector256.Create(s0, s0, s0, s0, s1, s1, s1, s1));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> MaddScale(Vector256<float> acc, float d, Vector256<int> sumi)
    {
        Vector256<float> dv = Vector256.Create(d);
        Vector256<float> v = Avx.ConvertToVector256Single(sumi);
        return Fma.IsSupported ? Fma.MultiplyAdd(dv, v, acc) : Avx.Add(acc, Avx.Multiply(dv, v));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<sbyte> ScaleShuffle(int i)
    {
        ReadOnlySpan<byte> shuffle = ScaleShuffleBytes;
        return Vector128.LoadUnsafe(ref System.Runtime.InteropServices.MemoryMarshal.GetReference(shuffle), (nuint)(i * 16)).AsSByte();
    }

    private static ReadOnlySpan<byte> ScaleShuffleBytes =>
    [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7,
        8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9,
        10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11,
        12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13,
        14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];

    public static void Gemv(BlockQ6K* weights, float* input, float* output, int nIn, int nOut, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        nuint q8Bytes = (nuint)Q8K.RowBytes(nIn);
        NativeBuffer? owned = scratch == null ? new NativeBuffer(q8Bytes) : null;
        BlockQ8K* y = (BlockQ8K*)(scratch != null ? scratch.D(q8Bytes) : owned!.Pointer);
        try
        {
            Q8K.QuantizeRow(input, y, nIn);
            GemvPrequant(weights, y, output, nIn, nOut, pool);
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void Gemm(BlockQ6K* weights, float* input, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool = null, ScratchArena? scratch = null)
    {
        if (tokens == 1)
        {
            Gemv(weights, input, output, nIn, nOut, pool, scratch);
            return;
        }

        nuint q8Bytes = (nuint)((long)tokens * Q8K.RowBytes(nIn));
        NativeBuffer? owned = scratch == null ? new NativeBuffer(q8Bytes) : null;
        BlockQ8K* y = (BlockQ8K*)(scratch != null ? scratch.A(q8Bytes) : owned!.Pointer);
        try
        {
            if (pool == null)
            {
                Quantize(input, y, nIn, tokens, null);
                GemmFromQ8(weights, y, output, nIn, nOut, tokens, null);
            }
            else
            {
                int nb = nIn / Qk.SuperBlock;
                pool.For(Math.Max(tokens, nOut), (int worker, int workers) =>
                {
                    int t0 = tokens * worker / workers;
                    int t1 = tokens * (worker + 1) / workers;
                    for (int t = t0; t < t1; t++)
                        Q8K.QuantizeRow(input + t * nIn, y + t * nb, nIn);
                    pool.Barrier();
                    GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, worker, workers);
                });
            }
        }
        finally
        {
            owned?.Dispose();
        }
    }

    public static void Quantize(float* input, BlockQ8K* y, int nIn, int tokens, CpuThreadPool? pool)
    {
        int nb = nIn / Qk.SuperBlock;
        void QuantBody(int worker, int workers)
        {
            int begin = tokens * worker / workers;
            int end = tokens * (worker + 1) / workers;
            for (int t = begin; t < end; t++)
                Q8K.QuantizeRow(input + t * nIn, y + t * nb, nIn);
        }

        if (pool == null)
            QuantBody(0, 1);
        else
            pool.For(tokens, QuantBody);
    }

    public static void GemmFromQ8(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, int tokens, CpuThreadPool? pool)
    {
        if (pool == null)
            GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, 0, 1);
        else
            pool.For(nOut, (int worker, int workers) => GemmFromQ8Range(weights, y, output, nIn, nOut, tokens, worker, workers));
    }

    [SkipLocalsInit]
    private static void GemmFromQ8Range(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, int tokens, int worker, int workers)
    {
        int nb = nIn / Qk.SuperBlock;
        int begin = nOut * worker / workers;
        int end = nOut * (worker + 1) / workers;
        float* tmp = stackalloc float[4];
        int t = 0;
        if (Avx2.IsSupported)
        {
            for (; t + 3 < tokens; t += 4)
            {
                BlockQ8K* y0 = y + t * nb;
                BlockQ8K* y1 = y + (t + 1) * nb;
                BlockQ8K* y2 = y + (t + 2) * nb;
                BlockQ8K* y3 = y + (t + 3) * nb;
                for (int row = begin; row < end; row++)
                {
                    DotAvx2x4(weights + row * nb, y0, y1, y2, y3, nIn, tmp);
                    output[(t + 0) * nOut + row] = tmp[0];
                    output[(t + 1) * nOut + row] = tmp[1];
                    output[(t + 2) * nOut + row] = tmp[2];
                    output[(t + 3) * nOut + row] = tmp[3];
                }
            }
        }

        for (; t < tokens; t++)
        {
            BlockQ8K* yt = y + t * nb;
            for (int row = begin; row < end; row++)
                output[t * nOut + row] = Dot(weights + row * nb, yt, nIn);
        }
    }

    public static void GemvPrequant(BlockQ6K* weights, BlockQ8K* y, float* output, int nIn, int nOut, CpuThreadPool? pool)
    {
        int nb = nIn / Qk.SuperBlock;
        int next = 0;
        void Body(int worker, int workers)
        {
            // Dynamic chunks: a worker on a slow memory patch takes fewer rows
            // instead of gating the whole GEMV (same trick as ggml mul_mat).
            // ~16 claims per worker keeps each claim a long contiguous stream.
            int chunk = Math.Max(8, nOut / (workers * 16));
            int begin;
            while ((begin = Interlocked.Add(ref next, chunk) - chunk) < nOut)
            {
                int end = Math.Min(begin + chunk, nOut);
                for (int row = begin; row < end; row++)
                    output[row] = Dot(weights + row * nb, y, nIn);
            }
        }

        if (pool == null)
            Body(0, 1);
        else
            pool.For(nOut, Body);
    }

    /// <summary>
    /// One parallel region over several weights sharing the same Q8_K activation:
    /// decode pays one dispatch for Q,K,V or gate,up instead of one per weight.
    /// </summary>
    public static void GemvPrequantMulti(BlockQ8K* y, int nIn, CpuThreadPool? pool, BlockQ6K* w0, float* d0, int n0, BlockQ6K* w1, float* d1, int n1, BlockQ6K* w2 = null, float* d2 = null, int n2 = 0)
    {
        int nb = nIn / Qk.SuperBlock;
        int total = n0 + n1 + n2;
        int next = 0;
        void Body(int worker, int workers)
        {
            int chunk = Math.Max(8, total / (workers * 16));
            int begin;
            while ((begin = Interlocked.Add(ref next, chunk) - chunk) < total)
            {
                int end = Math.Min(begin + chunk, total);
                int r0 = Math.Clamp(begin, 0, n0);
                int r1 = Math.Clamp(end, 0, n0);
                for (int row = r0; row < r1; row++)
                    d0[row] = Dot(w0 + row * nb, y, nIn);
                r0 = Math.Clamp(begin - n0, 0, n1);
                r1 = Math.Clamp(end - n0, 0, n1);
                for (int row = r0; row < r1; row++)
                    d1[row] = Dot(w1 + row * nb, y, nIn);
                r0 = Math.Clamp(begin - n0 - n1, 0, n2);
                r1 = Math.Clamp(end - n0 - n1, 0, n2);
                for (int row = r0; row < r1; row++)
                    d2[row] = Dot(w2 + row * nb, y, nIn);
            }
        }

        if (pool == null || total == 0)
            Body(0, 1);
        else
            pool.For(total, Body);
    }
}
