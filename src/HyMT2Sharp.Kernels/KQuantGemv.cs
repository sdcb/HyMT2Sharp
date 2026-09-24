namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>One Q4_K or Q6_K GEMV output sharing a Q8_K activation with others.</summary>
public readonly unsafe struct KGemvTarget(BlockQ4K* q4, BlockQ6K* q6, float* dst, int rows)
{
    public readonly BlockQ4K* Q4 = q4;
    public readonly BlockQ6K* Q6 = q6;
    public readonly float* Dst = dst;
    public readonly int Rows = rows;
}

/// <summary>
/// Decode GEMV over up to three Q4_K/Q6_K weights (mixed types allowed, as in Q4_K_M
/// where attn_v may be Q6_K) in one parallel region with dynamic row chunks. The
/// portable tier builds the <see cref="BlockQ8KAct"/> view once for all of them.
/// </summary>
public static unsafe class KQuantGemv
{
    public static void Multi(BlockQ8K* y, int nIn, CpuThreadPool? pool, KGemvTarget t0, KGemvTarget t1, KGemvTarget t2 = default)
    {
        int nb = nIn / Qk.SuperBlock;
        int n0 = t0.Rows, n1 = t1.Rows, n2 = t2.Rows;
        int total = n0 + n1 + n2;
        bool vec = !Simd.UseAvx2 && VecI8.PairLayoutSupported;
        NativeBuffer? actOwned = vec && nb > 64 ? new NativeBuffer((nuint)(nb * sizeof(BlockQ8KAct))) : null;
        BlockQ8KAct* actStack = stackalloc BlockQ8KAct[vec && actOwned == null ? nb : 0];
        BlockQ8KAct* act = actOwned != null ? (BlockQ8KAct*)actOwned.Pointer : vec ? actStack : null;
        if (act != null)
            Q8K.ToVecAct(y, act, nb);

        int next = 0;
        void Body(int worker, int workers)
        {
            int chunk = Math.Max(8, total / (workers * 16));
            int begin;
            while ((begin = Interlocked.Add(ref next, chunk) - chunk) < total)
            {
                int end = Math.Min(begin + chunk, total);
                Range(t0, begin, end);
                Range(t1, begin - n0, end - n0);
                Range(t2, begin - n0 - n1, end - n0 - n1);
            }
        }

        void Range(KGemvTarget t, int r0, int r1)
        {
            r0 = Math.Clamp(r0, 0, t.Rows);
            r1 = Math.Clamp(r1, 0, t.Rows);
            if (t.Q6 != null)
            {
                for (int row = r0; row < r1; row++)
                    t.Dst[row] = act != null ? Q6K.DotAct(t.Q6 + row * nb, act, nb) : Q6K.Dot(t.Q6 + row * nb, y, nIn);
            }
            else
            {
                for (int row = r0; row < r1; row++)
                    t.Dst[row] = act != null ? VecDotQ4K.DotAct(t.Q4 + row * nb, act, nb) : VecDotQ4K.Dot(t.Q4 + row * nb, y, nIn);
            }
        }

        try
        {
            if (pool == null || total == 0)
                Body(0, 1);
            else
                pool.For(total, Body);
        }
        finally
        {
            actOwned?.Dispose();
        }
    }
}
