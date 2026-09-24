using System.Numerics;
using System.Runtime.CompilerServices;

namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Sampling configuration. Defaults are greedy-equivalent-neutral; set
/// <see cref="Temperature"/> to 0 for deterministic argmax decoding.
/// </summary>
public readonly record struct SamplingParams
{
    /// <summary>0 = greedy argmax, bypasses every other field.</summary>
    public float Temperature { get; init; } = 1f;

    /// <summary>Keep only the k highest-logit candidates. 0 = whole vocab.</summary>
    public int TopK { get; init; } = 0;

    /// <summary>Nucleus cutoff on sorted cumulative probability. 1 = disabled.</summary>
    public float TopP { get; init; } = 1f;

    /// <summary>Drop candidates below MinP * (max prob). 0 = disabled.</summary>
    public float MinP { get; init; } = 0f;

    /// <summary>llama.cpp-style repetition penalty over <see cref="PenaltyWindow"/>. 1 = disabled.</summary>
    public float RepeatPenalty { get; init; } = 1f;

    /// <summary>Number of recent generated tokens the repetition penalty covers.</summary>
    public int PenaltyWindow { get; init; } = 64;

    /// <summary>Fixed RNG seed for reproducible sampling; null = entropy.</summary>
    public int? Seed { get; init; } = null;

    public SamplingParams() { }
}

/// <summary>
/// Low-allocation token sampler. Per-token work is a handful of O(vocab) passes
/// over the logits buffer plus sorting of at most TopK candidates; steady-state
/// allocation is zero after the first call per vocab size. Mutates the logits
/// span in place (callers discard the buffer anyway).
/// </summary>
public sealed class Sampler
{
    private readonly SamplingParams _p;
    private Xoshiro256StarStar _rng;

    // penalty history ring (empty when penalty disabled)
    private readonly int[] _window;
    private int _winPos, _winLen;

    // scratch, lazily sized on first Sample call to the vocab length
    private float[]? _topBuf;   // top-k running buffer, size TopK
    private int[]? _sortIds;    // top-p sort, size min(topK > 0 ? topK : vocab, vocab)
    private float[]? _sortVals; // negated probs so ascending sort = descending prob

    public Sampler() : this(new SamplingParams { Temperature = 0f }) { }

    public Sampler(SamplingParams p)
    {
        _p = p;
        _rng = new Xoshiro256StarStar(p.Seed is int s ? (ulong)s : (ulong)Environment.TickCount64 * 0x9E3779B97F4A7C15UL + 0x165667B19E3779F9UL);
        _window = p.RepeatPenalty != 1f && p.PenaltyWindow > 0 ? new int[p.PenaltyWindow] : [];
    }

    /// <summary>Seeds the penalty window with tokens that were not sampled by this
    /// instance (e.g. the prompt), so repeated prompt text gets penalized too.</summary>
    public void FeedHistory(ReadOnlySpan<int> tokens)
    {
        if (_window.Length == 0)
            return;
        int from = Math.Max(0, tokens.Length - _window.Length);
        for (int i = from; i < tokens.Length; i++)
            Record(tokens[i]);
    }

    public int Sample(Span<float> logits)
    {
        int token;

        ApplyPenalty(logits);
        if (_p.Temperature <= 0f)
        {
            token = ArgMax(logits);
        }
        else
        {
            token = Stochastic(logits);
        }

        Record(token);
        return token;
    }

    private int Stochastic(Span<float> logits)
    {
        int n = logits.Length;
        float invT = 1f / _p.Temperature;
        if (invT != 1f)
            ScaleInPlace(logits, invT);

        if (_p.TopK > 0 && _p.TopK < n)
            MaskBelowKthLargest(logits, _p.TopK);

        // in-place exp(logit - max); masked entries (-inf) become 0 prob
        float max = VectorMax(logits);
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            float l = logits[i];
            float p = l == float.NegativeInfinity ? 0f : MathF.Exp(l - max);
            logits[i] = p;
            sum += p;
        }

        ApplyTopPMinP(logits, ref sum);

        // multinomial: cumulative-prob walk, no sort needed. Zero-prob
        // entries must be skipped so an r of exactly 0 can't select them.
        float r = _rng.NextSingle() * sum;
        float acc = 0;
        for (int i = 0; i < n; i++)
        {
            float p = logits[i];
            if (p <= 0f)
                continue;
            acc += p;
            if (acc >= r)
                return i;
        }

        // r landed past the last survivor due to fp rounding: take last nonzero
        for (int i = n - 1; i >= 0; i--)
        {
            if (logits[i] > 0)
                return i;
        }

        return ArgMax(logits); // every prob zero (degenerate) — pick anything
    }

    private void ApplyPenalty(Span<float> logits)
    {
        float pen = _p.RepeatPenalty;
        int len = _winLen;
        if (len == 0 || pen == 1f)
            return;
        int cap = _window.Length;
        for (int i = 0; i < len; i++)
        {
            int t = _window[(int)((uint)(_winPos - 1 - i) % (uint)cap)];
            if ((uint)t >= (uint)logits.Length)
                continue;
            ref float l = ref logits[t];
            l = l <= 0f ? l * pen : l / pen;
        }
    }

    private void Record(int token)
    {
        int cap = _window.Length;
        if (cap == 0)
            return;
        _window[_winPos] = token;
        _winPos = (_winPos + 1) % cap;
        if (_winLen < cap)
            _winLen++;
    }

    /// <summary>Keeps a running buffer of the k largest logits; everything below
    /// the k-th largest is masked to -inf. Single pass, no sort.</summary>
    private void MaskBelowKthLargest(Span<float> logits, int k)
    {
        int n = logits.Length;
        if (k >= n)
            return;
        _topBuf ??= new float[k];
        float[] buf = _topBuf;

        logits.Slice(0, k).CopyTo(buf);
        float min = float.MaxValue;
        int minIdx = 0;
        for (int j = 0; j < k; j++)
        {
            if (buf[j] < min)
            {
                min = buf[j];
                minIdx = j;
            }
        }

        for (int i = k; i < n; i++)
        {
            float l = logits[i];
            if (l <= min)
                continue;
            buf[minIdx] = l;
            // rescan for the new weakest candidate
            min = float.MaxValue;
            minIdx = 0;
            for (int j = 0; j < k; j++)
            {
                if (buf[j] < min)
                {
                    min = buf[j];
                    minIdx = j;
                }
            }
        }

        // Entries equal to the cutoff can push survivors past k; keep only as
        // many ties as there are free slots so the candidate count is exact.
        int free = k;
        for (int i = 0; i < n; i++)
        {
            if (logits[i] > min)
                free--;
        }

        for (int i = 0; i < n; i++)
        {
            float l = logits[i];
            if (l < min || (l == min && free-- <= 0))
                logits[i] = float.NegativeInfinity;
        }
    }

    /// <summary>Sorts only the surviving (nonzero-prob) candidates, then drops the
    /// tail beyond TopP cumulative mass and below MinP * maxProb.</summary>
    private void ApplyTopPMinP(Span<float> probs, ref float sum)
    {
        if (_p.TopP >= 1f && _p.MinP <= 0f)
            return;

        int n = probs.Length;
        int needed = _p.TopK > 0 && _p.TopK < n ? _p.TopK : n;
        if (_sortIds == null || _sortIds.Length < needed)
        {
            _sortIds = new int[needed];
            _sortVals = new float[needed];
        }

        int[] ids = _sortIds;
        float[] vals = _sortVals!;
        int c = 0;
        for (int i = 0; i < n; i++)
        {
            float p = probs[i];
            if (p > 0f)
            {
                ids[c] = i;
                vals[c] = -p; // ascending sort of negated = descending
                c++;
            }
        }

        if (c == 0)
            return;

        Array.Sort(vals, ids, 0, c);

        // min_p: keep p >= MinP * maxProb
        int keep = c;
        if (_p.MinP > 0f)
        {
            float thr = -vals[0] * _p.MinP;
            int j = 1;
            while (j < keep && -vals[j] >= thr)
                j++;
            keep = j;
        }

        // top_p: keep the smallest prefix whose cumulative mass >= TopP * sum
        if (_p.TopP < 1f)
        {
            float limit = _p.TopP * sum;
            float cum = 0;
            int j = 0;
            while (j < keep && cum < limit)
                cum += -vals[j++];
            keep = j;
        }

        for (int j = keep; j < c; j++)
            probs[ids[j]] = 0f;

        if (keep < c)
        {
            float dropped = 0;
            for (int j = keep; j < c; j++)
                dropped -= vals[j];
            sum -= dropped;
        }
    }

    /// <summary>SIMD argmax over the whole buffer. Two passes worst case:
    /// vector max reduction, then a scan for the first index hitting the max.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int ArgMax(ReadOnlySpan<float> logits)
    {
        int n = logits.Length;
        int i = 0;
        float max;
        if (n >= Vector<float>.Count)
        {
            Vector<float> acc = new Vector<float>(logits.Slice(0, Vector<float>.Count));
            for (i = Vector<float>.Count; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                acc = Vector.Max(acc, new Vector<float>(logits.Slice(i, Vector<float>.Count)));
            max = acc[0];
            for (int j = 1; j < Vector<float>.Count; j++)
                if (acc[j] > max)
                    max = acc[j];
            for (; i < n; i++)
                if (logits[i] > max)
                    max = logits[i];
            // find first index holding max
            for (i = 0; i < n; i++)
                if (logits[i] == max)
                    return i;
            return 0;
        }

        int best = 0;
        max = logits[0];
        for (i = 1; i < n; i++)
        {
            if (logits[i] > max)
            {
                max = logits[i];
                best = i;
            }
        }

        return best;
    }

    private static float VectorMax(ReadOnlySpan<float> x)
    {
        int n = x.Length, i = 0;
        if (n >= Vector<float>.Count)
        {
            Vector<float> acc = new Vector<float>(x.Slice(0, Vector<float>.Count));
            for (i = Vector<float>.Count; i + Vector<float>.Count <= n; i += Vector<float>.Count)
                acc = Vector.Max(acc, new Vector<float>(x.Slice(i, Vector<float>.Count)));
            float m = acc[0];
            for (int j = 1; j < Vector<float>.Count; j++)
                if (acc[j] > m)
                    m = acc[j];
            for (; i < n; i++)
                if (x[i] > m)
                    m = x[i];
            return m;
        }

        float max = x[0];
        for (i = 1; i < n; i++)
            if (x[i] > max)
                max = x[i];
        return max;
    }

    private static void ScaleInPlace(Span<float> x, float s)
    {
        int n = x.Length, i = 0;
        Vector<float> vs = new(s);
        for (; i + Vector<float>.Count <= n; i += Vector<float>.Count)
        {
            Vector<float> v = new Vector<float>(x.Slice(i, Vector<float>.Count));
            (v * vs).CopyTo(x.Slice(i));
        }

        for (; i < n; i++)
            x[i] *= s;
    }

    private struct Xoshiro256StarStar(ulong seed)
    {
        private ulong _s0 = SplitMix(ref seed);
        private ulong _s1 = SplitMix(ref seed);
        private ulong _s2 = SplitMix(ref seed);
        private ulong _s3 = SplitMix(ref seed);

        private static ulong SplitMix(ref ulong z)
        {
            z += 0x9E3779B97F4A7C15UL;
            ulong x = z;
            x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9UL;
            x = (x ^ (x >> 27)) * 0x94D049BB133111EBUL;
            return x ^ (x >> 31);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static ulong Rotl(ulong x, int k) => (x << k) | (x >> (64 - k));

        public ulong Next()
        {
            ulong r = Rotl(_s1 * 5, 7) * 9;
            ulong t = _s1 << 17;
            _s2 ^= _s0;
            _s3 ^= _s1;
            _s1 ^= _s2;
            _s0 ^= _s3;
            _s2 ^= t;
            _s3 = Rotl(_s3, 45);
            return r;
        }

        /// <summary>Uniform float in [0, 1).</summary>
        public float NextSingle() => (Next() >> 40) * (1f / 16777216f);
    }
}
