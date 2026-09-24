using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class SamplerTests
{
    private static float[] Logits(int n, Func<int, float> gen)
    {
        float[] x = new float[n];
        for (int i = 0; i < n; i++)
            x[i] = gen(i);
        return x;
    }

    [Fact]
    public void Greedy_PicksArgMax()
    {
        Sampler s = new(new SamplingParams { Temperature = 0f });
        float[] logits = Logits(1024, i => i == 777 ? 42f : i * 0.01f);
        Assert.Equal(777, s.Sample(logits));
    }

    [Fact]
    public void TopK1_EqualsArgMax()
    {
        Sampler s = new(new SamplingParams { Temperature = 0.7f, TopK = 1, Seed = 1 });
        float[] logits = Logits(4096, i => i == 2000 ? 10f : (i % 37) * 0.1f);
        Assert.Equal(2000, s.Sample(logits));
    }

    [Fact]
    public void SameSeed_IsReproducible()
    {
        int n = 512;
        int[] a = new int[32], b = new int[32];
        Sampler sa = new(new SamplingParams { Temperature = 1f, TopP = 0.9f, Seed = 1234 });
        Sampler sb = new(new SamplingParams { Temperature = 1f, TopP = 0.9f, Seed = 1234 });
        for (int i = 0; i < 32; i++)
        {
            float[] la = Logits(n, j => MathF.Sin(j * 12.9898f + i) * 10f);
            float[] lb = (float[])la.Clone();
            a[i] = sa.Sample(la);
            b[i] = sb.Sample(lb);
        }

        Assert.Equal(a, b);
    }

    [Fact]
    public void TopP_RespectsCumulativeMass()
    {
        // logits: one dominant (prob ~0.73), rest tiny. top_p=0.5 keeps only the top token.
        Sampler s = new(new SamplingParams { Temperature = 1f, TopP = 0.5f, Seed = 7 });
        for (int i = 0; i < 32; i++)
        {
            float[] logits = Logits(256, j => j == 5 ? 10f : 0f);
            Assert.Equal(5, s.Sample(logits));
        }
    }

    [Fact]
    public void MinP_DropsWeakCandidates()
    {
        // max prob token + others below min_p*max → always argmax
        Sampler s = new(new SamplingParams { Temperature = 1f, MinP = 0.5f, Seed = 9 });
        for (int i = 0; i < 32; i++)
        {
            float[] logits = Logits(256, j => j == 9 ? 5f : 0f);
            Assert.Equal(9, s.Sample(logits));
        }
    }

    [Fact]
    public void RepeatPenalty_SuppressesRepeats()
    {
        // Token 3 starts as argmax; once sampled it should get penalized and
        // a different token must eventually win.
        Sampler s = new(new SamplingParams
        {
            Temperature = 0f,
            RepeatPenalty = 2f,
            PenaltyWindow = 8,
        });
        int first = s.Sample(Logits(64, j => j == 3 ? 10f : 5f));
        Assert.Equal(3, first);
        int second = s.Sample(Logits(64, j => j == 3 ? 10f : 5f));
        Assert.Equal(0, second); // token 3 penalized to 5; others tie at 5 → argmax takes 0
    }

    [Fact]
    public void Sample_ZeroAllocations_SteadyState()
    {
        const int vocab = 32000;
        Sampler s = new(new SamplingParams
        {
            Temperature = 0.9f,
            TopK = 40,
            TopP = 0.95f,
            RepeatPenalty = 1.1f,
            PenaltyWindow = 64,
            Seed = 42,
        });
        float[] logits = Logits(vocab, i => MathF.Cos(i * 0.001f) * 8f);

        for (int i = 0; i < 8; i++)
            s.Sample(logits); // warm up lazy scratch

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 64; i++)
            s.Sample(logits);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(0, allocated);
    }

    [Fact]
    public void FullVocab_TopP_NoTopK_Works()
    {
        // top_p over the whole vocab without top_k exercises the vocab-sized sort path.
        Sampler s = new(new SamplingParams { Temperature = 1f, TopP = 0.95f, Seed = 3 });
        float[] logits = Logits(512, i => MathF.Sin(i) * 3f);
        int t = s.Sample(logits);
        Assert.InRange(t, 0, 511);
    }
}
