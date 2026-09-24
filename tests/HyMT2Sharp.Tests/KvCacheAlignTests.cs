using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class KvCacheAlignTests
{
    [Fact]
    public void DivergingTail_KeepsSharedPrefix()
    {
        int[] cached = [1, 2, 3, 4, 5];
        int[] prompt = [1, 2, 3, 6, 7];
        PromptReuse plan = KvCacheAlign.Plan(cached, prompt);
        Assert.Equal(3, plan.TruncateTo);
        Assert.Equal(3, plan.SuffixStart);
        Assert.Equal([6, 7], prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void LongerCacheThanPrompt_DoesNotDropPrefix()
    {
        int[] cached = [1, 2, 3, 10, 11];
        int[] prompt = [1, 2, 3, 6];
        PromptReuse plan = KvCacheAlign.Plan(cached, prompt);
        Assert.Equal(3, plan.TruncateTo);
        Assert.Equal([6], prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void FullHit_ReplaysLastToken()
    {
        int[] ids = [1, 2, 3];
        PromptReuse plan = KvCacheAlign.Plan(ids, ids);
        Assert.Equal(2, plan.TruncateTo);
        Assert.Equal(2, plan.SuffixStart);
        Assert.Equal([3], ids[plan.SuffixStart..]);
    }

    [Fact]
    public void EmptyCache_ForwardsWholePrompt()
    {
        int[] prompt = [4, 5, 6];
        PromptReuse plan = KvCacheAlign.Plan([], prompt);
        Assert.Equal(0, plan.TruncateTo);
        Assert.Equal(prompt, prompt[plan.SuffixStart..]);
    }

    [Fact]
    public void NoOverlap_Resets()
    {
        PromptReuse plan = KvCacheAlign.Plan([1, 2, 3], [9, 8]);
        Assert.Equal(0, plan.TruncateTo);
        Assert.Equal(0, plan.SuffixStart);
    }

    [Fact]
    public void PrefixMatches_IdenticalRegion()
    {
        int[] cached = [10, 20, 30, 40];
        Assert.True(KvCacheAlign.PrefixMatches(cached, [10, 20, 30], 3));
        Assert.True(KvCacheAlign.PrefixMatches(cached, cached, 4));
    }

    [Fact]
    public void PrefixMatches_StaleTail_DetectsPrefixDivergence()
    {
        // Warm cache = D-prefix + A-tail; prompt C shares the A-tail at
        // [64..100) but its block-prefix differs — the tail KV was computed
        // under D's prefix, so positions beyond n must not be reused.
        int[] cached = Enumerable.Repeat(1, 64).Concat(Enumerable.Repeat(2, 36)).ToArray();  // D[..64) + A[64..100)
        int[] prompt = Enumerable.Repeat(9, 64).Concat(Enumerable.Repeat(2, 36)).ToArray();  // C[..64) + A[64..100)
        Assert.False(KvCacheAlign.PrefixMatches(cached, prompt, 64));
    }

    [Fact]
    public void PrefixMatches_ShortCache_IsFalse()
    {
        int[] cached = [1, 2];
        Assert.False(KvCacheAlign.PrefixMatches(cached, [1, 2, 3, 4], 4));
    }
}
