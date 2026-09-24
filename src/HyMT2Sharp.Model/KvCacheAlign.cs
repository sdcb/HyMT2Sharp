namespace Sdcb.HyMT2Sharp.Model;

public readonly record struct PromptAlignment(int Cached, int[] Suffix);

public static class KvCacheAlign
{
    public static int CommonPrefixLength(ReadOnlySpan<int> cached, ReadOnlySpan<int> prompt)
    {
        int n = Math.Min(cached.Length, prompt.Length);
        int i = 0;
        while (i < n && cached[i] == prompt[i])
            i++;
        return i;
    }

    /// <summary>
    /// Whether the first <paramref name="length"/> cached tokens are identical
    /// to the prompt's — the validity condition for KV beyond a restored
    /// block region (KV at position j is only valid under token history
    /// [0..j]).
    /// </summary>
    public static bool PrefixMatches(ReadOnlySpan<int> cached, ReadOnlySpan<int> prompt, int length)
        => length <= cached.Length && cached[..length].SequenceEqual(prompt[..length]);

    /// <summary>
    /// Reuse the longest matching prefix. If the prompt is already fully cached,
    /// keep all but the last token and replay it so logits can be refreshed.
    /// </summary>
    public static PromptReuse Plan(ReadOnlySpan<int> cached, ReadOnlySpan<int> prompt)
    {
        if (prompt.Length == 0)
            return new PromptReuse(0, 0);
        int common = CommonPrefixLength(cached, prompt);
        if (common == prompt.Length)
            return new PromptReuse(prompt.Length - 1, prompt.Length - 1);
        return new PromptReuse(common, common);
    }
}

public readonly record struct PromptReuse(int TruncateTo, int SuffixStart);
