namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Cross-request KV cache retention configuration — one mechanism, presets
/// for policy. Two independent knobs:
/// <see cref="KeepFlatCache"/> keeps the live arena warm across requests so
/// the next prompt reuses its longest matching prefix at token granularity
/// (zero copies; on GPU this is the VRAM-resident session KV);
/// <see cref="Blocks"/> additionally snapshots completed prefix blocks into a
/// bounded store so prefixes survive across unrelated requests (one memcpy
/// per hit; a GPU backend snapshots D2H, restores H2D).
/// Presets: <see cref="None"/> drops everything at end of request;
/// <see cref="Memory"/> keeps the flat cache only; <see cref="BlockPolicy"/>
/// adds a block pool.
/// </summary>
public sealed record KvCacheConfig
{
    public bool KeepFlatCache { get; init; }
    public KvBlockStoreConfig? Blocks { get; init; }

    public static KvCacheConfig None { get; } = new();
    public static KvCacheConfig Memory { get; } = new() { KeepFlatCache = true };

    public static KvCacheConfig BlockPolicy(long capBytes = KvBlockStoreConfig.DefaultCapBytes, int blockTokens = KvBlockStoreConfig.DefaultBlockTokens)
        => new() { KeepFlatCache = true, Blocks = new(capBytes, blockTokens) };

    /// <summary>Display name used in logs, derived from the knobs.</summary>
    public string PolicyName => this switch
    {
        { KeepFlatCache: false } => "none",
        { Blocks: not null } => "blocks",
        _ => "memory",
    };
}

/// <summary>Backing tier for the block pool. Disk is reserved — the table's
/// values are opaque blobs, so a file backend needs no model changes.</summary>
public enum KvBlockBackend
{
    Memory,
    // Disk,
}

/// <param name="CapBytes">Pool capacity. 4 MB per 64-token block on this
/// model (bf16), so the default 1 GB holds ~256 blocks ≈ 16K tokens.</param>
/// <param name="BlockTokens">Tokens per block; a hit loses at most
/// BlockTokens-1 tail tokens to recompute.</param>
public sealed record KvBlockStoreConfig(
    long CapBytes = KvBlockStoreConfig.DefaultCapBytes,
    int BlockTokens = KvBlockStoreConfig.DefaultBlockTokens,
    KvBlockBackend Backend = KvBlockBackend.Memory)
{
    public const long DefaultCapBytes = 1L << 30;
    public const int DefaultBlockTokens = 64;
}
