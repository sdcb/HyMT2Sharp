using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Cross-request prefix cache at <see cref="BlockTokens"/>-token granularity.
/// Keys are chained FNV-1a hashes — block i's key covers all tokens before it,
/// so a hit at depth i certifies an identical prefix of (i+1)*BlockTokens
/// tokens, which is the only region whose KV may be reused (KV is causally
/// dependent). Values are opaque serialized blobs: the CPU backend memcpy's
/// them in and out of the per-layer arenas, but the table itself never sees a
/// raw pointer, so a GPU/MLX backend can reuse it unchanged (its blobs are
/// D2H snapshots). Capacity-bounded, LRU-evicted. Not thread-safe: callers
/// serialize requests already (the server's single-slot gate).
/// </summary>
public sealed unsafe class KvBlockStore
{
    public sealed class Entry
    {
        public required ulong Hash;
        public required int[] Tokens;
        public required byte[] Blob;
        public LinkedListNode<Entry>? Node;
    }

    private readonly Dictionary<ulong, LinkedListNode<Entry>> _map = [];
    private readonly LinkedList<Entry> _lru = new(); // front = MRU
    private readonly long _capBytes;
    private long _bytes;

    public int BlockTokens { get; }
    public int Layers { get; }
    public int KvStride { get; } // elements (ushort) per token per layer per K/V

    public long BlockBytes => (long)BlockTokens * KvStride * sizeof(ushort) * 2 * Layers;
    public int Count => _map.Count;
    public long Bytes => _bytes;
    public long Hits { get; private set; }
    public long Misses { get; private set; }

    public KvBlockStore(int layers, int kvStride, int blockTokens, long capBytes)
    {
        Layers = layers;
        KvStride = kvStride;
        BlockTokens = blockTokens;
        _capBytes = capBytes;
    }

    public const ulong Seed = 14695981039346656037UL;

    public static ulong ChainHash(ulong prev, ReadOnlySpan<int> tokens)
    {
        ulong h = prev;
        foreach (int t in tokens)
        {
            h ^= (uint)t;
            h *= 1099511628211UL;
        }
        return h;
    }

    /// <summary>Look up and mark MRU. Token ids are verified, so a hash
    /// collision degrades to a miss instead of restoring wrong KV.</summary>
    public Entry? Find(ulong hash, ReadOnlySpan<int> tokens)
    {
        if (_map.TryGetValue(hash, out LinkedListNode<Entry>? node)
            && node.Value.Tokens.AsSpan().SequenceEqual(tokens))
        {
            _lru.Remove(node);
            _lru.AddFirst(node);
            Hits++;
            return node.Value;
        }
        Misses++;
        return null;
    }

    /// <summary>Copy the KV of the block at <paramref name="blockIndex"/>
    /// out of the per-layer arenas into the store. No-op if already present.</summary>
    public void Put(ulong hash, int[] tokens, ushort*[] cacheK, ushort*[] cacheV, int blockIndex)
    {
        if (_map.ContainsKey(hash))
            return;
        byte[] blob = new byte[BlockBytes];
        CopyToBlob(blob, cacheK, cacheV, blockIndex);
        Entry e = new() { Hash = hash, Tokens = tokens, Blob = blob };
        e.Node = _lru.AddFirst(e);
        _map[hash] = e.Node;
        _bytes += blob.Length;
        Evict();
    }

    /// <summary>Copy a stored block back into the arenas at
    /// <paramref name="blockIndex"/> (position must be inside capacity).</summary>
    public void Restore(Entry e, ushort*[] cacheK, ushort*[] cacheV, int blockIndex)
    {
        CopyFromBlob(e.Blob, cacheK, cacheV, blockIndex);
    }

    // Blob layout: [layer][K,V][BlockTokens * KvStride ushorts].
    private int BlobOffset(int layer, bool v) =>
        (layer * 2 + (v ? 1 : 0)) * BlockTokens * KvStride * sizeof(ushort);

    private void CopyToBlob(byte[] blob, ushort*[] cacheK, ushort*[] cacheV, int blockIndex)
    {
        long srcOff = (long)blockIndex * BlockTokens * KvStride;
        int bytes = BlockTokens * KvStride * sizeof(ushort);
        fixed (byte* bp = blob)
        {
            for (int l = 0; l < Layers; l++)
            {
                Buffer.MemoryCopy(cacheK[l] + srcOff, bp + BlobOffset(l, false), bytes, bytes);
                Buffer.MemoryCopy(cacheV[l] + srcOff, bp + BlobOffset(l, true), bytes, bytes);
            }
        }
    }

    private void CopyFromBlob(byte[] blob, ushort*[] cacheK, ushort*[] cacheV, int blockIndex)
    {
        long dstOff = (long)blockIndex * BlockTokens * KvStride;
        int bytes = BlockTokens * KvStride * sizeof(ushort);
        fixed (byte* bp = blob)
        {
            for (int l = 0; l < Layers; l++)
            {
                Buffer.MemoryCopy(bp + BlobOffset(l, false), cacheK[l] + dstOff, bytes, bytes);
                Buffer.MemoryCopy(bp + BlobOffset(l, true), cacheV[l] + dstOff, bytes, bytes);
            }
        }
    }

    private void Evict()
    {
        while (_bytes > _capBytes && _lru.Last is LinkedListNode<Entry> tail)
        {
            _bytes -= tail.Value.Blob.Length;
            _map.Remove(tail.Value.Hash);
            _lru.RemoveLast();
        }
    }
}
