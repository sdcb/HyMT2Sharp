using Sdcb.HyMT2Sharp.Model;

namespace HyMT2Sharp.Tests;

public unsafe class KvBlockStoreTests
{
    private const int Layers = 2;
    private const int KvStride = 4;
    private const int BlockTokens = 2;
    private const int Cap = 8; // enough capacity for several blocks

    private static (ushort*[] k, ushort*[] v, ushort[] kMem, ushort[] vMem) MakeCache()
    {
        ushort[] kMem = new ushort[Cap * KvStride * Layers];
        ushort[] vMem = new ushort[Cap * KvStride * Layers];
        ushort*[] k = new ushort*[Layers];
        ushort*[] v = new ushort*[Layers];
        fixed (ushort* kp = kMem, vp = vMem)
        {
            for (int l = 0; l < Layers; l++)
            {
                k[l] = kp + l * Cap * KvStride;
                v[l] = vp + l * Cap * KvStride;
            }
        }
        return (k, v, kMem, vMem);
    }

    [Fact]
    public void PutFind_RestoresExactBlock()
    {
        KvBlockStore store = new(Layers, KvStride, BlockTokens, 1 << 20);
        (ushort*[] k, ushort*[] v, ushort[] kMem, ushort[] vMem) = MakeCache();
        for (int i = 0; i < kMem.Length; i++) { kMem[i] = (ushort)(i + 1); vMem[i] = (ushort)(i + 1000); }

        int[] tokens = [7, 8];
        ulong h = KvBlockStore.ChainHash(KvBlockStore.Seed, tokens);
        store.Put(h, tokens, k, v, blockIndex: 0);

        Array.Clear(kMem); Array.Clear(vMem);
        KvBlockStore.Entry? e = store.Find(h, tokens);
        Assert.NotNull(e);
        store.Restore(e, k, v, blockIndex: 0);
        for (int i = 0; i < BlockTokens * KvStride; i++)
        {
            Assert.Equal((ushort)(i + 1), kMem[i]);
            Assert.Equal((ushort)(i + 1000), vMem[i]);
            Assert.Equal(0, kMem[i + BlockTokens * KvStride]); // block 1 untouched
        }
    }

    [Fact]
    public void Find_TokenMismatch_IsMiss()
    {
        KvBlockStore store = new(Layers, KvStride, BlockTokens, 1 << 20);
        (ushort*[] k, ushort*[] v, _, _) = MakeCache();
        int[] tokens = [7, 8];
        ulong h = KvBlockStore.ChainHash(KvBlockStore.Seed, tokens);
        store.Put(h, tokens, k, v, 0);
        Assert.Null(store.Find(h, [7, 9]));
    }

    [Fact]
    public void ChainHash_IsOrderDependent()
    {
        ulong a = KvBlockStore.ChainHash(KvBlockStore.ChainHash(KvBlockStore.Seed, [1, 2]), [3, 4]);
        ulong b = KvBlockStore.ChainHash(KvBlockStore.ChainHash(KvBlockStore.Seed, [9, 9]), [3, 4]);
        Assert.NotEqual(a, b); // same second block, different prefix => different key
    }

    [Fact]
    public void EvictsLeastRecentlyUsed_WhenOverCap()
    {
        KvBlockStore store = new(Layers, KvStride, BlockTokens, 2L * BlockTokens * KvStride * sizeof(ushort) * 2 * Layers);
        (ushort*[] k, ushort*[] v, _, _) = MakeCache();
        int[] t0 = [1, 1], t1 = [2, 2], t2 = [3, 3];
        ulong h0 = KvBlockStore.ChainHash(KvBlockStore.Seed, t0);
        ulong h1 = KvBlockStore.ChainHash(h0, t1);
        ulong h2 = KvBlockStore.ChainHash(h1, t2);
        store.Put(h0, t0, k, v, 0);
        store.Put(h1, t1, k, v, 1);
        store.Put(h2, t2, k, v, 2); // evicts t0 (LRU)

        Assert.Null(store.Find(h0, t0));
        Assert.NotNull(store.Find(h1, t1));
        Assert.NotNull(store.Find(h2, t2));
    }
}
