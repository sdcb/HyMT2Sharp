using System.Diagnostics;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Backends.Metal;

/// <summary>
/// Metal decode backend (M2): one full decode step encoded as a flat sequence
/// of compute dispatches into a single command buffer per token. Weights live
/// in shared-storage MTLBuffers uploaded once at load; activation/KV buffers
/// persist on device. Only the final logits row comes back to the host.
/// Supported weight types: Q4_K + F32 (Q4_K_M model); flat bf16 KV capped at
/// ContextLength (paged KV is M4).
/// </summary>
public sealed unsafe class MetalBackend : IComputeBackend
{
    private readonly MtlDevice _dev;
    private ModelConfig _cfg = null!;
    private readonly Dictionary<string, IntPtr> _w = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgmlTensorType> _wtype = new(StringComparer.Ordinal);
    private IntPtr _psoGemv, _psoGemvQ6, _psoGemvQ8, _psoGemvQ2, _psoGemvStq;
    private IntPtr _psoSgemm, _psoMma, _psoXt, _psoDeqQ4, _psoDeqQ6, _psoDeqQ8, _psoDeqQ2, _psoDeqStq;
    // Persistently dequantized fp32 weight copies for prefill GEMM (M6):
    // allocated on first use so decode-only runs pay nothing.
    private readonly Dictionary<string, IntPtr> _wfp32 = new(StringComparer.Ordinal);
    private IntPtr _psoEmbedQ4, _psoEmbedQ6, _psoEmbedQ8, _psoRms, _psoRope, _psoKvAppend, _psoAttn, _psoSilu, _psoAdd;
    private IntPtr _psoAttnSplit, _psoAttnMerge;
    // Max split-K chunks per head for decode attention (chunk >= kvLen/16).
    private const int MaxSplit = 16;
    private IntPtr _attnPart;
    private IntPtr _psoGemvMulti, _psoRopeRms;
    private IntPtr _psoEmbedRowsQ4, _psoEmbedRowsQ6, _psoEmbedRowsQ8, _psoRopeMulti, _psoKvAppendMulti;
    private IntPtr _psoAttnScores, _psoAttnCombine, _psoKvCopy;
    private IntPtr[] _kvK = null!, _kvV = null!;
    private int _kvCap;
    private int _kvStride;

    // ---- paged KV + device block store (M4) ----
    // Device KV pools stay contiguous per layer, but logical positions reach
    // them through _kvTab: tab[logicalBlk] -> physical block (64 positions).
    // _store hashes token blocks to phys blocks; restoring a shared prefix is
    // a table rewrite — zero KV copies, zero uploads.
    private const int BlkShift = 6;                  // 64 positions per physical block
    private const int BlkTok = 1 << BlkShift;
    private IntPtr _kvTab;
    private int[] _tab = null!;                      // host shadow, -1 = unmapped
    private bool _tabDirty;
    private int _maxBlk;
    private int _physNext;
    private readonly Stack<int> _freePhys = new();
    private int _liveLen;                            // positions the model may still read
    private sealed class DevBlk { public int Phys; public int[] Toks = null!; public LinkedListNode<ulong>? Node; }
    private readonly Dictionary<ulong, DevBlk> _store = new();
    private readonly LinkedList<ulong> _storeLru = new();   // front = oldest
    private readonly Dictionary<int, ulong> _physToHash = new();
    private IntPtr _embd, _outNorm;
    private IntPtr _h, _n1, _n2, _q, _k, _v, _ao, _attnOut, _gate, _up, _down, _normed, _logitsBuf, _tok;
    // ForwardStep returns this shared buffer — callers must consume it before the next call.
    private float[] _logitsHost = null!;
    private int _vocab;
    private readonly bool _timing = Environment.GetEnvironmentVariable("HYMT_METAL_TIMING") == "1";
    private readonly Stopwatch _sw = new();
    private double _encMs, _gpuMs;
    private int _n;

    public string Name => "metal";

    public bool SupportsPrefill => true;

    public MetalBackend()
    {
        _dev = MtlDevice.Create();
    }

    private IntPtr W(string name) => _w.TryGetValue(name, out IntPtr b) ? b : 0;

    private static void* Contents(IntPtr buf) => (void*)ObjC.Send0(buf, ObjC.Sel("contents"));

    private IntPtr NewPso(IntPtr lib, string name) =>
        _dev.NewPso(_dev.NewFunction(lib, name));

    public void LoadModel(GgufFile gguf, ModelConfig cfg)
    {
        _cfg = cfg;
        IntPtr lib1 = _dev.NewLibraryFromSource(MslKernels.Source);
        IntPtr lib2 = _dev.NewLibraryFromSource(MslDecodeKernels.Source);
        _psoGemv = NewPso(lib1, "q4k_gemv_fast4");
        _psoGemvQ6 = NewPso(lib2, "q6k_gemv_fast4");
        _psoGemvQ8 = NewPso(lib2, "q8_0_gemv_fast4");
        _psoGemvQ2 = NewPso(lib2, "q2c_gemv_fast4");
        _psoGemvStq = NewPso(lib2, "stq_gemv_fast4");
        _psoEmbedQ4 = NewPso(lib2, "q4k_embed_row");
        _psoEmbedQ6 = NewPso(lib2, "q6k_embed_row");
        _psoEmbedQ8 = NewPso(lib2, "q8_0_embed_row");
        _psoRms = NewPso(lib2, "rmsnorm_rows");
        _psoRope = NewPso(lib2, "rope_neox");
        _psoKvAppend = NewPso(lib2, "kv_append_bf16");
        _psoAttn = NewPso(lib2, "attn_decode");
        _psoAttnSplit = NewPso(lib2, "attn_split");
        _psoAttnMerge = NewPso(lib2, "attn_merge");
        _psoSilu = NewPso(lib2, "silu_mul");
        _psoAdd = NewPso(lib2, "add_inplace");
        _psoSgemm = NewPso(lib2, "sgemm4x4");
        try { _psoMma = NewPso(lib2, "mma_gemm"); }
        catch { _psoMma = IntPtr.Zero; }   // simdgroup MMA unavailable -> sgemm4x4 covers everything
        _psoXt = NewPso(lib2, "xtranspose");
        _psoDeqQ4 = NewPso(lib2, "q4k_deq");
        _psoDeqQ6 = NewPso(lib2, "q6k_deq");
        _psoDeqQ8 = NewPso(lib2, "q8_0_deq");
        _psoDeqQ2 = NewPso(lib2, "q2c_deq");
        _psoDeqStq = NewPso(lib2, "stq_deq");
        _psoEmbedRowsQ4 = NewPso(lib2, "q4k_embed_rows");
        _psoEmbedRowsQ6 = NewPso(lib2, "q6k_embed_rows");
        _psoEmbedRowsQ8 = NewPso(lib2, "q8_0_embed_rows");
        _psoRopeMulti = NewPso(lib2, "rope_neox_multi");
        _psoKvAppendMulti = NewPso(lib2, "kv_append_multi");
        if (cfg.HeadDim > 128)
            throw new NotSupportedException(
                $"attn_scores/attn_combine stage q in threadgroup float[128] — headDim {cfg.HeadDim} unsupported");
        _psoAttnScores = NewPso(lib2, "attn_scores");
        _psoAttnCombine = NewPso(lib2, "attn_combine");
        _psoKvCopy = NewPso(lib2, "kv_copy_block");
        _psoGemvMulti = NewPso(lib2, "gemv_multi");
        _psoRopeRms = NewPso(lib2, "rope_rms");

        foreach ((string name, GgufTensorInfo info) in gguf.Tensors)
        {
            ulong bytes = info.Type switch
            {
                GgmlTensorType.F32 => info.NumElements * sizeof(float),
                GgmlTensorType.Q4_K => info.NumElements / 256 * (ulong)144,
                GgmlTensorType.Q6_K => info.NumElements / 256 * (ulong)210,
                GgmlTensorType.Q8_0 => info.NumElements / 32 * (ulong)34,
                GgmlTensorType.Q2_0C => info.NumElements / 512 * (ulong)130,
                GgmlTensorType.STQ1_0 => info.NumElements / 256 * (ulong)42,
                _ => throw new NotSupportedException(
                    $"Metal backend supports F32/Q4_K/Q6_K/Q8_0/Q2_0C/STQ1_0 only; {name} is {info.Type} — use --backend cpu"),
            };
            byte* data = gguf.DataBase + (long)info.Offset;
            _w[name] = _dev.NewBufferBytes(data, (nuint)bytes);
            _wtype[name] = info.Type;
        }

        _embd = W("token_embd.weight");
        if (_embd == 0) throw new KeyNotFoundException("token_embd.weight");
        _outNorm = W("output_norm.weight");
        if (W("output.weight") == 0)
        {
            _w["output.weight"] = _embd;  // tied lm_head
            _wtype["output.weight"] = _wtype["token_embd.weight"];
        }
        if (_outNorm == 0) throw new KeyNotFoundException("output_norm.weight");

        // Buffers: bf16 KV pools indexed through the block table
        // (cap = context length, clamped to threadgroup score capacity of
        // attn_decode/attn_combine: kvLen <= 4096)
        _kvStride = cfg.NumKvHeads * cfg.HeadDim;
        _kvCap = Math.Min(cfg.ContextLength, 4096);
        _kvK = new IntPtr[cfg.NumLayers];
        _kvV = new IntPtr[cfg.NumLayers];
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            _kvK[l] = _dev.NewBuffer((nuint)(_kvCap * _kvStride * sizeof(ushort)));
            _kvV[l] = _dev.NewBuffer((nuint)(_kvCap * _kvStride * sizeof(ushort)));
        }
        _maxBlk = _kvCap >> BlkShift;
        _tab = new int[_maxBlk];
        Array.Fill(_tab, -1);
        fixed (int* tp = _tab)
            _kvTab = _dev.NewBufferBytes(tp, (nuint)(_tab.Length * sizeof(int)));

        int hidden = cfg.HiddenSize, ffn = cfg.FfnSize;
        int qDim = cfg.NumHeads * cfg.HeadDim, kDim = _kvStride;
        _vocab = cfg.VocabSize;
        _h = _dev.NewBuffer((nuint)(hidden * 4));
        _n1 = _dev.NewBuffer((nuint)(hidden * 4));
        _n2 = _dev.NewBuffer((nuint)(hidden * 4));
        _q = _dev.NewBuffer((nuint)(qDim * 4));
        _k = _dev.NewBuffer((nuint)(kDim * 4));
        _v = _dev.NewBuffer((nuint)(kDim * 4));
        _ao = _dev.NewBuffer((nuint)(qDim * 4));
        // split-K decode attention partials: heads * MaxSplit * (2 + headDim)
        _attnPart = _dev.NewBuffer((nuint)(cfg.NumHeads * MaxSplit * (2 + cfg.HeadDim) * 4));
        _attnOut = _dev.NewBuffer((nuint)(hidden * 4));
        _gate = _dev.NewBuffer((nuint)(ffn * 4));
        _up = _dev.NewBuffer((nuint)(ffn * 4));
        _down = _dev.NewBuffer((nuint)(hidden * 4));
        _normed = _dev.NewBuffer((nuint)(hidden * 4));
        _logitsBuf = _dev.NewBuffer((nuint)(_vocab * 4));
        _tok = _dev.NewBuffer(4);
        _logitsHost = new float[_vocab];
    }

    public void UploadKv(int layer, int pos, int len, ushort* k, ushort* v)
    {
        if (pos + len > _kvCap)
            throw new NotSupportedException($"Metal KV capacity {_kvCap} exceeded at {pos + len}");
        EnsureWrittenRange(pos, len);
        int rowBytes = _kvStride * sizeof(ushort);
        long loRow = long.MaxValue, hiRow = 0;
        for (int j = pos; j < pos + len; j++)
        {
            long prow = (long)_tab[j >> BlkShift] * BlkTok + (j & (BlkTok - 1));
            loRow = Math.Min(loRow, prow); hiRow = Math.Max(hiRow, prow);
            nuint off = (nuint)prow * (nuint)rowBytes;
            Buffer.MemoryCopy(k + j * _kvStride, (byte*)Contents(_kvK[layer]) + off, rowBytes, rowBytes);
            Buffer.MemoryCopy(v + j * _kvStride, (byte*)Contents(_kvV[layer]) + off, rowBytes, rowBytes);
        }
        nuint dirtyOff = (nuint)loRow * (nuint)rowBytes;
        nuint dirtyLen = (nuint)(hiRow - loRow + 1) * (nuint)rowBytes;
        MtlDevice.DidModifyRange(_kvK[layer], dirtyOff, dirtyLen);
        MtlDevice.DidModifyRange(_kvV[layer], dirtyOff, dirtyLen);
    }

    public bool SupportsBlockStore => true;
    public int DeviceBlockTokens => BlkTok;

    public void SetCacheLen(int len) => _liveLen = len;

    // Physical block allocator + hash store. A physical block is owned by the
    // current table mapping and/or one store entry; freeing a phys clears every
    // table slot still pointing at it so no two logical blocks ever alias.
    private int AllocPhys()
    {
        if (_freePhys.Count > 0) return _freePhys.Pop();
        if (_physNext < _maxBlk) return _physNext++;
        // Evict LRU entries until a physical block outside the live range frees.
        int liveBlk = (_liveLen + BlkTok - 1) >> BlkShift;
        var live = new HashSet<int>();
        for (int i = 0; i < liveBlk; i++) if (_tab[i] >= 0) live.Add(_tab[i]);
        var node = _storeLru.First;
        while (node != null)
        {
            var next = node.Next;
            if (!live.Contains(_store[node.Value].Phys))
            {
                RemoveEntry(node.Value);
                return _freePhys.Pop();
            }
            node = next;
        }
        throw new NotSupportedException($"Metal KV block pool exhausted ({_maxBlk} blocks x {BlkTok} tok)");
    }

    private void Unpin(ulong hash)
    {
        DevBlk e = _store[hash];
        _store.Remove(hash);
        _physToHash.Remove(e.Phys);
        if (e.Node is not null) _storeLru.Remove(e.Node);
    }

    // Eviction path: the phys is guaranteed outside the live range — clear any
    // stale (dead) table slots pointing at it, then recycle.
    private void RemoveEntry(ulong hash)
    {
        int phys = _store[hash].Phys;
        Unpin(hash);
        for (int i = 0; i < _maxBlk; i++) if (_tab[i] == phys) _tab[i] = -1;
        _freePhys.Push(phys);
    }

    // Free phys only if no table slot references it anymore and no store
    // entry pins it.
    private void ReleaseOrFree(int phys)
    {
        if (phys < 0 || _physToHash.ContainsKey(phys)) return;
        for (int i = 0; i < _maxBlk; i++) if (_tab[i] == phys) return;
        _freePhys.Push(phys);
    }

    // Map every logical block covering [pos, pos+len) to a physical block.
    // A phys pinned by a store entry is never written — the logical block gets
    // a fresh phys instead (copy-on-write). For a partial first block the rows
    // before pos must be copied over; they queue into _copies for the next
    // command buffer.
    private readonly List<(int srcPhys, int dstPhys, int rows)> _copies = new();
    private void EnsureWrittenRange(int pos, int len)
    {
        int first = pos >> BlkShift, last = (pos + len - 1) >> BlkShift;
        for (int i = first; i <= last && i < _maxBlk; i++)
        {
            if (_tab[i] < 0)
            {
                _tab[i] = AllocPhys();
                _tabDirty = true;
            }
            else if (_physToHash.ContainsKey(_tab[i]))
            {
                // Phys is pinned by a store entry describing its current
                // content — keep entry and block intact, write elsewhere.
                int fresh = AllocPhys();
                if (i == first && (pos & (BlkTok - 1)) != 0)
                    _copies.Add((_tab[i], fresh, pos & (BlkTok - 1)));
                _tab[i] = fresh;
                _tabDirty = true;
            }
        }
    }

    private void FlushTab()
    {
        if (!_tabDirty) return;
        _tabDirty = false;
        fixed (int* tp = _tab)
            Buffer.MemoryCopy(tp, Contents(_kvTab), (nuint)(_tab.Length * sizeof(int)), (nuint)(_tab.Length * sizeof(int)));
        MtlDevice.DidModifyRange(_kvTab, 0, (nuint)(_tab.Length * sizeof(int)));
    }

    private void ExecCopies(CmdCtx cc)
    {
        if (_copies.Count == 0) return;
        foreach ((int srcPhys, int dstPhys, int rows) in _copies)
        {
            for (int l = 0; l < _cfg.NumLayers; l++)
            {
                cc.SetPso(_psoKvCopy);
                cc.SetBuffer(_kvK[l], 0, 0); cc.SetBuffer(_kvV[l], 0, 1);
                cc.SetBuffer(_kvK[l], 0, 2); cc.SetBuffer(_kvV[l], 0, 3);
                cc.SetInt(4, _kvStride); cc.SetInt(5, srcPhys); cc.SetInt(6, dstPhys); cc.SetInt(7, rows);
                cc.Dispatch((nuint)((rows * _kvStride + 255) / 256), 1, 1, 256, 1, 1);
            }
        }
        _copies.Clear();
    }

    public void HarvestPrefix(ReadOnlySpan<int> tokens, int blockTokens)
    {
        if (Environment.GetEnvironmentVariable("HYMT_METAL_DEBUG") == "1")
            Console.Error.WriteLine($"[blk] harvest tokens={tokens.Length} store={_store.Count} tab0={_tab[0]}");
        if (blockTokens != BlkTok)
            throw new NotSupportedException($"Metal block store requires --kv-block-tokens {BlkTok}, got {blockTokens}");
        ulong h = KvBlockStore.Seed;
        for (int b = 0; b * BlkTok + BlkTok <= tokens.Length && b < _maxBlk; b++)
        {
            ReadOnlySpan<int> toks = tokens.Slice(b * BlkTok, BlkTok);
            h = KvBlockStore.ChainHash(h, toks);
            int phys = _tab[b];
            if (phys < 0) continue;  // block never touched device KV
            if (_store.TryGetValue(h, out DevBlk? cur) && cur.Toks.AsSpan().SequenceEqual(toks))
            {
                if (cur.Node is not null) { _storeLru.Remove(cur.Node); _storeLru.AddLast(cur.Node); }
                continue;
            }
            if (cur is not null) { int op = cur.Phys; Unpin(h); ReleaseOrFree(op); }
            if (_physToHash.TryGetValue(phys, out ulong prev)) Unpin(prev);
            var e2 = new DevBlk { Phys = phys, Toks = toks.ToArray() };
            e2.Node = _storeLru.AddLast(h);
            _store[h] = e2;
            _physToHash[phys] = h;
        }
    }

    public int RestorePrefix(ReadOnlySpan<int> tokens, int blockTokens)
    {
        if (blockTokens != BlkTok)
            throw new NotSupportedException($"Metal block store requires --kv-block-tokens {BlkTok}, got {blockTokens}");
        ulong h = KvBlockStore.Seed;
        int hit = 0;
        for (int i = 0; i * BlkTok + BlkTok <= tokens.Length && i < _maxBlk; i++)
        {
            ReadOnlySpan<int> toks = tokens.Slice(i * BlkTok, BlkTok);
            h = KvBlockStore.ChainHash(h, toks);
            if (!_store.TryGetValue(h, out DevBlk? e) || !e.Toks.AsSpan().SequenceEqual(toks))
                break;
            int old = _tab[i];
            _tab[i] = e.Phys;
            if (old != e.Phys)
                ReleaseOrFree(old);
            hit++;
        }
        if (Environment.GetEnvironmentVariable("HYMT_METAL_DEBUG") == "1")
            Console.Error.WriteLine($"[blk] restore tokens={tokens.Length} hit={hit} store={_store.Count}");
        if (hit == 0) return 0;
        _tabDirty = true;
        FlushTab();
        return hit * BlkTok;
    }

    public float[] ForwardStep(ReadOnlySpan<int> tokens, int pos)
    {
        if (tokens.Length != 1)
            return PrefillStep(tokens, pos);
        int tokenId = tokens[0];
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, heads = c.NumHeads, kvHeads = c.NumKvHeads;
        int dim = c.HeadDim, qDim = heads * dim, kDim = kvHeads * dim, kvLen = pos + 1;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"Metal KV capacity {_kvCap} exceeded at {kvLen}");
        float scale = 1f / MathF.Sqrt(dim);
        EnsureWrittenRange(pos, 1);
        FlushTab();
        _liveLen = Math.Max(_liveLen, kvLen);

        _sw.Restart();
        *(int*)Contents(_tok) = tokenId;
        MtlDevice.DidModifyRange(_tok, 0, 4);

        var cc = CmdCtx.Begin(_dev.Queue);  // owns the autorelease pool for this step
        ExecCopies(cc);
        // embed: h = dequant(embd[token])
        cc.SetPso(_wtype["token_embd.weight"] switch
        {
            GgmlTensorType.Q4_K => _psoEmbedQ4,
            GgmlTensorType.Q6_K => _psoEmbedQ6,
            GgmlTensorType.Q8_0 => _psoEmbedQ8,
            GgmlTensorType t => throw new NotSupportedException(
                $"embed token_embd.weight: {t} not supported on Metal backend — use --backend cpu"),
        });
        cc.SetBuffer(_embd, 0, 0); cc.SetBuffer(_tok, 0, 1); cc.SetBuffer(_h, 0, 2);
        cc.SetInt(3, hidden);
        cc.Dispatch((nuint)((hidden + 255) / 256), 1, 1, 256, 1, 1);

        // Fused decode (M7): 8 dispatches per layer instead of 17 — q/k/v
        // ride one sectioned gemv (v lands directly in the KV row as bf16),
        // rope+per-head-rmsnorm is one kernel writing k straight into KV,
        // gate+up+silu is one pair-dispatch, and residuals accumulate inside
        // the gemv stores instead of separate add_inplace kernels.
        long prow = (long)_tab[pos >> BlkShift] * BlkTok + (pos & (BlkTok - 1));
        nuint kvOff = (nuint)prow * (nuint)_kvStride * sizeof(ushort);
        int maxLayers = c.NumLayers;
        if (int.TryParse(Environment.GetEnvironmentVariable("HYMT_METAL_LAYERS"), out int ml))
            maxLayers = Math.Min(ml, c.NumLayers);
        for (int l = 0; l < maxLayers; l++)
        {
            Rms(cc, _h, W($"blk.{l}.attn_norm.weight"), _n1, 1, hidden);
            GemvF(cc, _n1, hidden, IntPtr.Zero,
                new GSec($"blk.{l}.attn_q.weight", _q, 0, qDim),
                new GSec($"blk.{l}.attn_k.weight", _k, 0, kDim),
                new GSec($"blk.{l}.attn_v.weight", _kvV[l], kvOff, kDim, Bf16: true));
            RopeRms(cc, l, pos);
            // attention -> ao: split-K flash-decoding past 256 positions
            // (attn_decode's serial position scan underuses the GPU).
            int split = Math.Min((kvLen + 255) / 256, MaxSplit);
            if (split <= 1)
            {
                cc.SetPso(_psoAttn);
                cc.SetBuffer(_q, 0, 0); cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(_kvV[l], 0, 2);
                cc.SetBuffer(_ao, 0, 3);
                cc.SetInt(4, heads); cc.SetInt(5, kvHeads); cc.SetInt(6, dim);
                cc.SetInt(7, _kvStride); cc.SetInt(8, kvLen); cc.SetFloat(9, scale);
                cc.SetBuffer(_kvTab, 0, 10); cc.SetInt(11, BlkShift);
                cc.Dispatch((nuint)heads, 1, 1, 128, 1, 1);
            }
            else
            {
                cc.SetPso(_psoAttnSplit);
                cc.SetBuffer(_q, 0, 0); cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(_kvV[l], 0, 2);
                cc.SetBuffer(_attnPart, 0, 3);
                cc.SetInt(4, heads); cc.SetInt(5, kvHeads); cc.SetInt(6, dim);
                cc.SetInt(7, _kvStride); cc.SetInt(8, kvLen); cc.SetFloat(9, scale);
                cc.SetBuffer(_kvTab, 0, 10); cc.SetInt(11, BlkShift);
                cc.SetInt(12, split);
                cc.Dispatch((nuint)(heads * split), 1, 1, 128, 1, 1);
                cc.SetPso(_psoAttnMerge);
                cc.SetBuffer(_attnPart, 0, 0); cc.SetBuffer(_ao, 0, 1);
                cc.SetInt(2, heads); cc.SetInt(3, dim); cc.SetInt(4, split);
                cc.Dispatch((nuint)heads, 1, 1, 128, 1, 1);
            }
            GemvF(cc, _ao, qDim, _h,
                new GSec($"blk.{l}.attn_output.weight", _h, 0, hidden));

            // FFN: gate+up+silu fused, then down accumulates into _h
            Rms(cc, _h, W($"blk.{l}.ffn_norm.weight"), _n2, 1, hidden);
            GemvPair(cc, _n2, hidden,
                $"blk.{l}.ffn_gate.weight", $"blk.{l}.ffn_up.weight", _gate, c.FfnSize);
            GemvF(cc, _gate, c.FfnSize, _h,
                new GSec($"blk.{l}.ffn_down.weight", _h, 0, hidden));
        }

        Rms(cc, _h, _outNorm, _normed, 1, hidden);
        GemvF(cc, _normed, hidden, IntPtr.Zero,
            new GSec("output.weight", _logitsBuf, 0, _vocab));

        cc.EndEnc();
        cc.Commit();
        double encMs = _timing ? _sw.Elapsed.TotalMilliseconds : 0;
        cc.Wait();
        if (_timing)
        {
            _encMs += encMs; _gpuMs += _sw.Elapsed.TotalMilliseconds - encMs; _n++;
            if (_n % 64 == 0)
            {
                Console.WriteLine($"[metal] enc={_encMs / _n:F2}ms gpu={_gpuMs / _n:F2}ms tok");
                _encMs = _gpuMs = 0; _n = 0;
            }
        }
        cc.Drain();

        Marshal.Copy((IntPtr)Contents(_logitsBuf), _logitsHost, 0, _vocab);
        return _logitsHost;
    }

    // seq > 1 forward: same layer sequence as decode but GEMM instead of GEMV,
    // multi-token rope/KV-append variants, and attention dispatched per position
    // (kvLen = start + t + 1). Per-call scratch buffers are +1 MTLBuffers —
    // they must be released by hand after waitUntilCompleted.
    private float[] PrefillStep(ReadOnlySpan<int> tokens, int start)
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, heads = c.NumHeads, kvHeads = c.NumKvHeads;
        int dim = c.HeadDim, qDim = heads * dim, kDim = kvHeads * dim, ffn = c.FfnSize;
        int T = tokens.Length, kvLen = start + T;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"Metal KV capacity {_kvCap} exceeded at {kvLen}");
        float scale = 1f / MathF.Sqrt(dim);
        EnsureWrittenRange(start, T);
        FlushTab();
        _liveLen = Math.Max(_liveLen, kvLen);

        IntPtr toks;
        fixed (int* tp = tokens) toks = _dev.NewBufferBytes(tp, (nuint)(T * 4));
        IntPtr h = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr n1 = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr n2 = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr q = _dev.NewBuffer((nuint)(T * qDim * 4));
        IntPtr kb = _dev.NewBuffer((nuint)(T * kDim * 4));
        IntPtr vb = _dev.NewBuffer((nuint)(T * kDim * 4));
        IntPtr ao = _dev.NewBuffer((nuint)(T * qDim * 4));
        IntPtr attnOut = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr gate = _dev.NewBuffer((nuint)(T * ffn * 4));
        IntPtr up = _dev.NewBuffer((nuint)(T * ffn * 4));
        IntPtr down = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr normed = _dev.NewBuffer((nuint)(T * hidden * 4));
        IntPtr scores = _dev.NewBuffer((nuint)heads * (nuint)T * (nuint)kvLen * 4);
        // Transposed copies of the four per-layer GEMM inputs (xT[k][t]).
        int Tpad = (T + 63) & ~63;
        IntPtr xT1 = _dev.NewBuffer((nuint)(hidden * Tpad * 4));   // n1 (q/k/v)
        IntPtr xT2 = _dev.NewBuffer((nuint)(hidden * Tpad * 4));   // n2 (gate/up)
        IntPtr xT3 = _dev.NewBuffer((nuint)(qDim * Tpad * 4));     // ao (attn_output)
        IntPtr xT4 = _dev.NewBuffer((nuint)(ffn * Tpad * 4));      // silu(gate) (down)

        var cc = CmdCtx.Begin(_dev.Queue);
        ExecCopies(cc);
        cc.SetPso(PsoFor("token_embd.weight", hidden,
            (_psoEmbedRowsQ4, _psoEmbedRowsQ6, _psoEmbedRowsQ8, default, default)));
        cc.SetBuffer(_embd, 0, 0); cc.SetBuffer(toks, 0, 1); cc.SetBuffer(h, 0, 2);
        cc.SetInt(3, hidden);
        cc.Dispatch((nuint)T, 1, 1, 256, 1, 1);
        if (Environment.GetEnvironmentVariable("HYMT_METAL_EARLYD") == "1")
        {
            cc.EndEnc(); cc.Commit(); cc.Wait();
            DumpF(h, T * hidden, "h_early");
            cc.Drain();  // release this ctx's autorelease pool before abandoning it
            cc = CmdCtx.Begin(_dev.Queue);
        }

        int maxLayers = c.NumLayers;
        if (int.TryParse(Environment.GetEnvironmentVariable("HYMT_METAL_LAYERS"), out int ml))
            maxLayers = Math.Min(ml, c.NumLayers);
        for (int l = 0; l < maxLayers; l++)
        {
            Rms(cc, h, W($"blk.{l}.attn_norm.weight"), n1, T, hidden);
            Xt(cc, n1, xT1, hidden, T, Tpad);
            Gemm(cc, $"blk.{l}.attn_q.weight", xT1, q, hidden, qDim, T, Tpad);
            Gemm(cc, $"blk.{l}.attn_k.weight", xT1, kb, hidden, kDim, T, Tpad);
            Gemm(cc, $"blk.{l}.attn_v.weight", xT1, vb, hidden, kDim, T, Tpad);
            RopeMulti(cc, q, heads, dim, start, T);
            RopeMulti(cc, kb, kvHeads, dim, start, T);
            Rms(cc, q, W($"blk.{l}.attn_q_norm.weight"), q, T * heads, dim);
            Rms(cc, kb, W($"blk.{l}.attn_k_norm.weight"), kb, T * kvHeads, dim);
            cc.SetPso(_psoKvAppendMulti);
            cc.SetBuffer(kb, 0, 0); cc.SetBuffer(vb, 0, 1);
            cc.SetBuffer(_kvK[l], 0, 2); cc.SetBuffer(_kvV[l], 0, 3);
            cc.SetInt(4, kDim); cc.SetInt(5, _kvStride); cc.SetInt(6, start);
            cc.SetBuffer(_kvTab, 0, 7); cc.SetInt(8, BlkShift); cc.SetInt(9, T);
            cc.Dispatch((nuint)((T * kDim + 255) / 256), 1, 1, 256, 1, 1);
            if (Environment.GetEnvironmentVariable("HYMT_METAL_SLOWATTN") == "1")
            {
                for (int t = 0; t < T; t++)
                {
                    cc.SetPso(_psoAttn);
                    cc.SetBuffer(q, (nuint)(t * qDim * 4), 0);
                    cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(_kvV[l], 0, 2);
                    cc.SetBuffer(ao, (nuint)(t * qDim * 4), 3);
                    cc.SetInt(4, heads); cc.SetInt(5, kvHeads); cc.SetInt(6, dim);
                    cc.SetInt(7, _kvStride); cc.SetInt(8, start + t + 1); cc.SetFloat(9, scale);
                    cc.SetBuffer(_kvTab, 0, 10); cc.SetInt(11, BlkShift);
                    cc.Dispatch((nuint)heads, 1, 1, 128, 1, 1);
                }
            }
            else
            {
                cc.SetPso(_psoAttnScores);
                cc.SetBuffer(q, 0, 0); cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(scores, 0, 2);
                cc.SetInt(3, heads); cc.SetInt(4, kvHeads); cc.SetInt(5, dim);
                cc.SetInt(6, _kvStride); cc.SetInt(7, kvLen); cc.SetInt(8, start); cc.SetInt(9, T);
                cc.SetFloat(10, scale);
                cc.SetBuffer(_kvTab, 0, 11); cc.SetInt(12, BlkShift);
                cc.Dispatch((nuint)T, (nuint)heads, 1, 128, 1, 1);
                cc.SetPso(_psoAttnCombine);
                cc.SetBuffer(scores, 0, 0); cc.SetBuffer(_kvV[l], 0, 1); cc.SetBuffer(ao, 0, 2);
                cc.SetInt(3, heads); cc.SetInt(4, kvHeads); cc.SetInt(5, dim);
                cc.SetInt(6, _kvStride); cc.SetInt(7, kvLen); cc.SetInt(8, start); cc.SetInt(9, T);
                cc.SetBuffer(_kvTab, 0, 10); cc.SetInt(11, BlkShift);
                cc.Dispatch((nuint)T, (nuint)heads, 1, 128, 1, 1);
            }
            Xt(cc, ao, xT3, qDim, T, Tpad);
            Gemm(cc, $"blk.{l}.attn_output.weight", xT3, attnOut, qDim, hidden, T, Tpad);
            Add(cc, h, attnOut, T * hidden);

            Rms(cc, h, W($"blk.{l}.ffn_norm.weight"), n2, T, hidden);
            Xt(cc, n2, xT2, hidden, T, Tpad);
            Gemm(cc, $"blk.{l}.ffn_gate.weight", xT2, gate, hidden, ffn, T, Tpad);
            Gemm(cc, $"blk.{l}.ffn_up.weight", xT2, up, hidden, ffn, T, Tpad);
            cc.SetPso(_psoSilu);
            cc.SetBuffer(gate, 0, 0); cc.SetBuffer(up, 0, 1);
            cc.SetInt(2, T * ffn);
            cc.Dispatch((nuint)((T * ffn + 255) / 256), 1, 1, 256, 1, 1);
            Xt(cc, gate, xT4, ffn, T, Tpad);
            Gemm(cc, $"blk.{l}.ffn_down.weight", xT4, down, ffn, hidden, T, Tpad);
            Add(cc, h, down, T * hidden);
        }

        Rms(cc, h, _outNorm, normed, T, hidden);
        Gemv(cc, "output.weight", normed, (nuint)((T - 1) * hidden * 4), _logitsBuf, 0, hidden, _vocab);

        cc.EndEnc();
        cc.Commit();
        cc.Wait();
        if (Environment.GetEnvironmentVariable("HYMT_METAL_DUMP") == "1")
        {
            DumpF(n1, T * hidden, "n1"); DumpF(q, T * qDim, "q");
            DumpF(kb, T * kDim, "kb"); DumpF(vb, T * kDim, "vb"); DumpF(ao, T * qDim, "ao");
            DumpF(gate, T * ffn, "gate"); DumpF(down, T * hidden, "down");
            DumpF(normed, T * hidden, "normed"); DumpF(xT1, hidden * Tpad, "xT1");
            DumpF(h, T * hidden, "h");
        }
        // GPU is done with the scratch; newBuffer* returned +1 objects.
        foreach (IntPtr b in new[]
            { toks, h, n1, n2, q, kb, vb, ao, attnOut, gate, up, down, normed, scores,
              xT1, xT2, xT3, xT4 })
            ObjC.Release(b);
        cc.Drain();

        Marshal.Copy((IntPtr)Contents(_logitsBuf), _logitsHost, 0, _vocab);
        return _logitsHost;
    }

    private void RopeMulti(CmdCtx c, IntPtr x, int heads, int headDim, int posBase, int T)
    {
        int half = _cfg.RopeDim / 2;
        int n = T * heads * half;
        c.SetPso(_psoRopeMulti);
        c.SetBuffer(x, 0, 0);
        c.SetInt(1, headDim); c.SetInt(2, _cfg.RopeDim); c.SetInt(3, posBase);
        c.SetFloat(4, _cfg.RopeBase); c.SetInt(5, heads); c.SetInt(6, n);
        c.Dispatch((nuint)((n + 255) / 256), 1, 1, 256, 1, 1);
    }

    // (q4k, q6k, q8_0, q2_0c, stq1_0)
    private IntPtr PsoFor(string name, int inDim,
        (IntPtr q4, IntPtr q6, IntPtr q8, IntPtr q2, IntPtr stq) p)
    {
        GgmlTensorType t = _wtype[name];
        int granule = t switch
        {
            GgmlTensorType.Q4_K or GgmlTensorType.Q6_K or GgmlTensorType.STQ1_0 => 256,
            GgmlTensorType.Q2_0C => 512,
            GgmlTensorType.Q8_0 => 32,
            _ => 1,
        };
        if (inDim % granule != 0)
            throw new NotSupportedException($"gemv/gemm {name}: in_dim {inDim} not a multiple of {t} granule {granule}");
        IntPtr pso = t switch
        {
            GgmlTensorType.Q4_K => p.q4,
            GgmlTensorType.Q6_K => p.q6,
            GgmlTensorType.Q8_0 => p.q8,
            GgmlTensorType.Q2_0C => p.q2,
            GgmlTensorType.STQ1_0 => p.stq,
            _ => throw new NotSupportedException(
                $"gemv/gemm {name}: {t} not supported on Metal backend — use --backend cpu"),
        };
        // A nil PSO would make setComputePipelineState: a silent no-op; fail fast instead.
        if (pso == IntPtr.Zero)
            throw new NotSupportedException(
                $"gemv/gemm {name}: {t} not supported on Metal backend — use --backend cpu");
        return pso;
    }

    private void Gemv(CmdCtx c, string name, IntPtr x, IntPtr y, int inDim, int outDim)
        => Gemv(c, name, x, 0, y, 0, inDim, outDim);

    private void Gemv(CmdCtx c, string name, IntPtr x, nuint xOff, IntPtr y, nuint yOff, int inDim, int outDim)
    {
        IntPtr w = W(name);
        // fast4 K-quant kernels stage scales in threadgroup sf[]:
        // q4k -> in_dim <= 6144 (sf[4*192] groups of 32), q6k -> sf[4*384] groups of 16.
        // q8_0/q2c/stq read weights per-block without that table — the bound doesn't apply.
        if (inDim > 6144 && _wtype[name] is GgmlTensorType.Q4_K or GgmlTensorType.Q6_K)
            throw new NotSupportedException($"gemv {name}: in_dim {inDim} exceeds fast4 K-quant staging limit 6144");
        c.SetPso(PsoFor(name, inDim, (_psoGemv, _psoGemvQ6, _psoGemvQ8, _psoGemvQ2, _psoGemvStq)));
        c.SetBuffer(w, 0, 0); c.SetBuffer(x, xOff, 1); c.SetBuffer(y, yOff, 2);
        c.SetInt(3, inDim); c.SetInt(4, outDim);
        c.Dispatch((nuint)((outDim + 3) / 4), 1, 1, 256, 1, 1);
    }

    // ---- M7 fused decode helpers ----
    // gemv_multi section: weight tensor name, output buffer + byte offset,
    // column count, and whether the dst stores bf16 (KV rows).
    private sealed record GSec(string W, IntPtr Dst, nuint DstOff, int Cols, bool Bf16 = false);

    private int TypeCode(string name, int inDim)
    {
        GgmlTensorType t = _wtype[name];
        int code = t switch
        {
            GgmlTensorType.Q4_K => 0,
            GgmlTensorType.Q6_K => 1,
            GgmlTensorType.Q8_0 => 2,
            GgmlTensorType.Q2_0C => 3,
            GgmlTensorType.STQ1_0 => 4,
            _ => -1,
        };
        if (code < 0)
            throw new NotSupportedException(
                $"gemv {name}: {t} not supported on Metal backend — use --backend cpu");
        // gemv_multi shares the fast4 sf[] staging bounds: q4k 192 entries,
        // q6k 384 → in_dim <= 6144 for K-quants.
        if (inDim > 6144 && code is 0 or 1)
            throw new NotSupportedException(
                $"gemv {name}: in_dim {inDim} exceeds the sf staging limit 6144 — use --backend cpu");
        return code;
    }

    // One dispatch computing up to 3 weight sections sharing input x.
    // accIn != 0 → each output accumulates: y[i] = accIn[i] + dot.
    private void GemvF(CmdCtx c, IntPtr x, int inDim, IntPtr accIn, params GSec[] secs)
    {
        if (secs.Length is < 1 or > 3)
            throw new ArgumentException("gemv_multi needs 1-3 sections", nameof(secs));
        c.SetPso(_psoGemvMulti);
        c.SetBuffer(x, 0, 0);
        int[] ncols = new int[4], types = new int[4];
        int flags = accIn != IntPtr.Zero ? 1 : 0;
        for (int s = 0; s < 3; s++)
        {
            GSec? g = s < secs.Length ? secs[s] : null;
            string wname = g?.W ?? secs[0].W;
            types[s] = TypeCode(wname, inDim);
            c.SetBuffer(g is null ? W(secs[0].W) : W(wname), 0, (nuint)(1 + s));
            c.SetBuffer(g?.Dst ?? secs[0].Dst, g?.DstOff ?? 0, (nuint)(4 + s));
            ncols[s] = g?.Cols ?? 0;
            if (g?.Bf16 == true) flags |= 1 << (2 + s);
        }
        c.SetBuffer(accIn != IntPtr.Zero ? accIn : x, 0, 7);
        c.SetInt(8, inDim);
        c.SetInt4(9, ncols[0], ncols[1], ncols[2], 0);
        c.SetInt4(10, types[0], types[1], types[2], 0);
        c.SetInt(11, flags);
        int total = secs.Sum(g => g.Cols);
        c.Dispatch((nuint)((total + 3) / 4), 1, 1, 256, 1, 1);
    }

    // Staged scale entries a section needs: q4k → in/32, q6k → in/16.
    private static int SfEntries(int code, int inDim) =>
        code == 0 ? inDim >> 5 : (code == 1 ? inDim >> 4 : 0);

    // gate+up pair in one dispatch: y[i] = silu(dot_gate_i) * dot_up_i.
    // Falls back to gate+up+silu as three dispatches when both sections'
    // staged scales don't fit the 384-entry subgroup slice (e.g. q6k+q6k).
    private void GemvPair(CmdCtx c, IntPtr x, int inDim,
        string wGate, string wUp, IntPtr y, int cols)
    {
        int t0 = TypeCode(wGate, inDim), t1 = TypeCode(wUp, inDim);
        if (SfEntries(t0, inDim) + SfEntries(t1, inDim) > 384)
        {
            GemvF(c, x, inDim, IntPtr.Zero, new GSec(wGate, y, 0, cols));
            GemvF(c, x, inDim, IntPtr.Zero, new GSec(wUp, _up, 0, cols));
            c.SetPso(_psoSilu);
            c.SetBuffer(y, 0, 0); c.SetBuffer(_up, 0, 1);
            c.SetInt(2, cols);
            c.Dispatch((nuint)((cols + 255) / 256), 1, 1, 256, 1, 1);
            return;
        }
        c.SetPso(_psoGemvMulti);
        c.SetBuffer(x, 0, 0);
        c.SetBuffer(W(wGate), 0, 1); c.SetBuffer(W(wUp), 0, 2); c.SetBuffer(W(wUp), 0, 3);
        c.SetBuffer(y, 0, 4); c.SetBuffer(y, 0, 5); c.SetBuffer(y, 0, 6);
        c.SetBuffer(x, 0, 7);
        c.SetInt(8, inDim);
        c.SetInt4(9, cols, cols, 0, 0);
        c.SetInt4(10, t0, t1, 0, 0);
        c.SetInt(11, 2);  // pair-silu flag
        c.Dispatch((nuint)((cols + 3) / 4), 1, 1, 256, 1, 1);
    }

    // rope + per-head rmsnorm for q (in place) and k (straight into the
    // bf16 KV row for pos) — replaces rope×2 + rms×2 + kv_append.
    private void RopeRms(CmdCtx c, int l, int pos)
    {
        int heads = _cfg.NumHeads, kvHeads = _cfg.NumKvHeads, dim = _cfg.HeadDim;
        if (dim > 256)
            throw new NotSupportedException(
                $"rope_rms stages a head row in threadgroup memory — headDim {dim} > 256 unsupported");
        long prow = (long)_tab[pos >> BlkShift] * BlkTok + (pos & (BlkTok - 1));
        nuint kOff = (nuint)prow * (nuint)_kvStride * sizeof(ushort);
        c.SetPso(_psoRopeRms);
        c.SetBuffer(_q, 0, 0);
        c.SetBuffer(_k, 0, 1);
        c.SetBuffer(_kvK[l], kOff, 2);
        c.SetBuffer(W($"blk.{l}.attn_q_norm.weight"), 0, 3);
        c.SetBuffer(W($"blk.{l}.attn_k_norm.weight"), 0, 4);
        c.SetInt(5, heads); c.SetInt(6, dim); c.SetInt(7, _cfg.RopeDim);
        c.SetInt(8, pos); c.SetFloat(9, _cfg.RopeBase); c.SetFloat(10, _cfg.Eps);
        c.SetInt(11, _kvStride);
        c.Dispatch((nuint)(heads + kvHeads), 1, 1, 256, 1, 1);
    }

    // x -> xT[k][t] (stride Tpad), 32x32 tiled transpose.
    private void Xt(CmdCtx c, IntPtr x, IntPtr xT, int dim, int T, int Tpad)
    {
        c.SetPso(_psoXt);
        c.SetBuffer(x, 0, 0); c.SetBuffer(xT, 0, 1);
        c.SetInt(2, dim); c.SetInt(3, T); c.SetInt(4, Tpad);
        c.Dispatch((nuint)((dim + 31) / 32), (nuint)(Tpad / 32), 1, 256, 1, 1);
    }

    // Lazily dequantize a weight tensor into a persistent fp32 buffer
    // ([out][in] rows) and return it; later calls reuse the cached copy.
    private IntPtr WFp32(CmdCtx c, string name, int inDim, int outDim)
    {
        if (_wfp32.TryGetValue(name, out IntPtr buf)) return buf;
        buf = _dev.NewBuffer((nuint)inDim * (nuint)outDim * 4);
        _wfp32[name] = buf;
        IntPtr w = W(name);
        c.SetPso(PsoFor(name, inDim, (_psoDeqQ4, _psoDeqQ6, _psoDeqQ8, _psoDeqQ2, _psoDeqStq)));
        c.SetBuffer(w, 0, 0); c.SetBuffer(buf, 0, 1);
        c.SetInt(2, inDim); c.SetInt(3, outDim);
        int cover = _wtype[name] switch   // elements each dequant thread writes
        {
            GgmlTensorType.Q6_K => 16,
            GgmlTensorType.Q8_0 => 32,
            _ => 64,
        };
        int units = (inDim + cover - 1) / cover;
        c.Dispatch((nuint)((units + 31) / 32), (nuint)outDim, 1, 32, 1, 1);
        return buf;
    }

    // [T x inDim] · W^T -> [T x outDim] via mma_gemm (simdgroup MMA, ~2x
    // scalar FMA on this GPU) on fp32 weights + xT, with sgemm4x4 covering
    // the ragged row/col edges mma's 64x64 tiles don't reach.
    private void Gemm(CmdCtx c, string name, IntPtr xT, IntPtr y, int inDim, int outDim, int T, int Tpad)
    {
        // sgemm4x4 writes y rows as float4 when c+3 < out_dim — requires out_dim % 4 == 0
        // so every row stays 16B-aligned (this model: 2048/512/6144/3072 all qualify).
        if (outDim % 4 != 0)
            throw new NotSupportedException($"gemm {name}: out_dim {outDim} not a multiple of 4 — use --backend cpu");
        IntPtr w32 = WFp32(c, name, inDim, outDim);

        int tFull = T & ~63, cFull = outDim & ~63;
        bool useMma = _psoMma != IntPtr.Zero && inDim % 16 == 0 && tFull > 0 && cFull > 0;
        if (useMma)
        {
            c.SetPso(_psoMma);
            c.SetBuffer(w32, 0, 0); c.SetBuffer(xT, 0, 1); c.SetBuffer(y, 0, 2);
            c.SetInt(3, inDim); c.SetInt(4, outDim); c.SetInt(5, T); c.SetInt(6, Tpad);
            c.Dispatch((nuint)(cFull / 64), (nuint)(tFull / 64), 1, 256, 1, 1);
        }
        // column tail (all rows) + row tail (full-tile cols only) via sgemm4x4
        int cTail = outDim - cFull;
        if (!useMma || cTail > 0)
        {
            int cBase = useMma ? cFull : 0;
            Sgemm(c, w32, xT, y, inDim, outDim, T, Tpad, 0, cBase,
                (outDim - cBase + 63) / 64, Tpad / 64);
        }
        if (useMma && T - tFull > 0)
        {
            Sgemm(c, w32, xT, y, inDim, outDim, T, Tpad, tFull, 0,
                cFull / 64, (T - tFull + 63) / 64);
        }
    }

    private void Sgemm(CmdCtx c, IntPtr w32, IntPtr xT, IntPtr y,
        int inDim, int outDim, int T, int Tpad, int tBase, int cBase, int gx, int gy)
    {
        c.SetPso(_psoSgemm);
        c.SetBuffer(w32, 0, 0); c.SetBuffer(xT, 0, 1); c.SetBuffer(y, 0, 2);
        c.SetInt(3, inDim); c.SetInt(4, outDim); c.SetInt(5, T); c.SetInt(6, Tpad);
        c.SetInt(7, tBase); c.SetInt(8, cBase);
        c.Dispatch((nuint)gx, (nuint)gy, 1, 256, 1, 1);
    }

    private void Rms(CmdCtx c, IntPtr x, IntPtr w, IntPtr y, int rows, int dim)
    {
        c.SetPso(_psoRms);
        c.SetBuffer(x, 0, 0); c.SetBuffer(w, 0, 1); c.SetBuffer(y, 0, 2);
        c.SetInt(3, dim); c.SetFloat(4, _cfg.Eps);
        c.Dispatch((nuint)rows, 1, 1, 256, 1, 1);
    }

    private void Rope(CmdCtx c, IntPtr x, int heads, int headDim, int pos)
    {
        int half = _cfg.RopeDim / 2;
        c.SetPso(_psoRope);
        c.SetBuffer(x, 0, 0);
        c.SetInt(1, headDim); c.SetInt(2, _cfg.RopeDim); c.SetInt(3, pos); c.SetFloat(4, _cfg.RopeBase);
        c.SetInt(5, heads * half);
        c.Dispatch((nuint)((heads * half + 255) / 256), 1, 1, 256, 1, 1);
    }

    private void Add(CmdCtx c, IntPtr a, IntPtr b, int n)
    {
        c.SetPso(_psoAdd);
        c.SetBuffer(a, 0, 0); c.SetBuffer(b, 0, 1);
        c.SetInt(2, n);
        c.Dispatch((nuint)((n + 255) / 256), 1, 1, 256, 1, 1);
    }

    private static void DumpF(IntPtr buf, int n, string name)
    {
        float[] a = new float[n];
        Marshal.Copy((IntPtr)Contents(buf), a, 0, n);
        File.WriteAllBytes($"/tmp/metal_{name}.bin",
            a.SelectMany(BitConverter.GetBytes).ToArray());
    }

    public void Dispose()
    {
        foreach (IntPtr b in _wfp32.Values) ObjC.Release(b);
        /* TODO(M2): release device/queue/buffers on the real backend lifecycle */
    }
}
