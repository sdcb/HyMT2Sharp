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
    private IntPtr _psoGemm, _psoGemmQ6, _psoGemmQ8, _psoGemmQ2, _psoGemmStq;
    private IntPtr _psoEmbedQ4, _psoEmbedQ6, _psoEmbedQ8, _psoRms, _psoRope, _psoKvAppend, _psoAttn, _psoSilu, _psoAdd;
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
        _psoSilu = NewPso(lib2, "silu_mul");
        _psoAdd = NewPso(lib2, "add_inplace");
        _psoGemm = NewPso(lib2, "q4k_gemm");
        _psoGemmQ6 = NewPso(lib2, "q6k_gemm");
        _psoGemmQ8 = NewPso(lib2, "q8_0_gemm");
        _psoGemmQ2 = NewPso(lib2, "q2c_gemm");
        _psoGemmStq = NewPso(lib2, "stq_gemm");
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

        for (int l = 0; l < c.NumLayers; l++)
        {
            // n1 = rmsnorm(h, attn_norm)
            Rms(cc, _h, W($"blk.{l}.attn_norm.weight"), _n1, 1, hidden);
            Gemv(cc, $"blk.{l}.attn_q.weight", _n1, _q, hidden, qDim);
            Gemv(cc, $"blk.{l}.attn_k.weight", _n1, _k, hidden, kDim);
            Gemv(cc, $"blk.{l}.attn_v.weight", _n1, _v, hidden, kDim);
            Rope(cc, _q, heads, dim, pos);
            Rope(cc, _k, kvHeads, dim, pos);
            Rms(cc, _q, W($"blk.{l}.attn_q_norm.weight"), _q, heads, dim);
            Rms(cc, _k, W($"blk.{l}.attn_k_norm.weight"), _k, kvHeads, dim);
            // append K/V at pos
            cc.SetPso(_psoKvAppend);
            cc.SetBuffer(_k, 0, 0); cc.SetBuffer(_v, 0, 1);
            cc.SetBuffer(_kvK[l], 0, 2); cc.SetBuffer(_kvV[l], 0, 3);
            cc.SetInt(4, kDim); cc.SetInt(5, _kvStride); cc.SetInt(6, pos);
            cc.SetBuffer(_kvTab, 0, 7); cc.SetInt(8, BlkShift);
            cc.Dispatch((nuint)((kDim + 255) / 256), 1, 1, 256, 1, 1);
            // attention -> ao
            cc.SetPso(_psoAttn);
            cc.SetBuffer(_q, 0, 0); cc.SetBuffer(_kvK[l], 0, 1); cc.SetBuffer(_kvV[l], 0, 2);
            cc.SetBuffer(_ao, 0, 3);
            cc.SetInt(4, heads); cc.SetInt(5, kvHeads); cc.SetInt(6, dim);
            cc.SetInt(7, _kvStride); cc.SetInt(8, kvLen); cc.SetFloat(9, scale);
            cc.SetBuffer(_kvTab, 0, 10); cc.SetInt(11, BlkShift);
            cc.Dispatch((nuint)heads, 1, 1, 128, 1, 1);
            Gemv(cc, $"blk.{l}.attn_output.weight", _ao, _attnOut, qDim, hidden);
            Add(cc, _h, _attnOut, hidden);

            // FFN
            Rms(cc, _h, W($"blk.{l}.ffn_norm.weight"), _n2, 1, hidden);
            Gemv(cc, $"blk.{l}.ffn_gate.weight", _n2, _gate, hidden, c.FfnSize);
            Gemv(cc, $"blk.{l}.ffn_up.weight", _n2, _up, hidden, c.FfnSize);
            cc.SetPso(_psoSilu);
            cc.SetBuffer(_gate, 0, 0); cc.SetBuffer(_up, 0, 1);
            cc.SetInt(2, c.FfnSize);
            cc.Dispatch((nuint)((c.FfnSize + 255) / 256), 1, 1, 256, 1, 1);
            Gemv(cc, $"blk.{l}.ffn_down.weight", _gate, _down, c.FfnSize, hidden);
            Add(cc, _h, _down, hidden);
        }

        Rms(cc, _h, _outNorm, _normed, 1, hidden);
        Gemv(cc, "output.weight", _normed, _logitsBuf, hidden, _vocab);

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

        var cc = CmdCtx.Begin(_dev.Queue);
        ExecCopies(cc);
        cc.SetPso(PsoFor("token_embd.weight", hidden,
            (_psoEmbedRowsQ4, _psoEmbedRowsQ6, _psoEmbedRowsQ8, default, default)));
        cc.SetBuffer(_embd, 0, 0); cc.SetBuffer(toks, 0, 1); cc.SetBuffer(h, 0, 2);
        cc.SetInt(3, hidden);
        cc.Dispatch((nuint)T, 1, 1, 256, 1, 1);

        for (int l = 0; l < c.NumLayers; l++)
        {
            Rms(cc, h, W($"blk.{l}.attn_norm.weight"), n1, T, hidden);
            Gemm(cc, $"blk.{l}.attn_q.weight", n1, q, hidden, qDim, T);
            Gemm(cc, $"blk.{l}.attn_k.weight", n1, kb, hidden, kDim, T);
            Gemm(cc, $"blk.{l}.attn_v.weight", n1, vb, hidden, kDim, T);
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
            Gemm(cc, $"blk.{l}.attn_output.weight", ao, attnOut, qDim, hidden, T);
            Add(cc, h, attnOut, T * hidden);

            Rms(cc, h, W($"blk.{l}.ffn_norm.weight"), n2, T, hidden);
            Gemm(cc, $"blk.{l}.ffn_gate.weight", n2, gate, hidden, ffn, T);
            Gemm(cc, $"blk.{l}.ffn_up.weight", n2, up, hidden, ffn, T);
            cc.SetPso(_psoSilu);
            cc.SetBuffer(gate, 0, 0); cc.SetBuffer(up, 0, 1);
            cc.SetInt(2, T * ffn);
            cc.Dispatch((nuint)((T * ffn + 255) / 256), 1, 1, 256, 1, 1);
            Gemm(cc, $"blk.{l}.ffn_down.weight", gate, down, ffn, hidden, T);
            Add(cc, h, down, T * hidden);
        }

        Rms(cc, h, _outNorm, normed, T, hidden);
        Gemv(cc, "output.weight", normed, (nuint)((T - 1) * hidden * 4), _logitsBuf, 0, hidden, _vocab);

        cc.EndEnc();
        cc.Commit();
        cc.Wait();
        // GPU is done with the scratch; newBuffer* returned +1 objects.
        foreach (IntPtr b in new[]
            { toks, h, n1, n2, q, kb, vb, ao, attnOut, gate, up, down, normed, scores })
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
        return t switch
        {
            GgmlTensorType.Q4_K => p.q4,
            GgmlTensorType.Q6_K => p.q6,
            GgmlTensorType.Q8_0 => p.q8,
            GgmlTensorType.Q2_0C => p.q2,
            GgmlTensorType.STQ1_0 => p.stq,
            _ => throw new NotSupportedException(
                $"gemv/gemm {name}: {t} not supported on Metal backend — use --backend cpu"),
        };
    }

    private void Gemv(CmdCtx c, string name, IntPtr x, IntPtr y, int inDim, int outDim)
        => Gemv(c, name, x, 0, y, 0, inDim, outDim);

    private void Gemv(CmdCtx c, string name, IntPtr x, nuint xOff, IntPtr y, nuint yOff, int inDim, int outDim)
    {
        IntPtr w = W(name);
        // fast4 kernels stage scales in threadgroup sf[]:
        // q4k -> in_dim <= 6144 (sf[4*192] groups of 32), q6k -> sf[4*384] groups of 16.
        if (inDim > 6144)
            throw new NotSupportedException($"gemv {name}: in_dim {inDim} exceeds fast4 limit 6144");
        c.SetPso(PsoFor(name, inDim, (_psoGemv, _psoGemvQ6, _psoGemvQ8, _psoGemvQ2, _psoGemvStq)));
        c.SetBuffer(w, 0, 0); c.SetBuffer(x, xOff, 1); c.SetBuffer(y, yOff, 2);
        c.SetInt(3, inDim); c.SetInt(4, outDim);
        c.Dispatch((nuint)((outDim + 3) / 4), 1, 1, 256, 1, 1);
    }

    // [T x inDim] · W^T -> [T x outDim]; TILE_T=8 token rows per threadgroup tile.
    private void Gemm(CmdCtx c, string name, IntPtr x, IntPtr y, int inDim, int outDim, int T)
    {
        IntPtr w = W(name);
        if (inDim > 6144)
            throw new NotSupportedException($"gemm {name}: in_dim {inDim} exceeds fast4 limit 6144");
        c.SetPso(PsoFor(name, inDim, (_psoGemm, _psoGemmQ6, _psoGemmQ8, _psoGemmQ2, _psoGemmStq)));
        c.SetBuffer(w, 0, 0); c.SetBuffer(x, 0, 1); c.SetBuffer(y, 0, 2);
        c.SetInt(3, inDim); c.SetInt(4, outDim); c.SetInt(5, T);
        c.Dispatch((nuint)((outDim + 3) / 4), (nuint)((T + 7) / 8), 1, 256, 1, 1);  // TILE_T=8
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

    public void Dispose() { /* TODO(M2): release device/queue/buffers on the real backend lifecycle */ }
}
