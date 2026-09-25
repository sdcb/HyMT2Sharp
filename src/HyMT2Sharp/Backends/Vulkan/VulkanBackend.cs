// Vulkan decode backend (M2): the whole decode step is pre-recorded once into
// a reusable command buffer at LoadModel; per-token params (tokenId, pos,
// kvLen) flow through a HOST_VISIBLE SSBO read by kernels — nothing in the CB
// is patched per token. Descriptor sets are allocated once at record time and
// never updated while in flight.
using System.Diagnostics;
using System.Reflection;
using System.Text;
using System.Runtime.InteropServices;
using Sdcb.HyMT2Sharp.Gguf;
using Sdcb.HyMT2Sharp.Model;

namespace Sdcb.HyMT2Sharp.Backends.Vulkan;

public sealed unsafe class VulkanBackend : IComputeBackend
{
    private readonly VkDevice _dev;
    private ModelConfig _cfg = null!;
    private readonly Dictionary<string, VkBuffer> _w = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgmlTensorType> _wtype = new(StringComparer.Ordinal);
    private VkPipeline _pGemv4 = null!, _pGemv6 = null!, _pEmbed4 = null!, _pEmbed6 = null!,
        _pRms = null!,
        _pPreKv = null!, _pGemvAdd4 = null!, _pGemvAdd6 = null!, _pPreFfn = null!,
        _pAttn2 = null!, _pFfnGu = null!;
    private VkPipeline _pPfDeq4 = null!, _pPfDeq6 = null!, _pPfEmbed = null!, _pPfRms = null!,
        _pPfGemm = null!, _pPfKvPrep = null!, _pPfAttn = null!, _pPfSilu = null!, _pPfRmsX = null!,
        _pPfQprep = null!, _pPfQk = null!, _pPfSoft = null!, _pPfPv = null!;
    private VkBuffer[] _kvK = null!, _kvV = null!;
    private ushort*[] _kvKHost = null!, _kvVHost = null!; // persistent fp16 maps for UploadKv
    private int _kvCap, _kvStride;
    private VkBuffer _embd = null!, _outNorm = null!;
    private VkBuffer _h = null!, _n1 = null!, _n2 = null!, _q = null!, _k = null!, _v = null!,
        _ao = null!, _attnOut = null!, _gate = null!, _up = null!, _down = null!, _normed = null!,
        _logitsBuf = null!, _prm = null!, _logitsStage = null!;
    private int* _prmPtr;                     // {tokenId, pos, kvLen}
    private IntPtr _cmd, _fence, _qpool;

    // ---- M3 prefill state ----
    private readonly Dictionary<string, long> _welems = new(StringComparer.Ordinal);
    private VkBuffer[] _wq16 = null!, _wo16 = null!, _wgu16 = null!, _wd16 = null!;
    private VkBuffer _embd16 = null!, _pfTok = null!, _pfX = null!, _pfXs = null!, _pfXs2 = null!,
        _pfQkv = null!, _pfAo = null!, _pfGu = null!, _pfH = null!;
    private int* _pfTokPtr;
    private IntPtr _cmdPf;
    private readonly List<IntPtr> _pfSetCache = new();
    private int _pfSetIdx;
    private const int PfCap = 1024;
    private readonly bool _noPf = Environment.GetEnvironmentVariable("HYMT_VK_NOPF") == "1";

    private float[] _logitsHost = null!;
    private int _vocab;
    private int _pfGemmTM;
    private int _pfGemmTN = 128;
    private readonly bool _timing = Environment.GetEnvironmentVariable("HYMT_VK_TIMING") == "1";
    private readonly bool _noSmall = Environment.GetEnvironmentVariable("HYMT_VK_NOSMALL") == "1";
    private readonly bool _noAttn = Environment.GetEnvironmentVariable("HYMT_VK_NOATTN") == "1";
    private readonly bool _noBarrier = Environment.GetEnvironmentVariable("HYMT_VK_NOBARRIER") == "1";
    private readonly bool _noPush = Environment.GetEnvironmentVariable("HYMT_VK_NOPUSH") == "1";
    private int _seq;
    private bool _kvF32;
    private bool _splitFfn = Environment.GetEnvironmentVariable("HYMT_VK_SPLITFFN") == "1";
    private bool _dump;
    private bool _pfFastAttn;
    private VkBuffer _pfQ16 = null!, _pfS32 = null!, _pfP16 = null!;
    private bool _gVar;
    private int _step;
    private readonly Stopwatch _sw = new();
    private double _gpuMs; private int _n;
    private double[] _segMs = new double[4];
    private double[] _opMs = new double[6];

    public string Name => "vulkan";
    // pf kernels assume the fp16 KV layout; KVF32 (debug knob) can't prefill.
    public bool SupportsPrefill => !_kvF32 && !_noPf;

    public VulkanBackend()
    {
        _dev = VkDevice.Create();
        if (!_dev.Storage16Bit)
        {
            _dev.Dispose();
            throw new PlatformNotSupportedException(
                "Vulkan backend needs storageBuffer16BitAccess + shaderInt16 (fp16 KV, Q6_K views)");
        }
    }

    /// <summary>Cheap probe: a real (non-software) Vulkan device with compute exists.</summary>
    public static bool IsSupported()
    {
        try
        {
            using VkDevice d = VkDevice.Create();
            return d.Storage16Bit;
        }
        catch { return false; }
    }

    private VkBuffer W(string name) => _w.TryGetValue(name, out VkBuffer? b) ? b : throw new KeyNotFoundException(name);

    public void LoadModel(GgufFile gguf, ModelConfig cfg)
    {
        _cfg = cfg;
        VkPipeline Mk(string spv, int bindings, int pushBytes)
            => _dev.NewPipeline(_dev.NewShaderModule(LoadSpv(spv)), bindings, pushBytes);

        _pGemv4 = Mk("q4k_gemv3", bindings: 3, pushBytes: 8);
        _pGemv6 = Mk("q6k_gemv2", bindings: 3, pushBytes: 8);
        _pEmbed4 = Mk("dec_embed_q4k", bindings: 3, pushBytes: 4);
        _pEmbed6 = Mk("dec_embed_q6k", bindings: 3, pushBytes: 4);
        _pRms = Mk("dec_rmsnorm", bindings: 3, pushBytes: 8);
        _kvF32 = Environment.GetEnvironmentVariable("HYMT_VK_KVF32") == "1";
        _pAttn2 = Mk("dec_attn2", bindings: 11, pushBytes: 32);
        _pFfnGu = Mk("dec_ffngu", bindings: 4, pushBytes: 16);
        _pPreKv = Mk("dec_prekv", bindings: 8, pushBytes: 28);
        _pGemvAdd4 = Mk("dec_gemvadd_q4k", bindings: 3, pushBytes: 8);
        _pGemvAdd6 = Mk("dec_gemvadd_q6k", bindings: 3, pushBytes: 8);
        _pPreFfn = Mk("dec_preffn", bindings: 5, pushBytes: 20);
        _pPfDeq4 = Mk("pf_deq_q4k", bindings: 2, pushBytes: 12);
        _pPfDeq6 = Mk("pf_deq_q6k", bindings: 2, pushBytes: 12);
        _pPfEmbed = Mk("pf_embed", bindings: 3, pushBytes: 4);
        _pPfRms = Mk("pf_rms", bindings: 3, pushBytes: 16);
        bool useCm = _dev.CoopMatrix && _dev.SubgroupSizeControl
            && Environment.GetEnvironmentVariable("HYMT_VK_NOCM") != "1";
        string cmVar = Environment.GetEnvironmentVariable("HYMT_VK_GEMMV") ?? "g28";
        _gVar = useCm && cmVar.StartsWith("g");
        _pfGemmTM = useCm
            ? (int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_GEMMTM"), out int tm) ? tm : 512)
            : 128;
        _pfGemmTN = useCm
            ? (int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_GEMMTN"), out int tn) ? tn : 64)
            : 128;
        try
        {
            _pPfGemm = useCm
                ? _dev.NewPipeline(_dev.NewShaderModule(LoadSpv(
                    Environment.GetEnvironmentVariable("HYMT_VK_F16ACC") == "1" ? "pf_gemm_cm_f16"
                    : "pf_gemm_cm_" + cmVar)),
                    4, 16, requiredSubgroupSize: 16)
                : Mk("pf_gemm", bindings: 4, pushBytes: 16);
            _pPfRmsX = _gVar ? Mk("pf_rms16", bindings: 3, pushBytes: 16) : _pPfRms;
            _pPfKvPrep = Mk("pf_kvprep", bindings: 5, pushBytes: 32);
            _pPfAttn = Mk(_gVar ? "pf_attn16" : "pf_attn", bindings: 6, pushBytes: 36);
            _pPfSilu = Mk(_gVar ? "pf_silumul16" : "pf_silumul", bindings: 2, pushBytes: 8);
            _pfFastAttn = _gVar && useCm && Environment.GetEnvironmentVariable("HYMT_VK_NOFASTATTN") != "1";
            if (_pfFastAttn)
            {
                _pPfQprep = Mk("pf_qprep", bindings: 4, pushBytes: 24);
                _pPfQk = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_qk")), 3, 28, requiredSubgroupSize: 16);
                _pPfSoft = Mk("pf_soft", bindings: 3, pushBytes: 16);
                _pPfPv = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_pv")), 3, 32, requiredSubgroupSize: 16);
            }
        }
        catch (Exception e) when (useCm)
        {
            // e.g. sg32 devices (NVIDIA) reject requiredSubgroupSize=16 even with
            // VK_EXT_subgroup_size_control — fall back to the scalar GEMM path.
            Console.Error.WriteLine($"[vk] coopmat pipeline failed ({e.Message}) — scalar pf_gemm fallback");
            useCm = false; _gVar = false; _pfFastAttn = false;
            _pfGemmTM = _pfGemmTN = 128;
            _pPfGemm = Mk("pf_gemm", bindings: 4, pushBytes: 16);
            _pPfRmsX = _pPfRms;
            _pPfKvPrep = Mk("pf_kvprep", bindings: 5, pushBytes: 32);
            _pPfAttn = Mk("pf_attn", bindings: 6, pushBytes: 36);
            _pPfSilu = Mk("pf_silumul", bindings: 2, pushBytes: 8);
        }

        foreach ((string name, GgufTensorInfo info) in gguf.Tensors)
        {
            ulong bytes = info.Type switch
            {
                GgmlTensorType.F32 => info.NumElements * sizeof(float),
                GgmlTensorType.Q4_K => info.NumElements / 256 * (ulong)144,
                GgmlTensorType.Q6_K => info.NumElements / 256 * (ulong)210,
                _ => throw new NotSupportedException(
                    $"Vulkan backend (M2) supports F32/Q4_K/Q6_K only; {name} is {info.Type} — use --backend cpu"),
            };
            VkBuffer b = _dev.NewStorageBuffer(bytes, hostVisible: _dev.CoherentDeviceLocal);
            _dev.Upload(b, gguf.DataBase + (long)info.Offset, bytes);
            _w[name] = b;
            _wtype[name] = info.Type;
            _welems[name] = (long)info.NumElements;
        }

        _embd = W("token_embd.weight");
        _outNorm = W("output_norm.weight");
        if (!_w.ContainsKey("output.weight"))
        {
            _w["output.weight"] = _embd;      // tied lm_head
            _wtype["output.weight"] = _wtype["token_embd.weight"];
        }

        _kvStride = cfg.NumKvHeads * cfg.HeadDim;
        _kvCap = Math.Min(cfg.ContextLength, 4096);   // attn kernel shared-mem cap
        _kvK = new VkBuffer[cfg.NumLayers];
        _kvV = new VkBuffer[cfg.NumLayers];
        _kvKHost = new ushort*[cfg.NumLayers];
        _kvVHost = new ushort*[cfg.NumLayers];
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            int kvBytes = _kvF32 ? 4 : 2;
            _kvK[l] = _dev.NewStorageBuffer((ulong)(_kvCap * _kvStride * kvBytes), hostVisible: true);
            _kvV[l] = _dev.NewStorageBuffer((ulong)(_kvCap * _kvStride * kvBytes), hostVisible: true);
            _kvKHost[l] = (ushort*)_kvK[l].Map();
            _kvVHost[l] = (ushort*)_kvV[l].Map();
        }

        int hidden = cfg.HiddenSize, ffn = cfg.FfnSize;
        int qDim = cfg.NumHeads * cfg.HeadDim, kDim = _kvStride;
        _vocab = cfg.VocabSize;
        _dump = Environment.GetEnvironmentVariable("HYMT_VK_DUMP") == "1";
        VkBuffer DevBuf(int bytes) => _dev.NewStorageBuffer((ulong)bytes, hostVisible: _dump);
        _h = DevBuf(hidden * 4); _n1 = DevBuf(hidden * 4); _n2 = DevBuf(hidden * 4);
        _q = DevBuf(qDim * 4); _k = DevBuf(kDim * 4); _v = DevBuf(kDim * 4);
        _ao = DevBuf(qDim * 4); _attnOut = DevBuf(hidden * 4);
        _gate = DevBuf(ffn * 4); _up = DevBuf(ffn * 4); _down = DevBuf(hidden * 4);
        _normed = DevBuf(hidden * 4);
        _logitsBuf = DevBuf(_vocab * 4);
        // CPU reads of device-local (WC) memory are ~15MB/s: the kernel writes logits
        // to VRAM, then a burst copy at CB end lands them in a host-side staging buffer.
        _logitsStage = _dev.NewStorageBuffer((ulong)(_vocab * 4), hostVisible: true, preferHost: true);
        // Host writes to device-local are fast (write-combining); only reads are slow.
        _prm = _dev.NewStorageBuffer(16, hostVisible: true);
        _prmPtr = (int*)_prm.Map();
        _logitsHost = new float[_vocab];

        _cmd = _dev.NewCommandBuffer();
        _fence = _dev.NewFence();
        Vk.VkQueryPoolCreateInfo qpci = new() { SType = 11, QueryType = 2, QueryCount = 16 };
        Vk.Check(Vk.vkCreateQueryPool(_dev.Device, &qpci, null, out _qpool), "vkCreateQueryPool");
        RecordDecodeGraph();
        if (SupportsPrefill) InitPrefill();
    }

    // ---- descriptor-set helpers ----

    private (IntPtr set, VkBuffer[] bufs) Set(VkPipeline p, params VkBuffer[] bufs)
    {
        IntPtr s = IntPtr.Zero;
        if (!_dev.PushDescriptors || _noPush)
        {
            s = _dev.NewDescriptorSet(p.SetLayout);
            for (uint i = 0; i < bufs.Length; i++) _dev.BindBuffer(s, i, bufs[i]);
        }
        return (s, bufs);
    }

    /// <summary>Record one compute dispatch + write->read barrier into `cmd`.</summary>
    private void CmdRun(IntPtr cmd, VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, uint gy, params VkBuffer[] written)
    {
        Vk.vkCmdBindPipeline(cmd, VkConst.BindPointCompute, p.Pipeline);
        if (_dev.PushDescriptors && !_noPush)
        {
            int n = d.bufs.Length;
            Vk.VkWriteDescriptorSet* w = stackalloc Vk.VkWriteDescriptorSet[n];
            Vk.VkDescriptorBufferInfo* bi = stackalloc Vk.VkDescriptorBufferInfo[n];
            for (int i = 0; i < n; i++)
            {
                bi[i] = new Vk.VkDescriptorBufferInfo { Buffer = d.bufs[i].Buffer, Offset = 0, Range = d.bufs[i].Size };
                w[i] = new Vk.VkWriteDescriptorSet
                {
                    SType = VkConst.StWriteDescriptorSet, DstBinding = (uint)i,
                    DescriptorCount = 1, DescriptorType = VkConst.DescStorageBuffer,
                    PBufferInfo = &bi[i],
                };
            }
            _dev.CmdPushDescriptors(cmd, p.Layout, w, (uint)n);
        }
        else
        {
            IntPtr set = d.set;
            Vk.vkCmdBindDescriptorSets(cmd, VkConst.BindPointCompute, p.Layout, 0, 1, &set, 0, null);
        }
        if (pcBytes > 0) Vk.vkCmdPushConstants(cmd, p.Layout, VkConst.StageComputeShader, 0, pcBytes, pc);
        Vk.vkCmdDispatch(cmd, gx, gy, 1);
        if (!_noBarrier && written.Length > 0)
            _dev.CmdBufferBarrier(cmd, written);
    }

    // ---- recording ----

    private void RecordDecodeGraph()
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, heads = c.NumHeads, kvHeads = c.NumKvHeads;
        int dim = c.HeadDim, qDim = heads * dim, kDim = _kvStride;
        float scale = 1f / MathF.Sqrt(dim);

        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmd, &begin), "vkBeginCommandBuffer");

        Vk.vkCmdResetQueryPool(_cmd, _qpool, 0, 16);
        Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageTopOfPipe, _qpool, 0);

        uint* pcs = stackalloc uint[10]; // shared push-constant scratch (dims baked at record)

        void Run(VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, params VkBuffer[] written)
            => CmdRun(_cmd, p, d, pc, pcBytes, gx, 1, written);

        void Gemv(string name, VkBuffer x, VkBuffer y, int inDim, int outDim)
        {
            GgmlTensorType t = _wtype[name];
            VkPipeline p = t == GgmlTensorType.Q4_K ? _pGemv4
                : t == GgmlTensorType.Q6_K ? _pGemv6
                : throw new NotSupportedException($"gemv {name}: {t} unsupported on Vulkan — use --backend cpu");
            var s = t == GgmlTensorType.Q4_K
                ? Set(p, W(name), x, y, W(name))
                : Set(p, W(name), x, y);
            pcs[0] = (uint)inDim; pcs[1] = (uint)outDim;
            Run(p, s, pcs, 8, ((uint)outDim + 15u) / 16u, y);
        }

        void Rms(VkBuffer x, VkBuffer wbuf, VkBuffer y, int rows, int rdim)
        {
            var s = Set(_pRms, x, wbuf, y);
            pcs[0] = (uint)rdim; pcs[1] = BitConverter.SingleToUInt32Bits(c.Eps);
            Run(_pRms, s, pcs, 8, (uint)rows, y);
        }

        // embed h = dequant(embd[tokenId])
        {
            VkPipeline p = _wtype["token_embd.weight"] == GgmlTensorType.Q6_K ? _pEmbed6 : _pEmbed4;
            var s = Set(p, _embd, _prm, _h);
            pcs[0] = (uint)hidden;
            Run(p, s, pcs, 4, (uint)((hidden + 255) / 256), _h);
        }

        Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 1);

        if (Environment.GetEnvironmentVariable("HYMT_VK_OUTFIRST") == "1")
            Gemv("output.weight", _normed, _logitsBuf, hidden, _vocab);

        uint DType(string name) => _wtype[name] switch
        {
            GgmlTensorType.Q4_K => 0u,
            GgmlTensorType.Q6_K => 1u,
            var t => throw new NotSupportedException($"vulkan decode: {name} is {t} — only Q4_K/Q6_K supported"),
        };

        // gemv accumulating into a residual buffer: h[row] += W*x
        void GemvAdd(VkPipeline p, string name, VkBuffer x, VkBuffer y, int inDim, int outDim)
        {
            var s = Set(p, W(name), x, y);
            pcs[0] = (uint)inDim; pcs[1] = (uint)outDim;
            Run(p, s, pcs, 8, ((uint)outDim + 15u) / 16u, y);
        }

        uint* pcs7 = stackalloc uint[11];
        int layers = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_LAYERS"), out int nl) ? nl : c.NumLayers;
        for (int l = 0; l < layers; l++)
        {
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 11);
            // 1) rms(h,attn_norm) -> xs(shared); gemv -> q,k,v
            {
                var s = Set(_pPreKv, _h, W($"blk.{l}.attn_norm.weight"),
                    W($"blk.{l}.attn_q.weight"), W($"blk.{l}.attn_k.weight"), W($"blk.{l}.attn_v.weight"),
                    _q, _k, _v);
                pcs7[0] = (uint)hidden; pcs7[1] = (uint)qDim; pcs7[2] = (uint)kDim;
                pcs7[3] = BitConverter.SingleToUInt32Bits(c.Eps);
                pcs7[4] = DType($"blk.{l}.attn_q.weight"); pcs7[5] = DType($"blk.{l}.attn_k.weight"); pcs7[6] = DType($"blk.{l}.attn_v.weight");
                Run(_pPreKv, s, pcs7, 28, (uint)((qDim + 2 * kDim + 15) / 16), _q, _k, _v);
            }
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 5);
            // 2) fused: rope(q,k) + per-head rmsnorm + kv append + attention
            {
                var s = Set(_pAttn2, _q, _k, _v, _kvK[l], _kvV[l], _kvK[l], _kvV[l],
                    W($"blk.{l}.attn_q_norm.weight"), W($"blk.{l}.attn_k_norm.weight"), _prm, _ao);
                pcs7[0] = (uint)heads; pcs7[1] = (uint)kvHeads; pcs7[2] = (uint)dim;
                pcs7[3] = (uint)c.RopeDim; pcs7[4] = (uint)_kvStride;
                pcs7[5] = BitConverter.SingleToUInt32Bits(scale);
                pcs7[6] = BitConverter.SingleToUInt32Bits(c.RopeBase);
                pcs7[7] = BitConverter.SingleToUInt32Bits(c.Eps);
                Run(_pAttn2, s, pcs7, 32, (uint)heads, _q, _k, _ao, _kvK[l], _kvV[l]);
            }
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 6);
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 7);
            // 4) h += attn_output * ao
            GemvAdd(_wtype[$"blk.{l}.attn_output.weight"] == GgmlTensorType.Q6_K ? _pGemvAdd6 : _pGemvAdd4,
                $"blk.{l}.attn_output.weight", _ao, _h, qDim, hidden);
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 8);
            // 5) rms(h,ffn_norm) -> xs; out = silu(gate(xs)) * up(xs) -> _gate
            if (_splitFfn)
            {
                Rms(_h, W($"blk.{l}.ffn_norm.weight"), _n1, 1, hidden);
                {
                    var s = Set(_pFfnGu, _n1, W($"blk.{l}.ffn_gate.weight"), W($"blk.{l}.ffn_up.weight"), _gate);
                    pcs[0] = (uint)hidden; pcs[1] = (uint)c.FfnSize;
                    pcs[2] = DType($"blk.{l}.ffn_gate.weight"); pcs[3] = DType($"blk.{l}.ffn_up.weight");
                    Run(_pFfnGu, s, pcs, 16, (uint)((c.FfnSize + 15) / 16), _gate);
                }
            }
            else
            {
                var s = Set(_pPreFfn, _h, W($"blk.{l}.ffn_norm.weight"),
                    W($"blk.{l}.ffn_gate.weight"), W($"blk.{l}.ffn_up.weight"), _gate);
                pcs[0] = (uint)hidden; pcs[1] = (uint)c.FfnSize; pcs[2] = BitConverter.SingleToUInt32Bits(c.Eps);
                pcs[3] = DType($"blk.{l}.ffn_gate.weight"); pcs[4] = DType($"blk.{l}.ffn_up.weight");
                Run(_pPreFfn, s, pcs, 20, (uint)((c.FfnSize + 15) / 16), _gate);
            }
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 9);
            // 6) h += ffn_down * gate
            GemvAdd(_wtype[$"blk.{l}.ffn_down.weight"] == GgmlTensorType.Q6_K ? _pGemvAdd6 : _pGemvAdd4,
                $"blk.{l}.ffn_down.weight", _gate, _h, c.FfnSize, hidden);
            if (l == layers - 1) Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 10);
        }

        Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 2);

        Rms(_h, _outNorm, _normed, 1, hidden);
        if (Environment.GetEnvironmentVariable("HYMT_VK_NOOUT") != "1"
            && Environment.GetEnvironmentVariable("HYMT_VK_OUTFIRST") != "1")
            Gemv("output.weight", _normed, _logitsBuf, hidden, _vocab);

        Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 3);

        // logits -> host staging in one burst copy (GPU writes to VRAM are fast;
        // scattered kernel writes to sysmem over PCIe are not).
        {
            Vk.VkMemoryBarrier mb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead,
            };
            Vk.vkCmdPipelineBarrier(_cmd, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0,
                1, &mb, 0, null, 0, null);
            Vk.VkBufferCopy r = new() { Size = (ulong)(_vocab * 4) };
            Vk.vkCmdCopyBuffer(_cmd, _logitsBuf.Buffer, _logitsStage.Buffer, 1, &r);
        }
        Vk.vkCmdWriteTimestamp(_cmd, VkConst.PipelineStageBottomOfPipe, _qpool, 4);

        Vk.Check(Vk.vkEndCommandBuffer(_cmd), "vkEndCommandBuffer");
    }

    // ---- M3 prefill: fp16 weight copies + per-call-recorded graph ----

    private void InitPrefill()
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, ffn = c.FfnSize, dim = c.HeadDim;
        int qDim = c.NumHeads * dim, kDim = _kvStride;
        int qkvDim = qDim + 2 * kDim;

        VkBuffer F16(long elems) => _dev.NewStorageBuffer((ulong)(elems * 2), hostVisible: _dump);
        _wq16 = new VkBuffer[c.NumLayers];
        _wo16 = new VkBuffer[c.NumLayers];
        _wgu16 = new VkBuffer[c.NumLayers];
        _wd16 = new VkBuffer[c.NumLayers];
        for (int l = 0; l < c.NumLayers; l++)
        {
            _wq16[l] = F16((long)qkvDim * hidden);
            _wo16[l] = F16((long)hidden * qDim);
            _wgu16[l] = F16(2L * ffn * hidden);
            _wd16[l] = F16((long)hidden * ffn);
        }
        _embd16 = F16((long)_vocab * hidden);

        _pfTok = _dev.NewStorageBuffer(PfCap * 4, hostVisible: true);
        _pfTokPtr = (int*)_pfTok.Map();
        VkBuffer PfF(long elems) => _dev.NewStorageBuffer((ulong)(elems * 4), hostVisible: _dump);
        VkBuffer PfA(long elems) => _dev.NewStorageBuffer((ulong)(elems * (_gVar ? 2 : 4)), hostVisible: _dump);
        _pfX = PfF((long)PfCap * hidden);
        _pfXs = PfA((long)PfCap * hidden);
        _pfXs2 = PfA((long)PfCap * hidden);
        _pfQkv = PfF((long)PfCap * qkvDim);
        _pfAo = PfA((long)PfCap * qDim);
        _pfGu = PfF((long)PfCap * 2 * ffn);
        _pfH = PfA((long)PfCap * ffn);
        if (_pfFastAttn)
        {
            _pfQ16 = _dev.NewStorageBuffer((ulong)(PfCap * c.NumHeads * dim * 2), hostVisible: _dump);
            _pfS32 = _dev.NewStorageBuffer((ulong)(c.NumHeads * PfCap * _kvCap * 4), hostVisible: _dump);
            _pfP16 = _dev.NewStorageBuffer((ulong)(c.NumHeads * PfCap * _kvCap * 2), hostVisible: _dump);
        }
        _cmdPf = _dev.NewCommandBuffer();

        // One-shot dequant submission: every GEMM weight + embed -> fp16 copies.
        IntPtr cmd = _dev.NewCommandBuffer();
        IntPtr fence = _dev.NewFence();
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(cmd, &begin), "vkBeginCommandBuffer");
        uint* pc = stackalloc uint[3];
        void Deq(string name, VkBuffer dst, long outOff)
        {
            GgmlTensorType t = _wtype[name];
            VkPipeline p = t == GgmlTensorType.Q4_K ? _pPfDeq4
                : t == GgmlTensorType.Q6_K ? _pPfDeq6
                : throw new NotSupportedException($"prefill dequant {name}: {t} unsupported on Vulkan");
            var s = Set(p, W(name), dst);
            long nblk = _welems[name] / 256;
            uint gx = (uint)Math.Min(nblk, 65535);
            pc[0] = (uint)outOff; pc[1] = gx; pc[2] = (uint)nblk;
            CmdRun(cmd, p, s, pc, 12, gx, (uint)((nblk + gx - 1) / gx), dst);
        }
        for (int l = 0; l < c.NumLayers; l++)
        {
            Deq($"blk.{l}.attn_q.weight", _wq16[l], 0);
            Deq($"blk.{l}.attn_k.weight", _wq16[l], (long)qDim * hidden);
            Deq($"blk.{l}.attn_v.weight", _wq16[l], (long)(qDim + kDim) * hidden);
            Deq($"blk.{l}.attn_output.weight", _wo16[l], 0);
            Deq($"blk.{l}.ffn_gate.weight", _wgu16[l], 0);
            Deq($"blk.{l}.ffn_up.weight", _wgu16[l], (long)ffn * hidden);
            Deq($"blk.{l}.ffn_down.weight", _wd16[l], 0);
        }
        Deq("token_embd.weight", _embd16, 0);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
    }

    private void RecordPrefill(int seq, int pos)
    {
        ModelConfig c = _cfg;
        int hidden = c.HiddenSize, heads = c.NumHeads, kvHeads = c.NumKvHeads, dim = c.HeadDim;
        int qDim = heads * dim, kDim = _kvStride, qkvDim = qDim + 2 * kDim, ffn = c.FfnSize;
        float scale = 1f / MathF.Sqrt(dim);
        uint M = (uint)seq;

        Vk.Check(Vk.vkResetCommandBuffer(_cmdPf, 0), "vkResetCommandBuffer");
        var begin = new Vk.VkCommandBufferBeginInfo { SType = VkConst.StCommandBufferBeginInfo };
        Vk.Check(Vk.vkBeginCommandBuffer(_cmdPf, &begin), "vkBeginCommandBuffer");

        // Non-push-descriptor path: sets are allocated once (first call) and
        // rebound every record — op order is seq-independent so indices align.
        _pfSetIdx = 0;
        (IntPtr set, VkBuffer[] bufs) PfSet(VkPipeline p, params VkBuffer[] bufs)
        {
            IntPtr s = IntPtr.Zero;
            if (!_dev.PushDescriptors || _noPush)
            {
                if (_pfSetIdx == _pfSetCache.Count)
                {
                    s = _dev.NewDescriptorSet(p.SetLayout);
                    for (uint i = 0; i < bufs.Length; i++) _dev.BindBuffer(s, i, bufs[i]);
                    _pfSetCache.Add(s);
                }
                else s = _pfSetCache[_pfSetIdx];
                _pfSetIdx++;
            }
            return (s, bufs);
        }
        void Run(VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, uint gy, params VkBuffer[] written)
            => CmdRun(_cmdPf, p, d, pc, pcBytes, gx, gy, written);

        uint* pc = stackalloc uint[10];
        uint epsB = BitConverter.SingleToUInt32Bits(c.Eps);
        int pfLayers = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_PFLAYERS"), out int pfl)
            ? Math.Min(pfl, c.NumLayers) : c.NumLayers;

        {   // x[m,:] = embd16[tok[m],:]
            var s = PfSet(_pPfEmbed, _embd16, _pfTok, _pfX);
            pc[0] = (uint)hidden;
            Run(_pPfEmbed, s, pc, 4, M, 1, _pfX);
        }

        if (_timing) { Vk.vkCmdResetQueryPool(_cmdPf, _qpool, 0, 16); Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageTopOfPipe, _qpool, 12); }
        for (int l = 0; l < pfLayers; l++)
        {
            if (!_noSmall) {   // xs = rmsnorm(x) * attn_norm
                var s = PfSet(_pPfRmsX, _pfX, W($"blk.{l}.attn_norm.weight"), _pfXs);
                pc[0] = (uint)hidden; pc[1] = 0; pc[2] = 0; pc[3] = epsB;
                Run(_pPfRmsX, s, pc, 16, M, 1, _pfXs);
            }
            {   // qkv = xs @ (wq|wk|wv)^T
                var s = PfSet(_pPfGemm, _pfXs, _wq16[l], _pfX, _pfQkv);
                pc[0] = M; pc[1] = (uint)qkvDim; pc[2] = (uint)hidden; pc[3] = 0;
                Run(_pPfGemm, s, pc, 16, (uint)((M + _pfGemmTM - 1) / _pfGemmTM), (uint)((qkvDim + _pfGemmTN - 1) / _pfGemmTN), _pfQkv);
            }
            if (!_noSmall) {   // rope+rmsnorm k rows -> fp16 K; v rows -> fp16 V
                var s = PfSet(_pPfKvPrep, _pfQkv, _kvK[l], _kvV[l], W($"blk.{l}.attn_k_norm.weight"), _prm);
                pc[0] = (uint)dim; pc[1] = (uint)c.RopeDim; pc[2] = (uint)_kvStride; pc[3] = (uint)qkvDim;
                pc[4] = (uint)qDim; pc[5] = (uint)(qDim + kDim);
                pc[6] = BitConverter.SingleToUInt32Bits(c.RopeBase); pc[7] = epsB;
                Run(_pPfKvPrep, s, pc, 32, (uint)kvHeads, M, _kvK[l], _kvV[l]);
            }
            if (!(_noSmall || _noAttn))
            {
                if (_pfFastAttn)
                {
                    {   // q16[m][h*headDim+d] = rmsnorm(rope(q row)) in fp16
                        var s = PfSet(_pPfQprep, _pfQkv, W($"blk.{l}.attn_q_norm.weight"), _prm, _pfQ16);
                        pc[0] = (uint)dim; pc[1] = (uint)c.RopeDim; pc[2] = (uint)qkvDim; pc[3] = (uint)qDim;
                        pc[4] = BitConverter.SingleToUInt32Bits(c.RopeBase); pc[5] = epsB;
                        Run(_pPfQprep, s, pc, 24, (uint)heads, M, _pfQ16);
                    }
                    {   // S = q16 . K^T per (head, m-tile)
                        var s = PfSet(_pPfQk, _pfQ16, _kvK[l], _pfS32);
                        pc[0] = (uint)qDim; pc[1] = (uint)_kvStride; pc[2] = (uint)_kvCap;
                        pc[3] = (uint)(PfCap * _kvCap); pc[4] = (uint)dim;
                        pc[5] = (uint)(heads / kvHeads); pc[6] = (uint)pos;
                        Run(_pPfQk, s, pc, 28, (uint)heads, (M + 127) / 128, _pfS32);
                    }
                    {   // P = causal softmax(S) row-wise
                        var s = PfSet(_pPfSoft, _pfS32, _pfP16, _prm);
                        pc[0] = (uint)_kvCap; pc[1] = (uint)PfCap; pc[2] = M;
                        pc[3] = BitConverter.SingleToUInt32Bits(scale);
                        Run(_pPfSoft, s, pc, 16, (uint)(heads * (int)M), 1, _pfP16);
                    }
                    {   // ao = P . V per (head, m-tile)
                        var s = PfSet(_pPfPv, _pfP16, _kvV[l], _pfAo);
                        pc[0] = (uint)_kvCap; pc[1] = (uint)(PfCap * _kvCap); pc[2] = (uint)_kvStride;
                        pc[3] = (uint)qDim; pc[4] = (uint)dim; pc[5] = (uint)(heads / kvHeads);
                        pc[6] = M; pc[7] = (uint)pos;
                        Run(_pPfPv, s, pc, 32, (uint)heads, (M + 127) / 128, _pfAo);
                    }
                }
                else
                {   // causal attention per (head, m) -> _pfAo
                    var s = PfSet(_pPfAttn, _pfQkv, _kvK[l], _kvV[l], W($"blk.{l}.attn_q_norm.weight"), _prm, _pfAo);
                    pc[0] = (uint)heads; pc[1] = (uint)kvHeads; pc[2] = (uint)dim; pc[3] = (uint)c.RopeDim;
                    pc[4] = (uint)_kvStride; pc[5] = (uint)qkvDim;
                    pc[6] = BitConverter.SingleToUInt32Bits(scale);
                    pc[7] = BitConverter.SingleToUInt32Bits(c.RopeBase); pc[8] = epsB;
                    Run(_pPfAttn, s, pc, 36, (uint)heads, M, _pfAo);
                }
            }
            {   // x += ao @ wo^T
                var s = PfSet(_pPfGemm, _pfAo, _wo16[l], _pfX, _pfX);
                pc[0] = M; pc[1] = (uint)hidden; pc[2] = (uint)qDim; pc[3] = 1;
                Run(_pPfGemm, s, pc, 16, (uint)((M + _pfGemmTM - 1) / _pfGemmTM), (uint)((hidden + _pfGemmTN - 1) / _pfGemmTN), _pfX);
            }
            if (!_noSmall) {   // xs = rmsnorm(x) * ffn_norm
                var s = PfSet(_pPfRmsX, _pfX, W($"blk.{l}.ffn_norm.weight"), _pfXs2);
                pc[0] = (uint)hidden; pc[1] = 0; pc[2] = 0; pc[3] = epsB;
                Run(_pPfRmsX, s, pc, 16, M, 1, _pfXs2);
            }
            {   // gu = xs @ (gate|up)^T
                var s = PfSet(_pPfGemm, _pfXs2, _wgu16[l], _pfX, _pfGu);
                pc[0] = M; pc[1] = (uint)(2 * ffn); pc[2] = (uint)hidden; pc[3] = 0;
                Run(_pPfGemm, s, pc, 16, (uint)((M + _pfGemmTM - 1) / _pfGemmTM), (uint)((2 * ffn + _pfGemmTN - 1) / _pfGemmTN), _pfGu);
            }
            if (!_noSmall) {   // h = silu(gate) * up
                var s = PfSet(_pPfSilu, _pfGu, _pfH);
                uint total = (uint)(seq * ffn);
                pc[0] = (uint)ffn; pc[1] = total;
                Run(_pPfSilu, s, pc, 8, (total + 255) / 256, 1, _pfH);
            }
            {   // x += h @ ffn_down^T
                var s = PfSet(_pPfGemm, _pfH, _wd16[l], _pfX, _pfX);
                pc[0] = M; pc[1] = (uint)hidden; pc[2] = (uint)ffn; pc[3] = 1;
                Run(_pPfGemm, s, pc, 16, (uint)((M + _pfGemmTM - 1) / _pfGemmTM), (uint)((hidden + _pfGemmTN - 1) / _pfGemmTN), _pfX);
            }
        }

        if (_timing) Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageBottomOfPipe, _qpool, 13);
        {   // normed = rmsnorm(x[last]) * out_norm, single row at index 0
            var s = PfSet(_pPfRms, _pfX, _outNorm, _normed);
            pc[0] = (uint)hidden; pc[1] = (uint)(seq - 1); pc[2] = 0; pc[3] = epsB;
            Run(_pPfRms, s, pc, 16, 1, 1, _normed);
        }
        {   // logits = normed @ output^T — the quant gemv (same kernel as decode)
            GgmlTensorType t = _wtype["output.weight"];
            VkPipeline p = t == GgmlTensorType.Q4_K ? _pGemv4 : _pGemv6;
            var s = t == GgmlTensorType.Q4_K
                ? PfSet(p, W("output.weight"), _normed, _logitsBuf, W("output.weight"))
                : PfSet(p, W("output.weight"), _normed, _logitsBuf);
            pc[0] = (uint)hidden; pc[1] = (uint)_vocab;
            Run(p, s, pc, 8, (uint)((_vocab + 15) / 16), 1, _logitsBuf);
        }
        if (_timing) Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageBottomOfPipe, _qpool, 14);
        {   // logits -> host staging in one burst copy
            Vk.VkMemoryBarrier mb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite, DstAccessMask = VkConst.AccessTransferRead,
            };
            Vk.vkCmdPipelineBarrier(_cmdPf, VkConst.PipelineStageComputeShader, VkConst.PipelineStageTransfer, 0,
                1, &mb, 0, null, 0, null);
            Vk.VkBufferCopy r = new() { Size = (ulong)(_vocab * 4) };
            Vk.vkCmdCopyBuffer(_cmdPf, _logitsBuf.Buffer, _logitsStage.Buffer, 1, &r);
        }
        Vk.Check(Vk.vkEndCommandBuffer(_cmdPf), "vkEndCommandBuffer");
    }

    public void UploadKv(int layer, int pos, int len, ushort* k, ushort* v)
    {
        if (pos + len > _kvCap)
            throw new NotSupportedException($"Vulkan KV capacity {_kvCap} exceeded at {pos + len} (flat cap)");
        // CPU KV is bf16; device KV is fp16 — convert on host.
        ushort* ks = k + pos * _kvStride;
        ushort* vs = v + pos * _kvStride;
        if (_kvF32)
        {
            float* kdf = (float*)_kvKHost[layer] + pos * _kvStride;
            float* vdf = (float*)_kvVHost[layer] + pos * _kvStride;
            for (long i = 0; i < (long)len * _kvStride; i++)
            {
                kdf[i] = BitConverter.UInt32BitsToSingle(((uint)ks[i]) << 16);
                vdf[i] = BitConverter.UInt32BitsToSingle(((uint)vs[i]) << 16);
            }
        }
        else
        {
            ushort* kd = _kvKHost[layer] + pos * _kvStride;
            ushort* vd = _kvVHost[layer] + pos * _kvStride;
            for (long i = 0; i < (long)len * _kvStride; i++)
            {
                kd[i] = BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(((uint)ks[i]) << 16));
                vd[i] = BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(((uint)vs[i]) << 16));
            }
        }
        int kvb = _kvF32 ? 4 : 2;
        _kvK[layer].Flush((ulong)(pos * _kvStride * kvb), (ulong)(len * _kvStride * kvb));
        _kvV[layer].Flush((ulong)(pos * _kvStride * kvb), (ulong)(len * _kvStride * kvb));
    }

    private void DumpBuffers()
    {
        static void Dump(StringBuilder sb, string name, VkBuffer b, int n)
        {
            b.Invalidate(0, (ulong)n * 4);
            float* p = (float*)b.Map();
            sb.Append(name).Append(':');
            for (int i = 0; i < n; i++) sb.Append(' ').Append(p[i].ToString("R"));
            b.Unmap();
            sb.Append('\n');
        }
        var sb = new StringBuilder();
        Dump(sb, "h", _h, 8);
        Dump(sb, "n1", _n1, 8);
        Dump(sb, "q", _q, 64);
        Dump(sb, "k", _k, 8);
        Dump(sb, "v", _v, 64);
        Dump(sb, "ao", _ao, 16);
        Dump(sb, "attnOut", _attnOut, 8);
        Dump(sb, "n2", _n2, 8);
        Dump(sb, "gate", _gate, 8);
        Dump(sb, "up", _up, 8);
        Dump(sb, "down", _down, 8);
        Dump(sb, "normed", _normed, 8);
        File.AppendAllText("vk_dump.txt", sb.ToString());

        static void Bin(string name, VkBuffer b, int n)
        {
            b.Invalidate(0, (ulong)n * 4);
            float* p = (float*)b.Map();
            using var fs = File.Create($"vk_{name}.bin");
            using var bw = new BinaryWriter(fs);
            for (int i = 0; i < n; i++) bw.Write(p[i]);
            b.Unmap();
        }
        Bin("n1", _n1, 2048);
        Bin("v", _v, 512);
        Bin("q", _q, 2048);
        Bin("k", _k, 512);
        Bin("ao", _ao, 2048);
        Bin("h", _h, 2048);
        Bin("gate", _gate, 6144);
    }

    private void DumpPrefill(int seq)
    {
        static void BinRow(string name, VkBuffer b, long rowStart, int n)
        {
            b.Invalidate((ulong)(rowStart * 4), (ulong)(n * 4));
            float* p = (float*)b.Map() + rowStart;
            using var fs = File.Create($"vk_{name}.bin");
            using var bw = new BinaryWriter(fs);
            for (int i = 0; i < n; i++) bw.Write(p[i]);
            b.Unmap();
        }
        int hidden = _cfg.HiddenSize, qDim = _cfg.NumHeads * _cfg.HeadDim, ffn = _cfg.FfnSize;
        int qkvDim = qDim + 2 * _kvStride;
        long last = seq - 1;
        BinRow("pfx", _pfX, last * hidden, hidden);
        BinRow("pfxs", _pfXs, last * hidden, hidden);
        BinRow("pfqkv", _pfQkv, last * qkvDim, qkvDim);
        BinRow("pfao", _pfAo, last * qDim, qDim);
        BinRow("pfgu", _pfGu, last * 2 * ffn, 2 * ffn);
        BinRow("pfh", _pfH, last * ffn, ffn);
        // whole-tensor dumps for stage isolation (first rows too)
        BinRow("pfx0", _pfX, 0, hidden);
        BinRow("wq", _wq16[0], 0, 512);
        BinRow("pfqkv0", _pfQkv, 0, qkvDim);
        BinRow("pfn", _normed, 0, hidden);
        if (_pfFastAttn)
        {
            static void Bin16(string name, VkBuffer b, long n)
            {
                b.Invalidate(0, (ulong)(n * 2));
                ushort* p = (ushort*)b.Map();
                using var fs = File.Create($"vk_{name}.bin");
                using var bw = new BinaryWriter(fs);
                for (long i = 0; i < n; i++) bw.Write((float)BitConverter.UInt16BitsToHalf(p[i]));
                b.Unmap();
            }
            static void Bin32(string name, VkBuffer b, long n)
            {
                b.Invalidate(0, (ulong)(n * 4));
                float* p = (float*)b.Map();
                using var fs = File.Create($"vk_{name}.bin");
                using var bw = new BinaryWriter(fs);
                for (long i = 0; i < n; i++) bw.Write(p[i]);
                b.Unmap();
            }
            int heads = _cfg.NumHeads, dim = _cfg.HeadDim;
            Bin16("pfq16", _pfQ16, (long)seq * heads * dim);
            Bin16("pfp16", _pfP16, (long)heads * seq * _kvCap);
            Bin32("pfs32", _pfS32, (long)heads * seq * _kvCap);
            Bin16("pfa16", _pfAo, (long)seq * qDim);
        }
    }

    private float[] PrefillGpu(ReadOnlySpan<int> tokens, int pos)
    {
        int seq = tokens.Length;
        if (seq > PfCap)
            throw new NotSupportedException($"Vulkan prefill chunk {seq} exceeds cap {PfCap}");
        if (pos + seq > _kvCap)
            throw new NotSupportedException($"Vulkan KV capacity {_kvCap} exceeded at {pos + seq}");

        for (int i = 0; i < seq; i++) _pfTokPtr[i] = tokens[i];
        _pfTok.Flush(0, (ulong)(seq * 4));
        _prmPtr[0] = 0; _prmPtr[1] = pos; _prmPtr[2] = pos + seq; _prmPtr[3] = _seq++;
        _prm.Flush(0, 16);

        _sw.Restart();
        RecordPrefill(seq, pos);
        _dev.Submit(_cmdPf, _fence);
        _dev.WaitFence(_fence);
        IntPtr f = _fence;
        Vk.vkResetFences(_dev.Device, 1, &f);
        if (_timing)
        {
            Console.WriteLine($"[vk] prefill seq={seq} pos={pos} {_sw.Elapsed.TotalMilliseconds:F1}ms");
            ulong* ts = stackalloc ulong[16];
            VkResult qr = Vk.vkGetQueryPoolResults(_dev.Device, _qpool, 0, 16, 128, ts, 8, 1u | 2u);
            double per = _dev.TimestampPeriodNs * 1e-6;
            Console.WriteLine($"[vk] pf-seg qr={qr} layers={(ts[13] - ts[12]) * per:F2}ms outgemv={(ts[14] - ts[13]) * per:F2}ms");
        }
        if (_dump) DumpPrefill(seq);

        if (Environment.GetEnvironmentVariable("HYMT_VK_NOREAD") != "1")
        {
            _logitsStage.Invalidate(0, _logitsStage.Size);
            Marshal.Copy((IntPtr)_logitsStage.Map(), _logitsHost, 0, _vocab);
            _logitsStage.Unmap();
        }
        return _logitsHost;
    }

    public float[] ForwardStep(ReadOnlySpan<int> tokens, int pos)
    {
        if (tokens.Length > 1) return PrefillGpu(tokens, pos);
        int kvLen = pos + 1;
        if (kvLen > _kvCap)
            throw new NotSupportedException($"Vulkan KV capacity {_kvCap} exceeded at {kvLen} (flat cap)");

        _prmPtr[0] = tokens[0]; _prmPtr[1] = pos; _prmPtr[2] = kvLen; _prmPtr[3] = _seq;
        _prm.Flush(0, 16);
        if (_dump) File.AppendAllText("vk_dump.txt", $"step {_step} token={tokens[0]} pos={pos}\n");

        _sw.Restart();
        int reps = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_REPS"), out int nr) ? nr : 1;
        for (int i = 0; i < reps; i++)
        {
            // each submission consumes its own epoch of the monotonic flag counters
            _prmPtr[3] = _seq++;
            _prm.Flush(0, 16);
            _dev.Submit(_cmd, _fence);
            if (i < reps - 1)
            {
                _dev.WaitFence(_fence);
                IntPtr f0 = _fence;
                Vk.vkResetFences(_dev.Device, 1, &f0);
            }
        }
        _dev.WaitFence(_fence);
        IntPtr f = _fence;
        Vk.vkResetFences(_dev.Device, 1, &f);
        if (_dump) { DumpBuffers(); _step++; }
        if (_timing)
        {
            _gpuMs += _sw.Elapsed.TotalMilliseconds; _n++;
            ulong* ts = stackalloc ulong[16];
            double[] seg = new double[4];
            VkResult qr = Vk.vkGetQueryPoolResults(_dev.Device, _qpool, 0, 12, 128, ts, 8, 1u | 2u);
            if (_n == 1) Console.WriteLine($"[vk] qr={qr} period={_dev.TimestampPeriodNs} ts0={ts[0]} ts1={ts[1]} ts2={ts[2]} ts3={ts[3]} ts4={ts[4]}");
            if (qr == VkResult.Success)
            {
                double per = _dev.TimestampPeriodNs * 1e-6;
                for (int i = 0; i < 4; i++) seg[i] = (ts[i + 1] - ts[i]) * per;
                _segMs[0] += seg[0]; _segMs[1] += seg[1]; _segMs[2] += seg[2]; _segMs[3] += seg[3];
                // per-op in last layer: slot 11 = layer start, 5..10 = end of each op
                uint[] map = { 11, 5, 6, 7, 8, 9, 10 };
                for (int i = 0; i < 6; i++) _opMs[i] += (ts[map[i + 1]] - ts[map[i]]) * per;
            }
            if (_n % 64 == 0)
            {
                Console.WriteLine($"[vk] submit+wait={_gpuMs / _n:F2}ms/tok  embed={_segMs[0] / _n:F3} layers={_segMs[1] / _n:F2} outgemv={_segMs[2] / _n:F2} copy={_segMs[3] / _n:F3}");
                Console.WriteLine($"[vk] per-op last-layer: prekv={_opMs[0] / _n * 1000:F0}us kvprep={_opMs[1] / _n * 1000:F0}us attn={_opMs[2] / _n * 1000:F0}us outadd={_opMs[3] / _n * 1000:F0}us preffn={_opMs[4] / _n * 1000:F0}us downadd={_opMs[5] / _n * 1000:F0}us");
                _gpuMs = 0; _n = 0; _segMs = new double[4]; _opMs = new double[6];
            }
        }

        if (Environment.GetEnvironmentVariable("HYMT_VK_NOREAD") != "1")
        {
            _logitsStage.Invalidate(0, _logitsStage.Size);
            Marshal.Copy((IntPtr)_logitsStage.Map(), _logitsHost, 0, _vocab);
            _logitsStage.Unmap();
        }
        return _logitsHost;
    }

    private static byte[] LoadSpv(string name)
    {
        var asm = typeof(VulkanBackend).Assembly;
        string res = $"Sdcb.HyMT2Sharp.Backends.Vulkan.Shaders.{name}.spv";
        using Stream s = asm.GetManifestResourceStream(res)
            ?? throw new FileNotFoundException($"embedded shader {res} — run tools/build-shaders.bat");
        byte[] b = new byte[s.Length];
        s.ReadExactly(b);
        return b;
    }

    public void Dispose() => _dev.Dispose();
}
