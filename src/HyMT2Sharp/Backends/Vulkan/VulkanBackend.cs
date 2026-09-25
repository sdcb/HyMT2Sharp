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
    private VkPipeline _pGemv4 = null!, _pGemv6 = null!, _pGemv8 = null!, _pGemvS = null!, _pGemv2 = null!,
        _pEmbed4 = null!, _pEmbed6 = null!, _pEmbed8 = null!,
        _pRms = null!,
        _pPreKv = null!, _pGemvAdd4 = null!, _pGemvAdd6 = null!, _pGemvAdd8 = null!, _pGemvAddS = null!, _pGemvAdd2 = null!, _pPreFfn = null!,
        _pAttn2 = null!, _pFfnGu = null!;
    private VkPipeline _pPfDeq4 = null!, _pPfDeq6 = null!, _pPfDeq8 = null!, _pPfDeqS = null!, _pPfDeq2 = null!,
        _pPfEmbed = null!, _pPfRms = null!,
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
    private long _pfRecKey = -1;            // (pos,seq) the prefill CB was last recorded for
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
    private readonly bool _pfProf = Environment.GetEnvironmentVariable("HYMT_VK_PFPROF") == "1";
    private const uint PfProfCap = 1024;
    private IntPtr _pfProfPool;
    private readonly List<string> _pfProfNames = new();

    public string Name => "vulkan";
    // pf kernels assume the fp16 KV layout (device KV is always fp16).
    public bool SupportsPrefill => !_noPf;

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
        {
            VkPipeline p = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv(spv)), bindings, pushBytes);
            p.Name = spv;
            return p;
        }

        _pGemv4 = Mk("q4k_gemv3", bindings: 3, pushBytes: 8);
        _pGemv6 = Mk("q6k_gemv2", bindings: 3, pushBytes: 8);
        _pGemv8 = Mk("q8_gemv", bindings: 3, pushBytes: 8);
        _pGemvS = Mk("stq_gemv", bindings: 3, pushBytes: 8);
        _pGemv2 = Mk("q2_gemv", bindings: 3, pushBytes: 8);
        _pEmbed4 = Mk("dec_embed_q4k", bindings: 3, pushBytes: 4);
        _pEmbed6 = Mk("dec_embed_q6k", bindings: 3, pushBytes: 4);
        _pEmbed8 = Mk("dec_embed_q8", bindings: 3, pushBytes: 4);
        _pRms = Mk("dec_rmsnorm", bindings: 3, pushBytes: 8);
        _pAttn2 = Mk("dec_attn2", bindings: 11, pushBytes: 32);
        _pFfnGu = Mk("dec_ffngu", bindings: 4, pushBytes: 16);
        _pPreKv = Mk("dec_prekv", bindings: 8, pushBytes: 28);
        _pGemvAdd4 = Mk("dec_gemvadd_q4k", bindings: 3, pushBytes: 8);
        _pGemvAdd6 = Mk("dec_gemvadd_q6k", bindings: 3, pushBytes: 8);
        _pGemvAdd8 = Mk("dec_gemvadd_q8", bindings: 3, pushBytes: 8);
        _pGemvAddS = Mk("dec_gemvadd_stq", bindings: 3, pushBytes: 8);
        _pGemvAdd2 = Mk("dec_gemvadd_q2", bindings: 3, pushBytes: 8);
        _pPreFfn = Mk("dec_preffn", bindings: 5, pushBytes: 20);
        _pPfDeq4 = Mk("pf_deq_q4k", bindings: 2, pushBytes: 12);
        _pPfDeq6 = Mk("pf_deq_q6k", bindings: 2, pushBytes: 12);
        _pPfDeq8 = Mk("pf_deq_q8", bindings: 2, pushBytes: 12);
        _pPfDeqS = Mk("pf_deq_stq", bindings: 2, pushBytes: 12);
        _pPfDeq2 = Mk("pf_deq_q2", bindings: 2, pushBytes: 12);
        _pPfEmbed = Mk("pf_embed", bindings: 3, pushBytes: 4);
        _pPfRms = Mk("pf_rms", bindings: 3, pushBytes: 16);
        // Cooperative-matrix subgroup size the device can guarantee: sg16 (Intel
        // Xe) or sg32 (NVIDIA/AMD); requiredSubgroupSize must lie in [min, max].
        int cmSg = _dev.CoopMatrix && _dev.SubgroupSizeControl
            ? (_dev.SubgroupMin <= 16 && _dev.SubgroupMax >= 16 ? 16
            : _dev.SubgroupMin <= 32 && _dev.SubgroupMax >= 32 ? 32 : 0)
            : 0;
        bool useCm = cmSg != 0 && Environment.GetEnvironmentVariable("HYMT_VK_NOCM") != "1";
        if (Environment.GetEnvironmentVariable("HYMT_VK_DEBUG") == "1")
            Console.Error.WriteLine($"vk-dbg cmSg={cmSg} useCm={useCm} coop={_dev.CoopMatrix} sgc={_dev.SubgroupSizeControl} sgMin={_dev.SubgroupMin} sgMax={_dev.SubgroupMax}");
        string cmVar = cmSg == 32
            ? (Environment.GetEnvironmentVariable("HYMT_VK_GEMMV32") ?? "sg32")
            : (Environment.GetEnvironmentVariable("HYMT_VK_GEMMV") ?? "g28");
        _gVar = useCm && (cmSg == 32 || cmVar.StartsWith("g"));
        _pfGemmTM = useCm
            ? (int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_GEMMTM"), out int tm) ? tm : (cmSg == 32 ? 64 : 512))
            : 128;
        _pfGemmTN = useCm
            ? (int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_GEMMTN"), out int tn) ? tn : (cmSg == 32 ? 64 : 64))
            : 128;
        try
        {
            _pPfGemm = useCm
                ? _dev.NewPipeline(_dev.NewShaderModule(LoadSpv(
                    cmSg == 32 ? "pf_gemm_cm_" + cmVar
                    : Environment.GetEnvironmentVariable("HYMT_VK_F16ACC") == "1" ? "pf_gemm_cm_f16"
                    : "pf_gemm_cm_" + cmVar)),
                    4, 16, requiredSubgroupSize: (uint)cmSg)
                : Mk("pf_gemm", bindings: 4, pushBytes: 16);
            _pPfGemm.Name = "gemm";
            if (useCm && cmSg == 32 && Environment.GetEnvironmentVariable("HYMT_VK_NOT32") != "1")
            {
                // per-GEMM tile variants (qkv, wo, gu, down); HYMT_VK_T32[_QKV|_WO|_GU|_DOWN] override
                string def = Environment.GetEnvironmentVariable("HYMT_VK_T32") ?? "64x64_32x32";
                string[] roles = { "QKV", "WO", "GU", "DOWN" };
                var cache = new Dictionary<string, GemmT>();
                _gemmT = new GemmT[4];
                for (int i = 0; i < 4; i++)
                {
                    string v = Environment.GetEnvironmentVariable("HYMT_VK_T32_" + roles[i]) ?? def;
                    if (!cache.TryGetValue(v, out GemmT? g)) cache[v] = g = LoadGemmT(v);
                    _gemmT[i] = g;
                }
                if (cfg.HiddenSize <= 2048 && cfg.HiddenSize % 4 == 0)
                {
                    _pPfAddRms = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_addrms16")), 4, 16, requiredSubgroupSize: 32);
                    _pPfAddRms.Name = "pf_addrms16";
                    _splitKWo = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_SPLITK_WO"), out int sw) ? sw : 1;
                    _splitKDown = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_SPLITK_DOWN"), out int sd) ? sd : 2;
                }
                if (cfg.HeadDim == 128 && cfg.RopeDim == 128 && cfg.NumHeads == 4 * cfg.NumKvHeads && Environment.GetEnvironmentVariable("HYMT_VK_NOFA") != "1")
                {
                    _pPfFa = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_fa32")), 6, 36, requiredSubgroupSize: 32);
                    _pPfFa.Name = "pf_fa32";
                }
            }
            // sg32 (NVIDIA/AMD): glslc-compiled fp16+subgroup shaders produce no
            // output there — the *_sg32 variants are the same sources rebuilt
            // with glslang. Intel sg16 keeps the committed glslc SPIR-V.
            _pPfRmsX = _gVar ? Mk(cmSg == 32 ? "pf_rms16_sg32" : "pf_rms16", bindings: 3, pushBytes: 16) : _pPfRms;
            _pPfKvPrep = Mk(_gVar && cmSg == 32 ? "pf_kvprep_sg32" : "pf_kvprep", bindings: 5, pushBytes: 32);
            _pPfAttn = Mk(_gVar ? (cmSg == 32 ? "pf_attn16_sg32" : "pf_attn16") : "pf_attn", bindings: 6, pushBytes: 36);
            _pPfSilu = Mk(_gVar ? (cmSg == 32 ? "pf_silumul16_sg32" : "pf_silumul16") : "pf_silumul", bindings: 2, pushBytes: 8);
            // coopmat attention (qk/pv) has no sg32 variant yet — scalar attn16 path.
            _pfFastAttn = _gVar && useCm && cmSg != 32 && Environment.GetEnvironmentVariable("HYMT_VK_NOFASTATTN") != "1";
            if (_pfFastAttn)
            {
                _pPfQprep = Mk("pf_qprep", bindings: 4, pushBytes: 24);
                _pPfQk = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_qk")), 3, 28, requiredSubgroupSize: (uint)cmSg);
                _pPfSoft = Mk("pf_soft", bindings: 3, pushBytes: 16);
                _pPfPv = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_pv")), 3, 32, requiredSubgroupSize: (uint)cmSg);
            }
        }
        catch (Exception e) when (useCm)
        {
            // Unsupported requiredSubgroupSize, missing SPV variant, or rejected
            // coopmat shapes — fall back to the scalar GEMM path.
            Console.Error.WriteLine($"[vk] coopmat pipeline failed ({e.Message}) — scalar pf_gemm fallback");
            useCm = false; _gVar = false; _pfFastAttn = false; _gemmT = null; _pPfFa = null; _pPfAddRms = null;
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
                GgmlTensorType.Q8_0 => info.NumElements / 32 * (ulong)34,
                GgmlTensorType.STQ1_0 => info.NumElements / 256 * (ulong)42,
                GgmlTensorType.Q2_0C => info.NumElements / 512 * (ulong)130,
                _ => throw new NotSupportedException(
                    $"Vulkan backend supports F32/Q4_K/Q6_K/Q8_0/STQ1_0/Q2_0C only; {name} is {info.Type} — use --backend cpu"),
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
        // Rows rounded up to 32 and zero-filled: pf_fa32 reads whole 32-row KV
        // tiles straight from the cache, so the tail past kvLen must be finite.
        ulong kvBytes = (ulong)((_kvCap + 31) / 32 * 32 * _kvStride * 2);
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            _kvK[l] = _dev.NewStorageBuffer(kvBytes, hostVisible: true);
            _kvV[l] = _dev.NewStorageBuffer(kvBytes, hostVisible: true);
            _kvKHost[l] = (ushort*)_kvK[l].Map();
            _kvVHost[l] = (ushort*)_kvV[l].Map();
            NativeMemory.Clear(_kvKHost[l], (nuint)kvBytes);
            NativeMemory.Clear(_kvVHost[l], (nuint)kvBytes);
            _kvK[l].Flush(0, kvBytes);
            _kvV[l].Flush(0, kvBytes);
        }
        if (Environment.GetEnvironmentVariable("HYMT_VK_VERBOSE") == "1")
            Console.Error.WriteLine($"[vk] kv mem flags=0x{_kvK[0].Flags:X} (last layer 0x{_kvK[cfg.NumLayers - 1].Flags:X}) weights flags=0x{_embd.Flags:X}");

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
        if (_pfProf)
        {
            Vk.VkQueryPoolCreateInfo ppci = new() { SType = 11, QueryType = 2, QueryCount = PfProfCap };
            Vk.Check(Vk.vkCreateQueryPool(_dev.Device, &ppci, null, out _pfProfPool), "vkCreateQueryPool");
        }
        RecordDecodeGraph();
        if (SupportsPrefill) InitPrefill();
    }

    // sg32 tensor-core GEMM variants: pf_gemm_t32_{BM}x{BN}_{WM}x{WN}[_k{BK}].spv
    private sealed record GemmT(VkPipeline P, int BM, int BN, int BK, int WN);
    private GemmT[]? _gemmT;                // indexed by GEMM role: 0 qkv, 1 wo, 2 gu, 3 down
    private bool _swiglu;                   // gu GEMM writes h = silu(g)*u directly (Wgu rows interleaved in 16-row gate/up groups)
    private VkPipeline? _pPfFa;             // sg32 tensor-core causal attention (16 rows x 4 GQA heads / WG)
    private VkPipeline? _pPfAddRms;         // x += split-K partials; xs = rmsnorm(x)
    private int _splitKWo = 1, _splitKDown = 1;
    private VkBuffer _pfPart = null!;       // split-K partial slices [(splitK-1)][PfCap][hidden] fp32
    private long _pfPartElems;

    private GemmT LoadGemmT(string v)
    {
        string[] parts = v.Split('_');
        string[] t = parts[0].Split('x');
        VkPipeline p = _dev.NewPipeline(_dev.NewShaderModule(LoadSpv("pf_gemm_t32_" + v)), 4, 16, requiredSubgroupSize: 32);
        p.Name = "gemm_t" + v;
        int k = parts.Length == 3 ? int.Parse(parts[2][1..]) : 32;
        return new GemmT(p, int.Parse(t[0]), int.Parse(t[1]), k, int.Parse(parts[1].Split('x')[1]));
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
        => CmdRun(cmd, p, d, pc, pcBytes, gx, gy, 1, written);

    private void CmdRun(IntPtr cmd, VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, uint gy, uint gz, params VkBuffer[] written)
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
        Vk.vkCmdDispatch(cmd, gx, gy, gz);
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
                : t == GgmlTensorType.Q8_0 ? _pGemv8
                : t == GgmlTensorType.STQ1_0 ? _pGemvS
                : t == GgmlTensorType.Q2_0C ? _pGemv2
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
            VkPipeline p = _wtype["token_embd.weight"] switch
            {
                GgmlTensorType.Q6_K => _pEmbed6,
                GgmlTensorType.Q8_0 => _pEmbed8,
                GgmlTensorType.Q4_K => _pEmbed4,
                var t => throw new NotSupportedException($"vulkan embed: token_embd is {t} — only Q4_K/Q6_K/Q8_0 supported"),
            };
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
            GgmlTensorType.Q8_0 => 2u,
            GgmlTensorType.STQ1_0 => 3u,
            GgmlTensorType.Q2_0C => 4u,
            var t => throw new NotSupportedException($"vulkan decode: {name} is {t} — only Q4_K/Q6_K/Q8_0/STQ1_0/Q2_0C supported"),
        };

        VkPipeline GaPipe(string name) => _wtype[name] switch
        {
            GgmlTensorType.Q6_K => _pGemvAdd6,
            GgmlTensorType.Q8_0 => _pGemvAdd8,
            GgmlTensorType.STQ1_0 => _pGemvAddS,
            GgmlTensorType.Q2_0C => _pGemvAdd2,
            GgmlTensorType.Q4_K => _pGemvAdd4,
            var t => throw new NotSupportedException($"vulkan gemvadd: {name} is {t} — only Q4_K/Q6_K/Q8_0/STQ1_0/Q2_0C supported"),
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
            GemvAdd(GaPipe($"blk.{l}.attn_output.weight"),
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
            GemvAdd(GaPipe($"blk.{l}.ffn_down.weight"),
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
        _swiglu = _gemmT?[2] is { } t && t.WN % 32 == 0 && 2 * ffn % t.BN == 0 && ffn % 16 == 0 && hidden % t.BK == 0
            && Environment.GetEnvironmentVariable("HYMT_VK_NOSWIGLU") != "1";
        _pfGu = _swiglu ? _pfQkv : PfF((long)PfCap * 2 * ffn);
        _pfH = PfA((long)PfCap * ffn);
        _pfPartElems = (long)(Math.Max(_splitKWo, _splitKDown) - 1) * PfCap * hidden;
        _pfPart = _pfPartElems > 0 ? PfF(_pfPartElems) : _pfX;
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
                : t == GgmlTensorType.Q8_0 ? _pPfDeq8
                : t == GgmlTensorType.STQ1_0 ? _pPfDeqS
                : t == GgmlTensorType.Q2_0C ? _pPfDeq2
                : throw new NotSupportedException($"prefill dequant {name}: {t} unsupported on Vulkan");
            var s = Set(p, W(name), dst);
            long nblk = _welems[name] / 256;
            uint gx = (uint)Math.Min(nblk, 65535);
            pc[0] = (uint)outOff; pc[1] = gx; pc[2] = (uint)nblk;
            CmdRun(cmd, p, s, pc, 12, gx, (uint)((nblk + gx - 1) / gx), dst);
        }
        // SwiGLU-fused gu GEMM wants gate/up rows interleaved in 16-row groups:
        // dequant into a scratch copy, then scatter with one multi-region copy.
        VkBuffer? guTmp = _swiglu ? F16(2L * ffn * hidden) : null;
        Vk.VkBufferCopy[] regs = new Vk.VkBufferCopy[_swiglu ? 2 * ffn / 16 : 0];
        for (int g = 0; g < regs.Length / 2; g++)
        {
            ulong grp = (ulong)(16 * hidden * 2);
            regs[2 * g] = new Vk.VkBufferCopy { SrcOffset = (ulong)g * grp, DstOffset = (ulong)(2 * g) * grp, Size = grp };
            regs[2 * g + 1] = new Vk.VkBufferCopy { SrcOffset = (ulong)(ffn / 16 + g) * grp, DstOffset = (ulong)(2 * g + 1) * grp, Size = grp };
        }
        void AllBarrier()
        {
            Vk.VkMemoryBarrier mb = new()
            {
                SType = VkConst.StMemoryBarrier,
                SrcAccessMask = VkConst.AccessShaderWrite | VkConst.AccessTransferWrite,
                DstAccessMask = VkConst.AccessShaderRead | VkConst.AccessShaderWrite | VkConst.AccessTransferRead | VkConst.AccessTransferWrite,
            };
            uint st = VkConst.PipelineStageComputeShader | VkConst.PipelineStageTransfer;
            Vk.vkCmdPipelineBarrier(cmd, st, st, 0, 1, &mb, 0, null, 0, null);
        }
        for (int l = 0; l < c.NumLayers; l++)
        {
            Deq($"blk.{l}.attn_q.weight", _wq16[l], 0);
            Deq($"blk.{l}.attn_k.weight", _wq16[l], (long)qDim * hidden);
            Deq($"blk.{l}.attn_v.weight", _wq16[l], (long)(qDim + kDim) * hidden);
            Deq($"blk.{l}.attn_output.weight", _wo16[l], 0);
            Deq($"blk.{l}.ffn_gate.weight", guTmp ?? _wgu16[l], 0);
            Deq($"blk.{l}.ffn_up.weight", guTmp ?? _wgu16[l], (long)ffn * hidden);
            if (guTmp != null)
            {
                AllBarrier();
                fixed (Vk.VkBufferCopy* r = regs)
                    Vk.vkCmdCopyBuffer(cmd, guTmp.Buffer, _wgu16[l].Buffer, (uint)regs.Length, r);
                AllBarrier();
            }
            Deq($"blk.{l}.ffn_down.weight", _wd16[l], 0);
        }
        Deq("token_embd.weight", _embd16, 0);
        Vk.Check(Vk.vkEndCommandBuffer(cmd), "vkEndCommandBuffer");
        _dev.Submit(cmd, fence);
        _dev.WaitFence(fence);
        if (guTmp != null)
        {
            Vk.vkDestroyBuffer(_dev.Device, guTmp.Buffer, null);
            Vk.vkFreeMemory(_dev.Device, guTmp.Memory, null);
        }
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
        if (_pfProf)
        {
            _pfProfNames.Clear();
            Vk.vkCmdResetQueryPool(_cmdPf, _pfProfPool, 0, PfProfCap);
            Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageBottomOfPipe, _pfProfPool, 0);
        }
        void RunZ(VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, uint gy, uint gz, params VkBuffer[] written)
        {
            CmdRun(_cmdPf, p, d, pc, pcBytes, gx, gy, gz, written);
            if (_pfProf && _pfProfNames.Count + 1 < PfProfCap)
            {
                _pfProfNames.Add(p.Name.StartsWith("gemm") ? $"{p.Name}[{((uint*)pc)[1]}x{((uint*)pc)[2]}{(gz > 1 ? $"/{gz}" : "")}]" : p.Name);
                Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageBottomOfPipe, _pfProfPool, (uint)_pfProfNames.Count);
            }
        }
        void Run(VkPipeline p, (IntPtr set, VkBuffer[] bufs) d, void* pc, uint pcBytes, uint gx, uint gy, params VkBuffer[] written)
            => RunZ(p, d, pc, pcBytes, gx, gy, 1, written);

        uint* pc = stackalloc uint[10];
        // out[M,N] (+)= xin[M,K] @ w[N,K]^T. splitK > 1 (tensor-core path only):
        // slices 1.. write raw partials to _pfPart for pf_addrms16 to fold in.
        // Returns the split count actually used.
        int Gemm(int role, VkBuffer xin, VkBuffer w, VkBuffer outp, int n, int k, bool add, bool swiglu = false, int splitK = 1)
        {
            GemmT? t = _gemmT?[role];
            bool useT = t != null && n % t.BN == 0 && k % t.BK == 0;
            VkPipeline p = useT ? t!.P : _pPfGemm;
            int tm = useT ? t!.BM : _pfGemmTM, tn = useT ? t!.BN : _pfGemmTN;
            if (!useT || k / t!.BK < splitK || (long)(splitK - 1) * M * n > _pfPartElems) splitK = 1;
            var s = PfSet(p, xin, w, splitK > 1 ? _pfPart : _pfX, outp);
            pc[0] = M; pc[1] = (uint)n; pc[2] = (uint)k; pc[3] = (add ? 1u : 0u) | (swiglu ? 2u : 0u);
            if (splitK > 1)
                RunZ(p, s, pc, 16, (uint)((M + tm - 1) / tm), (uint)((n + tn - 1) / tn), (uint)splitK, outp, _pfPart);
            else
                Run(p, s, pc, 16, (uint)((M + tm - 1) / tm), (uint)((n + tn - 1) / tn), outp);
            return splitK;
        }
        // xs = rmsnorm(x (+ split-K partials)) * wn
        void Rms(VkBuffer wn, VkBuffer xs, int parts)
        {
            if (_pPfAddRms != null)
            {
                var s = PfSet(_pPfAddRms!, _pfX, wn, xs, _pfPart);
                pc[0] = (uint)hidden; pc[1] = (uint)(parts - 1); pc[2] = M * (uint)hidden; pc[3] = BitConverter.SingleToUInt32Bits(c.Eps);
                if (parts > 1) Run(_pPfAddRms!, s, pc, 16, M, 1, _pfX, xs);
                else Run(_pPfAddRms!, s, pc, 16, M, 1, xs);
            }
            else
            {
                var s = PfSet(_pPfRmsX, _pfX, wn, xs);
                pc[0] = (uint)hidden; pc[1] = 0; pc[2] = 0; pc[3] = BitConverter.SingleToUInt32Bits(c.Eps);
                Run(_pPfRmsX, s, pc, 16, M, 1, xs);
            }
        }
        uint epsB = BitConverter.SingleToUInt32Bits(c.Eps);
        int pfLayers = int.TryParse(Environment.GetEnvironmentVariable("HYMT_VK_PFLAYERS"), out int pfl)
            ? Math.Min(pfl, c.NumLayers) : c.NumLayers;

        {   // x[m,:] = embd16[tok[m],:]
            var s = PfSet(_pPfEmbed, _embd16, _pfTok, _pfX);
            pc[0] = (uint)hidden;
            Run(_pPfEmbed, s, pc, 4, M, 1, _pfX);
        }

        if (_timing) { Vk.vkCmdResetQueryPool(_cmdPf, _qpool, 0, 16); Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageTopOfPipe, _qpool, 12); }
        // split-K slices for the residual GEMMs (wo, down); partials are folded
        // into x by the following rms (pf_addrms16), so the last layer's down stays whole.
        int skWo = _noSmall || _pPfAddRms == null ? 1 : _splitKWo, skDown = _noSmall || _pPfAddRms == null ? 1 : _splitKDown;
        int parts = 1;
        for (int l = 0; l < pfLayers; l++)
        {
            if (!_noSmall) Rms(W($"blk.{l}.attn_norm.weight"), _pfXs, parts);   // xs = rmsnorm(x) * attn_norm
            Gemm(0, _pfXs, _wq16[l], _pfQkv, qkvDim, hidden, false);   // qkv = xs @ (wq|wk|wv)^T
            if (!_noSmall) {   // rope+rmsnorm k rows -> fp16 K; v rows -> fp16 V
                var s = PfSet(_pPfKvPrep, _pfQkv, _kvK[l], _kvV[l], W($"blk.{l}.attn_k_norm.weight"), _prm);
                pc[0] = (uint)dim; pc[1] = (uint)c.RopeDim; pc[2] = (uint)_kvStride; pc[3] = (uint)qkvDim;
                pc[4] = (uint)qDim; pc[5] = (uint)(qDim + kDim);
                pc[6] = BitConverter.SingleToUInt32Bits(c.RopeBase); pc[7] = epsB;
                Run(_pPfKvPrep, s, pc, 32, (uint)kvHeads, M, _kvK[l], _kvV[l]);
            }
            if (!(_noSmall || _noAttn))
            {
                if (_pPfFa != null)
                {
                    var s = PfSet(_pPfFa, _pfQkv, _kvK[l], _kvV[l], W($"blk.{l}.attn_q_norm.weight"), _prm, _pfAo);
                    pc[0] = (uint)heads; pc[1] = (uint)kvHeads; pc[2] = (uint)dim; pc[3] = (uint)c.RopeDim;
                    pc[4] = (uint)_kvStride; pc[5] = (uint)qkvDim;
                    pc[6] = BitConverter.SingleToUInt32Bits(scale);
                    pc[7] = BitConverter.SingleToUInt32Bits(c.RopeBase); pc[8] = epsB;
                    Run(_pPfFa, s, pc, 36, (uint)kvHeads, (M + 15) / 16, _pfAo);
                }
                else if (_pfFastAttn)
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
            parts = Gemm(1, _pfAo, _wo16[l], _pfX, hidden, qDim, true, splitK: skWo);   // x += ao @ wo^T
            if (!_noSmall) Rms(W($"blk.{l}.ffn_norm.weight"), _pfXs2, parts);   // xs = rmsnorm(x) * ffn_norm
            if (_swiglu)
                Gemm(2, _pfXs2, _wgu16[l], _pfH, 2 * ffn, hidden, false, swiglu: true);   // h = silu(xs@Wg^T) * (xs@Wu^T)
            else
                Gemm(2, _pfXs2, _wgu16[l], _pfGu, 2 * ffn, hidden, false);   // gu = xs @ (gate|up)^T
            if (!_noSmall && !_swiglu) {   // h = silu(gate) * up
                var s = PfSet(_pPfSilu, _pfGu, _pfH);
                uint total = (uint)(seq * ffn);
                pc[0] = (uint)ffn; pc[1] = total;
                Run(_pPfSilu, s, pc, 8, (total + 255) / 256, 1, _pfH);
            }
            parts = Gemm(3, _pfH, _wd16[l], _pfX, hidden, ffn, true, splitK: l + 1 < pfLayers ? skDown : 1);   // x += h @ ffn_down^T
        }

        if (_timing) Vk.vkCmdWriteTimestamp(_cmdPf, VkConst.PipelineStageBottomOfPipe, _qpool, 13);
        {   // normed = rmsnorm(x[last]) * out_norm, single row at index 0
            var s = PfSet(_pPfRms, _pfX, _outNorm, _normed);
            pc[0] = (uint)hidden; pc[1] = (uint)(seq - 1); pc[2] = 0; pc[3] = epsB;
            Run(_pPfRms, s, pc, 16, 1, 1, _normed);
        }
        {   // logits = normed @ output^T — the quant gemv (same kernel as decode)
            GgmlTensorType t = _wtype["output.weight"];
            VkPipeline p = t == GgmlTensorType.Q4_K ? _pGemv4
                : t == GgmlTensorType.Q8_0 ? _pGemv8
                : _pGemv6;
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
        ushort* kd = _kvKHost[layer] + pos * _kvStride;
        ushort* vd = _kvVHost[layer] + pos * _kvStride;
        for (long i = 0; i < (long)len * _kvStride; i++)
        {
            kd[i] = BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(((uint)ks[i]) << 16));
            vd[i] = BitConverter.HalfToUInt16Bits((Half)BitConverter.UInt32BitsToSingle(((uint)vs[i]) << 16));
        }
        _kvK[layer].Flush((ulong)(pos * _kvStride * 2), (ulong)(len * _kvStride * 2));
        _kvV[layer].Flush((ulong)(pos * _kvStride * 2), (ulong)(len * _kvStride * 2));
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
        if (_gVar)
        {
            _pfXs.Invalidate((ulong)(last * hidden * 2), (ulong)(hidden * 2));
            ushort* h16 = (ushort*)_pfXs.Map() + last * hidden;
            using (var fs = File.Create("vk_pfxs.bin"))
            using (var bw = new BinaryWriter(fs))
            {
                for (int i = 0; i < hidden; i++) bw.Write((float)BitConverter.UInt16BitsToHalf(h16[i]));
            }
            _pfXs.Unmap();
        }
        else BinRow("pfxs", _pfXs, last * hidden, hidden);
        BinRow("pfqkv", _pfQkv, last * qkvDim, qkvDim);
        BinRow("pfao", _pfAo, last * qDim, qDim);
        if (!_swiglu) BinRow("pfgu", _pfGu, last * 2 * ffn, 2 * ffn);
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
        // The graph depends only on seq (pos/kvLen flow through _prm), except the
        // Intel fast-attn path which bakes pos into push constants.
        long key = _pfFastAttn ? ((long)pos << 32) | (uint)seq : seq;
        if (key != _pfRecKey)
        {
            RecordPrefill(seq, pos);
            _pfRecKey = key;
        }
        if (_pfProf) Console.WriteLine($"[vk-prof] record={_sw.Elapsed.TotalMilliseconds:F2}ms");
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
        if (_pfProf) { Console.WriteLine($"[vk-prof] record+submit+wait={_sw.Elapsed.TotalMilliseconds:F2}ms"); PrintPfProf(seq); }

        if (Environment.GetEnvironmentVariable("HYMT_VK_NOREAD") != "1")
        {
            _logitsStage.Invalidate(0, _logitsStage.Size);
            Marshal.Copy((IntPtr)_logitsStage.Map(), _logitsHost, 0, _vocab);
            _logitsStage.Unmap();
        }
        return _logitsHost;
    }

    private void PrintPfProf(int seq)
    {
        int n = _pfProfNames.Count + 1;
        ulong[] ts = new ulong[n];
        fixed (ulong* p = ts)
            Vk.vkGetQueryPoolResults(_dev.Device, _pfProfPool, 0, (uint)n, (nuint)(n * 8), p, 8, 1u | 2u);
        double per = _dev.TimestampPeriodNs * 1e-6;
        var agg = new Dictionary<string, (double ms, int cnt)>();
        for (int i = 1; i < n; i++)
        {
            string k = _pfProfNames[i - 1];
            agg.TryGetValue(k, out var a);
            agg[k] = (a.ms + (ts[i] - ts[i - 1]) * per, a.cnt + 1);
        }
        double tot = (ts[n - 1] - ts[0]) * per;
        Console.WriteLine($"[vk-prof] seq={seq} total={tot:F2}ms");
        foreach (var kv in agg.OrderByDescending(x => x.Value.ms))
            Console.WriteLine($"[vk-prof] {kv.Key,-36} {kv.Value.ms,8:F2}ms {kv.Value.ms / tot * 100,5:F1}%  n={kv.Value.cnt} avg={kv.Value.ms / kv.Value.cnt * 1000:F0}us");
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
