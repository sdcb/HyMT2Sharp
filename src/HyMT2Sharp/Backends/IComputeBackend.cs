using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Compute backend for the model forward pass. When set on
/// <see cref="HunyuanDenseModel"/>, single-token forwards always run on the
/// device through <see cref="ForwardStep"/>; multi-token forwards (prefill)
/// also run on device when <see cref="SupportsPrefill"/> is true, otherwise
/// prefill stays on the CPU engine and its bf16 KV is pushed to the device
/// via <see cref="UploadKv"/>. Implemented by MetalBackend; the default
/// (null) path is the existing CPU engine.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }

    /// <summary>
    /// Whether <see cref="ForwardStep"/> accepts more than one token, i.e.
    /// the backend has a prefill (GEMM + causal attention) path. Decode
    /// (seq == 1) is always supported.
    /// </summary>
    bool SupportsPrefill { get; }

    /// <summary>
    /// Upload the tensors this backend needs (quantized weights and F32 norms).
    /// Throws <see cref="NotSupportedException"/> for weight types it cannot run.
    /// </summary>
    unsafe void LoadModel(GgufFile gguf, ModelConfig config);

    /// <summary>
    /// Copy <paramref name="len"/> bf16 KV positions written by the CPU prefill
    /// into the device KV cache for one layer. <paramref name="k"/>/
    /// <paramref name="v"/> point at the model's flat caches laid out as
    /// pos*kvStride; the backend copies rows [pos, pos+len). Only used when
    /// prefill runs on the CPU (i.e. <see cref="SupportsPrefill"/> is false).
    /// </summary>
    unsafe void UploadKv(int layer, int pos, int len, ushort* k, ushort* v);

    /// <summary>
    /// Run one full forward for <paramref name="tokens"/> appended at KV
    /// position <paramref name="pos"/> and return the logits of the last
    /// token. seq == 1 is the decode path (GEMV); seq &gt; 1 is prefill and
    /// requires <see cref="SupportsPrefill"/>. The device KV must already
    /// hold positions [0, pos); on return it holds [0, pos+seq) for every layer.
    /// </summary>
    float[] ForwardStep(ReadOnlySpan<int> tokens, int pos);

    /// <summary>
    /// Whether the backend keeps its own device-side prefix block store.
    /// When true and the model is configured with KvCacheConfig.Blocks, the
    /// model delegates harvest/restore to <see cref="HarvestPrefix"/>/
    /// <see cref="RestorePrefix"/> instead of the CPU-side KvBlockStore:
    /// blocks then reference KV already resident on the device, so restore
    /// is a zero-copy table update with no upload.
    /// </summary>
    bool SupportsBlockStore => false;

    /// <summary>
    /// Block size in tokens the device store operates on. The model falls back
    /// to the CPU-side KvBlockStore when it differs from the configured
    /// block granularity.
    /// </summary>
    int DeviceBlockTokens => 0;

    /// <summary>
    /// Snapshot device KV for every complete <paramref name="blockTokens"/>-sized
    /// block inside <paramref name="tokens"/> (the live sequence prefix) into
    /// the device block store. Called before the cache is disturbed; only
    /// positions already written on device are harvestable.
    /// </summary>
    void HarvestPrefix(ReadOnlySpan<int> tokens, int blockTokens) { }

    /// <summary>
    /// Restore the longest hash-chained prefix of <paramref name="tokens"/>
    /// found in the device block store. On return the device KV block table
    /// covers [0, n) where n is a multiple of
    /// <paramref name="blockTokens"/> (0 when nothing matched).
    /// </summary>
    int RestorePrefix(ReadOnlySpan<int> tokens, int blockTokens) => 0;

    /// <summary>
    /// Tell the backend the live cache prefix length: positions at or beyond
    /// <paramref name="len"/> may be recycled freely. Called when the model
    /// truncates its cache so a paged backend can protect only live blocks.
    /// </summary>
    void SetCacheLen(int len) { }
}
