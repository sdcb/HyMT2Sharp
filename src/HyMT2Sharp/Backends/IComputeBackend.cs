using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Model;

/// <summary>
/// Decode-path compute backend. When set on <see cref="HunyuanDenseModel"/>,
/// prefill still runs on the CPU engine (its bf16 KV is pushed to the device
/// via <see cref="UploadKv"/>), while single-token forwards run entirely on
/// the device through <see cref="DecodeStep"/>. Implemented by MetalBackend;
/// the default (null) path is the existing CPU engine.
/// </summary>
public interface IComputeBackend : IDisposable
{
    string Name { get; }

    /// <summary>
    /// Upload the tensors this backend needs (quantized weights and F32 norms).
    /// Throws <see cref="NotSupportedException"/> for weight types it cannot run.
    /// </summary>
    unsafe void LoadModel(GgufFile gguf, ModelConfig config);

    /// <summary>
    /// Copy <paramref name="len"/> bf16 KV positions written by the CPU prefill
    /// into the device KV cache for one layer. <paramref name="k"/>/
    /// <paramref name="v"/> point at the model's flat caches laid out as
    /// pos*kvStride; the backend copies rows [pos, pos+len).
    /// </summary>
    unsafe void UploadKv(int layer, int pos, int len, ushort* k, ushort* v);

    /// <summary>
    /// Run one full decode step for <paramref name="tokenId"/> at KV position
    /// <paramref name="pos"/> and return its logits. The device KV must already
    /// hold positions [0, pos).
    /// </summary>
    float[] DecodeStep(int tokenId, int pos);
}
