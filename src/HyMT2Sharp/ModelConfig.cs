using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Model;

public sealed class ModelConfig
{
    public string Architecture { get; init; } = "hunyuan-dense";
    public int NumLayers { get; init; }
    public int HiddenSize { get; init; }
    public int FfnSize { get; init; }
    public int NumHeads { get; init; }
    public int NumKvHeads { get; init; }
    public int HeadDim { get; init; }
    public int VocabSize { get; set; }
    public int ContextLength { get; init; }
    public float RopeBase { get; init; }
    public float RopeScale { get; init; } = 1f;
    public int RopeDim { get; init; }
    public float Eps { get; init; } = 1e-5f;

    public static ModelConfig FromGguf(GgufFile gguf)
    {
        string arch = gguf.GetString("general.architecture", "hunyuan-dense");
        int hidden = (int)gguf.GetUint32($"{arch}.embedding_length");
        int heads = (int)gguf.GetUint32($"{arch}.attention.head_count");
        int kvHeads = (int)gguf.GetUint32($"{arch}.attention.head_count_kv", (uint)heads);
        int headDim = (int)gguf.GetUint32($"{arch}.attention.key_length", 0);
        if (headDim == 0)
            headDim = hidden / heads;
        int ropeDim = (int)gguf.GetUint32($"{arch}.rope.dimension_count", (uint)headDim);
        float ropeBase = gguf.GetFloat32($"{arch}.rope.freq_base", 10000f);
        float alpha = gguf.GetFloat32($"{arch}.rope.scaling.alpha", 0f);
        if (alpha > 0 && headDim > 2)
            ropeBase *= MathF.Pow(alpha, (float)headDim / (headDim - 2));

        return new ModelConfig
        {
            Architecture = arch,
            NumLayers = (int)gguf.GetUint32($"{arch}.block_count"),
            HiddenSize = hidden,
            FfnSize = (int)gguf.GetUint32($"{arch}.feed_forward_length"),
            NumHeads = heads,
            NumKvHeads = kvHeads,
            HeadDim = headDim,
            VocabSize = (int)gguf.GetUint32($"{arch}.vocab_size", 0),
            ContextLength = (int)gguf.GetUint32($"{arch}.context_length", 4096),
            RopeBase = ropeBase,
            RopeDim = ropeDim,
            Eps = gguf.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon", 1e-5f),
        };
    }
}
