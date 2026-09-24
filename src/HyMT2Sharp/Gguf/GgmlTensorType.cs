namespace Sdcb.HyMT2Sharp.Gguf;

/// <summary>
/// On-disk <c>ggml_type</c> codes from llama.cpp <c>ggml.h</c>.
/// </summary>
public enum GgmlTensorType
{
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    TQ1_0 = 34,
    TQ2_0 = 35,
    MXFP4 = 39,
    Q2_0C = 40,
    Q2_0 = 42,
    // Canonical llama.cpp PR #22836 value.  GgufFile normalizes the older
    // HyMT2 Stride16 files (raw type 42) to this value using their metadata.
    STQ1_0 = 43,
}
