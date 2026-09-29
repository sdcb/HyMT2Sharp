namespace Sdcb.HyMT2Sharp.Gguf;

public readonly struct GgufTensorInfo
{
    public GgmlTensorType Type { get; init; }

    /// <summary>Byte offset from the aligned tensor data section (<see cref="GgufFile.DataOffset"/>).</summary>
    public ulong Offset { get; init; }

    /// <summary>GGUF dims, dim 0 is the innermost (row) length.</summary>
    public ulong[] Shape { get; init; }

    public ulong NumElements
    {
        get
        {
            ulong n = 1;
            for (int i = 0; i < Shape.Length; i++)
                n = checked(n * Shape[i]);
            return n;
        }
    }
}
