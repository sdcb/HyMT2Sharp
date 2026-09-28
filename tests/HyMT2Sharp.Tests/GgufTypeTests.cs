using System.Text;
using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class GgufTypeTests
{
    [Fact]
    public void LegacyStride16Type42IsNormalizedWithoutBreakingQ2()
    {
        string legacy = Path.GetTempFileName();
        string standard = Path.GetTempFileName();
        try
        {
            WriteMinimal(legacy, "Hy-MT2 1.25Bit Stride16", 41, 42);
            WriteMinimal(standard, "plain Q2 model", 41, 42);
            using (GgufFile a = new(legacy))
                Assert.Equal(GgmlTensorType.STQ1_0, a.Tensors["x"].Type);
            using (FileStream fs = File.OpenRead(legacy))
            using (GgufFile aStream = new(fs))
                Assert.Equal(GgmlTensorType.STQ1_0, aStream.Tensors["x"].Type);
            using (GgufFile b = new(standard))
                Assert.Equal(GgmlTensorType.Q2_0, b.Tensors["x"].Type);
            using (FileStream fs = File.OpenRead(standard))
            using (GgufFile bStream = new(fs))
                Assert.Equal(GgmlTensorType.Q2_0, bStream.Tensors["x"].Type);
        }
        finally
        {
            File.Delete(legacy);
            File.Delete(standard);
        }
    }

    private static void WriteMinimal(string path, string name, int fileType, int tensorType)
    {
        using FileStream fs = new(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using BinaryWriter w = new(fs, Encoding.UTF8, leaveOpen: false);
        w.Write(0x46554747u); // GGUF
        w.Write(3u);
        w.Write(1ul); // tensors
        w.Write(2ul); // metadata
        WriteString(w, "general.name");
        w.Write(8); // string
        WriteString(w, name);
        WriteString(w, "general.file_type");
        w.Write(5); // int32
        w.Write(fileType);
        WriteString(w, "x");
        w.Write(1u); // one-dimensional shape
        w.Write(256L);
        w.Write(tensorType);
        w.Write(0ul); // data offset
        while ((fs.Position & 31) != 0) w.Write((byte)0);
        w.Write(new byte[64]);
    }

    private static void WriteString(BinaryWriter w, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }
}
