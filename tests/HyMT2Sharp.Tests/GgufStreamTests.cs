using System.Security.Cryptography;
using System.Text;
using Sdcb.HyMT2Sharp.Gguf;

namespace Sdcb.HyMT2Sharp.Tests;

public sealed class GgufStreamTests
{
    private const int CopyChunkLimit = 4 * 1024 * 1024;

    [Fact]
    public void PathFileAndShortReadMatch()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 160_000, blobChars: 1_500_000);
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, bytes);
            using GgufFile fromPath = new(path);
            using FileStream fs = new(path, FileMode.Open, FileAccess.Read, FileShare.Read);
            fs.Position = fs.Length;
            using GgufFile fromFile = new(fs, leaveOpen: true);
            ShortReadStream shortReads = new(bytes, maxRead: 13, boundary: 100);
            using GgufFile fromShort = new(shortReads, leaveOpen: true);

            AssertSameMetadata(fromPath, fromFile);
            AssertSameMetadata(fromPath, fromShort);
            AssertSameTensors(fromPath, fromFile);
            AssertSameTensors(fromPath, fromShort);
            AssertCopiedTwice(fromShort, fromShort.Tensors["big.weight"]);
            Assert.True(shortReads.MaxRequested <= CopyChunkLimit);
            Assert.True(shortReads.ReadCount > bytes.Length / 13);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void SpanShortReadMatchesPath()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 160_000, blobChars: 0);
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, bytes);
            using GgufFile fromPath = new(path);
            SpanShortReadStream spans = new(bytes, maxRead: 13, boundary: 100);
            using GgufFile fromSpan = new(spans, leaveOpen: true);
            AssertSameMetadata(fromPath, fromSpan);
            AssertSameTensors(fromPath, fromSpan);
            Assert.True(spans.SpanReads > 0);
            Assert.True(spans.MaxSpan <= CopyChunkLimit);
        }
        finally
        {
            File.Delete(path);
        }
    }

    private const string RealModelPath = @"D:\_\model\Hy-MT2-1.8B-1.25Bit.gguf";

    [FactIfFileExists(RealModelPath)]
    public void RealModel_PathMatchesStream()
    {
        using GgufFile fromPath = new(RealModelPath);
        using FileStream fs = new(RealModelPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.RandomAccess);
        using GgufFile fromStream = new(fs, leaveOpen: true);

        Assert.Equal(fromPath.DataOffset, fromStream.DataOffset);
        Assert.Equal(fromPath.DataLength, fromStream.DataLength);
        Assert.Equal(fromPath.GetString("general.architecture"), fromStream.GetString("general.architecture"));
        Assert.Equal(fromPath.GetString("general.name"), fromStream.GetString("general.name"));
        Assert.Equal(fromPath.GetInt32("general.file_type", -1), fromStream.GetInt32("general.file_type", -1));
        Assert.Equal(fromPath.GetUint32("general.alignment", 32), fromStream.GetUint32("general.alignment", 32));
        Assert.Equal(fromPath.GetStringArray("tokenizer.ggml.tokens"), fromStream.GetStringArray("tokenizer.ggml.tokens"));
        Assert.Equal(fromPath.GetStringArray("tokenizer.ggml.merges"), fromStream.GetStringArray("tokenizer.ggml.merges"));
        Assert.Equal(fromPath.Tensors.Count, fromStream.Tensors.Count);

        ulong maxEnd = 0;
        foreach ((string name, GgufTensorInfo info) in fromPath.Tensors)
        {
            Assert.True(fromStream.Tensors.TryGetValue(name, out GgufTensorInfo other), name);
            Assert.Equal(info.Type, other.Type);
            Assert.Equal(info.Offset, other.Offset);
            Assert.Equal(info.Shape, other.Shape);
            ulong size = TensorBytes(info);
            ulong end = info.Offset + size;
            Assert.True(end <= (ulong)fromPath.DataLength, name);
            if (end > maxEnd)
                maxEnd = end;

            byte[] buf = new byte[size];
            string hashA, hashB;
            unsafe
            {
                fixed (byte* p = buf)
                    fromPath.CopyTensor(info, p, size);
                hashA = Convert.ToHexString(SHA256.HashData(buf));
                fixed (byte* p = buf)
                    fromStream.CopyTensor(other, p, size);
                hashB = Convert.ToHexString(SHA256.HashData(buf));
            }

            Assert.Equal(hashA, hashB);
        }

        Assert.Equal((ulong)fromPath.DataLength, maxEnd);
    }

    [Fact]
    public void HeaderLargerThanLimitIsRejected()
    {
        using MemoryStream raw = new();
        using (BinaryWriter w = new(raw, Encoding.UTF8, leaveOpen: true))
        {
            w.Write(0x46554747u);
            w.Write(3u);
            w.Write(0ul);
            w.Write(1ul);
            WriteString(w, "k");
            w.Write(8);
            w.Write(300L << 20);
        }

        using ReportedLengthStream stream = new(raw.ToArray(), reportedLength: 400L << 20);
        InvalidDataException ex = Assert.Throws<InvalidDataException>(() => new GgufFile(stream, leaveOpen: true));
        Assert.Contains("header", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RejectsNonSeekableStream()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 1, blobChars: 0);
        using NonSeekableStream stream = new(bytes);
        ArgumentException ex = Assert.Throws<ArgumentException>(() => new GgufFile(stream));
        Assert.Contains("seek", ex.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void TruncatedStreamThrows()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 1, blobChars: 0);
        byte[] cut = bytes[..80];
        using MemoryStream stream = new(cut);
        Exception ex = Assert.ThrowsAny<Exception>(() => new GgufFile(stream, leaveOpen: true));
        Assert.True(ex is EndOfStreamException or InvalidDataException, ex.GetType().Name);
    }

    [Fact]
    public void CopyTensorRejectsOutOfRange()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 1, blobChars: 0);
        using MemoryStream stream = new(bytes);
        using GgufFile gguf = new(stream, leaveOpen: true);
        GgufTensorInfo info = gguf.Tensors["big.weight"];
        ulong size = TensorBytes(info);
        Assert.Equal((ulong)gguf.DataLength, info.Offset + size);
        byte[] buf = new byte[size + 1];
        unsafe
        {
            fixed (byte* p = buf)
            {
                try
                {
                    gguf.CopyTensor(info, p, size + 1);
                    Assert.Fail("CopyTensor accepted an out-of-range length");
                }
                catch (ArgumentOutOfRangeException ex)
                {
                    Assert.Equal("bytes", ex.ParamName);
                }
            }
        }
    }

    [Fact]
    public void LeaveOpenControlsStreamLifetime()
    {
        byte[] bytes = BuildSynthetic(q8Rows: 1, blobChars: 0);
        MemoryStream kept = new(bytes);
        using (GgufFile gguf = new(kept, leaveOpen: true))
            Assert.True(gguf.Tensors.ContainsKey("small.weight"));
        kept.Position = 0;
        Assert.Equal(bytes[0], kept.ReadByte());
        kept.Dispose();

        MemoryStream owned = new(bytes);
        using (GgufFile gguf = new(owned, leaveOpen: false))
        {
            using MemoryStream other = new(bytes);
            using GgufFile again = new(other, leaveOpen: false);
            Assert.Equal(gguf.DataLength, again.DataLength);
        }

        Assert.Throws<ObjectDisposedException>(() => owned.ReadByte());
    }

    private static void AssertSameMetadata(GgufFile a, GgufFile b)
    {
        Assert.Equal(a.DataOffset, b.DataOffset);
        Assert.Equal(a.DataLength, b.DataLength);
        Assert.Equal("hunyuan-dense", a.GetString("general.architecture"));
        Assert.Equal(a.GetString("general.architecture"), b.GetString("general.architecture"));
        Assert.Equal(a.GetString("general.name"), b.GetString("general.name"));
        Assert.Equal(64u, a.GetUint32("general.alignment", 0));
        Assert.Equal(a.GetUint32("general.alignment", 0), b.GetUint32("general.alignment", 0));
        Assert.Equal(1.5f, a.GetFloat32("synthetic.score"));
        Assert.Equal(a.GetFloat32("synthetic.score"), b.GetFloat32("synthetic.score"));
        Assert.Equal(7, a.GetInt32("synthetic.flag"));
        Assert.Equal(a.GetInt32("synthetic.flag"), b.GetInt32("synthetic.flag"));
        Assert.Equal(new[] { "<a>", "hello", "世界" }, a.GetStringArray("tokenizer.ggml.tokens"));
        Assert.Equal(a.GetStringArray("tokenizer.ggml.tokens"), b.GetStringArray("tokenizer.ggml.tokens"));
        if (a.GetString("synthetic.blob").Length > 0)
            Assert.Equal(a.GetString("synthetic.blob"), b.GetString("synthetic.blob"));
    }

    private static void AssertSameTensors(GgufFile a, GgufFile b)
    {
        Assert.Equal(a.Tensors.Count, b.Tensors.Count);
        foreach ((string name, GgufTensorInfo info) in a.Tensors)
        {
            Assert.True(b.Tensors.TryGetValue(name, out GgufTensorInfo other), name);
            Assert.Equal(info.Type, other.Type);
            Assert.Equal(info.Offset, other.Offset);
            Assert.Equal(info.Shape, other.Shape);
            ulong size = TensorBytes(info);
            byte[] left = new byte[size];
            byte[] right = new byte[size];
            unsafe
            {
                fixed (byte* p = left)
                    a.CopyTensor(info, p, size);
                fixed (byte* p = right)
                    b.CopyTensor(other, p, size);
            }

            Assert.True(left.AsSpan().SequenceEqual(right), name);
        }
    }

    private static void AssertCopiedTwice(GgufFile gguf, GgufTensorInfo info)
    {
        ulong size = TensorBytes(info);
        byte[] first = new byte[size];
        byte[] second = new byte[size];
        unsafe
        {
            fixed (byte* p = first)
                gguf.CopyTensor(info, p, size);
            fixed (byte* p = second)
                gguf.CopyTensor(info, p, size);
        }

        Assert.True(first.AsSpan().SequenceEqual(second));
    }

    private static ulong TensorBytes(GgufTensorInfo info)
    {
        ulong n = info.NumElements;
        (int block, int size) = info.Type switch
        {
            GgmlTensorType.F32 => (1, 4),
            GgmlTensorType.F16 or GgmlTensorType.BF16 => (1, 2),
            GgmlTensorType.Q8_0 => (32, 34),
            GgmlTensorType.Q4_K => (256, 144),
            GgmlTensorType.Q5_K => (256, 176),
            GgmlTensorType.Q6_K => (256, 210),
            GgmlTensorType.Q8_K => (256, 292),
            GgmlTensorType.Q2_0C => (512, 130),
            GgmlTensorType.STQ1_0 => (256, 42),
            _ => throw new NotSupportedException(info.Type.ToString()),
        };
        if (n % (ulong)block != 0)
            throw new InvalidDataException($"{info.Type} elements {n} not a multiple of {block}");
        return n / (ulong)block * (ulong)size;
    }

    private static byte[] BuildSynthetic(int q8Rows, int blobChars)
    {
        using MemoryStream ms = new();
        using BinaryWriter w = new(ms, Encoding.UTF8, leaveOpen: true);
        w.Write(0x46554747u);
        w.Write(3u);
        w.Write(2ul);
        w.Write(blobChars > 0 ? 7ul : 6ul);
        WriteKvString(w, "general.architecture", "hunyuan-dense");
        WriteKvString(w, "general.name", "synthetic");
        WriteKvU32(w, "general.alignment", 64);
        WriteKvF32(w, "synthetic.score", 1.5f);
        WriteKvI32(w, "synthetic.flag", 7);
        WriteKvStringArray(w, "tokenizer.ggml.tokens", ["<a>", "hello", "世界"]);
        if (blobChars > 0)
            WriteKvString(w, "synthetic.blob", new string('x', blobChars));

        WriteString(w, "small.weight");
        w.Write(1u);
        w.Write(4L);
        w.Write(0);
        w.Write(0ul);

        WriteString(w, "big.weight");
        w.Write(2u);
        w.Write(32L);
        w.Write((long)q8Rows);
        w.Write(8);
        w.Write(64ul);

        long dataStart = (ms.Position + 63) & ~63L;
        while (ms.Position < dataStart)
            w.Write((byte)0);
        w.Write(1f);
        w.Write(2f);
        w.Write(3f);
        w.Write(4f);
        while (ms.Position < dataStart + 64)
            w.Write((byte)0);
        int q8Bytes = q8Rows * 34;
        byte[] q8 = new byte[q8Bytes];
        for (int i = 0; i < q8.Length; i++)
            q8[i] = (byte)(i * 17);
        w.Write(q8);
        return ms.ToArray();
    }

    private static void WriteKvString(BinaryWriter w, string key, string value)
    {
        WriteString(w, key);
        w.Write(8);
        WriteString(w, value);
    }

    private static void WriteKvU32(BinaryWriter w, string key, uint value)
    {
        WriteString(w, key);
        w.Write(4);
        w.Write(value);
    }

    private static void WriteKvF32(BinaryWriter w, string key, float value)
    {
        WriteString(w, key);
        w.Write(6);
        w.Write(value);
    }

    private static void WriteKvI32(BinaryWriter w, string key, int value)
    {
        WriteString(w, key);
        w.Write(5);
        w.Write(value);
    }

    private static void WriteKvStringArray(BinaryWriter w, string key, string[] values)
    {
        WriteString(w, key);
        w.Write(9);
        w.Write(8);
        w.Write((ulong)values.Length);
        foreach (string value in values)
            WriteString(w, value);
    }

    private static void WriteString(BinaryWriter w, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        w.Write((ulong)bytes.Length);
        w.Write(bytes);
    }

    private sealed class NonSeekableStream : MemoryStream
    {
        public NonSeekableStream(byte[] data) : base(data, writable: false) { }
        public override bool CanSeek => false;
    }

    private class ShortReadStream : Stream
    {
        private readonly byte[] _data;
        private readonly int _maxRead;
        private readonly int _boundary;
        private long _pos;

        public ShortReadStream(byte[] data, int maxRead, int boundary)
        {
            _data = data;
            _maxRead = maxRead;
            _boundary = boundary;
        }

        public int MaxRequested { get; private set; }
        public int ReadCount { get; private set; }
        public override bool CanRead => true;
        public override bool CanSeek => true;
        public override bool CanWrite => false;
        public override long Length => _data.Length;
        public override long Position
        {
            get => _pos;
            set => Seek(value, SeekOrigin.Begin);
        }

        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = CopyShort(buffer.AsSpan(offset, count));
            if (n > 0)
                ReadCount++;
            return n;
        }

        protected int CopyShort(Span<byte> destination)
        {
            if (destination.Length > 0)
                MaxRequested = Math.Max(MaxRequested, destination.Length);
            if (_pos >= _data.Length || destination.Length == 0)
                return 0;
            int room = (int)Math.Min(destination.Length, _data.Length - _pos);
            int toEdge = _boundary - (int)(_pos % _boundary);
            int n = Math.Min(room, Math.Min(_maxRead, toEdge));
            _data.AsSpan((int)_pos, n).CopyTo(destination);
            _pos += n;
            return n;
        }

        public override long Seek(long offset, SeekOrigin origin)
        {
            long next = origin switch
            {
                SeekOrigin.Begin => offset,
                SeekOrigin.Current => _pos + offset,
                SeekOrigin.End => _data.Length + offset,
                _ => throw new ArgumentOutOfRangeException(nameof(origin)),
            };
            if (next < 0 || next > _data.Length)
                throw new IOException($"seek out of range: {next}");
            _pos = next;
            return _pos;
        }

        public override void Flush() { }
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    private sealed class SpanShortReadStream : ShortReadStream
    {
        public SpanShortReadStream(byte[] data, int maxRead, int boundary) : base(data, maxRead, boundary) { }

        public int SpanReads { get; private set; }
        public int MaxSpan { get; private set; }

        public override int Read(Span<byte> buffer)
        {
            if (buffer.Length > 0)
                MaxSpan = Math.Max(MaxSpan, buffer.Length);
            int n = CopyShort(buffer);
            if (n > 0)
                SpanReads++;
            return n;
        }
    }

    private sealed class ReportedLengthStream : MemoryStream
    {
        private readonly long _reportedLength;
        public ReportedLengthStream(byte[] data, long reportedLength) : base(data, writable: false)
            => _reportedLength = reportedLength;
        public override long Length => _reportedLength;
    }
}

public sealed class FactIfFileExistsAttribute : FactAttribute
{
    public FactIfFileExistsAttribute(string path)
    {
        if (!File.Exists(path))
            Skip = $"GGUF model not present: {path}";
    }
}
