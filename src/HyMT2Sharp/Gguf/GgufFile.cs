using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Text;

namespace Sdcb.HyMT2Sharp.Gguf;

/// <summary>
/// GGUF v2/v3 reader. A path is memory-mapped; a <see cref="Stream"/> is
/// parsed from a growable header buffer and tensor bytes are copied on
/// demand via <see cref="CopyTensor"/>. Both modes share <see cref="Cursor"/>.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    private const uint Magic = 0x46554747; // "GGUF" LE
    private const uint DefaultAlignment = 32;
    // Each stream read is at most 4MB. A stream that does not override
    // Read(Span<byte>) rents an array the size of the span; the cap keeps
    // that rent small. SegmentStream writes the span straight through.
    private const int StreamCopyChunk = 4 * 1024 * 1024;
    // Tokenizer tables are a few MB. A corrupt length must not grow the
    // header buffer toward the size of the whole weight file.
    private const int HeaderLimit = 256 * 1024 * 1024;

    private readonly MemoryMappedFile? _mmf;
    private readonly MemoryMappedViewAccessor? _view;
    private readonly Dictionary<string, object> _kv = new(StringComparer.Ordinal);
    private readonly Dictionary<string, GgufTensorInfo> _tensors = new(StringComparer.Ordinal);
    private readonly Stream? _stream;
    private readonly bool _leaveOpen;
    private byte* _fileBase;
    private byte* _dataBase;
    private long _dataOffset;
    private long _dataLength;
    private bool _disposed;

    /// <summary>Byte offset of the aligned tensor data section.</summary>
    public long DataOffset => _dataOffset;

    /// <summary>Length in bytes of the tensor data section.</summary>
    public long DataLength => _dataLength;

    public IReadOnlyDictionary<string, GgufTensorInfo> Tensors => _tensors;

    public GgufFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        FileStream stream = new(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.RandomAccess);
        long length = stream.Length;
        if (length < 24)
        {
            stream.Dispose();
            throw new InvalidDataException($"GGUF file is too small: {length} bytes");
        }

        try
        {
            _mmf = MemoryMappedFile.CreateFromFile(stream, mapName: null, length, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
            stream = null!;
            _view = _mmf.CreateViewAccessor(0, length, MemoryMappedFileAccess.Read);
            byte* ptr = null;
            _view.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
            _fileBase = ptr;
            Parse(_fileBase, length, length);
        }
        catch (Exception ex)
        {
            stream?.Dispose();
            Dispose();
            if (ex is HeaderShortException)
                throw new EndOfStreamException("Unexpected end of GGUF file");
            throw;
        }
    }

    /// <summary>
    /// Read a GGUF v2/v3 stream. The stream must be readable and seekable.
    /// The image is read from offset 0; <see cref="Stream.Position"/> is ignored.
    /// The stream is closed on <see cref="Dispose"/> unless <paramref name="leaveOpen"/> is set.
    /// <see cref="CopyTensor"/> seeks this stream and is not thread-safe.
    /// </summary>
    public GgufFile(Stream stream, bool leaveOpen = false)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead || !stream.CanSeek)
            throw new ArgumentException("GGUF stream must be readable and seekable.", nameof(stream));

        _stream = stream;
        _leaveOpen = leaveOpen;
        try
        {
            long length = stream.Length;
            if (length < 24)
                throw new InvalidDataException($"GGUF file is too small: {length} bytes");
            ReadHeader(stream, length);
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    /// <summary>
    /// Copy <paramref name="bytes"/> of tensor payload at <paramref name="info"/>'s
    /// data-section offset into <paramref name="dst"/>. Path mode copies from the
    /// mapping. Stream mode seeks, then reads at most 4MB at a time directly into
    /// <paramref name="dst"/>, looping when a read stops early (a stitched weight
    /// stream stops at each segment). Stream mode is not thread-safe.
    /// </summary>
    public void CopyTensor(GgufTensorInfo info, void* dst, ulong bytes)
    {
        if (bytes > (ulong)_dataLength || info.Offset > (ulong)_dataLength - bytes)
            throw new ArgumentOutOfRangeException(nameof(bytes), $"Tensor slice at offset {info.Offset} length {bytes} exceeds the GGUF data section ({_dataLength} bytes).");
        if (bytes == 0)
            return;
        if (_stream is null)
        {
            Buffer.MemoryCopy(_dataBase + (long)info.Offset, dst, bytes, bytes);
            return;
        }

        _stream.Seek(checked(_dataOffset + (long)info.Offset), SeekOrigin.Begin);
        byte* p = (byte*)dst;
        ulong left = bytes;
        while (left != 0)
        {
            int want = left > StreamCopyChunk ? StreamCopyChunk : (int)left;
            int n = _stream.Read(new Span<byte>(p, want));
            if (n <= 0)
                throw new EndOfStreamException("Unexpected end of GGUF stream while reading tensor data.");
            p += n;
            left -= (ulong)n;
        }
    }

    public string GetString(string key, string defaultValue = "")
        => TryGet(key, out object? value) && value is not null && CoerceString(value, out string s) ? s : defaultValue;

    public string[] GetStringArray(string key)
        => TryGet(key, out object? value) && value is object[] arr ? CoerceStringArray(arr) : [];

    public int GetInt32(string key, int defaultValue = 0)
        => TryGet(key, out object? value) && CoerceInt64(value, out long n) ? checked((int)n) : defaultValue;

    public int[] GetInt32Array(string key)
    {
        if (!TryGet(key, out object? value) || value is not object[] arr)
            return [];
        int[] result = new int[arr.Length];
        for (int i = 0; i < arr.Length; i++)
        {
            if (!CoerceInt64(arr[i], out long n))
                return [];
            result[i] = checked((int)n);
        }

        return result;
    }

    public uint GetUint32(string key)
    {
        if (!TryGet(key, out object? value) || !CoerceInt64(value, out long n))
            throw new KeyNotFoundException($"GGUF missing or non-integer key '{key}'");
        return checked((uint)n);
    }

    public uint GetUint32(string key, uint defaultValue)
        => TryGet(key, out object? value) && CoerceInt64(value, out long n) ? checked((uint)n) : defaultValue;

    public float GetFloat32(string key, float defaultValue = 0f)
        => TryGet(key, out object? value) && CoerceFloat(value, out float f) ? f : defaultValue;

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        if (_fileBase != null)
        {
            _view?.SafeMemoryMappedViewHandle.ReleasePointer();
            _fileBase = null;
            _dataBase = null;
        }

        _view?.Dispose();
        _mmf?.Dispose();
        if (!_leaveOpen)
            _stream?.Dispose();
        GC.SuppressFinalize(this);
    }

    private bool TryGet(string key, out object? value) => _kv.TryGetValue(key, out value);

    private void ReadHeader(Stream stream, long fileLength)
    {
        int target = (int)Math.Min(fileLength, 1 << 20);
        byte[] buf = ArrayPool<byte>.Shared.Rent(target);
        try
        {
            stream.Seek(0, SeekOrigin.Begin);
            int filled = Fill(stream, buf, 0, target);
            while (true)
            {
                try
                {
                    fixed (byte* p = buf)
                        Parse(p, filled, fileLength);
                    return;
                }
                catch (HeaderShortException ex)
                {
                    if (ex.Required > fileLength)
                        throw new EndOfStreamException("Unexpected end of GGUF stream.");
                    if (ex.Required > HeaderLimit)
                        throw new InvalidDataException($"GGUF header needs {ex.Required} bytes, above the {HeaderLimit}-byte limit.");
                    // Jump to the bytes this read needs, then overshoot by at
                    // least 1MB so the following small fields (tokenizer
                    // tokens) do not each reparse the header.
                    long slack = Math.Max((long)filled * 2, filled + (1 << 20));
                    long grown = Math.Min(fileLength, Math.Max(ex.Required, slack));
                    if (grown > HeaderLimit)
                        grown = HeaderLimit;
                    int next = (int)grown;
                    if (next <= filled)
                        throw new EndOfStreamException("Unexpected end of GGUF stream.");
                    if (buf.Length < next)
                    {
                        byte[] bigger = ArrayPool<byte>.Shared.Rent(next);
                        Buffer.BlockCopy(buf, 0, bigger, 0, filled);
                        byte[] old = buf;
                        buf = bigger;
                        ArrayPool<byte>.Shared.Return(old);
                    }

                    int got = Fill(stream, buf, filled, next - filled);
                    if (filled + got < next)
                        throw new EndOfStreamException("Unexpected end of GGUF stream.");
                    filled += got;
                    target = next;
                }
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buf);
        }
    }

    private static int Fill(Stream stream, byte[] buf, int offset, int count)
    {
        int got = 0;
        while (got < count)
        {
            int n = stream.Read(buf, offset + got, count - got);
            if (n == 0)
                break;
            got += n;
        }

        return got;
    }

    private void Parse(byte* fileBase, long available, long fileLength)
    {
        _kv.Clear();
        _tensors.Clear();
        Cursor cur = new(fileBase, available);
        uint magic = cur.ReadU32();
        if (magic != Magic)
            throw new InvalidDataException($"Not a GGUF file (magic 0x{magic:X8})");

        uint version = cur.ReadU32();
        if (version is < 2 or > 3)
            throw new NotSupportedException($"GGUF version {version} is not supported (need 2 or 3)");

        ulong tensorCount = cur.ReadU64();
        ulong kvCount = cur.ReadU64();
        if (tensorCount > int.MaxValue || kvCount > int.MaxValue)
            throw new InvalidDataException("GGUF header counts are implausibly large");

        for (ulong i = 0; i < kvCount; i++)
        {
            string key = cur.ReadString();
            if (key.Length == 0)
                throw new InvalidDataException($"GGUF key {i} is empty");
            _kv[key] = ReadValue(ref cur);
        }

        uint alignment = GetUint32("general.alignment", DefaultAlignment);
        if (alignment == 0 || (alignment & (alignment - 1)) != 0)
            throw new InvalidDataException($"GGUF alignment {alignment} is not a power of two");

        for (ulong i = 0; i < tensorCount; i++)
        {
            string name = cur.ReadString();
            uint nDims = cur.ReadU32();
            if (nDims == 0 || nDims > 4)
                throw new InvalidDataException($"Tensor '{name}' has invalid dim count {nDims}");
            ulong[] shape = new ulong[nDims];
            for (int d = 0; d < nDims; d++)
            {
                long dim = cur.ReadI64();
                if (dim < 0)
                    throw new InvalidDataException($"Tensor '{name}' dim {d} is negative");
                shape[d] = (ulong)dim;
            }

            int type = cur.ReadI32();
            ulong offset = cur.ReadU64();
            _tensors[name] = new GgufTensorInfo
            {
                Type = CanonicalTensorType(type),
                Offset = offset,
                Shape = shape,
            };
        }

        long pos = cur.Offset;
        if (tensorCount > 0)
            pos = Align(pos, alignment);
        if (pos > fileLength)
            throw new InvalidDataException("GGUF data section starts past end of file");
        _dataOffset = pos;
        _dataLength = fileLength - pos;
        if (_fileBase != null)
            _dataBase = _fileBase + pos;
    }

    private GgmlTensorType CanonicalTensorType(int rawType)
    {
        // The released HyMT2 Stride16 model predates the final STQ type
        // assignment in llama.cpp PR #22836.  Its raw type 42/file type 41
        // now collide with plain Q2_0.  Require the legacy format marker so
        // a standard Q2_0 file is never silently interpreted as STQ bytes.
        if (rawType == 42 && GetInt32("general.file_type", -1) == 41 &&
            GetString("general.name").Contains("Stride16", StringComparison.OrdinalIgnoreCase))
            return GgmlTensorType.STQ1_0;
        return (GgmlTensorType)rawType;
    }

    private static object ReadValue(ref Cursor cur)
    {
        GgufValueType type = (GgufValueType)cur.ReadI32();
        if (type == GgufValueType.Array)
        {
            GgufValueType elem = (GgufValueType)cur.ReadI32();
            ulong n = cur.ReadU64();
            if (n > int.MaxValue)
                throw new InvalidDataException($"GGUF array length {n} exceeds int.MaxValue");
            object[] arr = new object[(int)n];
            for (int i = 0; i < arr.Length; i++)
                arr[i] = ReadScalar(ref cur, elem);
            return arr;
        }

        return ReadScalar(ref cur, type);
    }

    private static object ReadScalar(ref Cursor cur, GgufValueType type) => type switch
    {
        GgufValueType.Uint8 => cur.ReadU8(),
        GgufValueType.Int8 => cur.ReadI8(),
        GgufValueType.Uint16 => cur.ReadU16(),
        GgufValueType.Int16 => cur.ReadI16(),
        GgufValueType.Uint32 => cur.ReadU32(),
        GgufValueType.Int32 => cur.ReadI32(),
        GgufValueType.Float32 => cur.ReadF32(),
        GgufValueType.Bool => cur.ReadU8() != 0,
        GgufValueType.String => cur.ReadString(),
        GgufValueType.Uint64 => cur.ReadU64(),
        GgufValueType.Int64 => cur.ReadI64(),
        GgufValueType.Float64 => cur.ReadF64(),
        GgufValueType.Array => throw new InvalidDataException("Nested GGUF arrays are not supported"),
        _ => throw new InvalidDataException($"Unknown GGUF value type {(int)type}"),
    };

    private static bool CoerceString(object value, out string s)
    {
        if (value is string str)
        {
            s = str;
            return true;
        }

        s = "";
        return false;
    }

    private static string[] CoerceStringArray(object[] arr)
    {
        string[] result = new string[arr.Length];
        for (int i = 0; i < arr.Length; i++)
            result[i] = arr[i] as string ?? arr[i]?.ToString() ?? "";
        return result;
    }

    private static bool CoerceInt64(object? value, out long n)
    {
        switch (value)
        {
            case byte u8:
                n = u8;
                return true;
            case sbyte i8:
                n = i8;
                return true;
            case ushort u16:
                n = u16;
                return true;
            case short i16:
                n = i16;
                return true;
            case uint u32:
                n = u32;
                return true;
            case int i32:
                n = i32;
                return true;
            case ulong u64:
                n = checked((long)u64);
                return true;
            case long i64:
                n = i64;
                return true;
            default:
                n = 0;
                return false;
        }
    }

    private static bool CoerceFloat(object? value, out float f)
    {
        switch (value)
        {
            case float f32:
                f = f32;
                return true;
            case double f64:
                f = (float)f64;
                return true;
            default:
                if (CoerceInt64(value, out long n))
                {
                    f = n;
                    return true;
                }

                f = 0;
                return false;
        }
    }

    private static long Align(long value, uint alignment)
        => (value + alignment - 1) & ~((long)alignment - 1);

    private enum GgufValueType
    {
        Uint8 = 0,
        Int8 = 1,
        Uint16 = 2,
        Int16 = 3,
        Uint32 = 4,
        Int32 = 5,
        Float32 = 6,
        Bool = 7,
        String = 8,
        Array = 9,
        Uint64 = 10,
        Int64 = 11,
        Float64 = 12,
    }

    private struct Cursor
    {
        private readonly byte* _base;
        private readonly long _length;
        private long _pos;

        public Cursor(byte* fileBase, long length)
        {
            _base = fileBase;
            _length = length;
            _pos = 0;
        }

        public readonly long Offset => _pos;

        public byte ReadU8()
        {
            Ensure(1);
            byte v = _base[_pos];
            _pos += 1;
            return v;
        }

        public sbyte ReadI8() => (sbyte)ReadU8();

        public ushort ReadU16()
        {
            Ensure(2);
            ushort v = Unsafe.ReadUnaligned<ushort>(_base + _pos);
            _pos += 2;
            return v;
        }

        public short ReadI16() => (short)ReadU16();

        public uint ReadU32()
        {
            Ensure(4);
            uint v = Unsafe.ReadUnaligned<uint>(_base + _pos);
            _pos += 4;
            return v;
        }

        public int ReadI32() => (int)ReadU32();

        public ulong ReadU64()
        {
            Ensure(8);
            ulong v = Unsafe.ReadUnaligned<ulong>(_base + _pos);
            _pos += 8;
            return v;
        }

        public long ReadI64() => (long)ReadU64();

        public float ReadF32()
        {
            Ensure(4);
            float v = Unsafe.ReadUnaligned<float>(_base + _pos);
            _pos += 4;
            return v;
        }

        public double ReadF64()
        {
            Ensure(8);
            double v = Unsafe.ReadUnaligned<double>(_base + _pos);
            _pos += 8;
            return v;
        }

        public string ReadString()
        {
            ulong len = ReadU64();
            if (len > int.MaxValue)
                throw new InvalidDataException($"GGUF string length {len} exceeds 2 GB");
            Ensure((long)len);
            string s = Encoding.UTF8.GetString(new ReadOnlySpan<byte>(_base + _pos, (int)len));
            _pos += (long)len;
            return s;
        }

        private readonly void Ensure(long bytes)
        {
            if (_pos < 0 || bytes < 0)
                throw new EndOfStreamException("Unexpected end of GGUF file");
            long end = _pos + bytes;
            if (end > _length)
                throw new HeaderShortException(end);
        }
    }

    /// <summary>Header parse ran past the bytes buffered so far. <see cref="Required"/> is the file offset needed.</summary>
    private sealed class HeaderShortException : Exception
    {
        public HeaderShortException(long required) => Required = required;
        public long Required { get; }
    }
}
