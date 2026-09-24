using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Kernels;

public sealed unsafe class NativeBuffer : IDisposable
{
    private void* _ptr;
    private nuint _bytes;

    public NativeBuffer(nuint bytes)
    {
        _bytes = bytes;
        _ptr = NativeMemory.AlignedAlloc(bytes == 0 ? 1 : bytes, 64);
        NativeMemory.Clear(_ptr, bytes);
    }

    public void* Pointer => _ptr;
    public nuint Bytes => _bytes;
    public Span<byte> Span => new(_ptr, checked((int)_bytes));

    public void Dispose()
    {
        if (_ptr == null)
            return;
        NativeMemory.AlignedFree(_ptr);
        _ptr = null;
        GC.SuppressFinalize(this);
    }

    ~NativeBuffer()
    {
        if (_ptr != null)
            NativeMemory.AlignedFree(_ptr);
    }
}
