namespace Sdcb.HyMT2Sharp.Kernels;

/// <summary>
/// Reusable 64-byte-aligned scratch. Callers overwrite the bytes they need;
/// reuse does not re-clear the whole region.
/// </summary>
public sealed unsafe class ScratchArena : IDisposable
{
    private NativeBuffer? _a;
    private NativeBuffer? _b;
    private NativeBuffer? _c;
    private NativeBuffer? _d;
    private NativeBuffer? _e;
    private NativeBuffer? _f;

    public void* A(nuint bytes) => Ensure(ref _a, bytes);
    public void* B(nuint bytes) => Ensure(ref _b, bytes);
    public void* C(nuint bytes) => Ensure(ref _c, bytes);
    public void* D(nuint bytes) => Ensure(ref _d, bytes);
    public void* E(nuint bytes) => Ensure(ref _e, bytes);
    public void* F(nuint bytes) => Ensure(ref _f, bytes);

    private static void* Ensure(ref NativeBuffer? buf, nuint bytes)
    {
        if (buf != null && buf.Bytes >= bytes)
            return buf.Pointer;
        buf?.Dispose();
        buf = new NativeBuffer(bytes);
        return buf.Pointer;
    }

    public void Dispose()
    {
        _a?.Dispose();
        _b?.Dispose();
        _c?.Dispose();
        _d?.Dispose();
        _e?.Dispose();
        _f?.Dispose();
        _a = null;
        _b = null;
        _c = null;
        _d = null;
        _e = null;
        _f = null;
    }
}
