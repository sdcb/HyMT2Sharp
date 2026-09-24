using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Backends.Metal;

// Thin Metal wrappers over ObjC msgSend. +1-owned objects are released on Dispose.
// TODO(M2): MtlDevice/Queue/library/PSO/buffers are intentionally never released in this
// spike — wire Dispose into the real backend's lifetime management.
public sealed class MtlDevice
{
    public readonly IntPtr H;
    public readonly IntPtr Queue;   // +1 via newCommandQueue — TODO(M2): release in Dispose
    private MtlDevice(IntPtr h) { H = h; Queue = ObjC.Send0(h, ObjC.Sel("newCommandQueue")); }
    public static MtlDevice Create()
    {
        // MTLCreateSystemDefaultDevice is a C function returning id — autoreleased by ARC
        // default, not +1. Retain so the device survives pool drains.
        IntPtr d = ObjC.Retain(ObjC.CreateSystemDefaultDevice());
        if (d == 0) throw new InvalidOperationException("no Metal device");
        return new MtlDevice(d);
    }

    public IntPtr NewBuffer(nuint bytes)
    {
        // newBufferWithLength:options:  StorageModeShared=0
        return ObjC.Send1P1N(H, ObjC.Sel("newBufferWithLength:options:"), IntPtr.Zero + (nint)bytes, 0);
    }

    public unsafe IntPtr NewBufferBytes(void* src, nuint bytes)
    {
        // newBufferWithBytes:length:options:  StorageModeShared=0
        return ObjC.Send1P2N(H, ObjC.Sel("newBufferWithBytes:length:options:"), (IntPtr)src, bytes, 0);
    }

    public IntPtr NewLibraryFromSource(string msl)
    {
        IntPtr src = ObjC.NsStr(msl);
        IntPtr err = IntPtr.Zero;
        IntPtr lib;
        unsafe { lib = ObjC.Send3P(H, ObjC.Sel("newLibraryWithSource:options:error:"), src, IntPtr.Zero, (IntPtr)(&err)); }
        ObjC.Release(src);
        if (lib == 0)
            throw new InvalidOperationException($"MSL compile failed: {ErrorMessage(err)}");
        return lib;
    }

    public IntPtr NewPso(IntPtr fn)
    {
        IntPtr err = IntPtr.Zero;
        IntPtr pso;
        unsafe { pso = ObjC.Send2P(H, ObjC.Sel("newComputePipelineStateWithFunction:error:"), fn, (IntPtr)(&err)); }
        if (pso == 0) throw new InvalidOperationException($"PSO failed: {ErrorMessage(err)}");
        return pso;
    }

    private static string ErrorMessage(IntPtr err)
    {
        if (err == 0) return "unknown";
        IntPtr desc = ObjC.Send0(err, ObjC.Sel("localizedDescription"));
        if (desc == 0) return "unknown";
        IntPtr cstr = ObjC.Send0(desc, ObjC.Sel("UTF8String"));
        return cstr == 0 ? "unknown" : Marshal.PtrToStringUTF8(cstr) ?? "unknown";
    }

    // Required after CPU writes into a StorageModeShared buffer so the GPU sees them.
    public static void DidModifyRange(IntPtr buf, nuint offset, nuint length) =>
        ObjC.SendV1Range(buf, ObjC.Sel("didModifyRange:"), new ObjC.NSRange(offset, length));

    public IntPtr NewFunction(IntPtr lib, string name)
    {
        IntPtr n = ObjC.NsStr(name);
        IntPtr fn = ObjC.Send1P(lib, ObjC.Sel("newFunctionWithName:"), n);
        ObjC.Release(n);
        if (fn == 0) throw new InvalidOperationException($"function not found: {name}");
        return fn;
    }
}

public struct CmdCtx : IDisposable
{
    public IntPtr Cmd;   // autoreleased
    public IntPtr Enc;   // autoreleased
    private AutoReleasePool _pool;

    public static CmdCtx Begin(IntPtr queue)
    {
        var c = new CmdCtx { _pool = AutoReleasePool.Create() };
        c.Cmd = ObjC.Send0(queue, ObjC.Sel("commandBuffer"));
        c.Enc = ObjC.Send0(c.Cmd, ObjC.Sel("computeCommandEncoder"));
        return c;
    }

    public void SetPso(IntPtr pso) => ObjC.Send1P(Enc, ObjC.Sel("setComputePipelineState:"), pso);
    public void SetBuffer(IntPtr buf, nuint offset, nuint index) =>
        ObjC.SendV1P2N(Enc, ObjC.Sel("setBuffer:offset:atIndex:"), buf, offset, index);
    public void SetInt(nuint index, int v)
    {
        unsafe { ObjC.SendV1P2N(Enc, ObjC.Sel("setBytes:length:atIndex:"), (IntPtr)(&v), 4, index); }
    }

    public void SetFloat(nuint index, float v)
    {
        unsafe { ObjC.SendV1P2N(Enc, ObjC.Sel("setBytes:length:atIndex:"), (IntPtr)(&v), 4, index); }
    }

    public void EndEnc() => ObjC.Send0(Enc, ObjC.Sel("endEncoding"));
    public void Commit() => ObjC.Send0(Cmd, ObjC.Sel("commit"));
    public void Wait() => ObjC.Send0(Cmd, ObjC.Sel("waitUntilCompleted"));
    public void Drain() => _pool.Dispose();
    public void Dispatch(nuint gx, nuint gy, nuint gz, nuint tx, nuint ty, nuint tz) =>
        ObjC.SendV2Size(Enc, ObjC.Sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            new ObjC.MTLSize(gx, gy, gz), new ObjC.MTLSize(tx, ty, tz));

    public void Finish()
    {
        ObjC.Send0(Enc, ObjC.Sel("endEncoding"));
        ObjC.Send0(Cmd, ObjC.Sel("commit"));
        ObjC.Send0(Cmd, ObjC.Sel("waitUntilCompleted"));
        _pool.Dispose();
    }
    public void Dispose() { /* Finish() must be called */ }
}
