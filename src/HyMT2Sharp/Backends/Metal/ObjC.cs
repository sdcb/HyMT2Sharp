using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Backends.Metal;

// Minimal libobjc P/Invoke layer — pure net10.0, no macOS workload.
// Ownership follows Cocoa rules: alloc/init/newXxx/copy return +1 (caller releases);
// everything else is autoreleased within the enclosing NSAutoreleasePool.
public static unsafe partial class ObjC
{
    private const string LibObjC = "libobjc.dylib";
    private const string Metal = "/System/Library/Frameworks/Metal.framework/Metal";

    [LibraryImport(LibObjC, EntryPoint = "objc_getClass")]
    public static partial IntPtr GetClass(IntPtr name);

    [LibraryImport(LibObjC, EntryPoint = "sel_registerName")]
    public static partial IntPtr RegSel(IntPtr name);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send0(IntPtr recv, IntPtr sel);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1P(IntPtr recv, IntPtr sel, IntPtr a);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send1P1N(IntPtr recv, IntPtr sel, IntPtr a, nuint b);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send3P(IntPtr recv, IntPtr sel, IntPtr a, IntPtr b, IntPtr c);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial IntPtr Send2P1N(IntPtr recv, IntPtr sel, IntPtr a, IntPtr b, nuint c);

    // setBuffer:offset:atIndex: / setBytes:length:atIndex:
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV1P2N(IntPtr recv, IntPtr sel, IntPtr a, nuint b, nuint c);

    // dispatchThreadgroups:threadsPerThreadgroup: — MTLSize by value (24B, stack)
    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial void SendV2Size(IntPtr recv, IntPtr sel, MTLSize a, MTLSize b);

    [LibraryImport(LibObjC, EntryPoint = "objc_msgSend")]
    public static partial byte SendBool(IntPtr recv, IntPtr sel);

    [LibraryImport(Metal, EntryPoint = "MTLCreateSystemDefaultDevice")]
    public static partial IntPtr CreateSystemDefaultDevice();

    [StructLayout(LayoutKind.Sequential)]
    public struct MTLSize { public nuint X, Y, Z; public MTLSize(nuint x, nuint y, nuint z) { X = x; Y = y; Z = z; } }

    public static IntPtr GetClass(string name)
    {
        IntPtr n = Marshal.StringToHGlobalAnsi(name);
        try { return GetClass(n); } finally { Marshal.FreeHGlobal(n); }
    }

    public static IntPtr Sel(string name)
    {
        IntPtr n = Marshal.StringToHGlobalAnsi(name);
        try { return RegSel(n); } finally { Marshal.FreeHGlobal(n); }
    }

    // NSString* with UTF-8 — caller owns (+1 via alloc/init)
    public static IntPtr NsStr(string s)
    {
        IntPtr cls = GetClass("NSString");
        IntPtr obj = Send0(cls, Sel("alloc"));
        IntPtr cstr = Marshal.StringToHGlobalAnsi(s);
        try { return Send1P(obj, Sel("initWithUTF8String:"), cstr); }
        finally { Marshal.FreeHGlobal(cstr); }
    }

    public static void Release(IntPtr obj) { if (obj != 0) Send0(obj, Sel("release")); }
    public static IntPtr Retain(IntPtr obj) => Send0(obj, Sel("retain"));
}

// NSAutoreleasePool: alloc/init owned; Dispose drains autoreleased then releases.
public readonly struct AutoReleasePool : IDisposable
{
    private readonly IntPtr _pool;
    private AutoReleasePool(IntPtr pool) => _pool = pool;
    public static AutoReleasePool Create() =>
        new(ObjC.Send0(ObjC.Send0(ObjC.GetClass("NSAutoreleasePool"), ObjC.Sel("alloc")), ObjC.Sel("init")));
    public void Dispose() { ObjC.Send0(_pool, ObjC.Sel("drain")); }
}
