using System.Diagnostics;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Backends.Metal;

// M1 microbenchmark: Q4_K GEMV correctness vs C# dequant + effective bandwidth
// over the hunyuan-dense decode shape set (~780MB weights/token).
public static class MetalGemvMicro
{
    public static void Run()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            throw new PlatformNotSupportedException("Metal backend requires macOS.");

        var dev = MtlDevice.Create();
        var lib = dev.NewLibraryFromSource(MslKernels.Source);
        Console.WriteLine("MSL compiled OK");

        IntPtr psoP = dev.NewPso(dev.NewFunction(lib, "q4k_gemv_portable"));
        IntPtr psoF = dev.NewPso(dev.NewFunction(lib, "q4k_gemv_fast"));
        IntPtr psoF4 = dev.NewPso(dev.NewFunction(lib, "q4k_gemv_fast4"));
        IntPtr psoS = 0;
        try { psoS = dev.NewPso(dev.NewFunction(lib, "q4k_gemv_simd")); Console.WriteLine("simdgroup PSO built"); }
        catch (Exception e) { Console.WriteLine($"simdgroup PSO failed: {e.Message}"); }

        var kernels = new[] { (psoP, "portable", 1), (psoF, "fast", 1), (psoF4, "fast4", 4), (psoS, "simd", 1) };

        // ---------- correctness on a small matrix ----------
        {
            int inDim = 512, outDim = 512;
            var rng = new Random(42);
            var w = new byte[outDim * (inDim / 256) * 144];
            rng.NextBytes(w);
            var dB = BitConverter.GetBytes((Half)0.01f); var mB = BitConverter.GetBytes((Half)0.001f);
            for (int b = 0; b < w.Length / 144; b++) { Array.Copy(dB, 0, w, b * 144, 2); Array.Copy(mB, 0, w, b * 144 + 2, 2); }
            var x = new float[inDim];
            for (int i = 0; i < inDim; i++) x[i] = (float)(rng.NextDouble() - 0.5);

            IntPtr wb = dev.NewBuffer((nuint)w.Length);
            IntPtr xb = dev.NewBuffer((nuint)(inDim * 4));
            IntPtr yb = dev.NewBuffer((nuint)(outDim * 4));
            unsafe
            {
                Marshal.Copy(w, 0, ObjC.Send0(wb, ObjC.Sel("contents")), w.Length);
                Marshal.Copy(x, 0, ObjC.Send0(xb, ObjC.Sel("contents")), x.Length);
            }
            MtlDevice.DidModifyRange(wb, 0, (nuint)w.Length);
            MtlDevice.DidModifyRange(xb, 0, (nuint)(inDim * 4));
            foreach (var (pso, name, colsPerTg) in kernels)
            {
                if (pso == 0) continue;
                CheckKernelLimit(name, inDim);
                var c = CmdCtx.Begin(dev.Queue);
                c.SetPso(pso); c.SetBuffer(wb, 0, 0); c.SetBuffer(xb, 0, 1); c.SetBuffer(yb, 0, 2);
                c.SetInt(3, inDim); c.SetInt(4, outDim);
                c.Dispatch((nuint)((outDim + colsPerTg - 1) / colsPerTg), 1, 1, 256, 1, 1);
                c.Finish();
                var y = new float[outDim];
                unsafe { Marshal.Copy(ObjC.Send0(yb, ObjC.Sel("contents")), y, 0, outDim); }
                double maxErr = 0;
                for (int r = 0; r < outDim; r++)
                {
                    float acc = 0;
                    for (int k = 0; k < inDim; k++)
                        acc += x[k] * DequantQ4K(w, r * (inDim / 256) * 144 + (k >> 8) * 144, k & 255);
                    maxErr = Math.Max(maxErr, Math.Abs(y[r] - acc));
                }
                Console.WriteLine($"{name}: correctness maxErr={maxErr:F4}");
            }
        }

        // ---------- bandwidth bench: hunyuan decode shape set ----------
        var shapes = new (int InDim, int OutDim)[] {
            (2048, 98304), (3072, 65536), (2048, 180224), (5632, 65536), (2048, 120818)
        };
        long totalBytes = 0;

        var bufs = new (IntPtr W, IntPtr X, IntPtr Y, int InDim, int OutDim)[shapes.Length];
        for (int i = 0; i < shapes.Length; i++)
        {
            var (inDim, outDim) = shapes[i];
            long wb2 = (long)outDim * (inDim / 256) * 144;
            totalBytes += wb2;
            bufs[i].W = dev.NewBuffer((nuint)wb2);
            bufs[i].X = dev.NewBuffer((nuint)(inDim * 4));
            bufs[i].Y = dev.NewBuffer((nuint)(outDim * 4));
            bufs[i].InDim = inDim; bufs[i].OutDim = outDim;
            unsafe
            {
                var p = (byte*)ObjC.Send0(bufs[i].W, ObjC.Sel("contents"));
                for (long j = 0; j < wb2; j++) p[j] = (byte)(j * 31 + 7);
                var xp = (float*)ObjC.Send0(bufs[i].X, ObjC.Sel("contents"));
                for (int j = 0; j < inDim; j++) xp[j] = 0.001f * j;
            }
            MtlDevice.DidModifyRange(bufs[i].W, 0, (nuint)wb2);
            MtlDevice.DidModifyRange(bufs[i].X, 0, (nuint)(inDim * 4));
        }
        Console.WriteLine($"weights/token = {totalBytes / 1e9:F3} GB");

        foreach (var (pso, name, colsPerTg) in kernels)
        {
            if (pso == 0) continue;
            double encMs = 0, gpuMs = 0;
            for (int t = 0; t < 13; t++)
            {
                var sw = Stopwatch.StartNew();
                var c = CmdCtx.Begin(dev.Queue);
                foreach (var b in bufs)
                {
                    c.SetPso(pso);
                    CheckKernelLimit(name, b.InDim);
                    c.SetBuffer(b.W, 0, 0); c.SetBuffer(b.X, 0, 1); c.SetBuffer(b.Y, 0, 2);
                    c.SetInt(3, b.InDim); c.SetInt(4, b.OutDim);
                    c.Dispatch((nuint)((b.OutDim + colsPerTg - 1) / colsPerTg), 1, 1, 256, 1, 1);
                }
                c.EndEnc();
                c.Commit();
                double enc = sw.Elapsed.TotalMilliseconds;
                c.Wait();
                c.Drain();
                double gpu = sw.Elapsed.TotalMilliseconds - enc;
                if (t >= 3) { encMs += enc; gpuMs += gpu; }
            }
            encMs /= 10; gpuMs /= 10;
            Console.WriteLine($"{name}: enc={encMs:F2}ms gpu={gpuMs:F2}ms total={encMs + gpuMs:F2}ms -> {totalBytes / gpuMs / 1e6:F0} GB/s, ~{1000 / (encMs + gpuMs):F0} tok/s");
        }
    }

    // q4k_gemv_fast/fast4 stage their per-group scales in threadgroup sf[176]/sf[4*176] —
    // any tensor with in_dim > 5632 would overflow threadgroup memory. Guard on the host side.
    private static void CheckKernelLimit(string name, int inDim)
    {
        if ((name is "fast" or "fast4") && inDim > 176 * 32)
            throw new NotSupportedException($"{name} kernel supports in_dim <= {176 * 32} (sf[176] groups/row), got {inDim}");
    }

    private static float DequantQ4K(byte[] blk, int off, int i)
    {
        int group = i >> 5, w32 = i & 31, pi = group >> 1;
        bool hi = (group & 1) != 0;
        int sc, mn;
        if (group < 4) { sc = blk[off + 4 + group] & 63; mn = blk[off + 8 + group] & 63; }
        else { sc = (blk[off + 8 + group] & 0x0f) | ((blk[off + group] >> 6) << 4);
               mn = (blk[off + 8 + group] >> 4) | ((blk[off + 4 + group] >> 6) << 4); }
        float d = (float)BitConverter.ToHalf(blk, off);
        float dm = (float)BitConverter.ToHalf(blk, off + 2);
        int q = hi ? (blk[off + 16 + pi * 32 + w32] >> 4) : (blk[off + 16 + pi * 32 + w32] & 15);
        return d * sc * q - dm * mn;
    }
}
