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

    // Q6_K gemv bandwidth on the real model shapes (v=2048x512, down=6144x2048,
    // embd-as-lm_head=2048x120818). --micro-metal-q6
    public static void RunQ6()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            throw new PlatformNotSupportedException("Metal backend requires macOS.");

        var dev = MtlDevice.Create();
        var lib = dev.NewLibraryFromSource(MslDecodeKernels.Source);
        IntPtr pso = dev.NewPso(dev.NewFunction(lib, "q6k_gemv_fast4"));

        var shapes = new (int InDim, int OutDim)[] {
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 512), (6144, 2048), (2048, 512), (6144, 2048),
            (2048, 120818)
        };
        long totalBytes = 0;
        var bufs = new List<(IntPtr W, IntPtr X, IntPtr Y, int InDim, int OutDim)>();
        foreach (var (inDim, outDim) in shapes)
        {
            long wb = (long)outDim * (inDim / 256) * 210;
            totalBytes += wb;
            var (W, X, Y) = (dev.NewBuffer((nuint)wb), dev.NewBuffer((nuint)(inDim * 4)), dev.NewBuffer((nuint)(outDim * 4)));
            unsafe
            {
                var p = (byte*)ObjC.Send0(W, ObjC.Sel("contents"));
                for (long j = 0; j < wb; j++) p[j] = (byte)(j * 31 + 7);
                var xp = (float*)ObjC.Send0(X, ObjC.Sel("contents"));
                for (int j = 0; j < inDim; j++) xp[j] = 0.001f * j;
            }
            MtlDevice.DidModifyRange(W, 0, (nuint)wb);
            MtlDevice.DidModifyRange(X, 0, (nuint)(inDim * 4));
            bufs.Add((W, X, Y, inDim, outDim));
        }
        Console.WriteLine($"q6k weights/pass = {totalBytes / 1e9:F3} GB");

        // correctness vs managed dequant on a small shape
        {
            int inDim = 512, outDim = 64;
            var rng = new Random(42);
            var w = new byte[outDim * (inDim / 256) * 210];
            rng.NextBytes(w);
            var dB = BitConverter.GetBytes((Half)0.01f);
            for (int b = 0; b < w.Length / 210; b++) Array.Copy(dB, 0, w, b * 210 + 208, 2);
            var x = new float[inDim];
            for (int i = 0; i < inDim; i++) x[i] = (float)(rng.NextDouble() - 0.5);
            IntPtr wb = dev.NewBuffer((nuint)w.Length), xb = dev.NewBuffer((nuint)(inDim * 4)), yb = dev.NewBuffer((nuint)(outDim * 4));
            Marshal.Copy(w, 0, ObjC.Send0(wb, ObjC.Sel("contents")), w.Length);
            Marshal.Copy(x, 0, ObjC.Send0(xb, ObjC.Sel("contents")), inDim);
            MtlDevice.DidModifyRange(wb, 0, (nuint)w.Length);
            MtlDevice.DidModifyRange(xb, 0, (nuint)(inDim * 4));
            var c = CmdCtx.Begin(dev.Queue);
            c.SetPso(pso); c.SetBuffer(wb, 0, 0); c.SetBuffer(xb, 0, 1); c.SetBuffer(yb, 0, 2);
            c.SetInt(3, inDim); c.SetInt(4, outDim);
            c.Dispatch((nuint)((outDim + 3) / 4), 1, 1, 256, 1, 1);
            c.Finish();
            var y = new float[outDim];
            Marshal.Copy(ObjC.Send0(yb, ObjC.Sel("contents")), y, 0, outDim);
            double maxErr = 0;
            for (int r = 0; r < outDim; r++)
            {
                float acc = 0;
                for (int k = 0; k < inDim; k++)
                    acc += x[k] * DequantQ6K(w, r * (inDim / 256) * 210 + (k >> 8) * 210, k & 255);
                maxErr = Math.Max(maxErr, Math.Abs(y[r] - acc));
            }
            Console.WriteLine($"q6k correctness maxErr={maxErr:F4}");

            // bisect: raw q6 codes via scalar kernel vs unit-unpack kernel
            IntPtr psD = dev.NewPso(dev.NewFunction(lib, "q6k_debug_row"));
            IntPtr psU = dev.NewPso(dev.NewFunction(lib, "q6k_debug_row_u"));
            var ya = new float[inDim]; var yu = new float[inDim];
            var c2 = CmdCtx.Begin(dev.Queue);
            c2.SetPso(psD); c2.SetBuffer(wb, 0, 0); c2.SetBuffer(yb, 0, 1); c2.SetInt(2, inDim);
            c2.Dispatch((nuint)inDim, 1, 1, 64, 1, 1);
            c2.Finish();
            Marshal.Copy(ObjC.Send0(yb, ObjC.Sel("contents")), ya, 0, inDim);
            c2 = CmdCtx.Begin(dev.Queue);
            c2.SetPso(psU); c2.SetBuffer(wb, 0, 0); c2.SetBuffer(yb, 0, 1); c2.SetInt(2, inDim);
            c2.Dispatch((nuint)(inDim >> 4), 1, 1, 32, 1, 1);
            c2.Finish();
            Marshal.Copy(ObjC.Send0(yb, ObjC.Sel("contents")), yu, 0, inDim);
            int diffs = 0;
            for (int i = 0; i < inDim; i++)
            {
                int q = Q6Code(w, (i >> 8) * 210, i & 255);
                if (ya[i] != q || yu[i] != q)
                {
                    if (diffs++ < 8)
                        Console.WriteLine($"  i={i} q={q} scalar={ya[i]:F0} unit={yu[i]:F0}");
                }
            }
            Console.WriteLine($"q6 code diffs: {diffs}/{inDim}");
        }

        double encMs = 0, gpuMs = 0;
        for (int t = 0; t < 13; t++)
        {
            var sw = Stopwatch.StartNew();
            var c = CmdCtx.Begin(dev.Queue);
            foreach (var b in bufs)
            {
                c.SetPso(pso);
                c.SetBuffer(b.W, 0, 0); c.SetBuffer(b.X, 0, 1); c.SetBuffer(b.Y, 0, 2);
                c.SetInt(3, b.InDim); c.SetInt(4, b.OutDim);
                c.Dispatch((nuint)((b.OutDim + 3) / 4), 1, 1, 256, 1, 1);
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
        Console.WriteLine($"q6k_gemv_fast4: enc={encMs:F2}ms gpu={gpuMs:F2}ms -> {totalBytes / gpuMs / 1e6:F0} GB/s");
    }

    private static int Q6Code(byte[] blk, int off, int i)
    {
        int j = i >> 7, w = i & 127, l = w & 31, sub = w >> 5;
        int ql = blk[off + j * 64 + l + ((sub & 1) << 5)];
        return ((sub & 2) == 0 ? ql & 15 : ql >> 4) | (((blk[off + 128 + j * 32 + l] >> (sub * 2)) & 3) << 4);
    }

    private static float DequantQ6K(byte[] blk, int off, int i)
    {
        int j = i >> 7, w = i & 127, l = w & 31, sub = w >> 5;
        int ql = blk[off + j * 64 + l + ((sub & 1) << 5)];
        int q6 = ((sub & 2) == 0 ? ql & 15 : ql >> 4) | (((blk[off + 128 + j * 32 + l] >> (sub * 2)) & 3) << 4);
        float d = (float)BitConverter.ToHalf(blk, off + 208);
        return d * (sbyte)blk[off + 192 + j * 8 + (l >> 4) + sub * 2] * (q6 - 32);
    }

    // q4k_gemv_fast/fast4 stage their per-group scales in threadgroup sf[192]/sf[4*192] —
    // any tensor with in_dim > 6144 would overflow threadgroup memory. Guard on the host side.
    private static void CheckKernelLimit(string name, int inDim)
    {
        if ((name is "fast" or "fast4") && inDim > 192 * 32)
            throw new NotSupportedException($"{name} kernel supports in_dim <= {192 * 32} (sf[192] groups/row), got {inDim}");
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

    // attn_scores/attn_combine synthetic check: random q, KV; compare vs managed.
    // --micro-metal-attn
    public static unsafe void RunAttn()
    {
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            throw new PlatformNotSupportedException("Metal backend requires macOS.");
        var dev = MtlDevice.Create();
        var lib = dev.NewLibraryFromSource(MslDecodeKernels.Source);
        IntPtr psS = dev.NewPso(dev.NewFunction(lib, "attn_scores"));
        IntPtr psC = dev.NewPso(dev.NewFunction(lib, "attn_combine"));

        int heads = 4, kvHeads = 2, headDim = 128, kvStride = kvHeads * headDim;
        int T = 9, posBase = 3, kvLen = posBase + T;
        var rng = new Random(7);
        var q = new float[T * heads * headDim];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() - 0.5);
        // bf16 KV: write fp32 value as bf16 bits in ushort[]
        var kb16 = new ushort[kvLen * kvStride];
        var vb16 = new ushort[kvLen * kvStride];
        var kvF = new float[kvLen * kvStride];
        for (int i = 0; i < kvF.Length; i++)
        {
            float f = (float)(rng.NextDouble() - 0.5);
            kvF[i] = f;
            uint b = BitConverter.SingleToUInt32Bits(f);
            kb16[i] = vb16[i] = (ushort)(b >> 16);
        }
        IntPtr qb = dev.NewBuffer((nuint)(q.Length * 4));
        IntPtr kb = dev.NewBuffer((nuint)(kb16.Length * 2));
        IntPtr vb = dev.NewBuffer((nuint)(vb16.Length * 2));
        IntPtr sb = dev.NewBuffer((nuint)(heads * T * kvLen * 4));
        IntPtr ob = dev.NewBuffer((nuint)(T * heads * headDim * 4));
        unsafe
        {
            fixed (float* pq = q) Marshal.Copy(q, 0, ObjC.Send0(qb, ObjC.Sel("contents")), q.Length);
            fixed (ushort* pk = kb16) Buffer.MemoryCopy(pk, (void*)ObjC.Send0(kb, ObjC.Sel("contents")), kb16.Length * 2, kb16.Length * 2);
            fixed (ushort* pv = vb16) Buffer.MemoryCopy(pv, (void*)ObjC.Send0(vb, ObjC.Sel("contents")), vb16.Length * 2, vb16.Length * 2);
        }
        MtlDevice.DidModifyRange(qb, 0, (nuint)(q.Length * 4));
        MtlDevice.DidModifyRange(kb, 0, (nuint)(kb16.Length * 2));
        MtlDevice.DidModifyRange(vb, 0, (nuint)(vb16.Length * 2));

        var c = CmdCtx.Begin(dev.Queue);
        c.SetPso(psS);
        c.SetBuffer(qb, 0, 0); c.SetBuffer(kb, 0, 1); c.SetBuffer(sb, 0, 2);
        c.SetInt(3, heads); c.SetInt(4, kvHeads); c.SetInt(5, headDim);
        c.SetInt(6, kvStride); c.SetInt(7, kvLen); c.SetInt(8, posBase); c.SetInt(9, T);
        int[] tab = new int[64];
        for (int i = 0; i < tab.Length; i++) tab[i] = i;  // identity table
        IntPtr tabB;
        fixed (int* tp = tab) tabB = dev.NewBufferBytes(tp, (nuint)(tab.Length * 4));
        c.SetFloat(10, 1f / MathF.Sqrt(headDim));
        c.SetBuffer(tabB, 0, 11); c.SetInt(12, 6);
        c.Dispatch((nuint)T, (nuint)heads, 1, 128, 1, 1);
        c.SetPso(psC);
        c.SetBuffer(sb, 0, 0); c.SetBuffer(vb, 0, 1); c.SetBuffer(ob, 0, 2);
        c.SetInt(3, heads); c.SetInt(4, kvHeads); c.SetInt(5, headDim);
        c.SetInt(6, kvStride); c.SetInt(7, kvLen); c.SetInt(8, posBase); c.SetInt(9, T);
        c.SetBuffer(tabB, 0, 10); c.SetInt(11, 6);
        c.Dispatch((nuint)T, (nuint)heads, 1, 128, 1, 1);
        c.Finish();

        var o = new float[T * heads * headDim];
        Marshal.Copy(ObjC.Send0(ob, ObjC.Sel("contents")), o, 0, o.Length);
        double maxErr = 0; int bad = 0;
        float scale = 1f / MathF.Sqrt(headDim);
        for (int t = 0; t < T; t++)
        for (int h = 0; h < heads; h++)
        {
            int kvh = h * kvHeads / heads;
            int kvHere = posBase + t + 1;
            var sc = new float[kvHere];
            for (int j = 0; j < kvHere; j++)
            {
                double acc = 0;
                for (int i = 0; i < headDim; i++)
                    acc += q[t * heads * headDim + h * headDim + i] * kvF[j * kvStride + kvh * headDim + i];
                sc[j] = (float)(acc * scale);
            }
            float mx = sc.Max(), sm = 0;
            for (int j = 0; j < kvHere; j++) sm += sc[j] = (float)Math.Exp(sc[j] - mx);
            for (int d = 0; d < headDim; d++)
            {
                double acc = 0;
                for (int j = 0; j < kvHere; j++)
                    acc += sc[j] * kvF[j * kvStride + kvh * headDim + d];
                float want = (float)(acc / sm);
                float got = o[t * heads * headDim + h * headDim + d];
                double err = Math.Abs(want - got);
                if (err > 1e-3) bad++;
                if (err > maxErr) maxErr = err;
            }
        }
        Console.WriteLine($"attn prefill: maxErr={maxErr:F5} bad={bad}/{T * heads * headDim}");
    }

}
