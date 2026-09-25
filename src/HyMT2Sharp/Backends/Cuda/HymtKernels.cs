using System;
using Sdcb.HyMT2Sharp.Backends.Cuda.Interop;

namespace Sdcb.HyMT2Sharp.Backends.Cuda;

/// <summary>
/// Loader + launch wrappers for the small HyMT2-specific kernels in
/// hymt_kernels.ptx (built from native/cuda/hymt_kernels.cu). Kept in its own
/// module so the vendored TensorSharp PTX stays byte-identical.
/// </summary>
internal sealed unsafe class HymtKernels : IDisposable
{
    private const int BlockSize = 256;
    private readonly CudaModule module;
    private readonly IntPtr rowsToHeadFirstF32;
    private readonly IntPtr rowsToHeadFirstBf16;
    private readonly IntPtr rowsToHeadFirstF32Dyn;
    private readonly IntPtr addRmsNormF32;
    private readonly IntPtr attnPrepF32;
    private readonly IntPtr matvecQ4KQ81F32;
    private readonly IntPtr matvecQ6KQ81F32;

    private HymtKernels(CudaModule module)
    {
        this.module = module;
        rowsToHeadFirstF32 = module.GetFunction("hymt_rows_to_head_first_f32");
        rowsToHeadFirstBf16 = module.GetFunction("hymt_rows_to_head_first_bf16");
        rowsToHeadFirstF32Dyn = module.GetFunction("hymt_rows_to_head_first_f32_dyn");
        addRmsNormF32 = module.GetFunction("hymt_add_rmsnorm_f32");
        attnPrepF32 = module.GetFunction("hymt_attn_prep_f32");
        matvecQ4KQ81F32 = module.GetFunction("hymt_matvec_q4k_q81_f32");
        matvecQ6KQ81F32 = module.GetFunction("hymt_matvec_q6k_q81_f32");
    }

    public static HymtKernels? TryCreate()
    {
        string? path = CudaKernels.LocatePtxPath("hymt_kernels.ptx");
        if (path == null)
        {
            Console.Error.WriteLine(
                "WARNING: HyMT2Sharp CUDA kernels are unavailable: hymt_kernels.ptx not found.");
            return null;
        }

        try
        {
            return new HymtKernels(CudaModule.LoadFromFile(path));
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine(
                $"WARNING: HyMT2Sharp CUDA kernels are unavailable: loading '{path}' failed: {ex.Message}");
            return null;
        }
    }

    /// <summary>
    /// src[seq][heads*dim] -> dst[heads][dstSeqStride][dim] with seq rows written
    /// starting at position <paramref name="startPos"/>. For seq == 1 this is a
    /// plain KV append; for seq > 1 it transposes row-major activation rows into
    /// the head-first layout the GQA kernels read.
    /// </summary>
    public void LaunchRowsToHeadFirstF32(
        IntPtr src, IntPtr dst, int heads, int seq, int dim, int startPos, int dstSeqStride, IntPtr stream)
    {
        IntPtr srcArg = src, dstArg = dst;
        int headsArg = heads, seqArg = seq, dimArg = dim, startPosArg = startPos, dstSeqStrideArg = dstSeqStride;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &headsArg, &seqArg, &dimArg, &startPosArg, &dstSeqStrideArg };
        long count = (long)heads * seq * dim;
        Launch(rowsToHeadFirstF32, Grid(count), stream, args);
    }

    /// <summary>Same layout transform, reading bf16 rows and writing f32.</summary>
    public void LaunchRowsToHeadFirstBf16(
        IntPtr src, IntPtr dst, int heads, int seq, int dim, int startPos, int dstSeqStride, IntPtr stream)
    {
        IntPtr srcArg = src, dstArg = dst;
        int headsArg = heads, seqArg = seq, dimArg = dim, startPosArg = startPos, dstSeqStrideArg = dstSeqStride;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &headsArg, &seqArg, &dimArg, &startPosArg, &dstSeqStrideArg };
        long count = (long)heads * seq * dim;
        Launch(rowsToHeadFirstBf16, Grid(count), stream, args);
    }

    /// <summary>
    /// Same append as <see cref="LaunchRowsToHeadFirstF32"/>, but the write
    /// position comes from the dyn buffer (slot 1 = KV_WRITE_POS) so a captured
    /// CUDA-graph decode step appends at the per-replay position.
    /// </summary>
    public void LaunchRowsToHeadFirstF32Dyn(
        IntPtr src, IntPtr dst, int heads, int seq, int dim, IntPtr dyn, int dstSeqStride, IntPtr stream)
    {
        IntPtr srcArg = src, dstArg = dst, dynArg = dyn;
        int headsArg = heads, seqArg = seq, dimArg = dim, dstSeqStrideArg = dstSeqStride;
        void** args = stackalloc void*[] { &srcArg, &dstArg, &headsArg, &seqArg, &dimArg, &dynArg, &dstSeqStrideArg };
        long count = (long)heads * seq * dim;
        Launch(rowsToHeadFirstF32Dyn, Grid(count), stream, args);
    }

    /// <summary>
    /// h += b then out = rmsnorm(h) * w — fuses residual add + the following
    /// RMSNorm into one node. Single block.
    /// </summary>
    public void LaunchAddRmsNormF32(
        IntPtr h, IntPtr b, IntPtr w, IntPtr out_, int n, float eps, IntPtr stream)
    {
        IntPtr hArg = h, bArg = b, wArg = w, outArg = out_;
        int nArg = n;
        float epsArg = eps;
        void** args = stackalloc void*[] { &hArg, &bArg, &wArg, &outArg, &nArg, &epsArg };
        // Single-block kernel: use the widest CTA the kernel supports so a
        // 2048-element row keeps more loads in flight per element.
        CudaDriverApi.cuLaunchKernel(
            addRmsNormF32, 1, 1, 1, 1024, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }

    /// <summary>
    /// One launch = rope(Q)+rope(K)+rmsnorm(Q)+rmsnorm(K)+append(K)+append(V)
    /// for a decode step. Grid = heads + 2*kvHeads blocks; pos comes from
    /// dyn[KV_WRITE_POS], the rope table is the single-position row filled by
    /// ts_fill_neox_rope_tables_dyn_f32.
    /// </summary>
    public void LaunchAttnPrepF32(
        IntPtr q, IntPtr k, IntPtr v,
        IntPtr qNormW, IntPtr kNormW,
        IntPtr cosTab, IntPtr sinTab,
        IntPtr kvK, IntPtr kvV,
        int heads, int kvHeads, int dim, int ropeHalf, int kvCap,
        float eps, IntPtr dyn, IntPtr stream)
    {
        IntPtr qArg = q, kArg = k, vArg = v, qNormWArg = qNormW, kNormWArg = kNormW;
        IntPtr cosArg = cosTab, sinArg = sinTab, kvKArg = kvK, kvVArg = kvV, dynArg = dyn;
        int headsArg = heads, kvHeadsArg = kvHeads, dimArg = dim, ropeHalfArg = ropeHalf, kvCapArg = kvCap;
        float epsArg = eps;
        void** args = stackalloc void*[]
        {
            &qArg, &kArg, &vArg, &qNormWArg, &kNormWArg, &cosArg, &sinArg,
            &kvKArg, &kvVArg, &headsArg, &kvHeadsArg, &dimArg, &ropeHalfArg,
            &kvCapArg, &epsArg, &dynArg
        };
        CudaDriverApi.cuLaunchKernel(
            attnPrepF32, (uint)(heads + 2 * kvHeads), 1, 1, BlockSize, 1, 1, 0,
            stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }

    /// <summary>
    /// Decode matvec (rows == 1): warp-per-row Q4_K/Q6_K kernel over a q8_1
    /// activation row. 256 threads = 8 warps = 8 output rows per CTA.
    /// </summary>
    public void LaunchMatvecQ81F32(
        int ggmlType, IntPtr weights, IntPtr xq, IntPtr output,
        int inDim, int outDim, IntPtr stream)
    {
        IntPtr wArg = weights, xqArg = xq, outArg = output;
        int inDimArg = inDim, outDimArg = outDim;
        void** args = stackalloc void*[] { &wArg, &xqArg, &outArg, &inDimArg, &outDimArg };
        IntPtr fn = ggmlType switch
        {
            12 => matvecQ4KQ81F32,
            14 => matvecQ6KQ81F32,
            _ => throw new ArgumentOutOfRangeException(nameof(ggmlType), ggmlType, null),
        };
        // 8 warps per CTA -> one CTA covers 8 output rows.
        uint grid = (uint)((outDim + 7) / 8);
        CudaDriverApi.cuLaunchKernel(
            fn, grid, 1, 1, BlockSize, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }

    private static uint Grid(long count) => (uint)Math.Max(1, (count + BlockSize - 1) / BlockSize);

    private static void Launch(IntPtr function, uint grid, IntPtr stream, void** args)
    {
        CudaDriverApi.cuLaunchKernel(
            function, grid, 1, 1, BlockSize, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
    }

    public void Dispose() => module.Dispose();
}
