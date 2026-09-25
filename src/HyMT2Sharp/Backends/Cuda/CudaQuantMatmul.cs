using System;
using System.Collections.Generic;
using Sdcb.HyMT2Sharp.Backends.Cuda.Interop;

namespace Sdcb.HyMT2Sharp.Backends.Cuda;

/// <summary>
/// Kernel-selection core for quantized/dense linear layers, ported from
/// TensorSharp.Backends.Cuda's CudaQuantizedOps.RunResidentMatmul and adapted
/// to <see cref="CudaEnv"/> (raw device pointers, no Tensor/Storage types).
/// Computes result[rows, outDim] = input[rows, inDim] x W^T for a resident
/// weight. Dispatch depends on quant type and row count only: dp4a/vec matvec
/// for decode (rows == 1), dp4a-tiled/MMQ/batched quant kernels for small
/// batches, dequant + cuBLAS F16 GEMM for prefill-sized rows.
/// </summary>
internal static class CudaQuantMatmul
{
    private sealed class ScratchAllocation
    {
        public IntPtr Ptr;
        public long Bytes;
        public readonly List<IntPtr> Retired = new();
    }

    private sealed class ScratchPool
    {
        public readonly Dictionary<CudaEnv, ScratchAllocation> Allocations = new();
    }

    private static readonly object Sync = new();

    // q8_1-quantized activation scratch for the dp4a/MMQ matmul paths.
    private static readonly ScratchPool Q81Scratch = new();

    // Row-batched quantized matmul (weight-reuse across small row counts).
    // TS_CUDA_QMM_BATCHED=0 forces the legacy per-row kernels (A/B switch).
    internal static readonly bool BatchedMatmulEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_BATCHED"), "0", StringComparison.Ordinal);

    // Single-row (decode) quantized matmul for generic quant types: one block
    // per output column rather than the batched kernel's warp-per-column.
    internal static readonly bool VecMatmulEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_VEC"), "0", StringComparison.Ordinal);

    // Large-row (prefill) matmuls: dequantize the weight ONCE to f16 and run a
    // tensor-core cuBLAS GEMM (mirrors ggml_cuda's dequant+cuBLAS route).
    internal static readonly bool F16GemmEnabled =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QMM_F16GEMM"), "0", StringComparison.Ordinal);
    internal static readonly int F16GemmMinRows = EnvInt("TS_CUDA_QMM_F16GEMM_MIN_ROWS", 32);
    internal static readonly long F16GemmMaxWeightBytes = EnvInt("TS_CUDA_QMM_F16GEMM_MAX_MB", 768) * 1024L * 1024L;

    // ggml-style warp-cooperative Q8_0 -> F16 whole-weight conversion for
    // RunF16Gemm. TS_CUDA_Q80_F16_DEQUANT=0 keeps the generic dequantizer.
    internal static bool Q80F16DequantEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_F16_DEQUANT"), "0", StringComparison.Ordinal);

    private static int EnvInt(string name, int fallback)
    {
        string? s = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
    }

    // Per-env reusable f16 scratch for the dequant+GEMM path (weight and
    // activation panels).
    private static readonly ScratchPool WF16Scratch = new();
    private static readonly ScratchPool AF16Scratch = new();

    // Split q8_1 activation scratch for the cp.async MMQ path: dense qs byte
    // rows + separate float scales (ts_quantize_q8_1_split_rows_f32).
    private static readonly ScratchPool Q81SplitQsScratch = new();
    private static readonly ScratchPool Q81SplitDScratch = new();

    private static IntPtr EnsureScratch(ScratchPool pool, CudaEnv env, long bytes)
    {
        lock (Sync)
        {
            if (pool.Allocations.TryGetValue(env, out ScratchAllocation? scratch) &&
                scratch.Bytes >= bytes)
            {
                return scratch.Ptr;
            }

            env.Context.MakeCurrent();
            long doubled = scratch == null || scratch.Bytes > long.MaxValue / 2
                ? 0
                : scratch.Bytes * 2;
            long alloc = Math.Max(bytes, Math.Max(64 * 1024L, doubled));
            CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)alloc)).ThrowOnError();

            if (scratch == null)
            {
                scratch = new ScratchAllocation();
                pool.Allocations.Add(env, scratch);
            }
            else if (scratch.Ptr != IntPtr.Zero)
            {
                scratch.Retired.Add(scratch.Ptr);
            }

            scratch.Ptr = ptr;
            scratch.Bytes = alloc;
            return ptr;
        }
    }

    internal static void ReleaseScratch(CudaEnv env)
    {
        lock (Sync)
        {
            env.Context.MakeCurrent();
            ReleaseScratch(Q81Scratch, env);
            ReleaseScratch(WF16Scratch, env);
            ReleaseScratch(AF16Scratch, env);
            ReleaseScratch(Q81SplitQsScratch, env);
            ReleaseScratch(Q81SplitDScratch, env);
        }
    }

    private static void ReleaseScratch(ScratchPool pool, CudaEnv env)
    {
        if (!pool.Allocations.TryGetValue(env, out ScratchAllocation? scratch))
            return;

        pool.Allocations.Remove(env);
        if (scratch.Ptr != IntPtr.Zero)
            CudaDriverApi.cuMemFree(scratch.Ptr);
        foreach (IntPtr ptr in scratch.Retired)
            CudaDriverApi.cuMemFree(ptr);
    }

    /// <summary>
    /// TS_CUDA_BF16_MATVEC=0 sends single-row BF16 projections through cuBLAS
    /// instead of the dedicated matvec (the matvec measured ~9% faster on a
    /// bandwidth-bound BF16 decode).
    /// </summary>
    internal static bool Bf16MatvecEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_BF16_MATVEC"), "0", StringComparison.Ordinal) &&
        !string.Equals(Environment.GetEnvironmentVariable("TS_DSV4_BF16_MATVEC"), "0", StringComparison.Ordinal);

    /// <summary>
    /// C[rows, outDim] (row-major) = A[rows, inDim] x W[outDim, inDim]^T via
    /// cuBLAS, for operands already in wType/aType device layout. F32
    /// accumulate, F32 output.
    /// </summary>
    private static void RunGemm(
        CudaEnv env, IntPtr weightPtr, int wType, IntPtr aPtr, int aType,
        IntPtr resultPtr, int inDim, int outDim, int rows)
    {
        env.Blas.SetStream(env.Stream.Handle);
        float alpha = 1.0f, beta = 0.0f;
        CublasApi.cublasGemmEx(
            env.Blas.Handle,
            CublasApi.CUBLAS_OP_T, CublasApi.CUBLAS_OP_N,
            outDim, rows, inDim,
            ref alpha,
            weightPtr, wType, inDim,
            aPtr, aType, inDim,
            ref beta,
            resultPtr, CublasApi.CUDA_R_32F, outDim,
            CublasApi.CUBLAS_COMPUTE_32F, CublasApi.CUBLAS_GEMM_DEFAULT).ThrowOnCublasError();
    }

    private static void RunF16Gemm(
        CudaEnv env, CudaKernels kernels,
        IntPtr weightPtr, int ggmlType, IntPtr inputPtr, IntPtr resultPtr,
        int inDim, int outDim, int rows)
    {
        long wElems = (long)inDim * outDim;
        long aElems = (long)rows * inDim;
        IntPtr aF16 = EnsureScratch(AF16Scratch, env, aElems * 2);
        env.Context.MakeCurrent();
        IntPtr stream = env.Stream.Handle;
        IntPtr wF16;
        if (ggmlType == 1)
        {
            // F16 weights are already the cuBLAS operand in the same
            // [outDim, inDim] row-major layout the dequant kernels emit.
            wF16 = weightPtr;
        }
        else
        {
            wF16 = EnsureScratch(WF16Scratch, env, wElems * 2);
            if (ggmlType == 8 && Q80F16DequantEnabled)
                kernels.LaunchDequantWeightQ80F16(weightPtr, wF16, wElems, stream);
            else
                kernels.LaunchDequantWeightF16(weightPtr, wF16, ggmlType, inDim, wElems, stream);
        }
        kernels.LaunchConvertF32F16(inputPtr, aF16, aElems, stream);

        // C[rows, outDim] (row-major) == C_col[outDim, rows]:
        //   C_col = (W_col[inDim, outDim])^T x A_col[inDim, rows]
        // f16 inputs, f32 accumulate + f32 output (CUBLAS_COMPUTE_32F).
        RunGemm(env, wF16, CublasApi.CUDA_R_16F, aF16, CublasApi.CUDA_R_16F,
            resultPtr, inDim, outDim, rows);
    }

    // Q4_0 uses the int8 dp4a matmul for decode AND verify by default
    // (~memory-bound, matches ggml's mul_mat_vec_q); TS_CUDA_Q40_DP4A=0 reverts
    // to the FP32 dequant kernels.
    internal static bool Q40Dp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q40_DP4A"), "0", StringComparison.Ordinal);

    // Q8_0 single-row decode uses the warp-per-column dp4a matvec by default;
    // TS_CUDA_Q80_VEC=0 reverts to the exact FP32 dequant kernel.
    internal static bool Q80VecDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_VEC"), "0", StringComparison.Ordinal);

    // Q4_K single-token decode via q8_1-quantized activation + dp4a (ggml
    // mul_mat_vec_q4_K). TS_CUDA_Q4K_DP4A=0 reverts to the scalar kernel.
    internal static bool Q4KDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q4K_DP4A"), "0", StringComparison.Ordinal);

    // Q5_K / Q6_K single-token decode via ggml-style q8_1 activation dots.
    internal static bool Q5KDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q5K_DP4A"), "0", StringComparison.Ordinal);

    internal static bool Q6KDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q6K_DP4A"), "0", StringComparison.Ordinal);

    // IQ2_XXS / IQ2_S decode via q8_1 + dp4a vec dots.
    internal static bool Iq2xxsDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_IQ2XXS_DP4A"), "0", StringComparison.Ordinal);

    internal static bool Iq2sDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_IQ2S_DP4A"), "0", StringComparison.Ordinal);

    // Single-row IQ2_XXS/IQ2_S matvec using one global q8_1 scratch row.
    internal static bool Iq2VecDp4aEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_IQ2_VEC"), "0", StringComparison.Ordinal);

    // Quantize each 32-value activation block cooperatively with one warp.
    // TS_CUDA_Q81_WARP=0 keeps the legacy one-thread-per-block kernel.
    internal static bool Q81WarpQuantizeEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q81_WARP"), "0", StringComparison.Ordinal);

    // Only use the q8_1 vec matvec when the output is at least this wide
    // (0 = always).
    internal static readonly int Q80VecMinOutDim = EnvInt("TS_CUDA_Q80_VEC_MIN_OUT", 0);

    // Direct int8 tensor-core GEMM over raw Q8_0 blocks for prefill-sized rows
    // (mma.m16n8k32, ggml MMQ-style). TS_CUDA_Q80_MMQ=0 falls back to the
    // dequant+cuBLAS F16 route; MAX_ROWS is the crossover where cuBLAS wins.
    internal static bool Q80MmqEnabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_MMQ"), "0", StringComparison.Ordinal);
    internal static readonly int Q80MmqMaxRows = EnvInt("TS_CUDA_Q80_MMQ_MAX_ROWS", 512);

    // cp.async staging variant of the MMQ kernel (split q8_1 scratch, raw
    // weight windows async-copied to shared). Requires inDim % 256 == 0.
    // TS_CUDA_Q80_MMQ2=0 pins the register-prefetch variant.
    internal static bool Q80Mmq2Enabled { get; set; } =
        !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_Q80_MMQ2"), "0", StringComparison.Ordinal);

    internal static bool SupportsQuantizedType(int ggmlType)
    {
        return ggmlType == 2 ||   // Q4_0
            ggmlType == 3 ||      // Q4_1
            ggmlType == 6 ||      // Q5_0
            ggmlType == 7 ||      // Q5_1
            ggmlType == 8 ||      // Q8_0
            ggmlType == 9 ||      // Q8_1
            ggmlType == 10 ||     // Q2_K
            ggmlType == 11 ||     // Q3_K
            ggmlType == 12 ||     // Q4_K
            ggmlType == 13 ||     // Q5_K
            ggmlType == 14 ||     // Q6_K
            ggmlType == 16 ||     // IQ2_XXS
            ggmlType == 18 ||     // IQ3_XXS
            ggmlType == 20 ||     // IQ4_NL
            ggmlType == 21 ||     // IQ3_S
            ggmlType == 22 ||     // IQ2_S
            ggmlType == 23 ||     // IQ4_XS
            ggmlType == 39 ||     // MXFP4
            ggmlType == 40;       // NVFP4
    }

    /// <summary>
    /// Unquantized weight types whose resident bytes are directly a cuBLAS
    /// operand: F32 (0), F16 (1) and BF16 (30). These skip the dequant pass.
    /// </summary>
    internal static bool IsDenseGemmType(int ggmlType)
        => ggmlType == 0 || ggmlType == 1 || ggmlType == 30;

    /// <summary>Every weight type <see cref="RunResidentMatmul"/> can serve.</summary>
    internal static bool SupportsMatmulType(int ggmlType)
        => IsDenseGemmType(ggmlType) || SupportsQuantizedType(ggmlType);

    /// <summary>
    /// result[rows, outDim] = input[rows, inDim] x W^T for a resident weight at
    /// weightPtr. The ONE place kernel routing lives: everything dispatches on
    /// quant type and row count only.
    /// </summary>
    internal static void RunResidentMatmul(
        CudaEnv env,
        CudaKernels kernels,
        IntPtr weightPtr,
        int ggmlType,
        IntPtr inputPtr,
        IntPtr resultPtr,
        int inDim,
        int outDim,
        int rows,
        int q8Kernel = 0)
    {
        env.Context.MakeCurrent();

        // F16 weights: the resident bytes are already the GEMM operand.
        if (ggmlType == 1)
        {
            RunF16Gemm(env, kernels, weightPtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
            return;
        }

        // BF16 weights stay bit-exact in VRAM; decode (rows == 1) is
        // bandwidth-bound so the dedicated matvec beats a cuBLAS GEMM.
        if (ggmlType == 30)
        {
            if (rows == 1 && Bf16MatvecEnabled && (inDim & 7) == 0)
            {
                kernels.LaunchMatvecBf16(weightPtr, inputPtr, resultPtr, inDim, outDim, env.Stream.Handle);
            }
            else
            {
                IntPtr aBf16 = EnsureScratch(AF16Scratch, env, (long)rows * inDim * 2);
                kernels.LaunchConvertF32Bf16(inputPtr, aBf16, (long)rows * inDim, env.Stream.Handle);
                RunGemm(env, weightPtr, CublasApi.CUDA_R_16BF, aBf16, CublasApi.CUDA_R_16BF,
                    resultPtr, inDim, outDim, rows);
            }
            return;
        }

        // F32 weights (norm-scale-sized dense tensors) go straight to cuBLAS.
        if (ggmlType == 0)
        {
            RunGemm(env, weightPtr, CublasApi.CUDA_R_32F, inputPtr, CublasApi.CUDA_R_32F,
                resultPtr, inDim, outDim, rows);
            return;
        }

        // Q8_0 prefill-sized batches: direct int8 tensor-core GEMM over the raw
        // Q8_0 blocks (mma.m16n8k32, ggml MMQ-style). Weight DRAM traffic is
        // ceil(rows/128) sweeps with NO f16 dequant round trip, so it beats the
        // dequant+cuBLAS route up to ~TS_CUDA_Q80_MMQ_MAX_ROWS rows.
        if (Q80MmqEnabled && ggmlType == 8 && q8Kernel == 0
            && rows >= F16GemmMinRows && rows <= Q80MmqMaxRows && (inDim & 31) == 0)
        {
            if (Q80Mmq2Enabled && (inDim & 255) == 0)
            {
                IntPtr qsScratch = EnsureScratch(Q81SplitQsScratch, env, (long)rows * inDim);
                IntPtr dScratch = EnsureScratch(Q81SplitDScratch, env, (long)rows * (inDim / 32) * sizeof(float));
                kernels.LaunchQuantizeQ81SplitRows(inputPtr, qsScratch, dScratch, inDim, rows, env.Stream.Handle);
                kernels.LaunchQuantMatmulQ80Mmq2(
                    weightPtr, qsScratch, dScratch, resultPtr, inDim, outDim, rows, env.Stream.Handle);
            }
            else
            {
                long mmqScratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr mmqXq = EnsureQ81Scratch(env, mmqScratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, mmqXq, inDim, rows, env.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulQ80Mmq(
                    weightPtr, mmqXq, resultPtr, inDim, outDim, rows, env.Stream.Handle);
            }
            return;
        }

        // Prefill-sized batches: dequant-once + tensor-core cuBLAS GEMM.
        if (F16GemmEnabled && q8Kernel == 0 && rows >= F16GemmMinRows
            && 2L * inDim * outDim <= F16GemmMaxWeightBytes)
        {
            RunF16Gemm(env, kernels, weightPtr, ggmlType, inputPtr, resultPtr, inDim, outDim, rows);
            return;
        }
        // Small multi-row batches run the per-row kernels once per output row;
        // the generic scalar batched kernel tiles TS_QMM_ROW_TILE rows per
        // block, streaming each weight ONCE and reusing it across the tile.
        // rows == 1 (decode) deliberately stays out of this path.
        if (BatchedMatmulEnabled && rows >= 2
            && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
        {
            kernels.LaunchQuantMatmulBatchedF32(
                weightPtr, inputPtr, resultPtr,
                ggmlType, inDim, outDim, rows, env.Stream.Handle);
            return;
        }
        // Q4_K single-token decode: q8_1 activation + dp4a.
        if (Q4KDp4aEnabled && ggmlType == 12 && rows == 1 && (inDim & 255) == 0)
        {
            long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
            IntPtr xqScratch = EnsureQ81Scratch(env, scratchBytes);
            kernels.LaunchQuantizeQ81Rows(
                inputPtr, xqScratch, inDim, rows, env.Stream.Handle, Q81WarpQuantizeEnabled);
            kernels.LaunchQuantMatmulQ4KDp4a(
                weightPtr, xqScratch, resultPtr, inDim, outDim, env.Stream.Handle);
            return;
        }
        // Q5_K/Q6_K decode: same global q8_1 scratch, format-specific dots.
        if (rows == 1 && (inDim & 255) == 0
            && ((ggmlType == 13 && Q5KDp4aEnabled)
                || (ggmlType == 14 && Q6KDp4aEnabled)))
        {
            long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
            IntPtr xqScratch = EnsureQ81Scratch(env, scratchBytes);
            kernels.LaunchQuantizeQ81Rows(
                inputPtr, xqScratch, inDim, 1,
                env.Stream.Handle, Q81WarpQuantizeEnabled);
            if (ggmlType == 13)
            {
                kernels.LaunchQuantMatmulQ5KDp4a(
                    weightPtr, xqScratch, resultPtr,
                    inDim, outDim, env.Stream.Handle);
            }
            else
            {
                kernels.LaunchQuantMatmulQ6KDp4a(
                    weightPtr, xqScratch, resultPtr,
                    inDim, outDim, env.Stream.Handle);
            }
            return;
        }
        // Decode IQ2 matvec: quantize the activation row once, then four
        // warp-owned output columns per CTA reuse it.
        if (Iq2VecDp4aEnabled && rows == 1
            && (ggmlType == 16 || ggmlType == 22)
            && (inDim & 255) == 0)
        {
            long scratchBytes = (long)(inDim / 32) * CudaKernels.Q81BlockBytes;
            IntPtr xqScratch = EnsureQ81Scratch(env, scratchBytes);
            kernels.LaunchQuantizeQ81Rows(
                inputPtr, xqScratch, inDim, 1,
                env.Stream.Handle, Q81WarpQuantizeEnabled);
            kernels.LaunchQuantMatmulIq2VecQ81F32(
                weightPtr, xqScratch, resultPtr,
                ggmlType, inDim, outDim, env.Stream.Handle);
            return;
        }
        if (VecMatmulEnabled && rows == 1
            && ggmlType != 2 && ggmlType != 8 && ggmlType != 16)
        {
            kernels.LaunchQuantMatmulVecF32(
                weightPtr, inputPtr, resultPtr,
                ggmlType, inDim, outDim, env.Stream.Handle);
            return;
        }
        if (ggmlType == 2)
        {
            // Q4_0 int8 dp4a: quantize activation rows to q8_1 once, then the
            // block-tile dp4a GEMM.
            if (Q40Dp4aEnabled && (inDim & 31) == 0)
            {
                long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
                IntPtr xqScratch = EnsureQ81Scratch(env, scratchBytes);
                kernels.LaunchQuantizeQ81Rows(
                    inputPtr, xqScratch, inDim, rows, env.Stream.Handle, Q81WarpQuantizeEnabled);
                kernels.LaunchQuantMatmulQ40Dp4a(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, env.Stream.Handle);
                return;
            }
            if (BatchedMatmulEnabled && rows >= 2 && rows <= CudaKernels.QuantMatmulBatchMaxRows)
            {
                kernels.LaunchQuantMatmulQ40BatchedF32(
                    weightPtr, inputPtr, resultPtr,
                    inDim, outDim, rows, env.Stream.Handle);
                return;
            }
            kernels.LaunchQuantMatmulQ40F32(
                weightPtr,
                inputPtr,
                resultPtr,
                inDim,
                outDim,
                rows,
                env.Stream.Handle);
        }
        else if (ggmlType == 8 && (inDim & 31) == 0
                 && q8Kernel != 3
                 && (rows >= 2 || (Q80VecDp4aEnabled && outDim >= Q80VecMinOutDim))
                 && (q8Kernel == 1 || q8Kernel == 2
                     || (q8Kernel == 0 && (CudaKernels.Q8MmaEnabled || CudaKernels.Q8Dp4aEnabled))))
        {
            // Q8_0 int8 fast path (rows >= 1): quantize activation rows to q8_1
            // once, then either the tensor-core MMA GEMM or the block-tile dp4a
            // GEMM. dp4a covers single-token decode too.
            bool useMma = q8Kernel == 2 || (q8Kernel == 0 && CudaKernels.Q8MmaEnabled && rows >= 2);
            long scratchBytes = (long)rows * (inDim / 32) * CudaKernels.Q81BlockBytes;
            IntPtr xqScratch = EnsureQ81Scratch(env, scratchBytes);
            kernels.LaunchQuantizeQ81Rows(
                inputPtr, xqScratch, inDim, rows, env.Stream.Handle, Q81WarpQuantizeEnabled);
            if (useMma)
                kernels.LaunchQuantMatmulQ80Mma(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, env.Stream.Handle);
            else if (rows == 1)
                kernels.LaunchQuantMatmulQ80Vec(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, env.Stream.Handle);
            else
                kernels.LaunchQuantMatmulQ80Dp4a(
                    weightPtr, xqScratch, resultPtr, inDim, outDim, rows, env.Stream.Handle);
        }
        else if (ggmlType == 8)
        {
            kernels.LaunchQuantMatmulQ80F32(
                weightPtr,
                inputPtr,
                resultPtr,
                inDim,
                outDim,
                rows,
                env.Stream.Handle);
        }
        else if (ggmlType == 16 && (inDim & 255) == 0 && ((inDim / 32) * 36) <= 48 * 1024)
        {
            kernels.LaunchQuantMatmulIq2XxsQ81F32(
                weightPtr,
                inputPtr,
                resultPtr,
                inDim,
                outDim,
                rows,
                env.Stream.Handle);
        }
        else
        {
            kernels.LaunchQuantMatmulF32(
                weightPtr,
                inputPtr,
                resultPtr,
                ggmlType,
                inDim,
                outDim,
                rows,
                env.Stream.Handle);
        }
    }

    private static IntPtr EnsureQ81Scratch(CudaEnv env, long bytes)
        => EnsureScratch(Q81Scratch, env, bytes);
}
