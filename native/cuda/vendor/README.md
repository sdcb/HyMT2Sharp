# Vendored CUDA kernels

`tensorsharp_kernels.cu` is vendored verbatim from TensorSharp
(`TensorSharp.Backends.Cuda/native/kernels/tensorsharp_kernels.cu`), upstream
commit `fe0be2749b6e3055543c33e36d76f7468d8947e4` (2026-08-30).

It supplies the runtime for `src/HyMT2Sharp/Backends/Cuda`: quantized matmul
(dp4a vec/MMQ/batched + cuBLAS F16 GEMM routing), GQA decode/prefill
attention, rmsnorm, NeoX rope, silu-mul, index/gather and elementwise ops.

The companion managed layer (`Backends/Cuda/Interop/*`, `CudaModule.cs`,
`CudaKernels.cs`, `CudaContext.cs`, `CudaStream.cs`, `CudaCublasHandle.cs`)
is vendored from the same commit with only the namespace renamed to
`Sdcb.HyMT2Sharp.Backends.Cuda`.

Rebuild `native/ptx/tensorsharp_kernels.ptx` with `../build-ptx.ps1` after
syncing this file from upstream.
