using System;
using System.Collections.Generic;
using Sdcb.HyMT2Sharp.Backends.Cuda.Interop;

namespace Sdcb.HyMT2Sharp.Backends.Cuda;

/// <summary>
/// Minimal replacement for TensorSharp's CudaAllocator: owns the device
/// primary context, one stream, the cuBLAS handle, the loaded PTX kernel
/// modules and the backend-owned device buffers. All work for this backend
/// runs on the single stream, so kernel order is program order.
/// </summary>
internal sealed class CudaEnv : IDisposable
{
    public CudaContext Context { get; }
    public CudaStream Stream { get; }
    public CudaCublasHandle Blas { get; }
    public CudaKernels? Kernels { get; }
    public HymtKernels? HymtKernels { get; }
    public int DeviceId { get; }

    private readonly List<IntPtr> _owned = new();
    private bool _disposed;

    private CudaEnv(CudaContext context, CudaStream stream, CudaCublasHandle blas,
        CudaKernels? kernels, HymtKernels? hymtKernels)
    {
        Context = context;
        DeviceId = context.DeviceId;
        Stream = stream;
        Blas = blas;
        Kernels = kernels;
        HymtKernels = hymtKernels;
        Blas.SetStream(stream.Handle);
    }

    /// <summary>
    /// Creates an env on device 0, or returns null when CUDA is unavailable
    /// (no nvcuda, no device, or PTX module load failure). Failures are
    /// swallowed so the caller can fall back to the CPU backend.
    /// </summary>
    public static CudaEnv? TryCreate(int deviceId = 0)
    {
        try
        {
            CudaContext context = CudaContext.Create(deviceId);
            try
            {
                CudaStream stream = CudaStream.Create();
                try
                {
                    CudaCublasHandle blas = CudaCublasHandle.Create();
                    CudaKernels? kernels = CudaKernels.TryCreate();
                    HymtKernels? hymt = HymtKernels.TryCreate();
                    if (kernels is null || hymt is null)
                    {
                        kernels?.Dispose();
                        hymt?.Dispose();
                        blas.Dispose();
                        stream.Dispose();
                        context.Dispose();
                        return null;
                    }
                    return new CudaEnv(context, stream, blas, kernels, hymt);
                }
                catch
                {
                    stream.Dispose();
                    throw;
                }
            }
            catch
            {
                context.Dispose();
                throw;
            }
        }
        catch
        {
            return null;
        }
    }

    /// <summary>cuMemAlloc + track the pointer for Dispose.</summary>
    public IntPtr Alloc(long bytes)
    {
        Context.MakeCurrent();
        CudaDriverApi.cuMemAlloc(out IntPtr ptr, new UIntPtr((ulong)bytes)).ThrowOnError();
        _owned.Add(ptr);
        return ptr;
    }

    public unsafe IntPtr AllocUpload(void* src, long bytes)
    {
        IntPtr ptr = Alloc(bytes);
        CudaDriverApi.cuMemcpyHtoD(ptr, (IntPtr)src, new UIntPtr((ulong)bytes)).ThrowOnError();
        return ptr;
    }

    public void Synchronize() => Stream.Synchronize();

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            Context.MakeCurrent();
            Synchronize();
        }
        catch
        {
        }
        CudaQuantMatmul.ReleaseScratch(this);
        foreach (IntPtr ptr in _owned)
        {
            if (ptr != IntPtr.Zero)
                CudaDriverApi.cuMemFree(ptr);
        }
        _owned.Clear();
        Kernels?.Dispose();
        HymtKernels?.Dispose();
        Blas.Dispose();
        Stream.Dispose();
        Context.Dispose();
    }
}
