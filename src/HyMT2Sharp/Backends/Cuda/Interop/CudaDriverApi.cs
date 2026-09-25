using System;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Backends.Cuda.Interop
{
    internal static partial class CudaDriverApi
    {
        private const string LibName = "cuda";

        [LibraryImport(LibName)]
        public static partial int cuInit(uint flags);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGet(out int device, int ordinal);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetCount(out int count);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetName(byte[] name, int len, int device);

        [LibraryImport(LibName, EntryPoint = "cuDeviceTotalMem_v2")]
        public static partial int cuDeviceTotalMem(out UIntPtr bytes, int device);

        [LibraryImport(LibName, EntryPoint = "cuMemGetInfo_v2")]
        public static partial int cuMemGetInfo(out UIntPtr free, out UIntPtr total);

        [LibraryImport(LibName)]
        public static partial int cuDeviceGetAttribute(out int value, int attribute, int device);

        [LibraryImport(LibName, EntryPoint = "cuCtxCreate_v2")]
        public static partial int cuCtxCreate(out IntPtr ctx, uint flags, int device);

        [LibraryImport(LibName, EntryPoint = "cuCtxDestroy_v2")]
        public static partial int cuCtxDestroy(IntPtr ctx);

        [LibraryImport(LibName)]
        public static partial int cuCtxSetCurrent(IntPtr ctx);

        [LibraryImport(LibName)]
        public static partial int cuCtxGetCurrent(out IntPtr ctx);

        [LibraryImport(LibName)]
        public static partial int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

        [LibraryImport(LibName)]
        public static partial int cuDevicePrimaryCtxRelease(int device);

        [LibraryImport(LibName, EntryPoint = "cuMemAlloc_v2")]
        public static partial int cuMemAlloc(out IntPtr devicePtr, UIntPtr byteSize);

        [LibraryImport(LibName, EntryPoint = "cuMemFree_v2")]
        public static partial int cuMemFree(IntPtr devicePtr);

        /// <summary>Page-locked (pinned) host allocation. A captured HtoD memcpy
        /// node whose source is pinned host memory re-reads the buffer on every
        /// graph launch, which is how decode-graph replays receive per-token
        /// parameters (see CudaDecodeDynParams).</summary>
        [LibraryImport(LibName, EntryPoint = "cuMemHostAlloc")]
        public static partial int cuMemHostAlloc(out IntPtr hostPtr, UIntPtr byteSize, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuMemFreeHost")]
        public static partial int cuMemFreeHost(IntPtr hostPtr);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoD_v2")]
        public static partial int cuMemcpyHtoD(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyHtoDAsync_v2")]
        public static partial int cuMemcpyHtoDAsync(IntPtr dstDevice, IntPtr srcHost, UIntPtr byteCount, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoH_v2")]
        public static partial int cuMemcpyDtoH(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoHAsync_v2")]
        public static partial int cuMemcpyDtoHAsync(IntPtr dstHost, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoD_v2")]
        public static partial int cuMemcpyDtoD(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount);

        [LibraryImport(LibName, EntryPoint = "cuMemcpyDtoDAsync_v2")]
        public static partial int cuMemcpyDtoDAsync(IntPtr dstDevice, IntPtr srcDevice, UIntPtr byteCount, IntPtr stream);

        [LibraryImport(LibName, EntryPoint = "cuMemsetD8_v2")]
        public static partial int cuMemsetD8(IntPtr dstDevice, byte value, UIntPtr count);

        [LibraryImport(LibName, EntryPoint = "cuMemsetD8Async")]
        public static partial int cuMemsetD8Async(IntPtr dstDevice, byte value, UIntPtr count, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuModuleLoadData(out IntPtr module, IntPtr image);

        [LibraryImport(LibName, StringMarshalling = StringMarshalling.Utf8)]
        public static partial int cuModuleGetFunction(out IntPtr function, IntPtr module, string name);

        [LibraryImport(LibName)]
        public static partial int cuModuleUnload(IntPtr module);

        [LibraryImport(LibName)]
        public static partial int cuFuncSetAttribute(IntPtr function, int attribute, int value);

        // CUfunction_attribute: opt-in cap for dynamic shared memory per block
        // (default launches are limited to 48 KB without it).
        public const int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8;

        [LibraryImport(LibName)]
        public static partial int cuLaunchKernel(
            IntPtr function,
            uint gridDimX,
            uint gridDimY,
            uint gridDimZ,
            uint blockDimX,
            uint blockDimY,
            uint blockDimZ,
            uint sharedMemBytes,
            IntPtr stream,
            IntPtr kernelParams,
            IntPtr extra);

        [LibraryImport(LibName)]
        public static partial int cuStreamCreate(out IntPtr stream, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuStreamDestroy_v2")]
        public static partial int cuStreamDestroy(IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuStreamSynchronize(IntPtr stream);

        // ---- CUDA Graphs (stream capture) ----
        [LibraryImport(LibName, EntryPoint = "cuStreamBeginCapture_v2")]
        public static partial int cuStreamBeginCapture(IntPtr stream, int mode);

        [LibraryImport(LibName)]
        public static partial int cuStreamEndCapture(IntPtr stream, out IntPtr graph);

        [LibraryImport(LibName)]
        public static partial int cuGraphInstantiateWithFlags(out IntPtr graphExec, IntPtr graph, ulong flags);

        [LibraryImport(LibName)]
        public static partial int cuGraphLaunch(IntPtr graphExec, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cuGraphExecDestroy(IntPtr graphExec);

        [LibraryImport(LibName)]
        public static partial int cuGraphDestroy(IntPtr graph);

        // CUstreamCaptureMode: 0 = GLOBAL, 1 = THREAD_LOCAL, 2 = RELAXED.
        public const int CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1;
        // Relaxed capture permits API calls that are unsafe-by-default during
        // capture (notably cuMemAlloc for pool misses); the capture owner takes
        // responsibility for the referenced memory's lifetime.
        public const int CU_STREAM_CAPTURE_MODE_RELAXED = 2;

        [LibraryImport(LibName)]
        public static partial int cuGetErrorString(int error, out IntPtr str);

        // ---- CUDA Events ----
        [LibraryImport(LibName)]
        public static partial int cuEventCreate(out IntPtr phEvent, uint flags);

        [LibraryImport(LibName, EntryPoint = "cuEventDestroy_v2")]
        public static partial int cuEventDestroy(IntPtr hEvent);

        [LibraryImport(LibName, EntryPoint = "cuEventRecord")]
        public static partial int cuEventRecord(IntPtr hEvent, IntPtr hStream);

        [LibraryImport(LibName)]
        public static partial int cuEventSynchronize(IntPtr hEvent);

        [LibraryImport(LibName)]
        public static partial int cuStreamWaitEvent(IntPtr hStream, IntPtr hEvent, uint flags);

        // HyMT addition (not in the vendored set): pair with cuEventRecord for
        // GPU-side step timing diagnostics.
        [LibraryImport(LibName)]
        public static partial int cuEventElapsedTime(out float pMilliseconds, IntPtr hStart, IntPtr hEnd);

        // ---- Peer-to-Peer (multi-GPU) ----
        [LibraryImport(LibName)]
        public static partial int cuDeviceCanAccessPeer(out int canAccessPeer, int dev, int peerDev);

        [LibraryImport(LibName)]
        public static partial int cuCtxEnablePeerAccess(IntPtr peerContext, uint flags);

        [LibraryImport(LibName)]
        public static partial int cuMemcpyPeerAsync(
            IntPtr dstDevice, IntPtr dstContext,
            IntPtr srcDevice, IntPtr srcContext,
            UIntPtr byteCount, IntPtr hStream);

        // CUstream flags
        public const uint CU_STREAM_DEFAULT = 0;
        public const uint CU_STREAM_NON_BLOCKING = 1;

        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
        public const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
        public const int CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16;
    }
}
