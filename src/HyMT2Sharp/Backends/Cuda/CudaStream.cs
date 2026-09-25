using System;
using System.Threading;
using Sdcb.HyMT2Sharp.Backends.Cuda.Interop;

namespace Sdcb.HyMT2Sharp.Backends.Cuda
{
    public sealed class CudaStream : IDisposable
    {
        private IntPtr stream;

        private CudaStream(IntPtr stream)
        {
            this.stream = stream;
        }

        public IntPtr Handle => stream;

        public static CudaStream Create()
        {
            CudaDriverApi.cuStreamCreate(out IntPtr stream, 0).ThrowOnError();
            return new CudaStream(stream);
        }

        public void Synchronize()
        {
            if (stream == IntPtr.Zero)
                throw new ObjectDisposedException(nameof(CudaStream));

            CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
        }

        public void Dispose()
        {
            IntPtr handle = Interlocked.Exchange(ref stream, IntPtr.Zero);
            if (handle != IntPtr.Zero)
                CudaDriverApi.cuStreamDestroy(handle);
        }
    }
}
