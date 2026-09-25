using System;

namespace Sdcb.HyMT2Sharp.Backends.Cuda.Interop
{
    public sealed class CudaException : Exception
    {
        public CudaException(int errorCode, string message)
            : base($"CUDA error {errorCode}: {message}")
        {
            ErrorCode = errorCode;
        }

        public int ErrorCode { get; }
    }
}
