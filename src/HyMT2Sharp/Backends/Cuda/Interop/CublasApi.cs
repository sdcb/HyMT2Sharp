using System;
using System.Runtime.InteropServices;

namespace Sdcb.HyMT2Sharp.Backends.Cuda.Interop
{
    internal static partial class CublasApi
    {
        private const string LibName = "cublas";

        [LibraryImport(LibName, EntryPoint = "cublasCreate_v2")]
        public static partial int cublasCreate(out IntPtr handle);

        [LibraryImport(LibName, EntryPoint = "cublasDestroy_v2")]
        public static partial int cublasDestroy(IntPtr handle);

        [LibraryImport(LibName, EntryPoint = "cublasSetStream_v2")]
        public static partial int cublasSetStream(IntPtr handle, IntPtr stream);

        [LibraryImport(LibName)]
        public static partial int cublasSetMathMode(IntPtr handle, int mode);

        [LibraryImport(LibName, EntryPoint = "cublasSgemm_v2")]
        public static partial int cublasSgemm(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            IntPtr b,
            int ldb,
            ref float beta,
            IntPtr c,
            int ldc);

        [LibraryImport(LibName)]
        public static partial int cublasSgemmStridedBatched(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int lda,
            long strideA,
            IntPtr b,
            int ldb,
            long strideB,
            ref float beta,
            IntPtr c,
            int ldc,
            long strideC,
            int batchCount);

        [LibraryImport(LibName)]
        public static partial int cublasGemmEx(
            IntPtr handle,
            int transa,
            int transb,
            int m,
            int n,
            int k,
            ref float alpha,
            IntPtr a,
            int aType,
            int lda,
            IntPtr b,
            int bType,
            int ldb,
            ref float beta,
            IntPtr c,
            int cType,
            int ldc,
            int computeType,
            int algo);

        public const int CUBLAS_OP_N = 0;
        public const int CUBLAS_OP_T = 1;
        public const int CUBLAS_TENSOR_OP_MATH = 1;
        public const int CUDA_R_16F = 2;
        public const int CUDA_R_32F = 0;
        public const int CUDA_R_16BF = 14;
        public const int CUBLAS_COMPUTE_32F = 68;
        public const int CUBLAS_GEMM_DEFAULT = -1;
    }
}
