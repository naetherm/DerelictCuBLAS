
module derelict.cublas;

import derelict.util.loader;
import derelict.cuda.runtimeapi : cuComplex, cuDoubleComplex, cudaStream_t, cudaDataType, libraryPropertyType;


private
{
  import derelict.util.system;

  static if(Derelict_OS_Windows)
    // Don't know, please check!
    enum libNames = "cublas32_1000.dll,cublas64_1000.dll";
  else static if (Derelict_OS_Mac)
    // Don't know, please check!
    enum libNames = "libcublas.dylib,/usr/local/lib/libcublas.dylib";
  else static if (Derelict_OS_Linux)
  {
    version(X86)
      enum libNames = "libcublas.so,libcublas.so.10.0,/opt/cuda/lib/libcublas.so";
    else version(X86_64)
      enum libNames = "libcublas.so,libcublas.so.10.0,/opt/cuda/lib64/libcublas.so,/usr/lib/x86_64-linux-gnu/libcublas.so.10.0";
    else
      static assert(0, "Need to implement CUDA libNames for this arch.");
  }
  else
    static assert(0, "Need to implement CUDA libNames for this operating system.");
}



/* CUBLAS status type returns */
alias cublasStatus_t = int;
enum : cublasStatus_t {
  CUBLAS_STATUS_SUCCESS         =0,
  CUBLAS_STATUS_NOT_INITIALIZED =1,
  CUBLAS_STATUS_ALLOC_FAILED    =3,
  CUBLAS_STATUS_INVALID_VALUE   =7,
  CUBLAS_STATUS_ARCH_MISMATCH   =8,
  CUBLAS_STATUS_MAPPING_ERROR   =11,
  CUBLAS_STATUS_EXECUTION_FAILED=13,
  CUBLAS_STATUS_INTERNAL_ERROR  =14,
  CUBLAS_STATUS_NOT_SUPPORTED   =15,
  CUBLAS_STATUS_LICENSE_ERROR   =16
}


alias cublasFillMode_t = int;
enum : cublasFillMode_t {
  CUBLAS_FILL_MODE_LOWER=0,
  CUBLAS_FILL_MODE_UPPER=1
}

alias cublasDiagType_t = int;
enum : cublasDiagType_t {
  CUBLAS_DIAG_NON_UNIT=0,
  CUBLAS_DIAG_UNIT=1
}

alias cublasSideMode_t = int;
enum : cublasSideMode_t {
  CUBLAS_SIDE_LEFT =0,
  CUBLAS_SIDE_RIGHT=1
}


alias cublasOperation_t = int;
enum : cublasOperation_t {
  CUBLAS_OP_N=0,
  CUBLAS_OP_T=1,
  CUBLAS_OP_C=2
}


alias cublasPointerMode_t = int;
enum : cublasPointerMode_t {
  CUBLAS_POINTER_MODE_HOST   = 0,
  CUBLAS_POINTER_MODE_DEVICE = 1
}

alias cublasAtomicsMode_t = int;
enum : cublasAtomicsMode_t {
  CUBLAS_ATOMICS_NOT_ALLOWED   = 0,
  CUBLAS_ATOMICS_ALLOWED       = 1
}

/*For different GEMM algorithm */
alias cublasGemmAlgo_t = int;
enum : cublasGemmAlgo_t {
  CUBLAS_GEMM_DFALT               = -1,
  CUBLAS_GEMM_DEFAULT             = -1,
  CUBLAS_GEMM_ALGO0               =  0,
  CUBLAS_GEMM_ALGO1               =  1,
  CUBLAS_GEMM_ALGO2               =  2,
  CUBLAS_GEMM_ALGO3               =  3,
  CUBLAS_GEMM_ALGO4               =  4,
  CUBLAS_GEMM_ALGO5               =  5,
  CUBLAS_GEMM_ALGO6               =  6,
  CUBLAS_GEMM_ALGO7               =  7,
  CUBLAS_GEMM_ALGO8               =  8,
  CUBLAS_GEMM_ALGO9               =  9,
  CUBLAS_GEMM_ALGO10              =  10,
  CUBLAS_GEMM_ALGO11              =  11,
  CUBLAS_GEMM_ALGO12              =  12,
  CUBLAS_GEMM_ALGO13              =  13,
  CUBLAS_GEMM_ALGO14              =  14,
  CUBLAS_GEMM_ALGO15              =  15,
  CUBLAS_GEMM_ALGO16              =  16,
  CUBLAS_GEMM_ALGO17              =  17,
  CUBLAS_GEMM_ALGO18              =  18, //sliced 32x32
  CUBLAS_GEMM_ALGO19              =  19, //sliced 64x32
  CUBLAS_GEMM_ALGO20              =  20, //sliced 128x32
  CUBLAS_GEMM_ALGO21              =  21, //sliced 32x32  -splitK
  CUBLAS_GEMM_ALGO22              =  22, //sliced 64x32  -splitK
  CUBLAS_GEMM_ALGO23              =  23, //sliced 128x32 -splitK
  CUBLAS_GEMM_DEFAULT_TENSOR_OP   =  99,
  CUBLAS_GEMM_DFALT_TENSOR_OP     =  99,
  CUBLAS_GEMM_ALGO0_TENSOR_OP     =  100,
  CUBLAS_GEMM_ALGO1_TENSOR_OP     =  101,
  CUBLAS_GEMM_ALGO2_TENSOR_OP     =  102,
  CUBLAS_GEMM_ALGO3_TENSOR_OP     =  103,
  CUBLAS_GEMM_ALGO4_TENSOR_OP     =  104,
  CUBLAS_GEMM_ALGO5_TENSOR_OP     =  105,
  CUBLAS_GEMM_ALGO6_TENSOR_OP     =  106,
  CUBLAS_GEMM_ALGO7_TENSOR_OP     =  107,
  CUBLAS_GEMM_ALGO8_TENSOR_OP     =  108,
  CUBLAS_GEMM_ALGO9_TENSOR_OP     =  109,
  CUBLAS_GEMM_ALGO10_TENSOR_OP     =  110,
  CUBLAS_GEMM_ALGO11_TENSOR_OP     =  111,
  CUBLAS_GEMM_ALGO12_TENSOR_OP     =  112,
  CUBLAS_GEMM_ALGO13_TENSOR_OP     =  113,
  CUBLAS_GEMM_ALGO14_TENSOR_OP     =  114,
  CUBLAS_GEMM_ALGO15_TENSOR_OP     =  115
}

/*Enum for default math mode/tensor operation*/
alias cublasMath_t = int;
enum : cublasMath_t {
  CUBLAS_DEFAULT_MATH = 0,
  CUBLAS_TENSOR_OP_MATH = 1
}

/* For backward compatibility purposes */
alias cublasDataType_t = cudaDataType;

/* Opaque structure holding CUBLAS library context */
struct cublasContext;
alias cublasHandle_t = cublasContext*;


extern(System) nothrow {
  alias cublasLogCallback = void function(const char *msg);
}


extern(System) @nogc nothrow {
  alias da_cublasCreate_v2 = cublasStatus_t function(cublasHandle_t *handle);
  alias da_cublasDestroy_v2 = cublasStatus_t function(cublasHandle_t handle);
  alias da_cublasGetVersion_v2 = cublasStatus_t function(cublasHandle_t handle, int *pVersion);
  alias da_cublasGetProperty = cublasStatus_t function(libraryPropertyType type, int *value);
  alias da_cublasSetStream_v2 = cublasStatus_t function(cublasHandle_t handle, cudaStream_t streamId);
  alias da_cublasGetStream_v2 = cublasStatus_t function(cublasHandle_t handle, cudaStream_t *streamId);
  alias da_cublasGetPointerMode_v2 = cublasStatus_t function(cublasHandle_t handle, cublasPointerMode_t *mode);
  alias da_cublasSetPointerMode_v2 = cublasStatus_t function(cublasHandle_t handle, cublasPointerMode_t mode);
  alias da_cublasGetAtomicsMode = cublasStatus_t function(cublasHandle_t handle, cublasAtomicsMode_t *mode);
  alias da_cublasSetAtomicsMode = cublasStatus_t function(cublasHandle_t handle, cublasAtomicsMode_t mode);
  alias da_cublasGetMathMode = cublasStatus_t function(cublasHandle_t handle, cublasMath_t *mode);
  alias da_cublasSetMathMode = cublasStatus_t function(cublasHandle_t handle, cublasMath_t mode);
  alias da_cublasLoggerConfigure = cublasStatus_t function(int logIsOn, int logToStdOut, int logToStdErr, const char* logFileName);
  alias da_cublasSetLoggerCallback = cublasStatus_t function(cublasLogCallback userCallback);
  alias da_cublasGetLoggerCallback = cublasStatus_t function(cublasLogCallback* userCallback);
  alias da_cublasSetVector = cublasStatus_t function(int n, int elemSize, const void *x,int incx, void *devicePtr, int incy);
  alias da_cublasGetVector = cublasStatus_t function(int n, int elemSize, const void *x,int incx, void *y, int incy);
  alias da_cublasSetMatrix = cublasStatus_t function(int rows, int cols, int elemSize,const void *A, int lda, void *B,int ldb);
  alias da_cublasGetMatrix = cublasStatus_t function(int rows, int cols, int elemSize,const void *A, int lda, void *B,int ldb);
  alias da_cublasSetVectorAsync = cublasStatus_t function(int n, int elemSize,const void *hostPtr, int incx,void *devicePtr, int incy,cudaStream_t stream);
  alias da_cublasGetVectorAsync = cublasStatus_t function(int n, int elemSize,const void *devicePtr, int incx,void *hostPtr, int incy,cudaStream_t stream);
  alias da_cublasSetMatrixAsync = cublasStatus_t function(int rows, int cols, int elemSize,const void *A, int lda, void *B,int ldb, cudaStream_t stream);
  alias da_cublasGetMatrixAsync = cublasStatus_t function(int rows, int cols, int elemSize,const void *A, int lda, void *B,int ldb, cudaStream_t stream);
  alias da_cublasXerbla = void function(const char *srName, int info);
  alias da_cublasNrm2Ex = cublasStatus_t function(cublasHandle_t handle,int n,const void *x,cudaDataType xType,int incx,void *result,cudaDataType resultType,cudaDataType executionType);
  alias da_cublasSnrm2_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,float *result);
  alias da_cublasDnrm2_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,double *result);
  alias da_cublasScnrm2_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,float *result);
  alias da_cublasDznrm2_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,double *result);
  alias da_cublasDotEx = cublasStatus_t function(cublasHandle_t handle,int n,const void *x,cudaDataType xType,int incx,const void *y,cudaDataType yType,int incy,void *result,cudaDataType resultType,cudaDataType executionType);
  alias da_cublasDotcEx = cublasStatus_t function(cublasHandle_t handle,int n,const void *x,cudaDataType xType,int incx,const void *y,cudaDataType yType,int incy,void *result,cudaDataType resultType,cudaDataType executionType);
  alias da_cublasSdot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,const float *y,int incy,float *result);
  alias da_cublasDdot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,const double *y,int incy,double *result);
  alias da_cublasCdotu_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *result);
  alias da_cublasCdotc_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *result);
  alias da_cublasZdotu_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *result);
  alias da_cublasZdotc_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *result);
  alias da_cublasScalEx = cublasStatus_t function(cublasHandle_t handle,int n,const void *alpha, cudaDataType alphaType,void *x,cudaDataType xType,int incx,cudaDataType executionType);
  alias da_cublasSscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *alpha, float *x,int incx);
  alias da_cublasDscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *alpha, double *x,int incx);
  alias da_cublasCscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *alpha,cuComplex *x,int incx);
  alias da_cublasCsscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *alpha,cuComplex *x,int incx);
  alias da_cublasZscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *alpha,cuDoubleComplex *x,int incx);
  alias da_cublasZdscal_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *alpha,cuDoubleComplex *x,int incx);
  alias da_cublasAxpyEx = cublasStatus_t function(cublasHandle_t handle,int n,const void *alpha,cudaDataType alphaType,const void *x,cudaDataType xType,int incx,void *y,cudaDataType yType,int incy,cudaDataType executiontype);
  alias da_cublasSaxpy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *alpha,const float *x,int incx,float *y,int incy);
  alias da_cublasDaxpy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *alpha,const double *x,int incx,double *y,int incy);
  alias da_cublasCaxpy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *alpha,const cuComplex *x,int incx,cuComplex *y,int incy);
  alias da_cublasZaxpy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *x,int incx,cuDoubleComplex *y,int incy);
  alias da_cublasScopy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,float *y,int incy);
  alias da_cublasDcopy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,double *y,int incy);
  alias da_cublasCcopy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,cuComplex *y,int incy);
  alias da_cublasZcopy_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,cuDoubleComplex *y,int incy);
  alias da_cublasSswap_v2 = cublasStatus_t function(cublasHandle_t handle,int n,float *x,int incx,float *y,int incy);
  alias da_cublasDswap_v2 = cublasStatus_t function(cublasHandle_t handle,int n,double *x,int incx,double *y,int incy);
  alias da_cublasCswap_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuComplex *x,int incx,cuComplex *y,int incy);
  alias da_cublasZswap_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuDoubleComplex *x,int incx,cuDoubleComplex *y,int incy);
  alias da_cublasIsamax_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,int *result);
  alias da_cublasIdamax_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,int *result);
  alias da_cublasIcamax_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,int *result);
  alias da_cublasIzamax_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,int *result);
  alias da_cublasIsamin_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,int *result);
  alias da_cublasIdamin_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,int *result);
  alias da_cublasIcamin_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,int *result);
  alias da_cublasIzamin_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,int *result);
  alias da_cublasSasum_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const float *x,int incx,float *result);
  alias da_cublasDasum_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const double *x,int incx,double *result);
  alias da_cublasScasum_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *x,int incx,float *result);
  alias da_cublasDzasum_v2 = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *x,int incx,double *result);
  alias da_cublasSrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,float *x,int incx,float *y,int incy,const float *c, const float *s);
  alias da_cublasDrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,double *x,int incx,double *y,int incy,const double *c, const double *s);
  alias da_cublasCrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuComplex *x,int incx,cuComplex *y,int incy,const float *c,     const cuComplex *s);
  alias da_cublasCsrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuComplex *x,int incx,cuComplex *y,int incy,const float *c, const float *s);
  alias da_cublasZrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuDoubleComplex *x,int incx,cuDoubleComplex *y,int incy,const double *c,           const cuDoubleComplex *s);
  alias da_cublasZdrot_v2 = cublasStatus_t function(cublasHandle_t handle,int n,cuDoubleComplex *x,int incx,cuDoubleComplex *y,int incy,const double *c, const double *s);
  alias da_cublasSrotg_v2 = cublasStatus_t function(cublasHandle_t handle,float *a,  float *b,  float *c,  float *s);
  alias da_cublasDrotg_v2 = cublasStatus_t function(cublasHandle_t handle,double *a, double *b, double *c, double *s);
  alias da_cublasCrotg_v2 = cublasStatus_t function(cublasHandle_t handle,cuComplex *a, cuComplex *b, float *c,     cuComplex *s);
  alias da_cublasZrotg_v2 = cublasStatus_t function(cublasHandle_t handle,cuDoubleComplex *a, cuDoubleComplex *b, double *c,          cuDoubleComplex *s);
  alias da_cublasSrotm_v2 = cublasStatus_t function(cublasHandle_t handle,int n,float *x,int incx,float *y,int incy,const float* param);
  alias da_cublasDrotm_v2 = cublasStatus_t function(cublasHandle_t handle,int n,double *x,int incx,double *y,int incy,const double* param);
  alias da_cublasSrotmg_v2 = cublasStatus_t function(cublasHandle_t handle,float *d1,       float *d2,       float *x1,       const float *y1, float *param);
  alias da_cublasDrotmg_v2 = cublasStatus_t function(cublasHandle_t handle,double *d1,       double *d2,       double *x1,       const double *y1, double *param);
  alias da_cublasSgemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,const float *alpha,const float *A,int lda,const float *x,int incx,const float *beta, float *y,int incy);
  alias da_cublasDgemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,const double *alpha,const double *A,int lda,const double *x,int incx,const double *beta,double *y,int incy);
  alias da_cublasCgemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZgemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,cuDoubleComplex *y,int incy);
  alias da_cublasSgbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,int kl,int ku,const float *alpha,const float *A,int lda,const float *x,int incx,const float *beta,float *y,int incy);
  alias da_cublasDgbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,int kl,int ku,const double *alpha,const double *A,int lda,const double *x,int incx,const double *beta,double *y,int incy);
  alias da_cublasCgbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,int kl,int ku,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZgbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t trans,int m,int n,int kl,int ku,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,cuDoubleComplex *y,int incy);
  alias da_cublasStrmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const float *A,int lda,float *x,int incx);
  alias da_cublasDtrmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const double *A,int lda,double *x,int incx);
  alias da_cublasCtrmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuComplex *A,int lda,cuComplex *x,int incx);
  alias da_cublasZtrmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuDoubleComplex *A,int lda,cuDoubleComplex *x,int incx);
  alias da_cublasStbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const float *A,int lda,float *x,int incx);
  alias da_cublasDtbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const double *A,int lda,double *x,int incx);
  alias da_cublasCtbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const cuComplex *A,int lda,cuComplex *x,int incx);
  alias da_cublasZtbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const cuDoubleComplex *A,int lda,cuDoubleComplex *x,int incx);
  alias da_cublasStpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const float *AP,float *x,int incx);
  alias da_cublasDtpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const double *AP,double *x,int incx);
  alias da_cublasCtpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuComplex *AP,cuComplex *x,int incx);
  alias da_cublasZtpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuDoubleComplex *AP,cuDoubleComplex *x,int incx);
  alias da_cublasStrsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const float *A,int lda,float *x,int incx);
  alias da_cublasDtrsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const double *A,int lda,double *x,int incx);
  alias da_cublasCtrsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuComplex *A,int lda,cuComplex *x,int incx);
  alias da_cublasZtrsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuDoubleComplex *A,int lda,cuDoubleComplex *x,int incx);
  alias da_cublasStpsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const float *AP,float *x,int incx);
  alias da_cublasDtpsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const double *AP,double *x,int incx);
  alias da_cublasCtpsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuComplex *AP,cuComplex *x,int incx);
  alias da_cublasZtpsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,const cuDoubleComplex *AP,cuDoubleComplex *x,int incx);
  alias da_cublasStbsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const float *A,int lda,float *x,int incx);
  alias da_cublasDtbsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const double *A,int lda,double *x,int incx);
  alias da_cublasCtbsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const cuComplex *A,int lda,cuComplex *x,int incx);
  alias da_cublasZtbsv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int n,int k,const cuDoubleComplex *A,int lda,cuDoubleComplex *x,int incx);
  alias da_cublasSsymv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const float *A,int lda,const float *x,int incx,const float *beta,float *y,int incy);
  alias da_cublasDsymv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const double *A,int lda,const double *x,int incx,const double *beta,double *y,int incy);
  alias da_cublasCsymv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZsymv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha, const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,  cuDoubleComplex *y,int incy);
  alias da_cublasChemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZhemv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha, const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,  cuDoubleComplex *y,int incy);
  alias da_cublasSsbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,int k,const float *alpha,  const float *A,int lda,const float *x,int incx,const float *beta, float *y,int incy);
  alias da_cublasDsbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,int k,const double *alpha,  const double *A,int lda,const double *x,int incx,const double *beta,  double *y,int incy);
  alias da_cublasChbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZhbmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,cuDoubleComplex *y,int incy);
  alias da_cublasSspmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha, const float *AP,const float *x,int incx,const float *beta,  float *y,int incy);
  alias da_cublasDspmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const double *AP,const double *x,int incx,const double *beta, double *y,int incy);
  alias da_cublasChpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *alpha,const cuComplex *AP,const cuComplex *x,int incx,const cuComplex *beta,cuComplex *y,int incy);
  alias da_cublasZhpmv_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *AP,const cuDoubleComplex *x,int incx,const cuDoubleComplex *beta,cuDoubleComplex *y,int incy);
  alias da_cublasSger_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const float *alpha,const float *x,int incx,const float *y,int incy,float *A,int lda);
  alias da_cublasDger_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const double *alpha,const double *x,int incx,const double *y,int incy,double *A,int lda);
  alias da_cublasCgeru_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const cuComplex *alpha,const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *A,int lda);
  alias da_cublasCgerc_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const cuComplex *alpha,const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *A,int lda);
  alias da_cublasZgeru_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *A,int lda);
  alias da_cublasZgerc_v2 = cublasStatus_t function(cublasHandle_t handle,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *A,int lda);
  alias da_cublasSsyr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const float *x,int incx,float *A,int lda);
  alias da_cublasDsyr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const double *x,int incx,double *A,int lda);
  alias da_cublasCsyr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *alpha,const cuComplex *x,int incx,cuComplex *A,int lda);
  alias da_cublasZsyr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *x,int incx,cuDoubleComplex *A,int lda);
  alias da_cublasCher_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const cuComplex *x,int incx,cuComplex *A,int lda);
  alias da_cublasZher_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const cuDoubleComplex *x,int incx,cuDoubleComplex *A,int lda);
  alias da_cublasSspr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const float *x,int incx,float *AP);
  alias da_cublasDspr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const double *x,int incx,double *AP);
  alias da_cublasChpr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const cuComplex *x,int incx,cuComplex *AP);
  alias da_cublasZhpr_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const cuDoubleComplex *x,int incx,cuDoubleComplex *AP);
  alias da_cublasSsyr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha,const float *x,int incx,const float *y,int incy,float *A,int lda);
  alias da_cublasDsyr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha,const double *x,int incx,const double *y,int incy,double *A,int lda);
  alias da_cublasCsyr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo, int n,const cuComplex *alpha, const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *A,int lda);
  alias da_cublasZsyr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha, const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *A,int lda);
  alias da_cublasCher2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo, int n,const cuComplex *alpha, const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *A,int lda);
  alias da_cublasZher2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha, const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *A,int lda);
  alias da_cublasSspr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *alpha, const float *x,int incx,const float *y,int incy,float *AP);
  alias da_cublasDspr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *alpha, const double *x,int incx,const double *y,int incy,double *AP);
  alias da_cublasChpr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *alpha,const cuComplex *x,int incx,const cuComplex *y,int incy,cuComplex *AP);
  alias da_cublasZhpr2_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *x,int incx,const cuDoubleComplex *y,int incy,cuDoubleComplex *AP);
  alias da_cublasSgemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const float *alpha,const float *A,int lda,const float *B,int ldb,const float *beta,float *C,int ldc);
  alias da_cublasDgemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const double *alpha,const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc);
  alias da_cublasCgemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasCgemm3m = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasCgemm3mEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa, cublasOperation_t transb,int m, int n, int k,const cuComplex *alpha,const void *A,cudaDataType Atype,int lda,const void  *B,cudaDataType Btype,int ldb,const cuComplex *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasZgemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasZgemm3m = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  //alias da_cublasHgemm = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const __half *alpha,const __half *A,int lda,const __half *B,int ldb,const __half *beta,__half *C,int ldc);
  alias da_cublasSgemmEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const float *alpha,const void *A,cudaDataType Atype,int lda,const void *B,cudaDataType Btype,int ldb,const float *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasGemmEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const void *alpha,const void *A,cudaDataType Atype,int lda,const void *B,cudaDataType Btype,int ldb,const void *beta,void *C,cudaDataType Ctype,int ldc,cudaDataType computeType,cublasGemmAlgo_t algo);
  alias da_cublasCgemmEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa, cublasOperation_t transb,int m, int n, int k,const cuComplex *alpha,const void *A,cudaDataType Atype,int lda,const void *B,cudaDataType Btype,int ldb,const cuComplex *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasUint8gemmBias = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa, cublasOperation_t transb, cublasOperation_t transc,int m, int n, int k,const dchar *A, int A_bias, int lda,const dchar *B, int B_bias, int ldb,dchar *C, int C_bias, int ldc,int C_mult, int C_shift);
  alias da_cublasSsyrk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha,const float *A,int lda,const float *beta,float *C,int ldc);
  alias da_cublasDsyrk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const double *alpha, const double *A,int lda,const double *beta, double *C,int ldc);
  alias da_cublasCsyrk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasZsyrk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasCsyrkEx = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const void *A,cudaDataType Atype,int lda,const cuComplex *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasCsyrk3mEx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const void *A,cudaDataType Atype,int lda,const cuComplex *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasCherk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha, const cuComplex *A,int lda,const float *beta,  cuComplex *C,int ldc);
  alias da_cublasZherk_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const double *alpha, const cuDoubleComplex *A,int lda,const double *beta, cuDoubleComplex *C,int ldc);
  alias da_cublasCherkEx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha, const void *A,cudaDataType Atype,int lda,const float *beta,  void *C,cudaDataType Ctype,int ldc);
  alias da_cublasCherk3mEx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha,const void *A, cudaDataType Atype,int lda,const float *beta,void *C,cudaDataType Ctype,int ldc);
  alias da_cublasSsyr2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha,const float *A,int lda,const float *B,int ldb,const float *beta,float *C,int ldc);
  alias da_cublasDsyr2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const double *alpha,const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc);
  alias da_cublasCsyr2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasZsyr2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuDoubleComplex *alpha, const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta, cuDoubleComplex *C,int ldc);
  alias da_cublasCher2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const float *beta,  cuComplex *C,int ldc);
  alias da_cublasZher2k_v2 = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const double *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasSsyrkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const float *alpha,const float *A,int lda,const float *B,int ldb,const float *beta,float *C,int ldc);
  alias da_cublasDsyrkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const double *alpha,const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc);
  alias da_cublasCsyrkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasZsyrkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasCherkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const float *beta,cuComplex *C,int ldc);
  alias da_cublasZherkx = cublasStatus_t function(cublasHandle_t handle,cublasFillMode_t uplo,cublasOperation_t trans,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const double *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasSsymm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const float *alpha,const float *A,int lda,const float *B,int ldb,const float *beta,float *C,int ldc);
  alias da_cublasDsymm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const double *alpha,const double *A,int lda,const double *B,int ldb,const double *beta,double *C,int ldc);
  alias da_cublasCsymm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasZsymm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasChemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,const cuComplex *beta,cuComplex *C,int ldc);
  alias da_cublasZhemm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *C,int ldc);
  alias da_cublasStrsm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const float *alpha,const float *A,int lda,float *B,int ldb);
  alias da_cublasDtrsm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const double *alpha,const double *A,int lda,double *B,int ldb);
  alias da_cublasCtrsm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,cuComplex *B,int ldb);
  alias da_cublasZtrsm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,cuDoubleComplex *B,int ldb);
  alias da_cublasStrmm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const float *alpha,const float *A,int lda,const float *B,int ldb,float *C,int ldc);
  alias da_cublasDtrmm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const double *alpha,const double *A,int lda,const double *B,int ldb,double *C,int ldc);
  alias da_cublasCtrmm_v2 = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *B,int ldb,cuComplex *C,int ldc);
  alias da_cublasZtrmm_v2 = cublasStatus_t function(cublasHandle_t handle, cublasSideMode_t side,cublasFillMode_t uplo,cublasOperation_t trans,cublasDiagType_t diag,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *B,int ldb,cuDoubleComplex *C,int ldc);
  //alias da_cublasHgemmBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const __half *alpha, const __half *const Aarray[],int lda,const __half *const Barray[],int ldb,const __half *beta,  __half *const Carray[],int ldc,int batchCount);
  alias da_cublasSgemmBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const float *alpha, const float *[] Aarray,int lda,const float *[] Barray,int ldb,const float *beta,  float *[] Carray,int ldc,int batchCount);
  alias da_cublasDgemmBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const double *alpha, const double *[] Aarray,int lda,const double *[] Barray,int ldb,const double *beta, double *[] Carray,int ldc,int batchCount);
  alias da_cublasCgemmBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha,const cuComplex *[] Aarray,int lda,const cuComplex *[] Barray,int ldb,const cuComplex *beta,cuComplex *[] Carray,int ldc,int batchCount);
  alias da_cublasCgemm3mBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha,const cuComplex *[] Aarray,int lda,const cuComplex *[] Barray,int ldb,const cuComplex *beta,cuComplex *[] Carray,int ldc,int batchCount);
  alias da_cublasZgemmBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuDoubleComplex *alpha,const cuDoubleComplex *[] Aarray,int lda,const cuDoubleComplex *[] Barray,int ldb,const cuDoubleComplex *beta,cuDoubleComplex *[] Carray,int ldc,int batchCount);
  alias da_cublasGemmBatchedEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const void *alpha,const void *[] Aarray,cudaDataType Atype,int lda,const void *[] Barray,cudaDataType Btype,int ldb,const void *beta,void *[] Carray,cudaDataType Ctype,int ldc,int batchCount,cudaDataType computeType,cublasGemmAlgo_t algo);
  alias da_cublasGemmStridedBatchedEx = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const void *alpha, const void *A,cudaDataType Atype,int lda,long strideA,   const void *B,cudaDataType Btype,int ldb,long strideB,const void *beta,  void *C,cudaDataType Ctype,int ldc,long strideC,int batchCount,cudaDataType computeType,cublasGemmAlgo_t algo);
  alias da_cublasSgemmStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const float *alpha, const float *A,int lda,long strideA,   const float *B,int ldb,long strideB,const float *beta,  float *C,int ldc,long strideC,int batchCount);
  alias da_cublasDgemmStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const double *alpha, const double *A,int lda,long strideA,   const double *B,int ldb,long strideB,const double *beta,  double *C,int ldc,long strideC,int batchCount);
  alias da_cublasCgemmStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha, const cuComplex *A,int lda,long strideA,   const cuComplex *B,int ldb,long strideB,const cuComplex *beta,  cuComplex *C,int ldc,long strideC,int batchCount);
  alias da_cublasCgemm3mStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuComplex *alpha, const cuComplex *A,int lda,long strideA,   const cuComplex *B,int ldb,long strideB,const cuComplex *beta,  cuComplex *C,int ldc,long strideC,int batchCount);
  alias da_cublasZgemmStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const cuDoubleComplex *alpha, const cuDoubleComplex *A,int lda,long strideA,   const cuDoubleComplex *B,int ldb,long strideB,const cuDoubleComplex *beta,   /* host or device poi */cuDoubleComplex *C,int ldc,long strideC,int batchCount);
  //alias da_cublasHgemmStridedBatched = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,int k,const __half *alpha, const __half *A,int lda,long strideA,   const __half *B,int ldb,long strideB,const __half *beta,  __half *C,int ldc,long strideC,int batchCount);
  alias da_cublasSgeam = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,const float *alpha,const float *A,int lda,const float *beta ,const float *B,int ldb,float *C,int ldc);
  alias da_cublasDgeam = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m, int n,const double *alpha,const double *A,int lda,const double *beta,const double *B,int ldb,double *C,int ldc);
  alias da_cublasCgeam = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,const cuComplex *alpha,const cuComplex *A,int lda,const cuComplex *beta,const cuComplex *B,int ldb,cuComplex *C,int ldc);
  alias da_cublasZgeam = cublasStatus_t function(cublasHandle_t handle,cublasOperation_t transa,cublasOperation_t transb,int m,int n,const cuDoubleComplex *alpha,const cuDoubleComplex *A,int lda,const cuDoubleComplex *beta,const cuDoubleComplex *B,int ldb,cuDoubleComplex *C,int ldc);
  alias da_cublasSgetrfBatched = cublasStatus_t function(cublasHandle_t handle,int n,float *[] A,                int lda,int *P,                          int *info,                       int batchSize);
  alias da_cublasDgetrfBatched = cublasStatus_t function(cublasHandle_t handle,int n,double *[] A,               int lda,int *P,                          int *info,                       int batchSize);
  alias da_cublasCgetrfBatched = cublasStatus_t function(cublasHandle_t handle,int n,cuComplex *[] A,           int lda,int *P,                         int *info,                      int batchSize);
  alias da_cublasZgetrfBatched = cublasStatus_t function(cublasHandle_t handle,int n,cuDoubleComplex *[] A,     int lda,int *P,                         int *info,                      int batchSize);
  alias da_cublasSgetriBatched = cublasStatus_t function(cublasHandle_t handle,int n,const float *[] A,         int lda,const int *P,                   float *[] C,               int ldc,int *info,int batchSize);
  alias da_cublasDgetriBatched = cublasStatus_t function(cublasHandle_t handle,int n,const double *[] A,        int lda,const int *P,                   double *[] C,              int ldc,int *info,int batchSize);
  alias da_cublasCgetriBatched = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *[] A,     int lda,const int *P,                   cuComplex *[] C,           int ldc,int *info,int batchSize);
  alias da_cublasZgetriBatched = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *[] A, int lda,const int *P,                     cuDoubleComplex *[] C,       int ldc,int *info,int batchSize);
  alias da_cublasSgetrsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int n,int nrhs,const float *[] Aarray,int lda,const int *devIpiv,float *[] Barray,int ldb,int *info,int batchSize);
  alias da_cublasDgetrsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int n,int nrhs,const double *[] Aarray,int lda,const int *devIpiv,double *[] Barray,int ldb,int *info,int batchSize);
  alias da_cublasCgetrsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int n,int nrhs,const cuComplex *[] Aarray,int lda,const int *devIpiv,cuComplex *[] Barray,int ldb,int *info,int batchSize);
  alias da_cublasZgetrsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int n,int nrhs,const cuDoubleComplex *[] Aarray,int lda,const int *devIpiv,cuDoubleComplex *[] Barray,int ldb,int *info,int batchSize);
  alias da_cublasStrsmBatched = cublasStatus_t function( cublasHandle_t    handle,cublasSideMode_t  side,cublasFillMode_t  uplo,cublasOperation_t trans,cublasDiagType_t  diag,int m,int n,const float *alpha,           const float *[] A,int lda,float *[] B,int ldb,int batchCount);
  alias da_cublasDtrsmBatched = cublasStatus_t function( cublasHandle_t    handle,cublasSideMode_t  side,cublasFillMode_t  uplo,cublasOperation_t trans,cublasDiagType_t  diag,int m,int n,const double *alpha,          const double *[] A,int lda,double *[] B,int ldb,int batchCount);
  alias da_cublasCtrsmBatched = cublasStatus_t function( cublasHandle_t    handle,cublasSideMode_t  side,cublasFillMode_t  uplo,cublasOperation_t trans,cublasDiagType_t  diag,int m,int n,const cuComplex *alpha,       const cuComplex *[] A,int lda,cuComplex *[] B,int ldb,int batchCount);
  alias da_cublasZtrsmBatched = cublasStatus_t function( cublasHandle_t    handle,cublasSideMode_t  side,cublasFillMode_t  uplo,cublasOperation_t trans,cublasDiagType_t  diag,int m,int n,const cuDoubleComplex *alpha, const cuDoubleComplex *[] A,int lda,cuDoubleComplex *[] B,int ldb,int batchCount);
  alias da_cublasSmatinvBatched = cublasStatus_t function(cublasHandle_t handle,int n,const float *[] A,      int lda,float *[] Ainv,         int lda_inv,int *info,                   int batchSize);
  alias da_cublasDmatinvBatched = cublasStatus_t function(cublasHandle_t handle,int n,const double *[] A,     int lda,double *[] Ainv,        int lda_inv,int *info,                   int batchSize);
  alias da_cublasCmatinvBatched = cublasStatus_t function(cublasHandle_t handle,int n,const cuComplex *[] A,  int lda,cuComplex *[] Ainv,     int lda_inv,int *info,                   int batchSize);
  alias da_cublasZmatinvBatched = cublasStatus_t function(cublasHandle_t handle,int n,const cuDoubleComplex *[] A, int lda,cuDoubleComplex *[] Ainv,    int lda_inv,int *info,                        int batchSize);
  alias da_cublasSgeqrfBatched = cublasStatus_t function( cublasHandle_t handle,int m,int n,float *[] Aarray,      int lda,float *[] TauArray,    int *info,int batchSize);
  alias da_cublasDgeqrfBatched = cublasStatus_t function( cublasHandle_t handle,int m,int n,double *[] Aarray,     int lda,double *[] TauArray,   int *info,int batchSize);
  alias da_cublasCgeqrfBatched = cublasStatus_t function( cublasHandle_t handle,int m,int n,cuComplex *[] Aarray,          int lda,cuComplex *[] TauArray,        int *info,int batchSize);
  alias da_cublasZgeqrfBatched = cublasStatus_t function( cublasHandle_t handle,int m,int n,cuDoubleComplex *[] Aarray,    int lda,cuDoubleComplex *[] TauArray,  int *info,int batchSize);
  alias da_cublasSgelsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int m,int n,int nrhs,float *[] Aarray,      int lda,float *[] Carray,      int ldc,int *info,int *devInfoArray,          int batchSize );
  alias da_cublasDgelsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int m,int n,int nrhs,double *[] Aarray,     int lda,double *[] Carray,     int ldc,int *info,int *devInfoArray,          int batchSize);
  alias da_cublasCgelsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int m,int n,int nrhs,cuComplex *[] Aarray,  int lda,cuComplex *[] Carray,  int ldc,int *info,int *devInfoArray,int batchSize);
  alias da_cublasZgelsBatched = cublasStatus_t function( cublasHandle_t handle,cublasOperation_t trans,int m,int n,int nrhs,cuDoubleComplex *[] Aarray,  int lda,cuDoubleComplex *[] Carray,  int ldc,int *info,int *devInfoArray,int batchSize);
  alias da_cublasSdgmm = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t mode,int m,int n,const float *A,int lda,const float *x,int incx,float *C,int ldc);
  alias da_cublasDdgmm = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t mode,int m,int n,const double *A,int lda,const double *x,int incx,double *C,int ldc);
  alias da_cublasCdgmm = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t mode,int m,int n,const cuComplex *A,int lda,const cuComplex *x,int incx,cuComplex *C,int ldc);
  alias da_cublasZdgmm = cublasStatus_t function(cublasHandle_t handle,cublasSideMode_t mode,int m,int n,const cuDoubleComplex *A,int lda,const cuDoubleComplex *x,int incx,cuDoubleComplex *C,int ldc);
  alias da_cublasStpttr = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *AP,float *A,int lda );
  alias da_cublasDtpttr = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *AP,double *A,int lda );
  alias da_cublasCtpttr = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *AP,cuComplex *A,int lda );
  alias da_cublasZtpttr = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *AP,cuDoubleComplex *A,int lda );
  alias da_cublasStrttp = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const float *A,int lda,float *AP );
  alias da_cublasDtrttp = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const double *A,int lda,double *AP );
  alias da_cublasCtrttp = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuComplex *A,int lda,cuComplex *AP );
  alias da_cublasZtrttp = cublasStatus_t function( cublasHandle_t handle,cublasFillMode_t uplo,int n,const cuDoubleComplex *A,int lda,cuDoubleComplex *AP );
}

__gshared
{
  da_cublasCreate_v2 cublasCreate_v2;
  da_cublasDestroy_v2 cublasDestroy_v2;
  da_cublasGetVersion_v2 cublasGetVersion_v2;
  da_cublasGetProperty cublasGetProperty;
  da_cublasSetStream_v2 cublasSetStream_v2;
  da_cublasGetStream_v2 cublasGetStream_v2;
  da_cublasGetPointerMode_v2 cublasGetPointerMode_v2;
  da_cublasSetPointerMode_v2 cublasSetPointerMode_v2;
  da_cublasGetAtomicsMode cublasGetAtomicsMode;
  da_cublasSetAtomicsMode cublasSetAtomicsMode;
  da_cublasGetMathMode cublasGetMathMode;
  da_cublasSetMathMode cublasSetMathMode;
  da_cublasLoggerConfigure cublasLoggerConfigure;
  da_cublasSetLoggerCallback cublasSetLoggerCallback;
  da_cublasGetLoggerCallback cublasGetLoggerCallback;
  da_cublasSetVector cublasSetVector;
  da_cublasGetVector cublasGetVector;
  da_cublasSetMatrix cublasSetMatrix;
  da_cublasGetMatrix cublasGetMatrix;
  da_cublasSetVectorAsync cublasSetVectorAsync;
  da_cublasGetVectorAsync cublasGetVectorAsync;
  da_cublasSetMatrixAsync cublasSetMatrixAsync;
  da_cublasGetMatrixAsync cublasGetMatrixAsync;
  da_cublasXerbla cublasXerbla;
  da_cublasNrm2Ex cublasNrm2Ex;
  da_cublasSnrm2_v2 cublasSnrm2_v2;
  da_cublasDnrm2_v2 cublasDnrm2_v2;
  da_cublasScnrm2_v2 cublasScnrm2_v2;
  da_cublasDznrm2_v2 cublasDznrm2_v2;
  da_cublasDotEx cublasDotEx;
  da_cublasDotcEx cublasDotcEx;
  da_cublasSdot_v2 cublasSdot_v2;
  da_cublasDdot_v2 cublasDdot_v2;
  da_cublasCdotu_v2 cublasCdotu_v2;
  da_cublasCdotc_v2 cublasCdotc_v2;
  da_cublasZdotu_v2 cublasZdotu_v2;
  da_cublasZdotc_v2 cublasZdotc_v2;
  da_cublasScalEx cublasScalEx;
  da_cublasSscal_v2 cublasSscal_v2;
  da_cublasDscal_v2 cublasDscal_v2;
  da_cublasCscal_v2 cublasCscal_v2;
  da_cublasCsscal_v2 cublasCsscal_v2;
  da_cublasZscal_v2 cublasZscal_v2;
  da_cublasZdscal_v2 cublasZdscal_v2;
  da_cublasAxpyEx cublasAxpyEx;
  da_cublasSaxpy_v2 cublasSaxpy_v2;
  da_cublasDaxpy_v2 cublasDaxpy_v2;
  da_cublasCaxpy_v2 cublasCaxpy_v2;
  da_cublasZaxpy_v2 cublasZaxpy_v2;
  da_cublasScopy_v2 cublasScopy_v2;
  da_cublasDcopy_v2 cublasDcopy_v2;
  da_cublasCcopy_v2 cublasCcopy_v2;
  da_cublasZcopy_v2 cublasZcopy_v2;
  da_cublasSswap_v2 cublasSswap_v2;
  da_cublasDswap_v2 cublasDswap_v2;
  da_cublasCswap_v2 cublasCswap_v2;
  da_cublasZswap_v2 cublasZswap_v2;
  da_cublasIsamax_v2 cublasIsamax_v2;
  da_cublasIdamax_v2 cublasIdamax_v2;
  da_cublasIcamax_v2 cublasIcamax_v2;
  da_cublasIzamax_v2 cublasIzamax_v2;
  da_cublasIsamin_v2 cublasIsamin_v2;
  da_cublasIdamin_v2 cublasIdamin_v2;
  da_cublasIcamin_v2 cublasIcamin_v2;
  da_cublasIzamin_v2 cublasIzamin_v2;
  da_cublasSasum_v2 cublasSasum_v2;
  da_cublasDasum_v2 cublasDasum_v2;
  da_cublasScasum_v2 cublasScasum_v2;
  da_cublasDzasum_v2 cublasDzasum_v2;
  da_cublasSrot_v2 cublasSrot_v2;
  da_cublasDrot_v2 cublasDrot_v2;
  da_cublasCrot_v2 cublasCrot_v2;
  da_cublasCsrot_v2 cublasCsrot_v2;
  da_cublasZrot_v2 cublasZrot_v2;
  da_cublasZdrot_v2 cublasZdrot_v2;
  da_cublasSrotg_v2 cublasSrotg_v2;
  da_cublasDrotg_v2 cublasDrotg_v2;
  da_cublasCrotg_v2 cublasCrotg_v2;
  da_cublasZrotg_v2 cublasZrotg_v2;
  da_cublasSrotm_v2 cublasSrotm_v2;
  da_cublasDrotm_v2 cublasDrotm_v2;
  da_cublasSrotmg_v2 cublasSrotmg_v2;
  da_cublasDrotmg_v2 cublasDrotmg_v2;
  da_cublasSgemv_v2 cublasSgemv_v2;
  da_cublasDgemv_v2 cublasDgemv_v2;
  da_cublasCgemv_v2 cublasCgemv_v2;
  da_cublasZgemv_v2 cublasZgemv_v2;
  da_cublasSgbmv_v2 cublasSgbmv_v2;
  da_cublasDgbmv_v2 cublasDgbmv_v2;
  da_cublasCgbmv_v2 cublasCgbmv_v2;
  da_cublasZgbmv_v2 cublasZgbmv_v2;
  da_cublasStrmv_v2 cublasStrmv_v2;
  da_cublasDtrmv_v2 cublasDtrmv_v2;
  da_cublasCtrmv_v2 cublasCtrmv_v2;
  da_cublasZtrmv_v2 cublasZtrmv_v2;
  da_cublasStbmv_v2 cublasStbmv_v2;
  da_cublasDtbmv_v2 cublasDtbmv_v2;
  da_cublasCtbmv_v2 cublasCtbmv_v2;
  da_cublasZtbmv_v2 cublasZtbmv_v2;
  da_cublasStpmv_v2 cublasStpmv_v2;
  da_cublasDtpmv_v2 cublasDtpmv_v2;
  da_cublasCtpmv_v2 cublasCtpmv_v2;
  da_cublasZtpmv_v2 cublasZtpmv_v2;
  da_cublasStrsv_v2 cublasStrsv_v2;
  da_cublasDtrsv_v2 cublasDtrsv_v2;
  da_cublasCtrsv_v2 cublasCtrsv_v2;
  da_cublasZtrsv_v2 cublasZtrsv_v2;
  da_cublasStpsv_v2 cublasStpsv_v2;
  da_cublasDtpsv_v2 cublasDtpsv_v2;
  da_cublasCtpsv_v2 cublasCtpsv_v2;
  da_cublasZtpsv_v2 cublasZtpsv_v2;
  da_cublasStbsv_v2 cublasStbsv_v2;
  da_cublasDtbsv_v2 cublasDtbsv_v2;
  da_cublasCtbsv_v2 cublasCtbsv_v2;
  da_cublasZtbsv_v2 cublasZtbsv_v2;
  da_cublasSsymv_v2 cublasSsymv_v2;
  da_cublasDsymv_v2 cublasDsymv_v2;
  da_cublasCsymv_v2 cublasCsymv_v2;
  da_cublasZsymv_v2 cublasZsymv_v2;
  da_cublasChemv_v2 cublasChemv_v2;
  da_cublasZhemv_v2 cublasZhemv_v2;
  da_cublasSsbmv_v2 cublasSsbmv_v2;
  da_cublasDsbmv_v2 cublasDsbmv_v2;
  da_cublasChbmv_v2 cublasChbmv_v2;
  da_cublasZhbmv_v2 cublasZhbmv_v2;
  da_cublasSspmv_v2 cublasSspmv_v2;
  da_cublasDspmv_v2 cublasDspmv_v2;
  da_cublasChpmv_v2 cublasChpmv_v2;
  da_cublasZhpmv_v2 cublasZhpmv_v2;
  da_cublasSger_v2 cublasSger_v2;
  da_cublasDger_v2 cublasDger_v2;
  da_cublasCgeru_v2 cublasCgeru_v2;
  da_cublasCgerc_v2 cublasCgerc_v2;
  da_cublasZgeru_v2 cublasZgeru_v2;
  da_cublasZgerc_v2 cublasZgerc_v2;
  da_cublasSsyr_v2 cublasSsyr_v2;
  da_cublasDsyr_v2 cublasDsyr_v2;
  da_cublasCsyr_v2 cublasCsyr_v2;
  da_cublasZsyr_v2 cublasZsyr_v2;
  da_cublasCher_v2 cublasCher_v2;
  da_cublasZher_v2 cublasZher_v2;
  da_cublasSspr_v2 cublasSspr_v2;
  da_cublasDspr_v2 cublasDspr_v2;
  da_cublasChpr_v2 cublasChpr_v2;
  da_cublasZhpr_v2 cublasZhpr_v2;
  da_cublasSsyr2_v2 cublasSsyr2_v2;
  da_cublasDsyr2_v2 cublasDsyr2_v2;
  da_cublasCsyr2_v2 cublasCsyr2_v2;
  da_cublasZsyr2_v2 cublasZsyr2_v2;
  da_cublasCher2_v2 cublasCher2_v2;
  da_cublasZher2_v2 cublasZher2_v2;
  da_cublasSspr2_v2 cublasSspr2_v2;
  da_cublasDspr2_v2 cublasDspr2_v2;
  da_cublasChpr2_v2 cublasChpr2_v2;
  da_cublasZhpr2_v2 cublasZhpr2_v2;
  da_cublasSgemm_v2 cublasSgemm_v2;
  da_cublasDgemm_v2 cublasDgemm_v2;
  da_cublasCgemm_v2 cublasCgemm_v2;
  da_cublasCgemm3m cublasCgemm3m;
  da_cublasCgemm3mEx cublasCgemm3mEx;
  da_cublasZgemm_v2 cublasZgemm_v2;
  da_cublasZgemm3m cublasZgemm3m;
  //da_cublasHgemm cublasHgemm;
  da_cublasSgemmEx cublasSgemmEx;
  da_cublasGemmEx cublasGemmEx;
  da_cublasCgemmEx cublasCgemmEx;
  da_cublasUint8gemmBias cublasUint8gemmBias;
  da_cublasSsyrk_v2 cublasSsyrk_v2;
  da_cublasDsyrk_v2 cublasDsyrk_v2;
  da_cublasCsyrk_v2 cublasCsyrk_v2;
  da_cublasZsyrk_v2 cublasZsyrk_v2;
  da_cublasCsyrkEx cublasCsyrkEx;
  da_cublasCsyrk3mEx cublasCsyrk3mEx;
  da_cublasCherk_v2 cublasCherk_v2;
  da_cublasZherk_v2 cublasZherk_v2;
  da_cublasCherkEx cublasCherkEx;
  da_cublasCherk3mEx cublasCherk3mEx;
  da_cublasSsyr2k_v2 cublasSsyr2k_v2;
  da_cublasDsyr2k_v2 cublasDsyr2k_v2;
  da_cublasCsyr2k_v2 cublasCsyr2k_v2;
  da_cublasZsyr2k_v2 cublasZsyr2k_v2;
  da_cublasCher2k_v2 cublasCher2k_v2;
  da_cublasZher2k_v2 cublasZher2k_v2;
  da_cublasSsyrkx cublasSsyrkx;
  da_cublasDsyrkx cublasDsyrkx;
  da_cublasCsyrkx cublasCsyrkx;
  da_cublasZsyrkx cublasZsyrkx;
  da_cublasCherkx cublasCherkx;
  da_cublasZherkx cublasZherkx;
  da_cublasSsymm_v2 cublasSsymm_v2;
  da_cublasDsymm_v2 cublasDsymm_v2;
  da_cublasCsymm_v2 cublasCsymm_v2;
  da_cublasZsymm_v2 cublasZsymm_v2;
  da_cublasChemm_v2 cublasChemm_v2;
  da_cublasZhemm_v2 cublasZhemm_v2;
  da_cublasStrsm_v2 cublasStrsm_v2;
  da_cublasDtrsm_v2 cublasDtrsm_v2;
  da_cublasCtrsm_v2 cublasCtrsm_v2;
  da_cublasZtrsm_v2 cublasZtrsm_v2;
  da_cublasStrmm_v2 cublasStrmm_v2;
  da_cublasDtrmm_v2 cublasDtrmm_v2;
  da_cublasCtrmm_v2 cublasCtrmm_v2;
  da_cublasZtrmm_v2 cublasZtrmm_v2;
  //da_cublasHgemmBatched cublasHgemmBatched;
  da_cublasSgemmBatched cublasSgemmBatched;
  da_cublasDgemmBatched cublasDgemmBatched;
  da_cublasCgemmBatched cublasCgemmBatched;
  da_cublasCgemm3mBatched cublasCgemm3mBatched;
  da_cublasZgemmBatched cublasZgemmBatched;
  da_cublasGemmBatchedEx cublasGemmBatchedEx;
  da_cublasGemmStridedBatchedEx cublasGemmStridedBatchedEx;
  da_cublasSgemmStridedBatched cublasSgemmStridedBatched;
  da_cublasDgemmStridedBatched cublasDgemmStridedBatched;
  da_cublasCgemmStridedBatched cublasCgemmStridedBatched;
  da_cublasCgemm3mStridedBatched cublasCgemm3mStridedBatched;
  da_cublasZgemmStridedBatched cublasZgemmStridedBatched;
  //da_cublasHgemmStridedBatched cublasHgemmStridedBatched;
  da_cublasSgeam cublasSgeam;
  da_cublasDgeam cublasDgeam;
  da_cublasCgeam cublasCgeam;
  da_cublasZgeam cublasZgeam;
  da_cublasSgetrfBatched cublasSgetrfBatched;
  da_cublasDgetrfBatched cublasDgetrfBatched;
  da_cublasCgetrfBatched cublasCgetrfBatched;
  da_cublasZgetrfBatched cublasZgetrfBatched;
  da_cublasSgetriBatched cublasSgetriBatched;
  da_cublasDgetriBatched cublasDgetriBatched;
  da_cublasCgetriBatched cublasCgetriBatched;
  da_cublasZgetriBatched cublasZgetriBatched;
  da_cublasSgetrsBatched cublasSgetrsBatched;
  da_cublasDgetrsBatched cublasDgetrsBatched;
  da_cublasCgetrsBatched cublasCgetrsBatched;
  da_cublasZgetrsBatched cublasZgetrsBatched;
  da_cublasStrsmBatched cublasStrsmBatched;
  da_cublasDtrsmBatched cublasDtrsmBatched;
  da_cublasCtrsmBatched cublasCtrsmBatched;
  da_cublasZtrsmBatched cublasZtrsmBatched;
  da_cublasSmatinvBatched cublasSmatinvBatched;
  da_cublasDmatinvBatched cublasDmatinvBatched;
  da_cublasCmatinvBatched cublasCmatinvBatched;
  da_cublasZmatinvBatched cublasZmatinvBatched;
  da_cublasSgeqrfBatched cublasSgeqrfBatched;
  da_cublasDgeqrfBatched cublasDgeqrfBatched;
  da_cublasCgeqrfBatched cublasCgeqrfBatched;
  da_cublasZgeqrfBatched cublasZgeqrfBatched;
  da_cublasSgelsBatched cublasSgelsBatched;
  da_cublasDgelsBatched cublasDgelsBatched;
  da_cublasCgelsBatched cublasCgelsBatched;
  da_cublasZgelsBatched cublasZgelsBatched;
  da_cublasSdgmm cublasSdgmm;
  da_cublasDdgmm cublasDdgmm;
  da_cublasCdgmm cublasCdgmm;
  da_cublasZdgmm cublasZdgmm;
  da_cublasStpttr cublasStpttr;
  da_cublasDtpttr cublasDtpttr;
  da_cublasCtpttr cublasCtpttr;
  da_cublasZtpttr cublasZtpttr;
  da_cublasStrttp cublasStrttp;
  da_cublasDtrttp cublasDtrttp;
  da_cublasCtrttp cublasCtrttp;
  da_cublasZtrttp cublasZtrttp;
}

class DerelictCuBLASLoader : SharedLibLoader
{
  protected
  {
    override void loadSymbols()
    {
      bindFunc(cast(void**)&cublasCreate_v2, "cublasCreate_v2");
      bindFunc(cast(void**)&cublasDestroy_v2, "cublasDestroy_v2");
      bindFunc(cast(void**)&cublasGetVersion_v2, "cublasGetVersion_v2");
      bindFunc(cast(void**)&cublasGetProperty, "cublasGetProperty");
      bindFunc(cast(void**)&cublasSetStream_v2, "cublasSetStream_v2");
      bindFunc(cast(void**)&cublasGetStream_v2, "cublasGetStream_v2");
      bindFunc(cast(void**)&cublasGetPointerMode_v2, "cublasGetPointerMode_v2");
      bindFunc(cast(void**)&cublasSetPointerMode_v2, "cublasSetPointerMode_v2");
      bindFunc(cast(void**)&cublasGetAtomicsMode, "cublasGetAtomicsMode");
      bindFunc(cast(void**)&cublasSetAtomicsMode, "cublasSetAtomicsMode");
      bindFunc(cast(void**)&cublasGetMathMode, "cublasGetMathMode");
      bindFunc(cast(void**)&cublasSetMathMode, "cublasSetMathMode");
      bindFunc(cast(void**)&cublasLoggerConfigure, "cublasLoggerConfigure");
      bindFunc(cast(void**)&cublasSetLoggerCallback, "cublasSetLoggerCallback");
      bindFunc(cast(void**)&cublasGetLoggerCallback, "cublasGetLoggerCallback");
      bindFunc(cast(void**)&cublasSetVector, "cublasSetVector");
      bindFunc(cast(void**)&cublasGetVector, "cublasGetVector");
      bindFunc(cast(void**)&cublasSetMatrix, "cublasSetMatrix");
      bindFunc(cast(void**)&cublasGetMatrix, "cublasGetMatrix");
      bindFunc(cast(void**)&cublasSetVectorAsync, "cublasSetVectorAsync");
      bindFunc(cast(void**)&cublasGetVectorAsync, "cublasGetVectorAsync");
      bindFunc(cast(void**)&cublasSetMatrixAsync, "cublasSetMatrixAsync");
      bindFunc(cast(void**)&cublasGetMatrixAsync, "cublasGetMatrixAsync");
      bindFunc(cast(void**)&cublasXerbla, "cublasXerbla");
      bindFunc(cast(void**)&cublasNrm2Ex, "cublasNrm2Ex");
      bindFunc(cast(void**)&cublasSnrm2_v2, "cublasSnrm2_v2");
      bindFunc(cast(void**)&cublasDnrm2_v2, "cublasDnrm2_v2");
      bindFunc(cast(void**)&cublasScnrm2_v2, "cublasScnrm2_v2");
      bindFunc(cast(void**)&cublasDznrm2_v2, "cublasDznrm2_v2");
      bindFunc(cast(void**)&cublasDotEx, "cublasDotEx");
      bindFunc(cast(void**)&cublasDotcEx, "cublasDotcEx");
      bindFunc(cast(void**)&cublasSdot_v2, "cublasSdot_v2");
      bindFunc(cast(void**)&cublasDdot_v2, "cublasDdot_v2");
      bindFunc(cast(void**)&cublasCdotu_v2, "cublasCdotu_v2");
      bindFunc(cast(void**)&cublasCdotc_v2, "cublasCdotc_v2");
      bindFunc(cast(void**)&cublasZdotu_v2, "cublasZdotu_v2");
      bindFunc(cast(void**)&cublasZdotc_v2, "cublasZdotc_v2");
      bindFunc(cast(void**)&cublasScalEx, "cublasScalEx");
      bindFunc(cast(void**)&cublasSscal_v2, "cublasSscal_v2");
      bindFunc(cast(void**)&cublasDscal_v2, "cublasDscal_v2");
      bindFunc(cast(void**)&cublasCscal_v2, "cublasCscal_v2");
      bindFunc(cast(void**)&cublasCsscal_v2, "cublasCsscal_v2");
      bindFunc(cast(void**)&cublasZscal_v2, "cublasZscal_v2");
      bindFunc(cast(void**)&cublasZdscal_v2, "cublasZdscal_v2");
      bindFunc(cast(void**)&cublasAxpyEx, "cublasAxpyEx");
      bindFunc(cast(void**)&cublasSaxpy_v2, "cublasSaxpy_v2");
      bindFunc(cast(void**)&cublasDaxpy_v2, "cublasDaxpy_v2");
      bindFunc(cast(void**)&cublasCaxpy_v2, "cublasCaxpy_v2");
      bindFunc(cast(void**)&cublasZaxpy_v2, "cublasZaxpy_v2");
      bindFunc(cast(void**)&cublasScopy_v2, "cublasScopy_v2");
      bindFunc(cast(void**)&cublasDcopy_v2, "cublasDcopy_v2");
      bindFunc(cast(void**)&cublasCcopy_v2, "cublasCcopy_v2");
      bindFunc(cast(void**)&cublasZcopy_v2, "cublasZcopy_v2");
      bindFunc(cast(void**)&cublasSswap_v2, "cublasSswap_v2");
      bindFunc(cast(void**)&cublasDswap_v2, "cublasDswap_v2");
      bindFunc(cast(void**)&cublasCswap_v2, "cublasCswap_v2");
      bindFunc(cast(void**)&cublasZswap_v2, "cublasZswap_v2");
      bindFunc(cast(void**)&cublasIsamax_v2, "cublasIsamax_v2");
      bindFunc(cast(void**)&cublasIdamax_v2, "cublasIdamax_v2");
      bindFunc(cast(void**)&cublasIcamax_v2, "cublasIcamax_v2");
      bindFunc(cast(void**)&cublasIzamax_v2, "cublasIzamax_v2");
      bindFunc(cast(void**)&cublasIsamin_v2, "cublasIsamin_v2");
      bindFunc(cast(void**)&cublasIdamin_v2, "cublasIdamin_v2");
      bindFunc(cast(void**)&cublasIcamin_v2, "cublasIcamin_v2");
      bindFunc(cast(void**)&cublasIzamin_v2, "cublasIzamin_v2");
      bindFunc(cast(void**)&cublasSasum_v2, "cublasSasum_v2");
      bindFunc(cast(void**)&cublasDasum_v2, "cublasDasum_v2");
      bindFunc(cast(void**)&cublasScasum_v2, "cublasScasum_v2");
      bindFunc(cast(void**)&cublasDzasum_v2, "cublasDzasum_v2");
      bindFunc(cast(void**)&cublasSrot_v2, "cublasSrot_v2");
      bindFunc(cast(void**)&cublasDrot_v2, "cublasDrot_v2");
      bindFunc(cast(void**)&cublasCrot_v2, "cublasCrot_v2");
      bindFunc(cast(void**)&cublasCsrot_v2, "cublasCsrot_v2");
      bindFunc(cast(void**)&cublasZrot_v2, "cublasZrot_v2");
      bindFunc(cast(void**)&cublasZdrot_v2, "cublasZdrot_v2");
      bindFunc(cast(void**)&cublasSrotg_v2, "cublasSrotg_v2");
      bindFunc(cast(void**)&cublasDrotg_v2, "cublasDrotg_v2");
      bindFunc(cast(void**)&cublasCrotg_v2, "cublasCrotg_v2");
      bindFunc(cast(void**)&cublasZrotg_v2, "cublasZrotg_v2");
      bindFunc(cast(void**)&cublasSrotm_v2, "cublasSrotm_v2");
      bindFunc(cast(void**)&cublasDrotm_v2, "cublasDrotm_v2");
      bindFunc(cast(void**)&cublasSrotmg_v2, "cublasSrotmg_v2");
      bindFunc(cast(void**)&cublasDrotmg_v2, "cublasDrotmg_v2");
      bindFunc(cast(void**)&cublasSgemv_v2, "cublasSgemv_v2");
      bindFunc(cast(void**)&cublasDgemv_v2, "cublasDgemv_v2");
      bindFunc(cast(void**)&cublasCgemv_v2, "cublasCgemv_v2");
      bindFunc(cast(void**)&cublasZgemv_v2, "cublasZgemv_v2");
      bindFunc(cast(void**)&cublasSgbmv_v2, "cublasSgbmv_v2");
      bindFunc(cast(void**)&cublasDgbmv_v2, "cublasDgbmv_v2");
      bindFunc(cast(void**)&cublasCgbmv_v2, "cublasCgbmv_v2");
      bindFunc(cast(void**)&cublasZgbmv_v2, "cublasZgbmv_v2");
      bindFunc(cast(void**)&cublasStrmv_v2, "cublasStrmv_v2");
      bindFunc(cast(void**)&cublasDtrmv_v2, "cublasDtrmv_v2");
      bindFunc(cast(void**)&cublasCtrmv_v2, "cublasCtrmv_v2");
      bindFunc(cast(void**)&cublasZtrmv_v2, "cublasZtrmv_v2");
      bindFunc(cast(void**)&cublasStbmv_v2, "cublasStbmv_v2");
      bindFunc(cast(void**)&cublasDtbmv_v2, "cublasDtbmv_v2");
      bindFunc(cast(void**)&cublasCtbmv_v2, "cublasCtbmv_v2");
      bindFunc(cast(void**)&cublasZtbmv_v2, "cublasZtbmv_v2");
      bindFunc(cast(void**)&cublasStpmv_v2, "cublasStpmv_v2");
      bindFunc(cast(void**)&cublasDtpmv_v2, "cublasDtpmv_v2");
      bindFunc(cast(void**)&cublasCtpmv_v2, "cublasCtpmv_v2");
      bindFunc(cast(void**)&cublasZtpmv_v2, "cublasZtpmv_v2");
      bindFunc(cast(void**)&cublasStrsv_v2, "cublasStrsv_v2");
      bindFunc(cast(void**)&cublasDtrsv_v2, "cublasDtrsv_v2");
      bindFunc(cast(void**)&cublasCtrsv_v2, "cublasCtrsv_v2");
      bindFunc(cast(void**)&cublasZtrsv_v2, "cublasZtrsv_v2");
      bindFunc(cast(void**)&cublasStpsv_v2, "cublasStpsv_v2");
      bindFunc(cast(void**)&cublasDtpsv_v2, "cublasDtpsv_v2");
      bindFunc(cast(void**)&cublasCtpsv_v2, "cublasCtpsv_v2");
      bindFunc(cast(void**)&cublasZtpsv_v2, "cublasZtpsv_v2");
      bindFunc(cast(void**)&cublasStbsv_v2, "cublasStbsv_v2");
      bindFunc(cast(void**)&cublasDtbsv_v2, "cublasDtbsv_v2");
      bindFunc(cast(void**)&cublasCtbsv_v2, "cublasCtbsv_v2");
      bindFunc(cast(void**)&cublasZtbsv_v2, "cublasZtbsv_v2");
      bindFunc(cast(void**)&cublasSsymv_v2, "cublasSsymv_v2");
      bindFunc(cast(void**)&cublasDsymv_v2, "cublasDsymv_v2");
      bindFunc(cast(void**)&cublasCsymv_v2, "cublasCsymv_v2");
      bindFunc(cast(void**)&cublasZsymv_v2, "cublasZsymv_v2");
      bindFunc(cast(void**)&cublasChemv_v2, "cublasChemv_v2");
      bindFunc(cast(void**)&cublasZhemv_v2, "cublasZhemv_v2");
      bindFunc(cast(void**)&cublasSsbmv_v2, "cublasSsbmv_v2");
      bindFunc(cast(void**)&cublasDsbmv_v2, "cublasDsbmv_v2");
      bindFunc(cast(void**)&cublasChbmv_v2, "cublasChbmv_v2");
      bindFunc(cast(void**)&cublasZhbmv_v2, "cublasZhbmv_v2");
      bindFunc(cast(void**)&cublasSspmv_v2, "cublasSspmv_v2");
      bindFunc(cast(void**)&cublasDspmv_v2, "cublasDspmv_v2");
      bindFunc(cast(void**)&cublasChpmv_v2, "cublasChpmv_v2");
      bindFunc(cast(void**)&cublasZhpmv_v2, "cublasZhpmv_v2");
      bindFunc(cast(void**)&cublasSger_v2, "cublasSger_v2");
      bindFunc(cast(void**)&cublasDger_v2, "cublasDger_v2");
      bindFunc(cast(void**)&cublasCgeru_v2, "cublasCgeru_v2");
      bindFunc(cast(void**)&cublasCgerc_v2, "cublasCgerc_v2");
      bindFunc(cast(void**)&cublasZgeru_v2, "cublasZgeru_v2");
      bindFunc(cast(void**)&cublasZgerc_v2, "cublasZgerc_v2");
      bindFunc(cast(void**)&cublasSsyr_v2, "cublasSsyr_v2");
      bindFunc(cast(void**)&cublasDsyr_v2, "cublasDsyr_v2");
      bindFunc(cast(void**)&cublasCsyr_v2, "cublasCsyr_v2");
      bindFunc(cast(void**)&cublasZsyr_v2, "cublasZsyr_v2");
      bindFunc(cast(void**)&cublasCher_v2, "cublasCher_v2");
      bindFunc(cast(void**)&cublasZher_v2, "cublasZher_v2");
      bindFunc(cast(void**)&cublasSspr_v2, "cublasSspr_v2");
      bindFunc(cast(void**)&cublasDspr_v2, "cublasDspr_v2");
      bindFunc(cast(void**)&cublasChpr_v2, "cublasChpr_v2");
      bindFunc(cast(void**)&cublasZhpr_v2, "cublasZhpr_v2");
      bindFunc(cast(void**)&cublasSsyr2_v2, "cublasSsyr2_v2");
      bindFunc(cast(void**)&cublasDsyr2_v2, "cublasDsyr2_v2");
      bindFunc(cast(void**)&cublasCsyr2_v2, "cublasCsyr2_v2");
      bindFunc(cast(void**)&cublasZsyr2_v2, "cublasZsyr2_v2");
      bindFunc(cast(void**)&cublasCher2_v2, "cublasCher2_v2");
      bindFunc(cast(void**)&cublasZher2_v2, "cublasZher2_v2");
      bindFunc(cast(void**)&cublasSspr2_v2, "cublasSspr2_v2");
      bindFunc(cast(void**)&cublasDspr2_v2, "cublasDspr2_v2");
      bindFunc(cast(void**)&cublasChpr2_v2, "cublasChpr2_v2");
      bindFunc(cast(void**)&cublasZhpr2_v2, "cublasZhpr2_v2");
      bindFunc(cast(void**)&cublasSgemm_v2, "cublasSgemm_v2");
      bindFunc(cast(void**)&cublasDgemm_v2, "cublasDgemm_v2");
      bindFunc(cast(void**)&cublasCgemm_v2, "cublasCgemm_v2");
      bindFunc(cast(void**)&cublasCgemm3m, "cublasCgemm3m");
      bindFunc(cast(void**)&cublasCgemm3mEx, "cublasCgemm3mEx");
      bindFunc(cast(void**)&cublasZgemm_v2, "cublasZgemm_v2");
      bindFunc(cast(void**)&cublasZgemm3m, "cublasZgemm3m");
      //bindFunc(cast(void**)&cublasHgemm, "cublasHgemm");
      bindFunc(cast(void**)&cublasSgemmEx, "cublasSgemmEx");
      bindFunc(cast(void**)&cublasGemmEx, "cublasGemmEx");
      bindFunc(cast(void**)&cublasCgemmEx, "cublasCgemmEx");
      bindFunc(cast(void**)&cublasUint8gemmBias, "cublasUint8gemmBias");
      bindFunc(cast(void**)&cublasSsyrk_v2, "cublasSsyrk_v2");
      bindFunc(cast(void**)&cublasDsyrk_v2, "cublasDsyrk_v2");
      bindFunc(cast(void**)&cublasCsyrk_v2, "cublasCsyrk_v2");
      bindFunc(cast(void**)&cublasZsyrk_v2, "cublasZsyrk_v2");
      bindFunc(cast(void**)&cublasCsyrkEx, "cublasCsyrkEx");
      bindFunc(cast(void**)&cublasCsyrk3mEx, "cublasCsyrk3mEx");
      bindFunc(cast(void**)&cublasCherk_v2, "cublasCherk_v2");
      bindFunc(cast(void**)&cublasZherk_v2, "cublasZherk_v2");
      bindFunc(cast(void**)&cublasCherkEx, "cublasCherkEx");
      bindFunc(cast(void**)&cublasCherk3mEx, "cublasCherk3mEx");
      bindFunc(cast(void**)&cublasSsyr2k_v2, "cublasSsyr2k_v2");
      bindFunc(cast(void**)&cublasDsyr2k_v2, "cublasDsyr2k_v2");
      bindFunc(cast(void**)&cublasCsyr2k_v2, "cublasCsyr2k_v2");
      bindFunc(cast(void**)&cublasZsyr2k_v2, "cublasZsyr2k_v2");
      bindFunc(cast(void**)&cublasCher2k_v2, "cublasCher2k_v2");
      bindFunc(cast(void**)&cublasZher2k_v2, "cublasZher2k_v2");
      bindFunc(cast(void**)&cublasSsyrkx, "cublasSsyrkx");
      bindFunc(cast(void**)&cublasDsyrkx, "cublasDsyrkx");
      bindFunc(cast(void**)&cublasCsyrkx, "cublasCsyrkx");
      bindFunc(cast(void**)&cublasZsyrkx, "cublasZsyrkx");
      bindFunc(cast(void**)&cublasCherkx, "cublasCherkx");
      bindFunc(cast(void**)&cublasZherkx, "cublasZherkx");
      bindFunc(cast(void**)&cublasSsymm_v2, "cublasSsymm_v2");
      bindFunc(cast(void**)&cublasDsymm_v2, "cublasDsymm_v2");
      bindFunc(cast(void**)&cublasCsymm_v2, "cublasCsymm_v2");
      bindFunc(cast(void**)&cublasZsymm_v2, "cublasZsymm_v2");
      bindFunc(cast(void**)&cublasChemm_v2, "cublasChemm_v2");
      bindFunc(cast(void**)&cublasZhemm_v2, "cublasZhemm_v2");
      bindFunc(cast(void**)&cublasStrsm_v2, "cublasStrsm_v2");
      bindFunc(cast(void**)&cublasDtrsm_v2, "cublasDtrsm_v2");
      bindFunc(cast(void**)&cublasCtrsm_v2, "cublasCtrsm_v2");
      bindFunc(cast(void**)&cublasZtrsm_v2, "cublasZtrsm_v2");
      bindFunc(cast(void**)&cublasStrmm_v2, "cublasStrmm_v2");
      bindFunc(cast(void**)&cublasDtrmm_v2, "cublasDtrmm_v2");
      bindFunc(cast(void**)&cublasCtrmm_v2, "cublasCtrmm_v2");
      bindFunc(cast(void**)&cublasZtrmm_v2, "cublasZtrmm_v2");
      //bindFunc(cast(void**)&cublasHgemmBatched, "cublasHgemmBatched");
      bindFunc(cast(void**)&cublasSgemmBatched, "cublasSgemmBatched");
      bindFunc(cast(void**)&cublasDgemmBatched, "cublasDgemmBatched");
      bindFunc(cast(void**)&cublasCgemmBatched, "cublasCgemmBatched");
      bindFunc(cast(void**)&cublasCgemm3mBatched, "cublasCgemm3mBatched");
      bindFunc(cast(void**)&cublasZgemmBatched, "cublasZgemmBatched");
      bindFunc(cast(void**)&cublasGemmBatchedEx, "cublasGemmBatchedEx");
      bindFunc(cast(void**)&cublasGemmStridedBatchedEx, "cublasGemmStridedBatchedEx");
      bindFunc(cast(void**)&cublasSgemmStridedBatched, "cublasSgemmStridedBatched");
      bindFunc(cast(void**)&cublasDgemmStridedBatched, "cublasDgemmStridedBatched");
      bindFunc(cast(void**)&cublasCgemmStridedBatched, "cublasCgemmStridedBatched");
      bindFunc(cast(void**)&cublasCgemm3mStridedBatched, "cublasCgemm3mStridedBatched");
      bindFunc(cast(void**)&cublasZgemmStridedBatched, "cublasZgemmStridedBatched");
      //bindFunc(cast(void**)&cublasHgemmStridedBatched, "cublasHgemmStridedBatched");
      bindFunc(cast(void**)&cublasSgeam, "cublasSgeam");
      bindFunc(cast(void**)&cublasDgeam, "cublasDgeam");
      bindFunc(cast(void**)&cublasCgeam, "cublasCgeam");
      bindFunc(cast(void**)&cublasZgeam, "cublasZgeam");
      bindFunc(cast(void**)&cublasSgetrfBatched, "cublasSgetrfBatched");
      bindFunc(cast(void**)&cublasDgetrfBatched, "cublasDgetrfBatched");
      bindFunc(cast(void**)&cublasCgetrfBatched, "cublasCgetrfBatched");
      bindFunc(cast(void**)&cublasZgetrfBatched, "cublasZgetrfBatched");
      bindFunc(cast(void**)&cublasSgetriBatched, "cublasSgetriBatched");
      bindFunc(cast(void**)&cublasDgetriBatched, "cublasDgetriBatched");
      bindFunc(cast(void**)&cublasCgetriBatched, "cublasCgetriBatched");
      bindFunc(cast(void**)&cublasZgetriBatched, "cublasZgetriBatched");
      bindFunc(cast(void**)&cublasSgetrsBatched, "cublasSgetrsBatched");
      bindFunc(cast(void**)&cublasDgetrsBatched, "cublasDgetrsBatched");
      bindFunc(cast(void**)&cublasCgetrsBatched, "cublasCgetrsBatched");
      bindFunc(cast(void**)&cublasZgetrsBatched, "cublasZgetrsBatched");
      bindFunc(cast(void**)&cublasStrsmBatched, "cublasStrsmBatched");
      bindFunc(cast(void**)&cublasDtrsmBatched, "cublasDtrsmBatched");
      bindFunc(cast(void**)&cublasCtrsmBatched, "cublasCtrsmBatched");
      bindFunc(cast(void**)&cublasZtrsmBatched, "cublasZtrsmBatched");
      bindFunc(cast(void**)&cublasSmatinvBatched, "cublasSmatinvBatched");
      bindFunc(cast(void**)&cublasDmatinvBatched, "cublasDmatinvBatched");
      bindFunc(cast(void**)&cublasCmatinvBatched, "cublasCmatinvBatched");
      bindFunc(cast(void**)&cublasZmatinvBatched, "cublasZmatinvBatched");
      bindFunc(cast(void**)&cublasSgeqrfBatched, "cublasSgeqrfBatched");
      bindFunc(cast(void**)&cublasDgeqrfBatched, "cublasDgeqrfBatched");
      bindFunc(cast(void**)&cublasCgeqrfBatched, "cublasCgeqrfBatched");
      bindFunc(cast(void**)&cublasZgeqrfBatched, "cublasZgeqrfBatched");
      bindFunc(cast(void**)&cublasSgelsBatched, "cublasSgelsBatched");
      bindFunc(cast(void**)&cublasDgelsBatched, "cublasDgelsBatched");
      bindFunc(cast(void**)&cublasCgelsBatched, "cublasCgelsBatched");
      bindFunc(cast(void**)&cublasZgelsBatched, "cublasZgelsBatched");
      bindFunc(cast(void**)&cublasSdgmm, "cublasSdgmm");
      bindFunc(cast(void**)&cublasDdgmm, "cublasDdgmm");
      bindFunc(cast(void**)&cublasCdgmm, "cublasCdgmm");
      bindFunc(cast(void**)&cublasZdgmm, "cublasZdgmm");
      bindFunc(cast(void**)&cublasStpttr, "cublasStpttr");
      bindFunc(cast(void**)&cublasDtpttr, "cublasDtpttr");
      bindFunc(cast(void**)&cublasCtpttr, "cublasCtpttr");
      bindFunc(cast(void**)&cublasZtpttr, "cublasZtpttr");
      bindFunc(cast(void**)&cublasStrttp, "cublasStrttp");
      bindFunc(cast(void**)&cublasDtrttp, "cublasDtrttp");
      bindFunc(cast(void**)&cublasCtrttp, "cublasCtrttp");
      bindFunc(cast(void**)&cublasZtrttp, "cublasZtrttp");
    }
  }

  public
  {
    this()
    {
      super(libNames);
    }
  }
}


__gshared DerelictCuBLASLoader DerelictCuBLAS;

shared static this()
{
    DerelictCuBLAS = new DerelictCuBLASLoader();
}
