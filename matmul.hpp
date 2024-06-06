#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace cuda_kernels {
void mmult_naive(float *out, float *a, float *b, int m, int k, int n);
void mmult_cublas(cublasHandle_t cublas_handle, float *out, float *a, float *b,
                  int m, int k, int n);
} // namespace cuda_kernels
