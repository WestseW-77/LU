#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>

int main() {
    int n = 32768;
    int lda = n;
    
    half *dA, *dB, *dC;
    cudaMalloc(&dA, sizeof(half) * n * n);
    cudaMalloc(&dB, sizeof(half) * n * n);
    cudaMalloc(&dC, sizeof(half) * n * n);
    cudaMemset(dA, 0, sizeof(half) * n * n);
    cudaMemset(dB, 0, sizeof(half) * n * n);
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // 预热
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                 n, n, n, &alpha,
                 dA, CUDA_R_16F, lda,
                 dB, CUDA_R_16F, lda,
                 &beta,
                 dC, CUDA_R_16F, lda,
                 CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                     n, n, n, &alpha,
                     dA, CUDA_R_16F, lda,
                     dB, CUDA_R_16F, lda,
                     &beta,
                     dC, CUDA_R_16F, lda,
                     CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    
    double flops = 2.0 * (double)n * n * n;
    double tflops = (flops * 10 / (ms * 1e-3)) * 1e-12;
    
    printf("Pure cuBLAS GEMM (n=%d):\n", n);
    printf("  Time: %.2f ms\n", ms / 10);
    printf("  Performance: %.2f TFLOPS\n", tflops);
    
    return 0;
}