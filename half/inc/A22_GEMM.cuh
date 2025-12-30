// A22_GEMM.cuh
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdint.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",                 \
                    __FILE__, __LINE__, (int)st__);                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

// 如果你真的想让 GEMM 内部 setStream（不建议），可以打开这个宏
// #define A22_GEMM_SET_STREAM_INSIDE 1

/**
 * Tensor Core GEMM（带列范围）:
 *
 *  A22(col0:col0+n2-1) -= A21 * A12(col0:col0+n2-1)
 */
inline void launch_A22_gemm_tc_range(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    int   col0, int n2,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    if (!dA || !handle) return;
    if (ib <= 0 || n2 <= 0) return;

    const int row0 = j0 + ib;
    const int m2   = m - row0;
    if (m2 <= 0) return;
    if (col0 >= n) return;
    if (col0 + n2 > n) n2 = n - col0;

    half* A21 = dA + row0 + (size_t)j0   * lda;  // (m2 x ib)
    half* A12 = dA + j0   + (size_t)col0 * lda;  // (ib x n2)
    half* A22 = dA + row0 + (size_t)col0 * lda;  // (m2 x n2)

    const float alpha = -1.0f;
    const float beta  =  1.0f;

#if defined(A22_GEMM_SET_STREAM_INSIDE)
    CUBLAS_CHECK(cublasSetStream(handle, stream));
#else
    // 期望外层已经把 handle 的 stream 设置为 stream
    (void)stream;
#endif

    CUBLAS_CHECK(
        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            m2,         // m
            n2,         // n
            ib,         // k
            &alpha,
            A21, CUDA_R_16F, lda,
            A12, CUDA_R_16F, lda,
            &beta,
            A22, CUDA_R_16F, lda,
            CUDA_R_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}


/**
 * naive GEMM（一次性完整 tail）
 */
__global__ void A22_gemm_naive_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    int col0, int n2)
{
    int row0 = j0 + ib;
    int m2   = m - row0;
    if (m2 <= 0 || n2 <= 0) return;

    int j_rel = blockIdx.x * blockDim.x + threadIdx.x;
    int i_rel = blockIdx.y * blockDim.y + threadIdx.y;

    if (j_rel >= n2 || i_rel >= m2) return;

    int i = row0 + i_rel;
    int j = col0 + j_rel;

    float sum = 0.0f;
    for (int k = 0; k < ib; ++k) {
        int kcol = j0 + k;

        half a_h = A[i + (size_t)kcol * lda];
        half b_h = A[(j0 + k) + (size_t)j * lda];

        float a = __half2float(a_h);
        float b = __half2float(b_h);

        sum += a * b;
    }

    half c_h = A[i + (size_t)j * lda];
    float c  = __half2float(c_h);

    float res = c - sum;
    A[i + (size_t)j * lda] = __float2half(res);
}

inline void launch_A22_gemm_naive_range(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    int   col0, int n2,
    cudaStream_t stream)
{
    if (!dA) return;
    int row0 = j0 + ib;
    int m2   = m - row0;
    if (m2 <= 0 || n2 <= 0) return;

    if (col0 >= n) return;
    if (col0 + n2 > n) n2 = n - col0;

    dim3 block(16, 16);
    dim3 grid((n2 + block.x - 1) / block.x,
              (m2 + block.y - 1) / block.y);

    A22_gemm_naive_kernel<<<grid, block, 0, stream>>>(
        dA, m, n, lda, j0, ib, col0, n2
    );
    CUDA_CHECK(cudaGetLastError());
}

/**
 * 兼容老接口：一次性更新整个 tail
 */
inline void launch_A22_gemm_tc(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    int col0 = j0 + ib;
    int n2   = n - col0;
    launch_A22_gemm_tc_range(dA, m, n, lda, j0, ib, col0, n2, handle, stream);
}

inline void launch_A22_gemm_naive(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    cudaStream_t stream)
{
    int col0 = j0 + ib;
    int n2   = n - col0;
    launch_A22_gemm_naive_range(dA, m, n, lda, j0, ib, col0, n2, stream);
}
