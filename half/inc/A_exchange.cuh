#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
        }                                                                      \
    } while (0)
#endif

using half = __half;

/**
 * ============================================================================
 * A_exchange.cuh - 原始稳定版本
 * ============================================================================
 * 
 * 这是经过验证的稳定实现，保证正确性
 * 
 * ============================================================================
 */

/**
 * 单行交换kernel - 原始版本
 */
__global__ void swap_rows_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int row1, int row2,
    int j0, int ib)
{
    const int j_panel_begin = j0;
    const int j_panel_end = j0 + ib;
    
    // 每个线程处理一列
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n) return;
    
    // 跳过panel内的列
    if (j >= j_panel_begin && j < j_panel_end) return;
    
    const size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
    const size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;
    
    // 交换
    const half tmp = A[idx1];
    A[idx1] = A[idx2];
    A[idx2] = tmp;
}

/**
 * 主接口 - 原始稳定版本
 */
inline void launch_A_exchange_trailing(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* h_ipiv_rel,
    cudaStream_t stream = 0)
{
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    // 配置
    const int THREADS = 256;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks == 0) num_blocks = 1;
    
    dim3 grid(num_blocks);
    dim3 block(THREADS);

    // 逐个swap（保持顺序）
    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + h_ipiv_rel[k];

        // 跳过无效交换
        if (r1 == r2) continue;
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;

        swap_rows_kernel<<<grid, block, 0, stream>>>(
            dA, m, n, lda, r1, r2, j0, ib);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_exchange_buffers() {
}