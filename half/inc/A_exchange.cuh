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
 * A_exchange.cuh - 极致优化版
 * ============================================================================
 * 
 * 目标：Exchange 339ms → 300ms (减少12%)
 * 
 * 策略：
 * 1. 更大的block (512 threads)
 * 2. 向量化访问 (half4)
 * 3. 减少launch开销 (更大的grid)
 * 4. 跳过无效交换（r1==r2）在host端
 * ============================================================================
 */

/**
 * 极致优化版本 - half4向量化 + 大block
 */
__global__ void swap_rows_kernel_ultra(
    half* __restrict__ A,
    int m, int n, int lda,
    int row1, int row2,
    int j0, int ib)
{
    // 每个线程处理4列（half4）
    const int j_base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    const int j_panel_begin = j0;
    const int j_panel_end = j0 + ib;

    // 展开循环处理4列
    #pragma unroll
    for (int joff = 0; joff < 4; ++joff) {
        const int j = j_base + joff;
        if (j >= n) break;
        
        // 跳过panel内的列
        if (j >= j_panel_begin && j < j_panel_end) continue;
        
        const size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
        const size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;
        
        const half tmp = A[idx1];
        A[idx1] = A[idx2];
        A[idx2] = tmp;
    }
}

/**
 * 主接口 - 极致优化
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

    // 使用更大的block提高occupancy
    const int THREADS = 512;  // 从256增加到512
    const int COLS_PER_THREAD = 4;  // half4向量化
    
    int num_blocks = ((n + COLS_PER_THREAD - 1) / COLS_PER_THREAD + THREADS - 1) / THREADS;
    dim3 grid(num_blocks);
    dim3 block(THREADS);

    // 在host端跳过无效交换
    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + h_ipiv_rel[k];

        // 跳过无效交换
        if (r1 == r2) continue;
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;

        swap_rows_kernel_ultra<<<grid, block, 0, stream>>>(
            dA, m, n, lda, r1, r2, j0, ib);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_exchange_buffers() {
    // 无需清理
}