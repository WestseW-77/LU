#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

// ============================================================================
// 批量行交换 kernel：对 trailing columns 应用 ib 次 pivot swap
// pivot 来自全局 ipiv：d_ipiv[j0 + k] = pivot_row + 1 (1-based)
// 内部转成 0-based 执行 swap
// ============================================================================
__global__ void batch_swap_rows_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv) // global ipiv (1-based)
{
    extern __shared__ int s_ipiv[];

    // 协作加载当前 panel 的 pivot 行号到 shared memory（转为 0-based）
    for (int k = threadIdx.x; k < ib; k += blockDim.x) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    const int j_panel_begin = j0;
    const int j_panel_end   = j0 + ib;

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    if (j >= j_panel_begin && j < j_panel_end) return;

    const size_t col_offset = (size_t)j * (size_t)lda;

    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = s_ipiv[k]; // absolute pivot row, 0-based

        if (r1 == r2) continue;
        if ((unsigned)r1 >= (unsigned)m || (unsigned)r2 >= (unsigned)m) continue;

        const size_t idx1 = (size_t)r1 + col_offset;
        const size_t idx2 = (size_t)r2 + col_offset;

        const half tmp = A[idx1];
        A[idx1] = A[idx2];
        A[idx2] = tmp;
    }
}

// ============================================================================
// 对 trailing 部分应用 pivot：device pivot 版本（使用 global ipiv，无 malloc/free）
// ============================================================================
inline void launch_A_exchange_trailing_device_piv(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,   // global ipiv (1-based)
    cudaStream_t stream = 0)
{
    if (!dA || !d_ipiv) return;
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    const int THREADS = 256;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks <= 0) num_blocks = 1;

    dim3 grid(num_blocks);
    dim3 block(THREADS);

    size_t shmem_size = sizeof(int) * (size_t)ib;

    batch_swap_rows_kernel<<<grid, block, shmem_size, stream>>>(
        dA, m, n, lda, j0, ib, d_ipiv);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 兼容旧接口：直接报错，避免偷偷 malloc/memcpy
// ============================================================================
inline void launch_A_exchange_trailing(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* h_ipiv_rel,
    cudaStream_t stream = 0)
{
    (void)dA; (void)m; (void)n; (void)lda;
    (void)j0; (void)ib;
    (void)h_ipiv_rel;
    (void)stream;

    fprintf(stderr,
            "[A_exchange] ERROR: launch_A_exchange_trailing(host piv) is removed. "
            "Use launch_A_exchange_trailing_device_piv(global ipiv, 1-based) instead.\n");
    std::exit(EXIT_FAILURE);
}

inline void cleanup_exchange_buffers() {}
