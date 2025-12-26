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
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

// ============================================================================
// 批量行交换 kernel：对 trailing columns 应用 ib 次 pivot swap
// 每个线程负责一列 j（除去 panel 内的列），并顺序执行所有 k swap
// ============================================================================

__global__ void batch_swap_rows_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv_rel)
{
    // shared memory 缓存 pivot 信息
    extern __shared__ int s_ipiv[];

    // 协作加载 ipiv_rel 到 shared memory
    for (int k = threadIdx.x; k < ib; k += blockDim.x) {
        s_ipiv[k] = d_ipiv_rel[k];
    }
    __syncthreads();

    const int j_panel_begin = j0;
    const int j_panel_end   = j0 + ib;

    // 每个线程负责一列
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j >= n) return;
    if (j >= j_panel_begin && j < j_panel_end) return;

    const size_t col_offset = (size_t)j * (size_t)lda;

    // 对这一列顺序应用所有 ib 次行交换
    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + s_ipiv[k];

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
// 对 trailing 部分应用 pivot：device pivot 版本（无 H2D / malloc / free）
// ============================================================================

inline void launch_A_exchange_trailing_device_piv(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv_rel,
    cudaStream_t stream = 0)
{
    if (!dA || !d_ipiv_rel) return;
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    const int THREADS = 256;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks <= 0) num_blocks = 1;

    dim3 grid(num_blocks);
    dim3 block(THREADS);

    // shared memory 大小：存储 ib 个 int
    size_t shmem_size = sizeof(int) * (size_t)ib;

    batch_swap_rows_kernel<<<grid, block, shmem_size, stream>>>(
        dA, m, n, lda, j0, ib, d_ipiv_rel);

    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 兼容旧接口（如果你不想动其它地方，可以保留这个 wrapper）
// 注意：这里直接报错或 assert，比默默 malloc+memcpy 更合理
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
            "Use launch_A_exchange_trailing_device_piv() instead.\n");
    std::exit(EXIT_FAILURE);
}

inline void cleanup_exchange_buffers() {}
