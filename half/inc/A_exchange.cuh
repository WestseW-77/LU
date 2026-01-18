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

// 默认 64：8192 列 -> 128 blocks，4090 的 128 SM 终于不闲着了
#ifndef EXCH_THREADS
#define EXCH_THREADS 64
#endif

template<int THREADS>
__global__ void batch_swap_rows_kernel_v2(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv) // global ipiv (1-based)
{
    extern __shared__ int s_ipiv[];

    for (int k = threadIdx.x; k < ib; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1; // to 0-based
    }
    __syncthreads();

    // 仍然对所有列做（除了 panel 本身），保持你现在语义不变
    const int j_panel_begin = j0;
    const int j_panel_end   = j0 + ib;

    const int j = (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= n) return;
    if (j >= j_panel_begin && j < j_panel_end) return;

    const size_t col_offset = (size_t)j * (size_t)lda;

    // 简单 unroll 降低一点 loop overhead（不会神迹，但便宜）
    int k = 0;
    for (; k + 3 < ib; k += 4) {
        int r1, r2;

        r1 = j0 + (k + 0); r2 = s_ipiv[k + 0];
        if (r1 != r2 && (unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
            size_t idx1 = (size_t)r1 + col_offset;
            size_t idx2 = (size_t)r2 + col_offset;
            half tmp = A[idx1]; A[idx1] = A[idx2]; A[idx2] = tmp;
        }

        r1 = j0 + (k + 1); r2 = s_ipiv[k + 1];
        if (r1 != r2 && (unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
            size_t idx1 = (size_t)r1 + col_offset;
            size_t idx2 = (size_t)r2 + col_offset;
            half tmp = A[idx1]; A[idx1] = A[idx2]; A[idx2] = tmp;
        }

        r1 = j0 + (k + 2); r2 = s_ipiv[k + 2];
        if (r1 != r2 && (unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
            size_t idx1 = (size_t)r1 + col_offset;
            size_t idx2 = (size_t)r2 + col_offset;
            half tmp = A[idx1]; A[idx1] = A[idx2]; A[idx2] = tmp;
        }

        r1 = j0 + (k + 3); r2 = s_ipiv[k + 3];
        if (r1 != r2 && (unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
            size_t idx1 = (size_t)r1 + col_offset;
            size_t idx2 = (size_t)r2 + col_offset;
            half tmp = A[idx1]; A[idx1] = A[idx2]; A[idx2] = tmp;
        }
    }

    for (; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = s_ipiv[k];
        if (r1 == r2) continue;
        if ((unsigned)r1 >= (unsigned)m || (unsigned)r2 >= (unsigned)m) continue;

        size_t idx1 = (size_t)r1 + col_offset;
        size_t idx2 = (size_t)r2 + col_offset;
        half tmp = A[idx1]; A[idx1] = A[idx2]; A[idx2] = tmp;
    }
}

inline void launch_A_exchange_trailing_device_piv(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,
    cudaStream_t stream = 0)
{
    if (!dA || !d_ipiv) return;
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    constexpr int THREADS = EXCH_THREADS;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks < 1) num_blocks = 1;

    size_t shmem = sizeof(int) * (size_t)ib;

    batch_swap_rows_kernel_v2<THREADS><<<num_blocks, THREADS, shmem, stream>>>(
        dA, m, n, lda, j0, ib, d_ipiv);

    CUDA_CHECK(cudaGetLastError());
}
