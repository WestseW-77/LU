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

// -----------------------------------------------------------------------------
// Tunables (compile-time)
// -----------------------------------------------------------------------------
// 线程块按列并行：每个 thread 负责 1 列（你原始思路）。
// 建议你扫：64 / 128 / 256
#ifndef EXCH_THREADS
#define EXCH_THREADS 256
#endif

// unroll 深度（建议扫：4 / 8 / 16）
// 过大可能增寄存器导致 occupancy 下降
#ifndef EXCH_UNROLL
#define EXCH_UNROLL 8
#endif

// ib==256 专用路径
#ifndef EXCH_USE_IB256_SPECIALIZED
#define EXCH_USE_IB256_SPECIALIZED 1
#endif

// ib==128 专用路径（当你把 hgetrf panel_width 设为 128 时很常用）
#ifndef EXCH_USE_IB128_SPECIALIZED
#define EXCH_USE_IB128_SPECIALIZED 1
#endif

// 如果你确认所有调用都满足：m >= n 且 (j0+ib)<=n，那么 row1 永远 < m
// 可把它设为 1，少一层判断（可能带来几个百分点）
// 建议先 0 保守跑通，再开 1 测性能
#ifndef EXCH_ASSUME_PANELROWS_IN_RANGE
#define EXCH_ASSUME_PANELROWS_IN_RANGE 0
#endif

// -----------------------------------------------------------------------------
// Generic kernel: works for any ib (uses dynamic shared)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_generic(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    extern __shared__ int s_ipiv[];

    for (int k = threadIdx.x; k < ib; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    // 注意：为了省掉“panel overlap”的分支，这个 kernel 假设 [col_begin, col_end)
    // 不与 [j0, j0+ib) 重叠。launcher 会在必要时自动 split。

    const size_t col_offset = (size_t)j * (size_t)lda;

    for (int k = 0; k < ib; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= ib) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Specialized kernel: ib == 256 (static shared, fixed loop bounds)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_ib256(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    __shared__ int s_ipiv[256];

    for (int k = threadIdx.x; k < 256; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    // 同 generic：假设 range 不与 panel 列重叠（由 launcher 保证）

    const size_t col_offset = (size_t)j * (size_t)lda;

    // fixed 256 steps
    for (int k = 0; k < 256; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= 256) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

#if EXCH_ASSUME_PANELROWS_IN_RANGE
            // r1 guaranteed in range
            if ((unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#else
            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#endif
        }
    }
}

// -----------------------------------------------------------------------------
// Specialized kernel: ib == 128 (static shared, fixed loop bounds)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_ib128(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    __shared__ int s_ipiv[128];

    for (int k = threadIdx.x; k < 128; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    const size_t col_offset = (size_t)j * (size_t)lda;

    for (int k = 0; k < 128; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= 128) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

#if EXCH_ASSUME_PANELROWS_IN_RANGE
            if ((unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#else
            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#endif
        }
    }
}

// -----------------------------------------------------------------------------
// Public launchers (保持你现有 hgetrf 接口不变)
// -----------------------------------------------------------------------------
inline void launch_A_exchange_trailing_device_piv_range(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,
    int col_begin,
    int col_end,
    cudaStream_t stream = 0)
{
    if (!dA || !d_ipiv) return;
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    if (col_begin < 0) col_begin = 0;
    if (col_end > n) col_end = n;
    if (col_begin >= col_end) return;

    const int panel_begin = j0;
    const int panel_end   = j0 + ib;

    // 内部 helper：假设 [cb, ce) 不与 [panel_begin, panel_end) 重叠
    auto launch_one = [&](int cb, int ce) {
        if (cb >= ce) return;

        constexpr int THREADS = EXCH_THREADS;
        constexpr int UNROLL  = EXCH_UNROLL;

        int cols = ce - cb;
        int num_blocks = (cols + THREADS - 1) / THREADS;
        if (num_blocks < 1) num_blocks = 1;

#if EXCH_USE_IB256_SPECIALIZED
        if (ib == 256) {
            batch_swap_rows_kernel_range_ib256<THREADS, UNROLL>
                <<<num_blocks, THREADS, 0, stream>>>(
                    dA, m, n, lda, j0, d_ipiv, cb, ce);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#endif

#if EXCH_USE_IB128_SPECIALIZED
        if (ib == 128) {
            batch_swap_rows_kernel_range_ib128<THREADS, UNROLL>
                <<<num_blocks, THREADS, 0, stream>>>(
                    dA, m, n, lda, j0, d_ipiv, cb, ce);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#endif

        // generic fallback
        size_t shmem = sizeof(int) * (size_t)ib;
        batch_swap_rows_kernel_range_generic<THREADS, UNROLL>
            <<<num_blocks, THREADS, shmem, stream>>>(
                dA, m, n, lda, j0, ib, d_ipiv, cb, ce);
        CUDA_CHECK(cudaGetLastError());
    };

    // 如果 range 和 panel 列有交叠：自动 split 成左右两段（避免 kernel 内部分支）
    const bool overlap = !(col_end <= panel_begin || col_begin >= panel_end);
    if (overlap) {
        int left_end = (col_end < panel_begin) ? col_end : panel_begin;
        int right_begin = (col_begin > panel_end) ? col_begin : panel_end;

        launch_one(col_begin, left_end);
        launch_one(right_begin, col_end);
        return;
    }

    // no overlap
    launch_one(col_begin, col_end);
}

inline void launch_A_exchange_trailing_device_piv(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,
    cudaStream_t stream = 0)
{
    // full range [0, n)
    launch_A_exchange_trailing_device_piv_range(
        dA, m, n, lda, j0, ib, d_ipiv, 0, n, stream);
}
