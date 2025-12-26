#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__), __FILE__, __LINE__);           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// 初始化 piv_rows[i] = i
__global__ void init_piv_rows_kernel(int* piv_rows, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) piv_rows[i] = i;
}

// 顺序应用一个 panel 的 pivot swap 到 piv_rows
// 注意：必须严格按 k=0..ib-1 顺序应用，否则结果会变
__global__ void apply_panel_pivots_to_pivrows_kernel(
    int* piv_rows,
    int m,
    int j0,
    int ib,
    const int* __restrict__ d_ipiv_rel)
{
    if (threadIdx.x != 0) return;

    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + d_ipiv_rel[k];

        if (r1 == r2) continue;
        if ((unsigned)r1 >= (unsigned)m || (unsigned)r2 >= (unsigned)m) continue;

        int tmp = piv_rows[r1];
        piv_rows[r1] = piv_rows[r2];
        piv_rows[r2] = tmp;
    }
}

inline void launch_init_piv_rows(int* d_piv_rows, int m, cudaStream_t stream = 0)
{
    if (!d_piv_rows || m <= 0) return;
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    init_piv_rows_kernel<<<blocks, threads, 0, stream>>>(d_piv_rows, m);
    CUDA_CHECK(cudaGetLastError());
}

inline void launch_apply_panel_pivots_to_pivrows(
    int* d_piv_rows,
    int m,
    int j0,
    int ib,
    const int* d_ipiv_rel,
    cudaStream_t stream = 0)
{
    if (!d_piv_rows || !d_ipiv_rel || ib <= 0) return;
    apply_panel_pivots_to_pivrows_kernel<<<1, 32, 0, stream>>>(
        d_piv_rows, m, j0, ib, d_ipiv_rel);
    CUDA_CHECK(cudaGetLastError());
}
