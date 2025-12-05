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
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

__global__ void swap_rows_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int row1, int row2,
    int j0, int ib)
{
    const int j_panel_begin = j0;
    const int j_panel_end = j0 + ib;
    
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n) return;
    if (j >= j_panel_begin && j < j_panel_end) return;
    
    const size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
    const size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;
    
    const half tmp = A[idx1];
    A[idx1] = A[idx2];
    A[idx2] = tmp;
}

inline void launch_A_exchange_trailing(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* h_ipiv_rel,
    cudaStream_t stream = 0)
{
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    const int THREADS = 256;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks == 0) num_blocks = 1;
    
    dim3 grid(num_blocks);
    dim3 block(THREADS);

    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + h_ipiv_rel[k];

        if (r1 == r2) continue;
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;

        swap_rows_kernel<<<grid, block, 0, stream>>>(
            dA, m, n, lda, r1, r2, j0, ib);
    }
    
    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_exchange_buffers() {
}