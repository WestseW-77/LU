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

// 批量行交换 kernel
__global__ void batch_swap_rows_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv_rel)
{
    // 使用 shared memory 缓存 pivot 信息
    extern __shared__ int s_ipiv[];
    
    // 协作加载 ipiv_rel 到 shared memory
    for (int k = threadIdx.x; k < ib; k += blockDim.x) {
        s_ipiv[k] = d_ipiv_rel[k];
    }
    __syncthreads();
    
    const int j_panel_begin = j0;
    const int j_panel_end = j0 + ib;
    
    // 每个线程负责一列
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (j >= n) return;
    if (j >= j_panel_begin && j < j_panel_end) return;
    
    const size_t col_offset = (size_t)j * (size_t)lda;
    
    // 对这一列，顺序应用所有 ib 次行交换
    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + s_ipiv[k];
        
        if (r1 == r2) continue;
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;
        
        // 交换这一列的两个元素
        const size_t idx1 = (size_t)r1 + col_offset;
        const size_t idx2 = (size_t)r2 + col_offset;
        
        const half tmp = A[idx1];
        A[idx1] = A[idx2];
        A[idx2] = tmp;
    }
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

    // 分配设备端内存并拷贝 pivot 数组
    int* d_ipiv_rel = nullptr;
    cudaError_t err = cudaMallocAsync(&d_ipiv_rel, sizeof(int) * ib, stream);
    if (err != cudaSuccess) {
        // 分配失败，回退到原始版本
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
        return;
    }
    
    // H2D 拷贝
    cudaMemcpyAsync(d_ipiv_rel, h_ipiv_rel, sizeof(int) * ib, 
                    cudaMemcpyHostToDevice, stream);
    
    // 使用批量交换 kernel
    const int THREADS = 256;
    int num_blocks = (n + THREADS - 1) / THREADS;
    if (num_blocks == 0) num_blocks = 1;
    
    dim3 grid(num_blocks);
    dim3 block(THREADS);
    
    // shared memory 大小：存储 ib 个 int
    size_t shmem_size = sizeof(int) * ib;
    
    batch_swap_rows_kernel<<<grid, block, shmem_size, stream>>>(
        dA, m, n, lda, j0, ib, d_ipiv_rel);
    
    CUDA_CHECK(cudaGetLastError());
    
    // 释放临时内存
    cudaFreeAsync(d_ipiv_rel, stream);
}

inline void cleanup_exchange_buffers() {
}