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
 * 基础版本：在给定列区间 [col_begin, col_end) 内交换两行
 *   对所有 j ∈ [col_begin .. col_end):
 *       swap( A[row1, j], A[row2, j] )
 */
__global__ void swap_rows_cols_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int row1, int row2,
    int col_begin, int col_end)
{
    if (row1 == row2) return;
    if (row1 < 0 || row1 >= m || row2 < 0 || row2 >= m) return;
    if (col_begin >= col_end) return;
    if (col_begin >= n) return;

    int j = col_begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= col_end || j >= n) return;

    size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
    size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;

    half tmp = A[idx1];
    A[idx1]  = A[idx2];
    A[idx2]  = tmp;
}

/**
 * 改进结构的版本：单次 kernel 覆盖整行，但跳过 panel 列块
 *
 * 对所有 j ∈ [0 .. n)，若 j 不在 [j0 .. j0+ib):
 *     swap( A[row1, j], A[row2, j] )
 *
 * 这样可以用一次 kernel 同时完成“左侧列”和“右侧列”的行交换，
 * 避免之前对同一对 (row1,row2) 启动两次 kernel。
 */
__global__ void swap_rows_cols_kernel_skip_panel(
    half* __restrict__ A,
    int m, int n, int lda,
    int row1, int row2,
    int j0, int ib)
{
    if (row1 == row2) return;
    if (row1 < 0 || row1 >= m || row2 < 0 || row2 >= m) return;
    if (n <= 0) return;
    if (ib <= 0) {
        // 没有 panel，退化成对全行做交换
        int j = blockIdx.x * blockDim.x + threadIdx.x;
        if (j >= n) return;
        size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
        size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;
        half tmp = A[idx1];
        A[idx1]  = A[idx2];
        A[idx2]  = tmp;
        return;
    }

    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    int j_panel_begin = j0;
    int j_panel_end   = j0 + ib; // [j0, j0+ib) 为 panel 列块

    // 跳过 panel 列块（panel_TSLU 已经对这些列做过行交换）
    if (j >= j_panel_begin && j < j_panel_end) return;

    size_t idx1 = (size_t)row1 + (size_t)j * (size_t)lda;
    size_t idx2 = (size_t)row2 + (size_t)j * (size_t)lda;

    half tmp = A[idx1];
    A[idx1]  = A[idx2];
    A[idx2]  = tmp;
}

/**
 * launch_A_exchange_trailing
 *
 * 功能：在 panel_TSLU 之后，把该 panel 的 pivot 行交换
 *       补充传播到 panel 左侧和右侧所有列，从而实现
 *       “对整行做一次 row-swap”（panel 列已经在 panel_TSLU 中交换过）。
 *
 * 输入：
 *   dA         : 整个矩阵 (half, col-major)
 *   m, n, lda  : 行数, 列数, leading dimension
 *   j0         : 当前 panel 起始列
 *   ib         : panel 宽度
 *   h_ipiv_rel : host 端 pivot 相对位移数组, 长度 >= ib,
 *                第 k 列 pivot 行号 p = (j0 + k) + h_ipiv_rel[k]
 *
 * 逻辑：
 *   对 k = 0..ib-1：
 *      r1 = j0 + k
 *      r2 = r1 + h_ipiv_rel[k]
 *      若 r1!=r2，则对整行执行一次行交换：
 *         对所有 j∈[0..n)，但跳过 panel 列 [j0..j0+ib)
 *         swap( A[r1,j], A[r2,j] )
 *
 *   这样在一次 kernel 内同时完成原来“左侧列”和“右侧列”的行交换，
 *   panel 列 [j0..j0+ib-1] 仍然只在 panel_TSLU 内被交换一次。
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

    const int block_x = 256;  // 可以根据 profiling 适当调整

    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + h_ipiv_rel[k];

        if (r1 == r2) continue;
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;

        int ncols = n;
        if (ncols <= 0) continue;

        dim3 block(block_x);
        dim3 grid((ncols + block_x - 1) / block_x);

        // 单次 kernel 覆盖整行，内部自行跳过 panel 列块
        swap_rows_cols_kernel_skip_panel<<<grid, block, 0, stream>>>(
            dA, m, n, lda, r1, r2, j0, ib
        );
        CUDA_CHECK(cudaGetLastError());
    }
}
