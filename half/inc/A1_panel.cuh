#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/**
 * ============================================================================
 * A1_panel.cuh - 稳定版本（经过验证）
 * ============================================================================
 * 
 * 这是您原始的稳定实现，只做了最小的自适应优化
 * 
 * ============================================================================
 */

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err__), __FILE__, __LINE__);            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

static __device__ __forceinline__ half half_abs(half x) {
    return __habs(x);
}

// ============================================================================
// Pivot kernels - 完全保持原样
// ============================================================================

__global__ void panel_pivot_search_kernel(
    const half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val,
    int*   __restrict__ block_idx)
{
    extern __shared__ unsigned char smem[];
    half* s_val = reinterpret_cast<half*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    const int col = j0 + k;
    const int base_row = j0 + k;
    if (col >= m) return;

    half local_max_val = __float2half(0.0f);
    int  local_max_idx = base_row;

    const int tid = threadIdx.x;
    const int global_stride = blockDim.x * gridDim.x;

    for (int idx = base_row + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col * lda];
        half v = half_abs(a);
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = idx;
        }
    }

    s_val[tid] = local_max_val;
    s_idx[tid] = local_max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_val[blockIdx.x] = s_val[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

__global__ void panel_pivot_reduce_kernel(
    const half* __restrict__ block_val,
    const int*   __restrict__ block_idx,
    int num_blocks,
    int j0, int k,
    int* __restrict__ d_ipiv_rel)
{
    half max_val = 0.0f;
    int  max_idx = j0 + k;
    int  tid = threadIdx.x;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        half v = block_val[i];
        int  r = block_idx[i];
        if (v > max_val) {
            max_val = v;
            max_idx = r;
        }
    }

    extern __shared__ unsigned char smem[];
    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        int pivot_row = s_idx[0];
        int rel = pivot_row - (j0 + k);
        d_ipiv_rel[k] = rel;
    }
}

__global__ void panel_row_swap_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    int rel = d_ipiv_rel[k];
    int pivot_row = (j0 + k) + rel;
    int row_k     = j0 + k;

    if (pivot_row == row_k) return;

    int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j0 + ib) return;

    size_t col_offset = (size_t)j * lda;
    half tmp = A[row_k + col_offset];
    A[row_k + col_offset] = A[pivot_row + col_offset];
    A[pivot_row + col_offset] = tmp;
}

__global__ void panel_column_scale_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    half pivot = A[col_k + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) return;

    int r = col_k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;

    half val = A[r + (size_t)col_k * lda];
    A[r + (size_t)col_k * lda] = val / pivot;
}

__global__ void panel_update_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k)
{
    int col_k = j0 + k;
    int row_k = j0 + k;
    int col_start = col_k + 1;
    int col_end   = j0 + ib;

    if (col_k >= m) return;

    int r = row_k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int c = col_start + blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= m || c >= col_end) return;

    size_t col_k_off = (size_t)col_k * lda;
    size_t col_c_off = (size_t)c * lda;

    half L = A[r + col_k_off];
    half U = A[row_k + col_c_off];
    half A_h = A[r + col_c_off];
    
    half res = __hsub(A_h, __hmul(L, U));
    A[r + col_c_off] = res;
}

// ============================================================================
// 主入口 - 使用原始稳定实现
// ============================================================================

inline void launch_panel_TSLU(
    half* A,
    int   m,
    int   lda,
    int   j0,
    int   ib,
    int   uc,
    int*  d_ipiv_rel,
    cudaStream_t stream)
{
    if (!A || !d_ipiv_rel) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= m) return;

    const int threads_pivot = 256;
    const int rows_per_block = 128;

    int max_rows = m - j0;
    if (max_rows <= 0) return;

    int num_blocks_pivot = (max_rows + rows_per_block - 1) / rows_per_block;
    if (num_blocks_pivot <= 0) num_blocks_pivot = 1;

    half* d_block_val = nullptr;
    int*  d_block_idx = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_block_val, sizeof(half) * num_blocks_pivot, stream));
    CUDA_CHECK(cudaMallocAsync(&d_block_idx, sizeof(int) * num_blocks_pivot, stream));

    dim3 grid_row_swap((unsigned)((ib + 255) / 256));
    dim3 block_row_swap(256);

    int tile_c = (uc > 0) ? uc : 8;
    if (tile_c > 32) tile_c = 32;
    dim3 block_upd(tile_c, 8);

    for (int k = 0; k < ib; ++k) {
        int col = j0 + k;
        if (col >= m) break;

        // 1) pivot 搜索
        {
            dim3 grid_pivot(num_blocks_pivot);
            size_t shmem = sizeof(half) * threads_pivot + sizeof(int) * threads_pivot;
            panel_pivot_search_kernel<<<grid_pivot, threads_pivot, shmem, stream>>>(
                A, m, lda, j0, k, d_block_val, d_block_idx);
        }

        // 2) pivot 规约
        {
            dim3 grid_red(1);
            dim3 block_red(128);
            int threads_red = block_red.x;
            size_t shmem_red = sizeof(half) * threads_red + sizeof(int) * threads_red;
            panel_pivot_reduce_kernel<<<grid_red, block_red, shmem_red, stream>>>(
                d_block_val, d_block_idx, num_blocks_pivot, j0, k, d_ipiv_rel);
        }

        // 3) 行交换
        {
            panel_row_swap_kernel<<<grid_row_swap, block_row_swap, 0, stream>>>(
                A, m, lda, j0, ib, k, d_ipiv_rel);
        }

        // 4) 列缩放
        {
            int rows_remaining = m - (j0 + k + 1);
            if (rows_remaining > 0) {
                int blocks_scale = (rows_remaining + 255) / 256;
                dim3 grid_scale(blocks_scale);
                dim3 block_scale(256);
                panel_column_scale_kernel<<<grid_scale, block_scale, 0, stream>>>(
                    A, m, lda, j0, k, d_ipiv_rel);
            }
        }

        // 5) panel 更新
        {
            int rows_rem = m - (j0 + k + 1);
            int cols_rem = ib - (k + 1);
            if (rows_rem > 0 && cols_rem > 0) {
                int grid_x = (cols_rem + block_upd.x - 1) / block_upd.x;
                int grid_y = (rows_rem + block_upd.y - 1) / block_upd.y;
                dim3 grid_upd(grid_x, grid_y);
                panel_update_kernel<<<grid_upd, block_upd, 0, stream>>>(
                    A, m, lda, j0, ib, k);
            }
        }
    }

    CUDA_CHECK(cudaFreeAsync(d_block_val, stream));
    CUDA_CHECK(cudaFreeAsync(d_block_idx, stream));
    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_panel_buffers() {
}