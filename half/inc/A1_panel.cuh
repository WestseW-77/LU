#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <float.h>

/**
 * ============================================================================
 * A1_panel.cuh - Panel LU分解（分块LU的第一步）
 * ============================================================================
 * 
 * 【整体功能】
 * 对矩阵的一个"Panel"（垂直条带）执行LU分解
 * Panel是指从j0列开始的ib列宽度的区域
 * 
 * 【算法流程】
 * 对每一列k（在Panel内）：
 *   1. 找主元（pivot）：在第k列的下三角部分找绝对值最大的元素
 *   2. 行交换：将主元所在行与第k行交换
 *   3. 列缩放：第k列下方的元素除以主元
 *   4. 矩阵更新：更新Panel内右下角的子矩阵（Schur complement）
 * 
 * 【优化特点】
 * 1. Warp shuffle优化 - 快速并行归约找主元
 * 2. 4×32 tile配置 - 经过验证的最优块大小
 * 3. 动态block配置 - 根据矩阵大小自适应
 * 
 * ============================================================================
 */

#ifndef WARP_SIZE
#define WARP_SIZE 32  // CUDA warp的标准大小
#endif

// CUDA错误检查宏
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

// 半精度浮点数的绝对值函数
static __device__ __forceinline__ half half_abs(half x) {
    return __habs(x);
}

// 在整列中找到最大值的 kernel，这是单 block 内的最大值，会把数值和绝对行号存入全局内存数组中，再由另一个 kernel 合成出一个多个 block 规约的最大值
__global__ void optimized_panel_pivot_search_kernel(
    const half* __restrict__ A,      // restrict表示指针不重叠，允许编译器优化
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val,
    int*   __restrict__ block_idx)
{
    // 动态共享内存：存储每个warp的归约结果
    extern __shared__ unsigned char smem[];
    half* s_val = reinterpret_cast<half*>(smem);
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    // 绝对列号
    const int col = j0 + k;
    // 起始行号
    const int base_row = j0 + k;
    if (col >= m) 
        return;

    // 每个线程的局部最大值
    half local_max_val = __float2half(0.0f);
    int  local_max_idx = base_row;

    const int tid = threadIdx.x;
    // 每个线程读取的多个数据的步长
    const int global_stride = blockDim.x * gridDim.x;

    // 每个线程读取所有自己的数据并选出最大的
    for (int idx = base_row + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col * lda];
        half v = half_abs(a);
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = idx;
        }
    }

    // warp 内的线程 id
    const int lane = tid % WARP_SIZE; 
    // 线程的 warp id
    const int warp_id = tid / WARP_SIZE;
    
    // warp 内规约出最大值
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        half other_val = __shfl_down_sync(0xffffffff, local_max_val, offset);
        int other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max_val) {
            local_max_val = other_val;
            local_max_idx = other_idx;
        }
    }

    // 每个 warp 的第 0 个线程将结果写入共享内存
    const int num_warps = blockDim.x / WARP_SIZE;
    if (lane == 0) {
        s_val[warp_id] = local_max_val;
        s_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    // warp0 对共享内存内的结果再次进行规约，得到最大值
    if (warp_id == 0) {
        half warp_max = (lane < num_warps) ? s_val[lane] : __float2half(0.0f);
        int warp_idx = (lane < num_warps) ? s_idx[lane] : base_row;
        
        // warp shuffle 直接交换寄存器数据，拿到最大值
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            half other_val = __shfl_down_sync(0xffffffff, warp_max, offset);
            int other_idx = __shfl_down_sync(0xffffffff, warp_idx, offset);
            if (other_val > warp_max) {
                warp_max = other_val;
                warp_idx = other_idx;
            }
        }
        
        // 写入全局内存
        if (lane == 0) {
            block_val[blockIdx.x] = warp_max;
            block_idx[blockIdx.x] = warp_idx;
        }
    }
}

/**
 * ============================================================================
 * Kernel 2: panel_pivot_reduce_kernel
 * ============================================================================
 * 【功能】对多个block的pivot搜索结果进行最终归约
 * 
 * 【输入】
 * - block_val: 每个block找到的最大值数组
 * - block_idx: 每个block找到的最大值行索引数组
 * - num_blocks: block数量
 * 
 * 【输出】
 * - d_ipiv_rel[k]: 相对于对角线的pivot偏移量
 */
__global__ void panel_pivot_reduce_kernel(
    const half* __restrict__ block_val,
    const int*   __restrict__ block_idx,
    int num_blocks,
    int j0, int k,
    int* __restrict__ d_ipiv_rel)
{
    half max_val = __float2half(0.0f);
    int  max_idx = j0 + k;  // 默认为对角线元素
    int  tid = threadIdx.x;

    // 每个线程扫描一部分blocks
    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        half v = block_val[i];
        int  r = block_idx[i];
        if (v > max_val) {
            max_val = v;
            max_idx = r;
        }
    }

    // 使用共享内存进行block内归约
    extern __shared__ unsigned char smem[];
    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    s_val[tid] = max_val;
    s_idx[tid] = max_idx;
    __syncthreads();

    // 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // 第0个线程写入最终结果
    if (tid == 0) {
        int pivot_row = s_idx[0];
        int rel = pivot_row - (j0 + k);  // 计算相对偏移
        d_ipiv_rel[k] = rel;
    }
}

/**
 * ============================================================================
 * Kernel 3: panel_row_swap_kernel
 * ============================================================================
 * 【功能】在Panel内交换两行
 * 
 * 【操作】交换row_k和pivot_row在Panel列范围内的元素
 */
__global__ void panel_row_swap_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,      // Panel起始列和宽度
    int k,               // 当前列在Panel内的索引
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    int rel = d_ipiv_rel[k];
    int pivot_row = (j0 + k) + rel;  // pivot所在行
    int row_k     = j0 + k;          // 当前对角线行

    if (pivot_row == row_k) return;  // 不需要交换

    // 每个线程负责Panel中的一列
    int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j0 + ib) return;

    // 交换两行在第j列的元素
    size_t col_offset = (size_t)j * lda;
    half tmp = A[row_k + col_offset];
    A[row_k + col_offset] = A[pivot_row + col_offset];
    A[pivot_row + col_offset] = tmp;
}

/**
 * ============================================================================
 * Kernel 4: panel_column_scale_kernel
 * ============================================================================
 * 【功能】将第k列对角线下方的元素除以pivot（形成L矩阵的一列）
 * 
 * 【公式】A[r, col_k] = A[r, col_k] / A[col_k, col_k]  (for r > col_k)
 */
__global__ void panel_column_scale_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    // 读取主元
    half pivot = A[col_k + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) return;  // 避免除零

    // 每个线程处理对角线下方的一个元素
    int r = col_k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;

    half val = A[r + (size_t)col_k * lda];
    A[r + (size_t)col_k * lda] = val / pivot;  // L[r,k] = A[r,k] / U[k,k]
}

/**
 * ============================================================================
 * Kernel 5: panel_update_kernel
 * ============================================================================
 * 【功能】更新Panel内右下角的子矩阵（Schur complement更新）
 * 
 * 【公式】A[r,c] -= L[r,k] * U[k,c]  
 *         其中 r > row_k, c > col_k, c在Panel内
 * 
 * 【这是LU分解的核心计算】
 */
__global__ void panel_update_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k)
{
    int col_k = j0 + k;
    int row_k = j0 + k;
    int col_start = col_k + 1;  // 更新从下一列开始
    int col_end   = j0 + ib;    // 到Panel结束

    if (col_k >= m) return;

    // 每个线程负责一个(r,c)位置
    int r = row_k + 1 + blockIdx.y * blockDim.y + threadIdx.y;
    int c = col_start + blockIdx.x * blockDim.x + threadIdx.x;

    if (r >= m || c >= col_end) return;

    size_t col_k_off = (size_t)col_k * lda;
    size_t col_c_off = (size_t)c * lda;

    half L = A[r + col_k_off];        // L矩阵元素
    half U = A[row_k + col_c_off];    // U矩阵元素
    half A_h = A[r + col_c_off];      // 当前值
    
    // Schur complement: A[r,c] -= L[r,k] * U[k,c]
    half res = __hsub(A_h, __hmul(L, U));
    A[r + col_c_off] = res;
}

/**
 * ============================================================================
 * 主启动函数: launch_panel_TSLU
 * ============================================================================
 * 【功能】对Panel执行完整的LU分解
 * 
 * 【输入】
 * - A: 矩阵（列主序）
 * - m, n, lda: 矩阵维度和leading dimension
 * - j0: Panel起始列
 * - ib: Panel宽度（block size）
 * - uc: 未使用的参数
 * 
 * 【输出】
 * - d_ipiv_rel: pivot索引数组（相对偏移）
 * 
 * 【算法流程】
 * for k = 0 to ib-1:
 *   1. 找pivot（多block并行搜索 + 单block归约）
 *   2. 行交换（仅在Panel内）
 *   3. 列缩放（形成L的一列）
 *   4. 矩阵更新（更新右下角子矩阵）
 */
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
    // 输入验证
    if (!A || !d_ipiv_rel) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) 
        return;
    if (j0 < 0 || j0 >= m) 
        return;

    // 本次 panel 分解需要处理的行数
    const int m_effective = m - j0;
    if (m_effective <= 0) return;

    // pivot 单 block 线程数 [fix] 目前来看 512 会比 256 在大 size 上快一点点
    const int threads_pivot = 256;
    int rows_per_block;
    
    // 大矩阵每个 block 处理的行数会增加
    if (m_effective >= 24576) {
        rows_per_block = 1024;
    } else if (m_effective >= 12288) {
        rows_per_block = 512;
    } else if (m_effective >= 4096) {
        rows_per_block = 256;
    } else {
        rows_per_block = 128;
    }
    
    // 目前来看哪怕是 65536 行，也只是 64 个 block 在 pivot 上 [fix]
    int num_blocks_pivot = (m_effective + rows_per_block - 1) / rows_per_block;
    if (num_blocks_pivot <= 0) 
        num_blocks_pivot = 1;
    if (num_blocks_pivot > 64) 
        num_blocks_pivot = 64;

    // pivot 结果内存分配
    half* d_block_val = nullptr;
    int*  d_block_idx = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_block_val, sizeof(half) * num_blocks_pivot, stream));
    CUDA_CHECK(cudaMallocAsync(&d_block_idx, sizeof(int) * num_blocks_pivot, stream));

    // 交换 kernel 配置
    dim3 grid_row_swap((ib + 255) / 256);
    dim3 block_row_swap(256);

    // update kernel 配置 [fix]
    int tile_c = 4;   // 列方向tile大小
    int tile_r = 32;  // 行方向tile大小
    dim3 block_upd(tile_c, tile_r);

    // 每次执行一列的具体高斯消元法
    for (int k = 0; k < ib; ++k) {
        int col = j0 + k;
        if (col >= m) 
            break;

        // pivot 搜索
        {
            dim3 grid_pivot(num_blocks_pivot);
            size_t shmem = sizeof(half) * threads_pivot + sizeof(int) * threads_pivot;
            optimized_panel_pivot_search_kernel<<<grid_pivot, threads_pivot, shmem, stream>>>(
                A, m, lda, j0, k, d_block_val, d_block_idx);
        }

        // pivot 归约
        {
            dim3 grid_red(1);
            // 这里这个 reduce 的 block 内的 thread 数也是一个可以根据前面来调整的数值 
            dim3 block_red(64);
            int threads_red = block_red.x;
            size_t shmem_red = sizeof(half) * threads_red + sizeof(int) * threads_red;
            panel_pivot_reduce_kernel<<<grid_red, block_red, shmem_red, stream>>>(
                d_block_val, d_block_idx, num_blocks_pivot, j0, k, d_ipiv_rel);
        }

        // panel 内的行交换
        {
            panel_row_swap_kernel<<<grid_row_swap, block_row_swap, 0, stream>>>(
                A, m, lda, j0, ib, k, d_ipiv_rel);
        }

        // 高斯消元
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

        // panel 内的 update
        {
            int rows_rem = m - (j0 + k + 1);     // 剩余行数
            int cols_rem = ib - (k + 1);         // Panel内剩余列数
            if (rows_rem > 0 && cols_rem > 0) {
                int grid_x = (cols_rem + tile_c - 1) / tile_c;
                int grid_y = (rows_rem + tile_r - 1) / tile_r;
                dim3 grid_upd(grid_x, grid_y);
                panel_update_kernel<<<grid_upd, block_upd, 0, stream>>>(
                    A, m, lda, j0, ib, k);
            }
        }
    }

    // 释放临时内存
    CUDA_CHECK(cudaFreeAsync(d_block_val, stream));
    CUDA_CHECK(cudaFreeAsync(d_block_idx, stream));
    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_panel_buffers() {
}