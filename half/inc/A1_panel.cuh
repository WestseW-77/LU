#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <type_traits>

/**
 *
 * 设计思路：
 *   每个 panel 内部对列 k = 0..ib-1 依次执行如下 4 步：
 *     1) pivot 搜索：在全局内 [j0+k, m) 行范围上做绝对值最大元搜索，多 block 并行；
 *     2) 写入相对 pivot：把 pivot 行号减去 (j0+k) 写入 d_ipiv_rel[k]；
 *     3) panel 内行交换：仅在列 [j0, j0+ib) 范围内交换 pivot 行与 (j0+k) 行；
 *     4) 列缩放 + 面板更新：对列 j0+k 做 L 列缩放，再用该列更新 panel 内右侧列。
 *
 * 说明：
 *   这里为了突出“多 block 并行”的结构，尽量使用简单、直观的 kernel 组合，
 *   没有再引入复杂的模板 IB/UC 调度。原来的 uc 参数仍然保留在接口中，
 *   但在实现里暂未深度利用，只预留了按列方向做小块的可能。
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

// 简单的 half 绝对值辅助函数
static __device__ __forceinline__ half half_abs(half x) {
#if __CUDA_ARCH__ >= 530
    return fabsf(x);
#else
    return fabsf(x);
#endif
}

/*
 * kernel 1: 对单列 j0+k 做分块 pivot 搜索
 *
 * 输入:
 *   A         : 列主序 half 矩阵
 *   m, lda    : 行数、leading dimension
 *   j0        : 当前 panel 的起始列/行
 *   k         : panel 内相对列号 (0 <= k < ib)
 *   block_val : 每个 block 的局部最大值 
 *   block_idx : 每个 block 的局部最大值所在的“全局行号”
 */
__global__ void panel_pivot_search_kernel(
    const half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val,
    int*   __restrict__ block_idx)
{
    // 动态共享内存，下面两个是指向这一块 shared memory 的指针
    extern __shared__ unsigned char smem[];
    // 强制类型转换
    half* s_val = reinterpret_cast<half*>(smem);
    // 一个 block 中选择的最大值应该存在哪里，也就是说这里一次运行解决了一个 block 访问的所有的当中的最大值
    int* s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    // 列号
    const int col = j0 + k;
    // 从哪一行开始选主元
    const int base_row = j0 + k;
    // 列号不可以超过行数，这里后续是可以进行[健壮性]完整的
    if (col >= m) 
        return;

    half local_max_val = __float2half(0.0f);
    int  local_max_idx = base_row;

    const int tid = threadIdx.x;
    // 全局的步长
    const int global_stride = blockDim.x * gridDim.x;

    // 每个线程选出自己所访问数据的最大值
    for (int idx = base_row + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col * lda];
        half v = half_abs(a);
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = idx;
        }
    }

    // 拿到 block 中所有线程最大数据数组
    s_val[tid] = local_max_val;
    s_idx[tid] = local_max_idx;
    __syncthreads();

    // 二分规约 , 但这是 block 内的
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
        }
        __syncthreads();
    }

    // 最终把本 block 最大的值选出
    if (tid == 0) {
        block_val[blockIdx.x] = s_val[0];
        block_idx[blockIdx.x] = s_idx[0];
    }
}

/**
 * kernel 2: 对 block 级结果做最终规约，写入 d_ipiv_rel[k]
 *
 * 输入:
 *   block_val, block_idx : 来自 kernel1 的每个 block 的局部最大值/行号
 *   num_blocks           : 有效 block 数
 *   j0, k                : panel 起点 + 相对列号
 * 输出:
 *   d_ipiv_rel           : pivot 相对位移数组，d_ipiv_rel[k] = pivot_row - (j0+k)
 */
__global__ void panel_pivot_reduce_kernel(
    const half* __restrict__ block_val,
    const int*   __restrict__ block_idx,
    int num_blocks,
    int j0, int k,
    int* __restrict__ d_ipiv_rel)
{
    half max_val = 0.0f;
    int  max_idx = j0 + k;

    for (int i = threadIdx.x; i < num_blocks; i += blockDim.x) {
        half v = block_val[i];
        int   r = block_idx[i];
        if (v > max_val) {
            max_val = v;
            max_idx = r;
        }
    }

    // 用一个 block，线程 0 写结果即可
    __shared__ half s_val;
    __shared__ int   s_idx;
    if (threadIdx.x == 0) {
        s_val = max_val;
        s_idx = max_idx;
    }
    __syncthreads();

    // 为了简单，这里假定 blockDim.x 足够小，直接让 0 号线程写回
    if (threadIdx.x == 0) {
        int pivot_row = s_idx;
        int rel = pivot_row - (j0 + k);
        d_ipiv_rel[k] = rel;
    }
}

/**
 * kernel 3: 在 panel 列块 [j0, j0+ib) 内执行行交换
 *
 * 输入:
 *   A         : half 矩阵
 *   m, lda    : 行数、leading dimension
 *   j0, ib    : panel 起始列、宽度
 *   k         : panel 内相对列号
 *   d_ipiv_rel: pivot 相对位移数组
 *
 * 说明:
 *   这里只在 panel 内做行交换，整块矩阵左右两侧的扩散由 A_exchange.cuh 处理。
 */
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
    half tmp = A[row_k   + col_offset];
    A[row_k   + col_offset] = A[pivot_row + col_offset];
    A[pivot_row + col_offset] = tmp;
}

/**
 * kernel 4: 对列 j0+k 做 L 列缩放：A[r, col_k] /= A[col_k, col_k]
 */
__global__ void panel_column_scale_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    // 行交换之后，pivot 已经在 (row_k=col_k) 行
    half pivot_h = A[col_k + (size_t)col_k * lda];
    float pivot  = __half2float(pivot_h);
    if (pivot == 0.0f) return; // 简单防护，实际应用中可考虑加小阈值

    int r = col_k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;

    half val_h = A[r + (size_t)col_k * lda];
    float val  = __half2float(val_h);
    float l    = val / pivot;
    A[r + (size_t)col_k * lda] = __float2half(l);
}

/**
 * kernel 5: 用 L 列更新 panel 内右侧列：
 *   对所有 r > col_k, c in (col_k, j0+ib):
 *     A[r,c] -= A[r,col_k] * A[col_k,c]
 */
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
    size_t col_c_off = (size_t)c     * lda;

    half  L_h = A[r + col_k_off];
    half  U_h = A[row_k + col_c_off];

    float L_f = __half2float(L_h);
    float U_f = __half2float(U_h);

    half  A_h = A[r + col_c_off];
    float A_f = __half2float(A_h);

    float res = A_f - L_f * U_f;
    A[r + col_c_off] = __float2half(res);
}

/**
 * panel_TSLU 的 host 端入口。
 *
 * 输入:
 *   A         : [in/out] half 矩阵，列主序
 *   m, lda    : 行数、leading dimension
 *   j0        : 当前 panel 起始列
 *   ib        : 当前 panel 宽度
 *   uc        : 微块宽度（这里暂未深入使用，仅做列方向 block 大小时的参考）
 *   d_ipiv_rel: [out] pivot 相对位移数组，长度至少 ib
 *   stream    : CUDA stream
 *
 * 说明：
 *   相比原始版本，最大的结构性变化是：
 *     - 不再依赖模板 IB/UC 生成单一 kernel；
 *     - 改为在 host 侧对 k 做 for 循环，每一步用多个 kernel 分别处理
 *       pivot 搜索、行交换、缩放和更新；
 *     - 每个 kernel 内使用多 block 布局，在 H100/4090 上都可以充分利用 SM。
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
    if (!A || !d_ipiv_rel) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) 
        return;
    if (j0 < 0 || j0 >= m) 
        return;

    // 这里涉及了不同操作中会用到多少 block 多少 thread 是可以调参的地方
    const int threads_pivot = 256;
    const int rows_per_block = 128;

    int max_rows = m - j0;
    if (max_rows <= 0) 
        return;

    // 计算要分配多少个 block 用于 pivot 这里我觉得也是一个优化点
    int num_blocks_pivot = (max_rows + rows_per_block - 1) / rows_per_block;
    if (num_blocks_pivot <= 0) 
        num_blocks_pivot = 1;

    // 分配临时缓冲：每个 block 的局部最大值及其行号
    float* d_block_val = nullptr;
    int*   d_block_idx = nullptr;
    CUDA_CHECK(cudaMallocAsync(&d_block_val, sizeof(float) * num_blocks_pivot, stream));
    CUDA_CHECK(cudaMallocAsync(&d_block_idx, sizeof(int)   * num_blocks_pivot, stream));

    dim3 grid_row_swap((unsigned)((ib + 255) / 256));
    dim3 block_row_swap(256);

    // update kernel 的 block 形状：行 x 列
    int tile_c = (uc > 0) ? uc : 8;
    if (tile_c > 32) tile_c = 32;
    dim3 block_upd(tile_c, 8);

    for (int k = 0; k < ib; ++k) {
        int col = j0 + k;
        if (col >= m) break;

        // 1) pivot 搜索：多 block 扫描 [j0+k, m)
        {
            dim3 grid_pivot(num_blocks_pivot);
            size_t shmem = sizeof(float) * threads_pivot + sizeof(int) * threads_pivot;
            panel_pivot_search_kernel<<<grid_pivot, threads_pivot, shmem, stream>>>(
                A, m, lda, j0, k, d_block_val, d_block_idx);
        }

        // 2) block 结果规约，写入 d_ipiv_rel[k]
        {
            dim3 grid_red(1);
            dim3 block_red(128);
            panel_pivot_reduce_kernel<<<grid_red, block_red, 0, stream>>>(
                d_block_val, d_block_idx, num_blocks_pivot, j0, k, d_ipiv_rel);
        }

        // 3) 在 panel 内做行交换
        {
            panel_row_swap_kernel<<<grid_row_swap, block_row_swap, 0, stream>>>(
                A, m, lda, j0, ib, k, d_ipiv_rel);
        }

        // 4) 列缩放：计算 L 列
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

        // 5) 用该列更新 panel 内右侧列
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
}
