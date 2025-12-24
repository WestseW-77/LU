#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <math.h>
#include <stdint.h>
#include <float.h>
#include <cstdio>
#include <cstdlib>

namespace cg = cooperative_groups;

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

static __device__ __forceinline__ half half_abs(half x) { return __habs(x); }

/**
 * ============================================================================
 * 本文件对外接口变更（按你的要求）：
 *
 * 1) 提供接口：计算 pivot workspace 需要的元素数（以及 bytes）
 *    - 由外部提前 cudaMalloc 并在多次调用中复用
 *
 * 2) launch_panel_TSLU 仍保留原名字，但参数新增/替换为外部传入 workspace
 *    - 不再在函数内部 cudaMalloc/cudaFree
 *    - 不再使用静态全局 workspace
 *
 * 3) 仅 cooperative 路线（4090/更新设备），无 fallback
 * ============================================================================
 */

// -------------------------------------------
// 设备侧：cooperative pivot + pre-swap scale
// -------------------------------------------
__global__ void panel_pivot_and_prescale_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val, // 每个 block 选择到的最大值
    int*  __restrict__ block_idx, // 每个 block 选择到的最大值的索引
    int num_blocks,
    int* __restrict__ d_ipiv_rel) // pivot 结果的相对偏移量
{
    extern __shared__ unsigned char smem[];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // 前半段放具体的数值，后半段放对应的索引
    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    // 当前处理的列号
    const int col_k = j0 + k;
    // 对角线行号
    const int row_k = j0 + k;
    if (col_k >= m) 
        return;

    // 每个 block 找自己负责数据中的最大值
    half local_max_val = __float2half(0.0f);
    int  local_max_idx = row_k;

    const int global_stride = blockDim.x * gridDim.x;
    for (int idx = row_k + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col_k * lda];
        half v = half_abs(a);
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = idx;
        }
    }

    // warp 级别 reduce，得到每个 warp 内的最大值
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        half other_val = __shfl_down_sync(0xffffffff, local_max_val, offset);
        int  other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max_val) {
            local_max_val = other_val;
            local_max_idx = other_idx;
        }
    }

    if (lane == 0) {
        s_val[warp_id] = local_max_val;
        s_idx[warp_id] = local_max_idx;
    }
    // [question] 这个同步是不是少了一级
    __syncthreads();

    // warp0 规约，把不同 warp 内的最大值规约成为 block 最大值
    if (warp_id == 0) {
        half warp_max = (lane < num_warps) ? s_val[lane] : __float2half(0.0f);
        int  warp_idx = (lane < num_warps) ? s_idx[lane] : row_k;

        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            half other_val = __shfl_down_sync(0xffffffff, warp_max, offset);
            int  other_idx = __shfl_down_sync(0xffffffff, warp_idx, offset);
            if (other_val > warp_max) {
                warp_max = other_val;
                warp_idx = other_idx;
            }
        }

        if (lane == 0) {
            block_val[blockIdx.x] = warp_max;
            block_idx[blockIdx.x] = warp_idx;
        }
    }

    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // 选出所有 block 加在一起的最大值，并写入数组中传出，只由第一个 block 进行操作
    if (blockIdx.x == 0) {
        half max_val = __float2half(0.0f);
        int  max_idx = row_k;

        // 同理，先选出每个线程接触到的最大值，存入 shared memory
        for (int i = tid; i < num_blocks; i += blockDim.x) {
            half v = block_val[i];
            int  r = block_idx[i];
            if (v > max_val) {
                max_val = v;
                max_idx = r;
            }
        }

        s_val[tid] = max_val;
        s_idx[tid] = max_idx;
        __syncthreads();

        // 再做一次 reduce 得到最终的最大值
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                if (s_val[tid + stride] > s_val[tid]) {
                    s_val[tid] = s_val[tid + stride];
                    s_idx[tid] = s_idx[tid + stride];
                }
            }
            __syncthreads();
        }

        // 传出的索引是相对位置，这里会是需要统一的地方 [fix]
        if (tid == 0) {
            d_ipiv_rel[k] = s_idx[0] - row_k;
        }
    }

    grid.sync();

    // 这里进行了顺序调整，选择先进行高斯消元再进行 pivot 最终的 exchange 操作
    const int rel = d_ipiv_rel[k];
    const int pivot_row = row_k + rel;

    const half pivot = A[pivot_row + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) 
        return;

    // 这边的除法可能是可以优化数值精度的地方 [fix]
    if (pivot_row == row_k) {
        // 如果 pivot 行就是对角线行，那么直接进行高斯消元
        for (int r = row_k + 1 + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            half val = A[r + (size_t)col_k * lda];
            A[r + (size_t)col_k * lda] = val / pivot;
        }
    } else {
        for (int r = row_k + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            if (r == pivot_row) 
                continue;
            half val = A[r + (size_t)col_k * lda];
            A[r + (size_t)col_k * lda] = val / pivot;
        }
    }
}

// 对 pivot 结果进行交换 ，目前是每个线程都只交换了一个数据，拉更多的 block 参加 [fix]
__global__ void panel_row_swap_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    const int* __restrict__ d_ipiv_rel)
{
    const int col_k = j0 + k;
    if (col_k >= m) 
        return;
    // 当前的行号
    const int row_k = j0 + k;
    // 目标交换的行号
    const int pivot_row = row_k + d_ipiv_rel[k];
    if (pivot_row == row_k) 
        return;
    // 自己负责交换的行号
    const int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j0 + ib) 
        return;

    const size_t col_off = (size_t)j * lda;
    const half tmp = A[row_k + col_off];
    A[row_k + col_off] = A[pivot_row + col_off];
    A[pivot_row + col_off] = tmp;
}

// 对操作后的矩阵右边进行更新
__global__ void panel_update_kernel_cacheU(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k)
{
    const int col_k = j0 + k;
    const int row_k = j0 + k;
    const int col_start = col_k + 1;
    const int col_end = j0 + ib;
    
    if (col_k >= m) 
        return;
    
    // 简化：每个 block 处理 1 列
    const int c = col_start + blockIdx.x;
    const int r = row_k + 1 + blockIdx.y * blockDim.x + threadIdx.x;
    
    if (c >= col_end || r >= m) 
        return;
    
    // 直接从全局内存读取（L1 cache 会处理 U 的重复读取）
    const half L = A[r + (size_t)col_k * lda];      // coalesced 读取
    const half U = A[row_k + (size_t)c * lda];      // 所有线程读同一个值，会被 broadcast
    const half A_val = A[r + (size_t)c * lda];      // coalesced 读取
    
    A[r + (size_t)c * lda] = __hsub(A_val, __hmul(L, U));  // coalesced 写入
}

/**
 * ============================================================================
 * 对外接口1：计算 cooperative pivot 能用的最大 grid，以及推荐 num_blocks_pivot
 * ============================================================================
 *
 * 说明：
 * - 这个函数会查询设备属性与 occupancy（依赖 kernel 的寄存器/shared 使用）。
 * - 返回值 num_blocks_pivot 是你在 launch_panel_TSLU 中应使用的 blocks 数。
 * - 外部通常只需对“最坏情况 j0=0”调用一次，得到最大 workspace 并复用。
 */
inline int panel_TSLU_required_pivot_blocks(int m, int j0)
{
    const int m_effective = m - j0;
    if (m_effective <= 0) return 1;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "panel_TSLU_required_pivot_blocks: cooperative launch not supported.\n");
        std::exit(EXIT_FAILURE);
    }

    const int threads_pivot = 256;
    const size_t shmem_coop = (size_t)threads_pivot * (sizeof(half) + sizeof(int));

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        panel_pivot_and_prescale_coop_kernel,
        threads_pivot,
        (int)shmem_coop));

    int max_coop_grid = sm_count * max_blocks_per_sm;
    if (max_coop_grid < 1) max_coop_grid = 1;

    int rows_per_block;
    if (m_effective >= 24576)      
        rows_per_block = 1024;
    else if (m_effective >= 12288) 
        rows_per_block = 512;
    else if (m_effective >= 4096)  
        rows_per_block = 256;
    else                           
        rows_per_block = 128;

    int num_blocks = (m_effective + rows_per_block - 1) / rows_per_block;
    if (num_blocks < 1) 
        num_blocks = 1;
    if (num_blocks > 64) 
        num_blocks = 64;

    if (num_blocks > max_coop_grid) {
        int min_rows_per_block = (m_effective + max_coop_grid - 1) / max_coop_grid;
        if (min_rows_per_block < 1) min_rows_per_block = 1;
        rows_per_block = min_rows_per_block;
        num_blocks = (m_effective + rows_per_block - 1) / rows_per_block;
        if (num_blocks < 1) num_blocks = 1;
    }

    if (num_blocks > max_coop_grid) {
        fprintf(stderr,
                "panel_TSLU_required_pivot_blocks: cooperative grid too large. need=%d max=%d\n",
                num_blocks, max_coop_grid);
        std::exit(EXIT_FAILURE);
    }

    return num_blocks;
}

/**
 * 对外接口2：给定 num_blocks_pivot，计算 workspace bytes
 * 你也可以直接用：bytes = blocks*(sizeof(half)+sizeof(int))
 */
inline size_t panel_TSLU_workspace_bytes_from_blocks(int num_blocks_pivot)
{
    return (size_t)num_blocks_pivot * (sizeof(half) + sizeof(int));
}

/**
 * ============================================================================
 * 主启动函数（保留原名字，但改为外部传入 workspace）
 * ============================================================================
 *
 * 你需要外部提前分配：
 *   - half* d_block_val   (至少 num_blocks_pivot 个元素)
 *   - int*  d_block_idx   (至少 num_blocks_pivot 个元素)
 *
 * 传入：
 *   - num_blocks_pivot：建议使用 panel_TSLU_required_pivot_blocks(m, j0) 的返回值
 *
 * 注意：本函数不再 malloc/free。
 */
inline void launch_panel_TSLU(
    half* A,
    int   m,
    int   lda,
    int   j0,
    int   ib,
    int   uc,
    int*  d_ipiv_rel,
    cudaStream_t stream,
    half* d_block_val,
    int*  d_block_idx,
    int   num_blocks_pivot)  // 要在函数体外规定好 pivot 会使用多少 block
{
    // 暂时还不会使用 uc，后续可以加上这个参数，对应更新时候的参数设置
    (void)uc; 

    if (!A || !d_ipiv_rel || !d_block_val || !d_block_idx) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) 
        return;
    if (j0 < 0 || j0 >= m) 
        return;

    // 需要更新的行数
    const int m_effective = m - j0;
    if (m_effective <= 0) 
        return;

    // 检查 GPU 是否支持协作组
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "launch_panel_TSLU: cooperative launch not supported.\n");
        std::exit(EXIT_FAILURE);
    }

    // 目前规定好 pivot 的 block 数在 1-64 之间 [fix]
    if (num_blocks_pivot < 1)  
        num_blocks_pivot = 1;
    if (num_blocks_pivot > 64) 
        num_blocks_pivot = 64;

    // pivot 每 block 的线程数 [fix]
    const int threads_pivot = 256;
    const size_t shmem_coop = (size_t)threads_pivot * (sizeof(half) + sizeof(int));

    // swap 的 grid 与 block 配置，核内写的是每个线程处理一列的交换，这里应当和外面我们规定好的相适配，实际上可以剩 block 数 [fix]
    dim3 grid_row_swap((ib + 127) / 128);
    dim3 block_row_swap(128);

    // update 的 block 配置，也是需要看一下的地方，目前来看会对性能有较大的影响，我认为可以考虑这里根据 size 采用动态分配 [fix]
    const int tile_col = 1;
    const int tile_row = 256;
    dim3 block_upd(tile_row);
    const size_t shmem_upd = 0;

    for (int k = 0; k < ib; ++k) {
        // 当前处理的列号
        const int col = j0 + k;
        if (col >= m) 
            break;

        // pivot 与高斯消元
        void* args[] = {
            (void*)&A,
            (void*)&m, (void*)&lda,
            (void*)&j0, (void*)&k,
            (void*)&d_block_val,
            (void*)&d_block_idx,
            (void*)&num_blocks_pivot,
            (void*)&d_ipiv_rel
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)panel_pivot_and_prescale_coop_kernel,
            dim3(num_blocks_pivot), dim3(threads_pivot),
            args,
            shmem_coop,
            stream));

        // 交换 pivot 行
        panel_row_swap_kernel<<<grid_row_swap, block_row_swap, 0, stream>>>(
            A, m, lda, j0, ib, k, d_ipiv_rel);

        // 更新尾矩阵

        // 需要更新的行数与列数
        const int rows_rem = m - (j0 + k + 1);
        const int cols_rem = ib - (k + 1);
        if (rows_rem > 0 && cols_rem > 0) {
            // 每个线程负责一个数据，看总共需要起多少个 block
            const int grid_x = (cols_rem + tile_col - 1) / tile_col;
            const int grid_y = (rows_rem + tile_row - 1) / tile_row;
            panel_update_kernel_cacheU<<<dim3(grid_x, grid_y), block_upd, shmem_upd, stream>>>(
                A, m, lda, j0, ib, k);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}
inline void cleanup_panel_buffers() {}
