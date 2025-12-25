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

static __device__ __forceinline__ half half_abs(half x) { 
    return __habs(x); 
}

// 选主元并高斯消元
__global__ void panel_pivot_and_prescale_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val,
    int*  __restrict__ block_idx,
    int num_blocks,
    int* __restrict__ d_ipiv_rel) // 偏移是相对行号
{
    extern __shared__ unsigned char smem[];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // 前半段存数，后半段存索引
    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    const int col_k = j0 + k;
    const int row_k = j0 + k;
    if (col_k >= m) 
        return;

    half local_max_val = __float2half(0.0f);
    int  local_max_idx = row_k;

    // 每个线程得到自己扫描的最大
    const int global_stride = blockDim.x * gridDim.x;
    for (int idx = row_k + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col_k * lda];
        half v = half_abs(a);
        if (v > local_max_val) {
            local_max_val = v;
            local_max_idx = idx;
        }
    }

    // warp 内规约得到最大值
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
    __syncthreads();

    // block 内规约得到最大值
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

    // 多 block 得到的结果再规约得到最终值
    if (blockIdx.x == 0) {
        half max_val = __float2half(0.0f);
        int  max_idx = row_k;

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
            d_ipiv_rel[k] = s_idx[0] - row_k;
        }
    }

    grid.sync();

    // 每个线程都拿到最大值，进行高斯消元
    const int rel = d_ipiv_rel[k];
    const int pivot_row = row_k + rel;

    const half pivot = A[pivot_row + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) 
        return;

    // 这里可以看有没有可以修改的为了精度 [fix]
    if (pivot_row == row_k) {
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


// 行交换
__global__ void panel_row_swap_kernel_vec(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    const int* __restrict__ d_ipiv_rel)
{
    const int col_k = j0 + k;
    if (col_k >= m) 
        return;

    const int row_k = j0 + k;
    const int pivot_row = row_k + d_ipiv_rel[k];
    if (pivot_row == row_k) 
        return;

    // 每个线程期望处理两列，所以要乘 2 往上增加
    const int j = j0 + (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    
    if (j + 1 < j0 + ib) {
        // 两列的列号
        const size_t col_off0 = (size_t)j * lda;
        const size_t col_off1 = (size_t)(j + 1) * lda;
        
        // 向量化交换，因为是列主序没法一次性全部装入
        half2 tmp;
        tmp.x = A[row_k + col_off0];
        tmp.y = A[row_k + col_off1];
        
        half2 pivot_val;
        pivot_val.x = A[pivot_row + col_off0];
        pivot_val.y = A[pivot_row + col_off1];
        
        A[row_k + col_off0] = pivot_val.x;
        A[row_k + col_off1] = pivot_val.y;
        
        A[pivot_row + col_off0] = tmp.x;
        A[pivot_row + col_off1] = tmp.y;
        
    } else if (j < j0 + ib) {
        // 边界情况，末尾只剩一列需要交换
        const size_t col_off = (size_t)j * lda;
        const half tmp = A[row_k + col_off];
        A[row_k + col_off] = A[pivot_row + col_off];
        A[pivot_row + col_off] = tmp;
    }
}

// ============================================================================
// ✨ 优化2：Update Kernel - half2 + ILP×2（每线程处理 4 行）
// ============================================================================

__global__ void panel_update_kernel_vec_ILP2(
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
    
    const int c = col_start + blockIdx.x;
    if (c >= col_end) 
        return;
    
    // ✨ 每个线程处理 4 行（2 个 half2）
    const int r_base = row_k + 1 + blockIdx.y * 1024;
    const int r0 = r_base + threadIdx.x * 2;        // 第一对
    const int r1 = r_base + threadIdx.x * 2 + 512;  // 第二对
    
    // 读取 U 一次，broadcast
    const half U_scalar = __ldg(&A[row_k + (size_t)c * lda]);
    const half2 U = __half2half2(U_scalar);
    
    // ========================================
    // 处理第一对（r0, r0+1）
    // ========================================
    if (r0 + 1 < m) {
        half2 L, Av;
        L.x = __ldg(&A[r0 + 0 + (size_t)col_k * lda]);
        L.y = __ldg(&A[r0 + 1 + (size_t)col_k * lda]);
        
        Av.x = A[r0 + 0 + (size_t)c * lda];
        Av.y = A[r0 + 1 + (size_t)c * lda];
        
        half2 result = __hsub2(Av, __hmul2(L, U));
        
        A[r0 + 0 + (size_t)c * lda] = result.x;
        A[r0 + 1 + (size_t)c * lda] = result.y;
        
    } else if (r0 < m) {
        // 只有 1 行
        const half L = __ldg(&A[r0 + (size_t)col_k * lda]);
        const half Av = A[r0 + (size_t)c * lda];
        A[r0 + (size_t)c * lda] = __hsub(Av, __hmul(L, U_scalar));
    }
    
    // ========================================
    // 处理第二对（r1, r1+1）
    // ========================================
    if (r1 + 1 < m) {
        half2 L, Av;
        L.x = __ldg(&A[r1 + 0 + (size_t)col_k * lda]);
        L.y = __ldg(&A[r1 + 1 + (size_t)col_k * lda]);
        
        Av.x = A[r1 + 0 + (size_t)c * lda];
        Av.y = A[r1 + 1 + (size_t)c * lda];
        
        half2 result = __hsub2(Av, __hmul2(L, U));
        
        A[r1 + 0 + (size_t)c * lda] = result.x;
        A[r1 + 1 + (size_t)c * lda] = result.y;
        
    } else if (r1 < m) {
        const half L = __ldg(&A[r1 + (size_t)col_k * lda]);
        const half Av = A[r1 + (size_t)c * lda];
        A[r1 + (size_t)c * lda] = __hsub(Av, __hmul(L, U_scalar));
    }
}

// ============================================================================
// ✨ 优化3：Update Kernel - 纯 half2（简化版）
// ============================================================================

__global__ void panel_update_kernel_vec(
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
    
    const int c = col_start + blockIdx.x;
    if (c >= col_end) 
        return;
    
    const int r_base = row_k + 1 + blockIdx.y * 512;
    const int r = r_base + threadIdx.x * 2;
    
    // 边界检查
    if (r + 1 >= m) {
        if (r < m) {
            const half U = __ldg(&A[row_k + (size_t)c * lda]);
            const half L = __ldg(&A[r + (size_t)col_k * lda]);
            const half Av = A[r + (size_t)c * lda];
            A[r + (size_t)c * lda] = __hsub(Av, __hmul(L, U));
        }
        return;
    }
    
    // ✨ 向量化处理
    const half U_scalar = __ldg(&A[row_k + (size_t)c * lda]);
    const half2 U = __half2half2(U_scalar);
    
    half2 L;
    L.x = __ldg(&A[r + 0 + (size_t)col_k * lda]);
    L.y = __ldg(&A[r + 1 + (size_t)col_k * lda]);
    
    half2 Av;
    Av.x = A[r + 0 + (size_t)c * lda];
    Av.y = A[r + 1 + (size_t)c * lda];
    
    half2 result = __hsub2(Av, __hmul2(L, U));
    
    A[r + 0 + (size_t)c * lda] = result.x;
    A[r + 1 + (size_t)c * lda] = result.y;
}

// ============================================================================
// 辅助函数
// ============================================================================

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
    if (m_effective >= 24576)      rows_per_block = 1024;
    else if (m_effective >= 12288) rows_per_block = 512;
    else if (m_effective >= 4096)  rows_per_block = 256;
    else                           rows_per_block = 128;

    int num_blocks = (m_effective + rows_per_block - 1) / rows_per_block;
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > 64) num_blocks = 64;

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

inline size_t panel_TSLU_workspace_bytes_from_blocks(int num_blocks_pivot)
{
    return (size_t)num_blocks_pivot * (sizeof(half) + sizeof(int));
}

// ============================================================================
// ✨ 主启动函数（简化版，只保留最优配置）
// ============================================================================

/**
 * use_aggressive: 
 *   false = 纯 half2（推荐，稳定）
 *   true  = half2 + ILP×2（更激进，可能更快）
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
    int   num_blocks_pivot,
    bool  use_aggressive = false)
{
    (void)uc;

    if (!A || !d_ipiv_rel || !d_block_val || !d_block_idx) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) 
        return;
    if (j0 < 0 || j0 >= m) 
        return;

    const int m_effective = m - j0;
    if (m_effective <= 0) 
        return;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "launch_panel_TSLU: cooperative launch not supported.\n");
        std::exit(EXIT_FAILURE);
    }

    if (num_blocks_pivot < 1)  
        num_blocks_pivot = 1;
    if (num_blocks_pivot > 64) 
        num_blocks_pivot = 64;

    const int threads_pivot = 256;
    const size_t shmem_coop = (size_t)threads_pivot * (sizeof(half) + sizeof(int));

    // ✨ Swap 配置（向量化：每线程处理 2 列）
    dim3 grid_row_swap((ib + 255) / 256);  // 128 threads × 2 cols = 256 cols/block
    dim3 block_row_swap(128);

    // Update 配置
    dim3 block_upd(256);
    const int rows_per_block = use_aggressive ? 1024 : 512;

    // ========================================
    // 主循环
    // ========================================
    for (int k = 0; k < ib; ++k) {
        const int col = j0 + k;
        if (col >= m) 
            break;

        // (1) Pivot + Scale
        void* args[] = {
            (void*)&A, (void*)&m, (void*)&lda,
            (void*)&j0, (void*)&k,
            (void*)&d_block_val, (void*)&d_block_idx,
            (void*)&num_blocks_pivot, (void*)&d_ipiv_rel
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)panel_pivot_and_prescale_coop_kernel,
            dim3(num_blocks_pivot), dim3(threads_pivot),
            args, shmem_coop, stream));

        // (2) ✨ Vectorized Row Swap
        panel_row_swap_kernel_vec<<<grid_row_swap, block_row_swap, 0, stream>>>(
            A, m, lda, j0, ib, k, d_ipiv_rel);

        // (3) ✨ Vectorized Update
        const int rows_rem = m - (j0 + k + 1);
        const int cols_rem = ib - (k + 1);
        
        if (rows_rem > 0 && cols_rem > 0) {
            const int grid_x = cols_rem;
            const int grid_y = (rows_rem + rows_per_block - 1) / rows_per_block;
            
            if (use_aggressive) {
                panel_update_kernel_vec_ILP2<<<dim3(grid_x, grid_y), block_upd, 0, stream>>>(
                    A, m, lda, j0, ib, k);
            } else {
                panel_update_kernel_vec<<<dim3(grid_x, grid_y), block_upd, 0, stream>>>(
                    A, m, lda, j0, ib, k);
            }
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_panel_buffers() {}