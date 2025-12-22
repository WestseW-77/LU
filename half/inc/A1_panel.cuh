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

/**
 * ============================================================================
 * A1_panel.cuh  (cooperative pivot+pre-swap scale) + (swap) + (update)
 * ============================================================================
 *
 * 本版本相对你“缓存U+缓存L + float inv_pivot”的版本，按更稳的方向收敛：
 *
 * 1) pre-swap scale：恢复为 half 除法（去掉 half<->float 转换与 float 乘法）
 *    - 保持你之前的数值/指令路径，更便于对比性能
 *
 * 2) update：只缓存 U 行（tile_c 个 U[k,c]）到 shared
 *    - 不缓存 L（避免为 L 增加 shared 写/读/额外开销）
 *    - 每个线程仍直接从 global 读取 L = A[r,col_k]
 *
 * 目标：减少 update 中“U 行重复被 tile_r=32 个线程反复读取”的 global load，
 * 同时把新增开销控制到最小。
 *
 * 约束：
 * - A 全程 half 存储/写回
 * - launch_panel_TSLU 签名不变
 * - 仅 cooperative 路线（无 fallback）
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

static __device__ __forceinline__ half half_abs(half x) { return __habs(x); }

// ----------------------------------------------------------------------------
// Workspace（复用 block_val/block_idx）
// 注意：默认不会多线程/多 stream 并发调用同一个 TU 的 workspace。
// ----------------------------------------------------------------------------
namespace panel_ws {
    inline half* g_block_val = nullptr;
    inline int*  g_block_idx = nullptr;
    inline int   g_capacity  = 0;

    inline void ensure_capacity(int needed) {
        if (needed <= g_capacity) return;

        if (g_block_val) CUDA_CHECK(cudaFree(g_block_val));
        if (g_block_idx) CUDA_CHECK(cudaFree(g_block_idx));

        CUDA_CHECK(cudaMalloc(&g_block_val, sizeof(half) * (size_t)needed));
        CUDA_CHECK(cudaMalloc(&g_block_idx, sizeof(int)  * (size_t)needed));
        g_capacity = needed;
    }
} // namespace panel_ws

// ----------------------------------------------------------------------------
// Cooperative kernel：pivot(search+reduce) + pre-swap scale（按你的“先除再换”规则）
// pre-swap scale：使用 half 除法（不引入 float inv）
// ----------------------------------------------------------------------------
__global__ void panel_pivot_and_prescale_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int k,
    half* __restrict__ block_val,
    int*  __restrict__ block_idx,
    int num_blocks,
    int* __restrict__ d_ipiv_rel)
{
    extern __shared__ unsigned char smem[];
    const int tid = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    // shared：按 threads 的 half+int 给足，用于 block0 最终 reduce
    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    const int col_k = j0 + k;
    const int row_k = j0 + k;
    if (col_k >= m) return;

    // -------------------------
    // Phase 1: 每个 block 求 block max（写到 block_val/block_idx）
    // -------------------------
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

    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        half other_val = __shfl_down_sync(0xffffffff, local_max_val, offset);
        int  other_idx = __shfl_down_sync(0xffffffff, local_max_idx, offset);
        if (other_val > local_max_val) {
            local_max_val = other_val;
            local_max_idx = other_idx;
        }
    }

    // shared 前 num_warps 槽位暂存 warp 结果（复用 s_val/s_idx 前段）
    if (lane == 0) {
        s_val[warp_id] = local_max_val;
        s_idx[warp_id] = local_max_idx;
    }
    __syncthreads();

    // warp0 reduce 得到 block max
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

    // -------------------------
    // Phase 2: block0 最终 reduce -> d_ipiv_rel[k]
    // -------------------------
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

    // -------------------------
    // Phase 3: pre-swap scale（half 除法版本）
    // -------------------------
    const int rel = d_ipiv_rel[k];
    const int pivot_row = row_k + rel;

    const half pivot = A[pivot_row + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) return;

    if (pivot_row == row_k) {
        // 标准：r = row_k+1..m-1
        for (int r = row_k + 1 + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            half val = A[r + (size_t)col_k * lda];
            A[r + (size_t)col_k * lda] = val / pivot;
        }
    } else {
        // 预交换缩放：r = row_k..m-1 且 r != pivot_row
        for (int r = row_k + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            if (r == pivot_row) continue;
            half val = A[r + (size_t)col_k * lda];
            A[r + (size_t)col_k * lda] = val / pivot;
        }
    }
}

// ----------------------------------------------------------------------------
// swap kernel：panel 内列范围交换 row_k 与 pivot_row
// ----------------------------------------------------------------------------
__global__ void panel_row_swap_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    const int* __restrict__ d_ipiv_rel)
{
    int col_k = j0 + k;
    if (col_k >= m) return;

    int row_k = j0 + k;
    int pivot_row = row_k + d_ipiv_rel[k];
    if (pivot_row == row_k) return;

    int j = j0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= j0 + ib) return;

    size_t col_offset = (size_t)j * lda;
    half tmp = A[row_k + col_offset];
    A[row_k + col_offset] = A[pivot_row + col_offset];
    A[pivot_row + col_offset] = tmp;
}

// ----------------------------------------------------------------------------
// update kernel（只缓存 U 行）：减少 U 被 tile_r 重复读的 global load
// ----------------------------------------------------------------------------
__global__ void panel_update_kernel_cacheU(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k)
{
    const int col_k = j0 + k;
    const int row_k = j0 + k;
    const int col_start = col_k + 1;
    const int col_end   = j0 + ib;

    if (col_k >= m) return;

    // blockDim = (tile_c, tile_r)
    const int tile_c = (int)blockDim.x;
    const int tile_r = (int)blockDim.y;

    const int c0 = col_start + blockIdx.x * tile_c;

    const int r = row_k + 1 + blockIdx.y * tile_r + threadIdx.y;
    const int c = c0 + threadIdx.x;

    // shared: U tile（tile_c 个 half）
    extern __shared__ unsigned char smem[];
    half* sU = reinterpret_cast<half*>(smem);  // [tile_c]

    // 只让 threadIdx.y==0 的一行线程加载 U 行（tile_c 个）
    if (threadIdx.y == 0) {
        if (c < col_end) {
            sU[threadIdx.x] = A[row_k + (size_t)c * lda];
        } else {
            sU[threadIdx.x] = __float2half(0.0f);
        }
    }
    __syncthreads();

    if (r >= m || c >= col_end) return;

    // L 每个线程自己读一次（避免缓存 L 带来的额外开销）
    const half L = A[r + (size_t)col_k * lda];
    const half U = sU[threadIdx.x];

    const half A_h = A[r + (size_t)c * lda];
    A[r + (size_t)c * lda] = __hsub(A_h, __hmul(L, U));
}

// ----------------------------------------------------------------------------
// 主启动函数：纯 cooperative（无 fallback）
// ----------------------------------------------------------------------------
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
    (void)uc;

    if (!A || !d_ipiv_rel) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= m) return;

    const int m_effective = m - j0;
    if (m_effective <= 0) return;

    const int threads_pivot = 256;
    const size_t shmem_coop = sizeof(half) * (size_t)threads_pivot + sizeof(int) * (size_t)threads_pivot;

    // cooperative 支持性检查
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "Error: cooperative launch not supported on this device.\n");
        std::exit(EXIT_FAILURE);
    }

    // max cooperative grid
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

    // num_blocks_pivot（通过 rows_per_block 调整）
    int rows_per_block;
    if (m_effective >= 24576)      rows_per_block = 1024;
    else if (m_effective >= 12288) rows_per_block = 512;
    else if (m_effective >= 4096)  rows_per_block = 256;
    else                           rows_per_block = 128;

    int num_blocks_pivot = (m_effective + rows_per_block - 1) / rows_per_block;
    if (num_blocks_pivot < 1) num_blocks_pivot = 1;
    if (num_blocks_pivot > 64) num_blocks_pivot = 64;

    if (num_blocks_pivot > max_coop_grid) {
        int min_rows_per_block = (m_effective + max_coop_grid - 1) / max_coop_grid;
        if (min_rows_per_block < 1) min_rows_per_block = 1;
        rows_per_block = min_rows_per_block;
        num_blocks_pivot = (m_effective + rows_per_block - 1) / rows_per_block;
        if (num_blocks_pivot < 1) num_blocks_pivot = 1;
    }

    if (num_blocks_pivot > max_coop_grid) {
        fprintf(stderr,
                "Error: cooperative grid too large. num_blocks=%d, max_coop_grid=%d\n",
                num_blocks_pivot, max_coop_grid);
        std::exit(EXIT_FAILURE);
    }

    // workspace
    panel_ws::ensure_capacity(num_blocks_pivot);
    half* d_block_val = panel_ws::g_block_val;
    int*  d_block_idx = panel_ws::g_block_idx;

    // swap kernel config
    dim3 grid_row_swap((ib + 255) / 256);
    dim3 block_row_swap(256);

    // update kernel config（维持你原来的 tile）
    const int tile_c = 4;
    const int tile_r = 32;
    dim3 block_upd(tile_c, tile_r);

    // update shared: tile_c half
    const size_t shmem_upd = sizeof(half) * (size_t)tile_c;

    for (int k = 0; k < ib; ++k) {
        int col = j0 + k;
        if (col >= m) break;

        // (1) cooperative pivot + pre-swap scale
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

        // (2) swap（panel 内）
        panel_row_swap_kernel<<<grid_row_swap, block_row_swap, 0, stream>>>(
            A, m, lda, j0, ib, k, d_ipiv_rel);

        // (3) update（缓存 U 行）
        int rows_rem = m - (j0 + k + 1);
        int cols_rem = ib - (k + 1);
        if (rows_rem > 0 && cols_rem > 0) {
            int grid_x = (cols_rem + tile_c - 1) / tile_c;
            int grid_y = (rows_rem + tile_r - 1) / tile_r;
            panel_update_kernel_cacheU<<<dim3(grid_x, grid_y), block_upd, shmem_upd, stream>>>(
                A, m, lda, j0, ib, k);
        }
    }

    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// 释放 workspace
// ----------------------------------------------------------------------------
inline void cleanup_panel_buffers() {
    if (panel_ws::g_block_val) CUDA_CHECK(cudaFree(panel_ws::g_block_val));
    if (panel_ws::g_block_idx) CUDA_CHECK(cudaFree(panel_ws::g_block_idx));
    panel_ws::g_block_val = nullptr;
    panel_ws::g_block_idx = nullptr;
    panel_ws::g_capacity  = 0;
}
