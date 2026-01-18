// A1_panel.cuh
// Aggressive optimization version:
//   - Fuse per-column cooperative pivot + update into ONE cooperative kernel per micro-block (kb<=32)
//   - Drastically reduce kernel launch count (previously: ~256 coop launches + ~224 update launches per panel)
//   - Keep the existing block-out TRSM + GEMM inside panel (cuBLAS) for level-3 update
//
// Notes:
//   * This keeps the external API the same: launch_panel_TSLU(...)
//   * You SHOULD re-run hgetrf_bufferSize(...) after replacing this file (workspace size may change).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
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

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,            \
                    __FILE__, __LINE__);                                       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half  = __half;
using half2 = __half2;

static __device__ __forceinline__ half half_abs(half x) { return __habs(x); }

// =====================================================================================
// (1) Fused cooperative kernel: factorize a micro-block [k0, kend) within the panel
//     - Each iteration k does:
//         pivot search -> pivot selection -> row swap (panel columns) -> scale + rank-1 update within micro-block
//     - 1 cooperative launch per micro-block (kb<=32)
// =====================================================================================
__global__ void panel_getf2_microblock_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k0, int kend,
    half* __restrict__ block_val,
    int*  __restrict__ block_idx,
    int num_blocks,
    int* __restrict__ d_ipiv)
{
    // Design choice:
    //   ROW_TILE=512 means 1 block covers 512 rows using 256 threads (each thread handles 2 rows via half2).
    //   That scales naturally for your m sizes:
    //     m=32768 -> ~64 blocks; m=8192 -> ~16 blocks.
    constexpr int THREADS  = 256;
    constexpr int ROW_TILE = 512;
    constexpr int MAX_KB   = 32;

    cg::grid_group grid = cg::this_grid();

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE;

    __shared__ half s_warp_val[NUM_WARPS];
    __shared__ int  s_warp_idx[NUM_WARPS];
    __shared__ half sU[MAX_KB];

    // Loop columns in this micro-block
    for (int k = k0; k < kend; ++k) {
        const int col_k = j0 + k;
        const int row_k = j0 + k;
        if (row_k >= m) break; // uniform

        // ------------------------------
        // 1) Pivot search within column col_k, rows [row_k, m)
        //    Block i scans its tile starting at (row_k + i*ROW_TILE)
        // ------------------------------
        const int tile_start = row_k + (int)blockIdx.x * ROW_TILE;

        half local_max = __float2half(0.0f);
        int  local_idx = row_k;

        int r0 = tile_start + tid * 2;
        if (r0 < m) {
            half a0 = A[r0 + (size_t)col_k * lda];
            half v0 = half_abs(a0);
            if (v0 > local_max) { local_max = v0; local_idx = r0; }

            int r1 = r0 + 1;
            if (r1 < m) {
                half a1 = A[r1 + (size_t)col_k * lda];
                half v1 = half_abs(a1);
                if (v1 > local_max) { local_max = v1; local_idx = r1; }
            }
        }

        // Warp reduce (max)
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            half ov = __shfl_down_sync(0xffffffff, local_max, off);
            int  oi = __shfl_down_sync(0xffffffff, local_idx, off);
            if (ov > local_max) { local_max = ov; local_idx = oi; }
        }
        if (lane == 0) {
            s_warp_val[warp] = local_max;
            s_warp_idx[warp] = local_idx;
        }
        __syncthreads();

        // Warp0 reduces warps -> one value per block
        if (warp == 0) {
            half vmax = (lane < NUM_WARPS) ? s_warp_val[lane] : __float2half(0.0f);
            int  vidx = (lane < NUM_WARPS) ? s_warp_idx[lane] : row_k;

            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                half ov = __shfl_down_sync(0xffffffff, vmax, off);
                int  oi = __shfl_down_sync(0xffffffff, vidx, off);
                if (ov > vmax) { vmax = ov; vidx = oi; }
            }
            if (lane == 0) {
                block_val[blockIdx.x] = vmax;
                block_idx[blockIdx.x] = vidx;
            }
        }

        grid.sync();

        // ------------------------------
        // 2) Block0 reduces per-block maxima -> pivot row
        // ------------------------------
        if (blockIdx.x == 0) {
            if (warp == 0) {
                half vmax = __float2half(0.0f);
                int  vidx = row_k;

                for (int i = lane; i < num_blocks; i += WARP_SIZE) {
                    half v = block_val[i];
                    int  r = block_idx[i];
                    if (v > vmax) { vmax = v; vidx = r; }
                }

                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    half ov = __shfl_down_sync(0xffffffff, vmax, off);
                    int  oi = __shfl_down_sync(0xffffffff, vidx, off);
                    if (ov > vmax) { vmax = ov; vidx = oi; }
                }

                if (lane == 0) {
                    d_ipiv[row_k] = vidx + 1; // 1-based
                    block_idx[0]  = vidx;     // broadcast pivot row
                }
            }
        }

        grid.sync();

        const int pivot_row = block_idx[0];

        // ------------------------------
        // 3) Swap row_k <-> pivot_row inside the panel columns [j0, j0+ib)
        //    (Columns outside panel will be handled by A_exchange)
        // ------------------------------
        if (blockIdx.x == 0 && pivot_row != row_k) {
            for (int j = j0 + tid; j < j0 + ib; j += THREADS) {
                size_t off = (size_t)j * lda;
                half tmp = A[row_k + off];
                A[row_k + off] = A[pivot_row + off];
                A[pivot_row + off] = tmp;
            }
        }

        grid.sync();

        // ------------------------------
        // 4) Scale + rank-1 update within this micro-block:
        //      - Scale: A[r, col_k] /= pivot, for r > row_k
        //      - Update: A[r, c] -= A[r,col_k] * A[row_k,c], for c in (col_k, j0+kend)
        //    This replaces the old (scale kernel + update kernel) launches.
        // ------------------------------
        const half pivot = A[row_k + (size_t)col_k * lda];
        if (pivot == __float2half(0.0f)) {
            // keep grid.sync pattern identical
            grid.sync();
            continue;
        }

        const int col_begin = col_k + 1;
        const int col_end   = j0 + kend;
        int num_u = col_end - col_begin;  // <= 31 (since kend-k0<=32)
        if (num_u < 0) num_u = 0;
        if (num_u > MAX_KB) num_u = MAX_KB;

        // Load U row into shared (per-block)
        for (int t = tid; t < num_u; t += THREADS) {
            const int c = col_begin + t;
            sU[t] = A[row_k + (size_t)c * lda];
        }
        __syncthreads();

        // Rows to update start from row_k+1
        const int tile_u = row_k + 1 + (int)blockIdx.x * ROW_TILE;
        const int rr = tile_u + tid * 2;
        if (rr < m) {
            // Scale L entries in this column
            half a0 = A[rr + (size_t)col_k * lda];
            half l0 = __hdiv(a0, pivot);
            A[rr + (size_t)col_k * lda] = l0;

            half l1 = __float2half(0.0f);
            if (rr + 1 < m) {
                half a1 = A[(rr + 1) + (size_t)col_k * lda];
                l1 = __hdiv(a1, pivot);
                A[(rr + 1) + (size_t)col_k * lda] = l1;
            }

            half2 L2;
            L2.x = l0;
            L2.y = l1;

            // Update remaining columns inside the micro-block
            for (int t = 0; t < num_u; ++t) {
                const int c = col_begin + t;
                const half2 U2 = __half2half2(sU[t]);

                half2 Av2;
                Av2.x = A[rr + (size_t)c * lda];
                Av2.y = (rr + 1 < m) ? A[(rr + 1) + (size_t)c * lda] : __float2half(0.0f);

                const half2 R2 = __hsub2(Av2, __hmul2(L2, U2));

                A[rr + (size_t)c * lda] = R2.x;
                if (rr + 1 < m) A[(rr + 1) + (size_t)c * lda] = R2.y;
            }
        }

        // Ensure updates are visible before next pivot step
        grid.sync();
    }
}

// =====================================================================================
// (2) Existing panel block-out (TRSM + GEMM) inside panel: keep as-is
// =====================================================================================
template<int K_MAX>
__global__ void panel_trsm_u12_warp_kernel(
    const half* __restrict__ A,
    half* __restrict__ U12,
    int lda,
    int j0_k0,
    int K,
    int N)
{
    __shared__ half sL[K_MAX * K_MAX];

    for (int idx = threadIdx.x; idx < K * K; idx += blockDim.x) {
        int i = idx % K;
        int j = idx / K;
        sL[i + j * K_MAX] = A[(j0_k0 + i) + (size_t)(j0_k0 + j) * lda];
    }
    __syncthreads();

    const int warp = (int)threadIdx.x / WARP_SIZE;
    const int lane = (int)threadIdx.x & (WARP_SIZE - 1);

    const int rhs = (int)blockIdx.x * (int)(blockDim.x / WARP_SIZE) + warp;
    if (rhs >= N) return;

    half* colptr = U12 + (size_t)rhs * lda;

    for (int i = 0; i < K; ++i) {
        float bi = 0.0f;
        if (lane == 0) bi = __half2float(colptr[i]);

        float acc = 0.0f;
        for (int k = lane; k < i; k += WARP_SIZE) {
            float Lik = __half2float(sL[i + k * K_MAX]);
            float xk  = __half2float(colptr[k]);
            acc += Lik * xk;
        }
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, off);

        if (lane == 0) colptr[i] = __float2half(bi - acc);
        __syncwarp();
    }
}

static inline void panel_blockout_trsm_gemm_inside_panel(
    half* A, int m, int lda,
    int j0, int ib,
    int k0, int kend,
    cublasHandle_t cublas_handle,
    cudaStream_t stream)
{
    const int j0_k0 = j0 + k0;
    const int row2  = j0 + kend;
    const int col2  = j0 + kend;

    const int K = kend - k0;
    const int N = (j0 + ib) - col2;
    const int M = m - row2;

    if (K <= 0 || N <= 0)
        return;
    if (K > 32) {
        fprintf(stderr, "panel_blockout_trsm_gemm_inside_panel: K=%d > 32 not supported.\n", K);
        std::exit(EXIT_FAILURE);
    }

    half* U12 = A + j0_k0 + (size_t)col2 * lda;

    constexpr int WARPS_PER_BLOCK = 4;
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    panel_trsm_u12_warp_kernel<32><<<grid, block, 0, stream>>>(
        A, U12, lda, j0_k0, K, N);
    CUDA_CHECK(cudaGetLastError());

    if (M <= 0) return;

    half* L21 = A + row2 + (size_t)j0_k0 * lda;
    half* A22 = A + row2 + (size_t)col2  * lda;

    const float alpha = -1.0f;
    const float beta  =  1.0f;

    if (!cublas_handle) {
        fprintf(stderr, "panel_blockout_trsm_gemm_inside_panel: cublas_handle is null.\n");
        std::exit(EXIT_FAILURE);
    }

    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        L21, CUDA_R_16F, lda,
        U12, CUDA_R_16F, lda,
        &beta,
        A22, CUDA_R_16F, lda,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

// =====================================================================================
// (3) Workspace sizing: how many cooperative blocks we will use (max)
// =====================================================================================
inline int panel_TSLU_required_pivot_blocks(int m, int j0)
{
    const int m_effective = m - j0;
    if (m_effective <= 0)
        return 1;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "panel_TSLU_required_pivot_blocks: cooperative launch not supported.\n");
        std::exit(EXIT_FAILURE);
    }

    const int threads = 256;
    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

    int max_blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        panel_getf2_microblock_coop_kernel,
        threads,
        0));

    int max_coop_grid = sm_count * max_blocks_per_sm;
    if (max_coop_grid < 1) max_coop_grid = 1;

    // Heuristic tuned for your m values.
    // ROW_TILE=512 => blocks scale with m. m=32768 -> 64 blocks.
    const int rows_per_block = 512;
    int num_blocks = (m_effective + rows_per_block - 1) / rows_per_block;

    // Hard cap to avoid insane grid.sync overhead.
    const int CAP = 128;
    if (num_blocks < 1) num_blocks = 1;
    if (num_blocks > CAP) num_blocks = CAP;

    if (num_blocks > max_coop_grid) {
        num_blocks = max_coop_grid;
        if (num_blocks < 1) num_blocks = 1;
    }

    return num_blocks;
}

static inline int panel_TSLU_choose_blocks_fast(int m_effective, int num_blocks_max)
{
    if (m_effective <= 0) return 1;
    const int rows_per_block = 512;
    int nb = (m_effective + rows_per_block - 1) / rows_per_block;
    if (nb < 1) nb = 1;
    if (nb > num_blocks_max) nb = num_blocks_max;
    if (nb < 1) nb = 1;
    return nb;
}

// =====================================================================================
// (4) Public launcher: called by hgetrf.cuh
// =====================================================================================
inline void launch_panel_TSLU(
    half* A,
    int   m,
    int   lda,
    int   j0,
    int   ib,
    int   uc,
    int*  d_ipiv,
    cublasHandle_t cublas_handle,
    cudaStream_t stream,
    half* d_block_val,
    int*  d_block_idx,
    int   num_blocks_pivot_max)
{
    if (!A || !d_ipiv || !d_block_val || !d_block_idx) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!cublas_handle) {
        fprintf(stderr, "launch_panel_TSLU: cublas_handle is null.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= m) return;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int coop_supported = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&coop_supported, cudaDevAttrCooperativeLaunch, dev));
    if (!coop_supported) {
        fprintf(stderr, "launch_panel_TSLU: cooperative launch not supported.\n");
        std::exit(EXIT_FAILURE);
    }

    if (num_blocks_pivot_max < 1) num_blocks_pivot_max = 1;
    if (num_blocks_pivot_max > 128) num_blocks_pivot_max = 128;

    // Micro-block size (kb) controls how many cooperative launches per panel:
    //   launches per panel = ib / kb
    int kb = (uc > 0) ? uc : 32;
    if (kb < 1) kb = 1;
    if (kb > ib) kb = ib;
    if (kb > 32) kb = 32;

    const int threads = 256;

    for (int k0 = 0; k0 < ib; k0 += kb) {
        int kend = k0 + kb;
        if (kend > ib) kend = ib;

        // Choose smaller grid for later micro-blocks / later panels (cheap host-side heuristic)
        const int row_base = j0 + k0;
        const int m_eff = m - row_base;
        int num_blocks = panel_TSLU_choose_blocks_fast(m_eff, num_blocks_pivot_max);

        void* args[] = {
            (void*)&A,
            (void*)&m,
            (void*)&lda,
            (void*)&j0,
            (void*)&ib,
            (void*)&k0,
            (void*)&kend,
            (void*)&d_block_val,
            (void*)&d_block_idx,
            (void*)&num_blocks,
            (void*)&d_ipiv
        };

        CUDA_CHECK(cudaLaunchCooperativeKernel(
            (void*)panel_getf2_microblock_coop_kernel,
            dim3(num_blocks), dim3(threads),
            args, 0, stream));

        // Level-3 update for the rest of the panel columns
        panel_blockout_trsm_gemm_inside_panel(
            A, m, lda, j0, ib, k0, kend,
            cublas_handle,
            stream);
    }

    CUDA_CHECK(cudaGetLastError());
}
