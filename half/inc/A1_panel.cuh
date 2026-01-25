#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

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

#ifndef PANEL_USE_FAST_INV
#define PANEL_USE_FAST_INV 1
#endif

// 0: fast half2 micro-update (default)
// 1: fp32 micro-update (slower, for accuracy, keep OFF for now)
#ifndef PANEL_MICRO_UPDATE_FP32
#define PANEL_MICRO_UPDATE_FP32 0
#endif

// ---------------------------
// device helper: aligned half2 IO
// ---------------------------
static __device__ __forceinline__ bool is_half2_aligned_index(int r) {
    return ((r & 1) == 0);
}
static __device__ __forceinline__ half2 ld_half2_aligned(const half* p) {
    return *reinterpret_cast<const half2*>(p);
}
static __device__ __forceinline__ void st_half2_aligned(half* p, half2 v) {
    *reinterpret_cast<half2*>(p) = v;
}

// =====================================================================================
// Cooperative panel kernel: factorize micro-block [k0, kend) inside one panel
//
// Pivot 语义严格不变：每列全局选最大 |a| 的 pivot row，swap，写 ipiv(1-based)
//
// 性能刀：
//   - 保留 2 次 grid.sync/列：
//       (1) pivot candidates ready -> block0 reduce
//       (2) swap 完成 -> 才能安全更新（因为 row piv 会被更新，swap 必须先完成）
//   - 删除每列末尾那个 grid.sync：更新完成后不需要全网同步，因为：
//       * 每个 block 只写自己的 row tile
//       * 下一列 pivot search 每个 block 只读自己的 row tile
//     全网对齐自然由下一列 pivot-reduce 那个 grid.sync 完成
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
    constexpr int THREADS  = 256;
    constexpr int ROW_TILE = 512;   // 256 threads * 2 rows/thread
    constexpr int MAX_KB   = 32;

    cg::grid_group grid = cg::this_grid();

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE;

    // per-block pivot reduce
    __shared__ float s_warp_val_f[NUM_WARPS];
    __shared__ int   s_warp_idx[NUM_WARPS];

    // pivot row cache (per-block)
    __shared__ half  sU[MAX_KB];
    __shared__ half2 sNegU2[MAX_KB]; // -U (half2) to avoid extra ops in inner loop

    // block0 pivot row
    __shared__ int   s_piv_row;

    const half2 minus1 = __float2half2_rn(-1.0f);

    for (int k = k0; k < kend; ++k) {
        const int col_k = j0 + k;
        const int row_k = j0 + k;
        if (row_k >= m) break;

        const size_t col_off = (size_t)col_k * (size_t)lda;

        // ------------------------------
        // 1) Pivot search: each block scans its tile rows for column col_k
        // ------------------------------
        const int tile_start = row_k + (int)blockIdx.x * ROW_TILE;

        float local_max_f = 0.0f;
        int   local_idx   = row_k;

        int r0 = tile_start + tid * 2;
        int r1 = r0 + 1;

        if (r0 < m) {
            if (r1 < m && is_half2_aligned_index(r0)) {
                const half* p = A + (size_t)r0 + col_off;
                half2 a2 = ld_half2_aligned(p);
                float2 f2 = __half22float2(a2);
                float v0 = fabsf(f2.x);
                float v1 = fabsf(f2.y);
                if (v0 > local_max_f) { local_max_f = v0; local_idx = r0; }
                if (v1 > local_max_f) { local_max_f = v1; local_idx = r1; }
            } else {
                float v0 = fabsf(__half2float(A[(size_t)r0 + col_off]));
                if (v0 > local_max_f) { local_max_f = v0; local_idx = r0; }
                if (r1 < m) {
                    float v1 = fabsf(__half2float(A[(size_t)r1 + col_off]));
                    if (v1 > local_max_f) { local_max_f = v1; local_idx = r1; }
                }
            }
        }

        // warp reduce (float + idx)
        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            float ov = __shfl_down_sync(0xffffffff, local_max_f, off);
            int   oi = __shfl_down_sync(0xffffffff, local_idx,   off);
            if (ov > local_max_f) { local_max_f = ov; local_idx = oi; }
        }
        if (lane == 0) {
            s_warp_val_f[warp] = local_max_f;
            s_warp_idx[warp]   = local_idx;
        }
        __syncthreads();

        // warp0 -> per-block max
        if (warp == 0) {
            float vmax = (lane < NUM_WARPS) ? s_warp_val_f[lane] : 0.0f;
            int   vidx = (lane < NUM_WARPS) ? s_warp_idx[ lane] : row_k;

            for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                float ov = __shfl_down_sync(0xffffffff, vmax, off);
                int   oi = __shfl_down_sync(0xffffffff, vidx, off);
                if (ov > vmax) { vmax = ov; vidx = oi; }
            }
            if (lane == 0) {
                block_val[blockIdx.x] = __float2half_rn(vmax);
                block_idx[blockIdx.x] = vidx;
            }
        }

        // ✅ grid.sync #1: ensure block_val/block_idx are ready for block0 reduce
        grid.sync();

        // ------------------------------
        // 2) block0 reduce across blocks -> pivot row, then swap rows across panel columns
        // ------------------------------
        if (blockIdx.x == 0) {
            if (warp == 0) {
                float vmax = 0.0f;
                int   vidx = row_k;

                for (int i = lane; i < num_blocks; i += WARP_SIZE) {
                    float v = __half2float(block_val[i]);
                    int   r = block_idx[i];
                    if (v > vmax) { vmax = v; vidx = r; }
                }
                for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
                    float ov = __shfl_down_sync(0xffffffff, vmax, off);
                    int   oi = __shfl_down_sync(0xffffffff, vidx, off);
                    if (ov > vmax) { vmax = ov; vidx = oi; }
                }
                if (lane == 0) {
                    s_piv_row     = vidx;
                    d_ipiv[row_k] = vidx + 1; // 1-based
                }
            }
            __syncthreads();

            const int piv = s_piv_row;
            if (piv != row_k) {
                // swap rows across panel columns [j0, j0+ib)
                // 注意：这个 swap 必须发生在 update 之前，因为 row piv 会被 update
                for (int j = j0 + tid; j < j0 + ib; j += THREADS) {
                    size_t off = (size_t)j * (size_t)lda;
                    half tmp = A[(size_t)row_k + off];
                    A[(size_t)row_k + off] = A[(size_t)piv + off];
                    A[(size_t)piv + off] = tmp;
                }
            }
        }

        // ✅ grid.sync #2: swap finished; now safe to read pivot row at row_k and update rows below
        grid.sync();

        // ------------------------------
        // 3) Scale + update within this micro-block
        // ------------------------------
        const half pivot = A[(size_t)row_k + col_off];
        const bool singular = (pivot == __float2half(0.0f));

        float inv_piv_f = 0.0f;
        if (!singular) {
            float piv_f = __half2float(pivot);
#if PANEL_USE_FAST_INV
            inv_piv_f = __fdividef(1.0f, piv_f);
#else
            inv_piv_f = 1.0f / piv_f;
#endif
        }
        half  inv_piv_h  = __float2half(inv_piv_f);
        half2 inv_piv_h2 = __half2half2(inv_piv_h);

        const int col_begin = col_k + 1;
        const int col_end   = j0 + kend;
        int num_u = col_end - col_begin; // <= 31 for kend-k0<=32
        if (num_u < 0) num_u = 0;
        if (num_u > MAX_KB) num_u = MAX_KB;

        // Load pivot row U segment into shared, and -U as half2 for fast FMA
        if (tid < num_u) {
            const int c = col_begin + tid;
            half u = A[(size_t)row_k + (size_t)c * (size_t)lda];
            sU[tid] = u;
            half2 u2 = __half2half2(u);
            sNegU2[tid] = __hmul2(u2, minus1); // -u
        }
        __syncthreads();

        // Update rows below pivot (disjoint row tiles across blocks)
        const int tile_u = (row_k + 1) + (int)blockIdx.x * ROW_TILE;
        const int rr = tile_u + tid * 2;

        if (!singular && rr < m) {
            const bool has2 = (rr + 1 < m);
            const bool aligned = has2 && is_half2_aligned_index(rr);

#if PANEL_MICRO_UPDATE_FP32
            // optional accuracy path (OFF by default)
            const float inv_piv = inv_piv_f;

            float a0f = __half2float(A[(size_t)rr + col_off]);
            float a1f = has2 ? __half2float(A[(size_t)(rr + 1) + col_off]) : 0.0f;

            float l0 = a0f * inv_piv;
            float l1 = a1f * inv_piv;

            A[(size_t)rr + col_off] = __float2half_rn(l0);
            if (has2) A[(size_t)(rr + 1) + col_off] = __float2half_rn(l1);

            for (int t = 0; t < num_u; ++t) {
                const int c = col_begin + t;
                const size_t coff = (size_t)c * (size_t)lda;
                const float u = __half2float(sU[t]);

                float v0 = __half2float(A[(size_t)rr + coff]);
                v0 -= l0 * u;
                A[(size_t)rr + coff] = __float2half_rn(v0);

                if (has2) {
                    float v1 = __half2float(A[(size_t)(rr + 1) + coff]);
                    v1 -= l1 * u;
                    A[(size_t)(rr + 1) + coff] = __float2half_rn(v1);
                }
            }
#else
            // fast half2 path
            half2 a2;
            if (aligned) {
                a2 = ld_half2_aligned(A + (size_t)rr + col_off);
            } else {
                half a0 = A[(size_t)rr + col_off];
                half a1 = has2 ? A[(size_t)(rr + 1) + col_off] : __float2half(0.0f);
                a2 = __halves2half2(a0, a1);
            }

            half2 L2 = __hmul2(a2, inv_piv_h2);

            // store L back to column col_k
            if (aligned) {
                st_half2_aligned(A + (size_t)rr + col_off, L2);
            } else {
                A[(size_t)rr + col_off] = L2.x;
                if (has2) A[(size_t)(rr + 1) + col_off] = L2.y;
            }

            for (int t = 0; t < num_u; ++t) {
                const int c = col_begin + t;
                const size_t coff = (size_t)c * (size_t)lda;

                half2 Av2;
                if (aligned) {
                    Av2 = ld_half2_aligned(A + (size_t)rr + coff);
                } else {
                    half v0 = A[(size_t)rr + coff];
                    half v1 = has2 ? A[(size_t)(rr + 1) + coff] : __float2half(0.0f);
                    Av2 = __halves2half2(v0, v1);
                }

#if __CUDA_ARCH__ >= 530
                half2 R2 = __hfma2(L2, sNegU2[t], Av2); // Av2 - L2*U
#else
                // fallback
                half2 U2 = __hmul2(sNegU2[t], minus1);
                half2 R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif

                if (aligned) {
                    st_half2_aligned(A + (size_t)rr + coff, R2);
                } else {
                    A[(size_t)rr + coff] = R2.x;
                    if (has2) A[(size_t)(rr + 1) + coff] = R2.y;
                }
            }
#endif
        }

        // 关键：这里不再 grid.sync()！
        // 下一列 pivot reduction 那个 grid.sync 会自然对齐所有 block，
        // 且 pivot search 每个 block 只读自己更新过的 row tile，不需要全网同步。
    }
}

// =====================================================================================
// TRSM (panel-internal) + GEMM (panel-internal)
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
        sL[i + j * K_MAX] = A[(j0_k0 + i) + (size_t)(j0_k0 + j) * (size_t)lda];
    }
    __syncthreads();

    const int warp = (int)threadIdx.x / WARP_SIZE;
    const int lane = (int)threadIdx.x & (WARP_SIZE - 1);

    const int rhs = (int)blockIdx.x * (int)(blockDim.x / WARP_SIZE) + warp;
    if (rhs >= N) return;

    half* colptr = U12 + (size_t)rhs * (size_t)lda;

    for (int i = 0; i < K; ++i) {
        float bi = 0.0f;
        if (lane == 0) bi = __half2float(colptr[i]);

        float acc = 0.0f;
        for (int kk = lane; kk < i; kk += WARP_SIZE) {
            float Lik = __half2float(sL[i + kk * K_MAX]);
            float xk  = __half2float(colptr[kk]);
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

    if (K <= 0 || N <= 0) return;
    if (K > 32) {
        fprintf(stderr, "panel_blockout_trsm_gemm_inside_panel: K=%d > 32 not supported.\n", K);
        std::exit(EXIT_FAILURE);
    }

    half* U12 = A + j0_k0 + (size_t)col2 * (size_t)lda;

    constexpr int WARPS_PER_BLOCK = 4;
    dim3 block(WARPS_PER_BLOCK * WARP_SIZE);
    dim3 grid((N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);

    panel_trsm_u12_warp_kernel<32><<<grid, block, 0, stream>>>(
        A, U12, lda, j0_k0, K, N);
    CUDA_CHECK(cudaGetLastError());

    if (M <= 0) return;

    half* L21 = A + row2 + (size_t)j0_k0 * (size_t)lda;
    half* A22 = A + row2 + (size_t)col2  * (size_t)lda;

    const float alpha = -1.0f;
    const float beta  =  1.0f;

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

//  workspace 计算，返回需要多少个 block 对 panel 进行处理
inline int panel_TSLU_required_pivot_blocks(int m, int j0)
{
    // 计算有效行数
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

    int sm_count = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));

    int max_blocks_per_sm = 0;
    // 得到每个 sm 上最多可以同时跑几个我这样的 block
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm,
        panel_getf2_microblock_coop_kernel, // 函数指针
        256, // 每个 block 的 thread 数
        0)); // 动态共享内存大小

    // 最多可以搞多少个 block
    int max_coop_grid = sm_count * max_blocks_per_sm;
    if (max_coop_grid < 1) 
        max_coop_grid = 1;
    
    // 每个 block 可以处理 512 行，计算我们总共需要多少个 block 去处理，不允许超过最大值
    int nb = (m_effective + 512 - 1) / 512;
    if (nb < 1) 
        nb = 1;
    if (nb > max_coop_grid) 
        nb = max_coop_grid;
    if (nb < 1) 
        nb = 1;

    return nb;
}

static inline int panel_TSLU_choose_blocks_fast(int m_effective, int num_blocks_max)
{
    if (m_effective <= 0) return 1;
    int nb = (m_effective + 512 - 1) / 512;
    if (nb < 1) nb = 1;
    if (nb > num_blocks_max) nb = num_blocks_max;
    if (nb < 1) nb = 1;
    return nb;
}

// 对外暴露接口
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

    int kb = (uc > 0) ? uc : 16;
    if (kb < 1) kb = 1;
    if (kb > ib) kb = ib;
    if (kb > 32) kb = 32;

    const int threads = 256;

    for (int k0 = 0; k0 < ib; k0 += kb) {
        int kend = k0 + kb;
        if (kend > ib) kend = ib;

        const int row_base = j0 + k0;
        const int m_eff = m - row_base;

        int num_blocks = panel_TSLU_choose_blocks_fast(m_eff, num_blocks_pivot_max);
        if (num_blocks < 1) num_blocks = 1;

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

        panel_blockout_trsm_gemm_inside_panel(
            A, m, lda, j0, ib, k0, kend,
            cublas_handle,
            stream);
    }

    CUDA_CHECK(cudaGetLastError());
}
