// A1_panel.cuh  (next-step optimization: swap-first then prescale; removes one grid.sync)
// External API unchanged.
// Core change in cooperative kernel:
//
// BEFORE (your baseline):
//   grid-reduce -> publish pivot_row -> grid.sync
//   prescale (all blocks) -> grid.sync
//   swap (block0)
//
// NOW (this version):
//   grid-reduce -> publish pivot_row -> grid.sync
//   swap (block0, panel columns) -> grid.sync   <-- only one post-publish barrier
//   prescale (all blocks)                        <-- no barrier needed after; kernel end is fence
//
// This eliminates one expensive grid-wide barrier per k and also switches to the
// numerically standard ordering (swap first, then compute multipliers).

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

using half = __half;

static __device__ __forceinline__ half half_abs(half x) { return __habs(x); }

////////////////////////////////////////////////////////////////////////////////
// 1) cooperative pivot + panel swap + prescale
////////////////////////////////////////////////////////////////////////////////
__global__ void panel_pivot_prescale_and_swap_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    half* __restrict__ block_val,
    int*  __restrict__ block_idx,
    int num_blocks,
    int* __restrict__ d_ipiv) // 1-based output for outer code
{
    extern __shared__ unsigned char smem[];
    const int tid      = threadIdx.x;
    const int lane     = tid & (WARP_SIZE - 1);
    const int warp_id  = tid / WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    half* s_val = reinterpret_cast<half*>(smem);
    int*  s_idx = reinterpret_cast<int*>(s_val + blockDim.x);

    const int col_k = j0 + k;
    const int row_k = j0 + k;
    if (col_k >= m) return;

    // (1) thread-local scan
    half local_max_val = __float2half(0.0f);
    int  local_max_idx = row_k;

    const int global_stride = blockDim.x * gridDim.x;
    for (int idx = row_k + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col_k * lda];
        half v = half_abs(a);
        if (v > local_max_val) { local_max_val = v; local_max_idx = idx; }
    }

    // (2) warp reduce
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        half ov = __shfl_down_sync(0xffffffff, local_max_val, off);
        int  oi = __shfl_down_sync(0xffffffff, local_max_idx, off);
        if (ov > local_max_val) { local_max_val = ov; local_max_idx = oi; }
    }
    if (lane == 0) { s_val[warp_id] = local_max_val; s_idx[warp_id] = local_max_idx; }
    __syncthreads();

    // (3) block reduce (warp0)
    if (warp_id == 0) {
        half vmax = (lane < num_warps) ? s_val[lane] : __float2half(0.0f);
        int  vidx = (lane < num_warps) ? s_idx[lane] : row_k;

        for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
            half ov = __shfl_down_sync(0xffffffff, vmax, off);
            int  oi = __shfl_down_sync(0xffffffff, vidx, off);
            if (ov > vmax) { vmax = ov; vidx = oi; }
        }
        if (lane == 0) { block_val[blockIdx.x] = vmax; block_idx[blockIdx.x] = vidx; }
    }

    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // (4) grid reduce in block0 -> write ipiv (output) + publish pivot_row (internal)
    if (blockIdx.x == 0) {
        half vmax = __float2half(0.0f);
        int  vidx = row_k;

        for (int i = tid; i < num_blocks; i += blockDim.x) {
            half v = block_val[i];
            int  r = block_idx[i];
            if (v > vmax) { vmax = v; vidx = r; }
        }

        s_val[tid] = vmax;
        s_idx[tid] = vidx;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && s_val[tid + stride] > s_val[tid]) {
                s_val[tid] = s_val[tid + stride];
                s_idx[tid] = s_idx[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const int piv = s_idx[0];   // 0-based
            d_ipiv[row_k] = piv + 1;    // output for outer use
            block_idx[0]  = piv;        // publish pivot_row for internal use
        }
    }

    // make pivot_row visible
    grid.sync();

    const int pivot_row = block_idx[0]; // 0-based

    // (5) swap FIRST (panel columns only), then sync so all blocks see swapped A
    if (blockIdx.x == 0 && pivot_row != row_k) {
        for (int j = j0 + tid; j < j0 + ib; j += blockDim.x) {
            size_t off = (size_t)j * lda;
            half tmp = A[row_k + off];
            A[row_k + off] = A[pivot_row + off];
            A[pivot_row + off] = tmp;
        }
    }

    // ensure swap completed before prescale reads pivot and column
    grid.sync();

    // (6) prescale AFTER swap: standard LU multipliers
    // pivot is now at A[row_k, col_k]
    const half pivot = A[row_k + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) return;

    for (int r = row_k + 1 + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
        A[r + (size_t)col_k * lda] = __hdiv(A[r + (size_t)col_k * lda], pivot);
    }
    // no extra grid.sync: kernel end is a completion point for subsequent kernels in the stream
}

////////////////////////////////////////////////////////////////////////////////
// 2) panel-internal update (per k), half2 on rows, COL_TILE columns per block
////////////////////////////////////////////////////////////////////////////////
template<int ROW_TILE, int COL_TILE>
__global__ void panel_update_kernel_range_tiled_half2(
    half* __restrict__ A,
    int m, int lda,
    int j0,
    int col_end, // absolute exclusive
    int k)
{
    const int col_k = j0 + k;
    const int row_k = j0 + k;
    if (col_k >= m) return;

    const int c0 = (col_k + 1) + (int)blockIdx.x * COL_TILE;

    const int r0 = (row_k + 1) + (int)blockIdx.y * ROW_TILE + ((int)threadIdx.x * 2);
    if (r0 >= m) return;

    half2 L2;
    L2.x = __ldg(&A[(r0 + 0) + (size_t)col_k * lda]);
    L2.y = (r0 + 1 < m) ? __ldg(&A[(r0 + 1) + (size_t)col_k * lda]) : __float2half(0.0f);

    #pragma unroll
    for (int t = 0; t < COL_TILE; ++t) {
        const int c = c0 + t;
        if (c >= col_end) break;

        const half U  = __ldg(&A[row_k + (size_t)c * lda]);
        const half2 U2 = __half2half2(U);

        half2 Av2;
        Av2.x = A[(r0 + 0) + (size_t)c * lda];
        Av2.y = (r0 + 1 < m) ? A[(r0 + 1) + (size_t)c * lda] : __float2half(0.0f);

        const half2 R2 = __hsub2(Av2, __hmul2(L2, U2));

        A[(r0 + 0) + (size_t)c * lda] = R2.x;
        if (r0 + 1 < m) A[(r0 + 1) + (size_t)c * lda] = R2.y;
    }
}

////////////////////////////////////////////////////////////////////////////////
// 3) TRSM for panel-only U12 formation: U12 = L11^{-1} * A12 (K<=32)
//    One warp per RHS column.
////////////////////////////////////////////////////////////////////////////////
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

////////////////////////////////////////////////////////////////////////////////
// 4) cuBLAS handle (cached; only GEMM used)
////////////////////////////////////////////////////////////////////////////////
static inline cublasHandle_t& panel_cublas_handle_ref() { static cublasHandle_t h = nullptr; return h; }
static inline int& panel_cublas_dev_ref() { static int dev = -1; return dev; }

static inline cublasHandle_t panel_get_cublas_handle()
{
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cublasHandle_t& h = panel_cublas_handle_ref();
    int& cached = panel_cublas_dev_ref();

    if (!h || cached != dev) {
        if (h) { cublasDestroy(h); h = nullptr; }
        CUBLAS_CHECK(cublasCreate(&h));
        cached = dev;
    }
    return h;
}

////////////////////////////////////////////////////////////////////////////////
// 5) panel-only block-out update for a kb block: TRSM(U12) + GEMM(A22)
////////////////////////////////////////////////////////////////////////////////
static inline void panel_blockout_trsm_gemm_inside_panel(
    half* A, int m, int lda,
    int j0, int ib,
    int k0, int kend,
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

    cublasHandle_t h = panel_get_cublas_handle();
    CUBLAS_CHECK(cublasSetStream(h, stream));

    CUBLAS_CHECK(cublasGemmEx(
        h,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        L21, CUDA_R_16F, lda,
        U12, CUDA_R_16F, lda,
        &beta,
        A22, CUDA_R_16F, lda,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}

////////////////////////////////////////////////////////////////////////////////
// 6) required pivot blocks
////////////////////////////////////////////////////////////////////////////////
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
        panel_pivot_prescale_and_swap_coop_kernel,
        threads_pivot,
        (int)shmem_coop));

    int max_coop_grid = sm_count * max_blocks_per_sm;
    if (max_coop_grid < 1) max_coop_grid = 1;

    int rows_per_block;
    if (m_effective >= 24576) rows_per_block = 1024;
    else if (m_effective >= 12288) rows_per_block = 512;
    else if (m_effective >= 4096)  rows_per_block = 256;
    else rows_per_block = 128;

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

////////////////////////////////////////////////////////////////////////////////
// 7) external interface (UNCHANGED)
////////////////////////////////////////////////////////////////////////////////
inline void launch_panel_TSLU(
    half* A,
    int   m,
    int   lda,
    int   j0,
    int   ib,
    int   uc,
    int*  d_ipiv,          // 1-based output
    cudaStream_t stream,
    half* d_block_val,
    int*  d_block_idx,
    int   num_blocks_pivot)
{
    if (!A || !d_ipiv || !d_block_val || !d_block_idx) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
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

    if (num_blocks_pivot < 1) num_blocks_pivot = 1;
    if (num_blocks_pivot > 64) num_blocks_pivot = 64;

    const int threads_pivot = 256;
    const size_t shmem_coop = (size_t)threads_pivot * (sizeof(half) + sizeof(int));

    int kb = (uc > 0) ? uc : 16;
    if (kb < 1) kb = 1;
    if (kb > ib) kb = ib;
    if (kb > 32) kb = 32;

    constexpr int COL_TILE = 8;
    constexpr int ROW_TILE = 512;
    dim3 block_u(ROW_TILE / 2);

    for (int k0 = 0; k0 < ib; k0 += kb) {
        int kend = k0 + kb;
        if (kend > ib) kend = ib;

        for (int k = k0; k < kend; ++k) {
            const int col = j0 + k;
            if (col >= m) break;

            void* args[] = {
                (void*)&A, (void*)&m, (void*)&lda,
                (void*)&j0, (void*)&ib, (void*)&k,
                (void*)&d_block_val, (void*)&d_block_idx,
                (void*)&num_blocks_pivot, (void*)&d_ipiv
            };

            CUDA_CHECK(cudaLaunchCooperativeKernel(
                (void*)panel_pivot_prescale_and_swap_coop_kernel,
                dim3(num_blocks_pivot), dim3(threads_pivot),
                args, shmem_coop, stream));

            const int rows_rem = m - (j0 + k + 1);
            const int cols_in  = (j0 + kend) - (j0 + k + 1);
            if (rows_rem > 0 && cols_in > 0) {
                const int grid_x = (cols_in + COL_TILE - 1) / COL_TILE;
                const int grid_y = (rows_rem + ROW_TILE - 1) / ROW_TILE;

                panel_update_kernel_range_tiled_half2<ROW_TILE, COL_TILE>
                    <<<dim3(grid_x, grid_y), block_u, 0, stream>>>(
                        A, m, lda, j0, j0 + kend, k);
            }
        }

        panel_blockout_trsm_gemm_inside_panel(A, m, lda, j0, ib, k0, kend, stream);
    }

    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_panel_buffers()
{
    cublasHandle_t& h = panel_cublas_handle_ref();
    if (h) { cublasDestroy(h); h = nullptr; }
    panel_cublas_dev_ref() = -1;
}
