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

static __device__ __forceinline__ half half_abs(half x) { return __habs(x); }

////////////////////////////////////////////////////////////////////////////////
// cooperative pivot + prescale + panel swap（保留你的语义）
////////////////////////////////////////////////////////////////////////////////
__global__ void panel_pivot_prescale_and_swap_coop_kernel(
    half* __restrict__ A,
    int m, int lda,
    int j0, int ib,
    int k,
    half* __restrict__ block_val,
    int*  __restrict__ block_idx,
    int num_blocks,
    int* __restrict__ d_ipiv) // 1-based
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

    half local_max_val = __float2half(0.0f);
    int  local_max_idx = row_k;

    const int global_stride = blockDim.x * gridDim.x;

    for (int idx = row_k + blockIdx.x * blockDim.x + tid; idx < m; idx += global_stride) {
        half a = A[idx + (size_t)col_k * lda];
        half v = half_abs(a);
        if (v > local_max_val) { local_max_val = v; local_max_idx = idx; }
    }

    for (int off = WARP_SIZE / 2; off > 0; off >>= 1) {
        half ov = __shfl_down_sync(0xffffffff, local_max_val, off);
        int  oi = __shfl_down_sync(0xffffffff, local_max_idx, off);
        if (ov > local_max_val) { local_max_val = ov; local_max_idx = oi; }
    }

    if (lane == 0) { s_val[warp_id] = local_max_val; s_idx[warp_id] = local_max_idx; }
    __syncthreads();

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
        if (tid == 0) d_ipiv[row_k] = s_idx[0] + 1;
    }

    grid.sync();

    const int pivot_row = d_ipiv[row_k] - 1;
    const half pivot = A[pivot_row + (size_t)col_k * lda];
    if (pivot == __float2half(0.0f)) return;

    // prescale before swap（保持你的语义）
    if (pivot_row == row_k) {
        for (int r = row_k + 1 + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            A[r + (size_t)col_k * lda] = __hdiv(A[r + (size_t)col_k * lda], pivot);
        }
    } else {
        for (int r = row_k + blockIdx.x * blockDim.x + tid; r < m; r += global_stride) {
            if (r == pivot_row) continue;
            A[r + (size_t)col_k * lda] = __hdiv(A[r + (size_t)col_k * lda], pivot);
        }
    }

    grid.sync();

    // swap rows across panel columns [j0, j0+ib)
    if (blockIdx.x == 0 && pivot_row != row_k) {
        for (int j = j0 + tid; j < j0 + ib; j += blockDim.x) {
            size_t off = (size_t)j * lda;
            half tmp = A[row_k + off];
            A[row_k + off] = A[pivot_row + off];
            A[pivot_row + off] = tmp;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// block-internal update: update columns [col_k+1, col_end) for rows [row_k+1, m)
////////////////////////////////////////////////////////////////////////////////
__global__ void panel_update_kernel_range_scalar(
    half* __restrict__ A,
    int m, int lda,
    int j0,
    int col_end,
    int k)
{
    const int col_k = j0 + k;
    const int row_k = j0 + k;
    if (col_k >= m) return;

    const int c = (col_k + 1) + blockIdx.x;
    if (c >= col_end) return;

    const int r = row_k + 1 + blockIdx.y * 256 + threadIdx.x;
    if (r >= m) return;

    const half L = __ldg(&A[r + (size_t)col_k * lda]);
    const half U = __ldg(&A[row_k + (size_t)c * lda]);
    A[r + (size_t)c * lda] = __hsub(A[r + (size_t)c * lda], __hmul(L, U));
}

////////////////////////////////////////////////////////////////////////////////
// small TRSM kernel: solve unit-lower L (KxK) * X (KxN) = B (KxN) in-place
// K <= 32 recommended (use uc=16/32)
// Uses FP32 accumulate for stability.
////////////////////////////////////////////////////////////////////////////////
__global__ void trsm_unit_lower_small_kernel(
    const half* __restrict__ L, // (KxK) at A[j0+k0, j0+k0]
    half* __restrict__ B,       // (KxN) at A[j0+k0, col2]  (in/out)
    int lda,
    int K,
    int N)
{
    // One block per column of B (one RHS).
    const int j = (int)blockIdx.x;
    if (j >= N) return;

    // Use thread0 to do forward substitution for this RHS (K is small).
    if (threadIdx.x == 0) {
        // x[i] overwrites B[i, j]
        for (int i = 0; i < K; ++i) {
            float sum = __half2float(B[i + (size_t)j * lda]); // B is column-major
            // subtract L(i,0..i-1) * x(0..i-1)
            for (int t = 0; t < i; ++t) {
                float Lit = __half2float(L[i + (size_t)t * lda]);
                float xt  = __half2float(B[t + (size_t)j * lda]);
                sum -= Lit * xt;
            }
            // unit diagonal => no divide
            B[i + (size_t)j * lda] = __float2half(sum);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// cuBLAS handle management (only GEMM needed)
////////////////////////////////////////////////////////////////////////////////
static inline cublasHandle_t& panel_cublas_handle_ref() {
    static cublasHandle_t h = nullptr;
    return h;
}
static inline int& panel_cublas_dev_ref() {
    static int dev = -1;
    return dev;
}
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
// Update after each kb block:
// 1) U12 = L11^{-1} * A12 (small kernel)
// 2) A22 -= L21 * U12     (cublasGemmEx)
////////////////////////////////////////////////////////////////////////////////
static inline void panel_form_u12_and_gemm_update(
    half* A, int m, int lda,
    int j0, int ib,
    int k0, int kend,
    cudaStream_t stream)
{
    const int row2 = j0 + kend;
    const int col2 = j0 + kend;

    const int K = kend - k0;
    const int N = (j0 + ib) - col2;
    const int M = m - row2;

    if (K <= 0 || N <= 0) return;

    // L11: (KxK) lower unit at A[j0+k0, j0+k0]
    half* L11 = A + (j0 + k0) + (size_t)(j0 + k0) * lda;
    // U12 storage: overwrite A12 in place at A[j0+k0, col2]
    half* U12 = A + (j0 + k0) + (size_t)col2 * lda;

    // Solve L11 * U12 = A12 for each RHS column (N columns)
    // Grid: N blocks, 32 threads each (only thread0 active; could be optimized later)
    trsm_unit_lower_small_kernel<<<dim3(N), dim3(32), 0, stream>>>(
        L11, U12, lda, K, N);

    if (M <= 0) return;

    // GEMM: A22 -= L21 * U12
    half* L21 = A + row2 + (size_t)(j0 + k0) * lda; // (MxK)
    half* A22 = A + row2 + (size_t)col2 * lda;      // (MxN)

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
// required pivot blocks
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
    else if (m_effective >= 4096) rows_per_block = 256;
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
// external interface (UNCHANGED): uc used as kb (default 16)
////////////////////////////////////////////////////////////////////////////////
inline void launch_panel_TSLU(
    half* A,
    int   m,
    int   lda,
    int   j0,
    int   ib,
    int   uc,
    int*  d_ipiv,
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

    const int ROW_TILE = 256;
    dim3 block_u(ROW_TILE);

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

            // update only within block columns up to kend
            const int rows_rem = m - (j0 + k + 1);
            const int cols_in  = (j0 + kend) - (j0 + k + 1);
            if (rows_rem > 0 && cols_in > 0) {
                const int grid_x = cols_in;
                const int grid_y = (rows_rem + ROW_TILE - 1) / ROW_TILE;
                panel_update_kernel_range_scalar<<<dim3(grid_x, grid_y), block_u, 0, stream>>>(
                    A, m, lda, j0, j0 + kend, k);
            }
        }

        // form U12 and gemm update A22
        panel_form_u12_and_gemm_update(A, m, lda, j0, ib, k0, kend, stream);
    }

    CUDA_CHECK(cudaGetLastError());
}

inline void cleanup_panel_buffers()
{
    cublasHandle_t& h = panel_cublas_handle_ref();
    if (h) { cublasDestroy(h); h = nullptr; }
    panel_cublas_dev_ref() = -1;
}
