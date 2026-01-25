#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#ifndef HGETRF_USE_CUBLASLT
#define HGETRF_USE_CUBLASLT 1
#endif
#if HGETRF_USE_CUBLASLT
#include <cublasLt.h>
#endif
#include <cooperative_groups.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>
#include <vector>
#include <cmath>


#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d (status=%d)\n",                \
                    __FILE__, __LINE__, (int)st__);                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;
using half2 = __half2;

// 1: 开启左侧异步交换 (推荐)
// 0: 关闭
#ifndef HGETRF_EXCH_SPLIT_LEFT
#define HGETRF_EXCH_SPLIT_LEFT 1
#endif

// TRSM 路线
// 0: 原有 A12 kernel
// 1: inv(L11) + GEMM (类似 hpotrf 的 TRSM 处理)
#ifndef HGETRF_TRSM_MODE
#define HGETRF_TRSM_MODE 0
#endif

// hgetrf 句柄，管理所有的资源
struct hgetrfHandle {
    // panel 与 update 过程中需要使用的 cublas 句柄
    cublasHandle_t cublas_update = nullptr;
    cublasHandle_t cublas_panel  = nullptr;
#if HGETRF_USE_CUBLASLT
    cublasLtHandle_t cublaslt = nullptr;
#endif
    
    // 主流应当是外面传入的，销毁也应当是放在整个 hgetrf 之外的
    cudaStream_t stream_main = 0; 

    // panel 流
    cudaStream_t stream_panel = nullptr; 

#if HGETRF_EXCH_SPLIT_LEFT
    // stream_exch_left: 异步交换专用流（通常内部创建为低优先级），Handle 默认拥有
    cudaStream_t stream_exch_left = nullptr;
#endif

    // --- 同步事件 ---
    cudaEvent_t ev_piv_ready  = nullptr;
    cudaEvent_t ev_next_ready = nullptr;
#if HGETRF_EXCH_SPLIT_LEFT
    cudaEvent_t ev_exch_left_done = nullptr;
#endif

    // true 表示该资源由 Handle 创建，Destroy 时需要释放
    // false 表示该资源由用户 Set 进来，Destroy 时不释放
    bool owns_cublas_update = false;
    bool owns_cublas_panel  = false;
    bool owns_stream_panel  = false;
#if HGETRF_USE_CUBLASLT
    bool owns_cublaslt = false;
#endif
#if HGETRF_EXCH_SPLIT_LEFT
    bool owns_stream_exch_left = false;
#endif

    // 算法初始化参数可设置
    int panel_width = 256;
    int uc = 32;
    int trsm_mode = HGETRF_TRSM_MODE;
    int trsm_tile = 4096;
#if HGETRF_USE_CUBLASLT
    size_t lt_workspace_bytes = 32 * 1024 * 1024;
#endif

    // GEMM 计算类型：默认 32F accumulate；fastMath 时切到 16F
    cublasComputeType_t gemm_compute = CUBLAS_COMPUTE_32F;

    // workspace
    int    m_cached_max = 0;
    int    n_cached_max = 0;
    int    num_blocks_pivot_max = 0;
    size_t workspace_bytes = 0;
};

using hgetrfHandle_t = hgetrfHandle*;

// ----------------------------------------------------------------------------
// Setter API: 允许用户注入自定义资源
// ----------------------------------------------------------------------------

// 设置主流 (Update Stream)
// 这通常是用户的主计算流。Handle 只是引用它，不负责销毁。
inline void hgetrfSetStream(hgetrfHandle_t h, cudaStream_t stream) {
    if (!h) return;
    h->stream_main = stream;
    // 立即更新 update 句柄绑定的流
    if (h->cublas_update) {
        cublasSetStream(h->cublas_update, h->stream_main);
    }
}

// 设置 Panel 流 (建议高优先级)
// 用户可以传入自己管理的流。Handle 会释放原本内部创建的流（如果存在），并引用新流。
inline void hgetrfSetPanelStream(hgetrfHandle_t h, cudaStream_t stream) {
    if (!h) return;
    // 如果之前持有内部流，先销毁
    if (h->owns_stream_panel && h->stream_panel) {
        cudaStreamDestroy(h->stream_panel);
    }
    h->stream_panel = stream;
    h->owns_stream_panel = false; // 标记为外部注入，不拥有所有权
    
    // 立即更新 panel 句柄绑定的流
    if (h->cublas_panel) {
        cublasSetStream(h->cublas_panel, h->stream_panel);
    }
}

#if HGETRF_EXCH_SPLIT_LEFT
// 设置后台交换流 (建议低优先级)
inline void hgetrfSetExchangeLeftStream(hgetrfHandle_t h, cudaStream_t stream) {
    if (!h) return;
    if (h->owns_stream_exch_left && h->stream_exch_left) {
        cudaStreamDestroy(h->stream_exch_left);
    }
    h->stream_exch_left = stream;
    h->owns_stream_exch_left = false; // 标记为外部注入
}
#endif

inline void hgetrfSetPanelWidth(hgetrfHandle_t h, int width) { if(h) h->panel_width = width; }
inline void hgetrfSetUc(hgetrfHandle_t h, int uc) { if(h) h->uc = uc; }
inline void hgetrfSetTrsmMode(hgetrfHandle_t h, int mode) { if (h) h->trsm_mode = mode; }
inline void hgetrfSetTrsmTile(hgetrfHandle_t h, int tile) { if (h) h->trsm_tile = tile; }
#if HGETRF_USE_CUBLASLT
inline void hgetrfSetLtWorkspaceBytes(hgetrfHandle_t h, size_t bytes) { if (h) h->lt_workspace_bytes = bytes; }
#endif

// enable=1：TRSM/GEMM 用 CUBLAS_COMPUTE_16F（更快、精度更差）
// enable=0：回到 CUBLAS_COMPUTE_32F（更稳）
inline void hgetrfSetFastMath(hgetrfHandle_t h, int enable) {
    if (!h) return;
    h->gemm_compute = enable ? CUBLAS_COMPUTE_16F : CUBLAS_COMPUTE_32F;
}

// ----------------------------------------------------------------------------
// Create / Destroy: 生命周期管理
// ----------------------------------------------------------------------------
inline void hgetrfCreate(hgetrfHandle_t* out)
{
    if (!out) return;
    hgetrfHandle_t h = new hgetrfHandle;

    // 1. 创建 cuBLAS 句柄
    CUBLAS_CHECK(cublasCreate(&h->cublas_update));
    h->owns_cublas_update = true;
    CUBLAS_CHECK(cublasCreate(&h->cublas_panel));
    h->owns_cublas_panel = true;
#if HGETRF_USE_CUBLASLT
    CUBLAS_CHECK(cublasLtCreate(&h->cublaslt));
    h->owns_cublaslt = true;
#endif

    // 固定为 HOST 指针模式，避免外部改动导致 alpha/beta 解释错误
    CUBLAS_CHECK(cublasSetPointerMode(h->cublas_update, CUBLAS_POINTER_MODE_HOST));
    CUBLAS_CHECK(cublasSetPointerMode(h->cublas_panel,  CUBLAS_POINTER_MODE_HOST));

    // 2. 默认创建 Panel 流 (尝试高优先级)
    {
        int least = 0, greatest = 0;
        cudaError_t st = cudaDeviceGetStreamPriorityRange(&least, &greatest);
        if (st == cudaSuccess) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&h->stream_panel, cudaStreamNonBlocking, greatest));
        } else {
            cudaGetLastError();
            CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream_panel, cudaStreamNonBlocking));
        }
        h->owns_stream_panel = true; // 标记为内部创建
    }

#if HGETRF_EXCH_SPLIT_LEFT
    // 3. 默认创建 Exchange 流 (尝试低优先级)
    {
        int least = 0, greatest = 0;
        cudaError_t st = cudaDeviceGetStreamPriorityRange(&least, &greatest);
        if (st == cudaSuccess) {
            CUDA_CHECK(cudaStreamCreateWithPriority(&h->stream_exch_left, cudaStreamNonBlocking, least));
        } else {
            cudaGetLastError();
            CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream_exch_left, cudaStreamNonBlocking));
        }
        h->owns_stream_exch_left = true; // 标记为内部创建
    }
    // Event
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_exch_left_done, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(h->ev_exch_left_done, h->stream_exch_left));
#endif

    // 4. 创建同步 Event
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_piv_ready,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_next_ready, cudaEventDisableTiming));

    // 5. 设置 Tensor Core 模式
    CUBLAS_CHECK(cublasSetMathMode(h->cublas_update, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetMathMode(h->cublas_panel,  CUBLAS_TENSOR_OP_MATH));

    // 6. 初始绑定
    CUBLAS_CHECK(cublasSetStream(h->cublas_update, h->stream_main)); // default 0
    CUBLAS_CHECK(cublasSetStream(h->cublas_panel,  h->stream_panel));

    *out = h;
}

inline void hgetrfDestroy(hgetrfHandle_t h)
{
    if (!h) return;

    if (h->cublas_update && h->owns_cublas_update) cublasDestroy(h->cublas_update);
    if (h->cublas_panel  && h->owns_cublas_panel)  cublasDestroy(h->cublas_panel);
#if HGETRF_USE_CUBLASLT
    if (h->cublaslt && h->owns_cublaslt) cublasLtDestroy(h->cublaslt);
#endif

    if (h->ev_piv_ready)  cudaEventDestroy(h->ev_piv_ready);
    if (h->ev_next_ready) cudaEventDestroy(h->ev_next_ready);

    // 只有拥有所有权时才销毁流
    if (h->stream_panel && h->owns_stream_panel) {
        cudaStreamDestroy(h->stream_panel);
    }

#if HGETRF_EXCH_SPLIT_LEFT
    if (h->ev_exch_left_done) cudaEventDestroy(h->ev_exch_left_done);
    if (h->stream_exch_left && h->owns_stream_exch_left) {
        cudaStreamDestroy(h->stream_exch_left);
    }
#endif

    delete h;
}

// ----------------------------------------------------------------------------
// Workspace 辅助
// ----------------------------------------------------------------------------
struct HgetrfWorkspaceView {
    half* d_panel_block_val;
    int*  d_panel_block_idx;
    int   num_blocks_pivot_max;

    // TRSM inv+GEMM 路线需要的临时区
    float* d_L_f;   // ib x ib (float)
    float* d_inv_f; // ib x ib (float)
    half*  d_inv_h; // ib x ib (half)
    half*  d_trsm;  // ib x trsm_tile (half)

#if HGETRF_USE_CUBLASLT
    void*  d_lt_workspace;
    size_t lt_workspace_bytes;
#endif
};

static inline size_t align_up(size_t x, size_t a) { return (x + (a - 1)) & ~(a - 1); }

inline int panel_TSLU_required_pivot_blocks(int m, int j0);

inline void hgetrf_bufferSize(hgetrfHandle_t h, int m, int n, const half* dA, int lda, int* lwork) {
    (void)dA; (void)lda;
    if (!h || !lwork) return;

    const int k_total = (m < n) ? m : n;

    int nb = h->panel_width;
    if (nb <= 0) nb = 256;
    if (nb > k_total) nb = k_total;

    int trsm_tile = h->trsm_tile;
    if (trsm_tile <= 0) trsm_tile = 4096;
    if (trsm_tile > n) trsm_tile = n;

    // 得到需要多少个 block 对 panel 进行处理
    int num_blocks = panel_TSLU_required_pivot_blocks(m, 0);
    
    // 对齐到 256 字节，累计需要的字节数，这里的部分适用于存放 pivot 的值与索引
    size_t bytes = 0;
    bytes = align_up(bytes, 256); 
    bytes += sizeof(half) * num_blocks;
    bytes = align_up(bytes, 256); 
    bytes += sizeof(int)  * num_blocks;
    bytes = align_up(bytes, 256); 

    if (h->trsm_mode == 1 && nb > 0) {
        bytes = align_up(bytes, 256);
        bytes += sizeof(float) * (size_t)nb * (size_t)nb; // L_f

        bytes = align_up(bytes, 256);
        bytes += sizeof(float) * (size_t)nb * (size_t)nb; // inv_f

        bytes = align_up(bytes, 256);
        bytes += sizeof(half)  * (size_t)nb * (size_t)nb; // inv_h

        bytes = align_up(bytes, 256);
        bytes += sizeof(half)  * (size_t)nb * (size_t)trsm_tile; // trsm temp
    }

#if HGETRF_USE_CUBLASLT
    if (h->lt_workspace_bytes > 0) {
        bytes = align_up(bytes, 256);
        bytes += h->lt_workspace_bytes;
    }
#endif

    // 申请好的放到 handle 中等待被使用
    h->m_cached_max = m;
    h->n_cached_max = n;
    h->num_blocks_pivot_max = num_blocks;
    h->workspace_bytes = bytes;

    // 返回总字节数转换为 half 的数量,这样外层调用函数申请 workspace 大小时就可以用 half 的大小乘以返回值
    size_t elems = (bytes + sizeof(half) - 1) / sizeof(half);
    *lwork = (int)elems;
}

// 根据前面计算好的部分，把空间分开做好指针，这样就可以直接进行调用
inline HgetrfWorkspaceView hgetrf_workspace_bind(hgetrfHandle_t h, void* d_workspace) {
    HgetrfWorkspaceView ws{};
    if (!h || !d_workspace || h->workspace_bytes == 0) return ws;

    ws.num_blocks_pivot_max = h->num_blocks_pivot_max;

    int nb = h->panel_width;
    if (nb <= 0) nb = 256;
    if (h->m_cached_max > 0 && nb > h->m_cached_max) nb = h->m_cached_max;

    int trsm_tile = h->trsm_tile;
    if (trsm_tile <= 0) trsm_tile = 4096;
    if (h->n_cached_max > 0 && trsm_tile > h->n_cached_max) trsm_tile = h->n_cached_max;

    uint8_t* p = (uint8_t*)d_workspace;
    size_t off = 0;

    off = align_up(off, 256); ws.d_panel_block_val = (half*)(p + off);
    off += sizeof(half) * (size_t)h->num_blocks_pivot_max;

    off = align_up(off, 256); ws.d_panel_block_idx = (int*)(p + off);
    off += sizeof(int) * (size_t)h->num_blocks_pivot_max;

    if (h->trsm_mode == 1 && nb > 0) {
        off = align_up(off, 256); ws.d_L_f = (float*)(p + off);
        off += sizeof(float) * (size_t)nb * (size_t)nb;

        off = align_up(off, 256); ws.d_inv_f = (float*)(p + off);
        off += sizeof(float) * (size_t)nb * (size_t)nb;

        off = align_up(off, 256); ws.d_inv_h = (half*)(p + off);
        off += sizeof(half) * (size_t)nb * (size_t)nb;

        off = align_up(off, 256); ws.d_trsm = (half*)(p + off);
        off += sizeof(half) * (size_t)nb * (size_t)trsm_tile;
    }

#if HGETRF_USE_CUBLASLT
    ws.lt_workspace_bytes = h->lt_workspace_bytes;
    if (ws.lt_workspace_bytes > 0) {
        off = align_up(off, 256); ws.d_lt_workspace = (void*)(p + off);
        off += ws.lt_workspace_bytes;
    }
#endif

    return ws;
}

// ----------------------------------------------------------------------------
// TRSM inv+GEMM helpers (borrowed from hpotrf style)
// ----------------------------------------------------------------------------
__global__ void set_identity_f(float* M, int ld, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
        int row = t % n;
        int col = t / n;
        M[row + col * ld] = (row == col) ? 1.0f : 0.0f;
    }
}

__global__ void f16_to_f32_mat(const half* __restrict__ src, int ld_src,
                              float* __restrict__ dst, int ld_dst,
                              int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
        int row = t % n;
        int col = t / n;
        dst[row + col * ld_dst] = __half2float(src[row + col * ld_src]);
    }
}

__global__ void f32_to_f16_mat(const float* __restrict__ src, int ld_src,
                              half* __restrict__ dst, int ld_dst,
                              int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    for (int t = idx; t < total; t += blockDim.x * gridDim.x) {
        int row = t % n;
        int col = t / n;
        dst[row + col * ld_dst] = __float2half(src[row + col * ld_src]);
    }
}

static inline void hgetrf_cublas_gemm_ex(
    cublasHandle_t handle,
    cublasOperation_t opA, cublasOperation_t opB,
    int m, int n, int k,
    float alpha_f,
    const half* A, int lda,
    const half* B, int ldb,
    float beta_f,
    half* C, int ldc,
    cublasComputeType_t computeType)
{
    if (m <= 0 || n <= 0 || k <= 0) return;

    if (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC) {
        half alpha_h = __float2half(alpha_f);
        half beta_h  = __float2half(beta_f);
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            opA, opB,
            m, n, k,
            &alpha_h,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            &beta_h,
            C, CUDA_R_16F, ldc,
            computeType,
            CUBLAS_GEMM_DEFAULT));
    } else {
        float alpha = alpha_f;
        float beta  = beta_f;
        CUBLAS_CHECK(cublasGemmEx(
            handle,
            opA, opB,
            m, n, k,
            &alpha,
            A, CUDA_R_16F, lda,
            B, CUDA_R_16F, ldb,
            &beta,
            C, CUDA_R_16F, ldc,
            computeType,
            CUBLAS_GEMM_DEFAULT));
    }
}

// ----------------------------------------------------------------------------
// Internal helpers (merged from A_exchange.cuh)
// ----------------------------------------------------------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(err__));            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// -----------------------------------------------------------------------------
// Tunables (compile-time)
// -----------------------------------------------------------------------------
// 线程块按列并行：每个 thread 负责 1 列（你原始思路）。
// 建议你扫：64 / 128 / 256
#ifndef EXCH_THREADS
#define EXCH_THREADS 256
#endif

// unroll 深度（建议扫：4 / 8 / 16）
// 过大可能增寄存器导致 occupancy 下降
#ifndef EXCH_UNROLL
#define EXCH_UNROLL 8
#endif

// ib==256 专用路径
#ifndef EXCH_USE_IB256_SPECIALIZED
#define EXCH_USE_IB256_SPECIALIZED 1
#endif

// ib==128 专用路径（当你把 hgetrf panel_width 设为 128 时很常用）
#ifndef EXCH_USE_IB128_SPECIALIZED
#define EXCH_USE_IB128_SPECIALIZED 1
#endif

// 如果你确认所有调用都满足：m >= n 且 (j0+ib)<=n，那么 row1 永远 < m
// 可把它设为 1，少一层判断（可能带来几个百分点）
// 建议先 0 保守跑通，再开 1 测性能
#ifndef EXCH_ASSUME_PANELROWS_IN_RANGE
#define EXCH_ASSUME_PANELROWS_IN_RANGE 1
#endif

// -----------------------------------------------------------------------------
// Generic kernel: works for any ib (uses dynamic shared)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_generic(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    extern __shared__ int s_ipiv[];

    for (int k = threadIdx.x; k < ib; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    // 注意：为了省掉“panel overlap”的分支，这个 kernel 假设 [col_begin, col_end)
    // 不与 [j0, j0+ib) 重叠。launcher 会在必要时自动 split。

    const size_t col_offset = (size_t)j * (size_t)lda;

    for (int k = 0; k < ib; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= ib) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Specialized kernel: ib == 256 (static shared, fixed loop bounds)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_ib256(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    __shared__ int s_ipiv[256];

    for (int k = threadIdx.x; k < 256; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    // 同 generic：假设 range 不与 panel 列重叠（由 launcher 保证）

    const size_t col_offset = (size_t)j * (size_t)lda;

    // fixed 256 steps
    for (int k = 0; k < 256; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= 256) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

#if EXCH_ASSUME_PANELROWS_IN_RANGE
            // r1 guaranteed in range
            if ((unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#else
            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#endif
        }
    }
}

// -----------------------------------------------------------------------------
// Specialized kernel: ib == 128 (static shared, fixed loop bounds)
// -----------------------------------------------------------------------------
template<int THREADS, int UNROLL>
__global__ void batch_swap_rows_kernel_range_ib128(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0,
    const int* __restrict__ d_ipiv, // 1-based
    int col_begin,
    int col_end)
{
    __shared__ int s_ipiv[128];

    for (int k = threadIdx.x; k < 128; k += THREADS) {
        s_ipiv[k] = d_ipiv[j0 + k] - 1;
    }
    __syncthreads();

    int j = col_begin + (int)blockIdx.x * THREADS + (int)threadIdx.x;
    if (j >= col_end) return;
    if ((unsigned)j >= (unsigned)n) return;

    const size_t col_offset = (size_t)j * (size_t)lda;

    for (int k = 0; k < 128; k += UNROLL) {
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            int kk = k + u;
            if (kk >= 128) break;

            int r1 = j0 + kk;
            int r2 = s_ipiv[kk];
            if (r1 == r2) continue;

#if EXCH_ASSUME_PANELROWS_IN_RANGE
            if ((unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#else
            if ((unsigned)r1 < (unsigned)m && (unsigned)r2 < (unsigned)m) {
                size_t idx1 = (size_t)r1 + col_offset;
                size_t idx2 = (size_t)r2 + col_offset;
                half tmp = A[idx1];
                A[idx1] = A[idx2];
                A[idx2] = tmp;
            }
#endif
        }
    }
}

// -----------------------------------------------------------------------------
// Public launchers (保持你现有 hgetrf 接口不变)
// -----------------------------------------------------------------------------
inline void launch_A_exchange_trailing_device_piv_range(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,
    int col_begin,
    int col_end,
    cudaStream_t stream = 0)
{
    if (!dA || !d_ipiv) return;
    if (ib <= 0) return;
    if (j0 < 0 || j0 >= n) return;

    if (col_begin < 0) col_begin = 0;
    if (col_end > n) col_end = n;
    if (col_begin >= col_end) return;

    const int panel_begin = j0;
    const int panel_end   = j0 + ib;

    // 内部 helper：假设 [cb, ce) 不与 [panel_begin, panel_end) 重叠
    auto launch_one = [&](int cb, int ce) {
        if (cb >= ce) return;

        constexpr int THREADS = EXCH_THREADS;
        constexpr int UNROLL  = EXCH_UNROLL;

        int cols = ce - cb;
        int num_blocks = (cols + THREADS - 1) / THREADS;
        if (num_blocks < 1) num_blocks = 1;

#if EXCH_USE_IB256_SPECIALIZED
        if (ib == 256) {
            batch_swap_rows_kernel_range_ib256<THREADS, UNROLL>
                <<<num_blocks, THREADS, 0, stream>>>(
                    dA, m, n, lda, j0, d_ipiv, cb, ce);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#endif

#if EXCH_USE_IB128_SPECIALIZED
        if (ib == 128) {
            batch_swap_rows_kernel_range_ib128<THREADS, UNROLL>
                <<<num_blocks, THREADS, 0, stream>>>(
                    dA, m, n, lda, j0, d_ipiv, cb, ce);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#endif

        // generic fallback
        size_t shmem = sizeof(int) * (size_t)ib;
        batch_swap_rows_kernel_range_generic<THREADS, UNROLL>
            <<<num_blocks, THREADS, shmem, stream>>>(
                dA, m, n, lda, j0, ib, d_ipiv, cb, ce);
        CUDA_CHECK(cudaGetLastError());
    };

    // 如果 range 和 panel 列有交叠：自动 split 成左右两段（避免 kernel 内部分支）
    const bool overlap = !(col_end <= panel_begin || col_begin >= panel_end);
    if (overlap) {
        int left_end = (col_end < panel_begin) ? col_end : panel_begin;
        int right_begin = (col_begin > panel_end) ? col_begin : panel_end;

        launch_one(col_begin, left_end);
        launch_one(right_begin, col_end);
        return;
    }

    // no overlap
    launch_one(col_begin, col_end);
}

inline void launch_A_exchange_trailing_device_piv(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    const int* d_ipiv,
    cudaStream_t stream = 0)
{
    // full range [0, n)
    launch_A_exchange_trailing_device_piv_range(
        dA, m, n, lda, j0, ib, d_ipiv, 0, n, stream);
}

// ----------------------------------------------------------------------------
// Internal helpers (merged from A1_panel.cuh)
// ----------------------------------------------------------------------------
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

#ifndef PANEL_USE_FAST_INV
#define PANEL_USE_FAST_INV 1
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
    // 每线程两个数据
    constexpr int ROW_TILE = 512;
    // 限定了最大的步数
    constexpr int MAX_KB   = 32;

    cg::grid_group grid = cg::this_grid();

    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid / WARP_SIZE;
    constexpr int NUM_WARPS = THREADS / WARP_SIZE;

    // block 内规约
    __shared__ float s_warp_val_f[NUM_WARPS];
    __shared__ int   s_warp_idx[NUM_WARPS];

    // pivot row cache (per-block)
    // shared -U (half2), avoid extra ops in inner loop
    __shared__ half2 sNegU2[MAX_KB]; // -U (half2) to avoid extra ops in inner loop

    // 最后找到的 pivot 行号
    __shared__ int   s_piv_row;

    const half2 minus1 = __float2half2_rn(-1.0f);
    const half  hzero  = __float2half(0.0f);

    // 逐列处理
    for (int k = k0; k < kend; ++k) {
        const int col_k = j0 + k;
        const int row_k = j0 + k;
        if (row_k >= m) 
            break;

        const size_t col_off = (size_t)col_k * (size_t)lda;

        // pivot 操作
        const int tile_start = row_k + (int)blockIdx.x * ROW_TILE;

        float local_max_f = 0.0f;
        int   local_idx   = row_k;

        // 每个线程处理两行，先试图使用向量化进行访问
        int r0 = tile_start + tid * 2;
        int r1 = r0 + 1;
        
        // 会在 float 下进行大小比较
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
        const bool singular = __heq(pivot, hzero);

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

        // Load pivot row U segment into shared as -U (half2) for fast FMA
        if (tid < num_u) {
            const int c = col_begin + tid;
            half u = A[(size_t)row_k + (size_t)c * (size_t)lda];
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
            const size_t lda_s = (size_t)lda;

            // half2 path
            half* colk_ptr = A + (size_t)rr + col_off;
            half2 a2 = aligned ? ld_half2_aligned(colk_ptr)
                               : __halves2half2(colk_ptr[0], has2 ? colk_ptr[1] : hzero);

            half2 L2 = __hmul2(a2, inv_piv_h2);

            // store L back to column col_k
            if (aligned) st_half2_aligned(colk_ptr, L2);
            else {
                colk_ptr[0] = L2.x;
                if (has2) colk_ptr[1] = L2.y;
            }

            size_t coff = (size_t)col_begin * lda_s;
            if (aligned) {
                int t = 0;
                for (; t + 1 < num_u; t += 2) {
                    half2 Av2 = ld_half2_aligned(A + (size_t)rr + coff);
                    half2 R2;
#if __CUDA_ARCH__ < 530
                    half2 U2;
#endif
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t], Av2); // Av2 - L2*U
#else
                    // fallback
                    U2 = __hmul2(sNegU2[t], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    st_half2_aligned(A + (size_t)rr + coff, R2);
                    coff += lda_s;

                    Av2 = ld_half2_aligned(A + (size_t)rr + coff);
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t + 1], Av2); // Av2 - L2*U
#else
                    U2 = __hmul2(sNegU2[t + 1], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    st_half2_aligned(A + (size_t)rr + coff, R2);
                    coff += lda_s;
                }
                if (t < num_u) {
                    half2 Av2 = ld_half2_aligned(A + (size_t)rr + coff);
                    half2 R2;
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t], Av2); // Av2 - L2*U
#else
                    // fallback
                    half2 U2 = __hmul2(sNegU2[t], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    st_half2_aligned(A + (size_t)rr + coff, R2);
                }
            } else {
                int t = 0;
                for (; t + 1 < num_u; t += 2) {
                    half v0 = A[(size_t)rr + coff];
                    half v1 = has2 ? A[(size_t)(rr + 1) + coff] : hzero;
                    half2 Av2 = __halves2half2(v0, v1);
                    half2 R2;
#if __CUDA_ARCH__ < 530
                    half2 U2;
#endif
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t], Av2); // Av2 - L2*U
#else
                    // fallback
                    U2 = __hmul2(sNegU2[t], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    A[(size_t)rr + coff] = R2.x;
                    if (has2) A[(size_t)(rr + 1) + coff] = R2.y;
                    coff += lda_s;

                    v0 = A[(size_t)rr + coff];
                    v1 = has2 ? A[(size_t)(rr + 1) + coff] : hzero;
                    Av2 = __halves2half2(v0, v1);
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t + 1], Av2); // Av2 - L2*U
#else
                    U2 = __hmul2(sNegU2[t + 1], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    A[(size_t)rr + coff] = R2.x;
                    if (has2) A[(size_t)(rr + 1) + coff] = R2.y;
                    coff += lda_s;
                }
                if (t < num_u) {
                    half v0 = A[(size_t)rr + coff];
                    half v1 = has2 ? A[(size_t)(rr + 1) + coff] : hzero;
                    half2 Av2 = __halves2half2(v0, v1);
                    half2 R2;
#if __CUDA_ARCH__ >= 530
                    R2 = __hfma2(L2, sNegU2[t], Av2); // Av2 - L2*U
#else
                    // fallback
                    half2 U2 = __hmul2(sNegU2[t], minus1);
                    R2 = __hsub2(Av2, __hmul2(L2, U2));
#endif
                    A[(size_t)rr + coff] = R2.x;
                    if (has2) A[(size_t)(rr + 1) + coff] = R2.y;
                }
            }
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
    cudaStream_t stream,
    cublasComputeType_t computeType)
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

    if (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC) {
        half alpha_h = __float2half(-1.0f);
        half beta_h  = __float2half( 1.0f);
        CUBLAS_CHECK(cublasGemmEx(
            cublas_handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha_h,
            L21, CUDA_R_16F, lda,
            U12, CUDA_R_16F, lda,
            &beta_h,
            A22, CUDA_R_16F, lda,
            computeType,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
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
            computeType,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
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
// 512 对齐，但不可以超过允许的最大 block 数
static inline int panel_TSLU_choose_blocks_fast(int m_effective, int num_blocks_max)
{
    if (m_effective <= 0) 
        return 1;
    int nb = (m_effective + 512 - 1) / 512;
    if (nb < 1) 
        nb = 1;
    if (nb > num_blocks_max) 
        nb = num_blocks_max;
    if (nb < 1) 
        nb = 1;
    return nb;
}

// panel 对外接口
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
    int   num_blocks_pivot_max,
    cublasComputeType_t computeType)
{
    if (!A || !d_ipiv || !d_block_val || !d_block_idx) {
        fprintf(stderr, "launch_panel_TSLU: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!cublas_handle) {
        fprintf(stderr, "launch_panel_TSLU: cublas_handle is null.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib <= 0) 
        return;
    if (j0 < 0 || j0 >= m) 
        return;

    // 默认为 16 ，限制在 [1, 32] 范围内
    int kb = (uc > 0) ? uc : 16;
    if (kb < 1) 
        kb = 1;
    if (kb > ib) 
        kb = ib;
    if (kb > 32) 
        kb = 32;

    const int threads = 256;

    //每一次做 kb 列
    for (int k0 = 0; k0 < ib; k0 += kb) {
        // 做到那一列结束
        int kend = k0 + kb;
        if (kend > ib) 
            kend = ib;

        const int row_base = j0 + k0;
        const int m_eff = m - row_base;

        // 每个 block 处理 512 对齐，但不允许超过 block 数最大值
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

        panel_blockout_trsm_gemm_inside_panel(
            A, m, lda, j0, ib, k0, kend,
            cublas_handle,
            stream,
            computeType);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Internal helpers (merged from A12_TRSM.cuh)
// ----------------------------------------------------------------------------
// A12_TRSM_Half2_Simplified.cuh
// 功能：求解下三角方程组 L * X = B，其中L是下三角矩阵，B是右侧矩阵块
// 核心思想：对每一列，通过前向替换求解 X，然后更新后续行
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
do {                                                                       \
    cudaError_t err__ = (call);                                            \
    if (err__ != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                cudaGetErrorString(err__));                                \
        std::exit(EXIT_FAILURE);                                           \
    }                                                                      \
} while (0)
#endif

// ============================================================================
// 异步内存拷贝指令（Ampere架构及以上）
// 优势：CPU可以继续执行其他指令，而不用等待内存拷贝完成
// ============================================================================
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr,
                                               const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    // 一次拷贝16字节（128位）
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr)
                 : "memory");
#else
    // 旧架构回退到同步拷贝
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if __CUDA_ARCH__ >= 800
    // 等待所有异步拷贝完成
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

// 使用__ldg进行只读缓存优化的全局内存读取
#define L11_LOAD(i, j, j0, A, lda) __ldg(A + (j0 + i) + (size_t)(j0 + j) * lda)

// ============================================================================
// 主Kernel：使用__half2进行SIMD优化的A12 TRSM
// ============================================================================
template<int IB>  // IB是L11矩阵的大小（编译期常量）
__global__ __launch_bounds__(256) void A12_trsm_kernel_half_optimized(
    half* __restrict__ A,   // 输入输出矩阵（列主序）
    int m, int n, int lda,  // m行，n列，leading dimension
    int j0, int ib_actual)  // j0是起始列，ib_actual是L11的实际大小
{
    // ========================================================================
    // 线程组织和参数配置
    // ========================================================================
    constexpr int warp_size = 32;
    constexpr int col_process_size = 8;     // 每个warp处理8列
    constexpr int b_stage_count = 2;        // 双缓冲：同时处理2个stage
    constexpr int rows_per_lane = IB / warp_size;  // 每个线程处理的行数

    // 关键修复：IB=128 时 (IB/8)/warp_size == 0，会导致加载/重组/写回循环完全不执行
    // 这里用 ceil 计算每个 warp 需要的迭代次数，保证至少跑 1 次
    constexpr int VEC_CHUNKS = IB / 8;  // 每 chunk=8 行 half
    constexpr int VEC_ITERS  = (VEC_CHUNKS + warp_size - 1) / warp_size;

    int ib = ib_actual;
    if (ib <= 0 || ib > IB) return;

    const int col0 = j0 + ib;               // A12的起始列
    const int ntrail = n - col0;            // A12的列数
    if (ntrail <= 0) return;

    // 线程索引
    int lane_id = threadIdx.x % warp_size;  // warp内的线程ID [0, 31]
    int warp_id = threadIdx.x / warp_size;  // warp ID [0, 7]
    int bx = blockIdx.x;

    // 每个block处理 col_process_size * b_stage_count = 16 列
    int base_col = bx * col_process_size * b_stage_count;
    int b_warp_offset = warp_id * IB;       // 每个warp的共享内存偏移

    // ========================================================================
    // 共享内存布局（使用__half2优化）
    // B_sm存储A12矩阵块：[IB行 × 16列]，分2个stage，每个stage 8列
    // ========================================================================
    __shared__ __align__(16) half2 B_sm_h2[b_stage_count][IB * (col_process_size/2)];

    // 为每个warp提供独立的视图
    half* B_sm0 = reinterpret_cast<half*>(B_sm_h2[0]) + b_warp_offset;
    half* B_sm1 = reinterpret_cast<half*>(B_sm_h2[1]) + b_warp_offset;

    int row_base = lane_id;  // 每个线程的基础行索引

    // ========================================================================
    // 步骤1：异步加载A12矩阵到共享内存
    // ========================================================================
    struct alignas(16) half8_t {
        half data[8];
    };
    __shared__ __align__(16) half8_t B_temp[b_stage_count][col_process_size][IB / 8];

    // 加载2个stage的数据
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * col_process_size;
        if (global_col >= n) continue;

        // 每个warp加载IB行数据，使用向量化加载
#pragma unroll
        for (int it = 0; it < VEC_ITERS; ++it) {
            int chunk = lane_id + it * warp_size;  // chunk in [0, VEC_CHUNKS)
            if (chunk >= VEC_CHUNKS) continue;

            int load_row = chunk * 8;
            if (load_row < ib && j0 + load_row < m) {
                const half8_t* g_ptr = reinterpret_cast<const half8_t*>(
                    A + (j0 + load_row) + (size_t)global_col * lda);
                half8_t* s_ptr = &B_temp[stage][warp_id][chunk];
                cp_async_cg_16(s_ptr, g_ptr);  // 异步拷贝16字节
            }
        }
        cp_async_commit_group();  // 提交当前批次的拷贝
    }

    // ========================================================================
    // L11列的共享内存：存储L11矩阵的列向量
    // 使用双缓冲优化：当前计算使用一个buffer，同时预取下一个block到另一个buffer
    // ========================================================================
    constexpr int a_block_rows = 4;  // 每次处理4列
    constexpr bool USE_A_DOUBLE_BUFFER = (IB <= 128);
    constexpr int a_stage_count = USE_A_DOUBLE_BUFFER ? (a_block_rows * 2) : a_block_rows;

    __shared__ __align__(16) half L11_col[a_stage_count][IB];

    int warp_row = warp_id % a_block_rows;
    int warp_half = warp_id / a_block_rows;

    // 加载L11列的lambda函数
    auto load_L11_col = [&](int col_idx, int buf_idx) {
        if (col_idx < ib && buf_idx < a_stage_count) {
            // 每个线程加载多个元素
            for (int load_idx = lane_id + warp_half * warp_size;
                 load_idx < IB;
                 load_idx += warp_size * 2) {
                if (load_idx < ib && j0 + load_idx < m && j0 + col_idx < n) {
                    half val = L11_LOAD(load_idx, col_idx, j0, A, lda);
                    L11_col[buf_idx][load_idx] = val;
                } else if (load_idx < IB) {
                    L11_col[buf_idx][load_idx] = __float2half(0.0f);
                }
            }
        }
    };

    // 预加载第一批L11列
    load_L11_col(warp_row, warp_row);
    if constexpr (USE_A_DOUBLE_BUFFER) {
        load_L11_col(a_block_rows + warp_row, a_block_rows + warp_row);
    }

    // ========================================================================
    // 等待异步加载完成，然后重组数据到共享内存
    // ========================================================================
    cp_async_wait_group_0();
    __syncthreads();

    // 将向量化的数据重组到共享内存
    for (int stage = 0; stage < b_stage_count; ++stage) {
        half* B_sm_stage = (stage == 0) ? B_sm0 : B_sm1;

#pragma unroll
        for (int it = 0; it < VEC_ITERS; ++it) {
            int chunk = lane_id + it * warp_size;
            if (chunk >= VEC_CHUNKS) continue;

            int load_row = chunk * 8;
            half8_t temp = B_temp[stage][warp_id][chunk];

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (load_row + i < ib) {
                    B_sm_stage[load_row + i] = temp.data[i];
                } else {
                    B_sm_stage[load_row + i] = __float2half(0.0f);
                }
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // 寄存器缓存：缓存最后两行数据到寄存器
    // ========================================================================
    const int b_cache_i0 = rows_per_lane - 2;
    const int b_cache_i1 = rows_per_lane - 1;
    const int r_cache0 = row_base + b_cache_i0 * warp_size;
    const int r_cache1 = row_base + b_cache_i1 * warp_size;

    // 使用half2同时存储两列
    half2 b_cache0_h2 = (r_cache0 < ib) ?
        __halves2half2(B_sm0[r_cache0], B_sm1[r_cache0]) : __float2half2_rn(0.0f);
    half2 b_cache1_h2 = (r_cache1 < ib) ?
        __halves2half2(B_sm0[r_cache1], B_sm1[r_cache1]) : __float2half2_rn(0.0f);

    // ========================================================================
    // 步骤2：主循环 - 前向替换求解
    // ========================================================================
    constexpr int a_block_count = (IB + a_block_rows - 1) / a_block_rows;

    for (int block = 0; block < ib; block += a_block_rows) {
        int block_idx = block / a_block_rows;

        // 确定使用哪个buffer（双缓冲）
        int buf_base = 0;
        if constexpr (USE_A_DOUBLE_BUFFER) {
            int set = block_idx & 1;
            buf_base = set * a_block_rows;
        }

        // 处理当前block的每一列
        for (int col_offset = 0; col_offset < a_block_rows && (block + col_offset) < ib; ++col_offset) {
            int col = block + col_offset;
            int col_lane = col & (warp_size - 1);  // col在warp中的位置
            const half* L11_col_ptr = L11_col[buf_base + col_offset];

            // 缓存L11的4个关键元素到寄存器
            const int l_cache_i0 = rows_per_lane - 4;
            const int l_cache_i1 = rows_per_lane - 3;
            const int r_l0 = row_base + l_cache_i0 * warp_size;
            const int r_l1 = row_base + l_cache_i1 * warp_size;

            half l_cache0 = (r_l0 < ib) ? L11_col_ptr[r_l0] : __float2half(0.0f);
            half l_cache1 = (r_l1 < ib) ? L11_col_ptr[r_l1] : __float2half(0.0f);
            half l_cache2 = (r_cache0 < ib) ? L11_col_ptr[r_cache0] : __float2half(0.0f);
            half l_cache3 = (r_cache1 < ib) ? L11_col_ptr[r_cache1] : __float2half(0.0f);

            // 读取当前列的解 X[col]（2列打包成half2）
            half2 x_h2 = __float2half2_rn(0.0f);
            if (lane_id == col_lane && col < IB) {
                if (col == r_cache0) {
                    x_h2 = b_cache0_h2;
                } else if (col == r_cache1) {
                    x_h2 = b_cache1_h2;
                } else {
                    x_h2 = __halves2half2(B_sm0[col], B_sm1[col]);
                }
            }

            // warp 内广播
            half x0 = __shfl_sync(0xffffffff, __low2half(x_h2),  col_lane);
            half x1 = __shfl_sync(0xffffffff, __high2half(x_h2), col_lane);
            x_h2 = __halves2half2(x0, x1);

            // 更新后续行
            int first = (col >= row_base) ? ((col - row_base) / warp_size + 1) : 0;

#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                if (r >= ib) break;

                half  l   = L11_col_ptr[r];
                half2 l_h2 = __half2half2(l);
                half2 b_h2 = __halves2half2(B_sm0[r], B_sm1[r]);

                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));

                B_sm0[r] = __low2half(b_h2);
                B_sm1[r] = __high2half(b_h2);
            }

            // 更新缓存的4行
            if (first <= l_cache_i0 && r_l0 < ib) {
                half2 l_h2 = __half2half2(l_cache0);
                half2 b_h2 = __halves2half2(B_sm0[r_l0], B_sm1[r_l0]);
                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));
                B_sm0[r_l0] = __low2half(b_h2);
                B_sm1[r_l0] = __high2half(b_h2);
            }
            if (first <= l_cache_i1 && r_l1 < ib) {
                half2 l_h2 = __half2half2(l_cache1);
                half2 b_h2 = __halves2half2(B_sm0[r_l1], B_sm1[r_l1]);
                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));
                B_sm0[r_l1] = __low2half(b_h2);
                B_sm1[r_l1] = __high2half(b_h2);
            }
            if (first <= b_cache_i0 && r_cache0 < ib) {
                half2 l_h2 = __half2half2(l_cache2);
                b_cache0_h2 = __hsub2(b_cache0_h2, __hmul2(x_h2, l_h2));
            }
            if (first <= b_cache_i1 && r_cache1 < ib) {
                half2 l_h2 = __half2half2(l_cache3);
                b_cache1_h2 = __hsub2(b_cache1_h2, __hmul2(x_h2, l_h2));
            }

            // 写回解 X[col]
            if (lane_id == col_lane && col < IB) {
                if (col == r_cache0) {
                    b_cache0_h2 = x_h2;
                } else if (col == r_cache1) {
                    b_cache1_h2 = x_h2;
                } else {
                    B_sm0[col] = __low2half(x_h2);
                    B_sm1[col] = __high2half(x_h2);
                }
            }
            __syncthreads();
        }

        // ------------------------------------------------------------------
        // 预取下一个block的L11列（双缓冲优化）
        // ------------------------------------------------------------------
        if constexpr (USE_A_DOUBLE_BUFFER) {
            int next_block = block_idx + 2;
            if (next_block < a_block_count) {
                // 关键修复：next_block=block_idx+2 与当前 block_idx 同 parity
                // 必须写回同 parity 的 buffer(set)，否则会污染 block_idx+1 要用的 other buffer
                int set = (block_idx & 1);
                int next_col = next_block * a_block_rows + warp_row;
                load_L11_col(next_col, set * a_block_rows + warp_row);
            }
            __syncthreads();
        } else {
            int next_block = block_idx + 1;
            if (next_block < a_block_count) {
                int next_col = next_block * a_block_rows + warp_row;
                load_L11_col(next_col, warp_row);
            }
            __syncthreads();
        }
    }

    // ========================================================================
    // 步骤3：写回寄存器缓存到共享内存
    // ========================================================================
    if (r_cache0 < ib) {
        B_sm0[r_cache0] = __low2half(b_cache0_h2);
        B_sm1[r_cache0] = __high2half(b_cache0_h2);
    }
    if (r_cache1 < ib) {
        B_sm0[r_cache1] = __low2half(b_cache1_h2);
        B_sm1[r_cache1] = __high2half(b_cache1_h2);
    }
    __syncthreads();

    // ========================================================================
    // 步骤4：向量化写回全局内存
    // ========================================================================
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * col_process_size;
        if (global_col >= n) continue;

        half* B_sm_stage = (stage == 0) ? B_sm0 : B_sm1;
        half* B_out = A + j0 + (size_t)global_col * lda;

#pragma unroll
        for (int it = 0; it < VEC_ITERS; ++it) {
            int chunk = lane_id + it * warp_size;
            if (chunk >= VEC_CHUNKS) continue;

            int store_row = chunk * 8;
            if (store_row < ib && j0 + store_row + 7 < m) {
                half8_t temp;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    temp.data[i] = B_sm_stage[store_row + i];
                }
                half8_t* g_ptr = reinterpret_cast<half8_t*>(B_out + store_row);
                *g_ptr = temp;
            } else {
                for (int i = 0; i < 8; ++i) {
                    if (store_row + i < ib && j0 + store_row + i < m) {
                        B_out[store_row + i] = B_sm_stage[store_row + i];
                    }
                }
            }
        }
    }
}

// ============================================================================
// 启动函数：根据ib大小选择合适的模板实例
// ============================================================================
inline void launch_A12_trsm(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    cudaStream_t stream = 0)
{
    const int col0   = j0 + ib;
    const int ntrail = n - col0;
    if (ntrail <= 0 || ib <= 0) return;

    if (ib > 256) {
        fprintf(stderr, "[A12_TRSM] ERROR: ib=%d > 256 is not supported.\n", ib);
        std::exit(EXIT_FAILURE);
    }

    // 计算grid大小：每个block处理16列
    constexpr int col_process_size = 8;
    constexpr int b_stage_count = 2;
    int cols_per_block = col_process_size * b_stage_count;
    int grid_x = (ntrail + cols_per_block - 1) / cols_per_block;
    if (grid_x <= 0) grid_x = 1;

    dim3 grid(grid_x);
    dim3 block(256);  // 256线程 = 8 warps

    if (ib <= 32) {
        A12_trsm_kernel_half_optimized<32><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else if (ib <= 64) {
        A12_trsm_kernel_half_optimized<64><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else if (ib <= 128) {
        A12_trsm_kernel_half_optimized<128><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else {
        A12_trsm_kernel_half_optimized<256><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    }

    CUDA_CHECK(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Internal helpers (merged from A22_GEMM.cuh)
// ----------------------------------------------------------------------------
// A22_GEMM.cuh
#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#ifndef CUBLAS_CHECK
#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "cuBLAS error %s:%d: status=%d\n",                 \
                    __FILE__, __LINE__, (int)st__);                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

// 如果你真的想让 GEMM 内部 setStream（不建议），可以打开这个宏
// #define A22_GEMM_SET_STREAM_INSIDE 1
#if defined(A22_GEMM_SET_STREAM_INSIDE)
#error "A22_GEMM_SET_STREAM_INSIDE is not supported: set the stream on the cublas handle outside for accurate timing."
#endif

/**
 * Tensor Core GEMM（带列范围）:
 *
 *  A22(col0:col0+n2-1) -= A21 * A12(col0:col0+n2-1)
 */
inline void launch_A22_gemm_tc_range(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    int   col0, int n2,
    cublasHandle_t handle,
    cudaStream_t stream,
    cublasComputeType_t computeType
#if HGETRF_USE_CUBLASLT
    , cublasLtHandle_t lt_handle = nullptr,
    void* lt_workspace = nullptr,
    size_t lt_workspace_bytes = 0
#endif
    )
{
    if (!dA || !handle) return;
    if (ib <= 0 || n2 <= 0) return;

    const int row0 = j0 + ib;
    const int m2   = m - row0;
    if (m2 <= 0) return;
    if (col0 >= n) return;
    if (col0 + n2 > n) n2 = n - col0;

    half* A21 = dA + row0 + (size_t)j0   * lda;  // (m2 x ib)
    half* A12 = dA + j0   + (size_t)col0 * lda;  // (ib x n2)
    half* A22 = dA + row0 + (size_t)col0 * lda;  // (m2 x n2)

#if defined(A22_GEMM_SET_STREAM_INSIDE)
    CUBLAS_CHECK(cublasSetStream(handle, stream));
#else
    // 期望外层已经把 handle 的 stream 设置为 stream
    (void)stream;
#endif

#if HGETRF_USE_CUBLASLT
    if (lt_handle) {
        cublasLtMatmulDesc_t op_desc = nullptr;
        cublasLtMatrixLayout_t a_desc = nullptr;
        cublasLtMatrixLayout_t b_desc = nullptr;
        cublasLtMatrixLayout_t c_desc = nullptr;
        cublasLtMatrixLayout_t d_desc = nullptr;
        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned = 0;
        cublasOperation_t opn = CUBLAS_OP_N;
        cublasLtOrder_t order = CUBLASLT_ORDER_COL;
        cudaDataType_t scale_type =
            (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC)
                ? CUDA_R_16F
                : CUDA_R_32F;
        size_t ws_bytes = lt_workspace ? lt_workspace_bytes : 0;

        cublasStatus_t st = cublasLtMatmulDescCreate(&op_desc, computeType, scale_type);
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opn, sizeof(opn));
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opn, sizeof(opn));
        }

        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, m2, ib, lda);
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, ib, n2, lda);
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, m2, n2, lda);
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_16F, m2, n2, lda);
        }

        if (st == CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
            cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
            cublasLtMatrixLayoutSetAttribute(c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
            cublasLtMatrixLayoutSetAttribute(d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
        }

        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulPreferenceCreate(&pref);
        }
        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulPreferenceSetAttribute(
                pref,
                CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &ws_bytes,
                sizeof(ws_bytes));
        }

        if (st == CUBLAS_STATUS_SUCCESS) {
            st = cublasLtMatmulAlgoGetHeuristic(
                lt_handle, op_desc, a_desc, b_desc, c_desc, d_desc,
                pref, 1, &heuristic, &returned);
        }
        if (st == CUBLAS_STATUS_SUCCESS && returned > 0) {
            if (scale_type == CUDA_R_16F) {
                half alpha_h = __float2half(-1.0f);
                half beta_h  = __float2half( 1.0f);
                st = cublasLtMatmul(
                    lt_handle, op_desc,
                    &alpha_h,
                    A21, a_desc,
                    A12, b_desc,
                    &beta_h,
                    A22, c_desc,
                    A22, d_desc,
                    &heuristic.algo,
                    lt_workspace, ws_bytes, stream);
            } else {
                const float alpha = -1.0f;
                const float beta  =  1.0f;
                st = cublasLtMatmul(
                    lt_handle, op_desc,
                    &alpha,
                    A21, a_desc,
                    A12, b_desc,
                    &beta,
                    A22, c_desc,
                    A22, d_desc,
                    &heuristic.algo,
                    lt_workspace, ws_bytes, stream);
            }
        }

        if (pref)  cublasLtMatmulPreferenceDestroy(pref);
        if (d_desc) cublasLtMatrixLayoutDestroy(d_desc);
        if (c_desc) cublasLtMatrixLayoutDestroy(c_desc);
        if (b_desc) cublasLtMatrixLayoutDestroy(b_desc);
        if (a_desc) cublasLtMatrixLayoutDestroy(a_desc);
        if (op_desc) cublasLtMatmulDescDestroy(op_desc);

        if (st == CUBLAS_STATUS_SUCCESS && returned > 0) return;
    }
#endif

    if (computeType == CUBLAS_COMPUTE_16F || computeType == CUBLAS_COMPUTE_16F_PEDANTIC) {
        half alpha_h = __float2half(-1.0f);
        half beta_h  = __float2half( 1.0f);
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m2,         // m
                n2,         // n
                ib,         // k
                &alpha_h,
                A21, CUDA_R_16F, lda,
                A12, CUDA_R_16F, lda,
                &beta_h,
                A22, CUDA_R_16F, lda,
                computeType,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        const float alpha = -1.0f;
        const float beta  =  1.0f;
        CUBLAS_CHECK(
            cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m2,         // m
                n2,         // n
                ib,         // k
                &alpha,
                A21, CUDA_R_16F, lda,
                A12, CUDA_R_16F, lda,
                &beta,
                A22, CUDA_R_16F, lda,
                computeType,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
}


/**
 * naive GEMM（一次性完整 tail）
 */
#if defined(HGETRF_ENABLE_GEMM_FALLBACK)
__global__ void A22_gemm_naive_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib,
    int col0, int n2)
{
    int row0 = j0 + ib;
    int m2   = m - row0;
    if (m2 <= 0 || n2 <= 0) return;

    int j_rel = blockIdx.x * blockDim.x + threadIdx.x;
    int i_rel = blockIdx.y * blockDim.y + threadIdx.y;

    if (j_rel >= n2 || i_rel >= m2) return;

    int i = row0 + i_rel;
    int j = col0 + j_rel;

    float sum = 0.0f;
    for (int k = 0; k < ib; ++k) {
        int kcol = j0 + k;

        half a_h = A[i + (size_t)kcol * lda];
        half b_h = A[(j0 + k) + (size_t)j * lda];

        float a = __half2float(a_h);
        float b = __half2float(b_h);

        sum += a * b;
    }

    half c_h = A[i + (size_t)j * lda];
    float c  = __half2float(c_h);

    float res = c - sum;
    A[i + (size_t)j * lda] = __float2half(res);
}

inline void launch_A22_gemm_naive_range(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    int   col0, int n2,
    cudaStream_t stream)
{
    if (!dA) return;
    int row0 = j0 + ib;
    int m2   = m - row0;
    if (m2 <= 0 || n2 <= 0) return;

    if (col0 >= n) return;
    if (col0 + n2 > n) n2 = n - col0;

    dim3 block(16, 16);
    dim3 grid((n2 + block.x - 1) / block.x,
              (m2 + block.y - 1) / block.y);

    A22_gemm_naive_kernel<<<grid, block, 0, stream>>>(
        dA, m, n, lda, j0, ib, col0, n2
    );
    CUDA_CHECK(cudaGetLastError());
}
#endif

/**
 * 兼容老接口：一次性更新整个 tail
 */
#if defined(HGETRF_ENABLE_LEGACY_A22_GEMM_API)
inline void launch_A22_gemm_tc(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    cublasHandle_t handle,
    cudaStream_t stream)
{
    int col0 = j0 + ib;
    int n2   = n - col0;
    launch_A22_gemm_tc_range(dA, m, n, lda, j0, ib, col0, n2, handle, stream, CUBLAS_COMPUTE_32F
#if HGETRF_USE_CUBLASLT
        , nullptr, nullptr, 0
#endif
    );
}

#if defined(HGETRF_ENABLE_GEMM_FALLBACK)
inline void launch_A22_gemm_naive(
    half* dA,
    int   m, int n, int lda,
    int   j0, int ib,
    cudaStream_t stream)
{
    int col0 = j0 + ib;
    int n2   = n - col0;
    launch_A22_gemm_naive_range(dA, m, n, lda, j0, ib, col0, n2, stream);
}
#endif
#endif

// ----------------------------------------------------------------------------
// Kernel: Zero Check
// ----------------------------------------------------------------------------
__global__ void hgetrf_check_panel_pivots_zero_kernel(
    const half* __restrict__ A, int lda, int j0, int ib,
    const int* __restrict__ d_ipiv, int* __restrict__ d_info)
{
    int tid = threadIdx.x;
    if (*d_info != 0) return;
    for (int kk = tid; kk < ib; kk += blockDim.x) {
        int step = j0 + kk;
        half pv = A[(d_ipiv[step]-1) + (size_t)step * lda];
        if (pv == __float2half(0.0f)) atomicCAS(d_info, 0, step + 1);
    }
}

// 对外提供的接口
inline void hgetrf(
    hgetrfHandle_t h,
    int m, int n,
    half* dA, int lda,
    half* d_workspace,
    int* d_ipiv,
    int* d_info,
    bool profile = false)
{

    if (!h || !dA || !d_workspace || !d_ipiv || !d_info) {
        fprintf(stderr, "hgetrf: null input.\n"); std::exit(EXIT_FAILURE);
    }
    if (h->m_cached_max > 0 && m > h->m_cached_max) {
        fprintf(stderr, "hgetrf: m > cached_max. Re-call bufferSize.\n"); std::exit(EXIT_FAILURE);
    }

    // panel_width 对应了每次处理块的大小
    // uc 对应了 panel 内部的分块大小
    int panel_width = h->panel_width;
    int uc = h->uc;
    // 总共要进行 LU 分解的方阵维度
    const int k_total = (m < n) ? m : n;
    if (k_total <= 0) 
        return;

    if (panel_width <= 0) 
        panel_width = 256;
    if (panel_width > k_total) 
        panel_width = k_total;

    // 流
    cudaStream_t s_main = h->stream_main;
    cudaStream_t s_panel = h->stream_panel;
    // cublas 句柄
    cublasHandle_t cb_up = h->cublas_update;
    cublasHandle_t cb_pa = h->cublas_panel;
    HgetrfWorkspaceView ws = hgetrf_workspace_bind(h, d_workspace);

    // 标志位检查，在 panel 执行之前
    CUDA_CHECK(cudaMemsetAsync(d_info, 0, sizeof(int), s_panel));

    // 用于 testLU 时的 profile 优化程序
    cudaEvent_t ev_start = nullptr, ev_stop = nullptr;
    if (profile) {
        cudaEventCreate(&ev_start); 
        cudaEventCreate(&ev_stop);
        cudaEventRecord(ev_start, s_main);
    }

    // Lambda 函数定义各个阶段

    // Panel
    auto do_panel = [&](int j0, int ib_now) {
        // uc 为内部计算的小块 ， ib 为每次 panel 计算的维度大小
        int uc_now = (uc > ib_now) ? ib_now : uc;
        launch_panel_TSLU(dA, m, lda, j0, ib_now, uc_now, d_ipiv,
            cb_pa, s_panel, ws.d_panel_block_val, ws.d_panel_block_idx, ws.num_blocks_pivot_max, h->gemm_compute);
        
        int threads = (ib_now >= 256) ? 256 : ib_now;
        hgetrf_check_panel_pivots_zero_kernel<<<1, threads, 0, s_panel>>>(
            dA, lda, j0, ib_now, d_ipiv, d_info);
        
        CUDA_CHECK(cudaEventRecord(h->ev_piv_ready, s_panel));
    };

    // Exchange
    auto do_exchange = [&](int j0, int ib_now) {
        CUDA_CHECK(cudaStreamWaitEvent(s_main, h->ev_piv_ready, 0));
        
#if HGETRF_EXCH_SPLIT_LEFT
        int right_begin = j0 + ib_now;
        if (right_begin < n) {
            launch_A_exchange_trailing_device_piv_range(
                dA, m, n, lda, j0, ib_now, d_ipiv, right_begin, n, s_main);
        }
        
        // 左侧放到后台流
        if (j0 > 0) {
            cudaStream_t s_exch = h->stream_exch_left;
            CUDA_CHECK(cudaStreamWaitEvent(s_exch, h->ev_piv_ready, 0));
            launch_A_exchange_trailing_device_piv_range(
                dA, m, n, lda, j0, ib_now, d_ipiv, 0, j0, s_exch);
            CUDA_CHECK(cudaEventRecord(h->ev_exch_left_done, s_exch));
        } else {
            // 防止第一次 wait 失败
            CUDA_CHECK(cudaEventRecord(h->ev_exch_left_done, h->stream_exch_left));
        }
#else
        launch_A_exchange_trailing_device_piv(dA, m, n, lda, j0, ib_now, d_ipiv, s_main);
#endif
    };

    // TRSM
    auto do_trsm = [&](int j0, int ib_now) {
        int col0 = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0) return;

        if (h->trsm_mode == 1 && ws.d_L_f && ws.d_inv_f && ws.d_inv_h && ws.d_trsm) {
            const int ib = ib_now;
            const int tile = (h->trsm_tile > 0) ? h->trsm_tile : 4096;
            const int tile_use = (tile > ntrail) ? ntrail : tile;

            // L11 (lower, unit diag) -> float
            int total = ib * ib;
            int block = 256;
            int grid = std::min(1024, (total + block - 1) / block);
            f16_to_f32_mat<<<grid, block, 0, s_main>>>(dA + (size_t)j0 + (size_t)j0 * (size_t)lda,
                                                      lda, ws.d_L_f, ib, ib);
            CUDA_CHECK(cudaGetLastError());

            // inv_f = I
            set_identity_f<<<grid, block, 0, s_main>>>(ws.d_inv_f, ib, ib);
            CUDA_CHECK(cudaGetLastError());

            // inv_f = inv(L11) (diag = unit)
            const float one = 1.0f;
            CUBLAS_CHECK(cublasStrsm(
                cb_up,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                ib, ib,
                &one,
                ws.d_L_f, ib,
                ws.d_inv_f, ib));

            // inv_h = (half)inv_f
            f32_to_f16_mat<<<grid, block, 0, s_main>>>(ws.d_inv_f, ib, ws.d_inv_h, ib, ib);
            CUDA_CHECK(cudaGetLastError());

            // A12 = inv(L11) * A12 (tile by columns to avoid aliasing)
            for (int co = 0; co < ntrail; co += tile_use) {
                int mcol = std::min(tile_use, ntrail - co);
                half* B = dA + (size_t)j0 + (size_t)(col0 + co) * (size_t)lda; // (ib x mcol)

                CUDA_CHECK(cudaMemcpy2DAsync(
                    ws.d_trsm, (size_t)ib * sizeof(half),
                    B,         (size_t)lda * sizeof(half),
                    (size_t)ib * sizeof(half),
                    (size_t)mcol,
                    cudaMemcpyDeviceToDevice, s_main));

                hgetrf_cublas_gemm_ex(
                    cb_up,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    ib, mcol, ib,
                    1.0f,
                    ws.d_inv_h, ib,
                    ws.d_trsm,  ib,
                    0.0f,
                    B,          lda,
                    h->gemm_compute);
            }
            return;
        }

        launch_A12_trsm(dA, m, n, lda, j0, ib_now, s_main);
    };

    // GEMM
    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 > 0) 
            launch_A22_gemm_tc_range(dA, m, n, lda, j0, ib_now, col0, n2, cb_up, s_main, h->gemm_compute
#if HGETRF_USE_CUBLASLT
                , h->cublaslt, ws.d_lt_workspace, ws.lt_workspace_bytes
#endif
            );
    };

    // 流水线启动
    int j0 = 0;
    // 处理完之后剩余的块大小，在这里的 ktotal 是刚刚进入的 min(m,n)
    int rem0 = k_total - j0;
    // 如果小于 panel_width 则用实际大小，否则的话 panel_width 就是单次可处理的最大宽度
    int ib_now = (rem0 >= panel_width) ? panel_width : rem0;

    // 先做一次 panel 才轮到下一个，分别是开始的位置，以及本次要做多宽的 panel
    do_panel(j0, ib_now);

    while (j0 + ib_now < k_total) {
        int j0_next = j0 + ib_now;
        int rem_next = k_total - j0_next;
        int ib_next = (rem_next >= panel_width) ? panel_width : rem_next;

        do_exchange(j0, ib_now);
        do_trsm(j0, ib_now);

        // Look-ahead: 更新下一个 Panel 的区域
        do_gemm_range(j0, ib_now, j0 + ib_now, ib_next);
        CUDA_CHECK(cudaEventRecord(h->ev_next_ready, s_main));

        // 触发下一个 Panel (在 panel 流)
        CUDA_CHECK(cudaStreamWaitEvent(s_panel, h->ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        // 更新剩余尾部
        int col_tail = j0 + ib_now + ib_next;
        do_gemm_range(j0, ib_now, col_tail, n - col_tail);

        j0 = j0_next;
        ib_now = ib_next;
    }

    // Last block
    do_exchange(j0, ib_now);
    do_trsm(j0, ib_now);
    do_gemm_range(j0, ib_now, j0 + ib_now, n - (j0 + ib_now));

    // Wait for final panel (if any logic remains there)
    CUDA_CHECK(cudaEventRecord(h->ev_next_ready, s_panel));
    CUDA_CHECK(cudaStreamWaitEvent(s_main, h->ev_next_ready, 0));
    
#if HGETRF_EXCH_SPLIT_LEFT
    CUDA_CHECK(cudaStreamWaitEvent(s_main, h->ev_exch_left_done, 0));
#endif

    if (profile) {
        cudaEventRecord(ev_stop, s_main);
        cudaEventSynchronize(ev_stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, ev_start, ev_stop);
        printf("[hgetrf] Total time: %.3f ms (approx)\n", ms);
        cudaEventDestroy(ev_start); cudaEventDestroy(ev_stop);
    }
}
