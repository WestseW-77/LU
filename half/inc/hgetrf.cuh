#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <climits>

#include "A1_panel.cuh"
#include "A_exchange.cuh"
#include "A12_TRSM.cuh"
#include "A22_GEMM.cuh"

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

// ============================================================================
// 句柄：仿 cusolverDnHandle，内部只创建一次 stream/event
// ============================================================================
struct hgetrfHandle {
    // 为了支持 dual-stream pipeline：
    // - update stream 使用一个 cublas handle
    // - panel stream 也需要独立的 cublas handle（避免 setStream 互相抢）
    cublasHandle_t cublas_update = nullptr;
    cublasHandle_t cublas_panel  = nullptr;
    bool owns_cublas_update = false;
    bool owns_cublas_panel  = false;
    // 命令队列
    cudaStream_t   stream = 0;
    cudaStream_t   stream_panel = nullptr;
    // Event 用于跨 stream 同步
    cudaEvent_t  ev_piv_ready  = nullptr;
    cudaEvent_t  ev_next_ready = nullptr;

    int panel_width = 128;
    int uc = 8;

    int    m_cached_max = 0;
    int    num_blocks_pivot_max = 0;
    size_t workspace_bytes = 0;
};

using hgetrfHandle_t = hgetrfHandle*;

inline void hgetrfCreate(hgetrfHandle_t* out)
{
    // 要有目标指针才可以初始化
    if (!out) 
        return;
    hgetrfHandle_t h = new hgetrfHandle;
    // 不会因默认的 stream 同步规则而被阻塞
    CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream_panel, cudaStreamNonBlocking));
    // event 仅用于同步而不记录时间戳
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_piv_ready,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_next_ready, cudaEventDisableTiming));

    // cuBLAS 句柄也在这里创建（避免内部函数偷偷 create/destroy）
    CUBLAS_CHECK(cublasCreate(&h->cublas_update));
    h->owns_cublas_update = true;
    CUBLAS_CHECK(cublasCreate(&h->cublas_panel));
    h->owns_cublas_panel = true;

    // 让默认行为更接近“高性能”配置（half GEMM 更容易走 Tensor Core 路径）
    CUBLAS_CHECK(cublasSetMathMode(h->cublas_update, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetMathMode(h->cublas_panel,  CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSetStream(h->cublas_update, h->stream));
    CUBLAS_CHECK(cublasSetStream(h->cublas_panel,  h->stream_panel));

    *out = h;
}

inline void hgetrfDestroy(hgetrfHandle_t h)
{
    if (!h) 
        return;

    if (h->cublas_update && h->owns_cublas_update) {
        CUBLAS_CHECK(cublasDestroy(h->cublas_update));
        h->cublas_update = nullptr;
    }
    if (h->cublas_panel && h->owns_cublas_panel) {
        CUBLAS_CHECK(cublasDestroy(h->cublas_panel));
        h->cublas_panel = nullptr;
    }

    if (h->ev_piv_ready)
        CUDA_CHECK(cudaEventDestroy(h->ev_piv_ready));
    if (h->ev_next_ready)
        CUDA_CHECK(cudaEventDestroy(h->ev_next_ready));
    if (h->stream_panel)
        CUDA_CHECK(cudaStreamDestroy(h->stream_panel));
    delete h;
}

inline void hgetrfSetStream(hgetrfHandle_t h, cudaStream_t s)
{
    if (!h) return;
    h->stream = s;
    if (h->cublas_update) {
        CUBLAS_CHECK(cublasSetStream(h->cublas_update, h->stream));
    }
}

inline void hgetrfSetPanelWidth(hgetrfHandle_t h, int panel_width)
{
    if (!h) return;
    h->panel_width = panel_width;
}

inline void hgetrfSetUc(hgetrfHandle_t h, int uc)
{
    if (!h) return;
    h->uc = uc;
}

// 分配一个大块的 workspace 后，在这里又把这份大的 workspace 分割成不同部分给不同地方调用
struct HgetrfWorkspaceView {
    half* d_panel_block_val = nullptr;
    int*  d_panel_block_idx = nullptr;
    int   num_blocks_pivot_max = 0;
};

static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// 计算需要多大的 buffer
inline void hgetrf_bufferSizeBytes(
    hgetrfHandle_t /*h*/,
    int m, int /*n*/, int /*lda*/,
    int /*panel_width*/,
    size_t* device_bytes)
{
    if (!device_bytes) 
        return;

    const int num_blocks = panel_TSLU_required_pivot_blocks(m, 0);

    size_t bytes = 0;
    bytes = align_up(bytes, 256);
    bytes += sizeof(half) * (size_t)num_blocks;   // d_panel_block_val
    bytes = align_up(bytes, 256);
    bytes += sizeof(int)  * (size_t)num_blocks;   // d_panel_block_idx

    *device_bytes = bytes;
}

// cuSOLVER-like: lwork is in half elements (cudaMalloc sizeof(half)*lwork)
inline void hgetrf_bufferSize(
    hgetrfHandle_t h,
    int m, int n,
    const half* dA, int lda,
    int* lwork)
{
    (void)dA;
    if (!h) {
        fprintf(stderr, "hgetrf_bufferSize: handle is null.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!lwork) return;

    size_t bytes = 0;
    hgetrf_bufferSizeBytes(h, m, n, lda, h->panel_width, &bytes);

    size_t elems = (bytes + sizeof(half) - 1) / sizeof(half);
    if (elems > (size_t)INT32_MAX) {
        fprintf(stderr, "hgetrf_bufferSize: lwork overflow (%zu half elements)\n", elems);
        std::exit(EXIT_FAILURE);
    }
    *lwork = (int)elems;

    h->m_cached_max = m;
    h->num_blocks_pivot_max = panel_TSLU_required_pivot_blocks(m, 0);
    h->workspace_bytes = bytes;
}

// 分配的 workspace 切成一段段给不同地方使用
inline HgetrfWorkspaceView hgetrf_workspace_bind(
    void* d_workspace,
    size_t workspace_bytes,
    int num_blocks_pivot_max)
{
    HgetrfWorkspaceView ws;

    ws.num_blocks_pivot_max = num_blocks_pivot_max;

    uint8_t* p = (uint8_t*)d_workspace;
    size_t off = 0;

    off = align_up(off, 256);
    ws.d_panel_block_val = (half*)(p + off);
    off += sizeof(half) * (size_t)num_blocks_pivot_max;

    off = align_up(off, 256);
    ws.d_panel_block_idx = (int*)(p + off);
    off += sizeof(int) * (size_t)num_blocks_pivot_max;

    if (off > workspace_bytes) {
        fprintf(stderr, "hgetrf_workspace_bind: workspace_bytes too small (need %zu, got %zu)\n",
                off, workspace_bytes);
        std::exit(EXIT_FAILURE);
    }
    return ws;
}

// 看起来这个是一整个 panel 最后启动一次检查是否有出现 0，用 128 个线程就可以检查因为我们一个 panel 固定宽度为 128
__global__ void hgetrf_check_panel_pivots_zero_kernel(
    const half* __restrict__ A,
    int lda,
    int j0,
    int ib,
    const int* __restrict__ d_ipiv,  // 1-based
    int* __restrict__ d_info)        // device scalar
{
    // 这里用 1 个 block 够了：ib<=128，一般 128 线程一轮搞定
    int tid = threadIdx.x;

    // 读取当前 info，如果已经非 0，直接退出（不浪费）
    int cur = *d_info;
    if (cur != 0) 
        return;

    // 每个线程处理多步
    for (int kk = tid; kk < ib; kk += blockDim.x) {
        int step = j0 + kk;
        int piv  = d_ipiv[step] - 1;  // to 0-based
        half pv  = A[piv + (size_t)step * lda];

        if (pv == __float2half(0.0f)) {
            // 写最早的 step+1，避免竞态
            atomicCAS(d_info, 0, step + 1);
        }
    }
}

// ============================================================================
// 核心：blocked LU with dual stream
// - d_ipiv: global pivot (1-based), length >= min(m,n)
// - d_info: device scalar int (0 init)
// ============================================================================
inline void hgetrf_blocked_half_dualstream_ws(
    half* dA,
    int m, int n, int lda,
    int panel_width, int uc,
    cublasHandle_t cublas_update,
    cublasHandle_t cublas_panel,
    int* d_ipiv,        // global ipiv (1-based)
    int* d_info,        // device scalar (0 init)
    const HgetrfWorkspaceView& ws,
    cudaStream_t stream_update,
    cudaStream_t stream_panel,
    cudaEvent_t  ev_piv_ready,
    cudaEvent_t  ev_next_ready)
{
    if (!dA || !d_ipiv || !d_info) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!ws.d_panel_block_val || !ws.d_panel_block_idx) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: invalid workspace.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!cublas_update || !cublas_panel) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: cublas handles must be set.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!stream_panel || !ev_piv_ready || !ev_next_ready) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: stream/event not set.\n");
        std::exit(EXIT_FAILURE);
    }

    const int k_total = (m < n) ? m : n;
    if (k_total <= 0) 
        return;

    // 默认宽度 128，如果总体比 128 小则用小的
    if (panel_width <= 0) 
        panel_width = 128;
    if (panel_width > k_total) 
        panel_width = k_total;

    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;

        launch_panel_TSLU(
            dA, m, lda,
            j0, ib_now, uc_now,
            d_ipiv,               // global ipiv (1-based)
            cublas_panel,
            stream_panel,
            ws.d_panel_block_val,
            ws.d_panel_block_idx,
            ws.num_blocks_pivot_max);

        // ✅ 正确+快：每个 panel 只做 1 次 pivot==0 检测
        // 线程数设置到 128（与你 panel_width 一致）
        int threads = 128;
        hgetrf_check_panel_pivots_zero_kernel<<<1, threads, 0, stream_panel>>>(
            dA, lda, j0, ib_now, d_ipiv, d_info);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_piv_ready, stream_panel));
    };

    auto do_exchange = [&](int j0, int ib_now) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_piv_ready, 0));
        launch_A_exchange_trailing_device_piv(
            dA, m, n, lda, j0, ib_now,
            d_ipiv,             // global ipiv (1-based)
            stream_update);
    };

    auto do_trsm = [&](int j0, int ib_now) {
        int col0   = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0 || ib_now <= 0) return;

        launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;

        // ✅ 快：不在这里 setStream（外层 hgetrf 已经设过）
        launch_A22_gemm_tc_range(
            dA, m, n, lda,
            j0, ib_now,
            col0, n2,
            cublas_update, stream_update);
    };

    auto do_gemm_full = [&](int j0, int ib_now) {
        int col0 = j0 + ib_now;
        int n2   = n - col0;
        do_gemm_range(j0, ib_now, col0, n2);
    };

    // pipeline start
    int j0 = 0;
    int rem0 = k_total - j0;
    int ib_now = (rem0 >= panel_width) ? panel_width : rem0;
    if (ib_now <= 0) return;

    do_panel(j0, ib_now);

    while (j0 + ib_now < k_total) {
        int j0_next = j0 + ib_now;
        int rem_next = k_total - j0_next;
        int ib_next = (rem_next >= panel_width) ? panel_width : rem_next;

        do_exchange(j0, ib_now);
        do_trsm(j0, ib_now);

        // look-ahead: update only ib_next columns for next panel
        {
            int col0_next = j0 + ib_now;
            int n2_next   = ib_next;
            do_gemm_range(j0, ib_now, col0_next, n2_next);
            CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_update));
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream_panel, ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        // remaining tail
        {
            int col0_tail = j0 + ib_now + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) do_gemm_range(j0, ib_now, col0_tail, n2_tail);
        }

        j0 = j0_next;
        ib_now = ib_next;
    }

    do_exchange(j0, ib_now);
    do_trsm(j0, ib_now);
    do_gemm_full(j0, ib_now);

    // follow cuSolver style (caller controls sync/timing), but make the result
    // visible on stream_update by bridging stream_panel completion back.
    CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_panel));
    CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_next_ready, 0));
}

// ============================================================================
// Public API: hgetrf (cuSOLVER-like)
// - d_ipiv: device, length min(m,n), output (1-based pivot row)
// - d_info: device scalar int, output (0=success, >0 singular position)
// - workspace 外置
// ============================================================================
inline void hgetrf(
    hgetrfHandle_t h,
    int m, int n,
    half* dA, int lda,
    half* d_workspace,
    int* d_ipiv,          // device output: global ipiv (1-based), length min(m,n)
    int* d_info)          // device output: scalar int
{
    if (!h || !dA || !d_workspace || !d_ipiv || !d_info) {
        fprintf(stderr, "hgetrf: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!h->cublas_update || !h->cublas_panel) {
        fprintf(stderr, "hgetrf: cublas handles must be set.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!h->stream_panel || !h->ev_piv_ready || !h->ev_next_ready) {
        fprintf(stderr, "hgetrf: stream/event not set.\n");
        std::exit(EXIT_FAILURE);
    }
    if (h->workspace_bytes == 0 || h->num_blocks_pivot_max <= 0) {
        fprintf(stderr, "hgetrf: workspace not set, call hgetrf_bufferSize + allocate workspace first.\n");
        std::exit(EXIT_FAILURE);
    }
    if (h->m_cached_max > 0 && m > h->m_cached_max) {
        fprintf(stderr, "hgetrf: m=%d exceeds cached max m=%d, call hgetrf_bufferSize again.\n",
                m, h->m_cached_max);
        std::exit(EXIT_FAILURE);
    }

    const int k_total = (m < n) ? m : n;
    if (k_total <= 0) return;

    int panel_width = h->panel_width;
    int uc = h->uc;
    printf(" uc = %d\n", uc);
    if (panel_width <= 0) panel_width = 128;
    if (panel_width > k_total) panel_width = k_total;

    // bind workspace
    HgetrfWorkspaceView ws = hgetrf_workspace_bind(d_workspace, h->workspace_bytes, h->num_blocks_pivot_max);

    // run
    hgetrf_blocked_half_dualstream_ws(
        dA, m, n, lda,
        panel_width, uc,
        h->cublas_update,
        h->cublas_panel,
        d_ipiv,
        d_info,
        ws,
        h->stream,
        h->stream_panel,
        h->ev_piv_ready, 
        h->ev_next_ready);
}
