#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

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
    cublasHandle_t cublas = nullptr;
    // 命令队列
    cudaStream_t   stream = 0;
    cudaStream_t   stream_panel = nullptr;
    // Event 用于跨 stream 同步
    cudaEvent_t  ev_piv_ready  = nullptr;
    cudaEvent_t  ev_next_ready = nullptr;
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

    *out = h;
}

inline void hgetrfDestroy(hgetrfHandle_t h)
{
    if (!h) 
        return;
    if (h->ev_piv_ready)  
        CUDA_CHECK(cudaEventDestroy(h->ev_piv_ready));
    if (h->ev_next_ready) 
        CUDA_CHECK(cudaStreamDestroy(h->stream_panel));
    delete h;
}

inline void hgetrfSetStream(hgetrfHandle_t h, cudaStream_t s)
{
    if (!h) 
        return;
    h->stream = s;
}

inline void hgetrfSetCublas(hgetrfHandle_t h, cublasHandle_t c)
{
    if (!h) return;
    h->cublas = c;
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

// 计算需要多大的 buffer [fix]
inline void hgetrf_bufferSize(
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
// 分配的 workspace 切成一段段给不同地方使用
inline HgetrfWorkspaceView hgetrf_workspace_bind(
    void* d_workspace,
    size_t workspace_bytes,
    int m)
{
    HgetrfWorkspaceView ws;

    const int num_blocks = panel_TSLU_required_pivot_blocks(m, 0);
    ws.num_blocks_pivot_max = num_blocks;

    uint8_t* p = (uint8_t*)d_workspace;
    size_t off = 0;

    off = align_up(off, 256);
    ws.d_panel_block_val = (half*)(p + off);
    off += sizeof(half) * (size_t)num_blocks;

    off = align_up(off, 256);
    ws.d_panel_block_idx = (int*)(p + off);
    off += sizeof(int) * (size_t)num_blocks;

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
    cublasHandle_t cublas,
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

    // panel partition（只做 k_total 步）[fix] 我觉得这东西看起来有点多余了
    std::vector<int> panel_j0;
    std::vector<int> panel_ib;
    {
        int j0 = 0;
        // 把块先分好
        while (j0 < k_total) {
            int rem = k_total - j0;
            int ib_now = (rem >= panel_width) ? panel_width : rem;
            panel_j0.push_back(j0);
            panel_ib.push_back(ib_now);
            j0 += ib_now;
        }
    }
    const int num_panels = (int)panel_j0.size();
    if (num_panels == 0) return;

    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;

        launch_panel_TSLU(
            dA, m, lda,
            j0, ib_now, uc_now,
            d_ipiv,               // global ipiv (1-based)
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

        if (cublas) {
            // ✅ 快：不在这里 setStream（外层 hgetrf 已经设过）
            launch_A22_gemm_tc_range(
                dA, m, n, lda,
                j0, ib_now,
                col0, n2,
                cublas, stream_update);
        } else {
            launch_A22_gemm_naive_range(
                dA, m, n, lda,
                j0, ib_now,
                col0, n2,
                stream_update);
        }
    };

    auto do_gemm_full = [&](int j0, int ib_now) {
        int col0 = j0 + ib_now;
        int n2   = n - col0;
        do_gemm_range(j0, ib_now, col0, n2);
    };

    // pipeline start
    do_panel(panel_j0[0], panel_ib[0]);

    for (int p = 0; p < num_panels - 1; ++p) {
        int j0   = panel_j0[p];
        int ib_k = panel_ib[p];

        int j0_next = panel_j0[p + 1];
        int ib_next = panel_ib[p + 1];

        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);

        // look-ahead: update only ib_next columns for next panel
        {
            int col0_next = j0 + ib_k;
            int n2_next   = ib_next;
            do_gemm_range(j0, ib_k, col0_next, n2_next);
            CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_update));
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream_panel, ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        // remaining tail
        {
            int col0_tail = j0 + ib_k + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) do_gemm_range(j0, ib_k, col0_tail, n2_tail);
        }
    }

    // last panel
    {
        int j0_last = panel_j0[num_panels - 1];
        int ib_last = panel_ib[num_panels - 1];

        do_exchange(j0_last, ib_last);
        do_trsm(j0_last, ib_last);
        do_gemm_full(j0_last, ib_last);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_update));
    CUDA_CHECK(cudaStreamSynchronize(stream_panel));
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
    int panel_width,
    int uc,
    int* d_ipiv,          // device output: global ipiv (1-based), length min(m,n)
    int* d_info,          // device output: scalar int
    void* d_workspace,
    size_t workspace_bytes)
{
    if (!h || !dA || !d_ipiv || !d_info || !d_workspace) {
        fprintf(stderr, "hgetrf: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    const int k_total = (m < n) ? m : n;
    if (k_total <= 0) return;

    if (panel_width <= 0) panel_width = 128;
    if (panel_width > k_total) panel_width = k_total;

    // bind workspace
    HgetrfWorkspaceView ws = hgetrf_workspace_bind(d_workspace, workspace_bytes, m);

    // init info=0 on device
    CUDA_CHECK(cudaMemsetAsync(d_info, 0, sizeof(int), h->stream));

    // ✅ 快：只在这里设一次 stream，避免每次 GEMM setStream
    if (h->cublas) {
        CUBLAS_CHECK(cublasSetStream(h->cublas, h->stream));
    }

    // run
    hgetrf_blocked_half_dualstream_ws(
        dA, m, n, lda,
        panel_width, uc,
        h->cublas,
        d_ipiv,
        d_info,
        ws,
        h->stream,
        h->stream_panel,
        h->ev_piv_ready, 
        h->ev_next_ready);
}
