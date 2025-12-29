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
#include "A_pivrows.cuh"

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

// 句柄，仿 cusolver 的形式
struct hgetrfHandle {
    cublasHandle_t cublas = nullptr;
    // 双流设计，一条用于 panel，一条用于 update，并给出两个信号进行同步
    cudaStream_t   stream = 0;
    cudaStream_t stream_panel = nullptr;
    // panel 完成和 next panel 需求数据完成
    cudaEvent_t  ev_piv_ready = nullptr;
    cudaEvent_t  ev_next_ready = nullptr; 
};

using hgetrfHandle_t = hgetrfHandle*;

inline void hgetrfCreate(hgetrfHandle_t* out)
{
    if (!out) 
        return;
    hgetrfHandle_t h = new hgetrfHandle;

    CUDA_CHECK(cudaStreamCreateWithFlags(&h->stream_panel, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_piv_ready,  cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&h->ev_next_ready, cudaEventDisableTiming));

    *out = h;
}

inline void hgetrfDestroy(hgetrfHandle_t h)
{
    if (!h) return;
    if (h->ev_piv_ready)  CUDA_CHECK(cudaEventDestroy(h->ev_piv_ready));
    if (h->ev_next_ready) CUDA_CHECK(cudaEventDestroy(h->ev_next_ready));
    if (h->stream_panel)  CUDA_CHECK(cudaStreamDestroy(h->stream_panel));
    delete h;
}
// 外部控制主 stream 和 cublas handle
inline void hgetrfSetStream(hgetrfHandle_t h, cudaStream_t s)
{
    if (!h) return;
    h->stream = s;
}

inline void hgetrfSetCublas(hgetrfHandle_t h, cublasHandle_t c)
{
    if (!h) return;
    h->cublas = c;
}


// 这里可能是需要修改的，加上其他各个部分所需要的 workspace 数据 [fix]
struct HgetrfWorkspaceView {
    half* d_panel_block_val = nullptr;
    int*  d_panel_block_idx = nullptr;
    // 每行经过置换后对应的原始行号
    int*  d_piv_rows        = nullptr;
    // pivot 的最大块数
    int num_blocks_pivot_max = 0;
};

// 对齐 helper
static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// 计算需要算多大的 workspace
inline void hgetrf_bufferSize(
    hgetrfHandle_t /*h*/,
    int m, int /*n*/, int /*lda*/,
    int /*panel_width*/,
    size_t* device_bytes)
{
    if (!device_bytes) return;

    // 你原实现里就是这么算的
    const int num_blocks = panel_TSLU_required_pivot_blocks(m, 0);

    size_t bytes = 0;
    bytes = align_up(bytes, 256);
    bytes += sizeof(half) * (size_t)num_blocks;   // d_panel_block_val
    bytes = align_up(bytes, 256);
    bytes += sizeof(int)  * (size_t)num_blocks;   // d_panel_block_idx
    bytes = align_up(bytes, 256);
    // [fix] 感觉这里不需要整个 m 的空间，而是只需要申请本次 TSLU 的行数空间，比如说传入 32768 * 16384 的，则只需要 16384 个
    bytes += sizeof(int)  * (size_t)m;            // d_piv_rows

    *device_bytes = bytes;
}

// workspace 切分
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

    off = align_up(off, 256);
    ws.d_piv_rows = (int*)(p + off);
    off += sizeof(int) * (size_t)m;

    if (off > workspace_bytes) {
        fprintf(stderr, "hgetrf_workspace_bind: workspace_bytes too small (need %zu, got %zu)\n",
                off, workspace_bytes);
        std::exit(EXIT_FAILURE);
    }
    return ws;
}

// ============================================================================
// 核心实现：把你原 dualstream 版本改成“外置 workspace/stream/event”
// - 不 malloc/free
// - 不创建/销毁 stream/event
// - 输出 pivot rows 到 device: ws.d_piv_rows
// ============================================================================
inline void hgetrf_blocked_half_dualstream_ws(
    half* dA,
    int m, int n, int lda,
    int panel_width, int uc,
    cublasHandle_t cublas,
    int* d_ipiv_rel,
    const HgetrfWorkspaceView& ws,
    cudaStream_t stream_update,
    cudaStream_t stream_panel,
    cudaEvent_t  ev_piv_ready,
    cudaEvent_t  ev_next_ready)
{
    if (!dA || !d_ipiv_rel) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!ws.d_panel_block_val || !ws.d_panel_block_idx || !ws.d_piv_rows) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: invalid workspace.\n");
        std::exit(EXIT_FAILURE);
    }
    if (!stream_panel || !ev_piv_ready || !ev_next_ready) {
        fprintf(stderr, "hgetrf_blocked_half_dualstream_ws: stream/event not set.\n");
        std::exit(EXIT_FAILURE);
    }

    // init device piv_rows = [0..m-1]
    launch_init_piv_rows(ws.d_piv_rows, m, stream_update);

    // 划分 panels
    std::vector<int> panel_j0;
    std::vector<int> panel_ib;
    {
        int j0 = 0;
        while (j0 < n) {
            int rem = n - j0;
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
            d_ipiv_rel,
            stream_panel,
            ws.d_panel_block_val,
            ws.d_panel_block_idx,
            ws.num_blocks_pivot_max);

        // device piv_rows 更新（方向B）
        launch_apply_panel_pivots_to_pivrows(
            ws.d_piv_rows, m, j0, ib_now, d_ipiv_rel, stream_panel);

        CUDA_CHECK(cudaEventRecord(ev_piv_ready, stream_panel));
    };

    auto do_exchange = [&](int j0, int ib_now) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_piv_ready, 0));
        launch_A_exchange_trailing_device_piv(
            dA, m, n, lda, j0, ib_now, d_ipiv_rel, stream_update);
    };

    auto do_trsm = [&](int j0, int ib_now) {
        int col0   = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0 || ib_now <= 0) return;

        // 你目前 test 里用的是 TRSM_CUSTOM_KERNEL
        // 如需切换 cublasTrsmEx，可以在这里扩展
        launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;

        // test 里 use_tensor_core_gemm = true
        if (cublas) {
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

    // pipeline
    do_panel(panel_j0[0], panel_ib[0]);

    for (int k = 0; k < num_panels - 1; ++k) {
        int j0   = panel_j0[k];
        int ib_k = panel_ib[k];

        int j0_next = panel_j0[k + 1];
        int ib_next = panel_ib[k + 1];

        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);

        // next panel needs only ib_next columns
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
// - workspace 外置
// - 输出 piv_rows 在 device 上 (d_piv_rows_out)
// ============================================================================
inline void hgetrf(
    hgetrfHandle_t h,
    int m, int n,
    half* dA, int lda,
    int panel_width,
    int uc,
    int* d_ipiv_rel,     // device
    int* d_piv_rows_out, // device (输出：最终 piv_rows)
    void* d_workspace,
    size_t workspace_bytes,
    int* info_host)      // 简化：目前用 host info (0=success)
{
    if (!h || !dA || !d_ipiv_rel || !d_workspace || !d_piv_rows_out) {
        fprintf(stderr, "hgetrf: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (panel_width <= 0) panel_width = 128;

    // bind workspace view
    HgetrfWorkspaceView ws = hgetrf_workspace_bind(d_workspace, workspace_bytes, m);

    // run
    hgetrf_blocked_half_dualstream_ws(
        dA, m, n, lda,
        panel_width, uc,
        h->cublas,
        d_ipiv_rel,
        ws,
        h->stream,
        h->stream_panel,
        h->ev_piv_ready,
        h->ev_next_ready);

    // 把 device piv_rows 拷到用户提供的 d_piv_rows_out（device->device）
    CUDA_CHECK(cudaMemcpyAsync(
        d_piv_rows_out, ws.d_piv_rows,
        sizeof(int) * (size_t)m,
        cudaMemcpyDeviceToDevice,
        h->stream));
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    if (info_host) *info_host = 0;
}
