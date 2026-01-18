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
// Profiling helpers (enabled only when profile=true at runtime)
// ============================================================================
struct HgetrfProfileStat {
    float t_panel_ms    = 0.0f;
    float t_exchange_ms = 0.0f;
    float t_trsm_ms     = 0.0f;
    float t_gemm_ms     = 0.0f;

    int panel_calls   = 0;
    int exch_calls    = 0;
    int trsm_calls    = 0;
    int gemm_calls    = 0;

    // total (measured on update stream timeline)
    float t_total_ms = 0.0f;
};

static inline float safe_pct(float part, float total) {
    return (total > 0.0f) ? (100.0f * part / total) : 0.0f;
}

// ============================================================================
// 句柄：仿 cusolverDnHandle，内部只创建一次 stream/event
// ============================================================================
struct hgetrfHandle {
    cublasHandle_t cublas_update = nullptr;
    cublasHandle_t cublas_panel  = nullptr;
    bool owns_cublas_update = false;
    bool owns_cublas_panel  = false;

    cudaStream_t   stream = 0;
    cudaStream_t   stream_panel = nullptr;

    cudaEvent_t  ev_piv_ready  = nullptr;
    cudaEvent_t  ev_next_ready = nullptr;

    // ✅ 固定 256 panel 宽度（默认值）
    int panel_width = 256;
    int uc = 8;

    int    m_cached_max = 0;
    int    num_blocks_pivot_max = 0;
    size_t workspace_bytes = 0;
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

    CUBLAS_CHECK(cublasCreate(&h->cublas_update));
    h->owns_cublas_update = true;
    CUBLAS_CHECK(cublasCreate(&h->cublas_panel));
    h->owns_cublas_panel = true;

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

// ✅ 对外接口不变，但内部强制固定为 256
inline void hgetrfSetPanelWidth(hgetrfHandle_t h, int /*panel_width*/)
{
    if (!h) return;
    h->panel_width = 256;
}

inline void hgetrfSetUc(hgetrfHandle_t h, int uc)
{
    if (!h) return;
    h->uc = uc;
}

struct HgetrfWorkspaceView {
    half* d_panel_block_val = nullptr;
    int*  d_panel_block_idx = nullptr;
    int   num_blocks_pivot_max = 0;
};

static inline size_t align_up(size_t x, size_t a) {
    return (x + (a - 1)) & ~(a - 1);
}

// 计算需要多大的 buffer（✅ 不需要因为 panel 变 256 而改）
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

// 1 个 panel 最后启动一次检查 pivot==0
__global__ void hgetrf_check_panel_pivots_zero_kernel(
    const half* __restrict__ A,
    int lda,
    int j0,
    int ib,
    const int* __restrict__ d_ipiv,  // 1-based
    int* __restrict__ d_info)        // device scalar
{
    int tid = threadIdx.x;

    int cur = *d_info;
    if (cur != 0)
        return;

    for (int kk = tid; kk < ib; kk += blockDim.x) {
        int step = j0 + kk;
        int piv  = d_ipiv[step] - 1;  // to 0-based
        half pv  = A[piv + (size_t)step * lda];

        if (pv == __float2half(0.0f)) {
            atomicCAS(d_info, 0, step + 1);
        }
    }
}

// ============================================================================
// 核心：blocked LU with dual stream
// profile=true 时打印各阶段耗时占比
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
    cudaEvent_t  ev_next_ready,
    bool profile)       // ✅ 新增：运行时 profiling 开关
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

    // ✅ 固定 256（如果调用者传 0/负数，则兜底 256）
    if (panel_width <= 0)
        panel_width = 256;
    if (panel_width > k_total)
        panel_width = k_total;

    HgetrfProfileStat prof{};
    cudaEvent_t ev_total_begin = nullptr, ev_total_end = nullptr;

    // 每个阶段共用 2 个 event（profile=true 时才创建）
    cudaEvent_t ev_s = nullptr, ev_e = nullptr;
    cudaEvent_t ev_s_panel = nullptr, ev_e_panel = nullptr;

    if (profile) {
        CUDA_CHECK(cudaEventCreate(&ev_total_begin));
        CUDA_CHECK(cudaEventCreate(&ev_total_end));
        CUDA_CHECK(cudaEventCreate(&ev_s));
        CUDA_CHECK(cudaEventCreate(&ev_e));
        CUDA_CHECK(cudaEventCreate(&ev_s_panel));
        CUDA_CHECK(cudaEventCreate(&ev_e_panel));

        CUDA_CHECK(cudaEventRecord(ev_total_begin, stream_update));
    }

    auto do_panel = [&](int j0, int ib_now) {
        if (profile) CUDA_CHECK(cudaEventRecord(ev_s_panel, stream_panel));

        int uc_now = (uc > ib_now) ? ib_now : uc;

        launch_panel_TSLU(
            dA, m, lda,
            j0, ib_now, uc_now,
            d_ipiv,
            cublas_panel,
            stream_panel,
            ws.d_panel_block_val,
            ws.d_panel_block_idx,
            ws.num_blocks_pivot_max);

        // ✅ pivot==0 检测线程数跟 panel 走，最大 256
        int threads = (ib_now >= 256) ? 256 : ib_now;
        if (threads <= 0) threads = 1;

        hgetrf_check_panel_pivots_zero_kernel<<<1, threads, 0, stream_panel>>>(
            dA, lda, j0, ib_now, d_ipiv, d_info);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaEventRecord(ev_piv_ready, stream_panel));

        if (profile) {
            CUDA_CHECK(cudaEventRecord(ev_e_panel, stream_panel));
            CUDA_CHECK(cudaEventSynchronize(ev_e_panel));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s_panel, ev_e_panel));
            prof.t_panel_ms += ms;
            prof.panel_calls++;
        }
    };

    auto do_exchange = [&](int j0, int ib_now) {
        if (profile) CUDA_CHECK(cudaEventRecord(ev_s, stream_update));

        CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_piv_ready, 0));
        launch_A_exchange_trailing_device_piv(
            dA, m, n, lda, j0, ib_now,
            d_ipiv,
            stream_update);

        if (profile) {
            CUDA_CHECK(cudaEventRecord(ev_e, stream_update));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            prof.t_exchange_ms += ms;
            prof.exch_calls++;
        }
    };

    auto do_trsm = [&](int j0, int ib_now) {
        int col0   = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0 || ib_now <= 0) return;

        if (profile) CUDA_CHECK(cudaEventRecord(ev_s, stream_update));

        launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);

        if (profile) {
            CUDA_CHECK(cudaEventRecord(ev_e, stream_update));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            prof.t_trsm_ms += ms;
            prof.trsm_calls++;
        }
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;

        if (profile) CUDA_CHECK(cudaEventRecord(ev_s, stream_update));

        launch_A22_gemm_tc_range(
            dA, m, n, lda,
            j0, ib_now,
            col0, n2,
            cublas_update, stream_update);

        if (profile) {
            CUDA_CHECK(cudaEventRecord(ev_e, stream_update));
            CUDA_CHECK(cudaEventSynchronize(ev_e));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev_s, ev_e));
            prof.t_gemm_ms += ms;
            prof.gemm_calls++;
        }
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

    CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_panel));
    CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_next_ready, 0));

    if (profile) {
        // total end
        CUDA_CHECK(cudaEventRecord(ev_total_end, stream_update));
        CUDA_CHECK(cudaEventSynchronize(ev_total_end));
        CUDA_CHECK(cudaEventElapsedTime(&prof.t_total_ms, ev_total_begin, ev_total_end));

        float total = prof.t_total_ms;
        float parts = prof.t_panel_ms + prof.t_exchange_ms + prof.t_trsm_ms + prof.t_gemm_ms;

        printf("\n[hgetrf profile] (panel_width=%d, uc=%d)\n", panel_width, uc);
        printf("  total (update-stream wall): %8.3f ms\n", total);
        printf("  panel   : %8.3f ms  (%5.1f%%)  calls=%d\n",
               prof.t_panel_ms,    safe_pct(prof.t_panel_ms, total), prof.panel_calls);
        printf("  exchange: %8.3f ms  (%5.1f%%)  calls=%d\n",
               prof.t_exchange_ms, safe_pct(prof.t_exchange_ms, total), prof.exch_calls);
        printf("  trsm    : %8.3f ms  (%5.1f%%)  calls=%d\n",
               prof.t_trsm_ms,     safe_pct(prof.t_trsm_ms, total), prof.trsm_calls);
        printf("  gemm    : %8.3f ms  (%5.1f%%)  calls=%d\n",
               prof.t_gemm_ms,     safe_pct(prof.t_gemm_ms, total), prof.gemm_calls);
        printf("  parts sum: %8.3f ms  (%.1f%% of total)\n",
               parts, safe_pct(parts, total));

        // cleanup events
        CUDA_CHECK(cudaEventDestroy(ev_total_begin));
        CUDA_CHECK(cudaEventDestroy(ev_total_end));
        CUDA_CHECK(cudaEventDestroy(ev_s));
        CUDA_CHECK(cudaEventDestroy(ev_e));
        CUDA_CHECK(cudaEventDestroy(ev_s_panel));
        CUDA_CHECK(cudaEventDestroy(ev_e_panel));
    }
}

// ============================================================================
// Public API: hgetrf (cuSOLVER-like)
// ✅ 在原接口末尾增加 profile=false，可选参数，不影响旧调用点
// ============================================================================
inline void hgetrf(
    hgetrfHandle_t h,
    int m, int n,
    half* dA, int lda,
    half* d_workspace,
    int* d_ipiv,
    int* d_info,
    bool profile = false)   // ✅ 新增：默认 false
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

    // ✅ 内部固定 256，不改对外接口，但也不让你乱来
    int panel_width = 256;
    h->panel_width = 256;

    int uc = h->uc;
    if (panel_width > k_total) panel_width = k_total;

    // bind workspace（✅ 不需要因为 panel=256 改 workspace）
    HgetrfWorkspaceView ws = hgetrf_workspace_bind(
        d_workspace, h->workspace_bytes, h->num_blocks_pivot_max);

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
        h->ev_next_ready,
        profile);
}
