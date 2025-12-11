#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

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
// 配置结构体
// ============================================================================

struct HgetrfConfig {
    bool use_tensor_core_gemm = true;
    
    enum TrsmMode {
        TRSM_CUSTOM_KERNEL,
        TRSM_CUBLAS_HALF,
        TRSM_CUBLAS_FLOAT
    };
    TrsmMode trsm_mode = TRSM_CUSTOM_KERNEL;
    
    bool use_batched_exchange = true;
    bool use_multi_stream = false;  // 新增：启用多流水线
    bool verbose = false;
    bool check_errors = false;
    
    int fixed_panel_width = 128;
};

struct HgetrfTimers {
    double panel_ms    = 0.0;
    double exchange_ms = 0.0;
    double trsm_ms     = 0.0;
    double gemm_ms     = 0.0;
    double total_ms    = 0.0;
    int    panels      = 0;
};

// ============================================================================
// 常量定义
// ============================================================================

constexpr int MAX_PANEL_WIDTH = 256;
constexpr int DEFAULT_PANEL_WIDTH = 128;

// ============================================================================
// 辅助函数
// ============================================================================

inline void hgetrf_update_pivots_host(
    int j0, int ib,
    const int* h_ipiv_rel,
    int m,
    int* piv_rows)
{
    for (int k = 0; k < ib; ++k) {
        int r1 = j0 + k;
        int r2 = r1 + h_ipiv_rel[k];
        if (r1 < 0 || r1 >= m || r2 < 0 || r2 >= m) continue;
        if (r1 == r2) continue;
        int tmp      = piv_rows[r1];
        piv_rows[r1] = piv_rows[r2];
        piv_rows[r2] = tmp;
    }
}

// ============================================================================
// 核心接口：多流水线版本
// ============================================================================

inline void hgetrf_blocked_half_multistream(
    half* dA,
    int m, int n, int lda,
    int ib_pref, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream,
    HgetrfTimers* timers)
{
    if (!dA || !d_ipiv_rel || !h_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_blocked_half_multistream: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; ++i) {
        piv_rows[i] = i;
    }

    int panel_width = config.fixed_panel_width;
    if (panel_width <= 0 || panel_width > MAX_PANEL_WIDTH) {
        panel_width = DEFAULT_PANEL_WIDTH;
    }
    if (ib_pref > 0 && ib_pref < panel_width) {
        panel_width = ib_pref;
    }

    std::vector<int> panel_j0;
    std::vector<int> panel_ib;
    {
        int j0 = 0;
        while (j0 < n) {
            int remaining = n - j0;
            int ib_now = (remaining >= panel_width) ? panel_width : remaining;
            panel_j0.push_back(j0);
            panel_ib.push_back(ib_now);
            j0 += ib_now;
        }
    }
    const int num_panels = (int)panel_j0.size();
    if (num_panels == 0) return;

    if (config.verbose) {
        printf("[hgetrf] Multi-stream mode, panel width: %d, panels: %d\n",
               panel_width, num_panels);
    }

    // 创建4个独立stream
    cudaStream_t stream_panel = nullptr;
    cudaStream_t stream_exchange = nullptr;
    cudaStream_t stream_trsm = nullptr;
    cudaStream_t stream_gemm = nullptr;
    
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_panel, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_exchange, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_trsm, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_gemm, cudaStreamNonBlocking));

    // Events for synchronization
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev_total0 = nullptr, ev_total1 = nullptr;
    cudaEvent_t ev_panel_done = nullptr;
    cudaEvent_t ev_exchange_done = nullptr;
    cudaEvent_t ev_trsm_done = nullptr;

    if (timers) {
        *timers = HgetrfTimers{};
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev_total0));
        CUDA_CHECK(cudaEventCreate(&ev_total1));
        CUDA_CHECK(cudaEventRecord(ev_total0, stream_panel));
    }
    
    CUDA_CHECK(cudaEventCreate(&ev_panel_done));
    CUDA_CHECK(cudaEventCreate(&ev_exchange_done));
    CUDA_CHECK(cudaEventCreate(&ev_trsm_done));

    auto time_range = [&](cudaStream_t s, double* acc_ms, auto&& fn) {
        if (!timers || !acc_ms) {
            fn();
            return;
        }
        CUDA_CHECK(cudaEventRecord(ev0, s));
        fn();
        CUDA_CHECK(cudaEventRecord(ev1, s));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        *acc_ms += (double)ms;
    };

    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;
        
        time_range(stream_panel, timers ? &timers->panel_ms : nullptr, [&]() {
            launch_panel_TSLU(dA, m, lda, j0, ib_now, uc_now, d_ipiv_rel, stream_panel);
        });
        
        CUDA_CHECK(cudaMemcpyAsync(h_ipiv_rel, d_ipiv_rel,
                                   sizeof(int) * ib_now,
                                   cudaMemcpyDeviceToHost,
                                   stream_panel));
        CUDA_CHECK(cudaStreamSynchronize(stream_panel));
        hgetrf_update_pivots_host(j0, ib_now, h_ipiv_rel, m, piv_rows);
        if (timers) timers->panels += 1;
        
        CUDA_CHECK(cudaEventRecord(ev_panel_done, stream_panel));
    };

    auto do_exchange = [&](int j0, int ib_now) {
        CUDA_CHECK(cudaStreamWaitEvent(stream_exchange, ev_panel_done, 0));
        
        time_range(stream_exchange, timers ? &timers->exchange_ms : nullptr, [&]() {
            launch_A_exchange_trailing(dA, m, n, lda, j0, ib_now, h_ipiv_rel, stream_exchange);
        });
        
        CUDA_CHECK(cudaEventRecord(ev_exchange_done, stream_exchange));
    };

    auto do_trsm = [&](int j0, int ib_now) {
        int col0   = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0 || ib_now <= 0) return;

        CUDA_CHECK(cudaStreamWaitEvent(stream_trsm, ev_exchange_done, 0));

        time_range(stream_trsm, timers ? &timers->trsm_ms : nullptr, [&]() {
            if (config.trsm_mode == HgetrfConfig::TRSM_CUBLAS_HALF) {
#if defined(CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP)
                const float alpha = 1.0f;
                half* L11 = dA + j0 + (size_t)j0   * lda;
                half* A12 = dA + j0 + (size_t)col0 * lda;
                CUBLAS_CHECK(cublasSetStream(handle, stream_trsm));
                CUBLAS_CHECK(
                    cublasTrsmEx(
                        handle,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_UNIT,
                        ib_now,
                        ntrail,
                        &alpha,
                        L11, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        CUDA_R_32F,
                        CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP));
#else
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_trsm);
#endif
            } else {
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_trsm);
            }
        });
        
        CUDA_CHECK(cudaEventRecord(ev_trsm_done, stream_trsm));
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;
        
        CUDA_CHECK(cudaStreamWaitEvent(stream_gemm, ev_trsm_done, 0));
        
        time_range(stream_gemm, timers ? &timers->gemm_ms : nullptr, [&]() {
            if (config.use_tensor_core_gemm && handle) {
                launch_A22_gemm_tc_range(dA, m, n, lda, j0, ib_now, col0, n2, handle, stream_gemm);
            } else {
                launch_A22_gemm_naive_range(dA, m, n, lda, j0, ib_now, col0, n2, stream_gemm);
            }
        });
    };

    auto do_gemm_full = [&](int j0, int ib_now) {
        int col0 = j0 + ib_now;
        int n2   = n - col0;
        do_gemm_range(j0, ib_now, col0, n2);
    };

    // Pipeline with multi-stream overlap
    do_panel(panel_j0[0], panel_ib[0]);

    for (int k = 0; k < num_panels - 1; ++k) {
        int j0   = panel_j0[k];
        int ib_k = panel_ib[k];

        int j0_next   = panel_j0[k+1];
        int ib_next   = panel_ib[k+1];

        // 启动多个stream并行
        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);
        
        {
            int col0_next = j0 + ib_k;
            int n2_next   = ib_next;
            do_gemm_range(j0, ib_k, col0_next, n2_next);
        }

        do_panel(j0_next, ib_next);

        {
            int col0_tail = j0 + ib_k + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) {
                do_gemm_range(j0, ib_k, col0_tail, n2_tail);
            }
        }
    }

    {
        int k_last = num_panels - 1;
        int j0_last = panel_j0[k_last];
        int ib_last = panel_ib[k_last];

        do_exchange(j0_last, ib_last);
        do_trsm(j0_last, ib_last);
        do_gemm_full(j0_last, ib_last);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_panel));
    CUDA_CHECK(cudaStreamSynchronize(stream_exchange));
    CUDA_CHECK(cudaStreamSynchronize(stream_trsm));
    CUDA_CHECK(cudaStreamSynchronize(stream_gemm));

    if (timers) {
        CUDA_CHECK(cudaEventRecord(ev_total1, stream_panel));
        CUDA_CHECK(cudaEventSynchronize(ev_total1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_total0, ev_total1));
        timers->total_ms = (double)ms;

        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaEventDestroy(ev_total0));
        CUDA_CHECK(cudaEventDestroy(ev_total1));
    }
    
    CUDA_CHECK(cudaEventDestroy(ev_panel_done));
    CUDA_CHECK(cudaEventDestroy(ev_exchange_done));
    CUDA_CHECK(cudaEventDestroy(ev_trsm_done));

    CUDA_CHECK(cudaStreamDestroy(stream_panel));
    CUDA_CHECK(cudaStreamDestroy(stream_exchange));
    CUDA_CHECK(cudaStreamDestroy(stream_trsm));
    CUDA_CHECK(cudaStreamDestroy(stream_gemm));
}

// ============================================================================
// 原版双stream（保持不变）
// ============================================================================

inline void hgetrf_blocked_half_optimized(
    half* dA,
    int m, int n, int lda,
    int ib_pref, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream,
    HgetrfTimers* timers)
{
    if (!dA || !d_ipiv_rel || !h_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_blocked_half_optimized: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; ++i) {
        piv_rows[i] = i;
    }

    int panel_width = config.fixed_panel_width;
    if (panel_width <= 0 || panel_width > MAX_PANEL_WIDTH) {
        panel_width = DEFAULT_PANEL_WIDTH;
    }
    if (ib_pref > 0 && ib_pref < panel_width) {
        panel_width = ib_pref;
    }

    std::vector<int> panel_j0;
    std::vector<int> panel_ib;
    {
        int j0 = 0;
        while (j0 < n) {
            int remaining = n - j0;
            int ib_now = (remaining >= panel_width) ? panel_width : remaining;
            panel_j0.push_back(j0);
            panel_ib.push_back(ib_now);
            j0 += ib_now;
        }
    }
    const int num_panels = (int)panel_j0.size();
    if (num_panels == 0) return;

    if (config.verbose) {
        printf("[hgetrf] Fixed panel width: %d, Total panels: %d\n",
               panel_width, num_panels);
    }

    cudaStream_t stream_update = stream;
    cudaStream_t stream_panel  = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_panel, cudaStreamNonBlocking));

    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev_total0 = nullptr, ev_total1 = nullptr;
    cudaEvent_t ev_next_ready = nullptr;

    if (timers) {
        *timers = HgetrfTimers{};
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventCreate(&ev_total0));
        CUDA_CHECK(cudaEventCreate(&ev_total1));
        CUDA_CHECK(cudaEventCreate(&ev_next_ready));
        CUDA_CHECK(cudaEventRecord(ev_total0, stream_update));
    } else {
        CUDA_CHECK(cudaEventCreate(&ev_next_ready));
    }

    auto time_range = [&](cudaStream_t s, double* acc_ms, auto&& fn) {
        if (!timers || !acc_ms) {
            fn();
            return;
        }
        CUDA_CHECK(cudaEventRecord(ev0, s));
        fn();
        CUDA_CHECK(cudaEventRecord(ev1, s));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        *acc_ms += (double)ms;
    };

    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;
        
        time_range(stream_panel, timers ? &timers->panel_ms : nullptr, [&]() {
            launch_panel_TSLU(dA, m, lda, j0, ib_now, uc_now, d_ipiv_rel, stream_panel);
        });
        
        CUDA_CHECK(cudaMemcpyAsync(h_ipiv_rel, d_ipiv_rel,
                                   sizeof(int) * ib_now,
                                   cudaMemcpyDeviceToHost,
                                   stream_panel));
        CUDA_CHECK(cudaStreamSynchronize(stream_panel));
        hgetrf_update_pivots_host(j0, ib_now, h_ipiv_rel, m, piv_rows);
        if (timers) timers->panels += 1;
    };

    auto do_exchange = [&](int j0, int ib_now) {
        time_range(stream_update, timers ? &timers->exchange_ms : nullptr, [&]() {
            launch_A_exchange_trailing(dA, m, n, lda, j0, ib_now, h_ipiv_rel, stream_update);
        });
    };

    auto do_trsm = [&](int j0, int ib_now) {
        int col0   = j0 + ib_now;
        int ntrail = n - col0;
        if (ntrail <= 0 || ib_now <= 0) return;

        time_range(stream_update, timers ? &timers->trsm_ms : nullptr, [&]() {
            if (config.trsm_mode == HgetrfConfig::TRSM_CUBLAS_HALF) {
#if defined(CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP)
                const float alpha = 1.0f;
                half* L11 = dA + j0 + (size_t)j0   * lda;
                half* A12 = dA + j0 + (size_t)col0 * lda;
                CUBLAS_CHECK(cublasSetStream(handle, stream_update));
                CUBLAS_CHECK(
                    cublasTrsmEx(
                        handle,
                        CUBLAS_SIDE_LEFT,
                        CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N,
                        CUBLAS_DIAG_UNIT,
                        ib_now,
                        ntrail,
                        &alpha,
                        L11, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        CUDA_R_32F,
                        CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP));
#else
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
#endif
            } else {
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
            }
        });
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;
        time_range(stream_update, timers ? &timers->gemm_ms : nullptr, [&]() {
            if (config.use_tensor_core_gemm && handle) {
                launch_A22_gemm_tc_range(dA, m, n, lda, j0, ib_now, col0, n2, handle, stream_update);
            } else {
                launch_A22_gemm_naive_range(dA, m, n, lda, j0, ib_now, col0, n2, stream_update);
            }
        });
    };

    auto do_gemm_full = [&](int j0, int ib_now) {
        int col0 = j0 + ib_now;
        int n2   = n - col0;
        do_gemm_range(j0, ib_now, col0, n2);
    };

    // Pipeline: look-ahead=1
    do_panel(panel_j0[0], panel_ib[0]);

    for (int k = 0; k < num_panels - 1; ++k) {
        int j0   = panel_j0[k];
        int ib_k = panel_ib[k];

        int j0_next   = panel_j0[k+1];
        int ib_next   = panel_ib[k+1];

        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);

        {
            int col0_next = j0 + ib_k;
            int n2_next   = ib_next;
            do_gemm_range(j0, ib_k, col0_next, n2_next);
            CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_update));
        }

        CUDA_CHECK(cudaStreamWaitEvent(stream_panel, ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        {
            int col0_tail = j0 + ib_k + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) {
                do_gemm_range(j0, ib_k, col0_tail, n2_tail);
            }
        }
    }

    {
        int k_last = num_panels - 1;
        int j0_last = panel_j0[k_last];
        int ib_last = panel_ib[k_last];

        do_exchange(j0_last, ib_last);
        do_trsm(j0_last, ib_last);
        do_gemm_full(j0_last, ib_last);
    }

    CUDA_CHECK(cudaStreamSynchronize(stream_update));
    CUDA_CHECK(cudaStreamSynchronize(stream_panel));

    if (timers) {
        CUDA_CHECK(cudaEventRecord(ev_total1, stream_update));
        CUDA_CHECK(cudaEventSynchronize(ev_total1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev_total0, ev_total1));
        timers->total_ms = (double)ms;

        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));
        CUDA_CHECK(cudaEventDestroy(ev_total0));
        CUDA_CHECK(cudaEventDestroy(ev_total1));
        CUDA_CHECK(cudaEventDestroy(ev_next_ready));
    } else {
        CUDA_CHECK(cudaEventDestroy(ev_next_ready));
    }

    CUDA_CHECK(cudaStreamDestroy(stream_panel));
}

// ============================================================================
// 智能接口
// ============================================================================

inline void hgetrf_auto(
    half* dA,
    int m, int n, int lda,
    int ib_request, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream = 0,
    HgetrfTimers* timers = nullptr)
{
    if (!dA || !d_ipiv_rel || !h_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_auto: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    int panel_width = config.fixed_panel_width;
    if (panel_width <= 0 || panel_width > MAX_PANEL_WIDTH) {
        panel_width = DEFAULT_PANEL_WIDTH;
    }
    if (ib_request > 0 && ib_request < panel_width) {
        panel_width = ib_request;
    }

    if (n <= panel_width) {
        if (config.verbose) {
            printf("[hgetrf_auto] Single panel mode (n=%d)\n", n);
        }
        
        for (int i = 0; i < m; ++i) {
            piv_rows[i] = i;
        }
        
        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        if (timers) {
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            CUDA_CHECK(cudaEventRecord(ev0, stream));
        }
        
        launch_panel_TSLU(dA, m, lda, 0, n, uc, d_ipiv_rel, stream);
        
        if (timers) {
            CUDA_CHECK(cudaEventRecord(ev1, stream));
            CUDA_CHECK(cudaEventSynchronize(ev1));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
            timers->panel_ms = (double)ms;
            timers->total_ms = (double)ms;
            timers->panels = 1;
            CUDA_CHECK(cudaEventDestroy(ev0));
            CUDA_CHECK(cudaEventDestroy(ev1));
        }
        
        CUDA_CHECK(cudaMemcpyAsync(h_ipiv_rel, d_ipiv_rel,
                                   sizeof(int) * n,
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        hgetrf_update_pivots_host(0, n, h_ipiv_rel, m, piv_rows);
        
        return;
    }
    
    if (config.verbose) {
        printf("[hgetrf_auto] %s mode (n=%d, panel_width=%d)\n",
               config.use_multi_stream ? "Multi-stream" : "Dual-stream",
               n, panel_width);
    }
    
    if (config.use_multi_stream) {
        hgetrf_blocked_half_multistream(dA, m, n, lda,
                                       panel_width, uc,
                                       handle,
                                       d_ipiv_rel,
                                       h_ipiv_rel,
                                       piv_rows,
                                       config,
                                       stream,
                                       timers);
    } else {
        hgetrf_blocked_half_optimized(dA, m, n, lda,
                                      panel_width, uc,
                                      handle,
                                      d_ipiv_rel,
                                      h_ipiv_rel,
                                      piv_rows,
                                      config,
                                      stream,
                                      timers);
    }
}

// ============================================================================
// 兼容接口
// ============================================================================

inline void hgetrf_blocked_half(
    half* dA,
    int m, int n, int lda,
    int ib_pref, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    cudaStream_t stream = 0,
    bool use_cublas_tc   = true,
    bool use_cublas_trsm = false,
    HgetrfTimers* timers = nullptr)
{
    HgetrfConfig config;
    config.use_tensor_core_gemm = use_cublas_tc;
    config.trsm_mode = use_cublas_trsm ? 
                       HgetrfConfig::TRSM_CUBLAS_HALF : 
                       HgetrfConfig::TRSM_CUSTOM_KERNEL;
    config.verbose = false;
    config.use_multi_stream = false;
    config.fixed_panel_width = (ib_pref > 0 && ib_pref <= MAX_PANEL_WIDTH) ? 
                               ib_pref : DEFAULT_PANEL_WIDTH;
    
    hgetrf_auto(dA, m, n, lda,
               ib_pref, uc,
               handle,
               d_ipiv_rel,
               h_ipiv_rel,
               piv_rows,
               config,
               stream,
               timers);
}

inline void hgetrf_blocked_half_ex(
    half* dA,
    int m, int n, int lda,
    int ib_pref, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream,
    HgetrfTimers* timers)
{
    hgetrf_auto(dA, m, n, lda,
               ib_pref, uc,
               handle,
               d_ipiv_rel,
               h_ipiv_rel,
               piv_rows,
               config,
               stream,
               timers);
}

// ============================================================================
// cuSOLVER 风格兼容接口
// ============================================================================

/**
 * hgetrf - 半精度 LU 分解（cuSOLVER 风格接口）
 *
 * 功能等价于 cusolverDnSgetrf，但使用半精度
 *
 * @param handle    cuBLAS handle（用于 Tensor Core GEMM）
 * @param m         矩阵行数
 * @param n         矩阵列数
 * @param dA        设备端矩阵指针，分解后包含 L 和 U
 * @param lda       leading dimension
 * @param ipiv      设备端主元数组（输出，1-based 绝对索引，长度 min(m,n)）
 *                  如果为 nullptr，则不返回主元信息（无主元分解）
 * @param info      设备端状态信息（0 = 成功），可以为 nullptr
 * @param stream    CUDA stream（可选）
 * @param nb        面板宽度（可选，默认128）
 *
 * @return 0 成功，非0 失败
 */
inline int hgetrf(
    cublasHandle_t handle,
    int m, int n,
    half* dA,
    int lda,
    int* ipiv,
    int* info,
    cudaStream_t stream = 0,
    int nb = 128)
{
    // 参数检查
    if (!dA) {
        if (info) {
            int h_info = -4;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -4;
    }
    if (m < 0) {
        if (info) {
            int h_info = -2;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -2;
    }
    if (n < 0) {
        if (info) {
            int h_info = -3;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -3;
    }
    if (lda < std::max(1, m)) {
        if (info) {
            int h_info = -5;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -5;
    }
    
    // 空矩阵直接返回成功
    if (m == 0 || n == 0) {
        if (info) {
            int h_info = 0;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return 0;
    }

    int minmn = std::min(m, n);
    
    // 分配临时缓冲区
    int* d_ipiv_rel = nullptr;
    int* h_ipiv_rel = nullptr;
    int* piv_rows = nullptr;
    
    cudaError_t err = cudaMallocAsync(&d_ipiv_rel, sizeof(int) * minmn, stream);
    if (err != cudaSuccess) {
        if (info) {
            int h_info = -100;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -100;
    }
    
    h_ipiv_rel = (int*)std::malloc(sizeof(int) * minmn);
    piv_rows = (int*)std::malloc(sizeof(int) * m);
    
    if (!h_ipiv_rel || !piv_rows) {
        if (d_ipiv_rel) cudaFreeAsync(d_ipiv_rel, stream);
        if (h_ipiv_rel) std::free(h_ipiv_rel);
        if (piv_rows) std::free(piv_rows);
        if (info) {
            int h_info = -101;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -101;
    }

    // 配置
    HgetrfConfig config;
    config.use_tensor_core_gemm = (handle != nullptr);
    config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
    config.fixed_panel_width = nb;
    config.verbose = false;
    config.use_multi_stream = false;

    // 执行分解
    hgetrf_auto(dA, m, n, lda,
               nb, nb,
               handle,
               d_ipiv_rel,
               h_ipiv_rel,
               piv_rows,
               config,
               stream,
               nullptr);

    // 转换主元索引到 1-based 绝对索引（与 cuSOLVER 兼容）
    if (ipiv) {
        std::vector<int> h_ipiv(minmn);
        for (int i = 0; i < minmn; ++i) {
            h_ipiv[i] = piv_rows[i] + 1;  // 1-based
        }
        cudaMemcpyAsync(ipiv, h_ipiv.data(),
                        sizeof(int) * minmn,
                        cudaMemcpyHostToDevice, stream);
    }

    // 设置 info = 0（成功）
    if (info) {
        int h_info = 0;
        cudaMemcpyAsync(info, &h_info, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    // 同步并清理
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_ipiv_rel, stream);
    std::free(h_ipiv_rel);
    std::free(piv_rows);

    return 0;
}

/**
 * hgetrf_bufferSize - 获取工作空间大小（兼容接口）
 *
 * 注意：本实现内部自动分配，不需要外部工作空间
 */
inline int hgetrf_bufferSize(
    cublasHandle_t handle,
    int m, int n,
    half* dA,
    int lda,
    int* lwork)
{
    (void)handle;
    (void)m;
    (void)n;
    (void)dA;
    (void)lda;
    if (lwork) {
        *lwork = 0;  // 内部自动分配，不需要外部工作空间
    }
    return 0;
}

/**
 * hgetrf_simple - 最简接口（不返回主元信息）
 *
 * 适用于只需要分解结果，不关心主元的场景
 */
inline int hgetrf_simple(
    half* dA,
    int m, int n,
    int lda,
    cudaStream_t stream = 0,
    int nb = 128)
{
    cublasHandle_t handle = nullptr;
    cublasStatus_t st = cublasCreate(&handle);
    if (st != CUBLAS_STATUS_SUCCESS) {
        return -200;
    }
    cublasSetStream(handle, stream);
    
    int result = hgetrf(handle, m, n, dA, lda, nullptr, nullptr, stream, nb);
    
    cublasDestroy(handle);
    return result;
}

/**
 * hgetrf_with_timers - 带计时的接口
 *
 * 用于性能分析
 */
inline int hgetrf_with_timers(
    cublasHandle_t handle,
    int m, int n,
    half* dA,
    int lda,
    int* ipiv,
    int* info,
    HgetrfTimers* timers,
    cudaStream_t stream = 0,
    int nb = 128)
{
    // 参数检查
    if (!dA) {
        if (info) {
            int h_info = -4;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -4;
    }
    if (m <= 0 || n <= 0) {
        if (info) {
            int h_info = 0;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return 0;
    }
    if (lda < m) {
        if (info) {
            int h_info = -5;
            cudaMemcpyAsync(info, &h_info, sizeof(int), cudaMemcpyHostToDevice, stream);
        }
        return -5;
    }

    int minmn = std::min(m, n);
    
    // 分配临时缓冲区
    int* d_ipiv_rel = nullptr;
    int* h_ipiv_rel = nullptr;
    int* piv_rows = nullptr;
    
    cudaError_t err = cudaMallocAsync(&d_ipiv_rel, sizeof(int) * minmn, stream);
    if (err != cudaSuccess) {
        return -100;
    }
    
    h_ipiv_rel = (int*)std::malloc(sizeof(int) * minmn);
    piv_rows = (int*)std::malloc(sizeof(int) * m);
    
    if (!h_ipiv_rel || !piv_rows) {
        if (d_ipiv_rel) cudaFreeAsync(d_ipiv_rel, stream);
        if (h_ipiv_rel) std::free(h_ipiv_rel);
        if (piv_rows) std::free(piv_rows);
        return -101;
    }

    // 配置
    HgetrfConfig config;
    config.use_tensor_core_gemm = (handle != nullptr);
    config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
    config.fixed_panel_width = nb;
    config.verbose = false;
    config.use_multi_stream = false;

    // 执行分解（带计时）
    hgetrf_auto(dA, m, n, lda,
               nb, nb,
               handle,
               d_ipiv_rel,
               h_ipiv_rel,
               piv_rows,
               config,
               stream,
               timers);

    // 转换主元索引
    if (ipiv) {
        std::vector<int> h_ipiv(minmn);
        for (int i = 0; i < minmn; ++i) {
            h_ipiv[i] = piv_rows[i] + 1;
        }
        cudaMemcpyAsync(ipiv, h_ipiv.data(),
                        sizeof(int) * minmn,
                        cudaMemcpyHostToDevice, stream);
    }

    // 设置 info
    if (info) {
        int h_info = 0;
        cudaMemcpyAsync(info, &h_info, sizeof(int),
                        cudaMemcpyHostToDevice, stream);
    }

    // 同步并清理
    cudaStreamSynchronize(stream);
    cudaFreeAsync(d_ipiv_rel, stream);
    std::free(h_ipiv_rel);
    std::free(piv_rows);

    return 0;
}