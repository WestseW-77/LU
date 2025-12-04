#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstdio>
#include <cstdlib>

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

/**
 * LU 分解的详细配置选项
 */
struct HgetrfConfig {
    // GEMM 配置
    bool use_tensor_core_gemm = true;   // 是否使用 Tensor Core GEMM
    
    // TRSM 配置
    enum TrsmMode {
        TRSM_CUSTOM_KERNEL,             // 使用自定义 half kernel
        TRSM_CUBLAS_HALF,               // 使用 cuBLAS half TRSM (如果支持)
        TRSM_CUBLAS_FLOAT               // 使用 cuBLAS float TRSM (需要类型转换)
    };
    TrsmMode trsm_mode = TRSM_CUSTOM_KERNEL;
    
    // Exchange 配置
    bool use_batched_exchange = true;   // 是否批量交换
    
    // Stream 配置
    bool use_dual_stream = false;       // 是否使用双 stream overlap
    
    // 调试选项
    bool verbose = false;               // 是否输出详细信息
    bool check_errors = false;          // 是否每步检查错误
};

/**
 * 性能计时结果（毫秒）
 */
struct HgetrfTimers {
    double panel_ms    = 0.0;
    double exchange_ms = 0.0;
    double trsm_ms     = 0.0;
    double gemm_ms     = 0.0;
    double total_ms    = 0.0;
    int    panels      = 0;
};

// ============================================================================
// 辅助函数
// ============================================================================

/**
 * 根据 panel pivot 更新全局行置换
 */
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
// 主接口：标准版本（兼容原接口）
// ============================================================================

/**
 * 标准 Half 精度分块 LU 分解
 * 
 * 保持与原接口兼容，使用默认配置
 */
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
    bool use_cublas_trsm = false,  // 改为 false，默认用自定义 kernel
    HgetrfTimers* timers = nullptr)
{
    if (!dA || !d_ipiv_rel || !h_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_blocked_half: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }
    if (ib_pref <= 0) {
        fprintf(stderr, "hgetrf_blocked_half: ib_pref must be positive (ib_pref=%d).\n", ib_pref);
        std::exit(EXIT_FAILURE);
    }

    // 初始化 pivot 行号
    for (int i = 0; i < m; ++i) {
        piv_rows[i] = i;
    }

    // 计算所有 panel 的 (j0, ib)
    std::vector<int> panel_j0;
    std::vector<int> panel_ib;
    {
        int j0 = 0;
        const int candidates[] = {256, 128, 64, 32};
        while (j0 < n) {
            int remaining = n - j0;
            int ib_cap = (ib_pref < remaining) ? ib_pref : remaining;
            int ib_now = 0;
            for (int cand : candidates) {
                if (cand <= ib_cap) { ib_now = cand; break; }
            }
            if (ib_now == 0) {
                fprintf(stderr,
                        "hgetrf_blocked_half: remaining cols (%d) smaller than minimum supported panel width (32).\n",
                        remaining);
                std::exit(EXIT_FAILURE);
            }
            panel_j0.push_back(j0);
            panel_ib.push_back(ib_now);
            j0 += ib_now;
        }
    }
    const int num_panels = (int)panel_j0.size();
    if (num_panels == 0) return;

    // Stream & Event
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

    // 工具函数
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
            if (use_cublas_trsm) {
                // 尝试使用 cuBLAS (需要检查是否支持 half)
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
                // Fallback to custom kernel
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
#endif
            } else {
                // 使用自定义 kernel
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
            }
        });
    };

    auto do_gemm_range = [&](int j0, int ib_now, int col0, int n2) {
        if (n2 <= 0 || ib_now <= 0) return;
        time_range(stream_update, timers ? &timers->gemm_ms : nullptr, [&]() {
            if (use_cublas_tc && handle) {
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
// 增强接口：使用配置结构体
// ============================================================================

/**
 * 增强版 Half 精度分块 LU 分解
 * 
 * 使用 HgetrfConfig 结构体进行配置
 */
inline void hgetrf_blocked_half_ex(
    half* dA,
    int m, int n, int lda,
    int ib_pref, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream = 0,
    HgetrfTimers* timers = nullptr)
{
    // 根据配置调用标准接口
    bool use_tc = config.use_tensor_core_gemm;
    bool use_cublas_trsm = (config.trsm_mode == HgetrfConfig::TRSM_CUBLAS_HALF);
    
    if (config.verbose) {
        printf("[hgetrf] Configuration:\n");
        printf("  Tensor Core GEMM: %s\n", use_tc ? "enabled" : "disabled");
        printf("  TRSM mode: ");
        switch (config.trsm_mode) {
            case HgetrfConfig::TRSM_CUSTOM_KERNEL:
                printf("custom kernel\n");
                break;
            case HgetrfConfig::TRSM_CUBLAS_HALF:
                printf("cuBLAS half\n");
                break;
            case HgetrfConfig::TRSM_CUBLAS_FLOAT:
                printf("cuBLAS float (not implemented)\n");
                break;
        }
        printf("  Batched exchange: %s\n", config.use_batched_exchange ? "enabled" : "disabled");
    }
    
    hgetrf_blocked_half(
        dA, m, n, lda,
        ib_pref, uc,
        handle,
        d_ipiv_rel,
        h_ipiv_rel,
        piv_rows,
        stream,
        use_tc,
        use_cublas_trsm,
        timers
    );
}