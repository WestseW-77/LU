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

// ✅ 新增：device piv_rows 维护模块
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

// ============================================================================
// 配置结构体（清理后）
// ============================================================================

struct HgetrfConfig {
    bool use_tensor_core_gemm = true;
    
    enum TrsmMode {
        TRSM_CUSTOM_KERNEL,
        TRSM_CUBLAS_HALF,
        TRSM_CUBLAS_FLOAT
    };
    TrsmMode trsm_mode = TRSM_CUSTOM_KERNEL;

    int fixed_panel_width = 128;
    bool verbose = false;
};

// ============================================================================
// 计时器结构体
// ============================================================================

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
// 核心实现：双流流水线版本
// ============================================================================

inline void hgetrf_blocked_half_dualstream(
    half* dA,
    int m, int n, int lda,
    int panel_width, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    int* h_ipiv_rel,   // ✅ 不删，保留接口，但在性能路径中不再使用
    int* piv_rows,
    const HgetrfConfig& config,
    cudaStream_t stream,
    HgetrfTimers* timers)
{
    if (!dA || !d_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_dualstream: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    // ✅ host piv_rows 初始化仍保留（但最终结果会来自 device）
    for (int i = 0; i < m; ++i) {
        piv_rows[i] = i;
    }

    // 划分Panel
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
        printf("[hgetrf] Dual-stream mode, panel width: %d, panels: %d\n",
               panel_width, num_panels);
    }

    // 创建2个流
    cudaStream_t stream_update = stream;
    cudaStream_t stream_panel  = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_panel, cudaStreamNonBlocking));

    // 分配Panel工作空间
    const int num_blocks_pivot_max = panel_TSLU_required_pivot_blocks(m, 0);
    half* d_panel_block_val = nullptr;
    int*  d_panel_block_idx = nullptr;
    CUDA_CHECK(cudaMalloc(&d_panel_block_val, sizeof(half) * (size_t)num_blocks_pivot_max));
    CUDA_CHECK(cudaMalloc(&d_panel_block_idx, sizeof(int)  * (size_t)num_blocks_pivot_max));

    // ✅ 新增：device piv_rows
    int* d_piv_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_piv_rows, sizeof(int) * (size_t)m));
    launch_init_piv_rows(d_piv_rows, m, stream_update);

    // 创建事件
    cudaEvent_t ev0 = nullptr, ev1 = nullptr;
    cudaEvent_t ev_total0 = nullptr, ev_total1 = nullptr;
    cudaEvent_t ev_next_ready = nullptr;

    // ✅ 新增：pivot ready 事件（panel 完成 pivot + d_piv_rows 更新）
    cudaEvent_t ev_piv_ready = nullptr;
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_piv_ready, cudaEventDisableTiming));

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

    // 计时辅助函数
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

    // 定义各阶段操作
    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;

        time_range(stream_panel, timers ? &timers->panel_ms : nullptr, [&]() {
            launch_panel_TSLU(dA, m, lda, j0, ib_now, uc_now, d_ipiv_rel, stream_panel,
                             d_panel_block_val, d_panel_block_idx, num_blocks_pivot_max);
        });

        // ✅ 方向B核心：不再 D2H pivot，不再 host sync
        // 直接在 device 上更新 d_piv_rows
        launch_apply_panel_pivots_to_pivrows(
            d_piv_rows, m, j0, ib_now, d_ipiv_rel, stream_panel);

        // ✅ pivot + piv_rows ready
        CUDA_CHECK(cudaEventRecord(ev_piv_ready, stream_panel));

        if (timers) timers->panels += 1;
    };

    auto do_exchange = [&](int j0, int ib_now) {
        // ✅ 关键：exchange 必须等 pivot ready
        CUDA_CHECK(cudaStreamWaitEvent(stream_update, ev_piv_ready, 0));
        time_range(stream_update, timers ? &timers->exchange_ms : nullptr, [&]() {
            launch_A_exchange_trailing_device_piv(
                dA, m, n, lda, j0, ib_now, d_ipiv_rel, stream_update);
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

    // 流水线执行
    do_panel(panel_j0[0], panel_ib[0]);

    for (int k = 0; k < num_panels - 1; ++k) {
        int j0   = panel_j0[k];
        int ib_k = panel_ib[k];

        int j0_next   = panel_j0[k+1];
        int ib_next   = panel_ib[k+1];

        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);

        // 只更新下一个panel需要的部分
        {
            int col0_next = j0 + ib_k;
            int n2_next   = ib_next;
            do_gemm_range(j0, ib_k, col0_next, n2_next);
            CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_update));
        }

        // Panel与剩余GEMM并行
        CUDA_CHECK(cudaStreamWaitEvent(stream_panel, ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        // 更新剩余列
        {
            int col0_tail = j0 + ib_k + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) {
                do_gemm_range(j0, ib_k, col0_tail, n2_tail);
            }
        }
    }

    // 最后一个panel
    {
        int k_last = num_panels - 1;
        int j0_last = panel_j0[k_last];
        int ib_last = panel_ib[k_last];

        do_exchange(j0_last, ib_last);
        do_trsm(j0_last, ib_last);
        do_gemm_full(j0_last, ib_last);
    }

    // 同步
    CUDA_CHECK(cudaStreamSynchronize(stream_update));
    CUDA_CHECK(cudaStreamSynchronize(stream_panel));

    // ✅ 最后一次性把 device piv_rows 拷回 host piv_rows
    CUDA_CHECK(cudaMemcpyAsync(piv_rows, d_piv_rows,
                               sizeof(int) * (size_t)m,
                               cudaMemcpyDeviceToHost,
                               stream_update));
    CUDA_CHECK(cudaStreamSynchronize(stream_update));

    // 统计时间
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

    CUDA_CHECK(cudaEventDestroy(ev_piv_ready));

    // 清理
    CUDA_CHECK(cudaFree(d_piv_rows));
    CUDA_CHECK(cudaFree(d_panel_block_val));
    CUDA_CHECK(cudaFree(d_panel_block_idx));
    CUDA_CHECK(cudaStreamDestroy(stream_panel));
}

// ============================================================================
// 主接口
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
    if (!dA || !d_ipiv_rel || !piv_rows) {
        fprintf(stderr, "hgetrf_auto: null pointer input.\n");
        std::exit(EXIT_FAILURE);
    }

    // 确定panel宽度
    int panel_width = config.fixed_panel_width;
    if (panel_width <= 0 || panel_width > MAX_PANEL_WIDTH) {
        panel_width = DEFAULT_PANEL_WIDTH;
    }
    if (ib_request > 0 && ib_request < panel_width) {
        panel_width = ib_request;
    }

    // 如果矩阵很小（只有一个panel），直接分解（保留原逻辑）
    if (n <= panel_width) {
        if (config.verbose) {
            printf("[hgetrf_auto] Single panel mode (n=%d)\n", n);
        }

        for (int i = 0; i < m; ++i) {
            piv_rows[i] = i;
        }

        const int num_blocks_pivot_max = panel_TSLU_required_pivot_blocks(m, 0);
        half* d_panel_block_val = nullptr;
        int*  d_panel_block_idx = nullptr;
        CUDA_CHECK(cudaMalloc(&d_panel_block_val, sizeof(half) * (size_t)num_blocks_pivot_max));
        CUDA_CHECK(cudaMalloc(&d_panel_block_idx, sizeof(int)  * (size_t)num_blocks_pivot_max));

        cudaEvent_t ev0 = nullptr, ev1 = nullptr;
        if (timers) {
            CUDA_CHECK(cudaEventCreate(&ev0));
            CUDA_CHECK(cudaEventCreate(&ev1));
            CUDA_CHECK(cudaEventRecord(ev0, stream));
        }

        launch_panel_TSLU(dA, m, lda, 0, n, uc, d_ipiv_rel, stream,
                         d_panel_block_val, d_panel_block_idx, num_blocks_pivot_max);

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

        // 单 panel 模式依然走 host piv update（保持不动，简单正确）
        CUDA_CHECK(cudaMemcpyAsync(h_ipiv_rel, d_ipiv_rel,
                                   sizeof(int) * n,
                                   cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // host update piv_rows
        for (int k = 0; k < n; ++k) {
            int r1 = k;
            int r2 = r1 + h_ipiv_rel[k];
            if ((unsigned)r1 >= (unsigned)m || (unsigned)r2 >= (unsigned)m) continue;
            if (r1 == r2) continue;
            int tmp = piv_rows[r1];
            piv_rows[r1] = piv_rows[r2];
            piv_rows[r2] = tmp;
        }

        CUDA_CHECK(cudaFree(d_panel_block_val));
        CUDA_CHECK(cudaFree(d_panel_block_idx));
        return;
    }

    // 多panel情况：使用双流流水线
    if (config.verbose) {
        printf("[hgetrf_auto] Dual-stream mode (n=%d, panel_width=%d)\n", n, panel_width);
    }

    hgetrf_blocked_half_dualstream(dA, m, n, lda,
                                   panel_width, uc,
                                   handle,
                                   d_ipiv_rel,
                                   h_ipiv_rel,
                                   piv_rows,
                                   config,
                                   stream,
                                   timers);
}
