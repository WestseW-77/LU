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

// 可选的计时结果（毫秒）
struct HgetrfTimers {
    double panel_ms    = 0.0;
    double exchange_ms = 0.0;
    double trsm_ms     = 0.0;
    double gemm_ms     = 0.0;
    double total_ms    = 0.0;
    int    panels      = 0;
};

/**
 * Helper: 根据 panel pivot 更新全局行置换 piv_rows（host 端数组）。
 * piv_rows[i] 表示 LU 完成后第 i 行来源的原始行号。
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

/**
 * 全半精度分块 LU 分解入口 (blocking getrf)。
 *
 * 功能：
 *   逐 panel 执行 panel_TSLU -> 行交换扩散 -> A12 TRSM -> A22 GEMM，
 *   把结果直接写回 dA，输出 L/U（单位下三角 + 上三角）混存于 dA。
 *
 * 约束/假设：
 *   - 当前 panel kernel仅支持 IB in {32,64,128,256}。函数会在这些里选取
 *     不超过 ib_pref 且不超过剩余列数的最大候选。
 *   - 默认假设矩阵列数 n >= 32，且按列主序、lda>=m。
 *   - h_ipiv_rel、piv_rows 为 host 指针，容量分别 >= ib_pref、>= m。
 *   - d_ipiv_rel 为 device int 数组，容量 >= ib_pref。
 *   - 如果 handle 为空或 use_cublas_tc=false，则 trailing update 走 naive CUDA 版本。
 *
 * 输入/输出：
 *   dA         : [in/out] 设备端 half 矩阵，因地 factor，结果混存 L/U。
 *   m, n, lda  : 行数、列数、leading dimension。
 *   ib_pref    : 希望使用的最大 panel 宽度（将自动按 32/64/128/256 选不超过它的最佳）。
 *   uc         : panel 内核微块宽度（将截断到当前 panel 宽度）。
 *   handle     : cuBLAS 句柄，用于 A22 GEMM（可为 nullptr）。
 *   d_ipiv_rel : 设备端 pivot 相对位移 (长度 ib)。
 *   h_ipiv_rel : host 端 pivot 缓冲 (长度 ib)。
 *   piv_rows   : host 端全局行置换 (长度 m)，函数内会写入最终置换。
 *   stream     : 可选 CUDA stream。
 *   use_cublas_tc : true 时用 Tensor Core GEMM，否则用 naive CUDA GEMM。
 *   use_cublas_trsm : true 时 A12 使用 cublasTrsmEx，false 走定制 kernel。
 */
// hgetrf.cuh 中的一个实现版本
// 只给出 hgetrf_blocked_half，其他不动

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
    bool use_cublas_trsm = true,
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

    // ---- 预先计算所有 panel 的 (j0_k, ib_k) ----
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

    // ---- stream & event ----
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

    // ---- 工具 lambda：panel / exchange / TRSM / GEMM ----

    auto do_panel = [&](int j0, int ib_now) {
        int uc_now = (uc > ib_now) ? ib_now : uc;
        time_range(stream_panel, timers ? &timers->panel_ms : nullptr, [&]() {
            launch_panel_TSLU(dA, m, lda, j0, ib_now, uc_now, d_ipiv_rel, stream_panel);
        });
        // 取回 pivot 位移
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

#if defined(CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP)
        if (use_cublas_trsm && handle) {
            time_range(stream_update, timers ? &timers->trsm_ms : nullptr, [&]() {
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
                        ib_now,  // m
                        ntrail,  // n
                        &alpha,
                        L11, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        A12, CUDA_R_16F, lda,
                        CUDA_R_32F,
                        CUBLAS_TRSM_ALGO_DEFAULT_TENSOR_OP));
            });
        } else
#endif
        {
            time_range(stream_update, timers ? &timers->trsm_ms : nullptr, [&]() {
                launch_A12_trsm(dA, m, n, lda, j0, ib_now, stream_update);
            });
        }
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

    // ---- Pipeline 调度：look-ahead=1 ----

    // 先做 panel(0)
    do_panel(panel_j0[0], panel_ib[0]);

    // 主循环：处理 panel 0..num_panels-2
    for (int k = 0; k < num_panels - 1; ++k) {
        int j0   = panel_j0[k];
        int ib_k = panel_ib[k];

        int j0_next   = panel_j0[k+1];
        int ib_next   = panel_ib[k+1];

        // 1) 先在 update stream 上做 exchange(k), TRSM(k)
        do_exchange(j0, ib_k);
        do_trsm(j0, ib_k);

        // 2) GEMM1(k)：只更新下一块 panel 所在列块
        {
            int col0_next = j0 + ib_k;  // 下一 panel 的起始列
            int n2_next   = ib_next;    // 下一 panel 的宽度
            do_gemm_range(j0, ib_k, col0_next, n2_next);

            // GEMM1 完成后，下一 panel 所需数据已就绪，记录事件
            CUDA_CHECK(cudaEventRecord(ev_next_ready, stream_update));
        }

        // 3) 在 panel stream 上等待 GEMM1 完成，然后做 panel(k+1)
        CUDA_CHECK(cudaStreamWaitEvent(stream_panel, ev_next_ready, 0));
        do_panel(j0_next, ib_next);

        // 4) 在 update stream 上继续 GEMM2(k)：更新剩余尾部列
        {
            int col0_tail = j0 + ib_k + ib_next;
            int n2_tail   = n - col0_tail;
            if (n2_tail > 0) {
                do_gemm_range(j0, ib_k, col0_tail, n2_tail);
            }
        }

        // 5) panel(k+1) 已经完成，GEMM2(k) 也完成，
        //    下一轮循环会对 panel(k+1) 做 exchange/TRSM。
    }

    // 处理最后一个 panel: num_panels-1
    {
        int k_last = num_panels - 1;
        int j0_last = panel_j0[k_last];
        int ib_last = panel_ib[k_last];

        // 最后一块 panel 已经在上一步 do_panel() 完成
        // 这里只需要做 exchange + TRSM + full GEMM
        do_exchange(j0_last, ib_last);
        do_trsm(j0_last, ib_last);
        do_gemm_full(j0_last, ib_last);
    }

    // ---- 收尾 ----
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
