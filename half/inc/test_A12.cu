#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>

#include "A1_panel.cuh"
#include "A12_TRSM.cuh"

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

#ifndef CURAND_CHECK
#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t st__ = (call);                                          \
        if (st__ != CURAND_STATUS_SUCCESS) {                                   \
            fprintf(stderr, "CURAND error %s:%d (status=%d)\n",                \
                    __FILE__, __LINE__, (int)st__);                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

// float -> half
__global__ void float_to_half_kernel(
    const float* __restrict__ F,
    half* __restrict__ H,
    int m, int n, int ldF, int ldH)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= n || row >= m) return;

    float v = F[row + (size_t)col * ldF];
    H[row + (size_t)col * ldH] = __float2half(v);
}

// 把 panel pivot 应用到 trailing 列 (A12, A22)
__global__ void apply_pivots_trailing_kernel(
    half* __restrict__ A,
    int m, int n, int lda,
    int ib, int col0,
    const int* __restrict__ ipiv_rel)
{
    int j = col0 + blockIdx.x * blockDim.x + threadIdx.x; // trailing col index
    if (j >= n) return;

    for (int k = 0; k < ib; ++k) {
        int i1 = k;
        int p  = k + ipiv_rel[k];
        if (p == i1 || p < 0 || p >= m) continue;

        half tmp                = A[i1 + (size_t)j * lda];
        A[i1 + (size_t)j * lda] = A[p  + (size_t)j * lda];
        A[p  + (size_t)j * lda] = tmp;
    }
}

// 提取 A12_top: rows 0..ib-1, cols ib..ib+ntrail-1 => B(ib x ntrail)
__global__ void extract_A12_top_kernel(
    const half* __restrict__ A,
    int lda,
    half* __restrict__ B,
    int ib, int ntrail)
{
    int colB = blockIdx.x * blockDim.x + threadIdx.x; // 0..ntrail-1
    int row  = blockIdx.y * blockDim.y + threadIdx.y; // 0..ib-1

    if (colB >= ntrail || row >= ib) return;

    int jA = ib + colB;
    B[row + (size_t)colB * ib] = A[row + (size_t)jA * lda];
}

// 把 B 恢复到 A12_top
__global__ void restore_A12_from_B_kernel(
    half* __restrict__ A,
    int lda,
    const half* __restrict__ B,
    int ib, int ntrail)
{
    int colB = blockIdx.x * blockDim.x + threadIdx.x;
    int row  = blockIdx.y * blockDim.y + threadIdx.y;

    if (colB >= ntrail || row >= ib) return;

    int jA = ib + colB;
    A[row + (size_t)jA * lda] = B[row + (size_t)colB * ib];
}

int main(int argc, char** argv)
{
    int m       = 8192;  // 行数
    int ib      = 128;   // panel 宽度
    int ntrail  = 128;   // A12 列数
    int uc      = 8;     // panel 内核参数
    int iters   = 20;    // TRSM 迭代次数

    if (argc >= 2) m       = std::atoi(argv[1]);
    if (argc >= 3) ib      = std::atoi(argv[2]);
    if (argc >= 4) ntrail  = std::atoi(argv[3]);
    if (argc >= 5) uc      = std::atoi(argv[4]);
    if (argc >= 6) iters   = std::atoi(argv[5]);

    int n   = ib + ntrail; // 总列数
    int lda = m;

    printf("Testing panel_TSLU + A12_TRSM (all-half)\n");
    printf("  m = %d, ib = %d, ntrail = %d, n = %d, uc = %d, iters = %d\n",
           m, ib, ntrail, n, uc, iters);

    // ===== 1. 生成随机 half 矩阵 A (m x n) =====

    float* dF = nullptr;
    half*  dA = nullptr;

    CUDA_CHECK(cudaMalloc(&dF, sizeof(float) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half)  * (size_t)lda * n));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, 123456789ULL));
    CURAND_CHECK(curandGenerateNormal(gen, dF, (size_t)lda * n, 0.0f, 1.0f));

    dim3 blk(16, 16);
    dim3 grd((n   + blk.x - 1) / blk.x,
             (m   + blk.y - 1) / blk.y);

    float_to_half_kernel<<<grd, blk>>>(dF, dA, m, n, lda, lda);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dF));
    CURAND_CHECK(curandDestroyGenerator(gen));

    // 保存 panel 原始数据 (m x ib)
    half* dApanel_orig = nullptr;
    CUDA_CHECK(cudaMalloc(&dApanel_orig, sizeof(half) * (size_t)lda * ib));
    CUDA_CHECK(cudaMemcpy(dApanel_orig, dA,
                          sizeof(half) * (size_t)lda * ib,
                          cudaMemcpyDeviceToDevice));

    // ===== 2. panel_TSLU 计时 + factorization =====

    int* d_ipiv_rel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, sizeof(int) * ib));

    cudaEvent_t ev_p_start, ev_p_stop;
    CUDA_CHECK(cudaEventCreate(&ev_p_start));
    CUDA_CHECK(cudaEventCreate(&ev_p_stop));

    CUDA_CHECK(cudaEventRecord(ev_p_start));
    launch_panel_TSLU(dA, m, lda, 0, ib, uc, d_ipiv_rel);
    CUDA_CHECK(cudaEventRecord(ev_p_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_p_stop));

    float panel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&panel_ms, ev_p_start, ev_p_stop));

    // ===== 3. 把 pivot 应用到 trailing 列 =====

    int col0 = ib;
    dim3 blk_piv(128);
    dim3 grd_piv((n - col0 + blk_piv.x - 1) / blk_piv.x);

    apply_pivots_trailing_kernel<<<grd_piv, blk_piv>>>(
        dA, m, n, lda, ib, col0, d_ipiv_rel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===== 4. 提取 pivot 后的 A12_top 作为 B_orig =====

    half* dB_orig = nullptr; // ib x ntrail
    CUDA_CHECK(cudaMalloc(&dB_orig, sizeof(half) * (size_t)ib * ntrail));

    dim3 blk_ex(16, 16);
    dim3 grd_ex((ntrail + blk_ex.x - 1) / blk_ex.x,
                (ib      + blk_ex.y - 1) / blk_ex.y);

    extract_A12_top_kernel<<<grd_ex, blk_ex>>>(
        dA, lda, dB_orig, ib, ntrail);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===== 5. TRSM 性能测试 =====

    cudaEvent_t ev_t_start, ev_t_stop;
    CUDA_CHECK(cudaEventCreate(&ev_t_start));
    CUDA_CHECK(cudaEventCreate(&ev_t_stop));

    dim3 blk_rs(16, 16);
    dim3 grd_rs((ntrail + blk_rs.x - 1) / blk_rs.x,
                (ib      + blk_rs.y - 1) / blk_rs.y);

    // 预热一次
    restore_A12_from_B_kernel<<<grd_rs, blk_rs>>>(
        dA, lda, dB_orig, ib, ntrail);
    CUDA_CHECK(cudaGetLastError());
    launch_A12_trsm(dA, m, n, lda, 0, ib);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(ev_t_start));
    for (int it = 0; it < iters; ++it) {
        restore_A12_from_B_kernel<<<grd_rs, blk_rs>>>(
            dA, lda, dB_orig, ib, ntrail);
        CUDA_CHECK(cudaGetLastError());

        launch_A12_trsm(dA, m, n, lda, 0, ib);
    }
    CUDA_CHECK(cudaEventRecord(ev_t_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_t_stop));

    float trsm_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&trsm_ms, ev_t_start, ev_t_stop));
    trsm_ms /= iters;

    // TRSM FLOPs: per col ~ 0.5*ib*(ib-1)*2 => ib*(ib-1)
    double flops_trsm_per_col = (double)ib * (double)(ib - 1);
    double flops_trsm_total   = flops_trsm_per_col * (double)ntrail;
    double gflops_trsm        = (flops_trsm_total * 1e-9) / (trsm_ms * 1e-3);

    printf("\nTRSM performance (on LU panel L11, all-half):\n");
    printf("  Avg time per TRSM: %.4f ms\n", trsm_ms);
    printf("  Estimated FLOPs:   %.4f GFLOP\n", flops_trsm_total * 1e-9);
    printf("  Estimated perf:    %.4f GFLOP/s\n", gflops_trsm);

    // ===== 6. 精度检测：panel + TRSM =====

    // 先在 device 上做一次“干净”的 TRSM 解，用于精度
    restore_A12_from_B_kernel<<<grd_rs, blk_rs>>>(
        dA, lda, dB_orig, ib, ntrail);
    CUDA_CHECK(cudaGetLastError());
    launch_A12_trsm(dA, m, n, lda, 0, ib);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 拷回 host
    std::vector<half> hA((size_t)lda * n);
    std::vector<half> hB((size_t)ib  * ntrail);
    std::vector<half> hApanel_orig((size_t)lda * ib);
    std::vector<int>  h_ipiv(ib);

    CUDA_CHECK(cudaMemcpy(hA.data(), dA,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hB.data(), dB_orig,
                          sizeof(half) * (size_t)ib * ntrail,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hApanel_orig.data(), dApanel_orig,
                          sizeof(half) * (size_t)lda * ib,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ipiv.data(), d_ipiv_rel,
                          sizeof(int) * ib,
                          cudaMemcpyDeviceToHost));

    // ===== 6.1 panel 精度：P_panel*A_panel_orig vs L_panel*U_panel =====

    // Ap0: 原始 panel (m x ib, double)
    std::vector<double> Ap0((size_t)m * ib);
    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < m; ++i) {
            half hv = hApanel_orig[i + (size_t)j * lda];
            Ap0[i + (size_t)j * m] = (double)__half2float(hv);
        }
    }

    // 应用 pivot: P_panel * Ap0
    for (int k = 0; k < ib; ++k) {
        int p = k + h_ipiv[k];
        if (p == k || p < 0 || p >= m) continue;
        for (int j = 0; j < ib; ++j) {
            std::swap(Ap0[k + (size_t)j * m],
                      Ap0[p + (size_t)j * m]);
        }
    }

    // L_panel(m x ib), U_panel(ib x ib)
    std::vector<double> L_panel((size_t)m * ib);
    std::vector<double> U_panel((size_t)ib * ib);

    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < m; ++i) {
            double v;
            if (i < j) {
                v = 0.0;
            } else if (i == j) {
                v = 1.0;
            } else {
                half hv = hA[i + (size_t)j * lda]; // L11/L21 存在 A 里
                v = (double)__half2float(hv);
            }
            L_panel[i + (size_t)j * m] = v;
        }
    }

    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < ib; ++i) {
            double v;
            if (i > j) {
                v = 0.0;
            } else {
                half hv = hA[i + (size_t)j * lda];
                v = (double)__half2float(hv);
            }
            U_panel[i + (size_t)j * ib] = v;
        }
    }

    // A_fact_panel = L_panel * U_panel (m x ib)
    std::vector<double> A_fact_panel((size_t)m * ib);
    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int k = 0; k < ib; ++k) {
                sum += L_panel[i + (size_t)k * m] *
                       U_panel[k + (size_t)j * ib];
            }
            A_fact_panel[i + (size_t)j * m] = sum;
        }
    }

    // Frobenius norms for panel
    double normA_panel = 0.0;
    double normR_panel = 0.0;
    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < m; ++i) {
            double a   = Ap0[i + (size_t)j * m];
            double aff = A_fact_panel[i + (size_t)j * m];
            double r   = a - aff;
            normA_panel += a * a;
            normR_panel += r * r;
        }
    }
    normA_panel = std::sqrt(normA_panel);
    normR_panel = std::sqrt(normR_panel);
    double rel_panel = (normA_panel > 0.0) ? normR_panel / normA_panel : 0.0;

    printf("\nPanel accuracy (CPU double, m x ib panel):\n");
    printf("  ||A_panel||_F              = %.6f\n", normA_panel);
    printf("  ||P*A_panel - L_panel*U_panel||_F        = %.6f  (绝对误差)\n", normR_panel);
    printf("  ||P*A_panel - L_panel*U_panel||_F / ||A_panel||_F = %.6e  (相对误差)\n",
           rel_panel);

    // ===== 6.2 TRSM 精度：L11*X - B =====

    std::vector<double> L11((size_t)ib * ib);
    std::vector<double> Xd((size_t)ib * ntrail);
    std::vector<double> Bd((size_t)ib * ntrail);
    std::vector<double> Rd((size_t)ib * ntrail);

    // L11: unit-lower, from top-left ib x ib of A
    for (int j = 0; j < ib; ++j) {
        for (int i = 0; i < ib; ++i) {
            double v;
            if (i < j) {
                v = 0.0;
            } else if (i == j) {
                v = 1.0;
            } else {
                half hv = hA[i + (size_t)j * lda];
                v = (double)__half2float(hv);
            }
            L11[i + (size_t)j * ib] = v;
        }
    }

    // X: A12_top after TRSM
    for (int j = 0; j < ntrail; ++j) {
        int colA = ib + j;
        for (int i = 0; i < ib; ++i) {
            half hv = hA[i + (size_t)colA * lda];
            Xd[i + (size_t)j * ib] = (double)__half2float(hv);
        }
    }

    // B: B_orig
    for (int j = 0; j < ntrail; ++j) {
        for (int i = 0; i < ib; ++i) {
            half hv = hB[i + (size_t)j * ib];
            Bd[i + (size_t)j * ib] = (double)__half2float(hv);
        }
    }

    // R_trsm = L11*X - B
    for (int j = 0; j < ntrail; ++j) {
        for (int i = 0; i < ib; ++i) {
            double sum = 0.0;
            for (int k = 0; k < ib; ++k) {
                sum += L11[i + (size_t)k * ib] * Xd[k + (size_t)j * ib];
            }
            Rd[i + (size_t)j * ib] = sum - Bd[i + (size_t)j * ib];
        }
    }

    double normB = 0.0, normR_trsm = 0.0;
    for (size_t idx = 0; idx < (size_t)ib * ntrail; ++idx) {
        double vb = Bd[idx];
        double vr = Rd[idx];
        normB       += vb * vb;
        normR_trsm  += vr * vr;
    }
    normB      = std::sqrt(normB);
    normR_trsm = std::sqrt(normR_trsm);
    double rel_trsm = (normB > 0.0) ? normR_trsm / normB : 0.0;

    printf("\nTRSM accuracy (CPU double, L11*X - B):\n");
    printf("  ||B||_F              = %.6f\n", normB);
    printf("  ||L11*X - B||_F      = %.6f  (绝对误差)\n", normR_trsm);
    printf("  ||L11*X - B||_F / ||B||_F = %.6e  (相对误差)\n",
           rel_trsm);

    // ===== 7. 合并性能 + 粗略“前两步精度” =====

    // panel FLOPs 近似：LU(m x ib) ~ 2 m ib^2 - 2/3 ib^3
    double flops_panel = 2.0 * (double)m * (double)ib * (double)ib
                       - (2.0/3.0) * (double)ib * (double)ib * (double)ib;
    double gflops_panel = (flops_panel * 1e-9) / (panel_ms * 1e-3);

    double time_total_ms   = panel_ms + trsm_ms;
    double flops_total_all = flops_panel + flops_trsm_total;
    double gflops_all      = (flops_total_all * 1e-9) / (time_total_ms * 1e-3);

    double rel_combined = (rel_panel > rel_trsm) ? rel_panel : rel_trsm;

    printf("\nPanel performance:\n");
    printf("  Time(panel_TSLU): %.4f ms\n", panel_ms);
    printf("  FLOPs(panel_TSLU): %.4f GFLOP\n", flops_panel * 1e-9);
    printf("  Perf(panel_TSLU): %.4f GFLOP/s\n", gflops_panel);

    printf("\nCombined (panel_TSLU + TRSM) summary:\n");
    printf("  Time total: %.4f ms (panel %.4f + TRSM %.4f)\n",
           time_total_ms, panel_ms, trsm_ms);
    printf("  FLOPs total: %.4f GFLOP (panel %.4f + TRSM %.4f)\n",
           flops_total_all * 1e-9,
           flops_panel * 1e-9,
           flops_trsm_total * 1e-9);
    printf("  Perf total: %.4f GFLOP/s\n", gflops_all);
    printf("  Max relative error among {panel, TRSM}: %.6e\n", rel_combined);

    // 清理
    CUDA_CHECK(cudaEventDestroy(ev_p_start));
    CUDA_CHECK(cudaEventDestroy(ev_p_stop));
    CUDA_CHECK(cudaEventDestroy(ev_t_start));
    CUDA_CHECK(cudaEventDestroy(ev_t_stop));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dApanel_orig));
    CUDA_CHECK(cudaFree(dB_orig));
    CUDA_CHECK(cudaFree(d_ipiv_rel));

    return 0;
}
