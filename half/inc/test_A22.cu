#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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
            fprintf(stderr, "cuBLAS error %s:%d: status = %d\n",               \
                    __FILE__, __LINE__, (int)st__);                            \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

using half = __half;

// 把 host double 转成 host half（col-major）
static void pack_blocks_into_A(
    const std::vector<double>& A21_d, // m2 x ib, ldA21 = m2
    const std::vector<double>& A12_d, // ib x n2, ldA12 = ib
    const std::vector<double>& A22_d, // m2 x n2, ldA22 = m2
    int m, int n, int lda,
    int j0, int ib,
    std::vector<half>& hA)
{
    int row0 = j0 + ib;
    int col0 = j0 + ib;
    int m2   = m - row0;
    int n2   = n - col0;

    int ldA21 = m2;
    int ldA12 = ib;
    int ldA22 = m2;

    // 全部先置 0
    std::fill(hA.begin(), hA.end(), __float2half(0.0f));

    // A21 塞到 A(row0:row0+m2-1, j0:j0+ib-1)
    for (int j = 0; j < ib; ++j) {
        for (int i2 = 0; i2 < m2; ++i2) {
            int i = row0 + i2;
            double v = A21_d[i2 + (size_t)j * ldA21];
            hA[i + (size_t)(j0 + j) * lda] = __float2half((float)v);
        }
    }

    // A12 塞到 A(0:ib-1, col0:col0+n2-1)
    for (int j2 = 0; j2 < n2; ++j2) {
        int j = col0 + j2;
        for (int k = 0; k < ib; ++k) {
            double v = A12_d[k + (size_t)j2 * ldA12];
            hA[k + (size_t)j * lda] = __float2half((float)v);
        }
    }

    // A22 塞到 A(row0:row0+m2-1, col0:col0+n2-1)
    for (int j2 = 0; j2 < n2; ++j2) {
        int j = col0 + j2;
        for (int i2 = 0; i2 < m2; ++i2) {
            int i = row0 + i2;
            double v = A22_d[i2 + (size_t)j2 * ldA22];
            hA[i + (size_t)j * lda] = __float2half((float)v);
        }
    }
}

int main(int argc, char** argv)
{
    int m      = 8192; // 总行数
    int ib     = 128;  // panel 宽度 (K 维度)
    int ntrail = 128;  // trailing 列数 (N2)
    int iters  = 20;   // 每个版本重复次数

    if (argc >= 2) m      = std::atoi(argv[1]);
    if (argc >= 3) ib     = std::atoi(argv[2]);
    if (argc >= 4) ntrail = std::atoi(argv[3]);
    if (argc >= 5) iters  = std::atoi(argv[4]);

    int j0  = 0;              // panel 起始列
    int n   = ib + ntrail;    // 总列数
    int lda = m;

    int row0 = j0 + ib;
    int col0 = j0 + ib;
    int m2   = m - row0;      // A22 行
    int n2   = ntrail;        // A22 列

    if (m2 <= 0 || n2 <= 0 || ib <= 0) {
        printf("Invalid sizes: m2 <= 0 or n2 <= 0 or ib <= 0\n");
        return 0;
    }

    printf("Testing A22 GEMM (half-based, trailing update A22 <- A22 - A21*A12)\n");
    printf("  m = %d, ib = %d, ntrail(n2) = %d, n = %d, m2 = %d, iters = %d\n",
           m, ib, ntrail, n, m2, iters);

    // ===== 1. 在 CPU double 上生成 A21, A12, A22_orig =====
    std::mt19937 rng(1234567);
    std::normal_distribution<double> dist(0.0, 1.0);

    int ldA21 = m2;
    int ldA12 = ib;
    int ldA22 = m2;

    std::vector<double> A21_d((size_t)m2 * ib);
    std::vector<double> A12_d((size_t)ib * n2);
    std::vector<double> A22_orig_d((size_t)m2 * n2);

    // 生成 A21
    for (int j = 0; j < ib; ++j) {
        for (int i2 = 0; i2 < m2; ++i2) {
            A21_d[i2 + (size_t)j * ldA21] = dist(rng);
        }
    }

    // 生成 A12
    for (int j2 = 0; j2 < n2; ++j2) {
        for (int k = 0; k < ib; ++k) {
            A12_d[k + (size_t)j2 * ldA12] = dist(rng);
        }
    }

    // 生成 A22_orig
    for (int j2 = 0; j2 < n2; ++j2) {
        for (int i2 = 0; i2 < m2; ++i2) {
            A22_orig_d[i2 + (size_t)j2 * ldA22] = dist(rng);
        }
    }

    // ===== 2. 把这三个块打包到一个大矩阵 A (half, col-major) =====
    std::vector<half> hA_base((size_t)lda * n);
    pack_blocks_into_A(A21_d, A12_d, A22_orig_d,
                       m, n, lda, j0, ib, hA_base);

    // ===== 3. 分别为 naive 和 TC 版本准备 device 矩阵 =====
    half* dA_naive = nullptr;
    half* dA_tc    = nullptr;

    CUDA_CHECK(cudaMalloc(&dA_naive, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA_tc,    sizeof(half) * (size_t)lda * n));

    CUDA_CHECK(cudaMemcpy(dA_naive, hA_base.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA_tc,    hA_base.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    // ===== 4. 性能测试：naive 版本 =====
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 预热
    launch_A22_gemm_naive(dA_naive, m, n, lda, j0, ib);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dA_naive, hA_base.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int it = 0; it < iters; ++it) {
        // 每次都从相同初始状态开始
        if (it > 0) {
            CUDA_CHECK(cudaMemcpy(dA_naive, hA_base.data(),
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyHostToDevice));
        }
        launch_A22_gemm_naive(dA_naive, m, n, lda, j0, ib);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float naive_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&naive_ms, ev_start, ev_stop));
    naive_ms /= iters;

    // FLOPs ~ 2 * m2 * n2 * ib
    double flops_naive_total = 2.0 * (double)m2 * (double)n2 * (double)ib;
    double gflops_naive      = (flops_naive_total * 1e-9) / (naive_ms * 1e-3);

    printf("\nA22 GEMM naive (half->float) performance:\n");
    printf("  Avg time per update: %.4f ms\n", naive_ms);
    printf("  Estimated FLOPs:      %.4f GFLOP\n", flops_naive_total * 1e-9);
    printf("  Estimated perf:       %.4f GFLOP/s\n", gflops_naive);

    // ===== 5. 性能测试：cuBLAS Tensor Core 版本 =====
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 可选：强制 Tensor Core
    // CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 预热
    launch_A22_gemm_tc(dA_tc, m, n, lda, j0, ib, handle);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(dA_tc, hA_base.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            CUDA_CHECK(cudaMemcpy(dA_tc, hA_base.data(),
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyHostToDevice));
        }
        launch_A22_gemm_tc(dA_tc, m, n, lda, j0, ib, handle);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float tc_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&tc_ms, ev_start, ev_stop));
    tc_ms /= iters;

    double flops_tc_total = flops_naive_total; // 同一个 GEMM
    double gflops_tc      = (flops_tc_total * 1e-9) / (tc_ms * 1e-3);

    printf("\nA22 GEMM cuBLAS TensorCore performance:\n");
    printf("  Avg time per update: %.4f ms\n", tc_ms);
    printf("  Estimated FLOPs:      %.4f GFLOP\n", flops_tc_total * 1e-9);
    printf("  Estimated perf:       %.4f GFLOP/s\n", gflops_tc);

    // ===== 6. 精度检查：CPU double 上做 C_ref = A22_orig - A21*A12 =====
    std::vector<double> C_ref((size_t)m2 * n2);
    std::vector<double> C_naive((size_t)m2 * n2);
    std::vector<double> C_tc((size_t)m2 * n2);

    // 计算 C_ref
    for (int j2 = 0; j2 < n2; ++j2) {
        for (int i2 = 0; i2 < m2; ++i2) {
            double sum = 0.0;
            for (int k = 0; k < ib; ++k) {
                double a = A21_d[i2 + (size_t)k  * ldA21];
                double b = A12_d[k  + (size_t)j2 * ldA12];
                sum += a * b;
            }
            double c0 = A22_orig_d[i2 + (size_t)j2 * ldA22];
            C_ref[i2 + (size_t)j2 * m2] = c0 - sum;
        }
    }

    // 从 device 取回 naive 结果的 A22 区域
    std::vector<half> hA_naive((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hA_naive.data(), dA_naive,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));
    for (int j2 = 0; j2 < n2; ++j2) {
        int j = col0 + j2;
        for (int i2 = 0; i2 < m2; ++i2) {
            int i = row0 + i2;
            half hv = hA_naive[i + (size_t)j * lda];
            C_naive[i2 + (size_t)j2 * m2] = (double)__half2float(hv);
        }
    }

    // 从 device 取回 TC 结果的 A22 区域
    std::vector<half> hA_tc((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hA_tc.data(), dA_tc,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));
    for (int j2 = 0; j2 < n2; ++j2) {
        int j = col0 + j2;
        for (int i2 = 0; i2 < m2; ++i2) {
            int i = row0 + i2;
            half hv = hA_tc[i + (size_t)j * lda];
            C_tc[i2 + (size_t)j2 * m2] = (double)__half2float(hv);
        }
    }

    // 计算 Frobenius norm: ||C_ref||, ||C_naive - C_ref||, ||C_tc - C_ref||
    auto frob_norm = [](const std::vector<double>& M) {
        long long n = (long long)M.size();
        long double s = 0.0L;
        for (long long i = 0; i < n; ++i) {
            long double v = (long double)M[i];
            s += v * v;
        }
        return std::sqrt((double)s);
    };

    std::vector<double> diff_naive((size_t)m2 * n2);
    std::vector<double> diff_tc((size_t)m2 * n2);

    for (size_t i = 0; i < (size_t)m2 * n2; ++i) {
        diff_naive[i] = C_naive[i] - C_ref[i];
        diff_tc[i]    = C_tc[i]    - C_ref[i];
    }

    double norm_ref       = frob_norm(C_ref);
    double norm_diff_naiv = frob_norm(diff_naive);
    double norm_diff_tc   = frob_norm(diff_tc);

    printf("\nA22 GEMM accuracy (CPU double reference):\n");
    printf("  ||C_ref||_F                 = %.6f\n", norm_ref);
    printf("  Naive: ||C_naive - C_ref||_F        = %.6f  (绝对误差)\n",
           norm_diff_naiv);
    printf("         ||C_naive - C_ref||_F / ||C_ref||_F = %.6e  (相对误差)\n",
           norm_ref > 0.0 ? norm_diff_naiv / norm_ref : 0.0);
    printf("  TC:    ||C_tc - C_ref||_F            = %.6f  (绝对误差)\n",
           norm_diff_tc);
    printf("         ||C_tc - C_ref||_F / ||C_ref||_F     = %.6e  (相对误差)\n",
           norm_ref > 0.0 ? norm_diff_tc / norm_ref : 0.0);

    // ===== 清理 =====
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(dA_naive));
    CUDA_CHECK(cudaFree(dA_tc));

    return 0;
}
