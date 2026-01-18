#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <string>
#include <algorithm>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// 新的 cuSOLVER-like API
#include "hgetrf.cuh"

#include "A1_panel.cuh"
#include "A12_TRSM.cuh"
#include "A22_GEMM.cuh"
#include "A_exchange.cuh"

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

// ============================================================================
// 测试配置
// ============================================================================

struct TestConfig {
    int m       = 16384;
    int n       = 16384;
    int uc      = 16;
    int iters   = 10;
    int warmup  = 2;
    bool verbose = false;
};

// ============================================================================
// 辅助函数
// ============================================================================

void generate_random_A_float(std::vector<float>& hA, int m, int n, int lda, unsigned seed = 77777)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            hA[i + (size_t)j * lda] = dist(rng);
        }
    }
}

void convert_float_to_half(const std::vector<float>& hAf,
                           std::vector<half>& hAh,
                           int m, int n, int lda)
{
    hAh.resize((size_t)lda * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            size_t idx = i + (size_t)j * lda;
            hAh[idx] = __float2half(hAf[idx]);
        }
    }
}

__global__ void build_LU_half_kernel(const half* __restrict__ LU,
                                     half* __restrict__ L,
                                     half* __restrict__ U,
                                     int m, int n, int lda)
{
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= n || i >= m) return;

    half v = LU[i + (size_t)j * lda];

    if (i > j) {
        L[i + (size_t)j * lda] = v;
        U[i + (size_t)j * lda] = __float2half(0.0f);
    } else if (i == j) {
        L[i + (size_t)j * lda] = __float2half(1.0f);
        U[i + (size_t)j * lda] = v;
    } else {
        L[i + (size_t)j * lda] = __float2half(0.0f);
        U[i + (size_t)j * lda] = v;
    }
}

__global__ void build_PA_half_kernel(const half* __restrict__ A0,
                                     const int* __restrict__ piv_rows,
                                     half* __restrict__ PA,
                                     int m, int n, int lda)
{
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= n || i >= m)
        return;

    int src = piv_rows[i];
    if (src < 0 || src >= m)
        return;

    half v = A0[src + (size_t)j * lda];
    PA[i + (size_t)j * lda] = v;
}

__global__ void residual_half_kernel(const half* __restrict__ PA,
                                     const half* __restrict__ LU,
                                     half* __restrict__ R,
                                     int m, int n, int lda)
{
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= n || i >= m) return;

    size_t idx = i + (size_t)j * lda;
    half pa = PA[idx];
    half lu = LU[idx];
    R[idx] = __hsub(pa, lu);
}

double frob_norm_half_host(const std::vector<half>& H, int m, int n, int lda)
{
    long double s = 0.0L;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            half  hv = H[i + (size_t)j * lda];
            float fv = __half2float(hv);
            long double dv = (long double)fv;
            s += dv * dv;
        }
    }
    return std::sqrt((double)s);
}

// ============================================================================
// 从 ipiv (1-based, length k=min(m,n)) 构建 piv_rows（长度 m）
// piv_rows[r] = 原始哪一行被换到 r
// 这只用于 accuracy，不进入 hgetrf 热路径
// ============================================================================
__global__ void init_piv_rows_kernel(int* piv_rows, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m) piv_rows[i] = i;
}

__global__ void build_pivrows_from_ipiv_kernel(
    int* piv_rows,
    int m,
    int k,
    const int* __restrict__ ipiv) // 1-based
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    for (int i = 0; i < k; ++i) {
        int r1 = i;
        int r2 = ipiv[i] - 1; // to 0-based

        if (r1 == r2) continue;
        if ((unsigned)r1 >= (unsigned)m || (unsigned)r2 >= (unsigned)m) continue;

        int tmp = piv_rows[r1];
        piv_rows[r1] = piv_rows[r2];
        piv_rows[r2] = tmp;
    }
}

inline void launch_build_pivrows_from_ipiv(
    int* d_piv_rows,
    int m,
    int k,
    const int* d_ipiv,
    cudaStream_t stream = 0)
{
    int threads = 256;
    int blocks = (m + threads - 1) / threads;
    init_piv_rows_kernel<<<blocks, threads, 0, stream>>>(d_piv_rows, m);
    CUDA_CHECK(cudaGetLastError());

    build_pivrows_from_ipiv_kernel<<<1, 1, 0, stream>>>(d_piv_rows, m, k, d_ipiv);
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// 测试函数
// ============================================================================

struct TestResult {
    double avg_time_ms;
    double gflops;
    double normA;
    double normRes;
    double relErr;
};

TestResult test_matrix_lu(
    const std::vector<half>& hA0_half,
    int m, int n, int lda,
    int uc,
    int iters, int warmup,
    cublasHandle_t cublas,
    bool verbose)
{
    TestResult result{};

    half* dA_half  = nullptr;
    half* dA0_half = nullptr;

    CUDA_CHECK(cudaMalloc(&dA_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA0_half, sizeof(half) * (size_t)lda * n));

    CUDA_CHECK(cudaMemcpy(dA0_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    // ===== cuSOLVER-like handle + workspace =====
    hgetrfHandle_t h = nullptr;
    hgetrfCreate(&h);
    hgetrfSetStream(h, 0);

    const int panel_width = 128;
    const int k_total = std::min(m, n);
    hgetrfSetPanelWidth(h, panel_width);
    hgetrfSetUc(h, uc);

    // ipiv (global, 1-based) + info(device scalar)
    int* d_ipiv = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int) * (size_t)k_total));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

    int lwork = 0;
    hgetrf_bufferSize(h, m, n, dA_half, lda, &lwork);
    half* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, sizeof(half) * (size_t)lwork));

    printf("\n========================================\n");
    printf("Half 精度 LU 分解测试 (cuSOLVER-like)\n");
    printf("========================================\n");
    printf("矩阵规模: %d × %d\n", m, n);
    printf("panel宽度: %d\n", panel_width);
    printf("微块大小: %d\n", uc);
    printf("迭代次数: %d (warmup: %d)\n", iters, warmup);
    printf("workspace: %d half elements (%zu bytes)\n", lwork, sizeof(half) * (size_t)lwork);
    printf("========================================\n");

    // Warmup (copy 不计时)
    if (warmup > 0) {
        printf("\n[Warmup] %d iterations...\n", warmup);
        for (int w = 0; w < warmup; ++w) {
            CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemsetAsync(d_info, 0, sizeof(int), 0));

            hgetrf(h, m, n, dA_half, lda,
                   d_work,
                   d_ipiv, d_info);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Warmup] Complete!\n");
    }

    // Performance test：只计 hgetrf，不计 memcpy
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    printf("\n[Performance] Running %d iterations...\n", iters);

    float total_ms = 0.0f;

    for (int it = 0; it < iters; ++it) {
        // 这次 memcpy 不计时（但仍然执行，保证每次输入一致）
        CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                              sizeof(half) * (size_t)lda * n,
                              cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemsetAsync(d_info, 0, sizeof(int), 0));

        CUDA_CHECK(cudaEventRecord(ev_start));
        hgetrf(h, m, n, dA_half, lda,
               d_work,
               d_ipiv, d_info, true);
        CUDA_CHECK(cudaEventRecord(ev_stop));
        CUDA_CHECK(cudaEventSynchronize(ev_stop));

        float iter_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&iter_ms, ev_start, ev_stop));
        total_ms += iter_ms;
    }

    result.avg_time_ms = total_ms / iters;

    // FLOPs（方阵公式近似仍然沿用你的）
    double total_flops = (1.0) * m * n * n - (1.0 / 3.0) * n * n * n;
    result.gflops = (total_flops * 1e-9) / (result.avg_time_ms * 1e-3);

    printf("[Performance] Complete!\n");
    printf("\n性能指标 (仅 LU 内核，不含 memcpy):\n");
    printf("  平均时间: %.4f ms\n", result.avg_time_ms);
    printf("  性能:     %.2f GFLOP/s\n", result.gflops);

    // Profiling
    printf("\n[Profiling] (本版本不在 hgetrf 内部做分段 timers；建议用 Nsight)\n");
    (void)verbose;

    // Read info
    int h_info = 0;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info != 0) {
        printf("\n[Warning] LU 分解检测到奇异主元：info = %d (1-based)\n", h_info);
    }

    // Accuracy test - 使用最后一次迭代的 dA_half + d_ipiv
    printf("\n[Accuracy] 验证分解正确性...\n");

    half* dL_half  = nullptr;
    half* dU_half  = nullptr;
    half* dPA_half = nullptr;
    half* dLU_half = nullptr;
    half* dR_half  = nullptr;

    CUDA_CHECK(cudaMalloc(&dL_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dU_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dPA_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dLU_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dR_half,  sizeof(half) * (size_t)lda * n));

    // 构建 L 和 U
    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        build_LU_half_kernel<<<grid, block>>>(dA_half, dL_half, dU_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // 从 ipiv 构造 piv_rows（仅用于验证）
    int* d_piv_rows = nullptr;
    CUDA_CHECK(cudaMalloc(&d_piv_rows, sizeof(int) * (size_t)m));
    launch_build_pivrows_from_ipiv(d_piv_rows, m, k_total, d_ipiv, 0);

    // 构建 PA：用 piv_rows
    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        build_PA_half_kernel<<<grid, block>>>(dA0_half, d_piv_rows, dPA_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // LU = L * U
    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        CUBLAS_CHECK(cublasSetStream(cublas, 0));
        CUBLAS_CHECK(
            cublasGemmEx(cublas,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         m, n, n,
                         &alpha,
                         dL_half, CUDA_R_16F, lda,
                         dU_half, CUDA_R_16F, lda,
                         &beta,
                         dLU_half, CUDA_R_16F, lda,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT));
    }

    // 残差 R = PA - LU
    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        residual_half_kernel<<<grid, block>>>(dPA_half, dLU_half, dR_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 计算范数
    std::vector<half> hR_half((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hR_half.data(), dR_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    std::vector<half> hA0_copy((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hA0_copy.data(), dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    result.normA   = frob_norm_half_host(hA0_copy, m, n, lda);
    result.normRes = frob_norm_half_host(hR_half, m, n, lda);
    result.relErr  = (result.normA > 0.0) ? (result.normRes / result.normA) : 0.0;

    printf("[Accuracy] Complete!\n");
    printf("\n精度分析:\n");
    printf("  ||A||_F            = %.6f\n", result.normA);
    printf("  ||PA - L*U||_F     = %.6f\n", result.normRes);
    printf("  相对误差           = %.6e\n", result.relErr);

    if (result.relErr < 1e-3) {
        printf("  ✓ 精度优秀 (< 1e-3)\n");
    } else if (result.relErr < 1e-2) {
        printf("  ✓ 精度良好 (< 1e-2)\n");
    } else if (result.relErr < 0.1) {
        printf("  ○ 精度可接受 (< 0.1)\n");
    } else {
        printf("  ✗ 精度较差 (>= 0.1)\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    CUDA_CHECK(cudaFree(dL_half));
    CUDA_CHECK(cudaFree(dU_half));
    CUDA_CHECK(cudaFree(dPA_half));
    CUDA_CHECK(cudaFree(dLU_half));
    CUDA_CHECK(cudaFree(dR_half));

    CUDA_CHECK(cudaFree(d_piv_rows));

    CUDA_CHECK(cudaFree(dA_half));
    CUDA_CHECK(cudaFree(dA0_half));

    CUDA_CHECK(cudaFree(d_ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    hgetrfDestroy(h);

    return result;
}

// ============================================================================
// 命令行参数解析
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  -m <M>          矩阵行数 (default: 16384)\n");
    printf("  -n <N>          矩阵列数 (default: 16384)\n");
    printf("  -uc <UC>        微块大小 (default: 8)\n");
    printf("  -iters <N>      迭代次数 (default: 10)\n");
    printf("  -warmup <N>     预热次数 (default: 2)\n");
    printf("  -verbose        详细输出\n");
    printf("  -h, --help      显示帮助\n");
}

TestConfig parse_args(int argc, char** argv) {
    TestConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "-m" && i + 1 < argc) {
            config.m = std::atoi(argv[++i]);
        }
        else if (arg == "-n" && i + 1 < argc) {
            config.n = std::atoi(argv[++i]);
        }
        else if (arg == "-uc" && i + 1 < argc) {
            config.uc = std::atoi(argv[++i]);
        }
        else if (arg == "-iters" && i + 1 < argc) {
            config.iters = std::atoi(argv[++i]);
        }
        else if (arg == "-warmup" && i + 1 < argc) {
            config.warmup = std::atoi(argv[++i]);
        }
        else if (arg == "-verbose") {
            config.verbose = true;
        }
    }

    return config;
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv)
{
    TestConfig cfg = parse_args(argc, argv);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    int m = cfg.m;
    int n = cfg.n;
    int lda = m;

    std::vector<float> hA0_float((size_t)lda * n);
    generate_random_A_float(hA0_float, m, n, lda);

    std::vector<half> hA0_half;
    convert_float_to_half(hA0_float, hA0_half, m, n, lda);

    TestResult result = test_matrix_lu(
        hA0_half, m, n, lda, cfg.uc,
        cfg.iters, cfg.warmup,
        handle, cfg.verbose);
    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\n========================================\n");
    printf("测试完成!\n");
    printf("========================================\n");

    return 0;
}
