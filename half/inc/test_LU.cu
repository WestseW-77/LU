#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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
    int n          = 2048;
    int ib         = 128;
    int uc         = 8;
    int iters      = 10;
    int warmup     = 2;
    
    // 测试模式
    enum TestMode {
        MODE_FULL_LU,      // 完整 LU 分解
        MODE_PANEL_ONLY,   // 仅测试 panel TSLU (模拟真实调用)
        MODE_BOTH          // 两种都测试
    };
    TestMode mode = MODE_FULL_LU;
    
    bool test_tc_on    = true;
    bool test_tc_off   = false;
    bool test_trsm_custom = true;
    bool test_trsm_cublas = false;
    
    bool verify_accuracy = true;
    bool verbose         = false;
};

// ============================================================================
// 辅助函数
// ============================================================================

void generate_random_A_float(std::vector<float>& hA, int m, int n, unsigned seed = 1234567)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    int lda = m;
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            hA[i + (size_t)j * lda] = dist(rng);
        }
    }
}

void convert_float_to_half(const std::vector<float>& hAf,
                           std::vector<half>& hAh,
                           int m, int n)
{
    size_t N = (size_t)m * n;
    hAh.resize(N);
    for (size_t idx = 0; idx < N; ++idx) {
        hAh[idx] = __float2half(hAf[idx]);
    }
}

__global__ void build_LU_half_kernel(const half* __restrict__ LU,
                                     half* __restrict__ L,
                                     half* __restrict__ U,
                                     int n, int lda)
{
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= n || i >= n) return;

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
    if (j >= n || i >= m) return;

    int src = piv_rows[i];
    if (src < 0 || src >= m) return;

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
// 完整 LU 分解测试
// ============================================================================

void run_full_LU_once(
    half* dA_half,
    int m, int n, int lda,
    int ib, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    std::vector<int>& h_ipiv_rel,
    std::vector<int>& piv_rows,
    const HgetrfConfig& config,
    HgetrfTimers* timers = nullptr)
{
    if ((int)h_ipiv_rel.size() < ib) h_ipiv_rel.resize(ib);
    if ((int)piv_rows.size() < m)    piv_rows.resize(m);

    hgetrf_blocked_half_ex(dA_half,
                          m, n, lda,
                          ib, uc,
                          handle,
                          d_ipiv_rel,
                          h_ipiv_rel.data(),
                          piv_rows.data(),
                          config,
                          0,
                          timers);
}

// ============================================================================
// Panel TSLU 测试（模拟真实使用场景）
// ============================================================================

struct PanelTestResult {
    double avg_time_ms;
    double gflops;
    double normA;
    double normRes;
    double relErr;
};

PanelTestResult test_panel_tslu_only(
    const std::vector<half>& hA0_half,
    int m, int n, int lda,
    int j0, int ib, int uc,
    int iters, int warmup,
    cublasHandle_t handle,
    bool verbose)
{
    PanelTestResult result{};
    
    // 分配设备内存
    half* dA_half  = nullptr;
    half* dA0_half = nullptr;
    int* d_ipiv_rel = nullptr;

    CUDA_CHECK(cudaMalloc(&dA_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA0_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, sizeof(int) * ib));

    CUDA_CHECK(cudaMemcpy(dA0_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    std::vector<int> h_ipiv_rel(ib);

    printf("\n========================================\n");
    printf("Panel TSLU Test (模拟真实调用场景)\n");
    printf("========================================\n");
    printf("  Matrix size:        %d x %d\n", m, n);
    printf("  Panel position:     j0 = %d\n", j0);
    printf("  Panel width:        ib = %d\n", ib);
    printf("  Remaining rows:     %d\n", n - j0);
    printf("  Remaining cols:     %d\n", n - j0);
    printf("\n这模拟了在 double-blocking LU 中调用 cuSOLVER 的场景：\n");
    printf("  cusolverDnSgetrf(handle, n-j0, ib, A+j0+j0*lda, lda, ...)\n");
    printf("========================================\n");

    // Warmup
    if (warmup > 0) {
        printf("\n[Warmup] %d iterations...\n", warmup);
        for (int w = 0; w < warmup; ++w) {
            CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyDeviceToDevice));
            
            half* panel_ptr = dA_half + j0 + j0 * lda;
            launch_panel_TSLU(panel_ptr, n - j0, lda, 0, ib, uc, d_ipiv_rel, 0);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Warmup] Complete!\n");
    }

    // Performance test
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    printf("\n[Performance] Running %d iterations...\n", iters);
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                              sizeof(half) * (size_t)lda * n,
                              cudaMemcpyDeviceToDevice));
        
        half* panel_ptr = dA_half + j0 + j0 * lda;
        launch_panel_TSLU(panel_ptr, n - j0, lda, 0, ib, uc, d_ipiv_rel, 0);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    result.avg_time_ms = total_ms / iters;

    // 估算 FLOPs (panel TSLU: 1/3 * ib^2 * (n-j0))
    double panel_flops = (1.0/3.0) * ib * ib * (n - j0);
    result.gflops = (panel_flops * 1e-9) / (result.avg_time_ms * 1e-3);

    printf("[Performance] Complete!\n");
    printf("  Avg time:           %.4f ms\n", result.avg_time_ms);
    printf("  Est. performance:   %.4f GFLOP/s\n", result.gflops);

    // Accuracy test
    printf("\n[Accuracy] Verifying factorization...\n");
    CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToDevice));
    
    half* panel_ptr = dA_half + j0 + j0 * lda;
    launch_panel_TSLU(panel_ptr, n - j0, lda, 0, ib, uc, d_ipiv_rel, 0);
    
    CUDA_CHECK(cudaMemcpy(h_ipiv_rel.data(), d_ipiv_rel,
                          sizeof(int) * ib,
                          cudaMemcpyDeviceToHost));

    // Build piv_rows for panel region
    std::vector<int> piv_rows(n - j0);
    for (int i = 0; i < n - j0; ++i) {
        piv_rows[i] = i;
    }
    for (int k = 0; k < ib; ++k) {
        int r1 = k;
        int r2 = k + h_ipiv_rel[k];
        if (r1 >= 0 && r1 < n - j0 && r2 >= 0 && r2 < n - j0) {
            std::swap(piv_rows[r1], piv_rows[r2]);
        }
    }

    // 分配验证矩阵
    half* dL_half  = nullptr;
    half* dU_half  = nullptr;
    half* dPA_half = nullptr;
    half* dLU_half = nullptr;
    half* dR_half  = nullptr;
    int*  d_piv_rows = nullptr;

    int panel_m = n - j0;
    int panel_n = n - j0;

    CUDA_CHECK(cudaMalloc(&dL_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dU_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dPA_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dLU_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dR_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&d_piv_rows, sizeof(int) * panel_m));

    CUDA_CHECK(cudaMemcpy(d_piv_rows, piv_rows.data(),
                          sizeof(int) * panel_m,
                          cudaMemcpyHostToDevice));

    // Extract L and U from panel region
    {
        dim3 block(128);
        dim3 grid(ib, (panel_m + block.x - 1) / block.x);
        build_LU_half_kernel<<<grid, block>>>(panel_ptr, 
                                               dL_half + j0 + j0 * lda, 
                                               dU_half + j0 + j0 * lda, 
                                               ib, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // Build PA for panel region
    {
        dim3 block(128);
        dim3 grid(ib, (panel_m + block.x - 1) / block.x);
        build_PA_half_kernel<<<grid, block>>>(dA0_half + j0 + j0 * lda, 
                                               d_piv_rows, 
                                               dPA_half + j0 + j0 * lda, 
                                               panel_m, ib, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // LU = L * U (only panel region)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        CUBLAS_CHECK(
            cublasGemmEx(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         panel_m, ib, ib,
                         &alpha,
                         dL_half + j0 + j0 * lda, CUDA_R_16F, lda,
                         dU_half + j0 + j0 * lda, CUDA_R_16F, lda,
                         &beta,
                         dLU_half + j0 + j0 * lda, CUDA_R_16F, lda,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Residual
    {
        dim3 block(128);
        dim3 grid(ib, (panel_m + block.x - 1) / block.x);
        residual_half_kernel<<<grid, block>>>(dPA_half + j0 + j0 * lda, 
                                               dLU_half + j0 + j0 * lda, 
                                               dR_half + j0 + j0 * lda, 
                                               panel_m, ib, lda);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back for norm computation
    std::vector<half> hR_half((size_t)lda * n, __float2half(0.0f));
    CUDA_CHECK(cudaMemcpy(hR_half.data(), dR_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    std::vector<half> hA0_panel((size_t)lda * n, __float2half(0.0f));
    CUDA_CHECK(cudaMemcpy(hA0_panel.data(), dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    result.normA   = frob_norm_half_host(hA0_panel, panel_m, ib, lda);
    result.normRes = frob_norm_half_host(hR_half, panel_m, ib, lda);
    result.relErr  = (result.normA > 0.0) ? (result.normRes / result.normA) : 0.0;

    printf("[Accuracy] Complete!\n");
    printf("\n后向误差分析（Panel TSLU 验证）:\n");
    printf("  ||A||_F                  = %.6f\n", result.normA);
    printf("  ||PA - L*U||_F           = %.6f  (绝对误差)\n", result.normRes);
    printf("  ||PA - L*U||_F / ||A||_F = %.6e  (相对误差)\n", result.relErr);
    
    if (result.relErr < 1e-3) {
        printf("  ✓ 精度优秀 (< 1e-3)\n");
    } else if (result.relErr < 1e-2) {
        printf("  ✓ 精度良好 (< 1e-2)\n");
    } else if (result.relErr < 0.1) {
        printf("  ○ 精度可接受 (< 0.1) - half精度固有限制\n");
    } else {
        printf("  ✗ 精度较差 (>= 0.1) - 可能有bug\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(dA_half));
    CUDA_CHECK(cudaFree(dA0_half));
    CUDA_CHECK(cudaFree(d_ipiv_rel));
    CUDA_CHECK(cudaFree(dL_half));
    CUDA_CHECK(cudaFree(dU_half));
    CUDA_CHECK(cudaFree(dPA_half));
    CUDA_CHECK(cudaFree(dLU_half));
    CUDA_CHECK(cudaFree(dR_half));
    CUDA_CHECK(cudaFree(d_piv_rows));

    return result;
}

// ============================================================================
// 完整 LU 精度验证
// ============================================================================

struct FullLUAccuracy {
    double normA;
    double normRes;
    double relErr;
};

FullLUAccuracy verify_full_LU_accuracy(
    half* dA_half,
    half* dA0_half,
    const std::vector<int>& piv_rows,
    int m, int n, int lda,
    cublasHandle_t handle)
{
    FullLUAccuracy result{};
    
    half* dL_half  = nullptr;
    half* dU_half  = nullptr;
    half* dPA_half = nullptr;
    half* dLU_half = nullptr;
    half* dR_half  = nullptr;
    int*  d_piv_rows = nullptr;

    CUDA_CHECK(cudaMalloc(&dL_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dU_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dPA_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dLU_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dR_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&d_piv_rows, sizeof(int) * m));

    CUDA_CHECK(cudaMemcpy(d_piv_rows, piv_rows.data(),
                          sizeof(int) * m,
                          cudaMemcpyHostToDevice));

    {
        dim3 block(128);
        dim3 grid(n, (n + block.x - 1) / block.x);
        build_LU_half_kernel<<<grid, block>>>(dA_half, dL_half, dU_half, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        build_PA_half_kernel<<<grid, block>>>(dA0_half, d_piv_rows, dPA_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        CUBLAS_CHECK(
            cublasGemmEx(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         n, n, n,
                         &alpha,
                         dL_half, CUDA_R_16F, lda,
                         dU_half, CUDA_R_16F, lda,
                         &beta,
                         dLU_half, CUDA_R_16F, lda,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        residual_half_kernel<<<grid, block>>>(dPA_half, dLU_half, dR_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<half> hR_half((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hR_half.data(), dR_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    std::vector<half> hA0_half((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hA0_half.data(), dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    result.normA   = frob_norm_half_host(hA0_half, m, n, lda);
    result.normRes = frob_norm_half_host(hR_half,   m, n, lda);
    result.relErr  = (result.normA > 0.0) ? (result.normRes / result.normA) : 0.0;

    CUDA_CHECK(cudaFree(dL_half));
    CUDA_CHECK(cudaFree(dU_half));
    CUDA_CHECK(cudaFree(dPA_half));
    CUDA_CHECK(cudaFree(dLU_half));
    CUDA_CHECK(cudaFree(dR_half));
    CUDA_CHECK(cudaFree(d_piv_rows));

    return result;
}

// ============================================================================
// 完整 LU 性能测试
// ============================================================================

struct PerfResult {
    double avg_time_ms;
    double gflops;
    FullLUAccuracy accuracy;
};

PerfResult run_full_LU_perf_test(
    const std::vector<half>& hA0_half,
    int m, int n, int lda,
    int ib, int uc,
    int iters, int warmup,
    cublasHandle_t handle,
    HgetrfConfig& config,
    const char* test_name)
{
    PerfResult result{};
    
    half* dA_half  = nullptr;
    half* dA0_half = nullptr;
    int* d_ipiv_rel = nullptr;

    CUDA_CHECK(cudaMalloc(&dA_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA0_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, sizeof(int) * ib));

    CUDA_CHECK(cudaMemcpy(dA0_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    std::vector<int> h_ipiv_rel(ib);
    std::vector<int> piv_rows(m);

    printf("\n========================================\n");
    printf("Test: %s\n", test_name);
    printf("========================================\n");
    
    // 只在第一次打印配置
    if (config.verbose) {
        printf("[Configuration]\n");
        printf("  Tensor Core GEMM: %s\n", config.use_tensor_core_gemm ? "enabled" : "disabled");
        printf("  TRSM mode: ");
        switch (config.trsm_mode) {
            case HgetrfConfig::TRSM_CUSTOM_KERNEL:
                printf("custom kernel\n");
                break;
            case HgetrfConfig::TRSM_CUBLAS_HALF:
                printf("cuBLAS half\n");
                break;
            case HgetrfConfig::TRSM_CUBLAS_FLOAT:
                printf("cuBLAS float\n");
                break;
        }
        printf("  Batched exchange: %s\n", config.use_batched_exchange ? "enabled" : "disabled");
        config.verbose = false; // 关闭后续打印
    }
    
    // Warmup
    if (warmup > 0) {
        printf("\n[Warmup] %d iterations...\n", warmup);
        for (int w = 0; w < warmup; ++w) {
            CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyDeviceToDevice));
            run_full_LU_once(dA_half, m, n, lda, ib, uc, handle,
                           d_ipiv_rel, h_ipiv_rel, piv_rows, config);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("[Warmup] Complete!\n");
    }

    // Performance test
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    printf("\n[Performance] Running %d iterations...\n", iters);
    CUDA_CHECK(cudaEventRecord(ev_start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                              sizeof(half) * (size_t)lda * n,
                              cudaMemcpyDeviceToDevice));
        run_full_LU_once(dA_half, m, n, lda, ib, uc, handle,
                         d_ipiv_rel, h_ipiv_rel, piv_rows, config);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    result.avg_time_ms = total_ms / iters;

    double flops_total = 2.0 / 3.0 * (double)n * (double)n * (double)n;
    result.gflops = (flops_total * 1e-9) / (result.avg_time_ms * 1e-3);

    printf("[Performance] Complete!\n");
    printf("  Avg time:           %.4f ms\n", result.avg_time_ms);
    printf("  Est. FLOPs:         %.4f GFLOP\n", flops_total * 1e-9);
    printf("  Est. performance:   %.4f GFLOP/s\n", result.gflops);

    // Accuracy test with detailed timing
    printf("\n[Accuracy] Running single factorization with timing...\n");
    CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToDevice));
    
    HgetrfTimers timers{};
    run_full_LU_once(dA_half, m, n, lda, ib, uc, handle,
                     d_ipiv_rel, h_ipiv_rel, piv_rows, config, &timers);
    
    result.accuracy = verify_full_LU_accuracy(dA_half, dA0_half, piv_rows, m, n, lda, handle);

    printf("[Accuracy] Complete!\n");
    printf("\n后向误差分析（Full LU 验证）:\n");
    printf("  ||A||_F                  = %.6f\n", result.accuracy.normA);
    printf("  ||PA - L*U||_F           = %.6f  (绝对误差)\n", result.accuracy.normRes);
    printf("  ||PA - L*U||_F / ||A||_F = %.6e  (相对误差)\n", result.accuracy.relErr);
    
    if (result.accuracy.relErr < 1e-3) {
        printf("  ✓ 精度优秀 (< 1e-3)\n");
    } else if (result.accuracy.relErr < 1e-2) {
        printf("  ✓ 精度良好 (< 1e-2)\n");
    } else if (result.accuracy.relErr < 0.1) {
        printf("  ○ 精度可接受 (< 0.1) - half精度固有限制\n");
    } else {
        printf("  ✗ 精度较差 (>= 0.1) - 可能有bug\n");
    }
    
    printf("\n时间分解（单次分解）:\n");
    double total = timers.panel_ms + timers.exchange_ms + timers.trsm_ms + timers.gemm_ms;
    if (total > 0) {
        printf("  Panel:    %8.4f ms (%.1f%%)\n", timers.panel_ms, 100.0 * timers.panel_ms / total);
        printf("  Exchange: %8.4f ms (%.1f%%)\n", timers.exchange_ms, 100.0 * timers.exchange_ms / total);
        printf("  TRSM:     %8.4f ms (%.1f%%)\n", timers.trsm_ms, 100.0 * timers.trsm_ms / total);
        printf("  GEMM:     %8.4f ms (%.1f%%)\n", timers.gemm_ms, 100.0 * timers.gemm_ms / total);
        printf("  Total:    %8.4f ms\n", timers.total_ms);
    }

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(dA_half));
    CUDA_CHECK(cudaFree(dA0_half));
    CUDA_CHECK(cudaFree(d_ipiv_rel));

    return result;
}

// ============================================================================
// 命令行参数解析
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("Options:\n");
    printf("  -n <N>          Matrix size (default: 2048)\n");
    printf("  -ib <IB>        Panel width (default: 128)\n");
    printf("  -uc <UC>        Micro-block size (default: 8)\n");
    printf("  -iters <N>      Number of iterations (default: 10)\n");
    printf("  -warmup <N>     Warmup iterations (default: 2)\n");
    printf("  -mode <MODE>    Test mode: full/panel/both (default: full)\n");
    printf("  -j0 <J0>        Panel position for panel mode (default: 0)\n");
    printf("  -tc_on          Test Tensor Core ON (default)\n");
    printf("  -tc_off         Test Tensor Core OFF\n");
    printf("  -all            Test all combinations\n");
    printf("  -verbose        Verbose output\n");
    printf("  -h, --help      Show this help\n");
    printf("\nTest modes:\n");
    printf("  full   - Complete LU factorization\n");
    printf("  panel  - Panel TSLU only (simulates real usage)\n");
    printf("  both   - Run both tests\n");
}

TestConfig parse_args(int argc, char** argv) {
    TestConfig config;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        else if (arg == "-n" && i + 1 < argc) {
            config.n = std::atoi(argv[++i]);
        }
        else if (arg == "-ib" && i + 1 < argc) {
            config.ib = std::atoi(argv[++i]);
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
        else if (arg == "-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "full") {
                config.mode = TestConfig::MODE_FULL_LU;
            } else if (mode == "panel") {
                config.mode = TestConfig::MODE_PANEL_ONLY;
            } else if (mode == "both") {
                config.mode = TestConfig::MODE_BOTH;
            }
        }
        else if (arg == "-tc_off") {
            config.test_tc_off = true;
            config.test_tc_on = false;
        }
        else if (arg == "-all") {
            config.test_tc_on = true;
            config.test_tc_off = true;
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
    TestConfig test_cfg = parse_args(argc, argv);
    
    int n   = test_cfg.n;
    int m   = n;
    int lda = m;
    int ib  = test_cfg.ib;
    int uc  = test_cfg.uc;

    printf("========================================\n");
    printf("Half Precision LU Decomposition Test\n");
    printf("========================================\n");
    printf("Configuration:\n");
    printf("  Matrix size:       %d x %d\n", n, n);
    printf("  Panel width (ib):  %d\n", ib);
    printf("  Micro-block (uc):  %d\n", uc);
    printf("  Iterations:        %d\n", test_cfg.iters);
    printf("  Warmup:            %d\n", test_cfg.warmup);
    printf("  Test mode:         ");
    switch (test_cfg.mode) {
        case TestConfig::MODE_FULL_LU:
            printf("Full LU\n");
            break;
        case TestConfig::MODE_PANEL_ONLY:
            printf("Panel TSLU only\n");
            break;
        case TestConfig::MODE_BOTH:
            printf("Both\n");
            break;
    }
    printf("========================================\n");

    // 生成测试矩阵
    std::vector<float> hA0_float((size_t)m * n);
    generate_random_A_float(hA0_float, m, n, 7654321);

    std::vector<half> hA0_half;
    convert_float_to_half(hA0_float, hA0_half, m, n);

    // 初始化 cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Panel TSLU 测试
    if (test_cfg.mode == TestConfig::MODE_PANEL_ONLY || test_cfg.mode == TestConfig::MODE_BOTH) {
        int j0 = 0;  // 从矩阵开始位置测试
        test_panel_tslu_only(hA0_half, m, n, lda, j0, ib, uc,
                            test_cfg.iters, test_cfg.warmup,
                            handle, test_cfg.verbose);
    }

    // 完整 LU 测试
    if (test_cfg.mode == TestConfig::MODE_FULL_LU || test_cfg.mode == TestConfig::MODE_BOTH) {
        std::vector<std::pair<std::string, PerfResult>> results;

        if (test_cfg.test_tc_on) {
            HgetrfConfig config;
            config.use_tensor_core_gemm = true;
            config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
            config.verbose = test_cfg.verbose;
            
            auto result = run_full_LU_perf_test(hA0_half, m, n, lda, ib, uc,
                                                test_cfg.iters, test_cfg.warmup,
                                                handle, config,
                                                "TC ON + Custom TRSM");
            results.push_back({"TC ON + Custom TRSM", result});
        }

        if (test_cfg.test_tc_off) {
            HgetrfConfig config;
            config.use_tensor_core_gemm = false;
            config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
            config.verbose = test_cfg.verbose;
            
            auto result = run_full_LU_perf_test(hA0_half, m, n, lda, ib, uc,
                                                test_cfg.iters, test_cfg.warmup,
                                                handle, config,
                                                "TC OFF + Custom TRSM");
            results.push_back({"TC OFF + Custom TRSM", result});
        }

        // 打印汇总
        if (!results.empty()) {
            printf("\n========================================\n");
            printf("Summary (Full LU)\n");
            printf("========================================\n");
            printf("%-25s  %10s  %10s  %12s\n", 
                   "Configuration", "Time(ms)", "GFLOP/s", "Rel Error");
            printf("----------------------------------------\n");
            
            for (const auto& r : results) {
                printf("%-25s  %10.4f  %10.4f  %12.6e\n",
                       r.first.c_str(),
                       r.second.avg_time_ms,
                       r.second.gflops,
                       r.second.accuracy.relErr);
            }
        }
    }

    // 清理
    cleanup_panel_buffers();
    cleanup_exchange_buffers();
    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\n========================================\n");
    printf("Test completed!\n");
    printf("========================================\n");

    return 0;
}