#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// 注意：根据命令行参数选择不同的实现
// 需要准备两个版本的头文件：
// - A1_panel_baseline.cuh / A1_panel_optimized.cuh
// - A_exchange_baseline.cuh / A_exchange_optimized.cuh
// 或者在头文件中使用宏控制

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
// 测试配置（新增优化选项）
// ============================================================================

struct TestConfig {
    int m          = 32768;
    int n          = 128;
    int uc         = 8;
    int iters      = 10;
    int warmup     = 2;
    bool verbose   = false;
    
    // 新增：优化控制
    std::string optimization = "baseline";  // baseline, panelonly, exchangeonly, both
};

// ============================================================================
// 预定义场景
// ============================================================================

struct TSScenario {
    const char* name;
    int m;
    int n;
    const char* description;
    const char* use_case;
};

const TSScenario TS_SCENARIOS[] = {
    {"Tiny Panel",     1024,  32,   "超小 panel", "矩阵末尾"},
    {"Small Panel",    2048,  64,   "小 panel", "early stage"},
    {"Medium Panel",   4096,  128,  "中等 panel", "mid stage"},
    {"Large Panel",    8192,  128,  "大 panel", "mid stage"},
    {"XLarge Panel",   16384, 128,  "超大 panel", "late stage"},
    {"XXLarge Panel",  32768, 128,  "巨大 panel", "very late"},
    {"Wide Panel",     8192,  256,  "较宽 panel", "wider blocking"},
    {"Narrow Panel",   16384, 64,   "较窄 panel", "narrow blocking"},
    {"Small Square",   1024,  1024, "小方阵", "矩阵尾部剩余"},
    {"Medium Square",  2048,  2048, "中等方阵", "测试 blocking"},
};

// ============================================================================
// 辅助函数（保持不变）
// ============================================================================

void generate_random_A_float(std::vector<float>& hA, int m, int n, int lda, unsigned seed = 1234567)
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
// 测试函数（新增优化说明）
// ============================================================================

struct TSTestResult {
    double avg_time_ms;
    double gflops;
    double normA;
    double normRes;
    double relErr;
    int num_panels;
};

TSTestResult test_matrix_lu(
    const std::vector<half>& hA0_half,
    int m, int n, int lda,
    int uc,
    int iters, int warmup,
    cublasHandle_t handle,
    bool verbose,
    const char* scenario_name,
    const std::string& optimization)
{
    TSTestResult result{};
    
    half* dA_half  = nullptr;
    half* dA0_half = nullptr;
    int* d_ipiv_rel = nullptr;

    CUDA_CHECK(cudaMalloc(&dA_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA0_half, sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, sizeof(int) * std::max(m, n)));

    CUDA_CHECK(cudaMemcpy(dA0_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    std::vector<int> h_ipiv_rel(std::max(m, n));
    std::vector<int> piv_rows(m);

    printf("\n========================================\n");
    printf("测试场景: %s\n", scenario_name);
    printf("========================================\n");
    printf("矩阵形状:\n");
    printf("  行数 (m):           %d\n", m);
    printf("  列数 (n):           %d\n", n);
    printf("  lda:                %d\n", lda);
    printf("  微块大小 (uc):      %d\n", uc);
    
    // 新增：优化说明
    printf("\n优化配置: %s\n", optimization.c_str());
    if (optimization == "baseline") {
        printf("  说明: 原始实现（无优化）\n");
    } else if (optimization == "panelonly") {
        printf("  说明: Panel自适应优化（m_effective）\n");
    } else if (optimization == "exchangeonly") {
        printf("  说明: Exchange高效kernel\n");
    } else if (optimization == "both") {
        printf("  说明: Panel + Exchange双重优化\n");
    }
    printf("========================================\n");

    // Warmup
    if (warmup > 0) {
        printf("\n[Warmup] %d iterations...\n", warmup);
        for (int w = 0; w < warmup; ++w) {
            CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                                  sizeof(half) * (size_t)lda * n,
                                  cudaMemcpyDeviceToDevice));
            
            HgetrfConfig config;
            config.use_tensor_core_gemm = true;
            config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
            config.verbose = false;
            
            hgetrf_auto(dA_half, m, n, lda, 128, uc, handle,
                       d_ipiv_rel, h_ipiv_rel.data(), piv_rows.data(),
                       config, 0, nullptr);
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
        
        HgetrfConfig config;
        config.use_tensor_core_gemm = true;
        config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
        config.use_multi_stream = true;  // ← 启用多流水线测试
        config.verbose = false;
        
        hgetrf_auto(dA_half, m, n, lda, 128, uc, handle,
                   d_ipiv_rel, h_ipiv_rel.data(), piv_rows.data(),
                   config, 0, nullptr);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    result.avg_time_ms = total_ms / iters;

    double total_flops = (2.0 / 3.0) * n * n * m;
    result.gflops = (total_flops * 1e-9) / (result.avg_time_ms * 1e-3);

    printf("[Performance] Complete!\n");
    printf("\n性能指标:\n");
    printf("  平均时间:           %.4f ms\n", result.avg_time_ms);
    printf("  估算性能:           %.4f GFLOP/s\n", result.gflops);

    // Accuracy test with timing
    printf("\n[Accuracy] 验证分解正确性（含详细计时）...\n");
    CUDA_CHECK(cudaMemcpy(dA_half, dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToDevice));
    
    HgetrfConfig config;
    config.use_tensor_core_gemm = true;
    config.trsm_mode = HgetrfConfig::TRSM_CUSTOM_KERNEL;
    config.verbose = verbose;
    
    HgetrfTimers timers{};
    hgetrf_auto(dA_half, m, n, lda, 128, uc, handle,
               d_ipiv_rel, h_ipiv_rel.data(), piv_rows.data(),
               config, 0, &timers);
    
    result.num_panels = timers.panels;

    // 验证 PA = LU（保持不变）
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
        dim3 grid(n, (m + block.x - 1) / block.x);
        build_LU_half_kernel<<<grid, block>>>(dA_half, dL_half, dU_half, m, n, lda);
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
                         m, n, n,
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

    std::vector<half> hA0_copy((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hA0_copy.data(), dA0_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    result.normA   = frob_norm_half_host(hA0_copy, m, n, lda);
    result.normRes = frob_norm_half_host(hR_half, m, n, lda);
    result.relErr  = (result.normA > 0.0) ? (result.normRes / result.normA) : 0.0;

    printf("[Accuracy] Complete!\n");
    printf("\n后向误差分析:\n");
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

    if (timers.total_ms > 0) {
        printf("\n时间分解:\n");
        printf("  实际 panel 数:      %d\n", result.num_panels);
        double total = timers.panel_ms + timers.exchange_ms + timers.trsm_ms + timers.gemm_ms;
        printf("  Panel:              %.4f ms (%.1f%%)\n", timers.panel_ms, 100.0 * timers.panel_ms / total);
        printf("  Exchange:           %.4f ms (%.1f%%)\n", timers.exchange_ms, 100.0 * timers.exchange_ms / total);
        printf("  TRSM:               %.4f ms (%.1f%%)\n", timers.trsm_ms, 100.0 * timers.trsm_ms / total);
        printf("  GEMM:               %.4f ms (%.1f%%)\n", timers.gemm_ms, 100.0 * timers.gemm_ms / total);
        printf("  Total (measured):   %.4f ms\n", timers.total_ms);
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
// 命令行参数解析（新增优化选项）
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nBasic options:\n");
    printf("  -m <M>          Number of rows (default: 32768)\n");
    printf("  -n <N>          Number of columns (default: 128)\n");
    printf("  -uc <UC>        Micro-block size (default: 8)\n");
    printf("  -iters <N>      Number of iterations (default: 10)\n");
    printf("  -warmup <N>     Warmup iterations (default: 2)\n");
    printf("  -verbose        Verbose output\n");
    printf("\nOptimization options:\n");
    printf("  -opt <mode>     Optimization mode:\n");
    printf("    baseline      : 原始实现（无优化）[默认]\n");
    printf("    panelonly     : Panel自适应优化\n");
    printf("    exchangeonly  : Exchange高效kernel\n");
    printf("    both          : Panel + Exchange双重优化\n");
    printf("\nPreset scenarios:\n");
    printf("  -preset <n>     Use predefined scenario\n");
    printf("  -all_presets    Test all predefined scenarios\n");
    printf("  -compare        对比所有优化模式（32K×32K）\n");
    printf("\n  -h, --help      Show this help\n");
}

TestConfig parse_args(int argc, char** argv, bool& use_preset, int& preset_id, bool& test_all, bool& compare_mode) {
    TestConfig config;
    use_preset = false;
    preset_id = -1;
    test_all = false;
    compare_mode = false;
    
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
        else if (arg == "-opt" && i + 1 < argc) {
            config.optimization = argv[++i];
        }
        else if (arg == "-preset" && i + 1 < argc) {
            use_preset = true;
            preset_id = std::atoi(argv[++i]);
        }
        else if (arg == "-all_presets") {
            test_all = true;
        }
        else if (arg == "-compare") {
            compare_mode = true;
        }
    }
    
    return config;
}

// ============================================================================
// 对比模式（新增）
// ============================================================================

void run_comparison_mode(cublasHandle_t handle) {
    printf("\n========================================\n");
    printf("优化对比模式（32K×32K）\n");
    printf("========================================\n");
    
    int m = 32768, n = 32768, lda = m, uc = 8;
    int iters = 5, warmup = 2;
    
    // 生成测试矩阵
    std::vector<float> hA0_float((size_t)lda * n);
    generate_random_A_float(hA0_float, m, n, lda, 7654321);
    std::vector<half> hA0_half;
    convert_float_to_half(hA0_float, hA0_half, m, n, lda);
    
    // 测试所有模式
    const char* modes[] = {"baseline", "panelonly", "exchangeonly", "both"};
    TSTestResult results[4];
    
    for (int i = 0; i < 4; ++i) {
        results[i] = test_matrix_lu(hA0_half, m, n, lda, uc,
                                     iters, warmup, handle, false,
                                     "32K Comparison", modes[i]);
    }
    
    // 打印对比表
    printf("\n========================================\n");
    printf("对比汇总\n");
    printf("========================================\n");
    printf("%-15s %10s %10s %10s %10s %12s\n",
           "Mode", "Time(ms)", "Panel(ms)", "Exch(ms)", "GEMM(ms)", "RelError");
    printf("------------------------------------------------------------------------------\n");
    
    for (int i = 0; i < 4; ++i) {
        printf("%-15s %10.2f %10.2f %10.2f %10.2f %12.2e\n",
               modes[i],
               results[i].avg_time_ms,
               0.0,  // 需要从timers获取，暂时省略
               0.0,
               0.0,
               results[i].relErr);
    }
    
    // 计算提升
    printf("\n相对Baseline的提升:\n");
    double baseline_time = results[0].avg_time_ms;
    for (int i = 1; i < 4; ++i) {
        double improvement = 100.0 * (baseline_time - results[i].avg_time_ms) / baseline_time;
        printf("  %s: %.2f%%\n", modes[i], improvement);
    }
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char** argv)
{
    bool use_preset = false;
    int preset_id = -1;
    bool test_all = false;
    bool compare_mode = false;
    
    TestConfig test_cfg = parse_args(argc, argv, use_preset, preset_id, test_all, compare_mode);

    printf("========================================\n");
    printf("Half 精度 LU 分解优化测试\n");
    printf("========================================\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 对比模式
    if (compare_mode) {
        run_comparison_mode(handle);
        CUBLAS_CHECK(cublasDestroy(handle));
        return 0;
    }

    std::vector<TSTestResult> results;
    std::vector<std::string> names;

    // 测试所有预定义场景
    if (test_all) {
        printf("\n测试所有预定义场景（使用优化: %s）...\n", test_cfg.optimization.c_str());
        
        for (size_t i = 0; i < sizeof(TS_SCENARIOS)/sizeof(TS_SCENARIOS[0]); ++i) {
            int m = TS_SCENARIOS[i].m;
            int n = TS_SCENARIOS[i].n;
            int lda = m;

            std::vector<float> hA0_float((size_t)lda * n);
            generate_random_A_float(hA0_float, m, n, lda, 7654321 + i);

            std::vector<half> hA0_half;
            convert_float_to_half(hA0_float, hA0_half, m, n, lda);

            auto result = test_matrix_lu(hA0_half, m, n, lda, test_cfg.uc,
                                        test_cfg.iters, test_cfg.warmup,
                                        handle, test_cfg.verbose,
                                        TS_SCENARIOS[i].name,
                                        test_cfg.optimization);
            results.push_back(result);
            names.push_back(TS_SCENARIOS[i].name);
        }
    }
    // 测试单个场景
    else {
        int m = use_preset ? TS_SCENARIOS[preset_id].m : test_cfg.m;
        int n = use_preset ? TS_SCENARIOS[preset_id].n : test_cfg.n;
        int lda = m;

        std::vector<float> hA0_float((size_t)lda * n);
        generate_random_A_float(hA0_float, m, n, lda, 7654321);

        std::vector<half> hA0_half;
        convert_float_to_half(hA0_float, hA0_half, m, n, lda);

        const char* name = use_preset ? TS_SCENARIOS[preset_id].name : "Custom Matrix";
        auto result = test_matrix_lu(hA0_half, m, n, lda, test_cfg.uc,
                                     test_cfg.iters, test_cfg.warmup,
                                     handle, test_cfg.verbose,
                                     name,
                                     test_cfg.optimization);
        results.push_back(result);
        names.push_back(name);
    }

    // 清理
    cleanup_panel_buffers();
    cleanup_exchange_buffers();
    CUBLAS_CHECK(cublasDestroy(handle));

    printf("\n========================================\n");
    printf("测试完成!\n");
    printf("========================================\n");

    return 0;
}