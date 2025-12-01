#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

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

// ================== 工具：生成随机矩阵 (float host, col-major) ==================

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

// host: float -> half
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

// ================== kernel: 从 LU(half) 拆 L/U(half) ==================

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
        // strictly lower
        L[i + (size_t)j * lda] = v;
        U[i + (size_t)j * lda] = __float2half(0.0f);
    } else if (i == j) {
        L[i + (size_t)j * lda] = __float2half(1.0f);
        U[i + (size_t)j * lda] = v;
    } else {
        // i < j
        L[i + (size_t)j * lda] = __float2half(0.0f);
        U[i + (size_t)j * lda] = v;
    }
}

// ================== kernel: 构造 PA(half) ==================

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

// ================== kernel: R = PA - LU (half) ==================

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

// ================== host: Frobenius norm from half (CPU) ==================

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

// ================== 一个函数：对 dA_half 做一次完整 LU 分解 ==================
//
// 输入:
//   dA_half   : device half 矩阵, 大小 lda x n, 初始为待分解 A
//   m, n, lda : 行, 列, leading dimension
//   ib, uc    : panel 宽度 / panel 内微块宽度
//   handle    : cuBLAS handle, 用于 A22 GEMM
//
// 输出:
//   d_ipiv_rel: device pivot 相对位移（panel 用）
//   h_ipiv_rel/piv_rows: 需由外面传入引用以更新全局行置换
//
void run_full_LU_once(
    half* dA_half,
    int m, int n, int lda,
    int ib, int uc,
    cublasHandle_t handle,
    int* d_ipiv_rel,
    std::vector<int>& h_ipiv_rel,
    std::vector<int>& piv_rows,
    HgetrfTimers* timers = nullptr)
{
    // 使用封装好的接口完成一次完整的分块 LU
    if ((int)h_ipiv_rel.size() < ib) h_ipiv_rel.resize(ib);
    if ((int)piv_rows.size() < m)    piv_rows.resize(m);

    hgetrf_blocked_half(dA_half,
                        m, n, lda,
                        ib, uc,
                        handle,
                        d_ipiv_rel,
                        h_ipiv_rel.data(),
                        piv_rows.data(),
                        0,     // stream
                        true,  // use Tensor Core GEMM
                        true,  // use cuBLAS TRSM if available
                        timers);
}

// ================== 主程序：完整 LU + cuBLAS half 检验 ==================

int main(int argc, char** argv)
{
    int n     = 2048;
    // panel的大小
    int ib    = 128;
    // 理论上影响的是每次更新的大小
    int uc    = 8;
    int iters = 10;

    if (argc >= 2) n     = std::atoi(argv[1]);
    if (argc >= 3) ib    = std::atoi(argv[2]);
    if (argc >= 4) uc    = std::atoi(argv[3]);
    if (argc >= 5) iters = std::atoi(argv[4]);

    int m   = n;
    int lda = m;

    if (n <= 0 || ib <= 0 || iters <= 0) {
        printf("Invalid args.\n");
        return 0;
    }
    if (n % ib != 0) {
        printf("For this test, require n %% ib == 0.\n");
        printf("Given n=%d, ib=%d.\n", n, ib);
        return 0;
    }

    printf("Testing full blocked LU (half-based) with panel_TSLU + exchange + TRSM + GEMM\n");
    printf("  n = %d, m = %d, ib = %d, uc = %d, iters = %d\n", n, m, ib, uc, iters);

    // ===== 1. 生成随机 A0(float) -> A0(half) =====
    std::vector<float> hA0_float((size_t)m * n);
    generate_random_A_float(hA0_float, m, n, 7654321);

    std::vector<half> hA0_half;
    convert_float_to_half(hA0_float, hA0_half, m, n);

    // ===== 2. device 内存 =====
    half* dA_half  = nullptr;  // 用于 factorization
    half* dA0_half = nullptr;  // 保留原始 A0, 用于构造 PA

    CUDA_CHECK(cudaMalloc(&dA_half,  sizeof(half) * (size_t)lda * n));
    CUDA_CHECK(cudaMalloc(&dA0_half, sizeof(half) * (size_t)lda * n));

    CUDA_CHECK(cudaMemcpy(dA_half,  hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dA0_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    int* d_ipiv_rel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, sizeof(int) * ib));

    std::vector<int> h_ipiv_rel(ib);
    std::vector<int> piv_rows(m);

    // cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // 感觉上这个预热非常有问题，要作为一个修改的点来看[question]但是修改过后并没有很明显的区别

    // // ===== 3. 真正的预热：用 dummy，不碰真实数据、不更新 piv_rows =====
    // {
    //     half* dDummy = nullptr;
    //     int   dummy_m = ib;
    //     int   dummy_n = ib;
    //     int   dummy_lda = dummy_m;
    //     CUDA_CHECK(cudaMalloc(&dDummy, sizeof(half) * (size_t)dummy_lda * dummy_n));
    //     CUDA_CHECK(cudaMemset(dDummy, 0, sizeof(half) * (size_t)dummy_lda * dummy_n));

    //     // panel 预热
    //     launch_panel_TSLU(dDummy, dummy_m, dummy_lda, 0, ib, uc, d_ipiv_rel, 0);
    //     // TRSM 预热
    //     launch_A12_trsm(dDummy, dummy_m, dummy_n, dummy_lda, 0, ib, 0);
    //     // GEMM 预热（这里随便给个参数，主要是触发 kernel / cuBLAS 初始化）
    //     launch_A22_gemm_tc(dDummy, dummy_m, dummy_n, dummy_lda, 0, ib, handle, 0);

    //     CUDA_CHECK(cudaDeviceSynchronize());
    //     CUDA_CHECK(cudaFree(dDummy));
    // }

    // ===== 4. 计时循环：只测性能，每次自己重置 A，不拿这些结果算误差 =====

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    CUDA_CHECK(cudaEventRecord(ev_start));
    // 他这里的逻辑是做了一次并且时间计算其实也不算太标准吧，需要我进行改动 [question]
    for (int it = 0; it < iters; ++it) {
        // 重置 A 为原始矩阵
        CUDA_CHECK(cudaMemcpy(dA_half, hA0_half.data(),
                              sizeof(half) * (size_t)lda * n,
                              cudaMemcpyHostToDevice));
        // 做一次完整 LU，piv_rows 在这里更新，但我们只用来模拟真实开销
        run_full_LU_once(dA_half, m, n, lda, ib, uc,
                         handle, d_ipiv_rel, h_ipiv_rel, piv_rows);
    }
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    float avg_ms = total_ms / iters;

    double flops_total = 2.0 / 3.0 * (double)n * (double)n * (double)n;
    double gflops      = (flops_total * 1e-9) / (avg_ms * 1e-3);

    printf("\nFull LU (blocked, half-based) performance (timing only loop):\n");
    printf("  Avg time per factorization: %.4f ms\n", avg_ms);
    printf("  Estimated FLOPs:            %.4f GFLOP\n", flops_total * 1e-9);
    printf("  Estimated performance:      %.4f GFLOP/s\n", gflops);

    // ===== 5. 精度检查：单独再做一次干净的 LU，用这一次的结果算 PA vs LU =====

    // 重置矩阵
    CUDA_CHECK(cudaMemcpy(dA_half, hA0_half.data(),
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyHostToDevice));

    HgetrfTimers timers{};
    // 用 run_full_LU_once 再 factor 一次（这次的 piv_rows 就是误差用的 P）
    run_full_LU_once(dA_half, m, n, lda, ib, uc,
                     handle, d_ipiv_rel, h_ipiv_rel, piv_rows, &timers);

    printf("\nPer-step timing (single factorization, ms):\n");
    double panel_avg = (timers.panels > 0) ? timers.panel_ms / timers.panels : 0.0;
    double exch_avg  = (timers.panels > 0) ? timers.exchange_ms / timers.panels : 0.0;
    double trsm_avg  = (timers.panels > 0) ? timers.trsm_ms / timers.panels : 0.0;
    double gemm_avg  = (timers.panels > 0) ? timers.gemm_ms / timers.panels : 0.0;
    printf("  panels    : total %.4f ms, avg %.4f ms (panels=%d)\n", timers.panel_ms, panel_avg, timers.panels);
    printf("  exchange  : total %.4f ms, avg %.4f ms\n", timers.exchange_ms, exch_avg);
    printf("  TRSM      : total %.4f ms, avg %.4f ms\n", timers.trsm_ms, trsm_avg);
    printf("  GEMM      : total %.4f ms, avg %.4f ms\n", timers.gemm_ms, gemm_avg);
    printf("  overall   : total %.4f ms\n", timers.total_ms);

    // 5.1 分配 L_half, U_half, PA_half, LU_half, R_half
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

    // 5.2 build L / U (half)
    {
        dim3 block(128);
        dim3 grid(n, (n + block.x - 1) / block.x);
        build_LU_half_kernel<<<grid, block>>>(dA_half, dL_half, dU_half, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5.3 build PA (half)
    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        build_PA_half_kernel<<<grid, block>>>(dA0_half, d_piv_rows, dPA_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }

    // 5.4 LU_half = L_half * U_half (cublasGemmEx half in/out, float accumulate)
    {
        float alpha = 1.0f;
        float beta  = 0.0f;
        CUBLAS_CHECK(
            cublasGemmEx(handle,
                         CUBLAS_OP_N, CUBLAS_OP_N,
                         n,  // m
                         n,  // n
                         n,  // k
                         &alpha,
                         dL_half, CUDA_R_16F, lda,
                         dU_half, CUDA_R_16F, lda,
                         &beta,
                         dLU_half, CUDA_R_16F, lda,
                         CUDA_R_32F,
                         CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // 5.5 R_half = PA_half - LU_half
    {
        dim3 block(128);
        dim3 grid(n, (m + block.x - 1) / block.x);
        residual_half_kernel<<<grid, block>>>(dPA_half, dLU_half, dR_half, m, n, lda);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5.6 拷回 A0_half / R_half，算 Frobenius norm
    std::vector<half> hR_half((size_t)lda * n);
    CUDA_CHECK(cudaMemcpy(hR_half.data(), dR_half,
                          sizeof(half) * (size_t)lda * n,
                          cudaMemcpyDeviceToHost));

    double normA   = frob_norm_half_host(hA0_half, m, n, lda);
    double normRes = frob_norm_half_host(hR_half,   m, n, lda);
    double relErr  = (normA > 0.0) ? (normRes / normA) : 0.0;

    printf("\nFull LU accuracy (half-based storage, cuBLAS half GEMM, single factorization):\n");
    printf("  ||A||_F                  = %.6f\n", normA);
    printf("  ||PA - L*U||_F           = %.6f  (绝对误差)\n", normRes);
    printf("  ||PA - L*U||_F / ||A||_F = %.6e  (相对误差)\n", relErr);

    // ===== 6. 清理 =====
    CUBLAS_CHECK(cublasDestroy(handle));
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

    return 0;
}
