// test_A1.cu
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cublas_v2.h>

#include "A1_panel.cuh"  // 你的 panel_TSLU kernel & launch_panel_TSLU

using half = __half;

// ======================================================================
// 宏：尽量避免和 A1_panel.cuh 冲突，如果那边已经定义就不再重定义
// ======================================================================
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
            fprintf(stderr, "CUBLAS error %s:%d (status=%d)\n",                \
                    __FILE__, __LINE__, (int)st__);                            \
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

// ======================================================================
// 1. 在 GPU 上生成 half 矩阵：curand 生成 float，再转 half
// ======================================================================

__global__ void float_to_half_kernel(const float* __restrict__ in,
                                     half* __restrict__ out,
                                     int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __float2half(in[idx]);
    }
}

// 生成 m x n 的 N(0,1) half 矩阵（列主序）
void generate_normal_half_matrix(half* dA, int m, int n, unsigned long long seed)
{
    int N = m * n;

    float* dTmp = nullptr;
    CUDA_CHECK(cudaMalloc(&dTmp, N * sizeof(float)));

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateNormal(gen, dTmp, N, 0.0f, 1.0f));
    CURAND_CHECK(curandDestroyGenerator(gen));

    int block = 256;
    int grid  = (N + block - 1) / block;
    float_to_half_kernel<<<grid, block>>>(dTmp, dA, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(dTmp));
}

// ======================================================================
// 2. 从 panel LU 的 A_fact(half, m x n) 抽 L(m x n)、U(n x n)
// ======================================================================
//
// A_fact: m x n, 前 n 列被 factor，顶部 n x n 是 A11，下面 (m-n) x n 是 A21
// L: m x n (unit lower)   -> 下三角和 A21 部分来自 A_fact，下三角对角线=1，上三角=0
// U: n x n (upper)        -> 只从 A_fact 的前 n 行抽上三角（含对角线）
//
__global__ void extract_LU_panel_kernel(const half* __restrict__ Afact,
                                        half* __restrict__ L,
                                        half* __restrict__ U,
                                        int m, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列

    if (j >= n || i >= m) return;

    half aij = Afact[i + j * m];
    half h0  = __float2half(0.0f);
    half h1  = __float2half(1.0f);

    // ---- L: m x n ----
    if (i < j) {
        L[i + j * m] = h0;      // 上三角清零
    } else if (i == j) {
        L[i + j * m] = h1;      // 对角线=1
    } else {
        L[i + j * m] = aij;     // 下三角 + A21
    }

    // ---- U: n x n (只处理 i < n) ----
    if (i < n) {
        if (i <= j) {
            U[i + j * n] = aij; // 上三角 + 对角线
        } else {
            U[i + j * n] = h0;  // 下三角清零
        }
    }
}

// ======================================================================
// 3. 从 ipiv_rel 构造长度为 m 的行置换 row_perm（host）
//
// ipiv_rel[k] = p - k (j0=0)
// 实际 swap 序列：for k in [0..n-1]: swap(row k, row p)
// row_perm[i] = PA(i,:) 在原始 A 中来自哪一行
// ======================================================================
void build_row_perm_from_ipiv(const std::vector<int>& ipiv_rel,
                              std::vector<int>& row_perm,
                              int m, int n)
{
    row_perm.resize(m);
    for (int i = 0; i < m; ++i) {
        row_perm[i] = i;
    }
    int nk = std::min(m, n);
    for (int k = 0; k < nk; ++k) {
        int p = k + ipiv_rel[k];
        if (p < 0 || p >= m) continue;
        std::swap(row_perm[k], row_perm[p]);
    }
}

// ======================================================================
// 4. 应用 row_perm 到 panel: PA(i,j) = A(row_perm[i], j)
//
// A, PA: m x n, 列主序
// ======================================================================
__global__ void apply_row_perm_panel_kernel(const half* __restrict__ A,
                                            half* __restrict__ PA,
                                            const int* __restrict__ row_perm,
                                            int m, int n)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列
    if (i >= m || j >= n) return;

    int src = row_perm[i];
    PA[i + j * m] = A[src + j * m];
}

// ======================================================================
// 5. R = A - B (half)
// ======================================================================
__global__ void diff_kernel(const half* __restrict__ A,
                            const half* __restrict__ B,
                            half* __restrict__ C,
                            int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = __hsub(A[idx], B[idx]);  // C = A - B
    }
}

// ======================================================================
// 6. 在 host 上用 double 计算 Frobenius 范数（量尺）
// ======================================================================
double frob_norm_half_host(const std::vector<half>& hA, int m, int n)
{
    long double sum = 0.0L;
    size_t N = (size_t)m * n;
    for (size_t i = 0; i < N; ++i) {
        double v = (double)__half2float(hA[i]);
        sum += v * v;
    }
    return std::sqrt((double)sum);
}

// ======================================================================
// 7. 主程序：panel = m x ib，ib = panel 宽度（<= n）
//    我们只 factor 前 ib 列，误差也只对这块 panel 测
// ======================================================================
//
// args: m n ib uc iters
// 约定：当前版本常用 ib == n 的情况（整个 m×n 作为 panel）
//
int main(int argc, char** argv)
{
    int m     = 8192;  // 行数
    int n     = 64;    // 矩阵总列数（这里只用前 ib 列）
    int ib    = -1;    // panel 宽度
    int uc    = 8;     // 更新 tile 宽度
    int iters = 20;    // 重复次数

    if (argc >= 2) m  = std::atoi(argv[1]);
    if (argc >= 3) n  = std::atoi(argv[2]);
    if (argc >= 4) ib = std::atoi(argv[3]);
    if (argc >= 5) uc = std::atoi(argv[4]);
    if (argc >= 6) iters = std::atoi(argv[5]);

    if (ib <= 0) ib = n;
    if (ib > n)  ib = n; // 防一下乱填
    int panel_n = ib;    // panel 的列数
    int lda     = m;

    std::cout << "Testing panel_TSLU kernel (half-based, panel m x ib)\n";
    std::cout << "  m = " << m
              << ", n = " << n
              << ", ib = " << ib
              << ", uc = " << uc
              << ", iters = " << iters << "\n";

    // ---------- 生成原始 half 矩阵 A0_panel (m x panel_n) ----------
    half* dA0 = nullptr;
    CUDA_CHECK(cudaMalloc(&dA0, (size_t)m * panel_n * sizeof(half)));

    unsigned long long seed = 12345678ULL;
    generate_normal_half_matrix(dA0, m, panel_n, seed);

    // 拷一份 A0_panel 回 host，算 ||A||_F（panel 部分）
    std::vector<half> hA0((size_t)m * panel_n);
    CUDA_CHECK(cudaMemcpy(hA0.data(), dA0,
                          (size_t)m * panel_n * sizeof(half),
                          cudaMemcpyDeviceToHost));
    double normA = frob_norm_half_host(hA0, m, panel_n);

    // ---------- 设备端 buffers ----------
    half* dA_fact = nullptr;   // panel 上做 LU 的矩阵 (m x panel_n)
    half* dL      = nullptr;   // L: m x panel_n
    half* dU      = nullptr;   // U: panel_n x panel_n
    half* dPA     = nullptr;   // PA: m x panel_n
    half* dLU     = nullptr;   // LU: m x panel_n
    half* dR      = nullptr;   // R: m x panel_n

    CUDA_CHECK(cudaMalloc(&dA_fact, (size_t)m * panel_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dL,      (size_t)m * panel_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dU,      (size_t)panel_n * panel_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dPA,     (size_t)m * panel_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dLU,     (size_t)m * panel_n * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dR,      (size_t)m * panel_n * sizeof(half)));

    int* d_ipiv_rel = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ipiv_rel, panel_n * sizeof(int)));

    // ---------- 计时：多次 panel LU 以测性能 ----------
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // 预热
    CUDA_CHECK(cudaMemcpy(dA_fact, dA0,
                          (size_t)m * panel_n * sizeof(half),
                          cudaMemcpyDeviceToDevice));
    launch_panel_TSLU(
        dA_fact,
        m, lda,
        /*j0=*/0, ib, uc,
        d_ipiv_rel,
        /*stream=*/0,
        /*block_dim=*/256
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < iters; ++it) {
        CUDA_CHECK(cudaMemcpy(dA_fact, dA0,
                              (size_t)m * panel_n * sizeof(half),
                              cudaMemcpyDeviceToDevice));

        launch_panel_TSLU(
            dA_fact,
            m, lda,
            /*j0=*/0, ib, uc,
            d_ipiv_rel,
            /*stream=*/0,
            /*block_dim=*/256
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    double avg_ms = ms / iters;

    // Panel LU FLOPs: 2*m*ib^2 - 2/3*ib^3
    double md = (double)m;
    double id = (double)ib;
    double flops = 2.0 * md * id * id - (2.0 / 3.0) * id * id * id;
    if (flops < 0.0) flops = 0.0;
    double gflops = (flops * 1e-9) / (avg_ms * 1e-3);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nPerformance:\n";
    std::cout << "  Avg time per factorization: " << avg_ms << " ms\n";
    std::cout << "  Estimated FLOPs: " << (flops * 1e-9) << " GFLOP\n";
    std::cout << "  Estimated performance: " << gflops << " GFLOP/s\n";

    // ---------- 再做一次 panel LU，用于误差计算 ----------
    CUDA_CHECK(cudaMemcpy(dA_fact, dA0,
                          (size_t)m * panel_n * sizeof(half),
                          cudaMemcpyDeviceToDevice));

    launch_panel_TSLU(
        dA_fact,
        m, lda,
        /*j0=*/0, ib, uc,
        d_ipiv_rel,
        /*stream=*/0,
        /*block_dim=*/256
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 抽取 L(m x ib), U(ib x ib) ----------
    dim3 block2(16, 16);
    dim3 grid2((panel_n + block2.x - 1) / block2.x,
               (m        + block2.y - 1) / block2.y);
    extract_LU_panel_kernel<<<grid2, block2>>>(dA_fact, dL, dU, m, panel_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 从 ipiv_rel 构造长度 m 的 row_perm（host） ----------
    std::vector<int> h_ipiv_rel(panel_n);
    CUDA_CHECK(cudaMemcpy(h_ipiv_rel.data(), d_ipiv_rel,
                          panel_n * sizeof(int),
                          cudaMemcpyDeviceToHost));

    std::vector<int> h_row_perm;
    build_row_perm_from_ipiv(h_ipiv_rel, h_row_perm, m, panel_n);

    int* d_row_perm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_row_perm, m * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_row_perm, h_row_perm.data(),
                          m * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ---------- 构造 PA_panel: 对 half A0_panel 应用置换 ----------
    dim3 blockPA(16, 16);
    dim3 gridPA((panel_n + blockPA.x - 1) / blockPA.x,
                (m        + blockPA.y - 1) / blockPA.y);
    apply_row_perm_panel_kernel<<<gridPA, blockPA>>>(
        dA0, dPA, d_row_perm, m, panel_n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 用 cuBLAS half GEMM 计算 LU_panel = L * U ----------
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // half 输入/输出，FP32 accumulate（H100 上的主打配置）
    cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m,           // rows of C
        panel_n,     // cols of C
        panel_n,     // inner dim
        &alpha,
        dL, CUDA_R_16F, m,
        dU, CUDA_R_16F, panel_n,
        &beta,
        dLU, CUDA_R_16F, m,
        computeType,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // ---------- R_panel = PA_panel - LU_panel (half) ----------
    int Npanel = m * panel_n;
    int blockR = 256;
    int gridR  = (Npanel + blockR - 1) / blockR;
    diff_kernel<<<gridR, blockR>>>(dPA, dLU, dR, Npanel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---------- 拷回 host，算 ||PA-LU||_F 和相对误差 ----------
    std::vector<half> hR((size_t)Npanel);
    CUDA_CHECK(cudaMemcpy(hR.data(), dR,
                          (size_t)Npanel * sizeof(half),
                          cudaMemcpyDeviceToHost));

    double normRes = frob_norm_half_host(hR, m, panel_n);
    double relErr  = normRes / normA;

    std::cout << "\nAccuracy (Frobenius norm, half-based panel m x ib):\n";
    std::cout << "  ||A_panel||_F              = " << normA   << "\n";
    std::cout << "  ||PA_panel - L*U||_F        = " << normRes << "  (绝对误差)\n";
    std::cout << "  ||PA_panel - L*U||_F / ||A_panel||_F = "
              << relErr << "  (相对误差)\n";

    // ---------- 清理 ----------
    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_row_perm));
    CUDA_CHECK(cudaFree(dR));
    CUDA_CHECK(cudaFree(dLU));
    CUDA_CHECK(cudaFree(dPA));
    CUDA_CHECK(cudaFree(dU));
    CUDA_CHECK(cudaFree(dL));
    CUDA_CHECK(cudaFree(dA_fact));
    CUDA_CHECK(cudaFree(dA0));
    CUDA_CHECK(cudaFree(d_ipiv_rel));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
