// test_cusolver_style.cu - 测试 cuSOLVER 风格的 hgetrf 接口
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <random>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "hgetrf.cuh"

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

// 生成随机矩阵
void generate_random_matrix(std::vector<half>& hA, int m, int n) {
    std::mt19937 gen(12345);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float val = dist(gen);
            hA[i + j * m] = __float2half(val);
        }
    }
}

int main(int argc, char** argv) {
    // 解析参数
    int m = 8192;
    int n = 8192;
    int iters = 10;
    int nb = 128;  // panel width
    
    if (argc >= 2) m = std::atoi(argv[1]);
    if (argc >= 3) n = std::atoi(argv[2]);
    if (argc >= 4) iters = std::atoi(argv[3]);
    if (argc >= 5) nb = std::atoi(argv[4]);
    
    printf("========================================\n");
    printf("cuSOLVER 风格接口测试\n");
    printf("========================================\n");
    printf("矩阵大小: %d x %d\n", m, n);
    printf("Panel width: %d\n", nb);
    printf("迭代次数: %d\n", iters);
    printf("========================================\n\n");
    
    // 创建 cuBLAS handle
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    // 生成测试矩阵
    std::vector<half> hA(m * n);
    generate_random_matrix(hA, m, n);
    
    // 分配设备内存
    half *dA = nullptr;
    int *d_ipiv = nullptr;
    int *d_info = nullptr;
    
    CUDA_CHECK(cudaMalloc(&dA, sizeof(half) * m * n));
    CUDA_CHECK(cudaMalloc(&d_ipiv, sizeof(int) * std::min(m, n)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    
    // 拷贝数据到设备
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * m * n, cudaMemcpyHostToDevice));
    
    // 预热
    printf("预热中...\n");
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * m * n, cudaMemcpyHostToDevice));
        int ret = hgetrf(handle, m, n, dA, m, d_ipiv, d_info, 0, nb);
        if (ret != 0) {
            printf("hgetrf 返回错误: %d\n", ret);
            return 1;
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("预热完成！\n\n");
    
    // 性能测试
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    printf("性能测试中（%d 次迭代）...\n", iters);
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeof(half) * m * n, cudaMemcpyHostToDevice));
        int ret = hgetrf(handle, m, n, dA, m, d_ipiv, d_info, 0, nb);
        if (ret != 0) {
            printf("迭代 %d: hgetrf 返回错误: %d\n", i, ret);
            return 1;
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    
    float avg_ms = total_ms / iters;
    double flops = (2.0 / 3.0) * n * n * m;
    double gflops = (flops * 1e-9) / (avg_ms * 1e-3);
    
    printf("\n========================================\n");
    printf("性能结果\n");
    printf("========================================\n");
    printf("平均时间:   %.4f ms\n", avg_ms);
    printf("性能:       %.2f GFLOP/s (%.2f TFLOP/s)\n", gflops, gflops / 1000.0);
    printf("========================================\n");
    
    // 验证正确性（简单检查 info）
    int h_info = -999;
    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    printf("\ninfo = %d (0 = 成功)\n", h_info);
    
    // 清理
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(d_ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    printf("\n测试完成！\n");
    
    return 0;
}




