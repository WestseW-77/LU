#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <cstdio>
#include <cmath>
#include <cstring>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "getrf.cuh"
#include "inc/hgetrf.cuh"

#define SWAP_LEN_PANEL 2048

using data_type = __half;

// TRSM 路线：0=原有 half->float->half 的 cublasStrsm
// 1=inv(L) + GEMM (类似 hpotrf 的 TRSM 处理，减少 B 的类型转换)
#ifndef GETRF_TRSM_USE_INV_GEMM
#define GETRF_TRSM_USE_INV_GEMM 0
#endif

// inv+GEMM 路线的列分块大小
#ifndef GETRF_TRSM_TILE
#define GETRF_TRSM_TILE 4096
#endif

// ============================================
// 第一部分：基础工具函数
// ============================================

/**
 * 矩阵逐元素减法: C = A - B
 * 用途：验证时计算 PA - LU 的差值
 * 
 * @param A 输入矩阵A
 * @param B 输入矩阵B
 * @param C 输出矩阵C = A - B
 * @param N 矩阵元素总数
 */
template <typename T>
__global__ void elementwise_sub_kernel(const T* __restrict__ A,
                                       const T* __restrict__ B,
                                       T* __restrict__ C,
                                       int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] - B[idx];
    }
}

/**
 * 类型转换：half -> float (2D矩阵)
 * 用途：TRSM前将half数据转换为float，提高精度和性能
 * 
 * @param src 源half矩阵
 * @param src_ld 源矩阵leading dimension
 * @param dst 目标float矩阵
 * @param dst_ld 目标矩阵leading dimension
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
__global__ void half_to_float_2d_kernel(const half* __restrict__ src, int src_ld,
                                        float* __restrict__ dst, int dst_ld,
                                        int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        dst[i + j * (size_t)dst_ld] = __half2float(src[i + (size_t)j * src_ld]);
    }
}

/**
 * 类型转换：float -> half (2D矩阵)
 * 用途：TRSM后将float结果转回half存储
 * 
 * @param src 源float矩阵
 * @param src_ld 源矩阵leading dimension
 * @param dst 目标half矩阵
 * @param dst_ld 目标矩阵leading dimension
 * @param rows 矩阵行数
 * @param cols 矩阵列数
 */
__global__ void float_to_half_2d_kernel(const float* __restrict__ src, int src_ld,
                                        half* __restrict__ dst, int dst_ld,
                                        int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < cols) {
        dst[i + (size_t)j * dst_ld] = __float2half(src[i + (size_t)j * src_ld]);
    }
}

// ============================================
// 第二部分：TRSM Float工作空间管理
// ============================================

/**
 * TRSM Float 工作空间
 * 用途：存储转换后的float数据，避免频繁分配/释放内存
 * 
 * 设计原理：
 * - 一次性分配足够大的空间
 * - 在整个LU分解过程中重复使用
 * - 避免每次TRSM都重新分配内存的开销
 */
struct TrsmFloatWorkspace {
    float* A_float = nullptr;  // 存储转换后的A矩阵（L矩阵）
    float* B_float = nullptr;  // 存储转换后的B矩阵（待求解矩阵）
    float* inv_float = nullptr; // 存储 inv(L) (float)
    half*  inv_half = nullptr;  // 存储 inv(L) (half)
    half*  B_half_tmp = nullptr; // GEMM 临时块 (half)
    int max_m = 0;             // 最大行数
    int max_ncols = 0;         // 最大列数
    int ldA = 0;               // A的leading dimension
    int ldB = 0;               // B的leading dimension
    int trsm_tile = GETRF_TRSM_TILE;

    /**
     * 分配工作空间
     * @param max_m_ 最大行数（取决于矩阵规模和分块大小）
     * @param max_ncols_ 最大列数（取决于k和nb参数）
     */
    void alloc(int max_m_, int max_ncols_) {
        max_m = max_m_;
        max_ncols = max_ncols_;
        ldA = max_m;
        ldB = max_m;

        // 分配A矩阵空间（方阵）
        CUDA_CHECK(cudaMalloc(&A_float, (size_t)max_m * (size_t)max_m * sizeof(float)));
        // 分配B矩阵空间（可能是矩形）
        CUDA_CHECK(cudaMalloc(&B_float, (size_t)max_m * (size_t)max_ncols * sizeof(float)));
        // inv(L) 空间（float/half）
        CUDA_CHECK(cudaMalloc(&inv_float, (size_t)max_m * (size_t)max_m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&inv_half,  (size_t)max_m * (size_t)max_m * sizeof(half)));

        int tile = trsm_tile;
        if (tile <= 0) tile = max_ncols;
        if (tile > max_ncols) tile = max_ncols;
        trsm_tile = tile;
        if (trsm_tile < 1) trsm_tile = 1;
        CUDA_CHECK(cudaMalloc(&B_half_tmp, (size_t)max_m * (size_t)trsm_tile * sizeof(half)));
    }

    /**
     * 释放工作空间
     */
    void free() {
        if (A_float) CUDA_CHECK(cudaFree(A_float));
        if (B_float) CUDA_CHECK(cudaFree(B_float));
        if (inv_float) CUDA_CHECK(cudaFree(inv_float));
        if (inv_half) CUDA_CHECK(cudaFree(inv_half));
        if (B_half_tmp) CUDA_CHECK(cudaFree(B_half_tmp));
        A_float = nullptr;
        B_float = nullptr;
        inv_float = nullptr;
        inv_half = nullptr;
        B_half_tmp = nullptr;
    }
};

/**
 * TRSM实现（使用cuBLAS float + 递归分块）
 * 
 * 功能：求解 L * X = B，其中L是单位下三角矩阵
 * 
 * 算法原理：
 * 1. 基础情况（m <= nb）：直接调用cuBLAS的float版本TRSM
 *    - 先将half数据转换为float
 *    - 调用cublasStrsm求解
 *    - 将结果转回half
 * 
 * 2. 递归情况（m > nb）：使用分治策略
 *    矩阵分块：[L11  0 ] [X1]   [B1]
 *              [L21 L22] [X2] = [B2]
 *    步骤：
 *    a) 求解 L11 * X1 = B1
 *    b) 更新 B2 = B2 - L21 * X1
 *    c) 求解 L22 * X2 = B2
 * 
 * @param cublasH cuBLAS句柄
 * @param m 矩阵L的维度（m×m）
 * @param ncols B矩阵的列数
 * @param A L矩阵（单位下三角，half格式）
 * @param lda A的leading dimension
 * @param B 待求解矩阵（输入为B，输出为X，half格式）
 * @param ldb B的leading dimension
 * @param nb 基础情况的阈值（递归终止条件）
 * @param ws 工作空间（用于half<->float转换）
 */
static void trsm_float_recursive(
    cublasHandle_t cublasH, int m, int ncols,
    const half* A, int lda, half* B, int ldb,
    int nb, TrsmFloatWorkspace& ws) {

    if (m <= 0 || ncols <= 0) return;

    // 基础情况：m <= nb
    if (m <= nb) {
#if GETRF_TRSM_USE_INV_GEMM
        if (ws.inv_float && ws.inv_half && ws.B_half_tmp) {
            cudaStream_t stream = 0;
            CUBLAS_CHECK(cublasGetStream(cublasH, &stream));
            dim3 blockDim(16, 16);
            dim3 gridDim_A((m + 15) / 16, (m + 15) / 16);

            // L -> float
            half_to_float_2d_kernel<<<gridDim_A, blockDim, 0, stream>>>(
                A, lda, ws.A_float, ws.ldA, m, m);
            CUDA_CHECK(cudaGetLastError());

            // inv_float = I
            int total = m * m;
            int block = 256;
            int grid = std::min(1024, (total + block - 1) / block);
            set_identity_f<<<grid, block, 0, stream>>>(
                ws.inv_float, ws.ldA, m);
            CUDA_CHECK(cudaGetLastError());

            // inv_float = inv(L) (diag=unit)
            float alpha = 1.0f;
            CUBLAS_CHECK(cublasStrsm(
                cublasH,
                CUBLAS_SIDE_LEFT,
                CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N,
                CUBLAS_DIAG_UNIT,
                m, m, &alpha,
                ws.A_float, ws.ldA,
                ws.inv_float, ws.ldA));

            // inv_half
            float_to_half_2d_kernel<<<gridDim_A, blockDim, 0, stream>>>(
                ws.inv_float, ws.ldA, ws.inv_half, ws.ldA, m, m);
            CUDA_CHECK(cudaGetLastError());

            // B = inv(L) * B (tile by columns to avoid aliasing)
            int tile = ws.trsm_tile;
            if (tile <= 0) tile = ncols;
            if (tile > ncols) tile = ncols;
            if (tile < 1) tile = 1;

            for (int col = 0; col < ncols; col += tile) {
                int nc = std::min(tile, ncols - col);
                half* B_tile = B + (size_t)col * ldb;

                CUDA_CHECK(cudaMemcpy2DAsync(
                    ws.B_half_tmp, (size_t)ws.ldA * sizeof(half),
                    B_tile,        (size_t)ldb * sizeof(half),
                    (size_t)m * sizeof(half),
                    (size_t)nc,
                    cudaMemcpyDeviceToDevice, stream));

                float alpha_g = 1.0f;
                float beta_g  = 0.0f;
                CUBLAS_CHECK(cublasGemmEx(
                    cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, nc, m,
                    &alpha_g,
                    ws.inv_half,   CUDA_R_16F, ws.ldA,
                    ws.B_half_tmp, CUDA_R_16F, ws.ldA,
                    &beta_g,
                    B_tile, CUDA_R_16F, ldb,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }

            return;
        }
#endif

        dim3 blockDim(16, 16);
        dim3 gridDim_A((m + 15) / 16, (m + 15) / 16);
        dim3 gridDim_B((m + 15) / 16, (ncols + 15) / 16);

        // 步骤1：half -> float 转换
        half_to_float_2d_kernel<<<gridDim_A, blockDim>>>(
            A, lda, ws.A_float, ws.ldA, m, m);
        half_to_float_2d_kernel<<<gridDim_B, blockDim>>>(
            B, ldb, ws.B_float, ws.ldB, m, ncols);
        CUDA_CHECK(cudaGetLastError());

        // 步骤2：调用cuBLAS TRSM（float版本）
        // 求解 L * X = B，L是单位下三角（CUBLAS_DIAG_UNIT）
        float alpha = 1.0f;
        CUBLAS_CHECK(cublasStrsm(
            cublasH, 
            CUBLAS_SIDE_LEFT,        // L在左侧
            CUBLAS_FILL_MODE_LOWER,  // L是下三角
            CUBLAS_OP_N,             // 不转置
            CUBLAS_DIAG_UNIT,        // 单位对角线
            m, ncols, &alpha,
            ws.A_float, ws.ldA,
            ws.B_float, ws.ldB));

        // 步骤3：float -> half 转换
        float_to_half_2d_kernel<<<gridDim_B, blockDim>>>(
            ws.B_float, ws.ldB, B, ldb, m, ncols);
        CUDA_CHECK(cudaGetLastError());

        return;
    }

    // 递归情况：将矩阵分为四块
    int m1 = m / 2;      // 上半部分的行数
    int m2 = m - m1;     // 下半部分的行数

    // 分块指针
    const half* L11 = A;                                    // 左上块
    const half* L21 = A + m1;                               // 左下块
    const half* L22 = A + m1 + (size_t)m1 * lda;           // 右下块

    half* B1 = B;                                           // B的上半部分
    half* B2 = B + m1;                                      // B的下半部分

    // 步骤1：求解 L11 * X1 = B1（递归调用）
    trsm_float_recursive(cublasH, m1, ncols, L11, lda, B1, ldb, nb, ws);

    // 步骤2：更新 B2 = B2 - L21 * X1
    // 使用cuBLAS GEMM进行矩阵乘法和减法
    const float alpha = -1.0f;  // 负号用于减法
    const float beta  = 1.0f;   // 保留B2的原值
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        m2, ncols, m1,
        &alpha, L21, CUDA_R_16F, lda,  // L21: m2×m1
        B1,  CUDA_R_16F, ldb,           // X1:  m1×ncols
        &beta,  B2,  CUDA_R_16F, ldb,   // B2:  m2×ncols
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // 步骤3：求解 L22 * X2 = B2（递归调用）
    trsm_float_recursive(cublasH, m2, ncols, L22, lda, B2, ldb, nb, ws);
}

/**
 * TRSM预热函数
 * 
 * 目的：
 * - 触发cuBLAS的延迟初始化
 * - 编译和加载类型转换kernel
 * - 避免首次调用的额外开销污染性能测试
 * 
 * 策略：
 * - 使用真实会遇到的矩阵规模（panel和kpanel）
 * - 每种规模运行2次，确保充分预热
 * 
 * @param cublasH cuBLAS句柄
 * @param trsm_ws TRSM工作空间
 * @param n 矩阵总维度
 * @param k 外层分块大小
 * @param nb 内层分块大小（也是float TRSM的基础情况阈值）
 */
static void warmup_trsm(
    cublasHandle_t cublasH,
    TrsmFloatWorkspace& trsm_ws,
    int n, int k, int nb) {
    
    int kk = std::min(k, n);
    int bb = std::min(nb, n);

    // 预热两种典型规模：
    // 1. panel规模 (bb×bb)：对应内层分块
    // 2. kpanel规模 (kk×kk)：对应外层分块
    const int shapes[2][2] = {
        {bb, bb},    // panel
        {kk, kk}     // kpanel
    };

    for (int s = 0; s < 2; ++s) {
        const int m = shapes[s][0];
        const int ncols = shapes[s][1];
        if (m <= 0 || ncols <= 0) continue;

        // 使用紧凑布局（lda = ldb = m）
        const int lda = m;
        const int ldb = m;

        // 分配临时测试矩阵
        thrust::device_vector<half> L_d((size_t)lda * (size_t)m);
        thrust::device_vector<half> B_d((size_t)ldb * (size_t)ncols);

        // 填充随机数据
        generateNormalMatrix(L_d, m, m);
        generateNormalMatrix(B_d, m, ncols);

        // 运行2次确保充分预热
        for (int rep = 0; rep < 2; ++rep) {
            trsm_float_recursive(
                cublasH,
                m, ncols,
                (const half*)thrust::raw_pointer_cast(L_d.data()), lda,
                (half*)thrust::raw_pointer_cast(B_d.data()), ldb,
                nb,
                trsm_ws);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// ============================================
// 第三部分：行交换Kernel（用于主元选取后的行置换）
// ============================================

/**
 * 行交换Kernel（使用cooperative groups实现全局同步）
 * 
 * 功能：根据主元信息批量交换矩阵行
 * 
 * 算法：
 * 1. 第一阶段：将source行复制到临时缓冲区
 * 2. 全局同步（确保所有复制完成）
 * 3. 第二阶段：将临时缓冲区数据写到target行
 * 
 * 为什么需要两阶段：
 * - 避免读写冲突（如果行i和行j需要互换）
 * - 确保数据一致性
 * 
 * @tparam threadsPerRow 每行使用的线程数
 * @param source 源行索引数组
 * @param target 目标行索引数组
 * @param swap_size 需要交换的行对数量
 * @param swap_len 每行需要交换的元素数量
 * @param stride 矩阵的列跨度（leading dimension）
 * @param A_d 矩阵数据
 * @param temp 临时缓冲区
 */
template <size_t threadsPerRow>
__global__ void swapByPivotingKernel(int* __restrict__ source,
                                     int* __restrict__ target,
                                     int swap_size,
                                     int swap_len,
                                     int stride,
                                     half* __restrict__ A_d,
                                     half* __restrict__ temp) {
    auto grid = cooperative_groups::this_grid();
    auto gridStride = (int)(blockDim.x * gridDim.x);

    // 第一阶段：复制source行到临时缓冲区
    for (int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         idx < swap_size * (int)threadsPerRow;
         idx += gridStride) {
        const int row_idx = idx / (int)threadsPerRow;
        const int lane = idx % (int)threadsPerRow;
        for (int j = lane; j < swap_len; j += (int)threadsPerRow) {
            const int src_offset = source[row_idx] + j * stride;
            temp[row_idx * swap_len + j] = A_d[src_offset];
        }
    }
    
    // 全局同步：确保所有source行都已复制完成
    grid.sync();

    // 第二阶段：将临时缓冲区数据写到target行
    for (int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
         idx < swap_size * (int)threadsPerRow;
         idx += gridStride) {
        const int row_idx = idx / (int)threadsPerRow;
        const int lane = idx % (int)threadsPerRow;
        for (int j = lane; j < swap_len; j += (int)threadsPerRow) {
            const int tgt_offset = target[row_idx] + j * stride;
            A_d[tgt_offset] = temp[row_idx * swap_len + j];
        }
    }
}

/**
 * 启动行交换kernel的包装函数
 * 
 * 功能：处理不同的swap_len，自动分块启动kernel
 * 
 * 策略：
 * - swap_len < SWAP_LEN_PANEL：一次性处理
 * - swap_len >= SWAP_LEN_PANEL：分块处理（避免临时缓冲区过大）
 * 
 * @param source_raw_ptr 源行索引指针
 * @param target_raw_ptr 目标行索引指针
 * @param swap_size 交换行对数量
 * @param swap_len 每行交换的元素数
 * @param stride 矩阵列跨度
 * @param A_d 矩阵数据
 * @param temp_d 临时缓冲区
 * @param blocksPerGrid 网格大小
 * @param threadsPerBlock 块大小
 */
static void launchSwapByPivotingKernel(int* source_raw_ptr,
                                      int* target_raw_ptr,
                                      int swap_size,
                                      int swap_len,
                                      int stride,
                                      half* A_d,
                                      half* temp_d,
                                      int blocksPerGrid,
                                      int threadsPerBlock) {
    if (swap_size <= 0 || swap_len <= 0) return;

    if (swap_len < SWAP_LEN_PANEL) {
        // 小规模：一次性处理所有列
        void* args[] = {
            (void*)&source_raw_ptr, (void*)&target_raw_ptr,
            (void*)&swap_size, (void*)&swap_len,
            (void*)&stride, (void*)&A_d, (void*)&temp_d
        };
        using KernelType = decltype(&swapByPivotingKernel<16>);
        KernelType kernel_ptr = &swapByPivotingKernel<16>;
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_ptr,
                                               blocksPerGrid,
                                               threadsPerBlock,
                                               args));
    } else {
        // 大规模：分块处理，每块最多处理swap_len_panel列
        for (int i = 0; i < swap_len;
             i += (swap_len + threadsPerBlock - 1) / threadsPerBlock) {
            auto each_swap_len = (swap_len + threadsPerBlock - 1) / threadsPerBlock;
            auto swap_len_panel = std::min(swap_len - i, each_swap_len);
            auto A_d_panel = A_d + (size_t)i * stride;

            void* args[] = {
                (void*)&source_raw_ptr, (void*)&target_raw_ptr,
                (void*)&swap_size, (void*)&swap_len_panel,
                (void*)&stride, (void*)&A_d_panel, (void*)&temp_d
            };
            using KernelType = decltype(&swapByPivotingKernel<16>);
            KernelType kernel_ptr = &swapByPivotingKernel<16>;
            CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kernel_ptr,
                                                   blocksPerGrid,
                                                   threadsPerBlock,
                                                   args));
        }
    }
}

// ============================================
// 第四部分：辅助函数
// ============================================

/**
 * 命令行参数解析
 * 
 * 支持的格式：
 * - ./program <n> : 简化模式，k=n/2, nb=k/2
 * - ./program <n> <k> <nb> : 完整指定
 * - ./program <n> <k> <nb> [-v] [-p] [-d] [-c] : 带选项
 * 
 * 选项说明：
 * -v : 验证结果（计算|PA-LU|）
 * -p : 启用主元选取（默认启用）
 * -d : 调试模式
 * -c : 与cuSOLVER对比
 */
int parseArgs(int argc, char *argv[], size_t &n, size_t &k, size_t &nb,
              bool &verifyResult, bool &pivoting, bool &debug_mode,
              bool &compare_with_cusolver) {
    if (argc == 2) {
        n = std::stoul(argv[1]);
        k = n / 2;
        nb = k / 2;
        compare_with_cusolver = true;
        pivoting = true;
    } else if (argc == 4) {
        n = std::stoul(argv[1]);
        k = std::stoul(argv[2]);
        nb = std::stoul(argv[3]);
    } else if (argc > 4) {
        n = std::stoul(argv[1]);
        k = std::stoul(argv[2]);
        nb = std::stoul(argv[3]);
        for (int i = 4; i < argc; i++) {
            if (strcmp(argv[i], "-v") == 0) {
                verifyResult = true;
            } else if (strcmp(argv[i], "-p") == 0) {
                pivoting = true;
            } else if (strcmp(argv[i], "-d") == 0) {
                debug_mode = true;
            } else if (strcmp(argv[i], "-c") == 0) {
                compare_with_cusolver = true;
            }
        }
    } else {
        printf("Usage: %s <n> <k> <nb> [-v] [-p] [-d] [-c]\n", argv[0]);
        return 1;
    }

    pivoting = true;  // 默认启用主元选取
    return 0;
}

/**
 * 计算GETRF的理论浮点操作数
 * LU分解的复杂度：(2/3) * n^3
 */
static inline double getrf_flops(size_t n) {
    return (2.0 / 3.0) * std::pow((double)n, 3);
}

/**
 * 与cuSOLVER baseline对比
 * 
 * 用途：评估自定义实现相对于NVIDIA官方库的性能
 * 
 * 步骤：
 * 1. 将half数据转换为float（cuSOLVER只支持float/double）
 * 2. 调用cusolverDnSgetrf执行LU分解
 * 3. 测量时间并计算TFLOPS
 */
static void compareWithCusolver_half(const thrust::device_vector<__half>& A_seed,
                                     size_t n, int roll_num, bool pivoting) {
    printf("\n========== cuSOLVER Baseline ==========\n");

    // 创建cuSOLVER句柄
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // 分配float工作空间
    thrust::device_vector<float> A_float(n * n);
    int* dinfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dinfo, sizeof(int)));

    // 查询工作空间大小
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        cusolver_handle, (int)n, (int)n,
        thrust::raw_pointer_cast(A_float.data()), (int)n, &lwork));

    // 分配工作空间和主元数组
    float* dwork = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dwork, (size_t)lwork * sizeof(float)));

    int* dpiv = nullptr;
    if (pivoting) {
        CUDA_CHECK(cudaMalloc((void**)&dpiv, n * sizeof(int)));
    }

    // 类型转换kernel配置
    int block = 256;
    int grid = (int)((n * n + block - 1) / block);

    // 性能测试
    float cusolver_lu_time = 0.0f;
    for (int i = 0; i < roll_num; i++) {
        // half -> float转换
        half_to_float_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(A_seed.data()),
            thrust::raw_pointer_cast(A_float.data()), (int)(n * n));
        CUDA_CHECK(cudaDeviceSynchronize());

        // 执行LU分解并计时
        startTimer();
        CUSOLVER_CHECK(cusolverDnSgetrf(
            cusolver_handle, (int)n, (int)n,
            thrust::raw_pointer_cast(A_float.data()), (int)n,
            dwork, dpiv, dinfo));
        cusolver_lu_time += stopTimer();
    }

    // 输出性能结果
    printf("cuSOLVER time: %.2f ms\n", cusolver_lu_time / roll_num);
    printf("cuSOLVER: %.2f TFLOPS\n",
           (float)(getrf_flops(n) / ((cusolver_lu_time / roll_num) * 1e9)));

    // 清理资源
    if (dpiv) CUDA_CHECK(cudaFree(dpiv));
    if (dwork) CUDA_CHECK(cudaFree(dwork));
    if (dinfo) CUDA_CHECK(cudaFree(dinfo));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

/**
 * 验证LU分解结果的正确性
 * 
 * 原理：
 * - LU分解：PA = LU（其中P是置换矩阵）
 * - 验证方法：计算 |PA - LU| / |A|（相对误差）
 * 
 * 步骤：
 * 1. 计算 PA（置换后的原矩阵）
 * 2. 从分解结果中提取L（单位下三角）和U（上三角）
 * 3. 计算 LU
 * 4. 计算 PA - LU
 * 5. 计算Frobenius范数并输出相对误差
 */
static void computeMinusOfPAandLU_half(thrust::device_vector<__half>& A_device_vector,
                                       thrust::device_vector<__half>& oriA_device_vector,
                                       thrust::device_vector<__half>& P_device_vector,
                                       int n) {
    // 创建cuBLAS句柄（用于矩阵乘法）
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // 计算原矩阵的Frobenius范数
    auto A_norm = computeFrobeniusNorm(n, n, oriA_device_vector);

    auto A_d    = (half*)thrust::raw_pointer_cast(A_device_vector.data());
    auto oriA_d = (half*)thrust::raw_pointer_cast(oriA_device_vector.data());
    auto P_d    = (half*)thrust::raw_pointer_cast(P_device_vector.data());

    // 步骤1：计算 PA = P * A
    thrust::device_vector<__half> PA_device_vector(n * n);
    auto PA_d = (half*)thrust::raw_pointer_cast(PA_device_vector.data());

    float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        (const half*)P_d, CUDA_R_16F, n,
        (const half*)oriA_d, CUDA_R_16F, n,
        &beta,
        (half*)PA_d, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    dim3 blockDim(16, 16);

    // 步骤2：提取L矩阵（单位下三角）
    thrust::device_vector<__half> L_device_vector = A_device_vector;
    cleanMatrix<<<gridDim, blockDim>>>(
        (half*)thrust::raw_pointer_cast(L_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, half* L_d, int lda) {
            half zero = __float2half(0.0f);
            half one  = __float2half(1.0f);
            if (i < n && j < n) {
                if (i < j) L_d[i + (size_t)j * lda] = zero;  // 上三角置0
                if (i == j) L_d[i + (size_t)j * lda] = one;   // 对角置1
            }
        });
    CUDA_CHECK(cudaGetLastError());

    // 步骤3：提取U矩阵（上三角）
    thrust::device_vector<__half> U_device_vector = A_device_vector;
    cleanMatrix<<<gridDim, blockDim>>>(
        (half*)thrust::raw_pointer_cast(U_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, half* U_d, int lda) {
            half zero = __float2half(0.0f);
            if (i < n && j < n) {
                if (i > j) U_d[i + (size_t)j * lda] = zero;  // 下三角置0
            }
        });
    CUDA_CHECK(cudaGetLastError());

    // 步骤4：计算 LU = L * U
    thrust::device_vector<__half> LU_device_vector(n * n);
    auto LU_d = (half*)thrust::raw_pointer_cast(LU_device_vector.data());

    alpha = 1.0f; beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
        n, n, n,
        &alpha,
        (const half*)thrust::raw_pointer_cast(L_device_vector.data()), CUDA_R_16F, n,
        (const half*)thrust::raw_pointer_cast(U_device_vector.data()), CUDA_R_16F, n,
        &beta,
        (half*)LU_d, CUDA_R_16F, n,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // 步骤5：计算 PA - LU
    thrust::device_vector<__half> PAminusLU_device_vector(n * n);
    auto PAminusLU_d = (half*)thrust::raw_pointer_cast(PAminusLU_device_vector.data());

    int64_t total = (int64_t)n * (int64_t)n;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    elementwise_sub_kernel<<<(unsigned int)grid, block>>>(
        (const half*)PA_d, (const half*)LU_d, (half*)PAminusLU_d, total);
    CUDA_CHECK(cudaGetLastError());

    // 步骤6：计算误差范数并输出结果
    double Minus_norm = computeFrobeniusNorm(n, n, PAminusLU_device_vector);
    printf("\n========== Verification ==========\n");
    printf("|PA - LU|: %.6e\n", Minus_norm);
    printf("|A|: %.6e\n", A_norm);
    printf("Relative error: %.6e\n", Minus_norm / A_norm);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

// ============================================
// 第五部分：主函数 - 双分块LU分解
// ============================================

/**
 * 性能统计结构
 * 用于记录各个阶段的耗时和计算量
 */
struct DoubleBlockingLUBreakDownDetail {
    float tslu_time = 0.0f;           // Panel LU分解时间
    float swap_panel_time = 0.0f;     // Panel行交换时间
    float trsm_panel_time = 0.0f;     // Panel TRSM时间
    float gemm_panel_time = 0.0f;     // Panel GEMM时间
    float gemm_panel_ops = 0.0f;      // Panel GEMM操作数
    float swap_kpanel_time = 0.0f;    // Kpanel行交换时间
    float trsm_kpanel_time = 0.0f;    // Kpanel TRSM时间
    float gemm_kpanel_ops = 0.0f;     // Kpanel GEMM操作数
    float gemm_kpanel_time = 0.0f;    // Kpanel GEMM时间
};

/**
 * 主函数：双分块LU分解
 * 
 * 算法概述：
 * 
 * 双分块策略（Two-level blocking）：
 * - 外层分块：以k为步长，将n×n矩阵分为多个k×k块
 * - 内层分块：在每个k×k块内，以nb为步长进一步细分
 * 
 * 为什么使用双分块：
 * 1. 外层分块（k）：减少全局通信，提高数据局部性
 * 2. 内层分块（nb）：适应GPU的线程块结构，优化并行度
 * 3. 两层结合：在内存层次和并行粒度上都达到最优
 * 
 * 算法流程（对于第i个外层块）：
 * 
 * A. 内层Panel处理（j从i到i+k，步长nb）：
 *    1. TSLU：对当前nb列进行LU分解
 *    2. 行交换：根据主元信息交换panel内的行
 *    3. TRSM：求解三角方程组 L*X=B
 *    4. GEMM：更新剩余矩阵块 C = C - A*B
 * 
 * B. 外层Kpanel更新：
 *    1. 行交换：应用所有panel的行交换到左侧已完成块
 *    2. TRSM：对整个k列进行三角求解
 *    3. GEMM：更新右下角剩余矩阵
 * 
 * 矩阵布局示意（第i步）：
 * 
 *     0    i    i+k    n
 *   ┌─────┬─────┬──────┐
 * 0 │Done │     │      │
 *   ├─────┼─────┼──────┤
 * i │     │Panel│ Rest │  <- 当前处理的k×k块
 *   │     │(nb) │      │
 *   ├─────┼─────┼──────┤
 * i+k│     │     │      │  <- 待更新的部分
 *   └─────┴─────┴──────┘
 */
int main(int argc, char *argv[]) {
    // ========================================
    // 第一阶段：初始化和参数解析
    // ========================================
    
    size_t n, k, nb;
    bool verifyResult = false;
    bool pivoting = false;
    bool debug_mode = false;
    bool compare_with_cusolver = false;

    // 解析命令行参数
    int status = parseArgs(argc, argv, n, k, nb, verifyResult, pivoting, debug_mode,
                           compare_with_cusolver);
    if (status != 0) return 1;

    // 打印配置信息
    printf("==========================================\n");
    printf("双分块LU分解 (Double-Blocking LU Factorization)\n");
    printf("==========================================\n");
    printf("矩阵规模: %zu×%zu\n", n, n);
    printf("分块参数: k=%zu (外层块), nb=%zu (内层块)\n", k, nb);
    printf("数据类型: half (FP16)\n");
    printf("TRSM策略: cuBLAS float (递归分块, base_case=%d)\n", 512);
    printf("==========================================\n\n");

    // 获取GPU信息
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU设备: %s\n", prop.name);

    // 配置行交换kernel的grid/block大小
    int maxBlocksPerGrid = prop.multiProcessorCount;
    int threadsPerBlock = 256;
    int blocksPerGrid = maxBlocksPerGrid;

    // ========================================
    // 第二阶段：内存分配和数据初始化
    // ========================================
    
    // 分配主矩阵（n×n，half精度）
    thrust::device_vector<data_type> A_device_vector(n * n);
    generateNormalMatrix(A_device_vector, n, n);

    // 保存原始矩阵（用于验证或对比）
    thrust::device_vector<data_type> A_seed_device_vector;
    if (verifyResult || compare_with_cusolver) {
        A_seed_device_vector = A_device_vector;
    }

    // 分配置换矩阵P和原始矩阵副本（仅验证时需要）
    thrust::device_vector<data_type> P_device_vector;
    thrust::device_vector<data_type> oriA_device_vector;
    half* P_d = nullptr;

    if (verifyResult) {
        // 初始化P为单位矩阵
        P_device_vector.resize(n * n);
        P_d = (half*)thrust::raw_pointer_cast(P_device_vector.data());
        CUDA_CHECK(cudaMemset(P_d, 0, sizeof(data_type) * n * n));
        thrust::for_each(thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n),
                         [P_d, n] __device__(size_t i) {
                             P_d[i * n + i] = __float2half_rn(1.0f);
                         });
        oriA_device_vector = A_device_vector;
    }

    // ========================================
    // 第三阶段：初始化CUDA库句柄
    // ========================================
    
    // 初始化cuBLAS（用于GEMM和TRSM）
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    // 初始化hgetrf（自定义的panel LU分解）
    hgetrfHandle_t hgetrf_handle = nullptr;
    hgetrfCreate(&hgetrf_handle);
    hgetrfSetStream(hgetrf_handle, 0);
    hgetrfSetPanelWidth(hgetrf_handle, 256);  // Panel宽度
    hgetrfSetUc(hgetrf_handle, 16);            // 展开系数

    // ========================================
    // 第四阶段：分配工作空间
    // ========================================
    
    // TRSM float工作空间（一次性分配，重复使用）
    TrsmFloatWorkspace trsm_ws;
    {
        int trsm_base = 512;  // float TRSM的基础情况阈值
        int max_ncols = (int)std::max(k, nb);
        trsm_ws.alloc(trsm_base, max_ncols);
    }

    // 临时缓冲区（用于行交换）
    thrust::device_vector<data_type> temp_device_vector(n * n);
    auto temp_d = (half*)thrust::raw_pointer_cast(temp_device_vector.data());

    // hgetrf工作空间
    int buffer_size = 0;
    auto A_raw_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
    hgetrf_bufferSize(hgetrf_handle, (int)n, (int)nb,
                      (const half*)A_raw_d, (int)n, &buffer_size);

    data_type* tslu_workspace_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&tslu_workspace_d, sizeof(data_type) * (size_t)buffer_size));

    int* devInfo_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo_d, sizeof(int)));

    // 主元索引数组（device和host）
    thrust::device_vector<int> ipiv_d_vector(n);
    auto ipiv_d = thrust::raw_pointer_cast(ipiv_d_vector.data());
    thrust::host_vector<int> ipiv_h_vector(n);

    // ========================================
    // 第五阶段：GPU预热
    // ========================================
    
    printf("正在预热GPU...\n");
    
    // 预热hgetrf（panel LU分解）
    for (int w = 0; w < 2; ++w) {
        if (verifyResult || compare_with_cusolver) {
            A_device_vector = A_seed_device_vector;
        } else {
            generateNormalMatrix(A_device_vector, n, n, (size_t)(w + 12345));
        }

        CUDA_CHECK(cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
        CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
        thrust::fill(ipiv_d_vector.begin(), ipiv_d_vector.end(), 0);

        half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
        hgetrf(hgetrf_handle, (int)n, (int)nb,
               (half*)(A_d + 0 + 0 * n), (int)n,
               (half*)tslu_workspace_d,
               ipiv_d + 0, devInfo_d);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // 预热TRSM（包括类型转换kernel和cuBLAS）
    warmup_trsm(cublas_handle, trsm_ws, (int)n, (int)k, 512);

    printf("预热完成\n\n");

    // ========================================
    // 第六阶段：性能测试
    // ========================================
    
    int roll_num = verifyResult ? 1 : 3;  // 验证时只运行1次，性能测试运行3次
    DoubleBlockingLUBreakDownDetail detail;

    // 创建CUDA事件用于计时
    cudaEvent_t total_lu_begin, total_lu_end;
    CUDA_CHECK(cudaEventCreate(&total_lu_begin));
    CUDA_CHECK(cudaEventCreate(&total_lu_end));

    printf("开始LU分解... (运行 %d 次)\n", roll_num);
    CUDA_CHECK(cudaEventRecord(total_lu_begin));

    // ========================================
    // 主循环：重复运行LU分解以获取稳定的性能数据
    // ========================================
    for (int roll = 0; roll < roll_num; roll++) {
        // 重置工作空间和数据
        CUDA_CHECK(cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
        CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
        thrust::fill(ipiv_d_vector.begin(), ipiv_d_vector.end(), 0);
        thrust::fill(ipiv_h_vector.begin(), ipiv_h_vector.end(), 0);

        if (verifyResult || compare_with_cusolver) {
            A_device_vector = A_seed_device_vector;
        } else {
            generateNormalMatrix(A_device_vector, n, n, (size_t)roll);
        }

        if (verifyResult) {
            // 重置置换矩阵P为单位矩阵
            CUDA_CHECK(cudaMemset(P_d, 0, sizeof(data_type) * n * n));
            thrust::for_each(thrust::counting_iterator<size_t>(0),
                             thrust::counting_iterator<size_t>(n),
                             [P_d, n] __device__(size_t i) {
                                 P_d[i * n + i] = __float2half_rn(1.0f);
                             });
            oriA_device_vector = A_device_vector;
        }

        // ========================================
        // 外层循环：以k为步长遍历矩阵
        // ========================================
        for (size_t i = 0; i < n; i += k) {
            
            // ========================================
            // 内层循环：Panel分解（以nb为步长）
            // ========================================
            for (size_t j = i; j < i + k && j < n; j += nb) {
                CUDA_CHECK(cudaMemsetAsync(devInfo_d, 0, sizeof(int), 0));

                // ====================================
                // 步骤1：Panel LU分解 (TSLU)
                // ====================================
                // 对当前nb列进行LU分解，找出主元并进行局部行交换
                startTimer();
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                hgetrf(hgetrf_handle, 
                       (int)(n - j),           // 剩余行数
                       (int)nb,                // 当前panel宽度
                       (half*)(A_d + j + j * n), // panel起始位置
                       (int)n,                 // leading dimension
                       (half*)tslu_workspace_d,
                       ipiv_d + j,             // 主元索引数组
                       devInfo_d);
                float tslu_ms = stopTimer();
                detail.tslu_time += tslu_ms;

                // ====================================
                // 步骤2：更新置换矩阵P（仅验证时）
                // ====================================
                if (verifyResult) {
                    ipiv_h_vector = ipiv_d_vector;
                    for (size_t idx = 0; idx < nb && (j + idx) < n; idx++) {
                        size_t global_idx = j + idx;
                        size_t local_pivot = (size_t)ipiv_h_vector[global_idx] - 1;
                        size_t global_pivot = j + local_pivot;
                        if (global_idx != global_pivot) {
                            // 交换P矩阵的相应行
                            CUBLAS_CHECK(cublasSwapEx(
                                cublas_handle, (int)n,
                                (half*)(P_d + global_idx), CUDA_R_16F, (int)n,
                                (half*)(P_d + global_pivot), CUDA_R_16F, (int)n));
                        }
                    }
                }

                // ====================================
                // 步骤3：Panel行交换
                // ====================================
                // 将panel内的行交换应用到左侧已完成部分和右侧待处理部分
                int restnum_panel = (int)(i + k - j - nb);
                thrust::host_vector<int> source;
                thrust::host_vector<int> target;
                thrust::device_vector<int> source_d;
                thrust::device_vector<int> target_d;

                startTimer();

                ipiv_h_vector = ipiv_d_vector;
                
                // 构建行交换映射
                std::vector<int> ipiv_h_idx((int)n);
                for (int idx = 0; idx < (int)n; idx++) ipiv_h_idx[idx] = idx;

                for (int idx = 0; idx < (int)nb && (int)(idx + j) < (int)n; idx++) {
                    std::swap(ipiv_h_idx[ipiv_h_vector[idx + (int)j] - 1],
                              ipiv_h_idx[idx]);
                }

                for (int idx = 0; idx < (int)n; idx++) {
                    if (ipiv_h_idx[idx] != idx) {
                        source.push_back(ipiv_h_idx[idx]);
                        target.push_back(idx);
                    }
                }

                source_d = source;
                target_d = target;

                if (!source_d.empty()) {
                    // 交换左侧已完成部分（i到j之间）
                    if (restnum_panel < (int)k - (int)nb && (int)(j - i) > 0) {
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(source_d.data()),
                            thrust::raw_pointer_cast(target_d.data()),
                            (int)source_d.size(),
                            (int)(j - i),
                            (int)n,
                            (half*)(A_d + j + i * n),
                            temp_d,
                            blocksPerGrid,
                            threadsPerBlock);
                    }
                    // 交换右侧待处理部分
                    if (restnum_panel > 0) {
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(source_d.data()),
                            thrust::raw_pointer_cast(target_d.data()),
                            (int)source_d.size(),
                            restnum_panel,
                            (int)n,
                            (half*)(A_d + j + (nb + j) * n),
                            temp_d,
                            blocksPerGrid,
                            threadsPerBlock);
                    }
                }
                float swap_panel_ms = stopTimer();
                detail.swap_panel_time += swap_panel_ms;

                // 如果是最后一个panel，跳过TRSM和GEMM
                if (nb + j - i >= k) continue;

                // ====================================
                // 步骤4：Panel TRSM（三角求解）
                // ====================================
                // 求解 L * X = B，其中L是当前panel的下三角部分
                // X是右侧待处理的nb列
                startTimer();
                trsm_float_recursive(
                    cublas_handle,
                    (int)(nb + j - i),              // L的维度
                    (int)nb,                        // X的列数
                    (half*)(A_d + i + i * n),      // L矩阵
                    (int)n,
                    (half*)(A_d + i + (nb + j) * n), // B/X矩阵
                    (int)n,
                    512,                            // 递归base case
                    trsm_ws);
                float trsm_panel_ms = stopTimer();
                detail.trsm_panel_time += trsm_panel_ms;

                // ====================================
                // 步骤5：Panel GEMM更新
                // ====================================
                // 更新右下角矩阵：C = C - A * B
                // A: 下方剩余行
                // B: 当前panel求解结果
                // C: 右下角待更新块
                startTimer();
                float alpha_f = -1.0f;
                float beta_f  = 1.0f;
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)(n - nb - j),              // C的行数
                    (int)nb,                        // C的列数
                    (int)(nb + j - i),              // 收缩维度
                    &alpha_f,
                    (half*)(A_d + (nb + j) + i * n), CUDA_R_16F, (int)n,
                    (half*)(A_d + i + (nb + j) * n), CUDA_R_16F, (int)n,
                    &beta_f,
                    (half*)(A_d + (nb + j) + (nb + j) * n), CUDA_R_16F, (int)n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                float gemm_panel_ms = stopTimer();
                detail.gemm_panel_time += gemm_panel_ms;
                detail.gemm_panel_ops += 2.0f * (float)(n - nb - j) * (float)nb * (float)(nb + j - i);
            }

            // ========================================
            // Kpanel更新（外层块级别的更新）
            // ========================================
            
            int restnum_kpanel = (int)(n - i - k);

            thrust::host_vector<int> kpanel_source_i;
            thrust::host_vector<int> kpanel_target_i;
            thrust::device_vector<int> kpanel_source_i_d;
            thrust::device_vector<int> kpanel_target_i_d;

            thrust::host_vector<int> kpanel_source_r;
            thrust::host_vector<int> kpanel_target_r;
            thrust::device_vector<int> kpanel_source_r_d;
            thrust::device_vector<int> kpanel_target_r_d;

            // ====================================
            // 步骤6：Kpanel行交换
            // ====================================
            // 将当前k块内所有panel的行交换应用到左侧和右侧
            startTimer();
            ipiv_h_vector = ipiv_d_vector;

            // 交换左侧已完成部分（0到i）
            if (restnum_kpanel < (int)n - (int)k) {
                std::vector<int> ipiv_h_idx((int)n);
                for (int idx = 0; idx < (int)n; idx++) ipiv_h_idx[idx] = idx;

                // 应用所有panel的行交换
                for (int idx = 0; idx < (int)k && (int)(i + idx) < (int)n; idx++) {
                    int target_row = ipiv_h_vector[(int)i + idx] - 1;
                    if (idx >= (int)nb) {
                        target_row += (int)nb * (idx / (int)nb);
                    }
                    std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                }

                for (int idx = 0; idx < (int)n; idx++) {
                    if (idx != ipiv_h_idx[idx]) {
                        kpanel_source_i.push_back(ipiv_h_idx[idx]);
                        kpanel_target_i.push_back(idx);
                    }
                }

                kpanel_source_i_d = kpanel_source_i;
                kpanel_target_i_d = kpanel_target_i;

                if (!kpanel_source_i_d.empty() && (int)i > 0) {
                    half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                    launchSwapByPivotingKernel(
                        thrust::raw_pointer_cast(kpanel_source_i_d.data()),
                        thrust::raw_pointer_cast(kpanel_target_i_d.data()),
                        (int)kpanel_source_i_d.size(),
                        (int)i,
                        (int)n,
                        (half*)(A_d + i),
                        temp_d,
                        blocksPerGrid,
                        threadsPerBlock);
                }
            }

            // 交换右侧待处理部分（i+k到n）
            if (restnum_kpanel > 0) {
                std::vector<int> ipiv_h_idx((int)n);
                for (int idx = 0; idx < (int)n; idx++) ipiv_h_idx[idx] = idx;

                for (int idx = 0; idx < (int)(i + k) && idx < (int)n; idx++) {
                    int target_row = ipiv_h_vector[idx] - 1;
                    if (idx >= (int)nb) {
                        target_row += (int)nb * (idx / (int)nb);
                    }
                    std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                }

                for (int idx = 0; idx < (int)n; idx++) {
                    if (idx != ipiv_h_idx[idx]) {
                        kpanel_source_r.push_back(ipiv_h_idx[idx]);
                        kpanel_target_r.push_back(idx);
                    }
                }

                kpanel_source_r_d = kpanel_source_r;
                kpanel_target_r_d = kpanel_target_r;

                if (!kpanel_source_r_d.empty()) {
                    half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                    launchSwapByPivotingKernel(
                        thrust::raw_pointer_cast(kpanel_source_r_d.data()),
                        thrust::raw_pointer_cast(kpanel_target_r_d.data()),
                        (int)kpanel_source_r_d.size(),
                        (int)k,
                        (int)n,
                        (half*)(A_d + (k + i) * n),
                        temp_d,
                        blocksPerGrid,
                        threadsPerBlock);
                }
            }

            float swap_kpanel_ms = stopTimer();
            detail.swap_kpanel_time += swap_kpanel_ms;

            // 如果已经处理完所有行，跳过kpanel TRSM和GEMM
            if (k + i >= n) break;

            // ====================================
            // 步骤7：Kpanel TRSM
            // ====================================
            // 求解整个k列块：L * X = B
            // L: 左上角(k+i)×(k+i)块
            // X: 右侧k列
            startTimer();
            {
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                trsm_float_recursive(
                    cublas_handle,
                    (int)(k + i),                   // L的维度
                    (int)k,                         // X的列数
                    (half*)A_d,                     // L矩阵
                    (int)n,
                    (half*)(A_d + (k + i) * n),    // B/X矩阵
                    (int)n,
                    512,
                    trsm_ws);
            }
            float trsm_kpanel_ms = stopTimer();
            detail.trsm_kpanel_time += trsm_kpanel_ms;

            // ====================================
            // 步骤8：Kpanel GEMM更新
            // ====================================
            // 更新右下角大矩阵块：C = C - A * B
            startTimer();
            {
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                float alpha_f = -1.0f;
                float beta_f  = 1.0f;
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)(n - k - i),               // C的行数
                    (int)k,                         // C的列数
                    (int)(k + i),                   // 收缩维度
                    &alpha_f,
                    (half*)(A_d + (k + i)), CUDA_R_16F, (int)n,
                    (half*)(A_d + (k + i) * n), CUDA_R_16F, (int)n,
                    &beta_f,
                    (half*)(A_d + (k + i) + (k + i) * n), CUDA_R_16F, (int)n,
                    CUBLAS_COMPUTE_32F,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            float gemm_kpanel_ms = stopTimer();
            detail.gemm_kpanel_time += gemm_kpanel_ms;
            detail.gemm_kpanel_ops += 2.0f * (float)(n - k - i) * (float)k * (float)(k + i);
        }
    }

    CUDA_CHECK(cudaEventRecord(total_lu_end));
    CUDA_CHECK(cudaEventSynchronize(total_lu_end));

    float entire_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&entire_time_ms, total_lu_begin, total_lu_end));

    // ========================================
    // 第七阶段：性能报告
    // ========================================
    
    printf("\n==========================================\n");
    printf("性能统计 (平均 %d 次运行)\n", roll_num);
    printf("==========================================\n");
    printf("测量总时间:  %.2f ms\n", entire_time_ms / roll_num);
    printf("\n各部分耗时:\n");
    printf("  TSLU:          %.2f ms\n", detail.tslu_time / roll_num);
    printf("  Swap panel:    %.2f ms\n", detail.swap_panel_time / roll_num);
    printf("  TRSM panel:    %.2f ms\n", detail.trsm_panel_time / roll_num);
    printf("  GEMM panel:    %.2f ms (%.1f TFLOPS)\n",
           detail.gemm_panel_time / roll_num,
           detail.gemm_panel_ops / (detail.gemm_panel_time * 1e9f));
    printf("  Swap kpanel:   %.2f ms\n", detail.swap_kpanel_time / roll_num);
    printf("  TRSM kpanel:   %.2f ms\n", detail.trsm_kpanel_time / roll_num);
    printf("  GEMM kpanel:   %.2f ms (%.1f TFLOPS)\n",
           detail.gemm_kpanel_time / roll_num,
           detail.gemm_kpanel_ops / (detail.gemm_kpanel_time * 1e9f));

    float total_lu_time = detail.tslu_time + detail.swap_panel_time +
                          detail.trsm_panel_time + detail.gemm_panel_time +
                          detail.swap_kpanel_time + detail.trsm_kpanel_time +
                          detail.gemm_kpanel_time;

    printf("\n汇总:\n");
    printf("  各部分求和:    %.2f ms\n", total_lu_time / roll_num);
    printf("  整体性能:      %.2f TFLOPS\n",
           getrf_flops(n) * roll_num / (total_lu_time * 1e9f));
    printf("  同步开销:      %.2f ms (%.1f%%)\n",
           (entire_time_ms - total_lu_time) / roll_num,
           100.0f * (entire_time_ms - total_lu_time) / entire_time_ms);
    printf("==========================================\n");

    // ========================================
    // 第八阶段：验证和对比
    // ========================================
    
    if (verifyResult) {
        computeMinusOfPAandLU_half(A_device_vector, oriA_device_vector, P_device_vector, (int)n);
    }
    if (compare_with_cusolver) {
        compareWithCusolver_half(A_seed_device_vector, n, roll_num, pivoting);
    }

    // ========================================
    // 第九阶段：清理资源
    // ========================================
    
    trsm_ws.free();
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    hgetrfDestroy(hgetrf_handle);
    CUDA_CHECK(cudaFree(devInfo_d));
    CUDA_CHECK(cudaFree(tslu_workspace_d));
    CUDA_CHECK(cudaEventDestroy(total_lu_begin));
    CUDA_CHECK(cudaEventDestroy(total_lu_end));

    return 0;
}
