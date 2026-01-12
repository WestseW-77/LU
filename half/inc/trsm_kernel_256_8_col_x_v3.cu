#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cmath>
#include <iostream>
#include <vector>

// ============== 核心参数配置 ==============
constexpr int col_process_size = 8;        // 每个warp处理的B矩阵列数
constexpr int warp_size = 32;              // CUDA warp大小
constexpr int rows_per_lane = 8;           // 每个线程处理的行数
constexpr int row_process_size = warp_size * rows_per_lane;  // 256行，block处理的总行数
constexpr int b_stage_count = 2;           // B矩阵双缓冲stage数
constexpr int b_stage_cols = col_process_size;  // 每个stage的列数
constexpr int warp_num = b_stage_cols;     // 8个warp
constexpr int launch_bound = warp_size * warp_num;  // 256个线程
constexpr int a_block_rows = 4;            // A矩阵每次预取的行数

// A矩阵双缓冲开关（编译时选项）
#ifndef TRSM_A_DOUBLE_BUFFER
#define TRSM_A_DOUBLE_BUFFER 0
#endif
#if TRSM_A_DOUBLE_BUFFER
constexpr int a_stage_count = a_block_rows * 2;  // 双缓冲：8个stage
#else
constexpr int a_stage_count = a_block_rows;      // 单缓冲：4个stage
#endif
constexpr int a_block_count = row_process_size / a_block_rows;  // 64个block

// 从全局内存读取A矩阵元素的宏（使用__ldg进行只读缓存优化）
#define A_ori(i, j) __ldg(A + (i) + (j) * M)

// ============== 异步拷贝函数 ==============
// Ampere架构(SM80+)的异步拷贝：从全局内存到共享内存，16字节（float4）
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr,
                                               const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    // 使用PTX指令实现异步拷贝（不阻塞线程，由硬件在后台完成）
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr)
                 : "memory");
#else
    // 旧架构回退到同步拷贝
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

// 提交一组异步拷贝操作
__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

// 等待所有异步拷贝完成（wait_group 0表示等待所有group）
__device__ __forceinline__ void cp_async_wait_group_0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

// ============== 主TRSM内核 ==============
__global__ __launch_bounds__(launch_bound) void trsm_float_kernel(
    const float* __restrict__ A,  // 输入：M×M下三角矩阵（列主序）
    float* __restrict__ B,         // 输入/输出：M×Nrhs矩阵（列主序）
    int M,                         // 矩阵维度（固定为256）
    int Nrhs) {                    // 右端项数量
    
    // ========== 线程和block索引计算 ==========
    int lane_id = threadIdx.x;     // warp内线程ID [0, 31]
    int warp_id = threadIdx.y;     // warp ID [0, 7]
    int tx = lane_id + warp_id * warp_size;  // block内全局线程ID [0, 255]
    int bx = blockIdx.x;           // block索引
    
    // B矩阵的列索引计算
    int base_col = bx * b_stage_cols * b_stage_count;  // 当前block处理的起始列
    int b_warp_offset = warp_id * row_process_size;    // 当前warp在共享内存中的偏移
    int b_stage_stride = b_stage_cols * row_process_size;  // stage之间的跨度
    int b_tile_base = (base_col + warp_id) * row_process_size;  // 全局内存中的tile起始位置

    // ========== B矩阵双缓冲共享内存 ==========
    // 布局：[2个stage][256行 × 8列]
    __shared__ __align__(16) float B_sm[b_stage_count][row_process_size * warp_num];
    float* B_sm0 = B_sm[0] + b_warp_offset;  // stage 0的warp视图
    float* B_sm1 = B_sm[1] + b_warp_offset;  // stage 1的warp视图
    int row_base = lane_id;  // 当前线程处理的行基址（跨度为32）

    // ========== 异步加载B矩阵到共享内存（双stage） ==========
#pragma unroll
    for (int stage = 0; stage < b_stage_count; ++stage) {
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        // 每个线程加载两个float4（32字节），覆盖256行
        const float4* g_ptr = reinterpret_cast<const float4*>(
            B + b_tile_base + stage * b_stage_stride);
        float4* s_ptr = reinterpret_cast<float4*>(B_sm_stage);
        cp_async_cg_16(&s_ptr[lane_id], &g_ptr[lane_id]);
        cp_async_cg_16(&s_ptr[lane_id + warp_size],
                       &g_ptr[lane_id + warp_size]);
        cp_async_commit_group();
    }

    // ========== 计算A矩阵对角线倒数（用于除法） ==========
    __shared__ float A_diag[row_process_size];  // 256个元素
#pragma unroll
    for (int idx = tx; idx < row_process_size; idx += warp_size * warp_num) {
        A_diag[idx] = __frcp_rn(A_ori(idx, idx));  // 快速倒数指令
    }
    __syncthreads();

    // ========== 等待B矩阵加载完成 ==========
    cp_async_wait_group_0();
    __syncthreads();

    // ========== 寄存器缓存优化 ==========
    // 缓存每个线程最后两行的B值，减少共享内存访问
    const int b_cache_i0 = rows_per_lane - 2;  // 第6行（相对索引）
    const int b_cache_i1 = rows_per_lane - 1;  // 第7行（相对索引）
    const int r_cache0 = row_base + b_cache_i0 * warp_size;  // 绝对行索引
    const int r_cache1 = row_base + b_cache_i1 * warp_size;
    
    // 缓存A矩阵的4个元素位置
    const int a_cache_i0 = rows_per_lane - 4;
    const int a_cache_i1 = rows_per_lane - 3;
    const int a_cache_i2 = b_cache_i0;
    const int a_cache_i3 = b_cache_i1;
    const int r_a0 = row_base + a_cache_i0 * warp_size;
    const int r_a1 = row_base + a_cache_i1 * warp_size;
    const int r_a2 = r_cache0;
    const int r_a3 = r_cache1;
    
    // 将B的最后两行加载到寄存器（两个stage）
    float b0_cache0 = B_sm0[r_cache0];
    float b1_cache0 = B_sm1[r_cache0];
    float b0_cache1 = B_sm0[r_cache1];
    float b1_cache1 = B_sm1[r_cache1];

    // ========== A矩阵列的共享内存（多stage预取） ==========
    __shared__ __align__(16) float A_col[a_stage_count][row_process_size];
    int warp_row = warp_id & (a_block_rows - 1);  // warp在4行block内的行号

    // ========== 首次预取A矩阵 ==========
    int warp_half = warp_id / a_block_rows;  // warp分成两组（0-3, 4-7）
    int a_vec0 = lane_id + warp_half * warp_size;  // 预取向量索引
    int prefetch_row0 = warp_row;
    const float4* a_g_ptr0 =
        reinterpret_cast<const float4*>(A + prefetch_row0 * M);
    float4* a_s_ptr0 = reinterpret_cast<float4*>(A_col[warp_row]);
    cp_async_cg_16(&a_s_ptr0[a_vec0], &a_g_ptr0[a_vec0]);
    cp_async_commit_group();

#if TRSM_A_DOUBLE_BUFFER
    // 如果启用双缓冲，预取第二组4行
    int prefetch_row1 = a_block_rows + warp_row;
    const float4* a_g_ptr1 =
        reinterpret_cast<const float4*>(A + prefetch_row1 * M);
    float4* a_s_ptr1 =
        reinterpret_cast<float4*>(A_col[a_block_rows + warp_row]);
    cp_async_cg_16(&a_s_ptr1[a_vec0], &a_g_ptr1[a_vec0]);
    cp_async_commit_group();
#endif

    // ========== 主循环：按4行block处理256行 ==========
    for (int block = 0; block < row_process_size; block += a_block_rows) {
        int block_idx = block / a_block_rows;
        
#if TRSM_A_DOUBLE_BUFFER
        // 双缓冲：交替使用两个buffer
        int set = block_idx & 1;
        int buf_base = set * a_block_rows;
#else
        int buf_base = 0;
#endif

        // 等待当前block的A矩阵加载完成
        cp_async_wait_group_0();
        __syncthreads();

#if TRSM_A_DOUBLE_BUFFER
        // 预取下一个block（如果存在）
        if (block_idx >= 1) {
            int next_block = block_idx + 1;
            if (next_block < a_block_count) {
                int other = 1 - set;
                int prefetch_row = next_block * a_block_rows + warp_row;
                const float4* g_ptr =
                    reinterpret_cast<const float4*>(A + prefetch_row * M);
                float4* s_ptr = reinterpret_cast<float4*>(
                    A_col[other * a_block_rows + warp_row]);
                cp_async_cg_16(&s_ptr[a_vec0], &g_ptr[a_vec0]);
                cp_async_commit_group();
            }
        }
#endif

        // ========== 处理当前block的4行 ==========
        for (int row_offset = 0; row_offset < a_block_rows; ++row_offset) {
            int row = block + row_offset;  // 当前处理的绝对行号
            int row_lane = row & (warp_size - 1);  // 哪个lane负责这行
            const float* A_row = A_col[buf_base + row_offset];  // A的当前行
            
            // 从共享内存读取A的缓存值到寄存器
            float a_cache0 = A_row[r_a0];
            float a_cache1 = A_row[r_a1];
            float a_cache2 = A_row[r_a2];
            float a_cache3 = A_row[r_a3];
            
            // ========== 计算当前行的解 x0, x1 ==========
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (lane_id == row_lane) {
                // 负责该行的线程计算解：x = b[row] / A[row][row]
                int base = row;
                float inv = A_diag[row];
                if (row == r_cache0) {
                    x0 = b0_cache0 * inv;
                    x1 = b1_cache0 * inv;
                } else if (row == r_cache1) {
                    x0 = b0_cache1 * inv;
                    x1 = b1_cache1 * inv;
                } else {
                    x0 = B_sm0[base] * inv;
                    x1 = B_sm1[base] * inv;
                }
            }
            
            // ========== 广播解到warp内所有线程 ==========
            x0 = __shfl_sync(0xffffffff, x0, row_lane);
            x1 = __shfl_sync(0xffffffff, x1, row_lane);
            
            // ========== 计算需要更新的起始行 ==========
            int first = 0;
            if (row >= row_base) {
                // 只更新当前行之后的行（下三角性质）
                first = (row - row_base) / warp_size + 1;
            }
            
            // ========== 更新后续行：B[r] -= A[r][row] * x ==========
#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                int base = r;
                float a = A_row[r];
                B_sm0[base] -= x0 * a;  // stage 0
                B_sm1[base] -= x1 * a;  // stage 1
            }
            
            // 更新缓存的A和B值（最后4行）
            if (first <= a_cache_i0) {
                int base = r_a0;
                B_sm0[base] -= x0 * a_cache0;
                B_sm1[base] -= x1 * a_cache0;
            }
            if (first <= a_cache_i1) {
                int base = r_a1;
                B_sm0[base] -= x0 * a_cache1;
                B_sm1[base] -= x1 * a_cache1;
            }
            if (first <= a_cache_i2) {
                b0_cache0 -= x0 * a_cache2;
                b1_cache0 -= x1 * a_cache2;
            }
            if (first <= a_cache_i3) {
                b0_cache1 -= x0 * a_cache3;
                b1_cache1 -= x1 * a_cache3;
            }
            
            // ========== 写回当前行的解 ==========
            if (lane_id == row_lane) {
                int base = row;
                if (row == r_cache0) {
                    b0_cache0 = x0;
                    b1_cache0 = x1;
                } else if (row == r_cache1) {
                    b0_cache1 = x0;
                    b1_cache1 = x1;
                } else {
                    B_sm0[base] = x0;
                    B_sm1[base] = x1;
                }
            }
            __syncthreads();
        }

#if !TRSM_A_DOUBLE_BUFFER
        // 单缓冲模式：在处理完当前block后预取下一个
        int next_block = block_idx + 1;
        if (next_block < a_block_count) {
            int prefetch_row = next_block * a_block_rows + warp_row;
            const float4* g_ptr =
                reinterpret_cast<const float4*>(A + prefetch_row * M);
            float4* s_ptr = reinterpret_cast<float4*>(A_col[warp_row]);
            cp_async_cg_16(&s_ptr[a_vec0], &g_ptr[a_vec0]);
            cp_async_commit_group();
        }
#endif
    }

    // ========== 写回寄存器缓存的B值到共享内存 ==========
    B_sm0[r_cache0] = b0_cache0;
    B_sm1[r_cache0] = b1_cache0;
    B_sm0[r_cache1] = b0_cache1;
    B_sm1[r_cache1] = b1_cache1;
    __syncthreads();

    // ========== 将结果从共享内存写回全局内存 ==========
    // 使用合并访存：每个线程写两个float4
    for (int stage = 0; stage < b_stage_count; ++stage) {
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        float* B_tile_stage = B + b_tile_base + stage * b_stage_stride;
        const float4* s_ptr_out = reinterpret_cast<const float4*>(B_sm_stage);
        float4* g_ptr_out = reinterpret_cast<float4*>(B_tile_stage);
        g_ptr_out[lane_id] = s_ptr_out[lane_id];
        g_ptr_out[lane_id + warp_size] = s_ptr_out[lane_id + warp_size];
    }
}

// ========== 辅助kernel：清除A的上三角并增强对角线 ==========
__global__ __launch_bounds__(256) void clear_tril_upper(float* A, int lda) {
    auto row = threadIdx.x;
    auto col = blockIdx.x;
    if (row < col) {
        A[row + col * lda] = 0.0f;  // 上三角置零
    } else {
        A[row + col * lda] += 2.0f;  // 对角线和下三角加2（增强数值稳定性）
    }
}

// ========== 主函数：测试和性能评估 ==========
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <Nrhs>" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int Nrhs = atoi(argv[2]);

    // 强制M为256（硬编码优化）
    M = row_process_size;
    std::cout << "Now M only supports " << M << std::endl;
    std::cout << "Now Nrhs is " << Nrhs << std::endl;
    
    // 检查Nrhs是否是16的倍数（2 stages × 8 cols）
    int cols_per_block = b_stage_cols * b_stage_count;
    if (Nrhs % cols_per_block != 0) {
        std::cerr << "Nrhs must be divisible by " << cols_per_block << std::endl;
        return 1;
    }

    // ========== 分配GPU内存 ==========
    float* A = nullptr;              // 测试矩阵A
    float* B = nullptr;              // 测试矩阵B
    float* B_orig = nullptr;         // B的备份（用于验证）
    float* B_cublas = nullptr;       // cuBLAS结果
    float* A_warm = nullptr;         // 预热用矩阵
    float* B_warm = nullptr;
    float* B_warm_cublas = nullptr;
    
    cudaMalloc(&A, M * M * sizeof(float));
    cudaMalloc(&B, M * Nrhs * sizeof(float));
    cudaMalloc(&B_orig, M * Nrhs * sizeof(float));
    cudaMalloc(&B_cublas, M * Nrhs * sizeof(float));
    cudaMalloc(&A_warm, M * M * sizeof(float));
    cudaMalloc(&B_warm, M * Nrhs * sizeof(float));
    cudaMalloc(&B_warm_cublas, M * Nrhs * sizeof(float));
    cudaMemset(A, 0, M * M * sizeof(float));
    cudaMemset(B, 0, M * Nrhs * sizeof(float));

    // ========== 生成随机测试数据 ==========
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, A, M * M);
    clear_tril_upper<<<M, M>>>(A, M);  // 构造下三角矩阵
    curandGenerateUniform(gen, B, M * Nrhs);
    curandGenerateUniform(gen, A_warm, M * M);
    clear_tril_upper<<<M, M>>>(A_warm, M);
    curandGenerateUniform(gen, B_warm, M * Nrhs);
    cudaDeviceSynchronize();
    cudaMemcpy(B_orig, B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);

    // ========== cuBLAS初始化和预热 ==========
    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    float alpha = 1.0f;
    cudaMemcpy(B_warm_cublas, B_warm, M * Nrhs * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // 预热自定义kernel
    int grid = (Nrhs + cols_per_block - 1) / cols_per_block;
    trsm_float_kernel<<<grid, dim3(warp_size, warp_num)>>>(A_warm, B_warm, M, Nrhs);
    
    // 预热cuBLAS
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A_warm, M, B_warm_cublas, M);
    cudaDeviceSynchronize();
    cudaMemcpy(B_cublas, B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);

    // ========== 性能测试 ==========
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // 测试cuBLAS
    cudaEventRecord(start);
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A, M, B_cublas, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    
    // 测试自定义kernel
    cudaEventRecord(start);
    grid = (Nrhs + cols_per_block - 1) / cols_per_block;
    trsm_float_kernel<<<grid, dim3(warp_size, warp_num)>>>(A, B, M, Nrhs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // ========== 验证正确性 ==========
    std::vector<float> A_h(M * M);
    std::vector<float> B_h(M * Nrhs);
    std::vector<float> B_orig_h(M * Nrhs);
    cudaMemcpy(A_h.data(), A, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h.data(), B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_orig_h.data(), B_orig, M * Nrhs * sizeof(float),
               cudaMemcpyDeviceToHost);

    // 计算残差：||B_orig - L*X||
    double r_norm_sq = 0.0;
    double b_norm_sq = 0.0;
    double max_abs = 0.0;
    for (int col = 0; col < Nrhs; ++col) {
        int col_offset = col * M;
        for (int row = 0; row < M; ++row) {
            double sum = 0.0;
            for (int k = 0; k <= row; ++k) {  // 下三角矩阵乘法
                sum += static_cast<double>(A_h[row + k * M]) *
                       static_cast<double>(B_h[k + col_offset]);
            }
            double r = static_cast<double>(B_orig_h[row + col_offset]) - sum;
            r_norm_sq += r * r;
            b_norm_sq += static_cast<double>(B_orig_h[row + col_offset]) *
                         static_cast<double>(B_orig_h[row + col_offset]);
            double abs_r = std::abs(r);
            if (abs_r > max_abs) {
                max_abs = abs_r;
            }
        }
    }
    
    // ========== 输出结果 ==========
    double r_norm = std::sqrt(r_norm_sq);
    double b_norm = std::sqrt(b_norm_sq);
    double rel = r_norm / (b_norm + 1e-12);
    double flops = static_cast<double>(M) * static_cast<double>(M) *
                   static_cast<double>(Nrhs);
    double kernel_tflops = flops / (kernel_ms * 1e-3) * 1e-12;
    double cublas_tflops = flops / (cublas_ms * 1e-3) * 1e-12;
    
    std::cout << "Residual ||B_orig - L*X||_F = " << r_norm << std::endl;
    std::cout << "Relative residual = " << rel << std::endl;
    std::cout << "Max abs residual = " << max_abs << std::endl;
    std::cout << "Kernel time (ms) = " << kernel_ms << std::endl;
    std::cout << "cuBLAS trsm time (ms) = " << cublas_ms << std::endl;
    std::cout << "Kernel TFLOPS (assume M*M*Nrhs flops) = " << kernel_tflops << std::endl;
    std::cout << "cuBLAS TFLOPS (assume M*M*Nrhs flops) = " << cublas_tflops << std::endl;

    // ========== 清理资源 ==========
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A);
    cudaFree(B);
    cudaFree(B_orig);
    cudaFree(B_cublas);
    cudaFree(A_warm);
    cudaFree(B_warm);
    cudaFree(B_warm_cublas);

    return 0;
}