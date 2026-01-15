#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <cmath>
#include <iostream>
#include <vector>

// ============== 核心参数配置 ==============
constexpr int col_process_size = 8;        // 每个warp处理的B矩阵列数
constexpr int warp_size = 32;              // CUDA warp大小
constexpr int rows_per_lane = 8;           // 每个线程处理的行数
constexpr int row_process_size = warp_size * rows_per_lane;  // 256行
constexpr int b_stage_count = 2;           // B矩阵双缓冲stage数
constexpr int b_stage_cols = col_process_size;
constexpr int warp_num = b_stage_cols;     // 8个warp
constexpr int launch_bound = warp_size * warp_num;  // 256个线程
constexpr int a_block_rows = 4;            // A矩阵每次预取的行数

// A矩阵双缓冲开关
#ifndef TRSM_A_DOUBLE_BUFFER
#define TRSM_A_DOUBLE_BUFFER 0
#endif
#if TRSM_A_DOUBLE_BUFFER
constexpr int a_stage_count = a_block_rows * 2;
#else
constexpr int a_stage_count = a_block_rows;
#endif
constexpr int a_block_count = row_process_size / a_block_rows;

// 从全局内存读取A矩阵元素（half版本）
#define A_ori(i, j) __ldg(A + (i) + (j) * M)

// ============== 异步拷贝函数 ==============
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr,
                                               const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr)
                 : "memory");
#else
    *reinterpret_cast<int4*>(smem_ptr) =
        *reinterpret_cast<const int4*>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

// ============== Half精度TRSM内核 ==============
// 向量化策略：使用 int4 (16字节) 一次加载 8个half
__global__ __launch_bounds__(launch_bound) void trsm_half_kernel(
    const half* __restrict__ A,    // M×M下三角矩阵
    half* __restrict__ B,           // M×Nrhs矩阵
    int M,                          // 固定为256
    int Nrhs) {
    
    // ========== 线程和block索引 ==========
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    int tx = lane_id + warp_id * warp_size;
    int bx = blockIdx.x;
    
    int base_col = bx * b_stage_cols * b_stage_count;
    int b_warp_offset = warp_id * row_process_size;
    int b_stage_stride = b_stage_cols * row_process_size;
    int b_tile_base = (base_col + warp_id) * row_process_size;

    // ========== B矩阵共享内存（half） ==========
    __shared__ __align__(16) half B_sm[b_stage_count][row_process_size * warp_num];
    half* B_sm0 = B_sm[0] + b_warp_offset;
    half* B_sm1 = B_sm[1] + b_warp_offset;
    int row_base = lane_id;

    // ========== 异步加载B矩阵 ==========
    // 使用 int4 加载：16字节 = 8个half，每个线程加载两次覆盖256行
#pragma unroll
    for (int stage = 0; stage < b_stage_count; ++stage) {
        half* B_sm_stage = B_sm[stage] + b_warp_offset;
        const int4* g_ptr = reinterpret_cast<const int4*>(
            B + b_tile_base + stage * b_stage_stride);
        int4* s_ptr = reinterpret_cast<int4*>(B_sm_stage);
        cp_async_cg_16(&s_ptr[lane_id], &g_ptr[lane_id]);
        cp_async_cg_16(&s_ptr[lane_id + warp_size],
                       &g_ptr[lane_id + warp_size]);
        cp_async_commit_group();
    }

    // ========== 计算A对角线倒数（转float计算） ==========
    __shared__ float A_diag[row_process_size];
#pragma unroll
    for (int idx = tx; idx < row_process_size; idx += warp_size * warp_num) {
        float diag_val = __half2float(A_ori(idx, idx));
        A_diag[idx] = 1.0f / diag_val;  // float精度倒数
    }
    __syncthreads();

    cp_async_wait_group_0();
    __syncthreads();

    // ========== 寄存器缓存（存储为float以提高精度） ==========
    const int b_cache_i0 = rows_per_lane - 2;
    const int b_cache_i1 = rows_per_lane - 1;
    const int r_cache0 = row_base + b_cache_i0 * warp_size;
    const int r_cache1 = row_base + b_cache_i1 * warp_size;
    
    const int a_cache_i0 = rows_per_lane - 4;
    const int a_cache_i1 = rows_per_lane - 3;
    const int a_cache_i2 = b_cache_i0;
    const int a_cache_i3 = b_cache_i1;
    const int r_a0 = row_base + a_cache_i0 * warp_size;
    const int r_a1 = row_base + a_cache_i1 * warp_size;
    const int r_a2 = r_cache0;
    const int r_a3 = r_cache1;
    
    // B值缓存为float（提高累加精度）
    float b0_cache0 = __half2float(B_sm0[r_cache0]);
    float b1_cache0 = __half2float(B_sm1[r_cache0]);
    float b0_cache1 = __half2float(B_sm0[r_cache1]);
    float b1_cache1 = __half2float(B_sm1[r_cache1]);

    // ========== A矩阵列的共享内存 ==========
    __shared__ __align__(16) half A_col[a_stage_count][row_process_size];
    int warp_row = warp_id & (a_block_rows - 1);

    // ========== 首次预取A ==========
    int warp_half = warp_id / a_block_rows;
    int a_vec0 = lane_id + warp_half * warp_size;
    int prefetch_row0 = warp_row;
    const int4* a_g_ptr0 =
        reinterpret_cast<const int4*>(A + prefetch_row0 * M);
    int4* a_s_ptr0 = reinterpret_cast<int4*>(A_col[warp_row]);
    cp_async_cg_16(&a_s_ptr0[a_vec0], &a_g_ptr0[a_vec0]);
    cp_async_commit_group();

#if TRSM_A_DOUBLE_BUFFER
    int prefetch_row1 = a_block_rows + warp_row;
    const int4* a_g_ptr1 =
        reinterpret_cast<const int4*>(A + prefetch_row1 * M);
    int4* a_s_ptr1 =
        reinterpret_cast<int4*>(A_col[a_block_rows + warp_row]);
    cp_async_cg_16(&a_s_ptr1[a_vec0], &a_g_ptr1[a_vec0]);
    cp_async_commit_group();
#endif

    // ========== 主循环：前向替换 ==========
    for (int block = 0; block < row_process_size; block += a_block_rows) {
        int block_idx = block / a_block_rows;
        
#if TRSM_A_DOUBLE_BUFFER
        int set = block_idx & 1;
        int buf_base = set * a_block_rows;
#else
        int buf_base = 0;
#endif

        cp_async_wait_group_0();
        __syncthreads();

#if TRSM_A_DOUBLE_BUFFER
        if (block_idx >= 1) {
            int next_block = block_idx + 1;
            if (next_block < a_block_count) {
                int other = 1 - set;
                int prefetch_row = next_block * a_block_rows + warp_row;
                const int4* g_ptr =
                    reinterpret_cast<const int4*>(A + prefetch_row * M);
                int4* s_ptr = reinterpret_cast<int4*>(
                    A_col[other * a_block_rows + warp_row]);
                cp_async_cg_16(&s_ptr[a_vec0], &g_ptr[a_vec0]);
                cp_async_commit_group();
            }
        }
#endif

        for (int row_offset = 0; row_offset < a_block_rows; ++row_offset) {
            int row = block + row_offset;
            int row_lane = row & (warp_size - 1);
            const half* A_row = A_col[buf_base + row_offset];
            
            // A值转float提高精度
            float a_cache0 = __half2float(A_row[r_a0]);
            float a_cache1 = __half2float(A_row[r_a1]);
            float a_cache2 = __half2float(A_row[r_a2]);
            float a_cache3 = __half2float(A_row[r_a3]);
            
            // ========== 计算解（float精度） ==========
            float x0 = 0.0f;
            float x1 = 0.0f;
            if (lane_id == row_lane) {
                int base = row;
                float inv = A_diag[row];
                if (row == r_cache0) {
                    x0 = b0_cache0 * inv;
                    x1 = b1_cache0 * inv;
                } else if (row == r_cache1) {
                    x0 = b0_cache1 * inv;
                    x1 = b1_cache1 * inv;
                } else {
                    x0 = __half2float(B_sm0[base]) * inv;
                    x1 = __half2float(B_sm1[base]) * inv;
                }
            }
            
            // 广播
            x0 = __shfl_sync(0xffffffff, x0, row_lane);
            x1 = __shfl_sync(0xffffffff, x1, row_lane);
            
            int first = 0;
            if (row >= row_base) {
                first = (row - row_base) / warp_size + 1;
            }
            
            // ========== 更新（float累加再转half） ==========
#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                int base = r;
                float a = __half2float(A_row[r]);
                float b0_tmp = __half2float(B_sm0[base]);
                float b1_tmp = __half2float(B_sm1[base]);
                b0_tmp -= x0 * a;
                b1_tmp -= x1 * a;
                B_sm0[base] = __float2half(b0_tmp);
                B_sm1[base] = __float2half(b1_tmp);
            }
            
            // 更新缓存值（float运算）
            if (first <= a_cache_i0) {
                int base = r_a0;
                float b0_tmp = __half2float(B_sm0[base]);
                float b1_tmp = __half2float(B_sm1[base]);
                b0_tmp -= x0 * a_cache0;
                b1_tmp -= x1 * a_cache0;
                B_sm0[base] = __float2half(b0_tmp);
                B_sm1[base] = __float2half(b1_tmp);
            }
            if (first <= a_cache_i1) {
                int base = r_a1;
                float b0_tmp = __half2float(B_sm0[base]);
                float b1_tmp = __half2float(B_sm1[base]);
                b0_tmp -= x0 * a_cache1;
                b1_tmp -= x1 * a_cache1;
                B_sm0[base] = __float2half(b0_tmp);
                B_sm1[base] = __float2half(b1_tmp);
            }
            if (first <= a_cache_i2) {
                b0_cache0 -= x0 * a_cache2;
                b1_cache0 -= x1 * a_cache2;
            }
            if (first <= a_cache_i3) {
                b0_cache1 -= x0 * a_cache3;
                b1_cache1 -= x1 * a_cache3;
            }
            
            // 写回解
            if (lane_id == row_lane) {
                int base = row;
                if (row == r_cache0) {
                    b0_cache0 = x0;
                    b1_cache0 = x1;
                } else if (row == r_cache1) {
                    b0_cache1 = x0;
                    b1_cache1 = x1;
                } else {
                    B_sm0[base] = __float2half(x0);
                    B_sm1[base] = __float2half(x1);
                }
            }
            __syncthreads();
        }

#if !TRSM_A_DOUBLE_BUFFER
        int next_block = block_idx + 1;
        if (next_block < a_block_count) {
            int prefetch_row = next_block * a_block_rows + warp_row;
            const int4* g_ptr =
                reinterpret_cast<const int4*>(A + prefetch_row * M);
            int4* s_ptr = reinterpret_cast<int4*>(A_col[warp_row]);
            cp_async_cg_16(&s_ptr[a_vec0], &g_ptr[a_vec0]);
            cp_async_commit_group();
        }
#endif
    }

    // ========== 写回寄存器缓存 ==========
    B_sm0[r_cache0] = __float2half(b0_cache0);
    B_sm1[r_cache0] = __float2half(b1_cache0);
    B_sm0[r_cache1] = __float2half(b0_cache1);
    B_sm1[r_cache1] = __float2half(b1_cache1);
    __syncthreads();

    // ========== 写回全局内存 ==========
    for (int stage = 0; stage < b_stage_count; ++stage) {
        half* B_sm_stage = B_sm[stage] + b_warp_offset;
        half* B_tile_stage = B + b_tile_base + stage * b_stage_stride;
        const int4* s_ptr_out = reinterpret_cast<const int4*>(B_sm_stage);
        int4* g_ptr_out = reinterpret_cast<int4*>(B_tile_stage);
        g_ptr_out[lane_id] = s_ptr_out[lane_id];
        g_ptr_out[lane_id + warp_size] = s_ptr_out[lane_id + warp_size];
    }
}

// ========== 辅助kernel ==========
__global__ __launch_bounds__(256) void clear_tril_upper_half(half* A, int lda) {
    auto row = threadIdx.x;
    auto col = blockIdx.x;
    if (row < col) {
        A[row + col * lda] = __float2half(0.0f);
    } else {
        float val = __half2float(A[row + col * lda]);
        A[row + col * lda] = __float2half(val + 2.0f);
    }
}

__global__ __launch_bounds__(256) void clear_tril_upper_float(float* A, int lda) {
    auto row = threadIdx.x;
    auto col = blockIdx.x;
    if (row < col) {
        A[row + col * lda] = 0.0f;
    } else {
        A[row + col * lda] += 2.0f;
    }
}

// ========== 主函数 ==========
int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <Nrhs>" << std::endl;
        return 1;
    }

    int M = row_process_size;  // 固定256
    int Nrhs = atoi(argv[2]);
    
    std::cout << "=== Half Precision TRSM Test ===" << std::endl;
    std::cout << "M = " << M << std::endl;
    std::cout << "Nrhs = " << Nrhs << std::endl;
    
    int cols_per_block = b_stage_cols * b_stage_count;
    if (Nrhs % cols_per_block != 0) {
        std::cerr << "Nrhs must be divisible by " << cols_per_block << std::endl;
        return 1;
    }

    // ========== GPU内存分配 ==========
    half *A_half = nullptr, *B_half = nullptr, *B_orig_half = nullptr;
    float *A_float = nullptr, *B_float = nullptr, *B_cublas = nullptr;
    
    cudaMalloc(&A_half, M * M * sizeof(half));
    cudaMalloc(&B_half, M * Nrhs * sizeof(half));
    cudaMalloc(&B_orig_half, M * Nrhs * sizeof(half));
    cudaMalloc(&A_float, M * M * sizeof(float));
    cudaMalloc(&B_float, M * Nrhs * sizeof(float));
    cudaMalloc(&B_cublas, M * Nrhs * sizeof(float));

    // ========== 生成随机数据 ==========
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    
    // 生成float数据然后转half
    curandGenerateUniform(gen, A_float, M * M);
    clear_tril_upper_float<<<M, M>>>(A_float, M);
    curandGenerateUniform(gen, B_float, M * Nrhs);
    cudaDeviceSynchronize();
    
    // Float -> Half转换
    std::vector<float> A_h(M * M), B_h(M * Nrhs);
    cudaMemcpy(A_h.data(), A_float, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h.data(), B_float, M * Nrhs * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<half> A_half_h(M * M), B_half_h(M * Nrhs);
    for (int i = 0; i < M * M; ++i) A_half_h[i] = __float2half(A_h[i]);
    for (int i = 0; i < M * Nrhs; ++i) B_half_h[i] = __float2half(B_h[i]);
    
    cudaMemcpy(A_half, A_half_h.data(), M * M * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_half, B_half_h.data(), M * Nrhs * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(B_orig_half, B_half, M * Nrhs * sizeof(half), cudaMemcpyDeviceToDevice);
    cudaMemcpy(B_cublas, B_float, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);

    // ========== cuBLAS测试 (float) ==========
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A_float, M, B_cublas, M);
    cudaDeviceSynchronize();
    cudaMemcpy(B_cublas, B_float, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Timing
    cudaEventRecord(start);
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A_float, M, B_cublas, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);

    // ========== Half精度kernel测试 ==========
    int grid = (Nrhs + cols_per_block - 1) / cols_per_block;
    
    // Warmup
    trsm_half_kernel<<<grid, dim3(warp_size, warp_num)>>>(A_half, B_half, M, Nrhs);
    cudaDeviceSynchronize();
    cudaMemcpy(B_half, B_orig_half, M * Nrhs * sizeof(half), cudaMemcpyDeviceToDevice);
    
    // Timing
    cudaEventRecord(start);
    trsm_half_kernel<<<grid, dim3(warp_size, warp_num)>>>(A_half, B_half, M, Nrhs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    // ========== 验证正确性 ==========
    std::vector<half> B_result_h(M * Nrhs), B_orig_half_h(M * Nrhs);
    cudaMemcpy(B_result_h.data(), B_half, M * Nrhs * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_orig_half_h.data(), B_orig_half, M * Nrhs * sizeof(half), cudaMemcpyDeviceToHost);

    // 计算残差
    double r_norm_sq = 0.0, b_norm_sq = 0.0, max_abs = 0.0;
    for (int col = 0; col < Nrhs; ++col) {
        int col_offset = col * M;
        for (int row = 0; row < M; ++row) {
            double sum = 0.0;
            for (int k = 0; k <= row; ++k) {
                sum += static_cast<double>(__half2float(A_half_h[row + k * M])) *
                       static_cast<double>(__half2float(B_result_h[k + col_offset]));
            }
            double r = static_cast<double>(__half2float(B_orig_half_h[row + col_offset])) - sum;
            r_norm_sq += r * r;
            b_norm_sq += static_cast<double>(__half2float(B_orig_half_h[row + col_offset])) *
                         static_cast<double>(__half2float(B_orig_half_h[row + col_offset]));
            double abs_r = std::abs(r);
            if (abs_r > max_abs) max_abs = abs_r;
        }
    }

    // ========== 输出结果 ==========
    double r_norm = std::sqrt(r_norm_sq);
    double b_norm = std::sqrt(b_norm_sq);
    double rel = r_norm / (b_norm + 1e-12);
    double flops = static_cast<double>(M) * M * Nrhs;
    
    std::cout << "\n=== Correctness ===" << std::endl;
    std::cout << "Residual ||B_orig - L*X||_F = " << r_norm << std::endl;
    std::cout << "Relative residual = " << rel << std::endl;
    std::cout << "Max abs residual = " << max_abs << std::endl;
    
    std::cout << "\n=== Performance ===" << std::endl;
    std::cout << "Half kernel time (ms) = " << kernel_ms << std::endl;
    std::cout << "Float cuBLAS time (ms) = " << cublas_ms << std::endl;
    std::cout << "Half kernel TFLOPS = " << flops / (kernel_ms * 1e-3) * 1e-12 << std::endl;
    std::cout << "Float cuBLAS TFLOPS = " << flops / (cublas_ms * 1e-3) * 1e-12 << std::endl;
    std::cout << "Speedup vs cuBLAS = " << cublas_ms / kernel_ms << "x" << std::endl;

    // ========== 清理 ==========
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(A_half);
    cudaFree(B_half);
    cudaFree(B_orig_half);
    cudaFree(A_float);
    cudaFree(B_float);
    cudaFree(B_cublas);

    return 0;
}