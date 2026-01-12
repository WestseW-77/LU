#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

#include <cmath>
#include <iostream>
#include <vector>

constexpr int col_process_size = 8;
constexpr int warp_size = 32;
constexpr int rows_per_lane = 8;
constexpr int row_process_size = warp_size * rows_per_lane;
constexpr int b_stage_count = 2;
constexpr int b_stage_cols = col_process_size;
constexpr int warp_num = b_stage_cols;
constexpr int launch_bound = warp_size * warp_num;
constexpr int a_block_rows = 4;
#ifndef TRSM_A_DOUBLE_BUFFER
#define TRSM_A_DOUBLE_BUFFER 0
#endif
#if TRSM_A_DOUBLE_BUFFER
constexpr int a_stage_count = a_block_rows * 2;
#else
constexpr int a_stage_count = a_block_rows;
#endif
constexpr int a_block_count = row_process_size / a_block_rows;
#define A_ori(i, j) __ldg(A + (i) + (j) * M)

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

__global__ __launch_bounds__(launch_bound) void trsm_float_kernel(
    const float* __restrict__ A, float* __restrict__ B, int M, int Nrhs) {
    int lane_id = threadIdx.x;
    int warp_id = threadIdx.y;
    int tx = lane_id + warp_id * warp_size;
    int bx = blockIdx.x;
    int base_col = bx * b_stage_cols * b_stage_count;
    int b_warp_offset = warp_id * row_process_size;
    int b_stage_stride = b_stage_cols * row_process_size;
    int b_tile_base = (base_col + warp_id) * row_process_size;

    // copy B col to shared memory (double buffer, cp_async)
    __shared__ __align__(
        16) float B_sm[b_stage_count][row_process_size * warp_num];
    float* B_sm0 = B_sm[0] + b_warp_offset;
    float* B_sm1 = B_sm[1] + b_warp_offset;
    int row_base = lane_id;
#pragma unroll
    for (int stage = 0; stage < b_stage_count; ++stage) {
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        const float4* g_ptr = reinterpret_cast<const float4*>(
            B + b_tile_base + stage * b_stage_stride);
        float4* s_ptr = reinterpret_cast<float4*>(B_sm_stage);
        cp_async_cg_16(&s_ptr[lane_id], &g_ptr[lane_id]);
        cp_async_cg_16(&s_ptr[lane_id + warp_size],
                       &g_ptr[lane_id + warp_size]);
        cp_async_commit_group();
    }

    // compute A diag
    __shared__ float A_diag[row_process_size];
    #pragma unroll
    for (int idx = tx; idx < row_process_size; idx += warp_size * warp_num) {
        A_diag[idx] = __frcp_rn(A_ori(idx, idx));
    }
    __syncthreads();

    // ensure both B stages are resident since we use them together
    cp_async_wait_group_0();
    __syncthreads();

    // cache top rows per lane in registers to reduce shared traffic
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
    float b0_cache0 = B_sm0[r_cache0];
    float b1_cache0 = B_sm1[r_cache0];
    float b0_cache1 = B_sm0[r_cache1];
    float b1_cache1 = B_sm1[r_cache1];

    __shared__ __align__(16) float A_col[a_stage_count][row_process_size];
    int warp_row = warp_id & (a_block_rows - 1);

    int warp_half = warp_id / a_block_rows;
    int a_vec0 = lane_id + warp_half * warp_size;
    int prefetch_row0 = warp_row;
    const float4* a_g_ptr0 =
        reinterpret_cast<const float4*>(A + prefetch_row0 * M);
    float4* a_s_ptr0 = reinterpret_cast<float4*>(A_col[warp_row]);

    cp_async_cg_16(&a_s_ptr0[a_vec0], &a_g_ptr0[a_vec0]);

    cp_async_commit_group();

#if TRSM_A_DOUBLE_BUFFER
    int prefetch_row1 = a_block_rows + warp_row;
    const float4* a_g_ptr1 =
        reinterpret_cast<const float4*>(A + prefetch_row1 * M);
    float4* a_s_ptr1 =
        reinterpret_cast<float4*>(A_col[a_block_rows + warp_row]);

    cp_async_cg_16(&a_s_ptr1[a_vec0], &a_g_ptr1[a_vec0]);

    cp_async_commit_group();
#endif

    // process per row (A multi-stage prefetch in 4-row blocks)
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
                const float4* g_ptr =
                    reinterpret_cast<const float4*>(A + prefetch_row * M);
                float4* s_ptr = reinterpret_cast<float4*>(
                    A_col[other * a_block_rows + warp_row]);

                cp_async_cg_16(&s_ptr[a_vec0], &g_ptr[a_vec0]);

                cp_async_commit_group();
            }
        }
#endif

        for (int row_offset = 0; row_offset < a_block_rows; ++row_offset) {
            int row = block + row_offset;
            int row_lane = row & (warp_size - 1);
            const float* A_row = A_col[buf_base + row_offset];
            float a_cache0 = A_row[r_a0];
            float a_cache1 = A_row[r_a1];
            float a_cache2 = A_row[r_a2];
            float a_cache3 = A_row[r_a3];
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
                    x0 = B_sm0[base] * inv;
                    x1 = B_sm1[base] * inv;
                }
            }
            x0 = __shfl_sync(0xffffffff, x0, row_lane);
            x1 = __shfl_sync(0xffffffff, x1, row_lane);
            int first = 0;
            if (row >= row_base) {
                first = (row - row_base) / warp_size + 1;
            }
#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                int base = r;
                float a = A_row[r];
                B_sm0[base] -= x0 * a;
                B_sm1[base] -= x1 * a;
            }
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

    B_sm0[r_cache0] = b0_cache0;
    B_sm1[r_cache0] = b1_cache0;
    B_sm0[r_cache1] = b0_cache1;
    B_sm1[r_cache1] = b1_cache1;
    __syncthreads();

    // coalesced float4 stores across the warp
    for (int stage = 0; stage < b_stage_count; ++stage) {
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        float* B_tile_stage = B + b_tile_base + stage * b_stage_stride;
        const float4* s_ptr_out = reinterpret_cast<const float4*>(B_sm_stage);
        float4* g_ptr_out = reinterpret_cast<float4*>(B_tile_stage);
        g_ptr_out[lane_id] = s_ptr_out[lane_id];
        g_ptr_out[lane_id + warp_size] = s_ptr_out[lane_id + warp_size];
    }
}

__global__ __launch_bounds__(256) void clear_tril_upper(float* A, int lda) {
    auto row = threadIdx.x;
    auto col = blockIdx.x;
    if (row < col) {
        A[row + col * lda] = 0.0f;
    } else {
        A[row + col * lda] += 2.0f;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <Nrhs>" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int Nrhs = atoi(argv[2]);

    M = row_process_size;
    std::cout << "Now M only supports " << M << std::endl;
    std::cout << "Now Nrhs is " << Nrhs << std::endl;
    int cols_per_block = b_stage_cols * b_stage_count;
    if (Nrhs % cols_per_block != 0) {
        std::cerr << "Nrhs must be divisible by " << cols_per_block
                  << std::endl;
        return 1;
    }

    float* A = nullptr;
    float* B = nullptr;
    float* B_orig = nullptr;
    float* B_cublas = nullptr;
    float* A_warm = nullptr;
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

    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, A, M * M);
    clear_tril_upper<<<M, M>>>(A, M);
    curandGenerateUniform(gen, B, M * Nrhs);
    curandGenerateUniform(gen, A_warm, M * M);
    clear_tril_upper<<<M, M>>>(A_warm, M);
    curandGenerateUniform(gen, B_warm, M * Nrhs);
    cudaDeviceSynchronize();
    cudaMemcpy(B_orig, B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);
    float alpha = 1.0f;
    cudaMemcpy(B_warm_cublas, B_warm, M * Nrhs * sizeof(float),
               cudaMemcpyDeviceToDevice);
    int grid = (Nrhs + cols_per_block - 1) / cols_per_block;
    trsm_float_kernel<<<grid, dim3(warp_size, warp_num)>>>(A_warm, B_warm, M,
                                                           Nrhs);
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A_warm, M, B_warm_cublas,
                M);
    cudaDeviceSynchronize();
    cudaMemcpy(B_cublas, B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToDevice);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                CUBLAS_DIAG_NON_UNIT, M, Nrhs, &alpha, A, M, B_cublas, M);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float cublas_ms = 0.0f;
    cudaEventElapsedTime(&cublas_ms, start, stop);
    cudaEventRecord(start);
    grid = (Nrhs + cols_per_block - 1) / cols_per_block;
    trsm_float_kernel<<<grid, dim3(warp_size, warp_num)>>>(A, B, M, Nrhs);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, start, stop);

    std::vector<float> A_h(M * M);
    std::vector<float> B_h(M * Nrhs);
    std::vector<float> B_orig_h(M * Nrhs);
    cudaMemcpy(A_h.data(), A, M * M * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_h.data(), B, M * Nrhs * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(B_orig_h.data(), B_orig, M * Nrhs * sizeof(float),
               cudaMemcpyDeviceToHost);

    double r_norm_sq = 0.0;
    double b_norm_sq = 0.0;
    double max_abs = 0.0;
    for (int col = 0; col < Nrhs; ++col) {
        int col_offset = col * M;
        for (int row = 0; row < M; ++row) {
            double sum = 0.0;
            for (int k = 0; k <= row; ++k) {
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
    std::cout << "Kernel TFLOPS (assume M*M*Nrhs flops) = " << kernel_tflops
              << std::endl;
    std::cout << "cuBLAS TFLOPS (assume M*M*Nrhs flops) = " << cublas_tflops
              << std::endl;

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
