#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "getrf.cuh"
#include "inc/hgetrf.cuh"

#define SWAP_LEN_PANEL 2048

using data_type = __half;

// -----------------------------
// Helpers
// -----------------------------
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

// -----------------------------
// TRSM kernel - base case = 128 (保守但安全)
// -----------------------------
template <int K_MAX>
__global__ void trsm_unit_lower_half_kernel(
    const half* __restrict__ L, int lda,
    half* __restrict__ B, int ldb,
    int m, int ncols) {
    
    __shared__ half sL[K_MAX * K_MAX];  // 128*128*2 = 32KB (安全)
    
    // 加载L矩阵
    for (int idx = threadIdx.x; idx < m * m; idx += blockDim.x) {
        int i = idx % m;
        int j = idx / m;
        sL[i + j * K_MAX] = L[i + (size_t)j * lda];
    }
    __syncthreads();
    
    const int warp = (int)threadIdx.x / 32;
    const int lane = (int)threadIdx.x & 31;
    const int warps_per_block = (int)blockDim.x / 32;
    
    // 每个block的每个warp处理一列
    const int rhs = (int)blockIdx.x * warps_per_block + warp;
    if (rhs >= ncols) return;
    
    half* colptr = B + (size_t)rhs * ldb;
    
    // 前向替代
    for (int i = 0; i < m; ++i) {
        float bi = 0.0f;
        if (lane == 0) bi = __half2float(colptr[i]);
        
        float acc = 0.0f;
        for (int k = lane; k < i; k += 32) {
            float Lik = __half2float(sL[i + k * K_MAX]);
            float xk = __half2float(colptr[k]);
            acc += Lik * xk;
        }
        
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffff, acc, off);
        }
        
        if (lane == 0) colptr[i] = __float2half(bi - acc);
        __syncwarp();
    }
}

// -----------------------------
// 递归TRSM - base case = 128 (与原版相同，保证稳定)
// -----------------------------
static void trsm_half_recursive_optimized(
    cublasHandle_t cublasH, int m, int ncols,
    const half* A, int lda, half* B, int ldb) {
    
    if (m <= 0 || ncols <= 0) return;
    
    const int nb_base = 128;  // 保持原来的大小，避免shared memory问题
    
    if (m <= nb_base) {
        constexpr int block = 256;
        int warps_per_block = block / 32;
        int grid = (ncols + warps_per_block - 1) / warps_per_block;
        
        trsm_unit_lower_half_kernel<128><<<grid, block>>>(
            A, lda, B, ldb, m, ncols);
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    
    int m1 = m / 2;
    int m2 = m - m1;
    
    const half* A11 = A;
    const half* A21 = A + m1;
    const half* A22 = A + m1 + (size_t)m1 * lda;
    
    half* B1 = B;
    half* B2 = B + m1;
    
    trsm_half_recursive_optimized(cublasH, m1, ncols, A11, lda, B1, ldb);
    
    const float alpha = -1.0f;
    const float beta = 1.0f;
    CUBLAS_CHECK(cublasGemmEx(
        cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
        m2, ncols, m1,
        &alpha, A21, CUDA_R_16F, lda,
        B1, CUDA_R_16F, ldb,
        &beta, B2, CUDA_R_16F, ldb,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    trsm_half_recursive_optimized(cublasH, m2, ncols, A22, lda, B2, ldb);
}

// -----------------------------
// Swap kernel
// -----------------------------
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
    grid.sync();

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

// -----------------------------
// Args parsing
// -----------------------------
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

    pivoting = true;
    return 0;
}

// -----------------------------
// cuSOLVER compare
// -----------------------------
static inline double getrf_flops(size_t n) {
    return (2.0 / 3.0) * std::pow((double)n, 3);
}

static void compareWithCusolver_half(const thrust::device_vector<__half>& A_seed,
                                     size_t n, int roll_num, bool pivoting) {
    printf("\n========== cuSOLVER Baseline ==========\n");

    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    thrust::device_vector<float> A_float(n * n);
    int* dinfo = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dinfo, sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        cusolver_handle, (int)n, (int)n, 
        thrust::raw_pointer_cast(A_float.data()), (int)n, &lwork));

    float* dwork = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dwork, (size_t)lwork * sizeof(float)));

    int* dpiv = nullptr;
    if (pivoting) {
        CUDA_CHECK(cudaMalloc((void**)&dpiv, n * sizeof(int)));
    }

    int block = 256;
    int grid = (n * n + block - 1) / block;

    float cusolver_lu_time = 0.0f;
    for (int i = 0; i < roll_num; i++) {
        half_to_float_kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(A_seed.data()),
            thrust::raw_pointer_cast(A_float.data()), (int)(n * n));
        CUDA_CHECK(cudaDeviceSynchronize());

        startTimer();
        CUSOLVER_CHECK(cusolverDnSgetrf(
            cusolver_handle, (int)n, (int)n, 
            thrust::raw_pointer_cast(A_float.data()), (int)n, 
            dwork, dpiv, dinfo));
        cusolver_lu_time += stopTimer();
    }

    printf("cuSOLVER time: %.2f ms\n", cusolver_lu_time / roll_num);
    printf("cuSOLVER: %.2f TFLOPS\n",
           (float)(getrf_flops(n) / ((cusolver_lu_time / roll_num) * 1e9)));

    if (dpiv) CUDA_CHECK(cudaFree(dpiv));
    if (dwork) CUDA_CHECK(cudaFree(dwork));
    if (dinfo) CUDA_CHECK(cudaFree(dinfo));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

// -----------------------------
// Verify
// -----------------------------
static void computeMinusOfPAandLU_half(thrust::device_vector<__half>& A_device_vector,
                                       thrust::device_vector<__half>& oriA_device_vector,
                                       thrust::device_vector<__half>& P_device_vector,
                                       int n) {
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    auto A_norm = computeFrobeniusNorm(n, n, oriA_device_vector);

    auto A_d   = (half*)thrust::raw_pointer_cast(A_device_vector.data());
    auto oriA_d= (half*)thrust::raw_pointer_cast(oriA_device_vector.data());
    auto P_d   = (half*)thrust::raw_pointer_cast(P_device_vector.data());

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

    thrust::device_vector<__half> L_device_vector = A_device_vector;
    cleanMatrix<<<gridDim, blockDim>>>(
        (half*)thrust::raw_pointer_cast(L_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, half* L_d, int lda) {
            half zero = __float2half(0.0f);
            half one  = __float2half(1.0f);
            if (i < n && j < n) {
                if (i < j) L_d[i + (size_t)j * lda] = zero;
                if (i == j) L_d[i + (size_t)j * lda] = one;
            }
        });
    CUDA_CHECK(cudaGetLastError());

    thrust::device_vector<__half> U_device_vector = A_device_vector;
    cleanMatrix<<<gridDim, blockDim>>>(
        (half*)thrust::raw_pointer_cast(U_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, half* U_d, int lda) {
            half zero = __float2half(0.0f);
            if (i < n && j < n) {
                if (i > j) U_d[i + (size_t)j * lda] = zero;
            }
        });
    CUDA_CHECK(cudaGetLastError());

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

    thrust::device_vector<__half> PAminusLU_device_vector(n * n);
    auto PAminusLU_d = (half*)thrust::raw_pointer_cast(PAminusLU_device_vector.data());

    int64_t total = (int64_t)n * (int64_t)n;
    int block = 256;
    int64_t grid = (total + block - 1) / block;
    elementwise_sub_kernel<<<(unsigned int)grid, block>>>(
        (const half*)PA_d, (const half*)LU_d, (half*)PAminusLU_d, total);
    CUDA_CHECK(cudaGetLastError());

    double Minus_norm = computeFrobeniusNorm(n, n, PAminusLU_device_vector);
    printf("\n========== Verification ==========\n");
    printf("|PA - LU|: %.6e\n", Minus_norm);
    printf("|A|: %.6e\n", A_norm);
    printf("Relative error: %.6e\n", Minus_norm / A_norm);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

// -----------------------------
// Main
// -----------------------------
struct DoubleBlockingLUBreakDownDetail {
    float tslu_time = 0.0f;
    float swap_panel_time = 0.0f;
    float trsm_panel_time = 0.0f;
    float gemm_panel_time = 0.0f;
    float gemm_panel_ops = 0.0f;
    float swap_kpanel_time = 0.0f;
    float trsm_kpanel_time = 0.0f;
    float gemm_kpanel_ops = 0.0f;
    float gemm_kpanel_time = 0.0f;
};

int main(int argc, char *argv[]) {
    size_t n, k, nb;
    bool verifyResult = false;
    bool pivoting = false;
    bool debug_mode = false;
    bool compare_with_cusolver = false;

    int status = parseArgs(argc, argv, n, k, nb, verifyResult, pivoting, debug_mode,
                           compare_with_cusolver);
    if (status != 0) return 1;

    printf("Matrix: %zu×%zu, k=%zu, nb=%zu\n", n, n, k, nb);
    printf("Data type: half, TRSM base_case=128 (32KB smem)\n");

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s\n", prop.name);

    int maxBlocksPerGrid = prop.multiProcessorCount;
    int threadsPerBlock = 256;
    int blocksPerGrid = maxBlocksPerGrid;

    // ========== 准备数据（计时外）==========
    thrust::device_vector<data_type> A_device_vector(n * n);
    generateNormalMatrix(A_device_vector, n, n);

    thrust::device_vector<data_type> A_seed_device_vector;
    if (verifyResult || compare_with_cusolver) {
        A_seed_device_vector = A_device_vector;
    }

    thrust::device_vector<data_type> P_device_vector;
    thrust::device_vector<data_type> oriA_device_vector;
    half* P_d = nullptr;

    if (verifyResult) {
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

    // ========== 创建handles和workspace（计时外）==========
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));
    CUBLAS_CHECK(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));

    hgetrfHandle_t hgetrf_handle = nullptr;
    hgetrfCreate(&hgetrf_handle);
    hgetrfSetStream(hgetrf_handle, 0);
    hgetrfSetPanelWidth(hgetrf_handle, 128);
    hgetrfSetUc(hgetrf_handle, 8);

    thrust::device_vector<data_type> temp_device_vector(n * n);
    auto temp_d = (half*)thrust::raw_pointer_cast(temp_device_vector.data());

    int buffer_size = 0;
    auto A_raw_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
    hgetrf_bufferSize(hgetrf_handle, (int)n, (int)nb,
                      (const half*)A_raw_d, (int)n, &buffer_size);

    data_type* tslu_workspace_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&tslu_workspace_d, sizeof(data_type) * (size_t)buffer_size));

    int* devInfo_d = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devInfo_d, sizeof(int)));

    thrust::device_vector<int> ipiv_d_vector(n);
    auto ipiv_d = thrust::raw_pointer_cast(ipiv_d_vector.data());
    thrust::host_vector<int> ipiv_h_vector(n);

    // ========== Warmup（计时外）==========
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

    // ========== 正式测试 ==========
    int roll_num = verifyResult ? 1 : 3;
    DoubleBlockingLUBreakDownDetail detail;

    cudaEvent_t total_lu_begin, total_lu_end;
    CUDA_CHECK(cudaEventCreate(&total_lu_begin));
    CUDA_CHECK(cudaEventCreate(&total_lu_end));
    
    CUDA_CHECK(cudaEventRecord(total_lu_begin));

    for (int roll = 0; roll < roll_num; roll++) {
        // 重置（轻量级操作）
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
            CUDA_CHECK(cudaMemset(P_d, 0, sizeof(data_type) * n * n));
            thrust::for_each(thrust::counting_iterator<size_t>(0),
                             thrust::counting_iterator<size_t>(n),
                             [P_d, n] __device__(size_t i) {
                                 P_d[i * n + i] = __float2half_rn(1.0f);
                             });
            oriA_device_vector = A_device_vector;
        }

        size_t panel_calls = 0;

        // ====== LU分解核心循环 ======
        for (size_t i = 0; i < n; i += k) {
            size_t panels_in_block = 0;
            for (size_t j = i; j < i + k && j < n; j += nb) {
                CUDA_CHECK(cudaMemsetAsync(devInfo_d, 0, sizeof(int), 0));

                startTimer();
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                hgetrf(hgetrf_handle, (int)(n - j), (int)nb,
                       (half*)(A_d + j + j * n), (int)n,
                       (half*)tslu_workspace_d,
                       ipiv_d + j, devInfo_d);
                float tslu_ms = stopTimer();
                detail.tslu_time += tslu_ms;
                panel_calls++;
                panels_in_block++;

                if (verifyResult) {
                    ipiv_h_vector = ipiv_d_vector;
                    for (size_t idx = 0; idx < nb && (j + idx) < n; idx++) {
                        size_t global_idx = j + idx;
                        size_t local_pivot = (size_t)ipiv_h_vector[global_idx] - 1;
                        size_t global_pivot = j + local_pivot;
                        if (global_idx != global_pivot) {
                            CUBLAS_CHECK(cublasSwapEx(
                                cublas_handle, (int)n,
                                (half*)(P_d + global_idx), CUDA_R_16F, (int)n,
                                (half*)(P_d + global_pivot), CUDA_R_16F, (int)n));
                        }
                    }
                }

                int restnum_panel = (int)(i + k - j - nb);
                thrust::host_vector<int> source;
                thrust::host_vector<int> target;
                thrust::device_vector<int> source_d;
                thrust::device_vector<int> target_d;

                startTimer();
                ipiv_h_vector = ipiv_d_vector;
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

                if (nb + j - i >= k) {
                    if (debug_mode) {
                        printf("[%d] panel i=%zu j=%zu: tslu=%.2f swap=%.2f (skip trsm/gemm)\n",
                               roll, i, j, tslu_ms, swap_panel_ms);
                    }
                    continue;
                }

                // TRSM panel
                startTimer();
                trsm_half_recursive_optimized(
                    cublas_handle,
                    (int)(nb + j - i),
                    (int)nb,
                    (half*)(A_d + i + i * n),
                    (int)n,
                    (half*)(A_d + i + (nb + j) * n),
                    (int)n);
                float trsm_panel_ms = stopTimer();
                detail.trsm_panel_time += trsm_panel_ms;

                // GEMM panel update
                startTimer();
                float alpha_f = -1.0f;
                float beta_f  = 1.0f;
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)(n - nb - j), (int)nb, (int)(nb + j - i),
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

                if (debug_mode) {
                    printf("[%d] panel i=%zu j=%zu: tslu=%.2f swap=%.2f trsm=%.2f gemm=%.2f\n",
                           roll, i, j, tslu_ms, swap_panel_ms, trsm_panel_ms, gemm_panel_ms);
                }
            }

            // kpanel update
            int restnum_kpanel = (int)(n - i - k);

            thrust::host_vector<int> kpanel_source_i;
            thrust::host_vector<int> kpanel_target_i;
            thrust::device_vector<int> kpanel_source_i_d;
            thrust::device_vector<int> kpanel_target_i_d;

            thrust::host_vector<int> kpanel_source_r;
            thrust::host_vector<int> kpanel_target_r;
            thrust::device_vector<int> kpanel_source_r_d;
            thrust::device_vector<int> kpanel_target_r_d;

            startTimer();

            ipiv_h_vector = ipiv_d_vector;

            if (restnum_kpanel < (int)n - (int)k) {
                std::vector<int> ipiv_h_idx((int)n);
                for (int idx = 0; idx < (int)n; idx++) ipiv_h_idx[idx] = idx;

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

            if (k + i >= n) break;

            // TRSM kpanel
            startTimer();
            {
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                trsm_half_recursive_optimized(
                    cublas_handle,
                    (int)(k + i),
                    (int)k,
                    (half*)A_d,
                    (int)n,
                    (half*)(A_d + (k + i) * n),
                    (int)n);
            }
            float trsm_kpanel_ms = stopTimer();
            detail.trsm_kpanel_time += trsm_kpanel_ms;

            // GEMM kpanel update
            startTimer();
            {
                half* A_d = (half*)thrust::raw_pointer_cast(A_device_vector.data());
                float alpha_f = -1.0f;
                float beta_f  = 1.0f;
                CUBLAS_CHECK(cublasGemmEx(
                    cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)(n - k - i), (int)k, (int)(k + i),
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

            if (debug_mode) {
                printf("[%d] kpanel i=%zu: swap=%.2f trsm=%.2f gemm=%.2f (panels=%zu)\n",
                       roll, i, swap_kpanel_ms, trsm_kpanel_ms, gemm_kpanel_ms, panels_in_block);
            }
        }
    }

    CUDA_CHECK(cudaEventRecord(total_lu_end));
    CUDA_CHECK(cudaEventSynchronize(total_lu_end));
    
    float entire_time_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&entire_time_ms, total_lu_begin, total_lu_end));

    // ========== 输出性能统计 ==========
    printf("\n========== Performance (avg of %d runs) ==========\n", roll_num);
    printf("Measured total:  %.2f ms\n", entire_time_ms / roll_num);
    printf("  TSLU:          %.2f ms\n", detail.tslu_time / roll_num);
    printf("  Swap panel:    %.2f ms\n", detail.swap_panel_time / roll_num);
    printf("  TRSM panel:    %.2f ms\n", detail.trsm_panel_time / roll_num);
    printf("  GEMM panel:    %.2f ms (%.1f TFLOPS)\n", 
           detail.gemm_panel_time / roll_num,
           detail.gemm_panel_ops / (detail.gemm_panel_time * 1e9));
    printf("  Swap kpanel:   %.2f ms\n", detail.swap_kpanel_time / roll_num);
    printf("  TRSM kpanel:   %.2f ms\n", detail.trsm_kpanel_time / roll_num);
    printf("  GEMM kpanel:   %.2f ms (%.1f TFLOPS)\n", 
           detail.gemm_kpanel_time / roll_num,
           detail.gemm_kpanel_ops / (detail.gemm_kpanel_time * 1e9));

    float total_lu_time = detail.tslu_time + detail.swap_panel_time +
                          detail.trsm_panel_time + detail.gemm_panel_time +
                          detail.swap_kpanel_time + detail.trsm_kpanel_time +
                          detail.gemm_kpanel_time;

    printf("\nSum of parts:    %.2f ms\n", total_lu_time / roll_num);
    printf("Overall:         %.2f TFLOPS\n",
           getrf_flops(n) * roll_num / (total_lu_time * 1e9));
    printf("Overhead:        %.2f ms (%.1f%%)\n", 
           (entire_time_ms - total_lu_time) / roll_num,
           100.0f * (entire_time_ms - total_lu_time) / entire_time_ms);

    // ========== 验证和对比（不计入性能）==========
    if (verifyResult) {
        computeMinusOfPAandLU_half(A_device_vector, oriA_device_vector, P_device_vector, (int)n);
    }

    if (compare_with_cusolver) {
        compareWithCusolver_half(A_seed_device_vector, n, roll_num, pivoting);
    }

    // ========== 清理 ==========
    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    hgetrfDestroy(hgetrf_handle);
    CUDA_CHECK(cudaFree(devInfo_d));
    CUDA_CHECK(cudaFree(tslu_workspace_d));
    CUDA_CHECK(cudaEventDestroy(total_lu_begin));
    CUDA_CHECK(cudaEventDestroy(total_lu_end));

    return 0;
}