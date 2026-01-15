#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

// ================== 参数配置 ==================
constexpr int col_process_size = 8;                 // 8 warps
constexpr int warp_size = 32;
constexpr int rows_per_lane = 8;
constexpr int row_process_size = warp_size * rows_per_lane;   // 256

constexpr int b_stage_count = 2;                    // stage0 + stage1
constexpr int b_stage_cols = col_process_size;      // 8
constexpr int warp_num = b_stage_cols;              // 8
constexpr int launch_bound = warp_size * warp_num;  // 256 threads

constexpr int a_block_rows = 4;                     // 每次处理 4 个 pivot 行/列

// A、B 列主序
#define A_h(i, j) __ldg(A + (i) + (j) * M)

static void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << " : " << cudaGetErrorString(e) << "\n";
        std::exit(1);
    }
}
static void ckblas(cublasStatus_t s, const char* msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        std::cerr << msg << " : cublas status " << int(s) << "\n";
        std::exit(1);
    }
}

// ================== cp.async 16B ==================
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr, const void* gmem_ptr) {
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

// ================== float<->half 转换 ==================
__global__ void f2h(const float* __restrict__ in, __half* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}
__global__ void h2f(const __half* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

// ================== 构造下三角 + 强化对角线（float） ==================
__global__ __launch_bounds__(256) void clear_tril_upper(float* A, int lda) {
    int row = threadIdx.x;
    int col = blockIdx.x;
    if (row < col) {
        A[row + col * lda] = 0.0f;
    } else {
        A[row + col * lda] += 2.0f;
    }
}

// ================== half2 bit-cast helper（解决编译报错） ==================
__device__ __forceinline__ int half2_to_int(__half2 x) {
    return *reinterpret_cast<int*>(&x);
}
__device__ __forceinline__ __half2 int_to_half2(int x) {
    return *reinterpret_cast<__half2*>(&x);
}

// ================== half2 性能优先 TRSM kernel（内层同步改成 syncwarp） ==================
__global__ __launch_bounds__(launch_bound) void trsm_half2_kernel(
    const __half* __restrict__ A,   // half: MxM lower, col-major
    __half* __restrict__ B,         // half: MxNrhs, col-major, in/out
    int M,
    int Nrhs) {

    if (M != row_process_size) return;

    int lane_id = threadIdx.x;   // 0..31
    int warp_id = threadIdx.y;   // 0..7
    int tx      = lane_id + warp_id * warp_size;
    int bx      = blockIdx.x;

    constexpr int cols_per_block = b_stage_cols * b_stage_count; // 16
    int base_col = bx * cols_per_block;
    if (base_col + cols_per_block > Nrhs) return;

    int col0 = base_col + warp_id;
    int col1 = base_col + b_stage_cols + warp_id;

    // ================== B 双 stage shared half ==================
    __shared__ __align__(16) __half B_sm[b_stage_count][row_process_size * warp_num];
    int b_warp_offset = warp_id * row_process_size;
    __half* B0 = B_sm[0] + b_warp_offset;
    __half* B1 = B_sm[1] + b_warp_offset;

    // ================== B: global->shared half (cp.async 16B via uint4) ==================
    {
        const uint4* g0 = reinterpret_cast<const uint4*>(B + col0 * M);
        const uint4* g1 = reinterpret_cast<const uint4*>(B + col1 * M);
        uint4* s0 = reinterpret_cast<uint4*>(B0);
        uint4* s1 = reinterpret_cast<uint4*>(B1);

        cp_async_cg_16(&s0[lane_id], &g0[lane_id]);
        cp_async_commit_group();
        cp_async_cg_16(&s1[lane_id], &g1[lane_id]);
        cp_async_commit_group();
    }

    // ================== A 对角线倒数（float） ==================
    __shared__ float A_diag[row_process_size];
#pragma unroll
    for (int idx = tx; idx < row_process_size; idx += launch_bound) {
        float aii = __half2float(A_h(idx, idx));
        A_diag[idx] = __frcp_rn(aii);
    }

    // 等 B 搬完 + A_diag 写完
    cp_async_wait_group_0();
    __syncthreads();

    // ================== 寄存器缓存：用 half2 缓存两列 ==================
    int row_base = lane_id;

    const int b_cache_i0 = rows_per_lane - 2;  // 6
    const int b_cache_i1 = rows_per_lane - 1;  // 7
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

    __half2 b_cache0 = __halves2half2(B0[r_cache0], B1[r_cache0]);
    __half2 b_cache1 = __halves2half2(B0[r_cache1], B1[r_cache1]);

    // ================== A 预取块（4 列）到 shared half ==================
    __shared__ __align__(16) __half Ablk[a_block_rows][row_process_size];

    for (int block = 0; block < row_process_size; block += a_block_rows) {

        // 预取 A 的 4 列：A(:, block + c)
        if (warp_id < a_block_rows) {
            int col = block + warp_id;
            const uint4* g = reinterpret_cast<const uint4*>(A + col * M);
            uint4* s = reinterpret_cast<uint4*>(Ablk[warp_id]);
            cp_async_cg_16(&s[lane_id], &g[lane_id]);
            cp_async_commit_group();
        }

        cp_async_wait_group_0();
        __syncthreads();  // 需要：其他 warps 也要读 Ablk

#pragma unroll
        for (int row_offset = 0; row_offset < a_block_rows; ++row_offset) {
            int row = block + row_offset;
            int row_lane = row & (warp_size - 1);

            const __half* Acol = Ablk[row_offset];

            __half a0 = Acol[r_a0];
            __half a1 = Acol[r_a1];
            __half a2 = Acol[r_a2];
            __half a3 = Acol[r_a3];

            // 求解 x：float 算，再压 half2
            __half2 x2 = __float2half2_rn(0.0f);

            if (lane_id == row_lane) {
                float inv = A_diag[row];

                __half2 brow2;
                if (row == r_cache0) {
                    brow2 = b_cache0;
                } else if (row == r_cache1) {
                    brow2 = b_cache1;
                } else {
                    brow2 = __halves2half2(B0[row], B1[row]);
                }

                float2 bf = __half22float2(brow2);
                x2 = __floats2half2_rn(bf.x * inv, bf.y * inv);
            }

            // 广播 x2：用 bit-cast + shuffle
            int x2_i = half2_to_int(x2);
            x2_i = __shfl_sync(0xffffffff, x2_i, row_lane);
            x2 = int_to_half2(x2_i);

            int first = 0;
            if (row >= row_base) first = (row - row_base) / warp_size + 1;

#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;

                __half2 b2 = __halves2half2(B0[r], B1[r]);
                __half  a  = Acol[r];
                __half2 a2b = __halves2half2(a, a);

                b2 = __hsub2(b2, __hmul2(a2b, x2));

                B0[r] = __low2half(b2);
                B1[r] = __high2half(b2);
            }

            if (first <= a_cache_i0) {
                int r = r_a0;
                __half2 b2 = __halves2half2(B0[r], B1[r]);
                __half2 a2b = __halves2half2(a0, a0);
                b2 = __hsub2(b2, __hmul2(a2b, x2));
                B0[r] = __low2half(b2);
                B1[r] = __high2half(b2);
            }
            if (first <= a_cache_i1) {
                int r = r_a1;
                __half2 b2 = __halves2half2(B0[r], B1[r]);
                __half2 a2b = __halves2half2(a1, a1);
                b2 = __hsub2(b2, __hmul2(a2b, x2));
                B0[r] = __low2half(b2);
                B1[r] = __high2half(b2);
            }
            if (first <= a_cache_i2) {
                __half2 a2b = __halves2half2(a2, a2);
                b_cache0 = __hsub2(b_cache0, __hmul2(a2b, x2));
            }
            if (first <= a_cache_i3) {
                __half2 a2b = __halves2half2(a3, a3);
                b_cache1 = __hsub2(b_cache1, __hmul2(a2b, x2));
            }

            if (lane_id == row_lane) {
                if (row == r_cache0) {
                    b_cache0 = x2;
                } else if (row == r_cache1) {
                    b_cache1 = x2;
                } else {
                    B0[row] = __low2half(x2);
                    B1[row] = __high2half(x2);
                }
            }

            // ✅ warp 内就够了：每个 warp 只碰自己的 B0/B1 段
            __syncwarp(0xffffffff);
        }
    }

    // 写回寄存器缓存（warp 内就够）
    B0[r_cache0] = __low2half(b_cache0);
    B1[r_cache0] = __high2half(b_cache0);
    B0[r_cache1] = __low2half(b_cache1);
    B1[r_cache1] = __high2half(b_cache1);

    __syncwarp(0xffffffff);

    // shared -> global：16B store（uint4）
    {
        const uint4* s0 = reinterpret_cast<const uint4*>(B0);
        const uint4* s1 = reinterpret_cast<const uint4*>(B1);
        uint4* g0 = reinterpret_cast<uint4*>(B + col0 * M);
        uint4* g1 = reinterpret_cast<uint4*>(B + col1 * M);

        g0[lane_id] = s0[lane_id];
        g1[lane_id] = s1[lane_id];
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <M> <Nrhs>\n";
        return 1;
    }

    int M = std::atoi(argv[1]);
    int Nrhs = std::atoi(argv[2]);

    if (M != row_process_size) {
        std::cerr << "This kernel requires M == " << row_process_size << "\n";
        return 1;
    }

    constexpr int cols_per_block = b_stage_cols * b_stage_count; // 16
    if (Nrhs % cols_per_block != 0) {
        std::cerr << "Nrhs must be divisible by " << cols_per_block << "\n";
        return 1;
    }

    std::cout << "M = " << M << ", Nrhs = " << Nrhs << "\n";

    // float：生成数据 + 参考 + 残差
    float *A_f=nullptr, *B_in_f=nullptr, *B_orig_f=nullptr, *B_cublas_f=nullptr, *B_out_f=nullptr;
    // half：kernel 输入输出
    __half *A_h_dev=nullptr, *B_h_dev=nullptr;

    ck(cudaMalloc(&A_f, M*M*sizeof(float)), "cudaMalloc A_f");
    ck(cudaMalloc(&B_in_f, M*Nrhs*sizeof(float)), "cudaMalloc B_in_f");
    ck(cudaMalloc(&B_orig_f, M*Nrhs*sizeof(float)), "cudaMalloc B_orig_f");
    ck(cudaMalloc(&B_cublas_f, M*Nrhs*sizeof(float)), "cudaMalloc B_cublas_f");
    ck(cudaMalloc(&B_out_f, M*Nrhs*sizeof(float)), "cudaMalloc B_out_f");

    ck(cudaMalloc(&A_h_dev, M*M*sizeof(__half)), "cudaMalloc A_h");
    ck(cudaMalloc(&B_h_dev, M*Nrhs*sizeof(__half)), "cudaMalloc B_h");

    // 随机数据
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);

    curandGenerateUniform(gen, A_f, M*M);
    clear_tril_upper<<<M, 256>>>(A_f, M);

    curandGenerateUniform(gen, B_in_f, M*Nrhs);
    ck(cudaDeviceSynchronize(), "sync init");

    // 备份 B_orig
    ck(cudaMemcpy(B_orig_f, B_in_f, M*Nrhs*sizeof(float), cudaMemcpyDeviceToDevice), "copy B_orig");
    ck(cudaMemcpy(B_cublas_f, B_in_f, M*Nrhs*sizeof(float), cudaMemcpyDeviceToDevice), "copy B_cublas");

    // 转 half 给 kernel
    int nA = M*M, nB = M*Nrhs;
    int bs = 256;
    f2h<<<(nA+bs-1)/bs, bs>>>(A_f, A_h_dev, nA);
    f2h<<<(nB+bs-1)/bs, bs>>>(B_in_f, B_h_dev, nB);
    ck(cudaDeviceSynchronize(), "sync f2h");

    // cuBLAS float TRSM（参考性能）
    cublasHandle_t handle=nullptr;
    ckblas(cublasCreate(&handle), "cublasCreate");
    float alpha = 1.0f;

    cudaEvent_t start=nullptr, stop=nullptr;
    ck(cudaEventCreate(&start), "event create start");
    ck(cudaEventCreate(&stop), "event create stop");

    // warmup cuBLAS
    ckblas(cublasStrsm(handle,
                       CUBLAS_SIDE_LEFT,
                       CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_N,
                       CUBLAS_DIAG_NON_UNIT,
                       M, Nrhs,
                       &alpha,
                       A_f, M,
                       B_cublas_f, M),
           "cublasStrsm warmup");
    ck(cudaDeviceSynchronize(), "sync cublas warmup");

    // 重新拷回输入（避免 warmup 改掉 B）
    ck(cudaMemcpy(B_cublas_f, B_in_f, M*Nrhs*sizeof(float), cudaMemcpyDeviceToDevice), "restore B_cublas");

    // timing cuBLAS
    cudaEventRecord(start);
    ckblas(cublasStrsm(handle,
                       CUBLAS_SIDE_LEFT,
                       CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_N,
                       CUBLAS_DIAG_NON_UNIT,
                       M, Nrhs,
                       &alpha,
                       A_f, M,
                       B_cublas_f, M),
           "cublasStrsm");
    cudaEventRecord(stop);
    ck(cudaEventSynchronize(stop), "sync stop cublas");

    float cublas_ms=0.0f;
    ck(cudaEventElapsedTime(&cublas_ms, start, stop), "elapsed cublas");

    // warmup kernel
    int grid = Nrhs / cols_per_block;
    dim3 block(warp_size, warp_num);
    trsm_half2_kernel<<<grid, block>>>(A_h_dev, B_h_dev, M, Nrhs);
    ck(cudaDeviceSynchronize(), "sync kernel warmup");

    // 重新拷回输入 half（避免 warmup 改掉 B）
    f2h<<<(nB+bs-1)/bs, bs>>>(B_in_f, B_h_dev, nB);
    ck(cudaDeviceSynchronize(), "restore B_h");

    // timing kernel
    cudaEventRecord(start);
    trsm_half2_kernel<<<grid, block>>>(A_h_dev, B_h_dev, M, Nrhs);
    cudaEventRecord(stop);
    ck(cudaEventSynchronize(stop), "sync stop kernel");

    float kernel_ms=0.0f;
    ck(cudaEventElapsedTime(&kernel_ms, start, stop), "elapsed kernel");

    // half 输出转 float
    h2f<<<(nB+bs-1)/bs, bs>>>(B_h_dev, B_out_f, nB);
    ck(cudaDeviceSynchronize(), "sync h2f");

    // ===== 残差计算：||B_orig - L*X|| =====
    std::vector<float> A_host(M*M);
    std::vector<float> X_host(M*Nrhs);
    std::vector<float> Borig_host(M*Nrhs);

    ck(cudaMemcpy(A_host.data(), A_f, M*M*sizeof(float), cudaMemcpyDeviceToHost), "copy A host");
    ck(cudaMemcpy(X_host.data(), B_out_f, M*Nrhs*sizeof(float), cudaMemcpyDeviceToHost), "copy X host");
    ck(cudaMemcpy(Borig_host.data(), B_orig_f, M*Nrhs*sizeof(float), cudaMemcpyDeviceToHost), "copy Borig host");

    double r_norm_sq=0.0, b_norm_sq=0.0, max_abs=0.0;
    for (int col=0; col<Nrhs; ++col) {
        int col_off = col*M;
        for (int row=0; row<M; ++row) {
            double sum=0.0;
            for (int k=0; k<=row; ++k) {
                sum += (double)A_host[row + k*M] * (double)X_host[k + col_off];
            }
            double r = (double)Borig_host[row + col_off] - sum;
            r_norm_sq += r*r;
            b_norm_sq += (double)Borig_host[row + col_off] * (double)Borig_host[row + col_off];
            double ar = std::abs(r);
            if (ar > max_abs) max_abs = ar;
        }
    }

    double r_norm = std::sqrt(r_norm_sq);
    double b_norm = std::sqrt(b_norm_sq);
    double rel = r_norm / (b_norm + 1e-12);

    // 性能输出
    double flops = (double)M * (double)M * (double)Nrhs;
    double kernel_tflops = flops / (kernel_ms * 1e-3) * 1e-12;
    double cublas_tflops = flops / (cublas_ms * 1e-3) * 1e-12;

    std::cout << "Half2-kernel residual ||B_orig - L*X||_F = " << r_norm << "\n";
    std::cout << "Half2-kernel relative residual = " << rel << "\n";
    std::cout << "Half2-kernel max abs residual = " << max_abs << "\n";
    std::cout << "Half2-kernel time (ms) = " << kernel_ms << "\n";
    std::cout << "cuBLAS Strsm time (ms) = " << cublas_ms << "\n";
    std::cout << "Half2-kernel TFLOPS (M*M*Nrhs) = " << kernel_tflops << "\n";
    std::cout << "cuBLAS TFLOPS (M*M*Nrhs) = " << cublas_tflops << "\n";

    // cleanup
    curandDestroyGenerator(gen);
    cublasDestroy(handle);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(A_f);
    cudaFree(B_in_f);
    cudaFree(B_orig_f);
    cudaFree(B_cublas_f);
    cudaFree(B_out_f);
    cudaFree(A_h_dev);
    cudaFree(B_h_dev);

    return 0;
}
