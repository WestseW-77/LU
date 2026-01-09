#pragma once
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define MAGIC_SEED 261825ULL

// -----------------------------
// Error checking
// -----------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

static inline const char* cublasGetErrorString(cublasStatus_t status) {
    switch (status) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
        default: return "UNKNOWN CUBLAS STATUS";
    }
}

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t err = (call);                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cublasGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static inline const char* cusolverGetErrorString(cusolverStatus_t status) {
    switch (status) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
        default: return "UNKNOWN CUSOLVER STATUS";
    }
}

#define CUSOLVER_CHECK(call)                                                      \
    do {                                                                          \
        cusolverStatus_t err = (call);                                            \
        if (err != CUSOLVER_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuSolver error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cusolverGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// cuRAND minimal check
#ifndef CURAND_CHECK
#define CURAND_CHECK(call)                                                     \
    do {                                                                       \
        curandStatus_t err = (call);                                           \
        if (err != CURAND_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "cuRAND error at %s:%d - %d\n", __FILE__, __LINE__, \
                    (int)err);                                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)
#endif

// -----------------------------
// Timing (keep same API)
// -----------------------------
static cudaEvent_t default_start, default_end;

static inline void startTimer(cudaEvent_t& begin = default_start,
                              cudaEvent_t& end   = default_end) {
    CUDA_CHECK(cudaEventCreate(&begin));
    CUDA_CHECK(cudaEventRecord(begin));
    CUDA_CHECK(cudaEventCreate(&end));
}

static inline float stopTimer(cudaEvent_t& begin = default_start,
                              cudaEvent_t& end   = default_end) {
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, begin, end));
    CUDA_CHECK(cudaEventDestroy(begin));
    CUDA_CHECK(cudaEventDestroy(end));
    return ms;
}

// -----------------------------
// Conversion kernels
// -----------------------------
__global__ void float_to_half_kernel(const float* __restrict__ in,
                                     __half* __restrict__ out,
                                     int N) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx < N) out[idx] = __float2half(in[idx]);
}

__global__ void half_to_float_kernel(const __half* __restrict__ in,
                                     float* __restrict__ out,
                                     int N) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx < N) out[idx] = __half2float(in[idx]);
}

// -----------------------------
// Random normal matrix (half-only)
// Keep function name to avoid changing callers
// -----------------------------
template <typename T>
void generateNormalMatrix(thrust::device_vector<T>& A_device_vector, size_t m,
                          size_t n, size_t seed_offset = 0) {
    static_assert(std::is_same_v<T, __half>,
                  "This project is half-only now. generateNormalMatrix only supports __half.");

    T* dA = thrust::raw_pointer_cast(A_device_vector.data());

    thrust::device_vector<float> tmp(m * n);
    float* dtmp = thrust::raw_pointer_cast(tmp.data());

    curandGenerator_t gen;
    CURAND_CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    unsigned long long seed = (unsigned long long)MAGIC_SEED + (unsigned long long)seed_offset;
    CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed));
    CURAND_CHECK(curandGenerateNormal(gen, dtmp, m * n, 0.0f, 1.0f));
    CURAND_CHECK(curandDestroyGenerator(gen));

    int N = (int)(m * n);
    int block = 256;
    int grid = (N + block - 1) / block;
    float_to_half_kernel<<<grid, block>>>(dtmp, reinterpret_cast<__half*>(dA), N);
    CUDA_CHECK(cudaGetLastError());
}

// -----------------------------
// Print helpers (half-only)
// -----------------------------
template <typename T>
void printMatrixHost(thrust::host_vector<T>& A_host_vector, size_t m, size_t n) {
    static_assert(std::is_same_v<T, __half>,
                  "half-only: printMatrixHost only supports __half.");
    if (m <= 32 && n <= 32) {
        for (size_t i = 0; i < m; i++) {
            printf("%2zu: ", i);
            for (size_t j = 0; j < n; j++) {
                printf("%4.2f ", (double)__half2float(A_host_vector[j * m + i]));
            }
            printf("\n");
        }
    } else {
        printf("\nToo large to print\n");
    }
}

template <typename T>
void printMatrixDevice(thrust::device_vector<T>& A_device_vector, size_t m, size_t n,
                       const char* name) {
    static_assert(std::is_same_v<T, __half>,
                  "half-only: printMatrixDevice only supports __half.");
    bool donothing = false;
    if (donothing) return;
    printf("\n========== matrix %s ==========\n", name);
    if (m > 32 || n > 32) {
        printf("\nToo large to print\n");
    } else {
        thrust::host_vector<T> A_host_vector = A_device_vector;
        printMatrixHost(A_host_vector, m, n);
    }
    printf("\n========== matrix %s ==========\n", name);
}

// -----------------------------
// cleanMatrix: same signature as before
// IMPORTANT: lambda receives (i,j,A,lda)
// -----------------------------
template <typename T, class Func>
__global__ void cleanMatrix(T* A_d, size_t m, size_t n, size_t lda, Func func) {
    // grid stride loop
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < m;
         i += (size_t)blockDim.x * gridDim.x) {
        for (size_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n;
             j += (size_t)blockDim.y * gridDim.y) {
            func((int)i, (int)j, A_d, (int)lda);
        }
    }
}

// -----------------------------
// Frobenius norm (half-only) with custom reduction
// -----------------------------
__global__ void sum_squares_half_kernel(const __half* __restrict__ x,
                                       int64_t N,
                                       double* __restrict__ partials) {
    __shared__ double sh[256];
    double sum = 0.0;

    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) {
        float v = __half2float(x[idx]);
        sum += (double)v * (double)v;
    }

    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            sh[threadIdx.x] += sh[threadIdx.x + offset];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) partials[blockIdx.x] = sh[0];
}

template <typename T>
double computeFrobeniusNorm(size_t m, size_t n,
                            thrust::device_vector<T>& A_device_vector) {
    static_assert(std::is_same_v<T, __half>,
                  "half-only: computeFrobeniusNorm only supports __half.");

    const __half* dA = reinterpret_cast<const __half*>(
        thrust::raw_pointer_cast(A_device_vector.data()));
    const int64_t N = (int64_t)m * (int64_t)n;

    // fixed config: stable under memory pressure
    constexpr int threads = 256;
    int blocks = 4096;
    if (blocks <= 0) blocks = 1;

    thrust::device_vector<double> partials(blocks);
    double* partials_d = thrust::raw_pointer_cast(partials.data());

    sum_squares_half_kernel<<<blocks, threads>>>(dA, N, partials_d);
    CUDA_CHECK(cudaGetLastError());

    thrust::host_vector<double> partials_h = partials;
    double total = 0.0;
    for (double v : partials_h) total += v;
    return std::sqrt(total);
}
