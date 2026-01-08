// A12_TRSM.cuh
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

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

using half = __half;

// ========== 全 half 的 A12 = L11^{-1} * A12 TRSM ==========
//
//  L11: unit-lower，位置 [j0 .. j0+ib-1, j0 .. j0+ib-1]
//  A12: 同样行，列 [j0+ib .. n-1]，只更新顶部 ib 行
//
//  每个线程负责一列 A12(:, j) 的顶部 ib 元素，前代：
//      x[i] = b[i] - sum_{k=0..i-1} L(i,k) * x[k]
//  diag(L) = 1，不除
//

// 优化版本：L11 缓存在 shared，按模板 IB=32/64/128
template<int IB>
__global__ void A12_trsm_kernel_half_opt(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib_actual)
{
    int ib = ib_actual;
    if (ib <= 0) return;
    if (ib > IB) ib = IB;

    const int col0   = j0 + ib;
    const int ntrail = n - col0;
    if (ntrail <= 0) return;

    extern __shared__ half shmem_h[];
    half* L11_sh = shmem_h;
    const int stride = IB;

    const half h_zero = __float2half(0.0f);

    // 1) 把 L11 从 global 载入 shared(half)
    for (int col = threadIdx.x; col < ib; col += blockDim.x) {
        int gj = j0 + col;
        if (gj >= n) continue;
        for (int row = 0; row < ib; ++row) {
            int gi = j0 + row;
            half v = h_zero;
            if (gi < m) {
                v = A[gi + (size_t)gj * lda];
            }
            L11_sh[row * stride + col] = v;
        }
    }
    __syncthreads();

    // 2) 每个线程处理一列 trailing 列 j
    int j_trail = blockIdx.x * blockDim.x + threadIdx.x;
    if (j_trail >= ntrail) return;

    int j = col0 + j_trail;

    float x[IB];

    #pragma unroll
    for (int ii = 0; ii < IB; ++ii) {
        x[ii] = 0.0f;
    }

    #pragma unroll
    for (int ii = 0; ii < IB; ++ii) {
        if (ii >= ib) break;
        int gi = j0 + ii;
        float v = 0.0f;
        if (gi < m && j < n) {
            half hv = A[gi + (size_t)j * lda];
            v = __half2float(hv);
        }
        x[ii] = v;
    }

    // 3) forward substitution
    #pragma unroll
    for (int ii = 0; ii < IB; ++ii) {
        if (ii >= ib) break;

        float sum = x[ii];

        #pragma unroll
        for (int kk = 0; kk < IB; ++kk) {
            if (kk >= ii) break;
            half Lik_h = L11_sh[ii * stride + kk];
            float Lik  = __half2float(Lik_h);
            float xk   = x[kk];
            sum -= Lik * xk;
        }

        x[ii] = sum;
    }

    // 4) store back
    #pragma unroll
    for (int ii = 0; ii < IB; ++ii) {
        if (ii >= ib) break;
        int gi = j0 + ii;
        if (gi < m && j < n) {
            A[gi + (size_t)j * lda] = __float2half(x[ii]);
        }
    }
}


// fallback 版本：不用 shared 的通用实现
// 重要：这里明确只支持 ib <= 128（因为本 kernel 用了固定长度数组）
__global__ void A12_trsm_kernel_half_fallback(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib)
{
    if (ib <= 0 || ib > 128) return;

    int col0   = j0 + ib;
    int ntrail = n - col0;
    if (ntrail <= 0) return;

    int j_trail = blockIdx.x * blockDim.x + threadIdx.x;
    if (j_trail >= ntrail) return;

    int j = col0 + j_trail;

    const half h_zero = __float2half(0.0f);

    float x[128];

    // load
    for (int ii = 0; ii < ib; ++ii) {
        int gi = j0 + ii;
        float v = 0.0f;
        if (gi < m && j < n) {
            half hv = A[gi + (size_t)j * lda];
            v = __half2float(hv);
        }
        x[ii] = v;
    }

    // solve
    for (int ii = 0; ii < ib; ++ii) {
        int irow = j0 + ii;
        if (irow >= m) break;

        float sum = x[ii];

        for (int kk = 0; kk < ii; ++kk) {
            int kcol = j0 + kk;

            half Lik_h = h_zero;
            if (irow < m && kcol < n) {
                Lik_h = A[irow + (size_t)kcol * lda];
            }
            float Lik = __half2float(Lik_h);
            float xk  = x[kk];

            sum -= Lik * xk;
        }

        x[ii] = sum;
    }

    // store
    for (int ii = 0; ii < ib; ++ii) {
        int gi = j0 + ii;
        if (gi < m && j < n) {
            A[gi + (size_t)j * lda] = __float2half(x[ii]);
        }
    }
}


// 启动函数: A12(top) = L11^{-1} * A12(top)
inline void launch_A12_trsm(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    cudaStream_t stream = 0)
{
    int col0   = j0 + ib;
    int ntrail = n - col0;
    if (ntrail <= 0 || ib <= 0) return;

    // 保护：你的 opt/fallback 都默认支持到 128
    if (ib > 128) {
        // 这里你可以选择直接 exit（更像库行为），
        // 也可以 return（静默跳过会更难 debug，所以我建议 exit）
        fprintf(stderr, "[A12_TRSM] ERROR: ib=%d > 128 is not supported by custom TRSM.\n", ib);
        std::exit(EXIT_FAILURE);
    }

    const int block_x = 128;
    int grid_x = (ntrail + block_x - 1) / block_x;
    if (grid_x <= 0) grid_x = 1;

    dim3 grid(grid_x);
    dim3 block(block_x);

    auto launch_opt = [&](auto IB_tag) {
        constexpr int IB = decltype(IB_tag)::value;
        size_t shmem_size = sizeof(half) * IB * IB;
        A12_trsm_kernel_half_opt<IB><<<grid, block, shmem_size, stream>>>(
            dA, m, n, lda, j0, ib
        );
        CUDA_CHECK(cudaGetLastError());
    };

    switch (ib) {
    case 32:
        launch_opt(std::integral_constant<int,32>{});
        break;
    case 64:
        launch_opt(std::integral_constant<int,64>{});
        break;
    case 128:
        launch_opt(std::integral_constant<int,128>{});
        break;
    default:
        A12_trsm_kernel_half_fallback<<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib
        );
        CUDA_CHECK(cudaGetLastError());
        break;
    }
}
