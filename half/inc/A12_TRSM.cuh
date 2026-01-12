// A12_TRSM.cuh - 增强优化版本
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

// 从全局内存读取（只读缓存优化）
#define L11_LOAD(i, j, j0, A, lda) __ldg(A + (j0 + i) + (size_t)(j0 + j) * lda)

// ============== 完全优化的 A12 TRSM Kernel ==============
template<int IB>
__global__ __launch_bounds__(256) void A12_trsm_kernel_half_optimized(
    half* __restrict__ A,
    int m, int n, int lda,
    int j0, int ib_actual)
{
    // ============== 核心参数 ==============
    constexpr int col_process_size = 8;
    constexpr int warp_size = 32;
    constexpr int rows_per_lane = IB / warp_size;
    constexpr int row_process_size = IB;
    constexpr int b_stage_count = 2;
    constexpr int b_stage_cols = col_process_size;
    constexpr int warp_num = b_stage_cols;
    constexpr int a_block_rows = 4;
    
    // L11 双缓冲开关
    constexpr bool USE_A_DOUBLE_BUFFER = (IB <= 128);  // 大矩阵时共享内存不够
    constexpr int a_stage_count = USE_A_DOUBLE_BUFFER ? (a_block_rows * 2) : a_block_rows;
    
    int ib = ib_actual;
    if (ib <= 0 || ib > IB) return;
    
    const int col0 = j0 + ib;
    const int ntrail = n - col0;
    if (ntrail <= 0) return;
    
    // ========== 线程和 block 索引 ==========
    int lane_id = threadIdx.x % warp_size;
    int warp_id = threadIdx.x / warp_size;
    int bx = blockIdx.x;
    
    int base_col = bx * b_stage_cols * b_stage_count;
    int b_warp_offset = warp_id * row_process_size;
    
    // ========== A12 双缓冲共享内存 ==========
    __shared__ __align__(16) float B_sm[b_stage_count][row_process_size * warp_num];
    float* B_sm0 = B_sm[0] + b_warp_offset;
    float* B_sm1 = B_sm[1] + b_warp_offset;
    int row_base = lane_id;
    
    // ========== 使用异步拷贝加载 A12（16 字节对齐）==========
    // half8 = 16 bytes
    struct alignas(16) half8_t {
        half data[8];
    };
    
    // 临时共享内存用于异步加载（half 格式）
    __shared__ __align__(16) half8_t B_temp[b_stage_count][warp_num][IB / 8];
    
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * b_stage_cols;
        if (global_col >= n) continue;
        
        // 每个线程异步加载 half8（16 字节）
        #pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int load_row = (lane_id + vec_idx * warp_size) * 8;
            if (load_row < ib && j0 + load_row < m) {
                const half8_t* g_ptr = reinterpret_cast<const half8_t*>(
                    A + (j0 + load_row) + (size_t)global_col * lda);
                half8_t* s_ptr = &B_temp[stage][warp_id][lane_id + vec_idx * warp_size];
                cp_async_cg_16(s_ptr, g_ptr);
            }
        }
        cp_async_commit_group();
    }
    
    // ========== L11 列的共享内存（双缓冲可选）==========
    __shared__ __align__(16) float L11_col[a_stage_count][row_process_size];
    int warp_row = warp_id % a_block_rows;
    int warp_half = warp_id / a_block_rows;
    
    // ========== 异步预取首组 L11 列 ==========
    auto load_L11_col = [&](int col_idx, int buf_idx) {
        if (col_idx < ib && buf_idx < a_stage_count) {
            for (int load_idx = lane_id + warp_half * warp_size; load_idx < IB; load_idx += warp_size * 2) {
                if (load_idx < ib && j0 + load_idx < m && j0 + col_idx < n) {
                    half val = L11_LOAD(load_idx, col_idx, j0, A, lda);
                    L11_col[buf_idx][load_idx] = __half2float(val);
                } else if (load_idx < IB) {
                    L11_col[buf_idx][load_idx] = 0.0f;
                }
            }
        }
    };
    
    // 预取第一组
    load_L11_col(warp_row, warp_row);
    
    if constexpr (USE_A_DOUBLE_BUFFER) {
        // 预取第二组
        load_L11_col(a_block_rows + warp_row, a_block_rows + warp_row);
    }
    
    // ========== 等待 A12 异步加载完成并转换为 float ==========
    cp_async_wait_group_0();
    __syncthreads();
    
    // 转换 half -> float（提高精度）
    for (int stage = 0; stage < b_stage_count; ++stage) {
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        
        #pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int load_row = (lane_id + vec_idx * warp_size) * 8;
            if (load_row < IB) {
                half8_t temp = B_temp[stage][warp_id][lane_id + vec_idx * warp_size];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (load_row + i < ib) {
                        B_sm_stage[load_row + i] = __half2float(temp.data[i]);
                    } else {
                        B_sm_stage[load_row + i] = 0.0f;
                    }
                }
            }
        }
    }
    __syncthreads();
    
    // ========== 寄存器缓存 ==========
    const int b_cache_i0 = rows_per_lane - 2;
    const int b_cache_i1 = rows_per_lane - 1;
    const int r_cache0 = row_base + b_cache_i0 * warp_size;
    const int r_cache1 = row_base + b_cache_i1 * warp_size;
    
    float b0_cache0 = (r_cache0 < ib) ? B_sm0[r_cache0] : 0.0f;
    float b1_cache0 = (r_cache0 < ib) ? B_sm1[r_cache0] : 0.0f;
    float b0_cache1 = (r_cache1 < ib) ? B_sm0[r_cache1] : 0.0f;
    float b1_cache1 = (r_cache1 < ib) ? B_sm1[r_cache1] : 0.0f;
    
    // ========== 主循环：按 4 列 block 处理 ==========
    constexpr int a_block_count = (IB + a_block_rows - 1) / a_block_rows;
    
    for (int block = 0; block < ib; block += a_block_rows) {
        int block_idx = block / a_block_rows;
        
        int buf_base = 0;
        if constexpr (USE_A_DOUBLE_BUFFER) {
            int set = block_idx & 1;
            buf_base = set * a_block_rows;
        }
        
        // 处理当前 block 的 4 列
        for (int col_offset = 0; col_offset < a_block_rows && (block + col_offset) < ib; ++col_offset) {
            int col = block + col_offset;
            int col_lane = col & (warp_size - 1);
            const float* L11_col_ptr = L11_col[buf_base + col_offset];
            
            // 缓存 L11 的 4 个元素
            const int l_cache_i0 = rows_per_lane - 4;
            const int l_cache_i1 = rows_per_lane - 3;
            const int l_cache_i2 = b_cache_i0;
            const int l_cache_i3 = b_cache_i1;
            const int r_l0 = row_base + l_cache_i0 * warp_size;
            const int r_l1 = row_base + l_cache_i1 * warp_size;
            
            float l_cache0 = (r_l0 < ib) ? L11_col_ptr[r_l0] : 0.0f;
            float l_cache1 = (r_l1 < ib) ? L11_col_ptr[r_l1] : 0.0f;
            float l_cache2 = (r_cache0 < ib) ? L11_col_ptr[r_cache0] : 0.0f;
            float l_cache3 = (r_cache1 < ib) ? L11_col_ptr[r_cache1] : 0.0f;
            
            // 计算解
            float x0 = 0.0f, x1 = 0.0f;
            if (lane_id == col_lane && col < IB) {
                if (col == r_cache0) {
                    x0 = b0_cache0; x1 = b1_cache0;
                } else if (col == r_cache1) {
                    x0 = b0_cache1; x1 = b1_cache1;
                } else {
                    x0 = B_sm0[col]; x1 = B_sm1[col];
                }
            }
            
            // 广播
            x0 = __shfl_sync(0xffffffff, x0, col_lane);
            x1 = __shfl_sync(0xffffffff, x1, col_lane);
            
            // 更新
            int first = (col >= row_base) ? ((col - row_base) / warp_size + 1) : 0;
            
            #pragma unroll
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                if (r >= ib) break;
                float l = L11_col_ptr[r];
                B_sm0[r] -= x0 * l;
                B_sm1[r] -= x1 * l;
            }
            
            if (first <= l_cache_i0 && r_l0 < ib) {
                B_sm0[r_l0] -= x0 * l_cache0;
                B_sm1[r_l0] -= x1 * l_cache0;
            }
            if (first <= l_cache_i1 && r_l1 < ib) {
                B_sm0[r_l1] -= x0 * l_cache1;
                B_sm1[r_l1] -= x1 * l_cache1;
            }
            if (first <= l_cache_i2 && r_cache0 < ib) {
                b0_cache0 -= x0 * l_cache2;
                b1_cache0 -= x1 * l_cache2;
            }
            if (first <= l_cache_i3 && r_cache1 < ib) {
                b0_cache1 -= x0 * l_cache3;
                b1_cache1 -= x1 * l_cache3;
            }
            
            // 写回
            if (lane_id == col_lane && col < IB) {
                if (col == r_cache0) {
                    b0_cache0 = x0; b1_cache0 = x1;
                } else if (col == r_cache1) {
                    b0_cache1 = x0; b1_cache1 = x1;
                } else {
                    B_sm0[col] = x0; B_sm1[col] = x1;
                }
            }
            __syncthreads();
        }
        
        // 预取下一个 block
        if constexpr (USE_A_DOUBLE_BUFFER) {
            if (block_idx >= 0) {
                int next_block = block_idx + 2;
                if (next_block < a_block_count) {
                    int other = 1 - (block_idx & 1);
                    int next_col = next_block * a_block_rows + warp_row;
                    load_L11_col(next_col, other * a_block_rows + warp_row);
                }
            }
            __syncthreads();
        } else {
            int next_block = block_idx + 1;
            if (next_block < a_block_count) {
                int next_col = next_block * a_block_rows + warp_row;
                load_L11_col(next_col, warp_row);
            }
            __syncthreads();
        }
    }
    
    // 写回缓存
    if (r_cache0 < ib) {
        B_sm0[r_cache0] = b0_cache0;
        B_sm1[r_cache0] = b1_cache0;
    }
    if (r_cache1 < ib) {
        B_sm0[r_cache1] = b0_cache1;
        B_sm1[r_cache1] = b1_cache1;
    }
    __syncthreads();
    
    // ========== 向量化写回全局内存 ==========
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * b_stage_cols;
        if (global_col >= n) continue;
        
        float* B_sm_stage = B_sm[stage] + b_warp_offset;
        half* B_out = A + j0 + (size_t)global_col * lda;
        
        // 使用 half8 向量化写回
        #pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int store_row = (lane_id + vec_idx * warp_size) * 8;
            if (store_row < ib && j0 + store_row + 7 < m) {
                half8_t temp;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    temp.data[i] = __float2half(B_sm_stage[store_row + i]);
                }
                half8_t* g_ptr = reinterpret_cast<half8_t*>(B_out + store_row);
                *g_ptr = temp;
            } else {
                // 边界处理
                for (int i = 0; i < 8; ++i) {
                    if (store_row + i < ib && j0 + store_row + i < m) {
                        B_out[store_row + i] = __float2half(B_sm_stage[store_row + i]);
                    }
                }
            }
        }
    }
}

// ... 原有版本保持不变 ...

// 启动函数
inline void launch_A12_trsm(
    half* dA,
    int m, int n, int lda,
    int j0, int ib,
    cudaStream_t stream = 0)
{
    const int col0   = j0 + ib;
    const int ntrail = n - col0;
    if (ntrail <= 0 || ib <= 0) return;

    if (ib > 256) {
        fprintf(stderr, "[A12_TRSM] ERROR: ib=%d > 256 is not supported.\n", ib);
        std::exit(EXIT_FAILURE);
    }

    constexpr int col_process_size = 8;
    constexpr int b_stage_count = 2;
    int cols_per_block = col_process_size * b_stage_count;
    int grid_x = (ntrail + cols_per_block - 1) / cols_per_block;
    if (grid_x <= 0) grid_x = 1;
    
    dim3 grid(grid_x);
    dim3 block(256);
    
    if (ib <= 32) {
        A12_trsm_kernel_half_optimized<32><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else if (ib <= 64) {
        A12_trsm_kernel_half_optimized<64><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else if (ib <= 128) {
        A12_trsm_kernel_half_optimized<128><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    } else {
        A12_trsm_kernel_half_optimized<256><<<grid, block, 0, stream>>>(
            dA, m, n, lda, j0, ib);
    }
    
    CUDA_CHECK(cudaGetLastError());
}