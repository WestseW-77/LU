// A12_TRSM_Half2_Simplified.cuh
// 功能：求解下三角方程组 L * X = B，其中L是下三角矩阵，B是右侧矩阵块
// 核心思想：对每一列，通过前向替换求解 X，然后更新后续行
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

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
using half2 = __half2;

// ============================================================================
// 异步内存拷贝指令（Ampere架构及以上）
// 优势：CPU可以继续执行其他指令，而不用等待内存拷贝完成
// ============================================================================
__device__ __forceinline__ void cp_async_cg_16(void* smem_ptr,
                                                 const void* gmem_ptr) {
#if __CUDA_ARCH__ >= 800
    unsigned int smem_addr = __cvta_generic_to_shared(smem_ptr);
    // 一次拷贝16字节（128位）
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

__device__ __forceinline__ void cp_async_commit_group() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" : : : "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if __CUDA_ARCH__ >= 800
    // 等待所有异步拷贝完成
    asm volatile("cp.async.wait_group 0;\n" : : : "memory");
#endif
}

// 使用__ldg进行只读缓存优化的全局内存读取
#define L11_LOAD(i, j, j0, A, lda) __ldg(A + (j0 + i) + (size_t)(j0 + j) * lda)

// ============================================================================
// 主Kernel：使用__half2进行SIMD优化的A12 TRSM
// 
// 矩阵布局：
//   A = [ L11 | A12 ]
//       [ 0   | A22 ]
// 其中L11是j0:j0+ib的下三角矩阵（已求解）
// A12是j0:j0+ib × (j0+ib):n的矩阵块（待求解）
// 
// 求解：L11 * X = A12，结果写回A12
// 
// 优化策略：
// 1. __half2向量化：一次处理2列，利用SIMD指令加速
// 2. 异步拷贝：提前加载数据，隐藏内存延迟
// 3. 共享内存：减少全局内存访问
// 4. 寄存器缓存：缓存频繁访问的数据
// ============================================================================
template<int IB>  // IB是L11矩阵的大小（编译期常量）
__global__ __launch_bounds__(256) void A12_trsm_kernel_half_optimized(
    half* __restrict__ A,   // 输入输出矩阵（列主序）
    int m, int n, int lda,  // m行，n列，leading dimension
    int j0, int ib_actual)  // j0是起始列，ib_actual是L11的实际大小
{
    // ========================================================================
    // 线程组织和参数配置
    // ========================================================================
    constexpr int warp_size = 32;
    constexpr int col_process_size = 8;     // 每个warp处理8列
    constexpr int b_stage_count = 2;        // 双缓冲：同时处理2个stage
    constexpr int rows_per_lane = IB / warp_size;  // 每个线程处理的行数
    
    int ib = ib_actual;
    if (ib <= 0 || ib > IB) return;

    const int col0 = j0 + ib;               // A12的起始列
    const int ntrail = n - col0;            // A12的列数
    if (ntrail <= 0) return;

    // 线程索引
    int lane_id = threadIdx.x % warp_size;  // warp内的线程ID [0, 31]
    int warp_id = threadIdx.x / warp_size;  // warp ID [0, 7]
    int bx = blockIdx.x;
    
    // 每个block处理 col_process_size * b_stage_count = 16 列
    int base_col = bx * col_process_size * b_stage_count;
    int b_warp_offset = warp_id * IB;       // 每个warp的共享内存偏移

    // ========================================================================
    // 共享内存布局（使用__half2优化）
    // B_sm存储A12矩阵块：[IB行 × 16列]，分2个stage，每个stage 8列
    // 使用half2可以一次操作2列数据，提高内存带宽利用率
    // ========================================================================
    __shared__ __align__(16) half2 B_sm_h2[b_stage_count][IB * (col_process_size/2)];
    
    // 为每个warp提供独立的视图
    half* B_sm0 = reinterpret_cast<half*>(B_sm_h2[0]) + b_warp_offset;
    half* B_sm1 = reinterpret_cast<half*>(B_sm_h2[1]) + b_warp_offset;
    
    int row_base = lane_id;  // 每个线程的基础行索引

    // ========================================================================
    // 步骤1：异步加载A12矩阵到共享内存
    // 每8个half打包成一个128位向量，提高拷贝效率
    // ========================================================================
    struct alignas(16) half8_t {
        half data[8];
    };
    __shared__ __align__(16) half8_t B_temp[b_stage_count][col_process_size][IB / 8];

    // 加载2个stage的数据
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * col_process_size;
        if (global_col >= n) continue;

        // 每个warp加载IB行数据，使用向量化加载
#pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int load_row = (lane_id + vec_idx * warp_size) * 8;
            if (load_row < ib && j0 + load_row < m) {
                const half8_t* g_ptr = reinterpret_cast<const half8_t*>(
                    A + (j0 + load_row) + (size_t)global_col * lda);
                half8_t* s_ptr = &B_temp[stage][warp_id][lane_id + vec_idx * warp_size];
                cp_async_cg_16(s_ptr, g_ptr);  // 异步拷贝16字节
            }
        }
        cp_async_commit_group();  // 提交当前批次的拷贝
    }

    // ========================================================================
    // L11列的共享内存：存储L11矩阵的列向量
    // 使用双缓冲优化：当前计算使用一个buffer，同时预取下一个block到另一个buffer
    // ========================================================================
    constexpr int a_block_rows = 4;  // 每次处理4列
    constexpr bool USE_A_DOUBLE_BUFFER = (IB <= 128);
    constexpr int a_stage_count = USE_A_DOUBLE_BUFFER ? (a_block_rows * 2) : a_block_rows;
    
    __shared__ __align__(16) half L11_col[a_stage_count][IB];

    int warp_row = warp_id % a_block_rows;
    int warp_half = warp_id / a_block_rows;

    // 加载L11列的lambda函数
    auto load_L11_col = [&](int col_idx, int buf_idx) {
        if (col_idx < ib && buf_idx < a_stage_count) {
            // 每个线程加载多个元素
            for (int load_idx = lane_id + warp_half * warp_size; 
                 load_idx < IB; 
                 load_idx += warp_size * 2) {
                if (load_idx < ib && j0 + load_idx < m && j0 + col_idx < n) {
                    half val = L11_LOAD(load_idx, col_idx, j0, A, lda);
                    L11_col[buf_idx][load_idx] = val;
                } else if (load_idx < IB) {
                    L11_col[buf_idx][load_idx] = __float2half(0.0f);
                }
            }
        }
    };

    // 预加载第一批L11列
    load_L11_col(warp_row, warp_row);
    if constexpr (USE_A_DOUBLE_BUFFER) {
        load_L11_col(a_block_rows + warp_row, a_block_rows + warp_row);
    }

    // ========================================================================
    // 等待异步加载完成，然后重组数据到共享内存
    // ========================================================================
    cp_async_wait_group_0();
    __syncthreads();

    // 将向量化的数据重组到共享内存
    for (int stage = 0; stage < b_stage_count; ++stage) {
        half* B_sm_stage = (stage == 0) ? B_sm0 : B_sm1;
#pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int load_row = (lane_id + vec_idx * warp_size) * 8;
            if (load_row < IB) {
                half8_t temp = B_temp[stage][warp_id][lane_id + vec_idx * warp_size];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    if (load_row + i < ib) {
                        B_sm_stage[load_row + i] = temp.data[i];
                    } else {
                        B_sm_stage[load_row + i] = __float2half(0.0f);
                    }
                }
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // 寄存器缓存：缓存最后两行数据到寄存器
    // 这两行访问频率最高，放到寄存器可以减少共享内存访问
    // 使用half2可以同时缓存2列数据
    // ========================================================================
    const int b_cache_i0 = rows_per_lane - 2;
    const int b_cache_i1 = rows_per_lane - 1;
    const int r_cache0 = row_base + b_cache_i0 * warp_size;
    const int r_cache1 = row_base + b_cache_i1 * warp_size;

    // 使用half2同时存储两列
    half2 b_cache0_h2 = (r_cache0 < ib) ? 
        __halves2half2(B_sm0[r_cache0], B_sm1[r_cache0]) : __float2half2_rn(0.0f);
    half2 b_cache1_h2 = (r_cache1 < ib) ? 
        __halves2half2(B_sm0[r_cache1], B_sm1[r_cache1]) : __float2half2_rn(0.0f);

    // ========================================================================
    // 步骤2：主循环 - 前向替换求解
    // 对每一列j，计算 X[j] = B[j] - Σ(L[i,j] * X[i]) for i < j
    // ========================================================================
    constexpr int a_block_count = (IB + a_block_rows - 1) / a_block_rows;

    for (int block = 0; block < ib; block += a_block_rows) {
        int block_idx = block / a_block_rows;
        
        // 确定使用哪个buffer（双缓冲）
        int buf_base = 0;
        if constexpr (USE_A_DOUBLE_BUFFER) {
            int set = block_idx & 1;
            buf_base = set * a_block_rows;
        }

        // 处理当前block的每一列
        for (int col_offset = 0; col_offset < a_block_rows && (block + col_offset) < ib; ++col_offset) {
            int col = block + col_offset;
            int col_lane = col & (warp_size - 1);  // col在warp中的位置
            const half* L11_col_ptr = L11_col[buf_base + col_offset];

            // ------------------------------------------------------------------
            // 缓存L11的4个关键元素到寄存器
            // ------------------------------------------------------------------
            const int l_cache_i0 = rows_per_lane - 4;
            const int l_cache_i1 = rows_per_lane - 3;
            const int r_l0 = row_base + l_cache_i0 * warp_size;
            const int r_l1 = row_base + l_cache_i1 * warp_size;

            half l_cache0 = (r_l0 < ib) ? L11_col_ptr[r_l0] : __float2half(0.0f);
            half l_cache1 = (r_l1 < ib) ? L11_col_ptr[r_l1] : __float2half(0.0f);
            half l_cache2 = (r_cache0 < ib) ? L11_col_ptr[r_cache0] : __float2half(0.0f);
            half l_cache3 = (r_cache1 < ib) ? L11_col_ptr[r_cache1] : __float2half(0.0f);

            // ------------------------------------------------------------------
            // 读取当前列的解 X[col]（2列打包成half2）
            // ------------------------------------------------------------------
            half2 x_h2 = __float2half2_rn(0.0f);
            if (lane_id == col_lane && col < IB) {
                // 如果解在寄存器缓存中，直接读取
                if (col == r_cache0) {
                    x_h2 = b_cache0_h2;
                } else if (col == r_cache1) {
                    x_h2 = b_cache1_h2;
                } else {
                    // 否则从共享内存读取
                    x_h2 = __halves2half2(B_sm0[col], B_sm1[col]);
                }
            }

            // ------------------------------------------------------------------
            // 在warp内广播解 X[col]（使用shuffle指令）
            // half2可以一次广播2个half值
            // ------------------------------------------------------------------
            half x0 = __shfl_sync(0xffffffff, __low2half(x_h2), col_lane);
            half x1 = __shfl_sync(0xffffffff, __high2half(x_h2), col_lane);
            x_h2 = __halves2half2(x0, x1);

            // ------------------------------------------------------------------
            // 更新后续行：B[i] = B[i] - L[i,col] * X[col]
            // 使用half2 SIMD指令同时计算2列
            // ------------------------------------------------------------------
            int first = (col >= row_base) ? ((col - row_base) / warp_size + 1) : 0;

            // 主循环：更新大部分行（未缓存的行）
#pragma unroll 2
            for (int i = first; i < rows_per_lane - 4; ++i) {
                int r = row_base + i * warp_size;
                if (r >= ib) break;
                
                half l = L11_col_ptr[r];
                half2 l_h2 = __half2half2(l);  // 复制到half2的两个lane
                half2 b_h2 = __halves2half2(B_sm0[r], B_sm1[r]);  // 读取2列
                
                // SIMD计算：b = b - x * l（一次更新2列）
                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));
                
                // 写回共享内存
                B_sm0[r] = __low2half(b_h2);
                B_sm1[r] = __high2half(b_h2);
            }

            // 更新缓存的4行（使用寄存器缓存的L和B）
            if (first <= l_cache_i0 && r_l0 < ib) {
                half2 l_h2 = __half2half2(l_cache0);
                half2 b_h2 = __halves2half2(B_sm0[r_l0], B_sm1[r_l0]);
                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));
                B_sm0[r_l0] = __low2half(b_h2);
                B_sm1[r_l0] = __high2half(b_h2);
            }
            if (first <= l_cache_i1 && r_l1 < ib) {
                half2 l_h2 = __half2half2(l_cache1);
                half2 b_h2 = __halves2half2(B_sm0[r_l1], B_sm1[r_l1]);
                b_h2 = __hsub2(b_h2, __hmul2(x_h2, l_h2));
                B_sm0[r_l1] = __low2half(b_h2);
                B_sm1[r_l1] = __high2half(b_h2);
            }
            if (first <= b_cache_i0 && r_cache0 < ib) {
                half2 l_h2 = __half2half2(l_cache2);
                b_cache0_h2 = __hsub2(b_cache0_h2, __hmul2(x_h2, l_h2));
            }
            if (first <= b_cache_i1 && r_cache1 < ib) {
                half2 l_h2 = __half2half2(l_cache3);
                b_cache1_h2 = __hsub2(b_cache1_h2, __hmul2(x_h2, l_h2));
            }

            // ------------------------------------------------------------------
            // 写回解 X[col]
            // ------------------------------------------------------------------
            if (lane_id == col_lane && col < IB) {
                if (col == r_cache0) {
                    b_cache0_h2 = x_h2;
                } else if (col == r_cache1) {
                    b_cache1_h2 = x_h2;
                } else {
                    B_sm0[col] = __low2half(x_h2);
                    B_sm1[col] = __high2half(x_h2);
                }
            }
            __syncthreads();
        }

        // ------------------------------------------------------------------
        // 预取下一个block的L11列（双缓冲优化）
        // ------------------------------------------------------------------
        if constexpr (USE_A_DOUBLE_BUFFER) {
            int next_block = block_idx + 2;
            if (next_block < a_block_count) {
                int other = 1 - (block_idx & 1);
                int next_col = next_block * a_block_rows + warp_row;
                load_L11_col(next_col, other * a_block_rows + warp_row);
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

    // ========================================================================
    // 步骤3：写回寄存器缓存到共享内存
    // ========================================================================
    if (r_cache0 < ib) {
        B_sm0[r_cache0] = __low2half(b_cache0_h2);
        B_sm1[r_cache0] = __high2half(b_cache0_h2);
    }
    if (r_cache1 < ib) {
        B_sm0[r_cache1] = __low2half(b_cache1_h2);
        B_sm1[r_cache1] = __high2half(b_cache1_h2);
    }
    __syncthreads();

    // ========================================================================
    // 步骤4：向量化写回全局内存
    // 使用128位写入，提高内存带宽利用率
    // ========================================================================
    for (int stage = 0; stage < b_stage_count; ++stage) {
        int global_col = col0 + base_col + warp_id + stage * col_process_size;
        if (global_col >= n) continue;

        half* B_sm_stage = (stage == 0) ? B_sm0 : B_sm1;
        half* B_out = A + j0 + (size_t)global_col * lda;

        // 向量化写回（每次8个half = 128位）
#pragma unroll
        for (int vec_idx = 0; vec_idx < (IB / 8) / warp_size; ++vec_idx) {
            int store_row = (lane_id + vec_idx * warp_size) * 8;
            if (store_row < ib && j0 + store_row + 7 < m) {
                // 对齐写入：打包8个half一次写入
                half8_t temp;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    temp.data[i] = B_sm_stage[store_row + i];
                }
                half8_t* g_ptr = reinterpret_cast<half8_t*>(B_out + store_row);
                *g_ptr = temp;
            } else {
                // 边界情况：逐个元素写入
                for (int i = 0; i < 8; ++i) {
                    if (store_row + i < ib && j0 + store_row + i < m) {
                        B_out[store_row + i] = B_sm_stage[store_row + i];
                    }
                }
            }
        }
    }
}

// ============================================================================
// 启动函数：根据ib大小选择合适的模板实例
// ============================================================================
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

    // 计算grid大小：每个block处理16列
    constexpr int col_process_size = 8;
    constexpr int b_stage_count = 2;
    int cols_per_block = col_process_size * b_stage_count;
    int grid_x = (ntrail + cols_per_block - 1) / cols_per_block;
    if (grid_x <= 0) grid_x = 1;

    dim3 grid(grid_x);
    dim3 block(256);  // 256线程 = 8 warps

    // 根据ib选择合适的模板参数
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
