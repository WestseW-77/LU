#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <vector>

#include "getrf.cuh"

#define SWAP_LEN_PANEL 2048

using data_type = __half;

void trsm(cublasHandle_t cublasH, long m, long n, float alpha, float *A, long lda,
          float *B, long ldb, long nb) {
    float sonefloat = 1.0, snegonefloat = -1.0;
    if (m <= nb) {
        CUBLAS_CHECK(cublasStrsm(cublasH, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                                 CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha, A,
                                 lda, B, ldb));
        return;
    }

    trsm(cublasH, m / 2, n, alpha, A, lda, B, ldb, nb);

    long left = m - m / 2;
    CUBLAS_CHECK(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, left, n, m / 2,
                             &snegonefloat, A + m / 2, lda, B, ldb, &sonefloat,
                             B + m / 2, ldb));

    trsm(cublasH, left, n, alpha, A + m / 2 + m / 2 * lda, lda, B + m / 2, ldb, nb);
}

// 期望返回矩阵 A，其中上三角是 U，下三角是 L，另外返回矩阵 P 对后续矩阵更新的选主元操作进行指导
void TSLU(cublasHandle_t cublasH, long m, long n, float alpha, __half *A, long lda, half *Workspace, int *devIpiv, int *devInfo) {
    // 首先判断是执行选主元的 kernel 还是不选主元的
    if (devIpiv == NULL) { 
        noPviotingLU(A, n, m, lda, alpha, pivoting, devIpiv, devInfo);
    }
    else {
        PivotingLU(A, n, m, lda, alpha, pivoting, devIpiv, devInfo);
    }
}

int parseArgs(int argc, char *argv[], size_t &n, size_t &k, size_t &nb,
              bool &verifyResult, bool &pivoting, bool &debug_mode,
              bool &compare_with_cusolver);

template <typename T>
void compareWithCusolver(thrust::device_vector<T> &A_device_vector, size_t n,
                         int roll_num, float &cusolver_lu_time, bool pivoting);

template <typename T>
void computeMinusOfPAandLU(thrust::device_vector<T> &A_device_vector,
                           thrust::device_vector<T> &oriA_device_vector,
                           thrust::device_vector<T> &P_device_vector, int n);

template <typename T>
void launchSwapByPivotingKernel(int *source_raw_ptr, int *target_raw_ptr,
                                int swap_size, int swap_len, int stride, T *A_d,
                                T *temp_d, int blocksPerGrid, int threadsPerBlock);

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelRead(int *source, int *target, int swap_size,
                                         int swap_len, int stride, T *A, T *temp);

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelWrite(int *source, int *target, int swap_size,
                                          int swap_len, int stride, T *A, T *temp);

template <typename T, size_t threadsPerRow>
__global__ void swapByPivotingKernel(int *__restrict__ source,
                                     int *__restrict__ target, int swap_size,
                                     int swap_len, int stride, T *__restrict__ A_d,
                                     T *__restrict__ temp);

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

    if (status != 0) {
        printf("Please execute the above command\n");
        return 1;
    }

    if constexpr (std::is_same_v<data_type, float>) {
        printf("data type using float\n");
    } else if constexpr (std::is_same_v<data_type, double>) {
        printf("data type using double\n");
    } else if constexpr (std::is_same_v<data_type, __half>) {
        printf("data type using half\n");
    }

    // 打印当前显卡
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device name: %s\n", prop.name);

    int maxBlocksPerGrid = prop.multiProcessorCount;
    // 每个 block 包含 256 个线程
    int threadsPerBlock = 256;
    // h100 的话有 114 个 block
    int blocksPerGrid = maxBlocksPerGrid;

    // 初始化 A 矩阵，用来测试
    thrust::device_vector<data_type> A_device_vector(n * n);
    auto A_d = thrust::raw_pointer_cast(A_device_vector.data());
    generateNormalMatrix(A_device_vector, n, n);
    printMatrixDevice(A_device_vector, n, n, "A device");

    thrust::device_vector<data_type> P_device_vector;
    auto P_d = thrust::raw_pointer_cast(P_device_vector.data());
    thrust::device_vector<data_type> oriA_device_vector;
    auto oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());
    // 需要对比结果的话就初始化一个单位矩阵 P 用于记录，并且保存了一份 A 矩阵的副本，应该是最后用于检验的
    if (verifyResult) {
        P_device_vector.resize(n * n);
        P_d = thrust::raw_pointer_cast(P_device_vector.data());
        thrust::for_each(thrust::counting_iterator<size_t>(0),
                         thrust::counting_iterator<size_t>(n),
                         [P_d, n] __device__(size_t i) {
                             if constexpr (std::is_same_v<data_type, float>)
                                 P_d[i * n + i] = 1.0f;
                             else if constexpr (std::is_same_v<data_type, double>)
                                 P_d[i * n + i] = 1.0;
                            else if constexpr (std::is_same_v<data_type, __half>)
                                 P_d[i * n + i] = __float2half_rn(1.0f);
                         });
        oriA_device_vector = A_device_vector;
        oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());
    }

    // 初始化 cublas 相关句柄
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    // 初始化 cuSOLVER 相关句柄
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));

    // 迭代轮次
    int roll_num = 3;
    if (verifyResult) roll_num = 1;

    float cusolver_lu_time = 0.0f;
    if (compare_with_cusolver) {
        compareWithCusolver(A_device_vector, n, roll_num, cusolver_lu_time,
                            pivoting);
    }

    // 为了交换准备空间
    thrust::device_vector<data_type> temp_device_vector(n * n);
    auto temp_d = thrust::raw_pointer_cast(temp_device_vector.data());

    // 预分配 TSLU 工作空间
    int buffer_size;
    auto A_raw_d = thrust::raw_pointer_cast(A_device_vector.data());
    if constexpr (std::is_same_v<data_type, float>) {
        CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
            cusolver_handle, n, nb, reinterpret_cast<float *>(A_raw_d), n,
            &buffer_size));
    } else if constexpr (std::is_same_v<data_type, double>) {
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
            cusolver_handle, n, nb, reinterpret_cast<double *>(A_raw_d), n,
            &buffer_size));
    }

    data_type *tslu_workspace_d;
    CUDA_CHECK(
        cudaMalloc((void **)&tslu_workspace_d, sizeof(data_type) * buffer_size));
    CUDA_CHECK(cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
    // 预分配 TSLU 状态
    int *devInfo_d;
    CUDA_CHECK(cudaMalloc((void **)&devInfo_d, sizeof(int)));
    CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
    // 预分配 TSLU ipiv 数组
    thrust::device_vector<int> ipiv_d_vector(n);
    // 获取到原始指针
    auto ipiv_d = thrust::raw_pointer_cast(ipiv_d_vector.data());
    thrust::host_vector<int> ipiv_h_vector(n);

    DoubleBlockingLUBreakDownDetail detail;

    cudaEvent_t total_lu_begin, total_lu_end;

    startTimer(total_lu_begin, total_lu_end);

    for (int roll = 0; roll < roll_num; roll++) {
        //  执行 memset
        CUDA_CHECK(cudaMemset(tslu_workspace_d, 0, sizeof(data_type) * buffer_size));
        CUDA_CHECK(cudaMemset(devInfo_d, 0, sizeof(int)));
        thrust::fill(ipiv_d_vector.begin(), ipiv_d_vector.end(), 0);
        thrust::fill(ipiv_h_vector.begin(), ipiv_h_vector.end(), 0);
        generateNormalMatrix(A_device_vector, n, n);
        // n 是矩阵的维度， k 是 double blocking 中的一小块的长度， nb 是 double Blocking 中内层的再分一块，也是最小单位
        // 长度为 k 的一组 panel 的 double-blocking 划分
        for (size_t i = 0; i < n; i += k) {
            //  一组长度为 nb 的 panel 中执行对应操作
            for (size_t j = i; j < i + k; j += nb) {
                startTimer();
                // TSLU 分解
                data_type *A_d = thrust::raw_pointer_cast(A_device_vector.data());
                // 选主元，这边是调用 cusolver 的得到交换矩阵 P，这是局部的，以panel大小为分界线
                // devIpiv 为 NULL 就不执行选主元，为指针的话就是在这里存入换了谁，部分选主元只会选当前列下方最大的上来
                // 所以 +j 就是因为前面的部分不会再被碰到了，而且也存在因为 LU 分解的 panel 是越来越小的，所以是一个相对值
                if (pivoting) {
                    if constexpr (std::is_same_v<data_type, float>) {
                        CUSOLVER_CHECK(cusolverDnSgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<float *>(A_d + j + j * n), n,
                            reinterpret_cast<float *>(tslu_workspace_d), ipiv_d + j,
                            devInfo_d));
                    } else if constexpr (std::is_same_v<data_type, double>) {
                        CUSOLVER_CHECK(cusolverDnDgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<double *>(A_d + j + j * n), n,
                            reinterpret_cast<double *>(tslu_workspace_d), ipiv_d + j,
                            devInfo_d));
                    }
                    else if constexpr (std::is_same_v<data_type, __half>) { 
                        TSLU();
                    }
                } else {
                    if constexpr (std::is_same_v<data_type, float>) {
                        CUSOLVER_CHECK(cusolverDnSgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<float *>(A_d + j + j * n), n,
                            reinterpret_cast<float *>(tslu_workspace_d), NULL,
                            devInfo_d));
                    } else if constexpr (std::is_same_v<data_type, double>) {
                        CUSOLVER_CHECK(cusolverDnDgetrf(
                            cusolver_handle, n - j, nb,
                            reinterpret_cast<double *>(A_d + j + j * n), n,
                            reinterpret_cast<double *>(tslu_workspace_d), NULL,
                            devInfo_d));
                    }
                    else if constexpr (std::is_same_v<data_type, __half>) {
                        TSLU();
                    }
                }
                detail.tslu_time += stopTimer();

                // 求 P 用来做校验， 这是整体的 P，在做校验的时候直接乘到 A 上去
                if (pivoting && verifyResult) {
                    for (size_t idx = 0; idx < nb; idx++) {
                        // 主元选择结果
                        ipiv_h_vector = ipiv_d_vector;
                        // 索引是全局的哪一行
                        size_t global_idx = j + idx;
                        // 本来这个位置和谁互换了，但是这里的是相对位置，相对的是从 j 位置开始往后加这么多索引
                        size_t local_pivot = ipiv_h_vector[global_idx] - 1;
                        // 主元是全局的哪一行
                        size_t global_pivot = j + local_pivot;
                        // 如果不相等就是进行了选主元操作，进行交换操作
                        if (global_idx != global_pivot) {
                            if constexpr (std::is_same_v<data_type, float>) {
                                CUBLAS_CHECK(cublasSswap(
                                    cublas_handle, n,
                                    reinterpret_cast<float *>(P_d + global_idx), n,
                                    reinterpret_cast<float *>(P_d + global_pivot),
                                    n));
                            } else if constexpr (std::is_same_v<data_type, double>) {
                                CUBLAS_CHECK(cublasDswap(
                                    cublas_handle, n,
                                    reinterpret_cast<double *>(P_d + global_idx), n,
                                    reinterpret_cast<double *>(P_d + global_pivot),
                                    n));
                            }
                        }
                    }
                }
                // 处理了这一 panel 之后，还有多少列等待被处理
                int restnum_panel = i + k - j - nb;
                thrust::host_vector<int> source;
                thrust::host_vector<int> target;
                thrust::device_vector<int> source_d;
                thrust::device_vector<int> target_d;

                startTimer();

                if (pivoting) {
                    ipiv_h_vector = ipiv_d_vector;
                    std::vector<int> ipiv_h_idx(n);
                    // 在 host 端获取了一个换主元信息的数组，记录的是全局的索引
                    for (int idx = 0; idx < ipiv_h_vector.size(); idx++) {
                        ipiv_h_idx[idx] = idx;
                    }

                    for (int idx = 0; idx < nb; idx++) {
                        std::swap(ipiv_h_idx[ipiv_h_vector[idx + j] - 1],
                                  ipiv_h_idx[idx]);
                    }
                    // 记录下了源地址和目标地址，用于后面的交换操作
                    for (int idx = 0; idx < n; idx++) {
                        if (ipiv_h_idx[idx] != idx) {
                            source.push_back(ipiv_h_idx[idx]);
                            target.push_back(idx);
                        }
                    }

                    source_d = source;
                    target_d = target;
                    // 第一个 if 处进入的是本大块中，本身 panel 不会触及的地方
                    // GPU 上进行交换 ，第一个 if 应当只对第一小块无效，只有每个大块的第一小块才会显示剩余的块数等于 k-nb
                    // 目前的执行会在这里走入 6 次第一个 if
                    if (restnum_panel < k - nb) {
                        // swap_len 的含义, 因为不是第一个小块，所以不存在 0 次的情况，具体指的是要换多少列
                        // 之所以是这个条件，是因为这个 if 内部是对本块中 panel 前面的列应用交换，也就是说他的 panelLU 在小块内是需要换行的
                        // swap_size 的含义是有多少行要进行交换
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(source_d.data()),
                            thrust::raw_pointer_cast(target_d.data()),
                            source_d.size(), j - i, n, A_d + j + i * n, temp_d,
                            blocksPerGrid, threadsPerBlock);
                    }
                    // 这里的话则是除了最后一块的时候都会进来
                    // 作用是更新本块内 panelLU 的后续矩阵，所以本块内的最后一块不需要进入更新
                    if (restnum_panel > 0) {
                        launchSwapByPivotingKernel(
                            thrust::raw_pointer_cast(source_d.data()),
                            thrust::raw_pointer_cast(target_d.data()),
                            source_d.size(), restnum_panel, n,
                            A_d + j + (nb + j) * n, temp_d, blocksPerGrid,
                            threadsPerBlock);
                    }
                }

                detail.swap_panel_time += stopTimer();
                // 更新完一组后，直接 continue
                if (nb + j - i >= k) {
                    continue;
                }
                // 解三角方程，求 U
                startTimer();
                data_type alpha = 1.0f;
                if constexpr (std::is_same_v<data_type, float>) {
                    CUBLAS_CHECK(cublasStrsm(
                        cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, nb + j - i, nb,
                        reinterpret_cast<float *>(&alpha),
                        reinterpret_cast<float *>(A_d + i + i * n), n,
                        reinterpret_cast<float *>(A_d + i + (nb + j) * n), n));
                } else if constexpr (std::is_same_v<data_type, double>) {
                    CUBLAS_CHECK(cublasDtrsm(
                        cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_UNIT, nb + j - i, nb,
                        reinterpret_cast<double *>(&alpha),
                        reinterpret_cast<double *>(A_d + i + i * n), n,
                        reinterpret_cast<double *>(A_d + i + (nb + j) * n), n));
                }

                detail.trsm_panel_time += stopTimer();
                // 矩阵乘法更新
                alpha = -1.0f;
                data_type beta = 1.0f;
                startTimer();
                if constexpr (std::is_same_v<data_type, float>) {
                    CUBLAS_CHECK(cublasSgemm_v2(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - nb - j, nb,
                        nb + j - i, reinterpret_cast<float *>(&alpha),
                        reinterpret_cast<float *>(A_d + (nb + j) + i * n), n,
                        reinterpret_cast<float *>(A_d + i + (nb + j) * n), n,
                        reinterpret_cast<float *>(&beta),
                        reinterpret_cast<float *>(A_d + (nb + j) + (nb + j) * n),
                        n));
                } else if constexpr (std::is_same_v<data_type, double>) {
                    CUBLAS_CHECK(cublasDgemm_v2(
                        cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - nb - j, nb,
                        nb + j - i, reinterpret_cast<double *>(&alpha),
                        reinterpret_cast<double *>(A_d + (nb + j) + i * n), n,
                        reinterpret_cast<double *>(A_d + i + (nb + j) * n), n,
                        reinterpret_cast<double *>(&beta),
                        reinterpret_cast<double *>(A_d + (nb + j) + (nb + j) * n),
                        n));
                }

                detail.gemm_panel_time += stopTimer();
                detail.gemm_panel_ops += 2.0f * (n - nb - j) * nb * (nb + j - i);
            }

            // 上面结束了一个小块中的所有运算，现在要把这些看作一个块，对其他块进行运算所需要的更新操作
            int restnum_kpanel = n - i - k;

            thrust::host_vector<int> kpanel_source_i;
            thrust::host_vector<int> kpanel_target_i;
            thrust::device_vector<int> kpanel_source_i_d;
            thrust::device_vector<int> kpanel_target_i_d;
            thrust::host_vector<int> kpanel_source_r;
            thrust::host_vector<int> kpanel_target_r;
            thrust::device_vector<int> kpanel_source_r_d;
            thrust::device_vector<int> kpanel_target_r_d;

            startTimer();

            if (pivoting) {
                // 只要当前不是第一块
                if (restnum_kpanel < n - k) {
                    std::vector<int> ipiv_h_idx(n);
                    for (int idx = 0; idx < n; idx++) {
                        ipiv_h_idx[idx] = idx;
                    }
                    for (int idx = 0; idx < k; idx++) {
                        int target_row = ipiv_h_vector[i + idx] - 1;
                        if (idx >= nb) {
                            target_row += nb * (idx / nb);
                        }
                        std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                    }
                    for (int idx = 0; idx < ipiv_h_idx.size(); idx++) {
                        if (idx != ipiv_h_idx[idx]) {
                            kpanel_source_i.push_back(ipiv_h_idx[idx]);
                            kpanel_target_i.push_back(idx);
                        }
                    }
                    kpanel_source_i_d = kpanel_source_i;
                    kpanel_target_i_d = kpanel_target_i;
                    // 这里的 i 数字还在这边的开头，所以实际上这边的换行只会对已经处理的前块进行处理
                    launchSwapByPivotingKernel(
                        thrust::raw_pointer_cast(kpanel_source_i_d.data()),
                        thrust::raw_pointer_cast(kpanel_target_i_d.data()),
                        kpanel_source_i_d.size(), i, n, A_d + i, temp_d,
                        blocksPerGrid, threadsPerBlock);
                }
                // 只要当前不是最后一大块, 交换只换了后面的一个大块
                if (restnum_kpanel > 0) {
                    std::vector<int> ipiv_h_idx(n);
                    for (int idx = 0; idx < n; idx++) {
                        ipiv_h_idx[idx] = idx;
                    }
                    for (int idx = 0; idx < i + k; idx++) {
                        int target_row = ipiv_h_vector[idx] - 1;
                        if (idx >= nb) {
                            target_row += nb * (idx / nb);
                        }
                        std::swap(ipiv_h_idx[target_row], ipiv_h_idx[idx]);
                    }
                    for (int idx = 0; idx < ipiv_h_idx.size(); idx++) {
                        if (idx != ipiv_h_idx[idx]) {
                            kpanel_source_r.push_back(ipiv_h_idx[idx]);
                            kpanel_target_r.push_back(idx);
                        }
                    }
                    kpanel_source_r_d = kpanel_source_r;
                    kpanel_target_r_d = kpanel_target_r;
                    launchSwapByPivotingKernel(
                        thrust::raw_pointer_cast(kpanel_source_r_d.data()),
                        thrust::raw_pointer_cast(kpanel_target_r_d.data()),
                        kpanel_source_r_d.size(), k, n, A_d + (k + i) * n, temp_d,
                        blocksPerGrid, threadsPerBlock);
                }
            }
            detail.swap_kpanel_time += stopTimer();
            if (k + i >= n) break;

            // 解三角方程，求 U
            data_type alpha = 1.0f;
            startTimer();
            // 单精度的 trsm 有自己的 kernel，双精度则采用调取 cublas [question]:这里有个 512 小调料
            if constexpr (std::is_same_v<data_type, float>) {
                trsm(cublas_handle, k + i, k, alpha, reinterpret_cast<float *>(A_d), n, reinterpret_cast<float *>(A_d + (k + i) * n), n, 512);
            } else if constexpr (std::is_same_v<data_type, double>) {
                CUBLAS_CHECK(cublasDtrsm(
                    cublas_handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                    CUBLAS_OP_N, CUBLAS_DIAG_UNIT, k + i, k,
                    reinterpret_cast<double *>(&alpha),
                    reinterpret_cast<double *>(A_d), n,
                    reinterpret_cast<double *>(A_d + (k + i) * n), n));
            }

            detail.trsm_kpanel_time += stopTimer();
            startTimer();
            // 矩阵乘法更新
            alpha = -1.0f;
            data_type beta = 1.0f;
            if constexpr (std::is_same_v<data_type, float>) {
                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - i, k, k + i,
                    reinterpret_cast<float *>(&alpha),
                    reinterpret_cast<float *>(A_d + (k + i)), n,
                    reinterpret_cast<float *>(A_d + (k + i) * n), n,
                    reinterpret_cast<float *>(&beta),
                    reinterpret_cast<float *>(A_d + (k + i) + (k + i) * n), n));
            } else if constexpr (std::is_same_v<data_type, double>) {
                CUBLAS_CHECK(cublasDgemm(
                    cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n - k - i, k, k + i,
                    reinterpret_cast<double *>(&alpha),
                    reinterpret_cast<double *>(A_d + (k + i)), n,
                    reinterpret_cast<double *>(A_d + (k + i) * n), n,
                    reinterpret_cast<double *>(&beta),
                    reinterpret_cast<double *>(A_d + (k + i) + (k + i) * n), n));
            }

            detail.gemm_kpanel_time += stopTimer();
            detail.gemm_kpanel_ops += 2.0f * (n - k - i) * k * (k + i);
        }
    }

    printf("entire time: %f ms\n",
           stopTimer(total_lu_begin, total_lu_end) / roll_num);
    printf("TSLU time: %f ms\n", detail.tslu_time / roll_num);
    printf("Swap panel time: %f ms\n", detail.swap_panel_time / roll_num);
    printf("Trsm panel time: %f ms\n", detail.trsm_panel_time / roll_num);
    printf("Gemm panel time: %f ms\n", detail.gemm_panel_time / roll_num);
    printf("Gemm panel %f TFLOPS\n",
           detail.gemm_panel_ops / (detail.gemm_panel_time * 1e9));
    printf("Swap kpanel time: %f ms\n", detail.swap_kpanel_time / roll_num);
    printf("Trsm kpanel time: %f ms\n", detail.trsm_kpanel_time / roll_num);
    printf("Gemm kpanel time: %f ms\n", detail.gemm_kpanel_time / roll_num);
    printf("Gemm kpanel %f TFLOPS\n",
           detail.gemm_kpanel_ops / (detail.gemm_kpanel_time * 1e9));
    float total_lu_time = detail.tslu_time + detail.swap_panel_time +
                          detail.trsm_panel_time + detail.gemm_panel_time +
                          detail.swap_kpanel_time + detail.trsm_kpanel_time +
                          detail.gemm_kpanel_time;

    printf("double-blocking LU: %lf ms\n", total_lu_time / roll_num);
    printf("double-blocking LU: %lf TFLOPS\n",
           double(2.0 / 3.0) * std::pow(n, 3) * roll_num / (total_lu_time * 1e9));

    if (compare_with_cusolver) {
        printf("Speedup: %f\n", cusolver_lu_time / total_lu_time);
    }

    if (verifyResult) {
        computeMinusOfPAandLU<data_type>(A_device_vector, oriA_device_vector,
                                         P_device_vector, n);
    }

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
    CUDA_CHECK(cudaFree(devInfo_d));
    CUDA_CHECK(cudaFree(tslu_workspace_d));

    return 0;
}

/**
 * function: parse arguments
 * @param argc: number of arguments
 * @param argv: arguments
 * @param n: matrix size
 * @param k: matrix size
 * @param nb: block size
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
        for (int i = 4; i < argc; i++)
            if (strcmp(argv[i], "-v") == 0) {
                verifyResult = true;
            } else if (strcmp(argv[i], "-p") == 0) {
                pivoting = true;
            } else if (strcmp(argv[i], "-d") == 0) {
                debug_mode = true;
            } else if (strcmp(argv[i], "-c") == 0) {
                compare_with_cusolver = true;
            }
    } else {
        printf("Default usage: %s <n>\n", argv[0]);
        printf("Usage: %s <n> <k> <nb>\n", argv[0]);
        return 1;
    }
    return 0;
}

// 计算 PA - LU，得到算法的绝对误差和相对误差
template <typename T>
void computeMinusOfPAandLU(thrust::device_vector<T> &A_device_vector,
                           thrust::device_vector<T> &oriA_device_vector,
                           thrust::device_vector<T> &P_device_vector, int n) {
    cublasHandle_t cublas_handle;
    CUBLAS_CHECK(cublasCreate(&cublas_handle));

    auto A_norm = computeFrobeniusNorm(n, n, oriA_device_vector);
    auto A_d = thrust::raw_pointer_cast(A_device_vector.data());
    auto oriA_d = thrust::raw_pointer_cast(oriA_device_vector.data());
    auto P_d = thrust::raw_pointer_cast(P_device_vector.data());
    // compute PA
    thrust::device_vector<T> PA_device_vector(n * n);
    auto PA_d = thrust::raw_pointer_cast(PA_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm_v2(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            reinterpret_cast<float *>(&alpha), reinterpret_cast<float *>(P_d), n,
            reinterpret_cast<float *>(oriA_d), n, reinterpret_cast<float *>(&beta),
            reinterpret_cast<float *>(PA_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm_v2(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            reinterpret_cast<double *>(&alpha), reinterpret_cast<double *>(P_d), n,
            reinterpret_cast<double *>(oriA_d), n, reinterpret_cast<double *>(&beta),
            reinterpret_cast<double *>(PA_d), n));
    }

    dim3 gridDim((n + 15) / 16, (n + 15) / 16);
    dim3 blockDim(16, 16);

    // compute LU
    thrust::device_vector<T> L_device_vector(n * n);
    thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                 L_device_vector.begin());
    auto L_d = thrust::raw_pointer_cast(L_device_vector.data());
    cleanMatrix<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(L_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, T *L_d, int lda) {
            T zero, one;
            if constexpr (std::is_same_v<T, float>) {
                zero = 0.0f;
                one = 1.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                zero = 0.0;
                one = 1.0;
            }
            if (i < n && j < n) {
                if (i < j) {
                    L_d[i + j * lda] = zero;
                }
                if (i == j) {
                    L_d[i + j * lda] = one;
                }
            }
        });
    thrust::device_vector<T> U_device_vector(n * n);
    thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                 U_device_vector.begin());
    auto U_d = thrust::raw_pointer_cast(U_device_vector.data());
    cleanMatrix<<<gridDim, blockDim>>>(
        thrust::raw_pointer_cast(U_device_vector.data()), n, n, n,
        [=] __device__(int i, int j, T *U_d, int lda) {
            T zero, one;
            if constexpr (std::is_same_v<T, float>) {
                zero = 0.0f;
                one = 1.0f;
            } else if constexpr (std::is_same_v<T, double>) {
                zero = 0.0;
                one = 1.0;
            }
            if (i < n && j < n) {
                if (i > j) {
                    U_d[i + j * lda] = zero;
                }
            }
        });
    thrust::device_vector<T> LU_device_vector(n * n);
    auto LU_d = thrust::raw_pointer_cast(LU_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        float alpha = 1.0f;
        float beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm_v2(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            reinterpret_cast<float *>(&alpha), reinterpret_cast<float *>(L_d), n,
            reinterpret_cast<float *>(U_d), n, reinterpret_cast<float *>(&beta),
            reinterpret_cast<float *>(LU_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        double alpha = 1.0;
        double beta = 0.0;
        CUBLAS_CHECK(cublasDgemm_v2(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
            reinterpret_cast<double *>(&alpha), reinterpret_cast<double *>(L_d), n,
            reinterpret_cast<double *>(U_d), n, reinterpret_cast<double *>(&beta),
            reinterpret_cast<double *>(LU_d), n));
    }

    // compute PA - LU
    thrust::device_vector<T> PAminusLU_device_vector(n * n);
    auto PAminusLU_d = thrust::raw_pointer_cast(PAminusLU_device_vector.data());
    if constexpr (std::is_same_v<T, float>) {
        auto alpha = 1.0f;
        auto beta = -1.0f;
        CUBLAS_CHECK(cublasSgeam(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
            reinterpret_cast<float *>(&alpha), reinterpret_cast<float *>(PA_d), n,
            reinterpret_cast<float *>(&beta), reinterpret_cast<float *>(LU_d), n,
            reinterpret_cast<float *>(PAminusLU_d), n));
    } else if constexpr (std::is_same_v<T, double>) {
        auto alpha = 1.0;
        auto beta = -1.0;
        CUBLAS_CHECK(cublasDgeam(
            cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
            reinterpret_cast<double *>(&alpha), reinterpret_cast<double *>(PA_d), n,
            reinterpret_cast<double *>(&beta), reinterpret_cast<double *>(LU_d), n,
            reinterpret_cast<double *>(PAminusLU_d), n));
    }

    T Minus_norm = computeFrobeniusNorm(n, n, PAminusLU_device_vector);
    printf("|PA - LU| Frobenius norm: %.16f\n", Minus_norm);
    printf("|A| Frobenius norm: %f\n", A_norm);
    printf("PA - LU / |A| Frobenius norm: %.16f\n", Minus_norm / A_norm);

    CUBLAS_CHECK(cublasDestroy(cublas_handle));
}

template <typename T>
void compareWithCusolver(thrust::device_vector<T> &A_device_vector, size_t n,
                         int roll_num, float &cusolver_lu_time, bool pivoting) {
    printf("Starting cusolver LU decomposition...\n");
    thrust::device_vector<T> A_device_vector_copy(n * n);
    cusolverDnHandle_t cusolver_handle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
    for (int i = 0; i < roll_num; i++) {
        thrust::copy(A_device_vector.begin(), A_device_vector.end(),
                     A_device_vector_copy.begin());
        auto A_cusolver = thrust::raw_pointer_cast(A_device_vector_copy.data());
        int *dinfo;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dinfo), sizeof(int)));
        // 调用 cuSolver 的 lu 分解，注意这些全部是在 device 上进行的
        int lwork;
        if constexpr (std::is_same_v<T, float>) {
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(cusolver_handle, n, n,
                                                       A_cusolver, n, &lwork));
        } else if constexpr (std::is_same_v<T, double>) {
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolver_handle, n, n,
                                                       A_cusolver, n, &lwork))
        }
        T *dwork;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dwork), lwork * sizeof(T)));
        int *dpiv;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&dpiv), n * sizeof(int)));
        startTimer();
        if (pivoting) {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle, n, n, A_cusolver, n,
                                                dwork, dpiv, dinfo));
            } else if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_CHECK(cusolverDnDgetrf(cusolver_handle, n, n, A_cusolver, n,
                                                dwork, dpiv, dinfo))
            }
        } else {
            if constexpr (std::is_same_v<T, float>) {
                CUSOLVER_CHECK(cusolverDnSgetrf(cusolver_handle, n, n, A_cusolver, n,
                                                dwork, NULL, dinfo));
            } else if constexpr (std::is_same_v<T, double>) {
                CUSOLVER_CHECK(cusolverDnDgetrf(cusolver_handle, n, n, A_cusolver, n,
                                                dwork, NULL, dinfo))
            }
        }
        cusolver_lu_time += stopTimer();
    }

    printf("cusolver lu time: %f ms\n", cusolver_lu_time / roll_num);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolver_handle));
}

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelRead(int *source, int *target, int swap_size,
                                         int swap_len, int stride, T *A_d, T *temp) {
    // grid stride loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < swap_size;
         idx += blockDim.x * gridDim.x) {
        if (idx < swap_size) {
            auto temp4idx = temp + idx * swap_len;
            for (size_t j = 0; j < swap_len; j++) {
                temp4idx[j] = A_d[source[idx] + j * stride];
            }
        }
    }
}

template <typename T, int threadsPerRow>
__global__ void swapByPivotingKernelWrite(int *source, int *target, int swap_size,
                                          int swap_len, int stride, T *A_d,
                                          T *temp) {
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < swap_size;
         idx += blockDim.x * gridDim.x) {
        if (idx < swap_size) {
            auto temp4idx = temp + idx * swap_len;
            for (size_t j = 0; j < swap_len; j++) {
                A_d[target[idx] + j * stride] = temp4idx[j];
            }
        }
    }
}

// exchange 核函数
template <typename T, size_t threadsPerRow>
__global__ void swapByPivotingKernel(int *__restrict__ source,
                                     int *__restrict__ target, int swap_size,
                                     int swap_len, int stride, T *__restrict__ A_d,
                                     T *__restrict__ temp) {
    // 用于全局同步
    auto grid = cooperative_groups::this_grid();
    // 计算 grid 的步长，但感觉应该是一个 block 中有多少个线程才对 [question]
    // 目前的数值是 29184 ，即 114 ✖️ 256
    auto gridStride = blockDim.x * gridDim.x;
    //  当前线程的全局索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 这里应该是一整个 grid 把能用的线程都用掉了，所以每个线程下一次再进行交换做的就是下一个 grid 他所对应的位置了
    for (; idx < swap_size * threadsPerRow; idx += gridStride) {
        if (idx < swap_size * threadsPerRow) {
            // 按照 threadsPerRow 进行了一个分组，分组内的线程要负责 swap_len / threadsPerRow 个元素的交换
            // 该线程是哪一组 threadsPerRow
            const int row_idx = idx / threadsPerRow;
            // 该线程在分组内的索引
            const int lane = idx % threadsPerRow;
            // 循环交换完在本行中该线程要负责的元素， 结果暂存到 temp 数组中
            for (size_t j = lane; j < swap_len; j += threadsPerRow) {
                const size_t src_offset = source[row_idx] + j * stride;
                temp[row_idx * swap_len + j] = A_d[src_offset];
            }
        }
    }
    grid.sync();
    // 第一阶段完成， temp存放了每一行现在应该是哪一些数据，第二阶段就是把这些数据都存到原本数组当中，完成交换 
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < swap_size * threadsPerRow; idx += gridStride) {
        if (idx < swap_size * threadsPerRow) {
            const int row_idx = idx / threadsPerRow;
            const int lane = idx % threadsPerRow;
            for (size_t j = lane; j < swap_len; j += threadsPerRow) {
                const size_t tgt_offset = target[row_idx] + j * stride;
                A_d[tgt_offset] = temp[row_idx * swap_len + j];
            }
        }
    }
}

// 启动 exchange 核函数
template <typename T>
void launchSwapByPivotingKernel(int *source_raw_ptr, int *target_raw_ptr,
                                int swap_size, int swap_len, int stride, T *A_d,
                                T *temp_d, int blocksPerGrid, int threadsPerBlock) {
    //需要交换的列数小于 2048 认为是小，这里的需要交换的数字是无法进入的，与我们设置的 nb 值相关，设置的是小于 2048 时才会启用
    if (swap_len < SWAP_LEN_PANEL) {
        void *args[] = {reinterpret_cast<void *>(&source_raw_ptr),
                        reinterpret_cast<void *>(&target_raw_ptr),
                        reinterpret_cast<void *>(&swap_size),
                        reinterpret_cast<void *>(&swap_len),
                        reinterpret_cast<void *>(&stride),
                        reinterpret_cast<void *>(&A_d),
                        reinterpret_cast<void *>(&temp_d)};
        // 类型为指向该函数的函数指针
        using KernelType = decltype(&swapByPivotingKernel<T, 16>);
        // 目前的设定是每行交换都设置 16 行
        KernelType kernel_ptr = &swapByPivotingKernel<T, 16>;
        // 协作式内核启动，允许网格内所有的进程块进行协作和同步
        CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel_ptr),
                                               blocksPerGrid, threadsPerBlock,
                                               args));
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        // [question] 这个添加的数值非常的有待商榷
        for (int i = 0; i < swap_len;
             i += (swap_len + threadsPerBlock - 1) / threadsPerBlock) {
            // i 每次循环添加的数值是 这一次交换一共需要多少个 Block 才能够完成
            // 也就是这个 each_swap_len 
            auto each_swap_len = (swap_len + threadsPerBlock - 1) / threadsPerBlock;
            // swap_len - i 是本身需要交换多少行和已经进行了交换的长度相减，因为对于大的要交换的情况来说是循环进行的
            auto swap_len_panel = std::min(swap_len - i, each_swap_len);
            auto A_d_panel = A_d + i * stride;
            void *args[] = {reinterpret_cast<void *>(&source_raw_ptr),  
                            reinterpret_cast<void *>(&target_raw_ptr),
                            reinterpret_cast<void *>(&swap_size),
                            reinterpret_cast<void *>(&swap_len_panel),
                            reinterpret_cast<void *>(&stride),
                            reinterpret_cast<void *>(&A_d_panel),
                            reinterpret_cast<void *>(&temp_d)};
            using KernelType = decltype(&swapByPivotingKernel<T, 16>);
            KernelType kernel_ptr = &swapByPivotingKernel<T, 16>;
            CUDA_CHECK(
                cudaLaunchCooperativeKernel(reinterpret_cast<void *>(kernel_ptr),
                                            blocksPerGrid, threadsPerBlock, args));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
}
