#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>

#include <cusolverDn.h>

#include "TensorBLAS.h"

int64_t m, n, nb, trsm_nb;
double done = 1.0;
double dzero = 0.0;
double dnegone = -1.0;

float gemm_ms = 0.0;
float trsm_ms = 0.0;
float panel_ms = 0.0;
bool checkFlag = false;

int num_stms = 32;

int parseArguments(int argc,char *argv[])
{
    if(argc < 6)
    {
        printf("Needs m, n, nb, trsm_nb, check, as inputs\n");
        return -1;
    }
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    nb = atoi(argv[3]);
    trsm_nb = atoi(argv[4]);
    if(atoi(argv[5]) == 1)
        checkFlag = true;
    // trsm_nb = atoi(argv[3]);
    // syrk_nb = atoi(argv[4]);
    
    return 0;
}

int64_t* TrsmMatSize(int64_t n, int *length) 
{
    
    int64_t powers[] = {4096, 8192, 16384,  16384, 16384, 16384,16384, 16384, 16384, 16384, 16384};
    int64_t* result = (int64_t*) malloc((sizeof(powers) / sizeof(int64_t) + 1) * sizeof(int64_t)+1);
    int result_index = 0;

    for (int i = sizeof(powers) / sizeof(int64_t)-1 ; i >= -1; i--) 
    {
        if (n < 4096) {
            if(n == 0)
                break;
            result[result_index++] = n;
            break;
        }
        if (n >= powers[i]) {
            result[result_index++] = powers[i];
            n -= powers[i];
        }
    }

    result[result_index] = -1;
    *length = result_index-1;
    return result;
}

int64_t* LUMatSize(int64_t n, int *length) 
{
    
    int64_t powers[] = {1024, 2048, 4096, 8192, 16384, 32768, 65536};
    int64_t* result = (int64_t*) malloc((sizeof(powers) / sizeof(int64_t) + 1) * sizeof(int64_t)+1);
    int result_index = 0;

    for (int i = sizeof(powers) / sizeof(int64_t)-1 ; i >= -1; i--) 
    {
        if (n < 1024) {
            if(n == 0)
                break;
            result[result_index++] = n;
            break;
        }
        if (n >= powers[i]) {
            result[result_index++] = powers[i];
            n -= powers[i];
        }
    }

    result[result_index] = -1;
    *length = result_index-1;
    return result;
}

void rtrsm_p0(cublasHandle_t cublas_handle, long int m, long int n, 
        double *dA, long int lda, double *dB, long int ldb, int trsm_nb)
{
    if(m <= trsm_nb)
    {
        cublasDtrsm(cublas_handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            m, n, &done,
            dA, lda,
            dB, ldb
        );
        return;
    }
    rtrsm_p0(cublas_handle, m/2, n, dA, lda, dB, ldb, trsm_nb);

    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m/2, n, m/2, &dnegone,
                dA+m/2, lda, dB, ldb, &done,
                dB+m/2, ldb
    );
    rtrsm_p0(cublas_handle, m/2, n, dA+m/2+m/2*lda, lda, dB+m/2, ldb, trsm_nb);
    return;
}

void rtrsm(cublasHandle_t cublas_handle, long int m, long int n, double *dA, long int lda, double *dB, long int ldb, int trsm_nb)
{
    int length;
    int64_t* matSize = TrsmMatSize(m, &length);
    long int offset;
    long int rest_m = m;
    for(int i = length; i>=0; i--)
    {
        int64_t mm = matSize[i];
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(mm % 4096 == 0)
        {
            rtrsm_p0(cublas_handle, mm, n, dA+offset+offset*lda, lda, dB+offset, ldb, trsm_nb);
        }
        else
        {
            cublasDtrsm(cublas_handle,
                CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
                mm, n, &done,
                dA+offset+offset*lda, lda,
                dB+offset, ldb
            );
        }
        if(i != 0)
        {
            rest_m -= mm;
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                rest_m, n, mm, &dnegone,
                dA+offset+mm+offset*lda, lda, dB+offset, ldb, &done,
                dB+offset+mm, ldb
            );
        }
    }
    
}

int rn, orin;

void recLU_p0(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, int m, int n,
             double *dA, int lda, int nb, int64_t *ipiv, int64_t *hipiv, double *dwork, int dlwork,
            double *dhwork, int hlwork, int *devInfo)
{
    if(n <= nb)
    {
        //panel factorization
        auto info = cusolverDnXgetrf(cusolver_handle, NULL, m, nb, CUDA_R_64F, dA, lda, ipiv, CUDA_R_64F, 
                dwork, dlwork, dhwork, hlwork, devInfo);
            
        // printf("info = %d\n", info);
        cudaMemcpy(hipiv, ipiv, sizeof(int64_t)*nb, cudaMemcpyDeviceToHost);
        if(rn < orin)
        {
            //printf("rn = %d\n", rn);
            for(int i = 0; i < nb; i++)
            {
                cublasDswap(cublas_handle, orin-rn, dA-(orin-rn)*lda+i, lda, dA-(orin-rn)*lda+hipiv[i]-1, lda);
            }
            
        }

        if(rn - nb > 0)
        {
            for(int i = 0; i < nb; i++)
            {
                cublasDswap(cublas_handle, rn-nb, dA+nb*lda+i, lda, dA+nb*lda+hipiv[i]-1, lda);
            }
           rn-=nb;
        }
        //printf("another rn = %d\n", rn);
        
        
        return;
    }
        recLU_p0(cusolver_handle, cublas_handle, m, n/2, dA, lda, nb, ipiv, hipiv, dwork, dlwork, dhwork, hlwork, devInfo);
        rtrsm(cublas_handle, n/2, n/2, dA, lda, dA+n/2*lda, lda, trsm_nb);
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                m-n/2, n/2, n/2,
                &dnegone, dA+n/2, lda,
                dA+n/2*lda, lda,
                &done, dA+n/2+n/2*lda,
                lda);
        recLU_p0(cusolver_handle, cublas_handle, m-n/2, n/2, dA+n/2+n/2*lda, lda, nb, ipiv+n/2, hipiv+n/2, dwork, dlwork, dhwork, hlwork, devInfo);
        //printf("getrf info = %d\n", info);
    return;
}

// void recLU(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, int m, int n,
//              double *dA, int lda, int nb, int64_t *ipiv, int64_t *hipiv, double *dwork, int dlwork,
//             double *dhwork, int hlwork, int *devInfo)
// {
//     int length;
//     int64_t* matSize = LUMatSize(n, &length);
//     long int offset;
//     long int restn = n;
//     long int restm = m;
//     for(int i = length; i>=0; i--)
//     {
       
//         int64_t nn = matSize[i];
//         if(i < length)
//             offset += matSize[i + 1];
//         else
//             offset = 0;
//         printf("nn = %d, offset = %d, restm = %d, restn = %d\n", nn, offset, restm, restn);
//         if(nn % 1024 == 0)
//         {
//             orin = nn;
//             rn = nn;
//             recLU_p0(cusolver_handle, cublas_handle, restm, nn,
//                      dA+offset+offset*lda, lda, nb, ipiv+offset, hipiv+offset, dwork, dlwork,
//                      dhwork, hlwork, devInfo);
//         }
//         else
//         {
//             cusolverDnXgetrf(cusolver_handle, NULL, restm, nn, CUDA_R_64F, dA+offset+offset*lda, lda, ipiv+offset, CUDA_R_64F, 
//                              dwork, dlwork, dhwork, hlwork, devInfo);
//             cudaMemcpy(hipiv+offset, ipiv+offset, sizeof(int64_t)*nn, cudaMemcpyDeviceToHost);
//         }
//         if(i < length)
//         {
//             //printf("rn = %d\n", rn);
//             //printf("matSize[i+1] = %d\n", matSize[i+1]);
//             for(int j = 0; j < nn; j++)
//             {
//                 cublasDswap(cublas_handle, n-restn, dA+offset+j, lda, dA+offset+hipiv[offset+j]-1, lda);
//             }
            
//         }
//         if(i > 0)
//         {
//             for(int j = 0; j < nn; j++)
//             {
//                 cublasDswap(cublas_handle, restn-nn, dA+offset+offset*lda+nn*lda+j, lda, dA+offset+offset*lda+nn*lda+hipiv[offset+j]-1, lda);
//             }
//             rtrsm(cublas_handle, nn, restn-nn, dA+offset+offset*lda, lda, dA+offset+offset*lda+nn*lda, lda, trsm_nb);
//             cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
//                 restm-nn, restn-nn, nn,
//                 &dnegone, dA+offset+offset*lda+nn, lda,
//                 dA+offset+offset*lda+nn*lda, lda,
//                 &done, dA+offset+offset*lda+nn+nn*lda,
//                 lda);
//         }
//         restn -= nn;
//         restm -= nn;
//     }

// }

int luorin, lurn;
float pt = 0.0;
float ut = 0.0;
void recLU(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, int m, int n,
             double *dA, int lda, int nb, int64_t *ipiv, int64_t *hipiv, double *dwork, int dlwork,
            double *dhwork, int hlwork, int index, int length, int64_t *matSize, int *devInfo)
{
    if(n <= matSize[index])
    {
        //printf("n = %d\n", n);
        startTimer();
        recLU_p0(cusolver_handle, cublas_handle, m, n, dA, lda, nb, ipiv, hipiv, dwork, dlwork, dhwork, hlwork, devInfo);
        pt+=stopTimer();
        int64_t nipiv[n];
        int mul= 0;
        for(int i = 0; i < n; i+=nb)
        {   
            
            for(int j = 0; j<nb; j++)
                nipiv[i+j]=hipiv[i+j]+(nb*mul);
            mul++;
        }
        startTimer();
        if(lurn < luorin)
        {
            //printf("rn = %d\n", rn);
            for(int i = 0; i < n; i++)
            {
                //printf("%d ", nipiv[i]);
                cublasDswap(cublas_handle, luorin-lurn, dA-(luorin-lurn)*lda+i, lda, dA-(luorin-lurn)*lda+nipiv[i]-1, lda);
            }
            //printf("\n");
        }
        if(lurn - n > 0)
        {
            for(int i = 0; i < n; i++)
            {
                cublasDswap(cublas_handle, lurn-n, dA+n*lda+i, lda, dA+n*lda+nipiv[i]-1, lda);
            }
            lurn -= n;
        }
        pt+=stopTimer();
        return;
    }
    int nn = matSize[index];
    orin = nn;
    rn = nn;
    recLU(cusolver_handle, cublas_handle, m, nn, dA, lda, nb, ipiv, hipiv, dwork, dlwork, dhwork, hlwork, index, length, matSize, devInfo);
    if(index == 0)
        return;
    startTimer();
    rtrsm(cublas_handle, nn, n-nn, dA, lda, dA+nn*lda, lda, trsm_nb);
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                m-nn, n-nn, nn,
                &dnegone, dA+nn, lda,
                dA+nn*lda, lda,
                &done, dA+nn+nn*lda,
                lda);
    ut+=stopTimer();
    orin = n-nn;
    rn = n-nn;
    recLU(cusolver_handle, cublas_handle, m-nn, n-nn, dA+nn+nn*lda, lda, nb, ipiv+nn, hipiv+nn, dwork, dlwork, dhwork, hlwork, index-1, length, matSize, devInfo);
    if(index == 0)
        return;
}


void recLU_np(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, int m, int n, double *dA, int lda, int nb, int64_t *ipiv, double *dwork, int dlwork,
            double *dhwork, int hlwork, int *devInfo)
{
    //printf("n = %d\n",n);
    if(n <= nb)
    {
        //panel factorization
        auto info = cusolverDnXgetrf(cusolver_handle, NULL, m, nb, CUDA_R_64F, dA, lda, NULL, CUDA_R_64F, 
                dwork, dlwork, dhwork, hlwork, devInfo);
        //printf("getrf info = %d\n", info);
        return;
    }
    recLU_np(cusolver_handle, cublas_handle, m, n/2, dA, lda, nb, ipiv, dwork, dlwork, dhwork, hlwork, devInfo);
    //Solve U12
    rtrsm(cublas_handle, n/2, n/2, dA, lda, dA+n/2*lda, lda, trsm_nb);
    // auto info = cublasDtrsm(cublas_handle,
    //             CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
    //             CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
    //             n/2, n/2, &done,
    //             dA, lda,
    //             dA+n/2*lda, lda
    //             );
    //printf("trsm info = %d\n", info);
    //Trailing matrix update
    cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                m-n/2, n/2, n/2,
                &dnegone, dA+n/2, lda,
                dA+n/2*lda, lda,
                &done, dA+n/2+n/2*lda,
                 lda);
    //printf("gemm info = %d\n", info);
    recLU_np(cusolver_handle, cublas_handle, m-n/2, n/2, dA+n/2+n/2*lda, lda, nb, ipiv, dwork, dlwork, dhwork, hlwork, devInfo);
    return;

}

void rlunp(cusolverDnHandle_t cusolver_handle, cublasHandle_t cublas_handle, int m, int n, double *dA, int lda, int nb, int64_t *ipiv, double *dwork, int dlwork,
            double *dhwork, int hlwork, int *devInfo)
{
    int length;
    int64_t* matSize = TrsmMatSize(m, &length);
    long int offset;
    long int rest_m = m;
    for(int i = length; i>=0; i--)
    {
        int64_t nn = matSize[i];
        if(i < length)
            offset += matSize[i + 1];
        else
            offset = 0;
        if(nn % 4096 == 0)
        {
            recLU_np(cusolver_handle, cublas_handle, nn, nn, dA+offset+offset*lda, lda, nb, ipiv, dwork, dlwork, dhwork, hlwork, devInfo);
        }
        else
        {
           cusolverDnXgetrf(cusolver_handle, NULL, rest_m, nn, CUDA_R_64F, dA, lda, NULL, CUDA_R_64F, 
                dwork, dlwork, dhwork, hlwork, devInfo);
        }
        if(i != 0)
        {
           
        }
    }
}

__global__
void clearTriDouble_(char uplo, long int m, long int n, double *a, long int lda)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
   
	if (i<m && j<n) {
		if (uplo == 'l') {
			if (i>j) {
				a[i+j*lda] = 0.0;
			}
        } 
        else
        {
            if (i<j)
                a[i+j*lda] = 0.0;
            if (i>=j)
                a[i+j*lda] = 1.0;
            
            
		}
	}
}

__global__
void getLU(long int m, long int n, double *A, long int lda, double *L, double *U)
{
	long int i = threadIdx.x + blockDim.x * blockIdx.x;
	long int j = threadIdx.y + blockDim.y * blockIdx.y;
   
	if (i<m && j<n) {
		if (i>j) 
        {
			L[i+j*m] = A[i+j*lda];
		}
        else
        {
            U[i+j*n] = A[i+j*lda];
            if(i == j)
            {
                L[i+j*m] = 1.0;
            }
        }
	}
}

void test_trsm(cublasHandle_t cublas_handle, int m, int n)
{
    double *dA;
    cudaMalloc(&dA,sizeof(double)*m*m);
    double *dB;
    cudaMalloc(&dB,sizeof(double)*m*n);
    generateNormalMatrixDouble(dA, m, m);
    generateNormalMatrixDouble(dB, m, n);
    dim3 grid((m+31)/32, (m+31)/32);
    dim3 block(32,32);
    clearTriDouble_<<<grid,block>>>('u', m, m, dA, m);
    // printMatrixDeviceBlockDouble("AAAA.csv", m, m, dA, m);
    // printMatrixDeviceBlockDouble("BBBB.csv", m, n, dB, m);
    double *work;
    cudaMalloc(&work, sizeof(double)*m*n);
    cudaMemcpy(work, dB, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
    //printf("start\n");
    startTimer();
    rtrsm(cublas_handle, m, n, dA, m, dB, m, trsm_nb);
    float ms = stopTimer();
    //printf("recursive trsm takes %fms, TFLOPs is %f\n", ms, 1.0*m*m*n/ms/1e9);
    startTimer();
    cublasDtrsm(cublas_handle,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            m, n, &done,
            dA, m,
            work, m
        );
    ms = stopTimer();
    //  printMatrixDeviceBlockDouble("X.csv", m, n, work, m);
    //printf("Dtrsm takes %fms, TFLOPs is %f\n", ms, 1.0*m*m*n/ms/1e9);
    cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
                &done, dB, m, &dnegone, work, m,
                work, m);
    // printMatrixDeviceBlockDouble("work.csv", m, n, work, m);
    //printf("Forward error ||X_tc-X_cublas||/||X_cublas|| is %.6e\n", snormDouble(m,n,work,m)/snormDouble(m,n,dB,m));
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(work);
}

int main(int argc,char *argv[])
{
    if(parseArguments(argc, argv)==-1)
        return 0;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cusolverDnHandle_t cusolver_handle ;
    cusolverDnCreate(&cusolver_handle);
    // test_trsm(cublas_handle, m, n);
    // return;
    double *dA;
    cudaMalloc(&dA,sizeof(double)*m*n);
    int *devInfo;
    cudaMalloc(&devInfo, sizeof(int));
    generateUniformMatrixDouble(dA, m, n);
    
    double *dwork, *hwork;
    size_t dlwork, hlwork;
    cusolverDnXgetrf_bufferSize(
        cusolver_handle,
        NULL,
        m,
        n,
        CUDA_R_64F,
        dA,
        m,
        CUDA_R_64F,
        &dlwork,
        &hlwork); 
    
    //printf("lwork = %d, hlwork = %d\n", dlwork, hlwork);
    cudaMalloc(&dwork, dlwork);
    
    cudaMallocHost(&hwork, hlwork);

    

    int64_t *ipiv;
    cudaMalloc(&ipiv, sizeof(int64_t)*n);
    double *oriA, *L, *U;
    int64_t *hipiv;
    cudaError_t status = cudaMallocHost(&hipiv, sizeof(int64_t)*n);
    if (status != cudaSuccess)
        printf("Error allocating pinned host memory\n");
    if(checkFlag)
    {
        cudaMalloc(&oriA, sizeof(double)*m*n);
        cudaMemcpy(oriA, dA, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
        // printMatrixDeviceBlockDouble("testA.csv", m, n, dA, m);
    }

    startTimer();
    cusolverDnXgetrf(
        cusolver_handle,
        NULL,
        m,
        n,
        CUDA_R_64F,
        dA,
        m,
        ipiv,
        CUDA_R_64F,
        dwork,
        dlwork,
        hwork,
        hlwork,
        devInfo);
    float ms = stopTimer();
    //printf("double LU takes %f ms, gflops is %f\n", ms, 2.0/3.0*m*n*n/ms/1e6);
    if(checkFlag)
    {
        cudaMalloc(&L, sizeof(double)*m*n);
        cudaMalloc(&U, sizeof(double)*n*n);
        dim3 grid((m+31)/32, (n+31)/32);
        dim3 block(32,32);
        getLU<<<grid,block>>>(m, n, dA, m, L, U);
        // printMatrixDeviceBlockDouble("testL.csv", m, n, L, m);
        // printMatrixDeviceBlockDouble("testU.csv", n, n, U, n);
        double normA = snormDouble(m,n,oriA,m);
        // cublasHandle_t handle[num_stms];
        // cudaStream_t stream[num_stms];
        // for(int i = 0; i < num_stms; i++)
        // {
        //     cublasCreate(&handle[i]);
        //     cudaStreamCreate(&stream[i]);
        //     cublasSetStream(handle[i], stream[i]);
        // }
        startTimer();
        auto info = cudaMemcpy(hipiv, ipiv, sizeof(int64_t)*n, cudaMemcpyDeviceToHost);
        //printf("info = %d\n", info);
        //printMatrixDeviceBlockDouble("A.csv", m, n, oriA, m);
        float ms = stopTimer();
        printf("%d transfers takes %f ms\n", n, ms);
        startTimer();
        for(int i = 0; i < n; i++)
        {   
            //printf("%lld ", hipiv[i]);
            int info = cublasDswap(cublas_handle, n, oriA+i, m, oriA+hipiv[i]-1, m);
        }
        printf("\n");
        ms = stopTimer();
        printf("%d swaps takes %f ms\n", n, ms);
        
        //printMatrixDeviceBlockDouble("pA.csv", m, n, oriA, m);
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, n, &dnegone,
                L, m, U, n, &done,
                oriA, m
            );
        //printMatrixDeviceBlockDouble("L.csv", m, n, L, m);
        //printMatrixDeviceBlockDouble("U.csv", n, n, U, n);
        printf("backward error ||PA-LU||/||A|| is %.6e\n", snormDouble(m,n,oriA,m)/normA);
        cudaFree(L);
        cudaFree(U);
        cudaFree(oriA);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    generateUniformMatrixDouble(dA, m, n);
    if(checkFlag)
    {
        cudaMalloc(&oriA, sizeof(double)*m*n);
        cudaMemcpy(oriA, dA, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
        // printMatrixDeviceBlockDouble("A.csv", m, n, dA, m);
    }
    rn = n;
    orin = n;
    int length;
    int64_t* matSize = LUMatSize(n, &length);
    lurn = n;
    luorin = n;
    // startTimer();
    recLU(cusolver_handle, cublas_handle, m, n, dA, m, nb, ipiv, hipiv, dwork, dlwork, hwork, hlwork, length, length, matSize, devInfo);
    // ms = stopTimer();
    auto err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    
    //printf("double recursive LU takes %f ms, gflops is %f\n", pt+ut, 2.0/3.0*m*n*n/(pt+ut)/1e6);
    printf("%f\t%f\n", 2.0/3.0*m*n*n/(ms)/1e6, 2.0/3.0*m*n*n/(pt+ut)/1e6);
    if(checkFlag)
    {
        cudaMalloc(&L, sizeof(double)*m*n);
        cudaMalloc(&U, sizeof(double)*n*n);
        dim3 grid((m+31)/32, (n+31)/32);
        dim3 block(32,32);
        getLU<<<grid,block>>>(m, n, dA, m, L, U);
        // printMatrixDeviceBlockDouble("L.csv", m, n, L, m);
        // printMatrixDeviceBlockDouble("U.csv", n, n, U, n);
        double normA = snormDouble(m,n,oriA,m);
        // // cublasHandle_t handle[num_stms];
        // // cudaStream_t stream[num_stms];
        // // for(int i = 0; i < num_stms; i++)
        // // {
        // //     cublasCreate(&handle[i]);
        // //     cudaStreamCreate(&stream[i]);
        // //     cublasSetStream(handle[i], stream[i]);
        // // }
        // startTimer();
        // auto info = cudaMemcpy(hipiv, ipiv, sizeof(int64_t)*n, cudaMemcpyDeviceToHost);
        // //printf("info = %d\n", info);
        // //printMatrixDeviceBlockDouble("A.csv", m, n, oriA, m);
        // float ms = stopTimer();
        // printf("%d transfers takes %f ms\n", n, ms);
        // startTimer();
        int mul= 0;
        for(int i = 0; i < n; i+=nb)
        {   
            
            for(int j = 0; j<nb; j++)
                hipiv[i+j]+=(nb*mul);
            mul++;
            // printf("%lld ", hipiv[i]);
            // int info = cublasDswap(cublas_handle, n, oriA+i, m, oriA+hipiv[i]-1, m);
        }
        for(int i = 0; i < n; i++)
        {   
            //printf("%lld ", hipiv[i]);
            int info = cublasDswap(cublas_handle, n, oriA+i, m, oriA+hipiv[i]-1, m);
        }

        // ms = stopTimer();
        // printf("%d swaps takes %f ms\n", n, ms);
        printf("\n");
        // //printMatrixDeviceBlockDouble("pA.csv", m, n, oriA, m);
        cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, n, &dnegone,
                L, m, U, n, &done,
                oriA, m
            );
        // //printMatrixDeviceBlockDouble("L.csv", m, n, L, m);
        // //printMatrixDeviceBlockDouble("U.csv", n, n, U, n);
        printf("backward error ||PA-LU||/||A|| is %.6e\n", snormDouble(m,n,oriA,m)/normA);
        cudaFree(L);
        cudaFree(U);
        cudaFree(oriA);
        auto err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
    
    return;
    {
        //return;
        // startTimer();
        // for(int i = 0; i < nb; i++)
        // {
        //     cublasDswap(cublas_handle, n-nb, dA, m, dA+1, m);
        // }
        // ms = stopTimer();
        // printf("%d swap takes %fms\n", nb, ms);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
        generateUniformMatrixDouble(dA, m, n);
        
        if(checkFlag)
        {
            cudaMalloc(&oriA, sizeof(double)*m*n);
            cudaMemcpy(oriA, dA, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
        }

        //printMatrixDeviceBlockDouble("A.csv", m, n, dA, m);
        startTimer();
        cusolverDnXgetrf(
            cusolver_handle,
            NULL,
            m,
            n,
            CUDA_R_64F,
            dA,
            m,
            NULL,
            CUDA_R_64F,
            dwork,
            dlwork,
            hwork,
            hlwork,
            devInfo);
        ms = stopTimer();
        printf("double LU non-pivoting takes %f ms, gflops is %f\n", ms, 2.0/3.0*m*n*n/ms/1e6);
        if(checkFlag)
        {
            cudaMalloc(&L, sizeof(double)*m*n);
            cudaMalloc(&U, sizeof(double)*n*n);
            dim3 grid((m+31)/32, (n+31)/32);
            dim3 block(32,32);
            getLU<<<grid,block>>>(m, n, dA, m, L, U);
            double normA = snormDouble(m,n,oriA,m);
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, n, &dnegone,
                    L, m, U, n, &done,
                    oriA, m
                );
            printf("backward error ||A-LU||/||A|| is %.6e\n", snormDouble(m,n,oriA,m)/normA);
            cudaFree(L);
            cudaFree(U);
            cudaFree(oriA);
        }
        return;
        // double *LU;
        // cudaMalloc(&LU, sizeof(double)*m*n);
        // cudaMemcpy(LU, dA, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
        
        //printMatrixDeviceBlockDouble("LU.csv", m, n, dA, m);
        generateUniformMatrixDouble(dA, m, n);

        if(checkFlag)
        {
            cudaMalloc(&oriA, sizeof(double)*m*n);
            cudaMemcpy(oriA, dA, sizeof(double)*m*n, cudaMemcpyDeviceToDevice);
        }
        //printMatrixDeviceBlockDouble("a.csv", m, n, dA, m);
        startTimer();
        recLU_np(cusolver_handle, cublas_handle, m, n, dA, m, nb, ipiv, dwork, dlwork, hwork, hlwork, devInfo);
        ms = stopTimer();
        //printMatrixDeviceBlockDouble("lu.csv", m, n, dA, m);
        printf("double recursive non-pivoting LU takes %f ms, gflops is %f\n", ms, 2.0/3.0*m*n*n/ms/1e6);
        if(checkFlag)
        {
            cudaMalloc(&L, sizeof(double)*m*n);
            cudaMalloc(&U, sizeof(double)*n*n);
            dim3 grid((m+31)/32, (n+31)/32);
            dim3 block(32,32);
            getLU<<<grid,block>>>(m, n, dA, m, L, U);
            double normA = snormDouble(m,n,oriA,m);
            cublasDgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, n, &dnegone,
                    L, m, U, n, &done,
                    oriA, m
                );
            printf("backward error ||A-LU||/||A|| is %.6e\n", snormDouble(m,n,oriA,m)/normA);
            cudaFree(L);
            cudaFree(U);
            cudaFree(oriA);
        }
        // cublasDgeam(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n,
        //             &done, LU, m, &dnegone, dA, m,
        //             dA, m);
        //printMatrixDeviceBlockDouble("lu.csv", m, n, dA, m);
        //printf("Forward error ||lu||/||LU_cusolver|| is %.6e\n", snormDouble(m,n,dA,m)/snormDouble(m,n,LU,m));
        err = cudaGetLastError();
        if (err != cudaSuccess) 
        {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    }
}