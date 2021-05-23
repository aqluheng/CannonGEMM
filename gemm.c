#include <stdio.h>
#include <stdlib.h>
#include "cblas.h"
#include <mpi.h>

void gemmBench(float *A, float *B, float *C, int m, int n, int k)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, n, B, k, 0, C, k);
/*
    for (int x = 0; x < m; x++)
    {
        for (int y = 0; y < k; y++)
        {
            C[x * k + y] = 0;
        }
    }
    for (int x = 0; x < m; x++)
    {
        for (int y = 0; y < k; y++)
        {
            for (int z = 0; z < n; z++)
            {
                C[x * k + y] += A[x * n + z] * B[z * k + y];
            }
        }
    }
*/
}

// 只尝试16核
int worldSize;
int worldRank;
#define N 1024
#define partialN (N/4)

float tempA[N * N], tempB[N * N], *tempC;

// 只做1024*1024的方阵,使用cannon算法
void gemmOur(float *A, float *B, float *C, int n) 
{
    int id, coord[2];
    int dim[2], period[2];
    MPI_Comm comm;
    dim[0] = dim[1] = 4;
    period[0] = period[1] = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, 1, &comm);
    MPI_Comm_rank(comm, &id);
    MPI_Cart_coords(comm, id, 2, coord);

    float *partialA = (float *)malloc(sizeof(float) * partialN * partialN);
    float *partialB = (float *)malloc(sizeof(float) * partialN * partialN);
    float *partialC = (float *)malloc(sizeof(float) * partialN * partialN);
    if (worldRank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int x_idx = i / partialN;
                int y_idx = j / partialN;
                int begin = (y_idx * 4 + x_idx) * partialN * partialN;
                int x_add = i % partialN;
                int y_add = j % partialN;
                tempA[begin + y_add * partialN + x_add] = A[j * N + i];
                tempB[begin + y_add * partialN + x_add] = B[j * N + i];
            }
        }
    }
    MPI_Scatter(tempA, partialN * partialN, MPI_FLOAT, partialA, partialN * partialN, MPI_FLOAT, 0, comm);
    MPI_Scatter(tempB, partialN * partialN, MPI_FLOAT, partialB, partialN * partialN, MPI_FLOAT, 0, comm);
    for (int i = 0; i < partialN * partialN; i++)
        partialC[i] = 0;
    int source, dest;
    MPI_Cart_shift(comm, 1, -coord[0], &source, &dest);
    MPI_Sendrecv_replace(partialA, partialN * partialN, MPI_FLOAT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE);
    MPI_Cart_shift(comm, 0, -coord[1], &source, &dest);
    MPI_Sendrecv_replace(partialB, partialN * partialN, MPI_FLOAT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE);
    for (int i = 0; i < 4; i++)
    {
        // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, A, n, B, n, 1, C, n);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, partialN, partialN, partialN, 1, partialA, partialN, partialB, partialN, 1, partialC, partialN);
        MPI_Cart_shift(comm, 1, -1, &source, &dest);
        MPI_Sendrecv_replace(partialA, partialN * partialN, MPI_FLOAT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(comm, 0, -1, &source, &dest);
        MPI_Sendrecv_replace(partialB, partialN * partialN, MPI_FLOAT, dest, 0, source, 0, comm, MPI_STATUS_IGNORE);
    }
    if (worldRank == 0)
        tempC = (float *)malloc(sizeof(float) * N * N);
    MPI_Gather(partialC, partialN * partialN, MPI_FLOAT, tempC, partialN * partialN, MPI_FLOAT, 0, comm);
    if (worldRank == 0)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int x_idx = i / partialN;
                int y_idx = j / partialN;
                int begin = (y_idx * 4 + x_idx) * partialN * partialN;
                int x_add = i % partialN;
                int y_add = j % partialN;
                C[j * N + i] = tempC[begin + y_add * partialN + x_add];
            }
        }
    }
}

void pooling(int *A, int *kernel44, int *C, int m, int n)
{
    int *tempMatrix = (int *)malloc(sizeof(int) * (m - 3) * (n - 3) * 16);
    for (int i = 0; i < m - 3; i++)
    {
        for (int j = 0; j < n - 3; j++)
        {
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    tempMatrix[(i * (n - 3) + j) * 16 + k * 4 + l] = A[(i + k) * n + j + l];
                }
            }
        }
    }
}

void convolution(int **A, int **kernel44, int **C, int m, int n)
{
}

void printMat(float *A, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            printf("%.1f ", A[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    if (worldSize != 16)
    {
        if (worldRank == 0)
            printf("Please use 16 process.\nmpirun -np 16 ./gemm\n");
        MPI_Finalize();
        return 0;
    }
    float *C_bench;
    float *A = (float *)malloc(sizeof(float) * N * N);
    float *B = (float *)malloc(sizeof(float) * N * N);
    float *C_our = (float *)malloc(sizeof(float) * N * N);
    double startTime, endTime, originCostTime, ourCostTime;

    if (worldRank == 0)
    { // 0号线程先测试在单核情况下的程序性能
        C_bench = (float *)malloc(sizeof(float) * N * N);
        for (int i = 0; i < N * N; i++)
            A[i] = rand() & 0xF;
        for (int i = 0; i < N * N; i++)
            B[i] = rand() & 0xF;
        startTime = MPI_Wtime();
        gemmBench(A, B, C_bench, N, N, N);
        endTime = MPI_Wtime();
        originCostTime = endTime - startTime;
    }
    startTime = MPI_Wtime();
    gemmOur(A, B, C_our, N);
    endTime = MPI_Wtime();
    ourCostTime = endTime - startTime;

    if (worldRank == 0)
    {
        for (int i = 0; i < N * N; i++)
        {
            if (C_bench[i] != C_our[i])
            {
                printf("Dismatch happened.\n");
                break;
            }
        }
        // printMat(A, N, N);
        // printMat(B, N, N);
        // printMat(C_our, N, N);
        // printMat(C_bench, N, N);
        printf("origin Cost %.5f, Our Cost %.5f\n", originCostTime, ourCostTime);
    }
    MPI_Finalize();
}
