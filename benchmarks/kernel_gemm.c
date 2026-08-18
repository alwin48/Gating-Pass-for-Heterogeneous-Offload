/**
 * kernel_gemm.c - Dense General Matrix Multiply (GEMM) Benchmark
 *
 * Characteristics:
 * - High Arithmetic Intensity: O(N^3) FLOPs vs O(N^2) memory footprint
 * - Coalesced Memory Access: Stride-1 accesses in innermost reduction
 * - Loop Nest Depth: 3
 * - GPU Offload Suitability: Highly profitable (Speedup >= 1.20x)
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1024

void kernel_gemm(const double *__restrict__ A,
                 const double *__restrict__ B,
                 double *__restrict__ C) {
    #pragma omp target teams distribute parallel for if(target: N >= 256)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            #pragma omp simd reduction(+:sum)
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = (double *)malloc(bytes);
    double *B = (double *)malloc(bytes);
    double *C = (double *)malloc(bytes);

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N * N; ++i) {
        A[i] = 1.0;
        B[i] = 2.0;
        C[i] = 0.0;
    }

    kernel_gemm(A, B, C);

    printf("GEMM completed successfully. Sample output C[0]=%.2f, C[N*N-1]=%.2f\n",
           C[0], C[N * N - 1]);

    free(A);
    free(B);
    free(C);
    return 0;
}
