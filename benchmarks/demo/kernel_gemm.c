/* Demo kernel: large tiled GEMM-like nest (GPU-profitable narrative).
 * Prints GPHO_TIME_MS=<median-ready single run ms>
 *
 * Build (CPU): g++ -O3 -fopenmp kernel_gemm.c -o gemm_cpu
 * Build (GPU OpenMP if available):
 *   clang++ -O3 -fopenmp -fopenmp-targets=nvptx64 -DGPHO_GPU kernel_gemm.c -o gemm_gpu
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#ifndef N
#define N 512
#endif

static double now_ms(void) { return omp_get_wtime() * 1000.0; }

int main(void) {
  const int n = N;
  double *A = (double *)malloc(sizeof(double) * n * n);
  double *B = (double *)malloc(sizeof(double) * n * n);
  double *C = (double *)malloc(sizeof(double) * n * n);
  if (!A || !B || !C) return 1;
  for (int i = 0; i < n * n; i++) {
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }

  double t0 = now_ms();
#if defined(GPHO_GPU)
#pragma omp target teams distribute parallel for collapse(2) map(to : A[0 : n * n], B[0 : n * n]) map(tofrom : C[0 : n * n])
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += A[i * n + k] * B[k * n + j];
      C[i * n + j] = sum;
    }
  }
#else
#pragma omp parallel for collapse(2) if(n > 64)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      double sum = 0.0;
      for (int k = 0; k < n; k++)
        sum += A[i * n + k] * B[k * n + j];
      C[i * n + j] = sum;
    }
  }
#endif
  double t1 = now_ms();
  printf("GPHO_TIME_MS=%.4f\n", t1 - t0);
  /* Touch result to defeat DCE */
  fprintf(stderr, "C[0]=%.1f\n", C[0]);
  free(A);
  free(B);
  free(C);
  return 0;
}
