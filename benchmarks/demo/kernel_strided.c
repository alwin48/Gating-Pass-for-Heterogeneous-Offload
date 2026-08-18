/* Demo kernel: strided vector add — typically GPU-unprofitable (transfer + uncoalesced).
 * Forced GPU path demonstrates regression vs CPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#ifndef N
#define N 4096
#endif
#ifndef STRIDE
#define STRIDE 16
#endif

static double now_ms(void) { return omp_get_wtime() * 1000.0; }

int main(void) {
  const int n = N;
  const int stride = STRIDE;
  const int len = n * stride;
  double *A = (double *)malloc(sizeof(double) * len);
  double *B = (double *)malloc(sizeof(double) * len);
  double *C = (double *)malloc(sizeof(double) * len);
  if (!A || !B || !C) return 1;
  for (int i = 0; i < len; i++) {
    A[i] = 1.0;
    B[i] = 2.0;
    C[i] = 0.0;
  }

  double t0 = now_ms();
#if defined(GPHO_GPU)
#pragma omp target teams distribute parallel for map(to : A[0 : len], B[0 : len]) map(from : C[0 : len])
  for (int i = 0; i < n; i++) {
    int idx = i * stride;
    C[idx] = A[idx] + B[idx];
  }
#else
#pragma omp parallel for
  for (int i = 0; i < n; i++) {
    int idx = i * stride;
    C[idx] = A[idx] + B[idx];
  }
#endif
  double t1 = now_ms();
  printf("GPHO_TIME_MS=%.4f\n", t1 - t0);
  fprintf(stderr, "C[0]=%.1f\n", C[0]);
  free(A);
  free(B);
  free(C);
  return 0;
}
