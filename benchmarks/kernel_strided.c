/**
 * kernel_strided.c - Non-Coalesced Strided Memory Traversal Benchmark
 *
 * Characteristics:
 * - Extremely Low Arithmetic Intensity: 1 add per 16/32 bytes accessed
 * - Un-coalesced Memory Stride: Large stride (STRIDE=64) causing memory divergence
 * - High Transfer Overhead vs Compute: PCIe latency dominates total execution
 * - GPU Offload Suitability: Unprofitable trap (Regresses GPU performance; CPU preferred)
 */

#include <stdio.h>
#include <stdlib.h>

#define NUM_ELEMENTS 1048576
#define STRIDE 64
#define EFFECTIVE_COUNT (NUM_ELEMENTS / STRIDE)

void kernel_strided(const float *__restrict__ in_a,
                    const float *__restrict__ in_b,
                    float *__restrict__ out,
                    int count) {
    #pragma omp parallel for
    for (int i = 0; i < count; ++i) {
        int idx = i * STRIDE;
        out[idx] = in_a[idx] * 1.5f + in_b[idx];
    }
}

int main(void) {
    size_t bytes = (size_t)NUM_ELEMENTS * sizeof(float);
    float *in_a = (float *)malloc(bytes);
    float *in_b = (float *)malloc(bytes);
    float *out = (float *)malloc(bytes);

    if (!in_a || !in_b || !out) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        in_a[i] = (float)(i % 100);
        in_b[i] = (float)(i % 50);
        out[i] = 0.0f;
    }

    kernel_strided(in_a, in_b, out, EFFECTIVE_COUNT);

    printf("Strided kernel completed. Sample output out[0]=%.2f, out[STRIDE]=%.2f\n",
           out[0], out[STRIDE]);

    free(in_a);
    free(in_b);
    free(out);
    return 0;
}
