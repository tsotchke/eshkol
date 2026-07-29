/* bench_dot_exact.c — Phase E2.7: honest dot_exact vs f64 dot benchmark.
 *
 * Reports, on THIS machine:
 *   1. Correctness — the exact i128 integer dot vs a dequantized f64 dot, and how
 *      much the f64 result drifts (and DIFFERS across summation orders).
 *   2. Determinism — i128 is byte-identical across shuffles; f64 is not.
 *   3. Throughput — elements/s and GB/s for each path.
 *
 * Usage: bench_dot_exact [N] [reps]
 */
#if !defined(_WIN32) && !defined(_POSIX_C_SOURCE)
#define _POSIX_C_SOURCE 200809L
#endif

#include "eshkol_fixed_point.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* dequantized f64 dot: Σ (a[i]*sa)*(b[i]*sb), naive left-to-right accumulation */
static double f64_dot(const int16_t *a, const int16_t *b, size_t n, double sa, double sb) {
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) acc += ((double)a[i] * sa) * ((double)b[i] * sb);
    return acc;
}

static void shuffle(size_t *idx, size_t n, esk_rng *r) {
    for (size_t i = n; i > 1; i--) { size_t j = (size_t)(esk_rng_next_u64(r) % i);
        size_t t = idx[i-1]; idx[i-1] = idx[j]; idx[j] = t; }
}

int main(int argc, char **argv) {
    size_t N   = (argc > 1) ? (size_t)strtoull(argv[1], NULL, 10) : (size_t)1 << 20; /* 1,048,576 */
    int    reps= (argc > 2) ? atoi(argv[2]) : 50;
    double sa = 0.00787401575, sb = 0.0123456789;   /* representative dequant scales */
    uint8_t F = 40;

    esk_rng rng; esk_rng_seed(&rng, 12345u);
    int16_t *a = malloc(N * sizeof *a), *b = malloc(N * sizeof *b);
    size_t  *idx = malloc(N * sizeof *idx);
    int16_t *sa2 = malloc(N * sizeof *sa2), *sb2 = malloc(N * sizeof *sb2);
    for (size_t i = 0; i < N; i++) {
        a[i] = (int16_t)(esk_rng_next_u64(&rng) & 0xFFFF);
        b[i] = (int16_t)(esk_rng_next_u64(&rng) & 0xFFFF);
        idx[i] = i;
    }

    printf("=== dot_exact vs f64 dot ===\n");
    printf("platform: %s   N=%zu   reps=%d   F=%u\n",
#if defined(__aarch64__) || defined(__arm64__)
           "arm64",
#elif defined(__x86_64__)
           "x86-64",
#else
           "unknown",
#endif
           N, reps, F);

    /* ---- correctness + determinism ---- */
    esk_i128 gold = esk_idot_i16(a, b, N);
    bool ex; esk_fixed fx = esk_dot_exact_i16(a, b, N, sa, sb, F, &ex);
    double dgold = esk_fixed_to_double(fx, NULL);
    double f0 = f64_dot(a, b, N, sa, sb);

    /* shuffle and re-evaluate both */
    shuffle(idx, N, &rng);
    for (size_t i = 0; i < N; i++) { sa2[i] = a[idx[i]]; sb2[i] = b[idx[i]]; }
    esk_i128 gshuf = esk_idot_i16(sa2, sb2, N);
    double f1 = f64_dot(sa2, sb2, N, sa, sb);

    char gb[64]; esk_i128_to_string(gold, gb, sizeof gb);
    printf("\n[correctness]\n");
    printf("  exact integer dot (i128)     : %s\n", gb);
    printf("  exact scaled  (fixed<128,%u>) : %.15g\n", F, dgold);
    printf("  f64 dot (order A)            : %.15g\n", f0);
    printf("  f64 dot (order B, shuffled)  : %.15g\n", f1);
    printf("  f64 order A vs B abs diff    : %.3e   %s\n",
           f64_dot(a,b,N,sa,sb) == f1 ? 0.0 : (f0 - f1 < 0 ? f1 - f0 : f0 - f1),
           (f0 == f1) ? "(identical)" : "<-- f64 is ORDER-DEPENDENT");
    printf("[determinism]\n");
    printf("  i128 dot order A == order B  : %s\n", (gold == gshuf) ? "YES (byte-identical)" : "NO");
    printf("  f64  dot order A == order B  : %s\n", (f0 == f1) ? "YES" : "NO (differs)");

    /* ---- timing ---- */
    volatile esk_i128 sink_i = 0; volatile double sink_d = 0.0;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) sink_i ^= esk_idot_i16(a, b, N);
    double t_i = now_s() - t0;

    t0 = now_s();
    for (int r = 0; r < reps; r++) sink_d += f64_dot(a, b, N, sa, sb);
    double t_f = now_s() - t0;

    double elems = (double)N * reps;
    double bytes = elems * 2.0 * sizeof(int16_t);   /* a[i] and b[i] */
    printf("\n[throughput]\n");
    printf("  i128 exact dot : %6.2f ms/rep   %7.2f Melem/s   %6.2f GB/s\n",
           t_i / reps * 1e3, elems / t_i / 1e6, bytes / t_i / 1e9);
    printf("  f64  naive dot : %6.2f ms/rep   %7.2f Melem/s   %6.2f GB/s\n",
           t_f / reps * 1e3, elems / t_f / 1e6, bytes / t_f / 1e9);
    printf("  ratio (exact/f64 time) : %.2fx\n", t_i / t_f);
    printf("\nNote: the exact i128 path trades throughput for BIT-EXACT,\n"
           "order-independent results — the reference/certification property\n"
           "the f64 path cannot provide (see [determinism] above).\n");

    (void)sink_i; (void)sink_d;
    free(a); free(b); free(idx); free(sa2); free(sb2);
    return 0;
}
