/* test_dot_exact.c — Phase E2: exact accumulation.
 *   - dot_exact / idot correctness (vs a checked reference)
 *   - accum128 running-sum + partial merge
 *   - THE CONTRACT TEST: shuffled accumulation order => byte-identical result.
 *     Integer addition is associative and commutative, so the i128 accumulator
 *     is order-invariant to the bit. This is the whole point of the exact path.
 */
#include "eshkol_fixed_point.h"
#include "test_harness.h"
#include <stdlib.h>
#include <string.h>

/* Fisher-Yates shuffle of an index permutation using the library RNG. */
static void shuffle(size_t *idx, size_t n, esk_rng *r) {
    for (size_t i = n; i > 1; i--) {
        size_t j = (size_t)(esk_rng_next_u64(r) % i);
        size_t t = idx[i-1]; idx[i-1] = idx[j]; idx[j] = t;
    }
}

/* Reference integer dot with a wide accumulator, for correctness cross-check. */
static esk_i128 ref_idot16(const int16_t *a, const int16_t *b, size_t n) {
    esk_i128 acc = 0;
    for (size_t i = 0; i < n; i++) acc += (esk_i128)a[i] * (esk_i128)b[i];
    return acc;
}

static esk_i128 ref_idot32(const int32_t *a, const int32_t *b, size_t n) {
    esk_i128 acc = 0;
    for (size_t i = 0; i < n; i++) acc += (esk_i128)a[i] * (esk_i128)b[i];
    return acc;
}

int main(void) {
    esk_rng rng; esk_rng_seed(&rng, 20260717u);

    /* ===================== idot correctness ===================== */
    {
        int8_t a[5] = {1, 2, 3, 4, 5};
        int8_t b[5] = {5, 4, 3, 2, 1};   /* 5+8+9+8+5 = 35 */
        CHECK_EQ_I128(esk_idot_i8(a, b, 5), esk_i128_from_i64(35), "idot_i8 basic == 35");
        int16_t x[3] = {1000, -2000, 3000};
        int16_t y[3] = {3000, 2000, -1000};  /* 3e6 -4e6 -3e6 = -4e6 */
        CHECK_EQ_I128(esk_idot_i16(x, y, 3), esk_i128_from_i64(-4000000), "idot_i16 basic == -4e6");
        int32_t p[3] = {8388607, -8388608, 1234567};
        int32_t q[3] = {-7654321, 8388607, 7000000};
        CHECK_EQ_I128(esk_idot_i32(p, q, 3), ref_idot32(p, q, 3),
                       "idot_i32 matches wide reference for signed 24-bit operands");
    }

    /* ============ Attention P0 boundary: 24-bit operands, K=4096 ============ */
    {
        const size_t N = 4096;
        const int32_t qmax = (1 << 23) - 1;
        int32_t *a = malloc(N * sizeof *a), *b = malloc(N * sizeof *b);
        for (size_t i = 0; i < N; i++) { a[i] = qmax; b[i] = qmax; }
        esk_i128 want = esk_i128_widen_mul_i64(qmax, qmax) * (esk_i128)N;
        CHECK_EQ_I128(esk_idot_i32(a, b, N), want,
                       "idot_i32 exact at Attention 24-bit K=4096 bound");
        free(a); free(b);
    }

    /* ============ row-major i32 -> i128 ABI matmul ============ */
    {
        const int32_t a[6] = {2, -3, 5, 7, 11, -13};       /* [2,3] */
        const int32_t b[12] = {17, -19, 23, 29,             /* [3,4] */
                               -31, 37, -41, 43,
                               47, 53, -59, 61};
        esk_i128_abi out[8];
        int ok = esk_imatmul_i32(a, b, 2, 3, 4, out);
        int cells_ok = ok;
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 4; j++) {
                esk_i128 ref = 0;
                for (size_t k = 0; k < 3; k++)
                    ref += (esk_i128)a[i * 3 + k] * (esk_i128)b[k * 4 + j];
                if (esk_i128_from_abi(out[i * 4 + j]) != ref) cells_ok = 0;
            }
        }
        CHECK(cells_ok, "imatmul_i32 row-major layout and signed products are exact");
    }

    /* ============ matrix contraction order is byte-identical ============ */
    {
        const size_t R = 2, K = 257, C = 3;
        int32_t *a = malloc(R * K * sizeof *a), *b = malloc(K * C * sizeof *b);
        int32_t *sa = malloc(R * K * sizeof *sa), *sb = malloc(K * C * sizeof *sb);
        size_t *idx = malloc(K * sizeof *idx);
        esk_i128_abi gold[R * C], shuffled[R * C];
        for (size_t i = 0; i < R * K; i++)
            a[i] = (int32_t)(esk_rng_next_u64(&rng) & 0xFFFFFFu) - (1 << 23);
        for (size_t i = 0; i < K * C; i++)
            b[i] = (int32_t)(esk_rng_next_u64(&rng) & 0xFFFFFFu) - (1 << 23);
        for (size_t k = 0; k < K; k++) idx[k] = k;
        int order_ok = esk_imatmul_i32(a, b, R, K, C, gold);
        for (int trial = 0; trial < 50 && order_ok; trial++) {
            shuffle(idx, K, &rng);
            for (size_t i = 0; i < R; i++)
                for (size_t k = 0; k < K; k++) sa[i * K + k] = a[i * K + idx[k]];
            for (size_t k = 0; k < K; k++)
                for (size_t j = 0; j < C; j++) sb[k * C + j] = b[idx[k] * C + j];
            if (!esk_imatmul_i32(sa, sb, R, K, C, shuffled) ||
                memcmp(gold, shuffled, sizeof gold) != 0) order_ok = 0;
        }
        CHECK(order_ok, "CONTRACT: imatmul_i32 is byte-identical across 50 contraction orders");
        free(a); free(b); free(sa); free(sb); free(idx);
    }

    /* ============ matmul failure and degenerate-shape contract ============ */
    {
        int32_t one = 1;
        esk_i128_abi out[6];
        memset(out, 0xA5, sizeof out);
        CHECK(esk_imatmul_i32(NULL, NULL, 0, SIZE_MAX, 7, NULL),
              "imatmul_i32 empty output is a valid no-op");
        CHECK(esk_imatmul_i32(NULL, NULL, 2, 0, 3, out),
              "imatmul_i32 zero inner dimension accepts NULL inputs");
        int zeros = 1;
        for (size_t i = 0; i < 6; i++)
            if (esk_i128_from_abi(out[i]) != 0) zeros = 0;
        CHECK(zeros, "imatmul_i32 zero inner dimension writes exact zero cells");
        CHECK(!esk_imatmul_i32(NULL, &one, 1, 1, 1, out),
              "imatmul_i32 rejects NULL A for nonempty contraction");
        CHECK(!esk_imatmul_i32(&one, NULL, 1, 1, 1, out),
              "imatmul_i32 rejects NULL B for nonempty contraction");
        CHECK(!esk_imatmul_i32(&one, &one, 1, 1, 1, NULL),
              "imatmul_i32 rejects NULL output for nonempty output");
        CHECK(!esk_imatmul_i32(&one, &one, SIZE_MAX, 2, 1, out),
              "imatmul_i32 rejects overflowing A index product");
        CHECK(!esk_imatmul_i32(&one, &one, SIZE_MAX, 0, 2, out),
              "imatmul_i32 rejects overflowing output index product");
    }

    /* ===================== dot_exact scaling ===================== */
    {
        /* unit scales, F=0 -> fixed value equals the integer dot exactly */
        int16_t a[4] = {10, 20, 30, 40};
        int16_t b[4] = {1, 2, 3, 4};      /* 10+40+90+160 = 300 */
        bool ex;
        esk_fixed d = esk_dot_exact_i16(a, b, 4, 1.0, 1.0, 0, &ex);
        CHECK(d.raw == esk_i128_from_i64(300) && ex, "dot_exact_i16 unit-scale F=0 == 300 exact");
    }

    /* ============ THE CONTRACT: order-independence (byte-identical) ============ */
    {
        const size_t N = 4096;
        int16_t *a = malloc(N * sizeof *a), *b = malloc(N * sizeof *b);
        size_t  *idx = malloc(N * sizeof *idx);
        for (size_t i = 0; i < N; i++) {
            a[i] = (int16_t)(esk_rng_next_u64(&rng) & 0xFFFF);   /* full i16 range, signed */
            b[i] = (int16_t)(esk_rng_next_u64(&rng) & 0xFFFF);
            idx[i] = i;
        }
        esk_i128 gold = esk_idot_i16(a, b, N);          /* canonical order */
        esk_i128 gold_ref = ref_idot16(a, b, N);
        CHECK_EQ_I128(gold, gold_ref, "idot matches independent wide reference");

        int order_ok = 1;
        int16_t *sa = malloc(N * sizeof *sa), *sb = malloc(N * sizeof *sb);
        for (int trial = 0; trial < 200; trial++) {
            shuffle(idx, N, &rng);
            for (size_t i = 0; i < N; i++) { sa[i] = a[idx[i]]; sb[i] = b[idx[i]]; }
            esk_i128 s = esk_idot_i16(sa, sb, N);
            if (s != gold) order_ok = 0;
            /* also check the fully-scaled fixed<128,F> result is byte-identical */
            bool ex1, ex2;
            esk_fixed fg = esk_dot_exact_i16(a, b, N, 0.015625, 0.5, 20, &ex1);   /* dyadic scales */
            esk_fixed fs = esk_dot_exact_i16(sa, sb, N, 0.015625, 0.5, 20, &ex2);
            if (memcmp(&fg.raw, &fs.raw, sizeof fg.raw) != 0) order_ok = 0;
        }
        CHECK(order_ok, "CONTRACT: 200 shuffles x 4096 elems -> byte-identical dot (any order)");
        free(a); free(b); free(idx); free(sa); free(sb);
    }

    /* ============ contract also holds at extreme magnitudes (i64 would overflow) ============ */
    {
        const size_t N = 100000;
        int16_t *a = malloc(N * sizeof *a), *b = malloc(N * sizeof *b);
        size_t  *idx = malloc(N * sizeof *idx);
        for (size_t i = 0; i < N; i++) { a[i] = 32767; b[i] = 32767; idx[i] = i; }
        /* Σ = N * 32767^2 = 1.0737e14 — fits i64 here but we push i128 semantics */
        esk_i128 gold = esk_idot_i16(a, b, N);
        esk_i128 want = esk_i128_widen_mul_i64(32767, 32767) * (esk_i128)(int64_t)N;
        CHECK_EQ_I128(gold, want, "idot of 100k x (32767*32767) exact in i128");
        int ok = 1;
        for (int t = 0; t < 20; t++) {
            shuffle(idx, N, &rng);
            int16_t *sa = malloc(N * sizeof *sa), *sb = malloc(N * sizeof *sb);
            for (size_t i = 0; i < N; i++){ sa[i]=a[idx[i]]; sb[i]=b[idx[i]]; }
            if (esk_idot_i16(sa, sb, N) != gold) ok = 0;
            free(sa); free(sb);
        }
        CHECK(ok, "CONTRACT at max magnitude: 20 shuffles byte-identical");
        free(a); free(b); free(idx);
    }

    /* ===================== accum128 running sum + merge ===================== */
    {
        esk_accum128 acc; esk_accum128_init(&acc);
        for (int i = 1; i <= 1000; i++) esk_accum128_add_i64(&acc, i);   /* Σ1..1000 = 500500 */
        CHECK_EQ_I128(esk_accum128_value(&acc), esk_i128_from_i64(500500), "accum128 Σ(1..1000) == 500500");
        CHECK(acc.count == 1000, "accum128 tracks count");

        /* partial-merge equals single-stream (associativity of reduction) */
        esk_accum128 p1, p2; esk_accum128_init(&p1); esk_accum128_init(&p2);
        for (int i = 1; i <= 500; i++)     esk_accum128_add_i64(&p1, i);
        for (int i = 501; i <= 1000; i++)  esk_accum128_add_i64(&p2, i);
        esk_accum128_merge(&p1, &p2);
        CHECK_EQ_I128(esk_accum128_value(&p1), esk_i128_from_i64(500500), "accum128 partial-merge == single-stream");
        CHECK(p1.count == 1000, "accum128 merge sums counts");
    }

    return harness_summary("test_dot_exact");
}
