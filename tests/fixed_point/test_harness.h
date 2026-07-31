/* Tiny self-checking test harness for the fixed-point C tests.
 * PASS/FAIL lines mirror the .esk test style; exit code is nonzero on any fail. */
#ifndef ESHKOL_FP_TEST_HARNESS_H
#define ESHKOL_FP_TEST_HARNESS_H
#include <stdio.h>
#include <stdint.h>

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, label) do {                                        \
    if (cond) { g_pass++; printf("PASS: %s\n", (label)); }             \
    else      { g_fail++; printf("FAIL: %s\n", (label)); }             \
} while (0)

#define CHECK_EQ_I128(a, b, label) do {                                \
    esk_i128 _va = (a), _vb = (b);                                     \
    if (_va == _vb) { g_pass++; printf("PASS: %s\n", (label)); }       \
    else { g_fail++; char _sa[41], _sb[41];                           \
        esk_i128_to_string(_va, _sa, sizeof _sa);                      \
        esk_i128_to_string(_vb, _sb, sizeof _sb);                      \
        printf("FAIL: %s (got %s want %s)\n", (label), _sa, _sb); }    \
} while (0)

static int harness_summary(const char *suite) {
    printf("---\n%s: %d/%d checks passed", suite, g_pass, g_pass + g_fail);
    if (g_fail) printf(", %d FAILED", g_fail);
    printf("\n");
    return g_fail == 0 ? 0 : 1;
}

#endif
