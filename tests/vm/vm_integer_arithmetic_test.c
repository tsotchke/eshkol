#include <stdint.h>
#include <stdio.h>
#include "vm_integer_arithmetic.h"

/* An unsigned-magnitude oracle never performs the overflowing signed divide. */
static int64_t reference_remainder(int64_t a, int64_t b) {
    uint64_t magnitude_a = a < 0 ? UINT64_C(0) - (uint64_t)a : (uint64_t)a;
    uint64_t magnitude_b = b < 0 ? UINT64_C(0) - (uint64_t)b : (uint64_t)b;
    uint64_t remainder = magnitude_a % magnitude_b;
    return a < 0 ? -(int64_t)remainder : (int64_t)remainder;
}

int main(void) {
    const int64_t values[] = {
        INT64_MIN, INT64_MIN + 1, -4294967296LL, -17, -7, -3, -2, -1,
        0, 1, 2, 3, 7, 17, 4294967296LL, INT64_MAX - 1, INT64_MAX
    };
    unsigned checks = 0;
    for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        for (unsigned j = 0; j < sizeof(values) / sizeof(values[0]); ++j) {
            if (values[j] == 0) continue;
            int64_t actual = vm_i64_remainder_nonzero(values[i], values[j]);
            int64_t expected = reference_remainder(values[i], values[j]);
            if (actual != expected) {
                fprintf(stderr, "FAIL: remainder %lld %% %lld = %lld, expected %lld\n",
                        (long long)values[i], (long long)values[j],
                        (long long)actual, (long long)expected);
                return 1;
            }
            ++checks;
        }
    }
    printf("PASS: VM integer remainder boundaries (%u checks)\n", checks);
    return 0;
}
