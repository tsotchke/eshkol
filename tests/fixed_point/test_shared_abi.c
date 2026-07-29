/* test_shared_abi.c — external-consumer contract for libeshkol-fixedpoint.
 *
 * This file is intentionally linked against the shared target only: it must not
 * compile eshkol_fixed_point.c into the test executable. That pins the installed
 * C header + exported ABI that Attention's portable reference will consume.
 */
#include "eshkol_fixed_point.h"
#include "test_harness.h"

int main(void) {
    {
        const int32_t a[6] = {8388607, -8388608, 17,
                              -23, 29, -31};                 /* [2,3] */
        const int32_t b[6] = {37, -41,
                              43, 47,
                              -53, 59};                      /* [3,2] */
        esk_i128_abi out[4];
        CHECK(esk_imatmul_i32(a, b, 2, 3, 2, out),
              "shared ABI exact matmul call succeeds");

        int exact = 1;
        for (size_t i = 0; i < 2; i++) {
            for (size_t j = 0; j < 2; j++) {
                esk_i128 want = 0;
                for (size_t k = 0; k < 3; k++)
                    want += (esk_i128)a[i * 3 + k] * (esk_i128)b[k * 2 + j];
                if (esk_i128_from_abi(out[i * 2 + j]) != want) exact = 0;
            }
        }
        CHECK(exact, "shared ABI exports exact signed i32-to-i128 results");
    }

    {
        esk_i128_abi out[2];
        CHECK(esk_imatmul_i32(NULL, NULL, 1, 0, 2, out),
              "shared ABI preserves zero-inner contract");
        CHECK(esk_i128_from_abi(out[0]) == 0 && esk_i128_from_abi(out[1]) == 0,
              "shared ABI zero-inner cells decode as i128 zero");
    }

    return harness_summary("test_shared_abi");
}
