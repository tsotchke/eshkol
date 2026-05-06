/*
 * Hosted bignum display support.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/core/bignum.h"

#include <cstdio>
#include <cstring>

extern "C" void eshkol_bignum_display(const eshkol_bignum_t* a, void* file) {
    if (!a || !file) return;
    FILE* f = (FILE*)file;

    if (eshkol_bignum_is_zero(a)) {
        std::fprintf(f, "0");
        return;
    }

    if (a->sign) std::fprintf(f, "-");

    if (a->num_limbs == 1) {
        std::fprintf(f, "%llu", (unsigned long long)BIGNUM_LIMBS(a)[0]);
        return;
    }

    char stack_buf[256];
    size_t max_digits = (size_t)a->num_limbs * 20 + 2;

    if (max_digits <= sizeof(stack_buf)) {
        uint64_t work[13];
        std::memcpy(work, BIGNUM_LIMBS(a), a->num_limbs * sizeof(uint64_t));
        uint32_t work_limbs = a->num_limbs;

        size_t pos = 0;
        while (work_limbs > 0) {
            bool all_zero = true;
            for (uint32_t i = 0; i < work_limbs; ++i) {
                if (work[i] != 0) {
                    all_zero = false;
                    break;
                }
            }
            if (all_zero) break;

            __uint128_t rem = 0;
            for (int32_t i = (int32_t)work_limbs - 1; i >= 0; --i) {
                __uint128_t cur = (rem << 64) | work[i];
                work[i] = (uint64_t)(cur / 10);
                rem = cur % 10;
            }
            stack_buf[pos++] = '0' + (char)(uint64_t)rem;

            while (work_limbs > 0 && work[work_limbs - 1] == 0) {
                --work_limbs;
            }
        }

        for (size_t i = pos; i > 0; --i) {
            std::fputc(stack_buf[i - 1], f);
        }
    } else {
        std::fprintf(f, "<bignum:%u-limbs>", a->num_limbs);
    }
}
