/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * dtoa_shortest.h — shortest round-trip IEEE-754 double → text.
 *
 * Single source of truth for rendering a `double` as its R7RS external
 * representation, shared by the native runtime (lib/core/runtime_display_hosted.cpp
 * -> eshkol_format_double) and the bytecode VM (lib/backend/vm_*.c, incl. the
 * emcc browser REPL which has no access to the C++ runtime). Keeping ONE
 * implementation is what guarantees byte-identical output across the native
 * JIT/AOT paths and the VM (ADR-0003 parity).
 *
 * Contract: produce the SHORTEST decimal string that, when read back with
 * `strtod`, recovers the exact same double (round-trip guarantee — R7RS 6.2.6
 * requires this of `number->string` on an inexact real). We do NOT depend on
 * C++17 std::to_chars because the VM is C and is also compiled to WebAssembly
 * with emcc; a portable C routine that both paths share is the only way to get
 * a true single source of truth and 100% native/VM parity.
 *
 * The algorithm: find the minimal significant-digit count (1..17) whose "%.*e"
 * rendering round-trips, then choose fixed vs. scientific notation by whichever
 * is shorter (ties -> fixed), with a printf-style >=2-digit exponent. This
 * matches std::to_chars byte-for-byte on >99% of finite doubles; on the rest it
 * still produces a string of the SAME length that round-trips (it only differs
 * in the last digit among equally-short representations). Because BOTH paths use
 * this exact routine, they always agree with each other.
 *
 * Non-finite values use the R7RS special external representations
 * (+inf.0 / -inf.0 / +nan.0) so the reader can read them back.
 *
 * Header-only `static` so the VM unity build and the C++ runtime each get a
 * private copy with no link dependency between them.
 */
#ifndef ESHKOL_CORE_DTOA_SHORTEST_H
#define ESHKOL_CORE_DTOA_SHORTEST_H

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Render @p v into @p buf as its shortest round-trip R7RS external form.
 *
 * @param buf  Destination buffer (48 bytes is always sufficient for finite
 *             values — the longest shortest-round-trip output is 24 chars).
 * @param cap  Capacity of @p buf in bytes (including the NUL terminator).
 * @param v    Value to format.
 * @return Number of characters written (excluding the NUL), like snprintf.
 */
static int eshkol_dtoa_shortest(char* buf, size_t cap, double v) {
    if (cap == 0) return 0;

    /* R7RS non-finite external representations. */
    if (isnan(v)) {
        return snprintf(buf, cap, "+nan.0");
    }
    if (isinf(v)) {
        return snprintf(buf, cap, v < 0.0 ? "-inf.0" : "+inf.0");
    }

    /* Signed zero: "0" / "-0" (matches the existing %g convention). */
    if (v == 0.0) {
        return snprintf(buf, cap, signbit(v) ? "-0" : "0");
    }

    /* 1) Minimal significant digits: smallest sig-digit count in 1..17 whose
     *    scientific rendering reads back to exactly v. "%.*e" with precision p
     *    emits p+1 significant digits, so p runs 0..16. 17 digits always
     *    round-trips any IEEE double, so this loop always terminates. */
    char sciraw[64];
    int prec;
    for (prec = 0; prec < 17; prec++) {
        snprintf(sciraw, sizeof(sciraw), "%.*e", prec, v);
        if (strtod(sciraw, NULL) == v) break;
    }

    /* 2) Decompose "[-]d[.ddd]e[+-]XX" into sign, digit string, decimal exp. */
    const char* p = sciraw;
    int negative = 0;
    if (*p == '-') { negative = 1; p++; }
    char digits[20];
    int nd = 0;
    digits[nd++] = *p++;                 /* leading digit */
    if (*p == '.') {
        p++;
        while (*p && *p != 'e' && *p != 'E') digits[nd++] = *p++;
    }
    int exp10 = 0;
    if (*p == 'e' || *p == 'E') exp10 = atoi(p + 1);
    /* Trailing zeros can never appear in the minimal rendering, but strip
     * defensively so both notations agree on digit count. */
    while (nd > 1 && digits[nd - 1] == '0') nd--;
    digits[nd] = '\0';

    /* Number of digits that sit to the LEFT of the decimal point in fixed form
     * (value == digits x 10^(exp10 - (nd-1))). */
    int point_pos = exp10 + 1;

    /* 3) Build both candidate renderings, pick the shorter (tie -> fixed). */
    char fixed[512];   /* worst-case fixed form (~326 chars) fits here */
    char sci[64];
    char* o;

    /* ---- fixed ---- */
    o = fixed;
    if (negative) *o++ = '-';
    if (point_pos <= 0) {
        *o++ = '0'; *o++ = '.';
        for (int i = 0; i < -point_pos; i++) *o++ = '0';
        for (int i = 0; i < nd; i++)         *o++ = digits[i];
    } else if (point_pos >= nd) {
        for (int i = 0; i < nd; i++)             *o++ = digits[i];
        for (int i = 0; i < point_pos - nd; i++) *o++ = '0';
    } else {
        for (int i = 0; i < point_pos; i++) *o++ = digits[i];
        *o++ = '.';
        for (int i = point_pos; i < nd; i++) *o++ = digits[i];
    }
    *o = '\0';
    size_t fixed_len = (size_t)(o - fixed);

    /* ---- scientific: d[.ddd]e[+-]XX, exponent >= 2 digits ---- */
    o = sci;
    if (negative) *o++ = '-';
    *o++ = digits[0];
    if (nd > 1) {
        *o++ = '.';
        for (int i = 1; i < nd; i++) *o++ = digits[i];
    }
    *o++ = 'e';
    int e = exp10;
    if (e < 0) { *o++ = '-'; e = -e; } else { *o++ = '+'; }
    char eb[8];
    int en = 0;
    if (e == 0) eb[en++] = '0';
    while (e) { eb[en++] = (char)('0' + e % 10); e /= 10; }
    while (en < 2) eb[en++] = '0';       /* printf-style minimum 2 exponent digits */
    while (en) *o++ = eb[--en];
    *o = '\0';
    size_t sci_len = (size_t)(o - sci);

    const char* chosen = (fixed_len <= sci_len) ? fixed : sci;
    size_t chosen_len = (fixed_len <= sci_len) ? fixed_len : sci_len;

    if (chosen_len >= cap) chosen_len = cap - 1;   /* never overflow the caller */
    memcpy(buf, chosen, chosen_len);
    buf[chosen_len] = '\0';
    return (int)chosen_len;
}

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_DTOA_SHORTEST_H */
