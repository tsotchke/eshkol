/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Memory primitives for freestanding Eshkol images.  LLVM lowers aggregate
 * copies and zeroing (tagged values, structs, arrays) to these calls, so
 * every bare-metal image needs them.  Built with -fno-builtin so the loops
 * are not themselves turned back into calls.
 */
#include "eshkol_embedded.h"

void* memcpy(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    while (n--) *d++ = *s++;
    return dst;
}

void* memmove(void* dst, const void* src, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    if (d == s || n == 0) return dst;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else {
        d += n;
        s += n;
        while (n--) *--d = *--s;
    }
    return dst;
}

void* memset(void* dst, int c, size_t n) {
    uint8_t* d = (uint8_t*)dst;
    while (n--) *d++ = (uint8_t)c;
    return dst;
}

int memcmp(const void* a, const void* b, size_t n) {
    const uint8_t* x = (const uint8_t*)a;
    const uint8_t* y = (const uint8_t*)b;
    for (; n; --n, ++x, ++y) {
        if (*x != *y) return (int)*x - (int)*y;
    }
    return 0;
}
