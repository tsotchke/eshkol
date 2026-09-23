/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * number_syntax.h -- R7RS 7.1.1 <number> syntax: the one recognizer every
 * Eshkol reader uses to decide whether a token is a number and what its parts
 * are.
 *
 * Four readers turn text into numbers: the source parser (a literal in a
 * program), the runtime reader (`read`), `string->number`, and the bytecode
 * VM's parser and `string->number`. Each used to carry its own partial idea of
 * the grammar, so they disagreed: the parser split `1+1i` into `1` and `+1i`,
 * the VM read it as a call, `string->number` answered #f while `read` returned
 * a symbol, and `#i42` was the exact integer 42. This header is the grammar.
 * It never builds a value -- each engine has its own representation -- it
 * decides the form and hands each real part over in a canonical spelling that
 * every engine's existing real-number constructor already reads:
 *
 *   INTEGER   exact: optional '-' then decimal digits         "-255"
 *   RATIONAL  exact: optional '-' digits '/' digits (decimal)  "-3/4"
 *   DECIMAL   inexact: a decimal spelling whose correctly
 *             rounded double (strtod) is the value            "1.5e2"
 *   INFNAN    inexact: +inf.0 -inf.0 +nan.0 -nan.0             "+inf.0"
 *
 * A radix-2/8/16 part is converted to decimal digits here, bignum-sized parts
 * included, so no consumer needs a radix-aware bignum reader.
 *
 * ── Grammar (R7RS 7.1.1) ─────────────────────────────────────────────────
 *
 *   <number>    -> <prefix R> <complex R>        R = 2, 8, 10, 16
 *   <complex R> -> <real R> | <real R> @ <real R>
 *                | <real R> + <ureal R> i | <real R> - <ureal R> i
 *                | <real R> + i | <real R> - i | <real R> <infnan> i
 *                | + <ureal R> i | - <ureal R> i | <infnan> i | + i | - i
 *   <real R>    -> <sign> <ureal R> | <infnan>
 *   <ureal R>   -> <uinteger R> | <uinteger R> / <uinteger R> | <decimal R>
 *   <decimal 10> -> <uinteger 10> <suffix> | . <digit 10>+ <suffix>
 *                 | <digit 10>+ . <digit 10>* <suffix>
 *   <suffix>    -> <empty> | e <sign> <digit 10>+
 *   <infnan>    -> +inf.0 | -inf.0 | +nan.0 | -nan.0
 *   <prefix R>  -> <radix R> <exactness> | <exactness> <radix R>
 *   <radix R>   -> #b | #o | #d | #x   (#d or empty for R = 10)
 *   <exactness> -> <empty> | #i | #e
 *
 * Letters are case-insensitive (`#X1F`, `1E3`, `+INF.0`, `1+2I`). A pure
 * imaginary needs its sign: `+2i` is a number, `2i` is an identifier.
 *
 * ── Exactness (R7RS 6.2.5) ───────────────────────────────────────────────
 *
 * Without a prefix an integer or rational part is exact and a decimal or
 * <infnan> part inexact. `#e` turns a decimal part into the exact integer or
 * rational it spells (`#e1.5` is 3/2); an <infnan> has no exact form and is
 * refused (ESHKOL_NUMSYN_NO_EXACT_FORM). `#i` turns an exact part into the
 * DECIMAL spelling of its correctly rounded double (`#i1/3` is the double
 * nearest 1/3, `#i42` is 42.0), so exactness is carried by the part kind and
 * a consumer never converts exact to inexact itself.
 *
 * A rectangular form whose imaginary part is an exact zero IS the real number
 * (`1+0i` is 1, `#e1.5+0i` is 3/2), as is a polar form with an exact zero
 * angle; both come back as ESHKOL_NUMSYN_REAL. Eshkol's complex numbers carry
 * inexact parts (numeric-tower.md): the parts of a non-real result are always
 * DECIMAL or INFNAN, and a non-real complex under `#e` has no representation
 * and is refused with ESHKOL_NUMSYN_NO_EXACT_FORM.
 *
 * Header-only `static inline` so the VM unity build (native and WebAssembly),
 * the C++ runtime and the source parser each get a private copy with no link
 * dependency, like symbol_syntax.h and dtoa_shortest.h.
 */
#ifndef ESHKOL_CORE_NUMBER_SYNTAX_H
#define ESHKOL_CORE_NUMBER_SYNTAX_H

#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_NUMSYN_ZERO = 0,
    ESHKOL_NUMSYN_INTEGER,
    ESHKOL_NUMSYN_RATIONAL,
    ESHKOL_NUMSYN_DECIMAL,
    ESHKOL_NUMSYN_INFNAN
} eshkol_numsyn_part_kind_t;

typedef struct {
    eshkol_numsyn_part_kind_t kind;
    /* Canonical spelling (see the file comment), NUL-terminated and owned by
     * the result. ZERO appears only while parsing (the absent real part of a
     * pure imaginary) and never in a returned result. */
    char* text;
} eshkol_numsyn_part_t;

typedef enum {
    ESHKOL_NUMSYN_REAL = 0,     /* part[0] */
    ESHKOL_NUMSYN_RECTANGULAR,  /* part[0] + part[1] i */
    ESHKOL_NUMSYN_POLAR         /* part[0] @ part[1] */
} eshkol_numsyn_form_t;

typedef enum {
    ESHKOL_NUMSYN_OK = 0,
    ESHKOL_NUMSYN_NOT_A_NUMBER,   /* not <number>: an identifier, or garbage */
    ESHKOL_NUMSYN_DIVIDE_BY_ZERO, /* a rational part n/0 */
    ESHKOL_NUMSYN_NO_EXACT_FORM,  /* #e on <infnan>, or an exact non-real complex */
    ESHKOL_NUMSYN_TOO_LARGE,      /* #e exponent past ESHKOL_NUMSYN_MAX_EXACT_EXPONENT */
    ESHKOL_NUMSYN_NO_MEMORY
} eshkol_numsyn_status_t;

typedef struct {
    eshkol_numsyn_form_t form;
    eshkol_numsyn_part_t part[2];
} eshkol_numsyn_t;

/* `#e1e400` spells a 401-digit integer; the exact expansion of a decimal
 * exponent is bounded so a short token cannot demand gigabytes. */
#define ESHKOL_NUMSYN_MAX_EXACT_EXPONENT 4096

static inline const char* eshkol_number_syntax_status_message(eshkol_numsyn_status_t s) {
    switch (s) {
    case ESHKOL_NUMSYN_OK:             return "ok";
    case ESHKOL_NUMSYN_NOT_A_NUMBER:   return "not a number";
    case ESHKOL_NUMSYN_DIVIDE_BY_ZERO: return "division by zero in rational literal";
    case ESHKOL_NUMSYN_NO_EXACT_FORM:
        return "number has no exact representation (an infinity or NaN, or a "
               "non-real complex, whose parts are inexact)";
    case ESHKOL_NUMSYN_TOO_LARGE:      return "exact number exponent out of range";
    case ESHKOL_NUMSYN_NO_MEMORY:      return "out of memory";
    }
    return "invalid number";
}

static inline void eshkol_number_syntax_free(eshkol_numsyn_t* n) {
    if (!n) return;
    free(n->part[0].text);
    free(n->part[1].text);
    n->part[0].text = NULL;
    n->part[1].text = NULL;
}

/* ── internals ────────────────────────────────────────────────────────── */

static inline int eshkol_numsyn_lower(int c) {
    return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c;
}

static inline int eshkol_numsyn_digit_value(int c, int radix) {
    int v;
    c = eshkol_numsyn_lower(c);
    if (c >= '0' && c <= '9') v = c - '0';
    else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
    else return -1;
    return v < radix ? v : -1;
}

static inline char* eshkol_numsyn_strndup(const char* s, size_t n) {
    char* out = (char*)malloc(n + 1);
    if (!out) return NULL;
    if (n) memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

/* Append the decimal expansion of the radix-R digit string s[0..n) to out
 * (which has room for at least 4*n+2 bytes). Returns the digits written. */
static inline size_t eshkol_numsyn_to_decimal(const char* s, size_t n, int radix, char* out) {
    size_t i, k, used = 1;
    unsigned char* dec;          /* least-significant digit first */
    size_t cap = n * 4 + 2;
    if (radix == 10) {
        while (n > 1 && s[0] == '0') { s++; n--; }
        memcpy(out, s, n);
        return n;
    }
    dec = (unsigned char*)calloc(cap, 1);
    if (!dec) return 0;
    for (i = 0; i < n; ++i) {
        int carry = eshkol_numsyn_digit_value((unsigned char)s[i], radix);
        for (k = 0; k < used; ++k) {
            int p = dec[k] * radix + carry;
            dec[k] = (unsigned char)(p % 10);
            carry = p / 10;
        }
        while (carry && used < cap) { dec[used++] = (unsigned char)(carry % 10); carry /= 10; }
    }
    while (used > 1 && dec[used - 1] == 0) used--;
    for (k = 0; k < used; ++k) out[k] = (char)('0' + dec[used - 1 - k]);
    free(dec);
    return used;
}

/* Length of a run of radix digits starting at s[i]. */
static inline size_t eshkol_numsyn_digit_run(const char* s, size_t n, size_t i, int radix) {
    size_t j = i;
    while (j < n && eshkol_numsyn_digit_value((unsigned char)s[j], radix) >= 0) j++;
    return j - i;
}

/* Is s[0..n) exactly one of the four <infnan> spellings (case-insensitive)? */
static inline int eshkol_numsyn_is_infnan(const char* s, size_t n) {
    static const char* const tails[2] = { "inf.0", "nan.0" };
    int t;
    size_t k;
    if (n != 6 || (s[0] != '+' && s[0] != '-')) return 0;
    for (t = 0; t < 2; ++t) {
        for (k = 0; k < 5; ++k)
            if (eshkol_numsyn_lower((unsigned char)s[1 + k]) != tails[t][k]) break;
        if (k == 5) return 1;
    }
    return 0;
}

/* Recognize <real R> (when allow_bare_sign, also a lone sign meaning ±1, the
 * `+i` / `-i` unit) in s[0..n) and write its canonical part. `require_sign`
 * is the pure-imaginary rule. Returns OK or NOT_A_NUMBER / DIVIDE_BY_ZERO /
 * NO_MEMORY; `exact_prefix` is the #e/#i/none marker (1 / -1 / 0). */
static inline eshkol_numsyn_status_t eshkol_numsyn_real(const char* s, size_t n, int radix,
                                                        int require_sign, int allow_bare_sign,
                                                        eshkol_numsyn_part_t* out) {
    size_t i = 0;
    int negative = 0;
    size_t int_len, frac_len = 0, exp_start = 0, exp_len = 0;
    int has_dot = 0, has_exp = 0;
    size_t int_at, frac_at = 0;

    memset(out, 0, sizeof(*out));
    if (n == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;

    if (eshkol_numsyn_is_infnan(s, n)) {
        char buf[7];
        size_t k;
        buf[0] = s[0];
        for (k = 1; k < 6; ++k) buf[k] = (char)eshkol_numsyn_lower((unsigned char)s[k]);
        buf[6] = '\0';
        out->kind = ESHKOL_NUMSYN_INFNAN;
        out->text = eshkol_numsyn_strndup(buf, 6);
        return out->text ? ESHKOL_NUMSYN_OK : ESHKOL_NUMSYN_NO_MEMORY;
    }

    if (s[0] == '+' || s[0] == '-') { negative = s[0] == '-'; i = 1; }
    else if (require_sign) return ESHKOL_NUMSYN_NOT_A_NUMBER;

    if (i == n) {
        if (!allow_bare_sign || i == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;
        out->kind = ESHKOL_NUMSYN_INTEGER;
        out->text = eshkol_numsyn_strndup(negative ? "-1" : "1", negative ? 2 : 1);
        return out->text ? ESHKOL_NUMSYN_OK : ESHKOL_NUMSYN_NO_MEMORY;
    }

    int_at = i;
    int_len = eshkol_numsyn_digit_run(s, n, i, radix);
    i += int_len;

    /* <uinteger R> / <uinteger R> */
    if (i < n && s[i] == '/') {
        size_t den_at = i + 1;
        size_t den_len = eshkol_numsyn_digit_run(s, n, den_at, radix);
        size_t k, zero = 1;
        char* text;
        size_t w = 0;
        if (int_len == 0 || den_len == 0 || den_at + den_len != n) return ESHKOL_NUMSYN_NOT_A_NUMBER;
        for (k = 0; k < den_len; ++k) if (s[den_at + k] != '0') { zero = 0; break; }
        if (zero) return ESHKOL_NUMSYN_DIVIDE_BY_ZERO;
        text = (char*)malloc(4 * (int_len + den_len) + 8);
        if (!text) return ESHKOL_NUMSYN_NO_MEMORY;
        if (negative) text[w++] = '-';
        w += eshkol_numsyn_to_decimal(s + int_at, int_len, radix, text + w);
        text[w++] = '/';
        w += eshkol_numsyn_to_decimal(s + den_at, den_len, radix, text + w);
        text[w] = '\0';
        out->kind = ESHKOL_NUMSYN_RATIONAL;
        out->text = text;
        return ESHKOL_NUMSYN_OK;
    }

    /* <decimal 10>: only radix 10 has a point or an exponent. */
    if (radix == 10 && i < n && s[i] == '.') {
        has_dot = 1;
        frac_at = i + 1;
        frac_len = eshkol_numsyn_digit_run(s, n, frac_at, 10);
        i = frac_at + frac_len;
        if (int_len == 0 && frac_len == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;
    }
    if (radix == 10 && i < n && (s[i] == 'e' || s[i] == 'E')) {
        if (int_len == 0 && frac_len == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;
        has_exp = 1;
        exp_start = i + 1;
        i = exp_start;
        if (i < n && (s[i] == '+' || s[i] == '-')) i++;
        exp_len = eshkol_numsyn_digit_run(s, n, i, 10);
        if (exp_len == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;
        i += exp_len;
    }
    if (i != n) return ESHKOL_NUMSYN_NOT_A_NUMBER;
    if (int_len == 0 && !has_dot) return ESHKOL_NUMSYN_NOT_A_NUMBER;

    if (has_dot || has_exp) {
        out->kind = ESHKOL_NUMSYN_DECIMAL;
        out->text = eshkol_numsyn_strndup(s, n);
        if (out->text) {
            size_t k;
            for (k = 0; k < n; ++k) if (out->text[k] == 'E') out->text[k] = 'e';
        }
        return out->text ? ESHKOL_NUMSYN_OK : ESHKOL_NUMSYN_NO_MEMORY;
    }

    {
        char* text = (char*)malloc(4 * int_len + 4);
        size_t w = 0;
        if (!text) return ESHKOL_NUMSYN_NO_MEMORY;
        if (negative) text[w++] = '-';
        w += eshkol_numsyn_to_decimal(s + int_at, int_len, radix, text + w);
        text[w] = '\0';
        out->kind = ESHKOL_NUMSYN_INTEGER;
        out->text = text;
        (void)frac_at; (void)exp_start;
        return ESHKOL_NUMSYN_OK;
    }
}

/* Rewrite a DECIMAL part as the exact INTEGER or RATIONAL it spells (#e). */
static inline eshkol_numsyn_status_t eshkol_numsyn_make_exact(eshkol_numsyn_part_t* p) {
    const char* s = p->text;
    size_t n = strlen(s), i = 0, k;
    int negative = 0;
    long exp10 = 0;
    char* digits;
    size_t nd = 0;
    char* text;
    size_t w = 0;

    if (p->kind == ESHKOL_NUMSYN_INFNAN) return ESHKOL_NUMSYN_NO_EXACT_FORM;
    if (p->kind != ESHKOL_NUMSYN_DECIMAL) return ESHKOL_NUMSYN_OK;

    digits = (char*)malloc(n + 2);
    if (!digits) return ESHKOL_NUMSYN_NO_MEMORY;
    if (s[i] == '+' || s[i] == '-') { negative = s[i] == '-'; i++; }
    for (; i < n && s[i] != 'e'; ++i) {
        if (s[i] == '.') continue;
        digits[nd++] = s[i];
    }
    /* Count the fraction digits. */
    {
        const char* dot = strchr(s, '.');
        const char* e = strchr(s, 'e');
        if (dot) exp10 -= (long)((e ? (size_t)(e - dot) : (size_t)(s + n - dot)) - 1);
        if (e) {
            long v = 0;
            int eneg = 0;
            const char* q = e + 1;
            if (*q == '+' || *q == '-') { eneg = *q == '-'; q++; }
            for (; *q; ++q) {
                v = v * 10 + (*q - '0');
                if (v > 10L * ESHKOL_NUMSYN_MAX_EXACT_EXPONENT) break;
            }
            exp10 += eneg ? -v : v;
        }
    }
    /* Strip leading zeros from the mantissa. */
    k = 0;
    while (k + 1 < nd && digits[k] == '0') k++;
    memmove(digits, digits + k, nd - k);
    nd -= k;
    if (nd == 0) { digits[0] = '0'; nd = 1; }
    if (nd == 1 && digits[0] == '0') exp10 = 0;   /* 0.0e999 is exactly 0 */
    if (exp10 > ESHKOL_NUMSYN_MAX_EXACT_EXPONENT || exp10 < -ESHKOL_NUMSYN_MAX_EXACT_EXPONENT) {
        free(digits);
        return ESHKOL_NUMSYN_TOO_LARGE;
    }

    text = (char*)malloc(nd + (size_t)(exp10 < 0 ? -exp10 : exp10) + 8);
    if (!text) { free(digits); return ESHKOL_NUMSYN_NO_MEMORY; }
    if (negative && !(nd == 1 && digits[0] == '0')) text[w++] = '-';
    memcpy(text + w, digits, nd);
    w += nd;
    if (exp10 >= 0) {
        long z;
        for (z = 0; z < exp10; ++z) text[w++] = '0';
        p->kind = ESHKOL_NUMSYN_INTEGER;
    } else {
        long z;
        text[w++] = '/';
        text[w++] = '1';
        for (z = 0; z < -exp10; ++z) text[w++] = '0';
        p->kind = ESHKOL_NUMSYN_RATIONAL;
    }
    text[w] = '\0';
    free(digits);
    free(p->text);
    p->text = text;
    return ESHKOL_NUMSYN_OK;
}

/* Is this part an exact zero (0, -0, 0/n)? */
static inline int eshkol_numsyn_part_is_exact_zero(const eshkol_numsyn_part_t* p) {
    const char* s;
    if (p->kind == ESHKOL_NUMSYN_ZERO) return 1;
    if (p->kind != ESHKOL_NUMSYN_INTEGER && p->kind != ESHKOL_NUMSYN_RATIONAL) return 0;
    s = p->text;
    if (*s == '-') s++;
    return s[0] == '0' && (s[1] == '\0' || s[1] == '/');
}

/* Compare two decimal digit strings (no sign, no leading zeros). */
static inline int eshkol_numsyn_cmp(const char* a, size_t an, const char* b, size_t bn) {
    if (an != bn) return an < bn ? -1 : 1;
    return memcmp(a, b, an);
}

/* Make an exact part (ZERO, INTEGER, RATIONAL) the DECIMAL spelling of its
 * correctly rounded double. An integer's own digits already are one. A
 * rational n/d is expanded by long division to ESHKOL_NUMSYN_INEXACT_DIGITS
 * significant digits, and a final '1' is appended when the division is not
 * exact: the spelling then lies strictly between the truncated quotient and
 * the next decimal step, which no double rounding boundary (a dyadic
 * rational of at most 767 significant digits) separates from the true
 * quotient, so strtod rounds it exactly as it would round n/d. */
#define ESHKOL_NUMSYN_INEXACT_DIGITS 800
static inline eshkol_numsyn_status_t eshkol_numsyn_make_inexact(eshkol_numsyn_part_t* p) {
    const char* slash;
    const char* num;
    size_t nn, dn, rn = 0, k, sig = 0, point = 0, qn = 0;
    int negative = 0, started = 0;
    char *rem, *q, *text;
    size_t cap;

    if (p->kind == ESHKOL_NUMSYN_DECIMAL || p->kind == ESHKOL_NUMSYN_INFNAN) return ESHKOL_NUMSYN_OK;
    if (p->kind == ESHKOL_NUMSYN_ZERO) {
        p->text = eshkol_numsyn_strndup("0.0", 3);
        p->kind = ESHKOL_NUMSYN_DECIMAL;
        return p->text ? ESHKOL_NUMSYN_OK : ESHKOL_NUMSYN_NO_MEMORY;
    }
    if (p->kind == ESHKOL_NUMSYN_INTEGER) { p->kind = ESHKOL_NUMSYN_DECIMAL; return ESHKOL_NUMSYN_OK; }

    num = p->text;
    if (*num == '-') { negative = 1; num++; }
    slash = strchr(num, '/');
    nn = (size_t)(slash - num);
    dn = strlen(slash + 1);
    cap = nn + ESHKOL_NUMSYN_INEXACT_DIGITS + 8;
    rem = (char*)malloc(dn + 2);
    q = (char*)malloc(cap);
    if (!rem || !q) { free(rem); free(q); return ESHKOL_NUMSYN_NO_MEMORY; }
    /* Bring down the numerator's digits, then zeros, one quotient digit each. */
    for (k = 0; sig < ESHKOL_NUMSYN_INEXACT_DIGITS && qn + 2 < cap; ++k) {
        int digit = 0;
        char c = k < nn ? num[k] : '0';
        if (k == nn) point = qn;
        if (k >= nn && rn == 0) break;               /* exact quotient */
        if (!(rn == 0 && c == '0')) rem[rn++] = c;   /* rem = rem*10 + c, no leading zero */
        while (eshkol_numsyn_cmp(rem, rn, slash + 1, dn) >= 0) {
            /* rem -= den */
            size_t i2;
            int borrow = 0;
            for (i2 = 0; i2 < rn; ++i2) {
                int a = rem[rn - 1 - i2] - '0' - borrow;
                int b = i2 < dn ? slash[dn - i2] - '0' : 0;
                borrow = a < b;
                rem[rn - 1 - i2] = (char)('0' + (a - b + (borrow ? 10 : 0)));
            }
            while (rn > 0 && rem[0] == '0') { memmove(rem, rem + 1, rn - 1); rn--; }
            digit++;
        }
        q[qn++] = (char)('0' + digit);
        if (digit || started) { started = 1; sig++; }
    }
    if (k <= nn) point = qn < nn ? qn : nn;
    if (point > qn) point = qn;
    text = (char*)malloc(qn + 8);
    if (!text) { free(rem); free(q); return ESHKOL_NUMSYN_NO_MEMORY; }
    {
        size_t w = 0;
        size_t lead = 0;
        while (lead + 1 < point && q[lead] == '0') lead++;   /* no leading zeros */
        if (negative) text[w++] = '-';
        if (point == 0) text[w++] = '0';
        memcpy(text + w, q + lead, point - lead);
        w += point - lead;
        text[w++] = '.';
        memcpy(text + w, q + point, qn - point);
        w += qn - point;
        if (rn != 0) text[w++] = '1';                /* sticky: inexact quotient */
        else if (qn == point) text[w++] = '0';
        text[w] = '\0';
    }
    free(rem);
    free(q);
    free(p->text);
    p->text = text;
    p->kind = ESHKOL_NUMSYN_DECIMAL;
    return ESHKOL_NUMSYN_OK;
}

/**
 * @brief Recognize an R7RS <number> and split it into canonical parts.
 *
 * @param text          Token bytes (need not be NUL-terminated).
 * @param len           Token length.
 * @param default_radix Radix when the token has no radix prefix (10, or the
 *                      `string->number` radix argument: 2, 8, 10 or 16).
 * @param out           Result; release with eshkol_number_syntax_free() when
 *                      the status is ESHKOL_NUMSYN_OK (it is left empty
 *                      otherwise).
 * @return ESHKOL_NUMSYN_OK, NOT_A_NUMBER for a token that is not number
 *         syntax (read it as an identifier), or a refusal for number syntax
 *         that has no value (see eshkol_numsyn_status_t).
 */
static inline eshkol_numsyn_status_t eshkol_number_syntax_parse(const char* text, size_t len,
                                                                int default_radix,
                                                                eshkol_numsyn_t* out) {
    int radix = default_radix;
    int exactness = 0;            /* 1 = #e, -1 = #i */
    int seen_radix = 0, seen_exact = 0;
    size_t i = 0;
    const char* body;
    size_t n;
    eshkol_numsyn_status_t st = ESHKOL_NUMSYN_OK;
    int p;

    memset(out, 0, sizeof(*out));
    if (!text || len == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;
    if (radix != 2 && radix != 8 && radix != 10 && radix != 16) return ESHKOL_NUMSYN_NOT_A_NUMBER;

    while (i + 1 < len && text[i] == '#') {
        int c = eshkol_numsyn_lower((unsigned char)text[i + 1]);
        if (!seen_radix && (c == 'b' || c == 'o' || c == 'd' || c == 'x')) {
            radix = c == 'b' ? 2 : c == 'o' ? 8 : c == 'd' ? 10 : 16;
            seen_radix = 1;
        } else if (!seen_exact && (c == 'e' || c == 'i')) {
            exactness = c == 'e' ? 1 : -1;
            seen_exact = 1;
        } else {
            return ESHKOL_NUMSYN_NOT_A_NUMBER;
        }
        i += 2;
    }
    body = text + i;
    n = len - i;
    if (n == 0) return ESHKOL_NUMSYN_NOT_A_NUMBER;

    if (eshkol_numsyn_lower((unsigned char)body[n - 1]) == 'i' && !eshkol_numsyn_is_infnan(body, n)) {
        /* Rectangular: <real>? (+|-) <ureal>? i. The split is the sign that
         * starts the imaginary part; try every sign from the right and take
         * the first split whose two halves are both well formed (an exponent
         * sign in the real part fails its half). */
        size_t m = n - 1;
        size_t k;
        int found = 0;
        for (k = m; k-- > 0;) {
            if (body[k] != '+' && body[k] != '-') continue;
            if (k == 0) {
                /* Pure imaginary: the whole thing is the signed imaginary part. */
                st = eshkol_numsyn_real(body, m, radix, 1, 1, &out->part[1]);
                if (st == ESHKOL_NUMSYN_NOT_A_NUMBER) break;
                out->part[0].kind = ESHKOL_NUMSYN_ZERO;
                found = 1;
                break;
            }
            st = eshkol_numsyn_real(body + k, m - k, radix, 1, 1, &out->part[1]);
            if (st == ESHKOL_NUMSYN_NOT_A_NUMBER) continue;
            if (st != ESHKOL_NUMSYN_OK) { found = 1; break; }
            st = eshkol_numsyn_real(body, k, radix, 0, 0, &out->part[0]);
            if (st == ESHKOL_NUMSYN_NOT_A_NUMBER) {
                eshkol_number_syntax_free(out);
                memset(out, 0, sizeof(*out));
                continue;
            }
            found = 1;
            break;
        }
        if (!found) { eshkol_number_syntax_free(out); memset(out, 0, sizeof(*out)); return ESHKOL_NUMSYN_NOT_A_NUMBER; }
        if (st != ESHKOL_NUMSYN_OK) { eshkol_number_syntax_free(out); memset(out, 0, sizeof(*out)); return st; }
        out->form = ESHKOL_NUMSYN_RECTANGULAR;
    } else {
        const char* at = (const char*)memchr(body, '@', n);
        if (at) {
            size_t k = (size_t)(at - body);
            st = eshkol_numsyn_real(body, k, radix, 0, 0, &out->part[0]);
            if (st == ESHKOL_NUMSYN_OK)
                st = eshkol_numsyn_real(at + 1, n - k - 1, radix, 0, 0, &out->part[1]);
            out->form = ESHKOL_NUMSYN_POLAR;
        } else {
            st = eshkol_numsyn_real(body, n, radix, 0, 0, &out->part[0]);
            out->form = ESHKOL_NUMSYN_REAL;
        }
        if (st != ESHKOL_NUMSYN_OK) { eshkol_number_syntax_free(out); memset(out, 0, sizeof(*out)); return st; }
    }

    /* #e makes every part exact. */
    if (exactness > 0) {
        for (p = 0; p < (out->form == ESHKOL_NUMSYN_REAL ? 1 : 2); ++p) {
            st = eshkol_numsyn_make_exact(&out->part[p]);
            if (st != ESHKOL_NUMSYN_OK) { eshkol_number_syntax_free(out); memset(out, 0, sizeof(*out)); return st; }
        }
    }

    /* A zero imaginary part / zero angle that is exact makes the real number
     * (not under #i, which makes the zero inexact). */
    if (out->form != ESHKOL_NUMSYN_REAL && exactness >= 0 &&
        eshkol_numsyn_part_is_exact_zero(&out->part[1])) {
        free(out->part[1].text);
        memset(&out->part[1], 0, sizeof(out->part[1]));
        if (out->part[0].kind == ESHKOL_NUMSYN_ZERO) {
            out->part[0].kind = ESHKOL_NUMSYN_INTEGER;
            out->part[0].text = eshkol_numsyn_strndup("0", 1);
            if (!out->part[0].text) return ESHKOL_NUMSYN_NO_MEMORY;
        }
        out->form = ESHKOL_NUMSYN_REAL;
    }
    if (exactness > 0 && out->form != ESHKOL_NUMSYN_REAL) {
        eshkol_number_syntax_free(out);
        memset(out, 0, sizeof(*out));
        return ESHKOL_NUMSYN_NO_EXACT_FORM;
    }

    /* #i makes every part inexact, and a complex number's parts are inexact. */
    if (exactness < 0 || out->form != ESHKOL_NUMSYN_REAL) {
        for (p = 0; p < (out->form == ESHKOL_NUMSYN_REAL ? 1 : 2); ++p) {
            st = eshkol_numsyn_make_inexact(&out->part[p]);
            if (st != ESHKOL_NUMSYN_OK) { eshkol_number_syntax_free(out); memset(out, 0, sizeof(*out)); return st; }
        }
    }
    return ESHKOL_NUMSYN_OK;
}

/** True when @p text is number syntax (a valid number or a refused one). */
static inline int eshkol_number_syntax_is_number(const char* text, size_t len, int radix) {
    eshkol_numsyn_t n;
    eshkol_numsyn_status_t st = eshkol_number_syntax_parse(text, len, radix, &n);
    if (st == ESHKOL_NUMSYN_OK) eshkol_number_syntax_free(&n);
    return st != ESHKOL_NUMSYN_NOT_A_NUMBER;
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_CORE_NUMBER_SYNTAX_H */
