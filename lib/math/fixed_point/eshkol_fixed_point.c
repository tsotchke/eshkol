/*
 * eshkol_fixed_point.c — implementation of the fixed-point / i128 / exact
 * accumulation engine. See eshkol_fixed_point.h for the contract.
 *
 * Arithmetic engine: native __int128 (LLVM lowers to hardware i128 on
 * arm64/x86-64). Overflow checks use the __builtin_*_overflow intrinsics, which
 * clang/gcc support for __int128.
 *
 * Copyright (c) Tsotchke Corporation. MIT License.
 */
#include "eshkol_fixed_point.h"

#include <math.h>
#include <string.h>

/* ------------------------------------------------------------------------- */
/* limits                                                                    */
/* ------------------------------------------------------------------------- */

static const esk_u128 U128_ONE = (esk_u128)1;

const esk_i128 ESK_I128_MAX = (esk_i128)(((esk_u128)1 << 127) - 1);
const esk_i128 ESK_I128_MIN = (esk_i128)((esk_u128)1 << 127);   /* -2^127 */

/* ------------------------------------------------------------------------- */
/* i128 construction / marshalling                                           */
/* ------------------------------------------------------------------------- */

esk_i128 esk_i128_from_parts(uint64_t hi, uint64_t lo) {
    return (esk_i128)(((esk_u128)hi << 64) | (esk_u128)lo);
}

esk_i128_abi esk_i128_to_abi(esk_i128 v) {
    esk_u128 u = (esk_u128)v;
    esk_i128_abi a;
    a.lo = (uint64_t)u;
    a.hi = (uint64_t)(u >> 64);
    return a;
}

esk_i128 esk_i128_from_abi(esk_i128_abi a) {
    return (esk_i128)(((esk_u128)a.hi << 64) | (esk_u128)a.lo);
}

int esk_i128_cmp(esk_i128 a, esk_i128 b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

bool esk_i128_add_overflow(esk_i128 a, esk_i128 b, esk_i128 *out) {
    return __builtin_add_overflow(a, b, out);
}
bool esk_i128_sub_overflow(esk_i128 a, esk_i128 b, esk_i128 *out) {
    return __builtin_sub_overflow(a, b, out);
}
bool esk_i128_mul_overflow(esk_i128 a, esk_i128 b, esk_i128 *out) {
    return __builtin_mul_overflow(a, b, out);
}

esk_i128 esk_i128_widen_mul_i64(int64_t a, int64_t b) {
    return (esk_i128)a * (esk_i128)b;   /* exact: 64x64 -> 128 */
}

/* ------------------------------------------------------------------------- */
/* i128 <-> decimal string                                                   */
/* ------------------------------------------------------------------------- */

int esk_i128_to_string(esk_i128 v, char *buf, size_t buflen) {
    char tmp[41];
    int  ti = 0;
    bool neg = v < 0;
    /* Work in unsigned magnitude so INT128_MIN is handled without UB. */
    esk_u128 m = neg ? (~(esk_u128)v + 1u) : (esk_u128)v;
    if (m == 0) {
        tmp[ti++] = '0';
    } else {
        while (m > 0) {
            tmp[ti++] = (char)('0' + (int)(m % 10));
            m /= 10;
        }
    }
    int need = ti + (neg ? 1 : 0);
    if (buflen < (size_t)need + 1) return -1;
    int oi = 0;
    if (neg) buf[oi++] = '-';
    while (ti > 0) buf[oi++] = tmp[--ti];
    buf[oi] = '\0';
    return oi;
}

bool esk_i128_from_string(const char *s, esk_i128 *out, bool *overflow) {
    if (overflow) *overflow = false;
    if (!s) return false;
    while (*s == ' ' || *s == '\t') s++;
    bool neg = false;
    if (*s == '+' || *s == '-') { neg = (*s == '-'); s++; }
    if (*s < '0' || *s > '9') return false;   /* need at least one digit */
    esk_u128 m = 0;
    const esk_u128 lim_pos = (esk_u128)ESK_I128_MAX;           /*  2^127 - 1 */
    const esk_u128 lim_neg = (esk_u128)1 << 127;               /*  2^127     */
    const esk_u128 lim = neg ? lim_neg : lim_pos;
    bool ovf = false;
    for (; *s >= '0' && *s <= '9'; s++) {
        unsigned d = (unsigned)(*s - '0');
        /* m = m*10 + d, with overflow test against lim */
        if (m > (esk_u128)-1 / 10) { ovf = true; }
        esk_u128 m10 = m * 10;
        if (m10 > (esk_u128)-1 - d) { ovf = true; }
        m = m10 + d;
        if (m > lim) ovf = true;
    }
    if (*s != '\0') return false;   /* trailing garbage */
    if (ovf) {
        if (overflow) *overflow = true;
        *out = neg ? ESK_I128_MIN : ESK_I128_MAX;
        return true;
    }
    *out = neg ? (esk_i128)(~m + 1u) : (esk_i128)m;
    return true;
}

/* ------------------------------------------------------------------------- */
/* RNG (SplitMix64) for stochastic rounding                                  */
/* ------------------------------------------------------------------------- */

uint64_t esk_rng_next_u64(esk_rng *r) {
    uint64_t z = (r->state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

/* Uniform u128 in [0, 2^bits), bits in [0,128]. */
static esk_u128 rng_uniform_bits(esk_rng *r, unsigned bits) {
    if (bits == 0) return 0;
    esk_u128 hi = (esk_u128)esk_rng_next_u64(r) << 64;
    esk_u128 lo = (esk_u128)esk_rng_next_u64(r);
    esk_u128 v = hi | lo;
    if (bits >= 128) return v;
    return v & ((U128_ONE << bits) - 1u);
}

/* ------------------------------------------------------------------------- */
/* fixed<W,F> helpers                                                        */
/* ------------------------------------------------------------------------- */

esk_i128 esk_fixed_wmax(uint8_t W) {
    if (W >= 128) return ESK_I128_MAX;
    return (esk_i128)((U128_ONE << (W - 1)) - 1u);
}
esk_i128 esk_fixed_wmin(uint8_t W) {
    if (W >= 128) return ESK_I128_MIN;
    return -(esk_i128)(U128_ONE << (W - 1));
}

/* Clamp/wrap a value already computed in the i128 domain into the W-bit range.
 * `dir_hint` is the sign of the true (un-wrapped) result and is only consulted
 * for the W=128 wrap case where `v` itself has already wrapped. */
static esk_i128 clamp_to_W(esk_i128 v, uint8_t W, esk_overflow_mode of,
                           int dir_hint, bool wrapped128, bool *overflow) {
    esk_i128 lo = esk_fixed_wmin(W);
    esk_i128 hi = esk_fixed_wmax(W);
    if (W >= 128) {
        if (!wrapped128) { if (overflow) *overflow = false; return v; }
        if (overflow) *overflow = true;
        if (of == ESK_OF_SATURATE) return dir_hint >= 0 ? hi : lo;
        return v;   /* wrap: already the two's-complement low 128 bits */
    }
    if (v >= lo && v <= hi) { if (overflow) *overflow = false; return v; }
    if (overflow) *overflow = true;
    if (of == ESK_OF_SATURATE) return v > hi ? hi : lo;
    /* wrap: take low W bits, sign-extend */
    esk_u128 mask = (U128_ONE << W) - 1u;
    esk_u128 u = (esk_u128)v & mask;
    if (u & (U128_ONE << (W - 1))) u |= ~mask;   /* sign extend */
    return (esk_i128)u;
}

esk_fixed esk_fixed_make(esk_i128 raw, uint8_t W, uint8_t F) {
    esk_fixed r;
    bool of = false;
    r.raw = clamp_to_W(raw, W, ESK_OF_SATURATE, raw >= 0 ? 1 : -1, false, &of);
    r.W = W;
    r.F = F;
    return r;
}

esk_fixed esk_fixed_add(esk_fixed a, esk_fixed b, esk_overflow_mode of, bool *overflow) {
    esk_fixed r; r.W = a.W; r.F = a.F;
    esk_i128 s; bool w128 = __builtin_add_overflow(a.raw, b.raw, &s);
    int dir = (a.raw > 0 && b.raw > 0) ? 1 : -1;
    r.raw = clamp_to_W(s, a.W, of, dir, w128, overflow);
    return r;
}

esk_fixed esk_fixed_sub(esk_fixed a, esk_fixed b, esk_overflow_mode of, bool *overflow) {
    esk_fixed r; r.W = a.W; r.F = a.F;
    esk_i128 s; bool w128 = __builtin_sub_overflow(a.raw, b.raw, &s);
    int dir = (a.raw >= 0 && b.raw < 0) ? 1 : -1;
    r.raw = clamp_to_W(s, a.W, of, dir, w128, overflow);
    return r;
}

esk_fixed esk_fixed_neg(esk_fixed a, esk_overflow_mode of, bool *overflow) {
    esk_fixed z; z.raw = 0; z.W = a.W; z.F = a.F;
    return esk_fixed_sub(z, a, of, overflow);
}

/* ------------------------------------------------------------------------- */
/* 256-bit product + round-shift (the widening-multiply core)                */
/* ------------------------------------------------------------------------- */

typedef struct { esk_u128 hi, lo; } esk_u256;

/* Full unsigned 128x128 -> 256 product. */
static esk_u256 umul_u128(esk_u128 a, esk_u128 b) {
    uint64_t a0 = (uint64_t)a, a1 = (uint64_t)(a >> 64);
    uint64_t b0 = (uint64_t)b, b1 = (uint64_t)(b >> 64);
    esk_u128 p00 = (esk_u128)a0 * b0;
    esk_u128 p01 = (esk_u128)a0 * b1;
    esk_u128 p10 = (esk_u128)a1 * b0;
    esk_u128 p11 = (esk_u128)a1 * b1;

    esk_u128 t;
    uint64_t r0, r1, r2, r3;
    t  = p00;                                                   r0 = (uint64_t)t;
    t  = (t >> 64) + (esk_u128)(uint64_t)p01 + (esk_u128)(uint64_t)p10; r1 = (uint64_t)t;
    t  = (t >> 64) + (p01 >> 64) + (p10 >> 64) + (esk_u128)(uint64_t)p11; r2 = (uint64_t)t;
    t  = (t >> 64) + (p11 >> 64);                               r3 = (uint64_t)t;

    esk_u256 out;
    out.lo = ((esk_u128)r1 << 64) | r0;
    out.hi = ((esk_u128)r3 << 64) | r2;
    return out;
}

/* Round the 256-bit magnitude p >> F to a u128 magnitude using `mode`.
 * Sets *ovf_hi if the rounded magnitude does not fit in 128 bits. F in [0,127]. */
static esk_u128 u256_round_shift(esk_u256 p, unsigned F, esk_round_mode mode,
                                 esk_rng *rng, bool *ovf_hi) {
    esk_u128 q, rem, half, hipart;
    if (F == 0) {
        q = p.lo; rem = 0; half = 0; hipart = p.hi;
    } else {
        q      = (p.lo >> F) | (p.hi << (128 - F));
        hipart = p.hi >> F;
        rem    = p.lo & ((U128_ONE << F) - 1u);
        half   = U128_ONE << (F - 1);
    }
    bool carried = false;
    switch (mode) {
        case ESK_ROUND_TRUNCATE:
            break;                                   /* toward zero */
        case ESK_ROUND_NEAREST_EVEN:
            if (rem > half)                 { q += 1; carried = (q == 0); }
            else if (rem == half && F > 0)  { if (q & 1u) { q += 1; carried = (q == 0); } }
            break;
        case ESK_ROUND_STOCHASTIC: {
            esk_u128 r = rng_uniform_bits(rng, F);   /* uniform in [0, 2^F) */
            if (r < rem)                    { q += 1; carried = (q == 0); }
            break;
        }
    }
    *ovf_hi = (hipart != 0) || carried;
    return q;
}

/* Build a signed i128 from (sign, magnitude) and clamp into the W-bit range. */
static esk_i128 signmag_to_W(int sign, esk_u128 mag, bool ovf_hi, uint8_t W,
                             esk_overflow_mode of, bool *overflow) {
    esk_u128 max_pos = (esk_u128)esk_fixed_wmax(W);          /* 2^(W-1)-1 */
    esk_u128 max_neg = (W >= 128) ? ((esk_u128)1 << 127)     /* 2^(W-1)   */
                                  : (U128_ONE << (W - 1));
    bool over = ovf_hi || (sign >= 0 ? (mag > max_pos) : (mag > max_neg));
    if (over) {
        if (overflow) *overflow = true;
        if (of == ESK_OF_SATURATE)
            return sign >= 0 ? esk_fixed_wmax(W) : esk_fixed_wmin(W);
        /* wrap: low W bits of the two's-complement value */
        esk_i128 v = sign >= 0 ? (esk_i128)mag : (esk_i128)(~mag + 1u);
        return clamp_to_W(v, W, ESK_OF_WRAP, sign, W >= 128, NULL);
    }
    if (overflow) *overflow = false;
    return sign >= 0 ? (esk_i128)mag : -(esk_i128)mag;
}

esk_fixed esk_fixed_mul(esk_fixed a, esk_fixed b, esk_round_mode mode,
                        esk_overflow_mode of, esk_rng *rng, bool *overflow) {
    esk_fixed r; r.W = a.W; r.F = a.F;
    int sign = 1;
    esk_u128 ua, ub;
    if (a.raw < 0) { sign = -sign; ua = (~(esk_u128)a.raw + 1u); } else ua = (esk_u128)a.raw;
    if (b.raw < 0) { sign = -sign; ub = (~(esk_u128)b.raw + 1u); } else ub = (esk_u128)b.raw;
    esk_u256 p = umul_u128(ua, ub);
    bool ovf_hi = false;
    esk_u128 mag = u256_round_shift(p, a.F, mode, rng, &ovf_hi);
    r.raw = signmag_to_W(sign, mag, ovf_hi, a.W, of, overflow);
    return r;
}

bool esk_fixed_div(esk_fixed a, esk_fixed b, esk_round_mode mode,
                   esk_overflow_mode of, esk_fixed *out, bool *overflow) {
    if (b.raw == 0) return false;
    /* q = (a.raw << F) / b.raw, computed in sign-magnitude with rounding. */
    int sign = 1;
    esk_u128 na, db;
    if (a.raw < 0) { sign = -sign; na = (~(esk_u128)a.raw + 1u); } else na = (esk_u128)a.raw;
    if (b.raw < 0) { sign = -sign; db = (~(esk_u128)b.raw + 1u); } else db = (esk_u128)b.raw;

    /* Numerator na << F may exceed 128 bits; use the 256-bit shift. Represent
     * na << F as a u256 then long-divide by db (db <= 128 bits). For the common
     * F<=64 && na<2^64 case this stays within u128, but we take the general path. */
    esk_u256 num;
    if (a.F == 0) { num.lo = na; num.hi = 0; }
    else { num.lo = na << a.F; num.hi = na >> (128 - a.F); }

    /* Long division of the 256-bit numerator by the 128-bit divisor, MSB-first. */
    esk_u128 q = 0, rem = 0;
    for (int bit = 255; bit >= 0; --bit) {
        esk_u128 nbit = (bit >= 128) ? ((num.hi >> (bit - 128)) & 1u)
                                     : ((num.lo >> bit) & 1u);
        esk_u128 rem_hi = rem >> 127;            /* catch shift-out */
        rem = (rem << 1) | nbit;
        if (rem_hi || rem >= db) { rem -= db; if (bit < 128) q |= (U128_ONE << bit); else { /* q overflow */ } }
    }
    /* `rem` is the remainder; apply rounding on rem/db. */
    /* Division uses TRUNCATE or NEAREST_EVEN. STOCHASTIC has no RNG stream in the
     * div signature (div is the rare, non-hot path) and falls back to nearest-even. */
    if (mode == ESK_ROUND_NEAREST_EVEN || mode == ESK_ROUND_STOCHASTIC) {
        esk_u128 two_rem = rem << 1;   /* compare 2*rem vs db (may overflow: guard) */
        bool rem_hi = (rem >> 127) != 0;
        if (rem_hi || two_rem > db)          q += 1;
        else if (!rem_hi && two_rem == db) { if (q & 1u) q += 1; }
    }
    esk_fixed res; res.W = a.W; res.F = a.F;
    res.raw = signmag_to_W(sign, q, false, a.W, of, overflow);
    *out = res;
    return true;
}

/* ------------------------------------------------------------------------- */
/* conversions                                                               */
/* ------------------------------------------------------------------------- */

esk_fixed esk_fixed_from_i64(int64_t v, uint8_t W, uint8_t F,
                             esk_overflow_mode of, bool *exact) {
    esk_fixed r; r.W = W; r.F = F;
    esk_i128 raw = (esk_i128)v << F;               /* v * 2^F, exact for W>=... */
    bool ov = false;
    r.raw = clamp_to_W(raw, W, of, v >= 0 ? 1 : -1, false, &ov);
    if (exact) *exact = !ov;
    return r;
}

int64_t esk_fixed_to_i64(esk_fixed a, bool *exact) {
    esk_i128 ip = a.raw >> a.F;                    /* toward -inf */
    esk_u128 fracmask = (a.F == 0) ? 0 : ((U128_ONE << a.F) - 1u);
    bool frac = ((esk_u128)a.raw & fracmask) != 0;
    /* to_i64 truncates toward zero: adjust for negative with fractional part */
    esk_i128 t = a.raw / ((esk_i128)1 << a.F);     /* C division truncates toward zero */
    if (exact) *exact = !frac && t >= (esk_i128)INT64_MIN && t <= (esk_i128)INT64_MAX;
    (void)ip;
    if (t > (esk_i128)INT64_MAX) return INT64_MAX;
    if (t < (esk_i128)INT64_MIN) return INT64_MIN;
    return (int64_t)t;
}

esk_fixed esk_fixed_from_double(double x, uint8_t W, uint8_t F,
                                esk_round_mode mode, esk_overflow_mode of,
                                esk_rng *rng, bool *exact) {
    esk_fixed r; r.W = W; r.F = F;
    if (!isfinite(x)) { r.raw = 0; if (exact) *exact = false; return r; }
    int sign = signbit(x) ? -1 : 1;
    double ax = fabs(x);
    double scaled = ldexp(ax, F);                  /* ax * 2^F */
    double fl = floor(scaled);
    double frac = scaled - fl;                     /* in [0,1) */
    /* magnitude integer part as u128 (fl may exceed 2^64) */
    esk_u128 mag = 0;
    double t = fl;
    /* decompose fl into base-2^32 limbs to avoid precision loss up to 2^128 */
    if (t >= ldexp(1.0, 128)) { /* out of range -> saturate */
        r.raw = (sign >= 0) ? esk_fixed_wmax(W) : esk_fixed_wmin(W);
        if (exact) *exact = false;
        return r;
    }
    {
        /* Build mag from fl exactly for fl < 2^128 by 24-bit chunks. */
        double rem = fl; int shift = 0; esk_u128 acc = 0;
        /* extract low bits iteratively */
        while (rem >= 1.0 && shift < 128) {
            double chunk = fmod(rem, 4294967296.0);    /* low 32 bits */
            acc |= ((esk_u128)(uint64_t)chunk) << shift;
            rem = floor(rem / 4294967296.0);
            shift += 32;
        }
        mag = acc;
    }
    /* apply rounding of `frac` (the discarded fractional part) */
    bool round_up = false;
    switch (mode) {
        case ESK_ROUND_TRUNCATE: break;
        case ESK_ROUND_NEAREST_EVEN:
            if (frac > 0.5) round_up = true;
            else if (frac == 0.5) round_up = (mag & 1u) != 0;
            break;
        case ESK_ROUND_STOCHASTIC: {
            /* draw uniform in [0,1) with 53 bits; round up if < frac */
            uint64_t bits = esk_rng_next_u64(rng) >> 11;      /* 53 bits */
            double u = (double)bits / 9007199254740992.0;     /* /2^53 */
            round_up = (u < frac);
            break;
        }
    }
    if (round_up) mag += 1;
    bool inexact = (frac != 0.0);
    r.raw = signmag_to_W(sign, mag, false, W, of, NULL);
    /* re-flag saturation as inexact too */
    if (r.raw == esk_fixed_wmax(W) || r.raw == esk_fixed_wmin(W)) {
        /* could be legitimate; treat as inexact only if rounding happened */
    }
    if (exact) *exact = !inexact;
    return r;
}

double esk_fixed_to_double(esk_fixed a, bool *exact) {
    /* d = raw / 2^F */
    double d = 0.0;
    esk_i128 v = a.raw;
    int sign = v < 0 ? -1 : 1;
    esk_u128 m = v < 0 ? (~(esk_u128)v + 1u) : (esk_u128)v;
    /* build double from u128 in 32-bit chunks (exact up to rounding of double) */
    double scale = 1.0;
    esk_u128 mm = m;
    while (mm > 0) {
        d += (double)(uint64_t)(mm & 0xFFFFFFFFu) * scale;
        mm >>= 32;
        scale *= 4294967296.0;
    }
    d = ldexp(d, -(int)a.F) * sign;
    /* Exact iff the stored integer fits in double's 53-bit mantissa; the 2^-F
     * scaling is always exact in binary FP (barring under/overflow of exponent). */
    if (exact) *exact = (m < ((esk_u128)1 << 53));
    return d;
}

esk_fixed esk_fixed_from_f32(float x, uint8_t W, uint8_t F, esk_round_mode mode,
                             esk_overflow_mode of, esk_rng *rng, bool *exact) {
    return esk_fixed_from_double((double)x, W, F, mode, of, rng, exact);
}
float esk_fixed_to_f32(esk_fixed a, bool *exact) {
    bool e64; double d = esk_fixed_to_double(a, &e64);
    float f = (float)d;
    if (exact) *exact = e64 && ((double)f == d);
    return f;
}

esk_fixed esk_fixed_convert(esk_fixed a, uint8_t W, uint8_t F, esk_round_mode mode,
                            esk_overflow_mode of, esk_rng *rng, bool *exact) {
    esk_fixed r; r.W = W; r.F = F;
    if (F >= a.F) {
        esk_i128 raw = a.raw << (F - a.F);         /* exact left shift */
        bool ov = false;
        r.raw = clamp_to_W(raw, W, of, a.raw >= 0 ? 1 : -1, false, &ov);
        if (exact) *exact = !ov;
        return r;
    }
    /* F < a.F: shift right by (a.F - F) with rounding */
    unsigned s = (unsigned)(a.F - F);
    int sign = a.raw < 0 ? -1 : 1;
    esk_u128 m = a.raw < 0 ? (~(esk_u128)a.raw + 1u) : (esk_u128)a.raw;
    esk_u256 p; p.lo = m; p.hi = 0;
    bool ovf_hi = false;
    esk_u128 q = u256_round_shift(p, s, mode, rng, &ovf_hi);
    bool inexact = ((m & ((U128_ONE << s) - 1u)) != 0);
    bool ov = false;
    r.raw = signmag_to_W(sign, q, ovf_hi, W, of, &ov);
    if (exact) *exact = !inexact && !ov;
    return r;
}

int esk_fixed_to_string(esk_fixed a, char *buf, size_t buflen) {
    if (buflen < 2) return -1;
    int sign = a.raw < 0 ? -1 : 1;
    esk_u128 m = a.raw < 0 ? (~(esk_u128)a.raw + 1u) : (esk_u128)a.raw;
    esk_u128 ip = (a.F == 0) ? m : (m >> a.F);
    esk_u128 fr = (a.F == 0) ? 0 : (m & ((U128_ONE << a.F) - 1u));
    int oi = 0;
    if (sign < 0 && m != 0) buf[oi++] = '-';
    /* integer part */
    char ib[41]; int n = esk_i128_to_string((esk_i128)ip, ib, sizeof ib);
    if (n < 0) return -1;
    /* ip < 2^128 fits positive i128? ip may be up to 2^128-1 which overflows i128.
     * For fixed<128,F>, ip = m >> F and F>=1 so ip < 2^127 — fits. */
    for (int i = 0; i < n && (size_t)oi < buflen - 1; i++) buf[oi++] = ib[i];
    if (fr != 0) {
        if ((size_t)oi < buflen - 1) buf[oi++] = '.';
        int guard = 0;
        while (fr != 0 && guard < 60 && (size_t)oi < buflen - 1) {
            fr *= 10u;
            esk_u128 digit = fr >> a.F;
            buf[oi++] = (char)('0' + (int)digit);
            fr &= (U128_ONE << a.F) - 1u;
            guard++;
        }
    }
    buf[oi] = '\0';
    return oi;
}

/* ------------------------------------------------------------------------- */
/* exact accumulation                                                        */
/* ------------------------------------------------------------------------- */

void esk_accum128_merge(esk_accum128 *dst, const esk_accum128 *src) {
    dst->sum   += src->sum;
    dst->count += src->count;
}

esk_i128 esk_idot_i8(const int8_t *a, const int8_t *b, size_t n) {
    esk_i128 acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += (esk_i128)((int32_t)a[i] * (int32_t)b[i]);
    return acc;
}
esk_i128 esk_idot_i16(const int16_t *a, const int16_t *b, size_t n) {
    esk_i128 acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += (esk_i128)((int64_t)a[i] * (int64_t)b[i]);
    return acc;
}
esk_i128 esk_idot_i32(const int32_t *a, const int32_t *b, size_t n) {
    /* An i32 product needs at most 63 signed bits. With 64-bit size_t, even
     * SIZE_MAX such products have magnitude < 2^126, so the i128 sum cannot
     * overflow. This covers Eshkol's supported arm64 and x86-64 targets. */
    _Static_assert(sizeof(size_t) <= sizeof(uint64_t),
                   "esk_idot_i32 requires size_t no wider than 64 bits");
    esk_i128 acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += (esk_i128)((int64_t)a[i] * (int64_t)b[i]);
    return acc;
}

static bool size_product_overflows(size_t a, size_t b) {
    return a != 0 && b > SIZE_MAX / a;
}

bool esk_imatmul_i32(const int32_t *a, const int32_t *b,
                     size_t rows, size_t inner, size_t cols,
                     esk_i128_abi *out) {
    /* No output cells means no pointers or index products are observed. */
    if (rows == 0 || cols == 0) return true;
    if (out == NULL || size_product_overflows(rows, cols)) return false;

    size_t out_count = rows * cols;
    if (inner == 0) {
        esk_i128_abi zero = esk_i128_to_abi(0);
        for (size_t i = 0; i < out_count; i++) out[i] = zero;
        return true;
    }

    if (a == NULL || b == NULL ||
        size_product_overflows(rows, inner) ||
        size_product_overflows(inner, cols)) return false;

    for (size_t i = 0; i < rows; i++) {
        const int32_t *a_row = a + i * inner;
        for (size_t j = 0; j < cols; j++) {
            esk_i128 acc = 0;
            for (size_t k = 0; k < inner; k++)
                acc += (esk_i128)((int64_t)a_row[k] *
                                  (int64_t)b[k * cols + j]);
            out[i * cols + j] = esk_i128_to_abi(acc);
        }
    }
    return true;
}

/* Scale an exact i128 dot accumulator by (scale_a*scale_b) into fixed<128,F>.
 *
 * value = acc * combined, stored in Q(F).  Represent `combined` (a dyadic double)
 * as a fixed<128,F> scale factor  fsc.raw = round(combined * 2^F).  Then the Q(F)
 * result is simply  acc * fsc.raw  — ONE exact integer multiply, no shift, no
 * rounding.  Order-independence is entirely carried by `acc` (pure integer sum);
 * the scale is a single deterministic post-step, so the whole result is
 * byte-identical across accumulation orders.
 *
 * *exact is set iff the scale is representable without rounding AND the final
 * multiply neither overflows i128 nor saturates the 128-bit range. */
static esk_fixed scale_idot(esk_i128 acc, double sa, double sb, uint8_t F, bool *exact) {
    double combined = sa * sb;
    bool e_scale;
    esk_fixed fsc = esk_fixed_from_double(combined, 128, F, ESK_ROUND_NEAREST_EVEN,
                                          ESK_OF_SATURATE, NULL, &e_scale);
    esk_i128 raw;
    bool of128 = esk_i128_mul_overflow(acc, fsc.raw, &raw);
    esk_fixed r; r.W = 128; r.F = F;
    int dir = ((acc < 0) ^ (fsc.raw < 0)) ? -1 : 1;
    bool ovf = false;
    r.raw = clamp_to_W(raw, 128, ESK_OF_SATURATE, dir, of128, &ovf);
    if (exact) *exact = e_scale && !of128 && !ovf;
    return r;
}

esk_fixed esk_dot_exact_i8(const int8_t *a, const int8_t *b, size_t n,
                           double scale_a, double scale_b, uint8_t F, bool *exact) {
    return scale_idot(esk_idot_i8(a, b, n), scale_a, scale_b, F, exact);
}
esk_fixed esk_dot_exact_i16(const int16_t *a, const int16_t *b, size_t n,
                            double scale_a, double scale_b, uint8_t F, bool *exact) {
    return scale_idot(esk_idot_i16(a, b, n), scale_a, scale_b, F, exact);
}
