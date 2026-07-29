/*
 * eshkol_fixed_point.h — Parametric fixed-point + 128-bit integer + exact
 *                        accumulation primitives for the Eshkol runtime.
 *
 * This module is the numerical-precision engine for the compression stack:
 *   - i128            : fixed-width 128-bit signed integer (wrap/saturate defined,
 *                       distinct from Eshkol's arbitrary-precision bignum tower).
 *   - fixed<W,F>      : parametric fixed-point (W total bits in {32,64,128},
 *                       F fractional bits) with EXPLICIT per-op rounding mode.
 *   - dot_exact       : block-scaled integer dot product with an i128 accumulator
 *                       — bit-exact and ORDER-INDEPENDENT (the determinism payoff).
 *   - accum128        : running-sum type for layer reductions.
 *
 * It is deliberately ADDITIVE: a self-contained C ABI that depends only on the
 * C standard library. Native `__int128` is used as the arithmetic engine (LLVM
 * lowers it to hardware i128 on arm64/x86-64, or paired i64 elsewhere — the same
 * `__int128` the Eshkol bignum/rational core already relies on).
 *
 * FFI marshalling: `esk_i128` is exposed to foreign callers as a two-u64 struct
 * `esk_i128_abi` (little-endian limb order: lo then hi). See esk_i128_to_abi /
 * esk_i128_from_abi.
 *
 * Copyright (c) Tsotchke Corporation. MIT License (see repository LICENSE).
 */
#ifndef ESHKOL_FIXED_POINT_H
#define ESHKOL_FIXED_POINT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if !defined(__SIZEOF_INT128__)
#  error "eshkol_fixed_point requires a compiler with __int128 support. " \
         "All Eshkol LLVM targets (arm64, x86-64) provide it; the same type " \
         "backs the bignum/rational core. Portable soft-i128 fallback is "    \
         "documented in docs/FIXED_POINT_PLAN_20260717.md but not yet built."
#endif

/* ------------------------------------------------------------------------- */
/* 128-bit integer engine                                                    */
/* ------------------------------------------------------------------------- */

typedef __int128            esk_i128;   /* signed   128-bit */
typedef unsigned __int128   esk_u128;   /* unsigned 128-bit */

/* Two-u64 marshalling struct — the stable FFI representation of an i128.
 * Limb order is little-endian: `lo` holds bits [0..63], `hi` holds bits
 * [64..127]. This is the layout an Eshkol FFI declaration binds to. */
typedef struct {
    uint64_t lo;
    uint64_t hi;
} esk_i128_abi;

/* Construction */
static inline esk_i128 esk_i128_from_i64(int64_t v)  { return (esk_i128)v; }
static inline esk_i128 esk_i128_from_u64(uint64_t v) { return (esk_i128)(esk_u128)v; }
esk_i128 esk_i128_from_parts(uint64_t hi, uint64_t lo);

/* FFI marshalling: struct-of-two-u64 <-> native i128 */
esk_i128_abi esk_i128_to_abi(esk_i128 v);
esk_i128     esk_i128_from_abi(esk_i128_abi a);

/* Arithmetic (wrapping two's-complement — matches hardware i128) */
static inline esk_i128 esk_i128_add(esk_i128 a, esk_i128 b) { return a + b; }
static inline esk_i128 esk_i128_sub(esk_i128 a, esk_i128 b) { return a - b; }
static inline esk_i128 esk_i128_mul(esk_i128 a, esk_i128 b) { return a * b; }
static inline esk_i128 esk_i128_neg(esk_i128 a)             { return -a; }

/* Shifts: shl / arithmetic (sign-preserving) shr / logical shr. 0 <= s < 128. */
static inline esk_i128 esk_i128_shl(esk_i128 a, unsigned s) { return (esk_i128)((esk_u128)a << s); }
static inline esk_i128 esk_i128_asr(esk_i128 a, unsigned s) { return a >> s; }         /* arithmetic */
static inline esk_i128 esk_i128_lsr(esk_i128 a, unsigned s) { return (esk_i128)((esk_u128)a >> s); }

/* Compare: returns -1, 0, +1 */
int esk_i128_cmp(esk_i128 a, esk_i128 b);
static inline bool esk_i128_eq(esk_i128 a, esk_i128 b) { return a == b; }

/* Overflow-checked arithmetic. Returns true on overflow (result still holds the
 * wrapped two's-complement value). */
bool esk_i128_add_overflow(esk_i128 a, esk_i128 b, esk_i128 *out);
bool esk_i128_sub_overflow(esk_i128 a, esk_i128 b, esk_i128 *out);
bool esk_i128_mul_overflow(esk_i128 a, esk_i128 b, esk_i128 *out);

/* Widening multiply: exact 128-bit product of two 64-bit integers. */
esk_i128 esk_i128_widen_mul_i64(int64_t a, int64_t b);

/* Limits */
extern const esk_i128 ESK_I128_MAX;   /*  2^127 - 1 */
extern const esk_i128 ESK_I128_MIN;   /* -2^127     */

/* Decimal printing / parsing (for `display` and literal parsing).
 * esk_i128_to_string writes a NUL-terminated base-10 string into buf (needs >=41
 * bytes: sign + 39 digits + NUL); returns the number of chars written (excl NUL),
 * or -1 if buf too small. */
int  esk_i128_to_string(esk_i128 v, char *buf, size_t buflen);
/* Parse a base-10 signed integer. Returns true on success; sets *overflow=true if
 * the literal does not fit in i128 (value is then clamped/undefined). */
bool esk_i128_from_string(const char *s, esk_i128 *out, bool *overflow);

/* ------------------------------------------------------------------------- */
/* Parametric fixed-point: fixed<W,F>                                        */
/* ------------------------------------------------------------------------- */

/* Rounding mode is part of the OPERATION, never global state. */
typedef enum {
    ESK_ROUND_TRUNCATE      = 0,  /* truncate the fractional bits toward zero */
    ESK_ROUND_NEAREST_EVEN  = 1,  /* round to nearest, ties to even           */
    ESK_ROUND_STOCHASTIC    = 2   /* round up with probability == frac. part  */
} esk_round_mode;

/* Overflow policy for add/sub/mul results that leave the W-bit range. */
typedef enum {
    ESK_OF_WRAP     = 0,  /* two's-complement wrap into the W-bit range       */
    ESK_OF_SATURATE = 1   /* clamp to [min, max] of the W-bit range           */
} esk_overflow_mode;

/* A fixed<W,F> value. `raw` is the stored integer; the real value is raw / 2^F.
 * W and F travel with the value so a single ABI carries any instantiation.
 * Invariants: W in {32,64,128}, 0 <= F < W, raw fits the signed W-bit range. */
typedef struct {
    esk_i128 raw;
    uint8_t  W;
    uint8_t  F;
} esk_fixed;

/* Reproducible per-op RNG stream for stochastic rounding (SplitMix64). */
typedef struct { uint64_t state; } esk_rng;
static inline void esk_rng_seed(esk_rng *r, uint64_t seed) { r->state = seed; }
uint64_t esk_rng_next_u64(esk_rng *r);

/* Construction / limits */
esk_fixed esk_fixed_make(esk_i128 raw, uint8_t W, uint8_t F);   /* raw is masked/checked */
esk_i128  esk_fixed_wmin(uint8_t W);   /* -2^(W-1)     */
esk_i128  esk_fixed_wmax(uint8_t W);   /*  2^(W-1) - 1 */

/* Exact add / sub. `of` selects wrap vs saturate on W-bit overflow.
 * Operands must share (W,F). Sets *overflow (may be NULL). */
esk_fixed esk_fixed_add(esk_fixed a, esk_fixed b, esk_overflow_mode of, bool *overflow);
esk_fixed esk_fixed_sub(esk_fixed a, esk_fixed b, esk_overflow_mode of, bool *overflow);
esk_fixed esk_fixed_neg(esk_fixed a, esk_overflow_mode of, bool *overflow);

/* Widening multiply then shift-right F with the given rounding mode.
 * Operands must share (W,F); result is fixed<W,F>. The 2W-bit product is formed
 * exactly (256-bit path for W=128), rounded per `mode`, then narrowed to W bits
 * under `of`. `rng` is used only for ESK_ROUND_STOCHASTIC (may be NULL otherwise).
 * Sets *overflow (may be NULL). */
esk_fixed esk_fixed_mul(esk_fixed a, esk_fixed b, esk_round_mode mode,
                        esk_overflow_mode of, esk_rng *rng, bool *overflow);

/* Division: a / b as fixed<W,F>. Rare and comparatively expensive (dividend is
 * shifted left by F into a 2W-bit numerator). Documented cost, not for hot paths.
 * Returns false (result untouched) on divide-by-zero. */
bool esk_fixed_div(esk_fixed a, esk_fixed b, esk_round_mode mode,
                   esk_overflow_mode of, esk_fixed *out, bool *overflow);

/* Conversions with exactness flags. `*exact` (may be NULL) reports whether the
 * conversion lost information. */
esk_fixed esk_fixed_from_i64(int64_t v, uint8_t W, uint8_t F,
                             esk_overflow_mode of, bool *exact);
int64_t   esk_fixed_to_i64(esk_fixed a, bool *exact);          /* truncates frac toward zero */
esk_fixed esk_fixed_from_double(double x, uint8_t W, uint8_t F,
                                esk_round_mode mode, esk_overflow_mode of,
                                esk_rng *rng, bool *exact);
double    esk_fixed_to_double(esk_fixed a, bool *exact);
esk_fixed esk_fixed_from_f32(float x, uint8_t W, uint8_t F,
                             esk_round_mode mode, esk_overflow_mode of,
                             esk_rng *rng, bool *exact);
float     esk_fixed_to_f32(esk_fixed a, bool *exact);

/* Requantize an existing value to a different (W,F). Exactness reported. */
esk_fixed esk_fixed_convert(esk_fixed a, uint8_t W, uint8_t F,
                            esk_round_mode mode, esk_overflow_mode of,
                            esk_rng *rng, bool *exact);

/* Print fixed<W,F> as a decimal string (exact — it is a dyadic rational).
 * Returns chars written (excl NUL) or -1 if buf too small. */
int esk_fixed_to_string(esk_fixed a, char *buf, size_t buflen);

/* ------------------------------------------------------------------------- */
/* Exact accumulation (Phase E2)                                             */
/* ------------------------------------------------------------------------- */

/* Running-sum type. Integer addition is associative + commutative, so the sum
 * is INDEPENDENT of accumulation order — bit-for-bit. Used for LayerNorm /
 * softmax denominators and any exact reduction. */
typedef struct {
    esk_i128 sum;
    uint64_t count;
} esk_accum128;

static inline void esk_accum128_init(esk_accum128 *a) { a->sum = 0; a->count = 0; }
static inline void esk_accum128_add_i64(esk_accum128 *a, int64_t v) { a->sum += (esk_i128)v; a->count++; }
static inline void esk_accum128_add_i128(esk_accum128 *a, esk_i128 v) { a->sum += v; a->count++; }
void  esk_accum128_merge(esk_accum128 *dst, const esk_accum128 *src);  /* combine partials */
static inline esk_i128 esk_accum128_value(const esk_accum128 *a) { return a->sum; }

/* dot_exact — THE kernel. Block-scaled integer dot product with an i128
 * accumulator: computes  scale_a * scale_b * Σ_i a[i]*b[i]  as fixed<128,F>.
 *
 * The Σ a[i]*b[i] step is pure integer accumulation in i128 (each i8*i8,
 * i16*i16, or i32*i32 product is widened before multiplication), so it
 * is EXACT and byte-identical regardless of iteration order — that is the whole
 * point (see the contract test). The final scale is applied once, deterministically.
 *
 * `F` is the fractional-bit count of the fixed<128,F> result. Sets *exact (may be
 * NULL) if the scaled result is representable without rounding. */
esk_fixed esk_dot_exact_i8 (const int8_t  *a, const int8_t  *b, size_t n,
                            double scale_a, double scale_b, uint8_t F, bool *exact);
esk_fixed esk_dot_exact_i16(const int16_t *a, const int16_t *b, size_t n,
                            double scale_a, double scale_b, uint8_t F, bool *exact);

/* The raw integer dot product Σ a[i]*b[i] in i128, with no scaling — the
 * order-independent core that the contract test pins. */
esk_i128 esk_idot_i8 (const int8_t  *a, const int8_t  *b, size_t n);
esk_i128 esk_idot_i16(const int16_t *a, const int16_t *b, size_t n);
esk_i128 esk_idot_i32(const int32_t *a, const int32_t *b, size_t n);

/* Row-major exact integer matmul for portable-reference / FFI callers.
 * Computes A[rows,inner] x B[inner,cols] into row-major `out[rows,cols]`.
 * Every output cell is the stable two-u64 ABI form of an exact i128 sum.
 *
 * Returns false without reading or writing matrix elements when a required
 * pointer is NULL or any row-major size/index product would overflow size_t.
 * Empty output shapes are valid no-ops. When inner==0, A and B may be NULL and
 * every output cell is written as zero. */
bool esk_imatmul_i32(const int32_t *a, const int32_t *b,
                     size_t rows, size_t inner, size_t cols,
                     esk_i128_abi *out);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* ESHKOL_FIXED_POINT_H */
