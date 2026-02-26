/*
 * Metal SoftFloat - IEEE 754 Double-Precision Emulation for Metal
 *
 * Implements full f64 arithmetic using uint2 (two 32-bit integers)
 * Based on Berkeley SoftFloat library algorithms
 *
 * SINGLE SOURCE OF TRUTH for Metal SoftFloat f64 emulation.
 * At build time, CMake auto-generates metal_sf64_embedded.inc from this file,
 * which gpu_memory.mm includes as an NSString for runtime Metal compilation.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_METAL_SOFTFLOAT_H
#define ESHKOL_METAL_SOFTFLOAT_H

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// IEEE 754 Double-Precision Format
// ============================================================================
// Sign:     1 bit  (bit 63)
// Exponent: 11 bits (bits 62-52), bias = 1023
// Mantissa: 52 bits (bits 51-0), implicit leading 1 for normalized numbers

typedef uint2 sf64;  // .x = high 32 bits, .y = low 32 bits

// ============================================================================
// Constants
// ============================================================================

constant sf64 SF64_ZERO = sf64(0x00000000u, 0x00000000u);
constant sf64 SF64_NEG_ZERO = sf64(0x80000000u, 0x00000000u);
constant sf64 SF64_ONE = sf64(0x3FF00000u, 0x00000000u);
constant sf64 SF64_NEG_ONE = sf64(0xBFF00000u, 0x00000000u);
constant sf64 SF64_TWO = sf64(0x40000000u, 0x00000000u);
constant sf64 SF64_INF = sf64(0x7FF00000u, 0x00000000u);
constant sf64 SF64_NEG_INF = sf64(0xFFF00000u, 0x00000000u);
constant sf64 SF64_QNAN = sf64(0x7FF80000u, 0x00000000u);  // Quiet NaN

constant uint SF64_SIGN_MASK = 0x80000000u;
constant uint SF64_EXP_MASK = 0x7FF00000u;
constant uint SF64_MANT_HI_MASK = 0x000FFFFFu;
constant int SF64_EXP_BIAS = 1023;
constant int SF64_EXP_MAX = 2047;

// ============================================================================
// Bit Extraction
// ============================================================================

inline bool sf64_sign(sf64 x) {
    return (x.x >> 31) != 0;
}

inline int sf64_exp_raw(sf64 x) {
    return int((x.x >> 20) & 0x7FFu);
}

inline int sf64_exponent(sf64 x) {
    return sf64_exp_raw(x) - SF64_EXP_BIAS;
}

inline sf64 sf64_sig(sf64 x) {
    return sf64(x.x & SF64_MANT_HI_MASK, x.y);
}

// ============================================================================
// Classification
// ============================================================================

inline bool sf64_is_zero(sf64 x) {
    return ((x.x & 0x7FFFFFFFu) == 0) && (x.y == 0);
}

inline bool sf64_is_subnormal(sf64 x) {
    return (sf64_exp_raw(x) == 0) && !sf64_is_zero(x);
}

inline bool sf64_is_inf(sf64 x) {
    return ((x.x & 0x7FFFFFFFu) == SF64_EXP_MASK) && (x.y == 0);
}

inline bool sf64_is_nan(sf64 x) {
    return ((x.x & SF64_EXP_MASK) == SF64_EXP_MASK) &&
           (((x.x & SF64_MANT_HI_MASK) != 0) || (x.y != 0));
}

inline bool sf64_is_signaling_nan(sf64 x) {
    return sf64_is_nan(x) && ((x.x & 0x00080000u) == 0);
}

// ============================================================================
// NaN Propagation
// ============================================================================
// IEEE 754: preserve NaN payload, quiet signaling NaNs

inline sf64 sf64_propagate_nan(sf64 a, sf64 b) {
    // Prefer signaling NaN (to signal the exception), then first NaN operand
    if (sf64_is_signaling_nan(a)) {
        return sf64(a.x | 0x00080000u, a.y);  // Quiet the signaling NaN
    }
    if (sf64_is_signaling_nan(b)) {
        return sf64(b.x | 0x00080000u, b.y);
    }
    if (sf64_is_nan(a)) return a;
    return b;
}

inline sf64 sf64_propagate_nan3(sf64 a, sf64 b, sf64 c) {
    if (sf64_is_signaling_nan(a)) return sf64(a.x | 0x00080000u, a.y);
    if (sf64_is_signaling_nan(b)) return sf64(b.x | 0x00080000u, b.y);
    if (sf64_is_signaling_nan(c)) return sf64(c.x | 0x00080000u, c.y);
    if (sf64_is_nan(a)) return a;
    if (sf64_is_nan(b)) return b;
    return c;
}

// ============================================================================
// Packing
// ============================================================================

inline sf64 sf64_pack(bool sign, int exp_raw, uint mant_hi, uint mant_lo) {
    uint hi = (sign ? SF64_SIGN_MASK : 0u) |
              ((uint(exp_raw) & 0x7FFu) << 20) |
              (mant_hi & SF64_MANT_HI_MASK);
    return sf64(hi, mant_lo);
}

inline sf64 sf64_set_sign(sf64 x, bool sign) {
    uint hi = (x.x & 0x7FFFFFFFu) | (sign ? SF64_SIGN_MASK : 0u);
    return sf64(hi, x.y);
}

inline sf64 sf64_negate(sf64 x) {
    return sf64(x.x ^ SF64_SIGN_MASK, x.y);
}

inline sf64 sf64_abs(sf64 x) {
    return sf64(x.x & 0x7FFFFFFFu, x.y);
}

// ============================================================================
// 64-bit Arithmetic Helpers
// ============================================================================

inline sf64 shl64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) {
        return sf64(x.y << (n - 32), 0u);
    }
    return sf64((x.x << n) | (x.y >> (32 - n)), x.y << n);
}

inline sf64 shr64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) {
        return sf64(0u, x.x >> (n - 32));
    }
    return sf64(x.x >> n, (x.y >> n) | (x.x << (32 - n)));
}

// Right shift with sticky bit preservation (jam)
// Sets LSB if any bits are shifted out, preserving rounding information
inline sf64 shr64_jam(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) {
        uint sticky = ((x.x | x.y) != 0) ? 1u : 0u;
        return sf64(0u, sticky);
    }
    if (n >= 32) {
        uint lost_bits = x.y | ((n > 32) ? (x.x << (64 - n)) : 0u);
        uint sticky = (lost_bits != 0) ? 1u : 0u;
        return sf64(0u, (x.x >> (n - 32)) | sticky);
    }
    uint lost_bits = x.y << (32 - n);
    uint sticky = (lost_bits != 0) ? 1u : 0u;
    return sf64(x.x >> n, ((x.y >> n) | (x.x << (32 - n))) | sticky);
}

inline sf64 add64(sf64 a, sf64 b) {
    uint lo = a.y + b.y;
    uint carry = (lo < a.y) ? 1u : 0u;
    uint hi = a.x + b.x + carry;
    return sf64(hi, lo);
}

inline sf64 add64_carry(sf64 a, sf64 b, thread bool& carry_out) {
    uint lo = a.y + b.y;
    uint c1 = (lo < a.y) ? 1u : 0u;
    uint hi = a.x + b.x + c1;
    carry_out = (hi < a.x) || (c1 != 0 && hi == a.x + c1);
    return sf64(hi, lo);
}

inline sf64 sub64(sf64 a, sf64 b) {
    uint lo = a.y - b.y;
    uint borrow = (a.y < b.y) ? 1u : 0u;
    uint hi = a.x - b.x - borrow;
    return sf64(hi, lo);
}

// Compare magnitudes: returns -1 if a<b, 0 if a==b, 1 if a>b
inline int cmp64(sf64 a, sf64 b) {
    if (a.x != b.x) return (a.x < b.x) ? -1 : 1;
    if (a.y != b.y) return (a.y < b.y) ? -1 : 1;
    return 0;
}

// Count leading zeros in 64-bit value
inline int clz64(sf64 x) {
    if (x.x != 0) return clz(x.x);
    if (x.y != 0) return 32 + clz(x.y);
    return 64;
}

// ============================================================================
// 128-bit Arithmetic (required for true FMA and division)
// ============================================================================

struct sf128 { sf64 hi; sf64 lo; };

inline sf128 shr128(sf128 x, int n) {
    if (n <= 0) return x;
    if (n >= 128) return sf128{SF64_ZERO, SF64_ZERO};
    if (n >= 64) {
        return sf128{SF64_ZERO, shr64(x.hi, n - 64)};
    }
    sf64 new_lo;
    new_lo.x = (x.lo.x >> n) | (x.hi.y << (32 - n));
    new_lo.y = (x.lo.y >> n) | (x.lo.x << (32 - n));
    if (n >= 32) {
        new_lo.x = (x.hi.y >> (n - 32)) | (x.hi.x << (64 - n));
        new_lo.y = (x.lo.x >> (n - 32)) | (x.hi.y << (64 - n));
    }
    return sf128{shr64(x.hi, n), new_lo};
}

inline sf128 shr128_jam(sf128 x, int n) {
    if (n <= 0) return x;
    if (n >= 128) {
        uint sticky = ((x.hi.x | x.hi.y | x.lo.x | x.lo.y) != 0) ? 1u : 0u;
        return sf128{SF64_ZERO, sf64(0u, sticky)};
    }
    uint sticky = 0;
    if (n >= 64) {
        sticky = ((x.lo.x | x.lo.y) != 0) ? 1u : 0u;
        sf64 new_lo = shr64_jam(x.hi, n - 64);
        new_lo.y |= sticky;
        return sf128{SF64_ZERO, new_lo};
    }
    sticky = (x.lo.y << (32 - (n % 32))) != 0 ? 1u : 0u;
    sf128 result = shr128(x, n);
    result.lo.y |= sticky;
    return result;
}

inline sf128 shl128(sf128 x, int n) {
    if (n <= 0) return x;
    if (n >= 128) return sf128{SF64_ZERO, SF64_ZERO};
    if (n >= 64) {
        return sf128{shl64(x.lo, n - 64), SF64_ZERO};
    }
    sf64 new_hi;
    new_hi.x = (x.hi.x << n) | (x.hi.y >> (32 - n));
    new_hi.y = (x.hi.y << n) | (x.lo.x >> (32 - n));
    if (n >= 32) {
        new_hi.x = (x.hi.y << (n - 32)) | (x.lo.x >> (64 - n));
        new_hi.y = (x.lo.x << (n - 32)) | (x.lo.y >> (64 - n));
    }
    return sf128{new_hi, shl64(x.lo, n)};
}

inline sf128 add128(sf128 a, sf128 b) {
    sf64 lo = add64(a.lo, b.lo);
    uint carry = (cmp64(lo, a.lo) < 0 || cmp64(lo, b.lo) < 0) ? 1u : 0u;
    sf64 hi = add64(a.hi, b.hi);
    hi = add64(hi, sf64(0u, carry));
    return sf128{hi, lo};
}

inline sf128 sub128(sf128 a, sf128 b) {
    uint borrow = (cmp64(a.lo, b.lo) < 0) ? 1u : 0u;
    sf64 lo = sub64(a.lo, b.lo);
    sf64 hi = sub64(a.hi, b.hi);
    hi = sub64(hi, sf64(0u, borrow));
    return sf128{hi, lo};
}

inline int cmp128(sf128 a, sf128 b) {
    int cmp_hi = cmp64(a.hi, b.hi);
    if (cmp_hi != 0) return cmp_hi;
    return cmp64(a.lo, b.lo);
}

inline int clz128(sf128 x) {
    int clz_hi = clz64(x.hi);
    if (clz_hi < 64) return clz_hi;
    return 64 + clz64(x.lo);
}

inline bool is_zero128(sf128 x) {
    return (x.hi.x | x.hi.y | x.lo.x | x.lo.y) == 0;
}

// ============================================================================
// 128-bit Multiplication Support
// ============================================================================

struct uint128_t {
    uint w3, w2, w1, w0;  // w3 is MSW, w0 is LSW
};

// 64x64 -> 128 bit multiplication
// Uses 16-bit pieces to avoid overflow
inline uint128_t mul64x64(sf64 a, sf64 b) {
    uint a3 = a.x >> 16, a2 = a.x & 0xFFFFu;
    uint a1 = a.y >> 16, a0 = a.y & 0xFFFFu;
    uint b3 = b.x >> 16, b2 = b.x & 0xFFFFu;
    uint b1 = b.y >> 16, b0 = b.y & 0xFFFFu;

    uint p00 = a0*b0, p01 = a0*b1, p02 = a0*b2, p03 = a0*b3;
    uint p10 = a1*b0, p11 = a1*b1, p12 = a1*b2, p13 = a1*b3;
    uint p20 = a2*b0, p21 = a2*b1, p22 = a2*b2, p23 = a2*b3;
    uint p30 = a3*b0, p31 = a3*b1, p32 = a3*b2, p33 = a3*b3;

    uint c0 = p00 & 0xFFFFu;
    uint carry = p00 >> 16;
    uint c1 = p01 + p10 + carry; carry = c1 >> 16; c1 &= 0xFFFFu;
    uint c2 = p02 + p11 + p20 + carry; carry = c2 >> 16; c2 &= 0xFFFFu;
    uint c3 = p03 + p12 + p21 + p30 + carry; carry = c3 >> 16; c3 &= 0xFFFFu;
    uint c4 = p13 + p22 + p31 + carry; carry = c4 >> 16; c4 &= 0xFFFFu;
    uint c5 = p23 + p32 + carry; carry = c5 >> 16; c5 &= 0xFFFFu;
    uint c6 = p33 + carry; uint c7 = c6 >> 16; c6 &= 0xFFFFu;

    uint128_t r;
    r.w0 = (c1 << 16) | c0;
    r.w1 = (c3 << 16) | c2;
    r.w2 = (c5 << 16) | c4;
    r.w3 = (c7 << 16) | c6;
    return r;
}

// ============================================================================
// Normalization Helpers
// ============================================================================

inline int sf64_normalize_subnormal(thread sf64& sig) {
    int shift = clz64(sig) - 11;  // Target: leading 1 at bit 52
    if (shift > 0) {
        sig = shl64(sig, shift);
    }
    return shift;
}

// ============================================================================
// Rounding (Round-to-Nearest-Even)
// ============================================================================
// Synced to gpu_memory.mm canonical implementation.
// round_bits: 10 bits of rounding info, half-way point at 0x200

inline sf64 sf64_round_pack(bool sign, int exp_raw, sf64 sig, uint round_bits) {
    // Round to nearest, ties to even
    // round_bits: 10 bits, half-way point at 0x200
    bool round_up = (round_bits > 0x200u) ||
                    ((round_bits == 0x200u) && ((sig.y & 1u) != 0));

    if (round_up) {
        sig = add64(sig, sf64(0u, 1u));
        // Check if rounding caused mantissa overflow (bit 52 carried into bit 53)
        if ((sig.x & 0x00200000u) != 0) {
            sig = shr64(sig, 1);
            exp_raw++;
        }
    }

    // Check for overflow to infinity
    if (exp_raw >= SF64_EXP_MAX) {
        return sign ? SF64_NEG_INF : SF64_INF;
    }

    // Check for underflow to subnormal/zero
    if (exp_raw <= 0) {
        int shift = 1 - exp_raw;
        if (shift >= 64) {
            return sign ? SF64_NEG_ZERO : SF64_ZERO;
        }
        sig = shr64_jam(sig, shift);
        exp_raw = 0;
    }

    return sf64_pack(sign, exp_raw, sig.x, sig.y);
}

// ============================================================================
// Addition
// ============================================================================
// Uses 11 guard bits (shift left by 11) with 10 round bits and sticky.
// After operation, leading 1 is normalized to bit 62 (bit 30 of .x).
// Extract 10 round bits from low 10 bits, shift right by 10.

sf64 sf64_add(sf64 a, sf64 b) {
    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);
    int expA = sf64_exp_raw(a);
    int expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a);
    sf64 sigB = sf64_sig(b);

    // Handle NaN — propagate payload per IEEE 754
    if (sf64_is_nan(a) || sf64_is_nan(b)) return sf64_propagate_nan(a, b);

    // Handle infinity
    if (sf64_is_inf(a)) {
        if (sf64_is_inf(b) && (signA != signB)) {
            return SF64_QNAN;  // inf - inf = NaN
        }
        return a;
    }
    if (sf64_is_inf(b)) return b;

    // Handle zero
    if (sf64_is_zero(a)) {
        if (sf64_is_zero(b)) {
            return (signA && signB) ? SF64_NEG_ZERO : SF64_ZERO;
        }
        return b;
    }
    if (sf64_is_zero(b)) return a;

    // Add implicit bit for normalized numbers
    if (expA != 0) sigA.x |= 0x00100000u; else expA = 1;
    if (expB != 0) sigB.x |= 0x00100000u; else expB = 1;

    // Shift significands left by 11 bits for guard/round/sticky bits
    // This places the leading 1 at bit 63
    sigA = shl64(sigA, 11);
    sigB = shl64(sigB, 11);

    // Align exponents by shifting the smaller significand right
    int expDiff = expA - expB;
    int expZ;
    if (expDiff > 0) {
        sigB = shr64_jam(sigB, expDiff);
        expZ = expA;
    } else if (expDiff < 0) {
        sigA = shr64_jam(sigA, -expDiff);
        expZ = expB;
    } else {
        expZ = expA;
    }

    sf64 sigZ;
    bool signZ;

    if (signA == signB) {
        // Same sign: add magnitudes
        signZ = signA;
        sigZ = add64(sigA, sigB);

        // Check for 64-bit overflow (carry out) using magnitude comparison
        if (cmp64(sigZ, sigA) < 0 || cmp64(sigZ, sigB) < 0) {
            // Overflow: true sum >= 2^64, virtual leading 1 at bit 64
            // Shift right by 2 to normalize to bit 62, increment exponent
            sigZ = shr64_jam(sigZ, 2);
            sigZ.x |= 0x40000000u;  // Set leading 1 at bit 62
            expZ++;
        } else {
            // No overflow: leading 1 at bit 63
            // Shift right by 1 to normalize to bit 62
            sigZ = shr64_jam(sigZ, 1);
        }
    } else {
        // Different signs: subtract magnitudes
        int cmp = cmp64(sigA, sigB);
        if (cmp == 0) {
            return SF64_ZERO;  // Exact cancellation → positive zero
        }
        if (cmp > 0) {
            signZ = signA;
            sigZ = sub64(sigA, sigB);
        } else {
            signZ = signB;
            sigZ = sub64(sigB, sigA);
        }

        // Normalize: shift left until leading 1 is at bit 62 (not 63)
        int shift = clz64(sigZ) - 1;  // -1 targets bit 62
        if (shift > 0) {
            sigZ = shl64(sigZ, shift);
            expZ -= shift;
        } else if (shift < 0) {
            sigZ = shr64_jam(sigZ, -shift);
            expZ -= shift;
        }
    }

    // Leading 1 is now at bit 62 (bit 30 of sigZ.x)
    // Extract 10 round bits from low 10 bits of sigZ, then shift by 10
    uint round_bits = sigZ.y & 0x3FFu;
    sigZ = shr64(sigZ, 10);

    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ============================================================================
// Subtraction
// ============================================================================

inline sf64 sf64_sub(sf64 a, sf64 b) {
    return sf64_add(a, sf64_negate(b));
}

// ============================================================================
// Multiplication
// ============================================================================
// Both significands shifted left by 11 to place leading 1 at bit 63.
// Product has leading 1 at bit 126 or 127. Normalize to bit 62 of high word.
// Extract 10 round bits, shift by 10.

sf64 sf64_mul(sf64 a, sf64 b) {
    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);
    bool signZ = signA != signB;

    int expA = sf64_exp_raw(a);
    int expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a);
    sf64 sigB = sf64_sig(b);

    // Handle NaN — propagate payload
    if (sf64_is_nan(a) || sf64_is_nan(b)) return sf64_propagate_nan(a, b);

    // Handle infinity
    if (sf64_is_inf(a)) {
        if (sf64_is_zero(b)) return SF64_QNAN;  // inf * 0 = NaN
        return signZ ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_inf(b)) {
        if (sf64_is_zero(a)) return SF64_QNAN;  // 0 * inf = NaN
        return signZ ? SF64_NEG_INF : SF64_INF;
    }

    // Handle zero
    if (sf64_is_zero(a) || sf64_is_zero(b)) {
        return signZ ? SF64_NEG_ZERO : SF64_ZERO;
    }

    // Add implicit bit, normalize subnormals
    if (expA != 0) sigA.x |= 0x00100000u;
    else { int s = clz64(sigA) - 11; sigA = shl64(sigA, s); expA = 1 - s; }
    if (expB != 0) sigB.x |= 0x00100000u;
    else { int s = clz64(sigB) - 11; sigB = shl64(sigB, s); expB = 1 - s; }

    // Result exponent
    int expZ = expA + expB - SF64_EXP_BIAS;

    // Shift both significands to have leading 1 at bit 63
    sigA = shl64(sigA, 11);
    sigB = shl64(sigB, 11);

    // 64x64 → 128 multiplication
    uint128_t prod = mul64x64(sigA, sigB);
    sf64 sigZ = sf64(prod.w3, prod.w2);
    uint sticky = ((prod.w1 | prod.w0) != 0) ? 1u : 0u;

    // Normalize: product leading 1 at bit 126 or 127 of 128-bit result
    if ((sigZ.x & 0x80000000u) != 0) {
        // Overflow: product in [2,4), leading 1 at bit 63 of high word
        // Shift right by 1 to normalize to bit 62
        sticky |= (sigZ.y & 1u);
        sigZ = shr64(sigZ, 1);
        expZ++;
    }
    // Leading 1 now at bit 62 (bit 30 of sigZ.x)

    // Extract 10 round bits and shift by 10 to get mantissa at bit 52
    uint round_bits = (sigZ.y & 0x3FFu) | sticky;
    sigZ = shr64(sigZ, 10);

    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ============================================================================
// Division
// ============================================================================
// Implements long division with proper IEEE 754 rounding.
// Dividend significand is placed in a 128-bit register, divided by 64-bit divisor.

sf64 sf64_div(sf64 a, sf64 b) {
    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);
    bool signZ = signA != signB;

    int expA = sf64_exp_raw(a);
    int expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a);
    sf64 sigB = sf64_sig(b);

    // Handle NaN
    if (sf64_is_nan(a) || sf64_is_nan(b)) return sf64_propagate_nan(a, b);

    // Handle infinity
    if (sf64_is_inf(a)) {
        if (sf64_is_inf(b)) return SF64_QNAN;  // inf / inf = NaN
        return signZ ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_inf(b)) {
        return signZ ? SF64_NEG_ZERO : SF64_ZERO;  // x / inf = 0
    }

    // Handle zero
    if (sf64_is_zero(b)) {
        if (sf64_is_zero(a)) return SF64_QNAN;  // 0 / 0 = NaN
        return signZ ? SF64_NEG_INF : SF64_INF;  // x / 0 = inf
    }
    if (sf64_is_zero(a)) {
        return signZ ? SF64_NEG_ZERO : SF64_ZERO;
    }

    // Add implicit bit, normalize subnormals
    if (expA != 0) sigA.x |= 0x00100000u;
    else { int s = clz64(sigA) - 11; sigA = shl64(sigA, s); expA = 1 - s; }
    if (expB != 0) sigB.x |= 0x00100000u;
    else { int s = clz64(sigB) - 11; sigB = shl64(sigB, s); expB = 1 - s; }

    // Result exponent
    int expZ = expA - expB + SF64_EXP_BIAS;

    // Shift dividend to have leading 1 at bit 62 (aligned with our convention)
    sigA = shl64(sigA, 10);
    sigB = shl64(sigB, 10);

    // Check if dividend < divisor (quotient < 1.0)
    if (cmp64(sigA, sigB) < 0) {
        sigA = shl64(sigA, 1);
        expZ--;
    }

    // Long division: compute 64 bits of quotient
    // We compute quotient bit by bit for the top ~54 bits we need
    sf64 quotient = SF64_ZERO;
    sf64 remainder = sigA;

    // We need 53 bits of quotient plus rounding info
    for (int i = 62; i >= 0; i--) {
        // Shift quotient left by 1
        quotient = shl64(quotient, 1);

        if (cmp64(remainder, sigB) >= 0) {
            remainder = sub64(remainder, sigB);
            // Set bit i of quotient
            if (i >= 32) {
                quotient.x |= (1u << (i - 32));
            } else {
                quotient.y |= (1u << i);
            }
        }

        // Shift remainder left by 1 for next iteration
        remainder = shl64(remainder, 1);
    }

    // quotient has leading 1 at bit 62
    // Extract 10 round bits and compute sticky from remainder
    uint sticky = !sf64_is_zero(remainder) ? 1u : 0u;
    uint round_bits = (quotient.y & 0x3FFu) | sticky;
    sf64 sigZ = shr64(quotient, 10);

    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ============================================================================
// Fused Multiply-Add (FMA) — True IEEE 754 Single-Rounding
// ============================================================================
// Computes a*b+c with only ONE rounding at the end.
// Uses 128-bit intermediate to preserve full product precision.
// Synced to gpu_memory.mm canonical implementation.

sf64 sf64_fma(sf64 a, sf64 b, sf64 c) {
    // Handle special cases
    if (sf64_is_nan(a) || sf64_is_nan(b) || sf64_is_nan(c)) {
        return sf64_propagate_nan3(a, b, c);
    }

    bool signA = sf64_sign(a), signB = sf64_sign(b), signC = sf64_sign(c);
    bool signP = signA != signB;

    // Handle infinities
    if (sf64_is_inf(a)) {
        if (sf64_is_zero(b)) return SF64_QNAN;
        if (sf64_is_inf(c) && (signP != signC)) return SF64_QNAN;
        return signP ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_inf(b)) {
        if (sf64_is_zero(a)) return SF64_QNAN;
        if (sf64_is_inf(c) && (signP != signC)) return SF64_QNAN;
        return signP ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_inf(c)) return c;

    // Handle zeros — fast path when c==0 (common in matmul accumulation)
    if (sf64_is_zero(a) || sf64_is_zero(b)) return c;
    if (sf64_is_zero(c)) return sf64_mul(a, b);

    // Extract components
    int expA = sf64_exp_raw(a), expB = sf64_exp_raw(b), expC = sf64_exp_raw(c);
    sf64 sigA = sf64_sig(a), sigB = sf64_sig(b), sigC = sf64_sig(c);

    // Add implicit bits, handle denormals
    if (expA != 0) sigA.x |= 0x00100000u;
    else { int s = clz64(sigA) - 11; sigA = shl64(sigA, s); expA = 1 - s; }
    if (expB != 0) sigB.x |= 0x00100000u;
    else { int s = clz64(sigB) - 11; sigB = shl64(sigB, s); expB = 1 - s; }
    if (expC != 0) sigC.x |= 0x00100000u;
    else { int s = clz64(sigC) - 11; sigC = shl64(sigC, s); expC = 1 - s; }

    // Product exponent
    int expP = expA + expB - SF64_EXP_BIAS;

    // Shift significands to bit 63 for full-precision multiply
    sigA = shl64(sigA, 11);
    sigB = shl64(sigB, 11);

    // Full 128-bit product
    uint128_t prod = mul64x64(sigA, sigB);
    sf128 P = sf128{sf64(prod.w3, prod.w2), sf64(prod.w1, prod.w0)};

    // Normalize product to leading 1 at bit 126 (bit 62 of hi)
    if ((P.hi.x & 0x80000000u) != 0) {
        P = shr128_jam(P, 1);
        expP++;
    }

    // Prepare addend C: shift sigC (leading 1 at bit 52) to bit 126
    sf128 C = sf128{shl64(sigC, 10), SF64_ZERO};

    // Align to common exponent
    int expDiff = expP - expC;
    int expZ = expP;

    if (expDiff > 0) {
        C = shr128_jam(C, expDiff);
    } else if (expDiff < 0) {
        P = shr128_jam(P, -expDiff);
        expZ = expC;
    }

    // Add or subtract based on signs
    sf128 R;
    bool signZ;

    if (signP == signC) {
        signZ = signP;
        R = add128(P, C);
        // Check for overflow
        if ((R.hi.x & 0x80000000u) != 0) {
            R = shr128_jam(R, 1);
            expZ++;
        }
    } else {
        int cmp = cmp128(P, C);
        if (cmp == 0) return SF64_ZERO;

        if (cmp > 0) {
            signZ = signP;
            R = sub128(P, C);
        } else {
            signZ = signC;
            R = sub128(C, P);
        }

        // Normalize: shift left until leading 1 at bit 62 of hi
        int shift = clz128(R) - 1;
        if (shift > 0) {
            R = shl128(R, shift);
            expZ -= shift;
        } else if (shift < 0) {
            R = shr128_jam(R, -shift);
            expZ -= shift;
        }
    }

    // Extract result: leading 1 at bit 62, round bits in 9-0 of hi.lo
    uint sticky = (R.lo.x | R.lo.y) != 0 ? 1u : 0u;
    uint round_bits = (R.hi.y & 0x3FFu) | sticky;
    sf64 sigZ = shr64(R.hi, 10);

    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ============================================================================
// Comparison
// ============================================================================
// Returns -1 if a < b, 0 if a == b, 1 if a > b, 2 if unordered (NaN)

inline int sf64_compare(sf64 a, sf64 b) {
    // NaN comparisons are unordered — return 2 per IEEE 754
    if (sf64_is_nan(a) || sf64_is_nan(b)) return 2;

    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);

    // Handle zeros (both +0 and -0 are equal)
    if (sf64_is_zero(a) && sf64_is_zero(b)) return 0;

    // Different signs: negative < positive
    if (signA != signB) {
        return signA ? -1 : 1;
    }

    // Same sign: compare magnitudes
    int mag_cmp = cmp64(sf64(a.x & 0x7FFFFFFFu, a.y),
                        sf64(b.x & 0x7FFFFFFFu, b.y));

    // For negative numbers, larger magnitude means smaller value
    return signA ? -mag_cmp : mag_cmp;
}

inline bool sf64_eq(sf64 a, sf64 b) {
    if (sf64_is_nan(a) || sf64_is_nan(b)) return false;  // NaN != NaN
    if (sf64_is_zero(a) && sf64_is_zero(b)) return true;  // +0 == -0
    return (a.x == b.x) && (a.y == b.y);
}

inline bool sf64_ne(sf64 a, sf64 b) {
    return !sf64_eq(a, b);
}

inline bool sf64_lt(sf64 a, sf64 b) {
    int c = sf64_compare(a, b);
    return c == -1;  // Only true for ordered less-than (not NaN)
}

inline bool sf64_le(sf64 a, sf64 b) {
    int c = sf64_compare(a, b);
    return c == -1 || c == 0;  // Only true for ordered (not NaN)
}

inline bool sf64_gt(sf64 a, sf64 b) {
    int c = sf64_compare(a, b);
    return c == 1;  // Only true for ordered greater-than
}

inline bool sf64_ge(sf64 a, sf64 b) {
    int c = sf64_compare(a, b);
    return c == 1 || c == 0;  // Only true for ordered
}

// ============================================================================
// Conversions
// ============================================================================

// Convert integer to sf64
inline sf64 sf64_from_int(int val) {
    if (val == 0) return SF64_ZERO;
    bool sign = val < 0;
    uint abs_val = sign ? uint(-val) : uint(val);

    // Find position of leading 1
    int lz = clz(abs_val);
    int exp_raw = SF64_EXP_BIAS + 31 - lz;

    // Shift mantissa into position (leading 1 at bit 52)
    sf64 sig;
    int shift = 20 - (31 - lz);  // Position relative to bit 20 of .x
    if (shift >= 0) {
        sig = sf64(abs_val >> shift, abs_val << (32 - shift));
    } else {
        sig = sf64(abs_val << (-shift), 0u);
    }
    sig.x &= SF64_MANT_HI_MASK;  // Remove implicit bit

    return sf64_pack(sign, exp_raw, sig.x, sig.y);
}

// ============================================================================
// Transcendental Functions — SoftFloat f64 implementations
// ============================================================================
// Metal has NO native f64 math. These use sf64 FMA chains for precision.

// Helper constants for transcendentals
constant sf64 SF64_LN2_HI = sf64(0x3FE62E42u, 0xFEFA39EFu);  // ln(2) high part
constant sf64 SF64_LN2_LO = sf64(0x3C7ABC9Eu, 0x3B39803Fu);  // ln(2) low part
constant sf64 SF64_LN2     = sf64(0x3FE62E42u, 0xFEFA39EFu);  // ln(2) ≈ 0.6931471805599453
constant sf64 SF64_LOG2E   = sf64(0x3FF71547u, 0x652B82FEu);  // log2(e) ≈ 1.4426950408889634
constant sf64 SF64_INV_LN2 = sf64(0x3FF71547u, 0x652B82FEu);  // 1/ln(2)
// SF64_TWO already defined in constants section above
constant sf64 SF64_HALF    = sf64(0x3FE00000u, 0x00000000u);   // 0.5
constant sf64 SF64_PI      = sf64(0x400921FBu, 0x54442D18u);   // π
constant sf64 SF64_PI_2    = sf64(0x3FF921FBu, 0x54442D18u);   // π/2
constant sf64 SF64_PI_4    = sf64(0x3FE921FBu, 0x54442D18u);   // π/4
constant sf64 SF64_TWO_PI  = sf64(0x401921FBu, 0x54442D18u);   // 2π
constant sf64 SF64_INV_PI  = sf64(0x3FD45F30u, 0x6DC9C883u);   // 1/π
constant sf64 SF64_FOUR_PI = sf64(0x3FF45F30u, 0x6DC9C883u);   // 4/π

// sf64_sqrt: Newton-Raphson iteration
// Uses reciprocal sqrt estimate then multiply
inline sf64 sf64_sqrt(sf64 x) {
    if (sf64_is_zero(x)) return SF64_ZERO;
    if (sf64_is_nan(x) || (x.x >> 31)) return SF64_QNAN; // negative → NaN
    if (sf64_is_inf(x)) return SF64_INF;

    // Extract exponent for initial estimate
    int exp_raw = sf64_exp_raw(x);
    // Halve the exponent for sqrt estimate
    int new_exp = ((exp_raw - SF64_EXP_BIAS) / 2) + SF64_EXP_BIAS;
    sf64 y = sf64((uint(new_exp) << 20) | (x.x & 0x000FFFFFu), x.y);

    // 5 Newton-Raphson iterations: y = (y + x/y) * 0.5
    for (int i = 0; i < 5; i++) {
        sf64 xdivy = sf64_div(x, y);
        y = sf64_mul(sf64_add(y, xdivy), SF64_HALF);
    }
    return y;
}

// sf64_exp: e^x using range reduction and minimax polynomial
// Range reduction: x = n*ln2 + r, |r| < ln2/2
// Polynomial: e^r ≈ 1 + r + r²/2! + ... + r^13/13!
inline sf64 sf64_exp(sf64 x) {
    if (sf64_is_nan(x)) return SF64_QNAN;
    if (sf64_is_zero(x)) return SF64_ONE;

    // Check for overflow/underflow
    // e^709.78 ≈ DBL_MAX, e^(-745.13) ≈ 0
    sf64 max_val = sf64(0x40862E42u, 0xFEFA39EFu);  // ~709.78
    sf64 min_val = sf64(0xC0874910u, 0xD52D3052u);  // ~-745.13
    if (sf64_gt(x, max_val)) return SF64_INF;
    if (sf64_lt(x, min_val)) return SF64_ZERO;

    // n = round(x / ln2)
    sf64 x_over_ln2 = sf64_mul(x, SF64_INV_LN2);
    // Round to nearest integer
    int n_int;
    {
        // Extract integer part
        int exp_val = sf64_exp_raw(x_over_ln2) - SF64_EXP_BIAS;
        if (exp_val < 0) {
            n_int = 0;
        } else if (exp_val < 52) {
            // Extract integer part using 32-bit arithmetic (Metal has no 64-bit ints)
            uint mant_hi = (x_over_ln2.x & 0x000FFFFFu) | 0x00100000u;
            uint mant_lo = x_over_ln2.y;
            int shift = 52 - exp_val;
            uint val;
            if (shift >= 32) {
                val = mant_hi >> (shift - 32);
            } else if (shift > 0) {
                val = (mant_hi << (32 - shift)) | (mant_lo >> shift);
            } else {
                val = (mant_hi << 32) | mant_lo; // won't happen for exp range
            }
            // Round: check the bit just below the integer part
            if (shift > 0 && shift <= 52) {
                uint round_bit;
                if (shift >= 33) {
                    round_bit = (mant_hi >> (shift - 33)) & 1u;
                } else if (shift >= 1) {
                    round_bit = (shift > 32) ? ((mant_hi >> (shift - 33)) & 1u)
                                              : ((mant_lo >> (shift - 1)) & 1u);
                } else {
                    round_bit = 0;
                }
                val += round_bit;
            }
            n_int = (x_over_ln2.x >> 31) ? -int(val) : int(val);
        } else {
            n_int = (x_over_ln2.x >> 31) ? -1024 : 1024;
        }
    }

    // r = x - n * ln2 (Cody-Waite reduction)
    sf64 n_sf = sf64_from_int(n_int);
    sf64 r = sf64_sub(x, sf64_mul(n_sf, SF64_LN2_HI));
    r = sf64_sub(r, sf64_mul(n_sf, SF64_LN2_LO));

    // Polynomial approximation: e^r ≈ 1 + r(1 + r/2(1 + r/3(1 + r/4(...))))
    // Horner form with 13 terms for full double precision
    sf64 r2 = sf64_mul(r, r);
    // Coefficients: 1/n! for n=2..13
    sf64 c2  = SF64_HALF;                                    // 1/2
    sf64 c3  = sf64(0x3FC55555u, 0x55555555u);               // 1/6
    sf64 c4  = sf64(0x3FA55555u, 0x55555555u);               // 1/24
    sf64 c5  = sf64(0x3F811111u, 0x11111111u);               // 1/120
    sf64 c6  = sf64(0x3F56C16Cu, 0x16C16C17u);               // 1/720
    sf64 c7  = sf64(0x3F2A01A0u, 0x1A01A01Au);               // 1/5040
    sf64 c8  = sf64(0x3EFA01A0u, 0x1A01A01Au);               // 1/40320
    sf64 c9  = sf64(0x3EC71DE3u, 0xA556C734u);               // 1/362880
    sf64 c10 = sf64(0x3E927E4Fu, 0xB7789F5Cu);               // 1/3628800
    sf64 c11 = sf64(0x3E5AE64Du, 0x67F544E4u);               // 1/39916800
    sf64 c12 = sf64(0x3E21EED8u, 0xEFF8D898u);               // 1/479001600
    sf64 c13 = sf64(0x3DE6124Eu, 0x13B6CA51u);               // 1/6227020800

    // Horner evaluation: p = c13
    sf64 p = c13;
    p = sf64_fma(p, r, c12);
    p = sf64_fma(p, r, c11);
    p = sf64_fma(p, r, c10);
    p = sf64_fma(p, r, c9);
    p = sf64_fma(p, r, c8);
    p = sf64_fma(p, r, c7);
    p = sf64_fma(p, r, c6);
    p = sf64_fma(p, r, c5);
    p = sf64_fma(p, r, c4);
    p = sf64_fma(p, r, c3);
    p = sf64_fma(p, r, c2);
    // e^r = 1 + r + r^2*c2 + ... = 1 + r*(1 + r*p) where p starts at c2
    sf64 exp_r = sf64_add(SF64_ONE, sf64_mul(r, sf64_add(SF64_ONE, sf64_mul(r, p))));

    // Scale by 2^n: adjust exponent
    if (n_int == 0) return exp_r;
    int exp_result = sf64_exp_raw(exp_r) + n_int;
    if (exp_result >= 2047) return SF64_INF;
    if (exp_result <= 0) return SF64_ZERO;
    return sf64(((uint(exp_result) << 20) | (exp_r.x & 0x800FFFFFu)), exp_r.y);
}

// sf64_log: natural logarithm
// Range reduction: x = 2^n * m, 1 ≤ m < 2
// log(x) = n*ln2 + log(m)
// log(m) via minimax polynomial on f = m - 1
inline sf64 sf64_log(sf64 x) {
    if (sf64_is_nan(x) || (x.x >> 31)) return SF64_QNAN; // NaN or negative
    if (sf64_is_zero(x)) return sf64(0xFFF00000u, 0x00000000u); // -Inf
    if (sf64_is_inf(x)) return SF64_INF;

    // Extract exponent and mantissa
    int exp_raw = sf64_exp_raw(x);
    int n = exp_raw - SF64_EXP_BIAS;

    // f = mantissa in [1, 2): set exponent to bias (1.xxx)
    sf64 f = sf64((0x3FF00000u | (x.x & 0x000FFFFFu)), x.y);

    // Reduce to [sqrt(2)/2, sqrt(2)] for better convergence
    // If f > sqrt(2), use f/2 and n+1
    sf64 sqrt2 = sf64(0x3FF6A09Eu, 0x667F3BCDu); // sqrt(2)
    if (sf64_gt(f, sqrt2)) {
        f = sf64_mul(f, SF64_HALF);
        n++;
    }

    // s = (f - 1) / (f + 1)
    sf64 f_m1 = sf64_sub(f, SF64_ONE);
    sf64 f_p1 = sf64_add(f, SF64_ONE);
    sf64 s = sf64_div(f_m1, f_p1);
    sf64 s2 = sf64_mul(s, s);

    // log(f) = 2*s + 2*s^3/3 + 2*s^5/5 + ... (odd powers only)
    // Coefficients: 2/(2k+1) for k=1,2,...
    sf64 c1 = sf64(0x3FE55555u, 0x55555555u);  // 2/3 = 0.6666...
    sf64 c2 = sf64(0x3FD99999u, 0x9999999Au);  // 2/5 = 0.4
    sf64 c3 = sf64(0x3FD24924u, 0x92492492u);  // 2/7 ≈ 0.2857
    sf64 c4 = sf64(0x3FCC71C7u, 0x1C71C71Cu);  // 2/9 ≈ 0.2222
    sf64 c5 = sf64(0x3FC745D1u, 0x745D1746u);  // 2/11 ≈ 0.1818
    sf64 c6 = sf64(0x3FC3B13Bu, 0x13B13B14u);  // 2/13 ≈ 0.1538

    sf64 p = c6;
    p = sf64_fma(p, s2, c5);
    p = sf64_fma(p, s2, c4);
    p = sf64_fma(p, s2, c3);
    p = sf64_fma(p, s2, c2);
    p = sf64_fma(p, s2, c1);
    // log(f) = 2*s*(1 + s²*p)
    sf64 log_f = sf64_mul(SF64_TWO, sf64_mul(s, sf64_add(SF64_ONE, sf64_mul(s2, p))));

    // log(x) = n * ln(2) + log(f)
    sf64 n_sf = sf64_from_int(n);
    return sf64_add(sf64_mul(n_sf, SF64_LN2), log_f);
}

// sf64_sin / sf64_cos: Cody-Waite range reduction to [-π/4, π/4]
// Then minimax polynomial

// Helper: reduce x to [-π/4, π/4], return quadrant (0-3)
inline sf64 sf64_trig_reduce(sf64 x, thread int& quadrant) {
    // |x| / (π/4) to find quadrant
    sf64 abs_x = sf64_abs(x);
    sf64 four_over_pi = SF64_FOUR_PI;
    sf64 j_sf = sf64_mul(abs_x, four_over_pi);

    // Extract integer part
    int j;
    {
        int exp_val = sf64_exp_raw(j_sf) - SF64_EXP_BIAS;
        if (exp_val < 0) {
            j = 0;
        } else if (exp_val < 30) {
            // Extract integer part using 32-bit arithmetic (Metal has no 64-bit ints)
            uint mant_hi = (j_sf.x & 0x000FFFFFu) | 0x00100000u;
            uint mant_lo = j_sf.y;
            int shift = 52 - exp_val;
            uint val;
            if (shift >= 32) {
                val = mant_hi >> (shift - 32);
            } else if (shift > 0) {
                val = (mant_hi << (32 - shift)) | (mant_lo >> shift);
            } else {
                val = mant_hi;  // exp_val >= 52 won't reach here (cap is 30)
            }
            j = int(val);
        } else {
            j = 0; // Very large: use modular reduction (simplified)
        }
    }
    // Make j odd for rounding
    j = (j + 1) & ~1;
    quadrant = j & 7;

    sf64 j_val = sf64_from_int(j);
    // r = |x| - j * π/4
    sf64 r = sf64_sub(abs_x, sf64_mul(j_val, SF64_PI_4));

    // Handle sign of original x
    if (x.x >> 31) {
        quadrant = (8 - quadrant) & 7;
        r = sf64_negate(r);
    }
    return r;
}

// Sine polynomial on [-π/4, π/4]: sin(r) ≈ r - r³/6 + r⁵/120 - ...
inline sf64 sf64_sin_poly(sf64 r) {
    sf64 r2 = sf64_mul(r, r);
    // Coefficients: alternating 1/(2k+1)!
    sf64 s3 = sf64(0xBFC55555u, 0x55555549u);  // -1/6
    sf64 s5 = sf64(0x3F811111u, 0x1110F8A6u);  //  1/120
    sf64 s7 = sf64(0xBF2A01A0u, 0x19C161D5u);  // -1/5040
    sf64 s9 = sf64(0x3EC71DE3u, 0x57B1FE7Du);  //  1/362880
    sf64 s11= sf64(0xBE5AE5E6u, 0x8A2B9CEBu);  // -1/39916800

    sf64 p = s11;
    p = sf64_fma(p, r2, s9);
    p = sf64_fma(p, r2, s7);
    p = sf64_fma(p, r2, s5);
    p = sf64_fma(p, r2, s3);
    return sf64_add(r, sf64_mul(r, sf64_mul(r2, p)));
}

// Cosine polynomial on [-π/4, π/4]: cos(r) ≈ 1 - r²/2 + r⁴/24 - ...
inline sf64 sf64_cos_poly(sf64 r) {
    sf64 r2 = sf64_mul(r, r);
    sf64 c2 = sf64(0xBFE00000u, 0x00000000u);  // -1/2
    sf64 c4 = sf64(0x3FA55555u, 0x55555555u);  //  1/24
    sf64 c6 = sf64(0xBF56C16Cu, 0x16C16C17u);  // -1/720
    sf64 c8 = sf64(0x3EFA01A0u, 0x1A01A01Au);  //  1/40320
    sf64 c10= sf64(0xBE927E4Fu, 0xB7789F5Cu);  // -1/3628800

    sf64 p = c10;
    p = sf64_fma(p, r2, c8);
    p = sf64_fma(p, r2, c6);
    p = sf64_fma(p, r2, c4);
    p = sf64_fma(p, r2, c2);
    return sf64_add(SF64_ONE, sf64_mul(r2, p));
}

inline sf64 sf64_sin(sf64 x) {
    if (sf64_is_nan(x) || sf64_is_inf(x)) return SF64_QNAN;
    if (sf64_is_zero(x)) return x; // preserve sign of zero

    int quadrant;
    sf64 r = sf64_trig_reduce(x, quadrant);

    sf64 result;
    switch (quadrant & 3) {
        case 0: result = sf64_sin_poly(r); break;
        case 1: result = sf64_cos_poly(r); break;
        case 2: result = sf64_negate(sf64_sin_poly(r)); break;
        case 3: result = sf64_negate(sf64_cos_poly(r)); break;
        default: result = sf64_sin_poly(r); break;
    }
    return result;
}

inline sf64 sf64_cos(sf64 x) {
    if (sf64_is_nan(x) || sf64_is_inf(x)) return SF64_QNAN;
    if (sf64_is_zero(x)) return SF64_ONE;

    int quadrant;
    sf64 r = sf64_trig_reduce(x, quadrant);

    sf64 result;
    switch (quadrant & 3) {
        case 0: result = sf64_cos_poly(r); break;
        case 1: result = sf64_negate(sf64_sin_poly(r)); break;
        case 2: result = sf64_negate(sf64_cos_poly(r)); break;
        case 3: result = sf64_sin_poly(r); break;
        default: result = sf64_cos_poly(r); break;
    }
    return result;
}

// sf64_tanh: (e^2x - 1) / (e^2x + 1)
// For |x| >= 20: return ±1 (saturated)
inline sf64 sf64_tanh(sf64 x) {
    if (sf64_is_nan(x)) return SF64_QNAN;
    if (sf64_is_zero(x)) return x;

    sf64 abs_x = sf64_abs(x);
    sf64 twenty = sf64(0x40340000u, 0x00000000u); // 20.0
    if (sf64_ge(abs_x, twenty)) {
        return (x.x >> 31) ? sf64(0xBFF00000u, 0x00000000u) : SF64_ONE;
    }

    sf64 e2x = sf64_exp(sf64_mul(SF64_TWO, abs_x));
    sf64 num = sf64_sub(e2x, SF64_ONE);
    sf64 den = sf64_add(e2x, SF64_ONE);
    sf64 result = sf64_div(num, den);
    return (x.x >> 31) ? sf64_negate(result) : result;
}

// ============================================================================
// GPU Kernels — Elementwise, Reduce, Transpose
// ============================================================================

kernel void elementwise_sf64(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    sf64 a = A[tid];
    sf64 b = (op <= 3) ? B[tid] : SF64_ZERO; // Binary ops read B

    sf64 result;
    switch (op) {
        case 0:  result = sf64_add(a, b); break;           // ADD
        case 1:  result = sf64_sub(a, b); break;           // SUB
        case 2:  result = sf64_mul(a, b); break;           // MUL
        case 3:  result = sf64_div(a, b); break;           // DIV
        case 4:  result = sf64_negate(a); break;           // NEG
        case 5:  result = sf64_abs(a); break;              // ABS
        case 6:  result = sf64_exp(a); break;              // EXP
        case 7:  result = sf64_log(a); break;              // LOG
        case 8:  result = sf64_sin(a); break;              // SIN
        case 9:  result = sf64_cos(a); break;              // COS
        case 10: result = sf64_tanh(a); break;             // TANH
        case 11: result = sf64_gt(a, SF64_ZERO) ? a : SF64_ZERO; break;  // RELU
        case 12: {                                                         // SIGMOID
            sf64 neg_a = sf64_negate(a);
            sf64 exp_neg = sf64_exp(neg_a);
            result = sf64_div(SF64_ONE, sf64_add(SF64_ONE, exp_neg));
            break;
        }
        case 13: result = sf64_sqrt(a); break;             // SQRT
        case 14: result = sf64_div(SF64_ONE, a); break;    // RECIPROCAL
        default: result = SF64_ZERO; break;
    }
    C[tid] = result;
}

// Reduction kernel — two-pass design
// Pass 1: each threadgroup reduces 256 elements to 1 partial result
// The host then launches a second pass if needed
kernel void reduce_sf64(
    device const sf64* in [[buffer(0)]],
    device sf64* out [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup sf64 shared[256];

    // Load element or identity
    sf64 identity;
    switch (op) {
        case 0: identity = SF64_ZERO; break;  // SUM
        case 1: identity = SF64_ONE; break;   // PROD
        case 2: identity = SF64_INF; break;   // MIN
        case 3: identity = sf64(0xFFF00000u, 0x00000000u); break;  // MAX (-Inf)
        case 4: identity = SF64_ZERO; break;  // MEAN (sum then divide)
        default: identity = SF64_ZERO; break;
    }

    shared[lid] = (tid < N) ? in[tid] : identity;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction in shared memory
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            sf64 a = shared[lid];
            sf64 b = shared[lid + s];
            switch (op) {
                case 0: case 4: shared[lid] = sf64_add(a, b); break;
                case 1: shared[lid] = sf64_mul(a, b); break;
                case 2: shared[lid] = sf64_lt(a, b) ? a : b; break;
                case 3: shared[lid] = sf64_gt(a, b) ? a : b; break;
                default: break;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes partial result
    if (lid == 0) {
        out[gid] = shared[0];
    }
}

// Transpose kernel: out[j*rows + i] = in[i*cols + j]
kernel void transpose_sf64(
    device const sf64* in [[buffer(0)]],
    device sf64* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.y; // row
    uint j = tid.x; // col
    if (i >= rows || j >= cols) return;
    out[j * rows + i] = in[i * cols + j];
}

// ============================================================================
// Softmax Kernel — Numerically Stable, Fused (max, exp, sum, normalize)
// ============================================================================
// Each thread processes one contiguous slice of slice_len elements.
// softmax(x_k) = exp(x_k - max(x)) / sum(exp(x - max(x)))
// Handles axis-last layout (inner_stride=1). General axis handled on CPU.

kernel void softmax_sf64(
    device const sf64* in [[buffer(0)]],
    device sf64* out [[buffer(1)]],
    constant uint& slice_len [[buffer(2)]],
    constant uint& num_slices [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_slices) return;

    uint base = tid * slice_len;

    // Pass 1: find max for numerical stability
    sf64 max_val = in[base];
    for (uint k = 1; k < slice_len; k++) {
        sf64 v = in[base + k];
        if (sf64_gt(v, max_val)) max_val = v;
    }

    // Pass 2: exp(x - max) and accumulate sum
    sf64 sum_exp = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sf64 e = sf64_exp(sf64_sub(in[base + k], max_val));
        out[base + k] = e;
        sum_exp = sf64_add(sum_exp, e);
    }

    // Guard against division by zero
    if (sf64_is_zero(sum_exp)) sum_exp = SF64_ONE;

    // Pass 3: normalize
    for (uint k = 0; k < slice_len; k++) {
        out[base + k] = sf64_div(out[base + k], sum_exp);
    }
}

// ============================================================================
// Normalize Kernel — Layer Normalization (mean, variance, scale+shift)
// ============================================================================
// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
// Each thread processes one contiguous slice of slice_len elements.

kernel void normalize_sf64(
    device const sf64* in [[buffer(0)]],
    device sf64* out [[buffer(1)]],
    constant uint& slice_len [[buffer(2)]],
    constant uint& num_slices [[buffer(3)]],
    constant sf64& gamma_sf [[buffer(4)]],
    constant sf64& beta_sf [[buffer(5)]],
    constant sf64& epsilon_sf [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_slices) return;

    uint base = tid * slice_len;
    sf64 n_sf = sf64_from_int(int(slice_len));

    // Pass 1: compute mean
    sf64 sum = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sum = sf64_add(sum, in[base + k]);
    }
    sf64 mean = sf64_div(sum, n_sf);

    // Pass 2: compute variance
    sf64 var_sum = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sf64 diff = sf64_sub(in[base + k], mean);
        var_sum = sf64_add(var_sum, sf64_mul(diff, diff));
    }
    sf64 variance = sf64_div(var_sum, n_sf);

    // inv_std = 1 / sqrt(var + epsilon)
    sf64 inv_std = sf64_div(SF64_ONE, sf64_sqrt(sf64_add(variance, epsilon_sf)));

    // Pass 3: normalize with scale and shift
    for (uint k = 0; k < slice_len; k++) {
        sf64 x_norm = sf64_mul(sf64_sub(in[base + k], mean), inv_std);
        out[base + k] = sf64_add(sf64_mul(gamma_sf, x_norm), beta_sf);
    }
}

// ============================================================================
// Matrix Multiplication Kernel — Optimized with Threadgroup Shared Memory
// ============================================================================
// Each threadgroup (8×8 = 64 threads) computes a 32×32 output block.
// Each thread computes a 4×4 sub-tile of the output.
// K dimension is processed in chunks of TILE_K=8.
// A and B tiles are cooperatively loaded into threadgroup shared memory,
// reducing device memory bandwidth by ~8x vs naive per-thread loads.
// Register budget: ~100 registers/thread (safe for M1+ at 256 limit).
// Shared memory: 4KB per threadgroup (safe for 32KB limit).

constant uint SF64_TG = 8;       // Threadgroup dimension (8×8 = 64 threads)
constant uint SF64_TT = 4;       // Thread tile dimension (4×4 per thread)
constant uint SF64_BLK = 32;     // Block dimension (TG * TT = 8*4 = 32)
constant uint SF64_TILE_K = 8;   // K-dimension blocking factor

kernel void matmul_sf64(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    // Output block origin for this threadgroup
    uint baseRow = group_id.y * SF64_BLK;
    uint baseCol = group_id.x * SF64_BLK;

    // Thread's sub-tile origin within the 32×32 block
    uint threadRow = lid.y * SF64_TT;
    uint threadCol = lid.x * SF64_TT;

    // Accumulators for 4×4 sub-tile
    sf64 acc[SF64_TT][SF64_TT];
    for (uint i = 0; i < SF64_TT; i++)
        for (uint j = 0; j < SF64_TT; j++)
            acc[i][j] = SF64_ZERO;

    // Shared memory tiles — cooperatively loaded by all 64 threads
    // sA: 32 rows × 8 cols = 256 sf64 = 2KB
    // sB: 8 rows × 32 cols = 256 sf64 = 2KB
    threadgroup sf64 sA[SF64_BLK * SF64_TILE_K];   // [row * TILE_K + col]
    threadgroup sf64 sB[SF64_TILE_K * SF64_BLK];    // [row * BLK + col]

    uint tid = lid.y * SF64_TG + lid.x;  // Linear thread ID (0..63)

    // Process K in chunks of TILE_K
    for (uint kBlock = 0; kBlock < K; kBlock += SF64_TILE_K) {

        // === Cooperative load A tile [32 × TILE_K] ===
        // 256 elements / 64 threads = 4 elements per thread
        for (uint i = 0; i < 4; i++) {
            uint idx = tid * 4 + i;
            uint row = idx >> 3;       // idx / TILE_K (TILE_K=8)
            uint col = idx & 7u;       // idx % TILE_K
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;
            sA[row * SF64_TILE_K + col] =
                (gRow < M && gCol < K) ? A[gRow * K + gCol] : SF64_ZERO;
        }

        // === Cooperative load B tile [TILE_K × 32] ===
        // 256 elements / 64 threads = 4 elements per thread
        for (uint i = 0; i < 4; i++) {
            uint idx = tid * 4 + i;
            uint row = idx >> 5;       // idx / BLK (BLK=32)
            uint col = idx & 31u;      // idx % BLK
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;
            sB[row * SF64_BLK + col] =
                (gRow < K && gCol < N) ? B[gRow * N + gCol] : SF64_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute: each thread processes TILE_K values from shared memory ===
        // 8 K-values × 16 FMAs = 128 FMA calls per thread per K-chunk
        for (uint kk = 0; kk < SF64_TILE_K; kk++) {
            sf64 a_val[SF64_TT];
            sf64 b_val[SF64_TT];

            for (uint i = 0; i < SF64_TT; i++)
                a_val[i] = sA[(threadRow + i) * SF64_TILE_K + kk];
            for (uint j = 0; j < SF64_TT; j++)
                b_val[j] = sB[kk * SF64_BLK + threadCol + j];

            for (uint i = 0; i < SF64_TT; i++)
                for (uint j = 0; j < SF64_TT; j++)
                    acc[i][j] = sf64_fma(a_val[i], b_val[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store results ===
    for (uint i = 0; i < SF64_TT; i++) {
        for (uint j = 0; j < SF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = acc[i][j];
        }
    }
}

#endif // ESHKOL_METAL_SOFTFLOAT_H
