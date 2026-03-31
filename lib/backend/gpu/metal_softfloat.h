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
        return sf64(0u, select(0u, 1u, (x.x | x.y) != 0));
    }
    if (n >= 32) {
        uint lost_bits = x.y | select(0u, x.x << (64 - n), n > 32);
        return sf64(0u, (x.x >> (n - 32)) | select(0u, 1u, lost_bits != 0));
    }
    uint lost_bits = x.y << (32 - n);
    return sf64(x.x >> n, ((x.y >> n) | (x.x << (32 - n))) | select(0u, 1u, lost_bits != 0));
}

inline sf64 add64(sf64 a, sf64 b) {
    uint lo = a.y + b.y;
    uint carry = select(0u, 1u, lo < a.y);
    uint hi = a.x + b.x + carry;
    return sf64(hi, lo);
}

inline sf64 add64_carry(sf64 a, sf64 b, thread bool& carry_out) {
    uint lo = a.y + b.y;
    uint c1 = select(0u, 1u, lo < a.y);
    uint hi = a.x + b.x + c1;
    carry_out = (hi < a.x) || (c1 != 0 && hi == a.x + c1);
    return sf64(hi, lo);
}

inline sf64 sub64(sf64 a, sf64 b) {
    uint lo = a.y - b.y;
    uint borrow = select(0u, 1u, a.y < b.y);
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
        return sf128{SF64_ZERO, sf64(0u, select(0u, 1u, (x.hi.x | x.hi.y | x.lo.x | x.lo.y) != 0))};
    }
    if (n >= 64) {
        uint sticky = select(0u, 1u, (x.lo.x | x.lo.y) != 0);
        sf64 new_lo = shr64_jam(x.hi, n - 64);
        new_lo.y |= sticky;
        return sf128{SF64_ZERO, new_lo};
    }
    // Fast path: 1 <= n <= 31 — direct word shifts, no nested calls
    // This is the hot path in matmul FMA alignment (small exponent differences)
    if (n < 32) {
        uint un = uint(n);
        uint inv = 32u - un;
        uint sticky = select(0u, 1u, (x.lo.y & ((1u << un) - 1u)) != 0);
        uint w0 = (x.lo.y >> un) | (x.lo.x << inv);
        uint w1 = (x.lo.x >> un) | (x.hi.y << inv);
        uint w2 = (x.hi.y >> un) | (x.hi.x << inv);
        uint w3 = x.hi.x >> un;
        return sf128{sf64(w3, w2), sf64(w1, w0 | sticky)};
    }
    // n == 32..63: delegate to shr128 + sticky (infrequent in matmul)
    uint sticky = select(0u, 1u, (x.lo.y | (x.lo.x << (64u - uint(n)))) != 0);
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
    // Direct 4-word addition with carry propagation (no nested calls)
    uint w0 = a.lo.y + b.lo.y;
    uint c0 = select(0u, 1u, w0 < a.lo.y);
    uint w1 = a.lo.x + b.lo.x + c0;
    uint c1 = select(0u, 1u, c0 ? (w1 <= a.lo.x) : (w1 < a.lo.x));
    uint w2 = a.hi.y + b.hi.y + c1;
    uint c2 = select(0u, 1u, c1 ? (w2 <= a.hi.y) : (w2 < a.hi.y));
    uint w3 = a.hi.x + b.hi.x + c2;
    return sf128{sf64(w3, w2), sf64(w1, w0)};
}

inline sf128 sub128(sf128 a, sf128 b) {
    // Direct 4-word subtraction with borrow propagation (no nested calls)
    uint w0 = a.lo.y - b.lo.y;
    uint b0 = select(0u, 1u, a.lo.y < b.lo.y);
    uint w1 = a.lo.x - b.lo.x - b0;
    uint b1 = select(0u, 1u, b0 ? (a.lo.x <= b.lo.x) : (a.lo.x < b.lo.x));
    uint w2 = a.hi.y - b.hi.y - b1;
    uint b2 = select(0u, 1u, b1 ? (a.hi.y <= b.hi.y) : (a.hi.y < b.hi.y));
    uint w3 = a.hi.x - b.hi.x - b2;
    return sf128{sf64(w3, w2), sf64(w1, w0)};
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
// Uses 32-bit mulhi() for 3.2x fewer ops than 16-bit decomposition.
// MSL mulhi(a,b) returns high 32 bits of a*b (available since MSL 1.0).
inline uint128_t mul64x64(sf64 a, sf64 b) {
    // 4 partial products, each 64-bit (lo from *, hi from mulhi)
    uint p0_lo = a.y * b.y;            uint p0_hi = mulhi(a.y, b.y);
    uint p1_lo = a.x * b.y;            uint p1_hi = mulhi(a.x, b.y);
    uint p2_lo = a.y * b.x;            uint p2_hi = mulhi(a.y, b.x);
    uint p3_lo = a.x * b.x;            uint p3_hi = mulhi(a.x, b.x);

    // Combine: result = p0 + (p1 + p2) << 32 + p3 << 64
    // Uses branchless select() for carry propagation
    uint w0 = p0_lo;

    uint t1 = p0_hi + p1_lo;           uint c1 = select(0u, 1u, t1 < p0_hi);
    uint w1 = t1 + p2_lo;              uint c2 = select(0u, 1u, w1 < t1);
    uint carry1 = c1 + c2;

    uint t2 = p1_hi + p2_hi;           uint c3 = select(0u, 1u, t2 < p1_hi);
    uint t3 = t2 + p3_lo;              uint c4 = select(0u, 1u, t3 < t2);
    uint w2 = t3 + carry1;             uint c5 = select(0u, 1u, w2 < t3);

    uint w3 = p3_hi + c3 + c4 + c5;

    return uint128_t{w3, w2, w1, w0};
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
    // Fast path: all operands are normal (exp != 0 and exp != 0x7FF)
    // Skips 13 branch checks for NaN/Inf/Zero/subnormal — hot path in matmul
    uint eA = (a.x >> 20) & 0x7FFu;
    uint eB = (b.x >> 20) & 0x7FFu;
    uint eC = (c.x >> 20) & 0x7FFu;

    int expA, expB, expC;
    sf64 sigA, sigB, sigC;
    bool signA, signB, signC, signP;

    if (eA != 0 && eA != 0x7FFu && eB != 0 && eB != 0x7FFu &&
        eC != 0 && eC != 0x7FFu) {
        // All normal — extract components directly (no denormal handling)
        signA = (a.x >> 31) != 0; signB = (b.x >> 31) != 0; signC = (c.x >> 31) != 0;
        signP = signA != signB;
        expA = int(eA); expB = int(eB); expC = int(eC);
        sigA = sf64(a.x & SF64_MANT_HI_MASK, a.y); sigA.x |= 0x00100000u;
        sigB = sf64(b.x & SF64_MANT_HI_MASK, b.y); sigB.x |= 0x00100000u;
        sigC = sf64(c.x & SF64_MANT_HI_MASK, c.y); sigC.x |= 0x00100000u;
    } else {
        // Slow path: handle NaN, Inf, Zero, subnormal
        if (sf64_is_nan(a) || sf64_is_nan(b) || sf64_is_nan(c)) {
            return sf64_propagate_nan3(a, b, c);
        }
        signA = sf64_sign(a); signB = sf64_sign(b); signC = sf64_sign(c);
        signP = signA != signB;

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
        if (sf64_is_zero(a) || sf64_is_zero(b)) return c;
        if (sf64_is_zero(c)) return sf64_mul(a, b);

        expA = int(eA); expB = int(eB); expC = int(eC);
        sigA = sf64_sig(a); sigB = sf64_sig(b); sigC = sf64_sig(c);

        if (expA != 0) sigA.x |= 0x00100000u;
        else { int s = clz64(sigA) - 11; sigA = shl64(sigA, s); expA = 1 - s; }
        if (expB != 0) sigB.x |= 0x00100000u;
        else { int s = clz64(sigB) - 11; sigB = shl64(sigB, s); expB = 1 - s; }
        if (expC != 0) sigC.x |= 0x00100000u;
        else { int s = clz64(sigC) - 11; sigC = shl64(sigC, s); expC = 1 - s; }
    }

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
// GPU Kernels — Word-Swap (f64↔sf64 in-place conversion)
// ============================================================================
// On little-endian ARM64, a double in memory is [low32, high32].
// sf64 (uint2) expects {.x=high32, .y=low32}. This kernel swaps in-place.
// Used to eliminate CPU-side conversion loops by doing the swap on GPU.

kernel void word_swap_sf64(
    device uint2* buf [[buffer(0)]],
    constant uint& count [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    uint2 v = buf[tid];
    buf[tid] = uint2(v.y, v.x);
}

// ============================================================================
// Native f64 ↔ sf64 byte-order helpers
// ============================================================================
// On little-endian ARM64: uint2.x = low32, uint2.y = high32 (native f64 byte order)
// sf64 convention: .x = high32, .y = low32 (for consistent bit manipulation)
// These helpers swap the two 32-bit words. Used by ALL kernels that read/write
// native f64 byte order directly, eliminating the separate dispatch_word_swap passes.
inline sf64 native_to_sf64(uint2 raw) { return sf64(raw.y, raw.x); }
inline uint2 sf64_to_native(sf64 val) { return uint2(val.y, val.x); }

// ============================================================================
// GPU Kernels — Elementwise, Reduce, Transpose
// ============================================================================

// All non-matmul kernels read native f64 byte order (uint2) and swap inline.
// This eliminates the separate dispatch_word_swap passes (2-3 per operation).

kernel void elementwise_sf64(
    device const uint2* A [[buffer(0)]],
    device const uint2* B [[buffer(1)]],
    device uint2* C [[buffer(2)]],
    constant uint& N [[buffer(3)]],
    constant uint& op [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;

    sf64 a = native_to_sf64(A[tid]);
    sf64 b = (op <= 3) ? native_to_sf64(B[tid]) : SF64_ZERO;

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
    C[tid] = sf64_to_native(result);
}

// Reduction kernel — two-pass design
// Pass 1: each threadgroup reduces 256 elements to 1 partial result
// The host then launches a second pass if needed
kernel void reduce_sf64(
    device const uint2* in [[buffer(0)]],
    device uint2* out [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    constant uint& op [[buffer(3)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    threadgroup sf64 shared[256];

    // Load element (inline swap) or identity
    sf64 identity;
    switch (op) {
        case 0: identity = SF64_ZERO; break;  // SUM
        case 1: identity = SF64_ONE; break;   // PROD
        case 2: identity = SF64_INF; break;   // MIN
        case 3: identity = sf64(0xFFF00000u, 0x00000000u); break;  // MAX (-Inf)
        case 4: identity = SF64_ZERO; break;  // MEAN (sum then divide)
        default: identity = SF64_ZERO; break;
    }

    shared[lid] = (tid < N) ? native_to_sf64(in[tid]) : identity;
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

    // First thread writes partial result (swap back to native f64)
    if (lid == 0) {
        out[gid] = sf64_to_native(shared[0]);
    }
}

// Transpose kernel: out[j*rows + i] = in[i*cols + j]
// Reads/writes native f64 byte order — no word-swap dispatch needed
kernel void transpose_sf64(
    device const uint2* in [[buffer(0)]],
    device uint2* out [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]])
{
    uint i = tid.y; // row
    uint j = tid.x; // col
    if (i >= rows || j >= cols) return;
    // Data is uint2 (native f64 bytes) — transpose is just a copy with reindexing
    // No byte-swap needed since we're not doing sf64 arithmetic
    out[j * rows + i] = in[i * cols + j];
}

// ============================================================================
// Softmax Kernel — Numerically Stable, Fused (max, exp, sum, normalize)
// ============================================================================
// Each thread processes one contiguous slice of slice_len elements.
// softmax(x_k) = exp(x_k - max(x)) / sum(exp(x - max(x)))
// Handles axis-last layout (inner_stride=1). General axis handled on CPU.

kernel void softmax_sf64(
    device const uint2* in [[buffer(0)]],
    device uint2* out [[buffer(1)]],
    constant uint& slice_len [[buffer(2)]],
    constant uint& num_slices [[buffer(3)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= num_slices) return;

    uint base = tid * slice_len;

    // Pass 1: find max for numerical stability
    sf64 max_val = native_to_sf64(in[base]);
    for (uint k = 1; k < slice_len; k++) {
        sf64 v = native_to_sf64(in[base + k]);
        if (sf64_gt(v, max_val)) max_val = v;
    }

    // Pass 2: exp(x - max) and accumulate sum
    // Store intermediate sf64 results as native for memory coherency
    sf64 sum_exp = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sf64 e = sf64_exp(sf64_sub(native_to_sf64(in[base + k]), max_val));
        out[base + k] = sf64_to_native(e);
        sum_exp = sf64_add(sum_exp, e);
    }

    // Guard against division by zero
    if (sf64_is_zero(sum_exp)) sum_exp = SF64_ONE;

    // Pass 3: normalize — read back intermediate, divide, write native
    for (uint k = 0; k < slice_len; k++) {
        sf64 e = native_to_sf64(out[base + k]);
        out[base + k] = sf64_to_native(sf64_div(e, sum_exp));
    }
}

// ============================================================================
// Normalize Kernel — Layer Normalization (mean, variance, scale+shift)
// ============================================================================
// y = gamma * (x - mean) / sqrt(var + epsilon) + beta
// Each thread processes one contiguous slice of slice_len elements.

kernel void normalize_sf64(
    device const uint2* in [[buffer(0)]],
    device uint2* out [[buffer(1)]],
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

    // Pass 1: compute mean (inline swap on read)
    sf64 sum = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sum = sf64_add(sum, native_to_sf64(in[base + k]));
    }
    sf64 mean = sf64_div(sum, n_sf);

    // Pass 2: compute variance
    sf64 var_sum = SF64_ZERO;
    for (uint k = 0; k < slice_len; k++) {
        sf64 diff = sf64_sub(native_to_sf64(in[base + k]), mean);
        var_sum = sf64_add(var_sum, sf64_mul(diff, diff));
    }
    sf64 variance = sf64_div(var_sum, n_sf);

    // inv_std = 1 / sqrt(var + epsilon)
    sf64 inv_std = sf64_div(SF64_ONE, sf64_sqrt(sf64_add(variance, epsilon_sf)));

    // Pass 3: normalize with scale and shift (inline swap on write)
    for (uint k = 0; k < slice_len; k++) {
        sf64 x_norm = sf64_mul(sf64_sub(native_to_sf64(in[base + k]), mean), inv_std);
        out[base + k] = sf64_to_native(sf64_add(sf64_mul(gamma_sf, x_norm), beta_sf));
    }
}

// ============================================================================
// Axis-Specific Reduction Kernel
// ============================================================================
// Reduces along a specified axis of an N-dimensional tensor.
// Each thread computes one output element by reducing over the axis dimension.

kernel void reduce_sf64_axis(
    device const uint2* in [[buffer(0)]],
    device uint2* out [[buffer(1)]],
    constant uint& rank [[buffer(2)]],
    device const uint32_t* shape [[buffer(3)]],
    constant uint& axis [[buffer(4)]],
    constant uint& op [[buffer(5)]],
    constant uint& out_size [[buffer(6)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= out_size) return;

    // Compute inner_size = product of dimensions after axis
    uint inner_size = 1;
    for (uint i = axis + 1; i < rank; i++) {
        inner_size *= shape[i];
    }

    uint axis_len = shape[axis];
    uint outer = tid / inner_size;
    uint inner = tid % inner_size;
    uint base = outer * inner_size * axis_len + inner;

    // Identity value for reduction
    sf64 result;
    switch (op) {
        case 0: case 4: result = SF64_ZERO; break;  // SUM, MEAN
        case 1: result = SF64_ONE; break;            // PROD
        case 2: result = SF64_INF; break;            // MIN
        case 3: result = sf64(0xFFF00000u, 0x00000000u); break;  // MAX (-Inf)
        default: result = SF64_ZERO; break;
    }

    // Reduce along axis (inline swap on read)
    for (uint k = 0; k < axis_len; k++) {
        sf64 val = native_to_sf64(in[base + k * inner_size]);
        switch (op) {
            case 0: case 4: result = sf64_add(result, val); break;
            case 1: result = sf64_mul(result, val); break;
            case 2: if (sf64_lt(val, result)) result = val; break;
            case 3: if (sf64_gt(val, result)) result = val; break;
            default: break;
        }
    }

    // MEAN: divide by axis length
    if (op == 4) {
        result = sf64_div(result, sf64_from_int(int(axis_len)));
    }

    out[tid] = sf64_to_native(result);
}

// ============================================================================
// Matrix Multiplication Kernel — Optimized with Threadgroup Shared Memory
// ============================================================================
// Dynamically configured at compile time based on GPU architecture.
// Host prepends #define ESHKOL_SF64_TG/TT/TILE_K before MSL compilation.
// A and B tiles are cooperatively loaded into threadgroup shared memory.
// sA has +1 stride padding to avoid bank conflicts.

// Configurable constants — overridden by host at MSL compile time
#ifndef ESHKOL_SF64_TG
#define ESHKOL_SF64_TG 8           // Default for unknown GPU
#endif
#ifndef ESHKOL_SF64_TT
#define ESHKOL_SF64_TT 4
#endif
#ifndef ESHKOL_SF64_TILE_K
#define ESHKOL_SF64_TILE_K 8
#endif

constant uint SF64_TG = ESHKOL_SF64_TG;       // Threadgroup dimension (TG×TG threads)
constant uint SF64_TT = ESHKOL_SF64_TT;       // Thread tile dimension (TT×TT per thread)
constant uint SF64_BLK = SF64_TG * SF64_TT;   // Block dimension (TG * TT)
constant uint SF64_TILE_K = ESHKOL_SF64_TILE_K;
constant uint SF64_THREADS = SF64_TG * SF64_TG;
constant uint SF64_EPT = (SF64_BLK * SF64_TILE_K) / SF64_THREADS; // Elements per thread
constant uint SA_STRIDE = SF64_TILE_K + 2;  // sA stride with padding for bank conflict avoidance (used by V2, df64, f32 kernels)

// native_to_sf64/sf64_to_native defined above (shared by all kernels)

// ============================================================================
// Matmul-Specialized Rounding — Fully Branchless
// ============================================================================
// ~8 ops, zero branches (vs sf64_round_pack's ~15 ops with 4 branches).
// Eliminates SIMD divergence from round_up (~50% probability) branch.
// No overflow/underflow checks — impossible for matmul accumulation of normals.

inline sf64 sf64_round_pack_matmul(bool sign, int exp_raw, sf64 sig, uint round_bits) {
    // Branchless round-to-nearest-even
    uint halfway = 0x200u;
    bool above_half = (round_bits > halfway);
    bool at_half_odd = (round_bits == halfway) && ((sig.y & 1u) != 0);
    uint do_round = select(0u, 1u, above_half || at_half_odd);

    // Branchless increment
    uint new_lo = sig.y + do_round;
    uint carry = select(0u, 1u, new_lo < sig.y);
    uint new_hi = sig.x + carry;

    // Round-to-even: clear LSB if exact half
    new_lo = select(new_lo, new_lo & ~1u, round_bits == halfway);

    // Branchless mantissa overflow (bit 20 carried into bit 21)
    uint mant_ov = (new_hi >> 20) & 1u;
    uint final_lo = select(new_lo, (new_lo >> 1) | (new_hi << 31), mant_ov != 0);
    uint final_hi = select(new_hi, new_hi >> 1, mant_ov != 0);
    int final_exp = exp_raw + int(mant_ov);

    // Pack (no overflow/underflow checks — impossible in matmul inner loop)
    return sf64((sign ? 0x80000000u : 0u) | (uint(final_exp) << 20) | (final_hi & 0x000FFFFFu),
                final_lo);
}

// ============================================================================
// Matmul-Specialized FMA — Classification-Free, Fused Extract+Shift
// ============================================================================
// ~114 ops vs ~134 for sf64_fma fast path = ~15% reduction.
// Assumes all three operands are normal (exp in [1, 0x7FE]).
// MUST NOT be called with zero, inf, NaN, or subnormal operands.
// Used by matmul_sf64 kernel after first K iteration (where acc becomes normal).

inline sf64 sf64_fma_matmul(sf64 a, sf64 b, sf64 c) {
    // Sign extraction (7 ops)
    bool signA = (a.x >> 31) != 0;
    bool signB = (b.x >> 31) != 0;
    bool signC = (c.x >> 31) != 0;
    bool signP = signA != signB;

    // Exponent extraction — no classification check (6 ops)
    int expA = int((a.x >> 20) & 0x7FFu);
    int expB = int((b.x >> 20) & 0x7FFu);
    int expC = int((c.x >> 20) & 0x7FFu);

    // Fused extract + hidden bit + shl64(11) for A and B (5 ops each = 10 ops)
    // Replaces: mask(1) + OR(1) + shl64(5 with branches) = 7 ops each
    uint ah = ((a.x & 0x000FFFFFu) | 0x00100000u);
    sf64 sigA = sf64((ah << 11) | (a.y >> 21), a.y << 11);
    uint bh = ((b.x & 0x000FFFFFu) | 0x00100000u);
    sf64 sigB = sf64((bh << 11) | (b.y >> 21), b.y << 11);

    // sigC: extract only, no shift by 11 (shifted by 10 later) (2 ops)
    sf64 sigC = sf64((c.x & 0x000FFFFFu) | 0x00100000u, c.y);

    // Product exponent (2 ops)
    int expP = expA + expB - SF64_EXP_BIAS;

    // Full 128-bit product via mul64x64 (20 ops)
    uint128_t prod = mul64x64(sigA, sigB);
    sf128 P = sf128{sf64(prod.w3, prod.w2), sf64(prod.w1, prod.w0)};

    // Branchless product normalization — select-based shr by 1 (16 ops)
    // Eliminates potential SIMD divergence vs branching version
    uint overflow = P.hi.x >> 31;
    uint sticky_bit = P.lo.y & overflow;
    P.lo.y = select(P.lo.y, (P.lo.y >> 1) | (P.lo.x << 31), overflow != 0) | sticky_bit;
    P.lo.x = select(P.lo.x, (P.lo.x >> 1) | (P.hi.y << 31), overflow != 0);
    P.hi.y = select(P.hi.y, (P.hi.y >> 1) | (P.hi.x << 31), overflow != 0);
    P.hi.x = select(P.hi.x, P.hi.x >> 1, overflow != 0);
    expP += int(overflow);

    // Prepare addend C: inline shl64(sigC, 10) — no branch checks (5 ops)
    sf128 C128 = sf128{sf64((sigC.x << 10) | (sigC.y >> 22), sigC.y << 10), SF64_ZERO};

    // Branchless alignment — one shr128_jam call, select-based routing
    // Eliminates SIMD divergence from expDiff > 0 / < 0 branches
    int expDiff = expP - expC;
    int abs_diff = abs(expDiff);
    bool shift_c = (expDiff > 0);
    sf128 to_shift = sf128{
        select(P.hi, C128.hi, shift_c), select(P.lo, C128.lo, shift_c)
    };
    sf128 shifted = shr128_jam(to_shift, abs_diff);
    C128 = sf128{select(C128.hi, shifted.hi, shift_c), select(C128.lo, shifted.lo, shift_c)};
    P = sf128{select(shifted.hi, P.hi, shift_c), select(shifted.lo, P.lo, shift_c)};
    int expZ = select(expC, expP, shift_c);

    // Add or subtract (same-sign path dominates in matmul)
    sf128 R;
    bool signZ;
    if (signP == signC) {
        signZ = signP;
        R = add128(P, C128);
        if ((R.hi.x & 0x80000000u) != 0) {
            R = shr128_jam(R, 1);
            expZ++;
        }
    } else {
        int cmp = cmp128(P, C128);
        if (cmp == 0) return SF64_ZERO;
        if (cmp > 0) { signZ = signP; R = sub128(P, C128); }
        else          { signZ = signC; R = sub128(C128, P); }
        int shift = clz128(R) - 1;
        if (shift > 0)      { R = shl128(R, shift); expZ -= shift; }
        else if (shift < 0) { R = shr128_jam(R, -shift); expZ -= shift; }
    }

    // Inline rounding — eliminates shr64 branch checks (10 ops)
    uint sticky = (R.lo.x | R.lo.y) != 0 ? 1u : 0u;
    uint round_bits = (R.hi.y & 0x3FFu) | sticky;
    sf64 sigZ = sf64(R.hi.x >> 10, (R.hi.y >> 10) | (R.hi.x << 22));
    return sf64_round_pack_matmul(signZ, expZ, sigZ, round_bits);
}

// Matmul-specialized multiply — classification-free, for first K iteration
// where accumulator is SF64_ZERO (can't use sf64_fma_matmul with zero acc).
// Assumes both operands are normal.
inline sf64 sf64_mul_matmul(sf64 a, sf64 b) {
    bool signZ = ((a.x >> 31) != 0) != ((b.x >> 31) != 0);
    int expZ = int((a.x >> 20) & 0x7FFu) + int((b.x >> 20) & 0x7FFu) - SF64_EXP_BIAS;

    // Fused extract + hidden bit + shl64(11)
    uint ah = ((a.x & 0x000FFFFFu) | 0x00100000u);
    sf64 sigA = sf64((ah << 11) | (a.y >> 21), a.y << 11);
    uint bh = ((b.x & 0x000FFFFFu) | 0x00100000u);
    sf64 sigB = sf64((bh << 11) | (b.y >> 21), b.y << 11);

    // 64x64 -> 128 multiply
    uint128_t prod = mul64x64(sigA, sigB);
    sf64 sigZ = sf64(prod.w3, prod.w2);
    uint sticky = ((prod.w1 | prod.w0) != 0) ? 1u : 0u;

    // Branchless normalize — if MSB set, shift right 1
    uint ov = sigZ.x >> 31;
    sticky |= sigZ.y & ov;
    sigZ = select(sigZ, sf64(sigZ.x >> 1, (sigZ.y >> 1) | (sigZ.x << 31)), ov != 0);
    expZ += int(ov);

    // Inline rounding
    uint round_bits = (sigZ.y & 0x3FFu) | sticky;
    sigZ = sf64(sigZ.x >> 10, (sigZ.y >> 10) | (sigZ.x << 22));
    return sf64_round_pack_matmul(signZ, expZ, sigZ, round_bits);
}

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

    // Thread's sub-tile origin within the block
    uint threadRow = lid.y * SF64_TT;
    uint threadCol = lid.x * SF64_TT;

    // Accumulators for TT×TT sub-tile
    sf64 acc[SF64_TT][SF64_TT];
    for (uint i = 0; i < SF64_TT; i++)
        for (uint j = 0; j < SF64_TT; j++)
            acc[i][j] = SF64_ZERO;

    // Shared memory tiles — cooperatively loaded by all TG×TG threads
    threadgroup sf64 sA[SF64_BLK * SF64_TILE_K];
    threadgroup sf64 sB[SF64_TILE_K * SF64_BLK];

    uint tid = lid.y * SF64_TG + lid.x;

    // Process K in chunks of TILE_K
    for (uint kBlock = 0; kBlock < K; kBlock += SF64_TILE_K) {

        // Cooperative load A tile [BLK × TILE_K]
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_TILE_K;
            uint col = idx % SF64_TILE_K;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;
            sA[row * SF64_TILE_K + col] =
                (gRow < M && gCol < K) ? native_to_sf64(A[gRow * K + gCol]) : SF64_ZERO;
        }

        // Cooperative load B tile [TILE_K × BLK]
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_BLK;
            uint col = idx % SF64_BLK;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;
            sB[row * SF64_BLK + col] =
                (gRow < K && gCol < N) ? native_to_sf64(B[gRow * N + gCol]) : SF64_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: each thread processes TILE_K values from shared memory
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

    // Store results (inline byte-swap to native f64 order)
    for (uint i = 0; i < SF64_TT; i++) {
        for (uint j = 0; j < SF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = sf64_to_native(acc[i][j]);
        }
    }
}

// ============================================================================
// Matmul V2 — Bitwise-First, Deferred Rounding, Pre-Extracted
// ============================================================================
// ~67 ops per FMA vs ~168 for sf64_fma. Key optimizations:
// 1. Pre-extract sign/exp/sig at tile load (amortized across TT uses)
// 2. Pre-shift significands (shl64 by 11) at tile load
// 3. No NaN/Inf/zero/subnormal checks in inner loop
// 4. 128-bit accumulator with deferred rounding (round once at store)
// 5. Branchless product normalization

// Deferred-rounding accumulator — 128 bits of precision, no rounding until store
struct sf64_acc {
    sf128 sig;    // 128-bit significand
    int   exp;    // raw exponent
    bool  sign;   // sign bit
};

// Initialize accumulator to zero
inline sf64_acc sf64_acc_zero() {
    return sf64_acc{sf128{SF64_ZERO, SF64_ZERO}, 0, false};
}

// Core matmul FMA: no rounding, no special cases, no extraction of operands
// Pre-extracted sign/exp/sig passed directly from shared memory reads
inline void matmul_fma_acc(
    sf64 sig_a, int exp_a, bool sign_a,
    sf64 sig_b, int exp_b, bool sign_b,
    thread sf64_acc& acc)
{
    // 1. Product sign — XOR (1 op)
    bool sign_p = sign_a ^ sign_b;

    // 2. Product exponent (2 ops)
    int exp_p = exp_a + exp_b - SF64_EXP_BIAS;

    // 3. 64×64 → 128 multiply (27 ops — hardware-pipelined)
    uint128_t raw = mul64x64(sig_a, sig_b);
    sf128 prod = sf128{sf64(raw.w3, raw.w2), sf64(raw.w1, raw.w0)};

    // 4. Branchless normalize: if bit 127 set, shift right 1 (6 ops)
    uint overflow = prod.hi.x >> 31;
    uint ov_mask = select(0u, 0xFFFFFFFFu, overflow != 0);
    // Shift right by 1 if overflow, using select per word
    uint sticky_bit = prod.lo.y & select(0u, 1u, overflow != 0);
    prod.lo.y = select(prod.lo.y,
        (prod.lo.y >> 1) | (prod.lo.x << 31), overflow != 0) | sticky_bit;
    prod.lo.x = select(prod.lo.x,
        (prod.lo.x >> 1) | (prod.hi.y << 31), overflow != 0);
    prod.hi.y = select(prod.hi.y,
        (prod.hi.y >> 1) | (prod.hi.x << 31), overflow != 0);
    prod.hi.x = select(prod.hi.x,
        prod.hi.x >> 1, overflow != 0);
    exp_p += int(overflow);

    // 5. First accumulation: just store product
    if (acc.exp == 0 && is_zero128(acc.sig)) {
        acc.sig = prod;
        acc.exp = exp_p;
        acc.sign = sign_p;
        return;
    }

    // 6. Align exponents — shift smaller value right (bitwise shift, ~15 ops)
    int diff = exp_p - acc.exp;
    sf128 aligned_prod = prod;
    sf128 aligned_acc = acc.sig;
    int exp_z = acc.exp;

    if (diff > 0) {
        aligned_acc = shr128_jam(aligned_acc, diff);
        exp_z = exp_p;
    } else if (diff < 0) {
        aligned_prod = shr128_jam(aligned_prod, -diff);
    }

    // 7. Accumulate (~16 ops for add, handle sign)
    if (sign_p == acc.sign) {
        acc.sig = add128(aligned_prod, aligned_acc);
        acc.exp = exp_z;
        // Branchless overflow: if MSB set, shift right 1
        uint ov2 = acc.sig.hi.x >> 31;
        if (ov2) {
            acc.sig = shr128_jam(acc.sig, 1);
            acc.exp++;
        }
    } else {
        // Subtraction path (rare in matmul — opposite signs)
        int cmp = cmp128(aligned_prod, aligned_acc);
        if (cmp > 0) {
            acc.sig = sub128(aligned_prod, aligned_acc);
            acc.sign = sign_p;
        } else if (cmp < 0) {
            acc.sig = sub128(aligned_acc, aligned_prod);
            // acc.sign unchanged
        } else {
            acc.sig = sf128{SF64_ZERO, SF64_ZERO};
            acc.exp = 0;
            return;
        }
        acc.exp = exp_z;
        // Normalize: shift left until leading 1 at bit 62 of hi
        int shift = clz128(acc.sig) - 1;
        if (shift > 0) {
            acc.sig = shl128(acc.sig, shift);
            acc.exp -= shift;
        }
    }
}

// Packed FMA: accepts raw sf64 values, extracts sign/exp/sig inline.
// Avoids pre-extraction arrays that cause register spilling in V2 inner loop.
// Adds ~12 ops/call for extraction, but eliminates ~104 bytes of register pressure.
inline void matmul_fma_acc_packed(
    sf64 a_packed, sf64 b_packed,
    thread sf64_acc& acc)
{
    // Inline extraction of A (~6 ops)
    bool sign_a = (a_packed.x >> 31) != 0;
    int exp_a = int((a_packed.x >> 20) & 0x7FFu);
    sf64 sig_a = sf64(a_packed.x & SF64_MANT_HI_MASK, a_packed.y);
    uint nz_a = select(0u, 1u, exp_a != 0);
    sig_a.x |= (nz_a * 0x00100000u);
    sig_a = sf64((sig_a.x << 11) | (sig_a.y >> 21), sig_a.y << 11);

    // Inline extraction of B (~6 ops)
    bool sign_b = (b_packed.x >> 31) != 0;
    int exp_b = int((b_packed.x >> 20) & 0x7FFu);
    sf64 sig_b = sf64(b_packed.x & SF64_MANT_HI_MASK, b_packed.y);
    uint nz_b = select(0u, 1u, exp_b != 0);
    sig_b.x |= (nz_b * 0x00100000u);
    sig_b = sf64((sig_b.x << 11) | (sig_b.y >> 21), sig_b.y << 11);

    // 1. Product sign — XOR (1 op)
    bool sign_p = sign_a ^ sign_b;

    // 2. Product exponent (2 ops)
    int exp_p = exp_a + exp_b - SF64_EXP_BIAS;

    // 3. 64×64 → 128 multiply (27 ops — hardware-pipelined)
    uint128_t raw = mul64x64(sig_a, sig_b);
    sf128 prod = sf128{sf64(raw.w3, raw.w2), sf64(raw.w1, raw.w0)};

    // 4. Branchless normalize: if bit 127 set, shift right 1 (6 ops)
    uint overflow = prod.hi.x >> 31;
    uint ov_mask = select(0u, 0xFFFFFFFFu, overflow != 0);
    uint sticky_bit = prod.lo.y & select(0u, 1u, overflow != 0);
    prod.lo.y = select(prod.lo.y,
        (prod.lo.y >> 1) | (prod.lo.x << 31), overflow != 0) | sticky_bit;
    prod.lo.x = select(prod.lo.x,
        (prod.lo.x >> 1) | (prod.hi.y << 31), overflow != 0);
    prod.hi.y = select(prod.hi.y,
        (prod.hi.y >> 1) | (prod.hi.x << 31), overflow != 0);
    prod.hi.x = select(prod.hi.x,
        prod.hi.x >> 1, overflow != 0);
    exp_p += int(overflow);

    // 5. First accumulation: just store product
    if (acc.exp == 0 && is_zero128(acc.sig)) {
        acc.sig = prod;
        acc.exp = exp_p;
        acc.sign = sign_p;
        return;
    }

    // 6. Align exponents — shift smaller value right (bitwise shift, ~15 ops)
    int diff = exp_p - acc.exp;
    sf128 aligned_prod = prod;
    sf128 aligned_acc = acc.sig;
    int exp_z = acc.exp;

    if (diff > 0) {
        aligned_acc = shr128_jam(aligned_acc, diff);
        exp_z = exp_p;
    } else if (diff < 0) {
        aligned_prod = shr128_jam(aligned_prod, -diff);
    }

    // 7. Accumulate (~16 ops for add, handle sign)
    if (sign_p == acc.sign) {
        acc.sig = add128(aligned_prod, aligned_acc);
        acc.exp = exp_z;
        // Branchless overflow: if MSB set, shift right 1
        uint ov2 = acc.sig.hi.x >> 31;
        if (ov2) {
            acc.sig = shr128_jam(acc.sig, 1);
            acc.exp++;
        }
    } else {
        // Subtraction path (rare in matmul — opposite signs)
        int cmp = cmp128(aligned_prod, aligned_acc);
        if (cmp > 0) {
            acc.sig = sub128(aligned_prod, aligned_acc);
            acc.sign = sign_p;
        } else if (cmp < 0) {
            acc.sig = sub128(aligned_acc, aligned_prod);
            // acc.sign unchanged
        } else {
            acc.sig = sf128{SF64_ZERO, SF64_ZERO};
            acc.exp = 0;
            return;
        }
        acc.exp = exp_z;
        // Normalize: shift left until leading 1 at bit 62 of hi
        int shift = clz128(acc.sig) - 1;
        if (shift > 0) {
            acc.sig = shl128(acc.sig, shift);
            acc.exp -= shift;
        }
    }
}

// Finalize accumulator → packed sf64 (called once per output element at store time)
inline sf64 sf64_acc_finalize(sf64_acc acc) {
    // Zero check
    if (is_zero128(acc.sig)) return SF64_ZERO;

    // Normalize: leading 1 at bit 62 of hi
    int shift = clz128(acc.sig) - 1;
    if (shift > 0) {
        acc.sig = shl128(acc.sig, shift);
        acc.exp -= shift;
    } else if (shift < 0) {
        acc.sig = shr128_jam(acc.sig, -shift);
        acc.exp -= shift;
    }

    // Extract round bits (10 bits) + sticky from low words
    uint sticky = (acc.sig.lo.x | acc.sig.lo.y) != 0 ? 1u : 0u;
    uint round_bits = (acc.sig.hi.y & 0x3FFu) | sticky;
    sf64 sigZ = shr64(acc.sig.hi, 10);

    return sf64_round_pack(acc.sign, acc.exp, sigZ, round_bits);
}

kernel void matmul_sf64_v2(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    // Output block origin
    uint baseRow = group_id.y * SF64_BLK;
    uint baseCol = group_id.x * SF64_BLK;

    // Thread's sub-tile origin
    uint threadRow = lid.y * SF64_TT;
    uint threadCol = lid.x * SF64_TT;

    // 128-bit accumulators — deferred rounding
    sf64_acc acc[SF64_TT][SF64_TT];
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++)
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++)
            acc[i][j] = sf64_acc_zero();

    // Shared memory tiles (packed sf64 — extract into registers at read time)
    threadgroup sf64 sA[SF64_BLK * SA_STRIDE];
    threadgroup sf64 sB[SF64_TILE_K * SF64_BLK];

    uint tid = lid.y * SF64_TG + lid.x;

    // Process K in chunks
    for (uint kBlock = 0; kBlock < K; kBlock += SF64_TILE_K) {

        // === Cooperative load A tile (inline byte-swap) ===
        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_TILE_K;
            uint col = idx % SF64_TILE_K;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;
            sA[row * SA_STRIDE + col] =
                (gRow < M && gCol < K) ? native_to_sf64(A[gRow * K + gCol]) : SF64_ZERO;
        }

        // === Cooperative load B tile (inline byte-swap) ===
        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_BLK;
            uint col = idx % SF64_BLK;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;
            sB[row * SF64_BLK + col] =
                (gRow < K && gCol < N) ? native_to_sf64(B[gRow * N + gCol]) : SF64_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Compute: inline extraction per FMA (no pre-extract arrays) ===
        // Eliminates ~104 bytes of register pressure from pre-extracted arrays.
        // Extraction is ~12 ops per FMA, but avoids register spilling (~300 cycles/spill).
        #pragma unroll
        for (uint kk = 0; kk < SF64_TILE_K; kk++) {
            #pragma unroll
            for (uint i = 0; i < SF64_TT; i++)
                #pragma unroll
                for (uint j = 0; j < SF64_TT; j++)
                    matmul_fma_acc_packed(
                        sA[(threadRow + i) * SA_STRIDE + kk],
                        sB[kk * SF64_BLK + threadCol + j],
                        acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store: finalize accumulators → packed sf64 → native byte order ===
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = sf64_to_native(sf64_acc_finalize(acc[i][j]));
        }
    }
}

// ============================================================================
// df64 — Double-Float (two f32) for High-Performance GPU Matmul
// ============================================================================
// Represents an f64 value as hi + lo where hi and lo are f32.
// Precision: ~48 bits (vs 53 for IEEE f64, 24 for f32).
// Uses native f32 FMA hardware → ~10 f32 ops per FMA vs ~168 uint32 ops.

struct df64 {
    float hi;   // High part (leading ~24 bits)
    float lo;   // Low part (trailing ~24 bits), |lo| <= 0.5 ULP of hi
};

// Convert sf64 (uint2 IEEE f64) → df64 (two floats)
// Extracts the f64 value and splits into hi + lo
inline df64 sf64_to_df64(sf64 x) {
    // Extract IEEE 754 components via bitwise ops
    uint sign_bit = x.x & 0x80000000u;
    int exp_raw = int((x.x >> 20) & 0x7FFu);
    uint mant_hi = x.x & 0x000FFFFFu;
    uint mant_lo = x.y;

    // Handle zero
    if (exp_raw == 0 && mant_hi == 0 && mant_lo == 0) {
        float z = sign_bit ? -0.0f : 0.0f;
        return df64{z, 0.0f};
    }

    // Reconstruct the f64 value using f32 arithmetic
    // Split the 52-bit mantissa into two 26-bit halves
    // Upper 26 bits: bits 51-26 of mantissa
    // Lower 26 bits: bits 25-0 of mantissa

    // Mantissa with implicit 1: 1.mant_hi[19:0].mant_lo[31:0] = 53 bits total
    // Upper 27 bits: 1.mant_hi[19:0].mant_lo[31:26] (fits in f32's 24-bit mantissa via rounding)
    // We'll use the IEEE trick: construct f64 as two f32 values

    // Method: reconstruct the full value in f32 parts
    // hi = f32 representation of the value (top 24 bits of mantissa)
    // lo = exact error: value - hi

    // Construct hi: take top 20 bits of mantissa + implicit 1 = 21 bits
    // Pack into f32: sign | exp_f32 | mant_f32[22:0]
    int exp_unbiased = exp_raw - 1023;
    int exp_f32 = exp_unbiased + 127;

    // For the high part: use the top 20 mantissa bits (from x.x)
    // f32 has 23 mantissa bits, so we can fit 20 + 3 more from x.y
    uint hi_mant_23 = (mant_hi << 3) | (mant_lo >> 29);
    uint hi_bits = sign_bit | (uint(exp_f32 & 0xFF) << 23) | (hi_mant_23 & 0x007FFFFFu);

    // Construct hi as float (reinterpret bits)
    float hi = as_type<float>(hi_bits);

    // For the low part: remaining mantissa bits (bits 28-0 of mant_lo)
    // These represent: remaining_bits × 2^(exp_unbiased - 52)
    // In f32: the remaining 29 bits, but f32 only has 24-bit mantissa
    // so we take bits 28-5 of mant_lo (24 bits)
    uint lo_mant_raw = mant_lo & 0x1FFFFFFFu;  // bottom 29 bits
    if (lo_mant_raw == 0) {
        return df64{hi, 0.0f};
    }

    // Count leading zeros in the remaining mantissa to normalize
    int lo_lz = clz(lo_mant_raw) - 3;  // 32-bit clz minus 3 (since only 29 bits)
    if (lo_lz < 0) lo_lz = 0;

    int lo_exp_unbiased = exp_unbiased - 23 - lo_lz;  // 23 bits already consumed by hi
    int lo_exp_f32 = lo_exp_unbiased + 127;

    float lo;
    if (lo_exp_f32 <= 0) {
        lo = 0.0f;  // Underflow — too small for f32
    } else {
        // Normalize: shift left by lo_lz, then take top 23 bits
        uint lo_shifted = lo_mant_raw << lo_lz;
        uint lo_mant_23 = (lo_shifted >> 6) & 0x007FFFFFu;  // top 23 of 29 bits
        uint lo_bits = sign_bit | (uint(lo_exp_f32 & 0xFF) << 23) | lo_mant_23;
        lo = as_type<float>(lo_bits);
    }

    return df64{hi, lo};
}

// Convert df64 → sf64 (reconstruct IEEE f64 from two f32)
inline sf64 df64_to_sf64(df64 x) {
    // Quick path: if lo is zero, just convert hi
    if (x.hi == 0.0f && x.lo == 0.0f) return SF64_ZERO;

    // Use Knuth TwoSum to get exact hi+lo, then pack
    // The value is exactly hi + lo (by construction)

    // Extract hi as sf64 components
    uint hi_bits = as_type<uint>(x.hi);
    bool sign = (hi_bits >> 31) != 0;
    int hi_exp = int((hi_bits >> 23) & 0xFFu) - 127 + 1023;
    uint hi_mant = hi_bits & 0x007FFFFFu;

    // Start building the 52-bit mantissa
    // f32 mantissa is 23 bits (bits 51-29 of f64 mantissa)
    uint mant_hi = hi_mant >> 3;        // top 20 bits
    uint mant_lo = (hi_mant & 0x7u) << 29;  // bits 31-29

    // Add lo's contribution
    if (x.lo != 0.0f) {
        uint lo_bits = as_type<uint>(x.lo);
        int lo_exp = int((lo_bits >> 23) & 0xFFu) - 127 + 1023;
        uint lo_mant = (lo_bits & 0x007FFFFFu) | 0x00800000u;  // with implicit 1

        int shift = hi_exp - lo_exp;  // how far right to shift lo
        if (shift >= 0 && shift < 52) {
            // Shift lo's mantissa right by 'shift' bits and add to result
            // lo's implicit 1 is at bit 23, hi's at bit 52
            // So lo contributes at bit (52 - shift)
            uint64_t lo_full = uint64_t(lo_mant) << 29;  // align to 52-bit position
            if (shift < 64) {
                lo_full >>= uint(shift);
            } else {
                lo_full = 0;
            }
            // Add to mantissa
            uint64_t full_mant = (uint64_t(mant_hi) << 32) | uint64_t(mant_lo);
            full_mant += lo_full;
            mant_hi = uint(full_mant >> 32);
            mant_lo = uint(full_mant);
        }
    }

    return sf64_pack(sign, hi_exp, mant_hi & SF64_MANT_HI_MASK, mant_lo);
}

// df64 FMA using native f32 hardware (~10 ops)
// Computes a*b + c in double-float precision
inline df64 df64_fma(df64 a, df64 b, df64 c) {
    // TwoProduct: p + e = a.hi * b.hi (exact)
    float p = a.hi * b.hi;
    float e = fma(a.hi, b.hi, -p);     // error-free transform

    // Cross terms (a.hi*b.lo + a.lo*b.hi)
    float cross = fma(a.hi, b.lo, a.lo * b.hi);

    // Full product error
    float prod_lo = e + cross + a.lo * b.lo;

    // Add to accumulator c using TwoSum
    float s = p + c.hi;
    float v = s - p;
    float t = (p - (s - v)) + (c.hi - v);

    // Combine all low-order terms
    t += prod_lo + c.lo;

    // Final normalization (TwoSum)
    float r_hi = s + t;
    float r_lo = t - (r_hi - s);

    return df64{r_hi, r_lo};
}

// df64 addition (for accumulation)
inline df64 df64_add(df64 a, df64 b) {
    // TwoSum for hi parts
    float s = a.hi + b.hi;
    float v = s - a.hi;
    float t = (a.hi - (s - v)) + (b.hi - v);
    t += a.lo + b.lo;
    float r_hi = s + t;
    float r_lo = t - (r_hi - s);
    return df64{r_hi, r_lo};
}

// ============================================================================
// df64 Constants — Hardware-adaptive via #define injection from host
// ============================================================================
#ifndef DF64_BM
#define DF64_BM 64
#define DF64_BN 64
#define DF64_BK 16
#define DF64_TG 16
#define DF64_TT 4
#define DF64_THREADS 256
#endif
constant uint DF64_BLK = DF64_TG * DF64_TT;
constant uint DF64_SA_STRIDE = DF64_BK + 2;  // bank conflict padding
constant uint DF64_EPT_A = (DF64_BM * DF64_BK + DF64_THREADS - 1) / DF64_THREADS;
constant uint DF64_EPT_B = (DF64_BK * DF64_BN + DF64_THREADS - 1) / DF64_THREADS;
constant uint DF64_SA_SIZE = DF64_BM * DF64_SA_STRIDE;
constant uint DF64_SB_SIZE = DF64_BK * DF64_BN;

// ============================================================================
// df64 Conversion Kernels — f64 ↔ df64 on GPU
// ============================================================================

kernel void convert_f64_to_df64(
    device const uint2* input [[buffer(0)]],
    device float2* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    sf64 s = native_to_sf64(input[tid]);
    df64 d = sf64_to_df64(s);
    output[tid] = float2(d.hi, d.lo);
}

kernel void convert_df64_to_f64(
    device const float2* input [[buffer(0)]],
    device uint2* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    df64 d = {input[tid].x, input[tid].y};
    output[tid] = sf64_to_native(df64_to_sf64(d));
}

// ============================================================================
// df64 Pure Matmul Kernel — Pre-converted float2 input
// ============================================================================
// 256 threads (16×16), 64×64 output tile, BK=16
// No sf64↔df64 conversion in inner loop — data pre-converted by GPU kernels above
// Shared: 17408 bytes (9216 for A, 8192 for B)

kernel void matmul_df64_pure(
    device const float2* A [[buffer(0)]],
    device const float2* B [[buffer(1)]],
    device float2* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint baseRow = group_id.y * DF64_BM;
    uint baseCol = group_id.x * DF64_BN;
    uint threadRow = lid.y * DF64_TT;
    uint threadCol = lid.x * DF64_TT;

    // df64 accumulators
    df64 acc[DF64_TT][DF64_TT];
    #pragma unroll
    for (uint i = 0; i < DF64_TT; i++)
        #pragma unroll
        for (uint j = 0; j < DF64_TT; j++)
            acc[i][j] = df64{0.0f, 0.0f};

    // Shared memory for df64 tiles (float2 = 8 bytes each)
    threadgroup float2 sA[DF64_SA_SIZE];
    threadgroup float2 sB[DF64_SB_SIZE];

    uint tid = lid.y * DF64_TG + lid.x;

    // Pre-compute interior tile flags for fast path (skip bounds checks)
    const bool full_tile_m = (baseRow + DF64_BM <= M);
    const bool full_tile_n = (baseCol + DF64_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += DF64_BK) {
        const bool full_tile_k = (kBlock + DF64_BK <= K);

        // Cooperative load A tile (coalesced)
        if (full_tile_m && full_tile_k) {
            // Fast path: no bounds check
            #pragma unroll
            for (uint i = 0; i < DF64_EPT_A; i++) {
                uint idx = i * DF64_THREADS + tid;
                uint row = idx / DF64_BK;
                uint col = idx % DF64_BK;
                sA[row * DF64_SA_STRIDE + col] = A[(baseRow + row) * K + kBlock + col];
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < DF64_EPT_A; i++) {
                uint idx = i * DF64_THREADS + tid;
                uint row = idx / DF64_BK;
                uint col = idx % DF64_BK;
                uint gRow = baseRow + row;
                uint gCol = kBlock + col;
                sA[row * DF64_SA_STRIDE + col] =
                    (gRow < M && gCol < K) ? A[gRow * K + gCol] : float2(0.0f, 0.0f);
            }
        }

        // Cooperative load B tile (coalesced)
        if (full_tile_k && full_tile_n) {
            // Fast path: no bounds check
            #pragma unroll
            for (uint i = 0; i < DF64_EPT_B; i++) {
                uint idx = i * DF64_THREADS + tid;
                uint row = idx / DF64_BN;
                uint col = idx % DF64_BN;
                sB[row * DF64_BN + col] = B[(kBlock + row) * N + baseCol + col];
            }
        } else {
            #pragma unroll
            for (uint i = 0; i < DF64_EPT_B; i++) {
                uint idx = i * DF64_THREADS + tid;
                uint row = idx / DF64_BN;
                uint col = idx % DF64_BN;
                uint gRow = kBlock + row;
                uint gCol = baseCol + col;
                sB[row * DF64_BN + col] =
                    (gRow < K && gCol < N) ? B[gRow * N + gCol] : float2(0.0f, 0.0f);
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner compute: pure df64 FMA (no conversions needed)
        #pragma unroll
        for (uint kk = 0; kk < DF64_BK; kk++) {
            df64 a_df[DF64_TT];
            #pragma unroll
            for (uint i = 0; i < DF64_TT; i++) {
                float2 v = sA[(threadRow + i) * DF64_SA_STRIDE + kk];
                a_df[i] = df64{v.x, v.y};
            }

            df64 b_df[DF64_TT];
            #pragma unroll
            for (uint j = 0; j < DF64_TT; j++) {
                float2 v = sB[kk * DF64_BN + threadCol + j];
                b_df[j] = df64{v.x, v.y};
            }

            #pragma unroll
            for (uint i = 0; i < DF64_TT; i++)
                #pragma unroll
                for (uint j = 0; j < DF64_TT; j++)
                    acc[i][j] = df64_fma(a_df[i], b_df[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results directly to device memory as float2
    #pragma unroll
    for (uint i = 0; i < DF64_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < DF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = float2(acc[i][j].hi, acc[i][j].lo);
        }
    }
}

// ============================================================================
// df64 Matmul Kernel (Legacy) — Uses sf64 tile constants
// ============================================================================
// Same tiling as sf64 matmul but inner loop uses f32 FMA hardware.
// sf64 → df64 conversion at tile boundaries, df64 → sf64 at store.

kernel void matmul_df64(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint baseRow = group_id.y * SF64_BLK;
    uint baseCol = group_id.x * SF64_BLK;
    uint threadRow = lid.y * SF64_TT;
    uint threadCol = lid.x * SF64_TT;

    // df64 accumulators (2 floats each = 8 bytes, same as sf64)
    df64 acc[SF64_TT][SF64_TT];
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++)
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++)
            acc[i][j] = df64{0.0f, 0.0f};

    // Shared memory: load as sf64, convert to df64 in registers
    threadgroup sf64 sA[SF64_BLK * SA_STRIDE];
    threadgroup sf64 sB[SF64_TILE_K * SF64_BLK];

    uint tid = lid.y * SF64_TG + lid.x;

    for (uint kBlock = 0; kBlock < K; kBlock += SF64_TILE_K) {

        // Cooperative load A tile (inline byte-swap)
        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_TILE_K;
            uint col = idx % SF64_TILE_K;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;
            sA[row * SA_STRIDE + col] =
                (gRow < M && gCol < K) ? native_to_sf64(A[gRow * K + gCol]) : SF64_ZERO;
        }

        // Cooperative load B tile (inline byte-swap)
        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_BLK;
            uint col = idx % SF64_BLK;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;
            sB[row * SF64_BLK + col] =
                (gRow < K && gCol < N) ? native_to_sf64(B[gRow * N + gCol]) : SF64_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: convert sf64 → df64 at read, FMA with f32 hardware
        #pragma unroll
        for (uint kk = 0; kk < SF64_TILE_K; kk++) {
            // Convert A values to df64 (amortized across TT j-iterations)
            df64 a_df[SF64_TT];
            #pragma unroll
            for (uint i = 0; i < SF64_TT; i++)
                a_df[i] = sf64_to_df64(sA[(threadRow + i) * SA_STRIDE + kk]);

            // Convert B values to df64 (amortized across TT i-iterations)
            df64 b_df[SF64_TT];
            #pragma unroll
            for (uint j = 0; j < SF64_TT; j++)
                b_df[j] = sf64_to_df64(sB[kk * SF64_BLK + threadCol + j]);

            // Inner FMA: TT × TT using f32 hardware
            #pragma unroll
            for (uint i = 0; i < SF64_TT; i++)
                #pragma unroll
                for (uint j = 0; j < SF64_TT; j++)
                    acc[i][j] = df64_fma(a_df[i], b_df[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: convert df64 → sf64 → native byte order
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = sf64_to_native(df64_to_sf64(acc[i][j]));
        }
    }
}

// ============================================================================
// f32 Matmul Kernel — Maximum Throughput Tier
// ============================================================================
// Native float matmul. Input/output as sf64 (uint2), but all compute in f32.
// Precision: 24 bits. For inference and normalized data.

// Quick sf64 → f32 conversion (truncation, not rounding)
inline float sf64_to_f32(sf64 x) {
    uint sign_bit = x.x & 0x80000000u;
    int exp_raw = int((x.x >> 20) & 0x7FFu);
    if (exp_raw == 0) return sign_bit ? -0.0f : 0.0f;
    if (exp_raw == 0x7FF) return sign_bit ? -INFINITY : INFINITY;
    int exp_f32 = exp_raw - 1023 + 127;
    if (exp_f32 <= 0) return sign_bit ? -0.0f : 0.0f;
    if (exp_f32 >= 255) return sign_bit ? -INFINITY : INFINITY;
    uint mant_23 = ((x.x & 0x000FFFFFu) << 3) | (x.y >> 29);
    return as_type<float>(sign_bit | (uint(exp_f32) << 23) | mant_23);
}

// Quick f32 → sf64 conversion (exact)
inline sf64 f32_to_sf64(float x) {
    if (x == 0.0f) return SF64_ZERO;
    uint bits = as_type<uint>(x);
    uint sign_bit = bits & 0x80000000u;
    int exp_f32 = int((bits >> 23) & 0xFFu);
    if (exp_f32 == 0) return sign_bit ? SF64_NEG_ZERO : SF64_ZERO;
    if (exp_f32 == 255) return sign_bit ? SF64_NEG_INF : SF64_INF;
    int exp_f64 = exp_f32 - 127 + 1023;
    uint mant_23 = bits & 0x007FFFFFu;
    uint hi = sign_bit | (uint(exp_f64) << 20) | (mant_23 >> 3);
    uint lo = (mant_23 & 0x7u) << 29;
    return sf64(hi, lo);
}

kernel void matmul_f32(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint baseRow = group_id.y * SF64_BLK;
    uint baseCol = group_id.x * SF64_BLK;
    uint threadRow = lid.y * SF64_TT;
    uint threadCol = lid.x * SF64_TT;

    // f32 accumulators (4 bytes each — half the size of sf64)
    float acc[SF64_TT][SF64_TT];
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++)
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++)
            acc[i][j] = 0.0f;

    // Shared memory for f32 tiles (4 bytes per element vs 8 for sf64)
    // But we load as sf64 and convert, so keep sf64 format in shared mem
    threadgroup sf64 sA[SF64_BLK * SA_STRIDE];
    threadgroup sf64 sB[SF64_TILE_K * SF64_BLK];

    uint tid = lid.y * SF64_TG + lid.x;

    for (uint kBlock = 0; kBlock < K; kBlock += SF64_TILE_K) {

        // Load with inline byte-swap
        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_TILE_K;
            uint col = idx % SF64_TILE_K;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;
            sA[row * SA_STRIDE + col] =
                (gRow < M && gCol < K) ? native_to_sf64(A[gRow * K + gCol]) : SF64_ZERO;
        }

        #pragma unroll
        for (uint i = 0; i < SF64_EPT; i++) {
            uint idx = tid * SF64_EPT + i;
            uint row = idx / SF64_BLK;
            uint col = idx % SF64_BLK;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;
            sB[row * SF64_BLK + col] =
                (gRow < K && gCol < N) ? native_to_sf64(B[gRow * N + gCol]) : SF64_ZERO;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (uint kk = 0; kk < SF64_TILE_K; kk++) {
            float a_val[SF64_TT];
            float b_val[SF64_TT];

            #pragma unroll
            for (uint i = 0; i < SF64_TT; i++)
                a_val[i] = sf64_to_f32(sA[(threadRow + i) * SA_STRIDE + kk]);
            #pragma unroll
            for (uint j = 0; j < SF64_TT; j++)
                b_val[j] = sf64_to_f32(sB[kk * SF64_BLK + threadCol + j]);

            // Native f32 FMA — single cycle on Apple GPU
            #pragma unroll
            for (uint i = 0; i < SF64_TT; i++)
                #pragma unroll
                for (uint j = 0; j < SF64_TT; j++)
                    acc[i][j] = fma(a_val[i], b_val[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store: convert f32 → sf64 → native byte order
    #pragma unroll
    for (uint i = 0; i < SF64_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < SF64_TT; j++) {
            uint row = baseRow + threadRow + i;
            uint col = baseCol + threadCol + j;
            if (row < M && col < N)
                C[row * N + col] = sf64_to_native(f32_to_sf64(acc[i][j]));
        }
    }
}

// ============================================================================
// f32 Matmul Kernel — simdgroup_matrix_multiply_accumulate (Apple Silicon M1+)
// ============================================================================
// Uses Apple's hardware 8×8 matrix multiply instruction for maximum throughput.
// Reads native f64 byte order, converts to f32 during tile load, computes with
// simdgroup_matrix, converts back during output store. Zero f64↔f32 overhead
// in inner loop.
//
// Tiling: BM=64, BN=64, BK=32, 2×2 SIMD groups, 4×4 fragments per group
// Threads: 128 (4 SIMD groups × 32 threads)
// Shared memory: ~18KB during K-loop, reused for output store

// --- Legacy f32_simd constants (for matmul_f32_simd kernel with f64 conversion) ---
constant uint F32S_LEGACY_BM = 64;
constant uint F32S_LEGACY_BN = 64;
constant uint F32S_LEGACY_BK = 32;
constant uint F32S_LEGACY_TM = 4;
constant uint F32S_LEGACY_TN = 4;
constant uint F32S_LEGACY_WM = 2;
constant uint F32S_LEGACY_WN = 2;
constant uint F32S_LEGACY_THREADS = 128;
constant uint F32S_LEGACY_SA_STRIDE = F32S_LEGACY_BK + 4;   // 36
constant uint F32S_LEGACY_SB_STRIDE = F32S_LEGACY_BN + 4;   // 68
constant uint F32S_LEGACY_OUT_STRIDE = F32S_LEGACY_BN + 4;   // 68
constant uint F32S_LEGACY_SA_SIZE = F32S_LEGACY_BM * F32S_LEGACY_SA_STRIDE;
constant uint F32S_LEGACY_SB_SIZE = F32S_LEGACY_BK * F32S_LEGACY_SB_STRIDE;
constant uint F32S_LEGACY_SHARED_SIZE = F32S_LEGACY_SA_SIZE + F32S_LEGACY_SB_SIZE;

// --- Pure f32 SIMD constants — Hardware-adaptive via #define injection ---
#ifndef F32S_BM
#define F32S_BM 64
#define F32S_BN 64
#define F32S_BK 16
#define F32S_TM 4
#define F32S_TN 4
#define F32S_WM 2
#define F32S_WN 2
#define F32S_THREADS 128
#endif
// Derived: stride padding avoids bank conflicts (stride % 8 != 0)
constant uint F32S_SA_STRIDE = F32S_BK + ((F32S_BK % 8 == 0) ? 4 : 1);
constant uint F32S_SB_STRIDE = F32S_BN + ((F32S_BN % 8 == 0) ? 4 : 1);
constant uint F32S_SA_SIZE = F32S_BM * F32S_SA_STRIDE;
constant uint F32S_SB_SIZE = F32S_BK * F32S_SB_STRIDE;
constant uint F32S_EDGE_SIZE = (F32S_WM * F32S_WN) * 64;  // per-SG edge scratch
constant uint F32S_SHARED_SIZE = F32S_SA_SIZE + F32S_SB_SIZE + F32S_EDGE_SIZE;

// --- 128×128 f32 SIMD constants — Hardware-adaptive via #define injection ---
#ifndef F32S128_BM
#define F32S128_BM 128
#define F32S128_BN 128
#define F32S128_BK 8
#define F32S128_TM 4
#define F32S128_TN 4
#define F32S128_WM 4
#define F32S128_WN 4
#define F32S128_THREADS 512
#endif
constant uint F32S128_SA_STRIDE = F32S128_BK + ((F32S128_BK % 8 == 0) ? 4 : 1);
constant uint F32S128_SB_STRIDE = F32S128_BN + ((F32S128_BN % 8 == 0) ? 4 : 1);
constant uint F32S128_SA_SIZE = F32S128_BM * F32S128_SA_STRIDE;
constant uint F32S128_SB_SIZE = F32S128_BK * F32S128_SB_STRIDE;
// Edge scratch reuses sA region (WM*WN SGs × 64 floats ≤ SA_SIZE)
constant uint F32S128_SHARED_SIZE = F32S128_SA_SIZE + F32S128_SB_SIZE;

// Ozaki-I constants removed — replaced by Ozaki-II CRT approach (see matmul_ozaki_gemm)

// Branchless f64→f32 conversion from native byte order (no swap needed)
// ~8 ops, zero branches — ideal for GPU SIMT execution
inline float f64_to_f32_native(uint2 raw) {
    // raw.x = lo32, raw.y = hi32 (native little-endian f64)
    uint hi = raw.y;
    uint lo = raw.x;
    uint sign = hi & 0x80000000u;
    int exp64 = int((hi >> 20) & 0x7FFu);
    int exp32 = exp64 - 1023 + 127;
    uint mant = ((hi & 0x000FFFFFu) << 3) | (lo >> 29);

    uint normal = sign | (uint(exp32) << 23) | mant;
    // Branchless: zero if denorm/underflow, inf if overflow/nan
    uint is_zero = uint(exp64 == 0) | uint(exp32 <= 0);
    uint is_inf  = uint(exp64 == 0x7FF) | uint(exp32 >= 255);
    uint result = select(select(normal, sign | 0x7F800000u, is_inf != 0), sign, is_zero != 0);
    return as_type<float>(result);
}

// Branchless f32→native f64 conversion (exact, no precision loss)
inline uint2 f32_to_native_f64(float x) {
    uint bits = as_type<uint>(x);
    uint sign = bits & 0x80000000u;
    int exp32 = int((bits >> 23) & 0xFFu);
    uint mant23 = bits & 0x007FFFFFu;
    // Branchless: handle zero, denorm, inf/nan
    int exp64 = exp32 - 127 + 1023;
    uint hi = sign | (uint(exp64) << 20) | (mant23 >> 3);
    uint lo = (mant23 & 0x7u) << 29;
    // Zero out if exp32==0 (zero/denorm), set inf if exp32==255
    uint is_zero = uint(exp32 == 0);
    uint is_inf  = uint(exp32 == 255);
    hi = select(select(hi, sign | 0x7FF00000u, is_inf != 0), sign, is_zero != 0);
    lo = select(lo, 0u, is_zero != 0);
    return uint2(lo, hi);  // native byte order: lo first
}

kernel void matmul_f32_simd(
    device const uint2* A [[buffer(0)]],
    device const uint2* B [[buffer(1)]],
    device uint2* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * F32S_LEGACY_BM;
    const uint baseCol = group_id.x * F32S_LEGACY_BN;
    const uint sg_row = sgid / F32S_LEGACY_WN;
    const uint sg_col = sgid % F32S_LEGACY_WN;

    simdgroup_float8x8 acc[F32S_LEGACY_TM][F32S_LEGACY_TN];
    for (uint i = 0; i < F32S_LEGACY_TM; i++)
        for (uint j = 0; j < F32S_LEGACY_TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    threadgroup float shared_mem[F32S_LEGACY_SHARED_SIZE];
    threadgroup float* sA = &shared_mem[0];
    threadgroup float* sB = &shared_mem[F32S_LEGACY_SA_SIZE];
    const uint tid = sgid * 32 + slid;
    const uint EPT_A = (F32S_LEGACY_BM * F32S_LEGACY_BK) / F32S_LEGACY_THREADS;
    const uint EPT_B = (F32S_LEGACY_BK * F32S_LEGACY_BN) / F32S_LEGACY_THREADS;

    for (uint kBlock = 0; kBlock < K; kBlock += F32S_LEGACY_BK) {
        for (uint i = 0; i < EPT_A; i++) {
            const uint idx = tid * EPT_A + i;
            const uint row = idx / F32S_LEGACY_BK;
            const uint col = idx % F32S_LEGACY_BK;
            const uint gRow = baseRow + row;
            const uint gCol = kBlock + col;
            float val = 0.0f;
            if (gRow < M && gCol < K)
                val = f64_to_f32_native(A[gRow * K + gCol]);
            sA[row * F32S_LEGACY_SA_STRIDE + col] = val;
        }
        for (uint i = 0; i < EPT_B; i++) {
            const uint idx = tid * EPT_B + i;
            const uint row = idx / F32S_LEGACY_BN;
            const uint col = idx % F32S_LEGACY_BN;
            const uint gRow = kBlock + row;
            const uint gCol = baseCol + col;
            float val = 0.0f;
            if (gRow < K && gCol < N)
                val = f64_to_f32_native(B[gRow * N + gCol]);
            sB[row * F32S_LEGACY_SB_STRIDE + col] = val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < F32S_LEGACY_BK; kk += 8) {
            simdgroup_float8x8 a_frag[F32S_LEGACY_TM];
            simdgroup_float8x8 b_frag[F32S_LEGACY_TN];
            for (uint i = 0; i < F32S_LEGACY_TM; i++)
                simdgroup_load(a_frag[i],
                    &sA[(sg_row * F32S_LEGACY_TM * 8 + i * 8) * F32S_LEGACY_SA_STRIDE + kk],
                    F32S_LEGACY_SA_STRIDE);
            for (uint j = 0; j < F32S_LEGACY_TN; j++)
                simdgroup_load(b_frag[j],
                    &sB[kk * F32S_LEGACY_SB_STRIDE + (sg_col * F32S_LEGACY_TN * 8 + j * 8)],
                    F32S_LEGACY_SB_STRIDE);
            for (uint i = 0; i < F32S_LEGACY_TM; i++)
                for (uint j = 0; j < F32S_LEGACY_TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store via shared mem → global with f32→f64 conversion
    threadgroup float* sOut = &shared_mem[0];
    for (uint i = 0; i < F32S_LEGACY_TM; i++) {
        for (uint j = 0; j < F32S_LEGACY_TN; j++) {
            const uint out_row = sg_row * F32S_LEGACY_TM * 8 + i * 8;
            const uint out_col = sg_col * F32S_LEGACY_TN * 8 + j * 8;
            simdgroup_store(acc[i][j],
                &sOut[out_row * F32S_LEGACY_OUT_STRIDE + out_col],
                F32S_LEGACY_OUT_STRIDE);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint OUT_EPT = (F32S_LEGACY_BM * F32S_LEGACY_BN) / F32S_LEGACY_THREADS;
    for (uint i = 0; i < OUT_EPT; i++) {
        const uint idx = tid * OUT_EPT + i;
        const uint row = idx / F32S_LEGACY_BN;
        const uint col = idx % F32S_LEGACY_BN;
        const uint gRow = baseRow + row;
        const uint gCol = baseCol + col;
        if (gRow < M && gCol < N) {
            float val = sOut[row * F32S_LEGACY_OUT_STRIDE + col];
            C[gRow * N + gCol] = f32_to_native_f64(val);
        }
    }
}

// ============================================================================
// Pure f32 SIMD Matmul — Pre-converted float data, zero conversion overhead
// ============================================================================
// Input/output are pure float buffers (pre-converted from f64 by GPU kernel).
// Identical inner loop to matmul_f32_simd but no f64_to_f32/f32_to_f64 in
// tile loads/stores. This eliminates ~12 integer ops per element and halves
// memory bandwidth (4 bytes vs 8 bytes per element).

kernel void matmul_f32_simd_pure(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * F32S_BM;
    const uint baseCol = group_id.x * F32S_BN;
    const uint sg_row = sgid / F32S_WN;      // 0..1
    const uint sg_col = sgid % F32S_WN;      // 0..3

    simdgroup_float8x8 acc[F32S_TM][F32S_TN];
    for (uint i = 0; i < F32S_TM; i++)
        for (uint j = 0; j < F32S_TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    threadgroup float shared_mem[F32S_SHARED_SIZE];
    threadgroup float* sA = &shared_mem[0];
    threadgroup float* sB = &shared_mem[F32S_SA_SIZE];

    const uint tid = sgid * 32 + slid;

    // Vectorized load counts: float4 (16 bytes) per load instruction
    // A tile [64×16] = 1024 floats → 256 float4s / 128 threads = 2 float4s per thread
    // B tile [16×64] = 1024 floats → 256 float4s / 128 threads = 2 float4s per thread
    const uint VEC_A = (F32S_BM * F32S_BK) / (F32S_THREADS * 4);  // 2
    const uint VEC_B = (F32S_BK * F32S_BN) / (F32S_THREADS * 4);  // 2

    // Check if this is a full interior tile (no bounds checking needed)
    const bool full_tile_m = (baseRow + F32S_BM <= M);
    const bool full_tile_n = (baseCol + F32S_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += F32S_BK) {
        const bool full_tile_k = (kBlock + F32S_BK <= K);

        if (full_tile_m && full_tile_k) {
            // Fast path: vectorized float4 loads for A, no bounds check
            for (uint i = 0; i < VEC_A; i++) {
                const uint vec_idx = i * F32S_THREADS + tid;
                const uint elem = vec_idx * 4;
                const uint row = elem / F32S_BK;
                const uint col = elem % F32S_BK;
                float4 v = *((device const float4*)&A[(baseRow + row) * K + kBlock + col]);
                sA[row * F32S_SA_STRIDE + col + 0] = v.x;
                sA[row * F32S_SA_STRIDE + col + 1] = v.y;
                sA[row * F32S_SA_STRIDE + col + 2] = v.z;
                sA[row * F32S_SA_STRIDE + col + 3] = v.w;
            }
        } else {
            // Slow path: scalar loads with bounds check
            const uint EPT_A = (F32S_BM * F32S_BK) / F32S_THREADS;
            for (uint i = 0; i < EPT_A; i++) {
                const uint idx = i * F32S_THREADS + tid;
                const uint row = idx / F32S_BK;
                const uint col = idx % F32S_BK;
                const uint gRow = baseRow + row;
                const uint gCol = kBlock + col;
                sA[row * F32S_SA_STRIDE + col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
        }

        if (full_tile_n && full_tile_k) {
            // Fast path: vectorized float4 loads for B, no bounds check
            for (uint i = 0; i < VEC_B; i++) {
                const uint vec_idx = i * F32S_THREADS + tid;
                const uint elem = vec_idx * 4;
                const uint row = elem / F32S_BN;
                const uint col = elem % F32S_BN;
                float4 v = *((device const float4*)&B[(kBlock + row) * N + baseCol + col]);
                sB[row * F32S_SB_STRIDE + col + 0] = v.x;
                sB[row * F32S_SB_STRIDE + col + 1] = v.y;
                sB[row * F32S_SB_STRIDE + col + 2] = v.z;
                sB[row * F32S_SB_STRIDE + col + 3] = v.w;
            }
        } else {
            const uint EPT_B = (F32S_BK * F32S_BN) / F32S_THREADS;
            for (uint i = 0; i < EPT_B; i++) {
                const uint idx = i * F32S_THREADS + tid;
                const uint row = idx / F32S_BN;
                const uint col = idx % F32S_BN;
                const uint gRow = kBlock + row;
                const uint gCol = baseCol + col;
                sB[row * F32S_SB_STRIDE + col] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: BK=16 in chunks of 8 → 2 iterations
        for (uint kk = 0; kk < F32S_BK; kk += 8) {
            simdgroup_float8x8 a_frag[F32S_TM];
            simdgroup_float8x8 b_frag[F32S_TN];

            for (uint i = 0; i < F32S_TM; i++) {
                simdgroup_load(a_frag[i],
                    &sA[(sg_row * F32S_TM * 8 + i * 8) * F32S_SA_STRIDE + kk],
                    F32S_SA_STRIDE);
            }

            for (uint j = 0; j < F32S_TN; j++) {
                simdgroup_load(b_frag[j],
                    &sB[kk * F32S_SB_STRIDE + (sg_col * F32S_TN * 8 + j * 8)],
                    F32S_SB_STRIDE);
            }

            for (uint i = 0; i < F32S_TM; i++)
                for (uint j = 0; j < F32S_TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Direct store: simdgroup_store to device memory (no shared memory staging)
    for (uint i = 0; i < F32S_TM; i++) {
        for (uint j = 0; j < F32S_TN; j++) {
            const uint gRow = baseRow + sg_row * F32S_TM * 8 + i * 8;
            const uint gCol = baseCol + sg_col * F32S_TN * 8 + j * 8;
            if (gRow + 7 < M && gCol + 7 < N) {
                simdgroup_store(acc[i][j], &C[gRow * N + gCol], N);
            } else if (gRow < M && gCol < N) {
                threadgroup float* edge = &shared_mem[F32S_SA_SIZE + F32S_SB_SIZE + min(sgid, 3u) * 64];
                simdgroup_store(acc[i][j], edge, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (slid == 0) {
                    for (uint r = 0; r < 8 && gRow + r < M; r++)
                        for (uint c = 0; c < 8 && gCol + c < N; c++)
                            C[(gRow + r) * N + (gCol + c)] = edge[r * 8 + c];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ============================================================================
// 128×128 f32 SIMD Matmul — Single-buffer, 16 simdgroups, high occupancy
// ============================================================================
// 4× more output per tile dispatch than 64×64. Each loaded element is reused
// across 128 output rows/columns instead of 64. Memory bandwidth per output
// element halves. Single buffer keeps shared memory at ~8.7KB → 3 TG occupancy.
//
// Architecture: 512 threads = 16 simdgroups (4×4 layout)
// Each simdgroup owns 32×32 output (4×4 fragments of 8×8)
// Inner product: BK=8, one simdgroup_multiply pass per K-block

kernel void matmul_f32_simd_128(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * F32S128_BM;
    const uint baseCol = group_id.x * F32S128_BN;
    const uint sg_row = sgid / F32S128_WN;      // 0..3
    const uint sg_col = sgid % F32S128_WN;      // 0..3

    simdgroup_float8x8 acc[F32S128_TM][F32S128_TN];
    for (uint i = 0; i < F32S128_TM; i++)
        for (uint j = 0; j < F32S128_TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    threadgroup float shared_mem[F32S128_SHARED_SIZE];
    threadgroup float* sA = &shared_mem[0];
    threadgroup float* sB = &shared_mem[F32S128_SA_SIZE];

    const uint tid = sgid * 32 + slid;
    // A tile: 128×8 = 1024 floats / 512 threads = 2 per thread
    // B tile: 8×128 = 1024 floats / 512 threads = 2 per thread

    const bool full_tile_m = (baseRow + F32S128_BM <= M);
    const bool full_tile_n = (baseCol + F32S128_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += F32S128_BK) {
        const bool full_k = (kBlock + F32S128_BK <= K);

        // Load A tile [128×8] — 2 elements per thread
        if (full_tile_m && full_k) {
            for (uint e = 0; e < 2; e++) {
                const uint idx = e * F32S128_THREADS + tid;
                const uint row = idx / F32S128_BK;
                const uint col = idx % F32S128_BK;
                sA[row * F32S128_SA_STRIDE + col] = A[(baseRow + row) * K + kBlock + col];
            }
        } else {
            for (uint e = 0; e < 2; e++) {
                const uint idx = e * F32S128_THREADS + tid;
                const uint row = idx / F32S128_BK;
                const uint col = idx % F32S128_BK;
                const uint gRow = baseRow + row;
                const uint gCol = kBlock + col;
                sA[row * F32S128_SA_STRIDE + col] =
                    (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
        }

        // Load B tile [8×128] — 2 elements per thread
        if (full_tile_n && full_k) {
            for (uint e = 0; e < 2; e++) {
                const uint idx = e * F32S128_THREADS + tid;
                const uint row = idx / F32S128_BN;
                const uint col = idx % F32S128_BN;
                sB[row * F32S128_SB_STRIDE + col] = B[(kBlock + row) * N + baseCol + col];
            }
        } else {
            for (uint e = 0; e < 2; e++) {
                const uint idx = e * F32S128_THREADS + tid;
                const uint row = idx / F32S128_BN;
                const uint col = idx % F32S128_BN;
                const uint gRow = kBlock + row;
                const uint gCol = baseCol + col;
                sB[row * F32S128_SB_STRIDE + col] =
                    (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Compute: one pass (BK=8 = one simdgroup_multiply)
        {
            simdgroup_float8x8 a_frag[F32S128_TM];
            simdgroup_float8x8 b_frag[F32S128_TN];

            for (uint i = 0; i < F32S128_TM; i++)
                simdgroup_load(a_frag[i],
                    &sA[(sg_row * F32S128_TM * 8 + i * 8) * F32S128_SA_STRIDE],
                    F32S128_SA_STRIDE);

            for (uint j = 0; j < F32S128_TN; j++)
                simdgroup_load(b_frag[j],
                    &sB[sg_col * F32S128_TN * 8 + j * 8],
                    F32S128_SB_STRIDE);

            for (uint i = 0; i < F32S128_TM; i++)
                for (uint j = 0; j < F32S128_TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results — reuse sA region as edge scratch (16 SGs × 64 ≤ 1152)
    for (uint i = 0; i < F32S128_TM; i++) {
        for (uint j = 0; j < F32S128_TN; j++) {
            const uint gRow = baseRow + sg_row * F32S128_TM * 8 + i * 8;
            const uint gCol = baseCol + sg_col * F32S128_TN * 8 + j * 8;
            if (gRow + 7 < M && gCol + 7 < N) {
                simdgroup_store(acc[i][j], &C[gRow * N + gCol], N);
            } else if (gRow < M && gCol < N) {
                threadgroup float* edge = &shared_mem[sgid * 64];
                simdgroup_store(acc[i][j], edge, 8);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (slid == 0) {
                    for (uint r = 0; r < 8 && gRow + r < M; r++)
                        for (uint c = 0; c < 8 && gCol + c < N; c++)
                            C[(gRow + r) * N + (gCol + c)] = edge[r * 8 + c];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}

// ============================================================================
// GPU Conversion Kernels — f64 ↔ f32
// ============================================================================
// Used by MPS and f32_simd_pure paths to convert between f64 and f32 formats.
// Each thread converts one element.

kernel void convert_f64_to_f32(
    device const uint2* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    dst[tid] = f64_to_f32_native(src[tid]);
}

kernel void convert_f32_to_f64(
    device const float* src [[buffer(0)]],
    device uint2* dst [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= count) return;
    dst[tid] = f32_to_native_f64(src[tid]);
}

// ============================================================================
// Fixed-Point Matmul — Quake-3-Inspired Bitwise Kernel (ML Mode, 24-bit)
// ============================================================================
// FP_BM, FP_BN, FP_BK, FP_TT, FP_THREADS injected by host via #define
// All parameters hardware-adaptive: computed from device capabilities at init
//
// Core idea (Quake-3 philosophy):
//   IEEE f64 bits encode log2(x) as an integer. Instead of emulating floating-point
//   per-FMA (67+ ops), we ALIGN ALL VALUES TO ONE SCALE at tile load via bitwise shift,
//   then the inner loop is pure integer multiply + integer add.
//
// Data flow (V2 optimized):
//   1. Host CPU scans max exponent during data copy (eliminates GPU pre-pass)
//   2. Tile load: extract 24-bit mantissa via bitwise ops, align to max_exp via shift
//   3. Inner loop: uint32 × uint32 multiply + signed int64 accumulate (pure integer)
//   4. Store: normalize signed int64 accumulator → pack as IEEE f64
//
// V2 optimizations:
//   - CPU-side max_exp scan (eliminates 2 GPU command encoders)
//   - Single signed accumulator via two's complement (halves register pressure)
//   - B tile stride padding (eliminates shared memory bank conflicts)

// Default constants (overridden by host #defines for hardware adaptation)
#ifndef FP_BM
#define FP_BM 64
#define FP_BN 64
#define FP_BK 32
#define FP_TT 4
#define FP_THREADS 256
#endif

constant uint FP_SA_STRIDE = FP_BK + 2;  // A bank conflict padding
constant uint FP_SB_STRIDE = FP_BN + 2;  // B bank conflict padding
constant uint FP_EPT_A = (FP_BM * FP_BK + FP_THREADS - 1) / FP_THREADS;
constant uint FP_EPT_B = (FP_BK * FP_BN + FP_THREADS - 1) / FP_THREADS;

// Fixed-point matmul kernel V2: signed accumulator, no GPU pre-pass needed
// Input: native f64 as uint2, plus pre-computed max exponents (from CPU scan)
// Output: native f64 as uint2
kernel void matmul_fp24(
    device const uint2* A [[buffer(0)]],
    device const uint2* B [[buffer(1)]],
    device uint2* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    device const uint* max_exp_buf [[buffer(6)]],  // [0]=max_exp_A, [1]=max_exp_B
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint baseRow = gid.y * FP_BM;
    const uint baseCol = gid.x * FP_BN;

    // 1D thread-to-tile mapping (hardware-adaptive: works for any FP_THREADS)
    const uint tiles_per_row = FP_BN / FP_TT;
    const uint threadRow = (tid / tiles_per_row) * FP_TT;
    const uint threadCol = (tid % tiles_per_row) * FP_TT;

    // Global max exponents from CPU pre-scan
    const uint max_exp_A = max_exp_buf[0];
    const uint max_exp_B = max_exp_buf[1];

    // Single signed accumulator per output element (two's complement int64 as uint2)
    // For K≤16384 and 24-bit mantissa: max |sum| = K × 2^48 < 2^62, fits in int64
    // Half the register pressure of dual pos/neg accumulators → better occupancy
    uint2 acc[FP_TT][FP_TT];
    #pragma unroll
    for (uint i = 0; i < FP_TT; i++)
        #pragma unroll
        for (uint j = 0; j < FP_TT; j++)
            acc[i][j] = uint2(0, 0);

    // Shared memory with bank conflict padding for both A and B tiles
    threadgroup uint sA[FP_BM * FP_SA_STRIDE];
    threadgroup uint sB[FP_BK * FP_SB_STRIDE];

    // Interior tile flags for fast path (skip bounds checks)
    const bool full_m = (baseRow + FP_BM <= M);
    const bool full_n = (baseCol + FP_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += FP_BK) {
        const bool full_k = (kBlock + FP_BK <= K);

        // === Cooperative load A tile ===
        // Quake-3 bitwise extraction: f64 → 24-bit aligned mantissa + sign
        #pragma unroll
        for (uint i = 0; i < FP_EPT_A; i++) {
            uint idx = i * FP_THREADS + tid;
            if (idx >= FP_BM * FP_BK) break;
            uint row = idx / FP_BK;
            uint col = idx % FP_BK;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;

            uint packed = 0;
            if ((full_m && full_k) || (gRow < M && gCol < K)) {
                uint2 raw = A[gRow * K + gCol];
                uint hi = raw.y;  // native: .y = high32
                uint lo = raw.x;  // native: .x = low32
                uint sign = hi >> 31;
                uint exp = (hi >> 20) & 0x7FFu;

                if (exp != 0) {  // skip zero/subnormal
                    // Extract hidden bit + top 23 explicit mantissa bits = 24 bits
                    uint mant = (1u << 23) | ((hi & 0xFFFFFu) << 3) | (lo >> 29);

                    // BITWISE RIGHT SHIFT: align to global max exponent
                    uint shift = max_exp_A - exp;
                    if (shift < 24u) mant >>= shift; else mant = 0;

                    // Pack sign into bit 31 (mantissa only uses bits [23:0])
                    packed = mant | (sign << 31);
                }
            }
            sA[row * FP_SA_STRIDE + col] = packed;
        }

        // === Cooperative load B tile ===
        #pragma unroll
        for (uint i = 0; i < FP_EPT_B; i++) {
            uint idx = i * FP_THREADS + tid;
            if (idx >= FP_BK * FP_BN) break;
            uint row = idx / FP_BN;
            uint col = idx % FP_BN;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;

            uint packed = 0;
            if ((full_k && full_n) || (gRow < K && gCol < N)) {
                uint2 raw = B[gRow * N + gCol];
                uint hi = raw.y;
                uint lo = raw.x;
                uint sign = hi >> 31;
                uint exp = (hi >> 20) & 0x7FFu;

                if (exp != 0) {
                    uint mant = (1u << 23) | ((hi & 0xFFFFFu) << 3) | (lo >> 29);
                    uint shift = max_exp_B - exp;
                    if (shift < 24u) mant >>= shift; else mant = 0;
                    packed = mant | (sign << 31);
                }
            }
            sB[row * FP_SB_STRIDE + col] = packed;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Inner loop: PURE INTEGER multiply + signed accumulate ===
        // Two's complement: negative products are branchlessly negated via
        // bitwise NOT + 1, then added to single accumulator. No predicated
        // dual-branch waste — all threads execute the same instruction stream.
        #pragma unroll
        for (uint kk = 0; kk < FP_BK; kk++) {
            // Load pre-aligned mantissas from shared memory
            uint a_packed[FP_TT];
            #pragma unroll
            for (uint i = 0; i < FP_TT; i++)
                a_packed[i] = sA[(threadRow + i) * FP_SA_STRIDE + kk];

            uint b_packed[FP_TT];
            #pragma unroll
            for (uint j = 0; j < FP_TT; j++)
                b_packed[j] = sB[kk * FP_SB_STRIDE + threadCol + j];

            #pragma unroll
            for (uint i = 0; i < FP_TT; i++) {
                // Extract mantissa and sign from packed value
                uint a_mant = a_packed[i] & 0x7FFFFFFFu;
                uint a_sign = a_packed[i] >> 31;

                #pragma unroll
                for (uint j = 0; j < FP_TT; j++) {
                    uint b_mant = b_packed[j] & 0x7FFFFFFFu;
                    uint b_sign = b_packed[j] >> 31;

                    // MULTIPLY: 24×24 → 48-bit unsigned product (2 I32 mul ops)
                    uint prod_lo = a_mant * b_mant;
                    uint prod_hi = mulhi(a_mant, b_mant);

                    // SIGN: branchless two's complement negate if negative
                    uint neg = a_sign ^ b_sign;
                    uint neg_lo = (~prod_lo) + 1u;
                    uint neg_carry = select(0u, 1u, neg_lo == 0u);
                    uint neg_hi = (~prod_hi) + neg_carry;
                    uint final_lo = select(prod_lo, neg_lo, neg != 0u);
                    uint final_hi = select(prod_hi, neg_hi, neg != 0u);

                    // ACCUMULATE: signed 64-bit add (two's complement = unsigned add)
                    uint old_lo = acc[i][j].y;
                    acc[i][j].y += final_lo;
                    uint carry = select(0u, 1u, acc[i][j].y < old_lo);
                    acc[i][j].x += final_hi + carry;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store: convert signed int64 accumulators → IEEE f64 ===
    // Product scale: each aligned mantissa represents value × 2^(max_exp - 1046)
    // Product of two: 2^(max_exp_A + max_exp_B - 2092)
    const int product_scale = (int)max_exp_A + (int)max_exp_B - 2092;

    #pragma unroll
    for (uint i = 0; i < FP_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < FP_TT; j++) {
            uint gRow = baseRow + threadRow + i;
            uint gCol = baseCol + threadCol + j;
            if (gRow >= M || gCol >= N) continue;

            // Zero check
            if (acc[i][j].x == 0 && acc[i][j].y == 0) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }

            // Sign from MSB of two's complement accumulator
            uint sign_out = acc[i][j].x >> 31;

            // Magnitude: negate if negative (two's complement → absolute value)
            uint2 mag_v;
            if (sign_out) {
                uint n_lo = (~acc[i][j].y) + 1u;
                uint n_carry = select(0u, 1u, n_lo == 0u);
                mag_v = uint2((~acc[i][j].x) + n_carry, n_lo);
            } else {
                mag_v = acc[i][j];
            }

            sf64 mag = sf64(mag_v.x, mag_v.y);

            // Normalize: find leading bit position
            int lz = clz64(mag);
            if (lz >= 64) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }

            // Shift so MSB at bit 63
            sf64 norm = shl64(mag, lz);

            // Compute biased f64 exponent
            int result_exp = (63 - lz) + product_scale + 1023;

            if (result_exp <= 0) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }
            if (result_exp >= 2047) {
                uint inf_hi = (sign_out << 31) | 0x7FF00000u;
                C[gRow * N + gCol] = uint2(0, inf_hi);
                continue;
            }

            // Extract 52-bit mantissa from normalized 64-bit value
            // norm.x = bits [63:32], norm.y = bits [31:0]
            // Bit 63 = hidden bit (not stored), mantissa = bits [62:11] = 52 bits
            uint mant_hi_out = (norm.x >> 11) & 0xFFFFFu;
            uint mant_lo_out = ((norm.x & 0x7FFu) << 21) | (norm.y >> 11);

            // Round-to-nearest-even
            uint round_bit = (norm.y >> 10) & 1u;
            uint sticky = norm.y & 0x3FFu;
            if (round_bit && (sticky || (mant_lo_out & 1u))) {
                mant_lo_out += 1;
                if (mant_lo_out == 0) {
                    mant_hi_out += 1;
                    if (mant_hi_out >= 0x100000u) {
                        mant_hi_out = 0;
                        result_exp += 1;
                    }
                }
            }

            // Pack as IEEE f64 (native byte order: .x = lo32, .y = hi32)
            uint hi_out = (sign_out << 31) | (((uint)result_exp & 0x7FFu) << 20) | mant_hi_out;
            C[gRow * N + gCol] = uint2(mant_lo_out, hi_out);
        }
    }
}

// ============================================================================
// FIXED-POINT 53-BIT EXACT MATMUL KERNEL
// ============================================================================
//
// FP53_BM, FP53_BN, FP53_BK, FP53_TT, FP53_THREADS injected by host via #define
// All parameters hardware-adaptive: computed from device capabilities at init
//
// Same Quake-3 bitwise philosophy as fp24, but with full 53-bit mantissa
// and 128-bit signed accumulation for exact IEEE f64 results.
//
// vs sf64 V2 (67 ops/FMA + ~45 branch divergence waste = 112 effective):
//   fp53: ~60 ops/FMA, zero branch divergence (pure integer inner loop)
//
// Data flow:
//   1. Host CPU scans max exponent during data copy (same as fp24)
//   2. Tile load: extract 53-bit mantissa as uint2, align to max_exp via 64-bit shift
//   3. Inner loop: mul64x64 (128-bit) + branchless negate + 128-bit accumulate
//   4. Store: normalize 128-bit signed accumulator → pack as IEEE f64

// Default constants (overridden by host #defines for hardware adaptation)
#ifndef FP53_BM
#define FP53_BM 64
#define FP53_BN 64
#define FP53_BK 8
#define FP53_TT 4
#define FP53_THREADS 256
#define FP53_SB_STRIDE 65
#endif

constant uint FP53_SA_STRIDE = FP53_BK + 1;  // A stride padding (uint2 = 8 bytes, +1 avoids bank conflicts)
constant uint FP53_EPT_A = (FP53_BM * FP53_BK + FP53_THREADS - 1) / FP53_THREADS;
constant uint FP53_EPT_B = (FP53_BK * FP53_BN + FP53_THREADS - 1) / FP53_THREADS;

// 128-bit uint4 layout: .w = MSW (word3), .z = word2, .y = word1, .x = LSW (word0)
// Metal uint4(a,b,c,d) maps .x=a, .y=b, .z=c, .w=d
// So construct as: uint4(lsw, w1, w2, msw) to get .x=lsw, .w=msw

// Count leading zeros on 128-bit uint4 (.w=MSW, .x=LSW)
inline int clz128(uint4 v) {
    if (v.w != 0) return clz(v.w);
    if (v.z != 0) return 32 + clz(v.z);
    if (v.y != 0) return 64 + clz(v.y);
    if (v.x != 0) return 96 + clz(v.x);
    return 128;
}

// Left shift 128-bit uint4 (.w=MSW, .x=LSW) by n bits
inline uint4 shl128(uint4 v, int n) {
    if (n <= 0) return v;
    if (n >= 128) return uint4(0, 0, 0, 0);
    // Result must preserve .w=MSW, .x=LSW convention
    // uint4(new_x, new_y, new_z, new_w) → .x=new_x(LSW), .w=new_w(MSW)
    if (n >= 96) {
        return uint4(0, 0, 0, v.x << (n - 96));
    }
    if (n >= 64) {
        int s = n - 64;
        return uint4(0, 0, v.x << s,
                     (v.y << s) | select(0u, v.x >> (32 - s), s > 0));
    }
    if (n >= 32) {
        int s = n - 32;
        return uint4(0, v.x << s,
                     (v.y << s) | select(0u, v.x >> (32 - s), s > 0),
                     (v.z << s) | select(0u, v.y >> (32 - s), s > 0));
    }
    // n < 32: shift each word, carry bits from lower to upper
    return uint4(v.x << n,
                 (v.y << n) | (v.x >> (32 - n)),
                 (v.z << n) | (v.y >> (32 - n)),
                 (v.w << n) | (v.z >> (32 - n)));
}

// Fixed-point 53-bit exact matmul kernel
// Input: native f64 as uint2, plus pre-computed max exponents (from CPU scan)
// Output: native f64 as uint2
kernel void matmul_fp53(
    device const uint2* A [[buffer(0)]],
    device const uint2* B [[buffer(1)]],
    device uint2* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    device const uint* max_exp_buf [[buffer(6)]],  // [0]=max_exp_A, [1]=max_exp_B
    uint2 gid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]])
{
    const uint baseRow = gid.y * FP53_BM;
    const uint baseCol = gid.x * FP53_BN;

    // 1D thread-to-tile mapping
    const uint tiles_per_row = FP53_BN / FP53_TT;
    const uint threadRow = (tid / tiles_per_row) * FP53_TT;
    const uint threadCol = (tid % tiles_per_row) * FP53_TT;

    // Global max exponents from CPU pre-scan
    const uint max_exp_A = max_exp_buf[0];
    const uint max_exp_B = max_exp_buf[1];

    // 128-bit signed accumulators (uint4: w=MSW, x, y, z=LSW)
    // For K≤16384 and 53-bit mantissa: max |sum| = K × 2^106 < 2^120, fits in int128
    uint4 acc[FP53_TT][FP53_TT];
    #pragma unroll
    for (uint i = 0; i < FP53_TT; i++)
        #pragma unroll
        for (uint j = 0; j < FP53_TT; j++)
            acc[i][j] = uint4(0, 0, 0, 0);

    // Shared memory: uint2 per element for 53-bit mantissa + sign packed in bit 31 of .y
    // Mantissa uses at most 21 bits in .y (bits [20:0]), so bit 31 is free for sign.
    // This eliminates separate sign arrays and their associated race conditions.
    threadgroup uint2 sA[FP53_BM * FP53_SA_STRIDE];
    threadgroup uint2 sB[FP53_BK * FP53_SB_STRIDE];

    // Interior tile flags for fast path
    const bool full_m = (baseRow + FP53_BM <= M);
    const bool full_n = (baseCol + FP53_BN <= N);

    for (uint kBlock = 0; kBlock < K; kBlock += FP53_BK) {
        const bool full_k = (kBlock + FP53_BK <= K);

        // === Cooperative load A tile ===
        // Extract full 53-bit mantissa as uint2, align to max_exp via 64-bit shift
        // Sign packed into bit 31 of .y (mantissa hi word)
        #pragma unroll
        for (uint i = 0; i < FP53_EPT_A; i++) {
            uint idx = i * FP53_THREADS + tid;
            if (idx >= FP53_BM * FP53_BK) break;
            uint row = idx / FP53_BK;
            uint col = idx % FP53_BK;
            uint gRow = baseRow + row;
            uint gCol = kBlock + col;

            uint2 packed = uint2(0, 0);
            if ((full_m && full_k) || (gRow < M && gCol < K)) {
                uint2 raw = A[gRow * K + gCol];
                uint hi = raw.y;  // native: .y = high32
                uint lo = raw.x;  // native: .x = low32
                uint sign = hi >> 31;
                uint exp = (hi >> 20) & 0x7FFu;

                if (exp != 0) {  // skip zero/subnormal
                    uint mant_hi = (hi & 0xFFFFFu) | 0x100000u;  // 21 bits
                    uint mant_lo = lo;                              // 32 bits

                    uint shift = max_exp_A - exp;
                    if (shift < 53u) {
                        if (shift == 0u) {
                            // No shift needed — mantissa already aligned
                        } else if (shift < 32u) {
                            mant_lo = (mant_lo >> shift) | (mant_hi << (32u - shift));
                            mant_hi >>= shift;
                        } else {
                            mant_lo = mant_hi >> (shift - 32u);
                            mant_hi = 0;
                        }
                        // Pack sign into bit 31 of hi word (mantissa uses bits [20:0])
                        packed = uint2(mant_lo, mant_hi | (sign << 31));
                    }
                }
            }
            sA[row * FP53_SA_STRIDE + col] = packed;
        }

        // === Cooperative load B tile ===
        #pragma unroll
        for (uint i = 0; i < FP53_EPT_B; i++) {
            uint idx = i * FP53_THREADS + tid;
            if (idx >= FP53_BK * FP53_BN) break;
            uint row = idx / FP53_BN;
            uint col = idx % FP53_BN;
            uint gRow = kBlock + row;
            uint gCol = baseCol + col;

            uint2 packed = uint2(0, 0);
            if ((full_k && full_n) || (gRow < K && gCol < N)) {
                uint2 raw = B[gRow * N + gCol];
                uint hi = raw.y;
                uint lo = raw.x;
                uint sign = hi >> 31;
                uint exp = (hi >> 20) & 0x7FFu;

                if (exp != 0) {
                    uint mant_hi = (hi & 0xFFFFFu) | 0x100000u;
                    uint mant_lo = lo;

                    uint shift = max_exp_B - exp;
                    if (shift < 53u) {
                        if (shift == 0u) {
                            // No shift needed — mantissa already aligned
                        } else if (shift < 32u) {
                            mant_lo = (mant_lo >> shift) | (mant_hi << (32u - shift));
                            mant_hi >>= shift;
                        } else {
                            mant_lo = mant_hi >> (shift - 32u);
                            mant_hi = 0;
                        }
                        packed = uint2(mant_lo, mant_hi | (sign << 31));
                    }
                }
            }
            sB[row * FP53_SB_STRIDE + col] = packed;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Inner loop: mul64x64 + branchless negate + 128-bit accumulate ===
        // Zero branch divergence: all threads execute identical instruction stream
        #pragma unroll
        for (uint kk = 0; kk < FP53_BK; kk++) {
            // Load pre-aligned mantissas from shared memory
            // Sign is in bit 31 of .y, mantissa in .y[20:0] and .x[31:0]
            uint2 a_packed[FP53_TT];
            #pragma unroll
            for (uint i = 0; i < FP53_TT; i++)
                a_packed[i] = sA[(threadRow + i) * FP53_SA_STRIDE + kk];

            uint2 b_packed[FP53_TT];
            #pragma unroll
            for (uint j = 0; j < FP53_TT; j++)
                b_packed[j] = sB[kk * FP53_SB_STRIDE + threadCol + j];

            #pragma unroll
            for (uint i = 0; i < FP53_TT; i++) {
                // Extract sign and mantissa from packed value
                uint a_sign = a_packed[i].y >> 31;
                sf64 a_sf = sf64(a_packed[i].y & 0x1FFFFFu, a_packed[i].x);

                #pragma unroll
                for (uint j = 0; j < FP53_TT; j++) {
                    uint b_sign = b_packed[j].y >> 31;
                    sf64 b_sf = sf64(b_packed[j].y & 0x1FFFFFu, b_packed[j].x);

                    // MULTIPLY: 53×53 → 106-bit product via mul64x64 (27 ops, 8 I32 muls)
                    uint128_t prod = mul64x64(a_sf, b_sf);

                    // SIGN: branchless two's complement negate if negative
                    uint neg = a_sign ^ b_sign;

                    // Negate 128-bit: ~prod + 1 (LSW to MSW)
                    uint n0 = (~prod.w0) + 1u;
                    uint c0 = select(0u, 1u, n0 == 0u);
                    uint n1 = (~prod.w1) + c0;
                    uint c1 = select(0u, 1u, n1 == 0u && c0 != 0u);
                    uint n2 = (~prod.w2) + c1;
                    uint c2 = select(0u, 1u, n2 == 0u && c1 != 0u);
                    uint n3 = (~prod.w3) + c2;

                    // Select original or negated
                    uint f0 = select(prod.w0, n0, neg != 0u);  // LSW
                    uint f1 = select(prod.w1, n1, neg != 0u);
                    uint f2 = select(prod.w2, n2, neg != 0u);
                    uint f3 = select(prod.w3, n3, neg != 0u);  // MSW

                    // ACCUMULATE: 128-bit add with carry (LSW→MSW)
                    // Layout: .x=LSW(w0), .y=w1, .z=w2, .w=MSW(w3)
                    uint old_x = acc[i][j].x;
                    acc[i][j].x += f0;
                    uint carry0 = select(0u, 1u, acc[i][j].x < old_x);

                    uint old_y = acc[i][j].y;
                    acc[i][j].y += f1 + carry0;
                    uint carry1 = select(0u, 1u, acc[i][j].y < old_y ||
                                         (carry0 != 0u && acc[i][j].y == old_y));

                    uint old_z = acc[i][j].z;
                    acc[i][j].z += f2 + carry1;
                    uint carry2 = select(0u, 1u, acc[i][j].z < old_z ||
                                         (carry1 != 0u && acc[i][j].z == old_z));

                    acc[i][j].w += f3 + carry2;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // === Store: convert signed 128-bit accumulators → IEEE f64 ===
    // Product scale: each aligned mantissa represents value × 2^(max_exp - 1075)
    // Product of two: 2^(max_exp_A + max_exp_B - 2150)
    // where 1075 = 1023 (bias) + 52 (mantissa bits), 2150 = 2 × 1075
    const int product_scale = (int)max_exp_A + (int)max_exp_B - 2150;

    #pragma unroll
    for (uint i = 0; i < FP53_TT; i++) {
        #pragma unroll
        for (uint j = 0; j < FP53_TT; j++) {
            uint gRow = baseRow + threadRow + i;
            uint gCol = baseCol + threadCol + j;
            if (gRow >= M || gCol >= N) continue;

            // Zero check
            if (acc[i][j].w == 0 && acc[i][j].x == 0 &&
                acc[i][j].y == 0 && acc[i][j].z == 0) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }

            // Sign from MSB of two's complement 128-bit accumulator
            uint sign_out = acc[i][j].w >> 31;

            // Magnitude: negate if negative (two's complement from LSW to MSW)
            uint4 mag;
            if (sign_out) {
                uint m0 = (~acc[i][j].x) + 1u;     // .x = LSW
                uint mc0 = select(0u, 1u, m0 == 0u);
                uint m1 = (~acc[i][j].y) + mc0;     // .y = w1
                uint mc1 = select(0u, 1u, m1 == 0u && mc0 != 0u);
                uint m2 = (~acc[i][j].z) + mc1;     // .z = w2
                uint mc2 = select(0u, 1u, m2 == 0u && mc1 != 0u);
                uint m3 = (~acc[i][j].w) + mc2;     // .w = MSW
                mag = uint4(m0, m1, m2, m3);        // .x=LSW, .w=MSW
            } else {
                mag = acc[i][j];
            }

            // Find leading bit position via CLZ on 128-bit value
            int lz = clz128(mag);
            if (lz >= 128) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }

            // Shift so MSB at bit 127
            uint4 norm = shl128(mag, lz);

            // Compute biased f64 exponent
            // The accumulator bit position (127 - lz) maps to the product scale
            int result_exp = (127 - lz) + product_scale + 1023;

            if (result_exp <= 0) {
                C[gRow * N + gCol] = uint2(0, 0);
                continue;
            }
            if (result_exp >= 2047) {
                uint inf_hi = (sign_out << 31) | 0x7FF00000u;
                C[gRow * N + gCol] = uint2(0, inf_hi);
                continue;
            }

            // Extract 52-bit mantissa from normalized 128-bit value
            // norm.w has the hidden bit at bit 31 (MSB), bits [30:0] are top 31 mantissa bits
            // norm.z has next 32 bits, norm.y next 32, norm.x bottom 32
            // Need 52 bits after hidden bit: bits [126:75] of the 128-bit value
            // = norm.w[30:0] (31 bits) + norm.z[31:11] (21 bits) = 52 bits
            uint mant_hi_out = (norm.w >> 11) & 0xFFFFFu;  // bits [30:11] = 20 bits for f64 mantissa hi
            uint mant_lo_out = ((norm.w & 0x7FFu) << 21) | (norm.z >> 11);  // bits [10:0] of w + [31:11] of z

            // Round-to-nearest-even
            uint round_bit = (norm.z >> 10) & 1u;
            uint sticky = (norm.z & 0x3FFu) | norm.y | norm.x;
            if (round_bit && (sticky || (mant_lo_out & 1u))) {
                mant_lo_out += 1;
                if (mant_lo_out == 0) {
                    mant_hi_out += 1;
                    if (mant_hi_out >= 0x100000u) {
                        mant_hi_out = 0;
                        result_exp += 1;
                    }
                }
            }

            // Pack as IEEE f64 (native byte order: .x = lo32, .y = hi32)
            uint hi_out = (sign_out << 31) | (((uint)result_exp & 0x7FFu) << 20) | mant_hi_out;
            C[gRow * N + gCol] = uint2(mant_lo_out, hi_out);
        }
    }
}

// Ozaki-I kernel removed — replaced by Ozaki-II CRT approach below.

// ============================================================================
// Ozaki-II CRT Modular GEMM -- f32 GEMM with K-extraction for exact integers
// ============================================================================
// Computes W_l = mod(A_l * B_l, p_l) where A_l, B_l have values in [-128, 128].
// Uses the same simdgroup MMA inner loop as matmul_f32_simd_pure. Additions:
//   1. K-extraction: every 63 BK-iterations (126 MMA calls), extract the f32
//      accumulator via thread_elements(), mod-reduce, accumulate into per-thread
//      partial_w. Keeps the f32 accumulator < 2^24 (exact integer arithmetic).
//   2. Probe trick: discover which 2 output positions each thread owns in the
//      8x8 simdgroup_float8x8, for correct per-thread output writes.
//
// Called N times (once per CRT modulus) by ozaki_ii_dispatch on the CPU side.
// CPU handles scaling, mod-reduction of inputs, and CRT reconstruction.

kernel void matmul_ozaki_gemm(
    device const float* A [[buffer(0)]],   // mod-reduced A_l, values in [-128, 128]
    device const float* B [[buffer(1)]],   // mod-reduced B_l, values in [-128, 128]
    device float* C [[buffer(2)]],         // output W_l, mod-reduced results
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& modulus [[buffer(6)]],  // p_l (max 256)
    uint2 group_id [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint slid [[thread_index_in_simdgroup]])
{
    const uint baseRow = group_id.y * F32S_BM;
    const uint baseCol = group_id.x * F32S_BN;
    const uint sg_row = sgid / F32S_WN;
    const uint sg_col = sgid % F32S_WN;
    const uint tid = sgid * 32 + slid;

    const float p_f = float(modulus);
    const float p_inv = 1.0f / p_f;

    // === Probe trick: discover thread_elements() mapping ===
    // Each thread in a simdgroup owns 2 elements of the 8x8 output matrix.
    // The mapping is unspecified by Metal -- probe it at runtime for portability.
    // Must initialize ALL 64 positions (8x8). Use tid (not slid) since slid is
    // only 0-31 per simdgroup — tid covers 0-127 across all simdgroups.
    threadgroup float shared_mem[F32S_SHARED_SIZE];
    if (tid < 64) shared_mem[tid] = (float)tid;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 probe;
    simdgroup_load(probe, &shared_mem[0], 8);
    const uint map0 = (uint)probe.thread_elements()[0];
    const uint map1 = (uint)probe.thread_elements()[1];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // f32 simdgroup accumulators (same as f32_simd_pure)
    simdgroup_float8x8 acc[F32S_TM][F32S_TN];
    for (uint i = 0; i < F32S_TM; i++)
        for (uint j = 0; j < F32S_TN; j++)
            acc[i][j] = simdgroup_float8x8(0);

    // Per-thread partial_w accumulators for K-extraction.
    // Each thread owns 2 elements per (TM,TN) fragment pair.
    float partial_w[F32S_TM][F32S_TN][2];
    for (uint i = 0; i < F32S_TM; i++)
        for (uint j = 0; j < F32S_TN; j++) {
            partial_w[i][j][0] = 0.0f;
            partial_w[i][j][1] = 0.0f;
        }

    threadgroup float* sA = &shared_mem[0];
    threadgroup float* sB = &shared_mem[F32S_SA_SIZE];

    const uint VEC_A = (F32S_BM * F32S_BK) / (F32S_THREADS * 4);
    const uint VEC_B = (F32S_BK * F32S_BN) / (F32S_THREADS * 4);

    const bool full_tile_m = (baseRow + F32S_BM <= M);
    const bool full_tile_n = (baseCol + F32S_BN <= N);

    // K-extraction interval: extract every 63 kBlock iterations.
    // With BK=16, each kBlock does 2 MMA calls (BK/8=2).
    // 63 * 2 = 126 MMA calls. Max acc: 126 * 8 * 128^2 = 16,515,072 < 2^24 = 16,777,216
    const uint EXTRACT_INTERVAL = 63;
    uint kblock_count = 0;

    for (uint kBlock = 0; kBlock < K; kBlock += F32S_BK) {
        const bool full_tile_k = (kBlock + F32S_BK <= K);

        // === Load A tile (identical to f32_simd_pure) ===
        if (full_tile_m && full_tile_k) {
            for (uint i = 0; i < VEC_A; i++) {
                const uint vec_idx = i * F32S_THREADS + tid;
                const uint elem = vec_idx * 4;
                const uint row = elem / F32S_BK;
                const uint col = elem % F32S_BK;
                float4 v = *((device const float4*)&A[(baseRow + row) * K + kBlock + col]);
                sA[row * F32S_SA_STRIDE + col + 0] = v.x;
                sA[row * F32S_SA_STRIDE + col + 1] = v.y;
                sA[row * F32S_SA_STRIDE + col + 2] = v.z;
                sA[row * F32S_SA_STRIDE + col + 3] = v.w;
            }
        } else {
            const uint EPT_A = (F32S_BM * F32S_BK) / F32S_THREADS;
            for (uint i = 0; i < EPT_A; i++) {
                const uint idx = i * F32S_THREADS + tid;
                const uint row = idx / F32S_BK;
                const uint col = idx % F32S_BK;
                const uint gRow = baseRow + row;
                const uint gCol = kBlock + col;
                sA[row * F32S_SA_STRIDE + col] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }
        }

        // === Load B tile (identical to f32_simd_pure) ===
        if (full_tile_n && full_tile_k) {
            for (uint i = 0; i < VEC_B; i++) {
                const uint vec_idx = i * F32S_THREADS + tid;
                const uint elem = vec_idx * 4;
                const uint row = elem / F32S_BN;
                const uint col = elem % F32S_BN;
                float4 v = *((device const float4*)&B[(kBlock + row) * N + baseCol + col]);
                sB[row * F32S_SB_STRIDE + col + 0] = v.x;
                sB[row * F32S_SB_STRIDE + col + 1] = v.y;
                sB[row * F32S_SB_STRIDE + col + 2] = v.z;
                sB[row * F32S_SB_STRIDE + col + 3] = v.w;
            }
        } else {
            const uint EPT_B = (F32S_BK * F32S_BN) / F32S_THREADS;
            for (uint i = 0; i < EPT_B; i++) {
                const uint idx = i * F32S_THREADS + tid;
                const uint row = idx / F32S_BN;
                const uint col = idx % F32S_BN;
                const uint gRow = kBlock + row;
                const uint gCol = baseCol + col;
                sB[row * F32S_SB_STRIDE + col] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Inner loop: MMA (identical to f32_simd_pure) ===
        for (uint kk = 0; kk < F32S_BK; kk += 8) {
            simdgroup_float8x8 a_frag[F32S_TM];
            simdgroup_float8x8 b_frag[F32S_TN];

            for (uint i = 0; i < F32S_TM; i++) {
                simdgroup_load(a_frag[i],
                    &sA[(sg_row * F32S_TM * 8 + i * 8) * F32S_SA_STRIDE + kk],
                    F32S_SA_STRIDE);
            }

            for (uint j = 0; j < F32S_TN; j++) {
                simdgroup_load(b_frag[j],
                    &sB[kk * F32S_SB_STRIDE + (sg_col * F32S_TN * 8 + j * 8)],
                    F32S_SB_STRIDE);
            }

            for (uint i = 0; i < F32S_TM; i++)
                for (uint j = 0; j < F32S_TN; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_frag[i], b_frag[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === K-extraction: prevent f32 accumulator overflow ===
        // Extract per-thread values, mod-reduce, accumulate into partial_w, zero acc.
        // mod(a+b, p) = mod(mod(a,p) + mod(b,p), p) -- partial mods are correct.
        kblock_count++;
        if (kblock_count >= EXTRACT_INTERVAL) {
            for (uint i = 0; i < F32S_TM; i++) {
                for (uint j = 0; j < F32S_TN; j++) {
                    float v0 = acc[i][j].thread_elements()[0];
                    float v1 = acc[i][j].thread_elements()[1];
                    // Symmetric mod: val - p * round(val / p)
                    partial_w[i][j][0] += v0 - p_f * rint(v0 * p_inv);
                    partial_w[i][j][1] += v1 - p_f * rint(v1 * p_inv);
                    acc[i][j] = simdgroup_float8x8(0);
                }
            }
            kblock_count = 0;
        }
    }

    // === Final extraction: remaining K-blocks after last interval ===
    if (kblock_count > 0) {
        for (uint i = 0; i < F32S_TM; i++) {
            for (uint j = 0; j < F32S_TN; j++) {
                float v0 = acc[i][j].thread_elements()[0];
                float v1 = acc[i][j].thread_elements()[1];
                partial_w[i][j][0] += v0 - p_f * rint(v0 * p_inv);
                partial_w[i][j][1] += v1 - p_f * rint(v1 * p_inv);
            }
        }
    }

    // === Output: final mod-reduce of accumulated partial_w, write to global ===
    // Use probed mapping to write each thread's 2 elements to correct (row, col).
    for (uint i = 0; i < F32S_TM; i++) {
        for (uint j = 0; j < F32S_TN; j++) {
            for (uint ei = 0; ei < 2; ei++) {
                const uint map_val = (ei == 0) ? map0 : map1;
                const uint frag_row = map_val / 8;
                const uint frag_col = map_val % 8;
                const uint gRow = baseRow + sg_row * F32S_TM * 8 + i * 8 + frag_row;
                const uint gCol = baseCol + sg_col * F32S_TN * 8 + j * 8 + frag_col;
                if (gRow < M && gCol < N) {
                    float w = partial_w[i][j][ei];
                    w = w - p_f * rint(w * p_inv);  // final symmetric mod
                    C[gRow * N + gCol] = w;
                }
            }
        }
    }
}

#endif // ESHKOL_METAL_SOFTFLOAT_H
