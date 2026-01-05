/*
 * Metal SoftFloat - IEEE 754 Double-Precision Emulation for Metal
 *
 * Implements full f64 arithmetic using uint2 (two 32-bit integers)
 * Based on Berkeley SoftFloat library algorithms
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

inline int sf64_exp(sf64 x) {
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

// 64-bit left shift
inline sf64 shl64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) {
        return sf64(x.y << (n - 32), 0u);
    }
    return sf64((x.x << n) | (x.y >> (32 - n)), x.y << n);
}

// 64-bit right shift (logical)
inline sf64 shr64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) {
        return sf64(0u, x.x >> (n - 32));
    }
    return sf64(x.x >> n, (x.y >> n) | (x.x << (32 - n)));
}

// 64-bit right shift with sticky bit (jam)
// Preserves rounding information by setting LSB if any bits are lost
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

// 64-bit addition
inline sf64 add64(sf64 a, sf64 b) {
    uint lo = a.y + b.y;
    uint carry = (lo < a.y) ? 1u : 0u;
    uint hi = a.x + b.x + carry;
    return sf64(hi, lo);
}

// 64-bit addition with carry out
inline sf64 add64_carry(sf64 a, sf64 b, thread bool& carry_out) {
    uint lo = a.y + b.y;
    uint c1 = (lo < a.y) ? 1u : 0u;
    uint hi = a.x + b.x + c1;
    carry_out = (hi < a.x) || (c1 != 0 && hi == a.x + c1);
    return sf64(hi, lo);
}

// 64-bit subtraction
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
// 128-bit Multiplication Support
// ============================================================================

struct uint128_t {
    uint w3, w2, w1, w0;  // w3 is MSW, w0 is LSW
};

// 64x64 -> 128 bit multiplication
// Uses 16-bit pieces to avoid overflow
inline uint128_t mul64x64(sf64 a, sf64 b) {
    // Split each 64-bit value into four 16-bit pieces
    uint a3 = a.x >> 16;
    uint a2 = a.x & 0xFFFFu;
    uint a1 = a.y >> 16;
    uint a0 = a.y & 0xFFFFu;
    uint b3 = b.x >> 16;
    uint b2 = b.x & 0xFFFFu;
    uint b1 = b.y >> 16;
    uint b0 = b.y & 0xFFFFu;

    // Compute all 16 partial products (each max 32 bits)
    uint p00 = a0 * b0;
    uint p01 = a0 * b1;
    uint p02 = a0 * b2;
    uint p03 = a0 * b3;
    uint p10 = a1 * b0;
    uint p11 = a1 * b1;
    uint p12 = a1 * b2;
    uint p13 = a1 * b3;
    uint p20 = a2 * b0;
    uint p21 = a2 * b1;
    uint p22 = a2 * b2;
    uint p23 = a2 * b3;
    uint p30 = a3 * b0;
    uint p31 = a3 * b1;
    uint p32 = a3 * b2;
    uint p33 = a3 * b3;

    // Accumulate partial products column by column
    // Column 0 (bits 0-15)
    uint c0 = p00 & 0xFFFFu;
    uint carry = p00 >> 16;

    // Column 1 (bits 16-31)
    uint c1 = p01 + p10 + carry;
    carry = c1 >> 16;
    c1 &= 0xFFFFu;

    // Column 2 (bits 32-47)
    uint c2 = p02 + p11 + p20 + carry;
    carry = c2 >> 16;
    c2 &= 0xFFFFu;

    // Column 3 (bits 48-63)
    uint c3 = p03 + p12 + p21 + p30 + carry;
    carry = c3 >> 16;
    c3 &= 0xFFFFu;

    // Column 4 (bits 64-79)
    uint c4 = p13 + p22 + p31 + carry;
    carry = c4 >> 16;
    c4 &= 0xFFFFu;

    // Column 5 (bits 80-95)
    uint c5 = p23 + p32 + carry;
    carry = c5 >> 16;
    c5 &= 0xFFFFu;

    // Column 6 (bits 96-111)
    uint c6 = p33 + carry;
    carry = c6 >> 16;
    c6 &= 0xFFFFu;

    // Column 7 (bits 112-127)
    uint c7 = carry;

    uint128_t result;
    result.w0 = (c1 << 16) | c0;
    result.w1 = (c3 << 16) | c2;
    result.w2 = (c5 << 16) | c4;
    result.w3 = (c7 << 16) | c6;
    return result;
}

// ============================================================================
// Normalization Helpers
// ============================================================================

// Normalize subnormal significand and return shift amount
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

// Round and pack f64 result
// sig is assumed to be shifted left with leading 1 at appropriate position
// round_bits contains the bits to be rounded off
inline sf64 sf64_round_pack(bool sign, int exp_raw, sf64 sig, uint round_bits) {
    // Round to nearest even
    // round_bits: bit 10 is the rounding bit, bits 0-9 are sticky
    bool round_up = false;
    if (round_bits > 0x400u) {
        round_up = true;
    } else if (round_bits == 0x400u) {
        // Tie: round to even (round up if LSB of result is 1)
        round_up = (sig.y & 1u) != 0;
    }

    if (round_up) {
        sig = add64(sig, sf64(0u, 1u));
        // Check for overflow from rounding
        if ((sig.x & 0x00200000u) != 0) {
            sig = shr64(sig, 1);
            exp_raw++;
        }
    }

    // Check for overflow to infinity
    if (exp_raw >= SF64_EXP_MAX) {
        return sign ? SF64_NEG_INF : SF64_INF;
    }

    // Check for underflow to zero/subnormal
    if (exp_raw <= 0) {
        // Subnormal result
        int shift = 1 - exp_raw;
        if (shift >= 64) {
            // Complete underflow to zero
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

sf64 sf64_add(sf64 a, sf64 b) {
    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);
    int expA = sf64_exp_raw(a);
    int expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a);
    sf64 sigB = sf64_sig(b);

    // Handle NaN
    if (sf64_is_nan(a)) return SF64_QNAN;
    if (sf64_is_nan(b)) return SF64_QNAN;

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
            // -0 + -0 = -0, otherwise +0
            return (signA && signB) ? SF64_NEG_ZERO : SF64_ZERO;
        }
        return b;
    }
    if (sf64_is_zero(b)) return a;

    // Add implicit bit for normalized numbers
    if (expA != 0) {
        sigA.x |= 0x00100000u;
    } else {
        // Subnormal: normalize
        expA = 1;
    }
    if (expB != 0) {
        sigB.x |= 0x00100000u;
    } else {
        expB = 1;
    }

    // Shift significands left by 10 bits for guard/round/sticky bits
    sigA = shl64(sigA, 10);
    sigB = shl64(sigB, 10);

    // Align exponents by shifting the smaller one
    int expDiff = expA - expB;
    int expZ;
    sf64 sigZ;
    bool signZ;

    if (expDiff > 0) {
        sigB = shr64_jam(sigB, expDiff);
        expZ = expA;
    } else if (expDiff < 0) {
        sigA = shr64_jam(sigA, -expDiff);
        expZ = expB;
    } else {
        expZ = expA;
    }

    // Add or subtract based on signs
    if (signA == signB) {
        // Same sign: add magnitudes
        signZ = signA;
        sigZ = add64(sigA, sigB);

        // Check for overflow (carry out)
        if ((sigZ.x & 0x00400000u) != 0) {
            sigZ = shr64_jam(sigZ, 1);
            expZ++;
        }
    } else {
        // Different signs: subtract magnitudes
        int cmp = cmp64(sigA, sigB);
        if (cmp == 0) {
            // Exact cancellation
            return SF64_ZERO;  // Positive zero for round-to-nearest
        }
        if (cmp > 0) {
            signZ = signA;
            sigZ = sub64(sigA, sigB);
        } else {
            signZ = signB;
            sigZ = sub64(sigB, sigA);
        }

        // Normalize: shift left until leading bit is in position
        int shift = clz64(sigZ) - 11;
        if (shift > 0) {
            sigZ = shl64(sigZ, shift);
            expZ -= shift;
        }
    }

    // Extract rounding bits and shift down
    uint round_bits = sigZ.y & 0x7FFu;
    sigZ = shr64(sigZ, 11);

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

sf64 sf64_mul(sf64 a, sf64 b) {
    bool signA = sf64_sign(a);
    bool signB = sf64_sign(b);
    bool signZ = signA != signB;

    int expA = sf64_exp_raw(a);
    int expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a);
    sf64 sigB = sf64_sig(b);

    // Handle NaN
    if (sf64_is_nan(a) || sf64_is_nan(b)) return SF64_QNAN;

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

    // Add implicit bit for normalized numbers
    if (expA != 0) {
        sigA.x |= 0x00100000u;
    } else {
        // Subnormal: normalize
        int shift = sf64_normalize_subnormal(sigA);
        expA = 1 - shift;
    }
    if (expB != 0) {
        sigB.x |= 0x00100000u;
    } else {
        int shift = sf64_normalize_subnormal(sigB);
        expB = 1 - shift;
    }

    // Compute result exponent: expA + expB - bias
    int expZ = expA + expB - SF64_EXP_BIAS;

    // Shift significands for multiplication
    // sigA has 53 bits (1.52), sigB has 53 bits
    // After shifting: sigA << 10 and sigB << 11 gives proper alignment
    sigA = shl64(sigA, 10);
    sigB = shl64(sigB, 11);

    // 64x64 -> 128 multiplication
    uint128_t prod = mul64x64(sigA, sigB);

    // The product has leading 1 at bit 125 or 126 (depending on whether
    // the two 1.xxx values multiplied to >= 2.0)
    // We need to get the top 53 bits plus rounding info

    sf64 sigZ = sf64(prod.w3, prod.w2);

    // Include sticky bits from low 64 bits
    uint sticky = ((prod.w1 | prod.w0) != 0) ? 1u : 0u;

    // Normalize: check if leading bit is at position 63 or 62
    if ((sigZ.x & 0x80000000u) == 0) {
        // Leading bit at position 62, shift left
        sigZ = shl64(sigZ, 1);
        sigZ.y |= (prod.w1 >> 31);
        expZ--;
    }

    // Now leading bit is at position 63
    // We need bits 63-11 for the result (53 bits)
    // Bits 10-0 are round/sticky

    uint round_bits = ((sigZ.y & 0x7FFu) | sticky);
    sigZ = shr64(sigZ, 11);

    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ============================================================================
// Fused Multiply-Add (FMA)
// ============================================================================

// Note: This is a simplified FMA using mul + add (two roundings)
// A true IEEE FMA would keep the full product precision before adding
sf64 sf64_fma(sf64 a, sf64 b, sf64 c) {
    // For most practical purposes, mul + add is sufficient
    // A true single-rounding FMA is much more complex
    sf64 prod = sf64_mul(a, b);
    return sf64_add(prod, c);
}

// ============================================================================
// Comparison
// ============================================================================

// Returns -1 if a < b, 0 if a == b, 1 if a > b
// NaN comparisons return 0 (unordered)
inline int sf64_compare(sf64 a, sf64 b) {
    if (sf64_is_nan(a) || sf64_is_nan(b)) return 0;

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
    if (sf64_is_nan(a) || sf64_is_nan(b)) return false;
    if (sf64_is_zero(a) && sf64_is_zero(b)) return true;
    return (a.x == b.x) && (a.y == b.y);
}

inline bool sf64_lt(sf64 a, sf64 b) {
    return sf64_compare(a, b) < 0;
}

inline bool sf64_le(sf64 a, sf64 b) {
    return sf64_compare(a, b) <= 0;
}

inline bool sf64_gt(sf64 a, sf64 b) {
    return sf64_compare(a, b) > 0;
}

inline bool sf64_ge(sf64 a, sf64 b) {
    return sf64_compare(a, b) >= 0;
}

// ============================================================================
// Matrix Multiplication Kernel
// ============================================================================

// Each thread computes a 4x4 sub-matrix of C for optimal performance
// Based on Metal Performance Testing optimization techniques
constant uint SF64_TILE_SIZE = 4;

kernel void matmul_sf64(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    // Each thread handles a 4x4 tile of output
    uint baseRow = gid.y * SF64_TILE_SIZE;
    uint baseCol = gid.x * SF64_TILE_SIZE;

    // Accumulator for 4x4 tile (16 sf64 values)
    sf64 acc[SF64_TILE_SIZE][SF64_TILE_SIZE];
    for (uint i = 0; i < SF64_TILE_SIZE; i++) {
        for (uint j = 0; j < SF64_TILE_SIZE; j++) {
            acc[i][j] = SF64_ZERO;
        }
    }

    // Compute dot products for the 4x4 tile
    for (uint k = 0; k < K; k++) {
        // Load 4 elements from A column (for this k)
        sf64 a_vals[SF64_TILE_SIZE];
        for (uint i = 0; i < SF64_TILE_SIZE; i++) {
            uint row = baseRow + i;
            a_vals[i] = (row < M) ? A[row * K + k] : SF64_ZERO;
        }

        // Load 4 elements from B row (for this k)
        sf64 b_vals[SF64_TILE_SIZE];
        for (uint j = 0; j < SF64_TILE_SIZE; j++) {
            uint col = baseCol + j;
            b_vals[j] = (col < N) ? B[k * N + col] : SF64_ZERO;
        }

        // Accumulate 4x4 outer product: acc += a_vals * b_vals^T
        for (uint i = 0; i < SF64_TILE_SIZE; i++) {
            for (uint j = 0; j < SF64_TILE_SIZE; j++) {
                acc[i][j] = sf64_fma(a_vals[i], b_vals[j], acc[i][j]);
            }
        }
    }

    // Store results to C
    for (uint i = 0; i < SF64_TILE_SIZE; i++) {
        for (uint j = 0; j < SF64_TILE_SIZE; j++) {
            uint row = baseRow + i;
            uint col = baseCol + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

#endif // ESHKOL_METAL_SOFTFLOAT_H
