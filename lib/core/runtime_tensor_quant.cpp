/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * GGUF quantized-weight dequantization (ESH-0025).
 *
 * Kimi K2.6 ships as a GGUF model: experts q4_0 (~97% of the weights),
 * attention/shared/head q8_0. To consume those weights directly, Eshkol reads
 * the packed GGUF block layout from a byte buffer (e.g. an mmap'd tensor) and
 * dequantizes into f64 bit patterns (tensor element storage), which the f16
 * cuBLAS GemmEx path (ESH-0021) then consumes.
 *
 * GGUF block layouts (llama.cpp/ggml convention, QK=32 weights per block):
 *   q8_0: [ f16 scale d ][ 32 × int8 q ]              = 34 bytes/block
 *         x[i] = d * q[i]
 *   q4_0: [ f16 scale d ][ 16 × uint8 (packed nibbles) ] = 18 bytes/block
 *         for j in 0..15: x[j] = d*((qs[j] & 0xF) - 8),  x[j+16] = d*((qs[j]>>4) - 8)
 *
 * These are called from LLVM-generated code through extern "C" names.
 */

#include "arena_memory.h"

#include <cstdint>
#include <cstring>

namespace {

// IEEE-754 half (binary16) bit pattern -> float. Handles subnormals/inf/nan.
float half_bits_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;  // signed zero
        } else {
            // Subnormal: normalise into a single-precision normal.
            uint32_t e = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; e--; }
            mant &= 0x3FFu;
            f = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        f = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Little-endian 2-byte half scale at the start of each block.
inline uint16_t read_half(const uint8_t* p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

}  // namespace

extern "C" {

// Dequantize `n` weights of GGUF q8_0 into f64 bit patterns in `out`.
void eshkol_dequant_q8_0(const uint8_t* blocks, int64_t* out, int64_t n) {
    if (!blocks || !out || n <= 0) return;
    double* o = reinterpret_cast<double*>(out);
    const int QK = 32, BLOCK = 34;
    const int64_t nb = (n + QK - 1) / QK;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk = blocks + b * BLOCK;
        const float d = half_bits_to_float(read_half(blk));
        const int8_t* qs = reinterpret_cast<const int8_t*>(blk + 2);
        for (int j = 0; j < QK; j++) {
            const int64_t idx = b * QK + j;
            if (idx >= n) return;
            o[idx] = (double)(d * (float)qs[j]);
        }
    }
}

// Dequantize `n` weights of GGUF q4_0 into f64 bit patterns in `out`.
void eshkol_dequant_q4_0(const uint8_t* blocks, int64_t* out, int64_t n) {
    if (!blocks || !out || n <= 0) return;
    double* o = reinterpret_cast<double*>(out);
    const int QK = 32, BLOCK = 18;
    const int64_t nb = (n + QK - 1) / QK;
    for (int64_t b = 0; b < nb; b++) {
        const uint8_t* blk = blocks + b * BLOCK;
        const float d = half_bits_to_float(read_half(blk));
        const uint8_t* qs = blk + 2;
        for (int j = 0; j < 16; j++) {
            const int x0 = (int)(qs[j] & 0x0F) - 8;
            const int x1 = (int)(qs[j] >> 4)   - 8;
            const int64_t i0 = b * QK + j;
            const int64_t i1 = b * QK + j + 16;
            if (i0 < n) o[i0] = (double)(d * (float)x0);
            if (i1 < n) o[i1] = (double)(d * (float)x1);
        }
    }
}

// GGUF block sizes (bytes) for a buffer of `n` weights — lets callers size the
// byte buffer / validate an mmap'd region. QK=32 weights per block.
int64_t eshkol_q8_0_buffer_bytes(int64_t n) {
    return n <= 0 ? 0 : ((n + 31) / 32) * 34;
}
int64_t eshkol_q4_0_buffer_bytes(int64_t n) {
    return n <= 0 ? 0 : ((n + 31) / 32) * 18;
}

}  // extern "C"
