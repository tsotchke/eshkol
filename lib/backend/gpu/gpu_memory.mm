/*
 * Cross-Platform GPU Memory Implementation for Eshkol
 *
 * Provides unified GPU acceleration across:
 * - Metal (macOS/iOS) with unified memory
 * - CUDA (Linux/Windows) with discrete GPU memory
 * - Fallback to CPU when no GPU available
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/gpu/gpu_memory.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

// ============================================================================
// Platform Detection
// ============================================================================

// Metal (macOS/iOS with Apple GPU)
#if defined(__APPLE__) && defined(ESHKOL_GPU_METAL_ENABLED)
#define ESHKOL_GPU_METAL_AVAILABLE 1
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

// CUDA (NVIDIA GPU)
#if defined(ESHKOL_GPU_CUDA_ENABLED)
#define ESHKOL_GPU_CUDA_AVAILABLE 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

// ============================================================================
// Global State
// ============================================================================

// GPU threshold (elements) - default 100K
size_t g_gpu_threshold = 100000;

// Active backend
static EshkolGPUBackend g_active_backend = ESHKOL_GPU_NONE;
static bool g_gpu_initialized = false;

// ============================================================================
// Metal Backend State
// ============================================================================

#if ESHKOL_GPU_METAL_AVAILABLE

static id<MTLDevice> g_metal_device = nil;
static id<MTLCommandQueue> g_metal_queue = nil;
static id<MTLComputePipelineState> g_matmul_sf64_pipeline = nil;
static id<MTLLibrary> g_metal_library = nil;
static bool g_metal_unified_memory = false;

// ============================================================================
// SoftFloat IEEE 754 f64 Emulation for Metal
// ============================================================================
// Apple Silicon GPUs lack native f64 hardware. We emulate f64 using uint2
// (two 32-bit integers) with full IEEE 754 compliance including:
// - 52-bit mantissa precision (bit-exact with CPU)
// - Proper rounding (round-to-nearest-even)
// - Special value handling (zero, infinity, NaN)
// Based on Berkeley SoftFloat library algorithms.

static NSString* g_matmul_sf64_source = @R"(
#include <metal_stdlib>
using namespace metal;

// ===== IEEE 754 Double-Precision as uint2 =====
typedef uint2 sf64;  // .x = high 32 bits, .y = low 32 bits

// Constants
constant sf64 SF64_ZERO = sf64(0x00000000u, 0x00000000u);
constant sf64 SF64_NEG_ZERO = sf64(0x80000000u, 0x00000000u);
constant sf64 SF64_INF = sf64(0x7FF00000u, 0x00000000u);
constant sf64 SF64_NEG_INF = sf64(0xFFF00000u, 0x00000000u);
constant sf64 SF64_QNAN = sf64(0x7FF80000u, 0x00000000u);

// ===== Bit Extraction =====
inline bool sf64_sign(sf64 x) { return (x.x >> 31) != 0; }
inline int sf64_exp_raw(sf64 x) { return int((x.x >> 20) & 0x7FFu); }
inline sf64 sf64_sig(sf64 x) { return sf64(x.x & 0x000FFFFFu, x.y); }

// ===== Classification =====
inline bool sf64_is_zero(sf64 x) { return ((x.x & 0x7FFFFFFFu) == 0) && (x.y == 0); }
inline bool sf64_is_inf(sf64 x) { return ((x.x & 0x7FFFFFFFu) == 0x7FF00000u) && (x.y == 0); }
inline bool sf64_is_nan(sf64 x) {
    return ((x.x & 0x7FF00000u) == 0x7FF00000u) &&
           (((x.x & 0x000FFFFFu) != 0) || (x.y != 0));
}

// ===== Packing =====
inline sf64 sf64_pack(bool sign, int exp_raw, uint mant_hi, uint mant_lo) {
    uint hi = (sign ? 0x80000000u : 0u) | ((uint(exp_raw) & 0x7FFu) << 20) | (mant_hi & 0x000FFFFFu);
    return sf64(hi, mant_lo);
}
inline sf64 sf64_negate(sf64 x) { return sf64(x.x ^ 0x80000000u, x.y); }

// ===== 64-bit Arithmetic =====
inline sf64 shl64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) return sf64(x.y << (n - 32), 0u);
    return sf64((x.x << n) | (x.y >> (32 - n)), x.y << n);
}

inline sf64 shr64(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return SF64_ZERO;
    if (n >= 32) return sf64(0u, x.x >> (n - 32));
    return sf64(x.x >> n, (x.y >> n) | (x.x << (32 - n)));
}

inline sf64 shr64_jam(sf64 x, int n) {
    if (n <= 0) return x;
    if (n >= 64) return sf64(0u, ((x.x | x.y) != 0) ? 1u : 0u);
    if (n >= 32) {
        uint lost = x.y | ((n > 32) ? (x.x << (64 - n)) : 0u);
        return sf64(0u, (x.x >> (n - 32)) | ((lost != 0) ? 1u : 0u));
    }
    uint lost = x.y << (32 - n);
    return sf64(x.x >> n, ((x.y >> n) | (x.x << (32 - n))) | ((lost != 0) ? 1u : 0u));
}

inline sf64 add64(sf64 a, sf64 b) {
    uint lo = a.y + b.y;
    uint carry = (lo < a.y) ? 1u : 0u;
    return sf64(a.x + b.x + carry, lo);
}

inline sf64 sub64(sf64 a, sf64 b) {
    uint lo = a.y - b.y;
    uint borrow = (a.y < b.y) ? 1u : 0u;
    return sf64(a.x - b.x - borrow, lo);
}

inline int cmp64(sf64 a, sf64 b) {
    if (a.x != b.x) return (a.x < b.x) ? -1 : 1;
    if (a.y != b.y) return (a.y < b.y) ? -1 : 1;
    return 0;
}

inline int clz64(sf64 x) {
    if (x.x != 0) return clz(x.x);
    if (x.y != 0) return 32 + clz(x.y);
    return 64;
}

// ===== 64x64 -> 128 bit multiply =====
struct uint128_t { uint w3, w2, w1, w0; };

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

// ===== Rounding =====
// round_bits: 10 bits of rounding info, half-way point at 0x200
inline sf64 sf64_round_pack(bool sign, int exp_raw, sf64 sig, uint round_bits) {
    // Round to nearest, ties to even
    bool round_up = (round_bits > 0x200u) || ((round_bits == 0x200u) && ((sig.y & 1u) != 0));
    if (round_up) {
        sig = add64(sig, sf64(0u, 1u));
        // Check if rounding caused overflow (implicit bit moved from bit 52 to bit 53)
        if ((sig.x & 0x00200000u) != 0) { sig = shr64(sig, 1); exp_raw++; }
    }
    if (exp_raw >= 2047) return sign ? SF64_NEG_INF : SF64_INF;
    if (exp_raw <= 0) {
        int shift = 1 - exp_raw;
        if (shift >= 64) return sign ? SF64_NEG_ZERO : SF64_ZERO;
        sig = shr64_jam(sig, shift);
        exp_raw = 0;
    }
    return sf64_pack(sign, exp_raw, sig.x, sig.y);
}

// ===== Addition =====
sf64 sf64_add(sf64 a, sf64 b) {
    bool signA = sf64_sign(a), signB = sf64_sign(b);
    int expA = sf64_exp_raw(a), expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a), sigB = sf64_sig(b);

    if (sf64_is_nan(a)) return SF64_QNAN;
    if (sf64_is_nan(b)) return SF64_QNAN;
    if (sf64_is_inf(a)) {
        if (sf64_is_inf(b) && (signA != signB)) return SF64_QNAN;
        return a;
    }
    if (sf64_is_inf(b)) return b;
    if (sf64_is_zero(a)) {
        if (sf64_is_zero(b)) return (signA && signB) ? SF64_NEG_ZERO : SF64_ZERO;
        return b;
    }
    if (sf64_is_zero(b)) return a;

    // Add implicit bit for normalized numbers
    if (expA != 0) sigA.x |= 0x00100000u; else expA = 1;
    if (expB != 0) sigB.x |= 0x00100000u; else expB = 1;

    // Shift to have leading 1 at bit 63 (for 11 guard bits)
    sigA = shl64(sigA, 11);
    sigB = shl64(sigB, 11);

    // Align exponents
    int expDiff = expA - expB;
    int expZ;
    if (expDiff > 0) { sigB = shr64_jam(sigB, expDiff); expZ = expA; }
    else if (expDiff < 0) { sigA = shr64_jam(sigA, -expDiff); expZ = expB; }
    else expZ = expA;

    sf64 sigZ;
    bool signZ;
    if (signA == signB) {
        // Same sign: add magnitudes
        signZ = signA;
        sigZ = add64(sigA, sigB);
        // Check for overflow: sum wrapped if result < either operand
        if (cmp64(sigZ, sigA) < 0 || cmp64(sigZ, sigB) < 0) {
            // Overflow: true sum >= 2^64, leading 1 at virtual bit 64
            // Shift right by 2 to normalize to bit 62, increment exp
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
        if (cmp == 0) return SF64_ZERO;
        if (cmp > 0) { signZ = signA; sigZ = sub64(sigA, sigB); }
        else { signZ = signB; sigZ = sub64(sigB, sigA); }
        // Normalize: shift left until leading 1 is at bit 62 (not 63)
        int shift = clz64(sigZ) - 1;  // -1 to target bit 62 instead of 63
        if (shift > 0) { sigZ = shl64(sigZ, shift); expZ -= shift; }
        else if (shift < 0) { sigZ = shr64_jam(sigZ, -shift); expZ -= shift; }
    }
    // Now leading 1 is at bit 62 (bit 30 of sigZ.x)

    // Extract 10 round bits and shift by 10 to get mantissa at bit 52
    uint round_bits = sigZ.y & 0x3FFu;
    sigZ = shr64(sigZ, 10);
    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ===== Multiplication =====
sf64 sf64_mul(sf64 a, sf64 b) {
    bool signA = sf64_sign(a), signB = sf64_sign(b);
    bool signZ = signA != signB;
    int expA = sf64_exp_raw(a), expB = sf64_exp_raw(b);
    sf64 sigA = sf64_sig(a), sigB = sf64_sig(b);

    if (sf64_is_nan(a) || sf64_is_nan(b)) return SF64_QNAN;
    if (sf64_is_inf(a)) {
        if (sf64_is_zero(b)) return SF64_QNAN;
        return signZ ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_inf(b)) {
        if (sf64_is_zero(a)) return SF64_QNAN;
        return signZ ? SF64_NEG_INF : SF64_INF;
    }
    if (sf64_is_zero(a) || sf64_is_zero(b)) return signZ ? SF64_NEG_ZERO : SF64_ZERO;

    // Add implicit bit for normalized numbers
    if (expA != 0) sigA.x |= 0x00100000u;
    else { int s = clz64(sigA) - 11; sigA = shl64(sigA, s); expA = 1 - s; }
    if (expB != 0) sigB.x |= 0x00100000u;
    else { int s = clz64(sigB) - 11; sigB = shl64(sigB, s); expB = 1 - s; }

    // Compute result exponent (may need adjustment after normalization)
    int expZ = expA + expB - 1023;

    // Shift both significands to have leading 1 at bit 63
    // After multiply, product has leading 1 at bit 126 (no overflow) or 127 (overflow)
    sigA = shl64(sigA, 11);
    sigB = shl64(sigB, 11);

    uint128_t prod = mul64x64(sigA, sigB);
    sf64 sigZ = sf64(prod.w3, prod.w2);
    uint sticky = ((prod.w1 | prod.w0) != 0) ? 1u : 0u;

    // Normalize: product leading 1 is at bit 126 or 127 (bits 62 or 63 of high 64)
    if ((sigZ.x & 0x80000000u) != 0) {
        // Overflow: product in [2,4), leading 1 at bit 63
        // Shift right by 1 to normalize to bit 62, increment exponent
        sticky |= (sigZ.y & 1u);
        sigZ = shr64(sigZ, 1);
        expZ++;
    }
    // Now leading 1 is at bit 62 (bit 30 of sigZ.x) in both cases

    // Extract round bits (bits 9-0) and shift by 10 to get mantissa at bit 52
    uint round_bits = (sigZ.y & 0x3FFu) | sticky;
    sigZ = shr64(sigZ, 10);
    return sf64_round_pack(signZ, expZ, sigZ, round_bits);
}

// ===== FMA =====
sf64 sf64_fma(sf64 a, sf64 b, sf64 c) {
    return sf64_add(sf64_mul(a, b), c);
}

// ===== Matrix Multiplication Kernel (4x4 tiling) =====
constant uint TILE_SIZE = 4;

kernel void matmul_sf64(
    device const sf64* A [[buffer(0)]],
    device const sf64* B [[buffer(1)]],
    device sf64* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& K [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint baseRow = gid.y * TILE_SIZE;
    uint baseCol = gid.x * TILE_SIZE;

    sf64 acc[TILE_SIZE][TILE_SIZE];
    for (uint i = 0; i < TILE_SIZE; i++)
        for (uint j = 0; j < TILE_SIZE; j++)
            acc[i][j] = SF64_ZERO;

    for (uint k = 0; k < K; k++) {
        sf64 a_vals[TILE_SIZE];
        sf64 b_vals[TILE_SIZE];
        for (uint i = 0; i < TILE_SIZE; i++)
            a_vals[i] = (baseRow + i < M) ? A[(baseRow + i) * K + k] : SF64_ZERO;
        for (uint j = 0; j < TILE_SIZE; j++)
            b_vals[j] = (baseCol + j < N) ? B[k * N + (baseCol + j)] : SF64_ZERO;

        for (uint i = 0; i < TILE_SIZE; i++)
            for (uint j = 0; j < TILE_SIZE; j++)
                acc[i][j] = sf64_fma(a_vals[i], b_vals[j], acc[i][j]);
    }

    for (uint i = 0; i < TILE_SIZE; i++)
        for (uint j = 0; j < TILE_SIZE; j++)
            if (baseRow + i < M && baseCol + j < N)
                C[(baseRow + i) * N + (baseCol + j)] = acc[i][j];
}
)";

static int metal_init(void) {
    @autoreleasepool {
        g_metal_device = MTLCreateSystemDefaultDevice();
        if (!g_metal_device) return -1;

        g_metal_queue = [g_metal_device newCommandQueue];
        if (!g_metal_queue) {
            g_metal_device = nil;
            return -1;
        }

        // Check for unified memory (Apple Silicon)
        if (@available(macOS 10.15, *)) {
            g_metal_unified_memory = [g_metal_device supportsFamily:MTLGPUFamilyApple1];
        }

        // Compile sf64 (SoftFloat) matmul shader
        // This uses uint2 to emulate f64 with full IEEE 754 compliance
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        // Disable fast-math to ensure exact IEEE 754 rounding behavior
        options.fastMathEnabled = NO;

        g_metal_library = [g_metal_device newLibraryWithSource:g_matmul_sf64_source
                                                       options:options
                                                         error:&error];
        if (!g_metal_library) {
            fprintf(stderr, "Metal: Failed to compile sf64 shader: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;  // GPU detected but shader failed, fall back to CPU
        }

        id<MTLFunction> func = [g_metal_library newFunctionWithName:@"matmul_sf64"];
        if (!func) {
            fprintf(stderr, "Metal: Failed to find matmul_sf64 kernel\n");
            return 0;
        }

        g_matmul_sf64_pipeline = [g_metal_device newComputePipelineStateWithFunction:func
                                                                               error:&error];
        if (!g_matmul_sf64_pipeline) {
            fprintf(stderr, "Metal: Failed to create sf64 pipeline: %s\n",
                    [[error localizedDescription] UTF8String]);
            return 0;
        }

        return 0;
    }
}

static void metal_shutdown(void) {
    @autoreleasepool {
        g_matmul_sf64_pipeline = nil;
        g_metal_library = nil;
        g_metal_queue = nil;
        g_metal_device = nil;
    }
}

static int metal_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    @autoreleasepool {
        MTLResourceOptions options;
        switch (mem_type) {
            case ESHKOL_MEM_DEVICE:
                options = MTLResourceStorageModePrivate;
                break;
            case ESHKOL_MEM_HOST_PINNED:
                options = g_metal_unified_memory ? MTLResourceStorageModeShared
                                                  : MTLResourceStorageModeManaged;
                break;
            default:
                options = MTLResourceStorageModeShared;
                break;
        }

        id<MTLBuffer> buffer = [g_metal_device newBufferWithLength:size_bytes options:options];
        if (!buffer) return -1;

        out->size_bytes = size_bytes;
        out->mem_type = mem_type;
        out->backend = ESHKOL_GPU_METAL;
        out->backend_data = (__bridge_retained void*)buffer;

        if (options != MTLResourceStorageModePrivate) {
            out->host_ptr = [buffer contents];
            out->device_ptr = out->host_ptr;
        } else {
            out->host_ptr = nullptr;
            out->device_ptr = (__bridge void*)buffer;
        }

        return 0;
    }
}

static void metal_free(EshkolGPUBuffer* buffer) {
    @autoreleasepool {
        if (buffer->backend_data) {
            id<MTLBuffer> mtl_buf = (__bridge_transfer id<MTLBuffer>)buffer->backend_data;
            mtl_buf = nil;
        }
    }
}

static int metal_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    @autoreleasepool {
        // Unified memory: no sync needed
        if (g_metal_unified_memory && buffer->mem_type != ESHKOL_MEM_DEVICE) {
            return 0;
        }

        id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buffer->backend_data;

        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            if (@available(macOS 10.11, *)) {
                [mtl_buf didModifyRange:NSMakeRange(0, buffer->size_bytes)];
            }
        }

        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
            id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
            if (@available(macOS 10.11, *)) {
                [blit synchronizeResource:mtl_buf];
            }
            [blit endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
        }

        return 0;
    }
}

// ============================================================================
// f64 <-> sf64 Conversion (bit-exact reinterpretation)
// ============================================================================
// IEEE 754 f64 bits are split into two uint32: high word (.x) and low word (.y)

static void convert_f64_to_sf64(const double* src, uint32_t* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t bits;
        memcpy(&bits, &src[i], sizeof(double));
        dst[i * 2] = static_cast<uint32_t>(bits >> 32);      // High word
        dst[i * 2 + 1] = static_cast<uint32_t>(bits);        // Low word
    }
}

static void convert_sf64_to_f64(const uint32_t* src, double* dst, size_t count) {
    for (size_t i = 0; i < count; i++) {
        uint64_t bits = (static_cast<uint64_t>(src[i * 2]) << 32) | src[i * 2 + 1];
        memcpy(&dst[i], &bits, sizeof(double));
    }
}

static int metal_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        if (!g_matmul_sf64_pipeline) return -1;

        size_t elementsA = M * K;
        size_t elementsB = K * N;
        size_t elementsC = M * N;

        // Allocate GPU buffers for sf64 format (8 bytes per element, same as f64)
        // Using uint2 representation: each f64 becomes (high32, low32)
        id<MTLBuffer> buf_a = [g_metal_device newBufferWithLength:elementsA * 8
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [g_metal_device newBufferWithLength:elementsB * 8
                                                          options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_c = [g_metal_device newBufferWithLength:elementsC * 8
                                                          options:MTLResourceStorageModeShared];

        if (!buf_a || !buf_b || !buf_c) {
            return -1;
        }

        // Convert f64 inputs to sf64 format (bit reinterpretation)
        convert_f64_to_sf64(static_cast<const double*>(A->host_ptr),
                            static_cast<uint32_t*>([buf_a contents]), elementsA);
        convert_f64_to_sf64(static_cast<const double*>(B->host_ptr),
                            static_cast<uint32_t*>([buf_b contents]), elementsB);

        // Create command buffer
        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd computeCommandEncoder];

        [encoder setComputePipelineState:g_matmul_sf64_pipeline];
        [encoder setBuffer:buf_a offset:0 atIndex:0];
        [encoder setBuffer:buf_b offset:0 atIndex:1];
        [encoder setBuffer:buf_c offset:0 atIndex:2];

        // Pass dimensions as bytes
        uint32_t dims[3] = {static_cast<uint32_t>(M), static_cast<uint32_t>(K), static_cast<uint32_t>(N)};
        [encoder setBytes:&dims[0] length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&dims[1] length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&dims[2] length:sizeof(uint32_t) atIndex:5];

        // Dispatch with 4x4 tiling: each thread handles a 4x4 tile
        // Total threads needed = (N/4, M/4), one thread per tile
        const uint32_t TILE_SIZE = 4;
        NSUInteger tilesX = (N + TILE_SIZE - 1) / TILE_SIZE;
        NSUInteger tilesY = (M + TILE_SIZE - 1) / TILE_SIZE;
        MTLSize threadsTotal = MTLSizeMake(tilesX, tilesY, 1);
        MTLSize threadgroupSize = MTLSizeMake(8, 8, 1);

        // dispatchThreads dispatches exactly threadsTotal threads
        [encoder dispatchThreads:threadsTotal threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // Convert sf64 result back to f64
        convert_sf64_to_f64(static_cast<const uint32_t*>([buf_c contents]),
                            static_cast<double*>(C->host_ptr), elementsC);

        return 0;
    }
}

static int metal_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    @autoreleasepool {
        id<MTLBuffer> buf_a = (__bridge id<MTLBuffer>)A->backend_data;
        id<MTLBuffer> buf_b = (__bridge id<MTLBuffer>)B->backend_data;
        id<MTLBuffer> buf_c = (__bridge id<MTLBuffer>)C->backend_data;

        // Use MPS for f32 (highly optimized)
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K
                                                                           rowBytes:K * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor matrixDescriptorWithRows:K columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:N
                                                                           rowBytes:N * sizeof(float)
                                                                           dataType:MPSDataTypeFloat32];

        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:buf_a descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:buf_b descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:buf_c descriptor:desc_c];

        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_metal_device
            transposeLeft:NO transposeRight:NO
            resultRows:M resultColumns:N interiorColumns:K
            alpha:1.0 beta:0.0];

        id<MTLCommandBuffer> cmd = [g_metal_queue commandBuffer];
        [matmul encodeToCommandBuffer:cmd leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cmd commit];
        [cmd waitUntilCompleted];

        return 0;
    }
}

static int metal_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    @autoreleasepool {
        id<MTLBuffer> buffer = [g_metal_device newBufferWithBytesNoCopy:host_ptr
                                                                 length:size_bytes
                                                                options:MTLResourceStorageModeShared
                                                            deallocator:nil];
        if (!buffer) {
            // Fallback: allocate and copy
            int result = metal_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
            if (result != 0) return result;
            memcpy(out->host_ptr, host_ptr, size_bytes);
            return 0;
        }

        out->host_ptr = host_ptr;
        out->device_ptr = host_ptr;
        out->size_bytes = size_bytes;
        out->mem_type = ESHKOL_MEM_UNIFIED;
        out->backend = ESHKOL_GPU_METAL;
        out->flags = 1;  // Wrapped, don't free underlying memory
        out->backend_data = (__bridge_retained void*)buffer;

        return 0;
    }
}

#endif // ESHKOL_GPU_METAL_AVAILABLE

// ============================================================================
// CUDA Backend State
// ============================================================================

#if ESHKOL_GPU_CUDA_AVAILABLE

static cudaStream_t g_cuda_stream = nullptr;
static cublasHandle_t g_cublas_handle = nullptr;

static int cuda_init(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) return -1;

    err = cudaSetDevice(0);
    if (err != cudaSuccess) return -1;

    err = cudaStreamCreate(&g_cuda_stream);
    if (err != cudaSuccess) return -1;

    cublasStatus_t status = cublasCreate(&g_cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_cuda_stream);
        return -1;
    }

    cublasSetStream(g_cublas_handle, g_cuda_stream);

    return 0;
}

static void cuda_shutdown(void) {
    if (g_cublas_handle) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    if (g_cuda_stream) {
        cudaStreamDestroy(g_cuda_stream);
        g_cuda_stream = nullptr;
    }
}

static int cuda_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out) {
    cudaError_t err;
    void* ptr = nullptr;

    switch (mem_type) {
        case ESHKOL_MEM_UNIFIED:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_DEVICE:
            err = cudaMalloc(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = nullptr;
            out->device_ptr = ptr;
            break;

        case ESHKOL_MEM_HOST_PINNED:
            err = cudaMallocHost(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = nullptr;  // Need separate device alloc for pinned
            break;

        default:
            err = cudaMallocManaged(&ptr, size_bytes);
            if (err != cudaSuccess) return -1;
            out->host_ptr = ptr;
            out->device_ptr = ptr;
            break;
    }

    out->size_bytes = size_bytes;
    out->mem_type = mem_type;
    out->backend = ESHKOL_GPU_CUDA;
    out->backend_data = nullptr;
    out->flags = 0;

    return 0;
}

static void cuda_free(EshkolGPUBuffer* buffer) {
    switch (buffer->mem_type) {
        case ESHKOL_MEM_HOST_PINNED:
            if (buffer->host_ptr) cudaFreeHost(buffer->host_ptr);
            if (buffer->device_ptr && buffer->device_ptr != buffer->host_ptr) {
                cudaFree(buffer->device_ptr);
            }
            break;
        default:
            if (buffer->device_ptr) cudaFree(buffer->device_ptr);
            break;
    }
}

static int cuda_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (buffer->mem_type == ESHKOL_MEM_UNIFIED) {
        cudaStreamSynchronize(g_cuda_stream);
        return 0;
    }

    if (buffer->mem_type == ESHKOL_MEM_HOST_PINNED && buffer->device_ptr) {
        cudaError_t err;
        if (direction == ESHKOL_SYNC_HOST_TO_DEVICE || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->device_ptr, buffer->host_ptr, buffer->size_bytes,
                                   cudaMemcpyHostToDevice, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        if (direction == ESHKOL_SYNC_DEVICE_TO_HOST || direction == ESHKOL_SYNC_BIDIRECTIONAL) {
            err = cudaMemcpyAsync(buffer->host_ptr, buffer->device_ptr, buffer->size_bytes,
                                   cudaMemcpyDeviceToHost, g_cuda_stream);
            if (err != cudaSuccess) return -1;
        }
        cudaStreamSynchronize(g_cuda_stream);
    }

    return 0;
}

static int cuda_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const double alpha = 1.0;
    const double beta = 0.0;

    // cuBLAS uses column-major, so we compute C = B^T * A^T = (A * B)^T
    // But since we want row-major C, we do: C^T = B * A (in cuBLAS terms)
    // which gives us row-major C = A * B
    cublasStatus_t status = cublasDgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const double*)B->device_ptr, (int)N,
        (const double*)A->device_ptr, (int)K,
        &beta,
        (double*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                            uint64_t M, uint64_t K, uint64_t N) {
    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasStatus_t status = cublasSgemm(g_cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        (int)N, (int)M, (int)K,
        &alpha,
        (const float*)B->device_ptr, (int)N,
        (const float*)A->device_ptr, (int)K,
        &beta,
        (float*)C->device_ptr, (int)N);

    cudaStreamSynchronize(g_cuda_stream);
    return (status == CUBLAS_STATUS_SUCCESS) ? 0 : -1;
}

static int cuda_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out) {
    // Register host memory with CUDA
    cudaError_t err = cudaHostRegister(host_ptr, size_bytes, cudaHostRegisterDefault);
    if (err != cudaSuccess) {
        // Fallback: allocate and copy
        int result = cuda_alloc(size_bytes, ESHKOL_MEM_UNIFIED, out);
        if (result != 0) return result;
        memcpy(out->host_ptr, host_ptr, size_bytes);
        return 0;
    }

    out->host_ptr = host_ptr;
    out->device_ptr = host_ptr;  // Managed access
    out->size_bytes = size_bytes;
    out->mem_type = ESHKOL_MEM_HOST_PINNED;
    out->backend = ESHKOL_GPU_CUDA;
    out->flags = 1;  // Wrapped
    out->backend_data = nullptr;

    return 0;
}

#endif // ESHKOL_GPU_CUDA_AVAILABLE

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

int eshkol_gpu_init(void) {
    if (g_gpu_initialized) {
        return (g_active_backend != ESHKOL_GPU_NONE) ? 1 : 0;
    }

    g_gpu_initialized = true;

    // Try Metal first (macOS)
#if ESHKOL_GPU_METAL_AVAILABLE
    if (metal_init() == 0) {
        g_active_backend = ESHKOL_GPU_METAL;
        return 1;
    }
#endif

    // Try CUDA
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (cuda_init() == 0) {
        g_active_backend = ESHKOL_GPU_CUDA;
        return 1;
    }
#endif

    // No GPU available
    g_active_backend = ESHKOL_GPU_NONE;
    return 0;
}

void eshkol_gpu_shutdown(void) {
#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        metal_shutdown();
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        cuda_shutdown();
    }
#endif

    g_active_backend = ESHKOL_GPU_NONE;
    g_gpu_initialized = false;
}

EshkolGPUBackend eshkol_gpu_get_backend(void) {
    return g_active_backend;
}

const char* eshkol_gpu_backend_name(EshkolGPUBackend backend) {
    switch (backend) {
        case ESHKOL_GPU_NONE: return "CPU (no GPU)";
        case ESHKOL_GPU_METAL: return "Apple Metal";
        case ESHKOL_GPU_CUDA: return "NVIDIA CUDA";
        case ESHKOL_GPU_VULKAN: return "Vulkan";
    }
    return "Unknown";
}

int eshkol_gpu_backend_available(EshkolGPUBackend backend) {
    return (g_active_backend == backend) ? 1 : 0;
}

int eshkol_gpu_supports_f64(void) {
    // Metal does NOT support f64 (double precision) in compute shaders
    // CUDA supports f64 on all modern GPUs
    switch (g_active_backend) {
        case ESHKOL_GPU_CUDA:
            return 1;  // CUDA supports f64
        case ESHKOL_GPU_METAL:
            return 0;  // Metal does NOT support f64
        default:
            return 0;
    }
}

int eshkol_gpu_alloc(size_t size_bytes, EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    if (!out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_alloc(size_bytes, mem_type, out_buffer);
    }
#endif

    // CPU fallback
    out_buffer->host_ptr = malloc(size_bytes);
    if (!out_buffer->host_ptr) return -1;
    out_buffer->device_ptr = out_buffer->host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    return 0;
}

int eshkol_gpu_alloc_aligned(size_t size_bytes, size_t alignment,
                              EshkolMemoryType mem_type, EshkolGPUBuffer* out_buffer) {
    size_t aligned_size = (size_bytes + alignment - 1) & ~(alignment - 1);
    return eshkol_gpu_alloc(aligned_size, mem_type, out_buffer);
}

void eshkol_gpu_free(EshkolGPUBuffer* buffer) {
    if (!buffer) return;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        metal_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        cuda_free(buffer);
        memset(buffer, 0, sizeof(EshkolGPUBuffer));
        return;
    }
#endif

    // CPU fallback
    if (buffer->host_ptr && !(buffer->flags & 1)) {
        free(buffer->host_ptr);
    }
    memset(buffer, 0, sizeof(EshkolGPUBuffer));
}

int eshkol_gpu_wrap_host(void* host_ptr, size_t size_bytes, EshkolGPUBuffer* out_buffer) {
    if (!host_ptr || !out_buffer || size_bytes == 0) return -1;

    memset(out_buffer, 0, sizeof(EshkolGPUBuffer));

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_wrap_host(host_ptr, size_bytes, out_buffer);
    }
#endif

    // CPU: just use the pointer directly
    out_buffer->host_ptr = host_ptr;
    out_buffer->device_ptr = host_ptr;
    out_buffer->size_bytes = size_bytes;
    out_buffer->mem_type = ESHKOL_MEM_HOST;
    out_buffer->backend = ESHKOL_GPU_NONE;
    out_buffer->flags = 1;  // Wrapped
    return 0;
}

int eshkol_gpu_sync(EshkolGPUBuffer* buffer, EshkolSyncDirection direction) {
    if (!buffer) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_METAL) {
        return metal_sync(buffer, direction);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (buffer->backend == ESHKOL_GPU_CUDA) {
        return cuda_sync(buffer, direction);
    }
#endif

    return 0;  // CPU: no sync needed
}

int eshkol_gpu_sync_async(EshkolGPUBuffer* buffer, EshkolSyncDirection direction, void* stream) {
    (void)stream;
    return eshkol_gpu_sync(buffer, direction);
}

void eshkol_gpu_wait(EshkolGPUBuffer* buffer) {
    (void)buffer;
#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA && g_cuda_stream) {
        cudaStreamSynchronize(g_cuda_stream);
    }
#endif
}

int eshkol_gpu_matmul_f64(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f64(A, B, C, M, K, N);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f64(A, B, C, M, K, N);
    }
#endif

    // CPU fallback
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64((const double*)A->host_ptr, (const double*)B->host_ptr,
                      (double*)C->host_ptr, M, K, N);
    return 0;
}

int eshkol_gpu_matmul_f32(EshkolGPUBuffer* A, EshkolGPUBuffer* B, EshkolGPUBuffer* C,
                           uint64_t M, uint64_t K, uint64_t N) {
    if (!A || !B || !C) return -1;

#if ESHKOL_GPU_METAL_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_METAL) {
        return metal_matmul_f32(A, B, C, M, K, N);
    }
#endif

#if ESHKOL_GPU_CUDA_AVAILABLE
    if (g_active_backend == ESHKOL_GPU_CUDA) {
        return cuda_matmul_f32(A, B, C, M, K, N);
    }
#endif

    // CPU fallback: convert and use f64
    // TODO: Add f32 SIMD path
    return -1;
}

void eshkol_gpu_set_threshold(size_t threshold) {
    g_gpu_threshold = threshold;
}

size_t eshkol_gpu_get_threshold(void) {
    return g_gpu_threshold;
}

int eshkol_gpu_should_use(size_t num_elements) {
    return (g_active_backend != ESHKOL_GPU_NONE && num_elements >= g_gpu_threshold) ? 1 : 0;
}

void eshkol_matmul_dispatch(const double* A, const double* B, double* C,
                             uint64_t M, uint64_t K, uint64_t N) {
    size_t num_elements = M * N;

    // GPU path if available and large enough
    if (eshkol_gpu_should_use(num_elements)) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        if (eshkol_gpu_wrap_host((void*)A, M * K * sizeof(double), &buf_a) == 0 &&
            eshkol_gpu_wrap_host((void*)B, K * N * sizeof(double), &buf_b) == 0 &&
            eshkol_gpu_wrap_host((void*)C, M * N * sizeof(double), &buf_c) == 0) {

            if (eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0) {
                eshkol_gpu_free(&buf_a);
                eshkol_gpu_free(&buf_b);
                eshkol_gpu_free(&buf_c);
                return;
            }
        }
        // GPU failed, fall through to CPU
    }

    // CPU fallback (BLAS/SIMD/scalar)
    extern void eshkol_matmul_f64(const double*, const double*, double*, uint64_t, uint64_t, uint64_t);
    eshkol_matmul_f64(A, B, C, M, K, N);
}

} // extern "C"
