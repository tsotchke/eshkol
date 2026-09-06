/*
 * XLA Runtime Implementation for Eshkol
 *
 * Provides runtime support for XLA-compiled tensor operations.
 * Currently delegates to BLAS/SIMD while XLA JIT compilation is developed.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/device_lowering.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <future>
#include <unordered_map>
#include <chrono>
#include <atomic>
#include <mutex>
#include <set>
#include <string>
#include <vector>

// Use the existing BLAS matmul for now
extern "C" {
    void eshkol_matmul_f64(const double* A, const double* B, double* C,
                           uint64_t M, uint64_t K, uint64_t N);
}

// GPU acceleration
#include "eshkol/backend/gpu/gpu_memory.h"

// Optional PJRT device-execution path.
//
// pjrt_client.h wraps the PJRT C API (see that header's DESIGN NOTES) and is
// the actual XLA device runtime — the block below is what turns this file's
// LLVM-direct JIT dispatch into an alternative, opt-in path onto real
// hardware (TPU, or any other backend a PJRT plugin covers).
//
// Every other optional piece in lib/backend/xla/ (xla_codegen.cpp,
// xla_compiler.cpp, xla_types.cpp, stablehlo_emitter.cpp) is gated by a
// file-local macro derived from CMake-provided defines:
//
//     #if defined(ESHKOL_MLIR_AVAILABLE) && defined(ESHKOL_STABLEHLO_AVAILABLE)
//     #define ESHKOL_XLA_FULL_MLIR 1
//     #endif
//
// That pattern needs CMakeLists.txt to define the upstream macros, which it
// does for MLIR/StableHLO but not (yet) for PJRT — deps/pjrt is only added to
// the include path inside the STABLEHLO_ROOT-bundled-LLVM branch, and
// pjrt_client.cpp isn't wired into any source list yet. CMakeLists.txt is
// outside this file's scope for this change. `__has_include` gives the same
// "compiles either way" guarantee without requiring any build-system change:
// it is standard C++ (available since C++17; this project targets C++20) and
// asks the question this file actually cares about — is the header physically
// present in this checkout — directly, rather than through an intermediate
// macro. If PJRT is later given a proper CMake-level macro (mirroring
// ESHKOL_MLIR_AVAILABLE) this can be tightened to match that convention; until
// then this is the smallest change that does not touch CMakeLists.txt.
#if __has_include("eshkol/backend/xla/pjrt_client.h")
#include "eshkol/backend/xla/pjrt_client.h"
#define ESHKOL_XLA_PJRT_AVAILABLE 1
#endif

// Lazy GPU initialization for XLA runtime functions
// Uses std::call_once for thread-safe one-time initialization
static std::once_flag g_xla_gpu_init_flag;

static void ensure_gpu_initialized() {
    std::call_once(g_xla_gpu_init_flag, []() {
        eshkol_gpu_init();
    });
}

// Canonical tensor type + arena allocator.
//
// This file used to carry its own forward-declared copy of `struct
// eshkol_tensor` (4 fields, 32 bytes) instead of including the canonical
// definition in arena_memory.h (5 fields, 40 bytes — dtype was added at idx 4
// after this file was written and never updated to match). That is an ODR
// violation: the same struct tag with two different definitions in the same
// program. It stayed latent here only because every use in this file goes
// through pointers returned by arena_allocate_tensor_full — whose real,
// canonical 40-byte layout is what actually gets allocated — with no
// by-value copy or local sizeof(eshkol_tensor_t)/eshkol_tensor_t-on-the-stack
// use in this file. It also meant `->dtype` could never be set here, so
// every XLA result silently depended on arena_allocate_tensor_full's default
// rather than this file asserting its own dtype.
//
// There is no header-cycle or C/C++ linkage reason to avoid the canonical
// header: it is extern "C"-wrapped throughout and already included from
// other lib/backend/*.cpp translation units (collection_codegen.cpp,
// llvm_codegen.cpp, parallel_codegen.cpp, tensor_backward.cpp,
// tensor_conv_kernel.cpp, thread_pool.cpp). Including it here instead of
// re-declaring removes the divergent copy outright.
#include "../../core/arena_memory.h"

// Compile-time guard: if this TU's view of eshkol_tensor_t is ever no longer
// the canonical (dtype-bearing) one — e.g. a local re-typedef creeps back
// in, or arena_memory.h's own layout assert is bypassed for this TU — fail
// the build here rather than silently truncating tensors allocated by
// arena_allocate_tensor_full.
static_assert(sizeof(eshkol_tensor_t) == 40,
    "xla_runtime.cpp must use the canonical, dtype-bearing eshkol_tensor_t "
    "from arena_memory.h (4 core fields + dtype) — do not reintroduce a "
    "local re-typedef of struct eshkol_tensor");

// All eshkol_xla_* entry points below dispatch to the BLAS/SIMD/GPU f64
// kernels above and only ever read/write `double` data — none of them
// produce or consume reduced-precision (f32/f16/bf16) or dual-number tensor
// storage. So every tensor allocated or mutated in this file is logically
// f64 (ESHKOL_TENSOR_DTYPE_F64). arena_allocate_tensor_full's header
// allocator already defaults a fresh tensor's dtype to F64, but each
// function below sets `result->dtype` explicitly too, so this file's
// correctness does not depend on that default.

// XLA matmul runtime function - called by generated LLVM IR
// Performs matrix multiplication using the existing BLAS/SIMD backend
// Parameters:
//   arena - arena allocator for result tensor
//   a_data, b_data - input tensor data pointers
//   a_shape, b_shape - shape arrays (dimensions)
//   a_rank, b_rank - number of dimensions
// Returns: pointer to result tensor struct
extern "C" void* eshkol_xla_matmul_host(
    void* arena,
    const double* a_data,
    const double* b_data,
    const int64_t* a_shape,
    const int64_t* b_shape,
    int64_t a_rank,
    int64_t b_rank) {

    // Currently only support 2D matmul
    if (a_rank != 2 || b_rank != 2) {
        return nullptr;
    }

    // Extract dimensions: A is MxK, B is KxN, result is MxN
    uint64_t M = static_cast<uint64_t>(a_shape[0]);
    uint64_t K = static_cast<uint64_t>(a_shape[1]);
    uint64_t K2 = static_cast<uint64_t>(b_shape[0]);
    uint64_t N = static_cast<uint64_t>(b_shape[1]);

    // Verify inner dimensions match
    if (K != K2) {
        return nullptr;
    }

    // Allocate result tensor with object header (HEAP_SUBTYPE_TENSOR)
    // arena_allocate_tensor_full gives us: header + tensor struct + dims + elements
    eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), 2, M * N);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;

    // Set dimension sizes
    result->dimensions[0] = M;
    result->dimensions[1] = N;

    // Perform matrix multiplication using GPU/BLAS/SIMD cascade
    // elements is int64_t* but stores doubles as bit patterns — safe to cast
    // since sizeof(double) == sizeof(int64_t) == 8
    double* out = reinterpret_cast<double*>(result->elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for large matrices
    int gpu_should = eshkol_gpu_should_use(M * N);
    if (gpu_should) {
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        // P1: free every successfully-wrapped buffer on ALL exit paths. The old
        // short-circuit &&-chain leaked buf_a when buf_b's wrap failed, and
        // leaked all three when the matmul itself failed before falling back.
        bool wa = eshkol_gpu_wrap_host((void*)a_data, M * K * sizeof(double), &buf_a) == 0;
        bool wb = wa && eshkol_gpu_wrap_host((void*)b_data, K * N * sizeof(double), &buf_b) == 0;
        bool wc = wb && eshkol_gpu_wrap_host((void*)out, M * N * sizeof(double), &buf_c) == 0;
        bool done = wa && wb && wc &&
            eshkol_gpu_matmul_f64(&buf_a, &buf_b, &buf_c, M, K, N) == 0;
        if (wc) eshkol_gpu_free(&buf_c);
        if (wb) eshkol_gpu_free(&buf_b);
        if (wa) eshkol_gpu_free(&buf_a);
        if (done) return result;
    }
    // CPU fallback (BLAS/SIMD)
    eshkol_matmul_f64(a_data, b_data, out, M, K, N);

    return result;
}

// ===== XLA Elementwise Runtime =====
// Applies binary or unary operations element-wise across tensors.
// Op codes match ElementwiseOp enum: ADD=0,SUB=1,MUL=2,DIV=3,
//   EXP=4,LOG=5,SIN=6,COS=7,TANH=8,RELU=9,SIGMOID=10
// The broadcast runtime the CPU path uses. Reused here so the two paths cannot
// disagree about what a broadcast means, and so this path inherits its exact
// output allocation rather than guessing one.
extern "C" int64_t eshkol_broadcast_shape_f64(
    const int64_t* a_dims, int64_t a_ndim, const int64_t* b_dims, int64_t b_ndim,
    int64_t* out_dims, int64_t* out_ndim_out, int64_t* out_total_out);
extern "C" int64_t eshkol_broadcast_elementwise_f64(
    int64_t op,
    const double* a_data, const int64_t* a_dims, int64_t a_ndim,
    const double* b_data, const int64_t* b_dims, int64_t b_ndim,
    double* out_data, int64_t* out_dims, int64_t* out_ndim_out, int64_t* out_total_out);

static bool xla_same_shape(int64_t a_total, const uint64_t* a_shape, int64_t a_rank,
                           int64_t b_total, const uint64_t* b_shape, int64_t b_rank) {
    if (a_total != b_total || a_rank != b_rank) return false;
    for (int64_t i = 0; i < a_rank; i++) if (a_shape[i] != b_shape[i]) return false;
    return true;
}

/**
 * @brief Whether an elementwise op-code takes two operands.
 *
 * ADD..DIV (0..3) are binary and the unary ops follow them, so this test used
 * to be written inline as `op_code <= 3` in every caller. That stopped being
 * true when POW/MAX/MIN were appended at 16..18: the ABI numbering is frozen
 * (see the ElementwiseOp comment in xla_codegen.h), so a new binary op cannot
 * be given a code below 4, and an arity test by range now reads a binary op as
 * unary and dereferences nothing where it should dereference `b`.
 *
 * Exported rather than static so the parity harnesses ask the same question of
 * the same function the runtime asks it of. Two copies of an arity table is
 * how one of them ends up describing a different ABI.
 */
extern "C" int eshkol_xla_elementwise_is_binary(int64_t op_code) {
    switch (op_code) {
        case 0:   // ADD
        case 1:   // SUB
        case 2:   // MUL
        case 3:   // DIV
        case 16:  // POW
        case 17:  // MAX
        case 18:  // MIN
            return 1;
        default:
            return 0;
    }
}

extern "C" void* eshkol_xla_elementwise_host(
    void* arena,
    const double* a_data,
    const double* b_data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t b_total,
    const uint64_t* b_shape,
    int64_t b_rank,
    int64_t op_code) {

    if (total_elements <= 0 || !a_data) return nullptr;
    const bool binary = eshkol_xla_elementwise_is_binary(op_code) != 0;
    if (binary) {
        if (!b_data || !b_shape) return nullptr;
        if (!xla_same_shape(total_elements, shape, rank, b_total, b_shape, b_rank)) {
            // eshkol_broadcast_elementwise_f64 implements ADD/SUB/MUL/DIV only
            // and refuses anything else, so a shape-mismatched POW/MAX/MIN is
            // refused HERE, by name, rather than reaching a helper that would
            // have to report the same refusal with less context. Equal-shape
            // POW/MAX/MIN take the CPU loop below and are fully supported.
            if (op_code > 3) return nullptr;
            // Broadcast, or refuse. Returning null reaches the emitted fallback
            // branch, which takes the CPU path and raises a real type error for
            // shapes that cannot broadcast; nothing is ever read past the end
            // of the smaller operand.
            if (rank > 16 || b_rank > 16) return nullptr;
            int64_t a_dims[16], b_dims[16], out_dims[16], out_ndim = 0, out_total = 0;
            for (int64_t i = 0; i < rank; i++) a_dims[i] = static_cast<int64_t>(shape[i]);
            for (int64_t i = 0; i < b_rank; i++) b_dims[i] = static_cast<int64_t>(b_shape[i]);
            if (eshkol_broadcast_shape_f64(a_dims, rank, b_dims, b_rank, out_dims, &out_ndim, &out_total) != 0) {
                return nullptr;
            }
            eshkol_tensor_t* bres = arena_allocate_tensor_full(
                reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(out_ndim), static_cast<uint64_t>(out_total));
            if (!bres) return nullptr;
            bres->dtype = ESHKOL_TENSOR_DTYPE_F64;
            int64_t written_dims[16], written_ndim = 0, written_total = 0;
            if (eshkol_broadcast_elementwise_f64(op_code, a_data, a_dims, rank, b_data, b_dims, b_rank,
                    reinterpret_cast<double*>(bres->elements), written_dims, &written_ndim, &written_total) != 0) {
                return nullptr;
            }
            for (int64_t i = 0; i < written_ndim; i++) bres->dimensions[i] = static_cast<uint64_t>(written_dims[i]);
            return bres;
        }
    }

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(rank), static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;

    // Copy shape
    for (int64_t i = 0; i < rank; i++) {
        result->dimensions[i] = shape[i];
    }

    double* out = reinterpret_cast<double*>(result->elements);
    uint64_t n = static_cast<uint64_t>(total_elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for large tensors
    // XLA ElementwiseOp: ADD=0,SUB=1,MUL=2,DIV=3,EXP=4,LOG=5,SIN=6,COS=7,TANH=8,RELU=9,SIGMOID=10
    // GPU EshkolElementwiseOp: ADD=0,SUB=1,MUL=2,DIV=3,NEG=4,ABS=5,EXP=6,LOG=7,SIN=8,COS=9,TANH=10,RELU=11,SIGMOID=12
    // GPU enum has NEG(4) and ABS(5) inserted, so XLA unary ops 4-10 map to GPU 6-12
    static const int xla_to_gpu_elemwise[] = {0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12};
    static const int xla_elemwise_count = sizeof(xla_to_gpu_elemwise) / sizeof(xla_to_gpu_elemwise[0]);

    if (eshkol_gpu_should_use(n) && op_code >= 0 && op_code < xla_elemwise_count) {
        int gpu_op = xla_to_gpu_elemwise[op_code];
        EshkolGPUBuffer buf_a, buf_b, buf_c;
        // P1: free every wrapped buffer on ALL paths (partial-wrap failure and
        // kernel failure both previously leaked the buffers before CPU fallback).
        const bool need_b = (b_data && op_code <= 3);
        bool wa = eshkol_gpu_wrap_host((void*)a_data, n * sizeof(double), &buf_a) == 0;
        bool wb = wa && need_b && eshkol_gpu_wrap_host((void*)b_data, n * sizeof(double), &buf_b) == 0;
        bool wc = wa && (!need_b || wb) &&
            eshkol_gpu_wrap_host((void*)out, n * sizeof(double), &buf_c) == 0;
        bool done = wc && eshkol_gpu_elementwise_f64(&buf_a, need_b ? &buf_b : nullptr, &buf_c, n,
                        static_cast<EshkolElementwiseOp>(gpu_op)) == 0;
        if (wc) eshkol_gpu_free(&buf_c);
        if (wb) eshkol_gpu_free(&buf_b);
        if (wa) eshkol_gpu_free(&buf_a);
        if (done) return result;
    }

    // CPU fallback
    // P0: binary ops (ADD/SUB/MUL/DIV, op_code 0-3) dereference b_data; the GPU
    // path above already treats b_data as nullable, so guard the CPU path too
    // rather than dereferencing NULL.
    if (binary && !b_data) return nullptr;
    switch (op_code) {
        case 0: // ADD
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] + b_data[i];
            break;
        case 1: // SUB
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] - b_data[i];
            break;
        case 2: // MUL
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] * b_data[i];
            break;
        case 3: // DIV
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] / b_data[i];
            break;
        case 4: // EXP
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::exp(a_data[i]);
            break;
        case 5: // LOG
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::log(a_data[i]);
            break;
        case 6: // SIN
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::sin(a_data[i]);
            break;
        case 7: // COS
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::cos(a_data[i]);
            break;
        case 8: // TANH
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::tanh(a_data[i]);
            break;
        case 9: // RELU
            for (int64_t i = 0; i < total_elements; i++) out[i] = a_data[i] > 0.0 ? a_data[i] : 0.0;
            break;
        case 10: // SIGMOID
            for (int64_t i = 0; i < total_elements; i++) out[i] = 1.0 / (1.0 + std::exp(-a_data[i]));
            break;
        // Appended for the geometric primitives; see the ElementwiseOp comment
        // in xla_codegen.h for why these codes start at 11 rather than being
        // interleaved with the ops they resemble.
        case 11: // SQRT
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::sqrt(a_data[i]);
            break;
        case 12: // RSQRT
            // 1/sqrt(x), not std::pow(x, -0.5): the two round differently and
            // the device has a reciprocal-sqrt unit this is the reference for.
            for (int64_t i = 0; i < total_elements; i++) out[i] = 1.0 / std::sqrt(a_data[i]);
            break;
        case 13: // ABS
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::fabs(a_data[i]);
            break;
        case 14: // NEG
            for (int64_t i = 0; i < total_elements; i++) out[i] = -a_data[i];
            break;
        case 15: // ATANH
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::atanh(a_data[i]);
            break;
        case 16: // POW
            for (int64_t i = 0; i < total_elements; i++) out[i] = std::pow(a_data[i], b_data[i]);
            break;
        case 17: // MAX
            // Spelled as the same comparison Eshkol's host AD rule for max is
            // spelled with (AD_NODE_MAX in lib/backend/autodiff_codegen.cpp:
            // strictly greater wins, everything else goes to the right-hand
            // operand). The VALUE is the same either way; writing the forward
            // in the form the derivative was derived from is what keeps the
            // two from being revised apart. NaN behaviour is not specified
            // here and is not exercised by any parity row.
            for (int64_t i = 0; i < total_elements; i++)
                out[i] = (a_data[i] > b_data[i]) ? a_data[i] : b_data[i];
            break;
        case 18: // MIN
            for (int64_t i = 0; i < total_elements; i++)
                out[i] = (a_data[i] < b_data[i]) ? a_data[i] : b_data[i];
            break;
        default:
            return nullptr;
    }

    return result;
}

/**
 * @brief The host answer for an elementwise comparison, as an f64 tensor of
 *        0.0 / 1.0.
 *
 * This is the host reference entry point for the six Compare* device ops. It
 * exists because the host's own `<`, `>`, `=` return #t/#f for scalars and
 * have no tensor form at all, so there was nothing a parity row could call:
 * a comparison could be emitted to the device and never measured. The
 * encoding is the one every predicate takes when it leaves a device graph
 * (i1 converted to the float element type): the region emitter, the single-op
 * executor and this function all agree that a predicate on the host is 0/1.
 *
 * @param direction 0 EQ, 1 NE, 2 LT, 3 LE, 4 GT, 5 GE (the order of the
 *        DeviceOpKind::Compare* enumerators and of stablehlo's directions).
 *
 * Operands broadcast right-aligned, exactly as eshkol_broadcast_shape_f64
 * defines it, so a scalar can be compared against a tensor. Returns NULL for
 * shapes that do not broadcast or a direction outside 0..5.
 */
extern "C" void* eshkol_xla_compare_host(
    void* arena,
    const double* a_data,
    const double* b_data,
    int64_t a_total,
    const uint64_t* a_shape,
    int64_t a_rank,
    int64_t b_total,
    const uint64_t* b_shape,
    int64_t b_rank,
    int64_t direction) {

    if (!arena || !a_data || !b_data || a_total <= 0 || b_total <= 0) return nullptr;
    if (direction < 0 || direction > 5) return nullptr;
    if (a_rank > 16 || b_rank > 16 || a_rank < 0 || b_rank < 0) return nullptr;

    int64_t a_dims[16], b_dims[16], out_dims[16], out_ndim = 0, out_total = 0;
    for (int64_t i = 0; i < a_rank; i++) a_dims[i] = static_cast<int64_t>(a_shape[i]);
    for (int64_t i = 0; i < b_rank; i++) b_dims[i] = static_cast<int64_t>(b_shape[i]);
    if (a_rank == 0) { out_ndim = b_rank; for (int64_t i = 0; i < b_rank; i++) out_dims[i] = b_dims[i]; out_total = b_total; }
    else if (b_rank == 0) { out_ndim = a_rank; for (int64_t i = 0; i < a_rank; i++) out_dims[i] = a_dims[i]; out_total = a_total; }
    else if (eshkol_broadcast_shape_f64(a_dims, a_rank, b_dims, b_rank, out_dims, &out_ndim, &out_total) != 0) {
        return nullptr;
    }

    // A rank-0 answer is allocated the way every other rank-0 result here is:
    // one element, one dimension of extent one (see xla_alloc_result).
    const uint64_t alloc_rank = out_ndim == 0 ? 1u : static_cast<uint64_t>(out_ndim);
    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), alloc_rank, static_cast<uint64_t>(out_total));
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    if (out_ndim == 0) result->dimensions[0] = 1;
    for (int64_t i = 0; i < out_ndim; i++) result->dimensions[i] = static_cast<uint64_t>(out_dims[i]);

    double* out = reinterpret_cast<double*>(result->elements);
    int64_t idx[16] = {0};
    for (int64_t flat = 0; flat < out_total; flat++) {
        // Right-aligned index mapping: an operand dimension of extent 1, or a
        // dimension the operand does not have, contributes nothing to its
        // offset.
        int64_t ai = 0, bi = 0;
        for (int64_t d = 0; d < out_ndim; d++) {
            const int64_t ad = d - (out_ndim - a_rank);
            const int64_t bd = d - (out_ndim - b_rank);
            if (ad >= 0) ai = ai * a_dims[ad] + (a_dims[ad] == 1 ? 0 : idx[d]);
            if (bd >= 0) bi = bi * b_dims[bd] + (b_dims[bd] == 1 ? 0 : idx[d]);
        }
        const double x = a_data[ai], y = b_data[bi];
        bool p = false;
        switch (direction) {
            case 0: p = x == y; break;
            case 1: p = x != y; break;
            case 2: p = x <  y; break;
            case 3: p = x <= y; break;
            case 4: p = x >  y; break;
            case 5: p = x >= y; break;
        }
        out[flat] = p ? 1.0 : 0.0;
        for (int64_t d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < out_dims[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

// ===== XLA Reduce Runtime =====
// Reduces a tensor along an axis (or all axes if axis == -1).
// Op codes match ReduceOp enum: SUM=0,MEAN=1,MAX=2,MIN=3,PROD=4
extern "C" void* eshkol_xla_reduce_host(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t op_code) {

    if (total_elements <= 0 || !data) return nullptr;

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    if (axis == -1) {
        // Reduce all — result is a scalar tensor (rank 1, 1 element)
        eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), 1, 1);
        if (!result) return nullptr;
        result->dtype = ESHKOL_TENSOR_DTYPE_F64;
        result->dimensions[0] = 1;

        double* out = reinterpret_cast<double*>(result->elements);
        uint64_t n = static_cast<uint64_t>(total_elements);

        // Try GPU dispatch for large reductions
        // XLA op codes: SUM=0,MEAN=1,MAX=2,MIN=3,PROD=4
        // GPU op codes: SUM=0,PROD=1,MIN=2,MAX=3,MEAN=4
        if (eshkol_gpu_should_use(n) && op_code >= 0 && op_code <= 4) {
            static const int xla_to_gpu_reduce[] = {0, 4, 3, 2, 1};
            EshkolGPUBuffer buf_in, buf_out;
            // P1: free every wrapped buffer on all paths (partial-wrap / kernel
            // failure previously leaked before CPU fallback).
            bool wi = eshkol_gpu_wrap_host((void*)data, n * sizeof(double), &buf_in) == 0;
            bool wo = wi && eshkol_gpu_wrap_host((void*)out, sizeof(double), &buf_out) == 0;
            bool done = wo && eshkol_gpu_reduce_f64(&buf_in, &buf_out, n,
                            static_cast<EshkolReduceOp>(xla_to_gpu_reduce[op_code])) == 0;
            if (wo) eshkol_gpu_free(&buf_out);
            if (wi) eshkol_gpu_free(&buf_in);
            if (done) return result;
        }

        // CPU fallback
        double acc;
        switch (op_code) {
            case 0: // SUM
                acc = 0.0;
                for (int64_t i = 0; i < total_elements; i++) acc += data[i];
                break;
            case 1: // MEAN
                acc = 0.0;
                for (int64_t i = 0; i < total_elements; i++) acc += data[i];
                acc /= static_cast<double>(total_elements);
                break;
            case 2: // MAX
                acc = data[0];
                for (int64_t i = 1; i < total_elements; i++) acc = std::fmax(acc, data[i]);
                break;
            case 3: // MIN
                acc = data[0];
                for (int64_t i = 1; i < total_elements; i++) acc = std::fmin(acc, data[i]);
                break;
            case 4: // PROD
                acc = 1.0;
                for (int64_t i = 0; i < total_elements; i++) acc *= data[i];
                break;
            default:
                return nullptr;
        }

        out[0] = acc;
        return result;
    }

    // Reduce along specific axis
    if (axis < 0 || axis >= rank) return nullptr;

    // Compute output shape (remove the reduced axis)
    if (rank > 16) return nullptr; // max 16D tensors supported
    uint64_t out_rank = static_cast<uint64_t>(rank - 1);
    if (out_rank == 0) out_rank = 1; // scalar result

    // Compute strides and output dimensions
    uint64_t out_total = 1;
    uint64_t out_dims[16];
    uint64_t j = 0;
    for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
            out_dims[j++] = shape[i];
            out_total *= shape[i];
        }
    }
    if (j == 0) { out_dims[0] = 1; out_total = 1; }

    eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), out_rank, out_total);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (uint64_t i = 0; i < out_rank; i++) result->dimensions[i] = out_dims[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Try GPU dispatch for axis-reduce (large tensors)
    if (eshkol_gpu_should_use(static_cast<size_t>(total_elements)) && op_code >= 0 && op_code <= 4) {
        // XLA op codes: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4
        // GPU op codes: SUM=0, PROD=1, MIN=2, MAX=3, MEAN=4
        static const int xla_to_gpu_reduce[] = {0, 4, 3, 2, 1};
        EshkolGPUBuffer buf_in, buf_out;
        if (eshkol_gpu_wrap_host((void*)data, static_cast<size_t>(total_elements) * sizeof(double), &buf_in) == 0 &&
            eshkol_gpu_wrap_host((void*)out, out_total * sizeof(double), &buf_out) == 0) {
            if (eshkol_gpu_reduce_axis_f64(&buf_in, &buf_out, static_cast<uint64_t>(rank),
                    shape, static_cast<uint64_t>(axis),
                    static_cast<EshkolReduceOp>(xla_to_gpu_reduce[op_code])) == 0) {
                eshkol_gpu_free(&buf_in);
                eshkol_gpu_free(&buf_out);
                return result;
            }
            eshkol_gpu_free(&buf_in);
            eshkol_gpu_free(&buf_out);
        }
    }

    // CPU fallback: compute stride for the reduced axis
    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
    uint64_t outer_stride = axis_len * inner_stride;

    // Reduce
    for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            uint64_t out_idx = outer * inner_stride + inner;
            double acc;
            switch (op_code) {
                case 0: case 1: acc = 0.0; break;
                case 2: acc = -INFINITY; break;
                case 3: acc = INFINITY; break;
                case 4: acc = 1.0; break;
                default: acc = 0.0; break;
            }
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                double val = data[src_idx];
                switch (op_code) {
                    case 0: case 1: acc += val; break;
                    case 2: acc = std::fmax(acc, val); break;
                    case 3: acc = std::fmin(acc, val); break;
                    case 4: acc *= val; break;
                }
            }
            if (op_code == 1) acc /= static_cast<double>(axis_len); // MEAN
            out[out_idx] = acc;
        }
    }

    return result;
}

// ===== XLA Scale In-Place Runtime =====
// Multiplies every element of a tensor data buffer by a scalar.
// Used by MEAN gradient to divide broadcasted gradient by n.
extern "C" void* eshkol_xla_scale_inplace(
    double* data,
    int64_t total_elements,
    double scale) {
    for (int64_t i = 0; i < total_elements; i++) {
        data[i] *= scale;
    }
    return data;
}

// ===== XLA Softmax Runtime =====
// Numerically stable softmax along a specified axis.
// axis == -1 means softmax over all elements (global softmax).
extern "C" void* eshkol_xla_softmax_host(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis) {

    if (total_elements <= 0 || !data) return nullptr;

    // Handle negative axis
    if (axis < -1) axis = axis + rank;

    // Allocate result tensor with same shape
    uint64_t out_rank = static_cast<uint64_t>(rank);
    eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), out_rank, static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // GPU dispatch for contiguous softmax (global or last-axis)
    uint64_t num_slices = 0, slice_len = 0;
    bool gpu_eligible = false;
    if (axis == -1) {
        num_slices = 1;
        slice_len = static_cast<uint64_t>(total_elements);
        gpu_eligible = true;
    } else if (axis >= 0 && axis < rank) {
        // Check if axis is contiguous (inner_stride == 1, i.e. last axis)
        uint64_t inner_stride = 1;
        for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];
        if (inner_stride == 1) {
            slice_len = shape[axis];
            num_slices = static_cast<uint64_t>(total_elements) / slice_len;
            gpu_eligible = true;
        }
    }

    if (gpu_eligible && eshkol_gpu_should_use(static_cast<size_t>(total_elements))) {
        EshkolGPUBuffer in_buf = {const_cast<double*>(data), nullptr,
                                   static_cast<size_t>(total_elements) * sizeof(double),
                                   ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        EshkolGPUBuffer out_buf = {out, nullptr,
                                    static_cast<size_t>(total_elements) * sizeof(double),
                                    ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        if (eshkol_gpu_softmax_f64(&in_buf, &out_buf, num_slices, slice_len) == 0)
            return result;
    }

    if (axis == -1) {
        // Global softmax: max, exp, sum, normalize over all elements
        double max_val = data[0];
        for (int64_t i = 1; i < total_elements; i++)
            if (data[i] > max_val) max_val = data[i];

        double sum_exp = 0.0;
        for (int64_t i = 0; i < total_elements; i++) {
            out[i] = std::exp(data[i] - max_val);
            sum_exp += out[i];
        }
        if (sum_exp == 0.0) sum_exp = 1e-10;
        for (int64_t i = 0; i < total_elements; i++)
            out[i] /= sum_exp;
        return result;
    }

    // Axis-specific softmax
    if (axis < 0 || axis >= rank) return nullptr;

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    uint64_t outer_count = static_cast<uint64_t>(total_elements) / (axis_len * inner_stride);

    // For each "slice" perpendicular to the axis:
    // 1) find max, 2) compute exp(x - max), 3) sum, 4) normalize
    for (uint64_t outer = 0; outer < outer_count; outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            // Step 1: max
            double max_val = -INFINITY;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                if (data[idx] > max_val) max_val = data[idx];
            }
            // Step 2-3: exp and sum
            double sum_exp = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                double e = std::exp(data[idx] - max_val);
                out[idx] = e;
                sum_exp += e;
            }
            // Step 4: normalize
            if (sum_exp == 0.0) sum_exp = 1e-10;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                out[idx] /= sum_exp;
            }
        }
    }
    return result;
}

// ===== XLA Normalize Runtime =====
// Axis-aware normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
// Computes mean and variance along the specified axis.
// gamma and beta are scalar (applied uniformly).
extern "C" void* eshkol_xla_normalize(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    double gamma,
    double beta,
    double epsilon) {

    if (total_elements <= 0 || !data) return nullptr;

    // Handle negative axis
    if (axis < 0) axis = axis + rank;
    if (axis < 0 || axis >= rank) return nullptr;

    // Allocate result tensor with same shape
    eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena),
        static_cast<uint64_t>(rank), static_cast<uint64_t>(total_elements));
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    // GPU dispatch for contiguous normalize (last-axis, inner_stride == 1)
    if (inner_stride == 1 && eshkol_gpu_should_use(static_cast<size_t>(total_elements))) {
        uint64_t num_slices = static_cast<uint64_t>(total_elements) / axis_len;
        EshkolGPUBuffer in_buf = {const_cast<double*>(data), nullptr,
                                   static_cast<size_t>(total_elements) * sizeof(double),
                                   ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        EshkolGPUBuffer out_buf = {out, nullptr,
                                    static_cast<size_t>(total_elements) * sizeof(double),
                                    ESHKOL_MEM_HOST, ESHKOL_GPU_NONE, 0, nullptr};
        if (eshkol_gpu_normalize_f64(&in_buf, &out_buf, num_slices, axis_len,
                                      gamma, beta, epsilon) == 0)
            return result;
    }

    uint64_t outer_count = static_cast<uint64_t>(total_elements) / (axis_len * inner_stride);

    // For each slice perpendicular to the axis:
    // 1) compute mean, 2) compute variance, 3) normalize
    for (uint64_t outer = 0; outer < outer_count; outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            // Mean
            double sum = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                sum += data[idx];
            }
            double mean = sum / static_cast<double>(axis_len);

            // Variance
            double var_sum = 0.0;
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                double diff = data[idx] - mean;
                var_sum += diff * diff;
            }
            double var = var_sum / static_cast<double>(axis_len);

            // Normalize: y = gamma * (x - mean) / sqrt(var + eps) + beta
            double inv_std = 1.0 / std::sqrt(var + epsilon);
            for (uint64_t k = 0; k < axis_len; k++) {
                uint64_t idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                out[idx] = gamma * (data[idx] - mean) * inv_std + beta;
            }
        }
    }
    return result;
}

// ===== XLA Argreduce Runtime =====
// Returns tensor of indices (as doubles) for argmax/argmin along an axis.
// axis == -1 means argreduce over all elements (returns scalar index as double).
extern "C" void* eshkol_xla_argreduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t is_max) {  // 1=argmax, 0=argmin

    if (total_elements <= 0 || !data) return nullptr;

    if (axis == -1) {
        // Argreduce all — return scalar tensor with flattened index as double
        eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), 1, 1);
        if (!result) return nullptr;
        result->dtype = ESHKOL_TENSOR_DTYPE_F64;
        result->dimensions[0] = 1;

        int64_t best_idx = 0;
        double best_val = data[0];
        for (int64_t i = 1; i < total_elements; i++) {
            bool better = is_max ? (data[i] > best_val) : (data[i] < best_val);
            if (better) { best_val = data[i]; best_idx = i; }
        }
        double idx_as_double = static_cast<double>(best_idx);
        reinterpret_cast<double*>(result->elements)[0] = idx_as_double;
        return result;
    }

    // Argreduce along specific axis
    if (axis < 0 || axis >= rank) return nullptr;
    if (rank > 16) return nullptr; // max 16D tensors supported

    uint64_t out_rank = static_cast<uint64_t>(rank - 1);
    if (out_rank == 0) out_rank = 1;

    uint64_t out_total = 1;
    uint64_t out_dims[16];
    uint64_t j = 0;
    for (int64_t i = 0; i < rank; i++) {
        if (i != axis) {
            out_dims[j++] = shape[i];
            out_total *= shape[i];
        }
    }
    if (j == 0) { out_dims[0] = 1; out_total = 1; }

    eshkol_tensor_t* result = arena_allocate_tensor_full(reinterpret_cast<arena_t*>(arena), out_rank, out_total);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (uint64_t i = 0; i < out_rank; i++) result->dimensions[i] = out_dims[i];

    double* out = reinterpret_cast<double*>(result->elements);
    uint64_t axis_len = shape[axis];
    uint64_t inner_stride = 1;
    for (int64_t i = axis + 1; i < rank; i++) inner_stride *= shape[i];

    for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
        for (uint64_t inner = 0; inner < inner_stride; inner++) {
            uint64_t out_idx = outer * inner_stride + inner;
            double best_val = data[outer * axis_len * inner_stride + inner];
            int64_t best_k = 0;
            for (uint64_t k = 1; k < axis_len; k++) {
                uint64_t src_idx = outer * axis_len * inner_stride + k * inner_stride + inner;
                bool better = is_max ? (data[src_idx] > best_val) : (data[src_idx] < best_val);
                if (better) { best_val = data[src_idx]; best_k = static_cast<int64_t>(k); }
            }
            out[out_idx] = static_cast<double>(best_k);
        }
    }
    return result;
}

// ===== XLA Reduce Gradient Runtime =====
// Computes the gradient of a reduce operation w.r.t. the input.
// Supports MAX, MIN, PROD gradients (SUM/MEAN handled in codegen).
// Parameters:
//   arena       - arena allocator
//   grad_data   - upstream gradient elements (reduced shape)
//   input_data  - original input elements (full shape)
//   input_shape - shape of the original input
//   input_rank  - rank of the original input
//   total_input - total elements in input
//   axis        - axis that was reduced (-1 for reduce-all)
//   op_code     - 2=MAX, 3=MIN, 4=PROD
// Returns: tensor* with same shape as input, containing gradient
extern "C" void* eshkol_xla_reduce_gradient(
    void* arena,
    const double* grad_data,
    const double* input_data,
    const uint64_t* input_shape,
    int64_t input_rank,
    int64_t total_input,
    int64_t axis,
    int64_t op_code) {

    if (!grad_data || !input_data || total_input <= 0) return nullptr;

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(input_rank), static_cast<uint64_t>(total_input));
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < input_rank; i++) {
        result->dimensions[i] = input_shape[i];
    }
    double* out = reinterpret_cast<double*>(result->elements);

    if (axis == -1) {
        // Reduce-all gradient
        if (op_code == 2 || op_code == 3) {
            // MAX/MIN: gradient is upstream_grad / count where input == extremum
            double extremum = input_data[0];
            for (int64_t i = 1; i < total_input; i++) {
                if (op_code == 2) extremum = std::fmax(extremum, input_data[i]);
                else extremum = std::fmin(extremum, input_data[i]);
            }
            int64_t count = 0;
            for (int64_t i = 0; i < total_input; i++) {
                if (input_data[i] == extremum) count++;
            }
            double grad_val = (count > 0) ? grad_data[0] / static_cast<double>(count) : 0.0;
            for (int64_t i = 0; i < total_input; i++) {
                out[i] = (input_data[i] == extremum) ? grad_val : 0.0;
            }
        } else if (op_code == 4) {
            // PROD: gradient[i] = prod(x_j for j!=i) * upstream_grad
            // = total_product / x[i] * upstream_grad
            // Handle zeros: count zeros, if >1 all grads are 0
            int64_t zero_count = 0;
            int64_t zero_idx = -1;
            double total_product = 1.0;
            for (int64_t i = 0; i < total_input; i++) {
                if (input_data[i] == 0.0) {
                    zero_count++;
                    zero_idx = i;
                } else {
                    total_product *= input_data[i];
                }
            }
            if (zero_count > 1) {
                // More than one zero: all gradients are 0
                for (int64_t i = 0; i < total_input; i++) out[i] = 0.0;
            } else if (zero_count == 1) {
                // Exactly one zero: only the zero element gets a gradient
                for (int64_t i = 0; i < total_input; i++) out[i] = 0.0;
                out[zero_idx] = total_product * grad_data[0];
            } else {
                // No zeros: grad[i] = total_product / x[i] * upstream
                double upstream = grad_data[0];
                for (int64_t i = 0; i < total_input; i++) {
                    out[i] = (total_product / input_data[i]) * upstream;
                }
            }
        }
    } else {
        // Axis-specific gradient (same logic but per-slice along axis)
        if (axis < 0 || axis >= input_rank) return nullptr;

        uint64_t axis_len = input_shape[axis];
        uint64_t inner_stride = 1;
        for (int64_t i = axis + 1; i < input_rank; i++) inner_stride *= input_shape[i];
        uint64_t outer_stride = axis_len * inner_stride;
        uint64_t out_total = static_cast<uint64_t>(total_input) / axis_len;

        for (uint64_t outer = 0; outer < out_total / (inner_stride > 0 ? inner_stride : 1); outer++) {
            for (uint64_t inner = 0; inner < inner_stride; inner++) {
                uint64_t grad_idx = outer * inner_stride + inner;

                if (op_code == 2 || op_code == 3) {
                    // Find extremum along this slice
                    double extremum = input_data[outer * outer_stride + inner];
                    for (uint64_t k = 1; k < axis_len; k++) {
                        double val = input_data[outer * outer_stride + k * inner_stride + inner];
                        if (op_code == 2) extremum = std::fmax(extremum, val);
                        else extremum = std::fmin(extremum, val);
                    }
                    int64_t count = 0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        if (input_data[outer * outer_stride + k * inner_stride + inner] == extremum) count++;
                    }
                    double gv = (count > 0) ? grad_data[grad_idx] / static_cast<double>(count) : 0.0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                        out[src_idx] = (input_data[src_idx] == extremum) ? gv : 0.0;
                    }
                } else if (op_code == 4) {
                    // PROD gradient along axis
                    int64_t zc = 0;
                    int64_t zi = -1;
                    double tp = 1.0;
                    for (uint64_t k = 0; k < axis_len; k++) {
                        double val = input_data[outer * outer_stride + k * inner_stride + inner];
                        if (val == 0.0) { zc++; zi = static_cast<int64_t>(k); }
                        else tp *= val;
                    }
                    double upstream = grad_data[grad_idx];
                    for (uint64_t k = 0; k < axis_len; k++) {
                        uint64_t src_idx = outer * outer_stride + k * inner_stride + inner;
                        if (zc > 1) {
                            out[src_idx] = 0.0;
                        } else if (zc == 1) {
                            out[src_idx] = (static_cast<int64_t>(k) == zi) ? tp * upstream : 0.0;
                        } else {
                            out[src_idx] = (tp / input_data[src_idx]) * upstream;
                        }
                    }
                }
            }
        }
    }

    return result;
}

// ===== XLA Transpose Runtime =====
// Transposes a 2D tensor (matrix transpose).
// For higher-rank tensors, perm specifies the permutation of axes.
extern "C" void* eshkol_xla_transpose_host(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* perm) {

    if (!data || rank <= 0) return nullptr;
    if (rank > 16) return nullptr;  // P1: out_shape[16] is a fixed stack array

    // Compute transposed shape and total elements
    uint64_t total = 1;
    uint64_t out_shape[16];
    for (int64_t i = 0; i < rank; i++) {
        const int64_t p = perm[i];
        if (p < 0 || p >= rank) return nullptr;  // P1: validate permutation index ∈ [0,rank)
        out_shape[i] = shape[p];
        total *= out_shape[i];
    }

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(rank), total);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = out_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Ensure GPU subsystem is initialized before checking dispatch
    ensure_gpu_initialized();

    // Try GPU dispatch for 2D transpose of large matrices
    if (rank == 2 && eshkol_gpu_should_use(total)) {
        uint64_t rows = shape[0];
        uint64_t cols = shape[1];
        EshkolGPUBuffer buf_in, buf_out;
        if (eshkol_gpu_wrap_host((void*)data, total * sizeof(double), &buf_in) == 0 &&
            eshkol_gpu_wrap_host((void*)out, total * sizeof(double), &buf_out) == 0) {
            if (eshkol_gpu_transpose_f64(&buf_in, &buf_out, rows, cols) == 0) {
                eshkol_gpu_free(&buf_in);
                eshkol_gpu_free(&buf_out);
                return result;
            }
        }
    }

    // CPU fallback — general N-dimensional transpose
    // Compute source strides
    uint64_t src_strides[16];
    src_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute destination strides
    uint64_t dst_strides[16];
    dst_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        dst_strides[i] = dst_strides[i + 1] * out_shape[i + 1];
    }

    // Transpose via index mapping
    for (uint64_t flat = 0; flat < total; flat++) {
        // Convert flat index to multi-dimensional indices in output space
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < rank; d++) {
            uint64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            // This output dimension d corresponds to source dimension perm[d]
            src_flat += idx * src_strides[perm[d]];
        }
        out[flat] = data[src_flat];
    }

    return result;
}

// ===== XLA Broadcast Runtime =====
// Broadcasts a tensor from src_shape to tgt_shape.
extern "C" void* eshkol_xla_broadcast_host(
    void* arena,
    const double* data,
    const uint64_t* src_shape,
    int64_t src_rank,
    const uint64_t* tgt_shape,
    int64_t tgt_rank) {

    if (!data || tgt_rank <= 0) return nullptr;
    if (tgt_rank > 16 || src_rank > 16) return nullptr;  // P1: tgt_strides[16]/src_strides[16] stack arrays
    if (src_rank < 0) return nullptr;
    if (src_rank > tgt_rank) return nullptr;  // SW-22: broadcasting never reduces rank

    // SW-22: validate NumPy-style broadcast compatibility per right-aligned
    // dimension pair before computing any strides. A source dimension may
    // broadcast only if it equals the corresponding target dimension or
    // equals 1 (docs/breakdown/XLA_BACKEND.md "Broadcast Semantics"; mirrors
    // XLATypes::broadcastShape() in lib/backend/xla/xla_types.cpp, which
    // already applies this rule at compile time — this is the runtime's own
    // guard for callers that reach this entry point directly). Previously
    // nothing checked this here: an incompatible source dimension > 1 fed
    // straight into the stride multiplication below, and whenever that
    // dimension was NARROWER than the target it produced an out-of-bounds
    // read of `data` — a memory-safety defect, not merely a wrong answer.
    {
        const int64_t rank_offset = tgt_rank - src_rank;
        for (int64_t d = 0; d < tgt_rank; d++) {
            const int64_t src_d = d - rank_offset;
            if (src_d < 0) continue;  // implicit leading 1 — always compatible
            const uint64_t src_dim = src_shape[src_d];
            const uint64_t tgt_dim = tgt_shape[d];
            if (src_dim != tgt_dim && src_dim != 1) return nullptr;
        }
    }

    uint64_t total = 1;
    for (int64_t i = 0; i < tgt_rank; i++) total *= tgt_shape[i];

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(tgt_rank), total);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < tgt_rank; i++) result->dimensions[i] = tgt_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Compute source strides (right-aligned with target)
    int64_t offset = tgt_rank - src_rank;

    // Compute target strides
    uint64_t tgt_strides[16];
    tgt_strides[tgt_rank - 1] = 1;
    for (int64_t i = tgt_rank - 2; i >= 0; i--) {
        tgt_strides[i] = tgt_strides[i + 1] * tgt_shape[i + 1];
    }

    // Compute source strides
    uint64_t src_strides[16];
    if (src_rank > 0) {
        src_strides[src_rank - 1] = 1;
        for (int64_t i = src_rank - 2; i >= 0; i--) {
            src_strides[i] = src_strides[i + 1] * src_shape[i + 1];
        }
    }

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < tgt_rank; d++) {
            uint64_t idx = remaining / tgt_strides[d];
            remaining %= tgt_strides[d];
            int64_t src_d = d - offset;
            if (src_d >= 0 && src_d < src_rank && src_shape[src_d] > 1) {
                src_flat += idx * src_strides[src_d];
            }
        }
        out[flat] = data[src_flat];
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// Device dispatch: the five entry points below try StableHLO-on-PJRT first
// ═══════════════════════════════════════════════════════════════════════════
//
// Each of eshkol_xla_matmul / _elementwise / _reduce / _transpose / _broadcast
// is now a two-line function that asks the device, and on any answer other
// than "computed it" calls the *_host function above — which is the ENTIRE
// implementation that used to carry the public name, unchanged.
//
// WHY THE SPLIT IS A RENAME AND NOT A FLAG.
//
// The parity harness has to compare the same computation on both paths. If the
// only entry point were the public one, obtaining the host reference would
// mean toggling global state (an environment variable, a mutable switch) and
// hoping nothing else in the process observed the toggle. With the host
// implementation reachable under its own name, the harness calls
// eshkol_xla_add_host() for the reference and the public entry point for the
// device answer, in the same process, with no global mutated between them.
// There is nothing to get out of sync, and the thing being compared is the
// code Eshkol programs actually run rather than a copy of it written for the
// test.
//
// FALLBACK POLICY. A device failure falls back to the host and reports itself
// on stderr ONCE per op kind. Falling back keeps a program correct on a host
// where the plugin is missing a feature; reporting keeps that from being
// invisible, which is the failure mode that matters — a device path that
// silently never runs looks exactly like a device path that works.

namespace {

/** @brief Report a device failure to stderr once per op kind. */
void xla_report_device_failure(const char* op, const std::string& reason) {
    static std::mutex mutex;
    static std::set<std::string> reported;
    std::lock_guard<std::mutex> lock(mutex);
    if (!reported.insert(op).second) return;
    std::fprintf(stderr,
        "eshkol: XLA device path: '%s' fell back to the host runtime: %s\n"
        "        (further '%s' fallbacks are not repeated)\n",
        op, reason.c_str(), op);
}

/**
 * @brief Run one request on the device, or return false having said why.
 *
 * Returns false without any diagnostic when device execution was never
 * requested: that is the default configuration and is not an anomaly.
 */
bool xla_device_try(const eshkol::xla::DeviceOpRequest& request,
                    const std::vector<const double*>& operands,
                    double* out) {
    if (!eshkol::xla::deviceExecutionRequested()) return false;
    eshkol::xla::DeviceExecutor* executor = eshkol::xla::deviceExecutor();
    if (!executor) {
        xla_report_device_failure(eshkol::xla::deviceOpKindName(request.kind),
            "device execution was requested (ESHKOL_XLA_PJRT=1) but no device "
            "executor is installed in this binary; only a build that links the "
            "StableHLO emitter can install one (see device_lowering.h)");
        return false;
    }
    std::string error;
    if (executor->run(request, operands, out, &error)) return true;
    xla_report_device_failure(eshkol::xla::deviceOpKindName(request.kind), error);
    return false;
}

/** @brief Shape vector from the uint64 shape arrays the C ABI passes. */
std::vector<int64_t> xla_shape_of(const uint64_t* dims, int64_t rank) {
    std::vector<int64_t> shape;
    shape.reserve(rank > 0 ? static_cast<size_t>(rank) : 0u);
    for (int64_t i = 0; i < rank; i++) shape.push_back(static_cast<int64_t>(dims[i]));
    return shape;
}

/** @brief Allocate the f64 result tensor the device will be asked to fill. */
eshkol_tensor_t* xla_alloc_result(void* arena, const std::vector<int64_t>& shape,
                                  int64_t total) {
    const uint64_t rank = shape.empty() ? 1u : static_cast<uint64_t>(shape.size());
    eshkol_tensor_t* t = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), rank, static_cast<uint64_t>(total));
    if (!t) return nullptr;
    t->dtype = ESHKOL_TENSOR_DTYPE_F64;
    if (shape.empty()) {
        t->dimensions[0] = 1;
    } else {
        for (size_t i = 0; i < shape.size(); i++) {
            t->dimensions[i] = static_cast<uint64_t>(shape[i]);
        }
    }
    return t;
}

int64_t xla_num_elements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────
// Region execution seam.
//
// generated code -> eshkol_xla_region() [here, slim archive]
//                -> the installed RegionRunFn [region_execution.cpp, MLIR]
//
// Same shape as the DeviceExecutor split above and for the same link-time
// reason. The pointer lives here so it exists in every build and starts null;
// an AOT binary that links only this archive therefore has no region path at
// all, which is correct, because such a binary was never compiled with a
// region call in it.
// ─────────────────────────────────────────────────────────────────────────
namespace {
eshkol::xla::RegionRunFn g_region_runner = nullptr;
}

namespace eshkol {
namespace xla {
RegionRunFn regionRunner() { return g_region_runner; }
void setRegionRunner(RegionRunFn runner) { g_region_runner = runner; }
}  // namespace xla
}  // namespace eshkol

/**
 * @brief Execute region @p region_id over @p operand_tensors.
 *
 * Called by generated code where the outlined subtree used to be evaluated.
 * Returns the region's result as a tensor, or NULL.
 *
 * NULL IS NOT A FALLBACK SIGNAL. Every other eshkol_xla_* entry point here
 * returns NULL to mean "take the host path", because each of them has a host
 * path for the same op sitting beside it. This one does not: the subtree it
 * replaced is gone from the generated code, so there is nothing to fall back
 * to, and a caller that treated NULL as "use the host answer" would be using
 * no answer at all. It is a hard failure, and it says so on stderr rather than
 * letting a null propagate into arithmetic.
 */
extern "C" void* eshkol_xla_region(void* arena, int64_t region_id,
                                   int64_t num_operands,
                                   void* const* operand_tensors) {
    if (!arena || num_operands < 0 || (num_operands > 0 && !operand_tensors)) {
        std::fprintf(stderr, "eshkol: region %lld called with bad arguments\n",
                     static_cast<long long>(region_id));
        return nullptr;
    }
    eshkol::xla::RegionRunFn runner = eshkol::xla::regionRunner();
    if (!runner) {
        std::fprintf(stderr,
                     "eshkol: region %lld was compiled into this program but no "
                     "region runner is installed\n",
                     static_cast<long long>(region_id));
        return nullptr;
    }

    std::vector<std::vector<int64_t>> shapes;
    std::vector<const double*> operands;
    shapes.reserve(static_cast<size_t>(num_operands));
    operands.reserve(static_cast<size_t>(num_operands));
    for (int64_t i = 0; i < num_operands; ++i) {
        auto* t = static_cast<eshkol_tensor_t*>(operand_tensors[i]);
        if (!t) {
            std::fprintf(stderr, "eshkol: region %lld operand %lld is null\n",
                         static_cast<long long>(region_id),
                         static_cast<long long>(i));
            return nullptr;
        }
        std::vector<int64_t> shape;
        for (uint64_t d = 0; d < t->num_dimensions; ++d)
            shape.push_back(static_cast<int64_t>(t->dimensions[d]));
        shapes.push_back(std::move(shape));
        operands.push_back(reinterpret_cast<const double*>(t->elements));
    }

    std::vector<double> result;
    std::vector<int64_t> result_shape;
    std::string error;
    if (!runner(region_id, shapes, operands, &result, &result_shape, &error)) {
        std::fprintf(stderr, "eshkol: region %lld did not run on the device: %s\n",
                     static_cast<long long>(region_id), error.c_str());
        return nullptr;
    }

    eshkol_tensor_t* out = xla_alloc_result(
        arena, result_shape, static_cast<int64_t>(result.size()));
    if (!out) {
        std::fprintf(stderr, "eshkol: region %lld result could not be allocated\n",
                     static_cast<long long>(region_id));
        return nullptr;
    }
    double* dst = reinterpret_cast<double*>(out->elements);
    for (size_t i = 0; i < result.size(); ++i) dst[i] = result[i];
    return out;
}

/**
 * @brief Box a host scalar as a one-element f64 tensor, for a region input
 *        whose shape is rank 0.
 *
 * Generated code can hand eshkol_xla_region() only tensor pointers, and a
 * scalar Eshkol value (a threshold, a step count, a reduction's result) is a
 * tagged double, not a tensor. The box is how it crosses; the region runner
 * knows from the region's own input shapes that the value is a scalar and
 * builds the module over rank 0, not over [1] (see runRegisteredRegion).
 */
extern "C" void* eshkol_xla_scalar_tensor(void* arena, double value) {
    if (!arena) return nullptr;
    eshkol_tensor_t* t = xla_alloc_result(arena, {}, 1);
    if (!t) return nullptr;
    reinterpret_cast<double*>(t->elements)[0] = value;
    return t;
}

extern "C" void* eshkol_xla_matmul(
    void* arena,
    const double* a_data,
    const double* b_data,
    const int64_t* a_shape,
    const int64_t* b_shape,
    int64_t a_rank,
    int64_t b_rank) {

    if (eshkol::xla::deviceExecutionRequested() && arena && a_data && b_data &&
        a_rank == 2 && b_rank == 2 && a_shape[1] == b_shape[0]) {
        eshkol::xla::DeviceOpRequest request;
        request.kind = eshkol::xla::DeviceOpKind::Matmul;
        request.operand_shapes = {{a_shape[0], a_shape[1]}, {b_shape[0], b_shape[1]}};
        request.result_shape = {a_shape[0], b_shape[1]};
        eshkol_tensor_t* result = xla_alloc_result(
            arena, request.result_shape, xla_num_elements(request.result_shape));
        if (result &&
            xla_device_try(request, {a_data, b_data},
                           reinterpret_cast<double*>(result->elements))) {
            return result;
        }
    }
    return eshkol_xla_matmul_host(arena, a_data, b_data, a_shape, b_shape, a_rank, b_rank);
}

extern "C" void* eshkol_xla_elementwise(
    void* arena,
    const double* a_data,
    const double* b_data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t b_total,
    const uint64_t* b_shape,
    int64_t b_rank,
    int64_t op_code) {

    // XLA ElementwiseOp: ADD=0,SUB=1,MUL=2,DIV=3,EXP=4,LOG=5,SIN=6,COS=7,
    // TANH=8,RELU=9,SIGMOID=10,SQRT=11,RSQRT=12,ABS=13,NEG=14,ATANH=15,
    // POW=16,MAX=17,MIN=18.
    //
    // RELU (9) is here because it is now lowered (a maximum against a zero
    // splat) AND measured with an input that lands exactly on zero, which is
    // where its tie convention becomes visible. Nothing is routed to the
    // device before both of those are true.
    //
    // A table indexed by op_code rather than a switch would need a filler
    // entry for any code that is not lowered, and a filler that named a real
    // op would route that code to it. So this is a lookup that can say "no".
    auto device_kind_for = [](int64_t op, eshkol::xla::DeviceOpKind* out) -> bool {
        using K = eshkol::xla::DeviceOpKind;
        switch (op) {
            case 0:  *out = K::Add;      return true;
            case 1:  *out = K::Subtract; return true;
            case 2:  *out = K::Multiply; return true;
            case 3:  *out = K::Divide;   return true;
            case 4:  *out = K::Exp;      return true;
            case 5:  *out = K::Log;      return true;
            case 6:  *out = K::Sin;      return true;
            case 7:  *out = K::Cos;      return true;
            case 8:  *out = K::Tanh;     return true;
            case 9:  *out = K::Relu;     return true;
            case 10: *out = K::Sigmoid;  return true;
            case 11: *out = K::Sqrt;     return true;
            case 12: *out = K::Rsqrt;    return true;
            case 13: *out = K::Abs;      return true;
            case 14: *out = K::Negate;   return true;
            case 15: *out = K::Atanh;    return true;
            case 16: *out = K::Pow;      return true;
            case 17: *out = K::Maximum;  return true;
            case 18: *out = K::Minimum;  return true;
            default: return false;
        }
    };
    eshkol::xla::DeviceOpKind device_kind = eshkol::xla::DeviceOpKind::Add;

    if (eshkol::xla::deviceExecutionRequested() && arena && a_data &&
        total_elements > 0 && rank > 0 && device_kind_for(op_code, &device_kind)) {
        const bool binary = eshkol_xla_elementwise_is_binary(op_code) != 0;
        if (!binary || (b_data && b_shape && b_rank > 0)) {
            eshkol::xla::DeviceOpRequest request;
            request.kind = device_kind;
            std::vector<const double*> operands;
            request.operand_shapes.push_back(xla_shape_of(shape, rank));
            operands.push_back(a_data);
            if (binary) {
                request.operand_shapes.push_back(xla_shape_of(b_shape, b_rank));
                operands.push_back(b_data);
            }
            // Result shape: the broadcast of the operands for a binary op, the
            // operand shape for a unary one. The executor recomputes this
            // independently and refuses the request if it disagrees, so an
            // error here becomes a host fallback rather than a wrong answer.
            std::vector<int64_t> out_shape = request.operand_shapes[0];
            if (binary) {
                const std::vector<int64_t>& x = request.operand_shapes[0];
                const std::vector<int64_t>& y = request.operand_shapes[1];
                const size_t out_rank = x.size() > y.size() ? x.size() : y.size();
                out_shape.assign(out_rank, 1);
                bool ok = true;
                for (size_t i = 0; i < out_rank && ok; i++) {
                    const size_t xo = out_rank - x.size();
                    const size_t yo = out_rank - y.size();
                    const int64_t xd = i < xo ? 1 : x[i - xo];
                    const int64_t yd = i < yo ? 1 : y[i - yo];
                    if (xd != yd && xd != 1 && yd != 1) ok = false;
                    out_shape[i] = xd > yd ? xd : yd;
                }
                if (!ok) out_shape.clear();
            }
            if (!out_shape.empty()) {
                request.result_shape = out_shape;
                eshkol_tensor_t* result = xla_alloc_result(
                    arena, out_shape, xla_num_elements(out_shape));
                if (result &&
                    xla_device_try(request, operands,
                                   reinterpret_cast<double*>(result->elements))) {
                    return result;
                }
            }
        }
    }
    return eshkol_xla_elementwise_host(arena, a_data, b_data, total_elements, shape, rank,
                                       b_total, b_shape, b_rank, op_code);
}

extern "C" void* eshkol_xla_reduce(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis,
    int64_t op_code) {

    // XLA ReduceOp: SUM=0, MEAN=1, MAX=2, MIN=3, PROD=4.
    static const int kDeviceKind[] = {
        static_cast<int>(eshkol::xla::DeviceOpKind::ReduceSum),
        static_cast<int>(eshkol::xla::DeviceOpKind::ReduceMean),
        static_cast<int>(eshkol::xla::DeviceOpKind::ReduceMax),
        static_cast<int>(eshkol::xla::DeviceOpKind::ReduceMin),
        static_cast<int>(eshkol::xla::DeviceOpKind::ReduceProd),
    };

    if (eshkol::xla::deviceExecutionRequested() && arena && data &&
        total_elements > 0 && rank > 0 && op_code >= 0 && op_code <= 4 &&
        (axis == -1 || (axis >= 0 && axis < rank))) {
        eshkol::xla::DeviceOpRequest request;
        request.kind = static_cast<eshkol::xla::DeviceOpKind>(kDeviceKind[op_code]);
        request.operand_shapes = {xla_shape_of(shape, rank)};
        if (axis >= 0) request.axes = {axis};
        // Result shape: the input shape with the reduced axes removed. A full
        // reduction, and a rank-1 axis reduction, both leave nothing — which
        // the device expresses as rank 0 and the host tensor expresses as a
        // 1-element rank-1 tensor. xla_alloc_result bridges exactly that.
        std::vector<int64_t> out_shape;
        for (int64_t i = 0; i < rank; i++) {
            if (axis == -1 || i == axis) continue;
            out_shape.push_back(static_cast<int64_t>(shape[i]));
        }
        request.result_shape = out_shape;
        eshkol_tensor_t* result = xla_alloc_result(
            arena, out_shape, xla_num_elements(out_shape));
        if (result &&
            xla_device_try(request, {data},
                           reinterpret_cast<double*>(result->elements))) {
            return result;
        }
    }
    return eshkol_xla_reduce_host(arena, data, total_elements, shape, rank, axis, op_code);
}

extern "C" void* eshkol_xla_transpose(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* perm) {

    if (eshkol::xla::deviceExecutionRequested() && arena && data && perm &&
        rank > 0 && rank <= 16) {
        eshkol::xla::DeviceOpRequest request;
        request.kind = eshkol::xla::DeviceOpKind::Transpose;
        request.operand_shapes = {xla_shape_of(shape, rank)};
        bool ok = true;
        for (int64_t i = 0; i < rank; i++) {
            if (perm[i] < 0 || perm[i] >= rank) { ok = false; break; }
            request.axes.push_back(perm[i]);
            request.result_shape.push_back(static_cast<int64_t>(shape[perm[i]]));
        }
        if (ok) {
            eshkol_tensor_t* result = xla_alloc_result(
                arena, request.result_shape, xla_num_elements(request.result_shape));
            if (result &&
                xla_device_try(request, {data},
                               reinterpret_cast<double*>(result->elements))) {
                return result;
            }
        }
    }
    return eshkol_xla_transpose_host(arena, data, shape, rank, perm);
}

extern "C" void* eshkol_xla_broadcast(
    void* arena,
    const double* data,
    const uint64_t* src_shape,
    int64_t src_rank,
    const uint64_t* tgt_shape,
    int64_t tgt_rank) {

    if (eshkol::xla::deviceExecutionRequested() && arena && data &&
        src_rank > 0 && tgt_rank > 0 && src_rank <= tgt_rank && tgt_rank <= 16) {
        eshkol::xla::DeviceOpRequest request;
        request.kind = eshkol::xla::DeviceOpKind::Broadcast;
        request.operand_shapes = {xla_shape_of(src_shape, src_rank)};
        request.result_shape = xla_shape_of(tgt_shape, tgt_rank);
        // broadcast_in_dim wants, per operand dimension, the result dimension
        // it maps to. Eshkol's broadcast is right-aligned (NumPy), so operand
        // dimension i maps to result dimension (tgt_rank - src_rank + i).
        const int64_t offset = tgt_rank - src_rank;
        bool ok = true;
        for (int64_t i = 0; i < src_rank; i++) {
            const int64_t sd = static_cast<int64_t>(src_shape[i]);
            const int64_t td = static_cast<int64_t>(tgt_shape[offset + i]);
            if (sd != td && sd != 1) { ok = false; break; }
            request.axes.push_back(offset + i);
        }
        if (ok) {
            eshkol_tensor_t* result = xla_alloc_result(
                arena, request.result_shape, xla_num_elements(request.result_shape));
            if (result &&
                xla_device_try(request, {data},
                               reinterpret_cast<double*>(result->elements))) {
                return result;
            }
        }
    }
    return eshkol_xla_broadcast_host(arena, data, src_shape, src_rank, tgt_shape, tgt_rank);
}

extern "C" void* eshkol_xla_softmax(
    void* arena,
    const double* data,
    int64_t total_elements,
    const uint64_t* shape,
    int64_t rank,
    int64_t axis) {

    if (eshkol::xla::deviceExecutionRequested() && arena && data &&
        total_elements > 0 && rank > 0 && rank <= 16 &&
        (axis == -1 || (axis >= 0 && axis < rank))) {
        eshkol::xla::DeviceOpRequest request;
        request.kind = eshkol::xla::DeviceOpKind::Softmax;
        request.operand_shapes = {xla_shape_of(shape, rank)};
        // Softmax normalises along an axis; it does not remove one, so the
        // result has the operand's shape whichever axis was named.
        request.result_shape = request.operand_shapes[0];
        if (axis >= 0) request.axes = {axis};
        eshkol_tensor_t* result = xla_alloc_result(
            arena, request.result_shape, total_elements);
        if (result &&
            xla_device_try(request, {data},
                           reinterpret_cast<double*>(result->elements))) {
            return result;
        }
    }
    return eshkol_xla_softmax_host(arena, data, total_elements, shape, rank, axis);
}

// ===== XLA Device Gradient Runtime =====
//
// The reverse-mode counterpart of the entry points above: it computes the
// cotangents of one lowered tensor op ON THE DEVICE and hands one of them back
// as an ordinary Eshkol tensor, which is what a caller in this runtime can
// actually hold.
//
// WHY THIS ONE DOES NOT FALL BACK TO THE HOST.
//
// Every forward entry point above ends in a *_host call, because a host
// implementation of that op exists and computing it there is correct, merely
// slower. There is no such thing for most of these gradients: Eshkol's host
// reverse-mode AD has plain-buffer backward entry points for matmul,
// transpose, reshape, sum and mean (inc/eshkol/backend/tensor_backward.h) and
// nothing at all for elementwise, broadcast or max/min reductions. Falling
// back would therefore mean either refusing half the ops under a name that
// promises all of them, or quietly answering some of them from a different
// implementation than the one the caller asked for. So this returns NULL and
// says why, once, exactly as a missing device path should.
extern "C" void* eshkol_xla_gradient(
    void* arena,
    int64_t op_kind,
    int64_t num_operands,
    const double* const* operands,
    const uint64_t* const* operand_shapes,
    const int64_t* operand_ranks,
    const double* cotangent,
    const uint64_t* result_shape,
    int64_t result_rank,
    const int64_t* axes,
    int64_t num_axes,
    int64_t which_operand) {

    if (!arena || !operands || !operand_shapes || !operand_ranks) return nullptr;
    // Three, not two: clamp(lo, x, hi) is a lowered op with three operands and
    // a gradient for each of them. A bound of two would have refused it here
    // while runGradient() answered it perfectly well, i.e. the public entry
    // point would report a missing gradient the device path has.
    if (num_operands <= 0 || num_operands > 3) return nullptr;
    if (which_operand < 0 || which_operand >= num_operands) return nullptr;
    if (result_rank < 0 || result_rank > 16) return nullptr;
    if (op_kind < 0 || op_kind > static_cast<int64_t>(eshkol::xla::DeviceOpKind::ReduceProd)) {
        return nullptr;
    }

    if (!eshkol::xla::deviceExecutionRequested()) {
        xla_report_device_failure("gradient",
            "device execution was not requested (set ESHKOL_XLA_PJRT=1); there is no "
            "host reverse-mode AD entry point covering every lowered op, so no gradient "
            "is returned rather than one computed by a different implementation");
        return nullptr;
    }
    eshkol::xla::DeviceExecutor* executor = eshkol::xla::deviceExecutor();
    if (!executor) {
        xla_report_device_failure("gradient",
            "no device executor is installed in this binary; only a build that links the "
            "StableHLO emitter can install one (see device_lowering.h)");
        return nullptr;
    }

    eshkol::xla::DeviceOpRequest request;
    request.kind = static_cast<eshkol::xla::DeviceOpKind>(op_kind);
    for (int64_t i = 0; i < num_operands; i++) {
        if (!operands[i] || !operand_shapes[i] || operand_ranks[i] <= 0 ||
            operand_ranks[i] > 16) {
            return nullptr;
        }
        request.operand_shapes.push_back(xla_shape_of(operand_shapes[i], operand_ranks[i]));
    }
    request.result_shape = xla_shape_of(result_shape, result_rank);
    for (int64_t i = 0; i < num_axes; i++) request.axes.push_back(axes[i]);

    // Every operand's cotangent is produced by one device execution, because
    // the backward pass shares its forward graph; asking for one and
    // discarding the rest would compile and run the same module once per
    // operand. Only the requested one is wrapped as a tensor.
    std::vector<std::vector<double>> grad_storage;
    std::vector<double*> grad_ptrs;
    grad_storage.reserve(static_cast<size_t>(num_operands));
    for (int64_t i = 0; i < num_operands; i++) {
        grad_storage.emplace_back(
            static_cast<size_t>(xla_num_elements(request.operand_shapes[static_cast<size_t>(i)])),
            0.0);
        grad_ptrs.push_back(grad_storage.back().data());
    }

    std::vector<const double*> operand_ptrs;
    for (int64_t i = 0; i < num_operands; i++) operand_ptrs.push_back(operands[i]);

    std::string error;
    if (!executor->runGradient(request, operand_ptrs, cotangent, grad_ptrs, &error)) {
        xla_report_device_failure(eshkol::xla::deviceOpKindName(request.kind), error);
        return nullptr;
    }

    const std::vector<int64_t>& out_shape = request.operand_shapes[static_cast<size_t>(which_operand)];
    const int64_t out_total = xla_num_elements(out_shape);
    eshkol_tensor_t* result = xla_alloc_result(arena, out_shape, out_total);
    if (!result) return nullptr;
    double* dst = reinterpret_cast<double*>(result->elements);
    const std::vector<double>& src = grad_storage[static_cast<size_t>(which_operand)];
    for (int64_t i = 0; i < out_total; i++) dst[i] = src[static_cast<size_t>(i)];
    return result;
}

// ===== XLA Slice Runtime =====
// Slices a tensor with starts, limits, and strides per dimension.
extern "C" void* eshkol_xla_slice(
    void* arena,
    const double* data,
    const uint64_t* shape,
    int64_t rank,
    const int64_t* starts,
    const int64_t* limits,
    const int64_t* strides) {

    if (!data || rank <= 0) return nullptr;
    if (rank > 16) return nullptr;  // P1: out_shape[16] is a fixed stack array

    // Compute output shape
    uint64_t out_shape[16];
    uint64_t total = 1;
    for (int64_t i = 0; i < rank; i++) {
        int64_t stride = strides ? strides[i] : 1;
        // P2: reject degenerate/out-of-range slice params — stride<=0 (div-by-zero),
        // negative starts, limits<starts (unsigned underflow), or limits past the dim.
        if (stride <= 0 || starts[i] < 0 || limits[i] < starts[i] ||
            static_cast<uint64_t>(limits[i]) > shape[i]) return nullptr;
        out_shape[i] = static_cast<uint64_t>((limits[i] - starts[i] + stride - 1) / stride);
        total *= out_shape[i];
    }

    eshkol_tensor_t* result = arena_allocate_tensor_full(
        reinterpret_cast<arena_t*>(arena), static_cast<uint64_t>(rank), total);
    if (!result) return nullptr;
    result->dtype = ESHKOL_TENSOR_DTYPE_F64;
    for (int64_t i = 0; i < rank; i++) result->dimensions[i] = out_shape[i];

    double* out = reinterpret_cast<double*>(result->elements);

    // Compute source strides
    uint64_t src_strides[16];
    src_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        src_strides[i] = src_strides[i + 1] * shape[i + 1];
    }

    // Compute output strides
    uint64_t dst_strides[16];
    dst_strides[rank - 1] = 1;
    for (int64_t i = rank - 2; i >= 0; i--) {
        dst_strides[i] = dst_strides[i + 1] * out_shape[i + 1];
    }

    for (uint64_t flat = 0; flat < total; flat++) {
        uint64_t remaining = flat;
        uint64_t src_flat = 0;
        for (int64_t d = 0; d < rank; d++) {
            uint64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];
            int64_t stride = strides ? strides[d] : 1;
            uint64_t src_idx = static_cast<uint64_t>(starts[d]) + idx * static_cast<uint64_t>(stride);
            src_flat += src_idx * src_strides[d];
        }
        out[flat] = data[src_flat];
    }

    return result;
}

namespace eshkol {
namespace xla {

// ═══════════════════════════════════════════════════════════════════════════
// The device-lowering seam (see device_lowering.h)
// ═══════════════════════════════════════════════════════════════════════════
//
// These three live HERE, in the slim runtime archive, rather than beside the
// StableHLO executor that implements them. That placement is the whole point:
// the pointer below has to exist in every build, including AOT user binaries
// that link libeshkol-runtime.a alone and have no MLIR anywhere on their link
// line. It is null there, and the eshkol_xla_* entry points above take their
// host path exactly as they always did.

namespace {
/// The installed executor. Written once, early, by
/// registerStableHLODeviceExecutor(); read on every device-eligible op.
/// std::atomic rather than a plain pointer because the read happens from
/// whatever thread ran into a tensor op, including the parallel-map workers.
std::atomic<DeviceExecutor*> g_device_executor{nullptr};
}  // namespace

/**
 * @brief Human-readable name for a device op kind.
 *
 * DEFINED HERE, in the slim runtime archive, and not next to the executor
 * that uses it most. The reason is concrete: the eshkol_xla_* entry points
 * above name the op in their fallback diagnostic, so this symbol is
 * referenced from libeshkol-runtime.a. It was first written in
 * device_lowering.cpp, which lives in libeshkol-static.a, and every AOT link
 * of an .esk program then failed at link time:
 *
 *   Undefined symbols for architecture arm64:
 *     "eshkol::xla::deviceOpKindName(eshkol::xla::DeviceOpKind)",
 *       referenced from: xla_device_try(...) in libeshkol-runtime.a
 *
 * which is precisely the coupling device_lowering.h exists to prevent — an
 * AOT binary must never need anything from the MLIR-linked half. A pure
 * switch over an enum has no reason to be over there.
 */
const char* deviceOpKindName(DeviceOpKind kind) {
    switch (kind) {
        case DeviceOpKind::Add:        return "add";
        case DeviceOpKind::Subtract:   return "subtract";
        case DeviceOpKind::Multiply:   return "multiply";
        case DeviceOpKind::Divide:     return "divide";
        case DeviceOpKind::Exp:        return "exp";
        case DeviceOpKind::Log:        return "log";
        case DeviceOpKind::Sin:        return "sin";
        case DeviceOpKind::Cos:        return "cos";
        case DeviceOpKind::Tanh:       return "tanh";
        case DeviceOpKind::Sqrt:       return "sqrt";
        case DeviceOpKind::Rsqrt:      return "rsqrt";
        case DeviceOpKind::Abs:        return "abs";
        case DeviceOpKind::Negate:     return "negate";
        case DeviceOpKind::Relu:       return "relu";
        case DeviceOpKind::Sigmoid:    return "sigmoid";
        case DeviceOpKind::Atanh:      return "atanh";
        case DeviceOpKind::Softmax:    return "softmax";
        case DeviceOpKind::Pow:        return "pow";
        case DeviceOpKind::Maximum:    return "maximum";
        case DeviceOpKind::Minimum:    return "minimum";
        case DeviceOpKind::Clamp:      return "clamp";
        case DeviceOpKind::Matmul:     return "matmul";
        case DeviceOpKind::Transpose:  return "transpose";
        case DeviceOpKind::Reshape:    return "reshape";
        case DeviceOpKind::Broadcast:  return "broadcast";
        case DeviceOpKind::ReduceSum:  return "reduce_sum";
        case DeviceOpKind::ReduceMean: return "reduce_mean";
        case DeviceOpKind::ReduceMax:  return "reduce_max";
        case DeviceOpKind::ReduceMin:  return "reduce_min";
        case DeviceOpKind::ReduceProd: return "reduce_prod";
        case DeviceOpKind::CompareEq:  return "compare_eq";
        case DeviceOpKind::CompareNe:  return "compare_ne";
        case DeviceOpKind::CompareLt:  return "compare_lt";
        case DeviceOpKind::CompareLe:  return "compare_le";
        case DeviceOpKind::CompareGt:  return "compare_gt";
        case DeviceOpKind::CompareGe:  return "compare_ge";
    }
    return "unknown";
}

bool deviceOpYieldsPredicate(DeviceOpKind kind) {
    switch (kind) {
        case DeviceOpKind::CompareEq:
        case DeviceOpKind::CompareNe:
        case DeviceOpKind::CompareLt:
        case DeviceOpKind::CompareLe:
        case DeviceOpKind::CompareGt:
        case DeviceOpKind::CompareGe:
            return true;
        default:
            return false;
    }
}

DeviceExecutor* deviceExecutor() {
    return g_device_executor.load(std::memory_order_acquire);
}

void setDeviceExecutor(DeviceExecutor* executor) {
    g_device_executor.store(executor, std::memory_order_release);
}

bool deviceExecutionRequested() {
    // Read once per process. Re-reading getenv on every tensor op would put a
    // libc call on the hot path of every elementwise operation in the language,
    // and the answer cannot change usefully mid-run: the PJRT plugin selection
    // in initialize() is already latched at first use.
    static const bool requested = [] {
        const char* off = std::getenv("ESHKOL_XLA_DEVICE");
        if (off && std::strcmp(off, "0") == 0) return false;
        const char* want = std::getenv("ESHKOL_XLA_PJRT");
        return want != nullptr && std::strcmp(want, "1") == 0;
    }();
    return requested;
}

// ===== XLARuntime Implementation =====

class XLARuntime::Impl {
public:
    bool initialized_ = false;
    Target target_ = Target::CPU;
    size_t allocated_bytes_ = 0;
    size_t peak_bytes_ = 0;
    std::unordered_map<void*, std::future<ExecutionResult>> async_handles_;
    size_t next_handle_id_ = 1;

    // PJRT device execution state. `pjrt_active_` and `pjrt_status_` are
    // plain types so every read site below can check them unconditionally —
    // when ESHKOL_XLA_PJRT_AVAILABLE isn't defined, initialize() never sets
    // pjrt_active_ true and pjrt_status_ never leaves the empty string, so
    // execute()/toHost()/getDescription() fall straight through to the
    // existing behaviour with no ifdef needed at those call sites.
    bool pjrt_active_ = false;
    std::string pjrt_status_;   // empty unless PJRT was requested (diagnostic either way)
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    // Declared client-before-plugin so ~Impl() (reverse declaration order)
    // destroys the client first — pjrt_client.h requires the plugin to
    // outlive every client built over it, since every PJRT handle belongs to
    // the plugin's shared object.
    std::unique_ptr<PjrtClient> pjrt_client_;
    std::unique_ptr<PjrtPlugin> pjrt_plugin_;
    int pjrt_device_index_ = -1;   // index into pjrt_client_->devices() chosen at init
#endif
};

/** @brief Construct an uninitialized XLA runtime; call initialize() before use. */
XLARuntime::XLARuntime()
    : impl_(std::make_unique<Impl>()) {}

XLARuntime::~XLARuntime() = default;

// ===== Initialization =====

/** @brief Initialize the runtime for `target`. CPU always succeeds. For a
 *         GPU target, initializes the GPU subsystem and verifies the
 *         detected backend matches the requested one; if no GPU devices are
 *         found, still marks initialized=true since every XLA runtime
 *         function has a CPU fallback. Returns the resulting initialized state.
 *
 *         Independently of `target`, if ESHKOL_XLA_PJRT=1 is set in the
 *         environment (and this build has the PJRT client compiled in), also
 *         attempts to load a PJRT plugin and stand up a device client; on
 *         success, execute()/toHost() route through it instead of the
 *         LLVM-direct path (see execute() below). This never changes the
 *         bool this function returns: exactly as a GPU target with no devices
 *         still initializes to the CPU fallback, a PJRT request that fails
 *         still initializes to the LLVM-direct fallback — the failure is
 *         reported through getDescription() and stderr, not through this
 *         return value, so a caller that never asked for PJRT sees no
 *         behavior change at all. */
bool XLARuntime::initialize(Target target) {
    impl_->target_ = target;

    if (target == Target::CPU) {
        impl_->initialized_ = true;
    } else {
        // Initialize GPU backend if not already done
        int gpu_devices = eshkol_gpu_init();
        if (gpu_devices > 0) {
            // Verify the requested GPU backend is available
            EshkolGPUBackend backend = eshkol_gpu_get_backend();
            bool match = false;
            switch (target) {
                case Target::Metal:  match = (backend == ESHKOL_GPU_METAL);  break;
                case Target::CUDA:   match = (backend == ESHKOL_GPU_CUDA);   break;
                case Target::Vulkan: match = (backend == ESHKOL_GPU_VULKAN); break;
                default: break;
            }
            impl_->initialized_ = match;
        } else {
            // No GPU devices — fall back, but still mark as initialized
            // since XLA runtime functions have CPU fallbacks
            impl_->initialized_ = true;
        }
    }

    // ===== Optional PJRT device execution =====
    //
    // PJRT selection is deliberately independent of `target`: `target` names
    // what StableHLO was compiled for (CPU/CUDA/Metal/Vulkan, all via the
    // LLVM-direct JIT path above), while PJRT is a separate, later-binding
    // choice of which .so actually executes the compiled program — pjrt_client.h
    // puts it plainly: "Choosing a device is choosing which .so to dlopen."
    // A PJRT CPU plugin and a PJRT TPU plugin both exist; running the CPU
    // target's StableHLO through the PJRT CPU plugin (for testing, with no
    // TPU present) is a legitimate combination this design has to allow.
    //
    // Extending `Target` with a PJRT/TPU variant was considered and rejected:
    // every switch over Target in this codebase — the two in this file above
    // and below, plus xla_codegen.cpp, xla_compiler.cpp, and xla_types.cpp —
    // is a closed, default-less exhaustive dispatch by design (CMakeLists.txt's
    // EXHAUSTIVE CLOSED-ENUM DISPATCH section: no `default:` so an unhandled
    // enumerator is a build error, not a silent no-op). Adding a Target
    // enumerator here would require editing every one of those switches in
    // files this change does not own, to express a concern (which runtime
    // executes the program) that has nothing to do with what those switches
    // dispatch on (how the program was compiled). An environment variable
    // keeps PJRT activation entirely inside this file.
    impl_->pjrt_active_ = false;
    impl_->pjrt_status_.clear();
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    impl_->pjrt_client_.reset();   // destroyed before pjrt_plugin_ — see the Impl comment
    impl_->pjrt_plugin_.reset();
    impl_->pjrt_device_index_ = -1;

    const char* want_pjrt = std::getenv("ESHKOL_XLA_PJRT");
    if (want_pjrt && std::strcmp(want_pjrt, "1") == 0) {
        // findPjrtPlugin's own doc comment says its auto-discovery (the
        // vendor-wheel and system-path checks) only special-cases a "tpu" or
        // empty backend string; every other backend name only ever reaches
        // its unconditional ESHKOL_PJRT_PLUGIN_PATH check. There is no `Target`
        // value that means "tpu" to hand it (see the comment above), and
        // guessing a name for CPU/CUDA/Metal/Vulkan would buy nothing over the
        // wildcard, so "" is passed deliberately: it is exactly the case that
        // function's own logic already treats as "any backend".
        std::string plugin_path = findPjrtPlugin("");
        if (plugin_path.empty()) {
            impl_->pjrt_status_ =
                "ESHKOL_XLA_PJRT=1 was set but no PJRT plugin was found "
                "(checked ESHKOL_PJRT_PLUGIN_PATH and the usual TPU wheel/system "
                "locations); set ESHKOL_PJRT_PLUGIN_PATH to an explicit .so path. "
                "Falling back to LLVM-direct dispatch.";
            std::fprintf(stderr, "eshkol: XLA PJRT: %s\n", impl_->pjrt_status_.c_str());
        } else {
            std::string load_error;
            impl_->pjrt_plugin_ = PjrtPlugin::load(plugin_path, &load_error);
            if (!impl_->pjrt_plugin_) {
                impl_->pjrt_status_ =
                    "ESHKOL_XLA_PJRT=1 was set; failed to load PJRT plugin '" +
                    plugin_path + "': " + load_error +
                    ". Falling back to LLVM-direct dispatch.";
                std::fprintf(stderr, "eshkol: XLA PJRT: %s\n", impl_->pjrt_status_.c_str());
            } else {
                std::string create_error;
                impl_->pjrt_client_ =
                    PjrtClient::create(impl_->pjrt_plugin_.get(), &create_error);
                if (!impl_->pjrt_client_) {
                    impl_->pjrt_status_ =
                        "ESHKOL_XLA_PJRT=1 was set; plugin '" + plugin_path +
                        "' loaded but PjrtClient::create failed: " + create_error +
                        ". Falling back to LLVM-direct dispatch.";
                    std::fprintf(stderr, "eshkol: XLA PJRT: %s\n", impl_->pjrt_status_.c_str());
                    impl_->pjrt_plugin_.reset();
                } else {
                    // Pick the first addressable device up front, rather than
                    // deferring to the first execute() call, so a plugin that
                    // loads but exposes nothing this process can use is
                    // reported here — at init, where a human is looking —
                    // instead of surfacing later as an opaque execute() failure.
                    const auto& devices = impl_->pjrt_client_->devices();
                    for (size_t i = 0; i < devices.size(); ++i) {
                        if (devices[i].is_addressable) {
                            impl_->pjrt_device_index_ = static_cast<int>(i);
                            break;
                        }
                    }
                    if (impl_->pjrt_device_index_ < 0) {
                        impl_->pjrt_status_ =
                            "ESHKOL_XLA_PJRT=1 was set; plugin '" + plugin_path +
                            "' loaded (platform=" + impl_->pjrt_client_->platformName() +
                            ") but reports no addressable device. Falling back to "
                            "LLVM-direct dispatch.";
                        std::fprintf(stderr, "eshkol: XLA PJRT: %s\n", impl_->pjrt_status_.c_str());
                        impl_->pjrt_client_.reset();
                        impl_->pjrt_plugin_.reset();
                    } else {
                        impl_->pjrt_active_ = true;
                        impl_->pjrt_status_ =
                            "PJRT active: platform=" + impl_->pjrt_client_->platformName() +
                            ", plugin=" + plugin_path;
                        std::fprintf(stderr, "eshkol: XLA PJRT: %s\n", impl_->pjrt_status_.c_str());
                    }
                }
            }
        }
    }
#endif

    return impl_->initialized_;
}

/** @brief True if initialize() has completed successfully. */
bool XLARuntime::isInitialized() const {
    return impl_->initialized_;
}

/** @brief The compilation target this runtime was initialized for. */
Target XLARuntime::getTarget() const {
    return impl_->target_;
}

// ===== Execution =====
// Infrastructure for StableHLO JIT — execute/executeAsync provide the execution
// interface for compiled XLA programs. Currently unused; actual XLA ops dispatch
// through 11 C runtime functions called directly from codegen.

/** @brief Synchronously run a compiled `executable` against `inputs`/`outputs`
 *         buffer descriptors, timing the call. Returns a failure result if the
 *         runtime isn't initialized or `executable` is null.
 *
 *         What `executable` IS depends on which path initialize() selected,
 *         exactly the way it always has for the LLVM-direct/CPU-vs-GPU split
 *         below: in LLVM-direct mode it is a raw function pointer of type
 *         `void(const void* const*, void* const*)`; when PJRT is active (see
 *         initialize()) it is instead the `PJRT_LoadedExecutable*` returned by
 *         PjrtClient::compile(), reinterpret_cast through the same void*.
 *         There is no third representation and no runtime tag distinguishing
 *         them — the caller must compile through whichever path this runtime
 *         was initialized for. */
ExecutionResult XLARuntime::execute(void* executable,
                                     const std::vector<BufferDescriptor>& inputs,
                                     std::vector<BufferDescriptor>& outputs) {
    if (!impl_->initialized_ || !executable) {
        return ExecutionResult{
            .success = false,
            .error_message = "Runtime not initialized or null executable",
            .execution_time_ns = 0
        };
    }

    auto start = std::chrono::high_resolution_clock::now();

#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    if (impl_->pjrt_active_) {
        // Element type comes from the descriptor, not from an assumption.
        //
        // This used to pass PjrtElementType::kF64 for every buffer, on the
        // reasoning that every Eshkol tensor is f64. The tensors are — but the
        // buffers handed to a DEVICE are not: TPU has no f64 arithmetic, so
        // the device lowering path stages f32 (see device_lowering.cpp). A
        // PJRT plugin does not validate the claim; it takes the type at its
        // word and reads `element_count * sizeof(f64)` bytes out of a buffer
        // holding half that, which is a silent misread and an overrun rather
        // than a rejected transfer. PjrtElementType is checked against the
        // real PJRT_Buffer_Type by static_assert inside pjrt_client.cpp, so
        // these names cannot drift away from the ABI values without failing
        // the build.
        auto pjrt_type = [](BufferElementType e) {
            switch (e) {
                case BufferElementType::F32:  return PjrtElementType::kF32;
                case BufferElementType::S32:  return PjrtElementType::kS32;
                case BufferElementType::S64:  return PjrtElementType::kS64;
                case BufferElementType::PRED: return PjrtElementType::kPred;
                case BufferElementType::F64:  break;
            }
            return PjrtElementType::kF64;
        };
        auto expected_size = [](BufferElementType e) -> size_t {
            switch (e) {
                case BufferElementType::F32:  return sizeof(float);
                case BufferElementType::S32:  return sizeof(int32_t);
                case BufferElementType::S64:  return sizeof(int64_t);
                // PRED is a byte per element on the wire, not a bit: the ABI
                // has no sub-byte addressing and a packed bitmask would be a
                // different layout than any host buffer this runtime holds.
                case BufferElementType::PRED: return sizeof(uint8_t);
                case BufferElementType::F64:  break;
            }
            return sizeof(double);
        };
        auto elem_name = [](BufferElementType e) -> const char* {
            switch (e) {
                case BufferElementType::F32:  return "f32";
                case BufferElementType::S32:  return "s32";
                case BufferElementType::S64:  return "s64";
                case BufferElementType::PRED: return "pred";
                case BufferElementType::F64:  break;
            }
            return "f64";
        };

        auto* pjrt_executable = reinterpret_cast<PJRT_LoadedExecutable*>(executable);
        auto fail = [&](const std::string& message,
                        const std::vector<PJRT_Buffer*>& to_release1,
                        const std::vector<PJRT_Buffer*>& to_release2) -> ExecutionResult {
            for (auto* b : to_release1) impl_->pjrt_client_->destroyBuffer(b);
            for (auto* b : to_release2) impl_->pjrt_client_->destroyBuffer(b);
            auto end = std::chrono::high_resolution_clock::now();
            auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
            return ExecutionResult{.success = false, .error_message = message, .execution_time_ns = ns};
        };

        // Stage every host input onto the device chosen at initialize() time.
        std::vector<PJRT_Buffer*> pjrt_inputs;
        pjrt_inputs.reserve(inputs.size());
        for (const auto& in : inputs) {
            std::string stage_error;
            if (in.element_size != expected_size(in.elem)) {
                return fail("PJRT execute: input buffer declares element_size " +
                                std::to_string(in.element_size) + " but element type " +
                                elem_name(in.elem) +
                                " is " + std::to_string(expected_size(in.elem)) + " bytes",
                            pjrt_inputs, {});
            }
            PJRT_Buffer* buf = impl_->pjrt_client_->bufferFromHost(
                in.data, pjrt_type(in.elem), in.shape, impl_->pjrt_device_index_,
                &stage_error);
            if (!buf) {
                return fail("PJRT execute: bufferFromHost failed: " + stage_error,
                            pjrt_inputs, {});
            }
            pjrt_inputs.push_back(buf);
        }

        std::vector<PJRT_Buffer*> pjrt_outputs;
        PjrtStatus status = impl_->pjrt_client_->execute(pjrt_executable, pjrt_inputs, pjrt_outputs);
        if (!status.ok()) {
            return fail("PJRT execute failed: " + status.message(), pjrt_inputs, pjrt_outputs);
        }
        if (pjrt_outputs.size() != outputs.size()) {
            return fail("PJRT execute returned " + std::to_string(pjrt_outputs.size()) +
                             " output buffer(s) but " + std::to_string(outputs.size()) +
                             " were expected",
                         pjrt_inputs, pjrt_outputs);
        }

        // Copy every device result back into the caller's pre-allocated host
        // buffers before releasing anything, so a mid-loop failure still
        // leaves every buffer reachable for cleanup by `fail`.
        std::string copy_error;
        for (size_t i = 0; i < outputs.size(); ++i) {
            size_t total = 1;
            for (auto dim : outputs[i].shape) total *= static_cast<size_t>(dim);
            PjrtStatus copy_status = impl_->pjrt_client_->bufferToHost(
                pjrt_outputs[i], outputs[i].data, total * outputs[i].element_size);
            if (!copy_status.ok() && copy_error.empty()) {
                copy_error = "PJRT execute: bufferToHost failed for output " +
                             std::to_string(i) + ": " + copy_status.message();
            }
        }
        if (!copy_error.empty()) {
            return fail(copy_error, pjrt_inputs, pjrt_outputs);
        }

        for (auto* b : pjrt_inputs) impl_->pjrt_client_->destroyBuffer(b);
        for (auto* b : pjrt_outputs) impl_->pjrt_client_->destroyBuffer(b);

        auto end = std::chrono::high_resolution_clock::now();
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        return ExecutionResult{.success = true, .error_message = "", .execution_time_ns = ns};
    }
#endif

    // Cast executable to function pointer and call directly
    // In LLVM direct mode, the executable is a function pointer to the compiled IR
    using ExecFn = void(*)(const void* const*, void* const*);
    auto fn = reinterpret_cast<ExecFn>(executable);

    // Build input/output pointer arrays
    std::vector<const void*> input_ptrs;
    std::vector<void*> output_ptrs;
    input_ptrs.reserve(inputs.size());
    output_ptrs.reserve(outputs.size());
    for (const auto& in : inputs) input_ptrs.push_back(in.data);
    for (auto& out : outputs) output_ptrs.push_back(out.data);

    fn(input_ptrs.data(), output_ptrs.data());

    auto end = std::chrono::high_resolution_clock::now();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    return ExecutionResult{
        .success = true,
        .error_message = "",
        .execution_time_ns = ns
    };
}

/** @brief Launch execute() on a background thread (std::async), storing its
 *         future keyed by a newly minted opaque handle. Retrieve the result
 *         later via wait(). Returns nullptr if the runtime isn't initialized
 *         or `executable` is null. */
void* XLARuntime::executeAsync(void* executable,
                                const std::vector<BufferDescriptor>& inputs,
                                std::vector<BufferDescriptor>& outputs) {
    if (!impl_->initialized_ || !executable) return nullptr;

    // Capture copies for the async task
    auto inputs_copy = inputs;
    auto outputs_copy = outputs;

    auto future = std::async(std::launch::async, [this, executable, inputs_copy, outputs_copy]() mutable {
        return this->execute(executable, inputs_copy, outputs_copy);
    });

    // Store the future and return a handle
    void* handle = reinterpret_cast<void*>(impl_->next_handle_id_++);
    impl_->async_handles_[handle] = std::move(future);
    return handle;
}

/** @brief Block until an executeAsync() call identified by `handle`
 *         completes, and return (and forget) its result. Returns a failure
 *         result if `handle` is unknown/already consumed. */
ExecutionResult XLARuntime::wait(void* handle) {
    auto it = impl_->async_handles_.find(handle);
    if (it == impl_->async_handles_.end()) {
        return ExecutionResult{
            .success = false,
            .error_message = "Invalid async execution handle",
            .execution_time_ns = 0
        };
    }

    auto result = it->second.get();
    impl_->async_handles_.erase(it);
    return result;
}

/** @brief Compile a StableHLO module for the active PJRT device. See the
 *         header for why this lives on the runtime rather than beside the
 *         emitter that produces the module text. */
void* XLARuntime::compileStableHLO(const std::string& module_text, std::string* error) {
    std::string local;
    std::string* err = error ? error : &local;
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    if (!impl_->pjrt_active_) {
        *err = impl_->pjrt_status_.empty()
            ? "PJRT device execution is not active; set ESHKOL_XLA_PJRT=1 and make a "
              "PJRT plugin discoverable (ESHKOL_PJRT_PLUGIN_PATH)"
            : impl_->pjrt_status_;
        return nullptr;
    }
    if (module_text.empty()) {
        *err = "empty StableHLO module text";
        return nullptr;
    }
    // "mlir" is the PJRT program format name for StableHLO in its textual
    // form, the same one tests/xla/pjrt_roundtrip_test.cpp compiles with.
    return impl_->pjrt_client_->compile(module_text, "mlir", err);
#else
    (void)module_text;
    *err = "this build has no PJRT client compiled in";
    return nullptr;
#endif
}

/** @brief Release an executable from compileStableHLO(). */
void XLARuntime::releaseExecutable(void* executable) {
    if (!executable) return;
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    if (impl_->pjrt_active_ && impl_->pjrt_client_) {
        impl_->pjrt_client_->destroyExecutable(
            reinterpret_cast<PJRT_LoadedExecutable*>(executable));
    }
#endif
}

/** @brief Whether execute() will run on a PJRT device. */
bool XLARuntime::isDeviceExecutionActive() const {
    return impl_->pjrt_active_;
}

/** @brief Why the device is or is not active; empty if never requested. */
std::string XLARuntime::deviceStatus() const {
    return impl_->pjrt_status_;
}

// ===== Buffer Management =====

/** @brief Allocate a zero-initialized device buffer of the given shape/element
 *         size. On CPU this is a plain heap allocation (calloc), tracked
 *         against this runtime's allocation statistics. */
BufferDescriptor XLARuntime::allocateDevice(const std::vector<int64_t>& shape,
                                             size_t element_size) {
    // CPU: allocate on host. Compute total size from shape.
    size_t total = 1;
    for (auto dim : shape) total *= static_cast<size_t>(dim);
    size_t size_bytes = total * element_size;

    void* data = std::calloc(total, element_size);
    if (data) {
        impl_->allocated_bytes_ += size_bytes;
        if (impl_->allocated_bytes_ > impl_->peak_bytes_) {
            impl_->peak_bytes_ = impl_->allocated_bytes_;
        }
    }

    return BufferDescriptor{
        .data = data,
        .shape = shape,
        .element_size = element_size,
        .on_device = false
    };
}

/** @brief Wrap host memory as a device buffer descriptor. CPU path is
 *         zero-copy: the descriptor just points at `host_data` directly. */
BufferDescriptor XLARuntime::toDevice(void* host_data,
                                       const std::vector<int64_t>& shape,
                                       size_t element_size) {
    // CPU: data is already on host — zero-copy wrap
    return BufferDescriptor{
        .data = host_data,
        .shape = shape,
        .element_size = element_size,
        .on_device = false
    };
}

/** @brief Copy a device buffer's contents to host memory. CPU path is a
 *         no-op if the pointers already alias, otherwise a memcpy. Under
 *         PJRT, `device_buffer.data` for an actual on-device buffer is a
 *         `PJRT_Buffer*` handle rather than host memory, and the copy goes
 *         through PjrtClient::bufferToHost() instead of memcpy. */
void XLARuntime::toHost(const BufferDescriptor& device_buffer, void* host_data) {
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
    // `on_device` is what distinguishes the two: toDevice()/allocateDevice()
    // are unchanged by this file and still hand back host-resident,
    // on_device=false descriptors even when PJRT is active (pjrt_client.h has
    // no device-side "allocate empty buffer" call for them to use — see
    // s1-runtime-wiring-report.md). The only descriptors that are genuinely
    // on_device under PJRT are the ones execute() builds internally, and
    // execute() already copies those back out and destroys them itself before
    // this function would ever see them — this branch exists for symmetry and
    // for any future caller that holds one across a call boundary.
    if (impl_->pjrt_active_ && device_buffer.on_device) {
        auto* buffer = reinterpret_cast<PJRT_Buffer*>(device_buffer.data);
        size_t total = 1;
        for (auto dim : device_buffer.shape) total *= static_cast<size_t>(dim);
        PjrtStatus status = impl_->pjrt_client_->bufferToHost(
            buffer, host_data, total * device_buffer.element_size);
        if (!status.ok()) {
            // toHost() returns void — there is no channel back to the caller.
            // stderr is the only diagnostic available here; a silently
            // unfilled host_data would be far worse than a noisy failure.
            std::fprintf(stderr, "eshkol: XLA PJRT toHost failed: %s\n",
                         status.message().c_str());
        }
        return;
    }
#endif
    // CPU: data is already on host. If pointers differ, memcpy.
    if (device_buffer.data && host_data && device_buffer.data != host_data) {
        size_t total = 1;
        for (auto dim : device_buffer.shape) total *= static_cast<size_t>(dim);
        std::memcpy(host_data, device_buffer.data, total * device_buffer.element_size);
    }
}

/** @brief Release a buffer allocated by allocateDevice() and update
 *         allocation statistics. Does not attempt to free arena-managed
 *         buffers (those are owned elsewhere). */
void XLARuntime::freeBuffer(BufferDescriptor& buffer) {
    // Only free buffers we allocated (on_device=false for CPU allocs from allocateDevice)
    // Arena-managed buffers should NOT be freed here
    if (buffer.data) {
        size_t total = 1;
        for (auto dim : buffer.shape) total *= static_cast<size_t>(dim);
        size_t size_bytes = total * buffer.element_size;
        if (impl_->allocated_bytes_ >= size_bytes) {
            impl_->allocated_bytes_ -= size_bytes;
        }
    }
    buffer.data = nullptr;
}

// ===== Synchronization =====

/** @brief Wait for all pending executeAsync() operations to complete and
 *         clear their handles. On CPU, ops are already synchronous, so this
 *         only needs to drain outstanding async futures. */
void XLARuntime::synchronize() {
    // CPU: all operations are synchronous. Wait for any pending async executions.
    for (auto& [handle, future] : impl_->async_handles_) {
        if (future.valid()) future.wait();
    }
    impl_->async_handles_.clear();
}

// ===== Diagnostics =====

/** @brief Retrieve current and peak device-buffer allocation byte counts. */
void XLARuntime::getMemoryStats(size_t& allocated_bytes, size_t& peak_bytes) {
    allocated_bytes = impl_->allocated_bytes_;
    peak_bytes = impl_->peak_bytes_;
}

/** @brief Human-readable description of this runtime's target/state, for
 *         diagnostics. This is the string a human reads to find out whether
 *         they are actually executing on a device or still on the LLVM-direct
 *         CPU path — under PJRT it names the live platform and how many of
 *         its devices are addressable, rather than just echoing `target`
 *         (which, under PJRT, only says what StableHLO was compiled for, not
 *         what is executing it — see initialize()). */
std::string XLARuntime::getDescription() const {
    std::string desc = "XLA Runtime (";
    if (impl_->initialized_) {
#ifdef ESHKOL_XLA_PJRT_AVAILABLE
        if (impl_->pjrt_active_) {
            size_t total = impl_->pjrt_client_->devices().size();
            size_t addressable = impl_->pjrt_client_->addressableDevices().size();
            desc += "PJRT device execution, platform=" + impl_->pjrt_client_->platformName() +
                    ", devices=" + std::to_string(addressable) + "/" +
                    std::to_string(total) + " addressable";
            desc += ")";
            return desc;
        }
#endif
        switch (impl_->target_) {
            case Target::CPU: desc += "CPU, LLVM direct dispatch"; break;
            case Target::CUDA: desc += "CUDA"; break;
            case Target::Metal: desc += "Metal"; break;
            case Target::Vulkan: desc += "Vulkan"; break;
        }
        // Non-empty only when ESHKOL_XLA_PJRT=1 was requested and fell back
        // (see initialize()) — silent by default, so a caller that never asked
        // for PJRT sees exactly the pre-PJRT string, byte for byte.
        if (!impl_->pjrt_status_.empty()) {
            desc += "; " + impl_->pjrt_status_;
        }
        desc += ")";
    } else {
        desc += "not initialized)";
    }
    return desc;
}

// ===== Global Runtime =====

/** @brief Return the process-wide default XLARuntime singleton, lazily
 *         initializing it (once, thread-safe) to the best available GPU
 *         backend or CPU if none is detected. */
XLARuntime& getDefaultRuntime() {
    static XLARuntime runtime;
    static std::once_flag init_flag;
    std::call_once(init_flag, [&]() {
        // Detect best available GPU target
        eshkol_gpu_init();
        EshkolGPUBackend backend = eshkol_gpu_get_backend();
        Target target;
        switch (backend) {
            case ESHKOL_GPU_METAL:  target = Target::Metal;  break;
            case ESHKOL_GPU_CUDA:   target = Target::CUDA;   break;
            case ESHKOL_GPU_VULKAN: target = Target::Vulkan;  break;
            default:                target = Target::CPU;      break;
        }
        runtime.initialize(target);
    });
    return runtime;
}

} // namespace xla
} // namespace eshkol
