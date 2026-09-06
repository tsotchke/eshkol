/*
 * Device/host differential parity for Eshkol's lowered tensor operations.
 *
 * WHAT THIS PROVES.
 *
 * For each operation that lib/backend/xla/device_lowering.cpp lowers to
 * StableHLO, this computes the SAME inputs two ways in the SAME process:
 *
 *   host   — eshkol_xla_<op>_host(), the BLAS/SIMD/GPU implementation that has
 *            always answered these calls, in f64;
 *   device — the StableHLO module built by StableHLOEmitter, compiled through
 *            PjrtClient and executed on whatever PJRT device this host
 *            provides, in the device element type (f32 unless
 *            ESHKOL_XLA_DEVICE_DTYPE says otherwise);
 *
 * and reports max absolute and max relative error against the tolerance
 * docs/design/ESHKOL_S_FRAGMENT.md states for that dtype. It then calls the
 * PUBLIC entry point — eshkol_xla_<op>(), the one generated code calls — with
 * device execution enabled, and requires that to agree with the device answer.
 * That third comparison is what makes this a test of the wiring and not only
 * of the executor: a lowering that is correct but never reached by the runtime
 * would pass the first two comparisons and fail this one.
 *
 * WHY THE TOLERANCE IS NOT ZERO, AND WHY IT IS NOT ONE NUMBER.
 *
 * The host reference is f64 and the device computes in f32 on any TPU (which
 * has no f64 arithmetic at all). Comparing across those two precisions is the
 * point of the exercise, not a compromise in it: the alternative — computing
 * the reference in f32 as well — would stop testing whether the device
 * computed the right thing and start testing whether two f32 pipelines round
 * identically.
 *
 * But a single bound across every op is not the truth either. An add, a
 * matmul, a reduction and a transpose are exact operations whose only error is
 * the f32 rounding of their inputs and outputs; exp, log and tanh are not
 * operations at all on a TPU, they are approximations, evaluated by a
 * reduced-precision elementwise unit. Measured on TPU hardware with f32
 * device arithmetic against the f64 host, the first group agreed to at worst
 * 7.9e-8 relative (divide) and usually exactly, while the second reached
 * 2.2e-4 relative (log). Holding both to 1e-5 does not make the transcendental
 * ops more accurate; it makes the gate report a defect where there is only a
 * documented property of the hardware, which is how a gate stops being read.
 *
 * So each row names a tolerance CLASS, and docs/design/ESHKOL_S_FRAGMENT.md
 * states the bound for each: the arithmetic class keeps the per-dtype bound
 * (1e-5 f32, 1e-9 f64) and the transcendental class is 100x it (1e-3 f32,
 * 1e-7 f64). The rule in both cases is |device - host| <= tol absolutely OR
 * relatively, whichever is looser. See that document for the full rationale
 * and the measured numbers behind the factor.
 *
 * A GATE THAT CANNOT FAIL IS WORTHLESS.
 *
 * Two things here can fail, by construction:
 *   - A negative control (test_comparator_rejects_a_perturbed_result) runs
 *     BEFORE any device is required and feeds the comparator a deliberately
 *     wrong result. The comparator must reject it. Without this, a comparator
 *     that had regressed into `return true` would let every row below pass.
 *   - ESHKOL_XLA_PARITY_FORCE_FAIL=<op name> perturbs the host reference for
 *     that op by 1.0 before the comparison, so a real end-to-end FAIL can be
 *     demonstrated on demand against live hardware without editing this file.
 *
 * Exit status: 0 all rows agreed; 1 a row disagreed or a control failed;
 * 77 no PJRT device was reachable (the caller decides what that means — the
 * XLA gate treats it as FAIL, since a parity claim needs a device).
 *
 * Only builds when ESHKOL_XLA_ENABLED=ON (see CMakeLists.txt).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/xla_runtime.h"

#include "parity_compare.h"

#include "../../lib/core/arena_memory.h"

using eshkol::xla::DeviceExecutor;
using eshkol::xla::DeviceOpKind;
using eshkol::xla::DeviceOpRequest;
using eshkol::xla::deviceOpKindName;
using eshkol::xla::deviceOpYieldsPredicate;
using eshkol::xla::deviceExecutor;
using eshkol::xla::registerStableHLODeviceExecutor;

// The host implementations, under the names they carry since the device split
// in lib/backend/xla/xla_runtime.cpp. These are the reference.
extern "C" {
void* eshkol_xla_matmul_host(void* arena, const double* a, const double* b,
                             const int64_t* a_shape, const int64_t* b_shape,
                             int64_t a_rank, int64_t b_rank);
void* eshkol_xla_elementwise_host(void* arena, const double* a, const double* b,
                                  int64_t total, const uint64_t* shape, int64_t rank,
                                  int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                                  int64_t op_code);
// The runtime's own arity table for the elementwise ABI. Asked rather than
// restated: POW/MAX/MIN are binary at op codes 16..18, so `op_code <= 3` is no
// longer an arity test, and a second copy of the rule here would be the copy
// that goes stale.
int eshkol_xla_elementwise_is_binary(int64_t op_code);
void* eshkol_xla_reduce_host(void* arena, const double* data, int64_t total,
                             const uint64_t* shape, int64_t rank, int64_t axis,
                             int64_t op_code);
void* eshkol_xla_transpose_host(void* arena, const double* data, const uint64_t* shape,
                                int64_t rank, const int64_t* perm);
void* eshkol_xla_broadcast_host(void* arena, const double* data, const uint64_t* src_shape,
                                int64_t src_rank, const uint64_t* tgt_shape, int64_t tgt_rank);
void* eshkol_xla_softmax_host(void* arena, const double* data, int64_t total,
                              const uint64_t* shape, int64_t rank, int64_t axis);
// The host reference for the six comparison kinds: an f64 tensor of 0/1,
// which is the encoding a predicate takes at every host boundary.
void* eshkol_xla_compare_host(void* arena, const double* a, const double* b,
                              int64_t a_total, const uint64_t* a_shape, int64_t a_rank,
                              int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                              int64_t direction);

// The public entry points generated code calls.
void* eshkol_xla_matmul(void* arena, const double* a, const double* b,
                        const int64_t* a_shape, const int64_t* b_shape,
                        int64_t a_rank, int64_t b_rank);
void* eshkol_xla_elementwise(void* arena, const double* a, const double* b,
                             int64_t total, const uint64_t* shape, int64_t rank,
                             int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                             int64_t op_code);
void* eshkol_xla_reduce(void* arena, const double* data, int64_t total,
                        const uint64_t* shape, int64_t rank, int64_t axis,
                        int64_t op_code);
void* eshkol_xla_transpose(void* arena, const double* data, const uint64_t* shape,
                           int64_t rank, const int64_t* perm);
void* eshkol_xla_broadcast(void* arena, const double* data, const uint64_t* src_shape,
                           int64_t src_rank, const uint64_t* tgt_shape, int64_t tgt_rank);
void* eshkol_xla_softmax(void* arena, const double* data, int64_t total,
                         const uint64_t* shape, int64_t rank, int64_t axis);
}

namespace {

int g_rows_passed = 0;
int g_rows_failed = 0;
int g_controls_failed = 0;
std::string g_dtype = "f32";

// The comparator, the tolerance classes and the negative control live in
// tests/xla/parity_compare.h so that this harness and the gradient harness
// grade against one convention rather than two that drift apart.
using eshkol_parity::Comparison;
using eshkol_parity::ToleranceClass;
using eshkol_parity::compareArrays;
using eshkol_parity::makeData;
using eshkol_parity::numElements;
using eshkol_parity::shapeText;
using eshkol_parity::toleranceClassName;
using eshkol_parity::toleranceFor;
using eshkol_parity::test_comparator_rejects_a_perturbed_result;

/** @brief Read a tensor's f64 elements out of the bit-pattern storage. */
std::vector<double> tensorValues(void* tensor_ptr, int64_t expected) {
    std::vector<double> out;
    if (!tensor_ptr) return out;
    auto* t = static_cast<eshkol_tensor_t*>(tensor_ptr);
    const int64_t n = static_cast<int64_t>(t->total_elements);
    if (n != expected) return out;
    out.resize(static_cast<size_t>(n));
    const double* src = reinterpret_cast<const double*>(t->elements);
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = src[i];
    return out;
}

/**
 * @brief One parity row: an op, its shapes, its inputs, and the builtins it
 *        covers in lib/backend/xla/builtin_classification.yaml.
 *
 * `builtins` is what joins this harness to the classification the S2a lane
 * produced: the gate counts the device-labelled builtins named across every
 * row that passed, against the total the YAML labels device.
 */
struct ParityCase {
    const char* label;
    DeviceOpRequest request;
    std::vector<std::vector<double>> inputs;
    std::vector<const char*> builtins;
    ToleranceClass tolerance_class = ToleranceClass::Arithmetic;
    // False for an op the public elementwise ABI cannot express, so the row's
    // wiring leg is reported as not applicable instead of failing. Only clamp
    // is such an op today: eshkol_xla_elementwise carries two operands and a
    // clamp has three. Stating that is the point — a row silently graded on
    // one leg would look the same as a row graded on both.
    bool public_entry = true;
};

/**
 * @brief The integer op-code the elementwise C ABI uses for @p kind.
 *
 * Written as a switch rather than an index into DeviceOpKind, which happens to
 * agree today: the ABI numbering is frozen by every compiled object file while
 * DeviceOpKind is free to be reordered, so the coincidence is not something to
 * depend on silently. -1 means "not an elementwise op".
 */
int elementwiseOpCode(DeviceOpKind kind) {
    switch (kind) {
        case DeviceOpKind::Add:      return 0;
        case DeviceOpKind::Subtract: return 1;
        case DeviceOpKind::Multiply: return 2;
        case DeviceOpKind::Divide:   return 3;
        case DeviceOpKind::Exp:      return 4;
        case DeviceOpKind::Log:      return 5;
        case DeviceOpKind::Sin:      return 6;
        case DeviceOpKind::Cos:      return 7;
        case DeviceOpKind::Tanh:     return 8;
        case DeviceOpKind::Relu:     return 9;
        case DeviceOpKind::Sigmoid:  return 10;
        case DeviceOpKind::Sqrt:     return 11;
        case DeviceOpKind::Rsqrt:    return 12;
        case DeviceOpKind::Abs:      return 13;
        case DeviceOpKind::Negate:   return 14;
        case DeviceOpKind::Atanh:    return 15;
        case DeviceOpKind::Pow:      return 16;
        case DeviceOpKind::Maximum:  return 17;
        case DeviceOpKind::Minimum:  return 18;
        default:                     return -1;
    }
}

// makeData() now lives in tests/xla/parity_compare.h, so that both harnesses
// build their inputs the same way and a row in one can be reproduced in the
// other without transcribing numbers.

/** @brief eshkol_xla_compare_host's direction code for a comparison kind:
 *         the DeviceOpKind::Compare* order, EQ NE LT LE GT GE. */
int64_t compareDirection(DeviceOpKind kind) {
    switch (kind) {
        case DeviceOpKind::CompareEq: return 0;
        case DeviceOpKind::CompareNe: return 1;
        case DeviceOpKind::CompareLt: return 2;
        case DeviceOpKind::CompareLe: return 3;
        case DeviceOpKind::CompareGt: return 4;
        case DeviceOpKind::CompareGe: return 5;
        default: return -1;
    }
}

/** @brief The host answer for one case, through the *_host entry points. */
std::vector<double> hostReference(arena_t* arena, const ParityCase& c, std::string* error) {
    const int64_t expected = numElements(c.request.result_shape);
    const auto& shapes = c.request.operand_shapes;

    auto asU64 = [](const std::vector<int64_t>& s) {
        std::vector<uint64_t> u(s.size());
        for (size_t i = 0; i < s.size(); ++i) u[i] = static_cast<uint64_t>(s[i]);
        return u;
    };

    if (elementwiseOpCode(c.request.kind) >= 0) {
        const int op = elementwiseOpCode(c.request.kind);
        const bool binary = eshkol_xla_elementwise_is_binary(op) != 0;
        std::vector<uint64_t> a_shape = asU64(shapes[0]);
        std::vector<uint64_t> b_shape = binary ? asU64(shapes[1]) : std::vector<uint64_t>{};
        void* t = eshkol_xla_elementwise_host(
            arena, c.inputs[0].data(), binary ? c.inputs[1].data() : nullptr,
            numElements(shapes[0]), a_shape.data(), static_cast<int64_t>(a_shape.size()),
            binary ? numElements(shapes[1]) : 0,
            binary ? b_shape.data() : nullptr,
            binary ? static_cast<int64_t>(b_shape.size()) : 0,
            op);
        if (!t) { *error = "host elementwise returned null"; return {}; }
        return tensorValues(t, expected);
    }
    if (c.request.kind == DeviceOpKind::Clamp) {
        // clamp(lo, x, hi) has no op code of its own: the elementwise ABI
        // carries two operands and a clamp has three. The host reference is
        // therefore the composition the device emits — max(min(x, hi), lo) —
        // built from the host's OWN max and min entry points, so the reference
        // is host code rather than a formula written in this file. The bounds
        // are required to be the value's shape here (broadcast them in the
        // case, as the device path does).
        std::vector<uint64_t> vshape = asU64(shapes[1]);
        std::vector<uint64_t> hishape = asU64(shapes[2]);
        std::vector<uint64_t> loshape = asU64(shapes[0]);
        void* capped = eshkol_xla_elementwise_host(
            arena, c.inputs[1].data(), c.inputs[2].data(),
            numElements(shapes[1]), vshape.data(), static_cast<int64_t>(vshape.size()),
            numElements(shapes[2]), hishape.data(), static_cast<int64_t>(hishape.size()),
            18 /* MIN */);
        if (!capped) { *error = "host min returned null for clamp"; return {}; }
        std::vector<double> mid = tensorValues(capped, numElements(shapes[1]));
        if (mid.empty()) { *error = "host min produced no elements for clamp"; return {}; }
        void* t = eshkol_xla_elementwise_host(
            arena, mid.data(), c.inputs[0].data(),
            numElements(shapes[1]), vshape.data(), static_cast<int64_t>(vshape.size()),
            numElements(shapes[0]), loshape.data(), static_cast<int64_t>(loshape.size()),
            17 /* MAX */);
        if (!t) { *error = "host max returned null for clamp"; return {}; }
        return tensorValues(t, expected);
    }

    if (deviceOpYieldsPredicate(c.request.kind)) {
        std::vector<uint64_t> a_shape = asU64(shapes[0]);
        std::vector<uint64_t> b_shape = asU64(shapes[1]);
        void* t = eshkol_xla_compare_host(
            arena, c.inputs[0].data(), c.inputs[1].data(),
            numElements(shapes[0]), a_shape.data(), static_cast<int64_t>(a_shape.size()),
            numElements(shapes[1]), b_shape.data(), static_cast<int64_t>(b_shape.size()),
            compareDirection(c.request.kind));
        if (!t) { *error = "host compare returned null"; return {}; }
        return tensorValues(t, expected);
    }

    switch (c.request.kind) {
        case DeviceOpKind::Matmul: {
            void* t = eshkol_xla_matmul_host(arena, c.inputs[0].data(), c.inputs[1].data(),
                                             shapes[0].data(), shapes[1].data(), 2, 2);
            if (!t) { *error = "host matmul returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Transpose: {
            std::vector<uint64_t> shape = asU64(shapes[0]);
            void* t = eshkol_xla_transpose_host(arena, c.inputs[0].data(), shape.data(),
                                                static_cast<int64_t>(shape.size()),
                                                c.request.axes.data());
            if (!t) { *error = "host transpose returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Broadcast: {
            std::vector<uint64_t> src = asU64(shapes[0]);
            std::vector<uint64_t> tgt = asU64(c.request.result_shape);
            void* t = eshkol_xla_broadcast_host(arena, c.inputs[0].data(), src.data(),
                                                static_cast<int64_t>(src.size()),
                                                tgt.data(), static_cast<int64_t>(tgt.size()));
            if (!t) { *error = "host broadcast returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::ReduceSum:
        case DeviceOpKind::ReduceMean:
        case DeviceOpKind::ReduceMax:
        case DeviceOpKind::ReduceMin:
        case DeviceOpKind::ReduceProd: {
            int op = 0;
            switch (c.request.kind) {
                case DeviceOpKind::ReduceSum:  op = 0; break;
                case DeviceOpKind::ReduceMean: op = 1; break;
                case DeviceOpKind::ReduceMax:  op = 2; break;
                case DeviceOpKind::ReduceMin:  op = 3; break;
                default:                       op = 4; break;
            }
            std::vector<uint64_t> shape = asU64(shapes[0]);
            const int64_t axis = c.request.axes.empty() ? -1 : c.request.axes[0];
            void* t = eshkol_xla_reduce_host(arena, c.inputs[0].data(), numElements(shapes[0]),
                                             shape.data(), static_cast<int64_t>(shape.size()),
                                             axis, op);
            if (!t) { *error = "host reduce returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Softmax: {
            std::vector<uint64_t> shape = asU64(shapes[0]);
            const int64_t axis = c.request.axes.empty() ? -1 : c.request.axes[0];
            void* t = eshkol_xla_softmax_host(arena, c.inputs[0].data(), numElements(shapes[0]),
                                              shape.data(), static_cast<int64_t>(shape.size()), axis);
            if (!t) { *error = "host softmax returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Reshape:
            // Reshape has no host runtime entry point of its own — it is a
            // metadata change. It is not exercised as a parity row for that
            // reason, and saying so is more useful than inventing a reference.
            *error = "reshape has no host runtime entry point to compare against";
            return {};
        default:
            break;
    }
    *error = "unhandled op kind in hostReference";
    return {};
}

/** @brief The answer from the PUBLIC entry point, with the device enabled. */
std::vector<double> publicEntryPoint(arena_t* arena, const ParityCase& c, std::string* error) {
    const int64_t expected = numElements(c.request.result_shape);
    const auto& shapes = c.request.operand_shapes;
    auto asU64 = [](const std::vector<int64_t>& s) {
        std::vector<uint64_t> u(s.size());
        for (size_t i = 0; i < s.size(); ++i) u[i] = static_cast<uint64_t>(s[i]);
        return u;
    };

    if (elementwiseOpCode(c.request.kind) >= 0) {
        const int op = elementwiseOpCode(c.request.kind);
        const bool binary = eshkol_xla_elementwise_is_binary(op) != 0;
        std::vector<uint64_t> a_shape = asU64(shapes[0]);
        std::vector<uint64_t> b_shape = binary ? asU64(shapes[1]) : std::vector<uint64_t>{};
        void* t = eshkol_xla_elementwise(
            arena, c.inputs[0].data(), binary ? c.inputs[1].data() : nullptr,
            numElements(shapes[0]), a_shape.data(), static_cast<int64_t>(a_shape.size()),
            binary ? numElements(shapes[1]) : 0,
            binary ? b_shape.data() : nullptr,
            binary ? static_cast<int64_t>(b_shape.size()) : 0,
            op);
        if (!t) { *error = "public elementwise returned null"; return {}; }
        return tensorValues(t, expected);
    }

    switch (c.request.kind) {
        case DeviceOpKind::Matmul: {
            void* t = eshkol_xla_matmul(arena, c.inputs[0].data(), c.inputs[1].data(),
                                        shapes[0].data(), shapes[1].data(), 2, 2);
            if (!t) { *error = "public matmul returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Transpose: {
            std::vector<uint64_t> shape = asU64(shapes[0]);
            void* t = eshkol_xla_transpose(arena, c.inputs[0].data(), shape.data(),
                                           static_cast<int64_t>(shape.size()),
                                           c.request.axes.data());
            if (!t) { *error = "public transpose returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Broadcast: {
            std::vector<uint64_t> src = asU64(shapes[0]);
            std::vector<uint64_t> tgt = asU64(c.request.result_shape);
            void* t = eshkol_xla_broadcast(arena, c.inputs[0].data(), src.data(),
                                           static_cast<int64_t>(src.size()),
                                           tgt.data(), static_cast<int64_t>(tgt.size()));
            if (!t) { *error = "public broadcast returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::ReduceSum:
        case DeviceOpKind::ReduceMean:
        case DeviceOpKind::ReduceMax:
        case DeviceOpKind::ReduceMin:
        case DeviceOpKind::ReduceProd: {
            int op = 0;
            switch (c.request.kind) {
                case DeviceOpKind::ReduceSum:  op = 0; break;
                case DeviceOpKind::ReduceMean: op = 1; break;
                case DeviceOpKind::ReduceMax:  op = 2; break;
                case DeviceOpKind::ReduceMin:  op = 3; break;
                default:                       op = 4; break;
            }
            std::vector<uint64_t> shape = asU64(shapes[0]);
            const int64_t axis = c.request.axes.empty() ? -1 : c.request.axes[0];
            void* t = eshkol_xla_reduce(arena, c.inputs[0].data(), numElements(shapes[0]),
                                        shape.data(), static_cast<int64_t>(shape.size()),
                                        axis, op);
            if (!t) { *error = "public reduce returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Softmax: {
            std::vector<uint64_t> shape = asU64(shapes[0]);
            const int64_t axis = c.request.axes.empty() ? -1 : c.request.axes[0];
            void* t = eshkol_xla_softmax(arena, c.inputs[0].data(), numElements(shapes[0]),
                                         shape.data(), static_cast<int64_t>(shape.size()), axis);
            if (!t) { *error = "public softmax returned null"; return {}; }
            return tensorValues(t, expected);
        }
        default:
            *error = "no public entry point for this op";
            return {};
    }
}

/** @brief Build the parity table: every op lowered, with a rank-2 and a
 *         broadcast case among them as the brief requires. */
std::vector<ParityCase> buildCases() {
    std::vector<ParityCase> cases;

    auto elementwise2d = [&](const char* label, DeviceOpKind kind,
                             std::vector<const char*> builtins,
                             double b_base) {
        ParityCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(24, b_base, 0.125)};
        c.builtins = std::move(builtins);
        cases.push_back(c);
    };

    // ── Elementwise binary, non-trivial rank 2 ──
    elementwise2d("add   f64[4,6] + f64[4,6]", DeviceOpKind::Add,
                  {"tensor-add", "add2", "+"}, 1.25);
    elementwise2d("sub   f64[4,6] - f64[4,6]", DeviceOpKind::Subtract,
                  {"tensor-sub", "sub2", "-"}, 1.25);
    elementwise2d("mul   f64[4,6] * f64[4,6]", DeviceOpKind::Multiply,
                  {"tensor-mul", "mul2", "*"}, 1.25);
    elementwise2d("div   f64[4,6] / f64[4,6]", DeviceOpKind::Divide,
                  {"tensor-div", "div2", "/"}, 1.25);

    // ── Broadcast case: [4,6] against [6], the shape a bias vector has ──
    {
        ParityCase c;
        c.label = "add   f64[4,6] + f64[6]   (broadcast)";
        c.request.kind = DeviceOpKind::Add;
        c.request.operand_shapes = {{4, 6}, {6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(6, 2.0, 0.5)};
        c.builtins = {"tensor-add"};
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "mul   f64[4,6] * f64[6]   (broadcast)";
        c.request.kind = DeviceOpKind::Multiply;
        c.request.operand_shapes = {{4, 6}, {6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(6, 1.5, 0.25)};
        c.builtins = {"tensor-mul", "tensor-scale"};
        cases.push_back(c);
    }

    // ── Elementwise unary ──
    auto unary = [&](const char* label, DeviceOpKind kind,
                     std::vector<const char*> builtins, double base, double step) {
        ParityCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, base, step)};
        c.builtins = std::move(builtins);
        // Every op built through this helper is an approximated elementary
        // function, not an exact one.
        c.tolerance_class = ToleranceClass::Transcendental;
        cases.push_back(c);
    };
    // exp: kept inside [-2, 2] so the f32 result never approaches its dynamic
    // range and the comparison measures the op rather than an overflow.
    unary("exp   f64[4,6]", DeviceOpKind::Exp, {"tensor-exp", "exp"}, -2.0, 0.16);
    // log: strictly positive inputs — log of a non-positive is a domain
    // question, not a parity question, and belongs in its own test.
    unary("log   f64[4,6]", DeviceOpKind::Log, {"tensor-log", "log"}, 0.25, 0.25);
    unary("sin   f64[4,6]", DeviceOpKind::Sin, {"tensor-sin", "sin"}, -1.5, 0.125);
    unary("cos   f64[4,6]", DeviceOpKind::Cos, {"tensor-cos", "cos"}, -1.5, 0.125);
    unary("tanh  f64[4,6]", DeviceOpKind::Tanh, {"tanh"}, -1.5, 0.125);
    // sqrt / rsqrt: strictly positive. sqrt(0) is 0 and rsqrt(0) is infinite,
    // and neither is a parity question about the op.
    unary("sqrt  f64[4,6]", DeviceOpKind::Sqrt, {"sqrt", "tensor-sqrt"}, 0.25, 0.25);
    unary("rsqrt f64[4,6]", DeviceOpKind::Rsqrt, {}, 0.25, 0.25);
    unary("sigmoid f64[4,6]", DeviceOpKind::Sigmoid, {"sigmoid"}, -3.0, 0.25);
    // atanh: |x| < 1 by construction (-0.9 .. 0.825), away from both poles.
    unary("atanh f64[4,6]", DeviceOpKind::Atanh, {"atanh"}, -0.9, 0.075);

    // abs / negate are EXACT, not transcendental, so they are built directly
    // rather than through the unary() helper, which stamps every row it makes
    // with the transcendental class.
    //
    // The abs row's data crosses zero and HITS it exactly: -1.5 + 0.125*12 =
    // 0. That element is where the derivative convention lives (sign(0) = 0,
    // matching AD_ABS in lib/backend/vm_autodiff.c), so the gradient harness's
    // abs row reuses this data to exercise it.
    {
        ParityCase c;
        c.label = "abs   f64[4,6]";
        c.request.kind = DeviceOpKind::Abs;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, -1.5, 0.125)};
        c.builtins = {"abs", "tensor-abs"};
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "negate f64[4,6]";
        c.request.kind = DeviceOpKind::Negate;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, -1.5, 0.25)};
        c.builtins = {};
        cases.push_back(c);
    }

    // ── Elementwise binary: pow, maximum, minimum ──
    //
    // The max/min operands CROSS: makeData(24, 0.5, 0.25) rises from 0.5 and
    // makeData(24, 6.25, -0.25) falls from 6.25, so they are equal at index
    // 11.5 — i.e. never exactly, which is deliberate here. A tie is a
    // GRADIENT question (which operand receives the cotangent) and it is
    // exercised in the gradient harness, where the answer differs; the
    // forward value at a tie is the same either way, so a tie row here would
    // measure nothing this row does not.
    {
        ParityCase c;
        c.label = "pow   f64[4,6] ^ f64[4,6]";
        c.request.kind = DeviceOpKind::Pow;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.3, 0.2), makeData(24, 0.5, 0.125)};
        c.builtins = {"expt"};
        c.tolerance_class = ToleranceClass::Transcendental;
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "maximum f64[4,6]";
        c.request.kind = DeviceOpKind::Maximum;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(24, 6.25, -0.25)};
        c.builtins = {"max", "tensor-max"};
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "minimum f64[4,6]";
        c.request.kind = DeviceOpKind::Minimum;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(24, 6.25, -0.25)};
        c.builtins = {"min", "tensor-min"};
        cases.push_back(c);
    }
    // clamp(lo, x, hi): the value sweeps through both bounds, so the row
    // covers all three regimes (below lo, inside, above hi) rather than only
    // the pass-through one.
    {
        ParityCase c;
        c.label = "clamp f64[4,6] into [-1, 2]";
        c.request.kind = DeviceOpKind::Clamp;
        c.request.operand_shapes = {{4, 6}, {4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {std::vector<double>(24, -1.0),
                    makeData(24, -2.5, 0.25),
                    std::vector<double>(24, 2.0)};
        c.builtins = {};
        c.public_entry = false;
        cases.push_back(c);
    }

    // ── Activations. relu is exact (a maximum against zero), so it is held
    //    to the arithmetic bound; sigmoid goes through stablehlo.logistic and
    //    softmax through an exponential, so both are transcendental. ──
    {
        ParityCase c;
        c.label = "relu  f64[4,6]";
        c.request.kind = DeviceOpKind::Relu;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        // Spans zero in both directions, so the branch actually branches.
        c.inputs = {makeData(24, -1.5, 0.125)};
        c.builtins = {"relu"};
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "sigmoid f64[4,6]";
        c.request.kind = DeviceOpKind::Sigmoid;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, -1.5, 0.125)};
        c.builtins = {"sigmoid"};
        c.tolerance_class = ToleranceClass::Transcendental;
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "softmax f64[4,6] all";
        c.request.kind = DeviceOpKind::Softmax;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, -1.5, 0.125)};
        c.builtins = {"softmax"};
        c.tolerance_class = ToleranceClass::Transcendental;
        cases.push_back(c);
    }
    {
        ParityCase c;
        c.label = "softmax f64[4,6] axis 1";
        c.request.kind = DeviceOpKind::Softmax;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.request.axes = {1};
        c.inputs = {makeData(24, -1.5, 0.125)};
        c.builtins = {"softmax"};
        c.tolerance_class = ToleranceClass::Transcendental;
        cases.push_back(c);
    }

    // ── Matmul: non-square, so a transposed contraction would be visible ──
    {
        ParityCase c;
        c.label = "matmul f64[4,6] x f64[6,3]";
        c.request.kind = DeviceOpKind::Matmul;
        c.request.operand_shapes = {{4, 6}, {6, 3}};
        c.request.result_shape = {4, 3};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(18, -1.0, 0.125)};
        c.builtins = {"tensor-matmul", "matmul", "tensor-dot"};
        cases.push_back(c);
    }

    // ── Matmul with operands that are NOT bf16-exact ──
    //
    // The row above cannot observe the precision the dot is computed in.
    // makeData(24, 0.5, 0.25) and makeData(18, -1.0, 0.125) are multiples of
    // 0.25 and 0.125 at small magnitudes: every one of them is exactly
    // representable in bf16's 8 mantissa bits, so a matrix unit that rounds
    // its operands to bf16 — which is what a TPU does for an f32 dot at
    // DEFAULT precision — returns the identical answer and the row measures
    // exactly 0 error. That is how three decimal digits went missing from
    // every matmul in this program without any row reporting it; the gradient
    // harness's two-layer composite, whose dot operands are tanh outputs, is
    // where it finally showed up as 2^-9 relative.
    //
    // These steps are not dyadic. Measured on TPU for THIS row: 9.018e-8
    // relative with the HIGHEST precision_config
    // lib/backend/xla/stablehlo_emitter.cpp now emits, and 4.829e-3 with
    // ESHKOL_XLA_DOT_PRECISION=default, which is a FAIL against the 1e-5
    // arithmetic bound. The dyadic row above reads 0.000e+00 under both, which
    // is the whole point. The row exists so that a return to the silent
    // demotion cannot pass this gate again.
    {
        ParityCase c;
        c.label = "matmul f64[4,6] x f64[6,3] non-dyadic";
        c.request.kind = DeviceOpKind::Matmul;
        c.request.operand_shapes = {{4, 6}, {6, 3}};
        c.request.result_shape = {4, 3};
        c.inputs = {makeData(24, 0.31, 0.17), makeData(18, -0.83, 0.13)};
        c.builtins = {"tensor-matmul"};
        cases.push_back(c);
    }

    // ── Transpose ──
    {
        ParityCase c;
        c.label = "transpose f64[4,6] -> f64[6,4]";
        c.request.kind = DeviceOpKind::Transpose;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {6, 4};
        c.request.axes = {1, 0};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.builtins = {"tensor-transpose", "transpose"};
        cases.push_back(c);
    }

    // ── Transpose, rank 3, with a permutation that is NOT its own inverse ──
    //
    // The rank-2 case above cannot distinguish the two permutation
    // conventions (result.dim(i)=operand.dim(perm[i]) versus its inverse),
    // because {1,0} is self-inverse: both readings give the same answer, so a
    // convention error would pass. {2,0,1} is not self-inverse and its
    // inverse is {1,2,0}, so the two readings give different results and only
    // one of them matches the host.
    //
    // It is also the case that caught the layout defect: XLA may compile a
    // transpose to nothing at all and give the result buffer a permuted
    // minor-to-major layout, which a read-back under the buffer's own layout
    // returns as the operand, unpermuted and without any error. See
    // PjrtClient::bufferToHost.
    {
        ParityCase c;
        c.label = "transpose f64[2,3,4] perm{2,0,1}";
        c.request.kind = DeviceOpKind::Transpose;
        c.request.operand_shapes = {{2, 3, 4}};
        c.request.result_shape = {4, 2, 3};
        c.request.axes = {2, 0, 1};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.builtins = {"tensor-transpose"};
        cases.push_back(c);
    }

    // ── Broadcast op ──
    {
        ParityCase c;
        c.label = "broadcast f64[6] -> f64[4,6]";
        c.request.kind = DeviceOpKind::Broadcast;
        c.request.operand_shapes = {{6}};
        c.request.result_shape = {4, 6};
        c.request.axes = {1};
        c.inputs = {makeData(6, 2.0, 0.5)};
        c.builtins = {};
        cases.push_back(c);
    }

    // ── Reductions, full and along an axis ──
    auto reduce = [&](const char* label, DeviceOpKind kind, std::vector<int64_t> axes,
                      std::vector<int64_t> result_shape, std::vector<const char*> builtins) {
        ParityCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = std::move(result_shape);
        c.request.axes = std::move(axes);
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.builtins = std::move(builtins);
        cases.push_back(c);
    };
    reduce("reduce_sum  f64[4,6] all", DeviceOpKind::ReduceSum, {}, {},
           {"tensor-sum", "_tensor-reduce-sum", "tensor-reduce-all"});
    reduce("reduce_sum  f64[4,6] axis 1", DeviceOpKind::ReduceSum, {1}, {4},
           {"tensor-reduce"});
    reduce("reduce_mean f64[4,6] all", DeviceOpKind::ReduceMean, {}, {},
           {"tensor-mean", "_tensor-reduce-mean"});
    reduce("reduce_mean f64[4,6] axis 0", DeviceOpKind::ReduceMean, {0}, {6}, {});
    reduce("reduce_max  f64[4,6] all", DeviceOpKind::ReduceMax, {}, {},
           {"tensor-max", "_tensor-reduce-max"});
    reduce("reduce_min  f64[4,6] all", DeviceOpKind::ReduceMin, {}, {},
           {"tensor-min", "_tensor-reduce-min"});

    // ── Comparisons, all six directions (S5b). ──
    // The operands cross: a rises from 0.5 by 0.25 and b falls from 6.0 by
    // 0.25, so they are equal at exactly one index (11), a is below b before
    // it and above after. Every direction therefore has both outcomes in the
    // row AND a tie, which is what separates <= from < and >= from >; a row
    // without a tie would pass a lowering that emitted the strict direction
    // for both. The result is 0/1 in f64 on both sides.
    //
    // No public elementwise entry point carries a comparison (the ABI's op
    // codes are the arithmetic and elementary functions), so the wiring leg
    // is reported as not applicable rather than faked; the host reference is
    // eshkol_xla_compare_host(), added for exactly this measurement.
    auto compare = [&](const char* label, DeviceOpKind kind, std::vector<const char*> builtins) {
        ParityCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(24, 6.0, -0.25)};
        c.builtins = std::move(builtins);
        c.public_entry = false;
        cases.push_back(c);
    };
    compare("compare_eq f64[4,6] (tie at 11)", DeviceOpKind::CompareEq, {"="});
    compare("compare_ne f64[4,6] (tie at 11)", DeviceOpKind::CompareNe, {});
    compare("compare_lt f64[4,6] (tie at 11)", DeviceOpKind::CompareLt, {"<"});
    compare("compare_le f64[4,6] (tie at 11)", DeviceOpKind::CompareLe, {"<="});
    compare("compare_gt f64[4,6] (tie at 11)", DeviceOpKind::CompareGt, {">"});
    compare("compare_ge f64[4,6] (tie at 11)", DeviceOpKind::CompareGe, {">="});
    // A scalar against a tensor: the shape a conditional's predicate has when
    // it compares a reduction against a threshold, and the broadcast leg of
    // the comparison lowering.
    {
        ParityCase c;
        c.label = "compare_gt f64[4,6] > f64[] (broadcast)";
        c.request.kind = DeviceOpKind::CompareGt;
        c.request.operand_shapes = {{4, 6}, {}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), {3.25}};
        c.builtins = {">"};
        c.public_entry = false;
        cases.push_back(c);
    }

    return cases;
}

void printHeader() {
    std::printf("\n%-38s %-6s %-15s %-12s %-12s %-9s %s\n",
                "op / shapes", "dtype", "class", "max abs err", "max rel err",
                "tol", "result");
    std::printf("%-38s %-6s %-15s %-12s %-12s %-9s %s\n",
                "--------------------------------------", "------",
                "---------------", "------------", "------------",
                "---------", "------");
}

}  // namespace

int main() {
    std::cout << "=========================================" << std::endl;
    std::cout << "  XLA Op Surface Parity (device vs host)" << std::endl;
    std::cout << "=========================================" << std::endl;

    // Ask for device execution before anything latches the answer. This is set
    // here rather than left to the caller so the harness tests what it says it
    // tests no matter how it is invoked; deviceExecutionRequested() reads the
    // variable exactly once, on first use, and nothing above has used it yet.
    ::setenv("ESHKOL_XLA_PJRT", "1", 1);

    if (!test_comparator_rejects_a_perturbed_result()) {
        g_controls_failed++;
        std::cerr << "The comparator control failed; no parity row below would mean anything."
                  << std::endl;
        return 1;
    }

    DeviceExecutor* executor = registerStableHLODeviceExecutor();
    if (!executor) {
        std::cout << "SKIP: no device executor could be installed in this build." << std::endl;
        return 77;
    }
    std::string why;
    if (!executor->available(&why)) {
        std::cout << "SKIP: no PJRT device available: " << why << std::endl;
        return 77;
    }

    g_dtype = executor->dtypeName();
    // docs/design/ESHKOL_S_FRAGMENT.md, "parity rule": the arithmetic class
    // keeps the per-dtype bound (1e-5 f32, 1e-9 f64); the transcendental class
    // is 100x it. Absolute or relative, whichever is looser, in both cases.
    eshkol_parity::setTolerancesForDtype(g_dtype);

    std::cout << "Device: " << executor->description() << std::endl;
    std::cout << "Tolerance (" << g_dtype << ", absolute or relative, whichever is looser; "
                 "docs/design/ESHKOL_S_FRAGMENT.md): arithmetic="
              << eshkol_parity::g_tol_arithmetic
              << " transcendental=" << eshkol_parity::g_tol_transcendental
              << std::endl;

    const char* force_fail = std::getenv("ESHKOL_XLA_PARITY_FORCE_FAIL");
    if (force_fail && force_fail[0]) {
        std::cout << "FORCED FAIL requested for op '" << force_fail
                  << "': its host reference will be perturbed by 1.0." << std::endl;
    }

    arena_t* arena = arena_create(1024 * 1024);
    if (!arena) {
        std::cerr << "FAIL: could not create an arena" << std::endl;
        return 1;
    }

    std::vector<ParityCase> cases = buildCases();
    std::vector<std::string> covered_builtins;

    printHeader();
    for (const ParityCase& c : cases) {
        const int64_t expected = numElements(c.request.result_shape);

        std::string host_error;
        std::vector<double> host = hostReference(arena, c, &host_error);
        const double tol = toleranceFor(c.tolerance_class);
        const char* cls = toleranceClassName(c.tolerance_class);
        if (host.empty()) {
            std::printf("%-38s %-6s %-15s %-12s %-12s %-9s FAIL (host: %s)\n",
                        c.label, g_dtype.c_str(), cls, "-", "-", "-",
                        host_error.c_str());
            g_rows_failed++;
            continue;
        }
        if (force_fail && std::strcmp(force_fail, deviceOpKindName(c.request.kind)) == 0) {
            host[0] += 1.0;
        }

        std::vector<double> device(static_cast<size_t>(expected), 0.0);
        std::vector<const double*> operands;
        for (const auto& in : c.inputs) operands.push_back(in.data());
        std::string device_error;
        if (!executor->run(c.request, operands, device.data(), &device_error)) {
            std::printf("%-38s %-6s %-15s %-12s %-12s %-9s FAIL (device: %s)\n",
                        c.label, g_dtype.c_str(), cls, "-", "-", "-",
                        device_error.c_str());
            g_rows_failed++;
            continue;
        }

        Comparison cmp = compareArrays(device, host, tol);

        // The public entry point must reach the same device answer: this is
        // the wiring check described at the top of this file.
        std::string public_error;
        std::vector<double> via_public;
        bool wiring_ok = false;
        if (!c.public_entry) {
            wiring_ok = true;
            public_error = deviceOpYieldsPredicate(c.request.kind)
                ? "n/a (no public elementwise entry point carries a comparison)"
                : "n/a (no public elementwise entry point for a 3-operand op)";
        } else {
            via_public = publicEntryPoint(arena, c, &public_error);
        }
        if (wiring_ok) {
            // nothing further to check for this row
        } else if (via_public.size() == device.size()) {
            Comparison wiring = compareArrays(via_public, device, tol);
            wiring_ok = wiring.agreed;
            if (!wiring_ok) public_error = "public entry point disagreed with the device result";
        } else if (public_error.empty()) {
            public_error = "public entry point produced " + std::to_string(via_public.size()) +
                           " elements, expected " + std::to_string(device.size());
        }

        const bool row_ok = cmp.agreed && wiring_ok;
        std::printf("%-38s %-6s %-15s %-12.3e %-12.3e %-9.1e %s\n",
                    c.label, g_dtype.c_str(), cls, cmp.max_abs, cmp.max_rel, tol,
                    row_ok ? "PASS" : "FAIL");
        if (!row_ok) {
            g_rows_failed++;
            if (!cmp.agreed) {
                const int i = cmp.worst_index;
                std::printf("       first disagreement at index %d: device=%.17g host=%.17g\n",
                            i, i >= 0 ? device[static_cast<size_t>(i)] : 0.0,
                            i >= 0 ? host[static_cast<size_t>(i)] : 0.0);
            }
            if (!wiring_ok) {
                std::printf("       wiring: %s\n", public_error.c_str());
            }
        } else {
            g_rows_passed++;
            for (const char* b : c.builtins) covered_builtins.push_back(b);
        }
    }

    arena_destroy(arena);

    // Deduplicate the builtin coverage so the count is of distinct builtins.
    std::vector<std::string> distinct;
    for (const std::string& b : covered_builtins) {
        bool seen = false;
        for (const std::string& d : distinct) {
            if (d == b) { seen = true; break; }
        }
        if (!seen) distinct.push_back(b);
    }

    const eshkol::xla::DeviceStats stats = executor->stats();

    std::cout << std::endl;
    std::cout << "Rows passed: " << g_rows_passed << std::endl;
    std::cout << "Rows failed: " << g_rows_failed << std::endl;
    std::cout << "Executable cache: " << stats.compiled << " compiled, "
              << stats.cache_hits << " reused, " << stats.executed << " executions, "
              << stats.failures << " failures" << std::endl;

    // One machine-readable line the gate parses. COVERED_BUILTINS names the
    // device-labelled builtins in lib/backend/xla/builtin_classification.yaml
    // that a passing row exercised; the gate joins it against that file.
    std::cout << "SUMMARY: rows_passed=" << g_rows_passed
              << " rows_failed=" << g_rows_failed
              << " dtype=" << g_dtype
              << " tol_arithmetic=" << eshkol_parity::g_tol_arithmetic
              << " tol_transcendental=" << eshkol_parity::g_tol_transcendental
              << " compiled=" << stats.compiled
              << " cache_hits=" << stats.cache_hits << std::endl;
    std::cout << "COVERED_BUILTINS:";
    for (const std::string& b : distinct) std::cout << " " << b;
    std::cout << std::endl;

    if (g_controls_failed != 0 || g_rows_failed != 0) return 1;
    return 0;
}
