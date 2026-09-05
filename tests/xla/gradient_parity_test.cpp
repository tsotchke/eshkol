/*
 * Device/host differential parity for REVERSE-MODE GRADIENTS of Eshkol's
 * lowered tensor operations.
 *
 * WHAT THIS PROVES.
 *
 * For each operation lib/backend/xla/device_lowering.cpp lowers to StableHLO,
 * this computes the same vector-Jacobian product two ways in the same process:
 *
 *   device — the VJP emitted into StableHLO by StableHLOEmitter::emitVJP(),
 *            compiled through PJRT and executed on whatever device this host
 *            provides, in the device element type (f32 unless
 *            ESHKOL_XLA_DEVICE_DTYPE says otherwise). The forward pass and its
 *            backward pass are one `func.func @main`, so the backward is
 *            device code and not a host post-pass;
 *   host   — the reference cotangent, in f64, from the source named in the
 *            row's "reference" column (see WHAT THE HOST REFERENCE IS below);
 *
 * and grades them with the comparator in tests/xla/parity_compare.h, under the
 * tolerance class of the operation being DIFFERENTIATED — the VJP of tanh
 * evaluates tanh, so it inherits the transcendental bound; the VJP of a matmul
 * is two more matmuls and is exact.
 *
 * It then calls eshkol_xla_gradient(), the public runtime entry point that
 * returns a cotangent as an Eshkol tensor, and requires that to agree with the
 * device answer. That third comparison is the wiring check: a VJP that is
 * correct but not reachable from the runtime would pass the first two and fail
 * this one.
 *
 * WHAT THE HOST REFERENCE IS, AND WHY IT IS NOT ONE SOURCE.
 *
 * Eshkol's host reverse-mode AD exposes plain-buffer backward entry points for
 * exactly five of these operations — matmul, rank-2 transpose, reshape, full
 * sum and full mean (inc/eshkol/backend/tensor_backward.h). Those rows call
 * them directly, and the "reference" column says so.
 *
 * For the rest — every elementwise VJP, broadcast, axis reductions, and
 * max/min — there is no host AD entry point in this tree at all: no AD node
 * type in inc/eshkol/ad_node_registry.def carries a tensor reduce-max, and the
 * elementwise tensor backward is not present here. Inventing one and calling
 * it "the host" would be grading the device against a number written in this
 * file. So those rows state their rule explicitly AND every row that admits a
 * finite difference is cross-checked against a central difference of the HOST
 * FORWARD primitive (eshkol_xla_*_host) at h = 1e-5, reported in its own
 * column. A row passes only if both the device comparison and, where
 * admissible, the finite-difference cross-check agree. The finite difference
 * is an independent witness over the host's own forward code; it is the thing
 * that makes an analytic rule in this file a reference rather than an opinion.
 *
 * Rows where a finite difference is INADMISSIBLE say so and say why, rather
 * than being quietly graded on one leg: a max/min reduction at a tie is not
 * differentiable, and a central difference across the tie returns a confident
 * average of two different one-sided derivatives.
 *
 * THE TIE CONVENTION FOR REDUCE MAX/MIN.
 *
 * A reduction with repeated extrema has no derivative, only a choice, and the
 * choice must be the host's or a program's gradient would depend on whether it
 * ran on a device. Eshkol's host AD defines max and min only for SCALARS
 * (AD_NODE_MAX/AD_NODE_MIN, lib/backend/autodiff_codegen.cpp): the whole
 * gradient goes to the first operand when it is strictly greater, and
 * otherwise to the SECOND. A reduction spelled the only way Eshkol can spell
 * one — a fold of that scalar op — therefore gives the whole cotangent to the
 * LAST tied element in row-major order, and nothing to the others.
 *
 * That is the convention this harness requires of the device, and the
 * reference for a tie row is computed by folding the host's scalar rule
 * literally, element by element, rather than by restating it as a formula.
 * tests/xla/host_max_tie_convention.esk exercises the same rule through the
 * host language AD so the convention is a measured fact rather than a reading
 * of the code; the gate runs it alongside this harness.
 *
 * A GATE THAT CANNOT FAIL IS WORTHLESS.
 *
 * Three things here can fail, by construction:
 *   - The comparator control in tests/xla/parity_compare.h runs before any
 *     device is required and feeds the comparator a deliberately wrong result.
 *   - A perturbed-cotangent control: one row's expected cotangent is perturbed
 *     by 1.0 and must be REJECTED. A row that passed both ways would mean the
 *     grading is not connected to the numbers.
 *   - ESHKOL_XLA_GRADIENT_FORCE_FAIL=<op name> perturbs the host reference for
 *     that op by 1.0, so a real end-to-end FAIL can be demonstrated against
 *     live hardware without editing this file.
 *
 * And one row is a REFUSAL row: reduce_prod has no VJP rule on purpose (the
 * textbook grad*out/input is inf/NaN wherever the input contains a zero), and
 * the row passes only if the device refuses it with a diagnostic that names
 * the reason. Fail-closed is a behaviour, so it is tested like one.
 *
 * Exit status: 0 all rows agreed; 1 a row disagreed or a control failed;
 * 77 no PJRT device was reachable (the caller decides what that means; the XLA
 * gate treats it as FAIL, since a gradient parity claim needs a device).
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

#include "eshkol/backend/tensor_backward.h"
#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/stablehlo_emitter.h"
#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_types.h"

#include "parity_compare.h"

#include "../../lib/core/arena_memory.h"

using eshkol::xla::DeviceExecutor;
using eshkol::xla::DeviceOpKind;
using eshkol::xla::DeviceOpRequest;
using eshkol::xla::ElementType;
using eshkol::xla::StableHLOEmitter;
using eshkol::xla::VJPResult;
using eshkol::xla::deviceOpKindName;
using eshkol::xla::registerStableHLODeviceExecutor;

using eshkol_parity::Comparison;
using eshkol_parity::ToleranceClass;
using eshkol_parity::compareArrays;
using eshkol_parity::makeData;
using eshkol_parity::numElements;
using eshkol_parity::shapeText;
using eshkol_parity::toleranceClassName;
using eshkol_parity::toleranceFor;
using eshkol_parity::test_comparator_rejects_a_perturbed_result;

// The host forward implementations, under the names they carry since the
// device split in lib/backend/xla/xla_runtime.cpp. These are what the finite
// difference differentiates.
extern "C" {
void* eshkol_xla_matmul_host(void* arena, const double* a, const double* b,
                             const int64_t* a_shape, const int64_t* b_shape,
                             int64_t a_rank, int64_t b_rank);
void* eshkol_xla_elementwise_host(void* arena, const double* a, const double* b,
                                  int64_t total, const uint64_t* shape, int64_t rank,
                                  int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                                  int64_t op_code);
void* eshkol_xla_reduce_host(void* arena, const double* data, int64_t total,
                             const uint64_t* shape, int64_t rank, int64_t axis,
                             int64_t op_code);
void* eshkol_xla_transpose_host(void* arena, const double* data, const uint64_t* shape,
                                int64_t rank, const int64_t* perm);
void* eshkol_xla_broadcast_host(void* arena, const double* data, const uint64_t* src_shape,
                                int64_t src_rank, const uint64_t* tgt_shape, int64_t tgt_rank);

// The public gradient entry point, which returns a cotangent as an Eshkol
// tensor (lib/backend/xla/xla_runtime.cpp).
void* eshkol_xla_gradient(void* arena, int64_t op_kind, int64_t num_operands,
                          const double* const* operands,
                          const uint64_t* const* operand_shapes,
                          const int64_t* operand_ranks,
                          const double* cotangent,
                          const uint64_t* result_shape, int64_t result_rank,
                          const int64_t* axes, int64_t num_axes,
                          int64_t which_operand);
}

namespace {

int g_rows_passed = 0;
int g_rows_failed = 0;
int g_controls_failed = 0;
std::string g_dtype = "f32";

/** @brief The central-difference step, stated once. */
constexpr double kFdStep = 1e-5;
/** @brief How closely a host reference must match its finite difference.
 *         Absolute or relative, whichever is looser, as everywhere else. */
constexpr double kFdTolerance = 1e-6;

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

std::vector<uint64_t> asU64(const std::vector<int64_t>& s) {
    std::vector<uint64_t> u(s.size());
    for (size_t i = 0; i < s.size(); ++i) u[i] = static_cast<uint64_t>(s[i]);
    return u;
}

/** @brief Row-major strides of @p shape. */
std::vector<int64_t> stridesOf(const std::vector<int64_t>& shape) {
    std::vector<int64_t> st(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        st[static_cast<size_t>(i)] = st[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return st;
}

/**
 * @brief NumPy-style right-aligned broadcast of @p src onto @p dst_shape.
 *
 * Matches broadcastDims()/inferResultShape() in device_lowering.cpp: operand
 * dimension i maps to result dimension (rank_dst - rank_src + i), and an
 * extent of 1 is repeated. Written against that file rather than from memory,
 * because a reference that broadcasts differently from the thing it grades
 * measures the difference between two conventions and calls it an error.
 */
std::vector<double> broadcastTo(const std::vector<double>& src,
                                const std::vector<int64_t>& src_shape,
                                const std::vector<int64_t>& dst_shape) {
    const int64_t dst_n = numElements(dst_shape);
    std::vector<double> out(static_cast<size_t>(dst_n), 0.0);
    const auto dst_st = stridesOf(dst_shape);
    const auto src_st = stridesOf(src_shape);
    const size_t offset = dst_shape.size() - src_shape.size();
    for (int64_t flat = 0; flat < dst_n; ++flat) {
        int64_t rem = flat;
        int64_t src_flat = 0;
        for (size_t d = 0; d < dst_shape.size(); ++d) {
            const int64_t idx = rem / dst_st[d];
            rem %= dst_st[d];
            if (d < offset) continue;
            const size_t sd = d - offset;
            const int64_t si = (src_shape[sd] == 1) ? 0 : idx;
            src_flat += si * src_st[sd];
        }
        out[static_cast<size_t>(flat)] = src[static_cast<size_t>(src_flat)];
    }
    return out;
}

/** @brief The transpose of broadcastTo: sum @p g back onto @p target_shape. */
std::vector<double> unbroadcast(const std::vector<double>& g,
                                const std::vector<int64_t>& g_shape,
                                const std::vector<int64_t>& target_shape) {
    const int64_t g_n = numElements(g_shape);
    std::vector<double> out(static_cast<size_t>(numElements(target_shape)), 0.0);
    const auto g_st = stridesOf(g_shape);
    const auto t_st = stridesOf(target_shape);
    const size_t offset = g_shape.size() - target_shape.size();
    for (int64_t flat = 0; flat < g_n; ++flat) {
        int64_t rem = flat;
        int64_t t_flat = 0;
        for (size_t d = 0; d < g_shape.size(); ++d) {
            const int64_t idx = rem / g_st[d];
            rem %= g_st[d];
            if (d < offset) continue;
            const size_t td = d - offset;
            const int64_t ti = (target_shape[td] == 1) ? 0 : idx;
            t_flat += ti * t_st[td];
        }
        out[static_cast<size_t>(t_flat)] += g[static_cast<size_t>(flat)];
    }
    return out;
}

/**
 * @brief Broadcast a reduced cotangent back over the axes it was reduced along,
 *        scaled by @p scale.
 *
 * NOT broadcastTo(): that one right-aligns, which is the rule for elementwise
 * broadcasting and the wrong rule here. Reducing [4,6] along axis 1 leaves [4],
 * and putting it back means mapping it onto axis 0, not onto the trailing axis.
 * Right-aligning would map [4] onto the extent-6 axis and either fail or, for a
 * square tensor, silently transpose the gradient.
 */
std::vector<double> unreduceTo(const std::vector<double>& g,
                               const std::vector<int64_t>& in_shape,
                               const std::vector<int64_t>& res_shape,
                               const std::vector<int64_t>& axes,
                               double scale) {
    const int64_t n = numElements(in_shape);
    std::vector<double> out(static_cast<size_t>(n), 0.0);
    const auto in_st = stridesOf(in_shape);
    const auto res_st = stridesOf(res_shape);
    for (int64_t flat = 0; flat < n; ++flat) {
        int64_t rem = flat, o_flat = 0;
        size_t od = 0;
        for (size_t d = 0; d < in_shape.size(); ++d) {
            const int64_t idx = rem / in_st[d];
            rem %= in_st[d];
            bool reduced = axes.empty();
            for (int64_t ax : axes) if (ax == static_cast<int64_t>(d)) reduced = true;
            if (reduced) continue;
            o_flat += idx * res_st[od++];
        }
        out[static_cast<size_t>(flat)] = g[static_cast<size_t>(o_flat)] * scale;
    }
    return out;
}

/** @brief Permute @p src of @p shape by @p perm (result.dim(i) = src.dim(perm[i])). */
std::vector<double> transposeArray(const std::vector<double>& src,
                                   const std::vector<int64_t>& shape,
                                   const std::vector<int64_t>& perm) {
    std::vector<int64_t> out_shape;
    for (int64_t p : perm) out_shape.push_back(shape[static_cast<size_t>(p)]);
    const int64_t n = numElements(shape);
    std::vector<double> out(static_cast<size_t>(n), 0.0);
    const auto in_st = stridesOf(shape);
    const auto out_st = stridesOf(out_shape);
    for (int64_t flat = 0; flat < n; ++flat) {
        int64_t rem = flat;
        int64_t src_flat = 0;
        for (size_t d = 0; d < out_shape.size(); ++d) {
            const int64_t idx = rem / out_st[d];
            rem %= out_st[d];
            src_flat += idx * in_st[static_cast<size_t>(perm[d])];
        }
        out[static_cast<size_t>(flat)] = src[static_cast<size_t>(src_flat)];
    }
    return out;
}

/** @brief The integer op-code the elementwise C ABI uses for @p kind, or -1. */
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
        default:                     return -1;
    }
}

int reduceOpCode(DeviceOpKind kind) {
    switch (kind) {
        case DeviceOpKind::ReduceSum:  return 0;
        case DeviceOpKind::ReduceMean: return 1;
        case DeviceOpKind::ReduceMax:  return 2;
        case DeviceOpKind::ReduceMin:  return 3;
        case DeviceOpKind::ReduceProd: return 4;
        default:                       return -1;
    }
}

/**
 * @brief The HOST forward answer for one request at the given inputs.
 *
 * Exactly the entry points op_parity_test grades against, called here so the
 * finite difference differentiates the host's own forward code rather than a
 * second implementation of it.
 */
std::vector<double> hostForward(arena_t* arena, const DeviceOpRequest& req,
                                const std::vector<std::vector<double>>& inputs,
                                std::string* error) {
    const int64_t expected = numElements(req.result_shape);
    const auto& shapes = req.operand_shapes;

    if (elementwiseOpCode(req.kind) >= 0) {
        const int op = elementwiseOpCode(req.kind);
        const bool binary = op <= 3;
        std::vector<uint64_t> a_shape = asU64(shapes[0]);
        std::vector<uint64_t> b_shape = binary ? asU64(shapes[1]) : std::vector<uint64_t>{};
        void* t = eshkol_xla_elementwise_host(
            arena, inputs[0].data(), binary ? inputs[1].data() : nullptr,
            numElements(shapes[0]), a_shape.data(), static_cast<int64_t>(a_shape.size()),
            binary ? numElements(shapes[1]) : 0,
            binary ? b_shape.data() : nullptr,
            binary ? static_cast<int64_t>(b_shape.size()) : 0,
            op);
        if (!t) { *error = "host elementwise returned null"; return {}; }
        return tensorValues(t, expected);
    }
    if (reduceOpCode(req.kind) >= 0) {
        std::vector<uint64_t> shape = asU64(shapes[0]);
        const int64_t axis = req.axes.empty() ? -1 : req.axes[0];
        void* t = eshkol_xla_reduce_host(arena, inputs[0].data(), numElements(shapes[0]),
                                         shape.data(), static_cast<int64_t>(shape.size()),
                                         axis, reduceOpCode(req.kind));
        if (!t) { *error = "host reduce returned null"; return {}; }
        return tensorValues(t, expected);
    }
    switch (req.kind) {
        case DeviceOpKind::Matmul: {
            void* t = eshkol_xla_matmul_host(arena, inputs[0].data(), inputs[1].data(),
                                             shapes[0].data(), shapes[1].data(), 2, 2);
            if (!t) { *error = "host matmul returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Transpose: {
            std::vector<uint64_t> shape = asU64(shapes[0]);
            void* t = eshkol_xla_transpose_host(arena, inputs[0].data(), shape.data(),
                                                static_cast<int64_t>(shape.size()),
                                                req.axes.data());
            if (!t) { *error = "host transpose returned null"; return {}; }
            return tensorValues(t, expected);
        }
        case DeviceOpKind::Broadcast: {
            std::vector<uint64_t> src = asU64(shapes[0]);
            std::vector<uint64_t> tgt = asU64(req.result_shape);
            void* t = eshkol_xla_broadcast_host(arena, inputs[0].data(), src.data(),
                                                static_cast<int64_t>(src.size()),
                                                tgt.data(), static_cast<int64_t>(tgt.size()));
            if (!t) { *error = "host broadcast returned null"; return {}; }
            return tensorValues(t, expected);
        }
        default:
            *error = "no host forward entry point for this op";
            return {};
    }
}

/**
 * @brief Central-difference cotangent for one operand: (J^T g)_i estimated as
 *        sum_k g_k * (f_k(x + h e_i) - f_k(x - h e_i)) / 2h.
 *
 * O(n) forward evaluations of the HOST op, which for the shapes here is a few
 * dozen calls and costs nothing worth optimising.
 */
std::vector<double> fdCotangent(arena_t* arena, const DeviceOpRequest& req,
                                const std::vector<std::vector<double>>& inputs,
                                const std::vector<double>& cotangent,
                                size_t operand_index, std::string* error) {
    const size_t n = inputs[operand_index].size();
    std::vector<double> out(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::vector<double>> plus = inputs;
        std::vector<std::vector<double>> minus = inputs;
        plus[operand_index][i] += kFdStep;
        minus[operand_index][i] -= kFdStep;
        std::vector<double> fp = hostForward(arena, req, plus, error);
        std::vector<double> fm = hostForward(arena, req, minus, error);
        if (fp.empty() || fm.empty() || fp.size() != cotangent.size()) {
            if (error->empty()) *error = "finite difference could not evaluate the host forward";
            return {};
        }
        double acc = 0.0;
        for (size_t k = 0; k < fp.size(); ++k) {
            acc += cotangent[k] * (fp[k] - fm[k]) / (2.0 * kFdStep);
        }
        out[i] = acc;
    }
    return out;
}

/** @brief Where a row's reference cotangent comes from, printed per row. */
enum class RefSource {
    HostAd,       // an entry point in inc/eshkol/backend/tensor_backward.h
    StatedRule,   // the rule named in ref_note, cross-checked by finite difference
    Refusal       // the device must refuse this row
};

const char* refSourceName(RefSource s) {
    switch (s) {
        case RefSource::HostAd:     return "host-ad";
        case RefSource::StatedRule: return "rule+fd";
        case RefSource::Refusal:    return "refusal";
    }
    return "?";
}

/** @brief One gradient parity row. */
struct GradCase {
    const char* label;
    DeviceOpRequest request;                    // the FORWARD op
    std::vector<std::vector<double>> inputs;
    std::vector<double> cotangent;              // shaped request.result_shape
    ToleranceClass tolerance_class = ToleranceClass::Arithmetic;
    RefSource ref = RefSource::StatedRule;
    const char* ref_note = "";
    bool fd_admissible = true;
    const char* fd_note = "";
    const char* refusal_substring = "";         // required in the diagnostic, Refusal rows only
};

/**
 * @brief The host reference cotangents, one per operand.
 *
 * HostAd rows call inc/eshkol/backend/tensor_backward.h directly. StatedRule
 * rows apply the rule their ref_note names; every one of those that admits a
 * finite difference is checked against one before the row is graded.
 */
std::vector<std::vector<double>> hostCotangents(const GradCase& c, std::string* error) {
    const auto& shapes = c.request.operand_shapes;
    const auto& g = c.cotangent;
    const auto& res = c.request.result_shape;
    std::vector<std::vector<double>> out;

    auto bcast = [&](size_t idx) {
        return broadcastTo(c.inputs[idx], shapes[idx], res);
    };

    switch (c.request.kind) {
        case DeviceOpKind::Add: {
            out.push_back(unbroadcast(g, res, shapes[0]));
            out.push_back(unbroadcast(g, res, shapes[1]));
            return out;
        }
        case DeviceOpKind::Subtract: {
            std::vector<double> neg(g.size());
            for (size_t i = 0; i < g.size(); ++i) neg[i] = -g[i];
            out.push_back(unbroadcast(g, res, shapes[0]));
            out.push_back(unbroadcast(neg, res, shapes[1]));
            return out;
        }
        case DeviceOpKind::Multiply: {
            std::vector<double> a = bcast(0), b = bcast(1);
            std::vector<double> da(g.size()), db(g.size());
            for (size_t i = 0; i < g.size(); ++i) { da[i] = g[i] * b[i]; db[i] = g[i] * a[i]; }
            out.push_back(unbroadcast(da, res, shapes[0]));
            out.push_back(unbroadcast(db, res, shapes[1]));
            return out;
        }
        case DeviceOpKind::Divide: {
            std::vector<double> a = bcast(0), b = bcast(1);
            std::vector<double> da(g.size()), db(g.size());
            for (size_t i = 0; i < g.size(); ++i) {
                da[i] = g[i] / b[i];
                db[i] = -g[i] * a[i] / (b[i] * b[i]);
            }
            out.push_back(unbroadcast(da, res, shapes[0]));
            out.push_back(unbroadcast(db, res, shapes[1]));
            return out;
        }
        case DeviceOpKind::Exp:
        case DeviceOpKind::Log:
        case DeviceOpKind::Sin:
        case DeviceOpKind::Cos:
        case DeviceOpKind::Tanh: {
            const auto& x = c.inputs[0];
            std::vector<double> d(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                switch (c.request.kind) {
                    case DeviceOpKind::Exp:  d[i] = g[i] * std::exp(x[i]); break;
                    case DeviceOpKind::Log:  d[i] = g[i] / x[i]; break;
                    case DeviceOpKind::Sin:  d[i] = g[i] * std::cos(x[i]); break;
                    case DeviceOpKind::Cos:  d[i] = -g[i] * std::sin(x[i]); break;
                    default: {
                        const double t = std::tanh(x[i]);
                        d[i] = g[i] * (1.0 - t * t);
                        break;
                    }
                }
            }
            out.push_back(d);
            return out;
        }
        case DeviceOpKind::Matmul: {
            const int64_t M = shapes[0][0], K = shapes[0][1], N = shapes[1][1];
            std::vector<double> dA(static_cast<size_t>(M * K), 0.0);
            std::vector<double> dB(static_cast<size_t>(K * N), 0.0);
            eshkol_backward_matmul(g.data(), c.inputs[0].data(), c.inputs[1].data(),
                                   dA.data(), dB.data(), M, K, N);
            out.push_back(dA);
            out.push_back(dB);
            return out;
        }
        case DeviceOpKind::Transpose: {
            if (shapes[0].size() == 2) {
                // eshkol_backward_transpose is documented in terms of the
                // FORWARD input's (rows, cols): out is (rows, cols) and grad_in
                // is (cols, rows). Eshkol's rank-2 transpose is [r,c] -> [c,r],
                // so grad_out here is [c,r] and grad_in is [r,c]: the entry
                // point's `rows`/`cols` are the RESULT's, not the operand's.
                const int64_t rows = res[0], cols = res[1];
                std::vector<double> din(static_cast<size_t>(rows * cols), 0.0);
                eshkol_backward_transpose(g.data(), din.data(), rows, cols);
                out.push_back(din);
                return out;
            }
            // Rank > 2: the host entry point is rank-2 only, so the rule is
            // stated here — the VJP of a permutation is its INVERSE, which for
            // {2,0,1} is {1,2,0} and is not the same permutation.
            std::vector<int64_t> inv(c.request.axes.size(), 0);
            for (size_t i = 0; i < c.request.axes.size(); ++i) {
                inv[static_cast<size_t>(c.request.axes[i])] = static_cast<int64_t>(i);
            }
            out.push_back(transposeArray(g, res, inv));
            return out;
        }
        case DeviceOpKind::Reshape: {
            std::vector<double> din(g.size(), 0.0);
            eshkol_backward_reshape(g.data(), din.data(), static_cast<int64_t>(g.size()));
            out.push_back(din);
            return out;
        }
        case DeviceOpKind::Broadcast: {
            out.push_back(unbroadcast(g, res, shapes[0]));
            return out;
        }
        case DeviceOpKind::ReduceSum: {
            if (c.request.axes.empty()) {
                std::vector<double> din(static_cast<size_t>(numElements(shapes[0])), 0.0);
                eshkol_backward_sum(g[0], din.data(), numElements(shapes[0]));
                out.push_back(din);
                return out;
            }
            out.push_back(unreduceTo(g, shapes[0], res, c.request.axes, 1.0));
            return out;
        }
        case DeviceOpKind::ReduceMean: {
            if (c.request.axes.empty()) {
                std::vector<double> din(static_cast<size_t>(numElements(shapes[0])), 0.0);
                eshkol_backward_mean(g[0], din.data(), numElements(shapes[0]));
                out.push_back(din);
                return out;
            }
            const int64_t ax = c.request.axes[0];
            const double scale = 1.0 / static_cast<double>(shapes[0][static_cast<size_t>(ax)]);
            out.push_back(unreduceTo(g, shapes[0], res, c.request.axes, scale));
            return out;
        }
        case DeviceOpKind::ReduceMax:
        case DeviceOpKind::ReduceMin: {
            // The host's scalar rule, folded literally in row-major order:
            //   acc = max(acc, x_i);  on a strict win the gradient goes to the
            //   winner, and on a TIE it goes entirely to the second operand,
            //   which in this fold is x_i. So the last tied element takes all.
            // Only full reductions are built as rows here, so the fold runs
            // over the whole tensor and the winner is one flat index.
            const bool isMax = (c.request.kind == DeviceOpKind::ReduceMax);
            const auto& x = c.inputs[0];
            size_t winner = 0;
            double acc = x[0];
            for (size_t i = 1; i < x.size(); ++i) {
                const bool strict_first = isMax ? (acc > x[i]) : (acc < x[i]);
                if (!strict_first) { acc = x[i]; winner = i; }
            }
            std::vector<double> din(x.size(), 0.0);
            din[winner] = g[0];
            out.push_back(din);
            return out;
        }
        case DeviceOpKind::ReduceProd:
            *error = "reduce_prod is a refusal row and has no reference cotangent";
            return {};
    }
    *error = "unhandled op kind in hostCotangents";
    return {};
}

/** @brief Build the gradient parity table. */
std::vector<GradCase> buildCases() {
    std::vector<GradCase> cases;

    // A cotangent that is not all ones, so a rule that drops or duplicates it
    // is visible. Values are O(1) and non-repeating.
    auto cot = [](int64_t n) { return makeData(n, 0.75, 0.0625); };

    auto binary2d = [&](const char* label, DeviceOpKind kind, double b_base) {
        GradCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}, {4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(24, b_base, 0.125)};
        c.cotangent = cot(24);
        c.ref_note = "elementwise rule; no host tensor AD entry point in this tree";
        cases.push_back(c);
    };
    binary2d("d/dx add   f64[4,6],[4,6]", DeviceOpKind::Add, 1.25);
    binary2d("d/dx sub   f64[4,6],[4,6]", DeviceOpKind::Subtract, 1.25);
    binary2d("d/dx mul   f64[4,6],[4,6]", DeviceOpKind::Multiply, 1.25);
    binary2d("d/dx div   f64[4,6],[4,6]", DeviceOpKind::Divide, 1.25);

    // Broadcast rows. These are the ones a naive VJP gets shape-right and
    // value-wrong: the [6] operand's cotangent must be the [4,6] cotangent
    // SUMMED over axis 0, not a slice of it.
    {
        GradCase c;
        c.label = "d/dx add   f64[4,6],[6] (broadcast)";
        c.request.kind = DeviceOpKind::Add;
        c.request.operand_shapes = {{4, 6}, {6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(6, 2.0, 0.5)};
        c.cotangent = cot(24);
        c.ref_note = "un-broadcast: sum the cotangent over the axes the operand did not span";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx mul   f64[4,6],[6] (broadcast)";
        c.request.kind = DeviceOpKind::Multiply;
        c.request.operand_shapes = {{4, 6}, {6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(6, 1.5, 0.25)};
        c.cotangent = cot(24);
        c.ref_note = "un-broadcast of g*a; the [6] operand's gradient sums over axis 0";
        cases.push_back(c);
    }

    auto unary = [&](const char* label, DeviceOpKind kind, double base, double step,
                     const char* note) {
        GradCase c;
        c.label = label;
        c.request.kind = kind;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4, 6};
        c.inputs = {makeData(24, base, step)};
        c.cotangent = cot(24);
        c.tolerance_class = ToleranceClass::Transcendental;
        c.ref_note = note;
        cases.push_back(c);
    };
    unary("d/dx exp   f64[4,6]", DeviceOpKind::Exp, -2.0, 0.16, "g*exp(x)");
    unary("d/dx log   f64[4,6]", DeviceOpKind::Log, 0.25, 0.25, "g/x");
    unary("d/dx sin   f64[4,6]", DeviceOpKind::Sin, -1.5, 0.125, "g*cos(x)");
    unary("d/dx cos   f64[4,6]", DeviceOpKind::Cos, -1.5, 0.125, "-g*sin(x)");
    unary("d/dx tanh  f64[4,6]", DeviceOpKind::Tanh, -1.5, 0.125, "g*(1-tanh(x)^2)");

    // Matmul: non-square, so a transposed contraction in either gradient is
    // visible as a shape error rather than a plausible number.
    {
        GradCase c;
        c.label = "d/dx matmul f64[4,6]x[6,3]";
        c.request.kind = DeviceOpKind::Matmul;
        c.request.operand_shapes = {{4, 6}, {6, 3}};
        c.request.result_shape = {4, 3};
        c.inputs = {makeData(24, 0.5, 0.25), makeData(18, -1.0, 0.125)};
        c.cotangent = cot(12);
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_matmul";
        cases.push_back(c);
    }

    // A second matmul row whose operands are NOT exactly representable in
    // bf16. This is the row that measures the precision the device actually
    // computes a dot in, rather than the precision it was asked for: the row
    // above uses multiples of 0.25 and 0.0625, which survive a bf16 rounding
    // unchanged, so it measures 0 error whether the matrix unit rounded the
    // operands or not. These steps are not dyadic and do not.
    {
        GradCase c;
        c.label = "d/dx matmul f64[4,6]x[6,3] non-dyadic";
        c.request.kind = DeviceOpKind::Matmul;
        c.request.operand_shapes = {{4, 6}, {6, 3}};
        c.request.result_shape = {4, 3};
        c.inputs = {makeData(24, 0.31, 0.17), makeData(18, -0.83, 0.13)};
        c.cotangent = makeData(12, 0.73, 0.11);
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_matmul; operands are not bf16-exact, so this row "
                     "measures the precision the dot is actually computed in";
        cases.push_back(c);
    }

    {
        GradCase c;
        c.label = "d/dx transpose f64[4,6] {1,0}";
        c.request.kind = DeviceOpKind::Transpose;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {6, 4};
        c.request.axes = {1, 0};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = cot(24);
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_transpose";
        cases.push_back(c);
    }
    // Rank 3 with a permutation that is NOT its own inverse. The rank-2 case
    // cannot tell the VJP's inverse permutation from the forward one, because
    // {1,0} is self-inverse: a rule that reused the forward permutation would
    // pass it. {2,0,1} inverts to {1,2,0}, and only one of the two matches.
    {
        GradCase c;
        c.label = "d/dx transpose f64[2,3,4] {2,0,1}";
        c.request.kind = DeviceOpKind::Transpose;
        c.request.operand_shapes = {{2, 3, 4}};
        c.request.result_shape = {4, 2, 3};
        c.request.axes = {2, 0, 1};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = cot(24);
        c.ref_note = "inverse permutation {1,2,0}; eshkol_backward_transpose is rank-2 only";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reshape f64[4,6]->[3,8]";
        c.request.kind = DeviceOpKind::Reshape;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {3, 8};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = cot(24);
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_reshape";
        c.fd_admissible = false;
        c.fd_note = "no host forward entry point: reshape is a metadata change";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx broadcast f64[6]->[4,6]";
        c.request.kind = DeviceOpKind::Broadcast;
        c.request.operand_shapes = {{6}};
        c.request.result_shape = {4, 6};
        c.request.axes = {1};
        c.inputs = {makeData(6, 2.0, 0.5)};
        c.cotangent = cot(24);
        c.ref_note = "sum the cotangent over the broadcast axis";
        cases.push_back(c);
    }

    // Reductions. Full reductions use the host AD entry points; the axis
    // reductions have none and state their rule.
    {
        GradCase c;
        c.label = "d/dx reduce_sum f64[4,6] all";
        c.request.kind = DeviceOpKind::ReduceSum;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = {1.75};
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_sum";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reduce_sum f64[4,6] axis 1";
        c.request.kind = DeviceOpKind::ReduceSum;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {4};
        c.request.axes = {1};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = cot(4);
        c.ref_note = "broadcast the cotangent back over the reduced axis";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reduce_mean f64[4,6] all";
        c.request.kind = DeviceOpKind::ReduceMean;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = {1.75};
        c.ref = RefSource::HostAd;
        c.ref_note = "eshkol_backward_mean";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reduce_mean f64[4,6] axis 0";
        c.request.kind = DeviceOpKind::ReduceMean;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {6};
        c.request.axes = {0};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = cot(6);
        c.ref_note = "cotangent scaled by 1/extent, broadcast back over the reduced axis";
        cases.push_back(c);
    }
    // reduce_max / reduce_min with DISTINCT elements: no tie, so the gradient
    // is a genuine derivative and a central difference is admissible.
    {
        GradCase c;
        c.label = "d/dx reduce_max f64[4,6] all";
        c.request.kind = DeviceOpKind::ReduceMax;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = {1.75};
        c.ref_note = "cotangent to the extremum; inputs are distinct so there is no tie";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reduce_min f64[4,6] all";
        c.request.kind = DeviceOpKind::ReduceMin;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = {1.75};
        c.ref_note = "cotangent to the extremum; inputs are distinct so there is no tie";
        cases.push_back(c);
    }
    // The tie rows. See THE TIE CONVENTION at the top of this file: the whole
    // cotangent goes to the LAST tied element, which is what a fold of the
    // host's scalar max/min rule does. A central difference cannot grade these
    // — the function is not differentiable at a tie — so it is not run.
    {
        GradCase c;
        c.label = "d/dx reduce_max f64[3,4] TIED";
        c.request.kind = DeviceOpKind::ReduceMax;
        c.request.operand_shapes = {{3, 4}};
        c.request.result_shape = {};
        // Four elements tie at the maximum, at flat indices 2, 5, 10 and 11.
        c.inputs = {{0.5, 1.0, 3.0, 0.25,
                     2.0, 3.0, 1.5, 0.75,
                     1.25, 2.5, 3.0, 3.0}};
        c.cotangent = {1.75};
        c.ref_note = "host scalar-max fold: on a tie the gradient goes to the second operand, "
                     "so the last tied element (flat 11) takes all of it";
        c.fd_admissible = false;
        c.fd_note = "not differentiable at a tie; a central difference averages two "
                    "different one-sided derivatives and reports a confident wrong number";
        cases.push_back(c);
    }
    {
        GradCase c;
        c.label = "d/dx reduce_min f64[3,4] TIED";
        c.request.kind = DeviceOpKind::ReduceMin;
        c.request.operand_shapes = {{3, 4}};
        c.request.result_shape = {};
        c.inputs = {{0.5, 1.0, 0.25, 0.25,
                     2.0, 3.0, 1.5, 0.75,
                     1.25, 0.25, 3.0, 2.0}};
        c.cotangent = {1.75};
        c.ref_note = "host scalar-min fold: the last tied element (flat 9) takes all of it";
        c.fd_admissible = false;
        c.fd_note = "not differentiable at a tie";
        cases.push_back(c);
    }
    // Fail-closed row: reduce_prod has no VJP rule, deliberately. The device
    // must refuse and say why. A silent zero or a plausible number here is the
    // failure this row exists to catch.
    {
        GradCase c;
        c.label = "d/dx reduce_prod f64[4,6] all (must refuse)";
        c.request.kind = DeviceOpKind::ReduceProd;
        c.request.operand_shapes = {{4, 6}};
        c.request.result_shape = {};
        c.inputs = {makeData(24, 0.5, 0.25)};
        c.cotangent = {1.75};
        c.ref = RefSource::Refusal;
        c.ref_note = "no VJP rule: grad*out/input is undefined wherever the input has a zero";
        c.refusal_substring = "product";
        c.fd_admissible = false;
        c.fd_note = "refusal row";
        cases.push_back(c);
    }

    return cases;
}

/** @brief The cotangent from the public runtime entry point, as an Eshkol tensor. */
std::vector<double> viaPublicEntryPoint(arena_t* arena, const GradCase& c,
                                        size_t operand_index, std::string* error) {
    const auto& shapes = c.request.operand_shapes;
    std::vector<std::vector<uint64_t>> shape_storage;
    for (const auto& s : shapes) shape_storage.push_back(asU64(s));
    std::vector<const uint64_t*> shape_ptrs;
    std::vector<int64_t> ranks;
    std::vector<const double*> operand_ptrs;
    for (size_t i = 0; i < shapes.size(); ++i) {
        shape_ptrs.push_back(shape_storage[i].data());
        ranks.push_back(static_cast<int64_t>(shapes[i].size()));
        operand_ptrs.push_back(c.inputs[i].data());
    }
    std::vector<uint64_t> res = asU64(c.request.result_shape);

    void* t = eshkol_xla_gradient(
        arena, static_cast<int64_t>(c.request.kind),
        static_cast<int64_t>(shapes.size()),
        operand_ptrs.data(), shape_ptrs.data(), ranks.data(),
        c.cotangent.data(),
        res.empty() ? nullptr : res.data(), static_cast<int64_t>(res.size()),
        c.request.axes.empty() ? nullptr : c.request.axes.data(),
        static_cast<int64_t>(c.request.axes.size()),
        static_cast<int64_t>(operand_index));
    if (!t) {
        *error = "eshkol_xla_gradient returned null";
        return {};
    }
    return tensorValues(t, numElements(shapes[operand_index]));
}

void printHeader() {
    std::printf("\n%-38s %-6s %-15s %-9s %-12s %-12s %-9s %-11s %s\n",
                "gradient row", "dtype", "class", "reference", "max abs err",
                "max rel err", "tol", "fd rel diff", "result");
    std::printf("%-38s %-6s %-15s %-9s %-12s %-12s %-9s %-11s %s\n",
                "--------------------------------------", "------",
                "---------------", "---------", "------------", "------------",
                "---------", "-----------", "------");
}

// ─────────────────────────────────────────────────────────────────────────
// The composite and golden graphs. These are built here, with
// StableHLOEmitter directly, and executed through DeviceExecutor::runModule()
// so that the transfer, the dtype staging and the compile cache stay in
// device_lowering.cpp. A DeviceOpRequest cannot describe them: it names one
// op, and the point of these two is a backward pass SHARED across several.
// ─────────────────────────────────────────────────────────────────────────

ElementType deviceElementType() {
    return g_dtype == "f64" ? ElementType::F64 : ElementType::F32;
}

/** @brief Host matmul into a plain buffer (row-major [M,K] x [K,N]). */
std::vector<double> hostMatmul(arena_t* arena, const std::vector<double>& a,
                               const std::vector<double>& b,
                               int64_t M, int64_t K, int64_t N) {
    std::vector<int64_t> as = {M, K}, bs = {K, N};
    void* t = eshkol_xla_matmul_host(arena, a.data(), b.data(), as.data(), bs.data(), 2, 2);
    return tensorValues(t, M * N);
}

/**
 * @brief The first end-to-end reverse pass on silicon: a two-layer function
 *        matmul -> tanh -> matmul -> reduce_mean, differentiated with respect
 *        to both weight matrices.
 *
 * Every row above differentiates ONE op, which cannot expose the two things
 * that only appear in a real backward pass: a cotangent flowing through a
 * chain, and an intermediate (here the tanh output) whose value the backward
 * pass reuses rather than recomputing. This is the smallest graph that has
 * both, and it is the shape of a training step.
 */
bool runCompositeTwoLayer(DeviceExecutor* executor, arena_t* arena) {
    std::cout << "\nComposite: L = mean(tanh(X @ W1) @ W2), dL/dW1 and dL/dW2" << std::endl;

    const int64_t B = 2, D = 3, H = 4, O = 2;
    std::vector<int64_t> sX = {B, D}, sW1 = {D, H}, sW2 = {H, O};
    std::vector<double> X  = makeData(B * D, 0.25, 0.125);
    std::vector<double> W1 = makeData(D * H, -0.5, 0.0625);
    std::vector<double> W2 = makeData(H * O, 0.75, -0.125);

    // ---- device: one module, forward and backward together ----
    StableHLOEmitter emitter;
    if (!emitter.isAvailable()) {
        std::cout << "  FAIL: no StableHLO emitter" << std::endl;
        return false;
    }
    const ElementType elem = deviceElementType();
    std::vector<void*> args = emitter.beginFunction(
        "main", {{sX, elem}, {sW1, elem}, {sW2, elem}});
    if (args.size() != 3) {
        std::cout << "  FAIL: beginFunction did not open a 3-parameter function" << std::endl;
        return false;
    }
    eshkol::xla::DotDimensionNumbers dims;
    dims.lhs_contracting_dims = {1};
    dims.rhs_contracting_dims = {0};
    void* A = emitter.emitMatmul(args[0], args[1], dims);
    void* Hv = A ? emitter.emitTanh(A) : nullptr;
    void* Y = Hv ? emitter.emitMatmul(Hv, args[2], dims) : nullptr;
    // mean over both axes, built the way device_lowering.cpp builds it: a sum
    // divided by a sum of ones, so the whole mean is device arithmetic.
    void* sum = Y ? emitter.emitReduce(Y, {0, 1}, eshkol::xla::StableHLOOp::REDUCE_SUM) : nullptr;
    void* ones = Y ? emitter.emitOnesLike(Y) : nullptr;
    void* count = ones ? emitter.emitReduce(ones, {0, 1}, eshkol::xla::StableHLOOp::REDUCE_SUM)
                       : nullptr;
    void* L = (sum && count) ? emitter.emitDivide(sum, count) : nullptr;
    if (!L) {
        std::cout << "  FAIL: could not emit the forward graph" << std::endl;
        return false;
    }
    VJPResult vjp = emitter.emitVJP(L, {args[1], args[2]}, nullptr);
    if (!vjp.complete) {
        std::cout << "  FAIL: emitVJP refused the composite: " << vjp.diagnostic << std::endl;
        return false;
    }
    if (!emitter.endFunction({L, vjp.gradients[0], vjp.gradients[1]})) {
        std::cout << "  FAIL: endFunction failed" << std::endl;
        return false;
    }
    const std::string module_text = emitter.serializeToString();

    std::vector<double> dev_L(1, 0.0);
    std::vector<double> dev_dW1(static_cast<size_t>(D * H), 0.0);
    std::vector<double> dev_dW2(static_cast<size_t>(H * O), 0.0);
    std::string error;
    const std::vector<std::vector<int64_t>> in_shapes = {sX, sW1, sW2};
    const std::vector<const double*> in_ptrs = {X.data(), W1.data(), W2.data()};
    std::vector<std::vector<int64_t>> out_shapes;
    out_shapes.push_back(std::vector<int64_t>{});   // the scalar loss, rank 0
    out_shapes.push_back(sW1);
    out_shapes.push_back(sW2);
    const std::vector<double*> out_ptrs = {dev_L.data(), dev_dW1.data(), dev_dW2.data()};
    if (!executor->runModule(module_text, "composite-two-layer-2x3x4x2",
                             in_shapes, in_ptrs, out_shapes, out_ptrs, &error)) {
        std::cout << "  FAIL: device execution: " << error << std::endl;
        return false;
    }

    // ---- host: the same reverse pass, through the host AD entry points ----
    std::vector<double> hA = hostMatmul(arena, X, W1, B, D, H);
    if (hA.empty()) { std::cout << "  FAIL: host matmul 1" << std::endl; return false; }
    std::vector<double> hH(hA.size());
    for (size_t i = 0; i < hA.size(); ++i) hH[i] = std::tanh(hA[i]);
    std::vector<double> hY = hostMatmul(arena, hH, W2, B, H, O);
    if (hY.empty()) { std::cout << "  FAIL: host matmul 2" << std::endl; return false; }
    double hL = 0.0;
    for (double v : hY) hL += v;
    hL /= static_cast<double>(hY.size());

    std::vector<double> dY(hY.size(), 0.0);
    eshkol_backward_mean(1.0, dY.data(), static_cast<int64_t>(hY.size()));
    std::vector<double> dH(hH.size(), 0.0), host_dW2(static_cast<size_t>(H * O), 0.0);
    eshkol_backward_matmul(dY.data(), hH.data(), W2.data(), dH.data(), host_dW2.data(), B, H, O);
    // tanh's derivative is elementwise and has no host AD entry point; it is
    // (1 - tanh(x)^2) evaluated at the forward output, which is the value the
    // device's VJP reuses too.
    std::vector<double> dA(hA.size());
    for (size_t i = 0; i < hA.size(); ++i) dA[i] = dH[i] * (1.0 - hH[i] * hH[i]);
    std::vector<double> dX(static_cast<size_t>(B * D), 0.0), host_dW1(static_cast<size_t>(D * H), 0.0);
    eshkol_backward_matmul(dA.data(), X.data(), W1.data(), dX.data(), host_dW1.data(), B, D, H);

    // ---- independent witness: central difference of the scalar loss ----
    auto lossAt = [&](const std::vector<double>& w1, const std::vector<double>& w2) {
        std::vector<double> a = hostMatmul(arena, X, w1, B, D, H);
        std::vector<double> h(a.size());
        for (size_t i = 0; i < a.size(); ++i) h[i] = std::tanh(a[i]);
        std::vector<double> y = hostMatmul(arena, h, w2, B, H, O);
        double s = 0.0;
        for (double v : y) s += v;
        return s / static_cast<double>(y.size());
    };
    std::vector<double> fd_dW1(W1.size()), fd_dW2(W2.size());
    for (size_t i = 0; i < W1.size(); ++i) {
        std::vector<double> p = W1, m = W1;
        p[i] += kFdStep; m[i] -= kFdStep;
        fd_dW1[i] = (lossAt(p, W2) - lossAt(m, W2)) / (2.0 * kFdStep);
    }
    for (size_t i = 0; i < W2.size(); ++i) {
        std::vector<double> p = W2, m = W2;
        p[i] += kFdStep; m[i] -= kFdStep;
        fd_dW2[i] = (lossAt(W1, p) - lossAt(W1, m)) / (2.0 * kFdStep);
    }

    // The composite's tolerance class is transcendental: tanh is on the path.
    const double tol = toleranceFor(ToleranceClass::Transcendental);
    Comparison cL = compareArrays(dev_L, {hL}, tol);
    Comparison c1 = compareArrays(dev_dW1, host_dW1, tol);
    Comparison c2 = compareArrays(dev_dW2, host_dW2, tol);
    Comparison f1 = compareArrays(host_dW1, fd_dW1, kFdTolerance);
    Comparison f2 = compareArrays(host_dW2, fd_dW2, kFdTolerance);

    std::printf("  %-28s max abs %-11.3e max rel %-11.3e tol %-9.1e %s\n",
                "forward loss (scalar)", cL.max_abs, cL.max_rel, tol,
                cL.agreed ? "PASS" : "FAIL");
    std::printf("  %-28s max abs %-11.3e max rel %-11.3e tol %-9.1e %s\n",
                "dL/dW1 [3,4] vs host tape", c1.max_abs, c1.max_rel, tol,
                c1.agreed ? "PASS" : "FAIL");
    std::printf("  %-28s max abs %-11.3e max rel %-11.3e tol %-9.1e %s\n",
                "dL/dW2 [4,2] vs host tape", c2.max_abs, c2.max_rel, tol,
                c2.agreed ? "PASS" : "FAIL");
    std::printf("  %-28s max rel %-11.3e (h=%g) %-24s %s\n",
                "host tape vs central FD", std::fmax(f1.max_rel, f2.max_rel), kFdStep, "",
                (f1.agreed && f2.agreed) ? "PASS" : "FAIL");

    const bool ok = cL.agreed && c1.agreed && c2.agreed && f1.agreed && f2.agreed;
    std::cout << "  COMPOSITE: " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

// ─────────────────────────────────────────────────────────────────────────
// Golden vectors: sphere_project.
//
// out = grad - <grad, x> * x, from tests/qllm_oracle/golden/sphere_project.json
// (commit 83c87895 on feat/qllm-gradient-oracle), whose Jacobians are exact
// reverse-mode AD over Eshkol's own tape in f64 and were cross-checked there
// against a central difference at 2.7e-15 relative.
//
// WHY THIS ONE. The golden corpus covers qLLM's geometric primitives, and most
// of them need sqrt, artanh or a clamp — none of which is lowered yet, so they
// cannot run on the device at all. sphere_project is the one that decomposes
// ENTIRELY into ops lowered in S2: multiply, a full reduce_sum, a rank-0
// broadcast, another multiply and a subtract. So it is a device gradient
// graded against an exact reference computed by something other than this
// file, which is what a golden vector is for.
//
// Seeding the VJP with e_k reads row k of the Jacobian: (J^T e_k)_j = J[k][j].
// ─────────────────────────────────────────────────────────────────────────

struct SphereGolden {
    int dim;
    std::vector<double> x;
    std::vector<double> grad;
    std::vector<std::vector<double>> d_out_d_x;     // [row][col]
    std::vector<std::vector<double>> d_out_d_grad;
};

std::vector<SphereGolden> sphereGoldenCases() {
    return {
        {2,
         {0.59999999999999998, -0.80000000000000004},
         {0.40000000000000002, -0.59999999999999998},
         {{-0.95999999999999996, 0.35999999999999999},
          {0.32000000000000006, -1.2}},
         {{0.64000000000000001, 0.47999999999999998},
          {0.47999999999999998, 0.35999999999999988}}},
        {4,
         {0.5, 0.5, -0.5, 0.5},
         {0.40000000000000002, -0.59999999999999998, 0.25, 0.125},
         {{-0.037500000000000033, 0.29999999999999999, -0.125, -0.0625},
          {-0.20000000000000001, 0.46249999999999997, -0.125, -0.0625},
          {0.20000000000000001, -0.29999999999999999, 0.28749999999999998, 0.0625},
          {-0.20000000000000001, 0.29999999999999999, -0.125, 0.099999999999999978}},
         {{0.75, -0.25, 0.25, -0.25},
          {-0.25, 0.75, 0.25, -0.25},
          {0.25, 0.25, 0.75, 0.25},
          {-0.25, -0.25, 0.25, 0.75}}},
    };
}

bool runSphereProjectGolden(DeviceExecutor* executor) {
    std::cout << "\nGolden: sphere_project, out = grad - <grad,x>*x "
                 "(tests/qllm_oracle/golden/sphere_project.json, commit 83c87895)"
              << std::endl;
    const ElementType elem = deviceElementType();
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    bool all_ok = true;

    for (const SphereGolden& g : sphereGoldenCases()) {
        const int64_t d = g.dim;
        std::vector<int64_t> sd = {d};

        StableHLOEmitter emitter;
        if (!emitter.isAvailable()) { std::cout << "  FAIL: no emitter" << std::endl; return false; }
        std::vector<void*> args = emitter.beginFunction(
            "main", {{sd, elem}, {sd, elem}, {sd, elem}});   // x, grad, cotangent
        if (args.size() != 3) {
            std::cout << "  FAIL: beginFunction" << std::endl;
            return false;
        }
        void* p = emitter.emitMultiply(args[1], args[0]);                       // grad * x
        void* s = p ? emitter.emitReduce(p, {0}, eshkol::xla::StableHLOOp::REDUCE_SUM) : nullptr;
        void* sb = s ? emitter.emitBroadcastInDim(s, {d}, {}) : nullptr;        // splat
        void* t = sb ? emitter.emitMultiply(sb, args[0]) : nullptr;             // <g,x> * x
        void* out = t ? emitter.emitSubtract(args[1], t) : nullptr;
        if (!out) { std::cout << "  FAIL: could not emit sphere_project" << std::endl; return false; }
        VJPResult vjp = emitter.emitVJP(out, {args[0], args[1]}, args[2]);
        if (!vjp.complete) {
            std::cout << "  FAIL: emitVJP refused sphere_project: " << vjp.diagnostic << std::endl;
            return false;
        }
        if (!emitter.endFunction({vjp.gradients[0], vjp.gradients[1]})) {
            std::cout << "  FAIL: endFunction" << std::endl;
            return false;
        }
        const std::string module_text = emitter.serializeToString();
        const std::string key = "sphere-project-d" + std::to_string(d);

        for (int64_t k = 0; k < d; ++k) {
            std::vector<double> seed(static_cast<size_t>(d), 0.0);
            seed[static_cast<size_t>(k)] = 1.0;
            std::vector<double> dx(static_cast<size_t>(d), 0.0);
            std::vector<double> dg(static_cast<size_t>(d), 0.0);
            std::string error;
            if (!executor->runModule(module_text, key, {sd, sd, sd},
                                     {g.x.data(), g.grad.data(), seed.data()},
                                     {sd, sd}, {dx.data(), dg.data()}, &error)) {
                std::printf("  d=%lld row %lld: FAIL (device: %s)\n",
                            static_cast<long long>(d), static_cast<long long>(k), error.c_str());
                all_ok = false;
                continue;
            }
            Comparison cx = compareArrays(dx, g.d_out_d_x[static_cast<size_t>(k)], tol);
            Comparison cg = compareArrays(dg, g.d_out_d_grad[static_cast<size_t>(k)], tol);
            const bool ok = cx.agreed && cg.agreed;
            std::printf("  d=%lld J row %lld: d/dx max rel %-11.3e  d/dgrad max rel %-11.3e  %s\n",
                        static_cast<long long>(d), static_cast<long long>(k),
                        cx.max_rel, cg.max_rel, ok ? "PASS" : "FAIL");
            if (!ok) all_ok = false;
        }
    }
    std::cout << "  GOLDEN: " << (all_ok ? "PASS" : "FAIL") << std::endl;
    return all_ok;
}

/**
 * @brief Negative control: a perturbed expected cotangent must be rejected.
 *
 * Distinct from the comparator control in parity_compare.h, which grades the
 * comparator in isolation. This one perturbs a REAL device gradient's expected
 * value and requires the row's own grading path to reject it, so that a row
 * reported PASS is a row whose numbers were actually compared.
 */
bool control_perturbed_cotangent_is_rejected(DeviceExecutor* executor) {
    std::cout << "Control: a perturbed expected cotangent is rejected... ";
    GradCase c;
    c.label = "control";
    c.request.kind = DeviceOpKind::Multiply;
    c.request.operand_shapes = {{2, 3}, {2, 3}};
    c.request.result_shape = {2, 3};
    c.inputs = {makeData(6, 0.5, 0.25), makeData(6, 1.25, 0.125)};
    c.cotangent = makeData(6, 0.75, 0.0625);

    std::string error;
    std::vector<std::vector<double>> host = hostCotangents(c, &error);
    if (host.size() != 2) {
        std::cout << "FAIL (no host reference: " << error << ")" << std::endl;
        return false;
    }
    std::vector<double> d0(host[0].size(), 0.0), d1(host[1].size(), 0.0);
    if (!executor->runGradient(c.request, {c.inputs[0].data(), c.inputs[1].data()},
                               c.cotangent.data(), {d0.data(), d1.data()}, &error)) {
        std::cout << "FAIL (device refused the control row: " << error << ")" << std::endl;
        return false;
    }
    const double tol = toleranceFor(ToleranceClass::Arithmetic);
    Comparison good = compareArrays(d0, host[0], tol);
    std::vector<double> perturbed = host[0];
    perturbed[0] += 1.0;
    Comparison bad = compareArrays(d0, perturbed, tol);
    if (!good.agreed) {
        std::cout << "FAIL (the device gradient did not match its own reference)" << std::endl;
        return false;
    }
    if (bad.agreed) {
        std::cout << "FAIL (a cotangent off by 1.0 was accepted)" << std::endl;
        return false;
    }
    std::cout << "PASS (device gradient accepted, off-by-1.0 reference rejected)" << std::endl;
    return true;
}

}  // namespace

int main() {
    std::cout << "=============================================" << std::endl;
    std::cout << "  XLA Device Gradient Parity (device vs host)" << std::endl;
    std::cout << "=============================================" << std::endl;

    // Ask for device execution before anything latches the answer, exactly as
    // op_parity_test does: deviceExecutionRequested() reads the variable once,
    // on first use, and nothing above has used it yet.
    ::setenv("ESHKOL_XLA_PJRT", "1", 1);

    if (!test_comparator_rejects_a_perturbed_result()) {
        std::cerr << "The comparator control failed; no gradient row below would mean anything."
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
    eshkol_parity::setTolerancesForDtype(g_dtype);

    std::cout << "Device: " << executor->description() << std::endl;
    std::cout << "Tolerance (" << g_dtype << ", absolute or relative, whichever is looser; "
                 "docs/design/ESHKOL_S_FRAGMENT.md): arithmetic="
              << eshkol_parity::g_tol_arithmetic
              << " transcendental=" << eshkol_parity::g_tol_transcendental << std::endl;
    std::cout << "Finite-difference cross-check: central difference of the host forward at h="
              << kFdStep << ", required within " << kFdTolerance << std::endl;
    std::cout << "Reduce max/min tie convention: the whole cotangent goes to the LAST tied "
                 "element, matching a fold of Eshkol's scalar AD_NODE_MAX/MIN rule "
                 "(lib/backend/autodiff_codegen.cpp)." << std::endl;

    const char* force_fail = std::getenv("ESHKOL_XLA_GRADIENT_FORCE_FAIL");
    if (force_fail && force_fail[0]) {
        std::cout << "FORCED FAIL requested for op '" << force_fail
                  << "': its host reference will be perturbed by 1.0." << std::endl;
    }

    arena_t* arena = arena_create(16 * 1024 * 1024);
    if (!arena) {
        std::cerr << "FAIL: could not create an arena" << std::endl;
        return 1;
    }

    if (!control_perturbed_cotangent_is_rejected(executor)) {
        g_controls_failed++;
        arena_destroy(arena);
        std::cerr << "The perturbed-cotangent control failed; the rows below are not graded."
                  << std::endl;
        return 1;
    }

    printHeader();
    for (const GradCase& c : buildCases()) {
        const char* cls = toleranceClassName(c.tolerance_class);
        const double tol = toleranceFor(c.tolerance_class);
        const size_t n_operands = c.request.operand_shapes.size();

        std::vector<double*> grad_ptrs;
        std::vector<std::vector<double>> device(n_operands);
        std::vector<const double*> operand_ptrs;
        for (size_t i = 0; i < n_operands; ++i) {
            device[i].assign(static_cast<size_t>(numElements(c.request.operand_shapes[i])), 0.0);
            grad_ptrs.push_back(device[i].data());
            operand_ptrs.push_back(c.inputs[i].data());
        }

        std::string device_error;
        const bool device_ok = executor->runGradient(c.request, operand_ptrs,
                                                     c.cotangent.data(), grad_ptrs,
                                                     &device_error);

        // ---- refusal rows: the device MUST decline, with a reason ----
        if (c.ref == RefSource::Refusal) {
            bool ok = !device_ok;
            std::string note = device_error;
            if (ok && c.refusal_substring[0] &&
                device_error.find(c.refusal_substring) == std::string::npos) {
                ok = false;
                note = "refused, but the diagnostic does not mention '" +
                       std::string(c.refusal_substring) + "': " + device_error;
            }
            if (!ok && device_ok) note = "the device returned a gradient for an op with no VJP rule";
            std::printf("%-38s %-6s %-15s %-9s %-12s %-12s %-9s %-11s %s\n",
                        c.label, g_dtype.c_str(), cls, refSourceName(c.ref),
                        "-", "-", "-", "-", ok ? "PASS (refused)" : "FAIL");
            if (!ok) {
                std::printf("       %s\n", note.c_str());
                g_rows_failed++;
            } else {
                g_rows_passed++;
            }
            continue;
        }

        if (!device_ok) {
            std::printf("%-38s %-6s %-15s %-9s %-12s %-12s %-9s %-11s FAIL (device: %s)\n",
                        c.label, g_dtype.c_str(), cls, refSourceName(c.ref),
                        "-", "-", "-", "-", device_error.c_str());
            g_rows_failed++;
            continue;
        }

        std::string ref_error;
        std::vector<std::vector<double>> host = hostCotangents(c, &ref_error);
        if (host.size() != n_operands) {
            std::printf("%-38s %-6s %-15s %-9s %-12s %-12s %-9s %-11s FAIL (host: %s)\n",
                        c.label, g_dtype.c_str(), cls, refSourceName(c.ref),
                        "-", "-", "-", "-", ref_error.c_str());
            g_rows_failed++;
            continue;
        }
        if (force_fail && std::strcmp(force_fail, deviceOpKindName(c.request.kind)) == 0) {
            host[0][0] += 1.0;
        }

        // ---- finite-difference cross-check of the host reference ----
        double fd_worst = 0.0;
        bool fd_ok = true;
        std::string fd_error;
        if (c.fd_admissible) {
            for (size_t i = 0; i < n_operands && fd_ok; ++i) {
                std::vector<double> fd = fdCotangent(arena, c.request, c.inputs,
                                                     c.cotangent, i, &fd_error);
                if (fd.size() != host[i].size()) {
                    fd_ok = false;
                    if (fd_error.empty()) fd_error = "finite difference produced the wrong length";
                    break;
                }
                Comparison f = compareArrays(host[i], fd, kFdTolerance);
                fd_worst = std::fmax(fd_worst, f.max_rel);
                if (!f.agreed) {
                    fd_ok = false;
                    fd_error = "host reference disagrees with the central difference";
                }
            }
        }

        // ---- device vs host, per operand ----
        Comparison worst;
        worst.agreed = true;
        for (size_t i = 0; i < n_operands; ++i) {
            Comparison cmp = compareArrays(device[i], host[i], tol);
            worst.max_abs = std::fmax(worst.max_abs, cmp.max_abs);
            worst.max_rel = std::fmax(worst.max_rel, cmp.max_rel);
            if (!cmp.agreed) {
                worst.agreed = false;
                if (worst.worst_index < 0) worst.worst_index = cmp.worst_index;
            }
        }

        // ---- the public runtime entry point must reach the same answer ----
        std::string public_error;
        bool wiring_ok = true;
        for (size_t i = 0; i < n_operands && wiring_ok; ++i) {
            std::vector<double> pub = viaPublicEntryPoint(arena, c, i, &public_error);
            if (pub.size() != device[i].size()) {
                wiring_ok = false;
                if (public_error.empty()) public_error = "wrong element count from eshkol_xla_gradient";
                break;
            }
            Comparison w = compareArrays(pub, device[i], tol);
            if (!w.agreed) {
                wiring_ok = false;
                public_error = "eshkol_xla_gradient disagreed with the device result";
            }
        }

        const bool row_ok = worst.agreed && fd_ok && wiring_ok;
        char fd_text[32];
        if (c.fd_admissible) {
            std::snprintf(fd_text, sizeof fd_text, "%.3e", fd_worst);
        } else {
            std::snprintf(fd_text, sizeof fd_text, "n/a");
        }
        std::printf("%-38s %-6s %-15s %-9s %-12.3e %-12.3e %-9.1e %-11s %s\n",
                    c.label, g_dtype.c_str(), cls, refSourceName(c.ref),
                    worst.max_abs, worst.max_rel, tol, fd_text,
                    row_ok ? "PASS" : "FAIL");
        std::printf("       reference: %s%s%s\n", c.ref_note,
                    c.fd_admissible ? "" : "; no FD: ",
                    c.fd_admissible ? "" : c.fd_note);
        if (!row_ok) {
            g_rows_failed++;
            if (!worst.agreed) {
                std::printf("       device disagreed with the reference at index %d\n",
                            worst.worst_index);
            }
            if (!fd_ok) std::printf("       finite difference: %s\n", fd_error.c_str());
            if (!wiring_ok) std::printf("       wiring: %s\n", public_error.c_str());
        } else {
            g_rows_passed++;
        }
    }

    const bool composite_ok = runCompositeTwoLayer(executor, arena);
    if (!composite_ok) g_rows_failed++; else g_rows_passed++;

    const bool golden_ok = runSphereProjectGolden(executor);
    if (!golden_ok) g_rows_failed++; else g_rows_passed++;

    arena_destroy(arena);

    const eshkol::xla::DeviceStats stats = executor->stats();
    std::cout << std::endl;
    std::cout << "Rows passed: " << g_rows_passed << std::endl;
    std::cout << "Rows failed: " << g_rows_failed << std::endl;
    std::cout << "Executable cache: " << stats.compiled << " compiled, "
              << stats.cache_hits << " reused, " << stats.executed << " executions, "
              << stats.failures << " failures" << std::endl;
    std::cout << "SUMMARY: rows_passed=" << g_rows_passed
              << " rows_failed=" << g_rows_failed
              << " dtype=" << g_dtype
              << " tol_arithmetic=" << eshkol_parity::g_tol_arithmetic
              << " tol_transcendental=" << eshkol_parity::g_tol_transcendental
              << " fd_step=" << kFdStep
              << " composite=" << (composite_ok ? "PASS" : "FAIL")
              << " golden_sphere_project=" << (golden_ok ? "PASS" : "FAIL")
              << " compiled=" << stats.compiled
              << " cache_hits=" << stats.cache_hits << std::endl;

    if (g_controls_failed != 0 || g_rows_failed != 0) return 1;
    return 0;
}
