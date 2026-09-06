/*
 * tests/xla/region_host_reference.h — the host answer for an outlined region.
 *
 * It walks the SAME subtree the device emitter walks, but every node is
 * computed by the host runtime's own *_host entry point: BLAS, SIMD, the code
 * every Eshkol program has always run. The device side is one fused StableHLO
 * module. Two independent numeric paths over one expression is what makes the
 * comparison mean something; evaluating the same formula twice in double
 * precision would agree however wrong both were.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_TESTS_XLA_REGION_HOST_REFERENCE_H
#define ESHKOL_TESTS_XLA_REGION_HOST_REFERENCE_H

#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/backend/xla/region_formation.h"

namespace eshkol_region_host {

using eshkol::xla::DeviceOpKind;
using eshkol::xla::RegionFunction;

struct HostVal {
    std::vector<double> data;
    std::vector<int64_t> shape;
    bool ok = false;
};

inline int64_t elements(const std::vector<int64_t>& s) {
    int64_t n = 1;
    for (int64_t d : s) n *= d;
    return n;
}

inline std::vector<double> tensorElements(void* tensor_ptr, int64_t expected) {
    std::vector<double> out;
    if (!tensor_ptr) return out;
    auto* t = static_cast<eshkol_tensor_t*>(tensor_ptr);
    if (static_cast<int64_t>(t->total_elements) != expected) return out;
    out.resize(static_cast<size_t>(expected));
    const double* src = reinterpret_cast<const double*>(t->elements);
    for (int64_t i = 0; i < expected; ++i) out[static_cast<size_t>(i)] = src[i];
    return out;
}

/** @brief The elementwise ABI op code for a kind, or -1. The numbering is the
 *         one xla_runtime.cpp documents; asked for here rather than restated
 *         as a table so a new code cannot be silently mis-numbered. */
inline int elementwiseCode(DeviceOpKind k) {
    switch (k) {
        case DeviceOpKind::Add: return 0;
        case DeviceOpKind::Subtract: return 1;
        case DeviceOpKind::Multiply: return 2;
        case DeviceOpKind::Divide: return 3;
        case DeviceOpKind::Exp: return 4;
        case DeviceOpKind::Log: return 5;
        case DeviceOpKind::Sin: return 6;
        case DeviceOpKind::Cos: return 7;
        case DeviceOpKind::Tanh: return 8;
        case DeviceOpKind::Relu: return 9;
        case DeviceOpKind::Sigmoid: return 10;
        case DeviceOpKind::Sqrt: return 11;
        case DeviceOpKind::Rsqrt: return 12;
        case DeviceOpKind::Abs: return 13;
        case DeviceOpKind::Negate: return 14;
        case DeviceOpKind::Atanh: return 15;
        case DeviceOpKind::Pow: return 16;
        case DeviceOpKind::Maximum: return 17;
        case DeviceOpKind::Minimum: return 18;
        default: return -1;
    }
}

/** @brief eshkol_xla_compare_host's direction code, in DeviceOpKind order. */
inline int64_t compareDirection(DeviceOpKind k) {
    switch (k) {
        case DeviceOpKind::CompareEq: return 0;
        case DeviceOpKind::CompareNe: return 1;
        case DeviceOpKind::CompareLt: return 2;
        case DeviceOpKind::CompareLe: return 3;
        case DeviceOpKind::CompareGt: return 4;
        case DeviceOpKind::CompareGe: return 5;
        default: return -1;
    }
}

inline int reduceCode(DeviceOpKind k) {
    switch (k) {
        case DeviceOpKind::ReduceSum: return 0;
        case DeviceOpKind::ReduceMax: return 1;
        case DeviceOpKind::ReduceMin: return 2;
        case DeviceOpKind::ReduceMean: return 3;
        default: return -1;
    }
}

inline const char* calleeOf(const eshkol_ast_t* n) {
    if (!n) return nullptr;
    return n->type == ESHKOL_VAR ? n->variable.id : nullptr;
}
inline const char* bindNameOf(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS && b->cons_cell.car &&
        b->cons_cell.car->type == ESHKOL_VAR) return b->cons_cell.car->variable.id;
    if (b->type == ESHKOL_VAR) return b->variable.id;
    return nullptr;
}
inline const eshkol_ast_t* bindValueOf(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS) return b->cons_cell.cdr;
    if (b->type == ESHKOL_VAR) return b->variable.data;
    return b;
}
inline bool literalOf(const eshkol_ast_t* n, double* v) {
    if (!n) return false;
    switch (n->type) {
        case ESHKOL_DOUBLE: *v = n->double_val; return true;
        case ESHKOL_INT64:  *v = static_cast<double>(n->int64_val); return true;
        case ESHKOL_INT32:  *v = static_cast<double>(n->int32_val); return true;
        default: return false;
    }
}

class Evaluator {
public:
    Evaluator(void* arena, const std::map<std::string, RegionFunction>& fns)
        : arena_(arena), fns_(fns) { scopes_.emplace_back(); }

    void bind(const std::string& name, HostVal v) { scopes_.front()[name] = std::move(v); }

    HostVal eval(const eshkol_ast_t* node, const std::vector<int64_t>* like,
                 std::string* error);

private:
    HostVal call(const char* name, const eshkol_ast_t* argv, uint64_t argc,
                 std::string* error);
    HostVal conditional(const eshkol_ast_t* pred, const eshkol_ast_t* then_arm,
                        const eshkol_ast_t* else_arm, std::string* error);
    HostVal loop(const eshkol_operations_t& let, std::string* error);

    void* arena_;
    const std::map<std::string, RegionFunction>& fns_;
    std::vector<std::map<std::string, HostVal>> scopes_;
};

} // namespace eshkol_region_host

#endif // ESHKOL_TESTS_XLA_REGION_HOST_REFERENCE_H
