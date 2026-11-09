/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * The exact tensor element type: kernels over tagged exact slots (ADR-0055).
 *
 * An exact tensor (dtype ESHKOL_TENSOR_DTYPE_EXACT) stores each element as a
 * 16-byte tagged exact rational of the scalar tower -- a fixnum, a bignum or a
 * rational -- in the same slot layout as a jet tensor. The elementwise,
 * matmul and sum kernels for it are the generic tagged kernels in
 * runtime_tensor_math.cpp, which apply the language's own scalar operators
 * slot by slot. This file holds the kernels that need an algorithm of their
 * own over the exact ring:
 *
 *   - transpose, which moves tagged slots whole;
 *   - determinant, by Bareiss fraction-free elimination;
 *   - solve and inverse, by fraction-free elimination of the augmented
 *     system [A | B] followed by exact back substitution.
 *
 * Contagion (R7RS 6.2.2) decides the representation of every result from the
 * operand dtypes: when every operand is exact the result is exact; when an
 * exact operand meets an inexact one, the computation is the f64 kernel's,
 * on the projections of both operands, and the result is inexact. Nothing is
 * rounded without an inexact operand asking for it, and no f64 kernel ever
 * reads an exact tensor's slots: the f64 operand checks refuse it by name.
 */

#include "arena_memory.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

extern "C" {

void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);
void eshkol_enforce_tensor_elements(int64_t num_elements);
void* eshkol_tagged_as_tensor(const eshkol_tagged_value_t* value);
void* eshkol_tensor_from_collection(arena_t* arena, const eshkol_tagged_value_t* input);
int32_t eshkol_tensor_real_to_double(const eshkol_tagged_value_t* value, double* out);
void eshkol_taylor_binary_tagged(arena_t* arena, const eshkol_tagged_value_t* left,
                                 const eshkol_tagged_value_t* right, int op,
                                 eshkol_tagged_value_t* result);
int eshkol_taylor_binary_opcode(const char* spelling);
void eshkol_rational_compare_tagged_ptr(void* arena, const eshkol_tagged_value_t* a,
                                        const eshkol_tagged_value_t* b, int op,
                                        eshkol_tagged_value_t* result);
int64_t eshkol_lu_decompose(double* A, int64_t* piv, int64_t n);
double eshkol_det_from_lu(const double* LU, int64_t n, int64_t sign);
void eshkol_lu_solve(const double* LU, const int64_t* piv, double* b, int64_t n);

}  // extern "C"

namespace {

using tagged = eshkol_tagged_value_t;

// ── The exact ring, through the language's own scalar operators ─────────────

struct Ring {
    arena_t* arena;
    int add, sub, mul, div;
    explicit Ring(arena_t* a)
        : arena(a),
          add(eshkol_taylor_binary_opcode("+")),
          sub(eshkol_taylor_binary_opcode("-")),
          mul(eshkol_taylor_binary_opcode("*")),
          div(eshkol_taylor_binary_opcode("/")) {}
    tagged op(const tagged& x, const tagged& y, int code) const {
        tagged r{};
        eshkol_taylor_binary_tagged(arena, &x, &y, code, &r);
        return r;
    }
    tagged plus(const tagged& x, const tagged& y) const { return op(x, y, add); }
    tagged minus(const tagged& x, const tagged& y) const { return op(x, y, sub); }
    tagged times(const tagged& x, const tagged& y) const { return op(x, y, mul); }
    tagged over(const tagged& x, const tagged& y) const { return op(x, y, div); }
    bool is_zero(const tagged& x) const {
        const tagged zero = eshkol_make_int64(0, true);
        tagged r{};
        eshkol_rational_compare_tagged_ptr(arena, &x, &zero, /*eq=*/2, &r);
        return r.data.int_val != 0;
    }
    static tagged integer(int64_t v) { return eshkol_make_int64(v, true); }
};

const char* name_or(const char* name, const char* fallback) {
    return name ? name : fallback;
}

// ── Exact tensor allocation ─────────────────────────────────────────────────

eshkol_tensor_t* exact_tensor_alloc(arena_t* arena, uint64_t ndim, const uint64_t* dims,
                                    uint64_t total, const char* name) {
    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, ndim, 0);
    if (!t) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: out of memory", name);
        return nullptr;
    }
    for (uint64_t d = 0; d < ndim; ++d) t->dimensions[d] = dims[d];
    eshkol_enforce_tensor_elements((int64_t)total);
    t->elements = static_cast<int64_t*>(arena_allocate(
        arena, (size_t)(total ? total : 1) * sizeof(tagged)));
    if (!t->elements) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: out of memory", name);
        return nullptr;
    }
    t->total_elements = total;
    t->dtype = ESHKOL_TENSOR_DTYPE_EXACT;
    return t;
}

const tagged* exact_slots(const eshkol_tensor_t* t) {
    return reinterpret_cast<const tagged*>(t->elements);
}

// ── Operands ────────────────────────────────────────────────────────────────

// A linear-algebra operand: an exact tensor, or (for contagion) any numeric
// tensor or numeric collection, which converts as `(tensor X)` converts it.
const eshkol_tensor_t* operand_tensor(arena_t* arena, const tagged* value, const char* name) {
    if (!value) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: missing operand", name);
        return nullptr;
    }
    auto* t = static_cast<const eshkol_tensor_t*>(eshkol_tagged_as_tensor(value));
    if (!t) t = static_cast<const eshkol_tensor_t*>(eshkol_tensor_from_collection(arena, value));
    if (!t) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: expected a tensor operand", name);
        return nullptr;
    }
    if (t->dtype == ESHKOL_TENSOR_DTYPE_DUAL || t->dtype == ESHKOL_TENSOR_DTYPE_BOXED) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
            "%s: operand is a tensor of %s, which has no linear-algebra rule", name,
            t->dtype == ESHKOL_TENSOR_DTYPE_DUAL ? "dual numbers" : "non-numeric elements");
        return nullptr;
    }
    return t;
}

bool is_exact(const eshkol_tensor_t* t) {
    return t && t->dtype == ESHKOL_TENSOR_DTYPE_EXACT;
}

// The f64 projection of a numeric operand, as `inexact` gives each element.
std::vector<double> projected(const eshkol_tensor_t* t) {
    std::vector<double> out((size_t)t->total_elements);
    if (is_exact(t)) {
        for (uint64_t i = 0; i < t->total_elements; ++i) {
            eshkol_tensor_real_to_double(&exact_slots(t)[i], &out[(size_t)i]);
        }
    } else {
        if (!out.empty()) std::memcpy(out.data(), t->elements, out.size() * sizeof(double));
    }
    return out;
}

void require_square(const eshkol_tensor_t* a, const char* name) {
    if (a->num_dimensions != 2 || a->dimensions[0] != a->dimensions[1]) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: expected a square matrix (a rank-2 tensor with equal "
                             "dimensions)", name);
    }
}

tagged pack_tensor(eshkol_tensor_t* t) {
    tagged out{};
    out.type = ESHKOL_VALUE_HEAP_PTR;
    out.data.ptr_val = (uint64_t)(uintptr_t)t;
    return out;
}

eshkol_tensor_t* f64_tensor(arena_t* arena, uint64_t ndim, const uint64_t* dims,
                            const std::vector<double>& values, const char* name) {
    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, ndim, values.size());
    if (!t) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: out of memory", name);
        return nullptr;
    }
    for (uint64_t d = 0; d < ndim; ++d) t->dimensions[d] = dims[d];
    if (!values.empty()) {
        std::memcpy(t->elements, values.data(), values.size() * sizeof(double));
    }
    return t;
}

// ── Determinant: Bareiss fraction-free elimination ──────────────────────────
//
// Every division is exact (each intermediate entry is a minor of the input),
// so an integer matrix never produces a fraction and a rational one only the
// fractions its minors already have. Row exchanges flip the sign.
tagged exact_det(const Ring& R, const eshkol_tensor_t* a) {
    const int64_t n = (int64_t)a->dimensions[0];
    if (n == 0) return Ring::integer(1);
    std::vector<tagged> m(exact_slots(a), exact_slots(a) + n * n);
    tagged prev = Ring::integer(1);
    bool negate = false;
    for (int64_t k = 0; k + 1 < n; ++k) {
        if (R.is_zero(m[k * n + k])) {
            int64_t p = k + 1;
            while (p < n && R.is_zero(m[p * n + k])) ++p;
            if (p == n) return Ring::integer(0);
            for (int64_t j = 0; j < n; ++j) std::swap(m[k * n + j], m[p * n + j]);
            negate = !negate;
        }
        for (int64_t i = k + 1; i < n; ++i) {
            for (int64_t j = k + 1; j < n; ++j) {
                m[i * n + j] = R.over(R.minus(R.times(m[i * n + j], m[k * n + k]),
                                              R.times(m[i * n + k], m[k * n + j])),
                                      prev);
            }
        }
        prev = m[k * n + k];
    }
    const tagged det = m[(n - 1) * n + (n - 1)];
    return negate ? R.minus(Ring::integer(0), det) : det;
}

// ── Solve: fraction-free elimination of [A | B], exact back substitution ────
//
// Returns the n x m solution in row-major order. A singular A is reported
// exactly, from an exact zero pivot column, never from a tolerance.
std::vector<tagged> exact_solve(const Ring& R, const eshkol_tensor_t* a,
                                const tagged* rhs, int64_t m, const char* name) {
    const int64_t n = (int64_t)a->dimensions[0];
    const int64_t w = n + m;
    std::vector<tagged> aug((size_t)(n * w));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) aug[i * w + j] = exact_slots(a)[i * n + j];
        for (int64_t c = 0; c < m; ++c) aug[i * w + n + c] = rhs[i * m + c];
    }
    tagged prev = Ring::integer(1);
    for (int64_t k = 0; k < n; ++k) {
        int64_t p = k;
        while (p < n && R.is_zero(aug[p * w + k])) ++p;
        if (p == n) {
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                                 "%s: the matrix is singular (its exact determinant is 0)",
                                 name);
            return {};
        }
        if (p != k) {
            for (int64_t j = 0; j < w; ++j) std::swap(aug[k * w + j], aug[p * w + j]);
        }
        for (int64_t i = k + 1; i < n; ++i) {
            for (int64_t j = k + 1; j < w; ++j) {
                aug[i * w + j] = R.over(R.minus(R.times(aug[k * w + k], aug[i * w + j]),
                                                R.times(aug[i * w + k], aug[k * w + j])),
                                        prev);
            }
            aug[i * w + k] = Ring::integer(0);
        }
        prev = aug[k * w + k];
    }
    std::vector<tagged> x((size_t)(n * m));
    for (int64_t c = 0; c < m; ++c) {
        for (int64_t i = n - 1; i >= 0; --i) {
            tagged acc = aug[i * w + n + c];
            for (int64_t j = i + 1; j < n; ++j) {
                acc = R.minus(acc, R.times(aug[i * w + j], x[j * m + c]));
            }
            x[i * m + c] = R.over(acc, aug[i * w + i]);
        }
    }
    return x;
}

// The right-hand side of a solve as an n x m block: a rank-1 operand of
// length n is one column; a rank-2 operand n x m is m columns.
int64_t rhs_columns(const eshkol_tensor_t* a, const eshkol_tensor_t* b, const char* name) {
    const uint64_t n = a->dimensions[0];
    if (b->num_dimensions == 1 && b->dimensions[0] == n) return 1;
    if (b->num_dimensions == 2 && b->dimensions[0] == n) return (int64_t)b->dimensions[1];
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                         "%s: the right-hand side must have %llu rows", name,
                         (unsigned long long)n);
    return 0;
}

// The f64 solve the contagion case computes, on the projections of A and B.
std::vector<double> f64_solve(const eshkol_tensor_t* a, const eshkol_tensor_t* b, int64_t m,
                              const char* name) {
    const int64_t n = (int64_t)a->dimensions[0];
    std::vector<double> lu = projected(a);
    std::vector<int64_t> piv((size_t)(n ? n : 1));
    if (n > 0 && eshkol_lu_decompose(lu.data(), piv.data(), n) == 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: the matrix is singular", name);
    }
    const std::vector<double> rhs = projected(b);
    std::vector<double> x((size_t)(n * m));
    std::vector<double> col((size_t)(n ? n : 1));
    for (int64_t c = 0; c < m; ++c) {
        for (int64_t i = 0; i < n; ++i) col[i] = rhs[i * m + c];
        if (n > 0) eshkol_lu_solve(lu.data(), piv.data(), col.data(), n);
        for (int64_t i = 0; i < n; ++i) x[i * m + c] = col[i];
    }
    return x;
}

}  // namespace

extern "C" {

/* (transpose T) for a tensor with tagged numeric slots: a rank-2 tensor's
 * slots move whole, so an exact entry keeps its exact value. */
void* eshkol_tagged_tensor_transpose(arena_t* arena, const eshkol_tensor_t* t) {
    if (!arena) arena = get_global_arena();
    if (!t || t->num_dimensions != 2) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "transpose: only supports 2D tensors");
        return nullptr;
    }
    const uint64_t rows = t->dimensions[0], cols = t->dimensions[1];
    const uint64_t dims[2] = {cols, rows};
    eshkol_tensor_t* out = exact_tensor_alloc(arena, 2, dims, rows * cols, "transpose");
    if (!out) return nullptr;
    out->dtype = t->dtype;
    const tagged* src = exact_slots(t);
    auto* dst = reinterpret_cast<tagged*>(out->elements);
    for (uint64_t i = 0; i < rows; ++i) {
        for (uint64_t j = 0; j < cols; ++j) dst[j * rows + i] = src[i * cols + j];
    }
    return out;
}

/* True when a linear-algebra operator's operands include an exact tensor, so
 * the call takes the exact dispatch (eshkol_exact_linalg_tagged) instead of
 * the f64 kernel. `b` may be null for a unary operator. */
int32_t eshkol_linalg_operands_exact(const eshkol_tagged_value_t* a,
                                     const eshkol_tagged_value_t* b) {
    auto exact_tensor = [](const eshkol_tagged_value_t* v) {
        if (!v) return false;
        auto* t = static_cast<const eshkol_tensor_t*>(eshkol_tagged_as_tensor(v));
        return t && t->dtype == ESHKOL_TENSOR_DTYPE_EXACT;
    };
    return (exact_tensor(a) || exact_tensor(b)) ? 1 : 0;
}

/* The exact dispatch for tensor-det (kind 0), tensor-inverse (kind 1),
 * tensor-solve (kind 2) and linear-solve (kind 3). Writes the tagged result:
 *   - all operands exact: an exact rational (det) or an exact tensor;
 *   - an exact operand with an inexact one: the f64 kernel's result on the
 *     projections of both, by contagion.
 * A singular matrix raises a catchable error naming the operator. */
void eshkol_exact_linalg_tagged(arena_t* arena, int32_t kind,
                                const eshkol_tagged_value_t* a_val,
                                const eshkol_tagged_value_t* b_val,
                                const char* op_name, eshkol_tagged_value_t* out) {
    if (!arena) arena = get_global_arena();
    const char* name = name_or(op_name, kind == 0   ? "tensor-det"
                                        : kind == 1 ? "tensor-inverse"
                                        : kind == 2 ? "tensor-solve"
                                                    : "linear-solve");
    const Ring R(arena);
    const eshkol_tensor_t* a = operand_tensor(arena, a_val, name);
    require_square(a, name);
    const int64_t n = (int64_t)a->dimensions[0];

    if (kind == 0) {
        if (is_exact(a)) { *out = exact_det(R, a); return; }
        std::vector<double> lu = projected(a);
        std::vector<int64_t> piv((size_t)(n ? n : 1));
        const int64_t sign = n > 0 ? eshkol_lu_decompose(lu.data(), piv.data(), n) : 1;
        *out = eshkol_make_double(sign == 0 ? 0.0 : eshkol_det_from_lu(lu.data(), n, sign));
        return;
    }

    if (kind == 1) {
        // A^{-1} solves A X = I.
        if (is_exact(a)) {
            std::vector<tagged> identity((size_t)(n * n), Ring::integer(0));
            for (int64_t i = 0; i < n; ++i) identity[i * n + i] = Ring::integer(1);
            std::vector<tagged> x = exact_solve(R, a, identity.data(), n, name);
            const uint64_t dims[2] = {(uint64_t)n, (uint64_t)n};
            eshkol_tensor_t* t = exact_tensor_alloc(arena, 2, dims, (uint64_t)(n * n), name);
            if (!x.empty()) std::memcpy(t->elements, x.data(), x.size() * sizeof(tagged));
            *out = pack_tensor(t);
            return;
        }
        // The dispatch sends a unary call here only for an exact operand.
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR, "%s: expected an exact tensor", name);
        return;
    }

    const eshkol_tensor_t* b = operand_tensor(arena, b_val, name);
    const int64_t m = rhs_columns(a, b, name);
    // The solution has the right-hand side's rank: [n] or [n, m].
    const uint64_t dims[2] = {(uint64_t)n, (uint64_t)m};
    const uint64_t ndim = b->num_dimensions;
    if (is_exact(a) && is_exact(b)) {
        std::vector<tagged> x = exact_solve(R, a, exact_slots(b), m, name);
        eshkol_tensor_t* t = exact_tensor_alloc(arena, ndim, dims, (uint64_t)(n * m), name);
        if (!x.empty()) std::memcpy(t->elements, x.data(), x.size() * sizeof(tagged));
        *out = pack_tensor(t);
        return;
    }
    // Contagion: an inexact operand makes the solve the f64 one.
    *out = pack_tensor(f64_tensor(arena, ndim, dims, f64_solve(a, b, m, name), name));
}

}  // extern "C"
