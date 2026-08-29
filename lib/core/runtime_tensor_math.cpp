/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Tensor math runtime helpers.
 */

#include "arena_memory.h"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

// These helpers are called from LLVM-generated code through extern "C" names.
// They operate on raw double arrays in row-major order.

extern "C" void eshkol_type_error_with_operand(const char* proc_name,
                                                const char* expected_type,
                                                const eshkol_tagged_value_t* actual);
extern "C" void eshkol_runtime_fatal(eshkol_exception_type_t type,
                                      const char* fmt, ...);

namespace {

/* Forward-mode tensor boundaries use the same 64-byte jet layout emitted by
 * AutodiffCodegen::packDualToTagged: the first four coefficients are the
 * value jet and the second four are the reverse-seed derivative jet.  These
 * transformer kernels are first-order forward consumers, but retaining the
 * complete storage and copying the untouched slots makes the representation
 * safe for a caller that is itself nested in another AD operation. */
struct tensor_dual_jet {
    double c[8];
};

static tensor_dual_jet jet_zero(double value = 0.0) {
    tensor_dual_jet out{};
    out.c[0] = value;
    return out;
}

static tensor_dual_jet jet_from_tagged(const eshkol_tagged_value_t& value,
                                      const char* op_name) {
    const uint8_t base = (uint8_t)(value.type & 0x0F);
    if (base == ESHKOL_VALUE_DUAL_NUMBER) {
        if (!value.data.ptr_val) {
            eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                                 "%s: null forward dual element",
                                 op_name ? op_name : "tensor AD");
        }
        return *(const tensor_dual_jet*)(uintptr_t)value.data.ptr_val;
    }
    if (base == ESHKOL_VALUE_DOUBLE) return jet_zero(value.data.double_val);
    if (base == ESHKOL_VALUE_INT64) return jet_zero((double)value.data.int_val);
    eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                         "%s: tensor AD requires numeric elements",
                         op_name ? op_name : "tensor AD");
    return jet_zero();
}

static tensor_dual_jet jet_add(const tensor_dual_jet& a,
                               const tensor_dual_jet& b) {
    tensor_dual_jet out{};
    for (int i = 0; i < 8; ++i) out.c[i] = a.c[i] + b.c[i];
    return out;
}

static tensor_dual_jet jet_sub(const tensor_dual_jet& a,
                               const tensor_dual_jet& b) {
    tensor_dual_jet out{};
    for (int i = 0; i < 8; ++i) out.c[i] = a.c[i] - b.c[i];
    return out;
}

static tensor_dual_jet jet_mul(const tensor_dual_jet& a,
                               const tensor_dual_jet& b) {
    tensor_dual_jet out{};
    for (int i = 0; i < 8; ++i) out.c[i] = a.c[i] * b.c[0] + a.c[0] * b.c[i];
    out.c[0] = a.c[0] * b.c[0];
    return out;
}

static tensor_dual_jet jet_div(const tensor_dual_jet& a,
                               const tensor_dual_jet& b,
                               const char* op_name) {
    if (b.c[0] == 0.0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: division by zero in forward dual tensor",
                             op_name ? op_name : "tensor AD");
    }
    tensor_dual_jet out{};
    const double inv = 1.0 / b.c[0];
    const double inv2 = inv * inv;
    out.c[0] = a.c[0] * inv;
    for (int i = 1; i < 8; ++i)
        out.c[i] = a.c[i] * inv - a.c[0] * b.c[i] * inv2;
    return out;
}

static tensor_dual_jet jet_unary_sqrt(const tensor_dual_jet& a,
                                      const char* op_name) {
    if (a.c[0] < 0.0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: square root domain error in forward dual tensor",
                             op_name ? op_name : "tensor AD");
    }
    tensor_dual_jet out{};
    const double root = std::sqrt(a.c[0]);
    out.c[0] = root;
    if (root == 0.0) {
        for (int i = 1; i < 8; ++i) out.c[i] = 0.0;
    } else {
        for (int i = 1; i < 8; ++i) out.c[i] = a.c[i] / (2.0 * root);
    }
    return out;
}

static tensor_dual_jet jet_unary_exp(const tensor_dual_jet& a) {
    tensor_dual_jet out{};
    const double value = std::exp(a.c[0]);
    out.c[0] = value;
    for (int i = 1; i < 8; ++i) out.c[i] = value * a.c[i];
    return out;
}

static tensor_dual_jet jet_max(const tensor_dual_jet& a,
                               const tensor_dual_jet& b) {
    return a.c[0] > b.c[0] ? a : b;
}

static tensor_dual_jet tensor_jet_at(const eshkol_tensor_t* tensor,
                                     int64_t index, const char* op_name) {
    if (!tensor || index < 0 || (uint64_t)index >= tensor->total_elements) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: tensor element index out of bounds",
                             op_name ? op_name : "tensor AD");
    }
    if (tensor->dtype == ESHKOL_TENSOR_DTYPE_DUAL) {
        const auto* elems = (const eshkol_tagged_value_t*)tensor->elements;
        return jet_from_tagged(elems[index], op_name);
    }
    double value = 0.0;
    std::memcpy(&value, tensor->elements + index, sizeof(value));
    return jet_zero(value);
}

static eshkol_tagged_value_t jet_to_tagged(arena_t* arena,
                                           const tensor_dual_jet& jet,
                                           const char* op_name) {
    void* storage = arena_allocate(arena, sizeof(tensor_dual_jet));
    if (!storage) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: failed to allocate forward dual result",
                             op_name ? op_name : "tensor AD");
    }
    std::memcpy(storage, &jet, sizeof(jet));
    eshkol_tagged_value_t result{};
    result.type = ESHKOL_VALUE_DUAL_NUMBER;
    result.flags = ESHKOL_VALUE_INEXACT_FLAG;
    result.data.ptr_val = (uint64_t)(uintptr_t)storage;
    return result;
}

static eshkol_tensor_t* dual_tensor_result(arena_t* arena,
                                           const eshkol_tensor_t* shape_like,
                                           uint64_t total,
                                           const char* op_name) {
    if (!arena || !shape_like || shape_like->num_dimensions == 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: invalid tensor shape for forward dual result",
                             op_name ? op_name : "tensor AD");
    }
    auto* result = arena_allocate_tensor_with_header(arena);
    if (!result) return nullptr;
    const size_t dims_bytes = (size_t)shape_like->num_dimensions * sizeof(uint64_t);
    result->dimensions = (uint64_t*)arena_allocate(arena, dims_bytes);
    result->elements = (int64_t*)arena_allocate(
        arena, (size_t)total * sizeof(eshkol_tagged_value_t));
    if (!result->dimensions || !result->elements) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "%s: failed to allocate forward dual tensor",
                             op_name ? op_name : "tensor AD");
    }
    std::memcpy(result->dimensions, shape_like->dimensions, dims_bytes);
    result->num_dimensions = shape_like->num_dimensions;
    result->total_elements = total;
    result->dtype = ESHKOL_TENSOR_DTYPE_DUAL;
    return result;
}

static tensor_dual_jet parameter_jet(const eshkol_tagged_value_t* source,
                                     int64_t index, const char* op_name) {
    if (!source) return jet_zero();
    const uint8_t base = (uint8_t)(source->type & 0x0F);
    if (base == ESHKOL_VALUE_HEAP_PTR && source->data.ptr_val) {
        void* ptr = (void*)(uintptr_t)source->data.ptr_val;
        const auto* header = ESHKOL_GET_HEADER(ptr);
        if (header && header->subtype == HEAP_SUBTYPE_TENSOR) {
            const auto* tensor = (const eshkol_tensor_t*)ptr;
            if (tensor->total_elements == 0) return jet_zero();
            return tensor_jet_at(tensor, index % (int64_t)tensor->total_elements,
                                 op_name);
        }
    }
    return jet_from_tagged(*source, op_name);
}

}  // namespace

extern "C" eshkol_tensor_t* eshkol_tensor_layer_norm_dual(
    const eshkol_tensor_t* input,
    const eshkol_tagged_value_t* gamma,
    const eshkol_tagged_value_t* beta,
    double epsilon) {
    if (!input || input->num_dimensions < 1 || input->total_elements == 0 ||
        input->dimensions[input->num_dimensions - 1] == 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "layer-norm: invalid shape for forward dual input");
    }
    arena_t* arena = get_global_arena();
    const int64_t width = (int64_t)input->dimensions[input->num_dimensions - 1];
    const int64_t groups = (int64_t)(input->total_elements / (uint64_t)width);
    auto* result = dual_tensor_result(arena, input, input->total_elements,
                                       "layer-norm");
    auto* out = (eshkol_tagged_value_t*)result->elements;
    const tensor_dual_jet eps = jet_zero(epsilon);
    const tensor_dual_jet one = jet_zero(1.0);
    for (int64_t group = 0; group < groups; ++group) {
        tensor_dual_jet mean = jet_zero();
        for (int64_t i = 0; i < width; ++i)
            mean = jet_add(mean, tensor_jet_at(input, group * width + i,
                                               "layer-norm"));
        mean = jet_div(mean, jet_zero((double)width), "layer-norm");

        tensor_dual_jet variance = jet_zero();
        for (int64_t i = 0; i < width; ++i) {
            const auto centered = jet_sub(
                tensor_jet_at(input, group * width + i, "layer-norm"), mean);
            variance = jet_add(variance, jet_mul(centered, centered));
        }
        variance = jet_div(variance, jet_zero((double)width), "layer-norm");
        const auto stddev = jet_unary_sqrt(jet_add(variance, eps), "layer-norm");

        for (int64_t i = 0; i < width; ++i) {
            const int64_t index = group * width + i;
            const auto centered = jet_sub(tensor_jet_at(input, index,
                                                        "layer-norm"), mean);
            auto normalized = jet_div(centered, stddev, "layer-norm");
            normalized = jet_mul(normalized, parameter_jet(gamma, i,
                                                            "layer-norm gamma"));
            normalized = jet_add(normalized, parameter_jet(beta, i,
                                                           "layer-norm beta"));
            out[index] = jet_to_tagged(arena, normalized, "layer-norm");
        }
    }
    (void)one;
    return result;
}

extern "C" eshkol_tensor_t* eshkol_tensor_scaled_dot_attention_dual(
    const eshkol_tensor_t* q,
    const eshkol_tensor_t* k,
    const eshkol_tensor_t* v,
    const eshkol_tensor_t* mask) {
    if (!q || !k || !v || (q->num_dimensions != 2 && q->num_dimensions != 3) ||
        k->num_dimensions != q->num_dimensions ||
        v->num_dimensions != q->num_dimensions) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "scaled-dot-attention: forward dual inputs must have matching rank 2 or rank 3 tensors");
    }
    const bool batched = q->num_dimensions == 3;
    const int64_t batch = batched ? (int64_t)q->dimensions[0] : 1;
    const int64_t seq_q = (int64_t)q->dimensions[batched ? 1 : 0];
    const int64_t seq_k = (int64_t)k->dimensions[batched ? 1 : 0];
    const int64_t d_k = (int64_t)q->dimensions[batched ? 2 : 1];
    const int64_t k_d = (int64_t)k->dimensions[batched ? 2 : 1];
    const int64_t d_v = (int64_t)v->dimensions[batched ? 2 : 1];
    const int64_t v_seq = (int64_t)v->dimensions[batched ? 1 : 0];
    if (d_k <= 0 || k_d != d_k || v_seq != seq_k || d_v <= 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "scaled-dot-attention: incompatible Q/K/V dimensions for forward dual input");
    }
    arena_t* arena = get_global_arena();
    auto* result = dual_tensor_result(arena, v,
        (uint64_t)batch * (uint64_t)seq_q * (uint64_t)d_v,
        "scaled-dot-attention");
    auto* out = (eshkol_tagged_value_t*)result->elements;
    const size_t score_count = (size_t)batch * (size_t)seq_q * (size_t)seq_k;
    auto* scores = (tensor_dual_jet*)arena_allocate(
        arena, score_count * sizeof(tensor_dual_jet));
    if (!scores) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "scaled-dot-attention: failed to allocate forward dual scores");
    }
    const tensor_dual_jet scale = jet_zero(std::sqrt((double)d_k));
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < seq_q; ++i) {
            for (int64_t j = 0; j < seq_k; ++j) {
                tensor_dual_jet score = jet_zero();
                for (int64_t x = 0; x < d_k; ++x) {
                    const int64_t q_index = batched
                        ? b * seq_q * d_k + i * d_k + x : i * d_k + x;
                    const int64_t k_index = batched
                        ? b * seq_k * d_k + j * d_k + x : j * d_k + x;
                    score = jet_add(score, jet_mul(tensor_jet_at(q, q_index,
                        "scaled-dot-attention"), tensor_jet_at(k, k_index,
                        "scaled-dot-attention")));
                }
                score = jet_div(score, scale, "scaled-dot-attention");
                if (mask) {
                    const int64_t mask_index = i * seq_k + j;
                    score = jet_add(score, tensor_jet_at(mask, mask_index,
                                                         "scaled-dot-attention"));
                }
                scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j] = score;
            }
            tensor_dual_jet max_score = scores[(size_t)b * seq_q * seq_k +
                                               (size_t)i * seq_k];
            for (int64_t j = 1; j < seq_k; ++j)
                max_score = jet_max(max_score,
                    scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j]);
            tensor_dual_jet exp_sum = jet_zero();
            for (int64_t j = 0; j < seq_k; ++j) {
                auto shifted = jet_sub(scores[(size_t)b * seq_q * seq_k +
                                               (size_t)i * seq_k + j], max_score);
                scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j] =
                    jet_unary_exp(shifted);
                exp_sum = jet_add(exp_sum,
                    scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j]);
            }
            for (int64_t j = 0; j < seq_k; ++j)
                scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j] =
                    jet_div(scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + j],
                            exp_sum, "scaled-dot-attention");
        }
        for (int64_t i = 0; i < seq_q; ++i) {
            for (int64_t j = 0; j < d_v; ++j) {
                tensor_dual_jet value = jet_zero();
                for (int64_t x = 0; x < seq_k; ++x) {
                    const int64_t v_index = batched
                        ? b * seq_k * d_v + x * d_v + j : x * d_v + j;
                    value = jet_add(value, jet_mul(
                        scores[(size_t)b * seq_q * seq_k + (size_t)i * seq_k + x],
                        tensor_jet_at(v, v_index, "scaled-dot-attention")));
                }
                const int64_t out_index = batched
                    ? b * seq_q * d_v + i * d_v + j : i * d_v + j;
                out[out_index] = jet_to_tagged(arena, value,
                                               "scaled-dot-attention");
            }
        }
    }
    return result;
}

// LU decomposition with partial pivoting (in-place).
// A is n x n row-major, piv[i] stores the row swapped with row i.
// Returns the sign of the permutation (+1 or -1), or 0 if singular.
extern "C" int64_t eshkol_lu_decompose(double* A, int64_t* piv, int64_t n) {
    int64_t sign = 1;
    for (int64_t i = 0; i < n; i++) piv[i] = i;

    for (int64_t k = 0; k < n; k++) {
        double max_val = 0.0;
        int64_t max_row = k;
        for (int64_t i = k; i < n; i++) {
            double v = std::fabs(A[i * n + k]);
            if (v > max_val) {
                max_val = v;
                max_row = i;
            }
        }
        if (max_val < 1e-15) return 0;

        if (max_row != k) {
            sign = -sign;
            int64_t tmp_piv = piv[k];
            piv[k] = piv[max_row];
            piv[max_row] = tmp_piv;
            for (int64_t j = 0; j < n; j++) {
                double tmp = A[k * n + j];
                A[k * n + j] = A[max_row * n + j];
                A[max_row * n + j] = tmp;
            }
        }

        double pivot = A[k * n + k];
        for (int64_t i = k + 1; i < n; i++) {
            double factor = A[i * n + k] / pivot;
            A[i * n + k] = factor;
            for (int64_t j = k + 1; j < n; j++) {
                A[i * n + j] -= factor * A[k * n + j];
            }
        }
    }
    return sign;
}

/**
 * @brief Computes a matrix determinant from its LU decomposition.
 *
 * Multiplies the diagonal entries of the LU factor (as produced by
 * eshkol_lu_decompose()) together with the permutation `sign` (+1 or -1)
 * that pivoting introduced, since det(A) = sign * det(L) * det(U) and L has
 * unit diagonal.
 *
 * @param LU   n x n row-major combined L/U factors from eshkol_lu_decompose().
 * @param n    Matrix dimension.
 * @param sign Permutation sign from eshkol_lu_decompose() (+1, -1, or 0 for singular).
 * @return     The determinant of the original matrix.
 */
extern "C" double eshkol_det_from_lu(const double* LU, int64_t n, int64_t sign) {
    double det = (double)sign;
    for (int64_t i = 0; i < n; i++) {
        det *= LU[i * n + i];
    }
    return det;
}

// Solve Ax=b using LU decomposition. b is overwritten with x.
extern "C" void eshkol_lu_solve(const double* LU, const int64_t* piv, double* b, int64_t n) {
    double* pb = (double*)std::malloc((size_t)n * sizeof(double));
    if (!pb) return;
    for (int64_t i = 0; i < n; i++) pb[i] = b[piv[i]];

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < i; j++) {
            pb[i] -= LU[i * n + j] * pb[j];
        }
    }

    for (int64_t i = n - 1; i >= 0; i--) {
        for (int64_t j = i + 1; j < n; j++) {
            pb[i] -= LU[i * n + j] * pb[j];
        }
        pb[i] /= LU[i * n + i];
    }

    std::memcpy(b, pb, (size_t)n * sizeof(double));
    std::free(pb);
}

/**
 * @brief Computes the inverse of a matrix from its LU decomposition.
 *
 * For each column of the identity matrix, solves `LU x = e_col` via
 * eshkol_lu_solve() and stores the resulting column of the inverse into
 * `inv`. Requires n solves total (one per column) and heap-allocates a
 * scratch right-hand-side vector per column.
 *
 * @param LU  n x n row-major combined L/U factors from eshkol_lu_decompose().
 * @param piv Pivot/permutation array from eshkol_lu_decompose().
 * @param inv Output n x n row-major inverse matrix (caller-allocated).
 * @param n   Matrix dimension.
 */
extern "C" void eshkol_lu_inverse(const double* LU, const int64_t* piv, double* inv, int64_t n) {
    for (int64_t col = 0; col < n; col++) {
        double* b = (double*)std::malloc((size_t)n * sizeof(double));
        if (!b) return;
        std::memset(b, 0, (size_t)n * sizeof(double));
        b[col] = 1.0;

        eshkol_lu_solve(LU, piv, b, n);

        for (int64_t row = 0; row < n; row++) {
            inv[row * n + col] = b[row];
        }
        std::free(b);
    }
}

/**
 * @brief Computes the Cholesky decomposition A = L L^T of a symmetric positive-definite matrix.
 *
 * Fills the lower-triangular factor `L` (row-major) column-by-column using
 * the standard Cholesky-Banachiewicz recurrence. Bails out if a diagonal
 * pivot is non-positive, which indicates `A` is not positive-definite.
 *
 * @param A n x n row-major symmetric input matrix.
 * @param L Output n x n row-major lower-triangular factor (caller-allocated).
 * @param n Matrix dimension.
 * @return  0 on success, -1 if `A` is not positive-definite.
 */
extern "C" int64_t eshkol_cholesky(const double* A, double* L, int64_t n) {
    std::memset(L, 0, (size_t)n * (size_t)n * sizeof(double));

    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j <= i; j++) {
            double sum = 0.0;
            if (j == i) {
                for (int64_t k = 0; k < j; k++) {
                    sum += L[j * n + k] * L[j * n + k];
                }
                double val = A[j * n + j] - sum;
                if (val <= 0.0) return -1;
                L[j * n + j] = std::sqrt(val);
            } else {
                for (int64_t k = 0; k < j; k++) {
                    sum += L[i * n + k] * L[j * n + k];
                }
                L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return 0;
}

/**
 * @brief Computes the QR decomposition A = Q R of an m x n matrix via Householder reflections.
 *
 * Copies `A` into `R` and starts `Q` as the m x m identity, then for each of
 * the first min(m,n) columns builds a Householder reflector that zeroes the
 * sub-diagonal entries of that column of `R`, applying the same reflection
 * to `R` (from the left) and accumulating it into `Q` (from the right) so
 * that `Q` ends up orthogonal and `R` upper-triangular.
 *
 * @param A m x n row-major input matrix.
 * @param Q Output m x m row-major orthogonal factor (caller-allocated).
 * @param R Output m x n row-major upper-triangular factor (caller-allocated).
 * @param m Number of rows.
 * @param n Number of columns.
 */
extern "C" void eshkol_qr_decompose(const double* A, double* Q, double* R, int64_t m, int64_t n) {
    std::memcpy(R, A, (size_t)m * (size_t)n * sizeof(double));

    std::memset(Q, 0, (size_t)m * (size_t)m * sizeof(double));
    for (int64_t i = 0; i < m; i++) Q[i * m + i] = 1.0;

    int64_t min_mn = (m < n) ? m : n;

    for (int64_t k = 0; k < min_mn; k++) {
        double* v = (double*)std::malloc((size_t)m * sizeof(double));
        if (!v) return;
        std::memset(v, 0, (size_t)m * sizeof(double));

        double norm_sq = 0.0;
        for (int64_t i = k; i < m; i++) {
            v[i] = R[i * n + k];
            norm_sq += v[i] * v[i];
        }
        double norm = std::sqrt(norm_sq);
        if (norm < 1e-15) {
            std::free(v);
            continue;
        }

        double sign = (v[k] >= 0.0) ? 1.0 : -1.0;
        v[k] += sign * norm;

        double v_norm_sq = 0.0;
        for (int64_t i = k; i < m; i++) v_norm_sq += v[i] * v[i];
        if (v_norm_sq < 1e-30) {
            std::free(v);
            continue;
        }

        double scale = 2.0 / v_norm_sq;

        for (int64_t j = k; j < n; j++) {
            double dot = 0.0;
            for (int64_t i = k; i < m; i++) dot += v[i] * R[i * n + j];
            for (int64_t i = k; i < m; i++) R[i * n + j] -= scale * v[i] * dot;
        }

        for (int64_t i = 0; i < m; i++) {
            double dot = 0.0;
            for (int64_t j2 = k; j2 < m; j2++) dot += Q[i * m + j2] * v[j2];
            for (int64_t j2 = k; j2 < m; j2++) Q[i * m + j2] -= scale * dot * v[j2];
        }

        std::free(v);
    }
}

/**
 * @brief Computes the singular value decomposition A = U * diag(S) * V^T via one-sided Jacobi rotations.
 *
 * Iteratively applies Jacobi (Givens) rotations to pairs of columns of a
 * working copy of `A` to drive its columns toward orthogonality while
 * accumulating the same rotations into `V`; after convergence (or
 * `max_sweeps` sweeps) the singular values are the column norms of the
 * rotated matrix and `U` is that matrix with columns normalized. Singular
 * values (and the corresponding columns of `U`/`V`) are then sorted into
 * descending order.
 *
 * @param A m x n row-major input matrix.
 * @param m Number of rows.
 * @param n Number of columns.
 * @param U Output m x k row-major left singular vectors (k = min(m,n)), caller-allocated.
 * @param S Output length-k singular values, descending, caller-allocated.
 * @param V Output n x n row-major right singular vectors, caller-allocated.
 */
extern "C" void eshkol_tensor_svd(
    const double* A, int64_t m, int64_t n,
    double* U, double* S, double* V)
{
    int64_t k = (m < n) ? m : n;

    double* B = (double*)std::malloc((size_t)m * (size_t)n * sizeof(double));
    if (!B) return;
    std::memcpy(B, A, (size_t)m * (size_t)n * sizeof(double));

    std::memset(V, 0, (size_t)n * (size_t)n * sizeof(double));
    for (int64_t i = 0; i < n; i++) V[i * n + i] = 1.0;

    const double eps = 1e-15;
    const int max_sweeps = 100;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        double off_norm = 0.0;

        for (int64_t p = 0; p < n - 1; p++) {
            for (int64_t q = p + 1; q < n; q++) {
                double alpha = 0.0, beta = 0.0, gamma = 0.0;
                for (int64_t i = 0; i < m; i++) {
                    double bp = B[i * n + p];
                    double bq = B[i * n + q];
                    alpha += bp * bp;
                    beta  += bq * bq;
                    gamma += bp * bq;
                }

                off_norm += gamma * gamma;

                double threshold = eps * std::sqrt(alpha * beta);
                if (threshold < 1e-300) threshold = 1e-300;
                if (std::fabs(gamma) < threshold) continue;

                double zeta = (beta - alpha) / (2.0 * gamma);
                double t;
                if (zeta >= 0.0) {
                    t = 1.0 / (zeta + std::sqrt(zeta * zeta + 1.0));
                } else {
                    t = -1.0 / (-zeta + std::sqrt(zeta * zeta + 1.0));
                }
                double c = 1.0 / std::sqrt(1.0 + t * t);
                double s = t * c;

                for (int64_t i = 0; i < m; i++) {
                    double bp = B[i * n + p];
                    double bq = B[i * n + q];
                    B[i * n + p] =  c * bp + s * bq;
                    B[i * n + q] = -s * bp + c * bq;
                }

                for (int64_t i = 0; i < n; i++) {
                    double vp = V[i * n + p];
                    double vq = V[i * n + q];
                    V[i * n + p] =  c * vp + s * vq;
                    V[i * n + q] = -s * vp + c * vq;
                }
            }
        }

        if (off_norm < eps * eps) break;
    }

    for (int64_t j = 0; j < k; j++) {
        double norm = 0.0;
        for (int64_t i = 0; i < m; i++) {
            double v = B[i * n + j];
            norm += v * v;
        }
        norm = std::sqrt(norm);
        S[j] = norm;

        if (norm > eps) {
            for (int64_t i = 0; i < m; i++) {
                U[i * k + j] = B[i * n + j] / norm;
            }
        } else {
            for (int64_t i = 0; i < m; i++) {
                U[i * k + j] = 0.0;
            }
        }
    }

    for (int64_t i = 0; i < k - 1; i++) {
        for (int64_t j = 0; j < k - 1 - i; j++) {
            if (S[j] < S[j + 1]) {
                double tmp = S[j];
                S[j] = S[j + 1];
                S[j + 1] = tmp;
                for (int64_t r = 0; r < m; r++) {
                    double t2 = U[r * k + j];
                    U[r * k + j] = U[r * k + (j + 1)];
                    U[r * k + (j + 1)] = t2;
                }
                for (int64_t r = 0; r < n; r++) {
                    double t2 = V[r * n + j];
                    V[r * n + j] = V[r * n + (j + 1)];
                    V[r * n + (j + 1)] = t2;
                }
            }
        }
    }

    std::free(B);
}

/**
 * @brief Copies a source tensor's data into a destination buffer under NumPy-style broadcasting.
 *
 * Right-aligns `src_dims` against `dst_dims` (as in NumPy broadcasting
 * rules): a source dimension must equal the corresponding destination
 * dimension or be 1 (in which case it is stretched). Iterates every flat
 * index of the destination, maps it back to the corresponding (possibly
 * broadcast) source flat index using precomputed row-major strides, and
 * copies the value. Supports at most 16 dimensions per side.
 *
 * @param src_data  Source tensor's flat row-major element buffer.
 * @param src_dims  Source shape (length `src_ndim`).
 * @param src_ndim  Source rank.
 * @param dst_data  Destination flat row-major element buffer (caller-allocated,
 *                  sized to the product of `dst_dims`).
 * @param dst_dims  Destination shape (length `dst_ndim`).
 * @param dst_ndim  Destination rank.
 * @return          0 on success; -1 if a rank exceeds 16, a dimension is
 *                  negative, shapes are incompatible for broadcasting, or a
 *                  stride computation would overflow.
 */
extern "C" int64_t eshkol_broadcast_copy(
    const double* src_data, const int64_t* src_dims, int64_t src_ndim,
    double* dst_data, const int64_t* dst_dims, int64_t dst_ndim)
{
    int64_t dst_total = 1;
    for (int64_t d = 0; d < dst_ndim; d++) dst_total *= dst_dims[d];

    int64_t src_strides[16];
    if (src_ndim > 16) return -1;
    for (int64_t d = 0; d < src_ndim; d++) {
        if (src_dims[d] < 0) return -1;
    }
    if (src_ndim > 0) {
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; d--) {
            int64_t a = src_strides[d + 1];
            int64_t b = src_dims[d + 1];
            if (a > 0 && b > INT64_MAX / a) return -1;
            src_strides[d] = a * b;
        }
    }

    int64_t dst_strides[16];
    if (dst_ndim > 16) return -1;
    for (int64_t d = 0; d < dst_ndim; d++) {
        if (dst_dims[d] < 0) return -1;
    }
    if (dst_ndim > 0) {
        dst_strides[dst_ndim - 1] = 1;
        for (int64_t d = dst_ndim - 2; d >= 0; d--) {
            int64_t a = dst_strides[d + 1];
            int64_t b = dst_dims[d + 1];
            if (a > 0 && b > INT64_MAX / a) return -1;
            dst_strides[d] = a * b;
        }
    }

    int64_t offset = dst_ndim - src_ndim;
    for (int64_t d = 0; d < src_ndim; d++) {
        int64_t dd = d + offset;
        if (src_dims[d] != 1 && src_dims[d] != dst_dims[dd]) {
            return -1;
        }
    }

    for (int64_t flat = 0; flat < dst_total; flat++) {
        int64_t remaining = flat;
        int64_t src_flat = 0;

        for (int64_t d = 0; d < dst_ndim; d++) {
            int64_t idx = remaining / dst_strides[d];
            remaining %= dst_strides[d];

            int64_t src_d = d - offset;
            if (src_d >= 0 && src_d < src_ndim) {
                if (src_dims[src_d] != 1) {
                    src_flat += idx * src_strides[src_d];
                }
            }
        }

        dst_data[flat] = src_data[src_flat];
    }
    return 0;
}

/**
 * @brief Converts a Scheme cons-list of integers into a flat dims array.
 *
 * Walks `cons_ptr` as a chain of arena_tagged_cons_cell_t, appending each
 * car's int64 value to `dims_out` until the list ends (a cdr tagged
 * ESHKOL_VALUE_NULL or a null cdr pointer), `max_dims` entries have been
 * collected, or a non-integer element is encountered — in which case a type
 * error is raised via eshkol_type_error_with_operand() (tagged "reshape")
 * and the count collected so far is returned.
 *
 * @param cons_ptr  Head of the cons-list (as an arena_tagged_cons_cell_t*).
 * @param dims_out  Output array of length at least `max_dims`.
 * @param max_dims  Maximum number of dimensions to extract.
 * @return          Number of dimensions written to `dims_out`.
 */
/**
 * @brief Counts the dimensions in a Scheme cons-list shape without reading them.
 *
 * Codegen calls this first so it can size the dims array from the shape list
 * itself. Before it existed, `make-tensor` and `reshape` both allocated a
 * fixed 16-entry dims array and passed max_dims = 16 to
 * eshkol_cons_list_to_dims(), which silently dropped every dimension past the
 * sixteenth — a rank-19 reshape came back rank-16 with the wrong element
 * count instead of failing. Rank is now governed by the shape.
 *
 * @param cons_ptr Head of the cons-list (as an arena_tagged_cons_cell_t*).
 * @return         Number of cells in the list (0 for a null/empty list).
 */
extern "C" int64_t eshkol_cons_list_dim_count(const void* cons_ptr)
{
    int64_t count = 0;
    const arena_tagged_cons_cell_t* current =
        (const arena_tagged_cons_cell_t*)cons_ptr;
    while (current != NULL) {
        count++;
        uint8_t cdr_type = arena_tagged_cons_get_type(current, true);
        if (ESHKOL_GET_BASE_TYPE(cdr_type) == ESHKOL_VALUE_NULL) break;
        uint64_t cdr_ptr = arena_tagged_cons_get_ptr(current, true);
        if (cdr_ptr == 0) break;
        current = (const arena_tagged_cons_cell_t*)(uintptr_t)cdr_ptr;
    }
    return count;
}

/**
 * @brief Number of dimension entries a tensor-shaped shape argument carries,
 *        i.e. its element count. Companion to eshkol_cons_list_dim_count()
 *        for the `(reshape t other-tensor-shape)` form.
 */
extern "C" int64_t eshkol_tensor_dim_count(const void* tensor_ptr)
{
    const eshkol_tensor_t* t = (const eshkol_tensor_t*)tensor_ptr;
    return t ? (int64_t)t->total_elements : 0;
}

extern "C" int64_t eshkol_cons_list_to_dims(
    const void* cons_ptr, int64_t* dims_out, int64_t max_dims)
{
    int64_t count = 0;
    const arena_tagged_cons_cell_t* current =
        (const arena_tagged_cons_cell_t*)cons_ptr;

    while (current != NULL && count < max_dims) {
        if (current->car.type != ESHKOL_VALUE_INT64) {
            eshkol_type_error_with_operand("reshape", "integer dimension", &current->car);
            return count;
        }

        dims_out[count] = current->car.data.int_val;
        count++;

        uint8_t cdr_type = arena_tagged_cons_get_type(current, true);
        uint8_t cdr_base = ESHKOL_GET_BASE_TYPE(cdr_type);
        if (cdr_base == ESHKOL_VALUE_NULL) break;

        uint64_t cdr_ptr = arena_tagged_cons_get_ptr(current, true);
        if (cdr_ptr == 0) break;
        current = (const arena_tagged_cons_cell_t*)(uintptr_t)cdr_ptr;
    }

    return count;
}

/**
 * @brief Computes the total element count of a tensor shape.
 *
 * @param dims Array of dimension sizes.
 * @param ndim Number of dimensions.
 * @return     Product of all dimension sizes (1 if `ndim` is 0).
 */
extern "C" int64_t eshkol_compute_dims_total(
    const int64_t* dims, int64_t ndim)
{
    int64_t total = 1;
    for (int64_t i = 0; i < ndim; i++) {
        total *= dims[i];
    }
    return total;
}

/**
 * @brief Converts a runtime tensor whose elements encode dimension sizes into a flat dims array.
 *
 * Reads up to `max_dims` of the tensor's elements (capped by its
 * `total_elements`), reinterpreting each raw element slot's bit pattern as a
 * double (elements are stored as doubles even though the field type is
 * int64_t) and truncating it to an int64_t dimension size. Used where a
 * shape can be supplied as a tensor of values rather than a cons-list (see
 * eshkol_cons_list_to_dims()).
 *
 * @param tensor_ptr Tensor to read (as an eshkol_tensor_t*); returns 0 if null.
 * @param dims_out   Output array of length at least `max_dims`; returns 0 if null.
 * @param max_dims   Maximum number of dimensions to extract.
 * @return           Number of dimensions written to `dims_out`.
 */
extern "C" int64_t eshkol_tensor_to_dims(
    const void* tensor_ptr, int64_t* dims_out, int64_t max_dims)
{
    const eshkol_tensor_t* t = (const eshkol_tensor_t*)tensor_ptr;
    if (!t || !dims_out) return 0;
    int64_t count = (int64_t)t->total_elements;
    if (count > max_dims) count = max_dims;
    for (int64_t i = 0; i < count; i++) {
        double dval;
        std::memcpy(&dval, &t->elements[i], sizeof(double));
        dims_out[i] = (int64_t)dval;
    }
    return count;
}

/**
 * @brief Checks whether two tensor shapes are identical.
 *
 * @param dims1 First shape.
 * @param ndim1 Rank of the first shape.
 * @param dims2 Second shape.
 * @param ndim2 Rank of the second shape.
 * @return      1 if both ranks and all dimension sizes match, 0 otherwise.
 */
extern "C" int64_t eshkol_shapes_equal(
    const int64_t* dims1, int64_t ndim1,
    const int64_t* dims2, int64_t ndim2)
{
    if (ndim1 != ndim2) return 0;
    for (int64_t i = 0; i < ndim1; i++) {
        if (dims1[i] != dims2[i]) return 0;
    }
    return 1;
}

/**
 * @brief Computes the NumPy-style broadcast result shape of two shapes.
 *
 * Right-aligns `a_dims` and `b_dims` and, for each aligned position, takes
 * the larger of the two sizes when one side is 1, or requires them to match
 * exactly otherwise. Result rank is the larger of the two input ranks and
 * must not exceed 16.
 *
 * @param a_dims   First shape.
 * @param a_ndim   Rank of the first shape.
 * @param b_dims   Second shape.
 * @param b_ndim   Rank of the second shape.
 * @param out_dims Output broadcast shape (length at least the result rank, max 16).
 * @return         Result rank, or -1 if the shapes are incompatible or the
 *                 result rank would exceed 16.
 */
static int64_t compute_broadcast_shape(
    const int64_t* a_dims, int64_t a_ndim,
    const int64_t* b_dims, int64_t b_ndim,
    int64_t* out_dims)
{
    int64_t out_ndim = (a_ndim > b_ndim) ? a_ndim : b_ndim;
    if (out_ndim > 16) return -1;

    for (int64_t i = 0; i < out_ndim; i++) {
        int64_t ai = (i < a_ndim) ? a_dims[a_ndim - 1 - i] : 1;
        int64_t bi = (i < b_ndim) ? b_dims[b_ndim - 1 - i] : 1;

        if (ai == bi) {
            out_dims[out_ndim - 1 - i] = ai;
        } else if (ai == 1) {
            out_dims[out_ndim - 1 - i] = bi;
        } else if (bi == 1) {
            out_dims[out_ndim - 1 - i] = ai;
        } else {
            return -1;
        }
    }
    return out_ndim;
}

/** @brief Map a flat broadcast-output index back to one source tensor index.
 *         Used by the legacy scalarising AD oracle as well as runtime tests. */
extern "C" int64_t eshkol_broadcast_source_index(
    int64_t flat, const int64_t* out_dims, int64_t out_ndim,
    const int64_t* src_dims, int64_t src_ndim)
{
    if (flat < 0 || out_ndim < 0 || src_ndim < 0 || out_ndim > 16 ||
        src_ndim > out_ndim || (out_ndim > 0 && !out_dims) ||
        (src_ndim > 0 && !src_dims)) return -1;
    int64_t src_strides[16] = {0};
    if (src_ndim > 0) {
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; --d) {
            if (src_dims[d + 1] <= 0 ||
                src_strides[d + 1] > INT64_MAX / src_dims[d + 1]) return -1;
            src_strides[d] = src_strides[d + 1] * src_dims[d + 1];
        }
    }
    int64_t source_index = 0;
    int64_t remaining = flat;
    for (int64_t out_i = out_ndim - 1; out_i >= 0; --out_i) {
        int64_t dim = out_dims[out_i];
        if (dim <= 0) return -1;
        int64_t coordinate = remaining % dim;
        remaining /= dim;
        int64_t src_i = out_i - (out_ndim - src_ndim);
        if (src_i >= 0 && src_dims[src_i] != 1)
            source_index += coordinate * src_strides[src_i];
    }
    return source_index;
}

/**
 * @brief Applies a broadcasting elementwise binary operation to two tensors.
 *
 * Computes the NumPy-style broadcast shape of `a` and `b` (via
 * compute_broadcast_shape()), writes it to `out_dims`/`out_ndim_out`/
 * `out_total_out`, then for every flat index of the result maps back to the
 * corresponding (possibly broadcast) input elements via precomputed
 * row-major strides and combines them with `op`.
 *
 * @param op            Operation selector: 0 = add, 1 = subtract, 2 = multiply,
 *                      3 = divide (division by zero yields 0.0 rather than
 *                      trapping); any other value yields 0.0.
 * @param a_data        First operand's flat row-major elements.
 * @param a_dims        First operand's shape.
 * @param a_ndim        First operand's rank.
 * @param b_data        Second operand's flat row-major elements.
 * @param b_dims        Second operand's shape.
 * @param b_ndim        Second operand's rank.
 * @param out_data      Output flat row-major elements (caller-allocated to
 *                      the broadcast total size).
 * @param out_dims      Output broadcast shape (caller-allocated, length >=
 *                      max(a_ndim, b_ndim)).
 * @param out_ndim_out  Output broadcast rank.
 * @param out_total_out Output total element count of the broadcast result.
 * @return              0 on success, -1 if the shapes cannot be broadcast together.
 */
extern "C" int64_t eshkol_broadcast_elementwise_f64(
    int64_t op,
    const double* a_data, const int64_t* a_dims, int64_t a_ndim,
    const double* b_data, const int64_t* b_dims, int64_t b_ndim,
    double* out_data, int64_t* out_dims, int64_t* out_ndim_out,
    int64_t* out_total_out)
{
    int64_t bcast_dims[16];
    int64_t out_ndim = compute_broadcast_shape(a_dims, a_ndim, b_dims, b_ndim, bcast_dims);
    if (out_ndim < 0) return -1;

    for (int64_t i = 0; i < out_ndim; i++) out_dims[i] = bcast_dims[i];
    *out_ndim_out = out_ndim;

    int64_t out_total = 1;
    for (int64_t d = 0; d < out_ndim; d++) out_total *= bcast_dims[d];
    *out_total_out = out_total;

    int64_t out_strides[16], a_strides[16], b_strides[16];
    if (out_ndim > 0) {
        out_strides[out_ndim - 1] = 1;
        for (int64_t d = out_ndim - 2; d >= 0; d--)
            out_strides[d] = out_strides[d + 1] * bcast_dims[d + 1];
    }
    if (a_ndim > 0) {
        a_strides[a_ndim - 1] = 1;
        for (int64_t d = a_ndim - 2; d >= 0; d--)
            a_strides[d] = a_strides[d + 1] * a_dims[d + 1];
    }
    if (b_ndim > 0) {
        b_strides[b_ndim - 1] = 1;
        for (int64_t d = b_ndim - 2; d >= 0; d--)
            b_strides[d] = b_strides[d + 1] * b_dims[d + 1];
    }

    int64_t a_offset = out_ndim - a_ndim;
    int64_t b_offset = out_ndim - b_ndim;

    for (int64_t flat = 0; flat < out_total; flat++) {
        int64_t remaining = flat;
        int64_t a_flat = 0, b_flat = 0;

        for (int64_t d = 0; d < out_ndim; d++) {
            int64_t idx = remaining / out_strides[d];
            remaining %= out_strides[d];

            int64_t ad = d - a_offset;
            if (ad >= 0 && ad < a_ndim && a_dims[ad] != 1)
                a_flat += idx * a_strides[ad];

            int64_t bd = d - b_offset;
            if (bd >= 0 && bd < b_ndim && b_dims[bd] != 1)
                b_flat += idx * b_strides[bd];
        }

        double a_val = a_data[a_flat];
        double b_val = b_data[b_flat];
        double result;

        switch (op) {
            case 0: result = a_val + b_val; break;
            case 1: result = a_val - b_val; break;
            case 2: result = a_val * b_val; break;
            case 3: result = (b_val != 0.0) ? a_val / b_val : 0.0; break;
            default: result = 0.0; break;
        }

        out_data[flat] = result;
    }
    return 0;
}

/**
 * @brief Concatenates multiple tensors along one axis using precomputed strides.
 *
 * For each "outer" index (an iteration over the dimensions preceding the
 * concatenation axis) copies, in tensor order, a contiguous chunk of
 * `src_axis_dims[t] * stride_after` elements from each source tensor into
 * the next position of `result_data`. `stride_after` is the product of the
 * dimensions after the concatenation axis, so each chunk covers exactly one
 * source tensor's slice of the axis for that outer index.
 *
 * @param result_data   Output flat row-major buffer (caller-allocated to the
 *                      concatenated total size).
 * @param num_tensors   Number of source tensors being concatenated.
 * @param src_datas     Array of `num_tensors` pointers to each source's flat elements.
 * @param src_axis_dims Each source's size along the concatenation axis.
 * @param stride_after  Product of dimension sizes after the concatenation axis.
 * @param outer_count   Product of dimension sizes before the concatenation axis.
 */
extern "C" void eshkol_concat_strided(
    double* result_data,
    int64_t num_tensors,
    const double** src_datas,
    const int64_t* src_axis_dims,
    int64_t stride_after,
    int64_t outer_count)
{
    double* dst = result_data;
    for (int64_t outer = 0; outer < outer_count; outer++) {
        for (int64_t t = 0; t < num_tensors; t++) {
            int64_t chunk = src_axis_dims[t] * stride_after;
            int64_t src_offset = outer * chunk;
            std::memcpy(dst, src_datas[t] + src_offset, (size_t)(chunk * (int64_t)sizeof(double)));
            dst += chunk;
        }
    }
}

/**
 * @brief Computes a batch of independent row-major matrix multiplications C = A * B.
 *
 * For each of `batch` matrices, zero-initializes the corresponding M x N
 * slice of `c` and accumulates the naive triple-loop product of the M x K
 * slice of `a` and the K x N slice of `b`.
 *
 * @param a     Batched left operand, `batch` row-major M x K matrices, flattened.
 * @param b     Batched right operand, `batch` row-major K x N matrices, flattened.
 * @param c     Output, `batch` row-major M x N matrices, flattened (caller-allocated).
 * @param batch Number of independent matrix pairs.
 * @param M     Rows of each A slice / C slice.
 * @param K     Columns of each A slice / rows of each B slice.
 * @param N     Columns of each B slice / C slice.
 */
extern "C" void eshkol_batch_matmul_f64(
    const double* __restrict__ a,
    const double* __restrict__ b,
    double*       __restrict__ c,
    int64_t batch, int64_t M, int64_t K, int64_t N)
{
    for (int64_t bs = 0; bs < batch; bs++) {
        const double* A = a + bs * M * K;
        const double* B = b + bs * K * N;
        double*       C = c + bs * M * N;

        for (int64_t idx = 0; idx < M * N; idx++) C[idx] = 0.0;

        for (int64_t i = 0; i < M; i++) {
            for (int64_t kk = 0; kk < K; kk++) {
                double a_ik = A[i * K + kk];
                for (int64_t j = 0; j < N; j++) {
                    C[i * N + j] += a_ik * B[kk * N + j];
                }
            }
        }
    }
}
