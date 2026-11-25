/*
 * Staged Zero-Allocation Compiled Backward Graphs (Value + Gradient)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/staged_value_grad.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace {

inline size_t align_up(size_t n, size_t align) {
    if (align <= 1u) return n;
    return (n + align - 1u) & ~(align - 1u);
}

inline double softplus_stable(double x) {
    if (x > 30.0) return x;
    if (x < -30.0) return std::exp(x);
    return std::log1p(std::exp(x));
}

inline double sigmoid_stable(double x) {
    if (x >= 0.0) {
        double ez = std::exp(-x);
        return 1.0 / (1.0 + ez);
    }
    double ez = std::exp(x);
    return ez / (1.0 + ez);
}

struct StagedNodeDesc {
    eshkol_staged_op_kind_t op = ESHKOL_STAGED_OP_INPUT;
    uint32_t dim = 0;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t ext_offset = 0; /* Slice offset into inputs_flat or params_flat */
    double scalar = 1.0;
};

struct CompiledOp {
    eshkol_staged_op_kind_t op = ESHKOL_STAGED_OP_INPUT;
    uint32_t dim = 0;
    uint32_t lhs = UINT32_MAX;
    uint32_t rhs = UINT32_MAX;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t ext_offset = 0;
    double scalar = 1.0;
    size_t val_byte_offset = 0;
    size_t adj_byte_offset = 0;
};

} /* anonymous namespace */

struct eshkol_staged_builder {
    std::vector<StagedNodeDesc> nodes;
    uint32_t total_inputs = 0;
    uint32_t total_params = 0;
};

struct eshkol_staged_plan {
    std::vector<CompiledOp> ops;
    uint32_t loss_node = 0;
    uint32_t total_inputs = 0;
    uint32_t total_params = 0;
    size_t required_scratch_bytes = 0;
};

/* ============================================================================
 * Dedicated Caller-Owned Scratch Arena (Zero System Heap Calls)
 * ============================================================================ */

extern "C" bool eshkol_staged_scratch_init(eshkol_staged_scratch_arena_t* arena,
                                           uint8_t* backing_buffer,
                                           size_t capacity_bytes) {
    if (!arena || !backing_buffer || capacity_bytes == 0) return false;
    std::memset(arena, 0, sizeof(*arena));
    arena->buffer = backing_buffer;
    arena->capacity = capacity_bytes;
    return true;
}

extern "C" void eshkol_staged_scratch_reset(eshkol_staged_scratch_arena_t* arena) {
    if (!arena) return;
    arena->offset = 0;
    arena->resets++;
}

extern "C" void* eshkol_staged_scratch_alloc(eshkol_staged_scratch_arena_t* arena,
                                             size_t bytes,
                                             size_t alignment) {
    if (!arena || !arena->buffer) return nullptr;
    size_t aligned = align_up(arena->offset, alignment == 0 ? 16u : alignment);
    if (aligned + bytes > arena->capacity) return nullptr;
    void* ptr = arena->buffer + aligned;
    arena->offset = aligned + bytes;
    if (arena->offset > arena->high_water) arena->high_water = arena->offset;
    arena->bump_allocs++;
    return ptr;
}

/* ============================================================================
 * Graph Builder & Static Compiler
 * ============================================================================ */

extern "C" eshkol_staged_builder_t* eshkol_staged_builder_create(void) {
    return new (std::nothrow) eshkol_staged_builder();
}

extern "C" void eshkol_staged_builder_destroy(eshkol_staged_builder_t* b) {
    delete b;
}

extern "C" uint32_t eshkol_staged_add_input(eshkol_staged_builder_t* b, uint32_t dim) {
    if (!b || dim == 0) return UINT32_MAX;
    StagedNodeDesc d;
    d.op = ESHKOL_STAGED_OP_INPUT;
    d.dim = dim;
    d.ext_offset = b->total_inputs;
    b->total_inputs += dim;
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" uint32_t eshkol_staged_add_param(eshkol_staged_builder_t* b, uint32_t dim) {
    if (!b || dim == 0) return UINT32_MAX;
    StagedNodeDesc d;
    d.op = ESHKOL_STAGED_OP_PARAM;
    d.dim = dim;
    d.ext_offset = b->total_params;
    b->total_params += dim;
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" uint32_t eshkol_staged_add_matvec(eshkol_staged_builder_t* b,
                                             uint32_t w_node, uint32_t x_node,
                                             uint32_t rows, uint32_t cols) {
    if (!b || rows == 0 || cols == 0) return UINT32_MAX;
    if (w_node >= b->nodes.size() || x_node >= b->nodes.size()) return UINT32_MAX;
    if (b->nodes[w_node].dim != rows * cols || b->nodes[x_node].dim != cols) return UINT32_MAX;

    StagedNodeDesc d;
    d.op = ESHKOL_STAGED_OP_MATVEC;
    d.dim = rows;
    d.lhs = w_node;
    d.rhs = x_node;
    d.rows = rows;
    d.cols = cols;
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" uint32_t eshkol_staged_add_binary(eshkol_staged_builder_t* b,
                                             eshkol_staged_op_kind_t op,
                                             uint32_t lhs_node, uint32_t rhs_node) {
    if (!b || lhs_node >= b->nodes.size() || rhs_node >= b->nodes.size()) return UINT32_MAX;
    if (b->nodes[lhs_node].dim != b->nodes[rhs_node].dim) return UINT32_MAX;

    StagedNodeDesc d;
    d.op = op;
    d.lhs = lhs_node;
    d.rhs = rhs_node;
    switch (op) {
        case ESHKOL_STAGED_OP_ADD:
        case ESHKOL_STAGED_OP_MUL:
            d.dim = b->nodes[lhs_node].dim;
            break;
        case ESHKOL_STAGED_OP_DOT:
        case ESHKOL_STAGED_OP_MSE:
            d.dim = 1u;
            break;
        default:
            return UINT32_MAX;
    }
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" uint32_t eshkol_staged_add_unary(eshkol_staged_builder_t* b,
                                            eshkol_staged_op_kind_t op,
                                            uint32_t arg_node) {
    if (!b || arg_node >= b->nodes.size()) return UINT32_MAX;
    StagedNodeDesc d;
    d.op = op;
    d.lhs = arg_node;
    switch (op) {
        case ESHKOL_STAGED_OP_TANH:
        case ESHKOL_STAGED_OP_SOFTPLUS:
            d.dim = b->nodes[arg_node].dim;
            break;
        case ESHKOL_STAGED_OP_LOGSUMEXP:
        case ESHKOL_STAGED_OP_SUM:
            d.dim = 1u;
            break;
        default:
            return UINT32_MAX;
    }
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" uint32_t eshkol_staged_add_scale(eshkol_staged_builder_t* b,
                                            uint32_t arg_node, double alpha) {
    if (!b || arg_node >= b->nodes.size() || !std::isfinite(alpha)) return UINT32_MAX;
    StagedNodeDesc d;
    d.op = ESHKOL_STAGED_OP_SCALE;
    d.lhs = arg_node;
    d.dim = b->nodes[arg_node].dim;
    d.scalar = alpha;
    uint32_t id = static_cast<uint32_t>(b->nodes.size());
    b->nodes.push_back(d);
    return id;
}

extern "C" eshkol_staged_plan_t* eshkol_compile_staged_value_grad(
    const eshkol_staged_builder_t* b, uint32_t loss_node) {
    if (!b || loss_node >= b->nodes.size()) return nullptr;
    if (b->nodes[loss_node].dim != 1u) return nullptr;

    auto* plan = new (std::nothrow) eshkol_staged_plan();
    if (!plan) return nullptr;

    const uint32_t n_ops = loss_node + 1u;
    plan->ops.resize(n_ops);
    plan->loss_node = loss_node;
    plan->total_inputs = b->total_inputs;
    plan->total_params = b->total_params;

    size_t cursor = 0;
    for (uint32_t i = 0; i < n_ops; i++) {
        const StagedNodeDesc& src = b->nodes[i];
        CompiledOp& dst = plan->ops[i];
        dst.op = src.op;
        dst.dim = src.dim;
        dst.lhs = src.lhs;
        dst.rhs = src.rhs;
        dst.rows = src.rows;
        dst.cols = src.cols;
        dst.ext_offset = src.ext_offset;
        dst.scalar = src.scalar;

        cursor = align_up(cursor, 16u);
        dst.val_byte_offset = cursor;
        cursor += static_cast<size_t>(dst.dim) * sizeof(double);

        cursor = align_up(cursor, 16u);
        dst.adj_byte_offset = cursor;
        cursor += static_cast<size_t>(dst.dim) * sizeof(double);
    }
    plan->required_scratch_bytes = align_up(cursor, 16u);
    return plan;
}

extern "C" void eshkol_staged_plan_destroy(eshkol_staged_plan_t* plan) {
    delete plan;
}

extern "C" size_t eshkol_staged_plan_required_scratch(const eshkol_staged_plan_t* plan) {
    return plan ? plan->required_scratch_bytes : 0u;
}

extern "C" uint32_t eshkol_staged_plan_total_inputs(const eshkol_staged_plan_t* plan) {
    return plan ? plan->total_inputs : 0u;
}

extern "C" uint32_t eshkol_staged_plan_total_params(const eshkol_staged_plan_t* plan) {
    return plan ? plan->total_params : 0u;
}

/* ============================================================================
 * Zero-Allocation Forward + Reverse-Mode Execution
 * ============================================================================ */

extern "C" bool eshkol_staged_eval_value_grad(const eshkol_staged_plan_t* plan,
                                              eshkol_staged_scratch_arena_t* scratch_arena,
                                              const double* inputs_flat,
                                              uint32_t inputs_len,
                                              const double* params_flat,
                                              uint32_t params_len,
                                              double* out_value,
                                              double* out_param_grads,
                                              uint32_t grads_len,
                                              uint64_t* out_fd_evals) {
    if (!plan || !scratch_arena || !out_value) return false;
    if (plan->total_inputs > 0 && (!inputs_flat || inputs_len < plan->total_inputs)) return false;
    if (plan->total_params > 0 && (!params_flat || params_len < plan->total_params)) return false;
    if (out_param_grads && grads_len < plan->total_params) return false;
    if (out_fd_evals) *out_fd_evals = 0;

    /* Reset caller scratch arena and carve single contiguous tape block — ZERO heap calls! */
    eshkol_staged_scratch_reset(scratch_arena);
    uint8_t* tape = static_cast<uint8_t*>(
        eshkol_staged_scratch_alloc(scratch_arena, plan->required_scratch_bytes, 16u));
    if (!tape) return false;

    auto val_ptr = [tape, plan](uint32_t node) -> double* {
        return reinterpret_cast<double*>(tape + plan->ops[node].val_byte_offset);
    };
    auto adj_ptr = [tape, plan](uint32_t node) -> double* {
        return reinterpret_cast<double*>(tape + plan->ops[node].adj_byte_offset);
    };

    const uint32_t N = static_cast<uint32_t>(plan->ops.size());

    /* Forward pass */
    for (uint32_t i = 0; i < N; i++) {
        const CompiledOp& op = plan->ops[i];
        double* y = val_ptr(i);
        double* ay = adj_ptr(i);
        std::memset(ay, 0, static_cast<size_t>(op.dim) * sizeof(double));

        switch (op.op) {
            case ESHKOL_STAGED_OP_INPUT:
                std::memcpy(y, inputs_flat + op.ext_offset,
                            static_cast<size_t>(op.dim) * sizeof(double));
                break;
            case ESHKOL_STAGED_OP_PARAM:
                std::memcpy(y, params_flat + op.ext_offset,
                            static_cast<size_t>(op.dim) * sizeof(double));
                break;
            case ESHKOL_STAGED_OP_MATVEC: {
                const double* W = val_ptr(op.lhs);
                const double* x = val_ptr(op.rhs);
                for (uint32_t r = 0; r < op.rows; r++) {
                    double acc = 0.0;
                    const double* wrow = W + static_cast<size_t>(r) * op.cols;
                    for (uint32_t c = 0; c < op.cols; c++) acc += wrow[c] * x[c];
                    y[r] = acc;
                }
                break;
            }
            case ESHKOL_STAGED_OP_ADD: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                for (uint32_t k = 0; k < op.dim; k++) y[k] = a[k] + b[k];
                break;
            }
            case ESHKOL_STAGED_OP_MUL: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                for (uint32_t k = 0; k < op.dim; k++) y[k] = a[k] * b[k];
                break;
            }
            case ESHKOL_STAGED_OP_TANH: {
                const double* a = val_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) y[k] = std::tanh(a[k]);
                break;
            }
            case ESHKOL_STAGED_OP_SOFTPLUS: {
                const double* a = val_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) y[k] = softplus_stable(a[k]);
                break;
            }
            case ESHKOL_STAGED_OP_LOGSUMEXP: {
                const double* a = val_ptr(op.lhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                double m = a[0];
                for (uint32_t k = 1; k < d; k++) if (a[k] > m) m = a[k];
                double s = 0.0;
                for (uint32_t k = 0; k < d; k++) s += std::exp(a[k] - m);
                y[0] = m + std::log(s);
                break;
            }
            case ESHKOL_STAGED_OP_DOT: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                double acc = 0.0;
                for (uint32_t k = 0; k < d; k++) acc += a[k] * b[k];
                y[0] = acc;
                break;
            }
            case ESHKOL_STAGED_OP_MSE: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                double acc = 0.0;
                for (uint32_t k = 0; k < d; k++) {
                    double diff = a[k] - b[k];
                    acc += 0.5 * diff * diff;
                }
                y[0] = acc;
                break;
            }
            case ESHKOL_STAGED_OP_SUM: {
                const double* a = val_ptr(op.lhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                double acc = 0.0;
                for (uint32_t k = 0; k < d; k++) acc += a[k];
                y[0] = acc;
                break;
            }
            case ESHKOL_STAGED_OP_SCALE: {
                const double* a = val_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) y[k] = op.scalar * a[k];
                break;
            }
        }
    }

    *out_value = val_ptr(plan->loss_node)[0];
    if (!out_param_grads) return std::isfinite(*out_value);

    /* Reverse-mode adjoint pass */
    adj_ptr(plan->loss_node)[0] = 1.0;
    for (uint32_t i = N; i-- > 0; ) {
        const CompiledOp& op = plan->ops[i];
        const double* y = val_ptr(i);
        const double* gy = adj_ptr(i);

        switch (op.op) {
            case ESHKOL_STAGED_OP_INPUT:
            case ESHKOL_STAGED_OP_PARAM:
                break;
            case ESHKOL_STAGED_OP_MATVEC: {
                const double* W = val_ptr(op.lhs);
                const double* x = val_ptr(op.rhs);
                double* gW = adj_ptr(op.lhs);
                double* gx = adj_ptr(op.rhs);
                for (uint32_t r = 0; r < op.rows; r++) {
                    double gyr = gy[r];
                    const double* wrow = W + static_cast<size_t>(r) * op.cols;
                    double* gwrow = gW + static_cast<size_t>(r) * op.cols;
                    for (uint32_t c = 0; c < op.cols; c++) {
                        gwrow[c] += gyr * x[c];
                        gx[c] += gyr * wrow[c];
                    }
                }
                break;
            }
            case ESHKOL_STAGED_OP_ADD: {
                double* ga = adj_ptr(op.lhs);
                double* gb = adj_ptr(op.rhs);
                for (uint32_t k = 0; k < op.dim; k++) {
                    ga[k] += gy[k];
                    gb[k] += gy[k];
                }
                break;
            }
            case ESHKOL_STAGED_OP_MUL: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                double* ga = adj_ptr(op.lhs);
                double* gb = adj_ptr(op.rhs);
                for (uint32_t k = 0; k < op.dim; k++) {
                    ga[k] += gy[k] * b[k];
                    gb[k] += gy[k] * a[k];
                }
                break;
            }
            case ESHKOL_STAGED_OP_TANH: {
                double* ga = adj_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) {
                    ga[k] += gy[k] * (1.0 - y[k] * y[k]);
                }
                break;
            }
            case ESHKOL_STAGED_OP_SOFTPLUS: {
                const double* a = val_ptr(op.lhs);
                double* ga = adj_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) {
                    ga[k] += gy[k] * sigmoid_stable(a[k]);
                }
                break;
            }
            case ESHKOL_STAGED_OP_LOGSUMEXP: {
                const double* a = val_ptr(op.lhs);
                double* ga = adj_ptr(op.lhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                const double lse = y[0];
                const double g0 = gy[0];
                for (uint32_t k = 0; k < d; k++) {
                    ga[k] += g0 * std::exp(a[k] - lse);
                }
                break;
            }
            case ESHKOL_STAGED_OP_DOT: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                double* ga = adj_ptr(op.lhs);
                double* gb = adj_ptr(op.rhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                const double g0 = gy[0];
                for (uint32_t k = 0; k < d; k++) {
                    ga[k] += g0 * b[k];
                    gb[k] += g0 * a[k];
                }
                break;
            }
            case ESHKOL_STAGED_OP_MSE: {
                const double* a = val_ptr(op.lhs);
                const double* b = val_ptr(op.rhs);
                double* ga = adj_ptr(op.lhs);
                double* gb = adj_ptr(op.rhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                const double g0 = gy[0];
                for (uint32_t k = 0; k < d; k++) {
                    double diff = a[k] - b[k];
                    ga[k] += g0 * diff;
                    gb[k] -= g0 * diff;
                }
                break;
            }
            case ESHKOL_STAGED_OP_SUM: {
                double* ga = adj_ptr(op.lhs);
                const uint32_t d = plan->ops[op.lhs].dim;
                const double g0 = gy[0];
                for (uint32_t k = 0; k < d; k++) ga[k] += g0;
                break;
            }
            case ESHKOL_STAGED_OP_SCALE: {
                double* ga = adj_ptr(op.lhs);
                for (uint32_t k = 0; k < op.dim; k++) ga[k] += gy[k] * op.scalar;
                break;
            }
        }
    }

    if (plan->total_params > 0) {
        std::memset(out_param_grads, 0, static_cast<size_t>(plan->total_params) * sizeof(double));
        for (uint32_t i = 0; i < N; i++) {
            if (plan->ops[i].op == ESHKOL_STAGED_OP_PARAM) {
                std::memcpy(out_param_grads + plan->ops[i].ext_offset,
                            adj_ptr(i),
                            static_cast<size_t>(plan->ops[i].dim) * sizeof(double));
            }
        }
    }
    return std::isfinite(*out_value);
}
