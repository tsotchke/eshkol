/*
 * Staged Zero-Allocation Compiled Backward Graphs (Value + Gradient)
 *
 * Compiles differentiable DAGs (Forney factor graph energies, Einsum
 * probabilistic circuits, and energy-based MLP/transformer layers) once into a
 * static execution plan with precomputed byte offsets inside a caller-owned
 * pre-sized scratch arena.
 *
 * Execution performs zero heap allocations (zero malloc/new) and zero
 * finite-difference evaluations (exact analytical reverse-mode adjoints).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_STAGED_VALUE_GRAD_H
#define ESHKOL_CORE_STAGED_VALUE_GRAD_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Dedicated caller-owned pre-sized scratch arena (zero system heap allocations). */
typedef struct eshkol_staged_scratch_arena {
    uint8_t* buffer;
    size_t capacity;
    size_t offset;
    size_t high_water;
    uint64_t resets;
    uint64_t bump_allocs;
    uint64_t heap_allocs; /* Remains 0 during staged execution */
} eshkol_staged_scratch_arena_t;

bool eshkol_staged_scratch_init(eshkol_staged_scratch_arena_t* arena,
                                uint8_t* backing_buffer,
                                size_t capacity_bytes);

void eshkol_staged_scratch_reset(eshkol_staged_scratch_arena_t* arena);

void* eshkol_staged_scratch_alloc(eshkol_staged_scratch_arena_t* arena,
                                  size_t bytes,
                                  size_t alignment);

typedef enum eshkol_staged_op_kind {
    ESHKOL_STAGED_OP_INPUT     = 0,  /* External input vector slice */
    ESHKOL_STAGED_OP_PARAM     = 1,  /* Differentiable parameter vector/matrix slice */
    ESHKOL_STAGED_OP_MATVEC    = 2,  /* y = W * x (a=W [rows x cols], b=x [cols]) */
    ESHKOL_STAGED_OP_ADD       = 3,  /* y = a + b */
    ESHKOL_STAGED_OP_MUL       = 4,  /* y = a * b (Hadamard / Einsum PC product) */
    ESHKOL_STAGED_OP_TANH      = 5,  /* y = tanh(a) */
    ESHKOL_STAGED_OP_SOFTPLUS  = 6,  /* y = log(1 + exp(a)) */
    ESHKOL_STAGED_OP_LOGSUMEXP = 7,  /* scalar y = log(sum_i exp(a_i)) */
    ESHKOL_STAGED_OP_DOT       = 8,  /* scalar y = sum_i a_i * b_i */
    ESHKOL_STAGED_OP_MSE       = 9,  /* scalar y = 0.5 * sum_i (a_i - b_i)^2 */
    ESHKOL_STAGED_OP_SUM       = 10, /* scalar y = sum_i a_i */
    ESHKOL_STAGED_OP_SCALE     = 11  /* y = alpha * a */
} eshkol_staged_op_kind_t;

typedef struct eshkol_staged_builder eshkol_staged_builder_t;
typedef struct eshkol_staged_plan eshkol_staged_plan_t;

/* Create/destroy graph builder (allocates during staging/compile phase only). */
eshkol_staged_builder_t* eshkol_staged_builder_create(void);
void eshkol_staged_builder_destroy(eshkol_staged_builder_t* b);

/* Register input and parameter leaves (returns node id or UINT32_MAX). */
uint32_t eshkol_staged_add_input(eshkol_staged_builder_t* b, uint32_t dim);
uint32_t eshkol_staged_add_param(eshkol_staged_builder_t* b, uint32_t dim);

/* Add differentiable operations (returns node id or UINT32_MAX). */
uint32_t eshkol_staged_add_matvec(eshkol_staged_builder_t* b,
                                  uint32_t w_node, uint32_t x_node,
                                  uint32_t rows, uint32_t cols);
uint32_t eshkol_staged_add_binary(eshkol_staged_builder_t* b,
                                  eshkol_staged_op_kind_t op,
                                  uint32_t lhs_node, uint32_t rhs_node);
uint32_t eshkol_staged_add_unary(eshkol_staged_builder_t* b,
                                 eshkol_staged_op_kind_t op,
                                 uint32_t arg_node);
uint32_t eshkol_staged_add_scale(eshkol_staged_builder_t* b,
                                 uint32_t arg_node, double alpha);

/* Compile DAG rooted at scalar `loss_node` into an immutable zero-alloc execution plan. */
eshkol_staged_plan_t* eshkol_compile_staged_value_grad(const eshkol_staged_builder_t* b,
                                                       uint32_t loss_node);

void eshkol_staged_plan_destroy(eshkol_staged_plan_t* plan);

size_t eshkol_staged_plan_required_scratch(const eshkol_staged_plan_t* plan);
uint32_t eshkol_staged_plan_total_inputs(const eshkol_staged_plan_t* plan);
uint32_t eshkol_staged_plan_total_params(const eshkol_staged_plan_t* plan);

/*
 * Zero-allocation forward value + exact reverse-mode parameter gradient evaluation.
 * All temporary value and adjoint buffers are carved solely from `scratch_arena`.
 * Performs zero system heap allocations and sets `*out_fd_evals = 0`.
 */
bool eshkol_staged_eval_value_grad(const eshkol_staged_plan_t* plan,
                                   eshkol_staged_scratch_arena_t* scratch_arena,
                                   const double* inputs_flat,
                                   uint32_t inputs_len,
                                   const double* params_flat,
                                   uint32_t params_len,
                                   double* out_value,
                                   double* out_param_grads,
                                   uint32_t grads_len,
                                   uint64_t* out_fd_evals);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_STAGED_VALUE_GRAD_H */
