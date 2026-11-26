/*
 * Colored Differentiable E-Graph (diff-egglog + SmoothE Cost Extraction)
 *
 * Implements:
 *   - Hash-consed e-nodes with union-find e-classes and exact congruence closure
 *   - Hierarchical colored assumption layers (color 0 = unconditional base;
 *     child colors inherit ancestor merges with zero leakage to base or siblings)
 *   - SmoothE differentiable soft-minimum e-class extraction with exact
 *     analytical reverse-mode gradients w.r.t. e-node costs (zero FD evaluations)
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_DIFF_EGGLOG_H
#define ESHKOL_CORE_DIFF_EGGLOG_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_egraph eshkol_egraph_t;

typedef struct eshkol_smoothe_result {
    double root_cost;      /* SmoothE softmin extracted cost of root e-class */
    uint32_t steps_used;   /* Bellman unrolling steps executed */
    uint64_t fd_evals;     /* Always 0 (exact analytical reverse-mode gradients) */
    bool valid;
} eshkol_smoothe_result_t;

/* Create an empty colored e-graph with base color 0. */
eshkol_egraph_t* eshkol_egraph_create(void);
void eshkol_egraph_destroy(eshkol_egraph_t* eg);

/* Create a new assumption color inheriting merges from `parent_color` (0 = base).
 * Returns new color id (> 0), or UINT32_MAX on invalid parent_color. */
uint32_t eshkol_egraph_create_color(eshkol_egraph_t* eg, uint32_t parent_color);

/* Add (or base-color hash-cons) an e-node (op_id, children[], cost).
 * Returns canonical e-class id (equal to newly created node id when fresh),
 * or UINT32_MAX on invalid input. */
uint32_t eshkol_egraph_add_node(eshkol_egraph_t* eg,
                                uint32_t op_id,
                                const uint32_t* children,
                                uint32_t num_children,
                                double cost);

/* Update the differentiable local cost of e-node `node_id`. */
bool eshkol_egraph_set_node_cost(eshkol_egraph_t* eg, uint32_t node_id, double cost);

/* Record an equivalence merge between `class_a` and `class_b` inside `color_id`. */
bool eshkol_egraph_union(eshkol_egraph_t* eg,
                         uint32_t class_a,
                         uint32_t class_b,
                         uint32_t color_id);

/* Run exact congruence closure saturation across all colors. */
bool eshkol_egraph_rebuild(eshkol_egraph_t* eg);

/* Query canonical representative of `eclass` inside `color_id`. */
uint32_t eshkol_egraph_find(eshkol_egraph_t* eg, uint32_t eclass, uint32_t color_id);

/* Check whether `a` and `b` are equivalent inside `color_id`. */
bool eshkol_egraph_equiv(eshkol_egraph_t* eg,
                         uint32_t a,
                         uint32_t b,
                         uint32_t color_id);

/*
 * SmoothE differentiable extraction for `root_eclass` in `color_id` at
 * temperature `tau > 0` with `num_iters` Bellman steps (e.g., 16..64).
 * Writes exact analytical gradient d(root_cost)/d(node_cost[i]) into
 * `out_node_grads[0..num_nodes-1]` when non-NULL (zero FD evaluations).
 */
bool eshkol_egraph_smoothe_extract(eshkol_egraph_t* eg,
                                   uint32_t root_eclass,
                                   uint32_t color_id,
                                   double temperature,
                                   uint32_t num_iters,
                                   eshkol_smoothe_result_t* out_result,
                                   double* out_node_grads,
                                   uint32_t grads_len);

/* Read-only inspection helpers (used by persistence & tests). */
uint32_t eshkol_egraph_num_colors(const eshkol_egraph_t* eg);
uint32_t eshkol_egraph_color_parent(const eshkol_egraph_t* eg, uint32_t color_id);
uint32_t eshkol_egraph_num_nodes(const eshkol_egraph_t* eg);
bool eshkol_egraph_inspect_node(const eshkol_egraph_t* eg,
                                uint32_t node_id,
                                uint32_t* out_op_id,
                                const uint32_t** out_children,
                                uint32_t* out_num_children,
                                double* out_cost);
uint32_t eshkol_egraph_num_merges(const eshkol_egraph_t* eg);
bool eshkol_egraph_inspect_merge(const eshkol_egraph_t* eg,
                                 uint32_t merge_idx,
                                 uint32_t* out_class_a,
                                 uint32_t* out_class_b,
                                 uint32_t* out_color_id);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_DIFF_EGGLOG_H */
