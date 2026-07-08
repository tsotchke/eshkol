/**
 * @file logic_builtins.cpp
 * @brief Tagged-value wrappers for logic/inference engine operations.
 *
 * Provides extern "C" functions for fg-marginal, fg-entropy, and kb-retract!
 * callable from LLVM-compiled code via the sret calling convention.
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/logic.h>
#include <eshkol/core/inference.h>
#include "../../lib/core/arena_memory.h"

#include <cmath>
#include <cstring>

extern "C" {

/** @brief Compute the marginal belief distribution for one variable of a
 *  factor graph and return it as a newly allocated tensor of probabilities.
 *  Converts the stored log-beliefs to probabilities via a numerically
 *  stable exp/normalize (subtract max log-belief before exponentiating).
 *  @param arena Arena used to allocate the result tensor.
 *  @param fg_tv Tagged heap pointer to the factor graph.
 *  @param idx_tv Tagged int index of the variable.
 *  @param result Out param: tensor of per-state probabilities, or NULL
 *  (ESHKOL_VALUE_NULL) if `fg_tv` is not a heap pointer or the index is
 *  out of range. */
void eshkol_fg_marginal_tagged(arena_t* arena,
                                const eshkol_tagged_value_t* fg_tv,
                                const eshkol_tagged_value_t* idx_tv,
                                eshkol_tagged_value_t* result) {
    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        result->type = ESHKOL_VALUE_NULL;
        result->data.raw_val = 0;
        return;
    }

    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)(uintptr_t)fg_tv->data.ptr_val;
    int64_t var_idx = idx_tv->data.int_val;

    if (!fg || var_idx < 0 || var_idx >= (int64_t)fg->num_vars) {
        result->type = ESHKOL_VALUE_NULL;
        result->data.raw_val = 0;
        return;
    }

    /* Get marginal beliefs for this variable — return as tensor */
    uint32_t n_states = fg->var_dims[var_idx];
    eshkol_tensor_t* t = arena_allocate_tensor_full(arena, 1, n_states);
    if (!t) {
        result->type = ESHKOL_VALUE_NULL;
        result->data.raw_val = 0;
        return;
    }

    /* Beliefs are log-probabilities → convert to probabilities via exp.
     * Normalize so they sum to 1. */
    double* log_beliefs = fg->beliefs[var_idx];
    double max_log = log_beliefs[0];
    for (uint32_t i = 1; i < n_states; i++) {
        if (log_beliefs[i] > max_log) max_log = log_beliefs[i];
    }
    double sum = 0.0;
    for (uint32_t i = 0; i < n_states; i++) {
        double p = exp(log_beliefs[i] - max_log);
        sum += p;
        /* Store as int64 bit pattern of double */
        int64_t bits;
        memcpy(&bits, &p, sizeof(double));
        t->elements[i] = bits;
    }
    /* Normalize */
    for (uint32_t i = 0; i < n_states; i++) {
        double p;
        memcpy(&p, &t->elements[i], sizeof(double));
        p /= sum;
        int64_t bits;
        memcpy(&bits, &p, sizeof(double));
        t->elements[i] = bits;
    }

    result->type = ESHKOL_VALUE_HEAP_PTR;
    result->flags = 0;
    result->reserved = 0;
    result->data.ptr_val = (uint64_t)t;
}

/** @brief Compute the Shannon entropy (in nats) of the marginal belief
 *  distribution for one variable of a factor graph.
 *  @param arena Unused (kept for calling-convention symmetry with the
 *  other tagged-value builtins).
 *  @param fg_tv Tagged heap pointer to the factor graph.
 *  @param idx_tv Tagged int index of the variable.
 *  @param result Out param: double entropy value, or 0.0 if `fg_tv` is
 *  not a heap pointer or the index is out of range. */
void eshkol_fg_entropy_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* fg_tv,
                               const eshkol_tagged_value_t* idx_tv,
                               eshkol_tagged_value_t* result) {
    (void)arena;

    if (fg_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        *result = eshkol_make_double(0.0);
        return;
    }

    eshkol_factor_graph_t* fg = (eshkol_factor_graph_t*)(uintptr_t)fg_tv->data.ptr_val;
    int64_t var_idx = idx_tv->data.int_val;

    if (!fg || var_idx < 0 || var_idx >= (int64_t)fg->num_vars) {
        *result = eshkol_make_double(0.0);
        return;
    }

    /* Convert log-beliefs to probabilities and compute Shannon entropy */
    uint32_t n_states = fg->var_dims[var_idx];
    double* log_beliefs = fg->beliefs[var_idx];
    double max_log = log_beliefs[0];
    for (uint32_t i = 1; i < n_states; i++) {
        if (log_beliefs[i] > max_log) max_log = log_beliefs[i];
    }
    double sum = 0.0;
    double probs[256]; /* max states */
    for (uint32_t i = 0; i < n_states && i < 256; i++) {
        probs[i] = exp(log_beliefs[i] - max_log);
        sum += probs[i];
    }
    double entropy = 0.0;
    for (uint32_t i = 0; i < n_states && i < 256; i++) {
        double p = probs[i] / sum;
        if (p > 1e-15) {
            entropy -= p * log(p);
        }
    }

    *result = eshkol_make_double(entropy);
}

/** @brief Remove a fact from a knowledge base by pointer identity,
 *  shifting later facts down to fill the gap.
 *  @param arena Unused (kept for calling-convention symmetry with the
 *  other tagged-value builtins).
 *  @param kb_tv Tagged heap pointer to the knowledge base.
 *  @param fact_tv Tagged heap pointer to the fact to remove.
 *  @param result Out param: tagged bool, true if the fact was found and
 *  removed, false otherwise (including if either argument is not a
 *  heap pointer). */
void eshkol_kb_retract_tagged(arena_t* arena,
                               const eshkol_tagged_value_t* kb_tv,
                               const eshkol_tagged_value_t* fact_tv,
                               eshkol_tagged_value_t* result) {
    (void)arena;

    if (kb_tv->type != ESHKOL_VALUE_HEAP_PTR || fact_tv->type != ESHKOL_VALUE_HEAP_PTR) {
        result->type = ESHKOL_VALUE_BOOL;
        result->data.raw_val = 0;
        return;
    }

    eshkol_knowledge_base_t* kb = (eshkol_knowledge_base_t*)(uintptr_t)kb_tv->data.ptr_val;
    eshkol_fact_t* fact = (eshkol_fact_t*)(uintptr_t)fact_tv->data.ptr_val;

    if (!kb || !fact) {
        result->type = ESHKOL_VALUE_BOOL;
        result->data.raw_val = 0;
        return;
    }

    /* Remove fact by pointer identity */
    bool found = false;
    for (uint32_t i = 0; i < kb->num_facts; i++) {
        if (kb->facts[i] == fact) {
            for (uint32_t j = i; j < kb->num_facts - 1; j++) {
                kb->facts[j] = kb->facts[j + 1];
            }
            kb->num_facts--;
            found = true;
            break;
        }
    }

    result->type = ESHKOL_VALUE_BOOL;
    result->flags = 0;
    result->reserved = 0;
    result->data.raw_val = found ? 1 : 0;
}

} /* extern "C" */
