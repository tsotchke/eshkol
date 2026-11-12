/**
 * @file saturating_vjp.h
 * @brief One reverse-mode rule for the two saturating nonlinearities of the
 *        SDNC weight machine: the logistic gate and the attention softmax.
 *
 * Both the runtime weight kernels (lib/core/sdnc_core.h) and the training-mode
 * backward module (lib/backend/qllm_backward.c) differentiate the same
 * gated-sigmoid FFN and the same Layer-0 softmax. They take the vector-Jacobian
 * product from here, so the rule exists once.
 *
 * Each rule carries an inverse temperature `beta`:
 *
 *   logistic gate   s = sigma(beta * pre)
 *                   d pre = upstream * beta * s * (1 - s)
 *
 *   softmax         a = softmax(beta * z)
 *                   d z_p = beta * a_p * (da_p - sum_q a_q * da_q)
 *
 * With beta = 1 and `s` / `a` the forward values these are the exact
 * derivatives, evaluated in the same operation order as before the rule was
 * shared, so results are bit-identical. With beta < 1 and `s` / `a` evaluated
 * at the tempered pre-activation, they are the Jacobian of the
 * logistic-perturbed gate (and the tempered softmax): the expectation of a hard
 * threshold under logistic noise is sigma(beta * pre). That is what lets a
 * caller keep an exact, saturated forward pass and still obtain a non-zero,
 * informative derivative for control parameters whose exact derivative is
 * identically zero.
 *
 * The rules are precision-generic: ESHKOL_SATURATING_VJP_RULES(REAL, SUFFIX)
 * instantiates them for one scalar type (float for the runtime kernels,
 * `qllm_real` for qllm_backward.c, which the gradient-check test builds in
 * double).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_CORE_SATURATING_VJP_H
#define ESHKOL_CORE_SATURATING_VJP_H

#define ESHKOL_SATURATING_VJP_RULES(REAL, SUFFIX)                              \
    /** d pre for s = sigma(beta * pre); `s` is the gate value at `beta`. */  \
    static inline REAL eshkol_logistic_vjp##SUFFIX(REAL upstream, REAL s,     \
                                                  REAL beta) {                \
        return upstream * beta * s * ((REAL)1 - s);                           \
    }                                                                          \
    /** d z for a = softmax(beta * z); `a` holds the probabilities at `beta`. \
     *  Writes n entries of `dz`. */                                          \
    static inline void eshkol_softmax_vjp##SUFFIX(const REAL* a,              \
                                                 const REAL* da, int n,       \
                                                 REAL beta, REAL* dz) {       \
        REAL dot = 0;                                                          \
        for (int p = 0; p < n; p++) dot += a[p] * da[p];                       \
        for (int p = 0; p < n; p++) dz[p] = beta * a[p] * (da[p] - dot);       \
    }

#endif /* ESHKOL_CORE_SATURATING_VJP_H */
