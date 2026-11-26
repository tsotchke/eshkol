/*
 * Complementary Learning Systems (CLS) Memory: Hippocampal Episodic Z-Set Buffer
 * + Neocortical Semantic Network with Fisher-Protected Consolidation
 *
 * Strictly isolates raw episodic Z-set ingestion from neocortical semantic
 * weights so high-volume adversarial episodic writes produce zero drift on
 * semantic representations until guarded consolidation.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_CLS_ZSET_MEMORY_H
#define ESHKOL_CORE_CLS_ZSET_MEMORY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_cls_memory eshkol_cls_memory_t;

/*
 * Create a CLS memory module with an isolated hippocampal Z-set buffer and a
 * neocortical 2-layer tanh MLP (in_dim -> hidden_dim -> out_dim).
 */
eshkol_cls_memory_t* eshkol_cls_memory_create(uint32_t in_dim,
                                              uint32_t hidden_dim,
                                              uint32_t out_dim,
                                              uint64_t seed);
void eshkol_cls_memory_destroy(eshkol_cls_memory_t* cls);

/*
 * Train the neocortical semantic network on structured associations (xs, ys),
 * compute exact diagonal empirical Fisher information, and snapshot anchor
 * weights for protected consolidation.
 */
bool eshkol_cls_semantic_train(eshkol_cls_memory_t* cls,
                               const double* xs,
                               const double* ys,
                               size_t n_samples,
                               uint32_t epochs,
                               double lr);

/* Evaluate exact semantic MSE loss and analytical gradient L2 norm (fd_evals == 0). */
bool eshkol_cls_semantic_eval(const eshkol_cls_memory_t* cls,
                              const double* xs,
                              const double* ys,
                              size_t n_samples,
                              double* out_loss,
                              double* out_grad_norm,
                              uint64_t* out_fd_evals);

/*
 * Ingest a signed episodic delta into the isolated hippocampal Z-set buffer.
 * Zero-weight episodes are pruned immediately. Guaranteed NOT to mutate
 * neocortical semantic weights.
 */
bool eshkol_cls_episodic_ingest(eshkol_cls_memory_t* cls,
                                uint64_t episode_id,
                                const double* x,
                                const double* y,
                                int64_t weight_delta);

size_t eshkol_cls_episodic_active_count(const eshkol_cls_memory_t* cls);
uint64_t eshkol_cls_semantic_weights_checksum(const eshkol_cls_memory_t* cls);

/*
 * Guarded offline consolidation from the hippocampal Z-set buffer into the
 * neocortical network with diagonal Fisher EWC anchor regularization and
 * semantic-retention trust-region verification gate.
 */
bool eshkol_cls_consolidate_guarded(eshkol_cls_memory_t* cls,
                                    const double* anchor_xs,
                                    const double* anchor_ys,
                                    size_t n_anchor_samples,
                                    double lr,
                                    double fisher_lambda,
                                    double max_semantic_loss_delta,
                                    size_t* out_accepted_steps,
                                    size_t* out_rejected_steps);

/*
 * Unguarded direct-write baseline (bypasses CLS isolation & Fisher protection):
 * applies raw SGD directly to semantic weights for an episodic sample, used as
 * the catastrophic-forgetting negative control.
 */
bool eshkol_cls_unguarded_direct_step(eshkol_cls_memory_t* cls,
                                      const double* x,
                                      const double* y,
                                      double lr);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_CLS_ZSET_MEMORY_H */
