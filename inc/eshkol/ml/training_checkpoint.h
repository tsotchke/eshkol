/**
 * @file training_checkpoint.h
 * @brief Atomic ESKM checkpoints of the full mixed-curvature training state:
 *        parameters, optimizer moments, step count, curvature, and the RNG
 *        seed that reproduces the training batch. Restore-and-continue
 *        parity for the S9 stage of the XLA-to-TPU program.
 *
 * WHY A SEPARATE MODULE INSTEAD OF EXTENDING THE ESKM BINARY FORMAT.
 *
 * lib/ml/mixed_curvature_step.h's eshkol_mixed_curvature_init() generates the
 * training batch deterministically from one seed, and no step consumes any
 * further randomness afterward (docs/design/XLA_TRAINING_STEP.md section 2).
 * So the entire trajectory-determining state is: four parameter tensors,
 * eight moment tensors, the step count, the curvature, and that one seed --
 * every one of which already has a natural representation as a named f64
 * tensor in the existing ESKM tensor-list container that
 * lib/core/model_io.cpp's write_checkpoint()/parse_checkpoint() already read
 * and write. The container format is NOT missing a field for this job, so it
 * is not extended; this module writes the training state as one more named
 * tensor list through the same wire format (ESKM magic, format version 1,
 * CRC-32 footer), which keeps a training checkpoint loadable by the existing
 * `(model-load path)` builtin like any other model checkpoint.
 *
 * ATOMICITY.
 *
 * Both the checkpoint's .eskm payload and its .manifest sidecar are written
 * through lib/core/model_io_atomic.h (Gabriel Kahen's atomic-save helper,
 * PR #600): a uniquely created temporary file in the destination directory,
 * published by rename only after write, flush and close succeed. A reader
 * never observes a partially written file at the destination path.
 *
 * THE MANIFEST.
 *
 * <path>.manifest names the shape signature (n, d, c), the dtype policy
 * string, the step and seed recorded in the payload, and content_crc32 -- the
 * CRC-32 already computed while writing the payload, restated here so a
 * resume can validate a checkpoint's integrity from the small manifest file
 * before parsing the (potentially large) payload, and so a mismatch between
 * the two files alone is enough to refuse a checkpoint before ever opening
 * the payload.
 *
 * WHAT IS NOT BIT-EXACT ACROSS A RESTORE.
 *
 * The optimizer moments and parameters are restored bit-for-bit (raw f64
 * values, byte-identical round trip). What can differ is which libm the
 * restoring process links against, if it differs from the one that generated
 * the checkpoint (docs/design/XLA_PRODUCTION_TRAINING.md section on this).
 * eshkol_mixed_curvature_train_step performs the same floating-point
 * operations in the same order every time given the same inputs, so on one
 * platform / one libm, the post-restore trajectory equals the uninterrupted
 * one step for step. See docs/design/XLA_PRODUCTION_TRAINING.md.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_ML_TRAINING_CHECKPOINT_H
#define ESHKOL_ML_TRAINING_CHECKPOINT_H

#include "eshkol/ml/mixed_curvature_step.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief dtype policy recorded in the manifest. The ESKM payload is always
 *         f64 (model_io.cpp's only element dtype); this records what dtype
 *         policy PRODUCED the values being checkpointed, so a restore can
 *         tell an f32-host checkpoint from a bf16-mixed-device one apart. */
typedef enum {
    ESHKOL_TRAINING_DTYPE_F32 = 0,       /**< host reference step, f32 policy */
    ESHKOL_TRAINING_DTYPE_BF16_MIXED = 1 /**< device step, bf16-mixed policy (S7) */
} EshkolTrainingDtypePolicy;

/** @brief Everything eshkol_mixed_curvature_train_step needs to resume,
 *         plus the seed that reproduces the (otherwise unsaved) batch. */
typedef struct {
    EshkolMixedCurvatureShape shape;
    EshkolMixedCurvatureHyper hyper;
    uint64_t seed;
    EshkolTrainingDtypePolicy dtype_policy;
} EshkolTrainingCheckpointMeta;

/** @brief Save the full training state to @p path (ESKM payload) and
 *         @p path + ".manifest" (text sidecar), both published atomically.
 *
 * @return true if both files were written and committed; false on any
 *         allocation, I/O, or atomic-publish failure. On failure neither the
 *         payload nor the manifest at @p path is disturbed (the atomic
 *         helper aborts each side into its own temporary file).
 */
bool eshkol_training_checkpoint_save(const char* path,
                                     const EshkolTrainingCheckpointMeta* meta,
                                     const EshkolMixedCurvatureParams* params,
                                     const EshkolMixedCurvatureMoments* moments);

/** @brief Validate then load a training checkpoint written by
 *         eshkol_training_checkpoint_save().
 *
 * Validation, in order, any of which causes a refusal (false, no output
 * written): the manifest parses and its shape matches @p meta->shape (the
 * caller-declared shape a resuming process expects); the payload's CRC-32
 * matches the manifest's content_crc32; the payload's own trailing CRC-32
 * (independent of the manifest) matches its own bytes; every expected named
 * tensor is present with the expected element count.
 *
 * @param params, moments  Pre-allocated by the caller (buffer sizes from
 *                          eshkol_mixed_curvature_w_elements() etc.); filled
 *                          in place on success, left untouched on refusal.
 * @param out_meta          Filled with the manifest's recorded shape, hyper
 *                          (curvature only -- alpha/lr/etc. are a run
 *                          configuration, not trajectory state, and are the
 *                          resuming process's own to supply), seed and dtype
 *                          policy. May be null.
 * @return true if the checkpoint validated and was loaded; false if it is
 *         missing, corrupt, or shape-mismatched (a REFUSAL, not a crash).
 */
bool eshkol_training_checkpoint_load(const char* path,
                                     const EshkolMixedCurvatureShape* expected_shape,
                                     EshkolTrainingCheckpointMeta* out_meta,
                                     EshkolMixedCurvatureParams* params,
                                     EshkolMixedCurvatureMoments* moments);

/** @brief Validate a checkpoint's manifest + payload CRC without loading its
 *         tensors (the cheap check a resume runs over every candidate before
 *         picking the newest VALID one). Same validation order as
 *         eshkol_training_checkpoint_load(); returns false on ANY of the
 *         same refusal conditions, including a missing file. */
bool eshkol_training_checkpoint_is_valid(const char* path,
                                         const EshkolMixedCurvatureShape* expected_shape);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_ML_TRAINING_CHECKPOINT_H */
