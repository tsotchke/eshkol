/**
 * @file training_step_lowering.h
 * @brief One full training step of the mixed-curvature model, lowered as a
 *        single StableHLO program: forward, backward and optimizer update in
 *        one `func.func @main`.
 *
 * WHY ONE PROGRAM AND NOT THREE.
 *
 * A device forward followed by a host backward followed by a host optimizer is
 * three round trips per step, and each one is a place where the device's
 * numbers and the host's diverge for reasons that have nothing to do with the
 * model. More to the point it is not training on the device at all: the
 * parameters would live on the host and the device would be a matrix
 * accelerator. XLA also cannot fuse across a round trip, so the shape of the
 * program it is given would be the shape of the host's control flow rather
 * than the shape of the computation.
 *
 * So the whole step is one module. Its inputs are the parameters, the
 * optimizer moments, the batch, the step count, the curvature and the
 * hyperparameters; its outputs are the updated parameters, the updated moments
 * and the loss. Nothing crosses back to the host in between.
 *
 * WHERE THE BACKWARD COMES FROM.
 *
 * `StableHLOEmitter::emitVJP` over the forward graph, with no VJP rule written
 * for this step anywhere. The host's backward (lib/ml/mixed_curvature_step.cpp)
 * is instead derived by hand, stage by stage. That the two agree over five
 * steps is the claim `xla_training_step_parity` makes; if they were both
 * automatic, or both hand-written from the same notes, there would be no claim.
 *
 * WHY CURVATURE, LEARNING RATE AND STEP COUNT ARE OPERANDS.
 *
 * Exactly the reason geometric_lowering.h gives for curvature, extended to the
 * two other values that a shape-keyed cache would otherwise freeze. The cache
 * key here is `train|n|d|c|dtype` — the step index is NOT in it, so a second
 * step at the same shape is a cache hit and Adam's bias corrections are
 * computed on the device from the step operand with `stablehlo.power`. Baking
 * the step index in would mean one compile per step, which is both slow and a
 * cache that never hits, i.e. a cache whose correctness nothing tests.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_XLA_TRAINING_STEP_LOWERING_H
#define ESHKOL_BACKEND_XLA_TRAINING_STEP_LOWERING_H

#include <cstdint>
#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/xla_types.h"
#include "eshkol/ml/mixed_curvature_step.h"

namespace eshkol {
namespace xla {

/**
 * @brief The module's parameters, in the order `@main` declares them.
 *
 * Stated as an enum rather than left to a comment because the operand vector,
 * the shape vector and the emitter's argument list must agree three times over
 * and a mismatch between them is a wrong number, not a compile error.
 */
enum TrainingStepInput {
    kTsW = 0,      ///< [d,d]
    kTsPHyp,       ///< [c,d]
    kTsPSph,       ///< [c,d]
    kTsPEuc,       ///< [c,d]
    kTsMW,         ///< [d,d]
    kTsVW,         ///< [d,d]
    kTsMHyp,       ///< [c,d]
    kTsVHyp,       ///< [c,d]
    kTsMSph,       ///< [c,d]
    kTsVSph,       ///< [c,d]
    kTsMEuc,       ///< [c,d]
    kTsVEuc,       ///< [c,d]
    kTsBatch,      ///< [n,d]
    kTsTargets,    ///< [n,c]
    kTsCurvature,  ///< rank-0, from here down
    kTsAlpha,
    kTsLr,
    kTsBeta1,
    kTsBeta2,
    kTsAdamEps,
    kTsGuardEps,
    kTsStep,       ///< the count AFTER incrementing, as a real
    kTsWHyp,
    kTsWSph,
    kTsWEuc,
    kTsInputCount
};

/** @brief The module's results, in the order `@main` returns them. */
enum TrainingStepOutput {
    kTsOutW = 0,
    kTsOutPHyp,
    kTsOutPSph,
    kTsOutPEuc,
    kTsOutMW,
    kTsOutVW,
    kTsOutMHyp,
    kTsOutVHyp,
    kTsOutMSph,
    kTsOutVSph,
    kTsOutMEuc,
    kTsOutVEuc,
    kTsOutLoss,    ///< rank-0
    kTsOutputCount
};

/** @brief Operand shapes for @p s, in TrainingStepInput order. */
std::vector<std::vector<int64_t>> trainingStepInputShapes(EshkolMixedCurvatureShape s);

/** @brief Result shapes for @p s, in TrainingStepOutput order. */
std::vector<std::vector<int64_t>> trainingStepOutputShapes(EshkolMixedCurvatureShape s);

/**
 * @brief The cache key: `train|n|d|c|dtype`.
 *
 * Curvature, learning rate and step index are absent on purpose — they are
 * operands, so one executable serves every curvature and every step at a shape.
 */
std::string trainingStepCacheKey(EshkolMixedCurvatureShape s, ElementType elem);

/**
 * @brief Build the module.
 *
 * @param s           Shape signature.
 * @param elem        Element type the DEVICE computes in.
 * @param module_text Set to the module's textual form on success.
 * @param error       Set to a diagnostic on failure. A missing VJP rule is
 *                    reported by name here rather than producing a module with
 *                    a hole in its gradient.
 */
bool buildTrainingStepModule(EshkolMixedCurvatureShape s, ElementType elem,
                             std::string* module_text, std::string* error);

/**
 * @brief Run one training step on the device.
 *
 * @param params   Updated in place from the device's results. Untouched on failure.
 * @param moments  Updated in place, including `step`. Untouched on failure.
 * @param out_loss The loss BEFORE this step's update, i.e. of the parameters
 *                 that produced the gradient — matching
 *                 eshkol_mixed_curvature_train_step's contract exactly, since
 *                 a harness comparing the two would otherwise be comparing
 *                 losses one step apart. Optional.
 *
 * There is no partial success: the caller's parameters are written only once
 * every result has come back, because a step that updated three of four
 * parameters is a model that still runs and is wrong.
 */
bool runTrainingStep(DeviceExecutor* executor,
                     EshkolMixedCurvatureShape s,
                     const EshkolMixedCurvatureHyper& hyper,
                     EshkolMixedCurvatureParams* params,
                     EshkolMixedCurvatureMoments* moments,
                     const double* batch,
                     const double* targets,
                     double* out_loss,
                     std::string* error);

}  // namespace xla
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_XLA_TRAINING_STEP_LOWERING_H
