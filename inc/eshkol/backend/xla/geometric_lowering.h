/**
 * @file geometric_lowering.h
 * @brief The mixed-curvature model's geometric primitives, lowered to
 *        StableHLO as compositions of the ops in device_lowering.h.
 *
 * WHY THESE ARE COMPOSITIONS AND NOT OPS.
 *
 * A Poincare exponential map is not a hardware instruction and never will be.
 * It is a dot product, a square root, a tanh, a Mobius quotient and a handful
 * of scalar broadcasts. Lowering it means emitting THAT — one `func.func` per
 * primitive whose body is the decomposition — so the whole primitive, forward
 * and backward, is device code. The alternative, a host implementation with
 * device kernels underneath it, leaves the reverse pass on the host and leaves
 * XLA nothing to fuse.
 *
 * WHY CURVATURE IS AN OPERAND AND NOT A CONSTANT.
 *
 * Curvature enters every hyperbolic formula as a scalar. Baking it into the
 * module would be the obvious thing and it is wrong twice over:
 *
 *   - A mixed-curvature model LEARNS its curvature. A constant in the compiled
 *     executable would be the value curvature had when the graph was first
 *     traced, and training would silently optimise a parameter the device had
 *     already frozen.
 *   - The executable cache is keyed by shape. Two calls at the same shape and
 *     different K would collide, and the second would silently receive the
 *     first's curvature. That is a wrong number with nothing to see.
 *
 * So `c` is a rank-0 operand of every module that needs it, and one compiled
 * executable serves every curvature at a given dimension.
 *
 * WHAT CONVENTION THE FORMULAS ARE.
 *
 * The qLLM convention, transcribed from the exporters in tests/qllm_oracle/
 * that generated tests/qllm_oracle/golden/*.json (which are themselves
 * transcriptions of qLLM's own C and torch kernels). That is deliberate: the
 * golden corpus is an exact reverse-mode Jacobian for these exact operators,
 * so choosing any other convention would leave the lowering with no
 * independent reference to be graded against.
 *
 * The guards are PART of the operator, not sanitising. `clamp_min(a, eps)` is
 * `a < eps ? eps : a`, which is NOT `maximum(a, eps)` under Eshkol's tie
 * convention — max gives a tie to its right-hand operand, clamp_min keeps `a`
 * — so it is emitted as a select on the same comparison the reference uses.
 * The artanh clamp at 1 - 1e-7 is likewise the operator's, and above it the
 * true derivative is identically zero.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_XLA_GEOMETRIC_LOWERING_H
#define ESHKOL_BACKEND_XLA_GEOMETRIC_LOWERING_H

#include <cstdint>
#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/xla_types.h"

namespace eshkol {
namespace xla {

/**
 * @brief The geometric primitives that have a StableHLO lowering.
 *
 * Operand order is: every vector operand, in the order the formula names
 * them, then every scalar operand (rank-0), in the order the formula names
 * them. geometricOperandKinds() states it per primitive so no caller has to
 * remember.
 */
enum class GeometricPrimitive {
    // ----- hyperbolic: the Poincare ball of curvature -c, c > 0 -----
    /** x (+)_c y — Mobius addition. (x[d], y[d], c) -> [d] */
    MobiusAdd,
    /** exp_0(v) = v tanh(t)/t, t = sqrt(c)|v|. (v[d], c) -> [d] */
    PoincareExpMapOrigin,
    /** log_0(y) = y artanh(t)/t, t = sqrt(c)|y|, artanh clamped. (y[d], c) -> [d] */
    PoincareLogMapOrigin,
    /** exp_x(v), the Ganea formula. (x[d], v[d], c) -> [d] */
    PoincareExpMap,
    /** log_x(y), the Ganea formula. (x[d], y[d], c) -> [d] */
    PoincareLogMap,
    /** d_c(x,y) = acosh(1 + 2c|x-y|^2/((1-c|x|^2)(1-c|y|^2)))/sqrt(c).
     *  (x[d], y[d], c) -> rank-0 */
    HyperbolicDistance,
    /** Riemannian gradient rescaling: 0.25 conf^2 g, conf = clamp_min(1-c|x|^2, eps).
     *  (x[d], g[d], c, eps) -> [d] */
    PoincareProject,
    /** Radial clip back inside the ball. (x[d], step[d], c, eps) -> [d] */
    PoincareRetract,

    // ----- spherical: the unit sphere, curvature +1 -----
    /** Tangent projection g - <g,x> x. (x[d], g[d]) -> [d] */
    SphereProject,
    /** (x+step)/|x+step| with an eps guard. (x[d], step[d], eps) -> [d] */
    SphereRetract,
    /** exp_x(v) = cos|v| x + sin|v| v/|v|, on the unit sphere. (x[d], v[d]) -> [d] */
    SphereExpMap,
    /** log_x(y) = theta u/|u|, u = y - <x,y>x, theta = acos<x,y>. (x[d], y[d]) -> [d] */
    SphereLogMap,
    /** Great-circle distance acos(<x,y>/(|x||y|)). (x[d], y[d]) -> rank-0 */
    SphericalDistance,

    // ----- euclidean: curvature 0 -----
    /** exp_x(v) = x + v. (x[d], v[d]) -> [d] */
    EuclideanExpMap,
    /** log_x(y) = y - x. (x[d], y[d]) -> [d] */
    EuclideanLogMap,
    /** |x - y|. (x[d], y[d]) -> rank-0 */
    EuclideanDistance
};

/** @brief Stable name, used in cache keys, gate evidence and diagnostics. */
const char* geometricPrimitiveName(GeometricPrimitive p);

/** @brief How many [d]-shaped operands @p p takes, before its scalars. */
int geometricVectorOperands(GeometricPrimitive p);

/** @brief How many rank-0 operands @p p takes, after its vectors. */
int geometricScalarOperands(GeometricPrimitive p);

/** @brief True when @p p produces a rank-0 result rather than a [d] one. */
bool geometricResultIsScalar(GeometricPrimitive p);

/**
 * @brief Build the StableHLO module for @p p at dimension @p dim.
 *
 * @param p             Which primitive.
 * @param dim           d, the vector operands' single dimension.
 * @param elem          Element type the DEVICE computes in.
 * @param with_gradient When true the module's results are the cotangents of
 *                      the VECTOR operands (in order) and it takes one extra
 *                      trailing parameter, the upstream cotangent, shaped like
 *                      the result. When false the single result is the
 *                      primitive's value.
 * @param module_text   Set to the module's textual form on success.
 * @param error         Set to a diagnostic on failure.
 *
 * The gradient is with respect to the VECTOR operands only. Curvature and the
 * guard epsilons are not differentiated: Eshkol's host tape carries curvature
 * as a node PARAMETER rather than as a differentiable input (see
 * `node->params.curvature` in lib/bridge/qllm_bridge.cpp), so a device
 * cotangent for it would have nothing on the host to be graded against, and an
 * ungraded gradient is exactly what this program exists not to ship.
 */
bool buildGeometricModule(GeometricPrimitive p, int64_t dim, ElementType elem,
                          bool with_gradient, std::string* module_text,
                          std::string* error);

/**
 * @brief The cache key for that module.
 *
 * Includes the primitive, the dimension, the element type and the direction —
 * everything the module's TEXT depends on, and nothing else. Curvature is
 * absent on purpose: it is an operand, so two curvatures share one executable.
 */
std::string geometricCacheKey(GeometricPrimitive p, int64_t dim, ElementType elem,
                              bool with_gradient);

/**
 * @brief Evaluate @p p on the device.
 *
 * @param executor Device executor, from registerStableHLODeviceExecutor().
 * @param p        Which primitive.
 * @param dim      d.
 * @param operands One host f64 pointer per operand: the vector operands first,
 *                 then one single-element pointer per scalar operand.
 * @param result   Host f64 destination: `dim` elements, or 1 for a scalar
 *                 result. Untouched on failure.
 * @param error    Set to a diagnostic on failure.
 */
bool runGeometric(DeviceExecutor* executor, GeometricPrimitive p, int64_t dim,
                  const std::vector<const double*>& operands,
                  double* result, std::string* error);

/**
 * @brief Reverse-mode gradient of @p p on the device.
 *
 * @param cotangent Upstream cotangent, `dim` elements (1 for a scalar result).
 *                  nullptr means a ones seed.
 * @param gradients One destination per VECTOR operand, in order, each `dim`
 *                  elements. Untouched on failure.
 */
bool runGeometricGradient(DeviceExecutor* executor, GeometricPrimitive p, int64_t dim,
                          const std::vector<const double*>& operands,
                          const double* cotangent,
                          const std::vector<double*>& gradients,
                          std::string* error);

}  // namespace xla
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_XLA_GEOMETRIC_LOWERING_H
