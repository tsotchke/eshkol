/**
 * @file device_op_emission.h
 * @brief Emitting ONE device operation into an open StableHLO function, and
 *        the shape it produces.
 *
 * WHY THIS IS A HEADER AND NOT PRIVATE TO device_lowering.cpp.
 *
 * DeviceExecutor::run() answers "compute this one op on the device" and
 * builds one `func.func @main` per op. A region is several ops in ONE
 * function, so region formation cannot go through that interface — but every
 * node inside a region must be emitted EXACTLY the way the single-op path
 * emits it. If it were not, the per-op parity measurements
 * (tests/xla/op_parity_test, tests/xla/builtin_parity_test) would silently
 * stop covering the ops that appear inside regions: the same builtin would
 * have two lowerings, one measured and one not, and the unmeasured one would
 * be the one real programs ran.
 *
 * So the emission is one function with two callers rather than two copies.
 * Both live in the MLIR-linked archive; neither is reachable from the slim
 * runtime, which is why this header is separate from device_lowering.h (that
 * one is included by the slim half and must not mention StableHLOEmitter).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_XLA_DEVICE_OP_EMISSION_H
#define ESHKOL_BACKEND_XLA_DEVICE_OP_EMISSION_H

#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/stablehlo_emitter.h"

namespace eshkol {
namespace xla {

/**
 * @brief The shape @p req produces, from its operand shapes.
 *
 * @return false with @p error set for an op/operand combination that has no
 *         result shape — a matmul of mismatched inner dimensions, a reduction
 *         over an axis that does not exist. This refuses rather than guesses,
 *         because a guessed shape produces a module that compiles and returns
 *         the wrong number of elements.
 */
bool inferDeviceResultShape(const DeviceOpRequest& req,
                            std::vector<int64_t>* out,
                            std::string* error);

/**
 * @brief Broadcast @p value from @p from to @p to, or return it unchanged
 *        when the shapes already agree.
 *
 * StableHLO's binary ops do not broadcast their operands, so this is what
 * makes an Eshkol elementwise op over differently shaped tensors a device
 * computation rather than a host pre-pass.
 */
void* alignDeviceOperand(StableHLOEmitter& emitter, void* value,
                         const std::vector<int64_t>& from,
                         const std::vector<int64_t>& to,
                         std::string* error);

/**
 * @brief Emit @p req over @p args into the emitter's currently open function.
 *
 * @param args One value per entry of req.operand_shapes, already in the
 *             function (a block argument, or the result of an earlier call to
 *             this function — which is what makes a region a chain).
 * @return The result value, or nullptr with @p error set.
 */
void* emitDeviceOp(StableHLOEmitter& emitter, const DeviceOpRequest& req,
                   const std::vector<void*>& args, std::string* error);

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_BACKEND_XLA_DEVICE_OP_EMISSION_H
