/**
 * @file region_execution.h
 * @brief Compiling one outlined region into a StableHLO function and running
 *        it on the device.
 *
 * Region formation decides WHICH subgraph goes to the device and reports what
 * did not. This is the other half: taking that subgraph and making it a
 * `func.func @main` whose parameters are the region's live inputs and whose
 * result is the value the subtree produced, then executing it through the
 * PJRT path the rest of this backend already uses.
 *
 * TWO PROPERTIES THIS FILE EXISTS TO HOLD.
 *
 * ONE FUNCTION, NOT ONE PER OP. The whole point of a region is that the ops
 * inside it are compiled and run together; a chain emitted as N modules is
 * what the S2 per-op path already did, and it pays a host round trip between
 * every pair of ops. So every node of the region emits into the SAME open
 * function, and only the region's result leaves it.
 *
 * EMITTED THE SAME WAY THE MEASURED PATH EMITS. Each node goes through
 * emitDeviceOp() from device_op_emission.h, which is the identical function
 * DeviceExecutor::run() uses for a single op. A region therefore cannot
 * contain a lowering that no parity row has measured, and a correction to an
 * op's lowering reaches both paths at once.
 *
 * THE CACHE. A module is compiled once per (region, shapes) pair, keyed by the
 * region's shape signature — the same string the report prints. That is what
 * makes a region whose shapes are only known at run time compile once per
 * distinct shape rather than once per call.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_XLA_REGION_EXECUTION_H
#define ESHKOL_BACKEND_XLA_REGION_EXECUTION_H

#include <map>
#include <string>
#include <vector>

#include "eshkol/backend/xla/region_formation.h"

namespace eshkol {
namespace xla {

/** @brief One value entering a region from the host. */
struct RegionOperand {
    std::vector<int64_t> shape;   ///< Empty means rank 0
    const double* data = nullptr; ///< Host f64, row-major, shape-many elements
};

/**
 * @brief Compiles and runs outlined regions.
 *
 * Not thread-safe; one per compilation or per harness run. The executable
 * cache underneath it (in the installed DeviceExecutor) is shared and is
 * keyed by the module text's cache key, so two RegionExecutors that emit the
 * same module for the same shapes share the compiled executable.
 */
class RegionExecutor {
public:
    /**
     * @param functions The module's top-level functions, so a call to one that
     *        region formation counted as inlined is emitted as inlined here
     *        too. Both sides read the same table; a region whose op list the
     *        report shows as six ops emits six ops.
     */
    explicit RegionExecutor(const std::map<std::string, RegionFunction>& functions);
    ~RegionExecutor();

    RegionExecutor(const RegionExecutor&) = delete;
    RegionExecutor& operator=(const RegionExecutor&) = delete;

    /**
     * @brief Emit @p region as a StableHLO module for these operand shapes.
     *
     * Separated from execution because the emitted text is checkable without
     * a device: a machine with no accelerator can still assert that a region
     * became the function it was supposed to become.
     *
     * @param operand_shapes One per region input, in the report's input order.
     * @param module_text    The `func.func @main` on success.
     * @param result_shape   The shape @main returns.
     * @return false with @p error naming the node that could not be emitted.
     *         A region this refuses is a graph break that region formation
     *         should have reported and did not, so the error names the
     *         construct rather than saying "unsupported".
     */
    bool buildModule(const Region& region,
                     const std::vector<std::vector<int64_t>>& operand_shapes,
                     std::string* module_text,
                     std::vector<int64_t>* result_shape,
                     std::string* error);

    /**
     * @brief Emit @p region as forward pass AND its reverse-mode VJP in one
     *        module: (inputs..., cotangent) -> (d/d input...).
     *
     * A region inside a `gradient` operand must go this way. Executing only
     * its forward pass on the device would leave the host tape with no
     * derivative for those ops and train a model on a gradient that is
     * missing terms — silently, which is why a region whose VJP cannot be
     * built is refused here rather than downgraded to a forward-only run.
     */
    bool buildGradientModule(const Region& region,
                             const std::vector<std::vector<int64_t>>& operand_shapes,
                             std::string* module_text,
                             std::vector<int64_t>* result_shape,
                             std::string* error);

    /**
     * @brief Run @p region on the device over @p operands.
     *
     * @param result Host f64 destination, at least as large as the region's
     *               result. Untouched on failure.
     * @return false with @p error set when no device is installed, when the
     *         region could not be emitted, or when the module did not compile
     *         or execute. There is no fallback to the host here: a caller that
     *         wants the host answer asks for it, and a silent fallback is the
     *         defect this stage exists to make impossible.
     */
    bool execute(const Region& region,
                 const std::vector<RegionOperand>& operands,
                 double* result,
                 std::vector<int64_t>* result_shape,
                 std::string* error);

    /** @brief Modules emitted, and modules that reached the device. */
    uint64_t modulesBuilt() const;
    uint64_t executions() const;

private:
    class Impl;
    Impl* impl_;
};

/**
 * @brief Register @p region so that generated code can reach it by id.
 *
 * The id is what codegen bakes into the eshkol_xla_region() call it emits
 * where the outlined subtree used to be. Registration also installs the
 * region runner on first use, which is what joins the slim runtime's entry
 * point to this MLIR-linked half.
 *
 * @param region    Not copied. It must outlive every execution, which for a
 *                  compilation means the AST it points into must too.
 * @param functions The module's top-level functions, same lifetime rule.
 * @return The region id, or -1 if the region cannot be registered.
 */
int64_t registerRegionForExecution(const Region& region,
                                   const std::map<std::string, RegionFunction>& functions);

/** @brief Forget every registered region. For a harness that compiles more
 *         than one program in one process. */
void clearRegisteredRegions();

/** @brief How many registered regions have executed, and how many refused. */
void registeredRegionStats(uint64_t* executed, uint64_t* failed);

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_BACKEND_XLA_REGION_EXECUTION_H
