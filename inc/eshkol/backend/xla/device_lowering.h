/**
 * @file device_lowering.h
 * @brief The seam between Eshkol's host tensor runtime and real device
 *        execution of StableHLO through PJRT.
 *
 * WHY THIS FILE EXISTS, AND WHY IT IS AN INTERFACE RATHER THAN A CALL.
 *
 * Before this file, every `eshkol_xla_*` entry point in xla_runtime.cpp — the
 * functions generated code actually calls — computed its answer on the host:
 * BLAS, SIMD, or the GPU kernels in lib/backend/gpu. StableHLOEmitter could
 * build a module and PjrtClient could compile and run one, but nothing joined
 * the two: no Eshkol tensor expression had ever been evaluated by XLA.
 *
 * Joining them directly is not possible, and the reason is a link-time one
 * rather than a matter of taste.
 *
 *   - `xla_runtime.cpp` and `pjrt_client.cpp` are deliberately members of the
 *     SLIM runtime archive (`libeshkol-runtime.a`; see the ESHKOL_XLA_RUNTIME_SRC
 *     block in CMakeLists.txt). They must be, because AOT-compiled user
 *     binaries link that archive alone and have to resolve
 *     `eshkol_xla_elementwise` and friends. Neither depends on MLIR.
 *
 *   - `stablehlo_emitter.cpp` depends on MLIR and the StableHLO dialect
 *     libraries. It lives in `libeshkol-static.a`, which is linked only by the
 *     compiler executables and by tests, never by an AOT user binary.
 *
 * So if `eshkol_xla_elementwise` referenced StableHLOEmitter directly, every
 * AOT link would need the whole MLIR/StableHLO library set on its command
 * line. That is the failure the ESHKOL_XLA_RUNTIME_SRC comment in
 * CMakeLists.txt already records having been fixed once; reintroducing it
 * through a new call edge would be the same bug with a different cause.
 *
 * The interface below is how the two halves meet without that edge:
 *
 *   - `DeviceExecutor` is a pure interface over plain C++ types (no MLIR, no
 *     PJRT types) declared here and consumed by the slim half.
 *   - `deviceExecutor()` / `setDeviceExecutor()` are DEFINED IN
 *     xla_runtime.cpp, i.e. in the slim archive, so the pointer that holds the
 *     installed executor is always present and starts null.
 *   - `registerStableHLODeviceExecutor()` is DEFINED IN device_lowering.cpp,
 *     i.e. in the MLIR-linked archive. Calling it is what installs the
 *     executor, and a link that does not call it never pulls that translation
 *     unit — or MLIR — in.
 *
 * The consequence is precise and intended: a binary that links MLIR (the
 * compiler, the REPL/JIT path, the parity harness) can execute tensor ops on a
 * PJRT device; a bare AOT binary that links only the slim runtime keeps
 * exactly the host behaviour it had before this file existed, because the
 * executor pointer is still null there. Nothing silently changes for anyone
 * who did not ask for a device.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_BACKEND_XLA_DEVICE_LOWERING_H
#define ESHKOL_BACKEND_XLA_DEVICE_LOWERING_H

#include <cstdint>
#include <string>
#include <vector>

namespace eshkol {
namespace xla {

/**
 * @brief The tensor operations that have a StableHLO lowering.
 *
 * This is deliberately a separate enum from XLACodegen's ElementwiseOp /
 * ReduceOp rather than a reuse of them. Those two are numbered by the ABI the
 * generated LLVM IR passes as an integer op-code, and their numbering is
 * therefore frozen by every already-compiled object file. This enum is
 * internal to the device path and can grow in any order as ops are lowered.
 * Translation between the two happens once, in xla_runtime.cpp, where both
 * are already in scope.
 */
enum class DeviceOpKind {
    // Elementwise binary. Operands may have different shapes: a request whose
    // operand shapes differ from result_shape is lowered as an explicit
    // stablehlo.broadcast_in_dim on each operand followed by the binary op,
    // which is what makes the broadcast case a device computation rather than
    // a host pre-pass.
    Add,
    Subtract,
    Multiply,
    Divide,

    // Elementwise unary.
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,

    // Elementwise unary, added for the geometric primitives: every one of
    // these appears in the Poincare-ball and sphere formulas (a norm is a
    // sqrt, a conformal factor is a reciprocal, a Mobius quotient needs an
    // absolute value, a log map needs an artanh).
    Sqrt,
    Rsqrt,
    Abs,
    Negate,
    // Relu is maximum(x, 0) and Sigmoid is stablehlo.logistic. They are named
    // as ops here rather than left to the caller because the
    // eshkol_xla_elementwise ABI already numbers them as ops (codes 9 and 10),
    // and the device path has to answer at that same seam.
    Relu,
    Sigmoid,
    // Emitted as 0.5*(log(1+x) - log(1-x)); StableHLO has no artanh op. See
    // StableHLOEmitter::emitAtanh().
    Atanh,

    // Numerically stable softmax over `axes` (empty means every axis):
    // exp(x - max(x)) / sum(exp(x - max(x))).
    Softmax,

    // Elementwise binary. Operand shapes broadcast exactly as the four above.
    Pow,
    Maximum,
    Minimum,

    // Elementwise ternary: clamp(lo, x, hi), operands in that order. Emitted
    // as maximum(minimum(x, hi), lo) so that the gradient's tie convention at
    // a bound is the host's max/min convention by construction rather than a
    // second rule saying the same thing. See StableHLOEmitter::emitClamp().
    Clamp,

    // Matrix. Rank-2 only: operand_shapes[0] is [M,K], operand_shapes[1] is
    // [K,N], result_shape is [M,N].
    Matmul,

    // Shape. Transpose takes its permutation in `axes`; Reshape takes its
    // target shape in result_shape; Broadcast takes, in `axes`, the result
    // dimension each operand dimension maps to.
    Transpose,
    Reshape,
    Broadcast,

    // Reduction over `axes`. An empty `axes` means every axis (a full
    // reduction to a rank-0 result).
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,
    ReduceProd
};

/**
 * @brief Human-readable name for @p kind, used in cache keys and diagnostics.
 *
 * DEFINED IN xla_runtime.cpp, with the rest of the seam, because the slim
 * runtime archive's fallback diagnostics name the op — see the note on the
 * definition itself.
 */
const char* deviceOpKindName(DeviceOpKind kind);

/**
 * @brief One device computation request: what to compute, over what shapes.
 *
 * Shapes are row-major extents, matching Eshkol's tensor layout. `result_shape`
 * is stated by the caller rather than inferred here so that the shape the
 * device produces and the shape the host allocator already committed to are
 * checked against each other instead of being assumed equal — a mismatch is an
 * error the executor reports, never a silent reinterpretation of the bytes.
 *
 * An empty `result_shape` denotes a rank-0 (single element) result.
 */
struct DeviceOpRequest {
    DeviceOpKind kind = DeviceOpKind::Add;
    std::vector<std::vector<int64_t>> operand_shapes;
    std::vector<int64_t> result_shape;
    std::vector<int64_t> axes;
};

/** @brief Counters for what the device path actually did, for gates and tests. */
struct DeviceStats {
    uint64_t executed = 0;     ///< Requests that ran on the device and returned a result
    uint64_t compiled = 0;     ///< StableHLO modules handed to PJRT compile()
    uint64_t cache_hits = 0;   ///< Requests served by an already-compiled executable
    uint64_t failures = 0;     ///< Requests that could not run on the device
};

/**
 * @brief Executes one tensor operation on a PJRT device.
 *
 * Host data is f64 on both sides of this interface, because that is what every
 * Eshkol tensor is (see the dtype note at the top of xla_runtime.cpp). The
 * element type the DEVICE computes in is a separate choice — TPU has no f64 —
 * and is reported by dtypeName(); converting between the two is the executor's
 * job, not the caller's, so no caller has to know which device it is on.
 */
class DeviceExecutor {
public:
    virtual ~DeviceExecutor() = default;

    /**
     * @brief Whether a device is actually usable right now.
     * @param why Set to the reason when this returns false. Never set on true.
     *
     * This is a real check (plugin loaded, client created, addressable device
     * present, StableHLO emitter compiled in), not a compile-time constant, so
     * a caller can report why a requested device did not materialise instead
     * of quietly computing on the host.
     */
    virtual bool available(std::string* why) = 0;

    /**
     * @brief Compute @p request on the device.
     *
     * @param request  What to compute.
     * @param operands One host f64 pointer per entry of request.operand_shapes.
     * @param result   Host f64 destination, at least as many elements as
     *                 request.result_shape denotes. Untouched on failure.
     * @param error    Set to a diagnostic when this returns false.
     *
     * @return true only if the module compiled, executed, and the result was
     *         copied back in full. Any other outcome is false with @p error
     *         set — there is no partial success, because a caller that used a
     *         partially written result would produce wrong numbers rather than
     *         a visible failure.
     */
    virtual bool run(const DeviceOpRequest& request,
                     const std::vector<const double*>& operands,
                     double* result,
                     std::string* error) = 0;

    /**
     * @brief Compute the reverse-mode gradient of @p request on the device.
     *
     * The module built for this is the forward graph for @p request followed
     * by its vector-Jacobian product, both in the SAME `func.func @main`: its
     * parameters are the forward operands followed by ONE MORE parameter
     * holding the upstream cotangent (shaped `request.result_shape`), and its
     * results are one cotangent per operand, in operand order. So the backward
     * pass is device code, not a host post-pass over a device forward — which
     * is the whole point, since a host backward would leave the device with no
     * gradient of its own and nothing to train with.
     *
     * @param request    The FORWARD op. Its result_shape is the cotangent's shape.
     * @param operands   One host f64 pointer per entry of request.operand_shapes.
     * @param cotangent  Host f64 upstream cotangent, request.result_shape sized.
     *                   nullptr means a ones seed, which is the correct seed
     *                   for a scalar loss and, for a wider output, seeds the
     *                   gradient of the SUM of its elements.
     * @param gradients  One host f64 destination per operand, each at least as
     *                   large as that operand. Untouched on failure.
     * @param error      Set to a diagnostic when this returns false.
     *
     * @return true only if every operand had a VJP rule, the module compiled,
     *         executed, and every cotangent came back in full. There is no
     *         partial success: a gradient missing one term does not crash, it
     *         trains a model to garbage silently, so a request that cannot be
     *         answered completely is refused with a reason naming the op.
     */
    virtual bool runGradient(const DeviceOpRequest& request,
                             const std::vector<const double*>& operands,
                             const double* cotangent,
                             const std::vector<double*>& gradients,
                             std::string* error) = 0;

    /**
     * @brief Compile and execute a caller-supplied StableHLO module through
     *        the same PJRT path, executable cache and dtype staging as run().
     *
     * WHY THIS IS ON THE INTERFACE. A composite graph — several ops and their
     * shared backward pass in one program — cannot be described by a
     * DeviceOpRequest, which names exactly one op. The caller that CAN
     * describe it (a test, or later the region-formation path) already links
     * MLIR and can build the module with StableHLOEmitter; what it must not do
     * is re-implement the f64-to-device-dtype staging, the row-major read-back
     * and the compile cache, because a second copy of that is a second place
     * for the device and the host to disagree about layout. So the module text
     * crosses this interface as a string and everything numeric stays here.
     *
     * @param module_text    StableHLO in textual (MLIR) form, `func.func @main`.
     * @param cache_key      Caller's identity for this module. Two different
     *                       modules MUST NOT share a key; the cache would
     *                       return the wrong executable and the numbers would
     *                       be confidently wrong.
     * @param operand_shapes One per parameter of @main, in order.
     * @param operands       One host f64 pointer per operand shape.
     * @param result_shapes  One per result of @main, in order.
     * @param results        One host f64 destination per result shape.
     * @param error          Set to a diagnostic when this returns false.
     */
    virtual bool runModule(const std::string& module_text,
                           const std::string& cache_key,
                           const std::vector<std::vector<int64_t>>& operand_shapes,
                           const std::vector<const double*>& operands,
                           const std::vector<std::vector<int64_t>>& result_shapes,
                           const std::vector<double*>& results,
                           std::string* error) = 0;

    /** @brief Element type the device computes in: "f32", "f64", ... */
    virtual std::string dtypeName() const = 0;

    /** @brief One-line description of the live device, for gate evidence. */
    virtual std::string description() const = 0;

    /** @brief Counters since process start (or the last resetStats()). */
    virtual DeviceStats stats() const = 0;

    /** @brief Zero the counters. For tests that measure one phase at a time. */
    virtual void resetStats() = 0;
};

/**
 * @brief The installed device executor, or nullptr when none is installed.
 *
 * DEFINED IN xla_runtime.cpp (the slim runtime archive) so that this pointer
 * exists in every build, including ones with no MLIR anywhere on the link
 * line. See the file comment.
 */
DeviceExecutor* deviceExecutor();

/** @brief Install (or, with nullptr, uninstall) the device executor. */
void setDeviceExecutor(DeviceExecutor* executor);

/**
 * @brief Whether the caller ASKED for device execution.
 *
 * True when ESHKOL_XLA_PJRT=1 and ESHKOL_XLA_DEVICE is not "0". This is only
 * the request; whether a device is actually reachable is DeviceExecutor::
 * available(). Both must hold before anything runs on a device, and neither
 * implies the other: asking on a host with no plugin must produce a
 * diagnostic, and having a plugin must not silently move anyone's arithmetic
 * onto it.
 *
 * DEFINED IN xla_runtime.cpp for the same reason as deviceExecutor().
 */
bool deviceExecutionRequested();

/**
 * @brief Install the StableHLO/PJRT executor, and return it.
 *
 * DEFINED IN device_lowering.cpp, which depends on MLIR. Calling this from a
 * translation unit is what pulls the device path into a link; a binary that
 * never calls it never needs MLIR. Idempotent: repeated calls return the same
 * singleton and re-install it.
 *
 * The executor it returns is always non-null, but it is not necessarily
 * USABLE: in a build compiled without StableHLO support, or on a host with no
 * PJRT plugin, its available() reports exactly which of those is missing. That
 * split is deliberate — a build-configuration problem must surface as a named
 * diagnostic, never as a silent fall back to the host that leaves the caller
 * believing it ran on a device.
 */
DeviceExecutor* registerStableHLODeviceExecutor();

}  // namespace xla
}  // namespace eshkol

#endif  // ESHKOL_BACKEND_XLA_DEVICE_LOWERING_H
