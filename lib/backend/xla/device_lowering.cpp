/*
 * StableHLO device lowering for Eshkol tensor operations.
 *
 * This is the half of the seam described in device_lowering.h that depends on
 * the StableHLO emitter. It takes a DeviceOpRequest — an op plus the shapes it
 * runs over — builds a StableHLO `func.func @main` for exactly that op at
 * exactly those shapes through StableHLOEmitter, compiles it once through
 * XLARuntime's PJRT client, and thereafter executes the cached executable.
 *
 * WHY A MODULE PER (op, shapes, dtype), AND WHY IT IS CACHED.
 *
 * StableHLO is statically shaped: `tensor<128x128xf32>` is part of the program,
 * not an argument to it. So there is no single module that can serve `(+ a b)`
 * for every a and b; there is one module per shape signature. That is the
 * shape-specialisation the owner's ruling calls for, and its first slice is
 * here: the cache below is keyed by the whole signature (op, every operand
 * shape, the result shape, the axes, and the device element type), so a loop
 * that applies the same op to the same shapes compiles once and then only
 * transfers and executes. Without it every iteration would pay an XLA
 * compilation, which is between two and four orders of magnitude more than the
 * arithmetic and would make the device path slower than the host it replaced.
 *
 * WHY THE DEVICE ELEMENT TYPE IS NOT f64.
 *
 * Every Eshkol tensor is f64 (see the dtype note in xla_runtime.cpp). TPU has
 * no f64 arithmetic at all. So the device element type is a separate, explicit
 * choice — f32 by default, overridable with ESHKOL_XLA_DEVICE_DTYPE=f64 for a
 * CPU plugin where f64 is real — and the conversion happens here, on the way
 * in and on the way out. That conversion is the reason the parity tests state
 * a tolerance per op rather than asserting equality: comparing an f32 device
 * result against an f64 host reference is a comparison across precisions, and
 * pretending otherwise would either fail every test or, worse, force the
 * reference down to f32 and stop testing anything.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/device_lowering.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "eshkol/backend/xla/stablehlo_emitter.h"
#include "eshkol/backend/xla/xla_codegen.h"
#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_types.h"

namespace eshkol {
namespace xla {

namespace {

/** @brief Element count of a shape; an empty shape is rank-0, i.e. 1 element. */
int64_t numElements(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

std::string shapeKey(const std::vector<int64_t>& shape) {
    std::ostringstream os;
    os << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) os << 'x';
        os << shape[i];
    }
    os << ']';
    return os.str();
}

bool isBinary(DeviceOpKind k) {
    return k == DeviceOpKind::Add || k == DeviceOpKind::Subtract ||
           k == DeviceOpKind::Multiply || k == DeviceOpKind::Divide ||
           k == DeviceOpKind::Pow || k == DeviceOpKind::Maximum ||
           k == DeviceOpKind::Minimum ||
           k == DeviceOpKind::Matmul;
}

bool isReduce(DeviceOpKind k) {
    return k == DeviceOpKind::ReduceSum || k == DeviceOpKind::ReduceMean ||
           k == DeviceOpKind::ReduceMax || k == DeviceOpKind::ReduceMin ||
           k == DeviceOpKind::ReduceProd;
}

/**
 * @brief Right-aligned (NumPy) broadcast mapping from an operand shape to a
 *        result shape, as StableHLO's broadcast_in_dim wants it: one result
 *        dimension per operand dimension.
 *
 * Returns false when @p from cannot broadcast to @p to, which is a genuine
 * error here rather than something to work around: the host path has already
 * decided what the result shape is, so a disagreement means the two paths
 * would compute different things.
 */
bool broadcastDims(const std::vector<int64_t>& from, const std::vector<int64_t>& to,
                   std::vector<int64_t>* dims) {
    if (from.size() > to.size()) return false;
    const size_t offset = to.size() - from.size();
    dims->clear();
    dims->reserve(from.size());
    for (size_t i = 0; i < from.size(); ++i) {
        if (from[i] != to[offset + i] && from[i] != 1) return false;
        dims->push_back(static_cast<int64_t>(offset + i));
    }
    return true;
}

/**
 * @brief The shape this op actually produces, computed independently of what
 *        the caller claimed in request.result_shape.
 *
 * The two are then compared. That comparison is the point: the host allocator
 * has already committed to a result tensor of a particular shape by the time
 * this runs, and if the device would produce a different one, copying the
 * bytes back would silently reinterpret them. Better to refuse.
 */
bool inferResultShape(const DeviceOpRequest& req, std::vector<int64_t>* out,
                      std::string* error) {
    const auto& shapes = req.operand_shapes;
    switch (req.kind) {
        case DeviceOpKind::Add:
        case DeviceOpKind::Subtract:
        case DeviceOpKind::Multiply:
        case DeviceOpKind::Divide: {
            if (shapes.size() != 2) { *error = "binary elementwise op needs 2 operands"; return false; }
            // NumPy broadcasting: right-align and take the larger extent.
            const size_t rank = std::max(shapes[0].size(), shapes[1].size());
            out->assign(rank, 1);
            for (size_t i = 0; i < rank; ++i) {
                const int64_t a = i < rank - shapes[0].size() ? 1 : shapes[0][i - (rank - shapes[0].size())];
                const int64_t b = i < rank - shapes[1].size() ? 1 : shapes[1][i - (rank - shapes[1].size())];
                if (a != b && a != 1 && b != 1) {
                    *error = "operand shapes do not broadcast: " + shapeKey(shapes[0]) +
                             " vs " + shapeKey(shapes[1]);
                    return false;
                }
                (*out)[i] = std::max(a, b);
            }
            return true;
        }
        case DeviceOpKind::Pow:
        case DeviceOpKind::Maximum:
        case DeviceOpKind::Minimum: {
            if (shapes.size() != 2) { *error = "binary elementwise op needs 2 operands"; return false; }
            const size_t rank = std::max(shapes[0].size(), shapes[1].size());
            out->assign(rank, 1);
            for (size_t i = 0; i < rank; ++i) {
                const int64_t a = i < rank - shapes[0].size() ? 1 : shapes[0][i - (rank - shapes[0].size())];
                const int64_t b = i < rank - shapes[1].size() ? 1 : shapes[1][i - (rank - shapes[1].size())];
                if (a != b && a != 1 && b != 1) {
                    *error = "operand shapes do not broadcast: " + shapeKey(shapes[0]) +
                             " vs " + shapeKey(shapes[1]);
                    return false;
                }
                (*out)[i] = std::max(a, b);
            }
            return true;
        }
        case DeviceOpKind::Clamp: {
            // clamp(lo, x, hi). The bounds may broadcast to the value's shape
            // but the RESULT is the value's shape: a clamp whose bound is
            // wider than the value would be a different op, and silently
            // widening the result is exactly the reinterpretation
            // inferResultShape() exists to refuse.
            if (shapes.size() != 3) { *error = "clamp needs 3 operands (lo, x, hi)"; return false; }
            std::vector<int64_t> dims;
            if (!broadcastDims(shapes[0], shapes[1], &dims)) {
                *error = "clamp lower bound " + shapeKey(shapes[0]) +
                         " does not broadcast to the value shape " + shapeKey(shapes[1]);
                return false;
            }
            if (!broadcastDims(shapes[2], shapes[1], &dims)) {
                *error = "clamp upper bound " + shapeKey(shapes[2]) +
                         " does not broadcast to the value shape " + shapeKey(shapes[1]);
                return false;
            }
            *out = shapes[1];
            return true;
        }
        case DeviceOpKind::Exp:
        case DeviceOpKind::Log:
        case DeviceOpKind::Sin:
        case DeviceOpKind::Cos:
        case DeviceOpKind::Tanh:
        case DeviceOpKind::Sqrt:
        case DeviceOpKind::Rsqrt:
        case DeviceOpKind::Abs:
        case DeviceOpKind::Negate:
        case DeviceOpKind::Sigmoid:
        case DeviceOpKind::Atanh:
            if (shapes.size() != 1) { *error = "unary elementwise op needs 1 operand"; return false; }
            *out = shapes[0];
            return true;
        case DeviceOpKind::Matmul:
            if (shapes.size() != 2) { *error = "matmul needs 2 operands"; return false; }
            if (shapes[0].size() != 2 || shapes[1].size() != 2) {
                *error = "matmul lowering is rank-2 only, got " + shapeKey(shapes[0]) +
                         " x " + shapeKey(shapes[1]);
                return false;
            }
            if (shapes[0][1] != shapes[1][0]) {
                *error = "matmul inner dimensions disagree: " + shapeKey(shapes[0]) +
                         " x " + shapeKey(shapes[1]);
                return false;
            }
            *out = {shapes[0][0], shapes[1][1]};
            return true;
        case DeviceOpKind::Transpose: {
            if (shapes.size() != 1) { *error = "transpose needs 1 operand"; return false; }
            if (req.axes.size() != shapes[0].size()) {
                *error = "transpose permutation rank does not match operand rank";
                return false;
            }
            out->clear();
            for (int64_t p : req.axes) {
                if (p < 0 || p >= static_cast<int64_t>(shapes[0].size())) {
                    *error = "transpose permutation out of range";
                    return false;
                }
                out->push_back(shapes[0][p]);
            }
            return true;
        }
        case DeviceOpKind::Reshape:
            if (shapes.size() != 1) { *error = "reshape needs 1 operand"; return false; }
            if (numElements(shapes[0]) != numElements(req.result_shape)) {
                *error = "reshape changes the element count";
                return false;
            }
            *out = req.result_shape;
            return true;
        case DeviceOpKind::Broadcast: {
            if (shapes.size() != 1) { *error = "broadcast needs 1 operand"; return false; }
            if (req.axes.size() != shapes[0].size()) {
                *error = "broadcast dimension map rank does not match operand rank";
                return false;
            }
            *out = req.result_shape;
            return true;
        }
        case DeviceOpKind::ReduceSum:
        case DeviceOpKind::ReduceMean:
        case DeviceOpKind::ReduceMax:
        case DeviceOpKind::ReduceMin:
        case DeviceOpKind::ReduceProd: {
            if (shapes.size() != 1) { *error = "reduce needs 1 operand"; return false; }
            const auto& in = shapes[0];
            std::vector<bool> reduced(in.size(), req.axes.empty());
            for (int64_t ax : req.axes) {
                if (ax < 0 || ax >= static_cast<int64_t>(in.size())) {
                    *error = "reduce axis out of range";
                    return false;
                }
                reduced[static_cast<size_t>(ax)] = true;
            }
            out->clear();
            for (size_t i = 0; i < in.size(); ++i) {
                if (!reduced[i]) out->push_back(in[i]);
            }
            return true;
        }
    }
    *error = "unhandled op kind";
    return false;
}

/**
 * @brief The StableHLO/PJRT executor.
 *
 * One instance, installed by registerStableHLODeviceExecutor(). It owns the
 * executable cache and nothing else: the PJRT plugin, client and device
 * selection all live in XLARuntime, which is where PJRT was already stood up
 * (see XLARuntime::initialize) and which this deliberately does not duplicate.
 */
class StableHLODeviceExecutor final : public DeviceExecutor {
public:
    StableHLODeviceExecutor() {
        // Touch the runtime singleton here, in the constructor, so that it is
        // constructed BEFORE this object. Function-local statics are destroyed
        // in reverse order of construction, so this ordering is what
        // guarantees ~StableHLODeviceExecutor() can still call
        // releaseExecutable() on a live runtime rather than a destroyed one.
        (void)getDefaultRuntime();
        const char* dtype = std::getenv("ESHKOL_XLA_DEVICE_DTYPE");
        if (dtype && std::strcmp(dtype, "f64") == 0) {
            elem_ = ElementType::F64;
        } else {
            elem_ = ElementType::F32;
        }
    }

    ~StableHLODeviceExecutor() override {
        XLARuntime& rt = getDefaultRuntime();
        for (auto& entry : cache_) {
            rt.releaseExecutable(entry.second);
        }
        cache_.clear();
    }

    bool available(std::string* why) override {
        StableHLOEmitter probe;
        if (!probe.isAvailable()) {
            if (why) {
                *why = "this build has no StableHLO emitter (configure with "
                       "-DESHKOL_XLA_ENABLED=ON -DSTABLEHLO_ROOT=<path>)";
            }
            return false;
        }
        XLARuntime& rt = getDefaultRuntime();
        if (!rt.isDeviceExecutionActive()) {
            if (why) {
                std::string status = rt.deviceStatus();
                *why = status.empty()
                    ? "PJRT device execution is not active (set ESHKOL_XLA_PJRT=1 "
                      "and make a PJRT plugin discoverable)"
                    : status;
            }
            return false;
        }
        return true;
    }

    std::string dtypeName() const override {
        return elem_ == ElementType::F64 ? "f64" : "f32";
    }

    std::string description() const override {
        return getDefaultRuntime().getDescription() + "; device dtype=" + dtypeName();
    }

    DeviceStats stats() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }

    void resetStats() override {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_ = DeviceStats{};
    }

    bool run(const DeviceOpRequest& request,
             const std::vector<const double*>& operands,
             double* result,
             std::string* error) override {
        std::string local_error;
        std::string* err = error ? error : &local_error;
        err->clear();

        if (!available(err)) {
            bump(&DeviceStats::failures);
            return false;
        }
        if (operands.size() != request.operand_shapes.size()) {
            *err = "operand count does not match operand_shapes";
            bump(&DeviceStats::failures);
            return false;
        }
        for (const double* p : operands) {
            if (!p) {
                *err = "null operand pointer";
                bump(&DeviceStats::failures);
                return false;
            }
        }
        if (!result) {
            *err = "null result pointer";
            bump(&DeviceStats::failures);
            return false;
        }

        std::vector<int64_t> inferred;
        if (!inferResultShape(request, &inferred, err)) {
            bump(&DeviceStats::failures);
            return false;
        }
        if (inferred != request.result_shape) {
            *err = std::string("device would produce ") + shapeKey(inferred) +
                   " but caller allocated " + shapeKey(request.result_shape);
            bump(&DeviceStats::failures);
            return false;
        }

        void* executable = obtainExecutable(request, err);
        if (!executable) {
            bump(&DeviceStats::failures);
            return false;
        }

        return executeStaged(executable, request.operand_shapes, operands,
                             {request.result_shape}, {result},
                             deviceOpKindName(request.kind), err);
    }

    bool runGradient(const DeviceOpRequest& request,
                     const std::vector<const double*>& operands,
                     const double* cotangent,
                     const std::vector<double*>& gradients,
                     std::string* error) override {
        std::string local_error;
        std::string* err = error ? error : &local_error;
        err->clear();

        if (!available(err)) {
            bump(&DeviceStats::failures);
            return false;
        }
        if (operands.size() != request.operand_shapes.size()) {
            *err = "operand count does not match operand_shapes";
            bump(&DeviceStats::failures);
            return false;
        }
        if (gradients.size() != request.operand_shapes.size()) {
            *err = "gradient destination count does not match operand_shapes";
            bump(&DeviceStats::failures);
            return false;
        }
        for (const double* p : operands) {
            if (!p) { *err = "null operand pointer"; bump(&DeviceStats::failures); return false; }
        }
        for (double* p : gradients) {
            if (!p) { *err = "null gradient pointer"; bump(&DeviceStats::failures); return false; }
        }

        // The forward shape contract is checked before anything is emitted, so
        // that a shape disagreement is reported as a shape disagreement rather
        // than surfacing later as an opaque VJP or PJRT diagnostic.
        std::vector<int64_t> inferred;
        if (!inferResultShape(request, &inferred, err)) {
            bump(&DeviceStats::failures);
            return false;
        }
        if (inferred != request.result_shape) {
            *err = std::string("device would produce ") + shapeKey(inferred) +
                   " but caller stated a cotangent of " + shapeKey(request.result_shape);
            bump(&DeviceStats::failures);
            return false;
        }

        // A null cotangent means a ones seed. It is materialised HERE, on the
        // host, rather than left to the emitter's default: the gradient module
        // is cached by shape signature, and a module whose seed is baked in as
        // a constant would answer a later request that supplied a real
        // cotangent with the constant instead. One module shape, one meaning.
        const int64_t cot_n = numElements(request.result_shape);
        std::vector<double> ones_seed;
        if (!cotangent) {
            ones_seed.assign(static_cast<size_t>(cot_n), 1.0);
            cotangent = ones_seed.data();
        }

        const std::string key = "grad|" + cacheKey(request);
        void* executable = obtainExecutableForKey(
            key,
            [&](std::string* text, std::string* build_error) {
                return buildGradientModule(request, text, build_error);
            },
            err);
        if (!executable) {
            bump(&DeviceStats::failures);
            return false;
        }

        std::vector<std::vector<int64_t>> input_shapes = request.operand_shapes;
        input_shapes.push_back(request.result_shape);
        std::vector<const double*> inputs = operands;
        inputs.push_back(cotangent);

        return executeStaged(executable, input_shapes, inputs,
                             request.operand_shapes, gradients,
                             deviceOpKindName(request.kind), err);
    }

    bool runModule(const std::string& module_text,
                   const std::string& cache_key,
                   const std::vector<std::vector<int64_t>>& operand_shapes,
                   const std::vector<const double*>& operands,
                   const std::vector<std::vector<int64_t>>& result_shapes,
                   const std::vector<double*>& results,
                   std::string* error) override {
        std::string local_error;
        std::string* err = error ? error : &local_error;
        err->clear();

        if (!available(err)) {
            bump(&DeviceStats::failures);
            return false;
        }
        if (module_text.empty()) {
            *err = "empty module text";
            bump(&DeviceStats::failures);
            return false;
        }
        if (cache_key.empty()) {
            // Refused rather than defaulted: a shared or absent key makes the
            // cache hand back another module's executable, which produces
            // numbers that are wrong without anything being invalid.
            *err = "runModule requires a non-empty cache key";
            bump(&DeviceStats::failures);
            return false;
        }
        if (operands.size() != operand_shapes.size() || results.size() != result_shapes.size()) {
            *err = "buffer count does not match the stated shapes";
            bump(&DeviceStats::failures);
            return false;
        }
        for (const double* p : operands) {
            if (!p) { *err = "null operand pointer"; bump(&DeviceStats::failures); return false; }
        }
        for (double* p : results) {
            if (!p) { *err = "null result pointer"; bump(&DeviceStats::failures); return false; }
        }

        void* executable = obtainExecutableForKey(
            "mod|" + dtypeName() + "|" + cache_key,
            [&](std::string* text, std::string*) { *text = module_text; return true; },
            err);
        if (!executable) {
            bump(&DeviceStats::failures);
            return false;
        }

        return executeStaged(executable, operand_shapes, operands,
                             result_shapes, results, cache_key.c_str(), err);
    }

private:
    /**
     * @brief Transfer, execute and read back, converting between the host's
     *        f64 and the device element type on both sides.
     *
     * Shared by run(), runGradient() and runModule() rather than copied into
     * each: the f32 staging and the read-back are exactly where a device and a
     * host silently stop agreeing about layout and precision, and one copy is
     * the only way all three entry points keep agreeing.
     */
    bool executeStaged(void* executable,
                       const std::vector<std::vector<int64_t>>& input_shapes,
                       const std::vector<const double*>& inputs_host,
                       const std::vector<std::vector<int64_t>>& output_shapes,
                       const std::vector<double*>& outputs_host,
                       const char* what,
                       std::string* err) {
        XLARuntime& rt = getDefaultRuntime();

        // Stage the host f64 operands into the device element type. For f64
        // this is a pass-through (no copy); for f32 each operand is converted
        // into a staging vector that outlives the execute() call below.
        const bool narrow = (elem_ != ElementType::F64);
        std::vector<std::vector<float>> staged;
        if (narrow) staged.resize(inputs_host.size());

        std::vector<BufferDescriptor> inputs(inputs_host.size());
        for (size_t i = 0; i < inputs_host.size(); ++i) {
            const int64_t n = numElements(input_shapes[i]);
            inputs[i].shape = input_shapes[i];
            inputs[i].on_device = false;
            if (narrow) {
                staged[i].resize(static_cast<size_t>(n));
                for (int64_t j = 0; j < n; ++j) {
                    staged[i][static_cast<size_t>(j)] = static_cast<float>(inputs_host[i][j]);
                }
                inputs[i].data = staged[i].data();
                inputs[i].element_size = sizeof(float);
                inputs[i].elem = BufferElementType::F32;
            } else {
                inputs[i].data = const_cast<double*>(inputs_host[i]);
                inputs[i].element_size = sizeof(double);
                inputs[i].elem = BufferElementType::F64;
            }
        }

        std::vector<std::vector<float>> result_staged;
        if (narrow) result_staged.resize(outputs_host.size());
        std::vector<BufferDescriptor> outputs(outputs_host.size());
        for (size_t i = 0; i < outputs_host.size(); ++i) {
            const int64_t n = numElements(output_shapes[i]);
            outputs[i].shape = output_shapes[i];
            outputs[i].on_device = false;
            if (narrow) {
                result_staged[i].resize(static_cast<size_t>(n));
                outputs[i].data = result_staged[i].data();
                outputs[i].element_size = sizeof(float);
                outputs[i].elem = BufferElementType::F32;
            } else {
                outputs[i].data = outputs_host[i];
                outputs[i].element_size = sizeof(double);
                outputs[i].elem = BufferElementType::F64;
            }
        }

        ExecutionResult exec = rt.execute(executable, inputs, outputs);
        if (!exec.success) {
            *err = "device execution of " + std::string(what ? what : "<module>") +
                   " failed: " + exec.error_message;
            bump(&DeviceStats::failures);
            return false;
        }

        if (narrow) {
            for (size_t i = 0; i < outputs_host.size(); ++i) {
                const int64_t n = numElements(output_shapes[i]);
                for (int64_t j = 0; j < n; ++j) {
                    outputs_host[i][j] =
                        static_cast<double>(result_staged[i][static_cast<size_t>(j)]);
                }
            }
        }
        bump(&DeviceStats::executed);
        return true;
    }

    void bump(uint64_t DeviceStats::*field) {
        std::lock_guard<std::mutex> lock(mutex_);
        ++(stats_.*field);
    }

    std::string cacheKey(const DeviceOpRequest& req) const {
        std::ostringstream os;
        os << deviceOpKindName(req.kind) << '|' << dtypeName() << '|';
        for (const auto& s : req.operand_shapes) os << shapeKey(s);
        os << "->" << shapeKey(req.result_shape) << '|';
        os << shapeKey(req.axes);
        return os.str();
    }

    /** @brief Cached compile: build the module once per shape signature. */
    void* obtainExecutable(const DeviceOpRequest& req, std::string* error) {
        return obtainExecutableForKey(
            cacheKey(req),
            [&](std::string* text, std::string* build_error) {
                return buildModule(req, text, build_error);
            },
            error);
    }

    /**
     * @brief Cached compile for any module, keyed by @p key.
     *
     * @p build is only called on a miss, so a hit costs neither an emitter nor
     * an XLA compilation. The key must identify the module completely: it is
     * the only thing the cache compares, and two different modules under one
     * key would silently execute the wrong program.
     */
    void* obtainExecutableForKey(const std::string& key,
                                 const std::function<bool(std::string*, std::string*)>& build,
                                 std::string* error) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                ++stats_.cache_hits;
                return it->second;
            }
        }

        std::string module_text;
        if (!build(&module_text, error)) return nullptr;

        XLARuntime& rt = getDefaultRuntime();
        void* executable = rt.compileStableHLO(module_text, error);
        if (!executable) return nullptr;

        std::lock_guard<std::mutex> lock(mutex_);
        auto inserted = cache_.emplace(key, executable);
        if (!inserted.second) {
            // Another thread compiled the same signature while this one was
            // in PJRT. Keep the first, release the duplicate — never leak it
            // and never hand back a handle the map does not own.
            rt.releaseExecutable(executable);
            ++stats_.cache_hits;
            return inserted.first->second;
        }
        ++stats_.compiled;
        return executable;
    }

    /** @brief Build `func.func @main` for exactly this op at exactly these shapes. */
    bool buildModule(const DeviceOpRequest& req, std::string* text, std::string* error) {
        StableHLOEmitter emitter;
        if (!emitter.isAvailable()) {
            *error = "StableHLO emitter unavailable";
            return false;
        }

        std::vector<StableHLOEmitter::ParamSpec> params;
        params.reserve(req.operand_shapes.size());
        for (const auto& s : req.operand_shapes) {
            params.push_back(StableHLOEmitter::ParamSpec{s, elem_});
        }

        std::vector<void*> args = emitter.beginFunction("main", params);
        if (args.size() != params.size()) {
            *error = "beginFunction did not return one argument per parameter";
            return false;
        }

        void* value = emitOp(emitter, req, args, error);
        if (!value) return false;

        if (!emitter.endFunction({value})) {
            *error = "endFunction failed for " + std::string(deviceOpKindName(req.kind));
            return false;
        }

        *text = emitter.serializeToString();
        if (text->empty()) {
            *error = "serializeToString produced an empty module";
            return false;
        }
        return true;
    }

    /**
     * @brief Build `func.func @main` computing the VJP of exactly this op at
     *        exactly these shapes.
     *
     * Signature: (operand_0, ..., operand_n-1, cotangent) -> (d/d operand_0,
     * ..., d/d operand_n-1). The forward graph is built by the SAME emitOp()
     * the forward module uses — deliberately, because the backward pass
     * differentiates the graph that was actually emitted, and a second forward
     * written for the gradient's benefit could differ from the one that runs
     * without either being wrong on its own.
     *
     * The cotangent is a PARAMETER, not a constant. A baked-in ones seed would
     * make every gradient of a given shape share one cached executable that
     * ignores what the caller passed, which is the kind of failure that returns
     * a plausible number.
     *
     * Both halves are in one function on purpose. Splitting them would force
     * the forward intermediates across a function boundary as extra results,
     * and the values the VJP rules reuse (a tanh's own output, a divide's
     * quotient) would have to be re-derived on the far side.
     */
    bool buildGradientModule(const DeviceOpRequest& req, std::string* text, std::string* error) {
        StableHLOEmitter emitter;
        if (!emitter.isAvailable()) {
            *error = "StableHLO emitter unavailable";
            return false;
        }

        const size_t n_operands = req.operand_shapes.size();
        std::vector<StableHLOEmitter::ParamSpec> params;
        params.reserve(n_operands + 1);
        for (const auto& s : req.operand_shapes) {
            params.push_back(StableHLOEmitter::ParamSpec{s, elem_});
        }
        params.push_back(StableHLOEmitter::ParamSpec{req.result_shape, elem_});

        std::vector<void*> args = emitter.beginFunction("main", params);
        if (args.size() != params.size()) {
            *error = "beginFunction did not return one argument per parameter";
            return false;
        }
        std::vector<void*> operand_args(args.begin(), args.begin() + n_operands);
        void* seed = args[n_operands];

        void* forward = emitOp(emitter, req, operand_args, error);
        if (!forward) return false;

        VJPResult vjp = emitter.emitVJP(forward, operand_args, seed);
        if (!vjp.complete) {
            *error = std::string("no device gradient for ") + deviceOpKindName(req.kind) + ": " +
                     (vjp.diagnostic.empty() ? "emitVJP reported no diagnostic" : vjp.diagnostic);
            return false;
        }
        if (vjp.gradients.size() != n_operands) {
            // emitVJP's contract is one cotangent per wrt entry, in order.
            // Checked rather than assumed: a short vector here would be
            // returned to the caller as a gradient for the wrong operand.
            *error = std::string("emitVJP returned ") + std::to_string(vjp.gradients.size()) +
                     " gradients for " + std::to_string(n_operands) + " operands of " +
                     deviceOpKindName(req.kind);
            return false;
        }

        if (!emitter.endFunction(vjp.gradients)) {
            *error = std::string("endFunction failed for the gradient of ") +
                     deviceOpKindName(req.kind);
            return false;
        }

        *text = emitter.serializeToString();
        if (text->empty()) {
            *error = "serializeToString produced an empty gradient module";
            return false;
        }
        return true;
    }

    /**
     * @brief Broadcast an operand up to the result shape when it is not
     *        already that shape.
     *
     * This is what makes the broadcast case a DEVICE computation. The host
     * runtime broadcasts by materialising the expansion in a loop before the
     * arithmetic; doing the same thing here would transfer the expanded
     * operand and hide the broadcast from XLA entirely, so the test that
     * covers broadcasting would in fact be covering the host's expansion.
     */
    void* alignOperand(StableHLOEmitter& emitter, void* value,
                       const std::vector<int64_t>& from,
                       const std::vector<int64_t>& to,
                       std::string* error) {
        if (from == to) return value;
        std::vector<int64_t> dims;
        if (!broadcastDims(from, to, &dims)) {
            *error = "operand " + shapeKey(from) + " does not broadcast to " + shapeKey(to);
            return nullptr;
        }
        void* out = emitter.emitBroadcastInDim(value, to, dims);
        if (!out) *error = "emitBroadcastInDim failed";
        return out;
    }

    void* emitOp(StableHLOEmitter& emitter, const DeviceOpRequest& req,
                 const std::vector<void*>& args, std::string* error) {
        const auto& shapes = req.operand_shapes;

        if (isBinary(req.kind) && req.kind != DeviceOpKind::Matmul) {
            void* lhs = alignOperand(emitter, args[0], shapes[0], req.result_shape, error);
            if (!lhs) return nullptr;
            void* rhs = alignOperand(emitter, args[1], shapes[1], req.result_shape, error);
            if (!rhs) return nullptr;
            void* out = nullptr;
            switch (req.kind) {
                case DeviceOpKind::Add:      out = emitter.emitAdd(lhs, rhs); break;
                case DeviceOpKind::Subtract: out = emitter.emitSubtract(lhs, rhs); break;
                case DeviceOpKind::Multiply: out = emitter.emitMultiply(lhs, rhs); break;
                case DeviceOpKind::Divide:   out = emitter.emitDivide(lhs, rhs); break;
                case DeviceOpKind::Pow:      out = emitter.emitPow(lhs, rhs); break;
                case DeviceOpKind::Maximum:  out = emitter.emitMaximum(lhs, rhs); break;
                case DeviceOpKind::Minimum:  out = emitter.emitMinimum(lhs, rhs); break;
                default: break;
            }
            if (!out) *error = std::string("emit of ") + deviceOpKindName(req.kind) + " failed";
            return out;
        }

        switch (req.kind) {
            case DeviceOpKind::Exp:  { void* v = emitter.emitExp(args[0]);  if (!v) *error = "emitExp failed";  return v; }
            case DeviceOpKind::Log:  { void* v = emitter.emitLog(args[0]);  if (!v) *error = "emitLog failed";  return v; }
            case DeviceOpKind::Sin:  { void* v = emitter.emitSin(args[0]);  if (!v) *error = "emitSin failed";  return v; }
            case DeviceOpKind::Cos:  { void* v = emitter.emitCos(args[0]);  if (!v) *error = "emitCos failed";  return v; }
            case DeviceOpKind::Tanh: { void* v = emitter.emitTanh(args[0]); if (!v) *error = "emitTanh failed"; return v; }
            case DeviceOpKind::Sqrt:  { void* v = emitter.emitSqrt(args[0]);  if (!v) *error = "emitSqrt failed";  return v; }
            case DeviceOpKind::Rsqrt: { void* v = emitter.emitRsqrt(args[0]); if (!v) *error = "emitRsqrt failed"; return v; }
            case DeviceOpKind::Abs:   { void* v = emitter.emitAbs(args[0]);   if (!v) *error = "emitAbs failed";   return v; }
            case DeviceOpKind::Negate:{ void* v = emitter.emitNegate(args[0]);if (!v) *error = "emitNegate failed";return v; }
            case DeviceOpKind::Sigmoid:{ void* v = emitter.emitSigmoid(args[0]); if (!v) *error = "emitSigmoid failed"; return v; }
            case DeviceOpKind::Atanh: { void* v = emitter.emitAtanh(args[0]); if (!v) *error = "emitAtanh failed"; return v; }

            case DeviceOpKind::Clamp: {
                // The bounds are broadcast to the value's shape first: the
                // max/min VJP rule refuses implicitly broadcast operands, so a
                // clamp emitted against a rank-0 bound would have a forward
                // pass that runs and a backward pass that cannot be built.
                void* lo = alignOperand(emitter, args[0], shapes[0], req.result_shape, error);
                if (!lo) return nullptr;
                void* hi = alignOperand(emitter, args[2], shapes[2], req.result_shape, error);
                if (!hi) return nullptr;
                void* v = emitter.emitClamp(lo, args[1], hi);
                if (!v) *error = "emitClamp failed";
                return v;
            }

            case DeviceOpKind::Matmul: {
                DotDimensionNumbers dims;
                dims.lhs_contracting_dims = {1};
                dims.rhs_contracting_dims = {0};
                void* v = emitter.emitMatmul(args[0], args[1], dims);
                if (!v) *error = "emitMatmul failed";
                return v;
            }
            case DeviceOpKind::Transpose: {
                void* v = emitter.emitTranspose(args[0], req.axes);
                if (!v) *error = "emitTranspose failed";
                return v;
            }
            case DeviceOpKind::Reshape: {
                void* v = emitter.emitReshape(args[0], req.result_shape);
                if (!v) *error = "emitReshape failed";
                return v;
            }
            case DeviceOpKind::Broadcast: {
                void* v = emitter.emitBroadcastInDim(args[0], req.result_shape, req.axes);
                if (!v) *error = "emitBroadcastInDim failed";
                return v;
            }
            default:
                break;
        }

        if (isReduce(req.kind)) {
            std::vector<int64_t> axes = req.axes;
            if (axes.empty()) {
                for (size_t i = 0; i < shapes[0].size(); ++i) axes.push_back(static_cast<int64_t>(i));
            }
            if (req.kind == DeviceOpKind::ReduceMean) {
                // MEAN is not a StableHLO reduction, so it is built from ones:
                // sum(x) / sum(ones_like(x)), both reduced over the same axes.
                // The divisor is therefore computed on the device from the
                // shape itself rather than passed in as a constant, which
                // keeps the whole mean on the device — a host-side final
                // divide would leave the parity test unable to distinguish a
                // correct device sum from a broken one scaled back into range.
                void* sum = emitter.emitReduce(args[0], axes, StableHLOOp::REDUCE_SUM);
                if (!sum) { *error = "emitReduce(SUM) failed for mean"; return nullptr; }
                void* ones = emitter.emitOnesLike(args[0]);
                if (!ones) { *error = "emitOnesLike failed for mean"; return nullptr; }
                void* count = emitter.emitReduce(ones, axes, StableHLOOp::REDUCE_SUM);
                if (!count) { *error = "emitReduce(SUM) of ones failed for mean"; return nullptr; }
                void* mean = emitter.emitDivide(sum, count);
                if (!mean) { *error = "emitDivide failed for mean"; return nullptr; }
                return mean;
            }
            StableHLOOp op = StableHLOOp::REDUCE_SUM;
            switch (req.kind) {
                case DeviceOpKind::ReduceSum:  op = StableHLOOp::REDUCE_SUM;  break;
                case DeviceOpKind::ReduceMax:  op = StableHLOOp::REDUCE_MAX;  break;
                case DeviceOpKind::ReduceMin:  op = StableHLOOp::REDUCE_MIN;  break;
                case DeviceOpKind::ReduceProd: op = StableHLOOp::REDUCE_PROD; break;
                default: break;
            }
            void* v = emitter.emitReduce(args[0], axes, op);
            if (!v) *error = std::string("emitReduce failed for ") + deviceOpKindName(req.kind);
            return v;
        }

        *error = std::string("no StableHLO lowering for ") + deviceOpKindName(req.kind);
        return nullptr;
    }

    mutable std::mutex mutex_;
    ElementType elem_ = ElementType::F32;
    std::unordered_map<std::string, void*> cache_;
    DeviceStats stats_;
};

}  // namespace

DeviceExecutor* registerStableHLODeviceExecutor() {
    static StableHLODeviceExecutor executor;
    setDeviceExecutor(&executor);
    return &executor;
}

}  // namespace xla
}  // namespace eshkol
