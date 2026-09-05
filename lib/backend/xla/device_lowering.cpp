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
        case DeviceOpKind::Exp:
        case DeviceOpKind::Log:
        case DeviceOpKind::Sin:
        case DeviceOpKind::Cos:
        case DeviceOpKind::Tanh:
        case DeviceOpKind::Relu:
        case DeviceOpKind::Sigmoid:
        case DeviceOpKind::Softmax:
            // All shape-preserving, softmax included: it normalises along an
            // axis, it does not remove one.
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

        XLARuntime& rt = getDefaultRuntime();

        // Stage the host f64 operands into the device element type. For f64
        // this is a pass-through (no copy); for f32 each operand is converted
        // into a staging vector that outlives the execute() call below.
        const bool narrow = (elem_ != ElementType::F64);
        std::vector<std::vector<float>> staged;
        if (narrow) staged.resize(operands.size());

        std::vector<BufferDescriptor> inputs(operands.size());
        for (size_t i = 0; i < operands.size(); ++i) {
            const int64_t n = numElements(request.operand_shapes[i]);
            inputs[i].shape = request.operand_shapes[i];
            inputs[i].on_device = false;
            if (narrow) {
                staged[i].resize(static_cast<size_t>(n));
                for (int64_t j = 0; j < n; ++j) {
                    staged[i][static_cast<size_t>(j)] = static_cast<float>(operands[i][j]);
                }
                inputs[i].data = staged[i].data();
                inputs[i].element_size = sizeof(float);
                inputs[i].elem = BufferElementType::F32;
            } else {
                inputs[i].data = const_cast<double*>(operands[i]);
                inputs[i].element_size = sizeof(double);
                inputs[i].elem = BufferElementType::F64;
            }
        }

        const int64_t result_n = numElements(request.result_shape);
        std::vector<float> result_staged;
        std::vector<BufferDescriptor> outputs(1);
        outputs[0].shape = request.result_shape;
        outputs[0].on_device = false;
        if (narrow) {
            result_staged.resize(static_cast<size_t>(result_n));
            outputs[0].data = result_staged.data();
            outputs[0].element_size = sizeof(float);
            outputs[0].elem = BufferElementType::F32;
        } else {
            outputs[0].data = result;
            outputs[0].element_size = sizeof(double);
            outputs[0].elem = BufferElementType::F64;
        }

        ExecutionResult exec = rt.execute(executable, inputs, outputs);
        if (!exec.success) {
            *err = "device execution of " + std::string(deviceOpKindName(request.kind)) +
                   " failed: " + exec.error_message;
            bump(&DeviceStats::failures);
            return false;
        }

        if (narrow) {
            for (int64_t j = 0; j < result_n; ++j) {
                result[j] = static_cast<double>(result_staged[static_cast<size_t>(j)]);
            }
        }
        bump(&DeviceStats::executed);
        return true;
    }

private:
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
        const std::string key = cacheKey(req);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = cache_.find(key);
            if (it != cache_.end()) {
                ++stats_.cache_hits;
                return it->second;
            }
        }

        std::string module_text;
        if (!buildModule(req, &module_text, error)) return nullptr;

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

            case DeviceOpKind::Relu: {
                // max(x, 0). Not stablehlo.maximum against a bare zero
                // constant: the constant has to be a full-shape splat, which
                // is what emitConstantLike produces, because stablehlo's
                // binary ops do not broadcast their operands.
                void* zero = emitter.emitConstantLike(args[0], 0.0);
                if (!zero) { *error = "emitConstantLike(0) failed for relu"; return nullptr; }
                void* v = emitter.emitBinary(BinaryOp::Maximum, args[0], zero);
                if (!v) *error = "emitBinary(Maximum) failed for relu";
                return v;
            }
            case DeviceOpKind::Sigmoid: {
                // stablehlo.logistic IS 1/(1+e^-x). Composing it out of exp
                // and divide would be the same function with worse numerics
                // at the tails and no reason to prefer it.
                void* v = emitter.emitUnary(UnaryOp::Logistic, args[0]);
                if (!v) *error = "emitUnary(Logistic) failed for sigmoid";
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

        if (req.kind == DeviceOpKind::Softmax) {
            // Numerically stable softmax: subtract the max along the axes
            // before exponentiating, so a large input cannot overflow the
            // exponential. The host runtime (eshkol_xla_softmax) does exactly
            // the same thing, which is why the two agree to rounding rather
            // than merely to a tolerance.
            const std::vector<int64_t>& in_shape = shapes[0];
            std::vector<int64_t> axes = req.axes;
            if (axes.empty()) {
                for (size_t i = 0; i < in_shape.size(); ++i) axes.push_back(static_cast<int64_t>(i));
            }
            // Reducing removes the axes, so putting the reduced value back
            // against the input needs the map from each surviving result
            // dimension to the input dimension it came from.
            std::vector<bool> reduced(in_shape.size(), false);
            for (int64_t ax : axes) {
                if (ax < 0 || ax >= static_cast<int64_t>(in_shape.size())) {
                    *error = "softmax axis out of range";
                    return nullptr;
                }
                reduced[static_cast<size_t>(ax)] = true;
            }
            std::vector<int64_t> kept_dims;
            for (size_t i = 0; i < in_shape.size(); ++i) {
                if (!reduced[i]) kept_dims.push_back(static_cast<int64_t>(i));
            }

            void* mx = emitter.emitReduce(args[0], axes, StableHLOOp::REDUCE_MAX);
            if (!mx) { *error = "emitReduce(MAX) failed for softmax"; return nullptr; }
            void* mxb = emitter.emitBroadcastInDim(mx, in_shape, kept_dims);
            if (!mxb) { *error = "emitBroadcastInDim of the max failed for softmax"; return nullptr; }
            void* shifted = emitter.emitBinary(BinaryOp::Subtract, args[0], mxb);
            if (!shifted) { *error = "emitBinary(Subtract) failed for softmax"; return nullptr; }
            void* ex = emitter.emitUnary(UnaryOp::Exp, shifted);
            if (!ex) { *error = "emitUnary(Exp) failed for softmax"; return nullptr; }
            void* sum = emitter.emitReduce(ex, axes, StableHLOOp::REDUCE_SUM);
            if (!sum) { *error = "emitReduce(SUM) failed for softmax"; return nullptr; }
            void* sumb = emitter.emitBroadcastInDim(sum, in_shape, kept_dims);
            if (!sumb) { *error = "emitBroadcastInDim of the sum failed for softmax"; return nullptr; }
            void* out = emitter.emitBinary(BinaryOp::Divide, ex, sumb);
            if (!out) *error = "emitBinary(Divide) failed for softmax";
            return out;
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
