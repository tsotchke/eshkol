/*
 * lib/backend/xla/region_execution.cpp — emitting an outlined region as one
 * StableHLO function and running it through PJRT.
 *
 * See inc/eshkol/backend/xla/region_execution.h for why this is separate from
 * the pass and why every node goes through emitDeviceOp().
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/region_execution.h"

#include <cstdio>
#include <cstring>
#include <sstream>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/device_op_emission.h"
#include "eshkol/backend/xla/stablehlo_emitter.h"

namespace eshkol {
namespace xla {

namespace {

/** @brief A value inside the region under construction: the emitter handle
 *         and the shape it has, which the emitter does not tell us back. */
struct Val {
    void* handle = nullptr;
    std::vector<int64_t> shape;
    bool ok() const { return handle != nullptr; }
};

int64_t elementCount(const std::vector<int64_t>& shape) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
}

const char* calleeName(const eshkol_ast_t* node) {
    if (!node) return nullptr;
    return node->type == ESHKOL_VAR ? node->variable.id : nullptr;
}

const char* bindingName(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS && b->cons_cell.car &&
        b->cons_cell.car->type == ESHKOL_VAR)
        return b->cons_cell.car->variable.id;
    if (b->type == ESHKOL_VAR) return b->variable.id;
    return nullptr;
}

const eshkol_ast_t* bindingValue(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS) return b->cons_cell.cdr;
    if (b->type == ESHKOL_VAR) return b->variable.data;
    return b;
}

/** @brief Is this call a conditional written as a call to `if`? */
bool isIfCall(const eshkol_operations_t& op) {
    if (op.op != ESHKOL_CALL_OP) return false;
    const eshkol_ast_t* f = op.call_op.func;
    return f && f->type == ESHKOL_VAR && f->variable.id &&
           std::strcmp(f->variable.id, "if") == 0 && op.call_op.num_vars == 3;
}

/** @brief Is @p node a numeric literal, i.e. something that has a value but no
 *         shape of its own until it is put beside an operand? */
bool isNumericLiteral(const eshkol_ast_t* node, double* value) {
    if (!node) return false;
    switch (node->type) {
        case ESHKOL_DOUBLE: *value = node->double_val; return true;
        case ESHKOL_INT64:  *value = static_cast<double>(node->int64_val); return true;
        case ESHKOL_INT32:  *value = static_cast<double>(node->int32_val); return true;
        default: return false;
    }
}

/** @brief Is @p node a call to the named let @p name? */
bool isCallTo(const eshkol_ast_t* node, const char* name) {
    if (!node || !name || node->type != ESHKOL_OP) return false;
    if (node->operation.op != ESHKOL_CALL_OP) return false;
    const char* callee = calleeName(node->operation.call_op.func);
    return callee && std::strcmp(callee, name) == 0;
}

/** @brief Is @p node a leaf value — a variable or a numeric literal — rather
 *         than something with operations in it? A conditional whose arm is a
 *         leaf has nothing to skip by not evaluating that arm, which is what
 *         decides select against stablehlo.if. */
bool isLeaf(const eshkol_ast_t* node) {
    double ignored = 0.0;
    return node && (node->type == ESHKOL_VAR || isNumericLiteral(node, &ignored));
}

/**
 * @brief The shape of a named let's body as region formation admitted it:
 *        `(if pred (name args...) exit)` or `(if pred exit (name args...))`.
 *        Returns false when the body is not of that shape.
 */
bool loopStructure(const eshkol_operations_t& let, const eshkol_ast_t** pred,
                   const eshkol_ast_t** self_call, const eshkol_ast_t** exit) {
    const eshkol_ast_t* body = let.let_op.body;
    if (!body || body->type != ESHKOL_OP) return false;
    const eshkol_operations_t& b = body->operation;
    if (!(b.op == ESHKOL_IF_OP || isIfCall(b)) || b.call_op.num_vars != 3) return false;
    const eshkol_ast_t* p = &b.call_op.variables[0];
    const eshkol_ast_t* t = &b.call_op.variables[1];
    const eshkol_ast_t* e = &b.call_op.variables[2];
    if (isCallTo(t, let.let_op.name)) { *pred = p; *self_call = t; *exit = e; return true; }
    if (isCallTo(e, let.let_op.name)) { *pred = p; *self_call = e; *exit = t; return true; }
    return false;
}

/** @brief FNV-1a over the module text, as 16 hex digits. */
std::string moduleFingerprint(const std::string& text) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : text) {
        h ^= c;
        h *= 1099511628211ull;
    }
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

} // namespace

class RegionExecutor::Impl {
public:
    explicit Impl(const std::map<std::string, RegionFunction>& functions)
        : functions_(functions) {}

    const std::map<std::string, RegionFunction>& functions_;
    uint64_t modules_built_ = 0;
    uint64_t executions_ = 0;

    /** Name -> value, innermost scope last. */
    std::vector<std::map<std::string, Val>> scopes_;

    Val lookup(const char* name) const {
        if (!name) return Val{};
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return f->second;
        }
        return Val{};
    }

    /**
     * @brief Emit @p node into the open function.
     *
     * @param like When non-null, the value a bare numeric literal should take
     *        its shape and element type from. StableHLO has no shapeless
     *        scalar that broadcasts itself, so a literal only becomes a value
     *        once something has said what shape it must be — which is why a
     *        literal is emitted from its SIBLING rather than on its own.
     */
    Val emit(StableHLOEmitter& emitter, const eshkol_ast_t* node, void* like,
             std::string* error) {
        if (!node) { *error = "null node in region"; return Val{}; }

        double literal = 0.0;
        if (isNumericLiteral(node, &literal)) {
            if (!like) {
                *error = "a numeric literal in a region has no shape of its own; "
                         "it must appear beside a tensor operand";
                return Val{};
            }
            Val v;
            v.handle = emitter.emitConstantLike(like, literal);
            if (!v.handle) { *error = "emitConstantLike failed for a literal"; return Val{}; }
            v.shape = like_shape_;
            return v;
        }

        if (node->type == ESHKOL_VAR) {
            Val v = lookup(node->variable.id);
            if (!v.ok()) {
                *error = std::string("region references '") +
                         (node->variable.id ? node->variable.id : "?") +
                         "' which is not one of its inputs";
            }
            return v;
        }

        if (node->type != ESHKOL_OP) {
            *error = "a region node that is neither an operation nor a value";
            return Val{};
        }

        const eshkol_operations_t& op = node->operation;

        switch (op.op) {
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                if (op.let_op.name) return emitLoop(emitter, op, error);
                scopes_.emplace_back();
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const eshkol_ast_t* b = &op.let_op.bindings[i];
                    Val v = emit(emitter, bindingValue(b), nullptr, error);
                    if (!v.ok()) { scopes_.pop_back(); return Val{}; }
                    if (const char* n = bindingName(b)) scopes_.back()[n] = v;
                }
                Val body = emit(emitter, op.let_op.body, nullptr, error);
                scopes_.pop_back();
                return body;
            }
            case ESHKOL_SEQUENCE_OP: {
                Val last;
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i) {
                    last = emit(emitter, &op.sequence_op.expressions[i], nullptr, error);
                    if (!last.ok()) return Val{};
                }
                if (!last.ok()) *error = "an empty begin in a region has no value";
                return last;
            }
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP: case ESHKOL_DIV_OP: {
                const char* name = op.op == ESHKOL_ADD_OP ? "+"
                                 : op.op == ESHKOL_SUB_OP ? "-"
                                 : op.op == ESHKOL_MUL_OP ? "*" : "/";
                return emitBuiltin(emitter, name, op.call_op.variables,
                                   op.call_op.num_vars, error);
            }
            case ESHKOL_IF_OP:
                if (op.call_op.num_vars != 3) {
                    *error = "a conditional in a region must have a predicate and two arms";
                    return Val{};
                }
                return emitConditional(emitter, &op.call_op.variables[0],
                                       &op.call_op.variables[1], &op.call_op.variables[2],
                                       error);
            case ESHKOL_CALL_OP: {
                if (isIfCall(op)) {
                    return emitConditional(emitter, &op.call_op.variables[0],
                                           &op.call_op.variables[1], &op.call_op.variables[2],
                                           error);
                }
                const char* callee = calleeName(op.call_op.func);
                if (!callee) { *error = "a call in a region with no callee name"; return Val{}; }

                // A call into a top-level function region formation counted as
                // inlined is inlined here, with its parameters bound to the
                // emitted argument values. The report says the region contains
                // the callee's ops; this is what makes that true of the module.
                auto fn = functions_.find(callee);
                if (fn != functions_.end() && fn->second.body) {
                    std::vector<Val> args;
                    for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                        Val a = emit(emitter, &op.call_op.variables[i], nullptr, error);
                        if (!a.ok()) return Val{};
                        args.push_back(a);
                    }
                    if (args.size() != fn->second.params.size()) {
                        *error = std::string("call to '") + callee + "' has " +
                                 std::to_string(args.size()) + " arguments and it takes " +
                                 std::to_string(fn->second.params.size());
                        return Val{};
                    }
                    scopes_.emplace_back();
                    for (size_t i = 0; i < args.size(); ++i)
                        scopes_.back()[fn->second.params[i]] = args[i];
                    Val v = emit(emitter, fn->second.body, nullptr, error);
                    scopes_.pop_back();
                    return v;
                }
                return emitBuiltin(emitter, callee, op.call_op.variables,
                                   op.call_op.num_vars, error);
            }
            default:
                *error = "a form inside a region that the emitter cannot emit; "
                         "region formation should have reported it as a break";
                return Val{};
        }
    }

    /**
     * @brief Emit an `if` whose predicate and both arms are in the region.
     *
     * TWO LOWERINGS, CHOSEN HERE, AND THE RULE IS STATED ONCE:
     *
     *   stablehlo.select  when the region is inside a differentiated
     *                     expression, or when either arm is a leaf (a
     *                     variable or a literal).
     *   stablehlo.if      otherwise — both arms carry operations and the
     *                     region is not differentiated.
     *
     * Inside `gradient` it MUST be a select: emitVJP walks a flat use-def
     * graph and cannot enter stablehlo.if's regions, while select has a VJP
     * rule (the cotangent goes down the arm that was taken, zero down the
     * other). Both arms are computed, which is correct because a region
     * admits no effects — the only cost is the arithmetic of the arm not
     * taken. A leaf arm has no operations to skip, so a select is the
     * cheaper op there too. A stablehlo.if evaluates one arm and is what
     * condition 3 names; it is used when that actually saves work.
     *
     * The predicate is a rank-0 i1 (region formation admits only a
     * comparison whose shape is statically a scalar); for a select over
     * ranked arms it is broadcast to the arm shape.
     */
    Val emitConditional(StableHLOEmitter& emitter, const eshkol_ast_t* pred_node,
                        const eshkol_ast_t* then_node, const eshkol_ast_t* else_node,
                        std::string* error) {
        Val pred = emit(emitter, pred_node, nullptr, error);
        if (!pred.ok()) return Val{};
        if (!emitter.isPredicate(pred.handle)) {
            *error = "the predicate of a conditional in a region is not a comparison";
            return Val{};
        }
        if (!pred.shape.empty()) {
            *error = "the predicate of a conditional in a region is not a scalar";
            return Val{};
        }

        const bool use_select = in_gradient_ || isLeaf(then_node) || isLeaf(else_node);
        if (use_select) {
            // Non-literal arm first, so a literal arm can take its shape.
            double ignored = 0.0;
            const bool then_lit = isNumericLiteral(then_node, &ignored);
            const bool else_lit = isNumericLiteral(else_node, &ignored);
            Val t, e;
            if (then_lit && else_lit) {
                // Two literal arms: the value is a scalar shaped like the
                // predicate, which is rank 0.
                double tv = 0.0, ev = 0.0;
                isNumericLiteral(then_node, &tv);
                isNumericLiteral(else_node, &ev);
                t.handle = emitter.emitSplatConstant({}, elem_, tv);
                e.handle = emitter.emitSplatConstant({}, elem_, ev);
                if (!t.ok() || !e.ok()) { *error = "emitSplatConstant failed for a literal arm"; return Val{}; }
            } else if (then_lit) {
                e = emit(emitter, else_node, nullptr, error);
                if (!e.ok()) return Val{};
                like_shape_ = e.shape;
                t = emit(emitter, then_node, e.handle, error);
                if (!t.ok()) return Val{};
            } else if (else_lit) {
                t = emit(emitter, then_node, nullptr, error);
                if (!t.ok()) return Val{};
                like_shape_ = t.shape;
                e = emit(emitter, else_node, t.handle, error);
                if (!e.ok()) return Val{};
            } else {
                t = emit(emitter, then_node, nullptr, error);
                if (!t.ok()) return Val{};
                e = emit(emitter, else_node, nullptr, error);
                if (!e.ok()) return Val{};
            }
            if (t.shape != e.shape) {
                *error = "the two arms of a conditional in a region have different shapes";
                return Val{};
            }
            void* p = pred.handle;
            if (!t.shape.empty()) {
                p = emitter.emitBroadcastInDim(pred.handle, t.shape, {});
                if (!p) { *error = "could not broadcast the predicate to the arm shape"; return Val{}; }
            }
            Val out;
            out.handle = emitter.emitSelect(p, t.handle, e.handle);
            if (!out.handle) { *error = "emitSelect failed"; return Val{}; }
            out.shape = t.shape;
            return out;
        }

        Val then_val, else_val;
        std::string arm_error;
        void* result = emitter.emitIf(
            pred.handle,
            [&]() -> void* {
                then_val = emit(emitter, then_node, nullptr, &arm_error);
                return then_val.handle;
            },
            [&]() -> void* {
                else_val = emit(emitter, else_node, nullptr, &arm_error);
                return else_val.handle;
            });
        if (!result) {
            *error = arm_error.empty()
                ? "emitIf failed (the two arms of the conditional do not agree in type)"
                : arm_error;
            return Val{};
        }
        Val out;
        out.handle = result;
        out.shape = then_val.shape;
        return out;
    }

    /**
     * @brief Emit a named let as stablehlo.while over its bindings as the
     *        carried tuple.
     *
     * The body has the shape region formation admitted (R8):
     * `(if pred (name args...) exit)` in either arm order. The predicate is
     * the loop condition over the carried values, the self-call's arguments
     * are the next iteration's carried values, and the exit expression is
     * evaluated over the final carried values after the loop.
     */
    Val emitLoop(StableHLOEmitter& emitter, const eshkol_operations_t& op,
                 std::string* error) {
        if (in_gradient_) {
            *error = "a loop inside a differentiated expression has no device VJP; "
                     "region formation should have reported it as a break";
            return Val{};
        }
        const eshkol_ast_t *pred = nullptr, *self_call = nullptr, *exit = nullptr;
        if (!loopStructure(op, &pred, &self_call, &exit)) {
            *error = "a named let in a region is not of the admitted shape "
                     "(if pred (loop args...) exit)";
            return Val{};
        }
        const uint64_t n = op.let_op.num_bindings;
        if (self_call->operation.call_op.num_vars != n) {
            *error = "the loop's self-call does not pass one value per binding";
            return Val{};
        }

        std::vector<std::string> names;
        std::vector<Val> init;
        for (uint64_t i = 0; i < n; ++i) {
            const eshkol_ast_t* b = &op.let_op.bindings[i];
            const char* name = bindingName(b);
            if (!name) { *error = "a loop binding with no name"; return Val{}; }
            names.push_back(name);
            const eshkol_ast_t* value = bindingValue(b);
            double lit = 0.0;
            Val v;
            if (isNumericLiteral(value, &lit)) {
                // A counter starting from a literal: a rank-0 constant of the
                // graph's element type. Nothing beside it to copy a type from.
                v.handle = emitter.emitSplatConstant({}, elem_, lit);
                if (!v.handle) { *error = "emitSplatConstant failed for a loop's initial value"; return Val{}; }
            } else {
                v = emit(emitter, value, nullptr, error);
                if (!v.ok()) return Val{};
            }
            init.push_back(v);
        }

        std::vector<void*> init_handles;
        for (const Val& v : init) init_handles.push_back(v.handle);

        std::string inner_error;
        auto bindCarried = [&](const std::vector<void*>& args) {
            scopes_.emplace_back();
            for (uint64_t i = 0; i < n; ++i) {
                Val v;
                v.handle = args[i];
                v.shape = init[i].shape;
                scopes_.back()[names[i]] = v;
            }
        };

        std::vector<void*> results = emitter.emitWhile(
            init_handles,
            [&](const std::vector<void*>& args) -> void* {
                bindCarried(args);
                Val p = emit(emitter, pred, nullptr, &inner_error);
                scopes_.pop_back();
                if (!p.ok()) return nullptr;
                if (!emitter.isPredicate(p.handle) || !p.shape.empty()) {
                    inner_error = "the loop's trip predicate is not a scalar comparison";
                    return nullptr;
                }
                return p.handle;
            },
            [&](const std::vector<void*>& args) -> std::vector<void*> {
                bindCarried(args);
                std::vector<void*> next;
                const eshkol_ast_t* argv = self_call->operation.call_op.variables;
                for (uint64_t i = 0; i < n; ++i) {
                    like_shape_ = init[i].shape;
                    Val v = emit(emitter, &argv[i], args[i], &inner_error);
                    if (!v.ok()) { scopes_.pop_back(); return {}; }
                    if (v.shape != init[i].shape) {
                        inner_error = "the loop changes the shape of '" + names[i] + "'";
                        scopes_.pop_back();
                        return {};
                    }
                    next.push_back(v.handle);
                }
                scopes_.pop_back();
                return next;
            });
        if (results.size() != n) {
            *error = inner_error.empty()
                ? "emitWhile failed (the carried values do not agree in type across iterations)"
                : inner_error;
            return Val{};
        }

        bindCarried(results);
        Val out = emit(emitter, exit, nullptr, error);
        scopes_.pop_back();
        return out;
    }

    /** True while the region being built is inside a differentiated
     *  expression, which decides select against stablehlo.if above. */
    bool in_gradient_ = false;

    /** @brief Emit one builtin call: its operands, then the op itself. */
    Val emitBuiltin(StableHLOEmitter& emitter, const char* name,
                    const eshkol_ast_t* argv, uint64_t argc, std::string* error) {
        DeviceOpKind kind;
        if (!regionCoreOpKind(name, argc, &kind)) {
            *error = std::string("no device op for '") + name + "' at arity " +
                     std::to_string(argc);
            return Val{};
        }

        // Non-literal operands first: a literal takes its shape from one of
        // them, so at least one operand has to be a real value.
        std::vector<Val> vals(argc);
        void* like = nullptr;
        for (uint64_t i = 0; i < argc; ++i) {
            double ignored = 0.0;
            if (isNumericLiteral(&argv[i], &ignored)) continue;
            vals[i] = emit(emitter, &argv[i], nullptr, error);
            if (!vals[i].ok()) return Val{};
            if (!like) { like = vals[i].handle; like_shape_ = vals[i].shape; }
        }
        for (uint64_t i = 0; i < argc; ++i) {
            if (vals[i].ok()) continue;
            vals[i] = emit(emitter, &argv[i], like, error);
            if (!vals[i].ok()) return Val{};
        }

        DeviceOpRequest req;
        req.kind = kind;
        std::vector<void*> handles;
        for (uint64_t i = 0; i < argc; ++i) {
            req.operand_shapes.push_back(vals[i].shape);
            handles.push_back(vals[i].handle);
        }
        // A reduction with no axes named reduces every axis, which is what
        // (tensor-sum t) means.
        if (!inferDeviceResultShape(req, &req.result_shape, error)) return Val{};

        Val out;
        out.handle = emitDeviceOp(emitter, req, handles, error);
        if (!out.handle) return Val{};
        out.shape = req.result_shape;
        return out;
    }

    /** Shape of the value `like` currently points at. */
    std::vector<int64_t> like_shape_;

    bool build(const Region& region,
               const std::vector<std::vector<int64_t>>& operand_shapes,
               bool with_gradient,
               std::string* module_text,
               std::vector<int64_t>* result_shape,
               std::string* error) {
        if (!region.root) { *error = "region has no root node"; return false; }
        if (operand_shapes.size() != region.inputs.size()) {
            *error = "region takes " + std::to_string(region.inputs.size()) +
                     " inputs and " + std::to_string(operand_shapes.size()) +
                     " shapes were given";
            return false;
        }

        StableHLOEmitter emitter;
        if (!emitter.isAvailable()) {
            *error = "StableHLO emitter unavailable (built without MLIR)";
            return false;
        }

        std::vector<StableHLOEmitter::ParamSpec> params;
        for (const auto& s : operand_shapes)
            params.push_back(StableHLOEmitter::ParamSpec{s, elem_});

        std::vector<void*> args = emitter.beginFunction("main", params);
        if (args.size() != params.size()) {
            *error = "beginFunction did not return one argument per input";
            return false;
        }

        scopes_.clear();
        scopes_.emplace_back();
        for (size_t i = 0; i < args.size(); ++i) {
            Val v;
            v.handle = args[i];
            v.shape = operand_shapes[i];
            scopes_.back()[region.inputs[i].name] = v;
        }

        in_gradient_ = with_gradient;
        Val out = emit(emitter, region.root, nullptr, error);
        if (!out.ok()) return false;

        // A predicate leaves the region as 0/1 in the device element type;
        // the host buffer it is read into holds floats, and an i1 read into
        // it would be misread rather than refused.
        if (emitter.isPredicate(out.handle)) {
            out.handle = emitter.emitConvert(out.handle, elem_);
            if (!out.handle) { *error = "could not convert the region's predicate result"; return false; }
        }

        std::vector<void*> results;
        if (with_gradient) {
            // Forward and backward in ONE function, exactly as the single-op
            // gradient path does: the values a VJP rule reuses (a tanh's own
            // output) are already here, and splitting the two would have to
            // carry them across a function boundary.
            VJPResult vjp = emitter.emitVJP(out.handle, args, nullptr);
            if (!vjp.complete) {
                *error = "region inside a differentiated expression has no complete "
                         "device VJP: " + vjp.diagnostic;
                return false;
            }
            results = vjp.gradients;
        } else {
            results = {out.handle};
        }

        if (!emitter.endFunction(results)) {
            *error = "endFunction failed for the region";
            return false;
        }
        *module_text = emitter.serializeToString();
        if (module_text->empty()) {
            *error = "the region serialized to an empty module";
            return false;
        }
        *result_shape = out.shape;
        modules_built_++;
        return true;
    }

    ElementType elem_ = ElementType::F32;
};

RegionExecutor::RegionExecutor(const std::map<std::string, RegionFunction>& functions)
    : impl_(new Impl(functions)) {
    // The device decides the element type it can compute in; f64 is what every
    // Eshkol tensor is on the host, and TPU has no f64. Reading the installed
    // executor's answer rather than assuming keeps this in step with the
    // single-op path.
    if (DeviceExecutor* d = deviceExecutor()) {
        impl_->elem_ = d->dtypeName() == "f64" ? ElementType::F64 : ElementType::F32;
    }
}

RegionExecutor::~RegionExecutor() { delete impl_; }

bool RegionExecutor::buildModule(const Region& region,
                                 const std::vector<std::vector<int64_t>>& operand_shapes,
                                 std::string* module_text,
                                 std::vector<int64_t>* result_shape,
                                 std::string* error) {
    return impl_->build(region, operand_shapes, false, module_text, result_shape, error);
}

bool RegionExecutor::buildGradientModule(const Region& region,
                                         const std::vector<std::vector<int64_t>>& operand_shapes,
                                         std::string* module_text,
                                         std::vector<int64_t>* result_shape,
                                         std::string* error) {
    return impl_->build(region, operand_shapes, true, module_text, result_shape, error);
}

bool RegionExecutor::execute(const Region& region,
                             const std::vector<RegionOperand>& operands,
                             double* result,
                             std::vector<int64_t>* result_shape,
                             std::string* error) {
    DeviceExecutor* device = deviceExecutor();
    if (!device) {
        *error = "no device executor installed; call registerStableHLODeviceExecutor()";
        return false;
    }
    std::string why;
    if (!device->available(&why)) { *error = "no device available: " + why; return false; }

    std::vector<std::vector<int64_t>> shapes;
    std::vector<const double*> data;
    for (const RegionOperand& o : operands) {
        shapes.push_back(o.shape);
        data.push_back(o.data);
    }

    std::string module_text;
    std::vector<int64_t> out_shape;
    if (!impl_->build(region, shapes, region.inside_gradient,
                      &module_text, &out_shape, error))
        return false;

    // THE CACHE KEY IS DERIVED FROM THE MODULE, not only from the region's
    // shape signature.
    //
    // The signature names the region within its compilation unit — "R#1:
    // [4],[4]->[4]" — and two different programs both have a unit called R
    // with a region 1 over two rank-4 operands. Keyed on the signature alone,
    // the second program's tensor-add collected the executable the first
    // program's tensor-mul had compiled, and returned its numbers. That is
    // exactly the failure runModule's contract warns about: not a crash, a
    // confidently wrong answer, and the corpus measurement is where it
    // surfaced (two regions of 14_cond_break came back with O(1) errors
    // against the host while every other region was exact).
    //
    // Hashing the module text cannot collide with a different module by
    // accident and still gives one executable per (region, shapes): the same
    // region entered with the same shapes emits the same text.
    std::string key = region.shape_signature + "|" + moduleFingerprint(module_text);
    if (region.inside_gradient) key += "|vjp";

    std::vector<std::vector<int64_t>> result_shapes;
    std::vector<double*> results;
    if (region.inside_gradient) {
        // One cotangent per input, shaped like that input.
        for (const auto& s : shapes) result_shapes.push_back(s);
        // The caller's buffer holds them back to back, in input order.
        double* p = result;
        for (const auto& s : shapes) {
            results.push_back(p);
            p += elementCount(s);
        }
    } else {
        result_shapes.push_back(out_shape);
        results.push_back(result);
    }

    if (!device->runModule(module_text, key, shapes, data, result_shapes, results, error))
        return false;

    *result_shape = out_shape;
    impl_->executions_++;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// The registry generated code reaches through eshkol_xla_region().
// ─────────────────────────────────────────────────────────────────────────
namespace {

struct RegisteredRegion {
    const Region* region = nullptr;
    const std::map<std::string, RegionFunction>* functions = nullptr;
};

std::vector<RegisteredRegion>& registry() {
    static std::vector<RegisteredRegion> r;
    return r;
}

uint64_t g_executed = 0;
uint64_t g_failed = 0;

bool runRegisteredRegion(int64_t region_id,
                         const std::vector<std::vector<int64_t>>& shapes,
                         const std::vector<const double*>& operands,
                         std::vector<double>* result,
                         std::vector<int64_t>* result_shape,
                         std::string* error) {
    if (region_id < 0 || static_cast<size_t>(region_id) >= registry().size()) {
        *error = "no region is registered under that id";
        g_failed++;
        return false;
    }
    const RegisteredRegion& entry = registry()[static_cast<size_t>(region_id)];
    if (!entry.region || !entry.functions) {
        *error = "the registered region is empty";
        g_failed++;
        return false;
    }

    // A rank-0 input crosses the seam boxed as a one-element tensor, because
    // a tensor is the only thing generated code can hand this entry point
    // (see codegenRegionCall). The region knows the input is a scalar; the
    // shape the module is built for is the region's, not the box's, or a
    // comparison against the scalar would be emitted over [1] and its
    // predicate would not be the rank-0 one a conditional needs.
    std::vector<std::vector<int64_t>> region_shapes = shapes;
    for (size_t i = 0; i < region_shapes.size() && i < entry.region->inputs.size(); ++i) {
        const RegionShape& declared = entry.region->inputs[i].shape;
        if (declared.known && declared.dims.empty() &&
            region_shapes[i].size() == 1 && region_shapes[i][0] == 1)
            region_shapes[i].clear();
    }

    std::vector<RegionOperand> ops;
    for (size_t i = 0; i < region_shapes.size() && i < operands.size(); ++i)
        ops.push_back(RegionOperand{region_shapes[i], operands[i]});

    // Sized from the region's own emitted result rather than guessed: the
    // module is built first so the result shape is known before anything is
    // written into a buffer.
    std::string module_text;
    std::vector<int64_t> shape;
    RegionExecutor exec(*entry.functions);
    if (!exec.buildModule(*entry.region, region_shapes, &module_text, &shape, error)) {
        g_failed++;
        return false;
    }
    int64_t total = 1;
    for (int64_t d : shape) total *= d;
    result->assign(static_cast<size_t>(total), 0.0);

    if (!exec.execute(*entry.region, ops, result->data(), result_shape, error)) {
        g_failed++;
        return false;
    }
    g_executed++;
    return true;
}

} // namespace

int64_t registerRegionForExecution(const Region& region,
                                   const std::map<std::string, RegionFunction>& functions) {
    if (!region.root) return -1;
    setRegionRunner(&runRegisteredRegion);
    registry().push_back(RegisteredRegion{&region, &functions});
    return static_cast<int64_t>(registry().size()) - 1;
}

void clearRegisteredRegions() {
    registry().clear();
    g_executed = 0;
    g_failed = 0;
}

void registeredRegionStats(uint64_t* executed, uint64_t* failed) {
    if (executed) *executed = g_executed;
    if (failed) *failed = g_failed;
}

uint64_t RegionExecutor::modulesBuilt() const { return impl_->modules_built_; }
uint64_t RegionExecutor::executions() const { return impl_->executions_; }

} // namespace xla
} // namespace eshkol
