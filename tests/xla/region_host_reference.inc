/*
 * tests/xla/region_host_reference.inc — the bodies for
 * tests/xla/region_host_reference.h.
 *
 * Split out rather than inlined in the header only so that the header stays
 * readable as an interface; there is exactly one includer.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

namespace eshkol_region_host {

HostVal Evaluator::eval(const eshkol_ast_t* node,
                        const std::vector<int64_t>* like,
                        std::string* error) {
    HostVal bad;
    if (!node) { *error = "null node"; return bad; }

    double lit = 0.0;
    if (literalOf(node, &lit)) {
        // Beside an operand a literal takes that operand's shape; on its own
        // (a loop counter's initial value, a literal arm of a conditional) it
        // is the rank-0 scalar it reads as.
        HostVal v;
        if (like) v.shape = *like;
        v.data.assign(static_cast<size_t>(elements(v.shape)), lit);
        v.ok = true;
        return v;
    }
    if (node->type == ESHKOL_VAR) {
        const std::string name = node->variable.id ? node->variable.id : "";
        for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return f->second;
        }
        *error = "unbound '" + name + "'";
        return bad;
    }
    if (node->type != ESHKOL_OP) { *error = "not an operation"; return bad; }

    const eshkol_operations_t& op = node->operation;
    switch (op.op) {
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP: {
            if (op.let_op.name) return loop(op, error);
            scopes_.emplace_back();
            for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                const eshkol_ast_t* b = &op.let_op.bindings[i];
                HostVal v = eval(bindValueOf(b), nullptr, error);
                if (!v.ok) { scopes_.pop_back(); return bad; }
                if (const char* n = bindNameOf(b)) scopes_.back()[n] = v;
            }
            HostVal body = eval(op.let_op.body, nullptr, error);
            scopes_.pop_back();
            return body;
        }
        case ESHKOL_SEQUENCE_OP: {
            HostVal last;
            for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i) {
                last = eval(&op.sequence_op.expressions[i], nullptr, error);
                if (!last.ok) return bad;
            }
            if (!last.ok) *error = "an empty begin has no value";
            return last;
        }
        case ESHKOL_ADD_OP: case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP: case ESHKOL_DIV_OP: {
            const char* n = op.op == ESHKOL_ADD_OP ? "+"
                          : op.op == ESHKOL_SUB_OP ? "-"
                          : op.op == ESHKOL_MUL_OP ? "*" : "/";
            return call(n, op.call_op.variables, op.call_op.num_vars, error);
        }
        case ESHKOL_IF_OP:
            if (op.call_op.num_vars != 3) { *error = "a conditional without both arms"; return bad; }
            return conditional(&op.call_op.variables[0], &op.call_op.variables[1],
                               &op.call_op.variables[2], error);
        case ESHKOL_CALL_OP: {
            const char* callee = calleeOf(op.call_op.func);
            if (!callee) { *error = "a call with a computed callee"; return bad; }
            if (std::strcmp(callee, "if") == 0 && op.call_op.num_vars == 3)
                return conditional(&op.call_op.variables[0], &op.call_op.variables[1],
                                   &op.call_op.variables[2], error);
            auto fn = fns_.find(callee);
            if (fn != fns_.end() && fn->second.body) {
                std::vector<HostVal> args;
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                    HostVal a = eval(&op.call_op.variables[i], nullptr, error);
                    if (!a.ok) return bad;
                    args.push_back(a);
                }
                if (args.size() != fn->second.params.size()) {
                    *error = std::string("arity mismatch calling ") + callee;
                    return bad;
                }
                scopes_.emplace_back();
                for (size_t i = 0; i < args.size(); ++i)
                    scopes_.back()[fn->second.params[i]] = args[i];
                HostVal v = eval(fn->second.body, nullptr, error);
                scopes_.pop_back();
                return v;
            }
            return call(callee, op.call_op.variables, op.call_op.num_vars, error);
        }
        default:
            *error = "a form the host reference cannot evaluate";
            return bad;
    }
}

/** @brief The host `if`: the predicate is a 0/1 scalar (a comparison's host
 *         encoding) and the arm it selects is the one evaluated, which is
 *         exactly what the host program does with the same source. */
HostVal Evaluator::conditional(const eshkol_ast_t* pred, const eshkol_ast_t* then_arm,
                               const eshkol_ast_t* else_arm, std::string* error) {
    HostVal bad;
    HostVal p = eval(pred, nullptr, error);
    if (!p.ok) return bad;
    if (p.data.size() != 1) { *error = "the predicate of a conditional is not a scalar"; return bad; }
    const eshkol_ast_t* taken = p.data[0] != 0.0 ? then_arm : else_arm;
    const eshkol_ast_t* other = p.data[0] != 0.0 ? else_arm : then_arm;
    double lit = 0.0;
    if (literalOf(taken, &lit)) {
        // A literal arm is shaped like the other arm, as the device side
        // shapes it; if both are literals the value is a scalar.
        if (literalOf(other, &lit)) return eval(taken, nullptr, error);
        HostVal o = eval(other, nullptr, error);
        if (!o.ok) return bad;
        return eval(taken, &o.shape, error);
    }
    return eval(taken, nullptr, error);
}

/** @brief The host named let, run as the loop it is: bind, test, carry. The
 *         body shape is the one R8 admits, (if pred (loop args...) exit). */
HostVal Evaluator::loop(const eshkol_operations_t& let, std::string* error) {
    HostVal bad;
    const eshkol_ast_t* body = let.let_op.body;
    if (!body || body->type != ESHKOL_OP) { *error = "a named let with no body"; return bad; }
    const eshkol_operations_t& b = body->operation;
    const bool is_if = b.op == ESHKOL_IF_OP ||
        (b.op == ESHKOL_CALL_OP && calleeOf(b.call_op.func) &&
         std::strcmp(calleeOf(b.call_op.func), "if") == 0);
    if (!is_if || b.call_op.num_vars != 3) { *error = "a named let whose body is not a conditional"; return bad; }
    auto isSelf = [&](const eshkol_ast_t* n) {
        return n && n->type == ESHKOL_OP && n->operation.op == ESHKOL_CALL_OP &&
               calleeOf(n->operation.call_op.func) &&
               std::strcmp(calleeOf(n->operation.call_op.func), let.let_op.name) == 0;
    };
    const eshkol_ast_t* pred = &b.call_op.variables[0];
    const eshkol_ast_t* self_call = nullptr;
    const eshkol_ast_t* exit = nullptr;
    if (isSelf(&b.call_op.variables[1])) { self_call = &b.call_op.variables[1]; exit = &b.call_op.variables[2]; }
    else if (isSelf(&b.call_op.variables[2])) { self_call = &b.call_op.variables[2]; exit = &b.call_op.variables[1]; }
    else { *error = "a named let with no tail call to itself"; return bad; }

    const uint64_t n = let.let_op.num_bindings;
    if (self_call->operation.call_op.num_vars != n) { *error = "the loop's self-call arity differs from its bindings"; return bad; }
    std::vector<std::string> names;
    std::vector<HostVal> state;
    for (uint64_t i = 0; i < n; ++i) {
        const eshkol_ast_t* bnd = &let.let_op.bindings[i];
        const char* name = bindNameOf(bnd);
        if (!name) { *error = "a loop binding with no name"; return bad; }
        names.push_back(name);
        HostVal v = eval(bindValueOf(bnd), nullptr, error);
        if (!v.ok) return bad;
        state.push_back(v);
    }

    // A bound, so a loop whose predicate never turns false is reported
    // rather than run forever; the corpus loops run a handful of times.
    const int kMaxTrips = 1000000;
    for (int trip = 0; ; ++trip) {
        if (trip >= kMaxTrips) { *error = "the loop did not terminate within the reference's bound"; return bad; }
        scopes_.emplace_back();
        for (uint64_t i = 0; i < n; ++i) scopes_.back()[names[i]] = state[i];
        HostVal p = eval(pred, nullptr, error);
        if (!p.ok) { scopes_.pop_back(); return bad; }
        if (p.data.size() != 1) { scopes_.pop_back(); *error = "the loop's trip predicate is not a scalar"; return bad; }
        if (p.data[0] == 0.0) {
            HostVal out = eval(exit, nullptr, error);
            scopes_.pop_back();
            return out;
        }
        std::vector<HostVal> next;
        for (uint64_t i = 0; i < n; ++i) {
            HostVal v = eval(&self_call->operation.call_op.variables[i], &state[i].shape, error);
            if (!v.ok) { scopes_.pop_back(); return bad; }
            next.push_back(v);
        }
        scopes_.pop_back();
        state = std::move(next);
    }
}

HostVal Evaluator::call(const char* name, const eshkol_ast_t* argv, uint64_t argc,
                        std::string* error) {
    HostVal bad;
    DeviceOpKind kind;
    if (!eshkol::xla::regionCoreOpKind(name, argc, &kind)) {
        *error = std::string("no device op for ") + name;
        return bad;
    }

    std::vector<HostVal> vals(argc);
    const std::vector<int64_t>* like = nullptr;
    for (uint64_t i = 0; i < argc; ++i) {
        double ignored = 0.0;
        if (literalOf(&argv[i], &ignored)) continue;
        vals[i] = eval(&argv[i], nullptr, error);
        if (!vals[i].ok) return bad;
        if (!like) like = &vals[i].shape;
    }
    for (uint64_t i = 0; i < argc; ++i) {
        if (vals[i].ok) continue;
        vals[i] = eval(&argv[i], like, error);
        if (!vals[i].ok) return bad;
    }

    auto asU64 = [](const std::vector<int64_t>& s) {
        std::vector<uint64_t> u(s.size());
        for (size_t i = 0; i < s.size(); ++i) u[i] = static_cast<uint64_t>(s[i]);
        return u;
    };

    if (eshkol::xla::deviceOpYieldsPredicate(kind)) {
        // The host reference entry point for a comparison: 0/1 in f64, the
        // encoding every predicate has on the host.
        std::vector<uint64_t> a = asU64(vals[0].shape);
        std::vector<uint64_t> b = asU64(vals[1].shape);
        void* t = eshkol_xla_compare_host(
            arena_, vals[0].data.data(), vals[1].data.data(),
            elements(vals[0].shape), a.data(), static_cast<int64_t>(a.size()),
            elements(vals[1].shape), b.data(), static_cast<int64_t>(b.size()),
            compareDirection(kind));
        if (!t) { *error = std::string("the host compare path refused ") + name; return bad; }
        HostVal v;
        v.shape = (vals[1].shape.size() > vals[0].shape.size()) ? vals[1].shape : vals[0].shape;
        v.data = tensorElements(t, elements(v.shape));
        v.ok = static_cast<int64_t>(v.data.size()) == elements(v.shape);
        if (!v.ok) *error = std::string("the host compare path returned the wrong count for ") + name;
        return v;
    }

    const int ew = elementwiseCode(kind);
    if (ew >= 0) {
        const bool binary = eshkol_xla_elementwise_is_binary(ew) != 0;
        std::vector<uint64_t> a = asU64(vals[0].shape);
        std::vector<uint64_t> b = binary ? asU64(vals[1].shape) : std::vector<uint64_t>{};
        void* t = eshkol_xla_elementwise_host(
            arena_, vals[0].data.data(), binary ? vals[1].data.data() : nullptr,
            elements(vals[0].shape), a.data(), static_cast<int64_t>(a.size()),
            binary ? elements(vals[1].shape) : 0,
            binary ? b.data() : nullptr,
            binary ? static_cast<int64_t>(b.size()) : 0, ew);
        if (!t) { *error = std::string("the host elementwise path refused ") + name; return bad; }
        HostVal v;
        v.shape = (binary && vals[1].shape.size() > vals[0].shape.size())
                    ? vals[1].shape : vals[0].shape;
        v.data = tensorElements(t, elements(v.shape));
        v.ok = static_cast<int64_t>(v.data.size()) == elements(v.shape);
        if (!v.ok) *error = std::string("the host elementwise path returned the wrong count for ") + name;
        return v;
    }

    const int rc = reduceCode(kind);
    if (rc >= 0) {
        std::vector<uint64_t> a = asU64(vals[0].shape);
        // axis -1 is "every axis", which is what (tensor-sum t) means and what
        // the region emitter asks the device for.
        void* t = eshkol_xla_reduce_host(arena_, vals[0].data.data(),
                                         elements(vals[0].shape), a.data(),
                                         static_cast<int64_t>(a.size()), -1, rc);
        if (!t) { *error = std::string("the host reduce path refused ") + name; return bad; }
        HostVal v;
        v.data = tensorElements(t, 1);
        v.ok = v.data.size() == 1;
        if (!v.ok) *error = std::string("the host reduce path returned the wrong count for ") + name;
        return v;
    }

    if (kind == DeviceOpKind::Matmul) {
        if (vals[0].shape.size() != 2 || vals[1].shape.size() != 2) {
            *error = "the host matmul reference needs two rank-2 operands";
            return bad;
        }
        void* t = eshkol_xla_matmul_host(arena_, vals[0].data.data(), vals[1].data.data(),
                                         vals[0].shape.data(), vals[1].shape.data(), 2, 2);
        if (!t) { *error = "the host matmul path refused"; return bad; }
        HostVal v;
        v.shape = {vals[0].shape[0], vals[1].shape[1]};
        v.data = tensorElements(t, elements(v.shape));
        v.ok = static_cast<int64_t>(v.data.size()) == elements(v.shape);
        return v;
    }

    if (kind == DeviceOpKind::Transpose) {
        if (vals[0].shape.size() != 2) { *error = "the host transpose reference needs rank 2"; return bad; }
        std::vector<uint64_t> a = asU64(vals[0].shape);
        std::vector<int64_t> perm = {1, 0};
        void* t = eshkol_xla_transpose_host(arena_, vals[0].data.data(), a.data(), 2, perm.data());
        if (!t) { *error = "the host transpose path refused"; return bad; }
        HostVal v;
        v.shape = {vals[0].shape[1], vals[0].shape[0]};
        v.data = tensorElements(t, elements(v.shape));
        v.ok = static_cast<int64_t>(v.data.size()) == elements(v.shape);
        return v;
    }

    *error = std::string("no host reference for ") + name;
    return bad;
}

} // namespace eshkol_region_host
