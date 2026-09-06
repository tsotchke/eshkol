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
        if (!like) { *error = "a literal with no operand beside it to take a shape from"; return bad; }
        HostVal v;
        v.shape = *like;
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
        case ESHKOL_CALL_OP: {
            const char* callee = calleeOf(op.call_op.func);
            if (!callee) { *error = "a call with a computed callee"; return bad; }
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
