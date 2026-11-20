/**
 * @file quantum_types.cpp
 * @brief Quantum register types: index expressions, the qreg/qidx families,
 *        quantum signatures and the application check (ADR-0048).
 */

#include "eshkol/types/quantum_types.h"
#include "eshkol/types/dependent.h"

#include <algorithm>
#include <cctype>
#include <set>
#include <sstream>

namespace eshkol::hott::quantum {

// ===========================================================================
// Families
// ===========================================================================

TypeId familyBase(Family f) { return f == Family::QReg ? BuiltinTypes::QReg : BuiltinTypes::QIdx; }
const char* familyName(Family f) { return f == Family::QReg ? "qreg" : "qidx"; }

std::optional<std::pair<Family, IndexExpr>> parseIndexedSpelling(const std::string& name,
                                                                 std::string* error) {
    if (name.size() < 7 || name.back() != '>') return std::nullopt;
    std::string head = name.substr(0, 5);
    std::transform(head.begin(), head.end(), head.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    Family f;
    if (head == "qreg<") f = Family::QReg;
    else if (head == "qidx<") f = Family::QIdx;
    else return std::nullopt;
    const std::string inner = name.substr(5, name.size() - 6);
    auto index = IndexExpr::parse(inner);
    if (!index) {
        if (error) *error = "malformed index '" + inner + "' in " + name +
                            " (an index is a sum of naturals and names, e.g. 3, n, n+1, 2*n)";
        return std::make_pair(f, IndexExpr::unknown());
    }
    return std::make_pair(f, *index);
}

TypeId indexedType(TypeEnvironment& env, Family f, const IndexExpr& index) {
    if (!index.known) return familyBase(f);
    return env.internIndexedType(familyBase(f), index.toString());
}

std::optional<std::pair<Family, IndexExpr>> indexedTypeOf(const TypeEnvironment& env, TypeId t) {
    if (t == BuiltinTypes::QReg) return std::make_pair(Family::QReg, IndexExpr::unknown());
    if (t == BuiltinTypes::QIdx) return std::make_pair(Family::QIdx, IndexExpr::unknown());
    auto inst = env.indexedInstanceOf(t);
    if (!inst) return std::nullopt;
    const Family f = inst->first == BuiltinTypes::QReg ? Family::QReg : Family::QIdx;
    auto index = IndexExpr::parse(inst->second);
    return std::make_pair(f, index ? *index : IndexExpr::unknown());
}

bool isQuantumResource(const TypeEnvironment& env, TypeId t) {
    if (t == BuiltinTypes::Qubit || t == BuiltinTypes::QRegion) return true;
    auto f = indexedTypeOf(env, t);
    return f && f->first == Family::QReg;
}

// ===========================================================================
// Signatures
// ===========================================================================

std::string SpecType::toString(const TypeEnvironment& env) const {
    switch (kind) {
        case Kind::Plain: return env.getTypeName(plain);
        case Kind::Indexed: return std::string(familyName(family)) + "<" + index.toString() + ">";
        case Kind::Var: return "'" + var;
        case Kind::Arrow: {
            std::string out = "(->";
            for (const auto& p : params) out += " " + p.toString(env);
            return out + " " + (result ? result->toString(env) : std::string("Value")) + ")";
        }
    }
    return "?";
}

std::string Signature::key(const TypeEnvironment& env) const {
    std::string out = "quantum-sig:" + name;
    for (const auto& [n, t] : params) out += "(" + n + " " + t.toString(env) + ")";
    out += variadic ? " ... -> " : " -> ";
    return out + result.toString(env);
}

SpecType specFromAnnotation(TypeEnvironment& env, const hott_type_expr_t* annotation,
                            const std::function<TypeId(const hott_type_expr_t*)>& resolve,
                            const std::function<bool(const std::string&)>& is_alias) {
    SpecType s;
    if (!annotation) return s;
    if (annotation->kind == HOTT_TYPE_VAR && annotation->var_name) {
        const std::string name = annotation->var_name;
        if (auto spelled = parseIndexedSpelling(name)) {
            s.kind = SpecType::Kind::Indexed;
            s.family = spelled->first;
            s.index = spelled->second;
            return s;
        }
        TypeId t = resolve(annotation);
        if (t == BuiltinTypes::QReg || t == BuiltinTypes::QIdx) {
            s.kind = SpecType::Kind::Indexed;
            s.family = t == BuiltinTypes::QReg ? Family::QReg : Family::QIdx;
            s.index = IndexExpr::unknown();
            return s;
        }
        if (!isBuiltinTypeName(name) && !is_alias(name) && t == BuiltinTypes::Value &&
            !env.lookupType(name)) {
            s.kind = SpecType::Kind::Var;
            s.var = name;
            return s;
        }
        s.plain = t;
        return s;
    }
    if (annotation->kind == HOTT_TYPE_ARROW) {
        s.kind = SpecType::Kind::Arrow;
        for (uint64_t i = 0; i < annotation->arrow.num_params; ++i) {
            s.params.push_back(specFromAnnotation(env, annotation->arrow.param_types
                                                           ? annotation->arrow.param_types[i]
                                                           : nullptr,
                                                  resolve, is_alias));
        }
        s.result = std::make_shared<SpecType>(
            specFromAnnotation(env, annotation->arrow.return_type, resolve, is_alias));
        return s;
    }
    s.plain = resolve(annotation);
    return s;
}

TypeId eraseSpec(TypeEnvironment& env, const SpecType& spec) {
    switch (spec.kind) {
        case SpecType::Kind::Plain: return spec.plain;
        case SpecType::Kind::Indexed: return indexedType(env, spec.family, spec.index);
        case SpecType::Kind::Var: return BuiltinTypes::Value;
        case SpecType::Kind::Arrow: {
            std::vector<TypeId> ps;
            for (const auto& p : spec.params) ps.push_back(eraseSpec(env, p));
            return env.makeFunctionType(ps, spec.result ? eraseSpec(env, *spec.result)
                                                        : BuiltinTypes::Value);
        }
    }
    return BuiltinTypes::Value;
}

namespace {

bool specIsQuantum(const SpecType& s) {
    switch (s.kind) {
        case SpecType::Kind::Indexed: return true;
        case SpecType::Kind::Plain:
            return s.plain == BuiltinTypes::QRegion || s.plain == BuiltinTypes::QRegionBody;
        case SpecType::Kind::Arrow:
            for (const auto& p : s.params) if (specIsQuantum(p)) return true;
            return s.result && specIsQuantum(*s.result);
        case SpecType::Kind::Var: return false;
    }
    return false;
}

void collectIndexVariables(const SpecType& s, std::set<std::string>& out) {
    if (s.kind == SpecType::Kind::Indexed && s.index.known) {
        for (const auto& [v, a] : s.index.terms) { (void)a; out.insert(v); }
    }
    for (const auto& p : s.params) collectIndexVariables(p, out);
    if (s.result) collectIndexVariables(*s.result, out);
}

// ---------------------------------------------------------------------------
// The primitive signatures of core.quantum, in the annotation language.
// ---------------------------------------------------------------------------

const char* const kPrimitiveSignatures[] = {
    "(qalloc (qr qregion) (n integer)) qreg<n>",
    "(qh (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qx (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qy (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qz (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qs (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qsdg (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qt (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qtdg (r qreg<n>) (i qidx<n>)) qreg<n>",
    "(qp (r qreg<n>) (i qidx<n>) (p number)) qreg<n>",
    "(qrx (r qreg<n>) (i qidx<n>) (theta number)) qreg<n>",
    "(qry (r qreg<n>) (i qidx<n>) (theta number)) qreg<n>",
    "(qrz (r qreg<n>) (i qidx<n>) (theta number)) qreg<n>",
    "(qcnot (r qreg<n>) (c qidx<n>) (t qidx<n>)) qreg<n>",
    "(qcz (r qreg<n>) (a qidx<n>) (b qidx<n>)) qreg<n>",
    "(qswap (r qreg<n>) (a qidx<n>) (b qidx<n>)) qreg<n>",
    "(qrzz (r qreg<n>) (a qidx<n>) (b qidx<n>) (theta number)) qreg<n>",
    "(qccx (r qreg<n>) (a qidx<n>) (b qidx<n>) (t qidx<n>)) qreg<n>",
    "(qccz (r qreg<n>) (a qidx<n>) (b qidx<n>) (c qidx<n>)) qreg<n>",
    "(qapply-circuit (r qreg<n>) (c any)) qreg<n>",
    "(qmeasure (r qreg<n>)) list",
    "(qmeasure-bit (r qreg<1>)) integer",
    "(qrelease (r qreg<n>)) null",
    "(qsplit (r qreg<n>) (k integer) (f (-> qreg<k> qreg<n-k> a))) a",
    "(qjoin (a qreg<m>) (b qreg<n>)) qreg<m+n>",
    "(qreg->qubit (r qreg<1>)) qubit",
    "(qubit->qreg (q qubit)) qreg<1>",
    "(call-with-quantum-region (body qregion-body)) a",
};

// Table types become ordinary annotation trees, so primitives and user
// definitions take the same road to a SpecType.
hott_type_expr_t* sxToAnnotation(const detail::Sx& sx) {
    if (!sx.list) {
        const auto* spelling = lookupBuiltinTypeSpelling(sx.atom);
        if (spelling && spelling->kind != HOTT_TYPE_VAR) return hott_make_primitive_type(spelling->kind);
        return hott_make_type_var(sx.atom.c_str());
    }
    if (!sx.items.empty() && !sx.items[0].list && sx.items[0].atom == "->") {
        std::vector<hott_type_expr_t*> ps;
        for (size_t k = 1; k + 1 < sx.items.size(); ++k) ps.push_back(sxToAnnotation(sx.items[k]));
        hott_type_expr_t* ret = sxToAnnotation(sx.items.back());
        hott_type_expr_t* arrow = hott_make_arrow_type(ps.data(), ps.size(), ret);
        for (auto* p : ps) hott_free_type_expr(p);
        hott_free_type_expr(ret);
        return arrow;
    }
    return hott_make_primitive_type(HOTT_TYPE_ANY);
}

// ---------------------------------------------------------------------------
// The application check
// ---------------------------------------------------------------------------

// Tail expressions of a body: what a procedure can return.
void collectTails(const eshkol_ast_t* e, std::vector<const eshkol_ast_t*>& out, int depth = 0) {
    if (!e || depth > 64) return;
    if (e->type != ESHKOL_OP) { out.push_back(e); return; }
    const auto& op = e->operation;
    switch (op.op) {
        case ESHKOL_SEQUENCE_OP:
            if (op.sequence_op.num_expressions > 0)
                collectTails(&op.sequence_op.expressions[op.sequence_op.num_expressions - 1], out, depth + 1);
            return;
        case ESHKOL_LET_OP: case ESHKOL_LET_STAR_OP: case ESHKOL_LETREC_OP: case ESHKOL_LETREC_STAR_OP:
            collectTails(op.let_op.body, out, depth + 1);
            return;
        case ESHKOL_CALL_OP:
            if (op.call_op.func && op.call_op.func->type == ESHKOL_VAR && op.call_op.func->variable.id) {
                const std::string f = op.call_op.func->variable.id;
                if (f == "if" && op.call_op.num_vars >= 2) {
                    for (uint64_t k = 1; k < op.call_op.num_vars; ++k)
                        collectTails(&op.call_op.variables[k], out, depth + 1);
                    return;
                }
                if (f == "begin" && op.call_op.num_vars > 0) {
                    collectTails(&op.call_op.variables[op.call_op.num_vars - 1], out, depth + 1);
                    return;
                }
            }
            out.push_back(e);
            return;
        default:
            out.push_back(e);
            return;
    }
}

bool mentionsName(const eshkol_ast_t* e, const std::string& name, int depth = 0);

bool mentionsInOp(const eshkol_operations_t& op, const std::string& name, int depth) {
    switch (op.op) {
        case ESHKOL_CALL_OP:
            if (mentionsName(op.call_op.func, name, depth + 1)) return true;
            for (uint64_t k = 0; k < op.call_op.num_vars; ++k)
                if (mentionsName(&op.call_op.variables[k], name, depth + 1)) return true;
            return false;
        case ESHKOL_SEQUENCE_OP:
            for (uint64_t k = 0; k < op.sequence_op.num_expressions; ++k)
                if (mentionsName(&op.sequence_op.expressions[k], name, depth + 1)) return true;
            return false;
        case ESHKOL_LAMBDA_OP:
            for (uint64_t k = 0; k < op.lambda_op.num_params; ++k) {
                const auto& p = op.lambda_op.parameters[k];
                if (p.type == ESHKOL_VAR && p.variable.id && name == p.variable.id) return false;
            }
            return mentionsName(op.lambda_op.body, name, depth + 1);
        case ESHKOL_LET_OP: case ESHKOL_LET_STAR_OP: case ESHKOL_LETREC_OP: case ESHKOL_LETREC_STAR_OP:
            for (uint64_t k = 0; k < op.let_op.num_bindings; ++k) {
                const auto& b = op.let_op.bindings[k];
                if (b.type == ESHKOL_CONS && mentionsName(b.cons_cell.cdr, name, depth + 1)) return true;
            }
            return mentionsName(op.let_op.body, name, depth + 1);
        default:
            return false;
    }
}

bool mentionsName(const eshkol_ast_t* e, const std::string& name, int depth) {
    if (!e || depth > 256) return false;
    if (e->type == ESHKOL_VAR) return e->variable.id && name == e->variable.id;
    if (e->type == ESHKOL_OP) return mentionsInOp(e->operation, name, depth);
    if (e->type == ESHKOL_CONS) return mentionsName(e->cons_cell.car, name, depth + 1) ||
                                       mentionsName(e->cons_cell.cdr, name, depth + 1);
    return false;
}

std::string calleeName(const Signature& sig) { return sig.name.empty() ? std::string("<procedure>") : sig.name; }

}  // namespace

std::optional<int64_t> literalNatural(const eshkol_ast_t* expr) {
    if (!expr) return std::nullopt;
    if (expr->type == ESHKOL_INT64) return expr->int64_val;
    CTValue v = CTValue::makeExpr(expr);
    auto n = v.tryEvalNat();
    if (n) return static_cast<int64_t>(*n);
    return std::nullopt;
}

void Registry::bindPrimitives(TypeEnvironment& env,
                              const std::function<void(const std::string&, TypeId)>& bind) {
    const auto resolve = [&env](const hott_type_expr_t* t) -> TypeId {
        if (!t) return BuiltinTypes::Value;
        switch (t->kind) {
            case HOTT_TYPE_INTEGER: return BuiltinTypes::Int64;
            case HOTT_TYPE_REAL: return BuiltinTypes::Float64;
            case HOTT_TYPE_NUMBER: return BuiltinTypes::Number;
            case HOTT_TYPE_BOOLEAN: return BuiltinTypes::Boolean;
            case HOTT_TYPE_STRING: return BuiltinTypes::String;
            case HOTT_TYPE_NULL: return BuiltinTypes::Null;
            case HOTT_TYPE_ANY: return BuiltinTypes::Value;
            case HOTT_TYPE_VAR: {
                auto found = env.lookupType(t->var_name ? t->var_name : "");
                return found ? *found : BuiltinTypes::Value;
            }
            default: return BuiltinTypes::Value;
        }
    };
    const auto no_alias = [](const std::string&) { return false; };
    for (const char* text : kPrimitiveSignatures) {
        const std::string s(text);
        size_t i = 0;
        detail::Sx head, result;
        if (!detail::readSx(s, i, head) || !detail::readSx(s, i, result) || !head.list || head.items.empty()) continue;
        Signature sig;
        sig.name = head.items[0].atom;
        for (size_t k = 1; k < head.items.size(); ++k) {
            const detail::Sx& p = head.items[k];
            if (!p.list || p.items.size() != 2) continue;
            hott_type_expr_t* ann = sxToAnnotation(p.items[1]);
            sig.params.push_back({p.items[0].atom, specFromAnnotation(env, ann, resolve, no_alias)});
            hott_free_type_expr(ann);
        }
        hott_type_expr_t* ret = sxToAnnotation(result);
        sig.result = specFromAnnotation(env, ret, resolve, no_alias);
        hott_free_type_expr(ret);
        sig.quantum = true;
        bind(sig.name, registerSignature(env, sig));
    }
}

TypeId Registry::registerSignature(TypeEnvironment& env, const Signature& in) {
    Signature sig = in;
    if (!sig.quantum) {
        for (const auto& [n, t] : sig.params) { (void)n; if (specIsQuantum(t)) sig.quantum = true; }
        if (specIsQuantum(sig.result)) sig.quantum = true;
    }
    std::vector<PiType::Parameter> params;
    for (const auto& [n, t] : sig.params) params.push_back({n, eraseSpec(env, t), false});
    const std::string key = sig.key(env);
    TypeId id = env.makeDependentFunctionType(params, eraseSpec(env, sig.result), sig.variadic, key);
    auto existing = by_type_.find(id.id);
    if (existing == by_type_.end()) {
        by_type_[id.id] = sig;
    }
    return id;
}

const Signature* Registry::signatureOf(TypeId fn) const {
    auto it = by_type_.find(fn.id);
    return it == by_type_.end() ? nullptr : &it->second;
}

namespace {
// The callee's index variables are renamed apart ('%' cannot occur in a
// source index), so an argument's own variables (an enclosing signature's
// rigid n) never meet the callee's n of the same spelling.
IndexExpr renamedApart(const IndexExpr& e) {
    if (!e.known) return e;
    IndexExpr r = IndexExpr::nat(e.constant);
    for (const auto& [v, a] : e.terms) r.terms["%" + v] = a;
    return r;
}
SpecType renamedApart(const SpecType& t) {
    SpecType r = t;
    if (r.kind == SpecType::Kind::Indexed) r.index = renamedApart(t.index);
    for (auto& p : r.params) p = renamedApart(p);
    if (t.result) r.result = std::make_shared<SpecType>(renamedApart(*t.result));
    return r;
}
std::string sourceSpelling(const IndexExpr& e) {
    std::string s = e.toString();
    s.erase(std::remove(s.begin(), s.end(), '%'), s.end());
    return s;
}
}  // namespace

ApplicationOutcome Registry::checkApplication(TypeEnvironment& env, const Signature& sig_in,
                                              const eshkol_ast_t* args, size_t nargs,
                                              const std::vector<TypeId>& arg_types,
                                              const eshkol_ast_t* call_site,
                                              const ViolationReporter& report) const {
    Signature sig = sig_in;
    for (auto& [pn, t] : sig.params) t = renamedApart(t);
    sig.result = renamedApart(sig.result);
    ApplicationOutcome out;
    const size_t n = std::min(nargs, sig.params.size());
    out.expected.assign(nargs, BuiltinTypes::Value);
    for (size_t i = 0; i < nargs && i < arg_types.size(); ++i) out.expected[i] = arg_types[i];
    for (size_t i = 0; i < n; ++i) {
        if (sig.params[i].second.kind == SpecType::Kind::Plain)
            out.expected[i] = sig.params[i].second.plain;
    }
    const std::string fname = calleeName(sig);

    std::set<std::string> index_vars;
    for (const auto& [pn, t] : sig.params) { (void)pn; collectIndexVariables(t, index_vars); }
    collectIndexVariables(sig.result, index_vars);

    std::map<std::string, IndexExpr> binding;
    std::set<std::string> runtime_vars;      // value parameters with run-time arguments
    std::map<std::string, TypeId> type_vars;

    // 1. value parameters: an index named like an integer parameter is that
    //    argument's compile-time value
    for (size_t i = 0; i < n; ++i) {
        const auto& [pname, spec] = sig.params[i];
        if (spec.kind != SpecType::Kind::Plain || !index_vars.count("%" + pname)) continue;
        auto v = literalNatural(&args[i]);
        if (v && *v >= 0) {
            binding["%" + pname] = IndexExpr::nat(*v);
        } else if (args[i].type == ESHKOL_INT64 && args[i].int64_val < 0) {
            report("argument " + std::to_string(i + 1) + " of '" + fname + "' is the size " +
                       std::to_string(args[i].int64_val) + ": a quantum register size is a natural number",
                   &args[i]);
            runtime_vars.insert("%" + pname);
        } else {
            runtime_vars.insert("%" + pname);
        }
    }

    const auto mentionsRuntime = [&](const IndexExpr& e) {
        for (const auto& [v, a] : e.terms) { (void)a; if (runtime_vars.count(v)) return true; }
        return false;
    };
    const auto typeName = [&](Family f, const IndexExpr& e) {
        return std::string(familyName(f)) + "<" + sourceSpelling(e) + ">";
    };
    // unbound variables of the callee (the argument's own are rigid)
    const auto calleeFree = [&](const IndexExpr& e) {
        std::vector<std::string> out;
        for (const auto& [v, a] : e.terms) { (void)a; if (!v.empty() && v[0] == '%' && !binding.count(v)) out.push_back(v); }
        return out;
    };

    // Unify a pattern with an actual index. Returns false when the pattern
    // could not be decided yet (more than one unbound variable).
    const auto unify = [&](const IndexExpr& pattern, const IndexExpr& actual, Family f,
                           const std::string& where, const eshkol_ast_t* node) -> bool {
        if (!actual.known || !pattern.known) return true;
        IndexExpr p = pattern.substitute(binding);
        if (mentionsRuntime(p)) return true;
        auto free = calleeFree(p);
        if (free.empty()) {
            if (!p.sameAs(actual)) {
                report("quantum register dimension mismatch in " + where + " of '" + fname +
                           "': expected " + typeName(f, p) + ", got " + typeName(f, actual),
                       node);
            }
            return true;
        }
        if (free.size() == 1) {
            const std::string v = free[0];
            const int64_t a = p.terms.at(v);
            IndexExpr rest = p;
            rest.terms.erase(v);
            IndexExpr d = actual.plus(rest.scaled(-1));
            if (d.isConstant()) {
                if (d.constant % a != 0 || d.constant / a < 0) {
                    report("quantum register dimension mismatch in " + where + " of '" + fname +
                               "': expected " + typeName(f, p) + ", got " + typeName(f, actual) +
                               " (no natural " + v.substr(1) + " solves it)",
                           node);
                    binding[v] = IndexExpr::unknown();
                } else {
                    binding[v] = IndexExpr::nat(d.constant / a);
                }
            } else if (a == 1) {
                binding[v] = d;
            } else {
                binding[v] = IndexExpr::unknown();
            }
            return true;
        }
        return false;
    };

    // 2. register and address arguments, until no further variable is bound
    struct Pending { size_t i; };
    std::vector<size_t> pending;
    for (size_t i = 0; i < n; ++i) {
        const auto& spec = sig.params[i].second;
        if (spec.kind != SpecType::Kind::Indexed) continue;
        out.expected[i] = arg_types[i];
        auto actual = indexedTypeOf(env, arg_types[i]);
        if (!actual || actual->first != spec.family) continue;
        if (!unify(spec.index, actual->second, spec.family,
                   "argument " + std::to_string(i + 1), &args[i]))
            pending.push_back(i);
    }
    for (int pass = 0; pass < 2 && !pending.empty(); ++pass) {
        std::vector<size_t> still;
        for (size_t i : pending) {
            const auto& spec = sig.params[i].second;
            auto actual = indexedTypeOf(env, arg_types[i]);
            if (!unify(spec.index, actual->second, spec.family,
                       "argument " + std::to_string(i + 1), &args[i]))
                still.push_back(i);
        }
        pending.swap(still);
    }

    // 3. procedure arguments whose parameters receive registers
    for (size_t i = 0; i < n; ++i) {
        const auto& spec = sig.params[i].second;
        const bool region_body = spec.kind == SpecType::Kind::Plain && spec.plain == BuiltinTypes::QRegionBody;
        if (spec.kind != SpecType::Kind::Arrow && !region_body) continue;
        out.expected[i] = arg_types[i];
        const PiType* pi = env.getFunctionType(arg_types[i]);
        if (!pi) continue;
        if (region_body) {
            // (-> qregion a), with a classical
            const TypeId r = pi->return_type;
            if (isQuantumResource(env, r)) {
                report("the body of a quantum region returns " + env.getTypeName(r) +
                           ": quantum data cannot leave the region that owns it",
                       &args[i]);
            }
            const eshkol_ast_t* lam = &args[i];
            if (lam->type == ESHKOL_OP && lam->operation.op == ESHKOL_LAMBDA_OP &&
                lam->operation.lambda_op.num_params >= 1 &&
                lam->operation.lambda_op.parameters[0].type == ESHKOL_VAR &&
                lam->operation.lambda_op.parameters[0].variable.id) {
                const std::string rname = lam->operation.lambda_op.parameters[0].variable.id;
                std::vector<const eshkol_ast_t*> tails;
                collectTails(lam->operation.lambda_op.body, tails);
                for (const auto* t : tails) {
                    if (t->type == ESHKOL_OP && t->operation.op == ESHKOL_LAMBDA_OP &&
                        mentionsName(t, rname)) {
                        report("the quantum region '" + rname +
                                   "' escapes its region inside a returned procedure",
                               t);
                    }
                }
            }
            if (sig.result.kind == SpecType::Kind::Var) type_vars[sig.result.var] = r;
            continue;
        }
        if (pi->params.size() != spec.params.size()) {
            report("argument " + std::to_string(i + 1) + " of '" + fname + "' takes " +
                       std::to_string(pi->params.size()) + " parameters; it receives " +
                       std::to_string(spec.params.size()),
                   &args[i]);
            continue;
        }
        for (size_t j = 0; j < spec.params.size(); ++j) {
            const SpecType& want = spec.params[j];
            if (want.kind != SpecType::Kind::Indexed) continue;
            const IndexExpr expect = want.index.substitute(binding);
            const TypeId declared = pi->params[j].type;
            auto have = indexedTypeOf(env, declared);
            if (!have) {
                report("parameter " + std::to_string(j + 1) + " of the procedure passed to '" + fname +
                           "' receives " + typeName(want.family, expect) +
                           "; declare it with that type so its linear use is checked",
                       &args[i]);
                continue;
            }
            if (have->first != want.family) {
                report("parameter " + std::to_string(j + 1) + " of the procedure passed to '" + fname +
                           "' is declared " + env.getTypeName(declared) + " but receives " +
                           typeName(want.family, expect),
                       &args[i]);
                continue;
            }
            if (!mentionsRuntime(expect) && expect.known && have->second.known &&
                calleeFree(expect).empty() && !expect.sameAs(have->second)) {
                report("quantum register dimension mismatch in parameter " + std::to_string(j + 1) +
                           " of the procedure passed to '" + fname + "': it receives " +
                           typeName(want.family, expect) + ", declared " +
                           typeName(have->first, have->second),
                       &args[i]);
            }
        }
        if (spec.result && spec.result->kind == SpecType::Kind::Var) {
            type_vars[spec.result->var] = pi->return_type;
        }
    }

    // 4. qubit addresses: in range, and pairwise distinct within one register
    std::map<std::string, std::vector<std::pair<int64_t, size_t>>> addresses;
    for (size_t i = 0; i < n; ++i) {
        const auto& spec = sig.params[i].second;
        if (spec.kind != SpecType::Kind::Indexed || spec.family != Family::QIdx) continue;
        std::optional<int64_t> v;
        if (args[i].type == ESHKOL_INT64) v = args[i].int64_val;
        else v = literalNatural(&args[i]);
        if (!v) continue;
        const IndexExpr bound = spec.index.substitute(binding);
        if (*v < 0) {
            report("qubit index " + std::to_string(*v) + " in argument " + std::to_string(i + 1) +
                       " of '" + fname + "' is negative",
                   &args[i]);
            continue;
        }
        if (auto b = bound.value(); b && !mentionsRuntime(bound) && *v >= *b) {
            report("qubit index " + std::to_string(*v) + " in argument " + std::to_string(i + 1) +
                       " of '" + fname + "' is out of range for qreg<" + sourceSpelling(bound) + ">",
                   &args[i]);
        }
        auto& seen = addresses[spec.index.toString()];
        for (const auto& [w, j] : seen) {
            if (w == *v) {
                report("qubit " + std::to_string(*v) + " is used twice by one '" + fname +
                           "' (arguments " + std::to_string(j + 1) + " and " + std::to_string(i + 1) +
                           "): a gate cannot act on a copy of its own qubit (no-cloning)",
                       &args[i]);
            }
        }
        seen.push_back({*v, i});
    }

    // 5. the result
    const SpecType& r = sig.result;
    switch (r.kind) {
        case SpecType::Kind::Indexed: {
            IndexExpr e = r.index.substitute(binding);
            if (mentionsRuntime(e) || !calleeFree(e).empty()) e = IndexExpr::unknown();
            // a rigid variable of an enclosing signature stays symbolic
            out.result = indexedType(env, r.family, e);
            break;
        }
        case SpecType::Kind::Var: {
            auto it = type_vars.find(r.var);
            out.result = it != type_vars.end() ? it->second : BuiltinTypes::Value;
            break;
        }
        default:
            out.result = eraseSpec(env, r);
            break;
    }
    (void)call_site;
    return out;
}

bool checkDeclaredResult(const TypeEnvironment& env, TypeId declared, TypeId actual,
                         const std::string& what, const eshkol_ast_t* site,
                         const ViolationReporter& report) {
    auto d = indexedTypeOf(env, declared);
    auto a = indexedTypeOf(env, actual);
    if (!d || !a || d->first != a->first) return false;
    if (!d->second.known || !a->second.known || d->second.sameAs(a->second)) return false;
    report("quantum register dimension mismatch: " + what + " is declared " +
               std::string(familyName(d->first)) + "<" + d->second.toString() + "> but is " +
               std::string(familyName(a->first)) + "<" + a->second.toString() + ">",
           site);
    return true;
}

}  // namespace eshkol::hott::quantum
