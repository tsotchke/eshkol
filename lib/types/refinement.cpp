/**
 * @file refinement.cpp
 * @brief Refinement-type verification (ADR-0044). Rules and severities are in
 *        eshkol/types/refinement.h and docs/design/adr/0044-refinement-types.md.
 *
 * The walk is a symbolic evaluation of each top-level form. Every expression
 * evaluates to a symbolic value: a numeric term (Int, or Real with a flag
 * saying whether the arithmetic that produced it was exact), a formula for a
 * Boolean, an object term (vectors, strings, lists, tensors, closures), or
 * Unknown. Facts accumulate in two places:
 *
 *   - path facts: refinements assumed for parameters and call results, and
 *     the conditions of the branches the walk is inside; scoped;
 *   - definitions: constraints that only introduce a fresh variable (the
 *     quotient and remainder of a division by a constant, the rounding
 *     interval of an inexact operation, the non-negativity of a length);
 *     every such constraint is satisfiable for any value of the rest, so
 *     keeping them for the whole form is sound.
 *
 * Soundness choices:
 *   - a variable assigned by `set!` anywhere in its function is re-read as a
 *     fresh value (or as a fresh value satisfying its declared refinement);
 *   - inexact (floating-point) arithmetic yields a fresh real within the
 *     round-to-nearest error interval of the exact result;
 *   - a value of unknown numeric type is treated as inexact;
 *   - forms the walk does not model are still walked for obligations, in a
 *     scope that hides every outer binding and with no path facts.
 */
#include "eshkol/types/refinement.h"

#include "eshkol/types/smt.h"
#include "eshkol/types/type_checker.h"
#include <eshkol/frontend/syntax_color.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

namespace eshkol::hott {

using smt::Integer;
using smt::Rational;
using smt::Sort;
using smt::Term;

namespace {

// ---------------------------------------------------------------------------
// AST helpers
// ---------------------------------------------------------------------------

std::string baseName(const char* id) {
    if (!id) return {};
    return std::string(id, eshkol_syntax_base_length(id));
}

std::string varName(const eshkol_ast_t* a) {
    return (a && a->type == ESHKOL_VAR && a->variable.id) ? baseName(a->variable.id) : std::string();
}

bool isOp(const eshkol_ast_t* a, eshkol_op_t op) {
    return a && a->type == ESHKOL_OP && a->operation.op == op;
}

// Every evaluated sub-expression of @p a, in source order.
void children(const eshkol_ast_t* a, std::vector<const eshkol_ast_t*>& out) {
    if (!a) return;
    auto add = [&](const eshkol_ast_t* c) {
        if (c) out.push_back(c);
    };
    if (a->type == ESHKOL_CONS) {
        add(a->cons_cell.car);
        add(a->cons_cell.cdr);
        return;
    }
    if (a->type != ESHKOL_OP) return;
    const auto& op = a->operation;
    switch (op.op) {
        case ESHKOL_QUOTE_OP:
        case ESHKOL_DEFINE_SYNTAX_OP:
        case ESHKOL_LET_SYNTAX_OP:
        case ESHKOL_LETREC_SYNTAX_OP:
        case ESHKOL_DEFINE_TYPE_OP:
        case ESHKOL_TYPE_ANNOTATION_OP:
            return;
        case ESHKOL_LAMBDA_OP:
            add(op.lambda_op.body);
            return;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                const auto& b = op.let_op.bindings[i];
                if (b.type == ESHKOL_CONS) add(b.cons_cell.cdr);
                else add(&b);
            }
            add(op.let_op.body);
            return;
        case ESHKOL_DEFINE_OP:
            add(op.define_op.value);
            return;
        case ESHKOL_SET_OP:
            add(op.set_op.value);
            return;
        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                add(&op.sequence_op.expressions[i]);
            return;
        case ESHKOL_EXTERN_OP:
            return;
        case ESHKOL_WITH_REGION_OP:
            for (uint64_t i = 0; i < op.with_region_op.num_body_exprs; ++i)
                add(&op.with_region_op.body[i]);
            return;
        case ESHKOL_BORROW_OP:
            add(op.borrow_op.value);
            for (uint64_t i = 0; i < op.borrow_op.num_body_exprs; ++i) add(&op.borrow_op.body[i]);
            return;
        case ESHKOL_TENSOR_OP:
            for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i) add(&op.tensor_op.elements[i]);
            return;
        case ESHKOL_MATCH_OP:
            add(op.match_op.expr);
            for (uint64_t i = 0; i < op.match_op.num_clauses; ++i) {
                add(op.match_op.clauses[i].guard);
                add(op.match_op.clauses[i].body);
            }
            return;
        case ESHKOL_CALL_CC_OP:
            add(op.call_cc_op.proc);
            return;
        case ESHKOL_DYNAMIC_WIND_OP:
            add(op.dynamic_wind_op.before);
            add(op.dynamic_wind_op.thunk);
            add(op.dynamic_wind_op.after);
            return;
        case ESHKOL_GUARD_OP:
            for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i) add(&op.guard_op.clauses[i]);
            for (uint64_t i = 0; i < op.guard_op.num_body_exprs; ++i) add(&op.guard_op.body[i]);
            return;
        case ESHKOL_CALL_WITH_VALUES_OP:
            add(op.call_with_values_op.producer);
            add(op.call_with_values_op.consumer);
            return;
        case ESHKOL_LET_VALUES_OP:
        case ESHKOL_LET_STAR_VALUES_OP:
            for (uint64_t i = 0; i < op.let_values_op.num_bindings; ++i)
                add(&op.let_values_op.producers[i]);
            add(op.let_values_op.body);
            return;
        case ESHKOL_VALUES_OP:
            for (uint64_t i = 0; i < op.values_op.num_values; ++i) add(&op.values_op.expressions[i]);
            return;
        case ESHKOL_DIFF_OP:
            add(op.diff_op.expression);
            return;
        case ESHKOL_GRADIENT_OP:
            add(op.gradient_op.function);
            add(op.gradient_op.point);
            return;
        case ESHKOL_DERIVATIVE_OP:
            add(op.derivative_op.function);
            add(op.derivative_op.point);
            return;
        case ESHKOL_TAYLOR_OP:
        case ESHKOL_DERIVATIVE_N_OP:
            add(op.taylor_op.function);
            add(op.taylor_op.point);
            add(op.taylor_op.order);
            return;
        case ESHKOL_DIRECTIONAL_DERIV_OP:
            add(op.directional_deriv_op.function);
            add(op.directional_deriv_op.point);
            add(op.directional_deriv_op.direction);
            return;
        case ESHKOL_JACOBIAN_OP:
            add(op.jacobian_op.function);
            add(op.jacobian_op.point);
            return;
        case ESHKOL_HESSIAN_OP:
            add(op.hessian_op.function);
            add(op.hessian_op.point);
            return;
        case ESHKOL_DIVERGENCE_OP:
            add(op.divergence_op.function);
            add(op.divergence_op.point);
            return;
        case ESHKOL_CURL_OP:
            add(op.curl_op.function);
            add(op.curl_op.point);
            return;
        case ESHKOL_LAPLACIAN_OP:
            add(op.laplacian_op.function);
            add(op.laplacian_op.point);
            return;
        case ESHKOL_OWNED_OP:
            add(op.owned_op.value);
            return;
        case ESHKOL_MOVE_OP:
            add(op.move_op.value);
            return;
        case ESHKOL_SHARED_OP:
            add(op.shared_op.value);
            return;
        case ESHKOL_WEAK_REF_OP:
            add(op.weak_ref_op.value);
            return;
        case ESHKOL_THE_OP:
            add(op.the_op.expr);
            return;
        case ESHKOL_RAISE_OP:
            add(op.raise_op.exception);
            return;
        case ESHKOL_QUASIQUOTE_OP:
        case ESHKOL_CALL_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_DO_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_MAKE_PARAMETER_OP:
        default:
            if (op.op == ESHKOL_CALL_OP || op.op == ESHKOL_COND_OP || op.op == ESHKOL_IF_OP ||
                op.op == ESHKOL_CASE_OP || op.op == ESHKOL_DO_OP || op.op == ESHKOL_WHEN_OP ||
                op.op == ESHKOL_UNLESS_OP || op.op == ESHKOL_MAKE_PARAMETER_OP ||
                op.op == ESHKOL_QUASIQUOTE_OP ||
                (op.op >= ESHKOL_UNIFY_OP && op.op <= ESHKOL_WORKSPACE_PRED_OP) ||
                op.op == ESHKOL_KB_QUERY_PREFIX_OP ||
                (op.op >= ESHKOL_DNC_MAKE_OP && op.op <= ESHKOL_SDNC_PRED_OP)) {
                add(op.call_op.func);
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) add(&op.call_op.variables[i]);
            }
            return;
    }
}

void forEachNode(const eshkol_ast_t* a, const std::function<void(const eshkol_ast_t*)>& f) {
    if (!a) return;
    f(a);
    std::vector<const eshkol_ast_t*> cs;
    children(a, cs);
    for (const auto* c : cs) forEachNode(c, f);
}

bool typeHasRefinement(const hott_type_expr_t* t, int depth = 0) {
    if (!t || depth > 64) return false;
    switch (t->kind) {
        case HOTT_TYPE_REFINE:
            return true;
        case HOTT_TYPE_ARROW:
            for (uint64_t i = 0; i < t->arrow.num_params; ++i) {
                if (t->arrow.param_types && typeHasRefinement(t->arrow.param_types[i], depth + 1))
                    return true;
            }
            return typeHasRefinement(t->arrow.return_type, depth + 1);
        case HOTT_TYPE_FORALL:
            return typeHasRefinement(t->forall.body, depth + 1);
        default:
            return false;
    }
}

// Source text of a predicate, for diagnostics.
std::string show(const eshkol_ast_t* a) {
    if (!a) return "?";
    switch (a->type) {
        case ESHKOL_VAR: return varName(a);
        case ESHKOL_INT64: return std::to_string(a->int64_val);
        case ESHKOL_INT8: case ESHKOL_INT16: case ESHKOL_INT32:
            return std::to_string(a->int64_val);
        case ESHKOL_DOUBLE: {
            std::ostringstream o;
            o << a->double_val;
            return o.str();
        }
        case ESHKOL_BOOL: return a->int64_val ? "#t" : "#f";
        case ESHKOL_BIGNUM_LITERAL: return a->str_val.ptr ? std::string(a->str_val.ptr, a->str_val.size) : "?";
        default: break;
    }
    if (a->type != ESHKOL_OP) return "...";
    const auto& op = a->operation;
    std::ostringstream o;
    auto list = [&](const char* head, const eshkol_ast_t* xs, uint64_t n) {
        o << "(" << head;
        for (uint64_t i = 0; i < n; ++i) o << " " << show(&xs[i]);
        o << ")";
    };
    switch (op.op) {
        case ESHKOL_CALL_OP:
            if (varName(op.call_op.func) == "make-rational" && op.call_op.num_vars == 2) {
                return show(&op.call_op.variables[0]) + "/" + show(&op.call_op.variables[1]);
            }
            list(show(op.call_op.func).c_str(), op.call_op.variables, op.call_op.num_vars);
            return o.str();
        case ESHKOL_IF_OP:
            list("if", op.call_op.variables, op.call_op.num_vars);
            return o.str();
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            list(op.op == ESHKOL_AND_OP ? "and" : "or", op.sequence_op.expressions,
                 op.sequence_op.num_expressions);
            return o.str();
        default:
            return "(...)";
    }
}

// ---------------------------------------------------------------------------
// Symbolic values
// ---------------------------------------------------------------------------

enum class VK { Unknown, Num, Bool, Obj };

struct Cell;

struct SVal {
    VK kind = VK::Unknown;
    Term t = nullptr;
    bool inexact = false;
    std::shared_ptr<Cell> cell;  // stable identity of an Unknown variable
};

struct Refinement {
    const hott_type_expr_t* base = nullptr;
    std::string binder;  // empty: the predicate does not mention the subject
    const eshkol_ast_t* pred = nullptr;
    std::string text() const { return show(pred); }
};

enum class BaseKind { Unknown, Int, Nat, Rat, Real, Bool, Obj };

struct Cell {
    std::string name;
    SVal val;
    bool mutated = false;
    BaseKind kind = BaseKind::Unknown;
    std::optional<Refinement> ref;
    const struct Contract* contract = nullptr;  // function bindings with a contract
    std::map<int, Term> lazy;  // sort -> stable variable for an Unknown value
};

struct Scope {
    std::map<std::string, std::shared_ptr<Cell>> vars;
    const Scope* parent = nullptr;
    bool opaque = false;  // lookups stop here
    const std::shared_ptr<Cell>* lookup(const std::string& n) const {
        for (const Scope* s = this; s; s = s->parent) {
            auto it = s->vars.find(n);
            if (it != s->vars.end()) return &it->second;
            if (s->opaque) return nullptr;
        }
        return nullptr;
    }
};

struct Param {
    std::string name;
    const hott_type_expr_t* type = nullptr;
    std::optional<Refinement> ref;
};

struct Contract {
    std::string name;
    std::vector<Param> params;
    bool variadic = false;
    std::optional<Refinement> result;
    bool ambiguous = false;
};

struct Ctx {
    const Scope* scope = nullptr;
    std::vector<Term> path;
    bool typed = false;          // inside a function that carries refinement annotations
    std::string where;           // enclosing function, for diagnostics
    std::set<std::string> mutated;
    bool predicate = false;      // evaluating a refinement predicate
    bool malformed = false;      // predicate used a form outside the predicate language
    std::string malformed_what;
};

// Builtins a refinement predicate may call: pure, total on their domain.
const std::set<std::string>& predicateBuiltins() {
    static const std::set<std::string> s{
        "+", "-", "*", "/", "<", "<=", ">", ">=", "=", "not", "and", "or", "abs", "min", "max",
        "quotient", "remainder", "modulo", "zero?", "positive?", "negative?", "even?", "odd?",
        "vector-length", "string-length", "length", "tensor-length", "floor", "ceiling", "round",
        "truncate", "exact", "inexact", "exact->inexact", "inexact->exact", "make-rational",
        "eq?", "eqv?", "equal?", "null?", "pair?", "boolean=?", "implies"};
    return s;
}

// ---------------------------------------------------------------------------
// The checker
// ---------------------------------------------------------------------------

class Checker {
public:
    Checker(TypeChecker& tc, const RefinementOptions& opt) : tc_(tc), opt_(opt) {
        internal_ = smt::makeInternalSolver(smt::Budget{opt.budget});
        if (opt.solver == "z3" || opt.solver == "cross") {
            z3_ = smt::makeZ3Solver();
            if (!z3_) {
                std::fprintf(stderr,
                             "[WARN] Refinement: ESHKOL_REFINEMENT_SOLVER=%s, but this build has no "
                             "z3 backend (configure with -DESHKOL_REFINEMENT_Z3=ON); using the "
                             "built-in procedure\n",
                             opt.solver.c_str());
            }
        }
    }

    RefinementReport run(const eshkol_ast_t* asts, size_t n) {
        report_.ran = true;
        for (size_t i = 0; i < n; ++i) collectDeclarations(&asts[i]);
        for (size_t i = 0; i < n; ++i) {
            forEachNode(&asts[i], [&](const eshkol_ast_t* a) {
                if (isOp(a, ESHKOL_SET_OP) && a->operation.set_op.name) {
                    mutated_globals_.insert(baseName(a->operation.set_op.name));
                }
            });
        }
        for (size_t i = 0; i < n; ++i) collectContracts(&asts[i]);
        for (size_t i = 0; i < n; ++i) {
            defs_.clear();
            Ctx c;
            c.scope = &globals_;
            topLevel(&asts[i], c);
        }
        return report_;
    }

private:
    TypeChecker& tc_;
    RefinementOptions opt_;
    smt::ExactScope exact_;  // outlives tm_: declared first
    smt::TermManager tm_;
    std::unique_ptr<smt::Solver> internal_, z3_;
    RefinementReport report_;

    Scope globals_;
    std::set<std::string> mutated_globals_;
    std::map<std::string, const hott_type_expr_t*> aliases_;
    std::map<std::string, const hott_type_expr_t*> declared_;  // (: name type)
    std::map<std::string, Contract> contracts_;
    std::vector<Term> defs_;
    uint64_t fresh_ = 0;

    // -- declarations ---------------------------------------------------------

    void collectDeclarations(const eshkol_ast_t* a) {
        forEachNode(a, [&](const eshkol_ast_t* n) {
            if (isOp(n, ESHKOL_DEFINE_TYPE_OP) && n->operation.define_type_op.name &&
                n->operation.define_type_op.num_type_params == 0) {
                aliases_[baseName(n->operation.define_type_op.name)] =
                    n->operation.define_type_op.type_expr;
            }
        });
        if (isOp(a, ESHKOL_TYPE_ANNOTATION_OP) && a->operation.type_annotation_op.name) {
            declared_[baseName(a->operation.type_annotation_op.name)] =
                a->operation.type_annotation_op.type_expr;
        }
    }

    // The refinement a type denotes, following aliases.
    std::optional<Refinement> refinementOf(const hott_type_expr_t* t, int depth = 0) {
        if (!t || depth > 32) return std::nullopt;
        if (t->kind == HOTT_TYPE_REFINE) {
            Refinement r;
            r.base = t->refine.base;
            r.pred = t->refine.predicate;
            if (t->refine.binder) r.binder = baseName(t->refine.binder);
            return r;
        }
        if (t->kind == HOTT_TYPE_VAR && t->var_name) {
            auto it = aliases_.find(baseName(t->var_name));
            if (it != aliases_.end()) {
                auto r = refinementOf(it->second, depth + 1);
                if (r && r->binder.empty()) r->binder = soleFreeVariable(*r, {});
                return r;
            }
        }
        return std::nullopt;
    }

    BaseKind baseKindOf(const hott_type_expr_t* t) {
        if (!t) return BaseKind::Unknown;
        TypeId id = tc_.resolveType(t);
        namespace B = BuiltinTypes;
        for (TypeId nat : {B::Natural, B::UInt8, B::UInt16, B::UInt32, B::UInt64, B::USize}) {
            if (id == nat) return BaseKind::Nat;
        }
        for (TypeId in : {B::Integer, B::Int64, B::BigInt, B::Int8, B::Int16, B::Int32, B::ISize}) {
            if (id == in) return BaseKind::Int;
        }
        if (id == B::Rational) return BaseKind::Rat;
        if (id == B::Real || id == B::Float64 || id == B::Float32 || id == B::Number) return BaseKind::Real;
        if (id == B::Boolean) return BaseKind::Bool;
        if (id == B::Value) return BaseKind::Unknown;
        return BaseKind::Obj;
    }

    static void freeVariables(const eshkol_ast_t* a, std::vector<std::string>& out) {
        if (!a) return;
        if (a->type == ESHKOL_VAR) {
            std::string n = varName(a);
            if (!n.empty() && std::find(out.begin(), out.end(), n) == out.end()) out.push_back(n);
            return;
        }
        if (isOp(a, ESHKOL_CALL_OP)) {
            for (uint64_t i = 0; i < a->operation.call_op.num_vars; ++i)
                freeVariables(&a->operation.call_op.variables[i], out);
            return;
        }
        std::vector<const eshkol_ast_t*> cs;
        children(a, cs);
        for (const auto* c : cs) freeVariables(c, out);
    }

    // The binder of a 2-argument (Refine T phi): the single free variable of
    // phi that names nothing in @p in_scope ("" when phi mentions none).
    std::string soleFreeVariable(const Refinement& r, const std::set<std::string>& in_scope,
                                 bool* ambiguous = nullptr) {
        std::vector<std::string> fv;
        freeVariables(r.pred, fv);
        std::vector<std::string> cands;
        for (const auto& v : fv) {
            if (!in_scope.count(v) && !globals_.lookup(v)) cands.push_back(v);
        }
        if (cands.size() == 1) return cands[0];
        if (cands.size() > 1 && ambiguous) *ambiguous = true;
        return {};
    }

    std::optional<Refinement> refinementFor(const hott_type_expr_t* t, const std::string& subject,
                                            const std::set<std::string>& in_scope,
                                            const eshkol_ast_t* at) {
        auto r = refinementOf(t);
        if (!r || !r->binder.empty()) return r;
        std::vector<std::string> fv;
        freeVariables(r->pred, fv);
        if (!subject.empty() && std::find(fv.begin(), fv.end(), subject) != fv.end()) {
            r->binder = subject;
            return r;
        }
        bool ambiguous = false;
        r->binder = soleFreeVariable(*r, in_scope, &ambiguous);
        if (ambiguous) {
            fatal("cannot tell which variable of (Refine ... " + r->text() +
                      ") the refinement is about; write (Refine type binder predicate)",
                  at);
            return std::nullopt;
        }
        return r;
    }

    void collectContracts(const eshkol_ast_t* top) {
        forEachNode(top, [&](const eshkol_ast_t* a) {
            if (!isOp(a, ESHKOL_DEFINE_OP) || !a->operation.define_op.name) return;
            const auto& d = a->operation.define_op;
            std::string name = baseName(d.name);
            const eshkol_ast_t* params = nullptr;
            uint64_t nparams = 0;
            hott_type_expr_t** ptypes = nullptr;
            const hott_type_expr_t* rtype = nullptr;
            bool variadic = false;
            if (d.is_function) {
                params = d.parameters;
                nparams = d.num_params;
                ptypes = d.param_types;
                rtype = d.return_type;
                variadic = d.is_variadic;
            } else if (isOp(d.value, ESHKOL_LAMBDA_OP)) {
                const auto& l = d.value->operation.lambda_op;
                params = l.parameters;
                nparams = l.num_params;
                ptypes = l.param_types;
                rtype = l.return_type;
                variadic = l.is_variadic;
            } else {
                return;
            }
            const hott_type_expr_t* sig = nullptr;
            auto decl = declared_.find(name);
            if (decl != declared_.end() && decl->second && decl->second->kind == HOTT_TYPE_ARROW)
                sig = decl->second;
            Contract c;
            c.name = name;
            c.variadic = variadic;
            bool any = false;
            std::set<std::string> earlier;
            for (uint64_t i = 0; i < nparams; ++i) {
                Param p;
                p.name = varName(&params[i]);
                p.type = ptypes ? ptypes[i] : nullptr;
                if (!p.type && sig && i < sig->arrow.num_params && sig->arrow.param_types)
                    p.type = sig->arrow.param_types[i];
                p.ref = refinementFor(p.type, p.name, earlier, a);
                any |= p.ref.has_value();
                earlier.insert(p.name);
                c.params.push_back(std::move(p));
            }
            if (!rtype && sig) rtype = sig->arrow.return_type;
            c.result = refinementFor(rtype, "", earlier, a);
            any |= c.result.has_value();
            if (!any) return;
            auto it = contracts_.find(name);
            if (it != contracts_.end()) {
                it->second.ambiguous = true;
                return;
            }
            contracts_.emplace(name, std::move(c));
        });
    }

    // -- diagnostics ------------------------------------------------------------

    static std::string where(const eshkol_ast_t* at) {
        std::string s;
        if (at && at->line > 0) {
            s = " (line " + std::to_string(at->line);
            if (at->column > 0 && at->column < 100000) s += ":" + std::to_string(at->column);
            s += ")";
        }
        return s;
    }

    void fatal(const std::string& msg, const eshkol_ast_t* at) {
        if (opt_.unsafe_mode) return;
        std::fprintf(stderr, "[ERROR] Refinement error: %s%s\n", eshkol_syntax_source_text(msg).c_str(),
                     where(at).c_str());
        report_.errors++;
    }

    void issue(const std::string& msg, const eshkol_ast_t* at) {
        report_.issues++;
        tc_.reportTypeIssue("refinement " + msg, at);
    }

    // -- fresh terms ------------------------------------------------------------

    Term freshVar(const std::string& stem, Sort s) {
        return tm_.mkVar(tm_.freshName(stem), s);
    }

    Term namedVar(const std::string& source, Sort s) {
        // Source name first, so counterexamples read naturally; '@' starts the
        // disambiguating suffix that display strips.
        return tm_.mkVar(source + "@" + std::to_string(++fresh_), s);
    }

    static Sort sortOf(BaseKind k) {
        switch (k) {
            case BaseKind::Int:
            case BaseKind::Nat: return Sort::Int;
            case BaseKind::Rat:
            case BaseKind::Real:
            case BaseKind::Unknown: return Sort::Real;
            case BaseKind::Bool: return Sort::Bool;
            case BaseKind::Obj: return Sort::Obj;
        }
        return Sort::Real;
    }

    // A fresh value of base kind @p k named after @p source, with the facts the base type implies.
    SVal freshOf(BaseKind k, const std::string& source) {
        SVal v;
        switch (k) {
            case BaseKind::Unknown: {
                v.cell = std::make_shared<Cell>();
                v.cell->name = source;
                return v;
            }
            case BaseKind::Bool:
                v.kind = VK::Bool;
                v.t = namedVar(source, Sort::Bool);
                return v;
            case BaseKind::Obj:
                v.kind = VK::Obj;
                v.t = namedVar(source, Sort::Obj);
                return v;
            default:
                v.kind = VK::Num;
                v.t = namedVar(source, sortOf(k));
                v.inexact = k == BaseKind::Real;
                if (k == BaseKind::Nat) defs_.push_back(tm_.mkLe(tm_.mkInt(Integer(0)), v.t));
                return v;
        }
    }

    Term asNum(const SVal& v, bool* inexact = nullptr) {
        if (inexact) *inexact = v.inexact || v.kind != VK::Num;
        if (v.kind == VK::Num) return v.t;
        if (v.cell) return lazyTerm(*v.cell, Sort::Real);
        return freshVar("num", Sort::Real);
    }

    Term asObj(const SVal& v) {
        if (v.kind == VK::Obj) return v.t;
        if (v.cell) return lazyTerm(*v.cell, Sort::Obj);
        return freshVar("obj", Sort::Obj);
    }

    // The truth of a value used as a condition: everything but #f is true.
    Term truthy(const SVal& v) {
        switch (v.kind) {
            case VK::Bool: return v.t;
            case VK::Num:
            case VK::Obj: return tm_.mkTrue();
            case VK::Unknown:
                if (v.cell) return lazyTerm(*v.cell, Sort::Bool);
                return freshVar("cond", Sort::Bool);
        }
        return tm_.mkTrue();
    }

    Term lazyTerm(Cell& c, Sort s) {
        auto it = c.lazy.find(static_cast<int>(s));
        if (it != c.lazy.end()) return it->second;
        Term t = namedVar(c.name.empty() ? "value" : c.name, s);
        c.lazy.emplace(static_cast<int>(s), t);
        return t;
    }

    SVal num(Term t, bool inexact) {
        SVal v;
        v.kind = VK::Num;
        v.t = t;
        v.inexact = inexact;
        return v;
    }
    SVal boolean(Term t) {
        SVal v;
        v.kind = VK::Bool;
        v.t = t;
        return v;
    }
    SVal object(Term t) {
        SVal v;
        v.kind = VK::Obj;
        v.t = t;
        return v;
    }
    SVal unknown() { return SVal{}; }

    // r within the round-to-nearest error of the exact result s.
    SVal rounded(Term s) {
        static const Rational eps(Integer(1), Integer::parse("9007199254740992").value());  // 2^-53
        Term r = freshVar("fl", Sort::Real);
        Term zero = tm_.mkReal(Rational(0));
        Term mag = tm_.mkIte(tm_.mkLe(zero, s), s, tm_.mkNeg(s));
        // eta covers subnormal results: 2^-1074 < 2^-1000
        Integer two_1000(1);
        for (int i = 0; i < 1000; ++i) two_1000 *= Integer(2);
        Term slack = tm_.mkAdd(tm_.mkScale(eps, mag), tm_.mkReal(Rational(Integer(1), two_1000)));
        defs_.push_back(tm_.mkLe(tm_.mkSub(r, s), slack));
        defs_.push_back(tm_.mkLe(tm_.mkSub(s, r), slack));
        return num(r, true);
    }

    SVal arith(Term exact, bool inexact) { return inexact ? rounded(exact) : num(exact, false); }

    Term length(const char* fn, const SVal& v) {
        Term t = tm_.mkApp(fn, Sort::Int, {asObj(v)});
        defs_.push_back(tm_.mkLe(tm_.mkInt(Integer(0)), t));
        return t;
    }

    // -- obligations ------------------------------------------------------------

    enum class Verdict { Proved, NotProved, Contradicted, Undecided };

    struct Answer {
        Verdict verdict;
        smt::Model model;
        std::string reason;
        std::vector<Term> query;
    };

    smt::Outcome solve(const std::vector<Term>& q) {
        smt::Outcome o = (z3_ && opt_.solver == "z3") ? z3_->check(tm_, q) : internal_->check(tm_, q);
        if (z3_ && opt_.solver == "cross") {
            smt::Outcome z = z3_->check(tm_, q);
            if (o.result != smt::Result::Unknown && z.result != smt::Result::Unknown &&
                o.result != z.result) {
                fatal(std::string("solvers disagree (built-in ") + smt::resultName(o.result) +
                          ", z3 " + smt::resultName(z.result) + ") on:\n" +
                          smt::toSmtLib2Script(tm_, q, true),
                      nullptr);
            }
        }
        return o;
    }

    Answer prove(const Ctx& c, Term goal) {
        Answer a;
        std::vector<Term> q = defs_;
        q.insert(q.end(), c.path.begin(), c.path.end());
        std::vector<Term> base = q;
        q.push_back(tm_.mkNot(goal));
        smt::Outcome o = solve(q);
        a.query = q;
        if (o.result == smt::Result::Unsat) {
            a.verdict = Verdict::Proved;
            return a;
        }
        if (o.result == smt::Result::Unknown) {
            a.verdict = Verdict::Undecided;
            a.reason = o.reason;
            return a;
        }
        a.model = std::move(o.model);
        base.push_back(goal);
        smt::Outcome p = solve(base);
        a.verdict = p.result == smt::Result::Unsat ? Verdict::Contradicted : Verdict::NotProved;
        return a;
    }

    std::string counterexample(const Answer& a) {
        std::set<Term> seen;
        std::vector<Term> vars, apps;
        std::function<void(Term)> walk = [&](Term t) {
            if (!seen.insert(t).second) return;
            if (t->kind == smt::Kind::Var && t->name[0] != '%') vars.push_back(t);
            if (t->kind == smt::Kind::App && !smt::isOverApproximatedSymbol(t->name)) {
                bool named = true;
                for (Term x : t->args) named &= x->kind == smt::Kind::Var && x->name[0] != '%';
                if (named) apps.push_back(t);
            }
            for (Term x : t->args) walk(x);
        };
        for (Term t : a.query) walk(t);
        auto display = [](const std::string& n) { return n.substr(0, n.find('@')); };
        std::vector<std::string> parts;
        std::set<std::string> shown;
        for (Term v : vars) {
            if (v->sort == Sort::Obj) continue;
            auto val = smt::evaluateToString(v, a.model);
            std::string item = display(v->name) + " = " + (val ? *val : "?");
            if (val && shown.insert(item).second) parts.push_back(item);
        }
        for (Term app : apps) {
            auto val = smt::evaluateToString(app, a.model);
            if (!val) continue;
            std::string item = "(" + app->name;
            for (Term x : app->args) item += " " + display(x->name);
            item += ") = " + *val;
            if (shown.insert(item).second) parts.push_back(item);
        }
        std::string s;
        for (size_t i = 0; i < parts.size(); ++i) s += (i ? ", " : "") + parts[i];
        return s.empty() ? "any value" : s;
    }

    // Prove @p goal; report by severity. @p boundary: the obligation comes
    // from unannotated code calling an annotated function.
    void oblige(const Ctx& c, Term goal, const std::string& what, const std::string& pred,
                const eshkol_ast_t* at, bool boundary) {
        report_.obligations++;
        Answer a = prove(c, goal);
        std::string subject = what + " must satisfy " + pred;
        if (!c.where.empty()) subject += " in '" + c.where + "'";
        switch (a.verdict) {
            case Verdict::Proved:
                report_.proved++;
                return;
            case Verdict::Contradicted:
                fatal("refinement violated: " + subject +
                          ", and no execution reaching this point satisfies it (for example " +
                          counterexample(a) + ")",
                      at);
                return;
            case Verdict::NotProved:
                if (boundary) {
                    issue("not proved at a gradual boundary: " + subject + "; counterexample: " +
                              counterexample(a),
                          at);
                } else {
                    fatal("refinement not proved: " + subject + "; counterexample: " +
                              counterexample(a),
                          at);
                }
                return;
            case Verdict::Undecided:
                issue("undecided: " + subject + " (" + a.reason + ")", at);
                return;
        }
    }

    // Formula for refinement @p r of subject @p subject, with @p extra
    // bindings (parameters visible to the predicate). nullopt when malformed.
    std::optional<Term> predicate(const Refinement& r, const SVal& subject,
                                  const std::map<std::string, SVal>& extra, const Scope* outer,
                                  const eshkol_ast_t* at) {
        Scope s;
        s.parent = outer ? outer : &globals_;
        for (const auto& [n, v] : extra) {
            auto cell = std::make_shared<Cell>();
            cell->name = n;
            cell->val = v;
            s.vars[n] = cell;
        }
        if (!r.binder.empty()) {
            auto cell = std::make_shared<Cell>();
            cell->name = r.binder;
            cell->val = subject;
            s.vars[r.binder] = cell;
        }
        Ctx pc;
        pc.scope = &s;
        pc.predicate = true;
        SVal v = eval(r.pred, pc);
        if (pc.malformed) {
            fatal("refinement predicate " + r.text() + " uses " + pc.malformed_what +
                      ", which is not in the refinement predicate language (arithmetic, "
                      "comparisons, Boolean connectives, if, lengths)",
                  at);
            return std::nullopt;
        }
        return truthy(v);
    }

    // Contract predicates (parameters, results, loop variables) resolve their
    // free names among the contract's parameters and the globals; local
    // annotations (let, the, set!) also see the enclosing scope.
    void assume(Ctx& c, const Refinement& r, const SVal& subject,
                const std::map<std::string, SVal>& extra, const eshkol_ast_t* at,
                const Scope* names = nullptr) {
        auto f = predicate(r, subject, extra, names ? names : &globals_, at);
        if (f) c.path.push_back(*f);
    }

    void require(Ctx& c, const Refinement& r, const SVal& subject,
                 const std::map<std::string, SVal>& extra, const std::string& what,
                 const eshkol_ast_t* at, bool boundary, const Scope* names = nullptr) {
        auto f = predicate(r, subject, extra, names ? names : &globals_, at);
        if (f) oblige(c, *f, what, r.text(), at, boundary);
    }

    // Facts a branch learned (call results, ascriptions) hold outside it
    // whenever the branch was taken.
    void mergeBranch(Ctx& c, const Ctx& b, Term cond) {
        if (b.path.size() <= c.path.size() + 1) return;
        std::vector<Term> added(b.path.begin() + static_cast<long>(c.path.size()) + 1, b.path.end());
        c.path.push_back(tm_.mkImplies(cond, tm_.mkAnd(added)));
    }

    // -- walk ------------------------------------------------------------------

    void topLevel(const eshkol_ast_t* a, Ctx& c) {
        if (isOp(a, ESHKOL_DEFINE_OP) && a->operation.define_op.name) {
            const auto& d = a->operation.define_op;
            std::string name = baseName(d.name);
            if (d.is_function || isOp(d.value, ESHKOL_LAMBDA_OP)) {
                defineFunction(a, c);
                auto cell = std::make_shared<Cell>();
                cell->name = name;
                cell->val = object(freshVar("fn", Sort::Obj));
                cell->contract = contractFor(name);
                globals_.vars[d.name] = cell;
                return;
            }
            SVal v = eval(d.value, c);
            auto cell = std::make_shared<Cell>();
            cell->name = name;
            cell->val = v;
            cell->mutated = mutated_globals_.count(name) > 0;
            globals_.vars[d.name] = cell;
            return;
        }
        eval(a, c);
    }

    // Walk a function body against its contract.
    void defineFunction(const eshkol_ast_t* a, const Ctx& outer) {
        const auto& d = a->operation.define_op;
        std::string name = baseName(d.name);
        const eshkol_ast_t* params;
        uint64_t nparams;
        hott_type_expr_t** ptypes;
        const eshkol_ast_t* body;
        if (d.is_function) {
            params = d.parameters;
            nparams = d.num_params;
            ptypes = d.param_types;
            body = d.value;
        } else {
            const auto& l = d.value->operation.lambda_op;
            params = l.parameters;
            nparams = l.num_params;
            ptypes = l.param_types;
            body = l.body;
        }
        walkFunction(name, params, nparams, ptypes, body, contractFor(name), outer, a);
    }

    const Contract* contractFor(const std::string& name) const {
        auto it = contracts_.find(name);
        return (it != contracts_.end() && !it->second.ambiguous) ? &it->second : nullptr;
    }

    void walkFunction(const std::string& name, const eshkol_ast_t* params, uint64_t nparams,
                      hott_type_expr_t** ptypes, const eshkol_ast_t* body,
                      const Contract* contract, const Ctx& outer, const eshkol_ast_t* at) {
        Scope s;
        s.parent = outer.scope;
        Ctx c;
        c.scope = &s;
        c.path = outer.path;
        c.where = name;
        c.typed = outer.typed || contract != nullptr;
        forEachNode(body, [&](const eshkol_ast_t* n) {
            if (isOp(n, ESHKOL_SET_OP) && n->operation.set_op.name)
                c.mutated.insert(baseName(n->operation.set_op.name));
        });
        c.mutated.insert(outer.mutated.begin(), outer.mutated.end());
        std::map<std::string, SVal> visible;
        for (uint64_t i = 0; i < nparams; ++i) {
            std::string pn = varName(&params[i]);
            if (pn.empty()) continue;
            const hott_type_expr_t* pt = contract && i < contract->params.size()
                                             ? contract->params[i].type
                                             : (ptypes ? ptypes[i] : nullptr);
            BaseKind k = baseKindOf(pt);
            auto cell = std::make_shared<Cell>();
            cell->name = pn;
            cell->kind = k;
            cell->val = freshOf(k, pn);
            cell->mutated = c.mutated.count(pn) > 0;
            if (contract && i < contract->params.size()) cell->ref = contract->params[i].ref;
            s.vars[params[i].variable.id] = cell;
            if (cell->ref) assume(c, *cell->ref, cell->val, visible, at);
            visible[pn] = cell->val;
        }
        SVal result = eval(body, c);
        if (contract && contract->result) {
            require(c, *contract->result, result, visible, "result of '" + name + "'", at, false);
        }
    }

    SVal readVar(const eshkol_ast_t* a, Ctx& c) {
        const auto* cellp = c.scope ? c.scope->lookup(a->variable.id) : nullptr;
        if (!cellp) cellp = c.scope ? c.scope->lookup(varName(a)) : nullptr;
        if (!cellp) {
            if (c.predicate) {
                c.malformed = true;
                c.malformed_what = "the unbound name '" + varName(a) + "'";
            }
            return unknown();
        }
        Cell& cell = **cellp;
        if (!cell.mutated && !c.mutated.count(cell.name)) {
            if (cell.val.kind == VK::Unknown && !cell.val.cell) {
                // Give an Unknown binding a stable identity.
                cell.val.cell = std::make_shared<Cell>();
                cell.val.cell->name = cell.name;
            }
            return cell.val;
        }
        // Assigned somewhere: every read is a new value, satisfying the
        // declared refinement if there is one.
        SVal v = freshOf(cell.kind, cell.name);
        if (cell.ref) assume(c, *cell.ref, v, {}, a, c.scope);
        return v;
    }

    SVal evalSeq(const eshkol_ast_t* xs, uint64_t n, Ctx& c) {
        SVal last = unknown();
        Scope s;
        s.parent = c.scope;
        const Scope* saved = c.scope;
        c.scope = &s;
        for (uint64_t i = 0; i < n; ++i) {
            const eshkol_ast_t* x = &xs[i];
            if (isOp(x, ESHKOL_DEFINE_OP) && x->operation.define_op.name) {
                const auto& d = x->operation.define_op;
                if (d.is_function || isOp(d.value, ESHKOL_LAMBDA_OP)) {
                    auto cell = std::make_shared<Cell>();
                    cell->name = baseName(d.name);
                    cell->val = object(freshVar("fn", Sort::Obj));
                    cell->contract = contractFor(cell->name);
                    s.vars[d.name] = cell;
                    defineFunction(x, c);
                } else {
                    SVal v = eval(d.value, c);
                    auto cell = std::make_shared<Cell>();
                    cell->name = baseName(d.name);
                    cell->val = v;
                    cell->mutated = c.mutated.count(cell->name) > 0;
                    s.vars[d.name] = cell;
                }
                last = unknown();
                continue;
            }
            last = eval(x, c);
        }
        c.scope = saved;
        return last;
    }

    SVal join(Term cond, const SVal& a, const SVal& b) {
        if (a.kind == VK::Num && b.kind == VK::Num) {
            return num(tm_.mkIte(cond, a.t, b.t), a.inexact || b.inexact);
        }
        if (a.kind == VK::Bool && b.kind == VK::Bool) return boolean(tm_.mkIte(cond, a.t, b.t));
        if (a.kind == VK::Obj && b.kind == VK::Obj) return object(tm_.mkIte(cond, a.t, b.t));
        return unknown();
    }

    SVal evalIf(const eshkol_ast_t* cond, const eshkol_ast_t* then_e, const eshkol_ast_t* else_e,
                Ctx& c) {
        SVal cv = eval(cond, c);
        Term f = truthy(cv);
        Ctx tc = c;
        tc.path.push_back(f);
        SVal a = then_e ? eval(then_e, tc) : unknown();
        Ctx ec = c;
        ec.path.push_back(tm_.mkNot(f));
        SVal b = else_e ? eval(else_e, ec) : unknown();
        mergeBranch(c, tc, f);
        mergeBranch(c, ec, tm_.mkNot(f));
        return join(f, a, b);
    }

    SVal evalCond(const eshkol_ast_t* clauses, uint64_t n, uint64_t i, Ctx& c) {
        if (i >= n) return unknown();
        const eshkol_ast_t* cl = &clauses[i];
        if (!isOp(cl, ESHKOL_CALL_OP)) return evalCond(clauses, n, i + 1, c);
        const auto& k = cl->operation.call_op;
        if (varName(k.func) == "else") return evalSeq(k.variables, k.num_vars, c);
        if (k.num_vars > 0 && varName(&k.variables[0]) == "=>") {
            eval(k.func, c);
            Ctx o = opaque(c);
            for (uint64_t j = 1; j < k.num_vars; ++j) eval(&k.variables[j], o);
            return unknown();
        }
        SVal test = eval(k.func, c);
        Term f = truthy(test);
        Ctx tc = c;
        tc.path.push_back(f);
        SVal a = k.num_vars ? evalSeq(k.variables, k.num_vars, tc) : test;
        Ctx ec = c;
        ec.path.push_back(tm_.mkNot(f));
        SVal b = evalCond(clauses, n, i + 1, ec);
        mergeBranch(c, tc, f);
        mergeBranch(c, ec, tm_.mkNot(f));
        return join(f, a, b);
    }

    Ctx opaque(const Ctx& c) {
        Ctx o;
        o.typed = c.typed;
        o.where = c.where;
        o.predicate = c.predicate;
        return o;
    }

    // A form the walk does not model: still walk it for obligations, but
    // without any outer binding or fact (its own binders are unknown to us).
    SVal evalOpaque(const eshkol_ast_t* a, Ctx& c) {
        if (c.predicate) {
            c.malformed = true;
            c.malformed_what = "a form other than an expression";
            return unknown();
        }
        static Scope sealed = [] {
            Scope s;
            s.opaque = true;
            return s;
        }();
        Ctx o = opaque(c);
        o.scope = &sealed;
        std::vector<const eshkol_ast_t*> cs;
        children(a, cs);
        for (const auto* x : cs) eval(x, o);
        return unknown();
    }

    SVal evalLet(const eshkol_ast_t* a, Ctx& c) {
        const auto& l = a->operation.let_op;
        auto s = std::make_unique<Scope>();
        s->parent = c.scope;
        const Scope* saved = c.scope;
        bool sequential = a->operation.op != ESHKOL_LET_OP;
        bool recursive = a->operation.op == ESHKOL_LETREC_OP || a->operation.op == ESHKOL_LETREC_STAR_OP;
        if (l.name) {
            // Named let: the loop variables are arbitrary iterates; the walk
            // checks the body once for every value they may take.
            return evalNamedLet(a, c);
        }
        if (recursive) {
            for (uint64_t i = 0; i < l.num_bindings; ++i) {
                const auto& b = l.bindings[i];
                if (b.type != ESHKOL_CONS || !b.cons_cell.car || b.cons_cell.car->type != ESHKOL_VAR) continue;
                auto cell = std::make_shared<Cell>();
                cell->name = varName(b.cons_cell.car);
                cell->mutated = true;  // assigned by the letrec itself
                s->vars[b.cons_cell.car->variable.id] = cell;
            }
            c.scope = s.get();
        }
        for (uint64_t i = 0; i < l.num_bindings; ++i) {
            const auto& b = l.bindings[i];
            if (b.type != ESHKOL_CONS || !b.cons_cell.car || b.cons_cell.car->type != ESHKOL_VAR) {
                eval(&b, c);
                continue;
            }
            std::string n = varName(b.cons_cell.car);
            const hott_type_expr_t* ty = l.binding_types ? l.binding_types[i] : nullptr;
            SVal v = eval(b.cons_cell.cdr, c);
            auto cell = std::make_shared<Cell>();
            cell->name = n;
            cell->kind = ty ? baseKindOf(ty) : BaseKind::Unknown;
            cell->mutated = c.mutated.count(n) > 0;
            if (ty) {
                std::set<std::string> visible;
                for (const auto& [k, x] : s->vars) visible.insert(baseName(k.c_str()));
                cell->ref = refinementFor(ty, n, visible, a);
                if (cell->ref) require(c, *cell->ref, v, {}, "binding '" + n + "'", b.cons_cell.cdr, !c.typed, c.scope);
                if (cell->ref) assume(c, *cell->ref, v, {}, a, c.scope);
            }
            cell->val = v;
            if (recursive) {
                s->vars[b.cons_cell.car->variable.id] = cell;
                cell->mutated = true;
            } else {
                s->vars[b.cons_cell.car->variable.id] = cell;
                if (sequential) c.scope = s.get();
            }
        }
        c.scope = s.get();
        SVal r = eval(l.body, c);
        c.scope = saved;
        return r;
    }

    SVal evalNamedLet(const eshkol_ast_t* a, Ctx& c) {
        const auto& l = a->operation.let_op;
        Scope s;
        s.parent = c.scope;
        std::vector<std::shared_ptr<Cell>> loop_vars;
        for (uint64_t i = 0; i < l.num_bindings; ++i) {
            const auto& b = l.bindings[i];
            if (b.type != ESHKOL_CONS || !b.cons_cell.car || b.cons_cell.car->type != ESHKOL_VAR) continue;
            std::string n = varName(b.cons_cell.car);
            const hott_type_expr_t* ty = l.binding_types ? l.binding_types[i] : nullptr;
            SVal init = eval(b.cons_cell.cdr, c);
            auto cell = std::make_shared<Cell>();
            cell->name = n;
            cell->kind = ty ? baseKindOf(ty) : BaseKind::Unknown;
            if (ty) {
                cell->ref = refinementFor(ty, n, {}, a);
                if (cell->ref) require(c, *cell->ref, init, {}, "initial value of loop variable '" + n + "'", b.cons_cell.cdr, !c.typed);
            }
            cell->val = freshOf(cell->kind, n);
            cell->mutated = c.mutated.count(n) > 0;
            s.vars[b.cons_cell.car->variable.id] = cell;
            loop_vars.push_back(cell);
        }
        auto loop = std::make_shared<Cell>();
        loop->name = baseName(l.name);
        loop->val = object(freshVar("loop", Sort::Obj));
        s.vars[l.name] = loop;
        Ctx body = c;
        body.scope = &s;
        for (const auto& cell : loop_vars) {
            if (cell->ref) assume(body, *cell->ref, cell->val, {}, a);
        }
        Contract lc;
        lc.name = baseName(l.name);
        for (const auto& cell : loop_vars) {
            Param p;
            p.name = cell->name;
            p.ref = cell->ref;
            lc.params.push_back(p);
        }
        loops_.push_back({l.name, &s, lc});
        eval(l.body, body);
        loops_.pop_back();
        return unknown();
    }

    struct LoopFrame {
        const char* name;
        const Scope* scope;
        Contract contract;
    };
    std::vector<LoopFrame> loops_;

    SVal evalLambda(const eshkol_ast_t* a, Ctx& c) {
        if (c.predicate) {
            c.malformed = true;
            c.malformed_what = "lambda";
            return unknown();
        }
        const auto& l = a->operation.lambda_op;
        Contract contract;
        bool any = false;
        std::set<std::string> earlier;
        for (uint64_t i = 0; i < l.num_params; ++i) {
            Param p;
            p.name = varName(&l.parameters[i]);
            p.type = l.param_types ? l.param_types[i] : nullptr;
            p.ref = refinementFor(p.type, p.name, earlier, a);
            any |= p.ref.has_value();
            earlier.insert(p.name);
            contract.params.push_back(p);
        }
        contract.result = refinementFor(l.return_type, "", earlier, a);
        any |= contract.result.has_value();
        walkFunction(c.where.empty() ? "lambda" : c.where, l.parameters, l.num_params, l.param_types,
                     l.body, any ? &contract : nullptr, c, a);
        return object(freshVar("closure", Sort::Obj));
    }

    // -- calls ------------------------------------------------------------------

    SVal evalCall(const eshkol_ast_t* a, Ctx& c) {
        const auto& k = a->operation.call_op;
        std::string fn = varName(k.func);
        if (fn == "make-rational" && k.num_vars == 2) {
            const auto *n = &k.variables[0], *d = &k.variables[1];
            if (n->type == ESHKOL_INT64 && d->type == ESHKOL_INT64 && d->int64_val != 0) {
                return num(tm_.mkReal(Rational(Integer(n->int64_val), Integer(d->int64_val))), false);
            }
        }
        std::vector<SVal> args;
        for (uint64_t i = 0; i < k.num_vars; ++i) args.push_back(eval(&k.variables[i], c));
        const std::shared_ptr<Cell>* bound =
            (k.func && k.func->type == ESHKOL_VAR && c.scope) ? c.scope->lookup(k.func->variable.id) : nullptr;
        // A call to the enclosing named-let loop.
        for (auto it = loops_.rbegin(); it != loops_.rend(); ++it) {
            if (k.func && k.func->type == ESHKOL_VAR && c.scope &&
                c.scope->lookup(k.func->variable.id) == it->scope->lookup(it->name)) {
                const Contract& lc = it->contract;
                for (size_t i = 0; i < lc.params.size() && i < args.size(); ++i) {
                    if (lc.params[i].ref) {
                        require(c, *lc.params[i].ref, args[i], {},
                                "loop variable '" + lc.params[i].name + "' of '" + lc.name + "'",
                                &k.variables[i], !c.typed);
                    }
                }
                return unknown();
            }
        }
        if (bound && (*bound)->contract && !c.predicate) {
            return applyContract(*(*bound)->contract, args, a, c);
        }
        if (!bound) {
            const Contract* ct = contractFor(fn);
            if (ct && !c.predicate) return applyContract(*ct, args, a, c);
            if (auto v = builtin(fn, args, a, c)) return *v;
        }
        if (c.predicate) {
            c.malformed = true;
            c.malformed_what = "'" + (fn.empty() ? std::string("a computed function") : fn) + "'";
            return unknown();
        }
        if (!k.func || k.func->type != ESHKOL_VAR) eval(k.func, c);
        return unknown();
    }

    SVal applyContract(const Contract& ct, const std::vector<SVal>& args, const eshkol_ast_t* a,
                       Ctx& c) {
        const auto& k = a->operation.call_op;
        if (!ct.variadic && args.size() != ct.params.size()) return unknown();
        std::map<std::string, SVal> visible;
        for (size_t i = 0; i < ct.params.size() && i < args.size(); ++i) {
            const Param& p = ct.params[i];
            if (p.ref) {
                require(c, *p.ref, args[i], visible,
                        "argument " + std::to_string(i + 1) + " (" + p.name + ") of '" + ct.name + "'",
                        &k.variables[i], !c.typed);
            }
            visible[p.name] = args[i];
        }
        BaseKind rk = BaseKind::Unknown;
        if (ct.result) rk = baseKindOf(ct.result->base);
        SVal r = freshOf(rk, ct.name);
        if (ct.result) assume(c, *ct.result, r, visible, a);
        return r;
    }

    void needIndex(Ctx& c, const SVal& index, Term len, const std::string& fn,
                   const char* lenfn, const eshkol_ast_t* call) {
        if (!c.typed) return;
        const auto& k = call->operation.call_op;
        Term i = asNum(index);
        Term goal = tm_.mkAnd(tm_.mkLe(tm_.mkInt(Integer(0)), i), tm_.mkLt(i, len));
        std::string idx = show(&k.variables[1]);
        oblige(c, goal, "index " + idx + " of " + fn,
               "(and (<= 0 " + idx + ") (< " + idx + " (" + lenfn + " " + show(&k.variables[0]) + ")))",
               &k.variables[1], false);
    }

    void needNonzero(Ctx& c, const SVal& d, const char* fn, const eshkol_ast_t* at) {
        if (!c.typed) return;
        Term t = asNum(d);
        oblige(c, tm_.mkNot(tm_.mkEq(t, tm_.mkInt(Integer(0)))), std::string("divisor of ") + fn,
               "(not (= divisor 0))", at, false);
    }

    std::optional<SVal> builtin(const std::string& fn, const std::vector<SVal>& args,
                                const eshkol_ast_t* a, Ctx& c) {
        const auto& k = a->operation.call_op;
        if (c.predicate && !predicateBuiltins().count(fn)) return std::nullopt;
        auto inexactAny = [&]() {
            for (const auto& x : args) {
                if (x.kind != VK::Num || x.inexact) return true;
            }
            return false;
        };
        auto argNum = [&](size_t i) { return asNum(args[i]); };
        Term zero = tm_.mkInt(Integer(0));

        if (fn == "+" || fn == "*") {
            Term acc = tm_.mkInt(Integer(fn == "+" ? 0 : 1));
            for (size_t i = 0; i < args.size(); ++i) {
                acc = fn == "+" ? tm_.mkAdd(acc, argNum(i)) : tm_.mkMul(acc, argNum(i));
            }
            return arith(acc, inexactAny());
        }
        if (fn == "-" && !args.empty()) {
            if (args.size() == 1) return arith(tm_.mkNeg(argNum(0)), inexactAny());
            Term acc = argNum(0);
            for (size_t i = 1; i < args.size(); ++i) acc = tm_.mkSub(acc, argNum(i));
            return arith(acc, inexactAny());
        }
        if (fn == "/" && !args.empty()) {
            bool inexact = inexactAny();
            Term acc = args.size() == 1 ? tm_.mkReal(Rational(1)) : argNum(0);
            for (size_t i = args.size() == 1 ? 0 : 1; i < args.size(); ++i) {
                Term d = argNum(i);
                if (!inexact && !c.predicate) needNonzero(c, args[i], "/", &k.variables[i]);
                if (d->kind == smt::Kind::Numeral && !d->value.isZero()) {
                    acc = tm_.mkScale(Rational(1) / d->value, acc);
                } else {
                    acc = tm_.mkApp("/", Sort::Real, {acc, d});
                }
            }
            if (inexact) return rounded(acc);
            return num(acc, false);
        }
        if ((fn == "quotient" || fn == "remainder" || fn == "modulo") && args.size() == 2) {
            bool inexact = inexactAny();
            if (!inexact && !c.predicate) needNonzero(c, args[1], fn.c_str(), &k.variables[1]);
            Term n = argNum(0), d = argNum(1);
            if (inexact) return num(freshVar("div", Sort::Real), true);
            if (d->kind == smt::Kind::Numeral && !d->value.isZero() && d->value.isInteger()) {
                // n = d*q + r with Scheme's sign conventions, by fresh q and r.
                Integer kd = d->value.num();
                Term q = freshVar("q", Sort::Int), r = freshVar("r", Sort::Int);
                Term kabs = tm_.mkInt(kd.abs());
                defs_.push_back(tm_.mkEq(n, tm_.mkAdd(tm_.mkScale(Rational(kd), q), r)));
                if (fn == "modulo") {
                    if (kd.sign() > 0) {
                        defs_.push_back(tm_.mkLe(zero, r));
                        defs_.push_back(tm_.mkLt(r, kabs));
                    } else {
                        defs_.push_back(tm_.mkLt(tm_.mkNeg(kabs), r));
                        defs_.push_back(tm_.mkLe(r, zero));
                    }
                } else {
                    Term nonneg = tm_.mkLe(zero, n);
                    defs_.push_back(tm_.mkImplies(nonneg, tm_.mkAnd(tm_.mkLe(zero, r), tm_.mkLt(r, kabs))));
                    defs_.push_back(tm_.mkImplies(tm_.mkNot(nonneg),
                                                  tm_.mkAnd(tm_.mkLt(tm_.mkNeg(kabs), r), tm_.mkLe(r, zero))));
                }
                return num(fn == "quotient" ? q : r, false);
            }
            Term ni = n->sort == Sort::Int ? n : freshVar("n", Sort::Int);
            Term di = d->sort == Sort::Int ? d : freshVar("d", Sort::Int);
            return num(tm_.mkApp(fn, Sort::Int, {ni, di}), false);
        }
        if ((fn == "<" || fn == "<=" || fn == ">" || fn == ">=" || fn == "=") && args.size() >= 2) {
            std::vector<Term> cs;
            for (size_t i = 0; i + 1 < args.size(); ++i) {
                Term x = argNum(i), y = argNum(i + 1);
                if (fn == "<") cs.push_back(tm_.mkLt(x, y));
                else if (fn == "<=") cs.push_back(tm_.mkLe(x, y));
                else if (fn == ">") cs.push_back(tm_.mkGt(x, y));
                else if (fn == ">=") cs.push_back(tm_.mkGe(x, y));
                else cs.push_back(tm_.mkEq(x, y));
            }
            return boolean(tm_.mkAnd(cs));
        }
        if (fn == "not" && args.size() == 1) return boolean(tm_.mkNot(truthy(args[0])));
        if (fn == "implies" && args.size() == 2)
            return boolean(tm_.mkImplies(truthy(args[0]), truthy(args[1])));
        if (fn == "zero?" && args.size() == 1) return boolean(tm_.mkEq(argNum(0), zero));
        if (fn == "positive?" && args.size() == 1) return boolean(tm_.mkLt(zero, argNum(0)));
        if (fn == "negative?" && args.size() == 1) return boolean(tm_.mkLt(argNum(0), zero));
        if ((fn == "even?" || fn == "odd?") && args.size() == 1) {
            Term n = argNum(0);
            Term q = freshVar("half", Sort::Int), r = freshVar("parity", Sort::Int);
            defs_.push_back(tm_.mkEq(n, tm_.mkAdd(tm_.mkScale(Rational(2), q), r)));
            defs_.push_back(tm_.mkLe(zero, r));
            defs_.push_back(tm_.mkLe(r, tm_.mkInt(Integer(1))));
            return boolean(tm_.mkEq(r, tm_.mkInt(Integer(fn == "even?" ? 0 : 1))));
        }
        if (fn == "abs" && args.size() == 1) {
            Term x = argNum(0);
            return num(tm_.mkIte(tm_.mkLe(zero, x), x, tm_.mkNeg(x)), args[0].inexact || args[0].kind != VK::Num);
        }
        if ((fn == "min" || fn == "max") && !args.empty()) {
            Term acc = argNum(0);
            for (size_t i = 1; i < args.size(); ++i) {
                Term x = argNum(i);
                acc = fn == "min" ? tm_.mkIte(tm_.mkLe(acc, x), acc, x) : tm_.mkIte(tm_.mkLe(x, acc), acc, x);
            }
            return num(acc, inexactAny());
        }
        if ((fn == "floor" || fn == "ceiling" || fn == "round" || fn == "truncate") && args.size() == 1) {
            Term x = argNum(0);
            Term r = freshVar(fn, Sort::Int);
            Term one = tm_.mkInt(Integer(1));
            if (fn == "floor") {
                defs_.push_back(tm_.mkLe(r, x));
                defs_.push_back(tm_.mkLt(x, tm_.mkAdd(r, one)));
            } else if (fn == "ceiling") {
                defs_.push_back(tm_.mkLt(tm_.mkSub(r, one), x));
                defs_.push_back(tm_.mkLe(x, r));
            } else if (fn == "round") {
                Term half = tm_.mkReal(Rational(Integer(1), Integer(2)));
                defs_.push_back(tm_.mkLe(tm_.mkSub(x, r), half));
                defs_.push_back(tm_.mkLe(tm_.mkSub(r, x), half));
            } else {
                Term nonneg = tm_.mkLe(zero, x);
                defs_.push_back(tm_.mkImplies(nonneg, tm_.mkAnd(tm_.mkLe(r, x), tm_.mkLt(x, tm_.mkAdd(r, one)))));
                defs_.push_back(tm_.mkImplies(tm_.mkNot(nonneg),
                                              tm_.mkAnd(tm_.mkLt(tm_.mkSub(r, one), x), tm_.mkLe(x, r))));
            }
            return num(r, args[0].inexact || args[0].kind != VK::Num);
        }
        if ((fn == "exact->inexact" || fn == "inexact") && args.size() == 1) {
            if (args[0].kind == VK::Num && !args[0].inexact) return rounded(args[0].t);
            return num(argNum(0), true);
        }
        if ((fn == "inexact->exact" || fn == "exact") && args.size() == 1) {
            return num(argNum(0), false);  // the exact value of the double
        }
        if ((fn == "eq?" || fn == "eqv?" || fn == "equal?" || fn == "boolean=?") && args.size() == 2) {
            const SVal &x = args[0], &y = args[1];
            if (x.kind == VK::Num && y.kind == VK::Num) return boolean(tm_.mkEq(x.t, y.t));
            if (x.kind == VK::Bool && y.kind == VK::Bool) return boolean(tm_.mkIff(x.t, y.t));
            if (x.kind == VK::Obj && y.kind == VK::Obj && fn != "equal?") return boolean(tm_.mkEq(x.t, y.t));
            return SVal{};
        }
        // Lengths
        static const std::map<std::string, const char*> lengths{
            {"vector-length", "vector-length"}, {"string-length", "string-length"},
            {"length", "length"}, {"tensor-length", "tensor-length"}};
        auto lit = lengths.find(fn);
        if (lit != lengths.end() && args.size() == 1) return num(length(lit->second, args[0]), false);
        if (fn == "null?" && args.size() == 1) return boolean(tm_.mkEq(length("length", args[0]), zero));
        if (fn == "pair?" && args.size() == 1) return boolean(tm_.mkLt(zero, length("length", args[0])));
        if (c.predicate) return std::nullopt;

        // Constructors with known lengths
        auto withLength = [&](const char* lfn, Term len) {
            SVal o = object(freshVar("obj", Sort::Obj));
            defs_.push_back(tm_.mkEq(length(lfn, o), len));
            return o;
        };
        if ((fn == "make-vector" || fn == "make-string") && !args.empty()) {
            const char* lfn = fn == "make-vector" ? "vector-length" : "string-length";
            if (c.typed) {
                oblige(c, tm_.mkLe(zero, argNum(0)), "size of " + fn, "(<= 0 size)", &k.variables[0], false);
            }
            Term n = argNum(0);
            if (n->sort != Sort::Int) return object(freshVar("obj", Sort::Obj));
            return withLength(lfn, n);
        }
        if (fn == "vector") return withLength("vector-length", tm_.mkInt(Integer(static_cast<int64_t>(args.size()))));
        if (fn == "list") return withLength("length", tm_.mkInt(Integer(static_cast<int64_t>(args.size()))));
        if (fn == "cons" && args.size() == 2) {
            return withLength("length", tm_.mkAdd(length("length", args[1]), tm_.mkInt(Integer(1))));
        }
        if (fn == "cdr" && args.size() == 1) {
            return withLength("length", tm_.mkSub(length("length", args[0]), tm_.mkInt(Integer(1))));
        }
        if ((fn == "vector-copy" || fn == "reverse" || fn == "list-copy") && args.size() == 1) {
            const char* lfn = fn == "vector-copy" ? "vector-length" : "length";
            return withLength(lfn, length(lfn, args[0]));
        }
        if ((fn == "vector-append" || fn == "string-append" || fn == "append") && !args.empty()) {
            const char* lfn = fn == "vector-append" ? "vector-length"
                              : fn == "string-append" ? "string-length" : "length";
            std::vector<Term> parts;
            for (const auto& x : args) parts.push_back(length(lfn, x));
            return withLength(lfn, tm_.mkAdd(parts));
        }
        if ((fn == "map" || fn == "vector-map") && args.size() == 2) {
            const char* lfn = fn == "map" ? "length" : "vector-length";
            return withLength(lfn, length(lfn, args[1]));
        }
        // Indexing
        if ((fn == "vector-ref" || fn == "vector-set!") && args.size() >= 2) {
            needIndex(c, args[1], length("vector-length", args[0]), fn, "vector-length", a);
            return unknown();
        }
        if ((fn == "string-ref" || fn == "string-set!") && args.size() >= 2) {
            needIndex(c, args[1], length("string-length", args[0]), fn, "string-length", a);
            return unknown();
        }
        if ((fn == "list-ref" || fn == "list-tail") && args.size() == 2) {
            Term len = length("length", args[0]);
            if (fn == "list-tail") len = tm_.mkAdd(len, tm_.mkInt(Integer(1)));
            needIndex(c, args[1], len, fn, "length", a);
            return unknown();
        }
        if ((fn == "vref" || fn == "tensor-ref") && args.size() == 2) {
            needIndex(c, args[1], length("tensor-length", args[0]), fn, "tensor-length", a);
            return unknown();
        }
        if (fn == "substring" && args.size() == 3) {
            Term len = length("string-length", args[0]);
            Term s = argNum(1), e = argNum(2);
            if (c.typed) {
                oblige(c, tm_.mkAnd({tm_.mkLe(zero, s), tm_.mkLe(s, e), tm_.mkLe(e, len)}),
                       "bounds of substring", "(<= 0 start end (string-length s))", a, false);
            }
            return withLength("string-length", tm_.mkSub(e, s));
        }
        return std::nullopt;
    }

    // -- expressions -----------------------------------------------------------

    SVal eval(const eshkol_ast_t* a, Ctx& c) {
        if (!a) return unknown();
        switch (a->type) {
            case ESHKOL_INT8: case ESHKOL_INT16: case ESHKOL_INT32: case ESHKOL_INT64:
                return num(tm_.mkInt(Integer(a->int64_val)), false);
            case ESHKOL_UINT8: case ESHKOL_UINT16: case ESHKOL_UINT32:
                return num(tm_.mkInt(Integer(static_cast<int64_t>(a->uint64_val))), false);
            case ESHKOL_BIGNUM_LITERAL: {
                auto v = a->str_val.ptr ? Integer::parse(std::string(a->str_val.ptr, a->str_val.size))
                                        : std::nullopt;
                return v ? num(tm_.mkInt(*v), false) : unknown();
            }
            case ESHKOL_DOUBLE:
                return num(tm_.mkReal(Rational::fromDouble(a->double_val)), true);
            case ESHKOL_BOOL:
                return boolean(tm_.mkBool(a->int64_val != 0));
            case ESHKOL_STRING: {
                SVal o = object(freshVar("str", Sort::Obj));
                bool ascii = true;
                for (uint64_t i = 0; i < a->str_val.size && a->str_val.ptr; ++i)
                    ascii &= static_cast<unsigned char>(a->str_val.ptr[i]) < 0x80;
                if (ascii) {
                    defs_.push_back(tm_.mkEq(length("string-length", o),
                                             tm_.mkInt(Integer(static_cast<int64_t>(a->str_val.size)))));
                }
                return o;
            }
            case ESHKOL_NULL: {
                SVal o = object(freshVar("nil", Sort::Obj));
                defs_.push_back(tm_.mkEq(length("length", o), tm_.mkInt(Integer(0))));
                return o;
            }
            case ESHKOL_VAR:
                return readVar(a, c);
            case ESHKOL_OP:
                break;
            default:
                return unknown();
        }
        const auto& op = a->operation;
        switch (op.op) {
            case ESHKOL_CALL_OP:
                return evalCall(a, c);
            case ESHKOL_IF_OP:
                return evalIf(op.call_op.num_vars > 0 ? &op.call_op.variables[0] : nullptr,
                              op.call_op.num_vars > 1 ? &op.call_op.variables[1] : nullptr,
                              op.call_op.num_vars > 2 ? &op.call_op.variables[2] : nullptr, c);
            case ESHKOL_COND_OP:
                return evalCond(op.call_op.variables, op.call_op.num_vars, 0, c);
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP: {
                if (op.call_op.num_vars == 0) return unknown();
                SVal t = eval(&op.call_op.variables[0], c);
                Term f = truthy(t);
                Ctx bc = c;
                Term taken = op.op == ESHKOL_WHEN_OP ? f : tm_.mkNot(f);
                bc.path.push_back(taken);
                evalSeq(op.call_op.variables + 1, op.call_op.num_vars - 1, bc);
                mergeBranch(c, bc, taken);
                return unknown();
            }
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP: {
                bool is_and = op.op == ESHKOL_AND_OP;
                std::vector<Term> fs;
                Ctx inner = c;
                bool all_bool = true;
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i) {
                    SVal v = eval(&op.sequence_op.expressions[i], inner);
                    Term f = truthy(v);
                    all_bool &= v.kind == VK::Bool;
                    fs.push_back(f);
                    inner.path.push_back(is_and ? f : tm_.mkNot(f));
                }
                if (!all_bool && !c.predicate) {
                    // `and`/`or` return a value, not a Boolean: only its truth is known.
                    SVal v;
                    v.cell = std::make_shared<Cell>();
                    v.cell->name = is_and ? "and" : "or";
                    Term truth = lazyTerm(*v.cell, Sort::Bool);
                    defs_.push_back(tm_.mkIff(truth, is_and ? tm_.mkAnd(fs) : tm_.mkOr(fs)));
                    return v;
                }
                return boolean(is_and ? tm_.mkAnd(fs) : tm_.mkOr(fs));
            }
            case ESHKOL_SEQUENCE_OP:
                return evalSeq(op.sequence_op.expressions, op.sequence_op.num_expressions, c);
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP:
                if (c.predicate) return evalOpaque(a, c);
                return evalLet(a, c);
            case ESHKOL_LAMBDA_OP:
                return evalLambda(a, c);
            case ESHKOL_DEFINE_OP:
                if (c.predicate) return evalOpaque(a, c);
                if (op.define_op.is_function || isOp(op.define_op.value, ESHKOL_LAMBDA_OP)) {
                    defineFunction(a, c);
                } else {
                    eval(op.define_op.value, c);
                }
                return unknown();
            case ESHKOL_SET_OP: {
                if (c.predicate) return evalOpaque(a, c);
                SVal v = eval(op.set_op.value, c);
                const auto* cellp = c.scope ? c.scope->lookup(op.set_op.name) : nullptr;
                if (cellp && (*cellp)->ref) {
                    require(c, *(*cellp)->ref, v, {}, "value assigned to '" + (*cellp)->name + "'",
                            op.set_op.value, !c.typed, c.scope);
                }
                return unknown();
            }
            case ESHKOL_THE_OP: {
                SVal v = eval(op.the_op.expr, c);
                if (c.predicate) return v;
                std::set<std::string> visible;
                for (const Scope* s = c.scope; s; s = s->parent)
                    for (const auto& [n, x] : s->vars) visible.insert(baseName(n.c_str()));
                auto r = refinementFor(op.the_op.type_expr, "", visible, a);
                if (!r) return v;
                // `the` is the trusted ascription: a contradiction is an error,
                // an unproved refinement is a type issue, and either way the
                // refinement holds afterwards.
                auto f = predicate(*r, v, {}, c.scope, a);
                if (f) {
                    oblige(c, *f, "value of (the ...)", r->text(), a, true);
                    c.path.push_back(*f);
                }
                return v;
            }
            case ESHKOL_TENSOR_OP: {
                if (c.predicate) return evalOpaque(a, c);
                for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i) eval(&op.tensor_op.elements[i], c);
                SVal o = object(freshVar("vec", Sort::Obj));
                if (op.tensor_op.num_dimensions == 1) {
                    Term n = tm_.mkInt(Integer(static_cast<int64_t>(op.tensor_op.total_elements)));
                    defs_.push_back(tm_.mkEq(length("vector-length", o), n));
                    defs_.push_back(tm_.mkEq(length("tensor-length", o), n));
                }
                return o;
            }
            case ESHKOL_QUOTE_OP:
            case ESHKOL_DEFINE_TYPE_OP:
            case ESHKOL_TYPE_ANNOTATION_OP:
            case ESHKOL_DEFINE_SYNTAX_OP:
                return unknown();
            default:
                return evalOpaque(a, c);
        }
    }
};

}  // namespace

RefinementOptions RefinementOptions::fromEnvironment(bool strict_types, bool unsafe_mode) {
    RefinementOptions o;
    o.strict_types = strict_types;
    o.unsafe_mode = unsafe_mode;
    if (const char* s = std::getenv("ESHKOL_REFINEMENT_SOLVER")) {
        std::string v = s;
        if (v == "internal" || v == "z3" || v == "cross") o.solver = v;
    }
    if (const char* b = std::getenv("ESHKOL_REFINEMENT_BUDGET")) {
        char* end = nullptr;
        unsigned long long n = std::strtoull(b, &end, 10);
        if (end && *end == '\0' && n > 0) o.budget = n;
    }
    return o;
}

bool programHasRefinements(const eshkol_ast_t* asts, size_t num_asts) {
    bool found = false;
    for (size_t i = 0; i < num_asts && !found; ++i) {
        forEachNode(&asts[i], [&](const eshkol_ast_t* a) {
            if (found || a->type != ESHKOL_OP) return;
            const auto& op = a->operation;
            auto check = [&](const hott_type_expr_t* t) { found |= typeHasRefinement(t); };
            switch (op.op) {
                case ESHKOL_DEFINE_OP:
                    check(op.define_op.return_type);
                    for (uint64_t k = 0; op.define_op.param_types && k < op.define_op.num_params; ++k)
                        check(op.define_op.param_types[k]);
                    break;
                case ESHKOL_LAMBDA_OP:
                    check(op.lambda_op.return_type);
                    for (uint64_t k = 0; op.lambda_op.param_types && k < op.lambda_op.num_params; ++k)
                        check(op.lambda_op.param_types[k]);
                    break;
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    for (uint64_t k = 0; op.let_op.binding_types && k < op.let_op.num_bindings; ++k)
                        check(op.let_op.binding_types[k]);
                    break;
                case ESHKOL_THE_OP:
                    check(op.the_op.type_expr);
                    break;
                case ESHKOL_DEFINE_TYPE_OP:
                    check(op.define_type_op.type_expr);
                    break;
                case ESHKOL_TYPE_ANNOTATION_OP:
                    check(op.type_annotation_op.type_expr);
                    break;
                default:
                    break;
            }
        });
    }
    return found;
}

RefinementReport checkProgramRefinements(TypeChecker& checker, const eshkol_ast_t* asts,
                                         size_t num_asts, const RefinementOptions& options) {
    if (!programHasRefinements(asts, num_asts)) return RefinementReport{};
    Checker c(checker, options);
    return c.run(asts, num_asts);
}

}  // namespace eshkol::hott
