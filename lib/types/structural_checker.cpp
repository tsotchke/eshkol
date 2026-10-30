/**
 * @file structural_checker.cpp
 * @brief Structural layer of the type checker: rows, arbitrary rank, sessions
 *        (ADR-0046).
 *
 * Organisation:
 *   1. Type syntax: one neutral S-expression view of a type, read either from
 *      a parsed hott_type_expr_t or from a library signature string, and one
 *      conversion from that view to structural terms. User annotations and
 *      library signatures therefore mean the same thing by construction.
 *   2. Library signatures: the structural types of the operations of
 *      core.records and core.sessions (and the numeric core), active only
 *      when the program requires the library.
 *   3. The bidirectional checker (Peyton Jones et al. 2007): inferRho /
 *      checkRho / checkSigma / subsCheck, plus the label-literal rules of the
 *      record and session operations.
 *   4. The per-form finish: gradual boundary obligations, closure modes, and
 *      exactly-once enforcement of every linear binder through the
 *      TypeChecker's own linear judgment in strict mode.
 */
#include "eshkol/types/structural_checker.h"

#include "eshkol/types/hott_types.h"
#include "eshkol/types/type_checker.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <iterator>

namespace eshkol::hott {

using st::Binder;
using st::Kind;
using st::Mode;
using st::Tag;
using st::TyRef;

namespace {

// ============================================================================
// 1. Type syntax
// ============================================================================

/** Neutral S-expression view of a type. */
struct TypeSyntax {
    bool atom = true;
    bool invalid = false;          ///< a value argument, not a type
    std::string text;              ///< atom spelling
    std::vector<TypeSyntax> elems; ///< list elements (head first)

    static TypeSyntax mkAtom(const std::string& t) {
        TypeSyntax s;
        s.text = t;
        return s;
    }
    static TypeSyntax mkList(std::vector<TypeSyntax> e) {
        TypeSyntax s;
        s.atom = false;
        s.elems = std::move(e);
        return s;
    }
    static TypeSyntax mkInvalid() {
        TypeSyntax s;
        s.invalid = true;
        return s;
    }
    std::string head() const {
        return (!atom && !elems.empty() && elems[0].atom && !elems[0].invalid) ? elems[0].text : "";
    }
};

TypeSyntax syntaxOf(const hott_type_expr_t* t) {
    if (!t) return TypeSyntax::mkAtom("any");
    switch (t->kind) {
        case HOTT_TYPE_INTEGER:
        case HOTT_TYPE_REAL:
        case HOTT_TYPE_NUMBER:
            return TypeSyntax::mkAtom("number");
        case HOTT_TYPE_BOOLEAN: return TypeSyntax::mkAtom("boolean");
        case HOTT_TYPE_STRING:  return TypeSyntax::mkAtom("string");
        case HOTT_TYPE_CHAR:    return TypeSyntax::mkAtom("char");
        case HOTT_TYPE_SYMBOL:  return TypeSyntax::mkAtom("symbol");
        case HOTT_TYPE_NULL:    return TypeSyntax::mkAtom("null");
        case HOTT_TYPE_ANY:
        case HOTT_TYPE_NOTHING:
            return TypeSyntax::mkAtom("any");
        case HOTT_TYPE_VAR:
            return TypeSyntax::mkAtom(t->var_name ? t->var_name : "any");
        case HOTT_TYPE_ARROW: {
            std::vector<TypeSyntax> e{TypeSyntax::mkAtom("->")};
            for (uint64_t i = 0; i < t->arrow.num_params; ++i) {
                e.push_back(syntaxOf(t->arrow.param_types ? t->arrow.param_types[i] : nullptr));
            }
            e.push_back(syntaxOf(t->arrow.return_type));
            return TypeSyntax::mkList(std::move(e));
        }
        case HOTT_TYPE_FORALL: {
            std::vector<TypeSyntax> vars;
            for (uint64_t i = 0; i < t->forall.num_vars; ++i) {
                vars.push_back(TypeSyntax::mkAtom(t->forall.type_vars[i]));
            }
            return TypeSyntax::mkList({TypeSyntax::mkAtom("forall"), TypeSyntax::mkList(vars),
                                       syntaxOf(t->forall.body)});
        }
        case HOTT_TYPE_LIST:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("list"), syntaxOf(t->container.element_type)});
        case HOTT_TYPE_VECTOR:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("vector"), syntaxOf(t->container.element_type)});
        case HOTT_TYPE_TENSOR:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("tensor"), syntaxOf(t->container.element_type)});
        case HOTT_TYPE_POINTER:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("ptr"), syntaxOf(t->container.element_type)});
        case HOTT_TYPE_PAIR:
        case HOTT_TYPE_PRODUCT:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("pair"), syntaxOf(t->pair.left),
                                       syntaxOf(t->pair.right)});
        case HOTT_TYPE_SUM:
            return TypeSyntax::mkList({TypeSyntax::mkAtom("+"), syntaxOf(t->sum.left),
                                       syntaxOf(t->sum.right)});
        case HOTT_TYPE_APP: {
            std::vector<TypeSyntax> e{TypeSyntax::mkAtom(t->app.name ? t->app.name : "?")};
            for (uint64_t i = 0; i < t->app.num_args; ++i) {
                e.push_back(syntaxOf(t->app.args ? t->app.args[i] : nullptr));
            }
            return TypeSyntax::mkList(std::move(e));
        }
        default:
            return TypeSyntax::mkInvalid();
    }
}

/** Reader for library signature strings (the same S-expression syntax). */
TypeSyntax readSignature(const char*& p) {
    while (*p == ' ' || *p == '\n' || *p == '\t') ++p;
    if (*p == '(') {
        ++p;
        std::vector<TypeSyntax> elems;
        while (true) {
            while (*p == ' ' || *p == '\n' || *p == '\t') ++p;
            if (*p == ')' || *p == '\0') {
                if (*p) ++p;
                break;
            }
            elems.push_back(readSignature(p));
        }
        return TypeSyntax::mkList(std::move(elems));
    }
    std::string tok;
    while (*p && *p != ' ' && *p != '(' && *p != ')' && *p != '\n' && *p != '\t') tok += *p++;
    return TypeSyntax::mkAtom(tok);
}

/** Substitute atoms (alias parameters) in a syntax tree. */
TypeSyntax substSyntax(const TypeSyntax& t, const std::map<std::string, TypeSyntax>& sub) {
    if (t.atom) {
        if (!t.invalid) {
            auto it = sub.find(t.text);
            if (it != sub.end()) return it->second;
        }
        return t;
    }
    TypeSyntax out = t;
    for (auto& e : out.elems) e = substSyntax(e, sub);
    return out;
}

bool isNumericName(const std::string& n) {
    static const char* const kNames[] = {
        "number", "integer", "real", "int", "float", "double", "exact", "inexact",
        "Number", "Integer", "Real", "Int64", "Int32", "Int16", "Int8", "UInt64",
        "UInt32", "UInt16", "UInt8", "Float64", "Float32", "Natural", "Nat",
        "Rational", "Complex", "BigInt", "rational", "complex"};
    for (const char* k : kNames) {
        if (n == k) return true;
    }
    return false;
}

/** Spellings that denote the gradual unknown at this layer. */
bool isDynName(const std::string& n) {
    static const char* const kNames[] = {
        "any", "Any", "Value", "value", "nothing", "Nothing", "Function", "function",
        "procedure", "Procedure", "List", "Vector", "Pair", "Tensor", "Pointer", "Ptr"};
    for (const char* k : kNames) {
        if (n == k) return true;
    }
    return false;
}

const char* varName(const eshkol_ast_t* e) {
    return (e && e->type == ESHKOL_VAR && e->variable.id) ? e->variable.id : nullptr;
}

struct StructuralError {
    std::string msg;
    const eshkol_ast_t* node;
};

// ============================================================================
// 2. Library signatures
// ============================================================================

struct Signature {
    const char* name;
    const char* type;
};

/** The numeric and comparison core, used whenever a structural form calls it. */
const Signature kCoreSignatures[] = {
    {"+", "(-> (... number) number)"},
    {"*", "(-> (... number) number)"},
    {"-", "(-> number (... number) number)"},
    {"/", "(-> number (... number) number)"},
    {"=", "(-> number (... number) boolean)"},
    {"<", "(-> number (... number) boolean)"},
    {">", "(-> number (... number) boolean)"},
    {"<=", "(-> number (... number) boolean)"},
    {">=", "(-> number (... number) boolean)"},
    {"zero?", "(-> number boolean)"},
    {"abs", "(-> number number)"},
    {"not", "(forall (a) (-> a boolean))"},
    {"eq?", "(forall (a b) (-> a b boolean))"},
    {"eqv?", "(forall (a b) (-> a b boolean))"},
    {"equal?", "(forall (a b) (-> a b boolean))"},
    {"string-append", "(-> (... string) string)"},
    {"string-length", "(-> string number)"},
    {"number->string", "(-> number string)"},
    {"symbol->string", "(-> symbol string)"},
};

/** core.records: label-literal operations are rules (see recordRule()). */
const Signature kRecordSignatures[] = {
    {"record?", "(forall (a) (-> a boolean))"},
    {"record-has?", "(forall (r) (-> (record r) symbol boolean))"},
    {"record-labels", "(forall (r) (-> (record r) (list symbol)))"},
    {"record->alist", "(forall (r) (-> (record r) any))"},
};
const char* const kRecordRules[] = {"record", "record-ref", "record-extend",
                                    "record-remove", "record-update"};

/** core.sessions: label-literal operations are rules (see sessionRule()). */
const Signature kSessionSignatures[] = {
    {"session-send", "(forall (t p) (-> (chan (! t p)) t (chan p)))"},
    {"session-recv", "(forall (t p) (-> (chan (? t p)) (values t (chan p))))"},
    {"session-close", "(-> (chan end) null)"},
    {"make-session-pair", "(forall (p) (-> (values (chan p) (chan (dual p)))))"},
    {"session-fork", "(forall (p) (-> (-o (chan p) any) (chan (dual p))))"},
    {"tcp-session", "(forall (p) (-> any (chan p)))"},
    {"session-endpoint?", "(forall (a) (-> a boolean))"},
};
const char* const kSessionRules[] = {"session-select", "session-offer"};

bool requireNames(const eshkol_ast_t* form, const char* module) {
    const auto& r = form->operation.require_op;
    for (uint64_t i = 0; i < r.num_modules; ++i) {
        if (r.module_names && r.module_names[i] && std::strcmp(r.module_names[i], module) == 0) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Generic AST traversal (mentions and assignments only; over-approximate)
// ============================================================================

void forEachChild(const eshkol_ast_t* e, const std::function<void(const eshkol_ast_t*)>& f) {
    if (!e) return;
    if (e->type == ESHKOL_CONS) {
        f(e->cons_cell.car);
        f(e->cons_cell.cdr);
        return;
    }
    if (e->type != ESHKOL_OP) return;
    const auto& op = e->operation;
    switch (op.op) {
        case ESHKOL_QUOTE_OP:
            return;
        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i) {
                f(&op.sequence_op.expressions[i]);
            }
            return;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) f(&op.let_op.bindings[i]);
            f(op.let_op.body);
            return;
        case ESHKOL_LAMBDA_OP:
            f(op.lambda_op.body);
            return;
        case ESHKOL_DEFINE_OP:
            f(op.define_op.value);
            return;
        case ESHKOL_SET_OP:
            f(op.set_op.value);
            return;
        case ESHKOL_THE_OP:
            f(op.the_op.expr);
            return;
        case ESHKOL_VALUES_OP:
            for (uint64_t i = 0; i < op.values_op.num_values; ++i) f(&op.values_op.expressions[i]);
            return;
        case ESHKOL_LET_VALUES_OP:
        case ESHKOL_LET_STAR_VALUES_OP:
            for (uint64_t i = 0; i < op.let_values_op.num_bindings; ++i) {
                f(&op.let_values_op.producers[i]);
            }
            f(op.let_values_op.body);
            return;
        case ESHKOL_CALL_WITH_VALUES_OP:
            f(op.call_with_values_op.producer);
            f(op.call_with_values_op.consumer);
            return;
        case ESHKOL_GUARD_OP:
            for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i) f(&op.guard_op.clauses[i]);
            for (uint64_t i = 0; i < op.guard_op.num_body_exprs; ++i) f(&op.guard_op.body[i]);
            return;
        case ESHKOL_RAISE_OP:
            f(op.raise_op.exception);
            return;
        case ESHKOL_MATCH_OP:
            f(op.match_op.expr);
            for (uint64_t i = 0; i < op.match_op.num_clauses; ++i) {
                f(op.match_op.clauses[i].guard);
                f(op.match_op.clauses[i].body);
            }
            return;
        case ESHKOL_CALL_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_ADD_OP:
        case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP:
        case ESHKOL_DIV_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_DO_OP:
            f(op.call_op.func);
            for (uint64_t i = 0; i < op.call_op.num_vars; ++i) f(&op.call_op.variables[i]);
            return;
        default:
            return;
    }
}

void collectMentions(const eshkol_ast_t* e, std::set<std::string>& out) {
    if (!e) return;
    if (const char* n = varName(e)) {
        out.insert(n);
        return;
    }
    forEachChild(e, [&](const eshkol_ast_t* c) { collectMentions(c, out); });
}

bool assignsVar(const eshkol_ast_t* e, const std::string& name) {
    if (!e) return false;
    if (e->type == ESHKOL_OP && e->operation.op == ESHKOL_SET_OP && e->operation.set_op.name &&
        name == e->operation.set_op.name) {
        return true;
    }
    bool found = false;
    forEachChild(e, [&](const eshkol_ast_t* c) { found = found || assignsVar(c, name); });
    return found;
}

}  // namespace

// ============================================================================
// Implementation state
// ============================================================================

struct StructuralChecker::Impl {
    TypeChecker& tc;
    TypeEnvironment& env;
    st::TypeStore s;
    st::Solver solver{s};
    int level = 0;

    // ---- program-wide ----
    std::map<std::string, TyRef> globals;        // schemes of checked names and active signatures
    std::set<std::string> rules;                 // active label-literal operations
    std::set<std::string> structural;            // names whose mention makes a form structural
    std::set<std::string> user_defined;          // top-level names the program defines
    std::set<std::string> pending;               // structural defines not yet checked
    std::map<std::string, TyRef> declared;       // signatures from annotations
    std::map<std::string, TyRef> placeholders;   // forward references (monomorphic)
    struct Alias {
        std::vector<std::string> params;
        TypeSyntax body;
    };
    std::map<std::string, Alias> aliases;
    std::map<std::string, const hott_type_expr_t*> annotations;   // (: name T)
    std::set<std::string> modules;
    bool program_known = false;
    bool dump = false;

    // ---- per form ----
    struct Local {
        std::string name;
        TyRef type;
        int fn_depth;
        int binder;  // index into binders, or -1
        bool annotated = false;
    };
    std::vector<Local> locals;
    std::vector<const eshkol_ast_t*> fn_stack;
    struct BinderRec {
        std::string name;
        TyRef type;
        std::vector<const eshkol_ast_t*> region;
        const eshkol_ast_t* site;
        const void* loop_op = nullptr;
        size_t loop_index = 0;
    };
    std::vector<BinderRec> binders;
    struct Capture {
        const eshkol_ast_t* lambda;
        int binder;
    };
    std::vector<Capture> captures;
    std::map<const eshkol_ast_t*, TyRef> lambda_types;
    std::vector<std::pair<TyRef, const eshkol_ast_t*>> obligations;
    std::vector<TyRef> dyn_buffer;
    std::vector<std::pair<TyRef, const eshkol_ast_t*>> record_fields;
    std::vector<std::pair<TyRef, const eshkol_ast_t*>> discarded;   // values nobody receives
    std::vector<std::map<std::string, TyRef>> tyvar_scopes;
    const eshkol_ast_t* current = nullptr;

    Impl(TypeChecker& t, TypeEnvironment& e) : tc(t), env(e) {
        solver.dyn_meets = &dyn_buffer;
        const char* d = std::getenv("ESHKOL_DUMP_STRUCTURAL_TYPES");
        dump = d && *d && std::strcmp(d, "0") != 0;
        for (const auto& sig : kCoreSignatures) globals[sig.name] = signature(sig.type);
    }

    // ------------------------------------------------------------------
    // Errors and solver calls
    // ------------------------------------------------------------------
    [[noreturn]] void fail(const eshkol_ast_t* node, const std::string& msg) {
        throw StructuralError{msg, node ? node : current};
    }

    void drain(const eshkol_ast_t* node) {
        for (TyRef t : dyn_buffer) obligations.push_back({t, node});
        dyn_buffer.clear();
    }

    void unify(TyRef a, TyRef b, const eshkol_ast_t* node, const std::string& what) {
        solver.clearError();
        bool ok = solver.unify(a, b);
        drain(node);
        if (!ok) fail(node, what + ": " + solver.error());
    }

    // ------------------------------------------------------------------
    // Type syntax -> terms
    // ------------------------------------------------------------------
    enum class FreeVars { Quantify, Fresh };

    struct Slot {
        Binder binder;
        bool kinded = false;
        TyRef node = nullptr;
    };

    struct Conv {
        FreeVars mode = FreeVars::Fresh;
        std::vector<std::map<std::string, Slot*>> scopes;   // forall binders
        std::deque<Slot> slots;
        std::vector<Slot*> implicit;                        // Quantify: free variables
        std::vector<std::pair<std::string, std::pair<int, std::string>>> rec;   // rec vars
        const eshkol_ast_t* node = nullptr;
    };

    TyRef signature(const char* text) {
        const char* p = text;
        TypeSyntax ts = readSignature(p);
        Conv cv;
        cv.mode = FreeVars::Quantify;
        TyRef body = convert(ts, Kind::Type, Kind::Type, {}, cv);
        return closeImplicit(body, cv);
    }

    TyRef closeImplicit(TyRef body, Conv& cv) {
        std::vector<Binder> bs;
        for (Slot* sl : cv.implicit) bs.push_back(sl->binder);
        return s.forall(std::move(bs), body);
    }

    /** Convert a source annotation; free variables per @p mode. */
    TyRef annotation(const hott_type_expr_t* t, FreeVars mode, const eshkol_ast_t* node) {
        Conv cv;
        cv.mode = mode;
        cv.node = node;
        TyRef body = convert(syntaxOf(t), Kind::Type, Kind::Type, {}, cv);
        return mode == FreeVars::Quantify ? closeImplicit(body, cv) : body;
    }

    TyRef slotUse(Slot* sl, Kind k, Kind fk, const std::set<std::string>& lacks, Conv& cv,
                  const std::string& name) {
        if (!sl->kinded) {
            sl->kinded = true;
            sl->binder.kind = k;
            sl->binder.field_kind = fk;
            sl->binder.unrestricted = (k == Kind::Type);
            sl->node = s.bound(sl->binder);
        } else if (sl->binder.kind != k) {
            fail(cv.node, "the type variable `" + name +
                              "` is used both as a value type and as a row or protocol");
        }
        sl->binder.lacks.insert(lacks.begin(), lacks.end());
        return sl->node;
    }

    TyRef kindedTerm(TyRef t, Kind k, const std::set<std::string>& lacks, Conv& cv,
                     const std::string& name) {
        TyRef p = st::TypeStore::prune(t);
        if (p->tag != Tag::Dyn && p->kind != k) {
            fail(cv.node, "the type variable `" + name +
                              "` is used both as a value type and as a row or protocol");
        }
        if (k == Kind::Row) {
            if (p->tag == Tag::Meta) {
                p->lacks.insert(lacks.begin(), lacks.end());
            } else if (p->tag == Tag::Skolem) {
                for (const auto& l : lacks) {
                    if (!p->lacks.count(l)) {
                        fail(cv.node, "the row variable `" + name + "` may already contain `" +
                                          l + "`");
                    }
                }
            }
        }
        return t;
    }

    TyRef convert(const TypeSyntax& ts, Kind k, Kind fk, const std::set<std::string>& lacks,
                  Conv& cv) {
        if (ts.invalid) return s.dyn();
        if (ts.atom) return convertAtom(ts.text, k, fk, lacks, cv);
        const std::string head = ts.head();
        const size_t n = ts.elems.size();
        auto expectKind = [&](Kind want, const char* what) {
            if (k != want) {
                fail(cv.node, std::string("`") + head + "` forms " + what +
                                  ", which is not what this position expects");
            }
        };
        if (head == "->" || head == "-o") {
            expectKind(Kind::Type, "a function type");
            if (n < 2) fail(cv.node, "a function type needs a result type");
            std::vector<TyRef> params;
            TyRef rest = nullptr;
            for (size_t i = 1; i + 1 < n; ++i) {
                const TypeSyntax& p = ts.elems[i];
                if (p.head() == "..." && p.elems.size() == 2) {
                    rest = convert(p.elems[1], Kind::Type, Kind::Type, {}, cv);
                } else {
                    params.push_back(convert(p, Kind::Type, Kind::Type, {}, cv));
                }
            }
            TyRef res = convert(ts.elems[n - 1], Kind::Type, Kind::Type, {}, cv);
            return s.arrow(std::move(params), res, head == "-o" ? Mode::OneShot : Mode::Unrestricted,
                           rest);
        }
        if (head == "forall") {
            if (n != 3 || ts.elems[1].atom) fail(cv.node, "`forall` takes a variable list and a body");
            std::map<std::string, Slot*> scope;
            std::vector<Slot*> order;
            auto addVar = [&](const TypeSyntax& v) {
                if (!v.atom || v.invalid) fail(cv.node, "`forall` binds type variable names");
                cv.slots.emplace_back();
                Slot* sl = &cv.slots.back();
                sl->binder.id = s.freshId();
                sl->binder.name = v.text;
                scope[v.text] = sl;
                order.push_back(sl);
            };
            for (const auto& v : ts.elems[1].elems) addVar(v);
            cv.scopes.push_back(scope);
            TyRef body = convert(ts.elems[2], k, fk, lacks, cv);
            cv.scopes.pop_back();
            std::vector<Binder> bs;
            for (Slot* sl : order) {
                if (!sl->kinded) {
                    sl->binder.kind = Kind::Type;
                    sl->binder.unrestricted = true;
                }
                bs.push_back(sl->binder);
            }
            return s.forall(std::move(bs), body);
        }
        if (head == "record") {
            expectKind(Kind::Type, "a record type");
            return s.record(convertRow(ts, Kind::Type, cv, "record"));
        }
        if (head == "chan") {
            expectKind(Kind::Type, "a channel endpoint type");
            if (n != 2) fail(cv.node, "`chan` takes one protocol");
            return s.chan(convert(ts.elems[1], Kind::Session, Kind::Type, {}, cv));
        }
        if (head == "!" || head == "?" || head == "send" || head == "recv") {
            expectKind(Kind::Session, "a protocol");
            if (n != 3) fail(cv.node, "`" + head + "` takes a message type and a continuation");
            TyRef payload = convert(ts.elems[1], Kind::Type, Kind::Type, {}, cv);
            TyRef cont = convert(ts.elems[2], Kind::Session, Kind::Type, {}, cv);
            return (head == "!" || head == "send") ? s.send(payload, cont) : s.recv(payload, cont);
        }
        if (head == "choose" || head == "offer") {
            expectKind(Kind::Session, "a protocol");
            TyRef row = convertRow(ts, Kind::Session, cv, head.c_str());
            return head == "choose" ? s.choose(row) : s.offer(row);
        }
        if (head == "rec" || head == "mu") {
            expectKind(Kind::Session, "a protocol");
            if (n != 3 || !ts.elems[1].atom) fail(cv.node, "`rec` takes a variable and a protocol");
            const int id = s.freshId();
            cv.rec.push_back({ts.elems[1].text, {id, ts.elems[1].text}});
            TyRef body = convert(ts.elems[2], Kind::Session, Kind::Type, {}, cv);
            cv.rec.pop_back();
            return s.mu(id, ts.elems[1].text, body);
        }
        if (head == "dual") {
            expectKind(Kind::Session, "a protocol");
            if (n != 2) fail(cv.node, "`dual` takes one protocol");
            return s.dual(convert(ts.elems[1], Kind::Session, Kind::Type, {}, cv));
        }
        if (head == "values") {
            expectKind(Kind::Type, "multiple values");
            std::vector<TyRef> el;
            for (size_t i = 1; i < n; ++i) el.push_back(convert(ts.elems[i], Kind::Type, Kind::Type, {}, cv));
            return s.values(std::move(el));
        }
        if (head == "+") return s.dyn();
        if (head.empty()) {
            if (n == 0) return s.null();
            fail(cv.node, "a type application needs a constructor name");
        }
        // Parameterised alias.
        auto al = aliases.find(head);
        if (al != aliases.end() && !al->second.params.empty()) {
            std::map<std::string, TypeSyntax> sub;
            for (size_t i = 0; i < al->second.params.size() && i + 1 < n; ++i) {
                sub[al->second.params[i]] = ts.elems[i + 1];
            }
            return convert(substSyntax(al->second.body, sub), k, fk, lacks, cv);
        }
        expectKind(Kind::Type, "a value type");
        std::string name = head;
        if (name == "*" || name == "product") name = "pair";
        if (name == "pointer") name = "ptr";
        std::vector<TyRef> args;
        for (size_t i = 1; i < n; ++i) {
            if (ts.elems[i].invalid) return s.dyn();   // value-indexed family
            args.push_back(convert(ts.elems[i], Kind::Type, Kind::Type, {}, cv));
        }
        return s.con(name, std::move(args));
    }

    /** `(record (l T) ... [tail])` / `(choose (l S) ... [tail])`. */
    TyRef convertRow(const TypeSyntax& ts, Kind field_kind, Conv& cv, const char* what) {
        std::vector<std::pair<std::string, TyRef>> fields;
        std::set<std::string> labels;
        TyRef tail = nullptr;
        for (size_t i = 1; i < ts.elems.size(); ++i) {
            const TypeSyntax& f = ts.elems[i];
            if (f.atom) {
                if (i + 1 != ts.elems.size()) {
                    fail(cv.node, std::string("the row variable of a `") + what +
                                      "` type must come last");
                }
                tail = nullptr;
                // Converted below once every label is known (lacks set).
                continue;
            }
            if (f.elems.size() != 2 || !f.elems[0].atom) {
                fail(cv.node, std::string("each entry of a `") + what + "` type is (label type)");
            }
            const std::string label = f.elems[0].text;
            if (!labels.insert(label).second) {
                fail(cv.node, "the label `" + label + "` occurs twice in one `" + what + "` type");
            }
            fields.push_back({label, convert(f.elems[1], field_kind, Kind::Type, {}, cv)});
        }
        const TypeSyntax& last = ts.elems.back();
        if (ts.elems.size() > 1 && last.atom) {
            tail = convert(last, Kind::Row, field_kind, labels, cv);
        } else {
            tail = s.rowEmpty(field_kind);
        }
        TyRef row = tail;
        for (auto it = fields.rbegin(); it != fields.rend(); ++it) {
            row = s.rowExtend(it->first, it->second, row);
        }
        return row;
    }

    TyRef convertAtom(const std::string& name, Kind k, Kind fk, const std::set<std::string>& lacks,
                      Conv& cv) {
        // Recursion variables, then quantified variables, then scoped ones.
        for (auto it = cv.rec.rbegin(); it != cv.rec.rend(); ++it) {
            if (it->first == name) {
                if (k != Kind::Session) fail(cv.node, "`" + name + "` is a protocol variable");
                return s.recVar(it->second.first, it->second.second);
            }
        }
        for (auto it = cv.scopes.rbegin(); it != cv.scopes.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return slotUse(f->second, k, fk, lacks, cv, name);
        }
        for (Slot* sl : cv.implicit) {
            if (sl->binder.name == name) return slotUse(sl, k, fk, lacks, cv, name);
        }
        for (auto it = tyvar_scopes.rbegin(); it != tyvar_scopes.rend(); ++it) {
            auto f = it->find(name);
            if (f != it->end()) return kindedTerm(f->second, k, lacks, cv, name);
        }
        if (k == Kind::Session && name == "end") return s.end();
        // Aliases (also for protocols: `(define-type Echo (rec x ...))`).
        auto al = aliases.find(name);
        if (al != aliases.end() && al->second.params.empty()) {
            return convert(al->second.body, k, fk, lacks, cv);
        }
        if (k == Kind::Type) {
            if (isNumericName(name)) return s.number();
            if (name == "string" || name == "String") return s.string();
            if (name == "boolean" || name == "bool" || name == "Boolean" || name == "Bool") {
                return s.boolean();
            }
            if (name == "char" || name == "Char") return s.character();
            if (name == "symbol" || name == "Symbol") return s.symbol();
            if (name == "null" || name == "Null" || name == "unit") return s.null();
            if (isDynName(name)) return s.dyn();
            if (env.lookupType(name)) return s.con(name);
        }
        // A free type variable.
        if (cv.mode == FreeVars::Quantify) {
            cv.slots.emplace_back();
            Slot* sl = &cv.slots.back();
            sl->binder.id = s.freshId();
            sl->binder.name = name;
            cv.implicit.push_back(sl);
            return slotUse(sl, k, fk, lacks, cv, name);
        }
        TyRef m = s.meta(k, level, fk);
        m->name = name + std::to_string(m->id);
        m->lacks = lacks;
        if (!tyvar_scopes.empty()) tyvar_scopes.back()[name] = m;
        return m;
    }

    bool syntaxIsStructural(const TypeSyntax& ts, int depth = 0) {
        if (depth > 64) return false;
        if (ts.invalid) return false;
        if (ts.atom) {
            auto al = aliases.find(ts.text);
            return al != aliases.end() && syntaxIsStructural(al->second.body, depth + 1);
        }
        const std::string h = ts.head();
        static const char* const kFormers[] = {"forall", "record", "chan", "-o", "values", "!", "?",
                                               "send", "recv", "choose", "offer", "rec", "mu",
                                               "dual"};
        for (const char* f : kFormers) {
            if (h == f) return true;
        }
        static const char* const kPlain[] = {"->", "list", "vector", "tensor", "ptr", "pointer",
                                             "pair", "*", "product", "+", ""};
        bool plain = false;
        for (const char* f : kPlain) plain = plain || h == f;
        if (!plain) {
            auto al = aliases.find(h);
            if (al == aliases.end()) return true;      // a user type constructor
            return syntaxIsStructural(al->second.body, depth + 1);
        }
        for (size_t i = 1; i < ts.elems.size(); ++i) {
            if (syntaxIsStructural(ts.elems[i], depth + 1)) return true;
        }
        return false;
    }

    bool annotationIsStructural(const hott_type_expr_t* t) {
        return t && syntaxIsStructural(syntaxOf(t));
    }

    // ------------------------------------------------------------------
    // Program pre-pass
    // ------------------------------------------------------------------
    void activate(const eshkol_ast_t* form) {
        auto on = [&](const char* m, const Signature* sigs, size_t ns, const char* const* rl,
                      size_t nr) {
            if (!requireNames(form, m) || modules.count(m)) return;
            modules.insert(m);
            for (size_t i = 0; i < ns; ++i) {
                if (user_defined.count(sigs[i].name)) continue;
                globals[sigs[i].name] = signature(sigs[i].type);
                structural.insert(sigs[i].name);
            }
            for (size_t i = 0; i < nr; ++i) {
                if (user_defined.count(rl[i])) continue;
                rules.insert(rl[i]);
                structural.insert(rl[i]);
            }
        };
        on("core.records", kRecordSignatures, std::size(kRecordSignatures), kRecordRules,
           std::size(kRecordRules));
        on("core.sessions", kSessionSignatures, std::size(kSessionSignatures), kSessionRules,
           std::size(kSessionRules));
    }

    void noteAlias(const eshkol_ast_t* form) {
        const auto& d = form->operation.define_type_op;
        if (!d.name) return;
        Alias a;
        for (uint64_t i = 0; i < d.num_type_params; ++i) a.params.push_back(d.type_params[i]);
        a.body = syntaxOf(d.type_expr);
        aliases[d.name] = std::move(a);
    }

    bool defineHasStructuralAnnotation(const eshkol_ast_t* form) {
        const auto& d = form->operation.define_op;
        if (d.name) {
            auto an = annotations.find(d.name);
            if (an != annotations.end() && annotationIsStructural(an->second)) return true;
        }
        if (annotationIsStructural(d.return_type)) return true;
        for (uint64_t i = 0; d.param_types && i < d.num_params; ++i) {
            if (annotationIsStructural(d.param_types[i])) return true;
        }
        return formHasStructuralAnnotation(d.value);
    }

    /** A lambda, let or `the` inside @p e carries a structural annotation. */
    bool formHasStructuralAnnotation(const eshkol_ast_t* e) {
        if (!e) return false;
        if (e->type == ESHKOL_OP) {
            const auto& op = e->operation;
            if (op.op == ESHKOL_THE_OP && annotationIsStructural(op.the_op.type_expr)) return true;
            if (op.op == ESHKOL_LAMBDA_OP) {
                if (annotationIsStructural(op.lambda_op.return_type)) return true;
                for (uint64_t i = 0; op.lambda_op.param_types && i < op.lambda_op.num_params; ++i) {
                    if (annotationIsStructural(op.lambda_op.param_types[i])) return true;
                }
            }
            if ((op.op == ESHKOL_LET_OP || op.op == ESHKOL_LET_STAR_OP || op.op == ESHKOL_LETREC_OP ||
                 op.op == ESHKOL_LETREC_STAR_OP) && op.let_op.binding_types) {
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    if (annotationIsStructural(op.let_op.binding_types[i])) return true;
                }
            }
        }
        bool found = false;
        forEachChild(e, [&](const eshkol_ast_t* c) { found = found || formHasStructuralAnnotation(c); });
        return found;
    }

    bool mentionsStructural(const eshkol_ast_t* e) {
        std::set<std::string> m;
        collectMentions(e, m);
        for (const auto& n : m) {
            if (structural.count(n)) return true;
        }
        return false;
    }

    /** Signature from complete annotations of a define, or (: name T). */
    TyRef declaredSignature(const eshkol_ast_t* form) {
        const auto& d = form->operation.define_op;
        auto an = annotations.find(d.name);
        if (an != annotations.end()) return annotation(an->second, FreeVars::Quantify, form);
        if (!d.is_function || !d.return_type) return nullptr;
        for (uint64_t i = 0; i < d.num_params; ++i) {
            if (!d.param_types || !d.param_types[i]) return nullptr;
        }
        std::vector<TypeSyntax> e{TypeSyntax::mkAtom("->")};
        for (uint64_t i = 0; i < d.num_params; ++i) e.push_back(syntaxOf(d.param_types[i]));
        if (d.is_variadic && d.rest_param) return nullptr;
        e.push_back(syntaxOf(d.return_type));
        Conv cv;
        cv.mode = FreeVars::Quantify;
        cv.node = form;
        TyRef body = convert(TypeSyntax::mkList(std::move(e)), Kind::Type, Kind::Type, {}, cv);
        return closeImplicit(body, cv);
    }

    void prepare(const eshkol_ast_t* asts, size_t n) {
        program_known = true;
        std::vector<const eshkol_ast_t*> defines;
        for (size_t i = 0; i < n; ++i) {
            const eshkol_ast_t* f = &asts[i];
            if (f->type != ESHKOL_OP) continue;
            if (f->operation.op == ESHKOL_DEFINE_OP && f->operation.define_op.name) {
                user_defined.insert(f->operation.define_op.name);
                defines.push_back(f);
            } else if (f->operation.op == ESHKOL_DEFINE_TYPE_OP) {
                noteAlias(f);
            } else if (f->operation.op == ESHKOL_TYPE_ANNOTATION_OP &&
                       f->operation.type_annotation_op.name) {
                annotations[f->operation.type_annotation_op.name] =
                    f->operation.type_annotation_op.type_expr;
            }
        }
        for (size_t i = 0; i < n; ++i) {
            const eshkol_ast_t* f = &asts[i];
            if (f->type == ESHKOL_OP && f->operation.op == ESHKOL_REQUIRE_OP) activate(f);
        }
        std::set<const eshkol_ast_t*> chosen;
        for (const eshkol_ast_t* f : defines) {
            if (defineHasStructuralAnnotation(f)) {
                chosen.insert(f);
                structural.insert(f->operation.define_op.name);
            }
        }
        // Close over the call graph in both directions: a definition that
        // mentions structural code is structural, and so is a definition that
        // structural code mentions (its callers need its type).
        std::map<std::string, const eshkol_ast_t*> by_name;
        for (const eshkol_ast_t* f : defines) by_name[f->operation.define_op.name] = f;
        auto pull = [&](const eshkol_ast_t* body) {
            bool grew = false;
            std::set<std::string> m;
            collectMentions(body, m);
            for (const auto& n : m) {
                auto it = by_name.find(n);
                if (it != by_name.end() && !chosen.count(it->second)) {
                    chosen.insert(it->second);
                    structural.insert(n);
                    grew = true;
                }
            }
            return grew;
        };
        bool grew = true;
        while (grew) {
            grew = false;
            for (const eshkol_ast_t* f : defines) {
                if (!chosen.count(f) && mentionsStructural(f->operation.define_op.value)) {
                    chosen.insert(f);
                    structural.insert(f->operation.define_op.name);
                    grew = true;
                }
                if (chosen.count(f)) grew = pull(f->operation.define_op.value) || grew;
            }
            for (size_t i = 0; i < n; ++i) {
                const eshkol_ast_t* f = &asts[i];
                if (f->type == ESHKOL_OP && f->operation.op != ESHKOL_DEFINE_OP &&
                    f->operation.op != ESHKOL_REQUIRE_OP && f->operation.op != ESHKOL_DEFINE_TYPE_OP &&
                    (mentionsStructural(f) || formHasStructuralAnnotation(f))) {
                    grew = pull(f) || grew;
                }
            }
        }
        for (const eshkol_ast_t* f : chosen) {
            const char* name = f->operation.define_op.name;
            pending.insert(name);
            try {
                if (TyRef sig = declaredSignature(f)) declared[name] = sig;
            } catch (const StructuralError& e) {
                tc.reportStructuralViolation(e.msg, e.node);
            }
        }
    }

    // ------------------------------------------------------------------
    // Environment
    // ------------------------------------------------------------------
    int bindLocal(const std::string& name, TyRef type, std::vector<const eshkol_ast_t*> region,
                  const eshkol_ast_t* site, const void* loop_op = nullptr, size_t loop_index = 0) {
        BinderRec r;
        r.name = name;
        r.type = type;
        r.region = std::move(region);
        r.site = site;
        r.loop_op = loop_op;
        r.loop_index = loop_index;
        binders.push_back(std::move(r));
        const int idx = static_cast<int>(binders.size()) - 1;
        locals.push_back({name, type, static_cast<int>(fn_stack.size()), idx});
        return idx;
    }

    const Local* lookupLocal(const std::string& name) {
        for (auto it = locals.rbegin(); it != locals.rend(); ++it) {
            if (it->name == name) return &*it;
        }
        return nullptr;
    }

    TyRef placeholderFor(const std::string& name) {
        auto it = placeholders.find(name);
        if (it != placeholders.end()) return it->second;
        TyRef m = s.meta(Kind::Type, 0);
        m->name = name;
        placeholders[name] = m;
        return m;
    }

    /** Type (sigma) of a variable reference; records closure captures. */
    TyRef variableSigma(const eshkol_ast_t* e, const std::string& name) {
        if (const Local* l = lookupLocal(name)) {
            for (int d = static_cast<int>(fn_stack.size()) - 1; d >= l->fn_depth; --d) {
                if (l->binder >= 0) captures.push_back({fn_stack[d], l->binder});
            }
            return l->type;
        }
        if (rules.count(name)) {
            fail(e, "`" + name + "` takes a quoted label and must be applied directly");
        }
        auto g = globals.find(name);
        if (g != globals.end()) return g->second;
        auto d = declared.find(name);
        if (d != declared.end()) return d->second;
        if (pending.count(name)) return placeholderFor(name);
        return s.dyn();
    }

    // ------------------------------------------------------------------
    // Bidirectional checking
    // ------------------------------------------------------------------
    TyRef inferRho(eshkol_ast_t* e) {
        const eshkol_ast_t* saved = current;
        current = e;
        TyRef t = inferRhoInner(e);
        current = saved;
        return t;
    }

    TyRef inferRhoInner(eshkol_ast_t* e) {
        if (!e) return s.dyn();
        switch (e->type) {
            case ESHKOL_INT8: case ESHKOL_INT16: case ESHKOL_INT32: case ESHKOL_INT64:
            case ESHKOL_UINT8: case ESHKOL_UINT16: case ESHKOL_UINT32: case ESHKOL_UINT64:
            case ESHKOL_DOUBLE: case ESHKOL_BIGNUM_LITERAL:
                return s.number();
            case ESHKOL_STRING: return s.string();
            case ESHKOL_BOOL: return s.boolean();
            case ESHKOL_CHAR: return s.character();
            case ESHKOL_NULL: return s.null();
            case ESHKOL_SYMBOL: return s.symbol();
            case ESHKOL_VAR:
                return s.instantiate(variableSigma(e, e->variable.id), level);
            case ESHKOL_OP:
                break;
            default:
                return s.dyn();
        }
        auto& op = e->operation;
        switch (op.op) {
            case ESHKOL_QUOTE_OP: {
                if (op.call_op.num_vars == 1 && op.call_op.variables &&
                    (op.call_op.variables[0].type == ESHKOL_VAR ||
                     op.call_op.variables[0].type == ESHKOL_SYMBOL)) {
                    return s.symbol();
                }
                return s.dyn();
            }
            case ESHKOL_LAMBDA_OP:
                return inferLambda(e);
            case ESHKOL_IF_OP:
                return ifForm(e, nullptr);
            case ESHKOL_CALL_OP:
                if (isIfCall(e)) return ifForm(e, nullptr);
                return application(e, nullptr);
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP: case ESHKOL_MUL_OP: case ESHKOL_DIV_OP:
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                    checkRho(&op.call_op.variables[i], s.number());
                }
                return s.number();
            case ESHKOL_SEQUENCE_OP:
                return sequence(op.sequence_op.expressions, op.sequence_op.num_expressions, nullptr);
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i) {
                    effect(&op.sequence_op.expressions[i]);
                }
                return s.dyn();
            case ESHKOL_LET_OP: case ESHKOL_LET_STAR_OP: case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP:
                return letForm(e, nullptr);
            case ESHKOL_COND_OP:
                return condForm(e, nullptr);
            case ESHKOL_CASE_OP:
                return caseForm(e, nullptr);
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) effect(&op.call_op.variables[i]);
                return s.dyn();
            case ESHKOL_SET_OP:
                setForm(e);
                return s.dyn();
            case ESHKOL_THE_OP: {
                TyRef sigma = annotation(op.the_op.type_expr, FreeVars::Fresh, e);
                checkSigma(op.the_op.expr, sigma);
                return s.instantiate(sigma, level);
            }
            case ESHKOL_VALUES_OP: {
                std::vector<TyRef> el;
                for (uint64_t i = 0; i < op.values_op.num_values; ++i) {
                    el.push_back(inferRho(&op.values_op.expressions[i]));
                }
                if (el.size() == 1) return el[0];
                return s.values(std::move(el));
            }
            case ESHKOL_LET_VALUES_OP:
            case ESHKOL_LET_STAR_VALUES_OP:
                return letValues(e, nullptr);
            default:
                // Outside the structural fragment: an untyped island. Its
                // value is `any`; a linear variable used inside it is
                // rejected by the strict linear judgment (undecidable use).
                return s.dyn();
        }
    }

    void checkRho(eshkol_ast_t* e, TyRef rho) {
        const eshkol_ast_t* saved = current;
        current = e;
        checkRhoInner(e, rho);
        current = saved;
    }

    void checkRhoInner(eshkol_ast_t* e, TyRef rho) {
        rho = st::TypeStore::prune(rho);
        if (e && e->type == ESHKOL_OP) {
            auto& op = e->operation;
            switch (op.op) {
                case ESHKOL_LAMBDA_OP:
                    if (rho->tag == Tag::Arrow) {
                        checkLambda(e, rho);
                        return;
                    }
                    break;
                case ESHKOL_IF_OP:
                    ifForm(e, rho);
                    return;
                case ESHKOL_CALL_OP:
                    if (isIfCall(e)) {
                        ifForm(e, rho);
                        return;
                    }
                    application(e, rho);
                    return;
                case ESHKOL_SEQUENCE_OP:
                    sequence(op.sequence_op.expressions, op.sequence_op.num_expressions, rho);
                    return;
                case ESHKOL_LET_OP: case ESHKOL_LET_STAR_OP: case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    letForm(e, rho);
                    return;
                case ESHKOL_COND_OP:
                    condForm(e, rho);
                    return;
                case ESHKOL_CASE_OP:
                    caseForm(e, rho);
                    return;
                case ESHKOL_LET_VALUES_OP:
                case ESHKOL_LET_STAR_VALUES_OP:
                    letValues(e, rho);
                    return;
                default:
                    break;
            }
        }
        TyRef t = inferRho(e);
        subsCheckRho(t, rho, e);
    }

    void checkSigma(eshkol_ast_t* e, TyRef sigma) {
        std::vector<TyRef> skolems;
        ++level;
        TyRef rho = s.skolemise(sigma, level, skolems);
        checkRho(e, rho);
        --level;
    }

    /** sigma1 is at least as polymorphic as sigma2 (sigma1 <= sigma2). */
    void subsCheck(TyRef sigma1, TyRef sigma2, const eshkol_ast_t* node) {
        std::vector<TyRef> skolems;
        ++level;
        TyRef rho2 = s.skolemise(sigma2, level, skolems);
        subsCheckRho(sigma1, rho2, node);
        --level;
    }

    void subsCheckRho(TyRef sigma1, TyRef rho2, const eshkol_ast_t* node) {
        TyRef a = st::TypeStore::prune(sigma1);
        TyRef b = st::TypeStore::prune(rho2);
        if (a->tag == Tag::Forall) {
            subsCheckRho(s.instantiate(a, level), b, node);
            return;
        }
        if (b->tag == Tag::Forall) {
            subsCheck(a, b, node);
            return;
        }
        if (a->tag == Tag::Arrow && b->tag == Tag::Arrow && a->args.size() == b->args.size() &&
            (a->b == nullptr) == (b->b == nullptr)) {
            if (a->mode == Mode::OneShot && b->mode == Mode::Unrestricted) {
                fail(node, "a one-shot function (-o) is used where a function that may be "
                           "called many times is expected: " + s.show(a) + " vs " + s.show(b));
            }
            for (size_t i = 0; i < a->args.size(); ++i) subsCheck(b->args[i], a->args[i], node);
            if (a->b) subsCheck(b->b, a->b, node);
            subsCheckRho(a->a, b->a, node);
            return;
        }
        if (a->tag == Tag::Record && b->tag == Tag::Record) {
            // Width subtyping where the expected record is closed: the
            // provided record may carry more fields (ADR-0046). Open rows
            // are matched by unification (row polymorphism).
            TyRef bt = st::TypeStore::rowTail(b->a);
            if (bt && bt->tag == Tag::RowEmpty) {
                TyRef row = a->a;
                for (const auto& f : st::TypeStore::rowFields(b->a)) {
                    TyRef field = nullptr, rest = nullptr;
                    solver.clearError();
                    if (!solver.rewriteRow(row, f.first, field, rest)) {
                        fail(node, "the record " + s.show(a) + " has no field `" + f.first +
                                       "` required by " + s.show(b));
                    }
                    drain(node);
                    subsCheckRho(field, f.second, node);
                    row = rest;
                }
                return;
            }
        }
        unify(a, b, node, "type mismatch");
    }

    // ---- forms ---------------------------------------------------------
    bool isIfCall(const eshkol_ast_t* e) {
        const char* n = varName(e->operation.call_op.func);
        return n && std::strcmp(n, "if") == 0 && !lookupLocal("if") &&
               e->operation.call_op.num_vars >= 2;
    }

    /** Evaluate @p e for its effect only; its value must not be linear. */
    void effect(eshkol_ast_t* e) { discarded.push_back({inferRho(e), e}); }

    TyRef resultOrFresh(TyRef expected) {
        return expected ? expected : s.meta(Kind::Type, level);
    }

    /**
     * Gradual join of two branch types (ADR-0013): their unifier when one
     * exists, otherwise `any`. The failed trial is undone completely. A
     * linear type never disappears into `any` (the obligation rejects it).
     */
    TyRef join(TyRef a, TyRef b, const eshkol_ast_t* node) {
        const size_t m = solver.mark();
        solver.clearError();
        if (solver.unify(a, b)) {
            solver.commit(m);
            drain(node);
            return a;
        }
        solver.rollback(m);
        dyn_buffer.clear();
        obligations.push_back({a, node});
        obligations.push_back({b, node});
        return s.dyn();
    }

    /** One branch: checked against @p expected, or joined into @p acc. */
    TyRef branch(eshkol_ast_t* e, TyRef acc, TyRef expected) {
        if (expected) {
            checkRho(e, expected);
            return expected;
        }
        TyRef t = inferRho(e);
        return acc ? join(acc, t, e) : t;
    }

    TyRef ifForm(eshkol_ast_t* e, TyRef expected) {
        auto& c = e->operation.call_op;
        effect(&c.variables[0]);
        if (c.num_vars < 3) {
            effect(&c.variables[1]);
            return s.dyn();
        }
        TyRef acc = branch(&c.variables[1], nullptr, expected);
        return branch(&c.variables[2], acc, expected);
    }

    TyRef sequence(eshkol_ast_t* items, uint64_t n, TyRef expected) {
        if (n == 0) return s.null();
        for (uint64_t i = 0; i + 1 < n; ++i) effect(&items[i]);
        if (expected) {
            checkRho(&items[n - 1], expected);
            return expected;
        }
        return inferRho(&items[n - 1]);
    }

    TyRef condForm(eshkol_ast_t* e, TyRef expected) {
        auto& c = e->operation.call_op;
        TyRef acc = nullptr;
        bool has_else = false;
        for (uint64_t i = 0; i < c.num_vars; ++i) {
            eshkol_ast_t* clause = &c.variables[i];
            if (clause->type != ESHKOL_OP || clause->operation.op != ESHKOL_CALL_OP) return s.dyn();
            auto& cl = clause->operation.call_op;
            const char* test = varName(cl.func);
            const bool is_else = test && std::strcmp(test, "else") == 0;
            if (!is_else) effect(cl.func);
            if (cl.num_vars == 2 && varName(&cl.variables[0]) &&
                std::strcmp(varName(&cl.variables[0]), "=>") == 0) {
                effect(&cl.variables[1]);
                continue;
            }
            if (cl.num_vars == 0) continue;
            for (uint64_t j = 0; j + 1 < cl.num_vars; ++j) effect(&cl.variables[j]);
            acc = branch(&cl.variables[cl.num_vars - 1], acc, expected);
            if (is_else) {
                has_else = true;
                break;
            }
        }
        return (has_else && acc) ? acc : s.dyn();
    }

    TyRef caseForm(eshkol_ast_t* e, TyRef expected) {
        auto& c = e->operation.call_op;
        inferRho(c.func);
        TyRef acc = nullptr;
        bool has_else = false;
        for (uint64_t i = 0; i < c.num_vars; ++i) {
            eshkol_ast_t* clause = &c.variables[i];
            if (clause->type != ESHKOL_CONS) return s.dyn();
            const char* d = varName(clause->cons_cell.car);
            const bool is_else = d && std::strcmp(d, "else") == 0;
            eshkol_ast_t* body = clause->cons_cell.cdr;
            if (body && body->type == ESHKOL_OP && body->operation.op == ESHKOL_CALL_OP) {
                auto& bc = body->operation.call_op;
                if (bc.num_vars == 0) continue;
                for (uint64_t j = 0; j + 1 < bc.num_vars; ++j) effect(&bc.variables[j]);
                acc = branch(&bc.variables[bc.num_vars - 1], acc, expected);
            } else if (body) {
                acc = branch(body, acc, expected);
            }
            if (is_else) {
                has_else = true;
                break;
            }
        }
        return (has_else && acc) ? acc : s.dyn();
    }

    void setForm(eshkol_ast_t* e) {
        auto& so = e->operation.set_op;
        if (!so.name) return;
        if (const Local* l = lookupLocal(so.name)) {
            TyRef t = l->type;
            for (int d = static_cast<int>(fn_stack.size()) - 1; d >= l->fn_depth; --d) {
                if (l->binder >= 0) captures.push_back({fn_stack[d], l->binder});
            }
            if (l->annotated) {
                checkRho(so.value, s.instantiate(t, level));
            } else {
                // An unannotated variable may be re-assigned a value of a
                // different type (Scheme); the binding's type then joins to
                // `any` for this assignment, gradually.
                join(s.instantiate(t, level), inferRho(so.value), so.value);
            }
            return;
        }
        inferRho(so.value);
    }

    std::vector<const eshkol_ast_t*> letRegion(const eshkol_ast_t* e, uint64_t i) {
        const auto& let = e->operation.let_op;
        std::vector<const eshkol_ast_t*> region;
        const auto opk = e->operation.op;
        if (opk == ESHKOL_LETREC_OP || opk == ESHKOL_LETREC_STAR_OP) {
            for (uint64_t j = 0; j < let.num_bindings; ++j) region.push_back(let.bindings[j].cons_cell.cdr);
        } else if (opk == ESHKOL_LET_STAR_OP) {
            for (uint64_t j = i + 1; j < let.num_bindings; ++j) {
                region.push_back(let.bindings[j].cons_cell.cdr);
            }
        }
        region.push_back(let.body);
        return region;
    }

    TyRef letForm(eshkol_ast_t* e, TyRef expected) {
        auto& let = e->operation.let_op;
        const size_t mark = locals.size();
        const auto opk = e->operation.op;
        if (let.name) return namedLet(e, expected);
        for (uint64_t i = 0; i < let.num_bindings; ++i) {
            if (let.bindings[i].type != ESHKOL_CONS || !varName(let.bindings[i].cons_cell.car)) {
                fail(e, "malformed let binding");
            }
        }
        const bool rec = opk == ESHKOL_LETREC_OP || opk == ESHKOL_LETREC_STAR_OP;
        std::vector<TyRef> types(let.num_bindings, nullptr);
        auto annotated = [&](uint64_t i) -> TyRef {
            if (let.binding_types && let.binding_types[i]) {
                return annotation(let.binding_types[i], FreeVars::Fresh, e);
            }
            return nullptr;
        };
        if (rec) {
            for (uint64_t i = 0; i < let.num_bindings; ++i) {
                TyRef t = annotated(i);
                types[i] = t ? t : s.meta(Kind::Type, level);
                bindLocal(varName(let.bindings[i].cons_cell.car), types[i], letRegion(e, i), e);
                locals.back().annotated = let.binding_types && let.binding_types[i];
            }
            for (uint64_t i = 0; i < let.num_bindings; ++i) {
                eshkol_ast_t* value = let.bindings[i].cons_cell.cdr;
                const std::string name = varName(let.bindings[i].cons_cell.car);
                // Internal-define lowering: a '() placeholder assigned later.
                if (opk == ESHKOL_LETREC_STAR_OP && value && value->type == ESHKOL_NULL &&
                    let.body && assignsVar(let.body, name)) {
                    continue;
                }
                checkSigma(value, types[i]);
            }
        } else {
            for (uint64_t i = 0; i < let.num_bindings; ++i) {
                eshkol_ast_t* value = let.bindings[i].cons_cell.cdr;
                TyRef t = annotated(i);
                if (t) {
                    checkSigma(value, t);
                } else {
                    t = inferRho(value);
                }
                types[i] = t;
                if (opk == ESHKOL_LET_STAR_OP) {
                    bindLocal(varName(let.bindings[i].cons_cell.car), t, letRegion(e, i), e);
                    locals.back().annotated = let.binding_types && let.binding_types[i];
                }
            }
            if (opk != ESHKOL_LET_STAR_OP) {
                for (uint64_t i = 0; i < let.num_bindings; ++i) {
                    bindLocal(varName(let.bindings[i].cons_cell.car), types[i], letRegion(e, i), e);
                    locals.back().annotated = let.binding_types && let.binding_types[i];
                }
            }
        }
        TyRef r;
        if (expected) {
            checkRho(let.body, expected);
            r = expected;
        } else {
            r = inferRho(let.body);
        }
        locals.resize(mark);
        return r;
    }

    TyRef namedLet(eshkol_ast_t* e, TyRef expected) {
        auto& let = e->operation.let_op;
        const size_t mark = locals.size();
        std::vector<TyRef> params;
        for (uint64_t i = 0; i < let.num_bindings; ++i) {
            if (let.bindings[i].type != ESHKOL_CONS || !varName(let.bindings[i].cons_cell.car)) {
                fail(e, "malformed named let binding");
            }
            TyRef p = nullptr;
            if (let.binding_types && let.binding_types[i]) {
                p = annotation(let.binding_types[i], FreeVars::Fresh, e);
                checkSigma(let.bindings[i].cons_cell.cdr, p);
            } else {
                p = s.meta(Kind::Type, level);
                checkRho(let.bindings[i].cons_cell.cdr, p);
            }
            params.push_back(p);
        }
        TyRef result = resultOrFresh(expected);
        TyRef loop = s.arrow(params, result);
        // The loop procedure is a closure over the body: captures inside the
        // body are captures of the loop function.
        bindLocal(let.name, loop, {let.body}, e);
        fn_stack.push_back(e);
        lambda_types[e] = loop;
        for (uint64_t i = 0; i < let.num_bindings; ++i) {
            bindLocal(varName(let.bindings[i].cons_cell.car), params[i], {let.body}, e,
                      static_cast<const void*>(&e->operation), i);
        }
        checkRho(let.body, result);
        fn_stack.pop_back();
        locals.resize(mark);
        return result;
    }

    TyRef letValues(eshkol_ast_t* e, TyRef expected) {
        auto& lv = e->operation.let_values_op;
        const size_t mark = locals.size();
        const bool sequential = e->operation.op == ESHKOL_LET_STAR_VALUES_OP;
        std::vector<std::pair<std::string, TyRef>> pending_binds;
        for (uint64_t i = 0; i < lv.num_bindings; ++i) {
            TyRef t = st::TypeStore::prune(inferRho(&lv.producers[i]));
            const uint64_t count = lv.binding_var_counts ? lv.binding_var_counts[i] : 0;
            std::vector<TyRef> parts;
            if (t->tag == Tag::Values) {
                if (t->args.size() != count) {
                    fail(&lv.producers[i], "this produces " + std::to_string(t->args.size()) +
                                               " values, but " + std::to_string(count) +
                                               " are bound");
                }
                parts = t->args;
            } else if (t->tag == Tag::Dyn) {
                parts.assign(count, s.dyn());
            } else if (count == 1) {
                parts.push_back(t);
            } else {
                for (uint64_t j = 0; j < count; ++j) parts.push_back(s.meta(Kind::Type, level));
                unify(t, s.values(parts), &lv.producers[i], "multiple values");
            }
            std::vector<const eshkol_ast_t*> region;
            if (sequential) {
                for (uint64_t j = i + 1; j < lv.num_bindings; ++j) region.push_back(&lv.producers[j]);
            }
            region.push_back(lv.body);
            for (uint64_t j = 0; j < count; ++j) {
                const char* vn = lv.binding_vars[i][j];
                if (!vn) continue;
                if (sequential) {
                    bindLocal(vn, parts[j], region, e);
                } else {
                    pending_binds.push_back({vn, parts[j]});
                }
            }
        }
        for (auto& pb : pending_binds) bindLocal(pb.first, pb.second, {lv.body}, e);
        TyRef r;
        if (expected) {
            checkRho(lv.body, expected);
            r = expected;
        } else {
            r = inferRho(lv.body);
        }
        locals.resize(mark);
        return r;
    }

    // ---- functions -----------------------------------------------------
    struct FnShape {
        const eshkol_ast_t* node;
        eshkol_ast_t* params;
        uint64_t num_params;
        const char* rest;
        hott_type_expr_t** param_types;
        hott_type_expr_t* return_type;
        eshkol_ast_t* body;
    };

    FnShape shapeOfLambda(eshkol_ast_t* e) {
        auto& l = e->operation.lambda_op;
        return {e, l.parameters, l.num_params, l.is_variadic ? l.rest_param : nullptr,
                l.param_types, l.return_type, l.body};
    }

    FnShape shapeOfDefine(eshkol_ast_t* e) {
        auto& d = e->operation.define_op;
        return {e, d.parameters, d.num_params, d.is_variadic ? d.rest_param : nullptr,
                d.param_types, d.return_type, d.value};
    }

    std::string paramName(const FnShape& f, uint64_t i) {
        const char* n = varName(&f.params[i]);
        if (!n) fail(f.node, "malformed parameter");
        return n;
    }

    TyRef inferFunction(const FnShape& f) {
        const size_t mark = locals.size();
        tyvar_scopes.emplace_back();
        std::vector<TyRef> params;
        for (uint64_t i = 0; i < f.num_params; ++i) {
            TyRef p = (f.param_types && f.param_types[i])
                          ? annotation(f.param_types[i], FreeVars::Fresh, f.node)
                          : s.meta(Kind::Type, level);
            params.push_back(p);
        }
        TyRef rest = f.rest ? s.meta(Kind::Type, level) : nullptr;
        TyRef ret = f.return_type ? annotation(f.return_type, FreeVars::Fresh, f.node) : nullptr;
        fn_stack.push_back(f.node);
        for (uint64_t i = 0; i < f.num_params; ++i) {
            bindLocal(paramName(f, i), params[i], {f.body}, f.node);
            locals.back().annotated = f.param_types && f.param_types[i];
        }
        if (rest) bindLocal(f.rest, s.con("list", {rest}), {f.body}, f.node);
        TyRef result;
        if (ret) {
            checkSigma(f.body, ret);
            result = ret;
        } else {
            result = inferRho(f.body);
        }
        fn_stack.pop_back();
        tyvar_scopes.pop_back();
        locals.resize(mark);
        TyRef t = s.arrow(std::move(params), result, Mode::Unrestricted, rest);
        lambda_types[f.node] = t;
        return t;
    }

    void checkFunction(const FnShape& f, TyRef rho) {
        rho = st::TypeStore::prune(rho);
        if (rho->tag != Tag::Arrow) {
            unify(rho, inferFunction(f), f.node, "function type");
            return;
        }
        if (rho->args.size() != f.num_params || (rho->b != nullptr) != (f.rest != nullptr)) {
            fail(f.node, "this function takes " + std::to_string(f.num_params) +
                             " parameter(s), but its expected type is " + s.show(rho));
        }
        const size_t mark = locals.size();
        tyvar_scopes.emplace_back();
        std::vector<TyRef> params;
        for (uint64_t i = 0; i < f.num_params; ++i) {
            TyRef p = rho->args[i];
            if (f.param_types && f.param_types[i]) {
                TyRef ann = annotation(f.param_types[i], FreeVars::Fresh, f.node);
                subsCheck(p, ann, f.node);
                p = ann;
            }
            params.push_back(p);
        }
        fn_stack.push_back(f.node);
        for (uint64_t i = 0; i < f.num_params; ++i) {
            bindLocal(paramName(f, i), params[i], {f.body}, f.node);
            locals.back().annotated = true;   // typed by its signature or context
        }
        if (f.rest) bindLocal(f.rest, s.con("list", {rho->b}), {f.body}, f.node);
        if (f.return_type) {
            TyRef ret = annotation(f.return_type, FreeVars::Fresh, f.node);
            checkSigma(f.body, ret);
            subsCheck(ret, rho->a, f.node);
        } else {
            checkSigma(f.body, rho->a);
        }
        fn_stack.pop_back();
        tyvar_scopes.pop_back();
        locals.resize(mark);
        lambda_types[f.node] = rho;
    }

    TyRef inferLambda(eshkol_ast_t* e) { return inferFunction(shapeOfLambda(e)); }
    void checkLambda(eshkol_ast_t* e, TyRef rho) { checkFunction(shapeOfLambda(e), rho); }

    // ---- application -----------------------------------------------------
    static const char* symbolText(const eshkol_ast_t* q, std::string& buf) {
        if (!q) return nullptr;
        if (q->type == ESHKOL_VAR) return q->variable.id;
        if (q->type == ESHKOL_SYMBOL && q->str_val.ptr) {
            buf.assign(q->str_val.ptr, q->str_val.size);
            return buf.c_str();
        }
        return nullptr;
    }

    std::string labelOf(eshkol_ast_t* a, const char* form) {
        std::string buf;
        if (const char* t = symbolText(a && a->type == ESHKOL_SYMBOL ? a : nullptr, buf)) return t;
        if (a && a->type == ESHKOL_OP && a->operation.op == ESHKOL_QUOTE_OP &&
            a->operation.call_op.num_vars == 1 && a->operation.call_op.variables) {
            if (const char* t = symbolText(&a->operation.call_op.variables[0], buf)) return t;
        }
        fail(a, std::string("`") + form + "` needs a quoted label literal such as 'name here");
    }

    TyRef finish(TyRef t, TyRef expected, const eshkol_ast_t* node) {
        if (expected) {
            subsCheckRho(t, expected, node);
            return expected;
        }
        return t;
    }

    TyRef recordRule(eshkol_ast_t* e, const std::string& f, TyRef expected) {
        auto& c = e->operation.call_op;
        auto arity = [&](uint64_t n) {
            if (c.num_vars != n) {
                fail(e, "`" + f + "` takes " + std::to_string(n) + " arguments");
            }
        };
        if (f == "record") {
            if (c.num_vars % 2 != 0) fail(e, "`record` takes label/value pairs");
            std::vector<std::pair<std::string, TyRef>> fields;
            std::set<std::string> seen;
            for (uint64_t i = 0; i < c.num_vars; i += 2) {
                std::string l = labelOf(&c.variables[i], "record");
                if (!seen.insert(l).second) fail(e, "the label `" + l + "` occurs twice in one record");
                TyRef t = inferRho(&c.variables[i + 1]);
                record_fields.push_back({t, &c.variables[i + 1]});
                fields.push_back({l, t});
            }
            TyRef row = s.rowEmpty();
            for (auto it = fields.rbegin(); it != fields.rend(); ++it) row = s.rowExtend(it->first, it->second, row);
            return finish(s.record(row), expected, e);
        }
        if (f == "record-ref") {
            arity(2);
            std::string l = labelOf(&c.variables[1], "record-ref");
            TyRef a = s.meta(Kind::Type, level);
            TyRef r = s.meta(Kind::Row, level);
            r->lacks.insert(l);
            checkRho(&c.variables[0], s.record(s.rowExtend(l, a, r)));
            return finish(a, expected, e);
        }
        if (f == "record-extend") {
            arity(3);
            std::string l = labelOf(&c.variables[1], "record-extend");
            TyRef r = s.meta(Kind::Row, level);
            r->lacks.insert(l);
            checkRho(&c.variables[0], s.record(r));
            TyRef t = inferRho(&c.variables[2]);
            record_fields.push_back({t, &c.variables[2]});
            return finish(s.record(s.rowExtend(l, t, r)), expected, e);
        }
        if (f == "record-remove") {
            arity(2);
            std::string l = labelOf(&c.variables[1], "record-remove");
            TyRef a = s.meta(Kind::Type, level);
            TyRef r = s.meta(Kind::Row, level);
            r->lacks.insert(l);
            checkRho(&c.variables[0], s.record(s.rowExtend(l, a, r)));
            return finish(s.record(r), expected, e);
        }
        // record-update
        arity(3);
        std::string l = labelOf(&c.variables[1], "record-update");
        TyRef a = s.meta(Kind::Type, level);
        TyRef r = s.meta(Kind::Row, level);
        r->lacks.insert(l);
        checkRho(&c.variables[0], s.record(s.rowExtend(l, a, r)));
        TyRef t = inferRho(&c.variables[2]);
        record_fields.push_back({t, &c.variables[2]});
        return finish(s.record(s.rowExtend(l, t, r)), expected, e);
    }

    TyRef sessionRule(eshkol_ast_t* e, const std::string& f, TyRef expected) {
        auto& c = e->operation.call_op;
        if (f == "session-select") {
            if (c.num_vars != 2) fail(e, "`session-select` takes an endpoint and a label");
            std::string l = labelOf(&c.variables[1], "session-select");
            TyRef next = s.meta(Kind::Session, level);
            TyRef others = s.meta(Kind::Row, level, Kind::Session);
            others->lacks.insert(l);
            checkRho(&c.variables[0], s.chan(s.choose(s.rowExtend(l, next, others))));
            return finish(s.chan(next), expected, e);
        }
        // session-offer endpoint 'label handler ...
        if (c.num_vars < 3 || c.num_vars % 2 != 1) {
            fail(e, "`session-offer` takes an endpoint and label/handler pairs");
        }
        std::vector<std::pair<std::string, TyRef>> branches;
        std::set<std::string> seen;
        for (uint64_t i = 1; i < c.num_vars; i += 2) {
            std::string l = labelOf(&c.variables[i], "session-offer");
            if (!seen.insert(l).second) fail(e, "the label `" + l + "` is handled twice");
            branches.push_back({l, s.meta(Kind::Session, level)});
        }
        TyRef row = s.rowEmpty(Kind::Session);
        for (auto it = branches.rbegin(); it != branches.rend(); ++it) row = s.rowExtend(it->first, it->second, row);
        checkRho(&c.variables[0], s.chan(s.offer(row)));
        TyRef result = resultOrFresh(expected);
        for (size_t k = 0; k < branches.size(); ++k) {
            TyRef handler = s.arrow({s.chan(branches[k].second)}, result, Mode::OneShot);
            checkSigma(&c.variables[2 + 2 * k], handler);
        }
        return result;
    }

    TyRef application(eshkol_ast_t* e, TyRef expected) {
        auto& c = e->operation.call_op;
        if (const char* fname = varName(c.func)) {
            if (rules.count(fname) && !lookupLocal(fname)) {
                std::string f = fname;
                if (f.rfind("record", 0) == 0) return recordRule(e, f, expected);
                return sessionRule(e, f, expected);
            }
        }
        TyRef ft = st::TypeStore::prune(inferRho(c.func));
        if (ft->tag == Tag::Meta) {
            std::vector<TyRef> ps;
            for (uint64_t i = 0; i < c.num_vars; ++i) ps.push_back(s.meta(Kind::Type, level));
            TyRef arrow = s.arrow(ps, s.meta(Kind::Type, level));
            unify(ft, arrow, e, "call");
            ft = arrow;
        }
        if (ft->tag == Tag::Dyn) {
            for (uint64_t i = 0; i < c.num_vars; ++i) {
                TyRef t = inferRho(&c.variables[i]);
                obligations.push_back({t, &c.variables[i]});
            }
            return s.dyn();
        }
        if (ft->tag != Tag::Arrow) {
            fail(e, "this calls a value of type " + s.show(ft) + ", which is not a function");
        }
        const size_t np = ft->args.size();
        if (c.num_vars < np || (!ft->b && c.num_vars != np)) {
            const char* fname = varName(c.func);
            fail(e, std::string("`") + (fname ? fname : "this function") + "` expects " +
                        std::to_string(np) + (ft->b ? " or more" : "") + " argument(s), but is given " +
                        std::to_string(c.num_vars));
        }
        for (uint64_t i = 0; i < c.num_vars; ++i) {
            checkSigma(&c.variables[i], i < np ? ft->args[i] : ft->b);
        }
        return finish(s.instantiate(ft->a, level), expected, e);
    }

    // ------------------------------------------------------------------
    // Per-form driver
    // ------------------------------------------------------------------
    void resetForm() {
        locals.clear();
        fn_stack.clear();
        binders.clear();
        captures.clear();
        lambda_types.clear();
        obligations.clear();
        dyn_buffer.clear();
        record_fields.clear();
        discarded.clear();
        tyvar_scopes.clear();
        current = nullptr;
    }

    void finishForm() {
        for (const auto& d : discarded) {
            TyRef t = s.zonk(d.first);
            if (st::TypeStore::isLinear(t)) {
                fail(d.second, "the linear value of type " + s.show(t) +
                                   " produced here is discarded; it must be used exactly once");
            }
        }
        for (const auto& ob : obligations) {
            TyRef t = s.zonk(ob.first);
            if (st::TypeStore::isLinear(t)) {
                fail(ob.second, "a linear value of type " + s.show(t) +
                                    " flows into code whose type is not known here (`any`), "
                                    "where its use cannot be tracked");
            }
        }
        for (const auto& rf : record_fields) {
            TyRef t = s.zonk(rf.first);
            if (st::TypeStore::isLinear(t)) {
                fail(rf.second, "a record field cannot hold the linear value of type " + s.show(t));
            }
        }
        std::set<std::pair<const eshkol_ast_t*, int>> seen;
        for (const auto& cap : captures) {
            if (!seen.insert({cap.lambda, cap.binder}).second) continue;
            TyRef bt = s.zonk(binders[cap.binder].type);
            if (!st::TypeStore::isLinear(bt)) continue;
            auto lt = lambda_types.find(cap.lambda);
            TyRef ft = lt == lambda_types.end() ? nullptr : st::TypeStore::prune(lt->second);
            if (!ft || ft->tag != Tag::Arrow || ft->mode != Mode::OneShot) {
                // A named let is its own loop, not a closure value: it may
                // not capture a linear variable either (it could run twice).
                fail(cap.lambda, "this closure captures the linear variable `" +
                                     binders[cap.binder].name + "` (" + s.show(bt) +
                                     "), so it must be one-shot: give it a `-o` type or pass it "
                                     "where a `-o` function is expected");
            }
        }
        for (const auto& b : binders) {
            if (b.loop_op && st::TypeStore::isLinear(s.zonk(b.type))) {
                tc.markEnforcedLoopParam(b.loop_op, b.loop_index);
            }
        }
        for (const auto& b : binders) {
            if (!st::TypeStore::isLinear(s.zonk(b.type))) continue;
            tc.enforceLinearRegions({{b.name, b.region}}, b.site, /*strict=*/true);
        }
    }

    void checkDefine(eshkol_ast_t* form) {
        auto& d = form->operation.define_op;
        const std::string name = d.name;
        pending.erase(name);
        if (d.is_external) {
            auto dec = declared.find(name);
            globals[name] = dec != declared.end() ? dec->second : s.dyn();
            return;
        }
        auto dec = declared.find(name);
        TyRef sigma = dec != declared.end() ? dec->second : nullptr;
        level = 1;
        if (sigma) {
            globals[name] = sigma;   // polymorphic recursion through the signature
            std::vector<TyRef> skolems;
            ++level;
            TyRef rho = s.skolemise(sigma, level, skolems);
            std::map<std::string, TyRef> scope;
            for (TyRef k : skolems) scope[k->name] = k;
            tyvar_scopes.push_back(scope);
            if (d.is_function) {
                FnShape f = shapeOfDefine(form);
                f.param_types = nullptr;      // already part of the signature
                f.return_type = nullptr;
                checkFunction(f, rho);
            } else {
                checkRho(d.value, rho);
            }
            tyvar_scopes.pop_back();
            --level;
            finishForm();
            level = 0;
            report(name, sigma);
            return;
        }
        TyRef self = placeholders.count(name) ? placeholders[name] : s.meta(Kind::Type, level);
        TyRef t;
        if (d.is_function) {
            bindLocal(name, self, {}, form);
            binders.back().site = nullptr;   // the recursive reference is not a resource
            t = inferFunction(shapeOfDefine(form));
            unify(self, t, form, "recursive use of `" + name + "`");
        } else {
            t = inferRho(d.value);
            if (placeholders.count(name)) unify(self, t, form, "use of `" + name + "` before its definition");
        }
        finishForm();
        TyRef zt = s.zonk(t);
        if (st::TypeStore::isLinear(zt)) {
            fail(form, "the top-level binding `" + name + "` has the linear type " + s.show(zt) +
                           "; bind linear values inside a function or a let, where their use is "
                           "tracked");
        }
        const bool generalisable = d.is_function ||
                                   (d.value && d.value->type == ESHKOL_OP &&
                                    d.value->operation.op == ESHKOL_LAMBDA_OP);
        level = 0;
        TyRef scheme = (generalisable && !placeholders.count(name)) ? s.generalize(t, 0) : zt;
        globals[name] = scheme;
        report(name, scheme);
    }

    void report(const std::string& name, TyRef scheme) {
        if (dump) std::fprintf(stderr, "[types] %s : %s\n", name.c_str(), s.show(scheme).c_str());
    }

    void checkExpression(eshkol_ast_t* form) {
        level = 1;
        effect(form);
        finishForm();
        level = 0;
    }

    void topLevel(eshkol_ast_t* form) {
        if (!form || form->type != ESHKOL_OP) {
            return;
        }
        auto& op = form->operation;
        switch (op.op) {
            case ESHKOL_REQUIRE_OP:
                activate(form);
                return;
            case ESHKOL_DEFINE_TYPE_OP:
                if (!program_known) noteAlias(form);
                return;
            case ESHKOL_TYPE_ANNOTATION_OP:
                if (!program_known && op.type_annotation_op.name) {
                    annotations[op.type_annotation_op.name] = op.type_annotation_op.type_expr;
                    if (annotationIsStructural(op.type_annotation_op.type_expr)) {
                        structural.insert(op.type_annotation_op.name);
                        declared[op.type_annotation_op.name] =
                            annotation(op.type_annotation_op.type_expr, FreeVars::Quantify, form);
                    }
                }
                return;
            case ESHKOL_DEFINE_OP: {
                if (!op.define_op.name) return;
                const std::string name = op.define_op.name;
                if (!program_known && !structural.count(name) &&
                    (defineHasStructuralAnnotation(form) || mentionsStructural(op.define_op.value))) {
                    structural.insert(name);
                    if (TyRef sig = declaredSignature(form)) declared[name] = sig;
                }
                if (structural.count(name) && !rules.count(name)) checkDefine(form);
                return;
            }
            default:
                if (mentionsStructural(form) || formHasStructuralAnnotation(form)) checkExpression(form);
                return;
        }
    }
};

// ============================================================================
// Public surface
// ============================================================================

StructuralChecker::StructuralChecker(TypeChecker& tc, TypeEnvironment& env)
    : impl_(std::make_unique<Impl>(tc, env)) {}

StructuralChecker::~StructuralChecker() = default;

void StructuralChecker::beginProgram(const eshkol_ast_t* asts, size_t num_asts) {
    if (!asts || num_asts == 0) return;
    impl_->prepare(asts, num_asts);
}

void StructuralChecker::checkTopLevel(eshkol_ast_t* form) {
    try {
        impl_->topLevel(form);
    } catch (const StructuralError& e) {
        impl_->tc.reportStructuralViolation(e.msg, e.node);
        impl_->level = 0;
    }
    impl_->resetForm();
}

std::optional<std::string> StructuralChecker::schemeOf(const std::string& name) {
    auto it = impl_->globals.find(name);
    if (it == impl_->globals.end()) return std::nullopt;
    return impl_->s.show(it->second);
}

}  // namespace eshkol::hott
