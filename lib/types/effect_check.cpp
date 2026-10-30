/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Effect-row inference and checking (ADR-0045). See effect_check.h.
 */
#include <eshkol/types/effect_check.h>
#include <eshkol/types/type_checker.h>
#include <eshkol/frontend/syntax_color.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace eshkol::effects {
namespace {

constexpr const char* kUnknownTail = "?";

struct Witness {
    std::string what;
    uint32_t line = 0;
    uint32_t column = 0;
};

/** labels ∪ tails; a tail stands for the latent effects of something not yet
 *  known, minus the labels discharged between it and here. */
struct Row {
    std::map<std::string, Witness> labels;
    std::map<std::string, std::set<std::string>> tails;   // var -> excluded labels
    std::map<std::string, Witness> tail_witness;

    bool operator==(const Row& o) const {
        if (tails != o.tails || labels.size() != o.labels.size()) return false;
        for (const auto& [l, w] : labels) if (!o.labels.count(l)) return false;
        return true;
    }

    void addLabel(const std::string& l, const Witness& w) {
        if (!labels.count(l)) labels[l] = w;
    }

    void addTail(const std::string& var, const std::set<std::string>& minus, const Witness& w) {
        auto it = tails.find(var);
        if (it == tails.end()) {
            tails[var] = minus;
            tail_witness[var] = w;
            return;
        }
        // join: a label is excluded only if every contribution excluded it
        std::set<std::string> both;
        for (const auto& l : it->second) if (minus.count(l)) both.insert(l);
        it->second = both;
    }

    void join(const Row& o) {
        for (const auto& [l, w] : o.labels) addLabel(l, w);
        for (const auto& [v, m] : o.tails) {
            auto w = o.tail_witness.find(v);
            addTail(v, m, w == o.tail_witness.end() ? Witness{} : w->second);
        }
    }

    /** Remove @p handled from the row (a handler or catch-all guard). */
    void discharge(const std::set<std::string>& handled) {
        for (const auto& l : handled) labels.erase(l);
        for (auto& [v, m] : tails) m.insert(handled.begin(), handled.end());
    }

    /** This row with @p minus removed, as a contribution through a tail. */
    Row without(const std::set<std::string>& minus) const {
        Row r = *this;
        r.discharge(minus);
        return r;
    }

    std::string str() const {
        std::string s = "<";
        bool first = true;
        for (const auto& [l, w] : labels) {
            if (!first) s += ", ";
            s += l;
            first = false;
        }
        for (const auto& [v, m] : tails) {
            s += first ? "| " : " | ";
            s += v;
            if (!m.empty()) {
                s += " -";
                for (const auto& l : m) s += " " + l;
            }
            first = false;
        }
        return s + ">";
    }
};

/* ── the signature registry ──────────────────────────────────────────────── */

struct BuiltinSig {
    std::vector<std::string> labels;
    std::vector<int> applies;     // argument positions whose procedures it calls
    bool unknown = false;         // calls something the analysis cannot see
};

const std::map<std::string, BuiltinSig>& registry() {
    static const std::map<std::string, BuiltinSig> table = [] {
        std::map<std::string, BuiltinSig> t;
        auto io = [&](std::initializer_list<const char*> names) {
            for (const char* n : names) t[n] = BuiltinSig{{"IO"}, {}, false};
        };
        io({"display", "newline", "write", "write-string", "write-char", "write-u8",
            "write-bytevector", "write-shared", "write-simple", "print", "printf", "eprintf",
            "read", "read-line", "read-char", "read-string", "read-u8", "peek-char", "peek-u8",
            "char-ready?", "u8-ready?", "flush-output-port", "load", "exit", "emergency-exit",
            "system", "getenv", "get-environment-variable", "get-environment-variables",
            "command-line", "sleep", "sleep-ms", "make-pipe", "make-event-loop",
            "current-time-ms", "current-time-ns", "current-timestamp", "current-jiffy",
            "current-second", "random", "random-integer", "random-real", "delete-file",
            "file-exists?", "call-with-output-file", "call-with-input-file",
            "with-output-to-file", "with-input-from-file", "format", "error-output"});
        auto exn = [&](std::initializer_list<const char*> names) {
            for (const char* n : names) t[n] = BuiltinSig{{"Exception"}, {}, false};
        };
        exn({"raise", "raise-continuable", "error", "assert", "assertion-violation",
             "syntax-error"});
        auto ho = [&](const char* n, std::vector<int> applies) {
            t[n] = BuiltinSig{{}, std::move(applies), false};
        };
        for (const char* n : {"map", "for-each", "filter", "reduce", "fold", "fold-left",
                              "fold-right", "apply", "vector-map", "vector-for-each",
                              "string-map", "string-for-each", "find", "find-tail", "any",
                              "every", "count", "partition", "remove", "filter-map",
                              "append-map", "list-index", "delete-duplicates", "list-sort",
                              "vector-sort", "tabulate", "list-tabulate", "make-initialized-vector",
                              "vector-count", "assq-ref"})
            ho(n, {0});
        ho("sort", {1});
        ho("assoc", {2});
        ho("member", {2});
        ho("call-with-values", {0, 1});
        ho("with-exception-handler", {0, 1});
        ho("hash-table-walk", {1});
        ho("hash-table-for-each", {1});
        ho("hash-table-update!", {2, 3});
        t["hash-table-update!"].labels = {"State"};
        t["force"] = BuiltinSig{{}, {}, true};
        t["eval"] = BuiltinSig{{}, {}, true};
        return t;
    }();
    return table;
}

bool hasPrefix(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

/** The registry entry for a name the program does not define. */
BuiltinSig builtinSignature(const std::string& name) {
    const auto& t = registry();
    auto it = t.find(name);
    if (it != t.end()) return it->second;
    BuiltinSig sig;
    static const char* io_prefixes[] = {
        "display", "write", "print", "read-", "open-", "close-", "file-", "directory-",
        "http-", "socket-", "tcp-", "udp-", "net-", "fd-", "event-loop", "tls-", "process-",
        "random", "current-time", "current-output", "current-input", "current-error",
        "log-", "agent-", "subprocess-", "url-fetch", "path-", "env-"};
    for (const char* p : io_prefixes)
        if (hasPrefix(name, p)) { sig.labels.push_back("IO"); break; }
    if (!name.empty() && name.back() == '!') sig.labels.push_back("State");
    return sig;
}

/* ── programs ─────────────────────────────────────────────────────────────── */

struct Operation {
    std::string effect;
    std::string name;
};

struct Procedure {
    std::string name;
    const eshkol_ast_t* node = nullptr;            // the define
    std::vector<std::string> params;
    std::string rest;
    const eshkol_ast_t* body = nullptr;             // body (function define) or lambda
    bool body_is_lambda = false;
    Row latent;
};

struct Declaration {
    std::string name;
    std::set<std::string> labels;
    bool open = false;
    const eshkol_ast_t* node = nullptr;
};

const char* varName(const eshkol_ast_t* e) {
    return (e && e->type == ESHKOL_VAR && e->variable.id) ? e->variable.id : nullptr;
}

std::string sourceName(const char* name) {
    if (!name) return {};
    return std::string(name, eshkol_syntax_base_length(name));
}

std::string symbolText(const eshkol_ast_t* d) {
    std::string raw(d->str_val.ptr, d->str_val.size);
    return sourceName(raw.c_str());
}

bool isOp(const eshkol_ast_t* e, eshkol_op_t op) {
    return e && e->type == ESHKOL_OP && e->operation.op == op;
}

const eshkol_ast_t* callFunc(const eshkol_ast_t* e) {
    return isOp(e, ESHKOL_CALL_OP) ? e->operation.call_op.func : nullptr;
}

bool isCallTo(const eshkol_ast_t* e, const char* name) {
    const char* f = varName(callFunc(e));
    return f && sourceName(f) == name;
}

/** The symbol in `(quote sym)`, or empty. */
std::string quotedSymbol(const eshkol_ast_t* e) {
    if (!isOp(e, ESHKOL_QUOTE_OP) || e->operation.call_op.num_vars < 1) return {};
    const char* v = varName(&e->operation.call_op.variables[0]);
    if (v) return sourceName(v);
    const eshkol_ast_t* d = &e->operation.call_op.variables[0];
    if (d->type == ESHKOL_SYMBOL && d->str_val.ptr) return symbolText(d);
    return {};
}

/** Symbols of a quoted datum list `(a b . c)`; the dotted tail goes to *tail. */
void datumSymbols(const eshkol_ast_t* d, std::vector<std::string>& out, std::string* tail) {
    if (!d) return;
    if (const char* v = varName(d)) { if (tail) *tail = sourceName(v); return; }
    if (d->type == ESHKOL_SYMBOL && d->str_val.ptr) { if (tail) *tail = symbolText(d); return; }
    if (d->type == ESHKOL_CONS) {
        const eshkol_ast_t* car = d->cons_cell.car;
        if (const char* v = varName(car)) out.push_back(sourceName(v));
        else if (car && car->type == ESHKOL_SYMBOL && car->str_val.ptr) out.push_back(symbolText(car));
        datumSymbols(d->cons_cell.cdr, out, tail);
        return;
    }
    if (isOp(d, ESHKOL_QUOTE_OP) && d->operation.call_op.num_vars >= 1) {
        datumSymbols(&d->operation.call_op.variables[0], out, tail);
        return;
    }
    // The reader spells a quoted list as (list e ...) and a dotted one as
    // (cons e tail).
    if (d->type == ESHKOL_OP && d->operation.op == ESHKOL_CALL_OP) {
        const std::string head = sourceName(varName(d->operation.call_op.func));
        const auto& c = d->operation.call_op;
        auto symbolOf = [](const eshkol_ast_t* x) -> std::string {
            if (const char* v = varName(x)) return sourceName(v);
            if (x && x->type == ESHKOL_SYMBOL && x->str_val.ptr) return symbolText(x);
            if (isOp(x, ESHKOL_QUOTE_OP) && x->operation.call_op.num_vars >= 1) {
                const eshkol_ast_t* q = &x->operation.call_op.variables[0];
                if (const char* v = varName(q)) return sourceName(v);
                if (q->type == ESHKOL_SYMBOL && q->str_val.ptr) return symbolText(q);
            }
            return {};
        };
        if (head == "list") {
            for (uint64_t i = 0; i < c.num_vars; ++i) {
                std::string sym = symbolOf(&c.variables[i]);
                if (!sym.empty()) out.push_back(sym);
            }
        } else if (head == "cons" && c.num_vars == 2) {
            std::string sym = symbolOf(&c.variables[0]);
            if (!sym.empty()) out.push_back(sym);
            std::string rest = symbolOf(&c.variables[1]);
            if (!rest.empty()) { if (tail) *tail = rest; }
            else datumSymbols(&c.variables[1], out, tail);
        } else {
            // a proper list read as an application: (a b c)
            if (!head.empty()) out.push_back(head);
            for (uint64_t i = 0; i < c.num_vars; ++i) {
                std::string sym = symbolOf(&c.variables[i]);
                if (!sym.empty()) out.push_back(sym);
            }
        }
    }
}

std::vector<std::string> quotedList(const eshkol_ast_t* e, std::string* tail) {
    std::vector<std::string> out;
    if (!isOp(e, ESHKOL_QUOTE_OP) || e->operation.call_op.num_vars < 1) {
        if (const char* v = varName(e)) { if (tail) *tail = sourceName(v); }
        return out;
    }
    const eshkol_ast_t* d = &e->operation.call_op.variables[0];
    datumSymbols(d, out, tail);
    return out;
}

/* ── the analysis ─────────────────────────────────────────────────────────── */

struct Binding {
    enum Kind { Param, Lambda, Unknown, Resumption, Loop } kind = Unknown;
    int index = -1;             // Param: position in the top-level procedure
    Row latent;                 // Lambda
};

class Analyzer {
public:
    std::map<std::string, Operation> ops;               // operation procedure -> effect
    std::map<std::string, std::set<std::string>> effect_ops;  // effect -> operation names
    std::map<std::string, int> opArity;                       // operation -> arity (-1 variadic)
    std::map<std::string, Procedure> procs;
    std::set<std::string> globals;                      // every top-level definition
    std::vector<Declaration> decls;

    void collect(const eshkol_ast_t* e) {
        if (!e || e->type != ESHKOL_OP) return;
        const auto& op = e->operation;
        if (op.op == ESHKOL_SEQUENCE_OP) {
            for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                collect(&op.sequence_op.expressions[i]);
            return;
        }
        if (op.op == ESHKOL_DEFINE_OP && op.define_op.name) {
            const std::string name = sourceName(op.define_op.name);
            globals.insert(name);
            const eshkol_ast_t* v = op.define_op.value;
            if (!op.define_op.is_function && isCallTo(v, "%effect-operation") &&
                v->operation.call_op.num_vars >= 2) {
                const char* eff = varName(&v->operation.call_op.variables[0]);
                if (eff) {
                    ops[name] = Operation{sourceName(eff), quotedSymbol(&v->operation.call_op.variables[1])};
                    effect_ops[sourceName(eff)].insert(name);
                    if (v->operation.call_op.num_vars >= 3) {
                        std::string tail;
                        auto formals = quotedList(&v->operation.call_op.variables[2], &tail);
                        opArity[name] = tail.empty() ? static_cast<int>(formals.size()) : -1;
                    }
                }
                return;
            }
            Procedure p;
            p.name = name;
            p.node = e;
            if (op.define_op.is_function) {
                for (uint64_t i = 0; i < op.define_op.num_params; ++i) {
                    const char* pn = varName(&op.define_op.parameters[i]);
                    p.params.push_back(pn ? sourceName(pn) : std::string());
                }
                if (op.define_op.rest_param) p.rest = sourceName(op.define_op.rest_param);
                p.body = v;
                procs[name] = p;
            } else if (isOp(v, ESHKOL_LAMBDA_OP)) {
                const auto& l = v->operation.lambda_op;
                for (uint64_t i = 0; i < l.num_params; ++i) {
                    const char* pn = varName(&l.parameters[i]);
                    p.params.push_back(pn ? sourceName(pn) : std::string());
                }
                if (l.rest_param) p.rest = sourceName(l.rest_param);
                p.body = l.body;
                procs[name] = p;
            }
            return;
        }
        if (isCallTo(e, "%effect-declare") && op.call_op.num_vars == 2) {
            Declaration d;
            d.name = quotedSymbol(&op.call_op.variables[0]);
            std::string tail;
            const eshkol_ast_t* rowe = &op.call_op.variables[1];
            std::vector<std::string> labels = quotedList(rowe, &tail);
            if (labels.empty() && tail.empty()) {
                // a bare symbol: Pure, or a row variable (open, no fixed labels)
                std::string sym = quotedSymbol(rowe);
                if (!sym.empty() && sym != "Pure") tail = sym;
            }
            if (!tail.empty() && tail != "Pure") d.open = true;
            d.labels.insert(labels.begin(), labels.end());
            d.node = e;
            decls.push_back(d);
        }
    }

    /** Infer every procedure's latent row (least fixpoint). */
    void infer() {
        for (int pass = 0; pass < 64; ++pass) {
            bool changed = false;
            for (auto& [name, p] : procs) {
                Row r = analyzeProcedure(p);
                if (!(r == p.latent)) { p.latent = r; changed = true; }
            }
            if (!changed) return;
        }
    }

    Row analyzeProcedure(const Procedure& p) {
        std::vector<std::map<std::string, Binding>> scopes(1);
        for (size_t i = 0; i < p.params.size(); ++i) {
            Binding b;
            b.kind = Binding::Param;
            b.index = static_cast<int>(i);
            scopes[0][p.params[i]] = b;
        }
        if (!p.rest.empty()) scopes[0][p.rest] = Binding{};
        scopes_ = &scopes;
        lambda_floor_ = 0;
        Row r = walk(p.body);
        scopes_ = nullptr;
        return r;
    }

private:
    std::vector<std::map<std::string, Binding>>* scopes_ = nullptr;
    size_t lambda_floor_ = 0;     // first scope of the innermost procedure

    const Binding* lookup(const std::string& name, size_t* depth = nullptr) const {
        for (size_t i = scopes_->size(); i-- > 0;) {
            auto it = (*scopes_)[i].find(name);
            if (it != (*scopes_)[i].end()) { if (depth) *depth = i; return &it->second; }
        }
        return nullptr;
    }

    static Witness at(const eshkol_ast_t* e, const std::string& what) {
        Witness w;
        w.what = what;
        if (e) { w.line = e->line; w.column = e->column; }
        return w;
    }

    Row walkAll(const eshkol_ast_t* xs, uint64_t n) {
        Row r;
        for (uint64_t i = 0; i < n; ++i) r.join(walk(&xs[i]));
        return r;
    }

    /** The latent row of the procedure @p e evaluates to (when called). */
    Row latentOf(const eshkol_ast_t* e) {
        Row r;
        if (!e) return r;
        if (isOp(e, ESHKOL_LAMBDA_OP)) return analyzeLambda(e, -1);
        if (const char* v = varName(e)) {
            const std::string name = sourceName(v);
            if (const Binding* b = lookup(name)) {
                switch (b->kind) {
                    case Binding::Param: r.addTail("param:" + std::to_string(b->index), {}, at(e, name)); break;
                    case Binding::Lambda: r = b->latent; break;
                    case Binding::Resumption: case Binding::Loop: break;
                    case Binding::Unknown: r.addTail(kUnknownTail, {}, at(e, "calls " + name)); break;
                }
                return r;
            }
            auto op = ops.find(name);
            if (op != ops.end()) { r.addLabel(op->second.effect, at(e, name)); return r; }
            auto p = procs.find(name);
            if (p != procs.end()) {
                // passed as a value: its own procedure arguments are unknown here
                for (const auto& [l, w] : p->second.latent.labels) r.addLabel(l, w);
                for (const auto& [t, m] : p->second.latent.tails)
                    r.addTail(kUnknownTail, m, at(e, "calls " + name));
                return r;
            }
            if (globals.count(name)) { r.addTail(kUnknownTail, {}, at(e, "calls " + name)); return r; }
            BuiltinSig sig = builtinSignature(name);
            for (const auto& l : sig.labels) r.addLabel(l, at(e, name));
            if (sig.unknown || !sig.applies.empty()) r.addTail(kUnknownTail, {}, at(e, name));
            return r;
        }
        r.addTail(kUnknownTail, {}, at(e, "calls a computed procedure"));
        return r;
    }

    /** Latent row of a lambda literal; resumption_param >= 0 binds that
     *  parameter as a resumption. */
    Row analyzeLambda(const eshkol_ast_t* e, int resumption_param) {
        const auto& l = e->operation.lambda_op;
        std::map<std::string, Binding> frame;
        for (uint64_t i = 0; i < l.num_params; ++i) {
            const char* pn = varName(&l.parameters[i]);
            if (!pn) continue;
            Binding b;
            b.kind = (static_cast<int>(i) == resumption_param) ? Binding::Resumption : Binding::Unknown;
            frame[sourceName(pn)] = b;
        }
        if (l.rest_param) frame[sourceName(l.rest_param)] = Binding{};
        scopes_->push_back(frame);
        const size_t saved_floor = lambda_floor_;
        lambda_floor_ = scopes_->size() - 1;
        Row r = walk(l.body);
        lambda_floor_ = saved_floor;
        scopes_->pop_back();
        return r;
    }

    Row callEffect(const eshkol_ast_t* call) {
        const auto& c = call->operation.call_op;
        const eshkol_ast_t* f = c.func;
        Row r;
        const char* v = varName(f);
        if (!v) {
            if (isOp(f, ESHKOL_LAMBDA_OP)) return analyzeLambda(f, -1);
            r.join(walk(f));
            r.addTail(kUnknownTail, {}, at(call, "calls a computed procedure"));
            return r;
        }
        const std::string name = sourceName(v);
        if (const Binding* b = lookup(name)) {
            switch (b->kind) {
                case Binding::Param: r.addTail("param:" + std::to_string(b->index), {}, at(call, name)); break;
                case Binding::Lambda: r = b->latent; break;
                case Binding::Resumption: case Binding::Loop: break;
                case Binding::Unknown: r.addTail(kUnknownTail, {}, at(call, "calls " + name)); break;
            }
            return r;
        }
        auto op = ops.find(name);
        if (op != ops.end()) { r.addLabel(op->second.effect, at(call, name)); return r; }
        if (name == "%effect-handle") return handleEffect(call);
        if (name == "%effect-declare" || name == "%effect-make" || name == "%effect-operation" ||
            name == "%effect-clause")
            return r;
        auto p = procs.find(name);
        if (p != procs.end()) {
            const Row& lat = p->second.latent;
            for (const auto& [l, w] : lat.labels) r.addLabel(l, w);
            for (const auto& [t, minus] : lat.tails) {
                if (hasPrefix(t, "param:")) {
                    size_t i = std::strtoul(t.c_str() + 6, nullptr, 10);
                    if (i < c.num_vars) r.join(latentOf(&c.variables[i]).without(minus));
                    continue;
                }
                auto w = lat.tail_witness.find(t);
                r.addTail(t, minus, w == lat.tail_witness.end() ? Witness{} : w->second);
            }
            return r;
        }
        if (globals.count(name)) { r.addTail(kUnknownTail, {}, at(call, "calls " + name)); return r; }
        BuiltinSig sig = builtinSignature(name);
        for (const auto& l : sig.labels) r.addLabel(l, at(call, name));
        for (int i : sig.applies)
            if (i >= 0 && static_cast<uint64_t>(i) < c.num_vars) r.join(latentOf(&c.variables[i]));
        if (sig.unknown) r.addTail(kUnknownTail, {}, at(call, name));
        return r;
    }

    /** (%effect-handle 'mode 'depth return (list clause...) (lambda () body)) */
    Row handleEffect(const eshkol_ast_t* call) {
        const auto& c = call->operation.call_op;
        Row r;
        if (c.num_vars != 5) { r.addTail(kUnknownTail, {}, at(call, "malformed handler")); return r; }
        const bool deep = quotedSymbol(&c.variables[1]) != "shallow";
        std::set<std::string> handled_ops;
        Row clauses;
        const eshkol_ast_t* list = &c.variables[3];
        if (isCallTo(list, "list")) {
            const auto& l = list->operation.call_op;
            for (uint64_t i = 0; i < l.num_vars; ++i) {
                const eshkol_ast_t* cl = &l.variables[i];
                if (!isCallTo(cl, "%effect-clause") || cl->operation.call_op.num_vars != 3) {
                    clauses.join(walk(cl));
                    continue;
                }
                const auto& cc = cl->operation.call_op;
                if (const char* opn = varName(&cc.variables[0])) handled_ops.insert(sourceName(opn));
                const std::string kind = quotedSymbol(&cc.variables[1]);
                bool takes_k = kind == "general" || kind == "general-tail";
                if (kind.empty() && isOp(&cc.variables[2], ESHKOL_LAMBDA_OP)) {
                    // unrecognised kind: k is first exactly when the clause takes
                    // one parameter more than the operation
                    auto o = opArity.find(sourceName(varName(&cc.variables[0])));
                    if (o != opArity.end() && o->second >= 0)
                        takes_k = cc.variables[2].operation.lambda_op.num_params ==
                                  static_cast<uint64_t>(o->second) + 1;
                }
                if (isOp(&cc.variables[2], ESHKOL_LAMBDA_OP))
                    clauses.join(analyzeLambda(&cc.variables[2], takes_k ? 0 : -1));
                else
                    clauses.join(latentOf(&cc.variables[2]));
            }
        } else {
            clauses.join(walk(list));
        }
        Row body = latentOf(&c.variables[4]);
        if (deep) {
            std::set<std::string> discharged;
            for (const auto& [eff, names] : effect_ops) {
                bool all = !names.empty();
                for (const auto& n : names) if (!handled_ops.count(n)) { all = false; break; }
                if (all) discharged.insert(eff);
            }
            body.discharge(discharged);
        }
        r.join(body);
        r.join(clauses);
        if (!varName(&c.variables[2]) || sourceName(varName(&c.variables[2])) != "%effect-identity")
            r.join(latentOf(&c.variables[2]));
        return r;
    }

    void bindLocal(const std::string& name, Binding b) { scopes_->back()[name] = b; }

    Row walkLet(const eshkol_ast_t* e) {
        const auto& l = e->operation.let_op;
        Row r;
        const bool rec = e->operation.op == ESHKOL_LETREC_OP || e->operation.op == ESHKOL_LETREC_STAR_OP;
        scopes_->push_back({});
        if (l.name) {
            Binding loop;
            loop.kind = Binding::Loop;
            bindLocal(sourceName(l.name), loop);
        }
        auto bindingParts = [](const eshkol_ast_t& b, const char** var, const eshkol_ast_t** val) {
            *var = nullptr; *val = nullptr;
            if (b.type == ESHKOL_CONS) { *var = varName(b.cons_cell.car); *val = b.cons_cell.cdr; }
            else if (isOp(&b, ESHKOL_DEFINE_OP)) { *var = b.operation.define_op.name; *val = b.operation.define_op.value; }
        };
        if (rec) {
            for (uint64_t i = 0; i < l.num_bindings; ++i) {
                const char* var; const eshkol_ast_t* val;
                bindingParts(l.bindings[i], &var, &val);
                if (var) { Binding b; b.kind = isOp(val, ESHKOL_LAMBDA_OP) ? Binding::Lambda : Binding::Unknown; bindLocal(sourceName(var), b); }
            }
            for (int pass = 0; pass < 2; ++pass)
                for (uint64_t i = 0; i < l.num_bindings; ++i) {
                    const char* var; const eshkol_ast_t* val;
                    bindingParts(l.bindings[i], &var, &val);
                    if (var && isOp(val, ESHKOL_LAMBDA_OP))
                        scopes_->back()[sourceName(var)].latent = analyzeLambda(val, -1);
                }
        }
        for (uint64_t i = 0; i < l.num_bindings; ++i) {
            const char* var; const eshkol_ast_t* val;
            bindingParts(l.bindings[i], &var, &val);
            if (!val) { r.join(walk(&l.bindings[i])); continue; }
            r.join(walk(val));
            if (var && !rec) {
                Binding b;
                if (isOp(val, ESHKOL_LAMBDA_OP)) { b.kind = Binding::Lambda; b.latent = analyzeLambda(val, -1); }
                if (l.name) b.kind = Binding::Unknown, b.latent = Row{};
                bindLocal(sourceName(var), b);
            }
        }
        r.join(walk(l.body));
        scopes_->pop_back();
        return r;
    }

    /** A sequence of expressions stored in the call layout (func included). */
    Row walkContainer(const eshkol_ast_t* c) {
        if (!isOp(c, ESHKOL_CALL_OP)) return walk(c);
        Row r = walk(c->operation.call_op.func);
        r.join(walkAll(c->operation.call_op.variables, c->operation.call_op.num_vars));
        return r;
    }

    /** A cond/case/guard clause: (test body ...) in the call layout. */
    Row walkClause(const eshkol_ast_t* c, bool has_test) {
        if (!isOp(c, ESHKOL_CALL_OP)) return walk(c);
        Row r;
        if (has_test) r.join(walk(c->operation.call_op.func));
        r.join(walkAll(c->operation.call_op.variables, c->operation.call_op.num_vars));
        return r;
    }

    /** (do ((var init step) ...) (test result ...) body ...) */
    Row walkDo(const eshkol_ast_t* e) {
        const auto& op = e->operation;
        Row r;
        const eshkol_ast_t* main_cons = op.call_op.func;
        if (!main_cons || main_cons->type != ESHKOL_CONS) {
            r.addTail(kUnknownTail, {}, at(e, "a do form the effect analysis does not model"));
            return r;
        }
        scopes_->push_back({});
        const eshkol_ast_t* bindings = main_cons->cons_cell.car;
        if (isOp(bindings, ESHKOL_CALL_OP)) {
            for (uint64_t i = 0; i < bindings->operation.call_op.num_vars; ++i) {
                const eshkol_ast_t* b = &bindings->operation.call_op.variables[i];
                if (b->type != ESHKOL_CONS) continue;
                if (const char* v = varName(b->cons_cell.car)) bindLocal(sourceName(v), Binding{});
                r.join(walk(b->cons_cell.cdr));      // init and step
            }
        }
        const eshkol_ast_t* test_clause = main_cons->cons_cell.cdr;
        if (test_clause && test_clause->type == ESHKOL_CONS) {
            r.join(walk(test_clause->cons_cell.car));             // test
            r.join(walkContainer(test_clause->cons_cell.cdr));    // results
        } else {
            r.join(walk(test_clause));
        }
        r.join(walkAll(op.call_op.variables, op.call_op.num_vars));
        scopes_->pop_back();
        return r;
    }

    static bool catchAllGuard(const eshkol_operations_t& op) {
        for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i) {
            const eshkol_ast_t* clause = &op.guard_op.clauses[i];
            const eshkol_ast_t* test = clause->type == ESHKOL_OP ? clause->operation.call_op.func : nullptr;
            if (const char* v = varName(test)) if (sourceName(v) == "else") return true;
            if (test && test->type == ESHKOL_BOOL && test->int64_val) return true;
        }
        return false;
    }

    Row walk(const eshkol_ast_t* e) {
        Row r;
        if (!e) return r;
        if (e->type == ESHKOL_CONS) {
            r.join(walk(e->cons_cell.car));
            r.join(walk(e->cons_cell.cdr));
            return r;
        }
        if (e->type != ESHKOL_OP) return r;
        const auto& op = e->operation;
        switch (op.op) {
            case ESHKOL_QUOTE_OP:
            case ESHKOL_LAMBDA_OP:
            case ESHKOL_DEFINE_TYPE_OP:
            case ESHKOL_TYPE_ANNOTATION_OP:
            case ESHKOL_FORALL_OP:
            case ESHKOL_REQUIRE_OP:
            case ESHKOL_PROVIDE_OP:
            case ESHKOL_IMPORT_OP:
            case ESHKOL_DEFINE_SYNTAX_OP:
            case ESHKOL_EXTERN_OP:
            case ESHKOL_EXTERN_VAR_OP:
                return r;
            case ESHKOL_SEQUENCE_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
                return walkAll(op.sequence_op.expressions, op.sequence_op.num_expressions);
            case ESHKOL_CALL_OP: {
                r.join(walk(op.call_op.func));
                r.join(walkAll(op.call_op.variables, op.call_op.num_vars));
                r.join(callEffect(e));
                return r;
            }
            case ESHKOL_COND_OP:
                // clauses share the call layout: func = test, variables = body
                r.join(walk(op.call_op.func));
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) r.join(walkClause(&op.call_op.variables[i], true));
                return r;
            case ESHKOL_CASE_OP:
                // clauses are (datums . body); body is a container of expressions
                r.join(walk(op.call_op.func));
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                    const eshkol_ast_t* c = &op.call_op.variables[i];
                    if (c->type == ESHKOL_CONS) r.join(walkContainer(c->cons_cell.cdr));
                    else r.join(walkClause(c, false));
                }
                return r;
            case ESHKOL_DO_OP:
                return walkDo(e);
            case ESHKOL_IF_OP:
            case ESHKOL_WHEN_OP: case ESHKOL_UNLESS_OP:
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP: case ESHKOL_MUL_OP: case ESHKOL_DIV_OP:
            case ESHKOL_QUASIQUOTE_OP: case ESHKOL_UNQUOTE_OP: case ESHKOL_UNQUOTE_SPLICING_OP:
                r.join(walk(op.call_op.func));
                r.join(walkAll(op.call_op.variables, op.call_op.num_vars));
                return r;
            case ESHKOL_DEFINE_OP: {
                const std::string name = sourceName(op.define_op.name);
                Binding b;
                if (op.define_op.is_function || isOp(op.define_op.value, ESHKOL_LAMBDA_OP)) {
                    b.kind = Binding::Lambda;
                    bindLocal(name, b);          // visible to its own body
                    Row lat;
                    if (op.define_op.is_function) {
                        std::map<std::string, Binding> frame;
                        for (uint64_t i = 0; i < op.define_op.num_params; ++i)
                            if (const char* pn = varName(&op.define_op.parameters[i])) frame[sourceName(pn)] = Binding{};
                        if (op.define_op.rest_param) frame[sourceName(op.define_op.rest_param)] = Binding{};
                        scopes_->push_back(frame);
                        const size_t saved = lambda_floor_;
                        lambda_floor_ = scopes_->size() - 1;
                        lat = walk(op.define_op.value);
                        lambda_floor_ = saved;
                        scopes_->pop_back();
                    } else {
                        lat = analyzeLambda(op.define_op.value, -1);
                    }
                    b.latent = lat;
                    bindLocal(name, b);
                    return r;
                }
                r = walk(op.define_op.value);
                bindLocal(name, Binding{});
                return r;
            }
            case ESHKOL_LET_OP: case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP: case ESHKOL_LETREC_STAR_OP:
                return walkLet(e);
            case ESHKOL_SET_OP: {
                r = walk(op.set_op.value);
                // Mutating a variable of the enclosing top-level procedure's own
                // activation is invisible to its callers; anything else is State.
                const std::string name = sourceName(op.set_op.name);
                if (!lookup(name)) r.addLabel("State", at(e, "set! " + name));
                return r;
            }
            case ESHKOL_GUARD_OP: {
                Row body = walkAll(op.guard_op.body, op.guard_op.num_body_exprs);
                if (catchAllGuard(op)) body.discharge({"Exception"});
                r.join(body);
                scopes_->push_back({});
                if (op.guard_op.var_name) bindLocal(sourceName(op.guard_op.var_name), Binding{});
                for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i)
                    r.join(walkClause(&op.guard_op.clauses[i], true));
                scopes_->pop_back();
                return r;
            }
            case ESHKOL_RAISE_OP:
                r = walk(op.raise_op.exception);
                r.addLabel("Exception", at(e, "raise"));
                return r;
            case ESHKOL_CALL_CC_OP:
                r = walk(op.call_cc_op.proc);
                r.addLabel("Control", at(e, "call/cc"));
                if (isOp(op.call_cc_op.proc, ESHKOL_LAMBDA_OP)) r.join(analyzeLambda(op.call_cc_op.proc, 0));
                else r.join(latentOf(op.call_cc_op.proc));
                return r;
            case ESHKOL_DYNAMIC_WIND_OP:
                r.join(walk(op.dynamic_wind_op.before));
                r.join(walk(op.dynamic_wind_op.thunk));
                r.join(walk(op.dynamic_wind_op.after));
                r.join(latentOf(op.dynamic_wind_op.before));
                r.join(latentOf(op.dynamic_wind_op.thunk));
                r.join(latentOf(op.dynamic_wind_op.after));
                return r;
            case ESHKOL_WITH_REGION_OP:
                return walkAll(op.with_region_op.body, op.with_region_op.num_body_exprs);
            case ESHKOL_VALUES_OP:
                return walkAll(op.values_op.expressions, op.values_op.num_values);
            case ESHKOL_CALL_WITH_VALUES_OP:
                r.join(walk(op.call_with_values_op.producer));
                r.join(walk(op.call_with_values_op.consumer));
                r.join(latentOf(op.call_with_values_op.producer));
                r.join(latentOf(op.call_with_values_op.consumer));
                return r;
            case ESHKOL_LET_VALUES_OP: case ESHKOL_LET_STAR_VALUES_OP: {
                const auto& lv = op.let_values_op;
                r = walkAll(lv.producers, lv.num_bindings);
                scopes_->push_back({});
                for (uint64_t i = 0; i < lv.num_bindings; ++i)
                    for (uint64_t j = 0; lv.binding_vars && j < lv.binding_var_counts[i]; ++j)
                        bindLocal(sourceName(lv.binding_vars[i][j]), Binding{});
                r.join(walk(lv.body));
                scopes_->pop_back();
                return r;
            }
            case ESHKOL_MATCH_OP: {
                r = walk(op.match_op.expr);
                for (uint64_t i = 0; i < op.match_op.num_clauses; ++i) {
                    r.join(walk(op.match_op.clauses[i].guard));
                    r.join(walk(op.match_op.clauses[i].body));
                }
                return r;
            }
            case ESHKOL_THE_OP:
                return walk(op.the_op.expr);
            default:
                // A form this analysis does not model: never Pure (ADR-0004).
                r.addTail(kUnknownTail, {}, at(e, "a form the effect analysis does not model"));
                return r;
        }
    }
};

bool dumpEnabled() {
    const char* v = std::getenv("ESHKOL_DUMP_EFFECTS");
    return v && *v && std::strcmp(v, "0") != 0;
}

} // namespace

size_t checkProgram(const eshkol_ast_t* asts, size_t count, eshkol::hott::TypeChecker& checker) {
    Analyzer a;
    for (size_t i = 0; i < count; ++i) a.collect(&asts[i]);
    if (a.procs.empty() && a.decls.empty()) return 0;
    a.infer();

    if (dumpEnabled()) {
        const bool witnesses = std::getenv("ESHKOL_DUMP_EFFECTS")[0] == '2';
        for (const auto& [name, p] : a.procs) {
            std::fprintf(stderr, "effects: %s : %s\n", name.c_str(), p.latent.str().c_str());
            if (!witnesses) continue;
            for (const auto& [l, w] : p.latent.labels)
                std::fprintf(stderr, "  %s from %s (line %u)\n", l.c_str(), w.what.c_str(), w.line);
            for (const auto& [t, w] : p.latent.tail_witness)
                std::fprintf(stderr, "  %s from %s (line %u)\n", t.c_str(), w.what.c_str(), w.line);
        }
    }

    size_t violations = 0;
    for (const auto& d : a.decls) {
        auto p = a.procs.find(d.name);
        if (p == a.procs.end()) {
            checker.reportTypeIssue("effect declaration names '" + d.name +
                                    "', which is not a procedure defined in this program", d.node);
            ++violations;
            continue;
        }
        const Row& row = p->second.latent;
        std::string declared = "(";
        bool first = true;
        for (const auto& l : d.labels) { declared += (first ? "" : " ") + l; first = false; }
        if (d.open) declared += first ? ". e" : " . e";
        declared += ")";
        if (d.labels.empty() && !d.open) declared = "Pure";
        for (const auto& [l, w] : row.labels) {
            if (d.labels.count(l)) continue;
            std::string msg = "'" + d.name + "' is declared " + declared + " but may perform " + l +
                              " (" + w.what;
            if (w.line) msg += " at line " + std::to_string(w.line);
            msg += ")";
            checker.reportTypeIssue(msg, p->second.node);
            ++violations;
        }
        for (const auto& [t, minus] : row.tails) {
            const bool param = hasPrefix(t, "param:");
            if (param && d.open) continue;
            auto w = row.tail_witness.find(t);
            std::string what = w == row.tail_witness.end() ? std::string() : w->second.what;
            std::string msg = "'" + d.name + "' is declared " + declared + " but its effects are not closed: " +
                              (param ? "it calls its procedure argument " + what +
                                           " (declare an open row, e.g. (L ... . e))"
                                     : what + " has effects the checker cannot see");
            if (w != row.tail_witness.end() && w->second.line)
                msg += " at line " + std::to_string(w->second.line);
            checker.reportTypeIssue(msg, p->second.node);
            ++violations;
        }
    }
    return violations;
}

} // namespace eshkol::effects
