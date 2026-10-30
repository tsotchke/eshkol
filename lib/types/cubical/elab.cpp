/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Bidirectional elaboration for the cubical kernel (ADR-0043): S-expressions
 * to core terms, checking types as it goes.
 *
 *   check(e, T)  e is checked against the type T (lambdas, pairs, splits,
 *                glue elements, constructors of parameterized types);
 *   infer(e)     the type of e is computed (variables, applications, type
 *                formers, Kan operations, ascriptions).
 *
 * Every binder gets an internal name derived from its value-level identity
 * (`v#<id>` for variables, `i#<atom>` for dimensions), so a value can be read
 * back to a term that means the same thing in the elaboration context. That
 * read-back provides the motive of a split checked against a type value and
 * the endpoints stored in path applications.
 *
 * Soundness guards (the kernel is a logic, not only a calculator): universes
 * are predicative with explicit level arguments; data types must be strictly
 * positive; recursion is structural (a definition may call itself only on a
 * variable bound by a constructor pattern of its own split); a path
 * constructor's boundary is built from constructors and the constructor's own
 * arguments, the class for which transport commutes with constructors (CHM 2018).
 */

#include "cubical_internal.h"
#include "elab.h"

#include <algorithm>
#include <sstream>

namespace eshkol::hott::cubical {

namespace {

std::shared_ptr<Term> mk(Term::K k, const Sx* at = nullptr) {
    auto t = std::make_shared<Term>();
    t->k = k;
    if (at) { t->line = at->line; t->column = at->column; }
    return t;
}

IExprP iz() { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::Zero; return e; }
IExprP io() { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::One; return e; }
IExprP ivar(const std::string& n) { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::Var; e->name = n; return e; }
IExprP ineg(IExprP a) { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::Neg; e->a = std::move(a); return e; }
IExprP imin(IExprP a, IExprP b) { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::Min; e->a = std::move(a); e->b = std::move(b); return e; }
IExprP imax(IExprP a, IExprP b) { auto e = std::make_shared<IExpr>(); e->k = IExpr::K::Max; e->a = std::move(a); e->b = std::move(b); return e; }

LExprP lconst(uint32_t n) { auto e = std::make_shared<LExpr>(); e->k = LExpr::K::Const; e->n = n; return e; }
LExprP lvar(const std::string& n) { auto e = std::make_shared<LExpr>(); e->k = LExpr::K::Var; e->name = n; return e; }
LExprP lsuc(LExprP a) { auto e = std::make_shared<LExpr>(); e->k = LExpr::K::Suc; e->a = std::move(a); return e; }
LExprP lmax(LExprP a, LExprP b) { auto e = std::make_shared<LExpr>(); e->k = LExpr::K::Max; e->a = std::move(a); e->b = std::move(b); return e; }

std::string dimInternal(Dim d) { return "i#" + std::to_string(d); }
std::string varInternal(uint64_t id) { return "v#" + std::to_string(id); }

[[noreturn]] void fail(const Sx& at, const std::string& msg) {
    throw KernelError(msg, at.line, at.column);
}


}  // namespace

// ---------------------------------------------------------------------------
// Read-back of values to terms
// ---------------------------------------------------------------------------

LExprP levelTerm(const Level& l) {
    LExprP out;
    if (l.c > 0 || l.vars.empty()) out = lconst(l.c);
    for (auto& [v, k] : l.vars) {
        LExprP e = lvar(v);
        for (uint32_t i = 0; i < k; ++i) e = lsuc(e);
        out = out ? lmax(out, e) : e;
    }
    return out;
}

IExprP formulaTerm(const Formula& f) {
    if (f.isZero()) return iz();
    IExprP out;
    for (auto& c : f.dnf) {
        IExprP conj;
        for (auto& l : c) {
            IExprP a = ivar(dimInternal(l.d));
            if (!l.pos) a = ineg(a);
            conj = conj ? imin(conj, a) : a;
        }
        if (!conj) conj = io();
        out = out ? imax(out, conj) : conj;
    }
    return out;
}

static IExprP faceTerm(const Face& f) { return formulaTerm(faceFormula(f)); }

TermP readback(const V& v);

static TermP readbackLine(const V& line, std::string& binder) {
    Dim k = freshDim();
    binder = dimInternal(k);
    return readback(lineAt(line, Formula::atom(k)));
}

static std::vector<TSysClause> readbackSys(const VSys& s, bool lines, const std::string& binder, Dim k) {
    std::vector<TSysClause> out;
    for (auto& c : s) {
        TSysClause tc;
        tc.face = faceTerm(c.face);
        tc.t = lines ? readback(lineAt(c.v, Formula::atom(k))) : readback(c.v);
        if (c.v2) tc.t2 = readback(c.v2);
        out.push_back(tc);
    }
    (void)binder;
    return out;
}

TermP readback(const V& v) {
    switch (v->k) {
        case Val::K::U: {
            auto t = mk(Term::K::U);
            t->level = levelTerm(v->level);
            return t;
        }
        case Val::K::Pi:
        case Val::K::Sigma: {
            V x = mkVar(v->name.empty() ? "x" : v->name, v->a);
            auto t = mk(v->k == Val::K::Pi ? Term::K::Pi : Term::K::Sigma);
            t->name = varInternal(x->id);
            t->a = readback(v->a);
            t->b = readback(piCod(v, x));
            return t;
        }
        case Val::K::Lam: {
            V x = mkVar(v->name.empty() ? "x" : v->name, nullptr);
            auto t = mk(Term::K::Lam);
            t->name = varInternal(x->id);
            t->a = readback(app(v, x));
            return t;
        }
        case Val::K::Pair: {
            auto t = mk(Term::K::Pair);
            t->a = readback(v->a);
            t->b = readback(v->b);
            return t;
        }
        case Val::K::PathP: {
            auto line = mk(Term::K::PLam);
            line->a = readbackLine(v->a, line->name);
            auto t = mk(Term::K::PathP);
            t->a = line;
            t->b = readback(v->b);
            t->c = readback(v->c);
            return t;
        }
        case Val::K::PLam: {
            auto t = mk(Term::K::PLam);
            t->a = readbackLine(v, t->name);
            return t;
        }
        case Val::K::Data: {
            auto t = mk(Term::K::Data);
            t->data = v->data;
            for (auto& l : v->levels) t->levels.push_back(levelTerm(l));
            for (auto& p : v->params) t->params.push_back(readback(p));
            return t;
        }
        case Val::K::Con: {
            auto t = mk(Term::K::Con);
            t->data = v->data;
            t->con = v->con;
            for (auto& l : v->levels) t->levels.push_back(levelTerm(l));
            for (auto& p : v->params) t->params.push_back(readback(p));
            for (auto& p : v->args) t->args.push_back(readback(p));
            for (auto& d : v->dims) t->dims.push_back(formulaTerm(d));
            return t;
        }
        case Val::K::Split: {
            if (!v->name.empty() && v->name[0] == '@') {
                auto t = mk(Term::K::Global);
                t->name = v->name.substr(1);
                for (auto& l : v->levels) t->levels.push_back(levelTerm(l));
                return t;
            }
            // A local split: re-bind its environment around the split term.
            TermP body = v->split;
            std::vector<const Env*> chain;
            for (const Env* e = v->clo.env.get(); e; e = e->next.get()) chain.push_back(e);
            for (const Env* e : chain) {
                if (e->kind == Env::Kind::Value) {
                    auto let = mk(Term::K::Let);
                    let->name = e->name;
                    let->a = readback(e->v);
                    let->b = body;
                    body = let;
                } else {
                    throw KernelError("a split under a dimension or level binder cannot appear in a type; "
                                      "define it at top level");
                }
            }
            return body;
        }
        case Val::K::Glue: {
            auto t = mk(Term::K::Glue);
            t->a = readback(v->a);
            t->sys = readbackSys(v->sys, false, "", 0);
            return t;
        }
        case Val::K::GlueElem: {
            auto t = mk(Term::K::GlueElem);
            t->a = readback(v->a);
            t->sys = readbackSys(v->sys, false, "", 0);
            return t;
        }
        case Val::K::HComp: {
            auto t = mk(Term::K::HComp);
            Dim k = freshDim();
            t->name = dimInternal(k);
            t->a = readback(v->a);
            t->b = readback(v->b);
            t->sys = readbackSys(v->sys, true, t->name, k);
            return t;
        }
        case Val::K::Transp: {
            auto t = mk(Term::K::Transp);
            t->a = readbackLine(v->a, t->name);
            t->r = formulaTerm(v->f);
            t->b = readback(v->b);
            return t;
        }
        case Val::K::Var: {
            auto t = mk(Term::K::Var);
            t->name = varInternal(v->id);
            return t;
        }
        case Val::K::App:
        case Val::K::SplitApp: {
            auto t = mk(Term::K::App);
            t->a = readback(v->a);
            t->b = readback(v->b);
            return t;
        }
        case Val::K::Fst:
        case Val::K::Snd: {
            auto t = mk(v->k == Val::K::Fst ? Term::K::Fst : Term::K::Snd);
            t->a = readback(v->a);
            return t;
        }
        case Val::K::PApp: {
            auto t = mk(Term::K::PApp);
            t->a = readback(v->a);
            t->r = formulaTerm(v->f);
            t->b = readback(v->b);
            t->c = readback(v->c);
            return t;
        }
        case Val::K::Unglue: {
            auto t = mk(Term::K::Unglue);
            t->a = readback(v->a);
            t->b = readback(v->b);
            t->sys = readbackSys(v->sys, false, "", 0);
            return t;
        }
    }
    throw KernelError("internal: read-back of an unknown value");
}

// ---------------------------------------------------------------------------
// The elaborator
// ---------------------------------------------------------------------------

Elab::Elab(Signature& s) : S(s) {}

EnvP Elab::env() const { return envs.empty() ? EnvP{} : envs.back(); }

void Elab::push(Entry e) {
    EnvP base = env();
    EnvP n;
    switch (e.kind) {
        case Entry::Kind::Value: n = envValue(base, e.internal, e.value); break;
        case Entry::Kind::Dim: n = envDim(base, e.internal, e.f); break;
        case Entry::Kind::Level: n = base; break;
    }
    ctx.push_back(std::move(e));
    envs.push_back(n);
}

void Elab::pop() {
    ctx.pop_back();
    envs.pop_back();
}

V Elab::bindVar(const std::string& surface, const V& type) {
    V x = mkVar(surface, type);
    Entry e;
    e.kind = Entry::Kind::Value;
    e.surface = surface;
    e.internal = varInternal(x->id);
    e.type = type;
    e.value = x;
    push(std::move(e));
    return x;
}

Dim Elab::bindDim(const std::string& surface) {
    Dim k = freshDim();
    Entry e;
    e.kind = Entry::Kind::Dim;
    e.surface = surface;
    e.internal = dimInternal(k);
    e.f = Formula::atom(k);
    push(std::move(e));
    return k;
}

void Elab::bindLevel(const std::string& surface) {
    Entry e;
    e.kind = Entry::Kind::Level;
    e.surface = surface;
    e.internal = surface;
    push(std::move(e));
}

const Elab::Entry* Elab::find(const std::string& name) const {
    for (auto it = ctx.rbegin(); it != ctx.rend(); ++it)
        if (it->surface == name) return &*it;
    return nullptr;
}

void Elab::underFace(const Face& f, const std::function<void()>& body) {
    std::vector<Entry> saved = ctx;
    std::vector<EnvP> savedEnvs = envs;
    std::vector<Entry> restricted = ctx;
    ctx.clear();
    envs.clear();
    for (auto& e : restricted) {
        if (e.kind == Entry::Kind::Value) {
            e.type = restrict(e.type, f);
            e.value = restrict(e.value, f);
        } else if (e.kind == Entry::Kind::Dim) {
            for (auto& [d, b] : f) e.f = e.f.subst(d, Formula::endpoint(b));
        }
        push(e);
    }
    try {
        body();
    } catch (...) {
        ctx = std::move(saved);
        envs = std::move(savedEnvs);
        throw;
    }
    ctx = std::move(saved);
    envs = std::move(savedEnvs);
}

V Elab::evalT(const TermP& t) { return eval(env(), t); }

// ---- levels and the interval ----

LExprP Elab::level(const Sx& sx) {
    if (sx.isNumber()) return lconst((uint32_t)std::stoul(sx.text));
    if (sx.isSymbol()) {
        const Entry* e = find(sx.text);
        if (!e || e->kind != Entry::Kind::Level) fail(sx, "`" + sx.text + "` is not a level");
        return lvar(e->internal);
    }
    if (sx.isList() && !sx.items.empty() && sx.items[0].isSymbol("lsuc") && sx.items.size() == 2)
        return lsuc(level(sx.items[1]));
    if (sx.isList() && !sx.items.empty() && sx.items[0].isSymbol("lmax") && sx.items.size() >= 2) {
        LExprP out = level(sx.items[1]);
        for (size_t i = 2; i < sx.items.size(); ++i) out = lmax(out, level(sx.items[i]));
        return out;
    }
    fail(sx, "malformed level: " + printSx(sx));
}

bool Elab::isLevelExpr(const Sx& sx) const {
    if (sx.isNumber()) return true;
    if (sx.isSymbol()) {
        const Entry* e = find(sx.text);
        return e && e->kind == Entry::Kind::Level;
    }
    return sx.isList() && !sx.items.empty() && (sx.items[0].isSymbol("lsuc") || sx.items[0].isSymbol("lmax"));
}

IExprP Elab::interval(const Sx& sx) {
    if (sx.isSymbol("i0") || (sx.isNumber() && sx.text == "0")) return iz();
    if (sx.isSymbol("i1") || (sx.isNumber() && sx.text == "1")) return io();
    if (sx.isSymbol()) {
        const Entry* e = find(sx.text);
        if (!e || e->kind != Entry::Kind::Dim) fail(sx, "`" + sx.text + "` is not a dimension");
        return ivar(e->internal);
    }
    if (sx.isList() && sx.items.size() == 2 && sx.items[0].isSymbol("~")) return ineg(interval(sx.items[1]));
    if (sx.isList() && sx.items.size() >= 2 && (sx.items[0].isSymbol("min") || sx.items[0].isSymbol("max"))) {
        bool isMin = sx.items[0].isSymbol("min");
        IExprP out = interval(sx.items[1]);
        for (size_t i = 2; i < sx.items.size(); ++i)
            out = isMin ? imin(out, interval(sx.items[i])) : imax(out, interval(sx.items[i]));
        return out;
    }
    fail(sx, "malformed interval expression: " + printSx(sx));
}

IExprP Elab::face(const Sx& sx) {
    if (sx.isSymbol("i1")) return io();
    if (sx.isSymbol("i0")) return iz();
    if (sx.isList() && sx.items.size() == 3 && sx.items[0].isSymbol("=")) {
        IExprP r = interval(sx.items[1]);
        const Sx& e = sx.items[2];
        if (e.isSymbol("i1") || (e.isNumber() && e.text == "1")) return r;
        if (e.isSymbol("i0") || (e.isNumber() && e.text == "0")) return ineg(r);
        fail(e, "a face compares a dimension with 0 or 1");
    }
    if (sx.isList() && sx.items.size() >= 2 && (sx.items[0].isSymbol("and") || sx.items[0].isSymbol("or"))) {
        bool isAnd = sx.items[0].isSymbol("and");
        IExprP out = face(sx.items[1]);
        for (size_t i = 2; i < sx.items.size(); ++i)
            out = isAnd ? imin(out, face(sx.items[i])) : imax(out, face(sx.items[i]));
        return out;
    }
    fail(sx, "malformed face (expected (= i 0), (= i 1), (and ...) or (or ...)): " + printSx(sx));
}

// ---- types ----

std::pair<TermP, Level> Elab::inferType(const Sx& sx) {
    auto [t, T] = infer(sx);
    if (T->k != Val::K::U) fail(sx, "expected a type, but " + printSx(sx) + " has type " + show(T));
    return {t, T->level};
}

TermP Elab::checkTypeAt(const Sx& sx, const Level& l) {
    auto [t, lt] = inferType(sx);
    if (!lt.leq(l)) fail(sx, "type " + printSx(sx) + " lives in universe " + lt.str() + ", above " + l.str());
    return t;
}

void Elab::expect(const Sx& at, const V& T, const V& got, const V& want, const char* what) {
    if (!conv(T, got, want))
        fail(at, std::string(what) + ": " + show(got) + " is not definitionally equal to " + show(want));
}

// Binders: (x : A) or ((x : A) (y : B)) after a head.
std::vector<std::pair<std::string, Sx>> Elab::binders(const Sx& sx) {
    std::vector<std::pair<std::string, Sx>> out;
    auto one = [&](const Sx& b) {
        if (!b.isList() || b.items.size() < 3 || !b.items[b.items.size() - 2].isSymbol(":"))
            fail(b, "a binder is written (x : A) or (x y : A)");
        for (size_t i = 0; i + 2 < b.items.size(); ++i) {
            if (!b.items[i].isSymbol()) fail(b.items[i], "binder name must be a symbol");
            out.push_back({b.items[i].text, b.items.back()});
        }
    };
    if (sx.isList() && sx.items.size() >= 3 && sx.items[sx.items.size() - 2].isSymbol(":") &&
        sx.items[0].isSymbol())
        one(sx);
    else if (sx.isList())
        for (auto& b : sx.items) one(b);
    else
        fail(sx, "expected binders");
    return out;
}

TermP Elab::binderType(bool pi, const std::vector<std::pair<std::string, Sx>>& bs, size_t k,
                       const Sx& body, Level& lvl) {
    if (k == bs.size()) {
        auto [t, l] = inferType(body);
        lvl = l;
        return t;
    }
    auto [A, la] = inferType(bs[k].second);
    V Av = evalT(A);
    V x = bindVar(bs[k].first, Av);
    Level lb;
    TermP B;
    try {
        B = binderType(pi, bs, k + 1, body, lb);
    } catch (...) { pop(); throw; }
    pop();
    auto t = mk(pi ? Term::K::Pi : Term::K::Sigma, &bs[k].second);
    t->name = varInternal(x->id);
    t->a = A;
    t->b = B;
    lvl = la.max(lb);
    return t;
}

// ---- inference ----

std::pair<TermP, V> Elab::infer(const Sx& sx) {
    if (sx.isNumber()) {
        if (!S.natural) fail(sx, "numerals need (builtin natural ...)");
        uint64_t n = std::stoull(sx.text);
        auto t = mk(Term::K::Con, &sx);
        t->data = S.natural;
        t->con = S.natZero;
        for (uint64_t i = 0; i < n; ++i) {
            auto s = mk(Term::K::Con, &sx);
            s->data = S.natural;
            s->con = S.natSuc;
            s->args.push_back(t);
            t = s;
        }
        return {t, mkData(S.natural, {}, {})};
    }
    if (sx.isSymbol()) return inferSymbol(sx);
    if (!sx.isList() || sx.items.empty()) fail(sx, "cannot elaborate " + printSx(sx));
    const Sx& h = sx.items[0];
    const auto& it = sx.items;
    if (h.isSymbol()) {
        const std::string& n = h.text;
        if (n == "U") {
            if (it.size() != 2) fail(sx, "(U level)");
            auto t = mk(Term::K::U, &sx);
            t->level = level(it[1]);
            Level l = evalLevel(env(), t->level);
            return {t, mkU(l.suc())};
        }
        if (n == "Pi" || n == "Sigma") {
            if (it.size() != 3) fail(sx, "(" + n + " ((x : A) ...) B)");
            Level l;
            TermP t = binderType(n == "Pi", binders(it[1]), 0, it[2], l);
            return {t, mkU(l)};
        }
        if (n == "->") {
            if (it.size() < 3) fail(sx, "(-> A ... B)");
            std::vector<std::pair<std::string, Sx>> bs;
            for (size_t i = 1; i + 1 < it.size(); ++i) bs.push_back({"_", it[i]});
            Level l;
            TermP t = binderType(true, bs, 0, it.back(), l);
            return {t, mkU(l)};
        }
        if (n == "Path") {
            if (it.size() != 4) fail(sx, "(Path A a b)");
            auto [A, l] = inferType(it[1]);
            V Av = evalT(A);
            TermP a = check(it[2], Av), b = check(it[3], Av);
            auto line = mk(Term::K::PLam, &sx);
            line->name = dimInternal(freshDim());
            line->a = A;
            auto t = mk(Term::K::PathP, &sx);
            t->a = line; t->b = a; t->c = b;
            return {t, mkU(l)};
        }
        if (n == "PathP") return inferPathP(sx);
        if (n == "the") {
            if (it.size() != 3) fail(sx, "(the T e)");
            auto [T, l] = inferType(it[1]);
            (void)l;
            V Tv = evalT(T);
            return {check(it[2], Tv), Tv};
        }
        if (n == "fst" || n == "snd") {
            if (it.size() != 2) fail(sx, "(" + n + " p)");
            auto [p, T] = infer(it[1]);
            if (T->k != Val::K::Sigma) fail(sx, n + " of a non-pair of type " + show(T));
            auto t = mk(n == "fst" ? Term::K::Fst : Term::K::Snd, &sx);
            t->a = p;
            if (n == "fst") return {t, T->a};
            return {t, piCod(T, vfst(evalT(p)))};
        }
        if (n == "transp") return inferTransp(sx);
        if (n == "hcomp") return inferHComp(sx);
        if (n == "comp") return inferComp(sx);
        if (n == "Glue") return inferGlue(sx);
        if (n == "unglue") {
            if (it.size() != 2) fail(sx, "(unglue x)");
            auto [x, T] = infer(it[1]);
            if (T->k != Val::K::Glue) fail(sx, "unglue of a value of type " + show(T) + ", not a Glue type");
            auto t = mk(Term::K::Unglue, &sx);
            t->a = x;
            t->b = readback(T->a);
            t->sys = readbackGlueSys(T->sys);
            return {t, T->a};
        }
        if (n == "let") return inferLet(sx, nullptr);
        if (n == "lambda" || n == "λ" || n == "split" || n == "pair" || n == "glue")
            fail(sx, "cannot infer the type of (" + n + " ...); use it where a type is known, e.g. (the T e)");
        // Applications with a named head.
        if (const Entry* e = find(n)) {
            (void)e;
        } else if (n == selfName) {
            return inferSelfCall(sx);
        } else if (S.globals.count(n)) {
            const GlobalDef& g = *S.globals.at(n);
            size_t nl = g.levelParams.size();
            if (it.size() - 1 < nl) fail(sx, "`" + n + "` takes " + std::to_string(nl) + " level argument(s) first");
            auto t = mk(Term::K::Global, &h);
            t->name = n;
            for (size_t i = 0; i < nl; ++i) t->levels.push_back(level(it[1 + i]));
            V T = globalType(g, t->levels);
            return applyArgs(sx, t, T, 1 + nl);
        } else if (S.datas.count(n)) {
            return inferDataApp(sx, *S.datas.at(n));
        } else if (S.constructors.count(n)) {
            auto [d, c] = S.constructors.at(n);
            if (!d->params.empty() || !d->levelParams.empty())
                fail(sx, "constructor `" + n + "` of a parameterized type: use it where its type is known, "
                         "e.g. (the (" + d->name + " ...) (" + n + " ...))");
            return inferCon(sx, d, c);
        }
    }
    auto [f, T] = infer(h);
    return applyArgs(sx, f, T, 1);
}

std::pair<TermP, V> Elab::inferLet(const Sx& sx, const V* expected) {
    const auto& it = sx.items;
    if (it.size() != 3 || !it[1].isList()) fail(sx, "(let ((x : T e) ...) body)");
    std::vector<std::pair<std::string, TermP>> defs;
    size_t pushed = 0;
    TermP body;
    V T;
    try {
        for (auto& b : it[1].items) {
            if (!b.isList() || b.items.size() != 4 || !b.items[1].isSymbol(":") || !b.items[0].isSymbol())
                fail(b, "a let binding is written (x : T e)");
            auto [Ty, l] = inferType(b.items[2]);
            (void)l;
            V Tv = evalT(Ty);
            TermP e = check(b.items[3], Tv);
            V ev = evalT(e);
            Entry en;
            en.kind = Entry::Kind::Value;
            en.surface = b.items[0].text;
            V marker = mkVar(b.items[0].text, Tv);
            en.internal = varInternal(marker->id);
            en.type = Tv;
            en.value = ev;
            push(en);
            ++pushed;
            defs.push_back({en.internal, e});
        }
        if (expected) {
            body = check(it[2], *expected);
            T = *expected;
        } else {
            auto r = infer(it[2]);
            body = r.first;
            T = r.second;
        }
    } catch (...) {
        for (size_t i = 0; i < pushed; ++i) pop();
        throw;
    }
    for (size_t i = 0; i < pushed; ++i) pop();
    for (auto d = defs.rbegin(); d != defs.rend(); ++d) {
        auto l = mk(Term::K::Let, &sx);
        l->name = d->first;
        l->a = d->second;
        l->b = body;
        body = l;
    }
    return {body, T};
}

std::pair<TermP, V> Elab::inferSymbol(const Sx& sx) {
    const std::string& n = sx.text;
    if (n == "U") {
        auto t = mk(Term::K::U, &sx);
        t->level = lconst(0);
        return {t, mkU(Level::constant(1))};
    }
    if (const Entry* e = find(n)) {
        if (e->kind == Entry::Kind::Dim) fail(sx, "dimension `" + n + "` used where a term is expected");
        if (e->kind == Entry::Kind::Level) fail(sx, "level `" + n + "` used where a term is expected");
        auto t = mk(Term::K::Var, &sx);
        t->name = e->internal;
        return {t, e->type};
    }
    if (n == selfName) fail(sx, "`" + n + "` may only be called on a structurally smaller argument");
    if (auto g = S.globals.find(n); g != S.globals.end()) {
        if (!g->second->levelParams.empty())
            fail(sx, "`" + n + "` takes " + std::to_string(g->second->levelParams.size()) +
                         " level argument(s): (" + n + " l ...)");
        auto t = mk(Term::K::Global, &sx);
        t->name = n;
        return {t, globalType(*g->second, {})};
    }
    if (auto d = S.datas.find(n); d != S.datas.end()) {
        const DataDecl& dd = *d->second;
        if (!dd.params.empty() || !dd.levelParams.empty()) fail(sx, "`" + n + "` needs its parameters");
        auto t = mk(Term::K::Data, &sx);
        t->data = &dd;
        return {t, mkU(dataLevel(&dd, {}))};
    }
    if (auto c = S.constructors.find(n); c != S.constructors.end()) {
        auto [d, ci] = c->second;
        if (!d->params.empty() || !d->levelParams.empty())
            fail(sx, "constructor `" + n + "` of a parameterized type needs a known type: (the T " + n + ")");
        return inferCon(sx, d, ci);
    }
    fail(sx, "unbound name `" + n + "`");
}

V Elab::globalType(const GlobalDef& g, const std::vector<LExprP>& ls) {
    EnvP e;
    EnvP cur = env();
    for (size_t i = 0; i < g.levelParams.size() && i < ls.size(); ++i)
        e = envLevel(e, g.levelParams[i], evalLevel(cur, ls[i]));
    return eval(e, g.type);
}

std::pair<TermP, V> Elab::applyArgs(const Sx& sx, TermP f, V T, size_t from) {
    for (size_t i = from; i < sx.items.size(); ++i) {
        const Sx& a = sx.items[i];
        if (T->k == Val::K::Pi) {
            TermP at = check(a, T->a);
            auto t = mk(Term::K::App, &sx);
            t->a = f;
            t->b = at;
            T = piCod(T, evalT(at));
            f = t;
        } else if (T->k == Val::K::PathP) {
            IExprP r = interval(a);
            auto t = mk(Term::K::PApp, &sx);
            t->a = f;
            t->r = r;
            t->b = readback(T->b);
            t->c = readback(T->c);
            T = lineAt(T->a, evalI(env(), r));
            f = t;
        } else {
            fail(a, "too many arguments: " + printSx(sx) + " applies a value of type " + show(T));
        }
    }
    return {f, T};
}

std::pair<TermP, V> Elab::inferDataApp(const Sx& sx, const DataDecl& d) {
    const auto& it = sx.items;
    size_t nl = d.levelParams.size(), np = d.params.size();
    if (it.size() != 1 + nl + np)
        fail(sx, "`" + d.name + "` takes " + std::to_string(nl) + " level(s) and " + std::to_string(np) +
                     " parameter(s)");
    auto t = mk(Term::K::Data, &sx);
    t->data = &d;
    std::vector<Level> ls;
    for (size_t i = 0; i < nl; ++i) {
        t->levels.push_back(level(it[1 + i]));
        ls.push_back(evalLevel(env(), t->levels.back()));
    }
    std::vector<V> ps;
    for (size_t i = 0; i < np; ++i) {
        V ty = paramType(&d, ls, ps);
        TermP p = check(it[1 + nl + i], ty);
        t->params.push_back(p);
        ps.push_back(evalT(p));
    }
    if (dataInProgress == &d) {
        // A recursive occurrence: the parameters must be the declared ones.
        for (size_t i = 0; i < np; ++i)
            if (t->params[i]->k != Term::K::Var || t->params[i]->name != dataParamInternals[i])
                fail(sx, "a recursive occurrence of `" + d.name + "` must repeat its parameters");
    }
    return {t, mkU(dataLevel(&d, ls))};
}

// Constructor application against the data type D (levels and params from D).
TermP Elab::checkCon(const Sx& sx, const V& D, const DataDecl* d, int c) {
    const ConDecl& cd = d->cons[c];
    std::vector<Sx> args;
    if (sx.isList()) args.assign(sx.items.begin() + 1, sx.items.end());
    if (args.size() == cd.args.size() && !cd.dims.empty()) {
        // path constructor without its dimensions: a path (eta-expanded)
        fail(sx, "internal: path constructor eta handled by the caller");
    }
    if (args.size() != cd.args.size() + cd.dims.size())
        fail(sx, "constructor `" + cd.name + "` takes " + std::to_string(cd.args.size()) +
                     " argument(s) and " + std::to_string(cd.dims.size()) + " dimension(s)");
    auto t = mk(Term::K::Con, &sx);
    t->data = d;
    t->con = c;
    for (auto& l : D->levels) t->levels.push_back(levelTerm(l));
    for (auto& p : D->params) t->params.push_back(readback(p));
    std::vector<V> earlier;
    for (size_t i = 0; i < cd.args.size(); ++i) {
        V ty = conArgType(d, D->levels, D->params, c, earlier);
        TermP a = check(args[i], ty);
        t->args.push_back(a);
        earlier.push_back(evalT(a));
    }
    for (size_t i = 0; i < cd.dims.size(); ++i) t->dims.push_back(interval(args[cd.args.size() + i]));
    return t;
}

// A constructor of a non-parameterized type in inference position. Without
// its dimensions a path constructor denotes a path (eta-expanded), whose
// endpoints are the constructor's boundary.
std::pair<TermP, V> Elab::inferCon(const Sx& sx, const DataDecl* d, int c) {
    const ConDecl& cd = d->cons[c];
    V D = mkData(d, {}, {});
    size_t given = sx.isList() ? sx.items.size() - 1 : 0;
    if (given != cd.args.size() || cd.dims.empty()) return {checkCon(sx, D, d, c), D};
    std::vector<Sx> app{Sx::symbol(cd.name)};
    if (sx.isList()) app.insert(app.end(), sx.items.begin() + 1, sx.items.end());
    std::vector<Dim> ks;
    for (size_t k = 0; k < cd.dims.size(); ++k) {
        std::string dn = "%" + cd.name + "." + std::to_string(k);
        ks.push_back(bindDim(dn));
        app.push_back(Sx::symbol(dn));
    }
    Sx full = Sx::list(app);
    full.line = sx.line; full.column = sx.column;
    TermP body;
    try {
        body = checkCon(full, D, d, c);
    } catch (...) { for (size_t k = 0; k < ks.size(); ++k) pop(); throw; }
    for (size_t k = 0; k < ks.size(); ++k) pop();
    for (size_t k = ks.size(); k-- > 0;) {
        auto l = mk(Term::K::PLam, &sx);
        l->name = dimInternal(ks[k]);
        l->a = body;
        body = l;
    }
    V pv = evalT(body);
    std::function<V(const V&, size_t)> typeOf = [&](const V& p, size_t m) -> V {
        if (m == 0) return D;
        auto ty = std::make_shared<Val>();
        ty->k = Val::K::PathP;
        ty->a = mkLine([&](const Formula& i) { return typeOf(lineAt(p, i), m - 1); });
        ty->b = lineAt(p, Formula::zero());
        ty->c = lineAt(p, Formula::one());
        return ty;
    };
    return {body, typeOf(pv, cd.dims.size())};
}

// ---- checking ----

TermP Elab::check(const Sx& sx, const V& T) {
    if (sx.isList() && !sx.items.empty() && sx.items[0].isSymbol()) {
        const std::string& n = sx.items[0].text;
        if ((n == "lambda" || n == "λ") && !find(n)) return checkLambda(sx, T);
        if (n == "split" && !find(n)) return checkSplit(sx, T);
        if (n == "pair" && !find(n)) {
            if (sx.items.size() != 3) fail(sx, "(pair a b)");
            if (T->k != Val::K::Sigma) fail(sx, "a pair checked against " + show(T));
            TermP a = check(sx.items[1], T->a);
            TermP b = check(sx.items[2], piCod(T, evalT(a)));
            auto t = mk(Term::K::Pair, &sx);
            t->a = a; t->b = b;
            return t;
        }
        if (n == "glue" && !find(n)) return checkGlue(sx, T);
        if (n == "let" && !find(n)) return inferLet(sx, &T).first;
    }
    // Constructors: against their data type, or as paths (eta).
    std::string head;
    if (sx.isSymbol()) head = sx.text;
    else if (sx.isList() && !sx.items.empty() && sx.items[0].isSymbol()) head = sx.items[0].text;
    if (!head.empty() && !find(head) && S.constructors.count(head) && head != selfName) {
        auto [d, c] = S.constructors.at(head);
        const ConDecl& cd = d->cons[c];
        size_t given = sx.isList() ? sx.items.size() - 1 : 0;
        if (given == cd.args.size() && !cd.dims.empty()) {
            // eta: (c args) = (lambda (i ...) (c args i ...))
            std::vector<Sx> ds;
            Sx lam = Sx::list({Sx::symbol("lambda"), Sx::list({}), Sx::list({})});
            lam.line = sx.line; lam.column = sx.column;
            std::vector<Sx> app{Sx::symbol(head)};
            if (sx.isList()) app.insert(app.end(), sx.items.begin() + 1, sx.items.end());
            for (size_t k = 0; k < cd.dims.size(); ++k) {
                std::string dn = "%" + head + "." + std::to_string(k);
                lam.items[1].items.push_back(Sx::symbol(dn));
                app.push_back(Sx::symbol(dn));
            }
            lam.items[2] = Sx::list(app);
            lam.items[2].line = sx.line; lam.items[2].column = sx.column;
            return checkLambda(lam, T);
        }
        if (T->k == Val::K::Data && T->data == d) return checkCon(sx, T, d, c);
    }
    auto [t, got] = infer(sx);
    if (!subtype(got, T))
        fail(sx, "type mismatch: " + printSx(sx) + " has type " + show(got) + " but " + show(T) + " was expected");
    return t;
}

TermP Elab::checkLambda(const Sx& sx, const V& T) {
    if (sx.items.size() != 3 || !sx.items[1].isList()) fail(sx, "(lambda (x ...) body)");
    std::vector<Sx> params = sx.items[1].items;
    if (params.empty()) return check(sx.items[2], T);
    Sx p = params[0];
    Sx rest = Sx::list({sx.items[0], Sx::list(std::vector<Sx>(params.begin() + 1, params.end())), sx.items[2]});
    rest.line = sx.line; rest.column = sx.column;
    std::string name;
    if (p.isSymbol()) name = p.text;
    else if (p.isList() && p.items.size() == 3 && p.items[1].isSymbol(":") && p.items[0].isSymbol()) name = p.items[0].text;
    else fail(p, "lambda parameter must be x or (x : A)");
    if (T->k == Val::K::Pi) {
        if (p.isList()) {
            auto [A, l] = inferType(p.items[2]);
            (void)l;
            if (!convType(evalT(A), T->a))
                fail(p, "parameter annotation " + printSx(p.items[2]) + " does not match " + show(T->a));
        }
        V x = bindVar(name, T->a);
        TermP body;
        try {
            if (topSpine) ++spineArgs;
            body = check(rest, piCod(T, x));
        } catch (...) { pop(); throw; }
        pop();
        auto t = mk(Term::K::Lam, &sx);
        t->name = varInternal(x->id);
        t->a = body;
        return t;
    }
    if (T->k == Val::K::PathP) {
        if (p.isList()) fail(p, "a path abstraction binds a dimension: (lambda (i) ...)");
        bool savedSpine = topSpine;
        topSpine = false;
        Dim k = bindDim(name);
        TermP body;
        try {
            body = check(rest, lineAt(T->a, Formula::atom(k)));
            // boundary: body(0) = a0, body(1) = a1
            for (int end = 0; end < 2; ++end) {
                Face f{{k, end == 1}};
                V bv = restrict(evalT(body), f);
                V want = end ? T->c : T->b;
                expect(sx, lineAt(T->a, Formula::endpoint(end == 1)), bv, want,
                       end ? "path does not end where its type says" : "path does not start where its type says");
            }
        } catch (...) { pop(); topSpine = savedSpine; throw; }
        pop();
        topSpine = savedSpine;
        auto t = mk(Term::K::PLam, &sx);
        t->name = dimInternal(k);
        t->a = body;
        return t;
    }
    fail(sx, "a lambda checked against " + show(T) + ", which is neither a function nor a path type");
}

std::pair<TermP, V> Elab::inferPathP(const Sx& sx) {
    const auto& it = sx.items;
    // (PathP (i) A a b)  or  (PathP (lambda (i) A) a b)
    std::string iname;
    Sx A;
    size_t first;
    if (it.size() == 5 && it[1].isList() && it[1].items.size() == 1 && it[1].items[0].isSymbol()) {
        iname = it[1].items[0].text; A = it[2]; first = 3;
    } else if (it.size() == 4 && it[1].isList() && it[1].items.size() == 3 &&
               (it[1].items[0].isSymbol("lambda") || it[1].items[0].isSymbol("λ")) &&
               it[1].items[1].isList() && it[1].items[1].items.size() == 1) {
        iname = it[1].items[1].items[0].text; A = it[1].items[2]; first = 2;
    } else {
        fail(sx, "(PathP (i) A a b)");
    }
    Dim k = bindDim(iname);
    TermP At;
    Level l;
    try {
        auto r = inferType(A);
        At = r.first; l = r.second;
    } catch (...) { pop(); throw; }
    pop();
    auto line = mk(Term::K::PLam, &sx);
    line->name = dimInternal(k);
    line->a = At;
    V lineV = evalT(line);
    TermP a = check(it[first], lineAt(lineV, Formula::zero()));
    TermP b = check(it[first + 1], lineAt(lineV, Formula::one()));
    auto t = mk(Term::K::PathP, &sx);
    t->a = line; t->b = a; t->c = b;
    return {t, mkU(l)};
}

// A line binder and a type under it: (i) A
TermP Elab::typeLine(const Sx& binder, const Sx& A, std::string& internal, Level& l) {
    if (!binder.isList() || binder.items.size() != 1 || !binder.items[0].isSymbol())
        fail(binder, "expected a dimension binder (i)");
    Dim k = bindDim(binder.items[0].text);
    internal = dimInternal(k);
    TermP At;
    try {
        auto r = inferType(A);
        At = r.first; l = r.second;
    } catch (...) { pop(); throw; }
    pop();
    return At;
}

std::pair<TermP, V> Elab::inferTransp(const Sx& sx) {
    const auto& it = sx.items;
    if (it.size() != 5) fail(sx, "(transp (i) A phi u)");
    std::string name;
    Level l;
    TermP A = typeLine(it[1], it[2], name, l);
    auto line = mk(Term::K::PLam, &sx);
    line->name = name; line->a = A;
    V lineV = evalT(line);
    IExprP r = interval(it[3]);
    Formula phi = evalI(env(), r);
    for (auto& f : facesOf(phi)) {
        V lf = restrict(lineV, f);
        Formula k = Formula::atom(freshDim());
        if (!convType(lineAt(lf, k), lineAt(lf, Formula::zero())))
            fail(it[3], "transp: the type line is not constant where " + printSx(it[3]) + " holds");
    }
    TermP u = check(it[4], lineAt(lineV, Formula::zero()));
    auto t = mk(Term::K::Transp, &sx);
    t->name = name; t->a = A; t->r = r; t->b = u;
    return {t, lineAt(lineV, Formula::one())};
}

// Elaborate the clauses of a composition system; `typeAt(face)` gives the
// type of a clause's line at its dimension (already restricted).
std::vector<TSysClause> Elab::lineSystem(const Sx& clauses, const std::string& binderSurface,
                                         const std::string& binderInternal,
                                         const std::function<V(const Formula& i)>& lineType,
                                         const V& u0, const V& baseType) {
    if (!clauses.isList()) fail(clauses, "a system is a list of (face term) clauses");
    std::vector<TSysClause> out;
    struct Done { Face f; TermP t; };
    std::vector<Done> done;
    for (auto& c : clauses.items) {
        if (!c.isList() || c.items.size() != 2) fail(c, "a system clause is (face term)");
        IExprP fe = face(c.items[0]);
        Formula phi = evalI(env(), fe);
        TermP clauseTerm;
        auto fs = facesOf(phi);
        if (fs.empty()) continue;  // a face that never holds contributes nothing
        bool first = true;
        for (auto& f : fs) {
            underFace(f, [&] {
                Entry e;
                e.kind = Entry::Kind::Dim;
                e.surface = binderSurface;
                e.internal = binderInternal;
                Dim k = freshDim();
                e.f = Formula::atom(k);
                push(e);
                try {
                    V Ti = restrict(lineType(Formula::atom(k)), f);
                    TermP t = check(c.items[1], Ti);
                    if (first) { clauseTerm = t; first = false; }
                    // at i = 0 the clause meets the base
                    V tv = act(evalT(t), k, Formula::zero());
                    expect(c, restrict(baseType, f), tv, restrict(u0, f),
                           "a composition side does not meet the base at i = 0");
                    done.push_back({f, t});
                } catch (...) { pop(); throw; }
                pop();
            });
        }
        out.push_back({fe, clauseTerm, nullptr});
    }
    // Sides agree where faces overlap.
    for (size_t a = 0; a < done.size(); ++a)
        for (size_t b = a + 1; b < done.size(); ++b) {
            auto m = faceMeet(done[a].f, done[b].f);
            if (!m) continue;
            Dim k = freshDim();
            EnvP e = envDim(env(), binderInternal, Formula::atom(k));
            V va = restrict(eval(e, done[a].t), *m);
            V vb = restrict(eval(e, done[b].t), *m);
            V Ti = restrict(lineType(Formula::atom(k)), *m);
            expect(clauses, Ti, va, vb, "composition sides disagree where their faces overlap");
        }
    return out;
}

std::pair<TermP, V> Elab::inferHComp(const Sx& sx) {
    const auto& it = sx.items;
    if (it.size() != 5 || !it[2].isList() || it[2].items.size() != 1)
        fail(sx, "(hcomp A (i) ((face side) ...) base)");
    auto [A, l] = inferType(it[1]);
    (void)l;
    V Av = evalT(A);
    TermP u0 = check(it[4], Av);
    V u0v = evalT(u0);
    auto t = mk(Term::K::HComp, &sx);
    t->a = A;
    t->b = u0;
    t->name = dimInternal(freshDim());
    t->sys = lineSystem(it[3], it[2].items[0].text, t->name,
                        [&](const Formula&) { return Av; }, u0v, Av);
    return {t, Av};
}

std::pair<TermP, V> Elab::inferComp(const Sx& sx) {
    const auto& it = sx.items;
    if (it.size() != 5) fail(sx, "(comp (i) A ((face side) ...) base)");
    std::string name;
    Level l;
    TermP A = typeLine(it[1], it[2], name, l);
    auto line = mk(Term::K::PLam, &sx);
    line->name = name; line->a = A;
    V lineV = evalT(line);
    V A0 = lineAt(lineV, Formula::zero());
    TermP u0 = check(it[4], A0);
    V u0v = evalT(u0);
    auto t = mk(Term::K::Comp, &sx);
    t->name = name;
    t->a = A;
    t->b = u0;
    t->sys = lineSystem(it[3], it[1].items[0].text, name,
                        [&](const Formula& i) { return lineAt(lineV, i); }, u0v, A0);
    return {t, lineAt(lineV, Formula::one())};
}

V Elab::equivType(const V& T, const V& A, const Level& l) {
    auto g = S.globals.find(S.equivName);
    if (g == S.globals.end()) throw KernelError("Glue needs the kernel prelude (" + S.equivName + ")");
    V eq = globalValue(S.equivName, {l, l});
    return app(app(eq, T), A);
}

std::vector<TSysClause> Elab::readbackGlueSys(const VSys& s) {
    std::vector<TSysClause> out;
    for (auto& c : s) out.push_back({faceTerm(c.face), readback(c.v), readback(c.v2)});
    return out;
}

std::pair<TermP, V> Elab::inferGlue(const Sx& sx) {
    const auto& it = sx.items;
    if (it.size() != 3 || !it[2].isList()) fail(sx, "(Glue A ((face T e) ...))");
    auto [A, l] = inferType(it[1]);
    V Av = evalT(A);
    auto t = mk(Term::K::Glue, &sx);
    t->a = A;
    struct Done { Face f; TermP T, e; };
    std::vector<Done> done;
    for (auto& c : it[2].items) {
        if (!c.isList() || c.items.size() != 3) fail(c, "a Glue clause is (face T e)");
        IExprP fe = face(c.items[0]);
        auto fs = facesOf(evalI(env(), fe));
        if (fs.empty()) continue;  // a face that never holds contributes nothing
        TermP Tt, et;
        bool first = true;
        for (auto& f : fs) {
            underFace(f, [&] {
                TermP T1 = checkTypeAt(c.items[1], l);
                V Tv = evalT(T1);
                TermP e1 = check(c.items[2], equivType(Tv, restrict(Av, f), l));
                if (first) { Tt = T1; et = e1; first = false; }
                done.push_back({f, T1, e1});
            });
        }
        t->sys.push_back({fe, Tt, et});
    }
    for (size_t a = 0; a < done.size(); ++a)
        for (size_t b = a + 1; b < done.size(); ++b) {
            auto m = faceMeet(done[a].f, done[b].f);
            if (!m) continue;
            V Ta = restrict(evalT(done[a].T), *m), Tb = restrict(evalT(done[b].T), *m);
            if (!convType(Ta, Tb)) fail(sx, "Glue clauses disagree on their types where faces overlap");
            V ea = restrict(evalT(done[a].e), *m), eb = restrict(evalT(done[b].e), *m);
            expect(sx, equivType(Ta, restrict(Av, *m), l), ea, eb,
                   "Glue clauses disagree on their equivalences where faces overlap");
        }
    return {t, mkU(l)};
}

TermP Elab::checkGlue(const Sx& sx, const V& T) {
    const auto& it = sx.items;
    if (it.size() != 3 || !it[1].isList()) fail(sx, "(glue ((face t) ...) a)");
    if (T->k != Val::K::Glue) fail(sx, "glue checked against " + show(T) + ", which is not a Glue type");
    TermP a = check(it[2], T->a);
    V av = evalT(a);
    auto t = mk(Term::K::GlueElem, &sx);
    t->a = a;
    std::vector<Face> given;
    for (auto& c : it[1].items) {
        if (!c.isList() || c.items.size() != 2) fail(c, "a glue clause is (face t)");
        IExprP fe = face(c.items[0]);
        auto fs = facesOf(evalI(env(), fe));
        TermP tt;
        bool first = true;
        for (auto& f : fs) {
            given.push_back(f);
            underFace(f, [&] {
                V Tf = restrict(T, f);  // the Glue type collapses to its T on f
                TermP t1 = check(c.items[1], Tf);
                if (first) { tt = t1; first = false; }
                // e t = a on f
                VSys sysF = T->sys;
                for (auto& [d, b] : f) sysF = actSys(sysF, d, Formula::endpoint(b));
                const VClause* tc = trueClause(sysF);
                if (!tc) fail(c, "glue clause face is not a face of the Glue type");
                expect(c, restrict(T->a, f), app(vfst(tc->v2), evalT(t1)), restrict(av, f),
                       "glue: the equivalence does not send the clause to the base element");
            });
        }
        t->sys.push_back({fe, tt, nullptr});
    }
    if (!sameFaces(given, sysFaces(T->sys)))
        fail(sx, "glue: the clause faces must be exactly the faces of the Glue type");
    return t;
}

// ---- splits ----

TermP Elab::checkSplit(const Sx& sx, const V& T) {
    if (T->k != Val::K::Pi || T->a->k != Val::K::Data)
        fail(sx, "split checked against " + show(T) + ", which is not a function out of an inductive type");
    V D = T->a;
    const DataDecl* d = D->data;
    bool recursive = topSpine && !selfName.empty();
    if (recursive && selfArgPos < 0) selfArgPos = spineArgs;
    bool savedSpine = topSpine;
    topSpine = false;
    auto t = mk(Term::K::Split, &sx);
    t->data = d;
    // motive: (x : D) -> B x, read back to a term over this context
    V mx = mkVar("x", D);
    t->motiveVar = varInternal(mx->id);
    t->b = readback(piCod(T, mx));
    std::vector<bool> seen(d->cons.size(), false);
    std::vector<std::pair<int, std::pair<std::vector<V>, std::vector<Dim>>>> pathClauses;
    for (size_t ci = 1; ci < sx.items.size(); ++ci) {
        const Sx& cl = sx.items[ci];
        if (!cl.isList() || cl.items.size() != 2) fail(cl, "a split clause is (pattern body)");
        const Sx& pat = cl.items[0];
        std::string cname = pat.isSymbol() ? pat.text : (pat.isList() && !pat.items.empty() && pat.items[0].isSymbol() ? pat.items[0].text : "");
        int c = d->conIndex(cname);
        if (c < 0) fail(pat, "`" + cname + "` is not a constructor of " + d->name);
        if (seen[c]) fail(pat, "constructor `" + cname + "` matched twice");
        seen[c] = true;
        const ConDecl& cd = d->cons[c];
        size_t nvars = pat.isList() ? pat.items.size() - 1 : 0;
        if (nvars != cd.args.size() + cd.dims.size())
            fail(pat, "pattern for `" + cname + "` binds " + std::to_string(cd.args.size()) + " argument(s) and " +
                          std::to_string(cd.dims.size()) + " dimension(s)");
        SplitClause sc;
        sc.con = c;
        size_t pushed = 0;
        std::vector<V> argVals;
        std::vector<Dim> dimAtoms;
        std::set<std::string> savedSmaller = smaller;
        try {
            for (size_t k = 0; k < cd.args.size(); ++k) {
                const Sx& v = pat.items[1 + k];
                if (!v.isSymbol()) fail(v, "pattern variables are symbols");
                V ty = conArgType(d, D->levels, D->params, c, argVals);
                V x = bindVar(v.text, ty);
                ++pushed;
                argVals.push_back(x);
                sc.args.push_back(varInternal(x->id));
                if (recursive && cd.recursive[k]) smaller.insert(varInternal(x->id));
            }
            for (size_t k = 0; k < cd.dims.size(); ++k) {
                const Sx& v = pat.items[1 + cd.args.size() + k];
                if (!v.isSymbol()) fail(v, "pattern dimensions are symbols");
                Dim a = bindDim(v.text);
                ++pushed;
                dimAtoms.push_back(a);
                sc.dims.push_back(dimInternal(a));
            }
            std::vector<Formula> dims;
            for (Dim a : dimAtoms) dims.push_back(Formula::atom(a));
            V cv = mkCon(d, D->levels, D->params, c, argVals, dims);
            sc.body = check(cl.items[1], piCod(T, cv));
            if (!cd.boundary.empty()) {
                // Boundary checks run once the definition is in place (they may
                // unfold recursive calls).
                EnvP clauseEnv = env();
                EnvP splitEnv;
                {
                    // the environment of the split itself: the context before the pattern variables
                    splitEnv = envs.size() > pushed ? envs[envs.size() - pushed - 1] : EnvP{};
                }
                TermP body = sc.body;
                V Tcopy = T;
                const DataDecl* dd = d;
                std::vector<V> dParams = D->params;
                std::vector<Level> dLevels = D->levels;
                std::shared_ptr<const Term> splitTerm = t;
                std::vector<Dim> atoms = dimAtoms;
                Sx where = cl;
                deferred.push_back([=]() {
                    std::vector<Formula> fd;
                    for (Dim a : atoms) fd.push_back(Formula::atom(a));
                    V bodyV = eval(clauseEnv, body);
                    V splitV = eval(splitEnv, splitTerm);
                    // The constructor value with generic dimensions, before its boundary.
                    for (auto& [faceE, bterm] : dd->cons[c].boundary) {
                        EnvP be;
                        for (size_t q = 0; q < dd->levelParams.size() && q < dLevels.size(); ++q)
                            be = envLevel(be, dd->levelParams[q], dLevels[q]);
                        for (size_t q = 0; q < dd->params.size(); ++q)
                            be = envValue(be, dd->params[q].first, dParams[q]);
                        for (size_t q = 0; q < argVals.size(); ++q)
                            be = envValue(be, dd->cons[c].args[q].first, argVals[q]);
                        for (size_t q = 0; q < atoms.size(); ++q)
                            be = envDim(be, dd->cons[c].dims[q], fd[q]);
                        for (auto& f : facesOf(evalI(be, faceE))) {
                            V lhs = restrict(bodyV, f);
                            V bval = restrict(eval(be, bterm), f);
                            V rhs = app(restrict(splitV, f), bval);
                            V ty = restrict(piCod(Tcopy, bval), f);
                            if (!conv(ty, lhs, rhs))
                                throw KernelError("split clause for `" + dd->cons[c].name +
                                                      "` does not agree with the clauses of its boundary: " +
                                                      show(lhs) + " versus " + show(rhs),
                                                  where.line, where.column);
                        }
                    }
                });
            }
        } catch (...) {
            for (size_t i = 0; i < pushed; ++i) pop();
            smaller = savedSmaller;
            topSpine = savedSpine;
            throw;
        }
        for (size_t i = 0; i < pushed; ++i) pop();
        smaller = savedSmaller;
        t->clauses.push_back(std::move(sc));
    }
    for (size_t c = 0; c < seen.size(); ++c)
        if (!seen[c]) fail(sx, "split does not cover constructor `" + d->cons[c].name + "`");
    topSpine = savedSpine;
    // If no deferred definition will run the checks (a split outside a
    // define), run them now.
    if (selfName.empty()) runDeferred();
    return t;
}

void Elab::runDeferred() {
    auto checks = std::move(deferred);
    deferred.clear();
    for (auto& f : checks) f();
}

std::pair<TermP, V> Elab::inferSelfCall(const Sx& sx) {
    const auto& it = sx.items;
    const GlobalDef& g = *S.globals.at(selfName);
    size_t nl = g.levelParams.size();
    if (selfArgPos < 0) fail(sx, "`" + selfName + "` calls itself outside a split: recursion must be structural");
    size_t pos = 1 + nl + (size_t)selfArgPos;
    if (it.size() <= pos) fail(sx, "recursive call of `" + selfName + "` must pass its decreasing argument");
    const Sx& arg = it[pos];
    const Entry* e = arg.isSymbol() ? find(arg.text) : nullptr;
    if (!e || e->kind != Entry::Kind::Value || !smaller.count(e->internal))
        fail(arg, "recursive call of `" + selfName + "` on " + printSx(arg) +
                      ", which is not a structurally smaller pattern variable");
    auto t = mk(Term::K::Global, &it[0]);
    t->name = selfName;
    for (size_t i = 0; i < nl; ++i) {
        t->levels.push_back(level(it[1 + i]));
        if (!(evalLevel(env(), t->levels.back()) == Level::var(g.levelParams[i])))
            fail(it[1 + i], "recursive call of `" + selfName + "` must use its own levels");
    }
    V T = globalType(g, t->levels);
    return applyArgs(sx, t, T, 1 + nl);
}

}  // namespace eshkol::hott::cubical
