/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Evaluation for the cubical kernel (ADR-0043): weak head normal forms,
 * dimension substitution, and the Kan operations.
 *
 * The Kan operations follow CCHM (Cohen, Coquand, Huber, Mortberg, "Cubical
 * Type Theory: a constructive interpretation of the univalence axiom", 2018)
 * in the transp/hcomp presentation of Cubical Agda (Vezzosi, Mortberg, Abel,
 * JFP 2021), with higher inductive types as in Coquand, Huber, Mortberg, "On
 * Higher Inductive Types in Cubical Type Theory" (LICS 2018):
 *
 *   transp  computes on Pi, Sigma, PathP, U, Glue and inductive types;
 *   hcomp   computes on Pi, Sigma, PathP, Glue; in U it builds a Glue type;
 *           in an inductive type it is a constructor that eliminators compute
 *           on.
 *
 * Value invariants: every clause of a system is restricted to its face, and
 * every smart constructor re-checks the reductions a substitution can unblock
 * (a true face, a path constructor reaching its boundary, an endpoint applied
 * to a neutral path), so `act` reduces as it substitutes.
 */

#include "cubical_internal.h"

#include <algorithm>
#include <atomic>

namespace eshkol::hott::cubical {

namespace {
const Signature* g_sig = nullptr;

std::shared_ptr<Val> node(Val::K k) {
    auto v = std::make_shared<Val>();
    v->k = k;
    return v;
}

[[noreturn]] void internal(const std::string& what) {
    throw KernelError("cubical kernel invariant: " + what);
}
}  // namespace

const Signature& sig() {
    if (!g_sig) internal("no signature in scope");
    return *g_sig;
}
SignatureScope::SignatureScope(const Signature* s) : prev(g_sig) { g_sig = s; }
SignatureScope::~SignatureScope() { g_sig = prev; }

// ---------------------------------------------------------------------------
// Environments
// ---------------------------------------------------------------------------

EnvP envValue(const EnvP& e, const std::string& n, V v) {
    auto r = std::make_shared<Env>();
    r->kind = Env::Kind::Value; r->name = n; r->v = std::move(v); r->next = e;
    return r;
}
EnvP envDim(const EnvP& e, const std::string& n, Formula f) {
    auto r = std::make_shared<Env>();
    r->kind = Env::Kind::Dim; r->name = n; r->f = std::move(f); r->next = e;
    return r;
}
EnvP envLevel(const EnvP& e, const std::string& n, Level l) {
    auto r = std::make_shared<Env>();
    r->kind = Env::Kind::Level; r->name = n; r->l = std::move(l); r->next = e;
    return r;
}

static const Env* lookup(const EnvP& env, const std::string& n, Env::Kind k) {
    for (const Env* e = env.get(); e; e = e->next.get())
        if (e->kind == k && e->name == n) return e;
    return nullptr;
}

Level evalLevel(const EnvP& env, const LExprP& l) {
    if (!l) return Level::constant(0);
    switch (l->k) {
        case LExpr::K::Const: return Level::constant(l->n);
        case LExpr::K::Var: {
            const Env* e = lookup(env, l->name, Env::Kind::Level);
            if (!e) return Level::var(l->name);  // rigid level variable
            return e->l;
        }
        case LExpr::K::Suc: return evalLevel(env, l->a).suc();
        case LExpr::K::Max: return evalLevel(env, l->a).max(evalLevel(env, l->b));
    }
    return Level::constant(0);
}

Formula evalI(const EnvP& env, const IExprP& r) {
    switch (r->k) {
        case IExpr::K::Zero: return Formula::zero();
        case IExpr::K::One: return Formula::one();
        case IExpr::K::Var: {
            const Env* e = lookup(env, r->name, Env::Kind::Dim);
            if (!e) internal("unbound dimension " + r->name);
            return e->f;
        }
        case IExpr::K::Neg: return evalI(env, r->a).neg();
        case IExpr::K::Min: return evalI(env, r->a).meet(evalI(env, r->b));
        case IExpr::K::Max: return evalI(env, r->a).join(evalI(env, r->b));
    }
    return Formula::zero();
}

static EnvP actEnv(const EnvP& env, Dim d, const Formula& r) {
    if (!env) return env;
    // Rebuild the chain bottom-up.
    std::vector<const Env*> chain;
    for (const Env* e = env.get(); e; e = e->next.get()) chain.push_back(e);
    EnvP out;
    bool changed = false;
    for (auto it = chain.rbegin(); it != chain.rend(); ++it) {
        const Env* e = *it;
        auto n = std::make_shared<Env>(*e);
        n->next = out;
        if (e->kind == Env::Kind::Value) {
            V nv = act(e->v, d, r);
            if (nv != e->v) changed = true;
            n->v = nv;
        } else if (e->kind == Env::Kind::Dim) {
            if (e->f.mentions(d)) { n->f = e->f.subst(d, r); changed = true; }
        }
        out = n;
    }
    return changed ? out : env;
}

static bool envOccurs(const EnvP& env, Dim d) {
    for (const Env* e = env.get(); e; e = e->next.get()) {
        if (e->kind == Env::Kind::Value && occurs(e->v, d)) return true;
        if (e->kind == Env::Kind::Dim && e->f.mentions(d)) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// Constructors of values
// ---------------------------------------------------------------------------

V mkU(Level l) {
    auto v = node(Val::K::U);
    v->level = std::move(l);
    return v;
}

V mkVar(const std::string& name, V type) {
    static std::atomic<uint64_t> next{1};
    auto v = node(Val::K::Var);
    v->id = next.fetch_add(1);
    v->name = name;
    v->a = std::move(type);
    return v;
}

V mkPair(V a, V b) {
    auto v = node(Val::K::Pair);
    v->a = std::move(a);
    v->b = std::move(b);
    return v;
}

V mkPLam(Dim i, V body) {
    auto v = node(Val::K::PLam);
    v->dim = i;
    v->a = std::move(body);
    return v;
}

V mkLine(const std::function<V(const Formula&)>& f) {
    Dim k = freshDim();
    return mkPLam(k, f(Formula::atom(k)));
}

V lineAt(const V& line, const Formula& r) {
    if (line->k != Val::K::PLam) internal("lineAt on a non-line");
    return act(line->a, line->dim, r);
}

static V constLine(const V& v) {
    return mkPLam(freshDim(), v);
}

V mkData(const DataDecl* d, std::vector<Level> ls, std::vector<V> ps) {
    auto v = node(Val::K::Data);
    v->data = d;
    v->levels = std::move(ls);
    v->params = std::move(ps);
    return v;
}

static EnvP dataEnv(const DataDecl* d, const std::vector<Level>& ls, const std::vector<V>& ps) {
    EnvP env;
    for (size_t i = 0; i < d->levelParams.size() && i < ls.size(); ++i)
        env = envLevel(env, d->levelParams[i], ls[i]);
    for (size_t i = 0; i < d->params.size() && i < ps.size(); ++i)
        env = envValue(env, d->params[i].first, ps[i]);
    return env;
}

V paramType(const DataDecl* d, const std::vector<Level>& ls, const std::vector<V>& earlier) {
    EnvP env = dataEnv(d, ls, earlier);
    return eval(env, d->params[earlier.size()].second);
}

V conArgType(const DataDecl* d, const std::vector<Level>& ls, const std::vector<V>& ps,
             int con, const std::vector<V>& earlier) {
    EnvP env = dataEnv(d, ls, ps);
    const ConDecl& c = d->cons[con];
    for (size_t i = 0; i < earlier.size(); ++i) env = envValue(env, c.args[i].first, earlier[i]);
    return eval(env, c.args[earlier.size()].second);
}

Level dataLevel(const DataDecl* d, const std::vector<Level>& ls) {
    return evalLevel(dataEnv(d, ls, {}), d->universe);
}

V mkCon(const DataDecl* d, std::vector<Level> ls, std::vector<V> ps, int con,
        std::vector<V> args, std::vector<Formula> dims) {
    const ConDecl& c = d->cons[con];
    if (!c.boundary.empty()) {
        EnvP env = dataEnv(d, ls, ps);
        for (size_t i = 0; i < args.size(); ++i) env = envValue(env, c.args[i].first, args[i]);
        for (size_t i = 0; i < dims.size(); ++i) env = envDim(env, c.dims[i], dims[i]);
        for (auto& [face, term] : c.boundary)
            if (evalI(env, face).isOne()) return eval(env, term);
    }
    auto v = node(Val::K::Con);
    v->data = d;
    v->levels = std::move(ls);
    v->params = std::move(ps);
    v->con = con;
    v->args = std::move(args);
    v->dims = std::move(dims);
    return v;
}

const VClause* trueClause(const VSys& s) {
    for (auto& c : s)
        if (c.face.empty()) return &c;
    return nullptr;
}

std::vector<Face> sysFaces(const VSys& s) {
    std::vector<Face> fs;
    for (auto& c : s) fs.push_back(c.face);
    return normalizeFaces(fs);
}

V mkGlue(V A, VSys sys) {
    if (auto* c = trueClause(sys)) return c->v;
    auto v = node(Val::K::Glue);
    v->a = std::move(A);
    v->sys = std::move(sys);
    return v;
}

V mkGlueElem(VSys sys, V a) {
    if (auto* c = trueClause(sys)) return c->v;
    auto v = node(Val::K::GlueElem);
    v->a = std::move(a);
    v->sys = std::move(sys);
    return v;
}

V unglue(V x, V A, VSys sys) {
    if (auto* c = trueClause(sys)) return app(vfst(c->v2), x);
    if (x->k == Val::K::GlueElem) return x->a;
    auto v = node(Val::K::Unglue);
    v->a = std::move(x);
    v->b = std::move(A);
    v->sys = std::move(sys);
    return v;
}

// ---------------------------------------------------------------------------
// Substitution of dimensions
// ---------------------------------------------------------------------------

static Formula restrictFormula(Formula f, const Face& face) {
    for (auto& [d, b] : face) f = f.subst(d, Formula::endpoint(b));
    return f;
}

V restrict(const V& v, const Face& f) {
    V out = v;
    for (auto& [d, b] : f) out = act(out, d, Formula::endpoint(b));
    return out;
}

static VSys restrictSys(const VSys& s, const Face& g) {
    VSys out;
    for (auto& c : s) {
        auto m = faceMeet(c.face, g);
        if (!m) continue;
        // Faces of the result are over the dimensions g leaves free.
        Face rest;
        for (auto& p : *m)
            if (!faceMentions(g, p.first)) rest.push_back(p);
        out.push_back({rest, restrict(c.v, *m), c.v2 ? restrict(c.v2, *m) : nullptr});
    }
    return out;
}

VSys actSys(const VSys& s, Dim d, const Formula& r) {
    VSys out;
    std::set<Dim> rdims;
    r.collect(rdims);
    for (auto& c : s) {
        V v = act(c.v, d, r);
        V v2 = c.v2 ? act(c.v2, d, r) : nullptr;
        if (faceMentions(c.face, d)) {
            for (auto& f : faceSubst(c.face, d, r))
                out.push_back({f, restrict(v, f), v2 ? restrict(v2, f) : nullptr});
        } else {
            bool overlap = false;
            for (auto& p : c.face)
                if (rdims.count(p.first)) overlap = true;
            if (overlap) out.push_back({c.face, restrict(v, c.face), v2 ? restrict(v2, c.face) : nullptr});
            else out.push_back({c.face, v, v2});
        }
    }
    return out;
}

static bool sysOccurs(const VSys& s, Dim d) {
    for (auto& c : s) {
        if (faceMentions(c.face, d)) return true;
        if (occurs(c.v, d)) return true;
        if (c.v2 && occurs(c.v2, d)) return true;
    }
    return false;
}

bool occurs(const V& v, Dim d) {
    if (!v) return false;
    switch (v->k) {
        case Val::K::U: return false;
        case Val::K::Pi:
        case Val::K::Sigma: return occurs(v->a, d) || envOccurs(v->clo.env, d);
        case Val::K::Lam: return envOccurs(v->clo.env, d);
        case Val::K::Split: return envOccurs(v->clo.env, d);
        case Val::K::Pair: return occurs(v->a, d) || occurs(v->b, d);
        case Val::K::PathP: return occurs(v->a, d) || occurs(v->b, d) || occurs(v->c, d);
        case Val::K::PLam: return v->dim != d && occurs(v->a, d);
        case Val::K::Data:
            for (auto& p : v->params) if (occurs(p, d)) return true;
            return false;
        case Val::K::Con:
            for (auto& p : v->params) if (occurs(p, d)) return true;
            for (auto& p : v->args) if (occurs(p, d)) return true;
            for (auto& f : v->dims) if (f.mentions(d)) return true;
            return false;
        case Val::K::Glue:
        case Val::K::GlueElem: return occurs(v->a, d) || sysOccurs(v->sys, d);
        case Val::K::HComp: return occurs(v->a, d) || occurs(v->b, d) || sysOccurs(v->sys, d);
        case Val::K::Transp: return occurs(v->a, d) || occurs(v->b, d) || v->f.mentions(d);
        case Val::K::Var: return occurs(v->a, d);
        case Val::K::App: return occurs(v->a, d) || occurs(v->b, d);
        case Val::K::Fst:
        case Val::K::Snd: return occurs(v->a, d);
        case Val::K::PApp: return occurs(v->a, d) || v->f.mentions(d) || occurs(v->b, d) || occurs(v->c, d);
        case Val::K::SplitApp: return occurs(v->a, d) || occurs(v->b, d);
        case Val::K::Unglue: return occurs(v->a, d) || occurs(v->b, d) || sysOccurs(v->sys, d);
    }
    return false;
}

V act(const V& v, Dim d, const Formula& r) {
    if (!v) return v;
    switch (v->k) {
        case Val::K::U: return v;
        case Val::K::Pi:
        case Val::K::Sigma:
        case Val::K::Lam:
        case Val::K::Split: {
            EnvP env = actEnv(v->clo.env, d, r);
            V a = v->a ? act(v->a, d, r) : nullptr;
            if (env == v->clo.env && a == v->a) return v;
            auto n = std::make_shared<Val>(*v);
            n->clo.env = env;
            n->a = a;
            return n;
        }
        case Val::K::Pair: {
            V a = act(v->a, d, r), b = act(v->b, d, r);
            if (a == v->a && b == v->b) return v;
            return mkPair(a, b);
        }
        case Val::K::PathP: {
            auto n = std::make_shared<Val>(*v);
            n->a = act(v->a, d, r);
            n->b = act(v->b, d, r);
            n->c = act(v->c, d, r);
            return n;
        }
        case Val::K::PLam: {
            if (v->dim == d) return v;
            if (r.mentions(v->dim)) {
                Dim k = freshDim();
                V body = act(v->a, v->dim, Formula::atom(k));
                return mkPLam(k, act(body, d, r));
            }
            V body = act(v->a, d, r);
            if (body == v->a) return v;
            return mkPLam(v->dim, body);
        }
        case Val::K::Data: {
            std::vector<V> ps;
            for (auto& p : v->params) ps.push_back(act(p, d, r));
            return mkData(v->data, v->levels, std::move(ps));
        }
        case Val::K::Con: {
            std::vector<V> ps, as;
            std::vector<Formula> ds;
            for (auto& p : v->params) ps.push_back(act(p, d, r));
            for (auto& p : v->args) as.push_back(act(p, d, r));
            for (auto& f : v->dims) ds.push_back(f.subst(d, r));
            return mkCon(v->data, v->levels, std::move(ps), v->con, std::move(as), std::move(ds));
        }
        case Val::K::Glue: return mkGlue(act(v->a, d, r), actSys(v->sys, d, r));
        case Val::K::GlueElem: return mkGlueElem(actSys(v->sys, d, r), act(v->a, d, r));
        case Val::K::HComp: return hcomp(act(v->a, d, r), actSys(v->sys, d, r), act(v->b, d, r));
        case Val::K::Transp: return transp(act(v->a, d, r), v->f.subst(d, r), act(v->b, d, r));
        case Val::K::Var: {
            if (!v->a || !occurs(v->a, d)) return v;
            auto n = std::make_shared<Val>(*v);
            n->a = act(v->a, d, r);
            return n;
        }
        case Val::K::App: return app(act(v->a, d, r), act(v->b, d, r));
        case Val::K::Fst: return vfst(act(v->a, d, r));
        case Val::K::Snd: return vsnd(act(v->a, d, r));
        case Val::K::PApp:
            return papp(act(v->a, d, r), v->f.subst(d, r), act(v->b, d, r), act(v->c, d, r));
        case Val::K::SplitApp: return app(act(v->a, d, r), act(v->b, d, r));
        case Val::K::Unglue: return unglue(act(v->a, d, r), act(v->b, d, r), actSys(v->sys, d, r));
    }
    return v;
}

// ---------------------------------------------------------------------------
// Elimination
// ---------------------------------------------------------------------------

V apply(const Clo& c, const V& arg) {
    return eval(envValue(c.env, c.x, arg), c.body);
}

V piDom(const V& pi) {
    if (pi->k != Val::K::Pi && pi->k != Val::K::Sigma) internal("piDom on a non-binder type");
    return pi->a;
}
V piCod(const V& pi, const V& x) {
    if (pi->k != Val::K::Pi && pi->k != Val::K::Sigma) internal("piCod on a non-binder type");
    return apply(pi->clo, x);
}

static V splitApply(const V& s, const V& a);
static V transpPiApp(const V& line, const Formula& phi, const V& u0, const V& v);

V app(const V& f, const V& a) {
    switch (f->k) {
        case Val::K::Lam: return apply(f->clo, a);
        case Val::K::Split: return splitApply(f, a);
        case Val::K::HComp:
            if (f->a->k == Val::K::Pi) {
                V B = piCod(f->a, a);
                VSys sys;
                for (auto& c : f->sys) {
                    V line = c.v;
                    V ar = restrict(a, c.face);
                    sys.push_back({c.face, mkLine([&](const Formula& i) { return app(lineAt(line, i), ar); }), nullptr});
                }
                return hcomp(B, std::move(sys), app(f->b, a));
            }
            break;
        case Val::K::Transp: {
            V generic = lineAt(f->a, Formula::atom(freshDim()));
            if (generic->k == Val::K::Pi) return transpPiApp(f->a, f->f, f->b, a);
            break;
        }
        default: break;
    }
    auto v = node(Val::K::App);
    v->a = f;
    v->b = a;
    return v;
}

V vfst(const V& p) {
    if (p->k == Val::K::Pair) return p->a;
    auto v = node(Val::K::Fst);
    v->a = p;
    return v;
}

V vsnd(const V& p) {
    if (p->k == Val::K::Pair) return p->b;
    auto v = node(Val::K::Snd);
    v->a = p;
    return v;
}

V papp(const V& p, const Formula& r, const V& a0, const V& a1) {
    if (p->k == Val::K::PLam) return act(p->a, p->dim, r);
    if (r.isZero()) return a0;
    if (r.isOne()) return a1;
    auto v = node(Val::K::PApp);
    v->a = p;
    v->f = r;
    v->b = a0;
    v->c = a1;
    return v;
}

static V motiveAt(const V& s, const V& x) {
    const Term& t = *s->split;
    return eval(envValue(s->clo.env, t.motiveVar, x), t.b);
}

static V splitApply(const V& s, const V& a) {
    const Term& t = *s->split;
    if (a->k == Val::K::Con) {
        for (auto& cl : t.clauses) {
            if (cl.con != a->con) continue;
            EnvP env = s->clo.env;
            for (size_t i = 0; i < cl.args.size(); ++i) env = envValue(env, cl.args[i], a->args[i]);
            for (size_t i = 0; i < cl.dims.size(); ++i) env = envDim(env, cl.dims[i], a->dims[i]);
            return eval(env, cl.body);
        }
        internal("split has no clause for constructor " + a->data->cons[a->con].name);
    }
    if (a->k == Val::K::HComp && a->a->k == Val::K::Data) {
        // CHM18: f (hcomp^D [phi |-> u] u0)
        //      = comp^i (B (hfill^D [phi |-> u] u0 i)) [phi |-> f (u i)] (f u0)
        V fill = hfill(a->a, a->sys, a->b);
        V motive = mkLine([&](const Formula& i) { return motiveAt(s, lineAt(fill, i)); });
        VSys sys;
        for (auto& c : a->sys) {
            V sf = restrict(s, c.face);
            V line = c.v;
            sys.push_back({c.face, mkLine([&](const Formula& i) { return app(sf, lineAt(line, i)); }), nullptr});
        }
        return comp(motive, Formula::zero(), std::move(sys), app(s, a->b));
    }
    auto v = node(Val::K::SplitApp);
    v->a = s;
    v->b = a;
    return v;
}

// ---------------------------------------------------------------------------
// Kan operations
// ---------------------------------------------------------------------------

V hfill(const V& A, const VSys& sys, const V& u0) {
    return mkLine([&](const Formula& i) {
        Dim di = i.dnf[0][0].d;
        VSys s;
        for (auto& c : sys) {
            V line = c.v;
            s.push_back({c.face, mkLine([&](const Formula& j) { return lineAt(line, i.meet(j)); }), nullptr});
        }
        s.push_back({Face{{di, false}}, constLine(u0), nullptr});
        return hcomp(A, std::move(s), u0);
    });
}

V comp(const V& line, const Formula& psi, VSys sys, const V& u0) {
    VSys s;
    for (auto& c : sys) {
        V l = restrict(line, c.face);
        Formula p = restrictFormula(psi, c.face);
        V u = c.v;
        s.push_back({c.face, mkLine([&](const Formula& i) {
            V shifted = mkLine([&](const Formula& k) { return lineAt(l, i.join(k)); });
            return transp(shifted, i.join(p), lineAt(u, i));
        }), nullptr});
    }
    return hcomp(lineAt(line, Formula::one()), std::move(s), transp(line, psi, u0));
}

static V transpPiApp(const V& line, const Formula& phi, const V& u0, const V& v) {
    // transp (i. (x : A i) -> B i x) phi u0 = \v. transp (i. B i (x i)) phi (u0 (x 0))
    // with x i = transp (j. A (~j \/ i)) (phi \/ i) v.
    V x = mkLine([&](const Formula& i) {
        V dom = mkLine([&](const Formula& j) { return piDom(lineAt(line, j.neg().join(i))); });
        return transp(dom, phi.join(i), v);
    });
    V cod = mkLine([&](const Formula& i) { return piCod(lineAt(line, i), lineAt(x, i)); });
    return transp(cod, phi, app(u0, lineAt(x, Formula::zero())));
}

static V transpGlue(const V& line, const Formula& psi, const V& u0);
static V hcompGlue(const V& A, const VSys& sys, const V& u0);
static V transpData(const V& line, const Formula& phi, const V& u0, Dim i, const V& generic);

V transp(const V& line, const Formula& phi, const V& u0) {
    if (phi.isOne()) return u0;
    Dim i = freshDim();
    V A = lineAt(line, Formula::atom(i));
    switch (A->k) {
        case Val::K::U: return u0;
        case Val::K::Pi: break;  // kept as a value; reduces on application
        case Val::K::Sigma: {
            V a = mkLine([&](const Formula& r) {
                V dom = mkLine([&](const Formula& j) { return piDom(lineAt(line, r.meet(j))); });
                return transp(dom, phi.join(r.neg()), vfst(u0));
            });
            V cod = mkLine([&](const Formula& r) { return piCod(lineAt(line, r), lineAt(a, r)); });
            return mkPair(lineAt(a, Formula::one()), transp(cod, phi, vsnd(u0)));
        }
        case Val::K::PathP: {
            // transp (i. PathP (P i) (a0 i) (a1 i)) phi p
            //   = <j> comp (i. P i j) phi [phi |-> p j, j=0 |-> a0 i, j=1 |-> a1 i] (p j)
            V A0 = lineAt(line, Formula::zero());
            return mkLine([&](const Formula& j) {
                Dim dj = j.dnf[0][0].d;
                V pj = mkLine([&](const Formula& r) { return lineAt(lineAt(line, r)->a, j); });
                VSys s;
                for (auto& f : facesOf(phi)) {
                    V p = restrict(u0, f);
                    V A0f = restrict(A0, f);
                    s.push_back({f, constLine(papp(p, j, A0f->b, A0f->c)), nullptr});
                }
                s.push_back({Face{{dj, false}}, mkLine([&](const Formula& r) { return lineAt(line, r)->b; }), nullptr});
                s.push_back({Face{{dj, true}}, mkLine([&](const Formula& r) { return lineAt(line, r)->c; }), nullptr});
                // the endpoint clauses are restricted by construction: a0/a1 do not mention j
                return comp(pj, phi, std::move(s), papp(u0, j, A0->b, A0->c));
            });
        }
        case Val::K::Data: return transpData(line, phi, u0, i, A);
        case Val::K::Glue: return transpGlue(line, phi, u0);
        default: break;
    }
    auto v = node(Val::K::Transp);
    v->a = line;
    v->f = phi;
    v->b = u0;
    return v;
}

static V transpData(const V& line, const Formula& phi, const V& u0, Dim i, const V& generic) {
    bool constant = true;
    for (auto& p : generic->params)
        if (occurs(p, i)) { constant = false; break; }
    if (constant) return u0;  // a data type along a constant line: transport is the identity
    const DataDecl* d = generic->data;
    if (u0->k == Val::K::Con) {
        // Transport the arguments through the constructor's telescope: argument k
        // lives over the parameters and the fillers of the earlier arguments.
        std::vector<V> fills;  // lines
        const ConDecl& c = d->cons[u0->con];
        for (size_t k = 0; k < c.args.size(); ++k) {
            auto typeAt = [&](const Formula& r) {
                V Dr = lineAt(line, r);
                std::vector<V> earlier;
                for (auto& f : fills) earlier.push_back(lineAt(f, r));
                return conArgType(d, Dr->levels, Dr->params, u0->con, earlier);
            };
            V arg = u0->args[k];
            fills.push_back(mkLine([&](const Formula& r) {
                V ty = mkLine([&](const Formula& j) { return typeAt(r.meet(j)); });
                return transp(ty, phi.join(r.neg()), arg);
            }));
        }
        V D1 = lineAt(line, Formula::one());
        std::vector<V> args;
        for (auto& f : fills) args.push_back(lineAt(f, Formula::one()));
        return mkCon(d, D1->levels, D1->params, u0->con, std::move(args), u0->dims);
    }
    if (u0->k == Val::K::HComp) {
        VSys s;
        for (auto& c : u0->sys) {
            V l = restrict(line, c.face);
            Formula p = restrictFormula(phi, c.face);
            V u = c.v;
            s.push_back({c.face, mkLine([&](const Formula& j) { return transp(l, p, lineAt(u, j)); }), nullptr});
        }
        return hcomp(lineAt(line, Formula::one()), std::move(s), transp(line, phi, u0->b));
    }
    auto v = node(Val::K::Transp);
    v->a = line;
    v->f = phi;
    v->b = u0;
    return v;
}

static V fiberType(const V& T, const V& A, const V& f, const V& y) {
    V fib = globalValue("fiber", {Level::constant(0), Level::constant(0)});
    return app(app(app(app(fib, T), A), f), y);
}

/**
 * equivProof (Agda.Builtin.Cubical.Equiv; CCHM Lemma 5 with the phi = 0 face
 * added): extend the partial element `partial` of the fiber of the
 * equivalence `e` over `y` to a total one, using contractibility.
 */
static V extendFiber(const V& T, const V& A, const V& e, const V& y, const VSys& partial) {
    V cp = app(vsnd(e), y);
    V center = vfst(cp);
    V contraction = vsnd(cp);
    V F = fiberType(T, A, vfst(e), y);
    VSys s;
    std::vector<Face> pf;
    for (auto& c : partial) {
        V con = restrict(contraction, c.face);
        V cen = restrict(center, c.face);
        V u = c.v;
        s.push_back({c.face, mkLine([&](const Formula& i) { return papp(app(con, u), i, cen, u); }), nullptr});
        pf.push_back(c.face);
    }
    for (auto& f : facesOf(facesFormula(pf).neg()))
        s.push_back({f, constLine(restrict(center, f)), nullptr});
    return hcomp(F, std::move(s), center);
}

static V transpGlue(const V& line, const Formula& psi, const V& u0) {
    Dim i = freshDim();
    V gi = lineAt(line, Formula::atom(i));  // Glue (B i) [phi i |-> (T i, e i)]
    auto at = [&](const V& v, const Formula& r) { return act(v, i, r); };
    V B0 = at(gi->a, Formula::zero());
    V B1 = at(gi->a, Formula::one());
    V bLine = mkLine([&](const Formula& r) { return at(gi->a, r); });
    V a0 = unglue(u0, B0, actSys(gi->sys, i, Formula::zero()));

    // delta = forall i. phi(i): the clauses that do not mention i.
    VSys deltaFill;  // face |-> line r |-> t(r) in T(r)
    VSys deltaE;     // face |-> line r |-> e(r)
    for (auto& c : gi->sys) {
        if (faceMentions(c.face, i)) continue;
        V T = c.v, e = c.v2;
        V u = restrict(u0, c.face);
        Formula p = restrictFormula(psi, c.face);
        V tLine = mkLine([&](const Formula& r) {
            V ty = mkLine([&](const Formula& j) { return at(T, r.meet(j)); });
            return transp(ty, p.join(r.neg()), u);
        });
        deltaFill.push_back({c.face, tLine, nullptr});
        deltaE.push_back({c.face, mkLine([&](const Formula& r) { return at(e, r); }), nullptr});
    }

    // a1' = comp (i. B i) psi [psi |-> a0, delta |-> e i (t i)] a0
    VSys s1;
    for (auto& f : facesOf(psi)) s1.push_back({f, constLine(restrict(a0, f)), nullptr});
    for (size_t k = 0; k < deltaFill.size(); ++k) {
        V tl = deltaFill[k].v, el = deltaE[k].v;
        s1.push_back({deltaFill[k].face, mkLine([&](const Formula& r) {
            return app(vfst(lineAt(el, r)), lineAt(tl, r));
        }), nullptr});
    }
    V a1p = comp(bLine, psi, s1, a0);

    // On each face G of phi(1): the fiber of e(1) over a1', extended from
    // psi |-> (u0, refl) and delta |-> (t 1, refl).
    VSys sys1 = actSys(gi->sys, i, Formula::one());
    VSys tSys;
    VSys aSys;
    for (auto& G : sys1) {
        V T1 = G.v, e1 = G.v2;
        V y = restrict(a1p, G.face);
        VSys partial;
        auto addPartial = [&](const Face& f, const V& t) {
            auto m = faceMeet(f, G.face);
            if (!m) return;
            Face rest;
            for (auto& p : *m)
                if (!faceMentions(G.face, p.first)) rest.push_back(p);
            V tr = restrict(t, *m);
            V yr = restrict(y, rest);
            partial.push_back({rest, mkPair(tr, constLine(yr)), nullptr});
        };
        for (auto& f : facesOf(psi)) addPartial(f, u0);
        for (auto& c : deltaFill) addPartial(c.face, lineAt(c.v, Formula::one()));
        V B1G = restrict(B1, G.face);
        V fib = extendFiber(T1, B1G, e1, y, partial);
        V t1 = vfst(fib);
        V alpha = vsnd(fib);  // Path B1 (e1 t1) y
        tSys.push_back({G.face, t1, nullptr});
        V ft1 = app(vfst(e1), t1);
        aSys.push_back({G.face, mkLine([&](const Formula& j) { return papp(alpha, j.neg(), ft1, y); }), nullptr});
    }
    for (auto& f : facesOf(psi)) aSys.push_back({f, constLine(restrict(a1p, f)), nullptr});
    V a1 = hcomp(B1, std::move(aSys), a1p);
    return mkGlueElem(std::move(tSys), a1);
}

static V hcompGlue(const V& A, const VSys& sys, const V& u0) {
    // hcomp (Glue B [phi |-> (T, e)]) [psi |-> u] u0
    //   = glue [phi |-> hcomp T [psi |-> u] u0]
    //          (hcomp B [psi |-> unglue u, phi |-> e (hfill T [psi |-> u] u0)] (unglue u0))
    V B = A->a;
    VSys tSys;
    VSys aSys;
    for (auto& c : sys) {
        V line = c.v;
        V Bf = restrict(B, c.face);
        VSys gf = restrictSys(A->sys, c.face);
        aSys.push_back({c.face, mkLine([&](const Formula& i) { return unglue(lineAt(line, i), Bf, gf); }), nullptr});
    }
    for (auto& G : A->sys) {
        VSys sG = restrictSys(sys, G.face);
        V uG = restrict(u0, G.face);
        tSys.push_back({G.face, hcomp(G.v, sG, uG), nullptr});
        V fill = hfill(G.v, sG, uG);
        V e = G.v2;
        aSys.push_back({G.face, mkLine([&](const Formula& i) { return app(vfst(e), lineAt(fill, i)); }), nullptr});
    }
    V a1 = hcomp(B, std::move(aSys), unglue(u0, B, A->sys));
    return mkGlueElem(std::move(tSys), a1);
}

V hcomp(const V& A, VSys sys, const V& u0) {
    if (auto* c = trueClause(sys)) return lineAt(c->v, Formula::one());
    if (sys.empty()) return u0;
    switch (A->k) {
        case Val::K::Sigma: {
            VSys fs, ss;
            for (auto& c : sys) {
                V line = c.v;
                fs.push_back({c.face, mkLine([&](const Formula& i) { return vfst(lineAt(line, i)); }), nullptr});
                ss.push_back({c.face, mkLine([&](const Formula& i) { return vsnd(lineAt(line, i)); }), nullptr});
            }
            V a = hfill(A->a, fs, vfst(u0));
            V cod = mkLine([&](const Formula& i) { return piCod(A, lineAt(a, i)); });
            return mkPair(lineAt(a, Formula::one()), comp(cod, Formula::zero(), std::move(ss), vsnd(u0)));
        }
        case Val::K::PathP:
            return mkLine([&](const Formula& j) {
                Dim dj = j.dnf[0][0].d;
                VSys s;
                for (auto& c : sys) {
                    V line = c.v;
                    V a0 = restrict(A->b, c.face), a1 = restrict(A->c, c.face);
                    s.push_back({c.face, mkLine([&](const Formula& i) { return papp(lineAt(line, i), j, a0, a1); }), nullptr});
                }
                s.push_back({Face{{dj, false}}, constLine(A->b), nullptr});
                s.push_back({Face{{dj, true}}, constLine(A->c), nullptr});
                return hcomp(lineAt(A->a, j), std::move(s), papp(u0, j, A->b, A->c));
            });
        case Val::K::U: {
            // CCHM 7.2: composition in the universe is a Glue type,
            // hcomp U [phi |-> T] A = Glue A [phi |-> (T 1, pathToEquiv (i. T (~i)))].
            if (!sig().globals.count(sig().pathToEquivName))
                throw KernelError("composition in a universe needs the kernel prelude (" +
                                  sig().pathToEquivName + ")");
            VSys g;
            for (auto& c : sys) {
                V line = c.v;
                V T1 = lineAt(line, Formula::one());
                V T0 = lineAt(line, Formula::zero());
                V rev = mkLine([&](const Formula& i) { return lineAt(line, i.neg()); });
                V p2e = globalValue(sig().pathToEquivName, {A->level});
                V e = app(app(app(p2e, T1), T0), rev);
                g.push_back({c.face, T1, e});
            }
            return mkGlue(u0, std::move(g));
        }
        case Val::K::Glue: return hcompGlue(A, sys, u0);
        default: break;  // Pi: reduces on application; Data: the hcomp constructor; neutral: stuck
    }
    auto v = node(Val::K::HComp);
    v->a = A;
    v->b = u0;
    v->sys = std::move(sys);
    return v;
}

// ---------------------------------------------------------------------------
// Globals
// ---------------------------------------------------------------------------

V globalValue(const std::string& name, const std::vector<Level>& ls) {
    auto it = sig().globals.find(name);
    if (it == sig().globals.end()) internal("unknown global " + name);
    GlobalDef& g = *it->second;
    if (!g.body) return g.opaque;
    std::string key;
    for (auto& l : ls) key += l.str() + ";";
    auto c = g.cache.find(key);
    if (c != g.cache.end()) return c->second;
    EnvP env;
    for (size_t k = 0; k < g.levelParams.size() && k < ls.size(); ++k)
        env = envLevel(env, g.levelParams[k], ls[k]);
    V v = eval(env, g.body);
    if (v->k == Val::K::Split && v->name.empty()) {
        // remember the origin, so a stuck application reads back to the global
        auto n = std::make_shared<Val>(*v);
        n->name = "@" + g.name;
        n->levels = ls;
        v = n;
    }
    g.cache[key] = v;
    return v;
}

// ---------------------------------------------------------------------------
// eval
// ---------------------------------------------------------------------------

static VSys evalLines(const EnvP& env, const std::string& binder, const std::vector<TSysClause>& sys) {
    VSys out;
    for (auto& c : sys) {
        Formula phi = evalI(env, c.face);
        for (auto& f : facesOf(phi)) {
            Dim k = freshDim();
            V body = eval(envDim(env, binder, Formula::atom(k)), c.t);
            out.push_back({f, restrict(mkPLam(k, body), f), nullptr});
        }
    }
    return out;
}

static VSys evalClauses(const EnvP& env, const std::vector<TSysClause>& sys) {
    VSys out;
    for (auto& c : sys) {
        Formula phi = evalI(env, c.face);
        for (auto& f : facesOf(phi)) {
            V v = restrict(eval(env, c.t), f);
            V v2 = c.t2 ? restrict(eval(env, c.t2), f) : nullptr;
            out.push_back({f, v, v2});
        }
    }
    return out;
}

V eval(const EnvP& env, const TermP& t) {
    switch (t->k) {
        case Term::K::Var: {
            const Env* e = lookup(env, t->name, Env::Kind::Value);
            if (!e) internal("unbound variable " + t->name);
            return e->v;
        }
        case Term::K::Global: {
            std::vector<Level> ls;
            for (auto& l : t->levels) ls.push_back(evalLevel(env, l));
            return globalValue(t->name, ls);
        }
        case Term::K::U: return mkU(evalLevel(env, t->level));
        case Term::K::Pi:
        case Term::K::Sigma: {
            auto v = node(t->k == Term::K::Pi ? Val::K::Pi : Val::K::Sigma);
            v->name = t->name;
            v->a = eval(env, t->a);
            v->clo = Clo{env, t->name, t->b};
            return v;
        }
        case Term::K::Lam: {
            auto v = node(Val::K::Lam);
            v->name = t->name;
            v->clo = Clo{env, t->name, t->a};
            return v;
        }
        case Term::K::App: return app(eval(env, t->a), eval(env, t->b));
        case Term::K::Pair: return mkPair(eval(env, t->a), eval(env, t->b));
        case Term::K::Fst: return vfst(eval(env, t->a));
        case Term::K::Snd: return vsnd(eval(env, t->a));
        case Term::K::Let: return eval(envValue(env, t->name, eval(env, t->a)), t->b);
        case Term::K::PathP: {
            auto v = node(Val::K::PathP);
            v->a = eval(env, t->a);
            v->b = eval(env, t->b);
            v->c = eval(env, t->c);
            return v;
        }
        case Term::K::PLam: {
            Dim k = freshDim();
            return mkPLam(k, eval(envDim(env, t->name, Formula::atom(k)), t->a));
        }
        case Term::K::PApp: return papp(eval(env, t->a), evalI(env, t->r), eval(env, t->b), eval(env, t->c));
        case Term::K::Data: {
            std::vector<Level> ls;
            for (auto& l : t->levels) ls.push_back(evalLevel(env, l));
            std::vector<V> ps;
            for (auto& p : t->params) ps.push_back(eval(env, p));
            return mkData(t->data, std::move(ls), std::move(ps));
        }
        case Term::K::Con: {
            std::vector<Level> ls;
            for (auto& l : t->levels) ls.push_back(evalLevel(env, l));
            std::vector<V> ps, as;
            for (auto& p : t->params) ps.push_back(eval(env, p));
            for (auto& p : t->args) as.push_back(eval(env, p));
            std::vector<Formula> ds;
            for (auto& r : t->dims) ds.push_back(evalI(env, r));
            return mkCon(t->data, std::move(ls), std::move(ps), t->con, std::move(as), std::move(ds));
        }
        case Term::K::Split: {
            auto v = node(Val::K::Split);
            v->split = t;
            v->clo = Clo{env, t->motiveVar, nullptr};
            return v;
        }
        case Term::K::Transp: {
            Dim k = freshDim();
            V line = mkPLam(k, eval(envDim(env, t->name, Formula::atom(k)), t->a));
            return transp(line, evalI(env, t->r), eval(env, t->b));
        }
        case Term::K::HComp:
            return hcomp(eval(env, t->a), evalLines(env, t->name, t->sys), eval(env, t->b));
        case Term::K::Comp: {
            Dim k = freshDim();
            V line = mkPLam(k, eval(envDim(env, t->name, Formula::atom(k)), t->a));
            return comp(line, Formula::zero(), evalLines(env, t->name, t->sys), eval(env, t->b));
        }
        case Term::K::Glue: return mkGlue(eval(env, t->a), evalClauses(env, t->sys));
        case Term::K::GlueElem: return mkGlueElem(evalClauses(env, t->sys), eval(env, t->a));
        case Term::K::Unglue: return unglue(eval(env, t->a), eval(env, t->b), evalClauses(env, t->sys));
    }
    internal("unknown term");
}

}  // namespace eshkol::hott::cubical
