/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Definitional equality for the cubical kernel (ADR-0043), type-directed so
 * that it includes the eta laws: functions (Pi), pairs (Sigma), paths (PathP:
 * compared at a fresh dimension) and Glue elements (compared through unglue
 * and on every face of the Glue system). Types are compared structurally,
 * universe levels by their normal forms. Stuck terms are compared spine by
 * spine; comparing a spine computes its type, so arguments are compared at
 * the right types.
 *
 * quote() reads a value back to an S-expression: the normal form printed by
 * `hott-eval` and by diagnostics.
 */

#include "cubical_internal.h"

#include <map>

namespace eshkol::hott::cubical {

static V motiveOf(const V& s, const V& x) {
    const Term& t = *s->split;
    return eval(envValue(s->clo.env, t.motiveVar, x), t.b);
}

bool isStuck(const V& v) {
    switch (v->k) {
        case Val::K::Var:
        case Val::K::App:
        case Val::K::Fst:
        case Val::K::Snd:
        case Val::K::PApp:
        case Val::K::SplitApp:
        case Val::K::Unglue:
        case Val::K::Transp:
            return true;
        case Val::K::HComp:
            return v->a->k != Val::K::Data && v->a->k != Val::K::Pi;
        default:
            return false;
    }
}

V typeOfStuck(const V& v) {
    switch (v->k) {
        case Val::K::Var:
            if (!v->a) throw KernelError("internal: untyped variable " + v->name);
            return v->a;
        case Val::K::App: {
            V T = typeOfStuck(v->a);
            if (T->k != Val::K::Pi) throw KernelError("internal: application of a non-function");
            return piCod(T, v->b);
        }
        case Val::K::Fst: return piDom(typeOfStuck(v->a));
        case Val::K::Snd: return piCod(typeOfStuck(v->a), vfst(v->a));
        case Val::K::PApp: {
            V T = typeOfStuck(v->a);
            if (T->k != Val::K::PathP) throw KernelError("internal: path application of a non-path");
            return lineAt(T->a, v->f);
        }
        case Val::K::SplitApp: return motiveOf(v->a, v->b);
        case Val::K::Unglue: return v->b;
        case Val::K::HComp: return v->a;
        case Val::K::Transp: return lineAt(v->a, Formula::one());
        default: break;
    }
    throw KernelError("internal: type of a canonical value requested as stuck");
}

// ---------------------------------------------------------------------------

static bool convSysLines(const V& A, const VSys& s1, const VSys& s2) {
    if (!sameFaces(sysFaces(s1), sysFaces(s2))) return false;
    for (auto& c1 : s1)
        for (auto& c2 : s2) {
            auto m = faceMeet(c1.face, c2.face);
            if (!m) continue;
            Dim k = freshDim();
            Formula fk = Formula::atom(k);
            V Am = restrict(A, *m);
            if (!conv(Am, lineAt(restrict(c1.v, *m), fk), lineAt(restrict(c2.v, *m), fk))) return false;
        }
    return true;
}

static V equivType(const V& T, const V& A) {
    V eq = globalValue(sig().equivName, {Level::constant(0), Level::constant(0)});
    return app(app(eq, T), A);
}

static bool convGlueSys(const V& A, const VSys& s1, const VSys& s2) {
    if (!sameFaces(sysFaces(s1), sysFaces(s2))) return false;
    for (auto& c1 : s1)
        for (auto& c2 : s2) {
            auto m = faceMeet(c1.face, c2.face);
            if (!m) continue;
            V T1 = restrict(c1.v, *m), T2 = restrict(c2.v, *m);
            if (!convType(T1, T2)) return false;
            if (!conv(equivType(T1, restrict(A, *m)), restrict(c1.v2, *m), restrict(c2.v2, *m))) return false;
        }
    return true;
}

static V convNe(const V& a, const V& b);

static bool sameSplit(const V& s1, const V& s2) {
    if (s1 == s2) return true;
    if (s1->split != s2->split) return false;
    // Same split term: compare the captured environments as normal forms.
    // Level bindings are skipped: elaboration leaves levels rigid (no
    // binding) while an instantiated global binds them; the comparison is
    // typed by the caller, which already determines the levels.
    auto skipLevels = [](const Env* e) {
        while (e && e->kind == Env::Kind::Level) e = e->next.get();
        return e;
    };
    const Env* e1 = skipLevels(s1->clo.env.get());
    const Env* e2 = skipLevels(s2->clo.env.get());
    while (e1 && e2) {
        if (e1->kind != e2->kind || e1->name != e2->name) return false;
        if (e1->kind == Env::Kind::Value && e1->v != e2->v &&
            printSx(quote(e1->v)) != printSx(quote(e2->v)))
            return false;
        if (e1->kind == Env::Kind::Dim && e1->f != e2->f) return false;
        e1 = skipLevels(e1->next.get());
        e2 = skipLevels(e2->next.get());
    }
    return !e1 && !e2;
}

static V convNe(const V& a, const V& b) {
    if (a->k != b->k) return nullptr;
    switch (a->k) {
        case Val::K::Var:
            return a->id == b->id ? a->a : nullptr;
        case Val::K::App: {
            V T = convNe(a->a, b->a);
            if (!T || T->k != Val::K::Pi) return nullptr;
            if (!conv(T->a, a->b, b->b)) return nullptr;
            return piCod(T, a->b);
        }
        case Val::K::Fst: {
            V T = convNe(a->a, b->a);
            return T ? piDom(T) : nullptr;
        }
        case Val::K::Snd: {
            V T = convNe(a->a, b->a);
            return T ? piCod(T, vfst(a->a)) : nullptr;
        }
        case Val::K::PApp: {
            if (a->f != b->f) return nullptr;
            V T = convNe(a->a, b->a);
            if (!T || T->k != Val::K::PathP) return nullptr;
            return lineAt(T->a, a->f);
        }
        case Val::K::SplitApp: {
            if (!sameSplit(a->a, b->a)) return nullptr;
            if (!convNe(a->b, b->b)) return nullptr;
            return motiveOf(a->a, a->b);
        }
        case Val::K::Unglue: {
            if (!convType(a->b, b->b)) return nullptr;
            if (!convNe(a->a, b->a)) return nullptr;
            return a->b;
        }
        case Val::K::HComp: {
            if (!convType(a->a, b->a)) return nullptr;
            if (!conv(a->a, a->b, b->b)) return nullptr;
            if (!convSysLines(a->a, a->sys, b->sys)) return nullptr;
            return a->a;
        }
        case Val::K::Transp: {
            if (a->f != b->f) return nullptr;
            Dim k = freshDim();
            if (!convType(lineAt(a->a, Formula::atom(k)), lineAt(b->a, Formula::atom(k)))) return nullptr;
            V A0 = lineAt(a->a, Formula::zero());
            if (!conv(A0, a->b, b->b)) return nullptr;
            return lineAt(a->a, Formula::one());
        }
        default:
            return nullptr;
    }
}

bool conv(const V& T, const V& a, const V& b) {
    if (a == b) return true;
    switch (T->k) {
        case Val::K::Pi: {
            V x = mkVar("x", T->a);
            return conv(piCod(T, x), app(a, x), app(b, x));
        }
        case Val::K::Sigma: {
            V a1 = vfst(a);
            return conv(T->a, a1, vfst(b)) && conv(piCod(T, a1), vsnd(a), vsnd(b));
        }
        case Val::K::PathP: {
            Formula k = Formula::atom(freshDim());
            return conv(lineAt(T->a, k), papp(a, k, T->b, T->c), papp(b, k, T->b, T->c));
        }
        case Val::K::U:
            return convType(a, b);
        case Val::K::Glue: {
            if (!conv(T->a, unglue(a, T->a, T->sys), unglue(b, T->a, T->sys))) return false;
            for (auto& c : T->sys)
                if (!conv(c.v, restrict(a, c.face), restrict(b, c.face))) return false;
            return true;
        }
        case Val::K::Data: {
            if (a->k == Val::K::Con && b->k == Val::K::Con) {
                if (a->con != b->con || a->dims != b->dims) return false;
                std::vector<V> earlier;
                for (size_t i = 0; i < a->args.size(); ++i) {
                    V ty = conArgType(T->data, T->levels, T->params, a->con, earlier);
                    if (!conv(ty, a->args[i], b->args[i])) return false;
                    earlier.push_back(a->args[i]);
                }
                return true;
            }
            if (a->k == Val::K::HComp && b->k == Val::K::HComp)
                return conv(T, a->b, b->b) && convSysLines(T, a->sys, b->sys);
            if (isStuck(a) && isStuck(b)) return convNe(a, b) != nullptr;
            return false;
        }
        default:
            break;
    }
    if (isStuck(a) && isStuck(b)) return convNe(a, b) != nullptr;
    return false;
}

bool convType(const V& A, const V& B) {
    if (A == B) return true;
    if (A->k != B->k) return false;
    switch (A->k) {
        case Val::K::U: return A->level == B->level;
        case Val::K::Pi:
        case Val::K::Sigma: {
            if (!convType(A->a, B->a)) return false;
            V x = mkVar(A->name.empty() ? "x" : A->name, A->a);
            return convType(piCod(A, x), piCod(B, x));
        }
        case Val::K::PathP: {
            Formula k = Formula::atom(freshDim());
            if (!convType(lineAt(A->a, k), lineAt(B->a, k))) return false;
            return conv(lineAt(A->a, Formula::zero()), A->b, B->b) &&
                   conv(lineAt(A->a, Formula::one()), A->c, B->c);
        }
        case Val::K::Data: {
            if (A->data != B->data || A->levels.size() != B->levels.size()) return false;
            for (size_t i = 0; i < A->levels.size(); ++i)
                if (!(A->levels[i] == B->levels[i])) return false;
            std::vector<V> earlier;
            for (size_t i = 0; i < A->params.size(); ++i) {
                V ty = paramType(A->data, A->levels, earlier);
                if (!conv(ty, A->params[i], B->params[i])) return false;
                earlier.push_back(A->params[i]);
            }
            return true;
        }
        case Val::K::Glue:
            return convType(A->a, B->a) && convGlueSys(A->a, A->sys, B->sys);
        default:
            break;
    }
    if (isStuck(A) && isStuck(B)) return convNe(A, B) != nullptr;
    return false;
}

bool subtype(const V& A, const V& B) {
    if (A->k == Val::K::U && B->k == Val::K::U) return A->level.leq(B->level);
    if (A->k == Val::K::Pi && B->k == Val::K::Pi) {
        if (!convType(A->a, B->a)) return false;
        V x = mkVar(A->name.empty() ? "x" : A->name, A->a);
        return subtype(piCod(A, x), piCod(B, x));
    }
    return convType(A, B);
}

// ---------------------------------------------------------------------------
// Read-back
// ---------------------------------------------------------------------------

namespace {

struct Quoter {
    std::map<Dim, std::string> dimNames;
    int counter = 0;

    std::string freshName(const std::string& hint) {
        std::string base = hint.empty() ? "x" : hint;
        return base + "." + std::to_string(counter++);
    }
    std::string dimName(Dim d) {
        auto it = dimNames.find(d);
        if (it != dimNames.end()) return it->second;
        std::string n = "i." + std::to_string(counter++);
        dimNames[d] = n;
        return n;
    }
    Sx formula(const Formula& f) {
        if (f.isZero()) return Sx::symbol("i0");
        if (f.isOne()) return Sx::symbol("i1");
        std::vector<Sx> disj;
        for (auto& c : f.dnf) {
            std::vector<Sx> conj;
            for (auto& l : c) {
                Sx a = Sx::symbol(dimName(l.d));
                conj.push_back(l.pos ? a : Sx::list({Sx::symbol("~"), a}));
            }
            if (conj.size() == 1) disj.push_back(conj[0]);
            else {
                conj.insert(conj.begin(), Sx::symbol("min"));
                disj.push_back(Sx::list(conj));
            }
        }
        if (disj.size() == 1) return disj[0];
        disj.insert(disj.begin(), Sx::symbol("max"));
        return Sx::list(disj);
    }
    Sx face(const Face& f) {
        std::vector<Sx> cs;
        for (auto& [d, b] : f)
            cs.push_back(Sx::list({Sx::symbol("="), Sx::symbol(dimName(d)), Sx::number(b ? "1" : "0")}));
        if (cs.empty()) return Sx::symbol("i1");
        if (cs.size() == 1) return cs[0];
        cs.insert(cs.begin(), Sx::symbol("and"));
        return Sx::list(cs);
    }
    Sx lineBinder(const V& line, Sx& body) {
        Dim k = freshDim();
        std::string n = dimName(k);
        body = q(lineAt(line, Formula::atom(k)));
        return Sx::symbol(n);
    }
    Sx sys(const VSys& s, bool lines, Sx* binder) {
        std::vector<Sx> out;
        Dim k = freshDim();
        if (binder) *binder = Sx::symbol(dimName(k));
        for (auto& c : s) {
            std::vector<Sx> clause{face(c.face)};
            clause.push_back(lines ? q(lineAt(c.v, Formula::atom(k))) : q(c.v));
            if (c.v2) clause.push_back(q(c.v2));
            out.push_back(Sx::list(clause));
        }
        return Sx::list(out);
    }

    Sx q(const V& v) {
        switch (v->k) {
            case Val::K::U:
                if (v->level == Level::constant(0)) return Sx::symbol("U");
                return Sx::list({Sx::symbol("U"), Sx::symbol(v->level.str())});
            case Val::K::Pi:
            case Val::K::Sigma: {
                std::string n = freshName(v->name);
                V x = mkVar(n, v->a);
                Sx dom = q(v->a);
                Sx cod = q(piCod(v, x));
                return Sx::list({Sx::symbol(v->k == Val::K::Pi ? "Pi" : "Sigma"),
                                 Sx::list({Sx::list({Sx::symbol(n), Sx::symbol(":"), dom})}), cod});
            }
            case Val::K::Lam: {
                std::string n = freshName(v->name);
                V x = mkVar(n, nullptr);
                return Sx::list({Sx::symbol("lambda"), Sx::list({Sx::symbol(n)}), q(app(v, x))});
            }
            case Val::K::Pair: return Sx::list({Sx::symbol("pair"), q(v->a), q(v->b)});
            case Val::K::PathP: {
                Sx body;
                Sx b = lineBinder(v->a, body);
                return Sx::list({Sx::symbol("PathP"), Sx::list({b}), body, q(v->b), q(v->c)});
            }
            case Val::K::PLam: {
                Sx body;
                Sx b = lineBinder(v, body);
                return Sx::list({Sx::symbol("lambda"), Sx::list({b}), body});
            }
            case Val::K::Data: {
                if (v->params.empty() && v->levels.empty()) return Sx::symbol(v->data->name);
                std::vector<Sx> items{Sx::symbol(v->data->name)};
                for (auto& l : v->levels) items.push_back(Sx::symbol(l.str()));
                for (auto& p : v->params) items.push_back(q(p));
                return Sx::list(items);
            }
            case Val::K::Con: {
                const Signature& s = sig();
                if (v->data == s.natural) {
                    uint64_t n = 0;
                    V cur = v;
                    while (cur->k == Val::K::Con && cur->con == s.natSuc) { ++n; cur = cur->args[0]; }
                    if (cur->k == Val::K::Con && cur->con == s.natZero) return Sx::number(std::to_string(n));
                }
                const std::string& name = v->data->cons[v->con].name;
                if (v->args.empty() && v->dims.empty()) return Sx::symbol(name);
                std::vector<Sx> items{Sx::symbol(name)};
                for (auto& a : v->args) items.push_back(q(a));
                for (auto& d : v->dims) items.push_back(formula(d));
                return Sx::list(items);
            }
            case Val::K::Split: return Sx::symbol("#<split>");
            case Val::K::Glue: return Sx::list({Sx::symbol("Glue"), q(v->a), sys(v->sys, false, nullptr)});
            case Val::K::GlueElem: return Sx::list({Sx::symbol("glue"), sys(v->sys, false, nullptr), q(v->a)});
            case Val::K::HComp: {
                Sx binder;
                Sx s = sys(v->sys, true, &binder);
                return Sx::list({Sx::symbol("hcomp"), q(v->a), Sx::list({binder}), s, q(v->b)});
            }
            case Val::K::Transp: {
                Sx body;
                Sx b = lineBinder(v->a, body);
                return Sx::list({Sx::symbol("transp"), Sx::list({b}), body, formula(v->f), q(v->b)});
            }
            case Val::K::Var: return Sx::symbol(!v->name.empty() && v->name[0] == '@' ? v->name.substr(1) : v->name);
            case Val::K::App: {
                Sx f = q(v->a);
                if (f.isList() && v->a->k == Val::K::App) {
                    f.items.push_back(q(v->b));
                    return f;
                }
                return Sx::list({f, q(v->b)});
            }
            case Val::K::Fst: return Sx::list({Sx::symbol("fst"), q(v->a)});
            case Val::K::Snd: return Sx::list({Sx::symbol("snd"), q(v->a)});
            case Val::K::PApp: return Sx::list({q(v->a), formula(v->f)});
            case Val::K::SplitApp: return Sx::list({q(v->a), q(v->b)});
            case Val::K::Unglue: return Sx::list({Sx::symbol("unglue"), q(v->a)});
        }
        return Sx::symbol("?");
    }
};

}  // namespace

Sx quote(const V& v) {
    Quoter qt;
    return qt.q(v);
}

std::string show(const V& v) {
    try {
        return printSx(quote(v));
    } catch (const std::exception&) {
        return "<value>";
    }
}

}  // namespace eshkol::hott::cubical
