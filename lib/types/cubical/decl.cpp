/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Declarations and the public API of the cubical kernel (ADR-0043).
 *
 *   (data Name ctor ...)                          an inductive type
 *   (data (Name (l : Level) (A : (U l)) ...) : (U l) ctor ...)
 *       ctor ::= name | (name (x : A) ... [(dims i ...)] [(boundary (face t) ...)])
 *   (define name : T body)
 *   (define (name (x : A) ...) : T body)         levels first: (l : Level)
 *   (example T body)                              checked, not named
 *   (import module)                               a HoTT library module
 *   (builtin natural Nat)                         numerals denote Nat
 *
 * The kernel prelude below defines the notions the kernel itself relies on:
 * contractibility, fibers, equivalences (CCHM section 5), the identity
 * equivalence and the equivalence of a line of types, used to compose in a
 * universe (CCHM section 7.2).
 */

#include "cubical_internal.h"
#include "elab.h"

#include <algorithm>

namespace eshkol::hott::cubical {

namespace {

[[noreturn]] void fail(const Sx& at, const std::string& msg) {
    throw KernelError(msg, at.line, at.column);
}

bool termMentions(const TermP& t, const DataDecl* d) {
    if (!t) return false;
    if ((t->k == Term::K::Data || t->k == Term::K::Con) && t->data == d) return true;
    for (auto* p : {&t->a, &t->b, &t->c, &t->d})
        if (termMentions(*p, d)) return true;
    for (auto& p : t->params) if (termMentions(p, d)) return true;
    for (auto& p : t->args) if (termMentions(p, d)) return true;
    for (auto& c : t->sys) if (termMentions(c.t, d) || termMentions(c.t2, d)) return true;
    for (auto& c : t->clauses) if (termMentions(c.body, d)) return true;
    return false;
}

bool strictlyPositive(const TermP& t, const DataDecl* d) {
    if (!termMentions(t, d)) return true;
    if (t->k == Term::K::Data && t->data == d) return true;
    if (t->k == Term::K::Pi) return !termMentions(t->a, d) && strictlyPositive(t->b, d);
    // a path type in D is a function out of the interval: positive when its
    // line is; the endpoints are elements, not occurrences of the type
    if (t->k == Term::K::PathP) return t->a->k == Term::K::PLam && strictlyPositive(t->a->a, d);
    return false;
}

// Structural boundary terms: constructors of the type being declared, applied
// to the constructor's own arguments (or to such terms) and dimensions.
// Transport commutes with such constructors; for other boundaries (paths
// among the arguments, as in set truncation) transpData adds the CHM18
// correcting composition.
bool constructorExpression(const TermP& t, const DataDecl* d, const std::set<std::string>& args) {
    if (t->k == Term::K::Var) return args.count(t->name) > 0;
    if (t->k == Term::K::Con && t->data == d) {
        for (auto& a : t->args)
            if (!constructorExpression(a, d, args)) return false;
        return true;
    }
    return false;
}

const char* kKernelPrelude = R"HOTT(
(define (isContr (l : Level) (A : (U l))) : (U l)
  (Sigma ((c : A)) (Pi ((y : A)) (Path A c y))))
(define (fiber (l k : Level) (A : (U l)) (B : (U k)) (f : (-> A B)) (y : B)) : (U (lmax l k))
  (Sigma ((x : A)) (Path B (f x) y)))
(define (isEquiv (l k : Level) (A : (U l)) (B : (U k)) (f : (-> A B))) : (U (lmax l k))
  (Pi ((y : B)) (isContr (lmax l k) (fiber l k A B f y))))
(define (Equiv (l k : Level) (A : (U l)) (B : (U k))) : (U (lmax l k))
  (Sigma ((f : (-> A B))) (isEquiv l k A B f)))
(define (idEquiv (l : Level) (A : (U l))) : (Equiv l l A A)
  (pair (lambda (x) x)
        (lambda (y)
          (pair (pair y (lambda (i) y))
                (lambda (z)
                  (lambda (i)
                    (pair ((snd z) (~ i))
                          (lambda (j) ((snd z) (max (~ i) j))))))))))
(define (path-to-equiv (l : Level) (A B : (U l)) (P : (Path (U l) A B))) : (Equiv l l A B)
  (transp (i) (Equiv l l A (P i)) i0 (idEquiv l A)))
)HOTT";

}  // namespace

void Elab::checkNameFree(const Sx& at, const std::string& name) {
    if (name.empty() || name[0] == '%') fail(at, "invalid name `" + name + "`");
    static const std::set<std::string> reserved = {
        "lambda", "λ", "Pi", "->", "Sigma", "pair", "fst", "snd", "Path", "PathP", "the", "transp",
        "hcomp", "comp", "Glue", "glue", "unglue", "split", "U", "let", "i0", "i1", "~", "min", "max",
        "Level", "lsuc", "lmax", ":", "=", "and", "or", "data", "define", "example", "import", "builtin",
        "dims", "boundary"};
    if (reserved.count(name)) fail(at, "`" + name + "` is reserved");
    if (S.globals.count(name) || S.datas.count(name) || S.constructors.count(name))
        fail(at, "`" + name + "` is already defined");
}

void Elab::declare(const Sx& decl) {
    if (!decl.isList() || decl.items.empty() || !decl.items[0].isSymbol())
        fail(decl, "expected a declaration: (data ...), (define ...), (example ...), (import ...)");
    const std::string& h = decl.items[0].text;
    if (h == "data") return declData(decl);
    if (h == "define") return declDefine(decl, false);
    if (h == "example") return declDefine(decl, true);
    if (h == "builtin") return declBuiltin(decl);
    if (h == "import") {
        if (decl.items.size() != 2 || !decl.items[1].isSymbol()) fail(decl, "(import module)");
        if (!importModule) fail(decl, "imports are not available here");
        importModule(decl.items[1].text);
        return;
    }
    fail(decl.items[0], "unknown declaration `" + h + "`");
}

void Elab::declBuiltin(const Sx& sx) {
    if (sx.items.size() != 3 || !sx.items[1].isSymbol("natural") || !sx.items[2].isSymbol())
        fail(sx, "(builtin natural Nat)");
    auto it = S.datas.find(sx.items[2].text);
    if (it == S.datas.end()) fail(sx.items[2], "unknown data type " + sx.items[2].text);
    const DataDecl* d = it->second.get();
    int zero = -1, suc = -1;
    for (size_t i = 0; i < d->cons.size(); ++i) {
        const ConDecl& c = d->cons[i];
        if (c.args.empty() && c.dims.empty()) zero = (int)i;
        else if (c.args.size() == 1 && c.recursive[0] && c.dims.empty()) suc = (int)i;
    }
    if (d->cons.size() != 2 || zero < 0 || suc < 0 || !d->params.empty())
        fail(sx, "a natural-number type has exactly a constant and a successor constructor");
    S.natural = d;
    S.natZero = zero;
    S.natSuc = suc;
}

void Elab::declData(const Sx& sx) {
    const auto& it = sx.items;
    if (it.size() < 2) fail(sx, "(data Name ctor ...)");
    std::string name;
    std::vector<Sx> header;
    if (it[1].isSymbol()) name = it[1].text;
    else if (it[1].isList() && !it[1].items.empty() && it[1].items[0].isSymbol()) {
        name = it[1].items[0].text;
        header.assign(it[1].items.begin() + 1, it[1].items.end());
    } else fail(it[1], "(data Name ...) or (data (Name params ...) ...)");
    checkNameFree(it[1], name);
    size_t next = 2;
    Sx universe = Sx::symbol("U");
    if (next + 1 < it.size() + 1 && next < it.size() && it[next].isSymbol(":")) {
        if (next + 1 >= it.size()) fail(sx, "expected a universe after :");
        universe = it[next + 1];
        next += 2;
    }

    auto dd = std::make_shared<DataDecl>();
    dd->name = name;
    size_t pushed = 0;
    auto cleanup = [&] {
        for (size_t i = 0; i < pushed; ++i) pop();
        dataInProgress = nullptr;
        dataParamInternals.clear();
    };
    try {
        for (auto& b : header) {
            if (!b.isList() || b.items.size() < 3 || !b.items[b.items.size() - 2].isSymbol(":"))
                fail(b, "a parameter is written (x : A)");
            const Sx& ty = b.items.back();
            for (size_t i = 0; i + 2 < b.items.size(); ++i) {
                const std::string& pn = b.items[i].text;
                if (ty.isSymbol("Level")) {
                    if (!dd->params.empty()) fail(b, "level parameters come first");
                    bindLevel(pn);
                    ++pushed;
                    dd->levelParams.push_back(pn);
                } else {
                    auto [A, l] = inferType(ty);
                    (void)l;
                    V x = bindVar(pn, evalT(A));
                    ++pushed;
                    dd->params.push_back({"v#" + std::to_string(x->id), A});
                }
            }
        }
        // universe
        if (universe.isSymbol("U")) {
            auto l = std::make_shared<LExpr>();
            l->k = LExpr::K::Const;
            dd->universe = l;
        } else if (universe.isList() && universe.items.size() == 2 && universe.items[0].isSymbol("U")) {
            dd->universe = level(universe.items[1]);
        } else {
            fail(universe, "the universe of a data type is U or (U level)");
        }
        Level lev = evalLevel(env(), dd->universe);
        S.datas[name] = dd;
        dataInProgress = dd.get();
        for (auto& p : dd->params) dataParamInternals.push_back(p.first);
        std::vector<Level> rigid;
        for (auto& l : dd->levelParams) rigid.push_back(Level::var(l));
        std::vector<V> paramVals;
        for (auto& p : dd->params) {
            Elab::Entry const* e = nullptr;
            for (auto& en : ctx)
                if (en.internal == p.first) e = &en;
            paramVals.push_back(e->value);
        }

        for (size_t ci = next; ci < it.size(); ++ci) {
            const Sx& c = it[ci];
            ConDecl cd;
            std::vector<Sx> parts;
            if (c.isSymbol()) cd.name = c.text;
            else if (c.isList() && !c.items.empty() && c.items[0].isSymbol()) {
                cd.name = c.items[0].text;
                parts.assign(c.items.begin() + 1, c.items.end());
            } else fail(c, "a constructor is name or (name (x : A) ... (dims ...) (boundary ...))");
            checkNameFree(c, cd.name);
            for (auto& o : dd->cons)
                if (o.name == cd.name) fail(c, "constructor `" + cd.name + "` declared twice");
            size_t cpushed = 0;
            const Sx* boundary = nullptr;
            std::set<std::string> argNames;
            try {
                for (auto& p : parts) {
                    if (p.isList() && !p.items.empty() && p.items[0].isSymbol("dims")) {
                        if (!cd.dims.empty()) fail(p, "dims given twice");
                        for (size_t k = 1; k < p.items.size(); ++k) {
                            if (!p.items[k].isSymbol()) fail(p.items[k], "dimension names are symbols");
                            Dim a = bindDim(p.items[k].text);
                            ++cpushed;
                            cd.dims.push_back("i#" + std::to_string(a));
                        }
                        continue;
                    }
                    if (p.isList() && !p.items.empty() && p.items[0].isSymbol("boundary")) {
                        boundary = &p;
                        continue;
                    }
                    if (!cd.dims.empty() || boundary) fail(p, "constructor arguments come before (dims ...)");
                    if (!p.isList() || p.items.size() < 3 || !p.items[p.items.size() - 2].isSymbol(":"))
                        fail(p, "a constructor argument is written (x : A)");
                    for (size_t k = 0; k + 2 < p.items.size(); ++k) {
                        auto [A, l] = inferType(p.items.back());
                        if (!l.leq(lev))
                            fail(p.items.back(), "constructor argument type lives in universe " + l.str() +
                                                     ", above the universe " + lev.str() + " of " + name);
                        if (!strictlyPositive(A, dd.get()))
                            fail(p.items.back(), "`" + name + "` occurs in a non-strictly-positive position");
                        bool rec = A->k == Term::K::Data && A->data == dd.get();
                        V x = bindVar(p.items[k].text, evalT(A));
                        ++cpushed;
                        std::string in = "v#" + std::to_string(x->id);
                        cd.args.push_back({in, A});
                        cd.recursive.push_back(rec);
                        argNames.insert(in);
                    }
                }
                if (boundary && cd.dims.empty()) fail(*boundary, "a boundary needs (dims ...)");
                if (boundary) {
                    V D = mkData(dd.get(), rigid, paramVals);
                    struct Done { Face f; TermP t; };
                    std::vector<Done> done;
                    for (size_t k = 1; k < boundary->items.size(); ++k) {
                        const Sx& cl = boundary->items[k];
                        if (!cl.isList() || cl.items.size() != 2) fail(cl, "a boundary clause is (face term)");
                        IExprP fe = face(cl.items[0]);
                        std::function<bool(const IExprP&)> onlyDims = [&](const IExprP& e) -> bool {
                            if (!e) return true;
                            if (e->k == IExpr::K::Var)
                                return std::find(cd.dims.begin(), cd.dims.end(), e->name) != cd.dims.end();
                            return onlyDims(e->a) && onlyDims(e->b);
                        };
                        if (!onlyDims(fe)) fail(cl.items[0], "a boundary face mentions only the constructor's dimensions");
                        auto fs = facesOf(evalI(env(), fe));
                        if (fs.empty()) continue;
                        TermP bt;
                        for (auto& f : fs) {
                            underFace(f, [&] {
                                TermP t = check(cl.items[1], D);
                                if (!constructorExpression(t, dd.get(), argNames))
                                    cd.structuralBoundary = false;
                                if (!bt) bt = t;
                                done.push_back({f, t});
                            });
                        }
                        cd.boundary.push_back({fe, bt});
                    }
                    for (size_t a = 0; a < done.size(); ++a)
                        for (size_t b = a + 1; b < done.size(); ++b) {
                            auto m = faceMeet(done[a].f, done[b].f);
                            if (!m) continue;
                            V va = restrict(evalT(done[a].t), *m), vb = restrict(evalT(done[b].t), *m);
                            if (!conv(D, va, vb)) fail(*boundary, "boundary clauses disagree where their faces overlap");
                        }
                }
            } catch (...) {
                for (size_t i = 0; i < cpushed; ++i) pop();
                throw;
            }
            for (size_t i = 0; i < cpushed; ++i) pop();
            if (!cd.dims.empty()) dd->higher = true;
            int idx = (int)dd->cons.size();
            S.constructors[cd.name] = {dd.get(), idx};
            dd->cons.push_back(std::move(cd));
        }
    } catch (...) {
        cleanup();
        throw;
    }
    cleanup();
}

void Elab::declDefine(const Sx& sx, bool example) {
    const auto& it = sx.items;
    std::string name;
    std::vector<Sx> header;
    Sx typeSx, bodySx;
    if (example) {
        if (it.size() != 3) fail(sx, "(example T body)");
        typeSx = it[1];
        bodySx = it[2];
    } else {
        if (it.size() != 5 || !it[2].isSymbol(":")) fail(sx, "(define name : T body) or (define (name (x : A) ...) : T body)");
        if (it[1].isSymbol()) name = it[1].text;
        else if (it[1].isList() && !it[1].items.empty() && it[1].items[0].isSymbol()) {
            name = it[1].items[0].text;
            header.assign(it[1].items.begin() + 1, it[1].items.end());
        } else fail(it[1], "definition name");
        checkNameFree(it[1], name);
        typeSx = it[3];
        bodySx = it[4];
    }

    std::vector<std::string> levelParams;
    struct B { std::string internal; TermP type; };
    std::vector<B> bs;
    size_t pushed = 0;
    auto unwind = [&] {
        for (size_t i = 0; i < pushed; ++i) pop();
        selfName.clear();
        selfArgPos = -1;
        topSpine = false;
        spineArgs = 0;
        smaller.clear();
    };
    try {
        for (auto& b : header) {
            if (!b.isList() || b.items.size() < 3 || !b.items[b.items.size() - 2].isSymbol(":"))
                fail(b, "a parameter is written (x : A)");
            const Sx& ty = b.items.back();
            for (size_t i = 0; i + 2 < b.items.size(); ++i) {
                if (!b.items[i].isSymbol()) fail(b.items[i], "parameter names are symbols");
                if (ty.isSymbol("Level")) {
                    if (!bs.empty()) fail(b, "level parameters come first");
                    bindLevel(b.items[i].text);
                    ++pushed;
                    levelParams.push_back(b.items[i].text);
                } else {
                    auto [A, l] = inferType(ty);
                    (void)l;
                    V x = bindVar(b.items[i].text, evalT(A));
                    ++pushed;
                    bs.push_back({"v#" + std::to_string(x->id), A});
                }
            }
        }
        auto [T, lt] = inferType(typeSx);
        (void)lt;
        V Tv = evalT(T);

        // the full type and the global (opaque while its body is checked)
        TermP full = T;
        for (size_t k = bs.size(); k-- > 0;) {
            auto p = std::make_shared<Term>();
            p->k = Term::K::Pi;
            p->name = bs[k].internal;
            p->a = bs[k].type;
            p->b = full;
            full = p;
        }
        std::shared_ptr<GlobalDef> g;
        if (!example) {
            g = std::make_shared<GlobalDef>();
            g->name = name;
            g->levelParams = levelParams;
            g->type = full;
            // the definition stands for itself while its body is checked; it
            // reads back to the global, so terms mentioning it stay closed
            {
                auto op = std::make_shared<Val>(*mkVar("@" + name, eval(EnvP{}, full)));
                for (auto& lp : levelParams) op->levels.push_back(Level::var(lp));
                g->opaque = op;
            }
            S.globals[name] = g;
            selfName = name;
        }
        selfArgPos = -1;
        topSpine = true;
        spineArgs = (int)bs.size();
        TermP body = check(bodySx, Tv);
        for (size_t k = bs.size(); k-- > 0;) {
            auto l = std::make_shared<Term>();
            l->k = Term::K::Lam;
            l->name = bs[k].internal;
            l->a = body;
            body = l;
        }
        unwind();
        pushed = 0;
        if (g) {
            g->body = body;
            g->cache.clear();
        }
        runDeferred();
    } catch (...) {
        unwind();
        deferred.clear();
        throw;
    }
}

// ---------------------------------------------------------------------------
// Kernel
// ---------------------------------------------------------------------------

#include "hott_library.inc"

struct Kernel::Impl {
    Signature sig;
    std::set<std::string> imported;
    std::string preludeError;

    void declareAll(Elab& e, const std::vector<Sx>& decls) {
        for (auto& d : decls) e.declare(d);
    }

    void importModule(const std::string& name) {
        if (imported.count(name)) return;
        const char* text = nullptr;
        for (auto& m : kHottLibrary)
            if (name == m.name) text = m.text;
        if (!text) throw KernelError("unknown HoTT library module `" + name + "`");
        imported.insert(name);
        std::vector<Sx> forms;
        std::string err;
        if (!readSx(text, forms, err)) throw KernelError("library module " + name + ": " + err);
        for (auto& f : forms) {
            if (!f.isList() || f.items.empty() || !f.items[0].isSymbol("hott")) continue;
            std::vector<Sx> decls(f.items.begin() + 1, f.items.end());
            Elab e(sig);
            e.importModule = [this](const std::string& n) { importModule(n); };
            try {
                declareAll(e, decls);
            } catch (const KernelError& k) {
                throw KernelError("in HoTT library module `" + name + "`: " + k.what(), k.line, k.column);
            }
        }
    }
};

Kernel::Kernel() : impl_(std::make_unique<Impl>()) {
    SignatureScope scope(&impl_->sig);
    std::vector<Sx> forms;
    std::string err;
    if (!readSx(kKernelPrelude, forms, err)) {
        impl_->preludeError = "kernel prelude: " + err;
        return;
    }
    try {
        Elab e(impl_->sig);
        for (auto& f : forms) e.declare(f);
    } catch (const KernelError& k) {
        impl_->preludeError = std::string("kernel prelude: ") + k.what();
    }
}

Kernel::~Kernel() = default;

std::vector<std::string> Kernel::libraryModules() {
    std::vector<std::string> out;
    for (auto& m : kHottLibrary) out.push_back(m.name);
    return out;
}

KernelResult Kernel::declare(const std::vector<Sx>& decls) {
    KernelResult r;
    if (!impl_->preludeError.empty()) {
        r.ok = false;
        r.message = impl_->preludeError;
        return r;
    }
    SignatureScope scope(&impl_->sig);
    for (auto& d : decls) {
        Signature snapshot = impl_->sig;
        auto importedSnapshot = impl_->imported;
        try {
            Elab e(impl_->sig);
            e.importModule = [this](const std::string& n) { impl_->importModule(n); };
            e.declare(d);
        } catch (const KernelError& k) {
            impl_->sig = snapshot;
            impl_->imported = importedSnapshot;
            r.ok = false;
            r.message = k.what();
            r.line = k.line ? k.line : d.line;
            r.column = k.line ? k.column : d.column;
            return r;
        } catch (const std::exception& ex) {
            impl_->sig = snapshot;
            impl_->imported = importedSnapshot;
            r.ok = false;
            r.message = std::string("HoTT kernel: ") + ex.what();
            r.line = d.line;
            r.column = d.column;
            return r;
        }
    }
    return r;
}

KernelResult Kernel::evaluate(const Sx& term) {
    KernelResult r;
    if (!impl_->preludeError.empty()) {
        r.ok = false;
        r.message = impl_->preludeError;
        return r;
    }
    SignatureScope scope(&impl_->sig);
    try {
        Elab e(impl_->sig);
        auto [t, T] = e.infer(term);
        V v = eval(EnvP{}, t);
        r.value = quote(v);
        r.type = show(T);
    } catch (const KernelError& k) {
        r.ok = false;
        r.message = k.what();
        r.line = k.line ? k.line : term.line;
        r.column = k.line ? k.column : term.column;
    } catch (const std::exception& ex) {
        r.ok = false;
        r.message = std::string("HoTT kernel: ") + ex.what();
        r.line = term.line;
        r.column = term.column;
    }
    return r;
}

}  // namespace eshkol::hott::cubical
