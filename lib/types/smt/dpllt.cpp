/**
 * @file dpllt.cpp
 * @brief DPLL(T) search with delayed theory combination of linear arithmetic
 *        and congruence closure (ADR-0044).
 *
 * Encoding. Every numeric term is purified into a linear form over arithmetic
 * variables: an `ite` becomes a fresh variable constrained on both branches,
 * an application f(t...) becomes a fresh result variable plus a definition
 * r = f(v...) for the congruence closure, with each argument itself purified
 * to a variable. The Boolean structure is Tseitin-encoded; its atoms are
 * linear constraints, equalities between purified variables of sort Obj, and
 * Boolean variables.
 *
 * An arithmetic equality atom e: (t = c) carries two helper atoms lt: (t < c)
 * and gt: (t > c) with the clauses (~e | ~lt), (~e | ~gt), (e | lt | gt), so a
 * false equality reaches the arithmetic engine as a strict inequality chosen
 * by the search, and the engine only ever sees conjunctions.
 *
 * Combination. Every pair of numeric variables that occur both in arithmetic
 * and in a function application gets an interface equality atom that is
 * shared by both theories. The search therefore fixes one arrangement of the
 * shared variables and each theory checks it independently; for stably
 * infinite theories over disjoint signatures this is complete (Nelson-Oppen,
 * in its delayed-combination form). Boolean variables that occur as
 * application arguments or results are related to the constants true/false
 * inside the congruence closure, so the finite Boolean sort needs no
 * interface atoms.
 *
 * Search. Chronological DPLL with unit propagation. After each propagation
 * fixpoint the assigned theory literals are checked; a theory conflict is
 * shrunk to a minimal core by deletion and learned as a clause. A complete
 * assignment that both theories accept yields a model, which is then
 * re-evaluated against the original assertions with exact semantics. If an
 * over-approximated symbol (a nonlinear product, Scheme division) makes that
 * evaluation fail, the search continues; if nothing else is found the answer
 * is Unknown, never Sat.
 */
#include "eshkol/types/smt.h"

#include <algorithm>
#include <set>
#include <sstream>

namespace eshkol::smt {

namespace {

struct BudgetOut {};

// ---------------------------------------------------------------------------
// Congruence closure
// ---------------------------------------------------------------------------

struct Def {
    int result;               // EUF node
    std::string fn;
    std::vector<int> args;    // EUF nodes
};

class Congruence {
public:
    explicit Congruence(int n) : parent_(n) {
        for (int i = 0; i < n; ++i) parent_[i] = i;
    }
    int find(int x) {
        while (parent_[x] != x) {
            parent_[x] = parent_[parent_[x]];
            x = parent_[x];
        }
        return x;
    }
    void unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) parent_[std::max(a, b)] = std::min(a, b);
    }
    // Close under congruence over @p defs.
    void close(const std::vector<Def>& defs) {
        bool changed = true;
        while (changed) {
            changed = false;
            std::map<std::string, int> sig;
            for (const auto& d : defs) {
                std::string key = d.fn + "(";
                for (int a : d.args) key += std::to_string(find(a)) + ",";
                auto it = sig.find(key);
                if (it == sig.end()) {
                    sig.emplace(key, d.result);
                } else if (find(it->second) != find(d.result)) {
                    unite(it->second, d.result);
                    changed = true;
                }
            }
        }
    }

private:
    std::vector<int> parent_;
};

// ---------------------------------------------------------------------------
// Theory atoms
// ---------------------------------------------------------------------------

enum class AtomKind : uint8_t { Plain, Arith, EufEq, BoolShared };

struct Atom {
    AtomKind kind = AtomKind::Plain;
    // Arith: form rel rhs
    std::map<uint32_t, Rational> form;
    Rel rel = Rel::Le;
    Rational rhs;
    // EufEq: nodes a == b (also set for interface equalities, which are Arith too)
    int euf_a = -1, euf_b = -1;
    bool interface = false;
    // BoolShared: EUF node of the Boolean variable
    int euf_node = -1;
};

struct TheoryLit {
    int lit;  // signed SAT literal as assigned
};

class Engine {
public:
    Engine(TermManager& tm, uint64_t max_steps) : tm_(tm), max_steps_(max_steps) {
        atoms_.emplace_back();  // variable 0 unused
        true_var_ = newVar();
        clauses_.push_back({true_var_});
        euf_true_ = newEufNode("%true", Sort::Bool);
        euf_false_ = newEufNode("%false", Sort::Bool);
    }

    uint64_t steps = 0;

    void assertFormula(Term t) { clauses_.push_back({encode(t)}); }

    Outcome run(const std::vector<Term>& originals) {
        addInterfaceEqualities();
        Outcome out;
        try {
            out = search(originals);
        } catch (const BudgetOut&) {
            out.result = Result::Unknown;
            out.reason = "step budget exhausted";
        }
        out.steps = steps;
        return out;
    }

private:
    TermManager& tm_;
    uint64_t max_steps_;
    std::vector<Atom> atoms_;
    std::vector<std::vector<int>> clauses_;
    int true_var_ = 0;

    // Arithmetic variables
    std::map<uint32_t, uint32_t> arith_of_term_;  // term id -> arith index
    std::vector<Term> arith_terms_;
    std::vector<bool> is_int_;

    // EUF nodes
    std::map<uint32_t, int> euf_of_term_;  // term id -> node
    std::vector<Term> euf_terms_;          // nullptr for the Boolean constants
    std::vector<Sort> euf_sorts_;
    int euf_true_ = -1, euf_false_ = -1;
    std::vector<Def> defs_;

    // Bool variables
    std::map<uint32_t, int> bool_var_of_term_;  // Bool Var term id -> SAT var
    std::vector<Term> bool_terms_;
    std::map<uint32_t, int> encoded_;           // formula term id -> SAT literal
    std::map<uint32_t, Term> purified_;         // app/ite term id -> purified variable term
    std::map<std::string, int> arith_atoms_;    // canonical text -> SAT var
    std::map<std::pair<int, int>, int> euf_atoms_;

    bool over_approximated_ = false;

    void tick(uint64_t n = 1) {
        steps += n;
        if (steps > max_steps_) throw BudgetOut{};
    }

    int newVar() {
        atoms_.emplace_back();
        return static_cast<int>(atoms_.size() - 1);
    }

    int newEufNode(const std::string&, Sort s, Term t = nullptr) {
        euf_terms_.push_back(t);
        euf_sorts_.push_back(s);
        return static_cast<int>(euf_terms_.size() - 1);
    }

    int eufNode(Term var) {
        auto it = euf_of_term_.find(var->id);
        if (it != euf_of_term_.end()) return it->second;
        int n = newEufNode(var->name, var->sort, var);
        euf_of_term_.emplace(var->id, n);
        if (var->sort == Sort::Bool) {
            int v = boolVar(var);
            atoms_[v].kind = AtomKind::BoolShared;
            atoms_[v].euf_node = n;
        }
        return n;
    }

    uint32_t arithVar(Term var) {
        auto it = arith_of_term_.find(var->id);
        if (it != arith_of_term_.end()) return it->second;
        uint32_t i = static_cast<uint32_t>(arith_terms_.size());
        arith_terms_.push_back(var);
        is_int_.push_back(var->sort == Sort::Int);
        arith_of_term_.emplace(var->id, i);
        return i;
    }

    int boolVar(Term var) {
        auto it = bool_var_of_term_.find(var->id);
        if (it != bool_var_of_term_.end()) return it->second;
        int v = newVar();
        bool_var_of_term_.emplace(var->id, v);
        bool_terms_.push_back(var);
        return v;
    }

    // -- purification ------------------------------------------------------

    Term freshVar(const char* stem, Sort s) { return tm_.mkVar(tm_.freshName(stem), s); }

    // A variable term equal to @p t (t itself when it already is one).
    Term toVar(Term t) {
        if (t->kind == Kind::Var) return t;
        auto it = purified_.find(t->id);
        if (it != purified_.end()) return it->second;
        Term v;
        if (t->kind == Kind::App) {
            v = freshVar("app", t->sort);
            purified_.emplace(t->id, v);
            if (isOverApproximatedSymbol(t->name)) over_approximated_ = true;
            Def d;
            d.result = eufNode(v);
            d.fn = t->name;
            for (Term a : t->args) d.args.push_back(eufNode(toVar(a)));
            defs_.push_back(std::move(d));
            return v;
        }
        v = freshVar(t->kind == Kind::Ite ? "ite" : "val", t->sort);
        purified_.emplace(t->id, v);
        if (t->kind == Kind::Ite) {
            assertFormula(tm_.mkImplies(t->args[0], tm_.mkEq(v, t->args[1])));
            assertFormula(tm_.mkImplies(tm_.mkNot(t->args[0]), tm_.mkEq(v, t->args[2])));
        } else {
            assertFormula(tm_.mkEq(v, t));
        }
        return v;
    }

    struct Lin {
        std::map<uint32_t, Rational> c;
        Rational k;
    };

    void linearize(Term t, const Rational& scale, Lin& out) {
        switch (t->kind) {
            case Kind::Numeral:
                out.k += scale * t->value;
                return;
            case Kind::Add:
                for (Term a : t->args) linearize(a, scale, out);
                return;
            case Kind::Scale:
                linearize(t->args[0], scale * t->value, out);
                return;
            case Kind::Var: {
                uint32_t i = arithVar(t);
                out.c[i] += scale;
                if (out.c[i].isZero()) out.c.erase(i);
                return;
            }
            default: {
                Term v = toVar(t);
                uint32_t i = arithVar(v);
                out.c[i] += scale;
                if (out.c[i].isZero()) out.c.erase(i);
                return;
            }
        }
    }

    // -- Tseitin -----------------------------------------------------------

    int constLit(bool b) { return b ? true_var_ : -true_var_; }

    int arithAtom(const std::map<uint32_t, Rational>& form, Rel rel, const Rational& rhs) {
        if (form.empty()) {
            int s = Rational(0).compare(rhs);
            return constLit(rel == Rel::Le ? s <= 0 : (rel == Rel::Lt ? s < 0 : s == 0));
        }
        std::ostringstream key;
        for (const auto& [v, a] : form) key << v << '*' << a.str() << '+';
        key << (rel == Rel::Le ? "<=" : (rel == Rel::Lt ? "<" : "=")) << rhs.str();
        auto it = arith_atoms_.find(key.str());
        if (it != arith_atoms_.end()) return it->second;
        int v = newVar();
        atoms_[v].kind = AtomKind::Arith;
        atoms_[v].form = form;
        atoms_[v].rel = rel;
        atoms_[v].rhs = rhs;
        arith_atoms_.emplace(key.str(), v);
        if (rel == Rel::Eq) {
            std::map<uint32_t, Rational> neg;
            for (const auto& [x, a] : form) neg[x] = -a;
            int lt = arithAtom(form, Rel::Lt, rhs);
            int gt = arithAtom(neg, Rel::Lt, -rhs);
            clauses_.push_back({-v, -lt});
            clauses_.push_back({-v, -gt});
            clauses_.push_back({v, lt, gt});
        }
        return v;
    }

    int eufEqAtom(int a, int b) {
        if (a == b) return constLit(true);
        if (b < a) std::swap(a, b);
        auto it = euf_atoms_.find({a, b});
        if (it != euf_atoms_.end()) return it->second;
        int v = newVar();
        atoms_[v].kind = AtomKind::EufEq;
        atoms_[v].euf_a = a;
        atoms_[v].euf_b = b;
        euf_atoms_.emplace(std::make_pair(a, b), v);
        return v;
    }

    int encode(Term t) {
        auto it = encoded_.find(t->id);
        if (it != encoded_.end()) return it->second;
        int lit = encodeUncached(t);
        encoded_.emplace(t->id, lit);
        return lit;
    }

    int encodeUncached(Term t) {
        tick();
        switch (t->kind) {
            case Kind::BoolConst:
                return constLit(t->bval);
            case Kind::Var:
                return boolVar(t);
            case Kind::App:
            case Kind::Ite:
                return encode(toVar(t));
            case Kind::Not:
                return -encode(t->args[0]);
            case Kind::And:
            case Kind::Or: {
                std::vector<int> lits;
                for (Term a : t->args) lits.push_back(encode(a));
                int v = newVar();
                if (t->kind == Kind::And) {
                    std::vector<int> big{v};
                    for (int l : lits) {
                        clauses_.push_back({-v, l});
                        big.push_back(-l);
                    }
                    clauses_.push_back(big);
                } else {
                    std::vector<int> big{-v};
                    for (int l : lits) {
                        clauses_.push_back({v, -l});
                        big.push_back(l);
                    }
                    clauses_.push_back(big);
                }
                return v;
            }
            case Kind::Implies: {
                int a = encode(t->args[0]), b = encode(t->args[1]);
                int v = newVar();
                clauses_.push_back({-v, -a, b});
                clauses_.push_back({v, a});
                clauses_.push_back({v, -b});
                return v;
            }
            case Kind::Eq: {
                Term a = t->args[0], b = t->args[1];
                if (a->sort == Sort::Bool) {
                    int x = encode(a), y = encode(b);
                    int v = newVar();
                    clauses_.push_back({-v, -x, y});
                    clauses_.push_back({-v, x, -y});
                    clauses_.push_back({v, x, y});
                    clauses_.push_back({v, -x, -y});
                    return v;
                }
                if (a->sort == Sort::Obj) return eufEqAtom(eufNode(toVar(a)), eufNode(toVar(b)));
                return relAtom(a, b, Rel::Eq);
            }
            case Kind::Le:
                return relAtom(t->args[0], t->args[1], Rel::Le);
            case Kind::Lt:
                return relAtom(t->args[0], t->args[1], Rel::Lt);
            default:
                break;
        }
        // Numeric terms never reach formula position.
        return constLit(false);
    }

    int relAtom(Term a, Term b, Rel rel) {
        Lin l;
        linearize(a, Rational(1), l);
        linearize(b, Rational(-1), l);
        return arithAtom(l.c, rel, -l.k);
    }

    void addInterfaceEqualities() {
        std::vector<std::pair<uint32_t, int>> shared;  // arith index, euf node
        for (const auto& [term_id, node] : euf_of_term_) {
            auto it = arith_of_term_.find(term_id);
            if (it != arith_of_term_.end()) shared.emplace_back(it->second, node);
        }
        // A purified numeric application result or argument that never
        // appeared in arithmetic still needs an arithmetic index to be shared.
        for (int n = 0; n < static_cast<int>(euf_terms_.size()); ++n) {
            Term t = euf_terms_[n];
            if (!t || (t->sort != Sort::Int && t->sort != Sort::Real)) continue;
            if (!arith_of_term_.count(t->id)) shared.emplace_back(arithVar(t), n);
        }
        std::sort(shared.begin(), shared.end());
        shared.erase(std::unique(shared.begin(), shared.end()), shared.end());
        for (size_t i = 0; i < shared.size(); ++i) {
            for (size_t j = i + 1; j < shared.size(); ++j) {
                std::map<uint32_t, Rational> form;
                form[shared[i].first] = Rational(1);
                form[shared[j].first] = Rational(-1);
                int v = arithAtom(form, Rel::Eq, Rational(0));
                atoms_[v].interface = true;
                atoms_[v].euf_a = shared[i].second;
                atoms_[v].euf_b = shared[j].second;
            }
        }
    }

    // -- theory checks -------------------------------------------------------

    std::vector<int8_t> val_;  // per SAT var: 0 unassigned, 1 true, -1 false

    bool litTrue(int lit) const { return val_[std::abs(lit)] == (lit > 0 ? 1 : -1); }

    void arithConstraint(int lit, std::vector<LinearConstraint>& out) const {
        const Atom& a = atoms_[std::abs(lit)];
        bool pos = lit > 0;
        LinearConstraint lc;
        if (pos) {
            lc.coeffs = a.form;
            lc.rel = a.rel;
            lc.rhs = a.rhs;
        } else {
            if (a.rel == Rel::Eq) return;  // carried by the helper atoms
            for (const auto& [x, c] : a.form) lc.coeffs[x] = -c;
            lc.rhs = -a.rhs;
            lc.rel = a.rel == Rel::Le ? Rel::Lt : Rel::Le;  // ~(f<=c): -f<-c ; ~(f<c): -f<=-c
        }
        out.push_back(std::move(lc));
    }

    Result arithCheck(const std::vector<int>& lits, std::vector<Rational>* model) {
        std::vector<LinearConstraint> cs;
        for (int l : lits) arithConstraint(l, cs);
        uint64_t before = steps;
        Result r = decideLinear(cs, is_int_, steps, max_steps_, model);
        if (r == Result::Unknown && steps > max_steps_) throw BudgetOut{};
        (void)before;
        return r;
    }

    bool eufLiteralApplies(int lit) const {
        const Atom& a = atoms_[std::abs(lit)];
        return a.kind == AtomKind::EufEq || a.kind == AtomKind::BoolShared ||
               (a.kind == AtomKind::Arith && a.interface);
    }

    bool eufConsistent(const std::vector<int>& lits, Congruence* out = nullptr) {
        tick();
        Congruence cc(static_cast<int>(euf_terms_.size()));
        std::vector<std::pair<int, int>> diseq{{euf_true_, euf_false_}};
        for (int l : lits) {
            const Atom& a = atoms_[std::abs(l)];
            if (a.kind == AtomKind::BoolShared) {
                cc.unite(a.euf_node, l > 0 ? euf_true_ : euf_false_);
            } else if (l > 0) {
                cc.unite(a.euf_a, a.euf_b);
            } else {
                diseq.emplace_back(a.euf_a, a.euf_b);
            }
        }
        cc.close(defs_);
        for (const auto& [x, y] : diseq) {
            if (cc.find(x) == cc.find(y)) return false;
        }
        if (out) *out = cc;
        return true;
    }

    template <typename Check>
    std::vector<int> minimizeCore(std::vector<int> core, Check unsat) {
        for (size_t i = 0; i < core.size();) {
            std::vector<int> trial = core;
            trial.erase(trial.begin() + static_cast<long>(i));
            if (unsat(trial)) core = std::move(trial);
            else ++i;
        }
        return core;
    }

    // Theory status of the current (possibly partial) assignment. On Unsat
    // @p conflict receives a minimal set of assigned literals that is already
    // inconsistent.
    Result theoryCheck(std::vector<int>& conflict, std::vector<Rational>* arith_model) {
        std::vector<int> arith_lits, euf_lits;
        for (size_t v = 1; v < atoms_.size(); ++v) {
            if (val_[v] == 0) continue;
            int lit = val_[v] > 0 ? static_cast<int>(v) : -static_cast<int>(v);
            if (atoms_[v].kind == AtomKind::Arith) arith_lits.push_back(lit);
            if (eufLiteralApplies(lit)) euf_lits.push_back(lit);
        }
        Result ar = arithCheck(arith_lits, arith_model);
        if (ar == Result::Unsat) {
            conflict = minimizeCore(arith_lits, [&](const std::vector<int>& ls) {
                return arithCheck(ls, nullptr) == Result::Unsat;
            });
            return Result::Unsat;
        }
        if (!euf_lits.empty() && !eufConsistent(euf_lits)) {
            conflict = minimizeCore(euf_lits, [&](const std::vector<int>& ls) {
                return !eufConsistent(ls);
            });
            return Result::Unsat;
        }
        return ar;
    }

    // -- model -------------------------------------------------------------

    Model buildModel(const std::vector<Rational>& arith) {
        Model m;
        for (size_t i = 0; i < arith_terms_.size(); ++i) {
            Rational v = i < arith.size() ? arith[i] : Rational(0);
            m.numbers[arith_terms_[i]->name] = v;
        }
        for (const auto& [tid, v] : bool_var_of_term_) {
            (void)tid;
        }
        for (Term t : bool_terms_) m.booleans[t->name] = litTrue(bool_var_of_term_.at(t->id));
        std::vector<int> euf_lits;
        for (size_t v = 1; v < atoms_.size(); ++v) {
            int lit = val_[v] > 0 ? static_cast<int>(v) : -static_cast<int>(v);
            if (val_[v] != 0 && eufLiteralApplies(lit)) euf_lits.push_back(lit);
        }
        Congruence cc(static_cast<int>(euf_terms_.size()));
        eufConsistent(euf_lits, &cc);
        int fresh_obj = static_cast<int>(euf_terms_.size());
        for (Term v : tm_.variables()) {
            if (v->sort == Sort::Obj) {
                auto it = euf_of_term_.find(v->id);
                m.objects[v->name] = it != euf_of_term_.end() ? cc.find(it->second) : fresh_obj++;
            } else if (v->sort == Sort::Bool) {
                m.booleans.emplace(v->name, false);
            } else {
                m.numbers.emplace(v->name, Rational(0));
            }
        }
        auto valueKey = [&](int node) -> std::string {
            Term t = euf_terms_[node];
            if (!t) return node == euf_true_ ? "true" : "false";
            if (t->sort == Sort::Bool) return m.booleans[t->name] ? "true" : "false";
            if (t->sort == Sort::Obj) return "@" + std::to_string(m.objects[t->name]);
            return m.numbers[t->name].str();
        };
        for (const auto& d : defs_) {
            std::string key = d.fn + "(";
            for (size_t i = 0; i < d.args.size(); ++i) key += (i ? "," : "") + valueKey(d.args[i]);
            key += ")";
            m.applications[key] = valueKey(d.result);
        }
        return m;
    }

    // -- search ------------------------------------------------------------

    struct TrailEntry {
        int lit;
        bool decision;
    };
    std::vector<TrailEntry> trail_;

    void assign(int lit, bool decision) {
        val_[std::abs(lit)] = lit > 0 ? 1 : -1;
        trail_.push_back({lit, decision});
    }

    bool propagate() {
        bool changed = true;
        while (changed) {
            changed = false;
            for (const auto& c : clauses_) {
                int unassigned = 0, last = 0;
                bool sat = false;
                for (int l : c) {
                    int8_t v = val_[std::abs(l)];
                    if (v == 0) {
                        ++unassigned;
                        last = l;
                    } else if ((v > 0) == (l > 0)) {
                        sat = true;
                        break;
                    }
                }
                if (sat) continue;
                if (unassigned == 0) return false;
                if (unassigned == 1) {
                    assign(last, false);
                    changed = true;
                }
            }
        }
        return true;
    }

    // Undo to the most recent untried decision and take its other branch.
    bool backtrack() {
        while (!trail_.empty()) {
            TrailEntry e = trail_.back();
            trail_.pop_back();
            val_[std::abs(e.lit)] = 0;
            if (e.decision) {
                assign(-e.lit, false);
                return true;
            }
        }
        return false;
    }

    Outcome search(const std::vector<Term>& originals) {
        Outcome out;
        val_.assign(atoms_.size(), 0);
        bool saw_unknown = false;
        std::string unknown_reason;
        size_t learned_from = clauses_.size();
        (void)learned_from;
        for (;;) {
            tick();
            if (!propagate()) {
                if (!backtrack()) break;
                continue;
            }
            std::vector<int> conflict;
            int next = 0;
            for (size_t v = 1; v < atoms_.size(); ++v) {
                if (val_[v] == 0) {
                    next = static_cast<int>(v);
                    break;
                }
            }
            std::vector<Rational> arith_model;
            Result tr = theoryCheck(conflict, next == 0 ? &arith_model : nullptr);
            if (tr == Result::Unsat) {
                std::vector<int> learned;
                for (int l : conflict) learned.push_back(-l);
                clauses_.push_back(std::move(learned));
                if (!backtrack()) break;
                continue;
            }
            if (next != 0) {
                assign(next, true);
                continue;
            }
            // Complete assignment.
            if (tr == Result::Sat) {
                Model m = buildModel(arith_model);
                bool valid = true;
                for (Term a : originals) {
                    auto e = evaluate(a, m);
                    if (!e || !*e) {
                        valid = false;
                        break;
                    }
                }
                if (valid) {
                    out.result = Result::Sat;
                    out.model = std::move(m);
                    return out;
                }
                saw_unknown = true;
                unknown_reason = over_approximated_
                                     ? "nonlinear arithmetic or division outside the decided fragment"
                                     : "internal model validation failed";
            } else {
                saw_unknown = true;
                unknown_reason = "arithmetic engine could not decide a branch";
            }
            if (!backtrack()) break;
        }
        out.result = saw_unknown ? Result::Unknown : Result::Unsat;
        out.reason = unknown_reason;
        return out;
    }
};

class InternalSolver final : public Solver {
public:
    explicit InternalSolver(Budget b) : budget_(b) {}
    std::string name() const override { return "internal"; }
    Outcome check(TermManager& tm, const std::vector<Term>& assertions) override {
        Engine engine(tm, budget_.max_steps);
        for (Term a : assertions) engine.assertFormula(a);
        return engine.run(assertions);
    }

private:
    Budget budget_;
};

}  // namespace

std::unique_ptr<Solver> makeInternalSolver(Budget budget) {
    return std::make_unique<InternalSolver>(budget);
}

}  // namespace eshkol::smt
