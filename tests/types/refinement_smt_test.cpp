/**
 * @file refinement_smt_test.cpp
 * @brief Known-answer corpus for the refinement decision procedure (ADR-0044).
 *
 * 1. Exact arithmetic across the int64/bignum boundary.
 * 2. Presburger formulas over a box, decided by the procedure and by brute
 *    force over every point of the box: every single atom over two variables
 *    exhaustively, every pair from a grid of atoms, and seeded random Boolean
 *    combinations over three variables, some with divisibility conjuncts that
 *    exercise the Omega test's non-unit equality elimination.
 * 3. Rational conjunctions decided by the simplex and by Fourier-Motzkin
 *    projection: both engines must agree and every model must satisfy.
 * 4. Unbounded integer, mixed, uninterpreted-function and combination cases
 *    with known answers.
 *
 * Every Sat answer's model is re-evaluated against its formula.
 *
 *   refinement_smt_test [--corpus DIR] [--z3]
 *
 * --corpus writes every query as DIR/qNNNNN.smt2 plus DIR/answers.tsv (the
 * procedure's answers), for comparison against another solver. --z3 also asks
 * the z3 backend (when this build has it) and fails on any disagreement.
 */
#include "eshkol/types/smt.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

using namespace eshkol::smt;

namespace {

int g_failures = 0;
int g_queries = 0;
std::string g_corpus;
std::ofstream g_answers;
std::unique_ptr<Solver> g_z3;
int g_z3_agree = 0;

#define CHECK(cond, ...)                                    \
    do {                                                    \
        if (!(cond)) {                                      \
            ++g_failures;                                   \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);              \
            std::fprintf(stderr, "\n");                     \
        }                                                   \
    } while (0)

struct Rng {
    uint64_t s;
    uint64_t next() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1)); }
};

// Decide, validate the model, record in the corpus, compare with z3.
Result decide(TermManager& tm, const std::vector<Term>& q, const char* category) {
    auto solver = makeInternalSolver();
    Outcome o = solver->check(tm, q);
    ++g_queries;
    if (o.result == Result::Sat) {
        for (Term a : q) {
            auto v = evaluate(a, o.model);
            CHECK(v && *v, "[%s] model does not satisfy %s", category, toSmtLib2(a).c_str());
        }
    }
    if (!g_corpus.empty() && o.result != Result::Unknown) {
        char name[64];
        std::snprintf(name, sizeof name, "q%05d", g_queries);
        std::ofstream f(g_corpus + "/" + name + ".smt2");
        f << toSmtLib2Script(tm, q, true);
        g_answers << name << '\t' << category << '\t' << resultName(o.result) << '\n';
    }
    if (g_z3 && o.result != Result::Unknown) {
        Outcome z = g_z3->check(tm, q);
        CHECK(z.result == o.result, "[%s] internal %s, z3 %s: %s", category, resultName(o.result),
              resultName(z.result), toSmtLib2Script(tm, q, false).c_str());
        if (z.result == o.result) ++g_z3_agree;
    }
    return o.result;
}

// ---------------------------------------------------------------------------

void testExact() {
    Integer big = Integer(INT64_MAX) + Integer(INT64_MAX);
    CHECK(big.str() == "18446744073709551614", "bignum sum %s", big.str().c_str());
    CHECK((big - Integer(INT64_MAX)) == Integer(INT64_MAX), "bignum renormalises to int64");
    Integer p = Integer(INT64_MAX) * Integer(INT64_MAX);
    CHECK(Integer::truncDiv(p, Integer(INT64_MAX)) == Integer(INT64_MAX), "bignum division");
    CHECK(Integer::floorDiv(Integer(-7), Integer(2)) == Integer(-4), "floorDiv");
    CHECK(Integer::ceilDiv(Integer(-7), Integer(2)) == Integer(-3), "ceilDiv");
    CHECK(Integer::floorMod(Integer(-7), Integer(3)) == Integer(2), "floorMod");
    CHECK(Integer::gcd(Integer(-12), Integer(18)) == Integer(6), "gcd");
    CHECK(Integer(INT64_MIN).abs().str() == "9223372036854775808", "abs of INT64_MIN");
    Rational r(Integer(6), Integer(-4));
    CHECK(r.str() == "-3/2", "rational normalisation %s", r.str().c_str());
    CHECK((Rational(Integer(1), Integer(3)) + Rational(Integer(1), Integer(6))).str() == "1/2",
          "rational sum");
    CHECK(Rational::fromDouble(0.1).str() == "3602879701896397/36028797018963968",
          "exact double %s", Rational::fromDouble(0.1).str().c_str());
    CHECK(Rational(Integer(7), Integer(2)).floor() == Integer(3), "rational floor");
    CHECK(Rational(Integer(-7), Integer(2)).ceil() == Integer(-3), "rational ceil");
}

// ---------------------------------------------------------------------------
// Presburger over a box versus brute force

struct Atom {
    int a[3];
    int c;
    int rel;  // 0 <=, 1 <, 2 =
};

Term mkAtom(TermManager& tm, const std::vector<Term>& xs, const Atom& at) {
    std::vector<Term> parts;
    for (size_t i = 0; i < xs.size(); ++i) parts.push_back(tm.mkScale(Rational(at.a[i]), xs[i]));
    Term lhs = tm.mkAdd(parts);
    Term rhs = tm.mkInt(Integer(at.c));
    return at.rel == 0 ? tm.mkLe(lhs, rhs) : (at.rel == 1 ? tm.mkLt(lhs, rhs) : tm.mkEq(lhs, rhs));
}

Term boxOf(TermManager& tm, const std::vector<Term>& xs, int B) {
    std::vector<Term> cs;
    for (Term x : xs) {
        cs.push_back(tm.mkLe(tm.mkInt(Integer(-B)), x));
        cs.push_back(tm.mkLe(x, tm.mkInt(Integer(B))));
    }
    return tm.mkAnd(cs);
}

// Brute force: is there a point of the box satisfying phi (and x0 = r mod m when m > 0)?
bool bruteForce(Term phi, const std::vector<Term>& xs, int B, int m = 0, int r = 0) {
    std::vector<int> v(xs.size(), -B);
    for (;;) {
        Model md;
        for (size_t i = 0; i < xs.size(); ++i) md.numbers[xs[i]->name] = Rational(v[i]);
        bool div_ok = m == 0 || (((v[0] - r) % m) + m) % m == 0;
        auto e = evaluate(phi, md);
        if (div_ok && e && *e) return true;
        size_t i = 0;
        while (i < v.size() && v[i] == B) v[i++] = -B;
        if (i == v.size()) return false;
        ++v[i];
    }
}

void testPresburgerExhaustive() {
    int count = 0;
    // Every single atom over two variables.
    for (int a = -3; a <= 3; ++a)
        for (int b = -3; b <= 3; ++b)
            for (int c = -5; c <= 5; ++c)
                for (int rel = 0; rel < 3; ++rel) {
                    ExactScope scope;
                    TermManager tm;
                    std::vector<Term> xs{tm.mkVar("x", Sort::Int), tm.mkVar("y", Sort::Int)};
                    Term phi = mkAtom(tm, xs, Atom{{a, b, 0}, c, rel});
                    bool expect = bruteForce(phi, xs, 3);
                    Result got = decide(tm, {phi, boxOf(tm, xs, 3)}, "lia-box-atom");
                    CHECK(got == (expect ? Result::Sat : Result::Unsat),
                          "atom %dx+%dy rel%d %d: expected %s got %s", a, b, rel, c,
                          expect ? "sat" : "unsat", resultName(got));
                    ++count;
                }
    // Every pair from a grid of atoms.
    std::vector<Atom> grid;
    for (int a = -2; a <= 2; ++a)
        for (int b = -2; b <= 2; ++b)
            for (int c : {-3, 0, 2})
                for (int rel : {0, 2}) grid.push_back(Atom{{a, b, 0}, c, rel});
    for (size_t i = 0; i < grid.size(); ++i)
        for (size_t j = i + 1; j < grid.size(); j += 3) {
            ExactScope scope;
            TermManager tm;
            std::vector<Term> xs{tm.mkVar("x", Sort::Int), tm.mkVar("y", Sort::Int)};
            Term phi = tm.mkAnd(mkAtom(tm, xs, grid[i]), mkAtom(tm, xs, grid[j]));
            bool expect = bruteForce(phi, xs, 3);
            Result got = decide(tm, {phi, boxOf(tm, xs, 3)}, "lia-box-pair");
            CHECK(got == (expect ? Result::Sat : Result::Unsat), "pair %zu %zu", i, j);
            ++count;
        }
    std::printf("presburger exhaustive: %d queries\n", count);
}

Term randomFormula(TermManager& tm, const std::vector<Term>& xs, Rng& rng, int depth) {
    if (depth == 0 || rng.range(0, 9) < 4) {
        Atom at{{rng.range(-3, 3), rng.range(-3, 3), rng.range(-3, 3)}, rng.range(-8, 8),
                rng.range(0, 2)};
        return mkAtom(tm, xs, at);
    }
    // Operands are drawn in sequence so every compiler builds the same corpus.
    int op = rng.range(0, 3);
    Term a = randomFormula(tm, xs, rng, depth - 1);
    if (op == 2) return tm.mkNot(a);
    Term b = randomFormula(tm, xs, rng, depth - 1);
    if (op == 0) return tm.mkAnd(a, b);
    if (op == 1) return tm.mkOr(a, b);
    return tm.mkImplies(a, b);
}

void testPresburgerRandom(int n) {
    Rng rng{20261030};
    int sat = 0;
    for (int i = 0; i < n; ++i) {
        ExactScope scope;
        TermManager tm;
        std::vector<Term> xs{tm.mkVar("x", Sort::Int), tm.mkVar("y", Sort::Int),
                             tm.mkVar("z", Sort::Int)};
        Term phi = randomFormula(tm, xs, rng, 3);
        int m = 0, r = 0;
        std::vector<Term> q{phi, boxOf(tm, xs, 3)};
        if (i % 3 == 0) {
            // x = m k + r for a fresh unbounded k: divisibility of x - r by m.
            m = rng.range(2, 5);
            r = rng.range(0, m - 1);
            Term k = tm.mkVar("k", Sort::Int);
            q.push_back(tm.mkEq(xs[0], tm.mkAdd(tm.mkScale(Rational(m), k), tm.mkInt(Integer(r)))));
        }
        bool expect = bruteForce(phi, xs, 3, m, r);
        Result got = decide(tm, q, m ? "lia-box-divisibility" : "lia-box-random");
        CHECK(got == (expect ? Result::Sat : Result::Unsat), "random formula %d: %s", i,
              toSmtLib2(phi).c_str());
        sat += expect;
    }
    std::printf("presburger random: %d queries (%d sat)\n", n, sat);
}

// ---------------------------------------------------------------------------
// Unbounded families: no brute force exists, so every Sat model is
// re-evaluated here and the Unsat answers are what the z3 comparison checks.

Term linear(TermManager& tm, const std::vector<Term>& xs, Rng& rng, int cmax) {
    std::vector<Term> parts;
    for (Term x : xs) {
        int a = rng.range(-cmax, cmax);
        if (a) parts.push_back(tm.mkScale(Rational(a), x));
    }
    return tm.mkAdd(parts);
}

Term relation(TermManager& tm, Term lhs, Term rhs, int rel) {
    return rel == 0 ? tm.mkLe(lhs, rhs) : (rel == 1 ? tm.mkLt(lhs, rhs) : tm.mkEq(lhs, rhs));
}

Term conjunctive(TermManager& tm, Rng& rng, const std::function<Term()>& atom) {
    std::vector<Term> cs;
    int k = rng.range(2, 5);
    for (int j = 0; j < k; ++j) {
        Term a = atom();
        if (rng.range(0, 4) == 0) {
            Term b = atom();
            a = tm.mkOr(a, b);
        }
        if (rng.range(0, 5) == 0) a = tm.mkNot(a);
        cs.push_back(a);
    }
    return tm.mkAnd(cs);
}

void testUnboundedFamilies(int n) {
    Rng rng{1729};
    int counts[3][3] = {};
    for (int i = 0; i < n; ++i) {
        ExactScope scope;
        TermManager tm;
        int family = i % 3;
        const char* category = family == 0 ? "lia-unbounded" : (family == 1 ? "uflia" : "mixed");
        std::vector<Term> vars;
        std::function<Term()> atom;
        if (family == 0) {
            vars = {tm.mkVar("x", Sort::Int), tm.mkVar("y", Sort::Int), tm.mkVar("z", Sort::Int)};
            atom = [&]() {
                Term lhs = linear(tm, vars, rng, 6);
                Term rhs = tm.mkInt(Integer(rng.range(-12, 12)));
                return relation(tm, lhs, rhs, rng.range(0, 2));
            };
        } else if (family == 1) {
            Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
            Term v = tm.mkVar("v", Sort::Obj), w = tm.mkVar("w", Sort::Obj);
            std::vector<Term> pool{x, y,
                                   tm.mkApp("f", Sort::Int, {x}),
                                   tm.mkApp("f", Sort::Int, {y}),
                                   tm.mkApp("f", Sort::Int, {tm.mkAdd(x, tm.mkInt(Integer(1)))}),
                                   tm.mkApp("g", Sort::Int, {x, y}),
                                   tm.mkApp("len", Sort::Int, {v}),
                                   tm.mkApp("len", Sort::Int, {w})};
            atom = [&, v, w, pool]() {
                int pick = rng.range(0, 9);
                if (pick == 0) return tm.mkEq(v, w);
                if (pick == 1) return tm.mkEq(tm.mkApp("h", Sort::Obj, {v}), w);
                Term p0 = pool[rng.range(0, 7)];
                Term p1 = pool[rng.range(0, 7)];
                std::vector<Term> two{p0, p1};
                Term lhs = linear(tm, two, rng, 3);
                Term rhs = tm.mkInt(Integer(rng.range(-4, 4)));
                return relation(tm, lhs, rhs, rng.range(0, 2));
            };
        } else {
            vars = {tm.mkVar("n", Sort::Int), tm.mkVar("m", Sort::Int), tm.mkVar("r", Sort::Real),
                    tm.mkVar("s", Sort::Real)};
            atom = [&]() {
                Term rhs;
                if (rng.range(0, 1)) {
                    rhs = tm.mkInt(Integer(rng.range(-8, 8)));
                } else {
                    int n = rng.range(-8, 8), d = rng.range(1, 3);
                    rhs = tm.mkReal(Rational(Integer(n), Integer(d)));
                }
                Term lhs = linear(tm, vars, rng, 4);
                return relation(tm, lhs, rhs, rng.range(0, 2));
            };
        }
        Term phi = conjunctive(tm, rng, atom);
        Result got = decide(tm, {phi}, category);
        CHECK(got != Result::Unknown, "[%s] %d undecided: %s", category, i, toSmtLib2(phi).c_str());
        counts[family][static_cast<int>(got)]++;
    }
    std::printf("unbounded families: %d queries (lia %d/%d, uflia %d/%d, mixed %d/%d sat/unsat)\n", n,
                counts[0][0], counts[0][1], counts[1][0], counts[1][1], counts[2][0], counts[2][1]);
}

// ---------------------------------------------------------------------------
// Rational arithmetic: simplex versus projection

void testRationalEngines(int n) {
    Rng rng{42};
    int sat = 0;
    for (int i = 0; i < n; ++i) {
        ExactScope scope;
        int k = rng.range(2, 6);
        std::vector<LinearConstraint> cs;
        for (int j = 0; j < k; ++j) {
            LinearConstraint lc;
            for (uint32_t v = 0; v < 3; ++v) {
                int a = rng.range(-4, 4);
                if (a) {
                    int d = rng.range(1, 3);
                    lc.coeffs[v] = Rational(Integer(a), Integer(d));
                }
            }
            int rn = rng.range(-6, 6), rd = rng.range(1, 4);
            lc.rhs = Rational(Integer(rn), Integer(rd));
            bool eq = rng.range(0, 2) == 2 && rng.range(0, 2) == 0;
            int strict = rng.range(0, 1);
            lc.rel = static_cast<Rel>(eq ? 2 : strict);
            cs.push_back(std::move(lc));
        }
        std::vector<bool> is_int(3, false);
        uint64_t s1 = 0, s2 = 0;
        std::vector<Rational> m1, m2;
        Result a = decideLinear(cs, is_int, s1, 1000000, &m1, ArithEngine::Simplex);
        Result b = decideLinear(cs, is_int, s2, 1000000, &m2, ArithEngine::Projection);
        CHECK(a == b && a != Result::Unknown, "engines disagree on rational problem %d: %s vs %s", i,
              resultName(a), resultName(b));
        auto holds = [&](const std::vector<Rational>& m) {
            for (const auto& lc : cs) {
                Rational s(0);
                for (const auto& [v, c] : lc.coeffs) s += c * m[v];
                int cmp = s.compare(lc.rhs);
                if (lc.rel == Rel::Le ? cmp > 0 : (lc.rel == Rel::Lt ? cmp >= 0 : cmp != 0)) return false;
            }
            return true;
        };
        if (a == Result::Sat) {
            CHECK(holds(m1), "simplex model violates problem %d", i);
            ++sat;
        }
        if (b == Result::Sat) CHECK(holds(m2), "projection model violates problem %d", i);
        // Also through the solver interface, for the corpus.
        TermManager tm;
        std::vector<Term> xs{tm.mkVar("p", Sort::Real), tm.mkVar("q", Sort::Real),
                             tm.mkVar("r", Sort::Real)};
        std::vector<Term> q;
        for (const auto& lc : cs) {
            std::vector<Term> parts;
            for (const auto& [v, c] : lc.coeffs) parts.push_back(tm.mkScale(c, xs[v]));
            Term lhs = tm.mkAdd(parts), rhs = tm.mkReal(lc.rhs);
            q.push_back(lc.rel == Rel::Le ? tm.mkLe(lhs, rhs)
                                          : (lc.rel == Rel::Lt ? tm.mkLt(lhs, rhs) : tm.mkEq(lhs, rhs)));
        }
        Result c = decide(tm, q, "lra");
        CHECK(c == a, "solver and engine disagree on rational problem %d", i);
    }
    std::printf("rational engines: %d problems (%d sat)\n", n, sat);
}

// ---------------------------------------------------------------------------
// Known answers: unbounded integers, mixed, UF, combination

void expect(const char* name, const char* category, Result want,
            const std::function<std::vector<Term>(TermManager&)>& build) {
    ExactScope scope;
    TermManager tm;
    std::vector<Term> q = build(tm);
    Result got = decide(tm, q, category);
    CHECK(got == want, "%s: expected %s got %s", name, resultName(want), resultName(got));
}

void testKnownAnswers() {
    auto I = [](TermManager& tm, int64_t v) { return tm.mkInt(Integer(v)); };
    expect("parity 2x = 2y + 1", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkEq(tm.mkScale(2, x), tm.mkAdd(tm.mkScale(2, y), I(tm, 1)))};
    });
    expect("3x + 6y = 4", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkEq(tm.mkAdd(tm.mkScale(3, x), tm.mkScale(6, y)), I(tm, 4))};
    });
    expect("7x + 12y = 1", "lia", Result::Sat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkEq(tm.mkAdd(tm.mkScale(7, x), tm.mkScale(12, y)), I(tm, 1))};
    });
    expect("0 < x < 1 over Int", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int);
        return std::vector<Term>{tm.mkLt(I(tm, 0), x), tm.mkLt(x, I(tm, 1))};
    });
    expect("0 < x < 1 over Real", "lra", Result::Sat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Real);
        return std::vector<Term>{tm.mkLt(tm.mkReal(0), x), tm.mkLt(x, tm.mkReal(1))};
    });
    // Pugh's dark-shadow example: rationally feasible, no integer point.
    expect("27 <= 11x+13y <= 45, -10 <= 7x-9y <= 4", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        Term s = tm.mkAdd(tm.mkScale(11, x), tm.mkScale(13, y));
        Term d = tm.mkAdd(tm.mkScale(7, x), tm.mkScale(-9, y));
        return std::vector<Term>{tm.mkLe(I(tm, 27), s), tm.mkLe(s, I(tm, 45)), tm.mkLe(I(tm, -10), d),
                                 tm.mkLe(d, I(tm, 4))};
    });
    expect("the same over Real", "lra", Result::Sat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Real), y = tm.mkVar("y", Sort::Real);
        Term s = tm.mkAdd(tm.mkScale(11, x), tm.mkScale(13, y));
        Term d = tm.mkAdd(tm.mkScale(7, x), tm.mkScale(-9, y));
        return std::vector<Term>{tm.mkLe(tm.mkReal(27), s), tm.mkLe(s, tm.mkReal(45)),
                                 tm.mkLe(tm.mkReal(-10), d), tm.mkLe(d, tm.mkReal(4))};
    });
    expect("splinter: 2 <= 3x - 3y <= 2 with y free", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        Term d = tm.mkSub(tm.mkScale(3, x), tm.mkScale(3, y));
        return std::vector<Term>{tm.mkLe(I(tm, 1), d), tm.mkLe(d, I(tm, 2))};
    });
    expect("large coefficients", "lia", Result::Sat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        Integer big = Integer(INT64_MAX) * Integer(3);
        return std::vector<Term>{tm.mkEq(tm.mkAdd(x, y), tm.mkInt(big)), tm.mkLe(tm.mkInt(big), tm.mkScale(2, x))};
    });
    expect("mixed: n integer, r real, n < r < n + 1, r = 5/2", "mixed", Result::Sat, [&](TermManager& tm) {
        Term n = tm.mkVar("n", Sort::Int), r = tm.mkVar("r", Sort::Real);
        return std::vector<Term>{tm.mkLt(n, r), tm.mkLt(r, tm.mkAdd(n, I(tm, 1))),
                                 tm.mkEq(r, tm.mkReal(Rational(Integer(5), Integer(2))))};
    });
    expect("mixed: n integer, n < r < n + 1, r = 3", "mixed", Result::Unsat, [&](TermManager& tm) {
        Term n = tm.mkVar("n", Sort::Int), r = tm.mkVar("r", Sort::Real);
        return std::vector<Term>{tm.mkLt(n, r), tm.mkLt(r, tm.mkAdd(n, I(tm, 1))), tm.mkEq(r, tm.mkReal(3))};
    });
    expect("congruence: x = y, f(x) != f(y)", "uf", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Obj), y = tm.mkVar("y", Sort::Obj);
        return std::vector<Term>{tm.mkEq(x, y), tm.mkDistinct(tm.mkApp("f", Sort::Obj, {x}), tm.mkApp("f", Sort::Obj, {y}))};
    });
    expect("f(f(f(a))) = a, f(f(f(f(f(a))))) = a, f(a) != a", "uf", Result::Unsat, [&](TermManager& tm) {
        Term a = tm.mkVar("a", Sort::Obj);
        auto f = [&](Term t) { return tm.mkApp("f", Sort::Obj, {t}); };
        return std::vector<Term>{tm.mkEq(f(f(f(a))), a), tm.mkEq(f(f(f(f(f(a))))), a), tm.mkDistinct(f(a), a)};
    });
    expect("f(f(a)) = a, f(a) != a", "uf", Result::Sat, [&](TermManager& tm) {
        Term a = tm.mkVar("a", Sort::Obj);
        auto f = [&](Term t) { return tm.mkApp("f", Sort::Obj, {t}); };
        return std::vector<Term>{tm.mkEq(f(f(a)), a), tm.mkDistinct(f(a), a)};
    });
    expect("combination: x = y + 1, len(x) != len(y + 1)", "uflia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkEq(x, tm.mkAdd(y, I(tm, 1))),
                                 tm.mkDistinct(tm.mkApp("len", Sort::Int, {x}),
                                               tm.mkApp("len", Sort::Int, {tm.mkAdd(y, I(tm, 1))}))};
    });
    expect("combination: x <= y <= x, f(x) < f(y)", "uflia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkLe(x, y), tm.mkLe(y, x),
                                 tm.mkLt(tm.mkApp("f", Sort::Int, {x}), tm.mkApp("f", Sort::Int, {y}))};
    });
    expect("combination (non-convex): 1 <= x <= 2, f(1) = 0, f(2) = 0, f(x) != 0", "uflia",
           Result::Unsat, [&](TermManager& tm) {
               Term x = tm.mkVar("x", Sort::Int);
               auto f = [&](Term t) { return tm.mkApp("f", Sort::Int, {t}); };
               return std::vector<Term>{tm.mkLe(I(tm, 1), x), tm.mkLe(x, I(tm, 2)), tm.mkEq(f(I(tm, 1)), I(tm, 0)),
                                        tm.mkEq(f(I(tm, 2)), I(tm, 0)), tm.mkDistinct(f(x), I(tm, 0))};
           });
    expect("combination sat: f(x) > f(y)", "uflia", Result::Sat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkLt(tm.mkApp("f", Sort::Int, {y}), tm.mkApp("f", Sort::Int, {x}))};
    });
    expect("Boolean predicate: p(x), not p(y), x = y", "uflia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkApp("p", Sort::Bool, {x}), tm.mkNot(tm.mkApp("p", Sort::Bool, {y})),
                                 tm.mkEq(x, y)};
    });
    expect("vector bound: 0 <= i < len(v), len(v) = n, n <= 3, i >= 3", "uflia", Result::Unsat,
           [&](TermManager& tm) {
               Term v = tm.mkVar("v", Sort::Obj), i = tm.mkVar("i", Sort::Int), n = tm.mkVar("n", Sort::Int);
               Term len = tm.mkApp("vector-length", Sort::Int, {v});
               return std::vector<Term>{tm.mkLe(I(tm, 0), i), tm.mkLt(i, len), tm.mkEq(len, n),
                                        tm.mkLe(n, I(tm, 3)), tm.mkLe(I(tm, 3), i)};
           });
    expect("ite: y = (if (< x 0) (- x) x), y < 0", "lia", Result::Unsat, [&](TermManager& tm) {
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        return std::vector<Term>{tm.mkEq(y, tm.mkIte(tm.mkLt(x, I(tm, 0)), tm.mkNeg(x), x)), tm.mkLt(y, I(tm, 0))};
    });
    // Over-approximated nonlinear product: never a false Sat.
    {
        ExactScope scope;
        TermManager tm;
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        std::vector<Term> q{tm.mkEq(tm.mkMul(x, y), I(tm, 1)), tm.mkEq(x, I(tm, 2))};
        Outcome o = makeInternalSolver()->check(tm, q);
        ++g_queries;
        CHECK(o.result != Result::Sat, "nonlinear x*y = 1, x = 2 must not be Sat");
        Term z = tm.mkVar("z", Sort::Int);
        std::vector<Term> q2{tm.mkEq(tm.mkMul(x, z), tm.mkMul(z, x)), tm.mkLt(tm.mkMul(x, z), tm.mkMul(z, x))};
        CHECK(makeInternalSolver()->check(tm, q2).result == Result::Unsat, "x*z < z*x is unsat");
        ++g_queries;
    }
    // A deterministic budget answers Unknown, never Sat or Unsat.
    {
        ExactScope scope;
        TermManager tm;
        Term x = tm.mkVar("x", Sort::Int), y = tm.mkVar("y", Sort::Int);
        Term s = tm.mkAdd(tm.mkScale(11, x), tm.mkScale(13, y));
        std::vector<Term> q{tm.mkLe(I(tm, 27), s), tm.mkLe(s, I(tm, 45))};
        Outcome o = makeInternalSolver(Budget{3})->check(tm, q);
        ++g_queries;
        CHECK(o.result == Result::Unknown, "tiny budget must give Unknown, got %s", resultName(o.result));
    }
}

}  // namespace

int main(int argc, char** argv) {
    bool use_z3 = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--corpus") && i + 1 < argc) g_corpus = argv[++i];
        else if (!std::strcmp(argv[i], "--z3")) use_z3 = true;
    }
    if (!g_corpus.empty()) g_answers.open(g_corpus + "/answers.tsv");
    if (use_z3) {
        g_z3 = makeZ3Solver();
        std::printf("z3 backend: %s\n", g_z3 ? "compiled in" : "not compiled in this build");
    }
    {
        ExactScope scope;
        testExact();
    }
    testKnownAnswers();
    testPresburgerExhaustive();
    testPresburgerRandom(3000);
    testRationalEngines(2000);
    testUnboundedFamilies(6000);
    if (g_z3) std::printf("z3 agreement: %d queries\n", g_z3_agree);
    if (g_failures) {
        std::printf("FAILED: %d check(s) over %d queries\n", g_failures, g_queries);
        return 1;
    }
    std::printf("PASS: refinement decision procedure (%d queries)\n", g_queries);
    return 0;
}
