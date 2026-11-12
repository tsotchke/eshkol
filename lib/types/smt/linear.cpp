/**
 * @file linear.cpp
 * @brief Linear arithmetic engines of the refinement decision procedure
 *        (ADR-0044): exact general simplex over Q, Fourier-Motzkin projection
 *        of rational variables, and the Omega test for the integers.
 *
 * Every engine works on exact Rationals/Integers and returns a model on Sat.
 *
 * Simplex: Dutertre and de Moura's general simplex. Each constraint with two
 * or more variables gets a slack variable equal to its linear form; strict
 * bounds are delta-rationals (c + k*delta), and Bland's rule (smallest index
 * first) guarantees termination. Used for purely rational conjunctions.
 *
 * Projection: rational variables are eliminated first, by substitution from
 * an equality when one mentions the variable and by Fourier-Motzkin
 * combination of each lower with each upper bound otherwise. Projection over
 * Q is exact for any values of the remaining variables, integers included, so
 * what is left is an integer problem with the same integer solutions.
 *
 * Omega test (Pugh 1991): equalities are eliminated exactly (a unit
 * coefficient is solved for directly; otherwise the "mod-hat" step introduces
 * a fresh variable that shrinks the coefficients), then inequalities by exact
 * projection when every lower or every upper coefficient is 1, otherwise by
 * the dark shadow, the real shadow and, if those disagree, the splinters. It
 * is a complete decision procedure for conjunctions of linear integer
 * constraints. Each elimination remembers the constraints it removed, so the
 * model is rebuilt in reverse order once the reduced problem is solved.
 */
#include "eshkol/types/smt.h"

#include <algorithm>
#include <set>
#include <sstream>

namespace eshkol::smt {

namespace {

struct BudgetExhausted {};

struct Ticker {
    uint64_t& steps;
    uint64_t max;
    void tick(uint64_t n = 1) {
        steps += n;
        if (steps > max) throw BudgetExhausted{};
    }
};

// ===========================================================================
// Simplex
// ===========================================================================

struct DR {  // c + k*delta
    Rational c;
    Rational k;
    int compare(const DR& o) const {
        int a = c.compare(o.c);
        return a != 0 ? a : k.compare(o.k);
    }
    DR operator+(const DR& o) const { return {c + o.c, k + o.k}; }
    DR operator-(const DR& o) const { return {c - o.c, k - o.k}; }
    DR scale(const Rational& r) const { return {c * r, k * r}; }
};

class Simplex {
public:
    Simplex(size_t n_original, Ticker& t) : n_orig_(n_original), t_(t) {
        lower_.resize(n_original);
        upper_.resize(n_original);
        beta_.resize(n_original);
        row_of_.assign(n_original, -1);
    }

    // Returns false when the bounds alone are already contradictory.
    bool add(const LinearConstraint& lc) {
        std::vector<std::pair<uint32_t, Rational>> terms;
        for (const auto& [v, a] : lc.coeffs) {
            if (!a.isZero()) terms.emplace_back(v, a);
        }
        if (terms.empty()) {
            int s = Rational(0).compare(lc.rhs);
            if (lc.rel == Rel::Le) return s <= 0;
            if (lc.rel == Rel::Lt) return s < 0;
            return s == 0;
        }
        size_t var;
        Rational scale(1);
        if (terms.size() == 1) {
            var = terms[0].first;
            scale = terms[0].second;
        } else {
            var = newSlack(terms);
        }
        Rational bound = lc.rhs / scale;
        bool flip = scale.sign() < 0;  // dividing by a negative swaps the direction
        if (lc.rel == Rel::Eq) {
            return assertLower(var, {bound, Rational(0)}) && assertUpper(var, {bound, Rational(0)});
        }
        Rational k = lc.rel == Rel::Lt ? Rational(1) : Rational(0);
        if (!flip) return assertUpper(var, {bound, -k});
        return assertLower(var, {bound, k});
    }

    Result check() {
        // Move every non-basic variable inside its bounds first.
        for (size_t v = 0; v < beta_.size(); ++v) {
            if (row_of_[v] >= 0) continue;
            if (lower_[v] && beta_[v].compare(*lower_[v]) < 0) update(v, *lower_[v]);
            else if (upper_[v] && beta_[v].compare(*upper_[v]) > 0) update(v, *upper_[v]);
        }
        for (;;) {
            t_.tick();
            int bad = -1;
            for (size_t v = 0; v < beta_.size(); ++v) {
                if (row_of_[v] < 0) continue;
                if ((lower_[v] && beta_[v].compare(*lower_[v]) < 0) ||
                    (upper_[v] && beta_[v].compare(*upper_[v]) > 0)) {
                    bad = static_cast<int>(v);
                    break;
                }
            }
            if (bad < 0) return Result::Sat;
            const std::vector<Rational>& row = rows_[row_of_[bad]];
            bool increase = lower_[bad] && beta_[bad].compare(*lower_[bad]) < 0;
            int entering = -1;
            for (size_t j = 0; j < beta_.size(); ++j) {
                if (row_of_[j] >= 0 || j >= row.size() || row[j].isZero()) continue;
                bool pos = row[j].sign() > 0;
                bool can_up = !upper_[j] || beta_[j].compare(*upper_[j]) < 0;
                bool can_down = !lower_[j] || beta_[j].compare(*lower_[j]) > 0;
                if (increase ? (pos ? can_up : can_down) : (pos ? can_down : can_up)) {
                    entering = static_cast<int>(j);
                    break;
                }
            }
            if (entering < 0) return Result::Unsat;
            pivotAndUpdate(bad, entering, increase ? *lower_[bad] : *upper_[bad]);
        }
    }

    std::vector<Rational> model() const {
        Rational delta(1);
        auto tighten = [&](const DR& lo, const DR& hi) {
            // Concretised lo <= hi must survive: lo.c + lo.k d <= hi.c + hi.k d.
            if (lo.c < hi.c && lo.k > hi.k) {
                Rational limit = (hi.c - lo.c) / (lo.k - hi.k);
                if (limit < delta) delta = limit;
            }
        };
        for (size_t v = 0; v < beta_.size(); ++v) {
            if (lower_[v]) tighten(*lower_[v], beta_[v]);
            if (upper_[v]) tighten(beta_[v], *upper_[v]);
        }
        std::vector<Rational> out;
        for (size_t v = 0; v < n_orig_; ++v) out.push_back(beta_[v].c + beta_[v].k * delta);
        return out;
    }

private:
    size_t n_orig_;
    Ticker& t_;
    std::vector<std::vector<Rational>> rows_;  // row r: basic var = sum row[j] * x_j
    std::vector<int> row_of_;                  // var -> row index, -1 if non-basic
    std::vector<std::optional<DR>> lower_, upper_;
    std::vector<DR> beta_;

    size_t newSlack(const std::vector<std::pair<uint32_t, Rational>>& terms) {
        size_t s = beta_.size();
        lower_.emplace_back();
        upper_.emplace_back();
        beta_.push_back({Rational(0), Rational(0)});
        row_of_.push_back(static_cast<int>(rows_.size()));
        std::vector<Rational> row(s + 1);
        DR value{Rational(0), Rational(0)};
        for (const auto& [v, a] : terms) {
            if (row_of_[v] >= 0) {
                // Express through the current tableau so the row is over non-basics.
                const auto& r = rows_[row_of_[v]];
                for (size_t j = 0; j < r.size(); ++j) {
                    if (!r[j].isZero()) row[j] += a * r[j];
                }
            } else {
                row[v] += a;
            }
            value = value + beta_[v].scale(a);
        }
        for (auto& r : rows_) r.resize(s + 1);
        rows_.push_back(std::move(row));
        beta_[s] = value;
        return s;
    }

    bool assertLower(size_t v, const DR& b) {
        if (!lower_[v] || b.compare(*lower_[v]) > 0) lower_[v] = b;
        return !(upper_[v] && lower_[v]->compare(*upper_[v]) > 0);
    }
    bool assertUpper(size_t v, const DR& b) {
        if (!upper_[v] || b.compare(*upper_[v]) < 0) upper_[v] = b;
        return !(lower_[v] && lower_[v]->compare(*upper_[v]) > 0);
    }

    void update(size_t v, const DR& value) {
        DR diff = value - beta_[v];
        for (size_t b = 0; b < beta_.size(); ++b) {
            if (row_of_[b] < 0) continue;
            const auto& r = rows_[row_of_[b]];
            if (v < r.size() && !r[v].isZero()) beta_[b] = beta_[b] + diff.scale(r[v]);
        }
        beta_[v] = value;
    }

    void pivotAndUpdate(size_t basic, size_t entering, const DR& value) {
        int r = row_of_[basic];
        Rational a = rows_[r][entering];
        DR theta = (value - beta_[basic]).scale(Rational(1) / a);
        beta_[basic] = value;
        beta_[entering] = beta_[entering] + theta;
        for (size_t b = 0; b < beta_.size(); ++b) {
            if (row_of_[b] < 0 || b == basic) continue;
            const auto& rr = rows_[row_of_[b]];
            if (entering < rr.size() && !rr[entering].isZero()) {
                beta_[b] = beta_[b] + theta.scale(rr[entering]);
            }
        }
        // Pivot: entering = (basic - sum_{j != entering} row[j] x_j) / a.
        std::vector<Rational> nrow(rows_[r].size());
        Rational inv = Rational(1) / a;
        for (size_t j = 0; j < nrow.size(); ++j) {
            if (j == entering) continue;
            if (!rows_[r][j].isZero()) nrow[j] = -rows_[r][j] * inv;
        }
        nrow[basic] = inv;
        for (size_t k = 0; k < rows_.size(); ++k) {
            if (static_cast<int>(k) == r) continue;
            Rational c = rows_[k][entering];
            if (c.isZero()) continue;
            rows_[k][entering] = Rational(0);
            for (size_t j = 0; j < nrow.size(); ++j) {
                if (!nrow[j].isZero()) rows_[k][j] += c * nrow[j];
            }
        }
        rows_[r] = std::move(nrow);
        row_of_[entering] = r;
        row_of_[basic] = -1;
        t_.tick();
    }
};

// ===========================================================================
// Rational projection (Fourier-Motzkin)
// ===========================================================================

enum class CT : uint8_t { GE, GT, EQ };

struct QCon {  // sum a_i x_i + c  (>= | > | =)  0
    std::map<uint32_t, Rational> a;
    Rational c;
    CT t = CT::GE;
};

struct IntCon {  // sum a_i x_i + c  (>= | =)  0, integer coefficients
    std::map<uint32_t, Integer> a;
    Integer c;
    bool eq = false;
};

struct RealElim {
    uint32_t var;
    bool by_eq;
    QCon eq;
    std::vector<QCon> touching;  // every constraint mentioning var at elimination time
};

Rational evalQ(const QCon& q, const std::vector<Rational>& vals, uint32_t skip) {
    Rational s = q.c;
    for (const auto& [v, a] : q.a) {
        if (v != skip) s += a * vals[v];
    }
    return s;
}

// ===========================================================================
// Omega test
// ===========================================================================

Integer modHat(const Integer& a, const Integer& m) {
    // a - m * floor(a/m + 1/2) = a - m * floor((2a + m) / 2m)
    Integer two(2);
    return a - m * Integer::floorDiv(a * two + m, m * two);
}

std::string coeffKey(const std::map<uint32_t, Integer>& a) {
    std::ostringstream o;
    for (const auto& [v, c] : a) o << v << ':' << c.str() << ';';
    return o.str();
}

Integer evalI(const std::map<uint32_t, Integer>& a, const Integer& c,
              const std::map<uint32_t, Integer>& model, uint32_t skip) {
    Integer s = c;
    for (const auto& [v, k] : a) {
        if (v == skip) continue;
        auto it = model.find(v);
        if (it != model.end()) s += k * it->second;
    }
    return s;
}

void substitute(IntCon& con, uint32_t x, const std::map<uint32_t, Integer>& expr,
                const Integer& expr_c) {
    auto it = con.a.find(x);
    if (it == con.a.end()) return;
    Integer b = it->second;
    con.a.erase(it);
    for (const auto& [v, k] : expr) {
        Integer nv = con.a.count(v) ? con.a[v] + b * k : b * k;
        if (nv.isZero()) con.a.erase(v);
        else con.a[v] = nv;
    }
    con.c += b * expr_c;
}

class Omega {
public:
    Omega(Ticker& t, uint32_t next_var) : t_(t), next_var_(next_var) {}

    // Normalise in place; false when a constraint is already contradictory.
    bool normalize(std::vector<IntCon>& P) {
        std::vector<IntCon> out;
        std::map<std::string, size_t> ge_index;
        for (auto& con : P) {
            for (auto it = con.a.begin(); it != con.a.end();) {
                if (it->second.isZero()) it = con.a.erase(it);
                else ++it;
            }
            Integer g(0);
            for (const auto& [v, k] : con.a) g = Integer::gcd(g, k);
            if (g.isZero()) {
                if (con.eq ? !con.c.isZero() : con.c.sign() < 0) return false;
                continue;
            }
            if (con.eq) {
                if (!Integer::floorMod(con.c, g).isZero()) return false;
                if (!g.isOne()) {
                    for (auto& [v, k] : con.a) k = Integer::truncDiv(k, g);
                    con.c = Integer::truncDiv(con.c, g);
                }
                out.push_back(std::move(con));
                continue;
            }
            if (!g.isOne()) {
                for (auto& [v, k] : con.a) k = Integer::truncDiv(k, g);
                con.c = Integer::floorDiv(con.c, g);
            }
            std::string key = coeffKey(con.a);
            auto it = ge_index.find(key);
            if (it != ge_index.end()) {
                if (con.c < out[it->second].c) out[it->second].c = con.c;  // keep the tighter
                continue;
            }
            ge_index.emplace(key, out.size());
            out.push_back(std::move(con));
        }
        // Opposite pairs: a.x + c1 >= 0 and -a.x + c2 >= 0.
        for (size_t i = 0; i < out.size(); ++i) {
            if (out[i].eq) continue;
            std::map<uint32_t, Integer> neg;
            for (const auto& [v, k] : out[i].a) neg[v] = -k;
            auto it = ge_index.find(coeffKey(neg));
            if (it == ge_index.end() || out[it->second].eq) continue;
            Integer sum = out[i].c + out[it->second].c;
            if (sum.sign() < 0) return false;
            if (sum.isZero()) out[i].eq = true;
        }
        P = std::move(out);
        return true;
    }

    bool solve(std::vector<IntCon> P, std::map<uint32_t, Integer>& M) {
        t_.tick();
        if (!normalize(P)) return false;
        for (size_t ei = 0; ei < P.size(); ++ei) {
            if (P[ei].eq) return eliminateEquality(std::move(P), ei, M);
        }
        if (P.empty()) return true;
        return eliminateInequality(std::move(P), M);
    }

private:
    Ticker& t_;
    uint32_t next_var_;

    bool eliminateEquality(std::vector<IntCon> P, size_t ei, std::map<uint32_t, Integer>& M) {
        IntCon e = P[ei];
        uint32_t k = 0;
        Integer best;
        bool have = false;
        for (const auto& [v, a] : e.a) {
            if (!have || a.abs() < best) {
                best = a.abs();
                k = v;
                have = true;
            }
        }
        Integer ak = e.a[k];
        Integer s(ak.sign());
        std::map<uint32_t, Integer> expr;
        Integer expr_c;
        if (best.isOne()) {
            // a_k x_k + rest = 0  =>  x_k = -s * rest
            for (const auto& [v, a] : e.a) {
                if (v != k) expr[v] = -(s * a);
            }
            expr_c = -(s * e.c);
            P.erase(P.begin() + static_cast<long>(ei));
        } else {
            // sigma-equation: -s x_k + sum modhat(a_i) x_i - m sigma + modhat(c) = 0
            Integer m = best + Integer(1);
            uint32_t sigma = next_var_++;
            for (const auto& [v, a] : e.a) {
                if (v == k) continue;
                Integer r = modHat(a, m);
                if (!r.isZero()) expr[v] = s * r;
            }
            expr[sigma] = -(s * m);
            expr_c = s * modHat(e.c, m);
        }
        for (auto& con : P) substitute(con, k, expr, expr_c);
        if (!solve(std::move(P), M)) return false;
        Integer val = expr_c;
        for (const auto& [v, a] : expr) {
            auto it = M.find(v);
            if (it != M.end()) val += a * it->second;
        }
        M[k] = val;
        return true;
    }

    bool eliminateInequality(std::vector<IntCon> P, std::map<uint32_t, Integer>& M) {
        // Choose the variable: unbounded first, then exact, then fewest pairs.
        std::map<uint32_t, std::pair<int, int>> counts;  // var -> (#lower, #upper)
        std::map<uint32_t, std::pair<bool, bool>> unit;   // var -> (all lower unit, all upper unit)
        for (const auto& con : P) {
            for (const auto& [v, a] : con.a) {
                auto& c = counts[v];
                auto u = unit.emplace(v, std::make_pair(true, true)).first;
                if (a.sign() > 0) {
                    c.first++;
                    if (!a.isOne()) u->second.first = false;
                } else {
                    c.second++;
                    if (!(-a).isOne()) u->second.second = false;
                }
            }
        }
        uint32_t x = 0;
        int best_rank = -1;
        long best_pairs = 0;
        for (const auto& [v, c] : counts) {
            int rank;
            long pairs = static_cast<long>(c.first) * c.second;
            if (c.first == 0 || c.second == 0) rank = 2;
            else if (unit[v].first || unit[v].second) rank = 1;
            else rank = 0;
            if (rank > best_rank || (rank == best_rank && pairs < best_pairs)) {
                best_rank = rank;
                best_pairs = pairs;
                x = v;
            }
        }
        std::vector<IntCon> lowers, uppers, others;
        for (auto& con : P) {
            auto it = con.a.find(x);
            if (it == con.a.end()) others.push_back(con);
            else if (it->second.sign() > 0) lowers.push_back(con);
            else uppers.push_back(con);
        }
        if (lowers.empty() || uppers.empty()) {
            if (!solve(others, M)) return false;
            M[x] = pick(x, lowers, uppers, M);
            return true;
        }
        std::vector<IntCon> real = others, dark = others;
        Integer max_upper_coeff(0);
        for (const auto& up : uppers) {
            Integer b = -up.a.at(x);
            if (b > max_upper_coeff) max_upper_coeff = b;
        }
        for (const auto& lo : lowers) {
            for (const auto& up : uppers) {
                t_.tick();
                Integer a = lo.a.at(x), b = -up.a.at(x);
                IntCon r;  // a*upper + b*lower: x cancels
                for (const auto& [v, k] : lo.a) {
                    if (v != x) r.a[v] += b * k;
                }
                for (const auto& [v, k] : up.a) {
                    if (v != x) r.a[v] += a * k;
                }
                r.c = b * lo.c + a * up.c;
                IntCon d = r;
                d.c = d.c - (a - Integer(1)) * (b - Integer(1));
                real.push_back(std::move(r));
                dark.push_back(std::move(d));
            }
        }
        if (best_rank == 1) {  // exact projection
            if (!solve(std::move(real), M)) return false;
            M[x] = pick(x, lowers, uppers, M);
            return true;
        }
        {
            std::map<uint32_t, Integer> Md = M;
            if (solve(dark, Md)) {
                M = std::move(Md);
                M[x] = pick(x, lowers, uppers, M);
                return true;
            }
        }
        {
            std::map<uint32_t, Integer> Mr = M;
            if (!solve(real, Mr)) return false;
        }
        // Splinters: for each lower a x + L >= 0 and i in 0..floor((m a - a - m)/m),
        // the problem with a x + L - i = 0 added.
        for (const auto& lo : lowers) {
            Integer a = lo.a.at(x);
            Integer limit = Integer::floorDiv(max_upper_coeff * a - a - max_upper_coeff,
                                              max_upper_coeff);
            for (Integer i(0); i <= limit; i += Integer(1)) {
                t_.tick();
                std::vector<IntCon> splinter = P;
                IntCon eq = lo;
                eq.eq = true;
                eq.c = eq.c - i;
                splinter.push_back(std::move(eq));
                std::map<uint32_t, Integer> Ms = M;
                if (solve(std::move(splinter), Ms)) {
                    M = std::move(Ms);
                    return true;
                }
            }
        }
        return false;
    }

    // Least value allowed by the lower bounds (or the greatest by the upper
    // bounds when there is no lower bound) under the model of the rest.
    Integer pick(uint32_t x, const std::vector<IntCon>& lowers, const std::vector<IntCon>& uppers,
                 const std::map<uint32_t, Integer>& M) {
        if (!lowers.empty()) {
            std::optional<Integer> lb;
            for (const auto& lo : lowers) {
                Integer a = lo.a.at(x);  // a x + rest >= 0  =>  x >= ceil(-rest / a)
                Integer v = Integer::ceilDiv(-evalI(lo.a, lo.c, M, x), a);
                if (!lb || v > *lb) lb = v;
            }
            return *lb;
        }
        if (!uppers.empty()) {
            std::optional<Integer> ub;
            for (const auto& up : uppers) {
                Integer b = -up.a.at(x);  // -b x + rest >= 0  =>  x <= floor(rest / b)
                Integer v = Integer::floorDiv(evalI(up.a, up.c, M, x), b);
                if (!ub || v < *ub) ub = v;
            }
            return *ub;
        }
        return Integer(0);
    }
};

// ===========================================================================
// Projection driver
// ===========================================================================

Result decideByProjection(const std::vector<LinearConstraint>& constraints,
                          const std::vector<bool>& is_int, Ticker& t,
                          std::vector<Rational>* model) {
    std::vector<QCon> Q;
    for (const auto& lc : constraints) {
        QCon q;  // rhs - sum a x  (>= | > | =)  0
        for (const auto& [v, a] : lc.coeffs) {
            if (!a.isZero()) q.a[v] = -a;
        }
        q.c = lc.rhs;
        q.t = lc.rel == Rel::Le ? CT::GE : (lc.rel == Rel::Lt ? CT::GT : CT::EQ);
        Q.push_back(std::move(q));
    }
    // Eliminate rational variables.
    std::vector<RealElim> elims;
    for (;;) {
        std::map<uint32_t, std::pair<long, long>> counts;
        std::set<uint32_t> in_eq;
        for (const auto& q : Q) {
            for (const auto& [v, a] : q.a) {
                if (v < is_int.size() && is_int[v]) continue;
                if (q.t == CT::EQ) in_eq.insert(v);
                auto& c = counts[v];
                (a.sign() > 0 ? c.first : c.second)++;
            }
        }
        if (counts.empty()) break;
        uint32_t x = counts.begin()->first;
        bool by_eq = false;
        if (!in_eq.empty()) {
            x = *in_eq.begin();
            by_eq = true;
        } else {
            long best = -1;
            for (const auto& [v, c] : counts) {
                long pairs = c.first * c.second;
                if (best < 0 || pairs < best) {
                    best = pairs;
                    x = v;
                }
            }
        }
        RealElim el{x, by_eq, {}, {}};
        std::vector<QCon> rest, lowers, uppers;
        for (auto& q : Q) {
            if (!q.a.count(x)) rest.push_back(std::move(q));
            else el.touching.push_back(q);
        }
        if (by_eq) {
            size_t ei = 0;
            while (el.touching[ei].t != CT::EQ) ++ei;
            el.eq = el.touching[ei];
            Rational ax = el.eq.a.at(x);
            for (size_t i = 0; i < el.touching.size(); ++i) {
                if (i == ei) continue;
                QCon q = el.touching[i];
                Rational f = q.a.at(x) / ax;  // q - f * eq cancels x
                for (const auto& [v, a] : el.eq.a) {
                    Rational nv = (q.a.count(v) ? q.a[v] : Rational(0)) - f * a;
                    if (nv.isZero()) q.a.erase(v);
                    else q.a[v] = nv;
                }
                q.a.erase(x);
                q.c -= f * el.eq.c;
                rest.push_back(std::move(q));
                t.tick();
            }
        } else {
            for (const auto& q : el.touching) {
                (q.a.at(x).sign() > 0 ? lowers : uppers).push_back(q);
            }
            for (const auto& lo : lowers) {
                for (const auto& up : uppers) {
                    t.tick();
                    Rational a = lo.a.at(x), b = -up.a.at(x);
                    QCon r;
                    for (const auto& [v, k] : lo.a) {
                        if (v != x) r.a[v] += b * k;
                    }
                    for (const auto& [v, k] : up.a) {
                        if (v != x) r.a[v] += a * k;
                    }
                    for (auto it = r.a.begin(); it != r.a.end();) {
                        if (it->second.isZero()) it = r.a.erase(it);
                        else ++it;
                    }
                    r.c = b * lo.c + a * up.c;
                    r.t = (lo.t == CT::GT || up.t == CT::GT) ? CT::GT : CT::GE;
                    rest.push_back(std::move(r));
                }
            }
        }
        elims.push_back(std::move(el));
        Q = std::move(rest);
    }
    // Integer problem.
    std::vector<IntCon> P;
    uint32_t max_var = static_cast<uint32_t>(is_int.size());
    for (const auto& q : Q) {
        Integer l(1);
        auto lcm = [](const Integer& x, const Integer& y) {
            return Integer::truncDiv(x * y, Integer::gcd(x, y));
        };
        for (const auto& [v, a] : q.a) l = lcm(l, a.den());
        l = lcm(l, q.c.den());
        IntCon ic;
        for (const auto& [v, a] : q.a) ic.a[v] = (a * Rational(l)).num();
        ic.c = (q.c * Rational(l)).num();
        ic.eq = q.t == CT::EQ;
        if (q.t == CT::GT) ic.c -= Integer(1);
        P.push_back(std::move(ic));
        for (const auto& [v, a] : q.a) max_var = std::max(max_var, v + 1);
    }
    Omega omega(t, max_var);
    std::map<uint32_t, Integer> M;
    if (!omega.solve(std::move(P), M)) return Result::Unsat;
    if (!model) return Result::Sat;
    std::vector<Rational> vals(is_int.size(), Rational(0));
    for (const auto& [v, k] : M) {
        if (v < vals.size()) vals[v] = Rational(k);
    }
    for (auto it = elims.rbegin(); it != elims.rend(); ++it) {
        const RealElim& el = *it;
        if (el.by_eq) {
            vals[el.var] = -evalQ(el.eq, vals, el.var) / el.eq.a.at(el.var);
            continue;
        }
        std::optional<Rational> lo, hi;
        bool lo_strict = false, hi_strict = false;
        for (const auto& q : el.touching) {
            Rational a = q.a.at(el.var);
            Rational bound = -evalQ(q, vals, el.var) / a;
            bool strict = q.t == CT::GT;
            if (a.sign() > 0) {
                if (!lo || bound > *lo || (bound == *lo && strict)) {
                    lo = bound;
                    lo_strict = strict;
                }
            } else {
                if (!hi || bound < *hi || (bound == *hi && strict)) {
                    hi = bound;
                    hi_strict = strict;
                }
            }
        }
        Rational v(0);
        if (lo && hi) v = (*lo == *hi) ? *lo : (*lo + *hi) / Rational(2);
        else if (lo) v = *lo + Rational(lo_strict ? 1 : 0);
        else if (hi) v = *hi - Rational(hi_strict ? 1 : 0);
        vals[el.var] = v;
    }
    *model = std::move(vals);
    return Result::Sat;
}

}  // namespace

Result decideLinear(const std::vector<LinearConstraint>& constraints,
                    const std::vector<bool>& is_int, uint64_t& steps, uint64_t max_steps,
                    std::vector<Rational>* model, ArithEngine engine) {
    Ticker t{steps, max_steps};
    bool any_int = false;
    for (const auto& lc : constraints) {
        for (const auto& [v, a] : lc.coeffs) {
            if (!a.isZero() && v < is_int.size() && is_int[v]) any_int = true;
        }
    }
    try {
        if (engine == ArithEngine::Simplex || (engine == ArithEngine::Auto && !any_int)) {
            if (any_int) return Result::Unknown;
            Simplex sx(is_int.size(), t);
            for (const auto& lc : constraints) {
                if (!sx.add(lc)) return Result::Unsat;
            }
            Result r = sx.check();
            if (r == Result::Sat && model) *model = sx.model();
            return r;
        }
        return decideByProjection(constraints, is_int, t, model);
    } catch (const BudgetExhausted&) {
        return Result::Unknown;
    }
}

}  // namespace eshkol::smt
