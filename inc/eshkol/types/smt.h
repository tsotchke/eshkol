/**
 * @file smt.h
 * @brief Built-in decision procedure for refinement-type verification
 *        conditions (ADR-0044).
 *
 * The procedure is self-contained: it needs no external solver and runs on
 * every platform Eshkol builds on. It decides quantifier-free formulas over
 *
 *   - linear integer arithmetic (Presburger, by the Omega test),
 *   - linear rational arithmetic (exact general simplex over Q, with strict
 *     bounds as delta-rationals; Fourier-Motzkin projection when integer and
 *     rational variables are mixed),
 *   - uninterpreted functions with equality (congruence closure),
 *
 * combined by a DPLL(T) search with delayed theory combination: every
 * equality between variables shared by the arithmetic and the equality
 * theories is a Boolean decision, so both theories agree on one arrangement
 * (Nelson-Oppen for stably infinite theories).
 *
 * Numbers are exact. Integer wraps Eshkol's arbitrary-precision bignum
 * (lib/core/bignum.cpp) with an int64 fast path; Rational is a normalised
 * pair of Integers. Nothing is rounded anywhere in the procedure.
 *
 * The answer is three-valued. Unsat is a proof (relative to the procedure).
 * Sat carries a model that has been re-evaluated against the original
 * formula with the exact semantics of every interpreted symbol; a model that
 * fails that evaluation (possible only when a nonlinear product was
 * over-approximated as an uninterpreted function) degrades to Unknown.
 * Unknown is also the answer when the deterministic step budget runs out.
 * Unknown never counts as success.
 *
 * An optional z3 backend (SMT-LIB2 over a pipe) is compiled when the build
 * option ESHKOL_REFINEMENT_Z3 is ON; makeZ3Solver() returns nullptr otherwise.
 */
#ifndef ESHKOL_TYPES_SMT_H
#define ESHKOL_TYPES_SMT_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

struct arena;
struct eshkol_bignum;

namespace eshkol::smt {

// ===========================================================================
// Exact numbers
// ===========================================================================

/**
 * @brief RAII owner of the arena that backs every arbitrary-precision value
 * created on this thread while it is alive.
 *
 * Integers whose magnitude exceeds int64 hold a pointer into this arena, so
 * every Integer/Rational/Term created inside a scope must be dropped before
 * the scope ends. Scopes nest; the innermost is current. Without any scope a
 * process-lifetime arena is used.
 */
class ExactScope {
public:
    ExactScope();
    ~ExactScope();
    ExactScope(const ExactScope&) = delete;
    ExactScope& operator=(const ExactScope&) = delete;

    /** The arena new bignums are allocated in on this thread. */
    static struct arena* current();

private:
    struct arena* arena_;
    struct arena* previous_;
};

/** Arbitrary-precision integer: int64 fast path, Eshkol bignum beyond it. */
class Integer {
public:
    Integer() : small_(0), big_(nullptr) {}
    Integer(int64_t v) : small_(v), big_(nullptr) {}  // NOLINT: implicit by design

    static std::optional<Integer> parse(const std::string& decimal);

    Integer operator+(const Integer& o) const;
    Integer operator-(const Integer& o) const;
    Integer operator*(const Integer& o) const;
    Integer operator-() const;
    Integer& operator+=(const Integer& o) { return *this = *this + o; }
    Integer& operator-=(const Integer& o) { return *this = *this - o; }
    Integer& operator*=(const Integer& o) { return *this = *this * o; }

    /** Quotient rounded toward negative infinity. @p d must be nonzero. */
    static Integer floorDiv(const Integer& n, const Integer& d);
    /** Quotient rounded toward positive infinity. @p d must be nonzero. */
    static Integer ceilDiv(const Integer& n, const Integer& d);
    /** Quotient rounded toward zero. @p d must be nonzero. */
    static Integer truncDiv(const Integer& n, const Integer& d);
    /** n - d*floorDiv(n,d): result has the sign of d (or is zero). */
    static Integer floorMod(const Integer& n, const Integer& d);
    /** Non-negative greatest common divisor; gcd(0,0) = 0. */
    static Integer gcd(const Integer& a, const Integer& b);

    int compare(const Integer& o) const;
    bool operator==(const Integer& o) const { return compare(o) == 0; }
    bool operator!=(const Integer& o) const { return compare(o) != 0; }
    bool operator<(const Integer& o) const { return compare(o) < 0; }
    bool operator<=(const Integer& o) const { return compare(o) <= 0; }
    bool operator>(const Integer& o) const { return compare(o) > 0; }
    bool operator>=(const Integer& o) const { return compare(o) >= 0; }

    int sign() const;
    bool isZero() const { return sign() == 0; }
    bool isOne() const { return !big_ && small_ == 1; }
    Integer abs() const { return sign() < 0 ? -*this : *this; }
    bool fitsInt64(int64_t* out) const;
    std::string str() const;
    size_t hash() const;

private:
    int64_t small_;
    const struct eshkol_bignum* big_;  // non-null only when the value does not fit int64

    static Integer fromBig(const struct eshkol_bignum* b);
    const struct eshkol_bignum* toBig() const;
};

/** Exact rational number, always normalised (gcd(num,den)=1, den>0). */
class Rational {
public:
    Rational() : num_(0), den_(1) {}
    Rational(int64_t v) : num_(v), den_(1) {}            // NOLINT
    Rational(const Integer& v) : num_(v), den_(1) {}      // NOLINT
    Rational(const Integer& n, const Integer& d);

    static std::optional<Rational> parse(const std::string& text);  // "n", "-n", "n/d"
    /** The exact value of a finite double. */
    static Rational fromDouble(double d);

    const Integer& num() const { return num_; }
    const Integer& den() const { return den_; }

    Rational operator+(const Rational& o) const;
    Rational operator-(const Rational& o) const;
    Rational operator*(const Rational& o) const;
    Rational operator/(const Rational& o) const;  // o must be nonzero
    Rational operator-() const;
    Rational& operator+=(const Rational& o) { return *this = *this + o; }
    Rational& operator-=(const Rational& o) { return *this = *this - o; }
    Rational& operator*=(const Rational& o) { return *this = *this * o; }

    int compare(const Rational& o) const;
    bool operator==(const Rational& o) const { return num_ == o.num_ && den_ == o.den_; }
    bool operator!=(const Rational& o) const { return !(*this == o); }
    bool operator<(const Rational& o) const { return compare(o) < 0; }
    bool operator<=(const Rational& o) const { return compare(o) <= 0; }
    bool operator>(const Rational& o) const { return compare(o) > 0; }
    bool operator>=(const Rational& o) const { return compare(o) >= 0; }

    int sign() const { return num_.sign(); }
    bool isZero() const { return num_.isZero(); }
    bool isInteger() const { return den_.isOne(); }
    Integer floor() const { return Integer::floorDiv(num_, den_); }
    Integer ceil() const { return Integer::ceilDiv(num_, den_); }
    Rational abs() const { return sign() < 0 ? -*this : *this; }
    std::string str() const;  // "n" or "n/d"
    size_t hash() const { return num_.hash() * 1000003u ^ den_.hash(); }

private:
    Integer num_;
    Integer den_;
};

// ===========================================================================
// Terms
// ===========================================================================

enum class Sort : uint8_t { Bool, Int, Real, Obj };

const char* sortName(Sort s);

enum class Kind : uint8_t {
    BoolConst,  // bval
    Numeral,    // value; sort Int when integral and requested so, else Real
    Var,        // name
    App,        // uninterpreted (or over-approximated interpreted) function
    Add,        // n-ary sum
    Scale,      // value * args[0]
    Ite,        // args = {cond, then, else}
    Eq,         // args = {a, b}; any sort
    Le,         // a <= b
    Lt,         // a < b
    Not,
    And,
    Or,
    Implies,
};

struct Node {
    Kind kind;
    Sort sort;
    bool bval = false;
    Rational value;
    std::string name;  // Var name or App function symbol
    std::vector<const Node*> args;
    uint32_t id = 0;
};
using Term = const Node*;

/**
 * Function symbols the procedure treats as uninterpreted during search but
 * evaluates exactly when it validates a model. A product of two non-constant
 * terms, integer division and remainder by a non-constant all land here.
 */
bool isOverApproximatedSymbol(const std::string& fn);

/** Hash-consing term factory with light, semantics-preserving simplification. */
class TermManager {
public:
    TermManager();
    ~TermManager();
    TermManager(const TermManager&) = delete;
    TermManager& operator=(const TermManager&) = delete;

    Term mkTrue();
    Term mkFalse();
    Term mkBool(bool b) { return b ? mkTrue() : mkFalse(); }
    Term mkInt(const Integer& v);
    Term mkReal(const Rational& v);
    Term mkNumeral(const Rational& v, Sort s);
    Term mkVar(const std::string& name, Sort s);
    /** f(args) with result sort @p s; the signature is recorded on first use. */
    Term mkApp(const std::string& fn, Sort s, const std::vector<Term>& args);

    Term mkAdd(const std::vector<Term>& args);
    Term mkAdd(Term a, Term b) { return mkAdd(std::vector<Term>{a, b}); }
    Term mkSub(Term a, Term b);
    Term mkNeg(Term a);
    Term mkScale(const Rational& c, Term t);
    /** Product: linear when either side is a numeral, else the symbol "*". */
    Term mkMul(Term a, Term b);
    Term mkIte(Term c, Term t, Term e);

    Term mkEq(Term a, Term b);
    Term mkDistinct(Term a, Term b) { return mkNot(mkEq(a, b)); }
    Term mkLe(Term a, Term b);
    Term mkLt(Term a, Term b);
    Term mkGe(Term a, Term b) { return mkLe(b, a); }
    Term mkGt(Term a, Term b) { return mkLt(b, a); }
    Term mkNot(Term a);
    Term mkAnd(const std::vector<Term>& args);
    Term mkAnd(Term a, Term b) { return mkAnd(std::vector<Term>{a, b}); }
    Term mkOr(const std::vector<Term>& args);
    Term mkOr(Term a, Term b) { return mkOr(std::vector<Term>{a, b}); }
    Term mkImplies(Term a, Term b);
    Term mkIff(Term a, Term b) { return mkEq(a, b); }

    /** Declared signature of an uninterpreted symbol, if it has been used. */
    struct Signature { std::vector<Sort> args; Sort result; };
    const std::map<std::string, Signature>& signatures() const { return signatures_; }
    /** Every variable created so far, in creation order. */
    const std::vector<Term>& variables() const { return vars_; }

    /** A variable name guaranteed fresh in this manager. */
    std::string freshName(const std::string& stem);

    size_t size() const { return nodes_.size(); }

private:
    Term intern(Node&& n);
    std::vector<std::unique_ptr<Node>> nodes_;
    std::unordered_map<std::string, Term> table_;
    std::map<std::string, Signature> signatures_;
    std::vector<Term> vars_;
    std::map<std::string, Sort> var_sorts_;
    uint64_t fresh_counter_ = 0;
};

/** Render a term in SMT-LIB2 syntax (Int terms are lifted with to_real where needed). */
std::string toSmtLib2(Term t);
/** A complete SMT-LIB2 script: declarations, one assert per formula, (check-sat). */
std::string toSmtLib2Script(const TermManager& tm, const std::vector<Term>& assertions,
                            bool with_check_sat = true);

// ===========================================================================
// Answers and models
// ===========================================================================

enum class Result { Sat, Unsat, Unknown };
const char* resultName(Result r);

/** A satisfying assignment, keyed by variable name. */
struct Model {
    std::map<std::string, Rational> numbers;   // Int and Real variables
    std::map<std::string, bool> booleans;      // Bool variables
    std::map<std::string, int> objects;        // Obj variables: equivalence-class ids
    /** Values of uninterpreted applications, keyed by their SMT-LIB2 text. */
    std::map<std::string, std::string> applications;
};

/** Evaluate a formula under @p m; nullopt when a variable is unassigned. */
std::optional<bool> evaluate(Term formula, const Model& m);

struct Outcome {
    Result result = Result::Unknown;
    Model model;          // meaningful when result == Sat
    std::string reason;   // why the answer is Unknown
    uint64_t steps = 0;   // budget consumed
};

/** Deterministic resource limit: one step = one search decision, pivot or elimination. */
struct Budget {
    uint64_t max_steps = 2000000;
};

/** A decision procedure for conjunctions of quantifier-free formulas. */
class Solver {
public:
    virtual ~Solver() = default;
    virtual Outcome check(TermManager& tm, const std::vector<Term>& assertions) = 0;
    virtual std::string name() const = 0;
};

/** The built-in procedure described at the top of this header. */
std::unique_ptr<Solver> makeInternalSolver(Budget budget = Budget{});

/** True when this build includes the z3 backend (ESHKOL_REFINEMENT_Z3=ON). */
bool z3BackendCompiled();
/**
 * The z3 backend: SMT-LIB2 over a pipe to `z3 -in -smt2`. @p z3_path empty
 * means the ESHKOL_Z3 environment variable, then the path found at configure
 * time. Returns nullptr when the backend is not compiled in.
 */
std::unique_ptr<Solver> makeZ3Solver(const std::string& z3_path = std::string(),
                                     uint64_t rlimit = 5000000);

// ===========================================================================
// Theory engines (exposed for unit tests and cross-checks)
// ===========================================================================

enum class Rel : uint8_t { Le, Lt, Eq };

/** sum(coeffs[i] * x_i) rel rhs, over variables 0..n-1. */
struct LinearConstraint {
    std::map<uint32_t, Rational> coeffs;
    Rel rel = Rel::Le;
    Rational rhs;
};

enum class ArithEngine { Auto, Simplex, Projection };

/**
 * Decide a conjunction of linear constraints. @p is_int[i] marks integer
 * variables. Auto uses the simplex for purely rational problems and
 * Fourier-Motzkin projection of the rational variables followed by the Omega
 * test otherwise; the other two force one engine (Simplex rejects integer
 * variables with Unknown). On Sat, @p model receives one value per variable.
 */
Result decideLinear(const std::vector<LinearConstraint>& constraints,
                    const std::vector<bool>& is_int,
                    uint64_t& steps, uint64_t max_steps,
                    std::vector<Rational>* model,
                    ArithEngine engine = ArithEngine::Auto);

}  // namespace eshkol::smt

#endif  // ESHKOL_TYPES_SMT_H
