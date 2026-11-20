/**
 * @file exact.cpp
 * @brief Exact Integer and Rational for the refinement decision procedure,
 *        built on Eshkol's arbitrary-precision bignum (lib/core/bignum.cpp).
 *
 * An Integer is an int64 whenever its value fits one; only results that
 * overflow are promoted to a bignum allocated in the current ExactScope's
 * arena. Every result is renormalised, so a value has exactly one
 * representation and equality is structural.
 */
#include "eshkol/types/smt.h"

#include "eshkol/core/bignum.h"
#include "../../core/arena_memory.h"

#include <cmath>
#include <cstring>
#include <functional>

namespace eshkol::smt {

// ---------------------------------------------------------------------------
// Arena scope
// ---------------------------------------------------------------------------

static thread_local arena_t* t_exact_arena = nullptr;

static arena_t* processArena() {
    static arena_t* a = arena_create_threadsafe(64 * 1024);
    return a;
}

ExactScope::ExactScope() : arena_(arena_create(64 * 1024)), previous_(t_exact_arena) {
    t_exact_arena = arena_;
}

ExactScope::~ExactScope() {
    t_exact_arena = previous_;
    if (arena_) arena_destroy(arena_);
}

arena_t* ExactScope::current() {
    return t_exact_arena ? t_exact_arena : processArena();
}

// ---------------------------------------------------------------------------
// Integer
// ---------------------------------------------------------------------------

Integer Integer::fromBig(const eshkol_bignum_t* b) {
    Integer r;
    int64_t v = 0;
    if (eshkol_bignum_fits_int64(b, &v)) {
        r.small_ = v;
        r.big_ = nullptr;
    } else {
        r.small_ = 0;
        r.big_ = b;
    }
    return r;
}

const eshkol_bignum_t* Integer::toBig() const {
    if (big_) return big_;
    return eshkol_bignum_from_int64(ExactScope::current(), small_);
}

std::optional<Integer> Integer::parse(const std::string& decimal) {
    if (decimal.empty()) return std::nullopt;
    size_t i = (decimal[0] == '-' || decimal[0] == '+') ? 1 : 0;
    if (i == decimal.size()) return std::nullopt;
    for (size_t k = i; k < decimal.size(); ++k) {
        if (decimal[k] < '0' || decimal[k] > '9') return std::nullopt;
    }
    eshkol_bignum_t* b = eshkol_bignum_from_string(ExactScope::current(), decimal.c_str(),
                                                   decimal.size());
    if (!b) return std::nullopt;
    return fromBig(b);
}

Integer Integer::operator+(const Integer& o) const {
    if (!big_ && !o.big_) {
        int64_t r;
        if (!__builtin_add_overflow(small_, o.small_, &r)) return Integer(r);
    }
    return fromBig(eshkol_bignum_add(ExactScope::current(), toBig(), o.toBig()));
}

Integer Integer::operator-(const Integer& o) const {
    if (!big_ && !o.big_) {
        int64_t r;
        if (!__builtin_sub_overflow(small_, o.small_, &r)) return Integer(r);
    }
    return fromBig(eshkol_bignum_sub(ExactScope::current(), toBig(), o.toBig()));
}

Integer Integer::operator*(const Integer& o) const {
    if (!big_ && !o.big_) {
        int64_t r;
        if (!__builtin_mul_overflow(small_, o.small_, &r)) return Integer(r);
    }
    return fromBig(eshkol_bignum_mul(ExactScope::current(), toBig(), o.toBig()));
}

Integer Integer::operator-() const {
    if (!big_ && small_ != INT64_MIN) return Integer(-small_);
    return fromBig(eshkol_bignum_neg(ExactScope::current(), toBig()));
}

int Integer::compare(const Integer& o) const {
    if (!big_ && !o.big_) return (small_ < o.small_) ? -1 : (small_ > o.small_ ? 1 : 0);
    int c = eshkol_bignum_compare(toBig(), o.toBig());
    return c < 0 ? -1 : (c > 0 ? 1 : 0);
}

int Integer::sign() const {
    if (!big_) return small_ < 0 ? -1 : (small_ > 0 ? 1 : 0);
    return eshkol_bignum_is_negative(big_) ? -1 : (eshkol_bignum_is_zero(big_) ? 0 : 1);
}

Integer Integer::truncDiv(const Integer& n, const Integer& d) {
    if (!n.big_ && !d.big_ && !(n.small_ == INT64_MIN && d.small_ == -1)) {
        return Integer(n.small_ / d.small_);
    }
    return fromBig(eshkol_bignum_div(ExactScope::current(), n.toBig(), d.toBig()));
}

Integer Integer::floorDiv(const Integer& n, const Integer& d) {
    Integer q = truncDiv(n, d);
    Integer r = n - q * d;
    if (!r.isZero() && (r.sign() != d.sign())) q -= Integer(1);
    return q;
}

Integer Integer::ceilDiv(const Integer& n, const Integer& d) {
    return -floorDiv(-n, d);
}

Integer Integer::floorMod(const Integer& n, const Integer& d) {
    return n - d * floorDiv(n, d);
}

Integer Integer::gcd(const Integer& a, const Integer& b) {
    Integer x = a.abs();
    Integer y = b.abs();
    while (!y.isZero()) {
        Integer t = x - y * truncDiv(x, y);
        x = y;
        y = t;
    }
    return x;
}

bool Integer::fitsInt64(int64_t* out) const {
    if (big_) return false;
    if (out) *out = small_;
    return true;
}

std::string Integer::str() const {
    if (!big_) return std::to_string(small_);
    char* s = eshkol_bignum_to_string(ExactScope::current(), big_);
    return s ? std::string(s) : std::string("?");
}

size_t Integer::hash() const {
    if (!big_) return std::hash<int64_t>()(small_);
    return std::hash<std::string>()(str());
}

// ---------------------------------------------------------------------------
// Rational
// ---------------------------------------------------------------------------

Rational::Rational(const Integer& n, const Integer& d) : num_(n), den_(d) {
    if (den_.sign() < 0) {
        num_ = -num_;
        den_ = -den_;
    }
    Integer g = Integer::gcd(num_, den_);
    if (!g.isZero() && !g.isOne()) {
        num_ = Integer::truncDiv(num_, g);
        den_ = Integer::truncDiv(den_, g);
    }
    if (num_.isZero()) den_ = Integer(1);
}

std::optional<Rational> Rational::parse(const std::string& text) {
    size_t slash = text.find('/');
    if (slash == std::string::npos) {
        auto n = Integer::parse(text);
        if (!n) return std::nullopt;
        return Rational(*n);
    }
    auto n = Integer::parse(text.substr(0, slash));
    auto d = Integer::parse(text.substr(slash + 1));
    if (!n || !d || d->isZero()) return std::nullopt;
    return Rational(*n, *d);
}

Rational Rational::fromDouble(double d) {
    // Every finite double is m * 2^e exactly, with m an integer of at most 53 bits.
    if (!std::isfinite(d) || d == 0.0) return Rational(0);
    int exp = 0;
    double frac = std::frexp(d, &exp);           // d = frac * 2^exp, 0.5 <= |frac| < 1
    double mant = std::ldexp(frac, 53);          // integral
    exp -= 53;
    Integer m(static_cast<int64_t>(mant));
    Integer pow2(1);
    int e = exp < 0 ? -exp : exp;
    Integer two(2);
    for (int i = 0; i < e; ++i) pow2 *= two;
    if (exp >= 0) return Rational(m * pow2);
    return Rational(m, pow2);
}

Rational Rational::operator+(const Rational& o) const {
    if (den_.isOne() && o.den_.isOne()) return Rational(num_ + o.num_);
    return Rational(num_ * o.den_ + o.num_ * den_, den_ * o.den_);
}

Rational Rational::operator-(const Rational& o) const {
    if (den_.isOne() && o.den_.isOne()) return Rational(num_ - o.num_);
    return Rational(num_ * o.den_ - o.num_ * den_, den_ * o.den_);
}

Rational Rational::operator*(const Rational& o) const {
    if (den_.isOne() && o.den_.isOne()) return Rational(num_ * o.num_);
    return Rational(num_ * o.num_, den_ * o.den_);
}

Rational Rational::operator/(const Rational& o) const {
    return Rational(num_ * o.den_, den_ * o.num_);
}

Rational Rational::operator-() const {
    Rational r;
    r.num_ = -num_;
    r.den_ = den_;
    return r;
}

int Rational::compare(const Rational& o) const {
    if (den_ == o.den_) return num_.compare(o.num_);
    return (num_ * o.den_).compare(o.num_ * den_);
}

std::string Rational::str() const {
    if (den_.isOne()) return num_.str();
    return num_.str() + "/" + den_.str();
}

}  // namespace eshkol::smt
