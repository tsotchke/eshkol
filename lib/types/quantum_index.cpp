/**
 * @file quantum_index.cpp
 * @brief Index expressions of the quantum families (ADR-0048): parsing,
 *        normal form and the canonical spelling of (qreg e) / (qidx e).
 *
 * Kept apart from quantum_types.cpp so the LLVM-free workspace library
 * (parser) depends only on this file.
 */

#include "eshkol/types/quantum_types.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>

namespace eshkol::hott::quantum {

// ===========================================================================
// IndexExpr
// ===========================================================================

IndexExpr IndexExpr::nat(int64_t n) { IndexExpr e; e.constant = n; return e; }
IndexExpr IndexExpr::var(const std::string& v) { IndexExpr e; e.terms[v] = 1; return e; }
IndexExpr IndexExpr::unknown() { IndexExpr e; e.known = false; return e; }

std::optional<int64_t> IndexExpr::value() const {
    if (isConstant()) return constant;
    return std::nullopt;
}

IndexExpr IndexExpr::plus(const IndexExpr& other) const {
    if (!known || !other.known) return unknown();
    IndexExpr r = *this;
    r.constant += other.constant;
    for (const auto& [v, a] : other.terms) {
        int64_t c = (r.terms.count(v) ? r.terms[v] : 0) + a;
        if (c == 0) r.terms.erase(v); else r.terms[v] = c;
    }
    return r;
}

IndexExpr IndexExpr::scaled(int64_t k) const {
    if (!known) return unknown();
    IndexExpr r;
    if (k == 0) return r;
    r.constant = constant * k;
    for (const auto& [v, a] : terms) r.terms[v] = a * k;
    return r;
}

IndexExpr IndexExpr::substitute(const std::map<std::string, IndexExpr>& binding) const {
    if (!known) return *this;
    IndexExpr r = nat(constant);
    for (const auto& [v, a] : terms) {
        auto it = binding.find(v);
        r = r.plus(it != binding.end() ? it->second.scaled(a) : var(v).scaled(a));
        if (!r.known) return r;
    }
    return r;
}

std::vector<std::string> IndexExpr::freeVariables(const std::map<std::string, IndexExpr>& binding) const {
    std::vector<std::string> out;
    for (const auto& [v, a] : terms) {
        (void)a;
        if (!binding.count(v)) out.push_back(v);
    }
    return out;
}

std::string IndexExpr::toString() const {
    if (!known) return "?";
    std::ostringstream os;
    bool first = true;
    for (const auto& [v, a] : terms) {
        if (a < 0) os << "-"; else if (!first) os << "+";
        const int64_t m = a < 0 ? -a : a;
        if (m != 1) os << m << "*";
        os << v;
        first = false;
    }
    if (first) {
        os << constant;
    } else if (constant > 0) {
        os << "+" << constant;
    } else if (constant < 0) {
        os << constant;
    }
    return os.str();
}

namespace {

bool identStart(char c) { return std::isalpha(static_cast<unsigned char>(c)) || c == '_'; }
bool identChar(char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }

}  // namespace

std::optional<IndexExpr> IndexExpr::parse(const std::string& text) {
    if (text == "?") return unknown();
    size_t i = 0;
    const size_t n = text.size();
    if (n == 0) return std::nullopt;
    IndexExpr acc = nat(0);
    int sign = 1;
    bool expect_term = true;
    if (text[0] == '-') { sign = -1; i = 1; }
    while (i < n) {
        if (!expect_term) {
            if (text[i] == '+') sign = 1;
            else if (text[i] == '-') sign = -1;
            else return std::nullopt;
            ++i;
            expect_term = true;
            continue;
        }
        // term := factor ('*' factor)?  with at most one variable factor
        int64_t coef = 1;
        std::string name;
        for (int factor = 0; factor < 2; ++factor) {
            if (i >= n) return std::nullopt;
            if (std::isdigit(static_cast<unsigned char>(text[i]))) {
                int64_t k = 0;
                while (i < n && std::isdigit(static_cast<unsigned char>(text[i]))) {
                    k = k * 10 + (text[i] - '0');
                    if (k > (int64_t{1} << 40)) return std::nullopt;
                    ++i;
                }
                coef *= k;
            } else if (identStart(text[i])) {
                if (!name.empty()) return std::nullopt;
                size_t s = i;
                while (i < n && identChar(text[i])) ++i;
                name = text.substr(s, i - s);
            } else {
                return std::nullopt;
            }
            if (i < n && text[i] == '*' && factor == 0) { ++i; continue; }
            break;
        }
        IndexExpr term = name.empty() ? nat(coef) : IndexExpr::var(name).scaled(coef);
        acc = acc.plus(term.scaled(sign));
        expect_term = false;
    }
    if (expect_term) return std::nullopt;
    return acc;
}

namespace detail {

// A tiny reader for the table above: atoms and parenthesised lists.
bool readSx(const std::string& s, size_t& i, Sx& out) {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i >= s.size()) return false;
    if (s[i] == '(') {
        ++i;
        out.list = true;
        while (true) {
            while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
            if (i >= s.size()) return false;
            if (s[i] == ')') { ++i; return true; }
            Sx item;
            if (!readSx(s, i, item)) return false;
            out.items.push_back(item);
        }
    }
    size_t start = i;
    while (i < s.size() && !std::isspace(static_cast<unsigned char>(s[i])) && s[i] != '(' && s[i] != ')') ++i;
    out.atom = s.substr(start, i - start);
    return !out.atom.empty();
}

}  // namespace detail

std::optional<std::string> canonicalIndexedSpelling(const std::string& family,
                                                    const std::string& index_sexpr) {
    detail::Sx sx;
    size_t i = 0;
    if (!detail::readSx(index_sexpr, i, sx)) return std::nullopt;
    std::function<std::optional<IndexExpr>(const detail::Sx&)> eval = [&](const detail::Sx& x) -> std::optional<IndexExpr> {
        if (!x.list) return IndexExpr::parse(x.atom);
        if (x.items.empty() || x.items[0].list) return std::nullopt;
        const std::string op = x.items[0].atom;
        std::vector<IndexExpr> vals;
        for (size_t k = 1; k < x.items.size(); ++k) {
            auto v = eval(x.items[k]);
            if (!v) return std::nullopt;
            vals.push_back(*v);
        }
        if (vals.empty()) return std::nullopt;
        if (op == "+") {
            IndexExpr acc = IndexExpr::nat(0);
            for (const auto& v : vals) acc = acc.plus(v);
            return acc;
        }
        if (op == "-") {
            if (vals.size() == 1) return vals[0].scaled(-1);
            IndexExpr acc = vals[0];
            for (size_t k = 1; k < vals.size(); ++k) acc = acc.plus(vals[k].scaled(-1));
            return acc;
        }
        if (op == "*" && vals.size() == 2) {
            if (auto c = vals[0].value()) return vals[1].scaled(*c);
            if (auto c = vals[1].value()) return vals[0].scaled(*c);
        }
        return std::nullopt;
    };
    auto e = eval(sx);
    if (!e) return std::nullopt;
    std::string fam = family;
    std::transform(fam.begin(), fam.end(), fam.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return fam + "<" + e->toString() + ">";
}

}  // namespace eshkol::hott::quantum
