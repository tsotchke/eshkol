/**
 * @file type_term.cpp
 * @brief Interned core terms of the dependent type language. See type_term.h
 *        and docs/design/adr/0042-dependent-type-enforcement.md.
 */
#include "eshkol/types/type_term.h"

#include <algorithm>
#include <limits>

namespace eshkol::hott {

namespace {

constexpr int kMaxDepth = 512;  // normalization and unification recursion bound

bool isArithmeticHead(const std::string& head) {
    return head == "+" || head == "-" || head == "*";
}

bool fitsInt64(__int128 v) {
    return v >= static_cast<__int128>(std::numeric_limits<int64_t>::min()) &&
           v <= static_cast<__int128>(std::numeric_limits<int64_t>::max());
}

}  // namespace

size_t TermStore::KeyHash::operator()(const Key& k) const {
    size_t h = std::hash<int>()(static_cast<int>(k.kind));
    h ^= std::hash<int64_t>()(k.value) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    h ^= std::hash<std::string>()(k.name) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    for (TermRef a : k.args) {
        h ^= std::hash<uint32_t>()(a) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return h;
}

TermStore::TermStore() {
    nodes_.push_back(TermNode{});  // index 0: kNoTerm
}

TermRef TermStore::intern(TermNode node, bool binder_name_in_key) {
    Key key{node.kind, node.value,
            (node.kind == TermKind::Pi || node.kind == TermKind::Sigma) && !binder_name_in_key
                ? std::string()
                : node.name,
            node.args};
    auto it = interned_.find(key);
    if (it != interned_.end()) return it->second;
    const TermRef ref = static_cast<TermRef>(nodes_.size());
    nodes_.push_back(std::move(node));
    interned_.emplace(std::move(key), ref);
    return ref;
}

TermRef TermStore::universe(uint32_t level) {
    TermNode n;
    n.kind = TermKind::Universe;
    n.value = level;
    return intern(std::move(n));
}

TermRef TermStore::constant(uint32_t packed_type_id) {
    TermNode n;
    n.kind = TermKind::Const;
    n.value = packed_type_id & 0xFFFFu;  // identity is the numeric id
    return intern(std::move(n));
}

TermRef TermStore::bound(uint32_t index) {
    TermNode n;
    n.kind = TermKind::Bound;
    n.value = index;
    return intern(std::move(n));
}

TermRef TermStore::freeVar(uint64_t stamp, const std::string& display) {
    // The stamp is the identity; the spelling is for printing, and is the
    // same for every occurrence of one binding.
    TermNode n;
    n.kind = TermKind::Free;
    n.value = static_cast<int64_t>(stamp);
    n.name = display;
    return intern(std::move(n));
}

TermRef TermStore::freshMeta() {
    TermNode n;
    n.kind = TermKind::Meta;
    n.value = next_meta_++;
    return intern(std::move(n));
}

TermRef TermStore::intLit(int64_t value) {
    TermNode n;
    n.kind = TermKind::IntLit;
    n.value = value;
    return intern(std::move(n));
}

TermRef TermStore::op(const std::string& head, std::vector<TermRef> args) {
    TermNode n;
    n.kind = TermKind::Op;
    n.name = head;
    n.args = std::move(args);
    return intern(std::move(n));
}

TermRef TermStore::pi(std::vector<TermRef> domains, TermRef body, int64_t flags,
                      std::vector<std::string> hints) {
    TermNode n;
    n.kind = TermKind::Pi;
    n.value = flags;
    n.args = std::move(domains);
    n.args.push_back(body);
    n.hints = std::move(hints);
    return intern(std::move(n));
}

TermRef TermStore::sigma(TermRef first, TermRef second, const std::string& hint) {
    TermNode n;
    n.kind = TermKind::Sigma;
    n.args = {first, second};
    n.hints = {hint};
    return intern(std::move(n));
}

TermRef TermStore::app(const std::string& family, std::vector<TermRef> args) {
    TermNode n;
    n.kind = TermKind::App;
    n.name = family;
    n.args = std::move(args);
    return intern(std::move(n));
}

TermRef TermStore::rebuild(TermRef t, const std::vector<TermRef>& args) {
    if (nodes_[t].args == args) return t;
    TermNode n = nodes_[t];
    n.args = args;
    return intern(std::move(n));
}

// ============================================================================
// Binders
// ============================================================================

TermRef TermStore::shift(TermRef t, int64_t delta, uint32_t cutoff) {
    if (t == kNoTerm || delta == 0) return t;
    const TermNode& n = nodes_[t];
    switch (n.kind) {
        case TermKind::Bound:
            if (n.value >= cutoff) return bound(static_cast<uint32_t>(n.value + delta));
            return t;
        case TermKind::Pi: {
            std::vector<TermRef> args = n.args;
            for (size_t i = 0; i < args.size(); ++i) {
                args[i] = shift(args[i], delta, cutoff + static_cast<uint32_t>(i));
            }
            return rebuild(t, args);
        }
        case TermKind::Sigma: {
            std::vector<TermRef> args = n.args;
            args[0] = shift(args[0], delta, cutoff);
            args[1] = shift(args[1], delta, cutoff + 1);
            return rebuild(t, args);
        }
        case TermKind::Op:
        case TermKind::App: {
            std::vector<TermRef> args = n.args;
            for (auto& a : args) a = shift(a, delta, cutoff);
            return rebuild(t, args);
        }
        default:
            return t;
    }
}

TermRef TermStore::substitute(TermRef body, uint32_t index, TermRef value) {
    if (body == kNoTerm) return body;
    const TermNode& n = nodes_[body];
    switch (n.kind) {
        case TermKind::Bound: {
            const int64_t k = n.value;
            if (k == index) return shift(value, index);
            if (k > index) return bound(static_cast<uint32_t>(k - 1));
            return body;
        }
        case TermKind::Pi: {
            std::vector<TermRef> args = n.args;
            for (size_t i = 0; i < args.size(); ++i) {
                args[i] = substitute(args[i], index + static_cast<uint32_t>(i), value);
            }
            return rebuild(body, args);
        }
        case TermKind::Sigma: {
            std::vector<TermRef> args = n.args;
            args[0] = substitute(args[0], index, value);
            args[1] = substitute(args[1], index + 1, value);
            return rebuild(body, args);
        }
        case TermKind::Op:
        case TermKind::App: {
            std::vector<TermRef> args = n.args;
            for (auto& a : args) a = substitute(a, index, value);
            return rebuild(body, args);
        }
        default:
            return body;
    }
}

TermRef TermStore::instantiate(TermRef body, const std::vector<TermRef>& values) {
    TermRef t = body;
    for (size_t k = values.size(); k-- > 0;) {
        t = substitute(t, 0, values[k]);
    }
    return t;
}

TermRef TermStore::abstractFrees(TermRef t, const std::vector<uint64_t>& stamps, uint32_t depth) {
    if (t == kNoTerm || stamps.empty()) return t;
    const TermNode& n = nodes_[t];
    switch (n.kind) {
        case TermKind::Free: {
            for (size_t k = 0; k < stamps.size(); ++k) {
                if (static_cast<uint64_t>(n.value) == stamps[k]) {
                    return bound(static_cast<uint32_t>(stamps.size() - 1 - k + depth));
                }
            }
            return t;
        }
        case TermKind::Pi: {
            std::vector<TermRef> args = n.args;
            for (size_t i = 0; i < args.size(); ++i) {
                args[i] = abstractFrees(args[i], stamps, depth + static_cast<uint32_t>(i));
            }
            return rebuild(t, args);
        }
        case TermKind::Sigma: {
            std::vector<TermRef> args = n.args;
            args[0] = abstractFrees(args[0], stamps, depth);
            args[1] = abstractFrees(args[1], stamps, depth + 1);
            return rebuild(t, args);
        }
        case TermKind::Op:
        case TermKind::App: {
            std::vector<TermRef> args = n.args;
            for (auto& a : args) a = abstractFrees(a, stamps, depth);
            return rebuild(t, args);
        }
        default:
            return t;
    }
}

bool TermStore::mentionsBound(TermRef t, uint32_t index) const {
    if (t == kNoTerm) return false;
    const TermNode& n = nodes_[t];
    switch (n.kind) {
        case TermKind::Bound:
            return n.value == index;
        case TermKind::Pi:
            for (size_t i = 0; i < n.args.size(); ++i) {
                if (mentionsBound(n.args[i], index + static_cast<uint32_t>(i))) return true;
            }
            return false;
        case TermKind::Sigma:
            return mentionsBound(n.args[0], index) || mentionsBound(n.args[1], index + 1);
        case TermKind::Op:
        case TermKind::App:
            for (TermRef a : n.args) {
                if (mentionsBound(a, index)) return true;
            }
            return false;
        default:
            return false;
    }
}

bool TermStore::hasLooseBound(TermRef t, uint32_t depth) const {
    if (t == kNoTerm) return false;
    const TermNode& n = nodes_[t];
    switch (n.kind) {
        case TermKind::Bound:
            return n.value >= depth;
        case TermKind::Pi:
            for (size_t i = 0; i < n.args.size(); ++i) {
                if (hasLooseBound(n.args[i], depth + static_cast<uint32_t>(i))) return true;
            }
            return false;
        case TermKind::Sigma:
            return hasLooseBound(n.args[0], depth) || hasLooseBound(n.args[1], depth + 1);
        case TermKind::Op:
        case TermKind::App:
            for (TermRef a : n.args) {
                if (hasLooseBound(a, depth)) return true;
            }
            return false;
        default:
            return false;
    }
}

bool TermStore::hasUnsolvedMetas(TermRef t) {
    if (t == kNoTerm) return false;
    const TermNode n = nodes_[t];
    if (n.kind == TermKind::Meta) {
        auto sol = metaSolution(n.value);
        return sol ? hasUnsolvedMetas(*sol) : true;
    }
    for (TermRef a : n.args) {
        if (hasUnsolvedMetas(a)) return true;
    }
    return false;
}

// ============================================================================
// Definitions and metavariables
// ============================================================================

void TermStore::defineFree(uint64_t stamp, TermRef value) {
    if (value != kNoTerm) free_definitions_[stamp] = value;
}

std::optional<TermRef> TermStore::metaSolution(int64_t id) const {
    auto it = meta_solutions_.find(id);
    if (it == meta_solutions_.end()) return std::nullopt;
    return it->second;
}

void TermStore::solveMeta(int64_t id, TermRef value) {
    meta_solutions_[id] = value;
}

bool TermStore::occurs(int64_t meta, TermRef t) {
    if (t == kNoTerm) return false;
    const TermNode n = nodes_[t];
    if (n.kind == TermKind::Meta) {
        if (n.value == meta) return true;
        auto sol = metaSolution(n.value);
        return sol ? occurs(meta, *sol) : false;
    }
    for (TermRef a : n.args) {
        if (occurs(meta, a)) return true;
    }
    return false;
}

// ============================================================================
// Normalization
// ============================================================================

bool TermStore::isIndexTerm(TermRef t) const {
    switch (nodes_[t].kind) {
        case TermKind::IntLit:
        case TermKind::Op:
        case TermKind::Free:
            return true;
        default:
            return false;
    }
}

bool TermStore::polynomialOf(TermRef t, Polynomial& out, int depth) {
    if (depth > kMaxDepth) return false;
    out.clear();
    const TermNode n = nodes_[t];
    if (n.kind == TermKind::IntLit) {
        if (n.value != 0) out[{}] = n.value;
        return true;
    }
    if (n.kind == TermKind::Meta) {
        if (auto sol = metaSolution(n.value)) return polynomialOf(*sol, out, depth + 1);
    }
    if (n.kind == TermKind::Free) {
        auto def = free_definitions_.find(static_cast<uint64_t>(n.value));
        if (def != free_definitions_.end()) return polynomialOf(def->second, out, depth + 1);
    }
    if (n.kind == TermKind::Op && isArithmeticHead(n.name) && !n.args.empty()) {
        std::vector<Polynomial> parts(n.args.size());
        for (size_t i = 0; i < n.args.size(); ++i) {
            if (!polynomialOf(n.args[i], parts[i], depth + 1)) return false;
        }
        auto add = [](Polynomial& acc, const Polynomial& p, __int128 sign) {
            for (const auto& [mono, c] : p) {
                acc[mono] += sign * c;
                if (acc[mono] == 0) acc.erase(mono);
            }
        };
        if (n.name == "+") {
            for (const auto& p : parts) add(out, p, 1);
        } else if (n.name == "-") {
            if (parts.size() == 1) {
                add(out, parts[0], -1);
            } else {
                add(out, parts[0], 1);
                for (size_t i = 1; i < parts.size(); ++i) add(out, parts[i], -1);
            }
        } else {  // "*"
            out = parts[0];
            for (size_t i = 1; i < parts.size(); ++i) {
                Polynomial product;
                for (const auto& [ma, ca] : out) {
                    for (const auto& [mb, cb] : parts[i]) {
                        Monomial m = ma;
                        m.insert(m.end(), mb.begin(), mb.end());
                        std::sort(m.begin(), m.end());
                        product[m] += ca * cb;
                        if (product[m] == 0) product.erase(m);
                    }
                }
                for (const auto& [mono, c] : product) {
                    if (!fitsInt64(c)) return false;
                    (void)mono;
                }
                out = std::move(product);
            }
        }
        for (const auto& [mono, c] : out) {
            if (!fitsInt64(c)) return false;
            (void)mono;
        }
        return true;
    }
    // An atom: a rigid variable, an unsolved metavariable, or an uninterpreted
    // stable head such as (car p), with its arguments in normal form.
    TermRef atom = t;
    if (n.kind == TermKind::Op) {
        std::vector<TermRef> args = n.args;
        for (auto& a : args) a = normalize(a);
        atom = rebuild(t, args);
    }
    out[{atom}] = 1;
    return true;
}

TermRef TermStore::termOfPolynomial(const Polynomial& p) {
    std::vector<TermRef> terms;
    TermRef constant_term = kNoTerm;
    for (const auto& [mono, c] : p) {
        const int64_t coeff = static_cast<int64_t>(c);
        if (mono.empty()) {
            constant_term = intLit(coeff);
            continue;
        }
        if (coeff == 1 && mono.size() == 1) {
            terms.push_back(mono[0]);
            continue;
        }
        std::vector<TermRef> factors;
        if (coeff != 1) factors.push_back(intLit(coeff));
        factors.insert(factors.end(), mono.begin(), mono.end());
        terms.push_back(op("*", std::move(factors)));
    }
    if (constant_term != kNoTerm) terms.push_back(constant_term);
    if (terms.empty()) return intLit(0);
    if (terms.size() == 1) return terms[0];
    return op("+", std::move(terms));
}

TermRef TermStore::normalizeIndex(TermRef t) {
    Polynomial p;
    if (polynomialOf(t, p, 0)) return termOfPolynomial(p);
    // Out of range or too deep: keep the structure, arguments normalized.
    const TermNode n = nodes_[t];
    std::vector<TermRef> args = n.args;
    for (auto& a : args) a = normalize(a);
    return rebuild(t, args);
}

TermRef TermStore::normalize(TermRef t) {
    if (t == kNoTerm) return t;
    const TermNode n = nodes_[t];
    switch (n.kind) {
        case TermKind::Universe:
        case TermKind::Const:
        case TermKind::Bound:
        case TermKind::IntLit:
            return t;
        case TermKind::Meta: {
            auto sol = metaSolution(n.value);
            return sol ? normalize(*sol) : t;
        }
        case TermKind::Free: {
            auto def = free_definitions_.find(static_cast<uint64_t>(n.value));
            return def != free_definitions_.end() ? normalize(def->second) : t;
        }
        case TermKind::Op:
            return normalizeIndex(t);
        case TermKind::Pi:
        case TermKind::Sigma:
        case TermKind::App: {
            std::vector<TermRef> args = n.args;
            for (auto& a : args) a = normalize(a);
            return rebuild(t, args);
        }
    }
    return t;
}

// ============================================================================
// Unification and equality
// ============================================================================

TermEq TermStore::unifyIndex(TermRef a, TermRef b, bool solve) {
    Polynomial pa, pb;
    if (!polynomialOf(a, pa, 0) || !polynomialOf(b, pb, 0)) {
        return normalize(a) == normalize(b) ? TermEq::Equal : TermEq::Unknown;
    }
    Polynomial diff = pa;
    for (const auto& [mono, c] : pb) {
        diff[mono] -= c;
        if (diff[mono] == 0) diff.erase(mono);
    }
    if (diff.empty()) return TermEq::Equal;

    // Which monomials mention an unsolved metavariable?
    auto meta_of = [this](TermRef atom) -> int64_t {
        const TermNode& n = nodes_[atom];
        return n.kind == TermKind::Meta ? n.value : 0;
    };
    int64_t solve_for = 0;
    __int128 coeff = 0;
    size_t meta_monomials = 0;
    bool any_meta = false;
    for (const auto& [mono, c] : diff) {
        bool has = false;
        for (TermRef atom : mono) {
            if (meta_of(atom) || hasUnsolvedMetas(atom)) has = true;
        }
        if (has) {
            any_meta = true;
            ++meta_monomials;
            if (mono.size() == 1 && meta_of(mono[0])) {
                solve_for = meta_of(mono[0]);
                coeff = c;
            }
        }
    }
    if (!any_meta) return TermEq::Unequal;
    if (!solve || meta_monomials != 1 || solve_for == 0) return TermEq::Unknown;

    // diff = coeff * ?m + rest = 0  =>  ?m = -rest / coeff, when it divides.
    Polynomial rest;
    for (const auto& [mono, c] : diff) {
        if (mono.size() == 1 && meta_of(mono[0]) == solve_for) continue;
        if (c % coeff != 0) return TermEq::Unknown;
        rest[mono] = -c / coeff;
    }
    const TermRef solution = termOfPolynomial(rest);
    if (occurs(solve_for, solution)) return TermEq::Unknown;
    solveMeta(solve_for, solution);
    return TermEq::Equal;
}

TermEq TermStore::unifyAt(TermRef a, TermRef b, const LeafJudge& leaf, bool solve, int depth) {
    if (depth > kMaxDepth) return TermEq::Unknown;
    const TermRef na = normalize(a);
    const TermRef nb = normalize(b);
    if (na == nb) return TermEq::Equal;
    const TermNode A = nodes_[na];
    const TermNode B = nodes_[nb];

    const bool a_meta = A.kind == TermKind::Meta;
    const bool b_meta = B.kind == TermKind::Meta;
    const bool a_index = A.kind == TermKind::IntLit || A.kind == TermKind::Op;
    const bool b_index = B.kind == TermKind::IntLit || B.kind == TermKind::Op;

    if ((a_index || b_index) &&
        (a_index || a_meta || A.kind == TermKind::Free || A.kind == TermKind::Bound) &&
        (b_index || b_meta || B.kind == TermKind::Free || B.kind == TermKind::Bound)) {
        return unifyIndex(na, nb, solve);
    }
    if (a_meta || b_meta) {
        const int64_t id = a_meta ? A.value : B.value;
        const TermRef other = a_meta ? nb : na;
        if (!solve || occurs(id, other)) return TermEq::Unknown;
        solveMeta(id, other);
        return TermEq::Equal;
    }
    if (A.kind != B.kind) return TermEq::Unequal;
    switch (A.kind) {
        case TermKind::Const:
            return leaf ? leaf(static_cast<uint32_t>(A.value), static_cast<uint32_t>(B.value))
                        : TermEq::Unequal;
        case TermKind::Universe:
        case TermKind::Bound:
        case TermKind::Free:
        case TermKind::IntLit:
            return TermEq::Unequal;  // distinct references of the same kind
        case TermKind::Pi:
            if (A.value != B.value || A.args.size() != B.args.size()) return TermEq::Unequal;
            break;
        case TermKind::App:
            if (A.name != B.name || A.args.size() != B.args.size()) return TermEq::Unequal;
            break;
        case TermKind::Sigma:
            break;
        default:
            return TermEq::Unequal;
    }
    TermEq result = TermEq::Equal;
    for (size_t i = 0; i < A.args.size(); ++i) {
        result = termEqAnd(result, unifyAt(A.args[i], B.args[i], leaf, solve, depth + 1));
        if (result == TermEq::Unequal) return result;
    }
    return result;
}

TermEq TermStore::unify(TermRef a, TermRef b, const LeafJudge& leaf, bool solve) {
    return unifyAt(a, b, leaf, solve, 0);
}

TermEq TermStore::definitionallyEqual(TermRef a, TermRef b) {
    return unify(a, b,
                 [](uint32_t x, uint32_t y) { return x == y ? TermEq::Equal : TermEq::Unequal; },
                 /*solve=*/false);
}

// ============================================================================
// Universes
// ============================================================================

std::optional<uint32_t> TermStore::universeAt(
    TermRef t, const std::function<uint32_t(uint32_t)>& const_level,
    std::vector<std::optional<uint32_t>>& binder_levels) {
    const TermRef nt = normalize(t);
    const TermNode n = nodes_[nt];
    // The universe a binder's variable lives in when the binder ranges over a
    // universe (it is then a type); nullopt when it ranges over values.
    auto binder_level_of = [this](TermRef domain) -> std::optional<uint32_t> {
        const TermNode& d = nodes_[normalize(domain)];
        if (d.kind == TermKind::Universe) return static_cast<uint32_t>(d.value);
        return std::nullopt;
    };
    switch (n.kind) {
        case TermKind::Universe:
            return static_cast<uint32_t>(n.value + 1);
        case TermKind::Const:
            return const_level(static_cast<uint32_t>(n.value));
        case TermKind::Bound: {
            const size_t k = static_cast<size_t>(n.value);
            if (k >= binder_levels.size()) return std::nullopt;
            return binder_levels[binder_levels.size() - 1 - k];
        }
        case TermKind::Pi: {
            uint32_t level = 0;
            const size_t pushed = n.args.size() - 1;
            for (size_t i = 0; i < pushed; ++i) {
                auto u = universeAt(n.args[i], const_level, binder_levels);
                if (!u) {
                    binder_levels.resize(binder_levels.size() - i);
                    return std::nullopt;
                }
                level = std::max(level, *u);
                binder_levels.push_back(binder_level_of(n.args[i]));
            }
            auto body = universeAt(n.args.back(), const_level, binder_levels);
            binder_levels.resize(binder_levels.size() - pushed);
            if (!body) return std::nullopt;
            return std::max(level, *body);
        }
        case TermKind::Sigma: {
            auto first = universeAt(n.args[0], const_level, binder_levels);
            if (!first) return std::nullopt;
            binder_levels.push_back(binder_level_of(n.args[0]));
            auto second = universeAt(n.args[1], const_level, binder_levels);
            binder_levels.pop_back();
            if (!second) return std::nullopt;
            return std::max(*first, *second);
        }
        case TermKind::App: {
            // A family application lives in the universe of its type
            // arguments (formation); index arguments do not raise it.
            uint32_t level = 0;
            for (TermRef a : n.args) {
                if (isIndexTerm(normalize(a))) continue;
                auto u = universeAt(a, const_level, binder_levels);
                if (u) level = std::max(level, *u);
            }
            return level;
        }
        default:
            return std::nullopt;  // index terms and metavariables are not types
    }
}

std::optional<uint32_t> TermStore::universeOf(
    TermRef t, const std::function<uint32_t(uint32_t)>& const_level) {
    std::vector<std::optional<uint32_t>> binder_levels;
    return universeAt(t, const_level, binder_levels);
}

// ============================================================================
// Printing
// ============================================================================

std::string TermStore::printAt(TermRef t, const std::function<std::string(uint32_t)>& const_name,
                               std::vector<std::string>& names) {
    const TermRef nt = normalize(t);
    const TermNode n = nodes_[nt];
    switch (n.kind) {
        case TermKind::Universe:
            return "(universe " + std::to_string(n.value) + ")";
        case TermKind::Const:
            return const_name(static_cast<uint32_t>(n.value));
        case TermKind::Bound: {
            const size_t k = static_cast<size_t>(n.value);
            if (k < names.size()) return names[names.size() - 1 - k];
            return "#" + std::to_string(k);
        }
        case TermKind::Free:
            return n.name.empty() ? "v" + std::to_string(n.value) : n.name;
        case TermKind::Meta:
            return "?" + std::to_string(n.value);
        case TermKind::IntLit:
            return std::to_string(n.value);
        case TermKind::Op: {
            std::string out = "(" + n.name;
            for (TermRef a : n.args) out += " " + printAt(a, const_name, names);
            return out + ")";
        }
        case TermKind::App: {
            std::string out = "(" + n.name;
            for (TermRef a : n.args) out += " " + printAt(a, const_name, names);
            return out + ")";
        }
        case TermKind::Pi: {
            const size_t count = n.args.size() - 1;
            std::string out = (n.value & kPiImplicit) ? "(Pi {" : "(Pi (";
            const size_t before = names.size();
            for (size_t i = 0; i < count; ++i) {
                std::string hint = i < n.hints.size() && !n.hints[i].empty()
                                       ? n.hints[i]
                                       : "x" + std::to_string(before + i);
                if (i > 0) out += " ";
                out += "(" + hint + " : " + printAt(n.args[i], const_name, names) + ")";
                names.push_back(hint);
            }
            if ((n.value & kPiVariadic) && count > 0) out += " ...";
            out += (n.value & kPiImplicit) ? "} " : ") ";
            out += printAt(n.args.back(), const_name, names);
            names.resize(before);
            return out + ")";
        }
        case TermKind::Sigma: {
            std::string hint = !n.hints.empty() && !n.hints[0].empty()
                                   ? n.hints[0]
                                   : "x" + std::to_string(names.size());
            std::string out = "(Sigma ((" + hint + " : " +
                              printAt(n.args[0], const_name, names) + ")) ";
            names.push_back(hint);
            out += printAt(n.args[1], const_name, names);
            names.pop_back();
            return out + ")";
        }
    }
    return "?";
}

std::string TermStore::print(TermRef t, const std::function<std::string(uint32_t)>& const_name) {
    std::vector<std::string> names;
    return printAt(t, const_name, names);
}

}  // namespace eshkol::hott
