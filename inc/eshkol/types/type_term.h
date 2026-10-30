/**
 * @file type_term.h
 * @brief Interned core terms of the dependent type language (ADR-0042).
 *
 * One term language serves every type-level judgment the checker makes about
 * dependent types: Pi and Sigma telescopes, universes, type-family
 * applications such as the length-indexed vector `(vector real n)`, and the
 * index terms those families are indexed by (integer literals, program
 * variables, arithmetic, and uninterpreted stable projections such as
 * `(car p)` and `(vector-length v)`).
 *
 * Terms are hash-consed: two structurally equal terms are the same TermRef.
 * Binders use de Bruijn indices, so alpha-equivalent terms are also the same
 * TermRef; binder names are kept only as printing hints. Definitional
 * equality is therefore "normalize, then compare references":
 *
 *   - delta: a stable program variable whose value is an index term unfolds to
 *     that value (`(let ((n 3)) ...)` makes `n` and `3` definitionally equal);
 *   - beta for type families happens when a family application is resolved
 *     (the checker instantiates a define-type body with its arguments), so a
 *     term never contains an unexpanded user family;
 *   - index arithmetic is put in polynomial normal form over the integers
 *     (sum of monomials with integer coefficients, monomials ordered), which
 *     decides equality of polynomial index expressions exactly.
 *
 * Metavariables stand for indices a call site has not supplied (the implicit
 * binders of a dependent signature). unify() solves them first-order and
 * solves a linear equation in a single metavariable (`?n + 1 = 4`).
 *
 * The store is owned by TypeEnvironment; the checker reaches terms through
 * TypeEnvironment::termOf() and TypeEnvironment::typeFromTerm(). Extensions
 * (identity types, refinement predicates, effect rows) add a TermKind and its
 * cases here rather than a second term language.
 */
#ifndef ESHKOL_TYPES_TYPE_TERM_H
#define ESHKOL_TYPES_TYPE_TERM_H

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace eshkol::hott {

struct TypeId;

/** @brief Handle of an interned term; 0 is "no term". */
using TermRef = uint32_t;
inline constexpr TermRef kNoTerm = 0;

/** @brief The constructors of the core term language. */
enum class TermKind : uint8_t {
    Universe,  ///< (universe n); value = n
    Const,     ///< a nominal type; value = packed TypeId
    Bound,     ///< a variable bound by an enclosing Pi/Sigma; value = de Bruijn index
    Free,      ///< a stable program variable; value = binding stamp, name = spelling
    Meta,      ///< a metavariable; value = id
    IntLit,    ///< an integer index; value = the integer
    Op,        ///< index operator or uninterpreted stable head; name = head, args
    Pi,        ///< dependent function; args = domains..., body; value = flags
    Sigma,     ///< dependent pair; args = {first, second}; name = binder hint
    App,       ///< type-family application; name = family, args
};

/** @brief Pi flags (TermNode::value). */
inline constexpr int64_t kPiVariadic = 1;  ///< the last domain is a rest argument
inline constexpr int64_t kPiImplicit = 2;  ///< binders are solved at the call site, not passed

/** @brief Three-valued outcome of an equality question. */
enum class TermEq : uint8_t {
    Equal,    ///< definitionally equal (after solving metavariables)
    Unequal,  ///< provably different
    Unknown,  ///< depends on information not available (an unsolved metavariable)
};

/** @brief Combine two outcomes of a conjunction: Unequal dominates, then Unknown. */
inline TermEq termEqAnd(TermEq a, TermEq b) {
    if (a == TermEq::Unequal || b == TermEq::Unequal) return TermEq::Unequal;
    if (a == TermEq::Unknown || b == TermEq::Unknown) return TermEq::Unknown;
    return TermEq::Equal;
}

/** @brief One interned term node. */
struct TermNode {
    TermKind kind = TermKind::IntLit;
    int64_t value = 0;
    std::string name;
    std::vector<TermRef> args;
    std::vector<std::string> hints;  ///< binder names of a Pi telescope (printing only)
};

/**
 * @brief The interned term store: construction, binders, normalization,
 *        unification and printing.
 */
class TermStore {
public:
    TermStore();

    // ---- construction (hash-consed) ------------------------------------
    TermRef universe(uint32_t level);
    TermRef constant(uint32_t packed_type_id);
    TermRef bound(uint32_t index);
    TermRef freeVar(uint64_t stamp, const std::string& display);
    TermRef freshMeta();
    TermRef intLit(int64_t value);
    TermRef op(const std::string& head, std::vector<TermRef> args);
    /** @brief A Pi telescope: domains[i] lives under i binders, body under domains.size(). */
    TermRef pi(std::vector<TermRef> domains, TermRef body, int64_t flags,
               std::vector<std::string> hints = {});
    TermRef sigma(TermRef first, TermRef second, const std::string& hint = "");
    TermRef app(const std::string& family, std::vector<TermRef> args);

    const TermNode& node(TermRef t) const { return nodes_[t]; }
    TermKind kind(TermRef t) const { return nodes_[t].kind; }

    /** @brief A fresh binding stamp for a program variable. */
    uint64_t freshStamp() { return next_stamp_++; }

    // ---- binders --------------------------------------------------------
    /** @brief Add @p delta to every Bound index >= @p cutoff. */
    TermRef shift(TermRef t, int64_t delta, uint32_t cutoff = 0);
    /** @brief Replace Bound(@p index) (at the top level) with @p value, lowering later indices. */
    TermRef substitute(TermRef body, uint32_t index, TermRef value);
    /**
     * @brief Instantiate a telescope body with @p values: values[0] for the
     *        outermost binder. The body lives under values.size() binders.
     */
    TermRef instantiate(TermRef body, const std::vector<TermRef>& values);
    /**
     * @brief Close @p t over the program variables @p stamps: stamps[0]
     *        becomes the outermost binder, stamps.back() the innermost. @p t
     *        must have no loose Bound index (it is built from Free variables).
     */
    TermRef abstractFrees(TermRef t, const std::vector<uint64_t>& stamps, uint32_t depth = 0);
    /** @brief True if Bound(@p index) occurs (relative to the top). */
    bool mentionsBound(TermRef t, uint32_t index) const;
    /** @brief True if an unsolved metavariable occurs. */
    bool hasUnsolvedMetas(TermRef t);
    /** @brief True if any Bound index escapes (the term is not closed under its binders). */
    bool hasLooseBound(TermRef t, uint32_t depth = 0) const;

    // ---- definitions and metavariables ----------------------------------
    /** @brief Record that the stable variable @p stamp is defined as the index term @p value (delta). */
    void defineFree(uint64_t stamp, TermRef value);
    /** @brief The solution of metavariable @p id, if solved. */
    std::optional<TermRef> metaSolution(int64_t id) const;
    /** @brief Solve metavariable @p id (no occurs check: callers use unify()). */
    void solveMeta(int64_t id, TermRef value);

    // ---- normalization and equality -------------------------------------
    /** @brief Normal form: solved metas and defined variables unfolded, index arithmetic canonical. */
    TermRef normalize(TermRef t);
    /** @brief True if @p t is an index term (not a type former). */
    bool isIndexTerm(TermRef t) const;

    /**
     * @brief Judges two nominal constants (Const leaves); returns Equal, Unequal or Unknown.
     */
    using LeafJudge = std::function<TermEq(uint32_t packed_a, uint32_t packed_b)>;

    /**
     * @brief Unify @p a and @p b: definitional equality that may solve
     *        metavariables when @p solve is true. Const leaves are judged by @p leaf.
     */
    TermEq unify(TermRef a, TermRef b, const LeafJudge& leaf, bool solve = true);

    /** @brief Definitional equality of two meta-free terms with nominal equality at the leaves. */
    TermEq definitionallyEqual(TermRef a, TermRef b);

    /**
     * @brief The universe level @p t lives in, given the level of each nominal
     *        constant. nullopt for an index term or an ill-formed type.
     */
    std::optional<uint32_t> universeOf(TermRef t,
                                       const std::function<uint32_t(uint32_t packed)>& const_level);

    // ---- presentation ----------------------------------------------------
    /** @brief Print @p t; @p const_name spells a nominal constant. */
    std::string print(TermRef t, const std::function<std::string(uint32_t packed)>& const_name);

private:
    struct Key {
        TermKind kind;
        int64_t value;
        std::string name;
        std::vector<TermRef> args;
        bool operator==(const Key& o) const {
            return kind == o.kind && value == o.value && name == o.name && args == o.args;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const;
    };

    std::vector<TermNode> nodes_;
    std::unordered_map<Key, TermRef, KeyHash> interned_;
    std::map<int64_t, TermRef> meta_solutions_;
    std::map<uint64_t, TermRef> free_definitions_;
    int64_t next_meta_ = 1;
    uint64_t next_stamp_ = 1;

    TermRef intern(TermNode node, bool binder_name_in_key = false);
    TermRef rebuild(TermRef t, const std::vector<TermRef>& args);

    // Polynomial normal form of index arithmetic.
    using Monomial = std::vector<TermRef>;           // sorted atoms (with multiplicity)
    using Polynomial = std::map<Monomial, __int128>;  // monomial -> coefficient
    bool polynomialOf(TermRef t, Polynomial& out, int depth);
    TermRef termOfPolynomial(const Polynomial& p);
    TermRef normalizeIndex(TermRef t);
    TermEq unifyIndex(TermRef a, TermRef b, bool solve);
    bool occurs(int64_t meta, TermRef t);
    TermEq unifyAt(TermRef a, TermRef b, const LeafJudge& leaf, bool solve, int depth);
    std::optional<uint32_t> universeAt(TermRef t,
                                       const std::function<uint32_t(uint32_t)>& const_level,
                                       std::vector<std::optional<uint32_t>>& binder_levels);
    std::string printAt(TermRef t, const std::function<std::string(uint32_t)>& const_name,
                        std::vector<std::string>& names);
};

}  // namespace eshkol::hott

#endif  // ESHKOL_TYPES_TYPE_TERM_H
