/**
 * @file structural_types.h
 * @brief Structural type terms of the Eshkol type system (ADR-0046).
 *
 * ADR-0004 calls for "hash-consed core terms with metavariables" beneath the
 * packed TypeId layer. This header is that term language for the structural
 * fragment: polymorphic types with metavariables and rigid variables,
 * extensible records over kinded rows, arbitrary-rank `forall`, binary
 * session protocols and their channel endpoints, one-shot (linear) arrows and
 * multiple values. One solver serves all of them:
 *
 *  - unification with occurs check, levels-based generalisation and
 *    level-based skolem escape (Kiselyov; Peyton Jones et al. 2007);
 *  - row unification by row rewriting (Remy; Leijen 2005) where every row
 *    variable carries the set of labels it must lack (Gaster and Jones 1996),
 *    which keeps rows duplicate-free and inference principal;
 *  - equi-recursive session protocols unified coinductively, with duality as
 *    an involution that also reaches unsolved protocol variables;
 *  - predicativity: a metavariable is only ever solved with a monotype.
 *
 * The same rows are the representation for effect rows (ADR-0045): an effect
 * row is a RowExtend chain whose labels are effect names and whose fields are
 * the effects' payload types; rowInclusion() is "fewer effects <: more".
 *
 * Nothing here knows about the AST; the checker in structural_checker.h walks
 * programs and calls into this layer.
 */
#ifndef ESHKOL_TYPES_STRUCTURAL_TYPES_H
#define ESHKOL_TYPES_STRUCTURAL_TYPES_H

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace eshkol::hott::st {

/** Kind of a structural type term. */
enum class Kind : uint8_t {
    Type,     ///< the type of a value
    Row,      ///< a row of labelled fields (records, choices, effect rows)
    Session   ///< a communication protocol
};

/** Usage mode of an arrow: how often the function value may be called. */
enum class Mode : uint8_t {
    Unrestricted,  ///< `->`: duplicable and droppable
    OneShot        ///< `-o`: linear; called (or handed on) exactly once
};

/** Node tag. */
enum class Tag : uint8_t {
    Meta,       ///< unification variable (any kind)
    Skolem,     ///< rigid variable introduced by skolemisation
    Bound,      ///< variable bound by an enclosing Forall
    Dyn,        ///< the gradual unknown (`any`)
    Con,        ///< nominal constructor applied to arguments
    Arrow,      ///< function: params, optional rest element type, result
    Forall,     ///< polymorphic type `(forall (binders) body)`
    Record,     ///< record over a row
    RowEmpty,   ///< closed end of a row
    RowExtend,  ///< `label : field` in front of a row
    Chan,       ///< channel endpoint carrying a protocol
    SEnd,       ///< protocol `end`
    SSend,      ///< protocol `(! payload continuation)`
    SRecv,      ///< protocol `(? payload continuation)`
    SChoose,    ///< internal choice over a row of protocols
    SOffer,     ///< external choice over a row of protocols
    SMu,        ///< recursive protocol `(rec x body)`
    SRecVar,    ///< occurrence of the recursion variable of an SMu
    SDual,      ///< dual of a protocol (or of a row of protocols) not yet known
    Values      ///< multiple values `(values a b ...)`
};

/** A quantified variable of a Forall (and the template for instantiation). */
struct Binder {
    int id = 0;
    std::string name;
    Kind kind = Kind::Type;
    Kind field_kind = Kind::Type;     ///< for Row binders: kind of the fields
    std::set<std::string> lacks;      ///< for Row binders: labels the row must lack
    bool unrestricted = false;        ///< may not be instantiated with a linear type
};

struct Type;
using TyRef = Type*;

/**
 * One term node. Fields are interpreted per tag (see each constructor in
 * TypeStore); unused fields keep their defaults.
 */
struct Type {
    Tag tag = Tag::Dyn;
    Kind kind = Kind::Type;
    Kind field_kind = Kind::Type;     ///< Row-kinded Meta/Skolem/RowEmpty: kind of fields
    int id = 0;                       ///< Meta/Skolem/Bound/SMu/SRecVar identity
    int level = 0;                    ///< Meta/Skolem level
    std::string name;                 ///< Con name, RowExtend label, variable display name
    TyRef solution = nullptr;         ///< Meta: its solution once solved
    std::set<std::string> lacks;      ///< Row Meta/Skolem: labels it must lack
    bool unrestricted = false;        ///< Meta: may not be solved with a linear type
    Mode mode = Mode::Unrestricted;   ///< Arrow
    std::vector<TyRef> args;          ///< Con args, Arrow params, Values elements
    TyRef a = nullptr;                ///< Arrow result, RowExtend field, SSend/SRecv payload,
                                      ///< Record/SChoose/SOffer row, Chan protocol,
                                      ///< SMu/Forall body, SDual operand
    TyRef b = nullptr;                ///< RowExtend rest, SSend/SRecv continuation,
                                      ///< Arrow rest-parameter element type (nullable)
    std::vector<Binder> binders;      ///< Forall binders
};

/**
 * Owner of every term node. Nodes are never freed individually; the store
 * lives as long as the checker that uses it.
 */
class TypeStore {
public:
    TypeStore();

    // ---- constructors -------------------------------------------------
    TyRef meta(Kind kind, int level, Kind field_kind = Kind::Type);
    TyRef skolem(const std::string& name, Kind kind, int level,
                 Kind field_kind = Kind::Type,
                 const std::set<std::string>& lacks = {});
    TyRef bound(const Binder& binder);
    TyRef dyn() { return dyn_; }
    TyRef con(const std::string& name, std::vector<TyRef> args = {});
    TyRef arrow(std::vector<TyRef> params, TyRef result,
                Mode mode = Mode::Unrestricted, TyRef rest = nullptr);
    TyRef forall(std::vector<Binder> binders, TyRef body);
    TyRef record(TyRef row);
    TyRef rowEmpty(Kind field_kind = Kind::Type);
    TyRef rowExtend(const std::string& label, TyRef field, TyRef rest);
    TyRef chan(TyRef protocol);
    TyRef end() { return end_; }
    TyRef send(TyRef payload, TyRef cont);
    TyRef recv(TyRef payload, TyRef cont);
    TyRef choose(TyRef row);
    TyRef offer(TyRef row);
    TyRef mu(int id, const std::string& name, TyRef body);
    TyRef recVar(int id, const std::string& name);
    TyRef sdual(TyRef operand);
    TyRef values(std::vector<TyRef> elems);

    /** Fresh identity for binders and recursion variables. */
    int freshId() { return ++next_id_; }

    // ---- common base types ------------------------------------------
    TyRef number() { return number_; }
    TyRef string() { return string_; }
    TyRef boolean() { return boolean_; }
    TyRef symbol() { return symbol_; }
    TyRef character() { return char_; }
    TyRef null() { return null_; }

    // ---- structure ----------------------------------------------------
    /** Follow solved metavariables (with path compression). */
    static TyRef prune(TyRef t);
    /** Fully substitute solved metavariables (fresh nodes where changed). */
    TyRef zonk(TyRef t);
    /** Replace Bound variables by the terms in @p subst (keyed by binder id). */
    TyRef substBound(TyRef t, const std::map<int, TyRef>& subst);
    /** One unfolding of a recursive protocol: body[x := the SMu itself]. */
    TyRef unfold(TyRef mu);
    /** Dual of a protocol, or of a row of protocols. */
    TyRef dual(TyRef t);
    /** The tail of a row (the first non-RowExtend term after pruning). */
    static TyRef rowTail(TyRef row);
    /** Labels of a row's known prefix, in order. */
    static std::vector<std::pair<std::string, TyRef>> rowFields(TyRef row);
    /** True if a value of this (zonked) type must be used exactly once. */
    static bool isLinear(TyRef t);
    /** True if the term mentions a Forall anywhere (not a monotype). */
    static bool hasForall(TyRef t);
    /** Unsolved metavariables of @p t in first-occurrence order. */
    static void freeMetas(TyRef t, std::vector<TyRef>& out);

    // ---- polymorphism (Peyton Jones, Vytiniotis, Weirich, Shields 2007) --
    /**
     * Deep instantiation: replace every outer quantifier, and the quantifiers
     * of a function's result, by fresh metavariables at @p level.
     */
    TyRef instantiate(TyRef sigma, int level);
    /**
     * Deep skolemisation (`pr` in the paper): replace every outer quantifier,
     * and the quantifiers of a function's result, by rigid variables at
     * @p level. The rigid variables created are appended to @p skolems.
     */
    TyRef skolemise(TyRef sigma, int level, std::vector<TyRef>& skolems);
    /**
     * Quantify the unsolved metavariables of @p t whose level exceeds
     * @p level. Value-kinded binders are unrestricted (ADR-0046): a
     * generalised variable never ranges over linear types.
     */
    TyRef generalize(TyRef t, int level);
    /** Replace the given metavariables by terms (non-destructive). */
    TyRef replaceMetas(TyRef t, const std::map<TyRef, TyRef>& subst);

    // ---- printing -----------------------------------------------------
    /** Render in Eshkol type syntax; unsolved metas print as `?N`. */
    std::string show(TyRef t);

private:
    TyRef make(Tag tag, Kind kind = Kind::Type);
    std::deque<Type> nodes_;
    int next_id_ = 0;
    TyRef dyn_, end_, number_, string_, boolean_, symbol_, char_, null_;
};

/**
 * The one solver. Errors are reported through error(); after a failed call
 * the partial solution is kept (diagnostics name the first conflict).
 */
class Solver {
public:
    explicit Solver(TypeStore& store) : s_(store) {}

    /** Make @p a and @p b equal. */
    bool unify(TyRef a, TyRef b);

    /**
     * Row inclusion for effect rows: every label of @p sub occurs in @p sup
     * with a unifiable field; an open tail of @p sub absorbs the rest of
     * @p sup. A closed @p sub with extra labels in @p sup is included.
     */
    bool rowInclusion(TyRef sub, TyRef sup);

    /**
     * Rewrite @p row so @p label is at its head: returns {field, rest}.
     * Solves a row metavariable tail when the label is not yet present.
     */
    bool rewriteRow(TyRef row, const std::string& label, TyRef& field, TyRef& rest);

    /** Bind an unsolved metavariable, with every side condition checked. */
    bool bindMeta(TyRef meta, TyRef t);

    /** Human-readable reason for the last failure. */
    const std::string& error() const { return error_; }
    void clearError() { error_.clear(); }

    /**
     * Gradual consistency: a type that meets Dyn. The checker installs a
     * sink so a linear type meeting Dyn can be rejected once types are known.
     */
    std::vector<TyRef>* dyn_meets = nullptr;

private:
    bool fail(const std::string& msg);
    bool unifyRows(TyRef a, TyRef b);
    bool unifyDual(TyRef dual_node, TyRef other);
    bool occursAndAdjust(TyRef meta, TyRef t);

    TypeStore& s_;
    std::string error_;
    std::set<std::pair<TyRef, TyRef>> assumed_;   // coinductive protocol pairs
    int depth_ = 0;
};

}  // namespace eshkol::hott::st

#endif  // ESHKOL_TYPES_STRUCTURAL_TYPES_H
