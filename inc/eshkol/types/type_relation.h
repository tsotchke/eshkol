/**
 * @file type_relation.h
 * @brief The gradual type relation: one home for every judgment that compares,
 *        combines or prints types.
 *
 * The checker used to carry a rule of its own at each place it compared two
 * types: a numeric exception at the call site, a Value refusal and a Boolean
 * exception in named-let widening, a callable exception in ascription, an
 * outright binding in occurrence typing. TypeRelation replaces all of them
 * with a small set of documented operations on one lattice. The design,
 * the rules and what later work may rely on are recorded in
 * docs/design/adr/0013-gradual-type-relation.md.
 *
 * The lattice. `Value` is the top and `Never` (the empty type) the bottom.
 * Between them sit the nominal graph (the numeric tower, text, booleans, the
 * collection and resource families), tracked pairs `Pair<A, B>` (covariant;
 * a bare `Pair` is `Pair<Value, Value>`), sums `(+ A B ...)`, and function
 * signatures `(-> A... R)` below the generic procedure types `Function` and
 * `Closure`, which name the same runtime closure.
 *
 * The four judgments of ADR-0004 that exist today:
 *  - `A <: B`   static subtyping. `Value` participates only as the top.
 *               Arrows are contravariant in parameters, covariant in results.
 *  - `A ~ B`    Siek-Taha consistency. `Value` is the unknown type `?` and is
 *               consistent with everything, structurally inside arrows and
 *               pairs. Symmetric, not transitive, never cached.
 *  - `A ≲ B`    consistent subtyping, the gradual check: some type consistent
 *               with A is a subtype of B.
 *  - flow       the verdict a check site acts on, returned as evidence: the
 *               coercion typed HIR will make explicit (ADR-0004 step 6).
 *
 * Every operation is pure over the TypeEnvironment's interned types and keys
 * nothing by expression or binding; per-expression facts belong to the
 * semantic substrate (eshkol_typed_expr_info), not to this module.
 */
#ifndef ESHKOL_TYPES_TYPE_RELATION_H
#define ESHKOL_TYPES_TYPE_RELATION_H

#include "eshkol/types/hott_types.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace eshkol::hott {

/**
 * @brief Why a value of one static type may flow where another is expected.
 *
 * Ordered from no runtime work to a rejection. Everything except
 * Incompatible is accepted; the kind is the coercion a typed-HIR elaborator
 * inserts (none, an upcast, a checked cast with a blame label, a numeric
 * conversion).
 */
enum class RelationEvidence : uint8_t {
    Identity,      ///< the same type
    Upcast,        ///< a static subtype: no runtime work
    Dynamic,       ///< consistent only through Value: a checked cast
    Numeric,       ///< both in the numeric tower: an R7RS numeric conversion
    Incompatible,  ///< no value of the source type can be used as the target
};

/** @brief How an inferred slot (a loop variable) absorbs a new incoming type. */
enum class WidenPolicy : uint8_t {
    /**
     * A named-let parameter: its type is inferred, and a recursive argument is
     * also checked against it. An incoming Value adds nothing. A join that
     * reaches the top is a conflict (the slot keeps its type, so the argument
     * is reported), except where either side is Boolean: Scheme's `#f` is the
     * universal "nothing yet", so a flag-or-value slot widens to Value.
     */
    InferenceSlot,
    /** A `do` variable: its step is not an argument check, so every join is adopted. */
    AdoptTop,
};

/** @brief The outcome of TypeRelation::widen(). */
struct WidenResult {
    TypeId type;    ///< the slot's type after absorbing the incoming type
    bool changed;   ///< the type moved up the lattice
    bool conflict;  ///< InferenceSlot only: the join reached the top and was refused
};

/** @brief Which component of a pair an eliminator projects. */
enum class PairSide : uint8_t { Car, Cdr };

/**
 * @brief The gradual type relation over one TypeEnvironment.
 *
 * A lightweight view: it holds a reference to the environment, whose
 * interned signatures, pairs and sums it reads and extends, and whose subtype
 * cache it fills. Construct it wherever it is needed.
 */
class TypeRelation {
public:
    explicit TypeRelation(const TypeEnvironment& env) : env_(env) {}

    // ---- the four judgments ------------------------------------------------

    /** @brief Static subtyping `sub <: super` (cached). */
    bool isSubtype(TypeId sub, TypeId super) const;

    /** @brief Siek-Taha consistency `a ~ b`: equal up to Value, structurally. Not cached. */
    bool isConsistent(TypeId a, TypeId b) const;

    /** @brief Consistent subtyping `sub ≲ super`: the gradual check. Not cached. */
    bool isConsistentSubtype(TypeId sub, TypeId super) const;

    /** @brief The evidence under which a value of type @p from may flow where @p to is expected. */
    RelationEvidence compatibility(TypeId from, TypeId to) const;

    /** @brief compatibility(from, to) is not Incompatible. The verdict of every flow check. */
    bool accepts(TypeId from, TypeId to) const;

    /**
     * @brief `(the ascribed e)` over `e : actual` is not a provable contradiction:
     *        some value could have both types, or one flows to the other.
     */
    bool castable(TypeId actual, TypeId ascribed) const;

    // ---- the lattice -------------------------------------------------------

    /** @brief Least upper bound. Never is the identity, Value absorbs. */
    TypeId join(TypeId a, TypeId b) const;

    /** @brief join() folded over @p types; Value for an empty list. */
    TypeId joinAll(const std::vector<TypeId>& types) const;

    /** @brief Greatest lower bound. Value is the identity, Never absorbs; disjoint types meet at Never. */
    TypeId meet(TypeId a, TypeId b) const;

    /**
     * @brief The type a successful runtime test proves: meet(current, proven),
     *        or @p proven when the meet is empty (the test's success is evidence
     *        that outranks a static type it contradicts).
     */
    TypeId narrow(TypeId current, TypeId proven) const;

    /** @brief Absorb @p incoming into the inferred @p slot under @p policy. */
    WidenResult widen(TypeId slot, TypeId incoming, WidenPolicy policy) const;

    // ---- elimination and presentation ------------------------------------

    /** @brief The type `car`/`cdr` yields: a tracked component, a List tail, else Value. */
    TypeId pairProjection(TypeId pair, PairSide side) const;

    /** @brief The one printing function: "(-> Number Int64)", "Pair<A, B>", "(+ A B)", "Function". */
    std::string print(TypeId type) const;

    /** @brief Stable lower-case name of an evidence kind, for diagnostics and traces. */
    static const char* evidenceName(RelationEvidence evidence);

private:
    const TypeEnvironment& env_;

    bool subtypeUncached(TypeId sub, TypeId super) const;
    bool signatureSubtype(const PiType& sub, const PiType& super, bool consistent) const;
    bool signatureConsistent(const PiType& a, const PiType& b) const;
    bool isDynamic(TypeId t) const;
    bool isCallableTop(TypeId t) const;
    bool isPairLike(TypeId t) const;
    std::pair<TypeId, TypeId> pairComponents(TypeId t) const;
    TypeId canonicalPair(TypeId car, TypeId cdr) const;
    bool isNumeric(TypeId t) const;
};

}  // namespace eshkol::hott

#endif  // ESHKOL_TYPES_TYPE_RELATION_H
