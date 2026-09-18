/**
 * @file type_relation.cpp
 * @brief The gradual type relation. Rules are in type_relation.h and
 *        docs/design/adr/0013-gradual-type-relation.md.
 */
#include "eshkol/types/type_relation.h"

#include <utility>

namespace eshkol::hott {

// ============================================================================
// Classification helpers
// ============================================================================

/** @brief Value (the unknown/top type) or an unresolved type: carries no static information. */
bool TypeRelation::isDynamic(TypeId t) const {
    return t == BuiltinTypes::Value || t == BuiltinTypes::Invalid;
}

/** @brief The generic procedure types. Function and Closure name the same runtime closure. */
bool TypeRelation::isCallableTop(TypeId t) const {
    return t == BuiltinTypes::Function || t == BuiltinTypes::Closure;
}

/** @brief A tracked Pair<A, B> or the bare Pair constructor. */
bool TypeRelation::isPairLike(TypeId t) const {
    return t == BuiltinTypes::Pair || env_.isTrackedPairType(t);
}

/** @brief The components of a pair-like type; a bare Pair is Pair<Value, Value>. */
std::pair<TypeId, TypeId> TypeRelation::pairComponents(TypeId t) const {
    if (auto elems = env_.getPairElementTypes(t)) return *elems;
    return {BuiltinTypes::Value, BuiltinTypes::Value};
}

/** @brief Pair<car, cdr>, spelled as the bare Pair when both components are Value. */
TypeId TypeRelation::canonicalPair(TypeId car, TypeId cdr) const {
    if (car == BuiltinTypes::Value && cdr == BuiltinTypes::Value) return BuiltinTypes::Pair;
    return env_.makePairType(car, cdr);
}

/** @brief A member of the numeric tower (not the empty type). */
bool TypeRelation::isNumeric(TypeId t) const {
    return t != BuiltinTypes::Never && !isDynamic(t) && isSubtype(t, BuiltinTypes::Number);
}

// ============================================================================
// Static subtyping  A <: B
// ============================================================================

bool TypeRelation::isSubtype(TypeId sub, TypeId super) const {
    const auto key = std::make_pair(sub.id, super.id);
    auto it = env_.subtype_cache_.find(key);
    if (it != env_.subtype_cache_.end()) return it->second;
    const bool result = subtypeUncached(sub, super);
    env_.subtype_cache_[key] = result;
    return result;
}

/**
 * @brief Arrow subtyping: @p sub accepts every argument @p super accepts
 *        (parameters contravariant) and returns only what @p super may return
 *        (result covariant).
 *
 * The parameter lists are the fixed parameters. A variadic signature accepts
 * any number of further arguments, so a variadic @p sub covers a @p super with
 * at least as many parameters, and a fixed @p sub never stands in for a
 * variadic @p super. With @p consistent, components use consistent subtyping.
 */
bool TypeRelation::signatureSubtype(const PiType& sub, const PiType& super, bool consistent) const {
    const auto component = [this, consistent](TypeId a, TypeId b) {
        return consistent ? isConsistentSubtype(a, b) : isSubtype(a, b);
    };
    const size_t sub_n = sub.params.size();
    const size_t super_n = super.params.size();
    if (sub.is_variadic) {
        if (sub_n > super_n) return false;
    } else if (super.is_variadic || sub_n != super_n) {
        return false;
    }
    for (size_t i = 0; i < sub_n; ++i) {
        if (!component(super.params[i].type, sub.params[i].type)) return false;
    }
    return component(sub.return_type, super.return_type);
}

bool TypeRelation::subtypeUncached(TypeId sub, TypeId super) const {
    if (sub == super) return true;
    if (sub == BuiltinTypes::Invalid || super == BuiltinTypes::Invalid) return false;
    if (sub == BuiltinTypes::Never) return true;       // bottom
    if (super == BuiltinTypes::Value) return true;     // top
    if (super == BuiltinTypes::Never) return false;
    if (sub == BuiltinTypes::Value) return false;

    // Sums: sub <: (+ A B) when sub fits an arm; (+ A B) <: super when every arm does.
    const auto super_members = env_.getSumMembers(super);
    if (super_members) {
        for (const auto& arm : *super_members) {
            if (isSubtype(sub, arm)) return true;
        }
    }
    if (const auto sub_members = env_.getSumMembers(sub)) {
        for (const auto& arm : *sub_members) {
            if (!isSubtype(arm, super)) return false;
        }
        return true;
    }
    if (super_members) return false;

    // Procedures. Every signature is a Function and a Closure; those two name
    // the same runtime closure and are subtypes of each other.
    const PiType* sub_pi = env_.getFunctionType(sub);
    const PiType* super_pi = env_.getFunctionType(super);
    if ((sub_pi || isCallableTop(sub)) && isCallableTop(super)) return true;
    if (sub_pi && super_pi) return signatureSubtype(*sub_pi, *super_pi, /*consistent=*/false);
    if (sub_pi || super_pi) return false;

    // Pairs, covariant. A bare Pair is Pair<Value, Value>.
    if (isPairLike(sub) && isPairLike(super)) {
        const auto [sa, sd] = pairComponents(sub);
        const auto [pa, pd] = pairComponents(super);
        return isSubtype(sa, pa) && isSubtype(sd, pd);
    }
    if (env_.isTrackedPairType(sub)) return isSubtype(BuiltinTypes::Pair, super);

    // The nominal graph.
    const TypeNode* node = env_.getTypeNode(sub);
    while (node && node->supertype.has_value()) {
        if (node->supertype.value() == super) return true;
        node = env_.getTypeNode(node->supertype.value());
    }
    return false;
}

// ============================================================================
// Consistency  A ~ B  and consistent subtyping  A ≲ B
// ============================================================================

bool TypeRelation::signatureConsistent(const PiType& a, const PiType& b) const {
    if (a.is_variadic != b.is_variadic || a.params.size() != b.params.size()) return false;
    for (size_t i = 0; i < a.params.size(); ++i) {
        if (!isConsistent(a.params[i].type, b.params[i].type)) return false;
    }
    return isConsistent(a.return_type, b.return_type);
}

bool TypeRelation::isConsistent(TypeId a, TypeId b) const {
    if (a == b) return true;
    if (isDynamic(a) || isDynamic(b)) return true;
    if (const auto arms = env_.getSumMembers(a)) {
        for (const auto& arm : *arms) {
            if (isConsistent(arm, b)) return true;
        }
        return false;
    }
    if (const auto arms = env_.getSumMembers(b)) {
        for (const auto& arm : *arms) {
            if (isConsistent(a, arm)) return true;
        }
        return false;
    }
    const PiType* pa = env_.getFunctionType(a);
    const PiType* pb = env_.getFunctionType(b);
    if (pa && pb) return signatureConsistent(*pa, *pb);
    // The generic procedure type is an arrow whose signature is unknown.
    if ((pa && isCallableTop(b)) || (pb && isCallableTop(a))) return true;
    if (isCallableTop(a) && isCallableTop(b)) return true;
    if (isPairLike(a) && isPairLike(b)) {
        const auto [aa, ad] = pairComponents(a);
        const auto [ba, bd] = pairComponents(b);
        return isConsistent(aa, ba) && isConsistent(ad, bd);
    }
    return false;
}

bool TypeRelation::isConsistentSubtype(TypeId sub, TypeId super) const {
    if (sub == super) return true;
    if (isDynamic(sub) || isDynamic(super)) return true;
    if (sub == BuiltinTypes::Never) return true;

    const auto super_members = env_.getSumMembers(super);
    if (super_members) {
        for (const auto& arm : *super_members) {
            if (isConsistentSubtype(sub, arm)) return true;
        }
    }
    if (const auto sub_members = env_.getSumMembers(sub)) {
        for (const auto& arm : *sub_members) {
            if (!isConsistentSubtype(arm, super)) return false;
        }
        return true;
    }
    if (super_members) return false;

    const PiType* sub_pi = env_.getFunctionType(sub);
    const PiType* super_pi = env_.getFunctionType(super);
    if (sub_pi && super_pi) return signatureSubtype(*sub_pi, *super_pi, /*consistent=*/true);
    if ((sub_pi || isCallableTop(sub)) && (super_pi || isCallableTop(super))) return true;

    if (isPairLike(sub) && isPairLike(super)) {
        const auto [sa, sd] = pairComponents(sub);
        const auto [pa, pd] = pairComponents(super);
        return isConsistentSubtype(sa, pa) && isConsistentSubtype(sd, pd);
    }
    return isSubtype(sub, super);
}

// ============================================================================
// Flow and ascription
// ============================================================================

RelationEvidence TypeRelation::compatibility(TypeId from, TypeId to) const {
    if (from == to) return RelationEvidence::Identity;
    if (isSubtype(from, to)) return RelationEvidence::Upcast;
    if (isConsistentSubtype(from, to)) return RelationEvidence::Dynamic;
    // R7RS numbers convert: exactness and representation are runtime
    // properties the tower promotes, so a numeric value flows to any numeric
    // slot. This is a conversion, not a subtype: Int64 is not a Float64.
    if (isNumeric(from) && isNumeric(to)) return RelationEvidence::Numeric;
    return RelationEvidence::Incompatible;
}

bool TypeRelation::accepts(TypeId from, TypeId to) const {
    return compatibility(from, to) != RelationEvidence::Incompatible;
}

bool TypeRelation::castable(TypeId actual, TypeId ascribed) const {
    if (isDynamic(actual) || isDynamic(ascribed)) return true;
    if (accepts(actual, ascribed) || accepts(ascribed, actual)) return true;
    return meet(actual, ascribed) != BuiltinTypes::Never;
}

// ============================================================================
// Join and meet
// ============================================================================

TypeId TypeRelation::join(TypeId a, TypeId b) const {
    if (a == b) return a;
    if (a == BuiltinTypes::Never) return b;
    if (b == BuiltinTypes::Never) return a;
    if (isDynamic(a) || isDynamic(b)) return BuiltinTypes::Value;
    if (isSubtype(a, b)) return b;
    if (isSubtype(b, a)) return a;

    // A sum absorbs a type that fits one of its arms.
    if (const auto arms = env_.getSumMembers(a)) {
        for (const auto& arm : *arms) {
            if (isSubtype(b, arm)) return a;
        }
    }
    if (const auto arms = env_.getSumMembers(b)) {
        for (const auto& arm : *arms) {
            if (isSubtype(a, arm)) return b;
        }
    }

    // Two signatures over the same parameters join to that signature over the
    // join of their results; any other two procedures meet at Function. The
    // PiTypes are copied: makeFunctionType() may add to the cache they live in.
    const PiType* pa = env_.getFunctionType(a);
    const PiType* pb = env_.getFunctionType(b);
    if (pa && pb) {
        const PiType sa = *pa;
        const PiType sb = *pb;
        bool same = sa.is_variadic == sb.is_variadic && sa.params.size() == sb.params.size();
        for (size_t i = 0; same && i < sa.params.size(); ++i) {
            same = sa.params[i].type == sb.params[i].type;
        }
        if (same) {
            std::vector<TypeId> params;
            params.reserve(sa.params.size());
            for (const auto& p : sa.params) params.push_back(p.type);
            return env_.makeFunctionType(params, join(sa.return_type, sb.return_type),
                                         sa.is_variadic);
        }
    }
    if ((pa || isCallableTop(a)) && (pb || isCallableTop(b))) return BuiltinTypes::Function;

    if (isPairLike(a) && isPairLike(b)) {
        const auto [aa, ad] = pairComponents(a);
        const auto [ba, bd] = pairComponents(b);
        return canonicalPair(join(aa, ba), join(ad, bd));
    }

    // The nominal graph: the most specific shared supertype, else the top.
    // Walk it here instead of calling TypeEnvironment::leastCommonSupertype():
    // that compatibility facade delegates back to this relation's join().
    const auto chain_a = env_.getSupertypeChain(a);
    const auto chain_b = env_.getSupertypeChain(b);
    for (const auto& candidate : chain_a) {
        for (const auto& other : chain_b) {
            if (candidate == other) return candidate;
        }
    }
    return BuiltinTypes::Value;
}

TypeId TypeRelation::joinAll(const std::vector<TypeId>& types) const {
    if (types.empty()) return BuiltinTypes::Value;
    TypeId acc = types.front();
    for (size_t i = 1; i < types.size() && acc != BuiltinTypes::Value; ++i) {
        acc = join(acc, types[i]);
    }
    return acc;
}

TypeId TypeRelation::meet(TypeId a, TypeId b) const {
    if (a == b) return a;
    if (a == BuiltinTypes::Never || b == BuiltinTypes::Never) return BuiltinTypes::Never;
    if (isDynamic(a)) return b;
    if (isDynamic(b)) return a;
    if (isSubtype(a, b)) return a;
    if (isSubtype(b, a)) return b;

    // A sum meets arm by arm.
    const auto distribute = [this](const std::vector<TypeId>& arms, TypeId other) {
        std::vector<TypeId> kept;
        for (const auto& arm : arms) {
            const TypeId m = meet(arm, other);
            if (m != BuiltinTypes::Never) kept.push_back(m);
        }
        if (kept.empty()) return BuiltinTypes::Never;
        if (kept.size() == 1) return kept.front();
        return env_.makeSumType(kept);
    };
    if (const auto arms = env_.getSumMembers(a)) return distribute(*arms, b);
    if (const auto arms = env_.getSumMembers(b)) return distribute(*arms, a);

    // Two signatures of one arity meet at the arrow over the joined parameters
    // and the met result. An arrow is inhabited even when its result is Never
    // (a function that does not return), so only an arity mismatch is empty.
    const PiType* pa = env_.getFunctionType(a);
    const PiType* pb = env_.getFunctionType(b);
    if (pa && pb) {
        const PiType sa = *pa;
        const PiType sb = *pb;
        if (sa.is_variadic != sb.is_variadic || sa.params.size() != sb.params.size()) {
            return BuiltinTypes::Never;
        }
        std::vector<TypeId> params;
        params.reserve(sa.params.size());
        for (size_t i = 0; i < sa.params.size(); ++i) {
            params.push_back(join(sa.params[i].type, sb.params[i].type));
        }
        return env_.makeFunctionType(params, meet(sa.return_type, sb.return_type),
                                     sa.is_variadic);
    }

    if (isPairLike(a) && isPairLike(b)) {
        const auto [aa, ad] = pairComponents(a);
        const auto [ba, bd] = pairComponents(b);
        const TypeId car = meet(aa, ba);
        const TypeId cdr = meet(ad, bd);
        if (car == BuiltinTypes::Never || cdr == BuiltinTypes::Never) return BuiltinTypes::Never;
        return canonicalPair(car, cdr);
    }

    // Distinct nominal types with no subtyping between them share no value.
    return BuiltinTypes::Never;
}

TypeId TypeRelation::narrow(TypeId current, TypeId proven) const {
    if (!proven.isValid()) return current;
    const TypeId m = meet(current, proven);
    return m == BuiltinTypes::Never ? proven : m;
}

WidenResult TypeRelation::widen(TypeId slot, TypeId incoming, WidenPolicy policy) const {
    const WidenResult unchanged{slot, false, false};
    if (incoming == slot) return unchanged;
    if (policy == WidenPolicy::InferenceSlot && isDynamic(incoming)) return unchanged;
    if (isSubtype(incoming, slot)) return unchanged;
    const TypeId next = join(slot, incoming);
    if (next == slot) return unchanged;
    if (policy == WidenPolicy::InferenceSlot && next == BuiltinTypes::Value &&
        slot != BuiltinTypes::Boolean && incoming != BuiltinTypes::Boolean) {
        return WidenResult{slot, false, true};
    }
    return WidenResult{next, true, false};
}

// ============================================================================
// Elimination and presentation
// ============================================================================

TypeId TypeRelation::pairProjection(TypeId pair, PairSide side) const {
    if (auto elems = env_.getPairElementTypes(pair)) {
        return side == PairSide::Car ? elems->first : elems->second;
    }
    // The tail of a proper list is a list. Nothing else proves more: the tail
    // of a dotted pair is whatever was consed there.
    if (side == PairSide::Cdr && pair == BuiltinTypes::List) return BuiltinTypes::List;
    return BuiltinTypes::Value;
}

std::string TypeRelation::print(TypeId type) const {
    if (const auto arms = env_.getSumMembers(type)) {
        std::string out = "(+";
        for (const auto& arm : *arms) out += " " + print(arm);
        return out + ")";
    }
    if (auto elems = env_.getPairElementTypes(type)) {
        if (elems->first == BuiltinTypes::Value && elems->second == BuiltinTypes::Value) {
            return "Pair";
        }
        return "Pair<" + print(elems->first) + ", " + print(elems->second) + ">";
    }
    if (const PiType* pi = env_.getFunctionType(type)) {
        std::string out = "(->";
        for (const auto& param : pi->params) out += " " + print(param.type);
        if (pi->is_variadic) out += " ...";
        return out + " " + print(pi->return_type) + ")";
    }
    // Registered under the arrow constructor's spelling "->", which reads as a
    // stray token in a sentence.
    if (type == BuiltinTypes::Function) return "Function";
    const TypeNode* node = env_.getTypeNode(type);
    return node ? node->name : "unknown";
}

const char* TypeRelation::evidenceName(RelationEvidence evidence) {
    switch (evidence) {
        case RelationEvidence::Identity: return "identity";
        case RelationEvidence::Upcast: return "upcast";
        case RelationEvidence::Dynamic: return "dynamic";
        case RelationEvidence::Numeric: return "numeric";
        case RelationEvidence::Incompatible: return "incompatible";
    }
    return "incompatible";
}

}  // namespace eshkol::hott
