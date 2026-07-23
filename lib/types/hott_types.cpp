/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HoTT Type System Implementation
 */

#include <eshkol/types/hott_types.h>
#include <eshkol/eshkol.h>
#include <algorithm>
#include <functional>
#include <stdexcept>

namespace eshkol::hott {

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * @brief Convert a Universe level to a human-readable label.
 * @param u Universe level to describe.
 * @return A static UTF-8 string such as "Type₀"/"Type₁"/"Type₂"/"Typeω", or "Type?" for an unrecognized value.
 */
const char* universeToString(Universe u) {
    switch (u) {
        case Universe::U0: return "Type\342\202\200";  // Type₀ (UTF-8 subscript 0)
        case Universe::U1: return "Type\342\202\201";  // Type₁
        case Universe::U2: return "Type\342\202\202";  // Type₂
        case Universe::UOmega: return "Type\317\211"; // Typeω
        default: return "Type?";
    }
}

/**
 * @brief Convert a RuntimeRep to its short lowercase name (e.g. "int64", "ptr").
 * @param rep Runtime representation to describe.
 * @return A static string naming the representation, or "unknown" if unrecognized.
 */
const char* runtimeRepToString(RuntimeRep rep) {
    switch (rep) {
        case RuntimeRep::Int8: return "int8";
        case RuntimeRep::Int16: return "int16";
        case RuntimeRep::Int32: return "int32";
        case RuntimeRep::Int64: return "int64";
        case RuntimeRep::UInt8: return "uint8";
        case RuntimeRep::UInt16: return "uint16";
        case RuntimeRep::UInt32: return "uint32";
        case RuntimeRep::UInt64: return "uint64";
        case RuntimeRep::Float32: return "float32";
        case RuntimeRep::Float64: return "float64";
        case RuntimeRep::Pointer: return "ptr";
        case RuntimeRep::TaggedValue: return "tagged";
        case RuntimeRep::Struct: return "struct";
        case RuntimeRep::Erased: return "erased";
        default: return "unknown";
    }
}

/**
 * @brief Check whether a type is numeric, i.e. Number or one of its subtypes.
 * @param env Type environment used to resolve the subtype relationship.
 * @param id Type to test.
 * @return True if @p id is Number or a subtype of Number.
 */
bool isNumericType(const TypeEnvironment& env, TypeId id) {
    return env.isSubtype(id, BuiltinTypes::Number);
}

/**
 * @brief Check whether a type is a collection type (List, Vector, or Tensor, or a subtype thereof).
 * @param env Type environment used to resolve the subtype relationships.
 * @param id Type to test.
 * @return True if @p id is a subtype of List, Vector, or Tensor.
 */
bool isCollectionType(const TypeEnvironment& env, TypeId id) {
    return env.isSubtype(id, BuiltinTypes::List) ||
           env.isSubtype(id, BuiltinTypes::Vector) ||
           env.isSubtype(id, BuiltinTypes::Tensor);
}

// ============================================================================
// TYPE ENVIRONMENT IMPLEMENTATION
// ============================================================================

/**
 * @brief Construct a TypeEnvironment and populate it with all builtin types via initializeBuiltinTypes().
 */
TypeEnvironment::TypeEnvironment() {
    initializeBuiltinTypes();
}

/**
 * @brief Register every builtin type (universes, the numeric tower, text types,
 * collection/type-constructor families, autodiff and hash types, resource types, and proposition
 * types) and populate the common name aliases (e.g. "int", "i32", "float64") used for parsing
 * type annotations.
 */
void TypeEnvironment::initializeBuiltinTypes() {
    using namespace BuiltinTypes;

    // Register universes (meta-types)
    registerBuiltinType(TypeU0.id, "Type0", Universe::U1, 0, RuntimeRep::Erased);
    registerBuiltinType(TypeU1.id, "Type1", Universe::U2, 0, RuntimeRep::Erased);
    registerBuiltinType(TypeU2.id, "Type2", Universe::UOmega, 0, RuntimeRep::Erased);

    // ===== Value hierarchy (root supertype) =====
    registerBuiltinType(Value.id, "Value", Universe::U0, 0, RuntimeRep::TaggedValue);

    // ===== Numeric tower =====
    // Number is the supertype for all numeric types
    registerBuiltinType(Number.id, "Number", Universe::U0, 0,
                        RuntimeRep::TaggedValue, Value);

    // Integer branch: exact numbers
    registerBuiltinType(Integer.id, "Integer", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Number);
    registerBuiltinType(Int8.id, "Int8", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int8, Integer);
    registerBuiltinType(Int16.id, "Int16", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int16, Integer);
    registerBuiltinType(Int32.id, "Int32", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int32, Integer);
    registerBuiltinType(Int64.id, "Int64", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Integer);
    registerBuiltinType(ISize.id, "ISize", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Integer);
    registerBuiltinType(Natural.id, "Natural", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt64, Integer);
    registerBuiltinType(UInt8.id, "UInt8", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt8, Integer);
    registerBuiltinType(UInt16.id, "UInt16", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt16, Integer);
    registerBuiltinType(UInt32.id, "UInt32", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt32, Integer);
    registerBuiltinType(UInt64.id, "UInt64", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt64, Integer);
    registerBuiltinType(USize.id, "USize", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::UInt64, Integer);
    registerBuiltinType(BigInt.id, "BigInt", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Pointer, Integer);  // Heap-allocated arbitrary precision

    // Real branch: inexact numbers
    registerBuiltinType(Real.id, "Real", Universe::U0, 0,
                        RuntimeRep::Float64, Number);
    registerBuiltinType(Float64.id, "Float64", Universe::U0, 0,
                        RuntimeRep::Float64, Real);
    registerBuiltinType(Float32.id, "Float32", Universe::U0, 0,
                        RuntimeRep::Float64, Real);  // Represented as Float64 for simplicity

    // Complex number types (subtype of Number)
    registerBuiltinType(Complex.id, "Complex", Universe::U0, 0,
                        RuntimeRep::Struct, Number);  // Struct with real and imaginary parts
    registerBuiltinType(Complex64.id, "Complex64", Universe::U0, 0,
                        RuntimeRep::Struct, Complex);  // 2x Float32
    registerBuiltinType(Complex128.id, "Complex128", Universe::U0, 0,
                        RuntimeRep::Struct, Complex);  // 2x Float64

    // ===== Text types =====
    registerBuiltinType(Text.id, "Text", Universe::U0, 0,
                        RuntimeRep::Pointer, Value);
    registerBuiltinType(String.id, "String", Universe::U0, 0,
                        RuntimeRep::Pointer, Text);
    registerBuiltinType(Char.id, "Char", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Text);

    // ===== Other ground types =====
    registerBuiltinType(Boolean.id, "Boolean", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Value);
    registerBuiltinType(Null.id, "Null", Universe::U0, 0,
                        RuntimeRep::Int64, Value);
    registerBuiltinType(Symbol.id, "Symbol", Universe::U0, 0,
                        RuntimeRep::Pointer, Value);

    // ===== Type constructors (U1) =====
    auto list_id = registerTypeFamily(List.id, "List", Universe::U1, {"a"},
                                       RuntimeRep::Pointer);
    auto vec_id = registerTypeFamily(Vector.id, "Vector", Universe::U1, {"a"},
                                      RuntimeRep::Pointer);
    registerTypeFamily(Pair.id, "Pair", Universe::U1, {"a", "b"},
                       RuntimeRep::Pointer);
    registerTypeFamily(Function.id, "->", Universe::U1, {"a", "b"},
                       RuntimeRep::Pointer);
    registerTypeFamily(Tensor.id, "Tensor", Universe::U1, {"a", "shape"},
                       RuntimeRep::Pointer);
    registerTypeFamily(Pointer.id, "Ptr", Universe::U1, {"a"},
                       RuntimeRep::Pointer);
    registerTypeFamily(Closure.id, "Closure", Universe::U1, {"a", "b"},
                       RuntimeRep::Pointer);

    // ===== Autodiff types (U1) - for forward and reverse mode AD =====
    // DualNumber<T> = (primal: T, tangent: Float64) for forward-mode AD
    registerTypeFamily(DualNumber.id, "Dual", Universe::U1, {"a"},
                       RuntimeRep::Struct);
    // ADNode<T> = computation graph node for reverse-mode AD
    registerTypeFamily(ADNode.id, "ADNode", Universe::U1, {"a"},
                       RuntimeRep::Pointer);

    // ===== Hash/Map types (U1) - key-value collections =====
    // HashTable<K, V> = mutable hash table mapping keys of type K to values of type V
    registerTypeFamily(HashTable.id, "HashTable", Universe::U1, {"k", "v"},
                       RuntimeRep::Pointer);

    // ===== Resource types (U1, Linear) =====
    registerTypeFamily(Handle.id, "Handle", Universe::U1, {"k"},
                       RuntimeRep::Pointer);
    types_[Handle.id].id.flags |= TYPE_FLAG_LINEAR;

    registerTypeFamily(Buffer.id, "Buffer", Universe::U1, {"a", "n"},
                       RuntimeRep::Pointer);
    registerTypeFamily(Stream.id, "Stream", Universe::U1, {"a"},
                       RuntimeRep::Pointer);
    types_[Stream.id].id.flags |= TYPE_FLAG_LINEAR;

    // Qubit — a linear quantum resource (no-cloning). Runtime is an integer
    // index into a Moonlab state; the type system forbids using it twice.
    registerBuiltinType(Qubit.id, "Qubit", Universe::U0, TYPE_FLAG_LINEAR,
                        RuntimeRep::Int64);

    // ===== Proposition types (U2, Proof - erased at runtime) =====
    registerBuiltinType(Eq.id, "Eq", Universe::U2, TYPE_FLAG_PROOF,
                        RuntimeRep::Erased);
    registerBuiltinType(LessThan.id, "<", Universe::U2, TYPE_FLAG_PROOF,
                        RuntimeRep::Erased);
    registerBuiltinType(Bounded.id, "Bounded", Universe::U2, TYPE_FLAG_PROOF,
                        RuntimeRep::Erased);
    registerBuiltinType(Subtype.id, "<:", Universe::U2, TYPE_FLAG_PROOF,
                        RuntimeRep::Erased);

    // ===== Add common aliases =====
    name_to_id_["int"] = Int64;
    name_to_id_["i8"] = Int8;
    name_to_id_["int8"] = Int8;
    name_to_id_["i16"] = Int16;
    name_to_id_["int16"] = Int16;
    name_to_id_["i32"] = Int32;
    name_to_id_["int32"] = Int32;
    name_to_id_["i64"] = Int64;
    name_to_id_["int64"] = Int64;
    name_to_id_["integer"] = Int64;
    name_to_id_["isize"] = ISize;
    name_to_id_["u8"] = UInt8;
    name_to_id_["uint8"] = UInt8;
    name_to_id_["u16"] = UInt16;
    name_to_id_["uint16"] = UInt16;
    name_to_id_["u32"] = UInt32;
    name_to_id_["uint32"] = UInt32;
    name_to_id_["u64"] = UInt64;
    name_to_id_["uint64"] = UInt64;
    name_to_id_["usize"] = USize;
    name_to_id_["float"] = Float64;
    name_to_id_["float64"] = Float64;
    name_to_id_["double"] = Float64;
    name_to_id_["float32"] = Float32;
    name_to_id_["string"] = String;
    name_to_id_["str"] = String;
    name_to_id_["bool"] = Boolean;
    name_to_id_["boolean"] = Boolean;
    name_to_id_["char"] = Char;
    name_to_id_["number"] = Number;
    name_to_id_["natural"] = Natural;
    name_to_id_["nat"] = Natural;
    name_to_id_["real"] = Real;
    name_to_id_["list"] = List;
    name_to_id_["vector"] = Vector;
    name_to_id_["tensor"] = Tensor;
    name_to_id_["ptr"] = Pointer;
    name_to_id_["pointer"] = Pointer;
    name_to_id_["null"] = Null;
    name_to_id_["nil"] = Null;
    name_to_id_["symbol"] = Symbol;

    // Autodiff type aliases
    name_to_id_["dual"] = DualNumber;
    name_to_id_["dual-number"] = DualNumber;
    name_to_id_["ad-node"] = ADNode;
    name_to_id_["adnode"] = ADNode;
}

/**
 * @brief Register a non-parameterized builtin type in the type graph.
 *
 * Creates a TypeNode for @p id with the given @p name, @p level, @p flags, and runtime
 * representation @p rep, records it under both its numeric id and its @p name, and (if
 * @p supertype is given) links it into the supertype's subtype list via addSubtype().
 *
 * @return The TypeId assigned to the newly registered type.
 */
TypeId TypeEnvironment::registerBuiltinType(uint16_t id, const std::string& name,
                                             Universe level, uint8_t flags,
                                             RuntimeRep rep,
                                             std::optional<TypeId> supertype) {
    TypeId type_id{id, level, flags};

    TypeNode node;
    node.id = type_id;
    node.name = name;
    node.supertype = supertype;
    node.runtime_rep = rep;
    node.is_type_family = false;
    node.origin = TypeNode::Origin::Builtin;

    types_[id] = std::move(node);
    name_to_id_[name] = type_id;

    // Register as subtype of parent
    if (supertype) {
        addSubtype(*supertype, type_id);
    }

    return type_id;
}

/**
 * @brief Register a parameterized type family (e.g. List<A>, Pair<A,B>).
 *
 * Creates a TypeNode marked as a type family (is_type_family = true) with the given
 * @p param_names, and records it under both its numeric id and its @p name.
 *
 * @return The TypeId assigned to the newly registered type family.
 */
TypeId TypeEnvironment::registerTypeFamily(uint16_t id, const std::string& name,
                                            Universe level,
                                            const std::vector<std::string>& param_names,
                                            RuntimeRep rep) {
    TypeId type_id{id, level, 0};

    TypeNode node;
    node.id = type_id;
    node.name = name;
    node.runtime_rep = rep;
    node.param_names = param_names;
    node.is_type_family = true;
    node.origin = TypeNode::Origin::Builtin;

    types_[id] = std::move(node);
    name_to_id_[name] = type_id;

    return type_id;
}

/**
 * @brief Register a user-defined type (from a `define-type` form), assigning it a fresh id
 * starting at 1000.
 *
 * Creates a TypeNode marked with Origin::UserDefined, records it by name, and (if @p supertype
 * is given) links it into the supertype's subtype list via addSubtype().
 *
 * @return The newly assigned TypeId.
 */
TypeId TypeEnvironment::registerUserType(const std::string& name, Universe level,
                                          uint8_t flags,
                                          std::optional<TypeId> supertype) {
    uint16_t id = next_user_id_++;
    TypeId type_id{id, level, flags};

    TypeNode node;
    node.id = type_id;
    node.name = name;
    node.supertype = supertype;
    node.runtime_rep = RuntimeRep::TaggedValue;
    node.is_type_family = false;
    node.origin = TypeNode::Origin::UserDefined;

    types_[id] = std::move(node);
    name_to_id_[name] = type_id;

    if (supertype) {
        addSubtype(*supertype, type_id);
    }

    return type_id;
}

/**
 * @brief Record @p subtype as a child of @p supertype in the type graph.
 *
 * No-op if @p supertype is not a registered type.
 */
void TypeEnvironment::addSubtype(TypeId supertype, TypeId subtype) {
    auto it = types_.find(supertype.id);
    if (it != types_.end()) {
        it->second.subtypes.push_back(subtype);
    }
}

/**
 * @brief Look up a type by name, trying an exact match first and then a lowercase fallback.
 * @return The TypeId if found, or std::nullopt otherwise.
 */
std::optional<TypeId> TypeEnvironment::lookupType(const std::string& name) const {
    // Try exact match first
    auto it = name_to_id_.find(name);
    if (it != name_to_id_.end()) {
        return it->second;
    }

    // Try lowercase
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    it = name_to_id_.find(lower);
    if (it != name_to_id_.end()) {
        return it->second;
    }

    return std::nullopt;
}

/**
 * @brief Look up the TypeNode for a given TypeId.
 * @return Pointer to the stored TypeNode, or nullptr if @p id is not registered.
 */
const TypeNode* TypeEnvironment::getTypeNode(TypeId id) const {
    auto it = types_.find(id.id);
    return (it != types_.end()) ? &it->second : nullptr;
}

/**
 * @brief Get a human-readable name for a type.
 *
 * If @p id is a tracked pair type, returns "Pair<car, cdr>" built recursively from the element
 * types; otherwise returns the registered name, or "unknown" if @p id is not registered.
 */
std::string TypeEnvironment::getTypeName(TypeId id) const {
    // Check if this is a synthetic sum type
    auto sum_members = getSumMembers(id);
    if (sum_members) {
        std::string result = "(+";
        for (const auto& m : *sum_members) {
            result += " " + getTypeName(m);
        }
        result += ")";
        return result;
    }
    // Check if this is a tracked pair type
    auto pair_elems = getPairElementTypes(id);
    if (pair_elems) {
        return "Pair<" + getTypeName(pair_elems->first) + ", " +
               getTypeName(pair_elems->second) + ">";
    }
    const TypeNode* node = getTypeNode(id);
    return node ? node->name : "unknown";
}

/**
 * @brief Check whether @p sub is a subtype of @p super, using a cache to avoid re-walking the
 * type graph.
 *
 * On a cache miss, delegates to isSubtypeUncached() and stores the result.
 */
bool TypeEnvironment::isSubtype(TypeId sub, TypeId super) const {
    // Check cache
    auto key = std::make_pair(sub.id, super.id);
    auto it = subtype_cache_.find(key);
    if (it != subtype_cache_.end()) {
        return it->second;
    }

    bool result = isSubtypeUncached(sub, super);
    subtype_cache_[key] = result;
    return result;
}

/**
 * @brief Compute (without using the cache) whether @p sub is a subtype of @p super.
 *
 * Handles reflexivity (a type is a subtype of itself), tracked pair types (which are treated as
 * subtypes of the generic Pair type), and otherwise walks the supertype chain from @p sub looking
 * for @p super.
 */
bool TypeEnvironment::isSubtypeUncached(TypeId sub, TypeId super) const {
    // Reflexivity
    if (sub == super) return true;

    // Sum types. `sub <: (+ A B ...)` holds when sub fits any one arm; this is
    // the rule that lets a value of a concrete arm type (e.g. Vector) satisfy a
    // parameter declared with an explicit sum annotation. `(+ A B ...) <: super`
    // holds when *every* arm is a subtype of super (the sum is subsumed only by
    // a type that covers all its cases, e.g. (+ integer real) <: number).
    {
        auto super_members = getSumMembers(super);
        if (super_members) {
            for (const auto& arm : *super_members) {
                if (isSubtype(sub, arm)) return true;
            }
            // Fall through so a sum sub can still be checked arm-by-arm below,
            // but a non-sum sub that matched no arm is not a subtype.
        }
        auto sub_members = getSumMembers(sub);
        if (sub_members) {
            for (const auto& arm : *sub_members) {
                if (!isSubtype(arm, super)) return false;
            }
            return true;
        }
        if (super_members) {
            return false;  // non-sum sub matched no arm of the sum super
        }
    }

    // Tracked pair types are subtypes of Pair (and Pair's supertypes)
    if (isTrackedPairType(sub)) {
        if (super == BuiltinTypes::Pair) return true;
        return isSubtype(BuiltinTypes::Pair, super);
    }

    // Walk supertype chain
    const TypeNode* node = getTypeNode(sub);
    if (!node) return false;

    while (node->supertype.has_value()) {
        if (node->supertype.value() == super) return true;
        node = getTypeNode(node->supertype.value());
        if (!node) return false;
    }

    return false;
}

/**
 * @brief Find the least (most specific) common supertype of two types.
 *
 * Walks both types' supertype chains (via getSupertypeChain()) from most specific to most
 * general and returns the first type that appears in both.
 *
 * @return The common supertype, or std::nullopt if none exists.
 */
std::optional<TypeId> TypeEnvironment::leastCommonSupertype(TypeId a, TypeId b) const {
    // Same type
    if (a == b) return a;

    // Get supertype chains
    auto chain_a = getSupertypeChain(a);
    auto chain_b = getSupertypeChain(b);

    // Find first common element (going from specific to general)
    for (const auto& t : chain_a) {
        for (const auto& u : chain_b) {
            if (t == u) return t;
        }
    }

    return std::nullopt;
}

/**
 * @brief Compute the chain of supertypes from @p type up to the root, inclusive of @p type itself.
 *
 * Tracked pair types delegate to the generic Pair type's chain (prefixed by the pair type itself).
 */
std::vector<TypeId> TypeEnvironment::getSupertypeChain(TypeId type) const {
    std::vector<TypeId> chain;
    chain.push_back(type);

    // Tracked pair types (Pair<A,B>) are subtypes of Pair — delegate to Pair's chain
    if (isTrackedPairType(type)) {
        auto pair_chain = getSupertypeChain(BuiltinTypes::Pair);
        chain.insert(chain.end(), pair_chain.begin(), pair_chain.end());
        return chain;
    }

    const TypeNode* node = getTypeNode(type);
    while (node && node->supertype.has_value()) {
        chain.push_back(node->supertype.value());
        node = getTypeNode(node->supertype.value());
    }

    return chain;
}

/**
 * @brief Determine the result type of an arithmetic operation between two types, implementing
 * the Scheme numeric tower promotion rules (exact/inexact contagion, Complex/BigInt/Real handling).
 *
 * @return The promoted TypeId, falling back to the least common supertype (or Number) if no
 * specific promotion rule applies.
 */
TypeId TypeEnvironment::promoteForArithmetic(TypeId a, TypeId b) const {
    using namespace BuiltinTypes;

    // Same type, no promotion needed
    if (a == b) return a;

    // If one operand is Value (top type), use the other operand's type
    // This handles cases like (+ (car lst) 5) where car returns Value
    if (a == Value) return b;
    if (b == Value) return a;

    // If one operand is Number, use the other operand's type if numeric
    if (a == Number && (isSubtype(b, Integer) || isSubtype(b, Real))) return b;
    if (b == Number && (isSubtype(a, Integer) || isSubtype(a, Real))) return a;

    // Complex + anything numeric -> Complex
    if (a == Complex || a == Complex64 || a == Complex128 ||
        b == Complex || b == Complex64 || b == Complex128) {
        return Complex;
    }

    // BigInt + Int64 -> BigInt (exact promotion)
    if ((a == BigInt && isSubtype(b, Integer)) ||
        (b == BigInt && isSubtype(a, Integer))) {
        return BigInt;
    }

    // BigInt + Float64/Real -> Float64 (R7RS: exact+inexact -> inexact)
    if ((a == BigInt && isSubtype(b, Real)) ||
        (b == BigInt && isSubtype(a, Real))) {
        return Float64;
    }

    // Integer + Real -> Real (Float64)
    if ((isSubtype(a, Integer) && isSubtype(b, Real)) ||
        (isSubtype(a, Real) && isSubtype(b, Integer))) {
        return Float64;
    }

    // Generic arithmetic currently normalizes the integer family to Int64.
    // Width-specific machine integers are preserved for annotations, ABI
    // descriptions, and future freestanding lowering, but generic numeric
    // operators still execute through the existing Int64/Float64 paths.
    if (isSubtype(a, Integer) && isSubtype(b, Integer)) {
        return Int64;
    }

    // Both reals -> Float64
    if (isSubtype(a, Real) && isSubtype(b, Real)) {
        return Float64;
    }

    // Fallback to common supertype or Number
    auto lcs = leastCommonSupertype(a, b);
    return lcs.value_or(Number);
}

/**
 * @brief Check whether two types are equivalent. Currently this is identity only; type aliasing
 * is not yet implemented.
 */
bool TypeEnvironment::areEquivalent(TypeId a, TypeId b) const {
    // For now, just identity. Will support type aliases later.
    return a == b;
}

/**
 * @brief Get the runtime representation (LLVM/C-level layout) for a type.
 * @return The registered RuntimeRep, or RuntimeRep::TaggedValue if @p id is not registered.
 */
RuntimeRep TypeEnvironment::getRuntimeRep(TypeId id) const {
    const TypeNode* node = getTypeNode(id);
    return node ? node->runtime_rep : RuntimeRep::TaggedValue;
}

/**
 * @brief Map a runtime `eshkol_value_type_t` tag to the corresponding HoTT TypeId.
 *
 * Strips exactness flags for immediate types (tag < 8) but passes legacy pointer tags (>= 8,
 * e.g. CONS_PTR, STRING_PTR) through unmasked.
 *
 * @return The corresponding TypeId, or BuiltinTypes::Value if the tag is not recognized.
 */
TypeId TypeEnvironment::fromRuntimeType(uint8_t runtime_type) const {
    using namespace BuiltinTypes;

    // Map eshkol_value_type_t to HoTT TypeId
    // CRITICAL: Handle legacy types (>=32) correctly - do NOT use 0x0F mask!
    // Legacy types: CONS_PTR=32, STRING_PTR=33, VECTOR_PTR=34, TENSOR_PTR=35,
    //               HASH_PTR=36, EXCEPTION=37, CLOSURE_PTR=38, LAMBDA_SEXPR=39, AD_NODE_PTR=40

    // For immediate types (0-7), strip exactness flags
    uint8_t base_type = (runtime_type >= 8) ? runtime_type : (runtime_type & 0x0F);

    switch (base_type) {
        case ESHKOL_VALUE_NULL:
            return Null;
        case ESHKOL_VALUE_INT64:
            return Int64;
        case ESHKOL_VALUE_DOUBLE:
            return Float64;
        case ESHKOL_VALUE_CHAR:
            return Char;
        case ESHKOL_VALUE_BOOL:
            return Boolean;
        case ESHKOL_VALUE_SYMBOL:
            return Symbol;
        case ESHKOL_VALUE_DUAL_NUMBER:
            return DualNumber;
        // Legacy pointer types (at 32+)
        case ESHKOL_VALUE_CONS_PTR:
            return List;
        case ESHKOL_VALUE_STRING_PTR:
            return String;
        case ESHKOL_VALUE_VECTOR_PTR:
            return Vector;
        case ESHKOL_VALUE_TENSOR_PTR:
            return Tensor;
        case ESHKOL_VALUE_HASH_PTR:
            return HashTable;
        case ESHKOL_VALUE_CLOSURE_PTR:
            return Closure;
        case ESHKOL_VALUE_AD_NODE_PTR:
            return ADNode;
        default:
            return Value;  // Fallback to root type
    }
}

/**
 * @brief Map a HoTT TypeId back to a runtime `eshkol_value_type_t` tag.
 * @return The corresponding runtime tag, or ESHKOL_VALUE_NULL as a fallback for supertypes/unknown types.
 */
uint8_t TypeEnvironment::toRuntimeType(TypeId id) const {
    using namespace BuiltinTypes;

    // Map HoTT TypeId to eshkol_value_type_t
    if (id == Null) return ESHKOL_VALUE_NULL;
    if (id == Int8 || id == Int16 || id == Int32 || id == Int64 ||
        id == ISize || id == Integer || id == Natural ||
        id == UInt8 || id == UInt16 || id == UInt32 ||
        id == UInt64 || id == USize) {
        return ESHKOL_VALUE_INT64;
    }
    if (id == Float64 || id == Real) return ESHKOL_VALUE_DOUBLE;
    if (id == List) return ESHKOL_VALUE_CONS_PTR;
    if (id == String) return ESHKOL_VALUE_STRING_PTR;
    if (id == Char) return ESHKOL_VALUE_CHAR;
    if (id == Boolean) return ESHKOL_VALUE_BOOL;
    if (id == Vector) return ESHKOL_VALUE_VECTOR_PTR;
    if (id == Tensor) return ESHKOL_VALUE_TENSOR_PTR;
    if (id == Symbol) return ESHKOL_VALUE_SYMBOL;
    if (id == Closure || id == Function) return ESHKOL_VALUE_CLOSURE_PTR;
    if (id == DualNumber) return ESHKOL_VALUE_DUAL_NUMBER;
    if (id == ADNode) return ESHKOL_VALUE_AD_NODE_PTR;
    if (id == HashTable) return ESHKOL_VALUE_HASH_PTR;

    // For supertypes, fall back to tagged value
    return ESHKOL_VALUE_NULL;
}

// ============================================================================
// PARAMETERIZED TYPES IMPLEMENTATION
// ============================================================================

/**
 * @brief Create (and cache) a ParameterizedType instance for @p base_type applied to @p type_args.
 *
 * For example, instantiateType(List, {Int64}) represents List<Int64>.
 */
ParameterizedType TypeEnvironment::instantiateType(TypeId base_type,
                                                    const std::vector<TypeId>& type_args) const {
    ParameterizedType ptype;
    ptype.base_type = base_type;
    ptype.type_args = type_args;

    // Cache the instantiation for quick lookup
    parameterized_cache_[ptype] = base_type;

    return ptype;
}

/** @brief Create a List<element_type> parameterized type. Convenience wrapper over instantiateType(). */
ParameterizedType TypeEnvironment::makeListType(TypeId element_type) const {
    return instantiateType(BuiltinTypes::List, {element_type});
}

/** @brief Create a Vector<element_type> parameterized type. Convenience wrapper over instantiateType(). */
ParameterizedType TypeEnvironment::makeVectorType(TypeId element_type) const {
    return instantiateType(BuiltinTypes::Vector, {element_type});
}

/** @brief Create a Ptr<element_type> parameterized type. Convenience wrapper over instantiateType(). */
ParameterizedType TypeEnvironment::makePointerType(TypeId element_type) const {
    return instantiateType(BuiltinTypes::Pointer, {element_type});
}

/** @brief Create a HashTable<key_type, value_type> parameterized type. */
ParameterizedType TypeEnvironment::makeHashTableType(TypeId key_type, TypeId value_type) const {
    return instantiateType(BuiltinTypes::HashTable, {key_type, value_type});
}

/** @brief Check whether a type is a type family (parameterized type constructor like List or Vector). */
bool TypeEnvironment::isTypeFamily(TypeId id) const {
    const TypeNode* node = getTypeNode(id);
    return node && node->is_type_family;
}

/**
 * @brief Get the element type of a parameterized collection type.
 * @return @p ptype's first type argument, or BuiltinTypes::Value if it has none.
 */
TypeId TypeEnvironment::getElementType(const ParameterizedType& ptype) const {
    return ptype.elementType();
}

/**
 * @brief Infer the common element type of a homogeneous list from its elements' types.
 *
 * Starts from the first element's type and, for each subsequent differing type, attempts to
 * widen to their least common supertype.
 *
 * @return The inferred common type, or BuiltinTypes::Value if the list is empty or no common
 * supertype exists among the element types.
 */
TypeId TypeEnvironment::inferListElementType(const std::vector<TypeId>& element_types) const {
    if (element_types.empty()) {
        return BuiltinTypes::Value;  // Empty list has unknown element type
    }

    // Start with the first element's type
    TypeId common_type = element_types[0];

    // Check if all elements have the same type
    for (size_t i = 1; i < element_types.size(); i++) {
        if (element_types[i] != common_type) {
            // Different types - try to find least common supertype
            auto lcs = leastCommonSupertype(common_type, element_types[i]);
            if (lcs) {
                common_type = *lcs;
            } else {
                return BuiltinTypes::Value;  // No common supertype
            }
        }
    }

    return common_type;
}

// ============================================================================
// Dependent Type Dimension Tracking
// ============================================================================

/**
 * @brief Create a Vector<element_type, dimension> parameterized type carrying a known
 * compile-time dimension, and cache the instantiation.
 */
ParameterizedType TypeEnvironment::makeVectorTypeWithDim(TypeId element_type, uint64_t dimension) const {
    ParameterizedType ptype;
    ptype.base_type = BuiltinTypes::Vector;
    ptype.type_args = {element_type};
    ptype.value_args = {CTValueSimple::makeNat(dimension)};

    // Cache the instantiation
    parameterized_cache_[ptype] = BuiltinTypes::Vector;

    return ptype;
}

/**
 * @brief Create a Tensor<element_type, dims...> parameterized type carrying known compile-time
 * dimensions, and cache the instantiation.
 */
ParameterizedType TypeEnvironment::makeTensorTypeWithDims(TypeId element_type,
                                                           const std::vector<uint64_t>& dimensions) const {
    ParameterizedType ptype;
    ptype.base_type = BuiltinTypes::Tensor;
    ptype.type_args = {element_type};

    for (uint64_t dim : dimensions) {
        ptype.value_args.push_back(CTValueSimple::makeNat(dim));
    }

    // Cache the instantiation
    parameterized_cache_[ptype] = BuiltinTypes::Tensor;

    return ptype;
}

/** @brief Store dimension info for @p type_id so it can later be retrieved via getDimensionInfo(). */
void TypeEnvironment::storeDimensionInfo(TypeId type_id, const std::vector<CTValueSimple>& dimensions) {
    dimension_cache_[type_id.id] = dimensions;
}

/**
 * @brief Retrieve previously stored dimension info for a TypeId.
 * @return The stored dimensions, or std::nullopt if none were stored via storeDimensionInfo().
 */
std::optional<std::vector<CTValueSimple>> TypeEnvironment::getDimensionInfo(TypeId type_id) const {
    auto it = dimension_cache_.find(type_id.id);
    if (it != dimension_cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

// ============================================================================
// FUNCTION TYPES (Π-TYPES)
// ============================================================================

/**
 * @brief Create a non-variadic function type. Convenience overload forwarding to the
 * 3-argument makeFunctionType() with is_variadic = false.
 */
TypeId TypeEnvironment::makeFunctionType(const std::vector<TypeId>& param_types, TypeId return_type) const {
    return makeFunctionType(param_types, return_type, /*is_variadic=*/false);
}

/**
 * @brief Create or retrieve a cached function type for the given parameter types, return type,
 * and variadic flag.
 *
 * Searches function_type_cache_ for an existing PiType with matching return type, parameter
 * count, variadic flag, and parameter types; if found, its TypeId is reused. Otherwise a new
 * PiType is built and assigned a fresh id from next_function_type_id_.
 *
 * @return The TypeId identifying this function type.
 */
TypeId TypeEnvironment::makeFunctionType(const std::vector<TypeId>& param_types, TypeId return_type,
                                          bool is_variadic) const {
    // Check if we already have this exact function type cached
    for (const auto& entry : function_type_cache_) {
        const PiType& pi = entry.second;
        if (pi.return_type == return_type && pi.params.size() == param_types.size() &&
            pi.is_variadic == is_variadic) {
            bool match = true;
            for (size_t i = 0; i < param_types.size() && match; i++) {
                if (pi.params[i].type != param_types[i]) {
                    match = false;
                }
            }
            if (match) {
                return TypeId{entry.first, Universe::U0, 0};
            }
        }
    }

    // Create a new function type
    PiType pi;
    for (const auto& pt : param_types) {
        pi.params.push_back({"", pt, false});
    }
    pi.return_type = return_type;
    pi.is_dependent = false;
    pi.is_variadic = is_variadic;

    uint16_t type_id = next_function_type_id_++;
    function_type_cache_[type_id] = pi;

    return TypeId{type_id, Universe::U0, 0};
}

/** @brief Create a simple unary function type `input -> output`. */
TypeId TypeEnvironment::makeSimpleFunctionType(TypeId input, TypeId output) const {
    return makeFunctionType({input}, output);
}

/**
 * @brief Look up the PiType describing a function TypeId.
 * @return Pointer to the cached PiType, or nullptr if @p id is not a function type.
 */
const PiType* TypeEnvironment::getFunctionType(TypeId id) const {
    // Check if it's in the function type cache
    auto it = function_type_cache_.find(id.id);
    if (it != function_type_cache_.end()) {
        return &it->second;
    }

    // Not a function type
    return nullptr;
}

/**
 * @brief Check whether a TypeId represents a function type (the generic Function type or a
 * specific cached function signature).
 */
bool TypeEnvironment::isFunctionType(TypeId id) const {
    // Check if it's the base Function type or a specific function type
    if (id == BuiltinTypes::Function) {
        return true;
    }
    return function_type_cache_.find(id.id) != function_type_cache_.end();
}

/**
 * @brief Get the return type of a function TypeId.
 * @return The function's return type, or BuiltinTypes::Value if @p id is not a function type.
 */
TypeId TypeEnvironment::getFunctionReturnType(TypeId id) const {
    const PiType* pi = getFunctionType(id);
    if (pi) {
        return pi->return_type;
    }
    return BuiltinTypes::Value;
}

/**
 * @brief Get the parameter types of a function TypeId.
 * @return The function's parameter types in order, or an empty vector if @p id is not a
 * function type.
 */
std::vector<TypeId> TypeEnvironment::getFunctionParamTypes(TypeId id) const {
    const PiType* pi = getFunctionType(id);
    if (pi) {
        std::vector<TypeId> result;
        result.reserve(pi->params.size());
        for (const auto& param : pi->params) {
            result.push_back(param.type);
        }
        return result;
    }
    return {};
}

/**
 * @brief Build a human-readable name for a function type, e.g. "(Int64, Float64) -> Boolean".
 * @return "Function" if @p id is not a registered function type.
 */
std::string TypeEnvironment::getFunctionTypeName(TypeId id) const {
    const PiType* pi = getFunctionType(id);
    if (!pi) {
        return "Function";
    }

    std::string result;
    if (pi->params.size() == 1) {
        result = getTypeName(pi->params[0].type);
    } else {
        result = "(";
        for (size_t i = 0; i < pi->params.size(); i++) {
            if (i > 0) result += ", ";
            result += getTypeName(pi->params[i].type);
        }
        result += ")";
    }

    result += " -> ";
    result += getTypeName(pi->return_type);

    return result;
}

// ============================================================================
// PAIR TYPE TRACKING
// ============================================================================

/**
 * @brief Create or retrieve a cached Pair<car_type, cdr_type> TypeId, used by cons to propagate
 * element types through car/cdr.
 *
 * Searches pair_element_cache_ for a matching (car_type, cdr_type) pair; if found, its TypeId is
 * reused, otherwise a new id is allocated from next_pair_type_id_.
 */
TypeId TypeEnvironment::makePairType(TypeId car_type, TypeId cdr_type) const {
    // Check if we already have this exact pair type cached
    for (const auto& entry : pair_element_cache_) {
        if (entry.second.first == car_type && entry.second.second == cdr_type) {
            return TypeId{entry.first, Universe::U1, 0};
        }
    }

    // Allocate a new pair type ID
    uint16_t type_id = next_pair_type_id_++;
    pair_element_cache_[type_id] = {car_type, cdr_type};

    return TypeId{type_id, Universe::U1, 0};
}

/**
 * @brief Retrieve the (car_type, cdr_type) element types for a tracked pair TypeId.
 * @return The element type pair, or std::nullopt if @p id is not a tracked pair type.
 */
std::optional<std::pair<TypeId, TypeId>> TypeEnvironment::getPairElementTypes(TypeId id) const {
    auto it = pair_element_cache_.find(id.id);
    if (it != pair_element_cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

/**
 * @brief Check whether a TypeId represents a tracked pair type (one with known element types
 * created via makePairType()).
 */
bool TypeEnvironment::isTrackedPairType(TypeId id) const {
    return pair_element_cache_.find(id.id) != pair_element_cache_.end();
}

// ============================================================================
// SUM TYPE TRACKING
// ============================================================================

/**
 * @brief Create or retrieve a synthetic sum type over @p members.
 *
 * Flattens nested sums, de-duplicates arms (by id), and collapses degenerate
 * cases (empty → Value, single arm → that arm). Otherwise searches the cache
 * for an existing sum with the same normalized arm set and reuses it, or
 * allocates a fresh id from next_sum_type_id_.
 */
TypeId TypeEnvironment::makeSumType(const std::vector<TypeId>& members) const {
    // Normalize: flatten nested sums and de-duplicate arms preserving order.
    std::vector<TypeId> arms;
    std::function<void(const std::vector<TypeId>&)> flatten =
        [&](const std::vector<TypeId>& ms) {
            for (const auto& m : ms) {
                auto nested = getSumMembers(m);
                if (nested) {
                    flatten(*nested);
                    continue;
                }
                bool seen = false;
                for (const auto& a : arms) {
                    if (a.id == m.id) { seen = true; break; }
                }
                if (!seen) arms.push_back(m);
            }
        };
    flatten(members);

    if (arms.empty()) return BuiltinTypes::Value;
    if (arms.size() == 1) return arms[0];

    // Reuse an existing sum with the same arm set (order-insensitive).
    for (const auto& entry : sum_member_cache_) {
        if (entry.second.size() != arms.size()) continue;
        bool match = true;
        for (const auto& a : arms) {
            bool found = false;
            for (const auto& b : entry.second) {
                if (a.id == b.id) { found = true; break; }
            }
            if (!found) { match = false; break; }
        }
        if (match) {
            return TypeId{entry.first, Universe::U1, 0};
        }
    }

    uint16_t type_id = next_sum_type_id_++;
    sum_member_cache_[type_id] = arms;
    return TypeId{type_id, Universe::U1, 0};
}

/**
 * @brief Get the member arms of a sum TypeId, or nullopt if not a sum.
 */
std::optional<std::vector<TypeId>> TypeEnvironment::getSumMembers(TypeId id) const {
    auto it = sum_member_cache_.find(id.id);
    if (it != sum_member_cache_.end()) {
        return it->second;
    }
    return std::nullopt;
}

/**
 * @brief Check whether a TypeId represents a synthetic sum type.
 */
bool TypeEnvironment::isSumType(TypeId id) const {
    return sum_member_cache_.find(id.id) != sum_member_cache_.end();
}

/**
 * @brief Collapse a sum type to Pair (its historical codegen representation);
 * pass non-sum types through unchanged.
 */
TypeId TypeEnvironment::codegenReprOf(TypeId id) const {
    if (isSumType(id)) {
        return BuiltinTypes::Pair;
    }
    return id;
}

} // namespace eshkol::hott
