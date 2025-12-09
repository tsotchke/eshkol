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
#include <stdexcept>

namespace eshkol::hott {

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const char* universeToString(Universe u) {
    switch (u) {
        case Universe::U0: return "Type\342\202\200";  // Type₀ (UTF-8 subscript 0)
        case Universe::U1: return "Type\342\202\201";  // Type₁
        case Universe::U2: return "Type\342\202\202";  // Type₂
        case Universe::UOmega: return "Type\317\211"; // Typeω
        default: return "Type?";
    }
}

const char* runtimeRepToString(RuntimeRep rep) {
    switch (rep) {
        case RuntimeRep::Int64: return "int64";
        case RuntimeRep::Float64: return "float64";
        case RuntimeRep::Pointer: return "ptr";
        case RuntimeRep::TaggedValue: return "tagged";
        case RuntimeRep::Struct: return "struct";
        case RuntimeRep::Erased: return "erased";
        default: return "unknown";
    }
}

bool isNumericType(const TypeEnvironment& env, TypeId id) {
    return env.isSubtype(id, BuiltinTypes::Number);
}

bool isCollectionType(const TypeEnvironment& env, TypeId id) {
    return env.isSubtype(id, BuiltinTypes::List) ||
           env.isSubtype(id, BuiltinTypes::Vector) ||
           env.isSubtype(id, BuiltinTypes::Tensor);
}

// ============================================================================
// TYPE ENVIRONMENT IMPLEMENTATION
// ============================================================================

TypeEnvironment::TypeEnvironment() {
    initializeBuiltinTypes();
}

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
    registerBuiltinType(Int64.id, "Int64", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Integer);
    registerBuiltinType(Natural.id, "Natural", Universe::U0, TYPE_FLAG_EXACT,
                        RuntimeRep::Int64, Integer);

    // Real branch: inexact numbers
    registerBuiltinType(Real.id, "Real", Universe::U0, 0,
                        RuntimeRep::Float64, Number);
    registerBuiltinType(Float64.id, "Float64", Universe::U0, 0,
                        RuntimeRep::Float64, Real);

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
    name_to_id_["int64"] = Int64;
    name_to_id_["integer"] = Int64;
    name_to_id_["float"] = Float64;
    name_to_id_["float64"] = Float64;
    name_to_id_["double"] = Float64;
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
    name_to_id_["null"] = Null;
    name_to_id_["nil"] = Null;
    name_to_id_["symbol"] = Symbol;

    // Autodiff type aliases
    name_to_id_["dual"] = DualNumber;
    name_to_id_["dual-number"] = DualNumber;
    name_to_id_["ad-node"] = ADNode;
    name_to_id_["adnode"] = ADNode;
}

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

void TypeEnvironment::addSubtype(TypeId supertype, TypeId subtype) {
    auto it = types_.find(supertype.id);
    if (it != types_.end()) {
        it->second.subtypes.push_back(subtype);
    }
}

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

const TypeNode* TypeEnvironment::getTypeNode(TypeId id) const {
    auto it = types_.find(id.id);
    return (it != types_.end()) ? &it->second : nullptr;
}

std::string TypeEnvironment::getTypeName(TypeId id) const {
    const TypeNode* node = getTypeNode(id);
    return node ? node->name : "unknown";
}

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

bool TypeEnvironment::isSubtypeUncached(TypeId sub, TypeId super) const {
    // Reflexivity
    if (sub == super) return true;

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

std::vector<TypeId> TypeEnvironment::getSupertypeChain(TypeId type) const {
    std::vector<TypeId> chain;
    chain.push_back(type);

    const TypeNode* node = getTypeNode(type);
    while (node && node->supertype.has_value()) {
        chain.push_back(node->supertype.value());
        node = getTypeNode(node->supertype.value());
    }

    return chain;
}

TypeId TypeEnvironment::promoteForArithmetic(TypeId a, TypeId b) const {
    using namespace BuiltinTypes;

    // Same type, no promotion needed
    if (a == b) return a;

    // Integer + Real -> Real (Float64)
    if ((isSubtype(a, Integer) && isSubtype(b, Real)) ||
        (isSubtype(a, Real) && isSubtype(b, Integer))) {
        return Float64;
    }

    // Both integers -> Int64
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

bool TypeEnvironment::areEquivalent(TypeId a, TypeId b) const {
    // For now, just identity. Will support type aliases later.
    return a == b;
}

RuntimeRep TypeEnvironment::getRuntimeRep(TypeId id) const {
    const TypeNode* node = getTypeNode(id);
    return node ? node->runtime_rep : RuntimeRep::TaggedValue;
}

TypeId TypeEnvironment::fromRuntimeType(uint8_t runtime_type) const {
    using namespace BuiltinTypes;

    // Map eshkol_value_type_t to HoTT TypeId
    switch (runtime_type & 0x0F) {
        case ESHKOL_VALUE_NULL:
            return Null;
        case ESHKOL_VALUE_INT64:
            return Int64;
        case ESHKOL_VALUE_DOUBLE:
            return Float64;
        case ESHKOL_VALUE_CONS_PTR:
            return List;
        case ESHKOL_VALUE_STRING_PTR:
            return String;
        case ESHKOL_VALUE_CHAR:
            return Char;
        case ESHKOL_VALUE_BOOL:
            return Boolean;
        case ESHKOL_VALUE_VECTOR_PTR:
            return Vector;
        case ESHKOL_VALUE_TENSOR_PTR:
            return Tensor;
        case ESHKOL_VALUE_SYMBOL:
            return Symbol;
        case ESHKOL_VALUE_CLOSURE_PTR:
            return Closure;
        case ESHKOL_VALUE_DUAL_NUMBER:
            return DualNumber;
        case ESHKOL_VALUE_AD_NODE_PTR:
            return ADNode;
        case ESHKOL_VALUE_HASH_PTR:
            return HashTable;
        default:
            return Value;  // Fallback to root type
    }
}

uint8_t TypeEnvironment::toRuntimeType(TypeId id) const {
    using namespace BuiltinTypes;

    // Map HoTT TypeId to eshkol_value_type_t
    if (id == Null) return ESHKOL_VALUE_NULL;
    if (id == Int64 || id == Integer || id == Natural) return ESHKOL_VALUE_INT64;
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

ParameterizedType TypeEnvironment::instantiateType(TypeId base_type,
                                                    const std::vector<TypeId>& type_args) const {
    ParameterizedType ptype;
    ptype.base_type = base_type;
    ptype.type_args = type_args;

    // Cache the instantiation for quick lookup
    parameterized_cache_[ptype] = base_type;

    return ptype;
}

ParameterizedType TypeEnvironment::makeListType(TypeId element_type) const {
    return instantiateType(BuiltinTypes::List, {element_type});
}

ParameterizedType TypeEnvironment::makeVectorType(TypeId element_type) const {
    return instantiateType(BuiltinTypes::Vector, {element_type});
}

ParameterizedType TypeEnvironment::makeHashTableType(TypeId key_type, TypeId value_type) const {
    return instantiateType(BuiltinTypes::HashTable, {key_type, value_type});
}

bool TypeEnvironment::isTypeFamily(TypeId id) const {
    const TypeNode* node = getTypeNode(id);
    return node && node->is_type_family;
}

TypeId TypeEnvironment::getElementType(const ParameterizedType& ptype) const {
    return ptype.elementType();
}

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

} // namespace eshkol::hott
