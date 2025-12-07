/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HoTT Type System - Homotopy Type Theory Foundation for Eshkol
 *
 * This module provides the compile-time type system with:
 * - Universe levels (U0, U1, U2)
 * - Supertype hierarchy (Value > Number > Integer > Int64)
 * - Type flags (exact, linear, proof)
 * - Subtype checking with caching
 * - Type promotion for arithmetic
 *
 * This is Phase 1 of the HoTT implementation, providing the foundation
 * for gradual typing in Eshkol v1.0-foundation.
 */
#ifndef ESHKOL_TYPES_HOTT_TYPES_H
#define ESHKOL_TYPES_HOTT_TYPES_H

#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <memory>

namespace eshkol::hott {

// ============================================================================
// UNIVERSE LEVELS - HoTT Foundation
// ============================================================================

/**
 * Universe levels for the type hierarchy.
 *
 * U0: Ground types (integers, floats, strings, chars, booleans)
 * U1: Type constructors (List, Vector, ->, Pair, Tensor)
 * U2: Propositions (Eq, <:, Bounded, Linear) - erased at runtime
 * UOmega: Universe polymorphic (for generic functions)
 */
enum class Universe : uint8_t {
    U0 = 0,      // Ground types: integer, float64, string, char, boolean
    U1 = 1,      // Type constructors: list, vector, ->, *, Handle, Buffer
    U2 = 2,      // Propositions: Eq, <:, Bounded, Linear (proof types)
    UOmega = 255 // Universe polymorphic
};

// ============================================================================
// TYPE FLAGS
// ============================================================================

/**
 * Type flags for additional type properties.
 */
enum TypeFlags : uint8_t {
    TYPE_FLAG_NONE     = 0,
    TYPE_FLAG_EXACT    = 1 << 0,  // Scheme exactness (integer vs inexact)
    TYPE_FLAG_LINEAR   = 1 << 1,  // Must use exactly once (linear types)
    TYPE_FLAG_PROOF    = 1 << 2,  // Compile-time only (erased at runtime)
    TYPE_FLAG_ABSTRACT = 1 << 3   // Cannot be instantiated directly
};

// ============================================================================
// RUNTIME REPRESENTATION MAPPING
// ============================================================================

/**
 * How a type is represented at runtime.
 * This maps HoTT types to LLVM/C representations.
 */
enum class RuntimeRep : uint8_t {
    Int64,       // 64-bit signed integer
    Float64,     // IEEE 754 double-precision
    Pointer,     // Pointer to heap object
    TaggedValue, // eshkol_tagged_value_t (16 bytes)
    Struct,      // LLVM struct (for dependent pairs, closures)
    Erased       // No runtime representation (proofs)
};

// ============================================================================
// TYPE IDENTIFIER
// ============================================================================

/**
 * Unique identifier for a type.
 *
 * Contains:
 * - id: Unique numeric identifier
 * - level: Universe membership
 * - flags: Type properties (exact, linear, proof)
 */
struct TypeId {
    uint16_t id;          // Unique identifier (0-999 builtin, 1000+ user)
    Universe level;       // Universe membership
    uint8_t flags;        // Type flags

    bool operator==(const TypeId& other) const { return id == other.id; }
    bool operator!=(const TypeId& other) const { return id != other.id; }
    bool operator<(const TypeId& other) const { return id < other.id; }

    // Flag queries
    bool isExact() const { return flags & TYPE_FLAG_EXACT; }
    bool isLinear() const { return flags & TYPE_FLAG_LINEAR; }
    bool isProof() const { return flags & TYPE_FLAG_PROOF; }
    bool isAbstract() const { return flags & TYPE_FLAG_ABSTRACT; }

    // Check if this is a valid (non-null) type
    bool isValid() const { return id != 0; }
};

// ============================================================================
// TYPE NODE - Full type definition
// ============================================================================

/**
 * Complete definition of a type in the type graph.
 *
 * Contains:
 * - Identity: id, name
 * - Hierarchy: supertype, subtypes
 * - Parameters: for type constructors like List<A>
 * - Runtime mapping
 */
struct TypeNode {
    TypeId id;
    std::string name;

    // Hierarchy
    std::optional<TypeId> supertype;
    std::vector<TypeId> subtypes;

    // Parameterization (for List<A>, Buffer<A,n>, etc.)
    std::vector<TypeId> parameters;       // Parameter type bounds
    std::vector<std::string> param_names; // Parameter names for display

    // Runtime mapping
    RuntimeRep runtime_rep;

    // For dependent types: is this a type family (takes value args)?
    bool is_type_family = false;

    // Source of definition
    enum class Origin { Builtin, UserDefined } origin = Origin::Builtin;
};

// ============================================================================
// BUILTIN TYPE IDS (Compile-time constants)
// ============================================================================

namespace BuiltinTypes {
    // Invalid/null type
    inline constexpr TypeId Invalid{0, Universe::U0, 0};

    // Universes (not really types, but useful for lookup)
    inline constexpr TypeId TypeU0{1, Universe::U0, 0};
    inline constexpr TypeId TypeU1{2, Universe::U1, 0};
    inline constexpr TypeId TypeU2{3, Universe::U2, 0};

    // Root supertype
    inline constexpr TypeId Value{10, Universe::U0, 0};

    // Numeric tower
    inline constexpr TypeId Number{11, Universe::U0, 0};
    inline constexpr TypeId Integer{12, Universe::U0, TYPE_FLAG_EXACT};
    inline constexpr TypeId Int64{13, Universe::U0, TYPE_FLAG_EXACT};
    inline constexpr TypeId Natural{14, Universe::U0, TYPE_FLAG_EXACT};
    inline constexpr TypeId Real{15, Universe::U0, 0};
    inline constexpr TypeId Float64{16, Universe::U0, 0};

    // Text types
    inline constexpr TypeId Text{20, Universe::U0, 0};
    inline constexpr TypeId String{21, Universe::U0, 0};
    inline constexpr TypeId Char{22, Universe::U0, TYPE_FLAG_EXACT};

    // Other ground types
    inline constexpr TypeId Boolean{25, Universe::U0, TYPE_FLAG_EXACT};
    inline constexpr TypeId Null{26, Universe::U0, 0};
    inline constexpr TypeId Symbol{27, Universe::U0, 0};

    // Type constructors (U1)
    inline constexpr TypeId List{100, Universe::U1, 0};
    inline constexpr TypeId Vector{101, Universe::U1, 0};
    inline constexpr TypeId Pair{102, Universe::U1, 0};
    inline constexpr TypeId Function{103, Universe::U1, 0};
    inline constexpr TypeId Tensor{104, Universe::U1, 0};
    inline constexpr TypeId Closure{105, Universe::U1, 0};

    // Autodiff types (U1) - for forward and reverse mode AD
    inline constexpr TypeId DualNumber{106, Universe::U1, 0};  // Forward-mode AD: (primal, tangent)
    inline constexpr TypeId ADNode{107, Universe::U1, 0};      // Reverse-mode AD: computation graph node

    // Resource types (U1, Linear)
    inline constexpr TypeId Handle{110, Universe::U1, TYPE_FLAG_LINEAR};
    inline constexpr TypeId Buffer{111, Universe::U1, 0};
    inline constexpr TypeId Stream{112, Universe::U1, TYPE_FLAG_LINEAR};

    // Proposition types (U2, Proof - erased at runtime)
    inline constexpr TypeId Eq{200, Universe::U2, TYPE_FLAG_PROOF};
    inline constexpr TypeId LessThan{201, Universe::U2, TYPE_FLAG_PROOF};
    inline constexpr TypeId Bounded{202, Universe::U2, TYPE_FLAG_PROOF};
    inline constexpr TypeId Subtype{203, Universe::U2, TYPE_FLAG_PROOF};
}

// ============================================================================
// PARAMETERIZED TYPE INSTANCE
// ============================================================================

/**
 * Represents an instantiated parameterized type like List<Int64>.
 *
 * This allows tracking the element type of collections at compile time.
 */
struct ParameterizedType {
    TypeId base_type;                    // The type family (e.g., List)
    std::vector<TypeId> type_args;       // The type arguments (e.g., [Int64])

    bool operator==(const ParameterizedType& other) const {
        return base_type == other.base_type && type_args == other.type_args;
    }

    bool operator<(const ParameterizedType& other) const {
        if (base_type.id != other.base_type.id) return base_type.id < other.base_type.id;
        return type_args < other.type_args;
    }

    // Check if this is a valid parameterized type (has type arguments)
    bool isParameterized() const { return !type_args.empty(); }

    // Get the element type for single-parameter types like List<T>
    TypeId elementType() const {
        return type_args.empty() ? BuiltinTypes::Value : type_args[0];
    }
};

// ============================================================================
// DEPENDENT FUNCTION TYPE (Π-TYPE)
// ============================================================================

/**
 * Represents a dependent function type (Π-type).
 *
 * A Π-type is a function where the return type can depend on the input value.
 * Examples:
 *   - (Π (n : Nat) (Vector Float64 n)) - function returning n-dimensional vector
 *   - (Π (T : Type) (T -> T)) - polymorphic identity function
 *
 * In HoTT, Π-types are more general than simple function types (->).
 * When the return type doesn't depend on the input, it degenerates to (A -> B).
 */
struct PiType {
    struct Parameter {
        std::string name;      // Parameter name (for dependent references)
        TypeId type;           // Parameter type
        bool is_value_param;   // True if this is a value parameter (for dimensions)
    };

    std::vector<Parameter> params;  // Input parameters (can be multiple for curried form)
    TypeId return_type;             // Return type (may reference param names)
    bool is_dependent;              // True if return type references params

    PiType() : return_type(BuiltinTypes::Value), is_dependent(false) {}

    PiType(std::vector<Parameter> p, TypeId ret, bool dep = false)
        : params(std::move(p)), return_type(ret), is_dependent(dep) {}

    // Create a simple non-dependent function type (A -> B)
    static PiType makeSimple(TypeId input, TypeId output) {
        return PiType({{std::string(), input, false}}, output, false);
    }

    // Create a multi-argument function type ((A, B) -> C)
    static PiType makeMulti(std::vector<TypeId> inputs, TypeId output) {
        std::vector<Parameter> params;
        for (auto& t : inputs) {
            params.push_back({std::string(), t, false});
        }
        return PiType(params, output, false);
    }

    // Get arity (number of parameters)
    size_t arity() const { return params.size(); }

    // Check if this is a simple (non-dependent) function type
    bool isSimpleFunction() const { return !is_dependent; }

    // Get first parameter type (for unary functions)
    TypeId inputType() const {
        return params.empty() ? BuiltinTypes::Value : params[0].type;
    }
};

// ============================================================================
// TYPE ENVIRONMENT
// ============================================================================

/**
 * Type environment maintaining all type information.
 *
 * Provides:
 * - Type registration (builtin and user-defined)
 * - Type lookup by name or id
 * - Subtype checking with caching
 * - Least common supertype computation
 * - Arithmetic type promotion
 * - Parameterized type instantiation (List<Int64>, etc.)
 */
class TypeEnvironment {
private:
    std::map<uint16_t, TypeNode> types_;
    std::map<std::string, TypeId> name_to_id_;
    uint16_t next_user_id_ = 1000;  // User-defined types start at 1000

    // Subtype cache for performance
    mutable std::map<std::pair<uint16_t, uint16_t>, bool> subtype_cache_;

    // Parameterized type cache: maps instantiated types to their info
    mutable std::map<ParameterizedType, TypeId> parameterized_cache_;

public:
    /**
     * Construct a TypeEnvironment with all builtin types registered.
     */
    TypeEnvironment();

    // ========== Type Registration ==========

    /**
     * Register a builtin type.
     */
    TypeId registerBuiltinType(uint16_t id, const std::string& name,
                               Universe level, uint8_t flags,
                               RuntimeRep rep,
                               std::optional<TypeId> supertype = std::nullopt);

    /**
     * Register a type family (parameterized type like List<A>).
     */
    TypeId registerTypeFamily(uint16_t id, const std::string& name,
                              Universe level,
                              const std::vector<std::string>& param_names,
                              RuntimeRep rep);

    /**
     * Register a user-defined type.
     * Returns the assigned TypeId.
     */
    TypeId registerUserType(const std::string& name, Universe level,
                            uint8_t flags, std::optional<TypeId> supertype);

    // ========== Type Lookup ==========

    /**
     * Look up a type by name.
     * Returns nullopt if not found.
     */
    std::optional<TypeId> lookupType(const std::string& name) const;

    /**
     * Get the TypeNode for a TypeId.
     * Returns nullptr if not found.
     */
    const TypeNode* getTypeNode(TypeId id) const;

    /**
     * Get type name by id.
     */
    std::string getTypeName(TypeId id) const;

    // ========== Subtyping ==========

    /**
     * Check if 'sub' is a subtype of 'super'.
     * Uses cached results for performance.
     */
    bool isSubtype(TypeId sub, TypeId super) const;

    /**
     * Find the least common supertype of two types.
     * Returns nullopt if no common supertype exists.
     */
    std::optional<TypeId> leastCommonSupertype(TypeId a, TypeId b) const;

    /**
     * Get the chain of supertypes from a type to the root.
     * The first element is the type itself.
     */
    std::vector<TypeId> getSupertypeChain(TypeId type) const;

    // ========== Arithmetic Promotion ==========

    /**
     * Determine the result type for arithmetic between two types.
     * Implements the Scheme numeric tower promotion rules.
     */
    TypeId promoteForArithmetic(TypeId a, TypeId b) const;

    // ========== Type Equivalence ==========

    /**
     * Check if two types are equivalent.
     * Currently just identity, but will support type aliases later.
     */
    bool areEquivalent(TypeId a, TypeId b) const;

    // ========== Runtime Mapping ==========

    /**
     * Get the runtime representation for a type.
     */
    RuntimeRep getRuntimeRep(TypeId id) const;

    /**
     * Map an eshkol_value_type_t to a HoTT TypeId.
     */
    TypeId fromRuntimeType(uint8_t runtime_type) const;

    /**
     * Map a HoTT TypeId to an eshkol_value_type_t.
     */
    uint8_t toRuntimeType(TypeId id) const;

    // ========== Parameterized Types ==========

    /**
     * Create or retrieve a parameterized type instance.
     * For example, instantiateType(List, {Int64}) creates List<Int64>.
     * Returns a ParameterizedType that can be used for element type tracking.
     */
    ParameterizedType instantiateType(TypeId base_type,
                                       const std::vector<TypeId>& type_args) const;

    /**
     * Create a List type with the given element type.
     * Convenience method for instantiateType(List, {element_type}).
     */
    ParameterizedType makeListType(TypeId element_type) const;

    /**
     * Create a Vector type with the given element type.
     */
    ParameterizedType makeVectorType(TypeId element_type) const;

    /**
     * Check if a type is a type family (parameterized type constructor).
     */
    bool isTypeFamily(TypeId id) const;

    /**
     * Get the element type from a parameterized collection type.
     * Returns Value (unknown) if the type is not parameterized.
     */
    TypeId getElementType(const ParameterizedType& ptype) const;

    /**
     * Infer the element type of a homogeneous list from its elements.
     * Returns Value if elements have different types, otherwise the common type.
     */
    TypeId inferListElementType(const std::vector<TypeId>& element_types) const;

private:
    /**
     * Initialize all builtin types.
     */
    void initializeBuiltinTypes();

    /**
     * Add a subtype relationship.
     */
    void addSubtype(TypeId supertype, TypeId subtype);

    /**
     * Uncached subtype check (walks the type graph).
     */
    bool isSubtypeUncached(TypeId sub, TypeId super) const;
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * Get a human-readable string for a Universe level.
 */
const char* universeToString(Universe u);

/**
 * Get a human-readable string for a RuntimeRep.
 */
const char* runtimeRepToString(RuntimeRep rep);

/**
 * Check if a type is numeric (subtype of Number).
 */
bool isNumericType(const TypeEnvironment& env, TypeId id);

/**
 * Check if a type is a collection (List, Vector, etc.).
 */
bool isCollectionType(const TypeEnvironment& env, TypeId id);

} // namespace eshkol::hott

#endif // ESHKOL_TYPES_HOTT_TYPES_H
