/**
 * @file dependent.h
 * @brief Dependent Type Support for Eshkol HoTT Type System
 *
 * This file provides compile-time value representation and dependent type
 * infrastructure for Phase 5 of the HoTT implementation.
 *
 * Key concepts:
 * - CTValue: Compile-time values for type indices (e.g., array dimensions)
 * - DependentType: Types parameterized by compile-time values
 * - Dimension checking: Static verification of array bounds
 */

#ifndef ESHKOL_TYPES_DEPENDENT_H
#define ESHKOL_TYPES_DEPENDENT_H

#include <cstdint>
#include <string>
#include <vector>
#include <optional>
#include <memory>
#include "hott_types.h"

// Forward declaration of AST type
struct eshkol_ast;
typedef struct eshkol_ast eshkol_ast_t;

namespace eshkol {
namespace hott {

/**
 * @class CTValue
 * @brief Compile-Time Value - represents values known at compile time
 *
 * CTValue is used for:
 * - Array dimension indices (e.g., the 100 in Vector<Float64, 100>)
 * - Type-level naturals for dependent types
 * - Compile-time boolean flags
 * - Symbolic expressions that may be reducible
 */
class CTValue {
public:
    enum class Kind {
        Nat,      // Natural number (uint64_t)
        Bool,     // Boolean value
        Expr,     // Symbolic expression (AST reference)
        Unknown   // Runtime-only value (cannot check statically)
    };

private:
    Kind kind_;
    union {
        uint64_t nat_val_;
        bool bool_val_;
    };
    // For symbolic expressions - stored as raw pointer (non-owning reference to AST)
    const eshkol_ast_t* expr_;

public:
    // Default constructor - creates Unknown value
    CTValue() : kind_(Kind::Unknown), nat_val_(0), expr_(nullptr) {}

    // Factory methods
    static CTValue makeNat(uint64_t n) {
        CTValue v;
        v.kind_ = Kind::Nat;
        v.nat_val_ = n;
        v.expr_ = nullptr;
        return v;
    }

    static CTValue makeBool(bool b) {
        CTValue v;
        v.kind_ = Kind::Bool;
        v.bool_val_ = b;
        v.expr_ = nullptr;
        return v;
    }

    static CTValue makeExpr(const eshkol_ast_t* e) {
        CTValue v;
        v.kind_ = Kind::Expr;
        v.nat_val_ = 0;
        v.expr_ = e;
        return v;
    }

    static CTValue makeUnknown() {
        return CTValue();
    }

    // Accessors
    Kind kind() const { return kind_; }
    bool isNat() const { return kind_ == Kind::Nat; }
    bool isBool() const { return kind_ == Kind::Bool; }
    bool isExpr() const { return kind_ == Kind::Expr; }
    bool isUnknown() const { return kind_ == Kind::Unknown; }
    bool isKnown() const { return kind_ != Kind::Unknown; }

    uint64_t natValue() const { return nat_val_; }
    bool boolValue() const { return bool_val_; }
    const eshkol_ast_t* exprValue() const { return expr_; }

    // Evaluation - try to get a concrete natural number value
    std::optional<uint64_t> tryEvalNat() const {
        if (kind_ == Kind::Nat) {
            return nat_val_;
        }
        // TODO: Add symbolic expression evaluation for constant expressions
        return std::nullopt;
    }

    // Evaluation - try to get a concrete boolean value
    std::optional<bool> tryEvalBool() const {
        if (kind_ == Kind::Bool) {
            return bool_val_;
        }
        return std::nullopt;
    }

    // Comparison operations for dimension checking
    enum class CompareResult {
        True,      // Definitely true
        False,     // Definitely false
        Unknown    // Cannot determine statically
    };

    // Check if this < other
    CompareResult lessThan(const CTValue& other) const {
        auto lhs = tryEvalNat();
        auto rhs = other.tryEvalNat();
        if (lhs && rhs) {
            return *lhs < *rhs ? CompareResult::True : CompareResult::False;
        }
        return CompareResult::Unknown;
    }

    // Check if this == other
    CompareResult equals(const CTValue& other) const {
        if (kind_ != other.kind_) {
            // Different kinds might still be equal in some cases
            return CompareResult::Unknown;
        }
        if (kind_ == Kind::Nat) {
            return nat_val_ == other.nat_val_ ? CompareResult::True : CompareResult::False;
        }
        if (kind_ == Kind::Bool) {
            return bool_val_ == other.bool_val_ ? CompareResult::True : CompareResult::False;
        }
        return CompareResult::Unknown;
    }

    // Arithmetic operations (for type-level computations)
    CTValue add(const CTValue& other) const {
        auto lhs = tryEvalNat();
        auto rhs = other.tryEvalNat();
        if (lhs && rhs) {
            return makeNat(*lhs + *rhs);
        }
        return makeUnknown();
    }

    CTValue mul(const CTValue& other) const {
        auto lhs = tryEvalNat();
        auto rhs = other.tryEvalNat();
        if (lhs && rhs) {
            return makeNat(*lhs * *rhs);
        }
        return makeUnknown();
    }

    // String representation
    std::string toString() const {
        switch (kind_) {
            case Kind::Nat: return std::to_string(nat_val_);
            case Kind::Bool: return bool_val_ ? "#t" : "#f";
            case Kind::Expr: return "<expr>";
            case Kind::Unknown: return "?";
        }
        return "<?>";
    }
};

/**
 * @class DependentType
 * @brief Represents a type parameterized by compile-time values
 *
 * Examples:
 * - Buffer<Float64, 100> -> base=Buffer, indices=[Float64, Nat(100)]
 * - Matrix<Int64, 3, 4> -> base=Matrix, indices=[Int64, Nat(3), Nat(4)]
 * - Vector<T, n> -> base=Vector, indices=[T, Expr(n)]
 */
class DependentType {
public:
    TypeId base;                       // Base type (e.g., Buffer, Vector, Matrix)
    std::vector<TypeId> type_indices;  // Type parameters
    std::vector<CTValue> value_indices; // Value parameters (dimensions, etc.)

    DependentType() : base(BuiltinTypes::Value) {}

    DependentType(TypeId b) : base(b) {}

    DependentType(TypeId b, std::vector<TypeId> types, std::vector<CTValue> values)
        : base(b), type_indices(std::move(types)), value_indices(std::move(values)) {}

    // Check if this is a simple type (no dependent parameters)
    bool isSimple() const {
        return value_indices.empty();
    }

    // Check if all value indices are known at compile time
    bool allValuesKnown() const {
        for (const auto& v : value_indices) {
            if (!v.isKnown()) return false;
        }
        return true;
    }

    // Get dimension (for vector/matrix types)
    std::optional<uint64_t> getDimension(size_t idx = 0) const {
        if (idx < value_indices.size()) {
            return value_indices[idx].tryEvalNat();
        }
        return std::nullopt;
    }

    // String representation
    std::string toString() const;

    // Type equality (structural)
    bool equals(const DependentType& other) const;

    // Subtype checking for dependent types
    bool isSubtypeOf(const DependentType& other, const TypeEnvironment& env) const;
};

/**
 * @class SigmaType
 * @brief Dependent pair type (Σ-type / existential type)
 *
 * A Σ-type represents an existential quantification: Σ(x:A).B(x)
 * meaning "there exists an x of type A such that B(x) holds".
 *
 * Examples:
 *   - Σ(n:Nat).Vector<Float64,n> - a vector of some length
 *   - Σ(T:Type).List<T> - a list of some element type
 *
 * In HoTT, Σ-types are dual to Π-types:
 *   - Π-type: "for all x:A, B(x)" (universal)
 *   - Σ-type: "exists x:A, B(x)" (existential)
 *
 * A Σ-type value is a dependent pair (a, b) where:
 *   - a : A (the witness)
 *   - b : B(a) (the dependent component)
 */
struct SigmaType {
    std::string witness_name;    // Name of the bound variable (e.g., "n", "T")
    TypeId witness_type;         // Type of the first component (e.g., Nat, Type)
    TypeId body_type;            // Type of second component (may reference witness)
    bool is_dependent;           // True if body_type references witness_name

    SigmaType()
        : witness_type(BuiltinTypes::Value)
        , body_type(BuiltinTypes::Value)
        , is_dependent(false) {}

    SigmaType(const std::string& name, TypeId wit_type, TypeId bod_type, bool dep = true)
        : witness_name(name)
        , witness_type(wit_type)
        , body_type(bod_type)
        , is_dependent(dep) {}

    // Create a simple (non-dependent) product type A × B
    static SigmaType makeProduct(TypeId left, TypeId right) {
        return SigmaType("", left, right, false);
    }

    // Check if this is a simple product (non-dependent pair)
    bool isSimpleProduct() const { return !is_dependent; }

    // Get the first projection type (always witness_type)
    TypeId firstType() const { return witness_type; }

    // Get the second projection type (body_type, but may depend on first value)
    TypeId secondType() const { return body_type; }
};

/**
 * @class SigmaValue
 * @brief Runtime representation of a dependent pair value
 *
 * Holds both components of a Σ-type value: (witness, body)
 */
struct SigmaValue {
    CTValue witness;             // The first component (compile-time if known)
    TypeId witness_runtime_type; // Runtime type of witness
    TypeId body_runtime_type;    // Runtime type of body (after substitution)

    SigmaValue()
        : witness_runtime_type(BuiltinTypes::Value)
        , body_runtime_type(BuiltinTypes::Value) {}

    SigmaValue(const CTValue& wit, TypeId wit_type, TypeId bod_type)
        : witness(wit)
        , witness_runtime_type(wit_type)
        , body_runtime_type(bod_type) {}

    // Check if the witness is known at compile time
    bool hasKnownWitness() const { return witness.isKnown(); }
};

/**
 * @class DimensionChecker
 * @brief Static verification of array/vector dimension bounds
 *
 * Used to verify at compile time that array indices are within bounds,
 * vector dimensions match for operations, etc.
 */
class DimensionChecker {
public:
    struct Result {
        bool valid;
        std::string error_message;

        static Result success() { return {true, ""}; }
        static Result failure(const std::string& msg) { return {false, msg}; }
    };

    // Check if index is within bounds: 0 <= idx < bound
    static Result checkBounds(const CTValue& idx, const CTValue& bound,
                              const std::string& context = "");

    // Check if two dimensions are equal
    static Result checkDimensionsEqual(const CTValue& dim1, const CTValue& dim2,
                                       const std::string& context = "");

    // Check matrix multiplication compatibility: (m x n) * (n x p) -> (m x p)
    static Result checkMatMulDimensions(const DependentType& left,
                                        const DependentType& right,
                                        const std::string& context = "");

    // Check vector dot product compatibility: same dimensions
    static Result checkDotProductDimensions(const DependentType& left,
                                            const DependentType& right,
                                            const std::string& context = "");
};

} // namespace hott
} // namespace eshkol

#endif // ESHKOL_TYPES_DEPENDENT_H
