/**
 * @file dependent.cpp
 * @brief Implementation of Dependent Type Support for Eshkol HoTT Type System
 */

#include "eshkol/types/dependent.h"
#include "eshkol/eshkol.h"
#include <sstream>

namespace eshkol {
namespace hott {

// ===== CTValue Implementation =====

namespace {

/**
 * Result of constant expression evaluation.
 * Supports both integer and floating-point values.
 */
struct ConstValue {
    enum class Kind { Integer, Float, None };
    Kind kind = Kind::None;
    int64_t int_val = 0;
    double float_val = 0.0;

    /** @brief Create an integer-kind ConstValue holding @p v (also caching the double-converted value). */
    static ConstValue makeInt(int64_t v) {
        ConstValue cv;
        cv.kind = Kind::Integer;
        cv.int_val = v;
        cv.float_val = static_cast<double>(v);
        return cv;
    }

    /** @brief Create a float-kind ConstValue holding @p v (also caching the truncated integer value). */
    static ConstValue makeFloat(double v) {
        ConstValue cv;
        cv.kind = Kind::Float;
        cv.float_val = v;
        cv.int_val = static_cast<int64_t>(v);
        return cv;
    }

    /** @brief Create an invalid/empty ConstValue representing "could not evaluate". */
    static ConstValue none() {
        return ConstValue();
    }

    /** @brief True if this ConstValue holds an actual evaluated result (kind != None). */
    bool isValid() const { return kind != Kind::None; }
    /** @brief True if this ConstValue holds an integer result. */
    bool isInt() const { return kind == Kind::Integer; }
    /** @brief True if this ConstValue holds a floating-point result. */
    bool isFloat() const { return kind == Kind::Float; }

    // Get as double (works for both int and float)
    /** @brief Get the value as a double, regardless of whether it was stored as int or float. */
    double asDouble() const { return float_val; }

    // Get as int64 (truncates floats)
    /** @brief Get the value as an int64_t, truncating toward zero if it was stored as a float. */
    int64_t asInt() const { return int_val; }

    // Promote to float if either operand is float
    /**
     * @brief Add two constant values, promoting to float if either operand is a float.
     * @return The sum, or an invalid ConstValue if either operand is invalid.
     */
    static ConstValue add(const ConstValue& a, const ConstValue& b) {
        if (!a.isValid() || !b.isValid()) return none();
        if (a.isFloat() || b.isFloat()) {
            return makeFloat(a.asDouble() + b.asDouble());
        }
        return makeInt(a.asInt() + b.asInt());
    }

    /**
     * @brief Subtract two constant values, promoting to float if either operand is a float.
     * @return The difference (a - b), or an invalid ConstValue if either operand is invalid.
     */
    static ConstValue sub(const ConstValue& a, const ConstValue& b) {
        if (!a.isValid() || !b.isValid()) return none();
        if (a.isFloat() || b.isFloat()) {
            return makeFloat(a.asDouble() - b.asDouble());
        }
        return makeInt(a.asInt() - b.asInt());
    }

    /**
     * @brief Multiply two constant values, promoting to float if either operand is a float.
     * @return The product, or an invalid ConstValue if either operand is invalid.
     */
    static ConstValue mul(const ConstValue& a, const ConstValue& b) {
        if (!a.isValid() || !b.isValid()) return none();
        if (a.isFloat() || b.isFloat()) {
            return makeFloat(a.asDouble() * b.asDouble());
        }
        return makeInt(a.asInt() * b.asInt());
    }

    /**
     * @brief Divide two constant values, promoting to float if either operand is a float.
     * @return The quotient (a / b), or an invalid ConstValue if either operand is invalid or if
     * division by zero is detected.
     */
    static ConstValue div(const ConstValue& a, const ConstValue& b) {
        if (!a.isValid() || !b.isValid()) return none();
        // Check for division by zero
        if (b.isFloat() && b.asDouble() == 0.0) return none();
        if (b.isInt() && b.asInt() == 0) return none();

        if (a.isFloat() || b.isFloat()) {
            return makeFloat(a.asDouble() / b.asDouble());
        }
        return makeInt(a.asInt() / b.asInt());
    }
};

/**
 * Recursively evaluate an AST expression to a compile-time value.
 * Supports both integer and floating-point arithmetic.
 * Returns an invalid ConstValue if the expression cannot be evaluated at compile time.
 */
ConstValue evaluateConstExpr(const eshkol_ast_t* expr) {
    if (!expr) return ConstValue::none();

    switch (expr->type) {
        // Integer literals
        case ESHKOL_INT64:
            return ConstValue::makeInt(expr->int64_val);

        case ESHKOL_INT32:
            return ConstValue::makeInt(static_cast<int64_t>(expr->int32_val));

        case ESHKOL_INT16:
            return ConstValue::makeInt(static_cast<int64_t>(expr->int16_val));

        case ESHKOL_INT8:
            return ConstValue::makeInt(static_cast<int64_t>(expr->int8_val));

        // Unsigned integers
        case ESHKOL_UINT64:
            return ConstValue::makeInt(static_cast<int64_t>(expr->uint64_val));

        case ESHKOL_UINT32:
            return ConstValue::makeInt(static_cast<int64_t>(expr->uint32_val));

        case ESHKOL_UINT16:
            return ConstValue::makeInt(static_cast<int64_t>(expr->uint16_val));

        case ESHKOL_UINT8:
            return ConstValue::makeInt(static_cast<int64_t>(expr->uint8_val));

        // Floating-point literal
        case ESHKOL_DOUBLE:
            return ConstValue::makeFloat(expr->double_val);

        // Arithmetic operations
        case ESHKOL_OP: {
            switch (expr->operation.op) {
                case ESHKOL_ADD_OP: {
                    // Addition: evaluate all operands and sum
                    if (expr->operation.call_op.num_vars < 2) return ConstValue::none();
                    ConstValue result = evaluateConstExpr(&expr->operation.call_op.variables[0]);
                    if (!result.isValid()) return ConstValue::none();
                    for (uint64_t i = 1; i < expr->operation.call_op.num_vars; i++) {
                        ConstValue operand = evaluateConstExpr(&expr->operation.call_op.variables[i]);
                        if (!operand.isValid()) return ConstValue::none();
                        result = ConstValue::add(result, operand);
                    }
                    return result;
                }

                case ESHKOL_SUB_OP: {
                    // Subtraction: supports multiple operands (a - b - c - ...)
                    if (expr->operation.call_op.num_vars < 2) return ConstValue::none();
                    ConstValue result = evaluateConstExpr(&expr->operation.call_op.variables[0]);
                    if (!result.isValid()) return ConstValue::none();
                    for (uint64_t i = 1; i < expr->operation.call_op.num_vars; i++) {
                        ConstValue operand = evaluateConstExpr(&expr->operation.call_op.variables[i]);
                        if (!operand.isValid()) return ConstValue::none();
                        result = ConstValue::sub(result, operand);
                    }
                    return result;
                }

                case ESHKOL_MUL_OP: {
                    // Multiplication: evaluate all operands and multiply
                    if (expr->operation.call_op.num_vars < 2) return ConstValue::none();
                    ConstValue result = evaluateConstExpr(&expr->operation.call_op.variables[0]);
                    if (!result.isValid()) return ConstValue::none();
                    for (uint64_t i = 1; i < expr->operation.call_op.num_vars; i++) {
                        ConstValue operand = evaluateConstExpr(&expr->operation.call_op.variables[i]);
                        if (!operand.isValid()) return ConstValue::none();
                        result = ConstValue::mul(result, operand);
                    }
                    return result;
                }

                case ESHKOL_DIV_OP: {
                    // Division: supports multiple operands (a / b / c / ...)
                    if (expr->operation.call_op.num_vars < 2) return ConstValue::none();
                    ConstValue result = evaluateConstExpr(&expr->operation.call_op.variables[0]);
                    if (!result.isValid()) return ConstValue::none();
                    for (uint64_t i = 1; i < expr->operation.call_op.num_vars; i++) {
                        ConstValue operand = evaluateConstExpr(&expr->operation.call_op.variables[i]);
                        if (!operand.isValid()) return ConstValue::none();
                        result = ConstValue::div(result, operand);
                        if (!result.isValid()) return ConstValue::none();  // Division by zero
                    }
                    return result;
                }

                default:
                    // Non-arithmetic operation, cannot evaluate
                    return ConstValue::none();
            }
        }

        // Variables, function calls, etc. cannot be evaluated at compile time
        default:
            return ConstValue::none();
    }
}

} // anonymous namespace

/**
 * @brief Try to evaluate this CTValue as a concrete natural number (uint64_t).
 *
 * For Kind::Nat returns the stored value directly. For Kind::Expr, folds the referenced AST
 * expression via evaluateConstExpr(), accepting non-negative integer literals and integer-valued
 * float literals (and arithmetic combinations thereof).
 *
 * @return The evaluated value, or std::nullopt if it depends on a runtime value or is negative/non-integral.
 */
std::optional<uint64_t> CTValue::tryEvalNat() const {
    switch (kind_) {
        case Kind::Nat:
            return nat_val_;

        case Kind::Expr: {
            // Try to evaluate the expression as a constant
            ConstValue result = evaluateConstExpr(expr_);
            if (result.isValid()) {
                // For Nat (natural numbers), we need a non-negative integer
                if (result.isInt() && result.asInt() >= 0) {
                    return static_cast<uint64_t>(result.asInt());
                }
                // Also accept integer-valued floats
                if (result.isFloat()) {
                    double fval = result.asDouble();
                    if (fval >= 0.0 && fval == static_cast<double>(static_cast<uint64_t>(fval))) {
                        return static_cast<uint64_t>(fval);
                    }
                }
            }
            return std::nullopt;
        }

        case Kind::Bool:
        case Kind::Unknown:
        default:
            return std::nullopt;
    }
}

/**
 * @brief Try to evaluate this CTValue as a concrete double.
 *
 * For Kind::Nat, converts the stored natural number to double. For Kind::Expr, folds the
 * referenced AST expression via evaluateConstExpr().
 *
 * @return The evaluated value, or std::nullopt if it depends on a runtime value.
 */
std::optional<double> CTValue::tryEvalFloat() const {
    switch (kind_) {
        case Kind::Nat:
            return static_cast<double>(nat_val_);

        case Kind::Expr: {
            // Try to evaluate the expression as a constant
            ConstValue result = evaluateConstExpr(expr_);
            if (result.isValid()) {
                return result.asDouble();
            }
            return std::nullopt;
        }

        case Kind::Bool:
        case Kind::Unknown:
        default:
            return std::nullopt;
    }
}

// ===== DependentType Implementation =====

/**
 * @brief Render this dependent type as a string such as "Type#104<Type#16, 3, 4>".
 *
 * Note: base and parameter type names are printed as "Type#<id>" since no TypeEnvironment is
 * available here to resolve human-readable names; value indices use CTValue::toString().
 */
std::string DependentType::toString() const {
    std::ostringstream ss;

    // Get base type name (would need TypeEnvironment for this)
    ss << "Type#" << base.id;

    if (!type_indices.empty() || !value_indices.empty()) {
        ss << "<";
        bool first = true;

        // Type parameters
        for (const auto& t : type_indices) {
            if (!first) ss << ", ";
            ss << "Type#" << t.id;
            first = false;
        }

        // Value parameters
        for (const auto& v : value_indices) {
            if (!first) ss << ", ";
            ss << v.toString();
            first = false;
        }

        ss << ">";
    }

    return ss.str();
}

/**
 * @brief Structural equality check for dependent types.
 *
 * Requires the same base type, the same type indices (in order), and the same value indices
 * (compared via CTValue::equals(), each of which must yield CompareResult::True).
 */
bool DependentType::equals(const DependentType& other) const {
    // Check base type
    if (base != other.base) return false;

    // Check type indices
    if (type_indices.size() != other.type_indices.size()) return false;
    for (size_t i = 0; i < type_indices.size(); i++) {
        if (type_indices[i] != other.type_indices[i]) return false;
    }

    // Check value indices (structural equality)
    if (value_indices.size() != other.value_indices.size()) return false;
    for (size_t i = 0; i < value_indices.size(); i++) {
        auto cmp = value_indices[i].equals(other.value_indices[i]);
        if (cmp != CTValue::CompareResult::True) return false;
    }

    return true;
}

/**
 * @brief Subtype check for dependent types.
 *
 * Requires the base types to be subtypes per @p env, requires type indices to be pairwise equal
 * or subtypes (covariant), and requires all value indices to be provably equal
 * (CompareResult::True) — dimensions cannot differ between subtype and supertype.
 *
 * @return True if this type is a subtype of @p other under @p env.
 */
bool DependentType::isSubtypeOf(const DependentType& other, const TypeEnvironment& env) const {
    // First check if base types are subtypes
    if (!env.isSubtype(base, other.base)) {
        return false;
    }

    // For dependent types, type parameters must be equal (covariance for now)
    if (type_indices.size() != other.type_indices.size()) return false;
    for (size_t i = 0; i < type_indices.size(); i++) {
        // For type parameters, we require equality (could add variance later)
        if (type_indices[i] != other.type_indices[i]) {
            // Check if subtype relationship exists
            if (!env.isSubtype(type_indices[i], other.type_indices[i])) {
                return false;
            }
        }
    }

    // Value parameters must be equal for subtyping
    if (value_indices.size() != other.value_indices.size()) return false;
    for (size_t i = 0; i < value_indices.size(); i++) {
        auto cmp = value_indices[i].equals(other.value_indices[i]);
        // If we can't prove equality, assume not subtype
        if (cmp != CTValue::CompareResult::True) return false;
    }

    return true;
}

// ===== DimensionChecker Implementation =====

/**
 * @brief Check that a compile-time index is within `[0, bound)`.
 *
 * If both @p idx and @p bound evaluate to concrete naturals, checks the bound directly. If only
 * @p idx is known, or if @p idx cannot be evaluated at all, the check conservatively fails since
 * it cannot be statically proven.
 *
 * @return Result::success() only when both values are known and the index is in bounds;
 * otherwise a Result::failure() carrying a diagnostic message that includes @p context when
 * non-empty.
 */
DimensionChecker::Result DimensionChecker::checkBounds(
    const CTValue& idx, const CTValue& bound, const std::string& context) {

    // If both are known, check statically
    auto idx_val = idx.tryEvalNat();
    auto bound_val = bound.tryEvalNat();

    if (idx_val && bound_val) {
        if (*idx_val >= *bound_val) {
            std::ostringstream ss;
            ss << "Index " << *idx_val << " out of bounds (< " << *bound_val << ")";
            if (!context.empty()) {
                ss << " in " << context;
            }
            return Result::failure(ss.str());
        }
        return Result::success();
    }

    // If index is known but bound isn't, we can't verify
    if (idx_val && !bound_val) {
        std::ostringstream ss;
        ss << "Cannot statically verify index " << *idx_val << " against unknown bound";
        if (!context.empty()) {
            ss << " in " << context;
        }
        return Result::failure(ss.str());
    }

    // If index is unknown, require explicit proof
    std::ostringstream ss;
    ss << "Cannot statically verify index bound";
    if (!context.empty()) {
        ss << " in " << context;
    }
    ss << " (proof required)";
    return Result::failure(ss.str());
}

/**
 * @brief Check that two compile-time dimensions are equal.
 *
 * @return Result::success() if CTValue::equals() proves equality; a Result::failure() with a
 * mismatch or "proof required" message otherwise, including @p context when non-empty.
 */
DimensionChecker::Result DimensionChecker::checkDimensionsEqual(
    const CTValue& dim1, const CTValue& dim2, const std::string& context) {

    auto cmp = dim1.equals(dim2);

    if (cmp == CTValue::CompareResult::True) {
        return Result::success();
    }

    if (cmp == CTValue::CompareResult::False) {
        std::ostringstream ss;
        ss << "Dimension mismatch: " << dim1.toString() << " != " << dim2.toString();
        if (!context.empty()) {
            ss << " in " << context;
        }
        return Result::failure(ss.str());
    }

    // Unknown - need proof
    std::ostringstream ss;
    ss << "Cannot statically verify dimension equality: "
       << dim1.toString() << " =? " << dim2.toString();
    if (!context.empty()) {
        ss << " in " << context;
    }
    ss << " (proof required)";
    return Result::failure(ss.str());
}

/**
 * @brief Check matrix-multiplication dimension compatibility: (m x n) * (n x p) requires the
 * left matrix's column count to equal the right matrix's row count.
 *
 * @return Result::failure() if either side has fewer than 2 dimension indices; otherwise the
 * result of checkDimensionsEqual() on the shared "n" dimension.
 */
DimensionChecker::Result DimensionChecker::checkMatMulDimensions(
    const DependentType& left, const DependentType& right, const std::string& context) {

    // Matrix multiplication requires: (m x n) * (n x p) -> (m x p)
    // left should have 2 dimension indices, right should have 2
    if (left.value_indices.size() < 2 || right.value_indices.size() < 2) {
        return Result::failure("Matrix multiplication requires 2D matrices");
    }

    // Check that left's columns == right's rows (n == n)
    const CTValue& left_cols = left.value_indices[1];   // n
    const CTValue& right_rows = right.value_indices[0]; // n

    return checkDimensionsEqual(left_cols, right_rows,
        context.empty() ? "matrix multiplication" : context);
}

/**
 * @brief Check vector dot-product dimension compatibility: both vectors must have the same
 * (first) dimension.
 *
 * @return Result::failure() if either side has no dimension indices; otherwise the result of
 * checkDimensionsEqual() on the first dimension of each.
 */
DimensionChecker::Result DimensionChecker::checkDotProductDimensions(
    const DependentType& left, const DependentType& right, const std::string& context) {

    // Dot product requires same-dimension vectors
    if (left.value_indices.empty() || right.value_indices.empty()) {
        return Result::failure("Dot product requires vectors with known dimensions");
    }

    const CTValue& left_dim = left.value_indices[0];
    const CTValue& right_dim = right.value_indices[0];

    return checkDimensionsEqual(left_dim, right_dim,
        context.empty() ? "dot product" : context);
}

} // namespace hott
} // namespace eshkol
