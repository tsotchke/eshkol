/**
 * @file dependent.cpp
 * @brief Implementation of Dependent Type Support for Eshkol HoTT Type System
 */

#include "eshkol/types/dependent.h"
#include <sstream>

namespace eshkol {
namespace hott {

// ===== DependentType Implementation =====

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
