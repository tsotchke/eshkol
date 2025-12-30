/*
 * XLA Type Mapping Implementation for Eshkol
 *
 * STUB IMPLEMENTATION - Phase 1
 * This is a skeleton that compiles but doesn't connect to MLIR yet.
 * Actual type mapping will be added in Phase 2.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_types.h"
#include <sstream>
#include <numeric>

namespace eshkol {
namespace xla {

// ===== TensorType Implementation =====

TensorType TensorType::scalar(ElementType elem) {
    return TensorType{.element_type = elem, .shape = {}, .is_dynamic = false};
}

TensorType TensorType::vector(ElementType elem, int64_t size) {
    return TensorType{
        .element_type = elem,
        .shape = {size},
        .is_dynamic = (size == kDynamic)
    };
}

TensorType TensorType::matrix(ElementType elem, int64_t rows, int64_t cols) {
    return TensorType{
        .element_type = elem,
        .shape = {rows, cols},
        .is_dynamic = (rows == kDynamic || cols == kDynamic)
    };
}

TensorType TensorType::tensor(ElementType elem, std::vector<int64_t> shape) {
    bool dynamic = false;
    for (auto dim : shape) {
        if (dim == kDynamic) {
            dynamic = true;
            break;
        }
    }
    return TensorType{
        .element_type = elem,
        .shape = std::move(shape),
        .is_dynamic = dynamic
    };
}

int64_t TensorType::numElements() const {
    if (shape.empty()) return 1;  // Scalar
    if (is_dynamic) return -1;    // Unknown

    int64_t total = 1;
    for (auto dim : shape) {
        if (dim == kDynamic) return -1;
        total *= dim;
    }
    return total;
}

size_t TensorType::sizeInBytes() const {
    int64_t elems = numElements();
    if (elems < 0) return 0;  // Dynamic
    return static_cast<size_t>(elems) * XLATypes::elementSize(element_type);
}

bool TensorType::operator==(const TensorType& other) const {
    return element_type == other.element_type &&
           shape == other.shape &&
           is_dynamic == other.is_dynamic;
}

std::string TensorType::toString() const {
    std::ostringstream oss;
    oss << XLATypes::elementTypeName(element_type) << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        if (shape[i] == kDynamic) {
            oss << "?";
        } else {
            oss << shape[i];
        }
    }
    oss << "]";
    return oss.str();
}

// ===== XLATypes Implementation =====

class XLATypes::Impl {
public:
    // TODO: Add MLIR context reference in Phase 2
};

XLATypes::XLATypes()
    : impl_(std::make_unique<Impl>()) {}

XLATypes::~XLATypes() = default;

// ===== HoTT to XLA (Stubs) =====

std::optional<TensorType> XLATypes::fromHoTT(const ParameterizedType& hott_type) {
    (void)hott_type;
    // STUB: Return a default tensor type for now
    // Will be implemented when HoTT types are integrated
    return TensorType::scalar(ElementType::F64);
}

ElementType XLATypes::elementTypeFromTag(uint8_t type_tag) {
    // Based on Eshkol tagged value types
    // 1 = int64, 2 = double, etc.
    switch (type_tag) {
        case 1: return ElementType::I64;   // Integer
        case 2: return ElementType::F64;   // Double
        default: return ElementType::F64;  // Default to double
    }
}

// ===== XLA to MLIR (Stubs) =====

void* XLATypes::toMLIRType(const TensorType& xla_type) {
    (void)xla_type;
    // STUB: Will be implemented with MLIR integration
    return nullptr;
}

void* XLATypes::toMLIRElementType(ElementType elem) {
    (void)elem;
    // STUB: Will be implemented with MLIR integration
    return nullptr;
}

// ===== Utilities =====

size_t XLATypes::elementSize(ElementType elem) {
    switch (elem) {
        case ElementType::F16:
        case ElementType::BF16:
            return 2;
        case ElementType::F32:
        case ElementType::I32:
        case ElementType::U32:
            return 4;
        case ElementType::F64:
        case ElementType::I64:
        case ElementType::U64:
        case ElementType::COMPLEX64:
            return 8;
        case ElementType::COMPLEX128:
            return 16;
        case ElementType::I8:
        case ElementType::U8:
        case ElementType::BOOL:
            return 1;
        case ElementType::I16:
        case ElementType::U16:
            return 2;
    }
    return 8;  // Default to 8 bytes
}

std::string XLATypes::elementTypeName(ElementType elem) {
    switch (elem) {
        case ElementType::F16: return "f16";
        case ElementType::F32: return "f32";
        case ElementType::F64: return "f64";
        case ElementType::BF16: return "bf16";
        case ElementType::I8: return "i8";
        case ElementType::I16: return "i16";
        case ElementType::I32: return "i32";
        case ElementType::I64: return "i64";
        case ElementType::U8: return "u8";
        case ElementType::U16: return "u16";
        case ElementType::U32: return "u32";
        case ElementType::U64: return "u64";
        case ElementType::BOOL: return "bool";
        case ElementType::COMPLEX64: return "complex64";
        case ElementType::COMPLEX128: return "complex128";
    }
    return "unknown";
}

std::vector<int64_t> XLATypes::broadcastShape(const TensorType& lhs,
                                                const TensorType& rhs) {
    // Implement NumPy-style broadcasting rules
    std::vector<int64_t> result;

    size_t max_rank = std::max(lhs.shape.size(), rhs.shape.size());
    result.resize(max_rank);

    for (size_t i = 0; i < max_rank; ++i) {
        int64_t lhs_dim = (i < lhs.shape.size())
            ? lhs.shape[lhs.shape.size() - 1 - i]
            : 1;
        int64_t rhs_dim = (i < rhs.shape.size())
            ? rhs.shape[rhs.shape.size() - 1 - i]
            : 1;

        // Check compatibility
        if (lhs_dim != rhs_dim && lhs_dim != 1 && rhs_dim != 1 &&
            lhs_dim != TensorType::kDynamic && rhs_dim != TensorType::kDynamic) {
            return {};  // Incompatible shapes
        }

        // Result dimension is the larger one (unless dynamic)
        if (lhs_dim == TensorType::kDynamic || rhs_dim == TensorType::kDynamic) {
            result[max_rank - 1 - i] = TensorType::kDynamic;
        } else {
            result[max_rank - 1 - i] = std::max(lhs_dim, rhs_dim);
        }
    }

    return result;
}

bool XLATypes::areBroadcastCompatible(const TensorType& lhs,
                                        const TensorType& rhs) {
    return !broadcastShape(lhs, rhs).empty();
}

} // namespace xla
} // namespace eshkol
