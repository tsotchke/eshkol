/*
 * XLA Type Mapping Implementation for Eshkol
 *
 * Maps Eshkol's type system to XLA tensor types. Provides conversion
 * utilities for element types, shapes, and broadcasting rules.
 *
 * MLIR type conversion (toMLIRType, toMLIRElementType) is conditional on
 * MLIR availability. When MLIR is not linked, these functions return nullptr,
 * and the LLVM-direct codegen path handles all type mapping natively.
 *
 * HoTT type integration (fromHoTT) returns F64 scalar as the default mapping.
 * Full HoTT-to-XLA type extraction requires the MLIR path and will be
 * activated when MLIR+StableHLO are available.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/xla_types.h"
#include "eshkol/types/hott_types.h"
#include <sstream>
#include <numeric>

// MLIR includes (conditional compilation)
#if defined(ESHKOL_MLIR_AVAILABLE) && defined(ESHKOL_STABLEHLO_AVAILABLE)
#define ESHKOL_XLA_FULL_MLIR 1
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#endif

namespace eshkol {
namespace xla {

// ===== TensorType Implementation =====

/// Create a scalar tensor type (rank 0, empty shape).
TensorType TensorType::scalar(ElementType elem) {
    return TensorType{.element_type = elem, .shape = {}, .is_dynamic = false};
}

/// Create a 1-D vector tensor type.
TensorType TensorType::vector(ElementType elem, int64_t size) {
    return TensorType{
        .element_type = elem,
        .shape = {size},
        .is_dynamic = (size == kDynamic)
    };
}

/// Create a 2-D matrix tensor type.
TensorType TensorType::matrix(ElementType elem, int64_t rows, int64_t cols) {
    return TensorType{
        .element_type = elem,
        .shape = {rows, cols},
        .is_dynamic = (rows == kDynamic || cols == kDynamic)
    };
}

/// Create an N-D tensor type with arbitrary shape.
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

/// Return total number of elements, or -1 for dynamic tensors.
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

/// Return total size in bytes, or 0 for dynamic tensors.
size_t TensorType::sizeInBytes() const {
    int64_t elems = numElements();
    if (elems < 0) return 0;  // Dynamic
    return static_cast<size_t>(elems) * XLATypes::elementSize(element_type);
}

/// Structural equality comparison.
bool TensorType::operator==(const TensorType& other) const {
    return element_type == other.element_type &&
           shape == other.shape &&
           is_dynamic == other.is_dynamic;
}

/// Human-readable string representation, e.g. "f64[3, 4]".
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
#ifdef ESHKOL_XLA_FULL_MLIR
    mlir::MLIRContext* mlir_ctx_ = nullptr;

    /// Convert ElementType to MLIR Type. Returns null type if context not set.
    mlir::Type getMLIRElementType(ElementType elem) {
        if (!mlir_ctx_) return mlir::Type();
        switch (elem) {
            case ElementType::F16:  return mlir::Float16Type::get(mlir_ctx_);
            case ElementType::F32:  return mlir::Float32Type::get(mlir_ctx_);
            case ElementType::F64:  return mlir::Float64Type::get(mlir_ctx_);
            case ElementType::BF16: return mlir::BFloat16Type::get(mlir_ctx_);
            case ElementType::I8:   return mlir::IntegerType::get(mlir_ctx_, 8);
            case ElementType::I16:  return mlir::IntegerType::get(mlir_ctx_, 16);
            case ElementType::I32:  return mlir::IntegerType::get(mlir_ctx_, 32);
            case ElementType::I64:  return mlir::IntegerType::get(mlir_ctx_, 64);
            case ElementType::U8:   return mlir::IntegerType::get(mlir_ctx_, 8, mlir::IntegerType::Unsigned);
            case ElementType::U16:  return mlir::IntegerType::get(mlir_ctx_, 16, mlir::IntegerType::Unsigned);
            case ElementType::U32:  return mlir::IntegerType::get(mlir_ctx_, 32, mlir::IntegerType::Unsigned);
            case ElementType::U64:  return mlir::IntegerType::get(mlir_ctx_, 64, mlir::IntegerType::Unsigned);
            case ElementType::BOOL: return mlir::IntegerType::get(mlir_ctx_, 1);
            case ElementType::COMPLEX64:
                return mlir::ComplexType::get(mlir::Float32Type::get(mlir_ctx_));
            case ElementType::COMPLEX128:
                return mlir::ComplexType::get(mlir::Float64Type::get(mlir_ctx_));
        }
        return mlir::Float64Type::get(mlir_ctx_);  // Default
    }

    /// Convert TensorType to MLIR RankedTensorType. Returns null type if context not set.
    mlir::Type getMLIRTensorType(const TensorType& xla_type) {
        if (!mlir_ctx_) return mlir::Type();
        auto elemType = getMLIRElementType(xla_type.element_type);
        if (!elemType) return mlir::Type();

        if (xla_type.shape.empty()) {
            // Scalar: rank-0 tensor
            return mlir::RankedTensorType::get({}, elemType);
        }
        return mlir::RankedTensorType::get(xla_type.shape, elemType);
    }
#endif
};

XLATypes::XLATypes()
    : impl_(std::make_unique<Impl>()) {}

XLATypes::~XLATypes() = default;

// ===== HoTT to XLA =====

/// Convert a HoTT ParameterizedType to an XLA TensorType.
/// Extracts element type from type_args and shape from value_args.
/// Tensor<Float64, 3, 4> → TensorType{F64, [3, 4]}
std::optional<TensorType> XLATypes::fromHoTT(const ParameterizedType& hott_type) {
    // Only handle Tensor types (base_type.id == 104)
    if (hott_type.base_type.id != hott::BuiltinTypes::Tensor.id) {
        // For non-tensor types, return F64 scalar as default
        return TensorType::scalar(ElementType::F64);
    }

    // Extract element type from type_args[0]
    ElementType elem = ElementType::F64;  // default
    if (!hott_type.type_args.empty()) {
        uint16_t elem_id = hott_type.type_args[0].id;
        switch (elem_id) {
            case 13: elem = ElementType::I64; break;     // Int64
            case 16: elem = ElementType::F64; break;     // Float64
            case 17: elem = ElementType::F32; break;     // Float32
            case 25: elem = ElementType::BOOL; break;    // Boolean
            case 23: elem = ElementType::COMPLEX128; break; // Complex128
            case 19: elem = ElementType::COMPLEX64; break;  // Complex64
            default: elem = ElementType::F64; break;     // Default to f64
        }
    }

    // Extract shape from value_args
    std::vector<int64_t> shape;
    for (const auto& val : hott_type.value_args) {
        if (val.isNat()) {
            shape.push_back(static_cast<int64_t>(val.nat_value));
        } else {
            shape.push_back(TensorType::kDynamic);
        }
    }

    // If no shape info, return scalar
    if (shape.empty()) {
        return TensorType::scalar(elem);
    }

    return TensorType::tensor(elem, std::move(shape));
}

/// Map an Eshkol type tag to an XLA element type.
/// Based on Eshkol's tagged value type system.
ElementType XLATypes::elementTypeFromTag(uint8_t type_tag) {
    switch (type_tag) {
        case 1: return ElementType::I64;   // Integer
        case 2: return ElementType::F64;   // Double
        default: return ElementType::F64;  // Default to double (Eshkol's native numeric)
    }
}

// ===== XLA to MLIR =====

/// Set the MLIR context for type conversion. When MLIR is available, this
/// enables toMLIRType/toMLIRElementType to construct real MLIR types.
void XLATypes::setMLIRContext(void* ctx) {
#ifdef ESHKOL_XLA_FULL_MLIR
    impl_->mlir_ctx_ = static_cast<mlir::MLIRContext*>(ctx);
#else
    (void)ctx;
#endif
}

/// Convert an XLA TensorType to an MLIR RankedTensorType.
/// When MLIR is available and context is set, constructs a real
/// mlir::RankedTensorType. Returns nullptr in LLVM-direct mode.
/// The returned void* is a heap-allocated mlir::Type* — caller owns it.
void* XLATypes::toMLIRType(const TensorType& xla_type) {
#ifdef ESHKOL_XLA_FULL_MLIR
    auto type = impl_->getMLIRTensorType(xla_type);
    if (!type) return nullptr;
    return static_cast<void*>(new mlir::Type(type));
#else
    (void)xla_type;
    return nullptr;
#endif
}

/// Convert an XLA ElementType to an MLIR element type.
/// When MLIR is available and context is set, constructs a real
/// mlir::Type (Float64Type, IntegerType, etc.). Returns nullptr in LLVM-direct mode.
/// The returned void* is a heap-allocated mlir::Type* — caller owns it.
void* XLATypes::toMLIRElementType(ElementType elem) {
#ifdef ESHKOL_XLA_FULL_MLIR
    auto type = impl_->getMLIRElementType(elem);
    if (!type) return nullptr;
    return static_cast<void*>(new mlir::Type(type));
#else
    (void)elem;
    return nullptr;
#endif
}

// ===== Utilities =====

/// Return the size in bytes for a given element type.
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
    return 8;  // Default to 8 bytes (f64)
}

/// Return the canonical string name for an element type.
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

/// Compute the broadcast-compatible result shape for two tensor types.
/// Follows NumPy-style broadcasting rules: dimensions are compared from
/// the trailing end, and must be equal, 1, or dynamic to be compatible.
std::vector<int64_t> XLATypes::broadcastShape(const TensorType& lhs,
                                                const TensorType& rhs) {
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

/// Check if two tensor types are broadcast-compatible.
bool XLATypes::areBroadcastCompatible(const TensorType& lhs,
                                        const TensorType& rhs) {
    return !broadcastShape(lhs, rhs).empty();
}

} // namespace xla
} // namespace eshkol
