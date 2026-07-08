/*
 * XLA Type Mapping for Eshkol
 *
 * Maps Eshkol's HoTT type system to MLIR/StableHLO types.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_XLA_TYPES_H
#define ESHKOL_XLA_TYPES_H

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace eshkol {
namespace hott {
struct ParameterizedType;
} // namespace hott

/**
 * Type alias for backward compatibility with code referring to the
 * un-namespaced ParameterizedType (now defined in eshkol::hott).
 */
using ParameterizedType = hott::ParameterizedType;

namespace xla {

/**
 * Element types supported by XLA
 */
enum class ElementType {
    F16,    // 16-bit float
    F32,    // 32-bit float
    F64,    // 64-bit float (Eshkol default)
    BF16,   // bfloat16
    I8,     // 8-bit signed integer
    I16,    // 16-bit signed integer
    I32,    // 32-bit signed integer
    I64,    // 64-bit signed integer
    U8,     // 8-bit unsigned integer
    U16,    // 16-bit unsigned integer
    U32,    // 32-bit unsigned integer
    U64,    // 64-bit unsigned integer
    BOOL,   // Boolean
    COMPLEX64,   // Complex with F32 components
    COMPLEX128   // Complex with F64 components
};

/**
 * Tensor type with shape and element type
 */
struct TensorType {
    ElementType element_type;         // Scalar element type
    std::vector<int64_t> shape;      // Empty for scalar
    bool is_dynamic = false;          // Any dynamic dimensions?

    // Static factory methods
    /**
     * Create a scalar (rank-0) tensor type.
     * @param elem Element type
     * @return Scalar tensor type
     */
    static TensorType scalar(ElementType elem);

    /**
     * Create a rank-1 (vector) tensor type.
     * @param elem Element type
     * @param size Vector length
     * @return Vector tensor type
     */
    static TensorType vector(ElementType elem, int64_t size);

    /**
     * Create a rank-2 (matrix) tensor type.
     * @param elem Element type
     * @param rows Number of rows
     * @param cols Number of columns
     * @return Matrix tensor type
     */
    static TensorType matrix(ElementType elem, int64_t rows, int64_t cols);

    /**
     * Create a tensor type with an arbitrary shape.
     * @param elem Element type
     * @param shape Dimension sizes
     * @return Tensor type with the given shape
     */
    static TensorType tensor(ElementType elem, std::vector<int64_t> shape);

    /** Sentinel dimension size used to mark a dynamic (runtime-determined) dimension. */
    static constexpr int64_t kDynamic = -1;

    // Queries
    /** Number of dimensions (0 for scalar). */
    size_t rank() const { return shape.size(); }

    /** Total number of elements (product of shape dimensions; 1 for scalar). */
    int64_t numElements() const;

    /** Total storage size in bytes (numElements() * element size). */
    size_t sizeInBytes() const;

    /** True if this is a scalar (rank-0) type. */
    bool isScalar() const { return shape.empty(); }

    /** True if this is a rank-1 (vector) type. */
    bool isVector() const { return shape.size() == 1; }

    /** True if this is a rank-2 (matrix) type. */
    bool isMatrix() const { return shape.size() == 2; }

    // Comparison
    /** Structural equality: same element type and shape. */
    bool operator==(const TensorType& other) const;

    /** Structural inequality. */
    bool operator!=(const TensorType& other) const { return !(*this == other); }

    // String representation
    /** Human-readable representation, e.g. "f64[3,4]". */
    std::string toString() const;
};

/**
 * XLATypes - Type mapping utilities
 *
 * Provides conversion between Eshkol's type system and
 * MLIR/StableHLO types used by XLA.
 */
class XLATypes {
public:
    /**
     * Construct a type mapper with no MLIR context bound.
     * Call setMLIRContext() before using toMLIRType()/toMLIRElementType().
     */
    XLATypes();

    /**
     * Destroy the type mapper.
     */
    ~XLATypes();

    // Non-copyable
    XLATypes(const XLATypes&) = delete;
    XLATypes& operator=(const XLATypes&) = delete;

    // ===== HoTT to XLA =====

    /**
     * Convert HoTT ParameterizedType to XLA TensorType.
     * @param hott_type HoTT type from Eshkol type system
     * @return XLA tensor type
     */
    std::optional<TensorType> fromHoTT(const ParameterizedType& hott_type);

    /**
     * Get element type from HoTT type tag.
     * @param type_tag Type tag (e.g., from tagged value)
     * @return Element type (F64 for numeric, etc.)
     */
    ElementType elementTypeFromTag(uint8_t type_tag);

    // ===== XLA to MLIR =====

    /**
     * Set the MLIR context for type conversion.
     * Must be called before toMLIRType/toMLIRElementType when MLIR is available.
     * @param ctx MLIR context pointer (mlir::MLIRContext* cast to void*)
     */
    void setMLIRContext(void* ctx);

    /**
     * Create MLIR RankedTensorType from XLA TensorType.
     * Requires setMLIRContext() to have been called first.
     * @param xla_type XLA tensor type
     * @return MLIR type pointer (nullptr if MLIR not initialized)
     */
    void* toMLIRType(const TensorType& xla_type);

    /**
     * Create MLIR element type.
     * Requires setMLIRContext() to have been called first.
     * @param elem Element type
     * @return MLIR type pointer (nullptr if MLIR not initialized)
     */
    void* toMLIRElementType(ElementType elem);

    // ===== Utilities =====

    /**
     * Get size in bytes for element type.
     * @param elem Element type
     * @return Size in bytes
     */
    static size_t elementSize(ElementType elem);

    /**
     * Get string name for element type.
     * @param elem Element type
     * @return Type name string
     */
    static std::string elementTypeName(ElementType elem);

    /**
     * Infer broadcast shape for two tensor types.
     * @param lhs Left tensor type
     * @param rhs Right tensor type
     * @return Broadcast result shape (empty if incompatible)
     */
    static std::vector<int64_t> broadcastShape(const TensorType& lhs,
                                                 const TensorType& rhs);

    /**
     * Check if types are broadcast-compatible.
     * @param lhs Left tensor type
     * @param rhs Right tensor type
     * @return true if compatible
     */
    static bool areBroadcastCompatible(const TensorType& lhs,
                                         const TensorType& rhs);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace xla
} // namespace eshkol

#endif // ESHKOL_XLA_TYPES_H
