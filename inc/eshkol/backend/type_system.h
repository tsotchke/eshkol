/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TypeSystem - LLVM type management for the Eshkol compiler
 *
 * This module encapsulates all LLVM type creation and caching,
 * providing a clean interface for the rest of the codegen system.
 */
#ifndef ESHKOL_BACKEND_TYPE_SYSTEM_H
#define ESHKOL_BACKEND_TYPE_SYSTEM_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Type.h>
#include <llvm/IR/DerivedTypes.h>

namespace eshkol {

/**
 * TypeSystem manages all LLVM types used in the Eshkol compiler.
 *
 * It provides:
 * - Cached primitive types (int64, double, ptr, etc.)
 * - Struct types for runtime values (tagged_value, dual_number, ad_node, tensor)
 * - Type creation helpers
 *
 * Usage:
 *   TypeSystem types(context);
 *   auto int64_t = types.getInt64Type();
 *   auto tagged = types.getTaggedValueType();
 */
class TypeSystem {
public:
    /**
     * Construct a TypeSystem for the given LLVM context.
     * Creates and caches all LLVM types used by the compiler.
     */
    /**
     * @param ctx LLVM context
     * @param is_wasm32 If true, size_t is i32 (wasm32 target). If false, i64 (native 64-bit).
     */
    explicit TypeSystem(llvm::LLVMContext& ctx, bool is_wasm32 = false);

    // Target-dependent size type (i32 on wasm32, i64 on native 64-bit)
    // Use this for: arena sizes, string lengths, array counts, memcpy sizes —
    // anything that maps to C's size_t or a pointer-width integer.
    // Do NOT use for tagged value data fields (always i64).
    llvm::IntegerType* getSizeType() const { return size_type_; }
    bool isWasm32() const { return is_wasm32_; }

    // Primitive types
    llvm::IntegerType* getInt64Type() const { return int64_type; }
    llvm::IntegerType* getInt32Type() const { return int32_type; }
    llvm::IntegerType* getInt16Type() const { return int16_type; }
    llvm::IntegerType* getInt8Type() const { return int8_type; }
    llvm::IntegerType* getInt1Type() const { return int1_type; }
    llvm::Type* getDoubleType() const { return double_type; }
    llvm::Type* getVoidType() const { return void_type; }
    llvm::PointerType* getPtrType() const { return ptr_type; }

    // SIMD vector types for tensor operations
    // SSE2: 128-bit = 2 x double
    llvm::VectorType* getDouble2Type() const { return double2_type; }
    // AVX/AVX2: 256-bit = 4 x double
    llvm::VectorType* getDouble4Type() const { return double4_type; }
    // AVX-512: 512-bit = 8 x double (for future use)
    llvm::VectorType* getDouble8Type() const { return double8_type; }

    // SIMD configuration
    static constexpr unsigned SIMD_SSE_WIDTH = 2;   // 2 doubles (128-bit)
    static constexpr unsigned SIMD_AVX_WIDTH = 4;   // 4 doubles (256-bit)
    static constexpr unsigned SIMD_AVX512_WIDTH = 8; // 8 doubles (512-bit)

    // Struct types for Eshkol runtime
    llvm::StructType* getTaggedValueType() const { return tagged_value_type; }
    llvm::StructType* getDualNumberType() const { return dual_number_type; }
    llvm::StructType* getComplexNumberType() const { return complex_number_type; }
    llvm::StructType* getAdNodeType() const { return ad_node_type; }
    llvm::StructType* getTensorType() const { return tensor_type; }

    // Context accessor (for when raw context is needed)
    llvm::LLVMContext& getContext() { return context; }

    // Tagged value field indices (matching C struct layout)
    static constexpr unsigned TAGGED_VALUE_TYPE_IDX = 0;    // uint8_t type
    static constexpr unsigned TAGGED_VALUE_FLAGS_IDX = 1;   // uint8_t flags
    static constexpr unsigned TAGGED_VALUE_RESERVED_IDX = 2; // uint16_t reserved
    static constexpr unsigned TAGGED_VALUE_PADDING_IDX = 3;  // uint32_t padding
    static constexpr unsigned TAGGED_VALUE_DATA_IDX = 4;     // int64_t/double/ptr data

    // Dual number field indices
    static constexpr unsigned DUAL_VALUE_IDX = 0;      // double value
    static constexpr unsigned DUAL_DERIVATIVE_IDX = 1; // double derivative

    // Complex number field indices
    static constexpr unsigned COMPLEX_REAL_IDX = 0;    // double real
    static constexpr unsigned COMPLEX_IMAG_IDX = 1;    // double imag

    // AD node field indices (matches ad_node_t in eshkol.h)
    // Base fields (scalar AD)
    static constexpr unsigned AD_NODE_TYPE_IDX = 0;    // int32_t type (enum)
    static constexpr unsigned AD_NODE_VALUE_IDX = 1;   // double value
    static constexpr unsigned AD_NODE_GRADIENT_IDX = 2; // double gradient
    static constexpr unsigned AD_NODE_INPUT1_IDX = 3;  // ad_node* input1
    static constexpr unsigned AD_NODE_INPUT2_IDX = 4;  // ad_node* input2
    static constexpr unsigned AD_NODE_ID_IDX = 5;      // size_t id
    // Extended fields (tensor AD)
    static constexpr unsigned AD_NODE_TENSOR_VALUE_IDX = 6;    // void* tensor_value
    static constexpr unsigned AD_NODE_TENSOR_GRADIENT_IDX = 7;  // void* tensor_gradient
    static constexpr unsigned AD_NODE_INPUT3_IDX = 8;           // ad_node* input3
    static constexpr unsigned AD_NODE_INPUT4_IDX = 9;           // ad_node* input4
    static constexpr unsigned AD_NODE_SAVED_TENSORS_IDX = 10;   // void** saved_tensors
    static constexpr unsigned AD_NODE_NUM_SAVED_IDX = 11;       // size_t num_saved
    static constexpr unsigned AD_NODE_PARAMS_IDX = 12;          // [6 x i64] params union
    static constexpr unsigned AD_NODE_SHAPE_IDX = 13;           // int64_t* shape
    static constexpr unsigned AD_NODE_NDIM_IDX = 14;            // size_t ndim

    // Tensor field indices
    static constexpr unsigned TENSOR_DIMENSIONS_IDX = 0;     // uint64_t* dimensions
    static constexpr unsigned TENSOR_NUM_DIMS_IDX = 1;       // uint64_t num_dimensions
    static constexpr unsigned TENSOR_ELEMENTS_IDX = 2;       // double* elements
    static constexpr unsigned TENSOR_TOTAL_ELEMENTS_IDX = 3; // uint64_t total_elements

private:
    llvm::LLVMContext& context;
    bool is_wasm32_;
    llvm::IntegerType* size_type_;  // i32 on wasm32, i64 on native

    // Cached primitive types
    llvm::IntegerType* int64_type;
    llvm::IntegerType* int32_type;
    llvm::IntegerType* int16_type;
    llvm::IntegerType* int8_type;
    llvm::IntegerType* int1_type;
    llvm::Type* double_type;
    llvm::Type* void_type;
    llvm::PointerType* ptr_type;

    // SIMD vector types
    llvm::VectorType* double2_type;  // <2 x double> for SSE2
    llvm::VectorType* double4_type;  // <4 x double> for AVX
    llvm::VectorType* double8_type;  // <8 x double> for AVX-512

    // Cached struct types
    llvm::StructType* tagged_value_type;
    llvm::StructType* dual_number_type;
    llvm::StructType* complex_number_type;
    llvm::StructType* ad_node_type;
    llvm::StructType* tensor_type;

    // Private helper to create struct types
    void createStructTypes();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TYPE_SYSTEM_H
