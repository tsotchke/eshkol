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
    explicit TypeSystem(llvm::LLVMContext& ctx);

    // Primitive types
    llvm::IntegerType* getInt64Type() const { return int64_type; }
    llvm::IntegerType* getInt32Type() const { return int32_type; }
    llvm::IntegerType* getInt16Type() const { return int16_type; }
    llvm::IntegerType* getInt8Type() const { return int8_type; }
    llvm::IntegerType* getInt1Type() const { return int1_type; }
    llvm::Type* getDoubleType() const { return double_type; }
    llvm::Type* getVoidType() const { return void_type; }
    llvm::PointerType* getPtrType() const { return ptr_type; }

    // Struct types for Eshkol runtime
    llvm::StructType* getTaggedValueType() const { return tagged_value_type; }
    llvm::StructType* getDualNumberType() const { return dual_number_type; }
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

    // AD node field indices
    static constexpr unsigned AD_NODE_TYPE_IDX = 0;    // int32_t type (enum)
    static constexpr unsigned AD_NODE_VALUE_IDX = 1;   // double value
    static constexpr unsigned AD_NODE_GRADIENT_IDX = 2; // double gradient
    static constexpr unsigned AD_NODE_INPUT1_IDX = 3;  // ad_node* input1
    static constexpr unsigned AD_NODE_INPUT2_IDX = 4;  // ad_node* input2
    static constexpr unsigned AD_NODE_ID_IDX = 5;      // size_t id

    // Tensor field indices
    static constexpr unsigned TENSOR_DIMENSIONS_IDX = 0;     // uint64_t* dimensions
    static constexpr unsigned TENSOR_NUM_DIMS_IDX = 1;       // uint64_t num_dimensions
    static constexpr unsigned TENSOR_ELEMENTS_IDX = 2;       // double* elements
    static constexpr unsigned TENSOR_TOTAL_ELEMENTS_IDX = 3; // uint64_t total_elements

private:
    llvm::LLVMContext& context;

    // Cached primitive types
    llvm::IntegerType* int64_type;
    llvm::IntegerType* int32_type;
    llvm::IntegerType* int16_type;
    llvm::IntegerType* int8_type;
    llvm::IntegerType* int1_type;
    llvm::Type* double_type;
    llvm::Type* void_type;
    llvm::PointerType* ptr_type;

    // Cached struct types
    llvm::StructType* tagged_value_type;
    llvm::StructType* dual_number_type;
    llvm::StructType* ad_node_type;
    llvm::StructType* tensor_type;

    // Private helper to create struct types
    void createStructTypes();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TYPE_SYSTEM_H
