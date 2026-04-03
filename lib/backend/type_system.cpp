/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TypeSystem implementation - LLVM type creation and caching
 */

#include <eshkol/backend/type_system.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <llvm/IR/Constants.h>
#include <vector>

namespace eshkol {

TypeSystem::TypeSystem(llvm::LLVMContext& ctx, bool is_wasm32)
    : context(ctx), is_wasm32_(is_wasm32) {
    // Cache primitive types (avoid repeated lookups)
    int64_type = llvm::Type::getInt64Ty(context);
    int32_type = llvm::Type::getInt32Ty(context);
    int16_type = llvm::Type::getInt16Ty(context);
    int8_type = llvm::Type::getInt8Ty(context);
    int1_type = llvm::Type::getInt1Ty(context);
    double_type = llvm::Type::getDoubleTy(context);
    void_type = llvm::Type::getVoidTy(context);
    ptr_type = llvm::PointerType::getUnqual(context);

    // Target-dependent types: i32 on wasm32, i64 on native 64-bit
    size_type_ = is_wasm32_ ? int32_type : int64_type;
    intptr_type_ = is_wasm32_ ? int32_type : int64_type;

    // Create SIMD vector types for tensor operations
    // These are fixed-size vector types that map to SIMD registers
    double2_type = llvm::FixedVectorType::get(double_type, 2);  // SSE2: 128-bit
    double4_type = llvm::FixedVectorType::get(double_type, 4);  // AVX: 256-bit
    double8_type = llvm::FixedVectorType::get(double_type, 8);  // AVX-512: 512-bit

    // Create struct types
    createStructTypes();
}

void TypeSystem::createStructTypes() {
    // Tagged value struct type (matches C struct exactly):
    // struct eshkol_tagged_value {
    //     uint8_t type;        // offset 0
    //     uint8_t flags;       // offset 1
    //     uint16_t reserved;   // offset 2
    //     // implicit 4 bytes padding for 8-byte alignment of data union
    //     union { int64_t, double, uint64_t } data;  // offset 8
    // } // Total: 16 bytes
    std::vector<llvm::Type*> tagged_value_fields;
    tagged_value_fields.push_back(int8_type);   // Field 0: type (offset 0)
    tagged_value_fields.push_back(int8_type);   // Field 1: flags (offset 1)
    tagged_value_fields.push_back(int16_type);  // Field 2: reserved (offset 2)
    tagged_value_fields.push_back(int32_type);  // Field 3: PADDING for 8-byte alignment (offset 4)
    tagged_value_fields.push_back(int64_type);  // Field 4: data union (offset 8, 8 bytes)
    tagged_value_type = llvm::StructType::create(context, tagged_value_fields, "eshkol_tagged_value");

    // Dual number struct type for forward-mode automatic differentiation
    // struct eshkol_dual_number {
    //     double value;       // f(x)
    //     double derivative;  // f'(x)
    // }
    std::vector<llvm::Type*> dual_fields;
    dual_fields.push_back(double_type);  // value
    dual_fields.push_back(double_type);  // derivative
    dual_number_type = llvm::StructType::create(context, dual_fields, "dual_number");

    // Complex number struct type for signal processing and complex analysis
    // struct eshkol_complex_number {
    //     double real;        // Real component
    //     double imag;        // Imaginary component
    // }
    std::vector<llvm::Type*> complex_fields;
    complex_fields.push_back(double_type);  // real
    complex_fields.push_back(double_type);  // imag
    complex_number_type = llvm::StructType::create(context, complex_fields, "complex_number");

    // AD node struct type for reverse-mode automatic differentiation
    // Matches ad_node_t in eshkol.h (15 fields total)
    // struct ad_node {
    //     int32_t type;          // 0: ad_node_type_t enum
    //     double value;          // 1: computed value (scalar)
    //     double gradient;       // 2: accumulated gradient (scalar)
    //     ad_node* input1;       // 3: first input (or null)
    //     ad_node* input2;       // 4: second input (or null)
    //     size_t id;             // 5: unique node ID
    //     void* tensor_value;    // 6: tensor value ptr (null for scalar)
    //     void* tensor_gradient; // 7: tensor gradient ptr (null for scalar)
    //     ad_node* input3;       // 8: third input (e.g. V in attention)
    //     ad_node* input4;       // 9: fourth input (e.g. mask)
    //     void** saved_tensors;  // 10: array of saved tensors for backward
    //     size_t num_saved;      // 11: number of saved tensors
    //     [6 x i64] params;     // 12: operation params union (conv, attention, etc.)
    //     int64_t* shape;        // 13: output shape
    //     size_t ndim;           // 14: number of dimensions
    // }
    std::vector<llvm::Type*> ad_node_fields;
    ad_node_fields.push_back(int32_type);  // 0: type (enum, 4 bytes)
    ad_node_fields.push_back(double_type); // 1: value
    ad_node_fields.push_back(double_type); // 2: gradient
    ad_node_fields.push_back(ptr_type);    // 3: input1
    ad_node_fields.push_back(ptr_type);    // 4: input2
    ad_node_fields.push_back(int64_type);  // 5: id
    ad_node_fields.push_back(ptr_type);    // 6: tensor_value
    ad_node_fields.push_back(ptr_type);    // 7: tensor_gradient
    ad_node_fields.push_back(ptr_type);    // 8: input3
    ad_node_fields.push_back(ptr_type);    // 9: input4
    ad_node_fields.push_back(ptr_type);    // 10: saved_tensors
    ad_node_fields.push_back(int64_type);  // 11: num_saved
    ad_node_fields.push_back(llvm::ArrayType::get(int64_type, 6)); // 12: params union (6 x i64)
    ad_node_fields.push_back(ptr_type);    // 13: shape
    ad_node_fields.push_back(int64_type);  // 14: ndim
    ad_node_type = llvm::StructType::create(context, ad_node_fields, "ad_node");

    // Tensor struct type for N-dimensional arrays
    // struct tensor {
    //     uint64_t* dimensions;     // array of dimension sizes
    //     uint64_t num_dimensions;  // number of dimensions
    //     double* elements;         // flat array of elements
    //     uint64_t total_elements;  // product of all dimensions
    // }
    std::vector<llvm::Type*> tensor_fields;
    tensor_fields.push_back(ptr_type);     // dimensions
    tensor_fields.push_back(int64_type);   // num_dimensions
    tensor_fields.push_back(ptr_type);     // elements
    tensor_fields.push_back(int64_type);   // total_elements
    tensor_type = llvm::StructType::create(context, tensor_fields, "tensor");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
