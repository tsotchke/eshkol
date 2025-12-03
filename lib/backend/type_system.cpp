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

TypeSystem::TypeSystem(llvm::LLVMContext& ctx) : context(ctx) {
    // Cache primitive types (avoid repeated lookups)
    int64_type = llvm::Type::getInt64Ty(context);
    int32_type = llvm::Type::getInt32Ty(context);
    int16_type = llvm::Type::getInt16Ty(context);
    int8_type = llvm::Type::getInt8Ty(context);
    int1_type = llvm::Type::getInt1Ty(context);
    double_type = llvm::Type::getDoubleTy(context);
    void_type = llvm::Type::getVoidTy(context);
    ptr_type = llvm::PointerType::getUnqual(context);

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

    // AD node struct type for reverse-mode automatic differentiation
    // struct ad_node {
    //     int32_t type;       // ad_node_type_t enum
    //     double value;       // computed value
    //     double gradient;    // accumulated gradient
    //     ad_node* input1;    // first input (or null)
    //     ad_node* input2;    // second input (or null)
    //     size_t id;          // unique node ID
    // }
    std::vector<llvm::Type*> ad_node_fields;
    ad_node_fields.push_back(int32_type);  // type (enum, 4 bytes)
    ad_node_fields.push_back(double_type); // value
    ad_node_fields.push_back(double_type); // gradient
    ad_node_fields.push_back(ptr_type);    // input1
    ad_node_fields.push_back(ptr_type);    // input2
    ad_node_fields.push_back(int64_type);  // id
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
