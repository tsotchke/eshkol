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
#include <eshkol/logger.h>
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

    // Dual number struct type for forward-mode automatic differentiation.
    //
    // SECOND-ORDER / NESTED AD: the dual carries two independent
    // perturbation slots plus their mixed (cross) term, encoding a truncated
    // bivariate Taylor expansion
    //   v = primal + d1*e1 + d2*e2 + d12*e1*e2   (with e1^2 = e2^2 = 0)
    // This is what makes 2-level nested `derivative` and EXACT Hessians
    // possible: perturbation tagging maps nesting depth -> slot index.
    // The first two fields (primal, d1) are UNCHANGED, so all single-level
    // forward-mode AD and the C runtime view of the struct (which reads only
    // value/derivative at offsets 0 and 8) keep working bit-compatibly; the
    // d2/d12 slots default to 0 for first-order use.
    // struct eshkol_dual_number {
    //     double value;       // f0: primal  f(x)
    //     double derivative;  // f1: d/de1   (single-level tangent)
    //     double d2;          // f2: d/de2   (second perturbation slot)
    //     double d12;         // f3: d2/de1 de2 (mixed second-order term)
    //     // ESH-0117: parallel "reverse-seed derivative" 4-jet dp = d(above)/dep,
    //     // where ep is the infinitesimal of the reverse-mode gradient's active
    //     // seed variable (published via eshkol_ad_seed_swap). This lets a nested
    //     // forward derivative running INSIDE a reverse gradient carry the
    //     // dependence of ALL its jet coefficients on the captured reverse
    //     // variable — even at 2-level forward nesting where e1/e2 are both used
    //     // by the forward perturbations, leaving no free jet slot for ep.
    //     double dp;          // f4: d(value)/dep
    //     double dp1;         // f5: d(d/de1)/dep
    //     double dp2;         // f6: d(d/de2)/dep
    //     double dp12;        // f7: d(d2/de1 de2)/dep   (the e1 e2 ep triple term)
    // }
    // Fields 0-3 are UNCHANGED (all first/second-order forward AD, the C runtime
    // view at offsets 0/8, and every consumer reading value/derivative keep
    // working bit-compatibly). Fields 4-7 default to 0 for non-mixed-mode use.
    std::vector<llvm::Type*> dual_fields;
    dual_fields.push_back(double_type);  // value  (primal)
    dual_fields.push_back(double_type);  // derivative (slot e1)
    dual_fields.push_back(double_type);  // slot e2
    dual_fields.push_back(double_type);  // mixed e1*e2
    dual_fields.push_back(double_type);  // dp    : d(value)/dep
    dual_fields.push_back(double_type);  // dp1   : d(e1)/dep
    dual_fields.push_back(double_type);  // dp2   : d(e2)/dep
    dual_fields.push_back(double_type);  // dp12  : d(e1e2)/dep
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
    // Verify field count matches the constant defined in TypeSystem
    if (ad_node_fields.size() != AD_NODE_FIELD_COUNT) {
        eshkol_error("AD node struct has %zu fields but AD_NODE_FIELD_COUNT is %u",
                     ad_node_fields.size(), AD_NODE_FIELD_COUNT);
    }

    // Tensor struct type for N-dimensional arrays
    // struct tensor {
    //     uint64_t* dimensions;     // array of dimension sizes
    //     uint64_t num_dimensions;  // number of dimensions
    //     double* elements;         // flat array of elements
    //     uint64_t total_elements;  // product of all dimensions
    // }
    std::vector<llvm::Type*> tensor_fields;
    tensor_fields.push_back(ptr_type);     // idx 0: dimensions
    tensor_fields.push_back(int64_type);   // idx 1: num_dimensions
    tensor_fields.push_back(ptr_type);     // idx 2: elements
    tensor_fields.push_back(int64_type);   // idx 3: total_elements
    tensor_fields.push_back(int64_type);   // idx 4: dtype (eshkol_tensor_dtype_t)
    tensor_type = llvm::StructType::create(context, tensor_fields, "tensor");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
