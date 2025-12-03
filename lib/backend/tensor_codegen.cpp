/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen implementation
 *
 * Note: The complex tensor implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen and autodiff operations.
 * This module provides the interface and will be populated as
 * dependencies are extracted.
 */

#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

TensorCodegen::TensorCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("TensorCodegen initialized");
}

// Note: All tensor implementations are complex and depend on:
// - AST code generation for nested expressions
// - Autodiff integration (dual numbers, AD nodes)
// - Arena memory allocation for results
// - Runtime library functions
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* TensorCodegen::createTensor(const eshkol_ast_t* ast) {
    eshkol_warn("TensorCodegen::createTensor called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorOperation(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorOperation called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorGet(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorGet called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::vectorRef(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::vectorRef called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorSet(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorSet called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorArithmetic(const eshkol_operations_t* op, const std::string& operation) {
    eshkol_warn("TensorCodegen::tensorArithmetic called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorArithmeticInternal(llvm::Value* left, llvm::Value* right, const std::string& operation) {
    eshkol_warn("TensorCodegen::tensorArithmeticInternal called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorDot(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorDot called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorApply(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorApply called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorReduceAll(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorReduceAll called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorReduceWithDim(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorReduceWithDim called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorSum(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorSum called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorMean(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorMean called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::tensorShape(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::tensorShape called - using fallback");
    return tagged_.packNull();
}

llvm::Value* TensorCodegen::transpose(const eshkol_operations_t* op) {
    eshkol_warn("TensorCodegen::transpose called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
