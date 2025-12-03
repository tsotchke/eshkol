/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * AutodiffCodegen implementation
 *
 * Note: The complex autodiff implementations remain in llvm_codegen.cpp
 * for now due to extensive dependencies on AST codegen, tape management,
 * and runtime library functions. This module provides the interface.
 */

#include <eshkol/backend/autodiff_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

AutodiffCodegen::AutodiffCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("AutodiffCodegen initialized");
}

// Note: All autodiff implementations are complex and depend on:
// - AST code generation for function bodies
// - Tape management for reverse-mode AD
// - Runtime library functions for backpropagation
// - Complex control flow for gradient/jacobian computation
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* AutodiffCodegen::createDualNumber(llvm::Value* primal, llvm::Value* tangent) {
    eshkol_warn("AutodiffCodegen::createDualNumber called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::getDualPrimal(llvm::Value* dual) {
    eshkol_warn("AutodiffCodegen::getDualPrimal called - using fallback");
    return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
}

llvm::Value* AutodiffCodegen::getDualTangent(llvm::Value* dual) {
    eshkol_warn("AutodiffCodegen::getDualTangent called - using fallback");
    return llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
}

llvm::Value* AutodiffCodegen::dualAdd(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("AutodiffCodegen::dualAdd called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::dualSub(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("AutodiffCodegen::dualSub called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::dualMul(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("AutodiffCodegen::dualMul called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::dualDiv(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("AutodiffCodegen::dualDiv called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::recordADNodeBinary(uint32_t op_type, llvm::Value* left, llvm::Value* right) {
    eshkol_warn("AutodiffCodegen::recordADNodeBinary called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::recordADNodeUnary(uint32_t op_type, llvm::Value* input) {
    eshkol_warn("AutodiffCodegen::recordADNodeUnary called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::createADConstant(llvm::Value* value) {
    eshkol_warn("AutodiffCodegen::createADConstant called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::gradient(const eshkol_operations_t* op) {
    eshkol_warn("AutodiffCodegen::gradient called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::jacobian(const eshkol_operations_t* op) {
    eshkol_warn("AutodiffCodegen::jacobian called - using fallback");
    return tagged_.packNull();
}

llvm::Value* AutodiffCodegen::createTape() {
    eshkol_warn("AutodiffCodegen::createTape called - using fallback");
    return llvm::ConstantPointerNull::get(ctx_.ptrType());
}

void AutodiffCodegen::backpropagate(llvm::Value* tape, llvm::Value* output_node) {
    eshkol_warn("AutodiffCodegen::backpropagate called - using fallback");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
