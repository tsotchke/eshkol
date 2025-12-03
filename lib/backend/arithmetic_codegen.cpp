/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ArithmeticCodegen implementation
 *
 * Note: The full polymorphic arithmetic implementations remain in llvm_codegen.cpp
 * for now due to dependencies on tensor and AD operations. This module provides
 * the interface and will be populated as those dependencies are extracted.
 */

#include <eshkol/backend/arithmetic_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>

namespace eshkol {

ArithmeticCodegen::ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged) {
    eshkol_debug("ArithmeticCodegen initialized");
}

// Note: The polymorphic add/sub/mul/div implementations are complex and depend on:
// - codegenTensorArithmeticInternal (tensor operations)
// - createADConstant, createADNode (autodiff operations)
// - dualNumberAdd/Sub/Mul/Div (forward-mode AD)
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.
// The functions below provide simple scalar fallbacks for basic operations.

llvm::Value* ArithmeticCodegen::add(llvm::Value* left, llvm::Value* right) {
    // Stub - actual implementation in llvm_codegen.cpp::polymorphicAdd
    // This will be populated when tensor/AD modules are extracted
    eshkol_warn("ArithmeticCodegen::add called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ArithmeticCodegen::sub(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("ArithmeticCodegen::sub called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ArithmeticCodegen::mul(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("ArithmeticCodegen::mul called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ArithmeticCodegen::div(llvm::Value* left, llvm::Value* right) {
    eshkol_warn("ArithmeticCodegen::div called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ArithmeticCodegen::mod(llvm::Value* left, llvm::Value* right) {
    // Integer modulo - simpler implementation
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    llvm::Value* result = ctx_.builder().CreateSRem(left_int, right_int, "mod_result");
    return tagged_.packInt64(result, true);
}

llvm::Value* ArithmeticCodegen::neg(llvm::Value* operand) {
    // Check type and negate appropriately
    llvm::Value* type_tag = tagged_.getType(operand);
    llvm::Value* base_type = ctx_.builder().CreateAnd(type_tag,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* double_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_double", func);
    llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_int", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "neg_merge", func);

    ctx_.builder().CreateCondBr(is_double, double_bb, int_bb);

    // Double negation
    ctx_.builder().SetInsertPoint(double_bb);
    llvm::Value* dbl_val = tagged_.unpackDouble(operand);
    llvm::Value* neg_dbl = ctx_.builder().CreateFNeg(dbl_val, "neg_double");
    llvm::Value* dbl_result = tagged_.packDouble(neg_dbl);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

    // Integer negation
    ctx_.builder().SetInsertPoint(int_bb);
    llvm::Value* int_val = tagged_.unpackInt64(operand);
    llvm::Value* neg_int = ctx_.builder().CreateNeg(int_val, "neg_int");
    llvm::Value* int_result = tagged_.packInt64(neg_int, true);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "neg_result");
    phi->addIncoming(dbl_result, double_exit);
    phi->addIncoming(int_result, int_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::abs(llvm::Value* operand) {
    llvm::Value* type_tag = tagged_.getType(operand);
    llvm::Value* base_type = ctx_.builder().CreateAnd(type_tag,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* double_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_double", func);
    llvm::BasicBlock* int_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_int", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "abs_merge", func);

    ctx_.builder().CreateCondBr(is_double, double_bb, int_bb);

    // Double abs using fabs intrinsic
    ctx_.builder().SetInsertPoint(double_bb);
    llvm::Value* dbl_val = tagged_.unpackDouble(operand);
    // Use fneg + select for abs since we may not have fabs intrinsic set up
    llvm::Value* neg_dbl = ctx_.builder().CreateFNeg(dbl_val);
    llvm::Value* is_neg = ctx_.builder().CreateFCmpOLT(dbl_val,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    llvm::Value* abs_dbl = ctx_.builder().CreateSelect(is_neg, neg_dbl, dbl_val, "abs_double");
    llvm::Value* dbl_result = tagged_.packDouble(abs_dbl);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* double_exit = ctx_.builder().GetInsertBlock();

    // Integer abs
    ctx_.builder().SetInsertPoint(int_bb);
    llvm::Value* int_val = tagged_.unpackInt64(operand);
    llvm::Value* neg_int = ctx_.builder().CreateNeg(int_val);
    llvm::Value* is_neg_int = ctx_.builder().CreateICmpSLT(int_val,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* abs_int = ctx_.builder().CreateSelect(is_neg_int, neg_int, int_val, "abs_int");
    llvm::Value* int_result = tagged_.packInt64(abs_int, true);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* int_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "abs_result");
    phi->addIncoming(dbl_result, double_exit);
    phi->addIncoming(int_result, int_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::intToDouble(llvm::Value* int_tagged) {
    llvm::Value* int_val = tagged_.unpackInt64(int_tagged);
    llvm::Value* dbl_val = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType(), "int_to_double");
    return tagged_.packDouble(dbl_val);
}

llvm::Value* ArithmeticCodegen::doubleToInt(llvm::Value* double_tagged) {
    llvm::Value* dbl_val = tagged_.unpackDouble(double_tagged);
    llvm::Value* int_val = ctx_.builder().CreateFPToSI(dbl_val, ctx_.int64Type(), "double_to_int");
    return tagged_.packInt64(int_val, true);
}

llvm::Value* ArithmeticCodegen::extractAsDouble(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    // Handle raw double - return as-is
    if (tagged_val->getType()->isDoubleTy()) return tagged_val;

    // Handle raw int64 - convert to double
    if (tagged_val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateSIToFP(tagged_val, ctx_.doubleType());
    }

    // Handle tagged value
    llvm::Value* type_tag = tagged_.getType(tagged_val);
    llvm::Value* base_type = ctx_.builder().CreateAnd(type_tag,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
    llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
    llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());

    return ctx_.builder().CreateSelect(is_double, dbl_val, int_as_dbl, "as_double");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
