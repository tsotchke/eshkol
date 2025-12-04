/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ArithmeticCodegen implementation
 *
 * This module implements fully polymorphic arithmetic operations that handle:
 * - Integer (exact) and floating-point (inexact) numbers
 * - Dual numbers for forward-mode automatic differentiation
 * - AD nodes for reverse-mode automatic differentiation (computational graphs)
 * - Vectors and tensors for element-wise operations
 */

#include <eshkol/backend/arithmetic_codegen.h>
#include <eshkol/eshkol.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>

namespace eshkol {

ArithmeticCodegen::ArithmeticCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged,
                                     TensorCodegen& tensor, AutodiffCodegen& autodiff)
    : ctx_(ctx)
    , tagged_(tagged)
    , tensor_(tensor)
    , autodiff_(autodiff) {
    eshkol_debug("ArithmeticCodegen initialized with all dependencies");
}

// === Helper Functions ===

llvm::Value* ArithmeticCodegen::convertToDual(llvm::Value* operand, llvm::Value* is_dual,
                                               llvm::Value* is_double) {
    // Create blocks for branching
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_dual_bb = llvm::BasicBlock::Create(ctx_.context(), "is_dual", func);
    llvm::BasicBlock* not_dual_bb = llvm::BasicBlock::Create(ctx_.context(), "not_dual", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "dual_merge", func);

    ctx_.builder().CreateCondBr(is_dual, is_dual_bb, not_dual_bb);

    // Already a dual number - unpack it
    ctx_.builder().SetInsertPoint(is_dual_bb);
    llvm::Value* dual_value = autodiff_.unpackDualFromTagged(operand);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* is_dual_exit = ctx_.builder().GetInsertBlock();

    // Not a dual number - convert to dual with zero tangent
    ctx_.builder().SetInsertPoint(not_dual_bb);
    llvm::Value* as_double = ctx_.builder().CreateSelect(is_double,
        tagged_.unpackDouble(operand),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(operand), ctx_.doubleType()));
    llvm::Value* non_dual = autodiff_.createDualNumber(as_double,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_dual_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.dualNumberType(), 2, "dual_phi");
    phi->addIncoming(dual_value, is_dual_exit);
    phi->addIncoming(non_dual, not_dual_exit);

    return phi;
}

llvm::Value* ArithmeticCodegen::convertToADNode(llvm::Value* operand, llvm::Value* is_ad,
                                                 llvm::Value* base_type) {
    // Create blocks for branching
    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* is_ad_bb = llvm::BasicBlock::Create(ctx_.context(), "is_ad", func);
    llvm::BasicBlock* not_ad_bb = llvm::BasicBlock::Create(ctx_.context(), "not_ad", func);
    llvm::BasicBlock* merge_bb = llvm::BasicBlock::Create(ctx_.context(), "ad_merge", func);

    ctx_.builder().CreateCondBr(is_ad, is_ad_bb, not_ad_bb);

    // Already an AD node - unpack pointer
    ctx_.builder().SetInsertPoint(is_ad_bb);
    llvm::Value* ad_ptr = tagged_.unpackPtr(operand);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* is_ad_exit = ctx_.builder().GetInsertBlock();

    // Not an AD node - create constant node
    ctx_.builder().SetInsertPoint(not_ad_bb);
    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* val = ctx_.builder().CreateSelect(is_double,
        tagged_.unpackDouble(operand),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(operand), ctx_.doubleType()));
    llvm::Value* ad_const = autodiff_.createADConstant(val);
    ctx_.builder().CreateBr(merge_bb);
    llvm::BasicBlock* not_ad_exit = ctx_.builder().GetInsertBlock();

    // Merge
    ctx_.builder().SetInsertPoint(merge_bb);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "ad_phi");
    phi->addIncoming(ad_ptr, is_ad_exit);
    phi->addIncoming(ad_const, not_ad_exit);

    return phi;
}

// === Polymorphic Addition ===

llvm::Value* ArithmeticCodegen::add(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type tags
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);

    llvm::Value* left_base = ctx_.builder().CreateAnd(left_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));
    llvm::Value* right_base = ctx_.builder().CreateAnd(right_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    // Check for vector/tensor types
    llvm::Value* left_is_vector = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* right_is_vector = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* left_is_tensor = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* right_is_tensor = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* any_vector = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(left_is_vector, right_is_vector),
        ctx_.builder().CreateOr(left_is_tensor, right_is_tensor));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "add_vector", func);
    llvm::BasicBlock* check_ad = llvm::BasicBlock::Create(ctx_.context(), "add_check_ad", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "add_ad", func);
    llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "add_check_dual", func);
    llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "add_dual", func);
    llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "add_check_double", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "add_double", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "add_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "add_merge", func);

    ctx_.builder().CreateCondBr(any_vector, vector_path, check_ad);

    // Vector/tensor path
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "add");
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

    // Check for AD nodes
    ctx_.builder().SetInsertPoint(check_ad);
    llvm::Value* left_is_ad = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* right_is_ad = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* any_ad = ctx_.builder().CreateOr(left_is_ad, right_is_ad);
    ctx_.builder().CreateCondBr(any_ad, ad_path, check_dual);

    // AD node path
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* left_ad = convertToADNode(left, left_is_ad, left_base);
    llvm::Value* right_ad = convertToADNode(right, right_is_ad, right_base);
    llvm::Value* ad_result_node = autodiff_.recordADNodeBinary(2, left_ad, right_ad); // AD_NODE_ADD = 2
    llvm::Value* ad_result = tagged_.packPtr(ad_result_node, ESHKOL_VALUE_AD_NODE_PTR);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Check for dual numbers
    ctx_.builder().SetInsertPoint(check_dual);
    llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
    ctx_.builder().CreateCondBr(any_dual, dual_path, check_double);

    // Dual number path
    ctx_.builder().SetInsertPoint(dual_path);
    llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
    llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
    llvm::Value* dual_result = autodiff_.dualAdd(left_dual, right_dual);
    llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

    // Check for doubles
    ctx_.builder().SetInsertPoint(check_double);
    llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
    ctx_.builder().CreateCondBr(any_double, double_path, int_path);

    // Double path
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
        tagged_.unpackDouble(left),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
    llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
        tagged_.unpackDouble(right),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
    llvm::Value* dbl_result = ctx_.builder().CreateFAdd(left_dbl, right_dbl);
    llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
    ctx_.builder().CreateBr(merge);

    // Integer path
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    llvm::Value* int_result = ctx_.builder().CreateAdd(left_int, right_int);
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);

    // Merge all paths
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5, "add_result");
    phi->addIncoming(vec_result, vector_exit);
    phi->addIncoming(ad_result, ad_exit);
    phi->addIncoming(dual_tagged, dual_exit);
    phi->addIncoming(dbl_tagged, double_path);
    phi->addIncoming(int_tagged, int_path);

    return phi;
}

// === Polymorphic Subtraction ===

llvm::Value* ArithmeticCodegen::sub(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type tags
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);

    llvm::Value* left_base = ctx_.builder().CreateAnd(left_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));
    llvm::Value* right_base = ctx_.builder().CreateAnd(right_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    // Check for vector/tensor types
    llvm::Value* left_is_vector = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* right_is_vector = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* left_is_tensor = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* right_is_tensor = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* any_vector = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(left_is_vector, right_is_vector),
        ctx_.builder().CreateOr(left_is_tensor, right_is_tensor));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "sub_vector", func);
    llvm::BasicBlock* check_ad = llvm::BasicBlock::Create(ctx_.context(), "sub_check_ad", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "sub_ad", func);
    llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "sub_check_dual", func);
    llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "sub_dual", func);
    llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "sub_check_double", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "sub_double", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "sub_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "sub_merge", func);

    ctx_.builder().CreateCondBr(any_vector, vector_path, check_ad);

    // Vector/tensor path
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "sub");
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

    // Check for AD nodes
    ctx_.builder().SetInsertPoint(check_ad);
    llvm::Value* left_is_ad = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* right_is_ad = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* any_ad = ctx_.builder().CreateOr(left_is_ad, right_is_ad);
    ctx_.builder().CreateCondBr(any_ad, ad_path, check_dual);

    // AD node path
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* left_ad = convertToADNode(left, left_is_ad, left_base);
    llvm::Value* right_ad = convertToADNode(right, right_is_ad, right_base);
    llvm::Value* ad_result_node = autodiff_.recordADNodeBinary(3, left_ad, right_ad); // AD_NODE_SUB = 3
    llvm::Value* ad_result = tagged_.packPtr(ad_result_node, ESHKOL_VALUE_AD_NODE_PTR);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Check for dual numbers
    ctx_.builder().SetInsertPoint(check_dual);
    llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
    ctx_.builder().CreateCondBr(any_dual, dual_path, check_double);

    // Dual number path
    ctx_.builder().SetInsertPoint(dual_path);
    llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
    llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
    llvm::Value* dual_result = autodiff_.dualSub(left_dual, right_dual);
    llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

    // Check for doubles
    ctx_.builder().SetInsertPoint(check_double);
    llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
    ctx_.builder().CreateCondBr(any_double, double_path, int_path);

    // Double path
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
        tagged_.unpackDouble(left),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
    llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
        tagged_.unpackDouble(right),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
    llvm::Value* dbl_result = ctx_.builder().CreateFSub(left_dbl, right_dbl);
    llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
    ctx_.builder().CreateBr(merge);

    // Integer path
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    llvm::Value* int_result = ctx_.builder().CreateSub(left_int, right_int);
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);

    // Merge all paths
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5, "sub_result");
    phi->addIncoming(vec_result, vector_exit);
    phi->addIncoming(ad_result, ad_exit);
    phi->addIncoming(dual_tagged, dual_exit);
    phi->addIncoming(dbl_tagged, double_path);
    phi->addIncoming(int_tagged, int_path);

    return phi;
}

// === Polymorphic Multiplication ===

llvm::Value* ArithmeticCodegen::mul(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type tags
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);

    llvm::Value* left_base = ctx_.builder().CreateAnd(left_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));
    llvm::Value* right_base = ctx_.builder().CreateAnd(right_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    // Check for vector/tensor types
    llvm::Value* left_is_vector = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* right_is_vector = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* left_is_tensor = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* right_is_tensor = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* any_vector = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(left_is_vector, right_is_vector),
        ctx_.builder().CreateOr(left_is_tensor, right_is_tensor));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "mul_vector", func);
    llvm::BasicBlock* check_ad = llvm::BasicBlock::Create(ctx_.context(), "mul_check_ad", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "mul_ad", func);
    llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "mul_check_dual", func);
    llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "mul_dual", func);
    llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "mul_check_double", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "mul_double", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "mul_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "mul_merge", func);

    ctx_.builder().CreateCondBr(any_vector, vector_path, check_ad);

    // Vector/tensor path
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "mul");
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

    // Check for AD nodes
    ctx_.builder().SetInsertPoint(check_ad);
    llvm::Value* left_is_ad = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* right_is_ad = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* any_ad = ctx_.builder().CreateOr(left_is_ad, right_is_ad);
    ctx_.builder().CreateCondBr(any_ad, ad_path, check_dual);

    // AD node path
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* left_ad = convertToADNode(left, left_is_ad, left_base);
    llvm::Value* right_ad = convertToADNode(right, right_is_ad, right_base);
    llvm::Value* ad_result_node = autodiff_.recordADNodeBinary(4, left_ad, right_ad); // AD_NODE_MUL = 4
    llvm::Value* ad_result = tagged_.packPtr(ad_result_node, ESHKOL_VALUE_AD_NODE_PTR);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Check for dual numbers
    ctx_.builder().SetInsertPoint(check_dual);
    llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
    ctx_.builder().CreateCondBr(any_dual, dual_path, check_double);

    // Dual number path
    ctx_.builder().SetInsertPoint(dual_path);
    llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
    llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
    llvm::Value* dual_result = autodiff_.dualMul(left_dual, right_dual);
    llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

    // Check for doubles
    ctx_.builder().SetInsertPoint(check_double);
    llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
    ctx_.builder().CreateCondBr(any_double, double_path, int_path);

    // Double path
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
        tagged_.unpackDouble(left),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
    llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
        tagged_.unpackDouble(right),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
    llvm::Value* dbl_result = ctx_.builder().CreateFMul(left_dbl, right_dbl);
    llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
    ctx_.builder().CreateBr(merge);

    // Integer path
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    llvm::Value* int_result = ctx_.builder().CreateMul(left_int, right_int);
    llvm::Value* int_tagged = tagged_.packInt64(int_result, true);
    ctx_.builder().CreateBr(merge);

    // Merge all paths
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5, "mul_result");
    phi->addIncoming(vec_result, vector_exit);
    phi->addIncoming(ad_result, ad_exit);
    phi->addIncoming(dual_tagged, dual_exit);
    phi->addIncoming(dbl_tagged, double_path);
    phi->addIncoming(int_tagged, int_path);

    return phi;
}

// === Polymorphic Division ===

llvm::Value* ArithmeticCodegen::div(llvm::Value* left, llvm::Value* right) {
    if (!left || !right) {
        return tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
    }

    // Extract type tags
    llvm::Value* left_type = tagged_.getType(left);
    llvm::Value* right_type = tagged_.getType(right);

    llvm::Value* left_base = ctx_.builder().CreateAnd(left_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));
    llvm::Value* right_base = ctx_.builder().CreateAnd(right_type,
        llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

    // Check for vector/tensor types
    llvm::Value* left_is_vector = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* right_is_vector = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_VECTOR_PTR));
    llvm::Value* left_is_tensor = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* right_is_tensor = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_TENSOR_PTR));
    llvm::Value* any_vector = ctx_.builder().CreateOr(
        ctx_.builder().CreateOr(left_is_vector, right_is_vector),
        ctx_.builder().CreateOr(left_is_tensor, right_is_tensor));

    llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* vector_path = llvm::BasicBlock::Create(ctx_.context(), "div_vector", func);
    llvm::BasicBlock* check_ad = llvm::BasicBlock::Create(ctx_.context(), "div_check_ad", func);
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "div_ad", func);
    llvm::BasicBlock* check_dual = llvm::BasicBlock::Create(ctx_.context(), "div_check_dual", func);
    llvm::BasicBlock* dual_path = llvm::BasicBlock::Create(ctx_.context(), "div_dual", func);
    llvm::BasicBlock* check_double = llvm::BasicBlock::Create(ctx_.context(), "div_check_double", func);
    llvm::BasicBlock* double_path = llvm::BasicBlock::Create(ctx_.context(), "div_double", func);
    llvm::BasicBlock* int_path = llvm::BasicBlock::Create(ctx_.context(), "div_int", func);
    llvm::BasicBlock* merge = llvm::BasicBlock::Create(ctx_.context(), "div_merge", func);

    ctx_.builder().CreateCondBr(any_vector, vector_path, check_ad);

    // Vector/tensor path
    ctx_.builder().SetInsertPoint(vector_path);
    llvm::Value* vec_result = tensor_.tensorArithmeticInternal(left, right, "div");
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* vector_exit = ctx_.builder().GetInsertBlock();

    // Check for AD nodes
    ctx_.builder().SetInsertPoint(check_ad);
    llvm::Value* left_is_ad = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* right_is_ad = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_AD_NODE_PTR));
    llvm::Value* any_ad = ctx_.builder().CreateOr(left_is_ad, right_is_ad);
    ctx_.builder().CreateCondBr(any_ad, ad_path, check_dual);

    // AD node path
    ctx_.builder().SetInsertPoint(ad_path);
    llvm::Value* left_ad = convertToADNode(left, left_is_ad, left_base);
    llvm::Value* right_ad = convertToADNode(right, right_is_ad, right_base);
    llvm::Value* ad_result_node = autodiff_.recordADNodeBinary(5, left_ad, right_ad); // AD_NODE_DIV = 5
    llvm::Value* ad_result = tagged_.packPtr(ad_result_node, ESHKOL_VALUE_AD_NODE_PTR);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* ad_exit = ctx_.builder().GetInsertBlock();

    // Check for dual numbers
    ctx_.builder().SetInsertPoint(check_dual);
    llvm::Value* left_is_dual = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* right_is_dual = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::Value* any_dual = ctx_.builder().CreateOr(left_is_dual, right_is_dual);
    ctx_.builder().CreateCondBr(any_dual, dual_path, check_double);

    // Dual number path
    ctx_.builder().SetInsertPoint(dual_path);
    llvm::Value* left_is_double = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_double = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* left_dual = convertToDual(left, left_is_dual, left_is_double);
    llvm::Value* right_dual = convertToDual(right, right_is_dual, right_is_double);
    llvm::Value* dual_result = autodiff_.dualDiv(left_dual, right_dual);
    llvm::Value* dual_tagged = autodiff_.packDualToTagged(dual_result);
    ctx_.builder().CreateBr(merge);
    llvm::BasicBlock* dual_exit = ctx_.builder().GetInsertBlock();

    // Check for doubles
    ctx_.builder().SetInsertPoint(check_double);
    llvm::Value* left_is_dbl = ctx_.builder().CreateICmpEQ(left_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* right_is_dbl = ctx_.builder().CreateICmpEQ(right_base,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
    llvm::Value* any_double = ctx_.builder().CreateOr(left_is_dbl, right_is_dbl);
    ctx_.builder().CreateCondBr(any_double, double_path, int_path);

    // Double path
    ctx_.builder().SetInsertPoint(double_path);
    llvm::Value* left_dbl = ctx_.builder().CreateSelect(left_is_dbl,
        tagged_.unpackDouble(left),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(left), ctx_.doubleType()));
    llvm::Value* right_dbl = ctx_.builder().CreateSelect(right_is_dbl,
        tagged_.unpackDouble(right),
        ctx_.builder().CreateSIToFP(tagged_.unpackInt64(right), ctx_.doubleType()));
    llvm::Value* dbl_result = ctx_.builder().CreateFDiv(left_dbl, right_dbl);
    llvm::Value* dbl_tagged = tagged_.packDouble(dbl_result);
    ctx_.builder().CreateBr(merge);

    // Integer path - Scheme uses exact division, promoting to double for non-exact
    ctx_.builder().SetInsertPoint(int_path);
    llvm::Value* left_int = tagged_.unpackInt64(left);
    llvm::Value* right_int = tagged_.unpackInt64(right);
    // Check if division is exact
    llvm::Value* remainder = ctx_.builder().CreateSRem(left_int, right_int);
    llvm::Value* is_exact = ctx_.builder().CreateICmpEQ(remainder,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* exact_div = ctx_.builder().CreateSDiv(left_int, right_int);
    llvm::Value* exact_tagged = tagged_.packInt64(exact_div, true);
    // If not exact, convert to floating point
    llvm::Value* left_as_dbl = ctx_.builder().CreateSIToFP(left_int, ctx_.doubleType());
    llvm::Value* right_as_dbl = ctx_.builder().CreateSIToFP(right_int, ctx_.doubleType());
    llvm::Value* inexact_div = ctx_.builder().CreateFDiv(left_as_dbl, right_as_dbl);
    llvm::Value* inexact_tagged = tagged_.packDouble(inexact_div);
    llvm::Value* int_tagged = ctx_.builder().CreateSelect(is_exact, exact_tagged, inexact_tagged);
    ctx_.builder().CreateBr(merge);

    // Merge all paths
    ctx_.builder().SetInsertPoint(merge);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 5, "div_result");
    phi->addIncoming(vec_result, vector_exit);
    phi->addIncoming(ad_result, ad_exit);
    phi->addIncoming(dual_tagged, dual_exit);
    phi->addIncoming(dbl_tagged, double_path);
    phi->addIncoming(int_tagged, int_path);

    return phi;
}

// === Other Operations (mod, neg, abs, type coercion) ===

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

    // Double abs using fneg + select for abs since we may not have fabs intrinsic set up
    ctx_.builder().SetInsertPoint(double_bb);
    llvm::Value* dbl_val = tagged_.unpackDouble(operand);
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
