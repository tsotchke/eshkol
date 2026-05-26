/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Activation Functions (forward + backward).
 * Extracted from tensor_codegen.cpp during the v1.2 mechanical split.
 *
 * Forward: relu, sigmoid, softmax, gelu, leaky-relu, silu, elu, selu,
 * mish, hard-swish, hard-sigmoid, softplus, dropout, celu.
 * Backward (for autodiff): softmax-backward, relu-backward,
 * sigmoid-backward, gelu-backward, leaky-relu-backward,
 * silu-backward, etc.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-activation-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/autodiff_codegen.h>
#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 21
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

namespace eshkol {

// ===== ACTIVATION FUNCTIONS (SIMD-ACCELERATED) =====

llvm::Value* TensorCodegen::tensorRelu(const eshkol_operations_t* op) {
    // ReLU: max(0, x) element-wise
    // Note: For large tensors (≥100K elements), GPU acceleration is available
    // through the XLA elementwise runtime (RELU = op code 9) when invoked via
    // tensor-apply or the general tensor arithmetic path.
    if (op->call_op.num_vars != 1) {
        eshkol_error("relu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor dimensions and elements
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor (same shape)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "relu_result");

    // Allocate and copy dimensions
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* relu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(relu_arena_alloc, {arena_ptr, dims_size}, "relu_dims");

    // Copy dimensions using memcpy intrinsic
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8),
                         dims_ptr, llvm::MaybeAlign(8), dims_size);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(relu_arena_alloc, {arena_ptr, elems_size}, "relu_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Create loop blocks
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "relu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "relu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "relu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 12, exit_block, "relu");

    // Counter
    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "relu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* vec_val = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8), "relu_vec");

        // Create zero vector
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

        // ReLU: max(0, x) = select(x > 0, x, 0)
        llvm::Value* cmp = builder.CreateFCmpOGT(vec_val, zero_vec, "relu_cmp");
        llvm::Value* result_vec = builder.CreateSelect(cmp, vec_val, zero_vec, "relu_result_vec");

        // Store result
        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_ptr, llvm::MaybeAlign(8));
    }

    // Increment counter
    llvm::Value* next_i = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i, counter);
    auto* reluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(reluSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // === Scalar Loop (remainder) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i_scalar = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* scalar_cmp = builder.CreateICmpULT(i_scalar, total_elements);
    builder.CreateCondBr(scalar_cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_scalar);
    llvm::Value* val = builder.CreateLoad(ctx_.doubleType(), src_scalar_ptr);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* cmp_scalar = builder.CreateFCmpOGT(val, zero);
    llvm::Value* result_scalar = builder.CreateSelect(cmp_scalar, val, zero);
    llvm::Value* dst_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_scalar);
    builder.CreateStore(result_scalar, dst_scalar_ptr);

    llvm::Value* next_i_scalar = builder.CreateAdd(i_scalar,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i_scalar, counter);
    auto* reluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(reluScalarBackEdge, false, 0, true, 4);

    // === Exit: populate result tensor struct ===
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSigmoid(const eshkol_operations_t* op) {
    // Sigmoid: 1 / (1 + exp(-x)) element-wise
    if (op->call_op.num_vars != 1) {
        eshkol_error("sigmoid requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor dimensions and elements
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sigmoid_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sig_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(sig_arena_alloc, {arena_ptr, dims_size}, "sig_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(sig_arena_alloc, {arena_ptr, elems_size}, "sig_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar exp intrinsic (for tail loop)
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "sig_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sig_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sig_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 13, exit_block, "sig");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sig_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count (rounds down to multiple of SIMD_WIDTH)
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop: process SIMD_WIDTH elements at a time ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x_vec = builder.CreateAlignedLoad(vec_type, src_vec_ptr, llvm::MaybeAlign(8), "sig_x_vec");

        // Vector intrinsics
        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {vec_type});
        llvm::Function* fabs_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::fabs, {vec_type});

        // Numerically stable sigmoid: avoid computing exp(large positive)
        // exp_neg_abs = exp(-|x|) — argument always <= 0, no overflow
        llvm::Value* abs_x_vec = builder.CreateCall(fabs_vec_func, {x_vec}, "sig_abs_x_vec");
        llvm::Value* neg_abs_x_vec = builder.CreateFNeg(abs_x_vec, "sig_neg_abs_vec");
        llvm::Value* exp_neg_abs_vec = builder.CreateCall(exp_vec_func, {neg_abs_x_vec}, "sig_exp_vec");

        llvm::Value* one_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 1.0));
        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));

        llvm::Value* denom_vec = builder.CreateFAdd(one_vec, exp_neg_abs_vec, "sig_denom_vec");
        // x >= 0: 1/denom;  x < 0: exp_neg_abs/denom
        llvm::Value* pos_result = builder.CreateFDiv(one_vec, denom_vec, "sig_pos_vec");
        llvm::Value* neg_result = builder.CreateFDiv(exp_neg_abs_vec, denom_vec, "sig_neg_vec");
        llvm::Value* is_positive = builder.CreateFCmpOGE(x_vec, zero_vec, "sig_ge_zero");
        llvm::Value* result_vec = builder.CreateSelect(is_positive, pos_result, neg_result, "sig_result_vec");

        // Store result
        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i_simd = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i_simd, counter);
    auto* sigSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(sigSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // === Scalar Loop (remainder elements) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // Numerically stable sigmoid: avoid computing exp(large positive)
    // exp_neg_abs = exp(-|x|) — argument always <= 0, no overflow
    llvm::Function* fabs_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Value* abs_x = builder.CreateCall(fabs_func, {x}, "sig_abs_x");
    llvm::Value* neg_abs_x = builder.CreateFNeg(abs_x, "sig_neg_abs");
    llvm::Value* exp_neg_abs = builder.CreateCall(exp_func, {neg_abs_x}, "sig_exp_neg_abs");
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg_abs, "sig_denom");
    // x >= 0: 1/denom;  x < 0: exp_neg_abs/denom
    llvm::Value* pos_result = builder.CreateFDiv(one, denom, "sig_pos");
    llvm::Value* neg_result = builder.CreateFDiv(exp_neg_abs, denom, "sig_neg");
    llvm::Value* is_positive = builder.CreateFCmpOGE(x,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0), "sig_ge_zero");
    llvm::Value* result = builder.CreateSelect(is_positive, pos_result, neg_result, "sig_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    auto* sigScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(sigScalarBackEdge, false, 0, true, 4);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSoftmax(const eshkol_operations_t* op) {
    // Softmax: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    // Numerically stable version
    // (softmax tensor) — global softmax
    // (softmax tensor axis) — softmax along specified axis
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("softmax requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (softmax tensor axis) → axis-aware softmax via runtime
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;

        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(tensor_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);

        if (autodiff_) {
            llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
            llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_ad_path", builder.GetInsertBlock()->getParent());
            llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_numeric_path", builder.GetInsertBlock()->getParent());
            llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_merge", builder.GetInsertBlock()->getParent());
            builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

            builder.SetInsertPoint(ad_path);
            llvm::Value* ad_arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
            llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
            llvm::Function* arena_alloc = mem_.getArenaAllocate();
            llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {ad_arena}, "softmax_axis_ad_result");

            llvm::Value* dims_size = builder.CreateMul(rank,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
            llvm::Value* result_dims = builder.CreateCall(arena_alloc, {ad_arena, dims_size}, "softmax_axis_ad_dims");
            builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims, llvm::MaybeAlign(8), dims_size);

            llvm::Value* elems_size = builder.CreateMul(total,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
            llvm::Value* result_elems = builder.CreateCall(arena_alloc, {ad_arena, elems_size}, "softmax_axis_ad_elems");

            llvm::Value* one_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 1);
            llvm::Value* axis_len = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, axis));

            llvm::Value* inner_stride = builder.CreateAlloca(ctx_.int64Type(), nullptr, "softmax_axis_inner_stride");
            llvm::Value* stride_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "softmax_axis_stride_i");
            builder.CreateStore(one_i64, inner_stride);
            builder.CreateStore(builder.CreateAdd(axis, one_i64), stride_i_alloca);

            llvm::Function* current_func = builder.GetInsertBlock()->getParent();
            llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_stride_cond", current_func);
            llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_stride_body", current_func);
            llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_stride_done", current_func);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_cond);
            llvm::Value* stride_i = builder.CreateLoad(ctx_.int64Type(), stride_i_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(stride_i, rank), stride_body, stride_done);

            builder.SetInsertPoint(stride_body);
            llvm::Value* stride_dim = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, stride_i));
            llvm::Value* old_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride);
            builder.CreateStore(builder.CreateMul(old_stride, stride_dim), inner_stride);
            builder.CreateStore(builder.CreateAdd(stride_i, one_i64), stride_i_alloca);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_done);
            llvm::Value* group_count = builder.CreateUDiv(total, axis_len);
            llvm::Value* group_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "softmax_axis_group_i");
            builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), group_i_alloca);
            llvm::BasicBlock* group_cond = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_group_cond", current_func);
            llvm::BasicBlock* group_body = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_group_body", current_func);
            llvm::BasicBlock* group_done = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_group_done", current_func);
            builder.CreateBr(group_cond);

            builder.SetInsertPoint(group_cond);
            llvm::Value* group_i = builder.CreateLoad(ctx_.int64Type(), group_i_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(group_i, group_count), group_body, group_done);

            builder.SetInsertPoint(group_body);
            llvm::Value* stride = builder.CreateLoad(ctx_.int64Type(), inner_stride);
            llvm::Value* inner = builder.CreateURem(group_i, stride);
            llvm::Value* outer = builder.CreateUDiv(group_i, stride);
            llvm::Value* outer_stride = builder.CreateMul(axis_len, stride);
            llvm::Value* base = builder.CreateAdd(builder.CreateMul(outer, outer_stride), inner);

            llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), elems, base));
            llvm::Value* max_node_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, "softmax_axis_ad_max");
            builder.CreateStore(adNodeFromTensorElementBits(first_bits, "softmax_axis_ad_first"), max_node_alloca);

            llvm::Value* k_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "softmax_axis_k");
            builder.CreateStore(one_i64, k_alloca);
            llvm::BasicBlock* max_cond = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_max_cond", current_func);
            llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_max_body", current_func);
            llvm::BasicBlock* max_done = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_max_done", current_func);
            builder.CreateBr(max_cond);

            builder.SetInsertPoint(max_cond);
            llvm::Value* max_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(max_k, axis_len), max_body, max_done);

            builder.SetInsertPoint(max_body);
            llvm::Value* max_index = builder.CreateAdd(base, builder.CreateMul(max_k, stride));
            llvm::Value* max_bits = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), elems, max_index));
            llvm::Value* max_elem_node = adNodeFromTensorElementBits(max_bits, "softmax_axis_ad_max_elem");
            llvm::Value* old_max = builder.CreateLoad(ctx_.ptrType(), max_node_alloca);
            builder.CreateStore(autodiff_->recordADNodeBinary(44, old_max, max_elem_node), max_node_alloca);
            builder.CreateStore(builder.CreateAdd(max_k, one_i64), k_alloca);
            builder.CreateBr(max_cond);

            builder.SetInsertPoint(max_done);
            llvm::Value* sum_node_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, "softmax_axis_ad_sum");
            builder.CreateStore(autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), sum_node_alloca);
            builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_alloca);
            llvm::BasicBlock* exp_cond = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_exp_cond", current_func);
            llvm::BasicBlock* exp_body = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_exp_body", current_func);
            llvm::BasicBlock* exp_done = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_exp_done", current_func);
            builder.CreateBr(exp_cond);

            builder.SetInsertPoint(exp_cond);
            llvm::Value* exp_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(exp_k, axis_len), exp_body, exp_done);

            builder.SetInsertPoint(exp_body);
            llvm::Value* exp_index = builder.CreateAdd(base, builder.CreateMul(exp_k, stride));
            llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), elems, exp_index));
            llvm::Value* elem_node = adNodeFromTensorElementBits(elem_bits, "softmax_axis_ad_elem");
            llvm::Value* max_node = builder.CreateLoad(ctx_.ptrType(), max_node_alloca);
            llvm::Value* shifted_node = autodiff_->recordADNodeBinary(3, elem_node, max_node);
            llvm::Value* exp_node = autodiff_->recordADNodeUnary(8, shifted_node);
            llvm::Value* old_sum = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
            builder.CreateStore(autodiff_->recordADNodeBinary(2, old_sum, exp_node), sum_node_alloca);
            builder.CreateStore(builder.CreatePtrToInt(exp_node, ctx_.int64Type()),
                builder.CreateGEP(ctx_.int64Type(), result_elems, exp_index));
            builder.CreateStore(builder.CreateAdd(exp_k, one_i64), k_alloca);
            builder.CreateBr(exp_cond);

            builder.SetInsertPoint(exp_done);
            builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_alloca);
            llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_norm_cond", current_func);
            llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_norm_body", current_func);
            llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "softmax_axis_norm_done", current_func);
            builder.CreateBr(norm_cond);

            builder.SetInsertPoint(norm_cond);
            llvm::Value* norm_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(norm_k, axis_len), norm_body, norm_done);

            builder.SetInsertPoint(norm_body);
            llvm::Value* norm_index = builder.CreateAdd(base, builder.CreateMul(norm_k, stride));
            llvm::Value* exp_bits = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), result_elems, norm_index));
            llvm::Value* exp_node_for_norm = adNodeFromTensorElementBits(exp_bits, "softmax_axis_ad_exp");
            llvm::Value* denom_node = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
            llvm::Value* softmax_node = autodiff_->recordADNodeBinary(5, exp_node_for_norm, denom_node);
            builder.CreateStore(builder.CreatePtrToInt(softmax_node, ctx_.int64Type()),
                builder.CreateGEP(ctx_.int64Type(), result_elems, norm_index));
            builder.CreateStore(builder.CreateAdd(norm_k, one_i64), k_alloca);
            builder.CreateBr(norm_cond);

            builder.SetInsertPoint(norm_done);
            builder.CreateStore(builder.CreateAdd(group_i, one_i64), group_i_alloca);
            builder.CreateBr(group_cond);

            builder.SetInsertPoint(group_done);
            builder.CreateStore(result_dims, builder.CreateStructGEP(ttype, result_ptr, 0));
            builder.CreateStore(rank, builder.CreateStructGEP(ttype, result_ptr, 1));
            builder.CreateStore(result_elems, builder.CreateStructGEP(ttype, result_ptr, 2));
            builder.CreateStore(total, builder.CreateStructGEP(ttype, result_ptr, 3));
            llvm::Value* ad_result = tagged_.packHeapPtr(result_ptr);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* ad_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(numeric_path);
            llvm::Value* numeric_arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
            auto* ptrTy = ctx_.ptrType();
            auto* i64Ty = ctx_.int64Type();
            llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
                {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty}, false);
            llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_softmax", fn_type);
            llvm::Value* numeric_result = builder.CreateCall(callee,
                {numeric_arena, elems, total, dims, rank, axis}, "softmax_axis_numeric_result");
            llvm::Value* numeric_packed = tagged_.packHeapPtr(numeric_result);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* numeric_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(merge_block);
            llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), 2, "softmax_axis_result_phi");
            result_phi->addIncoming(ad_result, ad_exit);
            result_phi->addIncoming(numeric_packed, numeric_exit);
            return result_phi;
        }

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_softmax", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis}, "softmax_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "softmax_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sm_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(sm_arena_alloc, {arena_ptr, dims_size}, "sm_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(sm_arena_alloc, {arena_ptr, elems_size}, "sm_elems");

    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sm_exit", current_func);

    if (autodiff_) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_path", current_func);
        llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "sm_numeric_path", current_func);
        builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

        builder.SetInsertPoint(ad_path);
        llvm::Value* ad_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sm_ad_i");
        llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_elems, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
        llvm::Value* max_node_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, "sm_ad_max");
        builder.CreateStore(adNodeFromTensorElementBits(first_bits, "sm_ad_max_first"), max_node_alloca);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), ad_counter);

        llvm::BasicBlock* max_ad_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_max_cond", current_func);
        llvm::BasicBlock* max_ad_body = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_max_body", current_func);
        llvm::BasicBlock* exp_ad_init = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_exp_init", current_func);
        builder.CreateBr(max_ad_cond);

        builder.SetInsertPoint(max_ad_cond);
        llvm::Value* max_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
        builder.CreateCondBr(builder.CreateICmpULT(max_i, total_elements), max_ad_body, exp_ad_init);

        builder.SetInsertPoint(max_ad_body);
        llvm::Value* max_elem_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_elems, max_i));
        llvm::Value* max_elem_node = adNodeFromTensorElementBits(max_elem_bits, "sm_ad_max_elem");
        llvm::Value* old_max_node = builder.CreateLoad(ctx_.ptrType(), max_node_alloca);
        llvm::Value* new_max_node = autodiff_->recordADNodeBinary(44, old_max_node, max_elem_node);
        builder.CreateStore(new_max_node, max_node_alloca);
        builder.CreateStore(builder.CreateAdd(max_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_counter);
        builder.CreateBr(max_ad_cond);

        builder.SetInsertPoint(exp_ad_init);
        llvm::Value* sum_node_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, "sm_ad_sum");
        builder.CreateStore(autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), sum_node_alloca);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);

        llvm::BasicBlock* exp_ad_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_exp_cond", current_func);
        llvm::BasicBlock* exp_ad_body = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_exp_body", current_func);
        llvm::BasicBlock* norm_ad_init = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_norm_init", current_func);
        builder.CreateBr(exp_ad_cond);

        builder.SetInsertPoint(exp_ad_cond);
        llvm::Value* exp_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
        builder.CreateCondBr(builder.CreateICmpULT(exp_i, total_elements), exp_ad_body, norm_ad_init);

        builder.SetInsertPoint(exp_ad_body);
        llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), src_elems, exp_i));
        llvm::Value* elem_node = adNodeFromTensorElementBits(elem_bits, "sm_ad_elem");
        llvm::Value* max_node = builder.CreateLoad(ctx_.ptrType(), max_node_alloca);
        llvm::Value* shifted_node = autodiff_->recordADNodeBinary(3, elem_node, max_node);
        llvm::Value* exp_node = autodiff_->recordADNodeUnary(8, shifted_node);
        llvm::Value* old_sum_node = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
        llvm::Value* new_sum_node = autodiff_->recordADNodeBinary(2, old_sum_node, exp_node);
        builder.CreateStore(new_sum_node, sum_node_alloca);
        builder.CreateStore(builder.CreatePtrToInt(exp_node, ctx_.int64Type()),
            builder.CreateGEP(ctx_.int64Type(), result_elems, exp_i));
        builder.CreateStore(builder.CreateAdd(exp_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_counter);
        builder.CreateBr(exp_ad_cond);

        builder.SetInsertPoint(norm_ad_init);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);
        llvm::BasicBlock* norm_ad_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_norm_cond", current_func);
        llvm::BasicBlock* norm_ad_body = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_norm_body", current_func);
        llvm::BasicBlock* norm_ad_done = llvm::BasicBlock::Create(ctx_.context(), "sm_ad_norm_done", current_func);
        builder.CreateBr(norm_ad_cond);

        builder.SetInsertPoint(norm_ad_cond);
        llvm::Value* norm_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
        builder.CreateCondBr(builder.CreateICmpULT(norm_i, total_elements), norm_ad_body, norm_ad_done);

        builder.SetInsertPoint(norm_ad_body);
        llvm::Value* exp_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), result_elems, norm_i));
        llvm::Value* exp_node_for_norm = adNodeFromTensorElementBits(exp_bits, "sm_ad_exp_node");
        llvm::Value* denom_node = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
        llvm::Value* softmax_node = autodiff_->recordADNodeBinary(5, exp_node_for_norm, denom_node);
        builder.CreateStore(builder.CreatePtrToInt(softmax_node, ctx_.int64Type()),
            builder.CreateGEP(ctx_.int64Type(), result_elems, norm_i));
        builder.CreateStore(builder.CreateAdd(norm_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_counter);
        builder.CreateBr(norm_ad_cond);

        builder.SetInsertPoint(norm_ad_done);
        builder.CreateBr(exit_block);

        builder.SetInsertPoint(numeric_path);
    }

    // Pass 1: Find maximum element
    llvm::BasicBlock* max_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_max_cond", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "sm_max_body", current_func);
    llvm::BasicBlock* sum_init = llvm::BasicBlock::Create(ctx_.context(), "sm_sum_init", current_func);

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sm_max");
    llvm::Value* first_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), src_elems, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(first_elem, max_val);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sm_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), counter);
    builder.CreateBr(max_cond);

    builder.SetInsertPoint(max_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, max_body, sum_init);

    builder.SetInsertPoint(max_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* elem = builder.CreateLoad(ctx_.doubleType(), elem_ptr);
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem, cur_max);
    builder.CreateStore(new_max, max_val);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(max_cond);

    // SIMD parameters for softmax
    const unsigned SM_SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* sm_vec_type = getSIMDVectorType();
    const bool sm_use_simd = (SM_SIMD_WIDTH > 1 && sm_vec_type != nullptr);

    // Pass 2: Compute exp(x - max) and sum — SIMD vectorized
    builder.SetInsertPoint(sum_init);
    llvm::BasicBlock* exp_simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_simd_cond", current_func);
    llvm::BasicBlock* exp_simd_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_simd_body", current_func);
    llvm::BasicBlock* exp_scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_scalar_cond", current_func);
    llvm::BasicBlock* exp_scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_scalar_body", current_func);
    llvm::BasicBlock* norm_init = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_init", current_func);

    llvm::Value* sum_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sm_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // SIMD iteration count for exp pass
    llvm::Value* sm_simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateBr(exp_simd_cond);

    // === SIMD exp loop ===
    builder.SetInsertPoint(exp_simd_cond);
    llvm::Value* i2s = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2s = builder.CreateICmpULT(i2s, sm_simd_count);
    builder.CreateCondBr(cmp2s, exp_simd_body, exp_scalar_cond);

    builder.SetInsertPoint(exp_simd_body);
    if (sm_use_simd) {
        llvm::Value* final_max_s = builder.CreateLoad(ctx_.doubleType(), max_val);
        llvm::Value* max_vec = builder.CreateVectorSplat(SM_SIMD_WIDTH, final_max_s, "max_splat");

        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i2s);
        llvm::Value* x_vec = builder.CreateAlignedLoad(sm_vec_type, src_vec_ptr, llvm::MaybeAlign(8), "sm_x_vec");
        llvm::Value* shifted_vec = builder.CreateFSub(x_vec, max_vec, "sm_shifted_vec");

        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {sm_vec_type});
        llvm::Value* exp_vec = builder.CreateCall(exp_vec_func, {shifted_vec}, "sm_exp_vec");

        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i2s);
        builder.CreateAlignedStore(exp_vec, dst_vec_ptr, llvm::MaybeAlign(8));

        // Horizontal sum of exp vector for running total
        llvm::Value* cur_sum_s = builder.CreateLoad(ctx_.doubleType(), sum_val);
        for (unsigned lane = 0; lane < SM_SIMD_WIDTH; ++lane) {
            llvm::Value* lane_val = builder.CreateExtractElement(exp_vec,
                llvm::ConstantInt::get(ctx_.int32Type(), lane), "sm_lane_" + std::to_string(lane));
            cur_sum_s = builder.CreateFAdd(cur_sum_s, lane_val);
        }
        builder.CreateStore(cur_sum_s, sum_val);
    }
    llvm::Value* next_i2s = builder.CreateAdd(i2s, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateStore(next_i2s, counter);
    builder.CreateBr(exp_simd_cond);

    // === Scalar exp tail loop ===
    builder.SetInsertPoint(exp_scalar_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, exp_scalar_body, norm_init);

    builder.SetInsertPoint(exp_scalar_body);
    llvm::Value* src_ptr2 = builder.CreateGEP(ctx_.doubleType(), src_elems, i2);
    llvm::Value* x2 = builder.CreateLoad(ctx_.doubleType(), src_ptr2);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* x_shifted = builder.CreateFSub(x2, final_max, "x_shifted");
    llvm::Value* exp_x = builder.CreateCall(exp_func, {x_shifted}, "exp_shifted");
    llvm::Value* dst_ptr2 = builder.CreateGEP(ctx_.doubleType(), result_elems, i2);
    builder.CreateStore(exp_x, dst_ptr2);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* new_sum = builder.CreateFAdd(cur_sum, exp_x);
    builder.CreateStore(new_sum, sum_val);
    llvm::Value* next_i2 = builder.CreateAdd(i2, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i2, counter);
    builder.CreateBr(exp_scalar_cond);

    // Pass 3: Normalize (divide by sum) — SIMD vectorized
    builder.SetInsertPoint(norm_init);

    // Zero-guard: prevent division by zero if all exp values underflowed to 0
    {
        llvm::Value* cur_total_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
        llvm::Value* sum_is_zero = builder.CreateFCmpOEQ(cur_total_sum,
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* safe_total_sum = builder.CreateSelect(sum_is_zero,
            llvm::ConstantFP::get(ctx_.doubleType(), 1e-10), cur_total_sum);
        builder.CreateStore(safe_total_sum, sum_val);
    }

    llvm::BasicBlock* norm_simd_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_simd_cond", current_func);
    llvm::BasicBlock* norm_simd_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_simd_body", current_func);
    llvm::BasicBlock* norm_scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_scalar_cond", current_func);
    llvm::BasicBlock* norm_scalar_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_scalar_body", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(norm_simd_cond);

    // === SIMD normalization loop ===
    builder.SetInsertPoint(norm_simd_cond);
    llvm::Value* i3s = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp3s = builder.CreateICmpULT(i3s, sm_simd_count);
    builder.CreateCondBr(cmp3s, norm_simd_body, norm_scalar_cond);

    builder.SetInsertPoint(norm_simd_body);
    if (sm_use_simd) {
        llvm::Value* total_sum_s = builder.CreateLoad(ctx_.doubleType(), sum_val);
        llvm::Value* sum_vec = builder.CreateVectorSplat(SM_SIMD_WIDTH, total_sum_s, "sum_splat");

        llvm::Value* res_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i3s);
        llvm::Value* exp_vals = builder.CreateAlignedLoad(sm_vec_type, res_vec_ptr, llvm::MaybeAlign(8), "sm_exp_vals");
        llvm::Value* norm_vec = builder.CreateFDiv(exp_vals, sum_vec, "sm_norm_vec");
        builder.CreateAlignedStore(norm_vec, res_vec_ptr, llvm::MaybeAlign(8));
    }
    llvm::Value* next_i3s = builder.CreateAdd(i3s, llvm::ConstantInt::get(ctx_.int64Type(), SM_SIMD_WIDTH));
    builder.CreateStore(next_i3s, counter);
    builder.CreateBr(norm_simd_cond);

    // === Scalar normalization tail loop ===
    builder.SetInsertPoint(norm_scalar_cond);
    llvm::Value* i3 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp3 = builder.CreateICmpULT(i3, total_elements);
    builder.CreateCondBr(cmp3, norm_scalar_body, exit_block);

    builder.SetInsertPoint(norm_scalar_body);
    llvm::Value* res_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i3);
    llvm::Value* exp_val = builder.CreateLoad(ctx_.doubleType(), res_ptr);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* normalized = builder.CreateFDiv(exp_val, total_sum, "normalized");
    builder.CreateStore(normalized, res_ptr);
    llvm::Value* next_i3 = builder.CreateAdd(i3, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i3, counter);
    builder.CreateBr(norm_scalar_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorGelu(const eshkol_operations_t* op) {
    // GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) — PyTorch standard
    if (op->call_op.num_vars != 1) {
        eshkol_error("gelu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "gelu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* gelu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(gelu_arena_alloc, {arena_ptr, dims_size}, "gelu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(gelu_arena_alloc, {arena_ptr, elems_size}, "gelu_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Scalar exp intrinsic (for tail loop)
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "gelu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 16, exit_block, "gelu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "gelu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // Calculate SIMD iteration count (rounds down to multiple of SIMD_WIDTH)
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD Loop: process SIMD_WIDTH elements at a time ===
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        // Load SIMD_WIDTH elements
        llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x_vec = builder.CreateAlignedLoad(vec_type, src_vec_ptr, llvm::MaybeAlign(8), "gelu_x_vec");

        // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        // Standard tanh approximation (PyTorch default)
        llvm::Value* half_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.5));
        llvm::Value* one_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 1.0));
        llvm::Value* sqrt2pi_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654)); // sqrt(2/π)
        llvm::Value* k_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.044715));
        llvm::Value* two_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 2.0));

        // x³ = x * x * x
        llvm::Value* x2_vec = builder.CreateFMul(x_vec, x_vec, "gelu_x2_vec");
        llvm::Value* x3_vec = builder.CreateFMul(x2_vec, x_vec, "gelu_x3_vec");
        // inner = x + 0.044715 * x³
        llvm::Value* kx3_vec = builder.CreateFMul(k_vec, x3_vec, "gelu_kx3_vec");
        llvm::Value* inner_vec = builder.CreateFAdd(x_vec, kx3_vec, "gelu_inner_vec");
        // arg = sqrt(2/π) * inner
        llvm::Value* arg_vec = builder.CreateFMul(sqrt2pi_vec, inner_vec, "gelu_arg_vec");
        // tanh(arg) via (exp(2*arg) - 1) / (exp(2*arg) + 1)
        llvm::Function* exp_vec_func = ESHKOL_GET_INTRINSIC(
            &ctx_.module(), llvm::Intrinsic::exp, {vec_type});
        llvm::Value* two_arg_vec = builder.CreateFMul(two_vec, arg_vec, "gelu_2arg_vec");
        llvm::Value* exp_2arg_vec = builder.CreateCall(exp_vec_func, {two_arg_vec}, "gelu_exp_vec");
        llvm::Value* tanh_num = builder.CreateFSub(exp_2arg_vec, one_vec, "gelu_tanh_num");
        llvm::Value* tanh_den = builder.CreateFAdd(exp_2arg_vec, one_vec, "gelu_tanh_den");
        llvm::Value* tanh_vec = builder.CreateFDiv(tanh_num, tanh_den, "gelu_tanh_vec");
        // result = 0.5 * x * (1 + tanh)
        llvm::Value* one_plus_tanh = builder.CreateFAdd(one_vec, tanh_vec, "gelu_1pt_vec");
        llvm::Value* half_x = builder.CreateFMul(half_vec, x_vec, "gelu_halfx_vec");
        llvm::Value* result_vec = builder.CreateFMul(half_x, one_plus_tanh, "gelu_result_vec");

        // Store result
        llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i_simd = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i_simd, counter);
    auto* geluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(geluSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // === Scalar Loop (remainder elements) ===
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
    llvm::Value* x = builder.CreateLoad(ctx_.doubleType(), src_ptr);

    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* sqrt2pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* k = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    // x³
    llvm::Value* x2 = builder.CreateFMul(x, x, "gelu_x2");
    llvm::Value* x3 = builder.CreateFMul(x2, x, "gelu_x3");
    // inner = x + 0.044715 * x³
    llvm::Value* kx3 = builder.CreateFMul(k, x3, "gelu_kx3");
    llvm::Value* inner = builder.CreateFAdd(x, kx3, "gelu_inner");
    // arg = sqrt(2/π) * inner
    llvm::Value* arg = builder.CreateFMul(sqrt2pi, inner, "gelu_arg");
    // tanh(arg) via (exp(2*arg) - 1) / (exp(2*arg) + 1)
    llvm::Value* two_arg = builder.CreateFMul(two, arg, "gelu_2arg");
    llvm::Value* exp_2arg = builder.CreateCall(exp_func, {two_arg}, "gelu_exp");
    llvm::Value* tanh_num = builder.CreateFSub(exp_2arg, one, "gelu_tanh_n");
    llvm::Value* tanh_den = builder.CreateFAdd(exp_2arg, one, "gelu_tanh_d");
    llvm::Value* tanh_val = builder.CreateFDiv(tanh_num, tanh_den, "gelu_tanh");
    // 0.5 * x * (1 + tanh)
    llvm::Value* one_plus_tanh = builder.CreateFAdd(one, tanh_val, "gelu_1pt");
    llvm::Value* half_x = builder.CreateFMul(half, x, "gelu_halfx");
    llvm::Value* result = builder.CreateFMul(half_x, one_plus_tanh, "gelu_result");

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(result, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    auto* geluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(geluScalarBackEdge, false, 0, true, 4);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorLeakyRelu(const eshkol_operations_t* op) {
    // Leaky ReLU: x if x > 0, else alpha * x (default alpha = 0.01)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("leaky-relu requires 1-2 arguments (tensor, optional alpha)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // Default alpha
    double alpha_val = 0.01;
    // Note: For now we use compile-time constant alpha; runtime alpha would need extraction

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "lrelu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* lrelu_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(lrelu_arena_alloc, {arena_ptr, dims_size}, "lrelu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(lrelu_arena_alloc, {arena_ptr, elems_size}, "lrelu_elems");

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_simd_cond", current_func);
    llvm::BasicBlock* simd_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_simd_body", current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_scalar_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "lrelu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 17, exit_block, "lrelu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "lrelu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // SIMD Loop
    builder.SetInsertPoint(simd_cond);
    llvm::Value* i_simd = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* simd_cmp = builder.CreateICmpULT(i_simd, simd_count);
    builder.CreateCondBr(simd_cmp, simd_body, scalar_cond);

    builder.SetInsertPoint(simd_body);
    if (use_simd) {
        llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_simd);
        llvm::Value* x = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8), "lrelu_vec");

        llvm::Value* zero_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* alpha_vec = llvm::ConstantVector::getSplat(
            llvm::ElementCount::getFixed(SIMD_WIDTH),
            llvm::ConstantFP::get(ctx_.doubleType(), alpha_val));

        // alpha * x
        llvm::Value* scaled = builder.CreateFMul(alpha_vec, x);

        // x > 0 ? x : alpha*x
        llvm::Value* cmp = builder.CreateFCmpOGT(x, zero_vec);
        llvm::Value* result_vec = builder.CreateSelect(cmp, x, scaled, "lrelu_result_vec");

        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_simd);
        builder.CreateAlignedStore(result_vec, dst_ptr, llvm::MaybeAlign(8));
    }

    llvm::Value* next_i = builder.CreateAdd(i_simd,
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
    builder.CreateStore(next_i, counter);
    auto* lreluSimdBackEdge = builder.CreateBr(simd_cond);
    attachLoopMetadata(lreluSimdBackEdge, true, SIMD_WIDTH, false, 0);

    // Scalar Loop (remainder)
    builder.SetInsertPoint(scalar_cond);
    llvm::Value* i_scalar = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* scalar_cmp = builder.CreateICmpULT(i_scalar, total_elements);
    builder.CreateCondBr(scalar_cmp, scalar_body, exit_block);

    builder.SetInsertPoint(scalar_body);
    llvm::Value* src_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i_scalar);
    llvm::Value* val = builder.CreateLoad(ctx_.doubleType(), src_scalar_ptr);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), alpha_val);
    llvm::Value* scaled_val = builder.CreateFMul(alpha, val);
    llvm::Value* cmp_scalar = builder.CreateFCmpOGT(val, zero);
    llvm::Value* result_scalar = builder.CreateSelect(cmp_scalar, val, scaled_val);
    llvm::Value* dst_scalar_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i_scalar);
    builder.CreateStore(result_scalar, dst_scalar_ptr);

    llvm::Value* next_i_scalar = builder.CreateAdd(i_scalar,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i_scalar, counter);
    auto* lreluScalarBackEdge = builder.CreateBr(scalar_cond);
    attachLoopMetadata(lreluScalarBackEdge, false, 0, true, 4);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSilu(const eshkol_operations_t* op) {
    // SiLU/Swish: x * sigmoid(x)
    if (op->call_op.num_vars != 1) {
        eshkol_error("silu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;
    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "silu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "result_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "result_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop to compute silu: x * (1 / (1 + exp(-x)))
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "silu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "silu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "silu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 18, exit_block, "silu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "silu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // sigmoid(x) = 1 / (1 + exp(-x))
    llvm::Value* neg_val = builder.CreateFNeg(val);
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg);
    llvm::Value* sigmoid_val = builder.CreateFDiv(one, denom);
    // silu = x * sigmoid(x)
    llvm::Value* result_val = builder.CreateFMul(val, sigmoid_val);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    auto* siluBackEdge = builder.CreateBr(loop_cond);
    attachLoopMetadata(siluBackEdge, false, 0, true, 4);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorElu(const eshkol_operations_t* op) {
    // ELU: x > 0 ? x : alpha * (exp(x) - 1)
    // Default alpha = 1.0
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("elu requires 1 or 2 arguments (tensor [, alpha])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* alpha;
    if (op->call_op.num_vars == 2) {
        llvm::Value* alpha_tagged = codegenAST(&op->call_op.variables[1]);
        if (!alpha_tagged) return nullptr;
        alpha = tagged_.unpackDouble(alpha_tagged);
    } else {
        alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "elu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "elu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "elu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "elu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "elu_body", current_func);
    llvm::BasicBlock* elu_pos = llvm::BasicBlock::Create(ctx_.context(), "elu_pos", current_func);
    llvm::BasicBlock* elu_neg = llvm::BasicBlock::Create(ctx_.context(), "elu_neg", current_func);
    llvm::BasicBlock* elu_merge = llvm::BasicBlock::Create(ctx_.context(), "elu_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "elu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 46, exit_block, "elu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(val, zero);
    builder.CreateCondBr(is_positive, elu_pos, elu_neg);

    // Positive branch: output = x
    builder.SetInsertPoint(elu_pos);
    builder.CreateBr(elu_merge);

    // Negative branch: output = alpha * (exp(x) - 1)
    builder.SetInsertPoint(elu_neg);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* neg_result = builder.CreateFMul(alpha, exp_minus_1);
    builder.CreateBr(elu_merge);

    // Merge
    builder.SetInsertPoint(elu_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "elu_val");
    result_val->addIncoming(val, elu_pos);
    result_val->addIncoming(neg_result, elu_neg);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSelu(const eshkol_operations_t* op) {
    // SELU: lambda * (x > 0 ? x : alpha * (exp(x) - 1))
    // lambda = 1.0507009873554804934193349852946
    // alpha  = 1.6732632423543772848170429916717
    if (op->call_op.num_vars != 1) {
        eshkol_error("selu requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "selu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "selu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "selu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Value* selu_lambda = llvm::ConstantFP::get(ctx_.doubleType(), 1.0507009873554804934193349852946);
    llvm::Value* selu_alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.6732632423543772848170429916717);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "selu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "selu_body", current_func);
    llvm::BasicBlock* selu_pos = llvm::BasicBlock::Create(ctx_.context(), "selu_pos", current_func);
    llvm::BasicBlock* selu_neg = llvm::BasicBlock::Create(ctx_.context(), "selu_neg", current_func);
    llvm::BasicBlock* selu_merge = llvm::BasicBlock::Create(ctx_.context(), "selu_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "selu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 47, exit_block, "selu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "selu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(val, zero);
    builder.CreateCondBr(is_positive, selu_pos, selu_neg);

    // Positive: lambda * x
    builder.SetInsertPoint(selu_pos);
    llvm::Value* pos_result = builder.CreateFMul(selu_lambda, val);
    builder.CreateBr(selu_merge);

    // Negative: lambda * alpha * (exp(x) - 1)
    builder.SetInsertPoint(selu_neg);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* alpha_exp = builder.CreateFMul(selu_alpha, exp_minus_1);
    llvm::Value* neg_result = builder.CreateFMul(selu_lambda, alpha_exp);
    builder.CreateBr(selu_merge);

    builder.SetInsertPoint(selu_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "selu_val");
    result_val->addIncoming(pos_result, selu_pos);
    result_val->addIncoming(neg_result, selu_neg);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorMish(const eshkol_operations_t* op) {
    // Mish: x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    // Numerically stable: for x > 20, softplus(x) ≈ x, so mish ≈ x * tanh(x)
    if (op->call_op.num_vars != 1) {
        eshkol_error("mish requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "mish_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "mish_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "mish_elems");

    // Declare math functions
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* tanh_func = ctx_.module().getFunction("tanh");
    if (!tanh_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        tanh_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "tanh", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "mish_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "mish_body", current_func);
    llvm::BasicBlock* mish_large = llvm::BasicBlock::Create(ctx_.context(), "mish_large", current_func);
    llvm::BasicBlock* mish_normal = llvm::BasicBlock::Create(ctx_.context(), "mish_normal", current_func);
    llvm::BasicBlock* mish_merge = llvm::BasicBlock::Create(ctx_.context(), "mish_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "mish_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 48, exit_block, "mish");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mish_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // Stability: if x > 20, softplus(x) ≈ x, so mish ≈ x * tanh(x)
    llvm::Value* threshold = llvm::ConstantFP::get(ctx_.doubleType(), 20.0);
    llvm::Value* is_large = builder.CreateFCmpOGT(val, threshold);
    builder.CreateCondBr(is_large, mish_large, mish_normal);

    // Large x: x * tanh(x)
    builder.SetInsertPoint(mish_large);
    llvm::Value* tanh_x = builder.CreateCall(tanh_func, {val});
    llvm::Value* large_result = builder.CreateFMul(val, tanh_x);
    builder.CreateBr(mish_merge);

    // Normal: x * tanh(log(1 + exp(x)))
    builder.SetInsertPoint(mish_normal);
    llvm::Value* exp_x = builder.CreateCall(exp_func, {val});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_plus_exp = builder.CreateFAdd(one, exp_x);
    llvm::Value* softplus = builder.CreateCall(log_func, {one_plus_exp});
    llvm::Value* tanh_sp = builder.CreateCall(tanh_func, {softplus});
    llvm::Value* normal_result = builder.CreateFMul(val, tanh_sp);
    builder.CreateBr(mish_merge);

    builder.SetInsertPoint(mish_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "mish_val");
    result_val->addIncoming(large_result, mish_large);
    result_val->addIncoming(normal_result, mish_normal);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorHardSwish(const eshkol_operations_t* op) {
    // Hard Swish: x * min(max(x + 3, 0), 6) / 6
    // Piecewise: x <= -3 → 0, x >= 3 → x, else → x * (x + 3) / 6
    if (op->call_op.num_vars != 1) {
        eshkol_error("hard-swish requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "hswish_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "hswish_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "hswish_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "hswish_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "hswish_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "hswish_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 49, exit_block, "hswish");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "hswish_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // Compute min(max(x + 3, 0), 6)
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    llvm::Value* x_plus_3 = builder.CreateFAdd(val, three);
    // max(x + 3, 0)
    llvm::Value* cmp_zero = builder.CreateFCmpOGT(x_plus_3, zero);
    llvm::Value* clamped_low = builder.CreateSelect(cmp_zero, x_plus_3, zero);
    // min(clamped, 6)
    llvm::Value* cmp_six = builder.CreateFCmpOLT(clamped_low, six);
    llvm::Value* clamped = builder.CreateSelect(cmp_six, clamped_low, six);
    // x * clamped / 6
    llvm::Value* scaled = builder.CreateFMul(val, clamped);
    llvm::Value* result_val = builder.CreateFDiv(scaled, six);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorHardSigmoid(const eshkol_operations_t* op) {
    // Hard Sigmoid: clip((x + 3) / 6, 0, 1) = min(max((x + 3) / 6, 0), 1)
    if (op->call_op.num_vars != 1) {
        eshkol_error("hard-sigmoid requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "hsigmoid_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "hsigmoid_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "hsigmoid_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "hsigmoid_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 50, exit_block, "hsigmoid");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "hsigmoid_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // (x + 3) / 6
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    llvm::Value* x_plus_3 = builder.CreateFAdd(val, three);
    llvm::Value* divided = builder.CreateFDiv(x_plus_3, six);
    // clip to [0, 1]
    llvm::Value* cmp_zero = builder.CreateFCmpOGT(divided, zero);
    llvm::Value* clamped_low = builder.CreateSelect(cmp_zero, divided, zero);
    llvm::Value* cmp_one = builder.CreateFCmpOLT(clamped_low, one);
    llvm::Value* result_val = builder.CreateSelect(cmp_one, clamped_low, one);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSoftplus(const eshkol_operations_t* op) {
    // Softplus: (1/beta) * log(1 + exp(beta * x))
    // Default beta = 1.0, threshold = 20.0
    // For numerical stability: if beta*x > threshold, return x
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("softplus requires 1 or 2 arguments (tensor [, beta])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* beta;
    if (op->call_op.num_vars == 2) {
        llvm::Value* beta_tagged = codegenAST(&op->call_op.variables[1]);
        if (!beta_tagged) return nullptr;
        beta = tagged_.unpackDouble(beta_tagged);
    } else {
        beta = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "softplus_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "softplus_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "softplus_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "sp_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "sp_body", current_func);
    llvm::BasicBlock* sp_large = llvm::BasicBlock::Create(ctx_.context(), "sp_large", current_func);
    llvm::BasicBlock* sp_normal = llvm::BasicBlock::Create(ctx_.context(), "sp_normal", current_func);
    llvm::BasicBlock* sp_merge = llvm::BasicBlock::Create(ctx_.context(), "sp_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sp_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 51, exit_block, "sp");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sp_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // beta * x
    llvm::Value* beta_x = builder.CreateFMul(beta, val);
    // Threshold check: if beta*x > 20, just return x
    llvm::Value* threshold = llvm::ConstantFP::get(ctx_.doubleType(), 20.0);
    llvm::Value* is_large = builder.CreateFCmpOGT(beta_x, threshold);
    builder.CreateCondBr(is_large, sp_large, sp_normal);

    // Large: return x directly (softplus saturates to identity)
    builder.SetInsertPoint(sp_large);
    builder.CreateBr(sp_merge);

    // Normal: (1/beta) * log(1 + exp(beta * x))
    builder.SetInsertPoint(sp_normal);
    llvm::Value* exp_bx = builder.CreateCall(exp_func, {beta_x});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_plus_exp = builder.CreateFAdd(one, exp_bx);
    llvm::Value* log_val = builder.CreateCall(log_func, {one_plus_exp});
    llvm::Value* inv_beta = builder.CreateFDiv(one, beta);
    llvm::Value* normal_result = builder.CreateFMul(inv_beta, log_val);
    builder.CreateBr(sp_merge);

    builder.SetInsertPoint(sp_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "sp_val");
    result_val->addIncoming(val, sp_large);
    result_val->addIncoming(normal_result, sp_normal);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorDropout(const eshkol_operations_t* op) {
    // Dropout: x_i * mask_i / (1 - p), where mask_i ~ Bernoulli(1 - p)
    // During training, randomly zeros elements with probability p and scales survivors
    // Args: tensor, p (drop probability, 0 < p < 1)
    if (op->call_op.num_vars != 2) {
        eshkol_error("dropout requires exactly 2 arguments (tensor, drop_probability)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;
    llvm::Value* p_tagged = codegenAST(&op->call_op.variables[1]);
    if (!p_tagged) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* p = tagged_.unpackDouble(p_tagged);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "dropout_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "dropout_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "dropout_elems");

    // Get drand48 for random number generation
    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    // Compute scale = 1.0 / (1.0 - p) for inverted dropout
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_p = builder.CreateFSub(one, p);
    llvm::Value* scale = builder.CreateFDiv(one, one_minus_p);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "drop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "drop_body", current_func);
    llvm::BasicBlock* drop_keep = llvm::BasicBlock::Create(ctx_.context(), "drop_keep", current_func);
    llvm::BasicBlock* drop_zero = llvm::BasicBlock::Create(ctx_.context(), "drop_zero", current_func);
    llvm::BasicBlock* drop_merge = llvm::BasicBlock::Create(ctx_.context(), "drop_merge", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "drop_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "drop_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    // Generate random number in [0, 1)
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    // Keep if rand >= p (probability 1-p of keeping)
    llvm::Value* keep = builder.CreateFCmpOGE(rand_val, p);
    builder.CreateCondBr(keep, drop_keep, drop_zero);

    // Keep: scale the value
    builder.SetInsertPoint(drop_keep);
    llvm::Value* scaled_val = builder.CreateFMul(val, scale);
    builder.CreateBr(drop_merge);

    // Drop: zero
    builder.SetInsertPoint(drop_zero);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    builder.CreateBr(drop_merge);

    builder.SetInsertPoint(drop_merge);
    llvm::PHINode* result_val = builder.CreatePHI(ctx_.doubleType(), 2, "drop_val");
    result_val->addIncoming(scaled_val, drop_keep);
    result_val->addIncoming(zero, drop_zero);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorCelu(const eshkol_operations_t* op) {
    // CELU: max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
    // Default alpha = 1.0
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("celu requires 1 or 2 arguments (tensor [, alpha])");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* alpha;
    if (op->call_op.num_vars == 2) {
        llvm::Value* alpha_tagged = codegenAST(&op->call_op.variables[1]);
        if (!alpha_tagged) return nullptr;
        alpha = tagged_.unpackDouble(alpha_tagged);
    } else {
        alpha = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "celu_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "celu_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "celu_elems");

    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "celu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "celu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "celu_exit", current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, 53, exit_block, "celu");

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "celu_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, i);
    llvm::Value* val_bits = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    llvm::Value* val = builder.CreateBitCast(val_bits, ctx_.doubleType());

    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    // max(0, x)
    llvm::Value* cmp_pos = builder.CreateFCmpOGT(val, zero);
    llvm::Value* pos_part = builder.CreateSelect(cmp_pos, val, zero);

    // min(0, alpha * (exp(x / alpha) - 1))
    llvm::Value* x_over_alpha = builder.CreateFDiv(val, alpha);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {x_over_alpha});
    llvm::Value* exp_minus_1 = builder.CreateFSub(exp_val, one);
    llvm::Value* alpha_exp = builder.CreateFMul(alpha, exp_minus_1);
    llvm::Value* cmp_neg = builder.CreateFCmpOLT(alpha_exp, zero);
    llvm::Value* neg_part = builder.CreateSelect(cmp_neg, alpha_exp, zero);

    // CELU = max(0,x) + min(0, alpha*(exp(x/alpha)-1))
    llvm::Value* result_val = builder.CreateFAdd(pos_part, neg_part);

    llvm::Value* result_bits = builder.CreateBitCast(result_val, ctx_.int64Type());
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, i);
    builder.CreateStore(result_bits, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// ============================================================
// Activation Backward Functions (for autodiff)
// ============================================================

llvm::Value* TensorCodegen::tensorSoftmaxBackward(llvm::Value* softmax_output, llvm::Value* upstream_grad) {
    // Full tensor softmax gradient:
    // dL/dx_i = s_i * (g_i - sum_j(g_j * s_j))
    // where s = softmax output, g = upstream gradient
    //
    // This is the correct Jacobian-vector product for softmax backprop.
    // Much more efficient than computing the full n×n Jacobian matrix.

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack softmax output tensor
    llvm::Value* softmax_ptr_int = tagged_.unpackInt64(softmax_output);
    llvm::Value* softmax_ptr = builder.CreateIntToPtr(softmax_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties from softmax output
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 2);
    llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, softmax_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get gradient elements
    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sm_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "sm_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "sm_back_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Pass 1: Compute dot product sum = sum_j(g_j * s_j)
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "sm_dot_body", current_func);
    llvm::BasicBlock* grad_init = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_init", current_func);

    llvm::Value* dot_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dot_sum);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sm_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, dot_body, grad_init);

    builder.SetInsertPoint(dot_body);
    // Load s_i
    llvm::Value* s_ptr = builder.CreateGEP(ctx_.doubleType(), s_elems, i);
    llvm::Value* s_i = builder.CreateLoad(ctx_.doubleType(), s_ptr);
    // Load g_i
    llvm::Value* gp_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), gp_ptr);
    // Accumulate g_i * s_i
    llvm::Value* prod = builder.CreateFMul(g_i, s_i);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    llvm::Value* new_sum = builder.CreateFAdd(cur_sum, prod);
    builder.CreateStore(new_sum, dot_sum);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(dot_cond);

    // Pass 2: Compute dx_i = s_i * (g_i - dot_product)
    builder.SetInsertPoint(grad_init);
    llvm::BasicBlock* grad_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_cond", current_func);
    llvm::BasicBlock* grad_body = llvm::BasicBlock::Create(ctx_.context(), "sm_grad_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sm_back_exit", current_func);

    llvm::Value* final_dot = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(grad_cond);

    builder.SetInsertPoint(grad_cond);
    llvm::Value* i2 = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp2 = builder.CreateICmpULT(i2, total_elements);
    builder.CreateCondBr(cmp2, grad_body, exit_block);

    builder.SetInsertPoint(grad_body);
    // Load s_i
    llvm::Value* s_ptr2 = builder.CreateGEP(ctx_.doubleType(), s_elems, i2);
    llvm::Value* s_i2 = builder.CreateLoad(ctx_.doubleType(), s_ptr2);
    // Load g_i
    llvm::Value* g_ptr2 = builder.CreateGEP(ctx_.doubleType(), g_elems, i2);
    llvm::Value* g_i2 = builder.CreateLoad(ctx_.doubleType(), g_ptr2);
    // dx_i = s_i * (g_i - dot_product)
    llvm::Value* g_minus_dot = builder.CreateFSub(g_i2, final_dot);
    llvm::Value* dx_i = builder.CreateFMul(s_i2, g_minus_dot);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i2);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i2 = builder.CreateAdd(i2, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i2, counter);
    builder.CreateBr(grad_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorReluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // ReLU backward: dL/dx = dL/dy * (x > 0 ? 1 : 0)

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "relu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "relu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "relu_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "relu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "relu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "relu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "relu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * (x_i > 0 ? 1 : 0)
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* is_positive = builder.CreateFCmpOGT(x_i, zero);
    llvm::Value* mask = builder.CreateSelect(is_positive, one, zero);
    llvm::Value* dx_i = builder.CreateFMul(g_i, mask);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSigmoidBackward(llvm::Value* sigmoid_output, llvm::Value* upstream_grad) {
    // Sigmoid backward: dL/dx = dL/dy * σ(x) * (1 - σ(x))
    // We use the output directly since σ(x) is already computed

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack sigmoid output tensor
    llvm::Value* sig_ptr_int = tagged_.unpackInt64(sigmoid_output);
    llvm::Value* sig_ptr = builder.CreateIntToPtr(sig_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, sig_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, sig_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* s_elems_field = builder.CreateStructGEP(tensor_type, sig_ptr, 2);
    llvm::Value* s_elems = builder.CreateLoad(ctx_.ptrType(), s_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, sig_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "sig_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "sig_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "sig_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "sig_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "sig_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "sig_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sig_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load σ(x)_i
    llvm::Value* s_ptr = builder.CreateGEP(ctx_.doubleType(), s_elems, i);
    llvm::Value* sig_i = builder.CreateLoad(ctx_.doubleType(), s_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * σ(x) * (1 - σ(x))
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_sig = builder.CreateFSub(one, sig_i);
    llvm::Value* sig_deriv = builder.CreateFMul(sig_i, one_minus_sig);
    llvm::Value* dx_i = builder.CreateFMul(g_i, sig_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorGeluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // GELU backward for the tanh approximation used by tensorGelu:
    // gelu(x) = 0.5*x*(1+tanh(u)), u = sqrt(2/pi)*(x + 0.044715*x^3)
    // gelu'(x) = 0.5*(1+tanh(u)) + 0.5*x*(1-tanh(u)^2)*u'

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "gelu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "gelu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "gelu_back_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "gelu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "gelu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);

    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* three = llvm::ConstantFP::get(ctx_.doubleType(), 3.0);
    llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);

    llvm::Value* x_sq = builder.CreateFMul(x_i, x_i);
    llvm::Value* x_cubed = builder.CreateFMul(x_sq, x_i);
    llvm::Value* inner = builder.CreateFAdd(x_i, builder.CreateFMul(coeff, x_cubed));
    llvm::Value* u = builder.CreateFMul(sqrt_2_pi, inner);
    llvm::Value* exp_2u = builder.CreateCall(exp_func, {builder.CreateFMul(two, u)});
    llvm::Value* tanh_u = builder.CreateFDiv(
        builder.CreateFSub(exp_2u, one),
        builder.CreateFAdd(exp_2u, one));
    llvm::Value* tanh_sq = builder.CreateFMul(tanh_u, tanh_u);
    llvm::Value* sech_sq = builder.CreateFSub(one, tanh_sq);
    llvm::Value* inner_prime = builder.CreateFAdd(
        one, builder.CreateFMul(three, builder.CreateFMul(coeff, x_sq)));
    llvm::Value* u_prime = builder.CreateFMul(sqrt_2_pi, inner_prime);
    llvm::Value* first = builder.CreateFMul(half, builder.CreateFAdd(one, tanh_u));
    llvm::Value* second = builder.CreateFMul(
        half, builder.CreateFMul(x_i, builder.CreateFMul(sech_sq, u_prime)));
    llvm::Value* gelu_deriv = builder.CreateFAdd(first, second);

    // dx_i = g_i * gelu'(x)
    llvm::Value* dx_i = builder.CreateFMul(g_i, gelu_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorLeakyReluBackward(llvm::Value* input, llvm::Value* upstream_grad, double alpha) {
    // Leaky ReLU backward: dL/dx = dL/dy * (x > 0 ? 1 : alpha)

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "lrelu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "lrelu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "lrelu_back_elems");

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "lrelu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "lrelu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);
    // dx_i = g_i * (x_i > 0 ? 1 : alpha)
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* alpha_val = llvm::ConstantFP::get(ctx_.doubleType(), alpha);
    llvm::Value* is_positive = builder.CreateFCmpOGT(x_i, zero);
    llvm::Value* mask = builder.CreateSelect(is_positive, one, alpha_val);
    llvm::Value* dx_i = builder.CreateFMul(g_i, mask);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorSiluBackward(llvm::Value* input, llvm::Value* upstream_grad) {
    // SiLU backward: dL/dx = dL/dy * (σ(x) + x * σ(x) * (1 - σ(x)))
    //                      = dL/dy * σ(x) * (1 + x * (1 - σ(x)))

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack upstream gradient tensor
    llvm::Value* grad_ptr_int = tagged_.unpackInt64(upstream_grad);
    llvm::Value* grad_ptr = builder.CreateIntToPtr(grad_ptr_int, ctx_.ptrType());

    // Get tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Value* g_elems_field = builder.CreateStructGEP(tensor_type, grad_ptr, 2);
    llvm::Value* g_elems = builder.CreateLoad(ctx_.ptrType(), g_elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "silu_back_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "silu_back_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);

    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "silu_back_elems");

    // Get exp function
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* exp_type = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(exp_type, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "silu_back_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "silu_back_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "silu_back_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "silu_back_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    llvm::Value* cmp = builder.CreateICmpULT(i, total_elements);
    builder.CreateCondBr(cmp, loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Load x_i
    llvm::Value* x_ptr = builder.CreateGEP(ctx_.doubleType(), x_elems, i);
    llvm::Value* x_i = builder.CreateLoad(ctx_.doubleType(), x_ptr);
    // Load g_i
    llvm::Value* g_ptr = builder.CreateGEP(ctx_.doubleType(), g_elems, i);
    llvm::Value* g_i = builder.CreateLoad(ctx_.doubleType(), g_ptr);

    // Compute σ(x)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* neg_x = builder.CreateFNeg(x_i);
    llvm::Value* exp_neg = builder.CreateCall(exp_func, {neg_x});
    llvm::Value* denom = builder.CreateFAdd(one, exp_neg);
    llvm::Value* sigma = builder.CreateFDiv(one, denom);

    // silu'(x) = σ(x) * (1 + x * (1 - σ(x)))
    llvm::Value* one_minus_sigma = builder.CreateFSub(one, sigma);
    llvm::Value* x_times_deriv = builder.CreateFMul(x_i, one_minus_sigma);
    llvm::Value* inner = builder.CreateFAdd(one, x_times_deriv);
    llvm::Value* silu_deriv = builder.CreateFMul(sigma, inner);

    // dx_i = g_i * silu'(x)
    llvm::Value* dx_i = builder.CreateFMul(g_i, silu_deriv);
    // Store result
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
    builder.CreateStore(dx_i, dst_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, counter);
    builder.CreateBr(loop_cond);

    // Populate result tensor
    builder.SetInsertPoint(exit_block);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
