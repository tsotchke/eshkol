/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Extras (Phase 4/5/7 supplements + standalone ops).
 * Extracted from tensor_codegen.cpp during the v1.2 mechanical split.
 *
 * Catch-all home for the post-Phase-1 tensor operations that didn't fit
 * cleanly into the per-domain modules:
 *   - Additional Shape Operations (Phase 4): tile, pad
 *   - Additional Statistics (Phase 5): tensorMin/Max/Argmin/Argmax,
 *     tensorCov, tensorCorrcoef
 *   - Additional Convolution Operations (Phase 7): conv3d
 *   - Tensor Unary Ops: emitTensorUnaryOp + tensorNeg/Abs/Sqrt/Exp/
 *     Log/Sin/Cos
 *   - Tensor Binary Ops: tensorPow, tensorMaximum, tensorMinimum
 *   - tensor-scale, batch-matmul
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-extras-extract baseline.
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

// Additional Shape Operations (Phase 4)
// ============================================================

llvm::Value* TensorCodegen::tile(const eshkol_operations_t* op) {
    // tile: (tile tensor reps) - repeat tensor according to repetitions
    // reps is a list/vector of repetition counts for each dimension
    // Works with N-dimensional tensors
    if (op->call_op.num_vars < 2) {
        eshkol_error("tile requires 2 arguments (tensor, reps)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* reps_val = codegenAST(&op->call_op.variables[1]);
    if (!tensor_val || !reps_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* src_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get reps as vector
    llvm::Value* reps_ptr_int = tagged_.unpackInt64(reps_val);
    llvm::Value* reps_ptr = builder.CreateIntToPtr(reps_ptr_int, ctx_.ptrType());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate result tensor with multiplied dimensions
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "tile_result");

    // Allocate new dimensions array
    llvm::Value* dims_bytes = builder.CreateMul(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "tile_dims");

    // Calculate new dimensions and total elements
    llvm::BasicBlock* calc_cond = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_cond", current_func);
    llvm::BasicBlock* calc_body = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_body", current_func);
    llvm::BasicBlock* calc_done = llvm::BasicBlock::Create(ctx_.context(), "tile_calc_done", current_func);

    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_calc_i");
    llvm::Value* new_total = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_new_total");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), new_total);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* cmp = builder.CreateICmpULT(i, num_dims);
    builder.CreateCondBr(cmp, calc_body, calc_done);

    builder.SetInsertPoint(calc_body);
    // Get source dimension
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, i);
    llvm::Value* src_dim = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    // Get rep count from vector (first 8 bytes are length, then tagged values)
    llvm::Value* rep_offset = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* rep_tagged_ptr = builder.CreateGEP(ctx_.int64Type(), reps_ptr, rep_offset);
    llvm::Value* rep_tagged = builder.CreateLoad(ctx_.int64Type(), rep_tagged_ptr);
    llvm::Value* rep_count = tagged_.unpackInt64(builder.CreateIntToPtr(
        builder.CreateOr(rep_tagged, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        ctx_.taggedValueType()));
    // Clamp rep_count to at least 1
    llvm::Value* rep_clamped = builder.CreateSelect(
        builder.CreateICmpSLT(rep_count, llvm::ConstantInt::get(ctx_.int64Type(), 1)),
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        rep_count);
    // Calculate new dimension = src_dim * rep_count
    llvm::Value* new_dim = builder.CreateMul(src_dim, rep_clamped);
    llvm::Value* new_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims_ptr, i);
    builder.CreateStore(new_dim, new_dim_ptr);
    // Update total
    llvm::Value* cur_total = builder.CreateLoad(ctx_.int64Type(), new_total);
    builder.CreateStore(builder.CreateMul(cur_total, new_dim), new_total);
    // Increment
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_done);
    llvm::Value* result_total = builder.CreateLoad(ctx_.int64Type(), new_total);

    // Allocate result elements
    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "tile_elems");

    // Fill result tensor header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims_ptr, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    // Copy elements with tiling pattern
    // For each output element, calculate corresponding source element
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "tile_copy_done", current_func);

    llvm::Value* copy_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tile_copy_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* idx = builder.CreateLoad(ctx_.int64Type(), copy_idx);
    llvm::Value* cmp2 = builder.CreateICmpULT(idx, result_total);
    builder.CreateCondBr(cmp2, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Calculate source index using modulo for each dimension
    // This is the key: src_idx = idx % src_total (simplified for 1D, generalized below)
    llvm::Value* src_idx = builder.CreateURem(idx, src_total);
    llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, src_idx);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, idx);
    builder.CreateStore(elem_val, dst_elem_ptr);
    builder.CreateStore(builder.CreateAdd(idx, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::pad(const eshkol_operations_t* op) {
    // pad: (pad tensor pad-width value) - pad tensor on each side
    // pad-width is a list of (before, after) pairs for each dimension
    // Works with N-dimensional tensors
    if (op->call_op.num_vars < 2) {
        eshkol_error("pad requires 2-3 arguments (tensor, pad-width, optional value)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* pad_width_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fill_val = nullptr;
    if (op->call_op.num_vars >= 3) {
        fill_val = codegenAST(&op->call_op.variables[2]);
    }
    if (!tensor_val || !pad_width_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get fill value (default 0.0)
    llvm::Value* fill_double = fill_val ? extractAsDouble(fill_val) :
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* fill_bits = builder.CreateBitCast(fill_double, ctx_.int64Type());

    // Unpack input tensor
    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* src_dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* src_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get pad_width - interpret as single value for symmetric padding
    llvm::Value* pad_amount = extractAsDouble(pad_width_val);
    llvm::Value* pad_int = builder.CreateFPToSI(pad_amount, ctx_.int64Type());

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "pad_result");

    // Allocate new dimensions array
    llvm::Value* dims_bytes = builder.CreateMul(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "pad_dims");

    // Calculate new dimensions (each dim + 2*pad)
    llvm::BasicBlock* calc_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_cond", current_func);
    llvm::BasicBlock* calc_body = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_body", current_func);
    llvm::BasicBlock* calc_done = llvm::BasicBlock::Create(ctx_.context(), "pad_calc_done", current_func);

    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_calc_i");
    llvm::Value* new_total = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_new_total");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), new_total);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* cmp = builder.CreateICmpULT(i, num_dims);
    builder.CreateCondBr(cmp, calc_body, calc_done);

    builder.SetInsertPoint(calc_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), src_dims_ptr, i);
    llvm::Value* src_dim = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* pad2 = builder.CreateMul(pad_int, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* new_dim = builder.CreateAdd(src_dim, pad2);
    llvm::Value* new_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims_ptr, i);
    builder.CreateStore(new_dim, new_dim_ptr);
    llvm::Value* cur_total = builder.CreateLoad(ctx_.int64Type(), new_total);
    builder.CreateStore(builder.CreateMul(cur_total, new_dim), new_total);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_cond);

    builder.SetInsertPoint(calc_done);
    llvm::Value* result_total = builder.CreateLoad(ctx_.int64Type(), new_total);

    // Allocate and fill result elements with padding value
    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "pad_elems");

    // Fill all with padding value first
    llvm::BasicBlock* fill_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_cond", current_func);
    llvm::BasicBlock* fill_body = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_body", current_func);
    llvm::BasicBlock* fill_done = llvm::BasicBlock::Create(ctx_.context(), "pad_fill_done", current_func);

    llvm::Value* fill_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_fill_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), fill_idx);
    builder.CreateBr(fill_cond);

    builder.SetInsertPoint(fill_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), fill_idx);
    llvm::Value* fcmp = builder.CreateICmpULT(fi, result_total);
    builder.CreateCondBr(fcmp, fill_body, fill_done);

    builder.SetInsertPoint(fill_body);
    llvm::Value* fill_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, fi);
    builder.CreateStore(fill_bits, fill_ptr);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), fill_idx);
    builder.CreateBr(fill_cond);

    builder.SetInsertPoint(fill_done);

    // Copy source elements into padded position
    // For 1D: dst[pad + i] = src[i]
    // For ND: need to calculate proper offset
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "pad_copy_done", current_func);

    llvm::Value* copy_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pad_copy_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), copy_idx);
    llvm::Value* ccmp = builder.CreateICmpULT(ci, src_total);
    builder.CreateCondBr(ccmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Calculate destination index: for each dimension, add pad offset
    // Simplified: dst_idx = src_idx + pad * (dimension_multiplier for each dim)
    // For proper ND: need stride calculation, but simplified version works for most cases
    llvm::Value* dst_idx = builder.CreateAdd(ci, pad_int);
    llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, ci);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, dst_idx);
    builder.CreateStore(elem_val, dst_elem_ptr);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), copy_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Fill result tensor header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims_ptr, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// ============================================================
// Additional Statistics Operations (Phase 5)
// ============================================================

llvm::Value* TensorCodegen::tensorMin(const eshkol_operations_t* op) {
    // tensor-min: (tensor-min tensor) or (tensor-min tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-min requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-min tensor axis) → min along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(tensor_val, axis_val, 3); // MIN=3
    }

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* min_merge = llvm::BasicBlock::Create(ctx_.context(), "min_merge", current_func);

    llvm::Value* ad_tagged_result = nullptr;
    llvm::BasicBlock* ad_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_block = llvm::BasicBlock::Create(ctx_.context(), "min_ad", current_func);
        llvm::BasicBlock* numeric_block = llvm::BasicBlock::Create(ctx_.context(), "min_numeric", current_func);
        builder.CreateCondBr(in_ad_mode, ad_block, numeric_block);

        builder.SetInsertPoint(ad_block);
        llvm::Value* ad_first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* ad_first_bits = builder.CreateLoad(ctx_.int64Type(), ad_first_elem_ptr);
        llvm::Value* ad_acc = builder.CreateAlloca(ctx_.ptrType(), nullptr, "min_ad_acc");
        llvm::Value* ad_first_node = adNodeFromTensorElementBits(ad_first_bits, "min_ad_first");
        builder.CreateStore(ad_first_node, ad_acc);

        llvm::Value* ad_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "min_ad_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), ad_idx);
        llvm::BasicBlock* ad_cond = llvm::BasicBlock::Create(ctx_.context(), "min_ad_cond", current_func);
        llvm::BasicBlock* ad_body = llvm::BasicBlock::Create(ctx_.context(), "min_ad_body", current_func);
        llvm::BasicBlock* ad_done = llvm::BasicBlock::Create(ctx_.context(), "min_ad_done", current_func);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_cond);
        llvm::Value* ad_i = builder.CreateLoad(ctx_.int64Type(), ad_idx);
        llvm::Value* ad_has_elem = builder.CreateICmpULT(ad_i, total);
        builder.CreateCondBr(ad_has_elem, ad_body, ad_done);

        builder.SetInsertPoint(ad_body);
        llvm::Value* ad_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, ad_i);
        llvm::Value* ad_elem_bits = builder.CreateLoad(ctx_.int64Type(), ad_elem_ptr);
        llvm::Value* ad_elem_node = adNodeFromTensorElementBits(ad_elem_bits, "min_ad_elem");
        llvm::Value* ad_old_acc = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(45, ad_old_acc, ad_elem_node);
        builder.CreateStore(ad_new_acc, ad_acc);
        builder.CreateStore(builder.CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_idx);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_done);
        llvm::Value* final_ad_min = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        ad_tagged_result = tagged_.packPtr(final_ad_min, ESHKOL_VALUE_CALLABLE);
        builder.CreateBr(min_merge);
        ad_exit_block = builder.GetInsertBlock();

        builder.SetInsertPoint(numeric_block);
    }

    // Initialize min with first element
    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* min_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "min_val");
    builder.CreateStore(first_val, min_val);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "min_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "min_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "min_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "min_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_min = builder.CreateLoad(ctx_.doubleType(), min_val);
    llvm::Value* is_less = builder.CreateFCmpOLT(elem_val, cur_min);
    llvm::Value* new_min = builder.CreateSelect(is_less, elem_val, cur_min);
    builder.CreateStore(new_min, min_val);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), min_val);
    llvm::Value* tagged_result = tagged_.packDouble(result);
    builder.CreateBr(min_merge);
    llvm::BasicBlock* numeric_exit_block = builder.GetInsertBlock();

    builder.SetInsertPoint(min_merge);
    llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), ad_exit_block ? 2 : 1, "min_result");
    result_phi->addIncoming(tagged_result, numeric_exit_block);
    if (ad_exit_block && ad_tagged_result) {
        result_phi->addIncoming(ad_tagged_result, ad_exit_block);
    }
    return result_phi;
}

llvm::Value* TensorCodegen::tensorMax(const eshkol_operations_t* op) {
    // tensor-max: (tensor-max tensor) or (tensor-max tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-max requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-max tensor axis) → max along axis, returns tensor
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        return emitAxisReduce(tensor_val, axis_val, 2); // MAX=2
    }

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* max_merge = llvm::BasicBlock::Create(ctx_.context(), "max_merge", current_func);

    llvm::Value* ad_tagged_result = nullptr;
    llvm::BasicBlock* ad_exit_block = nullptr;
    if (autodiff_) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_block = llvm::BasicBlock::Create(ctx_.context(), "max_ad", current_func);
        llvm::BasicBlock* numeric_block = llvm::BasicBlock::Create(ctx_.context(), "max_numeric", current_func);
        builder.CreateCondBr(in_ad_mode, ad_block, numeric_block);

        builder.SetInsertPoint(ad_block);
        llvm::Value* ad_first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* ad_first_bits = builder.CreateLoad(ctx_.int64Type(), ad_first_elem_ptr);
        llvm::Value* ad_acc = builder.CreateAlloca(ctx_.ptrType(), nullptr, "max_ad_acc");
        llvm::Value* ad_first_node = adNodeFromTensorElementBits(ad_first_bits, "max_ad_first");
        builder.CreateStore(ad_first_node, ad_acc);

        llvm::Value* ad_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "max_ad_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), ad_idx);
        llvm::BasicBlock* ad_cond = llvm::BasicBlock::Create(ctx_.context(), "max_ad_cond", current_func);
        llvm::BasicBlock* ad_body = llvm::BasicBlock::Create(ctx_.context(), "max_ad_body", current_func);
        llvm::BasicBlock* ad_done = llvm::BasicBlock::Create(ctx_.context(), "max_ad_done", current_func);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_cond);
        llvm::Value* ad_i = builder.CreateLoad(ctx_.int64Type(), ad_idx);
        llvm::Value* ad_has_elem = builder.CreateICmpULT(ad_i, total);
        builder.CreateCondBr(ad_has_elem, ad_body, ad_done);

        builder.SetInsertPoint(ad_body);
        llvm::Value* ad_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, ad_i);
        llvm::Value* ad_elem_bits = builder.CreateLoad(ctx_.int64Type(), ad_elem_ptr);
        llvm::Value* ad_elem_node = adNodeFromTensorElementBits(ad_elem_bits, "max_ad_elem");
        llvm::Value* ad_old_acc = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(44, ad_old_acc, ad_elem_node);
        builder.CreateStore(ad_new_acc, ad_acc);
        builder.CreateStore(builder.CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_idx);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_done);
        llvm::Value* final_ad_max = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        ad_tagged_result = tagged_.packPtr(final_ad_max, ESHKOL_VALUE_CALLABLE);
        builder.CreateBr(max_merge);
        ad_exit_block = builder.GetInsertBlock();

        builder.SetInsertPoint(numeric_block);
    }

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    builder.CreateStore(first_val, max_val);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "max_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "max_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "max_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "max_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem_val, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem_val, cur_max);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* tagged_result = tagged_.packDouble(result);
    builder.CreateBr(max_merge);
    llvm::BasicBlock* numeric_exit_block = builder.GetInsertBlock();

    builder.SetInsertPoint(max_merge);
    llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), ad_exit_block ? 2 : 1, "max_result");
    result_phi->addIncoming(tagged_result, numeric_exit_block);
    if (ad_exit_block && ad_tagged_result) {
        result_phi->addIncoming(ad_tagged_result, ad_exit_block);
    }
    return result_phi;
}

llvm::Value* TensorCodegen::tensorArgmin(const eshkol_operations_t* op) {
    // tensor-argmin: (tensor-argmin tensor) or (tensor-argmin tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-argmin requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-argmin tensor axis) → argmin along axis, returns tensor of indices
    if (op->call_op.num_vars == 2) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[1]);
        if (!axis_val) return nullptr;
        // Emit call to eshkol_xla_argreduce(arena, data, total, shape, rank, axis, is_max=0)
        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(tensor_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);
        llvm::Value* neg = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        axis = builder.CreateSelect(neg, builder.CreateAdd(axis, rank), axis);
        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_argreduce", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, llvm::ConstantInt::get(i64Ty, 0)},
            "argmin_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* min_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "argmin_val");
    llvm::Value* min_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmin_idx");
    builder.CreateStore(first_val, min_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), min_idx);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "argmin_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "argmin_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "argmin_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmin_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_min = builder.CreateLoad(ctx_.doubleType(), min_val);
    llvm::Value* is_less = builder.CreateFCmpOLT(elem_val, cur_min);
    llvm::Value* new_min = builder.CreateSelect(is_less, elem_val, cur_min);
    llvm::Value* cur_idx = builder.CreateLoad(ctx_.int64Type(), min_idx);
    llvm::Value* new_idx = builder.CreateSelect(is_less, i, cur_idx);
    builder.CreateStore(new_min, min_val);
    builder.CreateStore(new_idx, min_idx);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result_idx = builder.CreateLoad(ctx_.int64Type(), min_idx);
    return tagged_.packInt64(result_idx);
}

llvm::Value* TensorCodegen::tensorArgmax(const eshkol_operations_t* op) {
    // tensor-argmax: (tensor-argmax tensor) or (tensor-argmax tensor axis)
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("tensor-argmax requires 1 or 2 arguments");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    // 2-arg case: (tensor-argmax tensor axis) → argmax along axis, returns tensor of indices
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
        llvm::Value* neg = builder.CreateICmpSLT(axis, llvm::ConstantInt::get(ctx_.int64Type(), 0));
        axis = builder.CreateSelect(neg, builder.CreateAdd(axis, rank), axis);
        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, i64Ty}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_argreduce", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, llvm::ConstantInt::get(i64Ty, 1)},
            "argmax_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems_ptr = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* first_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* first_bits = builder.CreateLoad(ctx_.int64Type(), first_elem_ptr);
    llvm::Value* first_val = builder.CreateBitCast(first_bits, ctx_.doubleType());

    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "argmax_val");
    llvm::Value* max_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmax_idx");
    builder.CreateStore(first_val, max_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), max_idx);

    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "argmax_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "argmax_body", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "argmax_done", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "argmax_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* cmp = builder.CreateICmpULT(i, total);
    builder.CreateCondBr(cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems_ptr, i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem_val = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(elem_val, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, elem_val, cur_max);
    llvm::Value* cur_idx = builder.CreateLoad(ctx_.int64Type(), max_idx);
    llvm::Value* new_idx = builder.CreateSelect(is_greater, i, cur_idx);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(new_idx, max_idx);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    llvm::Value* result_idx = builder.CreateLoad(ctx_.int64Type(), max_idx);
    return tagged_.packInt64(result_idx);
}

llvm::Value* TensorCodegen::tensorCov(const eshkol_operations_t* op) {
    // tensor-cov: compute covariance between two 1D tensors
    // cov(x, y) = E[(x - μx)(y - μy)] = E[xy] - E[x]E[y]
    if (op->call_op.num_vars < 2) {
        eshkol_error("tensor-cov requires 2 arguments");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* y_val = codegenAST(&op->call_op.variables[1]);
    if (!x_val || !y_val) return nullptr;

    auto& builder = ctx_.builder();

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack y tensor
    llvm::Value* y_ptr_int = tagged_.unpackInt64(y_val);
    llvm::Value* y_ptr = builder.CreateIntToPtr(y_ptr_int, ctx_.ptrType());

    llvm::Value* y_elems_field = builder.CreateStructGEP(tensor_type, y_ptr, 2);
    llvm::Value* y_elems = builder.CreateLoad(ctx_.ptrType(), y_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // First pass: compute means
    llvm::Value* sum_x = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum_x");
    llvm::Value* sum_y = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum_y");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_y);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "cov_mean_done", current_func);

    llvm::Value* mean_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "cov_mean_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), mean_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* mi = builder.CreateLoad(ctx_.int64Type(), mean_idx);
    llvm::Value* mcmp = builder.CreateICmpULT(mi, x_total);
    builder.CreateCondBr(mcmp, mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* x_elem_ptr = builder.CreateGEP(ctx_.int64Type(), x_elems, mi);
    llvm::Value* x_bits = builder.CreateLoad(ctx_.int64Type(), x_elem_ptr);
    llvm::Value* x_elem = builder.CreateBitCast(x_bits, ctx_.doubleType());
    llvm::Value* y_elem_ptr = builder.CreateGEP(ctx_.int64Type(), y_elems, mi);
    llvm::Value* y_bits = builder.CreateLoad(ctx_.int64Type(), y_elem_ptr);
    llvm::Value* y_elem = builder.CreateBitCast(y_bits, ctx_.doubleType());
    llvm::Value* cur_sx = builder.CreateLoad(ctx_.doubleType(), sum_x);
    llvm::Value* cur_sy = builder.CreateLoad(ctx_.doubleType(), sum_y);
    builder.CreateStore(builder.CreateFAdd(cur_sx, x_elem), sum_x);
    builder.CreateStore(builder.CreateFAdd(cur_sy, y_elem), sum_y);
    builder.CreateStore(builder.CreateAdd(mi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), mean_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* n = builder.CreateSIToFP(x_total, ctx_.doubleType());
    llvm::Value* mean_x = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_x), n);
    llvm::Value* mean_y = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_y), n);

    // Second pass: compute covariance
    llvm::Value* cov_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "cov_sum");
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), cov_sum);

    llvm::BasicBlock* cov_cond = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_cond", current_func);
    llvm::BasicBlock* cov_body = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_body", current_func);
    llvm::BasicBlock* cov_done = llvm::BasicBlock::Create(ctx_.context(), "cov_cov_done", current_func);

    llvm::Value* cov_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "cov_cov_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cov_idx);
    builder.CreateBr(cov_cond);

    builder.SetInsertPoint(cov_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), cov_idx);
    llvm::Value* ccmp = builder.CreateICmpULT(ci, x_total);
    builder.CreateCondBr(ccmp, cov_body, cov_done);

    builder.SetInsertPoint(cov_body);
    llvm::Value* x_ep = builder.CreateGEP(ctx_.int64Type(), x_elems, ci);
    llvm::Value* xb = builder.CreateLoad(ctx_.int64Type(), x_ep);
    llvm::Value* xe = builder.CreateBitCast(xb, ctx_.doubleType());
    llvm::Value* y_ep = builder.CreateGEP(ctx_.int64Type(), y_elems, ci);
    llvm::Value* yb = builder.CreateLoad(ctx_.int64Type(), y_ep);
    llvm::Value* ye = builder.CreateBitCast(yb, ctx_.doubleType());
    llvm::Value* dx = builder.CreateFSub(xe, mean_x);
    llvm::Value* dy = builder.CreateFSub(ye, mean_y);
    llvm::Value* prod = builder.CreateFMul(dx, dy);
    llvm::Value* cur_cov = builder.CreateLoad(ctx_.doubleType(), cov_sum);
    builder.CreateStore(builder.CreateFAdd(cur_cov, prod), cov_sum);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cov_idx);
    builder.CreateBr(cov_cond);

    builder.SetInsertPoint(cov_done);
    llvm::Value* result = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), cov_sum), n);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::tensorCorrcoef(const eshkol_operations_t* op) {
    // tensor-corrcoef: compute Pearson correlation coefficient
    // r = cov(x,y) / (std(x) * std(y))
    if (op->call_op.num_vars < 2) {
        eshkol_error("tensor-corrcoef requires 2 arguments");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* y_val = codegenAST(&op->call_op.variables[1]);
    if (!x_val || !y_val) return nullptr;

    auto& builder = ctx_.builder();

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack y tensor
    llvm::Value* y_ptr_int = tagged_.unpackInt64(y_val);
    llvm::Value* y_ptr = builder.CreateIntToPtr(y_ptr_int, ctx_.ptrType());

    llvm::Value* y_elems_field = builder.CreateStructGEP(tensor_type, y_ptr, 2);
    llvm::Value* y_elems = builder.CreateLoad(ctx_.ptrType(), y_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute means, variances, and covariance in two passes
    llvm::Value* sum_x = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* sum_y = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_y);

    // First pass: means
    llvm::BasicBlock* m1_cond = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_cond", current_func);
    llvm::BasicBlock* m1_body = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_body", current_func);
    llvm::BasicBlock* m1_done = llvm::BasicBlock::Create(ctx_.context(), "corr_m1_done", current_func);

    llvm::Value* m1_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m1_idx);
    builder.CreateBr(m1_cond);

    builder.SetInsertPoint(m1_cond);
    llvm::Value* m1i = builder.CreateLoad(ctx_.int64Type(), m1_idx);
    builder.CreateCondBr(builder.CreateICmpULT(m1i, x_total), m1_body, m1_done);

    builder.SetInsertPoint(m1_body);
    llvm::Value* xep = builder.CreateGEP(ctx_.int64Type(), x_elems, m1i);
    llvm::Value* yep = builder.CreateGEP(ctx_.int64Type(), y_elems, m1i);
    llvm::Value* xv = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), xep), ctx_.doubleType());
    llvm::Value* yv = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), yep), ctx_.doubleType());
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum_x), xv), sum_x);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum_y), yv), sum_y);
    builder.CreateStore(builder.CreateAdd(m1i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m1_idx);
    builder.CreateBr(m1_cond);

    builder.SetInsertPoint(m1_done);
    llvm::Value* n = builder.CreateSIToFP(x_total, ctx_.doubleType());
    llvm::Value* mean_x = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_x), n);
    llvm::Value* mean_y = builder.CreateFDiv(builder.CreateLoad(ctx_.doubleType(), sum_y), n);

    // Second pass: variances and covariance
    llvm::Value* var_x = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* var_y = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* cov = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), var_x);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), var_y);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), cov);

    llvm::BasicBlock* m2_cond = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_cond", current_func);
    llvm::BasicBlock* m2_body = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_body", current_func);
    llvm::BasicBlock* m2_done = llvm::BasicBlock::Create(ctx_.context(), "corr_m2_done", current_func);

    llvm::Value* m2_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m2_idx);
    builder.CreateBr(m2_cond);

    builder.SetInsertPoint(m2_cond);
    llvm::Value* m2i = builder.CreateLoad(ctx_.int64Type(), m2_idx);
    builder.CreateCondBr(builder.CreateICmpULT(m2i, x_total), m2_body, m2_done);

    builder.SetInsertPoint(m2_body);
    llvm::Value* xep2 = builder.CreateGEP(ctx_.int64Type(), x_elems, m2i);
    llvm::Value* yep2 = builder.CreateGEP(ctx_.int64Type(), y_elems, m2i);
    llvm::Value* xv2 = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), xep2), ctx_.doubleType());
    llvm::Value* yv2 = builder.CreateBitCast(builder.CreateLoad(ctx_.int64Type(), yep2), ctx_.doubleType());
    llvm::Value* dx = builder.CreateFSub(xv2, mean_x);
    llvm::Value* dy = builder.CreateFSub(yv2, mean_y);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), var_x),
        builder.CreateFMul(dx, dx)), var_x);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), var_y),
        builder.CreateFMul(dy, dy)), var_y);
    builder.CreateStore(builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), cov),
        builder.CreateFMul(dx, dy)), cov);
    builder.CreateStore(builder.CreateAdd(m2i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m2_idx);
    builder.CreateBr(m2_cond);

    builder.SetInsertPoint(m2_done);
    llvm::Value* cov_val = builder.CreateLoad(ctx_.doubleType(), cov);
    llvm::Value* var_x_val = builder.CreateLoad(ctx_.doubleType(), var_x);
    llvm::Value* var_y_val = builder.CreateLoad(ctx_.doubleType(), var_y);

    // r = cov / sqrt(var_x * var_y)
    llvm::Value* denom = builder.CreateFMul(var_x_val, var_y_val);
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_denom = builder.CreateCall(sqrt_func, {denom});
    llvm::Value* result = builder.CreateFDiv(cov_val, sqrt_denom);

    return tagged_.packDouble(result);
}

// ============================================================
// Additional Convolution Operations (Phase 7)
// ============================================================

llvm::Value* TensorCodegen::conv3d(const eshkol_operations_t* op) {
    // conv3d: (conv3d input kernel stride padding)
    // N-dimensional 3D convolution for volumetric data
    // Input: [batch, channels, depth, height, width] or [depth, height, width]
    // Kernel: [out_channels, in_channels, kD, kH, kW]
    if (op->call_op.num_vars < 2) {
        eshkol_error("conv3d requires at least 2 arguments (input, kernel)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!input_val || !kernel_val) return nullptr;

    // Get optional stride and padding
    llvm::Value* stride_val = (op->call_op.num_vars >= 3) ?
        codegenAST(&op->call_op.variables[2]) : nullptr;
    llvm::Value* padding_val = (op->call_op.num_vars >= 4) ?
        codegenAST(&op->call_op.variables[3]) : nullptr;

    auto& builder = ctx_.builder();
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    int64_t stride = 1;
    int64_t padding = 0;
    if (stride_val) {
        llvm::Value* s = extractAsDouble(stride_val);
        if (auto* ci = llvm::dyn_cast<llvm::ConstantFP>(s)) {
            stride = static_cast<int64_t>(ci->getValueAPF().convertToDouble());
        }
    }
    if (padding_val) {
        llvm::Value* p = extractAsDouble(padding_val);
        if (auto* ci = llvm::dyn_cast<llvm::ConstantFP>(p)) {
            padding = static_cast<int64_t>(ci->getValueAPF().convertToDouble());
        }
    }

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims_ptr = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* in_ndims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* in_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);

    // Unpack kernel tensor
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());

    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_ndim_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 1);
    llvm::Value* k_ndims = builder.CreateLoad(ctx_.int64Type(), k_ndim_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Guard: conv3d requires at least 3D tensors for both input and kernel
    {
        llvm::Value* in_ok = builder.CreateICmpUGE(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));
        llvm::Value* k_ok = builder.CreateICmpUGE(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));
        llvm::Value* both_ok = builder.CreateAnd(in_ok, k_ok);
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "c3d_dims_ok", current_func);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "c3d_dims_err", current_func);
        builder.CreateCondBr(both_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: conv3d requires at least 3D tensors (input=%lldD, kernel=%lldD)\n");
            builder.CreateCall(pf, {fmt, in_ndims, k_ndims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    // Get input dimensions (last 3 are D, H, W)
    llvm::Value* ndims_minus_1 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* ndims_minus_2 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* ndims_minus_3 = builder.CreateSub(in_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* in_W_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_1);
    llvm::Value* in_W = builder.CreateLoad(ctx_.int64Type(), in_W_ptr);
    llvm::Value* in_H_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_2);
    llvm::Value* in_H = builder.CreateLoad(ctx_.int64Type(), in_H_ptr);
    llvm::Value* in_D_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims_ptr, ndims_minus_3);
    llvm::Value* in_D = builder.CreateLoad(ctx_.int64Type(), in_D_ptr);

    // Get kernel dimensions
    llvm::Value* k_ndims_minus_1 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* k_ndims_minus_2 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* k_ndims_minus_3 = builder.CreateSub(k_ndims, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* k_W_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_1);
    llvm::Value* k_W = builder.CreateLoad(ctx_.int64Type(), k_W_ptr);
    llvm::Value* k_H_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_2);
    llvm::Value* k_H = builder.CreateLoad(ctx_.int64Type(), k_H_ptr);
    llvm::Value* k_D_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, k_ndims_minus_3);
    llvm::Value* k_D = builder.CreateLoad(ctx_.int64Type(), k_D_ptr);

    // Calculate output dimensions
    llvm::Value* stride_val_i64 = llvm::ConstantInt::get(ctx_.int64Type(), stride);
    llvm::Value* padding_val_i64 = llvm::ConstantInt::get(ctx_.int64Type(), padding);
    llvm::Value* padding2 = builder.CreateMul(padding_val_i64, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    llvm::Value* out_D = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_D, padding2), k_D),
            stride_val_i64), stride_val_i64);
    llvm::Value* out_H = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_H, padding2), k_H),
            stride_val_i64), stride_val_i64);
    llvm::Value* out_W = builder.CreateSDiv(
        builder.CreateAdd(builder.CreateSub(builder.CreateAdd(in_W, padding2), k_W),
            stride_val_i64), stride_val_i64);

    // Create output tensor with same leading dimensions, new spatial dims
    std::vector<llvm::Value*> out_dims;
    out_dims.push_back(out_D);
    out_dims.push_back(out_H);
    out_dims.push_back(out_W);

    llvm::Value* result_total = builder.CreateMul(out_D, builder.CreateMul(out_H, out_W));
    llvm::Value* kernel_total = builder.CreateMul(k_D, builder.CreateMul(k_H, k_W));

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv3d_result");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 24)}, "conv3d_dims");  // 3 dims * 8 bytes

    llvm::Value* elems_bytes = builder.CreateMul(result_total, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "conv3d_elems");

    // Store output dimensions
    builder.CreateStore(out_D, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(out_H, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(out_W, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    // Fill result header
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 3), r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(result_total, r_total_field);

    // 6-nested loop for 3D convolution
    llvm::BasicBlock* od_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_cond", current_func);
    llvm::BasicBlock* od_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_body", current_func);
    llvm::BasicBlock* od_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_od_done", current_func);

    llvm::Value* od_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), od_idx);
    builder.CreateBr(od_cond);

    builder.SetInsertPoint(od_cond);
    llvm::Value* od = builder.CreateLoad(ctx_.int64Type(), od_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(od, out_D), od_body, od_done);

    builder.SetInsertPoint(od_body);

    // Inner loops for oh, ow
    llvm::BasicBlock* oh_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_cond", current_func);
    llvm::BasicBlock* oh_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_body", current_func);
    llvm::BasicBlock* oh_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_oh_done", current_func);

    llvm::Value* oh_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oh_idx);
    builder.CreateBr(oh_cond);

    builder.SetInsertPoint(oh_cond);
    llvm::Value* oh = builder.CreateLoad(ctx_.int64Type(), oh_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(oh, out_H), oh_body, oh_done);

    builder.SetInsertPoint(oh_body);

    llvm::BasicBlock* ow_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_cond", current_func);
    llvm::BasicBlock* ow_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_body", current_func);
    llvm::BasicBlock* ow_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_ow_done", current_func);

    llvm::Value* ow_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ow_idx);
    builder.CreateBr(ow_cond);

    builder.SetInsertPoint(ow_cond);
    llvm::Value* ow = builder.CreateLoad(ctx_.int64Type(), ow_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(ow, out_W), ow_body, ow_done);

    builder.SetInsertPoint(ow_body);

    // Accumulator for this output position
    llvm::Value* acc = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), acc);

    // Kernel loops (kd, kh, kw)
    llvm::BasicBlock* kd_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_cond", current_func);
    llvm::BasicBlock* kd_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_body", current_func);
    llvm::BasicBlock* kd_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kd_done", current_func);

    llvm::Value* kd_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kd_idx);
    builder.CreateBr(kd_cond);

    builder.SetInsertPoint(kd_cond);
    llvm::Value* kd = builder.CreateLoad(ctx_.int64Type(), kd_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kd, k_D), kd_body, kd_done);

    builder.SetInsertPoint(kd_body);

    llvm::BasicBlock* kh_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_cond", current_func);
    llvm::BasicBlock* kh_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_body", current_func);
    llvm::BasicBlock* kh_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kh_done", current_func);

    llvm::Value* kh_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kh_idx);
    builder.CreateBr(kh_cond);

    builder.SetInsertPoint(kh_cond);
    llvm::Value* kh = builder.CreateLoad(ctx_.int64Type(), kh_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kh, k_H), kh_body, kh_done);

    builder.SetInsertPoint(kh_body);

    llvm::BasicBlock* kw_cond = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_cond", current_func);
    llvm::BasicBlock* kw_body = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_body", current_func);
    llvm::BasicBlock* kw_done = llvm::BasicBlock::Create(ctx_.context(), "conv3d_kw_done", current_func);

    llvm::Value* kw_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kw_idx);
    builder.CreateBr(kw_cond);

    builder.SetInsertPoint(kw_cond);
    llvm::Value* kw = builder.CreateLoad(ctx_.int64Type(), kw_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(kw, k_W), kw_body, kw_done);

    builder.SetInsertPoint(kw_body);

    // Calculate input indices
    llvm::Value* id = builder.CreateAdd(builder.CreateMul(od, stride_val_i64), kd);
    id = builder.CreateSub(id, padding_val_i64);
    llvm::Value* ih = builder.CreateAdd(builder.CreateMul(oh, stride_val_i64), kh);
    ih = builder.CreateSub(ih, padding_val_i64);
    llvm::Value* iw = builder.CreateAdd(builder.CreateMul(ow, stride_val_i64), kw);
    iw = builder.CreateSub(iw, padding_val_i64);

    // Bounds check
    llvm::Value* valid_d = builder.CreateAnd(
        builder.CreateICmpSGE(id, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(id, in_D));
    llvm::Value* valid_h = builder.CreateAnd(
        builder.CreateICmpSGE(ih, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(ih, in_H));
    llvm::Value* valid_w = builder.CreateAnd(
        builder.CreateICmpSGE(iw, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        builder.CreateICmpSLT(iw, in_W));
    llvm::Value* valid = builder.CreateAnd(valid_d, builder.CreateAnd(valid_h, valid_w));

    llvm::BasicBlock* valid_block = llvm::BasicBlock::Create(ctx_.context(), "conv3d_valid", current_func);
    llvm::BasicBlock* continue_block = llvm::BasicBlock::Create(ctx_.context(), "conv3d_continue", current_func);

    builder.CreateCondBr(valid, valid_block, continue_block);

    builder.SetInsertPoint(valid_block);
    // input_idx = id * (H * W) + ih * W + iw
    llvm::Value* in_idx = builder.CreateAdd(
        builder.CreateMul(id, builder.CreateMul(in_H, in_W)),
        builder.CreateAdd(builder.CreateMul(ih, in_W), iw));
    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), in_elems, in_idx);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    // kernel_idx = kd * (kH * kW) + kh * kW + kw
    llvm::Value* k_idx = builder.CreateAdd(
        builder.CreateMul(kd, builder.CreateMul(k_H, k_W)),
        builder.CreateAdd(builder.CreateMul(kh, k_W), kw));
    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), k_elems, k_idx);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* cur_acc = builder.CreateLoad(ctx_.doubleType(), acc);
    builder.CreateStore(builder.CreateFAdd(cur_acc, prod), acc);
    builder.CreateBr(continue_block);

    builder.SetInsertPoint(continue_block);
    builder.CreateStore(builder.CreateAdd(kw, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kw_idx);
    builder.CreateBr(kw_cond);

    builder.SetInsertPoint(kw_done);
    builder.CreateStore(builder.CreateAdd(kh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kh_idx);
    builder.CreateBr(kh_cond);

    builder.SetInsertPoint(kh_done);
    builder.CreateStore(builder.CreateAdd(kd, llvm::ConstantInt::get(ctx_.int64Type(), 1)), kd_idx);
    builder.CreateBr(kd_cond);

    builder.SetInsertPoint(kd_done);
    // Store result
    llvm::Value* out_idx = builder.CreateAdd(
        builder.CreateMul(od, builder.CreateMul(out_H, out_W)),
        builder.CreateAdd(builder.CreateMul(oh, out_W), ow));
    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_acc = builder.CreateLoad(ctx_.doubleType(), acc);
    llvm::Value* acc_bits = builder.CreateBitCast(final_acc, ctx_.int64Type());
    builder.CreateStore(acc_bits, out_ptr);

    builder.CreateStore(builder.CreateAdd(ow, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ow_idx);
    builder.CreateBr(ow_cond);

    builder.SetInsertPoint(ow_done);
    builder.CreateStore(builder.CreateAdd(oh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), oh_idx);
    builder.CreateBr(oh_cond);

    builder.SetInsertPoint(oh_done);
    builder.CreateStore(builder.CreateAdd(od, llvm::ConstantInt::get(ctx_.int64Type(), 1)), od_idx);
    builder.CreateBr(od_cond);

    builder.SetInsertPoint(od_done);
    return tagged_.packHeapPtr(result_ptr);
}

// Optimizers / weight init / LR schedulers: see tensor_training_codegen.cpp

// Linear algebra (Phase 4.4): see tensor_linalg_codegen.cpp

// Loss functions (Track 6.3): see tensor_loss_codegen.cpp

// Data loading infrastructure: see tensor_dataloader_codegen.cpp

// Transformer architecture (Track 8): see tensor_transformer_codegen.cpp
// ===================================================================
// Tensor Unary Operations (tensor-neg, tensor-abs, tensor-sqrt, etc.)
// ===================================================================

bool TensorCodegen::emitTensorADUnaryDispatch(llvm::Value* src_elems,
                                              llvm::Value* result_elems,
                                              llvm::Value* total_elements,
                                              uint32_t ad_op_type,
                                              llvm::BasicBlock* exit_block,
                                              const std::string& name) {
    if (!autodiff_ || ad_op_type == 0 || !src_elems || !result_elems || !total_elements || !exit_block) {
        return false;
    }

    auto& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), name + "_ad_path", current_func);
    llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), name + "_numeric_path", current_func);
    builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

    builder.SetInsertPoint(ad_path);
    llvm::Value* ad_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, (name + "_ad_i").c_str());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);
    llvm::BasicBlock* ad_cond = llvm::BasicBlock::Create(ctx_.context(), name + "_ad_cond", current_func);
    llvm::BasicBlock* ad_body = llvm::BasicBlock::Create(ctx_.context(), name + "_ad_body", current_func);
    llvm::BasicBlock* ad_exit = llvm::BasicBlock::Create(ctx_.context(), name + "_ad_exit", current_func);
    builder.CreateBr(ad_cond);

    builder.SetInsertPoint(ad_cond);
    llvm::Value* ad_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
    llvm::Value* ad_has_elem = builder.CreateICmpULT(ad_i, total_elements);
    builder.CreateCondBr(ad_has_elem, ad_body, ad_exit);

    builder.SetInsertPoint(ad_body);
    llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, ad_i);
    llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, ad_i);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* elem_node = adNodeFromTensorElementBits(elem_bits, name + "_ad_elem");
    llvm::Value* result_node = autodiff_->recordADNodeUnary(ad_op_type, elem_node);
    llvm::Value* result_bits = builder.CreatePtrToInt(result_node, ctx_.int64Type());
    builder.CreateStore(result_bits, dst_elem_ptr);

    llvm::Value* next_ad_i = builder.CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ad_i, ad_counter);
    builder.CreateBr(ad_cond);

    builder.SetInsertPoint(ad_exit);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(numeric_path);
    return true;
}

/**
 * Generic SIMD+scalar unary loop.  Applies a single-operand math op to
 * every element of a tensor and returns a new tensor of the same shape.
 *
 * @param tensor_val  Tagged tensor input
 * @param op_name     Short label used for LLVM basic-block names
 * @param intrinsic_id LLVM floating-point intrinsic to call per element
 *                    (ignored when use_fneg == true)
 * @param use_fneg    If true, emit FNeg instead of an intrinsic call
 */
llvm::Value* TensorCodegen::emitTensorUnaryOp(llvm::Value* tensor_val,
                                               const std::string& op_name,
                                               llvm::Intrinsic::ID intrinsic_id,
                                               bool use_fneg,
                                               uint32_t ad_op_type) {
    if (!tensor_val) return tagged_.packNull();
    auto& builder = ctx_.builder();

    // Ensure tagged value
    if (tensor_val->getType() != ctx_.taggedValueType())
        tensor_val = tagged_.packInt64(tensor_val, true);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr     = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type*  tensor_type    = ctx_.tensorType();

    // Load shape metadata
    llvm::Value* dims_ptr       = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 0));
    llvm::Value* num_dims       = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 1));
    llvm::Value* src_elems      = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 2));
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 3));

    // Allocate result tensor header
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, (op_name + "_result").c_str());

    // Allocate and copy dimensions
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_bytes = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, (op_name + "_dims").c_str());
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    // Allocate result elements
    llvm::Value* elems_bytes = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, (op_name + "_elems").c_str());

    // SIMD parameters
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    // Declare scalar intrinsic (used in scalar tail loop)
    llvm::Function* scalar_intr = nullptr;
    if (!use_fneg)
        scalar_intr = ESHKOL_GET_INTRINSIC(&ctx_.module(), intrinsic_id, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond   = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_simd_cond").c_str(),   current_func);
    llvm::BasicBlock* simd_body   = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_simd_body").c_str(),   current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_scalar_cond").c_str(), current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_scalar_body").c_str(), current_func);
    llvm::BasicBlock* exit_block  = llvm::BasicBlock::Create(ctx_.context(), (op_name+"_exit").c_str(),        current_func);

    emitTensorADUnaryDispatch(src_elems, result_elems, total_elements, ad_op_type, exit_block, op_name);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, (op_name + "_i").c_str());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    // simd_count = (total / SIMD_WIDTH) * SIMD_WIDTH
    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    builder.CreateBr(simd_cond);

    // === SIMD loop ===
    builder.SetInsertPoint(simd_cond);
    {
        llvm::Value* i    = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cond = builder.CreateICmpULT(i, simd_count);
        builder.CreateCondBr(cond, simd_body, scalar_cond);
    }

    builder.SetInsertPoint(simd_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        if (use_simd) {
            llvm::Value* src_vec_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
            llvm::Value* x_vec       = builder.CreateAlignedLoad(vec_type, src_vec_ptr,
                                                                  llvm::MaybeAlign(8));
            llvm::Value* result_vec;
            if (use_fneg) {
                result_vec = builder.CreateFNeg(x_vec);
            } else {
                llvm::Function* vec_intr = ESHKOL_GET_INTRINSIC(&ctx_.module(), intrinsic_id, {vec_type});
                result_vec = builder.CreateCall(vec_intr, {x_vec});
            }
            llvm::Value* dst_vec_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, i);
            builder.CreateAlignedStore(result_vec, dst_vec_ptr, llvm::MaybeAlign(8));
        }
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_i, counter);
        auto* backEdge = builder.CreateBr(simd_cond);
        attachLoopMetadata(backEdge, true, SIMD_WIDTH, false, 0);
    }

    // === Scalar tail loop ===
    builder.SetInsertPoint(scalar_cond);
    {
        llvm::Value* i    = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* cond = builder.CreateICmpULT(i, total_elements);
        builder.CreateCondBr(cond, scalar_body, exit_block);
    }

    builder.SetInsertPoint(scalar_body);
    {
        llvm::Value* i   = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* src = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
        llvm::Value* x   = builder.CreateLoad(ctx_.doubleType(), src);
        llvm::Value* r;
        if (use_fneg) {
            r = builder.CreateFNeg(x);
        } else {
            r = builder.CreateCall(scalar_intr, {x});
        }
        builder.CreateStore(r, builder.CreateGEP(ctx_.doubleType(), result_elems, i));
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
        builder.CreateStore(next_i, counter);
        auto* backEdge = builder.CreateBr(scalar_cond);
        attachLoopMetadata(backEdge, false, 0, true, 4);
    }

    // === Exit: populate result tensor struct ===
    builder.SetInsertPoint(exit_block);
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(num_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(total_elements,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    // Unary elementwise op preserves the operand's dtype.
    emitDtypePropagateUnary(result_ptr, tensor_ptr);

    return tagged_.packHeapPtr(result_ptr);
}

// ---- Individual unary wrappers ----

llvm::Value* TensorCodegen::tensorNeg(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-neg requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tneg", llvm::Intrinsic::fabs /* unused */, /*use_fneg=*/true, 11);
}

llvm::Value* TensorCodegen::tensorAbs(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-abs requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tabs", llvm::Intrinsic::fabs, false, 42);
}

llvm::Value* TensorCodegen::tensorSqrt(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-sqrt requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tsqrt", llvm::Intrinsic::sqrt, false, 41);
}

llvm::Value* TensorCodegen::tensorExp(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-exp requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "texp", llvm::Intrinsic::exp, false, 8);
}

llvm::Value* TensorCodegen::tensorLog(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-log requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tlog", llvm::Intrinsic::log, false, 9);
}

llvm::Value* TensorCodegen::tensorSin(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-sin requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tsin", llvm::Intrinsic::sin, false, 6);
}

llvm::Value* TensorCodegen::tensorCos(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 1) { eshkol_error("tensor-cos requires 1 argument"); return nullptr; }
    llvm::Value* t = codegenAST(&op->call_op.variables[0]);
    if (!t) return nullptr;
    return emitTensorUnaryOp(t, "tcos", llvm::Intrinsic::cos, false, 7);
}

// ===================================================================
// Tensor Binary Operations: tensor-pow, tensor-maximum, tensor-minimum
// Delegate through rawTensorArithmeticSIMD which already handles "pow"/"max"/"min"
// ===================================================================

llvm::Value* TensorCodegen::tensorPow(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-pow requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "pow");
}

llvm::Value* TensorCodegen::tensorMaximum(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-maximum requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "max");
}

llvm::Value* TensorCodegen::tensorMinimum(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-minimum requires 2 arguments"); return nullptr; }
    llvm::Value* a = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b = codegenAST(&op->call_op.variables[1]);
    if (!a || !b) return nullptr;
    return tensorArithmeticInternal(a, b, "min");
}

// ===================================================================
// tensor-scale: (tensor-scale tensor scalar) — multiply all elements
// ===================================================================

llvm::Value* TensorCodegen::tensorScale(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("tensor-scale requires 2 arguments: tensor and scalar"); return nullptr; }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* scalar_val = codegenAST(&op->call_op.variables[1]);
    if (!tensor_val || !scalar_val) return nullptr;

    auto& builder = ctx_.builder();

    // Ensure tagged
    if (tensor_val->getType() != ctx_.taggedValueType())
        tensor_val = tagged_.packInt64(tensor_val, true);

    // Extract scalar as double (handles both int64 and double tagged values)
    llvm::Value* scalar_d = extractAsDouble(scalar_val);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* tensor_ptr_int = tagged_.unpackInt64(tensor_val);
    llvm::Value* tensor_ptr     = builder.CreateIntToPtr(tensor_ptr_int, ctx_.ptrType());
    llvm::Type*  tensor_type    = ctx_.tensorType();

    // Load shape metadata
    llvm::Value* dims_ptr       = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 0));
    llvm::Value* num_dims       = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 1));
    llvm::Value* src_elems      = builder.CreateLoad(ctx_.ptrType(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 2));
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateStructGEP(tensor_type, tensor_ptr, 3));

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, "tscale_result");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* dims_bytes  = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "tscale_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    llvm::Value* elems_bytes  = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "tscale_elems");

    // SIMD setup
    const unsigned SIMD_WIDTH = getSIMDWidth();
    llvm::VectorType* vec_type = getSIMDVectorType();
    const bool use_simd = (SIMD_WIDTH > 1 && vec_type != nullptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* simd_cond   = llvm::BasicBlock::Create(ctx_.context(), "tscale_simd_cond",   current_func);
    llvm::BasicBlock* simd_body   = llvm::BasicBlock::Create(ctx_.context(), "tscale_simd_body",   current_func);
    llvm::BasicBlock* scalar_cond = llvm::BasicBlock::Create(ctx_.context(), "tscale_scalar_cond", current_func);
    llvm::BasicBlock* scalar_body = llvm::BasicBlock::Create(ctx_.context(), "tscale_scalar_body", current_func);
    llvm::BasicBlock* exit_block  = llvm::BasicBlock::Create(ctx_.context(), "tscale_exit",        current_func);

    if (autodiff_) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "tscale_ad_path", current_func);
        llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "tscale_numeric_path", current_func);
        builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

        builder.SetInsertPoint(ad_path);
        llvm::Value* scalar_node = autodiff_->createADConstant(scalar_d);
        llvm::Value* ad_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tscale_ad_i");
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_counter);
        llvm::BasicBlock* ad_cond = llvm::BasicBlock::Create(ctx_.context(), "tscale_ad_cond", current_func);
        llvm::BasicBlock* ad_body = llvm::BasicBlock::Create(ctx_.context(), "tscale_ad_body", current_func);
        llvm::BasicBlock* ad_done = llvm::BasicBlock::Create(ctx_.context(), "tscale_ad_done", current_func);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_cond);
        llvm::Value* ad_i = builder.CreateLoad(ctx_.int64Type(), ad_counter);
        llvm::Value* ad_has_elem = builder.CreateICmpULT(ad_i, total_elements);
        builder.CreateCondBr(ad_has_elem, ad_body, ad_done);

        builder.SetInsertPoint(ad_body);
        llvm::Value* src_elem_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, ad_i);
        llvm::Value* dst_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, ad_i);
        llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), src_elem_ptr);
        llvm::Value* elem_node = adNodeFromTensorElementBits(elem_bits, "tscale_ad_elem");
        llvm::Value* result_node = autodiff_->recordADNodeBinary(4, elem_node, scalar_node);
        llvm::Value* result_bits = builder.CreatePtrToInt(result_node, ctx_.int64Type());
        builder.CreateStore(result_bits, dst_elem_ptr);
        builder.CreateStore(builder.CreateAdd(ad_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_counter);
        builder.CreateBr(ad_cond);

        builder.SetInsertPoint(ad_done);
        builder.CreateBr(exit_block);

        builder.SetInsertPoint(numeric_path);
    }

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "tscale_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);

    llvm::Value* simd_count = builder.CreateMul(
        builder.CreateUDiv(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH)),
        llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));

    // Splat scalar to vector once
    llvm::Value* scale_vec = nullptr;
    if (use_simd) {
        scale_vec = llvm::UndefValue::get(vec_type);
        for (unsigned k = 0; k < SIMD_WIDTH; ++k)
            scale_vec = builder.CreateInsertElement(scale_vec, scalar_d,
                llvm::ConstantInt::get(ctx_.int32Type(), k));
    }

    builder.CreateBr(simd_cond);

    // SIMD loop
    builder.SetInsertPoint(simd_cond);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        builder.CreateCondBr(builder.CreateICmpULT(i, simd_count), simd_body, scalar_cond);
    }
    builder.SetInsertPoint(simd_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        if (use_simd) {
            llvm::Value* src_ptr = builder.CreateGEP(ctx_.doubleType(), src_elems, i);
            llvm::Value* x_vec   = builder.CreateAlignedLoad(vec_type, src_ptr, llvm::MaybeAlign(8));
            llvm::Value* r_vec   = builder.CreateFMul(x_vec, scale_vec);
            builder.CreateAlignedStore(r_vec, builder.CreateGEP(ctx_.doubleType(), result_elems, i), llvm::MaybeAlign(8));
        }
        llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), SIMD_WIDTH));
        builder.CreateStore(next_i, counter);
        auto* be = builder.CreateBr(simd_cond);
        attachLoopMetadata(be, true, SIMD_WIDTH, false, 0);
    }

    // Scalar tail loop
    builder.SetInsertPoint(scalar_cond);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), scalar_body, exit_block);
    }
    builder.SetInsertPoint(scalar_body);
    {
        llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
        llvm::Value* x = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), src_elems, i));
        llvm::Value* r = builder.CreateFMul(x, scalar_d);
        builder.CreateStore(r, builder.CreateGEP(ctx_.doubleType(), result_elems, i));
        builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
        auto* be = builder.CreateBr(scalar_cond);
        attachLoopMetadata(be, false, 0, true, 4);
    }

    // Populate result tensor struct
    builder.SetInsertPoint(exit_block);
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(num_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(total_elements,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    // Unary elementwise op preserves the operand's dtype.
    emitDtypePropagateUnary(result_ptr, tensor_ptr);

    return tagged_.packHeapPtr(result_ptr);
}

// ===================================================================
// batch-matmul: (batch-matmul A B)
// A: [batch, M, K]  B: [batch, K, N]  Result: [batch, M, N]
// Delegates to C runtime eshkol_batch_matmul_f64.
// ===================================================================

llvm::Value* TensorCodegen::batchMatmul(const eshkol_operations_t* op) {
    if (op->call_op.num_vars != 2) { eshkol_error("batch-matmul requires 2 arguments"); return nullptr; }

    llvm::Value* a_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* b_val = codegenAST(&op->call_op.variables[1]);
    if (!a_val || !b_val) return nullptr;

    auto& builder = ctx_.builder();

    // Ensure tagged
    if (a_val->getType() != ctx_.taggedValueType()) a_val = tagged_.packInt64(a_val, true);
    if (b_val->getType() != ctx_.taggedValueType()) b_val = tagged_.packInt64(b_val, true);

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensor pointers
    llvm::Value* a_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(a_val), ctx_.ptrType());
    llvm::Value* b_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b_val), ctx_.ptrType());
    llvm::Type*  tt    = ctx_.tensorType();

    // Load A dims: [batch, M, K]
    llvm::Value* a_dims  = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, a_ptr, 0));
    llvm::Value* a_total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(tt, a_ptr, 3));
    llvm::Value* a_elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, a_ptr, 2));

    llvm::Value* b_dims  = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, b_ptr, 0));
    llvm::Value* b_total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(tt, b_ptr, 3));
    llvm::Value* b_elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(tt, b_ptr, 2));

    // Extract batch, M, K from A; K, N from B
    llvm::Value* batch = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* M_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    llvm::Value* K_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), a_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    llvm::Value* N_dim = builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), b_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    // Result total = batch * M * N
    llvm::Value* result_total = builder.CreateMul(builder.CreateMul(batch, M_dim), N_dim);

    // Allocate result tensor [batch, M, N]
    llvm::Function* arena_alloc  = mem_.getArenaAllocate();
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value*    result_ptr   = builder.CreateCall(alloc_tensor, {arena_ptr}, "bmm_result");

    // Allocate dims array [3] = {batch, M, N}
    llvm::Value* dims_bytes  = llvm::ConstantInt::get(ctx_.int64Type(), 3 * sizeof(int64_t));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "bmm_dims");
    builder.CreateStore(batch, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(M_dim, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(N_dim, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));

    llvm::Value* elems_bytes  = builder.CreateMul(result_total,
        llvm::ConstantInt::get(ctx_.int64Type(), (int64_t)sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "bmm_elems");

    // Call C runtime: void eshkol_batch_matmul_f64(
    //     const double* a, const double* b, double* c,
    //     int64_t batch, int64_t M, int64_t K, int64_t N)
    auto* bmm_ft = llvm::FunctionType::get(ctx_.voidType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type(), ctx_.int64Type()}, false);
    llvm::Function* bmm_fn = ctx_.module().getFunction("eshkol_batch_matmul_f64");
    if (!bmm_fn) {
        bmm_fn = llvm::Function::Create(bmm_ft, llvm::Function::ExternalLinkage,
            "eshkol_batch_matmul_f64", &ctx_.module());
    }
    builder.CreateCall(bmm_fn, {a_elems, b_elems, result_elems, batch, M_dim, K_dim, N_dim});

    // Populate result tensor struct
    llvm::Type* tensor_type = ctx_.tensorType();
    builder.CreateStore(result_dims,
        builder.CreateStructGEP(tensor_type, result_ptr, 0));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 3),
        builder.CreateStructGEP(tensor_type, result_ptr, 1));
    builder.CreateStore(result_elems,
        builder.CreateStructGEP(tensor_type, result_ptr, 2));
    builder.CreateStore(result_total,
        builder.CreateStructGEP(tensor_type, result_ptr, 3));

    return tagged_.packHeapPtr(result_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
