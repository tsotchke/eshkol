/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Shape Operations (Phase 4).
 * Extracted from tensor_codegen.cpp during the v1.2 mechanical split.
 *
 * Implements the shape-manipulating operations: reshape, transpose,
 * squeeze, unsqueeze, flatten, concatenate, stack, split, slice,
 * tensorShape, tensorLength.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-shape-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/Config/llvm-config.h>

#ifdef ESHKOL_XLA_ENABLED
#include <eshkol/backend/xla/xla_codegen.h>
#endif

// LLVM VERSION COMPATIBILITY
#if LLVM_VERSION_MAJOR >= 21
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getOrInsertDeclaration(mod, id, types)
#else
#define ESHKOL_GET_INTRINSIC(mod, id, types) llvm::Intrinsic::getDeclaration(mod, id, types)
#endif

namespace eshkol {

// ============================================================
// Shape Operations (Phase 4)
// ============================================================

llvm::Value* TensorCodegen::squeeze(const eshkol_operations_t* op) {
    // Squeeze: remove dimensions of size 1
    // (squeeze tensor) - remove all size-1 dims
    // (squeeze tensor dim) - remove specific dim if size is 1
    if (op->call_op.num_vars < 1 || op->call_op.num_vars > 2) {
        eshkol_error("squeeze requires 1-2 arguments (tensor, optional dim)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "squeeze");
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate result tensor (metadata only - shares element data)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "squeeze_result");

    // Count non-size-1 dimensions (dynamic loop)
    // First pass: count how many dims to keep
    llvm::BasicBlock* count_cond = llvm::BasicBlock::Create(ctx_.context(), "sq_count_cond", current_func);
    llvm::BasicBlock* count_body = llvm::BasicBlock::Create(ctx_.context(), "sq_count_body", current_func);
    llvm::BasicBlock* count_done = llvm::BasicBlock::Create(ctx_.context(), "sq_count_done", current_func);

    llvm::Value* count_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_count");
    llvm::Value* count_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_count_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), count_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), count_i);
    builder.CreateBr(count_cond);

    builder.SetInsertPoint(count_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), count_i);
    llvm::Value* count_cmp = builder.CreateICmpULT(ci, num_dims);
    builder.CreateCondBr(count_cmp, count_body, count_done);

    builder.SetInsertPoint(count_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, ci);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* is_not_one = builder.CreateICmpNE(dim_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* curr_count = builder.CreateLoad(ctx_.int64Type(), count_var);
    llvm::Value* new_count = builder.CreateSelect(is_not_one,
        builder.CreateAdd(curr_count, llvm::ConstantInt::get(ctx_.int64Type(), 1)),
        curr_count);
    builder.CreateStore(new_count, count_var);
    llvm::Value* next_ci = builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ci, count_i);
    builder.CreateBr(count_cond);

    builder.SetInsertPoint(count_done);
    llvm::Value* new_ndim = builder.CreateLoad(ctx_.int64Type(), count_var);

    // Ensure at least 1 dimension (scalar becomes 1D with single element)
    llvm::Value* is_zero = builder.CreateICmpEQ(new_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    new_ndim = builder.CreateSelect(is_zero, llvm::ConstantInt::get(ctx_.int64Type(), 1), new_ndim);

    // Allocate new dimensions array
    llvm::Value* new_dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* sq_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(sq_arena_alloc, {arena_ptr, new_dims_size}, "sq_new_dims");

    // Second pass: copy non-size-1 dimensions
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "sq_copy_done", current_func);

    llvm::Value* src_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_src_idx");
    llvm::Value* dst_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "sq_dst_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_idx);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_idx);
    llvm::Value* copy_cmp = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* keep_dim = builder.CreateICmpNE(src_dim_val, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Conditional copy block
    llvm::BasicBlock* do_copy = llvm::BasicBlock::Create(ctx_.context(), "sq_do_copy", current_func);
    llvm::BasicBlock* skip_copy = llvm::BasicBlock::Create(ctx_.context(), "sq_skip_copy", current_func);

    builder.CreateCondBr(keep_dim, do_copy, skip_copy);

    builder.SetInsertPoint(do_copy);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_idx);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(src_dim_val, dst_dim_ptr);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_idx);
    builder.CreateBr(skip_copy);

    builder.SetInsertPoint(skip_copy);
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_idx);
    builder.CreateBr(copy_cond);

    // Handle edge case: all dims were squeezed (scalar)
    builder.SetInsertPoint(copy_done);
    llvm::BasicBlock* scalar_case = llvm::BasicBlock::Create(ctx_.context(), "sq_scalar", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "sq_finalize", current_func);

    llvm::Value* final_dst_idx = builder.CreateLoad(ctx_.int64Type(), dst_idx);
    llvm::Value* was_scalar = builder.CreateICmpEQ(final_dst_idx, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateCondBr(was_scalar, scalar_case, finalize);

    builder.SetInsertPoint(scalar_case);
    // Set single dimension of 1 (or total_elements for proper shape)
    llvm::Value* scalar_dim_ptr = builder.CreateGEP(ctx_.int64Type(), new_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(total_elements, scalar_dim_ptr);
    builder.CreateBr(finalize);

    // Populate result tensor (shares element data with original)
    builder.SetInsertPoint(finalize);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::unsqueeze(const eshkol_operations_t* op) {
    // Unsqueeze: add a dimension of size 1 at specified position
    // (unsqueeze tensor dim) - adds size-1 dim at position dim
    if (op->call_op.num_vars != 2) {
        eshkol_error("unsqueeze requires 2 arguments (tensor, dim)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* dim_arg = codegenAST(&op->call_op.variables[1]);
    if (!dim_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "unsqueeze");
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Extract dim index - handle both raw int64 and tagged
    llvm::Value* dim_idx = dim_arg;
    if (dim_arg->getType() == ctx_.taggedValueType()) {
        dim_idx = tagged_.unpackInt64(dim_arg);
    } else if (!dim_arg->getType()->isIntegerTy(64)) {
        dim_idx = builder.CreateSExtOrTrunc(dim_arg, ctx_.int64Type());
    }

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "unsq_result");

    // New number of dimensions is old + 1
    llvm::Value* new_ndim = builder.CreateAdd(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate new dimensions array
    llvm::Value* new_dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* unsq_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(unsq_arena_alloc, {arena_ptr, new_dims_size}, "unsq_new_dims");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Copy dimensions, inserting 1 at dim_idx
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_done", current_func);

    llvm::Value* src_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "unsq_src_i");
    llvm::Value* dst_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "unsq_dst_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_i);
    llvm::Value* cmp = builder.CreateICmpULT(di, new_ndim);
    builder.CreateCondBr(cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    // Check if this is the insertion point
    llvm::Value* is_insert_pos = builder.CreateICmpEQ(di, dim_idx);

    llvm::BasicBlock* insert_one = llvm::BasicBlock::Create(ctx_.context(), "unsq_insert_one", current_func);
    llvm::BasicBlock* copy_old = llvm::BasicBlock::Create(ctx_.context(), "unsq_copy_old", current_func);
    llvm::BasicBlock* next_iter = llvm::BasicBlock::Create(ctx_.context(), "unsq_next_iter", current_func);

    builder.CreateCondBr(is_insert_pos, insert_one, copy_old);

    builder.SetInsertPoint(insert_one);
    llvm::Value* dst_ptr_ins = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), dst_ptr_ins);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(copy_old);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_i);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_ptr_cpy = builder.CreateGEP(ctx_.int64Type(), new_dims, di);
    builder.CreateStore(src_dim_val, dst_ptr_cpy);
    // Increment source index only when copying
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_i);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(next_iter);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_i);
    builder.CreateBr(copy_cond);

    // Populate result tensor (shares element data)
    builder.SetInsertPoint(copy_done);
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::flatten(const eshkol_operations_t* op) {
    // Flatten: convert tensor to 1D
    // (flatten tensor) - all dimensions become a single dimension
    if (op->call_op.num_vars != 1) {
        eshkol_error("flatten requires exactly 1 argument (tensor)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "flatten");
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load tensor properties
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "flat_result");

    // Allocate single dimension (1D)
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Function* flat_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* new_dims = builder.CreateCall(flat_arena_alloc, {arena_ptr, dims_size}, "flat_dims");

    // Set the single dimension to total_elements
    builder.CreateStore(total_elements, new_dims);

    // Populate result tensor (shares element data)
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(new_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), r_ndim_field);  // 1D
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(src_elems, r_elems_field);  // Share data - no copy!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::concatenate(const eshkol_operations_t* op) {
    // Concatenate: join tensors along specified axis
    // (concatenate axis tensor1 tensor2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("concatenate requires at least 3 arguments (axis, tensor1, tensor2)");
        return nullptr;
    }

    // Get axis
    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[0]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 for axis
    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Get first tensor to establish shape template
    llvm::Value* first_tensor = codegenAST(&op->call_op.variables[1]);
    if (!first_tensor) return nullptr;

    llvm::Value* first_ptr = unpackTensorOperandChecked(first_tensor, "concatenate");

    llvm::Value* first_dims_field = builder.CreateStructGEP(tensor_type, first_ptr, 0);
    llvm::Value* first_dims_ptr = builder.CreateLoad(ctx_.ptrType(), first_dims_field);
    llvm::Value* first_ndim_field = builder.CreateStructGEP(tensor_type, first_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), first_ndim_field);

    // Calculate total size along axis
    llvm::Value* concat_dim_sum = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_sum");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), concat_dim_sum);

    // Calculate product of other dimensions (for stride calculation)
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Collect all tensors and sum their axis dimensions
    std::vector<llvm::Value*> tensor_ptrs;
    for (size_t i = 1; i < op->call_op.num_vars; ++i) {
        llvm::Value* t = codegenAST(&op->call_op.variables[i]);
        if (!t) return nullptr;
        llvm::Value* t_ptr = unpackTensorOperandChecked(t, "concatenate");
        tensor_ptrs.push_back(t_ptr);

        // Add this tensor's dimension along axis
        llvm::Value* t_dims_field = builder.CreateStructGEP(tensor_type, t_ptr, 0);
        llvm::Value* t_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t_dims_field);
        llvm::Value* t_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), t_dims_ptr, axis);
        llvm::Value* t_axis_dim = builder.CreateLoad(ctx_.int64Type(), t_axis_dim_ptr);

        llvm::Value* curr_sum = builder.CreateLoad(ctx_.int64Type(), concat_dim_sum);
        builder.CreateStore(builder.CreateAdd(curr_sum, t_axis_dim), concat_dim_sum);
    }

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "concat_result");

    // Allocate and copy dimensions
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* concat_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(concat_arena_alloc, {arena_ptr, dims_size}, "concat_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), first_dims_ptr, llvm::MaybeAlign(8), dims_size);

    // Update the axis dimension with the sum
    llvm::Value* new_axis_dim = builder.CreateLoad(ctx_.int64Type(), concat_dim_sum);
    llvm::Value* result_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, axis);
    builder.CreateStore(new_axis_dim, result_axis_dim_ptr);

    // Calculate total elements
    llvm::BasicBlock* calc_total_cond = llvm::BasicBlock::Create(ctx_.context(), "concat_total_cond", current_func);
    llvm::BasicBlock* calc_total_body = llvm::BasicBlock::Create(ctx_.context(), "concat_total_body", current_func);
    llvm::BasicBlock* calc_total_done = llvm::BasicBlock::Create(ctx_.context(), "concat_total_done", current_func);

    llvm::Value* total_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_total");
    llvm::Value* calc_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "concat_calc_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), calc_i);
    builder.CreateBr(calc_total_cond);

    builder.SetInsertPoint(calc_total_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), calc_i);
    llvm::Value* calc_cmp = builder.CreateICmpULT(ci, num_dims);
    builder.CreateCondBr(calc_cmp, calc_total_body, calc_total_done);

    builder.SetInsertPoint(calc_total_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, ci);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* curr_total = builder.CreateLoad(ctx_.int64Type(), total_var);
    builder.CreateStore(builder.CreateMul(curr_total, dim_val), total_var);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), calc_i);
    builder.CreateBr(calc_total_cond);

    builder.SetInsertPoint(calc_total_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_var);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(concat_arena_alloc, {arena_ptr, elems_size}, "concat_elems");

    // Calculate strides before and after axis for copying
    // stride_after = product of dims after axis
    // stride_before = product of dims before axis * axis_dim

    llvm::Value* stride_after_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stride_after");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), stride_after_var);

    // Calculate stride_after (dims after axis)
    llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "stride_cond", current_func);
    llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "stride_body", current_func);
    llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "stride_done", current_func);

    llvm::Value* stride_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stride_i");
    llvm::Value* axis_plus_one = builder.CreateAdd(axis, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(axis_plus_one, stride_i);
    builder.CreateBr(stride_cond);

    builder.SetInsertPoint(stride_cond);
    llvm::Value* sti = builder.CreateLoad(ctx_.int64Type(), stride_i);
    llvm::Value* stride_cmp = builder.CreateICmpULT(sti, num_dims);
    builder.CreateCondBr(stride_cmp, stride_body, stride_done);

    builder.SetInsertPoint(stride_body);
    llvm::Value* st_dim_ptr = builder.CreateGEP(ctx_.int64Type(), first_dims_ptr, sti);
    llvm::Value* st_dim_val = builder.CreateLoad(ctx_.int64Type(), st_dim_ptr);
    llvm::Value* curr_stride = builder.CreateLoad(ctx_.int64Type(), stride_after_var);
    builder.CreateStore(builder.CreateMul(curr_stride, st_dim_val), stride_after_var);
    builder.CreateStore(builder.CreateAdd(sti, llvm::ConstantInt::get(ctx_.int64Type(), 1)), stride_i);
    builder.CreateBr(stride_cond);

    builder.SetInsertPoint(stride_done);
    llvm::Value* stride_after = builder.CreateLoad(ctx_.int64Type(), stride_after_var);

    // Stride-aware copy via runtime function for correct ND concatenation.
    // Compute outer_count = product of result_dims[0..axis-1]
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_body", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "concat_outer_done", current_func);

    llvm::Value* outer_count_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "outer_count");
    llvm::Value* outer_i_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "outer_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), outer_count_var);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), outer_i_var);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), outer_i_var);
    llvm::Value* outer_cmp = builder.CreateICmpULT(oi, axis);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    llvm::Value* od_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, oi);
    llvm::Value* od_val = builder.CreateLoad(ctx_.int64Type(), od_ptr);
    llvm::Value* oc_curr = builder.CreateLoad(ctx_.int64Type(), outer_count_var);
    builder.CreateStore(builder.CreateMul(oc_curr, od_val), outer_count_var);
    builder.CreateStore(builder.CreateAdd(oi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), outer_i_var);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    llvm::Value* outer_count = builder.CreateLoad(ctx_.int64Type(), outer_count_var);

    // Build arrays of tensor element pointers and axis dims for runtime function
    size_t num_tensors = tensor_ptrs.size();
    llvm::Value* src_datas_buf = builder.CreateCall(concat_arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors * sizeof(double*))}, "src_datas");
    llvm::Value* src_axis_dims_buf = builder.CreateCall(concat_arena_alloc,
        {arena_ptr, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors * sizeof(int64_t))}, "src_axis_dims");

    for (size_t i = 0; i < num_tensors; ++i) {
        llvm::Value* t_ptr = tensor_ptrs[i];
        llvm::Value* t_elems_field = builder.CreateStructGEP(tensor_type, t_ptr, 2);
        llvm::Value* t_elems = builder.CreateLoad(ctx_.ptrType(), t_elems_field);

        // Store element pointer
        llvm::Value* slot = builder.CreateGEP(ctx_.ptrType(), src_datas_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(t_elems, slot);

        // Store axis dimension
        llvm::Value* t_dims_field = builder.CreateStructGEP(tensor_type, t_ptr, 0);
        llvm::Value* t_dims_ptr = builder.CreateLoad(ctx_.ptrType(), t_dims_field);
        llvm::Value* t_axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), t_dims_ptr, axis);
        llvm::Value* t_axis_dim = builder.CreateLoad(ctx_.int64Type(), t_axis_dim_ptr);
        llvm::Value* adim_slot = builder.CreateGEP(ctx_.int64Type(), src_axis_dims_buf,
            llvm::ConstantInt::get(ctx_.int64Type(), i));
        builder.CreateStore(t_axis_dim, adim_slot);
    }

    // Call runtime: eshkol_concat_strided(result_data, num_tensors, src_datas, src_axis_dims, stride_after, outer_count)
    auto* concat_ft = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx_.context()),
        {ctx_.ptrType(), ctx_.int64Type(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type()}, false);
    llvm::Function* concat_fn = ctx_.module().getFunction("eshkol_concat_strided");
    if (!concat_fn) {
        concat_fn = llvm::Function::Create(concat_ft,
            llvm::Function::ExternalLinkage, "eshkol_concat_strided", &ctx_.module());
    }
    builder.CreateCall(concat_fn,
        {result_elems, llvm::ConstantInt::get(ctx_.int64Type(), num_tensors),
         src_datas_buf, src_axis_dims_buf, stride_after, outer_count});

    // Populate result tensor
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

llvm::Value* TensorCodegen::stack(const eshkol_operations_t* op) {
    // Stack: stack tensors on a new axis
    // (stack axis tensor1 tensor2 ...)
    if (op->call_op.num_vars < 3) {
        eshkol_error("stack requires at least 3 arguments (axis, tensor1, tensor2)");
        return nullptr;
    }

    // Get axis
    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[0]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 for axis
    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Get first tensor shape
    llvm::Value* first_tensor = codegenAST(&op->call_op.variables[1]);
    if (!first_tensor) return nullptr;

    llvm::Value* first_ptr = unpackTensorOperandChecked(first_tensor, "stack");

    llvm::Value* first_dims_field = builder.CreateStructGEP(tensor_type, first_ptr, 0);
    llvm::Value* first_dims_ptr = builder.CreateLoad(ctx_.ptrType(), first_dims_field);
    llvm::Value* first_ndim_field = builder.CreateStructGEP(tensor_type, first_ptr, 1);
    llvm::Value* old_ndim = builder.CreateLoad(ctx_.int64Type(), first_ndim_field);
    llvm::Value* first_total_field = builder.CreateStructGEP(tensor_type, first_ptr, 3);
    llvm::Value* tensor_elements = builder.CreateLoad(ctx_.int64Type(), first_total_field);

    size_t num_tensors = op->call_op.num_vars - 1;
    llvm::Value* num_tensors_val = llvm::ConstantInt::get(ctx_.int64Type(), num_tensors);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "stack_result");

    // New dimensions: insert num_tensors at axis position
    llvm::Value* new_ndim = builder.CreateAdd(old_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* dims_size = builder.CreateMul(new_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* stack_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(stack_arena_alloc, {arena_ptr, dims_size}, "stack_dims");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Copy dimensions, inserting num_tensors at axis position
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "stk_copy_done", current_func);

    llvm::Value* src_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_src_i");
    llvm::Value* dst_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_dst_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), src_i);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dst_i);
    llvm::Value* cmp = builder.CreateICmpULT(di, new_ndim);
    builder.CreateCondBr(cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* is_new_axis = builder.CreateICmpEQ(di, axis);

    llvm::BasicBlock* insert_new = llvm::BasicBlock::Create(ctx_.context(), "stk_insert", current_func);
    llvm::BasicBlock* copy_old = llvm::BasicBlock::Create(ctx_.context(), "stk_copy", current_func);
    llvm::BasicBlock* next_iter = llvm::BasicBlock::Create(ctx_.context(), "stk_next", current_func);

    builder.CreateCondBr(is_new_axis, insert_new, copy_old);

    builder.SetInsertPoint(insert_new);
    llvm::Value* dst_ptr_ins = builder.CreateGEP(ctx_.int64Type(), result_dims, di);
    builder.CreateStore(num_tensors_val, dst_ptr_ins);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(copy_old);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), src_i);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), first_dims_ptr, si);
    llvm::Value* src_dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_ptr_cpy = builder.CreateGEP(ctx_.int64Type(), result_dims, di);
    builder.CreateStore(src_dim_val, dst_ptr_cpy);
    llvm::Value* next_si = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_si, src_i);
    builder.CreateBr(next_iter);

    builder.SetInsertPoint(next_iter);
    llvm::Value* next_di = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_di, dst_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);

    // Total elements = num_tensors * elements_per_tensor
    llvm::Value* total_elements = builder.CreateMul(num_tensors_val, tensor_elements);

    // Allocate result elements
    llvm::Value* elems_size = builder.CreateMul(total_elements,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* result_elems = builder.CreateCall(stack_arena_alloc, {arena_ptr, elems_size}, "stack_elems");

    // Copy each tensor's elements
    llvm::Value* dst_offset = builder.CreateAlloca(ctx_.int64Type(), nullptr, "stk_dst_off");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dst_offset);

    for (size_t i = 1; i < op->call_op.num_vars; ++i) {
        llvm::Value* t = codegenAST(&op->call_op.variables[i]);
        if (!t) return nullptr;
        llvm::Value* t_ptr = unpackTensorOperandChecked(t, "stack");

        llvm::Value* t_elems_field = builder.CreateStructGEP(tensor_type, t_ptr, 2);
        llvm::Value* t_elems = builder.CreateLoad(ctx_.ptrType(), t_elems_field);
        llvm::Value* t_total_field = builder.CreateStructGEP(tensor_type, t_ptr, 3);
        llvm::Value* t_total = builder.CreateLoad(ctx_.int64Type(), t_total_field);

        llvm::Value* copy_size = builder.CreateMul(t_total,
            llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
        llvm::Value* curr_offset = builder.CreateLoad(ctx_.int64Type(), dst_offset);
        llvm::Value* dst_ptr = builder.CreateGEP(ctx_.doubleType(), result_elems, curr_offset);
        builder.CreateMemCpy(dst_ptr, llvm::MaybeAlign(8), t_elems, llvm::MaybeAlign(8), copy_size);

        builder.CreateStore(builder.CreateAdd(curr_offset, t_total), dst_offset);
    }

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(new_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::split(const eshkol_operations_t* op) {
    // Split: split tensor into chunks along an axis
    // (split tensor num-chunks axis)
    if (op->call_op.num_vars != 3) {
        eshkol_error("split requires 3 arguments (tensor, num-chunks, axis)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* chunks_arg = codegenAST(&op->call_op.variables[1]);
    if (!chunks_arg) return nullptr;

    llvm::Value* axis_arg = codegenAST(&op->call_op.variables[2]);
    if (!axis_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* num_chunks = chunks_arg;
    if (chunks_arg->getType() == ctx_.taggedValueType()) {
        num_chunks = tagged_.unpackInt64(chunks_arg);
    } else if (!chunks_arg->getType()->isIntegerTy(64)) {
        num_chunks = builder.CreateSExtOrTrunc(chunks_arg, ctx_.int64Type());
    }

    llvm::Value* axis = axis_arg;
    if (axis_arg->getType() == ctx_.taggedValueType()) {
        axis = tagged_.unpackInt64(axis_arg);
    } else if (!axis_arg->getType()->isIntegerTy(64)) {
        axis = builder.CreateSExtOrTrunc(axis_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "split");
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Get axis dimension
    llvm::Value* axis_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, axis);
    llvm::Value* axis_dim = builder.CreateLoad(ctx_.int64Type(), axis_dim_ptr);

    // chunk_size = axis_dim / num_chunks
    llvm::Value* chunk_size = builder.CreateUDiv(axis_dim, num_chunks);

    // Calculate elements per chunk
    // Product of all dimensions except axis, times chunk_size
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* other_prod_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "other_prod");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), other_prod_var);

    llvm::BasicBlock* prod_cond = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_cond", current_func);
    llvm::BasicBlock* prod_body = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_body", current_func);
    llvm::BasicBlock* prod_done = llvm::BasicBlock::Create(ctx_.context(), "spl_prod_done", current_func);

    llvm::Value* prod_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_prod_i");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), prod_i);
    builder.CreateBr(prod_cond);

    builder.SetInsertPoint(prod_cond);
    llvm::Value* pi = builder.CreateLoad(ctx_.int64Type(), prod_i);
    llvm::Value* prod_cmp = builder.CreateICmpULT(pi, num_dims);
    builder.CreateCondBr(prod_cmp, prod_body, prod_done);

    builder.SetInsertPoint(prod_body);
    llvm::Value* is_axis = builder.CreateICmpEQ(pi, axis);
    llvm::Value* d_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, pi);
    llvm::Value* d_val = builder.CreateLoad(ctx_.int64Type(), d_ptr);
    llvm::Value* curr_prod = builder.CreateLoad(ctx_.int64Type(), other_prod_var);
    llvm::Value* new_prod = builder.CreateSelect(is_axis, curr_prod, builder.CreateMul(curr_prod, d_val));
    builder.CreateStore(new_prod, other_prod_var);
    builder.CreateStore(builder.CreateAdd(pi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), prod_i);
    builder.CreateBr(prod_cond);

    builder.SetInsertPoint(prod_done);
    llvm::Value* other_prod = builder.CreateLoad(ctx_.int64Type(), other_prod_var);
    llvm::Value* chunk_elements = builder.CreateMul(other_prod, chunk_size);

    // Build list of chunks from back to front (cons builds backwards)
    // Use int64 alloca to store pointer as integer (same pattern as tensorShape)
    llvm::Value* list_acc = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_list");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), list_acc);  // null

    // Iterate from num_chunks-1 down to 0
    llvm::Value* chunk_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "spl_chunk_i");
    builder.CreateStore(builder.CreateSub(num_chunks, llvm::ConstantInt::get(ctx_.int64Type(), 1)), chunk_i);

    llvm::BasicBlock* chunk_cond = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_cond", current_func);
    llvm::BasicBlock* chunk_body = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_body", current_func);
    llvm::BasicBlock* chunk_done = llvm::BasicBlock::Create(ctx_.context(), "spl_chunk_done", current_func);

    builder.CreateBr(chunk_cond);

    builder.SetInsertPoint(chunk_cond);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), chunk_i);
    llvm::Value* chunk_cmp = builder.CreateICmpSGE(ci, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateCondBr(chunk_cmp, chunk_body, chunk_done);

    builder.SetInsertPoint(chunk_body);
    // Create tensor for this chunk using arena allocation
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* chunk_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "spl_chunk");

    // Allocate dims for chunk from arena
    llvm::Value* chunk_dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* spl_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* chunk_dims = builder.CreateCall(spl_arena_alloc, {arena_ptr, chunk_dims_size}, "spl_dims");
    builder.CreateMemCpy(chunk_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), chunk_dims_size);

    // Update axis dimension to chunk_size
    llvm::Value* chunk_axis_ptr = builder.CreateGEP(ctx_.int64Type(), chunk_dims, axis);
    builder.CreateStore(chunk_size, chunk_axis_ptr);

    // Calculate offset into source elements
    llvm::Value* elem_offset = builder.CreateMul(ci, chunk_elements);
    llvm::Value* chunk_elems = builder.CreateGEP(ctx_.int64Type(), src_elems, elem_offset);

    // Populate chunk tensor (view - shares data with original)
    llvm::Value* c_dims_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 0);
    builder.CreateStore(chunk_dims, c_dims_field);
    llvm::Value* c_ndim_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 1);
    builder.CreateStore(num_dims, c_ndim_field);
    llvm::Value* c_elems_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 2);
    builder.CreateStore(chunk_elems, c_elems_field);
    llvm::Value* c_total_field = builder.CreateStructGEP(tensor_type, chunk_ptr, 3);
    builder.CreateStore(chunk_elements, c_total_field);

    // Load current list tail (stored as int64)
    llvm::Value* current_tail_int = builder.CreateLoad(ctx_.int64Type(), list_acc);

    // Allocate cons cell with header from arena (consolidated pointer format)
    llvm::Value* cons_cell = builder.CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});

    // Set car to tensor using tagged heap pointer
    // Must pass pointer to tagged value, not value itself
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* chunk_tagged = tagged_.packHeapPtr(chunk_ptr);
    llvm::Value* chunk_ptr_alloca = builder.CreateAlloca(ctx_.taggedValueType());
    builder.CreateStore(chunk_tagged, chunk_ptr_alloca);
    builder.CreateCall(mem_.getTaggedConsSetTaggedValue(),
        {cons_cell, is_car, chunk_ptr_alloca});

    // Set cdr to current tail
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    llvm::Value* is_null_tail = builder.CreateICmpEQ(current_tail_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Branch for null vs non-null cdr
    llvm::BasicBlock* set_null_cdr = llvm::BasicBlock::Create(ctx_.context(), "spl_null_cdr", current_func);
    llvm::BasicBlock* set_cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "spl_cons_cdr", current_func);
    llvm::BasicBlock* cdr_done = llvm::BasicBlock::Create(ctx_.context(), "spl_cdr_done", current_func);

    builder.CreateCondBr(is_null_tail, set_null_cdr, set_cons_cdr);

    builder.SetInsertPoint(set_null_cdr);
    builder.CreateCall(mem_.getTaggedConsSetNull(), {cons_cell, is_cdr});
    builder.CreateBr(cdr_done);

    builder.SetInsertPoint(set_cons_cdr);
    llvm::Value* cons_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    builder.CreateCall(mem_.getTaggedConsSetPtr(),
        {cons_cell, is_cdr, current_tail_int, cons_type});
    builder.CreateBr(cdr_done);

    builder.SetInsertPoint(cdr_done);

    // Update list accumulator to point to new cons cell
    llvm::Value* cons_cell_int = builder.CreatePtrToInt(cons_cell, ctx_.int64Type());
    builder.CreateStore(cons_cell_int, list_acc);

    // Decrement chunk index
    builder.CreateStore(builder.CreateSub(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), chunk_i);
    builder.CreateBr(chunk_cond);

    builder.SetInsertPoint(chunk_done);
    llvm::Value* final_result_int = builder.CreateLoad(ctx_.int64Type(), list_acc);

    // Return the list (or null if empty)
    llvm::Value* is_null = builder.CreateICmpEQ(final_result_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    return builder.CreateSelect(is_null, tagged_.packNull(),
        tagged_.packHeapPtr(builder.CreateIntToPtr(final_result_int, ctx_.ptrType())));
}

llvm::Value* TensorCodegen::slice(const eshkol_operations_t* op) {
    // Slice: extract subtensor
    // (slice tensor start end) - 1D slice from start to end (exclusive)
    if (op->call_op.num_vars != 3) {
        eshkol_error("slice requires 3 arguments (tensor, start, end)");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::Value* start_arg = codegenAST(&op->call_op.variables[1]);
    if (!start_arg) return nullptr;

    llvm::Value* end_arg = codegenAST(&op->call_op.variables[2]);
    if (!end_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* start = start_arg;
    if (start_arg->getType() == ctx_.taggedValueType()) {
        start = tagged_.unpackInt64(start_arg);
    } else if (!start_arg->getType()->isIntegerTy(64)) {
        start = builder.CreateSExtOrTrunc(start_arg, ctx_.int64Type());
    }

    llvm::Value* end = end_arg;
    if (end_arg->getType() == ctx_.taggedValueType()) {
        end = tagged_.unpackInt64(end_arg);
    } else if (!end_arg->getType()->isIntegerTy(64)) {
        end = builder.CreateSExtOrTrunc(end_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "slice");
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);

    // Allocate result tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "slice_result");

    // Slice length
    llvm::Value* slice_len = builder.CreateSub(end, start);

    // Allocate dimensions (1D result)
    llvm::Value* dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Function* slice_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(slice_arena_alloc, {arena_ptr, dims_size}, "slice_dims");
    builder.CreateStore(slice_len, result_dims);

    // Point to slice of elements (view - no copy)
    llvm::Value* slice_elems = builder.CreateGEP(ctx_.int64Type(), src_elems, start);

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), r_ndim_field);  // 1D
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(slice_elems, r_elems_field);  // View - shares data!
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(slice_len, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::tensorShape(const eshkol_operations_t* op) {
    // tensor-shape: (tensor-shape tensor) -> returns dimensions as a Scheme list
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-shape requires exactly 1 tensor argument");
        return nullptr;
    }

    llvm::Value* tensor_val = codegenAST(&op->call_op.variables[0]);
    if (!tensor_val) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* tensor_ptr = unpackTensorOperandChecked(tensor_val, "tensor-shape");

    // Load num_dimensions
    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = ctx_.builder().CreateLoad(ctx_.int64Type(), num_dims_field_ptr);

    // Load dimensions array pointer
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), dims_field_ptr);

    // Build a proper cons-based list from dimensions (build from end to front)
    // Start with null (empty list) and prepend each dimension
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::GlobalVariable* arena_global = ctx_.globalArena();

    if (!arena_global) {
        eshkol_error("tensor-shape requires arena for list allocation");
        return tagged_.packNull();
    }
    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global);

    // Create alloca at function entry for the accumulator
    llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock& entry = current_func->getEntryBlock();
    llvm::IRBuilder<> entry_builder(&entry, entry.getFirstInsertionPt());
    llvm::Value* result_alloca = entry_builder.CreateAlloca(ctx_.int64Type(), nullptr, "shape_result");
    llvm::Value* counter_alloca = entry_builder.CreateAlloca(ctx_.int64Type(), nullptr, "shape_i");

    ctx_.builder().SetInsertPoint(current_block);

    // Initialize: result = 0 (null), counter = num_dims - 1 (iterate backwards)
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), result_alloca);
    llvm::Value* start_idx = ctx_.builder().CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(start_idx, counter_alloca);

    // Loop condition
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "shape_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "shape_body", current_func);
    llvm::BasicBlock* loop_exit = llvm::BasicBlock::Create(ctx_.context(), "shape_exit", current_func);

    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), counter_alloca);
    // Loop while i >= 0
    llvm::Value* cond = ctx_.builder().CreateICmpSGE(i, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    ctx_.builder().CreateCondBr(cond, loop_body, loop_exit);

    ctx_.builder().SetInsertPoint(loop_body);

    // Load dimension value at index i
    llvm::Value* dim_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), dims_ptr, i);
    llvm::Value* dim_val = ctx_.builder().CreateLoad(ctx_.int64Type(), dim_ptr);

    // Load current list tail
    llvm::Value* current_tail_int = ctx_.builder().CreateLoad(ctx_.int64Type(), result_alloca);

    // Allocate new cons cell with object header (consolidated pointer format)
    // M1 Migration: Use header allocator for HEAP_PTR compatibility
    llvm::Value* cons_cell = ctx_.builder().CreateCall(
        mem_.getArenaAllocateConsWithHeader(), {arena_ptr});

    // Set car to dimension value (int64)
    llvm::Value* is_car = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    llvm::Value* int_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetInt64(),
        {cons_cell, is_car, dim_val, int_type});

    // Set cdr to current tail
    llvm::Value* is_cdr = llvm::ConstantInt::get(ctx_.int1Type(), 1);
    llvm::Value* is_null_tail = ctx_.builder().CreateICmpEQ(current_tail_int,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    // Branch for null vs non-null cdr
    llvm::BasicBlock* set_null_cdr = llvm::BasicBlock::Create(ctx_.context(), "set_null_cdr", current_func);
    llvm::BasicBlock* set_cons_cdr = llvm::BasicBlock::Create(ctx_.context(), "set_cons_cdr", current_func);
    llvm::BasicBlock* cdr_done = llvm::BasicBlock::Create(ctx_.context(), "cdr_done", current_func);

    ctx_.builder().CreateCondBr(is_null_tail, set_null_cdr, set_cons_cdr);

    ctx_.builder().SetInsertPoint(set_null_cdr);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetNull(), {cons_cell, is_cdr});
    ctx_.builder().CreateBr(cdr_done);

    ctx_.builder().SetInsertPoint(set_cons_cdr);
    llvm::Value* cons_type = llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR);
    ctx_.builder().CreateCall(mem_.getTaggedConsSetPtr(),
        {cons_cell, is_cdr, current_tail_int, cons_type});
    ctx_.builder().CreateBr(cdr_done);

    ctx_.builder().SetInsertPoint(cdr_done);

    // Update result to point to new cons cell
    llvm::Value* cons_cell_int = ctx_.builder().CreatePtrToInt(cons_cell, ctx_.int64Type());
    ctx_.builder().CreateStore(cons_cell_int, result_alloca);

    // Decrement counter
    llvm::Value* next_i = ctx_.builder().CreateSub(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, counter_alloca);
    ctx_.builder().CreateBr(loop_cond);

    ctx_.builder().SetInsertPoint(loop_exit);

    // Load final result
    llvm::Value* final_result_int = ctx_.builder().CreateLoad(ctx_.int64Type(), result_alloca);
    return tagged_.packHeapPtr(ctx_.builder().CreateIntToPtr(final_result_int, ctx_.ptrType()));
}

llvm::Value* TensorCodegen::tensorLength(const eshkol_operations_t* op) {
    // tensor-length: (tensor-length tensor) -> total number of elements
    if (op->call_op.num_vars != 1) {
        eshkol_error("tensor-length requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* arg = codegenAST(&op->call_op.variables[0]);
    if (!arg) return nullptr;

    llvm::Value* tensor_ptr = unpackTensorOperandChecked(arg, "tensor-length");

    // Field 3 of tensor struct is total_elements (int64)
    llvm::Value* total_field = ctx_.builder().CreateStructGEP(ctx_.tensorType(), tensor_ptr, 3);
    llvm::Value* total = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field);

    return tagged_.packInt64(total);
}

llvm::Value* TensorCodegen::transpose(const eshkol_operations_t* op) {
    // transpose: (transpose tensor) - Transpose 2D matrix (swap rows and cols)
    if (op->call_op.num_vars != 1) {
        eshkol_error("transpose requires exactly 1 argument");
        return nullptr;
    }

    llvm::Value* src_tensor = codegenAST(&op->call_op.variables[0]);
    if (!src_tensor) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // ESH-0069: validate/coerce the operand up front. A non-tensor now raises a
    // catchable type error (a homogeneous numeric vector is coerced) instead of
    // the legacy behavior of silently returning null via the error_block below.
    llvm::Value* src_ptr = unpackTensorOperandChecked(src_tensor, "transpose");
    llvm::Value* is_tensor = llvm::ConstantInt::getTrue(ctx_.context());

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_tensor", current_func);
    llvm::BasicBlock* error_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_error", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "transpose_exit", current_func);

    // Use alloca-based merge (avoids PHI predecessor issues with XLA blocks)
    llvm::Value* result_alloca = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "trans_result");

    ctx_.builder().CreateCondBr(is_tensor, tensor_block, error_block);

    // Error path - return null for non-tensor inputs
    ctx_.builder().SetInsertPoint(error_block);
    llvm::Value* error_result = tagged_.packNull();
    ctx_.builder().CreateStore(error_result, result_alloca);
    ctx_.builder().CreateBr(exit_block);

    // Tensor path - proceed with normal transpose (src_ptr validated above)
    ctx_.builder().SetInsertPoint(tensor_block);

    // Get source tensor properties
    llvm::Value* src_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 0);
    llvm::Value* src_dims_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_dims_field_ptr);
    llvm::Value* typed_src_dims_ptr = ctx_.builder().CreatePointerCast(src_dims_ptr, ctx_.ptrType());

    // Guard: transpose only supports 2D tensors
    {
        llvm::Value* src_ndim_field = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 1);
        llvm::Value* src_ndim = ctx_.builder().CreateLoad(ctx_.int64Type(), src_ndim_field);
        llvm::Value* not_2d = ctx_.builder().CreateICmpNE(src_ndim,
            llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* trans_dims_ok = llvm::BasicBlock::Create(ctx_.context(), "transpose_dims_ok", current_func);
        llvm::BasicBlock* trans_dims_err = llvm::BasicBlock::Create(ctx_.context(), "transpose_dims_err", current_func);
        ctx_.builder().CreateCondBr(not_2d, trans_dims_err, trans_dims_ok);

        ctx_.builder().SetInsertPoint(trans_dims_err);
        llvm::Function* printf_fn_trans = ctx_.lookupFunction("printf");
        llvm::Function* exit_fn_trans = ctx_.lookupFunction("exit");
        if (printf_fn_trans && exit_fn_trans) {
            llvm::Value* fmt = ctx_.builder().CreateGlobalString(
                "Error: transpose only supports 2D tensors (got %lld dimensions)\n");
            ctx_.builder().CreateCall(printf_fn_trans, {fmt, src_ndim});
            ctx_.builder().CreateCall(exit_fn_trans, {llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(ctx_.context()), 1)});
        }
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(trans_dims_ok);
    }

#ifdef ESHKOL_XLA_ENABLED
    if (xla_ && xla_->isAvailable()) {
        // Check if tensor is large enough for XLA dispatch
        llvm::Value* total_field = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 3);
        llvm::Value* total_elements = ctx_.builder().CreateLoad(ctx_.int64Type(), total_field, "trans_total");
        llvm::Value* threshold = llvm::ConstantInt::get(ctx_.int64Type(), xla::xla_get_threshold());
        llvm::Value* use_xla = ctx_.builder().CreateICmpUGE(total_elements, threshold);

        llvm::BasicBlock* xla_block = llvm::BasicBlock::Create(ctx_.context(), "trans_xla", current_func);
        llvm::BasicBlock* cpu_block = llvm::BasicBlock::Create(ctx_.context(), "trans_cpu", current_func);
        ctx_.builder().CreateCondBr(use_xla, xla_block, cpu_block);

        // XLA path
        ctx_.builder().SetInsertPoint(xla_block);
        llvm::Value* xla_result = xla_->emitTranspose(src_ptr);
        if (xla_result) {
            llvm::Value* xla_tagged = tagged_.packHeapPtr(xla_result);
            ctx_.builder().CreateStore(xla_tagged, result_alloca);
            ctx_.builder().CreateBr(exit_block);
        } else {
            ctx_.builder().CreateBr(cpu_block);
        }

        // CPU fallback
        ctx_.builder().SetInsertPoint(cpu_block);
    }
#endif

    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);
    llvm::Value* typed_src_elements_ptr = ctx_.builder().CreatePointerCast(src_elements_ptr, ctx_.ptrType());

    // Get rows and cols
    llvm::Value* rows = ctx_.builder().CreateLoad(ctx_.int64Type(), typed_src_dims_ptr);
    llvm::Value* dim1_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* cols = ctx_.builder().CreateLoad(ctx_.int64Type(), dim1_ptr);

    // Create result tensor with swapped dimensions [cols, rows]
    std::vector<llvm::Value*> new_dims = {cols, rows};
    llvm::Value* result_ptr = createTensorWithDims(new_dims, nullptr, false);
    if (!result_ptr) return nullptr;

    // Get result elements pointer
    llvm::Value* result_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, result_ptr, 2);
    llvm::Value* result_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), result_elements_field_ptr);
    llvm::Value* typed_result_elements_ptr = ctx_.builder().CreatePointerCast(result_elements_ptr, ctx_.ptrType());

    // Transpose: result[j][i] = src[i][j]
    llvm::BasicBlock* row_cond = llvm::BasicBlock::Create(ctx_.context(), "trans_row_cond", current_func);
    llvm::BasicBlock* row_body = llvm::BasicBlock::Create(ctx_.context(), "trans_row_body", current_func);
    llvm::BasicBlock* col_cond = llvm::BasicBlock::Create(ctx_.context(), "trans_col_cond", current_func);
    llvm::BasicBlock* col_body = llvm::BasicBlock::Create(ctx_.context(), "trans_col_body", current_func);
    llvm::BasicBlock* col_exit = llvm::BasicBlock::Create(ctx_.context(), "trans_col_exit", current_func);
    llvm::BasicBlock* row_exit = llvm::BasicBlock::Create(ctx_.context(), "trans_row_exit", current_func);

    llvm::Value* row_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "trans_i");
    llvm::Value* col_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "trans_j");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), row_counter);
    ctx_.builder().CreateBr(row_cond);

    ctx_.builder().SetInsertPoint(row_cond);
    llvm::Value* i = ctx_.builder().CreateLoad(ctx_.int64Type(), row_counter);
    llvm::Value* row_cmp = ctx_.builder().CreateICmpULT(i, rows);
    ctx_.builder().CreateCondBr(row_cmp, row_body, row_exit);

    ctx_.builder().SetInsertPoint(row_body);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_counter);
    ctx_.builder().CreateBr(col_cond);

    ctx_.builder().SetInsertPoint(col_cond);
    llvm::Value* j = ctx_.builder().CreateLoad(ctx_.int64Type(), col_counter);
    llvm::Value* col_cmp = ctx_.builder().CreateICmpULT(j, cols);
    ctx_.builder().CreateCondBr(col_cmp, col_body, col_exit);

    ctx_.builder().SetInsertPoint(col_body);
    // src_idx = i * cols + j
    llvm::Value* src_idx = ctx_.builder().CreateMul(i, cols);
    src_idx = ctx_.builder().CreateAdd(src_idx, j);
    // dst_idx = j * rows + i
    llvm::Value* dst_idx = ctx_.builder().CreateMul(j, rows);
    dst_idx = ctx_.builder().CreateAdd(dst_idx, i);

    llvm::Value* src_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_src_elements_ptr, src_idx);
    llvm::Value* elem = ctx_.builder().CreateLoad(ctx_.int64Type(), src_elem_ptr);
    llvm::Value* dst_elem_ptr = ctx_.builder().CreateGEP(ctx_.int64Type(), typed_result_elements_ptr, dst_idx);
    ctx_.builder().CreateStore(elem, dst_elem_ptr);

    llvm::Value* next_j = ctx_.builder().CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_j, col_counter);
    ctx_.builder().CreateBr(col_cond);

    ctx_.builder().SetInsertPoint(col_exit);
    llvm::Value* next_i = ctx_.builder().CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_i, row_counter);
    ctx_.builder().CreateBr(row_cond);

    ctx_.builder().SetInsertPoint(row_exit);
    llvm::Value* tensor_result = tagged_.packHeapPtr(result_ptr);
    ctx_.builder().CreateStore(tensor_result, result_alloca);
    ctx_.builder().CreateBr(exit_block);

    // Merge — load from alloca (all paths store their result)
    ctx_.builder().SetInsertPoint(exit_block);
    return ctx_.builder().CreateLoad(ctx_.taggedValueType(), result_alloca, "transpose_result");
}

llvm::Value* TensorCodegen::reshape(const eshkol_operations_t* op) {
    // reshape: (reshape tensor dim1 dim2 ...) OR (reshape tensor (list d1 d2 ...))
    // Support both individual dimension args and a list of dimensions
    if (op->call_op.num_vars < 2) {
        eshkol_error("reshape requires tensor and at least 1 dimension");
        return nullptr;
    }

    // Get source value (may be tensor OR Scheme vector)
    llvm::Value* src_val = codegenAST(&op->call_op.variables[0]);
    if (!src_val) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Check type: Scheme vector vs Tensor (using consolidated type check)
    llvm::Value* is_scheme_vector = tagged_.isVector(src_val);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* scheme_vec_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_scheme_vec", current_func);
    llvm::BasicBlock* tensor_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_tensor", current_func);
    llvm::BasicBlock* type_merge = llvm::BasicBlock::Create(ctx_.context(), "reshape_type_merge", current_func);

    ctx_.builder().CreateCondBr(is_scheme_vector, scheme_vec_block, tensor_block);

    // === SCHEME VECTOR PATH: Convert to tensor first ===
    ctx_.builder().SetInsertPoint(scheme_vec_block);
    llvm::Value* svec_ptr_int = tagged_.unpackInt64(src_val);
    llvm::Value* svec_ptr = ctx_.builder().CreateIntToPtr(svec_ptr_int, ctx_.ptrType());

    // Scheme vector layout: [length:i64, element0:tagged_value, element1:tagged_value, ...]
    llvm::Value* svec_len = ctx_.builder().CreateLoad(ctx_.int64Type(), svec_ptr);

    // Allocate arena for conversion
    llvm::Value* conv_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate tensor structure with header
    llvm::Function* conv_alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* conv_tensor_ptr = ctx_.builder().CreateCall(conv_alloc_tensor_func, {conv_arena_ptr}, "vec_to_tensor");

    // Null check: arena allocation can fail on OOM
    {
        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(conv_tensor_ptr,
            llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_.context(), 0)));
        llvm::Function* curr_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* alloc_ok = llvm::BasicBlock::Create(ctx_.context(), "vec2tensor_ok", curr_fn);
        llvm::BasicBlock* alloc_fail = llvm::BasicBlock::Create(ctx_.context(), "vec2tensor_oom", curr_fn);
        ctx_.builder().CreateCondBr(is_null, alloc_fail, alloc_ok);
        ctx_.builder().SetInsertPoint(alloc_fail);
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* raise_type = llvm::FunctionType::get(ctx_.builder().getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc_func) {
            llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.builder().getInt32Ty(), ctx_.ptrType()}, false);
            make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* err_msg = ctx_.builder().CreateGlobalString("vector->tensor: allocation failed (out of memory)");
        llvm::Value* exception = ctx_.builder().CreateCall(make_exc_func,
            {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1), err_msg});
        ctx_.builder().CreateCall(raise_func, {exception});
        ctx_.builder().CreateUnreachable();
        ctx_.builder().SetInsertPoint(alloc_ok);
    }

    // Allocate dimensions array (1D tensor)
    llvm::Function* conv_arena_alloc = mem_.getArenaAllocate();
    llvm::Value* conv_dims_size = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
    llvm::Value* conv_dims_ptr = ctx_.builder().CreateCall(conv_arena_alloc, {conv_arena_ptr, conv_dims_size}, "conv_dims");
    ctx_.builder().CreateStore(svec_len, conv_dims_ptr);

    // ESH-0121: detect a Scheme vector of forward-mode DUAL_NUMBER jets (the
    // Hessian's forward-over-forward sweep). When present, build a *dual tensor*:
    // elements are copied verbatim as 16-byte tagged DUAL_NUMBER values and dtype
    // is set to DUAL, so the downstream dual-aware matmul/tensor-sum paths keep
    // the e1/e2/e1e2 second-order terms instead of flattening each jet to a plain
    // double (which silently zeros the Hessian). The non-dual path is unchanged.
    llvm::Value* svec_data_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), svec_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));  // Skip length
    llvm::Value* svec_typed_ptr = ctx_.builder().CreatePointerCast(svec_data_ptr, ctx_.ptrType());

    // Peek element 0's tag (guarded on a non-empty vector) to decide dual vs plain.
    llvm::Value* svec_nonempty = ctx_.builder().CreateICmpUGT(svec_len,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::BasicBlock* peek_bb = llvm::BasicBlock::Create(ctx_.context(), "reshape_peek", current_func);
    llvm::BasicBlock* peek_skip = llvm::BasicBlock::Create(ctx_.context(), "reshape_peek_skip", current_func);
    llvm::BasicBlock* peek_merge = llvm::BasicBlock::Create(ctx_.context(), "reshape_peek_merge", current_func);
    ctx_.builder().CreateCondBr(svec_nonempty, peek_bb, peek_skip);

    ctx_.builder().SetInsertPoint(peek_bb);
    llvm::Value* e0_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(),
        ctx_.builder().CreateGEP(ctx_.taggedValueType(), svec_typed_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    llvm::Value* e0_is_dual = ctx_.builder().CreateICmpEQ(
        tagged_.getBaseType(tagged_.getType(e0_tagged)),
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DUAL_NUMBER));
    llvm::BasicBlock* peek_bb_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(peek_merge);

    ctx_.builder().SetInsertPoint(peek_skip);
    llvm::BasicBlock* peek_skip_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(peek_merge);

    ctx_.builder().SetInsertPoint(peek_merge);
    llvm::PHINode* is_dual_vec = ctx_.builder().CreatePHI(ctx_.int1Type(), 2, "reshape_is_dual");
    is_dual_vec->addIncoming(e0_is_dual, peek_bb_exit);
    is_dual_vec->addIncoming(llvm::ConstantInt::getFalse(ctx_.context()), peek_skip_exit);

    // Element stride: 16 bytes (tagged jet) when dual, else 8 bytes (double).
    llvm::Value* conv_elem_stride = ctx_.builder().CreateSelect(is_dual_vec,
        llvm::ConstantInt::get(ctx_.int64Type(), 16),
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* conv_elems_size = ctx_.builder().CreateMul(svec_len, conv_elem_stride);
    llvm::Value* conv_elems_ptr = ctx_.builder().CreateCall(conv_arena_alloc, {conv_arena_ptr, conv_elems_size}, "conv_elems");

    // Copy loop
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_body", current_func);
    llvm::BasicBlock* copy_dual = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_dual", current_func);
    llvm::BasicBlock* copy_num = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_num", current_func);
    llvm::BasicBlock* copy_incr = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_incr", current_func);
    llvm::BasicBlock* copy_exit = llvm::BasicBlock::Create(ctx_.context(), "reshape_copy_exit", current_func);

    llvm::Value* copy_counter = ctx_.builder().CreateAlloca(ctx_.int64Type(), nullptr, "copy_i");
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_counter);
    ctx_.builder().CreateBr(copy_cond);

    ctx_.builder().SetInsertPoint(copy_cond);
    llvm::Value* copy_i = ctx_.builder().CreateLoad(ctx_.int64Type(), copy_counter);
    llvm::Value* copy_cmp = ctx_.builder().CreateICmpULT(copy_i, svec_len);
    ctx_.builder().CreateCondBr(copy_cmp, copy_body, copy_exit);

    ctx_.builder().SetInsertPoint(copy_body);
    // Load tagged value from vector
    llvm::Value* elem_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(),
        svec_typed_ptr, copy_i);
    llvm::Value* elem_tagged = ctx_.builder().CreateLoad(ctx_.taggedValueType(), elem_ptr);
    ctx_.builder().CreateCondBr(is_dual_vec, copy_dual, copy_num);

    // Dual path: store the 16-byte tagged jet verbatim (preserves e1/e2/e1e2).
    ctx_.builder().SetInsertPoint(copy_dual);
    llvm::Value* dual_dest_ptr = ctx_.builder().CreateGEP(ctx_.taggedValueType(), conv_elems_ptr, copy_i);
    ctx_.builder().CreateStore(elem_tagged, dual_dest_ptr);
    ctx_.builder().CreateBr(copy_incr);

    // Numeric path: flatten to a plain double (unchanged legacy behavior).
    ctx_.builder().SetInsertPoint(copy_num);
    llvm::Value* elem_double = extractAsDouble(elem_tagged);
    llvm::Value* dest_ptr = ctx_.builder().CreateGEP(ctx_.doubleType(), conv_elems_ptr, copy_i);
    ctx_.builder().CreateStore(elem_double, dest_ptr);
    ctx_.builder().CreateBr(copy_incr);

    ctx_.builder().SetInsertPoint(copy_incr);
    llvm::Value* next_copy_i = ctx_.builder().CreateAdd(copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    ctx_.builder().CreateStore(next_copy_i, copy_counter);
    ctx_.builder().CreateBr(copy_cond);

    // Exit copy loop: populate tensor struct
    ctx_.builder().SetInsertPoint(copy_exit);

    // Store dims pointer (field 0)
    llvm::Value* conv_dims_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 0);
    ctx_.builder().CreateStore(conv_dims_ptr, conv_dims_field);

    // Store num_dimensions = 1 (field 1)
    llvm::Value* conv_ndim_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 1);
    ctx_.builder().CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), conv_ndim_field);

    // Store elements pointer (field 2)
    llvm::Value* conv_elems_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 2);
    ctx_.builder().CreateStore(conv_elems_ptr, conv_elems_field);

    // Store total_elements (field 3)
    llvm::Value* conv_total_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 3);
    ctx_.builder().CreateStore(svec_len, conv_total_field);

    // ESH-0121: dtype (field 4) — DUAL when elements are tagged jets, else f64(0).
    llvm::Value* conv_dtype_field = ctx_.builder().CreateStructGEP(tensor_type, conv_tensor_ptr, 4);
    ctx_.builder().CreateStore(
        ctx_.builder().CreateSelect(is_dual_vec,
            llvm::ConstantInt::get(ctx_.int64Type(), TensorCodegen::TENSOR_DTYPE_DUAL),
            llvm::ConstantInt::get(ctx_.int64Type(), 0)),
        conv_dtype_field);

    ctx_.builder().CreateBr(type_merge);
    llvm::BasicBlock* svec_exit_block = ctx_.builder().GetInsertBlock();

    // === TENSOR PATH: Use existing tensor directly ===
    ctx_.builder().SetInsertPoint(tensor_block);
    llvm::Value* direct_tensor_ptr = unpackTensorOperandChecked(src_val, "reshape");
    ctx_.builder().CreateBr(type_merge);
    llvm::BasicBlock* tensor_exit_block = ctx_.builder().GetInsertBlock();

    // === MERGE: Get unified tensor pointer ===
    ctx_.builder().SetInsertPoint(type_merge);
    llvm::PHINode* src_ptr = ctx_.builder().CreatePHI(ctx_.ptrType(), 2, "unified_tensor_ptr");
    src_ptr->addIncoming(conv_tensor_ptr, svec_exit_block);
    src_ptr->addIncoming(direct_tensor_ptr, tensor_exit_block);

    // Get new dimensions - three cases:
    // 1. Individual args: (reshape tensor 3 3) -> num_vars > 2
    // 2. List arg: (reshape tensor (list 3 3 2)) -> num_vars == 2 and arg is list
    // 3. Single dim: (reshape tensor 9) -> num_vars == 2 and arg is integer
    // All paths produce: final_dims_ptr (int64_t*), final_ndim (i64), final_total (i64)

    llvm::Value* final_dims_ptr = nullptr;
    llvm::Value* final_ndim = nullptr;
    llvm::Value* final_total = nullptr;

    auto extract_structural_int = [&](llvm::Value* value,
                                      const char* proc_name,
                                      const char* expected_type) -> llvm::Value* {
        auto set_error_location = [&]() {
            uint32_t line = ctx_.currentSourceLine();
            if (line == 0) {
                return;
            }

            llvm::Function* set_loc_fn = ctx_.module().getFunction("eshkol_set_error_location");
            if (!set_loc_fn) {
                llvm::FunctionType* ft = llvm::FunctionType::get(
                    ctx_.builder().getVoidTy(),
                    {ctx_.builder().getPtrTy(), ctx_.int32Type(), ctx_.int32Type()},
                    false);
                set_loc_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                    "eshkol_set_error_location", &ctx_.module());
            }

            const std::string& file = ctx_.currentSourceFile();
            llvm::Value* file_val = file.empty()
                ? static_cast<llvm::Value*>(llvm::ConstantPointerNull::get(ctx_.builder().getPtrTy()))
                : static_cast<llvm::Value*>(ctx_.builder().CreateGlobalString(file, "struct_int_file"));
            ctx_.builder().CreateCall(set_loc_fn, {
                file_val,
                llvm::ConstantInt::get(ctx_.int32Type(), line),
                llvm::ConstantInt::get(ctx_.int32Type(), ctx_.currentSourceColumn())});
        };

        if (!value) {
            return llvm::ConstantInt::get(ctx_.int64Type(), 0);
        }

        llvm::Type* value_type = value->getType();
        if (value_type->isIntegerTy(64)) {
            return value;
        }
        if (value_type->isIntegerTy()) {
            return ctx_.builder().CreateSExtOrTrunc(value, ctx_.int64Type());
        }

        if (value_type != ctx_.taggedValueType()) {
            llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
            llvm::BasicBlock* err_block = llvm::BasicBlock::Create(
                ctx_.context(), "reshape_dim_type_error", func);
            ctx_.builder().CreateBr(err_block);
            ctx_.builder().SetInsertPoint(err_block);
            set_error_location();

            llvm::Function* type_error_fn = ctx_.module().getFunction("eshkol_type_error");
            if (!type_error_fn) {
                llvm::FunctionType* ft = llvm::FunctionType::get(
                    ctx_.builder().getVoidTy(),
                    {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()},
                    false);
                type_error_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                    "eshkol_type_error", &ctx_.module());
                type_error_fn->setDoesNotReturn();
            }

            llvm::Value* proc = ctx_.builder().CreateGlobalString(proc_name, "struct_int_proc");
            llvm::Value* expected = ctx_.builder().CreateGlobalString(expected_type, "struct_int_expected");
            ctx_.builder().CreateCall(type_error_fn, {proc, expected});
            ctx_.builder().CreateUnreachable();
            return llvm::ConstantInt::get(ctx_.int64Type(), 0);
        }

        llvm::Value* type_tag = tagged_.getType(value);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        llvm::Value* is_int = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* ok_block = llvm::BasicBlock::Create(
            ctx_.context(), "reshape_dim_int_ok", func);
        llvm::BasicBlock* err_block = llvm::BasicBlock::Create(
            ctx_.context(), "reshape_dim_type_error", func);
        ctx_.builder().CreateCondBr(is_int, ok_block, err_block);

        ctx_.builder().SetInsertPoint(err_block);
        set_error_location();
        llvm::Function* type_error_fn = ctx_.module().getFunction("eshkol_type_error_with_operand");
        if (!type_error_fn) {
            llvm::FunctionType* ft = llvm::FunctionType::get(
                ctx_.builder().getVoidTy(),
                {ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy(), ctx_.builder().getPtrTy()},
                false);
            type_error_fn = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                "eshkol_type_error_with_operand", &ctx_.module());
            type_error_fn->setDoesNotReturn();
        }
        llvm::Value* slot = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "reshape_dim_err_slot");
        ctx_.builder().CreateStore(value, slot);
        llvm::Value* proc = ctx_.builder().CreateGlobalString(proc_name, "struct_int_proc");
        llvm::Value* expected = ctx_.builder().CreateGlobalString(expected_type, "struct_int_expected");
        ctx_.builder().CreateCall(type_error_fn, {proc, expected, slot});
        ctx_.builder().CreateUnreachable();

        ctx_.builder().SetInsertPoint(ok_block);
        return tagged_.unpackInt64(value);
    };

    // Get source tensor properties (needed for all paths)
    llvm::Value* src_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 2);
    llvm::Value* src_elements_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), src_elements_field_ptr);

    llvm::Function* arena_alloc = mem_.getArenaAllocate();

    if (op->call_op.num_vars == 2) {
        // Could be a single dimension OR a list of dimensions
        llvm::Value* dim_arg = codegenAST(&op->call_op.variables[1]);
        if (!dim_arg) return nullptr;

        // Check if it's a HEAP_PTR (could be cons list OR tensor)
        llvm::Value* type_tag = tagged_.getType(dim_arg);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        llvm::Value* is_heap = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_HEAP_PTR));

        llvm::Function* func = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* check_subtype = llvm::BasicBlock::Create(ctx_.context(), "reshape_check_sub", func);
        llvm::BasicBlock* tensor_dims_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_tensor_dims", func);
        llvm::BasicBlock* list_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_list", func);
        llvm::BasicBlock* single_path = llvm::BasicBlock::Create(ctx_.context(), "reshape_single", func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "reshape_merge", func);

        ctx_.builder().CreateCondBr(is_heap, check_subtype, single_path);

        // CHECK SUBTYPE: is it a tensor or a cons list?
        ctx_.builder().SetInsertPoint(check_subtype);
        llvm::Value* heap_ptr_int = tagged_.unpackInt64(dim_arg);
        llvm::Value* heap_ptr = ctx_.builder().CreateIntToPtr(heap_ptr_int, ctx_.ptrType());
        // Header is at ptr - 8, subtype is the first byte
        llvm::Value* header_ptr = ctx_.builder().CreateGEP(ctx_.int8Type(), heap_ptr,
            llvm::ConstantInt::get(ctx_.int64Type(), -8));
        llvm::Value* subtype = ctx_.builder().CreateLoad(ctx_.int8Type(), header_ptr);
        llvm::Value* is_tensor = ctx_.builder().CreateICmpEQ(subtype,
            llvm::ConstantInt::get(ctx_.int8Type(), HEAP_SUBTYPE_TENSOR));
        ctx_.builder().CreateCondBr(is_tensor, tensor_dims_path, list_path);

        // TENSOR DIMS PATH: call runtime to extract dims from tensor elements
        ctx_.builder().SetInsertPoint(tensor_dims_path);
        llvm::Value* t_arena = ctx_.builder().CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        llvm::Value* td_max_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 16 * sizeof(int64_t));
        llvm::Value* td_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {t_arena, td_max_bytes}, "tensor_dims");

        auto* t2d_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* t2d_fn = ctx_.module().getFunction("eshkol_tensor_to_dims");
        if (!t2d_fn) {
            t2d_fn = llvm::Function::Create(t2d_ft,
                llvm::Function::ExternalLinkage, "eshkol_tensor_to_dims", &ctx_.module());
        }
        llvm::Value* td_ndim = ctx_.builder().CreateCall(
            t2d_fn, {heap_ptr, td_dims_array, llvm::ConstantInt::get(ctx_.int64Type(), 16)},
            "tensor_ndim");

        auto* td_total_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* td_total_fn = ctx_.module().getFunction("eshkol_compute_dims_total");
        if (!td_total_fn) {
            td_total_fn = llvm::Function::Create(td_total_ft,
                llvm::Function::ExternalLinkage, "eshkol_compute_dims_total", &ctx_.module());
        }
        llvm::Value* td_total = ctx_.builder().CreateCall(
            td_total_fn, {td_dims_array, td_ndim}, "tensor_total");

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* tensor_dims_exit = ctx_.builder().GetInsertBlock();

        // LIST PATH: Walk cons list to extract N dimensions via runtime helper
        ctx_.builder().SetInsertPoint(list_path);
        llvm::Value* cons_ptr = ctx_.builder().CreateIntToPtr(heap_ptr_int, ctx_.ptrType());

        // Allocate dims array for up to 16 dimensions
        llvm::Value* list_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* max_dims_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 16 * sizeof(int64_t));
        llvm::Value* list_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {list_arena, max_dims_bytes}, "list_dims");

        // Call runtime: int64_t eshkol_cons_list_to_dims(void*, int64_t*, int64_t)
        auto* list_to_dims_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* list_to_dims_fn = ctx_.module().getFunction("eshkol_cons_list_to_dims");
        if (!list_to_dims_fn) {
            list_to_dims_fn = llvm::Function::Create(list_to_dims_ft,
                llvm::Function::ExternalLinkage, "eshkol_cons_list_to_dims", &ctx_.module());
        }
        {
            uint32_t line = ctx_.currentSourceLine();
            if (line != 0) {
                llvm::Function* set_loc_fn = ctx_.module().getFunction("eshkol_set_error_location");
                if (!set_loc_fn) {
                    llvm::FunctionType* set_loc_ft = llvm::FunctionType::get(
                        ctx_.builder().getVoidTy(),
                        {ctx_.builder().getPtrTy(), ctx_.int32Type(), ctx_.int32Type()},
                        false);
                    set_loc_fn = llvm::Function::Create(set_loc_ft, llvm::Function::ExternalLinkage,
                        "eshkol_set_error_location", &ctx_.module());
                }
                const std::string& file = ctx_.currentSourceFile();
                llvm::Value* file_val = file.empty()
                    ? static_cast<llvm::Value*>(llvm::ConstantPointerNull::get(ctx_.builder().getPtrTy()))
                    : static_cast<llvm::Value*>(ctx_.builder().CreateGlobalString(file, "reshape_list_file"));
                ctx_.builder().CreateCall(set_loc_fn, {
                    file_val,
                    llvm::ConstantInt::get(ctx_.int32Type(), line),
                    llvm::ConstantInt::get(ctx_.int32Type(), ctx_.currentSourceColumn())});
            }
        }
        llvm::Value* list_ndim = ctx_.builder().CreateCall(
            list_to_dims_fn,
            {cons_ptr, list_dims_array, llvm::ConstantInt::get(ctx_.int64Type(), 16)},
            "list_ndim");

        // Compute total via runtime: int64_t eshkol_compute_dims_total(int64_t*, int64_t)
        auto* total_ft = llvm::FunctionType::get(ctx_.int64Type(),
            {ctx_.ptrType(), ctx_.int64Type()}, false);
        llvm::Function* total_fn = ctx_.module().getFunction("eshkol_compute_dims_total");
        if (!total_fn) {
            total_fn = llvm::Function::Create(total_ft,
                llvm::Function::ExternalLinkage, "eshkol_compute_dims_total", &ctx_.module());
        }
        llvm::Value* list_total = ctx_.builder().CreateCall(
            total_fn, {list_dims_array, list_ndim}, "list_total");

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* list_exit = ctx_.builder().GetInsertBlock();

        // SINGLE PATH: Treat as 1D reshape
        ctx_.builder().SetInsertPoint(single_path);
        llvm::Value* single_dim = extract_structural_int(
            dim_arg, "reshape", "integer dimension");
        // Allocate 1-element dims array
        llvm::Value* single_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* single_bytes = llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t));
        llvm::Value* single_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {single_arena, single_bytes}, "single_dims");
        ctx_.builder().CreateStore(single_dim, single_dims_array);
        llvm::Value* single_ndim = llvm::ConstantInt::get(ctx_.int64Type(), 1);
        llvm::Value* single_total = single_dim;

        ctx_.builder().CreateBr(merge_block);
        llvm::BasicBlock* single_exit = ctx_.builder().GetInsertBlock();

        // MERGE: PHI nodes for dims_ptr, ndim, total
        ctx_.builder().SetInsertPoint(merge_block);

        llvm::PHINode* dims_phi = ctx_.builder().CreatePHI(ctx_.ptrType(), 3, "merged_dims");
        dims_phi->addIncoming(td_dims_array, tensor_dims_exit);
        dims_phi->addIncoming(list_dims_array, list_exit);
        dims_phi->addIncoming(single_dims_array, single_exit);

        llvm::PHINode* ndim_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "merged_ndim");
        ndim_phi->addIncoming(td_ndim, tensor_dims_exit);
        ndim_phi->addIncoming(list_ndim, list_exit);
        ndim_phi->addIncoming(single_ndim, single_exit);

        llvm::PHINode* total_phi = ctx_.builder().CreatePHI(ctx_.int64Type(), 3, "merged_total");
        total_phi->addIncoming(td_total, tensor_dims_exit);
        total_phi->addIncoming(list_total, list_exit);
        total_phi->addIncoming(single_total, single_exit);

        final_dims_ptr = dims_phi;
        final_ndim = ndim_phi;
        final_total = total_phi;
    } else {
        // Multiple explicit dimension arguments: (reshape tensor 3 3 2)
        size_t ndim_count = op->call_op.num_vars - 1;

        llvm::Value* multi_arena = ctx_.builder().CreateLoad(
            llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
        llvm::Value* multi_bytes = llvm::ConstantInt::get(ctx_.int64Type(), ndim_count * sizeof(int64_t));
        llvm::Value* multi_dims_array = ctx_.builder().CreateCall(
            arena_alloc, {multi_arena, multi_bytes}, "multi_dims");

        llvm::Value* multi_total = nullptr;
        for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
            llvm::Value* dim = codegenAST(&op->call_op.variables[i]);
            if (!dim) return nullptr;
            dim = extract_structural_int(dim, "reshape", "integer dimension");
            llvm::Value* dim_slot = ctx_.builder().CreateGEP(
                ctx_.int64Type(), multi_dims_array,
                llvm::ConstantInt::get(ctx_.int64Type(), i - 1));
            ctx_.builder().CreateStore(dim, dim_slot);

            if (multi_total == nullptr) {
                multi_total = dim;
            } else {
                multi_total = ctx_.builder().CreateMul(multi_total, dim);
            }
        }

        final_dims_ptr = multi_dims_array;
        final_ndim = llvm::ConstantInt::get(ctx_.int64Type(), ndim_count);
        final_total = multi_total;
    }

    // Allocate using arena
    llvm::Value* reshape_arena_ptr = ctx_.builder().CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Create new tensor structure with header (reuse elements - no copy needed for reshape)
    llvm::Function* alloc_tensor_func = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* typed_new_tensor_ptr = ctx_.builder().CreateCall(
        alloc_tensor_func, {reshape_arena_ptr}, "reshape_tensor");

    // Null check: arena allocation can fail on OOM
    {
        llvm::Value* is_null = ctx_.builder().CreateICmpEQ(typed_new_tensor_ptr,
            llvm::ConstantPointerNull::get(llvm::PointerType::get(ctx_.context(), 0)));
        llvm::Function* curr_fn = ctx_.builder().GetInsertBlock()->getParent();
        llvm::BasicBlock* alloc_ok = llvm::BasicBlock::Create(ctx_.context(), "reshape_ok", curr_fn);
        llvm::BasicBlock* alloc_fail = llvm::BasicBlock::Create(ctx_.context(), "reshape_oom", curr_fn);
        ctx_.builder().CreateCondBr(is_null, alloc_fail, alloc_ok);
        ctx_.builder().SetInsertPoint(alloc_fail);
        llvm::Function* raise_func = ctx_.module().getFunction("eshkol_raise");
        if (!raise_func) {
            llvm::FunctionType* raise_type = llvm::FunctionType::get(ctx_.builder().getVoidTy(), {ctx_.ptrType()}, false);
            raise_func = llvm::Function::Create(raise_type, llvm::Function::ExternalLinkage, "eshkol_raise", &ctx_.module());
            raise_func->setDoesNotReturn();
        }
        llvm::Function* make_exc_func = ctx_.module().getFunction("eshkol_make_exception_with_header");
        if (!make_exc_func) {
            llvm::FunctionType* make_type = llvm::FunctionType::get(ctx_.ptrType(), {ctx_.builder().getInt32Ty(), ctx_.ptrType()}, false);
            make_exc_func = llvm::Function::Create(make_type, llvm::Function::ExternalLinkage, "eshkol_make_exception_with_header", &ctx_.module());
        }
        llvm::Value* err_msg = ctx_.builder().CreateGlobalString("reshape: allocation failed (out of memory)");
        llvm::Value* exception = ctx_.builder().CreateCall(make_exc_func,
            {llvm::ConstantInt::get(ctx_.builder().getInt32Ty(), 1), err_msg});
        ctx_.builder().CreateCall(raise_func, {exception});
        ctx_.builder().CreateUnreachable();
        ctx_.builder().SetInsertPoint(alloc_ok);
    }

    // Store tensor fields: dims_ptr, ndim, elements (reused), total
    llvm::Value* dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 0);
    ctx_.builder().CreateStore(final_dims_ptr, dims_field_ptr);

    llvm::Value* num_dims_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 1);
    ctx_.builder().CreateStore(final_ndim, num_dims_field_ptr);

    llvm::Value* elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 2);
    ctx_.builder().CreateStore(src_elements_ptr, elements_field_ptr);

    llvm::Value* total_elements_field_ptr = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 3);
    ctx_.builder().CreateStore(final_total, total_elements_field_ptr);

    // ESH-0121: propagate the source dtype (field 4). reshape reuses the source
    // elements buffer unchanged, so a DUAL source stays a dual tensor and a plain
    // f64 source stays f64 — the dual-aware matmul/tensor-sum paths dispatch on it.
    llvm::Value* src_dtype_field = ctx_.builder().CreateStructGEP(tensor_type, src_ptr, 4);
    llvm::Value* src_dtype = ctx_.builder().CreateLoad(ctx_.int64Type(), src_dtype_field);
    llvm::Value* out_dtype_field = ctx_.builder().CreateStructGEP(tensor_type, typed_new_tensor_ptr, 4);
    ctx_.builder().CreateStore(src_dtype, out_dtype_field);

    // Pack as consolidated HEAP_PTR
    return tagged_.packHeapPtr(typed_new_tensor_ptr);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
