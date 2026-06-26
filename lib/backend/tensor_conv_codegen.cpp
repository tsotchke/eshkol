/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Convolution & Pooling. Extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split. Implements
 * maxPool2d, avgPool2d, conv1d, conv2d, batchNorm, layerNorm, plus
 * the shared extractAsDouble helper used across the tensor splits.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-conv-extract baseline.
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

// === Convolution & Pooling Operations ===

llvm::Value* TensorCodegen::maxPool2d(const eshkol_operations_t* op) {
    // max-pool2d: (max-pool2d input kernel-size stride)
    // N-dimensional: pools over last 2 dims, preserves all batch dims
    // Input shape: (..., H, W) -> Output shape: (..., out_H, out_W)
    if (op->call_op.num_vars < 3) {
        eshkol_error("max-pool2d requires at least 3 arguments (input, kernel-size, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_arg = codegenAST(&op->call_op.variables[1]);
    if (!kernel_arg) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    // Handle both raw int64 and tagged int64 arguments
    llvm::Value* kernel_size = kernel_arg;
    if (kernel_arg->getType() == ctx_.taggedValueType()) {
        kernel_size = tagged_.unpackInt64(kernel_arg);
    } else if (!kernel_arg->getType()->isIntegerTy(64)) {
        kernel_size = builder.CreateSExtOrTrunc(kernel_arg, ctx_.int64Type());
    }

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Load input tensor properties
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Guard: maxpool2d requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "mp2d_dims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "mp2d_dims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: maxpool2d requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, num_dims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    // Get last 2 dimensions (spatial dims) - works for any number of batch dims
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* h_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, h_idx);
    llvm::Value* w_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), w_ptr);

    // Compute batch_size = total / (H * W)
    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    // Calculate output spatial dimensions: out_dim = (in_dim - kernel_size) / stride + 1
    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "maxpool_result");

    // Allocate dimensions array (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "maxpool_dims");

    // Copy batch dimensions from input, set last 2 to output spatial dims
    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "mp_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    // Store output spatial dimensions
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "maxpool_elems");

    // Populate result tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Create nested loops: batch -> output_h -> output_w -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_body", current_func);
    llvm::BasicBlock* pool_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_cond", current_func);
    llvm::BasicBlock* pool_body = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_body", current_func);
    llvm::BasicBlock* pool_inner_cond = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_cond", current_func);
    llvm::BasicBlock* pool_inner_body = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_body", current_func);
    llvm::BasicBlock* pool_inner_done = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_inner_done", current_func);
    llvm::BasicBlock* pool_done = llvm::BasicBlock::Create(ctx_.context(), "mp_pool_done", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "mp_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "mp_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "mp_batch_done", current_func);

    // Allocate loop variables
    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_oi");
    llvm::Value* oj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_oj");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_ki");
    llvm::Value* kj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mp_kj");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "mp_max");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    // Calculate batch offsets
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    // Outer loop (output row)
    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_h);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oj);
    builder.CreateBr(inner_cond);

    // Inner loop (output column)
    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_oj, out_w);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    builder.CreateStore(llvm::ConstantFP::getInfinity(ctx_.doubleType(), true), max_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(pool_cond);

    // Pool kernel row loop
    builder.SetInsertPoint(pool_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* pool_cmp = builder.CreateICmpSLT(curr_ki, kernel_size);
    builder.CreateCondBr(pool_cmp, pool_body, pool_done);

    builder.SetInsertPoint(pool_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kj);
    builder.CreateBr(pool_inner_cond);

    // Pool kernel column loop
    builder.SetInsertPoint(pool_inner_cond);
    llvm::Value* curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    llvm::Value* pool_inner_cmp = builder.CreateICmpSLT(curr_kj, kernel_size);
    builder.CreateCondBr(pool_inner_cmp, pool_inner_body, pool_inner_done);

    builder.SetInsertPoint(pool_inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(curr_oj, stride), curr_kj);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_elem_bits = builder.CreateLoad(ctx_.int64Type(), in_elem_ptr);
    llvm::Value* in_elem = builder.CreateBitCast(in_elem_bits, ctx_.doubleType());

    llvm::Value* curr_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* new_max = builder.CreateMaxNum(curr_max, in_elem);
    builder.CreateStore(new_max, max_val);

    llvm::Value* next_kj = builder.CreateAdd(curr_kj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_kj, kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_done);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, out_w), curr_oj);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, out_spatial_idx);
    llvm::Value* out_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* max_bits = builder.CreateBitCast(final_max, ctx_.int64Type());
    builder.CreateStore(max_bits, out_elem_ptr);

    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* next_oj = builder.CreateAdd(curr_oj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oj, oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::avgPool2d(const eshkol_operations_t* op) {
    // avg-pool2d: (avg-pool2d input kernel-size stride)
    // N-dimensional: pools over last 2 dims, preserves all batch dims
    // Input shape: (..., H, W) -> Output shape: (..., out_H, out_W)
    if (op->call_op.num_vars < 3) {
        eshkol_error("avg-pool2d requires at least 3 arguments (input, kernel-size, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_arg = codegenAST(&op->call_op.variables[1]);
    if (!kernel_arg) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* kernel_size = kernel_arg;
    if (kernel_arg->getType() == ctx_.taggedValueType()) {
        kernel_size = tagged_.unpackInt64(kernel_arg);
    } else if (!kernel_arg->getType()->isIntegerTy(64)) {
        kernel_size = builder.CreateSExtOrTrunc(kernel_arg, ctx_.int64Type());
    }

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Type* tensor_type = ctx_.tensorType();

    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), ndim_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Guard: avgpool2d requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "ap2d_dims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "ap2d_dims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: avgpool2d requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, num_dims});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    // Get last 2 dimensions (spatial dims)
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* h_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, h_idx);
    llvm::Value* w_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), w_ptr);

    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, kernel_size), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "avgpool_result");

    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "avgpool_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims loop
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "ap_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "avgpool_elems");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    llvm::Value* pool_count = builder.CreateMul(kernel_size, kernel_size);
    llvm::Value* pool_count_fp = builder.CreateSIToFP(pool_count, ctx_.doubleType());

    // Nested loops: batch -> output_h -> output_w -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_body", current_func);
    llvm::BasicBlock* pool_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_cond", current_func);
    llvm::BasicBlock* pool_body = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_body", current_func);
    llvm::BasicBlock* pool_inner_cond = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_cond", current_func);
    llvm::BasicBlock* pool_inner_body = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_body", current_func);
    llvm::BasicBlock* pool_inner_done = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_inner_done", current_func);
    llvm::BasicBlock* pool_done = llvm::BasicBlock::Create(ctx_.context(), "ap_pool_done", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "ap_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "ap_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "ap_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_oi");
    llvm::Value* oj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_oj");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_ki");
    llvm::Value* kj = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ap_kj");
    llvm::Value* sum_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ap_sum");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_h);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_oj, out_w);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_val);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* pool_cmp = builder.CreateICmpSLT(curr_ki, kernel_size);
    builder.CreateCondBr(pool_cmp, pool_body, pool_done);

    builder.SetInsertPoint(pool_body);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_cond);
    llvm::Value* curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    llvm::Value* pool_inner_cmp = builder.CreateICmpSLT(curr_kj, kernel_size);
    builder.CreateCondBr(pool_inner_cmp, pool_inner_body, pool_inner_done);

    builder.SetInsertPoint(pool_inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_kj = builder.CreateLoad(ctx_.int64Type(), kj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(curr_oj, stride), curr_kj);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_elem_bits = builder.CreateLoad(ctx_.int64Type(), in_elem_ptr);
    llvm::Value* in_elem = builder.CreateBitCast(in_elem_bits, ctx_.doubleType());

    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, in_elem);
    builder.CreateStore(new_sum, sum_val);

    llvm::Value* next_kj = builder.CreateAdd(curr_kj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_kj, kj);
    builder.CreateBr(pool_inner_cond);

    builder.SetInsertPoint(pool_inner_done);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(pool_cond);

    builder.SetInsertPoint(pool_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, out_w), curr_oj);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, out_spatial_idx);
    llvm::Value* out_elem_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum_val);
    llvm::Value* avg = builder.CreateFDiv(final_sum, pool_count_fp);
    llvm::Value* avg_bits = builder.CreateBitCast(avg, ctx_.int64Type());
    builder.CreateStore(avg_bits, out_elem_ptr);

    curr_oj = builder.CreateLoad(ctx_.int64Type(), oj);
    llvm::Value* next_oj = builder.CreateAdd(curr_oj, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oj, oj);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::conv1d(const eshkol_operations_t* op) {
    // conv1d: (conv1d input kernel stride)
    // N-dimensional: convolves over last dim, preserves all batch dims
    // Input: (..., L) -> Output: (..., (L-K)/stride + 1)
    if (op->call_op.num_vars < 3) {
        eshkol_error("conv1d requires at least 3 arguments (input, kernel, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!kernel_val) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Get last dimension (spatial dim to convolve)
    llvm::Value* last_dim_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* last_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, last_dim_idx);
    llvm::Value* in_len = builder.CreateLoad(ctx_.int64Type(), last_dim_ptr);

    // Compute batch_size = total / in_len
    llvm::Value* batch_size = builder.CreateSDiv(in_total, in_len);

    // Unpack kernel tensor (1D)
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());
    llvm::Value* k_total_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 3);
    llvm::Value* k_len = builder.CreateLoad(ctx_.int64Type(), k_total_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* kernel_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Calculate output length: (in_len - k_len) / stride + 1
    llvm::Value* out_len = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_len, k_len), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv1d_result");

    // Allocate dimensions (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "conv1d_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, out_len);

    // Copy batch dims, set last to out_len
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "c1_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_len_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, last_dim_idx);
    builder.CreateStore(out_len, out_len_ptr);

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(out_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "conv1d_elems");

    // Populate result tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Create convolution loops: batch -> output position -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_body", current_func);
    llvm::BasicBlock* outer_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_cond", current_func);
    llvm::BasicBlock* outer_body = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_body", current_func);
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_body", current_func);
    llvm::BasicBlock* inner_done = llvm::BasicBlock::Create(ctx_.context(), "c1_inner_done", current_func);
    llvm::BasicBlock* outer_done = llvm::BasicBlock::Create(ctx_.context(), "c1_outer_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "c1_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_bi");
    llvm::Value* oi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_oi");
    llvm::Value* ki = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c1_ki");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "c1_sum");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, in_len);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, out_len);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), oi);
    builder.CreateBr(outer_cond);

    // Output position loop
    builder.SetInsertPoint(outer_cond);
    llvm::Value* curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    llvm::Value* outer_cmp = builder.CreateICmpSLT(curr_oi, out_len);
    builder.CreateCondBr(outer_cmp, outer_body, outer_done);

    builder.SetInsertPoint(outer_body);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ki);
    builder.CreateBr(inner_cond);

    // Kernel loop
    builder.SetInsertPoint(inner_cond);
    llvm::Value* curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    llvm::Value* inner_cmp = builder.CreateICmpSLT(curr_ki, k_len);
    builder.CreateCondBr(inner_cmp, inner_body, inner_done);

    builder.SetInsertPoint(inner_body);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_ki = builder.CreateLoad(ctx_.int64Type(), ki);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, in_len);
    llvm::Value* in_spatial_idx = builder.CreateAdd(builder.CreateMul(curr_oi, stride), curr_ki);
    llvm::Value* in_idx = builder.CreateAdd(in_batch_offset, in_spatial_idx);

    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_idx);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), kernel_elems, curr_ki);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, prod);
    builder.CreateStore(new_sum, sum);

    llvm::Value* next_ki = builder.CreateAdd(curr_ki, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_ki, ki);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_done);
    curr_oi = builder.CreateLoad(ctx_.int64Type(), oi);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, out_len);
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset, curr_oi);
    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_idx);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* sum_bits = builder.CreateBitCast(final_sum, ctx_.int64Type());
    builder.CreateStore(sum_bits, out_ptr);

    llvm::Value* next_oi = builder.CreateAdd(curr_oi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_oi, oi);
    builder.CreateBr(outer_cond);

    builder.SetInsertPoint(outer_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::conv2d(const eshkol_operations_t* op) {
    // conv2d: (conv2d input kernel stride)
    // N-dimensional: convolves over last 2 dims, preserves all batch dims
    // Input: (..., H, W) -> Output: (..., (H-kH)/stride+1, (W-kW)/stride+1)
    if (op->call_op.num_vars < 3) {
        eshkol_error("conv2d requires at least 3 arguments (input, kernel, stride)");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* kernel_val = codegenAST(&op->call_op.variables[1]);
    if (!kernel_val) return nullptr;

    llvm::Value* stride_arg = codegenAST(&op->call_op.variables[2]);
    if (!stride_arg) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* stride = stride_arg;
    if (stride_arg->getType() == ctx_.taggedValueType()) {
        stride = tagged_.unpackInt64(stride_arg);
    } else if (!stride_arg->getType()->isIntegerTy(64)) {
        stride = builder.CreateSExtOrTrunc(stride_arg, ctx_.int64Type());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* in_total = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Get last 2 dimensions (H, W)
    llvm::Value* h_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* w_idx = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* in_h_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, h_idx);
    llvm::Value* in_w_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, w_idx);
    llvm::Value* in_h = builder.CreateLoad(ctx_.int64Type(), in_h_ptr);
    llvm::Value* in_w = builder.CreateLoad(ctx_.int64Type(), in_w_ptr);

    llvm::Value* spatial_in = builder.CreateMul(in_h, in_w);
    llvm::Value* batch_size = builder.CreateSDiv(in_total, spatial_in);

    // Unpack kernel tensor (2D: kH x kW)
    llvm::Value* kernel_ptr_int = tagged_.unpackInt64(kernel_val);
    llvm::Value* kernel_ptr = builder.CreateIntToPtr(kernel_ptr_int, ctx_.ptrType());
    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 0);
    llvm::Value* k_dims = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_ndim_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 1);
    llvm::Value* k_ndim = builder.CreateLoad(ctx_.int64Type(), k_ndim_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 2);
    llvm::Value* kernel_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);
    llvm::Value* k_total_field = builder.CreateStructGEP(tensor_type, kernel_ptr, 3);
    llvm::Value* k_total = builder.CreateLoad(ctx_.int64Type(), k_total_field);

    // Guard: conv2d kernel requires at least 2D tensor
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(k_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "c2d_kdims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "c2d_kdims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: conv2d kernel requires at least 2D tensor (got %lldD)\n");
            builder.CreateCall(pf, {fmt, k_ndim});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    llvm::Value* k_h_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* k_h = builder.CreateLoad(ctx_.int64Type(), k_h_ptr);
    llvm::Value* k_w_ptr = builder.CreateGEP(ctx_.int64Type(), k_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* k_w = builder.CreateLoad(ctx_.int64Type(), k_w_ptr);

    // Calculate output spatial dimensions
    llvm::Value* out_h = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_h, k_h), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* out_w = builder.CreateAdd(
        builder.CreateSDiv(builder.CreateSub(in_w, k_w), stride),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* spatial_out = builder.CreateMul(out_h, out_w);

    // Allocate output tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "conv2d_result");

    // Allocate dimensions (same rank as input)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "conv2d_dims");

    llvm::Value* out_total = builder.CreateMul(batch_size, spatial_out);

    // Copy batch dims
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* copy_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_cond", current_func);
    llvm::BasicBlock* copy_body = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_body", current_func);
    llvm::BasicBlock* copy_done = llvm::BasicBlock::Create(ctx_.context(), "c2_copy_done", current_func);

    llvm::Value* copy_i = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_copy_i");
    llvm::Value* batch_dims = builder.CreateSub(num_dims, llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_cond);
    llvm::Value* curr_copy_i = builder.CreateLoad(ctx_.int64Type(), copy_i);
    llvm::Value* copy_cmp = builder.CreateICmpSLT(curr_copy_i, batch_dims);
    builder.CreateCondBr(copy_cmp, copy_body, copy_done);

    builder.SetInsertPoint(copy_body);
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, curr_copy_i);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, curr_copy_i);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    builder.CreateStore(dim_val, dst_dim_ptr);
    llvm::Value* next_copy_i = builder.CreateAdd(curr_copy_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_copy_i, copy_i);
    builder.CreateBr(copy_cond);

    builder.SetInsertPoint(copy_done);
    llvm::Value* out_h_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, h_idx);
    builder.CreateStore(out_h, out_h_ptr);
    llvm::Value* out_w_ptr = builder.CreateGEP(ctx_.int64Type(), result_dims, w_idx);
    builder.CreateStore(out_w, out_w_ptr);

    // Allocate elements
    llvm::Value* elems_size = builder.CreateMul(out_total, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "conv2d_elems");

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(num_dims, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(out_total, r_total_field);

    // Slot holding the tensor pointer returned to the caller. The AD path uses
    // the pre-allocated `result_ptr`; the numeric path uses the shape-correct
    // tensor allocated by the shared runtime kernel eshkol_rt_conv2d.
    llvm::Value* result_slot = builder.CreateAlloca(ctx_.ptrType(), nullptr, "c2_result_slot");

    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "c2_exit", current_func);

    if (autodiff_) {
        llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
        llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_path", current_func);
        llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), "c2_numeric_path", current_func);
        builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

        builder.SetInsertPoint(ad_path);
        llvm::Value* ad_bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_ad_bi");
        llvm::Value* ad_out_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_ad_out_idx");
        llvm::Value* ad_k_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_ad_k_idx");
        llvm::Value* ad_acc = builder.CreateAlloca(ctx_.ptrType(), nullptr, "c2_ad_acc");

        llvm::BasicBlock* ad_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_batch_cond", current_func);
        llvm::BasicBlock* ad_batch_body = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_batch_body", current_func);
        llvm::BasicBlock* ad_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_loop_cond", current_func);
        llvm::BasicBlock* ad_loop_body = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_loop_body", current_func);
        llvm::BasicBlock* ad_kernel_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_kernel_cond", current_func);
        llvm::BasicBlock* ad_kernel_body = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_kernel_body", current_func);
        llvm::BasicBlock* ad_kernel_done = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_kernel_done", current_func);
        llvm::BasicBlock* ad_loop_done = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_loop_done", current_func);
        llvm::BasicBlock* ad_batch_done = llvm::BasicBlock::Create(ctx_.context(), "c2_ad_batch_done", current_func);

        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_bi);
        builder.CreateBr(ad_batch_cond);

        builder.SetInsertPoint(ad_batch_cond);
        llvm::Value* ad_curr_bi = builder.CreateLoad(ctx_.int64Type(), ad_bi);
        builder.CreateCondBr(builder.CreateICmpSLT(ad_curr_bi, batch_size), ad_batch_body, ad_batch_done);

        builder.SetInsertPoint(ad_batch_body);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_out_idx);
        builder.CreateBr(ad_loop_cond);

        builder.SetInsertPoint(ad_loop_cond);
        llvm::Value* ad_curr_out = builder.CreateLoad(ctx_.int64Type(), ad_out_idx);
        builder.CreateCondBr(builder.CreateICmpSLT(ad_curr_out, spatial_out), ad_loop_body, ad_loop_done);

        builder.SetInsertPoint(ad_loop_body);
        builder.CreateStore(autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0)), ad_acc);
        builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ad_k_idx);
        builder.CreateBr(ad_kernel_cond);

        builder.SetInsertPoint(ad_kernel_cond);
        llvm::Value* ad_curr_k = builder.CreateLoad(ctx_.int64Type(), ad_k_idx);
        builder.CreateCondBr(builder.CreateICmpSLT(ad_curr_k, k_total), ad_kernel_body, ad_kernel_done);

        builder.SetInsertPoint(ad_kernel_body);
        ad_curr_k = builder.CreateLoad(ctx_.int64Type(), ad_k_idx);
        ad_curr_out = builder.CreateLoad(ctx_.int64Type(), ad_out_idx);
        ad_curr_bi = builder.CreateLoad(ctx_.int64Type(), ad_bi);

        llvm::Value* ad_in_batch_offset = builder.CreateMul(ad_curr_bi, spatial_in);
        llvm::Value* ad_out_row = builder.CreateSDiv(ad_curr_out, out_w);
        llvm::Value* ad_out_col = builder.CreateSRem(ad_curr_out, out_w);
        llvm::Value* ad_k_row = builder.CreateSDiv(ad_curr_k, k_w);
        llvm::Value* ad_k_col = builder.CreateSRem(ad_curr_k, k_w);
        llvm::Value* ad_in_row = builder.CreateAdd(builder.CreateMul(ad_out_row, stride), ad_k_row);
        llvm::Value* ad_in_col = builder.CreateAdd(builder.CreateMul(ad_out_col, stride), ad_k_col);
        llvm::Value* ad_in_spatial = builder.CreateAdd(builder.CreateMul(ad_in_row, in_w), ad_in_col);
        llvm::Value* ad_in_linear = builder.CreateAdd(ad_in_batch_offset, ad_in_spatial);

        llvm::Value* ad_in_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), input_elems, ad_in_linear));
        llvm::Value* ad_k_bits = builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), kernel_elems, ad_curr_k));
        llvm::Value* ad_in_node = adNodeFromTensorElementBits(ad_in_bits, "c2_ad_input");
        llvm::Value* ad_kernel_node = adNodeFromTensorElementBits(ad_k_bits, "c2_ad_kernel");
        llvm::Value* ad_product = autodiff_->recordADNodeBinary(4, ad_in_node, ad_kernel_node);
        llvm::Value* ad_old_acc = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        llvm::Value* ad_new_acc = autodiff_->recordADNodeBinary(2, ad_old_acc, ad_product);
        builder.CreateStore(ad_new_acc, ad_acc);

        builder.CreateStore(builder.CreateAdd(ad_curr_k, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_k_idx);
        builder.CreateBr(ad_kernel_cond);

        builder.SetInsertPoint(ad_kernel_done);
        ad_curr_out = builder.CreateLoad(ctx_.int64Type(), ad_out_idx);
        ad_curr_bi = builder.CreateLoad(ctx_.int64Type(), ad_bi);
        llvm::Value* ad_out_batch_offset = builder.CreateMul(ad_curr_bi, spatial_out);
        llvm::Value* ad_out_linear = builder.CreateAdd(ad_out_batch_offset, ad_curr_out);
        llvm::Value* ad_final_node = builder.CreateLoad(ctx_.ptrType(), ad_acc);
        builder.CreateStore(builder.CreatePtrToInt(ad_final_node, ctx_.int64Type()),
            builder.CreateGEP(ctx_.int64Type(), result_elems, ad_out_linear));
        builder.CreateStore(builder.CreateAdd(ad_curr_out, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_out_idx);
        builder.CreateBr(ad_loop_cond);

        builder.SetInsertPoint(ad_loop_done);
        ad_curr_bi = builder.CreateLoad(ctx_.int64Type(), ad_bi);
        builder.CreateStore(builder.CreateAdd(ad_curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ad_bi);
        builder.CreateBr(ad_batch_cond);

        builder.SetInsertPoint(ad_batch_done);
        builder.CreateStore(result_ptr, result_slot);
        builder.CreateBr(exit_block);

        builder.SetInsertPoint(numeric_path);
    }

    // === Numeric forward path (ESH-0068) ===
    // Delegate to the shared C kernel eshkol_rt_conv2d (tensor_conv_kernel.cpp /
    // tensor_conv_kernel.h), the SAME kernel the embedded VM calls, so -r / AOT
    // / VM conv2d can never diverge again. This replaces the former inline IR
    // loops, which read the kernel's FIRST two dims (k_dims[0], k_dims[1]) as
    // the spatial extent — correct for a bare 2-D kernel but wrong for a rank-4
    // NCHW kernel [out_c, in_c, kH, kW] — and which never summed over
    // in_channels nor produced out_channels. The shared kernel implements the
    // canonical NCHW cross-correlation with stride + zero-padding.
    //
    // Optional 4th argument: symmetric zero-padding (default 0).
    llvm::Value* pad = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    if (op->call_op.num_vars >= 4) {
        llvm::Value* pad_arg = codegenAST(&op->call_op.variables[3]);
        if (pad_arg) {
            if (pad_arg->getType() == ctx_.taggedValueType()) {
                pad = tagged_.unpackInt64(pad_arg);
            } else if (pad_arg->getType()->isIntegerTy(64)) {
                pad = pad_arg;
            } else {
                pad = builder.CreateSExtOrTrunc(pad_arg, ctx_.int64Type());
            }
        }
    }

    llvm::FunctionType* conv_fn_ty = llvm::FunctionType::get(
        ctx_.ptrType(),
        {ctx_.ptrType(), ctx_.ptrType(), ctx_.ptrType(),
         ctx_.int64Type(), ctx_.int64Type()},
        false);
    llvm::FunctionCallee conv_fn =
        ctx_.module().getOrInsertFunction("eshkol_rt_conv2d", conv_fn_ty);
    llvm::Value* numeric_result = builder.CreateCall(
        conv_fn, {arena_ptr, input_ptr, kernel_ptr, stride, pad}, "conv2d_rt");
    builder.CreateStore(numeric_result, result_slot);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(exit_block);
    llvm::Value* final_result = builder.CreateLoad(ctx_.ptrType(), result_slot);
    return tagged_.packHeapPtr(final_result);
}

bool TensorCodegen::emitTensorADNormalizeDispatch(llvm::Value* src_elems,
                                                  llvm::Value* result_elems,
                                                  llvm::Value* total_elements,
                                                  llvm::Value* axis_len,
                                                  llvm::Value* inner_stride,
                                                  llvm::Value* gamma,
                                                  llvm::Value* gamma_source,
                                                  llvm::Value* beta,
                                                  llvm::Value* beta_source,
                                                  llvm::Value* epsilon,
                                                  llvm::Value* epsilon_source,
                                                  llvm::BasicBlock* exit_block,
                                                  const std::string& name) {
    if (!autodiff_ || !src_elems || !result_elems || !total_elements ||
        !axis_len || !inner_stride || !gamma || !beta || !epsilon || !exit_block) {
        return false;
    }

    auto& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Value* in_ad_mode = builder.CreateLoad(ctx_.int1Type(), ctx_.adModeActive());
    llvm::BasicBlock* ad_path = llvm::BasicBlock::Create(ctx_.context(), name + "_ad_path", current_func);
    llvm::BasicBlock* numeric_path = llvm::BasicBlock::Create(ctx_.context(), name + "_numeric_path", current_func);
    builder.CreateCondBr(in_ad_mode, ad_path, numeric_path);

    builder.SetInsertPoint(ad_path);
    llvm::Value* zero_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    llvm::Value* one_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 1);
    llvm::Value* group_count = builder.CreateUDiv(total_elements, axis_len);
    llvm::Value* count_fp = builder.CreateUIToFP(axis_len, ctx_.doubleType());
    llvm::Value* zero_node = autodiff_->createADConstant(llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    llvm::Value* count_node = autodiff_->createADConstant(count_fp);
    auto scalarToADNode = [&](llvm::Value* source, llvm::Value* numeric,
                              const std::string& node_name) -> llvm::Value* {
        if (!source || source->getType() != ctx_.taggedValueType()) {
            return autodiff_->createADConstant(numeric);
        }

        llvm::BasicBlock* callable_block = llvm::BasicBlock::Create(
            ctx_.context(), node_name + "_callable", current_func);
        llvm::BasicBlock* ad_block = llvm::BasicBlock::Create(
            ctx_.context(), node_name + "_ad", current_func);
        llvm::BasicBlock* const_block = llvm::BasicBlock::Create(
            ctx_.context(), node_name + "_const", current_func);
        llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(
            ctx_.context(), node_name + "_merge", current_func);

        llvm::Value* type_tag = tagged_.getType(source);
        llvm::Value* base_type = tagged_.getBaseType(type_tag);
        llvm::Value* is_callable = builder.CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_CALLABLE));
        builder.CreateCondBr(is_callable, callable_block, const_block);

        builder.SetInsertPoint(callable_block);
        llvm::Value* is_ad = tagged_.checkCallableSubtype(source, CALLABLE_SUBTYPE_AD_NODE);
        builder.CreateCondBr(is_ad, ad_block, const_block);

        builder.SetInsertPoint(ad_block);
        llvm::Value* ad_node = tagged_.unpackPtr(source);
        builder.CreateBr(merge_block);
        llvm::BasicBlock* ad_exit = builder.GetInsertBlock();

        builder.SetInsertPoint(const_block);
        llvm::Value* const_node = autodiff_->createADConstant(numeric);
        builder.CreateBr(merge_block);
        llvm::BasicBlock* const_exit = builder.GetInsertBlock();

        builder.SetInsertPoint(merge_block);
        llvm::PHINode* phi = builder.CreatePHI(ctx_.ptrType(), 2, node_name + "_node");
        phi->addIncoming(ad_node, ad_exit);
        phi->addIncoming(const_node, const_exit);
        return phi;
    };
    llvm::Value* gamma_node = scalarToADNode(gamma_source, gamma, name + "_gamma");
    llvm::Value* beta_node = scalarToADNode(beta_source, beta, name + "_beta");
    llvm::Value* epsilon_node = scalarToADNode(epsilon_source, epsilon, name + "_epsilon");

    llvm::Value* group_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, (name + "_group_i").c_str());
    builder.CreateStore(zero_i64, group_i_alloca);
    llvm::BasicBlock* group_cond = llvm::BasicBlock::Create(ctx_.context(), name + "_group_cond", current_func);
    llvm::BasicBlock* group_body = llvm::BasicBlock::Create(ctx_.context(), name + "_group_body", current_func);
    llvm::BasicBlock* group_done = llvm::BasicBlock::Create(ctx_.context(), name + "_group_done", current_func);
    builder.CreateBr(group_cond);

    builder.SetInsertPoint(group_cond);
    llvm::Value* group_i = builder.CreateLoad(ctx_.int64Type(), group_i_alloca);
    builder.CreateCondBr(builder.CreateICmpULT(group_i, group_count), group_body, group_done);

    builder.SetInsertPoint(group_body);
    llvm::Value* inner = builder.CreateURem(group_i, inner_stride);
    llvm::Value* outer = builder.CreateUDiv(group_i, inner_stride);
    llvm::Value* outer_stride = builder.CreateMul(axis_len, inner_stride);
    llvm::Value* base = builder.CreateAdd(builder.CreateMul(outer, outer_stride), inner);

    llvm::Value* sum_node_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, (name + "_sum").c_str());
    llvm::Value* k_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, (name + "_k").c_str());
    builder.CreateStore(zero_node, sum_node_alloca);
    builder.CreateStore(zero_i64, k_alloca);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), name + "_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), name + "_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), name + "_mean_done", current_func);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* mean_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
    builder.CreateCondBr(builder.CreateICmpULT(mean_k, axis_len), mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* mean_index = builder.CreateAdd(base, builder.CreateMul(mean_k, inner_stride));
    llvm::Value* mean_bits = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), src_elems, mean_index));
    llvm::Value* mean_elem_node = adNodeFromTensorElementBits(mean_bits, name + "_mean_elem");
    llvm::Value* old_sum = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
    llvm::Value* new_sum = autodiff_->recordADNodeBinary(2, old_sum, mean_elem_node);
    builder.CreateStore(new_sum, sum_node_alloca);
    builder.CreateStore(builder.CreateAdd(mean_k, one_i64), k_alloca);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.ptrType(), sum_node_alloca);
    llvm::Value* mean_node = autodiff_->recordADNodeBinary(5, final_sum, count_node);

    llvm::Value* var_sum_alloca = builder.CreateAlloca(ctx_.ptrType(), nullptr, (name + "_var_sum").c_str());
    builder.CreateStore(zero_node, var_sum_alloca);
    builder.CreateStore(zero_i64, k_alloca);
    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), name + "_var_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), name + "_var_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), name + "_var_done", current_func);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    llvm::Value* var_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
    builder.CreateCondBr(builder.CreateICmpULT(var_k, axis_len), var_body, var_done);

    builder.SetInsertPoint(var_body);
    llvm::Value* var_index = builder.CreateAdd(base, builder.CreateMul(var_k, inner_stride));
    llvm::Value* var_bits = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), src_elems, var_index));
    llvm::Value* var_elem_node = adNodeFromTensorElementBits(var_bits, name + "_var_elem");
    llvm::Value* centered_node = autodiff_->recordADNodeBinary(3, var_elem_node, mean_node);
    llvm::Value* sq_node = autodiff_->recordADNodeBinary(4, centered_node, centered_node);
    llvm::Value* old_var_sum = builder.CreateLoad(ctx_.ptrType(), var_sum_alloca);
    llvm::Value* new_var_sum = autodiff_->recordADNodeBinary(2, old_var_sum, sq_node);
    builder.CreateStore(new_var_sum, var_sum_alloca);
    builder.CreateStore(builder.CreateAdd(var_k, one_i64), k_alloca);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    llvm::Value* final_var_sum = builder.CreateLoad(ctx_.ptrType(), var_sum_alloca);
    llvm::Value* var_node = autodiff_->recordADNodeBinary(5, final_var_sum, count_node);
    llvm::Value* var_eps_node = autodiff_->recordADNodeBinary(2, var_node, epsilon_node);
    llvm::Value* std_node = autodiff_->recordADNodeUnary(41, var_eps_node);

    builder.CreateStore(zero_i64, k_alloca);
    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), name + "_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), name + "_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), name + "_norm_done", current_func);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    llvm::Value* norm_k = builder.CreateLoad(ctx_.int64Type(), k_alloca);
    builder.CreateCondBr(builder.CreateICmpULT(norm_k, axis_len), norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    llvm::Value* norm_index = builder.CreateAdd(base, builder.CreateMul(norm_k, inner_stride));
    llvm::Value* norm_bits = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), src_elems, norm_index));
    llvm::Value* norm_elem_node = adNodeFromTensorElementBits(norm_bits, name + "_norm_elem");
    llvm::Value* norm_centered = autodiff_->recordADNodeBinary(3, norm_elem_node, mean_node);
    llvm::Value* normalized = autodiff_->recordADNodeBinary(5, norm_centered, std_node);
    llvm::Value* scaled = autodiff_->recordADNodeBinary(4, normalized, gamma_node);
    llvm::Value* shifted = autodiff_->recordADNodeBinary(2, scaled, beta_node);
    builder.CreateStore(builder.CreatePtrToInt(shifted, ctx_.int64Type()),
        builder.CreateGEP(ctx_.int64Type(), result_elems, norm_index));
    builder.CreateStore(builder.CreateAdd(norm_k, one_i64), k_alloca);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_done);
    builder.CreateStore(builder.CreateAdd(group_i, one_i64), group_i_alloca);
    builder.CreateBr(group_cond);

    builder.SetInsertPoint(group_done);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(numeric_path);
    return true;
}

llvm::Value* TensorCodegen::batchNorm(const eshkol_operations_t* op) {
    // batch-norm: (batch-norm input gamma beta epsilon [axis])
    // Simplified batch normalization for inference
    // y = gamma * (x - mean) / sqrt(var + eps) + beta
    // axis defaults to 0 (batch dimension); optional 5th arg overrides
    if (op->call_op.num_vars < 4 || op->call_op.num_vars > 5) {
        eshkol_error("batch-norm requires 4-5 arguments (input, gamma, beta, epsilon, [axis])");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;

    llvm::Value* gamma_val = codegenAST(&op->call_op.variables[1]);
    if (!gamma_val) return nullptr;

    llvm::Value* beta_val = codegenAST(&op->call_op.variables[2]);
    if (!beta_val) return nullptr;

    llvm::Value* eps_arg = codegenAST(&op->call_op.variables[3]);
    if (!eps_arg) return nullptr;

    // 5-arg case: (batch-norm input gamma beta epsilon axis) → axis-aware via runtime
    if (op->call_op.num_vars == 5) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[4]);
        if (!axis_val) return nullptr;

        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(input_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);

        llvm::Value* gamma_d = gamma_val;
        if (gamma_val->getType() == ctx_.taggedValueType()) gamma_d = tagged_.unpackDouble(gamma_val);
        llvm::Value* beta_d = beta_val;
        if (beta_val->getType() == ctx_.taggedValueType()) beta_d = tagged_.unpackDouble(beta_val);
        llvm::Value* eps_d = eps_arg;
        if (eps_arg->getType() == ctx_.taggedValueType()) eps_d = tagged_.unpackDouble(eps_arg);
        else if (eps_arg->getType()->isIntegerTy(64)) eps_d = builder.CreateSIToFP(eps_arg, ctx_.doubleType());

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        auto* dblTy = ctx_.doubleType();

        if (autodiff_) {
            llvm::Function* current_func = builder.GetInsertBlock()->getParent();
            llvm::BasicBlock* ad_done = llvm::BasicBlock::Create(ctx_.context(), "bn_axis_ad_done", current_func);
            llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "bn_axis_merge", current_func);

            llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
            llvm::Function* arena_alloc = mem_.getArenaAllocate();
            llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena}, "bn_axis_ad_result");
            llvm::Value* dims_size = builder.CreateMul(rank,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
            llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena, dims_size}, "bn_axis_ad_dims");
            builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims, llvm::MaybeAlign(8), dims_size);
            llvm::Value* elems_size = builder.CreateMul(total,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
            llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena, elems_size}, "bn_axis_ad_elems");

            llvm::Value* is_negative_axis = builder.CreateICmpSLT(axis,
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* normalized_axis = builder.CreateSelect(is_negative_axis,
                builder.CreateAdd(axis, rank), axis);
            llvm::Value* axis_len = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, normalized_axis));

            llvm::Value* one_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 1);
            llvm::Value* inner_stride_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bn_axis_inner_stride");
            llvm::Value* stride_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bn_axis_stride_i");
            builder.CreateStore(one_i64, inner_stride_alloca);
            builder.CreateStore(builder.CreateAdd(normalized_axis, one_i64), stride_i_alloca);
            llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_axis_stride_cond", current_func);
            llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "bn_axis_stride_body", current_func);
            llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "bn_axis_stride_done", current_func);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_cond);
            llvm::Value* stride_i = builder.CreateLoad(ctx_.int64Type(), stride_i_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(stride_i, rank), stride_body, stride_done);

            builder.SetInsertPoint(stride_body);
            llvm::Value* stride_dim = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, stride_i));
            llvm::Value* old_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride_alloca);
            builder.CreateStore(builder.CreateMul(old_stride, stride_dim), inner_stride_alloca);
            builder.CreateStore(builder.CreateAdd(stride_i, one_i64), stride_i_alloca);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_done);
            llvm::Value* inner_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride_alloca);
            builder.CreateStore(result_dims, builder.CreateStructGEP(ttype, result_ptr, 0));
            builder.CreateStore(rank, builder.CreateStructGEP(ttype, result_ptr, 1));
            builder.CreateStore(result_elems, builder.CreateStructGEP(ttype, result_ptr, 2));
            builder.CreateStore(total, builder.CreateStructGEP(ttype, result_ptr, 3));

            emitTensorADNormalizeDispatch(elems, result_elems, total, axis_len,
                inner_stride, gamma_d, gamma_val, beta_d, beta_val, eps_d, eps_arg,
                ad_done, "bn_axis");

            llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
                {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
            llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
            llvm::Value* numeric_result = builder.CreateCall(callee,
                {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "bn_axis_numeric_result");
            llvm::Value* numeric_packed = tagged_.packHeapPtr(numeric_result);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* numeric_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(ad_done);
            llvm::Value* ad_packed = tagged_.packHeapPtr(result_ptr);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* ad_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(merge_block);
            llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), 2, "bn_axis_result_phi");
            result_phi->addIncoming(ad_packed, ad_exit);
            result_phi->addIncoming(numeric_packed, numeric_exit);
            return result_phi;
        }

        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "bn_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    llvm::Value* epsilon = eps_arg;
    if (eps_arg->getType() == ctx_.taggedValueType()) {
        epsilon = tagged_.unpackDouble(eps_arg);
    } else if (eps_arg->getType()->isIntegerTy(64)) {
        epsilon = builder.CreateSIToFP(eps_arg, ctx_.doubleType());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* in_ndim = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Extract gamma scalar
    llvm::Value* gamma = gamma_val;
    if (gamma_val->getType() == ctx_.taggedValueType()) {
        gamma = tagged_.unpackDouble(gamma_val);
    }

    // Extract beta scalar
    llvm::Value* beta = beta_val;
    if (beta_val->getType() == ctx_.taggedValueType()) {
        beta = tagged_.unpackDouble(beta_val);
    }

    // Allocate output tensor (same shape as input)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "bn_result");

    // Copy dimensions
    llvm::Value* dims_size = builder.CreateMul(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "bn_dims");

    // Copy dimension values
    llvm::Function* memcpy_func = ctx_.module().getFunction("memcpy");
    if (!memcpy_func) {
        llvm::FunctionType* memcpy_type = llvm::FunctionType::get(ctx_.ptrType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        memcpy_func = llvm::Function::Create(memcpy_type, llvm::Function::ExternalLinkage, "memcpy", &ctx_.module());
    }
    builder.CreateCall(memcpy_func, {result_dims, in_dims, dims_size});

    // Allocate output elements
    llvm::Value* elems_size = builder.CreateMul(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "bn_elems");

    // Populate result tensor
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(in_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "bn_exit", current_func);
    emitTensorADNormalizeDispatch(input_elems, result_elems, total_elements,
        total_elements, llvm::ConstantInt::get(ctx_.int64Type(), 1),
        gamma, gamma_val, beta, beta_val, epsilon, eps_arg, exit_block, "bn");

    // First pass: compute mean
    llvm::Value* mean = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_mean");
    llvm::Value* var = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_var");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "bn_sum");
    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "bn_idx");

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "bn_mean_done", current_func);

    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* mean_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(mean_cmp, mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, elem);
    builder.CreateStore(new_sum, sum);
    llvm::Value* next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* total_fp = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mean_val = builder.CreateFDiv(final_sum, total_fp);
    builder.CreateStore(mean_val, mean);

    // Second pass: compute variance
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_var_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), "bn_var_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), "bn_var_done", current_func);

    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* var_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(var_cmp, var_body, var_done);

    builder.SetInsertPoint(var_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), mean);
    llvm::Value* diff = builder.CreateFSub(elem, mean_val);
    llvm::Value* sq_diff = builder.CreateFMul(diff, diff);
    curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    new_sum = builder.CreateFAdd(curr_sum, sq_diff);
    builder.CreateStore(new_sum, sum);
    next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* var_val = builder.CreateFDiv(final_sum, total_fp);
    builder.CreateStore(var_val, var);

    // Third pass: normalize and scale
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);

    // Compute std = sqrt(var + eps)
    llvm::Value* var_plus_eps = builder.CreateFAdd(var_val, epsilon);
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* std_val = builder.CreateCall(sqrt_func, {var_plus_eps});

    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "bn_norm_done", current_func);

    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    llvm::Value* norm_cmp = builder.CreateICmpSLT(curr_idx, total_elements);
    builder.CreateCondBr(norm_cmp, norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    curr_idx = builder.CreateLoad(ctx_.int64Type(), idx);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, curr_idx);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), mean);

    // y = gamma * (x - mean) / std + beta
    llvm::Value* centered = builder.CreateFSub(elem, mean_val);
    llvm::Value* normalized = builder.CreateFDiv(centered, std_val);
    llvm::Value* scaled = builder.CreateFMul(normalized, gamma);
    llvm::Value* shifted = builder.CreateFAdd(scaled, beta);

    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, curr_idx);
    llvm::Value* out_bits = builder.CreateBitCast(shifted, ctx_.int64Type());
    builder.CreateStore(out_bits, out_ptr);

    next_idx = builder.CreateAdd(curr_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_idx, idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_done);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(exit_block);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::layerNorm(const eshkol_operations_t* op) {
    // layer-norm: (layer-norm input gamma beta epsilon [axis])
    // Normalizes across the LAST dimension (features) for each sample independently.
    // This is fundamentally different from batch-norm which normalizes across the batch dimension.
    // For 1D: normalizes all elements (single sample)
    // For 2D (batch×features): normalizes each row independently
    // For ND: last dim = features, everything else = batch
    // Optional 5th arg overrides axis (default: -1 = last dimension)
    if (op->call_op.num_vars < 4 || op->call_op.num_vars > 5) {
        eshkol_error("layer-norm requires 4-5 arguments (input, gamma, beta, epsilon, [axis])");
        return nullptr;
    }

    llvm::Value* input_val = codegenAST(&op->call_op.variables[0]);
    if (!input_val) return nullptr;
    llvm::Value* gamma_val = codegenAST(&op->call_op.variables[1]);
    if (!gamma_val) return nullptr;
    llvm::Value* beta_val = codegenAST(&op->call_op.variables[2]);
    if (!beta_val) return nullptr;
    llvm::Value* eps_arg = codegenAST(&op->call_op.variables[3]);
    if (!eps_arg) return nullptr;

    // 5-arg case: (layer-norm input gamma beta epsilon axis) → axis-aware via runtime
    if (op->call_op.num_vars == 5) {
        llvm::Value* axis_val = codegenAST(&op->call_op.variables[4]);
        if (!axis_val) return nullptr;

        auto& builder = ctx_.builder();
        llvm::Value* ptr_int = tagged_.safeExtractInt64(input_val);
        llvm::Value* ptr = builder.CreateIntToPtr(ptr_int, ctx_.ptrType());
        llvm::StructType* ttype = ctx_.tensorType();
        llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 2));
        llvm::Value* total = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 3));
        llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), builder.CreateStructGEP(ttype, ptr, 0));
        llvm::Value* rank = builder.CreateLoad(ctx_.int64Type(), builder.CreateStructGEP(ttype, ptr, 1));
        llvm::Value* axis = tagged_.safeExtractInt64(axis_val);

        llvm::Value* gamma_d = gamma_val;
        if (gamma_val->getType() == ctx_.taggedValueType()) gamma_d = tagged_.unpackDouble(gamma_val);
        llvm::Value* beta_d = beta_val;
        if (beta_val->getType() == ctx_.taggedValueType()) beta_d = tagged_.unpackDouble(beta_val);
        llvm::Value* eps_d = eps_arg;
        if (eps_arg->getType() == ctx_.taggedValueType()) eps_d = tagged_.unpackDouble(eps_arg);
        else if (eps_arg->getType()->isIntegerTy(64)) eps_d = builder.CreateSIToFP(eps_arg, ctx_.doubleType());

        llvm::Value* arena = builder.CreateLoad(ctx_.ptrType(), ctx_.globalArena());
        auto* ptrTy = ctx_.ptrType();
        auto* i64Ty = ctx_.int64Type();
        auto* dblTy = ctx_.doubleType();

        if (autodiff_) {
            llvm::Function* current_func = builder.GetInsertBlock()->getParent();
            llvm::BasicBlock* ad_done = llvm::BasicBlock::Create(ctx_.context(), "ln_axis_ad_done", current_func);
            llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "ln_axis_merge", current_func);

            llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
            llvm::Function* arena_alloc = mem_.getArenaAllocate();
            llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena}, "ln_axis_ad_result");
            llvm::Value* dims_size = builder.CreateMul(rank,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
            llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena, dims_size}, "ln_axis_ad_dims");
            builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), dims, llvm::MaybeAlign(8), dims_size);
            llvm::Value* elems_size = builder.CreateMul(total,
                llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
            llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena, elems_size}, "ln_axis_ad_elems");

            llvm::Value* is_negative_axis = builder.CreateICmpSLT(axis,
                llvm::ConstantInt::get(ctx_.int64Type(), 0));
            llvm::Value* normalized_axis = builder.CreateSelect(is_negative_axis,
                builder.CreateAdd(axis, rank), axis);
            llvm::Value* axis_len = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, normalized_axis));

            llvm::Value* one_i64 = llvm::ConstantInt::get(ctx_.int64Type(), 1);
            llvm::Value* inner_stride_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_axis_inner_stride");
            llvm::Value* stride_i_alloca = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_axis_stride_i");
            builder.CreateStore(one_i64, inner_stride_alloca);
            builder.CreateStore(builder.CreateAdd(normalized_axis, one_i64), stride_i_alloca);
            llvm::BasicBlock* stride_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_axis_stride_cond", current_func);
            llvm::BasicBlock* stride_body = llvm::BasicBlock::Create(ctx_.context(), "ln_axis_stride_body", current_func);
            llvm::BasicBlock* stride_done = llvm::BasicBlock::Create(ctx_.context(), "ln_axis_stride_done", current_func);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_cond);
            llvm::Value* stride_i = builder.CreateLoad(ctx_.int64Type(), stride_i_alloca);
            builder.CreateCondBr(builder.CreateICmpULT(stride_i, rank), stride_body, stride_done);

            builder.SetInsertPoint(stride_body);
            llvm::Value* stride_dim = builder.CreateLoad(ctx_.int64Type(),
                builder.CreateGEP(ctx_.int64Type(), dims, stride_i));
            llvm::Value* old_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride_alloca);
            builder.CreateStore(builder.CreateMul(old_stride, stride_dim), inner_stride_alloca);
            builder.CreateStore(builder.CreateAdd(stride_i, one_i64), stride_i_alloca);
            builder.CreateBr(stride_cond);

            builder.SetInsertPoint(stride_done);
            llvm::Value* inner_stride = builder.CreateLoad(ctx_.int64Type(), inner_stride_alloca);
            builder.CreateStore(result_dims, builder.CreateStructGEP(ttype, result_ptr, 0));
            builder.CreateStore(rank, builder.CreateStructGEP(ttype, result_ptr, 1));
            builder.CreateStore(result_elems, builder.CreateStructGEP(ttype, result_ptr, 2));
            builder.CreateStore(total, builder.CreateStructGEP(ttype, result_ptr, 3));

            emitTensorADNormalizeDispatch(elems, result_elems, total, axis_len,
                inner_stride, gamma_d, gamma_val, beta_d, beta_val, eps_d, eps_arg,
                ad_done, "ln_axis");

            llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
                {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
            llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
            llvm::Value* numeric_result = builder.CreateCall(callee,
                {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "ln_axis_numeric_result");
            llvm::Value* numeric_packed = tagged_.packHeapPtr(numeric_result);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* numeric_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(ad_done);
            llvm::Value* ad_packed = tagged_.packHeapPtr(result_ptr);
            builder.CreateBr(merge_block);
            llvm::BasicBlock* ad_exit = builder.GetInsertBlock();

            builder.SetInsertPoint(merge_block);
            llvm::PHINode* result_phi = builder.CreatePHI(ctx_.taggedValueType(), 2, "ln_axis_result_phi");
            result_phi->addIncoming(ad_packed, ad_exit);
            result_phi->addIncoming(numeric_packed, numeric_exit);
            return result_phi;
        }

        llvm::FunctionType* fn_type = llvm::FunctionType::get(ptrTy,
            {ptrTy, ptrTy, i64Ty, ptrTy, i64Ty, i64Ty, dblTy, dblTy, dblTy}, false);
        llvm::FunctionCallee callee = ctx_.module().getOrInsertFunction("eshkol_xla_normalize", fn_type);
        llvm::Value* result = builder.CreateCall(callee,
            {arena, elems, total, dims, rank, axis, gamma_d, beta_d, eps_d}, "ln_axis_result");
        return tagged_.packHeapPtr(result);
    }

    auto& builder = ctx_.builder();

    llvm::Value* epsilon = eps_arg;
    if (eps_arg->getType() == ctx_.taggedValueType()) {
        epsilon = tagged_.unpackDouble(eps_arg);
    } else if (eps_arg->getType()->isIntegerTy(64)) {
        epsilon = builder.CreateSIToFP(eps_arg, ctx_.doubleType());
    }

    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());
    llvm::Type* tensor_type = ctx_.tensorType();

    // Unpack input tensor
    llvm::Value* input_ptr_int = tagged_.unpackInt64(input_val);
    llvm::Value* input_ptr = builder.CreateIntToPtr(input_ptr_int, ctx_.ptrType());
    llvm::Value* in_dims_field = builder.CreateStructGEP(tensor_type, input_ptr, 0);
    llvm::Value* in_dims = builder.CreateLoad(ctx_.ptrType(), in_dims_field);
    llvm::Value* in_ndim_field = builder.CreateStructGEP(tensor_type, input_ptr, 1);
    llvm::Value* in_ndim = builder.CreateLoad(ctx_.int64Type(), in_ndim_field);
    llvm::Value* in_elems_field = builder.CreateStructGEP(tensor_type, input_ptr, 2);
    llvm::Value* input_elems = builder.CreateLoad(ctx_.ptrType(), in_elems_field);
    llvm::Value* in_total_field = builder.CreateStructGEP(tensor_type, input_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), in_total_field);

    // Extract gamma/beta scalars
    llvm::Value* gamma = gamma_val;
    if (gamma_val->getType() == ctx_.taggedValueType()) {
        gamma = tagged_.unpackDouble(gamma_val);
    }
    llvm::Value* beta = beta_val;
    if (beta_val->getType() == ctx_.taggedValueType()) {
        beta = tagged_.unpackDouble(beta_val);
    }

    // Allocate output tensor (same shape as input)
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "ln_result");

    llvm::Value* dims_size = builder.CreateMul(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "ln_dims");

    llvm::Function* memcpy_func = ctx_.module().getFunction("memcpy");
    if (!memcpy_func) {
        llvm::FunctionType* memcpy_type = llvm::FunctionType::get(ctx_.ptrType(),
            {ctx_.ptrType(), ctx_.ptrType(), ctx_.int64Type()}, false);
        memcpy_func = llvm::Function::Create(memcpy_type, llvm::Function::ExternalLinkage, "memcpy", &ctx_.module());
    }
    builder.CreateCall(memcpy_func, {result_dims, in_dims, dims_size});

    llvm::Value* elems_size = builder.CreateMul(total_elements, llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_size}, "ln_elems");

    // Populate result tensor metadata
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(in_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(result_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_elements, r_total_field);

    // Compute feature_size = dims[ndim-1] (last dimension)
    llvm::Value* last_dim_idx = builder.CreateSub(in_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* last_dim_ptr = builder.CreateGEP(ctx_.int64Type(), in_dims, last_dim_idx);
    llvm::Value* feature_size = builder.CreateLoad(ctx_.int64Type(), last_dim_ptr);

    // batch_size = total_elements / feature_size
    llvm::Value* batch_size = builder.CreateUDiv(total_elements, feature_size);
    llvm::Value* feature_fp = builder.CreateSIToFP(feature_size, ctx_.doubleType());

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "ln_exit", current_func);
    emitTensorADNormalizeDispatch(input_elems, result_elems, total_elements,
        feature_size, llvm::ConstantInt::get(ctx_.int64Type(), 1),
        gamma, gamma_val, beta, beta_val, epsilon, eps_arg, exit_block, "ln");

    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});

    // Allocas for loop variables
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_batch_idx");
    llvm::Value* feat_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "ln_feat_idx");
    llvm::Value* ln_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ln_sum");
    llvm::Value* ln_mean = builder.CreateAlloca(ctx_.doubleType(), nullptr, "ln_mean");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);

    // === Outer loop: iterate over samples ===
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_body", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "ln_batch_done", current_func);

    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* bi = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    builder.CreateCondBr(builder.CreateICmpULT(bi, batch_size), batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* base_offset = builder.CreateMul(bi, feature_size);

    // --- Pass 1: Compute mean for this sample ---
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ln_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* mean_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_cond", current_func);
    llvm::BasicBlock* mean_body = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_body", current_func);
    llvm::BasicBlock* mean_done = llvm::BasicBlock::Create(ctx_.context(), "ln_mean_done", current_func);

    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), mean_body, mean_done);

    builder.SetInsertPoint(mean_body);
    llvm::Value* elem_offset = builder.CreateAdd(base_offset, fi);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, elem), ln_sum);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(mean_cond);

    builder.SetInsertPoint(mean_done);
    llvm::Value* sum_val = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    llvm::Value* mean_val = builder.CreateFDiv(sum_val, feature_fp);
    builder.CreateStore(mean_val, ln_mean);

    // --- Pass 2: Compute variance for this sample ---
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ln_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* var_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_var_cond", current_func);
    llvm::BasicBlock* var_body = llvm::BasicBlock::Create(ctx_.context(), "ln_var_body", current_func);
    llvm::BasicBlock* var_done = llvm::BasicBlock::Create(ctx_.context(), "ln_var_done", current_func);

    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_cond);
    fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), var_body, var_done);

    builder.SetInsertPoint(var_body);
    elem_offset = builder.CreateAdd(base_offset, fi);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), ln_mean);
    llvm::Value* diff = builder.CreateFSub(elem, mean_val);
    llvm::Value* sq_diff = builder.CreateFMul(diff, diff);
    cur_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, sq_diff), ln_sum);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(var_cond);

    builder.SetInsertPoint(var_done);
    llvm::Value* var_sum = builder.CreateLoad(ctx_.doubleType(), ln_sum);
    llvm::Value* var_val = builder.CreateFDiv(var_sum, feature_fp);
    llvm::Value* var_plus_eps = builder.CreateFAdd(var_val, epsilon);
    llvm::Value* std_val = builder.CreateCall(sqrt_func, {var_plus_eps});

    // --- Pass 3: Normalize, scale, and shift for this sample ---
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), feat_idx);

    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "ln_norm_done", current_func);

    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    fi = builder.CreateLoad(ctx_.int64Type(), feat_idx);
    builder.CreateCondBr(builder.CreateICmpULT(fi, feature_size), norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    elem_offset = builder.CreateAdd(base_offset, fi);
    elem_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, elem_offset);
    elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    mean_val = builder.CreateLoad(ctx_.doubleType(), ln_mean);

    // y = gamma * (x - mean) / std + beta
    llvm::Value* centered = builder.CreateFSub(elem, mean_val);
    llvm::Value* normalized = builder.CreateFDiv(centered, std_val);
    llvm::Value* scaled = builder.CreateFMul(normalized, gamma);
    llvm::Value* shifted = builder.CreateFAdd(scaled, beta);

    llvm::Value* out_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, elem_offset);
    llvm::Value* out_bits = builder.CreateBitCast(shifted, ctx_.int64Type());
    builder.CreateStore(out_bits, out_ptr);

    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), feat_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_done);
    // Advance to next sample
    builder.CreateStore(builder.CreateAdd(bi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(exit_block);
    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::extractAsDouble(llvm::Value* tagged_val) {
    if (!tagged_val) return nullptr;

    // Handle raw double - return as-is
    if (tagged_val->getType()->isDoubleTy()) return tagged_val;

    // Handle raw int64 - convert to double
    if (tagged_val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateSIToFP(tagged_val, ctx_.doubleType());
    }

    // Handle tagged value - check type and extract appropriately
    llvm::Value* type_tag = tagged_.getType(tagged_val);
    // Use getBaseType() to properly handle legacy types (>=32)
    // DO NOT use 0x0F mask - 34 & 0x0F = 2 (DOUBLE) which is WRONG!
    llvm::Value* base_type = tagged_.getBaseType(type_tag);

    llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
        llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));

    llvm::Value* dbl_val = tagged_.unpackDouble(tagged_val);
    llvm::Value* int_val = tagged_.unpackInt64(tagged_val);
    llvm::Value* int_as_dbl = ctx_.builder().CreateSIToFP(int_val, ctx_.doubleType());

    return ctx_.builder().CreateSelect(is_double, dbl_val, int_as_dbl, "as_double");
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
