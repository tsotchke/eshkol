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

    // Nested loops: batch -> output spatial position -> kernel
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_body", current_func);
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_body", current_func);
    llvm::BasicBlock* kernel_loop_cond = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_cond", current_func);
    llvm::BasicBlock* kernel_loop_body = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_body", current_func);
    llvm::BasicBlock* kernel_done = llvm::BasicBlock::Create(ctx_.context(), "c2_kernel_done", current_func);
    llvm::BasicBlock* loop_done = llvm::BasicBlock::Create(ctx_.context(), "c2_loop_done", current_func);
    llvm::BasicBlock* batch_done = llvm::BasicBlock::Create(ctx_.context(), "c2_batch_done", current_func);

    llvm::Value* bi = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_bi");
    llvm::Value* out_spatial_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_out_idx");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "c2_sum");
    llvm::Value* k_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "c2_k_idx");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), bi);
    builder.CreateBr(batch_cond);

    // Batch loop
    builder.SetInsertPoint(batch_cond);
    llvm::Value* curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* batch_cmp = builder.CreateICmpSLT(curr_bi, batch_size);
    builder.CreateCondBr(batch_cmp, batch_body, batch_done);

    builder.SetInsertPoint(batch_body);
    llvm::Value* in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    llvm::Value* out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), out_spatial_idx);
    builder.CreateBr(loop_cond);

    // Output spatial loop
    builder.SetInsertPoint(loop_cond);
    llvm::Value* curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    llvm::Value* loop_cmp = builder.CreateICmpSLT(curr_out, spatial_out);
    builder.CreateCondBr(loop_cmp, loop_body, loop_done);

    builder.SetInsertPoint(loop_body);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    llvm::Value* out_row = builder.CreateSDiv(curr_out, out_w);
    llvm::Value* out_col = builder.CreateSRem(curr_out, out_w);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(kernel_loop_cond);

    // Kernel loop
    builder.SetInsertPoint(kernel_loop_cond);
    llvm::Value* curr_k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    llvm::Value* k_cmp = builder.CreateICmpSLT(curr_k, k_total);
    builder.CreateCondBr(k_cmp, kernel_loop_body, kernel_done);

    builder.SetInsertPoint(kernel_loop_body);
    curr_k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    in_batch_offset = builder.CreateMul(curr_bi, spatial_in);
    out_row = builder.CreateSDiv(curr_out, out_w);
    out_col = builder.CreateSRem(curr_out, out_w);

    llvm::Value* k_row = builder.CreateSDiv(curr_k, k_w);
    llvm::Value* k_col = builder.CreateSRem(curr_k, k_w);

    llvm::Value* in_row = builder.CreateAdd(builder.CreateMul(out_row, stride), k_row);
    llvm::Value* in_col = builder.CreateAdd(builder.CreateMul(out_col, stride), k_col);
    llvm::Value* in_spatial = builder.CreateAdd(builder.CreateMul(in_row, in_w), in_col);
    llvm::Value* in_linear = builder.CreateAdd(in_batch_offset, in_spatial);

    llvm::Value* in_ptr = builder.CreateGEP(ctx_.int64Type(), input_elems, in_linear);
    llvm::Value* in_bits = builder.CreateLoad(ctx_.int64Type(), in_ptr);
    llvm::Value* in_val = builder.CreateBitCast(in_bits, ctx_.doubleType());

    llvm::Value* k_ptr = builder.CreateGEP(ctx_.int64Type(), kernel_elems, curr_k);
    llvm::Value* k_bits = builder.CreateLoad(ctx_.int64Type(), k_ptr);
    llvm::Value* k_val = builder.CreateBitCast(k_bits, ctx_.doubleType());

    llvm::Value* prod = builder.CreateFMul(in_val, k_val);
    llvm::Value* curr_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* new_sum = builder.CreateFAdd(curr_sum, prod);
    builder.CreateStore(new_sum, sum);

    llvm::Value* next_k = builder.CreateAdd(curr_k, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_k, k_idx);
    builder.CreateBr(kernel_loop_cond);

    builder.SetInsertPoint(kernel_done);
    curr_out = builder.CreateLoad(ctx_.int64Type(), out_spatial_idx);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    out_batch_offset = builder.CreateMul(curr_bi, spatial_out);
    llvm::Value* out_linear = builder.CreateAdd(out_batch_offset, curr_out);
    llvm::Value* res_ptr = builder.CreateGEP(ctx_.int64Type(), result_elems, out_linear);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    llvm::Value* sum_bits = builder.CreateBitCast(final_sum, ctx_.int64Type());
    builder.CreateStore(sum_bits, res_ptr);

    llvm::Value* next_out = builder.CreateAdd(curr_out, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_out, out_spatial_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_done);
    curr_bi = builder.CreateLoad(ctx_.int64Type(), bi);
    llvm::Value* next_bi = builder.CreateAdd(curr_bi, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_bi, bi);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_done);
    return tagged_.packHeapPtr(result_ptr);
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

    // First pass: compute mean
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
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
