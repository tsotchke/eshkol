/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Data Loading Infrastructure (extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split). Implements
 * the dataloader struct, batched iteration, and train/test split.
 *
 * Dataloader struct layout (arena-allocated, 8 fields × 8 bytes = 64 bytes):
 *   [0] data_ptr      — pointer to source data tensor
 *   [1] batch_size
 *   [2] num_samples
 *   [3] current_idx
 *   [4] indices_ptr   — shuffled indices array
 *   [5] sample_size   — elements per sample
 *   [6] sample_num_dims
 *   [7] sample_dims_ptr
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-dataloader-extract baseline.
 */
#include <eshkol/backend/tensor_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/BasicBlock.h>

namespace eshkol {

llvm::Value* TensorCodegen::makeDataloader(const eshkol_operations_t* op) {
    // make-dataloader: (make-dataloader data-tensor batch-size [shuffle])
    if (op->call_op.num_vars < 2) {
        eshkol_error("make-dataloader requires at least 2 arguments: data, batch-size");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get data tensor
    llvm::Value* data_tagged = codegenAST(&op->call_op.variables[0]);
    if (!data_tagged) return nullptr;
    llvm::Value* data_ptr = tagged_.unpackPtr(data_tagged);

    // Get batch size
    llvm::Value* batch_size_tagged = codegenAST(&op->call_op.variables[1]);
    if (!batch_size_tagged) return nullptr;
    llvm::Value* batch_size = batch_size_tagged;
    if (batch_size->getType() == ctx_.taggedValueType()) {
        batch_size = tagged_.unpackInt64(batch_size_tagged);
    }

    // Get shuffle flag (optional, default false)
    llvm::Value* shuffle_flag = llvm::ConstantInt::get(ctx_.int1Type(), 0);
    if (op->call_op.num_vars >= 3) {
        llvm::Value* shuffle_tagged = codegenAST(&op->call_op.variables[2]);
        if (shuffle_tagged) {
            // Check if it's true (non-null, non-false)
            llvm::Value* type_val = tagged_.getType(shuffle_tagged);
            llvm::Value* is_bool = builder.CreateICmpEQ(type_val,
                llvm::ConstantInt::get(ctx_.int8Type(), 3)); // ESHKOL_VALUE_BOOL
            llvm::Value* data_val = tagged_.unpackInt64(shuffle_tagged);
            shuffle_flag = builder.CreateAnd(is_bool,
                builder.CreateICmpNE(data_val, llvm::ConstantInt::get(ctx_.int64Type(), 0)));
        }
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get tensor properties
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* num_dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, data_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get number of samples (first dimension)
    llvm::Value* first_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), first_dim_ptr);

    // Compute sample size (total_elements / num_samples)
    llvm::Value* sample_size = builder.CreateUDiv(total_elements, num_samples);

    // Allocate dataloader structure (8 fields * 8 bytes = 64 bytes)
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* loader_size = llvm::ConstantInt::get(ctx_.int64Type(), 64);
    llvm::Value* loader_ptr = builder.CreateCall(arena_alloc, {arena_ptr, loader_size}, "loader_ptr");

    // Allocate indices array
    llvm::Value* indices_size = builder.CreateMul(num_samples,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* indices_ptr = builder.CreateCall(arena_alloc, {arena_ptr, indices_size}, "indices_ptr");

    // Allocate sample_dims array (num_dims - 1 dimensions)
    llvm::Value* sample_num_dims = builder.CreateSub(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* sample_dims_size = builder.CreateMul(sample_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* sample_dims_ptr = builder.CreateCall(arena_alloc, {arena_ptr, sample_dims_size}, "sample_dims_ptr");

    // Store fields in loader structure
    // Field 0: data_ptr
    llvm::Value* field0 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* data_ptr_int = builder.CreatePtrToInt(data_ptr, ctx_.int64Type());
    builder.CreateStore(data_ptr_int, field0);

    // Field 1: batch_size
    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(batch_size, field1);

    // Field 2: num_samples
    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    builder.CreateStore(num_samples, field2);

    // Field 3: current_idx (start at 0)
    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), field3);

    // Field 4: indices_ptr
    llvm::Value* field4 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 4));
    llvm::Value* indices_ptr_int = builder.CreatePtrToInt(indices_ptr, ctx_.int64Type());
    builder.CreateStore(indices_ptr_int, field4);

    // Field 5: sample_size
    llvm::Value* field5 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 5));
    builder.CreateStore(sample_size, field5);

    // Field 6: sample_num_dims
    llvm::Value* field6 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 6));
    builder.CreateStore(sample_num_dims, field6);

    // Field 7: sample_dims_ptr
    llvm::Value* field7 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 7));
    llvm::Value* sample_dims_ptr_int = builder.CreatePtrToInt(sample_dims_ptr, ctx_.int64Type());
    builder.CreateStore(sample_dims_ptr_int, field7);

    // Initialize indices and sample_dims with loops
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Initialize indices: indices[i] = i (sequential initially)
    llvm::BasicBlock* idx_init_loop = llvm::BasicBlock::Create(ctx_.context(), "idx_init_loop", current_func);
    llvm::BasicBlock* idx_init_body = llvm::BasicBlock::Create(ctx_.context(), "idx_init_body", current_func);
    llvm::BasicBlock* copy_dims = llvm::BasicBlock::Create(ctx_.context(), "copy_dims", current_func);

    llvm::Value* idx_counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx_counter);
    builder.CreateBr(idx_init_loop);

    builder.SetInsertPoint(idx_init_loop);
    llvm::Value* idx_i = builder.CreateLoad(ctx_.int64Type(), idx_counter);
    builder.CreateCondBr(builder.CreateICmpULT(idx_i, num_samples), idx_init_body, copy_dims);

    builder.SetInsertPoint(idx_init_body);
    llvm::Value* idx_slot = builder.CreateGEP(ctx_.int64Type(), indices_ptr, idx_i);
    builder.CreateStore(idx_i, idx_slot);
    builder.CreateStore(builder.CreateAdd(idx_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), idx_counter);
    builder.CreateBr(idx_init_loop);

    // Copy sample dimensions (all dims except first)
    builder.SetInsertPoint(copy_dims);
    llvm::BasicBlock* dim_loop = llvm::BasicBlock::Create(ctx_.context(), "dim_loop", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "dim_body", current_func);
    llvm::BasicBlock* shuffle_check = llvm::BasicBlock::Create(ctx_.context(), "shuffle_check", current_func);

    llvm::Value* dim_counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_counter);
    builder.CreateBr(dim_loop);

    builder.SetInsertPoint(dim_loop);
    llvm::Value* dim_i = builder.CreateLoad(ctx_.int64Type(), dim_counter);
    builder.CreateCondBr(builder.CreateICmpULT(dim_i, sample_num_dims), dim_body, shuffle_check);

    builder.SetInsertPoint(dim_body);
    // Copy dims[i+1] to sample_dims[i]
    llvm::Value* src_dim_idx = builder.CreateAdd(dim_i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* src_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr, src_dim_idx);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim_ptr);
    llvm::Value* dst_dim_ptr = builder.CreateGEP(ctx_.int64Type(), sample_dims_ptr, dim_i);
    builder.CreateStore(dim_val, dst_dim_ptr);
    builder.CreateStore(builder.CreateAdd(dim_i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dim_counter);
    builder.CreateBr(dim_loop);

    // Shuffle indices if requested (Fisher-Yates shuffle)
    builder.SetInsertPoint(shuffle_check);
    llvm::BasicBlock* do_shuffle = llvm::BasicBlock::Create(ctx_.context(), "do_shuffle", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "exit_loader", current_func);
    builder.CreateCondBr(shuffle_flag, do_shuffle, exit_block);

    builder.SetInsertPoint(do_shuffle);
    // Simple LCG-based shuffle (not cryptographically secure, but good enough for ML)
    llvm::BasicBlock* shuffle_loop = llvm::BasicBlock::Create(ctx_.context(), "shuffle_loop", current_func);
    llvm::BasicBlock* shuffle_body = llvm::BasicBlock::Create(ctx_.context(), "shuffle_body", current_func);

    llvm::Value* shuffle_i = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* rng_state = builder.CreateAlloca(ctx_.int64Type());
    // Seed with a mix of pointer and sample count
    llvm::Value* seed = builder.CreateXor(data_ptr_int, num_samples);
    builder.CreateStore(seed, rng_state);
    builder.CreateStore(builder.CreateSub(num_samples, llvm::ConstantInt::get(ctx_.int64Type(), 1)), shuffle_i);
    builder.CreateBr(shuffle_loop);

    builder.SetInsertPoint(shuffle_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), shuffle_i);
    builder.CreateCondBr(builder.CreateICmpSGT(si, llvm::ConstantInt::get(ctx_.int64Type(), 0)),
                          shuffle_body, exit_block);

    builder.SetInsertPoint(shuffle_body);
    // LCG: state = (state * 6364136223846793005 + 1) % 2^64
    llvm::Value* state = builder.CreateLoad(ctx_.int64Type(), rng_state);
    llvm::Value* mult = llvm::ConstantInt::get(ctx_.int64Type(), 6364136223846793005ULL);
    llvm::Value* new_state = builder.CreateAdd(builder.CreateMul(state, mult),
                                                llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(new_state, rng_state);

    // j = new_state % (i + 1)
    llvm::Value* si_plus_1 = builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* j = builder.CreateURem(new_state, si_plus_1);

    // Swap indices[i] and indices[j]
    llvm::Value* idx_i_ptr = builder.CreateGEP(ctx_.int64Type(), indices_ptr, si);
    llvm::Value* idx_j_ptr = builder.CreateGEP(ctx_.int64Type(), indices_ptr, j);
    llvm::Value* val_i = builder.CreateLoad(ctx_.int64Type(), idx_i_ptr);
    llvm::Value* val_j = builder.CreateLoad(ctx_.int64Type(), idx_j_ptr);
    builder.CreateStore(val_j, idx_i_ptr);
    builder.CreateStore(val_i, idx_j_ptr);

    builder.CreateStore(builder.CreateSub(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), shuffle_i);
    builder.CreateBr(shuffle_loop);

    builder.SetInsertPoint(exit_block);

    // Pack as HEAP_PTR (using generic pointer packing)
    return tagged_.packPtr(loader_ptr, ESHKOL_VALUE_HEAP_PTR, 0);
}

llvm::Value* TensorCodegen::dataloaderNext(const eshkol_operations_t* op) {
    // dataloader-next: (dataloader-next loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-next requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Load loader fields
    llvm::Value* field0 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* data_ptr_int = builder.CreateLoad(ctx_.int64Type(), field0);
    llvm::Value* data_ptr = builder.CreateIntToPtr(data_ptr_int, ctx_.ptrType());

    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), field1);

    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    llvm::Value* current_idx = builder.CreateLoad(ctx_.int64Type(), field3);

    llvm::Value* field4 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 4));
    llvm::Value* indices_ptr_int = builder.CreateLoad(ctx_.int64Type(), field4);
    llvm::Value* indices_ptr = builder.CreateIntToPtr(indices_ptr_int, ctx_.ptrType());

    llvm::Value* field5 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 5));
    llvm::Value* sample_size = builder.CreateLoad(ctx_.int64Type(), field5);

    llvm::Value* field6 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 6));
    llvm::Value* sample_num_dims = builder.CreateLoad(ctx_.int64Type(), field6);

    llvm::Value* field7 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 7));
    llvm::Value* sample_dims_ptr_int = builder.CreateLoad(ctx_.int64Type(), field7);
    llvm::Value* sample_dims_ptr = builder.CreateIntToPtr(sample_dims_ptr_int, ctx_.ptrType());

    // Get source data elements
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* src_elems_field = builder.CreateStructGEP(tensor_type, data_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), src_elems_field);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Check if we have more samples
    llvm::BasicBlock* has_data = llvm::BasicBlock::Create(ctx_.context(), "has_data", current_func);
    llvm::BasicBlock* no_data = llvm::BasicBlock::Create(ctx_.context(), "no_data", current_func);
    llvm::BasicBlock* copy_batch = llvm::BasicBlock::Create(ctx_.context(), "copy_batch", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "exit_next", current_func);

    builder.CreateCondBr(builder.CreateICmpULT(current_idx, num_samples), has_data, no_data);

    builder.SetInsertPoint(no_data);
    llvm::Value* null_result = tagged_.packNull();
    builder.CreateBr(exit_block);

    builder.SetInsertPoint(has_data);

    // Calculate actual batch size (may be smaller for last batch)
    llvm::Value* remaining = builder.CreateSub(num_samples, current_idx);
    llvm::Value* actual_batch = builder.CreateSelect(
        builder.CreateICmpULT(remaining, batch_size), remaining, batch_size);

    // Allocate batch tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* batch_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "batch_ptr");

    // Batch dimensions: [actual_batch, sample_dims...]
    llvm::Value* batch_num_dims = builder.CreateAdd(sample_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_dims_size = builder.CreateMul(batch_num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* batch_dims = builder.CreateCall(arena_alloc, {arena_ptr, batch_dims_size}, "batch_dims");

    // Store batch dimension at index 0
    llvm::Value* batch_dim0 = builder.CreateGEP(ctx_.int64Type(), batch_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(actual_batch, batch_dim0);

    // Copy sample dimensions
    llvm::BasicBlock* copy_dim_loop = llvm::BasicBlock::Create(ctx_.context(), "copy_dim_loop", current_func);
    llvm::BasicBlock* copy_dim_body = llvm::BasicBlock::Create(ctx_.context(), "copy_dim_body", current_func);

    llvm::Value* dim_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dim_i);
    builder.CreateBr(copy_dim_loop);

    builder.SetInsertPoint(copy_dim_loop);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dim_i);
    builder.CreateCondBr(builder.CreateICmpULT(di, sample_num_dims), copy_dim_body, copy_batch);

    builder.SetInsertPoint(copy_dim_body);
    llvm::Value* src_dim = builder.CreateGEP(ctx_.int64Type(), sample_dims_ptr, di);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), src_dim);
    llvm::Value* dst_dim_idx = builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* dst_dim = builder.CreateGEP(ctx_.int64Type(), batch_dims, dst_dim_idx);
    builder.CreateStore(dim_val, dst_dim);
    builder.CreateStore(builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dim_i);
    builder.CreateBr(copy_dim_loop);

    // Copy batch data
    builder.SetInsertPoint(copy_batch);
    llvm::Value* batch_total_elems = builder.CreateMul(actual_batch, sample_size);
    llvm::Value* batch_elems_size = builder.CreateMul(batch_total_elems,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* batch_elems = builder.CreateCall(arena_alloc, {arena_ptr, batch_elems_size}, "batch_elems");

    llvm::BasicBlock* sample_loop = llvm::BasicBlock::Create(ctx_.context(), "sample_loop", current_func);
    llvm::BasicBlock* sample_body = llvm::BasicBlock::Create(ctx_.context(), "sample_body", current_func);
    llvm::BasicBlock* finalize_batch = llvm::BasicBlock::Create(ctx_.context(), "finalize_batch", current_func);

    llvm::Value* sample_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sample_i);
    builder.CreateBr(sample_loop);

    builder.SetInsertPoint(sample_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), sample_i);
    builder.CreateCondBr(builder.CreateICmpULT(si, actual_batch), sample_body, finalize_batch);

    builder.SetInsertPoint(sample_body);
    // Get the actual sample index from indices array
    llvm::Value* global_idx = builder.CreateAdd(current_idx, si);
    llvm::Value* idx_slot = builder.CreateGEP(ctx_.int64Type(), indices_ptr, global_idx);
    llvm::Value* real_sample_idx = builder.CreateLoad(ctx_.int64Type(), idx_slot);

    // Copy sample data
    llvm::Value* src_offset = builder.CreateMul(real_sample_idx, sample_size);
    llvm::Value* dst_offset = builder.CreateMul(si, sample_size);

    // Copy element by element
    llvm::BasicBlock* elem_loop = llvm::BasicBlock::Create(ctx_.context(), "elem_loop", current_func);
    llvm::BasicBlock* elem_body = llvm::BasicBlock::Create(ctx_.context(), "elem_body", current_func);
    llvm::BasicBlock* elem_done = llvm::BasicBlock::Create(ctx_.context(), "elem_done", current_func);

    llvm::Value* elem_i = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_i);
    builder.CreateBr(elem_loop);

    builder.SetInsertPoint(elem_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_i);
    builder.CreateCondBr(builder.CreateICmpULT(ei, sample_size), elem_body, elem_done);

    builder.SetInsertPoint(elem_body);
    llvm::Value* src_elem_idx = builder.CreateAdd(src_offset, ei);
    llvm::Value* dst_elem_idx = builder.CreateAdd(dst_offset, ei);
    llvm::Value* src_ptr = builder.CreateGEP(ctx_.int64Type(), src_elems, src_elem_idx);
    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), batch_elems, dst_elem_idx);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.int64Type(), src_ptr);
    builder.CreateStore(elem_val, dst_ptr);
    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_i);
    builder.CreateBr(elem_loop);

    builder.SetInsertPoint(elem_done);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sample_i);
    builder.CreateBr(sample_loop);

    // Finalize batch tensor
    builder.SetInsertPoint(finalize_batch);
    llvm::Value* b_dims_field = builder.CreateStructGEP(tensor_type, batch_ptr, 0);
    builder.CreateStore(batch_dims, b_dims_field);
    llvm::Value* b_ndim_field = builder.CreateStructGEP(tensor_type, batch_ptr, 1);
    builder.CreateStore(batch_num_dims, b_ndim_field);
    llvm::Value* b_elems_field = builder.CreateStructGEP(tensor_type, batch_ptr, 2);
    builder.CreateStore(batch_elems, b_elems_field);
    llvm::Value* b_total_field = builder.CreateStructGEP(tensor_type, batch_ptr, 3);
    builder.CreateStore(batch_total_elems, b_total_field);

    // Update current_idx
    llvm::Value* new_idx = builder.CreateAdd(current_idx, actual_batch);
    builder.CreateStore(new_idx, field3);

    // Compute batch result BEFORE branching (PHI incoming must be in predecessor block)
    llvm::Value* batch_result = tagged_.packHeapPtr(batch_ptr);
    builder.CreateBr(exit_block);

    // Exit block with PHI for result
    builder.SetInsertPoint(exit_block);
    llvm::PHINode* result = builder.CreatePHI(ctx_.taggedValueType(), 2, "next_result");
    result->addIncoming(null_result, no_data);
    result->addIncoming(batch_result, finalize_batch);

    return result;
}

llvm::Value* TensorCodegen::dataloaderReset(const eshkol_operations_t* op) {
    // dataloader-reset!: (dataloader-reset! loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-reset! requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Reset current_idx to 0
    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), field3);

    return loader_tagged;
}

llvm::Value* TensorCodegen::dataloaderLength(const eshkol_operations_t* op) {
    // dataloader-length: (dataloader-length loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-length requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Load batch_size and num_samples
    llvm::Value* field1 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), field1);

    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    // num_batches = ceil(num_samples / batch_size)
    llvm::Value* full_batches = builder.CreateUDiv(num_samples, batch_size);
    llvm::Value* remainder = builder.CreateURem(num_samples, batch_size);
    llvm::Value* has_partial = builder.CreateICmpNE(remainder,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* extra = builder.CreateZExt(has_partial, ctx_.int64Type());
    llvm::Value* num_batches = builder.CreateAdd(full_batches, extra);

    return tagged_.packInt64(num_batches);
}

llvm::Value* TensorCodegen::dataloaderHasNext(const eshkol_operations_t* op) {
    // dataloader-has-next?: (dataloader-has-next? loader)
    if (op->call_op.num_vars < 1) {
        eshkol_error("dataloader-has-next? requires 1 argument: loader");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* loader_tagged = codegenAST(&op->call_op.variables[0]);
    if (!loader_tagged) return nullptr;
    llvm::Value* loader_ptr = tagged_.unpackPtr(loader_tagged);

    // Load current_idx and num_samples
    llvm::Value* field2 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), field2);

    llvm::Value* field3 = builder.CreateGEP(ctx_.int64Type(), loader_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));
    llvm::Value* current_idx = builder.CreateLoad(ctx_.int64Type(), field3);

    // has_next = current_idx < num_samples
    llvm::Value* has_next = builder.CreateICmpULT(current_idx, num_samples);

    return tagged_.packBool(has_next);
}

llvm::Value* TensorCodegen::trainTestSplit(const eshkol_operations_t* op) {
    // train-test-split: (train-test-split data ratio [shuffle])
    // Returns a vector of (train-data test-data)
    if (op->call_op.num_vars < 2) {
        eshkol_error("train-test-split requires at least 2 arguments: data, ratio");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get data tensor
    llvm::Value* data_tagged = codegenAST(&op->call_op.variables[0]);
    if (!data_tagged) return nullptr;
    llvm::Value* data_ptr = tagged_.unpackPtr(data_tagged);

    // Get ratio
    llvm::Value* ratio_tagged = codegenAST(&op->call_op.variables[1]);
    if (!ratio_tagged) return nullptr;
    llvm::Value* ratio = ratio_tagged;
    if (ratio->getType() == ctx_.taggedValueType()) {
        ratio = tagged_.unpackDouble(ratio_tagged);
    } else if (ratio->getType()->isIntegerTy(64)) {
        ratio = ctx_.builder().CreateBitCast(ratio, ctx_.doubleType());
    }

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get tensor properties
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 0);
    llvm::Value* dims_ptr = builder.CreateLoad(ctx_.ptrType(), dims_field);
    llvm::Value* num_dims_field = builder.CreateStructGEP(tensor_type, data_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_field);
    llvm::Value* elems_field = builder.CreateStructGEP(tensor_type, data_ptr, 2);
    llvm::Value* src_elems = builder.CreateLoad(ctx_.ptrType(), elems_field);
    llvm::Value* total_field = builder.CreateStructGEP(tensor_type, data_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_field);

    // Get number of samples
    llvm::Value* first_dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* num_samples = builder.CreateLoad(ctx_.int64Type(), first_dim_ptr);
    llvm::Value* sample_size = builder.CreateUDiv(total_elements, num_samples);

    // Calculate split point
    llvm::Value* num_samples_f = builder.CreateSIToFP(num_samples, ctx_.doubleType());
    llvm::Value* train_size_f = builder.CreateFMul(num_samples_f, ratio);
    llvm::Value* train_size = builder.CreateFPToSI(train_size_f, ctx_.int64Type());
    llvm::Value* test_size = builder.CreateSub(num_samples, train_size);

    // Allocate train tensor
    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* train_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "train_ptr");

    // Allocate train dims (same as original)
    llvm::Value* dims_size = builder.CreateMul(num_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* train_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "train_dims");
    builder.CreateMemCpy(train_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);
    // Update first dimension for train
    llvm::Value* train_dim0 = builder.CreateGEP(ctx_.int64Type(), train_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(train_size, train_dim0);

    // Allocate train elements
    llvm::Value* train_total = builder.CreateMul(train_size, sample_size);
    llvm::Value* train_elems_size = builder.CreateMul(train_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* train_elems = builder.CreateCall(arena_alloc, {arena_ptr, train_elems_size}, "train_elems");
    builder.CreateMemCpy(train_elems, llvm::MaybeAlign(8), src_elems, llvm::MaybeAlign(8), train_elems_size);

    // Set train tensor fields
    llvm::Value* train_dims_field = builder.CreateStructGEP(tensor_type, train_ptr, 0);
    builder.CreateStore(train_dims, train_dims_field);
    llvm::Value* train_ndim_field = builder.CreateStructGEP(tensor_type, train_ptr, 1);
    builder.CreateStore(num_dims, train_ndim_field);
    llvm::Value* train_elems_field = builder.CreateStructGEP(tensor_type, train_ptr, 2);
    builder.CreateStore(train_elems, train_elems_field);
    llvm::Value* train_total_field = builder.CreateStructGEP(tensor_type, train_ptr, 3);
    builder.CreateStore(train_total, train_total_field);

    // Allocate test tensor
    llvm::Value* test_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "test_ptr");
    llvm::Value* test_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_size}, "test_dims");
    builder.CreateMemCpy(test_dims, llvm::MaybeAlign(8), dims_ptr, llvm::MaybeAlign(8), dims_size);
    llvm::Value* test_dim0 = builder.CreateGEP(ctx_.int64Type(), test_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(test_size, test_dim0);

    // Allocate test elements
    llvm::Value* test_total = builder.CreateMul(test_size, sample_size);
    llvm::Value* test_elems_size = builder.CreateMul(test_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* test_elems = builder.CreateCall(arena_alloc, {arena_ptr, test_elems_size}, "test_elems");
    // Copy from offset
    llvm::Value* test_offset_bytes = builder.CreateMul(train_total,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));
    llvm::Value* test_src = builder.CreateGEP(ctx_.int8Type(), src_elems, test_offset_bytes);
    builder.CreateMemCpy(test_elems, llvm::MaybeAlign(8), test_src, llvm::MaybeAlign(8), test_elems_size);

    // Set test tensor fields
    llvm::Value* test_dims_field = builder.CreateStructGEP(tensor_type, test_ptr, 0);
    builder.CreateStore(test_dims, test_dims_field);
    llvm::Value* test_ndim_field = builder.CreateStructGEP(tensor_type, test_ptr, 1);
    builder.CreateStore(num_dims, test_ndim_field);
    llvm::Value* test_elems_field = builder.CreateStructGEP(tensor_type, test_ptr, 2);
    builder.CreateStore(test_elems, test_elems_field);
    llvm::Value* test_total_field = builder.CreateStructGEP(tensor_type, test_ptr, 3);
    builder.CreateStore(test_total, test_total_field);

    // Create result vector with 2 elements
    // arena_allocate_vector_with_header creates: [header(8)] + [length(8)] + [elements(16*n)]
    llvm::Function* alloc_vec = mem_.getArenaAllocateVectorWithHeader();
    llvm::Value* result_vec = builder.CreateCall(alloc_vec, {arena_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)}, "split_result");

    // Set length field at offset 0
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 2), result_vec);

    // Elements start at offset 8 (after length field)
    llvm::Value* vec_data_ptr = builder.CreateGEP(ctx_.int8Type(), result_vec,
        llvm::ConstantInt::get(ctx_.int64Type(), 8));

    // Store train and test tensors in vector
    llvm::Value* vec_elem0 = builder.CreateGEP(ctx_.taggedValueType(), vec_data_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    builder.CreateStore(tagged_.packHeapPtr(train_ptr), vec_elem0);
    llvm::Value* vec_elem1 = builder.CreateGEP(ctx_.taggedValueType(), vec_data_ptr,
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(tagged_.packHeapPtr(test_ptr), vec_elem1);

    return tagged_.packHeapPtr(result_vec);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
