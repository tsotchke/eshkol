/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Transformer Architecture (Track 8). Extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split. Implements the
 * attention stack (scaled dot-product, multi-head), positional and
 * rotary embeddings, masking, feed-forward, dropout, and embedding
 * lookup.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-transformer-extract baseline.
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

// ════════════════════════════════════════════════════════════════════════════════
// TRANSFORMER ARCHITECTURE (Track 8)
// ════════════════════════════════════════════════════════════════════════════════

// === Track 8.1: Scaled Dot-Product Attention ===

llvm::Value* TensorCodegen::scaledDotProductAttention(const eshkol_operations_t* op) {
    // Scaled Dot-Product Attention from "Attention Is All You Need"
    // scores = Q @ K^T / sqrt(d_k)
    // attention_weights = softmax(scores + mask)
    // output = attention_weights @ V
    //
    // Supports both 2D (seq_len, d_k) and 3D (batch, seq_len, d_k) inputs

    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("scaled-dot-attention requires 3-4 arguments: Q K V [mask]");
        return nullptr;
    }

    llvm::Value* q_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* k_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* v_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* mask_val = (op->call_op.num_vars == 4)
        ? codegenAST(&op->call_op.variables[3]) : nullptr;

    if (!q_val || !k_val || !v_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack Q tensor
    llvm::Value* q_ptr_int = tagged_.unpackInt64(q_val);
    llvm::Value* q_ptr = builder.CreateIntToPtr(q_ptr_int, ctx_.ptrType());

    llvm::Value* q_dims_field = builder.CreateStructGEP(tensor_type, q_ptr, 0);
    llvm::Value* q_dims_ptr = builder.CreateLoad(ctx_.ptrType(), q_dims_field);
    llvm::Value* q_ndim_field = builder.CreateStructGEP(tensor_type, q_ptr, 1);
    llvm::Value* q_ndim = builder.CreateLoad(ctx_.int64Type(), q_ndim_field);
    llvm::Value* q_elems_field = builder.CreateStructGEP(tensor_type, q_ptr, 2);
    llvm::Value* q_elems = builder.CreateLoad(ctx_.ptrType(), q_elems_field);

    // Unpack K tensor
    llvm::Value* k_ptr_int = tagged_.unpackInt64(k_val);
    llvm::Value* k_ptr = builder.CreateIntToPtr(k_ptr_int, ctx_.ptrType());

    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, k_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, k_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Unpack V tensor
    llvm::Value* v_ptr_int = tagged_.unpackInt64(v_val);
    llvm::Value* v_ptr = builder.CreateIntToPtr(v_ptr_int, ctx_.ptrType());

    llvm::Value* v_dims_field = builder.CreateStructGEP(tensor_type, v_ptr, 0);
    llvm::Value* v_dims_ptr = builder.CreateLoad(ctx_.ptrType(), v_dims_field);
    llvm::Value* v_elems_field = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_field);

    // Determine dimensions based on 2D or 3D input
    llvm::Value* is_3d = builder.CreateICmpEQ(q_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));

    // Get dimensions
    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), q_dims_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_q_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* seq_q = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, seq_q_idx));

    llvm::Value* d_k_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* d_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, d_k_idx));

    llvm::Value* seq_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, seq_q_idx));

    llvm::Value* d_v = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), v_dims_ptr, d_k_idx));

    // Compute sqrt(d_k) for scaling — guard against d_k == 0
    llvm::Value* d_k_double = builder.CreateSIToFP(d_k, ctx_.doubleType());
    llvm::Value* d_k_safe = builder.CreateSelect(
        builder.CreateFCmpOLE(d_k_double, llvm::ConstantFP::get(ctx_.doubleType(), 0.0)),
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0), d_k_double, "dk_safe");
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* scale = builder.CreateCall(sqrt_func, {d_k_safe}, "sqrt_dk");

    // Allocate scores matrix: (batch, seq_q, seq_k)
    llvm::Value* scores_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, seq_k));
    llvm::Value* scores_bytes = builder.CreateMul(scores_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* scores_ptr = builder.CreateCall(arena_alloc,
        {arena_ptr, scores_bytes}, "attn_scores");

    // Allocate output tensor: (batch, seq_q, d_v)
    llvm::Value* output_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, d_v));
    llvm::Value* output_bytes = builder.CreateMul(output_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc,
        {arena_ptr, output_bytes}, "attn_output");

    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    llvm::Value* j_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "j_idx");
    llvm::Value* k_iter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_iter");
    llvm::Value* row_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "row_idx");
    llvm::Value* col_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "col_idx");
    llvm::Value* dot_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "dot_sum");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    llvm::Value* sum_exp = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum_exp");
    llvm::Value* out_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "out_sum");

    // === Compute Q @ K^T / sqrt(d_k) + mask ===
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_body", current_func);
    llvm::BasicBlock* softmax_init = llvm::BasicBlock::Create(ctx_.context(), "attn_softmax_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* batch_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(batch_done, softmax_init, batch_body);

    builder.SetInsertPoint(batch_body);

    // Compute base offsets for this batch
    llvm::Value* q_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, d_k));
    llvm::Value* k_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_k, d_k));
    llvm::Value* scores_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, seq_k));

    // Inner loops: for each (i, j) in (seq_q, seq_k), compute dot product
    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "attn_i_body", current_func);
    llvm::BasicBlock* batch_next = llvm::BasicBlock::Create(ctx_.context(), "attn_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, seq_q);
    builder.CreateCondBr(i_done, batch_next, i_body);

    builder.SetInsertPoint(i_body);

    // j loop
    llvm::BasicBlock* j_cond = llvm::BasicBlock::Create(ctx_.context(), "attn_j_cond", current_func);
    llvm::BasicBlock* j_body = llvm::BasicBlock::Create(ctx_.context(), "attn_j_body", current_func);
    llvm::BasicBlock* i_next = llvm::BasicBlock::Create(ctx_.context(), "attn_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_idx);
    builder.CreateBr(j_cond);

    builder.SetInsertPoint(j_cond);
    llvm::Value* j = builder.CreateLoad(ctx_.int64Type(), j_idx);
    llvm::Value* j_done = builder.CreateICmpUGE(j, seq_k);
    builder.CreateCondBr(j_done, i_next, j_body);

    builder.SetInsertPoint(j_body);

    // Compute dot product: Q[b,i,:] @ K[b,j,:]
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "dot_body", current_func);
    llvm::BasicBlock* dot_done = llvm::BasicBlock::Create(ctx_.context(), "dot_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dot_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_iter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* kk = builder.CreateLoad(ctx_.int64Type(), k_iter);
    llvm::Value* k_done = builder.CreateICmpUGE(kk, d_k);
    builder.CreateCondBr(k_done, dot_done, dot_body);

    builder.SetInsertPoint(dot_body);
    // Q[b, i, k] = q_elems[q_batch_offset + i * d_k + k]
    llvm::Value* q_idx = builder.CreateAdd(q_batch_offset,
        builder.CreateAdd(builder.CreateMul(i, d_k), kk));
    llvm::Value* q_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), q_elems, q_idx));
    // K[b, j, k] = k_elems[k_batch_offset + j * d_k + k]
    llvm::Value* k_idx = builder.CreateAdd(k_batch_offset,
        builder.CreateAdd(builder.CreateMul(j, d_k), kk));
    llvm::Value* k_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), k_elems, k_idx));

    llvm::Value* prod = builder.CreateFMul(q_elem, k_elem);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, prod), dot_sum);

    builder.CreateStore(builder.CreateAdd(kk,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_iter);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_done);
    // Scale by 1/sqrt(d_k)
    llvm::Value* final_dot = builder.CreateLoad(ctx_.doubleType(), dot_sum);
    llvm::Value* scaled_dot = builder.CreateFDiv(final_dot, scale, "scaled_score");

    // Add mask if provided
    if (mask_val) {
        llvm::Value* mask_ptr_int = tagged_.unpackInt64(mask_val);
        llvm::Value* mask_ptr = builder.CreateIntToPtr(mask_ptr_int, ctx_.ptrType());
        llvm::Value* mask_elems_field = builder.CreateStructGEP(tensor_type, mask_ptr, 2);
        llvm::Value* mask_elems = builder.CreateLoad(ctx_.ptrType(), mask_elems_field);
        // mask[i, j] or mask[b, i, j]
        llvm::Value* mask_idx = builder.CreateAdd(builder.CreateMul(i, seq_k), j);
        llvm::Value* mask_elem = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), mask_elems, mask_idx));
        scaled_dot = builder.CreateFAdd(scaled_dot, mask_elem, "masked_score");
    }

    // Store in scores[b, i, j]
    llvm::Value* scores_idx = builder.CreateAdd(scores_batch_offset,
        builder.CreateAdd(builder.CreateMul(i, seq_k), j));
    builder.CreateStore(scaled_dot,
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, scores_idx));

    // Next j
    builder.CreateStore(builder.CreateAdd(j,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_idx);
    builder.CreateBr(j_cond);

    // Next i
    builder.SetInsertPoint(i_next);
    builder.CreateStore(builder.CreateAdd(i,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(i_cond);

    // Next batch
    builder.SetInsertPoint(batch_next);
    builder.CreateStore(builder.CreateAdd(b,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    // === Apply softmax row-wise (along seq_k dimension) ===
    builder.SetInsertPoint(softmax_init);

    llvm::BasicBlock* sm_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_cond", current_func);
    llvm::BasicBlock* sm_batch_body = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_body", current_func);
    llvm::BasicBlock* output_init = llvm::BasicBlock::Create(ctx_.context(), "output_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(sm_batch_cond);

    builder.SetInsertPoint(sm_batch_cond);
    llvm::Value* b2 = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b2_done = builder.CreateICmpUGE(b2, batch_size);
    builder.CreateCondBr(b2_done, output_init, sm_batch_body);

    builder.SetInsertPoint(sm_batch_body);

    llvm::Value* sm_batch_offset = builder.CreateMul(b2, builder.CreateMul(seq_q, seq_k));

    // Softmax each row (for each query position)
    llvm::BasicBlock* sm_row_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_row_cond", current_func);
    llvm::BasicBlock* sm_row_body = llvm::BasicBlock::Create(ctx_.context(), "sm_row_body", current_func);
    llvm::BasicBlock* sm_batch_next = llvm::BasicBlock::Create(ctx_.context(), "sm_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), row_idx);
    builder.CreateBr(sm_row_cond);

    builder.SetInsertPoint(sm_row_cond);
    llvm::Value* row = builder.CreateLoad(ctx_.int64Type(), row_idx);
    llvm::Value* row_done = builder.CreateICmpUGE(row, seq_q);
    builder.CreateCondBr(row_done, sm_batch_next, sm_row_body);

    builder.SetInsertPoint(sm_row_body);

    llvm::Value* row_offset = builder.CreateAdd(sm_batch_offset,
        builder.CreateMul(row, seq_k));

    // Find max for numerical stability
    llvm::BasicBlock* max_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_max_cond", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "sm_max_body", current_func);
    llvm::BasicBlock* exp_init = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_init", current_func);

    llvm::Value* first_score = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, row_offset));
    builder.CreateStore(first_score, max_val);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), col_idx);
    builder.CreateBr(max_cond);

    builder.SetInsertPoint(max_cond);
    llvm::Value* col = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col_done = builder.CreateICmpUGE(col, seq_k);
    builder.CreateCondBr(col_done, exp_init, max_body);

    builder.SetInsertPoint(max_body);
    llvm::Value* score_idx = builder.CreateAdd(row_offset, col);
    llvm::Value* score_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx));
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(score_val, cur_max);
    builder.CreateStore(builder.CreateSelect(is_greater, score_val, cur_max), max_val);
    builder.CreateStore(builder.CreateAdd(col,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(max_cond);

    // Compute exp(x - max) and sum
    builder.SetInsertPoint(exp_init);
    llvm::BasicBlock* exp_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_cond", current_func);
    llvm::BasicBlock* exp_body = llvm::BasicBlock::Create(ctx_.context(), "sm_exp_body", current_func);
    llvm::BasicBlock* norm_init = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_init", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_exp);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_idx);
    builder.CreateBr(exp_cond);

    builder.SetInsertPoint(exp_cond);
    llvm::Value* col2 = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col2_done = builder.CreateICmpUGE(col2, seq_k);
    builder.CreateCondBr(col2_done, norm_init, exp_body);

    builder.SetInsertPoint(exp_body);
    llvm::Value* score_idx2 = builder.CreateAdd(row_offset, col2);
    llvm::Value* score_ptr = builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx2);
    llvm::Value* score_val2 = builder.CreateLoad(ctx_.doubleType(), score_ptr);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* shifted = builder.CreateFSub(score_val2, final_max);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {shifted});
    builder.CreateStore(exp_val, score_ptr);  // Store exp value back
    llvm::Value* cur_exp_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFAdd(cur_exp_sum, exp_val), sum_exp);
    builder.CreateStore(builder.CreateAdd(col2,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(exp_cond);

    // Normalize by sum
    builder.SetInsertPoint(norm_init);
    llvm::BasicBlock* norm_cond = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_cond", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "sm_norm_body", current_func);
    llvm::BasicBlock* row_next = llvm::BasicBlock::Create(ctx_.context(), "row_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), col_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(norm_cond);
    llvm::Value* col3 = builder.CreateLoad(ctx_.int64Type(), col_idx);
    llvm::Value* col3_done = builder.CreateICmpUGE(col3, seq_k);
    builder.CreateCondBr(col3_done, row_next, norm_body);

    builder.SetInsertPoint(norm_body);
    llvm::Value* score_idx3 = builder.CreateAdd(row_offset, col3);
    llvm::Value* score_ptr3 = builder.CreateGEP(ctx_.doubleType(), scores_ptr, score_idx3);
    llvm::Value* exp_val3 = builder.CreateLoad(ctx_.doubleType(), score_ptr3);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    llvm::Value* normalized = builder.CreateFDiv(exp_val3, total_sum);
    builder.CreateStore(normalized, score_ptr3);
    builder.CreateStore(builder.CreateAdd(col3,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), col_idx);
    builder.CreateBr(norm_cond);

    builder.SetInsertPoint(row_next);
    builder.CreateStore(builder.CreateAdd(row,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), row_idx);
    builder.CreateBr(sm_row_cond);

    builder.SetInsertPoint(sm_batch_next);
    builder.CreateStore(builder.CreateAdd(b2,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(sm_batch_cond);

    // === Compute attention_weights @ V ===
    builder.SetInsertPoint(output_init);

    llvm::BasicBlock* out_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "out_batch_cond", current_func);
    llvm::BasicBlock* out_batch_body = llvm::BasicBlock::Create(ctx_.context(), "out_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "attn_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(out_batch_cond);

    builder.SetInsertPoint(out_batch_cond);
    llvm::Value* b3 = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b3_done = builder.CreateICmpUGE(b3, batch_size);
    builder.CreateCondBr(b3_done, finalize, out_batch_body);

    builder.SetInsertPoint(out_batch_body);

    llvm::Value* attn_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_q, seq_k));
    llvm::Value* v_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_k, d_v));
    llvm::Value* out_batch_offset = builder.CreateMul(b3, builder.CreateMul(seq_q, d_v));

    // output[b, i, j] = sum_k(attention[b, i, k] * V[b, k, j])
    llvm::BasicBlock* out_i_cond = llvm::BasicBlock::Create(ctx_.context(), "out_i_cond", current_func);
    llvm::BasicBlock* out_i_body = llvm::BasicBlock::Create(ctx_.context(), "out_i_body", current_func);
    llvm::BasicBlock* out_batch_next = llvm::BasicBlock::Create(ctx_.context(), "out_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_i_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* oi_done = builder.CreateICmpUGE(oi, seq_q);
    builder.CreateCondBr(oi_done, out_batch_next, out_i_body);

    builder.SetInsertPoint(out_i_body);

    // j loop (d_v)
    llvm::BasicBlock* out_j_cond = llvm::BasicBlock::Create(ctx_.context(), "out_j_cond", current_func);
    llvm::BasicBlock* out_j_body = llvm::BasicBlock::Create(ctx_.context(), "out_j_body", current_func);
    llvm::BasicBlock* out_i_next = llvm::BasicBlock::Create(ctx_.context(), "out_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_idx);
    builder.CreateBr(out_j_cond);

    builder.SetInsertPoint(out_j_cond);
    llvm::Value* oj = builder.CreateLoad(ctx_.int64Type(), j_idx);
    llvm::Value* oj_done = builder.CreateICmpUGE(oj, d_v);
    builder.CreateCondBr(oj_done, out_i_next, out_j_body);

    builder.SetInsertPoint(out_j_body);

    // k loop (sum over seq_k)
    llvm::BasicBlock* out_k_cond = llvm::BasicBlock::Create(ctx_.context(), "out_k_cond", current_func);
    llvm::BasicBlock* out_k_body = llvm::BasicBlock::Create(ctx_.context(), "out_k_body", current_func);
    llvm::BasicBlock* out_k_done = llvm::BasicBlock::Create(ctx_.context(), "out_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), out_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_iter);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_cond);
    llvm::Value* ok = builder.CreateLoad(ctx_.int64Type(), k_iter);
    llvm::Value* ok_done = builder.CreateICmpUGE(ok, seq_k);
    builder.CreateCondBr(ok_done, out_k_done, out_k_body);

    builder.SetInsertPoint(out_k_body);
    // attention[b, i, k]
    llvm::Value* attn_idx = builder.CreateAdd(attn_batch_offset,
        builder.CreateAdd(builder.CreateMul(oi, seq_k), ok));
    llvm::Value* attn_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), scores_ptr, attn_idx));
    // V[b, k, j]
    llvm::Value* v_idx = builder.CreateAdd(v_batch_offset,
        builder.CreateAdd(builder.CreateMul(ok, d_v), oj));
    llvm::Value* v_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), v_elems, v_idx));

    llvm::Value* prod2 = builder.CreateFMul(attn_val, v_elem);
    llvm::Value* cur_out_sum = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(builder.CreateFAdd(cur_out_sum, prod2), out_sum);

    builder.CreateStore(builder.CreateAdd(ok,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_iter);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_done);
    // Store output[b, i, j]
    llvm::Value* out_idx = builder.CreateAdd(out_batch_offset,
        builder.CreateAdd(builder.CreateMul(oi, d_v), oj));
    llvm::Value* final_out = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(final_out,
        builder.CreateGEP(ctx_.doubleType(), output_elems, out_idx));

    builder.CreateStore(builder.CreateAdd(oj,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_idx);
    builder.CreateBr(out_j_cond);

    builder.SetInsertPoint(out_i_next);
    builder.CreateStore(builder.CreateAdd(oi,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_batch_next);
    builder.CreateStore(builder.CreateAdd(b3,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(out_batch_cond);

    // === Finalize result tensor ===
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "attn_result");

    // Allocate dims array - same shape as output: (batch, seq_q, d_v) or (seq_q, d_v)
    llvm::Value* dims_count = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 3),
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* dims_bytes = builder.CreateMul(dims_count,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "result_dims");

    // Store dimensions based on 2D or 3D
    llvm::BasicBlock* store_3d = llvm::BasicBlock::Create(ctx_.context(), "store_3d", current_func);
    llvm::BasicBlock* store_2d = llvm::BasicBlock::Create(ctx_.context(), "store_2d", current_func);
    llvm::BasicBlock* store_done = llvm::BasicBlock::Create(ctx_.context(), "store_done", current_func);

    builder.CreateCondBr(is_3d, store_3d, store_2d);

    builder.SetInsertPoint(store_3d);
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(d_v, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    builder.CreateBr(store_done);

    builder.SetInsertPoint(store_2d);
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(d_v, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateBr(store_done);

    builder.SetInsertPoint(store_done);

    // Populate tensor struct
    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(dims_count, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// === Track 8.2: Multi-Head Attention ===

llvm::Value* TensorCodegen::multiHeadAttention(const eshkol_operations_t* op) {
    // Multi-Head Attention
    // 1. Project Q, K, V through weight matrices
    // 2. Split into num_heads
    // 3. Apply scaled dot-product attention to each head
    // 4. Concatenate heads
    // 5. Project through output weights
    //
    // Args: Q K V num-heads W_Q W_K W_V W_O [mask]

    if (op->call_op.num_vars < 8 || op->call_op.num_vars > 9) {
        eshkol_error("multi-head-attention requires 8-9 arguments: Q K V num-heads W_Q W_K W_V W_O [mask]");
        return nullptr;
    }

    llvm::Value* q_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* k_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* v_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* num_heads_val = codegenAST(&op->call_op.variables[3]);
    llvm::Value* wq_val = codegenAST(&op->call_op.variables[4]);
    llvm::Value* wk_val = codegenAST(&op->call_op.variables[5]);
    llvm::Value* wv_val = codegenAST(&op->call_op.variables[6]);
    llvm::Value* wo_val = codegenAST(&op->call_op.variables[7]);
    llvm::Value* mask_val = (op->call_op.num_vars == 9)
        ? codegenAST(&op->call_op.variables[8]) : nullptr;

    if (!q_val || !k_val || !v_val || !num_heads_val ||
        !wq_val || !wk_val || !wv_val || !wo_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Get num_heads as integer
    llvm::Value* num_heads = tagged_.unpackInt64(num_heads_val);

    // Unpack Q tensor to get dimensions
    llvm::Value* q_ptr_int = tagged_.unpackInt64(q_val);
    llvm::Value* q_ptr = builder.CreateIntToPtr(q_ptr_int, ctx_.ptrType());

    llvm::Value* q_dims_field = builder.CreateStructGEP(tensor_type, q_ptr, 0);
    llvm::Value* q_dims_ptr = builder.CreateLoad(ctx_.ptrType(), q_dims_field);
    llvm::Value* q_ndim_field = builder.CreateStructGEP(tensor_type, q_ptr, 1);
    llvm::Value* q_ndim = builder.CreateLoad(ctx_.int64Type(), q_ndim_field);
    llvm::Value* q_elems_field = builder.CreateStructGEP(tensor_type, q_ptr, 2);
    llvm::Value* q_elems = builder.CreateLoad(ctx_.ptrType(), q_elems_field);

    // Unpack K tensor
    llvm::Value* k_ptr_int = tagged_.unpackInt64(k_val);
    llvm::Value* k_ptr = builder.CreateIntToPtr(k_ptr_int, ctx_.ptrType());
    llvm::Value* k_dims_field = builder.CreateStructGEP(tensor_type, k_ptr, 0);
    llvm::Value* k_dims_ptr = builder.CreateLoad(ctx_.ptrType(), k_dims_field);
    llvm::Value* k_elems_field = builder.CreateStructGEP(tensor_type, k_ptr, 2);
    llvm::Value* k_elems = builder.CreateLoad(ctx_.ptrType(), k_elems_field);

    // Unpack V tensor
    llvm::Value* v_ptr_int = tagged_.unpackInt64(v_val);
    llvm::Value* v_ptr = builder.CreateIntToPtr(v_ptr_int, ctx_.ptrType());
    llvm::Value* v_dims_field = builder.CreateStructGEP(tensor_type, v_ptr, 0);
    llvm::Value* v_dims_ptr = builder.CreateLoad(ctx_.ptrType(), v_dims_field);
    llvm::Value* v_elems_field = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_field);

    // Unpack weight matrices
    llvm::Value* wq_ptr_int = tagged_.unpackInt64(wq_val);
    llvm::Value* wq_ptr = builder.CreateIntToPtr(wq_ptr_int, ctx_.ptrType());
    llvm::Value* wq_elems_field = builder.CreateStructGEP(tensor_type, wq_ptr, 2);
    llvm::Value* wq_elems = builder.CreateLoad(ctx_.ptrType(), wq_elems_field);

    llvm::Value* wk_ptr_int = tagged_.unpackInt64(wk_val);
    llvm::Value* wk_ptr = builder.CreateIntToPtr(wk_ptr_int, ctx_.ptrType());
    llvm::Value* wk_elems_field = builder.CreateStructGEP(tensor_type, wk_ptr, 2);
    llvm::Value* wk_elems = builder.CreateLoad(ctx_.ptrType(), wk_elems_field);

    llvm::Value* wv_ptr_int = tagged_.unpackInt64(wv_val);
    llvm::Value* wv_ptr = builder.CreateIntToPtr(wv_ptr_int, ctx_.ptrType());
    llvm::Value* wv_elems_field = builder.CreateStructGEP(tensor_type, wv_ptr, 2);
    llvm::Value* wv_elems = builder.CreateLoad(ctx_.ptrType(), wv_elems_field);

    llvm::Value* wo_ptr_int = tagged_.unpackInt64(wo_val);
    llvm::Value* wo_ptr = builder.CreateIntToPtr(wo_ptr_int, ctx_.ptrType());
    llvm::Value* wo_elems_field = builder.CreateStructGEP(tensor_type, wo_ptr, 2);
    llvm::Value* wo_elems = builder.CreateLoad(ctx_.ptrType(), wo_elems_field);

    // Handle mask if provided
    llvm::Value* mask_elems = nullptr;
    if (mask_val) {
        llvm::Value* mask_ptr_int = tagged_.unpackInt64(mask_val);
        llvm::Value* mask_ptr = builder.CreateIntToPtr(mask_ptr_int, ctx_.ptrType());
        llvm::Value* mask_elems_field = builder.CreateStructGEP(tensor_type, mask_ptr, 2);
        mask_elems = builder.CreateLoad(ctx_.ptrType(), mask_elems_field);
    }

    // Determine dimensions
    llvm::Value* is_3d = builder.CreateICmpEQ(q_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), q_dims_ptr,
                llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* dim_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_q = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, seq_idx));
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), q_dims_ptr, dim_idx));

    llvm::Value* seq_k = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), k_dims_ptr, seq_idx));

    // d_k = d_model / num_heads
    llvm::Value* d_k = builder.CreateSDiv(d_model, num_heads, "d_k");

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate projected Q, K, V: (batch, seq, d_model)
    llvm::Value* proj_size = builder.CreateMul(batch_size,
        builder.CreateMul(seq_q, d_model));
    llvm::Value* proj_bytes = builder.CreateMul(proj_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* proj_q = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_q");
    llvm::Value* proj_k = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_k");
    llvm::Value* proj_v = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "proj_v");

    // Allocate per-head attention outputs: (batch, num_heads, seq_q, d_k)
    llvm::Value* head_out_size = builder.CreateMul(batch_size,
        builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k)));
    llvm::Value* head_out_bytes = builder.CreateMul(head_out_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* head_outputs = builder.CreateCall(arena_alloc,
        {arena_ptr, head_out_bytes}, "head_outputs");

    // Allocate final output: (batch, seq_q, d_model)
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, proj_bytes}, "mha_output");

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* seq_idx_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "seq_idx");
    llvm::Value* d_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    llvm::Value* m_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "m_idx");
    llvm::Value* q_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "q_sum");
    llvm::Value* k_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "k_sum");
    llvm::Value* v_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "v_sum");
    llvm::Value* head_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "head_idx");
    llvm::Value* i_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_var");
    llvm::Value* j_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "j_var");
    llvm::Value* k_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_var");
    llvm::Value* score_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "score_sum");
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType(), nullptr, "max_val");
    llvm::Value* sum_exp = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum_exp");
    llvm::Value* out_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "out_sum");
    llvm::Value* final_sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "final_sum");

    // === Step 1: Project Q, K, V ===
    // proj_Q = Q @ W_Q (matmul for each position)
    // For each batch b, seq position s: proj_Q[b,s,:] = Q[b,s,:] @ W_Q

    llvm::BasicBlock* proj_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_cond", current_func);
    llvm::BasicBlock* proj_batch_body = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_body", current_func);
    llvm::BasicBlock* heads_init = llvm::BasicBlock::Create(ctx_.context(), "heads_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(proj_batch_cond);

    builder.SetInsertPoint(proj_batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, heads_init, proj_batch_body);

    builder.SetInsertPoint(proj_batch_body);

    llvm::Value* batch_offset = builder.CreateMul(b, builder.CreateMul(seq_q, d_model));
    llvm::Value* k_batch_offset = builder.CreateMul(b, builder.CreateMul(seq_k, d_model));

    // Project each sequence position
    llvm::BasicBlock* proj_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_cond", current_func);
    llvm::BasicBlock* proj_seq_body = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_body", current_func);
    llvm::BasicBlock* proj_batch_next = llvm::BasicBlock::Create(ctx_.context(), "proj_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), seq_idx_var);
    builder.CreateBr(proj_seq_cond);

    builder.SetInsertPoint(proj_seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), seq_idx_var);
    llvm::Value* s_done = builder.CreateICmpUGE(s, seq_q);
    builder.CreateCondBr(s_done, proj_batch_next, proj_seq_body);

    builder.SetInsertPoint(proj_seq_body);

    llvm::Value* pos_offset = builder.CreateAdd(batch_offset, builder.CreateMul(s, d_model));

    // For each output dimension d
    llvm::BasicBlock* proj_d_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_d_cond", current_func);
    llvm::BasicBlock* proj_d_body = llvm::BasicBlock::Create(ctx_.context(), "proj_d_body", current_func);
    llvm::BasicBlock* proj_seq_next = llvm::BasicBlock::Create(ctx_.context(), "proj_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(proj_d_cond);

    builder.SetInsertPoint(proj_d_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* d_done = builder.CreateICmpUGE(d, d_model);
    builder.CreateCondBr(d_done, proj_seq_next, proj_d_body);

    builder.SetInsertPoint(proj_d_body);

    // Compute dot product: input[s,:] @ W[:,d]
    llvm::BasicBlock* dot_cond = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_cond", current_func);
    llvm::BasicBlock* dot_body = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_body", current_func);
    llvm::BasicBlock* dot_done = llvm::BasicBlock::Create(ctx_.context(), "proj_dot_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), q_sum);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), k_sum);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), v_sum);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), m_idx);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_cond);
    llvm::Value* m = builder.CreateLoad(ctx_.int64Type(), m_idx);
    llvm::Value* m_done = builder.CreateICmpUGE(m, d_model);
    builder.CreateCondBr(m_done, dot_done, dot_body);

    builder.SetInsertPoint(dot_body);
    // input element: Q[b, s, m], K[b, s, m], V[b, s, m]
    llvm::Value* in_idx = builder.CreateAdd(pos_offset, m);
    llvm::Value* q_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), q_elems, in_idx));
    llvm::Value* k_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), k_elems, in_idx));
    llvm::Value* v_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), v_elems, in_idx));

    // Weight element: W[m, d]
    llvm::Value* w_idx = builder.CreateAdd(builder.CreateMul(m, d_model), d);
    llvm::Value* wq_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wq_elems, w_idx));
    llvm::Value* wk_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wk_elems, w_idx));
    llvm::Value* wv_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wv_elems, w_idx));

    // Accumulate
    llvm::Value* cur_q = builder.CreateLoad(ctx_.doubleType(), q_sum);
    builder.CreateStore(builder.CreateFAdd(cur_q, builder.CreateFMul(q_elem, wq_elem)), q_sum);
    llvm::Value* cur_k = builder.CreateLoad(ctx_.doubleType(), k_sum);
    builder.CreateStore(builder.CreateFAdd(cur_k, builder.CreateFMul(k_elem, wk_elem)), k_sum);
    llvm::Value* cur_v = builder.CreateLoad(ctx_.doubleType(), v_sum);
    builder.CreateStore(builder.CreateFAdd(cur_v, builder.CreateFMul(v_elem, wv_elem)), v_sum);

    builder.CreateStore(builder.CreateAdd(m, llvm::ConstantInt::get(ctx_.int64Type(), 1)), m_idx);
    builder.CreateBr(dot_cond);

    builder.SetInsertPoint(dot_done);
    // Store projected values
    llvm::Value* out_idx = builder.CreateAdd(pos_offset, d);
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), q_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_q, out_idx));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), k_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_k, out_idx));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), v_sum),
        builder.CreateGEP(ctx_.doubleType(), proj_v, out_idx));

    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(proj_d_cond);

    builder.SetInsertPoint(proj_seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), seq_idx_var);
    builder.CreateBr(proj_seq_cond);

    builder.SetInsertPoint(proj_batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(proj_batch_cond);

    // === Step 2-3: Split into heads and apply attention ===
    builder.SetInsertPoint(heads_init);

    // For efficiency, we compute attention for all heads in parallel
    // Each head h gets: Q[:, :, h*d_k:(h+1)*d_k], K[:, :, h*d_k:(h+1)*d_k], V[:, :, h*d_k:(h+1)*d_k]

    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Value* d_k_double = builder.CreateSIToFP(d_k, ctx_.doubleType());
    llvm::Value* d_k_safe_mh = builder.CreateSelect(
        builder.CreateFCmpOLE(d_k_double, llvm::ConstantFP::get(ctx_.doubleType(), 0.0)),
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0), d_k_double, "dk_safe_mh");
    llvm::Value* scale = builder.CreateCall(sqrt_func, {d_k_safe_mh}, "scale");

    // Allocate attention scores for one head: (seq_q, seq_k)
    llvm::Value* scores_size = builder.CreateMul(seq_q, seq_k);
    llvm::Value* scores_bytes = builder.CreateMul(scores_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* attn_scores = builder.CreateCall(arena_alloc,
        {arena_ptr, scores_bytes}, "attn_scores");

    // Loop over batches and heads
    llvm::BasicBlock* head_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "head_batch_cond", current_func);
    llvm::BasicBlock* head_batch_body = llvm::BasicBlock::Create(ctx_.context(), "head_batch_body", current_func);
    llvm::BasicBlock* concat_init = llvm::BasicBlock::Create(ctx_.context(), "concat_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(head_batch_cond);

    builder.SetInsertPoint(head_batch_cond);
    llvm::Value* hb = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* hb_done = builder.CreateICmpUGE(hb, batch_size);
    builder.CreateCondBr(hb_done, concat_init, head_batch_body);

    builder.SetInsertPoint(head_batch_body);

    llvm::Value* hb_offset = builder.CreateMul(hb, builder.CreateMul(seq_q, d_model));

    // Loop over heads
    llvm::BasicBlock* head_cond = llvm::BasicBlock::Create(ctx_.context(), "head_cond", current_func);
    llvm::BasicBlock* head_body = llvm::BasicBlock::Create(ctx_.context(), "head_body", current_func);
    llvm::BasicBlock* head_batch_next = llvm::BasicBlock::Create(ctx_.context(), "head_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), head_idx);
    builder.CreateBr(head_cond);

    builder.SetInsertPoint(head_cond);
    llvm::Value* h = builder.CreateLoad(ctx_.int64Type(), head_idx);
    llvm::Value* h_done = builder.CreateICmpUGE(h, num_heads);
    builder.CreateCondBr(h_done, head_batch_next, head_body);

    builder.SetInsertPoint(head_body);

    llvm::Value* head_start = builder.CreateMul(h, d_k);  // Starting dimension for this head

    // Compute attention scores: Q_h @ K_h^T / sqrt(d_k)
    // scores[i, j] = sum_k(Q[i, h*d_k + k] * K[j, h*d_k + k]) / sqrt(d_k)

    llvm::BasicBlock* score_i_cond = llvm::BasicBlock::Create(ctx_.context(), "score_i_cond", current_func);
    llvm::BasicBlock* score_i_body = llvm::BasicBlock::Create(ctx_.context(), "score_i_body", current_func);
    llvm::BasicBlock* softmax_row = llvm::BasicBlock::Create(ctx_.context(), "softmax_row", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(score_i_cond);

    builder.SetInsertPoint(score_i_cond);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* si_done = builder.CreateICmpUGE(si, seq_q);
    builder.CreateCondBr(si_done, softmax_row, score_i_body);

    builder.SetInsertPoint(score_i_body);

    llvm::BasicBlock* score_j_cond = llvm::BasicBlock::Create(ctx_.context(), "score_j_cond", current_func);
    llvm::BasicBlock* score_j_body = llvm::BasicBlock::Create(ctx_.context(), "score_j_body", current_func);
    llvm::BasicBlock* score_i_next = llvm::BasicBlock::Create(ctx_.context(), "score_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(score_j_cond);

    builder.SetInsertPoint(score_j_cond);
    llvm::Value* sj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* sj_done = builder.CreateICmpUGE(sj, seq_k);
    builder.CreateCondBr(sj_done, score_i_next, score_j_body);

    builder.SetInsertPoint(score_j_body);

    // Dot product over d_k dimensions
    llvm::BasicBlock* score_k_cond = llvm::BasicBlock::Create(ctx_.context(), "score_k_cond", current_func);
    llvm::BasicBlock* score_k_body = llvm::BasicBlock::Create(ctx_.context(), "score_k_body", current_func);
    llvm::BasicBlock* score_k_done = llvm::BasicBlock::Create(ctx_.context(), "score_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), score_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(score_k_cond);

    builder.SetInsertPoint(score_k_cond);
    llvm::Value* sk = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* sk_done = builder.CreateICmpUGE(sk, d_k);
    builder.CreateCondBr(sk_done, score_k_done, score_k_body);

    builder.SetInsertPoint(score_k_body);
    // Q[b, i, head_start + k]
    llvm::Value* q_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(si, d_model),
            builder.CreateAdd(head_start, sk)));
    llvm::Value* q_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_q, q_idx));
    // K[b, j, head_start + k]
    llvm::Value* k_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(sj, d_model),
            builder.CreateAdd(head_start, sk)));
    llvm::Value* k_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_k, k_idx));

    llvm::Value* prod = builder.CreateFMul(q_val_elem, k_val_elem);
    llvm::Value* cur_score = builder.CreateLoad(ctx_.doubleType(), score_sum);
    builder.CreateStore(builder.CreateFAdd(cur_score, prod), score_sum);

    builder.CreateStore(builder.CreateAdd(sk, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(score_k_cond);

    builder.SetInsertPoint(score_k_done);
    llvm::Value* final_score = builder.CreateLoad(ctx_.doubleType(), score_sum);
    llvm::Value* scaled_score = builder.CreateFDiv(final_score, scale);

    // Add mask if provided
    if (mask_elems) {
        llvm::Value* mask_idx = builder.CreateAdd(builder.CreateMul(si, seq_k), sj);
        llvm::Value* mask_val_elem = builder.CreateLoad(ctx_.doubleType(),
            builder.CreateGEP(ctx_.doubleType(), mask_elems, mask_idx));
        scaled_score = builder.CreateFAdd(scaled_score, mask_val_elem);
    }

    // Store score
    llvm::Value* scores_idx = builder.CreateAdd(builder.CreateMul(si, seq_k), sj);
    builder.CreateStore(scaled_score,
        builder.CreateGEP(ctx_.doubleType(), attn_scores, scores_idx));

    builder.CreateStore(builder.CreateAdd(sj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(score_j_cond);

    builder.SetInsertPoint(score_i_next);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(score_i_cond);

    // Apply softmax to each row
    builder.SetInsertPoint(softmax_row);

    llvm::BasicBlock* sm_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_cond", current_func);
    llvm::BasicBlock* sm_i_body = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_body", current_func);
    llvm::BasicBlock* attn_output = llvm::BasicBlock::Create(ctx_.context(), "attn_output", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(sm_i_cond);

    builder.SetInsertPoint(sm_i_cond);
    llvm::Value* smi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* smi_done = builder.CreateICmpUGE(smi, seq_q);
    builder.CreateCondBr(smi_done, attn_output, sm_i_body);

    builder.SetInsertPoint(sm_i_body);

    llvm::Value* row_offset = builder.CreateMul(smi, seq_k);

    // Find max
    llvm::Value* first_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, row_offset));
    builder.CreateStore(first_val, max_val);

    llvm::BasicBlock* max_j_cond = llvm::BasicBlock::Create(ctx_.context(), "max_j_cond", current_func);
    llvm::BasicBlock* max_j_body = llvm::BasicBlock::Create(ctx_.context(), "max_j_body", current_func);
    llvm::BasicBlock* exp_j_init = llvm::BasicBlock::Create(ctx_.context(), "exp_j_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), j_var);
    builder.CreateBr(max_j_cond);

    builder.SetInsertPoint(max_j_cond);
    llvm::Value* mj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* mj_done = builder.CreateICmpUGE(mj, seq_k);
    builder.CreateCondBr(mj_done, exp_j_init, max_j_body);

    builder.SetInsertPoint(max_j_body);
    llvm::Value* elem_idx = builder.CreateAdd(row_offset, mj);
    llvm::Value* elem_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, elem_idx));
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_gt = builder.CreateFCmpOGT(elem_val, cur_max);
    builder.CreateStore(builder.CreateSelect(is_gt, elem_val, cur_max), max_val);
    builder.CreateStore(builder.CreateAdd(mj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(max_j_cond);

    // Compute exp and sum
    builder.SetInsertPoint(exp_j_init);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_exp);

    llvm::BasicBlock* exp_j_cond = llvm::BasicBlock::Create(ctx_.context(), "exp_j_cond", current_func);
    llvm::BasicBlock* exp_j_body = llvm::BasicBlock::Create(ctx_.context(), "exp_j_body", current_func);
    llvm::BasicBlock* norm_j_init = llvm::BasicBlock::Create(ctx_.context(), "norm_j_init", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(exp_j_cond);

    builder.SetInsertPoint(exp_j_cond);
    llvm::Value* ej = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* ej_done = builder.CreateICmpUGE(ej, seq_k);
    builder.CreateCondBr(ej_done, norm_j_init, exp_j_body);

    builder.SetInsertPoint(exp_j_body);
    llvm::Value* exp_idx = builder.CreateAdd(row_offset, ej);
    llvm::Value* exp_ptr = builder.CreateGEP(ctx_.doubleType(), attn_scores, exp_idx);
    llvm::Value* exp_input = builder.CreateLoad(ctx_.doubleType(), exp_ptr);
    llvm::Value* final_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* shifted = builder.CreateFSub(exp_input, final_max);
    llvm::Value* exp_out = builder.CreateCall(exp_func, {shifted});
    builder.CreateStore(exp_out, exp_ptr);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFAdd(cur_sum, exp_out), sum_exp);
    builder.CreateStore(builder.CreateAdd(ej, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(exp_j_cond);

    // Normalize
    builder.SetInsertPoint(norm_j_init);
    llvm::BasicBlock* norm_j_cond = llvm::BasicBlock::Create(ctx_.context(), "norm_j_cond", current_func);
    llvm::BasicBlock* norm_j_body = llvm::BasicBlock::Create(ctx_.context(), "norm_j_body", current_func);
    llvm::BasicBlock* sm_i_next = llvm::BasicBlock::Create(ctx_.context(), "mha_sm_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_var);
    builder.CreateBr(norm_j_cond);

    builder.SetInsertPoint(norm_j_cond);
    llvm::Value* nj = builder.CreateLoad(ctx_.int64Type(), j_var);
    llvm::Value* nj_done = builder.CreateICmpUGE(nj, seq_k);
    builder.CreateCondBr(nj_done, sm_i_next, norm_j_body);

    builder.SetInsertPoint(norm_j_body);
    llvm::Value* norm_idx = builder.CreateAdd(row_offset, nj);
    llvm::Value* norm_ptr = builder.CreateGEP(ctx_.doubleType(), attn_scores, norm_idx);
    llvm::Value* norm_val = builder.CreateLoad(ctx_.doubleType(), norm_ptr);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_exp);
    builder.CreateStore(builder.CreateFDiv(norm_val, total_sum), norm_ptr);
    builder.CreateStore(builder.CreateAdd(nj, llvm::ConstantInt::get(ctx_.int64Type(), 1)), j_var);
    builder.CreateBr(norm_j_cond);

    builder.SetInsertPoint(sm_i_next);
    builder.CreateStore(builder.CreateAdd(smi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(sm_i_cond);

    // Compute attention output: attn_weights @ V_h
    builder.SetInsertPoint(attn_output);

    llvm::BasicBlock* out_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_cond", current_func);
    llvm::BasicBlock* out_i_body = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_body", current_func);
    llvm::BasicBlock* head_next = llvm::BasicBlock::Create(ctx_.context(), "head_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(out_i_cond);
    llvm::Value* oi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* oi_done = builder.CreateICmpUGE(oi, seq_q);
    builder.CreateCondBr(oi_done, head_next, out_i_body);

    builder.SetInsertPoint(out_i_body);

    // For each output dimension in this head
    llvm::BasicBlock* out_d_cond = llvm::BasicBlock::Create(ctx_.context(), "out_d_cond", current_func);
    llvm::BasicBlock* out_d_body = llvm::BasicBlock::Create(ctx_.context(), "out_d_body", current_func);
    llvm::BasicBlock* out_i_next = llvm::BasicBlock::Create(ctx_.context(), "mha_out_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(out_d_cond);

    builder.SetInsertPoint(out_d_cond);
    llvm::Value* od = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* od_done = builder.CreateICmpUGE(od, d_k);
    builder.CreateCondBr(od_done, out_i_next, out_d_body);

    builder.SetInsertPoint(out_d_body);

    // Sum over seq_k: attn[i, k] * V[k, head_start + d]
    llvm::BasicBlock* out_k_cond = llvm::BasicBlock::Create(ctx_.context(), "out_k_cond", current_func);
    llvm::BasicBlock* out_k_body = llvm::BasicBlock::Create(ctx_.context(), "out_k_body", current_func);
    llvm::BasicBlock* out_k_done = llvm::BasicBlock::Create(ctx_.context(), "out_k_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), out_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_cond);
    llvm::Value* ok = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* ok_done = builder.CreateICmpUGE(ok, seq_k);
    builder.CreateCondBr(ok_done, out_k_done, out_k_body);

    builder.SetInsertPoint(out_k_body);
    // attn[i, k]
    llvm::Value* attn_idx = builder.CreateAdd(builder.CreateMul(oi, seq_k), ok);
    llvm::Value* attn_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), attn_scores, attn_idx));
    // V[b, k, head_start + d]
    llvm::Value* v_idx = builder.CreateAdd(hb_offset,
        builder.CreateAdd(builder.CreateMul(ok, d_model),
            builder.CreateAdd(head_start, od)));
    llvm::Value* v_val_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), proj_v, v_idx));

    llvm::Value* prod_out = builder.CreateFMul(attn_val, v_val_elem);
    llvm::Value* cur_out = builder.CreateLoad(ctx_.doubleType(), out_sum);
    builder.CreateStore(builder.CreateFAdd(cur_out, prod_out), out_sum);

    builder.CreateStore(builder.CreateAdd(ok, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(out_k_cond);

    builder.SetInsertPoint(out_k_done);
    // Store in head_outputs[b, h, i, d]
    llvm::Value* ho_idx = builder.CreateAdd(
        builder.CreateMul(hb, builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k))),
        builder.CreateAdd(
            builder.CreateMul(h, builder.CreateMul(seq_q, d_k)),
            builder.CreateAdd(builder.CreateMul(oi, d_k), od)));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), out_sum),
        builder.CreateGEP(ctx_.doubleType(), head_outputs, ho_idx));

    builder.CreateStore(builder.CreateAdd(od, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(out_d_cond);

    builder.SetInsertPoint(out_i_next);
    builder.CreateStore(builder.CreateAdd(oi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(out_i_cond);

    builder.SetInsertPoint(head_next);
    builder.CreateStore(builder.CreateAdd(h, llvm::ConstantInt::get(ctx_.int64Type(), 1)), head_idx);
    builder.CreateBr(head_cond);

    builder.SetInsertPoint(head_batch_next);
    builder.CreateStore(builder.CreateAdd(hb, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(head_batch_cond);

    // === Step 4-5: Concatenate heads and project through W_O ===
    builder.SetInsertPoint(concat_init);

    // The concatenation is implicit in head_outputs layout
    // Now we project: output = concat(heads) @ W_O
    // For each position, we compute: output[:, i, d] = sum_h sum_k head_outputs[:, h, i, k] * W_O[h*d_k + k, d]

    llvm::BasicBlock* final_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "final_batch_cond", current_func);
    llvm::BasicBlock* final_batch_body = llvm::BasicBlock::Create(ctx_.context(), "final_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "mha_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(final_batch_cond);

    builder.SetInsertPoint(final_batch_cond);
    llvm::Value* fb = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* fb_done = builder.CreateICmpUGE(fb, batch_size);
    builder.CreateCondBr(fb_done, finalize, final_batch_body);

    builder.SetInsertPoint(final_batch_body);

    llvm::Value* fb_offset = builder.CreateMul(fb, builder.CreateMul(seq_q, d_model));
    llvm::Value* fb_head_offset = builder.CreateMul(fb,
        builder.CreateMul(num_heads, builder.CreateMul(seq_q, d_k)));

    llvm::BasicBlock* final_i_cond = llvm::BasicBlock::Create(ctx_.context(), "final_i_cond", current_func);
    llvm::BasicBlock* final_i_body = llvm::BasicBlock::Create(ctx_.context(), "final_i_body", current_func);
    llvm::BasicBlock* final_batch_next = llvm::BasicBlock::Create(ctx_.context(), "final_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_var);
    builder.CreateBr(final_i_cond);

    builder.SetInsertPoint(final_i_cond);
    llvm::Value* fi = builder.CreateLoad(ctx_.int64Type(), i_var);
    llvm::Value* fi_done = builder.CreateICmpUGE(fi, seq_q);
    builder.CreateCondBr(fi_done, final_batch_next, final_i_body);

    builder.SetInsertPoint(final_i_body);

    // For each output dimension
    llvm::BasicBlock* final_d_cond = llvm::BasicBlock::Create(ctx_.context(), "final_d_cond", current_func);
    llvm::BasicBlock* final_d_body = llvm::BasicBlock::Create(ctx_.context(), "final_d_body", current_func);
    llvm::BasicBlock* final_i_next = llvm::BasicBlock::Create(ctx_.context(), "final_i_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(final_d_cond);

    builder.SetInsertPoint(final_d_cond);
    llvm::Value* fd = builder.CreateLoad(ctx_.int64Type(), d_idx);
    llvm::Value* fd_done = builder.CreateICmpUGE(fd, d_model);
    builder.CreateCondBr(fd_done, final_i_next, final_d_body);

    builder.SetInsertPoint(final_d_body);

    // Sum over all heads and their dimensions
    llvm::BasicBlock* final_h_cond = llvm::BasicBlock::Create(ctx_.context(), "final_h_cond", current_func);
    llvm::BasicBlock* final_h_body = llvm::BasicBlock::Create(ctx_.context(), "final_h_body", current_func);
    llvm::BasicBlock* final_d_done = llvm::BasicBlock::Create(ctx_.context(), "final_d_done", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), final_sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), head_idx);
    builder.CreateBr(final_h_cond);

    builder.SetInsertPoint(final_h_cond);
    llvm::Value* fh = builder.CreateLoad(ctx_.int64Type(), head_idx);
    llvm::Value* fh_done = builder.CreateICmpUGE(fh, num_heads);
    builder.CreateCondBr(fh_done, final_d_done, final_h_body);

    builder.SetInsertPoint(final_h_body);

    llvm::Value* fh_start = builder.CreateMul(fh, d_k);

    // Sum over d_k
    llvm::BasicBlock* final_k_cond = llvm::BasicBlock::Create(ctx_.context(), "final_k_cond", current_func);
    llvm::BasicBlock* final_k_body = llvm::BasicBlock::Create(ctx_.context(), "final_k_body", current_func);
    llvm::BasicBlock* final_h_next = llvm::BasicBlock::Create(ctx_.context(), "final_h_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_var);
    builder.CreateBr(final_k_cond);

    builder.SetInsertPoint(final_k_cond);
    llvm::Value* fk = builder.CreateLoad(ctx_.int64Type(), k_var);
    llvm::Value* fk_done = builder.CreateICmpUGE(fk, d_k);
    builder.CreateCondBr(fk_done, final_h_next, final_k_body);

    builder.SetInsertPoint(final_k_body);
    // head_outputs[b, h, i, k]
    llvm::Value* ho_read_idx = builder.CreateAdd(fb_head_offset,
        builder.CreateAdd(builder.CreateMul(fh, builder.CreateMul(seq_q, d_k)),
            builder.CreateAdd(builder.CreateMul(fi, d_k), fk)));
    llvm::Value* ho_val = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), head_outputs, ho_read_idx));
    // W_O[h*d_k + k, d]
    llvm::Value* wo_idx = builder.CreateAdd(
        builder.CreateMul(builder.CreateAdd(fh_start, fk), d_model), fd);
    llvm::Value* wo_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), wo_elems, wo_idx));

    llvm::Value* prod_final = builder.CreateFMul(ho_val, wo_elem);
    llvm::Value* cur_final = builder.CreateLoad(ctx_.doubleType(), final_sum);
    builder.CreateStore(builder.CreateFAdd(cur_final, prod_final), final_sum);

    builder.CreateStore(builder.CreateAdd(fk, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_var);
    builder.CreateBr(final_k_cond);

    builder.SetInsertPoint(final_h_next);
    builder.CreateStore(builder.CreateAdd(fh, llvm::ConstantInt::get(ctx_.int64Type(), 1)), head_idx);
    builder.CreateBr(final_h_cond);

    builder.SetInsertPoint(final_d_done);
    // Store output[b, i, d]
    llvm::Value* out_final_idx = builder.CreateAdd(fb_offset,
        builder.CreateAdd(builder.CreateMul(fi, d_model), fd));
    builder.CreateStore(builder.CreateLoad(ctx_.doubleType(), final_sum),
        builder.CreateGEP(ctx_.doubleType(), output_elems, out_final_idx));

    builder.CreateStore(builder.CreateAdd(fd, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(final_d_cond);

    builder.SetInsertPoint(final_i_next);
    builder.CreateStore(builder.CreateAdd(fi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_var);
    builder.CreateBr(final_i_cond);

    builder.SetInsertPoint(final_batch_next);
    builder.CreateStore(builder.CreateAdd(fb, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(final_batch_cond);

    // === Finalize result tensor ===
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "mha_result");

    // Allocate dims: (batch, seq_q, d_model) or (seq_q, d_model)
    llvm::Value* dims_count = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 3),
        llvm::ConstantInt::get(ctx_.int64Type(), 2));
    llvm::Value* dims_bytes = builder.CreateMul(dims_count,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "mha_dims");

    llvm::BasicBlock* store_3d_dims = llvm::BasicBlock::Create(ctx_.context(), "mha_store_3d", current_func);
    llvm::BasicBlock* store_2d_dims = llvm::BasicBlock::Create(ctx_.context(), "mha_store_2d", current_func);
    llvm::BasicBlock* store_dims_done = llvm::BasicBlock::Create(ctx_.context(), "mha_store_done", current_func);

    builder.CreateCondBr(is_3d, store_3d_dims, store_2d_dims);

    builder.SetInsertPoint(store_3d_dims);
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateStore(d_model, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 2)));
    builder.CreateBr(store_dims_done);

    builder.SetInsertPoint(store_2d_dims);
    builder.CreateStore(seq_q, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(d_model, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));
    builder.CreateBr(store_dims_done);

    builder.SetInsertPoint(store_dims_done);

    llvm::Value* total_size = builder.CreateMul(batch_size, builder.CreateMul(seq_q, d_model));

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(dims_count, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

// === Track 8.3: Positional Encoding ===

llvm::Value* TensorCodegen::positionalEncoding(const eshkol_operations_t* op) {
    // Sinusoidal positional encoding from "Attention Is All You Need"
    // PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    // PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    if (op->call_op.num_vars != 2) {
        eshkol_error("positional-encoding requires 2 arguments: max-len d-model");
        return nullptr;
    }

    llvm::Value* max_len_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* d_model_val = codegenAST(&op->call_op.variables[1]);

    if (!max_len_val || !d_model_val) return nullptr;

    auto& builder = ctx_.builder();

    // Get integer values
    llvm::Value* max_len = max_len_val;
    if (max_len->getType() == ctx_.taggedValueType()) {
        max_len = tagged_.unpackInt64(max_len_val);
    }
    llvm::Value* d_model = d_model_val;
    if (d_model->getType() == ctx_.taggedValueType()) {
        d_model = tagged_.unpackInt64(d_model_val);
    }

    // Create tensor using proper helper
    std::vector<llvm::Value*> dims = {max_len, d_model};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, nullptr, true);  // Zero fill
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = builder.CreateLoad(ctx_.ptrType(), elements_field_ptr);

    // Get math functions
    llvm::Function* sin_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
    llvm::Function* cos_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Precompute log(10000) = 9.210340371976184
    llvm::Value* log_10000 = llvm::ConstantFP::get(ctx_.doubleType(), 9.210340371976184);
    llvm::Value* d_model_double = builder.CreateSIToFP(d_model, ctx_.doubleType());
    llvm::Value* half_d = builder.CreateSDiv(d_model, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    // Allocate loop counters upfront (before any branches)
    llvm::Value* pos_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pe_pos");
    llvm::Value* i_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "pe_i");

    // Loop over positions
    llvm::BasicBlock* pos_cond = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_cond", current_func);
    llvm::BasicBlock* pos_body = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_body", current_func);
    llvm::BasicBlock* pos_exit = llvm::BasicBlock::Create(ctx_.context(), "pe_pos_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), pos_counter);
    builder.CreateBr(pos_cond);

    builder.SetInsertPoint(pos_cond);
    llvm::Value* pos = builder.CreateLoad(ctx_.int64Type(), pos_counter);
    llvm::Value* pos_done = builder.CreateICmpUGE(pos, max_len);
    builder.CreateCondBr(pos_done, pos_exit, pos_body);

    builder.SetInsertPoint(pos_body);

    llvm::Value* pos_double = builder.CreateSIToFP(pos, ctx_.doubleType());
    llvm::Value* row_offset = builder.CreateMul(pos, d_model);

    // Inner loop over dimension pairs
    llvm::BasicBlock* dim_cond = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_cond", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_body", current_func);
    llvm::BasicBlock* dim_exit = llvm::BasicBlock::Create(ctx_.context(), "pe_dim_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_counter);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* i_done = builder.CreateICmpUGE(i, half_d);
    builder.CreateCondBr(i_done, dim_exit, dim_body);

    builder.SetInsertPoint(dim_body);

    // Compute div_term = exp(2*i * -log(10000) / d_model) = 1 / 10000^(2i/d_model)
    llvm::Value* i_double = builder.CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* two_i = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), i_double);
    llvm::Value* exponent = builder.CreateFMul(two_i,
        builder.CreateFDiv(builder.CreateFNeg(log_10000), d_model_double));
    llvm::Value* div_term = builder.CreateCall(exp_func, {exponent}, "div_term");

    // angle = pos * div_term
    llvm::Value* angle = builder.CreateFMul(pos_double, div_term, "angle");

    // PE[pos, 2i] = sin(angle)
    llvm::Value* sin_val = builder.CreateCall(sin_func, {angle}, "sin_val");
    llvm::Value* sin_bits = builder.CreateBitCast(sin_val, ctx_.int64Type());
    llvm::Value* even_idx = builder.CreateAdd(row_offset,
        builder.CreateMul(llvm::ConstantInt::get(ctx_.int64Type(), 2), i));
    llvm::Value* even_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, even_idx);
    builder.CreateStore(sin_bits, even_ptr);

    // PE[pos, 2i+1] = cos(angle)
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle}, "cos_val");
    llvm::Value* cos_bits = builder.CreateBitCast(cos_val, ctx_.int64Type());
    llvm::Value* odd_idx = builder.CreateAdd(even_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* odd_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, odd_idx);
    builder.CreateStore(cos_bits, odd_ptr);

    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, i_counter);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_exit);
    llvm::Value* next_pos = builder.CreateAdd(pos, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_pos, pos_counter);
    builder.CreateBr(pos_cond);

    builder.SetInsertPoint(pos_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

// === Additional Transformer Helpers ===

llvm::Value* TensorCodegen::rotaryEmbedding(const eshkol_operations_t* op) {
    // RoPE (Rotary Position Embedding)
    // Applies rotation to pairs of dimensions based on position
    // x_rot[..., 2i] = x[..., 2i] * cos(theta) - x[..., 2i+1] * sin(theta)
    // x_rot[..., 2i+1] = x[..., 2i] * sin(theta) + x[..., 2i+1] * cos(theta)
    // where theta = pos / 10000^(2i/dim)

    if (op->call_op.num_vars != 3) {
        eshkol_error("rotary-embedding requires 3 arguments: x seq-positions dim");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* pos_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* dim_val = codegenAST(&op->call_op.variables[2]);

    if (!x_val || !pos_val || !dim_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack x tensor
    llvm::Value* x_ptr_int = tagged_.unpackInt64(x_val);
    llvm::Value* x_ptr = builder.CreateIntToPtr(x_ptr_int, ctx_.ptrType());

    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims_ptr = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    // Unpack positions tensor
    llvm::Value* pos_ptr_int = tagged_.unpackInt64(pos_val);
    llvm::Value* pos_ptr = builder.CreateIntToPtr(pos_ptr_int, ctx_.ptrType());
    llvm::Value* pos_elems_field = builder.CreateStructGEP(tensor_type, pos_ptr, 2);
    llvm::Value* pos_elems = builder.CreateLoad(ctx_.ptrType(), pos_elems_field);

    // Get dim
    llvm::Value* dim = tagged_.unpackInt64(dim_val);

    // Allocate output
    llvm::Value* output_bytes = builder.CreateMul(x_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "rope_output");

    // Copy dims
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "rope_dims");
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), x_dims_ptr, llvm::MaybeAlign(8), dims_bytes);

    // Get math functions
    llvm::Function* sin_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sin, {ctx_.doubleType()});
    llvm::Function* cos_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::cos, {ctx_.doubleType()});
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get dimensions - assume x is (batch, seq_len, dim) or (seq_len, dim)
    llvm::Value* is_3d = builder.CreateICmpEQ(x_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch_size = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(),
            builder.CreateGEP(ctx_.int64Type(), x_dims_ptr, llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* seq_len = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims_ptr, seq_idx));

    llvm::Value* log_10000 = llvm::ConstantFP::get(ctx_.doubleType(), 9.210340371976184);
    llvm::Value* dim_double = builder.CreateSIToFP(dim, ctx_.doubleType());
    llvm::Value* half_dim = builder.CreateSDiv(dim, llvm::ConstantInt::get(ctx_.int64Type(), 2));

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* batch_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "batch_idx");
    llvm::Value* s_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "s_idx");
    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");

    // Main loop
    llvm::BasicBlock* batch_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_cond", current_func);
    llvm::BasicBlock* batch_body = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "rope_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), batch_idx);
    builder.CreateBr(batch_cond);

    builder.SetInsertPoint(batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), batch_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, finalize, batch_body);

    builder.SetInsertPoint(batch_body);

    llvm::Value* batch_offset = builder.CreateMul(b, builder.CreateMul(seq_len, dim));

    // Sequence loop
    llvm::BasicBlock* seq_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_cond", current_func);
    llvm::BasicBlock* seq_body = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_body", current_func);
    llvm::BasicBlock* batch_next = llvm::BasicBlock::Create(ctx_.context(), "rope_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(seq_cond);

    builder.SetInsertPoint(seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), s_idx);
    llvm::Value* s_done = builder.CreateICmpUGE(s, seq_len);
    builder.CreateCondBr(s_done, batch_next, seq_body);

    builder.SetInsertPoint(seq_body);

    // Get position for this sequence element
    llvm::Value* pos_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), pos_elems, s));

    llvm::Value* seq_offset = builder.CreateAdd(batch_offset, builder.CreateMul(s, dim));

    // Dimension pair loop
    llvm::BasicBlock* dim_cond = llvm::BasicBlock::Create(ctx_.context(), "rope_dim_cond", current_func);
    llvm::BasicBlock* dim_body = llvm::BasicBlock::Create(ctx_.context(), "rope_dim_body", current_func);
    llvm::BasicBlock* seq_next = llvm::BasicBlock::Create(ctx_.context(), "rope_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(dim_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, half_dim);
    builder.CreateCondBr(i_done, seq_next, dim_body);

    builder.SetInsertPoint(dim_body);

    // Compute theta = pos / 10000^(2i/dim)
    llvm::Value* i_double = builder.CreateSIToFP(i, ctx_.doubleType());
    llvm::Value* two_i = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), i_double);
    llvm::Value* exponent = builder.CreateFMul(two_i,
        builder.CreateFDiv(builder.CreateFNeg(log_10000), dim_double));
    llvm::Value* freq = builder.CreateCall(exp_func, {exponent});
    llvm::Value* theta = builder.CreateFMul(pos_elem, freq);

    llvm::Value* cos_theta = builder.CreateCall(cos_func, {theta});
    llvm::Value* sin_theta = builder.CreateCall(sin_func, {theta});

    // Get input pair
    llvm::Value* even_idx = builder.CreateAdd(seq_offset,
        builder.CreateMul(llvm::ConstantInt::get(ctx_.int64Type(), 2), i));
    llvm::Value* odd_idx = builder.CreateAdd(even_idx, llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* x_even = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, even_idx));
    llvm::Value* x_odd = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, odd_idx));

    // Apply rotation
    // out_even = x_even * cos - x_odd * sin
    // out_odd = x_even * sin + x_odd * cos
    llvm::Value* out_even = builder.CreateFSub(
        builder.CreateFMul(x_even, cos_theta),
        builder.CreateFMul(x_odd, sin_theta));
    llvm::Value* out_odd = builder.CreateFAdd(
        builder.CreateFMul(x_even, sin_theta),
        builder.CreateFMul(x_odd, cos_theta));

    builder.CreateStore(out_even, builder.CreateGEP(ctx_.doubleType(), output_elems, even_idx));
    builder.CreateStore(out_odd, builder.CreateGEP(ctx_.doubleType(), output_elems, odd_idx));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(dim_cond);

    builder.SetInsertPoint(seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(seq_cond);

    builder.SetInsertPoint(batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), batch_idx);
    builder.CreateBr(batch_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "rope_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(x_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::causalMask(const eshkol_operations_t* op) {
    // Creates causal mask for decoder self-attention (additive convention)
    // mask[i, j] = 0 if j <= i (attend), -inf otherwise (don't attend)
    // Added to attention scores before softmax

    if (op->call_op.num_vars != 1) {
        eshkol_error("causal-mask requires 1 argument: seq-len");
        return nullptr;
    }

    llvm::Value* seq_len_val = codegenAST(&op->call_op.variables[0]);
    if (!seq_len_val) return nullptr;

    auto& builder = ctx_.builder();

    llvm::Value* seq_len = seq_len_val;
    if (seq_len->getType() == ctx_.taggedValueType()) {
        seq_len = tagged_.unpackInt64(seq_len_val);
    }

    // Create tensor - fill with -inf (masked positions)
    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(ctx_.doubleType(), true);
    llvm::Value* neg_inf_bits = builder.CreateBitCast(neg_inf, ctx_.int64Type());

    std::vector<llvm::Value*> dims = {seq_len, seq_len};
    llvm::Value* tensor_ptr = createTensorWithDims(dims, neg_inf_bits, false);  // Fill with -inf
    if (!tensor_ptr) return nullptr;

    llvm::StructType* tensor_type = ctx_.tensorType();

    // Get elements pointer
    llvm::Value* elements_field_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elements_ptr = builder.CreateLoad(ctx_.ptrType(), elements_field_ptr);

    // Value for attending positions (0)
    llvm::Value* zero_double = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* zero_bits = builder.CreateBitCast(zero_double, ctx_.int64Type());

    // Fill lower triangle with 0 (attend)
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_i_cond = llvm::BasicBlock::Create(ctx_.context(), "mask_i_cond", current_func);
    llvm::BasicBlock* loop_i_body = llvm::BasicBlock::Create(ctx_.context(), "mask_i_body", current_func);
    llvm::BasicBlock* loop_i_exit = llvm::BasicBlock::Create(ctx_.context(), "mask_i_exit", current_func);

    // Allocate loop counters at entry
    llvm::Value* i_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mask_i");
    llvm::Value* j_counter = builder.CreateAlloca(ctx_.int64Type(), nullptr, "mask_j");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_counter);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_counter);
    llvm::Value* i_cmp = builder.CreateICmpULT(i, seq_len);
    builder.CreateCondBr(i_cmp, loop_i_body, loop_i_exit);

    builder.SetInsertPoint(loop_i_body);

    // Inner loop: j from 0 to i (inclusive) - these are positions we attend to
    llvm::BasicBlock* loop_j_cond = llvm::BasicBlock::Create(ctx_.context(), "mask_j_cond", current_func);
    llvm::BasicBlock* loop_j_body = llvm::BasicBlock::Create(ctx_.context(), "mask_j_body", current_func);
    llvm::BasicBlock* loop_j_exit = llvm::BasicBlock::Create(ctx_.context(), "mask_j_exit", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), j_counter);
    llvm::Value* i_plus_one = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_cond);
    llvm::Value* j = builder.CreateLoad(ctx_.int64Type(), j_counter);
    llvm::Value* j_cmp = builder.CreateICmpULT(j, i_plus_one);  // j <= i
    builder.CreateCondBr(j_cmp, loop_j_body, loop_j_exit);

    builder.SetInsertPoint(loop_j_body);
    // mask[i, j] = 0 (attend)
    llvm::Value* row_offset = builder.CreateMul(i, seq_len);
    llvm::Value* elem_idx = builder.CreateAdd(row_offset, j);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elements_ptr, elem_idx);
    builder.CreateStore(zero_bits, elem_ptr);

    llvm::Value* next_j = builder.CreateAdd(j, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_j, j_counter);
    builder.CreateBr(loop_j_cond);

    builder.SetInsertPoint(loop_j_exit);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, i_counter);
    builder.CreateBr(loop_i_cond);

    builder.SetInsertPoint(loop_i_exit);

    return tagged_.packHeapPtr(tensor_ptr);
}

llvm::Value* TensorCodegen::paddingMask(const eshkol_operations_t* op) {
    // Creates padding mask from sequence lengths
    // mask[b, i] = 0 if i < lengths[b], else -inf

    if (op->call_op.num_vars != 2) {
        eshkol_error("padding-mask requires 2 arguments: lengths max-len");
        return nullptr;
    }

    llvm::Value* lengths_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* max_len_val = codegenAST(&op->call_op.variables[1]);

    if (!lengths_val || !max_len_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get max_len
    llvm::Value* max_len = max_len_val;
    if (max_len->getType() == ctx_.taggedValueType()) {
        max_len = tagged_.unpackInt64(max_len_val);
    }

    // Unpack lengths tensor
    llvm::Value* lengths_ptr_int = tagged_.unpackInt64(lengths_val);
    llvm::Value* lengths_ptr = builder.CreateIntToPtr(lengths_ptr_int, ctx_.ptrType());
    llvm::Value* lengths_total_field = builder.CreateStructGEP(tensor_type, lengths_ptr, 3);
    llvm::Value* batch_size = builder.CreateLoad(ctx_.int64Type(), lengths_total_field);
    llvm::Value* lengths_elems_field = builder.CreateStructGEP(tensor_type, lengths_ptr, 2);
    llvm::Value* lengths_elems = builder.CreateLoad(ctx_.ptrType(), lengths_elems_field);

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Allocate mask: (batch, max_len)
    llvm::Value* total_size = builder.CreateMul(batch_size, max_len);
    llvm::Value* elems_bytes = builder.CreateMul(total_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Value* mask_elems = builder.CreateCall(arena_alloc, {arena_ptr, elems_bytes}, "pad_mask_elems");

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* neg_inf = llvm::ConstantFP::getInfinity(ctx_.doubleType(), true);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);

    // Fill mask
    llvm::BasicBlock* b_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_b_cond", current_func);
    llvm::BasicBlock* b_body = llvm::BasicBlock::Create(ctx_.context(), "pad_b_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "pad_finalize", current_func);

    llvm::Value* b_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "b_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(b_cond);

    builder.SetInsertPoint(b_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), b_idx);
    llvm::Value* b_done = builder.CreateICmpUGE(b, batch_size);
    builder.CreateCondBr(b_done, finalize, b_body);

    builder.SetInsertPoint(b_body);

    // Get length for this batch element
    // Tensor elements are stored as int64: small integers are plain int64,
    // doubles are stored as int64 bitpatterns. Use heuristic to distinguish.
    llvm::Value* len_raw = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), lengths_elems, b));
    // Small values (< 1000) are integers — use directly
    // Large values are double bitpatterns — BitCast and FPToSI
    llvm::Value* is_small = builder.CreateICmpULT(len_raw,
        llvm::ConstantInt::get(ctx_.int64Type(), 1000));
    llvm::Value* len_from_double = builder.CreateFPToSI(
        builder.CreateBitCast(len_raw, ctx_.doubleType()), ctx_.int64Type());
    llvm::Value* len = builder.CreateSelect(is_small, len_raw, len_from_double);

    llvm::Value* row_offset = builder.CreateMul(b, max_len);

    llvm::BasicBlock* i_cond = llvm::BasicBlock::Create(ctx_.context(), "pad_i_cond", current_func);
    llvm::BasicBlock* i_body = llvm::BasicBlock::Create(ctx_.context(), "pad_i_body", current_func);
    llvm::BasicBlock* b_next = llvm::BasicBlock::Create(ctx_.context(), "pad_b_next", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(i_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    llvm::Value* i_done = builder.CreateICmpUGE(i, max_len);
    builder.CreateCondBr(i_done, b_next, i_body);

    builder.SetInsertPoint(i_body);

    // mask[b, i] = 0 if i < len[b], else -inf
    llvm::Value* is_valid = builder.CreateICmpULT(i, len);
    llvm::Value* mask_val = builder.CreateSelect(is_valid, zero, neg_inf);

    llvm::Value* elem_idx = builder.CreateAdd(row_offset, i);
    builder.CreateStore(mask_val, builder.CreateGEP(ctx_.doubleType(), mask_elems, elem_idx));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(i_cond);

    builder.SetInsertPoint(b_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(b_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "pad_mask_result");

    llvm::Value* dims_bytes = llvm::ConstantInt::get(ctx_.int64Type(), 2 * sizeof(int64_t));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "pad_mask_dims");
    builder.CreateStore(batch_size, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 0)));
    builder.CreateStore(max_len, builder.CreateGEP(ctx_.int64Type(), result_dims,
        llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 2), r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(mask_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(total_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::feedForward(const eshkol_operations_t* op) {
    // Feed-forward network: FFN(x) = W2 * GELU(W1 * x + b1) + b2

    if (op->call_op.num_vars != 5) {
        eshkol_error("feed-forward requires 5 arguments: x W1 b1 W2 b2");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* w1_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* b1_val = codegenAST(&op->call_op.variables[2]);
    llvm::Value* w2_val = codegenAST(&op->call_op.variables[3]);
    llvm::Value* b2_val = codegenAST(&op->call_op.variables[4]);

    if (!x_val || !w1_val || !b1_val || !w2_val || !b2_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack all tensors
    llvm::Value* x_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(x_val), ctx_.ptrType());
    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);

    llvm::Value* w1_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(w1_val), ctx_.ptrType());
    llvm::Value* w1_dims_field = builder.CreateStructGEP(tensor_type, w1_ptr, 0);
    llvm::Value* w1_dims = builder.CreateLoad(ctx_.ptrType(), w1_dims_field);
    llvm::Value* w1_ndim_field = builder.CreateStructGEP(tensor_type, w1_ptr, 1);
    llvm::Value* w1_ndim = builder.CreateLoad(ctx_.int64Type(), w1_ndim_field);
    llvm::Value* w1_elems_field = builder.CreateStructGEP(tensor_type, w1_ptr, 2);
    llvm::Value* w1_elems = builder.CreateLoad(ctx_.ptrType(), w1_elems_field);

    // Guard: FFN requires 2D weight matrix
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(w1_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "ffn_wdims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "ffn_wdims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: FFN requires 2D weight matrix (got %lldD)\n");
            builder.CreateCall(pf, {fmt, w1_ndim});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    llvm::Value* b1_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b1_val), ctx_.ptrType());
    llvm::Value* b1_elems_field = builder.CreateStructGEP(tensor_type, b1_ptr, 2);
    llvm::Value* b1_elems = builder.CreateLoad(ctx_.ptrType(), b1_elems_field);

    llvm::Value* w2_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(w2_val), ctx_.ptrType());
    llvm::Value* w2_dims_field = builder.CreateStructGEP(tensor_type, w2_ptr, 0);
    llvm::Value* w2_dims = builder.CreateLoad(ctx_.ptrType(), w2_dims_field);
    llvm::Value* w2_elems_field = builder.CreateStructGEP(tensor_type, w2_ptr, 2);
    llvm::Value* w2_elems = builder.CreateLoad(ctx_.ptrType(), w2_elems_field);

    llvm::Value* b2_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(b2_val), ctx_.ptrType());
    llvm::Value* b2_elems_field = builder.CreateStructGEP(tensor_type, b2_ptr, 2);
    llvm::Value* b2_elems = builder.CreateLoad(ctx_.ptrType(), b2_elems_field);

    // Get dimensions
    // x: (batch, seq, d_model) or (seq, d_model)
    // W1: (d_model, d_ff), b1: (d_ff,)
    // W2: (d_ff, d_model), b2: (d_model,)

    llvm::Value* is_3d = builder.CreateICmpEQ(x_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 3));

    llvm::Value* batch = builder.CreateSelect(is_3d,
        builder.CreateLoad(ctx_.int64Type(), builder.CreateGEP(ctx_.int64Type(), x_dims,
            llvm::ConstantInt::get(ctx_.int64Type(), 0))),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 1),
        llvm::ConstantInt::get(ctx_.int64Type(), 0));
    llvm::Value* dim_idx = builder.CreateSelect(is_3d,
        llvm::ConstantInt::get(ctx_.int64Type(), 2),
        llvm::ConstantInt::get(ctx_.int64Type(), 1));

    llvm::Value* seq_len = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims, seq_idx));
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), x_dims, dim_idx));
    llvm::Value* d_ff = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), w1_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate intermediate: (batch, seq, d_ff)
    llvm::Value* hidden_size = builder.CreateMul(batch, builder.CreateMul(seq_len, d_ff));
    llvm::Value* hidden_bytes = builder.CreateMul(hidden_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* hidden = builder.CreateCall(arena_alloc, {arena_ptr, hidden_bytes}, "ffn_hidden");

    // Allocate output: (batch, seq, d_model)
    llvm::Value* output_size = builder.CreateMul(batch, builder.CreateMul(seq_len, d_model));
    llvm::Value* output_bytes = builder.CreateMul(output_size,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "ffn_output");

    // GELU approximation constants
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* sqrt_2_pi = llvm::ConstantFP::get(ctx_.doubleType(), 0.7978845608028654);
    llvm::Value* coeff = llvm::ConstantFP::get(ctx_.doubleType(), 0.044715);

    llvm::Function* tanh_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    // Note: LLVM doesn't have intrinsic tanh, we'll compute it via exp
    llvm::Function* exp_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::exp, {ctx_.doubleType()});

    // === First layer: hidden = GELU(x @ W1 + b1) ===
    llvm::BasicBlock* layer1_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_cond", current_func);
    llvm::BasicBlock* layer1_batch_body = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_body", current_func);
    llvm::BasicBlock* layer2_init = llvm::BasicBlock::Create(ctx_.context(), "layer2_init", current_func);

    // === ALLOCATE ALL LOOP VARIABLES UPFRONT ===
    llvm::Value* b_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "b_idx");
    llvm::Value* s_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "s_idx");
    llvm::Value* d_idx_var = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    llvm::Value* sum = builder.CreateAlloca(ctx_.doubleType(), nullptr, "sum");
    llvm::Value* k_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "k_idx");

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(layer1_batch_cond);

    builder.SetInsertPoint(layer1_batch_cond);
    llvm::Value* b = builder.CreateLoad(ctx_.int64Type(), b_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(b, batch), layer2_init, layer1_batch_body);

    builder.SetInsertPoint(layer1_batch_body);

    llvm::Value* x_batch_off = builder.CreateMul(b, builder.CreateMul(seq_len, d_model));
    llvm::Value* h_batch_off = builder.CreateMul(b, builder.CreateMul(seq_len, d_ff));

    // Sequence loop
    llvm::BasicBlock* l1_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_cond", current_func);
    llvm::BasicBlock* l1_seq_body = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_body", current_func);
    llvm::BasicBlock* l1_batch_next = llvm::BasicBlock::Create(ctx_.context(), "l1_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(l1_seq_cond);

    builder.SetInsertPoint(l1_seq_cond);
    llvm::Value* s = builder.CreateLoad(ctx_.int64Type(), s_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(s, seq_len), l1_batch_next, l1_seq_body);

    builder.SetInsertPoint(l1_seq_body);

    llvm::Value* x_seq_off = builder.CreateAdd(x_batch_off, builder.CreateMul(s, d_model));
    llvm::Value* h_seq_off = builder.CreateAdd(h_batch_off, builder.CreateMul(s, d_ff));

    // Output dimension loop (d_ff)
    llvm::BasicBlock* l1_d_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_d_cond", current_func);
    llvm::BasicBlock* l1_d_body = llvm::BasicBlock::Create(ctx_.context(), "l1_d_body", current_func);
    llvm::BasicBlock* l1_seq_next = llvm::BasicBlock::Create(ctx_.context(), "l1_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx_var);
    builder.CreateBr(l1_d_cond);

    builder.SetInsertPoint(l1_d_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx_var);
    builder.CreateCondBr(builder.CreateICmpUGE(d, d_ff), l1_seq_next, l1_d_body);

    builder.SetInsertPoint(l1_d_body);

    // Compute dot product: x[s, :] @ W1[:, d] + b1[d]
    llvm::BasicBlock* l1_k_cond = llvm::BasicBlock::Create(ctx_.context(), "l1_k_cond", current_func);
    llvm::BasicBlock* l1_k_body = llvm::BasicBlock::Create(ctx_.context(), "l1_k_body", current_func);
    llvm::BasicBlock* l1_apply_gelu = llvm::BasicBlock::Create(ctx_.context(), "l1_apply_gelu", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(l1_k_cond);

    builder.SetInsertPoint(l1_k_cond);
    llvm::Value* k = builder.CreateLoad(ctx_.int64Type(), k_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(k, d_model), l1_apply_gelu, l1_k_body);

    builder.SetInsertPoint(l1_k_body);
    llvm::Value* x_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, builder.CreateAdd(x_seq_off, k)));
    llvm::Value* w1_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w1_elems, builder.CreateAdd(builder.CreateMul(k, d_ff), d)));
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, builder.CreateFMul(x_elem, w1_elem)), sum);
    builder.CreateStore(builder.CreateAdd(k, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_idx);
    builder.CreateBr(l1_k_cond);

    builder.SetInsertPoint(l1_apply_gelu);
    // Add bias
    llvm::Value* bias1 = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), b1_elems, d));
    llvm::Value* pre_act = builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum), bias1);

    // Apply GELU: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    llvm::Value* x3 = builder.CreateFMul(pre_act, builder.CreateFMul(pre_act, pre_act));
    llvm::Value* inner = builder.CreateFMul(sqrt_2_pi,
        builder.CreateFAdd(pre_act, builder.CreateFMul(coeff, x3)));
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    llvm::Value* two_inner = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), inner);
    llvm::Value* exp_2x = builder.CreateCall(exp_func, {two_inner});
    llvm::Value* tanh_val = builder.CreateFDiv(
        builder.CreateFSub(exp_2x, llvm::ConstantFP::get(ctx_.doubleType(), 1.0)),
        builder.CreateFAdd(exp_2x, llvm::ConstantFP::get(ctx_.doubleType(), 1.0)));
    llvm::Value* gelu_out = builder.CreateFMul(
        llvm::ConstantFP::get(ctx_.doubleType(), 0.5),
        builder.CreateFMul(pre_act,
            builder.CreateFAdd(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), tanh_val)));

    builder.CreateStore(gelu_out,
        builder.CreateGEP(ctx_.doubleType(), hidden, builder.CreateAdd(h_seq_off, d)));

    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx_var);
    builder.CreateBr(l1_d_cond);

    builder.SetInsertPoint(l1_seq_next);
    builder.CreateStore(builder.CreateAdd(s, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(l1_seq_cond);

    builder.SetInsertPoint(l1_batch_next);
    builder.CreateStore(builder.CreateAdd(b, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(layer1_batch_cond);

    // === Second layer: output = hidden @ W2 + b2 ===
    builder.SetInsertPoint(layer2_init);

    llvm::BasicBlock* l2_batch_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_cond", current_func);
    llvm::BasicBlock* l2_batch_body = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "ffn_finalize", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), b_idx);
    builder.CreateBr(l2_batch_cond);

    builder.SetInsertPoint(l2_batch_cond);
    llvm::Value* b2_loop = builder.CreateLoad(ctx_.int64Type(), b_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(b2_loop, batch), finalize, l2_batch_body);

    builder.SetInsertPoint(l2_batch_body);

    llvm::Value* h_batch_off2 = builder.CreateMul(b2_loop, builder.CreateMul(seq_len, d_ff));
    llvm::Value* o_batch_off = builder.CreateMul(b2_loop, builder.CreateMul(seq_len, d_model));

    llvm::BasicBlock* l2_seq_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_cond", current_func);
    llvm::BasicBlock* l2_seq_body = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_body", current_func);
    llvm::BasicBlock* l2_batch_next = llvm::BasicBlock::Create(ctx_.context(), "l2_batch_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), s_idx);
    builder.CreateBr(l2_seq_cond);

    builder.SetInsertPoint(l2_seq_cond);
    llvm::Value* s2 = builder.CreateLoad(ctx_.int64Type(), s_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(s2, seq_len), l2_batch_next, l2_seq_body);

    builder.SetInsertPoint(l2_seq_body);

    llvm::Value* h_seq_off2 = builder.CreateAdd(h_batch_off2, builder.CreateMul(s2, d_ff));
    llvm::Value* o_seq_off = builder.CreateAdd(o_batch_off, builder.CreateMul(s2, d_model));

    llvm::BasicBlock* l2_d_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_d_cond", current_func);
    llvm::BasicBlock* l2_d_body = llvm::BasicBlock::Create(ctx_.context(), "l2_d_body", current_func);
    llvm::BasicBlock* l2_seq_next = llvm::BasicBlock::Create(ctx_.context(), "l2_seq_next", current_func);

    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx_var);
    builder.CreateBr(l2_d_cond);

    builder.SetInsertPoint(l2_d_cond);
    llvm::Value* d2 = builder.CreateLoad(ctx_.int64Type(), d_idx_var);
    builder.CreateCondBr(builder.CreateICmpUGE(d2, d_model), l2_seq_next, l2_d_body);

    builder.SetInsertPoint(l2_d_body);

    llvm::BasicBlock* l2_k_cond = llvm::BasicBlock::Create(ctx_.context(), "l2_k_cond", current_func);
    llvm::BasicBlock* l2_k_body = llvm::BasicBlock::Create(ctx_.context(), "l2_k_body", current_func);
    llvm::BasicBlock* l2_store = llvm::BasicBlock::Create(ctx_.context(), "l2_store", current_func);

    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), k_idx);
    builder.CreateBr(l2_k_cond);

    builder.SetInsertPoint(l2_k_cond);
    llvm::Value* k2 = builder.CreateLoad(ctx_.int64Type(), k_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(k2, d_ff), l2_store, l2_k_body);

    builder.SetInsertPoint(l2_k_body);
    llvm::Value* h_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), hidden, builder.CreateAdd(h_seq_off2, k2)));
    llvm::Value* w2_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w2_elems, builder.CreateAdd(builder.CreateMul(k2, d_model), d2)));
    llvm::Value* cur_sum2 = builder.CreateLoad(ctx_.doubleType(), sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum2, builder.CreateFMul(h_elem, w2_elem)), sum);
    builder.CreateStore(builder.CreateAdd(k2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), k_idx);
    builder.CreateBr(l2_k_cond);

    builder.SetInsertPoint(l2_store);
    llvm::Value* bias2 = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), b2_elems, d2));
    llvm::Value* out_val = builder.CreateFAdd(builder.CreateLoad(ctx_.doubleType(), sum), bias2);
    builder.CreateStore(out_val,
        builder.CreateGEP(ctx_.doubleType(), output, builder.CreateAdd(o_seq_off, d2)));

    builder.CreateStore(builder.CreateAdd(d2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx_var);
    builder.CreateBr(l2_d_cond);

    builder.SetInsertPoint(l2_seq_next);
    builder.CreateStore(builder.CreateAdd(s2, llvm::ConstantInt::get(ctx_.int64Type(), 1)), s_idx);
    builder.CreateBr(l2_seq_cond);

    builder.SetInsertPoint(l2_batch_next);
    builder.CreateStore(builder.CreateAdd(b2_loop, llvm::ConstantInt::get(ctx_.int64Type(), 1)), b_idx);
    builder.CreateBr(l2_batch_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "ffn_result");

    // Copy dims from input
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* result_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "ffn_dims");
    builder.CreateMemCpy(result_dims, llvm::MaybeAlign(8), x_dims, llvm::MaybeAlign(8), dims_bytes);

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(result_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_size, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::dropout(const eshkol_operations_t* op) {
    // Dropout: randomly zero elements and scale by 1/(1-rate)
    // Only applied during training

    if (op->call_op.num_vars != 3) {
        eshkol_error("dropout requires 3 arguments: x rate training");
        return nullptr;
    }

    llvm::Value* x_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* rate_val = codegenAST(&op->call_op.variables[1]);
    llvm::Value* training_val = codegenAST(&op->call_op.variables[2]);

    if (!x_val || !rate_val || !training_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack tensors
    llvm::Value* x_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(x_val), ctx_.ptrType());
    llvm::Value* x_dims_field = builder.CreateStructGEP(tensor_type, x_ptr, 0);
    llvm::Value* x_dims = builder.CreateLoad(ctx_.ptrType(), x_dims_field);
    llvm::Value* x_ndim_field = builder.CreateStructGEP(tensor_type, x_ptr, 1);
    llvm::Value* x_ndim = builder.CreateLoad(ctx_.int64Type(), x_ndim_field);
    llvm::Value* x_elems_field = builder.CreateStructGEP(tensor_type, x_ptr, 2);
    llvm::Value* x_elems = builder.CreateLoad(ctx_.ptrType(), x_elems_field);
    llvm::Value* x_total_field = builder.CreateStructGEP(tensor_type, x_ptr, 3);
    llvm::Value* x_total = builder.CreateLoad(ctx_.int64Type(), x_total_field);

    llvm::Value* rate = tagged_.unpackDouble(rate_val);
    llvm::Value* training = tagged_.unpackInt64(training_val);
    llvm::Value* is_training = builder.CreateICmpNE(training,
        llvm::ConstantInt::get(ctx_.int64Type(), 0));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Allocate output
    llvm::Value* output_bytes = builder.CreateMul(x_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "drop_output");

    // Copy dims
    llvm::Value* dims_bytes = builder.CreateMul(x_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, dims_bytes}, "drop_dims");
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), x_dims, llvm::MaybeAlign(8), dims_bytes);

    // Compute scale = 1 / (1 - rate)
    llvm::Value* scale = builder.CreateFDiv(
        llvm::ConstantFP::get(ctx_.doubleType(), 1.0),
        builder.CreateFSub(llvm::ConstantFP::get(ctx_.doubleType(), 1.0), rate));

    // Simple LCG random number generator state
    llvm::Value* rng_state = builder.CreateAlloca(ctx_.int64Type(), nullptr, "rng_state");
    // Seed with a constant (in production, use better seeding)
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 123456789), rng_state);

    // Main loop
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "drop_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "drop_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "drop_finalize", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(i, x_total), finalize, loop_body);

    builder.SetInsertPoint(loop_body);

    llvm::Value* x_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), x_elems, i));

    // Generate random number using LCG: state = (a * state + c) mod m
    // a = 1103515245, c = 12345, m = 2^31
    llvm::Value* state = builder.CreateLoad(ctx_.int64Type(), rng_state);
    llvm::Value* new_state = builder.CreateAdd(
        builder.CreateMul(state, llvm::ConstantInt::get(ctx_.int64Type(), 1103515245)),
        llvm::ConstantInt::get(ctx_.int64Type(), 12345));
    new_state = builder.CreateAnd(new_state, llvm::ConstantInt::get(ctx_.int64Type(), 0x7FFFFFFF));
    builder.CreateStore(new_state, rng_state);

    // Convert to [0, 1) range
    llvm::Value* rand_double = builder.CreateUIToFP(new_state, ctx_.doubleType());
    rand_double = builder.CreateFDiv(rand_double,
        llvm::ConstantFP::get(ctx_.doubleType(), 2147483648.0));

    // Apply dropout: if rand < rate, zero out; else scale
    llvm::Value* should_drop = builder.CreateFCmpOLT(rand_double, rate);

    // Only apply during training
    llvm::Value* actually_drop = builder.CreateAnd(is_training, should_drop);

    llvm::Value* out_val = builder.CreateSelect(actually_drop,
        llvm::ConstantFP::get(ctx_.doubleType(), 0.0),
        builder.CreateSelect(is_training,
            builder.CreateFMul(x_elem, scale),
            x_elem));  // No scaling during inference

    builder.CreateStore(out_val, builder.CreateGEP(ctx_.doubleType(), output_elems, i));

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(loop_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "drop_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(x_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(x_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}

llvm::Value* TensorCodegen::embedding(const eshkol_operations_t* op) {
    // Embedding lookup: output[b, s, :] = weights[indices[b, s], :]

    if (op->call_op.num_vars != 2) {
        eshkol_error("embedding requires 2 arguments: indices weights");
        return nullptr;
    }

    llvm::Value* indices_val = codegenAST(&op->call_op.variables[0]);
    llvm::Value* weights_val = codegenAST(&op->call_op.variables[1]);

    if (!indices_val || !weights_val) return nullptr;

    auto& builder = ctx_.builder();
    llvm::Type* tensor_type = ctx_.tensorType();

    // Get arena
    llvm::Value* arena_ptr = builder.CreateLoad(
        llvm::PointerType::get(ctx_.context(), 0), ctx_.globalArena());

    // Unpack indices tensor
    llvm::Value* idx_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(indices_val), ctx_.ptrType());
    llvm::Value* idx_dims_field = builder.CreateStructGEP(tensor_type, idx_ptr, 0);
    llvm::Value* idx_dims = builder.CreateLoad(ctx_.ptrType(), idx_dims_field);
    llvm::Value* idx_ndim_field = builder.CreateStructGEP(tensor_type, idx_ptr, 1);
    llvm::Value* idx_ndim = builder.CreateLoad(ctx_.int64Type(), idx_ndim_field);
    llvm::Value* idx_elems_field = builder.CreateStructGEP(tensor_type, idx_ptr, 2);
    llvm::Value* idx_elems = builder.CreateLoad(ctx_.ptrType(), idx_elems_field);
    llvm::Value* idx_total_field = builder.CreateStructGEP(tensor_type, idx_ptr, 3);
    llvm::Value* idx_total = builder.CreateLoad(ctx_.int64Type(), idx_total_field);

    // Unpack weights tensor: (vocab_size, d_model)
    llvm::Value* w_ptr = builder.CreateIntToPtr(tagged_.unpackInt64(weights_val), ctx_.ptrType());
    llvm::Value* w_dims_field = builder.CreateStructGEP(tensor_type, w_ptr, 0);
    llvm::Value* w_dims = builder.CreateLoad(ctx_.ptrType(), w_dims_field);
    llvm::Value* w_ndim_field = builder.CreateStructGEP(tensor_type, w_ptr, 1);
    llvm::Value* w_ndim = builder.CreateLoad(ctx_.int64Type(), w_ndim_field);
    llvm::Value* w_elems_field = builder.CreateStructGEP(tensor_type, w_ptr, 2);
    llvm::Value* w_elems = builder.CreateLoad(ctx_.ptrType(), w_elems_field);

    // Guard: embedding requires 2D weight matrix
    {
        llvm::Function* cur_fn = builder.GetInsertBlock()->getParent();
        llvm::Value* dims_ok = builder.CreateICmpUGE(w_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 2));
        llvm::BasicBlock* ok_bb = llvm::BasicBlock::Create(ctx_.context(), "emb_wdims_ok", cur_fn);
        llvm::BasicBlock* err_bb = llvm::BasicBlock::Create(ctx_.context(), "emb_wdims_err", cur_fn);
        builder.CreateCondBr(dims_ok, ok_bb, err_bb);
        builder.SetInsertPoint(err_bb);
        llvm::Function* pf = ctx_.lookupFunction("printf");
        llvm::Function* ef = ctx_.lookupFunction("exit");
        if (pf && ef) {
            llvm::Value* fmt = builder.CreateGlobalString("Error: embedding requires 2D weight matrix (got %lldD)\n");
            builder.CreateCall(pf, {fmt, w_ndim});
            builder.CreateCall(ef, {llvm::ConstantInt::get(builder.getInt32Ty(), 1)});
        }
        builder.CreateUnreachable();
        builder.SetInsertPoint(ok_bb);
    }

    // Get d_model from weights
    llvm::Value* d_model = builder.CreateLoad(ctx_.int64Type(),
        builder.CreateGEP(ctx_.int64Type(), w_dims, llvm::ConstantInt::get(ctx_.int64Type(), 1)));

    llvm::Function* arena_alloc = mem_.getArenaAllocate();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Output shape: indices_shape + (d_model,)
    // For indices (batch, seq), output is (batch, seq, d_model)
    llvm::Value* output_ndim = builder.CreateAdd(idx_ndim, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    llvm::Value* output_total = builder.CreateMul(idx_total, d_model);

    llvm::Value* output_bytes = builder.CreateMul(output_total,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(double)));
    llvm::Value* output_elems = builder.CreateCall(arena_alloc, {arena_ptr, output_bytes}, "emb_output");

    // Allocate output dims
    llvm::Value* output_dims_bytes = builder.CreateMul(output_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    llvm::Value* output_dims = builder.CreateCall(arena_alloc, {arena_ptr, output_dims_bytes}, "emb_dims");

    // Copy indices dims and add d_model
    llvm::Value* idx_dims_bytes = builder.CreateMul(idx_ndim,
        llvm::ConstantInt::get(ctx_.int64Type(), sizeof(int64_t)));
    builder.CreateMemCpy(output_dims, llvm::MaybeAlign(8), idx_dims, llvm::MaybeAlign(8), idx_dims_bytes);
    builder.CreateStore(d_model,
        builder.CreateGEP(ctx_.int64Type(), output_dims, idx_ndim));

    // Main loop over indices
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "emb_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "emb_body", current_func);
    llvm::BasicBlock* finalize = llvm::BasicBlock::Create(ctx_.context(), "emb_finalize", current_func);

    llvm::Value* i_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "i_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), i_idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), i_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(i, idx_total), finalize, loop_body);

    builder.SetInsertPoint(loop_body);

    // Get index value (stored as double, convert to int)
    llvm::Value* idx_double = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), idx_elems, i));
    llvm::Value* idx_int = builder.CreateFPToSI(idx_double, ctx_.int64Type());

    // Copy embedding vector
    llvm::Value* w_offset = builder.CreateMul(idx_int, d_model);
    llvm::Value* out_offset = builder.CreateMul(i, d_model);

    // Inner loop over d_model
    llvm::BasicBlock* inner_cond = llvm::BasicBlock::Create(ctx_.context(), "emb_inner_cond", current_func);
    llvm::BasicBlock* inner_body = llvm::BasicBlock::Create(ctx_.context(), "emb_inner_body", current_func);
    llvm::BasicBlock* loop_next = llvm::BasicBlock::Create(ctx_.context(), "emb_next", current_func);

    llvm::Value* d_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "d_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), d_idx);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(inner_cond);
    llvm::Value* d = builder.CreateLoad(ctx_.int64Type(), d_idx);
    builder.CreateCondBr(builder.CreateICmpUGE(d, d_model), loop_next, inner_body);

    builder.SetInsertPoint(inner_body);
    llvm::Value* w_elem = builder.CreateLoad(ctx_.doubleType(),
        builder.CreateGEP(ctx_.doubleType(), w_elems, builder.CreateAdd(w_offset, d)));
    builder.CreateStore(w_elem,
        builder.CreateGEP(ctx_.doubleType(), output_elems, builder.CreateAdd(out_offset, d)));
    builder.CreateStore(builder.CreateAdd(d, llvm::ConstantInt::get(ctx_.int64Type(), 1)), d_idx);
    builder.CreateBr(inner_cond);

    builder.SetInsertPoint(loop_next);
    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), i_idx);
    builder.CreateBr(loop_cond);

    // Finalize
    builder.SetInsertPoint(finalize);

    llvm::Function* alloc_tensor = mem_.getArenaAllocateTensorWithHeader();
    llvm::Value* result_ptr = builder.CreateCall(alloc_tensor, {arena_ptr}, "emb_result");

    llvm::Value* r_dims_field = builder.CreateStructGEP(tensor_type, result_ptr, 0);
    builder.CreateStore(output_dims, r_dims_field);
    llvm::Value* r_ndim_field = builder.CreateStructGEP(tensor_type, result_ptr, 1);
    builder.CreateStore(output_ndim, r_ndim_field);
    llvm::Value* r_elems_field = builder.CreateStructGEP(tensor_type, result_ptr, 2);
    builder.CreateStore(output_elems, r_elems_field);
    llvm::Value* r_total_field = builder.CreateStructGEP(tensor_type, result_ptr, 3);
    builder.CreateStore(output_total, r_total_field);

    return tagged_.packHeapPtr(result_ptr);
}


} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
