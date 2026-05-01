/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Loss Functions (Track 6.3). Extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split. Implements
 * MSE, MAE, cross-entropy, BCE, KL divergence, Huber, hinge,
 * cosine-similarity, contrastive, and related losses.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-loss-extract baseline.
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

// ===== LOSS FUNCTIONS (Track 6.3) =====

llvm::Value* TensorCodegen::mseLoss(const eshkol_operations_t* op) {
    // mse-loss: (mse-loss predictions targets)
    // MSE = (1/n) * sum((pred - target)^2)
    if (op->call_op.num_vars < 2) {
        eshkol_error("mse-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "mse_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "mse_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "mse_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "mse_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Sum squared differences
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* sum_sq_diff = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_sq_diff);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Value* diff_sq = builder.CreateFMul(diff, diff);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq_diff);
    builder.CreateStore(builder.CreateFAdd(cur_sum, diff_sq), sum_sq_diff);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq_diff);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* mse = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(mse);
}

llvm::Value* TensorCodegen::crossEntropyLoss(const eshkol_operations_t* op) {
    // cross-entropy-loss: (cross-entropy-loss logits targets)
    // CE = -sum(target * log(softmax(logits)))
    // Numerically stable: CE = -sum(target * (logits - logsumexp(logits)))
    if (op->call_op.num_vars < 2) {
        eshkol_error("cross-entropy-loss requires 2 arguments: logits, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* logits_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* targets_tagged = codegenAST(&op->call_op.variables[1]);
    if (!logits_tagged || !targets_tagged) return nullptr;

    llvm::Value* logits_ptr = tagged_.unpackPtr(logits_tagged);
    llvm::Value* targets_ptr = tagged_.unpackPtr(targets_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* logits_elems_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 2);
    llvm::Value* logits_elems = builder.CreateLoad(ctx_.ptrType(), logits_elems_ptr);
    llvm::Value* targets_elems_ptr = builder.CreateStructGEP(tensor_type, targets_ptr, 2);
    llvm::Value* targets_elems = builder.CreateLoad(ctx_.ptrType(), targets_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", ctx_.module());
    }
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", ctx_.module());
    }

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "ce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "ce_size_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Find max for numerical stability
    llvm::BasicBlock* max_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_max_loop", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "ce_max_body", current_func);
    llvm::BasicBlock* max_done = llvm::BasicBlock::Create(ctx_.context(), "ce_max_done", current_func);

    llvm::Value* max_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), max_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), -1e308), max_val);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_loop);
    llvm::Value* mi = builder.CreateLoad(ctx_.int64Type(), max_idx);
    builder.CreateCondBr(builder.CreateICmpULT(mi, num_elements), max_body, max_done);

    builder.SetInsertPoint(max_body);
    llvm::Value* logit_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, mi);
    llvm::Value* logit_bits = builder.CreateLoad(ctx_.int64Type(), logit_ptr);
    llvm::Value* logit = builder.CreateBitCast(logit_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(logit, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, logit, cur_max);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(builder.CreateAdd(mi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), max_idx);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_done);
    llvm::Value* max_logit = builder.CreateLoad(ctx_.doubleType(), max_val);

    // Compute logsumexp = max + log(sum(exp(x - max)))
    llvm::BasicBlock* lse_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_loop", current_func);
    llvm::BasicBlock* lse_body = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_body", current_func);
    llvm::BasicBlock* lse_done = llvm::BasicBlock::Create(ctx_.context(), "ce_lse_done", current_func);

    llvm::Value* lse_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* exp_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), lse_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), exp_sum);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_loop);
    llvm::Value* li = builder.CreateLoad(ctx_.int64Type(), lse_idx);
    builder.CreateCondBr(builder.CreateICmpULT(li, num_elements), lse_body, lse_done);

    builder.SetInsertPoint(lse_body);
    llvm::Value* l_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, li);
    llvm::Value* l_bits = builder.CreateLoad(ctx_.int64Type(), l_ptr);
    llvm::Value* l_val = builder.CreateBitCast(l_bits, ctx_.doubleType());
    llvm::Value* shifted = builder.CreateFSub(l_val, max_logit);
    llvm::Value* exp_val = builder.CreateCall(exp_func, {shifted});
    llvm::Value* cur_exp_sum = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    builder.CreateStore(builder.CreateFAdd(cur_exp_sum, exp_val), exp_sum);
    builder.CreateStore(builder.CreateAdd(li, llvm::ConstantInt::get(ctx_.int64Type(), 1)), lse_idx);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_done);
    llvm::Value* total_exp_sum = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    llvm::Value* log_sum = builder.CreateCall(log_func, {total_exp_sum});
    llvm::Value* logsumexp = builder.CreateFAdd(max_logit, log_sum);

    // Compute cross entropy: -sum(target * (logit - logsumexp))
    llvm::BasicBlock* ce_loop = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_loop", current_func);
    llvm::BasicBlock* ce_body = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_body", current_func);
    llvm::BasicBlock* ce_done = llvm::BasicBlock::Create(ctx_.context(), "ce_sum_done", current_func);

    llvm::Value* ce_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* ce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), ce_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), ce_sum);
    builder.CreateBr(ce_loop);

    builder.SetInsertPoint(ce_loop);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), ce_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ci, num_elements), ce_body, ce_done);

    builder.SetInsertPoint(ce_body);
    llvm::Value* logit_p = builder.CreateGEP(ctx_.int64Type(), logits_elems, ci);
    llvm::Value* logit_b = builder.CreateLoad(ctx_.int64Type(), logit_p);
    llvm::Value* logit_v = builder.CreateBitCast(logit_b, ctx_.doubleType());

    llvm::Value* target_p = builder.CreateGEP(ctx_.int64Type(), targets_elems, ci);
    llvm::Value* target_b = builder.CreateLoad(ctx_.int64Type(), target_p);
    llvm::Value* target_v = builder.CreateBitCast(target_b, ctx_.doubleType());

    llvm::Value* log_prob = builder.CreateFSub(logit_v, logsumexp);
    llvm::Value* term = builder.CreateFMul(target_v, log_prob);
    llvm::Value* cur_ce = builder.CreateLoad(ctx_.doubleType(), ce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_ce, term), ce_sum);
    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), ce_idx);
    builder.CreateBr(ce_loop);

    builder.SetInsertPoint(ce_done);
    llvm::Value* total_ce = builder.CreateLoad(ctx_.doubleType(), ce_sum);
    llvm::Value* neg_ce = builder.CreateFNeg(total_ce);

    return tagged_.packDouble(neg_ce);
}

llvm::Value* TensorCodegen::bceLoss(const eshkol_operations_t* op) {
    // bce-loss: (bce-loss predictions targets)
    // BCE = -sum(target * log(pred) + (1-target) * log(1-pred)) / n
    if (op->call_op.num_vars < 2) {
        eshkol_error("bce-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "bce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "bce_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-4); // For numerical stability (JAX/PyTorch standard)

    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* bce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), bce_sum);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    // Clamp predictions for numerical stability
    llvm::Value* pred_clamp = builder.CreateSelect(
        builder.CreateFCmpOLT(pred_val, eps), eps,
        builder.CreateSelect(
            builder.CreateFCmpOGT(pred_val, builder.CreateFSub(one, eps)),
            builder.CreateFSub(one, eps), pred_val));

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // BCE term: target * log(pred) + (1-target) * log(1-pred)
    llvm::Value* log_pred = builder.CreateCall(log_func, {pred_clamp});
    llvm::Value* one_minus_pred = builder.CreateFSub(one, pred_clamp);
    llvm::Value* log_one_minus_pred = builder.CreateCall(log_func, {one_minus_pred});
    llvm::Value* one_minus_target = builder.CreateFSub(one, target_val);

    llvm::Value* term1 = builder.CreateFMul(target_val, log_pred);
    llvm::Value* term2 = builder.CreateFMul(one_minus_target, log_one_minus_pred);
    llvm::Value* term = builder.CreateFAdd(term1, term2);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, term), bce_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* bce = builder.CreateFDiv(builder.CreateFNeg(total_sum), n_double);

    return tagged_.packDouble(bce);
}

llvm::Value* TensorCodegen::huberLoss(const eshkol_operations_t* op) {
    // huber-loss: (huber-loss predictions targets [delta])
    // L_δ(a) = 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
    if (op->call_op.num_vars < 2) {
        eshkol_error("huber-loss requires at least 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::Value* delta = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    if (op->call_op.num_vars >= 3) {
        llvm::Value* delta_tagged = codegenAST(&op->call_op.variables[2]);
        if (delta_tagged) delta = tagged_.unpackDouble(delta_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    llvm::Function* fabs_intrinsic = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::fabs, {ctx_.doubleType()});

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "huber_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "huber_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "huber_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "huber_sum_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpULT(si, num_dims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);

    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* huber_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), huber_sum);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sui, num_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());

    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Value* abs_diff = builder.CreateCall(fabs_intrinsic, {diff});

    // Huber: 0.5 * a² if |a| ≤ δ, else δ * (|a| - 0.5 * δ)
    llvm::Value* is_small = builder.CreateFCmpOLE(abs_diff, delta);
    llvm::Value* quadratic = builder.CreateFMul(half, builder.CreateFMul(diff, diff));
    llvm::Value* linear = builder.CreateFMul(delta, builder.CreateFSub(abs_diff, builder.CreateFMul(half, delta)));
    llvm::Value* loss_term = builder.CreateSelect(is_small, quadratic, linear);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), huber_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, loss_term), huber_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), huber_sum);
    llvm::Value* n_double = builder.CreateSIToFP(num_elements, ctx_.doubleType());
    llvm::Value* huber = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(huber);
}

llvm::Value* TensorCodegen::maeLoss(const eshkol_operations_t* op) {
    // mae-loss: (mae-loss predictions targets)
    // MAE = (1/n) * sum(|pred - target|)
    if (op->call_op.num_vars != 2) {
        eshkol_error("mae-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get prediction and target tensors
    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    // Get tensor layouts (index 0 = dims ptr, index 1 = ndims, index 2 = elems)
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);

    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Calculate total number of elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "mae_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "mae_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "mae_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "mae_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpSLT(si, pred_ndims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);

    // Sum absolute differences
    llvm::Value* mae_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), mae_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    llvm::Value* sum_cond = builder.CreateICmpSLT(sui, total_elements);
    builder.CreateCondBr(sum_cond, sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // |pred - target|
    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Function* fabs_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});
    llvm::Value* abs_diff = builder.CreateCall(fabs_func, {diff});

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), mae_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, abs_diff), mae_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), mae_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mae = builder.CreateFDiv(total_sum, n_double);

    return tagged_.packDouble(mae);
}

llvm::Value* TensorCodegen::binaryCrossEntropyLoss(const eshkol_operations_t* op) {
    // binary-cross-entropy-loss: (binary-cross-entropy-loss predictions targets)
    // BCE = -(1/n) * sum(target * log(pred) + (1-target) * log(1-pred))
    if (op->call_op.num_vars != 2) {
        eshkol_error("binary-cross-entropy-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Get prediction and target tensors
    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    // Get tensor layouts (index 0 = dims ptr, index 1 = ndims, index 2 = elems)
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);

    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Calculate total number of elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "bce_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "bce_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "bce_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpSLT(si, pred_ndims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);

    // Sum BCE terms with numerical stability
    // Use clipping: pred_clipped = max(eps, min(1-eps, pred))
    llvm::Value* bce_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), bce_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    llvm::Value* sum_cond = builder.CreateICmpSLT(sui, total_elements);
    builder.CreateCondBr(sum_cond, sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // Clip predictions for numerical stability
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_eps = builder.CreateFSub(one, eps);

    // pred_clipped = max(eps, min(1-eps, pred))
    llvm::Value* cmp_high = builder.CreateFCmpOLT(pred_val, one_minus_eps);
    llvm::Value* pred_upper = builder.CreateSelect(cmp_high, pred_val, one_minus_eps);
    llvm::Value* cmp_low = builder.CreateFCmpOGT(pred_upper, eps);
    llvm::Value* pred_clipped = builder.CreateSelect(cmp_low, pred_upper, eps);

    // BCE term: target * log(pred) + (1-target) * log(1-pred)
    llvm::Function* log_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::log, {ctx_.doubleType()});

    llvm::Value* log_pred = builder.CreateCall(log_func, {pred_clipped});
    llvm::Value* one_minus_pred = builder.CreateFSub(one, pred_clipped);
    llvm::Value* log_one_minus_pred = builder.CreateCall(log_func, {one_minus_pred});

    llvm::Value* term1 = builder.CreateFMul(target_val, log_pred);
    llvm::Value* one_minus_target = builder.CreateFSub(one, target_val);
    llvm::Value* term2 = builder.CreateFMul(one_minus_target, log_one_minus_pred);
    llvm::Value* bce_term = builder.CreateFAdd(term1, term2);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, bce_term), bce_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), bce_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* mean_bce = builder.CreateFDiv(total_sum, n_double);
    // Negate to get the loss (BCE is negative log likelihood)
    llvm::Value* bce = builder.CreateFNeg(mean_bce);

    return tagged_.packDouble(bce);
}

llvm::Value* TensorCodegen::klDivLoss(const eshkol_operations_t* op) {
    // KL Divergence: KL(P || Q) = sum(P * log(P / Q))
    // Args: P (true distribution), Q (predicted distribution)
    // Both must be probability distributions (non-negative, sum to 1)
    // Returns: scalar KL divergence
    if (op->call_op.num_vars != 2) {
        eshkol_error("kl-div-loss requires 2 arguments: P (true), Q (predicted)");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* p_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* q_tagged = codegenAST(&op->call_op.variables[1]);
    if (!p_tagged || !q_tagged) return nullptr;

    llvm::Value* p_ptr = tagged_.unpackPtr(p_tagged);
    llvm::Value* q_ptr = tagged_.unpackPtr(q_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* p_dims_ptr = builder.CreateStructGEP(tensor_type, p_ptr, 0);
    llvm::Value* p_dims = builder.CreateLoad(ctx_.ptrType(), p_dims_ptr);
    llvm::Value* p_ndims_ptr = builder.CreateStructGEP(tensor_type, p_ptr, 1);
    llvm::Value* p_ndims = builder.CreateLoad(ctx_.int64Type(), p_ndims_ptr);
    llvm::Value* p_elems_ptr = builder.CreateStructGEP(tensor_type, p_ptr, 2);
    llvm::Value* p_elems = builder.CreateLoad(ctx_.ptrType(), p_elems_ptr);
    llvm::Value* q_elems_ptr = builder.CreateStructGEP(tensor_type, q_ptr, 2);
    llvm::Value* q_elems = builder.CreateLoad(ctx_.ptrType(), q_elems_ptr);

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "kl_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "kl_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "kl_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "kl_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "kl_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "kl_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(si, p_ndims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), p_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);

    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }

    // Sum P * log(P / Q) with epsilon guard
    llvm::Value* kl_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), kl_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(sui, total_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* p_elem_ptr = builder.CreateGEP(ctx_.int64Type(), p_elems, sui);
    llvm::Value* p_bits = builder.CreateLoad(ctx_.int64Type(), p_elem_ptr);
    llvm::Value* p_val = builder.CreateBitCast(p_bits, ctx_.doubleType());
    llvm::Value* q_elem_ptr = builder.CreateGEP(ctx_.int64Type(), q_elems, sui);
    llvm::Value* q_bits = builder.CreateLoad(ctx_.int64Type(), q_elem_ptr);
    llvm::Value* q_val = builder.CreateBitCast(q_bits, ctx_.doubleType());

    // Clamp Q to avoid log(0): max(Q, epsilon)
    llvm::Value* epsilon = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* q_clamped_cmp = builder.CreateFCmpOGT(q_val, epsilon);
    llvm::Value* q_safe = builder.CreateSelect(q_clamped_cmp, q_val, epsilon);

    // Only contribute when P > 0 (avoid 0 * log(0) = NaN)
    llvm::Value* p_positive = builder.CreateFCmpOGT(p_val, llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    // P * log(P / Q) = P * (log(P) - log(Q))
    llvm::Value* p_safe_cmp = builder.CreateFCmpOGT(p_val, epsilon);
    llvm::Value* p_safe = builder.CreateSelect(p_safe_cmp, p_val, epsilon);
    llvm::Value* log_p = builder.CreateCall(log_func, {p_safe});
    llvm::Value* log_q = builder.CreateCall(log_func, {q_safe});
    llvm::Value* log_ratio = builder.CreateFSub(log_p, log_q);
    llvm::Value* kl_term = builder.CreateFMul(p_val, log_ratio);
    // Zero out contribution if P <= 0
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* safe_term = builder.CreateSelect(p_positive, kl_term, zero);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), kl_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, safe_term), kl_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), kl_sum);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::hingeLoss(const eshkol_operations_t* op) {
    // Hinge Loss: (1/n) * sum(max(0, 1 - y * f(x)))
    // Args: predictions (f(x)), targets (y, should be +1 or -1)
    if (op->call_op.num_vars != 2) {
        eshkol_error("hinge-loss requires 2 arguments: predictions, targets");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "hinge_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "hinge_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "hinge_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "hinge_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "hinge_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "hinge_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(si, pred_ndims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);

    llvm::Value* hinge_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), hinge_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(sui, total_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // max(0, 1 - y * f(x))
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* y_fx = builder.CreateFMul(target_val, pred_val);
    llvm::Value* margin = builder.CreateFSub(one, y_fx);
    llvm::Value* cmp = builder.CreateFCmpOGT(margin, zero);
    llvm::Value* hinge_val = builder.CreateSelect(cmp, margin, zero);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), hinge_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, hinge_val), hinge_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), hinge_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* result = builder.CreateFDiv(total_sum, n_double);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::smoothL1Loss(const eshkol_operations_t* op) {
    // Smooth L1 (Huber variant):
    // if |x| < beta: 0.5 * x^2 / beta
    // else: |x| - 0.5 * beta
    // Default beta = 1.0 (reduces to standard Huber loss)
    // Args: predictions, targets [, beta]
    if (op->call_op.num_vars < 2 || op->call_op.num_vars > 3) {
        eshkol_error("smooth-l1-loss requires 2 or 3 arguments: predictions, targets [, beta]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* beta;
    if (op->call_op.num_vars == 3) {
        llvm::Value* beta_tagged = codegenAST(&op->call_op.variables[2]);
        if (!beta_tagged) return nullptr;
        beta = tagged_.unpackDouble(beta_tagged);
    } else {
        beta = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "sl1_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "sl1_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "sl1_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "sl1_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "sl1_sum_body", current_func);
    llvm::BasicBlock* sl1_quad = llvm::BasicBlock::Create(ctx_.context(), "sl1_quad", current_func);
    llvm::BasicBlock* sl1_linear = llvm::BasicBlock::Create(ctx_.context(), "sl1_linear", current_func);
    llvm::BasicBlock* sl1_merge = llvm::BasicBlock::Create(ctx_.context(), "sl1_merge", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "sl1_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(si, pred_ndims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);

    llvm::Function* fabs_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::fabs, {ctx_.doubleType()});

    llvm::Value* loss_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), loss_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(sui, total_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(pred_val, target_val);
    llvm::Value* abs_diff = builder.CreateCall(fabs_func, {diff});
    llvm::Value* is_small = builder.CreateFCmpOLT(abs_diff, beta);
    builder.CreateCondBr(is_small, sl1_quad, sl1_linear);

    // Quadratic region: 0.5 * diff^2 / beta
    builder.SetInsertPoint(sl1_quad);
    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* diff_sq = builder.CreateFMul(diff, diff);
    llvm::Value* quad_val = builder.CreateFMul(half, builder.CreateFDiv(diff_sq, beta));
    builder.CreateBr(sl1_merge);

    // Linear region: |diff| - 0.5 * beta
    builder.SetInsertPoint(sl1_linear);
    llvm::Value* half_beta = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 0.5), beta);
    llvm::Value* linear_val = builder.CreateFSub(abs_diff, half_beta);
    builder.CreateBr(sl1_merge);

    builder.SetInsertPoint(sl1_merge);
    llvm::PHINode* loss_val = builder.CreatePHI(ctx_.doubleType(), 2, "sl1_val");
    loss_val->addIncoming(quad_val, sl1_quad);
    loss_val->addIncoming(linear_val, sl1_linear);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), loss_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, loss_val), loss_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), loss_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* result = builder.CreateFDiv(total_sum, n_double);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::focalLoss(const eshkol_operations_t* op) {
    // Focal Loss: -(1 - p_t)^gamma * log(p_t)
    // where p_t = p if target=1, (1-p) if target=0
    // Default gamma = 2.0 (RetinaNet default)
    // Args: predictions (probabilities), targets (0 or 1) [, gamma]
    if (op->call_op.num_vars < 2 || op->call_op.num_vars > 3) {
        eshkol_error("focal-loss requires 2 or 3 arguments: predictions, targets [, gamma]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* pred_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* target_tagged = codegenAST(&op->call_op.variables[1]);
    if (!pred_tagged || !target_tagged) return nullptr;

    llvm::Value* gamma;
    if (op->call_op.num_vars == 3) {
        llvm::Value* gamma_tagged = codegenAST(&op->call_op.variables[2]);
        if (!gamma_tagged) return nullptr;
        gamma = tagged_.unpackDouble(gamma_tagged);
    } else {
        gamma = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    }

    llvm::Value* pred_ptr = tagged_.unpackPtr(pred_tagged);
    llvm::Value* target_ptr = tagged_.unpackPtr(target_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* pred_dims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 0);
    llvm::Value* pred_dims = builder.CreateLoad(ctx_.ptrType(), pred_dims_ptr);
    llvm::Value* pred_ndims_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 1);
    llvm::Value* pred_ndims = builder.CreateLoad(ctx_.int64Type(), pred_ndims_ptr);
    llvm::Value* pred_elems_ptr = builder.CreateStructGEP(tensor_type, pred_ptr, 2);
    llvm::Value* pred_elems = builder.CreateLoad(ctx_.ptrType(), pred_elems_ptr);
    llvm::Value* target_elems_ptr = builder.CreateStructGEP(tensor_type, target_ptr, 2);
    llvm::Value* target_elems = builder.CreateLoad(ctx_.ptrType(), target_elems_ptr);

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "focal_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "focal_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "focal_size_done", current_func);
    llvm::BasicBlock* sum_loop = llvm::BasicBlock::Create(ctx_.context(), "focal_sum_loop", current_func);
    llvm::BasicBlock* sum_body = llvm::BasicBlock::Create(ctx_.context(), "focal_sum_body", current_func);
    llvm::BasicBlock* sum_done = llvm::BasicBlock::Create(ctx_.context(), "focal_sum_done", current_func);

    llvm::Value* num_elements = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), num_elements);
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(si, pred_ndims), size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), pred_dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), num_elements);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), num_elements);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), num_elements);

    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pow", &ctx_.module());
    }

    llvm::Value* focal_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), focal_sum);
    llvm::Value* sum_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_loop);
    llvm::Value* sui = builder.CreateLoad(ctx_.int64Type(), sum_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(sui, total_elements), sum_body, sum_done);

    builder.SetInsertPoint(sum_body);
    llvm::Value* pred_elem_ptr = builder.CreateGEP(ctx_.int64Type(), pred_elems, sui);
    llvm::Value* pred_bits = builder.CreateLoad(ctx_.int64Type(), pred_elem_ptr);
    llvm::Value* pred_val = builder.CreateBitCast(pred_bits, ctx_.doubleType());
    llvm::Value* target_elem_ptr = builder.CreateGEP(ctx_.int64Type(), target_elems, sui);
    llvm::Value* target_bits = builder.CreateLoad(ctx_.int64Type(), target_elem_ptr);
    llvm::Value* target_val = builder.CreateBitCast(target_bits, ctx_.doubleType());

    // Clamp prediction to [epsilon, 1-epsilon]
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-7);
    llvm::Value* one_minus_eps = llvm::ConstantFP::get(ctx_.doubleType(), 1.0 - 1e-7);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* cmp_low = builder.CreateFCmpOGT(pred_val, eps);
    llvm::Value* p_clamped = builder.CreateSelect(cmp_low, pred_val, eps);
    llvm::Value* cmp_high = builder.CreateFCmpOLT(p_clamped, one_minus_eps);
    llvm::Value* p_safe = builder.CreateSelect(cmp_high, p_clamped, one_minus_eps);

    // p_t = target * p + (1 - target) * (1 - p)
    llvm::Value* one_minus_t = builder.CreateFSub(one, target_val);
    llvm::Value* one_minus_p = builder.CreateFSub(one, p_safe);
    llvm::Value* pt_pos = builder.CreateFMul(target_val, p_safe);
    llvm::Value* pt_neg = builder.CreateFMul(one_minus_t, one_minus_p);
    llvm::Value* pt = builder.CreateFAdd(pt_pos, pt_neg);

    // -(1 - p_t)^gamma * log(p_t)
    llvm::Value* one_minus_pt = builder.CreateFSub(one, pt);
    llvm::Value* modulating = builder.CreateCall(pow_func, {one_minus_pt, gamma});
    llvm::Value* log_pt = builder.CreateCall(log_func, {pt});
    llvm::Value* focal_term = builder.CreateFMul(modulating, log_pt);
    llvm::Value* neg_focal = builder.CreateFNeg(focal_term);

    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), focal_sum);
    builder.CreateStore(builder.CreateFAdd(cur_sum, neg_focal), focal_sum);
    builder.CreateStore(builder.CreateAdd(sui, llvm::ConstantInt::get(ctx_.int64Type(), 1)), sum_idx);
    builder.CreateBr(sum_loop);

    builder.SetInsertPoint(sum_done);
    llvm::Value* total_sum = builder.CreateLoad(ctx_.doubleType(), focal_sum);
    llvm::Value* n_double = builder.CreateSIToFP(total_elements, ctx_.doubleType());
    llvm::Value* result = builder.CreateFDiv(total_sum, n_double);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::tripletLoss(const eshkol_operations_t* op) {
    // Triplet Loss: max(d(anchor, positive) - d(anchor, negative) + margin, 0)
    // Args: anchor tensor, positive tensor, negative tensor [, margin]
    // Default margin = 1.0
    // Uses L2 (Euclidean) distance
    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("triplet-loss requires 3 or 4 arguments: anchor, positive, negative [, margin]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* anchor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* positive_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* negative_tagged = codegenAST(&op->call_op.variables[2]);
    if (!anchor_tagged || !positive_tagged || !negative_tagged) return nullptr;

    llvm::Value* margin;
    if (op->call_op.num_vars == 4) {
        llvm::Value* margin_tagged = codegenAST(&op->call_op.variables[3]);
        if (!margin_tagged) return nullptr;
        margin = tagged_.unpackDouble(margin_tagged);
    } else {
        margin = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* anchor_ptr = tagged_.unpackPtr(anchor_tagged);
    llvm::Value* pos_ptr = tagged_.unpackPtr(positive_tagged);
    llvm::Value* neg_ptr = tagged_.unpackPtr(negative_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    // Get anchor total elements from field 3
    llvm::Value* anchor_total_ptr = builder.CreateStructGEP(tensor_type, anchor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), anchor_total_ptr);
    llvm::Value* anchor_elems_ptr = builder.CreateStructGEP(tensor_type, anchor_ptr, 2);
    llvm::Value* anchor_elems = builder.CreateLoad(ctx_.ptrType(), anchor_elems_ptr);
    llvm::Value* pos_elems_ptr = builder.CreateStructGEP(tensor_type, pos_ptr, 2);
    llvm::Value* pos_elems = builder.CreateLoad(ctx_.ptrType(), pos_elems_ptr);
    llvm::Value* neg_elems_ptr = builder.CreateStructGEP(tensor_type, neg_ptr, 2);
    llvm::Value* neg_elems = builder.CreateLoad(ctx_.ptrType(), neg_elems_ptr);

    // Compute squared L2 distances: d(a,p)^2 and d(a,n)^2
    llvm::BasicBlock* dist_loop = llvm::BasicBlock::Create(ctx_.context(), "trip_dist_loop", current_func);
    llvm::BasicBlock* dist_body = llvm::BasicBlock::Create(ctx_.context(), "trip_dist_body", current_func);
    llvm::BasicBlock* dist_done = llvm::BasicBlock::Create(ctx_.context(), "trip_dist_done", current_func);

    llvm::Value* dist_pos_sq = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* dist_neg_sq = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dist_pos_sq);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dist_neg_sq);
    llvm::Value* dist_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dist_idx);
    builder.CreateBr(dist_loop);

    builder.SetInsertPoint(dist_loop);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dist_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(di, total_elements), dist_body, dist_done);

    builder.SetInsertPoint(dist_body);
    // Load anchor, positive, negative elements
    llvm::Value* a_ptr = builder.CreateGEP(ctx_.int64Type(), anchor_elems, di);
    llvm::Value* a_bits = builder.CreateLoad(ctx_.int64Type(), a_ptr);
    llvm::Value* a_val = builder.CreateBitCast(a_bits, ctx_.doubleType());
    llvm::Value* p_ptr = builder.CreateGEP(ctx_.int64Type(), pos_elems, di);
    llvm::Value* p_bits = builder.CreateLoad(ctx_.int64Type(), p_ptr);
    llvm::Value* p_val = builder.CreateBitCast(p_bits, ctx_.doubleType());
    llvm::Value* n_ptr = builder.CreateGEP(ctx_.int64Type(), neg_elems, di);
    llvm::Value* n_bits = builder.CreateLoad(ctx_.int64Type(), n_ptr);
    llvm::Value* n_val = builder.CreateBitCast(n_bits, ctx_.doubleType());

    // d(a,p)^2 += (a - p)^2
    llvm::Value* diff_ap = builder.CreateFSub(a_val, p_val);
    llvm::Value* sq_ap = builder.CreateFMul(diff_ap, diff_ap);
    llvm::Value* cur_dp = builder.CreateLoad(ctx_.doubleType(), dist_pos_sq);
    builder.CreateStore(builder.CreateFAdd(cur_dp, sq_ap), dist_pos_sq);

    // d(a,n)^2 += (a - n)^2
    llvm::Value* diff_an = builder.CreateFSub(a_val, n_val);
    llvm::Value* sq_an = builder.CreateFMul(diff_an, diff_an);
    llvm::Value* cur_dn = builder.CreateLoad(ctx_.doubleType(), dist_neg_sq);
    builder.CreateStore(builder.CreateFAdd(cur_dn, sq_an), dist_neg_sq);

    builder.CreateStore(builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dist_idx);
    builder.CreateBr(dist_loop);

    builder.SetInsertPoint(dist_done);
    // sqrt distances
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* dp_sq_final = builder.CreateLoad(ctx_.doubleType(), dist_pos_sq);
    llvm::Value* dn_sq_final = builder.CreateLoad(ctx_.doubleType(), dist_neg_sq);
    llvm::Value* dist_pos = builder.CreateCall(sqrt_func, {dp_sq_final});
    llvm::Value* dist_neg = builder.CreateCall(sqrt_func, {dn_sq_final});

    // max(d_pos - d_neg + margin, 0)
    llvm::Value* diff_dist = builder.CreateFSub(dist_pos, dist_neg);
    llvm::Value* with_margin = builder.CreateFAdd(diff_dist, margin);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* cmp = builder.CreateFCmpOGT(with_margin, zero);
    llvm::Value* result = builder.CreateSelect(cmp, with_margin, zero);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::contrastiveLoss(const eshkol_operations_t* op) {
    // Contrastive Loss: (1-y)*d^2 + y*max(margin-d, 0)^2
    // Args: tensor1, tensor2, labels (0=similar, 1=dissimilar) [, margin]
    // Default margin = 1.0
    // d = Euclidean distance between tensor1 and tensor2
    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("contrastive-loss requires 3 or 4 arguments: tensor1, tensor2, labels [, margin]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* t1_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* t2_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* label_tagged = codegenAST(&op->call_op.variables[2]);
    if (!t1_tagged || !t2_tagged || !label_tagged) return nullptr;

    llvm::Value* margin;
    if (op->call_op.num_vars == 4) {
        llvm::Value* margin_tagged = codegenAST(&op->call_op.variables[3]);
        if (!margin_tagged) return nullptr;
        margin = tagged_.unpackDouble(margin_tagged);
    } else {
        margin = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    }

    llvm::Value* y = tagged_.unpackDouble(label_tagged);

    llvm::Value* t1_ptr = tagged_.unpackPtr(t1_tagged);
    llvm::Value* t2_ptr = tagged_.unpackPtr(t2_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* t1_total_ptr = builder.CreateStructGEP(tensor_type, t1_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), t1_total_ptr);
    llvm::Value* t1_elems_ptr = builder.CreateStructGEP(tensor_type, t1_ptr, 2);
    llvm::Value* t1_elems = builder.CreateLoad(ctx_.ptrType(), t1_elems_ptr);
    llvm::Value* t2_elems_ptr = builder.CreateStructGEP(tensor_type, t2_ptr, 2);
    llvm::Value* t2_elems = builder.CreateLoad(ctx_.ptrType(), t2_elems_ptr);

    // Compute squared L2 distance
    llvm::BasicBlock* dist_loop = llvm::BasicBlock::Create(ctx_.context(), "contr_dist_loop", current_func);
    llvm::BasicBlock* dist_body = llvm::BasicBlock::Create(ctx_.context(), "contr_dist_body", current_func);
    llvm::BasicBlock* dist_done = llvm::BasicBlock::Create(ctx_.context(), "contr_dist_done", current_func);

    llvm::Value* dist_sq = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dist_sq);
    llvm::Value* dist_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), dist_idx);
    builder.CreateBr(dist_loop);

    builder.SetInsertPoint(dist_loop);
    llvm::Value* di = builder.CreateLoad(ctx_.int64Type(), dist_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(di, total_elements), dist_body, dist_done);

    builder.SetInsertPoint(dist_body);
    llvm::Value* e1_ptr = builder.CreateGEP(ctx_.int64Type(), t1_elems, di);
    llvm::Value* e1_bits = builder.CreateLoad(ctx_.int64Type(), e1_ptr);
    llvm::Value* e1_val = builder.CreateBitCast(e1_bits, ctx_.doubleType());
    llvm::Value* e2_ptr = builder.CreateGEP(ctx_.int64Type(), t2_elems, di);
    llvm::Value* e2_bits = builder.CreateLoad(ctx_.int64Type(), e2_ptr);
    llvm::Value* e2_val = builder.CreateBitCast(e2_bits, ctx_.doubleType());

    llvm::Value* diff = builder.CreateFSub(e1_val, e2_val);
    llvm::Value* sq = builder.CreateFMul(diff, diff);
    llvm::Value* cur_d = builder.CreateLoad(ctx_.doubleType(), dist_sq);
    builder.CreateStore(builder.CreateFAdd(cur_d, sq), dist_sq);

    builder.CreateStore(builder.CreateAdd(di, llvm::ConstantInt::get(ctx_.int64Type(), 1)), dist_idx);
    builder.CreateBr(dist_loop);

    builder.SetInsertPoint(dist_done);
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* d_sq_final = builder.CreateLoad(ctx_.doubleType(), dist_sq);
    llvm::Value* d = builder.CreateCall(sqrt_func, {d_sq_final});

    // (1 - y) * d^2 + y * max(margin - d, 0)^2
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* one_minus_y = builder.CreateFSub(one, y);
    llvm::Value* similar_term = builder.CreateFMul(one_minus_y, d_sq_final);

    llvm::Value* margin_minus_d = builder.CreateFSub(margin, d);
    llvm::Value* cmp = builder.CreateFCmpOGT(margin_minus_d, zero);
    llvm::Value* clamped = builder.CreateSelect(cmp, margin_minus_d, zero);
    llvm::Value* clamped_sq = builder.CreateFMul(clamped, clamped);
    llvm::Value* dissimilar_term = builder.CreateFMul(y, clamped_sq);

    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* result = builder.CreateFMul(half, builder.CreateFAdd(similar_term, dissimilar_term));
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::labelSmoothingLoss(const eshkol_operations_t* op) {
    // Label Smoothing Cross-Entropy: CE with softened targets
    // smoothed_target = (1 - epsilon) * one_hot + epsilon / num_classes
    // Loss = -sum(smoothed_target * log(softmax(logits)))
    // Args: logits, targets (class indices or one-hot), num_classes [, epsilon]
    // Default epsilon = 0.1
    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("label-smoothing-loss requires 3 or 4 arguments: logits, targets, num_classes [, epsilon]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* logits_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* targets_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* nclasses_tagged = codegenAST(&op->call_op.variables[2]);
    if (!logits_tagged || !targets_tagged || !nclasses_tagged) return nullptr;

    llvm::Value* epsilon;
    if (op->call_op.num_vars == 4) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[3]);
        if (!eps_tagged) return nullptr;
        epsilon = tagged_.unpackDouble(eps_tagged);
    } else {
        epsilon = llvm::ConstantFP::get(ctx_.doubleType(), 0.1);
    }

    llvm::Value* num_classes_double = tagged_.unpackDouble(nclasses_tagged);

    llvm::Value* logits_ptr = tagged_.unpackPtr(logits_tagged);
    llvm::Value* targets_ptr = tagged_.unpackPtr(targets_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* logits_total_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), logits_total_ptr);
    llvm::Value* logits_elems_ptr = builder.CreateStructGEP(tensor_type, logits_ptr, 2);
    llvm::Value* logits_elems = builder.CreateLoad(ctx_.ptrType(), logits_elems_ptr);
    llvm::Value* targets_elems_ptr = builder.CreateStructGEP(tensor_type, targets_ptr, 2);
    llvm::Value* targets_elems = builder.CreateLoad(ctx_.ptrType(), targets_elems_ptr);

    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* exp_func = ctx_.module().getFunction("exp");
    if (!exp_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        exp_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "exp", &ctx_.module());
    }

    // Compute logsumexp for log-softmax stability
    // First pass: find max for numerical stability
    llvm::BasicBlock* max_loop = llvm::BasicBlock::Create(ctx_.context(), "ls_max_loop", current_func);
    llvm::BasicBlock* max_body = llvm::BasicBlock::Create(ctx_.context(), "ls_max_body", current_func);
    llvm::BasicBlock* max_done = llvm::BasicBlock::Create(ctx_.context(), "ls_max_done", current_func);
    llvm::BasicBlock* lse_loop = llvm::BasicBlock::Create(ctx_.context(), "ls_lse_loop", current_func);
    llvm::BasicBlock* lse_body = llvm::BasicBlock::Create(ctx_.context(), "ls_lse_body", current_func);
    llvm::BasicBlock* lse_done = llvm::BasicBlock::Create(ctx_.context(), "ls_lse_done", current_func);
    llvm::BasicBlock* loss_loop = llvm::BasicBlock::Create(ctx_.context(), "ls_loss_loop", current_func);
    llvm::BasicBlock* loss_body = llvm::BasicBlock::Create(ctx_.context(), "ls_loss_body", current_func);
    llvm::BasicBlock* loss_done = llvm::BasicBlock::Create(ctx_.context(), "ls_loss_done", current_func);

    // Find max logit
    llvm::Value* max_val = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), -1e308), max_val);
    llvm::Value* max_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), max_idx);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_loop);
    llvm::Value* mi = builder.CreateLoad(ctx_.int64Type(), max_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(mi, total_elements), max_body, max_done);

    builder.SetInsertPoint(max_body);
    llvm::Value* l_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, mi);
    llvm::Value* l_bits = builder.CreateLoad(ctx_.int64Type(), l_ptr);
    llvm::Value* l_val = builder.CreateBitCast(l_bits, ctx_.doubleType());
    llvm::Value* cur_max = builder.CreateLoad(ctx_.doubleType(), max_val);
    llvm::Value* is_greater = builder.CreateFCmpOGT(l_val, cur_max);
    llvm::Value* new_max = builder.CreateSelect(is_greater, l_val, cur_max);
    builder.CreateStore(new_max, max_val);
    builder.CreateStore(builder.CreateAdd(mi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), max_idx);
    builder.CreateBr(max_loop);

    builder.SetInsertPoint(max_done);
    llvm::Value* logit_max = builder.CreateLoad(ctx_.doubleType(), max_val);

    // Compute logsumexp = max + log(sum(exp(x - max)))
    llvm::Value* exp_sum = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), exp_sum);
    llvm::Value* lse_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), lse_idx);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_loop);
    llvm::Value* li = builder.CreateLoad(ctx_.int64Type(), lse_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(li, total_elements), lse_body, lse_done);

    builder.SetInsertPoint(lse_body);
    llvm::Value* lse_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, li);
    llvm::Value* lse_bits = builder.CreateLoad(ctx_.int64Type(), lse_ptr);
    llvm::Value* lse_val = builder.CreateBitCast(lse_bits, ctx_.doubleType());
    llvm::Value* shifted = builder.CreateFSub(lse_val, logit_max);
    llvm::Value* exp_shifted = builder.CreateCall(exp_func, {shifted});
    llvm::Value* cur_exp_sum = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    builder.CreateStore(builder.CreateFAdd(cur_exp_sum, exp_shifted), exp_sum);
    builder.CreateStore(builder.CreateAdd(li, llvm::ConstantInt::get(ctx_.int64Type(), 1)), lse_idx);
    builder.CreateBr(lse_loop);

    builder.SetInsertPoint(lse_done);
    llvm::Value* total_exp = builder.CreateLoad(ctx_.doubleType(), exp_sum);
    llvm::Value* log_sum_exp = builder.CreateFAdd(logit_max, builder.CreateCall(log_func, {total_exp}));

    // Compute loss: -sum(smoothed_target * log_softmax)
    // log_softmax(x_i) = x_i - logsumexp
    // smoothed_target_i = (1-eps)*target_i + eps/K
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* one_minus_eps = builder.CreateFSub(one, epsilon);
    llvm::Value* eps_over_k = builder.CreateFDiv(epsilon, num_classes_double);

    llvm::Value* loss_acc = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), loss_acc);
    llvm::Value* loss_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), loss_idx);
    builder.CreateBr(loss_loop);

    builder.SetInsertPoint(loss_loop);
    llvm::Value* loi = builder.CreateLoad(ctx_.int64Type(), loss_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(loi, total_elements), loss_body, loss_done);

    builder.SetInsertPoint(loss_body);
    llvm::Value* logit_ptr = builder.CreateGEP(ctx_.int64Type(), logits_elems, loi);
    llvm::Value* logit_bits = builder.CreateLoad(ctx_.int64Type(), logit_ptr);
    llvm::Value* logit = builder.CreateBitCast(logit_bits, ctx_.doubleType());
    llvm::Value* target_ptr2 = builder.CreateGEP(ctx_.int64Type(), targets_elems, loi);
    llvm::Value* target_bits2 = builder.CreateLoad(ctx_.int64Type(), target_ptr2);
    llvm::Value* target_oh = builder.CreateBitCast(target_bits2, ctx_.doubleType());

    // log_softmax = logit - logsumexp
    llvm::Value* log_sm = builder.CreateFSub(logit, log_sum_exp);
    // smoothed_target = (1-eps)*target + eps/K
    llvm::Value* smooth_t = builder.CreateFAdd(
        builder.CreateFMul(one_minus_eps, target_oh), eps_over_k);
    // -smoothed_target * log_softmax
    llvm::Value* term = builder.CreateFMul(smooth_t, log_sm);
    llvm::Value* neg_term = builder.CreateFNeg(term);
    llvm::Value* cur_loss = builder.CreateLoad(ctx_.doubleType(), loss_acc);
    builder.CreateStore(builder.CreateFAdd(cur_loss, neg_term), loss_acc);
    builder.CreateStore(builder.CreateAdd(loi, llvm::ConstantInt::get(ctx_.int64Type(), 1)), loss_idx);
    builder.CreateBr(loss_loop);

    builder.SetInsertPoint(loss_done);
    llvm::Value* result = builder.CreateLoad(ctx_.doubleType(), loss_acc);
    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::cosineEmbeddingLoss(const eshkol_operations_t* op) {
    // Cosine Embedding Loss:
    // if y == 1:  1 - cos(x1, x2)
    // if y == -1: max(0, cos(x1, x2) - margin)
    // Args: tensor1, tensor2, label (+1 or -1) [, margin]
    // Default margin = 0.0
    if (op->call_op.num_vars < 3 || op->call_op.num_vars > 4) {
        eshkol_error("cosine-embedding-loss requires 3 or 4 arguments: tensor1, tensor2, label [, margin]");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    llvm::Value* t1_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* t2_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* label_tagged = codegenAST(&op->call_op.variables[2]);
    if (!t1_tagged || !t2_tagged || !label_tagged) return nullptr;

    llvm::Value* margin;
    if (op->call_op.num_vars == 4) {
        llvm::Value* margin_tagged = codegenAST(&op->call_op.variables[3]);
        if (!margin_tagged) return nullptr;
        margin = tagged_.unpackDouble(margin_tagged);
    } else {
        margin = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    }

    llvm::Value* y = tagged_.unpackDouble(label_tagged);

    llvm::Value* t1_ptr = tagged_.unpackPtr(t1_tagged);
    llvm::Value* t2_ptr = tagged_.unpackPtr(t2_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* t1_total_ptr = builder.CreateStructGEP(tensor_type, t1_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), t1_total_ptr);
    llvm::Value* t1_elems_ptr = builder.CreateStructGEP(tensor_type, t1_ptr, 2);
    llvm::Value* t1_elems = builder.CreateLoad(ctx_.ptrType(), t1_elems_ptr);
    llvm::Value* t2_elems_ptr = builder.CreateStructGEP(tensor_type, t2_ptr, 2);
    llvm::Value* t2_elems = builder.CreateLoad(ctx_.ptrType(), t2_elems_ptr);

    // Compute dot product and norms in single pass
    llvm::BasicBlock* cos_loop = llvm::BasicBlock::Create(ctx_.context(), "cos_loop", current_func);
    llvm::BasicBlock* cos_body = llvm::BasicBlock::Create(ctx_.context(), "cos_body", current_func);
    llvm::BasicBlock* cos_done = llvm::BasicBlock::Create(ctx_.context(), "cos_done", current_func);

    llvm::Value* dot_prod = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* norm1_sq = builder.CreateAlloca(ctx_.doubleType());
    llvm::Value* norm2_sq = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), dot_prod);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), norm1_sq);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), norm2_sq);
    llvm::Value* cos_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), cos_idx);
    builder.CreateBr(cos_loop);

    builder.SetInsertPoint(cos_loop);
    llvm::Value* ci = builder.CreateLoad(ctx_.int64Type(), cos_idx);
    builder.CreateCondBr(builder.CreateICmpSLT(ci, total_elements), cos_body, cos_done);

    builder.SetInsertPoint(cos_body);
    llvm::Value* e1_ptr = builder.CreateGEP(ctx_.int64Type(), t1_elems, ci);
    llvm::Value* e1_bits = builder.CreateLoad(ctx_.int64Type(), e1_ptr);
    llvm::Value* e1 = builder.CreateBitCast(e1_bits, ctx_.doubleType());
    llvm::Value* e2_ptr = builder.CreateGEP(ctx_.int64Type(), t2_elems, ci);
    llvm::Value* e2_bits = builder.CreateLoad(ctx_.int64Type(), e2_ptr);
    llvm::Value* e2 = builder.CreateBitCast(e2_bits, ctx_.doubleType());

    // Accumulate dot product and squared norms
    llvm::Value* prod = builder.CreateFMul(e1, e2);
    llvm::Value* cur_dot = builder.CreateLoad(ctx_.doubleType(), dot_prod);
    builder.CreateStore(builder.CreateFAdd(cur_dot, prod), dot_prod);
    llvm::Value* e1_sq = builder.CreateFMul(e1, e1);
    llvm::Value* cur_n1 = builder.CreateLoad(ctx_.doubleType(), norm1_sq);
    builder.CreateStore(builder.CreateFAdd(cur_n1, e1_sq), norm1_sq);
    llvm::Value* e2_sq = builder.CreateFMul(e2, e2);
    llvm::Value* cur_n2 = builder.CreateLoad(ctx_.doubleType(), norm2_sq);
    builder.CreateStore(builder.CreateFAdd(cur_n2, e2_sq), norm2_sq);

    builder.CreateStore(builder.CreateAdd(ci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), cos_idx);
    builder.CreateBr(cos_loop);

    builder.SetInsertPoint(cos_done);
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(
        &ctx_.module(), llvm::Intrinsic::sqrt, {ctx_.doubleType()});

    llvm::Value* final_dot = builder.CreateLoad(ctx_.doubleType(), dot_prod);
    llvm::Value* final_n1 = builder.CreateLoad(ctx_.doubleType(), norm1_sq);
    llvm::Value* final_n2 = builder.CreateLoad(ctx_.doubleType(), norm2_sq);
    llvm::Value* n1 = builder.CreateCall(sqrt_func, {final_n1});
    llvm::Value* n2 = builder.CreateCall(sqrt_func, {final_n2});
    llvm::Value* norm_product = builder.CreateFMul(n1, n2);

    // Guard against division by zero
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);
    llvm::Value* safe_denom_cmp = builder.CreateFCmpOGT(norm_product, eps);
    llvm::Value* safe_denom = builder.CreateSelect(safe_denom_cmp, norm_product, eps);

    llvm::Value* cosine_sim = builder.CreateFDiv(final_dot, safe_denom);

    // Result depends on label y
    // y == 1: loss = 1 - cos_sim
    // y == -1: loss = max(0, cos_sim - margin)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* zero = llvm::ConstantFP::get(ctx_.doubleType(), 0.0);
    llvm::Value* pos_loss = builder.CreateFSub(one, cosine_sim);
    llvm::Value* cos_minus_margin = builder.CreateFSub(cosine_sim, margin);
    llvm::Value* neg_cmp = builder.CreateFCmpOGT(cos_minus_margin, zero);
    llvm::Value* neg_loss = builder.CreateSelect(neg_cmp, cos_minus_margin, zero);

    // Select based on y: if y > 0, positive case; else negative case
    llvm::Value* is_similar = builder.CreateFCmpOGT(y, zero);
    llvm::Value* result = builder.CreateSelect(is_similar, pos_loss, neg_loss);

    return tagged_.packDouble(result);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
