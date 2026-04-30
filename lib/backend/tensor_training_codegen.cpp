/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TensorCodegen — Training Loop Support. Extracted from
 * tensor_codegen.cpp during the v1.2 mechanical split. Implements:
 *   - Optimizers (Track 10.1): SGD, Adam, AdamW, RMSprop, Adagrad,
 *     plus zeroGrad, clipGradNorm, checkGradHealth.
 *   - Weight initialisation: Xavier (uniform/normal), Kaiming
 *     (uniform/normal), LeCun normal.
 *   - Learning-rate schedulers: cosine annealing, step decay,
 *     linear warmup, exponential decay.
 *
 * IR-identical to the prior in-class implementations; verified via the
 * pre-training-extract baseline.
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

// ===== OPTIMIZERS (Track 10.1) =====
// Full production implementations for neural network training

llvm::Value* TensorCodegen::sgdStep(const eshkol_operations_t* op) {
    // sgd-step!: (sgd-step! params grads lr [momentum velocity])
    // Implements SGD with optional Nesterov momentum
    if (op->call_op.num_vars < 3) {
        eshkol_error("sgd-step! requires at least 3 arguments: params, grads, lr");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get params tensor
    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    if (!params_tagged) return nullptr;
    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);

    // Get grads tensor
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    if (!grads_tagged) return nullptr;
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);

    // Get learning rate
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    if (!lr_tagged) return nullptr;
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);

    // Get momentum and velocity if provided
    bool has_momentum = op->call_op.num_vars >= 5;
    llvm::Value* momentum = nullptr;
    llvm::Value* velocity_ptr = nullptr;

    if (has_momentum) {
        llvm::Value* momentum_tagged = codegenAST(&op->call_op.variables[3]);
        if (!momentum_tagged) return nullptr;
        momentum = tagged_.unpackDouble(momentum_tagged);

        llvm::Value* velocity_tagged = codegenAST(&op->call_op.variables[4]);
        if (!velocity_tagged) return nullptr;
        velocity_ptr = tagged_.unpackPtr(velocity_tagged);
    }

    // Get tensor dimensions
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);

    // Get element pointers
    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);

    // Compute total elements
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "sgd_size_loop", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "sgd_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "sgd_update_loop", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "sgd_update_done", current_func);

    // Compute total size
    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "size_idx");
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type(), nullptr, "total_size");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(size_cond, llvm::BasicBlock::Create(ctx_.context(), "size_body", current_func), size_done);

    llvm::BasicBlock* size_body = &current_func->back();
    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Update loop
    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elem_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);

    llvm::Value* velocity_elems = nullptr;
    if (has_momentum) {
        llvm::Value* vel_elems_ptr = builder.CreateStructGEP(tensor_type, velocity_ptr, 2);
        velocity_elems = builder.CreateLoad(ctx_.ptrType(), vel_elems_ptr);
    }

    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    llvm::Value* update_cond = builder.CreateICmpULT(ei, num_elements);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "update_body", current_func);
    builder.CreateCondBr(update_cond, update_body, update_done);

    builder.SetInsertPoint(update_body);

    // Load gradient
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    // Load current param
    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* update;
    if (has_momentum) {
        // v = momentum * v + grad
        llvm::Value* vel_ptr = builder.CreateGEP(ctx_.int64Type(), velocity_elems, ei);
        llvm::Value* vel_bits = builder.CreateLoad(ctx_.int64Type(), vel_ptr);
        llvm::Value* vel = builder.CreateBitCast(vel_bits, ctx_.doubleType());
        llvm::Value* new_vel = builder.CreateFAdd(builder.CreateFMul(momentum, vel), grad);
        llvm::Value* new_vel_bits = builder.CreateBitCast(new_vel, ctx_.int64Type());
        builder.CreateStore(new_vel_bits, vel_ptr);
        update = new_vel;
    } else {
        update = grad;
    }

    // param = param - lr * update
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    llvm::Value* new_param_bits = builder.CreateBitCast(new_param, ctx_.int64Type());
    builder.CreateStore(new_param_bits, param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::adamStep(const eshkol_operations_t* op) {
    // adam-step!: (adam-step! params grads lr m v t [beta1 beta2 eps])
    // Implements Adam optimizer (Kingma & Ba, 2014)
    if (op->call_op.num_vars < 6) {
        eshkol_error("adam-step! requires at least 6 arguments: params, grads, lr, m, v, t");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    // Get arguments
    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* m_tagged = codegenAST(&op->call_op.variables[3]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[4]);
    llvm::Value* t_tagged = codegenAST(&op->call_op.variables[5]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !m_tagged || !v_tagged || !t_tagged) {
        return nullptr;
    }

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* m_ptr = tagged_.unpackPtr(m_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);
    llvm::Value* t = tagged_.unpackInt64(t_tagged);
    llvm::Value* t_double = builder.CreateSIToFP(t, ctx_.doubleType());

    // Default hyperparameters
    llvm::Value* beta1 = llvm::ConstantFP::get(ctx_.doubleType(), 0.9);
    llvm::Value* beta2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.999);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);

    if (op->call_op.num_vars >= 7) {
        llvm::Value* b1_tagged = codegenAST(&op->call_op.variables[6]);
        if (b1_tagged) beta1 = tagged_.unpackDouble(b1_tagged);
    }
    if (op->call_op.num_vars >= 8) {
        llvm::Value* b2_tagged = codegenAST(&op->call_op.variables[7]);
        if (b2_tagged) beta2 = tagged_.unpackDouble(b2_tagged);
    }
    if (op->call_op.num_vars >= 9) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[8]);
        if (eps_tagged) eps = tagged_.unpackDouble(eps_tagged);
    }

    // Compute bias corrections: 1 - beta^t
    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* pow_type = llvm::FunctionType::get(ctx_.doubleType(),
            {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(pow_type, llvm::Function::ExternalLinkage, "pow", ctx_.module());
    }

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* beta1_t = builder.CreateCall(pow_func, {beta1, t_double});
    llvm::Value* beta2_t = builder.CreateCall(pow_func, {beta2, t_double});
    llvm::Value* bias_corr1 = builder.CreateFSub(one, beta1_t);
    llvm::Value* bias_corr2 = builder.CreateFSub(one, beta2_t);

    // Get tensor info
    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* m_elems_ptr = builder.CreateStructGEP(tensor_type, m_ptr, 2);
    llvm::Value* m_elems = builder.CreateLoad(ctx_.ptrType(), m_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    // Compute total elements
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "adam_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "adam_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "adam_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adam_update_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adam_update_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adam_update_done", current_func);

    llvm::Value* size_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "size_idx");
    llvm::Value* total_size = builder.CreateAlloca(ctx_.int64Type(), nullptr, "total_size");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), size_idx);
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 1), total_size);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_loop);
    llvm::Value* si = builder.CreateLoad(ctx_.int64Type(), size_idx);
    llvm::Value* size_cond = builder.CreateICmpULT(si, num_dims);
    builder.CreateCondBr(size_cond, size_body, size_done);

    builder.SetInsertPoint(size_body);
    llvm::Value* dim_ptr = builder.CreateGEP(ctx_.int64Type(), dims, si);
    llvm::Value* dim_val = builder.CreateLoad(ctx_.int64Type(), dim_ptr);
    llvm::Value* cur_size = builder.CreateLoad(ctx_.int64Type(), total_size);
    builder.CreateStore(builder.CreateMul(cur_size, dim_val), total_size);
    builder.CreateStore(builder.CreateAdd(si, llvm::ConstantInt::get(ctx_.int64Type(), 1)), size_idx);
    builder.CreateBr(size_loop);

    builder.SetInsertPoint(size_done);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_size);

    // Update loop
    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type(), nullptr, "elem_idx");
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    llvm::Value* update_cond = builder.CreateICmpULT(ei, num_elements);
    builder.CreateCondBr(update_cond, update_body, update_done);

    builder.SetInsertPoint(update_body);

    // Load values
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* m_elem_ptr = builder.CreateGEP(ctx_.int64Type(), m_elems, ei);
    llvm::Value* m_bits = builder.CreateLoad(ctx_.int64Type(), m_elem_ptr);
    llvm::Value* m_val = builder.CreateBitCast(m_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    llvm::Value* one_minus_beta1 = builder.CreateFSub(one, beta1);
    llvm::Value* m_new = builder.CreateFAdd(
        builder.CreateFMul(beta1, m_val),
        builder.CreateFMul(one_minus_beta1, grad));

    // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    llvm::Value* one_minus_beta2 = builder.CreateFSub(one, beta2);
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(beta2, v_val),
        builder.CreateFMul(one_minus_beta2, grad_sq));

    // Store updated moments
    builder.CreateStore(builder.CreateBitCast(m_new, ctx_.int64Type()), m_elem_ptr);
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // Bias-corrected estimates
    llvm::Value* m_hat = builder.CreateFDiv(m_new, bias_corr1);
    llvm::Value* v_hat = builder.CreateFDiv(v_new, bias_corr2);

    // Update: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v_hat = builder.CreateCall(sqrt_func, {v_hat});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v_hat, eps);
    llvm::Value* update = builder.CreateFDiv(m_hat, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);

    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::zeroGrad(const eshkol_operations_t* op) {
    // zero-grad!: (zero-grad! tensor)
    // Zero all elements in place
    if (op->call_op.num_vars < 1) {
        eshkol_error("zero-grad! requires 1 argument");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    if (!tensor_tagged) return nullptr;
    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // Compute total elements
    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "zero_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "zero_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "zero_size_done", current_func);

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

    // Use memset for efficiency
    llvm::Value* byte_size = builder.CreateMul(num_elements, llvm::ConstantInt::get(ctx_.int64Type(), 8));
    builder.CreateMemSet(elems, llvm::ConstantInt::get(ctx_.int8Type(), 0),
                         byte_size, llvm::MaybeAlign(8));

    return tensor_tagged;
}

llvm::Value* TensorCodegen::clipGradNorm(const eshkol_operations_t* op) {
    // clip-grad-norm!: (clip-grad-norm! grads max-norm)
    if (op->call_op.num_vars < 2) {
        eshkol_error("clip-grad-norm! requires 2 arguments: grads, max-norm");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* max_norm_tagged = codegenAST(&op->call_op.variables[1]);
    if (!grads_tagged || !max_norm_tagged) return nullptr;

    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* max_norm = tagged_.unpackDouble(max_norm_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "clip_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "clip_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "clip_size_done", current_func);

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

    // Compute L2 norm: sqrt(sum(grad^2))
    llvm::BasicBlock* norm_loop = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_loop", current_func);
    llvm::BasicBlock* norm_body = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_body", current_func);
    llvm::BasicBlock* norm_done = llvm::BasicBlock::Create(ctx_.context(), "clip_norm_done", current_func);

    llvm::Value* norm_idx = builder.CreateAlloca(ctx_.int64Type());
    llvm::Value* sum_sq = builder.CreateAlloca(ctx_.doubleType());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), norm_idx);
    builder.CreateStore(llvm::ConstantFP::get(ctx_.doubleType(), 0.0), sum_sq);
    builder.CreateBr(norm_loop);

    builder.SetInsertPoint(norm_loop);
    llvm::Value* ni = builder.CreateLoad(ctx_.int64Type(), norm_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ni, num_elements), norm_body, norm_done);

    builder.SetInsertPoint(norm_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, ni);
    llvm::Value* elem_bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* elem = builder.CreateBitCast(elem_bits, ctx_.doubleType());
    llvm::Value* elem_sq = builder.CreateFMul(elem, elem);
    llvm::Value* cur_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq);
    builder.CreateStore(builder.CreateFAdd(cur_sum, elem_sq), sum_sq);
    builder.CreateStore(builder.CreateAdd(ni, llvm::ConstantInt::get(ctx_.int64Type(), 1)), norm_idx);
    builder.CreateBr(norm_loop);

    builder.SetInsertPoint(norm_done);
    llvm::Value* final_sum = builder.CreateLoad(ctx_.doubleType(), sum_sq);
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* total_norm = builder.CreateCall(sqrt_func, {final_sum});

    // Clip if necessary
    llvm::BasicBlock* do_clip = llvm::BasicBlock::Create(ctx_.context(), "do_clip", current_func);
    llvm::BasicBlock* clip_done = llvm::BasicBlock::Create(ctx_.context(), "clip_done", current_func);

    llvm::Value* need_clip = builder.CreateFCmpOGT(total_norm, max_norm);
    builder.CreateCondBr(need_clip, do_clip, clip_done);

    builder.SetInsertPoint(do_clip);
    llvm::Value* scale = builder.CreateFDiv(max_norm, total_norm);

    // Scale all gradients
    llvm::BasicBlock* scale_loop = llvm::BasicBlock::Create(ctx_.context(), "scale_loop", current_func);
    llvm::BasicBlock* scale_body = llvm::BasicBlock::Create(ctx_.context(), "scale_body", current_func);
    llvm::BasicBlock* scale_done = llvm::BasicBlock::Create(ctx_.context(), "scale_done", current_func);

    llvm::Value* scale_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), scale_idx);
    builder.CreateBr(scale_loop);

    builder.SetInsertPoint(scale_loop);
    llvm::Value* sci = builder.CreateLoad(ctx_.int64Type(), scale_idx);
    builder.CreateCondBr(builder.CreateICmpULT(sci, num_elements), scale_body, scale_done);

    builder.SetInsertPoint(scale_body);
    llvm::Value* sc_elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, sci);
    llvm::Value* sc_elem_bits = builder.CreateLoad(ctx_.int64Type(), sc_elem_ptr);
    llvm::Value* sc_elem = builder.CreateBitCast(sc_elem_bits, ctx_.doubleType());
    llvm::Value* scaled_elem = builder.CreateFMul(sc_elem, scale);
    builder.CreateStore(builder.CreateBitCast(scaled_elem, ctx_.int64Type()), sc_elem_ptr);
    builder.CreateStore(builder.CreateAdd(sci, llvm::ConstantInt::get(ctx_.int64Type(), 1)), scale_idx);
    builder.CreateBr(scale_loop);

    builder.SetInsertPoint(scale_done);
    builder.CreateBr(clip_done);

    builder.SetInsertPoint(clip_done);
    return tagged_.packDouble(total_norm);
}

llvm::Value* TensorCodegen::rmspropStep(const eshkol_operations_t* op) {
    // rmsprop-step!: (rmsprop-step! params grads lr v [alpha eps])
    if (op->call_op.num_vars < 4) {
        eshkol_error("rmsprop-step! requires at least 4 arguments: params, grads, lr, v");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[3]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !v_tagged) return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);

    llvm::Value* alpha = llvm::ConstantFP::get(ctx_.doubleType(), 0.99);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);

    if (op->call_op.num_vars >= 5) {
        llvm::Value* a_tagged = codegenAST(&op->call_op.variables[4]);
        if (a_tagged) alpha = tagged_.unpackDouble(a_tagged);
    }
    if (op->call_op.num_vars >= 6) {
        llvm::Value* e_tagged = codegenAST(&op->call_op.variables[5]);
        if (e_tagged) eps = tagged_.unpackDouble(e_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* num_dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 1);
    llvm::Value* num_dims = builder.CreateLoad(ctx_.int64Type(), num_dims_ptr);
    llvm::Value* dims_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 0);
    llvm::Value* dims = builder.CreateLoad(ctx_.ptrType(), dims_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();

    // Compute total elements
    llvm::BasicBlock* size_loop = llvm::BasicBlock::Create(ctx_.context(), "rms_size_loop", current_func);
    llvm::BasicBlock* size_body = llvm::BasicBlock::Create(ctx_.context(), "rms_size_body", current_func);
    llvm::BasicBlock* size_done = llvm::BasicBlock::Create(ctx_.context(), "rms_size_done", current_func);
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "rms_update_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "rms_update_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "rms_update_done", current_func);

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
    llvm::Value* one_minus_alpha = builder.CreateFSub(one, alpha);

    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ei, num_elements), update_body, update_done);

    builder.SetInsertPoint(update_body);

    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // v = alpha * v + (1 - alpha) * g^2
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(alpha, v_val),
        builder.CreateFMul(one_minus_alpha, grad_sq));
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // param = param - lr * g / (sqrt(v) + eps)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v = builder.CreateCall(sqrt_func, {v_new});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v, eps);
    llvm::Value* update = builder.CreateFDiv(grad, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::adamwStep(const eshkol_operations_t* op) {
    // adamw-step!: (adamw-step! params grads lr m v t [beta1 beta2 eps weight_decay])
    // AdamW: Decoupled weight decay — weight decay is applied directly to params,
    // NOT through the gradient (unlike L2 regularization in vanilla Adam)
    if (op->call_op.num_vars < 6) {
        eshkol_error("adamw-step! requires at least 6 arguments: params, grads, lr, m, v, t");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* m_tagged = codegenAST(&op->call_op.variables[3]);
    llvm::Value* v_tagged = codegenAST(&op->call_op.variables[4]);
    llvm::Value* t_tagged = codegenAST(&op->call_op.variables[5]);

    if (!params_tagged || !grads_tagged || !lr_tagged || !m_tagged || !v_tagged || !t_tagged)
        return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* m_ptr = tagged_.unpackPtr(m_tagged);
    llvm::Value* v_ptr = tagged_.unpackPtr(v_tagged);
    llvm::Value* t = tagged_.unpackInt64(t_tagged);
    llvm::Value* t_double = builder.CreateSIToFP(t, ctx_.doubleType());

    llvm::Value* beta1 = llvm::ConstantFP::get(ctx_.doubleType(), 0.9);
    llvm::Value* beta2 = llvm::ConstantFP::get(ctx_.doubleType(), 0.999);
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-8);
    llvm::Value* weight_decay = llvm::ConstantFP::get(ctx_.doubleType(), 0.01);

    if (op->call_op.num_vars >= 7) {
        llvm::Value* b1_tagged = codegenAST(&op->call_op.variables[6]);
        if (b1_tagged) beta1 = tagged_.unpackDouble(b1_tagged);
    }
    if (op->call_op.num_vars >= 8) {
        llvm::Value* b2_tagged = codegenAST(&op->call_op.variables[7]);
        if (b2_tagged) beta2 = tagged_.unpackDouble(b2_tagged);
    }
    if (op->call_op.num_vars >= 9) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[8]);
        if (eps_tagged) eps = tagged_.unpackDouble(eps_tagged);
    }
    if (op->call_op.num_vars >= 10) {
        llvm::Value* wd_tagged = codegenAST(&op->call_op.variables[9]);
        if (wd_tagged) weight_decay = tagged_.unpackDouble(wd_tagged);
    }

    // Bias corrections
    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* pow_type = llvm::FunctionType::get(ctx_.doubleType(),
            {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(pow_type, llvm::Function::ExternalLinkage, "pow", ctx_.module());
    }

    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* beta1_t = builder.CreateCall(pow_func, {beta1, t_double});
    llvm::Value* beta2_t = builder.CreateCall(pow_func, {beta2, t_double});
    llvm::Value* bias_corr1 = builder.CreateFSub(one, beta1_t);
    llvm::Value* bias_corr2 = builder.CreateFSub(one, beta2_t);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 3);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* m_elems_ptr = builder.CreateStructGEP(tensor_type, m_ptr, 2);
    llvm::Value* m_elems = builder.CreateLoad(ctx_.ptrType(), m_elems_ptr);
    llvm::Value* v_elems_ptr = builder.CreateStructGEP(tensor_type, v_ptr, 2);
    llvm::Value* v_elems = builder.CreateLoad(ctx_.ptrType(), v_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adamw_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adamw_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adamw_done", current_func);

    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ei, num_elements), update_body, update_done);

    builder.SetInsertPoint(update_body);
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* m_elem_ptr = builder.CreateGEP(ctx_.int64Type(), m_elems, ei);
    llvm::Value* m_bits = builder.CreateLoad(ctx_.int64Type(), m_elem_ptr);
    llvm::Value* m_val = builder.CreateBitCast(m_bits, ctx_.doubleType());

    llvm::Value* v_elem_ptr = builder.CreateGEP(ctx_.int64Type(), v_elems, ei);
    llvm::Value* v_bits = builder.CreateLoad(ctx_.int64Type(), v_elem_ptr);
    llvm::Value* v_val = builder.CreateBitCast(v_bits, ctx_.doubleType());

    // STEP 1: Decoupled weight decay — applied to params directly, NOT through gradient
    // param = param * (1 - lr * weight_decay)
    llvm::Value* decay_factor = builder.CreateFSub(one, builder.CreateFMul(lr, weight_decay));
    llvm::Value* decayed_param = builder.CreateFMul(param, decay_factor);

    // STEP 2: Standard Adam moment updates
    llvm::Value* one_minus_beta1 = builder.CreateFSub(one, beta1);
    llvm::Value* m_new = builder.CreateFAdd(
        builder.CreateFMul(beta1, m_val),
        builder.CreateFMul(one_minus_beta1, grad));

    llvm::Value* one_minus_beta2 = builder.CreateFSub(one, beta2);
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* v_new = builder.CreateFAdd(
        builder.CreateFMul(beta2, v_val),
        builder.CreateFMul(one_minus_beta2, grad_sq));

    builder.CreateStore(builder.CreateBitCast(m_new, ctx_.int64Type()), m_elem_ptr);
    builder.CreateStore(builder.CreateBitCast(v_new, ctx_.int64Type()), v_elem_ptr);

    // STEP 3: Bias-corrected Adam update on already-decayed param
    llvm::Value* m_hat = builder.CreateFDiv(m_new, bias_corr1);
    llvm::Value* v_hat = builder.CreateFDiv(v_new, bias_corr2);

    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_v_hat = builder.CreateCall(sqrt_func, {v_hat});
    llvm::Value* denom = builder.CreateFAdd(sqrt_v_hat, eps);
    llvm::Value* update = builder.CreateFDiv(m_hat, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(decayed_param, scaled_update);

    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::adagradStep(const eshkol_operations_t* op) {
    // adagrad-step!: (adagrad-step! params grads lr accum [eps])
    // Adagrad: Per-parameter adaptive learning rate via accumulated squared gradients
    // accum_new = accum + grad^2
    // param = param - lr * grad / (sqrt(accum_new) + eps)
    if (op->call_op.num_vars < 4) {
        eshkol_error("adagrad-step! requires at least 4 arguments: params, grads, lr, accum");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* params_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* grads_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* lr_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* accum_tagged = codegenAST(&op->call_op.variables[3]);
    if (!params_tagged || !grads_tagged || !lr_tagged || !accum_tagged) return nullptr;

    llvm::Value* params_ptr = tagged_.unpackPtr(params_tagged);
    llvm::Value* grads_ptr = tagged_.unpackPtr(grads_tagged);
    llvm::Value* lr = tagged_.unpackDouble(lr_tagged);
    llvm::Value* accum_ptr = tagged_.unpackPtr(accum_tagged);

    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    if (op->call_op.num_vars >= 5) {
        llvm::Value* eps_tagged = codegenAST(&op->call_op.variables[4]);
        if (eps_tagged) eps = tagged_.unpackDouble(eps_tagged);
    }

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 3);
    llvm::Value* num_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);

    llvm::Value* params_elems_ptr = builder.CreateStructGEP(tensor_type, params_ptr, 2);
    llvm::Value* params_elems = builder.CreateLoad(ctx_.ptrType(), params_elems_ptr);
    llvm::Value* grads_elems_ptr = builder.CreateStructGEP(tensor_type, grads_ptr, 2);
    llvm::Value* grads_elems = builder.CreateLoad(ctx_.ptrType(), grads_elems_ptr);
    llvm::Value* accum_elems_ptr = builder.CreateStructGEP(tensor_type, accum_ptr, 2);
    llvm::Value* accum_elems = builder.CreateLoad(ctx_.ptrType(), accum_elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* update_loop = llvm::BasicBlock::Create(ctx_.context(), "adagrad_loop", current_func);
    llvm::BasicBlock* update_body = llvm::BasicBlock::Create(ctx_.context(), "adagrad_body", current_func);
    llvm::BasicBlock* update_done = llvm::BasicBlock::Create(ctx_.context(), "adagrad_done", current_func);

    llvm::Value* elem_idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_loop);
    llvm::Value* ei = builder.CreateLoad(ctx_.int64Type(), elem_idx);
    builder.CreateCondBr(builder.CreateICmpULT(ei, num_elements), update_body, update_done);

    builder.SetInsertPoint(update_body);
    llvm::Value* grad_ptr = builder.CreateGEP(ctx_.int64Type(), grads_elems, ei);
    llvm::Value* grad_bits = builder.CreateLoad(ctx_.int64Type(), grad_ptr);
    llvm::Value* grad = builder.CreateBitCast(grad_bits, ctx_.doubleType());

    llvm::Value* param_ptr = builder.CreateGEP(ctx_.int64Type(), params_elems, ei);
    llvm::Value* param_bits = builder.CreateLoad(ctx_.int64Type(), param_ptr);
    llvm::Value* param = builder.CreateBitCast(param_bits, ctx_.doubleType());

    llvm::Value* acc_ptr = builder.CreateGEP(ctx_.int64Type(), accum_elems, ei);
    llvm::Value* acc_bits = builder.CreateLoad(ctx_.int64Type(), acc_ptr);
    llvm::Value* acc_val = builder.CreateBitCast(acc_bits, ctx_.doubleType());

    // accum += grad^2
    llvm::Value* grad_sq = builder.CreateFMul(grad, grad);
    llvm::Value* acc_new = builder.CreateFAdd(acc_val, grad_sq);
    builder.CreateStore(builder.CreateBitCast(acc_new, ctx_.int64Type()), acc_ptr);

    // param -= lr * grad / (sqrt(accum) + eps)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* sqrt_acc = builder.CreateCall(sqrt_func, {acc_new});
    llvm::Value* denom = builder.CreateFAdd(sqrt_acc, eps);
    llvm::Value* update = builder.CreateFDiv(grad, denom);
    llvm::Value* scaled_update = builder.CreateFMul(lr, update);
    llvm::Value* new_param = builder.CreateFSub(param, scaled_update);
    builder.CreateStore(builder.CreateBitCast(new_param, ctx_.int64Type()), param_ptr);

    builder.CreateStore(builder.CreateAdd(ei, llvm::ConstantInt::get(ctx_.int64Type(), 1)), elem_idx);
    builder.CreateBr(update_loop);

    builder.SetInsertPoint(update_done);
    return params_tagged;
}

llvm::Value* TensorCodegen::checkGradHealth(const eshkol_operations_t* op) {
    // check-grad-health: (check-grad-health tensor) → #t if all finite, #f if NaN/Inf found
    if (op->call_op.num_vars != 1) {
        eshkol_error("check-grad-health requires exactly 1 argument");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    if (!tensor_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "health_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "health_body", current_func);
    llvm::BasicBlock* loop_inc = llvm::BasicBlock::Create(ctx_.context(), "health_inc", current_func);
    llvm::BasicBlock* health_bad = llvm::BasicBlock::Create(ctx_.context(), "health_bad", current_func);
    llvm::BasicBlock* health_good = llvm::BasicBlock::Create(ctx_.context(), "health_good", current_func);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "health_merge", current_func);

    llvm::Value* idx = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), idx);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), idx);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, health_good);

    builder.SetInsertPoint(loop_body);
    llvm::Value* elem_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    llvm::Value* bits = builder.CreateLoad(ctx_.int64Type(), elem_ptr);
    llvm::Value* val = builder.CreateBitCast(bits, ctx_.doubleType());

    // Detect NaN or Inf: for finite values, (x - x) == 0.0 (ordered).
    // For NaN or Inf, (x - x) is NaN (unordered).
    llvm::Value* diff = builder.CreateFSub(val, val);
    llvm::Value* not_finite = builder.CreateFCmpUNO(diff, diff);
    builder.CreateCondBr(not_finite, health_bad, loop_inc);

    // Increment and loop back
    builder.SetInsertPoint(loop_inc);
    llvm::Value* next_i = builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1));
    builder.CreateStore(next_i, idx);
    builder.CreateBr(loop_cond);

    // All elements finite → return true
    builder.SetInsertPoint(health_good);
    llvm::Value* true_val = tagged_.packBool(builder.getInt1(1));
    builder.CreateBr(merge_block);

    // Found NaN/Inf → return false
    builder.SetInsertPoint(health_bad);
    llvm::Value* false_val = tagged_.packBool(builder.getInt1(0));
    builder.CreateBr(merge_block);

    builder.SetInsertPoint(merge_block);
    llvm::PHINode* result = builder.CreatePHI(true_val->getType(), 2, "health_result");
    result->addIncoming(true_val, health_good);
    result->addIncoming(false_val, health_bad);
    return result;
}

// ===== WEIGHT INITIALIZATION =====

llvm::Value* TensorCodegen::xavierUniform(const eshkol_operations_t* op) {
    // xavier-uniform!: (xavier-uniform! tensor fan_in fan_out)
    // Fills tensor with U(-limit, limit) where limit = sqrt(6 / (fan_in + fan_out))
    if (op->call_op.num_vars != 3) {
        eshkol_error("xavier-uniform! requires 3 arguments: tensor, fan_in, fan_out");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fan_out_tagged = codegenAST(&op->call_op.variables[2]);
    if (!tensor_tagged || !fan_in_tagged || !fan_out_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);
    llvm::Value* fan_out = taggedNumericToDouble(ctx_, tagged_, fan_out_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // limit = sqrt(6 / (fan_in + fan_out))
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* fan_sum = builder.CreateFAdd(fan_in, fan_out);
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* ratio = builder.CreateFDiv(six, fan_sum);
    llvm::Value* limit = builder.CreateCall(sqrt_func, {ratio});
    llvm::Value* neg_limit = builder.CreateFNeg(limit);
    llvm::Value* range = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), limit);

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "xu_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "xu_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "xu_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Generate U(-limit, limit) = drand48() * 2 * limit - limit
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    llvm::Value* scaled = builder.CreateFMul(rand_val, range);
    llvm::Value* result = builder.CreateFAdd(scaled, neg_limit);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::xavierNormal(const eshkol_operations_t* op) {
    // xavier-normal!: (xavier-normal! tensor fan_in fan_out)
    // Fills tensor with N(0, std) where std = sqrt(2 / (fan_in + fan_out))
    // Uses Box-Muller transform for normal distribution
    if (op->call_op.num_vars != 3) {
        eshkol_error("xavier-normal! requires 3 arguments: tensor, fan_in, fan_out");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* fan_out_tagged = codegenAST(&op->call_op.variables[2]);
    if (!tensor_tagged || !fan_in_tagged || !fan_out_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);
    llvm::Value* fan_out = taggedNumericToDouble(ctx_, tagged_, fan_out_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(2 / (fan_in + fan_out))
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* fan_sum = builder.CreateFAdd(fan_in, fan_out);
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* ratio = builder.CreateFDiv(two, fan_sum);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "xn_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "xn_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "xn_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller transform: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    // Clamp u1 away from 0 for log safety
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::kaimingUniform(const eshkol_operations_t* op) {
    // kaiming-uniform!: (kaiming-uniform! tensor fan_in)
    // Fills tensor with U(-limit, limit) where limit = sqrt(6 / fan_in)
    // For ReLU networks (He initialization)
    if (op->call_op.num_vars < 2 || op->call_op.num_vars > 2) {
        eshkol_error("kaiming-uniform! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // limit = sqrt(6 / fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* six = llvm::ConstantFP::get(ctx_.doubleType(), 6.0);
    llvm::Value* ratio = builder.CreateFDiv(six, fan_in);
    llvm::Value* limit = builder.CreateCall(sqrt_func, {ratio});
    llvm::Value* neg_limit = builder.CreateFNeg(limit);
    llvm::Value* range = builder.CreateFMul(llvm::ConstantFP::get(ctx_.doubleType(), 2.0), limit);

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "ku_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "ku_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "ku_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    llvm::Value* rand_val = builder.CreateCall(drand_func, {});
    llvm::Value* scaled = builder.CreateFMul(rand_val, range);
    llvm::Value* result = builder.CreateFAdd(scaled, neg_limit);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::kaimingNormal(const eshkol_operations_t* op) {
    // kaiming-normal!: (kaiming-normal! tensor fan_in)
    // Fills tensor with N(0, std) where std = sqrt(2 / fan_in)
    if (op->call_op.num_vars != 2) {
        eshkol_error("kaiming-normal! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(2 / fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* two = llvm::ConstantFP::get(ctx_.doubleType(), 2.0);
    llvm::Value* ratio = builder.CreateFDiv(two, fan_in);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "kn_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "kn_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "kn_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

llvm::Value* TensorCodegen::lecunNormal(const eshkol_operations_t* op) {
    // lecun-normal!: (lecun-normal! tensor fan_in)
    // Fills tensor with N(0, std) where std = sqrt(1 / fan_in)
    // For SELU activation
    if (op->call_op.num_vars != 2) {
        eshkol_error("lecun-normal! requires 2 arguments: tensor, fan_in");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* tensor_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* fan_in_tagged = codegenAST(&op->call_op.variables[1]);
    if (!tensor_tagged || !fan_in_tagged) return nullptr;

    llvm::Value* tensor_ptr = tagged_.unpackPtr(tensor_tagged);
    llvm::Value* fan_in = taggedNumericToDouble(ctx_, tagged_, fan_in_tagged);

    llvm::StructType* tensor_type = ctx_.tensorType();
    llvm::Value* total_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 3);
    llvm::Value* total_elements = builder.CreateLoad(ctx_.int64Type(), total_ptr);
    llvm::Value* elems_ptr = builder.CreateStructGEP(tensor_type, tensor_ptr, 2);
    llvm::Value* elems = builder.CreateLoad(ctx_.ptrType(), elems_ptr);

    // std = sqrt(1 / fan_in) = 1 / sqrt(fan_in)
    llvm::Function* sqrt_func = ESHKOL_GET_INTRINSIC(&ctx_.module(),
        llvm::Intrinsic::sqrt, {ctx_.doubleType()});
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* ratio = builder.CreateFDiv(one, fan_in);
    llvm::Value* std_dev = builder.CreateCall(sqrt_func, {ratio});

    llvm::Function* drand_func = ctx_.module().getFunction("drand48");
    if (!drand_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {}, false);
        drand_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "drand48", &ctx_.module());
    }
    llvm::Function* log_func = ctx_.module().getFunction("log");
    if (!log_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        log_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "log", &ctx_.module());
    }
    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    llvm::Function* current_func = builder.GetInsertBlock()->getParent();
    llvm::BasicBlock* loop_cond = llvm::BasicBlock::Create(ctx_.context(), "ln_cond", current_func);
    llvm::BasicBlock* loop_body = llvm::BasicBlock::Create(ctx_.context(), "ln_body", current_func);
    llvm::BasicBlock* exit_block = llvm::BasicBlock::Create(ctx_.context(), "ln_exit", current_func);

    llvm::Value* counter = builder.CreateAlloca(ctx_.int64Type());
    builder.CreateStore(llvm::ConstantInt::get(ctx_.int64Type(), 0), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(loop_cond);
    llvm::Value* i = builder.CreateLoad(ctx_.int64Type(), counter);
    builder.CreateCondBr(builder.CreateICmpULT(i, total_elements), loop_body, exit_block);

    builder.SetInsertPoint(loop_body);
    // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
    llvm::Value* u1 = builder.CreateCall(drand_func, {});
    llvm::Value* u2 = builder.CreateCall(drand_func, {});
    llvm::Value* eps = llvm::ConstantFP::get(ctx_.doubleType(), 1e-10);
    llvm::Value* u1_cmp = builder.CreateFCmpOGT(u1, eps);
    llvm::Value* u1_safe = builder.CreateSelect(u1_cmp, u1, eps);

    llvm::Value* neg_two = llvm::ConstantFP::get(ctx_.doubleType(), -2.0);
    llvm::Value* log_u1 = builder.CreateCall(log_func, {u1_safe});
    llvm::Value* inner = builder.CreateFMul(neg_two, log_u1);
    llvm::Value* radius = builder.CreateCall(sqrt_func, {inner});

    llvm::Value* two_pi = llvm::ConstantFP::get(ctx_.doubleType(), 6.283185307179586);
    llvm::Value* angle = builder.CreateFMul(two_pi, u2);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});

    llvm::Value* z = builder.CreateFMul(radius, cos_val);
    llvm::Value* result = builder.CreateFMul(z, std_dev);

    llvm::Value* dst_ptr = builder.CreateGEP(ctx_.int64Type(), elems, i);
    builder.CreateStore(builder.CreateBitCast(result, ctx_.int64Type()), dst_ptr);

    builder.CreateStore(builder.CreateAdd(i, llvm::ConstantInt::get(ctx_.int64Type(), 1)), counter);
    builder.CreateBr(loop_cond);

    builder.SetInsertPoint(exit_block);
    return tensor_tagged;
}

// ===== LEARNING RATE SCHEDULERS =====

llvm::Value* TensorCodegen::cosineAnnealingLR(const eshkol_operations_t* op) {
    // cosine-annealing-lr: (cosine-annealing-lr base_lr min_lr current_step total_steps) → lr
    // lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * t / T))
    if (op->call_op.num_vars != 4) {
        eshkol_error("cosine-annealing-lr requires 4 arguments: base_lr, min_lr, current_step, total_steps");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* min_lr_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* total_tagged = codegenAST(&op->call_op.variables[3]);
    if (!base_lr_tagged || !min_lr_tagged || !step_tagged || !total_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* min_lr = tagged_.unpackDouble(min_lr_tagged);
    llvm::Value* step = tagged_.unpackDouble(step_tagged);
    llvm::Value* total = tagged_.unpackDouble(total_tagged);

    llvm::Function* cos_func = ctx_.module().getFunction("cos");
    if (!cos_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        cos_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "cos", &ctx_.module());
    }

    // lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / total))
    llvm::Value* pi = llvm::ConstantFP::get(ctx_.doubleType(), 3.14159265358979323846);
    llvm::Value* half = llvm::ConstantFP::get(ctx_.doubleType(), 0.5);
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);

    llvm::Value* lr_range = builder.CreateFSub(base_lr, min_lr);
    llvm::Value* progress = builder.CreateFDiv(step, total);
    llvm::Value* angle = builder.CreateFMul(pi, progress);
    llvm::Value* cos_val = builder.CreateCall(cos_func, {angle});
    llvm::Value* one_plus_cos = builder.CreateFAdd(one, cos_val);
    llvm::Value* scaled = builder.CreateFMul(half, builder.CreateFMul(lr_range, one_plus_cos));
    llvm::Value* result = builder.CreateFAdd(min_lr, scaled);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::stepDecayLR(const eshkol_operations_t* op) {
    // step-decay-lr: (step-decay-lr base_lr gamma epoch step_size) → lr
    // lr = base_lr * gamma^(floor(epoch / step_size))
    if (op->call_op.num_vars != 4) {
        eshkol_error("step-decay-lr requires 4 arguments: base_lr, gamma, epoch, step_size");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* gamma_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* epoch_tagged = codegenAST(&op->call_op.variables[2]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[3]);
    if (!base_lr_tagged || !gamma_tagged || !epoch_tagged || !step_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* gamma = tagged_.unpackDouble(gamma_tagged);
    llvm::Value* epoch = tagged_.unpackDouble(epoch_tagged);
    llvm::Value* step_size = tagged_.unpackDouble(step_tagged);

    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pow", &ctx_.module());
    }
    llvm::Function* floor_func = ctx_.module().getFunction("floor");
    if (!floor_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType()}, false);
        floor_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "floor", &ctx_.module());
    }

    // exponent = floor(epoch / step_size)
    llvm::Value* ratio = builder.CreateFDiv(epoch, step_size);
    llvm::Value* exponent = builder.CreateCall(floor_func, {ratio});
    // lr = base_lr * gamma^exponent
    llvm::Value* decay = builder.CreateCall(pow_func, {gamma, exponent});
    llvm::Value* result = builder.CreateFMul(base_lr, decay);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::linearWarmupLR(const eshkol_operations_t* op) {
    // linear-warmup-lr: (linear-warmup-lr base_lr current_step warmup_steps) → lr
    // lr = base_lr * min(1.0, current_step / warmup_steps)
    if (op->call_op.num_vars != 3) {
        eshkol_error("linear-warmup-lr requires 3 arguments: base_lr, current_step, warmup_steps");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* step_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* warmup_tagged = codegenAST(&op->call_op.variables[2]);
    if (!base_lr_tagged || !step_tagged || !warmup_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* step = tagged_.unpackDouble(step_tagged);
    llvm::Value* warmup = tagged_.unpackDouble(warmup_tagged);

    // ratio = step / warmup
    llvm::Value* ratio = builder.CreateFDiv(step, warmup);
    // min(1.0, ratio)
    llvm::Value* one = llvm::ConstantFP::get(ctx_.doubleType(), 1.0);
    llvm::Value* cmp = builder.CreateFCmpOLT(ratio, one);
    llvm::Value* factor = builder.CreateSelect(cmp, ratio, one);
    llvm::Value* result = builder.CreateFMul(base_lr, factor);

    return tagged_.packDouble(result);
}

llvm::Value* TensorCodegen::exponentialDecayLR(const eshkol_operations_t* op) {
    // exponential-decay-lr: (exponential-decay-lr base_lr gamma epoch) → lr
    // lr = base_lr * gamma^epoch
    if (op->call_op.num_vars != 3) {
        eshkol_error("exponential-decay-lr requires 3 arguments: base_lr, gamma, epoch");
        return nullptr;
    }

    llvm::IRBuilder<>& builder = ctx_.builder();

    llvm::Value* base_lr_tagged = codegenAST(&op->call_op.variables[0]);
    llvm::Value* gamma_tagged = codegenAST(&op->call_op.variables[1]);
    llvm::Value* epoch_tagged = codegenAST(&op->call_op.variables[2]);
    if (!base_lr_tagged || !gamma_tagged || !epoch_tagged) return nullptr;

    llvm::Value* base_lr = tagged_.unpackDouble(base_lr_tagged);
    llvm::Value* gamma = tagged_.unpackDouble(gamma_tagged);
    llvm::Value* epoch = tagged_.unpackDouble(epoch_tagged);

    llvm::Function* pow_func = ctx_.module().getFunction("pow");
    if (!pow_func) {
        llvm::FunctionType* ft = llvm::FunctionType::get(ctx_.doubleType(), {ctx_.doubleType(), ctx_.doubleType()}, false);
        pow_func = llvm::Function::Create(ft, llvm::Function::ExternalLinkage, "pow", &ctx_.module());
    }

    llvm::Value* decay = builder.CreateCall(pow_func, {gamma, epoch});
    llvm::Value* result = builder.CreateFMul(base_lr, decay);

    return tagged_.packDouble(result);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
