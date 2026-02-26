/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TailCallCodegen implementation
 *
 * Provides tail call optimization infrastructure for the Eshkol compiler.
 * This module offers:
 * - Utilities for marking LLVM calls as tail calls
 * - Trampoline runtime for bounded-stack closure calls
 * - Future: tail position detection for automatic TCO
 *
 * The implementation is designed to integrate with existing codegen modules
 * without breaking changes, allowing gradual adoption of TCO.
 */

#include <eshkol/backend/tail_call_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

TailCallCodegen::TailCallCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("TailCallCodegen initialized");
}

// === Tail Position Detection ===
// These functions determine if an expression is in tail position,
// which is required for safe tail call optimization.

bool TailCallCodegen::isTailPosition(const eshkol_ast_t* ast, const eshkol_ast_t* parent) const {
    if (!ast) return false;

    // Top-level expression in function body is in tail position
    if (!parent) return true;

    // Only operations can be in tail position
    if (parent->type != ESHKOL_OP) {
        return false;
    }

    // Check if ast is in tail position within parent operation
    if (ast->type == ESHKOL_OP) {
        return isOperationInTailPosition(&ast->operation, &parent->operation);
    }

    return false;
}

bool TailCallCodegen::isOperationInTailPosition(const eshkol_operations_t* op,
                                                 const eshkol_operations_t* parent) const {
    if (!parent) return true;

    switch (parent->op) {
        case ESHKOL_IF_OP:
            // Both branches of if are in tail position
            return (parent->if_op.if_true == op || parent->if_op.if_false == op);

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            // Body of let/let*/letrec/letrec* is in tail position
            if (parent->let_op.body && parent->let_op.body->type == ESHKOL_OP) {
                return &parent->let_op.body->operation == op;
            }
            return false;

        case ESHKOL_LAMBDA_OP:
            // Lambda body is in tail position
            if (parent->lambda_op.body && parent->lambda_op.body->type == ESHKOL_OP) {
                return &parent->lambda_op.body->operation == op;
            }
            return false;

        case ESHKOL_SEQUENCE_OP:
            // Last expression in sequence is in tail position
            if (parent->sequence_op.num_expressions > 0) {
                size_t last_idx = parent->sequence_op.num_expressions - 1;
                if (parent->sequence_op.expressions &&
                    parent->sequence_op.expressions[last_idx].type == ESHKOL_OP) {
                    return &parent->sequence_op.expressions[last_idx].operation == op;
                }
            }
            return false;

        default:
            // Other operations don't create tail contexts
            return false;
    }
}

// === Tail Call Marking ===
// These functions mark LLVM call instructions for tail call optimization.

llvm::CallInst* TailCallCodegen::markTailCall(llvm::CallInst* call, bool in_tail_position) {
    if (!call) return nullptr;

    if (in_tail_position) {
        // Set the tail call flag
        // LLVM will optimize this to reuse the stack frame when possible
        call->setTailCall(true);
        eshkol_debug("Marked call as tail call");
    }

    return call;
}

llvm::Value* TailCallCodegen::createTailCall(llvm::Function* func,
                                              const std::vector<llvm::Value*>& args) {
    llvm::CallInst* call = ctx_.builder().CreateCall(func, args);
    call->setTailCall(true);
    return call;
}

// === Trampoline Support ===
// The trampoline enables bounded-stack deep recursion by using a loop
// instead of stack frames for continuation calls.

llvm::Value* TailCallCodegen::isBounce(llvm::Value* value) {
    // Check if value has the bounce tag
    // First, extract the int64 data from the tagged value
    llvm::Value* data = tagged_.unpackInt64(value);
    llvm::Value* tag_mask = llvm::ConstantInt::get(ctx_.int64Type(), 0xFF00000000000000ULL);
    llvm::Value* tag = ctx_.builder().CreateAnd(data, tag_mask);
    return ctx_.builder().CreateICmpEQ(
        tag,
        llvm::ConstantInt::get(ctx_.int64Type(), BOUNCE_TAG)
    );
}

llvm::Value* TailCallCodegen::extractBounceThunk(llvm::Value* bounce) {
    // Extract the thunk pointer by clearing the tag bits
    // First, extract the int64 data from the tagged value
    llvm::Value* data = tagged_.unpackInt64(bounce);
    llvm::Value* ptr_mask = llvm::ConstantInt::get(ctx_.int64Type(), 0x00FFFFFFFFFFFFFFULL);
    llvm::Value* thunk_int = ctx_.builder().CreateAnd(data, ptr_mask);
    return ctx_.builder().CreateIntToPtr(thunk_int, ctx_.ptrType());
}

void TailCallCodegen::generateTrampolineRuntime() {
    if (trampoline_func_) return;  // Already generated

    // Create reusable trampoline function: tagged_value_t eshkol_trampoline(thunk_t)
    llvm::FunctionType* thunk_type = llvm::FunctionType::get(ctx_.taggedValueType(), {}, false);
    llvm::Type* thunk_ptr_type = ctx_.ptrType();  // Use opaque pointer

    llvm::FunctionType* trampoline_type = llvm::FunctionType::get(
        ctx_.taggedValueType(),
        {thunk_ptr_type},
        false
    );

    trampoline_func_ = llvm::Function::Create(
        trampoline_type,
        llvm::Function::InternalLinkage,
        "eshkol_trampoline",
        &ctx_.module()
    );

    // Save current insert point
    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();

    // Generate trampoline body
    llvm::BasicBlock* entry = llvm::BasicBlock::Create(ctx_.context(), "entry", trampoline_func_);
    llvm::BasicBlock* loop = llvm::BasicBlock::Create(ctx_.context(), "loop", trampoline_func_);
    llvm::BasicBlock* bounce = llvm::BasicBlock::Create(ctx_.context(), "bounce", trampoline_func_);
    llvm::BasicBlock* done = llvm::BasicBlock::Create(ctx_.context(), "done", trampoline_func_);

    ctx_.builder().SetInsertPoint(entry);
    llvm::Value* initial_thunk = trampoline_func_->arg_begin();
    llvm::Value* initial_result = ctx_.builder().CreateCall(thunk_type, initial_thunk, {});
    ctx_.builder().CreateBr(loop);

    ctx_.builder().SetInsertPoint(loop);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "result");
    result->addIncoming(initial_result, entry);
    llvm::Value* is_bounce_val = isBounce(result);
    ctx_.builder().CreateCondBr(is_bounce_val, bounce, done);

    ctx_.builder().SetInsertPoint(bounce);
    llvm::Value* thunk = extractBounceThunk(result);
    llvm::Value* new_result = ctx_.builder().CreateCall(thunk_type, thunk, {});
    result->addIncoming(new_result, bounce);
    ctx_.builder().CreateBr(loop);

    ctx_.builder().SetInsertPoint(done);
    ctx_.builder().CreateRet(result);

    // Restore insert point
    if (saved_bb) {
        ctx_.builder().SetInsertPoint(saved_bb);
    }

    eshkol_debug("Generated trampoline runtime function");
}

// === Tail-Recursive Analysis ===

// Helper: check if an AST expression contains a call to func_name that is NOT in tail position.
// Returns true if ALL self-calls are in tail position (or no self-calls exist).
static bool allSelfCallsInTailPosition(const eshkol_ast_t* expr,
                                        const std::string& func_name,
                                        bool in_tail_pos) {
    if (!expr) return true;

    if (expr->type == ESHKOL_VAR) {
        // Variable reference — not a call, fine
        return true;
    }

    if (expr->type != ESHKOL_OP) {
        // Literal or other non-operation — no calls possible
        return true;
    }

    const eshkol_operations_t* op = &expr->operation;

    switch (op->op) {
        case ESHKOL_CALL_OP: {
            // Check if this is a self-call
            if (op->call_op.func && op->call_op.func->type == ESHKOL_VAR &&
                op->call_op.func->variable.id &&
                std::string(op->call_op.func->variable.id) == func_name) {
                // Self-call found — must be in tail position
                if (!in_tail_pos) return false;
            }
            // Check arguments (they are NOT in tail position)
            for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                if (!allSelfCallsInTailPosition(&op->call_op.variables[i], func_name, false)) {
                    return false;
                }
            }
            // Also check the function expression itself
            if (!allSelfCallsInTailPosition(op->call_op.func, func_name, false)) {
                return false;
            }
            return true;
        }

        case ESHKOL_IF_OP: {
            // The condition is NOT in tail position
            // Both branches inherit the tail position status
            // Note: if_op uses eshkol_operations_t* not eshkol_ast_t*
            // We need to create a temporary wrapper... or handle directly
            // if_op has if_true and if_false as operations pointers
            // Wrap them for recursive check
            eshkol_ast_t true_wrapper = {};
            true_wrapper.type = ESHKOL_OP;
            if (op->if_op.if_true) {
                true_wrapper.operation = *op->if_op.if_true;
                if (!allSelfCallsInTailPosition(&true_wrapper, func_name, in_tail_pos))
                    return false;
            }
            if (op->if_op.if_false) {
                eshkol_ast_t false_wrapper = {};
                false_wrapper.type = ESHKOL_OP;
                false_wrapper.operation = *op->if_op.if_false;
                if (!allSelfCallsInTailPosition(&false_wrapper, func_name, in_tail_pos))
                    return false;
            }
            return true;
        }

        case ESHKOL_SEQUENCE_OP: {
            // Only the last expression is in tail position
            for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                bool is_last = (i == op->sequence_op.num_expressions - 1);
                if (!allSelfCallsInTailPosition(
                        &op->sequence_op.expressions[i], func_name,
                        is_last ? in_tail_pos : false)) {
                    return false;
                }
            }
            return true;
        }

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP: {
            // Binding values are NOT in tail position
            for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                if (!allSelfCallsInTailPosition(&op->let_op.bindings[i], func_name, false)) {
                    return false;
                }
            }
            // Body IS in tail position
            if (!allSelfCallsInTailPosition(op->let_op.body, func_name, in_tail_pos)) {
                return false;
            }
            return true;
        }

        case ESHKOL_COND_OP: {
            // COND_OP uses call_op structure: each variable is a clause
            // Each clause is itself a CALL_OP with test as func and body exprs as vars
            for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                const eshkol_ast_t* clause = &op->call_op.variables[i];
                if (clause->type == ESHKOL_OP && clause->operation.op == ESHKOL_CALL_OP) {
                    // Test is NOT in tail position
                    if (!allSelfCallsInTailPosition(clause->operation.call_op.func, func_name, false))
                        return false;
                    // Last body expression IS in tail position
                    for (uint64_t j = 0; j < clause->operation.call_op.num_vars; j++) {
                        bool is_last_body = (j == clause->operation.call_op.num_vars - 1);
                        if (!allSelfCallsInTailPosition(
                                &clause->operation.call_op.variables[j], func_name,
                                is_last_body ? in_tail_pos : false))
                            return false;
                    }
                }
            }
            return true;
        }

        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP: {
            // AND_OP and OR_OP use sequence_op structure
            // Only the last operand is in tail position
            for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                bool is_last = (i == op->sequence_op.num_expressions - 1);
                if (!allSelfCallsInTailPosition(
                        &op->sequence_op.expressions[i], func_name,
                        is_last ? in_tail_pos : false)) {
                    return false;
                }
            }
            return true;
        }

        case ESHKOL_LAMBDA_OP: {
            // A nested lambda creates its own tail context — self-calls inside
            // a nested lambda are not our self-calls (different function boundary)
            return true;
        }

        default:
            // For other ops (arithmetic, etc.), sub-expressions are not in tail position
            return true;
    }
}

bool TailCallCodegen::isTailRecursive(const eshkol_operations_t* lambda_op,
                                       const std::string& func_name) const {
    if (!lambda_op || lambda_op->op != ESHKOL_LAMBDA_OP) return false;
    if (!lambda_op->lambda_op.body) return false;

    // Walk the lambda body and verify ALL self-calls are in tail position
    // The body of the lambda is in tail position
    return allSelfCallsInTailPosition(lambda_op->lambda_op.body, func_name, true);
}

llvm::Function* TailCallCodegen::transformToIterative(
    const std::string& func_name,
    const std::vector<std::string>& params,
    const eshkol_ast_t* body) {
    // Create an iterative version of a tail-recursive function.
    //
    // The transformation replaces:
    //   (define (f x y) ... (f new_x new_y) ...)
    // With:
    //   function f(x, y):
    //     loop:
    //       x_mut = phi(x_init, x_next)
    //       y_mut = phi(y_init, y_next)
    //       <body code>
    //       // self-calls become: x_next = new_x; y_next = new_y; goto loop
    //       // non-self returns become: return result
    //
    // This function creates the function shell with allocas for mutable params
    // and the loop header. The caller (main codegen) fills in the body, using
    // iterative_info_ to redirect self-calls to store+branch.

    if (params.empty() || !body) {
        eshkol_error("transformToIterative: invalid params or body for '%s'", func_name.c_str());
        return nullptr;
    }

    // Build function type: all params are tagged values, returns tagged value
    std::vector<llvm::Type*> param_types(params.size(), ctx_.taggedValueType());
    llvm::FunctionType* func_type = llvm::FunctionType::get(
        ctx_.taggedValueType(), param_types, false);

    llvm::Function* func = llvm::Function::Create(
        func_type,
        llvm::Function::InternalLinkage,
        func_name + "_iterative",
        &ctx_.module());

    // Name the parameters
    size_t idx = 0;
    for (auto& arg : func->args()) {
        arg.setName(params[idx++]);
    }

    // Save current insert point
    llvm::BasicBlock* saved_bb = ctx_.builder().GetInsertBlock();

    // Create basic blocks
    llvm::BasicBlock* entry_bb = llvm::BasicBlock::Create(ctx_.context(), "entry", func);
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(ctx_.context(), "loop", func);

    // Entry block: create mutable allocas for each parameter
    ctx_.builder().SetInsertPoint(entry_bb);

    std::vector<llvm::AllocaInst*> param_allocas;
    param_allocas.reserve(params.size());
    idx = 0;
    for (auto& arg : func->args()) {
        llvm::AllocaInst* alloca = ctx_.builder().CreateAlloca(
            ctx_.taggedValueType(), nullptr, params[idx] + "_mut");
        ctx_.builder().CreateStore(&arg, alloca);
        param_allocas.push_back(alloca);
        idx++;
    }

    // Branch to loop header
    ctx_.builder().CreateBr(loop_bb);

    // Loop header: load current param values
    ctx_.builder().SetInsertPoint(loop_bb);

    // Store the iterative transformation info so the main codegen can use it
    // to redirect self-calls to store+branch-to-loop
    iterative_info_.func = func;
    iterative_info_.func_name = func_name;
    iterative_info_.param_allocas = param_allocas;
    iterative_info_.loop_header = loop_bb;
    iterative_info_.body = body;
    iterative_info_.active = true;

    eshkol_debug("Created iterative function shell for '%s' with %zu params",
                 func_name.c_str(), params.size());

    // Restore insert point — the caller will fill in loop_bb's body
    if (saved_bb) {
        ctx_.builder().SetInsertPoint(saved_bb);
    }

    return func;
}

bool TailCallCodegen::isLastInSequence(const eshkol_ast_t* expr,
                                        const eshkol_operations_t* parent) const {
    if (!parent || parent->op != ESHKOL_SEQUENCE_OP) {
        return false;
    }

    if (parent->sequence_op.num_expressions == 0) {
        return false;
    }

    // Check if expr is the last expression in the sequence
    size_t last_idx = parent->sequence_op.num_expressions - 1;
    return &parent->sequence_op.expressions[last_idx] == expr;
}

bool TailCallCodegen::canUseDirectTCO(llvm::Function* caller, llvm::Function* callee) const {
    if (!caller || !callee) return false;

    // Direct TCO requires matching calling conventions and return types
    return caller->getCallingConv() == callee->getCallingConv() &&
           caller->getReturnType() == callee->getReturnType();
}

void TailCallCodegen::emitIterativeSelfCall(const std::vector<llvm::Value*>& new_args) {
    if (!iterative_info_.active) {
        eshkol_error("emitIterativeSelfCall called without active iterative transform");
        return;
    }

    if (new_args.size() != iterative_info_.param_allocas.size()) {
        eshkol_error("emitIterativeSelfCall: arg count mismatch (%zu args, %zu params)",
                     new_args.size(), iterative_info_.param_allocas.size());
        return;
    }

    // Store new argument values to the mutable parameter allocas
    for (size_t i = 0; i < new_args.size(); i++) {
        ctx_.builder().CreateStore(new_args[i], iterative_info_.param_allocas[i]);
    }

    // Branch back to loop header
    ctx_.builder().CreateBr(iterative_info_.loop_header);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
