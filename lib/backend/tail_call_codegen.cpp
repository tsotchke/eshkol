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
            // Body of let/let*/letrec is in tail position
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

llvm::Value* TailCallCodegen::createClosureTailCall(llvm::Value* closure,
                                                     const std::vector<llvm::Value*>& args) {
    // Closure tail calls require special handling
    // For now, return nullptr to indicate the caller should use regular closure call
    // TODO: Implement trampoline-based closure tail calls
    return nullptr;
}

// === Trampoline Support ===
// The trampoline enables bounded-stack deep recursion by using a loop
// instead of stack frames for continuation calls.

llvm::Value* TailCallCodegen::createBounce(llvm::Value* thunk_func) {
    // Create a bounce value: a thunk tagged with BOUNCE_TAG
    llvm::Value* thunk_int = ctx_.builder().CreatePtrToInt(thunk_func, ctx_.int64Type());
    return ctx_.builder().CreateOr(
        thunk_int,
        llvm::ConstantInt::get(ctx_.int64Type(), BOUNCE_TAG)
    );
}

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

llvm::Value* TailCallCodegen::createTrampoline(llvm::Value* initial_thunk) {
    // Generate an inline trampoline loop
    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();

    llvm::BasicBlock* entry_bb = ctx_.builder().GetInsertBlock();
    llvm::BasicBlock* loop_bb = llvm::BasicBlock::Create(ctx_.context(), "trampoline_loop", current_func);
    llvm::BasicBlock* bounce_bb = llvm::BasicBlock::Create(ctx_.context(), "trampoline_bounce", current_func);
    llvm::BasicBlock* done_bb = llvm::BasicBlock::Create(ctx_.context(), "trampoline_done", current_func);

    // Create thunk function type: () -> tagged_value
    llvm::FunctionType* thunk_type = llvm::FunctionType::get(ctx_.taggedValueType(), {}, false);

    // Entry: call initial thunk
    llvm::Value* initial_result = ctx_.builder().CreateCall(thunk_type, initial_thunk, {}, "initial_result");
    ctx_.builder().CreateBr(loop_bb);

    // Loop: check if result is a bounce
    ctx_.builder().SetInsertPoint(loop_bb);
    llvm::PHINode* result_phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "result");
    result_phi->addIncoming(initial_result, entry_bb);

    llvm::Value* is_bounce_val = isBounce(result_phi);
    ctx_.builder().CreateCondBr(is_bounce_val, bounce_bb, done_bb);

    // Bounce: extract and call thunk
    ctx_.builder().SetInsertPoint(bounce_bb);
    llvm::Value* thunk = extractBounceThunk(result_phi);
    llvm::Value* bounce_result = ctx_.builder().CreateCall(thunk_type, thunk, {}, "bounce_result");
    result_phi->addIncoming(bounce_result, bounce_bb);
    ctx_.builder().CreateBr(loop_bb);

    // Done: return result
    ctx_.builder().SetInsertPoint(done_bb);
    return result_phi;
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

bool TailCallCodegen::isTailRecursive(const eshkol_operations_t* lambda_op,
                                       const std::string& func_name) const {
    // TODO: Implement full analysis to detect if all self-calls are in tail position
    return false;
}

llvm::Function* TailCallCodegen::transformToIterative(
    const std::string& func_name,
    const std::vector<std::string>& params,
    const eshkol_ast_t* body) {
    // TODO: Transform tail-recursive functions to iterative form
    eshkol_debug("transformToIterative not yet implemented for: %s", func_name.c_str());
    return nullptr;
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

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
