/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ControlFlowCodegen implementation
 *
 * Implements control flow operations:
 * - Boolean logic (and, or, not)
 * - Conditional expressions (if, cond, case)
 * - When/unless conditionals
 * - Sequencing (begin)
 */

#include <eshkol/backend/control_flow_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Constants.h>
#include <cstring>

namespace eshkol {

ControlFlowCodegen::ControlFlowCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged) {
    eshkol_debug("ControlFlowCodegen initialized");
}

llvm::Value* ControlFlowCodegen::isTruthy(llvm::Value* val) {
    if (!val) return llvm::ConstantInt::getFalse(ctx_.context());

    // Handle raw int64 - truthy if non-zero
    if (val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateICmpNE(val, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    }

    // Handle raw double - truthy if non-zero
    if (val->getType()->isDoubleTy()) {
        return ctx_.builder().CreateFCmpONE(val, llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    // Handle tagged_value
    if (val->getType() == ctx_.taggedValueType()) {
        llvm::Value* type = tagged_.getType(val);
        llvm::Value* base_type = ctx_.builder().CreateAnd(type,
            llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

        // Check for false/null (type 0 with value 0)
        llvm::Value* is_null_type = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
        llvm::Value* data = tagged_.unpackInt64(val);
        llvm::Value* is_false_val = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* is_null_or_false = ctx_.builder().CreateAnd(is_null_type, is_false_val);

        // Check for BOOL type with value 0 (#f)
        llvm::Value* is_bool = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
        llvm::Value* bool_is_false = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* bool_false = ctx_.builder().CreateAnd(is_bool, bool_is_false);

        // Check for zero (int or double)
        llvm::Value* is_int = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
        llvm::Value* int_is_zero = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* int_zero = ctx_.builder().CreateAnd(is_int, int_is_zero);

        llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* double_val = tagged_.unpackDouble(val);
        llvm::Value* double_is_zero = ctx_.builder().CreateFCmpOEQ(double_val,
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* double_zero = ctx_.builder().CreateAnd(is_double, double_is_zero);

        // Truthy = NOT (null/false OR bool-false OR int-zero OR double-zero)
        llvm::Value* is_falsy = ctx_.builder().CreateOr(is_null_or_false,
            ctx_.builder().CreateOr(bool_false, ctx_.builder().CreateOr(int_zero, double_zero)));
        return ctx_.builder().CreateNot(is_falsy);
    }

    // Default: assume truthy
    return llvm::ConstantInt::getTrue(ctx_.context());
}

// ============================================================================
// Boolean Logic
// ============================================================================

llvm::Value* ControlFlowCodegen::codegenAnd(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenAnd - callbacks not set");
        return tagged_.packNull();
    }

    // ESHKOL_AND_OP uses sequence_op, ESHKOL_CALL_OP uses call_op
    uint64_t num_args = (op->op == ESHKOL_AND_OP) ?
        op->sequence_op.num_expressions : op->call_op.num_vars;
    const eshkol_ast_t* args = (op->op == ESHKOL_AND_OP) ?
        op->sequence_op.expressions : op->call_op.variables;

    if (num_args == 0) {
        // (and) with no args returns #t
        return tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "and_merge", current_func);

    // For collecting PHI inputs
    std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>> phi_inputs;

    // Evaluate each argument with short-circuit
    for (uint64_t i = 0; i < num_args; i++) {
        // Use callback to get typed AST result
        void* tv_ptr = codegen_typed_ast_callback_(&args[i], callback_context_);
        if (!tv_ptr) return nullptr;
        llvm::Value* arg = typed_to_tagged_callback_(tv_ptr, callback_context_);

        llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();

        if (i == num_args - 1) {
            // Last argument - always return its value
            phi_inputs.push_back({arg, current_block});
            ctx_.builder().CreateBr(merge_block);
        } else {
            // Check if truthy
            llvm::Value* is_truthy_val = isTruthy(arg);
            llvm::BasicBlock* next_block = llvm::BasicBlock::Create(ctx_.context(), "and_next", current_func);
            llvm::BasicBlock* short_circuit_block = llvm::BasicBlock::Create(ctx_.context(), "and_short", current_func);

            // If truthy, continue to next arg; if falsy, short-circuit
            ctx_.builder().CreateCondBr(is_truthy_val, next_block, short_circuit_block);

            // Short-circuit: return this falsy value
            ctx_.builder().SetInsertPoint(short_circuit_block);
            phi_inputs.push_back({arg, short_circuit_block});
            ctx_.builder().CreateBr(merge_block);

            // Continue evaluation with next arg
            ctx_.builder().SetInsertPoint(next_block);
        }
    }

    // Merge block with PHI
    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "and_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }

    return result;
}

llvm::Value* ControlFlowCodegen::codegenOr(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenOr - callbacks not set");
        return tagged_.packNull();
    }

    // ESHKOL_OR_OP uses sequence_op, ESHKOL_CALL_OP uses call_op
    uint64_t num_args = (op->op == ESHKOL_OR_OP) ?
        op->sequence_op.num_expressions : op->call_op.num_vars;
    const eshkol_ast_t* args = (op->op == ESHKOL_OR_OP) ?
        op->sequence_op.expressions : op->call_op.variables;

    if (num_args == 0) {
        // (or) with no args returns #f
        return tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "or_merge", current_func);

    std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>> phi_inputs;

    for (uint64_t i = 0; i < num_args; i++) {
        void* tv_ptr = codegen_typed_ast_callback_(&args[i], callback_context_);
        if (!tv_ptr) return nullptr;
        llvm::Value* arg = typed_to_tagged_callback_(tv_ptr, callback_context_);

        llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();

        if (i == num_args - 1) {
            // Last arg - just branch to merge with this value
            phi_inputs.push_back({arg, current_block});
            ctx_.builder().CreateBr(merge_block);
        } else {
            llvm::Value* is_truthy_val = isTruthy(arg);
            llvm::BasicBlock* next_block = llvm::BasicBlock::Create(ctx_.context(), "or_next", current_func);
            llvm::BasicBlock* short_circuit_block = llvm::BasicBlock::Create(ctx_.context(), "or_short", current_func);

            // If truthy, go to short_circuit_block; otherwise continue to next_block
            ctx_.builder().CreateCondBr(is_truthy_val, short_circuit_block, next_block);

            // Short circuit block - branch to merge with the truthy value
            ctx_.builder().SetInsertPoint(short_circuit_block);
            phi_inputs.push_back({arg, short_circuit_block});
            ctx_.builder().CreateBr(merge_block);

            // Continue evaluation in next_block
            ctx_.builder().SetInsertPoint(next_block);
        }
    }

    ctx_.builder().SetInsertPoint(merge_block);
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "or_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }

    return result;
}

llvm::Value* ControlFlowCodegen::codegenNot(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenNot - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 1) {
        eshkol_warn("not requires exactly 1 argument");
        return nullptr;
    }

    void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!tv_ptr) return nullptr;
    llvm::Value* arg = typed_to_tagged_callback_(tv_ptr, callback_context_);

    llvm::Value* is_truthy_val = isTruthy(arg);
    llvm::Value* is_false = ctx_.builder().CreateNot(is_truthy_val);
    return tagged_.packBool(is_false);
}

// ============================================================================
// Conditional Expressions
// ============================================================================

llvm::Value* ControlFlowCodegen::codegenCond(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenCond - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars == 0) {
        eshkol_warn("cond requires at least one clause");
        return nullptr;
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "cond_done", current_func);

    std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>> phi_inputs;

    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        const eshkol_ast_t* clause = &op->call_op.variables[i];

        // Each clause should be a list (test expr...)
        if (clause->type != ESHKOL_OP || clause->operation.op != ESHKOL_CALL_OP) {
            eshkol_warn("cond clause must be a list");
            continue;
        }

        // Check if this is an 'else' clause
        bool is_else = false;
        if (clause->operation.call_op.func &&
            clause->operation.call_op.func->type == ESHKOL_VAR &&
            clause->operation.call_op.func->variable.id) {
            const char* sym = clause->operation.call_op.func->variable.id;
            is_else = (strcmp(sym, "else") == 0);
        }

        if (is_else) {
            // else clause - evaluate expressions and we're done
            llvm::Value* result = nullptr;
            for (uint64_t j = 0; j < clause->operation.call_op.num_vars; j++) {
                void* tv_ptr = codegen_typed_ast_callback_(&clause->operation.call_op.variables[j], callback_context_);
                if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
            }
            // TCO FIX: Check if block is already terminated
            llvm::BasicBlock* else_block = ctx_.builder().GetInsertBlock();
            bool else_terminated = else_block->getTerminator() != nullptr;
            if (result) {
                phi_inputs.push_back({result, else_block});
            }
            // TCO FIX: Only add branch if block isn't already terminated
            if (!else_terminated) {
                ctx_.builder().CreateBr(done_block);
            }
            break;
        } else {
            // Regular clause - evaluate test (func is the test condition)
            void* test_tv = codegen_typed_ast_callback_(clause->operation.call_op.func, callback_context_);
            if (!test_tv) continue;
            llvm::Value* test = typed_to_tagged_callback_(test_tv, callback_context_);

            llvm::Value* is_true = isTruthy(test);
            llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ctx_.context(), "cond_then", current_func);
            llvm::BasicBlock* next_block = llvm::BasicBlock::Create(ctx_.context(), "cond_next", current_func);

            ctx_.builder().CreateCondBr(is_true, then_block, next_block);

            // Then block - evaluate expressions (clause body)
            ctx_.builder().SetInsertPoint(then_block);
            llvm::Value* result = nullptr;
            for (uint64_t j = 0; j < clause->operation.call_op.num_vars; j++) {
                void* tv_ptr = codegen_typed_ast_callback_(&clause->operation.call_op.variables[j], callback_context_);
                if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
            }
            // TCO FIX: Update then_block to current block (code gen may have created new blocks)
            then_block = ctx_.builder().GetInsertBlock();
            bool then_terminated = then_block->getTerminator() != nullptr;
            if (result) {
                phi_inputs.push_back({result, then_block});
            }
            // TCO FIX: Only add branch if block isn't already terminated
            if (!then_terminated) {
                ctx_.builder().CreateBr(done_block);
            }

            // Continue to next clause
            ctx_.builder().SetInsertPoint(next_block);
        }
    }

    // If no clause matched, return false
    if (phi_inputs.empty() || ctx_.builder().GetInsertBlock()->getTerminator() == nullptr) {
        phi_inputs.push_back({tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context())), ctx_.builder().GetInsertBlock()});
        ctx_.builder().CreateBr(done_block);
    }

    ctx_.builder().SetInsertPoint(done_block);
    if (phi_inputs.size() == 1) {
        return phi_inputs[0].first;
    }
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "cond_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }

    return result;
}

llvm::Value* ControlFlowCodegen::codegenIf(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenIf - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars != 3) {
        eshkol_warn("if requires exactly 3 arguments: condition, then-expr, else-expr");
        return nullptr;
    }

    // Generate condition
    llvm::Value* condition = codegen_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!condition) return nullptr;

    // Convert condition to boolean
    llvm::Value* cond_bool;
    if (condition->getType() == ctx_.taggedValueType()) {
        cond_bool = isTruthy(condition);
    } else if (condition->getType()->isIntegerTy(64)) {
        cond_bool = ctx_.builder().CreateICmpNE(condition, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    } else {
        cond_bool = llvm::ConstantInt::getTrue(ctx_.context());
    }

    llvm::Function* function = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ctx_.context(), "then", function);
    llvm::BasicBlock* else_block = llvm::BasicBlock::Create(ctx_.context(), "else", function);
    llvm::BasicBlock* merge_block = llvm::BasicBlock::Create(ctx_.context(), "ifcont", function);

    ctx_.builder().CreateCondBr(cond_bool, then_block, else_block);

    // Generate then block
    ctx_.builder().SetInsertPoint(then_block);
    llvm::Value* then_value = codegen_ast_callback_(&op->call_op.variables[1], callback_context_);
    if (!then_value) {
        then_value = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }
    // TCO FIX: Only add branch if block isn't already terminated (tail calls terminate the block)
    then_block = ctx_.builder().GetInsertBlock();
    bool then_terminated = then_block->getTerminator() != nullptr;
    if (!then_terminated) {
        ctx_.builder().CreateBr(merge_block);
    }

    // Generate else block
    ctx_.builder().SetInsertPoint(else_block);
    llvm::Value* else_value = codegen_ast_callback_(&op->call_op.variables[2], callback_context_);
    if (!else_value) {
        else_value = llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }
    // TCO FIX: Only add branch if block isn't already terminated (tail calls terminate the block)
    else_block = ctx_.builder().GetInsertBlock();
    bool else_terminated = else_block->getTerminator() != nullptr;
    if (!else_terminated) {
        ctx_.builder().CreateBr(merge_block);
    }

    // TCO FIX: Handle cases where one or both branches are terminated by tail calls
    if (then_terminated && else_terminated) {
        // Both branches are tail calls - no merge block needed
        // Remove the unused merge block
        merge_block->eraseFromParent();
        // CRITICAL: Don't call packNull() here - we're in a terminated block!
        // packNull() tries to add instructions which would fail LLVM verification.
        // Return UndefValue since this value is never used (control flow diverged via TCO)
        return llvm::UndefValue::get(ctx_.taggedValueType());
    }

    // Determine result type
    // HoTT FIX: Use tagged_value when types differ to preserve type information
    llvm::Type* result_type;
    if (then_value->getType() == ctx_.taggedValueType() || else_value->getType() == ctx_.taggedValueType()) {
        result_type = ctx_.taggedValueType();
    } else if (then_value->getType() != else_value->getType()) {
        // Types differ (e.g., int64 vs double) - use tagged_value to preserve type info
        result_type = ctx_.taggedValueType();
    } else {
        result_type = then_value->getType();  // Both same type, use it directly
    }

    // Convert values to result type if needed (only for non-terminated branches)
    auto convertToResultType = [&](llvm::Value* val, llvm::BasicBlock* block, bool is_terminated) -> llvm::Value* {
        if (is_terminated) return val;  // Don't touch terminated blocks
        if (val->getType() == result_type) return val;

        llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();
        ctx_.builder().SetInsertPoint(block->getTerminator());

        llvm::Value* converted;
        if (result_type == ctx_.taggedValueType()) {
            // Use detect_and_pack_callback_ to properly identify CONS_PTR from PtrToInt values
            if (detect_and_pack_callback_ && val->getType()->isIntegerTy(64)) {
                converted = detect_and_pack_callback_(val, callback_context_);
            } else if (val->getType()->isDoubleTy()) {
                converted = tagged_.packDouble(val);
            } else if (val->getType()->isIntegerTy()) {
                llvm::Value* extended = ctx_.builder().CreateSExt(val, ctx_.int64Type());
                if (detect_and_pack_callback_) {
                    converted = detect_and_pack_callback_(extended, callback_context_);
                } else {
                    converted = tagged_.packInt64(extended, true);
                }
            } else {
                converted = tagged_.packInt64(llvm::ConstantInt::get(ctx_.int64Type(), 0), true);
            }
        } else {
            if (val->getType() == ctx_.taggedValueType()) {
                converted = tagged_.unpackInt64(val);
            } else if (val->getType()->isIntegerTy()) {
                if (val->getType()->getIntegerBitWidth() < 64) {
                    converted = ctx_.builder().CreateSExt(val, result_type);
                } else {
                    converted = ctx_.builder().CreateTrunc(val, result_type);
                }
            } else {
                converted = llvm::ConstantInt::get(result_type, 0);
            }
        }

        ctx_.builder().SetInsertPoint(current_block);
        return converted;
    };

    if (!then_terminated) {
        then_value = convertToResultType(then_value, then_block, then_terminated);
    }
    if (!else_terminated) {
        else_value = convertToResultType(else_value, else_block, else_terminated);
    }

    ctx_.builder().SetInsertPoint(merge_block);

    // Create PHI node only with branches that actually reach the merge block
    if (then_terminated) {
        // Only else branch reaches merge - no phi needed, just return else value
        return else_value;
    } else if (else_terminated) {
        // Only then branch reaches merge - no phi needed, just return then value
        return then_value;
    }

    // Both branches reach merge - use phi node
    llvm::PHINode* phi = ctx_.builder().CreatePHI(result_type, 2, "iftmp");
    phi->addIncoming(then_value, then_block);
    phi->addIncoming(else_value, else_block);

    return phi;
}

// ============================================================================
// When/Unless Conditionals
// ============================================================================

llvm::Value* ControlFlowCodegen::codegenWhen(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenWhen - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1) {
        eshkol_warn("when requires at least a test expression");
        return nullptr;
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ctx_.context(), "when_then", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "when_done", current_func);

    // Evaluate test condition
    void* test_tv = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!test_tv) return nullptr;
    llvm::Value* test = typed_to_tagged_callback_(test_tv, callback_context_);
    llvm::Value* is_true = isTruthy(test);

    // Create the false result before branching
    llvm::Value* false_result = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    llvm::BasicBlock* branch_block = ctx_.builder().GetInsertBlock();

    ctx_.builder().CreateCondBr(is_true, then_block, done_block);

    // Then block - evaluate body expressions
    ctx_.builder().SetInsertPoint(then_block);
    llvm::Value* result = nullptr;
    for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
        void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
    }
    if (!result) result = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    llvm::BasicBlock* then_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(done_block);

    // Done block - PHI for result
    ctx_.builder().SetInsertPoint(done_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "when_result");
    phi->addIncoming(result, then_exit);
    phi->addIncoming(false_result, branch_block);

    return phi;
}

llvm::Value* ControlFlowCodegen::codegenUnless(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenUnless - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars < 1) {
        eshkol_warn("unless requires at least a test expression");
        return nullptr;
    }

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* else_block = llvm::BasicBlock::Create(ctx_.context(), "unless_else", current_func);
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "unless_done", current_func);

    // Evaluate test condition
    void* test_tv = codegen_typed_ast_callback_(&op->call_op.variables[0], callback_context_);
    if (!test_tv) return nullptr;
    llvm::Value* test = typed_to_tagged_callback_(test_tv, callback_context_);
    llvm::Value* is_true = isTruthy(test);

    // Create the false result before branching
    llvm::Value* false_result = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
    llvm::BasicBlock* branch_block = ctx_.builder().GetInsertBlock();

    // Branch to else block if test is FALSE
    ctx_.builder().CreateCondBr(is_true, done_block, else_block);

    // Else block - evaluate body expressions (when test is false)
    ctx_.builder().SetInsertPoint(else_block);
    llvm::Value* result = nullptr;
    for (uint64_t i = 1; i < op->call_op.num_vars; i++) {
        void* tv_ptr = codegen_typed_ast_callback_(&op->call_op.variables[i], callback_context_);
        if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
    }
    if (!result) result = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    llvm::BasicBlock* else_exit = ctx_.builder().GetInsertBlock();
    ctx_.builder().CreateBr(done_block);

    // Done block - PHI for result
    ctx_.builder().SetInsertPoint(done_block);
    llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "unless_result");
    phi->addIncoming(result, else_exit);
    phi->addIncoming(false_result, branch_block);

    return phi;
}

// ============================================================================
// Case Expression
// ============================================================================

llvm::Value* ControlFlowCodegen::codegenCase(const eshkol_operations_t* op) {
    if (!codegen_typed_ast_callback_ || !typed_to_tagged_callback_ || !eqv_compare_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenCase - callbacks not set");
        return tagged_.packNull();
    }

    if (!op->call_op.func) {
        eshkol_warn("case requires a key expression");
        return nullptr;
    }

    // Evaluate the key expression once
    void* key_tv = codegen_typed_ast_callback_(op->call_op.func, callback_context_);
    if (!key_tv) return nullptr;
    llvm::Value* key = typed_to_tagged_callback_(key_tv, callback_context_);

    llvm::Function* current_func = ctx_.builder().GetInsertBlock()->getParent();
    llvm::BasicBlock* done_block = llvm::BasicBlock::Create(ctx_.context(), "case_done", current_func);

    std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>> phi_inputs;

    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        const eshkol_ast_t* clause = &op->call_op.variables[i];

        // Each clause is a CONS: car = datums, cdr = body
        if (clause->type != ESHKOL_CONS) {
            eshkol_warn("case clause must be a cons cell");
            continue;
        }

        const eshkol_ast_t* datums_ast = clause->cons_cell.car;
        const eshkol_ast_t* body_ast = clause->cons_cell.cdr;

        // Check if this is an 'else' clause
        bool is_else = false;
        if (datums_ast && datums_ast->type == ESHKOL_VAR && datums_ast->variable.id) {
            is_else = (strcmp(datums_ast->variable.id, "else") == 0);
        }

        if (is_else) {
            // else clause - evaluate body expressions
            llvm::Value* result = nullptr;
            if (body_ast && body_ast->type == ESHKOL_OP &&
                body_ast->operation.op == ESHKOL_CALL_OP) {
                for (uint64_t j = 0; j < body_ast->operation.call_op.num_vars; j++) {
                    void* tv_ptr = codegen_typed_ast_callback_(&body_ast->operation.call_op.variables[j], callback_context_);
                    if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
                }
            }
            if (result) {
                phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
            }
            ctx_.builder().CreateBr(done_block);
            break;
        } else {
            // Regular clause - datums_ast is a CALL_OP with variables containing datums
            llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ctx_.context(), "case_then", current_func);
            llvm::BasicBlock* next_block = llvm::BasicBlock::Create(ctx_.context(), "case_next", current_func);

            llvm::Value* any_match = llvm::ConstantInt::getFalse(ctx_.context());

            if (datums_ast && datums_ast->type == ESHKOL_OP &&
                datums_ast->operation.op == ESHKOL_CALL_OP) {
                // Check all datums
                for (uint64_t d = 0; d < datums_ast->operation.call_op.num_vars; d++) {
                    void* datum_tv = codegen_typed_ast_callback_(&datums_ast->operation.call_op.variables[d], callback_context_);
                    if (!datum_tv) continue;
                    llvm::Value* datum = typed_to_tagged_callback_(datum_tv, callback_context_);
                    llvm::Value* is_match = eqv_compare_callback_(key, datum, callback_context_);
                    any_match = ctx_.builder().CreateOr(any_match, is_match);
                }
            }

            ctx_.builder().CreateCondBr(any_match, then_block, next_block);

            // Then block - evaluate body expressions
            ctx_.builder().SetInsertPoint(then_block);
            llvm::Value* result = nullptr;
            if (body_ast && body_ast->type == ESHKOL_OP &&
                body_ast->operation.op == ESHKOL_CALL_OP) {
                for (uint64_t j = 0; j < body_ast->operation.call_op.num_vars; j++) {
                    void* tv_ptr = codegen_typed_ast_callback_(&body_ast->operation.call_op.variables[j], callback_context_);
                    if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
                }
            }
            if (result) {
                phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
            }
            ctx_.builder().CreateBr(done_block);

            ctx_.builder().SetInsertPoint(next_block);
        }
    }

    // If no clause matched, return unspecified (false)
    if (phi_inputs.empty() || ctx_.builder().GetInsertBlock()->getTerminator() == nullptr) {
        phi_inputs.push_back({tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context())), ctx_.builder().GetInsertBlock()});
        ctx_.builder().CreateBr(done_block);
    }

    ctx_.builder().SetInsertPoint(done_block);
    if (phi_inputs.size() == 1) {
        return phi_inputs[0].first;
    }
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "case_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }

    return result;
}

// ============================================================================
// Sequencing
// ============================================================================

llvm::Value* ControlFlowCodegen::codegenBegin(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
        eshkol_warn("ControlFlowCodegen::codegenBegin - callbacks not set");
        return tagged_.packNull();
    }

    if (op->call_op.num_vars == 0) {
        eshkol_warn("begin requires at least 1 expression");
        return llvm::ConstantInt::get(ctx_.int64Type(), 0);
    }

    // ESHKOL EXTENSION: Collect ALL defines from anywhere in the body
    // This allows statements before/between defines (unlike strict R5RS Scheme)
    std::vector<const eshkol_ast_t*> defines;
    std::vector<const eshkol_ast_t*> non_defines;

    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        const eshkol_ast_t* expr = &op->call_op.variables[i];
        if (expr->type == ESHKOL_OP && expr->operation.op == ESHKOL_DEFINE_OP) {
            defines.push_back(expr);
        } else {
            non_defines.push_back(expr);
        }
    }

    // If there are internal defines, handle them specially
    if (!defines.empty() && codegen_func_define_callback_ && codegen_var_define_callback_) {

        // First process ALL defines (letrec-like: all bindings visible to all)
        for (const eshkol_ast_t* def : defines) {
            if (def->operation.define_op.is_function) {
                codegen_func_define_callback_(&def->operation, callback_context_);
            } else {
                codegen_var_define_callback_(&def->operation, callback_context_);
            }
        }

        // Then execute non-define expressions
        llvm::Value* last_value = nullptr;
        for (const eshkol_ast_t* expr : non_defines) {
            last_value = codegen_ast_callback_(expr, callback_context_);
        }

        return last_value ? last_value : tagged_.packNull();
    }

    // No internal defines - just execute expressions in sequence
    llvm::Value* last_value = nullptr;
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        last_value = codegen_ast_callback_(&op->call_op.variables[i], callback_context_);
    }

    return last_value ? last_value : llvm::ConstantInt::get(ctx_.int64Type(), 0);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
