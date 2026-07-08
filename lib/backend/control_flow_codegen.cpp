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

/**
 * @brief Construct a ControlFlowCodegen bound to the shared codegen context and tagged-value helper.
 *
 * @param ctx Shared LLVM codegen context (IR builder, LLVM context, type helpers).
 * @param tagged Helper for packing/unpacking eshkol tagged values.
 */
ControlFlowCodegen::ControlFlowCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged) {
    eshkol_debug("ControlFlowCodegen initialized");
}

/**
 * @brief Compute the Scheme truthiness of an LLVM value as an i1 boolean.
 *
 * Raw i1 values are returned as-is. Raw int64 and double values are always
 * truthy (Scheme's 0 and 0.0 are not false). For an eshkol tagged_value, only
 * a BOOL-tagged value carrying 0 is falsy (#f) — every other tagged value,
 * including '(), "", and 0, is truthy. Any other/unrecognized LLVM type
 * defaults to truthy.
 *
 * @param val Value to test; may be null, i1, i64, double, or tagged_value.
 * @return i1 LLVM value that is true iff @p val is Scheme-truthy.
 */
llvm::Value* ControlFlowCodegen::isTruthy(llvm::Value* val) {
    if (!val) return llvm::ConstantInt::getFalse(ctx_.context());

    // Handle raw i1 boolean - truthy if true
    if (val->getType()->isIntegerTy(1)) {
        return val;  // Already i1, use directly
    }

    // Handle raw int64 - ALWAYS truthy in Scheme (0 is truthy!)
    if (val->getType()->isIntegerTy(64)) {
        return llvm::ConstantInt::getTrue(ctx_.context());
    }

    // Handle raw double - ALWAYS truthy in Scheme (0.0 is truthy!)
    if (val->getType()->isDoubleTy()) {
        return llvm::ConstantInt::getTrue(ctx_.context());
    }

    // Handle tagged_value
    if (val->getType() == ctx_.taggedValueType()) {
        llvm::Value* type = tagged_.getType(val);
        // Use getBaseType() to properly handle legacy types (>=32) and exactness flags
        // DO NOT use 0x0F mask - 32 & 0x0F = 0 (NULL) which is WRONG!
        llvm::Value* base_type = tagged_.getBaseType(type);

        // SCHEME TRUTHINESS: Only #f is false. Everything else is truthy.
        // This includes: 0, 0.0, '(), "", and all other values.
        // Check for BOOL type with value 0 (#f)
        llvm::Value* is_bool = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
        llvm::Value* data = tagged_.unpackInt64(val);
        llvm::Value* bool_is_false = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* is_false = ctx_.builder().CreateAnd(is_bool, bool_is_false);

        // Truthy = NOT false (only #f is false)
        return ctx_.builder().CreateNot(is_false);
    }

    // Default: assume truthy
    return llvm::ConstantInt::getTrue(ctx_.context());
}

// ============================================================================
// Boolean Logic
// ============================================================================

/**
 * @brief Generate short-circuiting code for `(and a b ...)`.
 *
 * Builds one basic block per argument: each non-final argument is evaluated,
 * tested with isTruthy(), and if falsy the walk short-circuits by branching
 * straight to a shared merge block carrying that falsy value; otherwise
 * control falls through to evaluate the next argument. The last argument's
 * value is always forwarded to the merge block unconditionally. All argument
 * values are coerced to tagged_value_type before reaching the merge PHI node
 * so the incoming edges are type-consistent. Handles the zero-argument case
 * `(and)` (returns #t) and blocks already terminated by a noreturn
 * sub-expression (e.g. `raise`).
 *
 * @param op AST operation node for `and` (ESHKOL_AND_OP) or an equivalent call op.
 * @return Tagged boolean/value result of the short-circuit evaluation.
 */
llvm::Value* ControlFlowCodegen::codegenAnd(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
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

    std::vector<std::pair<llvm::Value*, llvm::BasicBlock*>> phi_inputs;

    for (uint64_t i = 0; i < num_args; i++) {
        // Use direct AST callback (not typed path) to avoid emitUseAfterMoveCheck
        // creating intermediate blocks that shift the builder's insertion point
        llvm::Value* arg = codegen_ast_callback_(&args[i], callback_context_);
        if (!arg) return nullptr;
        // Ensure tagged_value_type for PHI consistency
        if (arg->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
            arg = detect_and_pack_callback_(arg, callback_context_);
        }

        llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();

        // NORETURN SAFETY: If block is already terminated (e.g., by raise in sub-expression),
        // we cannot emit any more instructions. Break out of the loop.
        if (current_block->getTerminator()) {
            break;
        }

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

    // Handle fallthrough: if the last block has no terminator, branch to merge
    llvm::BasicBlock* fallthrough_bb = ctx_.builder().GetInsertBlock();
    if (fallthrough_bb && !fallthrough_bb->getTerminator()) {
        // Last expression didn't terminate — need to branch to merge with a default value
        llvm::Value* default_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
        phi_inputs.push_back({default_val, fallthrough_bb});
        ctx_.builder().CreateBr(merge_block);
    }

    // If no inputs reach merge, remove it
    if (!merge_block->hasNPredecessorsOrMore(1)) {
        merge_block->eraseFromParent();
        return llvm::UndefValue::get(ctx_.taggedValueType());
    }

    ctx_.builder().SetInsertPoint(merge_block);
    if (phi_inputs.size() == 1) {
        return phi_inputs[0].first;
    }
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "and_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }
    return result;
}

/**
 * @brief Generate short-circuiting code for `(or a b ...)`.
 *
 * Mirrors codegenAnd()'s basic-block-per-argument structure, but branches to
 * the short-circuit block when an argument is truthy (returning that value)
 * and continues to the next argument otherwise. The last argument's value is
 * always forwarded to the merge block. Handles the zero-argument case `(or)`
 * (returns #f) and blocks already terminated by a noreturn sub-expression.
 *
 * @param op AST operation node for `or` (ESHKOL_OR_OP) or an equivalent call op.
 * @return Tagged boolean/value result of the short-circuit evaluation.
 */
llvm::Value* ControlFlowCodegen::codegenOr(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
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
        // Use direct AST callback (not typed path) to avoid emitUseAfterMoveCheck
        // creating intermediate blocks that shift the builder's insertion point
        llvm::Value* arg = codegen_ast_callback_(&args[i], callback_context_);
        if (!arg) return nullptr;
        // Ensure tagged_value_type for PHI consistency
        if (arg->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
            arg = detect_and_pack_callback_(arg, callback_context_);
        }

        llvm::BasicBlock* current_block = ctx_.builder().GetInsertBlock();

        // NORETURN SAFETY: If block is already terminated, stop
        if (current_block->getTerminator()) {
            break;
        }

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

    // Handle fallthrough: if the last block has no terminator, branch to merge
    llvm::BasicBlock* or_fallthrough = ctx_.builder().GetInsertBlock();
    if (or_fallthrough && !or_fallthrough->getTerminator()) {
        llvm::Value* default_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
        phi_inputs.push_back({default_val, or_fallthrough});
        ctx_.builder().CreateBr(merge_block);
    }

    // If no inputs reach merge, remove it
    if (!merge_block->hasNPredecessorsOrMore(1)) {
        merge_block->eraseFromParent();
        return llvm::UndefValue::get(ctx_.taggedValueType());
    }

    ctx_.builder().SetInsertPoint(merge_block);
    if (phi_inputs.size() == 1) {
        return phi_inputs[0].first;
    }
    llvm::PHINode* result = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "or_result");
    for (auto& [val, block] : phi_inputs) {
        result->addIncoming(val, block);
    }

    return result;
}

/**
 * @brief Generate code for `(not x)`.
 *
 * Evaluates the single argument via the typed AST callback, converts it to a
 * tagged value, tests it with isTruthy(), and packs the logical negation as a
 * tagged boolean.
 *
 * @param op Call operation AST node; must have exactly one argument.
 * @return Tagged boolean value: #t if the argument is falsy, #f otherwise (or
 *         nullptr if the argument count is wrong or codegen fails).
 */
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

/**
 * @brief Generate code for `(cond (test1 expr...) (test2 expr...) ... [else expr...])`.
 *
 * Walks the clauses in order, emitting a conditional branch per non-`else`
 * clause: a `then` block evaluates the clause body and branches to a shared
 * `done` block, and a `next` block continues the search. Supports the R7RS
 * §5.5 `(test => receiver)` arrow form by applying the receiver closure to
 * the test's value via the closure-call callback, and the R7RS §6.3 `(test)`
 * no-body clause by returning the test's own value when truthy. An `else`
 * clause (if present) always terminates the clause walk. Each reachable
 * clause exit is collected as a PHI input on the merge block; clauses whose
 * body diverges via TCO/tail-call/raise are excluded from the merge. If
 * control falls through every clause unmatched, a default `#f` PHI input is
 * added. If no clause exit reaches `done_block` (e.g. every clause tail-calls
 * out), the block is discarded and an undef value is returned.
 *
 * @param op Call operation AST node holding the cond clauses.
 * @return Tagged value result of whichever clause matched (or #f if none did).
 */
llvm::Value* ControlFlowCodegen::codegenCond(const eshkol_operations_t* op) {
    if (!codegen_ast_callback_) {
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
            // Else clause — evaluate body expressions using direct AST callback
            // (same path as codegenIf, avoids typed callback block creation issues)
            llvm::Value* result = nullptr;
            for (uint64_t j = 0; j < clause->operation.call_op.num_vars; j++) {
                if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
                result = codegen_ast_callback_(&clause->operation.call_op.variables[j], callback_context_);
            }
            llvm::BasicBlock* else_exit = ctx_.builder().GetInsertBlock();
            bool else_terminated = else_exit->getTerminator() != nullptr;
            if (!else_terminated) {
                if (!result) result = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
                // Ensure result is tagged_value_type for PHI consistency
                if (result->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
                    result = detect_and_pack_callback_(result, callback_context_);
                }
                phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
                ctx_.builder().CreateBr(done_block);
            }
            break;
        } else {
            // R7RS §5.5 `=>` receiver form: (test => receiver). The test is
            // evaluated once; if truthy, `receiver` (which must yield a
            // procedure) is applied to the test's value and the cond returns
            // that result. The parser lowers this to a clause whose body is
            // [`=>`, receiver], so detect a leading `=>` identifier.
            bool is_arrow = (clause->operation.call_op.num_vars == 2 &&
                clause->operation.call_op.variables[0].type == ESHKOL_VAR &&
                clause->operation.call_op.variables[0].variable.id &&
                strcmp(clause->operation.call_op.variables[0].variable.id, "=>") == 0);

            // Regular clause — evaluate test using direct AST callback
            llvm::Value* test = codegen_ast_callback_(clause->operation.call_op.func, callback_context_);
            if (!test) continue;

            llvm::Value* is_true = isTruthy(test);
            llvm::BasicBlock* then_block = llvm::BasicBlock::Create(ctx_.context(), "cond_then", current_func);
            llvm::BasicBlock* next_block = llvm::BasicBlock::Create(ctx_.context(), "cond_next", current_func);

            ctx_.builder().CreateCondBr(is_true, then_block, next_block);

            // Then block — evaluate body expressions using direct AST callback
            ctx_.builder().SetInsertPoint(then_block);
            llvm::Value* result = nullptr;
            if (is_arrow) {
                // Evaluate the receiver to a procedure value, then apply it to
                // the (tagged) test value via the closure-call dispatcher.
                llvm::Value* receiver = codegen_ast_callback_(
                    &clause->operation.call_op.variables[1], callback_context_);
                llvm::Value* test_val = test;
                if (test_val->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
                    test_val = detect_and_pack_callback_(test_val, callback_context_);
                }
                if (receiver && receiver->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
                    receiver = detect_and_pack_callback_(receiver, callback_context_);
                }
                if (receiver && closure_call_callback_) {
                    std::vector<llvm::Value*> args{test_val};
                    result = closure_call_callback_(receiver, args, callback_context_);
                } else {
                    eshkol_warn("cond `=>` requires a procedure receiver");
                    result = test_val;
                }
            } else
            for (uint64_t j = 0; j < clause->operation.call_op.num_vars; j++) {
                if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
                result = codegen_ast_callback_(&clause->operation.call_op.variables[j], callback_context_);
            }
            // Capture actual exit block (codegen may have created new blocks)
            then_block = ctx_.builder().GetInsertBlock();
            bool then_terminated = then_block->getTerminator() != nullptr;
            if (!then_terminated) {
                if (!result) {
                    // R7RS §6.3: a `(<test>)` clause with no body returns the
                    // value of the test itself when truthy.  `test` was
                    // computed before the conditional branch and dominates
                    // this then-block, so we can reuse it directly.  Without
                    // this special case, `(cond ((+ 1 2)) (else 'fail))`
                    // would silently return #f instead of 3 (Bug CC).
                    result = test;
                }
                // Ensure result is tagged_value_type for PHI consistency
                if (result->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
                    result = detect_and_pack_callback_(result, callback_context_);
                }
                phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
                ctx_.builder().CreateBr(done_block);
            }

            // Continue to next clause
            ctx_.builder().SetInsertPoint(next_block);
        }
    }

    // If we fell through all clauses without matching, branch to done with #f
    llvm::BasicBlock* fallthrough = ctx_.builder().GetInsertBlock();
    if (!fallthrough->getTerminator()) {
        llvm::Value* false_val = tagged_.packBool(llvm::ConstantInt::getFalse(ctx_.context()));
        phi_inputs.push_back({false_val, ctx_.builder().GetInsertBlock()});
        ctx_.builder().CreateBr(done_block);
    }

    // If done_block has no predecessors (all branches terminated via TCO), remove it
    if (done_block->hasNPredecessorsOrMore(1)) {
        ctx_.builder().SetInsertPoint(done_block);
        if (phi_inputs.size() == 1) {
            return phi_inputs[0].first;
        }
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), phi_inputs.size(), "cond_result");
        for (auto& [val, block] : phi_inputs) {
            phi->addIncoming(val, block);
        }
        return phi;
    } else {
        // All clauses terminated (e.g. all are tail calls) — no merge needed
        done_block->eraseFromParent();
        return llvm::UndefValue::get(ctx_.taggedValueType());
    }
}

/**
 * @brief Generate code for `(if condition then-expr else-expr)`.
 *
 * Evaluates the condition and converts it with isTruthy(), then emits
 * `then`/`else`/`ifcont` basic blocks joined by a conditional branch. Each
 * branch's value is coerced to a common result type: tagged_value_type is
 * used whenever either branch already produces a tagged value or the two
 * branches disagree on raw LLVM type (preserving type information), otherwise
 * the shared raw type is kept as-is. Handles tail-call optimization: if a
 * branch already terminated its block (e.g. via a tail call), it is excluded
 * from the final merge/PHI — if both branches terminated, the merge block is
 * discarded entirely and an undef value is returned (its result is never
 * observed), and if only one branch terminated the surviving branch's value
 * is returned directly without a PHI.
 *
 * @param op Call operation AST node; must have exactly 3 arguments (condition,
 *           then-expr, else-expr).
 * @return Tagged or raw value result of whichever branch was taken.
 */
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

    // Convert condition to boolean using isTruthy for consistent Scheme semantics
    llvm::Value* cond_bool = isTruthy(condition);

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

/**
 * @brief Generate code for `(when test expr...)`.
 *
 * Evaluates the test and branches directly to a `done` block if falsy
 * (skipping the body and carrying a precomputed #f value on that edge), or
 * into a `then` block if truthy. The `then` block evaluates the body
 * expressions in order, stopping early if any expression already terminates
 * the block (e.g. a tail call), and defaults to #t if the body is empty. The
 * two paths are joined at `done_block` with a PHI node; if the `then` block
 * never reaches `done_block` (its body diverged via a tail call), the PHI is
 * skipped and the precomputed #f value is returned directly, since that is
 * the only value still reaching this program point.
 *
 * @param op Call operation AST node; variables[0] is the test, remaining
 *           variables are the body expressions.
 * @return Tagged value: result of the last body expression if the test was
 *         truthy (or #t if the body is empty), otherwise #f.
 */
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
        if (ctx_.builder().GetInsertBlock()->getTerminator()) {
            break;
        }
        if (tv_ptr) {
            result = typed_to_tagged_callback_(tv_ptr, callback_context_);
        }
        if (ctx_.builder().GetInsertBlock()->getTerminator()) {
            break;
        }
    }
    llvm::BasicBlock* then_exit = ctx_.builder().GetInsertBlock();
    bool then_branches_to_done = !then_exit->getTerminator();
    if (!result && then_branches_to_done) {
        result = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    }

    // Only add branch if block doesn't already have a terminator (e.g., from TCO tail call)
    if (then_branches_to_done) {
        ctx_.builder().CreateBr(done_block);
    }

    // Done block - PHI for result
    ctx_.builder().SetInsertPoint(done_block);

    // If then block doesn't branch to done (TCO case), we only have one incoming edge
    if (then_branches_to_done) {
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "when_result");
        phi->addIncoming(result, then_exit);
        phi->addIncoming(false_result, branch_block);
        return phi;
    } else {
        // TCO case - condition was false, just return false_result
        // The then block jumped elsewhere, so only the false path reaches here
        return false_result;
    }
}

/**
 * @brief Generate code for `(unless test expr...)`.
 *
 * Mirrors codegenWhen() with the branches swapped: evaluates the test and
 * branches directly to `done_block` if truthy (carrying a precomputed #f
 * value on that edge), or into an `else` block if falsy, which evaluates the
 * body expressions in order (defaulting to #t if the body is empty) and
 * joins with the #f path via a PHI node at `done_block`. If the body's
 * evaluation already terminated the block (e.g. a tail call), the PHI is
 * skipped and the precomputed #f value is returned directly.
 *
 * @param op Call operation AST node; variables[0] is the test, remaining
 *           variables are the body expressions.
 * @return Tagged value: result of the last body expression if the test was
 *         falsy (or #t if the body is empty), otherwise #f.
 */
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
        if (ctx_.builder().GetInsertBlock()->getTerminator()) {
            break;
        }
        if (tv_ptr) {
            result = typed_to_tagged_callback_(tv_ptr, callback_context_);
        }
        if (ctx_.builder().GetInsertBlock()->getTerminator()) {
            break;
        }
    }
    llvm::BasicBlock* else_exit = ctx_.builder().GetInsertBlock();
    bool else_branches_to_done = !else_exit->getTerminator();
    if (!result && else_branches_to_done) {
        result = tagged_.packBool(llvm::ConstantInt::getTrue(ctx_.context()));
    }

    // Only add branch if block doesn't already have a terminator (e.g., from TCO tail call)
    if (else_branches_to_done) {
        ctx_.builder().CreateBr(done_block);
    }

    // Done block - PHI for result
    ctx_.builder().SetInsertPoint(done_block);

    // If else block doesn't branch to done (TCO case), we only have one incoming edge
    if (else_branches_to_done) {
        llvm::PHINode* phi = ctx_.builder().CreatePHI(ctx_.taggedValueType(), 2, "unless_result");
        phi->addIncoming(result, else_exit);
        phi->addIncoming(false_result, branch_block);
        return phi;
    } else {
        // TCO case - condition was true, just return false_result
        // The else block jumped elsewhere, so only the true path reaches here
        return false_result;
    }
}

// ============================================================================
// Case Expression
// ============================================================================

/**
 * @brief Generate code for `(case key ((datum...) expr...) ... [else expr...])`.
 *
 * Evaluates the key expression once, then for each clause builds a `then`/
 * `next` block pair: all datums in the clause are compared against the key
 * with the eqv?-comparison callback and OR'd together into a single match
 * condition, which conditionally branches into the clause body or continues
 * the scan. Supports the R7RS §5.5 `(datums => receiver)` arrow form by
 * applying the receiver closure to the key value. An `else` clause (if
 * present) always terminates the scan. Each clause whose body reaches the
 * end of its block (i.e. did not already terminate via a TCO tail call or
 * noreturn expression) contributes a PHI input to the shared `done_block`;
 * if no clause matched, a default `#f` input is added.
 *
 * @param op Call operation AST node; op->call_op.func is the key expression,
 *           op->call_op.variables holds the clauses (each a CONS of
 *           datums-AST . body-AST, or an `else` marker).
 * @return Tagged value result of the matched clause's body (or #f if none matched).
 */
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
                    // TCO / noreturn SAFETY (ESH-0211): a body expression in
                    // tail position (e.g. a named-let self-call) may compile
                    // to an unconditional branch back to a loop header,
                    // terminating this block mid-clause. Stop emitting
                    // further sibling expressions once that happens — they
                    // would land after the terminator and trip "Terminator
                    // found in the middle of a basic block!" (mirrors the
                    // same guard in codegenBegin).
                    if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
                    void* tv_ptr = codegen_typed_ast_callback_(&body_ast->operation.call_op.variables[j], callback_context_);
                    if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
                }
            }
            // If the clause body already terminated the block (TCO tail
            // jump / noreturn), there is no fallthrough edge into
            // done_block — do not add a phi input or branch for it.
            if (!ctx_.builder().GetInsertBlock()->getTerminator()) {
                if (result) {
                    phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
                }
                ctx_.builder().CreateBr(done_block);
            }
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
            // R7RS §5.5 `=>` receiver form in case: (datums => receiver).
            bool is_arrow = (body_ast && body_ast->type == ESHKOL_OP &&
                body_ast->operation.op == ESHKOL_CALL_OP &&
                body_ast->operation.call_op.num_vars == 2 &&
                body_ast->operation.call_op.variables[0].type == ESHKOL_VAR &&
                body_ast->operation.call_op.variables[0].variable.id &&
                strcmp(body_ast->operation.call_op.variables[0].variable.id, "=>") == 0);
            if (is_arrow && codegen_ast_callback_ && closure_call_callback_) {
                llvm::Value* receiver = codegen_ast_callback_(
                    &body_ast->operation.call_op.variables[1], callback_context_);
                if (receiver && receiver->getType() != ctx_.taggedValueType() && detect_and_pack_callback_) {
                    receiver = detect_and_pack_callback_(receiver, callback_context_);
                }
                if (receiver) {
                    std::vector<llvm::Value*> args{key};
                    result = closure_call_callback_(receiver, args, callback_context_);
                }
            } else if (body_ast && body_ast->type == ESHKOL_OP &&
                body_ast->operation.op == ESHKOL_CALL_OP) {
                for (uint64_t j = 0; j < body_ast->operation.call_op.num_vars; j++) {
                    // TCO / noreturn SAFETY (ESH-0211): see matching comment
                    // in the else-clause branch above.
                    if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
                    void* tv_ptr = codegen_typed_ast_callback_(&body_ast->operation.call_op.variables[j], callback_context_);
                    if (tv_ptr) result = typed_to_tagged_callback_(tv_ptr, callback_context_);
                }
            }
            // See matching comment in the else-clause branch above: skip the
            // phi input and branch entirely if the clause body already
            // terminated the block.
            if (!ctx_.builder().GetInsertBlock()->getTerminator()) {
                if (result) {
                    phi_inputs.push_back({result, ctx_.builder().GetInsertBlock()});
                }
                ctx_.builder().CreateBr(done_block);
            }

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

/**
 * @brief Generate code for `(begin expr1 expr2 ... exprN)`.
 *
 * Splits the body into internal `define`s and non-define expressions (an
 * Eshkol extension over strict R5RS/R7RS `begin`, which allows defines
 * interleaved with other statements). If any defines are present, all of
 * them are codegen'd first in letrec-like fashion (every binding visible to
 * every other), then the non-define expressions are evaluated in order;
 * otherwise all expressions are simply evaluated in sequence. Evaluation
 * stops early if an expression already terminates the current block (e.g.
 * `raise`), to avoid emitting unreachable instructions after a terminator.
 *
 * @param op Call operation AST node holding the body expressions.
 * @return Value of the last expression evaluated (a default zero constant if
 *         the body is empty or produced no value).
 */
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
            // NORETURN SAFETY (#244): if this expression terminated the
            // current block (e.g., `(raise ...)` emitted unreachable),
            // stop emitting further siblings — they would land after the
            // terminator and trip "Terminator found in the middle of a
            // basic block!" verifier failure.
            if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
        }

        return last_value ? last_value : tagged_.packNull();
    }

    // No internal defines - just execute expressions in sequence
    llvm::Value* last_value = nullptr;
    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
        last_value = codegen_ast_callback_(&op->call_op.variables[i], callback_context_);
        // NORETURN SAFETY (#244): see comment in the defines branch above.
        if (ctx_.builder().GetInsertBlock()->getTerminator()) break;
    }

    return last_value ? last_value : llvm::ConstantInt::get(ctx_.int64Type(), 0);
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
