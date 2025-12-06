/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * TailCallCodegen - Tail call optimization support
 *
 * This module provides:
 * - Tail position detection
 * - Tail call marking for LLVM
 * - Trampoline-based TCO for closures
 * - Stack-safe deep recursion
 */
#ifndef ESHKOL_BACKEND_TAIL_CALL_CODEGEN_H
#define ESHKOL_BACKEND_TAIL_CALL_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instructions.h>
#include <vector>
#include <string>

namespace eshkol {

/**
 * TailCallCodegen handles tail call optimization.
 *
 * LLVM TCO Requirements:
 * 1. Call must be immediately followed by a return
 * 2. Caller and callee must have compatible calling conventions
 * 3. The return value must be the result of the call (or void)
 * 4. No instructions between call and return (except bitcasts)
 *
 * For closures with captured environments, we use a trampoline approach:
 * - Functions return either a result or a "bounce" (thunk to continue)
 * - The trampoline loop evaluates bounces until a result is reached
 */
class TailCallCodegen {
public:
    /**
     * Construct TailCallCodegen with context and helpers.
     */
    TailCallCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // === Tail Position Detection ===

    /**
     * Check if an AST node is in tail position within its enclosing function.
     * @param ast The AST node to check
     * @param parent The parent AST node (or nullptr for top-level)
     * @return true if the node is in tail position
     */
    bool isTailPosition(const eshkol_ast_t* ast, const eshkol_ast_t* parent) const;

    /**
     * Check if an operation is in tail position.
     * @param op The operation node
     * @param parent The parent operation (or nullptr)
     * @return true if in tail position
     */
    bool isOperationInTailPosition(const eshkol_operations_t* op, const eshkol_operations_t* parent) const;

    // === Tail Call Marking ===

    /**
     * Mark a call instruction as a tail call if appropriate.
     * @param call The LLVM call instruction
     * @param in_tail_position Whether this call is in tail position
     * @return The call instruction (possibly modified)
     */
    llvm::CallInst* markTailCall(llvm::CallInst* call, bool in_tail_position);

    /**
     * Create a tail call that will reuse the current stack frame.
     * The call must be immediately followed by a return.
     * @param func The function to call
     * @param args The arguments
     * @return The result of the tail call
     */
    llvm::Value* createTailCall(llvm::Function* func, const std::vector<llvm::Value*>& args);

    /**
     * Create a tail call to a closure.
     * @param closure The closure tagged value
     * @param args The arguments
     * @return The result of the tail call
     */
    llvm::Value* createClosureTailCall(llvm::Value* closure, const std::vector<llvm::Value*>& args);

    // === Trampoline Support ===

    /**
     * Tag value indicating a bounce (continuation thunk).
     * Uses a special tag bit pattern to distinguish from regular values.
     */
    static constexpr uint64_t BOUNCE_TAG = 0x0F00000000000000ULL;

    /**
     * Create a bounce value (a thunk that continues the computation).
     * @param thunk_func The 0-argument function to call
     * @return Tagged bounce value
     */
    llvm::Value* createBounce(llvm::Value* thunk_func);

    /**
     * Check if a value is a bounce (continuation).
     * @param value The tagged value to check
     * @return Boolean indicating if it's a bounce
     */
    llvm::Value* isBounce(llvm::Value* value);

    /**
     * Extract the thunk function from a bounce value.
     * @param bounce The bounce value
     * @return The thunk function pointer
     */
    llvm::Value* extractBounceThunk(llvm::Value* bounce);

    /**
     * Create a trampoline that evaluates a thunk until it returns a non-bounce value.
     * @param initial_thunk The initial thunk to evaluate
     * @return The final result
     */
    llvm::Value* createTrampoline(llvm::Value* initial_thunk);

    /**
     * Generate the trampoline runtime function.
     * This function iteratively calls thunks until a result is reached.
     */
    void generateTrampolineRuntime();

    // === Tail-Recursive Lambda Transformation ===

    /**
     * Check if a lambda body contains only tail-recursive self-calls.
     * Such lambdas can be transformed to use iteration instead of recursion.
     * @param lambda_op The lambda operation
     * @param func_name The name of the function (for self-call detection)
     * @return true if all self-calls are in tail position
     */
    bool isTailRecursive(const eshkol_operations_t* lambda_op, const std::string& func_name) const;

    /**
     * Transform a tail-recursive lambda into an iterative form.
     * This avoids stack growth by using a loop instead of recursion.
     * @param func_name The function name
     * @param params The parameter names
     * @param body The function body
     * @return An LLVM function using iteration
     */
    llvm::Function* transformToIterative(
        const std::string& func_name,
        const std::vector<std::string>& params,
        const eshkol_ast_t* body
    );

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Trampoline runtime function
    llvm::Function* trampoline_func_ = nullptr;

    // Helper to determine if an expression is the last in a sequence
    bool isLastInSequence(const eshkol_ast_t* expr, const eshkol_operations_t* parent) const;

    // Helper to check if a call can use direct TCO (vs trampoline)
    bool canUseDirectTCO(llvm::Function* caller, llvm::Function* callee) const;
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_TAIL_CALL_CODEGEN_H
