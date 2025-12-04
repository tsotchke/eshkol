/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ControlFlowCodegen - Control flow code generation
 *
 * This module handles control flow operations on tagged values:
 * - Boolean logic (and, or)
 * - Conditional expressions (if, cond)
 * - Sequencing (begin)
 * - Truthiness testing
 */
#ifndef ESHKOL_BACKEND_CONTROL_FLOW_CODEGEN_H
#define ESHKOL_BACKEND_CONTROL_FLOW_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>

namespace eshkol {

/**
 * ControlFlowCodegen handles control flow operations.
 *
 * Control flow operations often need to evaluate AST nodes, so this class
 * uses callbacks to invoke the main codegen's AST processing methods.
 */
class ControlFlowCodegen {
public:
    /**
     * Construct ControlFlowCodegen with context and tagged value helper.
     */
    ControlFlowCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Truthiness Testing ===

    /**
     * Check if a value is "truthy" (non-false, non-null, non-zero).
     * Handles raw values and tagged values.
     * @param val Value to test
     * @return i1 boolean value (true if truthy)
     */
    llvm::Value* isTruthy(llvm::Value* val);

    // === Boolean Logic (Short-circuit) ===

    /**
     * Short-circuit AND: (and a b ...)
     * Returns the last truthy value or the first falsy value.
     * @param op The AND operation AST node
     * @return Result as tagged_value
     */
    llvm::Value* codegenAnd(const eshkol_operations_t* op);

    /**
     * Short-circuit OR: (or a b ...)
     * Returns the first truthy value or the last falsy value.
     * @param op The OR operation AST node
     * @return Result as tagged_value
     */
    llvm::Value* codegenOr(const eshkol_operations_t* op);

    // === Conditional Expressions ===

    /**
     * Conditional expression: (cond (test1 expr1) (test2 expr2) ... (else exprN))
     * @param op The COND operation AST node
     * @return Result as tagged_value
     */
    llvm::Value* codegenCond(const eshkol_operations_t* op);

    /**
     * If expression: (if condition then-expr else-expr)
     * @param op The IF call operation AST node
     * @return Result value
     */
    llvm::Value* codegenIf(const eshkol_operations_t* op);

    // === Sequencing ===

    /**
     * Begin sequence: (begin expr1 expr2 ... exprN)
     * Executes all expressions and returns the value of the last one.
     * Handles internal defines by treating them as letrec.
     * @param op The BEGIN call operation AST node
     * @return Result value
     */
    llvm::Value* codegenBegin(const eshkol_operations_t* op);

    // === Logical Operations ===

    /**
     * Logical NOT: (not x)
     * Returns #t if x is falsy, #f otherwise.
     * @param op The NOT call operation AST node
     * @return Result as tagged_value (boolean)
     */
    llvm::Value* codegenNot(const eshkol_operations_t* op);

    // === Additional Control Flow ===

    /**
     * When conditional: (when test expr...)
     * Evaluates expressions if test is true, returns last result or void.
     * @param op The WHEN call operation AST node
     * @return Result value
     */
    llvm::Value* codegenWhen(const eshkol_operations_t* op);

    /**
     * Unless conditional: (unless test expr...)
     * Evaluates expressions if test is false, returns last result or void.
     * @param op The UNLESS call operation AST node
     * @return Result value
     */
    llvm::Value* codegenUnless(const eshkol_operations_t* op);

    /**
     * Case expression: (case key ((datum1 datum2 ...) expr1 ...) ... (else exprN))
     * Matches key against datums using eqv? semantics.
     * @param op The CASE call operation AST node
     * @return Result value
     */
    llvm::Value* codegenCase(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Callback types for AST code generation
    // These are set by the main codegen to avoid circular dependencies
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);  // Returns TypedValue*
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);
    using CodegenDefineFunc = void (*)(const void* op, void* context);
    using EqvCompareFunc = llvm::Value* (*)(llvm::Value* a, llvm::Value* b, void* context);
    using DetectAndPackFunc = llvm::Value* (*)(llvm::Value* val, void* context);  // detectValueType + pack

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    CodegenDefineFunc codegen_func_define_callback_ = nullptr;
    CodegenDefineFunc codegen_var_define_callback_ = nullptr;
    EqvCompareFunc eqv_compare_callback_ = nullptr;
    DetectAndPackFunc detect_and_pack_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callbacks for AST code generation.
     * Called by EshkolLLVMCodeGen to inject dependencies.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        CodegenDefineFunc codegen_func_define,
        CodegenDefineFunc codegen_var_define,
        EqvCompareFunc eqv_compare,
        DetectAndPackFunc detect_and_pack,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        codegen_func_define_callback_ = codegen_func_define;
        codegen_var_define_callback_ = codegen_var_define;
        eqv_compare_callback_ = eqv_compare;
        detect_and_pack_callback_ = detect_and_pack;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_CONTROL_FLOW_CODEGEN_H
