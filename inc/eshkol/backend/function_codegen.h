/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * FunctionCodegen - Lambda and closure code generation
 *
 * This module handles:
 * - Lambda expression compilation
 * - Closure creation and capture
 * - Function definitions (top-level and nested)
 * - Function application
 */
#ifndef ESHKOL_BACKEND_FUNCTION_CODEGEN_H
#define ESHKOL_BACKEND_FUNCTION_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <vector>
#include <string>

namespace eshkol {

/**
 * FunctionCodegen handles lambda and closure operations.
 */
class FunctionCodegen {
public:
    /**
     * Construct FunctionCodegen with context and helpers.
     */
    FunctionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // === Lambda Expressions ===

    /**
     * Compile a lambda expression: (lambda (params) body)
     * @param op The LAMBDA operation AST node
     * @return Closure as tagged value (or raw function pointer if no captures)
     */
    llvm::Value* lambda(const eshkol_operations_t* op);

    /**
     * Convert a lambda to its S-expression representation.
     * @param ast The lambda AST node
     * @return i64 pointer to S-expression
     */
    llvm::Value* lambdaToSExpr(const eshkol_ast_t* ast);

    // === Function Definitions ===

    /**
     * Compile a function definition: (define (name params) body)
     * @param op The DEFINE operation AST node
     * @return The defined function value
     */
    llvm::Value* functionDefinition(const eshkol_operations_t* op);

    /**
     * Compile a nested function definition within another function.
     * @param op The DEFINE operation AST node
     * @return The defined function value
     */
    llvm::Value* nestedFunctionDefinition(const eshkol_operations_t* op);

    // === Closure Operations ===

    /**
     * Call a closure with arguments.
     * @param closure The closure value
     * @param args The arguments to pass
     * @return Result of the call
     */
    llvm::Value* closureCall(llvm::Value* closure, const std::vector<llvm::Value*>& args);

    /**
     * Create a closure capturing the given environment.
     * @param func The LLVM function
     * @param captures Values to capture in the closure
     * @return Closure as tagged value
     */
    llvm::Value* createClosure(llvm::Function* func, const std::vector<llvm::Value*>& captures);

    // === Application ===

    /**
     * Apply a function to arguments: (apply func args-list)
     * @param op The APPLY operation AST node
     * @return Result of the application
     */
    llvm::Value* apply(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Callback types for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_FUNCTION_CODEGEN_H
