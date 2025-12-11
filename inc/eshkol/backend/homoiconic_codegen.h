/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HomoiconicCodegen - Code-as-data (quote/S-expression) code generation
 *
 * This module handles Scheme's homoiconic features:
 * - quote: Convert compile-time AST to runtime S-expression
 * - Lambda-to-S-expr: Serialize lambda for display/introspection
 * - Quoted operations: Handle special forms in quoted context
 *
 * Key principle: Code is data. Lambdas can be displayed as their
 * source S-expression, enabling introspection and meta-programming.
 */
#ifndef ESHKOL_BACKEND_HOMOICONIC_CODEGEN_H
#define ESHKOL_BACKEND_HOMOICONIC_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/collection_codegen.h>
#include <eshkol/backend/string_io_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <string>

namespace eshkol {

/**
 * HomoiconicCodegen handles quote and S-expression operations.
 *
 * Converts compile-time AST structures to runtime cons-cell lists,
 * enabling Scheme's code-as-data philosophy.
 */
class HomoiconicCodegen {
public:
    /**
     * Construct HomoiconicCodegen with context and helpers.
     */
    HomoiconicCodegen(CodegenContext& ctx,
                      TaggedValueCodegen& tagged,
                      CollectionCodegen& collection,
                      StringIOCodegen& string_io);

    // === Quote Operations ===

    /**
     * Quote an AST node: (quote expr) or 'expr
     *
     * Converts the AST to a runtime S-expression (cons list).
     * Numbers, strings, bools become themselves.
     * Symbols become strings.
     * Lists become cons chains.
     *
     * @param ast The AST to quote
     * @return Tagged value representing the quoted expression
     */
    llvm::Value* quoteAST(const eshkol_ast_t* ast);

    /**
     * Quote an operation node.
     *
     * Handles special forms (if, and, or, cond, let, define, lambda)
     * by building appropriate S-expression lists.
     *
     * @param op The operation to quote
     * @return Tagged value representing the quoted operation
     */
    llvm::Value* quoteOperation(const eshkol_operations_t* op);

    // === Lambda S-Expression ===

    /**
     * Convert a lambda to its S-expression form.
     *
     * Returns (lambda (param1 param2 ...) body)
     * Used for displaying procedures as their source code.
     *
     * @param op The lambda or define operation
     * @return Cons list pointer (i64) representing the lambda
     */
    llvm::Value* lambdaToSExpr(const eshkol_operations_t* op);

    /**
     * Create an S-expression for a builtin primitive.
     *
     * Returns (primitive name) for display purposes.
     * Used when builtin operators like +, -, *, / are stored in variables.
     *
     * @param name The primitive name (e.g., "+", "-", "*", "/")
     * @return Cons list pointer (i64) representing the primitive
     */
    llvm::Value* builtinToSExpr(const std::string& name);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    CollectionCodegen& collection_;
    StringIOCodegen& string_io_;

    // === Internal Helpers ===

    /**
     * Build an N-ary operation: (op arg1 arg2 ...)
     */
    llvm::Value* quoteNaryOp(const char* op_name,
                              const eshkol_ast_t* args,
                              uint64_t num_args);

    /**
     * Build a list from call operation arguments.
     */
    llvm::Value* quoteList(const eshkol_operations_t* op);

    /**
     * Build a parameter list: (param1 param2 ...)
     */
    llvm::Value* buildParameterList(const eshkol_ast_t* params,
                                     uint64_t num_params);

    /**
     * Pack null value.
     */
    llvm::Value* packNull();

    /**
     * Pack pointer with type.
     */
    llvm::Value* packPtr(llvm::Value* ptr, eshkol_value_type_t type);

    /**
     * Create cons cell from two tagged values.
     */
    llvm::Value* consFromTagged(llvm::Value* car, llvm::Value* cdr);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_HOMOICONIC_CODEGEN_H
