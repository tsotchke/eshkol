/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * BindingCodegen - Variable binding code generation
 *
 * This module handles:
 * - define (top-level and nested variable definitions)
 * - let (local bindings with sequential evaluation)
 * - letrec (recursive bindings for mutual recursion)
 * - set! (mutation of existing bindings)
 *
 * Key design principle: ALL values are stored as tagged_value structs
 * to preserve type information across storage and retrieval.
 */
#ifndef ESHKOL_BACKEND_BINDING_CODEGEN_H
#define ESHKOL_BACKEND_BINDING_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/GlobalVariable.h>
#include <string>
#include <set>
#include <unordered_map>

namespace eshkol {

// Forward declaration for TypedValue
struct TypedValue;

/**
 * BindingCodegen handles variable binding operations.
 *
 * All bindings store values as tagged_value to preserve type information.
 * This ensures that ports, lists, lambdas, etc. can be correctly retrieved.
 */
class BindingCodegen {
public:
    /**
     * Construct BindingCodegen with context and tagged value helper.
     */
    BindingCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Variable Definition ===

    /**
     * Define a variable: (define name value)
     *
     * For top-level: creates GlobalVariable with tagged_value
     * For nested (inside function): creates AllocaInst with tagged_value
     *
     * @param op The define operation AST node
     * @return The stored value (or nullptr on failure)
     */
    llvm::Value* define(const eshkol_operations_t* op);

    // === Let Bindings ===

    /**
     * Let expression: (let ((x 1) (y 2)) body)
     *
     * Sequential evaluation - each binding evaluated in order,
     * but bindings are not visible to each other.
     *
     * @param op The let operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* let(const eshkol_operations_t* op);

    /**
     * Letrec expression: (letrec ((f ...) (g ...)) body)
     *
     * Recursive bindings - all bindings visible to all values.
     * Used for mutually recursive function definitions.
     *
     * @param op The letrec operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* letrec(const eshkol_operations_t* op);

    /**
     * Let* expression: (let* ((x 1) (y x)) body)
     *
     * Sequential evaluation - each binding visible to subsequent ones.
     *
     * @param op The let* operation AST node
     * @return Result of body evaluation
     */
    llvm::Value* letStar(const eshkol_operations_t* op);

    // === Mutation ===

    /**
     * Set! expression: (set! name value)
     *
     * Mutates an existing binding.
     *
     * @param op The set! operation AST node
     * @return The new value
     */
    llvm::Value* set(const eshkol_operations_t* op);

    // === Variable Lookup ===

    /**
     * Look up a variable by name.
     *
     * Searches local symbol table first, then global.
     *
     * @param name Variable name
     * @return The variable's storage location (alloca or global), or nullptr
     */
    llvm::Value* lookupVariable(const std::string& name);

    /**
     * Load a variable's value.
     *
     * @param name Variable name
     * @return The variable's tagged value, or nullptr
     */
    llvm::Value* loadVariable(const std::string& name);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Callbacks for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);
    // Get TypedValue type (returns eshkol_value_type_t)
    using GetTypedValueTypeFunc = int (*)(void* typed_value, void* context);
    // Register function binding (for apply/call resolution)
    using RegisterFuncBindingFunc = void (*)(const char* var_name, void* typed_value, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    GetTypedValueTypeFunc get_typed_value_type_callback_ = nullptr;
    RegisterFuncBindingFunc register_func_binding_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // Symbol tables (references to main codegen's tables)
    std::unordered_map<std::string, llvm::Value*>* symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Value*>* global_symbol_table_ = nullptr;

    // Current function context
    llvm::Function** current_function_ = nullptr;

    // REPL mode flag
    bool* repl_mode_ = nullptr;

    // Lambda tracking for function bindings
    std::string* last_generated_lambda_name_ = nullptr;
    std::unordered_map<std::string, llvm::Function*>* function_table_ = nullptr;

    // LETREC REFACTOR: Set of names to exclude from free variable capture
    // Populated before generating letrec lambda bindings, cleared after
    std::set<std::string>* letrec_excluded_capture_names_ = nullptr;

    /**
     * Store a value in a binding (alloca or global).
     * Always converts to tagged_value first.
     *
     * @param name Variable name
     * @param value The value to store
     * @param value_type The Eshkol type of the value
     * @param is_global Whether to create a global variable
     * @return The storage location
     */
    llvm::Value* storeBinding(
        const std::string& name,
        llvm::Value* value,
        eshkol_value_type_t value_type,
        bool is_global
    );

    /**
     * Ensure value is a tagged_value.
     * Converts raw i64, double, etc. to tagged_value struct.
     *
     * @param value The value to convert
     * @param value_type The Eshkol type
     * @return A tagged_value
     */
    llvm::Value* ensureTaggedValue(llvm::Value* value, eshkol_value_type_t value_type);

    /**
     * Register a lambda function binding.
     * Sets up _func and _sexpr entries in symbol tables.
     *
     * @param var_name Variable name
     * @param lambda_name Lambda function name
     */
    void registerLambdaBinding(const std::string& var_name, const std::string& lambda_name);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        GetTypedValueTypeFunc get_typed_value_type,
        RegisterFuncBindingFunc register_func_binding,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        get_typed_value_type_callback_ = get_typed_value_type;
        register_func_binding_callback_ = register_func_binding;
        callback_context_ = context;
    }

    /**
     * Set symbol table references.
     */
    void setSymbolTables(
        std::unordered_map<std::string, llvm::Value*>* local,
        std::unordered_map<std::string, llvm::Value*>* global
    ) {
        symbol_table_ = local;
        global_symbol_table_ = global;
    }

    /**
     * Set current function pointer reference.
     */
    void setCurrentFunction(llvm::Function** func) {
        current_function_ = func;
    }

    /**
     * Set REPL mode flag reference.
     */
    void setReplMode(bool* repl_mode) {
        repl_mode_ = repl_mode;
    }

    /**
     * Set lambda tracking references.
     */
    void setLambdaTracking(
        std::string* last_lambda_name,
        std::unordered_map<std::string, llvm::Function*>* func_table
    ) {
        last_generated_lambda_name_ = last_lambda_name;
        function_table_ = func_table;
    }

    /**
     * Set letrec excluded capture names reference.
     * Used to prevent letrec-bound lambdas from capturing themselves.
     */
    void setLetrecExcludedCaptureNames(std::set<std::string>* names) {
        letrec_excluded_capture_names_ = names;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_BINDING_CODEGEN_H
