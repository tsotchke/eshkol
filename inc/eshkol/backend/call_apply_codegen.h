/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CallApplyCodegen - Function call and apply code generation
 *
 * This module handles:
 * - Function calls (direct, closure, and indirect)
 * - apply (Scheme's apply - apply function to argument list)
 * - Closure dispatch with dynamic argument counts
 *
 * Performance optimizations:
 * - Direct dispatch for known functions
 * - Specialized paths for common arg/capture counts
 * - Efficient argument extraction from lists
 */
#ifndef ESHKOL_BACKEND_CALL_APPLY_CODEGEN_H
#define ESHKOL_BACKEND_CALL_APPLY_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/arithmetic_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace eshkol {

// Forward declarations
struct TypedValue;

/**
 * CallApplyCodegen handles function invocation operations.
 *
 * Key responsibilities:
 * - Dispatch function calls to appropriate handlers
 * - Handle closure calls with captured variables
 * - Implement Scheme's apply for runtime function application
 */
class CallApplyCodegen {
public:
    // Performance tuning constants
    static constexpr int MAX_APPLY_ARGS = 8;
    static constexpr int MAX_APPLY_CAPTURES = 8;

    /**
     * Construct CallApplyCodegen with context and helpers.
     */
    CallApplyCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, ArithmeticCodegen& arith);

    // === Apply Operations (Scheme apply) ===

    /**
     * Apply function to argument list: (apply func args)
     *
     * Handles:
     * - Built-in arithmetic (+, -, *, /)
     * - Built-in operations (cons, list)
     * - User-defined functions
     * - Closures with captures
     *
     * @param op The apply operation AST node
     * @return Result of function application
     */
    llvm::Value* apply(const eshkol_operations_t* op);

    /**
     * Apply cons to a 2-element list: (apply cons '(a b))
     *
     * @param list_int List as i64 pointer
     * @return Cons cell (a . b) as tagged value
     */
    llvm::Value* applyCons(llvm::Value* list_int);

    /**
     * Apply variadic arithmetic to list: (apply + '(1 2 3))
     *
     * @param op Operation string (+, -, *, /)
     * @param list_int Argument list as i64 pointer
     * @return Result of folding operation over list
     */
    llvm::Value* applyArithmetic(const std::string& op, llvm::Value* list_int);

    /**
     * Apply user-defined function to list arguments.
     *
     * Handles both fixed-arity and variadic functions.
     *
     * @param func The LLVM function to call
     * @param list_int Argument list as i64 pointer
     * @return Function result as tagged value
     */
    llvm::Value* applyUserFunction(llvm::Function* func, llvm::Value* list_int);

    /**
     * Apply closure to list arguments with capture support.
     *
     * This is the most general apply path - handles:
     * - Variable argument counts (0-8)
     * - Variable capture counts (0-8)
     * - Runtime dispatch based on actual counts
     *
     * @param func_value Closure as tagged value
     * @param list_int Argument list as i64 pointer
     * @return Function result as tagged value
     */
    llvm::Value* applyClosure(llvm::Value* func_value, llvm::Value* list_int);

    // === Closure Calls ===

    /**
     * Call a closure with pre-evaluated arguments.
     *
     * Used for direct closure invocation, not via apply.
     * Extracts captures from closure environment and appends to args.
     *
     * @param closure Closure as tagged value
     * @param args Pre-evaluated arguments
     * @return Function result as tagged value
     */
    llvm::Value* closureCall(llvm::Value* closure, const std::vector<llvm::Value*>& args);

    // === Configuration ===

    /**
     * Set symbol table references for function lookup.
     */
    void setSymbolTables(
        std::unordered_map<std::string, llvm::Value*>* symbol_table,
        std::unordered_map<std::string, llvm::Value*>* global_symbol_table
    ) {
        symbol_table_ = symbol_table;
        global_symbol_table_ = global_symbol_table;
    }

    /**
     * Set variadic function info for proper argument handling.
     */
    void setVariadicFunctionInfo(
        std::unordered_map<std::string, std::pair<uint64_t, bool>>* info
    ) {
        variadic_function_info_ = info;
    }

    /**
     * Set callback for general AST code generation.
     * Used when apply needs to evaluate function argument.
     */
    using CodegenASTCallback = llvm::Value* (*)(const eshkol_ast_t*, void*);
    void setCodegenASTCallback(CodegenASTCallback callback, void* context) {
        codegen_ast_callback_ = callback;
        callback_context_ = context;
    }

    // === Cons Cell Operation Callbacks ===
    // These are needed for list manipulation in apply

    /**
     * Callback type for extracting car as tagged value.
     */
    using ExtractConsCarCallback = llvm::Value* (*)(llvm::Value* cons_ptr, void* context);
    void setExtractConsCarCallback(ExtractConsCarCallback callback) {
        extract_cons_car_callback_ = callback;
    }

    /**
     * Callback type for getting cons cell accessor function.
     */
    using GetConsAccessorCallback = llvm::Function* (*)(void* context);
    void setGetConsAccessorCallback(GetConsAccessorCallback callback) {
        get_cons_accessor_callback_ = callback;
    }

    /**
     * Callback type for creating tagged cons cell.
     */
    using CreateConsCallback = llvm::Value* (*)(llvm::Value* car, llvm::Value* cdr, void* context);
    void setCreateConsCallback(CreateConsCallback callback) {
        create_cons_callback_ = callback;
    }

    /**
     * Callback type for getting builtin arithmetic function.
     * Returns the Function* for a given operation (+, -, *, /).
     */
    using GetBuiltinArithmeticCallback = llvm::Function* (*)(const std::string& op, void* context);
    void setGetBuiltinArithmeticCallback(GetBuiltinArithmeticCallback callback) {
        get_builtin_arithmetic_callback_ = callback;
    }

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    ArithmeticCodegen& arith_;

    // Symbol tables for function lookup
    std::unordered_map<std::string, llvm::Value*>* symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Value*>* global_symbol_table_ = nullptr;

    // Variadic function metadata
    std::unordered_map<std::string, std::pair<uint64_t, bool>>* variadic_function_info_ = nullptr;

    // Callback for AST codegen
    CodegenASTCallback codegen_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // Cons cell operation callbacks
    ExtractConsCarCallback extract_cons_car_callback_ = nullptr;
    GetConsAccessorCallback get_cons_accessor_callback_ = nullptr;
    CreateConsCallback create_cons_callback_ = nullptr;

    // Builtin arithmetic callback
    GetBuiltinArithmeticCallback get_builtin_arithmetic_callback_ = nullptr;

    // === Internal Helpers ===

    /**
     * Extract car of cons cell as tagged value using callback.
     */
    llvm::Value* extractConsCarAsTaggedValue(llvm::Value* cons_ptr);

    /**
     * Get runtime function for cons cell access using callback.
     */
    llvm::Function* getTaggedConsGetPtrFunc();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_CALL_APPLY_CODEGEN_H
