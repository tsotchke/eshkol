/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MapCodegen - Higher-order list mapping code generation
 *
 * This module handles:
 * - map (apply function to each element of list(s))
 * - Support for direct functions, closures, and builtins
 * - Single-list and multi-list map operations
 * - Closure capture handling for mapped functions
 */
#ifndef ESHKOL_BACKEND_MAP_CODEGEN_H
#define ESHKOL_BACKEND_MAP_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>

namespace eshkol {

/**
 * MapCodegen handles higher-order list mapping operations.
 *
 * Key responsibilities:
 * - Resolve procedure argument (lambda, closure, builtin)
 * - Apply procedure to each element of list(s)
 * - Build result list in arena memory
 * - Handle closure captures during mapping
 */
class MapCodegen {
public:
    /**
     * Construct MapCodegen with context and helpers.
     */
    MapCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === Main Entry Point ===

    /**
     * Generate code for (map proc list1 ...) operation.
     *
     * @param op The map operation AST node
     * @return Result list as tagged value
     */
    llvm::Value* map(const eshkol_operations_t* op);

    // === Configuration ===

    /**
     * Set symbol table references for function/variable lookup.
     */
    void setSymbolTables(
        std::unordered_map<std::string, llvm::Value*>* symbol_table,
        std::unordered_map<std::string, llvm::Value*>* global_symbol_table
    ) {
        symbol_table_ = symbol_table;
        global_symbol_table_ = global_symbol_table;
    }

    /**
     * Set function table for direct function lookup.
     */
    void setFunctionTable(
        std::unordered_map<std::string, llvm::Function*>* function_table
    ) {
        function_table_ = function_table;
    }

    /**
     * Set nested function captures info for closure handling.
     */
    void setNestedFunctionCaptures(
        std::unordered_map<std::string, std::vector<std::string>>* captures
    ) {
        nested_function_captures_ = captures;
    }

    /**
     * Set last generated lambda name reference.
     */
    void setLastGeneratedLambdaName(std::string* name) {
        last_generated_lambda_name_ = name;
    }

    /**
     * Set current function reference.
     */
    void setCurrentFunction(llvm::Function** current) {
        current_function_ = current;
    }

    // === Callbacks ===

    /**
     * Callback for general AST code generation.
     */
    using CodegenASTCallback = llvm::Value* (*)(const eshkol_ast_t*, void*);
    void setCodegenASTCallback(CodegenASTCallback callback, void* context) {
        codegen_ast_callback_ = callback;
        callback_context_ = context;
    }

    /**
     * Callback for lambda code generation.
     */
    using CodegenLambdaCallback = llvm::Value* (*)(const eshkol_operations_t*, void*);
    void setCodegenLambdaCallback(CodegenLambdaCallback callback) {
        codegen_lambda_callback_ = callback;
    }

    /**
     * Callback for closure call.
     */
    using ClosureCallCallback = llvm::Value* (*)(llvm::Value*, const std::vector<llvm::Value*>&, void*);
    void setClosureCallCallback(ClosureCallCallback callback) {
        closure_call_callback_ = callback;
    }

    /**
     * Callback for extracting car as tagged value.
     */
    using ExtractCarCallback = llvm::Value* (*)(llvm::Value*, void*);
    void setExtractCarCallback(ExtractCarCallback callback) {
        extract_car_callback_ = callback;
    }

    /**
     * Callback for creating cons cell from tagged values.
     */
    using CreateConsCallback = llvm::Value* (*)(llvm::Value*, llvm::Value*, void*);
    void setCreateConsCallback(CreateConsCallback callback) {
        create_cons_callback_ = callback;
    }

    /**
     * Callback for getting cons cell accessor function.
     */
    using GetConsAccessorCallback = llvm::Function* (*)(void*);
    void setGetConsGetPtrCallback(GetConsAccessorCallback callback) {
        get_cons_get_ptr_callback_ = callback;
    }
    void setGetConsSetPtrCallback(GetConsAccessorCallback callback) {
        get_cons_set_ptr_callback_ = callback;
    }

    /**
     * Callback for resolving lambda function from AST.
     */
    using ResolveLambdaCallback = llvm::Value* (*)(const eshkol_ast_t*, size_t, void*);
    void setResolveLambdaCallback(ResolveLambdaCallback callback) {
        resolve_lambda_callback_ = callback;
    }

    /**
     * Callback for indirect function call handling.
     */
    using IndirectCallCallback = llvm::Value* (*)(llvm::Value*, size_t, void*);
    void setIndirectCallCallback(IndirectCallCallback callback) {
        indirect_call_callback_ = callback;
    }

    /**
     * Callback for function context push/pop (isolation).
     */
    using FunctionContextCallback = void (*)(void*);
    void setFunctionContextCallbacks(FunctionContextCallback push, FunctionContextCallback pop) {
        push_function_context_ = push;
        pop_function_context_ = pop;
    }

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;

    // Symbol tables
    std::unordered_map<std::string, llvm::Value*>* symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Value*>* global_symbol_table_ = nullptr;
    std::unordered_map<std::string, llvm::Function*>* function_table_ = nullptr;
    std::unordered_map<std::string, std::vector<std::string>>* nested_function_captures_ = nullptr;

    // State references
    std::string* last_generated_lambda_name_ = nullptr;
    llvm::Function** current_function_ = nullptr;

    // Callbacks
    CodegenASTCallback codegen_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;
    CodegenLambdaCallback codegen_lambda_callback_ = nullptr;
    ClosureCallCallback closure_call_callback_ = nullptr;
    ExtractCarCallback extract_car_callback_ = nullptr;
    CreateConsCallback create_cons_callback_ = nullptr;
    GetConsAccessorCallback get_cons_get_ptr_callback_ = nullptr;
    GetConsAccessorCallback get_cons_set_ptr_callback_ = nullptr;
    ResolveLambdaCallback resolve_lambda_callback_ = nullptr;
    IndirectCallCallback indirect_call_callback_ = nullptr;
    FunctionContextCallback push_function_context_ = nullptr;
    FunctionContextCallback pop_function_context_ = nullptr;

    // === Internal Implementation ===

    /**
     * Map with runtime closure dispatch.
     * Used when procedure is a closure with captures.
     */
    llvm::Value* mapWithClosure(llvm::Value* closure_val, llvm::Value* list);

    /**
     * Map over a single list with known function.
     */
    llvm::Value* mapSingleList(llvm::Function* proc_func, llvm::Value* list);

    /**
     * Map over multiple lists with known function.
     */
    llvm::Value* mapMultiList(llvm::Function* proc_func, const std::vector<llvm::Value*>& lists);

    /**
     * Load captured values for a closure call.
     */
    void loadCapturedValues(
        llvm::Function* proc_func,
        const std::string& func_name,
        size_t first_capture_idx,
        std::vector<llvm::Value*>& args
    );

    /**
     * Get cons cell get pointer function.
     */
    llvm::Function* getConsGetPtrFunc();

    /**
     * Get cons cell set pointer function.
     */
    llvm::Function* getConsSetPtrFunc();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_MAP_CODEGEN_H
