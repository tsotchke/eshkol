/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * BuiltinDeclarations - External runtime function declarations
 *
 * This module declares external C runtime functions that the generated
 * LLVM IR calls into. These include:
 * - Deep equality comparison for nested structures
 * - Unified display system for homoiconicity
 * - Lambda registry for tracking lambda S-expressions
 */
#ifndef ESHKOL_BACKEND_BUILTIN_DECLARATIONS_H
#define ESHKOL_BACKEND_BUILTIN_DECLARATIONS_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <llvm/IR/Function.h>

namespace eshkol {

/**
 * BuiltinDeclarations manages external runtime function declarations.
 *
 * All functions are declared on construction and cached as members.
 * The CodegenContext is also updated with these functions for use by
 * other codegen modules.
 */
class BuiltinDeclarations {
public:
    /**
     * Construct BuiltinDeclarations for the given context.
     * Creates all runtime function declarations.
     */
    explicit BuiltinDeclarations(CodegenContext& ctx);

    // === Deep Equality ===
    /**
     * eshkol_deep_equal: bool(ptr val1, ptr val2)
     * Deep equality comparison for nested tagged values
     */
    llvm::Function* getDeepEqual() const { return deep_equal_func_; }

    // === Display System ===
    /**
     * eshkol_display_value: void(ptr value)
     * Unified display for all tagged value types
     */
    llvm::Function* getDisplayValue() const { return display_value_func_; }

    // === Lambda Registry ===
    /**
     * eshkol_lambda_registry_init: void()
     * Initialize the lambda registry
     */
    llvm::Function* getLambdaRegistryInit() const { return lambda_registry_init_func_; }

    /**
     * eshkol_lambda_registry_add: void(i64 func_ptr, i64 sexpr_ptr, ptr name)
     * Register a lambda's S-expression for homoiconic display
     */
    llvm::Function* getLambdaRegistryAdd() const { return lambda_registry_add_func_; }

    /**
     * eshkol_lambda_registry_lookup: i64(i64 func_ptr)
     * Look up a lambda's S-expression by function pointer
     */
    llvm::Function* getLambdaRegistryLookup() const { return lambda_registry_lookup_func_; }

private:
    CodegenContext& ctx_;

    // Cached function declarations
    llvm::Function* deep_equal_func_;
    llvm::Function* display_value_func_;
    llvm::Function* lambda_registry_init_func_;
    llvm::Function* lambda_registry_add_func_;
    llvm::Function* lambda_registry_lookup_func_;

    // Declaration helpers
    void declareDeepEqual();
    void declareDisplayValue();
    void declareLambdaRegistry();
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_BUILTIN_DECLARATIONS_H
