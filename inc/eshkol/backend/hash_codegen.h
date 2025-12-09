/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * HashCodegen - Hash table code generation for Eshkol
 *
 * This module provides code generation for hash table operations:
 * - make-hash-table: Create a new hash table
 * - hash-set!: Set a key-value pair
 * - hash-ref: Get value by key
 * - hash-has-key?: Check if key exists
 * - hash-remove!: Remove a key
 * - hash-keys: Get all keys as a list
 * - hash-values: Get all values as a list
 * - hash-count: Get number of entries
 * - hash-clear!: Clear all entries
 */
#ifndef ESHKOL_BACKEND_HASH_CODEGEN_H
#define ESHKOL_BACKEND_HASH_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <llvm/IR/Value.h>
#include <llvm/IR/Function.h>
#include <unordered_map>
#include <string>

// Forward declaration
struct eshkol_operation;
typedef struct eshkol_operation eshkol_operations_t;

namespace eshkol {

/**
 * HashCodegen provides LLVM IR code generation for hash table operations.
 *
 * It generates calls to the runtime hash table functions in arena_memory.cpp
 * and handles the conversion between Eshkol tagged values and the hash table API.
 */
class HashCodegen {
public:
    // Callback type for AST code generation
    using CodegenASTCallback = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTCallback = void* (*)(const void* ast, void* context);

    /**
     * Construct a HashCodegen for the given context, tagged value codegen, and memory codegen.
     */
    HashCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem,
                std::unordered_map<std::string, llvm::Function*>& func_table);

    /**
     * Set callback functions for AST code generation.
     * These are used to generate code for hash key/value expressions.
     */
    void setCodegenCallbacks(CodegenASTCallback ast_cb, CodegenTypedASTCallback typed_cb, void* ctx);

    // Hash table creation
    llvm::Value* makeHashTable(const eshkol_operations_t* op);

    // Hash table operations
    llvm::Value* hashSet(const eshkol_operations_t* op);
    llvm::Value* hashRef(const eshkol_operations_t* op);
    llvm::Value* hashHasKey(const eshkol_operations_t* op);
    llvm::Value* hashRemove(const eshkol_operations_t* op);

    // Hash table queries
    llvm::Value* hashKeys(const eshkol_operations_t* op);
    llvm::Value* hashValues(const eshkol_operations_t* op);
    llvm::Value* hashCount(const eshkol_operations_t* op);
    llvm::Value* hashClear(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;
    std::unordered_map<std::string, llvm::Function*>& function_table_;

    // Callbacks for AST evaluation
    CodegenASTCallback codegen_ast_cb_;
    CodegenTypedASTCallback codegen_typed_ast_cb_;
    void* callback_context_;

    // Cached runtime function declarations
    llvm::Function* hash_table_create_func_;
    llvm::Function* hash_table_set_func_;
    llvm::Function* hash_table_get_func_;
    llvm::Function* hash_table_has_key_func_;
    llvm::Function* hash_table_remove_func_;
    llvm::Function* hash_table_keys_func_;
    llvm::Function* hash_table_values_func_;
    llvm::Function* hash_table_count_func_;
    llvm::Function* hash_table_clear_func_;

    // Initialize runtime function declarations
    void initRuntimeFunctions();

    // Helper to generate code for an AST node
    llvm::Value* codegenAST(const void* ast);

    // Helper to ensure a value is a tagged value struct (packs raw values if needed)
    llvm::Value* ensureTaggedValue(llvm::Value* val, const std::string& name);

    // Helper to extract tagged value to stack-allocated struct pointer
    llvm::Value* extractTaggedValuePtr(llvm::Value* tagged_val, const std::string& name);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_HASH_CODEGEN_H
