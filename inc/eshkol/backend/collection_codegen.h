/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CollectionCodegen - Collection (list, vector) code generation
 *
 * This module handles:
 * - Pair/cons operations (cons, car, cdr)
 * - List construction and manipulation
 * - Vector construction and access
 */
#ifndef ESHKOL_BACKEND_COLLECTION_CODEGEN_H
#define ESHKOL_BACKEND_COLLECTION_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>

namespace eshkol {

/**
 * CollectionCodegen handles list and vector operations.
 */
class CollectionCodegen {
public:
    /**
     * Construct CollectionCodegen with context and helpers.
     */
    CollectionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem);

    // === Pair/Cons Operations ===

    /**
     * Create a cons cell: (cons car cdr)
     * @param op The operation AST node
     * @return Tagged cons pointer
     */
    llvm::Value* cons(const eshkol_operations_t* op);

    /**
     * Get the car of a pair: (car pair)
     * @param op The operation AST node
     * @return Tagged value
     */
    llvm::Value* car(const eshkol_operations_t* op);

    /**
     * Get the cdr of a pair: (cdr pair)
     * @param op The operation AST node
     * @return Tagged value
     */
    llvm::Value* cdr(const eshkol_operations_t* op);

    // === List Operations ===

    /**
     * Create a list: (list elem1 elem2 ...)
     * @param op The operation AST node
     * @return Tagged cons pointer to list head
     */
    llvm::Value* list(const eshkol_operations_t* op);

    /**
     * Create a list with last element as tail: (list* elem1 elem2 ... tail)
     * @param op The operation AST node
     * @return Tagged cons pointer to list head
     */
    llvm::Value* listStar(const eshkol_operations_t* op);

    /**
     * Check if value is null (empty list): (null? val)
     * @param op The operation AST node
     * @return Tagged boolean
     */
    llvm::Value* isNull(const eshkol_operations_t* op);

    /**
     * Check if value is a pair: (pair? val)
     * @param op The operation AST node
     * @return Tagged boolean
     */
    llvm::Value* isPair(const eshkol_operations_t* op);

    // === Vector Operations ===

    /**
     * Create a vector with size and fill: (make-vector size [fill])
     * @param op The operation AST node
     * @return Tagged vector pointer
     */
    llvm::Value* makeVector(const eshkol_operations_t* op);

    /**
     * Create a vector from elements: (vector elem1 elem2 ...)
     * @param op The operation AST node
     * @return Tagged vector pointer
     */
    llvm::Value* vector(const eshkol_operations_t* op);

    /**
     * Get vector length: (vector-length vec)
     * @param op The operation AST node
     * @return Tagged integer
     */
    llvm::Value* vectorLength(const eshkol_operations_t* op);

    /**
     * Get element at index: (vector-ref vec idx)
     * @param op The operation AST node
     * @return Tagged value
     */
    llvm::Value* vectorRef(const eshkol_operations_t* op);

    /**
     * Set element at index: (vector-set! vec idx val)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* vectorSet(const eshkol_operations_t* op);

    // === Helper: Arena-based Cons Cell ===

    /**
     * Allocate and initialize a cons cell in arena memory.
     * @param car_val The car value (tagged)
     * @param cdr_val The cdr value (tagged)
     * @return Tagged cons pointer
     */
    llvm::Value* allocConsCell(llvm::Value* car_val, llvm::Value* cdr_val);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;

    // Callback for AST code generation
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
#endif // ESHKOL_BACKEND_COLLECTION_CODEGEN_H
