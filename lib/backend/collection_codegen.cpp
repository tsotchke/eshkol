/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * CollectionCodegen implementation
 *
 * Note: The complex collection implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen methods. This module provides
 * the allocConsCell helper which is self-contained.
 */

#include <eshkol/backend/collection_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

CollectionCodegen::CollectionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("CollectionCodegen initialized");
}

llvm::Value* CollectionCodegen::allocConsCell(llvm::Value* car_val, llvm::Value* cdr_val) {
    // Use arena-based tagged cons cell allocation
    llvm::Function* alloc_func = mem_.getArenaAllocateTaggedConsCell();
    if (!alloc_func) {
        eshkol_warn("arena_allocate_tagged_cons_cell not available");
        return tagged_.packNull();
    }

    // Get global arena pointer
    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("global_arena");
    if (!arena_global) {
        eshkol_warn("global_arena not found");
        return tagged_.packNull();
    }

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global, "arena");

    // Allocate cons cell with car and cdr as tagged values
    // Create temporaries to pass by pointer
    llvm::Value* car_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "car_tmp");
    llvm::Value* cdr_ptr = ctx_.builder().CreateAlloca(ctx_.taggedValueType(), nullptr, "cdr_tmp");
    ctx_.builder().CreateStore(car_val, car_ptr);
    ctx_.builder().CreateStore(cdr_val, cdr_ptr);

    llvm::Value* cons_ptr = ctx_.builder().CreateCall(alloc_func, {arena_ptr, car_ptr, cdr_ptr}, "cons_cell");

    // Pack as tagged value with CONS_PTR type
    return tagged_.packPtr(cons_ptr, ESHKOL_VALUE_CONS_PTR);
}

// Note: The following implementations are complex and depend on:
// - codegenTypedAST, typedValueToTaggedValue (AST code generation)
// - S-expression building for proper display
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* CollectionCodegen::cons(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::cons called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::car(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::car called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::cdr(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::cdr called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::list(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::list called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::listStar(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::listStar called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::isNull(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::isNull called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::isPair(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::isPair called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::makeVector(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::makeVector called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::vector(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::vector called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::vectorLength(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::vectorLength called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::vectorRef(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::vectorRef called - using fallback");
    return tagged_.packNull();
}

llvm::Value* CollectionCodegen::vectorSet(const eshkol_operations_t* op) {
    eshkol_warn("CollectionCodegen::vectorSet called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
