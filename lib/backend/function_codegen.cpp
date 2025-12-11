/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * FunctionCodegen implementation
 *
 * Note: The complex lambda and closure implementations remain in llvm_codegen.cpp
 * for now due to extensive dependencies on AST codegen, symbol tables, and
 * closure capture analysis. This module provides the interface and helper functions.
 */

#include <eshkol/backend/function_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>

namespace eshkol {

FunctionCodegen::FunctionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("FunctionCodegen initialized");
}

llvm::Value* FunctionCodegen::createClosure(llvm::Function* func, const std::vector<llvm::Value*>& captures) {
    if (captures.empty()) {
        // No captures - just return function pointer as closure
        // NOTE: Uses legacy CLOSURE_PTR since there's no allocation (no header).
        // The isClosure() helper recognizes both legacy and consolidated types.
        llvm::Value* func_ptr = ctx_.builder().CreatePtrToInt(func, ctx_.int64Type());
        return tagged_.packPtr(
            ctx_.builder().CreateIntToPtr(func_ptr, ctx_.ptrType()),
            ESHKOL_VALUE_CALLABLE
        );
    }

    // Allocate closure structure in arena with header (consolidated CALLABLE type)
    llvm::Function* alloc_closure = mem_.getArenaAllocateClosureWithHeader();
    if (!alloc_closure) {
        eshkol_warn("arena_allocate_closure_with_header not available");
        return tagged_.packNull();
    }

    llvm::GlobalVariable* arena_global = ctx_.module().getNamedGlobal("global_arena");
    if (!arena_global) {
        eshkol_warn("global_arena not found");
        return tagged_.packNull();
    }

    llvm::Value* arena_ptr = ctx_.builder().CreateLoad(ctx_.ptrType(), arena_global, "arena");
    llvm::Value* func_ptr = ctx_.builder().CreatePtrToInt(func, ctx_.int64Type());
    llvm::Value* num_captures = llvm::ConstantInt::get(ctx_.int64Type(), captures.size());
    llvm::Value* sexpr_ptr = llvm::ConstantInt::get(ctx_.int64Type(), 0);  // No sexpr for now
    llvm::Value* return_type_info = llvm::ConstantInt::get(ctx_.int64Type(), 0);  // Default type info

    llvm::Value* closure_ptr = ctx_.builder().CreateCall(
        alloc_closure,
        {arena_ptr, func_ptr, num_captures, sexpr_ptr, return_type_info},
        "closure"
    );

    // Store captured values
    for (size_t i = 0; i < captures.size(); i++) {
        llvm::Value* capture_idx = llvm::ConstantInt::get(ctx_.int64Type(), i);
        // Calculate offset: closure_ptr + 16 + (i * 16) for tagged values
        llvm::Value* offset = ctx_.builder().CreateAdd(
            llvm::ConstantInt::get(ctx_.int64Type(), 16),
            ctx_.builder().CreateMul(capture_idx, llvm::ConstantInt::get(ctx_.int64Type(), 16))
        );
        llvm::Value* capture_ptr = ctx_.builder().CreateGEP(
            ctx_.int8Type(),
            closure_ptr,
            offset,
            "capture_ptr"
        );
        llvm::Value* capture_typed_ptr = ctx_.builder().CreateBitCast(
            capture_ptr,
            llvm::PointerType::getUnqual(ctx_.taggedValueType())
        );
        ctx_.builder().CreateStore(captures[i], capture_typed_ptr);
    }

    // Pack using consolidated CALLABLE type (header contains CALLABLE_SUBTYPE_CLOSURE)
    return tagged_.packCallable(closure_ptr);
}

// Note: The following implementations are complex and depend on:
// - Symbol table management for captures
// - Nested function handling
// - S-expression building
// - AST code generation
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.

llvm::Value* FunctionCodegen::lambda(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::lambda called - using fallback");
    return tagged_.packNull();
}

llvm::Value* FunctionCodegen::lambdaToSExpr(const eshkol_ast_t* ast) {
    eshkol_warn("FunctionCodegen::lambdaToSExpr called - using fallback");
    return llvm::ConstantInt::get(ctx_.int64Type(), 0);
}

llvm::Value* FunctionCodegen::functionDefinition(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::functionDefinition called - using fallback");
    return tagged_.packNull();
}

llvm::Value* FunctionCodegen::nestedFunctionDefinition(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::nestedFunctionDefinition called - using fallback");
    return tagged_.packNull();
}

llvm::Value* FunctionCodegen::closureCall(llvm::Value* closure, const std::vector<llvm::Value*>& args) {
    eshkol_warn("FunctionCodegen::closureCall called - using fallback");
    return tagged_.packNull();
}

llvm::Value* FunctionCodegen::apply(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::apply called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
