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

/**
 * @brief Construct a FunctionCodegen bound to the shared codegen context.
 *
 * @param ctx Shared LLVM codegen context (module, builder, type helpers).
 * @param tagged Tagged-value packing/unpacking helper used by closure creation.
 * @param mem Memory codegen helper providing arena allocation entry points.
 */
FunctionCodegen::FunctionCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem)
    : ctx_(ctx)
    , tagged_(tagged)
    , mem_(mem) {
    eshkol_debug("FunctionCodegen initialized");
}

/**
 * @brief Build a tagged CALLABLE value wrapping a function pointer and its captures.
 *
 * With no captures, this just packs the raw function pointer as a legacy
 * CLOSURE_PTR-compatible CALLABLE tagged value (no heap allocation, since
 * there is no environment to store). With captures, it allocates a closure
 * structure in the arena via arena_allocate_closure_with_header (16-byte
 * header + function pointer + capture count, followed by one 16-byte tagged
 * slot per capture), stores each captured tagged value at offset
 * `16 + i*16`, and packs the resulting pointer as a CALLABLE tagged value
 * (header carries CALLABLE_SUBTYPE_CLOSURE).
 *
 * @param func LLVM function to wrap as a closure.
 * @param captures Tagged values to store in the closure's environment, in order.
 * @return Tagged CALLABLE value, or a tagged null if arena allocation or the
 *         global arena pointer is unavailable.
 */
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
    llvm::Value* func_ptr = ctx_.builder().CreatePtrToInt(func, ctx_.intPtrType());
    llvm::Value* num_captures = llvm::ConstantInt::get(ctx_.sizeType(), captures.size());
    llvm::Value* sexpr_ptr = llvm::ConstantInt::get(ctx_.intPtrType(), 0);  // Compiled closures don't carry s-expression data
    llvm::Value* return_type_info = llvm::ConstantInt::get(ctx_.intPtrType(), 0);  // Default type info

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

/**
 * @brief Stub: lambda expression codegen has not been extracted to this module yet.
 *
 * Logs a warning and returns a tagged null. The real implementation still
 * lives in llvm_codegen.cpp because it depends on symbol-table management,
 * nested-function handling, and AST codegen not yet available here.
 *
 * @param op Lambda operation AST node (unused by this fallback).
 * @return Tagged null value.
 */
llvm::Value* FunctionCodegen::lambda(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::lambda called - using fallback");
    return tagged_.packNull();
}

/**
 * @brief Stub: lambda-to-S-expression conversion has not been extracted to this module yet.
 *
 * Logs a warning and returns a constant zero i64 in place of the real
 * S-expression encoding. See the class-level note about llvm_codegen.cpp
 * retaining the full implementation.
 *
 * @param ast Lambda AST node (unused by this fallback).
 * @return Constant i64 zero.
 */
llvm::Value* FunctionCodegen::lambdaToSExpr(const eshkol_ast_t* ast) {
    eshkol_warn("FunctionCodegen::lambdaToSExpr called - using fallback");
    return llvm::ConstantInt::get(ctx_.int64Type(), 0);
}

/**
 * @brief Stub: top-level function-definition codegen has not been extracted to this module yet.
 *
 * Logs a warning and returns a tagged null.
 *
 * @param op Function-definition operation AST node (unused by this fallback).
 * @return Tagged null value.
 */
llvm::Value* FunctionCodegen::functionDefinition(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::functionDefinition called - using fallback");
    return tagged_.packNull();
}

/**
 * @brief Stub: nested (inner) function-definition codegen has not been extracted to this module yet.
 *
 * Logs a warning and returns a tagged null.
 *
 * @param op Nested function-definition operation AST node (unused by this fallback).
 * @return Tagged null value.
 */
llvm::Value* FunctionCodegen::nestedFunctionDefinition(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::nestedFunctionDefinition called - using fallback");
    return tagged_.packNull();
}

/**
 * @brief Stub: direct closure-call codegen has not been extracted to this module yet.
 *
 * Logs a warning and returns a tagged null instead of emitting a call to the
 * closure's function pointer with its captured environment and @p args.
 *
 * @param closure Tagged closure value (unused by this fallback).
 * @param args Pre-evaluated argument values (unused by this fallback).
 * @return Tagged null value.
 */
llvm::Value* FunctionCodegen::closureCall(llvm::Value* closure, const std::vector<llvm::Value*>& args) {
    eshkol_warn("FunctionCodegen::closureCall called - using fallback");
    return tagged_.packNull();
}

/**
 * @brief Stub: `apply` codegen has not been extracted to this module yet.
 *
 * Logs a warning and returns a tagged null.
 *
 * @param op Apply operation AST node (unused by this fallback).
 * @return Tagged null value.
 */
llvm::Value* FunctionCodegen::apply(const eshkol_operations_t* op) {
    eshkol_warn("FunctionCodegen::apply called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
