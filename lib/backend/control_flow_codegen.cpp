/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ControlFlowCodegen implementation
 *
 * Note: The full control flow implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen methods. This module provides
 * the isTruthy helper which is self-contained, and stubs for the others
 * that will be populated as dependencies are extracted.
 */

#include <eshkol/backend/control_flow_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <eshkol/eshkol.h>

namespace eshkol {

ControlFlowCodegen::ControlFlowCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged) {
    eshkol_debug("ControlFlowCodegen initialized");
}

llvm::Value* ControlFlowCodegen::isTruthy(llvm::Value* val) {
    if (!val) return llvm::ConstantInt::getFalse(ctx_.context());

    // Handle raw int64 - truthy if non-zero
    if (val->getType()->isIntegerTy(64)) {
        return ctx_.builder().CreateICmpNE(val, llvm::ConstantInt::get(ctx_.int64Type(), 0));
    }

    // Handle raw double - truthy if non-zero
    if (val->getType()->isDoubleTy()) {
        return ctx_.builder().CreateFCmpONE(val, llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
    }

    // Handle tagged_value
    if (val->getType() == ctx_.taggedValueType()) {
        llvm::Value* type = tagged_.getType(val);
        llvm::Value* base_type = ctx_.builder().CreateAnd(type,
            llvm::ConstantInt::get(ctx_.int8Type(), 0x0F));

        // Check for false/null (type 0 with value 0)
        llvm::Value* is_null_type = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_NULL));
        llvm::Value* data = tagged_.unpackInt64(val);
        llvm::Value* is_false_val = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* is_null_or_false = ctx_.builder().CreateAnd(is_null_type, is_false_val);

        // Check for BOOL type with value 0 (#f)
        llvm::Value* is_bool = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_BOOL));
        llvm::Value* bool_is_false = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* bool_false = ctx_.builder().CreateAnd(is_bool, bool_is_false);

        // Check for zero (int or double)
        llvm::Value* is_int = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_INT64));
        llvm::Value* int_is_zero = ctx_.builder().CreateICmpEQ(data,
            llvm::ConstantInt::get(ctx_.int64Type(), 0));
        llvm::Value* int_zero = ctx_.builder().CreateAnd(is_int, int_is_zero);

        llvm::Value* is_double = ctx_.builder().CreateICmpEQ(base_type,
            llvm::ConstantInt::get(ctx_.int8Type(), ESHKOL_VALUE_DOUBLE));
        llvm::Value* double_val = tagged_.unpackDouble(val);
        llvm::Value* double_is_zero = ctx_.builder().CreateFCmpOEQ(double_val,
            llvm::ConstantFP::get(ctx_.doubleType(), 0.0));
        llvm::Value* double_zero = ctx_.builder().CreateAnd(is_double, double_is_zero);

        // Truthy = NOT (null/false OR bool-false OR int-zero OR double-zero)
        llvm::Value* is_falsy = ctx_.builder().CreateOr(is_null_or_false,
            ctx_.builder().CreateOr(bool_false, ctx_.builder().CreateOr(int_zero, double_zero)));
        return ctx_.builder().CreateNot(is_falsy);
    }

    // Default: assume truthy
    return llvm::ConstantInt::getTrue(ctx_.context());
}

// Note: The following implementations are complex and depend on:
// - codegenTypedAST, typedValueToTaggedValue (AST code generation)
// - codegenNestedFunctionDefinition, codegenVariableDefinition (defines in begin)
// - packBoolToTaggedValue
//
// These implementations remain in llvm_codegen.cpp until AST codegen is extracted.
// The functions below use callbacks to the main codegen when available.

llvm::Value* ControlFlowCodegen::codegenAnd(const eshkol_operations_t* op) {
    // Stub - actual implementation in llvm_codegen.cpp::codegenAnd
    // This will be populated when AST codegen is extracted
    eshkol_warn("ControlFlowCodegen::codegenAnd called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ControlFlowCodegen::codegenOr(const eshkol_operations_t* op) {
    eshkol_warn("ControlFlowCodegen::codegenOr called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ControlFlowCodegen::codegenCond(const eshkol_operations_t* op) {
    eshkol_warn("ControlFlowCodegen::codegenCond called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ControlFlowCodegen::codegenIf(const eshkol_operations_t* op) {
    eshkol_warn("ControlFlowCodegen::codegenIf called - using fallback");
    return tagged_.packNull();
}

llvm::Value* ControlFlowCodegen::codegenBegin(const eshkol_operations_t* op) {
    eshkol_warn("ControlFlowCodegen::codegenBegin called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
