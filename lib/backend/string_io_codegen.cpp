/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * StringIOCodegen implementation
 *
 * Note: The complex string and I/O implementations remain in llvm_codegen.cpp
 * for now due to dependencies on AST codegen methods. This module provides
 * the basic string creation and printf utilities which are self-contained.
 */

#include <eshkol/backend/string_io_codegen.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/logger.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Function.h>

namespace eshkol {

StringIOCodegen::StringIOCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged)
    : ctx_(ctx)
    , tagged_(tagged)
    , printf_func_(nullptr) {
    eshkol_debug("StringIOCodegen initialized");
}

llvm::Value* StringIOCodegen::createString(const char* str) {
    if (!str) return nullptr;

    // Use context's string interning
    return ctx_.internString(str);
}

llvm::Value* StringIOCodegen::packString(const char* str) {
    llvm::Value* str_ptr = createString(str);
    if (!str_ptr) return tagged_.packNull();
    return tagged_.packPtr(str_ptr, ESHKOL_VALUE_STRING_PTR);
}

llvm::Function* StringIOCodegen::getPrintf() {
    if (printf_func_) return printf_func_;

    // Declare printf if not already declared
    printf_func_ = ctx_.module().getFunction("printf");
    if (!printf_func_) {
        llvm::FunctionType* printf_type = llvm::FunctionType::get(
            ctx_.int32Type(),
            {ctx_.ptrType()},
            true  // variadic
        );
        printf_func_ = llvm::Function::Create(
            printf_type,
            llvm::Function::ExternalLinkage,
            "printf",
            &ctx_.module()
        );
    }
    return printf_func_;
}

llvm::Value* StringIOCodegen::newline(const eshkol_operations_t* op) {
    // Simple newline - just print \n
    llvm::Value* newline_str = createString("\n");
    ctx_.builder().CreateCall(getPrintf(), {newline_str});
    return tagged_.packNull();
}

// Note: The following implementations are complex and depend on:
// - codegenAST, codegenTypedAST (AST code generation)
// - Memory allocation for string manipulation
// - Runtime library functions
//
// These implementations remain in llvm_codegen.cpp until those modules are extracted.
// The functions below provide stubs that warn when called.

llvm::Value* StringIOCodegen::stringLength(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringLength called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringRef(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringRef called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringAppend(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringAppend called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringCompare(const eshkol_operations_t* op, const std::string& cmp_type) {
    eshkol_warn("StringIOCodegen::stringCompare called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringToNumber(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringToNumber called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringSet(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringSet called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringToList(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringToList called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringSplit(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringSplit called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringContains(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringContains called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringIndex(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringIndex called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringUpcase(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringUpcase called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::stringDowncase(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::stringDowncase called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::display(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::display called - using fallback");
    return tagged_.packNull();
}

llvm::Value* StringIOCodegen::readLine(const eshkol_operations_t* op) {
    eshkol_warn("StringIOCodegen::readLine called - using fallback");
    return tagged_.packNull();
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
