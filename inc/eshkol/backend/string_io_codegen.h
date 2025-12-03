/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * StringIOCodegen - String and I/O code generation
 *
 * This module handles:
 * - String literal creation with interning
 * - String operations (length, ref, append, compare, etc.)
 * - Basic I/O operations (display, newline, read)
 */
#ifndef ESHKOL_BACKEND_STRING_IO_CODEGEN_H
#define ESHKOL_BACKEND_STRING_IO_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <string>

namespace eshkol {

/**
 * StringIOCodegen handles string and I/O operations.
 */
class StringIOCodegen {
public:
    /**
     * Construct StringIOCodegen with context and tagged value helper.
     */
    StringIOCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged);

    // === String Literal Creation ===

    /**
     * Create a global string constant with interning.
     * @param str The string content
     * @return Pointer to the string constant
     */
    llvm::Value* createString(const char* str);

    /**
     * Create a string tagged value.
     * @param str The string content
     * @return Tagged value with STRING_PTR type
     */
    llvm::Value* packString(const char* str);

    // === String Operations ===

    /**
     * Get string length: (string-length s)
     * @param op The operation AST node
     * @return Length as tagged integer
     */
    llvm::Value* stringLength(const eshkol_operations_t* op);

    /**
     * Get character at index: (string-ref s idx)
     * @param op The operation AST node
     * @return Character as tagged value
     */
    llvm::Value* stringRef(const eshkol_operations_t* op);

    /**
     * Append strings: (string-append s1 s2 ...)
     * @param op The operation AST node
     * @return New string as tagged value
     */
    llvm::Value* stringAppend(const eshkol_operations_t* op);

    /**
     * Compare strings: (string=? s1 s2), (string<? s1 s2), etc.
     * @param op The operation AST node
     * @param cmp_type One of "eq", "lt", "gt", "le", "ge"
     * @return Boolean as tagged value
     */
    llvm::Value* stringCompare(const eshkol_operations_t* op, const std::string& cmp_type);

    /**
     * Convert string to number: (string->number s)
     * @param op The operation AST node
     * @return Number as tagged value
     */
    llvm::Value* stringToNumber(const eshkol_operations_t* op);

    /**
     * Set character at index: (string-set! s idx char)
     * @param op The operation AST node
     * @return Void/unspecified value
     */
    llvm::Value* stringSet(const eshkol_operations_t* op);

    /**
     * Convert string to list of characters: (string->list s)
     * @param op The operation AST node
     * @return List as tagged value
     */
    llvm::Value* stringToList(const eshkol_operations_t* op);

    /**
     * Split string by delimiter: (string-split s delim)
     * @param op The operation AST node
     * @return List of strings
     */
    llvm::Value* stringSplit(const eshkol_operations_t* op);

    /**
     * Check if string contains substring: (string-contains? s sub)
     * @param op The operation AST node
     * @return Boolean as tagged value
     */
    llvm::Value* stringContains(const eshkol_operations_t* op);

    /**
     * Find index of substring: (string-index s sub)
     * @param op The operation AST node
     * @return Index or #f as tagged value
     */
    llvm::Value* stringIndex(const eshkol_operations_t* op);

    /**
     * Convert to uppercase: (string-upcase s)
     * @param op The operation AST node
     * @return New string as tagged value
     */
    llvm::Value* stringUpcase(const eshkol_operations_t* op);

    /**
     * Convert to lowercase: (string-downcase s)
     * @param op The operation AST node
     * @return New string as tagged value
     */
    llvm::Value* stringDowncase(const eshkol_operations_t* op);

    // === I/O Operations ===

    /**
     * Display value: (display val)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* display(const eshkol_operations_t* op);

    /**
     * Print newline: (newline)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* newline(const eshkol_operations_t* op);

    /**
     * Read a line from input: (read-line)
     * @param op The operation AST node
     * @return String as tagged value
     */
    llvm::Value* readLine(const eshkol_operations_t* op);

    // === Printf Declaration ===

    /**
     * Get or declare the printf function.
     * @return LLVM Function for printf
     */
    llvm::Function* getPrintf();

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    llvm::Function* printf_func_ = nullptr;

    // Callback for AST code generation (set by main codegen)
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    CodegenASTFunc codegen_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callback for AST code generation.
     * Called by EshkolLLVMCodeGen to inject dependencies.
     */
    void setCodegenCallback(CodegenASTFunc callback, void* context) {
        codegen_ast_callback_ = callback;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_STRING_IO_CODEGEN_H
