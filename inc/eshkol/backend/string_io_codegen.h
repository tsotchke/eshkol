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
     * Create a global string constant with interning (NO header).
     * @param str The string content
     * @return Pointer to the string constant
     * @deprecated Use createStringWithHeader for HEAP_PTR compatibility
     */
    llvm::Value* createString(const char* str);

    /**
     * Create a global string constant WITH header for HEAP_PTR.
     * The header is prepended to the string data, and the returned pointer
     * points to the string data (after header). Use ESHKOL_GET_HEADER(ptr)
     * to access the header.
     * @param str The string content
     * @param subtype The subtype value (default: HEAP_SUBTYPE_STRING)
     * @return Pointer to the string data (header at ptr-8)
     */
    llvm::Value* createStringWithHeader(const char* str, uint8_t subtype = HEAP_SUBTYPE_STRING);

    /**
     * Create a string tagged value.
     * @param str The string content
     * @return Tagged value with HEAP_PTR type (uses header)
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
     * Extract substring: (substring s start end)
     * @param op The operation AST node
     * @return New string as tagged value
     */
    llvm::Value* substring(const eshkol_operations_t* op);

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
     * Convert number to string: (number->string n)
     * @param op The operation AST node
     * @return String as tagged value
     */
    llvm::Value* numberToString(const eshkol_operations_t* op);

    /**
     * Create string of given length: (make-string len [char])
     * @param op The operation AST node
     * @return String as tagged value
     */
    llvm::Value* makeString(const eshkol_operations_t* op);

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
     * Convert list of characters to string: (list->string chars)
     * @param op The operation AST node
     * @return String as tagged value
     */
    llvm::Value* listToString(const eshkol_operations_t* op);

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

    // === Character Operations ===

    /**
     * Convert character to integer: (char->integer c)
     * @param op The operation AST node
     * @return Integer as tagged value
     */
    llvm::Value* charToInteger(const eshkol_operations_t* op);

    /**
     * Convert integer to character: (integer->char n)
     * @param op The operation AST node
     * @return Character as tagged value
     */
    llvm::Value* integerToChar(const eshkol_operations_t* op);

    /**
     * Compare characters: (char=? c1 c2), (char<? c1 c2), etc.
     * @param op The operation AST node
     * @param cmp_type One of "eq", "lt", "gt", "le", "ge"
     * @return Boolean as tagged value
     */
    llvm::Value* charCompare(const eshkol_operations_t* op, const std::string& cmp_type);

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
     * Read a line from input: (read-line port)
     * @param op The operation AST node
     * @return String or eof-object as tagged value
     */
    llvm::Value* readLine(const eshkol_operations_t* op);

    /**
     * Open file for input: (open-input-file filename)
     * @param op The operation AST node
     * @return Port as tagged value
     */
    llvm::Value* openInputFile(const eshkol_operations_t* op);

    /**
     * Open file for output: (open-output-file filename)
     * @param op The operation AST node
     * @return Port as tagged value
     */
    llvm::Value* openOutputFile(const eshkol_operations_t* op);

    /**
     * Close a port: (close-port port)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* closePort(const eshkol_operations_t* op);

    /**
     * Check if value is eof-object: (eof-object? obj)
     * @param op The operation AST node
     * @return Boolean as tagged value
     */
    llvm::Value* eofObject(const eshkol_operations_t* op);

    /**
     * Write a string to a port: (write-string s port)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* writeString(const eshkol_operations_t* op);

    /**
     * Write a string followed by newline: (write-line s [port])
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* writeLine(const eshkol_operations_t* op);

    /**
     * Write a character to a port: (write-char c [port])
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* writeChar(const eshkol_operations_t* op);

    /**
     * Flush an output port: (flush-output-port port)
     * @param op The operation AST node
     * @return Unspecified value
     */
    llvm::Value* flushOutputPort(const eshkol_operations_t* op);

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
    llvm::Function* display_value_func_ = nullptr;

    /**
     * Ensure a value is a raw i64 integer, extracting from tagged_value if needed.
     * This is critical for GEP operations that require integer indices.
     * @param val The value (may be raw i64 or tagged_value struct)
     * @param name Name hint for the extracted value
     * @return Raw i64 value suitable for GEP indices
     */
    llvm::Value* ensureRawInt64(llvm::Value* val, const std::string& name = "raw_idx");

    // Callbacks for AST code generation (set by main codegen)
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);
    using TypedToTaggedFunc = llvm::Value* (*)(void* typed_value, void* context);
    using ConsCreateFunc = llvm::Value* (*)(llvm::Value* car, llvm::Value* cdr, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    TypedToTaggedFunc typed_to_tagged_callback_ = nullptr;
    ConsCreateFunc cons_create_callback_ = nullptr;
    void* callback_context_ = nullptr;

public:
    /**
     * Set callbacks for AST code generation.
     * Called by EshkolLLVMCodeGen to inject dependencies.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        TypedToTaggedFunc typed_to_tagged,
        ConsCreateFunc cons_create,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        typed_to_tagged_callback_ = typed_to_tagged;
        cons_create_callback_ = cons_create;
        callback_context_ = context;
    }

    /**
     * Set the display value function (C runtime).
     */
    void setDisplayValueFunc(llvm::Function* func) {
        display_value_func_ = func;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_STRING_IO_CODEGEN_H
