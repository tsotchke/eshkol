/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * SystemCodegen - System, environment, and file system code generation
 *
 * This module handles:
 * - Environment variables (getenv, setenv, unsetenv)
 * - System operations (system, sleep, current-seconds)
 * - File system operations (file-exists?, read-file, write-file, etc.)
 * - Directory operations (directory-list, make-directory, etc.)
 */
#ifndef ESHKOL_BACKEND_SYSTEM_CODEGEN_H
#define ESHKOL_BACKEND_SYSTEM_CODEGEN_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/codegen_context.h>
#include <eshkol/backend/tagged_value_codegen.h>
#include <eshkol/backend/memory_codegen.h>
#include <eshkol/eshkol.h>
#include <llvm/IR/Value.h>
#include <unordered_map>

namespace eshkol {

/**
 * SystemCodegen handles system, environment, and file operations.
 */
class SystemCodegen {
public:
    /**
     * Construct SystemCodegen with context and helpers.
     */
    SystemCodegen(CodegenContext& ctx, TaggedValueCodegen& tagged, MemoryCodegen& mem,
                  std::unordered_map<std::string, llvm::Function*>& function_table);

    // === Environment Operations ===

    /**
     * Get environment variable: (getenv "NAME")
     * @return String value or #f if not set
     */
    llvm::Value* getenv(const eshkol_operations_t* op);

    /**
     * Set environment variable: (setenv "NAME" "VALUE")
     * @return #t on success, #f on failure
     */
    llvm::Value* setenv(const eshkol_operations_t* op);

    /**
     * Unset environment variable: (unsetenv "NAME")
     * @return #t on success, #f on failure
     */
    llvm::Value* unsetenv(const eshkol_operations_t* op);

    // === System Operations ===

    /**
     * Execute shell command: (system "command")
     * @return Exit code as integer
     */
    llvm::Value* systemCall(const eshkol_operations_t* op);

    /**
     * Sleep for seconds: (sleep n)
     * @return null
     */
    llvm::Value* sleep(const eshkol_operations_t* op);

    /**
     * Get current Unix timestamp: (current-seconds)
     * @return Timestamp as integer
     */
    llvm::Value* currentSeconds(const eshkol_operations_t* op);

    /**
     * Exit the program: (exit code)
     * @param code Exit code (integer)
     * @return Does not return
     */
    llvm::Value* exitProgram(const eshkol_operations_t* op);

    /**
     * Get command-line arguments: (command-line)
     * @return List of strings (argv)
     */
    llvm::Value* commandLine(const eshkol_operations_t* op);

    // === File Operations ===

    /**
     * Check if file exists: (file-exists? "path")
     * @return #t or #f
     */
    llvm::Value* fileExists(const eshkol_operations_t* op);

    /**
     * Check if file is readable: (file-readable? "path")
     * @return #t or #f
     */
    llvm::Value* fileReadable(const eshkol_operations_t* op);

    /**
     * Check if file is writable: (file-writable? "path")
     * @return #t or #f
     */
    llvm::Value* fileWritable(const eshkol_operations_t* op);

    /**
     * Delete a file: (file-delete "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* fileDelete(const eshkol_operations_t* op);

    /**
     * Rename/move a file: (file-rename "old" "new")
     * @return #t on success, #f on failure
     */
    llvm::Value* fileRename(const eshkol_operations_t* op);

    /**
     * Get file size: (file-size "path")
     * @return Size in bytes, or #f on error
     */
    llvm::Value* fileSize(const eshkol_operations_t* op);

    /**
     * Read entire file: (read-file "path")
     * @return File contents as string, or #f on error
     */
    llvm::Value* readFile(const eshkol_operations_t* op);

    /**
     * Write string to file: (write-file "path" "contents")
     * @return #t on success, #f on failure
     */
    llvm::Value* writeFile(const eshkol_operations_t* op);

    /**
     * Append string to file: (append-file "path" "contents")
     * @return #t on success, #f on failure
     */
    llvm::Value* appendFile(const eshkol_operations_t* op);

    // === Directory Operations ===

    /**
     * Check if directory exists: (directory-exists? "path")
     * @return #t or #f
     */
    llvm::Value* directoryExists(const eshkol_operations_t* op);

    /**
     * Create directory: (make-directory "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* makeDirectory(const eshkol_operations_t* op);

    /**
     * Delete directory: (delete-directory "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* deleteDirectory(const eshkol_operations_t* op);

    /**
     * List directory contents: (directory-list "path")
     * @return List of filenames
     */
    llvm::Value* directoryList(const eshkol_operations_t* op);

    /**
     * Get current working directory: (current-directory)
     * @return Path as string
     */
    llvm::Value* currentDirectory(const eshkol_operations_t* op);

    /**
     * Set current working directory: (set-current-directory! "path")
     * @return #t on success, #f on failure
     */
    llvm::Value* setCurrentDirectory(const eshkol_operations_t* op);

private:
    CodegenContext& ctx_;
    TaggedValueCodegen& tagged_;
    MemoryCodegen& mem_;
    std::unordered_map<std::string, llvm::Function*>& function_table_;

    // Callback for AST code generation
    using CodegenASTFunc = llvm::Value* (*)(const void* ast, void* context);
    using CodegenTypedASTFunc = void* (*)(const void* ast, void* context);

    CodegenASTFunc codegen_ast_callback_ = nullptr;
    CodegenTypedASTFunc codegen_typed_ast_callback_ = nullptr;
    void* callback_context_ = nullptr;

    // Helper to extract string pointer from tagged value
    llvm::Value* extractStringPtr(llvm::Value* tagged_val);

public:
    /**
     * Set callbacks for AST code generation.
     */
    void setCodegenCallbacks(
        CodegenASTFunc codegen_ast,
        CodegenTypedASTFunc codegen_typed_ast,
        void* context
    ) {
        codegen_ast_callback_ = codegen_ast;
        codegen_typed_ast_callback_ = codegen_typed_ast;
        callback_context_ = context;
    }
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_SYSTEM_CODEGEN_H
