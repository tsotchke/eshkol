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
     * Get current time in seconds with microsecond precision: (current-time)
     * @return Time in seconds as double (e.g., 1709049600.123456)
     */
    llvm::Value* currentTime(const eshkol_operations_t* op);

    /**
     * Get current time in milliseconds: (current-time-ms)
     * @return Time in milliseconds as double
     */
    llvm::Value* currentTimeMs(const eshkol_operations_t* op);

    /**
     * Get current time in nanoseconds: (current-time-ns)
     * Uses clock_gettime with CLOCK_UPTIME_RAW (macOS) or CLOCK_MONOTONIC (Linux)
     * @return Time in nanoseconds as double
     */
    llvm::Value* currentTimeNs(const eshkol_operations_t* op);

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

    /* ── v1.2 system builtins (delegate to C runtime) ── */
    llvm::Value* osType(const eshkol_operations_t* op);
    llvm::Value* osArch(const eshkol_operations_t* op);
    llvm::Value* hostnameBuiltin(const eshkol_operations_t* op);
    llvm::Value* usernameBuiltin(const eshkol_operations_t* op);
    llvm::Value* cpuCount(const eshkol_operations_t* op);
    llvm::Value* getpidBuiltin(const eshkol_operations_t* op);
    llvm::Value* homeDirectory(const eshkol_operations_t* op);
    llvm::Value* sleepMs(const eshkol_operations_t* op);
    llvm::Value* executableExists(const eshkol_operations_t* op);
    llvm::Value* pathJoin(const eshkol_operations_t* op);
    llvm::Value* pathDirname(const eshkol_operations_t* op);
    llvm::Value* pathBasename(const eshkol_operations_t* op);
    llvm::Value* pathExtname(const eshkol_operations_t* op);
    llvm::Value* pathIsAbsolute(const eshkol_operations_t* op);
    llvm::Value* pathNormalize(const eshkol_operations_t* op);
    llvm::Value* realpathBuiltin(const eshkol_operations_t* op);
    llvm::Value* fileStat(const eshkol_operations_t* op);
    llvm::Value* fileCopy(const eshkol_operations_t* op);
    llvm::Value* mkdirRecursive(const eshkol_operations_t* op);
    llvm::Value* mkdtempBuiltin(const eshkol_operations_t* op);
    llvm::Value* directoryDeleteRecursive(const eshkol_operations_t* op);
    llvm::Value* shellQuote(const eshkol_operations_t* op);
    llvm::Value* processSpawn(const eshkol_operations_t* op);
    llvm::Value* processWait(const eshkol_operations_t* op);
    llvm::Value* pollFd(const eshkol_operations_t* op);
    llvm::Value* tensorSave(const eshkol_operations_t* op);
    llvm::Value* tensorLoad(const eshkol_operations_t* op);

    /* v1.2 batch 2: VM-parity + new builtins */
    llvm::Value* fileChmod(const eshkol_operations_t* op);
    llvm::Value* symlinkCreate(const eshkol_operations_t* op);
    llvm::Value* symlinkRead(const eshkol_operations_t* op);
    llvm::Value* directoryWalk(const eshkol_operations_t* op);
    llvm::Value* mkstempBuiltin(const eshkol_operations_t* op);
    llvm::Value* processKill(const eshkol_operations_t* op);
    llvm::Value* fileMtime(const eshkol_operations_t* op);
    llvm::Value* fileAtime(const eshkol_operations_t* op);
    llvm::Value* fileLock(const eshkol_operations_t* op);
    llvm::Value* fileUnlock(const eshkol_operations_t* op);
    llvm::Value* pathRelative(const eshkol_operations_t* op);
    llvm::Value* pathResolve(const eshkol_operations_t* op);
    llvm::Value* globExpand(const eshkol_operations_t* op);
    llvm::Value* globMatch(const eshkol_operations_t* op);

    /* v1.2 batch 3: advanced process management */
    llvm::Value* processSetpgid(const eshkol_operations_t* op);
    llvm::Value* processKillTree(const eshkol_operations_t* op);
    llvm::Value* processSpawnPty(const eshkol_operations_t* op);
    llvm::Value* processReadNonblocking(const eshkol_operations_t* op);

    /* v1.2 batch 4 */
    llvm::Value* processPid(const eshkol_operations_t* op);
    llvm::Value* fileMmap(const eshkol_operations_t* op);
    llvm::Value* fileMunmap(const eshkol_operations_t* op);
    llvm::Value* kbSave(const eshkol_operations_t* op);
    llvm::Value* kbLoad(const eshkol_operations_t* op);
    llvm::Value* tensorTokenEstimate(const eshkol_operations_t* op);

    /* Noesis requirements */
    llvm::Value* fgMarginal(const eshkol_operations_t* op);
    llvm::Value* fgEntropy(const eshkol_operations_t* op);
    llvm::Value* kbRetract(const eshkol_operations_t* op);

    /* Consciousness engine */
    llvm::Value* makeSubstitution(const eshkol_operations_t* op);
    llvm::Value* unifyBuiltin(const eshkol_operations_t* op);
    llvm::Value* walkBuiltin(const eshkol_operations_t* op);
    llvm::Value* makeFactBuiltin(const eshkol_operations_t* op);
    llvm::Value* makeKbBuiltin(const eshkol_operations_t* op);
    llvm::Value* kbAssertBuiltin(const eshkol_operations_t* op);
    llvm::Value* kbQueryBuiltin(const eshkol_operations_t* op);
    llvm::Value* makeFactorGraphBuiltin(const eshkol_operations_t* op);
    llvm::Value* fgAddFactorBuiltin(const eshkol_operations_t* op);
    llvm::Value* fgInferBuiltin(const eshkol_operations_t* op);
    llvm::Value* freeEnergyBuiltin(const eshkol_operations_t* op);
    llvm::Value* expectedFreeEnergyBuiltin(const eshkol_operations_t* op);
    llvm::Value* makeWorkspaceBuiltin(const eshkol_operations_t* op);
    llvm::Value* wsRegisterBuiltin(const eshkol_operations_t* op);
    llvm::Value* wsStepBuiltin(const eshkol_operations_t* op);

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
