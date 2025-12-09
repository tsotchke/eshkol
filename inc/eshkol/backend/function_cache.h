/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * FunctionCache - Lazy initialization and caching of C library functions
 *
 * This module provides lazy-loaded, cached declarations for common C library
 * functions used by the Eshkol compiler. Functions are only declared in the
 * LLVM module when first requested, and cached for subsequent uses.
 */
#ifndef ESHKOL_BACKEND_FUNCTION_CACHE_H
#define ESHKOL_BACKEND_FUNCTION_CACHE_H

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

#include <eshkol/backend/type_system.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Function.h>

namespace eshkol {

/**
 * FunctionCache provides lazy-loaded C library function declarations.
 *
 * Each function is declared in the LLVM module only when first requested,
 * then cached for subsequent calls. This avoids cluttering the module with
 * unused function declarations.
 *
 * Usage:
 *   FunctionCache funcs(module, types);
 *   auto malloc_fn = funcs.getMalloc();
 *   auto strlen_fn = funcs.getStrlen();
 */
class FunctionCache {
public:
    /**
     * Construct a FunctionCache for the given module and type system.
     */
    FunctionCache(llvm::Module& mod, TypeSystem& ts);

    // String functions
    llvm::Function* getStrlen();   // size_t strlen(const char*)
    llvm::Function* getStrcmp();   // int strcmp(const char*, const char*)
    llvm::Function* getStrncmp();  // int strncmp(const char*, const char*, size_t)
    llvm::Function* getStrcpy();   // char* strcpy(char*, const char*)
    llvm::Function* getStrcat();   // char* strcat(char*, const char*)
    llvm::Function* getStrstr();   // char* strstr(const char*, const char*)

    // Memory functions
    llvm::Function* getMalloc();   // void* malloc(size_t)
    llvm::Function* getMemcpy();   // void* memcpy(void*, const void*, size_t)
    llvm::Function* getMemset();   // void* memset(void*, int, size_t)

    // Formatting/conversion functions
    llvm::Function* getSnprintf(); // int snprintf(char*, size_t, const char*, ...)
    llvm::Function* getStrtod();   // double strtod(const char*, char**)

    /**
     * Reset all cached functions (needed for REPL mode between evaluations).
     */
    void reset();

private:
    llvm::Module& module;
    TypeSystem& types;

    // Cached function pointers (nullptr until first use)
    llvm::Function* strlen_func;
    llvm::Function* strcmp_func;
    llvm::Function* strncmp_func;
    llvm::Function* strcpy_func;
    llvm::Function* strcat_func;
    llvm::Function* strstr_func;
    llvm::Function* malloc_func;
    llvm::Function* memcpy_func;
    llvm::Function* memset_func;
    llvm::Function* snprintf_func;
    llvm::Function* strtod_func;

    // Helper to get or create a function declaration
    llvm::Function* getOrCreateFunction(const char* name,
                                        llvm::FunctionType* ft,
                                        llvm::Function*& cache);
};

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
#endif // ESHKOL_BACKEND_FUNCTION_CACHE_H
