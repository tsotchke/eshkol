/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * FunctionCache implementation - Lazy C library function declarations
 */

#include <eshkol/backend/function_cache.h>

#ifdef ESHKOL_LLVM_BACKEND_ENABLED

namespace eshkol {

FunctionCache::FunctionCache(llvm::Module& mod, TypeSystem& ts)
    : module(mod), types(ts),
      strlen_func(nullptr), strcmp_func(nullptr), strcpy_func(nullptr),
      strcat_func(nullptr), strstr_func(nullptr), malloc_func(nullptr),
      memcpy_func(nullptr), memset_func(nullptr), snprintf_func(nullptr),
      strtod_func(nullptr) {}

void FunctionCache::reset() {
    strlen_func = nullptr;
    strcmp_func = nullptr;
    strcpy_func = nullptr;
    strcat_func = nullptr;
    strstr_func = nullptr;
    malloc_func = nullptr;
    memcpy_func = nullptr;
    memset_func = nullptr;
    snprintf_func = nullptr;
    strtod_func = nullptr;
}

llvm::Function* FunctionCache::getOrCreateFunction(const char* name,
                                                    llvm::FunctionType* ft,
                                                    llvm::Function*& cache) {
    if (!cache) {
        cache = module.getFunction(name);
        if (!cache) {
            cache = llvm::Function::Create(ft, llvm::Function::ExternalLinkage,
                                           name, &module);
        }
    }
    return cache;
}

// size_t strlen(const char*)
llvm::Function* FunctionCache::getStrlen() {
    if (!strlen_func) {
        auto ft = llvm::FunctionType::get(types.getInt64Type(),
            {types.getPtrType()}, false);
        getOrCreateFunction("strlen", ft, strlen_func);
    }
    return strlen_func;
}

// int strcmp(const char*, const char*)
llvm::Function* FunctionCache::getStrcmp() {
    if (!strcmp_func) {
        auto ft = llvm::FunctionType::get(types.getInt32Type(),
            {types.getPtrType(), types.getPtrType()}, false);
        getOrCreateFunction("strcmp", ft, strcmp_func);
    }
    return strcmp_func;
}

// char* strcpy(char*, const char*)
llvm::Function* FunctionCache::getStrcpy() {
    if (!strcpy_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getPtrType(), types.getPtrType()}, false);
        getOrCreateFunction("strcpy", ft, strcpy_func);
    }
    return strcpy_func;
}

// char* strcat(char*, const char*)
llvm::Function* FunctionCache::getStrcat() {
    if (!strcat_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getPtrType(), types.getPtrType()}, false);
        getOrCreateFunction("strcat", ft, strcat_func);
    }
    return strcat_func;
}

// char* strstr(const char*, const char*)
llvm::Function* FunctionCache::getStrstr() {
    if (!strstr_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getPtrType(), types.getPtrType()}, false);
        getOrCreateFunction("strstr", ft, strstr_func);
    }
    return strstr_func;
}

// void* malloc(size_t)
llvm::Function* FunctionCache::getMalloc() {
    if (!malloc_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getInt64Type()}, false);
        getOrCreateFunction("malloc", ft, malloc_func);
    }
    return malloc_func;
}

// void* memcpy(void*, const void*, size_t)
llvm::Function* FunctionCache::getMemcpy() {
    if (!memcpy_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getPtrType(), types.getPtrType(), types.getInt64Type()}, false);
        getOrCreateFunction("memcpy", ft, memcpy_func);
    }
    return memcpy_func;
}

// void* memset(void*, int, size_t)
llvm::Function* FunctionCache::getMemset() {
    if (!memset_func) {
        auto ft = llvm::FunctionType::get(types.getPtrType(),
            {types.getPtrType(), types.getInt32Type(), types.getInt64Type()}, false);
        getOrCreateFunction("memset", ft, memset_func);
    }
    return memset_func;
}

// int snprintf(char*, size_t, const char*, ...)
llvm::Function* FunctionCache::getSnprintf() {
    if (!snprintf_func) {
        auto ft = llvm::FunctionType::get(types.getInt32Type(),
            {types.getPtrType(), types.getInt64Type(), types.getPtrType()}, true);
        getOrCreateFunction("snprintf", ft, snprintf_func);
    }
    return snprintf_func;
}

// double strtod(const char*, char**)
llvm::Function* FunctionCache::getStrtod() {
    if (!strtod_func) {
        auto ft = llvm::FunctionType::get(types.getDoubleType(),
            {types.getPtrType(), types.getPtrType()}, false);
        getOrCreateFunction("strtod", ft, strtod_func);
    }
    return strtod_func;
}

} // namespace eshkol

#endif // ESHKOL_LLVM_BACKEND_ENABLED
