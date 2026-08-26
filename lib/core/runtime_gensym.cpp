/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * gensym — process-wide unique (uninterned) symbol generation.
 *
 * This implementation used to live in lib/core/introspection.cpp, which is
 * correct for the API surface it declares (inc/eshkol/core/introspection.h)
 * but wrong for where it is *linked*: introspection.cpp includes
 * eshkol/llvm_backend.h and is compiled only into eshkol-static, the
 * compiler/tool aggregate — never into libeshkol-runtime.a, the slim
 * archive AOT and JIT user binaries actually link against (the runtime
 * archive is deliberately kept LLVM-free; see CMakeLists.txt's runtime
 * stratification comment). `gensym` is a native-codegen builtin that
 * *generated user programs* call directly, so its definition has to live
 * in a runtime-eligible translation unit or every compiled program fails
 * to link with "undefined symbol: eshkol_gensym_ptr" — exactly the same
 * reason lib/core/symbol_intern.cpp exists as a separate TU from
 * introspection.cpp for eshkol_intern_symbol_lookup.
 *
 * Declared in inc/eshkol/core/introspection.h; defined here instead.
 */

#include <eshkol/core/introspection.h>

#include "arena_memory.h"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {
std::atomic<uint64_t> g_gensym_counter{1};
}

/**
 * @brief Generate a unique (uninterned) symbol with the default "G" prefix.
 *
 * @param arena Arena for symbol string allocation.
 * @return New unique symbol as a tagged value (format G<counter>).
 */
eshkol_tagged_value_t eshkol_gensym(void* arena) {
    return eshkol_gensym_prefix("G", arena);
}

/**
 * @brief Generate a unique (uninterned) symbol with a caller-supplied prefix.
 *
 * Appends a process-wide monotonically increasing counter to @p prefix and
 * allocates the resulting string with a proper symbol object header (so
 * later ESHKOL_GET_HEADER dispatch correctly identifies it as
 * HEAP_SUBTYPE_SYMBOL) in the consolidated tagged-value encoding.
 *
 * @param prefix Prefix for the symbol name (defaults to "G" if NULL).
 * @param arena Arena for symbol string allocation.
 * @return New unique symbol as a tagged value, or a null value if @p arena is NULL or allocation fails.
 */
eshkol_tagged_value_t eshkol_gensym_prefix(const char* prefix, void* arena) {
    if (!arena) {
        eshkol_tagged_value_t null_val;
        null_val.type = ESHKOL_VALUE_NULL;
        null_val.flags = 0;
        null_val.reserved = 0;
        null_val.data.raw_val = 0;
        return null_val;
    }

    uint64_t counter = g_gensym_counter.fetch_add(1, std::memory_order_relaxed);

    // Format: <prefix><counter>
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "%s%llu",
             prefix ? prefix : "G",
             (unsigned long long)counter);

    // Allocate symbol string with a proper object header so ESHKOL_GET_HEADER
    // can identify the subtype. A headerless allocation would make
    // ESHKOL_GET_HEADER read arena bookkeeping bytes as the header, producing
    // garbage subtype values and occasional crashes in introspection code.
    arena_t* a = static_cast<arena_t*>(arena);
    size_t len = strlen(buffer);
    char* sym_str = static_cast<char*>(
        arena_allocate_symbol_with_header(a, len)
    );

    eshkol_tagged_value_t result;
    if (!sym_str) {
        result.type = ESHKOL_VALUE_NULL;
        result.flags = 0;
        result.reserved = 0;
        result.data.raw_val = 0;
        return result;
    }

    memcpy(sym_str, buffer, len + 1);

    // Gensym produces fresh (uninterned) symbols in the consolidated encoding
    // so header.subtype == HEAP_SUBTYPE_SYMBOL is authoritative for readers.
    result.type = ESHKOL_VALUE_HEAP_PTR;
    result.flags = 0;
    result.reserved = 0;
    result.data.ptr_val = reinterpret_cast<uint64_t>(sym_str);

    return result;
}

/**
 * @brief Raw-pointer entry point for eshkol_gensym used by native codegen.
 *
 * Mirrors eshkol_intern_symbol_lookup's calling convention (return the raw
 * heap pointer; the caller packs the HEAP_PTR tag) instead of returning a
 * by-value eshkol_tagged_value_t, so the native LLVM codegen's `(gensym)`
 * builtin can call this directly the same way it calls the runtime helper
 * behind `string->symbol`.
 *
 * @param arena Arena for symbol string allocation.
 * @return Pointer to the new symbol's string data, or NULL (NULL arena or
 *         allocation failure).
 */
void* eshkol_gensym_ptr(void* arena) {
    eshkol_tagged_value_t sym = eshkol_gensym(arena);
    if (sym.type != ESHKOL_VALUE_HEAP_PTR) {
        return nullptr;
    }
    return reinterpret_cast<void*>(sym.data.ptr_val);
}
