/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Symbol interning — process-global canonical symbol table.
 *
 * Split from introspection.cpp so codegen-emitted calls to
 * `eshkol_intern_symbol_lookup` (used for symbol literals in compiled
 * stdlib.o) don't force the linker to pull in introspection.cpp.o, which
 * has unresolved references to ReplJITContext (only defined in
 * eshkol-repl-lib).  User binaries compiled with eshkol-run link only
 * against libeshkol-static.a, so they must not drag in REPL symbols.
 *
 * All callers that need symbol interning — runtime helpers
 * (string->symbol, procedure-name, type-of) and codegen-emitted literals
 * — route through the shared table here via `eshkol_intern_symbol_lookup`.
 */

#include <eshkol/eshkol.h>
#include "arena_memory.h"

#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct InternedEntry {
    char* symbol_ptr;
};

std::mutex g_symbol_mutex;
std::unordered_map<std::string, InternedEntry> g_interned_symbols;

} /* anonymous namespace */

/*
 * Return the canonical char* pointer for a symbol spelled `name`. The
 * pointer refers to a NUL-terminated string living in the global arena,
 * with a SYMBOL header at ptr-8 (so ESHKOL_GET_HEADER works). Every
 * distinct spelling maps to exactly one pointer so `eq?` on symbol
 * literals across modules is pointer equality (R7RS §6.5).
 *
 * Codegen emits `call` to this function for every symbol literal and
 * every user-facing interning path (string->symbol, procedure-name,
 * type-of) routes through it too, so runtime and compile-time interning
 * stay unified.
 *
 * Thread-safe. Returns nullptr on allocation failure or if `name` is
 * NULL.
 */
extern "C" void* eshkol_intern_symbol_lookup(const char* name) {
    if (!name) return nullptr;

    std::string key(name);

    /* Fast path: already interned. */
    {
        std::lock_guard<std::mutex> lock(g_symbol_mutex);
        auto it = g_interned_symbols.find(key);
        if (it != g_interned_symbols.end()) {
            return it->second.symbol_ptr;
        }
    }

    /* Allocate outside the lock — arena allocation can be expensive,
     * and arena allocators have their own internal synchronization. */
    arena_t* a = get_global_arena();
    size_t len = key.size();
    char* sym_str = static_cast<char*>(
        arena_allocate_symbol_with_header(a, len)
    );
    if (!sym_str) return nullptr;
    std::memcpy(sym_str, name, len + 1);

    /* Insert. Re-check in case another thread won the race. */
    {
        std::lock_guard<std::mutex> lock(g_symbol_mutex);
        auto it = g_interned_symbols.find(key);
        if (it != g_interned_symbols.end()) {
            return it->second.symbol_ptr;
        }
        g_interned_symbols[key] = {sym_str};
    }
    return sym_str;
}
