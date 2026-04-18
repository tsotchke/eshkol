/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Symbol interning — process-global canonical symbol table with its
 * own process-lifetime backing store (NOT the global arena).
 *
 * Why a dedicated backing store:
 *   Symbols are process-lifetime by design — R7RS §6.5 requires eq?
 *   on symbol literals to hold across every module in the same image,
 *   so the canonical pointer must survive as long as any code that
 *   could reference it. The obvious backing store is the main
 *   arena, but the main arena gets reset during runtime lifecycle
 *   events (REPL restart, test-batch boundary, Python binding
 *   destroy) and between embedded instances — every reset dangles
 *   every interned symbol's char buffer. Storing them here, in a
 *   per-page malloc'd pool that's never freed, breaks that
 *   coupling: symbol pointers stay valid regardless of arena
 *   lifecycle, and dual-instance embedding becomes safe.
 *
 *   Each symbol needs an 8-byte ESHKOL_OBJECT_HEADER immediately
 *   preceding its char data so ESHKOL_GET_HEADER(ptr) at runtime
 *   reads HEAP_SUBTYPE_SYMBOL (not whatever happens to live there).
 *   We allocate (header + chars + NUL) as one block and return
 *   a pointer past the header, matching the invariant the runtime
 *   relies on.
 *
 * This TU also lives in eshkol-static so codegen-emitted calls to
 * eshkol_intern_symbol_lookup (symbol literals in compiled stdlib.o)
 * don't drag introspection.cpp.o into user binaries. All runtime
 * helpers that need interning — string->symbol, procedure-name,
 * type-of — route through the single shared table here.
 */

#include <eshkol/eshkol.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

struct InternedEntry {
    char* symbol_ptr;  /* points past the header to the NUL-terminated data */
};

std::mutex g_symbol_mutex;
std::unordered_map<std::string, InternedEntry> g_interned_symbols;

/*
 * Allocate `len+1` bytes for a symbol's character data, preceded by
 * an 8-byte ESHKOL_OBJECT_HEADER with subtype=HEAP_SUBTYPE_SYMBOL.
 * Returns a pointer to the char data (header sits at ptr-8).
 *
 * Backed by plain malloc because symbols are process-lifetime — we
 * never free them, so an arena's bulk-reset isn't helpful and in
 * fact is actively harmful. Small per-symbol overhead (~40 bytes
 * including malloc bookkeeping) times ~10k symbols in a large
 * process is ~400 KB — negligible.
 */
char* alloc_symbol_block(const char* src, size_t len) {
    size_t header_sz = sizeof(eshkol_object_header_t);
    size_t total = header_sz + len + 1;
    uint8_t* block = static_cast<uint8_t*>(std::malloc(total));
    if (!block) return nullptr;
    eshkol_object_header_t* hdr = reinterpret_cast<eshkol_object_header_t*>(block);
    hdr->subtype = HEAP_SUBTYPE_SYMBOL;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = static_cast<uint32_t>(len + 1);
    char* data = reinterpret_cast<char*>(block + header_sz);
    std::memcpy(data, src, len);
    data[len] = '\0';
    return data;
}

} /* anonymous namespace */

/*
 * Return the canonical char* pointer for a symbol spelled `name`.
 * The pointer refers to a NUL-terminated string with the SYMBOL
 * header at ptr-8 (so ESHKOL_GET_HEADER works). Every distinct
 * spelling maps to exactly one pointer so eq? on symbol literals
 * across modules is pointer equality (R7RS §6.5).
 *
 * Codegen emits `call` to this function for every symbol literal
 * and every user-facing interning path (string->symbol,
 * procedure-name, type-of) routes through it too, so runtime and
 * compile-time interning stay unified.
 *
 * Thread-safe. Returns nullptr on allocation failure or if `name`
 * is NULL.
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

    /* Allocate outside the lock — malloc is fast but we still don't
     * want to hold the interning lock across it. */
    char* sym_str = alloc_symbol_block(name, key.size());
    if (!sym_str) return nullptr;

    /* Insert. Re-check in case another thread won the race — if so,
     * free our speculative allocation so the canonical pointer
     * stays singular. */
    {
        std::lock_guard<std::mutex> lock(g_symbol_mutex);
        auto it = g_interned_symbols.find(key);
        if (it != g_interned_symbols.end()) {
            /* Free the redundant block (header is 8 bytes before data). */
            std::free(reinterpret_cast<uint8_t*>(sym_str) -
                      sizeof(eshkol_object_header_t));
            return it->second.symbol_ptr;
        }
        g_interned_symbols[key] = {sym_str};
    }
    return sym_str;
}
