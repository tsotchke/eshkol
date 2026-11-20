/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_strings.cpp
 * @brief Storage for AST string payloads (ADR-0021).
 *
 * One instance of the rooted, chunked owner arena (ast_owner_arena.h). A
 * request larger than a quarter chunk gets a chunk of its own, so a long
 * string literal never strands most of a standard one.
 *
 * The arena's root is what makes the storage *reachable*: LeakSanitizer scans
 * globals, follows the root and every chunk link, and classes each string as
 * live. Nothing in this file is, or needs to be, a suppression.
 */

#include <eshkol/frontend/ast_strings.h>

#include "ast_owner_arena.h"

#include <cstring>

namespace {

/* tests/memory/leak_audit_gate.sh adds the teardown report's `requested` to
 * LeakSanitizer's figure, so rooted retention stays measured rather than
 * disappearing from the per-line slope just because it is no longer a leak. */
eshkol::frontend::detail::OwnerArena g_strings("eshkol-ast-strings",
                                               "ESHKOL_AST_STRINGS_STATS");

}  // namespace

extern "C" char* eshkol_ast_string_alloc(size_t bytes) {
    return static_cast<char*>(g_strings.allocate(bytes));
}

extern "C" char* eshkol_ast_strndup(const char* s, size_t n) {
    if (!s && n > 0) return nullptr;
    char* out = eshkol_ast_string_alloc(n + 1);
    if (!out) return nullptr;
    if (n) std::memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

extern "C" char* eshkol_ast_strdup(const char* s) {
    if (!s) return nullptr;
    return eshkol_ast_strndup(s, std::strlen(s));
}

extern "C" bool eshkol_ast_string_is_owned(const void* p) {
    return g_strings.owns(p);
}

extern "C" void eshkol_ast_strings_stats(eshkol_ast_strings_stats_t* out) {
    if (!out) return;
    const auto c = g_strings.counters();
    out->allocations = c.allocations;
    out->bytes_requested = c.bytes_requested;
    out->live_chunks = c.live_chunks;
    out->live_bytes_reserved = c.live_bytes_reserved;
    out->teardowns = c.teardowns;
}

extern "C" void eshkol_ast_strings_teardown(void) {
    g_strings.teardown();
}
