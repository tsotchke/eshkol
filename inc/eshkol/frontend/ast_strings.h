/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_FRONTEND_AST_STRINGS_H
#define ESHKOL_FRONTEND_AST_STRINGS_H

/**
 * @file ast_strings.h
 * @brief The one owner of AST string payloads (ADR-0020).
 *
 * Every `char*` hung off a frontend structure -- `eshkol_ast_t` identifiers
 * (`variable.id`, `eshkol_func.id`), string and bignum literal text
 * (`str_val.ptr`), the name fields of `eshkol_operations_t` (define / set! /
 * let / lambda rest parameters, type-annotation, extern, import, require,
 * provide, guard, region and diff names), HoTT type-variable names, and the
 * names in `eshkol_macro_def_t` -- is allocated here and nowhere else.
 *
 * ## Ownership model
 *
 * The owner is a process-rooted, append-only, chunked arena that lives beside
 * the `NodeId -> SourceSpan` table in the frontend identity substrate
 * (inc/eshkol/frontend/node_identity.h). The two share one lifetime rule:
 * frontend data is valid for the whole compilation that produced it, because
 * a node built during parsing is read by macro expansion, module renaming,
 * type checking and codegen long after the parse unit is gone.
 *
 *  - **Producers** (the parser, the macro expander, module-private renaming,
 *    the driver's module rewriting, the REPL's import rewriting, the runtime
 *    `eval` reader bridge, the type checker's type substitution, codegen's
 *    synthesized nodes) allocate with eshkol_ast_strdup(),
 *    eshkol_ast_strndup() or eshkol_ast_string_alloc().
 *  - **Nobody frees an individual string.** There is no per-string release:
 *    rewriting a name stores a new pointer and leaves the old bytes to the
 *    owner, and eshkol_ast_clean() no longer touches string payloads. This is
 *    what removes the mixed `new[]` / `malloc` / `free` / `delete[]`
 *    ownership the frontend used to have.
 *  - **Reachability.** Every chunk is linked from a process-global root, so
 *    LeakSanitizer sees the strings as live for as long as the compilation
 *    lives; they are rooted, not suppressed.
 *  - **Teardown.** eshkol_ast_strings_teardown() releases every chunk at once.
 *    The batch driver (eshkol-run) calls it when main() returns; the
 *    interactive REPL calls it in its ordered exit, after the runtime has
 *    shut down and before its explicit leak check. Embedders
 *    that parse repeatedly in one process (the C FFI, runtime `eval`) keep the
 *    process-lifetime default, exactly like the NodeId table.
 *
 * Retention is proportional to source text read (plus names synthesized by
 * expansion), the same growth law as the NodeId table; a resident process
 * that reads new source forever grows with it, as it already does for spans.
 *
 * ## Sanitizer precision
 *
 * Under AddressSanitizer each allocation is followed by a poisoned redzone
 * and unused chunk space is poisoned, so reading past the end of an AST
 * string is still reported, as it was when each string had its own heap
 * block. After teardown the chunks are freed, so a stale AST pointer is a
 * heap-use-after-free rather than silent garbage.
 *
 * ## Thread-safety
 *
 * Allocation takes a mutex (the runtime `eval` bridge may run on worker
 * threads). Teardown must not race with allocation; it is called only once
 * all compilation work in the process is finished.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <string>
#include <string_view>
extern "C" {
#endif

/**
 * @brief Allocate @p bytes of zero-filled, owner-held storage for an AST string.
 *
 * The caller writes the text (including its NUL terminator) into the result.
 * Storage is 1-byte aligned text; it is never individually freed.
 *
 * @return Pointer to @p bytes writable bytes, or NULL on allocation failure.
 *         A request for 0 bytes returns a valid 1-byte empty string.
 */
char* eshkol_ast_string_alloc(size_t bytes);

/**
 * @brief Copy the NUL-terminated string @p s into the owner.
 * @return The owned copy, or NULL when @p s is NULL or allocation fails.
 */
char* eshkol_ast_strdup(const char* s);

/**
 * @brief Copy exactly @p n bytes of @p s into the owner and NUL-terminate.
 *
 * Embedded NULs are copied verbatim; the result has room for @p n + 1 bytes.
 * @return The owned copy, or NULL when @p s is NULL and @p n > 0, or on
 *         allocation failure.
 */
char* eshkol_ast_strndup(const char* s, size_t n);

/**
 * @brief True when @p p points into storage the owner currently holds.
 *
 * Linear in the number of chunks; meant for assertions and tests, not for
 * hot paths.
 */
bool eshkol_ast_string_is_owned(const void* p);

/** @brief Counters describing the owner, for tests and memory reports. */
typedef struct eshkol_ast_strings_stats {
    /** Allocations served since process start (never reset). */
    uint64_t allocations;
    /** Bytes requested by those allocations (never reset). */
    uint64_t bytes_requested;
    /** Chunks currently held. */
    uint64_t live_chunks;
    /** Bytes of chunk storage currently held (the owner's RSS footprint). */
    uint64_t live_bytes_reserved;
    /** Completed eshkol_ast_strings_teardown() calls that released chunks. */
    uint64_t teardowns;
} eshkol_ast_strings_stats_t;

/** @brief Fill @p out with the current counters. NULL is ignored. */
void eshkol_ast_strings_stats(eshkol_ast_strings_stats_t* out);

/**
 * @brief Release every AST string at once.
 *
 * Every pointer the owner handed out becomes invalid. Call only when no AST
 * produced in this process will be read again. The owner stays usable: a
 * later allocation starts a fresh chunk.
 *
 * When the environment variable `ESHKOL_AST_STRINGS_STATS` is set (to
 * anything but `0`/`false`), teardown first writes one line to stderr:
 * `eshkol-ast-strings: allocations=N requested=N chunks=N reserved=N`.
 */
void eshkol_ast_strings_teardown(void);

#ifdef __cplusplus
} /* extern "C" */

/** @brief Owned copy of @p text (length-exact, NUL-terminated). */
inline char* eshkol_ast_string_copy(std::string_view text) {
    return eshkol_ast_strndup(text.data(), text.size());
}

/** @brief Owned copy of @p text (length-exact, NUL-terminated). */
inline char* eshkol_ast_string_copy(const std::string& text) {
    return eshkol_ast_strndup(text.data(), text.size());
}

/** @brief Owned copy of @p text; NULL stays NULL. */
inline char* eshkol_ast_string_copy(const char* text) {
    return eshkol_ast_strdup(text);
}

namespace eshkol::frontend {

/**
 * @brief Scope guard that tears the AST string owner down when a driver's
 * main() returns.
 *
 * Declare it first in main(), before any object that could hold AST
 * pointers, so it is destroyed last. It does not run on std::exit() or
 * std::_Exit() paths, which is correct: the owner is rooted, so memory held
 * at exit is live, not leaked. A driver that always leaves through
 * std::_Exit() (the REPL) calls eshkol_ast_strings_teardown() itself.
 */
struct AstStringsTeardownOnReturn {
    AstStringsTeardownOnReturn() = default;
    AstStringsTeardownOnReturn(const AstStringsTeardownOnReturn&) = delete;
    AstStringsTeardownOnReturn& operator=(const AstStringsTeardownOnReturn&) = delete;
    ~AstStringsTeardownOnReturn() { eshkol_ast_strings_teardown(); }
};

}  // namespace eshkol::frontend
#endif

#endif /* ESHKOL_FRONTEND_AST_STRINGS_H */
