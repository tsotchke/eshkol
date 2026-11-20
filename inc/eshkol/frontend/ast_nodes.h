/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_AST_NODES_H
#define ESHKOL_FRONTEND_AST_NODES_H

/**
 * @file ast_nodes.h
 * @brief The one owner of AST node storage.
 *
 * The parser and the macro expander build `eshkol_ast_t` nodes and the side
 * structures hung off them: child and argument arrays, parameter and binding
 * name arrays, HoTT type-annotation arrays, patterns, match clauses, macro
 * definitions and tensor-literal dimension arrays. All of them are allocated
 * here and nowhere else, through eshkol_ast_new() and eshkol_ast_new_array().
 *
 * ## Ownership model
 *
 * The owner is a process-rooted, append-only, chunked arena, the sibling of
 * the AST string owner (ADR-0021, ast_strings.h) and built on the same arena.
 * It follows the same lifetime rule: frontend data is valid for the whole
 * compilation that produced it, because a node built during parsing is read
 * by macro expansion, module renaming, type checking and codegen long after
 * the parse unit is gone.
 *
 *  - **Nobody frees an individual node or array.** Rewriting a subtree stores
 *    new pointers and leaves the old storage to the owner. The types placed
 *    here are trivially destructible, so dropping them runs no code.
 *  - **Reachability.** Every chunk is linked from a process-global root, so
 *    LeakSanitizer sees the storage as live for as long as the compilation
 *    lives; it is rooted, not suppressed.
 *  - **Teardown.** eshkol_ast_nodes_teardown() releases every chunk at once.
 *    It is called at the same points as eshkol_ast_strings_teardown(): when
 *    the batch driver's main() returns, in the REPL's ordered exit, and after
 *    each parser-fuzzer input. Embedders that parse repeatedly in one process
 *    keep the process-lifetime default.
 *
 * Retention is proportional to source read plus the nodes expansion
 * synthesizes, the same growth law as the NodeId table and the string owner.
 *
 * ## Sanitizer precision
 *
 * Under AddressSanitizer each allocation is followed by a poisoned redzone,
 * so indexing past the end of a child array is still reported. After teardown
 * a stale node pointer is a heap-use-after-free.
 *
 * ## Thread-safety
 *
 * Allocation takes a mutex. Teardown must not race with allocation.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#include <new>
#include <type_traits>
#include <utility>
extern "C" {
#endif

/**
 * @brief Allocate @p bytes of zero-filled, owner-held, 8-byte-aligned storage.
 *
 * Storage is never individually freed. A request for 0 bytes returns a valid,
 * distinct pointer, as `new T[0]` does.
 *
 * @return The storage, or NULL on allocation failure.
 */
void* eshkol_ast_node_storage_alloc(size_t bytes);

/**
 * @brief True when @p p points into storage the node owner currently holds.
 *
 * Linear in the number of chunks; meant for assertions and tests.
 */
bool eshkol_ast_node_storage_is_owned(const void* p);

/** @brief Counters describing the node owner, for tests and memory reports. */
typedef struct eshkol_ast_nodes_stats {
    /** Allocations served since process start (never reset). */
    uint64_t allocations;
    /** Bytes requested by those allocations (never reset). */
    uint64_t bytes_requested;
    /** Chunks currently held. */
    uint64_t live_chunks;
    /** Bytes of chunk storage currently held. */
    uint64_t live_bytes_reserved;
    /** Completed eshkol_ast_nodes_teardown() calls that released chunks. */
    uint64_t teardowns;
} eshkol_ast_nodes_stats_t;

/** @brief Fill @p out with the current counters. NULL is ignored. */
void eshkol_ast_nodes_stats(eshkol_ast_nodes_stats_t* out);

/**
 * @brief Release all AST node storage at once.
 *
 * Every pointer the owner handed out becomes invalid. Call only when no AST
 * produced in this process will be read again. The owner stays usable.
 *
 * When the environment variable `ESHKOL_AST_NODES_STATS` is set (to anything
 * but `0`/`false`), teardown first writes one line to stderr:
 * `eshkol-ast-nodes: allocations=N requested=N chunks=N reserved=N`.
 */
void eshkol_ast_nodes_teardown(void);

#ifdef __cplusplus
} /* extern "C" */

/**
 * @brief One owner-held @p T, value-initialised (or constructed from @p args).
 *
 * `eshkol_ast_t` construction applies the thread's AST birth location exactly
 * as `new eshkol_ast_t` does.
 *
 * @throws std::bad_alloc when the owner cannot allocate.
 */
template <typename T, typename... Args>
inline T* eshkol_ast_new(Args&&... args) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "the AST node owner never runs destructors");
    static_assert(alignof(T) <= 8, "the AST node owner aligns to 8 bytes");
    void* storage = eshkol_ast_node_storage_alloc(sizeof(T));
    if (!storage) throw std::bad_alloc();
    if constexpr (sizeof...(Args) == 0) {
        return ::new (storage) T{};
    } else {
        return ::new (storage) T(std::forward<Args>(args)...);
    }
}

/**
 * @brief @p count owner-held, value-initialised @p T.
 *
 * Elements of `eshkol_ast_t` arrays receive the thread's AST birth location.
 * A count of 0 returns a valid, distinct pointer.
 *
 * @throws std::bad_alloc when the owner cannot allocate or the size overflows.
 */
template <typename T>
inline T* eshkol_ast_new_array(size_t count) {
    static_assert(std::is_trivially_destructible_v<T>,
                  "the AST node owner never runs destructors");
    static_assert(alignof(T) <= 8, "the AST node owner aligns to 8 bytes");
    if (count > SIZE_MAX / sizeof(T)) throw std::bad_alloc();
    void* storage = eshkol_ast_node_storage_alloc(count * sizeof(T));
    if (!storage) throw std::bad_alloc();
    T* items = static_cast<T*>(storage);
    for (size_t i = 0; i < count; ++i) {
        ::new (static_cast<void*>(&items[i])) T{};
    }
    return items;
}

namespace eshkol::frontend {

/**
 * @brief Scope guard that tears the AST node owner down when a driver's
 * main() returns. Same placement rule as AstStringsTeardownOnReturn.
 */
struct AstNodesTeardownOnReturn {
    AstNodesTeardownOnReturn() = default;
    AstNodesTeardownOnReturn(const AstNodesTeardownOnReturn&) = delete;
    AstNodesTeardownOnReturn& operator=(const AstNodesTeardownOnReturn&) = delete;
    ~AstNodesTeardownOnReturn() { eshkol_ast_nodes_teardown(); }
};

}  // namespace eshkol::frontend
#endif

#endif /* ESHKOL_FRONTEND_AST_NODES_H */
