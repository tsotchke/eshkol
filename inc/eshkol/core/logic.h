/*
 * Logic Engine for Eshkol Consciousness Engine
 *
 * Implements core symbolic reasoning primitives:
 * - Logic variables (?x syntax, R7RS compatible)
 * - Substitutions (immutable, copy-on-extend)
 * - Unification (Robinson's algorithm with occurs check)
 * - Knowledge base (facts, assert, query)
 *
 * All objects are arena-allocated with object headers.
 * Tagged value dispatch follows the bignum pattern.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#ifndef ESHKOL_CORE_LOGIC_H
#define ESHKOL_CORE_LOGIC_H

#include <eshkol/eshkol.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/* Forward declarations */
typedef struct arena arena_t;

/* Maximum number of logic variables (sufficient for v1.1) */
#define LOGIC_VAR_MAX 65536

/*
 * Substitution: immutable mapping from var_ids to terms.
 * Stored as parallel arrays for cache-friendly access.
 * Layout: [eshkol_object_header_t][eshkol_substitution_t][var_ids...][terms...]
 *
 * Immutability means copy-on-extend: each extend creates a new substitution
 * containing all old bindings plus the new one. This enables backtracking
 * without undo (just discard the extended substitution).
 */
typedef struct eshkol_substitution {
    uint32_t num_bindings;
    uint32_t capacity;
    /* Followed by: uint64_t var_ids[capacity] */
    /* Followed by: eshkol_tagged_value_t terms[capacity] */
} eshkol_substitution_t;

/* Access the var_ids array (immediately after the struct) */
#define SUBST_VAR_IDS(s) ((uint64_t*)((uint8_t*)(s) + sizeof(eshkol_substitution_t)))

/* Access the terms array (after var_ids) */
#define SUBST_TERMS(s)   ((eshkol_tagged_value_t*)((uint8_t*)(s) + sizeof(eshkol_substitution_t) + (s)->capacity * sizeof(uint64_t)))

/*
 * Fact: predicate symbol + argument terms.
 * Layout: [eshkol_object_header_t][eshkol_fact_t][args...]
 *
 * Predicate is stored as a pointer to an interned symbol (HEAP_SUBTYPE_SYMBOL).
 * Symbol comparison uses pointer equality for fast predicate matching.
 */
typedef struct eshkol_fact {
    uint64_t predicate;    /* Pointer to interned symbol (HEAP_SUBTYPE_SYMBOL) */
    uint32_t arity;        /* Number of arguments */
    uint32_t _pad;
    /* Followed by: eshkol_tagged_value_t args[arity] */
} eshkol_fact_t;

/* Access the args array (immediately after the struct) */
#define FACT_ARGS(f) ((eshkol_tagged_value_t*)((uint8_t*)(f) + sizeof(eshkol_fact_t)))

/*
 * Knowledge base: growable array of fact pointers.
 * Layout: [eshkol_object_header_t][eshkol_knowledge_base_t]
 *
 * Uses arena allocation — when full, allocates new larger array and copies.
 * v1.2 will add predicate indexing for O(1) lookup.
 */
typedef struct eshkol_knowledge_base {
    uint32_t num_facts;
    uint32_t capacity;
    eshkol_fact_t** facts;    /* Arena-allocated array of fact pointers */
} eshkol_knowledge_base_t;

#ifdef __cplusplus
extern "C" {
#endif

/* ===== Logic Variables ===== */

/*
 * Create or look up a logic variable by name.
 * Thread-safe global var_id counter. Names stored in static array.
 * Returns the var_id (unique, monotonically increasing).
 * Same name always returns the same var_id.
 */
uint64_t eshkol_make_logic_var(const char* name);

/* Get the name of a logic variable by its id. Returns NULL if invalid. */
const char* eshkol_logic_var_name(uint64_t var_id);

/* Check if a tagged value is a logic variable */
bool eshkol_is_logic_var(const eshkol_tagged_value_t* tv);

/*
 * Clear the process-global logic-variable name registry AND the
 * predicate interning pool. Used by (reset-tests!) to achieve full
 * test isolation — without this, two consecutive test runs share
 * logic-variable IDs and predicate canonical pointers, which can
 * surface as stale unification results or false-positive eq?
 * matches across test boundaries.
 *
 * NOT thread-safe. Call from the main thread only, between test
 * batches or REPL session resets. Calling while logic operations
 * are in flight on another thread is a data race.
 */
void eshkol_logic_registry_reset(void);

/* ===== Substitution ===== */

/*
 * Create an empty substitution with given capacity.
 * Returns NULL on allocation failure.
 */
eshkol_substitution_t* eshkol_make_substitution(arena_t* arena, uint32_t capacity);

/*
 * Immutable extend: returns NEW substitution with binding added.
 * The old substitution is NOT modified (arena-allocated, not freed).
 * Returns NULL on allocation failure.
 */
eshkol_substitution_t* eshkol_extend_subst(arena_t* arena,
    const eshkol_substitution_t* s, uint64_t var_id,
    const eshkol_tagged_value_t* term);

/*
 * Lookup: returns pointer to bound term, or NULL if unbound.
 * Linear scan over bindings (sufficient for v1.1).
 */
const eshkol_tagged_value_t* eshkol_subst_lookup(
    const eshkol_substitution_t* s, uint64_t var_id);

/* ===== Walk (deep resolution) ===== */

/*
 * Shallow walk: follows variable chains.
 * If ?x -> ?y and ?y -> 42, walk(?x, subst) -> 42
 * Returns the term itself if not a logic variable or unbound.
 */
eshkol_tagged_value_t eshkol_walk(const eshkol_tagged_value_t* term,
    const eshkol_substitution_t* subst);

/*
 * Deep walk: resolves inside compound structures (facts).
 * Returns a fully resolved copy of the term.
 */
eshkol_tagged_value_t eshkol_walk_deep(arena_t* arena,
    const eshkol_tagged_value_t* term, const eshkol_substitution_t* subst);

/* ===== Unification (Robinson's algorithm) ===== */

/*
 * Unify two terms under a substitution.
 * Returns extended substitution on success, NULL on failure.
 * Includes occurs check to prevent circular bindings.
 *
 * Handles: logic var + any, value == value, fact + fact (structural)
 */
eshkol_substitution_t* eshkol_unify(arena_t* arena,
    const eshkol_tagged_value_t* t1, const eshkol_tagged_value_t* t2,
    const eshkol_substitution_t* subst);

/* ===== Facts ===== */

/*
 * Create a fact with the given predicate symbol and arguments.
 * predicate should be a pointer to an interned symbol.
 */
eshkol_fact_t* eshkol_make_fact(arena_t* arena, uint64_t predicate,
    const eshkol_tagged_value_t* args, uint32_t arity);

/* ===== Knowledge Base ===== */

/* Create an empty knowledge base */
eshkol_knowledge_base_t* eshkol_make_kb(arena_t* arena);

/* Assert a fact into the knowledge base. KB grows as needed. */
void eshkol_kb_assert(arena_t* arena, eshkol_knowledge_base_t* kb,
    const eshkol_fact_t* fact);

/*
 * Query the KB: returns cons list of substitutions that unify pattern
 * with KB facts. Returns tagged NULL (empty list) if no matches.
 * initial_subst may be NULL (uses empty substitution).
 */
eshkol_tagged_value_t eshkol_kb_query(arena_t* arena,
    const eshkol_knowledge_base_t* kb, const eshkol_fact_t* pattern,
    const eshkol_substitution_t* initial_subst);

/*
 * Prefix variant of eshkol_kb_query. Accepts patterns whose arity is <= the
 * fact's arity, unifying only the first pattern-arity argument pairs. Useful
 * for KBs with mixed arities or provenance-extended tails where the caller
 * wants "match any fact with this head and these prefix args".
 */
eshkol_tagged_value_t eshkol_kb_query_prefix(arena_t* arena,
    const eshkol_knowledge_base_t* kb, const eshkol_fact_t* pattern,
    const eshkol_substitution_t* initial_subst);

/* ===== Tagged Value Dispatch ===== */
/* Called from LLVM codegen. Same alloca/store/call/load pattern as bignum. */

/**
 * @brief Tagged-value entry point for eshkol_unify(), called from LLVM codegen.
 *
 * Extracts a substitution pointer from @p subst (treated as the empty
 * substitution if it is not a HEAP_PTR) and unifies @p t1 with @p t2 under
 * it. On success @p result is set to a HEAP_PTR wrapping the extended
 * substitution; on unification failure @p result is set to the tagged NULL
 * value, which the Scheme surface displays as `#f`.
 *
 * @param arena Arena used for any new substitution allocated during unification.
 * @param t1 First term to unify.
 * @param t2 Second term to unify.
 * @param subst Tagged HEAP_PTR wrapping the substitution to unify under, or NULL/non-HEAP_PTR for empty.
 * @param[out] result Destination tagged value (HEAP_PTR substitution on success, NULL on failure).
 */
void eshkol_unify_tagged(arena_t* arena,
    const eshkol_tagged_value_t* t1, const eshkol_tagged_value_t* t2,
    const eshkol_tagged_value_t* subst, eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_walk(), called from LLVM codegen.
 *
 * Extracts a substitution pointer from @p subst (treated as the empty
 * substitution if it is not a HEAP_PTR) and copies the shallow-walked result
 * of @p term into @p result.
 *
 * @param term Term to resolve.
 * @param subst Tagged HEAP_PTR wrapping the substitution to walk under, or NULL/non-HEAP_PTR for empty.
 * @param[out] result Destination for the walked tagged value.
 */
void eshkol_walk_tagged(
    const eshkol_tagged_value_t* term, const eshkol_tagged_value_t* subst,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_make_fact(), called from LLVM codegen.
 *
 * @p pred must be a HEAP_PTR to a symbol/string; its text is interned via
 * eshkol_intern_predicate() so later unification and KB lookups can compare
 * predicates by pointer equality. Builds a fact with @p arity arguments
 * copied from @p args. On success @p result is set to a HEAP_PTR wrapping
 * the new fact; on missing @p arena/@p pred or allocation failure @p result
 * is set to the tagged NULL value.
 *
 * @param arena Arena used for the fact allocation.
 * @param pred Tagged HEAP_PTR to the predicate symbol/string.
 * @param args Array of @p arity tagged argument values (may be NULL if arity is 0).
 * @param arity Number of arguments in @p args.
 * @param[out] result Destination tagged value (HEAP_PTR fact on success, NULL on failure).
 */
void eshkol_make_fact_tagged(arena_t* arena,
    const eshkol_tagged_value_t* pred, const eshkol_tagged_value_t* args,
    int32_t arity, eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_make_kb(), called from LLVM codegen.
 *
 * On success @p result is set to a HEAP_PTR wrapping a new empty knowledge
 * base; on allocation failure @p result is set to the tagged NULL value.
 *
 * @param arena Arena used for the knowledge-base allocation.
 * @param[out] result Destination tagged value (HEAP_PTR knowledge base on success, NULL on failure).
 */
void eshkol_make_kb_tagged(arena_t* arena, eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_kb_assert(), called from LLVM codegen.
 *
 * Requires both @p kb and @p fact to be non-NULL tagged HEAP_PTR values;
 * otherwise this is a no-op. Mutates @p kb in place (growing its fact array
 * as needed) — there is no result parameter because the operation returns
 * no value on the Scheme side.
 *
 * @param arena Arena used to grow the KB's fact array if it is at capacity.
 * @param kb Tagged HEAP_PTR wrapping the target eshkol_knowledge_base_t.
 * @param fact Tagged HEAP_PTR wrapping the eshkol_fact_t to append.
 */
void eshkol_kb_assert_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb, const eshkol_tagged_value_t* fact);

/**
 * @brief Tagged-value entry point for eshkol_kb_query(), called from LLVM codegen.
 *
 * Requires @p kb and @p pattern to be non-NULL tagged HEAP_PTR values;
 * otherwise @p result is set to the tagged NULL value. Otherwise delegates
 * to eshkol_kb_query() with an empty initial substitution and copies the
 * resulting cons-list of matching substitutions into @p result.
 *
 * @param arena Arena used for substitutions and cons cells built during the query.
 * @param kb Tagged HEAP_PTR wrapping the eshkol_knowledge_base_t to search.
 * @param pattern Tagged HEAP_PTR wrapping the eshkol_fact_t pattern to match.
 * @param[out] result Destination tagged value: cons-list of matching substitutions, or tagged NULL if none/invalid input.
 */
void eshkol_kb_query_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb, const eshkol_tagged_value_t* pattern,
    eshkol_tagged_value_t* result);

/**
 * @brief Tagged-value entry point for eshkol_make_substitution(), called from LLVM codegen.
 *
 * Creates an empty substitution with a fixed default capacity of 8
 * bindings (it grows automatically as needed via eshkol_extend_subst()).
 * On success @p result is set to a HEAP_PTR wrapping the substitution; on
 * allocation failure @p result is set to the tagged NULL value.
 *
 * @param arena Arena used for the substitution allocation.
 * @param[out] result Destination tagged value (HEAP_PTR substitution on success, NULL on failure).
 */
void eshkol_make_substitution_tagged(arena_t* arena,
    eshkol_tagged_value_t* result);

/* ===== Display ===== */

/**
 * @brief Print a logic variable's name, e.g. `?x`.
 *
 * Looks up @p var_id via eshkol_logic_var_name(); if the id has no
 * registered name (out of range or never created), falls back to printing
 * `?_<id>`.
 *
 * @param var_id Logic variable id to print.
 * @param file Destination `FILE*`, or NULL to write to stdout.
 */
void eshkol_display_logic_var(uint64_t var_id, void* file);

/**
 * @brief Print a substitution as `{var -> term, var -> term, ...}`.
 *
 * Prints `{}` if @p s is NULL. Each bound term is formatted according to
 * its tagged type (integers, doubles via eshkol_fprint_double(), booleans as
 * `#t`/`#f`, nested logic variables, `()` for NULL, interned symbol strings
 * for HEAP_PTR/HEAP_SUBTYPE_SYMBOL, or a generic `#<heap:N>` / `#<type:N>`
 * placeholder for anything else).
 *
 * @param s Substitution to print, or NULL.
 * @param file Destination `FILE*`, or NULL to write to stdout.
 */
void eshkol_display_substitution(const eshkol_substitution_t* s, void* file);

/**
 * @brief Print a fact as `(predicate arg1 arg2 ...)`.
 *
 * Prints `(fact)` if @p f is NULL, and `?` in place of the predicate if it
 * is unset (0). Arguments use the same per-type formatting as
 * eshkol_display_substitution()'s bound terms.
 *
 * @param f Fact to print, or NULL.
 * @param file Destination `FILE*`, or NULL to write to stdout.
 */
void eshkol_display_fact(const eshkol_fact_t* f, void* file);

/**
 * @brief Print a short human-readable summary of a knowledge base.
 *
 * Writes `#<knowledge-base: empty>` if @p kb is NULL, otherwise
 * `#<knowledge-base: N facts>`.
 *
 * @param kb Knowledge base to describe, or NULL.
 * @param file Destination `FILE*`, or NULL to write to stdout.
 */
void eshkol_display_kb(const eshkol_knowledge_base_t* kb, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_LOGIC_H */
