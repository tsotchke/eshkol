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

/* ===== Tagged Value Dispatch ===== */
/* Called from LLVM codegen. Same alloca/store/call/load pattern as bignum. */

void eshkol_unify_tagged(arena_t* arena,
    const eshkol_tagged_value_t* t1, const eshkol_tagged_value_t* t2,
    const eshkol_tagged_value_t* subst, eshkol_tagged_value_t* result);

void eshkol_walk_tagged(
    const eshkol_tagged_value_t* term, const eshkol_tagged_value_t* subst,
    eshkol_tagged_value_t* result);

void eshkol_make_fact_tagged(arena_t* arena,
    const eshkol_tagged_value_t* pred, const eshkol_tagged_value_t* args,
    int32_t arity, eshkol_tagged_value_t* result);

void eshkol_make_kb_tagged(arena_t* arena, eshkol_tagged_value_t* result);

void eshkol_kb_assert_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb, const eshkol_tagged_value_t* fact);

void eshkol_kb_query_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb, const eshkol_tagged_value_t* pattern,
    eshkol_tagged_value_t* result);

void eshkol_make_substitution_tagged(arena_t* arena,
    eshkol_tagged_value_t* result);

/* ===== Display ===== */

void eshkol_display_logic_var(uint64_t var_id, void* file);
void eshkol_display_substitution(const eshkol_substitution_t* s, void* file);
void eshkol_display_fact(const eshkol_fact_t* f, void* file);
void eshkol_display_kb(const eshkol_knowledge_base_t* kb, void* file);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_LOGIC_H */
