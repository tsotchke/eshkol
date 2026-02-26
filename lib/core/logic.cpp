/*
 * Logic Engine Implementation for Eshkol Consciousness Engine
 *
 * Implements: logic variables, substitutions, unification, knowledge base.
 * All allocations use arena_allocate_with_header (bignum pattern).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <eshkol/core/logic.h>
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <atomic>
#include <mutex>

/* ===== Logic Variable Registry ===== */

/* Global variable name registry. Thread-safe via mutex (protects g_var_names array)
 * and atomic counter (g_var_count). The mutex serializes find+register in
 * eshkol_make_logic_var and reads in eshkol_logic_var_name. */
static const char* g_var_names[LOGIC_VAR_MAX];
static std::atomic<uint64_t> g_var_count{0};
static std::mutex g_var_mutex;

/* Map from name to var_id for deduplication */
static uint64_t find_var_by_name(const char* name) {
    for (uint64_t i = 0; i < g_var_count; i++) {
        if (g_var_names[i] && strcmp(g_var_names[i], name) == 0) {
            return i;
        }
    }
    return UINT64_MAX; /* not found */
}

/* Static string pool for logic variable names — avoids malloc leak. */
static char g_var_name_pool[LOGIC_VAR_MAX * 64]; /* 64 chars per name max */
static std::atomic<size_t> g_var_name_pool_offset{0};

static const char* intern_var_name(const char* name) {
    size_t len = strlen(name);
    if (len >= 63) len = 63; /* truncate to fit pool slot */
    size_t needed = len + 1;
    size_t offset = g_var_name_pool_offset.load(std::memory_order_relaxed);
    do {
        if (offset + needed > sizeof(g_var_name_pool)) {
            eshkol_error("logic variable name pool exhausted");
            return "<pool-exhausted>"; /* safe static string, not dangling caller ptr */
        }
    } while (!g_var_name_pool_offset.compare_exchange_weak(
                 offset, offset + needed,
                 std::memory_order_acq_rel, std::memory_order_relaxed));
    memcpy(g_var_name_pool + offset, name, len);
    g_var_name_pool[offset + len] = '\0';
    return g_var_name_pool + offset;
}

uint64_t eshkol_make_logic_var(const char* name) {
    if (!name) return UINT64_MAX;

    /* Lock to protect g_var_names[] array access (find + register must be atomic) */
    std::lock_guard<std::mutex> lock(g_var_mutex);

    /* Check if already registered */
    uint64_t existing = find_var_by_name(name);
    if (existing != UINT64_MAX) return existing;

    /* Register new variable */
    uint64_t count = g_var_count.load();
    if (count >= LOGIC_VAR_MAX) {
        eshkol_error("logic variable limit exceeded (%d)", LOGIC_VAR_MAX);
        return UINT64_MAX;
    }

    uint64_t id = g_var_count.fetch_add(1);
    if (id >= LOGIC_VAR_MAX) {
        /* Another thread beat us past the limit */
        eshkol_error("logic variable limit exceeded (%d)", LOGIC_VAR_MAX);
        return UINT64_MAX;
    }

    g_var_names[id] = intern_var_name(name);
    return id;
}

const char* eshkol_logic_var_name(uint64_t var_id) {
    if (var_id >= g_var_count) return NULL;
    std::lock_guard<std::mutex> lock(g_var_mutex);
    return g_var_names[var_id];
}

bool eshkol_is_logic_var(const eshkol_tagged_value_t* tv) {
    if (!tv) return false;
    return tv->type == ESHKOL_VALUE_LOGIC_VAR;
}

/* ===== Arena Allocation Helpers ===== */

/* Forward declaration — defined in arena_memory.cpp */
extern "C" void* arena_allocate_with_header(arena_t* arena, size_t data_size,
                                             uint8_t subtype, uint8_t flags);

/* Forward declaration for cons cell creation */
extern "C" void* arena_allocate_aligned(arena_t* arena, size_t size, size_t alignment);

static eshkol_substitution_t* alloc_substitution(arena_t* arena, uint32_t capacity) {
    size_t data_size = sizeof(eshkol_substitution_t)
                     + capacity * sizeof(uint64_t)
                     + capacity * sizeof(eshkol_tagged_value_t);
    eshkol_substitution_t* s = (eshkol_substitution_t*)arena_allocate_with_header(
        arena, data_size, HEAP_SUBTYPE_SUBSTITUTION, 0);
    if (s) {
        s->num_bindings = 0;
        s->capacity = capacity;
    }
    return s;
}

static eshkol_fact_t* alloc_fact(arena_t* arena, uint32_t arity) {
    size_t data_size = sizeof(eshkol_fact_t)
                     + arity * sizeof(eshkol_tagged_value_t);
    eshkol_fact_t* f = (eshkol_fact_t*)arena_allocate_with_header(
        arena, data_size, HEAP_SUBTYPE_FACT, 0);
    if (f) {
        f->predicate = 0;
        f->arity = arity;
        f->_pad = 0;
    }
    return f;
}

static eshkol_knowledge_base_t* alloc_kb(arena_t* arena) {
    eshkol_knowledge_base_t* kb = (eshkol_knowledge_base_t*)arena_allocate_with_header(
        arena, sizeof(eshkol_knowledge_base_t), HEAP_SUBTYPE_KNOWLEDGE_BASE, 0);
    return kb;
}

/* ===== Substitution ===== */

eshkol_substitution_t* eshkol_make_substitution(arena_t* arena, uint32_t capacity) {
    if (!arena) return NULL;
    if (capacity == 0) capacity = 8; /* default initial capacity */
    return alloc_substitution(arena, capacity);
}

eshkol_substitution_t* eshkol_extend_subst(arena_t* arena,
    const eshkol_substitution_t* s, uint64_t var_id,
    const eshkol_tagged_value_t* term) {
    if (!arena || !term) return NULL;

    uint32_t old_bindings = s ? s->num_bindings : 0;
    uint32_t new_capacity = old_bindings + 1;
    /* Grow capacity if needed */
    if (s && new_capacity <= s->capacity) {
        new_capacity = s->capacity;
    } else {
        /* At least double */
        uint32_t min_cap = s ? s->capacity * 2 : 8;
        if (new_capacity < min_cap) new_capacity = min_cap;
    }

    eshkol_substitution_t* new_s = alloc_substitution(arena, new_capacity);
    if (!new_s) return NULL;

    /* Copy old bindings */
    if (s && old_bindings > 0) {
        const uint64_t* old_ids = SUBST_VAR_IDS(s);
        const eshkol_tagged_value_t* old_terms = SUBST_TERMS(s);
        uint64_t* new_ids = SUBST_VAR_IDS(new_s);
        eshkol_tagged_value_t* new_terms = SUBST_TERMS(new_s);
        memcpy(new_ids, old_ids, old_bindings * sizeof(uint64_t));
        memcpy(new_terms, old_terms, old_bindings * sizeof(eshkol_tagged_value_t));
    }

    /* Add new binding */
    uint64_t* ids = SUBST_VAR_IDS(new_s);
    eshkol_tagged_value_t* terms = SUBST_TERMS(new_s);
    ids[old_bindings] = var_id;
    terms[old_bindings] = *term;
    new_s->num_bindings = old_bindings + 1;

    return new_s;
}

const eshkol_tagged_value_t* eshkol_subst_lookup(
    const eshkol_substitution_t* s, uint64_t var_id) {
    if (!s) return NULL;
    const uint64_t* ids = SUBST_VAR_IDS(s);
    const eshkol_tagged_value_t* terms = SUBST_TERMS(s);
    for (uint32_t i = 0; i < s->num_bindings; i++) {
        if (ids[i] == var_id) {
            return &terms[i];
        }
    }
    return NULL;
}

/* ===== Walk ===== */

eshkol_tagged_value_t eshkol_walk(const eshkol_tagged_value_t* term,
    const eshkol_substitution_t* subst) {
    if (!term) {
        eshkol_tagged_value_t null_val;
        memset(&null_val, 0, sizeof(null_val));
        null_val.type = ESHKOL_VALUE_NULL;
        return null_val;
    }

    eshkol_tagged_value_t current = *term;

    /* Follow variable chains */
    while (current.type == ESHKOL_VALUE_LOGIC_VAR && subst) {
        uint64_t var_id = current.data.int_val;
        const eshkol_tagged_value_t* bound = eshkol_subst_lookup(subst, var_id);
        if (!bound) break; /* unbound variable */
        current = *bound;
    }

    return current;
}

static const int WALK_DEEP_MAX_DEPTH = 10000;

static eshkol_tagged_value_t walk_deep_impl(arena_t* arena,
    const eshkol_tagged_value_t* term, const eshkol_substitution_t* subst, int depth) {
    if (!term || !arena) {
        eshkol_tagged_value_t null_val;
        memset(&null_val, 0, sizeof(null_val));
        null_val.type = ESHKOL_VALUE_NULL;
        return null_val;
    }

    if (depth > WALK_DEEP_MAX_DEPTH) {
        eshkol_warn("walk_deep: depth limit exceeded (%d), returning as-is", depth);
        return *term;
    }

    eshkol_tagged_value_t walked = eshkol_walk(term, subst);

    /* If it's a fact, recursively walk its arguments */
    if (walked.type == ESHKOL_VALUE_HEAP_PTR && walked.data.ptr_val) {
        eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)walked.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_FACT) {
            eshkol_fact_t* fact = (eshkol_fact_t*)walked.data.ptr_val;
            eshkol_tagged_value_t* args = FACT_ARGS(fact);

            /* Create new fact with walked arguments */
            eshkol_fact_t* new_fact = alloc_fact(arena, fact->arity);
            if (!new_fact) return walked;
            new_fact->predicate = fact->predicate;
            eshkol_tagged_value_t* new_args = FACT_ARGS(new_fact);

            for (uint32_t i = 0; i < fact->arity; i++) {
                new_args[i] = walk_deep_impl(arena, &args[i], subst, depth + 1);
            }

            eshkol_tagged_value_t result;
            memset(&result, 0, sizeof(result));
            result.type = ESHKOL_VALUE_HEAP_PTR;
            result.data.ptr_val = (uint64_t)new_fact;
            return result;
        }
    }

    return walked;
}

eshkol_tagged_value_t eshkol_walk_deep(arena_t* arena,
    const eshkol_tagged_value_t* term, const eshkol_substitution_t* subst) {
    return walk_deep_impl(arena, term, subst, 0);
}

/* ===== Occurs Check ===== */

static const int OCCURS_CHECK_MAX_DEPTH = 1000;

static bool occurs_impl(uint64_t var_id, const eshkol_tagged_value_t* term,
                        const eshkol_substitution_t* subst, int depth) {
    if (depth > OCCURS_CHECK_MAX_DEPTH) return false; /* safety limit */

    eshkol_tagged_value_t walked = eshkol_walk(term, subst);

    if (walked.type == ESHKOL_VALUE_LOGIC_VAR) {
        return walked.data.int_val == (int64_t)var_id;
    }

    /* Check inside facts */
    if (walked.type == ESHKOL_VALUE_HEAP_PTR && walked.data.ptr_val) {
        eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)walked.data.ptr_val);
        if (header->subtype == HEAP_SUBTYPE_FACT) {
            eshkol_fact_t* fact = (eshkol_fact_t*)walked.data.ptr_val;
            eshkol_tagged_value_t* args = FACT_ARGS(fact);
            for (uint32_t i = 0; i < fact->arity; i++) {
                if (occurs_impl(var_id, &args[i], subst, depth + 1)) return true;
            }
        }
    }

    return false;
}

static bool occurs(uint64_t var_id, const eshkol_tagged_value_t* term,
                   const eshkol_substitution_t* subst) {
    return occurs_impl(var_id, term, subst, 0);
}

/* ===== Tagged Value Comparison ===== */

static bool tagged_values_equal(const eshkol_tagged_value_t* a,
                                const eshkol_tagged_value_t* b) {
    if (a->type != b->type) return false;
    switch (a->type) {
        case ESHKOL_VALUE_NULL:
            return true;
        case ESHKOL_VALUE_INT64:
            return a->data.int_val == b->data.int_val;
        case ESHKOL_VALUE_DOUBLE:
            return a->data.double_val == b->data.double_val;
        case ESHKOL_VALUE_BOOL:
            return a->data.int_val == b->data.int_val;
        case ESHKOL_VALUE_CHAR:
            return a->data.int_val == b->data.int_val;
        case ESHKOL_VALUE_LOGIC_VAR:
            return a->data.int_val == b->data.int_val;
        case ESHKOL_VALUE_HEAP_PTR:
        case ESHKOL_VALUE_CALLABLE:
            /* Pointer equality for heap objects (symbols are interned) */
            return a->data.ptr_val == b->data.ptr_val;
        default:
            return false;
    }
}

/* ===== Unification ===== */

eshkol_substitution_t* eshkol_unify(arena_t* arena,
    const eshkol_tagged_value_t* t1, const eshkol_tagged_value_t* t2,
    const eshkol_substitution_t* subst) {
    if (!arena || !t1 || !t2) return NULL;

    /* Walk both terms */
    eshkol_tagged_value_t w1 = eshkol_walk(t1, subst);
    eshkol_tagged_value_t w2 = eshkol_walk(t2, subst);

    /* If identical (including same logic variable), succeed */
    if (tagged_values_equal(&w1, &w2)) {
        /* Return the substitution as-is (cast away const for return) */
        return (eshkol_substitution_t*)subst;
    }

    /* If w1 is a logic variable, bind it */
    if (w1.type == ESHKOL_VALUE_LOGIC_VAR) {
        uint64_t var_id = (uint64_t)w1.data.int_val;
        if (occurs(var_id, &w2, subst)) return NULL; /* occurs check */
        return eshkol_extend_subst(arena, subst, var_id, &w2);
    }

    /* If w2 is a logic variable, bind it */
    if (w2.type == ESHKOL_VALUE_LOGIC_VAR) {
        uint64_t var_id = (uint64_t)w2.data.int_val;
        if (occurs(var_id, &w1, subst)) return NULL; /* occurs check */
        return eshkol_extend_subst(arena, subst, var_id, &w1);
    }

    /* Structural unification of facts */
    if (w1.type == ESHKOL_VALUE_HEAP_PTR && w2.type == ESHKOL_VALUE_HEAP_PTR &&
        w1.data.ptr_val && w2.data.ptr_val) {
        eshkol_object_header_t* h1 = ESHKOL_GET_HEADER((void*)w1.data.ptr_val);
        eshkol_object_header_t* h2 = ESHKOL_GET_HEADER((void*)w2.data.ptr_val);

        if (h1->subtype == HEAP_SUBTYPE_FACT && h2->subtype == HEAP_SUBTYPE_FACT) {
            eshkol_fact_t* f1 = (eshkol_fact_t*)w1.data.ptr_val;
            eshkol_fact_t* f2 = (eshkol_fact_t*)w2.data.ptr_val;

            /* Check predicate equality (pointer comparison for interned symbols) */
            if (f1->predicate != f2->predicate) return NULL;
            /* Check arity */
            if (f1->arity != f2->arity) return NULL;

            /* Unify each argument pair */
            eshkol_substitution_t* current = (eshkol_substitution_t*)subst;
            eshkol_tagged_value_t* args1 = FACT_ARGS(f1);
            eshkol_tagged_value_t* args2 = FACT_ARGS(f2);

            for (uint32_t i = 0; i < f1->arity; i++) {
                current = eshkol_unify(arena, &args1[i], &args2[i], current);
                if (!current) return NULL; /* unification failed */
            }

            return current;
        }
    }

    /* No other cases match -> failure */
    return NULL;
}

/* ===== Facts ===== */

eshkol_fact_t* eshkol_make_fact(arena_t* arena, uint64_t predicate,
    const eshkol_tagged_value_t* args, uint32_t arity) {
    if (!arena) return NULL;

    eshkol_fact_t* f = alloc_fact(arena, arity);
    if (!f) return NULL;

    f->predicate = predicate;
    if (args && arity > 0) {
        eshkol_tagged_value_t* fact_args = FACT_ARGS(f);
        memcpy(fact_args, args, arity * sizeof(eshkol_tagged_value_t));
    }

    return f;
}

/* ===== Knowledge Base ===== */

#define KB_INITIAL_CAPACITY 16

eshkol_knowledge_base_t* eshkol_make_kb(arena_t* arena) {
    if (!arena) return NULL;

    eshkol_knowledge_base_t* kb = alloc_kb(arena);
    if (!kb) return NULL;

    kb->num_facts = 0;
    kb->capacity = KB_INITIAL_CAPACITY;
    kb->facts = (eshkol_fact_t**)arena_allocate_aligned(
        arena, KB_INITIAL_CAPACITY * sizeof(eshkol_fact_t*), 8);

    if (!kb->facts) {
        kb->capacity = 0;
        return NULL;
    }

    return kb;
}

void eshkol_kb_assert(arena_t* arena, eshkol_knowledge_base_t* kb,
    const eshkol_fact_t* fact) {
    if (!arena || !kb || !fact) return;

    /* Grow if needed */
    if (kb->num_facts >= kb->capacity) {
        uint32_t new_capacity = kb->capacity * 2;
        eshkol_fact_t** new_facts = (eshkol_fact_t**)arena_allocate_aligned(
            arena, new_capacity * sizeof(eshkol_fact_t*), 8);
        if (!new_facts) return;
        memcpy(new_facts, kb->facts, kb->num_facts * sizeof(eshkol_fact_t*));
        kb->facts = new_facts;
        kb->capacity = new_capacity;
    }

    kb->facts[kb->num_facts++] = (eshkol_fact_t*)fact;
}

eshkol_tagged_value_t eshkol_kb_query(arena_t* arena,
    const eshkol_knowledge_base_t* kb, const eshkol_fact_t* pattern,
    const eshkol_substitution_t* initial_subst) {
    eshkol_tagged_value_t null_val;
    memset(&null_val, 0, sizeof(null_val));
    null_val.type = ESHKOL_VALUE_NULL;

    if (!arena || !kb || !pattern) return null_val;

    /* Use empty substitution if none provided */
    const eshkol_substitution_t* base_subst = initial_subst;
    eshkol_substitution_t* empty = NULL;
    if (!base_subst) {
        empty = eshkol_make_substitution(arena, 8);
        base_subst = empty;
    }

    /* Build result list in reverse, then reverse at the end */
    /* For simplicity, build a cons list as we go */
    eshkol_tagged_value_t result_list = null_val;

    for (uint32_t i = 0; i < kb->num_facts; i++) {
        const eshkol_fact_t* fact = kb->facts[i];

        /* Quick check: predicate must match */
        if (pattern->predicate != 0 && fact->predicate != 0 &&
            pattern->predicate != fact->predicate) {
            continue;
        }

        /* Check arity */
        if (pattern->arity != fact->arity) continue;

        /* Try to unify pattern arguments with fact arguments */
        eshkol_substitution_t* subst = (eshkol_substitution_t*)base_subst;
        const eshkol_tagged_value_t* pat_args = FACT_ARGS(pattern);
        const eshkol_tagged_value_t* fact_args = FACT_ARGS(fact);

        bool success = true;
        for (uint32_t j = 0; j < pattern->arity; j++) {
            subst = eshkol_unify(arena, &pat_args[j], &fact_args[j], subst);
            if (!subst) {
                success = false;
                break;
            }
        }

        if (success && subst) {
            /* Create a cons cell: (subst . result_list) */
            /* Cons cell layout: [header][car (tagged_value)][cdr (tagged_value)] */
            size_t cons_size = 2 * sizeof(eshkol_tagged_value_t);
            void* cons_data = arena_allocate_with_header(arena, cons_size, HEAP_SUBTYPE_CONS, 0);
            if (cons_data) {
                eshkol_tagged_value_t* car = (eshkol_tagged_value_t*)cons_data;
                eshkol_tagged_value_t* cdr = car + 1;

                /* Car = the successful substitution */
                memset(car, 0, sizeof(*car));
                car->type = ESHKOL_VALUE_HEAP_PTR;
                car->data.ptr_val = (uint64_t)subst;

                /* Cdr = rest of list */
                *cdr = result_list;

                /* Update result list head */
                memset(&result_list, 0, sizeof(result_list));
                result_list.type = ESHKOL_VALUE_HEAP_PTR;
                result_list.data.ptr_val = (uint64_t)cons_data;
            }
        }
    }

    return result_list;
}

/* ===== Tagged Value Dispatch ===== */

void eshkol_unify_tagged(arena_t* arena,
    const eshkol_tagged_value_t* t1, const eshkol_tagged_value_t* t2,
    const eshkol_tagged_value_t* subst_tv, eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    const eshkol_substitution_t* subst = NULL;
    if (subst_tv && subst_tv->type == ESHKOL_VALUE_HEAP_PTR && subst_tv->data.ptr_val) {
        subst = (const eshkol_substitution_t*)subst_tv->data.ptr_val;
    }

    eshkol_substitution_t* unified = eshkol_unify(arena, t1, t2, subst);
    if (unified) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)unified;
    } else {
        /* Unification failed — return NULL (which displays as #f) */
        result->type = ESHKOL_VALUE_NULL;
    }
}

void eshkol_walk_tagged(
    const eshkol_tagged_value_t* term, const eshkol_tagged_value_t* subst_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    const eshkol_substitution_t* subst = NULL;
    if (subst_tv && subst_tv->type == ESHKOL_VALUE_HEAP_PTR && subst_tv->data.ptr_val) {
        subst = (const eshkol_substitution_t*)subst_tv->data.ptr_val;
    }

    *result = eshkol_walk(term, subst);
}

void eshkol_make_fact_tagged(arena_t* arena,
    const eshkol_tagged_value_t* pred, const eshkol_tagged_value_t* args,
    int32_t arity, eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !pred) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    /* Extract predicate pointer (should be a HEAP_PTR to interned symbol) */
    uint64_t predicate = 0;
    if (pred->type == ESHKOL_VALUE_HEAP_PTR) {
        predicate = pred->data.ptr_val;
    }

    eshkol_fact_t* fact = eshkol_make_fact(arena, predicate,
        args, (uint32_t)arity);

    if (fact) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)fact;
    } else {
        result->type = ESHKOL_VALUE_NULL;
    }
}

void eshkol_make_kb_tagged(arena_t* arena, eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    eshkol_knowledge_base_t* kb = eshkol_make_kb(arena);
    if (kb) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)kb;
    } else {
        result->type = ESHKOL_VALUE_NULL;
    }
}

void eshkol_kb_assert_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb_tv, const eshkol_tagged_value_t* fact_tv) {
    if (!arena || !kb_tv || !fact_tv) return;
    if (kb_tv->type != ESHKOL_VALUE_HEAP_PTR || !kb_tv->data.ptr_val) return;
    if (fact_tv->type != ESHKOL_VALUE_HEAP_PTR || !fact_tv->data.ptr_val) return;

    eshkol_knowledge_base_t* kb = (eshkol_knowledge_base_t*)kb_tv->data.ptr_val;
    const eshkol_fact_t* fact = (const eshkol_fact_t*)fact_tv->data.ptr_val;

    eshkol_kb_assert(arena, kb, fact);
}

void eshkol_kb_query_tagged(arena_t* arena,
    const eshkol_tagged_value_t* kb_tv, const eshkol_tagged_value_t* pattern_tv,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    if (!arena || !kb_tv || !pattern_tv) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }
    if (kb_tv->type != ESHKOL_VALUE_HEAP_PTR || !kb_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }
    if (pattern_tv->type != ESHKOL_VALUE_HEAP_PTR || !pattern_tv->data.ptr_val) {
        result->type = ESHKOL_VALUE_NULL;
        return;
    }

    const eshkol_knowledge_base_t* kb = (const eshkol_knowledge_base_t*)kb_tv->data.ptr_val;
    const eshkol_fact_t* pattern = (const eshkol_fact_t*)pattern_tv->data.ptr_val;

    *result = eshkol_kb_query(arena, kb, pattern, NULL);
}

void eshkol_make_substitution_tagged(arena_t* arena,
    eshkol_tagged_value_t* result) {
    if (!result) return;
    memset(result, 0, sizeof(*result));

    eshkol_substitution_t* s = eshkol_make_substitution(arena, 8);
    if (s) {
        result->type = ESHKOL_VALUE_HEAP_PTR;
        result->data.ptr_val = (uint64_t)s;
    } else {
        result->type = ESHKOL_VALUE_NULL;
    }
}

/* ===== Display ===== */

void eshkol_display_logic_var(uint64_t var_id, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    const char* name = eshkol_logic_var_name(var_id);
    if (name) {
        fprintf(f, "%s", name);
    } else {
        fprintf(f, "?_%llu", (unsigned long long)var_id);
    }
}

/* Forward declaration for recursive display */
extern "C" void eshkol_display_value_opts(const eshkol_tagged_value_t* value,
                                           eshkol_display_opts_t* opts);

void eshkol_display_substitution(const eshkol_substitution_t* s, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    if (!s) {
        fprintf(f, "{}");
        return;
    }

    fprintf(f, "{");
    const uint64_t* ids = SUBST_VAR_IDS(s);
    const eshkol_tagged_value_t* terms = SUBST_TERMS(s);

    for (uint32_t i = 0; i < s->num_bindings; i++) {
        if (i > 0) fprintf(f, ", ");
        const char* name = eshkol_logic_var_name(ids[i]);
        if (name) {
            fprintf(f, "%s", name);
        } else {
            fprintf(f, "?_%llu", (unsigned long long)ids[i]);
        }
        fprintf(f, " -> ");

        /* Display the bound term */
        const eshkol_tagged_value_t* term = &terms[i];
        switch (term->type) {
            case ESHKOL_VALUE_INT64:
                fprintf(f, "%lld", (long long)term->data.int_val);
                break;
            case ESHKOL_VALUE_DOUBLE:
                fprintf(f, "%g", term->data.double_val);
                break;
            case ESHKOL_VALUE_BOOL:
                fprintf(f, "%s", term->data.int_val ? "#t" : "#f");
                break;
            case ESHKOL_VALUE_LOGIC_VAR:
                eshkol_display_logic_var(term->data.int_val, file);
                break;
            case ESHKOL_VALUE_NULL:
                fprintf(f, "()");
                break;
            case ESHKOL_VALUE_HEAP_PTR:
                if (term->data.ptr_val) {
                    eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)term->data.ptr_val);
                    if (header->subtype == HEAP_SUBTYPE_SYMBOL) {
                        fprintf(f, "%s", (const char*)term->data.ptr_val);
                    } else {
                        fprintf(f, "#<heap:%d>", header->subtype);
                    }
                } else {
                    fprintf(f, "()");
                }
                break;
            default:
                fprintf(f, "#<type:%d>", term->type);
                break;
        }
    }

    fprintf(f, "}");
}

void eshkol_display_fact(const eshkol_fact_t* fact, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    if (!fact) {
        fprintf(f, "(fact)");
        return;
    }

    fprintf(f, "(");

    /* Display predicate */
    if (fact->predicate) {
        fprintf(f, "%s", (const char*)fact->predicate);
    } else {
        fprintf(f, "?");
    }

    /* Display arguments */
    const eshkol_tagged_value_t* args = FACT_ARGS(fact);
    for (uint32_t i = 0; i < fact->arity; i++) {
        fprintf(f, " ");
        const eshkol_tagged_value_t* arg = &args[i];
        switch (arg->type) {
            case ESHKOL_VALUE_INT64:
                fprintf(f, "%lld", (long long)arg->data.int_val);
                break;
            case ESHKOL_VALUE_DOUBLE:
                fprintf(f, "%g", arg->data.double_val);
                break;
            case ESHKOL_VALUE_BOOL:
                fprintf(f, "%s", arg->data.int_val ? "#t" : "#f");
                break;
            case ESHKOL_VALUE_LOGIC_VAR:
                eshkol_display_logic_var(arg->data.int_val, file);
                break;
            case ESHKOL_VALUE_NULL:
                fprintf(f, "()");
                break;
            case ESHKOL_VALUE_HEAP_PTR:
                if (arg->data.ptr_val) {
                    eshkol_object_header_t* header = ESHKOL_GET_HEADER((void*)arg->data.ptr_val);
                    if (header->subtype == HEAP_SUBTYPE_SYMBOL) {
                        fprintf(f, "%s", (const char*)arg->data.ptr_val);
                    } else {
                        fprintf(f, "#<heap:%d>", header->subtype);
                    }
                } else {
                    fprintf(f, "()");
                }
                break;
            default:
                fprintf(f, "#<type:%d>", arg->type);
                break;
        }
    }

    fprintf(f, ")");
}

void eshkol_display_kb(const eshkol_knowledge_base_t* kb, void* file) {
    FILE* f = file ? (FILE*)file : stdout;
    if (!kb) {
        fprintf(f, "#<knowledge-base: empty>");
        return;
    }
    fprintf(f, "#<knowledge-base: %u facts>", kb->num_facts);
}
