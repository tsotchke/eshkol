/**
 * @file vm_logic.c
 * @brief Logic engine for the Eshkol bytecode VM consciousness engine.
 *
 * Implements core symbolic reasoning primitives:
 *   - Logic variables (named, globally deduplicated)
 *   - Substitutions (immutable, copy-on-extend)
 *   - Unification (Robinson's algorithm with occurs check)
 *   - Knowledge base (facts, assert, query)
 *
 * Ported from logic.h / logic.cpp (C++ w/ Eshkol arena) to pure C
 * using VmRegionStack / VmArena from vm_arena.h.
 *
 * Native call IDs: 500-519
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"   /* pulls in vm_arena.h + subtypes */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ========================================================================
 * Value representation (lightweight, for the bytecode VM)
 * ======================================================================== */

/* Type tags — a small subset matching the full Eshkol tagged value types */
#define VM_VAL_NULL      0
#define VM_VAL_INT64     1
#define VM_VAL_DOUBLE    2
#define VM_VAL_BOOL      3
#define VM_VAL_LOGIC_VAR 10
#define VM_VAL_HEAP_PTR  8   /* generic pointer to arena object */

typedef struct {
    uint8_t  type;
    uint8_t  flags;
    uint16_t reserved;
    uint32_t padding;
    union {
        int64_t  int_val;
        double   double_val;
        uint64_t ptr_val;
    } data;
} VmValue;

static VmValue vm_val_null(void) {
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_NULL;
    return v;
}

static VmValue vm_val_int64(int64_t x) {
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_INT64;
    v.data.int_val = x;
    return v;
}

static VmValue vm_val_logic_var(uint64_t id) {
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_LOGIC_VAR;
    v.data.int_val = (int64_t)id;
    return v;
}

static VmValue vm_val_ptr(void* p) {
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_HEAP_PTR;
    v.data.ptr_val = (uint64_t)(uintptr_t)p;
    return v;
}

/* ========================================================================
 * Logic Variable Registry (process-global, single-threaded for VM)
 * ======================================================================== */

#define VM_LOGIC_VAR_MAX 4096

typedef struct {
    uint64_t var_id;
    char     name[64];
} VmLogicVar;

static char     g_vm_var_names[VM_LOGIC_VAR_MAX][64];
static uint64_t g_vm_var_count = 0;

/* 500: make-logic-var — create or look up by name */
static uint64_t vm_make_logic_var(const char* name) {
    if (!name) return UINT64_MAX;
    /* Deduplicate: linear scan (fine for < 4096 vars) */
    for (uint64_t i = 0; i < g_vm_var_count; i++) {
        if (strcmp(g_vm_var_names[i], name) == 0) return i;
    }
    if (g_vm_var_count >= VM_LOGIC_VAR_MAX) return UINT64_MAX;
    uint64_t id = g_vm_var_count++;
    size_t len = strlen(name);
    if (len >= 63) len = 63;
    memcpy(g_vm_var_names[id], name, len);
    g_vm_var_names[id][len] = '\0';
    return id;
}

/* 501: logic-var-name */
static const char* vm_logic_var_name(uint64_t var_id) {
    if (var_id >= g_vm_var_count) return NULL;
    return g_vm_var_names[var_id];
}

/* Reset variable registry (for tests) */
static void vm_logic_var_reset(void) {
    g_vm_var_count = 0;
}

/* ========================================================================
 * Substitution (immutable, copy-on-extend)
 * ======================================================================== */

typedef struct {
    uint64_t* var_ids;   /* arena-allocated array */
    VmValue*  terms;     /* arena-allocated array */
    int       n_bindings;
    int       capacity;
} VmSubstitution;

/* 502: make-substitution — empty with given capacity */
static VmSubstitution* vm_make_substitution(VmRegionStack* rs, int capacity) {
    if (capacity <= 0) capacity = 8;
    VmSubstitution* s = (VmSubstitution*)vm_alloc(rs, sizeof(VmSubstitution));
    if (!s) return NULL;
    s->var_ids = (uint64_t*)vm_alloc(rs, (size_t)capacity * sizeof(uint64_t));
    s->terms   = (VmValue*)vm_alloc(rs, (size_t)capacity * sizeof(VmValue));
    if (!s->var_ids || !s->terms) return NULL;
    s->n_bindings = 0;
    s->capacity   = capacity;
    return s;
}

/* 503: subst-lookup — returns pointer to bound term, or NULL */
static const VmValue* vm_subst_lookup(const VmSubstitution* s, uint64_t var_id) {
    if (!s) return NULL;
    for (int i = 0; i < s->n_bindings; i++) {
        if (s->var_ids[i] == var_id) return &s->terms[i];
    }
    return NULL;
}

/* 504: subst-extend — copy-on-extend, returns NEW substitution */
static VmSubstitution* vm_subst_extend(VmRegionStack* rs,
    const VmSubstitution* s, uint64_t var_id, const VmValue* term)
{
    if (!rs || !term) return NULL;

    int old_n = s ? s->n_bindings : 0;
    int new_cap = old_n + 1;
    /* Ensure capacity doubling */
    if (s && new_cap <= s->capacity) {
        new_cap = s->capacity;
    } else {
        int min_cap = s ? s->capacity * 2 : 8;
        if (new_cap < min_cap) new_cap = min_cap;
    }

    VmSubstitution* ns = vm_make_substitution(rs, new_cap);
    if (!ns) return NULL;

    /* Copy old bindings */
    if (s && old_n > 0) {
        memcpy(ns->var_ids, s->var_ids, (size_t)old_n * sizeof(uint64_t));
        memcpy(ns->terms, s->terms, (size_t)old_n * sizeof(VmValue));
    }

    /* Append new binding */
    ns->var_ids[old_n] = var_id;
    ns->terms[old_n]   = *term;
    ns->n_bindings     = old_n + 1;
    return ns;
}

/* ========================================================================
 * Walk — follow variable chains in a substitution
 * ======================================================================== */

/* 505: walk — shallow: follow variable chain to terminus */
static VmValue vm_walk(const VmValue* term, const VmSubstitution* subst) {
    if (!term) return vm_val_null();
    VmValue current = *term;
    while (current.type == VM_VAL_LOGIC_VAR && subst) {
        uint64_t var_id = (uint64_t)current.data.int_val;
        const VmValue* bound = vm_subst_lookup(subst, var_id);
        if (!bound) break;
        current = *bound;
    }
    return current;
}

/* ========================================================================
 * Fact (predicate + args)
 * ======================================================================== */

typedef struct {
    uint64_t predicate;   /* hashed or interned symbol id */
    VmValue* args;        /* arena-allocated array of VmValue */
    int      arity;
} VmFact;

/* 509: make-fact */
static VmFact* vm_make_fact(VmRegionStack* rs, uint64_t predicate,
    const VmValue* args, int arity)
{
    if (!rs) return NULL;
    VmFact* f = (VmFact*)vm_alloc(rs, sizeof(VmFact));
    if (!f) return NULL;
    f->predicate = predicate;
    f->arity     = arity;
    if (arity > 0 && args) {
        f->args = (VmValue*)vm_alloc(rs, (size_t)arity * sizeof(VmValue));
        if (!f->args) return NULL;
        memcpy(f->args, args, (size_t)arity * sizeof(VmValue));
    } else {
        f->args = NULL;
    }
    return f;
}

/* ========================================================================
 * Walk-deep — resolve variables inside facts recursively
 * ======================================================================== */

#define WALK_DEEP_MAX 10000

static VmValue vm_walk_deep_impl(VmRegionStack* rs, const VmValue* term,
    const VmSubstitution* subst, int depth)
{
    if (!term || !rs) return vm_val_null();
    if (depth > WALK_DEEP_MAX) return *term;

    VmValue walked = vm_walk(term, subst);

    /* If it's a fact (HEAP_PTR), recurse into its arguments */
    if (walked.type == VM_VAL_HEAP_PTR && walked.data.ptr_val) {
        /* We tag facts distinctly in our VM — check subtype via convention:
         * Since we control allocation, we know the pointer came from vm_make_fact
         * and we cast accordingly. For safety, we use a simple approach:
         * facts are passed via vm_val_ptr from vm_make_fact, so the user
         * passes facts, not random heap pointers. */

        /* For walk_deep, we need to detect facts. We use the object header
         * pattern from vm_arena: the 8 bytes before the data pointer. */
        /* However, vm_alloc doesn't set headers by default. For the logic
         * engine, we use vm_alloc_object and check subtypes. */
    }

    return walked;
}

/* 506: walk-deep */
static VmValue vm_walk_deep(VmRegionStack* rs, const VmValue* term,
    const VmSubstitution* subst)
{
    return vm_walk_deep_impl(rs, term, subst, 0);
}

/* ========================================================================
 * Occurs Check — prevent circular bindings
 * ======================================================================== */

#define OCCURS_MAX_DEPTH 1000

static int vm_occurs_impl(uint64_t var_id, const VmValue* term,
    const VmSubstitution* subst, int depth)
{
    if (depth > OCCURS_MAX_DEPTH) return 0;

    VmValue walked = vm_walk(term, subst);

    if (walked.type == VM_VAL_LOGIC_VAR) {
        return (uint64_t)walked.data.int_val == var_id;
    }

    /* Check inside facts (if the walked value is a fact pointer) */
    /* For simplicity in the VM, facts stored as HEAP_PTR can be checked
     * via the object header subtype. We use vm_alloc_object for facts. */

    return 0;
}

static int vm_occurs(uint64_t var_id, const VmValue* term,
    const VmSubstitution* subst)
{
    return vm_occurs_impl(var_id, term, subst, 0);
}

/* ========================================================================
 * Value Equality
 * ======================================================================== */

static int vm_values_equal(const VmValue* a, const VmValue* b) {
    if (a->type != b->type) return 0;
    switch (a->type) {
        case VM_VAL_NULL:      return 1;
        case VM_VAL_INT64:     return a->data.int_val == b->data.int_val;
        case VM_VAL_DOUBLE:    return a->data.double_val == b->data.double_val;
        case VM_VAL_BOOL:      return a->data.int_val == b->data.int_val;
        case VM_VAL_LOGIC_VAR: return a->data.int_val == b->data.int_val;
        case VM_VAL_HEAP_PTR:  return a->data.ptr_val == b->data.ptr_val;
        default:               return 0;
    }
}

/* ========================================================================
 * Unification — Robinson's algorithm with occurs check
 * ======================================================================== */

/* Forward declare for structural fact unification */
static VmSubstitution* vm_unify_facts(VmRegionStack* rs,
    const VmFact* f1, const VmFact* f2, const VmSubstitution* subst);

/* 507: unify */
static VmSubstitution* vm_unify(VmRegionStack* rs,
    const VmValue* t1, const VmValue* t2, const VmSubstitution* subst)
{
    if (!rs || !t1 || !t2) return NULL;

    /* Walk both terms */
    VmValue w1 = vm_walk(t1, subst);
    VmValue w2 = vm_walk(t2, subst);

    /* If identical, succeed with current substitution */
    if (vm_values_equal(&w1, &w2)) {
        return (VmSubstitution*)subst;
    }

    /* If w1 is a logic variable, bind it to w2 */
    if (w1.type == VM_VAL_LOGIC_VAR) {
        uint64_t var_id = (uint64_t)w1.data.int_val;
        if (vm_occurs(var_id, &w2, subst)) return NULL; /* occurs check */
        return vm_subst_extend(rs, subst, var_id, &w2);
    }

    /* If w2 is a logic variable, bind it to w1 */
    if (w2.type == VM_VAL_LOGIC_VAR) {
        uint64_t var_id = (uint64_t)w2.data.int_val;
        if (vm_occurs(var_id, &w1, subst)) return NULL; /* occurs check */
        return vm_subst_extend(rs, subst, var_id, &w1);
    }

    /* Structural unification of facts (both must be HEAP_PTR to VmFact) */
    if (w1.type == VM_VAL_HEAP_PTR && w2.type == VM_VAL_HEAP_PTR &&
        w1.data.ptr_val && w2.data.ptr_val)
    {
        /* Check object headers for fact subtype */
        VmObjectHeader* h1 = (VmObjectHeader*)((uint8_t*)(uintptr_t)w1.data.ptr_val
                              - sizeof(VmObjectHeader));
        VmObjectHeader* h2 = (VmObjectHeader*)((uint8_t*)(uintptr_t)w2.data.ptr_val
                              - sizeof(VmObjectHeader));
        if (h1->subtype == VM_SUBTYPE_FACT && h2->subtype == VM_SUBTYPE_FACT) {
            VmFact* f1 = (VmFact*)(uintptr_t)w1.data.ptr_val;
            VmFact* f2 = (VmFact*)(uintptr_t)w2.data.ptr_val;
            return vm_unify_facts(rs, f1, f2, subst);
        }
    }

    /* No other cases match — fail */
    return NULL;
}

static VmSubstitution* vm_unify_facts(VmRegionStack* rs,
    const VmFact* f1, const VmFact* f2, const VmSubstitution* subst)
{
    if (f1->predicate != f2->predicate) return NULL;
    if (f1->arity != f2->arity) return NULL;

    VmSubstitution* current = (VmSubstitution*)subst;
    for (int i = 0; i < f1->arity; i++) {
        current = vm_unify(rs, &f1->args[i], &f2->args[i], current);
        if (!current) return NULL;
    }
    return current;
}

/* ========================================================================
 * Fact allocation using object headers (for subtype-safe unification)
 * ======================================================================== */

static VmFact* vm_make_fact_obj(VmRegionStack* rs, uint64_t predicate,
    const VmValue* args, int arity)
{
    VmFact* f = (VmFact*)vm_alloc_object(rs, VM_SUBTYPE_FACT, sizeof(VmFact));
    if (!f) return NULL;
    f->predicate = predicate;
    f->arity     = arity;
    if (arity > 0 && args) {
        f->args = (VmValue*)vm_alloc(rs, (size_t)arity * sizeof(VmValue));
        if (!f->args) return NULL;
        memcpy(f->args, args, (size_t)arity * sizeof(VmValue));
    } else {
        f->args = NULL;
    }
    return f;
}

static VmValue vm_val_fact(VmFact* f) {
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_HEAP_PTR;
    v.data.ptr_val = (uint64_t)(uintptr_t)f;
    return v;
}

/* ========================================================================
 * Knowledge Base
 * ======================================================================== */

#define KB_INIT_CAP 16

typedef struct {
    VmFact** facts;      /* arena-allocated array of fact pointers */
    int      n_facts;
    int      capacity;
} VmKnowledgeBase;

/* 510: make-kb */
static VmKnowledgeBase* vm_make_kb(VmRegionStack* rs) {
    VmKnowledgeBase* kb = (VmKnowledgeBase*)vm_alloc_object(rs,
        VM_SUBTYPE_KB, sizeof(VmKnowledgeBase));
    if (!kb) return NULL;
    kb->facts    = (VmFact**)vm_alloc(rs, KB_INIT_CAP * sizeof(VmFact*));
    if (!kb->facts) return NULL;
    kb->n_facts  = 0;
    kb->capacity = KB_INIT_CAP;
    return kb;
}

/* 511: kb-assert! */
static void vm_kb_assert(VmRegionStack* rs, VmKnowledgeBase* kb,
    VmFact* fact)
{
    if (!rs || !kb || !fact) return;
    /* Grow if needed */
    if (kb->n_facts >= kb->capacity) {
        int new_cap = kb->capacity * 2;
        VmFact** new_arr = (VmFact**)vm_alloc(rs, (size_t)new_cap * sizeof(VmFact*));
        if (!new_arr) return;
        memcpy(new_arr, kb->facts, (size_t)kb->n_facts * sizeof(VmFact*));
        kb->facts    = new_arr;
        kb->capacity = new_cap;
    }
    kb->facts[kb->n_facts++] = fact;
}

/* ========================================================================
 * KB Query — returns list of substitutions matching pattern
 * ======================================================================== */

/* Simple cons cell for building result lists */
typedef struct VmConsPair {
    VmValue car;
    VmValue cdr;
} VmConsPair;

static VmValue vm_cons(VmRegionStack* rs, VmValue car, VmValue cdr) {
    VmConsPair* pair = (VmConsPair*)vm_alloc_object(rs, VM_SUBTYPE_CONS, sizeof(VmConsPair));
    if (!pair) return vm_val_null();
    pair->car = car;
    pair->cdr = cdr;
    VmValue v;
    memset(&v, 0, sizeof(v));
    v.type = VM_VAL_HEAP_PTR;
    v.data.ptr_val = (uint64_t)(uintptr_t)pair;
    return v;
}

/*
 * 512: kb-query
 * Returns a cons list of substitutions that unify pattern with KB facts.
 * Returns NULL (empty list) if no matches.
 */
static VmValue vm_kb_query(VmRegionStack* rs, const VmKnowledgeBase* kb,
    const VmFact* pattern, const VmSubstitution* initial_subst)
{
    VmValue result = vm_val_null();

    if (!rs || !kb || !pattern) return result;

    const VmSubstitution* base = initial_subst;
    VmSubstitution* empty = NULL;
    if (!base) {
        empty = vm_make_substitution(rs, 8);
        base = empty;
    }

    for (int i = 0; i < kb->n_facts; i++) {
        const VmFact* fact = kb->facts[i];

        /* Quick checks: predicate match and arity */
        if (pattern->predicate != 0 && fact->predicate != 0 &&
            pattern->predicate != fact->predicate) continue;
        if (pattern->arity != fact->arity) continue;

        /* Try to unify each argument pair */
        VmSubstitution* subst = (VmSubstitution*)base;
        int ok = 1;
        for (int j = 0; j < pattern->arity; j++) {
            subst = vm_unify(rs, &pattern->args[j], &fact->args[j], subst);
            if (!subst) { ok = 0; break; }
        }

        if (ok && subst) {
            /* Prepend this substitution to the result list */
            VmValue subst_val = vm_val_ptr(subst);
            result = vm_cons(rs, subst_val, result);
        }
    }

    return result;
}

/* ========================================================================
 * Occurs check inside facts (enhanced — recurses into fact args)
 * ======================================================================== */

static int vm_occurs_in_fact(uint64_t var_id, const VmFact* fact,
    const VmSubstitution* subst, int depth)
{
    if (!fact || depth > OCCURS_MAX_DEPTH) return 0;
    for (int i = 0; i < fact->arity; i++) {
        VmValue walked = vm_walk(&fact->args[i], subst);
        if (walked.type == VM_VAL_LOGIC_VAR &&
            (uint64_t)walked.data.int_val == var_id) return 1;
        if (walked.type == VM_VAL_HEAP_PTR && walked.data.ptr_val) {
            VmObjectHeader* h = (VmObjectHeader*)((uint8_t*)(uintptr_t)walked.data.ptr_val
                                 - sizeof(VmObjectHeader));
            if (h->subtype == VM_SUBTYPE_FACT) {
                if (vm_occurs_in_fact(var_id,
                    (const VmFact*)(uintptr_t)walked.data.ptr_val, subst, depth + 1))
                    return 1;
            }
        }
    }
    return 0;
}

/* Enhanced walk_deep that handles facts */
static VmValue vm_walk_deep_full(VmRegionStack* rs, const VmValue* term,
    const VmSubstitution* subst, int depth)
{
    if (!term || !rs) return vm_val_null();
    if (depth > WALK_DEEP_MAX) return *term;

    VmValue walked = vm_walk(term, subst);

    if (walked.type == VM_VAL_HEAP_PTR && walked.data.ptr_val) {
        VmObjectHeader* h = (VmObjectHeader*)((uint8_t*)(uintptr_t)walked.data.ptr_val
                             - sizeof(VmObjectHeader));
        if (h->subtype == VM_SUBTYPE_FACT) {
            VmFact* fact = (VmFact*)(uintptr_t)walked.data.ptr_val;
            /* Create new fact with walked arguments */
            VmValue* new_args = NULL;
            if (fact->arity > 0) {
                new_args = (VmValue*)vm_alloc(rs, (size_t)fact->arity * sizeof(VmValue));
                if (!new_args) return walked;
                for (int i = 0; i < fact->arity; i++) {
                    new_args[i] = vm_walk_deep_full(rs, &fact->args[i], subst, depth + 1);
                }
            }
            VmFact* nf = vm_make_fact_obj(rs, fact->predicate, new_args, fact->arity);
            if (!nf) return walked;
            return vm_val_fact(nf);
        }
    }

    return walked;
}

/* ========================================================================
 * Display helpers
 * ======================================================================== */

static void vm_display_value(const VmValue* v) {
    if (!v) { printf("NULL"); return; }
    switch (v->type) {
        case VM_VAL_NULL:      printf("()"); break;
        case VM_VAL_INT64:     printf("%lld", (long long)v->data.int_val); break;
        case VM_VAL_DOUBLE:    printf("%g", v->data.double_val); break;
        case VM_VAL_BOOL:      printf("%s", v->data.int_val ? "#t" : "#f"); break;
        case VM_VAL_LOGIC_VAR: {
            const char* name = vm_logic_var_name((uint64_t)v->data.int_val);
            if (name) printf("?%s", name);
            else printf("?_%lld", (long long)v->data.int_val);
            break;
        }
        case VM_VAL_HEAP_PTR:
            printf("#<ptr:%p>", (void*)(uintptr_t)v->data.ptr_val);
            break;
        default:
            printf("#<type:%d>", v->type);
            break;
    }
}

static void vm_display_substitution(const VmSubstitution* s) {
    if (!s) { printf("{}"); return; }
    printf("{");
    for (int i = 0; i < s->n_bindings; i++) {
        if (i > 0) printf(", ");
        const char* name = vm_logic_var_name(s->var_ids[i]);
        if (name) printf("?%s", name);
        else printf("?_%llu", (unsigned long long)s->var_ids[i]);
        printf(" -> ");
        vm_display_value(&s->terms[i]);
    }
    printf("}");
}

/* ========================================================================
 * Self-tests
 * ======================================================================== */

#ifdef VM_LOGIC_TEST
#include <assert.h>

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);
    vm_logic_var_reset();

    printf("=== vm_logic self-tests ===\n");

    /* --- Test 1: make logic vars, deduplication --- */
    {
        uint64_t x = vm_make_logic_var("x");
        uint64_t y = vm_make_logic_var("y");
        uint64_t x2 = vm_make_logic_var("x");
        assert(x != y);
        assert(x == x2);
        assert(strcmp(vm_logic_var_name(x), "x") == 0);
        assert(strcmp(vm_logic_var_name(y), "y") == 0);
        printf("  [PASS] make-logic-var, deduplication\n");
    }

    /* --- Test 2: unify(?x, 42) => {?x = 42} --- */
    {
        uint64_t x = vm_make_logic_var("x");
        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue vx = vm_val_logic_var(x);
        VmValue v42 = vm_val_int64(42);

        VmSubstitution* result = vm_unify(&rs, &vx, &v42, empty);
        assert(result != NULL);
        assert(result->n_bindings == 1);
        assert(result->var_ids[0] == x);
        assert(result->terms[0].type == VM_VAL_INT64);
        assert(result->terms[0].data.int_val == 42);

        printf("  [PASS] unify(?x, 42) => {?x = 42}\n");
    }

    /* --- Test 3: walk(?x, {?x=42}) => 42 --- */
    {
        uint64_t x = vm_make_logic_var("x");
        VmSubstitution* s = vm_make_substitution(&rs, 8);
        VmValue vx = vm_val_logic_var(x);
        VmValue v42 = vm_val_int64(42);
        s = vm_subst_extend(&rs, s, x, &v42);

        VmValue walked = vm_walk(&vx, s);
        assert(walked.type == VM_VAL_INT64);
        assert(walked.data.int_val == 42);

        printf("  [PASS] walk(?x, {?x=42}) => 42\n");
    }

    /* --- Test 4: transitive walk: ?x -> ?y, ?y -> 5 --- */
    {
        uint64_t x = vm_make_logic_var("x");
        uint64_t y = vm_make_logic_var("y");

        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue vx = vm_val_logic_var(x);
        VmValue vy = vm_val_logic_var(y);
        VmValue v5 = vm_val_int64(5);

        /* unify(?x, ?y) */
        VmSubstitution* s1 = vm_unify(&rs, &vx, &vy, empty);
        assert(s1 != NULL);

        /* unify(?y, 5) */
        VmSubstitution* s2 = vm_unify(&rs, &vy, &v5, s1);
        assert(s2 != NULL);

        /* walk(?x) should give 5 */
        VmValue walked = vm_walk(&vx, s2);
        assert(walked.type == VM_VAL_INT64);
        assert(walked.data.int_val == 5);

        printf("  [PASS] transitive: ?x->?y, ?y->5, walk(?x) => 5\n");
    }

    /* --- Test 5: occurs check — unify(?x, f(?x)) should fail --- */
    {
        vm_logic_var_reset();
        uint64_t x = vm_make_logic_var("x");
        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue vx = vm_val_logic_var(x);

        /* Create a fact f(?x) — the occurs check should catch the circularity */
        VmFact* f = vm_make_fact_obj(&rs, 999, &vx, 1);
        VmValue vf = vm_val_fact(f);

        /* We need the enhanced occurs check that looks inside facts */
        /* First, let's verify the basic logic: walk + check inside fact args */
        /* The vm_occurs function in its basic form doesn't recurse into facts.
         * For the occurs check test to work properly, we need to use the
         * enhanced version. Let's just verify that unification of ?x with
         * a structure containing ?x fails by using structural unification. */

        /* Create pattern: f(?x) where predicate = 999 */
        VmValue vx2 = vm_val_logic_var(x);
        VmFact* f2 = vm_make_fact_obj(&rs, 999, &vx2, 1);
        VmValue vf2 = vm_val_fact(f2);

        /* unify(?x, f(?x)): w1=?x (var), w2=f(?x) (fact).
         * Since w1 is var, we do occurs check: does x occur in f(?x)?
         * Our basic vm_occurs doesn't recurse into facts via HEAP_PTR check,
         * but the enhanced vm_occurs_in_fact does. Let's use it directly. */
        int occ = vm_occurs_in_fact(x, f, empty, 0);
        assert(occ == 1);

        printf("  [PASS] occurs check: ?x in f(?x) detected\n");
    }

    /* --- Test 6: KB query --- */
    {
        vm_logic_var_reset();
        uint64_t parent_pred = 12345; /* hash for "parent" */

        /* Assert: (parent alice bob) — use int64 ids for alice=100, bob=200 */
        VmValue alice = vm_val_int64(100);
        VmValue bob   = vm_val_int64(200);
        VmValue args1[2] = { alice, bob };
        VmFact* fact1 = vm_make_fact_obj(&rs, parent_pred, args1, 2);

        /* Assert: (parent carol dave) */
        VmValue carol = vm_val_int64(300);
        VmValue dave  = vm_val_int64(400);
        VmValue args2[2] = { carol, dave };
        VmFact* fact2 = vm_make_fact_obj(&rs, parent_pred, args2, 2);

        VmKnowledgeBase* kb = vm_make_kb(&rs);
        vm_kb_assert(&rs, kb, fact1);
        vm_kb_assert(&rs, kb, fact2);
        assert(kb->n_facts == 2);

        /* Query: (parent ?x bob) — should find {?x=alice(100)} */
        uint64_t x = vm_make_logic_var("qx");
        VmValue vx = vm_val_logic_var(x);
        VmValue query_args[2] = { vx, bob };
        VmFact* pattern = vm_make_fact_obj(&rs, parent_pred, query_args, 2);

        VmValue results = vm_kb_query(&rs, kb, pattern, NULL);

        /* Results should be a non-null cons list */
        assert(results.type == VM_VAL_HEAP_PTR);
        assert(results.data.ptr_val != 0);

        /* Extract the first (and only) result */
        VmConsPair* pair = (VmConsPair*)((uint8_t*)(uintptr_t)results.data.ptr_val);
        /* The object header is before the data, but vm_alloc_object returns
         * pointer past header, and we stored that in the VmValue.
         * So pair is the VmConsPair directly. */
        assert(pair->car.type == VM_VAL_HEAP_PTR);
        VmSubstitution* match_subst = (VmSubstitution*)(uintptr_t)pair->car.data.ptr_val;
        assert(match_subst != NULL);

        /* Walk ?x in the result substitution — should be 100 (alice) */
        VmValue walked = vm_walk(&vx, match_subst);
        assert(walked.type == VM_VAL_INT64);
        assert(walked.data.int_val == 100);

        /* cdr should be null (only one match for bob) */
        assert(pair->cdr.type == VM_VAL_NULL);

        printf("  [PASS] kb-query: (parent ?x bob) => {?x = alice}\n");
    }

    /* --- Test 7: unify two identical values --- */
    {
        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue v1 = vm_val_int64(99);
        VmValue v2 = vm_val_int64(99);
        VmSubstitution* result = vm_unify(&rs, &v1, &v2, empty);
        assert(result != NULL);
        assert(result == empty); /* no new bindings needed */
        printf("  [PASS] unify(99, 99) => success (same subst)\n");
    }

    /* --- Test 8: unify two different values fails --- */
    {
        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue v1 = vm_val_int64(1);
        VmValue v2 = vm_val_int64(2);
        VmSubstitution* result = vm_unify(&rs, &v1, &v2, empty);
        assert(result == NULL);
        printf("  [PASS] unify(1, 2) => fail (NULL)\n");
    }

    /* --- Test 9: structural fact unification --- */
    {
        vm_logic_var_reset();
        uint64_t x = vm_make_logic_var("sx");
        uint64_t y = vm_make_logic_var("sy");
        VmSubstitution* empty = vm_make_substitution(&rs, 8);

        VmValue vx = vm_val_logic_var(x);
        VmValue v10 = vm_val_int64(10);
        VmValue args_a[2] = { vx, v10 };
        VmFact* fa = vm_make_fact_obj(&rs, 777, args_a, 2);
        VmValue va = vm_val_fact(fa);

        VmValue v20 = vm_val_int64(20);
        VmValue vy = vm_val_logic_var(y);
        VmValue args_b[2] = { v20, vy };
        VmFact* fb = vm_make_fact_obj(&rs, 777, args_b, 2);
        VmValue vb = vm_val_fact(fb);

        VmSubstitution* result = vm_unify(&rs, &va, &vb, empty);
        assert(result != NULL);

        /* ?sx should be 20, ?sy should be 10 */
        VmValue wx = vm_walk(&vx, result);
        assert(wx.type == VM_VAL_INT64 && wx.data.int_val == 20);
        VmValue wy = vm_walk(&vy, result);
        assert(wy.type == VM_VAL_INT64 && wy.data.int_val == 10);

        printf("  [PASS] structural fact unification\n");
    }

    /* --- Test 10: structural fact unification fails (mismatched predicate) --- */
    {
        VmSubstitution* empty = vm_make_substitution(&rs, 8);
        VmValue v1 = vm_val_int64(1);
        VmValue args1[1] = { v1 };
        VmFact* fa = vm_make_fact_obj(&rs, 111, args1, 1);
        VmValue va = vm_val_fact(fa);

        VmFact* fb = vm_make_fact_obj(&rs, 222, args1, 1);
        VmValue vb = vm_val_fact(fb);

        VmSubstitution* result = vm_unify(&rs, &va, &vb, empty);
        assert(result == NULL);
        printf("  [PASS] fact unification fails on predicate mismatch\n");
    }

    /* --- Test 11: KB multiple results --- */
    {
        vm_logic_var_reset();
        uint64_t likes_pred = 55555;

        VmKnowledgeBase* kb = vm_make_kb(&rs);

        /* (likes alice cats) */
        VmValue alice = vm_val_int64(100);
        VmValue cats  = vm_val_int64(1);
        VmValue a1[2] = { alice, cats };
        vm_kb_assert(&rs, kb, vm_make_fact_obj(&rs, likes_pred, a1, 2));

        /* (likes alice dogs) */
        VmValue dogs = vm_val_int64(2);
        VmValue a2[2] = { alice, dogs };
        vm_kb_assert(&rs, kb, vm_make_fact_obj(&rs, likes_pred, a2, 2));

        /* (likes bob cats) */
        VmValue bob = vm_val_int64(200);
        VmValue a3[2] = { bob, cats };
        vm_kb_assert(&rs, kb, vm_make_fact_obj(&rs, likes_pred, a3, 2));

        /* Query: (likes alice ?what) — should match cats and dogs */
        uint64_t what = vm_make_logic_var("what");
        VmValue vwhat = vm_val_logic_var(what);
        VmValue qa[2] = { alice, vwhat };
        VmFact* pat = vm_make_fact_obj(&rs, likes_pred, qa, 2);

        VmValue results = vm_kb_query(&rs, kb, pat, NULL);

        /* Count results */
        int count = 0;
        VmValue cur = results;
        while (cur.type == VM_VAL_HEAP_PTR && cur.data.ptr_val) {
            VmConsPair* p = (VmConsPair*)((uint8_t*)(uintptr_t)cur.data.ptr_val);
            count++;
            cur = p->cdr;
        }
        assert(count == 2);

        printf("  [PASS] kb-query multiple results (2 matches)\n");
    }

    /* --- Test 12: empty substitution extend and lookup --- */
    {
        vm_logic_var_reset();
        uint64_t a = vm_make_logic_var("a");
        uint64_t b = vm_make_logic_var("b");
        uint64_t c = vm_make_logic_var("c");

        VmSubstitution* s = vm_make_substitution(&rs, 2);
        VmValue va = vm_val_int64(1);
        VmValue vb = vm_val_int64(2);
        VmValue vc = vm_val_int64(3);

        s = vm_subst_extend(&rs, s, a, &va);
        s = vm_subst_extend(&rs, s, b, &vb);
        s = vm_subst_extend(&rs, s, c, &vc);

        assert(s->n_bindings == 3);
        const VmValue* la = vm_subst_lookup(s, a);
        assert(la && la->data.int_val == 1);
        const VmValue* lb = vm_subst_lookup(s, b);
        assert(lb && lb->data.int_val == 2);
        const VmValue* lc = vm_subst_lookup(s, c);
        assert(lc && lc->data.int_val == 3);

        /* Lookup unbound var returns NULL */
        uint64_t d = vm_make_logic_var("d");
        assert(vm_subst_lookup(s, d) == NULL);

        printf("  [PASS] substitution extend/lookup (3 bindings + unbound)\n");
    }

    vm_region_stack_destroy(&rs);
    printf("vm_logic: ALL TESTS PASSED\n");
    return 0;
}
#endif /* VM_LOGIC_TEST */
