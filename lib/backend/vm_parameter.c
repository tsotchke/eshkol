/**
 * @file vm_parameter.c
 * @brief Dynamic parameters (R7RS parameterize).
 *
 * Parameters are dynamic bindings with optional converter functions.
 * parameterize-push/pop save and restore values on a per-parameter
 * stack, supporting arbitrary nesting.
 *
 * Native call IDs: 700-704
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>

/* ── Parameter Object ── */

typedef struct {
    void* current_value;
    void* converter;      /* optional converter closure, NULL if none */
    void** save_stack;    /* arena-allocated stack for parameterize nesting */
    int stack_depth;
    int stack_cap;
} VmParameter;

#define PARAM_INITIAL_STACK 8

/* ── Public API ── */

/* 700: make-parameter(default, [converter]) */
VmParameter* vm_param_make(VmRegionStack* rs, void* default_val, void* converter) {
    VmParameter* p = (VmParameter*)vm_alloc_object(rs, VM_SUBTYPE_PARAMETER,
                                                    sizeof(VmParameter));
    if (!p) return NULL;
    p->current_value = default_val;
    p->converter = converter;
    p->stack_depth = 0;
    p->stack_cap = PARAM_INITIAL_STACK;
    p->save_stack = (void**)vm_alloc(rs, (size_t)p->stack_cap * sizeof(void*));
    if (!p->save_stack) return NULL;
    return p;
}

/* 701: parameter-ref(param) → current value */
void* vm_param_ref(const VmParameter* p) {
    return p ? p->current_value : NULL;
}

/* 702: parameterize-push(param, new-value) → save current, set new
 * If a converter is set, the new value would be passed through it.
 * Since we don't have VM-level closure calls here, the converter
 * is expected to be applied by the VM before calling push. */
void vm_param_push(VmRegionStack* rs, VmParameter* p, void* new_value) {
    if (!p) return;

    /* Grow stack if needed */
    if (p->stack_depth >= p->stack_cap) {
        int new_cap = p->stack_cap * 2;
        void** new_stack = (void**)vm_alloc(rs, (size_t)new_cap * sizeof(void*));
        if (!new_stack) {
            fprintf(stderr, "ERROR: parameterize stack growth failed\n");
            return;
        }
        memcpy(new_stack, p->save_stack, (size_t)p->stack_depth * sizeof(void*));
        p->save_stack = new_stack;
        p->stack_cap = new_cap;
    }

    /* Save current value */
    p->save_stack[p->stack_depth++] = p->current_value;
    p->current_value = new_value;
}

/* 703: parameterize-pop(param) → restore saved value */
void vm_param_pop(VmParameter* p) {
    if (!p || p->stack_depth <= 0) {
        fprintf(stderr, "ERROR: parameterize-pop on empty stack\n");
        return;
    }
    p->current_value = p->save_stack[--p->stack_depth];
}

/* 704: parameter? → type check */
int vm_param_is_parameter(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_PARAMETER;
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

void* vm_param_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 700: return vm_param_make(rs, nargs >= 1 ? args[0] : NULL,
                                        nargs >= 2 ? args[1] : NULL);
    case 701: return vm_param_ref((VmParameter*)args[0]);
    case 702: vm_param_push(rs, (VmParameter*)args[0], args[1]); return NULL;
    case 703: vm_param_pop((VmParameter*)args[0]); return NULL;
    case 704: { static int r; r = vm_param_is_parameter(args[0]); return &r; }
    default:
        fprintf(stderr, "ERROR: unknown parameter native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_PARAMETER_TEST

#include <assert.h>

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_parameter self-test ===\n\n");

    /* make-parameter with default */
    VmParameter* p = vm_param_make(&rs, (void*)(uintptr_t)100, NULL);
    CHECK("make returns non-null", p != NULL);
    CHECK("parameter? true", vm_param_is_parameter(p));
    CHECK("initial value is 100", (uintptr_t)vm_param_ref(p) == 100);

    /* Single push/pop */
    vm_param_push(&rs, p, (void*)(uintptr_t)200);
    CHECK("after push value is 200", (uintptr_t)vm_param_ref(p) == 200);
    vm_param_pop(p);
    CHECK("after pop value is 100", (uintptr_t)vm_param_ref(p) == 100);

    /* Nested push/pop */
    vm_param_push(&rs, p, (void*)(uintptr_t)10);
    CHECK("nest1: value is 10", (uintptr_t)vm_param_ref(p) == 10);
    vm_param_push(&rs, p, (void*)(uintptr_t)20);
    CHECK("nest2: value is 20", (uintptr_t)vm_param_ref(p) == 20);
    vm_param_push(&rs, p, (void*)(uintptr_t)30);
    CHECK("nest3: value is 30", (uintptr_t)vm_param_ref(p) == 30);
    vm_param_pop(p);
    CHECK("pop nest3: value is 20", (uintptr_t)vm_param_ref(p) == 20);
    vm_param_pop(p);
    CHECK("pop nest2: value is 10", (uintptr_t)vm_param_ref(p) == 10);
    vm_param_pop(p);
    CHECK("pop nest1: value is 100", (uintptr_t)vm_param_ref(p) == 100);

    /* Multiple parameters (independent) */
    VmParameter* q = vm_param_make(&rs, (void*)(uintptr_t)999, NULL);
    vm_param_push(&rs, p, (void*)(uintptr_t)1);
    vm_param_push(&rs, q, (void*)(uintptr_t)2);
    CHECK("p is 1", (uintptr_t)vm_param_ref(p) == 1);
    CHECK("q is 2", (uintptr_t)vm_param_ref(q) == 2);
    vm_param_pop(q);
    CHECK("q restored to 999", (uintptr_t)vm_param_ref(q) == 999);
    CHECK("p still 1", (uintptr_t)vm_param_ref(p) == 1);
    vm_param_pop(p);
    CHECK("p restored to 100", (uintptr_t)vm_param_ref(p) == 100);

    /* Stack growth: push > initial capacity */
    for (int i = 0; i < 20; i++)
        vm_param_push(&rs, p, (void*)(uintptr_t)(1000 + i));
    CHECK("after 20 pushes, value = 1019", (uintptr_t)vm_param_ref(p) == 1019);
    for (int i = 19; i >= 0; i--) {
        vm_param_pop(p);
        uintptr_t expected = (i > 0) ? (uintptr_t)(1000 + i - 1) : 100;
        if ((uintptr_t)vm_param_ref(p) != expected) {
            printf("  FAIL: pop %d expected %lu got %lu\n", i, (unsigned long)expected,
                   (unsigned long)(uintptr_t)vm_param_ref(p));
            fail++;
            break;
        }
    }
    CHECK("all 20 pops restored correctly", (uintptr_t)vm_param_ref(p) == 100);

    /* parameter? false for non-parameter (use a valid arena-allocated object) */
    void* not_param = vm_alloc_object(&rs, VM_SUBTYPE_STRING, 16);
    CHECK("parameter? false for non-param", !vm_param_is_parameter(not_param));

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_PARAMETER_TEST */
