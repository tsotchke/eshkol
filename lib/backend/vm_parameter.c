/**
 * @file vm_parameter.c
 * @brief Dynamic parameters (R7RS parameterize).
 *
 * Parameters are dynamic bindings with optional converter functions.
 * parameterize-push/pop save and restore values on a per-parameter
 * stack, supporting arbitrary nesting.
 *
 * Native call IDs: 700-704 (the interpreter-only converter invocation is
 * native ID 705 and lives in vm_native.c because it must call closures).
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>

/* ── Parameter Object ── */

/* This has the exact memory layout of vm_core.c's Value without depending on
 * the unity include order.  The pointer APIs copy it to/from Value objects. */
typedef struct {
    int type;
    union {
        int64_t i;
        double f;
        int b;
        int32_t ptr;
    } as;
} VmParameterValue;

typedef struct {
    VmParameterValue current_value;
    VmParameterValue converter;  /* VAL_NIL when absent */
    VmParameterValue* save_stack;
    int stack_depth;
    int stack_cap;
} VmParameter;

#define PARAM_INITIAL_STACK 8

/* ── Public API ── */

/** @brief Native call 700: `(make-parameter default [converter])` —
 *         allocate a dynamic parameter object with its save/restore stack
 *         pre-sized to PARAM_INITIAL_STACK. */
VmParameter* vm_param_make(VmRegionStack* rs, const void* default_val,
                           const void* converter) {
    VmParameter* p = (VmParameter*)vm_alloc_object(rs, VM_SUBTYPE_PARAMETER,
                                                    sizeof(VmParameter));
    if (!p) return NULL;
    memset(&p->current_value, 0, sizeof(p->current_value));
    memset(&p->converter, 0, sizeof(p->converter));
    if (default_val) memcpy(&p->current_value, default_val, sizeof(p->current_value));
    if (converter) memcpy(&p->converter, converter, sizeof(p->converter));
    p->stack_depth = 0;
    p->stack_cap = PARAM_INITIAL_STACK;
    p->save_stack = (VmParameterValue*)vm_alloc(
        rs, (size_t)p->stack_cap * sizeof(VmParameterValue));
    if (!p->save_stack) return NULL;
    return p;
}

/** @brief Native call 701: copy the current parameter value to @p out. */
void vm_param_ref(const VmParameter* p, void* out) {
    if (!out) return;
    if (p) memcpy(out, &p->current_value, sizeof(p->current_value));
    else memset(out, 0, sizeof(VmParameterValue));
}

/** Copy the optional converter to @p out. */
void vm_param_converter_ref(const VmParameter* p, void* out) {
    if (!out) return;
    if (p) memcpy(out, &p->converter, sizeof(p->converter));
    else memset(out, 0, sizeof(VmParameterValue));
}

/**
 * @brief Native call 702: `parameterize` entry — push @p p's current value
 *        onto its save stack (growing it if full) and set @p new_value as
 *        the current value. If @p p has a converter, the VM is expected to
 *        apply it to @p new_value before calling this (this module has no
 *        VM-level closure-call capability).
 */
void vm_param_push(VmRegionStack* rs, VmParameter* p, const void* new_value) {
    if (!p) return;

    /* Grow stack if needed */
    if (p->stack_depth >= p->stack_cap) {
        int new_cap = p->stack_cap * 2;
        VmParameterValue* new_stack = (VmParameterValue*)vm_alloc(
            rs, (size_t)new_cap * sizeof(VmParameterValue));
        if (!new_stack) {
            fprintf(stderr, "ERROR: parameterize stack growth failed\n");
            return;
        }
        memcpy(new_stack, p->save_stack,
               (size_t)p->stack_depth * sizeof(VmParameterValue));
        p->save_stack = new_stack;
        p->stack_cap = new_cap;
    }

    /* Save current value */
    p->save_stack[p->stack_depth++] = p->current_value;
    if (new_value) memcpy(&p->current_value, new_value, sizeof(p->current_value));
}

/** @brief Native call 703: `parameterize` exit — restore @p p's value from
 *         the top of its save stack (error if the stack is empty). */
void vm_param_pop(VmParameter* p) {
    if (!p || p->stack_depth <= 0) {
        fprintf(stderr, "ERROR: parameterize-pop on empty stack\n");
        return;
    }
    p->current_value = p->save_stack[--p->stack_depth];
}

/** Replace the current binding without changing dynamic nesting depth. */
void vm_param_set(VmParameter* p, const void* new_value) {
    if (!p || !new_value) return;
    memcpy(&p->current_value, new_value, sizeof(p->current_value));
}

/** @brief Native call 704: `(parameter? obj)` — check the heap object
 *         header subtype. */
int vm_param_is_parameter(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_PARAMETER;
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_PARAMETER_TEST

#include <assert.h>

/** @brief Standalone self-test (built when VM_PARAMETER_TEST is defined):
 *         exercises make/ref/push/pop, nested and multi-parameter
 *         independence, save-stack growth beyond the initial capacity, and
 *         the parameter? type check. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_parameter self-test ===\n\n");

    /* This source is also built independently of vm_core.c, so use the
     * layout-compatible transport type instead of the VM's Value typedef. */
    VmParameterValue value = {.type = 1, .as.i = 100}; /* VAL_INT */
    VmParameterValue result;

    /* make-parameter with default */
    VmParameter* p = vm_param_make(&rs, &value, NULL);
    CHECK("make returns non-null", p != NULL);
    CHECK("parameter? true", vm_param_is_parameter(p));
    vm_param_ref(p, &result);
    CHECK("initial value is 100", result.type == 1 && result.as.i == 100);

    /* Single push/pop */
    value.as.i = 200; vm_param_push(&rs, p, &value);
    vm_param_ref(p, &result);
    CHECK("after push value is 200", result.as.i == 200);
    vm_param_pop(p);
    vm_param_ref(p, &result);
    CHECK("after pop value is 100", result.as.i == 100);

    /* Nested push/pop */
    value.as.i = 10; vm_param_push(&rs, p, &value); vm_param_ref(p, &result);
    CHECK("nest1: value is 10", result.as.i == 10);
    value.as.i = 20; vm_param_push(&rs, p, &value); vm_param_ref(p, &result);
    CHECK("nest2: value is 20", result.as.i == 20);
    value.as.i = 30; vm_param_push(&rs, p, &value); vm_param_ref(p, &result);
    CHECK("nest3: value is 30", result.as.i == 30);
    vm_param_pop(p);
    vm_param_ref(p, &result); CHECK("pop nest3: value is 20", result.as.i == 20);
    vm_param_pop(p);
    vm_param_ref(p, &result); CHECK("pop nest2: value is 10", result.as.i == 10);
    vm_param_pop(p);
    vm_param_ref(p, &result); CHECK("pop nest1: value is 100", result.as.i == 100);

    /* Multiple parameters (independent) */
    value.as.i = 999; VmParameter* q = vm_param_make(&rs, &value, NULL);
    value.as.i = 1; vm_param_push(&rs, p, &value);
    value.as.i = 2; vm_param_push(&rs, q, &value);
    vm_param_ref(p, &result); CHECK("p is 1", result.as.i == 1);
    vm_param_ref(q, &result); CHECK("q is 2", result.as.i == 2);
    vm_param_pop(q);
    vm_param_ref(q, &result); CHECK("q restored to 999", result.as.i == 999);
    vm_param_ref(p, &result); CHECK("p still 1", result.as.i == 1);
    vm_param_pop(p);
    vm_param_ref(p, &result); CHECK("p restored to 100", result.as.i == 100);

    /* Stack growth: push > initial capacity */
    for (int i = 0; i < 20; i++) { value.as.i = 1000 + i; vm_param_push(&rs, p, &value); }
    vm_param_ref(p, &result);
    CHECK("after 20 pushes, value = 1019", result.as.i == 1019);
    for (int i = 19; i >= 0; i--) {
        vm_param_pop(p);
        uintptr_t expected = (i > 0) ? (uintptr_t)(1000 + i - 1) : 100;
        vm_param_ref(p, &result);
        if ((uintptr_t)result.as.i != expected) {
            printf("  FAIL: pop %d expected %lu got %lu\n", i, (unsigned long)expected,
                   (unsigned long)result.as.i);
            fail++;
            break;
        }
    }
    vm_param_ref(p, &result);
    CHECK("all 20 pops restored correctly", result.as.i == 100);

    /* parameter? false for non-parameter (use a valid arena-allocated object) */
    void* not_param = vm_alloc_object(&rs, VM_SUBTYPE_STRING, 16);
    CHECK("parameter? false for non-param", !vm_param_is_parameter(not_param));

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_PARAMETER_TEST */
