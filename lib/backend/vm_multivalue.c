/**
 * @file vm_multivalue.c
 * @brief Multiple return values (R7RS values/call-with-values).
 *
 * Implements the R7RS multiple-value mechanism:
 *   (values v1 v2 ... vn) → multi-value container
 *   (call-with-values producer consumer) → unpack and apply
 *
 * Native call IDs: 650-654
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>

/* ── Multi-Value Container ── */

typedef struct {
    void** values;   /* arena-allocated array of tagged values */
    int count;
} VmMultiValue;

/* ── Allocation ── */

/** @brief Native call 650: `(values v1 v2 ... vn)` — allocate a multi-value
 *         container copying the @p n values from @p vals. */
VmMultiValue* vm_mv_values(VmRegionStack* rs, void** vals, int n) {
    VmMultiValue* mv = (VmMultiValue*)vm_alloc_object(rs, VM_SUBTYPE_MULTI_VAL,
                                                       sizeof(VmMultiValue));
    if (!mv) return NULL;
    mv->count = n;
    mv->values = (void**)vm_alloc(rs, (size_t)(n > 0 ? n : 1) * sizeof(void*));
    if (!mv->values) return NULL;
    for (int i = 0; i < n; i++)
        mv->values[i] = vals[i];
    return mv;
}

/** @brief Native call 651: extract the value at @p index from @p mv,
 *         printing an error and returning NULL if out of range. */
void* vm_mv_ref(const VmMultiValue* mv, int index) {
    if (!mv || index < 0 || index >= mv->count) {
        fprintf(stderr, "ERROR: multi-value-ref index %d out of range [0,%d)\n",
                index, mv ? mv->count : 0);
        return NULL;
    }
    return mv->values[index];
}

/** @brief Native call 652: number of values held by @p mv. */
int vm_mv_count(const VmMultiValue* mv) {
    return mv ? mv->count : 0;
}

/** @brief Native call 653: `(multi-value? obj)` — check the heap object
 *         header subtype. */
int vm_mv_is_multivalue(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_MULTI_VAL;
}

/**
 * @brief Native call 654: support for `call-with-values` — exposes @p mv's
 *        backing values array and writes its count to @p out_count so the
 *        VM's CALL instruction can unpack them as consumer arguments. The
 *        actual producer/consumer call dispatch happens in the VM itself;
 *        this is purely the unpacking helper.
 */
void** vm_mv_unpack(const VmMultiValue* mv, int* out_count) {
    if (!mv) { *out_count = 0; return NULL; }
    *out_count = mv->count;
    return mv->values;
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

/** @brief Native-call dispatcher for the multi-value primitives (IDs
 *         650-654). */
void* vm_mv_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 650: return vm_mv_values(rs, args, nargs);
    case 651: return vm_mv_ref((VmMultiValue*)args[0], (int)(intptr_t)args[1]);
    case 652: { static int r; r = vm_mv_count((VmMultiValue*)args[0]); return &r; }
    case 653: { static int r; r = vm_mv_is_multivalue(args[0]); return &r; }
    case 654: {
        int cnt;
        void** unpacked = vm_mv_unpack((VmMultiValue*)args[0], &cnt);
        (void)cnt;
        return unpacked;
    }
    default:
        fprintf(stderr, "ERROR: unknown multi-value native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_MULTIVALUE_TEST

#include <assert.h>

/** @brief Standalone self-test (built when VM_MULTIVALUE_TEST is defined):
 *         exercises values creation/ref/count, single/zero-value edge
 *         cases, unpack, and the multi-value? type check. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_multivalue self-test ===\n\n");

    /* Create (values 1 2 3) */
    void* vals3[3] = { (void*)(uintptr_t)1, (void*)(uintptr_t)2, (void*)(uintptr_t)3 };
    VmMultiValue* mv3 = vm_mv_values(&rs, vals3, 3);
    CHECK("values returns non-null", mv3 != NULL);
    CHECK("count is 3", vm_mv_count(mv3) == 3);
    CHECK("multi-value? true", vm_mv_is_multivalue(mv3));
    CHECK("ref 0 = 1", (uintptr_t)vm_mv_ref(mv3, 0) == 1);
    CHECK("ref 1 = 2", (uintptr_t)vm_mv_ref(mv3, 1) == 2);
    CHECK("ref 2 = 3", (uintptr_t)vm_mv_ref(mv3, 2) == 3);

    /* Single value */
    void* val1[1] = { (void*)(uintptr_t)42 };
    VmMultiValue* mv1 = vm_mv_values(&rs, val1, 1);
    CHECK("single value count", vm_mv_count(mv1) == 1);
    CHECK("single value ref", (uintptr_t)vm_mv_ref(mv1, 0) == 42);

    /* Zero values */
    VmMultiValue* mv0 = vm_mv_values(&rs, NULL, 0);
    CHECK("zero values count", vm_mv_count(mv0) == 0);

    /* Unpack for call-with-values */
    int cnt;
    void** unpacked = vm_mv_unpack(mv3, &cnt);
    CHECK("unpack count is 3", cnt == 3);
    CHECK("unpack[0] = 1", (uintptr_t)unpacked[0] == 1);
    CHECK("unpack[1] = 2", (uintptr_t)unpacked[1] == 2);
    CHECK("unpack[2] = 3", (uintptr_t)unpacked[2] == 3);

    /* Type check on non-multivalue (use a valid arena-allocated object) */
    void* not_mv = vm_alloc_object(&rs, VM_SUBTYPE_STRING, 16);
    CHECK("multi-value? false for non-mv", !vm_mv_is_multivalue(not_mv));

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_MULTIVALUE_TEST */
