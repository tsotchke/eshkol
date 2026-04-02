/**
 * @file vm_error.c
 * @brief R7RS error objects for structured exception handling.
 *
 * Error objects carry a message, type symbol, and irritant values.
 * The actual exception dispatch (guard/raise) is handled by the VM;
 * this module provides error object creation and accessors.
 *
 * Native call IDs: 710-714
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

/* ── Error Object ── */

typedef struct {
    char message[256];
    void* irritants;   /* cons list of irritant values (or NULL) */
    char type[64];     /* error type symbol (e.g., "read-error") */
} VmError;

/* Simple cons cell for irritant list (arena-allocated) */
typedef struct VmCons {
    void* car;
    void* cdr;  /* next VmCons* or NULL */
} VmCons;

/* ── Helpers ── */

static VmCons* vm_err_cons(VmRegionStack* rs, void* car, void* cdr) {
    VmCons* c = (VmCons*)vm_alloc_object(rs, VM_SUBTYPE_CONS, sizeof(VmCons));
    if (!c) return NULL;
    c->car = car;
    c->cdr = cdr;
    return c;
}

/* Build a cons list from an array of values (in order) */
static void* vm_err_list_from_array(VmRegionStack* rs, void** items, int n) {
    void* list = NULL;
    for (int i = n - 1; i >= 0; i--) {
        list = vm_err_cons(rs, items[i], list);
    }
    return list;
}

/* ── Public API ── */

/* 710: error(type, message, irritants...) → allocate error object */
VmError* vm_error_make(VmRegionStack* rs, const char* type, const char* message,
                       void** irritants, int n_irritants) {
    VmError* e = (VmError*)vm_alloc_object(rs, VM_SUBTYPE_ERROR, sizeof(VmError));
    if (!e) return NULL;

    if (message) {
        size_t mlen = strlen(message);
        if (mlen >= sizeof(e->message)) mlen = sizeof(e->message) - 1;
        memcpy(e->message, message, mlen);
        e->message[mlen] = '\0';
    } else {
        e->message[0] = '\0';
    }

    if (type) {
        size_t tlen = strlen(type);
        if (tlen >= sizeof(e->type)) tlen = sizeof(e->type) - 1;
        memcpy(e->type, type, tlen);
        e->type[tlen] = '\0';
    } else {
        strcpy(e->type, "error");
    }

    e->irritants = (n_irritants > 0)
        ? vm_err_list_from_array(rs, irritants, n_irritants)
        : NULL;

    return e;
}

/* 711: error-object? */
int vm_error_is_error(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_ERROR;
}

/* 712: error-object-message → returns pointer to message string */
const char* vm_error_message(const VmError* e) {
    return e ? e->message : "";
}

/* 713: error-object-irritants → returns irritant cons list */
void* vm_error_irritants(const VmError* e) {
    return e ? e->irritants : NULL;
}

/* 714: error-object-type → returns type symbol string */
const char* vm_error_type(const VmError* e) {
    return e ? e->type : "";
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

void* vm_error_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 710: {
        /* error(type, message, irritant1, irritant2, ...) */
        const char* type = (nargs >= 1) ? (const char*)args[0] : NULL;
        const char* msg  = (nargs >= 2) ? (const char*)args[1] : NULL;
        int nirr = (nargs > 2) ? nargs - 2 : 0;
        void** irritants = (nirr > 0) ? &args[2] : NULL;
        return vm_error_make(rs, type, msg, irritants, nirr);
    }
    case 711: { static int r; r = vm_error_is_error(args[0]); return &r; }
    case 712: return (void*)vm_error_message((VmError*)args[0]);
    case 713: return vm_error_irritants((VmError*)args[0]);
    case 714: return (void*)vm_error_type((VmError*)args[0]);
    default:
        fprintf(stderr, "ERROR: unknown error native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_ERROR_TEST

#include <assert.h>

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_error self-test ===\n\n");

    /* Basic error with no irritants */
    VmError* e1 = vm_error_make(&rs, "read-error", "unexpected EOF", NULL, 0);
    CHECK("make returns non-null", e1 != NULL);
    CHECK("error-object? true", vm_error_is_error(e1));
    CHECK("message correct", strcmp(vm_error_message(e1), "unexpected EOF") == 0);
    CHECK("type correct", strcmp(vm_error_type(e1), "read-error") == 0);
    CHECK("no irritants", vm_error_irritants(e1) == NULL);

    /* Error with irritants */
    void* irr[3] = { (void*)(uintptr_t)10, (void*)(uintptr_t)20, (void*)(uintptr_t)30 };
    VmError* e2 = vm_error_make(&rs, "range-error", "index out of bounds", irr, 3);
    CHECK("e2 non-null", e2 != NULL);
    CHECK("e2 message", strcmp(vm_error_message(e2), "index out of bounds") == 0);
    CHECK("e2 type", strcmp(vm_error_type(e2), "range-error") == 0);

    /* Walk irritant list */
    VmCons* list = (VmCons*)vm_error_irritants(e2);
    CHECK("irritant list non-null", list != NULL);
    CHECK("irritant[0] = 10", (uintptr_t)list->car == 10);
    list = (VmCons*)list->cdr;
    CHECK("irritant[1] = 20", list != NULL && (uintptr_t)list->car == 20);
    list = (VmCons*)list->cdr;
    CHECK("irritant[2] = 30", list != NULL && (uintptr_t)list->car == 30);
    CHECK("irritant list ends", list->cdr == NULL);

    /* Default type when NULL */
    VmError* e3 = vm_error_make(&rs, NULL, "oops", NULL, 0);
    CHECK("default type is 'error'", strcmp(vm_error_type(e3), "error") == 0);

    /* error-object? false for non-error (use a valid arena-allocated object) */
    void* not_err = vm_alloc_object(&rs, VM_SUBTYPE_STRING, 16);
    CHECK("error-object? false for non-error", !vm_error_is_error(not_err));

    /* Long message truncation (no crash) */
    char longmsg[512];
    memset(longmsg, 'A', 511);
    longmsg[511] = '\0';
    VmError* e4 = vm_error_make(&rs, "test", longmsg, NULL, 0);
    CHECK("long message truncated", strlen(vm_error_message(e4)) == 255);

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_ERROR_TEST */
