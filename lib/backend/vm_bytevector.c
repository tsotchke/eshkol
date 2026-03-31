/**
 * @file vm_bytevector.c
 * @brief Raw byte arrays (R7RS bytevectors) for the Eshkol bytecode VM.
 *
 * Bytevectors are mutable sequences of exact integers in [0, 255].
 * Arena-allocated, no GC.
 *
 * Native call IDs: 680-699
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>

/* ── Bytevector ── */

typedef struct {
    int len;
    uint8_t* data;   /* arena-allocated */
} VmBytevector;

/* Forward-declare string struct for utf8 conversions.
 * Full definition lives in vm_string.c — skip if already included. */
#ifndef VM_STRING_C_INCLUDED
typedef struct {
    int byte_len;
    int char_len;
    char* data;
} VmString;
#endif

/* ── Allocation ── */

static VmBytevector* vm_bv_alloc(VmRegionStack* rs, int len) {
    VmBytevector* bv = (VmBytevector*)vm_alloc_object(rs, VM_SUBTYPE_BYTEVEC,
                                                       sizeof(VmBytevector));
    if (!bv) return NULL;
    bv->len = len;
    bv->data = (uint8_t*)vm_alloc(rs, (size_t)(len > 0 ? len : 1));
    if (!bv->data) return NULL;
    return bv;
}

/* ── Public API ── */

/* 680: make-bytevector(n, fill) */
VmBytevector* vm_bv_make(VmRegionStack* rs, int n, int fill) {
    VmBytevector* bv = vm_bv_alloc(rs, n);
    if (!bv) return NULL;
    memset(bv->data, (uint8_t)(fill & 0xFF), (size_t)n);
    return bv;
}

/* 681: bytevector-length */
int vm_bv_length(const VmBytevector* bv) {
    return bv ? bv->len : 0;
}

/* 682: bytevector-u8-ref */
int vm_bv_u8_ref(const VmBytevector* bv, int k) {
    if (!bv || k < 0 || k >= bv->len) {
        fprintf(stderr, "ERROR: bytevector-u8-ref index %d out of range [0,%d)\n",
                k, bv ? bv->len : 0);
        return -1;
    }
    return bv->data[k];
}

/* 683: bytevector-u8-set! */
void vm_bv_u8_set(VmBytevector* bv, int k, int byte) {
    if (!bv || k < 0 || k >= bv->len) {
        fprintf(stderr, "ERROR: bytevector-u8-set! index %d out of range [0,%d)\n",
                k, bv ? bv->len : 0);
        return;
    }
    bv->data[k] = (uint8_t)(byte & 0xFF);
}

/* 684: bytevector-copy(bv, [start], [end]) */
VmBytevector* vm_bv_copy(VmRegionStack* rs, const VmBytevector* bv, int start, int end) {
    if (!bv) return NULL;
    if (start < 0) start = 0;
    if (end < 0 || end > bv->len) end = bv->len;
    if (start > end) start = end;
    int n = end - start;
    VmBytevector* out = vm_bv_alloc(rs, n);
    if (!out) return NULL;
    if (n > 0) memcpy(out->data, bv->data + start, (size_t)n);
    return out;
}

/* 685: bytevector-copy!(to, at, from, start, end) */
void vm_bv_copy_to(VmBytevector* to, int at, const VmBytevector* from, int start, int end) {
    if (!to || !from) return;
    if (start < 0) start = 0;
    if (end < 0 || end > from->len) end = from->len;
    if (start > end) start = end;
    int n = end - start;
    if (at < 0 || at + n > to->len) {
        fprintf(stderr, "ERROR: bytevector-copy! destination overflow\n");
        return;
    }
    /* memmove handles overlapping regions (same bytevector copy) */
    memmove(to->data + at, from->data + start, (size_t)n);
}

/* 686: bytevector-append(bv1, bv2) */
VmBytevector* vm_bv_append(VmRegionStack* rs, const VmBytevector* a, const VmBytevector* b) {
    int alen = a ? a->len : 0;
    int blen = b ? b->len : 0;
    VmBytevector* out = vm_bv_alloc(rs, alen + blen);
    if (!out) return NULL;
    if (alen > 0) memcpy(out->data, a->data, (size_t)alen);
    if (blen > 0) memcpy(out->data + alen, b->data, (size_t)blen);
    return out;
}

/* 687: bytevector? */
int vm_bv_is_bytevector(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_BYTEVEC;
}

/* 688: bytevector (literal constructor from list of bytes) */
VmBytevector* vm_bv_from_bytes(VmRegionStack* rs, const int* bytes, int n) {
    VmBytevector* bv = vm_bv_alloc(rs, n);
    if (!bv) return NULL;
    for (int i = 0; i < n; i++)
        bv->data[i] = (uint8_t)(bytes[i] & 0xFF);
    return bv;
}

/* 689: utf8->string(bv, start, end) — interpret bytes as UTF-8, create string */
VmString* vm_bv_utf8_to_string(VmRegionStack* rs, const VmBytevector* bv, int start, int end) {
    if (!bv) return NULL;
    if (start < 0) start = 0;
    if (end < 0 || end > bv->len) end = bv->len;
    if (start > end) start = end;
    int n = end - start;

    VmString* s = (VmString*)vm_alloc_object(rs, VM_SUBTYPE_STRING, sizeof(VmString));
    if (!s) return NULL;
#ifdef VM_STRING_C_INCLUDED
    s->byte_len = n;
    s->char_len = n; /* approximate: assumes ASCII-like for raw bv->string */
#else
    s->len = n;
    s->cap = n + 1;
#endif
    s->data = (char*)vm_alloc(rs, (size_t)(n + 1));
    if (!s->data) return NULL;
    memcpy(s->data, bv->data + start, (size_t)n);
    s->data[n] = '\0';
    return s;
}

/* 690: string->utf8(str, start, end) — encode string as UTF-8 bytevector */
VmBytevector* vm_bv_string_to_utf8(VmRegionStack* rs, const VmString* s, int start, int end) {
    if (!s) return NULL;
    if (start < 0) start = 0;
#ifdef VM_STRING_C_INCLUDED
    if (end < 0 || end > s->byte_len) end = s->byte_len;
#else
    if (end < 0 || end > s->len) end = s->len;
#endif
    if (start > end) start = end;
    int n = end - start;

    VmBytevector* bv = vm_bv_alloc(rs, n);
    if (!bv) return NULL;
    memcpy(bv->data, s->data + start, (size_t)n);
    return bv;
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

void* vm_bv_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 680: return vm_bv_make(rs, nargs >= 1 ? (int)(intptr_t)args[0] : 0,
                                     nargs >= 2 ? (int)(intptr_t)args[1] : 0);
    case 681: { static int r; r = vm_bv_length((VmBytevector*)args[0]); return &r; }
    case 682: { static int r; r = vm_bv_u8_ref((VmBytevector*)args[0], (int)(intptr_t)args[1]); return &r; }
    case 683: vm_bv_u8_set((VmBytevector*)args[0], (int)(intptr_t)args[1], (int)(intptr_t)args[2]); return NULL;
    case 684: return vm_bv_copy(rs, (VmBytevector*)args[0],
                                nargs >= 2 ? (int)(intptr_t)args[1] : 0,
                                nargs >= 3 ? (int)(intptr_t)args[2] : -1);
    case 685: vm_bv_copy_to((VmBytevector*)args[0], (int)(intptr_t)args[1],
                            (VmBytevector*)args[2],
                            nargs >= 4 ? (int)(intptr_t)args[3] : 0,
                            nargs >= 5 ? (int)(intptr_t)args[4] : -1);
              return NULL;
    case 686: return vm_bv_append(rs, (VmBytevector*)args[0],
                                       nargs >= 2 ? (VmBytevector*)args[1] : NULL);
    case 687: { static int r; r = vm_bv_is_bytevector(args[0]); return &r; }
    case 689: return vm_bv_utf8_to_string(rs, (VmBytevector*)args[0],
                                          nargs >= 2 ? (int)(intptr_t)args[1] : 0,
                                          nargs >= 3 ? (int)(intptr_t)args[2] : -1);
    case 690: return vm_bv_string_to_utf8(rs, (VmString*)args[0],
                                          nargs >= 2 ? (int)(intptr_t)args[1] : 0,
                                          nargs >= 3 ? (int)(intptr_t)args[2] : -1);
    default:
        fprintf(stderr, "ERROR: unknown bytevector native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_BYTEVECTOR_TEST

#include <assert.h>

int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_bytevector self-test ===\n\n");

    /* make-bytevector with fill */
    VmBytevector* bv = vm_bv_make(&rs, 8, 0xAB);
    CHECK("make returns non-null", bv != NULL);
    CHECK("length is 8", vm_bv_length(bv) == 8);
    CHECK("fill byte correct", vm_bv_u8_ref(bv, 0) == 0xAB);
    CHECK("fill byte correct [7]", vm_bv_u8_ref(bv, 7) == 0xAB);
    CHECK("bytevector? true", vm_bv_is_bytevector(bv));

    /* u8-set! / u8-ref */
    vm_bv_u8_set(bv, 3, 42);
    CHECK("u8-set!/u8-ref at 3", vm_bv_u8_ref(bv, 3) == 42);
    vm_bv_u8_set(bv, 0, 0);
    CHECK("u8-set!/u8-ref at 0", vm_bv_u8_ref(bv, 0) == 0);
    vm_bv_u8_set(bv, 7, 255);
    CHECK("u8-set!/u8-ref at 7", vm_bv_u8_ref(bv, 7) == 255);

    /* bytevector-copy */
    VmBytevector* c = vm_bv_copy(&rs, bv, 2, 6);
    CHECK("copy length is 4", vm_bv_length(c) == 4);
    CHECK("copy[0] = bv[2]", vm_bv_u8_ref(c, 0) == vm_bv_u8_ref(bv, 2));
    CHECK("copy[1] = bv[3]", vm_bv_u8_ref(c, 1) == 42);

    /* bytevector-copy! */
    VmBytevector* dst = vm_bv_make(&rs, 10, 0);
    vm_bv_copy_to(dst, 2, bv, 0, 4);
    CHECK("copy! dst[2] = bv[0]", vm_bv_u8_ref(dst, 2) == vm_bv_u8_ref(bv, 0));
    CHECK("copy! dst[5] = bv[3]", vm_bv_u8_ref(dst, 5) == 42);
    CHECK("copy! dst[0] untouched", vm_bv_u8_ref(dst, 0) == 0);

    /* bytevector-append */
    VmBytevector* a = vm_bv_make(&rs, 3, 1);
    VmBytevector* b = vm_bv_make(&rs, 4, 2);
    VmBytevector* ab = vm_bv_append(&rs, a, b);
    CHECK("append length is 7", vm_bv_length(ab) == 7);
    CHECK("append[0] from a", vm_bv_u8_ref(ab, 0) == 1);
    CHECK("append[3] from b", vm_bv_u8_ref(ab, 3) == 2);
    CHECK("append[6] from b", vm_bv_u8_ref(ab, 6) == 2);

    /* utf8->string and string->utf8 round-trip */
    int hello[] = { 'H', 'e', 'l', 'l', 'o' };
    VmBytevector* hbv = vm_bv_from_bytes(&rs, hello, 5);
    CHECK("from_bytes length", vm_bv_length(hbv) == 5);
    CHECK("from_bytes[0] = 'H'", vm_bv_u8_ref(hbv, 0) == 'H');

    VmString* s = vm_bv_utf8_to_string(&rs, hbv, 0, -1);
    CHECK("utf8->string non-null", s != NULL);
#ifdef VM_STRING_C_INCLUDED
    CHECK("utf8->string len is 5", s->byte_len == 5);
#else
    CHECK("utf8->string len is 5", s->len == 5);
#endif
    CHECK("utf8->string content", memcmp(s->data, "Hello", 5) == 0);

    VmBytevector* rt = vm_bv_string_to_utf8(&rs, s, 0, -1);
    CHECK("string->utf8 round-trip len", vm_bv_length(rt) == 5);
    CHECK("string->utf8 round-trip data", vm_bv_u8_ref(rt, 0) == 'H' && vm_bv_u8_ref(rt, 4) == 'o');

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_BYTEVECTOR_TEST */
