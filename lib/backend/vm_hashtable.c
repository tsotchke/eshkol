/**
 * @file vm_hashtable.c
 * @brief Hash table with open addressing and linear probing.
 *
 * R7RS-compatible hash tables for the Eshkol bytecode VM.
 * FNV-1a hash, linear probing, rehash at 75% load.
 *
 * Native call IDs: 660-679
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>

/* ── Hash Table ── */

typedef struct {
    int capacity, count;
    uint64_t* hashes;   /* pre-computed hashes, 0 = empty slot */
    void** keys;        /* arena-allocated, NULL = empty slot */
    void** values;      /* arena-allocated */
} VmHashTable;

#define HT_INITIAL_CAP 16
#define HT_EMPTY_HASH  0
#define HT_TOMBSTONE   1  /* deleted marker hash */

/* ── FNV-1a Hash ── */

/** @brief FNV-1a hash of a byte buffer, remapped away from the
 *         HT_EMPTY_HASH/HT_TOMBSTONE sentinel values. */
static uint64_t vm_ht_fnv1a(const void* data, size_t len) {
    const uint8_t* p = (const uint8_t*)data;
    uint64_t h = 14695981039346656037ULL;
    for (size_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
    /* Ensure we never produce EMPTY or TOMBSTONE sentinels */
    if (h <= HT_TOMBSTONE) h += 2;
    return h;
}

/** @brief Hash a tagged value by hashing its raw 8-byte pointer/value
 *         bits. */
static uint64_t vm_ht_hash_value(void* val) {
    uint64_t bits = (uint64_t)(uintptr_t)val;
    return vm_ht_fnv1a(&bits, sizeof(bits));
}

/** @brief Key equality for the default hash table: pointer equality (R7RS
 *         `eqv?` semantics). */
static int vm_ht_keys_equal(void* a, void* b) {
    return a == b;
}

/* ── Allocation ── */

/** @brief Allocate a hash table with @p capacity slots (must be a power of
 *         two — see the `& mask` probing in vm_ht_probe()), all initially
 *         empty. */
static VmHashTable* vm_ht_alloc(VmRegionStack* rs, int capacity) {
    VmHashTable* ht = (VmHashTable*)vm_alloc_object(rs, VM_SUBTYPE_HASH,
                                                     sizeof(VmHashTable));
    if (!ht) return NULL;
    ht->capacity = capacity;
    ht->count = 0;
    ht->hashes = (uint64_t*)vm_alloc(rs, (size_t)capacity * sizeof(uint64_t));
    ht->keys   = (void**)vm_alloc(rs, (size_t)capacity * sizeof(void*));
    ht->values = (void**)vm_alloc(rs, (size_t)capacity * sizeof(void*));
    if (!ht->hashes || !ht->keys || !ht->values) return NULL;
    memset(ht->hashes, 0, (size_t)capacity * sizeof(uint64_t));
    memset(ht->keys,   0, (size_t)capacity * sizeof(void*));
    memset(ht->values,  0, (size_t)capacity * sizeof(void*));
    return ht;
}

/* ── Probe for a key ── */

/**
 * @brief Linear-probe @p ht starting at slot `h & (capacity-1)` for @p key
 *        (already hashed to @p h), tracking the first tombstone seen so
 *        insertions can reuse deleted slots.
 * @return The index where @p key lives, or (if absent) the first
 *         empty/tombstone slot suitable for insertion; sets *found
 *         accordingly.
 */
static int vm_ht_probe(const VmHashTable* ht, void* key, uint64_t h, int* found) {
    int mask = ht->capacity - 1;
    int idx = (int)(h & (uint64_t)mask);
    int first_tombstone = -1;

    for (int i = 0; i < ht->capacity; i++) {
        uint64_t sh = ht->hashes[idx];
        if (sh == HT_EMPTY_HASH) {
            /* Empty slot: key not in table */
            *found = 0;
            return (first_tombstone >= 0) ? first_tombstone : idx;
        }
        if (sh == HT_TOMBSTONE) {
            if (first_tombstone < 0) first_tombstone = idx;
        } else if (sh == h && vm_ht_keys_equal(ht->keys[idx], key)) {
            *found = 1;
            return idx;
        }
        idx = (idx + 1) & mask;
    }
    /* Table is full of tombstones */
    *found = 0;
    return (first_tombstone >= 0) ? first_tombstone : 0;
}

/* ── Rehash ── */

/** @brief Double @p ht's capacity and re-insert all live (non-tombstone)
 *         entries into the new arrays; the old arena-allocated arrays are
 *         simply abandoned (freed when their region pops). */
static void vm_ht_rehash(VmRegionStack* rs, VmHashTable* ht) {
    int old_cap = ht->capacity;
    uint64_t* old_hashes = ht->hashes;
    void** old_keys = ht->keys;
    void** old_values = ht->values;

    int new_cap = old_cap * 2;
    ht->capacity = new_cap;
    ht->count = 0;
    ht->hashes = (uint64_t*)vm_alloc(rs, (size_t)new_cap * sizeof(uint64_t));
    ht->keys   = (void**)vm_alloc(rs, (size_t)new_cap * sizeof(void*));
    ht->values = (void**)vm_alloc(rs, (size_t)new_cap * sizeof(void*));
    memset(ht->hashes, 0, (size_t)new_cap * sizeof(uint64_t));
    memset(ht->keys,   0, (size_t)new_cap * sizeof(void*));
    memset(ht->values,  0, (size_t)new_cap * sizeof(void*));

    for (int i = 0; i < old_cap; i++) {
        if (old_hashes[i] > HT_TOMBSTONE) {
            int found;
            int idx = vm_ht_probe(ht, old_keys[i], old_hashes[i], &found);
            ht->hashes[idx] = old_hashes[i];
            ht->keys[idx]   = old_keys[i];
            ht->values[idx] = old_values[i];
            ht->count++;
        }
    }
    /* Old arrays are arena-allocated, freed when region pops */
}

/* ── Public API ── */

/** @brief Native call 660: `(make-hash-table)`. */
VmHashTable* vm_ht_make(VmRegionStack* rs) {
    return vm_ht_alloc(rs, HT_INITIAL_CAP);
}

/** @brief Native call 661: `(hash-table-ref ht key [default])`. */
void* vm_ht_ref(VmHashTable* ht, void* key, void* dflt) {
    if (!ht) return dflt;
    uint64_t h = vm_ht_hash_value(key);
    int found;
    int idx = vm_ht_probe(ht, key, h, &found);
    return found ? ht->values[idx] : dflt;
}

/** @brief Native call 662: `(hash-table-set! ht key value)`; rehashes
 *         first if the load factor would exceed 75%. */
void vm_ht_set(VmRegionStack* rs, VmHashTable* ht, void* key, void* value) {
    if (!ht) return;
    /* Check load factor: rehash at 75% */
    if (ht->count * 4 >= ht->capacity * 3) {
        vm_ht_rehash(rs, ht);
    }
    uint64_t h = vm_ht_hash_value(key);
    int found;
    int idx = vm_ht_probe(ht, key, h, &found);
    if (!found) ht->count++;
    ht->hashes[idx] = h;
    ht->keys[idx]   = key;
    ht->values[idx] = value;
}

/** @brief Native call 663: `(hash-table-has-key? ht key)`. */
int vm_ht_has_key(VmHashTable* ht, void* key) {
    if (!ht) return 0;
    uint64_t h = vm_ht_hash_value(key);
    int found;
    vm_ht_probe(ht, key, h, &found);
    return found;
}

/** @brief Native call 664: `(hash-table-remove! ht key)`; marks the slot
 *         as a tombstone so subsequent probes keep working. */
void vm_ht_remove(VmHashTable* ht, void* key) {
    if (!ht) return;
    uint64_t h = vm_ht_hash_value(key);
    int found;
    int idx = vm_ht_probe(ht, key, h, &found);
    if (found) {
        ht->hashes[idx] = HT_TOMBSTONE;
        ht->keys[idx]   = NULL;
        ht->values[idx] = NULL;
        ht->count--;
    }
}

/** @brief Native call 665: `(hash-table-keys ht)` — build an
 *         arena-allocated array of all live keys, writing the count to
 *         @p out_n. */
void** vm_ht_keys(VmRegionStack* rs, VmHashTable* ht, int* out_n) {
    if (!ht || ht->count == 0) { *out_n = 0; return NULL; }
    void** result = (void**)vm_alloc(rs, (size_t)ht->count * sizeof(void*));
    int j = 0;
    for (int i = 0; i < ht->capacity && j < ht->count; i++) {
        if (ht->hashes[i] > HT_TOMBSTONE)
            result[j++] = ht->keys[i];
    }
    *out_n = j;
    return result;
}

/** @brief Native call 666: `(hash-table-values ht)` — build an
 *         arena-allocated array of all live values, writing the count to
 *         @p out_n. */
void** vm_ht_values(VmRegionStack* rs, VmHashTable* ht, int* out_n) {
    if (!ht || ht->count == 0) { *out_n = 0; return NULL; }
    void** result = (void**)vm_alloc(rs, (size_t)ht->count * sizeof(void*));
    int j = 0;
    for (int i = 0; i < ht->capacity && j < ht->count; i++) {
        if (ht->hashes[i] > HT_TOMBSTONE)
            result[j++] = ht->values[i];
    }
    *out_n = j;
    return result;
}

/** @brief Native call 667: `(hash-table-count ht)`. */
int vm_ht_count(VmHashTable* ht) {
    return ht ? ht->count : 0;
}

/** @brief Native call 668: `(hash-table-clear! ht)`. */
void vm_ht_clear(VmHashTable* ht) {
    if (!ht) return;
    memset(ht->hashes, 0, (size_t)ht->capacity * sizeof(uint64_t));
    memset(ht->keys,   0, (size_t)ht->capacity * sizeof(void*));
    memset(ht->values,  0, (size_t)ht->capacity * sizeof(void*));
    ht->count = 0;
}

/** @brief Native call 669: `(hash-table? obj)` — check the heap object
 *         header subtype. */
int vm_ht_is_hashtable(void* obj) {
    if (!obj) return 0;
    VmObjectHeader* hdr = (VmObjectHeader*)((uint8_t*)obj - sizeof(VmObjectHeader));
    return hdr->subtype == VM_SUBTYPE_HASH;
}

/*******************************************************************************
 * Dispatch
 ******************************************************************************/

/** @brief Native-call dispatcher for the hash-table primitives (IDs
 *         660-669). */
void* vm_ht_dispatch(VmRegionStack* rs, int id, void** args, int nargs) {
    switch (id) {
    case 660: return vm_ht_make(rs);
    case 661: return (nargs >= 3) ? vm_ht_ref((VmHashTable*)args[0], args[1], args[2])
                                  : vm_ht_ref((VmHashTable*)args[0], args[1], NULL);
    case 662: vm_ht_set(rs, (VmHashTable*)args[0], args[1], args[2]); return NULL;
    case 663: { static int r; r = vm_ht_has_key((VmHashTable*)args[0], args[1]); return &r; }
    case 664: vm_ht_remove((VmHashTable*)args[0], args[1]); return NULL;
    case 665: { int n; void** k = vm_ht_keys(rs, (VmHashTable*)args[0], &n); (void)n; return k; }
    case 666: { int n; void** v = vm_ht_values(rs, (VmHashTable*)args[0], &n); (void)n; return v; }
    case 667: { static int r; r = vm_ht_count((VmHashTable*)args[0]); return &r; }
    case 668: vm_ht_clear((VmHashTable*)args[0]); return NULL;
    case 669: { static int r; r = vm_ht_is_hashtable(args[0]); return &r; }
    default:
        fprintf(stderr, "ERROR: unknown hash-table native ID %d\n", id);
        return NULL;
    }
}

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_HASHTABLE_TEST

#include <assert.h>

/** @brief Standalone self-test (built when VM_HASHTABLE_TEST is defined):
 *         exercises make/ref/set!/has-key?/remove!/keys/values/clear and
 *         verifies data survives a triggered rehash. */
int main(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    int pass = 0, fail = 0;

#define CHECK(name, cond) do { \
    if (cond) { pass++; printf("  PASS: %s\n", name); } \
    else { fail++; printf("  FAIL: %s\n", name); } \
} while(0)

    printf("=== vm_hashtable self-test ===\n\n");

    VmHashTable* ht = vm_ht_make(&rs);
    CHECK("make-hash-table returns non-null", ht != NULL);
    CHECK("initial count is 0", vm_ht_count(ht) == 0);
    CHECK("hash-table? true", vm_ht_is_hashtable(ht));

    /* Insert 10 pairs: keys = (void*)1..10, values = (void*)100..109 */
    for (int i = 1; i <= 10; i++) {
        vm_ht_set(&rs, ht, (void*)(uintptr_t)i, (void*)(uintptr_t)(100 + i - 1));
    }
    CHECK("count after 10 inserts is 10", vm_ht_count(ht) == 10);

    /* Lookup all 10 */
    int all_found = 1;
    for (int i = 1; i <= 10; i++) {
        void* v = vm_ht_ref(ht, (void*)(uintptr_t)i, NULL);
        if ((uintptr_t)v != (uintptr_t)(100 + i - 1)) { all_found = 0; break; }
    }
    CHECK("all 10 lookups correct", all_found);

    /* has-key? for existing and missing */
    CHECK("has-key? for key 5", vm_ht_has_key(ht, (void*)(uintptr_t)5));
    CHECK("!has-key? for key 42", !vm_ht_has_key(ht, (void*)(uintptr_t)42));

    /* Default value for missing key */
    void* d = vm_ht_ref(ht, (void*)(uintptr_t)42, (void*)(uintptr_t)999);
    CHECK("ref missing key returns default", (uintptr_t)d == 999);

    /* Update existing key */
    vm_ht_set(&rs, ht, (void*)(uintptr_t)3, (void*)(uintptr_t)333);
    CHECK("update key 3", (uintptr_t)vm_ht_ref(ht, (void*)(uintptr_t)3, NULL) == 333);
    CHECK("count still 10 after update", vm_ht_count(ht) == 10);

    /* Remove key */
    vm_ht_remove(ht, (void*)(uintptr_t)5);
    CHECK("count after remove is 9", vm_ht_count(ht) == 9);
    CHECK("removed key not found", !vm_ht_has_key(ht, (void*)(uintptr_t)5));
    CHECK("other keys still found", vm_ht_has_key(ht, (void*)(uintptr_t)7));

    /* keys and values */
    int nk;
    void** keys = vm_ht_keys(&rs, ht, &nk);
    CHECK("keys count matches", nk == 9);
    CHECK("keys array non-null", keys != NULL);

    int nv;
    void** vals = vm_ht_values(&rs, ht, &nv);
    CHECK("values count matches", nv == 9);
    CHECK("values array non-null", vals != NULL);

    /* Force rehash by inserting more entries (initial cap = 16, 75% = 12) */
    for (int i = 11; i <= 20; i++) {
        vm_ht_set(&rs, ht, (void*)(uintptr_t)i, (void*)(uintptr_t)(200 + i));
    }
    CHECK("count after bulk insert is 19", vm_ht_count(ht) == 19);
    CHECK("capacity grew (rehash triggered)", ht->capacity > HT_INITIAL_CAP);
    /* Verify data survived rehash */
    CHECK("key 7 survived rehash", (uintptr_t)vm_ht_ref(ht, (void*)(uintptr_t)7, NULL) == 106);
    CHECK("key 15 after rehash", (uintptr_t)vm_ht_ref(ht, (void*)(uintptr_t)15, NULL) == 215);

    /* Clear */
    vm_ht_clear(ht);
    CHECK("count after clear is 0", vm_ht_count(ht) == 0);
    CHECK("lookup after clear returns default",
          vm_ht_ref(ht, (void*)(uintptr_t)1, (void*)(uintptr_t)0) == NULL);

    printf("\n%d passed, %d failed out of %d total\n", pass, fail, pass + fail);

    vm_region_stack_destroy(&rs);
    return fail > 0 ? 1 : 0;

#undef CHECK
}

#endif /* VM_HASHTABLE_TEST */
