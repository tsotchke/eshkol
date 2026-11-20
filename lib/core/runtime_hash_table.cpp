/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hash-table runtime helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/core/bignum.h"

#include <cstring>
#include <eshkol/core/table_hash.h>
#include <eshkol/core/rational.h>
#include <eshkol/core/i128.h>

void eshkol_hash_table_lock(void);
void eshkol_hash_table_unlock(void);

extern "C" {

static uint64_t hash_exact_integer(const eshkol_bignum_t* bn) {
    int64_t small;
    if (eshkol_bignum_fits_int64(bn, &small))
        return eshkol_table_hash_pair(ESHKOL_VALUE_INT64, (uint64_t)small);
    return eshkol_table_hash_pair(bn->sign, eshkol_table_hash(BIGNUM_LIMBS(bn),
                                  (size_t)bn->num_limbs * sizeof(uint64_t)));
}

uint64_t hash_tagged_value(const eshkol_tagged_value_t* value) {
    if (!value) return 0;
    uint8_t type = value->type;
    uint64_t hash = type;
    switch (type) {
        case ESHKOL_VALUE_NULL:
            return eshkol_table_hash_pair(type, 0);
        case ESHKOL_VALUE_COMPLEX:
            if (value->data.ptr_val)
                return eshkol_table_hash((const void*)value->data.ptr_val,
                                         sizeof(eshkol_complex_number_t));
            break;
        case ESHKOL_VALUE_STRING_PTR:
            if (value->data.ptr_val) {
                const char* s = (const char*)value->data.ptr_val;
                return eshkol_table_hash(s, std::strlen(s));
            }
            break;
        case ESHKOL_VALUE_HEAP_PTR:
            if (value->data.ptr_val) {
                uint8_t subtype = ESHKOL_GET_SUBTYPE((void*)value->data.ptr_val);
                if (subtype == HEAP_SUBTYPE_STRING || subtype == HEAP_SUBTYPE_SYMBOL) {
                    const char* s = (const char*)value->data.ptr_val;
                    return eshkol_table_hash(s, std::strlen(s));
                } else if (subtype == HEAP_SUBTYPE_BIGNUM) {
                    return hash_exact_integer((const eshkol_bignum_t*)value->data.ptr_val);
                } else if (subtype == HEAP_SUBTYPE_RATIONAL) {
                    const auto* r = (const eshkol_rational_t*)value->data.ptr_val;
                    return r->is_big
                        ? eshkol_table_hash_pair(hash_exact_integer(r->big_num), hash_exact_integer(r->big_den))
                        : eshkol_table_hash_pair(r->numerator, r->denominator);
                } else if (subtype == HEAP_SUBTYPE_I128) {
                    return eshkol_table_hash((const void*)value->data.ptr_val, sizeof(eshkol_i128_abi));
                } else if (subtype == HEAP_SUBTYPE_CONS) {
                    // Structural hash of a pair: combine hash(car) and hash(cdr)
                    // recursively, so equal lists/pairs hash equal (ESH-0064).
                    arena_tagged_cons_cell_t* cell =
                        (arena_tagged_cons_cell_t*)value->data.ptr_val;
                    eshkol_tagged_value_t car = arena_tagged_cons_get_tagged_value(cell, false);
                    eshkol_tagged_value_t cdr = arena_tagged_cons_get_tagged_value(cell, true);
                    return eshkol_table_hash_pair(hash_tagged_value(&car), hash_tagged_value(&cdr));
                } else if (subtype == HEAP_SUBTYPE_VECTOR) {
                    // Structural hash of a heterogeneous vector: [len:i64][elems...].
                    int64_t len = *(int64_t*)(uintptr_t)value->data.ptr_val;
                    eshkol_tagged_value_t* elems =
                        (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)value->data.ptr_val + 8);
                    for (int64_t i = 0; i < len; i++) {
                        hash = eshkol_table_hash_pair(hash, hash_tagged_value(&elems[i]));
                    }
                } else {
                    return eshkol_table_hash_pair(type, value->data.ptr_val);
                }
            }
            break;

        default:
            return eshkol_table_hash_pair(type, value->data.raw_val);
    }

    return hash;
}

// Compare table keys using their active numeric or structural representation.
bool hash_keys_equal(const eshkol_tagged_value_t* a, const eshkol_tagged_value_t* b) {
    if (!a || !b) return a == b;

    auto get_base_type = [](uint8_t t) -> uint8_t {
        return (t >= 8) ? t : (t & 0x0F);
    };
    uint8_t type_a = get_base_type(a->type);
    uint8_t type_b = get_base_type(b->type);

    if (type_a != type_b) {
        const auto* integer = type_a == ESHKOL_VALUE_INT64 ? a : b;
        const auto* boxed = integer == a ? b : a;
        return integer->type == ESHKOL_VALUE_INT64 && eshkol_is_bignum_tagged(boxed) &&
               eshkol_bignum_compare_int64((const eshkol_bignum_t*)boxed->data.ptr_val, integer->data.int_val) == 0;
    }

    switch (type_a) {
        case ESHKOL_VALUE_INT64:
        case ESHKOL_VALUE_BOOL:
        case ESHKOL_VALUE_CHAR:
            return a->data.int_val == b->data.int_val;

        case ESHKOL_VALUE_DOUBLE:
            return a->data.raw_val == b->data.raw_val;

        case ESHKOL_VALUE_COMPLEX:
            return a->data.ptr_val == b->data.ptr_val ||
                   (a->data.ptr_val && b->data.ptr_val &&
                    std::memcmp((const void*)a->data.ptr_val, (const void*)b->data.ptr_val,
                                sizeof(eshkol_complex_number_t)) == 0);

        case ESHKOL_VALUE_STRING_PTR:
            if (a->data.ptr_val == b->data.ptr_val) return true;
            if (!a->data.ptr_val || !b->data.ptr_val) return false;
            return std::strcmp((const char*)a->data.ptr_val,
                               (const char*)b->data.ptr_val) == 0;

        case ESHKOL_VALUE_SYMBOL:
            return a->data.ptr_val == b->data.ptr_val;

        case ESHKOL_VALUE_NULL:
            return true;

        case ESHKOL_VALUE_HEAP_PTR: {
            if (a->data.ptr_val == b->data.ptr_val) return true;
            if (!a->data.ptr_val || !b->data.ptr_val) return false;

            uint8_t subtype_a = ESHKOL_GET_SUBTYPE((void*)a->data.ptr_val);
            uint8_t subtype_b = ESHKOL_GET_SUBTYPE((void*)b->data.ptr_val);

            if (subtype_a != subtype_b) return false;

            if (subtype_a == HEAP_SUBTYPE_RATIONAL)
                return eshkol_rational_equal((void*)a->data.ptr_val, (void*)b->data.ptr_val) != 0;
            if (subtype_a == HEAP_SUBTYPE_I128)
                return std::memcmp((const void*)a->data.ptr_val, (const void*)b->data.ptr_val,
                                   sizeof(eshkol_i128_abi)) == 0;
            if (subtype_a == HEAP_SUBTYPE_STRING || subtype_a == HEAP_SUBTYPE_SYMBOL) {
                return std::strcmp((const char*)a->data.ptr_val,
                                   (const char*)b->data.ptr_val) == 0;
            }

            if (subtype_a == HEAP_SUBTYPE_BIGNUM) {
                return eshkol_bignum_compare((const eshkol_bignum_t*)a->data.ptr_val,
                                             (const eshkol_bignum_t*)b->data.ptr_val) == 0;
            }

            if (subtype_a == HEAP_SUBTYPE_CONS) {
                // Structural pair equality, recursing through hash_keys_equal so
                // hash and equality stay consistent (ESH-0064). Compound keys
                // (e.g. SICP data-directed (op . type) keys) now match by value.
                arena_tagged_cons_cell_t* ca = (arena_tagged_cons_cell_t*)a->data.ptr_val;
                arena_tagged_cons_cell_t* cb = (arena_tagged_cons_cell_t*)b->data.ptr_val;
                eshkol_tagged_value_t car_a = arena_tagged_cons_get_tagged_value(ca, false);
                eshkol_tagged_value_t car_b = arena_tagged_cons_get_tagged_value(cb, false);
                if (!hash_keys_equal(&car_a, &car_b)) return false;
                eshkol_tagged_value_t cdr_a = arena_tagged_cons_get_tagged_value(ca, true);
                eshkol_tagged_value_t cdr_b = arena_tagged_cons_get_tagged_value(cb, true);
                return hash_keys_equal(&cdr_a, &cdr_b);
            }

            if (subtype_a == HEAP_SUBTYPE_VECTOR) {
                int64_t len_a = *(int64_t*)(uintptr_t)a->data.ptr_val;
                int64_t len_b = *(int64_t*)(uintptr_t)b->data.ptr_val;
                if (len_a != len_b) return false;
                eshkol_tagged_value_t* ea =
                    (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)a->data.ptr_val + 8);
                eshkol_tagged_value_t* eb =
                    (eshkol_tagged_value_t*)((uint8_t*)(uintptr_t)b->data.ptr_val + 8);
                for (int64_t i = 0; i < len_a; i++) {
                    if (!hash_keys_equal(&ea[i], &eb[i])) return false;
                }
                return true;
            }

            return a->data.ptr_val == b->data.ptr_val;
        }

        default:
            return a->data.raw_val == b->data.raw_val;
    }
}

/**
 * @brief Allocate a bare hash table (no object header) with a given initial capacity.
 *
 * Allocates the eshkol_hash_table_t control struct plus its keys/values/status
 * backing arrays from @p arena (zero-initialized arrays), and records @p arena
 * as the table's home_arena (ESH-0039) so later resizes always grow the
 * backing arrays in the arena the table was created in, not whatever arena
 * happens to be active at set!-time.
 *
 * @param arena             Arena to allocate the table and its arrays from.
 * @param initial_capacity  Number of key/value slots to allocate; must be > 0.
 * @return                  Newly allocated, empty hash table, or NULL on
 *                          invalid parameters or allocation failure.
 */
eshkol_hash_table_t* arena_allocate_hash_table(arena_t* arena, size_t initial_capacity) {
    if (!arena || initial_capacity == 0) {
        eshkol_error("Invalid parameters for hash table allocation");
        return nullptr;
    }

    eshkol_hash_table_t* table = (eshkol_hash_table_t*)
        arena_allocate_aligned(arena, sizeof(eshkol_hash_table_t), 8);
    if (!table) return nullptr;

    table->keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * initial_capacity);

    if (!table->keys || !table->values || !table->status) {
        eshkol_error("Failed to allocate hash table arrays");
        return nullptr;
    }

    table->capacity = initial_capacity;
    table->size = 0;
    table->tombstones = 0;
    table->home_arena = arena;  // ESH-0039: remember where the table lives

    return table;
}

/**
 * @brief Allocate a bare hash table using the runtime's default initial capacity.
 *
 * @param arena Arena to allocate the table and its arrays from.
 * @return      Newly allocated, empty hash table, or NULL on failure.
 */
eshkol_hash_table_t* arena_hash_table_create(arena_t* arena) {
    return arena_allocate_hash_table(arena, HASH_TABLE_INITIAL_CAPACITY);
}

/**
 * @brief Allocate a hash table prefixed with an eshkol_object_header_t (HEAP_SUBTYPE_HASH).
 *
 * Used when the table itself needs to be a first-class heap object (e.g.
 * referenced by a tagged HEAP_PTR value). Allocates the header + table struct
 * as one 8-byte-aligned block, then allocates the keys/values/status backing
 * arrays at the runtime's default initial capacity and records @p arena as
 * the table's home_arena (ESH-0039), mirroring arena_allocate_hash_table().
 *
 * @param arena Arena to allocate the header, table, and its arrays from.
 * @return      Pointer to the table struct (immediately after its header),
 *              or NULL on invalid parameters or allocation failure.
 */
eshkol_hash_table_t* arena_hash_table_create_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Invalid arena for hash table allocation");
        return nullptr;
    }

    size_t data_size = sizeof(eshkol_hash_table_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~7;

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate hash table with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_HASH;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    eshkol_hash_table_t* table =
        (eshkol_hash_table_t*)(mem + sizeof(eshkol_object_header_t));

    size_t initial_capacity = HASH_TABLE_INITIAL_CAPACITY;
    table->keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * initial_capacity);
    table->status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * initial_capacity);

    if (!table->keys || !table->values || !table->status) {
        eshkol_error("Failed to allocate hash table arrays");
        return nullptr;
    }

    table->capacity = initial_capacity;
    table->size = 0;
    table->tombstones = 0;
    table->home_arena = arena;  // ESH-0039: remember where the table lives

    return table;
}

/**
 * @brief Locate a key's slot (or its insertion point) using linear-probing open addressing.
 *
 * Starting from `hash(key) % capacity`, probes consecutive slots (wrapping
 * around the array) until an empty slot is found (key not present) or a
 * matching occupied slot is found. Deleted (tombstoned) slots are skipped
 * over but the first one encountered is recorded in @p tombstone_slot so
 * callers can reuse it on insert, keeping tombstones from accumulating
 * indefinitely.
 *
 * @param table           Hash table to search.
 * @param key             Key to look up.
 * @param tombstone_slot  Out param (may be NULL): set to the index of the
 *                         first tombstone seen along the probe sequence, or
 *                         -1 if none was seen or the key was found directly.
 * @return                Index of the slot holding an equal key, or -1 if
 *                         the key is not present.
 */
static int64_t find_slot(const eshkol_hash_table_t* table, const eshkol_tagged_value_t* key,
                         int64_t* tombstone_slot) {
    uint64_t hash = hash_tagged_value(key);
    size_t index = hash % table->capacity;
    int64_t first_tombstone = -1;

    for (size_t i = 0; i < table->capacity; i++) {
        size_t probe_index = (index + i) % table->capacity;
        uint8_t status = table->status[probe_index];

        if (status == HASH_ENTRY_EMPTY) {
            if (tombstone_slot) *tombstone_slot = first_tombstone;
            return -1;
        }

        if (status == HASH_ENTRY_DELETED) {
            if (first_tombstone == -1) {
                first_tombstone = (int64_t)probe_index;
            }
            continue;
        }

        if (hash_keys_equal(&table->keys[probe_index], key)) {
            if (tombstone_slot) *tombstone_slot = -1;
            return (int64_t)probe_index;
        }
    }

    if (tombstone_slot) *tombstone_slot = first_tombstone;
    return -1;
}

/** @brief RAII guard that takes the global hash-table lock on construction and releases it on destruction. */
struct hash_table_lock_guard {
    hash_table_lock_guard() { eshkol_hash_table_lock(); }
    ~hash_table_lock_guard() { eshkol_hash_table_unlock(); }
    hash_table_lock_guard(const hash_table_lock_guard&) = delete;
    hash_table_lock_guard& operator=(const hash_table_lock_guard&) = delete;
};

/**
 * @brief Grow the table's backing arrays to @p new_capacity and rehash all occupied entries.
 *
 * Always allocates the new keys/values/status arrays in the table's
 * home_arena (ESH-0039) rather than the @p arena passed in, so a table
 * created outside a `(with-region ...)` isn't corrupted by having its arrays
 * reallocated into a transient region arena that gets freed at region_pop.
 * Occupied slots are rehashed and reinserted via linear probing into the new
 * array (tombstones are dropped, not carried over, so `tombstones` resets to
 * 0). On allocation failure the original table is left untouched.
 *
 * @param arena        Arena hint; overridden by table->home_arena if set.
 * @param table        Table whose backing arrays are being grown in place.
 * @param new_capacity New slot-array capacity (must be able to hold all
 *                     currently occupied entries without a further resize).
 * @return             true on success; false if allocation of the new
 *                     arrays failed (table is left unmodified).
 */
static bool hash_table_resize(arena_t* arena, eshkol_hash_table_t* table, size_t new_capacity) {
    // ESH-0039: grow the backing arrays in the arena the table was BORN in, not
    // whatever transient region arena is active during this set!. Otherwise a
    // table created outside a (with-region ...) would have its arrays reallocated
    // into the region arena and freed (corrupted) at region_pop.
    if (table->home_arena) arena = table->home_arena;

    eshkol_tagged_value_t* new_keys = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * new_capacity);
    eshkol_tagged_value_t* new_values = (eshkol_tagged_value_t*)
        arena_allocate_zeroed(arena, sizeof(eshkol_tagged_value_t) * new_capacity);
    uint8_t* new_status = (uint8_t*)
        arena_allocate_zeroed(arena, sizeof(uint8_t) * new_capacity);

    if (!new_keys || !new_values || !new_status) {
        return false;
    }

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            uint64_t hash = hash_tagged_value(&table->keys[i]);
            size_t index = hash % new_capacity;

            while (new_status[index] != HASH_ENTRY_EMPTY) {
                index = (index + 1) % new_capacity;
            }

            new_keys[index] = table->keys[i];
            new_values[index] = table->values[i];
            new_status[index] = HASH_ENTRY_OCCUPIED;
        }
    }

    table->keys = new_keys;
    table->values = new_values;
    table->status = new_status;
    table->capacity = new_capacity;
    table->tombstones = 0;

    return true;
}

/**
 * @brief Insert or update a key/value pair in the hash table.
 *
 * Takes the global hash-table lock for the duration of the call. If the
 * current load factor (size + tombstones, over capacity) exceeds
 * HASH_TABLE_LOAD_FACTOR, the table is grown (doubled) before inserting. If
 * the key already exists, its value is overwritten in place; otherwise the
 * key is inserted into a reused tombstone slot if one was seen during the
 * probe, or the next empty slot via linear probing.
 *
 * @param arena Arena to allocate from if a resize is triggered (see
 *              hash_table_resize() for home_arena override behavior).
 * @param table Hash table to modify.
 * @param key   Key to insert or update.
 * @param value Value to associate with @p key.
 * @return      true on success; false if any parameter is NULL or a
 *              required resize allocation failed.
 */
bool hash_table_set(arena_t* arena, eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, const eshkol_tagged_value_t* value) {
    if (!arena || !table || !key || !value) return false;
    hash_table_lock_guard lock;

    double load = (double)(table->size + table->tombstones) / table->capacity;
    if (load > HASH_TABLE_LOAD_FACTOR) {
        if (!hash_table_resize(arena, table, table->capacity * 2)) {
            return false;
        }
    }

    int64_t tombstone_slot;
    int64_t slot = find_slot(table, key, &tombstone_slot);

    if (slot >= 0) {
        table->values[slot] = *value;
        return true;
    }

    size_t insert_index;
    if (tombstone_slot >= 0) {
        insert_index = (size_t)tombstone_slot;
        table->tombstones--;
    } else {
        uint64_t hash = hash_tagged_value(key);
        insert_index = hash % table->capacity;
        while (table->status[insert_index] != HASH_ENTRY_EMPTY) {
            insert_index = (insert_index + 1) % table->capacity;
        }
    }

    table->keys[insert_index] = *key;
    table->values[insert_index] = *value;
    table->status[insert_index] = HASH_ENTRY_OCCUPIED;
    table->size++;

    return true;
}

/**
 * @brief Look up a key in the hash table and copy its value out if present.
 *
 * @param table     Hash table to query.
 * @param key       Key to look up.
 * @param out_value Out param (may be NULL): receives a copy of the stored
 *                  value when the key is found.
 * @return          true if @p key is present in the table; false otherwise
 *                  (including NULL @p table or @p key).
 */
bool hash_table_get(const eshkol_hash_table_t* table,
                    const eshkol_tagged_value_t* key, eshkol_tagged_value_t* out_value) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;

    int64_t slot = find_slot(table, key, nullptr);
    if (slot < 0) return false;

    if (out_value) {
        *out_value = table->values[slot];
    }
    return true;
}

/**
 * @brief Test whether a key is present in the hash table, without retrieving its value.
 *
 * @param table Hash table to query.
 * @param key   Key to look up.
 * @return      true if @p key is present; false otherwise (including NULL
 *              @p table or @p key).
 */
bool hash_table_has_key(const eshkol_hash_table_t* table, const eshkol_tagged_value_t* key) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;
    return find_slot(table, key, nullptr) >= 0;
}

/**
 * @brief Remove a key/value pair from the hash table.
 *
 * Marks the key's slot as a tombstone (HASH_ENTRY_DELETED) rather than
 * empty, so later linear probes for other keys that hashed past this slot
 * still find them; the tombstone count is used by hash_table_set() to
 * decide when to resize.
 *
 * @param table Hash table to modify.
 * @param key   Key to remove.
 * @return      true if the key was present and removed; false otherwise.
 */
bool hash_table_remove(eshkol_hash_table_t* table, const eshkol_tagged_value_t* key) {
    if (!table || !key) return false;
    hash_table_lock_guard lock;

    int64_t slot = find_slot(table, key, nullptr);
    if (slot < 0) return false;

    table->status[slot] = HASH_ENTRY_DELETED;
    table->size--;
    table->tombstones++;

    return true;
}

/**
 * @brief Remove all entries from the hash table without shrinking its backing arrays.
 *
 * Resets every slot's status byte to HASH_ENTRY_EMPTY and zeroes size and
 * tombstones; the keys/values arrays and capacity are left as-is for reuse.
 *
 * @param table Hash table to clear (no-op if NULL).
 */
void hash_table_clear(eshkol_hash_table_t* table) {
    if (!table) return;
    hash_table_lock_guard lock;

    std::memset(table->status, HASH_ENTRY_EMPTY, table->capacity);
    table->size = 0;
    table->tombstones = 0;
}

/** @brief Return the number of live (non-tombstoned) entries in the hash table, or 0 if NULL. */
size_t hash_table_count(const eshkol_hash_table_t* table) {
    if (!table) return 0;
    hash_table_lock_guard lock;
    return table->size;
}

/**
 * @brief Build a tagged-cons list of all keys currently stored in the hash table.
 *
 * Walks the table's slot array in bucket order (not insertion order) and
 * allocates one new cons cell per occupied slot, chaining them into a
 * singly-linked list terminated by NULL. If a cons-cell allocation fails
 * partway through, the list built so far is returned rather than NULL.
 *
 * @param arena Arena to allocate the resulting cons cells from.
 * @param table Hash table to enumerate.
 * @return      Head of the newly built list of keys, or NULL if @p arena or
 *              @p table is NULL or the table is empty.
 */
arena_tagged_cons_cell_t* hash_table_keys(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table) return nullptr;
    hash_table_lock_guard lock;
    if (table->size == 0) return nullptr;

    arena_tagged_cons_cell_t* head = nullptr;
    arena_tagged_cons_cell_t* tail = nullptr;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
            if (!cell) return head;

            arena_tagged_cons_set_tagged_value(cell, false, &table->keys[i]);
            arena_tagged_cons_set_null(cell, true);

            if (!head) {
                head = tail = cell;
            } else {
                eshkol_tagged_value_t cell_val;
                cell_val.type = ESHKOL_VALUE_HEAP_PTR;
                cell_val.flags = 0;
                cell_val.reserved = 0;
                cell_val.data.ptr_val = (uint64_t)cell;
                arena_tagged_cons_set_tagged_value(tail, true, &cell_val);
                tail = cell;
            }
        }
    }

    return head;
}

/**
 * @brief Build a tagged-cons list of all values currently stored in the hash table.
 *
 * Same traversal and list-construction strategy as hash_table_keys(), but
 * collects each occupied slot's value rather than its key.
 *
 * @param arena Arena to allocate the resulting cons cells from.
 * @param table Hash table to enumerate.
 * @return      Head of the newly built list of values, or NULL if @p arena
 *              or @p table is NULL or the table is empty.
 */
arena_tagged_cons_cell_t* hash_table_values(arena_t* arena, const eshkol_hash_table_t* table) {
    if (!arena || !table) return nullptr;
    hash_table_lock_guard lock;
    if (table->size == 0) return nullptr;

    arena_tagged_cons_cell_t* head = nullptr;
    arena_tagged_cons_cell_t* tail = nullptr;

    for (size_t i = 0; i < table->capacity; i++) {
        if (table->status[i] == HASH_ENTRY_OCCUPIED) {
            arena_tagged_cons_cell_t* cell = arena_allocate_cons_with_header(arena);
            if (!cell) return head;

            arena_tagged_cons_set_tagged_value(cell, false, &table->values[i]);
            arena_tagged_cons_set_null(cell, true);

            if (!head) {
                head = tail = cell;
            } else {
                eshkol_tagged_value_t cell_val;
                cell_val.type = ESHKOL_VALUE_HEAP_PTR;
                cell_val.flags = 0;
                cell_val.reserved = 0;
                cell_val.data.ptr_val = (uint64_t)cell;
                arena_tagged_cons_set_tagged_value(tail, true, &cell_val);
                tail = cell;
            }
        }
    }

    return head;
}

}  // extern "C"
