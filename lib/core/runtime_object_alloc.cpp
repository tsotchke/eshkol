/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe header-aware object allocation helpers.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

// Allocate object with header prepended.
// Returns pointer to data (after header), or nullptr on failure.
void* arena_allocate_with_header(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags) {
    if (!arena || data_size == 0) {
        return nullptr;
    }

    // #192 CRITICAL: integer-overflow guard on total_size. If data_size
    // is close to SIZE_MAX, adding sizeof(header) + alignment rounding
    // wraps to a small value; arena_allocate_aligned then returns a
    // buffer much smaller than the caller expects. Callers often derive
    // data_size from file contents (kb-load) or tensor shapes
    // (user-controlled), so any wrap yields an immediate heap-overflow
    // primitive.
    if (data_size > SIZE_MAX - sizeof(eshkol_object_header_t) - 8) {
        eshkol_error("arena_allocate_with_header: data_size=%zu would overflow", data_size);
        return nullptr;
    }

    // #192 HIGH: header->size is uint32_t. Silently truncating a 4GB+
    // allocation to its low 32 bits makes downstream readers think the
    // object is tiny and under-copy it. Reject rather than truncate.
    if (data_size > UINT32_MAX) {
        eshkol_error("arena_allocate_with_header: data_size=%zu exceeds UINT32_MAX", data_size);
        return nullptr;
    }

    // Total size: header + data, aligned to 8 bytes.
    size_t total_size = sizeof(eshkol_object_header_t) + data_size;
    total_size = (total_size + 7) & ~((size_t)7);

    void* raw = arena_allocate_aligned(arena, total_size, 8);
    if (!raw) {
        eshkol_error("Failed to allocate object with header (size=%zu)", data_size);
        return nullptr;
    }

    eshkol_object_header_t* header = (eshkol_object_header_t*)raw;
    header->subtype = subtype;
    header->flags = flags;
    header->ref_count = 0;
    header->size = (uint32_t)data_size;

    return (void*)((uint8_t*)raw + sizeof(eshkol_object_header_t));
}

/**
 * @brief Like arena_allocate_with_header, but zero-fills the returned data region.
 *
 * @param arena      Arena to allocate from.
 * @param data_size  Size in bytes of the payload (excluding the header).
 * @param subtype    Heap object subtype to record in the header.
 * @param flags      Object flag bits to record in the header.
 * @return           Pointer to the zeroed data region (after the header), or nullptr on failure.
 */
void* arena_allocate_with_header_zeroed(arena_t* arena, size_t data_size, uint8_t subtype, uint8_t flags) {
    void* ptr = arena_allocate_with_header(arena, data_size, subtype, flags);
    if (ptr) {
        std::memset(ptr, 0, data_size);
    }
    return ptr;
}

/**
 * @brief Allocate a multiple-return-values container (HEAP_SUBTYPE_MULTI_VALUE).
 *
 * Layout is a leading size_t element count followed by `count` tagged
 * values, allocated via arena_allocate_with_header (so it carries an object
 * header with subtype HEAP_SUBTYPE_MULTI_VALUE). The element count is
 * written into the first `sizeof(size_t)` bytes of the returned region;
 * callers are responsible for filling in the tagged values that follow.
 * Guards against integer overflow in the size computation before
 * allocating.
 *
 * @param arena  Arena to allocate from.
 * @param count  Number of values the container will hold.
 * @return       Pointer to the container's data region (count header + values),
 *               or nullptr on overflow/allocation failure.
 */
void* arena_allocate_multi_value(arena_t* arena, size_t count) {
    if (count > (SIZE_MAX - sizeof(size_t)) / sizeof(eshkol_tagged_value_t)) {
        eshkol_error("integer overflow in multi-value allocation size (count=%zu)", count);
        return nullptr;
    }

    size_t data_size = sizeof(size_t) + count * sizeof(eshkol_tagged_value_t);
    void* ptr = arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_MULTI_VALUE, 0);
    if (ptr) {
        *((size_t*)ptr) = count;
    }
    return ptr;
}

/**
 * @brief Allocate a single cons cell (pair) with a heap object header.
 *
 * Allocates `sizeof(eshkol_object_header_t) + sizeof(arena_tagged_cons_cell_t)`
 * bytes, 16-byte aligned, tags the header with HEAP_SUBTYPE_CONS, and
 * initializes both car and cdr to null tagged values. The cell is
 * arena-owned and lives until the arena is freed/reset.
 *
 * @param arena  Arena to allocate from (must not be null).
 * @return       Newly allocated cons cell, or nullptr on failure.
 */
arena_tagged_cons_cell_t* arena_allocate_cons_with_header(arena_t* arena) {
    if (!arena) {
        eshkol_error("Cannot allocate cons with header: null arena");
        return nullptr;
    }

    size_t total = sizeof(eshkol_object_header_t) + sizeof(arena_tagged_cons_cell_t);
    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate cons cell with header");
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_CONS;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = sizeof(arena_tagged_cons_cell_t);

    arena_tagged_cons_cell_t* cell =
        (arena_tagged_cons_cell_t*)(mem + sizeof(eshkol_object_header_t));
    cell->car.type = ESHKOL_VALUE_NULL;
    cell->car.flags = 0;
    cell->car.reserved = 0;
    cell->car.data.raw_val = 0;
    cell->cdr.type = ESHKOL_VALUE_NULL;
    cell->cdr.flags = 0;
    cell->cdr.reserved = 0;
    cell->cdr.data.raw_val = 0;

    return cell;
}

/**
 * @brief Allocate a NUL-terminated string buffer with a heap object header.
 *
 * Allocates `length + 1` bytes for the string data (room for the trailing
 * NUL), 8-byte aligned, tagged with header subtype HEAP_SUBTYPE_STRING and
 * header->size set to `length + 1`. The buffer is pre-terminated (byte 0
 * set to '\0') but otherwise left uninitialized for the caller to fill in
 * up to `length` bytes. Guards against overflow of the header+data size
 * computation. The string is arena-owned and lives until the arena is
 * freed/reset.
 *
 * @param arena   Arena to allocate from (must not be null).
 * @param length  Number of characters in the string, excluding the NUL terminator.
 * @return        Pointer to the (empty, NUL-terminated) string data, or nullptr on failure.
 */
char* arena_allocate_string_with_header(arena_t* arena, size_t length) {
    if (!arena) {
        eshkol_error("Cannot allocate string with header: null arena");
        return nullptr;
    }

    if (length >= SIZE_MAX - sizeof(eshkol_object_header_t) - 8) {
        eshkol_error("String length overflow (length=%zu)", length);
        return nullptr;
    }

    size_t data_size = length + 1;
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~((size_t)7);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate string with header (length=%zu)", length);
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_STRING;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    char* str = (char*)(mem + sizeof(eshkol_object_header_t));
    str[0] = '\0';

    return str;
}

/**
 * @brief Allocate a heterogeneous vector buffer with a heap object header.
 *
 * Layout is an 8-byte length/metadata slot followed by `capacity` tagged
 * values, allocated 16-byte aligned via a single header+data block tagged
 * with HEAP_SUBTYPE_VECTOR. Guards against overflow of the capacity-to-bytes
 * computation before allocating. Contents beyond the header are left
 * uninitialized — the caller is responsible for filling in the length slot
 * and the tagged-value elements. The buffer is arena-owned and lives until
 * the arena is freed/reset.
 *
 * @param arena     Arena to allocate from (must not be null).
 * @param capacity  Number of tagged-value elements to reserve space for.
 * @return          Pointer to the vector's data region (length slot + elements),
 *                  or nullptr on overflow/allocation failure.
 */
void* arena_allocate_vector_with_header(arena_t* arena, size_t capacity) {
    if (!arena) {
        eshkol_error("Cannot allocate vector with header: null arena");
        return nullptr;
    }

    if (capacity > (SIZE_MAX - 8 - sizeof(eshkol_object_header_t)) / sizeof(eshkol_tagged_value_t)) {
        eshkol_error("Vector capacity overflow (capacity=%zu)", capacity);
        return nullptr;
    }

    size_t data_size = 8 + capacity * sizeof(eshkol_tagged_value_t);
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 15) & ~((size_t)15);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 16);
    if (!mem) {
        eshkol_error("Failed to allocate vector with header (capacity=%zu)", capacity);
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_VECTOR;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    return mem + sizeof(eshkol_object_header_t);
}

/**
 * @brief Allocate an interned-symbol buffer with a heap object header.
 *
 * Layout is an 8-byte metadata slot (e.g. cached hash/length) followed by
 * `length + 1` bytes of NUL-terminated symbol-name storage, allocated
 * 8-byte aligned via a single header+data block tagged with
 * HEAP_SUBTYPE_SYMBOL. Unlike arena_allocate_string_with_header, this does
 * not guard the size computation against overflow before allocating.
 * Contents beyond the header are left uninitialized for the caller to fill
 * in. The buffer is arena-owned and lives until the arena is freed/reset.
 *
 * @param arena   Arena to allocate from (must not be null).
 * @param length  Number of characters in the symbol name, excluding the NUL terminator.
 * @return        Pointer to the symbol's data region (metadata slot + name bytes),
 *                or nullptr on allocation failure.
 */
void* arena_allocate_symbol_with_header(arena_t* arena, size_t length) {
    if (!arena) {
        eshkol_error("Cannot allocate symbol with header: null arena");
        return nullptr;
    }

    size_t data_size = 8 + length + 1;
    size_t total = sizeof(eshkol_object_header_t) + data_size;
    total = (total + 7) & ~((size_t)7);

    uint8_t* mem = (uint8_t*)arena_allocate_aligned(arena, total, 8);
    if (!mem) {
        eshkol_error("Failed to allocate symbol with header (length=%zu)", length);
        return nullptr;
    }

    eshkol_object_header_t* hdr = (eshkol_object_header_t*)mem;
    hdr->subtype = HEAP_SUBTYPE_SYMBOL;
    hdr->flags = 0;
    hdr->ref_count = 0;
    hdr->size = (uint32_t)data_size;

    return mem + sizeof(eshkol_object_header_t);
}
