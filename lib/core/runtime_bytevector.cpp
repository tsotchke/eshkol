/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe bytevector runtime helpers.
 *
 * These helpers allocate and access Eshkol bytevector payloads through the
 * arena object allocator. They do not depend on host files, environment state,
 * process APIs, signals, or threads. Error reporting still delegates to the
 * shared runtime fatal path until the fatal/panic hook ABI is split.
 */

#include <eshkol/eshkol.h>

#include <cstdint>
#include <cstring>

extern "C" {

extern void* arena_allocate_with_header(void* arena, uint64_t data_size,
                                        uint8_t subtype, uint8_t flags);
extern void eshkol_runtime_fatal(eshkol_exception_type_t type, const char* fmt, ...);

/* Create a new bytevector of given length, filled with fill_byte. */
void* eshkol_make_bytevector(void* arena, int64_t len, int64_t fill_byte) {
    if (!arena || len < 0) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in make-bytevector: invalid arguments (len=%lld)",
                             (long long)len);
    }

    const uint64_t data_size = (uint64_t)(8 + len);
    void* ptr = arena_allocate_with_header(arena, data_size, HEAP_SUBTYPE_BYTEVECTOR, 0);
    if (!ptr) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in make-bytevector: out of memory (len=%lld)",
                             (long long)len);
    }

    int64_t* length_ptr = (int64_t*)ptr;
    *length_ptr = len;

    uint8_t* data = (uint8_t*)ptr + 8;
    const uint8_t fill = (uint8_t)(fill_byte & 0xFF);
    std::memset(data, fill, (size_t)len);

    return ptr;
}

/* Get byte at index k (returns int64_t for tagged value compatibility). */
int64_t eshkol_bytevector_u8_ref(void* bv, int64_t k) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-u8-ref: null bytevector");
    }

    const int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-ref: index %lld out of range [0, %lld)",
                             (long long)k, (long long)len);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    return (int64_t)data[k];
}

/* Set byte at index k. */
void eshkol_bytevector_u8_set(void* bv, int64_t k, int64_t byte_val) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-u8-set!: null bytevector");
    }

    const int64_t len = *((int64_t*)bv);
    if (k < 0 || k >= len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-set!: index %lld out of range [0, %lld)",
                             (long long)k, (long long)len);
    }

    if (byte_val < 0 || byte_val > 255) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-u8-set!: byte value %lld out of range [0, 255]",
                             (long long)byte_val);
    }

    uint8_t* data = (uint8_t*)bv + 8;
    data[k] = (uint8_t)byte_val;
}

/* Get length of bytevector. */
int64_t eshkol_bytevector_length(void* bv) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-length: null bytevector");
    }

    return *((int64_t*)bv);
}

/* Copy bytevector or sub-range into a new arena-allocated bytevector. */
void* eshkol_bytevector_copy(void* arena, void* bv, int64_t start, int64_t end) {
    if (!bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: null bytevector");
    }
    if (!arena) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: null arena");
    }

    const int64_t len = *((int64_t*)bv);

    if (end < 0) end = len;

    if (start < 0 || start > len || end < start || end > len) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_RANGE_ERROR,
                             "Error in bytevector-copy: range [%lld, %lld) out of bounds [0, %lld)",
                             (long long)start, (long long)end, (long long)len);
    }

    const int64_t copy_len = end - start;
    const uint64_t data_size = (uint64_t)(8 + copy_len);
    void* new_bv = arena_allocate_with_header(
        arena, data_size, HEAP_SUBTYPE_BYTEVECTOR, 0);
    if (!new_bv) {
        eshkol_runtime_fatal(ESHKOL_EXCEPTION_ERROR,
                             "Error in bytevector-copy: out of memory (len=%lld)",
                             (long long)copy_len);
    }

    *((int64_t*)new_bv) = copy_len;

    if (copy_len > 0) {
        uint8_t* src_data = (uint8_t*)bv + 8 + start;
        uint8_t* dst_data = (uint8_t*)new_bv + 8;
        std::memcpy(dst_data, src_data, (size_t)copy_len);
    }

    return new_bv;
}

}  // extern "C"
