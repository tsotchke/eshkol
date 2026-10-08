#ifndef ESHKOL_ESKM_V2_EXPERIMENTAL_H
#define ESHKOL_ESKM_V2_EXPERIMENTAL_H

/* Private, provisional hosted-engine policy. Wire validity is kept separate. */
#include "eskm_v2_preflight.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ESKM_V2_BACKEND_RECORDS UINT64_C(4096)
#define ESKM_V2_BACKEND_NAME UINT64_C(4096)
#define ESKM_V2_BACKEND_RANK UINT64_C(8)
#define ESKM_V2_BACKEND_ELEMENTS UINT64_C(16777216)
#define ESKM_V2_BACKEND_MEMORY UINT64_C(536870912)

/* 0: ordinary v1; 1: experimental reader; 2: experimental reader/writer.
 * Unknown values fail closed, including requests in builds without the option. */
static inline int eskm_v2_mode(void) {
    const char* mode = getenv("ESHKOL_EXPERIMENTAL_ESKM_V2");
    if (!mode || !*mode) return 0;
#ifdef ESHKOL_ENABLE_EXPERIMENTAL_ESKM_V2
    if (strcmp(mode, "read") == 0) return 1;
    if (strcmp(mode, "write") == 0) return 2;
#endif
    fprintf(stderr, "ERROR: experimental ESKM v2 requires an enabled build and mode read or write\n");
    return -1;
}

static inline uint64_t eskm_v2_uint(const unsigned char* p, unsigned n) {
    uint64_t value = 0;
    for (unsigned i = 0; i < n; ++i) value |= (uint64_t)p[i] << (8 * i);
    return value;
}

/* Peek before any whole-file allocation. v1 admission remains unchanged. */
static inline int eskm_v2_file_admit(FILE* file, uint64_t size) {
    unsigned char header[8];
    int mode = eskm_v2_mode();
    if (mode < 0) return 0;
    size_t got = fread(header, 1, sizeof(header), file);
    if (fseek(file, 0, SEEK_SET) != 0) return 0;
    if (got != sizeof(header) || memcmp(header, "ESKM", 4) != 0) return 0;
    uint64_t version = eskm_v2_uint(header + 4, 4);
    if (version == 2) return mode > 0 && size <= ESKM_V2_MAX_BUFFER_BYTES;
    return version == 1;
}

typedef struct eskm_v2_backend_limits {
    uint64_t records, name_bytes, rank, elements, memory_bytes;
} eskm_v2_backend_limits;
#define ESKM_V2_BACKEND_DEFAULT { ESKM_V2_BACKEND_RECORDS, ESKM_V2_BACKEND_NAME, \
    ESKM_V2_BACKEND_RANK, ESKM_V2_BACKEND_ELEMENTS, ESKM_V2_BACKEND_MEMORY }

typedef struct eskm_v2_budget {
    uint64_t elements, names, records, wire_bytes, materialized_bytes, charged_bytes;
} eskm_v2_budget;

static inline int eskm_v2_account(eskm_v2_budget* budget,
                                 const eskm_v2_backend_limits* limits,
                                 uint64_t name, uint64_t rank,
                                 const uint64_t* dims, uint64_t* count) {
    if (name > limits->name_bytes || rank > limits->rank ||
        budget->records >= limits->records) return 0;
    uint64_t total = 1, stride = 1;
    for (uint64_t i = rank; i > 0; --i) {
        uint64_t dim = dims[i - 1];
        if (dim > INT64_MAX) return 0;
        if (i > 1) {
            if (dim && stride > (uint64_t)INT64_MAX / dim) return 0;
            stride *= dim;
        }
    }
    for (uint64_t i = 0; i < rank; ++i) {
        if (dims[i] && total > (uint64_t)INT64_MAX / dims[i]) return 0;
        total *= dims[i];
    }
    if (total > limits->elements || budget->elements > limits->elements - total) return 0;
    budget->elements += total;
    budget->names += name;
    ++budget->records;
    budget->wire_bytes += 9 + name + rank * 8 + total * 8;
    /* Includes alignment, tensor/string/list headers and VM slot bookkeeping.
     * 4 payload/name copies cover native staging and VM block over-allocation;
     * this is checkpoint incremental memory, not a whole-process RSS limit. */
    budget->materialized_bytes = budget->records * 4096 + budget->names + budget->elements * 8;
    uint64_t memory = budget->wire_bytes + budget->elements * 32 +
                      budget->names * 4 + budget->records * 8192 + 16384;
    budget->charged_bytes = memory;
    if (budget->wire_bytes > ESKM_V2_MAX_BUFFER_BYTES || memory > limits->memory_bytes) return 0;
    if (count) *count = total;
    return 1;
}

#ifdef ESHKOL_ENABLE_EXPERIMENTAL_ESKM_V2
/* Full preflight then backend admission, before any proportional allocation. */
static inline int eskm_v2_backend_admit(const unsigned char* bytes, size_t size,
                                      const eskm_v2_backend_limits* limits,
                                      eskm_v2_result* result, eskm_v2_budget* budget) {
    eskm_v2_limits parser_limits = ESKM_V2_LIMITS_DEFAULT;
    eskm_v2_workspace workspace;
    if (!limits || limits->records > ESKM_V2_BACKEND_RECORDS ||
        limits->name_bytes > ESKM_V2_BACKEND_NAME || limits->rank > ESKM_V2_BACKEND_RANK ||
        limits->elements > ESKM_V2_BACKEND_ELEMENTS || limits->memory_bytes > ESKM_V2_BACKEND_MEMORY)
        return 0;
    if (eskm_v2_preflight(bytes, size, &parser_limits, &workspace, result).status != ESKM_V2_OK)
        return 0;
    if (result->record_count > limits->records) return 0;
    memset(budget, 0, sizeof(*budget));
    budget->wire_bytes = 28 + result->extension_bytes;
    budget->charged_bytes = budget->wire_bytes + 16384;
    if (budget->charged_bytes > limits->memory_bytes) return 0;
    size_t cursor = (size_t)result->records_offset;
    for (uint32_t i = 0; i < result->record_count; ++i) {
        uint64_t name = eskm_v2_uint(bytes + cursor, 4);
        cursor += 4;
        /* Native Scheme strings use NUL termination at the writer boundary. */
        if (name > limits->name_bytes || memchr(bytes + cursor, 0, (size_t)name)) return 0;
        cursor += (size_t)name;
        uint64_t rank = eskm_v2_uint(bytes + cursor, 4), dims[8], count;
        cursor += 4;
        if (rank > limits->rank) return 0;
        for (uint64_t d = 0; d < rank; ++d, cursor += 8) dims[d] = eskm_v2_uint(bytes + cursor, 8);
        if (!eskm_v2_account(budget, limits, name, rank, dims, &count)) return 0;
        cursor += 1 + (size_t)count * 8;
    }
    return cursor == size - 4;
}
#endif
#endif
