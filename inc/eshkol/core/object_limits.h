/*
 * Limits imposed by the shipped object-header representation.
 */
#ifndef ESHKOL_CORE_OBJECT_LIMITS_H
#define ESHKOL_CORE_OBJECT_LIMITS_H

#include <stdint.h>
#include <stddef.h>

/* eshkol_object_header_t::size and VmObjectHeader::size are uint32_t. */
#define ESHKOL_OBJECT_MAX_PAYLOAD_BYTES UINT32_MAX

/* A vector has an 8-byte length slot followed by 16-byte tagged values.  Keep
 * the element-count boundary below the first value whose payload would exceed
 * the header representation.  The strict predicate is shared by every
 * vector-producing path so the boundary itself is rejected before allocation.
 */
#define ESHKOL_MAX_VECTOR_CAPACITY (UINT32_C(1) << 28)

static inline int eshkol_object_payload_fits(size_t payload_bytes) {
    return payload_bytes <= (size_t)ESHKOL_OBJECT_MAX_PAYLOAD_BYTES;
}

static inline int eshkol_vector_capacity_fits(size_t capacity) {
    return capacity < (size_t)ESHKOL_MAX_VECTOR_CAPACITY;
}

#endif
