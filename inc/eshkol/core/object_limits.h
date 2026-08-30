/*
 * Limits imposed by the shipped object-header representation.
 */
#ifndef ESHKOL_CORE_OBJECT_LIMITS_H
#define ESHKOL_CORE_OBJECT_LIMITS_H

#include <stdint.h>
#include <stddef.h>

/* eshkol_object_header_t::size and VmObjectHeader::size are uint32_t. */
#define ESHKOL_OBJECT_MAX_PAYLOAD_BYTES UINT32_MAX

static inline int eshkol_object_payload_fits(size_t payload_bytes) {
    return payload_bytes <= (size_t)ESHKOL_OBJECT_MAX_PAYLOAD_BYTES;
}

#endif
