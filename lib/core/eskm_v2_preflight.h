#ifndef ESHKOL_ESKM_V2_PREFLIGHT_H
#define ESHKOL_ESKM_V2_PREFLIGHT_H

/* Experimental private contract; not a public v2 loading API. */
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESKM_V2_MAX_BUFFER_BYTES UINT64_C(268435456)
#define ESKM_V2_MAX_EXTENSION_BYTES UINT64_C(1048576)
#define ESKM_V2_MAX_TLVS UINT64_C(1024)
#define ESKM_V2_MAX_ANNOTATIONS UINT64_C(1024)
#define ESKM_V2_MAX_KEY_BYTES UINT64_C(4096)
#define ESKM_V2_MAX_VALUE_BYTES UINT64_C(65536)

/* Limits are inclusive. Zero means zero; limits cannot exceed these ceilings. */
typedef struct eskm_v2_limits {
    uint64_t buffer_bytes;
    uint64_t extension_bytes;
    uint64_t tlvs;
    uint64_t annotations;
    uint64_t key_bytes;
    uint64_t value_bytes;
} eskm_v2_limits;

#define ESKM_V2_LIMITS_DEFAULT \
    { ESKM_V2_MAX_BUFFER_BYTES, ESKM_V2_MAX_EXTENSION_BYTES, \
      ESKM_V2_MAX_TLVS, ESKM_V2_MAX_ANNOTATIONS, \
      ESKM_V2_MAX_KEY_BYTES, ESKM_V2_MAX_VALUE_BYTES }

typedef struct eskm_v2_key_span {
    uint64_t offset;
    uint64_t length;
} eskm_v2_key_span;

/* Exactly 16 KiB; contents are scratch and unspecified after any call. */
typedef struct eskm_v2_workspace {
    eskm_v2_key_span keys[1024];
} eskm_v2_workspace;

typedef enum eskm_v2_status {
    ESKM_V2_OK = 0,
    ESKM_V2_INVALID_ARGUMENT,
    ESKM_V2_INVALID_LIMITS,
    ESKM_V2_BAD_MAGIC,
    ESKM_V2_UNSUPPORTED_VERSION,
    ESKM_V2_UNSUPPORTED_FEATURES,
    ESKM_V2_CHECKSUM_MISMATCH,
    ESKM_V2_MALFORMED,
    ESKM_V2_RESOURCE_LIMIT
} eskm_v2_status;

typedef struct eskm_v2_error {
    eskm_v2_status status;
    uint64_t offset;
} eskm_v2_error;

/* Borrowed byte ranges into the original immutable buffer, excluding the CRC. */
typedef struct eskm_v2_result {
    uint64_t extension_offset;
    uint64_t extension_bytes;
    uint64_t records_offset;
    uint64_t records_bytes;
    uint32_t record_count;
    uint32_t tlv_count;
    uint32_t annotation_count;
    uint32_t has_annotations;
} eskm_v2_result;

/* All pointers must be nonnull, even for a zero-sized input. Input, limits,
 * workspace and result objects must be disjoint and correctly aligned for their
 * types. The caller must keep input alive and immutable while using its ranges.
 * No heap allocation, callbacks, recursion or tensor materialization occurs.
 * On failure, result is entirely zeroed (when nonnull). The returned offset is
 * the offending field's first byte; argument/limit errors use zero. No partial
 * view is published. ESKM_V2_OK always has offset zero.
 *
 * Order: arguments/limits, buffer cap/minimum size, magic/version/features, CRC,
 * then extensions and records in wire order. A missing field reports its start;
 * an invalid length reports the length field. Duplicate keys report the later
 * key's length field; duplicate type 1 reports its type field. Shape overflow
 * reports the dimension field; payload size/bounds errors report the dtype.
 * Duplicate checking follows complete validation of the annotation entries.
 */
eskm_v2_error eskm_v2_preflight(const uint8_t *input, size_t size,
                              const eskm_v2_limits *limits,
                              eskm_v2_workspace *workspace,
                              eskm_v2_result *result);

#ifdef __cplusplus
}
static_assert(sizeof(eskm_v2_workspace) == 16384, "ESKM v2 workspace budget");
#else
_Static_assert(sizeof(eskm_v2_workspace) == 16384, "ESKM v2 workspace budget");
#endif

#endif
