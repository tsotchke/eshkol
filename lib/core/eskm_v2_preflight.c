#include "eskm_v2_preflight.h"

#include <string.h>

static eskm_v2_error fail(eskm_v2_status status, uint64_t offset) {
    eskm_v2_error error = {status, offset};
    return error;
}

static uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 |
           (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static uint64_t read_u64(const uint8_t *p) {
    return (uint64_t)read_u32(p) | (uint64_t)read_u32(p + 4) << 32;
}

static uint32_t crc32(const uint8_t *input, size_t size) {
    uint32_t crc = UINT32_MAX;
    for (size_t i = 0; i < size; ++i) {
        crc ^= input[i];
        for (unsigned bit = 0; bit < 8; ++bit)
            crc = (crc >> 1) ^ (UINT32_C(0xedb88320) & (0u - (crc & 1u)));
    }
    return ~crc;
}

static int key_compare(const uint8_t *input, eskm_v2_key_span a,
                       eskm_v2_key_span b) {
    size_t common = (size_t)(a.length < b.length ? a.length : b.length);
    int order = memcmp(input + (size_t)a.offset, input + (size_t)b.offset, common);
    if (order != 0) return order;
    return (a.length > b.length) - (a.length < b.length);
}

/* Offset breaks equal-key ties, so adjacent duplicates remain in wire order. */
static int span_compare(const uint8_t *input, eskm_v2_key_span a,
                        eskm_v2_key_span b) {
    int order = key_compare(input, a, b);
    return order != 0 ? order : (a.offset > b.offset) - (a.offset < b.offset);
}

static void sift_down(const uint8_t *input, eskm_v2_key_span *keys,
                      uint32_t count, uint32_t root) {
    while (root < count / 2) {
        uint32_t child = root * 2 + 1;
        if (child + 1 < count && span_compare(input, keys[child], keys[child + 1]) < 0)
            ++child;
        if (span_compare(input, keys[root], keys[child]) >= 0) return;
        eskm_v2_key_span swap = keys[root];
        keys[root] = keys[child];
        keys[child] = swap;
        root = child;
    }
}

static uint64_t duplicate_key(const uint8_t *input, eskm_v2_key_span *keys,
                              uint32_t count) {
    for (uint32_t i = count / 2; i > 0; --i)
        sift_down(input, keys, count, i - 1);
    for (uint32_t i = count; i > 1; --i) {
        eskm_v2_key_span swap = keys[0];
        keys[0] = keys[i - 1];
        keys[i - 1] = swap;
        sift_down(input, keys, i - 1, 0);
    }
    uint64_t duplicate = UINT64_MAX;
    for (uint32_t i = 1; i < count; ++i) {
        if (key_compare(input, keys[i - 1], keys[i]) == 0 && keys[i].offset < duplicate)
            duplicate = keys[i].offset;
    }
    return duplicate;
}

static eskm_v2_error annotations(const uint8_t *input, size_t start, size_t end,
                                const eskm_v2_limits *limits,
                                eskm_v2_workspace *workspace, uint32_t *count_out) {
    if (end - start < 4) return fail(ESKM_V2_MALFORMED, start);
    uint32_t count = read_u32(input + start);
    if (count > limits->annotations) return fail(ESKM_V2_RESOURCE_LIMIT, start);
    size_t cursor = start + 4;
    for (uint32_t i = 0; i < count; ++i) {
        size_t field = cursor;
        if (end - cursor < 4) return fail(ESKM_V2_MALFORMED, field);
        uint32_t key_bytes = read_u32(input + cursor);
        if (key_bytes == 0) return fail(ESKM_V2_MALFORMED, field);
        if (key_bytes > limits->key_bytes) return fail(ESKM_V2_RESOURCE_LIMIT, field);
        cursor += 4;
        if (end - cursor < 4) return fail(ESKM_V2_MALFORMED, cursor);
        uint32_t value_bytes = read_u32(input + cursor);
        if (value_bytes > limits->value_bytes)
            return fail(ESKM_V2_RESOURCE_LIMIT, cursor);
        cursor += 4;
        if (key_bytes > end - cursor) return fail(ESKM_V2_MALFORMED, field);
        workspace->keys[i].offset = cursor;
        workspace->keys[i].length = key_bytes;
        cursor += key_bytes;
        if (value_bytes > end - cursor) return fail(ESKM_V2_MALFORMED, field + 4);
        cursor += value_bytes;
    }
    if (cursor != end) return fail(ESKM_V2_MALFORMED, cursor);
    uint64_t duplicate = duplicate_key(input, workspace->keys, count);
    if (duplicate != UINT64_MAX) return fail(ESKM_V2_MALFORMED, duplicate - 8);
    *count_out = count;
    return fail(ESKM_V2_OK, 0);
}

static int valid_limits(const eskm_v2_limits *limits) {
    return limits->buffer_bytes <= ESKM_V2_MAX_BUFFER_BYTES &&
           limits->extension_bytes <= ESKM_V2_MAX_EXTENSION_BYTES &&
           limits->tlvs <= ESKM_V2_MAX_TLVS &&
           limits->annotations <= ESKM_V2_MAX_ANNOTATIONS &&
           limits->key_bytes <= ESKM_V2_MAX_KEY_BYTES &&
           limits->value_bytes <= ESKM_V2_MAX_VALUE_BYTES;
}

eskm_v2_error eskm_v2_preflight(const uint8_t *input, size_t size,
                              const eskm_v2_limits *limits,
                              eskm_v2_workspace *workspace,
                              eskm_v2_result *result) {
    if (result != NULL) memset(result, 0, sizeof(*result));
    if (input == NULL || limits == NULL || workspace == NULL || result == NULL)
        return fail(ESKM_V2_INVALID_ARGUMENT, 0);
    if (!valid_limits(limits)) return fail(ESKM_V2_INVALID_LIMITS, 0);
    if (size > limits->buffer_bytes) return fail(ESKM_V2_RESOURCE_LIMIT, 0);
    if (size < 28) return fail(ESKM_V2_MALFORMED, 0);
    if (memcmp(input, "ESKM", 4) != 0) return fail(ESKM_V2_BAD_MAGIC, 0);
    if (read_u32(input + 4) != 2) return fail(ESKM_V2_UNSUPPORTED_VERSION, 4);
    if (read_u32(input + 12) != 0) return fail(ESKM_V2_UNSUPPORTED_FEATURES, 12);
    size_t end = size - 4;
    if (crc32(input, end) != read_u32(input + end))
        return fail(ESKM_V2_CHECKSUM_MISMATCH, end);

    eskm_v2_result parsed = {0};
    parsed.record_count = read_u32(input + 8);
    parsed.extension_offset = 24;
    parsed.extension_bytes = read_u64(input + 16);
    if (parsed.extension_bytes > limits->extension_bytes)
        return fail(ESKM_V2_RESOURCE_LIMIT, 16);
    if (parsed.extension_bytes > end - 24) return fail(ESKM_V2_MALFORMED, 16);
    size_t extension_end = 24 + (size_t)parsed.extension_bytes;
    size_t cursor = 24;
    while (cursor < extension_end) {
        size_t type_field = cursor;
        if (parsed.tlv_count >= limits->tlvs)
            return fail(ESKM_V2_RESOURCE_LIMIT, type_field);
        if (extension_end - cursor < 4) return fail(ESKM_V2_MALFORMED, cursor);
        uint32_t type = read_u32(input + cursor);
        if (type == 0) return fail(ESKM_V2_MALFORMED, type_field);
        if (type == 1 && parsed.has_annotations)
            return fail(ESKM_V2_MALFORMED, type_field);
        cursor += 4;
        if (extension_end - cursor < 4) return fail(ESKM_V2_MALFORMED, cursor);
        uint32_t length = read_u32(input + cursor);
        cursor += 4;
        if (length > extension_end - cursor)
            return fail(ESKM_V2_MALFORMED, type_field + 4);
        if (type == 1) {
            eskm_v2_error error = annotations(input, cursor, cursor + length,
                                            limits, workspace, &parsed.annotation_count);
            if (error.status != ESKM_V2_OK) return error;
            parsed.has_annotations = 1;
        }
        cursor += length;
        ++parsed.tlv_count;
    }
    parsed.records_offset = cursor;
    for (uint32_t record = 0; record < parsed.record_count; ++record) {
        size_t name_field = cursor;
        if (end - cursor < 4) return fail(ESKM_V2_MALFORMED, cursor);
        uint32_t name_bytes = read_u32(input + cursor);
        cursor += 4;
        if (name_bytes > end - cursor) return fail(ESKM_V2_MALFORMED, name_field);
        cursor += name_bytes;
        size_t rank_field = cursor;
        if (end - cursor < 4) return fail(ESKM_V2_MALFORMED, cursor);
        uint32_t rank = read_u32(input + cursor);
        cursor += 4;
        if (rank > (end - cursor) / 8) return fail(ESKM_V2_MALFORMED, rank_field);
        uint64_t elements = 1;
        for (uint32_t dim = 0; dim < rank; ++dim) {
            uint64_t dimension = read_u64(input + cursor);
            if (elements != 0) {
                if (dimension != 0 && elements > UINT64_MAX / dimension)
                    return fail(ESKM_V2_MALFORMED, cursor);
                elements *= dimension;
            }
            cursor += 8;
        }
        size_t dtype_field = cursor;
        if (cursor == end || input[cursor] != 0)
            return fail(ESKM_V2_MALFORMED, dtype_field);
        ++cursor;
        if (elements > UINT64_MAX / 8 || elements * 8 > end - cursor)
            return fail(ESKM_V2_MALFORMED, dtype_field);
        cursor += (size_t)(elements * 8);
    }
    if (cursor != end) return fail(ESKM_V2_MALFORMED, cursor);
    parsed.records_bytes = cursor - parsed.records_offset;
    *result = parsed;
    return fail(ESKM_V2_OK, 0);
}
