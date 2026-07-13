/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Freestanding-safe string and UTF-8 runtime helpers.
 *
 * These helpers inspect Eshkol string headers, walk UTF-8 byte sequences, and
 * allocate substring results through the arena string allocator. They do not
 * depend on host files, environment state, process APIs, signals, or threads.
 */

#include <eshkol/eshkol.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

extern "C" {

// Stable hosted symbols used by the executable FFI/low-level language-surface
// conformance probe. Keeping them in the core runtime makes the extern-var and
// raw-pointer tests portable across every supported host linker.
uint64_t eshkol_runtime_ffi_surface_probe = 0;

const char* eshkol_runtime_ffi_surface_cstring(void) {
    return "ffi-surface";
}

extern void* arena_allocate_string_with_header(void* arena, uint64_t byte_len);

/** Copy a temporary C buffer into Eshkol's canonical header-tagged string shape. */
void* eshkol_runtime_copy_string(void* arena, const char* source) {
    if (!arena || !source) return nullptr;
    const uint64_t byte_len = static_cast<uint64_t>(std::strlen(source));
    char* result = static_cast<char*>(
        arena_allocate_string_with_header(arena, byte_len));
    if (!result) return nullptr;
    std::memcpy(result, source, static_cast<std::size_t>(byte_len) + 1);
    return result;
}

/* Public byte-count for an Eshkol string. Reads the header size field, which
 * includes the trailing NUL, and falls back to strlen for raw C strings. */
int64_t eshkol_string_byte_length(const char* s) {
    if (!s) return 0;

    const eshkol_object_header_t* hdr =
        (const eshkol_object_header_t*)((const uint8_t*)s - sizeof(eshkol_object_header_t));
    if (hdr->subtype == HEAP_SUBTYPE_STRING && hdr->size > 0) {
        return (int64_t)hdr->size - 1;
    }

    return (int64_t)std::strlen(s);
}

/**
 * @brief Count the number of UTF-8 codepoints (characters) in an Eshkol string.
 *
 * Scans the byte range reported by eshkol_string_byte_length and counts every
 * byte that is not a UTF-8 continuation byte (top two bits != 10), i.e. every
 * lead byte of a codepoint.
 *
 * @param s  Eshkol/C string (may be null, which yields 0).
 * @return   Number of codepoints.
 */
int64_t eshkol_utf8_strlen(const char* s) {
    if (!s) return 0;
    const int64_t byte_len = eshkol_string_byte_length(s);
    int64_t count = 0;
    for (int64_t i = 0; i < byte_len; i++) {
        if ((s[i] & 0xC0) != 0x80) count++;
    }
    return count;
}

/**
 * @brief Decode one UTF-8 codepoint starting at `*s`, advancing `*s` past it.
 *
 * Handles 1-4 byte sequences per the standard UTF-8 lead-byte patterns
 * (0xxxxxxx, 110xxxxx, 1110xxxx, 11110xxx). Malformed lead bytes decode to the
 * replacement codepoint U+FFFD and advance by one byte so callers make
 * forward progress on invalid input.
 *
 * @param s  In/out cursor into a UTF-8 byte sequence; advanced past the decoded codepoint.
 * @return   The decoded Unicode codepoint.
 */
static int64_t decode_utf8_codepoint(const char** s) {
    const unsigned char* p = (const unsigned char*)*s;
    int64_t cp;
    if (*p < 0x80) {
        cp = *p;
        *s += 1;
    } else if ((*p & 0xE0) == 0xC0) {
        cp = (*p & 0x1F) << 6 | (p[1] & 0x3F);
        *s += 2;
    } else if ((*p & 0xF0) == 0xE0) {
        cp = (*p & 0x0F) << 12 | (p[1] & 0x3F) << 6 | (p[2] & 0x3F);
        *s += 3;
    } else if ((*p & 0xF8) == 0xF0) {
        cp = (*p & 0x07) << 18 | (p[1] & 0x3F) << 12 |
             (p[2] & 0x3F) << 6 | (p[3] & 0x3F);
        *s += 4;
    } else {
        cp = 0xFFFD;
        *s += 1;
    }
    return cp;
}

/**
 * @brief Return the codepoint at UTF-8 character index `k` in string `s` (`string-ref` support).
 *
 * Walks byte-by-byte counting codepoint (lead-byte) boundaries until the k-th
 * one is reached, then decodes it. Runs in O(byte length), not O(1), since
 * UTF-8 codepoints are variable-width.
 *
 * @param s  Eshkol/C string (must be non-null).
 * @param k  Zero-based codepoint index.
 * @return   The codepoint at index k, or -1 if `s` is null, `k` is negative, or `k` is out of range.
 */
int64_t eshkol_utf8_ref(const char* s, int64_t k) {
    if (!s || k < 0) return -1;
    const int64_t byte_len = eshkol_string_byte_length(s);
    int64_t cp_idx = 0;
    int64_t i = 0;
    while (i < byte_len && cp_idx < k) {
        if ((s[i] & 0xC0) != 0x80) cp_idx++;
        i++;
    }
    if (i >= byte_len) return -1;
    const char* p = s + i;
    return decode_utf8_codepoint(&p);
}

/**
 * @brief Extract the UTF-8 substring spanning codepoint indices [start, end) (`substring` support).
 *
 * Walks `s` codepoint-by-codepoint (via the local advance_one_codepoint
 * lambda, which mirrors decode_utf8_codepoint's lead-byte width logic without
 * decoding values) to find the byte offsets corresponding to `start` and
 * `end`, then copies that byte range into a new arena-allocated,
 * NUL-terminated HEAP_SUBTYPE_STRING buffer via
 * arena_allocate_string_with_header.
 *
 * @param s      Source Eshkol/C string.
 * @param start  Start codepoint index (inclusive).
 * @param end    End codepoint index (exclusive); must be >= start.
 * @param arena  Arena to allocate the result string from.
 * @return       A new arena-owned string, or null if `s`/`arena` is null or the range is invalid.
 */
char* eshkol_utf8_substring(const char* s, int64_t start, int64_t end, void* arena) {
    if (!s || !arena || start < 0 || end < start) return nullptr;

    const int64_t byte_len_total = eshkol_string_byte_length(s);
    auto advance_one_codepoint = [&](int64_t& i) {
        if (i >= byte_len_total) return;
        const unsigned char b = (unsigned char)s[i];
        if ((b & 0x80) == 0) {
            i += 1;
        } else if ((b & 0xE0) == 0xC0) {
            i += 2;
        } else if ((b & 0xF0) == 0xE0) {
            i += 3;
        } else if ((b & 0xF8) == 0xF0) {
            i += 4;
        } else {
            i += 1;
        }
        if (i > byte_len_total) i = byte_len_total;
    };

    int64_t i = 0;
    int64_t cp_idx = 0;
    while (i < byte_len_total && cp_idx < start) {
        advance_one_codepoint(i);
        cp_idx++;
    }
    const int64_t start_off = i;
    while (i < byte_len_total && cp_idx < end) {
        advance_one_codepoint(i);
        cp_idx++;
    }

    const int64_t byte_len = i - start_off;
    char* buf = (char*)arena_allocate_string_with_header(arena, (uint64_t)byte_len);
    if (buf) {
        std::memcpy(buf, s + start_off, (std::size_t)byte_len);
        buf[byte_len] = '\0';
    }
    return buf;
}

}  // extern "C"
