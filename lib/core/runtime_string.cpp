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

extern void* arena_allocate_string_with_header(void* arena, uint64_t byte_len);

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

int64_t eshkol_utf8_strlen(const char* s) {
    if (!s) return 0;
    const int64_t byte_len = eshkol_string_byte_length(s);
    int64_t count = 0;
    for (int64_t i = 0; i < byte_len; i++) {
        if ((s[i] & 0xC0) != 0x80) count++;
    }
    return count;
}

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
