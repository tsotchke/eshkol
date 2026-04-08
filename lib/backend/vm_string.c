/**
 * @file vm_string.c
 * @brief UTF-8 string operations for the Eshkol bytecode VM.
 *
 * Implements R7RS string operations with full UTF-8 support.
 * Strings are immutable: mutation returns a new string.
 * All allocation via OALR arena (vm_arena.h), no GC.
 *
 * Native call IDs: 550-579
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_STRING_C_INCLUDED
#define VM_STRING_C_INCLUDED

#include "vm_numeric.h"
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <stdlib.h>
#include <errno.h>

/* ── VmString Type ── */

typedef struct {
    int byte_len;
    int char_len;    /* codepoint count */
    char* data;      /* arena-allocated, null-terminated */
} VmString;

/* Replacement character for invalid UTF-8 */
#define VM_UNICODE_REPLACEMENT 0xFFFD

/* ── UTF-8 Decode ──
 * Decodes one codepoint from buf[pos..byte_len), advances pos.
 * Returns codepoint, or U+FFFD on invalid sequence.
 */
static int vm_utf8_decode(const char* buf, int byte_len, int* pos) {
    if (*pos >= byte_len) return -1;
    unsigned char b0 = (unsigned char)buf[*pos];

    /* 1-byte ASCII: 0xxxxxxx */
    if (b0 < 0x80) {
        (*pos)++;
        return (int)b0;
    }

    int cp;
    int expect; /* expected continuation bytes */

    if ((b0 & 0xE0) == 0xC0) {
        /* 2-byte: 110xxxxx 10xxxxxx */
        cp = b0 & 0x1F;
        expect = 1;
        if (cp < 2) { (*pos)++; return VM_UNICODE_REPLACEMENT; } /* overlong */
    } else if ((b0 & 0xF0) == 0xE0) {
        /* 3-byte: 1110xxxx 10xxxxxx 10xxxxxx */
        cp = b0 & 0x0F;
        expect = 2;
    } else if ((b0 & 0xF8) == 0xF0) {
        /* 4-byte: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx */
        cp = b0 & 0x07;
        expect = 3;
    } else {
        /* Invalid lead byte */
        (*pos)++;
        return VM_UNICODE_REPLACEMENT;
    }

    (*pos)++;
    for (int i = 0; i < expect; i++) {
        if (*pos >= byte_len) return VM_UNICODE_REPLACEMENT;
        unsigned char c = (unsigned char)buf[*pos];
        if ((c & 0xC0) != 0x80) return VM_UNICODE_REPLACEMENT; /* bad continuation */
        cp = (cp << 6) | (c & 0x3F);
        (*pos)++;
    }

    /* Reject overlong encodings */
    if (expect == 1 && cp < 0x80) return VM_UNICODE_REPLACEMENT;
    if (expect == 2 && cp < 0x800) return VM_UNICODE_REPLACEMENT;
    if (expect == 3 && cp < 0x10000) return VM_UNICODE_REPLACEMENT;

    /* Reject surrogates and out-of-range */
    if (cp >= 0xD800 && cp <= 0xDFFF) return VM_UNICODE_REPLACEMENT;
    if (cp > 0x10FFFF) return VM_UNICODE_REPLACEMENT;

    return cp;
}

/* ── UTF-8 Encode ──
 * Encodes codepoint cp into buf (must have room for 4 bytes).
 * Returns number of bytes written (1-4), or 0 on invalid cp.
 */
static int vm_utf8_encode(int cp, char* buf) {
    if (cp < 0 || (cp >= 0xD800 && cp <= 0xDFFF) || cp > 0x10FFFF) {
        /* Encode replacement character U+FFFD */
        buf[0] = (char)0xEF;
        buf[1] = (char)0xBF;
        buf[2] = (char)0xBD;
        return 3;
    }
    if (cp < 0x80) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp < 0x800) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp < 0x10000) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    buf[0] = (char)(0xF0 | (cp >> 18));
    buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    buf[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* ── Count codepoints in a UTF-8 buffer ── */
static int vm_utf8_char_count(const char* buf, int byte_len) {
    int count = 0;
    int pos = 0;
    while (pos < byte_len) {
        vm_utf8_decode(buf, byte_len, &pos);
        count++;
    }
    return count;
}

/* ── Find byte offset of the n-th codepoint (0-indexed) ──
 * Returns byte offset, or -1 if idx is out of range.
 */
static int vm_utf8_byte_offset(const char* buf, int byte_len, int idx) {
    int pos = 0;
    for (int i = 0; i < idx; i++) {
        if (pos >= byte_len) return -1;
        vm_utf8_decode(buf, byte_len, &pos);
    }
    return pos;
}

/* ── Allocation ── */

/* 550: vm_string_new — create string from raw bytes + byte_len */
static VmString* vm_string_new(VmRegionStack* rs, const char* data, int byte_len) {
    VmString* s = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!s) return NULL;
    s->data = (char*)vm_alloc(rs, byte_len + 1);
    if (!s->data) return NULL;
    memcpy(s->data, data, byte_len);
    s->data[byte_len] = '\0';
    s->byte_len = byte_len;
    s->char_len = vm_utf8_char_count(data, byte_len);
    return s;
}

/* 551: vm_string_from_cstr — create string from C string */
static VmString* vm_string_from_cstr(VmRegionStack* rs, const char* cstr) {
    if (!cstr) return vm_string_new(rs, "", 0);
    return vm_string_new(rs, cstr, (int)strlen(cstr));
}

/* ── Core Operations ── */

/* 552: string-length → codepoint count */
static int vm_string_length(const VmString* s) {
    if (!s) return 0;
    return s->char_len;
}

/* 553: string-ref → codepoint at char index (O(n) scan) */
static int vm_string_ref(const VmString* s, int idx) {
    if (!s || idx < 0 || idx >= s->char_len) return -1;
    int pos = vm_utf8_byte_offset(s->data, s->byte_len, idx);
    if (pos < 0 || pos >= s->byte_len) return -1;
    return vm_utf8_decode(s->data, s->byte_len, &pos);
}

/* 554: string-set → new string with codepoint replaced at idx (immutable) */
static VmString* vm_string_set(VmRegionStack* rs, const VmString* s, int idx, int cp) {
    if (!s || idx < 0 || idx >= s->char_len) return NULL;

    /* Find byte range of the character at idx */
    int start_byte = vm_utf8_byte_offset(s->data, s->byte_len, idx);
    if (start_byte < 0) return NULL;
    int end_byte = start_byte;
    vm_utf8_decode(s->data, s->byte_len, &end_byte);

    /* Encode the new codepoint */
    char enc[4];
    int enc_len = vm_utf8_encode(cp, enc);

    /* Build result: prefix + encoded char + suffix */
    int old_char_bytes = end_byte - start_byte;
    int new_byte_len = s->byte_len - old_char_bytes + enc_len;

    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, new_byte_len + 1);
    if (!result->data) return NULL;

    memcpy(result->data, s->data, start_byte);
    memcpy(result->data + start_byte, enc, enc_len);
    memcpy(result->data + start_byte + enc_len, s->data + end_byte, s->byte_len - end_byte);
    result->data[new_byte_len] = '\0';
    result->byte_len = new_byte_len;
    result->char_len = s->char_len; /* same number of codepoints */
    return result;
}

/* 555: substring → new string from char indices [start, end) */
static VmString* vm_string_substring(VmRegionStack* rs, const VmString* s, int start, int end) {
    if (!s) return NULL;
    if (start < 0) start = 0;
    if (end > s->char_len) end = s->char_len;
    if (start >= end) return vm_string_new(rs, "", 0);

    int start_byte = vm_utf8_byte_offset(s->data, s->byte_len, start);
    int end_byte = vm_utf8_byte_offset(s->data, s->byte_len, end);
    if (start_byte < 0) start_byte = 0;
    if (end_byte < 0) end_byte = s->byte_len;

    return vm_string_new(rs, s->data + start_byte, end_byte - start_byte);
}

/* 556: string-append → concatenated string */
static VmString* vm_string_append(VmRegionStack* rs, const VmString* a, const VmString* b) {
    if (!a && !b) return vm_string_new(rs, "", 0);
    if (!a) return vm_string_new(rs, b->data, b->byte_len);
    if (!b) return vm_string_new(rs, a->data, a->byte_len);

    int new_byte_len = a->byte_len + b->byte_len;
    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, new_byte_len + 1);
    if (!result->data) return NULL;

    memcpy(result->data, a->data, a->byte_len);
    memcpy(result->data + a->byte_len, b->data, b->byte_len);
    result->data[new_byte_len] = '\0';
    result->byte_len = new_byte_len;
    result->char_len = a->char_len + b->char_len;
    return result;
}

/* 557: string-upcase → uppercased (ASCII codepoints only) */
static VmString* vm_string_upcase(VmRegionStack* rs, const VmString* s) {
    if (!s) return NULL;
    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, s->byte_len + 1);
    if (!result->data) return NULL;

    int pos = 0, out = 0;
    while (pos < s->byte_len) {
        int cp = vm_utf8_decode(s->data, s->byte_len, &pos);
        if (cp >= 'a' && cp <= 'z') cp -= 32;
        out += vm_utf8_encode(cp, result->data + out);
    }
    result->data[out] = '\0';
    result->byte_len = out;
    result->char_len = s->char_len;
    return result;
}

/* 558: string-downcase → lowercased (ASCII codepoints only) */
static VmString* vm_string_downcase(VmRegionStack* rs, const VmString* s) {
    if (!s) return NULL;
    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, s->byte_len + 1);
    if (!result->data) return NULL;

    int pos = 0, out = 0;
    while (pos < s->byte_len) {
        int cp = vm_utf8_decode(s->data, s->byte_len, &pos);
        if (cp >= 'A' && cp <= 'Z') cp += 32;
        out += vm_utf8_encode(cp, result->data + out);
    }
    result->data[out] = '\0';
    result->byte_len = out;
    result->char_len = s->char_len;
    return result;
}

/* 559: string-contains → byte index of first occurrence, or -1
 * Returns codepoint index, not byte index. */
static int vm_string_contains(const VmString* s, const VmString* substr) {
    if (!s || !substr) return -1;
    if (substr->byte_len == 0) return 0;
    if (substr->byte_len > s->byte_len) return -1;

    /* Byte-level search, then convert to codepoint index */
    for (int i = 0; i <= s->byte_len - substr->byte_len; i++) {
        if (memcmp(s->data + i, substr->data, substr->byte_len) == 0) {
            /* Convert byte offset i to codepoint index */
            return vm_utf8_char_count(s->data, i);
        }
    }
    return -1;
}

/* ── Comparison ── */

/* 560: string=? */
static int vm_string_eq(const VmString* a, const VmString* b) {
    if (!a || !b) return (!a && !b);
    if (a->byte_len != b->byte_len) return 0;
    return memcmp(a->data, b->data, a->byte_len) == 0;
}

/* 561: string<? (lexicographic by codepoint) */
static int vm_string_lt(const VmString* a, const VmString* b) {
    if (!a || !b) return (!a && b);
    int pa = 0, pb = 0;
    while (pa < a->byte_len && pb < b->byte_len) {
        int ca = vm_utf8_decode(a->data, a->byte_len, &pa);
        int cb = vm_utf8_decode(b->data, b->byte_len, &pb);
        if (ca < cb) return 1;
        if (ca > cb) return 0;
    }
    return (pa >= a->byte_len && pb < b->byte_len);
}

/* 562: string-ci=? (case-insensitive equality, ASCII folding) */
static int vm_string_ci_eq(const VmString* a, const VmString* b) {
    if (!a || !b) return (!a && !b);
    int pa = 0, pb = 0;
    while (pa < a->byte_len && pb < b->byte_len) {
        int ca = vm_utf8_decode(a->data, a->byte_len, &pa);
        int cb = vm_utf8_decode(b->data, b->byte_len, &pb);
        /* ASCII case fold */
        if (ca >= 'A' && ca <= 'Z') ca += 32;
        if (cb >= 'A' && cb <= 'Z') cb += 32;
        if (ca != cb) return 0;
    }
    return (pa >= a->byte_len && pb >= b->byte_len);
}

/* ── Conversion ── */

/* 563: string->number → parse numeric string, returns NaN on failure */
static double vm_string_to_number(const VmString* s) {
    if (!s || s->byte_len == 0) return NAN;
    const char* p = s->data;
    /* Handle Scheme radix prefixes: #x (hex), #b (binary), #o (octal), #d (decimal) */
    if (p[0] == '#' && p[1]) {
        char pfx = (char)(p[1] | 32); /* to lowercase */
        int radix = 0;
        if      (pfx == 'x') { radix = 16; p += 2; }
        else if (pfx == 'b') { radix =  2; p += 2; }
        else if (pfx == 'o') { radix =  8; p += 2; }
        else if (pfx == 'd') { radix = 10; p += 2; }
        if (radix && *p) {
            char* end;
            long long iv = strtoll(p, &end, radix);
            while (*end && isspace((unsigned char)*end)) end++;
            if (*end == '\0') return (double)iv;
            return NAN;
        }
        return NAN; /* unknown prefix */
    }
    /* Standard decimal/float parsing */
    char* end;
    errno = 0;
    double val = strtod(p, &end);
    while (*end && isspace((unsigned char)*end)) end++;
    if (*end != '\0' || errno == ERANGE) return NAN;
    return val;
}

/* 564: number->string → format double */
static VmString* vm_number_to_string(VmRegionStack* rs, double n) {
    char buf[64];
    if (n == (int64_t)n && fabs(n) < 1e15) {
        snprintf(buf, sizeof(buf), "%lld", (long long)(int64_t)n);
    } else {
        snprintf(buf, sizeof(buf), "%.17g", n);
    }
    return vm_string_from_cstr(rs, buf);
}

/* 565: string-copy → deep copy */
static VmString* vm_string_copy(VmRegionStack* rs, const VmString* s) {
    if (!s) return NULL;
    return vm_string_new(rs, s->data, s->byte_len);
}

/* 566: string->list → simple list of codepoints as int array
 * Returns array of ints (arena-allocated), writes count to *out_len.
 * For the bytecode VM, this produces a cons-list via the VM's CONS op.
 * Here we return the raw array of codepoints. */
static int* vm_string_to_list(VmRegionStack* rs, const VmString* s, int* out_len) {
    if (!s || s->char_len == 0) {
        *out_len = 0;
        return NULL;
    }
    *out_len = s->char_len;
    int* cps = (int*)vm_alloc(rs, s->char_len * sizeof(int));
    if (!cps) { *out_len = 0; return NULL; }

    int pos = 0;
    for (int i = 0; i < s->char_len; i++) {
        cps[i] = vm_utf8_decode(s->data, s->byte_len, &pos);
    }
    return cps;
}

/* 567: list->string → build string from codepoint array */
static VmString* vm_string_from_list(VmRegionStack* rs, const int* cps, int count) {
    if (!cps || count <= 0) return vm_string_new(rs, "", 0);

    /* First pass: compute total byte length */
    int total_bytes = 0;
    char tmp[4];
    for (int i = 0; i < count; i++) {
        total_bytes += vm_utf8_encode(cps[i], tmp);
    }

    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, total_bytes + 1);
    if (!result->data) return NULL;

    int pos = 0;
    for (int i = 0; i < count; i++) {
        pos += vm_utf8_encode(cps[i], result->data + pos);
    }
    result->data[total_bytes] = '\0';
    result->byte_len = total_bytes;
    result->char_len = count;
    return result;
}

/* 568: make-string → string of n copies of char cp */
static VmString* vm_string_make(VmRegionStack* rs, int n, int cp) {
    if (n <= 0) return vm_string_new(rs, "", 0);

    char enc[4];
    int enc_len = vm_utf8_encode(cp, enc);

    int total_bytes = n * enc_len;
    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, total_bytes + 1);
    if (!result->data) return NULL;

    for (int i = 0; i < n; i++) {
        memcpy(result->data + i * enc_len, enc, enc_len);
    }
    result->data[total_bytes] = '\0';
    result->byte_len = total_bytes;
    result->char_len = n;
    return result;
}

/* 569: string-hash → simple FNV-1a hash */
static uint64_t vm_string_hash(const VmString* s) {
    if (!s) return 0;
    uint64_t h = 14695981039346656037ULL;
    for (int i = 0; i < s->byte_len; i++) {
        h ^= (uint64_t)(unsigned char)s->data[i];
        h *= 1099511628211ULL;
    }
    return h;
}

/* 570: string-reverse → reversed string (by codepoints) */
static VmString* vm_string_reverse(VmRegionStack* rs, const VmString* s) {
    if (!s || s->char_len <= 1) {
        return s ? vm_string_copy(rs, s) : NULL;
    }

    /* Extract codepoints, then reverse */
    int count;
    int* cps = vm_string_to_list(rs, s, &count);
    if (!cps) return NULL;

    /* Reverse in place */
    for (int i = 0; i < count / 2; i++) {
        int tmp = cps[i];
        cps[i] = cps[count - 1 - i];
        cps[count - 1 - i] = tmp;
    }
    return vm_string_from_list(rs, cps, count);
}

/* 571: string-trim → remove leading/trailing whitespace */
static VmString* vm_string_trim(VmRegionStack* rs, const VmString* s) {
    if (!s || s->byte_len == 0) return vm_string_new(rs, "", 0);

    /* Extract codepoints */
    int count;
    int* cps = vm_string_to_list(rs, s, &count);
    if (!cps || count == 0) return vm_string_new(rs, "", 0);

    /* Find first non-whitespace */
    int start = 0;
    while (start < count && (cps[start] == ' ' || cps[start] == '\t' ||
           cps[start] == '\n' || cps[start] == '\r')) start++;

    /* Find last non-whitespace */
    int end = count;
    while (end > start && (cps[end - 1] == ' ' || cps[end - 1] == '\t' ||
           cps[end - 1] == '\n' || cps[end - 1] == '\r')) end--;

    if (start >= end) return vm_string_new(rs, "", 0);
    return vm_string_from_list(rs, cps + start, end - start);
}

/* 572: string-split → array of VmString* (split by single-char delimiter)
 * Returns arena-allocated array. Writes count to *out_count. */
static VmString** vm_string_split(VmRegionStack* rs, const VmString* s, int delim_cp,
                                   int* out_count) {
    if (!s || s->byte_len == 0) {
        *out_count = 0;
        return NULL;
    }

    /* Count segments */
    int n_segments = 1;
    int pos = 0;
    while (pos < s->byte_len) {
        int cp = vm_utf8_decode(s->data, s->byte_len, &pos);
        if (cp == delim_cp) n_segments++;
    }

    VmString** parts = (VmString**)vm_alloc(rs, n_segments * sizeof(VmString*));
    if (!parts) { *out_count = 0; return NULL; }

    /* Split */
    int seg = 0;
    int seg_start = 0;
    pos = 0;
    while (pos < s->byte_len) {
        int prev_pos = pos;
        int cp = vm_utf8_decode(s->data, s->byte_len, &pos);
        if (cp == delim_cp) {
            parts[seg++] = vm_string_new(rs, s->data + seg_start, prev_pos - seg_start);
            seg_start = pos;
        }
    }
    /* Last segment */
    parts[seg++] = vm_string_new(rs, s->data + seg_start, s->byte_len - seg_start);

    *out_count = seg;
    return parts;
}

/* 573: string-join → join array of VmString* with separator */
static VmString* vm_string_join(VmRegionStack* rs, VmString** parts, int count,
                                 const VmString* sep) {
    if (!parts || count <= 0) return vm_string_new(rs, "", 0);
    if (count == 1) return vm_string_copy(rs, parts[0]);

    /* Compute total byte length */
    int total = 0;
    for (int i = 0; i < count; i++) {
        if (parts[i]) total += parts[i]->byte_len;
    }
    if (sep) total += sep->byte_len * (count - 1);

    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, total + 1);
    if (!result->data) return NULL;

    int pos = 0;
    int char_total = 0;
    for (int i = 0; i < count; i++) {
        if (i > 0 && sep) {
            memcpy(result->data + pos, sep->data, sep->byte_len);
            pos += sep->byte_len;
            char_total += sep->char_len;
        }
        if (parts[i]) {
            memcpy(result->data + pos, parts[i]->data, parts[i]->byte_len);
            pos += parts[i]->byte_len;
            char_total += parts[i]->char_len;
        }
    }
    result->data[total] = '\0';
    result->byte_len = total;
    result->char_len = char_total;
    return result;
}

/* 574: string-starts-with? */
static int vm_string_starts_with(const VmString* s, const VmString* prefix) {
    if (!s || !prefix) return 0;
    if (prefix->byte_len > s->byte_len) return 0;
    return memcmp(s->data, prefix->data, prefix->byte_len) == 0;
}

/* 575: string-ends-with? */
static int vm_string_ends_with(const VmString* s, const VmString* suffix) {
    if (!s || !suffix) return 0;
    if (suffix->byte_len > s->byte_len) return 0;
    return memcmp(s->data + s->byte_len - suffix->byte_len,
                  suffix->data, suffix->byte_len) == 0;
}

/* 576: string-replace → replace first occurrence of old with new */
static VmString* vm_string_replace(VmRegionStack* rs, const VmString* s,
                                     const VmString* old_sub, const VmString* new_sub) {
    if (!s || !old_sub || old_sub->byte_len == 0) return s ? vm_string_copy(rs, s) : NULL;
    if (!new_sub) new_sub = &(VmString){0, 0, ""};

    /* Find first occurrence */
    const char* found = NULL;
    for (int i = 0; i <= s->byte_len - old_sub->byte_len; i++) {
        if (memcmp(s->data + i, old_sub->data, old_sub->byte_len) == 0) {
            found = s->data + i;
            break;
        }
    }
    if (!found) return vm_string_copy(rs, s);

    int prefix_len = (int)(found - s->data);
    int suffix_start = prefix_len + old_sub->byte_len;
    int suffix_len = s->byte_len - suffix_start;
    int new_byte_len = prefix_len + new_sub->byte_len + suffix_len;

    VmString* result = (VmString*)vm_alloc(rs, sizeof(VmString));
    if (!result) return NULL;
    result->data = (char*)vm_alloc(rs, new_byte_len + 1);
    if (!result->data) return NULL;

    memcpy(result->data, s->data, prefix_len);
    memcpy(result->data + prefix_len, new_sub->data, new_sub->byte_len);
    memcpy(result->data + prefix_len + new_sub->byte_len, s->data + suffix_start, suffix_len);
    result->data[new_byte_len] = '\0';
    result->byte_len = new_byte_len;
    result->char_len = vm_utf8_char_count(result->data, new_byte_len);
    return result;
}

/* 577: integer->char (codepoint to single-char string) */
static VmString* vm_integer_to_char(VmRegionStack* rs, int cp) {
    char buf[4];
    int len = vm_utf8_encode(cp, buf);
    return vm_string_new(rs, buf, len);
}

/* 578: char->integer (first char of string to codepoint) */
static int vm_char_to_integer(const VmString* s) {
    if (!s || s->byte_len == 0) return -1;
    int pos = 0;
    return vm_utf8_decode(s->data, s->byte_len, &pos);
}

/* ── Self-Test ── */

#ifdef VM_STRING_TEST
#include <assert.h>

static void test_utf8_roundtrip(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* ASCII roundtrip */
    char buf[4];
    int len = vm_utf8_encode('A', buf);
    assert(len == 1 && buf[0] == 'A');

    /* 2-byte: e-acute (U+00E9) */
    len = vm_utf8_encode(0xE9, buf);
    assert(len == 2);
    int pos = 0;
    int cp = vm_utf8_decode(buf, len, &pos);
    assert(cp == 0xE9 && pos == 2);

    /* 3-byte: Euro sign (U+20AC) */
    len = vm_utf8_encode(0x20AC, buf);
    assert(len == 3);
    pos = 0;
    cp = vm_utf8_decode(buf, len, &pos);
    assert(cp == 0x20AC && pos == 3);

    /* 4-byte: Musical symbol G-clef (U+1D11E) */
    len = vm_utf8_encode(0x1D11E, buf);
    assert(len == 4);
    pos = 0;
    cp = vm_utf8_decode(buf, len, &pos);
    assert(cp == 0x1D11E && pos == 4);

    /* Invalid: surrogate */
    len = vm_utf8_encode(0xD800, buf);
    assert(len == 3); /* should encode replacement char */
    pos = 0;
    cp = vm_utf8_decode(buf, len, &pos);
    assert(cp == VM_UNICODE_REPLACEMENT);

    vm_region_stack_destroy(&rs);
    printf("  utf8_roundtrip: PASS\n");
}

static void test_string_basic(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* length("hello") = 5 */
    VmString* s = vm_string_from_cstr(&rs, "hello");
    assert(s && vm_string_length(s) == 5);

    /* length("héllo") = 5 (é is 2 bytes, 1 codepoint) */
    VmString* s2 = vm_string_from_cstr(&rs, "h\xc3\xa9llo");
    assert(s2 && vm_string_length(s2) == 5);
    assert(s2->byte_len == 6); /* 'h' + 2-byte é + 'l' + 'l' + 'o' */

    /* ref("abc", 1) = 'b' (98) */
    VmString* abc = vm_string_from_cstr(&rs, "abc");
    assert(vm_string_ref(abc, 0) == 'a');
    assert(vm_string_ref(abc, 1) == 'b');
    assert(vm_string_ref(abc, 2) == 'c');
    assert(vm_string_ref(abc, 3) == -1); /* out of bounds */

    /* ref with multibyte */
    assert(vm_string_ref(s2, 1) == 0xE9); /* é */
    assert(vm_string_ref(s2, 2) == 'l');

    vm_region_stack_destroy(&rs);
    printf("  string_basic: PASS\n");
}

static void test_string_set(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "abc");
    VmString* s2 = vm_string_set(&rs, s, 1, 'x');
    assert(s2 && vm_string_length(s2) == 3);
    assert(vm_string_ref(s2, 0) == 'a');
    assert(vm_string_ref(s2, 1) == 'x');
    assert(vm_string_ref(s2, 2) == 'c');

    /* Replace ASCII with multibyte */
    VmString* s3 = vm_string_set(&rs, s, 1, 0xE9); /* é */
    assert(s3 && vm_string_length(s3) == 3);
    assert(vm_string_ref(s3, 1) == 0xE9);
    assert(s3->byte_len == 4); /* a + 2-byte é + c */

    vm_region_stack_destroy(&rs);
    printf("  string_set: PASS\n");
}

static void test_substring(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello");
    VmString* sub = vm_string_substring(&rs, s, 1, 4);
    assert(sub && vm_string_length(sub) == 3);
    assert(strcmp(sub->data, "ell") == 0);

    /* Empty substring */
    VmString* empty = vm_string_substring(&rs, s, 2, 2);
    assert(empty && vm_string_length(empty) == 0);

    /* Full string */
    VmString* full = vm_string_substring(&rs, s, 0, 5);
    assert(full && strcmp(full->data, "hello") == 0);

    vm_region_stack_destroy(&rs);
    printf("  substring: PASS\n");
}

static void test_append(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* a = vm_string_from_cstr(&rs, "hello");
    VmString* b = vm_string_from_cstr(&rs, " world");
    VmString* c = vm_string_append(&rs, a, b);
    assert(c && strcmp(c->data, "hello world") == 0);
    assert(vm_string_length(c) == 11);

    /* Append with NULL */
    VmString* d = vm_string_append(&rs, a, NULL);
    assert(d && strcmp(d->data, "hello") == 0);

    vm_region_stack_destroy(&rs);
    printf("  append: PASS\n");
}

static void test_case(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello");
    VmString* up = vm_string_upcase(&rs, s);
    assert(up && strcmp(up->data, "HELLO") == 0);

    VmString* s2 = vm_string_from_cstr(&rs, "WORLD");
    VmString* lo = vm_string_downcase(&rs, s2);
    assert(lo && strcmp(lo->data, "world") == 0);

    /* Mixed case */
    VmString* s3 = vm_string_from_cstr(&rs, "Hello World 123!");
    VmString* up3 = vm_string_upcase(&rs, s3);
    assert(up3 && strcmp(up3->data, "HELLO WORLD 123!") == 0);

    vm_region_stack_destroy(&rs);
    printf("  case: PASS\n");
}

static void test_contains(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello world");
    VmString* sub = vm_string_from_cstr(&rs, "world");
    assert(vm_string_contains(s, sub) == 6);

    VmString* sub2 = vm_string_from_cstr(&rs, "xyz");
    assert(vm_string_contains(s, sub2) == -1);

    VmString* sub3 = vm_string_from_cstr(&rs, "hello");
    assert(vm_string_contains(s, sub3) == 0);

    /* Empty substring */
    VmString* empty = vm_string_from_cstr(&rs, "");
    assert(vm_string_contains(s, empty) == 0);

    vm_region_stack_destroy(&rs);
    printf("  contains: PASS\n");
}

static void test_comparison(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* a = vm_string_from_cstr(&rs, "abc");
    VmString* b = vm_string_from_cstr(&rs, "abc");
    VmString* c = vm_string_from_cstr(&rs, "abd");
    VmString* d = vm_string_from_cstr(&rs, "ABC");

    assert(vm_string_eq(a, b) == 1);
    assert(vm_string_eq(a, c) == 0);

    assert(vm_string_lt(a, c) == 1);  /* "abc" < "abd" */
    assert(vm_string_lt(c, a) == 0);
    assert(vm_string_lt(a, b) == 0);  /* equal → not less */

    assert(vm_string_ci_eq(a, d) == 1);  /* case-insensitive */
    assert(vm_string_ci_eq(a, c) == 0);

    /* Length-based ordering */
    VmString* short_s = vm_string_from_cstr(&rs, "ab");
    assert(vm_string_lt(short_s, a) == 1);  /* "ab" < "abc" */
    assert(vm_string_lt(a, short_s) == 0);

    vm_region_stack_destroy(&rs);
    printf("  comparison: PASS\n");
}

static void test_conversion(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    /* string->number */
    VmString* n1 = vm_string_from_cstr(&rs, "42.5");
    assert(vm_string_to_number(n1) == 42.5);

    VmString* n2 = vm_string_from_cstr(&rs, "-17");
    assert(vm_string_to_number(n2) == -17.0);

    VmString* n3 = vm_string_from_cstr(&rs, "3.14e2");
    assert(fabs(vm_string_to_number(n3) - 314.0) < 1e-10);

    VmString* bad = vm_string_from_cstr(&rs, "not-a-number");
    assert(isnan(vm_string_to_number(bad)));

    /* number->string */
    VmString* s1 = vm_number_to_string(&rs, 42.0);
    assert(s1 && strcmp(s1->data, "42") == 0);

    VmString* s2 = vm_number_to_string(&rs, 3.14);
    assert(s2 && vm_string_to_number(s2) == 3.14);

    vm_region_stack_destroy(&rs);
    printf("  conversion: PASS\n");
}

static void test_list_roundtrip(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello");
    int count;
    int* cps = vm_string_to_list(&rs, s, &count);
    assert(count == 5);
    assert(cps[0] == 'h' && cps[1] == 'e' && cps[2] == 'l' &&
           cps[3] == 'l' && cps[4] == 'o');

    VmString* s2 = vm_string_from_list(&rs, cps, count);
    assert(s2 && vm_string_eq(s, s2));

    /* Multibyte roundtrip */
    VmString* mb = vm_string_from_cstr(&rs, "caf\xc3\xa9");
    int* mb_cps = vm_string_to_list(&rs, mb, &count);
    assert(count == 4);
    assert(mb_cps[3] == 0xE9);
    VmString* mb2 = vm_string_from_list(&rs, mb_cps, count);
    assert(mb2 && vm_string_eq(mb, mb2));

    vm_region_stack_destroy(&rs);
    printf("  list_roundtrip: PASS\n");
}

static void test_make_string(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_make(&rs, 5, 'x');
    assert(s && vm_string_length(s) == 5);
    assert(strcmp(s->data, "xxxxx") == 0);

    /* Multibyte fill char */
    VmString* s2 = vm_string_make(&rs, 3, 0xE9); /* 3 x é */
    assert(s2 && vm_string_length(s2) == 3);
    assert(s2->byte_len == 6); /* 3 x 2-byte */

    vm_region_stack_destroy(&rs);
    printf("  make_string: PASS\n");
}

static void test_hash(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* a = vm_string_from_cstr(&rs, "hello");
    VmString* b = vm_string_from_cstr(&rs, "hello");
    VmString* c = vm_string_from_cstr(&rs, "world");

    uint64_t ha = vm_string_hash(a);
    uint64_t hb = vm_string_hash(b);
    uint64_t hc = vm_string_hash(c);

    assert(ha == hb); /* same content → same hash */
    assert(ha != hc); /* different content → different hash (with high probability) */

    vm_region_stack_destroy(&rs);
    printf("  hash: PASS\n");
}

static void test_reverse(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "abcde");
    VmString* rev = vm_string_reverse(&rs, s);
    assert(rev && strcmp(rev->data, "edcba") == 0);

    /* Multibyte reverse */
    VmString* mb = vm_string_from_cstr(&rs, "caf\xc3\xa9");
    VmString* rev_mb = vm_string_reverse(&rs, mb);
    assert(rev_mb && vm_string_length(rev_mb) == 4);
    assert(vm_string_ref(rev_mb, 0) == 0xE9);
    assert(vm_string_ref(rev_mb, 1) == 'f');

    vm_region_stack_destroy(&rs);
    printf("  reverse: PASS\n");
}

static void test_trim(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "  hello  ");
    VmString* trimmed = vm_string_trim(&rs, s);
    assert(trimmed && strcmp(trimmed->data, "hello") == 0);

    VmString* s2 = vm_string_from_cstr(&rs, "\t\n hello \r\n");
    VmString* t2 = vm_string_trim(&rs, s2);
    assert(t2 && strcmp(t2->data, "hello") == 0);

    vm_region_stack_destroy(&rs);
    printf("  trim: PASS\n");
}

static void test_split_join(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "a,b,c,d");
    int count;
    VmString** parts = vm_string_split(&rs, s, ',', &count);
    assert(count == 4);
    assert(strcmp(parts[0]->data, "a") == 0);
    assert(strcmp(parts[1]->data, "b") == 0);
    assert(strcmp(parts[2]->data, "c") == 0);
    assert(strcmp(parts[3]->data, "d") == 0);

    VmString* sep = vm_string_from_cstr(&rs, ",");
    VmString* joined = vm_string_join(&rs, parts, count, sep);
    assert(joined && strcmp(joined->data, "a,b,c,d") == 0);

    vm_region_stack_destroy(&rs);
    printf("  split_join: PASS\n");
}

static void test_starts_ends_with(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello world");
    VmString* pre = vm_string_from_cstr(&rs, "hello");
    VmString* suf = vm_string_from_cstr(&rs, "world");
    VmString* bad = vm_string_from_cstr(&rs, "xyz");

    assert(vm_string_starts_with(s, pre) == 1);
    assert(vm_string_starts_with(s, bad) == 0);
    assert(vm_string_ends_with(s, suf) == 1);
    assert(vm_string_ends_with(s, bad) == 0);

    vm_region_stack_destroy(&rs);
    printf("  starts_ends_with: PASS\n");
}

static void test_replace(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* s = vm_string_from_cstr(&rs, "hello world");
    VmString* old_s = vm_string_from_cstr(&rs, "world");
    VmString* new_s = vm_string_from_cstr(&rs, "there");
    VmString* result = vm_string_replace(&rs, s, old_s, new_s);
    assert(result && strcmp(result->data, "hello there") == 0);

    /* No match → copy */
    VmString* nomatch = vm_string_from_cstr(&rs, "xyz");
    VmString* result2 = vm_string_replace(&rs, s, nomatch, new_s);
    assert(result2 && strcmp(result2->data, "hello world") == 0);

    vm_region_stack_destroy(&rs);
    printf("  replace: PASS\n");
}

static void test_char_conversion(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* ch = vm_integer_to_char(&rs, 0xE9);
    assert(ch && vm_string_length(ch) == 1);
    assert(vm_char_to_integer(ch) == 0xE9);

    VmString* ascii = vm_integer_to_char(&rs, 'A');
    assert(ascii && vm_string_length(ascii) == 1);
    assert(vm_char_to_integer(ascii) == 'A');

    vm_region_stack_destroy(&rs);
    printf("  char_conversion: PASS\n");
}

static void test_empty_string(void) {
    VmRegionStack rs;
    vm_region_stack_init(&rs);

    VmString* empty = vm_string_from_cstr(&rs, "");
    assert(empty && vm_string_length(empty) == 0);
    assert(empty->byte_len == 0);

    /* Operations on empty strings */
    VmString* up = vm_string_upcase(&rs, empty);
    assert(up && vm_string_length(up) == 0);

    VmString* rev = vm_string_reverse(&rs, empty);
    assert(rev && vm_string_length(rev) == 0);

    VmString* copy = vm_string_copy(&rs, empty);
    assert(copy && vm_string_length(copy) == 0);

    assert(vm_string_eq(empty, copy));
    assert(isnan(vm_string_to_number(empty)));

    vm_region_stack_destroy(&rs);
    printf("  empty_string: PASS\n");
}

int main(void) {
    printf("vm_string self-tests:\n");
    test_utf8_roundtrip();
    test_string_basic();
    test_string_set();
    test_substring();
    test_append();
    test_case();
    test_contains();
    test_comparison();
    test_conversion();
    test_list_roundtrip();
    test_make_string();
    test_hash();
    test_reverse();
    test_trim();
    test_split_join();
    test_starts_ends_with();
    test_replace();
    test_char_conversion();
    test_empty_string();
    printf("vm_string: ALL 19 TESTS PASSED\n");
    return 0;
}
#endif

#endif /* VM_STRING_C_INCLUDED */
