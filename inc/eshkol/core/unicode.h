/*
 * Unicode character classification shared by the native and VM engines.
 */
#ifndef ESHKOL_CORE_UNICODE_H
#define ESHKOL_CORE_UNICODE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Decode one RFC 3629 scalar. Return its byte width, or zero on invalid input. */
static inline size_t eshkol_utf8_read_scalar(const char* data, size_t available,
                                             uint32_t* result) {
    if (!available) return 0;
    const unsigned char* bytes = (const unsigned char*)data;
    const unsigned char lead = bytes[0];
    size_t width;
    uint32_t value;
    if (lead < 0x80) { width = 1; value = lead; }
    else if (lead >= 0xC2 && lead <= 0xDF) { width = 2; value = lead & 31; }
    else if (lead >= 0xE0 && lead <= 0xEF) { width = 3; value = lead & 15; }
    else if (lead >= 0xF0 && lead <= 0xF4) { width = 4; value = lead & 7; }
    else return 0;

    if (width > available) return 0;
    for (size_t i = 1; i < width; ++i) {
        if ((bytes[i] & 0xC0) != 0x80) return 0;
        value = (value << 6) | (bytes[i] & 63);
    }
    if ((width == 3 && value < 0x800) || (width == 4 && value < 0x10000) ||
        (value >= 0xD800 && value <= 0xDFFF) || value > 0x10FFFF) return 0;
    *result = value;
    return width;
}

/**
 * R7RS 7.1.1 `#\x<hex scalar value>`, shared by both readers. @p data points
 * just past `#\`; a NUL byte or @p available ends the input. When the
 * spelling is `x`, one or more hexadecimal digits and then a delimiter
 * (whitespace, `(`, `)`, `"`, `;`, `|`) or the end, returns the bytes it
 * occupies and stores the scalar in *@p result, or sets *@p invalid when the
 * digits name a surrogate or a value above U+10FFFF. Any other spelling --
 * `#\x` alone, `#\xylophone` -- returns 0 and is not a hex character.
 */
static inline size_t eshkol_char_hex_literal(const char* data, size_t available,
                                             uint32_t* result, int* invalid) {
    *invalid = 0;
    if (!available || data[0] != 'x') return 0;
    size_t i = 1;
    uint32_t value = 0;
    int too_large = 0;
    for (; i < available && data[i]; ++i) {
        const char c = data[i];
        int digit;
        if (c >= '0' && c <= '9') digit = c - '0';
        else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
        else break;
        if (value > 0x10FFFFu) too_large = 1;
        else value = value * 16u + (uint32_t)digit;
    }
    if (i == 1) return 0;
    if (i < available && data[i]) {
        const char c = data[i];
        if (!(c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
              c == '\v' || c == '(' || c == ')' || c == '"' || c == ';' || c == '|'))
            return 0;
    }
    if (too_large || value > 0x10FFFFu || (value >= 0xD800u && value <= 0xDFFFu)) {
        *invalid = 1;
        return i;
    }
    *result = value;
    return i;
}
int eshkol_unicode_is_alphabetic(int64_t codepoint);
int eshkol_unicode_is_numeric(int64_t codepoint);
int eshkol_unicode_is_whitespace(int64_t codepoint);
int eshkol_unicode_is_uppercase(int64_t codepoint);
int eshkol_unicode_is_lowercase(int64_t codepoint);

#ifdef __cplusplus
}
#endif

#endif
