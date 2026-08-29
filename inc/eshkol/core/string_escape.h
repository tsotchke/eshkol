/*
 * Shared string-literal escape decoder.
 *
 * The native tokenizer, bytecode source reader, and hosted datum reader all
 * consume the same escape grammar through this header.  The decoder writes
 * bytes, not C strings: an output length of zero is meaningful for a line
 * continuation and an embedded NUL is an ordinary one-byte result.
 */
#ifndef ESHKOL_CORE_STRING_ESCAPE_H
#define ESHKOL_CORE_STRING_ESCAPE_H

#include <stddef.h>
#include <stdint.h>

enum {
    ESHKOL_STRING_ESCAPE_MALFORMED = -1,
    ESHKOL_STRING_ESCAPE_INCOMPLETE = 0,
    ESHKOL_STRING_ESCAPE_OK = 1
};

static inline int eshkol_string_escape_hex_digit(unsigned char c) {
    if (c >= (unsigned char)'0' && c <= (unsigned char)'9') return (int)c - '0';
    if (c >= (unsigned char)'a' && c <= (unsigned char)'f') return (int)c - 'a' + 10;
    if (c >= (unsigned char)'A' && c <= (unsigned char)'F') return (int)c - 'A' + 10;
    return -1;
}

static inline int eshkol_string_escape_append_utf8(uint32_t cp,
                                                    unsigned char out[4],
                                                    size_t* out_len) {
    if (cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) return 0;
    if (cp < 0x80u) {
        out[0] = (unsigned char)cp;
        *out_len = 1;
    } else if (cp < 0x800u) {
        out[0] = (unsigned char)(0xC0u | (cp >> 6));
        out[1] = (unsigned char)(0x80u | (cp & 0x3Fu));
        *out_len = 2;
    } else if (cp < 0x10000u) {
        out[0] = (unsigned char)(0xE0u | (cp >> 12));
        out[1] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (unsigned char)(0x80u | (cp & 0x3Fu));
        *out_len = 3;
    } else {
        out[0] = (unsigned char)(0xF0u | (cp >> 18));
        out[1] = (unsigned char)(0x80u | ((cp >> 12) & 0x3Fu));
        out[2] = (unsigned char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[3] = (unsigned char)(0x80u | (cp & 0x3Fu));
        *out_len = 4;
    }
    return 1;
}

/*
 * Decode the escape beginning at source[slash_pos].  `available` is the
 * number of bytes available from slash_pos onward.  `consumed` includes the
 * leading backslash.  R7RS hex escapes are variable length and terminate at
 * ';'.  Common bounded octal escapes consume one through three octal digits;
 * this makes "\033[" ESC followed by '[' while preserving "\0" as NUL.
 * U+NNNN remains accepted as the documented interoperability extension.
 */
static inline int eshkol_decode_string_escape(const unsigned char* source,
                                              size_t available,
                                              unsigned char out[4],
                                              size_t* out_len,
                                              size_t* consumed) {
    size_t i;
    unsigned char esc;
    if (!source || !out || !out_len || !consumed) return ESHKOL_STRING_ESCAPE_MALFORMED;
    *out_len = 0;
    *consumed = 0;
    if (available == 0 || source[0] != '\\') return ESHKOL_STRING_ESCAPE_MALFORMED;
    if (available == 1) return ESHKOL_STRING_ESCAPE_INCOMPLETE;

    esc = source[1];
    switch (esc) {
        case 'a': out[0] = 7;  *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'b': out[0] = 8;  *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 't': out[0] = 9;  *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'n': out[0] = 10; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'r': out[0] = 13; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'v': out[0] = 11; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'f': out[0] = 12; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case 'e': out[0] = 27; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case '\\': out[0] = '\\'; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case '"': out[0] = '"'; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        case '|': out[0] = '|'; *out_len = 1; *consumed = 2; return ESHKOL_STRING_ESCAPE_OK;
        default: break;
    }

    if (esc >= (unsigned char)'0' && esc <= (unsigned char)'7') {
        uint32_t value = (uint32_t)(esc - (unsigned char)'0');
        size_t digits = 1;
        while (digits < 3 && 1 + digits < available &&
               source[1 + digits] >= (unsigned char)'0' &&
               source[1 + digits] <= (unsigned char)'7') {
            value = (value << 3) | (uint32_t)(source[1 + digits] - (unsigned char)'0');
            digits++;
        }
        if (value > 0xFFu) return ESHKOL_STRING_ESCAPE_MALFORMED;
        out[0] = (unsigned char)value;
        *out_len = 1;
        *consumed = 1 + digits;
        return ESHKOL_STRING_ESCAPE_OK;
    }

    if (esc == (unsigned char)'x' || esc == (unsigned char)'X') {
        uint32_t value = 0;
        size_t digits = 0;
        i = 2;
        while (i < available && digits < 8) {
            int digit = eshkol_string_escape_hex_digit(source[i]);
            if (digit < 0) break;
            value = (value << 4) | (uint32_t)digit;
            i++;
            digits++;
        }
        if (digits == 0 || i >= available || source[i] != ';')
            return ESHKOL_STRING_ESCAPE_MALFORMED;
        if (!eshkol_string_escape_append_utf8(value, out, out_len))
            return ESHKOL_STRING_ESCAPE_MALFORMED;
        *consumed = i + 1;
        return ESHKOL_STRING_ESCAPE_OK;
    }

    if (esc == (unsigned char)'u') {
        uint32_t value = 0;
        if (available < 6) return ESHKOL_STRING_ESCAPE_INCOMPLETE;
        for (i = 2; i < 6; i++) {
            int digit = eshkol_string_escape_hex_digit(source[i]);
            if (digit < 0) return ESHKOL_STRING_ESCAPE_MALFORMED;
            value = (value << 4) | (uint32_t)digit;
        }
        if (!eshkol_string_escape_append_utf8(value, out, out_len))
            return ESHKOL_STRING_ESCAPE_MALFORMED;
        *consumed = 6;
        return ESHKOL_STRING_ESCAPE_OK;
    }

    /* A backslash followed by whitespace is a line continuation only when a
     * newline follows the intraline whitespace.  Otherwise preserve that
     * whitespace, matching the historical reader behavior. */
    if (esc == ' ' || esc == '\t') {
        i = 1;
        while (i < available && (source[i] == ' ' || source[i] == '\t')) i++;
        if (i < available && source[i] == '\n') {
            i++;
            while (i < available && (source[i] == ' ' || source[i] == '\t')) i++;
            *consumed = i;
            return ESHKOL_STRING_ESCAPE_OK;
        }
        if (i < available && source[i] == '\r') {
            i++;
            if (i < available && source[i] == '\n') i++;
            while (i < available && (source[i] == ' ' || source[i] == '\t')) i++;
            *consumed = i;
            return ESHKOL_STRING_ESCAPE_OK;
        }
        out[0] = esc;
        *out_len = 1;
        *consumed = 2;
        return ESHKOL_STRING_ESCAPE_OK;
    }
    if (esc == '\n') {
        *consumed = 2;
        return ESHKOL_STRING_ESCAPE_OK;
    }
    if (esc == '\r') {
        *consumed = (available >= 3 && source[2] == '\n') ? 3 : 2;
        return ESHKOL_STRING_ESCAPE_OK;
    }

    return ESHKOL_STRING_ESCAPE_MALFORMED;
}

#endif
