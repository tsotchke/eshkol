#ifndef ESHKOL_AGENT_HTTP_INTERNAL_H
#define ESHKOL_AGENT_HTTP_INTERNAL_H

#include <stddef.h>

/* RFC 9110 token grammar used by both request methods and field names. */
static inline int eshkol_http_is_token_char(unsigned char c) {
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z')) return 1;
    switch (c) {
        case '!': case '#': case '$': case '%': case '&': case '\'':
        case '*': case '+': case '-': case '.': case '^': case '_':
        case '`': case '|': case '~':
            return 1;
        default:
            return 0;
    }
}

static inline int eshkol_http_valid_token(const char* text, size_t length) {
    if (!text || length == 0) return 0;
    for (size_t i = 0; i < length; ++i) {
        if (!eshkol_http_is_token_char((unsigned char)text[i])) return 0;
    }
    return 1;
}

static inline int eshkol_http_valid_method(const char* method) {
    if (!method) return 0;
    size_t length = 0;
    while (method[length]) length++;
    return eshkol_http_valid_token(method, length);
}

/* Eshkol deliberately accepts the stricter visible-ASCII field-value subset.
 * It prevents request splitting and ambiguous whitespace across proxies. */
static inline int eshkol_http_valid_field_value(const char* value,
                                                size_t length) {
    if (!value && length != 0) return 0;
    for (size_t i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)value[i];
        if (c < 0x20 || c == 0x7f) return 0;
    }
    return 1;
}

static inline int eshkol_http_valid_header_line(const char* line) {
    if (!line || !*line) return 0;
    const char* colon = NULL;
    size_t length = 0;
    while (line[length]) {
        if (line[length] == '\r' || line[length] == '\n') return 0;
        if (!colon && line[length] == ':') colon = line + length;
        length++;
    }
    if (!colon) return 0;
    size_t name_length = (size_t)(colon - line);
    const char* value = colon + 1;
    size_t value_length = length - name_length - 1;
    while (value_length && *value == ' ') {
        value++;
        value_length--;
    }
    return eshkol_http_valid_token(line, name_length) &&
           eshkol_http_valid_field_value(value, value_length);
}

#endif
