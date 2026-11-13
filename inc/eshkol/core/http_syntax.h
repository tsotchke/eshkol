/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * http_syntax.h — the one HTTP request-syntax grammar every Eshkol engine
 * shares.
 *
 * A request line and a header block are built from values a program supplies,
 * and a program's values are data. This header is where Eshkol decides what
 * that data is allowed to be, so that the bytes that reach a socket are always
 * a request the grammar describes:
 *
 *   - METHOD is an RFC 9110 section 5.6.2 token.
 *   - REQUEST TARGET is one of the four RFC 9112 section 3.2 forms
 *     (origin, absolute, authority, asterisk), built from visible ASCII with
 *     no space and no control byte.
 *   - A HEADER NAME is an RFC 9110 token; a HEADER VALUE is visible ASCII and
 *     SP, which is the field-value subset that cannot be read two ways by a
 *     proxy on the path.
 *
 * Callers never hand a caller's bytes to the transport. They validate, and
 * then re-render the header block from the validated parse with canonical
 * CRLF line endings (eshkol_http_render_header_block), so the request on the
 * wire is the request the grammar admits and nothing else.
 *
 * eshkol_http_request_offending_field() names the one field that does not fit,
 * which is what the engines put in the error a program catches.
 *
 * Header-only and free of dependencies beyond <stddef.h>, because it is
 * compiled into the native runtime, the bytecode VM, the agent HTTP client
 * and the test binaries alike, in C and in C++.
 */
#ifndef ESHKOL_CORE_HTTP_SYNTAX_H
#define ESHKOL_CORE_HTTP_SYNTAX_H

#include <stddef.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @brief True for a byte in the RFC 9110 section 5.6.2 `tchar` set, the
 *         grammar both request methods and header field names follow. */
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

/** @brief True when `length` bytes at `text` are a non-empty RFC 9110 token. */
static inline int eshkol_http_valid_token(const char* text, size_t length) {
    size_t i;
    if (!text || length == 0) return 0;
    for (i = 0; i < length; ++i) {
        if (!eshkol_http_is_token_char((unsigned char)text[i])) return 0;
    }
    return 1;
}

/** @brief True when `method` is a NUL-terminated RFC 9110 method token. */
static inline int eshkol_http_valid_method(const char* method) {
    size_t length = 0;
    if (!method) return 0;
    while (method[length]) length++;
    return eshkol_http_valid_token(method, length);
}

/* Eshkol deliberately accepts the stricter visible-ASCII field-value subset.
 * It prevents request splitting and ambiguous whitespace across proxies. */
/** @brief True when `length` bytes at `value` are a field value Eshkol will
 *         send: visible ASCII and SP, no control byte, no DEL. An empty value
 *         is admitted; RFC 9110 allows one. */
static inline int eshkol_http_valid_field_value(const char* value,
                                                size_t length) {
    size_t i;
    if (!value && length != 0) return 0;
    for (i = 0; i < length; ++i) {
        unsigned char c = (unsigned char)value[i];
        if (c < 0x20 || c == 0x7f) return 0;
    }
    return 1;
}

/** @brief True when `line` is a complete `Name: value` header line with a
 *         token name, an admissible value, and no CR or LF anywhere in it. */
static inline int eshkol_http_valid_header_line(const char* line) {
    const char* colon = NULL;
    size_t length = 0;
    size_t name_length;
    const char* value;
    size_t value_length;
    if (!line || !*line) return 0;
    while (line[length]) {
        if (line[length] == '\r' || line[length] == '\n') return 0;
        if (!colon && line[length] == ':') colon = line + length;
        length++;
    }
    if (!colon) return 0;
    name_length = (size_t)(colon - line);
    value = colon + 1;
    value_length = length - name_length - 1;
    while (value_length && *value == ' ') {
        value++;
        value_length--;
    }
    return eshkol_http_valid_token(line, name_length) &&
           eshkol_http_valid_field_value(value, value_length);
}

/* ─── Request target (RFC 9112 section 3.2) ───────────────────────────────── */

/** @brief True for a byte admissible anywhere in a request target: the
 *         unreserved, sub-delims, pct-encoding and path/query punctuation of
 *         RFC 3986, and nothing else. SP, CR, LF, DEL, every control byte and
 *         every byte above ASCII are refused. */
static inline int eshkol_http_is_target_char(unsigned char c) {
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z')) return 1;
    switch (c) {
        /* unreserved */
        case '-': case '.': case '_': case '~':
        /* pct-encoded introducer */
        case '%':
        /* sub-delims */
        case '!': case '$': case '&': case '\'': case '(': case ')':
        case '*': case '+': case ',': case ';': case '=':
        /* pchar extras, path and query separators, userinfo/host/port */
        case ':': case '@': case '/': case '?': case '[': case ']':
            return 1;
        default:
            return 0;
    }
}

/**
 * @brief True when `target` is one of the four request-target forms.
 *
 * Accepts the asterisk form `*`, the origin form (`/` followed by path and an
 * optional `?query`), the absolute form (`scheme://` followed by authority and
 * path), and the authority form (`host:port`, which CONNECT uses). Every byte
 * must satisfy eshkol_http_is_target_char(), so a space, a CR, an LF, a NUL or
 * any other control byte refuses the target whatever form it otherwise fits.
 */
static inline int eshkol_http_valid_request_target(const char* target) {
    size_t length = 0;
    size_t i;
    if (!target || !*target) return 0;
    while (target[length]) length++;
    if (length == 1 && target[0] == '*') return 1;
    for (i = 0; i < length; ++i) {
        if (!eshkol_http_is_target_char((unsigned char)target[i])) return 0;
    }
    if (target[0] == '/') {
        /* Origin form. A second leading slash would read as an authority to a
         * proxy that re-parses the line, so it is not an origin form here. */
        return target[1] != '/';
    }
    /* Absolute form: scheme ":" "//" authority ... — the scheme is a token of
     * letters, digits and "+-." starting with a letter. */
    {
        size_t scheme_end = 0;
        while (scheme_end < length && target[scheme_end] != ':') scheme_end++;
        if (scheme_end == 0 || scheme_end >= length) return 0;
        if (!((target[0] >= 'A' && target[0] <= 'Z') ||
              (target[0] >= 'a' && target[0] <= 'z'))) return 0;
        for (i = 1; i < scheme_end; ++i) {
            char c = target[i];
            if (!((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') ||
                  (c >= 'a' && c <= 'z') || c == '+' || c == '-' || c == '.'))
                return 0;
        }
        if (scheme_end + 2 < length &&
            target[scheme_end + 1] == '/' && target[scheme_end + 2] == '/')
            return 1; /* absolute form */
        /* Authority form: host ":" port, the only remaining shape. The bytes
         * after the colon must be decimal digits and there must be some. */
        if (scheme_end + 1 >= length) return 0;
        for (i = scheme_end + 1; i < length; ++i) {
            if (target[i] < '0' || target[i] > '9') return 0;
        }
        return 1;
    }
}

/**
 * @brief True when `host` is admissible as the authority of a request: one or
 *        more request-target bytes, which admits a name, an IPv4 or bracketed
 *        IPv6 literal and a port, and refuses a space, a CR, an LF, a NUL and
 *        every other control byte.
 */
static inline int eshkol_http_valid_authority(const char* host) {
    size_t i = 0;
    size_t length = 0;
    if (!host || !*host) return 0;
    while (host[length]) length++;

    /* A bracketed IPv6 literal, optionally followed by ":" port. */
    if (host[0] == '[') {
        size_t close = 1;
        while (close < length && host[close] != ']') {
            const unsigned char c = (unsigned char)host[close];
            if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') ||
                  (c >= 'A' && c <= 'F') || c == ':' || c == '.' || c == '%'))
                return 0;
            close++;
        }
        if (close == 1 || close >= length) return 0;
        i = close + 1;
    } else {
        /* A registered name or an IPv4 literal: no "/", "?", "#", "@", "[",
         * "]" or ":" among them, so `evil.test/x?y` is not a host and a
         * userinfo cannot be smuggled in front of one. */
        while (i < length && host[i] != ':') {
            const unsigned char c = (unsigned char)host[i];
            if (!eshkol_http_is_target_char(c)) return 0;
            if (c == '/' || c == '?' || c == '@' || c == '[' || c == ']')
                return 0;
            i++;
        }
        if (i == 0) return 0;  /* an empty host */
    }

    if (i == length) return 1;          /* no port */
    if (host[i] != ':') return 0;
    if (i + 1 == length) return 0;      /* a colon with no port */
    for (i = i + 1; i < length; ++i) {
        if (host[i] < '0' || host[i] > '9') return 0;
    }
    return 1;
}

/**
 * @brief True when `url` is an absolute-form target: a scheme, `://`, an
 *        authority and an optional path, built only of target bytes.
 *
 * This is the shape every Eshkol HTTP client takes as a URL. A space, a CR,
 * an LF, a NUL or any other control byte refuses it before the transport sees
 * it, whatever the transport's own parser would have made of it.
 */
static inline int eshkol_http_valid_absolute_url(const char* url) {
    size_t i = 0;
    if (!url || !*url) return 0;
    if (!eshkol_http_valid_request_target(url)) return 0;
    while (url[i] && url[i] != ':') i++;
    return url[i] == ':' && url[i + 1] == '/' && url[i + 2] == '/' &&
           url[i + 3] != '\0';
}

/* ─── Header blocks ───────────────────────────────────────────────────────── */

/**
 * @brief Validates a newline-delimited header block.
 *
 * The block is the shape every Eshkol HTTP entry point takes for extra
 * headers: zero or more `Name: value` lines separated by LF, each line
 * optionally ending in CR, with a trailing separator permitted. Each line must
 * satisfy eshkol_http_valid_header_line().
 *
 * @param block      Header block; NULL or empty is a valid empty block.
 * @param line_index When the block is refused and this is non-NULL, receives
 *                   the zero-based index of the line that was refused.
 * @return 1 when every line fits the grammar, 0 otherwise.
 */
static inline int eshkol_http_valid_header_block(const char* block,
                                                 size_t* line_index) {
    char line[1024];
    size_t pos = 0;
    size_t index = 0;
    if (!block || !*block) return 1;
    while (block[pos]) {
        size_t len = 0;
        while (block[pos + len] && block[pos + len] != '\n') len++;
        /* A CR immediately before the LF is the line terminator, not content. */
        {
            size_t content = len;
            if (content > 0 && block[pos + content - 1] == '\r') content--;
            if (content > 0) {
                if (content >= sizeof(line)) {
                    if (line_index) *line_index = index;
                    return 0;
                }
                {
                    size_t i;
                    for (i = 0; i < content; ++i) line[i] = block[pos + i];
                }
                line[content] = '\0';
                if (!eshkol_http_valid_header_line(line)) {
                    if (line_index) *line_index = index;
                    return 0;
                }
                index++;
            }
        }
        pos += len;
        if (block[pos] == '\n') pos++;
    }
    return 1;
}

/**
 * @brief Re-renders a validated header block with canonical CRLF endings.
 *
 * The caller's bytes are never forwarded; the block written to `out` is built
 * line by line from the parse, each line terminated with CRLF, so a request
 * assembled from `out` has exactly the lines the grammar admitted. Empty lines
 * in the input are dropped rather than carried through.
 *
 * @param block   Header block to render; NULL or empty writes an empty string.
 * @param out     Destination buffer, NUL-terminated on success.
 * @param out_cap Capacity of `out` in bytes, including the NUL.
 * @param out_len When non-NULL, receives the rendered length excluding the NUL.
 * @return 1 on success; 0 when the block does not fit the grammar or `out` is
 *         too small for the rendering.
 */
static inline int eshkol_http_render_header_block(const char* block,
                                                  char* out, size_t out_cap,
                                                  size_t* out_len) {
    size_t pos = 0;
    size_t written = 0;
    if (!out || out_cap == 0) return 0;
    out[0] = '\0';
    if (out_len) *out_len = 0;
    if (!block || !*block) return 1;
    if (!eshkol_http_valid_header_block(block, NULL)) return 0;
    while (block[pos]) {
        size_t len = 0;
        size_t content;
        while (block[pos + len] && block[pos + len] != '\n') len++;
        content = len;
        if (content > 0 && block[pos + content - 1] == '\r') content--;
        if (content > 0) {
            size_t i;
            if (written + content + 2 >= out_cap) return 0;
            for (i = 0; i < content; ++i) out[written + i] = block[pos + i];
            written += content;
            out[written++] = '\r';
            out[written++] = '\n';
        }
        pos += len;
        if (block[pos] == '\n') pos++;
    }
    out[written] = '\0';
    if (out_len) *out_len = written;
    return 1;
}

/* ─── Length-aware forms ──────────────────────────────────────────────────── */
/*
 * A header section may contain a NUL: the server accumulates bytes by count
 * and scans only for the blank line. Validating through a NUL-terminated
 * pointer would then check the bytes before the first NUL and silently ignore
 * the rest, so `G\0ET` would validate as the method `G`. These forms take the
 * length the caller actually holds and refuse an embedded NUL outright, which
 * is also what the grammar says: NUL is neither a tchar nor a target char.
 *
 * Each requires `text` to be NUL-terminated at `length`, which is what
 * std::string::data() and a C string both give.
 */

/** @brief eshkol_http_valid_method() over exactly `length` bytes. */
static inline int eshkol_http_valid_method_n(const char* text, size_t length) {
    return eshkol_http_valid_token(text, length);
}

/* The two forms below copy `length` bytes into a local buffer and terminate
 * it there, so the terminated check sees exactly those bytes and not whatever
 * follows them in the caller's buffer. A NUL among them refuses the input: it
 * is neither a target char nor a field-value byte. */

/** @brief eshkol_http_valid_request_target() over exactly `length` bytes. */
static inline int eshkol_http_valid_request_target_n(const char* text,
                                                     size_t length) {
    char buffer[8192];
    size_t i;
    if (!text || length == 0 || length >= sizeof(buffer)) return 0;
    for (i = 0; i < length; ++i) {
        if (text[i] == '\0') return 0;
        buffer[i] = text[i];
    }
    buffer[length] = '\0';
    return eshkol_http_valid_request_target(buffer);
}

/**
 * @brief A received field line: the sending grammar, plus HTAB in the value.
 *
 * Eshkol refuses HTAB in a value it *sends*, because a proxy on the path may
 * re-split a value on it. RFC 9110 section 5.5 nonetheless admits HTAB in a
 * field value, and real `Cookie` and `User-Agent` strings carry it, so a
 * server that refused one would reject conforming traffic. Receiving is
 * therefore held to the RFC's set and sending to the stricter one; every
 * other control byte, CR, LF, NUL and DEL included, is refused on both.
 *
 * @param text   Field line, not NUL-terminated.
 * @param length Length of `text` in bytes.
 * @return 1 when the line is one a peer may legitimately have sent.
 */
static inline int eshkol_http_valid_received_header_line_n(const char* text,
                                                           size_t length) {
    size_t name_length = 0;
    size_t value_start;
    size_t value_end;
    size_t i;
    if (!text || length == 0) return 0;
    while (name_length < length && text[name_length] != ':') name_length++;
    if (name_length == length) return 0;  /* no colon: not a field line */
    if (!eshkol_http_valid_token(text, name_length)) return 0;
    value_start = name_length + 1;
    value_end = length;
    while (value_start < value_end &&
           (text[value_start] == ' ' || text[value_start] == '\t'))
        value_start++;
    while (value_end > value_start &&
           (text[value_end - 1] == ' ' || text[value_end - 1] == '\t'))
        value_end--;
    for (i = value_start; i < value_end; ++i) {
        const unsigned char c = (unsigned char)text[i];
        if (c == '\t') continue;  /* RFC 9110 admits HTAB inside a value */
        if (c < 0x20 || c == 0x7f) return 0;
    }
    return 1;
}

/**
 * @brief eshkol_http_valid_header_line() over exactly `length` bytes.
 *
 * Length-aware throughout rather than by copying, so a legal field line of
 * any length is judged on its own bytes: the server's header budget, not a
 * buffer here, is what bounds a header section.
 */
static inline int eshkol_http_valid_header_line_n(const char* text,
                                                  size_t length) {
    size_t name_length = 0;
    size_t value_start;
    size_t value_length;
    if (!text || length == 0) return 0;
    while (name_length < length && text[name_length] != ':') name_length++;
    if (name_length == length) return 0;  /* no colon: not a field line */
    value_start = name_length + 1;
    value_length = length - value_start;
    while (value_length && text[value_start] == ' ') {
        value_start++;
        value_length--;
    }
    return eshkol_http_valid_token(text, name_length) &&
           eshkol_http_valid_field_value(text + value_start, value_length);
}

/* ─── Fields the client sets itself ───────────────────────────────────────── */

/**
 * @brief True when `length` bytes at `name` name a field an Eshkol client
 *        writes itself, so a caller may not also supply it.
 *
 * Two kinds of field are reserved. The framing and connection-control fields
 * (`Content-Length`, `Transfer-Encoding`, `Host`, `Connection`, `Upgrade`) are
 * emitted by every client from the request it was given; a caller-supplied
 * second copy puts two framing headers on the wire, and RFC 9112 section 6.3
 * forbids a sender emitting both a `Content-Length` and a `Transfer-Encoding`.
 * A front end honouring one and a back end honouring the other desynchronise,
 * which is request smuggling. The handshake fields (`Sec-WebSocket-Key`,
 * `-Version`, `-Accept`, `-Extensions`, `-Protocol`) are derived per connection
 * from the CSPRNG and must not be forced by a caller.
 *
 * @param name       Field name, not NUL-terminated.
 * @param length     Length of `name` in bytes.
 * @param websocket  Non-zero to reserve the `Sec-WebSocket-*` set as well.
 * @return 1 when the name is reserved, 0 otherwise.
 */
static inline int eshkol_http_reserved_field_name(const char* name,
                                                  size_t length,
                                                  int websocket) {
    static const char* const framing[] = {
        "content-length", "transfer-encoding", "host", "connection", "upgrade"
    };
    static const char* const handshake[] = {
        "sec-websocket-key", "sec-websocket-version", "sec-websocket-accept",
        "sec-websocket-extensions", "sec-websocket-protocol"
    };
    size_t set;
    if (!name) return 0;
    for (set = 0; set < 2; ++set) {
        const char* const* names = set == 0 ? framing : handshake;
        size_t count = set == 0 ? sizeof(framing) / sizeof(framing[0])
                                : sizeof(handshake) / sizeof(handshake[0]);
        size_t n;
        if (set == 1 && !websocket) break;
        for (n = 0; n < count; ++n) {
            size_t i;
            if (strlen(names[n]) != length) continue;
            for (i = 0; i < length; ++i) {
                unsigned char c = (unsigned char)name[i];
                if (c >= 'A' && c <= 'Z') c = (unsigned char)(c - 'A' + 'a');
                if ((char)c != names[n][i]) break;
            }
            if (i == length) return 1;
        }
    }
    return 0;
}

/**
 * @brief True when any line of `block` names a reserved field.
 *
 * The block is walked the same way eshkol_http_valid_header_block() walks it,
 * so a name is recognised only at a line start: a reserved name appearing
 * inside another field's value is not a match.
 *
 * @param block      Header block; NULL or empty names nothing.
 * @param websocket  Non-zero to reserve the `Sec-WebSocket-*` set as well.
 * @param line_index When a line is refused and this is non-NULL, receives the
 *                   zero-based index of that line.
 * @return 1 when a reserved field is named, 0 otherwise.
 */
static inline int eshkol_http_block_names_reserved_field(const char* block,
                                                         int websocket,
                                                         size_t* line_index) {
    size_t pos = 0;
    size_t index = 0;
    if (!block || !*block) return 0;
    while (block[pos]) {
        size_t len = 0;
        size_t content;
        while (block[pos + len] && block[pos + len] != '\n') len++;
        content = len;
        if (content > 0 && block[pos + content - 1] == '\r') content--;
        if (content > 0) {
            size_t name_length = 0;
            while (name_length < content && block[pos + name_length] != ':')
                name_length++;
            if (name_length < content &&
                eshkol_http_reserved_field_name(block + pos, name_length,
                                                websocket)) {
                if (line_index) *line_index = index;
                return 1;
            }
            index++;
        }
        pos += len;
        if (block[pos] == '\n') pos++;
    }
    return 0;
}

/* ─── One answer for a whole request line ─────────────────────────────────── */

/**
 * @brief Names the field of a request that does not fit the grammar.
 *
 * @param method       Request method, or NULL to skip the method check.
 * @param target       Request target, or NULL to skip the target check.
 * @param header_block Newline-delimited extra headers, or NULL for none.
 * @return NULL when the request line and headers are admissible; otherwise a
 *         static string naming the field: "method", "request-target",
 *         "header" or "framing header". The string is stable and safe to put
 *         in an error message.
 * @param websocket    Non-zero to reserve the `Sec-WebSocket-*` set as well.
 */
static inline const char* eshkol_http_request_offending_field_ex(
    const char* method, const char* target, const char* header_block,
    int websocket) {
    if (method && !eshkol_http_valid_method(method)) return "method";
    if (target && !eshkol_http_valid_request_target(target))
        return "request-target";
    if (!eshkol_http_valid_header_block(header_block, NULL)) return "header";
    if (eshkol_http_block_names_reserved_field(header_block, websocket, NULL))
        return "framing header";
    return NULL;
}

/**
 * @brief eshkol_http_request_offending_field_ex() for a plain HTTP request.
 */
static inline const char* eshkol_http_request_offending_field(
    const char* method, const char* target, const char* header_block) {
    return eshkol_http_request_offending_field_ex(method, target, header_block,
                                                  0);
}

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif /* ESHKOL_CORE_HTTP_SYNTAX_H */
