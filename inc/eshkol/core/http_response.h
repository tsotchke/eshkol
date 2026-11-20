/* Bounded HTTP/1.x response framing for the POSIX raw socket clients. */
#ifndef ESHKOL_CORE_HTTP_RESPONSE_H
#define ESHKOL_CORE_HTTP_RESPONSE_H

#include "http_syntax.h"
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <sys/socket.h>

#define ESHKOL_HTTP_RESPONSE_METADATA_MAX (64u * 1024u)
#define ESHKOL_HTTP_RESPONSE_BODY_MAX (10u * 1024u * 1024u)
#define ESHKOL_HTTP_RESPONSE_WIRE_MAX (ESHKOL_HTTP_RESPONSE_METADATA_MAX + ESHKOL_HTTP_RESPONSE_BODY_MAX)

typedef struct {
    int fd, status;
    char *wire;
    size_t used, capacity, pos, metadata;
    size_t head, head_len, body, body_len;
} eshkol_http_response_reader;

static inline ssize_t eshkol_http_recv(int fd, char *buf, size_t size) {
    ssize_t n;
    do { n = recv(fd, buf, size, 0); } while (n < 0 && errno == EINTR);
    return n;
}

/* Append a block. EOF is distinct from a timeout or reset, including at
 * the exact wire limit, where only a one-byte probe is needed. */
static inline const char *eshkol_http_fill(eshkol_http_response_reader *r, int *eof) {
    ssize_t n;
    *eof = 0;
    if (r->used == ESHKOL_HTTP_RESPONSE_WIRE_MAX) {
        char extra;
        n = eshkol_http_recv(r->fd, &extra, 1);
        if (n > 0) return "http-request: response size limit";
    } else {
        if (r->used == r->capacity) {
            size_t cap = r->capacity ? 2 * r->capacity : 4096;
            if (cap > ESHKOL_HTTP_RESPONSE_WIRE_MAX) cap = ESHKOL_HTTP_RESPONSE_WIRE_MAX;
            char *p = (char *)realloc(r->wire, cap);
            if (!p) return "http-request: response allocation failed";
            r->wire = p;
            r->capacity = cap;
        }
        size_t room = r->capacity - r->used;
        n = eshkol_http_recv(r->fd, r->wire + r->used, room < 8192 ? room : 8192);
    }
    if (n < 0) return "http-request: response read failed";
    *eof = n == 0;
    r->used += (size_t)n;
    return NULL;
}

static inline const char *eshkol_http_need(eshkol_http_response_reader *r, size_t count) {
    while (r->used - r->pos < count) {
        int eof;
        const char *err = eshkol_http_fill(r, &eof);
        if (err) return err;
        if (eof) return "http-request: truncated response";
    }
    return NULL;
}

/* Return offsets since subsequent reads may relocate the buffer. Charge all
 * non-body octets, including interim responses, chunk delimiters and trailers. */
static inline const char *eshkol_http_line(eshkol_http_response_reader *r, size_t *start,
                                           size_t *length) {
    *start = r->pos;
    for (;;) {
        if (r->metadata == ESHKOL_HTTP_RESPONSE_METADATA_MAX)
            return "http-request: response metadata limit";
        const char *err = eshkol_http_need(r, 1);
        if (err) return err;
        char c = r->wire[r->pos++];
        ++r->metadata;
        if (c == '\n') return "http-request: invalid response line ending";
        if (c == '\r') {
            if (r->metadata == ESHKOL_HTTP_RESPONSE_METADATA_MAX)
                return "http-request: response metadata limit";
            err = eshkol_http_need(r, 1);
            if (err) return err;
            if (r->wire[r->pos++] != '\n') return "http-request: invalid response line ending";
            ++r->metadata;
            *length = r->pos - *start - 2;
            return NULL;
        }
    }
}

static inline int eshkol_http_name_is(const char *p, size_t n, const char *lower) {
    if (strlen(lower) != n) return 0;
    for (size_t i = 0; i < n; ++i) {
        char c = p[i];
        if (c >= 'A' && c <= 'Z') c = (char)(c + ('a' - 'A'));
        if (c != lower[i]) return 0;
    }
    return 1;
}

static inline int eshkol_http_ows(char c) { return c == ' ' || c == '\t'; }

static inline int eshkol_http_digits(const char *s, size_t n, unsigned base,
                                     uint64_t *value, size_t *digits) {
    uint64_t v = 0;
    size_t i;
    for (i = 0; i < n; ++i) {
        unsigned d;
        if (s[i] >= '0' && s[i] <= '9') d = (unsigned)(s[i] - '0');
        else if (base == 16 && s[i] >= 'a' && s[i] <= 'f') d = (unsigned)(s[i] - 'a') + 10;
        else if (base == 16 && s[i] >= 'A' && s[i] <= 'F') d = (unsigned)(s[i] - 'A') + 10;
        else break;
        if (v > (UINT64_MAX - d) / base) return 0;
        v = v * base + d;
    }
    *digits = i;
    *value = v;
    return i > 0;
}

/* chunk-ext = *( BWS ";" BWS token [ BWS "=" BWS (token / quoted-string) ] ) */
static inline int eshkol_http_chunk_extensions(const char *p, size_t n) {
    size_t i = 0;
    while (i < n) {
        while (i < n && eshkol_http_ows(p[i])) ++i;
        if (i == n || p[i++] != ';') return 0;
        while (i < n && eshkol_http_ows(p[i])) ++i;
        size_t start = i;
        while (i < n && eshkol_http_is_token_char((unsigned char)p[i])) ++i;
        if (i == start) return 0;
        size_t end = i;
        while (i < n && eshkol_http_ows(p[i])) ++i;
        if (i == n || p[i] != '=') {
            i = end;
            continue;
        }
        ++i;
        while (i < n && eshkol_http_ows(p[i])) ++i;
        if (i < n && p[i] == '"') {
            ++i;
            while (i < n && p[i] != '"') {
                if (p[i] == '\\' && ++i == n) return 0;
                unsigned char c = (unsigned char)p[i++];
                if ((c < 32 && c != '\t') || c == 127) return 0;
            }
            if (i == n) return 0;
            ++i;
        } else {
            start = i;
            while (i < n && eshkol_http_is_token_char((unsigned char)p[i])) ++i;
            if (start == i) return 0;
        }
    }
    return 1;
}

typedef struct {
    int has_length, chunked;
    uint64_t length;
} eshkol_http_response_framing;

static inline const char *eshkol_http_fields(eshkol_http_response_reader *r,
                                             eshkol_http_response_framing *f, int trailers) {
    for (;;) {
        size_t start, n;
        const char *err = eshkol_http_line(r, &start, &n);
        if (err || !n) return err;
        const char *line = r->wire + start;
        if (!eshkol_http_valid_received_header_line_n(line, n))
            return "http-request: invalid response field";
        size_t name = (size_t)((const char *)memchr(line, ':', n) - line);
        if (trailers) {
            if (eshkol_http_reserved_field_name(line, name, 0) ||
                eshkol_http_name_is(line, name, "trailer"))
                return "http-request: forbidden response trailer";
            continue;
        }
        size_t i = name + 1;
        while (i < n && eshkol_http_ows(line[i])) ++i;
        while (n > i && eshkol_http_ows(line[n - 1])) --n;
        if (eshkol_http_name_is(line, name, "content-length")) {
            size_t digits;
            if (f->has_length || f->chunked ||
                !eshkol_http_digits(line + i, n - i, 10, &f->length, &digits) ||
                digits != n - i)
                return "http-request: invalid response Content-Length";
            f->has_length = 1;
        } else if (eshkol_http_name_is(line, name, "transfer-encoding")) {
            if (f->chunked || f->has_length ||
                !eshkol_http_name_is(line + i, n - i, "chunked"))
                return "http-request: invalid response Transfer-Encoding";
            f->chunked = 1;
        }
    }
}

static inline const char *eshkol_http_chunks(eshkol_http_response_reader *r) {
    for (;;) {
        size_t start, n, digits;
        uint64_t size;
        const char *err = eshkol_http_line(r, &start, &n);
        if (err) return err;
        const char *line = r->wire + start;
        if (!eshkol_http_digits(line, n, 16, &size, &digits) ||
            !eshkol_http_chunk_extensions(line + digits, n - digits))
            return "http-request: invalid response chunk size";
        if (!size) return eshkol_http_fields(r, NULL, 1);
        if (size > ESHKOL_HTTP_RESPONSE_BODY_MAX - r->body_len)
            return "http-request: response body limit";
        err = eshkol_http_need(r, (size_t)size);
        if (err) return err;
        memmove(r->wire + r->body + r->body_len, r->wire + r->pos, (size_t)size);
        r->pos += (size_t)size;
        r->body_len += (size_t)size;
        if (ESHKOL_HTTP_RESPONSE_METADATA_MAX - r->metadata < 2)
            return "http-request: response metadata limit";
        err = eshkol_http_need(r, 2);
        if (err) return err;
        if (memcmp(r->wire + r->pos, "\r\n", 2))
            return "http-request: invalid response chunk terminator";
        r->pos += 2;
        r->metadata += 2;
    }
}

/* Caller zero-initializes the reader and frees wire on every exit. The
 * header slice includes the final status line and excludes the final CRLFCRLF. */
static inline const char *eshkol_http_read_response(eshkol_http_response_reader *r,
                                                     int fd, const char *method) {
    r->fd = fd;
    for (;;) {
        size_t start, n;
        const char *err = eshkol_http_line(r, &start, &n);
        if (err) return err;
        const char *p = r->wire + start;
        if (n < 13 || (memcmp(p, "HTTP/1.1 ", 9) && memcmp(p, "HTTP/1.0 ", 9)) ||
            p[9] < '1' || p[9] > '5' || p[10] < '0' || p[10] > '9' ||
            p[11] < '0' || p[11] > '9' || p[12] != ' ')
            return "http-request: invalid response status line";
        for (size_t i = 13; i < n; ++i) {
            unsigned char c = (unsigned char)p[i];
            if ((c < 32 && c != '\t') || c == 127) return "http-request: invalid response reason";
        }
        int http10 = p[7] == '0';
        r->status = 100 * (p[9] - '0') + 10 * (p[10] - '0') + p[11] - '0';
        r->head = start;
        eshkol_http_response_framing f = {0, 0, 0};
        err = eshkol_http_fields(r, &f, 0);
        if (err) return err;
        if ((r->status < 200 || r->status == 204) && (f.has_length || f.chunked))
            return "http-request: framing on bodyless response";
        if (http10 && f.chunked) return "http-request: HTTP/1.0 transfer coding";
        if (r->status < 200 && r->status != 101) continue;
        r->head_len = r->pos - start - 4;
        r->body = r->pos;
        if (!strcmp(method, "HEAD") || r->status == 204 || r->status == 304 ||
            r->status == 101 || (!strcmp(method, "CONNECT") && r->status / 100 == 2))
            return NULL;
        if (f.chunked) return eshkol_http_chunks(r);
        if (f.has_length) {
            if (f.length > ESHKOL_HTTP_RESPONSE_BODY_MAX)
                return "http-request: response body limit";
            r->body_len = (size_t)f.length;
            return eshkol_http_need(r, r->body_len);
        }
        for (;;) {
            int eof;
            r->body_len = r->used - r->body;
            if (r->body_len > ESHKOL_HTTP_RESPONSE_BODY_MAX)
                return "http-request: response body limit";
            err = eshkol_http_fill(r, &eof);
            if (err || eof) return err;
        }
    }
}
#endif
