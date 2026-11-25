/*
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 *
 * Unit tests for RFC 9112 raw-socket HTTP response status-line, header,
 * framing-smuggling, and chunked-body validation in http_syntax.h.
 */

#include "eshkol/core/http_syntax.h"

#include <stdio.h>
#include <string.h>

static int g_failures = 0;

static void check(int cond, const char* name) {
    if (!cond) {
        fprintf(stderr, "FAIL: %s\n", name);
        g_failures++;
    }
}

int main(void) {
    eshkol_http_parsed_response_t parsed;
    const char* offending = NULL;

    /* 1. Valid Content-Length response */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\nContent-Type: text/plain\r\n\r\nhello";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 1,
              "valid Content-Length response accepted");
        check(parsed.status == 200, "status 200 parsed");
        check(parsed.has_content_length == 1 && parsed.content_length == 5, "Content-Length 5");
        check(parsed.is_chunked == 0, "not chunked");
        check(parsed.body_length == 5 && memcmp(raw + parsed.body_offset, "hello", 5) == 0,
              "body matches hello");
    }

    /* 2. Identical repeated Content-Length allowed by RFC 9110 */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nContent-Length: 4, 4\r\nContent-Length: 4\r\n\r\ntest";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 1,
              "identical repeated Content-Length accepted");
    }

    /* 3. Conflicting Content-Length rejected */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nContent-Length: 5\r\n\r\nhello";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "conflicting Content-Length rejected");
    }

    /* 4. Simultaneous Transfer-Encoding + Content-Length (smuggling) rejected */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 5\r\n\r\n0\r\n\r\n";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "simultaneous TE + CL rejected");
    }

    /* 5. Truncated Content-Length body rejected */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "truncated CL body rejected");
    }

    /* 6. Excess trailing bytes after Content-Length rejected */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhelloSMUGGLED";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "excess bytes after CL rejected");
    }

    /* 7. Bare LF in response header rejected */
    {
        const char* raw = "HTTP/1.1 200 OK\r\nX-A: 1\nX-B: 2\r\n\r\n";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "bare LF in response header rejected");
    }

    /* 8. NUL byte inside response headers rejected */
    {
        const char raw[] = "HTTP/1.1 200 OK\r\nX-A: a\0b\r\n\r\n";
        check(eshkol_http_parse_response(raw, sizeof(raw) - 1, &parsed, &offending) == 0,
              "NUL byte in response headers rejected");
    }

    /* 9. Malformed status line rejected */
    {
        const char* raw = "HTTP/2.0 200 OK\r\n\r\n";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 0,
              "non-1.x status line rejected");
        const char* raw2 = "HTTP/1.1 999 Bad\r\n\r\n";
        check(eshkol_http_parse_response(raw2, strlen(raw2), &parsed, &offending) == 0,
              "out-of-range status code rejected");
    }

    /* 10. Valid chunked transfer decoding with extension and trailer */
    {
        const char* raw =
            "HTTP/1.1 200 OK\r\n"
            "Transfer-Encoding: chunked\r\n\r\n"
            "5;ext=1\r\nhello\r\n"
            "6\r\n world\r\n"
            "0\r\n"
            "X-Trailer: ok\r\n\r\n";
        check(eshkol_http_parse_response(raw, strlen(raw), &parsed, &offending) == 1,
              "chunked headers accepted");
        check(parsed.is_chunked == 1, "is_chunked set");
        char decoded[64];
        size_t decoded_len = 0;
        check(eshkol_http_decode_chunked(raw + parsed.body_offset, parsed.body_length,
                                         decoded, sizeof(decoded), &decoded_len) == 1,
              "chunked body decoded");
        check(decoded_len == 11 && memcmp(decoded, "hello world", 11) == 0,
              "chunked body equals 'hello world'");
    }

    /* 11. Malformed / truncated / trailing-garbage chunked bodies rejected */
    {
        char decoded[64];
        size_t decoded_len = 0;
        const char* bad_trunc = "5\r\nhel";
        check(eshkol_http_decode_chunked(bad_trunc, strlen(bad_trunc),
                                         decoded, sizeof(decoded), &decoded_len) == 0,
              "truncated chunk rejected");
        const char* bad_trail = "0\r\n\r\nEXTRA";
        check(eshkol_http_decode_chunked(bad_trail, strlen(bad_trail),
                                         decoded, sizeof(decoded), &decoded_len) == 0,
              "trailing bytes after terminal chunk rejected");
        const char* bad_size = "ZZ\r\nhi\r\n0\r\n\r\n";
        check(eshkol_http_decode_chunked(bad_size, strlen(bad_size),
                                         decoded, sizeof(decoded), &decoded_len) == 0,
              "non-hex chunk size rejected");
    }

    if (g_failures > 0) {
        fprintf(stderr, "http_response_framing_test: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("PASS: http_response_framing_test\n");
    return 0;
}
