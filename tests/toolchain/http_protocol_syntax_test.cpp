/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * http_protocol_syntax_test.cpp — what <eshkol/core/http_syntax.h> admits and
 * what it refuses.
 *
 * This is the grammar every Eshkol HTTP client reads, so this test is the
 * statement of record for it: each rule gets a value that satisfies it and a
 * value that does not, and the request lines a program might be handed from
 * elsewhere appear verbatim.
 */

#include <eshkol/core/http_syntax.h>

#include <cstring>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void expect(bool condition, const std::string& label) {
    if (!condition) {
        std::cerr << "FAIL: " << label << std::endl;
        ++g_failures;
    }
}

void expect_field(const char* method, const char* target, const char* headers,
                  const char* expected_field, const std::string& label) {
    const char* actual =
        eshkol_http_request_offending_field(method, target, headers);
    const bool ok = expected_field
                        ? (actual != nullptr && std::strcmp(actual, expected_field) == 0)
                        : (actual == nullptr);
    if (!ok) {
        std::cerr << "FAIL: " << label << "\n  expected: "
                  << (expected_field ? expected_field : "(admitted)")
                  << "\n  actual:   " << (actual ? actual : "(admitted)")
                  << std::endl;
        ++g_failures;
    }
}

}  // namespace

int main() {
    // ─── Method: an RFC 9110 token, and nothing else ────────────────────────
    expect(eshkol_http_valid_method("GET"), "GET is a token");
    expect(eshkol_http_valid_method("PROPFIND"), "PROPFIND is a token");
    expect(eshkol_http_valid_method("X-Custom!method"), "tchar set is admitted");
    expect(!eshkol_http_valid_method(""), "an empty method is not a token");
    expect(!eshkol_http_valid_method("GET "), "a trailing space ends the token");
    expect(!eshkol_http_valid_method("GET\r\nX: 1"), "CRLF in a method is refused");
    expect(!eshkol_http_valid_method("GET\n"), "LF in a method is refused");
    expect(!eshkol_http_valid_method("G/ET"), "a solidus is not a tchar");
    expect(!eshkol_http_valid_method(nullptr), "a missing method is refused");

    // ─── Request target: the four RFC 9112 forms ────────────────────────────
    expect(eshkol_http_valid_request_target("/"), "origin form, root");
    expect(eshkol_http_valid_request_target("/a/b?c=d&e=%20f"),
           "origin form with a query");
    expect(eshkol_http_valid_request_target("*"), "asterisk form");
    expect(eshkol_http_valid_request_target("http://example.test/a"),
           "absolute form");
    expect(eshkol_http_valid_request_target("example.test:443"),
           "authority form");
    expect(!eshkol_http_valid_request_target(""), "an empty target is refused");
    expect(!eshkol_http_valid_request_target("/a b"),
           "a space in a target is refused");
    expect(!eshkol_http_valid_request_target("/a\tb"),
           "a tab in a target is refused");
    expect(!eshkol_http_valid_request_target("/a\x7f"),
           "DEL in a target is refused");
    expect(!eshkol_http_valid_request_target("//evil.test/a"),
           "a protocol-relative target is not an origin form");
    expect(!eshkol_http_valid_request_target("example.test:http"),
           "an authority form needs a decimal port");
    expect(!eshkol_http_valid_request_target("relative/path"),
           "a target with no leading solidus and no scheme is refused");

    // Exactly the target from the audit's injection example. Everything after
    // the space is a second request line to any parser that reads the line.
    expect(!eshkol_http_valid_request_target(
               "/a HTTP/1.1\r\nX: 1\r\n\r\nGET /internal HTTP/1.1\r\n"
               "Host: 10.0.0.5"),
           "a target carrying a whole second request is refused");

    // ─── Authority: what may appear in a Host field ─────────────────────────
    expect(eshkol_http_valid_authority("example.test"), "a host name");
    expect(eshkol_http_valid_authority("example.test:8080"), "a host and port");
    expect(eshkol_http_valid_authority("127.0.0.1"), "an IPv4 literal");
    expect(eshkol_http_valid_authority("[::1]:443"),
           "a bracketed IPv6 literal and port");
    expect(!eshkol_http_valid_authority(""), "an empty authority is refused");
    expect(!eshkol_http_valid_authority("a b"),
           "a space in an authority is refused");
    expect(!eshkol_http_valid_authority("evil\r\nX: 1"),
           "CRLF in an authority is refused");
    // A Host is host[:port] and nothing else, so a path, a query, a userinfo
    // or a non-numeric port cannot ride along in one.
    expect(!eshkol_http_valid_authority("evil.test/x?y"),
           "a path in an authority is refused");
    expect(!eshkol_http_valid_authority("evil.test?y"),
           "a query in an authority is refused");
    expect(!eshkol_http_valid_authority("user@evil.test"),
           "a userinfo in an authority is refused");
    expect(!eshkol_http_valid_authority("example.test:http"),
           "a non-numeric port is refused");
    expect(!eshkol_http_valid_authority("example.test:"),
           "a colon with no port is refused");
    expect(!eshkol_http_valid_authority("[::1"),
           "an unterminated IPv6 literal is refused");
    expect(eshkol_http_valid_authority("[::1]"),
           "a bracketed IPv6 literal with no port");

    // ─── Absolute URLs, the shape the clients take ──────────────────────────
    expect(eshkol_http_valid_absolute_url("https://example.test/a?b=c"),
           "an https URL is absolute");
    expect(!eshkol_http_valid_absolute_url("/a"),
           "an origin form is not an absolute URL");
    expect(!eshkol_http_valid_absolute_url("http://"),
           "a scheme with no authority is refused");
    expect(!eshkol_http_valid_absolute_url(
               "http://10.0.0.5/a HTTP/1.1\r\nX: 1\r\n\r\nGET /internal "
               "HTTP/1.1\r\nHost: 10.0.0.5"),
           "the audit's injected URL is refused as a URL");

    // ─── Header lines and blocks ────────────────────────────────────────────
    expect(eshkol_http_valid_header_line("Accept: application/json"),
           "an ordinary header line");
    expect(eshkol_http_valid_header_line("X-Empty:"),
           "an empty field value is admitted");
    expect(!eshkol_http_valid_header_line("Accept application/json"),
           "a header line needs a colon");
    expect(!eshkol_http_valid_header_line("Ac cept: x"),
           "a space in a field name is refused");
    expect(!eshkol_http_valid_header_line("X: a\rY: b"),
           "a CR in a field value is refused");
    expect(!eshkol_http_valid_header_line("X: a\nY: b"),
           "an LF in a field value is refused");
    expect(!eshkol_http_valid_header_line("X: a\tY: b"),
           "a tab in a field value is refused");

    size_t line_index = 0;
    expect(eshkol_http_valid_header_block(nullptr, &line_index),
           "no header block is a valid empty block");
    expect(eshkol_http_valid_header_block("", &line_index),
           "an empty header block is valid");
    expect(eshkol_http_valid_header_block("A: 1\nB: 2\n", &line_index),
           "two well-formed lines");
    expect(eshkol_http_valid_header_block("A: 1\r\nB: 2\r\n", &line_index),
           "CRLF line endings in the block");
    expect(!eshkol_http_valid_header_block("A: 1\nB 2\n", &line_index),
           "a block with a colonless line is refused");
    expect(line_index == 1, "the refused line is named by index");

    // ─── Rendering: the caller's bytes never reach the transport ────────────
    char rendered[256];
    size_t rendered_length = 0;
    expect(eshkol_http_render_header_block("A: 1\nB: 2\n", rendered,
                                           sizeof(rendered), &rendered_length),
           "a valid block renders");
    expect(std::string(rendered) == "A: 1\r\nB: 2\r\n",
           "rendering uses canonical CRLF endings");
    expect(rendered_length == 12, "the rendered length is reported");

    expect(eshkol_http_render_header_block("A: 1\n\n\nB: 2", rendered,
                                           sizeof(rendered), nullptr),
           "blank lines in the input are dropped");
    expect(std::string(rendered) == "A: 1\r\nB: 2\r\n",
           "a block with blank lines renders as its two lines");

    expect(!eshkol_http_render_header_block("X: a\rY: b", rendered,
                                            sizeof(rendered), nullptr),
           "a block that does not fit the grammar does not render");

    char tiny[8];
    expect(!eshkol_http_render_header_block("A: 1\nB: 2\n", tiny, sizeof(tiny),
                                            nullptr),
           "a rendering that does not fit the buffer is refused");

    // ─── One answer per request, naming the field ───────────────────────────
    expect_field("GET", "/health", "Accept: application/json\n", nullptr,
                 "a well-formed request is admitted");
    expect_field("GET\r\nX: 1", "/health", "", "method",
                 "a method carrying CRLF names the method");
    expect_field("GET", "/a HTTP/1.1\r\nX: 1", "", "request-target",
                 "a target carrying CRLF names the request target");
    expect_field("GET", "/health", "X: a\rInjected: 1", "header",
                 "a bare CR inside a header line names the header");
    expect(!eshkol_http_valid_header_line("X: a\r\nInjected: 1"),
           "CRLF inside one field value is refused");
    expect(!eshkol_http_valid_field_value("a\rb", 3),
           "a bare CR is not a field value byte");
    expect_field("GET", "/health", "X: a\r\nInjected: 1", nullptr,
                 "CRLF separates two header lines, each validated on its own");
    expect_field("GET", "/health", "Bad Header: 1", "header",
                 "a field name with a space names the header");

    // ─── Fields the client sets itself (RFC 9112 section 6.3) ──────────────
    // The reviewer's input: every byte inside the grammar, and the request it
    // would have produced carries both a Transfer-Encoding and the client's
    // own Content-Length.
    expect_field("POST", "/x", "Transfer-Encoding: chunked", "framing header",
                 "a caller Transfer-Encoding names the framing header");
    expect_field("POST", "/x", "Content-Length: 0", "framing header",
                 "a caller Content-Length names the framing header");
    expect_field("GET", "/x", "Host: internal", "framing header",
                 "a caller Host names the framing header");
    expect_field("GET", "/x", "Connection: keep-alive", "framing header",
                 "a caller Connection names the framing header");
    expect_field("GET", "/x", "Upgrade: h2c", "framing header",
                 "a caller Upgrade names the framing header");
    expect_field("POST", "/x", "transfer-ENCODING: chunked", "framing header",
                 "the match is case-insensitive");
    expect_field("POST", "/x", "Accept: text/plain\nContent-Length: 5",
                 "framing header",
                 "a reserved field on a later line is found");
    expect_field("GET", "/x", "X-Note: Content-Length: 5", nullptr,
                 "a reserved name inside another value is not a match");
    expect_field("GET", "/x", "Content-Length-Note: 5", nullptr,
                 "a longer name that starts with a reserved one is admitted");

    // The Sec-WebSocket-* set is reserved only for the WebSocket client, which
    // derives those values per connection from the CSPRNG.
    expect(eshkol_http_request_offending_field_ex(
               "GET", "/x", "Sec-WebSocket-Key: AAAAAAAAAAAAAAAAAAAAAA==", 1) !=
               nullptr,
           "a caller WebSocket key is refused on the WebSocket path");
    expect(eshkol_http_request_offending_field_ex(
               "GET", "/x", "Sec-WebSocket-Version: 13", 1) != nullptr,
           "a caller WebSocket version is refused on the WebSocket path");
    expect(eshkol_http_request_offending_field_ex(
               "GET", "/x", "Sec-WebSocket-Protocol: chat", 0) == nullptr,
           "the WebSocket set is not reserved on the plain HTTP path");

    expect(eshkol_http_reserved_field_name("Content-Length", 14, 0),
           "the predicate matches a reserved name");
    expect(!eshkol_http_reserved_field_name("Accept", 6, 0),
           "the predicate admits an ordinary name");

    if (g_failures == 0) {
        std::cout << "http_protocol_syntax_test: all cases pass" << std::endl;
        return 0;
    }
    std::cerr << "http_protocol_syntax_test: " << g_failures << " case(s) failed"
              << std::endl;
    return 1;
}
