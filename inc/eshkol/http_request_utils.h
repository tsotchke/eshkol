/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * http_request_utils.h — header-only HTTP request parsing utilities shared
 * by eshkol-server and its regression tests.
 *
 * Defence-in-depth for untrusted HTTP input:
 *  1. eshkol_parse_content_length — validates and caps Content-Length
 *  2. eshkol_normalize_header_name — lowercases header names so
 *     case-insensitive lookup works (HTTP/1.1 requirement)
 *  3. eshkol_is_safe_url_path — rejects path traversal after URL decoding
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

// ─── Request budgets ────────────────────────────────────────────────────────
//
// Three numbers bound what one connection can cost: how many bytes of header
// section the server will hold, how many bytes of body, and how long a single
// request has from the moment it was accepted to arrive in full. Each has a
// default below and an environment variable that overrides it, read once per
// process on first use; the accessors are the only readers, so a value is
// consistent for the life of the run.
//
// This follows the shape the reader's own input-size cap uses
// (eshkol_reader_max_source_bytes and ESHKOL_MAX_SOURCE_BYTES): a documented
// default macro, one accessor, one setter, one environment variable. It is a
// separate budget because it governs a different thing — an HTTP request in
// flight, not a source text handed to the reader — and the reader's cap still
// applies downstream to whatever source a request carries.

// Maximum request body the server will accept (10 MB).
// Requests with a Content-Length above this get 413 Payload Too Large.
constexpr size_t kMaxRequestBodySize = 10 * 1024 * 1024;

// Maximum header section (request line and fields, up to and including the
// blank line). A request whose header section passes this gets 431 Request
// Header Fields Too Large.
constexpr size_t kMaxRequestHeaderSize = 64 * 1024;

// Maximum total request size (headers + body) before the read loop gives up.
constexpr size_t kMaxRequestTotalSize = kMaxRequestBodySize + kMaxRequestHeaderSize;

// Seconds a single request has, measured from accept, to arrive complete.
// A connection that has not delivered a whole request by then gets 408
// Request Timeout and is closed; other connections are unaffected.
constexpr int kRequestDeadlineSeconds = 15;

// Seconds a single recv() will block. Bounded well below the request
// deadline so the read loop returns to re-check the deadline.
constexpr int kReceiveTimeoutSeconds = 5;

// Maximum simultaneous connections the server will handle.
constexpr int kMaxConcurrentConnections = 128;

/** @brief Reads a positive-integer environment override, or the fallback. */
inline size_t eshkol_server_size_from_env(const char* name, size_t fallback) {
    const char* raw = std::getenv(name);
    if (!raw || !*raw) return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (!end || *end != '\0' || parsed == 0ULL) return fallback;
    if (parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max()))
        return fallback;
    return static_cast<size_t>(parsed);
}

/** @brief Header-section cap in bytes. Override: ESHKOL_SERVER_MAX_HEADER_BYTES. */
inline size_t eshkol_server_max_header_bytes() {
    static const size_t value =
        eshkol_server_size_from_env("ESHKOL_SERVER_MAX_HEADER_BYTES",
                                    kMaxRequestHeaderSize);
    return value;
}

/** @brief Body cap in bytes. Override: ESHKOL_SERVER_MAX_BODY_BYTES. */
inline size_t eshkol_server_max_body_bytes() {
    static const size_t value =
        eshkol_server_size_from_env("ESHKOL_SERVER_MAX_BODY_BYTES",
                                    kMaxRequestBodySize);
    return value;
}

/** @brief Per-request deadline in seconds, measured from accept.
 *         Override: ESHKOL_SERVER_REQUEST_DEADLINE_SECONDS. */
inline int eshkol_server_request_deadline_seconds() {
    static const int value = static_cast<int>(
        eshkol_server_size_from_env("ESHKOL_SERVER_REQUEST_DEADLINE_SECONDS",
                                    static_cast<size_t>(kRequestDeadlineSeconds)));
    return value;
}

// ─── Content-Length parser ───────────────────────────────────────────────────

/**
 * @brief Parses and validates the value of an HTTP Content-Length header.
 *
 * Trims leading/trailing whitespace, requires the remaining text to be
 * composed entirely of decimal digits, and rejects values that would not
 * fit in a size_t. This function only validates syntax and range; callers
 * are responsible for separately comparing the result against server-side
 * limits such as kMaxRequestBodySize.
 *
 * @param raw_value Raw header value as received from the client (may
 *        contain surrounding whitespace).
 * @param content_length Output parameter set to the parsed length on
 *        success; left unmodified on failure.
 * @return true if raw_value is a valid non-negative integer that fits in
 *         size_t; false if raw_value is empty/whitespace-only, contains
 *         non-digit characters, or overflows size_t.
 */
inline bool eshkol_parse_content_length(const std::string& raw_value, size_t& content_length) {
    size_t start = 0;
    while (start < raw_value.size() && std::isspace(static_cast<unsigned char>(raw_value[start]))) {
        ++start;
    }

    size_t end = raw_value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(raw_value[end - 1]))) {
        --end;
    }

    if (start == end) {
        return false;
    }

    const std::string trimmed = raw_value.substr(start, end - start);
    for (char ch : trimmed) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            return false;
        }
    }

    try {
        const unsigned long long parsed = std::stoull(trimmed);
        if (parsed > static_cast<unsigned long long>(std::numeric_limits<size_t>::max())) {
            return false;
        }
        content_length = static_cast<size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

// ─── Header name normalisation ──────────────────────────────────────────────
//
// HTTP/1.1 header field names are case-insensitive (RFC 7230 sec 3.2).
// Normalise to lowercase so a single unordered_map lookup works regardless
// of how the client capitalises header names. Without this, a client
// sending `content-length: abc` instead of `Content-Length: abc` would
// bypass all Content-Length validation.

/**
 * @brief Returns an ASCII-lowercased copy of an HTTP header field name.
 *
 * @param name Header name to normalize; not modified.
 * @return A new std::string containing the lowercased header name, suitable
 *         as a case-insensitive lookup key (e.g. in an unordered_map keyed
 *         by header name).
 */
inline std::string eshkol_normalize_header_name(const std::string& name) {
    std::string lower = name;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return lower;
}

// ─── URL path safety ────────────────────────────────────────────────────────
//
// Rejects paths that could escape the served directory via:
//   - literal ".." component
//   - percent-encoded ".." (%2e%2e, %2E%2E, mixed case)
//   - backslash separators (Windows)
//   - null bytes
//
// Call AFTER URL-decoding if the server does its own decoding, or call on
// the raw path to catch encoded traversals that the server doesn't decode.

/**
 * @brief Checks whether a request path is safe to resolve against the
 *        server's document root.
 *
 * Rejects the path if it is empty, does not start with '/', contains a
 * null byte or backslash, or — after internally percent-decoding it purely
 * for inspection — contains a ".." path component. This catches both
 * literal and percent-encoded (including mixed-case, e.g. %2E%2e)
 * traversal attempts.
 *
 * @param path Request path to validate.
 * @return true if the path is safe to use; false otherwise.
 * @note This function does not return the decoded path; it only inspects a
 *       local copy to detect traversal. Callers that need the decoded path
 *       must decode it themselves.
 */
inline bool eshkol_is_safe_url_path(const std::string& path) {
    if (path.empty() || path[0] != '/') return false;

    // Reject null bytes
    if (path.find('\0') != std::string::npos) return false;

    // Reject backslash (Windows path separator)
    if (path.find('\\') != std::string::npos) return false;

    // Decode percent-encoded characters for traversal check
    std::string decoded;
    decoded.reserve(path.size());
    for (size_t i = 0; i < path.size(); ++i) {
        if (path[i] == '%' && i + 2 < path.size()) {
            char hi = path[i + 1];
            char lo = path[i + 2];
            auto hex_val = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int h = hex_val(hi), l = hex_val(lo);
            if (h >= 0 && l >= 0) {
                decoded += static_cast<char>((h << 4) | l);
                i += 2;
                continue;
            }
        }
        decoded += path[i];
    }

    // Check for ".." path component in decoded form
    // Split on '/' and reject any component that is exactly ".."
    size_t pos = 0;
    while (pos < decoded.size()) {
        size_t next = decoded.find('/', pos + 1);
        if (next == std::string::npos) next = decoded.size();
        std::string component = decoded.substr(pos, next - pos);
        // Strip leading slash
        if (!component.empty() && component[0] == '/') {
            component = component.substr(1);
        }
        if (component == "..") return false;
        pos = next;
    }

    return true;
}

// ─── Message framing (RFC 9112 section 6) ───────────────────────────────────
//
// How long a request body is, is decided once, from one reading of the header
// section, and that one answer is what both the read loop and the request
// handler use. Reading it twice is how a request comes to mean one length to
// the loop that accumulates it and another to the code that interprets it.
//
// The rules this implements:
//
//   * A field name is recognised only at the start of a line, so
//     `X-Note: content-length: 5` names no framing field.
//   * RFC 9112 section 6.3: a message carrying more than one Content-Length,
//     or a Content-Length whose value is a list, is refused — including the
//     case where the values agree, because agreement is not the rule.
//   * A message carrying both Transfer-Encoding and Content-Length is
//     refused: the two name different framings and a recipient that prefers
//     either one disagrees with a recipient that prefers the other.
//   * A Transfer-Encoding this server does not implement is refused rather
//     than ignored.

enum class eshkol_http_framing_verdict {
    ok,
    invalid_content_length,
    repeated_content_length,
    transfer_encoding_with_content_length,
    unsupported_transfer_encoding,
    header_section_too_large,
    body_too_large
};

struct eshkol_http_framing {
    eshkol_http_framing_verdict verdict = eshkol_http_framing_verdict::ok;
    bool has_content_length = false;
    size_t content_length = 0;
};

/**
 * @brief Collects the values of one header field from a header section,
 *        matching the name only where a line begins.
 *
 * @param head Header section: the request line, the field lines, and
 *        optionally the terminating blank line.
 * @param lowercase_name Field name to collect, in lowercase.
 * @return One entry per field line that carries the name, in order, each
 *         trimmed of surrounding spaces and tabs.
 */
inline std::vector<std::string> eshkol_http_field_values(
    const std::string& head, const std::string& lowercase_name) {
    std::vector<std::string> values;
    size_t line_start = head.find('\n');
    if (line_start == std::string::npos) return values;
    ++line_start;
    while (line_start < head.size()) {
        size_t line_end = head.find('\n', line_start);
        if (line_end == std::string::npos) line_end = head.size();
        size_t content_end = line_end;
        if (content_end > line_start && head[content_end - 1] == '\r') --content_end;
        const size_t line_length = content_end - line_start;
        if (line_length == 0) break;  // end of the header section
        if (line_length > lowercase_name.size() &&
            head[line_start + lowercase_name.size()] == ':') {
            bool name_matches = true;
            for (size_t i = 0; i < lowercase_name.size(); ++i) {
                const unsigned char c =
                    static_cast<unsigned char>(head[line_start + i]);
                if (static_cast<char>(std::tolower(c)) != lowercase_name[i]) {
                    name_matches = false;
                    break;
                }
            }
            if (name_matches) {
                size_t value_start = line_start + lowercase_name.size() + 1;
                size_t value_end = content_end;
                while (value_start < value_end &&
                       (head[value_start] == ' ' || head[value_start] == '\t'))
                    ++value_start;
                while (value_end > value_start &&
                       (head[value_end - 1] == ' ' || head[value_end - 1] == '\t'))
                    --value_end;
                values.emplace_back(head, value_start, value_end - value_start);
            }
        }
        line_start = line_end < head.size() ? line_end + 1 : head.size();
    }
    return values;
}

/**
 * @brief Decides a request's framing from one reading of its header section.
 *
 * @param head Header section, including the terminating blank line.
 * @return The verdict, and when it is `ok` the body length the message
 *         declares (zero when it declares none).
 */
inline eshkol_http_framing eshkol_http_examine_framing(const std::string& head) {
    eshkol_http_framing framing;
    if (head.size() > eshkol_server_max_header_bytes()) {
        framing.verdict = eshkol_http_framing_verdict::header_section_too_large;
        return framing;
    }

    const std::vector<std::string> transfer_encodings =
        eshkol_http_field_values(head, "transfer-encoding");
    const std::vector<std::string> content_lengths =
        eshkol_http_field_values(head, "content-length");

    if (!transfer_encodings.empty() && !content_lengths.empty()) {
        framing.verdict =
            eshkol_http_framing_verdict::transfer_encoding_with_content_length;
        return framing;
    }
    if (!transfer_encodings.empty()) {
        // This server reads a request body only from a declared length. A
        // message framed another way is refused with its name, not read as
        // though it had no body.
        framing.verdict = eshkol_http_framing_verdict::unsupported_transfer_encoding;
        return framing;
    }
    if (content_lengths.size() > 1) {
        framing.verdict = eshkol_http_framing_verdict::repeated_content_length;
        return framing;
    }
    if (content_lengths.empty()) return framing;

    const std::string& raw = content_lengths.front();
    if (raw.find(',') != std::string::npos) {
        // A list-valued Content-Length is the same message as a repeated one.
        framing.verdict = eshkol_http_framing_verdict::repeated_content_length;
        return framing;
    }
    size_t parsed = 0;
    if (!eshkol_parse_content_length(raw, parsed)) {
        framing.verdict = eshkol_http_framing_verdict::invalid_content_length;
        return framing;
    }
    if (parsed > eshkol_server_max_body_bytes()) {
        framing.verdict = eshkol_http_framing_verdict::body_too_large;
        framing.content_length = parsed;
        return framing;
    }
    framing.has_content_length = true;
    framing.content_length = parsed;
    return framing;
}

/** @brief The HTTP status a framing verdict is answered with. */
inline int eshkol_http_framing_status(eshkol_http_framing_verdict verdict) {
    switch (verdict) {
        case eshkol_http_framing_verdict::ok: return 200;
        case eshkol_http_framing_verdict::header_section_too_large: return 431;
        case eshkol_http_framing_verdict::body_too_large: return 413;
        default: return 400;
    }
}

/** @brief A short, stable description of a framing verdict, for the response
 *         body. It names the rule, not the client's bytes. */
inline const char* eshkol_http_framing_message(
    eshkol_http_framing_verdict verdict) {
    switch (verdict) {
        case eshkol_http_framing_verdict::ok:
            return "ok";
        case eshkol_http_framing_verdict::invalid_content_length:
            return "Content-Length is not a single decimal length";
        case eshkol_http_framing_verdict::repeated_content_length:
            return "Content-Length appears more than once";
        case eshkol_http_framing_verdict::transfer_encoding_with_content_length:
            return "Transfer-Encoding and Content-Length both present";
        case eshkol_http_framing_verdict::unsupported_transfer_encoding:
            return "Transfer-Encoding is not supported by this server";
        case eshkol_http_framing_verdict::header_section_too_large:
            return "Header fields exceed the configured limit";
        case eshkol_http_framing_verdict::body_too_large:
            return "Payload exceeds the configured limit";
    }
    return "Request could not be framed";
}

// ─── Request accumulation ───────────────────────────────────────────────────
//
// One object owns the whole question "have I got a request yet, and is it one
// I will answer?". It is fed the exact bytes a read returned — a count, never
// a NUL-terminated string, so a body containing NUL bytes arrives whole — and
// it answers need_more, complete or refused. The framing is examined once, on
// the header section it has in hand, and the answer it gives is the one the
// caller uses to decide when the body has arrived and the one the handler
// uses to interpret it. There is no second reading to disagree with the
// first.
//
// Both the server's read loop and the tests drive this same object, so what
// the tests observe is what a connection gets.

class eshkol_http_request_accumulator {
public:
    enum class state { need_more, complete, refused };

    /** @brief Feeds exactly `length` bytes. NUL bytes are content. */
    state append(const char* bytes, size_t length) {
        if (state_ != state::need_more) return state_;
        if (bytes && length > 0) {
            const size_t resume = data_.size() >= 3 ? data_.size() - 3 : 0;
            data_.append(bytes, length);
            if (header_end_ == std::string::npos) {
                header_end_ = data_.find("\r\n\r\n", resume);
                if (header_end_ != std::string::npos) header_end_ += 4;
            }
        }
        return reconsider();
    }

    /** @brief Marks the peer as having stopped sending. A request that is
     *         still incomplete at that point is not one this server answers. */
    state close() {
        if (state_ == state::need_more) {
            if (data_.empty()) {
                state_ = state::complete;  // nothing was sent; nothing to answer
            } else {
                refuse(400, "Request ended before it was complete");
            }
        }
        return state_;
    }

    state current() const { return state_; }
    const std::string& data() const { return data_; }
    size_t header_end() const { return header_end_; }
    const eshkol_http_framing& framing() const { return framing_; }
    int refusal_status() const { return refusal_status_; }
    const std::string& refusal_message() const { return refusal_message_; }

    /** @brief Declared body length once the header section has been read. */
    size_t declared_body_length() const {
        return framing_.has_content_length ? framing_.content_length : 0;
    }

private:
    state reconsider() {
        if (header_end_ == std::string::npos) {
            if (data_.size() > eshkol_server_max_header_bytes()) {
                refuse(431, eshkol_http_framing_message(
                                eshkol_http_framing_verdict::header_section_too_large));
            }
            return state_;
        }
        if (!framing_examined_) {
            framing_examined_ = true;
            framing_ = eshkol_http_examine_framing(data_.substr(0, header_end_));
            if (framing_.verdict != eshkol_http_framing_verdict::ok) {
                refuse(eshkol_http_framing_status(framing_.verdict),
                       eshkol_http_framing_message(framing_.verdict));
                return state_;
            }
        }
        const size_t expected = header_end_ + declared_body_length();
        if (data_.size() > eshkol_server_max_header_bytes() +
                               eshkol_server_max_body_bytes()) {
            refuse(413, eshkol_http_framing_message(
                            eshkol_http_framing_verdict::body_too_large));
            return state_;
        }
        if (data_.size() >= expected) {
            data_.resize(expected);  // a pipelined follower is not this request
            state_ = state::complete;
        }
        return state_;
    }

    void refuse(int status, const char* message) {
        state_ = state::refused;
        refusal_status_ = status;
        refusal_message_ = message;
    }

    std::string data_;
    size_t header_end_ = std::string::npos;
    bool framing_examined_ = false;
    eshkol_http_framing framing_;
    state state_ = state::need_more;
    int refusal_status_ = 400;
    std::string refusal_message_;
};

// ─── JSON strings ───────────────────────────────────────────────────────────
//
// Every string that goes into a JSON document goes through here. A value the
// client chose — a session identifier, an error detail, a file name — is a
// string like any other, and the one way a string becomes JSON is
// eshkol_json_string(). Concatenating a value between two quote characters is
// not a second way of doing it; it is a way of letting the value decide where
// the string ends.

/** @brief Escapes a string's contents for a JSON string literal: the two
 *         characters JSON reserves, the five with short escapes, and every
 *         remaining control character as \\u00XX. */
inline std::string eshkol_json_escape(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    for (char c : text) {
        switch (c) {
            case '"':  result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            case '\b': result += "\\b"; break;
            case '\f': result += "\\f"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* hex = "0123456789abcdef";
                    const unsigned char byte = static_cast<unsigned char>(c);
                    result += "\\u00";
                    result += hex[(byte >> 4) & 0x0f];
                    result += hex[byte & 0x0f];
                } else {
                    result += c;
                }
                break;
        }
    }
    return result;
}

/** @brief Renders a string as a complete JSON string literal, quotes and all. */
inline std::string eshkol_json_string(const std::string& text) {
    return "\"" + eshkol_json_escape(text) + "\"";
}
