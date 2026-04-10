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
#include <limits>
#include <string>

// Maximum request body the server will accept (10 MB).
// Requests with a Content-Length above this get 413 Payload Too Large.
constexpr size_t kMaxRequestBodySize = 10 * 1024 * 1024;

// Maximum total request size (headers + body) before the read loop gives up.
constexpr size_t kMaxRequestTotalSize = kMaxRequestBodySize + 65536;

// Maximum simultaneous connections the server will handle.
constexpr int kMaxConcurrentConnections = 128;

// ─── Content-Length parser ───────────────────────────────────────────────────

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
