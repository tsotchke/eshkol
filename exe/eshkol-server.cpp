/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol WASM Compilation Server
 * A minimal HTTP server that compiles Eshkol Scheme code to WebAssembly.
 *
 * Security hardening (v1.1.13):
 *   - Content-Length validation + max body size cap (413 Payload Too Large)
 *   - Case-insensitive header lookup (RFC 7230)
 *   - Path traversal defence with percent-decode + component check
 *   - Bounded request accumulation (headers + body)
 *   - Connection limiter (atomic counter, kMaxConcurrentConnections)
 *   - Thread-safe session ID generation
 *   - Partial-send loop for large responses
 *   - Increased listen backlog (SOMAXCONN)
 *   - Loopback bind by default; public binds require a compile token
 *   - Optional exact-origin CORS (no wildcard)
 *   - JSON-only /compile requests to block simple cross-site form posts
 */

#ifdef ESHKOL_SELF_HOSTED_INCREMENTAL
#include <eshkol/frontend/incrementalize.h>
#endif
#ifdef ESHKOL_SELF_HOSTED_SCALAR_LOWERING
#include <eshkol/frontend/scalar_lowering.h>
#endif
#include "eshkol/eshkol.h"
#include "eshkol/core/http_syntax.h"
#include "eshkol/http_request_utils.h"
#include <eshkol/llvm_backend.h>
#include <eshkol/logger.h>

#include <iostream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <new>
#include <stdexcept>
#include <fstream>

// Networking headers
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <winsock2.h>
#include <ws2tcpip.h>
#include <io.h>
#include <signal.h>
#pragma comment(lib, "Ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#endif

#ifdef _WIN32
using socket_handle_t = SOCKET;
using socket_length_t = int;
constexpr socket_handle_t k_invalid_socket = INVALID_SOCKET;
#else
using socket_handle_t = int;
using socket_length_t = socklen_t;
constexpr socket_handle_t k_invalid_socket = -1;
#endif

namespace {

bool init_socket_runtime() {
#ifdef _WIN32
    WSADATA wsa_data{};
    return WSAStartup(MAKEWORD(2, 2), &wsa_data) == 0;
#else
    return true;
#endif
}

void shutdown_socket_runtime() {
#ifdef _WIN32
    WSACleanup();
#endif
}

void close_socket(socket_handle_t socket) {
#ifdef _WIN32
    if (socket != INVALID_SOCKET) {
        closesocket(socket);
    }
#else
    if (socket >= 0) {
        close(socket);
    }
#endif
}

bool set_socket_receive_timeout(socket_handle_t socket, int seconds) {
#ifdef _WIN32
    const DWORD timeout_ms = static_cast<DWORD>(seconds * 1000);
    return setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO,
                      reinterpret_cast<const char*>(&timeout_ms),
                      sizeof(timeout_ms)) == 0;
#else
    struct timeval tv;
    tv.tv_sec = seconds;
    tv.tv_usec = 0;
    return setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) == 0;
#endif
}

bool path_exists(const char* path) {
#ifdef _WIN32
    return _access(path, 0) == 0;
#else
    return access(path, F_OK) == 0;
#endif
}

// Send the full response, handling partial writes.
bool send_all(socket_handle_t sock, const std::string& data) {
    size_t total_sent = 0;
    while (total_sent < data.size()) {
        int chunk = send(sock, data.c_str() + total_sent,
                         static_cast<int>(data.size() - total_sent), 0);
        if (chunk <= 0) return false;
        total_sent += static_cast<size_t>(chunk);
    }
    return true;
}

bool parse_bind_address(const std::string& host, in_addr& addr) {
    if (host == "localhost") {
        addr.s_addr = htonl(INADDR_LOOPBACK);
        return true;
    }

#ifdef _WIN32
    return InetPtonA(AF_INET, host.c_str(), &addr) == 1;
#else
    return inet_pton(AF_INET, host.c_str(), &addr) == 1;
#endif
}

bool is_loopback_address(const in_addr& addr) {
    const uint32_t host_order = ntohl(addr.s_addr);
    return (host_order & 0xff000000u) == 0x7f000000u;
}

bool is_safe_header_value(const std::string& value) {
    if (value.empty()) return false;
    for (unsigned char ch : value) {
        if (ch < 0x20 || ch == 0x7f) return false;
    }
    return true;
}

bool constant_time_equals(const std::string& a, const std::string& b) {
    const size_t max_len = std::max(a.size(), b.size());
    unsigned char diff = static_cast<unsigned char>(a.size() ^ b.size());
    for (size_t i = 0; i < max_len; ++i) {
        const unsigned char ac = i < a.size() ? static_cast<unsigned char>(a[i]) : 0;
        const unsigned char bc = i < b.size() ? static_cast<unsigned char>(b[i]) : 0;
        diff |= static_cast<unsigned char>(ac ^ bc);
    }
    return diff == 0;
}

} // namespace

static std::atomic<bool> g_running{true};
static socket_handle_t g_server_socket = k_invalid_socket;

// Connection limiter — prevents unbounded thread creation under load.
static std::atomic<int> g_active_connections{0};

static std::string g_bind_host = "127.0.0.1";
static std::string g_compile_token;
static std::string g_cors_origin;

// The platform cryptographic source (lib/core/crypto_primitives.c):
// SecRandomCopyBytes, BCryptGenRandom, RAND_bytes or /dev/urandom.
extern "C" int eshkol_random_bytes(char* buf, size_t len);

// A session identifier names a compilation a client will come back for, so
// it is drawn from the operating system's cryptographic source and from
// nowhere else — 16 bytes, rendered as 32 hex characters. The seeded
// generator behind Eshkol's own `(random)` is a reproducible stream for
// simulation and replay and has no part in this. When the source cannot be
// read the function returns an empty string and the caller declines the
// request rather than issuing a guessable identifier.
std::string generate_session_id() {
    static const char* hex = "0123456789abcdef";
    unsigned char bytes[16];
    if (eshkol_random_bytes(reinterpret_cast<char*>(bytes), sizeof(bytes)) != 0) {
        return std::string();
    }
    std::string id;
    id.reserve(sizeof(bytes) * 2);
    for (unsigned char byte : bytes) {
        id += hex[(byte >> 4) & 0x0f];
        id += hex[byte & 0x0f];
    }
    return id;
}

// Simple HTTP response builder
std::string build_response(int status, const std::string& content_type,
                          const std::string& body) {
    std::ostringstream response;

    std::string status_text;
    switch (status) {
        case 200: status_text = "OK"; break;
        case 400: status_text = "Bad Request"; break;
        case 401: status_text = "Unauthorized"; break;
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 408: status_text = "Request Timeout"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 415: status_text = "Unsupported Media Type"; break;
        case 431: status_text = "Request Header Fields Too Large"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
        default: status_text = "Unknown"; break;
    }

    response << "HTTP/1.1 " << status << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.size() << "\r\n";
    if (!g_cors_origin.empty()) {
        response << "Access-Control-Allow-Origin: " << g_cors_origin << "\r\n";
        response << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        response << "Access-Control-Allow-Headers: Content-Type, Authorization, X-Eshkol-Compile-Token\r\n";
        response << "Vary: Origin\r\n";
    }
    response << "Connection: close\r\n";
    response << "\r\n";

    return response.str() + body;
}

// JSON strings are built by the one serializer in
// <eshkol/http_request_utils.h>, which both this server and its tests use.
std::string json_escape(const std::string& s) { return eshkol_json_escape(s); }

// Build JSON error response
std::string json_error(const std::string& message) {
    return "{\"success\":false,\"error\":" + eshkol_json_string(message) + "}";
}

// Build JSON success response with WASM (base64 encoded)
std::string json_success_wasm(const std::string& session_id, const uint8_t* wasm, size_t size) {
    // Base64 encode the WASM
    static const char* b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;

    for (size_t i = 0; i < size; i += 3) {
        uint32_t n = (uint32_t)wasm[i] << 16;
        if (i + 1 < size) n |= (uint32_t)wasm[i + 1] << 8;
        if (i + 2 < size) n |= wasm[i + 2];

        encoded += b64[(n >> 18) & 0x3F];
        encoded += b64[(n >> 12) & 0x3F];
        encoded += (i + 1 < size) ? b64[(n >> 6) & 0x3F] : '=';
        encoded += (i + 2 < size) ? b64[n & 0x3F] : '=';
    }

    // Every string that goes into a JSON document goes through the one
    // serializer that knows what JSON strings may contain. session_id can be
    // a value the client chose, so it is escaped on the way out exactly as
    // every other string value is.
    std::ostringstream json;
    json << "{\"success\":true,\"session_id\":" << eshkol_json_string(session_id) << ",";
    json << "\"wasm\":\"" << encoded << "\",";
    json << "\"size\":" << size << "}";
    return json.str();
}

// Compile code to WASM
std::string compile_to_wasm(const std::string& code, const std::string& session_id) {
    // Parse the code using a stringstream
    std::istringstream code_stream(code);

    // Fresh parse session — reset cumulative line counter so the first
    // form starts at line 1 in `code`.
    eshkol_reset_parse_line_counter();

    std::vector<eshkol_ast_t> asts;
    eshkol_ast_t ast;

    while ((ast = eshkol_parse_next_ast_from_stream(code_stream)).type != ESHKOL_INVALID) {
        asts.push_back(ast);
    }

    if (asts.empty()) {
        return json_error("Failed to parse code or empty input");
    }

    // Generate LLVM IR (library mode - no main function)
    LLVMModuleRef module = eshkol_generate_llvm_ir_library(asts.data(), asts.size(), "wasm_module");
    if (!module) {
        return json_error("Failed to generate LLVM IR");
    }

    // Compile to WASM
    uint8_t* wasm_buffer = nullptr;
    size_t wasm_size = 0;

    int result = eshkol_compile_llvm_ir_to_wasm(module, &wasm_buffer, &wasm_size);

    eshkol_dispose_llvm_module(module);

    if (result != 0 || !wasm_buffer) {
        return json_error("Failed to compile to WebAssembly");
    }

    std::string response = json_success_wasm(session_id, wasm_buffer, wasm_size);
    free(wasm_buffer);

    return response;
}

// Parse HTTP request
struct HttpRequest {
    std::string method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    bool malformed = false;
    int error_status = 400;
    std::string error_message;
};

// Interpret a complete request. `header_end` and `body_length` come from the
// one framing answer the accumulator produced while reading, so the body this
// returns is exactly the body the read loop waited for — there is no second
// reading of Content-Length here to disagree with the first.
HttpRequest parse_request(const std::string& raw, size_t header_end,
                          size_t body_length) {
    HttpRequest req;
    std::istringstream stream(raw.substr(0, header_end));
    std::string line;

    // Parse request line
    std::getline(stream, line);
    if (line.size() > 0 && line.back() == '\r') line.pop_back();

    size_t space1 = line.find(' ');
    size_t space2 = line.find(' ', space1 + 1);

    if (space1 != std::string::npos && space2 != std::string::npos) {
        req.method = line.substr(0, space1);
        req.path = line.substr(space1 + 1, space2 - space1 - 1);
    }

    // Parse headers — normalise names to lowercase (RFC 7230 sec 3.2)
    while (std::getline(stream, line) && line != "\r" && !line.empty()) {
        if (line.back() == '\r') line.pop_back();
        size_t colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = eshkol_normalize_header_name(line.substr(0, colon));
            std::string value = line.substr(colon + 1);
            // Trim leading whitespace
            while (!value.empty() && value[0] == ' ') value.erase(0, 1);
            req.headers[key] = value;
        }
    }

    // The body is the declared number of bytes that follow the header
    // section, taken by length from the accumulated request. NUL bytes in it
    // are content and are preserved.
    const size_t available = raw.size() > header_end ? raw.size() - header_end : 0;
    req.body.assign(raw, header_end, std::min(body_length, available));

    // The request line must itself be a request line: a method token and one
    // of the request-target forms. A request whose line is neither is refused
    // before any route sees it.
    if (req.method.empty() || !eshkol_http_valid_method(req.method.c_str())) {
        req.malformed = true;
        req.error_status = 400;
        req.error_message = "Request method is not a token";
        return req;
    }
    if (req.path.empty() || !eshkol_http_valid_request_target(req.path.c_str())) {
        req.malformed = true;
        req.error_status = 400;
        req.error_message = "Request target is not a request-target form";
        return req;
    }

    return req;
}

bool is_json_content_type(const HttpRequest& req) {
    auto it = req.headers.find("content-type");
    if (it == req.headers.end()) return false;
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    const size_t parameter_start = value.find(';');
    if (parameter_start != std::string::npos) {
        value = value.substr(0, parameter_start);
    }
    while (!value.empty() && (value.front() == ' ' || value.front() == '\t')) {
        value.erase(0, 1);
    }
    while (!value.empty() && (value.back() == ' ' || value.back() == '\t')) {
        value.pop_back();
    }
    return value == "application/json";
}

std::string compile_request_token(const HttpRequest& req) {
    auto token_it = req.headers.find("x-eshkol-compile-token");
    if (token_it != req.headers.end()) return token_it->second;

    auto auth_it = req.headers.find("authorization");
    if (auth_it != req.headers.end()) {
        const std::string prefix = "Bearer ";
        if (auth_it->second.compare(0, prefix.size(), prefix) == 0) {
            return auth_it->second.substr(prefix.size());
        }
    }
    return "";
}

bool compile_request_authorized(const HttpRequest& req) {
    if (g_compile_token.empty()) return true;
    return constant_time_equals(compile_request_token(req), g_compile_token);
}

// Extract JSON string value (simple parser)
std::string get_json_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos += search.size();
    size_t end = json.find("\"", pos);
    if (end == std::string::npos) return "";

    // Unescape
    std::string value;
    for (size_t i = pos; i < end; ++i) {
        if (json[i] == '\\' && i + 1 < end) {
            ++i;
            switch (json[i]) {
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                case '"': value += '"'; break;
                case '\\': value += '\\'; break;
                default: value += json[i]; break;
            }
        } else {
            value += json[i];
        }
    }
    return value;
}

// Static file directory (relative to executable)
static std::string g_web_dir = "";

// Get MIME type from file extension
std::string get_mime_type(const std::string& path) {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) return "application/octet-stream";

    std::string ext = path.substr(dot);
    if (ext == ".html" || ext == ".htm") return "text/html";
    if (ext == ".css") return "text/css";
    if (ext == ".js") return "application/javascript";
    if (ext == ".json") return "application/json";
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".svg") return "image/svg+xml";
    if (ext == ".wasm") return "application/wasm";
    if (ext == ".ico") return "image/x-icon";
    return "text/plain";
}

// Read file contents
std::string read_file(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) return "";

    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// Serve static file
std::string serve_static(const std::string& url_path) {
    // Security: reject path traversal (percent-encoded, backslash, null bytes)
    if (!eshkol_is_safe_url_path(url_path)) {
        return build_response(403, "text/plain", "Forbidden");
    }

    std::string file_path = g_web_dir;
    if (url_path == "/") {
        file_path += "/index.html";
    } else {
        file_path += url_path;
    }

    std::string content = read_file(file_path);
    if (content.empty()) {
        return build_response(404, "text/plain", "Not found");
    }

    return build_response(200, get_mime_type(file_path), content);
}

// RAII guard for the connection counter.
struct ConnectionGuard {
    ConnectionGuard() { g_active_connections.fetch_add(1, std::memory_order_relaxed); }
    ~ConnectionGuard() { g_active_connections.fetch_sub(1, std::memory_order_relaxed); }
    ConnectionGuard(const ConnectionGuard&) = delete;
    ConnectionGuard& operator=(const ConnectionGuard&) = delete;
};

// Handle client connection
void handle_client(socket_handle_t client_socket) {
    ConnectionGuard guard;

    // Read the request. The accumulator owns both questions — is it complete,
    // and is it one this server will answer — from one reading of the header
    // section, and it is fed exact byte counts, so a body carrying NUL bytes
    // arrives whole.
    char buffer[65536];
    eshkol_http_request_accumulator accumulator;

    // Two clocks bound the read: each recv() blocks for at most
    // kReceiveTimeoutSeconds, and the whole request must arrive within the
    // per-request deadline measured from the moment it was accepted. A client
    // that sends a byte at a time therefore holds this one connection for a
    // bounded time and no other connection at all.
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(eshkol_server_request_deadline_seconds());
    set_socket_receive_timeout(client_socket, kReceiveTimeoutSeconds);

    bool timed_out = false;
    while (accumulator.current() ==
           eshkol_http_request_accumulator::state::need_more) {
        if (std::chrono::steady_clock::now() >= deadline) {
            timed_out = true;
            break;
        }
        const int bytes_read = recv(client_socket, buffer, sizeof(buffer), 0);
        if (bytes_read > 0) {
            accumulator.append(buffer, static_cast<size_t>(bytes_read));
            continue;
        }
        if (bytes_read == 0) {
            accumulator.close();  // the peer stopped sending
            break;
        }
        // A recv() that expired is not an error: the deadline above decides
        // whether this connection still has time.
#ifndef _WIN32
        if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
#else
        if (WSAGetLastError() == WSAETIMEDOUT) continue;
#endif
        close_socket(client_socket);
        return;
    }

    if (timed_out) {
        send_all(client_socket,
                 build_response(408, "application/json",
                                json_error("Request did not arrive within the "
                                           "per-request deadline")));
        close_socket(client_socket);
        return;
    }
    if (accumulator.current() ==
        eshkol_http_request_accumulator::state::refused) {
        send_all(client_socket,
                 build_response(accumulator.refusal_status(), "application/json",
                                json_error(accumulator.refusal_message())));
        close_socket(client_socket);
        return;
    }

    const std::string& request_data = accumulator.data();
    if (request_data.empty()) {
        close_socket(client_socket);
        return;
    }

    HttpRequest req = parse_request(request_data, accumulator.header_end(),
                                    accumulator.declared_body_length());
    std::string response;

    if (req.malformed) {
        response = build_response(req.error_status, "application/json",
                                  json_error(req.error_message));
        send_all(client_socket, response);
        close_socket(client_socket);
        return;
    }

    // Handle CORS preflight
    if (req.method == "OPTIONS") {
        response = build_response(200, "text/plain", "");
    }
    // Handle POST /compile
    else if (req.method == "POST" && req.path == "/compile") {
        std::string code = get_json_string(req.body, "code");
        std::string session_id = get_json_string(req.body, "session_id");

        if (!is_json_content_type(req)) {
            response = build_response(415, "application/json",
                json_error("POST /compile requires Content-Type: application/json"));
        } else if (!compile_request_authorized(req)) {
            response = build_response(401, "application/json", json_error("Unauthorized"));
        } else if (code.empty()) {
            response = build_response(400, "application/json", json_error("Missing 'code' field"));
        } else {
            if (session_id.empty()) {
                session_id = generate_session_id();
            }
            if (session_id.empty()) {
                response = build_response(503, "application/json",
                    json_error("The cryptographic source for session "
                               "identifiers is unavailable"));
            } else {
                // One request's compilation is one request's business. A
                // source that the compiler cannot find memory for, or that
                // fails anywhere in the pipeline, is answered on this
                // connection; the process stays up and every other
                // connection is unaffected.
                try {
                    std::string result = compile_to_wasm(code, session_id);
                    response = build_response(200, "application/json", result);
                } catch (const std::bad_alloc&) {
                    response = build_response(413, "application/json",
                        json_error("The submitted source needs more memory "
                                   "than one request is given"));
                } catch (const std::exception& error) {
                    response = build_response(400, "application/json",
                        json_error(std::string("Compilation declined: ") +
                                   error.what()));
                }
            }
        }
    }
    // Handle GET /health
    else if (req.method == "GET" && req.path == "/health") {
        response = build_response(200, "application/json", "{\"status\":\"ok\"}");
    }
    // Serve static files from web directory
    else if (req.method == "GET" && !g_web_dir.empty()) {
        response = serve_static(req.path);
    }
    // 404 for everything else
    else {
        response = build_response(404, "application/json", json_error("Not found"));
    }

    // Send response (handles partial writes)
    send_all(client_socket, response);
    close_socket(client_socket);
}

// Signal handler for graceful shutdown
void signal_handler(int sig) {
    (void)sig;
    g_running = false;
    if (g_server_socket != k_invalid_socket) {
        close_socket(g_server_socket);
    }
}

void print_usage() {
    std::cout << "Usage: eshkol-server [options]\n"
              << "Options:\n"
              << "  --port <port>      Port to listen on (default: 8080)\n"
              << "  --host <addr>      IPv4 address to bind (default: 127.0.0.1)\n"
              << "  --public           Bind 0.0.0.0 (requires a compile token)\n"
              << "  --compile-token <token>\n"
              << "                     Require token for POST /compile\n"
              << "                     (or set ESHKOL_SERVER_TOKEN)\n"
              << "  --cors-origin <origin>\n"
              << "                     Allow CORS for one exact origin (no wildcard)\n"
              << "  --web-dir <path>   Directory to serve static files from\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char** argv) {
#ifdef ESHKOL_SELF_HOSTED_INCREMENTAL
    eshkol::frontend::enable_self_hosted_incremental();
#endif
#ifdef ESHKOL_SELF_HOSTED_SCALAR_LOWERING
    eshkol::frontend::enable_self_hosted_scalar_lowering();
#endif
    int port = 8080;
    const char* env_token = std::getenv("ESHKOL_SERVER_TOKEN");
    if (env_token && env_token[0] != '\0') {
        g_compile_token = env_token;
    }

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--host") == 0 && i + 1 < argc) {
            g_bind_host = argv[++i];
        } else if (strcmp(argv[i], "--public") == 0) {
            g_bind_host = "0.0.0.0";
        } else if (strcmp(argv[i], "--compile-token") == 0 && i + 1 < argc) {
            g_compile_token = argv[++i];
        } else if (strcmp(argv[i], "--cors-origin") == 0 && i + 1 < argc) {
            g_cors_origin = argv[++i];
        } else if (strcmp(argv[i], "--web-dir") == 0 && i + 1 < argc) {
            g_web_dir = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage();
            return 0;
        }
    }

    // Try to auto-detect web directory if not specified
    if (g_web_dir.empty()) {
        // Check for web/ relative to current directory
        if (path_exists("web/index.html")) {
            g_web_dir = "web";
        }
        // Check for ../web/ (if running from build/)
        else if (path_exists("../web/index.html")) {
            g_web_dir = "../web";
        }
    }

    if (!init_socket_runtime()) {
        std::cerr << "Failed to initialize socket runtime\n";
        return 1;
    }

    in_addr bind_addr{};
    if (!parse_bind_address(g_bind_host, bind_addr)) {
        std::cerr << "Invalid --host value '" << g_bind_host
                  << "'; expected an IPv4 address or localhost\n";
        shutdown_socket_runtime();
        return 1;
    }
    if (!is_loopback_address(bind_addr) && g_compile_token.empty()) {
        std::cerr << "Refusing to expose /compile on " << g_bind_host
                  << " without --compile-token or ESHKOL_SERVER_TOKEN\n";
        shutdown_socket_runtime();
        return 1;
    }
    if (!g_cors_origin.empty() &&
        (g_cors_origin == "*" || !is_safe_header_value(g_cors_origin))) {
        std::cerr << "Invalid --cors-origin; use one exact safe origin, not '*'\n";
        shutdown_socket_runtime();
        return 1;
    }

    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);  // Ignore broken pipe
#endif

    // Create socket
    g_server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (g_server_socket == k_invalid_socket) {
        std::cerr << "Failed to create socket\n";
        shutdown_socket_runtime();
        return 1;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(g_server_socket, SOL_SOCKET, SO_REUSEADDR,
#ifdef _WIN32
               reinterpret_cast<const char*>(&opt),
#else
               &opt,
#endif
               sizeof(opt));

    // Bind
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr = bind_addr;
    addr.sin_port = htons(port);

    if (bind(g_server_socket, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "Failed to bind to port " << port << "\n";
        close_socket(g_server_socket);
        shutdown_socket_runtime();
        return 1;
    }

    // Listen with a reasonable backlog
    if (listen(g_server_socket, SOMAXCONN) < 0) {
        std::cerr << "Failed to listen\n";
        close_socket(g_server_socket);
        shutdown_socket_runtime();
        return 1;
    }

    std::cout << "Eshkol WASM Server running on http://" << g_bind_host << ":" << port << "\n";
    if (!g_web_dir.empty()) {
        std::cout << "Serving static files from: " << g_web_dir << "\n";
    }
    if (!g_compile_token.empty()) {
        std::cout << "POST /compile requires Authorization: Bearer <token>\n";
    }
    if (!g_cors_origin.empty()) {
        std::cout << "CORS allowed origin: " << g_cors_origin << "\n";
    }
    std::cout << "Press Ctrl+C to stop\n";

    // Accept connections
    while (g_running) {
        struct sockaddr_in client_addr;
        socket_length_t client_len = sizeof(client_addr);

        socket_handle_t client_socket = accept(g_server_socket, (struct sockaddr*)&client_addr, &client_len);
        if (client_socket == k_invalid_socket) {
            if (g_running) {
                std::cerr << "Accept failed\n";
            }
            continue;
        }

        // Connection limiter — reject with 503 if at capacity.
        if (g_active_connections.load(std::memory_order_relaxed) >= kMaxConcurrentConnections) {
            std::string response = build_response(503, "application/json",
                json_error("Server at capacity"));
            send(client_socket, response.c_str(), static_cast<int>(response.size()), 0);
            close_socket(client_socket);
            continue;
        }

        // Handle in separate thread
        std::thread(handle_client, client_socket).detach();
    }

    close_socket(g_server_socket);
    shutdown_socket_runtime();
    std::cout << "\nServer stopped\n";
    return 0;
}
