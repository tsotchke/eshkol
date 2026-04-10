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
 */

#include "eshkol/eshkol.h"
#include "eshkol/http_request_utils.h"
#include <eshkol/llvm_backend.h>
#include <eshkol/logger.h>

#include <iostream>
#include <sstream>
#include <string>
#include <cstring>
#include <cstdlib>
#include <unordered_map>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <random>
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

} // namespace

static std::atomic<bool> g_running{true};
static socket_handle_t g_server_socket = k_invalid_socket;

// Connection limiter — prevents unbounded thread creation under load.
static std::atomic<int> g_active_connections{0};

// Thread-safe session ID generator.
std::string generate_session_id() {
    static std::mutex rng_mutex;
    static std::mt19937 gen(std::random_device{}());
    static std::uniform_int_distribution<> dis(0, 15);
    static const char* hex = "0123456789abcdef";

    std::lock_guard<std::mutex> lock(rng_mutex);
    std::string id;
    for (int i = 0; i < 32; ++i) {
        id += hex[dis(gen)];
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
        case 403: status_text = "Forbidden"; break;
        case 404: status_text = "Not Found"; break;
        case 413: status_text = "Payload Too Large"; break;
        case 500: status_text = "Internal Server Error"; break;
        case 503: status_text = "Service Unavailable"; break;
        default: status_text = "Unknown"; break;
    }

    response << "HTTP/1.1 " << status << " " << status_text << "\r\n";
    response << "Content-Type: " << content_type << "\r\n";
    response << "Content-Length: " << body.size() << "\r\n";
    response << "Access-Control-Allow-Origin: *\r\n";
    response << "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
    response << "Access-Control-Allow-Headers: Content-Type\r\n";
    response << "Connection: close\r\n";
    response << "\r\n";

    return response.str() + body;
}

// JSON escape helper
std::string json_escape(const std::string& s) {
    std::string result;
    for (char c : s) {
        switch (c) {
            case '"': result += "\\\""; break;
            case '\\': result += "\\\\"; break;
            case '\n': result += "\\n"; break;
            case '\r': result += "\\r"; break;
            case '\t': result += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    // Escape control characters as \u00XX
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned char>(c));
                    result += buf;
                } else {
                    result += c;
                }
                break;
        }
    }
    return result;
}

// Build JSON error response
std::string json_error(const std::string& message) {
    return "{\"success\":false,\"error\":\"" + json_escape(message) + "\"}";
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

    std::ostringstream json;
    json << "{\"success\":true,\"session_id\":\"" << session_id << "\",";
    json << "\"wasm\":\"" << encoded << "\",";
    json << "\"size\":" << size << "}";
    return json.str();
}

// Compile code to WASM
std::string compile_to_wasm(const std::string& code, const std::string& session_id) {
    // Parse the code using a stringstream
    std::istringstream code_stream(code);

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

HttpRequest parse_request(const std::string& raw) {
    HttpRequest req;
    std::istringstream stream(raw);
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

    // Get body (rest of stream)
    std::ostringstream body_stream;
    body_stream << stream.rdbuf();
    req.body = body_stream.str();

    // Validate and enforce Content-Length (now case-insensitive)
    auto it = req.headers.find("content-length");
    if (it != req.headers.end()) {
        size_t len = 0;
        if (!eshkol_parse_content_length(it->second, len)) {
            req.malformed = true;
            req.error_status = 400;
            req.error_message = "Invalid Content-Length";
            return req;
        }
        if (len > kMaxRequestBodySize) {
            req.malformed = true;
            req.error_status = 413;
            req.error_message = "Payload too large";
            return req;
        }
        if (req.body.size() > len) {
            req.body = req.body.substr(0, len);
        }
    }

    return req;
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

    // Read request
    char buffer[65536];
    std::string request_data;
    int bytes_read;

    set_socket_receive_timeout(client_socket, 5);

    // Read until we have the full request (bounded by kMaxRequestTotalSize)
    while ((bytes_read = recv(client_socket, buffer, sizeof(buffer) - 1, 0)) > 0) {
        buffer[bytes_read] = '\0';
        request_data += buffer;

        // Bound total request size to prevent OOM from unbounded header
        // accumulation or oversized bodies.
        if (request_data.size() > kMaxRequestTotalSize) {
            std::string response = build_response(413, "application/json",
                json_error("Request too large"));
            send_all(client_socket, response);
            close_socket(client_socket);
            return;
        }

        // Check if we have the complete request
        size_t header_end = request_data.find("\r\n\r\n");
        if (header_end != std::string::npos) {
            // Case-insensitive Content-Length search in raw data.
            // We need to find it before full header parsing because the read
            // loop uses it to decide when the body is complete.
            // Include the trailing \r\n\r\n so the last header's \r\n is findable.
            std::string lower_data = request_data.substr(0, header_end + 4);
            std::transform(lower_data.begin(), lower_data.end(), lower_data.begin(),
                           [](unsigned char c) { return std::tolower(c); });

            size_t cl_pos = lower_data.find("content-length:");
            if (cl_pos != std::string::npos) {
                size_t cl_end = lower_data.find("\r\n", cl_pos);
                if (cl_end == std::string::npos) {
                    std::string response = build_response(400, "application/json",
                        json_error("Invalid Content-Length"));
                    send_all(client_socket, response);
                    close_socket(client_socket);
                    return;
                }

                // Extract from the ORIGINAL data (preserving case for the value)
                std::string cl_str = request_data.substr(cl_pos + 15, cl_end - cl_pos - 15);
                size_t content_length = 0;
                if (!eshkol_parse_content_length(cl_str, content_length)) {
                    std::string response = build_response(400, "application/json",
                        json_error("Invalid Content-Length"));
                    send_all(client_socket, response);
                    close_socket(client_socket);
                    return;
                }
                if (content_length > kMaxRequestBodySize) {
                    std::string response = build_response(413, "application/json",
                        json_error("Payload too large"));
                    send_all(client_socket, response);
                    close_socket(client_socket);
                    return;
                }
                size_t body_start = header_end + 4;
                if (request_data.size() >= body_start + content_length) {
                    break;
                }
            } else {
                break;
            }
        }
    }

    if (request_data.empty()) {
        close_socket(client_socket);
        return;
    }

    HttpRequest req = parse_request(request_data);
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

        if (session_id.empty()) {
            session_id = generate_session_id();
        }

        if (code.empty()) {
            response = build_response(400, "application/json", json_error("Missing 'code' field"));
        } else {
            std::string result = compile_to_wasm(code, session_id);
            response = build_response(200, "application/json", result);
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
              << "  --web-dir <path>   Directory to serve static files from\n"
              << "  --help             Show this help message\n";
}

int main(int argc, char** argv) {
    int port = 8080;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--port") == 0 && i + 1 < argc) {
            port = atoi(argv[++i]);
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
    addr.sin_addr.s_addr = INADDR_ANY;
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

    std::cout << "Eshkol WASM Server running on http://localhost:" << port << "\n";
    if (!g_web_dir.empty()) {
        std::cout << "Serving static files from: " << g_web_dir << "\n";
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
