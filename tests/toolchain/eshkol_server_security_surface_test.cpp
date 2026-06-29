#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

bool expect_contains(const std::string& haystack, const std::string& needle,
                     const std::string& label) {
    if (haystack.find(needle) == std::string::npos) {
        std::cerr << "missing: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

bool expect_not_contains(const std::string& haystack, const std::string& needle,
                         const std::string& label) {
    if (haystack.find(needle) != std::string::npos) {
        std::cerr << "unexpected: " << label << "\nneedle: " << needle
                  << std::endl;
        return false;
    }
    return true;
}

int fail(const std::string& message) {
    std::cerr << "FAIL: " << message << std::endl;
    return 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        return fail("usage: eshkol_server_security_surface_test <source-root>");
    }

    const fs::path source_root = argv[1];
    const fs::path server_path = source_root / "exe" / "eshkol-server.cpp";
    const fs::path repl_path = source_root / "web" / "eshkol-repl.js";
    if (!fs::exists(server_path)) {
        return fail("eshkol-server.cpp not found under source root");
    }
    if (!fs::exists(repl_path)) {
        return fail("eshkol-repl.js not found under source root");
    }

    const std::string server = read_file(server_path);
    const std::string repl = read_file(repl_path);
    bool ok = true;

    ok = ok &&
         expect_contains(server, "static std::string g_bind_host = \"127.0.0.1\";",
                         "server binds to loopback by default") &&
         expect_contains(server, "addr.sin_addr = bind_addr;",
                         "server binds the parsed host address") &&
         expect_contains(server, "!is_loopback_address(bind_addr) && g_compile_token.empty()",
                         "public binds require a compile token") &&
         expect_contains(server, "Refusing to expose /compile",
                         "server reports public bind refusal") &&
         expect_contains(server, "ESHKOL_SERVER_TOKEN",
                         "server supports token configuration through the environment") &&
         expect_contains(server, "compile_request_authorized(req)",
                         "compile route enforces configured token") &&
         expect_contains(server, "is_json_content_type(req)",
                         "compile route requires JSON content type") &&
         expect_contains(server, "g_cors_origin == \"*\"",
                         "server rejects wildcard CORS configuration") &&
         expect_contains(server, "Access-Control-Allow-Origin: \" << g_cors_origin",
                         "server emits only the configured CORS origin") &&
         expect_contains(server, "X-Eshkol-Compile-Token",
                         "server accepts explicit compile-token header") &&
         expect_contains(server, "Authorization: Bearer <token>",
                         "server documents bearer token usage");

    ok = ok &&
         expect_not_contains(server, "Access-Control-Allow-Origin: *",
                             "server should not emit wildcard CORS") &&
         expect_not_contains(server, "addr.sin_addr.s_addr = INADDR_ANY",
                             "server should not bind all interfaces by default");

    ok = ok &&
         expect_contains(repl, "constructor(serverUrl = 'http://localhost:8080', compileToken = null)",
                         "browser client accepts optional compile token") &&
         expect_contains(repl, "setCompileToken(token)",
                         "browser client can update compile token") &&
         expect_contains(repl, "headers.Authorization = `Bearer ${this.compileToken}`;",
                         "browser client sends bearer token for compile requests");

    if (!ok) {
        return 1;
    }

    std::cout << "PASS" << std::endl;
    return 0;
}
