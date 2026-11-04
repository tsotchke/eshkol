// Native regression fixture. The target driver links the runtime sources and
// invokes this test; this file intentionally does not prescribe build wiring.
#include <eshkol/eshkol_ffi.h>
#include <eshkol/core/introspection.h>
#include "../../../lib/core/arena_memory.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

extern "C" int64_t eshkol_string_byte_length(const char*);
static unsigned checks = 0;
static void checked(bool ok, const char* expression) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", expression); std::exit(1); }
}
#define CHECK(expression) checked((expression), #expression)

// Mirrors system_builtins.c's explicitly documented 16-byte tagged layout.
struct builtin_value {
    uint8_t type;
    uint8_t flags;
    uint16_t reserved;
    uint32_t padding;
    uint64_t data;
};
static_assert(sizeof(builtin_value) == 16, "builtin ABI layout changed");
extern "C" void eshkol_builtin_process_read_nonblocking(
    builtin_value*, const builtin_value*, const builtin_value*);
extern "C" void eshkol_builtin_socket_recv(
    builtin_value*, const builtin_value*, const builtin_value*);

static builtin_value integer(int64_t n) {
    builtin_value v{};
    v.type = ESHKOL_FFI_TYPE_INT64; // layout-compatible system builtin value
    std::memcpy(&v.data, &n, sizeof(n));
    return v;
}

static void check_string(const builtin_value& v, const char expected[3]) {
    CHECK(v.type == 8); // SYS_TYPE_HEAP_PTR
    // Heap subtype comes from the canonical object header; value flags may be zero.
    const char* s = reinterpret_cast<const char*>(v.data);
    const eshkol_object_header_t* header = ESHKOL_GET_HEADER(s);
    CHECK(header->subtype == HEAP_SUBTYPE_STRING);
    CHECK(header->size == 4); // three payload bytes plus terminator
    CHECK(eshkol_string_byte_length(s) == 3);
    CHECK(std::memcmp(s, expected, 3) == 0);
    CHECK(s[3] == '\0');
}

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "all";
    const bool all = std::strcmp(mode, "all") == 0;
    CHECK(all || std::strcmp(mode, "ffi") == 0 || std::strcmp(mode, "pipe") == 0 ||
          std::strcmp(mode, "socket") == 0 || std::strcmp(mode, "symbol") == 0);
    arena_t* arena = get_global_arena();
    CHECK(arena != nullptr);
    const char bytes[3] = {'A', '\0', 'Z'};
    builtin_value fd, limit = integer(16), out{};
    if (all || std::strcmp(mode, "ffi") == 0) {
    eshkol_ffi_context_t* ctx = nullptr; // eshkol_ffi_string ignores ctx.
    eshkol_ffi_value_t ffi = eshkol_ffi_string(ctx, "ffi-string");
    CHECK(ffi.type == ESHKOL_FFI_TYPE_HEAP_PTR && ffi.flags == ESHKOL_FFI_SUBTYPE_STRING);
    const char* ffi_s = reinterpret_cast<const char*>(ffi.data.ptr_val);
    CHECK(ESHKOL_GET_HEADER(ffi_s)->subtype == HEAP_SUBTYPE_STRING);
    CHECK(ESHKOL_GET_HEADER(ffi_s)->size == sizeof("ffi-string"));
    CHECK(eshkol_string_byte_length(ffi_s) == 10);
    CHECK(std::strcmp(ffi_s, "ffi-string") == 0);
    CHECK(eshkol_ffi_string(ctx, nullptr).type == ESHKOL_FFI_TYPE_NULL);

    }

    if (all || std::strcmp(mode, "symbol") == 0) {
        const char name[] = "symbol-string-producer";
        char* input = arena_allocate_string_with_header(arena, sizeof(name) - 1);
        CHECK(input != nullptr);
        std::memcpy(input, name, sizeof(name));
        eshkol_tagged_value_t text{}; text.type = ESHKOL_VALUE_HEAP_PTR;
        text.data.ptr_val = reinterpret_cast<uint64_t>(input);
        auto symbol = eshkol_string_to_symbol(text);
        auto converted = eshkol_symbol_to_string(symbol, nullptr);
        CHECK(converted.type == ESHKOL_VALUE_HEAP_PTR);
        auto* copy = reinterpret_cast<char*>(converted.data.ptr_val);
        CHECK(copy != nullptr);
        CHECK(ESHKOL_GET_HEADER(copy)->subtype == HEAP_SUBTYPE_STRING);
        CHECK(ESHKOL_GET_HEADER(copy)->size == sizeof(name));
        CHECK(std::strcmp(copy, name) == 0);
        CHECK(copy != reinterpret_cast<char*>(symbol.data.ptr_val));
        copy[0] = 'X';
        CHECK(std::strcmp(reinterpret_cast<const char*>(symbol.data.ptr_val), name) == 0);
        CHECK(eshkol_symbol_to_string(symbol, arena).type == ESHKOL_VALUE_HEAP_PTR);
    }

    if (all || std::strcmp(mode, "pipe") == 0) {
    int pipefd[2];
    CHECK(pipe(pipefd) == 0);
    CHECK(write(pipefd[1], bytes, sizeof(bytes)) == 3);
    CHECK((fcntl(pipefd[0], F_GETFL) & O_NONBLOCK) == 0);
    fd = integer(pipefd[0]);
    eshkol_builtin_process_read_nonblocking(&out, &fd, &limit);
    check_string(out, bytes);
    CHECK((fcntl(pipefd[0], F_GETFL) & O_NONBLOCK) == 0);
    close(pipefd[0]);
    close(pipefd[1]);

    }

    if (all || std::strcmp(mode, "socket") == 0) {
    int sockets[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sockets) == 0);
    CHECK(write(sockets[0], bytes, sizeof(bytes)) == 3);
    CHECK((fcntl(sockets[1], F_GETFL) & O_NONBLOCK) == 0);
    fd = integer(sockets[1]);
    eshkol_builtin_socket_recv(&out, &fd, &limit);
    check_string(out, bytes);
    CHECK((fcntl(sockets[1], F_GETFL) & O_NONBLOCK) == 0);
    close(sockets[0]);
    close(sockets[1]);
    }
    std::printf("string_producers_native: %u checks, 0 failures\n", checks);
    return 0;
}
