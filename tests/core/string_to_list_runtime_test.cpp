#include "../../lib/core/arena_memory.h"

#include <cstdint>
#include <cstdio>
#include <cstring>

extern "C" void* eshkol_runtime_string_to_list(void* arena, const char* s);

namespace {

bool check(arena_t* arena, const unsigned char* input, size_t length,
           const int64_t* expected, size_t expected_count, const char* name) {
    char* string = arena_allocate_string_with_header(arena, length);
    if (!string) {
        std::printf("FAIL: %s: string allocation\n", name);
        return false;
    }
    std::memcpy(string, input, length);
    string[length] = '\0';

    auto* cell = static_cast<arena_tagged_cons_cell_t*>(
        eshkol_runtime_string_to_list(arena, string));
    for (size_t i = 0; i < expected_count; ++i) {
        if (!cell || cell->car.type != ESHKOL_VALUE_CHAR ||
            cell->car.data.int_val != expected[i]) {
            std::printf("FAIL: %s: codepoint %zu\n", name, i);
            return false;
        }
        if (cell->cdr.type != ESHKOL_VALUE_HEAP_PTR &&
            cell->cdr.type != ESHKOL_VALUE_NULL) {
            std::printf("FAIL: %s: improper list tail at %zu\n", name, i);
            return false;
        }
        if (cell->cdr.type == ESHKOL_VALUE_HEAP_PTR) {
            cell = reinterpret_cast<arena_tagged_cons_cell_t*>(cell->cdr.data.ptr_val);
        } else {
            cell = nullptr;
        }
    }
    if (cell) {
        std::printf("FAIL: %s: list has extra elements\n", name);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    arena_t* arena = arena_create(4096);
    if (!arena) return 1;
    int failures = 0;
    const unsigned char empty[] = {0};
    failures += !check(arena, empty, 0, nullptr, 0, "empty");
    const unsigned char ascii[] = {'A', 'z'};
    const int64_t ascii_expected[] = {'A', 'z'};
    failures += !check(arena, ascii, sizeof(ascii), ascii_expected, 2, "ASCII");
    const unsigned char embedded_nul[] = {'a', 0, 'b'};
    const int64_t nul_expected[] = {'a', 0, 'b'};
    failures += !check(arena, embedded_nul, sizeof(embedded_nul), nul_expected, 3,
                       "embedded NUL");
    const unsigned char multibyte[] = {0xC2, 0xA2, 0xE2, 0x82, 0xAC,
                                       0xF0, 0x9F, 0x98, 0x80};
    const int64_t multibyte_expected[] = {0xA2, 0x20AC, 0x1F600};
    failures += !check(arena, multibyte, sizeof(multibyte), multibyte_expected, 3,
                       "valid 2/3/4-byte sequences");
    const unsigned char max_scalar[] = {0xF4, 0x8F, 0xBF, 0xBF};
    const int64_t max_expected[] = {0x10FFFF};
    failures += !check(arena, max_scalar, sizeof(max_scalar), max_expected, 1,
                       "maximum scalar");

    const unsigned char malformed[] = {
        0x80,                         // stray continuation
        0xC0, 0xAF,                   // overlong 2-byte
        0xE0, 0x80, 0x80,             // overlong 3-byte
        0xED, 0xA0, 0x80,             // surrogate
        0xF4, 0x90, 0x80, 0x80,       // above U+10FFFF
        0xE2, 'X',                    // invalid continuation, then ASCII
        0xF0, 0x9F                    // truncated sequence
    };
    const int64_t malformed_expected[] = {
        0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD, 0xFFFD,
        0xFFFD, 'X', 0xFFFD
    };
    failures += !check(arena, malformed, sizeof(malformed), malformed_expected, 9,
                       "malformed and truncated sequences");
    const unsigned char truncated_two[] = {0xC2};
    const unsigned char truncated_three[] = {0xE2, 0x82};
    const int64_t replacement[] = {0xFFFD};
    failures += !check(arena, truncated_two, sizeof(truncated_two), replacement, 1,
                       "truncated two-byte sequence");
    failures += !check(arena, truncated_three, sizeof(truncated_three), replacement, 1,
                       "truncated three-byte sequence");
    arena_destroy(arena);
    if (!failures) std::printf("PASS: runtime string->list Unicode decoding\n");
    return failures ? 1 : 0;
}
