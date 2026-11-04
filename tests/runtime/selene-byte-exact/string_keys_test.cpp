// Full-byte string keys, independent allocations, rehash and legacy C strings.
#include "../../../lib/core/arena_memory.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
unsigned checks = 0, failures = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { ++failures; std::cerr << "FAIL: " << label << '\n'; }
}

eshkol_tagged_value_t integer(int64_t n) {
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_INT64;
    v.data.int_val = n;
    return v;
}

eshkol_tagged_value_t legacy(const char* s) {
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_STRING_PTR;
    v.data.ptr_val = reinterpret_cast<uint64_t>(s);
    return v;
}

eshkol_tagged_value_t string(arena_t* arena, const std::string& bytes) {
    char* p = arena_allocate_string_with_header(arena, bytes.size());
    if (!p) { std::cerr << "FAIL: allocation\n"; std::exit(2); }
    std::memcpy(p, bytes.data(), bytes.size());
    p[bytes.size()] = '\0';
    eshkol_tagged_value_t v{};
    v.type = ESHKOL_VALUE_HEAP_PTR;
    v.data.ptr_val = reinterpret_cast<uint64_t>(p);
    return v;
}

void key_pair(arena_t* arena, const std::string& a, const std::string& b) {
    const auto first = string(arena, a), copy = string(arena, a);
    const auto second = string(arena, b);
    check(eshkol_deep_equal(&first, &copy), "equal bytes across allocations");
    check(hash_tagged_value(&first) == hash_tagged_value(&copy), "equal keys hash equally");
    check(!eshkol_deep_equal(&first, &second), "distinct full-byte strings are unequal");
    auto* table = arena_hash_table_create_with_header(arena);
    if (!table) std::exit(2);
    const auto one = integer(1), two = integer(2), three = integer(3);
    eshkol_tagged_value_t result{};
    check(hash_table_set(arena, table, &first, &one), "insert first key");
    check(hash_table_set(arena, table, &second, &two), "insert second key");
    check(hash_table_count(table) == 2, "NUL suffixes retain distinct entries");
    check(hash_table_get(table, &copy, &result) && result.data.int_val == 1,
          "lookup uses complete bytes, independent allocation");
    check(hash_table_get(table, &second, &result) && result.data.int_val == 2,
          "second key has its own value");
    check(hash_table_set(arena, table, &copy, &three), "update equivalent key");
    check(hash_table_count(table) == 2, "equivalent-key update retains size");
    check(hash_table_get(table, &first, &result) && result.data.int_val == 3,
          "equivalent-key update affects only first key");
    check(hash_table_get(table, &second, &result) && result.data.int_val == 2,
          "update cannot overwrite distinct suffix");
    check(hash_table_remove(table, &copy), "remove equivalent key");
    check(!hash_table_has_key(table, &first), "removed key is absent");
    check(hash_table_has_key(table, &second), "distinct key survives removal");
}
}  // namespace

int main() {
    arena_t* arena = arena_create(65536);
    if (!arena) return 2;
    key_pair(arena, std::string("key\0a", 5), std::string("key\0z", 5));
    key_pair(arena, std::string("\0a", 2), std::string("\0z", 2));
    key_pair(arena, std::string("key\0", 4), "key");
    key_pair(arena, std::string("a\0\0b", 4), std::string("a\0\0c", 4));
    key_pair(arena, std::string("\0", 1), "");
    key_pair(arena, std::string("\xc3\xa9\0\xe9\x9b\xa8", 6),
                    std::string("\xc3\xa9\0\xe9\x9b\xaa", 6));

    auto plain = legacy("plain");
    auto same = string(arena, "plain");
    auto suffix = string(arena, std::string("plain\0tail", 10));
    check(eshkol_deep_equal(&plain, &same), "raw legacy/header equality remains supported");
    check(!eshkol_deep_equal(&plain, &suffix), "legacy C prefix differs from longer header string");
    auto* table = arena_hash_table_create_with_header(arena);
    const auto one = integer(1), two = integer(2);
    auto raw_other = legacy("other");
    check(hash_table_set(arena, table, &plain, &one), "raw legacy key remains hashable");
    check(hash_table_set(arena, table, &raw_other, &two), "second raw legacy key remains hashable");
    check(hash_table_count(table) == 2, "legacy key count");

    auto* growing = arena_hash_table_create_with_header(arena);
    std::vector<std::string> keys;
    for (int i = 0; i < 260; ++i) {
        keys.push_back(std::string("shared\0", 7) + std::to_string(i));
        auto key = string(arena, keys.back());
        auto value = integer(i);
        check(hash_table_set(arena, growing, &key, &value), "insert NUL suffix across resize");
    }
    check(hash_table_count(growing) == keys.size(), "all NUL suffixes survive rehash");
    for (size_t i = 0; i < keys.size(); ++i) {
        auto key = string(arena, keys[i]);
        eshkol_tagged_value_t value{};
        check(hash_table_get(growing, &key, &value) && value.data.int_val == static_cast<int64_t>(i),
              "rehash lookup with fresh equivalent allocation");
    }
    arena_destroy(arena);
    std::cout << "string_keys: " << checks << " checks, " << failures << " failures\n";
    return failures ? 1 : 0;
}
