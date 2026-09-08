/*
 * Load one ESKM input through the public native model/tensor entry points and
 * print a stable digest of every successfully materialized value.  The seeded
 * driver runs this probe out of process so malformed files cannot take down the
 * campaign.
 */

#include <eshkol/model_io.h>

#include "arena_memory.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace {

constexpr std::size_t kArenaBytes = 32u * 1024u * 1024u;
constexpr std::size_t kMaxObservedRecords = 4096;
constexpr std::size_t kMaxObservedName = 1024u * 1024u;
constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct Observation {
    bool accepted = false;
    bool malformed_result = false;
    std::uint64_t digest = kFnvOffset;
};

eshkol_tagged_value_t make_heap_ptr(void* ptr) {
    eshkol_tagged_value_t value{};
    value.type = ESHKOL_VALUE_HEAP_PTR;
    value.data.ptr_val = reinterpret_cast<std::uint64_t>(ptr);
    return value;
}

eshkol_tagged_value_t make_path(arena_t* arena, const std::string& path) {
    char* bytes = arena_allocate_string_with_header(arena, path.size());
    if (!bytes) return {};
    std::memcpy(bytes, path.data(), path.size());
    bytes[path.size()] = '\0';
    return make_heap_ptr(bytes);
}

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= kFnvPrime;
    }
}

void hash_u32(std::uint64_t& hash, std::uint32_t value) {
    unsigned char bytes[4];
    for (unsigned i = 0; i < 4; ++i) bytes[i] = static_cast<unsigned char>(value >> (8u * i));
    hash_bytes(hash, bytes, sizeof(bytes));
}

void hash_u64(std::uint64_t& hash, std::uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = static_cast<unsigned char>(value >> (8u * i));
    hash_bytes(hash, bytes, sizeof(bytes));
}

bool is_pair(const eshkol_tagged_value_t& value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

const arena_tagged_cons_cell_t* as_pair(const eshkol_tagged_value_t& value) {
    if (!is_pair(value) || value.data.ptr_val == 0) return nullptr;
    return reinterpret_cast<const arena_tagged_cons_cell_t*>(value.data.ptr_val);
}

bool tagged_name(const eshkol_tagged_value_t& value, const char** data, std::size_t* size) {
    if (!data || !size || value.type != ESHKOL_VALUE_HEAP_PTR || value.data.ptr_val == 0) return false;
    const auto* ptr = reinterpret_cast<const void*>(value.data.ptr_val);
    const auto subtype = ESHKOL_GET_SUBTYPE(ptr);
    if (subtype != HEAP_SUBTYPE_STRING && subtype != HEAP_SUBTYPE_SYMBOL) return false;
    const auto* chars = reinterpret_cast<const char*>(ptr);
    std::size_t length = 0;
    while (length <= kMaxObservedName && chars[length] != '\0') ++length;
    if (length > kMaxObservedName) return false;
    *data = chars;
    *size = length;
    return true;
}

const eshkol_tensor_t* tagged_tensor(const eshkol_tagged_value_t& value) {
    if (!ESHKOL_IS_TENSOR_COMPAT(value) || value.data.ptr_val == 0) return nullptr;
    return reinterpret_cast<const eshkol_tensor_t*>(value.data.ptr_val);
}

bool hash_tensor(std::uint64_t& hash, const eshkol_tensor_t* tensor) {
    if (!tensor) return false;
    if (tensor->dtype != ESHKOL_TENSOR_DTYPE_F64) return false;
    if (tensor->num_dimensions > 0 && !tensor->dimensions) return false;
    if (tensor->total_elements > 0 && !tensor->elements) return false;
    hash_u32(hash, static_cast<std::uint32_t>(tensor->num_dimensions));
    for (std::uint64_t i = 0; i < tensor->num_dimensions; ++i) {
        hash_u64(hash, tensor->dimensions[i]);
    }
    hash_u64(hash, tensor->total_elements);
    for (std::uint64_t i = 0; i < tensor->total_elements; ++i) {
        hash_u64(hash, static_cast<std::uint64_t>(tensor->elements[i]));
    }
    return true;
}

Observation observe_model(const std::string& path) {
    Observation observation;
    arena_t* arena = arena_create_bounded(kArenaBytes);
    if (!arena) {
        observation.malformed_result = true;
        return observation;
    }
    const eshkol_tagged_value_t path_value = make_path(arena, path);
    eshkol_tagged_value_t result{};
    eshkol_model_load_tagged(arena, &path_value, &result);
    if (result.type == ESHKOL_VALUE_NULL) {
        arena_destroy(arena);
        return observation;
    }

    eshkol_tagged_value_t cursor = result;
    std::uint32_t count = 0;
    while (is_pair(cursor) && count < kMaxObservedRecords) {
        const auto* list_node = as_pair(cursor);
        const eshkol_tagged_value_t entry = list_node->car;
        const auto* entry_pair = as_pair(entry);
        if (!entry_pair) {
            observation.malformed_result = true;
            break;
        }
        const char* name = nullptr;
        std::size_t name_size = 0;
        if (!tagged_name(entry_pair->car, &name, &name_size) || name_size > UINT32_MAX) {
            observation.malformed_result = true;
            break;
        }
        hash_u32(observation.digest, static_cast<std::uint32_t>(name_size));
        hash_bytes(observation.digest, name, name_size);
        if (!hash_tensor(observation.digest, tagged_tensor(entry_pair->cdr))) {
            observation.malformed_result = true;
            break;
        }
        ++count;
        cursor = list_node->cdr;
    }
    if (cursor.type != ESHKOL_VALUE_NULL) {
        observation.malformed_result = true;
    }
    if (!observation.malformed_result) {
        hash_u32(observation.digest, count);
        observation.accepted = true;
    }
    arena_destroy(arena);
    return observation;
}

Observation observe_tensor(const std::string& path) {
    Observation observation;
    arena_t* arena = arena_create_bounded(kArenaBytes);
    if (!arena) {
        observation.malformed_result = true;
        return observation;
    }
    const eshkol_tagged_value_t path_value = make_path(arena, path);
    eshkol_tagged_value_t result{};
    eshkol_tensor_load_tagged(arena, &path_value, &result);
    if (result.type == ESHKOL_VALUE_NULL) {
        arena_destroy(arena);
        return observation;
    }
    const auto* tensor = tagged_tensor(result);
    if (!tensor || !hash_tensor(observation.digest, tensor)) {
        observation.malformed_result = true;
    } else {
        observation.accepted = true;
    }
    arena_destroy(arena);
    return observation;
}

std::string render(const Observation& observation) {
    if (observation.malformed_result) return "malformed";
    if (!observation.accepted) return "reject";
    std::ostringstream out;
    out << "accept:" << std::hex << std::setw(16) << std::setfill('0') << observation.digest;
    return out.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s INPUT.eskm\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const Observation model = observe_model(path);
    const Observation tensor = observe_tensor(path);
    std::cout << "model=" << render(model) << " tensor=" << render(tensor) << '\n';
    return (model.malformed_result || tensor.malformed_result) ? 3 : 0;
}
