#ifndef ESHKOL_RUNTIME_STRING_BYTES_H
#define ESHKOL_RUNTIME_STRING_BYTES_H

#include "arena_memory.h"
#include <cstring>
#include <string_view>

// Call only after validating the value's string type and non-null payload.
// HEAP_PTR strings carry an object header; legacy STRING_PTR also permits raw
// C strings, so it must never be probed before the payload to guess a header.
// Canonical producers must preserve the header-bearing representation.
static inline std::string_view eshkol_tagged_string_bytes(
        const eshkol_tagged_value_t* value) {
    const char* bytes = reinterpret_cast<const char*>(value->data.ptr_val);
    if (value->type == ESHKOL_VALUE_HEAP_PTR) {
        const auto* header = ESHKOL_GET_HEADER(bytes);
        if (header->size > 0) return {bytes, header->size - 1};
    }
    return {bytes, std::strlen(bytes)};
}

#endif
