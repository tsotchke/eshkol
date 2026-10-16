/* Copyright (C) tsotchke. SPDX-License-Identifier: MIT */
#ifndef ESHKOL_COMPILER_PASS_VALUES_H
#define ESHKOL_COMPILER_PASS_VALUES_H
#include "../core/arena_memory.h"
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <vector>

namespace eshkol::frontend::compiler_pass {
using Value = eshkol_tagged_value_t;
static_assert(sizeof(Value) == 16 && offsetof(Value, data) == 8,
              "compiler passes require the native 16-byte tagged ABI");
struct PlanningArena {
    std::unique_ptr<arena_t, decltype(&arena_destroy)> arena{arena_create(65536), arena_destroy};
    eshkol_memctx_t* context = eshkol_memctx_current();
    arena_t* previous = context->allocation_domain;
    PlanningArena() {
        if (!arena) throw std::bad_alloc();
        context->allocation_domain = arena.get();
        if (eshkol_current_arena() != arena.get()) {
            context->allocation_domain = previous;
            throw std::logic_error("compiler pass requires an isolated compiler memory context");
        }
    }
    ~PlanningArena() { context->allocation_domain = previous; }
};
inline Value integer(size_t n) { return eshkol_make_int64(static_cast<int64_t>(n), true); }
inline Value vector(arena_t* arena, const std::vector<Value>& values) {
    void* data = arena_allocate_vector_with_header(arena, values.size());
    if (!data) throw std::bad_alloc();
    *static_cast<uint64_t*>(data) = values.size();
    if (!values.empty()) std::memcpy(static_cast<char*>(data) + 8, values.data(), values.size() * sizeof(Value));
    Value value{}; value.type = ESHKOL_VALUE_HEAP_PTR;
    value.data.ptr_val = reinterpret_cast<uintptr_t>(data);
    return value;
}
inline size_t index(Value value, size_t bound) {
    if (value.type != ESHKOL_VALUE_INT64 || value.data.int_val < 0 ||
        static_cast<uint64_t>(value.data.int_val) >= bound)
        throw std::logic_error("compiler pass returned an invalid integer");
    return static_cast<size_t>(value.data.int_val);
}
inline std::vector<Value> elements(Value value, size_t bound) {
    if (value.type != ESHKOL_VALUE_HEAP_PTR || !value.data.ptr_val)
        throw std::logic_error("compiler pass refused the graph");
    auto* data = reinterpret_cast<const uint64_t*>(static_cast<uintptr_t>(value.data.ptr_val));
    const auto* header = reinterpret_cast<const eshkol_object_header_t*>(data) - 1;
    const auto count = *data;
    if (header->subtype != HEAP_SUBTYPE_VECTOR || count > bound ||
        header->size != 8 + count * sizeof(Value))
        throw std::logic_error("compiler pass returned an invalid vector");
    auto* begin = reinterpret_cast<const Value*>(data + 1);
    return {begin, begin + count};
}
}
#endif
