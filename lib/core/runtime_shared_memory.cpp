/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Shared and weak-reference runtime helpers.
 *
 * These helpers implement the reference-counted allocation ABI registered with
 * the REPL JIT and used by generated ownership paths. They are independent of
 * arena region state and host process/file/thread APIs.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"

#include <cstdlib>

namespace {

constexpr uint8_t ESHKOL_SHARED_DEALLOCATED = 0x01;

}  // namespace

extern "C" {

// Get the shared header from a user pointer (header is before user data).
eshkol_shared_header_t* shared_get_header(void* ptr) {
    if (!ptr) return nullptr;
    return (eshkol_shared_header_t*)((uint8_t*)ptr - sizeof(eshkol_shared_header_t));
}

// Allocate a shared (reference-counted) value.
void* shared_allocate(size_t size, void (*destructor)(void*)) {
    return shared_allocate_typed(size, ESHKOL_VALUE_NULL, destructor);
}

void* shared_allocate_typed(size_t size, uint8_t value_type, void (*destructor)(void*)) {
    const size_t total_size = sizeof(eshkol_shared_header_t) + size;

    // Shared allocations intentionally outlive lexical regions.
    auto* memory = (uint8_t*)std::malloc(total_size);
    if (!memory) {
        eshkol_error("Failed to allocate shared memory of size %zu", size);
        return nullptr;
    }

    auto* header = (eshkol_shared_header_t*)memory;
    header->destructor = destructor;
    header->ref_count = 1;
    header->weak_count = 0;
    header->flags = 0;
    header->value_type = value_type;
    header->reserved = 0;
    header->reserved2 = 0;

    void* user_data = memory + sizeof(eshkol_shared_header_t);

    eshkol_debug("Allocated shared memory at %p (header at %p), size=%zu, type=%d",
                 user_data, (void*)header, size, value_type);

    return user_data;
}

void shared_retain(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    header->ref_count++;

    eshkol_debug("Retained shared at %p, ref_count now %u", ptr, header->ref_count);
}

void shared_release(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    if (header->ref_count == 0) {
        eshkol_warn("Releasing shared with zero ref count at %p", ptr);
        return;
    }

    header->ref_count--;

    eshkol_debug("Released shared at %p, ref_count now %u", ptr, header->ref_count);

    if (header->ref_count == 0) {
        if (header->destructor) {
            eshkol_debug("Calling destructor for shared at %p", ptr);
            header->destructor(ptr);
        }

        if (header->weak_count == 0) {
            eshkol_debug("Freeing shared memory at %p", ptr);
            std::free(header);
        } else {
            header->flags |= ESHKOL_SHARED_DEALLOCATED;
            eshkol_debug("Shared at %p deallocated but %u weak refs remain",
                         ptr, header->weak_count);
        }
    }
}

uint32_t shared_ref_count(void* ptr) {
    if (!ptr) return 0;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return 0;

    return header->ref_count;
}

eshkol_weak_ref_t* weak_ref_create(void* shared_ptr) {
    if (!shared_ptr) return nullptr;

    eshkol_shared_header_t* header = shared_get_header(shared_ptr);
    if (!header) return nullptr;

    auto* weak = (eshkol_weak_ref_t*)std::malloc(sizeof(eshkol_weak_ref_t));
    if (!weak) {
        eshkol_error("Failed to allocate weak reference");
        return nullptr;
    }

    weak->header = header;
    weak->data = shared_ptr;
    header->weak_count++;

    eshkol_debug("Created weak ref at %p to shared %p, weak_count now %u",
                 (void*)weak, shared_ptr, header->weak_count);

    return weak;
}

void* weak_ref_upgrade(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return nullptr;

    if (weak->header->flags & ESHKOL_SHARED_DEALLOCATED) {
        eshkol_debug("Cannot upgrade weak ref - target deallocated");
        return nullptr;
    }

    if (weak->header->ref_count == 0) {
        return nullptr;
    }

    weak->header->ref_count++;

    eshkol_debug("Upgraded weak ref at %p to strong, ref_count now %u",
                 (void*)weak, weak->header->ref_count);

    return weak->data;
}

void weak_ref_release(eshkol_weak_ref_t* weak) {
    if (!weak) return;

    if (weak->header) {
        weak->header->weak_count--;

        eshkol_debug("Released weak ref at %p, weak_count now %u",
                     (void*)weak, weak->header->weak_count);

        if ((weak->header->flags & ESHKOL_SHARED_DEALLOCATED) &&
            weak->header->weak_count == 0) {
            eshkol_debug("Freeing shared header at %p (all refs gone)",
                         (void*)weak->header);
            std::free(weak->header);
        }
    }

    std::free(weak);
}

bool weak_ref_is_alive(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return false;
    if (weak->header->flags & ESHKOL_SHARED_DEALLOCATED) return false;
    return weak->header->ref_count > 0;
}

}  // extern "C"
