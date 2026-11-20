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

#include <atomic>
#include <cstdlib>

namespace {

// Set in the header once the payload destructor has run.
constexpr uint8_t ESHKOL_SHARED_DEALLOCATED = 0x01;

// Increment a count that is currently positive. Returns false, without
// changing it, when the count has already reached zero.
bool retain_count(uint32_t& field, std::memory_order order) {
    std::atomic_ref<uint32_t> counter(field);
    uint32_t count = counter.load(std::memory_order_relaxed);
    while (count) {
        if (count == UINT32_MAX) std::abort();
        if (counter.compare_exchange_weak(count, count + 1, order, std::memory_order_relaxed))
            return true;
    }
    return false;
}

// Each weak handle owns the header. Strong references collectively hold one
// additional owner through completion of the payload destructor.
void release_header(eshkol_shared_header_t* header) {
    if (std::atomic_ref(header->header_owners).fetch_sub(1, std::memory_order_acq_rel) == 1)
        std::free(header);
}

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

/**
 * @brief Allocate a reference-counted shared value with an explicit tagged
 * value type, and return a pointer to its user data.
 *
 * Allocates `sizeof(eshkol_shared_header_t) + size` bytes with malloc(),
 * initializes the header in place (ref_count=1, weak_count=0, flags=0,
 * header_owners=1) with the given destructor and value_type, and returns a
 * pointer just past the header. The allocation is heap-owned and intentionally outlives any
 * lexical arena region — it is freed only when the reference count (and any
 * outstanding weak references) drop to zero via shared_release() /
 * weak_ref_release().
 *
 * @param size        Size in bytes of the user payload (excluding the header).
 * @param value_type  Tagged-value type tag stored in the header, used later
 *                    to interpret the payload.
 * @param destructor  Optional cleanup callback invoked with the user pointer
 *                    when the reference count reaches zero; may be nullptr.
 * @return            Pointer to the user data region, or nullptr if the
 *                    header plus `size` is not representable in size_t or
 *                    the underlying malloc() failed.
 */
void* shared_allocate_typed(size_t size, uint8_t value_type, void (*destructor)(void*)) {
    constexpr size_t header_size = sizeof(eshkol_shared_header_t);
    if (size > SIZE_MAX - header_size) return nullptr;
    const size_t total_size = header_size + size;

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
    header->header_owners = 1;

    void* user_data = memory + sizeof(eshkol_shared_header_t);

    eshkol_debug("Allocated shared memory at %p (header at %p), size=%zu, type=%d",
                 user_data, (void*)header, size, value_type);

    return user_data;
}

/**
 * @brief Add a strong reference to a shared allocation.
 *
 * The caller must already own a strong reference, so the count is positive on
 * entry. The increment is atomic and may race with shared_retain(),
 * shared_release(), weak_ref_upgrade() and weak_ref_release() on other
 * threads. A count of UINT32_MAX terminates the process instead of wrapping.
 *
 * @param ptr  User pointer from shared_allocate() / shared_allocate_typed();
 *             NULL is a no-op.
 */
void shared_retain(void* ptr) {
    if (ptr) retain_count(shared_get_header(ptr)->ref_count, std::memory_order_relaxed);
}

/**
 * @brief Drop a strong reference to a shared allocation.
 *
 * The decrement is atomic. The thread that drops the last strong reference
 * runs the destructor (if any), marks the header ESHKOL_SHARED_DEALLOCATED,
 * and releases the strong group's header owner. The header and payload block
 * is freed when its last owner goes: here when no weak handle exists,
 * otherwise in the weak_ref_release() of the last weak handle. Releasing a
 * value whose count is already zero logs a warning and changes nothing.
 *
 * @param ptr  User pointer from shared_allocate() / shared_allocate_typed();
 *             NULL is a no-op.
 */
void shared_release(void* ptr) {
    if (!ptr) return;
    auto* header = shared_get_header(ptr);
    uint32_t count;
    {
        std::atomic_ref counter(header->ref_count);
        count = counter.load(std::memory_order_relaxed);
        do {
            if (!count) {
                eshkol_warn("Releasing shared with zero ref count at %p", ptr);
                return;
            }
        } while (!counter.compare_exchange_weak(count, count - 1, std::memory_order_acq_rel,
                                                std::memory_order_relaxed));
    }
    if (count == 1) {
        if (header->destructor) header->destructor(ptr);
        header->flags |= ESHKOL_SHARED_DEALLOCATED;
        release_header(header);
    }
}

/**
 * @brief Read the strong reference count of a shared allocation (diagnostic).
 *
 * @param ptr  User pointer from shared_allocate() / shared_allocate_typed().
 * @return     The current count, or 0 for NULL. Other threads may change it
 *             immediately after the read.
 */
uint32_t shared_ref_count(void* ptr) {
    return ptr ? std::atomic_ref(shared_get_header(ptr)->ref_count).load(std::memory_order_relaxed) : 0;
}

/**
 * @brief Create a weak handle to a shared allocation.
 *
 * The caller must own a strong reference while creating the handle. The handle
 * owns the header (not the payload): the header stays allocated until the
 * handle is released with weak_ref_release(), so upgrading or querying a
 * handle never reads freed memory. weak_count counts the explicit handles.
 *
 * @param shared_ptr  User pointer from shared_allocate() /
 *                    shared_allocate_typed().
 * @return            A new handle owned by the caller, or nullptr if
 *                    `shared_ptr` is NULL or the handle allocation fails.
 */
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
    retain_count(header->header_owners, std::memory_order_relaxed);
    std::atomic_ref(header->weak_count).fetch_add(1, std::memory_order_relaxed);
    return weak;
}

/**
 * @brief Obtain a strong reference from a weak handle.
 *
 * Atomically increments the strong count only while it is positive, so an
 * upgrade that races with the final shared_release() either wins (and keeps
 * the payload alive) or observes zero and fails; it never revives a payload
 * whose destructor has started.
 *
 * @param weak  Handle from weak_ref_create().
 * @return      The payload pointer, now strongly referenced and released with
 *              shared_release(), or nullptr if the payload is gone or `weak`
 *              is NULL.
 */
void* weak_ref_upgrade(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return nullptr;
    return retain_count(weak->header->ref_count, std::memory_order_acquire) ? weak->data : nullptr;
}

/**
 * @brief Release a weak handle and its ownership of the header.
 *
 * Frees the handle. When this was the last owner of the header (no strong
 * references and no other weak handles remain), also frees the header and
 * payload block.
 *
 * @param weak  Handle from weak_ref_create(); NULL is a no-op.
 */
void weak_ref_release(eshkol_weak_ref_t* weak) {
    if (!weak) return;
    if (weak->header) {
        std::atomic_ref(weak->header->weak_count).fetch_sub(1, std::memory_order_relaxed);
        release_header(weak->header);
    }
    std::free(weak);
}

/**
 * @brief Report whether a weak handle's payload still has a strong owner.
 *
 * The answer can change as soon as it is returned; use weak_ref_upgrade() to
 * obtain the payload.
 *
 * @param weak  Handle from weak_ref_create().
 * @return      true if the strong count is positive.
 */
bool weak_ref_is_alive(eshkol_weak_ref_t* weak) {
    return weak && weak->header &&
           std::atomic_ref(weak->header->ref_count).load(std::memory_order_acquire) != 0;
}

}  // extern "C"
