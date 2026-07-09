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

/**
 * @brief Allocate a reference-counted shared value with an explicit tagged
 * value type, and return a pointer to its user data.
 *
 * Allocates `sizeof(eshkol_shared_header_t) + size` bytes with malloc(),
 * initializes the header in place (ref_count=1, weak_count=0, flags=0) with
 * the given destructor and value_type, and returns a pointer just past the
 * header. The allocation is heap-owned and intentionally outlives any
 * lexical arena region — it is freed only when the reference count (and any
 * outstanding weak references) drop to zero via shared_release() /
 * weak_ref_release(). Not thread-safe by itself: the header fields it
 * initializes are plain (non-atomic) values, so callers sharing this pointer
 * across threads must provide their own synchronization.
 *
 * @param size        Size in bytes of the user payload (excluding the header).
 * @param value_type  Tagged-value type tag stored in the header, used later
 *                    to interpret the payload.
 * @param destructor  Optional cleanup callback invoked with the user pointer
 *                    when the reference count reaches zero; may be nullptr.
 * @return            Pointer to the user data region, or nullptr if the
 *                    underlying malloc() failed.
 */
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

/**
 * @brief Increment the strong reference count of a shared allocation.
 *
 * No-op if `ptr` is NULL or has no header. Not thread-safe: `ref_count` is a
 * plain uint32_t incremented without a lock or atomic op, so concurrent
 * shared_retain()/shared_release() calls on the same value from multiple
 * threads can race.
 *
 * @param ptr  User pointer previously returned by shared_allocate() /
 *             shared_allocate_typed().
 */
void shared_retain(void* ptr) {
    if (!ptr) return;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return;

    header->ref_count++;

    eshkol_debug("Retained shared at %p, ref_count now %u", ptr, header->ref_count);
}

/**
 * @brief Decrement the strong reference count of a shared allocation,
 * running its destructor and/or freeing it once the count reaches zero.
 *
 * No-op if `ptr` is NULL or has no header. Warns (via eshkol_warn()) and
 * returns without decrementing if the ref count is already zero (double
 * release). When the count drops to zero: invokes the stored destructor (if
 * any) with `ptr`, then frees the header+payload block immediately if there
 * are no outstanding weak references (weak_count == 0); otherwise it leaves
 * the memory allocated but marks it ESHKOL_SHARED_DEALLOCATED so
 * weak_ref_upgrade() can detect the value is gone, and the final free is
 * deferred to weak_ref_release() once the last weak reference goes away.
 * Not thread-safe: `ref_count`/`flags` are plain fields updated without a
 * lock or atomic op.
 *
 * @param ptr  User pointer previously returned by shared_allocate() /
 *             shared_allocate_typed().
 */
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

/**
 * @brief Read the current strong reference count of a shared allocation
 * (diagnostic use).
 *
 * @param ptr  User pointer previously returned by shared_allocate() /
 *             shared_allocate_typed(); NULL or headerless returns 0.
 * @return     Current ref_count value.
 */
uint32_t shared_ref_count(void* ptr) {
    if (!ptr) return 0;

    eshkol_shared_header_t* header = shared_get_header(ptr);
    if (!header) return 0;

    return header->ref_count;
}

/**
 * @brief Create a weak reference to a shared allocation, without affecting
 * its strong reference count.
 *
 * Allocates a small eshkol_weak_ref_t block (owned by the caller — release
 * with weak_ref_release()) that records the target's header and original
 * data pointer, and increments the header's `weak_count`. A weak reference
 * does not keep the value alive; use weak_ref_upgrade() to attempt to obtain
 * a strong reference. Not thread-safe: `weak_count` is a plain uint32_t
 * incremented without a lock or atomic op.
 *
 * @param shared_ptr  User pointer previously returned by shared_allocate() /
 *                    shared_allocate_typed().
 * @return            A newly allocated weak reference, or nullptr if
 *                    `shared_ptr` is NULL/headerless or the weak-ref
 *                    allocation itself fails.
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
    header->weak_count++;

    eshkol_debug("Created weak ref at %p to shared %p, weak_count now %u",
                 (void*)weak, shared_ptr, header->weak_count);

    return weak;
}

/**
 * @brief Attempt to obtain a strong reference from a weak reference.
 *
 * Fails if the target has already been deallocated (its header's
 * ESHKOL_SHARED_DEALLOCATED flag is set, meaning shared_release() dropped
 * the strong count to zero while weak references still existed) or if its
 * ref_count is already zero. Otherwise increments ref_count and returns the
 * original data pointer as a new strong reference that the caller must
 * eventually release with shared_release(). Not thread-safe: `ref_count`
 * and `flags` are plain fields read/updated without a lock or atomic op, so
 * this is not safe to race against a concurrent shared_release() on the
 * same value.
 *
 * @param weak  Weak reference from weak_ref_create().
 * @return      The shared value's user pointer (now strongly referenced) on
 *              success, or nullptr if `weak` is NULL/headerless or the
 *              target is gone.
 */
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

/**
 * @brief Release a weak reference, freeing the target's header if it was
 * already deallocated and this was the last weak reference to it.
 *
 * Decrements the target header's `weak_count`. If the target was already
 * strong-released (ESHKOL_SHARED_DEALLOCATED set, meaning
 * shared_release() deferred the final free pending outstanding weak refs)
 * and this was the last weak reference (weak_count reaches 0), frees the
 * header+payload block here. Always frees the eshkol_weak_ref_t block
 * itself. Not thread-safe: `weak_count` is a plain uint32_t decremented
 * without a lock or atomic op.
 *
 * @param weak  Weak reference from weak_ref_create(); NULL is a no-op.
 */
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

/**
 * @brief Check whether a weak reference's target is still alive, without
 * upgrading to a strong reference.
 *
 * @param weak  Weak reference from weak_ref_create().
 * @return      true if `weak` and its header are non-null, the target has
 *              not been marked ESHKOL_SHARED_DEALLOCATED, and its
 *              ref_count is greater than zero; false otherwise.
 */
bool weak_ref_is_alive(eshkol_weak_ref_t* weak) {
    if (!weak || !weak->header) return false;
    if (weak->header->flags & ESHKOL_SHARED_DEALLOCATED) return false;
    return weak->header->ref_count > 0;
}

}  // extern "C"
