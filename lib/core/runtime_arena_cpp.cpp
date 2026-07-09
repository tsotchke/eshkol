/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * C++ RAII wrapper for the C arena runtime.
 */

#include "arena_memory.h"

#include <new>

/**
 * @brief Construct an Arena, creating the underlying C arena_t.
 * @throws std::bad_alloc if the underlying arena_create() allocation fails.
 */
Arena::Arena(size_t default_block_size) : arena_(arena_create(default_block_size)) {
    if (!arena_) {
        throw std::bad_alloc();
    }
}

/** @brief Destroy the Arena, releasing the underlying C arena_t (and all memory it owns). */
Arena::~Arena() {
    arena_destroy(arena_);
}

/** @brief Move-construct: take ownership of `other`'s underlying arena_t, leaving `other` empty. */
Arena::Arena(Arena&& other) noexcept : arena_(other.arena_) {
    other.arena_ = nullptr;
}

/** @brief Move-assign: destroy this arena's current arena_t, then take ownership of `other`'s, leaving `other` empty. */
Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        arena_destroy(arena_);
        arena_ = other.arena_;
        other.arena_ = nullptr;
    }
    return *this;
}

/** @brief Allocate `size` bytes from the arena (default alignment). @return Pointer to the allocation, or NULL on failure. */
void* Arena::allocate(size_t size) {
    return arena_allocate(arena_, size);
}

/** @brief Allocate `size` bytes from the arena aligned to `alignment`. @return Pointer to the allocation, or NULL on failure. */
void* Arena::allocate_aligned(size_t size, size_t alignment) {
    return arena_allocate_aligned(arena_, size, alignment);
}

/** @brief Allocate `size` zero-initialised bytes from the arena. @return Pointer to the allocation, or NULL on failure. */
void* Arena::allocate_zeroed(size_t size) {
    return arena_allocate_zeroed(arena_, size);
}

/** @brief Total bytes currently in use across all blocks of the underlying arena. */
size_t Arena::get_used_memory() const {
    return arena_get_used_memory(arena_);
}

/** @brief Total bytes reserved (allocated from the OS/malloc) across all blocks of the underlying arena. */
size_t Arena::get_total_memory() const {
    return arena_get_total_memory(arena_);
}

/** @brief Number of memory blocks currently chained in the underlying arena. */
size_t Arena::get_block_count() const {
    return arena_get_block_count(arena_);
}

/** @brief Reset the arena, reclaiming all allocations made since it was created (or last reset). */
void Arena::reset() {
    arena_reset(arena_);
}

/** @brief Push a new nested allocation scope on `arena`, recording the current allocation mark so it can be rewound on destruction. */
Arena::Scope::Scope(Arena& arena) : arena_(&arena), active_(true) {
    arena_push_scope(arena.arena_);
}

/** @brief Pop the scope (if still active), rewinding the arena to the mark recorded at construction and reclaiming everything allocated within the scope. */
Arena::Scope::~Scope() {
    if (active_) {
        arena_pop_scope(arena_->arena_);
    }
}
