/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * C++ RAII wrapper for the C arena runtime.
 */

#include "arena_memory.h"

#include <new>

Arena::Arena(size_t default_block_size) : arena_(arena_create(default_block_size)) {
    if (!arena_) {
        throw std::bad_alloc();
    }
}

Arena::~Arena() {
    arena_destroy(arena_);
}

Arena::Arena(Arena&& other) noexcept : arena_(other.arena_) {
    other.arena_ = nullptr;
}

Arena& Arena::operator=(Arena&& other) noexcept {
    if (this != &other) {
        arena_destroy(arena_);
        arena_ = other.arena_;
        other.arena_ = nullptr;
    }
    return *this;
}

void* Arena::allocate(size_t size) {
    return arena_allocate(arena_, size);
}

void* Arena::allocate_aligned(size_t size, size_t alignment) {
    return arena_allocate_aligned(arena_, size, alignment);
}

void* Arena::allocate_zeroed(size_t size) {
    return arena_allocate_zeroed(arena_, size);
}

size_t Arena::get_used_memory() const {
    return arena_get_used_memory(arena_);
}

size_t Arena::get_total_memory() const {
    return arena_get_total_memory(arena_);
}

size_t Arena::get_block_count() const {
    return arena_get_block_count(arena_);
}

void Arena::reset() {
    arena_reset(arena_);
}

Arena::Scope::Scope(Arena& arena) : arena_(&arena), active_(true) {
    arena_push_scope(arena.arena_);
}

Arena::Scope::~Scope() {
    if (active_) {
        arena_pop_scope(arena_->arena_);
    }
}
