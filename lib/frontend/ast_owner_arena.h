/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_owner_arena.h
 * @brief The rooted, chunked, append-only arena behind the frontend owners.
 *
 * Internal to lib/frontend. The AST string owner (ADR-0021,
 * inc/eshkol/frontend/ast_strings.h) and the AST node-storage owner
 * (inc/eshkol/frontend/ast_nodes.h) are two instances of this arena with the
 * same lifetime rule: storage is valid for the whole compilation that
 * produced it and is released all at once by a teardown call.
 *
 * Each instance is a singly linked list of chunks rooted at a member of a
 * namespace-scope object, so LeakSanitizer scans the root, follows every
 * `next` link and classes the storage as live. Allocation bumps a cursor in
 * the newest chunk under a mutex; a request that does not fit starts a new
 * chunk, and a request larger than a quarter chunk gets a chunk of its own
 * placed behind the head, so the head keeps serving small requests.
 *
 * Under AddressSanitizer each allocation is followed by a poisoned redzone and
 * unused chunk space stays poisoned, so an overread is still reported as it
 * was when each object had its own heap block. After teardown the chunks are
 * freed, so a stale pointer is a heap-use-after-free.
 */
#ifndef ESHKOL_FRONTEND_AST_OWNER_ARENA_H
#define ESHKOL_FRONTEND_AST_OWNER_ARENA_H

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ESHKOL_AST_OWNER_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define ESHKOL_AST_OWNER_ASAN 1
#endif

#if defined(ESHKOL_AST_OWNER_ASAN)
#include <sanitizer/asan_interface.h>
#define ESHKOL_AST_OWNER_POISON(p, n) ASAN_POISON_MEMORY_REGION((p), (n))
#define ESHKOL_AST_OWNER_UNPOISON(p, n) ASAN_UNPOISON_MEMORY_REGION((p), (n))
#else
#define ESHKOL_AST_OWNER_POISON(p, n) ((void)(p), (void)(n))
#define ESHKOL_AST_OWNER_UNPOISON(p, n) ((void)(p), (void)(n))
#endif

namespace eshkol::frontend::detail {

/** Counters common to every owner arena. */
struct OwnerArenaCounters {
    uint64_t allocations;
    uint64_t bytes_requested;
    uint64_t live_chunks;
    uint64_t live_bytes_reserved;
    uint64_t teardowns;
};

class OwnerArena {
public:
    /** Payload bytes in a standard chunk. */
    static constexpr size_t kChunkPayload = 64u * 1024u;
    /** Requests above this size get a dedicated chunk. */
    static constexpr size_t kLargeRequest = kChunkPayload / 4u;
    /**
     * Allocation granule and alignment. ASan tracks addressability in 8-byte
     * granules, so every allocation starts on a granule boundary; that lets
     * the redzone after one object be poisoned without also poisoning the
     * start of the next. It is also the alignment every owned type needs.
     */
    static constexpr size_t kGranule = 8u;
#if defined(ESHKOL_AST_OWNER_ASAN)
    /** Poisoned gap after each allocation, so an overread is reported. */
    static constexpr size_t kRedzone = 16u;
#else
    static constexpr size_t kRedzone = 0u;
#endif

    /**
     * @p stats_label names the teardown report line; @p stats_env is the
     * environment variable that enables it.
     */
    constexpr OwnerArena(const char* stats_label, const char* stats_env) noexcept
        : stats_label_(stats_label), stats_env_(stats_env) {}

    OwnerArena(const OwnerArena&) = delete;
    OwnerArena& operator=(const OwnerArena&) = delete;

    /**
     * Zero-filled, granule-aligned storage for @p bytes (at least one byte).
     * Returns nullptr on allocation failure or size overflow.
     */
    void* allocate(size_t bytes) {
        if (bytes == 0) bytes = 1;
        const size_t need = round_up(bytes + kRedzone, kGranule);
        if (need < bytes) return nullptr; /* overflow */
        std::lock_guard<std::mutex> lock(mutex_);
        Chunk* chunk = head_;
        if (need > kLargeRequest || !chunk || chunk->capacity - chunk->used < need) {
            chunk = new_chunk_locked(need);
            if (!chunk) return nullptr;
        }
        unsigned char* out = chunk->payload() + chunk->used;
        chunk->used += need;
        /* Only the requested bytes become addressable; the tail of the
         * granule run and the redzone stay poisoned. Chunk memory came from
         * calloc and is never reused before teardown, so it is still zero. */
        ESHKOL_AST_OWNER_UNPOISON(out, bytes);
        allocations_.fetch_add(1, std::memory_order_relaxed);
        bytes_requested_.fetch_add(bytes, std::memory_order_relaxed);
        return out;
    }

    /** True when @p p points into storage this arena currently holds. */
    bool owns(const void* p) {
        if (!p) return false;
        const unsigned char* q = static_cast<const unsigned char*>(p);
        std::lock_guard<std::mutex> lock(mutex_);
        for (const Chunk* c = head_; c; c = c->next) {
            const unsigned char* begin = c->payload();
            if (q >= begin && q < begin + c->used) return true;
        }
        return false;
    }

    OwnerArenaCounters counters() {
        OwnerArenaCounters out{};
        out.allocations = allocations_.load(std::memory_order_relaxed);
        out.bytes_requested = bytes_requested_.load(std::memory_order_relaxed);
        std::lock_guard<std::mutex> lock(mutex_);
        out.live_chunks = live_chunks_;
        out.live_bytes_reserved = live_bytes_reserved_;
        out.teardowns = teardowns_;
        return out;
    }

    /**
     * Release every chunk. When the stats variable is set (to anything but
     * `0`/`false`), first write one line to stderr:
     * `<label>: allocations=N requested=N chunks=N reserved=N`.
     */
    void teardown() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (stats_enabled()) {
            std::fprintf(stderr,
                         "%s: allocations=%llu requested=%llu chunks=%llu reserved=%llu\n",
                         stats_label_,
                         (unsigned long long)allocations_.load(std::memory_order_relaxed),
                         (unsigned long long)bytes_requested_.load(std::memory_order_relaxed),
                         (unsigned long long)live_chunks_,
                         (unsigned long long)live_bytes_reserved_);
            std::fflush(stderr);
        }
        if (!head_) return;
        Chunk* c = head_;
        head_ = nullptr;
        while (c) {
            Chunk* next = c->next;
            ESHKOL_AST_OWNER_UNPOISON(c->payload(), c->capacity);
            std::free(c);
            c = next;
        }
        live_chunks_ = 0;
        live_bytes_reserved_ = 0;
        ++teardowns_;
    }

private:
    struct Chunk {
        Chunk* next;
        size_t capacity; /* payload bytes */
        size_t used;     /* payload bytes handed out, including redzones */
        /* Payload follows the header, on an allocation granule. */
        unsigned char* payload() { return reinterpret_cast<unsigned char*>(this + 1); }
        const unsigned char* payload() const {
            return reinterpret_cast<const unsigned char*>(this + 1);
        }
    };
    static_assert(sizeof(Chunk) % kGranule == 0,
                  "chunk payload must start on an allocation granule");

    static size_t round_up(size_t n, size_t to) { return (n + to - 1u) / to * to; }

    /** Allocate and link a chunk able to hold @p need payload bytes. */
    Chunk* new_chunk_locked(size_t need) {
        const size_t capacity = need > kChunkPayload ? round_up(need, kGranule) : kChunkPayload;
        void* raw = std::calloc(1, sizeof(Chunk) + capacity);
        if (!raw) return nullptr;
        Chunk* chunk = static_cast<Chunk*>(raw);
        chunk->capacity = capacity;
        chunk->used = 0;
        ESHKOL_AST_OWNER_POISON(chunk->payload(), capacity);
        if (need > kLargeRequest && head_) {
            chunk->next = head_->next;
            head_->next = chunk;
        } else {
            chunk->next = head_;
            head_ = chunk;
        }
        ++live_chunks_;
        live_bytes_reserved_ += sizeof(Chunk) + capacity;
        return chunk;
    }

    /* Resolved once per arena: a measurement must not change discipline
     * halfway through a process. Called with mutex_ held. */
    bool stats_enabled() {
        if (stats_state_ == 0) {
            const char* raw = std::getenv(stats_env_);
            const bool on = raw && raw[0] != '\0' && std::strcmp(raw, "0") != 0 &&
                            std::strcmp(raw, "false") != 0 && std::strcmp(raw, "FALSE") != 0;
            stats_state_ = on ? 1 : 2;
        }
        return stats_state_ == 1;
    }

    const char* stats_label_;
    const char* stats_env_;
    /** The process root. Newest chunk first. */
    Chunk* head_ = nullptr;
    std::mutex mutex_;
    std::atomic<uint64_t> allocations_{0};
    std::atomic<uint64_t> bytes_requested_{0};
    uint64_t live_chunks_ = 0;         /* guarded by mutex_ */
    uint64_t live_bytes_reserved_ = 0; /* guarded by mutex_ */
    uint64_t teardowns_ = 0;           /* guarded by mutex_ */
    int stats_state_ = 0;              /* guarded by mutex_; 0 unknown, 1 on, 2 off */
};

}  // namespace eshkol::frontend::detail

#endif /* ESHKOL_FRONTEND_AST_OWNER_ARENA_H */
