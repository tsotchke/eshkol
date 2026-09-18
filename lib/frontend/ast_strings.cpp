/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file ast_strings.cpp
 * @brief Storage for AST string payloads (ADR-0021).
 *
 * A singly linked list of chunks rooted at one global. Allocation bumps a
 * cursor in the newest chunk under a mutex; a request that does not fit
 * starts a new chunk, and a request larger than a quarter chunk gets a chunk
 * of its own so a long string literal never strands most of a standard one.
 *
 * The list root is what makes the storage *reachable*: LeakSanitizer scans
 * globals, follows `g_head` and every `next` link, and classes each string as
 * live. Nothing in this file is, or needs to be, a suppression.
 */

#include <eshkol/frontend/ast_strings.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define ESHKOL_AST_STRINGS_ASAN 1
#endif
#endif
#if defined(__SANITIZE_ADDRESS__)
#define ESHKOL_AST_STRINGS_ASAN 1
#endif

#if defined(ESHKOL_AST_STRINGS_ASAN)
#include <sanitizer/asan_interface.h>
#define AST_STRINGS_POISON(p, n) ASAN_POISON_MEMORY_REGION((p), (n))
#define AST_STRINGS_UNPOISON(p, n) ASAN_UNPOISON_MEMORY_REGION((p), (n))
#else
#define AST_STRINGS_POISON(p, n) ((void)(p), (void)(n))
#define AST_STRINGS_UNPOISON(p, n) ((void)(p), (void)(n))
#endif

namespace {

/** Payload bytes in a standard chunk. */
constexpr size_t kChunkPayload = 64u * 1024u;

/** Requests above this size get a dedicated chunk. */
constexpr size_t kLargeRequest = kChunkPayload / 4u;

/**
 * Allocation granule. ASan tracks addressability in 8-byte granules, so
 * every allocation starts on a granule boundary; that is what lets the
 * redzone after one string be poisoned without also poisoning the start of
 * the next one.
 */
constexpr size_t kGranule = 8u;

#if defined(ESHKOL_AST_STRINGS_ASAN)
/** Poisoned gap after each allocation, so an overread is reported. */
constexpr size_t kRedzone = 16u;
#else
constexpr size_t kRedzone = 0u;
#endif

struct Chunk {
    Chunk* next;
    size_t capacity;  /* payload bytes */
    size_t used;      /* payload bytes handed out, including redzones */
    /* Payload follows the header, on an allocation granule. */
    unsigned char* payload() { return reinterpret_cast<unsigned char*>(this + 1); }
    const unsigned char* payload() const {
        return reinterpret_cast<const unsigned char*>(this + 1);
    }
};

static_assert(sizeof(Chunk) % kGranule == 0,
              "chunk payload must start on an allocation granule");

/** The process root. Newest chunk first. */
Chunk* g_head = nullptr;

std::mutex g_mutex;

std::atomic<uint64_t> g_allocations{0};
std::atomic<uint64_t> g_bytes_requested{0};
uint64_t g_live_chunks = 0;         /* guarded by g_mutex */
uint64_t g_live_bytes_reserved = 0; /* guarded by g_mutex */
uint64_t g_teardowns = 0;           /* guarded by g_mutex */

inline size_t round_up(size_t n, size_t to) { return (n + to - 1u) / to * to; }

/** Allocate and link a chunk able to hold @p need payload bytes. */
Chunk* new_chunk_locked(size_t need) {
    const size_t capacity = need > kChunkPayload ? round_up(need, kGranule) : kChunkPayload;
    void* raw = std::calloc(1, sizeof(Chunk) + capacity);
    if (!raw) return nullptr;
    Chunk* chunk = static_cast<Chunk*>(raw);
    chunk->capacity = capacity;
    chunk->used = 0;
    AST_STRINGS_POISON(chunk->payload(), capacity);
    /* A dedicated (large) chunk goes behind the current head so the head's
     * remaining space keeps serving small requests. */
    if (need > kLargeRequest && g_head) {
        chunk->next = g_head->next;
        g_head->next = chunk;
    } else {
        chunk->next = g_head;
        g_head = chunk;
    }
    ++g_live_chunks;
    g_live_bytes_reserved += sizeof(Chunk) + capacity;
    return chunk;
}

}  // namespace

extern "C" char* eshkol_ast_string_alloc(size_t bytes) {
    if (bytes == 0) bytes = 1;
    const size_t need = round_up(bytes + kRedzone, kGranule);
    if (need < bytes) return nullptr; /* overflow */

    std::lock_guard<std::mutex> lock(g_mutex);
    Chunk* chunk = g_head;
    if (need > kLargeRequest || !chunk || chunk->capacity - chunk->used < need) {
        chunk = new_chunk_locked(need);
        if (!chunk) return nullptr;
    }
    unsigned char* out = chunk->payload() + chunk->used;
    chunk->used += need;
    /* Only the requested bytes become addressable; the tail of the granule
     * run and the redzone stay poisoned. Chunk memory came from calloc and is
     * never reused before teardown, so it is still zero. */
    AST_STRINGS_UNPOISON(out, bytes);

    g_allocations.fetch_add(1, std::memory_order_relaxed);
    g_bytes_requested.fetch_add(bytes, std::memory_order_relaxed);
    return reinterpret_cast<char*>(out);
}

extern "C" char* eshkol_ast_strndup(const char* s, size_t n) {
    if (!s && n > 0) return nullptr;
    char* out = eshkol_ast_string_alloc(n + 1);
    if (!out) return nullptr;
    if (n) std::memcpy(out, s, n);
    out[n] = '\0';
    return out;
}

extern "C" char* eshkol_ast_strdup(const char* s) {
    if (!s) return nullptr;
    return eshkol_ast_strndup(s, std::strlen(s));
}

extern "C" bool eshkol_ast_string_is_owned(const void* p) {
    if (!p) return false;
    const unsigned char* q = static_cast<const unsigned char*>(p);
    std::lock_guard<std::mutex> lock(g_mutex);
    for (const Chunk* c = g_head; c; c = c->next) {
        const unsigned char* begin = c->payload();
        if (q >= begin && q < begin + c->used) return true;
    }
    return false;
}

extern "C" void eshkol_ast_strings_stats(eshkol_ast_strings_stats_t* out) {
    if (!out) return;
    out->allocations = g_allocations.load(std::memory_order_relaxed);
    out->bytes_requested = g_bytes_requested.load(std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_mutex);
    out->live_chunks = g_live_chunks;
    out->live_bytes_reserved = g_live_bytes_reserved;
    out->teardowns = g_teardowns;
}

namespace {

bool stats_enabled() {
    /* Resolved once, like ESHKOL_NODE_IDENTITY_STATS: a measurement must not
     * change discipline halfway through a process. */
    static const bool enabled = [] {
        const char* raw = std::getenv("ESHKOL_AST_STRINGS_STATS");
        return raw && raw[0] != '\0' && std::strcmp(raw, "0") != 0 &&
               std::strcmp(raw, "false") != 0 && std::strcmp(raw, "FALSE") != 0;
    }();
    return enabled;
}

}  // namespace

extern "C" void eshkol_ast_strings_teardown(void) {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (stats_enabled()) {
        /* One stable, machine-readable line, printed while the numbers still
         * describe the compilation being torn down. tests/memory/
         * leak_audit_gate.sh adds `requested` to LeakSanitizer's figure so
         * rooted retention stays measured rather than disappearing from the
         * per-line slope just because it is no longer a leak. */
        std::fprintf(stderr,
                     "eshkol-ast-strings: allocations=%llu requested=%llu "
                     "chunks=%llu reserved=%llu\n",
                     (unsigned long long)g_allocations.load(std::memory_order_relaxed),
                     (unsigned long long)g_bytes_requested.load(std::memory_order_relaxed),
                     (unsigned long long)g_live_chunks,
                     (unsigned long long)g_live_bytes_reserved);
        std::fflush(stderr);
    }
    if (!g_head) return;
    Chunk* c = g_head;
    g_head = nullptr;
    while (c) {
        Chunk* next = c->next;
        AST_STRINGS_UNPOISON(c->payload(), c->capacity);
        std::free(c);
        c = next;
    }
    g_live_chunks = 0;
    g_live_bytes_reserved = 0;
    ++g_teardowns;
}
