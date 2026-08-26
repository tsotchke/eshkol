/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file node_identity.cpp
 * @brief Storage for the `NodeId -> SourceSpan` substrate (ADR-0000 Stage 1).
 *
 * The table is a fixed-size array of chunk pointers, each chunk holding
 * ::kChunkSpans spans. Allocation appends under a mutex; lookup is lock-free.
 * A chunk array of ::kMaxChunks entries times ::kChunkSpans spans is exactly
 * the 2^24 ids the tagged NodeId encoding can name, so the two limits agree
 * by construction rather than by comment.
 *
 * Chunks are never freed. They are process-lifetime for the same reason the
 * interned source-file table is: an id stamped during parsing must resolve at
 * codegen time, long after the parse unit that produced it is gone, and a
 * diagnostic emitted at the very end of a compile must still be able to name
 * the line it is about.
 */

#include <eshkol/frontend/node_identity.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace {

/** Spans per chunk. 4096 * 4096 == 2^24 == ESHKOL_NODE_ID_MAX_INDEX + 1. */
constexpr uint32_t kChunkSpans = 4096u;
/** Chunk-pointer slots. */
constexpr uint32_t kMaxChunks = 4096u;

static_assert((uint64_t)kChunkSpans * (uint64_t)kMaxChunks ==
                  (uint64_t)ESHKOL_NODE_ID_MAX_INDEX + 1u,
              "chunked span table must cover exactly the tagged NodeId index space");

std::atomic<eshkol_source_span_t*> g_chunks[kMaxChunks];

/** Number of spans published. Read with acquire, written with release. */
std::atomic<uint32_t> g_published{0};

/** Number of spans handed out (may briefly exceed g_published mid-append). */
uint32_t g_allocated = 0;

std::mutex g_alloc_mutex;

/* Coverage counters. Relaxed: these are a measurement, not a synchronizer,
 * and the gate reads them after the compile has finished. */
std::atomic<uint64_t> g_queried{0};
std::atomic<uint64_t> g_resolved{0};
std::atomic<uint64_t> g_with_location{0};
std::atomic<uint64_t> g_with_extent{0};

std::once_flag g_atexit_once;

/**
 * Arrange for the coverage line to be printed on the way out, whatever exit
 * path the process takes.
 *
 * Registered from the substrate rather than from a driver on purpose: every
 * binary that parses Eshkol (eshkol-run, the REPL, the server, a test rig)
 * is a measurement site, and a driver-side hook would have to be repeated in
 * each of them and would be silently missing from the next one.
 */
void ensure_report_at_exit();

bool env_flag_enabled(const char* name) {
    const char* raw = std::getenv(name);
    return raw && raw[0] != '\0' && std::strcmp(raw, "0") != 0 &&
           std::strcmp(raw, "false") != 0 && std::strcmp(raw, "FALSE") != 0;
}

/** Decode a candidate NodeId to a 0-based slot, or return false. */
inline bool decode(eshkol_node_id_t id, uint32_t* slot_out) {
    if (id == ESHKOL_NODE_ID_NONE) return false;
    if ((id >> 24) != ESHKOL_NODE_ID_TAG) return false;  /* garbage word */
    const uint32_t index = id & ESHKOL_NODE_ID_MAX_INDEX;
    if (index == 0) return false;                        /* index is 1-based */
    if (index > g_published.load(std::memory_order_acquire)) return false;
    *slot_out = index - 1u;
    return true;
}

/** Address of a published slot. Caller must have gone through decode(). */
inline eshkol_source_span_t* slot_ptr(uint32_t slot) {
    eshkol_source_span_t* chunk =
        g_chunks[slot / kChunkSpans].load(std::memory_order_acquire);
    return chunk ? &chunk[slot % kChunkSpans] : nullptr;
}

void ensure_report_at_exit() {
    if (!eshkol_node_identity_stats_enabled()) return;
    std::call_once(g_atexit_once, [] { std::atexit(eshkol_node_identity_report); });
}

}  // namespace

extern "C" eshkol_node_id_t eshkol_node_id_new(uint32_t file_id,
                                               uint32_t line,
                                               uint32_t column) {
    ensure_report_at_exit();
    std::lock_guard<std::mutex> lock(g_alloc_mutex);

    if (g_allocated >= ESHKOL_NODE_ID_MAX_INDEX) {
        /* Id space exhausted. Returning NONE degrades to "unknown location",
         * which is what the frontend already does for unstamped nodes — never
         * to a wrong location, and never to a crash. */
        return ESHKOL_NODE_ID_NONE;
    }

    const uint32_t slot = g_allocated;
    const uint32_t chunk_index = slot / kChunkSpans;
    eshkol_source_span_t* chunk = g_chunks[chunk_index].load(std::memory_order_relaxed);
    if (chunk == nullptr) {
        chunk = new (std::nothrow) eshkol_source_span_t[kChunkSpans];
        if (chunk == nullptr) return ESHKOL_NODE_ID_NONE;
        std::memset(chunk, 0, sizeof(eshkol_source_span_t) * kChunkSpans);
        g_chunks[chunk_index].store(chunk, std::memory_order_release);
    }

    eshkol_source_span_t& span = chunk[slot % kChunkSpans];
    span.file_id = file_id;
    span.start_line = line;
    span.start_column = column;
    span.end_line = line;
    span.end_column = column;
    span.has_extent = false;

    g_allocated = slot + 1u;
    /* Publish last: a reader that can see this index sees a complete span. */
    g_published.store(g_allocated, std::memory_order_release);

    return (eshkol_node_id_t)((ESHKOL_NODE_ID_TAG << 24) | (slot + 1u));
}

extern "C" void eshkol_node_span_set_extent(eshkol_node_id_t id,
                                            uint32_t end_line,
                                            uint32_t end_column) {
    uint32_t slot;
    if (!decode(id, &slot)) return;
    eshkol_source_span_t* span = slot_ptr(slot);
    if (!span) return;
    if (end_line == 0) return;
    /* Monotone: never move an extent backwards past the start. */
    if (end_line < span->start_line) return;
    if (end_line == span->start_line && end_column < span->start_column) return;
    span->end_line = end_line;
    span->end_column = end_column;
    span->has_extent = true;
}

extern "C" bool eshkol_node_span_lookup(eshkol_node_id_t id,
                                        eshkol_source_span_t* out) {
    uint32_t slot;
    if (!decode(id, &slot)) return false;
    const eshkol_source_span_t* span = slot_ptr(slot);
    if (!span) return false;
    if (out) *out = *span;
    return true;
}

extern "C" uint64_t eshkol_node_id_count(void) {
    return (uint64_t)g_published.load(std::memory_order_acquire);
}

extern "C" bool eshkol_node_identity_stats_enabled(void) {
    /* Resolved once: the gate sets the variable before exec, and a compile
     * must not change its measurement discipline halfway through. */
    static const bool enabled = env_flag_enabled("ESHKOL_NODE_IDENTITY_STATS");
    return enabled;
}

extern "C" void eshkol_node_identity_record_lookup(bool resolved,
                                                   bool has_location,
                                                   bool has_extent) {
    ensure_report_at_exit();
    g_queried.fetch_add(1, std::memory_order_relaxed);
    if (resolved) g_resolved.fetch_add(1, std::memory_order_relaxed);
    if (has_location) g_with_location.fetch_add(1, std::memory_order_relaxed);
    if (has_extent) g_with_extent.fetch_add(1, std::memory_order_relaxed);
}

extern "C" void eshkol_node_identity_stats(uint64_t* queried,
                                           uint64_t* resolved,
                                           uint64_t* with_location,
                                           uint64_t* with_extent,
                                           uint64_t* allocated) {
    if (queried) *queried = g_queried.load(std::memory_order_relaxed);
    if (resolved) *resolved = g_resolved.load(std::memory_order_relaxed);
    if (with_location) *with_location = g_with_location.load(std::memory_order_relaxed);
    if (with_extent) *with_extent = g_with_extent.load(std::memory_order_relaxed);
    if (allocated) *allocated = eshkol_node_id_count();
}

extern "C" void eshkol_node_identity_reset_stats(void) {
    g_queried.store(0, std::memory_order_relaxed);
    g_resolved.store(0, std::memory_order_relaxed);
    g_with_location.store(0, std::memory_order_relaxed);
    g_with_extent.store(0, std::memory_order_relaxed);
}

extern "C" void eshkol_node_identity_report(void) {
    if (!eshkol_node_identity_stats_enabled()) return;
    uint64_t queried = 0, resolved = 0, located = 0, extent = 0, allocated = 0;
    eshkol_node_identity_stats(&queried, &resolved, &located, &extent, &allocated);
    std::fprintf(stderr,
                 "eshkol-node-identity: allocated=%llu queried=%llu resolved=%llu "
                 "located=%llu extent=%llu\n",
                 (unsigned long long)allocated, (unsigned long long)queried,
                 (unsigned long long)resolved, (unsigned long long)located,
                 (unsigned long long)extent);
    std::fflush(stderr);
}
