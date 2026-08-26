/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena diagnostics policy.
 */

#include "arena_memory.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

/**
 * @brief Report whether arena allocation poisoning is enabled for this process.
 *
 * Reads the ESHKOL_ARENA_POISON environment variable once (cached in an atomic
 * function-local static) and enables poisoning unless it is unset, empty, or
 * "0".
 *
 * The cache is an atomic tri-state (-1 = uncomputed, 0/1 = resolved) because
 * arena allocation runs concurrently on pool workers: the plain-int cache
 * previously raced (benign, same value, but a real TSan-visible data race) when
 * multiple workers first touched it simultaneously. Relaxed ordering suffices —
 * the computed value is idempotent, so a redundant recompute by a racing thread
 * writes the identical result.
 *
 * @return Non-zero if arena poisoning is enabled, zero otherwise.
 */
extern "C" int eshkol_arena_poison_enabled(void) {
    static std::atomic<int> poison_enabled{-1};
    int cached = poison_enabled.load(std::memory_order_relaxed);
    if (cached < 0) {
        const char* env = std::getenv("ESHKOL_ARENA_POISON");
        cached = (env && env[0] && env[0] != '0') ? 1 : 0;
        poison_enabled.store(cached, std::memory_order_relaxed);
    }
    return cached;
}

// ───────────────────────────────────────────────────────────────────────────
// SW-57: exact, load-independent retention probe for the process-global arena.
//
// Peak RSS was the only flat-memory signal this repo's gates had, and it is a
// poor one to gate on. `maximum resident set size` is a high-water mark of
// INSTANTANEOUS residency, so on a loaded host the memory compressor evicts
// pages and the recorded maximum comes back LOWER than what the process
// actually retains — measuring the SW-57 repro on a 24-core box at load average
// ~200 gave 97 MB and 193 MB for the same binary on consecutive runs. A leak
// gate built on that number is quietest exactly when CI is busiest.
//
// The arena's own `total_allocated` is the honest number: deterministic to the
// byte, unaffected by system load, and it is what the flat-memory claim is
// really about — bytes the process can never hand back. ESHKOL_ARENA_REPORT=1
// prints it once at exit, on stderr, in a single grep-able line, which is what
// tests/memory/resident_longrun_flat_gate.sh gates on.
//
// Diagnostic only: off unless the variable is exactly "1", and it never changes
// allocation behavior. Registered from a static constructor rather than from
// arena creation because the arena lives in the freestanding-clean core, which
// may not call getenv(); reading __global_arena at exit is safe because nothing
// destroys the process-global arena.
// ───────────────────────────────────────────────────────────────────────────
static void eshkol_arena_report_at_exit(void) {
    arena_t* arena = __global_arena;
    std::fprintf(stderr, "[eshkol-arena] global_total_allocated_bytes=%zu\n",
                 arena ? arena_get_total_memory(arena) : (size_t)0);
}

namespace {
struct EshkolArenaReportInstaller {
    EshkolArenaReportInstaller() {
        const char* env = std::getenv("ESHKOL_ARENA_REPORT");
        if (env && env[0] == '1' && env[1] == '\0') {
            std::atexit(eshkol_arena_report_at_exit);
        }
    }
};
const EshkolArenaReportInstaller g_eshkol_arena_report_installer;
}  // namespace
