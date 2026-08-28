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
#include <cstring>

/**
 * @brief Report whether arena allocation poisoning is enabled for this process.
 *
 * Reads the ESHKOL_ARENA_POISON environment variable once (cached in an atomic
 * function-local static) and enables poisoning unless it is unset, empty, or
 * exactly "0" — the WHOLE value is compared, not just its first byte, so e.g.
 * "01" counts as set. This is the native engine's own accessor: it cannot see
 * the freestanding VM's vm_arena_poison_enabled() (lib/backend/vm_arena.h),
 * which is the bytecode VM's single reader of the same variable, but both
 * apply this identical rule so the two substrates never disagree about
 * whether a given value arms poisoning.
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
        cached = (env && env[0] && std::strcmp(env, "0") != 0) ? 1 : 0;
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

// ───────────────────────────────────────────────────────────────────────────
// SW-59: native-engine parity for the bytecode VM's region-pin stderr note.
//
// A continuation captured inside `with-region` pins every region open at
// capture time on both engines (heap_region_pin_all() in lib/backend/vm_core.c
// for the VM, eshkol_region_pin_all() in runtime_regions.cpp for native): the
// region's arena is promoted/leaked rather than freed, because the
// continuation's saved state may hold interior pointers into it. The VM has
// always announced this unconditionally on stderr (vm_evac_pin_notice(),
// lib/backend/vm_region_evac.c); native only ever logged it through
// eshkol_debug(), which is silent unless the process log level is raised to
// DEBUG. That asymmetry meant the same event was visible on one engine and
// invisible on the other by default.
//
// This gives native the same unconditional stderr note, gated by the same
// ESHKOL_VM_REGION_QUIET=1 the VM already honors (reused rather than given a
// native-specific name, since it already means "quiet about region pinning"
// regardless of which engine is asking) and read here rather than in
// runtime_regions.cpp because that file is ESHKOL_RUNTIME_CORE_SRC and must
// not call getenv()/fprintf() directly (see eshkol_arena_poison_enabled()
// above for the same split, applied to ESHKOL_ARENA_POISON).
// ───────────────────────────────────────────────────────────────────────────
extern "C" void eshkol_region_pin_notice(void) {
    static std::atomic<int> said{0};
    int expected = 0;
    if (!said.compare_exchange_strong(expected, 1, std::memory_order_relaxed)) return;
    const char* quiet = std::getenv("ESHKOL_VM_REGION_QUIET");
    if (quiet && quiet[0] && quiet[0] != '0') return;
    std::fprintf(stderr,
            "eshkol: note: a `with-region` body could not be reclaimed because "
            "a continuation was captured inside it; its arena is leaked instead "
            "of freed (the continuation's saved stack may hold interior "
            "pointers into it that this call site cannot see and therefore "
            "cannot promote). The answer is unaffected: nothing is dangling, "
            "but the memory is not returned for the rest of the process "
            "(lib/core/runtime_regions.cpp). Set ESHKOL_VM_REGION_QUIET=1 to "
            "silence this note.\n");
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
