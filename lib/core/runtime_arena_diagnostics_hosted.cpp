/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena diagnostics policy.
 */

#include <atomic>
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
