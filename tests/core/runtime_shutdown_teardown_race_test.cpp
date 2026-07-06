/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * ESH-0216 regression test: a graceful SIGTERM shutdown must not race a
 * still-running worker-thread pool against teardown of shared, arena-backed
 * state.
 *
 * Shape: mimics a long-running Eshkol-compiled "resident" — a main loop
 * driving the global parallel thread pool (as parallel-map/parallel-execute
 * do), a shutdown hook that frees a shared arena on graceful shutdown (the
 * documented way for a module to clean up its own state), and a real SIGTERM
 * delivered to this process, handled by the runtime's own hosted signal
 * handler (lib/core/runtime_signals_hosted.cpp).
 *
 * Before ESH-0216, eshkol_runtime_shutdown() ran shutdown hooks (and
 * restored signal handlers) without first stopping/joining the global
 * thread pool, so a worker still executing a task could dereference the
 * arena a hook had just freed — reported as
 * "[Eshkol] fatal signal: SIGSEGV" during/after the "graceful shutdown"
 * log line. This single run is inherently racy (the crash was intermittent
 * in production too); the driver script
 * tests/core/run_shutdown_teardown_race_cycles.sh runs this binary
 * ESH-0216's mandated 50 times to make the absence of the race a reliable
 * signal rather than a single lucky pass.
 */

#include <eshkol/backend/thread_pool.h>
#include <eshkol/core/runtime.h>

#include "../../lib/core/arena_memory.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

namespace {

std::atomic<bool> g_keep_spawning{true};
arena_t* g_shared_arena = nullptr;

int fail(const char* message) {
    std::fprintf(stderr, "runtime_shutdown_teardown_race_test: %s\n", message);
    return 1;
}

void* worker_task(void* /*arg*/) {
    // Workers share ONE arena, not their own thread-local one — this is the
    // shape of a resident module's session/faculty state shared across
    // parallel-map/parallel-execute calls.
    for (int i = 0; i < 200; ++i) {
        void* p = arena_allocate(g_shared_arena, 128);
        if (p) {
            std::memset(p, 0x5a, 128);
        }
    }
    return nullptr;
}

int free_shared_arena_hook(void* /*ctx*/, eshkol_shutdown_reason_t /*reason*/) {
    // Mimics a resident module (e.g. a session/memory faculty) freeing its
    // own shared state during graceful shutdown — exactly what
    // eshkol_register_shutdown_hook() exists for.
    arena_destroy(g_shared_arena);
    g_shared_arena = nullptr;
    return 0;
}

}  // namespace

int main() {
    if (eshkol_runtime_init() != 0) {
        return fail("runtime init failed");
    }

    g_shared_arena = arena_create_threadsafe(1024 * 1024);
    if (!g_shared_arena) {
        return fail("failed to create shared arena");
    }

    if (eshkol_register_shutdown_hook(free_shared_arena_hook, nullptr,
                                       "test-shared-arena-owner") == 0) {
        return fail("failed to register shutdown hook");
    }

    std::vector<std::thread> spawners;
    for (int s = 0; s < 4; ++s) {
        spawners.emplace_back([]() {
            // Stop dispatching *new* work the moment shutdown is observed —
            // exactly what any real Eshkol program's control flow does once
            // it branches into its own shutdown path — so the only race
            // left is against tasks already in flight in the pool, which is
            // precisely what eshkol_runtime_shutdown() must serialize
            // against.
            while (g_keep_spawning.load(std::memory_order_relaxed) &&
                   !eshkol_runtime_interrupt_flag_is_set()) {
                eshkol_thread_pool_t* pool = thread_pool_global();
                for (int i = 0; i < 32; ++i) {
                    thread_pool_submit_detached(pool, worker_task, nullptr);
                }
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        });
    }

    // Give the pool a moment to actually get busy before the signal lands.
    std::this_thread::sleep_for(std::chrono::milliseconds(15));

    if (std::raise(SIGTERM) != 0) {
        return fail("failed to raise SIGTERM");
    }

    int ticks = 0;
    while (!eshkol_runtime_interrupt_flag_is_set()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if (++ticks > 5000) {
            return fail("interrupt flag was never observed after SIGTERM");
        }
    }

    g_keep_spawning.store(false, std::memory_order_relaxed);
    for (auto& t : spawners) {
        t.join();
    }

    // The actual thing under test: this must not touch the arena the hook
    // frees from a still-running worker thread.
    eshkol_runtime_shutdown(ESHKOL_SHUTDOWN_REQUESTED);

    if (eshkol_runtime_get_state() != ESHKOL_RUNTIME_TERMINATED) {
        return fail("runtime did not terminate");
    }

    return 0;
}
