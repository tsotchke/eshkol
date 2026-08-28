/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted process stack-limit setup.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/resource_limits.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>

#ifndef _WIN32
#include <sys/resource.h>
#include <pthread.h>
#else
#include <windows.h>
#endif

/**
 * @brief Highest address of the calling thread's stack, or 0 if unknown.
 *
 * Queried per call rather than cached, so a continuation captured on a worker
 * thread gets that thread's geometry. A 0 return leaves `call/cc` escape-only
 * for that capture instead of copying a stack region we cannot bound.
 */
static uintptr_t eshkol_hosted_stack_base(void) {
#if defined(_WIN32)
    ULONG_PTR low = 0, high = 0;
    GetCurrentThreadStackLimits(&low, &high);
    return (uintptr_t)high;
#elif defined(__APPLE__)
    return (uintptr_t)pthread_get_stackaddr_np(pthread_self());
#elif defined(__linux__)
    pthread_attr_t attr;
    void* addr = nullptr;
    size_t size = 0;
    if (pthread_getattr_np(pthread_self(), &attr) == 0) {
        int ok = (pthread_attr_getstack(&attr, &addr, &size) == 0);
        pthread_attr_destroy(&attr);
        // The stack grows down, so the base is the far end of the mapping.
        if (ok) return (uintptr_t)addr + size;
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief Raise the process's stack rlimit for hosted (non-Windows) builds.
 *
 * No-op on Windows (thread stacks are sized at creation time instead). On
 * other platforms, reads a target size from the ESHKOL_STACK_SIZE
 * environment variable — accepting a bare byte count or a value with a
 * K/M/G (or KiB/MiB/GiB) suffix via eshkol_parse_size(), the same parser
 * every other ESHKOL_* size variable uses — falling back to a 512MB default
 * when unset or below the 1MB floor, then raises RLIMIT_STACK's soft limit
 * to that target via getrlimit()/setrlimit(), clamped to the hard limit if
 * one is set. Only ever increases the current soft limit; never lowers it.
 *
 * A value that fails to parse at all (empty, non-numeric, unrecognized
 * suffix, or trailing garbage after a valid one) is reported to stderr
 * naming the variable and the offending value, then falls back to the
 * default; a value that parses but is below the 1MB floor falls back
 * silently, per the documented floor.
 */
extern "C" void eshkol_init_stack_size(void) {
    // Hand the freestanding runtime core a platform probe so `call/cc` can
    // snapshot the stack and stay re-invocable after its frame returns. Runs
    // from the program entry the codegen emits, before any user code.
    eshkol_set_stack_base_hook(&eshkol_hosted_stack_base);
#ifdef _WIN32
    // Windows thread stack sizing is handled at link/thread creation time.
    return;
#else
    const rlim_t default_stack = 512ULL * 1024 * 1024;  // 512MB
    const size_t floor_bytes = 1024ULL * 1024;           // 1MB
    rlim_t target = default_stack;

    const char* env_val = std::getenv("ESHKOL_STACK_SIZE");
    if (env_val) {
        size_t parsed = 0;
        if (!eshkol_parse_size(env_val, &parsed)) {
            std::fprintf(stderr,
                "eshkol: warning: ESHKOL_STACK_SIZE=\"%s\" is not a valid "
                "size (expected a byte count or a value with a K/M/G or "
                "KiB/MiB/GiB suffix); using default stack size\n",
                env_val);
        } else if (parsed >= floor_bytes) {
            target = (rlim_t)parsed;
        }
        // parsed < floor_bytes: too small to be useful, silently keep the
        // default (the documented 1MB floor).
    }

    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0) {
        if (rl.rlim_cur < target) {
            rl.rlim_cur = target;
            if (rl.rlim_max != RLIM_INFINITY && rl.rlim_max < target) {
                rl.rlim_cur = rl.rlim_max;
            }
            setrlimit(RLIMIT_STACK, &rl);
        }
    }
#endif
}
