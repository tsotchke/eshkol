/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Native-stack headroom guard.
 *
 * Reading a datum nests as deeply as memory allows (see reader_limits.h), so
 * a pass that walks the resulting tree with native recursion can be handed a
 * tree far deeper than its own stack.  Such a pass asks this guard for
 * headroom on entry and reports an ordinary error when the answer is no.
 *
 * The guard measures what the thread's stack actually has left rather than
 * counting frames, so it holds for a worker thread with a small stack and for
 * the main thread with a large one, and it needs no per-pass frame-size
 * estimate.  A fixed margin (64 KB) is reserved below the bound so the pass
 * has room to build and report its diagnostic.
 *
 * The thread's stack extent is measured once per thread and kept: on Linux
 * glibc derives it by reading and parsing /proc/self/maps, which would
 * otherwise make the probe's cost grow with the program's address space.  The
 * extent is re-measured only when the current frame lies outside the kept one
 * (a different stack).
 *
 * The probe is always __builtin_frame_address(0) and never the address of a
 * local: under AddressSanitizer with detect_stack_use_after_return a local
 * lives on a heap fake stack, whose address is outside the thread's bounds and
 * would read as exhaustion on the very first frame.
 */
#ifndef ESHKOL_CORE_STACK_GUARD_H
#define ESHKOL_CORE_STACK_GUARD_H

#include <stddef.h>
#include <stdint.h>

#if !defined(_WIN32)
#include <pthread.h>
#endif

#ifdef __cplusplus
#define ESHKOL_STACK_GUARD_THREAD_LOCAL thread_local
#else
#define ESHKOL_STACK_GUARD_THREAD_LOCAL _Thread_local
#endif

/* Reserved below the stack bound, for the refusing pass to report itself. */
#define ESHKOL_STACK_SAFETY_MARGIN ((size_t)65536)

/**
 * @brief Whether the current thread has more than ESHKOL_STACK_SAFETY_MARGIN
 *        of stack left.
 *
 * @return 1 when there is headroom, or when the platform offers no way to
 *         ask (in which case the answer is conservatively yes); 0 when the
 *         thread is close enough to its bound that the caller must stop
 *         descending and report.
 */
static inline int eshkol_stack_headroom_ok(void) {
#ifdef _WIN32
    return 1;
#elif defined(__APPLE__)
    pthread_t self = pthread_self();
    void* stack_addr = pthread_get_stackaddr_np(self);
    size_t stack_size = pthread_get_stacksize_np(self);
    /* On macOS stack_addr is the TOP (highest address) of the stack. */
    const char* here = (const char*)__builtin_frame_address(0);
    size_t used = (size_t)((const char*)stack_addr - here);
    return (stack_size > used) &&
           ((stack_size - used) > ESHKOL_STACK_SAFETY_MARGIN) ? 1 : 0;
#elif defined(__linux__)
    static ESHKOL_STACK_GUARD_THREAD_LOCAL uintptr_t t_bottom = 0;
    static ESHKOL_STACK_GUARD_THREAD_LOCAL size_t t_extent = 0;
    static ESHKOL_STACK_GUARD_THREAD_LOCAL int t_measured = 0;
    const uintptr_t here = (uintptr_t)__builtin_frame_address(0);
    if (!t_measured || here < t_bottom || (here - t_bottom) > t_extent) {
        pthread_attr_t attr;
        if (pthread_getattr_np(pthread_self(), &attr) != 0) return 1;
        void* stack_addr = NULL;
        size_t stack_size = 0;
        pthread_attr_getstack(&attr, &stack_addr, &stack_size);
        pthread_attr_destroy(&attr);
        t_bottom = (uintptr_t)stack_addr;
        t_extent = stack_size;
        t_measured = 1;
    }
    /* On Linux the stack grows toward its BOTTOM (lowest address); the
     * distance to that bound is the space remaining. */
    return (here >= t_bottom && (here - t_bottom) <= t_extent &&
            (here - t_bottom) > ESHKOL_STACK_SAFETY_MARGIN) ? 1 : 0;
#else
    return 1;
#endif
}

#endif /* ESHKOL_CORE_STACK_GUARD_H */
