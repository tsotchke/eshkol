/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted process stack-limit setup and native stack-overflow guard.
 */

#include <eshkol/eshkol.h>
#include <eshkol/core/resource_limits.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#ifndef _WIN32
#include <pthread.h>
#include <sys/resource.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#if defined(__APPLE__)
#include <pthread.h>
#endif

namespace {

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
        if (ok) return (uintptr_t)addr + size;
    }
    return 0;
#else
    return 0;
#endif
}

/**
 * @brief Bytes kept in reserve below the guard's floor.
 *
 * Two jobs. (1) The diagnostic path itself runs on the stack it is about to
 * declare exhausted, so it needs room for fprintf and its buffers. (2) The
 * check happens at function ENTRY, so between one check and the next the
 * stack pointer moves by that function's whole frame plus whatever a runtime
 * helper it calls consumes; the reserve absorbs that overshoot. A frame
 * larger than this margin can still jump the floor in one step — that case
 * lands on the fatal-signal handler, which is why both mechanisms exist.
 */
constexpr uint64_t kEshkolStackGuardMargin = 256ULL * 1024ULL;
constexpr uint64_t kEshkolDefaultStackSize = 512ULL * 1024ULL * 1024ULL;

/**
 * @brief Linux keeps an unmapped gap below a growable stack VMA
 * (`stack_guard_gap`, 256 pages by default) that no other mapping may use.
 * Growth stops one gap above the nearest mapping below, so the reachable
 * floor is that mapping's end plus the gap.
 */
constexpr uint64_t kEshkolLinuxStackGuardGap = 1024ULL * 1024ULL;

// Per-thread guard state. 0 = not yet probed, 1 = floor is valid,
// -1 = bounds unknown, guard disabled for this thread.
thread_local int t_stack_guard_state = 0;
thread_local uintptr_t t_stack_guard_region_low = 0;
thread_local uintptr_t t_stack_guard_region_high = 0;
thread_local uintptr_t t_stack_floor = 0;
thread_local uint64_t t_stack_size = 0;
thread_local uint64_t t_stack_usable = 0;
thread_local bool t_stack_is_program_thread = false;

#ifndef _WIN32
// Which thread ESHKOL_STACK_SIZE describes. Worker stacks are sized by
// ESHKOL_WORKER_STACK_BYTES at pthread_create time and their bounds are
// already exact, so the ESHKOL_STACK_SIZE ceiling must not be applied to
// them. Set by eshkol_init_stack_size(); if that never ran (an embedder that
// only links the runtime), no thread claims the ceiling and every thread is
// measured as-is.
pthread_t g_program_thread;
bool g_program_thread_known = false;
#endif

/**
 * @brief Parse the target stack size from ESHKOL_STACK_SIZE.
 *
 * @param fallback Value returned when the variable is unset, unparseable, or
 *                 below the documented 1 MiB minimum.
 * @return The configured size in bytes.
 */
uint64_t eshkol_stack_size_target(uint64_t fallback) {
    const char* env_val = std::getenv("ESHKOL_STACK_SIZE");
    if (!env_val) {
        return fallback;
    }
    size_t parsed = 0;
    if (!eshkol_parse_size(env_val, &parsed) || parsed < 1024ULL * 1024ULL) {
        return fallback;
    }
    return (uint64_t)parsed;
}

#ifndef _WIN32

#if defined(__linux__)
/**
 * @brief Find the reachable low address of the initial thread's stack.
 *
 * glibc's pthread_getattr_np() derives the main thread's extent from the
 * CURRENT RLIMIT_STACK, which eshkol_init_stack_size() has usually just
 * raised — and the kernel places the mmap region using the limit in force at
 * exec(), so a raised limit does not make the stack reachable any further
 * down. Reading /proc/self/maps gives the truth: the floor is whichever is
 * higher of (a) one guard gap above the nearest mapping below the stack and
 * (b) the rlimit-derived floor.
 *
 * @param[out] out_low  Reachable low address of the main stack.
 * @return true if the [stack] mapping was found and @p out_low was written.
 */
bool linux_main_stack_low(uintptr_t* out_low) {
    std::FILE* f = std::fopen("/proc/self/maps", "re");
    if (!f) {
        return false;
    }

    char line[512];
    uintptr_t stack_lo = 0, stack_hi = 0;
    uintptr_t prev_end_below = 0;
    bool found = false;

    // One pass: remember the highest mapping end that lies at or below the
    // stack mapping's start. /proc/self/maps is sorted ascending, so the last
    // such end seen before the [stack] line is the neighbour we want.
    while (std::fgets(line, sizeof(line), f)) {
        unsigned long long lo = 0, hi = 0;
        if (std::sscanf(line, "%llx-%llx", &lo, &hi) != 2) {
            continue;
        }
        if (std::strstr(line, "[stack]") != nullptr) {
            stack_lo = (uintptr_t)lo;
            stack_hi = (uintptr_t)hi;
            found = true;
            break;
        }
        prev_end_below = (uintptr_t)hi;
    }
    std::fclose(f);

    if (!found || stack_hi <= stack_lo) {
        return false;
    }

    uintptr_t floor_from_maps = 0;
    if (prev_end_below != 0 && prev_end_below <= stack_lo) {
        floor_from_maps = prev_end_below + (uintptr_t)kEshkolLinuxStackGuardGap;
    }

    uintptr_t floor_from_rlimit = 0;
    struct rlimit rl;
    if (getrlimit(RLIMIT_STACK, &rl) == 0 && rl.rlim_cur != RLIM_INFINITY) {
        if ((uintptr_t)rl.rlim_cur < stack_hi) {
            floor_from_rlimit = stack_hi - (uintptr_t)rl.rlim_cur;
        }
    }

    uintptr_t low = floor_from_maps > floor_from_rlimit ? floor_from_maps
                                                        : floor_from_rlimit;
    if (low == 0 || low >= stack_hi) {
        // Nothing below the stack and no finite limit: fall back to the
        // currently mapped extent rather than claiming unbounded headroom.
        low = stack_lo;
    }
    *out_low = low;
    return true;
}
#endif  // __linux__

/**
 * @brief Discover the calling thread's usable stack range.
 *
 * @param[out] out_low   Lowest address the stack can reach.
 * @param[out] out_size  Bytes between @p out_low and the stack base.
 * @return true if both were determined.
 */
bool current_thread_stack_bounds(uintptr_t* out_low, uint64_t* out_size) {
#if defined(__APPLE__)
    // Darwin reports the true mapped extent for both the main thread (sized
    // by the -stack_size link flag) and pthreads (sized at creation).
    void* base = pthread_get_stackaddr_np(pthread_self());  // high address
    size_t size = pthread_get_stacksize_np(pthread_self());
    if (base == nullptr || size == 0) {
        return false;
    }
    *out_low = (uintptr_t)base - (uintptr_t)size;
    *out_size = (uint64_t)size;
    return true;
#elif defined(__linux__)
    // The initial thread needs the /proc reading above; created threads have
    // exact bounds in their pthread attributes.
    pthread_attr_t attr;
    if (pthread_getattr_np(pthread_self(), &attr) != 0) {
        return false;
    }
    void* addr = nullptr;
    size_t size = 0;
    int rc = pthread_attr_getstack(&attr, &addr, &size);
    pthread_attr_destroy(&attr);
    if (rc != 0 || addr == nullptr || size == 0) {
        return false;
    }
    uintptr_t low = (uintptr_t)addr;
    uintptr_t high = low + (uintptr_t)size;

    uintptr_t main_low = 0;
    if (linux_main_stack_low(&main_low)) {
        // Only trust the /proc floor when this thread IS the one whose stack
        // /proc calls [stack]; a worker's mmap'd stack lives elsewhere.
        char probe;
        uintptr_t sp = (uintptr_t)&probe;
        if (sp >= main_low && sp < high && main_low > low) {
            low = main_low;
        }
    }
    if (high <= low) {
        return false;
    }
    *out_low = low;
    *out_size = (uint64_t)(high - low);
    return true;
#else
    (void)out_low;
    (void)out_size;
    return false;
#endif
}

#endif  // !_WIN32

/** @brief Probe this thread's stack bounds once and latch the guard floor. */
void stack_guard_init_thread(void) {
#ifdef _WIN32
    // Windows commits and guards thread stacks itself, and raises
    // EXCEPTION_STACK_OVERFLOW, which the runtime's unhandled-exception
    // filter already turns into a diagnostic. Nothing to probe.
    t_stack_guard_state = -1;
#else
    uintptr_t low = 0;
    uint64_t size = 0;
    if (!current_thread_stack_bounds(&low, &size) ||
        size <= kEshkolStackGuardMargin) {
        t_stack_guard_state = -1;
        return;
    }

    // ESHKOL_STACK_SIZE is a ceiling as well as a request. eshkol_init_stack_size()
    // raises RLIMIT_STACK toward it, but the kernel fixes the initial thread's
    // reachable extent at exec() and macOS fixes it at link time, so on most
    // systems the raise alone changes nothing and the documented variable would
    // be inert. Honouring it here is what makes it real in both directions: it
    // can only ever move the floor UP (never claim stack the thread does not
    // have), so a smaller value bounds recursion sooner and a larger one is
    // allowed exactly as far as the OS actually granted.
    t_stack_is_program_thread = g_program_thread_known &&
        pthread_equal(pthread_self(), g_program_thread);
    if (t_stack_is_program_thread) {
        // The default is a real guard target even when the inherited OS limit
        // is larger. This keeps the default and explicit 1G gate legs distinct.
        uint64_t requested = eshkol_stack_size_target(kEshkolDefaultStackSize);
        if (requested < size) {
            low = (uintptr_t)((low + (uintptr_t)size) - (uintptr_t)requested);
            size = requested;
            if (size <= kEshkolStackGuardMargin) {
                t_stack_guard_state = -1;
                return;
            }
        }
    }

    t_stack_size = size;
    t_stack_floor = low + (uintptr_t)kEshkolStackGuardMargin;
    const uintptr_t guard_region_bytes = (uintptr_t)kEshkolLinuxStackGuardGap;
    t_stack_guard_region_high = low;
    t_stack_guard_region_low = low > guard_region_bytes
        ? low - guard_region_bytes : 0;
    t_stack_usable = size - kEshkolStackGuardMargin;
    t_stack_guard_state = 1;
#endif
}

/** @brief Render a byte count as a whole number of MiB (at least 1). */
uint64_t as_mib(uint64_t bytes) {
    uint64_t mib = bytes / (1024ULL * 1024ULL);
    return mib == 0 ? 1 : mib;
}

/**
 * @brief Report native stack exhaustion and terminate.
 *
 * Deliberately fatal rather than a catchable Eshkol condition: the whole
 * point of this path is that only the guard margin is left, and unwinding
 * through an arbitrary depth of user frames is exactly the thing that cannot
 * be guaranteed there. Exits with ESHKOL_EXIT_LIMIT_STACK, the documented
 * status for a stack-limit breach, so callers see the same code they get
 * from ESHKOL_MAX_STACK.
 */
[[noreturn]] void stack_overflow_fatal(void) {
    std::fflush(stdout);
    const char* stack_env = t_stack_is_program_thread
        ? "ESHKOL_STACK_SIZE" : "ESHKOL_WORKER_STACK_BYTES";
    std::fprintf(stderr,
                 "eshkol: stack overflow: recursion depth exceeded the "
                 "%llu MiB stack (%s); use tail "
                 "recursion, or raise %s and the OS stack "
                 "limit to allow deeper recursion\n",
                 (unsigned long long)as_mib(t_stack_size), stack_env,
                 stack_env);
    std::fflush(stderr);
    _Exit(ESHKOL_EXIT_LIMIT_STACK);
}

}  // namespace

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
 *
 * Raising the limit does NOT retroactively enlarge the initial thread's
 * stack: the kernel lays out the process at exec() using the limit in force
 * then, so the guard measures the stack it actually has rather than the one
 * that was asked for.
 */
extern "C" void eshkol_init_stack_size(void) {
    // Hand the freestanding runtime core a platform probe so call/cc can
    // snapshot the stack and stay re-invocable after its frame returns.
    eshkol_set_stack_base_hook(&eshkol_hosted_stack_base);
#ifdef _WIN32
    // Windows thread stack sizing is handled at link/thread creation time.
    return;
#else
    const rlim_t default_stack = 512ULL * 1024 * 1024;  // 512MB
<<<<<<< HEAD
    const size_t floor_bytes = 1024ULL * 1024;           // 1MB
=======
    const size_t floor_bytes = 1024ULL * 1024ULL;        // 1MB
>>>>>>> eadbc0bb (fix(runtime): stack overflow in user recursion reports a diagnostic instead of SIGILL (ESH-0101, SW-81))
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
    // Keep the documented parser and warning behavior for malformed values.

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

    // Probe after the limit is settled so the program thread's floor reflects it.
    g_program_thread = pthread_self();
    g_program_thread_known = true;
    t_stack_guard_state = 0;
    stack_guard_init_thread();
#endif
}

/** @copydoc eshkol_stack_guard_check */
extern "C" void eshkol_stack_guard_check(void) {
    char probe;
    if (t_stack_guard_state == 0) {
        stack_guard_init_thread();
    }
    if (t_stack_guard_state < 0) {
        return;
    }
    if ((uintptr_t)&probe <= t_stack_floor) {
        stack_overflow_fatal();
    }
}

/**
 * @brief Signal-safe test for a fault address in this thread's stack guard.
 *
 * The returned value is meaningful only for SIGSEGV/SIGBUS fault addresses.
 * It reads latched thread-local integers and performs no allocation, locking,
 * I/O, or other work that could recurse on an exhausted stack.
 */
extern "C" bool eshkol_stack_guard_fault_in_region(const void* fault_address) {
    if (t_stack_guard_state != 1 || fault_address == nullptr) {
        return false;
    }
    uintptr_t address = (uintptr_t)fault_address;
    return address >= t_stack_guard_region_low &&
           address < t_stack_guard_region_high;
}

/** @copydoc eshkol_stack_guard_headroom */
extern "C" uint64_t eshkol_stack_guard_headroom(void) {
    char probe;
    if (t_stack_guard_state == 0) {
        stack_guard_init_thread();
    }
    if (t_stack_guard_state < 0) {
        return 0;
    }
    uintptr_t sp = (uintptr_t)&probe;
    return sp > t_stack_floor ? (uint64_t)(sp - t_stack_floor) : 0;
}
