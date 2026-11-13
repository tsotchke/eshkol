/*
 * threads.c — POSIX thread synchronization primitives for Eshkol.
 *
 * Exposes mutex + condvar to Eshkol code via the FFI (extern ...) path,
 * mirroring the agent_crypto.c pattern.  Mutex/condvar storage is
 * malloc'd; the caller is responsible for the matching *-destroy! call.
 *
 * Thread creation itself (make-thread / thread-join) is intentionally
 * not exposed here — calling Eshkol closures from a freshly-created
 * thread requires the closure-call ABI (per-thread arena, AD tape,
 * region stack), which is non-trivial.  For parallel work, use the
 * existing thread-pool primitives: (future thunk), (force fut),
 * (parallel-execute thunk1 thunk2 ...).  Mutex/condvar are useful
 * regardless to coordinate worker-pool tasks and to protect shared
 * state inside parallel-map workers.
 *
 * Counting semaphores and reusable barriers (lib/core/threads.esk) are
 * built on this same mutex + condvar pair; the only primitive they add
 * here is the bounded wait eshkol_condvar_timedwait().  Condition
 * variables measure that bound on the monotonic clock: Linux and other
 * POSIX hosts bind CLOCK_MONOTONIC at creation, macOS uses the relative
 * wait pthread_cond_timedwait_relative_np(), so a wall-clock step never
 * lengthens or shortens a timed wait.  Unnamed POSIX semaphores (sem_t)
 * are not used: macOS does not implement sem_init().
 *
 * Windows stub: returns NULL / non-zero error from every entry.  Real
 * Windows implementation (SRWLOCK + CONDITION_VARIABLE) ships with
 * the cross-platform threading work in v1.4-platform.
 */

#include <stdlib.h>
#include <string.h>

#include <stdint.h>

#ifndef _WIN32
#include <pthread.h>
#include <errno.h>
#include <time.h>
#endif

/* ----------------------------------------------------------------------
 * Mutex
 * ---------------------------------------------------------------------- */

/* (make-mutex) → ptr  (NULL on allocation failure) */
void* eshkol_mutex_create(void) {
#ifndef _WIN32
    pthread_mutex_t* m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (!m) return NULL;
    pthread_mutexattr_t attr;
    if (pthread_mutexattr_init(&attr) != 0) { free(m); return NULL; }
    /* PTHREAD_MUTEX_RECURSIVE so an Eshkol closure that re-enters
     * (with-mutex m ...) on the same thread doesn't deadlock.  This
     * matches the SRFI 18 mutex semantics most users expect from a
     * high-level language. */
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    int rc = pthread_mutex_init(m, &attr);
    pthread_mutexattr_destroy(&attr);
    if (rc != 0) { free(m); return NULL; }
    return (void*)m;
#else
    return NULL;
#endif
}

/* (mutex-lock! ptr) → int (0=success, errno on failure) */
int eshkol_mutex_lock(void* mutex_ptr) {
#ifndef _WIN32
    if (!mutex_ptr) return EINVAL;
    return pthread_mutex_lock((pthread_mutex_t*)mutex_ptr);
#else
    (void)mutex_ptr;
    return -1;
#endif
}

/* (mutex-trylock! ptr) → int (0=acquired, EBUSY=already locked, other=error) */
int eshkol_mutex_trylock(void* mutex_ptr) {
#ifndef _WIN32
    if (!mutex_ptr) return EINVAL;
    return pthread_mutex_trylock((pthread_mutex_t*)mutex_ptr);
#else
    (void)mutex_ptr;
    return -1;
#endif
}

/* (mutex-unlock! ptr) → int (0=success, errno on failure) */
int eshkol_mutex_unlock(void* mutex_ptr) {
#ifndef _WIN32
    if (!mutex_ptr) return EINVAL;
    return pthread_mutex_unlock((pthread_mutex_t*)mutex_ptr);
#else
    (void)mutex_ptr;
    return -1;
#endif
}

/* (mutex-destroy! ptr) → int (0=success).  Frees the underlying storage;
 * caller MUST NOT use the pointer afterwards. */
int eshkol_mutex_destroy(void* mutex_ptr) {
#ifndef _WIN32
    if (!mutex_ptr) return EINVAL;
    pthread_mutex_t* m = (pthread_mutex_t*)mutex_ptr;
    int rc = pthread_mutex_destroy(m);
    free(m);
    return rc;
#else
    (void)mutex_ptr;
    return -1;
#endif
}

/* ----------------------------------------------------------------------
 * Condition variable
 * ---------------------------------------------------------------------- */

/* (make-condvar) → ptr (NULL on allocation failure) */
void* eshkol_condvar_create(void) {
#ifndef _WIN32
    pthread_cond_t* c = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
    if (!c) return NULL;
    pthread_condattr_t attr;
    if (pthread_condattr_init(&attr) != 0) { free(c); return NULL; }
#if !defined(__APPLE__)
    /* Timed waits measure their bound on the monotonic clock (see
     * eshkol_condvar_timedwait).  macOS has no pthread_condattr_setclock
     * and uses the relative-wait entry point instead. */
    if (pthread_condattr_setclock(&attr, CLOCK_MONOTONIC) != 0) {
        pthread_condattr_destroy(&attr);
        free(c);
        return NULL;
    }
#endif
    int rc = pthread_cond_init(c, &attr);
    pthread_condattr_destroy(&attr);
    if (rc != 0) { free(c); return NULL; }
    return (void*)c;
#else
    return NULL;
#endif
}

/* (condvar-wait! cond-ptr mutex-ptr) → int (0=success).
 * Atomically releases the mutex, blocks until signalled, reacquires
 * the mutex before returning.  The mutex MUST be held by the calling
 * thread on entry — POSIX requirement. */
int eshkol_condvar_wait(void* cond_ptr, void* mutex_ptr) {
#ifndef _WIN32
    if (!cond_ptr || !mutex_ptr) return EINVAL;
    return pthread_cond_wait((pthread_cond_t*)cond_ptr,
                             (pthread_mutex_t*)mutex_ptr);
#else
    (void)cond_ptr; (void)mutex_ptr;
    return -1;
#endif
}

/* (condvar-wait-timeout! cond-ptr mutex-ptr ms) → int.
 * Like eshkol_condvar_wait, but returns ETIMEDOUT once timeout_ms
 * milliseconds of monotonic time pass without a wakeup.  A negative
 * timeout is treated as zero.  Returns 0 when woken (which, as with any
 * condition variable, may be spurious: callers re-check their predicate),
 * ETIMEDOUT on expiry, or another errno value on failure. */
int eshkol_condvar_timedwait(void* cond_ptr, void* mutex_ptr, int64_t timeout_ms) {
#ifndef _WIN32
    if (!cond_ptr || !mutex_ptr) return EINVAL;
    if (timeout_ms < 0) timeout_ms = 0;
    struct timespec ts;
#if defined(__APPLE__)
    ts.tv_sec = (time_t)(timeout_ms / 1000);
    ts.tv_nsec = (long)((timeout_ms % 1000) * 1000000L);
    return pthread_cond_timedwait_relative_np((pthread_cond_t*)cond_ptr,
                                              (pthread_mutex_t*)mutex_ptr, &ts);
#else
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return errno;
    ts.tv_sec += (time_t)(timeout_ms / 1000);
    ts.tv_nsec += (long)((timeout_ms % 1000) * 1000000L);
    if (ts.tv_nsec >= 1000000000L) {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000L;
    }
    return pthread_cond_timedwait((pthread_cond_t*)cond_ptr,
                                  (pthread_mutex_t*)mutex_ptr, &ts);
#endif
#else
    (void)cond_ptr; (void)mutex_ptr; (void)timeout_ms;
    return -1;
#endif
}

/* The errno value eshkol_condvar_timedwait returns on expiry (ETIMEDOUT
 * differs between platforms: 60 on macOS, 110 on Linux). */
int eshkol_condvar_timeout_code(void) {
#ifndef _WIN32
    return ETIMEDOUT;
#else
    return -2;
#endif
}

/* (condvar-signal! ptr) → int.  Wakes one waiter. */
int eshkol_condvar_signal(void* cond_ptr) {
#ifndef _WIN32
    if (!cond_ptr) return EINVAL;
    return pthread_cond_signal((pthread_cond_t*)cond_ptr);
#else
    (void)cond_ptr;
    return -1;
#endif
}

/* (condvar-broadcast! ptr) → int.  Wakes all waiters. */
int eshkol_condvar_broadcast(void* cond_ptr) {
#ifndef _WIN32
    if (!cond_ptr) return EINVAL;
    return pthread_cond_broadcast((pthread_cond_t*)cond_ptr);
#else
    (void)cond_ptr;
    return -1;
#endif
}

/* (condvar-destroy! ptr) → int.  Frees the underlying storage. */
int eshkol_condvar_destroy(void* cond_ptr) {
#ifndef _WIN32
    if (!cond_ptr) return EINVAL;
    pthread_cond_t* c = (pthread_cond_t*)cond_ptr;
    int rc = pthread_cond_destroy(c);
    free(c);
    return rc;
#else
    (void)cond_ptr;
    return -1;
#endif
}
