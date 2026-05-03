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
 * Windows stub: returns NULL / non-zero error from every entry.  Real
 * Windows implementation (SRWLOCK + CONDITION_VARIABLE) ships with
 * the cross-platform threading work in v1.4-platform.
 */

#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <pthread.h>
#include <errno.h>
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
    if (pthread_cond_init(c, NULL) != 0) { free(c); return NULL; }
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
