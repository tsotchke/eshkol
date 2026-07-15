#ifndef ESHKOL_AGENT_NATIVE_MUTEX_H
#define ESHKOL_AGENT_NATIVE_MUTEX_H

/* Small internal static-mutex abstraction for process-global handle tables. */
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

typedef SRWLOCK eshkol_agent_mutex_t;
#define ESHKOL_AGENT_MUTEX_INITIALIZER SRWLOCK_INIT

static inline void eshkol_agent_mutex_lock(eshkol_agent_mutex_t* mutex) {
    AcquireSRWLockExclusive(mutex);
}

static inline void eshkol_agent_mutex_unlock(eshkol_agent_mutex_t* mutex) {
    ReleaseSRWLockExclusive(mutex);
}
#else
#include <pthread.h>

typedef pthread_mutex_t eshkol_agent_mutex_t;
#define ESHKOL_AGENT_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER

static inline void eshkol_agent_mutex_lock(eshkol_agent_mutex_t* mutex) {
    (void)pthread_mutex_lock(mutex);
}

static inline void eshkol_agent_mutex_unlock(eshkol_agent_mutex_t* mutex) {
    (void)pthread_mutex_unlock(mutex);
}
#endif

#endif
