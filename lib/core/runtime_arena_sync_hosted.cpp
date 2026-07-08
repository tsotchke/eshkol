/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena synchronization primitives.
 */

#ifdef _WIN32
#include <mutex>
#include <new>
#else
#include <pthread.h>
#include <cstdlib>
#endif

namespace {
#ifdef _WIN32
std::mutex g_hash_table_mutex;
#else
pthread_mutex_t g_hash_table_mutex = PTHREAD_MUTEX_INITIALIZER;
#endif
}

/**
 * @brief Allocate and initialise a platform mutex for a thread-safe arena.
 *
 * On Windows, heap-allocates a std::mutex. Elsewhere, heap-allocates and
 * pthread_mutex_init()s a pthread_mutex_t, freeing it again if init fails.
 *
 * @return Opaque mutex handle (to be passed to the other eshkol_arena_mutex_*
 *         functions), or NULL on allocation/initialisation failure.
 */
void* eshkol_arena_mutex_create(void) {
#ifdef _WIN32
    return new (std::nothrow) std::mutex();
#else
    auto* mutex = static_cast<pthread_mutex_t*>(std::malloc(sizeof(pthread_mutex_t)));
    if (!mutex) {
        return nullptr;
    }
    if (pthread_mutex_init(mutex, nullptr) != 0) {
        std::free(mutex);
        return nullptr;
    }
    return mutex;
#endif
}

/**
 * @brief Destroy and free a mutex previously created by eshkol_arena_mutex_create().
 *
 * No-op if `mutex` is NULL.
 */
void eshkol_arena_mutex_destroy(void* mutex) {
    if (!mutex) {
        return;
    }
#ifdef _WIN32
    delete static_cast<std::mutex*>(mutex);
#else
    pthread_mutex_destroy(static_cast<pthread_mutex_t*>(mutex));
    std::free(mutex);
#endif
}

/**
 * @brief Lock a mutex previously created by eshkol_arena_mutex_create(). No-op if `mutex` is NULL.
 */
void eshkol_arena_mutex_lock(void* mutex) {
    if (!mutex) {
        return;
    }
#ifdef _WIN32
    static_cast<std::mutex*>(mutex)->lock();
#else
    pthread_mutex_lock(static_cast<pthread_mutex_t*>(mutex));
#endif
}

/**
 * @brief Unlock a mutex previously created by eshkol_arena_mutex_create(). No-op if `mutex` is NULL.
 */
void eshkol_arena_mutex_unlock(void* mutex) {
    if (!mutex) {
        return;
    }
#ifdef _WIN32
    static_cast<std::mutex*>(mutex)->unlock();
#else
    pthread_mutex_unlock(static_cast<pthread_mutex_t*>(mutex));
#endif
}

/**
 * @brief Run `init` exactly once for the lifetime of the process, across all callers/threads.
 *
 * Backed by std::call_once (Windows) or pthread_once (elsewhere) with a
 * function-local static once-control object.
 *
 * @param init  Zero-argument function to invoke exactly once.
 */
void eshkol_arena_global_once(void (*init)(void)) {
#ifdef _WIN32
    static std::once_flag once;
    std::call_once(once, init);
#else
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init);
#endif
}

/** @brief Lock the process-wide global hash-table mutex. */
void eshkol_hash_table_lock(void) {
#ifdef _WIN32
    g_hash_table_mutex.lock();
#else
    pthread_mutex_lock(&g_hash_table_mutex);
#endif
}

/** @brief Unlock the process-wide global hash-table mutex. */
void eshkol_hash_table_unlock(void) {
#ifdef _WIN32
    g_hash_table_mutex.unlock();
#else
    pthread_mutex_unlock(&g_hash_table_mutex);
#endif
}
