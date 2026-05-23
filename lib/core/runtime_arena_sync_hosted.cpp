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

void eshkol_arena_global_once(void (*init)(void)) {
#ifdef _WIN32
    static std::once_flag once;
    std::call_once(once, init);
#else
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, init);
#endif
}

void eshkol_hash_table_lock(void) {
#ifdef _WIN32
    g_hash_table_mutex.lock();
#else
    pthread_mutex_lock(&g_hash_table_mutex);
#endif
}

void eshkol_hash_table_unlock(void) {
#ifdef _WIN32
    g_hash_table_mutex.unlock();
#else
    pthread_mutex_unlock(&g_hash_table_mutex);
#endif
}
