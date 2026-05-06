/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted arena threading and process-tuning support.
 */

#include "arena_memory.h"
#include "../../inc/eshkol/logger.h"
#include "../../inc/eshkol/eshkol.h"

#include <cstdio>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#include <mutex>
#else
#include <pthread.h>
#include <sys/resource.h>
#endif

namespace eshkol::runtime {

void destroy_arena_threading_state(arena_t* arena) {
    if (!arena || !arena->thread_safe || !arena->mutex) {
        return;
    }

#ifdef _WIN32
    delete static_cast<std::mutex*>(arena->mutex);
#else
    pthread_mutex_destroy(static_cast<pthread_mutex_t*>(arena->mutex));
    free(arena->mutex);
#endif

    arena->mutex = nullptr;
    arena->thread_safe = false;
}

} // namespace eshkol::runtime

extern "C" void eshkol_init_stack_size() {
#ifdef _WIN32
    return;
#else
    const rlim_t default_stack = 512ULL * 1024 * 1024;
    rlim_t target = default_stack;

    const char* env_val = getenv("ESHKOL_STACK_SIZE");
    if (env_val) {
        char* end = nullptr;
        unsigned long long parsed = strtoull(env_val, &end, 0);
        if (end != env_val && parsed >= 1024 * 1024) {
            target = static_cast<rlim_t>(parsed);
        }
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

#if !defined(_WIN32)
__attribute__((constructor)) static void eshkol_init_stack_size_before_main() {
    eshkol_init_stack_size();
}
#endif

void debug_print_ad_mode(const char* context) {
    fprintf(stderr, "[AD_MODE_DEBUG] %s: __ad_mode_active = %s\n",
            context,
            __ad_mode_active ? "TRUE" : "FALSE");
}

void debug_print_ptr(const char* context, void* ptr) {
    fprintf(stderr, "[PTR_DEBUG] %s: ptr = %p (0x%llx)\n",
            context,
            ptr,
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(ptr)));
}

arena_t* arena_create_threadsafe(size_t default_block_size) {
    arena_t* arena = arena_create(default_block_size);
    if (!arena) {
        return nullptr;
    }

#ifdef _WIN32
    std::mutex* mutex = new (std::nothrow) std::mutex();
    if (!mutex) {
        eshkol_error("Failed to allocate mutex for thread-safe arena");
        arena_destroy(arena);
        return nullptr;
    }
#else
    pthread_mutex_t* mutex = static_cast<pthread_mutex_t*>(malloc(sizeof(pthread_mutex_t)));
    if (!mutex) {
        eshkol_error("Failed to allocate mutex for thread-safe arena");
        arena_destroy(arena);
        return nullptr;
    }

    if (pthread_mutex_init(mutex, nullptr) != 0) {
        eshkol_error("Failed to initialize mutex for thread-safe arena");
        free(mutex);
        arena_destroy(arena);
        return nullptr;
    }
#endif

    arena->mutex = mutex;
    arena->thread_safe = true;

    eshkol_debug("Created thread-safe arena with default block size %zu", default_block_size);
    return arena;
}

void arena_lock(arena_t* arena) {
    if (!arena || !arena->thread_safe || !arena->mutex) {
        return;
    }

#ifdef _WIN32
    static_cast<std::mutex*>(arena->mutex)->lock();
#else
    pthread_mutex_lock(static_cast<pthread_mutex_t*>(arena->mutex));
#endif
}

void arena_unlock(arena_t* arena) {
    if (!arena || !arena->thread_safe || !arena->mutex) {
        return;
    }

#ifdef _WIN32
    static_cast<std::mutex*>(arena->mutex)->unlock();
#else
    pthread_mutex_unlock(static_cast<pthread_mutex_t*>(arena->mutex));
#endif
}

__attribute__((weak)) arena_t* __global_arena = nullptr;

#ifdef _WIN32
static std::once_flag global_arena_once;
#else
static pthread_once_t global_arena_once = PTHREAD_ONCE_INIT;
#endif

static void init_global_arena_internal() {
    if (__global_arena) {
        return;
    }
    __global_arena = arena_create_threadsafe(65536);
    if (!__global_arena) {
        eshkol_error("Failed to create global arena");
    }
}

arena_t* get_global_arena() {
    if (__global_arena) {
        return __global_arena;
    }
#ifdef _WIN32
    std::call_once(global_arena_once, init_global_arena_internal);
#else
    pthread_once(&global_arena_once, init_global_arena_internal);
#endif
    return __global_arena;
}

static thread_local arena_t* __thread_local_arena = nullptr;

arena_t* arena_get_thread_local(void) {
    if (__thread_local_arena) {
        return __thread_local_arena;
    }
    return get_global_arena();
}

arena_t* arena_create_thread_local(size_t size_hint) {
    if (__thread_local_arena) {
        return __thread_local_arena;
    }

    const size_t block_size = size_hint > 0 ? size_hint : (1024 * 1024);
    __thread_local_arena = arena_create(block_size);
    return __thread_local_arena;
}

void arena_merge_to_parent(arena_t* dest, arena_t* src) {
    if (!dest || !src || dest == src) {
        return;
    }

    if (dest->thread_safe) {
        arena_lock(dest);
    }

    if (src->current_block) {
        if (dest->current_block) {
            arena_block_t* dest_tail = dest->current_block;
            while (dest_tail->next) {
                dest_tail = dest_tail->next;
            }
            dest_tail->next = src->current_block;
        } else {
            dest->current_block = src->current_block;
        }

        dest->total_allocated += src->total_allocated;
        src->current_block = nullptr;
        src->total_allocated = 0;
    }

    if (dest->thread_safe) {
        arena_unlock(dest);
    }
}

extern "C" int eshkol_thread_pool_is_worker(void) __attribute__((weak));

int arena_is_worker_thread(void) {
    if (eshkol_thread_pool_is_worker) {
        return eshkol_thread_pool_is_worker();
    }
    return 0;
}

void eshkol_thread_init_worker(size_t arena_size_hint) {
    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;

    __outer_ad_node_storage = nullptr;
    __outer_ad_node_to_inner = nullptr;
    __outer_grad_accumulator = nullptr;
    __inner_var_node_ptr = nullptr;
    __gradient_x_degree = 0;

    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;

    (void)arena_create_thread_local(arena_size_hint);
}

void eshkol_thread_shutdown_worker(void) {
    if (__thread_local_arena) {
        arena_destroy(__thread_local_arena);
        __thread_local_arena = nullptr;
    }

    for (size_t i = 0; i < ESHKOL_ARENA_MAX_TAPE_DEPTH; ++i) {
        __ad_tape_stack[i] = nullptr;
        __outer_ad_node_stack[i] = nullptr;
    }
    __ad_tape_depth = 0;
    __outer_ad_node_depth = 0;

    __outer_ad_node_storage = nullptr;
    __outer_ad_node_to_inner = nullptr;
    __outer_grad_accumulator = nullptr;
    __inner_var_node_ptr = nullptr;
    __gradient_x_degree = 0;

    for (size_t i = 0; i < MAX_REGION_DEPTH; ++i) {
        __region_stack[i] = nullptr;
    }
    __region_stack_depth = 0;
}
