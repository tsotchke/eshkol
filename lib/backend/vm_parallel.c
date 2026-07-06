/**
 * @file vm_parallel.c
 * @brief Thread pool and parallel primitives for the Eshkol bytecode VM.
 *
 * Provides a pthread-backed task queue and parallel higher-order primitives
 * (parallel-map, parallel-filter, parallel-for-each, parallel-reduce).
 *
 * Workers execute read-only closure bytecode in isolated VM/arena snapshots and
 * publish returned values back to the main VM heap. Closures that may mutate
 * shared VM state use a serialized fallback through the main VM.
 *
 * Native call IDs: 620-639
 *
 * Copyright (C) Tsotchke Corporation. MIT License.
 */

#ifndef VM_PARALLEL_C_INCLUDED
#define VM_PARALLEL_C_INCLUDED

#include "vm_arena.h"
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdatomic.h>
#ifdef _WIN32
#include <windows.h>
#endif

/* ── Task ── */

typedef enum {
    TASK_PENDING,
    TASK_RUNNING,
    TASK_DONE
} VmTaskState;

typedef struct {
    void (*fn)(void* arg, void* result);
    void*       arg;
    void*       result;
    VmTaskState state;
} VmTask;

/* ── Thread Pool ── */

#define VM_POOL_MAX_QUEUE 4096

typedef struct {
    pthread_t*      threads;
    int             n_threads;

    /* Circular task queue (mutex-protected) */
    VmTask          queue[VM_POOL_MAX_QUEUE];
    int             head;           /* next slot to dequeue */
    int             tail;           /* next slot to enqueue */
    int             count;          /* tasks in queue */

    pthread_mutex_t mutex;
    pthread_cond_t  work_avail;     /* signalled when task enqueued or shutdown */
    pthread_cond_t  all_done;       /* signalled when active_tasks drops to 0 */

    atomic_int      active_tasks;   /* tasks currently being executed */
    int             shutdown;       /* 1 = workers should exit */

    VmRegionStack*  thread_arenas;  /* one per worker, for arena-based allocation */
} VmThreadPool;

/* Single global pool */
static VmThreadPool* g_pool = NULL;
static pthread_mutex_t g_pool_init_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_pool_atexit_registered = 0;

/* ESH-0216: exported (not static) so lib/core/runtime_lifecycle_hosted.cpp can
 * stop/join this pool deterministically as part of eshkol_runtime_shutdown(),
 * before shutdown hooks or any other teardown step runs. Still also
 * registered via atexit() below as a fallback for processes that never call
 * eshkol_runtime_shutdown() explicitly; both call sites are idempotent (see
 * definition) so invoking it from either or both paths is safe. */
void eshkol_vm_parallel_shutdown_global(void);

/*******************************************************************************
 * Worker Thread
 ******************************************************************************/

static void* vm_pool_worker(void* arg) {
    VmThreadPool* pool = (VmThreadPool*)arg;

    for (;;) {
        VmTask task;
        int got_task = 0;

        /* ── Dequeue ── */
        pthread_mutex_lock(&pool->mutex);
        while (pool->count == 0 && !pool->shutdown) {
            pthread_cond_wait(&pool->work_avail, &pool->mutex);
        }
        if (pool->shutdown && pool->count == 0) {
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        if (pool->count > 0) {
            task = pool->queue[pool->head];
            pool->queue[pool->head].state = TASK_RUNNING;
            pool->head = (pool->head + 1) % VM_POOL_MAX_QUEUE;
            pool->count--;
            atomic_fetch_add(&pool->active_tasks, 1);
            got_task = 1;
        }
        pthread_mutex_unlock(&pool->mutex);

        /* ── Execute ── */
        if (got_task) {
            task.fn(task.arg, task.result);

            int remaining = atomic_fetch_sub(&pool->active_tasks, 1) - 1;
            if (remaining == 0) {
                pthread_mutex_lock(&pool->mutex);
                if (pool->count == 0 && atomic_load(&pool->active_tasks) == 0) {
                    pthread_cond_broadcast(&pool->all_done);
                }
                pthread_mutex_unlock(&pool->mutex);
            }
        }
    }
    return NULL;
}

/*******************************************************************************
 * Pool Lifecycle
 ******************************************************************************/

/* 620: vm_pool_init — create thread pool with n_threads workers */
static VmThreadPool* vm_pool_init(int n_threads) {
    if (n_threads <= 0) {
        /* Auto-detect: number of CPUs, clamped to [1, 64] */
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        long n = (long)sysinfo.dwNumberOfProcessors;
#else
        long n = sysconf(_SC_NPROCESSORS_ONLN);
#endif
        if (n <= 0) n = 1;
        if (n > 64) n = 64;
        n_threads = (int)n;
    }

    VmThreadPool* pool = (VmThreadPool*)calloc(1, sizeof(VmThreadPool));
    if (!pool) {
        fprintf(stderr, "ERROR: vm_pool_init: alloc failed\n");
        return NULL;
    }

    pool->n_threads = n_threads;
    pool->head = 0;
    pool->tail = 0;
    pool->count = 0;
    pool->shutdown = 0;
    atomic_store(&pool->active_tasks, 0);

    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->work_avail, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->all_done, NULL) != 0) {
        pthread_cond_destroy(&pool->work_avail);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }

    /* Per-thread arenas */
    pool->thread_arenas = (VmRegionStack*)calloc(n_threads, sizeof(VmRegionStack));
    if (!pool->thread_arenas) {
        pthread_cond_destroy(&pool->all_done);
        pthread_cond_destroy(&pool->work_avail);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    for (int i = 0; i < n_threads; i++) {
        vm_region_stack_init(&pool->thread_arenas[i]);
    }

    /* Spawn workers */
    pool->threads = (pthread_t*)calloc(n_threads, sizeof(pthread_t));
    if (!pool->threads) {
        for (int i = 0; i < n_threads; i++)
            vm_region_stack_destroy(&pool->thread_arenas[i]);
        free(pool->thread_arenas);
        pthread_cond_destroy(&pool->all_done);
        pthread_cond_destroy(&pool->work_avail);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    for (int i = 0; i < n_threads; i++) {
        if (pthread_create(&pool->threads[i], NULL, vm_pool_worker, pool) != 0) {
            fprintf(stderr, "ERROR: vm_pool_init: pthread_create failed for thread %d\n", i);
            /* Shut down threads we already started */
            pthread_mutex_lock(&pool->mutex);
            pool->shutdown = 1;
            pthread_cond_broadcast(&pool->work_avail);
            pthread_mutex_unlock(&pool->mutex);
            for (int j = 0; j < i; j++)
                pthread_join(pool->threads[j], NULL);
            for (int j = 0; j < n_threads; j++)
                vm_region_stack_destroy(&pool->thread_arenas[j]);
            free(pool->thread_arenas);
            free(pool->threads);
            pthread_cond_destroy(&pool->all_done);
            pthread_cond_destroy(&pool->work_avail);
            pthread_mutex_destroy(&pool->mutex);
            free(pool);
            return NULL;
        }
    }

    g_pool = pool;
    return pool;
}

/* 621: vm_pool_shutdown — signal shutdown, join all workers, free resources */
static void vm_pool_shutdown(VmThreadPool* pool) {
    if (!pool) return;

    /* Signal shutdown */
    pthread_mutex_lock(&pool->mutex);
    pool->shutdown = 1;
    pthread_cond_broadcast(&pool->work_avail);
    pthread_mutex_unlock(&pool->mutex);

    /* Join all worker threads */
    for (int i = 0; i < pool->n_threads; i++) {
        pthread_join(pool->threads[i], NULL);
    }

    /* Destroy per-thread arenas */
    for (int i = 0; i < pool->n_threads; i++) {
        vm_region_stack_destroy(&pool->thread_arenas[i]);
    }

    /* Free resources */
    free(pool->thread_arenas);
    free(pool->threads);
    pthread_cond_destroy(&pool->all_done);
    pthread_cond_destroy(&pool->work_avail);
    pthread_mutex_destroy(&pool->mutex);

    if (g_pool == pool) g_pool = NULL;
    free(pool);
}

static VmThreadPool* vm_parallel_ensure_pool(void) {
    pthread_mutex_lock(&g_pool_init_mutex);
    if (!g_pool) {
        g_pool = vm_pool_init(0);
        if (g_pool && !g_pool_atexit_registered) {
            atexit(eshkol_vm_parallel_shutdown_global);
            g_pool_atexit_registered = 1;
        }
    }
    VmThreadPool* pool = g_pool;
    pthread_mutex_unlock(&g_pool_init_mutex);
    return pool;
}

/* 621b: eshkol_vm_parallel_shutdown_global — idempotent; safe to call even if
 * no pool was ever created (g_pool is NULL) and safe to call more than once
 * (the second call finds g_pool already NULL and is a no-op). */
void eshkol_vm_parallel_shutdown_global(void) {
    pthread_mutex_lock(&g_pool_init_mutex);
    VmThreadPool* pool = g_pool;
    g_pool = NULL;
    pthread_mutex_unlock(&g_pool_init_mutex);
    if (pool) vm_pool_shutdown(pool);
}

/* 622: vm_pool_submit — enqueue a task, returns 0 on success */
static int vm_pool_submit(VmThreadPool* pool, void (*fn)(void* arg, void* result),
                          void* arg, void* result) {
    if (!pool || !fn) return -1;

    pthread_mutex_lock(&pool->mutex);
    if (pool->count >= VM_POOL_MAX_QUEUE) {
        pthread_mutex_unlock(&pool->mutex);
        fprintf(stderr, "ERROR: vm_pool_submit: queue full (%d tasks)\n", VM_POOL_MAX_QUEUE);
        return -1;
    }

    VmTask* t = &pool->queue[pool->tail];
    t->fn = fn;
    t->arg = arg;
    t->result = result;
    t->state = TASK_PENDING;
    pool->tail = (pool->tail + 1) % VM_POOL_MAX_QUEUE;
    pool->count++;

    pthread_cond_signal(&pool->work_avail);
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

/* 623: vm_pool_wait_all — block until all submitted tasks complete */
static void vm_pool_wait_all(VmThreadPool* pool) {
    if (!pool) return;

    pthread_mutex_lock(&pool->mutex);
    while (pool->count > 0 || atomic_load(&pool->active_tasks) > 0) {
        pthread_cond_wait(&pool->all_done, &pool->mutex);
    }
    pthread_mutex_unlock(&pool->mutex);
}

/* 624: vm_pool_thread_count — return number of worker threads */
static int vm_pool_thread_count(const VmThreadPool* pool) {
    return pool ? pool->n_threads : 0;
}

/* 625: vm_pool_pending_count — return number of pending tasks */
static int vm_pool_pending_count(VmThreadPool* pool) {
    if (!pool) return 0;
    pthread_mutex_lock(&pool->mutex);
    int c = pool->count;
    pthread_mutex_unlock(&pool->mutex);
    return c;
}

/*******************************************************************************
 * Worker VM Context
 *
 * Eshkol's VM heap is arena-backed and indexes heap objects by small integer
 * handles. Worker execution therefore cannot share Heap.next_free or arena state
 * with the main VM. For read-only closure bytecode, each worker receives:
 *
 *   - shared immutable code/constants,
 *   - a private heap object table and private arena,
 *   - cloned reachable heap objects at the same indexes as the main VM,
 *   - new worker-local allocations starting at main_heap.next_free.
 *
 * Returned worker-local heap objects are deep-copied into the main heap under
 * g_heap_mutex. Mutating or otherwise unsafe closures fall back to the old
 * serialized bridge.
 ******************************************************************************/

typedef struct VmWorkerContext {
    int unused;
} VmWorkerContext;

static pthread_mutex_t g_heap_mutex = PTHREAD_MUTEX_INITIALIZER;

static int vm_value_has_heap_index(Value v) {
    switch ((int)v.type) {
        case VAL_PAIR:
        case VAL_CLOSURE:
        case VAL_STRING:
        case VAL_VECTOR:
        case VAL_CONTINUATION:
        case VAL_TENSOR:
        case VAL_KB:
        case VAL_COMPLEX:
        case VAL_RATIONAL:
        case VAL_BIGNUM:
        case VAL_DUAL:
        case VAL_FACTOR_GRAPH:
        case VAL_WORKSPACE:
        case VAL_SUBST:
        case VAL_HASH:
        case VAL_BYTEVECTOR:
        case VAL_PARAMETER_OBJ:
        case VAL_AD_TAPE:
        case VAL_ERROR_OBJ:
        case VAL_MANIFOLD:
        case VAL_PORT:
        case VAL_HYPER_DUAL:
        case VAL_RIEMANNIAN_ADAM_STATE:
        case VAL_FUTURE:
            return 1;
        default:
            return 0;
    }
}

static VmBignum* vm_bignum_clone_to(VmRegionStack* rs, const VmBignum* src) {
    if (!src) return NULL;
    VmBignum* dst = (VmBignum*)vm_alloc(rs, sizeof(VmBignum));
    if (!dst) return NULL;
    *dst = *src;
    if (src->capacity > 0 && src->limbs) {
        dst->limbs = (uint32_t*)vm_alloc(rs, (size_t)src->capacity * sizeof(uint32_t));
        if (!dst->limbs) return NULL;
        memcpy(dst->limbs, src->limbs, (size_t)src->n_limbs * sizeof(uint32_t));
    } else {
        dst->limbs = NULL;
    }
    return dst;
}

static int vm_clone_value_graph(VM* worker, VM* main_vm, Value v,
                                int32_t base_next, int depth);

static int vm_clone_object_at(VM* worker, VM* main_vm, int32_t idx,
                              int32_t base_next, int depth) {
    if (idx < 0 || idx >= base_next || depth > 256) return 0;
    if (worker->heap.objects[idx]) return 1;

    HeapObject* src = main_vm->heap.objects[idx];
    if (!src) return 0;

    HeapObject* dst = (HeapObject*)vm_alloc(&worker->heap.regions, sizeof(HeapObject));
    if (!dst) return 0;
    *dst = *src;
    worker->heap.objects[idx] = dst;

    switch (src->type) {
        case HEAP_CONS:
            return vm_clone_value_graph(worker, main_vm, src->cons.car, base_next, depth + 1) &&
                   vm_clone_value_graph(worker, main_vm, src->cons.cdr, base_next, depth + 1);

        case HEAP_CLOSURE:
            for (int i = 0; i < src->closure.n_upvalues && i < 16; i++) {
                if (!vm_clone_value_graph(worker, main_vm, src->closure.upvalues[i],
                                          base_next, depth + 1)) {
                    return 0;
                }
            }
            return 1;

        case HEAP_STRING: {
            VmString* s = (VmString*)src->opaque.ptr;
            dst->opaque.ptr = s ? vm_string_new(&worker->heap.regions, s->data, s->byte_len) : NULL;
            return !s || dst->opaque.ptr != NULL;
        }

        case HEAP_VECTOR: {
            VmVector* sv = (VmVector*)src->opaque.ptr;
            if (!sv) { dst->opaque.ptr = NULL; return 1; }
            VmVector* dv = (VmVector*)vm_alloc(&worker->heap.regions, sizeof(VmVector));
            if (!dv) return 0;
            dv->len = sv->len;
            dv->cap = sv->cap;
            dv->items = NULL;
            if (sv->len > 0) {
                dv->items = (Value*)vm_alloc(&worker->heap.regions,
                                             (size_t)sv->len * sizeof(Value));
                if (!dv->items) return 0;
                memcpy(dv->items, sv->items, (size_t)sv->len * sizeof(Value));
                for (int i = 0; i < sv->len; i++) {
                    if (!vm_clone_value_graph(worker, main_vm, dv->items[i],
                                              base_next, depth + 1)) {
                        return 0;
                    }
                }
            }
            dst->opaque.ptr = dv;
            return 1;
        }

        case HEAP_COMPLEX: {
            VmComplex* src_z = (VmComplex*)src->opaque.ptr;
            VmComplex* dst_z = src_z ? (VmComplex*)vm_alloc(&worker->heap.regions, sizeof(VmComplex)) : NULL;
            if (src_z && !dst_z) return 0;
            if (src_z) *dst_z = *src_z;
            dst->opaque.ptr = dst_z;
            return 1;
        }

        case HEAP_RATIONAL: {
            VmRational* src_r = (VmRational*)src->opaque.ptr;
            VmRational* dst_r = src_r ? (VmRational*)vm_alloc(&worker->heap.regions, sizeof(VmRational)) : NULL;
            if (src_r && !dst_r) return 0;
            if (src_r) *dst_r = *src_r;
            dst->opaque.ptr = dst_r;
            return 1;
        }

        case HEAP_BIGNUM:
            dst->opaque.ptr = vm_bignum_clone_to(&worker->heap.regions,
                                                 (VmBignum*)src->opaque.ptr);
            return !src->opaque.ptr || dst->opaque.ptr != NULL;

        case HEAP_DUAL: {
            VmDual* src_d = (VmDual*)src->opaque.ptr;
            VmDual* dst_d = src_d ? (VmDual*)vm_alloc(&worker->heap.regions, sizeof(VmDual)) : NULL;
            if (src_d && !dst_d) return 0;
            if (src_d) *dst_d = *src_d;
            dst->opaque.ptr = dst_d;
            return 1;
        }

        case HEAP_TENSOR:
            dst->opaque.ptr = vm_tensor_copy(&worker->heap.regions, (VmTensor*)src->opaque.ptr);
            return !src->opaque.ptr || dst->opaque.ptr != NULL;

        case HEAP_BYTEVECTOR: {
            VmBytevector* bv = (VmBytevector*)src->opaque.ptr;
            dst->opaque.ptr = bv ? vm_bv_copy(&worker->heap.regions, bv, 0, bv->len) : NULL;
            return !bv || dst->opaque.ptr != NULL;
        }

        case HEAP_HYPER_DUAL: {
            VmHyperDual* src_h = (VmHyperDual*)src->opaque.ptr;
            VmHyperDual* dst_h = src_h ? (VmHyperDual*)vm_alloc(&worker->heap.regions, sizeof(VmHyperDual)) : NULL;
            if (src_h && !dst_h) return 0;
            if (src_h) *dst_h = *src_h;
            dst->opaque.ptr = dst_h;
            return 1;
        }

        default:
            return 1;
    }
}

static int vm_clone_value_graph(VM* worker, VM* main_vm, Value v,
                                int32_t base_next, int depth) {
    if (!vm_value_has_heap_index(v)) return 1;
    return vm_clone_object_at(worker, main_vm, v.as.ptr, base_next, depth);
}

static int vm_native_is_worker_safe(int fid) {
    if ((fid >= 20 && fid <= 38) || (fid >= 40 && fid <= 51) || fid == 55 ||
        (fid >= 71 && fid <= 73) || (fid >= 137 && fid <= 139) ||
        (fid >= 160 && fid <= 166) || (fid >= 186 && fid <= 189) ||
        fid == 235 || (fid >= 300 && fid <= 319) || (fid >= 330 && fid <= 350) ||
        (fid >= 353 && fid <= 389) ||
        (fid >= 720 && fid <= 722) || (fid >= 1680 && fid <= 1699) ||
        (fid >= 1900 && fid <= 1921)) {
        return 1;
    }
    return 0;
}

static int vm_opcode_is_worker_safe(Instr instr) {
    switch (instr.op) {
        case OP_NOP:
        case OP_CONST:
        case OP_NIL:
        case OP_TRUE:
        case OP_FALSE:
        case OP_POP:
        case OP_DUP:
        case OP_ADD:
        case OP_SUB:
        case OP_MUL:
        case OP_DIV:
        case OP_MOD:
        case OP_NEG:
        case OP_ABS:
        case OP_EQ:
        case OP_LT:
        case OP_GT:
        case OP_LE:
        case OP_GE:
        case OP_NOT:
        case OP_GET_LOCAL:
        case OP_SET_LOCAL:
        case OP_GET_UPVALUE:
        case OP_CLOSURE:
        case OP_RETURN:
        case OP_JUMP:
        case OP_JUMP_IF_FALSE:
        case OP_LOOP:
        case OP_CONS:
        case OP_CAR:
        case OP_CDR:
        case OP_NULL_P:
        case OP_VEC_CREATE:
        case OP_VEC_REF:
        case OP_VEC_LEN:
        case OP_STR_REF:
        case OP_STR_LEN:
        case OP_PAIR_P:
        case OP_NUM_P:
        case OP_STR_P:
        case OP_BOOL_P:
        case OP_PROC_P:
        case OP_VEC_P:
        case OP_POPN:
        case OP_PACK_REST:
        case OP_VOID:
            return 1;
        case OP_NATIVE_CALL:
            return vm_native_is_worker_safe(instr.operand);
        default:
            return 0;
    }
}

static int vm_closure_is_worker_safe(VM* vm, Value closure) {
    if (closure.type != VAL_CLOSURE || closure.as.ptr < 0 ||
        closure.as.ptr >= vm->heap.next_free) {
        return 0;
    }
    HeapObject* cl = vm->heap.objects[closure.as.ptr];
    if (!cl || cl->type != HEAP_CLOSURE) return 0;

    int start_pc = cl->closure.func_pc;
    if (start_pc < 0 || start_pc >= vm->code_len) return 0;

    uint8_t* seen = (uint8_t*)calloc((size_t)vm->code_len, sizeof(uint8_t));
    int* work = (int*)calloc((size_t)vm->code_len, sizeof(int));
    if (!seen || !work) {
        free(seen);
        free(work);
        return 0;
    }

    int safe = 1;
    int top = 0;
    work[top++] = start_pc;

    while (top > 0 && safe) {
        int pc = work[--top];
        if (pc < 0 || pc >= vm->code_len) { safe = 0; break; }
        if (seen[pc]) continue;
        seen[pc] = 1;

        Instr instr = vm->code[pc];
        if (!vm_opcode_is_worker_safe(instr)) { safe = 0; break; }

        switch (instr.op) {
            case OP_RETURN:
                break;
            case OP_JUMP:
            case OP_LOOP:
                if (top < vm->code_len) work[top++] = instr.operand;
                else safe = 0;
                break;
            case OP_JUMP_IF_FALSE:
                if (top + 2 <= vm->code_len) {
                    work[top++] = instr.operand;
                    work[top++] = pc + 1;
                } else safe = 0;
                break;
            default:
                if (pc + 1 < vm->code_len) {
                    if (top < vm->code_len) work[top++] = pc + 1;
                    else safe = 0;
                }
                break;
        }
    }

    free(seen);
    free(work);
    return safe;
}

static int vm_publish_value_locked(VM* main_vm, VM* worker, Value in,
                                   int32_t base_next, int32_t* remap,
                                   int remap_len, Value* out, int depth);

static int vm_publish_object_locked(VM* main_vm, VM* worker, Value in,
                                    int32_t base_next, int32_t* remap,
                                    int remap_len, Value* out, int depth) {
    int32_t idx = in.as.ptr;
    if (idx < 0 || idx >= worker->heap.next_free || depth > 256) return 0;
    if (idx < base_next) { *out = in; return 1; }

    int map_idx = idx - base_next;
    if (map_idx < 0 || map_idx >= remap_len) return 0;
    if (remap[map_idx] >= 0) {
        *out = in;
        out->as.ptr = remap[map_idx];
        return 1;
    }

    HeapObject* src = worker->heap.objects[idx];
    if (!src) return 0;

    int32_t dst_idx = heap_alloc(&main_vm->heap);
    if (dst_idx < 0) return 0;
    remap[map_idx] = dst_idx;
    HeapObject* dst = main_vm->heap.objects[dst_idx];
    memset(dst, 0, sizeof(HeapObject));
    dst->type = src->type;

    *out = in;
    out->as.ptr = dst_idx;

    switch (src->type) {
        case HEAP_CONS:
            return vm_publish_value_locked(main_vm, worker, src->cons.car,
                                           base_next, remap, remap_len,
                                           &dst->cons.car, depth + 1) &&
                   vm_publish_value_locked(main_vm, worker, src->cons.cdr,
                                           base_next, remap, remap_len,
                                           &dst->cons.cdr, depth + 1);

        case HEAP_CLOSURE:
            dst->closure.func_pc = src->closure.func_pc;
            dst->closure.n_upvalues = src->closure.n_upvalues;
            for (int i = 0; i < src->closure.n_upvalues && i < 16; i++) {
                if (!vm_publish_value_locked(main_vm, worker, src->closure.upvalues[i],
                                             base_next, remap, remap_len,
                                             &dst->closure.upvalues[i], depth + 1)) {
                    return 0;
                }
            }
            return 1;

        case HEAP_STRING: {
            VmString* s = (VmString*)src->opaque.ptr;
            dst->opaque.ptr = s ? vm_string_new(&main_vm->heap.regions, s->data, s->byte_len) : NULL;
            return !s || dst->opaque.ptr != NULL;
        }

        case HEAP_VECTOR: {
            VmVector* sv = (VmVector*)src->opaque.ptr;
            if (!sv) { dst->opaque.ptr = NULL; return 1; }
            VmVector* dv = (VmVector*)vm_alloc(&main_vm->heap.regions, sizeof(VmVector));
            if (!dv) return 0;
            dv->len = sv->len;
            dv->cap = sv->cap;
            dv->items = NULL;
            if (sv->len > 0) {
                dv->items = (Value*)vm_alloc(&main_vm->heap.regions,
                                             (size_t)sv->len * sizeof(Value));
                if (!dv->items) return 0;
                for (int i = 0; i < sv->len; i++) {
                    if (!vm_publish_value_locked(main_vm, worker, sv->items[i],
                                                 base_next, remap, remap_len,
                                                 &dv->items[i], depth + 1)) {
                        return 0;
                    }
                }
            }
            dst->opaque.ptr = dv;
            return 1;
        }

        case HEAP_COMPLEX: {
            VmComplex* src_z = (VmComplex*)src->opaque.ptr;
            VmComplex* dst_z = src_z ? (VmComplex*)vm_alloc(&main_vm->heap.regions, sizeof(VmComplex)) : NULL;
            if (src_z && !dst_z) return 0;
            if (src_z) *dst_z = *src_z;
            dst->opaque.ptr = dst_z;
            return 1;
        }

        case HEAP_RATIONAL: {
            VmRational* src_r = (VmRational*)src->opaque.ptr;
            VmRational* dst_r = src_r ? (VmRational*)vm_alloc(&main_vm->heap.regions, sizeof(VmRational)) : NULL;
            if (src_r && !dst_r) return 0;
            if (src_r) *dst_r = *src_r;
            dst->opaque.ptr = dst_r;
            return 1;
        }

        case HEAP_BIGNUM:
            dst->opaque.ptr = vm_bignum_clone_to(&main_vm->heap.regions,
                                                 (VmBignum*)src->opaque.ptr);
            return !src->opaque.ptr || dst->opaque.ptr != NULL;

        case HEAP_DUAL: {
            VmDual* src_d = (VmDual*)src->opaque.ptr;
            VmDual* dst_d = src_d ? (VmDual*)vm_alloc(&main_vm->heap.regions, sizeof(VmDual)) : NULL;
            if (src_d && !dst_d) return 0;
            if (src_d) *dst_d = *src_d;
            dst->opaque.ptr = dst_d;
            return 1;
        }

        case HEAP_TENSOR:
            dst->opaque.ptr = vm_tensor_copy(&main_vm->heap.regions, (VmTensor*)src->opaque.ptr);
            return !src->opaque.ptr || dst->opaque.ptr != NULL;

        case HEAP_BYTEVECTOR: {
            VmBytevector* bv = (VmBytevector*)src->opaque.ptr;
            dst->opaque.ptr = bv ? vm_bv_copy(&main_vm->heap.regions, bv, 0, bv->len) : NULL;
            return !bv || dst->opaque.ptr != NULL;
        }

        case HEAP_HYPER_DUAL: {
            VmHyperDual* src_h = (VmHyperDual*)src->opaque.ptr;
            VmHyperDual* dst_h = src_h ? (VmHyperDual*)vm_alloc(&main_vm->heap.regions, sizeof(VmHyperDual)) : NULL;
            if (src_h && !dst_h) return 0;
            if (src_h) *dst_h = *src_h;
            dst->opaque.ptr = dst_h;
            return 1;
        }

        default:
            return 0;
    }
}

static int vm_publish_value_locked(VM* main_vm, VM* worker, Value in,
                                   int32_t base_next, int32_t* remap,
                                   int remap_len, Value* out, int depth) {
    if (!vm_value_has_heap_index(in)) {
        *out = in;
        return 1;
    }
    return vm_publish_object_locked(main_vm, worker, in, base_next,
                                    remap, remap_len, out, depth);
}

static int vm_call_closure_from_native_isolated(VM* main_vm, Value closure,
                                                Value* args, int argc,
                                                Value* out) {
    VM worker;
    vm_init(&worker);
    worker.code = main_vm->code;
    worker.code_len = main_vm->code_len;
    worker.n_constants = main_vm->n_constants;
    memcpy(worker.constants, main_vm->constants,
           (size_t)main_vm->n_constants * sizeof(Value));
    memset(worker.ad_node_map, -1, sizeof(worker.ad_node_map));

    int ok = 1;
    int32_t base_next = 0;

    pthread_mutex_lock(&g_heap_mutex);
    base_next = main_vm->heap.next_free;
    worker.heap.next_free = base_next;
    for (int i = 0; i < worker.n_constants && ok; i++) {
        ok = vm_clone_value_graph(&worker, main_vm, worker.constants[i],
                                  base_next, 0);
    }
    ok = ok && vm_clone_value_graph(&worker, main_vm, closure, base_next, 0);
    for (int i = 0; i < argc && ok; i++) {
        ok = vm_clone_value_graph(&worker, main_vm, args[i], base_next, 0);
    }
    pthread_mutex_unlock(&g_heap_mutex);

    if (!ok) {
        heap_destroy(&worker.heap);
        return 0;
    }

    Value worker_result = vm_call_closure_from_native(&worker, closure, args, argc);
    int remap_len = worker.heap.next_free - base_next;
    int32_t* remap = NULL;
    if (remap_len > 0) {
        remap = (int32_t*)malloc((size_t)remap_len * sizeof(int32_t));
        if (!remap) {
            heap_destroy(&worker.heap);
            return 0;
        }
        for (int i = 0; i < remap_len; i++) remap[i] = -1;
    }

    pthread_mutex_lock(&g_heap_mutex);
    ok = vm_publish_value_locked(main_vm, &worker, worker_result, base_next,
                                 remap, remap_len, out, 0);
    pthread_mutex_unlock(&g_heap_mutex);

    free(remap);
    heap_destroy(&worker.heap);
    return ok;
}

/* Execute a closure call in a worker context.
 * The closure's bytecode address and captured values are in the closure struct.
 * Returns the result as a Value. */
static Value vm_worker_call_closure(VmWorkerContext* wctx, VM* main_vm,
                                    Value closure, Value* args, int nargs) {
    (void)wctx;
    if (vm_closure_is_worker_safe(main_vm, closure)) {
        Value isolated_result = NIL_VAL;
        if (vm_call_closure_from_native_isolated(main_vm, closure, args, nargs,
                                                 &isolated_result)) {
            return isolated_result;
        }
    }

    Value result;
    pthread_mutex_lock(&g_heap_mutex);
    result = vm_call_closure_from_native(main_vm, closure, args, nargs);
    pthread_mutex_unlock(&g_heap_mutex);
    return result;
}

/*******************************************************************************
 * Parallel Map Task — applies closure to one element
 ******************************************************************************/

typedef struct {
    VM*     main_vm;
    Value   closure;
    Value   input;
    Value   output;
} VmParMapTask;

static void vm_parmap_task_fn(void* arg, void* result) {
    VmParMapTask* task = (VmParMapTask*)arg;
    (void)result;
    task->output = vm_worker_call_closure(NULL, task->main_vm,
                                          task->closure, &task->input, 1);
}

typedef struct {
    VM*     main_vm;
    Value   closure;
    Value   output;
} VmParThunkTask;

static void vm_parthunk_task_fn(void* arg, void* result) {
    VmParThunkTask* task = (VmParThunkTask*)arg;
    (void)result;
    task->output = vm_worker_call_closure(NULL, task->main_vm,
                                          task->closure, NULL, 0);
}

/*******************************************************************************
 * Future Handles
 ******************************************************************************/

typedef struct {
    VM*             main_vm;
    Value           thunk_or_value;
    Value           result;
    int             ready;
    pthread_mutex_t mutex;
    pthread_cond_t  done;
} VmFuture;

static VmFuture* vm_future_create(VM* vm, Value thunk_or_value) {
    VmFuture* fut = (VmFuture*)vm_alloc(&vm->heap.regions, sizeof(VmFuture));
    if (!fut) return NULL;
    memset(fut, 0, sizeof(VmFuture));
    fut->main_vm = vm;
    fut->thunk_or_value = thunk_or_value;
    fut->result = NIL_VAL;
    fut->ready = 0;
    if (pthread_mutex_init(&fut->mutex, NULL) != 0) return NULL;
    if (pthread_cond_init(&fut->done, NULL) != 0) {
        pthread_mutex_destroy(&fut->mutex);
        return NULL;
    }
    return fut;
}

static void vm_future_mark_ready(VmFuture* fut, Value result) {
    pthread_mutex_lock(&fut->mutex);
    fut->result = result;
    fut->ready = 1;
    pthread_cond_broadcast(&fut->done);
    pthread_mutex_unlock(&fut->mutex);
}

static void vm_future_task_fn(void* arg, void* result_slot) {
    (void)result_slot;
    VmFuture* fut = (VmFuture*)arg;
    if (!fut) return;
    Value result = fut->thunk_or_value;
    if (fut->thunk_or_value.type == VAL_CLOSURE) {
        result = vm_worker_call_closure(NULL, fut->main_vm,
                                        fut->thunk_or_value, NULL, 0);
    }
    vm_future_mark_ready(fut, result);
}

static int vm_future_is_ready(VmFuture* fut) {
    if (!fut) return 1;
    pthread_mutex_lock(&fut->mutex);
    int ready = fut->ready;
    pthread_mutex_unlock(&fut->mutex);
    return ready;
}

static Value vm_future_force(VmFuture* fut) {
    if (!fut) return NIL_VAL;
    pthread_mutex_lock(&fut->mutex);
    while (!fut->ready) {
        pthread_cond_wait(&fut->done, &fut->mutex);
    }
    Value result = fut->result;
    pthread_mutex_unlock(&fut->mutex);
    return result;
}

/*******************************************************************************
 * Parallel Primitives
 ******************************************************************************/

/* Callback type: applies a user function to an element, writes result */
typedef int64_t (*VmMapFn)(int64_t element, void* closure_data);
typedef int     (*VmFilterFn)(int64_t element, void* closure_data);
typedef int64_t (*VmReduceFn)(int64_t accumulator, int64_t element, void* closure_data);

/* 626: parallel-map — apply fn to each element, produce output array.
 * Currently sequential; API is parallel-ready.
 *
 * items:    input array of tagged values
 * n_items:  number of elements
 * fn:       mapping function
 * closure:  opaque closure data passed to fn
 * results:  output array (caller-allocated, same length)
 */
static void vm_parallel_map(const int64_t* items, int n_items,
                            VmMapFn fn, void* closure, int64_t* results) {
    if (!items || !fn || !results || n_items <= 0) return;
    for (int i = 0; i < n_items; i++) {
        results[i] = fn(items[i], closure);
    }
}

/* 627: parallel-filter — keep elements where predicate returns true.
 * Returns count of elements kept. Results written to `results`.
 */
static int vm_parallel_filter(const int64_t* items, int n_items,
                              VmFilterFn pred, void* closure, int64_t* results) {
    if (!items || !pred || !results || n_items <= 0) return 0;
    int out = 0;
    for (int i = 0; i < n_items; i++) {
        if (pred(items[i], closure)) {
            results[out++] = items[i];
        }
    }
    return out;
}

/* 628: parallel-for-each — apply fn for side effects, no output */
static void vm_parallel_for_each(const int64_t* items, int n_items,
                                 VmMapFn fn, void* closure) {
    if (!items || !fn || n_items <= 0) return;
    for (int i = 0; i < n_items; i++) {
        (void)fn(items[i], closure);
    }
}

/* 629: parallel-reduce — left fold with associative fn.
 * init:  initial accumulator value
 * fn:    (accumulator, element) -> new_accumulator
 */
static int64_t vm_parallel_reduce(const int64_t* items, int n_items,
                                  int64_t init, VmReduceFn fn, void* closure) {
    if (!items || !fn || n_items <= 0) return init;
    int64_t acc = init;
    for (int i = 0; i < n_items; i++) {
        acc = fn(acc, items[i], closure);
    }
    return acc;
}

/*******************************************************************************
 * Chunked Parallel Primitives (for when threading is enabled)
 *
 * These split work into chunks, submit each chunk to the thread pool,
 * then merge results. Ready for real parallelism.
 ******************************************************************************/

/* Per-chunk work descriptor for parallel-map */
typedef struct {
    const int64_t* src;
    int64_t*       dst;
    int            start;
    int            end;
    VmMapFn        fn;
    void*          closure;
} VmMapChunk;

static void vm_map_chunk_worker(void* arg, void* result) {
    (void)result;
    VmMapChunk* chunk = (VmMapChunk*)arg;
    for (int i = chunk->start; i < chunk->end; i++) {
        chunk->dst[i] = chunk->fn(chunk->src[i], chunk->closure);
    }
}

/* 630: parallel-map with thread pool (real parallelism) */
static void vm_parallel_map_threaded(VmThreadPool* pool,
                                     const int64_t* items, int n_items,
                                     VmMapFn fn, void* closure,
                                     int64_t* results) {
    if (!pool || n_items <= 0 || !fn || !items || !results) {
        /* Fallback to sequential */
        vm_parallel_map(items, n_items, fn, closure, results);
        return;
    }

    int n_workers = pool->n_threads;
    if (n_items < n_workers * 4) {
        /* Too few items to justify overhead — run sequentially */
        vm_parallel_map(items, n_items, fn, closure, results);
        return;
    }

    int chunk_size = (n_items + n_workers - 1) / n_workers;
    VmMapChunk* chunks = (VmMapChunk*)calloc(n_workers, sizeof(VmMapChunk));
    if (!chunks) {
        vm_parallel_map(items, n_items, fn, closure, results);
        return;
    }

    int n_chunks = 0;
    for (int i = 0; i < n_items; i += chunk_size) {
        int end = i + chunk_size;
        if (end > n_items) end = n_items;
        chunks[n_chunks].src = items;
        chunks[n_chunks].dst = results;
        chunks[n_chunks].start = i;
        chunks[n_chunks].end = end;
        chunks[n_chunks].fn = fn;
        chunks[n_chunks].closure = closure;
        vm_pool_submit(pool, vm_map_chunk_worker, &chunks[n_chunks], NULL);
        n_chunks++;
    }

    vm_pool_wait_all(pool);
    free(chunks);
}

/* Per-chunk work descriptor for parallel-filter */
typedef struct {
    const int64_t* src;
    int64_t*       dst;         /* thread-local output buffer */
    int            start;
    int            end;
    VmFilterFn     pred;
    void*          closure;
    int            out_count;   /* written by worker */
} VmFilterChunk;

static void vm_filter_chunk_worker(void* arg, void* result) {
    (void)result;
    VmFilterChunk* chunk = (VmFilterChunk*)arg;
    int out = 0;
    for (int i = chunk->start; i < chunk->end; i++) {
        if (chunk->pred(chunk->src[i], chunk->closure)) {
            chunk->dst[out++] = chunk->src[i];
        }
    }
    chunk->out_count = out;
}

/* 631: parallel-filter with thread pool */
static int vm_parallel_filter_threaded(VmThreadPool* pool,
                                       const int64_t* items, int n_items,
                                       VmFilterFn pred, void* closure,
                                       int64_t* results) {
    if (!pool || n_items <= 0 || !pred || !items || !results) {
        return vm_parallel_filter(items, n_items, pred, closure, results);
    }

    int n_workers = pool->n_threads;
    if (n_items < n_workers * 4) {
        return vm_parallel_filter(items, n_items, pred, closure, results);
    }

    int chunk_size = (n_items + n_workers - 1) / n_workers;

    /* Each chunk gets a thread-local output buffer */
    VmFilterChunk* chunks = (VmFilterChunk*)calloc(n_workers, sizeof(VmFilterChunk));
    int64_t** local_bufs = (int64_t**)calloc(n_workers, sizeof(int64_t*));
    if (!chunks || !local_bufs) {
        free(chunks);
        free(local_bufs);
        return vm_parallel_filter(items, n_items, pred, closure, results);
    }

    int n_chunks = 0;
    for (int i = 0; i < n_items; i += chunk_size) {
        int end = i + chunk_size;
        if (end > n_items) end = n_items;
        int len = end - i;
        local_bufs[n_chunks] = (int64_t*)calloc(len, sizeof(int64_t));
        if (!local_bufs[n_chunks]) {
            /* Cleanup and fallback */
            for (int j = 0; j < n_chunks; j++) free(local_bufs[j]);
            free(local_bufs);
            free(chunks);
            return vm_parallel_filter(items, n_items, pred, closure, results);
        }
        chunks[n_chunks].src = items;
        chunks[n_chunks].dst = local_bufs[n_chunks];
        chunks[n_chunks].start = i;
        chunks[n_chunks].end = end;
        chunks[n_chunks].pred = pred;
        chunks[n_chunks].closure = closure;
        chunks[n_chunks].out_count = 0;
        vm_pool_submit(pool, vm_filter_chunk_worker, &chunks[n_chunks], NULL);
        n_chunks++;
    }

    vm_pool_wait_all(pool);

    /* Merge chunk outputs into results */
    int total = 0;
    for (int i = 0; i < n_chunks; i++) {
        memcpy(results + total, local_bufs[i], chunks[i].out_count * sizeof(int64_t));
        total += chunks[i].out_count;
        free(local_bufs[i]);
    }
    free(local_bufs);
    free(chunks);
    return total;
}

/*******************************************************************************
 * Native Call Dispatch (IDs 620-639)
 ******************************************************************************/

/*
 * ID  Function
 * 620 vm_pool_init(n_threads)
 * 621 vm_pool_shutdown()
 * 622 vm_pool_submit(fn, arg)
 * 623 vm_pool_wait_all()
 * 624 vm_pool_thread_count()
 * 625 vm_pool_pending_count()
 * 626 parallel-map (sequential)
 * 627 parallel-filter (sequential)
 * 628 parallel-for-each (sequential)
 * 629 parallel-reduce (sequential)
 * 630 parallel-map (threaded)
 * 631 parallel-filter (threaded)
 * 632-639 reserved
 */

#endif /* VM_PARALLEL_C_INCLUDED */

/*******************************************************************************
 * Self-Test
 ******************************************************************************/

#ifdef VM_PARALLEL_TEST

#include <assert.h>

/* ── Test helpers ── */

typedef struct {
    int input;
    int output;
} TestPair;

static void test_square_task(void* arg, void* result) {
    TestPair* pair = (TestPair*)arg;
    pair->output = pair->input * pair->input;
    (void)result;
}

static void test_add_one_task(void* arg, void* result) {
    TestPair* pair = (TestPair*)arg;
    pair->output = pair->input + 1;
    (void)result;
}

/* For parallel-map test */
static int64_t square_fn(int64_t x, void* closure) {
    (void)closure;
    return x * x;
}

static int is_even_fn(int64_t x, void* closure) {
    (void)closure;
    return (x % 2) == 0;
}

static int64_t sum_fn(int64_t acc, int64_t elem, void* closure) {
    (void)closure;
    return acc + elem;
}

/* ── Tests ── */

static void test_pool_lifecycle(void) {
    printf("  test_pool_lifecycle... ");
    VmThreadPool* pool = vm_pool_init(4);
    assert(pool != NULL);
    assert(pool->n_threads == 4);
    assert(vm_pool_thread_count(pool) == 4);
    assert(vm_pool_pending_count(pool) == 0);
    vm_pool_shutdown(pool);
    printf("OK\n");
}

static void test_submit_and_wait(void) {
    printf("  test_submit_and_wait... ");
    VmThreadPool* pool = vm_pool_init(2);
    assert(pool != NULL);

    TestPair pairs[10];
    for (int i = 0; i < 10; i++) {
        pairs[i].input = i + 1;
        pairs[i].output = 0;
        int rc = vm_pool_submit(pool, test_square_task, &pairs[i], NULL);
        assert(rc == 0);
    }

    vm_pool_wait_all(pool);

    for (int i = 0; i < 10; i++) {
        assert(pairs[i].output == (i + 1) * (i + 1));
    }
    vm_pool_shutdown(pool);
    printf("OK\n");
}

static void test_many_tasks(void) {
    printf("  test_many_tasks... ");
    VmThreadPool* pool = vm_pool_init(4);
    assert(pool != NULL);

    #define N_TASKS 500
    TestPair pairs[N_TASKS];
    for (int i = 0; i < N_TASKS; i++) {
        pairs[i].input = i;
        pairs[i].output = -1;
        vm_pool_submit(pool, test_add_one_task, &pairs[i], NULL);
    }

    vm_pool_wait_all(pool);

    for (int i = 0; i < N_TASKS; i++) {
        assert(pairs[i].output == i + 1);
    }
    #undef N_TASKS
    vm_pool_shutdown(pool);
    printf("OK\n");
}

static void test_parallel_map_sequential(void) {
    printf("  test_parallel_map_sequential... ");
    int64_t items[] = {1, 2, 3, 4, 5};
    int64_t results[5] = {0};
    vm_parallel_map(items, 5, square_fn, NULL, results);
    assert(results[0] == 1);
    assert(results[1] == 4);
    assert(results[2] == 9);
    assert(results[3] == 16);
    assert(results[4] == 25);
    printf("OK\n");
}

static void test_parallel_filter_sequential(void) {
    printf("  test_parallel_filter_sequential... ");
    int64_t items[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int64_t results[8] = {0};
    int count = vm_parallel_filter(items, 8, is_even_fn, NULL, results);
    assert(count == 4);
    assert(results[0] == 2);
    assert(results[1] == 4);
    assert(results[2] == 6);
    assert(results[3] == 8);
    printf("OK\n");
}

static void test_parallel_reduce_sequential(void) {
    printf("  test_parallel_reduce_sequential... ");
    int64_t items[] = {1, 2, 3, 4, 5};
    int64_t result = vm_parallel_reduce(items, 5, 0, sum_fn, NULL);
    assert(result == 15);
    printf("OK\n");
}

static void test_parallel_map_threaded(void) {
    printf("  test_parallel_map_threaded... ");
    VmThreadPool* pool = vm_pool_init(4);
    assert(pool != NULL);

    #define N_ITEMS 100
    int64_t items[N_ITEMS];
    int64_t results[N_ITEMS];
    for (int i = 0; i < N_ITEMS; i++) {
        items[i] = i + 1;
        results[i] = 0;
    }

    vm_parallel_map_threaded(pool, items, N_ITEMS, square_fn, NULL, results);

    for (int i = 0; i < N_ITEMS; i++) {
        assert(results[i] == (int64_t)(i + 1) * (i + 1));
    }
    #undef N_ITEMS
    vm_pool_shutdown(pool);
    printf("OK\n");
}

static void test_parallel_filter_threaded(void) {
    printf("  test_parallel_filter_threaded... ");
    VmThreadPool* pool = vm_pool_init(4);
    assert(pool != NULL);

    #define N_ITEMS 100
    int64_t items[N_ITEMS];
    int64_t results[N_ITEMS];
    for (int i = 0; i < N_ITEMS; i++) {
        items[i] = i + 1;
        results[i] = 0;
    }

    int count = vm_parallel_filter_threaded(pool, items, N_ITEMS, is_even_fn, NULL, results);

    assert(count == 50);
    for (int i = 0; i < count; i++) {
        assert(results[i] % 2 == 0);
    }
    #undef N_ITEMS
    vm_pool_shutdown(pool);
    printf("OK\n");
}

static void test_empty_inputs(void) {
    printf("  test_empty_inputs... ");
    int64_t results[1] = {0};
    vm_parallel_map(NULL, 0, square_fn, NULL, results);
    int c = vm_parallel_filter(NULL, 0, is_even_fn, NULL, results);
    assert(c == 0);
    int64_t r = vm_parallel_reduce(NULL, 0, 42, sum_fn, NULL);
    assert(r == 42);
    vm_parallel_for_each(NULL, 0, square_fn, NULL);
    printf("OK\n");
}

static void test_single_element(void) {
    printf("  test_single_element... ");
    int64_t items[] = {7};
    int64_t results[1] = {0};
    vm_parallel_map(items, 1, square_fn, NULL, results);
    assert(results[0] == 49);

    int count = vm_parallel_filter(items, 1, is_even_fn, NULL, results);
    assert(count == 0);

    int64_t r = vm_parallel_reduce(items, 1, 100, sum_fn, NULL);
    assert(r == 107);
    printf("OK\n");
}

static void test_pool_auto_detect(void) {
    printf("  test_pool_auto_detect... ");
    VmThreadPool* pool = vm_pool_init(0); /* 0 = auto-detect */
    assert(pool != NULL);
    assert(pool->n_threads >= 1);
    printf("(%d threads) OK\n", pool->n_threads);
    vm_pool_shutdown(pool);
}

int main(void) {
    printf("vm_parallel self-test\n");
    printf("=====================\n");

    test_pool_lifecycle();
    test_submit_and_wait();
    test_many_tasks();
    test_parallel_map_sequential();
    test_parallel_filter_sequential();
    test_parallel_reduce_sequential();
    test_parallel_map_threaded();
    test_parallel_filter_threaded();
    test_empty_inputs();
    test_single_element();
    test_pool_auto_detect();

    printf("=====================\n");
    printf("All 11 tests passed.\n");
    return 0;
}

#endif /* VM_PARALLEL_TEST */
