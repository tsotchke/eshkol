/**
 * @file vm_parallel.c
 * @brief Thread pool and parallel primitives for the Eshkol bytecode VM.
 *
 * Provides a work-stealing thread pool and parallel higher-order primitives
 * (parallel-map, parallel-filter, parallel-for-each, parallel-reduce).
 *
 * The bytecode VM is single-threaded, so parallel-map etc. currently execute
 * sequentially with the correct API — drop-in ready for true parallelism
 * when per-thread VM instances are added.
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
 * Parallel Primitives (sequential until per-thread VM is ready)
 *
 * These operate on arrays of tagged values represented as int64_t.
 * The callback is a C function pointer wrapping a VM closure call.
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
