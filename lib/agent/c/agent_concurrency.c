/*******************************************************************************
 * Concurrency Primitives for Eshkol Agent
 *
 * Provides: channels (bounded message queues), mutexes, condition variables,
 * and timer scheduling — all built on pthreads.
 *
 * These are LLVM-path extern "C" functions. The VM parallel primitives
 * (cases 620-628) are separate and live in vm_native.c.
 *
 * B.10: make-channel, channel-send!, channel-receive, channel-try-receive,
 *       channel-close!, make-mutex, mutex-lock!, mutex-unlock!, with-mutex,
 *       make-condition-variable, condition-wait, condition-signal,
 *       condition-broadcast, make-timer, timer-cancel!, make-interval,
 *       interval-cancel!
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

/*******************************************************************************
 * Mutex
 *
 * Simple pthread_mutex wrapper with handle table.
 ******************************************************************************/

#define MAX_MUTEXES 64

static pthread_mutex_t* g_mutexes[MAX_MUTEXES] = {0};

/**
 * @brief Finds the first free (NULL) slot in the mutex handle table, skipping index 0 so 0 is never a valid handle.
 *
 * @return Index of a free slot (>= 1), or -1 if all MAX_MUTEXES slots are in use.
 */
static int alloc_mutex_slot(void) {
    for (int i = 1; i < MAX_MUTEXES; i++) {
        if (!g_mutexes[i]) return i;
    }
    return -1;
}

/*
 * Create a new mutex.
 * Returns: handle (>= 1), -1 if slots full
 */
/**
 * @brief Allocates and initializes a new pthread mutex and stores it in the mutex handle table.
 *
 * @return Mutex handle (>= 1) on success, -1 if the slot table is full, allocation fails, or pthread_mutex_init() fails.
 */
int64_t eshkol_make_mutex(void) {
    int slot = alloc_mutex_slot();
    if (slot < 0) return -1;
    pthread_mutex_t* m = (pthread_mutex_t*)malloc(sizeof(pthread_mutex_t));
    if (!m) return -1;
    if (pthread_mutex_init(m, NULL) != 0) { free(m); return -1; }
    g_mutexes[slot] = m;
    return (int64_t)slot;
}

/**
 * @brief Locks the mutex at @p handle, blocking until it is acquired.
 *
 * @param handle Mutex handle from eshkol_make_mutex().
 * @return 0 on success, -1 on invalid handle or pthread_mutex_lock() failure.
 */
int32_t eshkol_mutex_lock(int64_t handle) {
    if (handle < 1 || handle >= MAX_MUTEXES || !g_mutexes[handle]) return -1;
    return pthread_mutex_lock(g_mutexes[handle]) == 0 ? 0 : -1;
}

/**
 * @brief Unlocks the mutex at @p handle.
 *
 * @param handle Mutex handle from eshkol_make_mutex().
 * @return 0 on success, -1 on invalid handle or pthread_mutex_unlock() failure.
 */
int32_t eshkol_mutex_unlock(int64_t handle) {
    if (handle < 1 || handle >= MAX_MUTEXES || !g_mutexes[handle]) return -1;
    return pthread_mutex_unlock(g_mutexes[handle]) == 0 ? 0 : -1;
}

/**
 * @brief Attempts to lock the mutex at @p handle without blocking.
 *
 * @param handle Mutex handle from eshkol_make_mutex().
 * @return 1 if the lock was acquired, 0 if it is already held by someone else, -1 on invalid handle or other error.
 */
int32_t eshkol_mutex_trylock(int64_t handle) {
    if (handle < 1 || handle >= MAX_MUTEXES || !g_mutexes[handle]) return -1;
    int r = pthread_mutex_trylock(g_mutexes[handle]);
    if (r == 0) return 1;           /* Locked successfully */
    if (r == EBUSY) return 0;       /* Already locked by someone else */
    return -1;                       /* Error */
}

/**
 * @brief Destroys the pthread mutex at @p handle, frees it, and clears its slot.
 *
 * @param handle Mutex handle from eshkol_make_mutex(). No-op if out of range or already destroyed.
 */
void eshkol_mutex_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_MUTEXES || !g_mutexes[handle]) return;
    pthread_mutex_destroy(g_mutexes[handle]);
    free(g_mutexes[handle]);
    g_mutexes[handle] = NULL;
}

/*******************************************************************************
 * Condition Variable
 ******************************************************************************/

#define MAX_CONDVARS 64

static pthread_cond_t* g_condvars[MAX_CONDVARS] = {0};

/**
 * @brief Allocates and initializes a new pthread condition variable and stores it in the condition-variable handle table.
 *
 * @return Condition-variable handle (>= 1) on success, -1 if the slot table is full, allocation fails, or pthread_cond_init() fails.
 */
int64_t eshkol_make_condition_variable(void) {
    for (int i = 1; i < MAX_CONDVARS; i++) {
        if (!g_condvars[i]) {
            pthread_cond_t* cv = (pthread_cond_t*)malloc(sizeof(pthread_cond_t));
            if (!cv) return -1;
            if (pthread_cond_init(cv, NULL) != 0) { free(cv); return -1; }
            g_condvars[i] = cv;
            return (int64_t)i;
        }
    }
    return -1;
}

/**
 * @brief Atomically unlocks @p mutex_handle and waits on @p cv_handle, relocking the mutex before returning.
 *
 * Mirrors pthread_cond_wait() semantics: the caller must already hold the
 * mutex, and must be prepared to re-check its wait condition on return
 * since condition variables can wake spuriously.
 *
 * @param cv_handle Condition-variable handle from eshkol_make_condition_variable().
 * @param mutex_handle Mutex handle from eshkol_make_mutex(), currently held by the caller.
 * @return 0 on success, -1 on invalid handles or pthread_cond_wait() failure.
 */
int32_t eshkol_condition_wait(int64_t cv_handle, int64_t mutex_handle) {
    if (cv_handle < 1 || cv_handle >= MAX_CONDVARS || !g_condvars[cv_handle]) return -1;
    if (mutex_handle < 1 || mutex_handle >= MAX_MUTEXES || !g_mutexes[mutex_handle]) return -1;
    return pthread_cond_wait(g_condvars[cv_handle], g_mutexes[mutex_handle]) == 0 ? 0 : -1;
}

/**
 * @brief Wakes at least one thread waiting on the condition variable at @p handle.
 *
 * @param handle Condition-variable handle from eshkol_make_condition_variable().
 * @return 0 on success, -1 on invalid handle or pthread_cond_signal() failure.
 */
int32_t eshkol_condition_signal(int64_t handle) {
    if (handle < 1 || handle >= MAX_CONDVARS || !g_condvars[handle]) return -1;
    return pthread_cond_signal(g_condvars[handle]) == 0 ? 0 : -1;
}

/**
 * @brief Wakes all threads waiting on the condition variable at @p handle.
 *
 * @param handle Condition-variable handle from eshkol_make_condition_variable().
 * @return 0 on success, -1 on invalid handle or pthread_cond_broadcast() failure.
 */
int32_t eshkol_condition_broadcast(int64_t handle) {
    if (handle < 1 || handle >= MAX_CONDVARS || !g_condvars[handle]) return -1;
    return pthread_cond_broadcast(g_condvars[handle]) == 0 ? 0 : -1;
}

/**
 * @brief Destroys the pthread condition variable at @p handle, frees it, and clears its slot.
 *
 * @param handle Condition-variable handle from eshkol_make_condition_variable(). No-op if out of range or already destroyed.
 */
void eshkol_condition_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_CONDVARS || !g_condvars[handle]) return;
    pthread_cond_destroy(g_condvars[handle]);
    free(g_condvars[handle]);
    g_condvars[handle] = NULL;
}

/*******************************************************************************
 * Channel — Bounded thread-safe message queue
 *
 * Capacity 0 = rendezvous (synchronous: sender blocks until receiver arrives).
 * Capacity N = buffer N values (sender blocks when full, receiver blocks when
 * empty).
 *
 * Values are stored as opaque int64_t (tagged values from Eshkol runtime).
 ******************************************************************************/

#define MAX_CHANNELS 32
#define MAX_CHANNEL_BUF 4096

typedef struct {
    int64_t* buf;           /* Circular buffer */
    int32_t  capacity;
    int32_t  head;          /* Next write position */
    int32_t  tail;          /* Next read position */
    int32_t  count;         /* Current items in buffer */
    int32_t  closed;        /* 1 if channel is closed */
    pthread_mutex_t lock;
    pthread_cond_t  not_full;
    pthread_cond_t  not_empty;
} Channel;

static Channel* g_channels[MAX_CHANNELS] = {0};

/*
 * Create a bounded channel.
 * capacity: max buffered values. 0 = synchronous (rendezvous).
 * Returns: handle (>= 1), -1 error
 */
/**
 * @brief Allocates a bounded, thread-safe FIFO channel with room for @p capacity buffered values.
 *
 * A @p capacity of 0 is stored internally with one buffer slot but behaves
 * as a synchronous rendezvous, since eshkol_channel_send() and
 * eshkol_channel_receive() coordinate via the same not_full/not_empty
 * condition variables regardless of buffer size.
 *
 * @param capacity Maximum number of buffered values (0 for synchronous rendezvous); must be between 0 and MAX_CHANNEL_BUF.
 * @return Channel handle (>= 1) on success, -1 if @p capacity is out of range, the slot table is full, or allocation fails.
 */
int64_t eshkol_make_channel(int32_t capacity) {
    if (capacity < 0 || capacity > MAX_CHANNEL_BUF) return -1;
    int effective_cap = capacity > 0 ? capacity : 1;  /* Need at least 1 slot for rendezvous */

    for (int i = 1; i < MAX_CHANNELS; i++) {
        if (!g_channels[i]) {
            Channel* ch = (Channel*)calloc(1, sizeof(Channel));
            if (!ch) return -1;
            ch->buf = (int64_t*)malloc(sizeof(int64_t) * (size_t)effective_cap);
            if (!ch->buf) { free(ch); return -1; }
            ch->capacity = effective_cap;
            ch->head = ch->tail = ch->count = ch->closed = 0;
            pthread_mutex_init(&ch->lock, NULL);
            pthread_cond_init(&ch->not_full, NULL);
            pthread_cond_init(&ch->not_empty, NULL);
            g_channels[i] = ch;
            return (int64_t)i;
        }
    }
    return -1;
}

/*
 * Send a value to the channel. Blocks if buffer is full.
 * Returns: 0 success, -1 error (channel closed)
 */
/**
 * @brief Pushes @p value onto the channel at @p handle, blocking while the buffer is full.
 *
 * @param handle Channel handle from eshkol_make_channel().
 * @param value Opaque tagged Eshkol value to enqueue.
 * @return 0 on success, -1 on invalid handle or if the channel is (or becomes) closed while waiting.
 */
int32_t eshkol_channel_send(int64_t handle, int64_t value) {
    if (handle < 1 || handle >= MAX_CHANNELS || !g_channels[handle]) return -1;
    Channel* ch = g_channels[handle];

    pthread_mutex_lock(&ch->lock);
    while (ch->count >= ch->capacity && !ch->closed) {
        pthread_cond_wait(&ch->not_full, &ch->lock);
    }
    if (ch->closed) {
        pthread_mutex_unlock(&ch->lock);
        return -1;
    }
    ch->buf[ch->head] = value;
    ch->head = (ch->head + 1) % ch->capacity;
    ch->count++;
    pthread_cond_signal(&ch->not_empty);
    pthread_mutex_unlock(&ch->lock);
    return 0;
}

/*
 * Receive a value from the channel. Blocks until available or timeout.
 * timeout_ms: -1 = infinite, 0 = immediate, >0 = milliseconds
 * value_out: receives the value
 * Returns: 1 = value received, 0 = timeout, -1 = closed/error
 */
/**
 * @brief Pops the next value from the channel at @p handle, waiting according to @p timeout_ms if it is empty.
 *
 * @param handle Channel handle from eshkol_make_channel().
 * @param timeout_ms -1 to wait indefinitely, 0 to try without blocking, or a positive number of milliseconds to wait.
 * @param value_out Output: receives the dequeued value when the return value is 1.
 * @return 1 if a value was received, 0 on timeout (or empty with timeout_ms 0), -1 on invalid handle/arguments or if the channel is closed and empty.
 */
int32_t eshkol_channel_receive(int64_t handle, int32_t timeout_ms, int64_t* value_out) {
    if (handle < 1 || handle >= MAX_CHANNELS || !g_channels[handle] || !value_out) return -1;
    Channel* ch = g_channels[handle];

    pthread_mutex_lock(&ch->lock);

    if (timeout_ms == 0) {
        /* Non-blocking try */
        if (ch->count == 0) {
            pthread_mutex_unlock(&ch->lock);
            return ch->closed ? -1 : 0;
        }
    } else if (timeout_ms > 0) {
        /* Timed wait */
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        ts.tv_sec += timeout_ms / 1000;
        ts.tv_nsec += (timeout_ms % 1000) * 1000000L;
        if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }

        while (ch->count == 0 && !ch->closed) {
            int r = pthread_cond_timedwait(&ch->not_empty, &ch->lock, &ts);
            if (r == ETIMEDOUT) {
                pthread_mutex_unlock(&ch->lock);
                return 0;  /* Timeout */
            }
        }
    } else {
        /* Infinite wait */
        while (ch->count == 0 && !ch->closed) {
            pthread_cond_wait(&ch->not_empty, &ch->lock);
        }
    }

    if (ch->count == 0 && ch->closed) {
        pthread_mutex_unlock(&ch->lock);
        return -1;
    }

    *value_out = ch->buf[ch->tail];
    ch->tail = (ch->tail + 1) % ch->capacity;
    ch->count--;
    pthread_cond_signal(&ch->not_full);
    pthread_mutex_unlock(&ch->lock);
    return 1;
}

/*
 * Close a channel. Wakes all waiting senders/receivers.
 */
/**
 * @brief Marks the channel at @p handle closed and wakes all threads blocked in eshkol_channel_send() or eshkol_channel_receive().
 *
 * @param handle Channel handle from eshkol_make_channel(). No-op if out of range or unallocated.
 */
void eshkol_channel_close(int64_t handle) {
    if (handle < 1 || handle >= MAX_CHANNELS || !g_channels[handle]) return;
    Channel* ch = g_channels[handle];
    pthread_mutex_lock(&ch->lock);
    ch->closed = 1;
    pthread_cond_broadcast(&ch->not_full);
    pthread_cond_broadcast(&ch->not_empty);
    pthread_mutex_unlock(&ch->lock);
}

/*
 * Destroy a channel and free resources.
 */
/**
 * @brief Closes the channel at @p handle, destroys its mutex/condition variables, frees its buffer, and clears its slot.
 *
 * @param handle Channel handle from eshkol_make_channel(). No-op if out of range or unallocated.
 */
void eshkol_channel_destroy(int64_t handle) {
    if (handle < 1 || handle >= MAX_CHANNELS || !g_channels[handle]) return;
    Channel* ch = g_channels[handle];
    eshkol_channel_close(handle);
    pthread_mutex_destroy(&ch->lock);
    pthread_cond_destroy(&ch->not_full);
    pthread_cond_destroy(&ch->not_empty);
    free(ch->buf);
    free(ch);
    g_channels[handle] = NULL;
}

/*******************************************************************************
 * Timer — One-shot delayed callback
 *
 * Timers run on a background thread. When the delay expires, the timer
 * sets a "fired" flag that can be polled from Eshkol code. (We cannot
 * safely call Eshkol closures from a pthread — the Eshkol runtime is not
 * thread-safe for closure invocation. Instead, the agent polls timers
 * in its event loop.)
 ******************************************************************************/

#define MAX_TIMERS 64

typedef struct {
    pthread_t thread;
    int64_t   delay_ms;
    int64_t   interval_ms;   /* 0 = one-shot, >0 = repeating */
    volatile int32_t fired;  /* 1 when timer expired */
    volatile int32_t cancelled;
    volatile int32_t active;
} Timer;

static Timer* g_timers[MAX_TIMERS] = {0};

/**
 * @brief Background thread body for both one-shot timers and repeating intervals; sleeps for the configured delay and sets the "fired" flag.
 *
 * For a repeating timer (@c interval_ms > 0), loops sleeping and setting
 * @c fired until cancelled. For a one-shot timer, sleeps once for
 * @c delay_ms and sets @c fired, then exits. Never invokes an Eshkol
 * closure directly from this thread — the Eshkol runtime is not
 * thread-safe for closure invocation, so callers must poll
 * eshkol_timer_check() from the main event loop instead.
 *
 * @param arg Pointer to the owning Timer.
 * @return Always NULL.
 */
static void* timer_thread_func(void* arg) {
    Timer* t = (Timer*)arg;

    if (t->interval_ms > 0) {
        /* Repeating interval */
        while (!t->cancelled) {
            struct timespec ts;
            ts.tv_sec = t->interval_ms / 1000;
            ts.tv_nsec = (t->interval_ms % 1000) * 1000000L;
            nanosleep(&ts, NULL);
            if (!t->cancelled) t->fired = 1;
        }
    } else {
        /* One-shot delay */
        struct timespec ts;
        ts.tv_sec = t->delay_ms / 1000;
        ts.tv_nsec = (t->delay_ms % 1000) * 1000000L;
        nanosleep(&ts, NULL);
        if (!t->cancelled) t->fired = 1;
    }

    t->active = 0;
    return NULL;
}

/*
 * Create a one-shot timer.
 * delay_ms: milliseconds until timer fires
 * Returns: handle (>= 1), -1 error
 */
/**
 * @brief Creates and starts a detached background thread that sets a "fired" flag once after @p delay_ms.
 *
 * @param delay_ms Milliseconds to wait before firing; must be >= 0.
 * @return Timer handle (>= 1) on success, -1 if @p delay_ms is negative, the slot table is full, allocation fails, or the thread cannot be created.
 */
int64_t eshkol_make_timer(int64_t delay_ms) {
    if (delay_ms < 0) return -1;
    for (int i = 1; i < MAX_TIMERS; i++) {
        if (!g_timers[i]) {
            Timer* t = (Timer*)calloc(1, sizeof(Timer));
            if (!t) return -1;
            t->delay_ms = delay_ms;
            t->interval_ms = 0;
            t->fired = 0;
            t->cancelled = 0;
            t->active = 1;
            g_timers[i] = t;
            if (pthread_create(&t->thread, NULL, timer_thread_func, t) != 0) {
                free(t);
                g_timers[i] = NULL;
                return -1;
            }
            pthread_detach(t->thread);
            return (int64_t)i;
        }
    }
    return -1;
}

/*
 * Create a repeating interval timer.
 * interval_ms: milliseconds between firings
 * Returns: handle (>= 1), -1 error
 */
/**
 * @brief Creates and starts a detached background thread that sets a "fired" flag repeatedly every @p interval_ms until cancelled.
 *
 * @param interval_ms Milliseconds between firings; must be > 0.
 * @return Timer handle (>= 1) on success, -1 if @p interval_ms is not positive, the slot table is full, allocation fails, or the thread cannot be created.
 */
int64_t eshkol_make_interval(int64_t interval_ms) {
    if (interval_ms <= 0) return -1;
    for (int i = 1; i < MAX_TIMERS; i++) {
        if (!g_timers[i]) {
            Timer* t = (Timer*)calloc(1, sizeof(Timer));
            if (!t) return -1;
            t->delay_ms = interval_ms;
            t->interval_ms = interval_ms;
            t->fired = 0;
            t->cancelled = 0;
            t->active = 1;
            g_timers[i] = t;
            if (pthread_create(&t->thread, NULL, timer_thread_func, t) != 0) {
                free(t);
                g_timers[i] = NULL;
                return -1;
            }
            pthread_detach(t->thread);
            return (int64_t)i;
        }
    }
    return -1;
}

/*
 * Check if timer has fired (and clear the flag).
 * Returns: 1 if fired since last check, 0 if not yet, -1 if handle invalid
 */
/**
 * @brief Checks whether the timer at @p handle has fired since the last check, clearing the flag if so.
 *
 * @param handle Timer handle from eshkol_make_timer() or eshkol_make_interval().
 * @return 1 if the timer fired since the last check, 0 if not yet fired, -1 if @p handle is invalid.
 */
int32_t eshkol_timer_check(int64_t handle) {
    if (handle < 1 || handle >= MAX_TIMERS || !g_timers[handle]) return -1;
    Timer* t = g_timers[handle];
    if (t->fired) {
        t->fired = 0;
        return 1;
    }
    return 0;
}

/*
 * Cancel a timer or interval.
 */
/**
 * @brief Cancels the timer at @p handle, waits briefly for its background thread to notice, then frees it and clears its slot.
 *
 * Waits up to ~100ms (100 x 1ms sleeps) for the timer thread to observe
 * the cancellation flag and exit before freeing the Timer struct out from
 * under it; this is a best-effort wait, not a hard join.
 *
 * @param handle Timer handle from eshkol_make_timer() or eshkol_make_interval(). No-op if out of range or unallocated.
 */
void eshkol_timer_cancel(int64_t handle) {
    if (handle < 1 || handle >= MAX_TIMERS || !g_timers[handle]) return;
    Timer* t = g_timers[handle];
    t->cancelled = 1;
    /* Wait briefly for thread to notice */
    for (int i = 0; i < 100 && t->active; i++) {
        usleep(1000);  /* 1ms */
    }
    free(t);
    g_timers[handle] = NULL;
}
