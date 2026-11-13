/*
 * event_queue.c — bounded, thread-safe per-window event queue (ADR-0053).
 *
 * Producers are the window's owner thread (backend pump) and any thread that
 * calls eshkol_window_push_event(); the consumer is the owner thread. A full
 * queue drops the new event and counts it, so a stalled consumer never blocks
 * a producer.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_POSIX_C_SOURCE) && !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../platform_internal.h"

#include <errno.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

int eshkol_plat_queue_init(eshkol_plat_event_queue_t* q) {
    memset(q, 0, sizeof(*q));
    if (pthread_mutex_init(&q->lock, NULL) != 0) return ESHKOL_PLATFORM_ENOMEM;
    if (pthread_cond_init(&q->ready, NULL) != 0) {
        pthread_mutex_destroy(&q->lock);
        return ESHKOL_PLATFORM_ENOMEM;
    }
    return ESHKOL_PLATFORM_OK;
}

void eshkol_plat_queue_destroy(eshkol_plat_event_queue_t* q) {
    pthread_cond_destroy(&q->ready);
    pthread_mutex_destroy(&q->lock);
}

int eshkol_plat_queue_push(eshkol_plat_event_queue_t* q, const eshkol_event_t* ev) {
    pthread_mutex_lock(&q->lock);
    if (q->count == ESHKOL_PLAT_EVENT_QUEUE_CAP) {
        q->dropped++;
        pthread_mutex_unlock(&q->lock);
        return ESHKOL_PLATFORM_EFULL;
    }
    q->items[(q->head + q->count) % ESHKOL_PLAT_EVENT_QUEUE_CAP] = *ev;
    q->count++;
    pthread_cond_signal(&q->ready);
    pthread_mutex_unlock(&q->lock);
    return ESHKOL_PLATFORM_OK;
}

bool eshkol_plat_queue_pop(eshkol_plat_event_queue_t* q, eshkol_event_t* ev) {
    pthread_mutex_lock(&q->lock);
    if (q->count == 0) {
        pthread_mutex_unlock(&q->lock);
        return false;
    }
    *ev = q->items[q->head];
    q->head = (q->head + 1) % ESHKOL_PLAT_EVENT_QUEUE_CAP;
    q->count--;
    pthread_mutex_unlock(&q->lock);
    return true;
}

int eshkol_plat_queue_wait(eshkol_plat_event_queue_t* q, int timeout_ms) {
    pthread_mutex_lock(&q->lock);
    if (timeout_ms < 0) {
        while (q->count == 0) pthread_cond_wait(&q->ready, &q->lock);
    } else if (q->count == 0 && timeout_ms > 0) {
        /* pthread_cond_timedwait measures CLOCK_REALTIME by default. */
        struct timeval now;
        gettimeofday(&now, NULL);
        struct timespec until;
        uint64_t ns = (uint64_t)now.tv_usec * 1000ull + (uint64_t)timeout_ms * 1000000ull;
        until.tv_sec = now.tv_sec + (time_t)(ns / 1000000000ull);
        until.tv_nsec = (long)(ns % 1000000000ull);
        while (q->count == 0) {
            if (pthread_cond_timedwait(&q->ready, &q->lock, &until) == ETIMEDOUT) break;
        }
    }
    int ready = q->count > 0;
    pthread_mutex_unlock(&q->lock);
    return ready;
}

uint64_t eshkol_plat_queue_dropped(eshkol_plat_event_queue_t* q) {
    pthread_mutex_lock(&q->lock);
    uint64_t d = q->dropped;
    pthread_mutex_unlock(&q->lock);
    return d;
}
