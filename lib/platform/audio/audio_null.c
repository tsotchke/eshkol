/*
 * audio_null.c — null audio backend: a device-free real-time clock
 * (ADR-0053).
 *
 * A dedicated thread runs one period every buffer_frames / sample_rate
 * seconds against absolute monotonic deadlines (no drift accumulation) and
 * calls the same eshkol_plat_audio_process() path as the device backends.
 * Output periods are consumed and discarded; input periods deliver silence.
 * Headless test suites and CI exercise the full callback, queue, underrun and
 * timing machinery through this backend.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_POSIX_C_SOURCE) && !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../platform_internal.h"

#include <sched.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    pthread_t thread;
    atomic_int run;
    int started;
    uint8_t* buffer;
} null_audio_t;

static int null_available(void) { return 1; }

static void* null_thread(void* arg) {
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)arg;
    null_audio_t* n = (null_audio_t*)d->impl;
    /* Best effort: real-time priority where the process is allowed it. */
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = sched_get_priority_min(SCHED_FIFO);
    (void)pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);

    const uint64_t period_ns =
        (uint64_t)d->config.buffer_frames * 1000000000ull / d->config.sample_rate;
    const size_t bytes = (size_t)d->config.buffer_frames * d->frame_bytes;
    uint64_t deadline = eshkol_plat_now_ns();
    while (atomic_load_explicit(&n->run, memory_order_acquire)) {
        if (d->direction == ESHKOL_AUDIO_INPUT)
            memset(n->buffer, d->config.format == ESHKOL_AUDIO_FORMAT_U8 ? 0x80 : 0, bytes);
        eshkol_plat_audio_process(d, n->buffer, d->config.buffer_frames);
        deadline += period_ns;
        uint64_t now = eshkol_plat_now_ns();
        if (now > deadline + 4 * period_ns) deadline = now;  /* resynchronise after a stall */
        eshkol_plat_sleep_until_ns(deadline);
    }
    return NULL;
}

static int null_open(eshkol_audio_device_t* d, const char* device_name) {
    (void)device_name;
    null_audio_t* n = (null_audio_t*)calloc(1, sizeof(*n));
    if (!n) return ESHKOL_PLATFORM_ENOMEM;
    uint32_t fb = eshkol_audio_format_bytes(d->config.format) * d->config.channels;
    n->buffer = (uint8_t*)calloc(d->config.buffer_frames, fb);
    if (!n->buffer) { free(n); return ESHKOL_PLATFORM_ENOMEM; }
    atomic_init(&n->run, 0);
    d->impl = n;
    atomic_store(&d->device_latency_frames, d->config.buffer_frames);
    return ESHKOL_PLATFORM_OK;
}

static int null_start(eshkol_audio_device_t* d) {
    null_audio_t* n = (null_audio_t*)d->impl;
    atomic_store_explicit(&n->run, 1, memory_order_release);
    if (pthread_create(&n->thread, NULL, null_thread, d) != 0) {
        atomic_store(&n->run, 0);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    n->started = 1;
    return ESHKOL_PLATFORM_OK;
}

static int null_stop(eshkol_audio_device_t* d) {
    null_audio_t* n = (null_audio_t*)d->impl;
    if (n->started) {
        atomic_store_explicit(&n->run, 0, memory_order_release);
        pthread_join(n->thread, NULL);
        n->started = 0;
    }
    return ESHKOL_PLATFORM_OK;
}

static void null_close(eshkol_audio_device_t* d) {
    null_audio_t* n = (null_audio_t*)d->impl;
    if (!n) return;
    (void)null_stop(d);
    free(n->buffer);
    free(n);
    d->impl = NULL;
}

const eshkol_audio_backend_ops_t eshkol_plat_audio_null_ops = {
    ESHKOL_BACKEND_AUDIO_NULL, null_available, null_open, null_start, null_stop, null_close
};
