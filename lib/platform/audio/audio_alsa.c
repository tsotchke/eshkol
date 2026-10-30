/*
 * audio_alsa.c — ALSA PCM backend (ADR-0053).
 *
 * libasound.so.2 is bound at run time. The few entry points used are declared
 * here from alsa-lib's stable C ABI (opaque handles, int enums), so the
 * backend builds without the alsa-lib development headers. The device runs
 * on a dedicated thread in blocking read/write mode: one period per
 * iteration, xruns recovered with snd_pcm_recover() and counted, latency taken
 * from snd_pcm_delay().
 *
 * Device names are ALSA PCM names: NULL means "default"; "null" is alsa-lib's
 * device-free PCM, which the headless suite uses to drive this exact path.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "../platform_internal.h"

#if defined(__linux__)

#include <errno.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>

#if !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
#error "the ALSA backend maps formats to their little-endian ALSA codes"
#endif

/* alsa-lib ABI (asoundlib.h / pcm.h). */
typedef struct _snd_pcm snd_pcm_t;
enum { SND_PCM_STREAM_PLAYBACK = 0, SND_PCM_STREAM_CAPTURE = 1 };
enum { SND_PCM_ACCESS_RW_INTERLEAVED = 3 };
enum {
    SND_PCM_FORMAT_U8 = 1, SND_PCM_FORMAT_S16_LE = 2, SND_PCM_FORMAT_S32_LE = 10,
    SND_PCM_FORMAT_FLOAT_LE = 14, SND_PCM_FORMAT_S24_3LE = 32
};

static struct {
    int  (*snd_pcm_open)(snd_pcm_t**, const char*, int, int);
    int  (*snd_pcm_set_params)(snd_pcm_t*, int, int, unsigned int, unsigned int, int, unsigned int);
    int  (*snd_pcm_get_params)(snd_pcm_t*, unsigned long*, unsigned long*);
    long (*snd_pcm_writei)(snd_pcm_t*, const void*, unsigned long);
    long (*snd_pcm_readi)(snd_pcm_t*, void*, unsigned long);
    int  (*snd_pcm_recover)(snd_pcm_t*, int, int);
    int  (*snd_pcm_delay)(snd_pcm_t*, long*);
    int  (*snd_pcm_drop)(snd_pcm_t*);
    int  (*snd_pcm_prepare)(snd_pcm_t*);
    int  (*snd_pcm_close)(snd_pcm_t*);
} A;

static const char* const k_alsa_symbols[] = {
    "snd_pcm_open", "snd_pcm_set_params", "snd_pcm_get_params", "snd_pcm_writei",
    "snd_pcm_readi", "snd_pcm_recover", "snd_pcm_delay", "snd_pcm_drop",
    "snd_pcm_prepare", "snd_pcm_close", NULL
};

static pthread_once_t g_alsa_once = PTHREAD_ONCE_INIT;
static int g_alsa_loaded = 0;

static void alsa_load(void) {
    static const char* const libs[] = { "libasound.so.2", "libasound.so", NULL };
    g_alsa_loaded = eshkol_plat_dlsym_all(eshkol_plat_dlopen_any(libs), k_alsa_symbols,
                                          (void**)&A);
}

static int alsa_available(void) {
    pthread_once(&g_alsa_once, alsa_load);
    return g_alsa_loaded;
}

static int alsa_format(int f) {
    switch (f) {
        case ESHKOL_AUDIO_FORMAT_U8:  return SND_PCM_FORMAT_U8;
        case ESHKOL_AUDIO_FORMAT_S16: return SND_PCM_FORMAT_S16_LE;
        case ESHKOL_AUDIO_FORMAT_S24: return SND_PCM_FORMAT_S24_3LE;
        case ESHKOL_AUDIO_FORMAT_S32: return SND_PCM_FORMAT_S32_LE;
        default:                      return SND_PCM_FORMAT_FLOAT_LE;
    }
}

typedef struct {
    snd_pcm_t* pcm;
    pthread_t thread;
    atomic_int run;
    int started;
    uint8_t* buffer;
} alsa_audio_t;

static int alsa_open(eshkol_audio_device_t* d, const char* device_name) {
    alsa_audio_t* a = (alsa_audio_t*)calloc(1, sizeof(*a));
    if (!a) return ESHKOL_PLATFORM_ENOMEM;
    int stream = d->direction == ESHKOL_AUDIO_INPUT ? SND_PCM_STREAM_CAPTURE
                                                    : SND_PCM_STREAM_PLAYBACK;
    if (A.snd_pcm_open(&a->pcm, device_name ? device_name : "default", stream, 0) < 0) {
        free(a);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    /* Four periods of buffering: latency = 4 x period. */
    unsigned int latency_us = (unsigned int)((uint64_t)d->config.buffer_frames * 4u *
                                             1000000ull / d->config.sample_rate);
    if (A.snd_pcm_set_params(a->pcm, alsa_format(d->config.format),
                             SND_PCM_ACCESS_RW_INTERLEAVED, d->config.channels,
                             d->config.sample_rate, 1, latency_us) < 0) {
        A.snd_pcm_close(a->pcm);
        free(a);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    unsigned long buffer_size = 0, period_size = 0;
    if (A.snd_pcm_get_params(a->pcm, &buffer_size, &period_size) == 0 && period_size >= 16 &&
        period_size <= 16384)
        d->config.buffer_frames = (uint32_t)period_size;
    uint32_t fb = eshkol_audio_format_bytes(d->config.format) * d->config.channels;
    a->buffer = (uint8_t*)calloc(d->config.buffer_frames, fb);
    if (!a->buffer) {
        A.snd_pcm_close(a->pcm);
        free(a);
        return ESHKOL_PLATFORM_ENOMEM;
    }
    atomic_init(&a->run, 0);
    atomic_store(&d->device_latency_frames, (unsigned)buffer_size);
    d->impl = a;
    return ESHKOL_PLATFORM_OK;
}

static void* alsa_thread(void* arg) {
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)arg;
    alsa_audio_t* a = (alsa_audio_t*)d->impl;
    struct sched_param sp;
    memset(&sp, 0, sizeof(sp));
    sp.sched_priority = sched_get_priority_min(SCHED_FIFO);
    (void)pthread_setschedparam(pthread_self(), SCHED_FIFO, &sp);
    const uint32_t period = d->config.buffer_frames;
    while (atomic_load_explicit(&a->run, memory_order_acquire)) {
        if (d->direction == ESHKOL_AUDIO_OUTPUT) {
            eshkol_plat_audio_process(d, a->buffer, period);
            uint32_t done = 0;
            while (done < period && atomic_load_explicit(&a->run, memory_order_relaxed)) {
                long n = A.snd_pcm_writei(a->pcm, a->buffer + (size_t)done * d->frame_bytes,
                                          period - done);
                if (n < 0) {
                    if (n == -EPIPE || n == -ESTRPIPE)
                        atomic_fetch_add_explicit(&d->device_xruns, 1u, memory_order_relaxed);
                    if (A.snd_pcm_recover(a->pcm, (int)n, 1) < 0) {
                        atomic_store(&a->run, 0);
                        break;
                    }
                    continue;
                }
                done += (uint32_t)n;
            }
        } else {
            long n = A.snd_pcm_readi(a->pcm, a->buffer, period);
            if (n < 0) {
                if (n == -EPIPE || n == -ESTRPIPE)
                    atomic_fetch_add_explicit(&d->device_xruns, 1u, memory_order_relaxed);
                if (A.snd_pcm_recover(a->pcm, (int)n, 1) < 0) atomic_store(&a->run, 0);
                continue;
            }
            if (n > 0) eshkol_plat_audio_process(d, a->buffer, (uint32_t)n);
        }
        long delay = 0;
        if (A.snd_pcm_delay(a->pcm, &delay) == 0 && delay >= 0)
            atomic_store_explicit(&d->device_latency_frames, (unsigned)delay,
                                  memory_order_relaxed);
    }
    return NULL;
}

static int alsa_start(eshkol_audio_device_t* d) {
    alsa_audio_t* a = (alsa_audio_t*)d->impl;
    (void)A.snd_pcm_prepare(a->pcm);
    atomic_store_explicit(&a->run, 1, memory_order_release);
    if (pthread_create(&a->thread, NULL, alsa_thread, d) != 0) {
        atomic_store(&a->run, 0);
        return ESHKOL_PLATFORM_EDEVICE;
    }
    a->started = 1;
    return ESHKOL_PLATFORM_OK;
}

static int alsa_stop(eshkol_audio_device_t* d) {
    alsa_audio_t* a = (alsa_audio_t*)d->impl;
    if (a->started) {
        atomic_store_explicit(&a->run, 0, memory_order_release);
        pthread_join(a->thread, NULL);  /* a blocking call returns within one period */
        a->started = 0;
        (void)A.snd_pcm_drop(a->pcm);
    }
    return ESHKOL_PLATFORM_OK;
}

static void alsa_close(eshkol_audio_device_t* d) {
    alsa_audio_t* a = (alsa_audio_t*)d->impl;
    if (!a) return;
    (void)alsa_stop(d);
    A.snd_pcm_close(a->pcm);
    free(a->buffer);
    free(a);
    d->impl = NULL;
}

const eshkol_audio_backend_ops_t eshkol_plat_audio_alsa_ops = {
    ESHKOL_BACKEND_ALSA, alsa_available, alsa_open, alsa_start, alsa_stop, alsa_close
};

#endif /* __linux__ */
