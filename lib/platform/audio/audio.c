/*
 * audio.c — backend-independent audio device, queue-mode ring, real-time
 * period processing and the Eshkol language bridge (ADR-0053).
 *
 * Real-time path: eshkol_plat_audio_process() runs on the backend's audio
 * thread. It reads the clock, calls the device callback and updates counters
 * with relaxed/release atomics; it never allocates, locks or blocks. The
 * queue-mode ring is single-producer/single-consumer: for output the owner
 * thread produces and the audio thread consumes; for input the reverse.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

const eshkol_audio_backend_ops_t* eshkol_plat_audio_ops_for(int backend) {
    switch (backend) {
        case ESHKOL_BACKEND_AUDIO_NULL: return &eshkol_plat_audio_null_ops;
#if defined(__linux__)
        case ESHKOL_BACKEND_ALSA:       return &eshkol_plat_audio_alsa_ops;
#endif
#if defined(__APPLE__)
        case ESHKOL_BACKEND_COREAUDIO:  return &eshkol_plat_audio_coreaudio_ops;
#endif
        default:                        return NULL;
    }
}

uint32_t eshkol_audio_format_bytes(int format) {
    switch (format) {
        case ESHKOL_AUDIO_FORMAT_U8:  return 1;
        case ESHKOL_AUDIO_FORMAT_S16: return 2;
        case ESHKOL_AUDIO_FORMAT_S24: return 3;
        case ESHKOL_AUDIO_FORMAT_S32: return 4;
        case ESHKOL_AUDIO_FORMAT_F32: return 4;
        default:                      return 0;
    }
}

/* ── sample conversion ([-1, 1] doubles <-> device format) ─────────── */

static void store_sample(uint8_t* p, int format, double v) {
    if (!(v == v)) v = 0.0;       /* NaN -> silence */
    if (v > 1.0) v = 1.0;
    if (v < -1.0) v = -1.0;
    switch (format) {
        case ESHKOL_AUDIO_FORMAT_U8:
            p[0] = (uint8_t)lrint(v * 127.0 + 128.0);
            break;
        case ESHKOL_AUDIO_FORMAT_S16: {
            int16_t s = (int16_t)lrint(v * 32767.0);
            memcpy(p, &s, 2);
            break;
        }
        case ESHKOL_AUDIO_FORMAT_S24: {
            int32_t s = (int32_t)lrint(v * 8388607.0);
            p[0] = (uint8_t)(s & 0xFF);
            p[1] = (uint8_t)((s >> 8) & 0xFF);
            p[2] = (uint8_t)((s >> 16) & 0xFF);
            break;
        }
        case ESHKOL_AUDIO_FORMAT_S32: {
            int32_t s = (int32_t)llrint(v * 2147483647.0);
            memcpy(p, &s, 4);
            break;
        }
        default: {
            float f = (float)v;
            memcpy(p, &f, 4);
            break;
        }
    }
}

static double load_sample(const uint8_t* p, int format) {
    switch (format) {
        case ESHKOL_AUDIO_FORMAT_U8:
            return ((double)p[0] - 128.0) / 127.0;
        case ESHKOL_AUDIO_FORMAT_S16: {
            int16_t s; memcpy(&s, p, 2);
            return (double)s / 32767.0;
        }
        case ESHKOL_AUDIO_FORMAT_S24: {
            int32_t s = (int32_t)((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16));
            if (s & 0x800000) s -= 0x1000000;
            return (double)s / 8388607.0;
        }
        case ESHKOL_AUDIO_FORMAT_S32: {
            int32_t s; memcpy(&s, p, 4);
            return (double)s / 2147483647.0;
        }
        default: {
            float f; memcpy(&f, p, 4);
            return (double)f;
        }
    }
}

static void fill_silence(uint8_t* p, int format, size_t bytes) {
    memset(p, format == ESHKOL_AUDIO_FORMAT_U8 ? 0x80 : 0x00, bytes);
}

/* ── ring helpers (positions in frames, ring_frames a power of two) ── */

static void ring_copy_out(eshkol_audio_device_t* d, uint64_t pos, uint8_t* dst, uint32_t frames) {
    uint32_t mask = d->ring_frames - 1u;
    uint32_t start = (uint32_t)(pos & mask);
    uint32_t first = frames < d->ring_frames - start ? frames : d->ring_frames - start;
    memcpy(dst, d->ring + (size_t)start * d->frame_bytes, (size_t)first * d->frame_bytes);
    if (frames > first)
        memcpy(dst + (size_t)first * d->frame_bytes, d->ring, (size_t)(frames - first) * d->frame_bytes);
}

static void ring_copy_in(eshkol_audio_device_t* d, uint64_t pos, const uint8_t* src, uint32_t frames) {
    uint32_t mask = d->ring_frames - 1u;
    uint32_t start = (uint32_t)(pos & mask);
    uint32_t first = frames < d->ring_frames - start ? frames : d->ring_frames - start;
    memcpy(d->ring + (size_t)start * d->frame_bytes, src, (size_t)first * d->frame_bytes);
    if (frames > first)
        memcpy(d->ring, src + (size_t)first * d->frame_bytes, (size_t)(frames - first) * d->frame_bytes);
}

/* Queue-mode callbacks: run on the audio thread. */
static void queue_output_callback(void* userdata, uint8_t* buffer, uint32_t frames) {
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)userdata;
    uint64_t r = atomic_load_explicit(&d->ring_read, memory_order_relaxed);
    uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_acquire);
    uint64_t avail = wpos - r;
    uint32_t take = avail < frames ? (uint32_t)avail : frames;
    if (take) ring_copy_out(d, r, buffer, take);
    if (take < frames) {
        fill_silence(buffer + (size_t)take * d->frame_bytes, d->config.format,
                     (size_t)(frames - take) * d->frame_bytes);
        atomic_fetch_add_explicit(&d->underruns, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&d->underrun_frames, frames - take, memory_order_relaxed);
    }
    atomic_store_explicit(&d->ring_read, r + take, memory_order_release);
}

static void queue_input_callback(void* userdata, uint8_t* buffer, uint32_t frames) {
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)userdata;
    uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&d->ring_read, memory_order_acquire);
    uint64_t space = d->ring_frames - (wpos - r);
    uint32_t put = space < frames ? (uint32_t)space : frames;
    if (put) ring_copy_in(d, wpos, buffer, put);
    if (put < frames) {
        atomic_fetch_add_explicit(&d->overruns, 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&d->overrun_frames, frames - put, memory_order_relaxed);
    }
    atomic_store_explicit(&d->ring_write, wpos + put, memory_order_release);
}

static void atomic_max_u64(atomic_uint_fast64_t* a, uint64_t v) {
    uint64_t cur = atomic_load_explicit(a, memory_order_relaxed);
    while (v > cur &&
           !atomic_compare_exchange_weak_explicit(a, &cur, v, memory_order_relaxed,
                                                  memory_order_relaxed)) {
    }
}

void eshkol_plat_audio_process(eshkol_audio_device_t* d, uint8_t* buffer, uint32_t frames) {
    uint64_t t0 = eshkol_plat_now_ns();
    if (d->last_period_start_ns && d->config.sample_rate) {
        uint64_t nominal = (uint64_t)frames * 1000000000ull / d->config.sample_rate;
        uint64_t actual = t0 - d->last_period_start_ns;
        uint64_t jitter = actual > nominal ? actual - nominal : nominal - actual;
        atomic_max_u64(&d->max_jitter_ns, jitter);
    }
    d->last_period_start_ns = t0;
    d->callback(d->userdata, buffer, frames);
    uint64_t dt = eshkol_plat_now_ns() - t0;
    atomic_store_explicit(&d->last_callback_ns, dt, memory_order_relaxed);
    atomic_max_u64(&d->max_callback_ns, dt);
    atomic_fetch_add_explicit(&d->frames, frames, memory_order_relaxed);
    atomic_fetch_add_explicit(&d->callbacks, 1u, memory_order_release);
}

/* ── lifecycle ─────────────────────────────────────────────────────── */

static int valid_config(const eshkol_audio_config_t* c) {
    return c && c->sample_rate >= 8000 && c->sample_rate <= 384000 &&
           c->channels >= 1 && c->channels <= 8 &&
           eshkol_audio_format_bytes(c->format) != 0 &&
           c->buffer_frames >= 16 && c->buffer_frames <= 16384;
}

static uint32_t next_pow2(uint32_t v) {
    uint32_t p = 1;
    while (p < v && p < (1u << 24)) p <<= 1;
    return p;
}

int eshkol_audio_open_ex(int backend, int direction, const char* device,
                         const eshkol_audio_config_t* config,
                         eshkol_audio_callback_t callback, void* userdata,
                         uint32_t ring_frames, eshkol_audio_device_t** out) {
    if (!out) return ESHKOL_PLATFORM_EINVAL;
    *out = NULL;
    if (!valid_config(config)) return ESHKOL_PLATFORM_EINVAL;
    if (direction != ESHKOL_AUDIO_OUTPUT && direction != ESHKOL_AUDIO_INPUT)
        return ESHKOL_PLATFORM_EINVAL;
    const eshkol_audio_backend_ops_t* ops = eshkol_plat_audio_ops_for(backend);
    if (!ops || !ops->available()) return ESHKOL_PLATFORM_EUNAVAILABLE;

    eshkol_audio_device_t* d = (eshkol_audio_device_t*)calloc(1, sizeof(*d));
    if (!d) return ESHKOL_PLATFORM_ENOMEM;
    d->magic = ESHKOL_PLAT_KIND_AUDIO;
    d->ops = ops;
    d->direction = direction;
    d->config = *config;
    d->frame_bytes = eshkol_audio_format_bytes(config->format) * config->channels;
    atomic_init(&d->state, ESHKOL_AUDIO_STATE_STOPPED);
    atomic_init(&d->ring_write, 0);
    atomic_init(&d->ring_read, 0);
    if (callback) {
        d->callback = callback;
        d->userdata = userdata;
    } else {
        d->queue_mode = 1;
        d->callback = direction == ESHKOL_AUDIO_OUTPUT ? queue_output_callback
                                                       : queue_input_callback;
        d->userdata = d;
    }

    int st = ops->open(d, device);
    if (st != ESHKOL_PLATFORM_OK) { free(d); return st; }
    d->frame_bytes = eshkol_audio_format_bytes(d->config.format) * d->config.channels;
    if (d->queue_mode) {
        uint32_t want = ring_frames ? ring_frames : d->config.buffer_frames * 8u;
        if (want < d->config.buffer_frames * 2u) want = d->config.buffer_frames * 2u;
        d->ring_frames = next_pow2(want);
        d->ring = (uint8_t*)calloc(d->ring_frames, d->frame_bytes);
        if (!d->ring) { ops->close(d); free(d); return ESHKOL_PLATFORM_ENOMEM; }
    }
    *out = d;
    return ESHKOL_PLATFORM_OK;
}

eshkol_audio_device_t* eshkol_audio_open_output(const eshkol_audio_config_t* config,
                                                eshkol_audio_callback_t callback,
                                                void* userdata) {
    eshkol_audio_device_t* d = NULL;
    (void)eshkol_audio_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_AUDIO),
                               ESHKOL_AUDIO_OUTPUT, NULL, config, callback, userdata, 0, &d);
    return d;
}

eshkol_audio_device_t* eshkol_audio_open_input(const eshkol_audio_config_t* config,
                                               eshkol_audio_callback_t callback,
                                               void* userdata) {
    eshkol_audio_device_t* d = NULL;
    (void)eshkol_audio_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_AUDIO),
                               ESHKOL_AUDIO_INPUT, NULL, config, callback, userdata, 0, &d);
    return d;
}

int eshkol_audio_start(eshkol_audio_device_t* d) {
    if (!d) return ESHKOL_PLATFORM_EINVAL;
    if (atomic_load(&d->state) == ESHKOL_AUDIO_STATE_RUNNING) return ESHKOL_PLATFORM_OK;
    d->last_period_start_ns = 0;
    int st = d->ops->start(d);
    if (st == ESHKOL_PLATFORM_OK) atomic_store(&d->state, ESHKOL_AUDIO_STATE_RUNNING);
    return st;
}

static int halt(eshkol_audio_device_t* d, int new_state) {
    if (!d) return ESHKOL_PLATFORM_EINVAL;
    if (atomic_load(&d->state) == ESHKOL_AUDIO_STATE_RUNNING) {
        int st = d->ops->stop(d);
        if (st != ESHKOL_PLATFORM_OK) return st;
    }
    atomic_store(&d->state, new_state);
    return ESHKOL_PLATFORM_OK;
}

int eshkol_audio_stop(eshkol_audio_device_t* d) { return halt(d, ESHKOL_AUDIO_STATE_STOPPED); }
int eshkol_audio_pause(eshkol_audio_device_t* d) { return halt(d, ESHKOL_AUDIO_STATE_PAUSED); }

void eshkol_audio_close(eshkol_audio_device_t* d) {
    if (!d) return;
    (void)halt(d, ESHKOL_AUDIO_STATE_STOPPED);
    d->ops->close(d);
    free(d->ring);
    d->magic = 0;
    free(d);
}

/* ── queue mode (owner thread) ─────────────────────────────────────── */

bool eshkol_audio_queue(eshkol_audio_device_t* d, const void* data, uint32_t frames) {
    if (!d || !d->queue_mode || d->direction != ESHKOL_AUDIO_OUTPUT || (!data && frames))
        return false;
    uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&d->ring_read, memory_order_acquire);
    if (d->ring_frames - (wpos - r) < frames) return false;
    ring_copy_in(d, wpos, (const uint8_t*)data, frames);
    atomic_store_explicit(&d->ring_write, wpos + frames, memory_order_release);
    d->stage_write = wpos + frames;
    d->stage_sample = 0;
    return true;
}

uint32_t eshkol_audio_queued_frames(eshkol_audio_device_t* d) {
    if (!d || !d->queue_mode) return 0;
    uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_acquire);
    uint64_t r = atomic_load_explicit(&d->ring_read, memory_order_acquire);
    return (uint32_t)(wpos - r);
}

uint32_t eshkol_audio_dequeue(eshkol_audio_device_t* d, void* data, uint32_t frames) {
    if (!d || !d->queue_mode || d->direction != ESHKOL_AUDIO_INPUT || !data) return 0;
    uint64_t r = atomic_load_explicit(&d->ring_read, memory_order_relaxed);
    uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_acquire);
    uint32_t take = wpos - r < frames ? (uint32_t)(wpos - r) : frames;
    if (take) ring_copy_out(d, r, (uint8_t*)data, take);
    atomic_store_explicit(&d->ring_read, r + take, memory_order_release);
    d->read_pos_private = r + take;
    d->read_sample = 0;
    return take;
}

int eshkol_audio_get_stats(eshkol_audio_device_t* d, eshkol_audio_stats_t* s) {
    if (!d || !s) return ESHKOL_PLATFORM_EINVAL;
    memset(s, 0, sizeof(*s));
    s->callbacks = atomic_load_explicit(&d->callbacks, memory_order_acquire);
    s->frames = atomic_load(&d->frames);
    s->underruns = atomic_load(&d->underruns);
    s->underrun_frames = atomic_load(&d->underrun_frames);
    s->overruns = atomic_load(&d->overruns);
    s->overrun_frames = atomic_load(&d->overrun_frames);
    s->device_xruns = atomic_load(&d->device_xruns);
    s->last_callback_ns = atomic_load(&d->last_callback_ns);
    s->max_callback_ns = atomic_load(&d->max_callback_ns);
    s->max_period_jitter_ns = atomic_load(&d->max_jitter_ns);
    s->device_latency_frames = atomic_load(&d->device_latency_frames);
    s->queued_frames = eshkol_audio_queued_frames(d);
    s->period_frames = d->config.buffer_frames;
    s->sample_rate = d->config.sample_rate;
    return ESHKOL_PLATFORM_OK;
}

int eshkol_audio_get_config(eshkol_audio_device_t* d, eshkol_audio_config_t* out) {
    if (!d || !out) return ESHKOL_PLATFORM_EINVAL;
    *out = d->config;
    return ESHKOL_PLATFORM_OK;
}

int eshkol_audio_backend_of(eshkol_audio_device_t* d) {
    return d ? d->ops->backend : ESHKOL_BACKEND_NONE;
}

/* ── Eshkol language bridge ────────────────────────────────────────── */

#define WITH_AUDIO(handle)                                                    \
    uint64_t lease_ = 0; int st_ = 0;                                         \
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)eshkol_plat_handle_borrow( \
        (handle), ESHKOL_PLAT_KIND_AUDIO, &lease_, &st_);                     \
    if (!d) return st_;
#define END_AUDIO(handle) eshkol_plat_handle_release((handle), lease_)

int64_t eshkol_plat_audio_open(int32_t direction, int32_t sample_rate, int32_t channels,
                               int32_t format, int32_t period_frames, int32_t ring_frames) {
    if (sample_rate <= 0 || channels <= 0 || channels > 8 || period_frames <= 0 ||
        ring_frames < 0)
        return ESHKOL_PLATFORM_EINVAL;
    eshkol_audio_config_t cfg;
    cfg.sample_rate = (uint32_t)sample_rate;
    cfg.channels = (uint8_t)channels;
    cfg.format = (eshkol_audio_format_t)format;
    cfg.buffer_frames = (uint32_t)period_frames;
    eshkol_audio_device_t* d = NULL;
    int st = eshkol_audio_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_AUDIO), direction,
                                  NULL, &cfg, NULL, NULL, (uint32_t)ring_frames, &d);
    if (st != ESHKOL_PLATFORM_OK) return st;
    int64_t id = eshkol_plat_handle_adopt(ESHKOL_PLAT_KIND_AUDIO, d);
    if (id < 0) eshkol_audio_close(d);
    return id;
}

int32_t eshkol_plat_audio_close(int64_t handle) {
    return eshkol_plat_handle_close(handle);
}

int32_t eshkol_plat_audio_control(int64_t handle, int32_t op) {
    WITH_AUDIO(handle)
    int r = op == 0 ? eshkol_audio_start(d) : op == 1 ? eshkol_audio_stop(d)
          : op == 2 ? eshkol_audio_pause(d) : ESHKOL_PLATFORM_EINVAL;
    END_AUDIO(handle);
    return r;
}

int32_t eshkol_plat_audio_write(int64_t handle, double sample) {
    WITH_AUDIO(handle)
    int r = ESHKOL_PLATFORM_OK;
    if (!d->queue_mode || d->direction != ESHKOL_AUDIO_OUTPUT) {
        r = ESHKOL_PLATFORM_ESTATE;
    } else {
        uint64_t published = atomic_load_explicit(&d->ring_write, memory_order_relaxed);
        if (d->stage_write < published) { d->stage_write = published; d->stage_sample = 0; }
        uint64_t rpos = atomic_load_explicit(&d->ring_read, memory_order_acquire);
        if (d->stage_write - rpos >= d->ring_frames) {
            r = ESHKOL_PLATFORM_EFULL;
        } else {
            uint32_t bps = eshkol_audio_format_bytes(d->config.format);
            uint8_t* frame = d->ring + (size_t)(d->stage_write & (d->ring_frames - 1u)) * d->frame_bytes;
            store_sample(frame + (size_t)d->stage_sample * bps, d->config.format, sample);
            if (++d->stage_sample == d->config.channels) {
                d->stage_sample = 0;
                d->stage_write++;
            }
        }
    }
    END_AUDIO(handle);
    return r;
}

int32_t eshkol_plat_audio_commit(int64_t handle) {
    WITH_AUDIO(handle)
    int32_t r = 0;
    if (!d->queue_mode || d->direction != ESHKOL_AUDIO_OUTPUT) {
        r = ESHKOL_PLATFORM_ESTATE;
    } else {
        uint64_t published = atomic_load_explicit(&d->ring_write, memory_order_relaxed);
        if (d->stage_write > published) {
            r = (int32_t)(d->stage_write - published);
            atomic_store_explicit(&d->ring_write, d->stage_write, memory_order_release);
        }
    }
    END_AUDIO(handle);
    return r;
}

int32_t eshkol_plat_audio_available(int64_t handle) {
    WITH_AUDIO(handle)
    int32_t r = 0;
    if (!d->queue_mode || d->direction != ESHKOL_AUDIO_INPUT) {
        r = ESHKOL_PLATFORM_ESTATE;
    } else {
        uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_acquire);
        uint64_t rpos = atomic_load_explicit(&d->ring_read, memory_order_relaxed);
        r = (int32_t)(wpos - rpos);
    }
    END_AUDIO(handle);
    return r;
}

double eshkol_plat_audio_read(int64_t handle) {
    uint64_t lease_ = 0; int st_ = 0;
    eshkol_audio_device_t* d = (eshkol_audio_device_t*)eshkol_plat_handle_borrow(
        handle, ESHKOL_PLAT_KIND_AUDIO, &lease_, &st_);
    if (!d) return NAN;
    double v = NAN;
    if (d->queue_mode && d->direction == ESHKOL_AUDIO_INPUT) {
        uint64_t rpos = atomic_load_explicit(&d->ring_read, memory_order_relaxed);
        uint64_t wpos = atomic_load_explicit(&d->ring_write, memory_order_acquire);
        if (wpos > rpos) {
            uint32_t bps = eshkol_audio_format_bytes(d->config.format);
            const uint8_t* frame = d->ring + (size_t)(rpos & (d->ring_frames - 1u)) * d->frame_bytes;
            v = load_sample(frame + (size_t)d->read_sample * bps, d->config.format);
            if (++d->read_sample == d->config.channels) {
                d->read_sample = 0;
                atomic_store_explicit(&d->ring_read, rpos + 1, memory_order_release);
            }
        }
    }
    eshkol_plat_handle_release(handle, lease_);
    return v;
}

int64_t eshkol_plat_audio_stat(int64_t handle, int32_t field) {
    WITH_AUDIO(handle)
    eshkol_audio_stats_t s;
    eshkol_audio_get_stats(d, &s);
    int64_t r;
    switch (field) {
        case 0:  r = (int64_t)s.callbacks; break;
        case 1:  r = (int64_t)s.frames; break;
        case 2:  r = (int64_t)s.underruns; break;
        case 3:  r = (int64_t)s.underrun_frames; break;
        case 4:  r = (int64_t)s.overruns; break;
        case 5:  r = (int64_t)s.overrun_frames; break;
        case 6:  r = (int64_t)s.device_xruns; break;
        case 7:  r = (int64_t)s.last_callback_ns; break;
        case 8:  r = (int64_t)s.max_callback_ns; break;
        case 9:  r = (int64_t)s.max_period_jitter_ns; break;
        case 10: r = s.device_latency_frames; break;
        case 11: r = s.queued_frames; break;
        case 12: r = s.period_frames; break;
        case 13: r = s.sample_rate; break;
        case 14: r = d->ops->backend; break;
        default: r = ESHKOL_PLATFORM_EINVAL; break;
    }
    END_AUDIO(handle);
    return r;
}
