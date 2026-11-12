/**
 * @file audio.h
 * @brief Real-time audio with callback-based I/O (ADR-0053; interface of
 *        docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md section 8.3).
 *
 * Backends: null (a clock-driven real-time thread with no device; always
 * present), ALSA (libasound loaded at run time), CoreAudio (AudioToolbox
 * loaded at run time).
 *
 * Real-time contract. The backend's audio thread calls the device callback
 * once per period with `frames` interleaved frames in the configured format.
 * The callback must not block, allocate or take locks. When no callback is
 * given the device runs in queue mode: the built-in callback moves frames
 * between the device and a lock-free single-producer/single-consumer ring.
 * Output underruns (the ring had fewer frames than the period) are filled
 * with silence and counted; input overruns (the ring was full) drop the newest
 * frames and are counted. Every counter is read with atomic loads, so
 * eshkol_audio_get_stats() is safe while the stream runs.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_AUDIO_H
#define ESHKOL_PLATFORM_AUDIO_H

#include <stdbool.h>
#include <stdint.h>
#include <eshkol/platform/platform.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_audio_device eshkol_audio_device_t;

typedef enum {
    ESHKOL_AUDIO_FORMAT_U8  = 0,  /**< unsigned 8-bit */
    ESHKOL_AUDIO_FORMAT_S16 = 1,  /**< signed 16-bit, native endian */
    ESHKOL_AUDIO_FORMAT_S24 = 2,  /**< signed 24-bit packed in 3 bytes, little endian */
    ESHKOL_AUDIO_FORMAT_S32 = 3,  /**< signed 32-bit, native endian */
    ESHKOL_AUDIO_FORMAT_F32 = 4   /**< IEEE float 32-bit, native endian */
} eshkol_audio_format_t;

typedef enum {
    ESHKOL_AUDIO_OUTPUT = 0,
    ESHKOL_AUDIO_INPUT  = 1
} eshkol_audio_direction_t;

typedef struct {
    uint32_t sample_rate;     /**< frames per second, 8000..384000 */
    uint8_t  channels;        /**< 1..8 */
    eshkol_audio_format_t format;
    uint32_t buffer_frames;   /**< frames per period (callback size), 16..16384 */
} eshkol_audio_config_t;

/** Called on the audio thread once per period. */
typedef void (*eshkol_audio_callback_t)(void* userdata, uint8_t* buffer,
                                        uint32_t frames);

typedef struct {
    uint64_t callbacks;          /**< periods processed */
    uint64_t frames;             /**< frames moved to/from the device */
    uint64_t underruns;          /**< output periods that ran short in queue mode */
    uint64_t underrun_frames;    /**< silence frames inserted */
    uint64_t overruns;           /**< input periods that found the ring full */
    uint64_t overrun_frames;     /**< input frames dropped */
    uint64_t device_xruns;       /**< xruns reported by the OS device */
    uint64_t last_callback_ns;   /**< duration of the last callback */
    uint64_t max_callback_ns;    /**< longest callback */
    uint64_t max_period_jitter_ns; /**< largest |actual - nominal| period start spacing */
    uint32_t device_latency_frames; /**< latency the backend reports */
    uint32_t queued_frames;      /**< frames waiting in the queue-mode ring */
    uint32_t period_frames;      /**< negotiated frames per period */
    uint32_t sample_rate;        /**< negotiated sample rate */
} eshkol_audio_stats_t;

/* ── Device management ─────────────────────────────────────────────── */
eshkol_audio_device_t* eshkol_audio_open_output(const eshkol_audio_config_t* config,
                                                eshkol_audio_callback_t callback,
                                                void* userdata);
eshkol_audio_device_t* eshkol_audio_open_input(const eshkol_audio_config_t* config,
                                               eshkol_audio_callback_t callback,
                                               void* userdata);
/**
 * Open on an explicit backend. `device` names an OS device (NULL = default;
 * ALSA accepts PCM names such as "default" or "null"). `ring_frames` sizes the
 * queue-mode ring (0 = 8 periods). Returns a status code.
 */
int eshkol_audio_open_ex(int backend, int direction, const char* device,
                         const eshkol_audio_config_t* config,
                         eshkol_audio_callback_t callback, void* userdata,
                         uint32_t ring_frames, eshkol_audio_device_t** out);
void eshkol_audio_close(eshkol_audio_device_t* device);

/* ── Playback control ──────────────────────────────────────────────── */
int eshkol_audio_start(eshkol_audio_device_t* device);
int eshkol_audio_stop(eshkol_audio_device_t* device);
int eshkol_audio_pause(eshkol_audio_device_t* device);

/* ── Queue mode ────────────────────────────────────────────────────── */
/** Append frames (config format) to the output ring; false when they do not fit. */
bool eshkol_audio_queue(eshkol_audio_device_t* device, const void* data,
                        uint32_t frames);
uint32_t eshkol_audio_queued_frames(eshkol_audio_device_t* device);
/** Take up to `frames` captured frames (config format); returns frames taken. */
uint32_t eshkol_audio_dequeue(eshkol_audio_device_t* device, void* data,
                              uint32_t frames);

int eshkol_audio_get_stats(eshkol_audio_device_t* device, eshkol_audio_stats_t* out);
int eshkol_audio_get_config(eshkol_audio_device_t* device, eshkol_audio_config_t* out);
int eshkol_audio_backend_of(eshkol_audio_device_t* device);
/** Bytes per sample of a format, 0 for an unknown format. */
uint32_t eshkol_audio_format_bytes(int format);

/* ── Eshkol language bridge (generation-checked integer handles) ───── */
/** Open in queue mode on the selected backend. */
int64_t eshkol_plat_audio_open(int32_t direction, int32_t sample_rate,
                               int32_t channels, int32_t format,
                               int32_t period_frames, int32_t ring_frames);
int32_t eshkol_plat_audio_close(int64_t handle);
/** op: 0 start, 1 stop, 2 pause. */
int32_t eshkol_plat_audio_control(int64_t handle, int32_t op);
/** Stage one interleaved output sample in [-1, 1]; visible after commit. */
int32_t eshkol_plat_audio_write(int64_t handle, double sample);
/** Publish staged samples to the audio thread; returns frames published. */
int32_t eshkol_plat_audio_commit(int64_t handle);
/** Captured frames ready for reading. */
int32_t eshkol_plat_audio_available(int64_t handle);
/** Next captured sample in [-1, 1] (interleaved); NaN when none. */
double  eshkol_plat_audio_read(int64_t handle);
/** field: 0 callbacks, 1 frames, 2 underruns, 3 underrun-frames, 4 overruns,
 *  5 overrun-frames, 6 device-xruns, 7 last-callback-ns, 8 max-callback-ns,
 *  9 max-jitter-ns, 10 device-latency-frames, 11 queued-frames,
 *  12 period-frames, 13 sample-rate, 14 backend. */
int64_t eshkol_plat_audio_stat(int64_t handle, int32_t field);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_AUDIO_H */
