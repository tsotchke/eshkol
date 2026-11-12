/*
 * platform_internal.h — shared internals of the Eshkol platform layer
 * (ADR-0053): synchronisation, run-time library binding, backend tables,
 * the window/audio/MIDI object layouts and the lock-free rings.
 *
 * The layer targets POSIX hosts (Linux, BSD, macOS). Win32 windows, WASAPI
 * and WinMM are Planned; CMake does not compile this directory on Windows.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_INTERNAL_H
#define ESHKOL_PLATFORM_INTERNAL_H

#if defined(_WIN32)
#error "the Eshkol platform layer is compiled on POSIX hosts; Win32 backends are Planned"
#endif

#include <eshkol/platform/platform.h>
#include <eshkol/platform/event.h>
#include <eshkol/platform/window.h>
#include <eshkol/platform/audio.h>
#include <eshkol/platform/midi.h>

#if defined(__has_include) && !defined(__APPLE__)
#  if __has_include(<X11/Xlib.h>) && __has_include(<X11/Xutil.h>)
#    define ESHKOL_PLATFORM_HAVE_X11 1
#  endif
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Clock / sleep ─────────────────────────────────────────────────── */
uint64_t eshkol_plat_now_ns(void);
/** Sleep until the monotonic deadline (ns). */
void eshkol_plat_sleep_until_ns(uint64_t deadline_ns);
void eshkol_plat_sleep_ms(int ms);

/* ── Run-time library binding ──────────────────────────────────────── */
/** dlopen the first loadable name (RTLD_NOW|RTLD_LOCAL); NULL if none. */
void* eshkol_plat_dlopen_any(const char* const* names);
/** Resolve every symbol of a NULL-terminated name list into `slots`;
 *  returns 1 when all resolved. */
int eshkol_plat_dlsym_all(void* lib, const char* const* names, void** slots);

/* ── Backend selection ─────────────────────────────────────────────── */
/** Selected backend for a subsystem, resolving `auto` once. */
int eshkol_plat_selected(int subsystem);

/* ── Handle registry (generation-checked ids for the Eshkol bridge) ─── */
#include <eshkol/core/handle.h>
enum { ESHKOL_PLAT_KIND_WINDOW = ESHKOL_HANDLE_KIND_WINDOW,
       ESHKOL_PLAT_KIND_AUDIO = ESHKOL_HANDLE_KIND_AUDIO_DEVICE,
       ESHKOL_PLAT_KIND_MIDI = ESHKOL_HANDLE_KIND_MIDI_PORT };
/** Adopt `object`; returns a positive id or a negative status. */
int64_t eshkol_plat_handle_adopt(uint32_t kind, void* object);
/** Borrow the object behind `id`; returns NULL (and sets *status) on failure. */
void* eshkol_plat_handle_borrow(int64_t id, uint32_t kind, uint64_t* lease, int* status);
void  eshkol_plat_handle_release(int64_t id, uint64_t lease);
/** Close `id` (runs the kind's destructor once). */
int   eshkol_plat_handle_close(int64_t id);

/* ── Event queue ───────────────────────────────────────────────────── */
#define ESHKOL_PLAT_EVENT_QUEUE_CAP 1024u
typedef struct {
    pthread_mutex_t lock;
    pthread_cond_t  ready;
    eshkol_event_t  items[ESHKOL_PLAT_EVENT_QUEUE_CAP];
    uint32_t head, count;
    uint64_t dropped;
} eshkol_plat_event_queue_t;

int  eshkol_plat_queue_init(eshkol_plat_event_queue_t* q);
void eshkol_plat_queue_destroy(eshkol_plat_event_queue_t* q);
int  eshkol_plat_queue_push(eshkol_plat_event_queue_t* q, const eshkol_event_t* ev);
bool eshkol_plat_queue_pop(eshkol_plat_event_queue_t* q, eshkol_event_t* ev);
/** Wait up to timeout_ms (< 0 forever) for a queued event; 1 ready, 0 timeout. */
int  eshkol_plat_queue_wait(eshkol_plat_event_queue_t* q, int timeout_ms);
uint64_t eshkol_plat_queue_dropped(eshkol_plat_event_queue_t* q);

/* ── Windows ───────────────────────────────────────────────────────── */
typedef struct eshkol_window_backend_ops {
    int backend;
    int  (*available)(void);
    int  (*create)(eshkol_window_t* w, const char* title);
    void (*destroy)(eshkol_window_t* w);
    int  (*set_size)(eshkol_window_t* w, int width, int height);
    int  (*set_position)(eshkol_window_t* w, int x, int y);
    int  (*set_title)(eshkol_window_t* w, const char* title);
    int  (*present)(eshkol_window_t* w);
    int  (*read_presented)(eshkol_window_t* w, int x, int y, uint32_t* out);
    /** Move pending OS events into w->queue without blocking. */
    void (*pump)(eshkol_window_t* w);
    /** Block until the OS has input for w or timeout_ms passes (< 0 forever). */
    void (*wait_os)(eshkol_window_t* w, int timeout_ms);
} eshkol_window_backend_ops_t;

struct eshkol_window {
    uint32_t magic;                  /* ESHKOL_PLAT_KIND_WINDOW: kind tag checked on borrow */
    const eshkol_window_backend_ops_t* ops;
    void* impl;
    uint32_t id;
    uint32_t flags;
    int x, y, width, height;
    atomic_int should_close;
    eshkol_surface_t surface;
    atomic_uint_fast64_t presents;
    eshkol_plat_event_queue_t queue;
    eshkol_event_t current;          /* Eshkol bridge slot */
};

extern const eshkol_window_backend_ops_t eshkol_plat_window_offscreen_ops;
#if defined(ESHKOL_PLATFORM_HAVE_X11)
extern const eshkol_window_backend_ops_t eshkol_plat_window_x11_ops;
#endif
#if defined(__APPLE__)
extern const eshkol_window_backend_ops_t eshkol_plat_window_cocoa_ops;
#endif
const eshkol_window_backend_ops_t* eshkol_plat_window_ops_for(int backend);

/** Reallocate the surface to width x height, keeping the overlapping pixels. */
int  eshkol_plat_window_resize_surface(eshkol_window_t* w, int width, int height);
/** Queue an event originating from the backend (fills window_id/timestamp). */
void eshkol_plat_window_emit(eshkol_window_t* w, eshkol_event_t* ev);

/* ── Audio ─────────────────────────────────────────────────────────── */
typedef struct eshkol_audio_backend_ops {
    int backend;
    int  (*available)(void);
    /** Open the OS device; may adjust dev->config (period/rate) to what the
     *  device granted. */
    int  (*open)(eshkol_audio_device_t* dev, const char* device_name);
    int  (*start)(eshkol_audio_device_t* dev);
    int  (*stop)(eshkol_audio_device_t* dev);
    void (*close)(eshkol_audio_device_t* dev);
} eshkol_audio_backend_ops_t;

enum { ESHKOL_AUDIO_STATE_STOPPED = 0, ESHKOL_AUDIO_STATE_RUNNING = 1,
       ESHKOL_AUDIO_STATE_PAUSED = 2 };

struct eshkol_audio_device {
    uint32_t magic;                  /* ESHKOL_PLAT_KIND_AUDIO */
    const eshkol_audio_backend_ops_t* ops;
    void* impl;
    int direction;
    eshkol_audio_config_t config;
    uint32_t frame_bytes;
    eshkol_audio_callback_t callback;
    void* userdata;
    int queue_mode;
    atomic_int state;

    /* queue-mode ring: ring_frames is a power of two; positions count frames */
    uint8_t* ring;
    uint32_t ring_frames;
    _Alignas(64) atomic_uint_fast64_t ring_write;
    _Alignas(64) atomic_uint_fast64_t ring_read;

    /* Eshkol bridge cursors (owner thread only) */
    uint64_t stage_write;            /* staged, unpublished write position */
    uint32_t stage_sample;           /* samples written into the staged frame */
    uint64_t read_pos_private;       /* consumer position for bridge reads */
    uint32_t read_sample;            /* sample index inside the current frame */

    /* counters (written by the audio thread, read anywhere) */
    _Alignas(64) atomic_uint_fast64_t callbacks;
    atomic_uint_fast64_t frames;
    atomic_uint_fast64_t underruns, underrun_frames;
    atomic_uint_fast64_t overruns, overrun_frames;
    atomic_uint_fast64_t device_xruns;
    atomic_uint_fast64_t last_callback_ns, max_callback_ns, max_jitter_ns;
    atomic_uint device_latency_frames;
    uint64_t last_period_start_ns;   /* audio thread only */
};

extern const eshkol_audio_backend_ops_t eshkol_plat_audio_null_ops;
#if defined(__linux__)
extern const eshkol_audio_backend_ops_t eshkol_plat_audio_alsa_ops;
#endif
#if defined(__APPLE__)
extern const eshkol_audio_backend_ops_t eshkol_plat_audio_coreaudio_ops;
#endif
const eshkol_audio_backend_ops_t* eshkol_plat_audio_ops_for(int backend);

/** Run one period on the audio thread: timing, callback, counters. */
void eshkol_plat_audio_process(eshkol_audio_device_t* dev, uint8_t* buffer,
                               uint32_t frames);

/* ── MIDI ──────────────────────────────────────────────────────────── */
typedef struct eshkol_midi_backend_ops {
    int backend;
    int  (*available)(void);
    int  (*open)(eshkol_midi_port_t* port);
    void (*close)(eshkol_midi_port_t* port);
    int  (*send)(eshkol_midi_port_t* port, const uint8_t* bytes, uint32_t len);
    int  (*source_count)(void);
    int  (*source_name)(int index, char* buf, uint32_t cap);
} eshkol_midi_backend_ops_t;

#define ESHKOL_PLAT_MIDI_RING_BYTES 65536u

struct eshkol_midi_port {
    uint32_t magic;                  /* ESHKOL_PLAT_KIND_MIDI */
    const eshkol_midi_backend_ops_t* ops;
    void* impl;
    int direction;
    char name[128];

    /* inbound ring of records [u32 len][u64 timestamp][len bytes], 4-aligned */
    uint8_t* ring;
    _Alignas(64) atomic_uint_fast64_t ring_write;
    _Alignas(64) atomic_uint_fast64_t ring_read;

    /* producer-side stream parser (running status, SysEx) */
    uint8_t parse_buf[ESHKOL_MIDI_MAX_MESSAGE];
    uint32_t parse_len, parse_expected;
    uint8_t running_status;
    int in_sysex;

    _Alignas(64) atomic_uint_fast64_t sent, received, dropped, malformed;

    /* Eshkol bridge slots (owner thread only) */
    uint8_t stage[ESHKOL_MIDI_MAX_MESSAGE];
    uint32_t stage_len;
    uint8_t current[ESHKOL_MIDI_MAX_MESSAGE];
    int32_t current_len;
    uint64_t current_ts;
};

extern const eshkol_midi_backend_ops_t eshkol_plat_midi_loopback_ops;
#if defined(__linux__)
extern const eshkol_midi_backend_ops_t eshkol_plat_midi_alsa_ops;
#endif
#if defined(__APPLE__)
extern const eshkol_midi_backend_ops_t eshkol_plat_midi_coremidi_ops;
#endif
const eshkol_midi_backend_ops_t* eshkol_plat_midi_ops_for(int backend);

/** Deliver one complete message into an input port's ring (producer side). */
void eshkol_plat_midi_deliver(eshkol_midi_port_t* port, const uint8_t* msg,
                              uint32_t len, uint64_t timestamp_us);
/** Feed a raw byte stream (running status / split SysEx allowed). */
void eshkol_plat_midi_feed(eshkol_midi_port_t* port, const uint8_t* bytes,
                           uint32_t len, uint64_t timestamp_us);
/** Validate one complete message; returns 1 when well formed. */
int eshkol_plat_midi_valid(const uint8_t* bytes, uint32_t len);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_INTERNAL_H */
