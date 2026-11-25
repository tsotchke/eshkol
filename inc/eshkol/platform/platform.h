/**
 * @file platform.h
 * @brief Eshkol platform layer: one C ABI for windows, events, real-time
 *        audio and MIDI across operating systems (ADR-0053).
 *
 * Every native backend (X11, Cocoa, ALSA, CoreAudio, ALSA sequencer,
 * CoreMIDI) is bound at run time with the system dynamic loader, so a program
 * that never opens a window or device acquires no link-time or load-time
 * dependency, and a binary built on one host runs on a host without those
 * libraries. Each subsystem also has a headless backend (offscreen display,
 * null audio clock, MIDI loopback) that exercises the same public paths and
 * is what the test suites run against.
 *
 * Backend selection per subsystem:
 *   ESHKOL_PLATFORM_DISPLAY = auto | offscreen | x11 | cocoa
 *   ESHKOL_PLATFORM_AUDIO   = auto | null | alsa | coreaudio
 *   ESHKOL_PLATFORM_MIDI    = auto | loopback | alsa | coremidi
 * `auto` (the default) selects the native backend whose system library loads
 * and otherwise the headless backend. An explicitly named backend that is not
 * available is reported as ESHKOL_PLATFORM_EUNAVAILABLE; it never silently
 * falls back.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_PLATFORM_H
#define ESHKOL_PLATFORM_PLATFORM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ESHKOL_PLATFORM_ABI_VERSION 1

/** Status codes shared by every platform entry point (0 = success). */
typedef enum {
    ESHKOL_PLATFORM_OK           = 0,
    ESHKOL_PLATFORM_EINVAL       = -1,  /**< invalid argument */
    ESHKOL_PLATFORM_ENOMEM       = -2,  /**< allocation failed */
    ESHKOL_PLATFORM_EUNAVAILABLE = -3,  /**< backend not compiled in or its library did not load */
    ESHKOL_PLATFORM_EDEVICE      = -4,  /**< the OS or device refused the request */
    ESHKOL_PLATFORM_ESTATE       = -5,  /**< operation not valid in the current state */
    ESHKOL_PLATFORM_EBUSY        = -6,  /**< handle is in use by another operation */
    ESHKOL_PLATFORM_ENOTFOUND    = -7,  /**< unknown handle, endpoint or name */
    ESHKOL_PLATFORM_EFULL        = -8,  /**< bounded queue is full */
    ESHKOL_PLATFORM_EFORMAT      = -9   /**< malformed data (e.g. not a complete MIDI message) */
} eshkol_platform_status_t;

/** Subsystems with independently selected backends. */
typedef enum {
    ESHKOL_PLATFORM_DISPLAY = 0,
    ESHKOL_PLATFORM_AUDIO   = 1,
    ESHKOL_PLATFORM_MIDI    = 2
} eshkol_platform_subsystem_t;

/** Backend identifiers. Values are stable ABI. */
typedef enum {
    ESHKOL_BACKEND_NONE       = 0,
    /* display */
    ESHKOL_BACKEND_OFFSCREEN  = 1,
    ESHKOL_BACKEND_X11        = 2,
    ESHKOL_BACKEND_COCOA      = 3,
    ESHKOL_BACKEND_WAYLAND    = 4,   /**< Planned */
    ESHKOL_BACKEND_WIN32      = 5,   /**< Planned */
    /* audio */
    ESHKOL_BACKEND_AUDIO_NULL = 16,
    ESHKOL_BACKEND_ALSA       = 17,
    ESHKOL_BACKEND_COREAUDIO  = 18,
    ESHKOL_BACKEND_WASAPI     = 19,  /**< Planned */
    /* midi */
    ESHKOL_BACKEND_LOOPBACK   = 32,
    ESHKOL_BACKEND_ALSA_SEQ   = 33,
    ESHKOL_BACKEND_COREMIDI   = 34,
    ESHKOL_BACKEND_WINMM      = 35   /**< Planned */
} eshkol_platform_backend_t;

/** Initialise the layer (idempotent, thread-safe). Returns ESHKOL_PLATFORM_OK. */
int eshkol_platform_init(void);
/** Release process-wide platform state. Open handles must be closed first. */
void eshkol_platform_quit(void);

/** Monotonic clock in microseconds, the timestamp base of every event. */
uint64_t eshkol_platform_time_us(void);

/** Backend currently selected for a subsystem (resolving `auto`). */
int eshkol_platform_backend(int subsystem);
/** Select a backend explicitly (overrides the environment). */
int eshkol_platform_select_backend(int subsystem, int backend);
/** 1 when the backend is compiled into this build, else 0. */
int eshkol_platform_backend_compiled(int backend);
/** 1 when the backend is compiled in and its system library loads. */
int eshkol_platform_backend_available(int backend);
/** Stable lower-case backend name ("offscreen", "x11", "alsa", ...). */
const char* eshkol_platform_backend_name(int backend);
/** Human-readable text for a status code. */
const char* eshkol_platform_status_string(int status);

/* ── Eshkol language bridge ────────────────────────────────────────── */
/** Selected backend for a subsystem. */
int32_t eshkol_plat_backend(int32_t subsystem);
/** Backend name as a NUL-terminated static string. */
const char* eshkol_plat_backend_name(int32_t backend);
/** Select a backend by name ("auto" re-resolves); returns a status code. */
int32_t eshkol_plat_select_backend(int32_t subsystem, const char* name);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_PLATFORM_H */
