/*
 * platform.c — process-wide state of the Eshkol platform layer (ADR-0053):
 * monotonic clock, run-time library binding, backend selection and the
 * generation-checked handle registry used by the Eshkol language bridge.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_POSIX_C_SOURCE) && !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "platform_internal.h"

#include <dlfcn.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ── Clock ─────────────────────────────────────────────────────────── */

uint64_t eshkol_plat_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

uint64_t eshkol_platform_time_us(void) {
    return eshkol_plat_now_ns() / 1000ull;
}

void eshkol_plat_sleep_until_ns(uint64_t deadline_ns) {
    for (;;) {
        uint64_t now = eshkol_plat_now_ns();
        if (now >= deadline_ns) return;
        uint64_t left = deadline_ns - now;
        struct timespec req = { (time_t)(left / 1000000000ull),
                                (long)(left % 1000000000ull) };
        if (nanosleep(&req, NULL) == 0) return;
        if (errno != EINTR) return;
    }
}

void eshkol_plat_sleep_ms(int ms) {
    if (ms <= 0) return;
    eshkol_plat_sleep_until_ns(eshkol_plat_now_ns() + (uint64_t)ms * 1000000ull);
}

/* ── Run-time library binding ──────────────────────────────────────── */

void* eshkol_plat_dlopen_any(const char* const* names) {
    for (const char* const* n = names; n && *n; ++n) {
        void* h = dlopen(*n, RTLD_NOW | RTLD_LOCAL);
        if (h) return h;
    }
    return NULL;
}

int eshkol_plat_dlsym_all(void* lib, const char* const* names, void** slots) {
    if (!lib) return 0;
    for (size_t i = 0; names[i]; ++i) {
        slots[i] = dlsym(lib, names[i]);
        if (!slots[i]) return 0;
    }
    return 1;
}

/* ── Backend names and compiled set ────────────────────────────────── */

const char* eshkol_platform_backend_name(int backend) {
    switch (backend) {
        case ESHKOL_BACKEND_NONE:       return "none";
        case ESHKOL_BACKEND_OFFSCREEN:  return "offscreen";
        case ESHKOL_BACKEND_X11:        return "x11";
        case ESHKOL_BACKEND_COCOA:      return "cocoa";
        case ESHKOL_BACKEND_WAYLAND:    return "wayland";
        case ESHKOL_BACKEND_WIN32:      return "win32";
        case ESHKOL_BACKEND_AUDIO_NULL: return "null";
        case ESHKOL_BACKEND_ALSA:       return "alsa";
        case ESHKOL_BACKEND_COREAUDIO:  return "coreaudio";
        case ESHKOL_BACKEND_WASAPI:     return "wasapi";
        case ESHKOL_BACKEND_LOOPBACK:   return "loopback";
        case ESHKOL_BACKEND_ALSA_SEQ:   return "alsa";
        case ESHKOL_BACKEND_COREMIDI:   return "coremidi";
        case ESHKOL_BACKEND_WINMM:      return "winmm";
        default:                        return "unknown";
    }
}

const char* eshkol_platform_status_string(int status) {
    switch (status) {
        case ESHKOL_PLATFORM_OK:           return "ok";
        case ESHKOL_PLATFORM_EINVAL:       return "invalid argument";
        case ESHKOL_PLATFORM_ENOMEM:       return "out of memory";
        case ESHKOL_PLATFORM_EUNAVAILABLE: return "backend unavailable";
        case ESHKOL_PLATFORM_EDEVICE:      return "device error";
        case ESHKOL_PLATFORM_ESTATE:       return "invalid state";
        case ESHKOL_PLATFORM_EBUSY:        return "handle busy";
        case ESHKOL_PLATFORM_ENOTFOUND:    return "not found";
        case ESHKOL_PLATFORM_EFULL:        return "queue full";
        case ESHKOL_PLATFORM_EFORMAT:      return "malformed data";
        default:                           return "unknown status";
    }
}

static int subsystem_of(int backend) {
    if (backend >= ESHKOL_BACKEND_OFFSCREEN && backend <= ESHKOL_BACKEND_WIN32)
        return ESHKOL_PLATFORM_DISPLAY;
    if (backend >= ESHKOL_BACKEND_AUDIO_NULL && backend <= ESHKOL_BACKEND_WASAPI)
        return ESHKOL_PLATFORM_AUDIO;
    if (backend >= ESHKOL_BACKEND_LOOPBACK && backend <= ESHKOL_BACKEND_WINMM)
        return ESHKOL_PLATFORM_MIDI;
    return -1;
}

static int backend_available_flag(int backend, int need_library) {
    switch (subsystem_of(backend)) {
        case ESHKOL_PLATFORM_DISPLAY: {
            const eshkol_window_backend_ops_t* ops = eshkol_plat_window_ops_for(backend);
            return ops ? (need_library ? ops->available() : 1) : 0;
        }
        case ESHKOL_PLATFORM_AUDIO: {
            const eshkol_audio_backend_ops_t* ops = eshkol_plat_audio_ops_for(backend);
            return ops ? (need_library ? ops->available() : 1) : 0;
        }
        case ESHKOL_PLATFORM_MIDI: {
            const eshkol_midi_backend_ops_t* ops = eshkol_plat_midi_ops_for(backend);
            return ops ? (need_library ? ops->available() : 1) : 0;
        }
        default:
            return 0;
    }
}

int eshkol_platform_backend_compiled(int backend) {
    return backend_available_flag(backend, 0);
}

int eshkol_platform_backend_available(int backend) {
    return backend_available_flag(backend, 1);
}

/* ── Selection ─────────────────────────────────────────────────────── */

static pthread_mutex_t g_select_lock = PTHREAD_MUTEX_INITIALIZER;
/* -1 = not resolved yet */
static int g_selected[3] = { -1, -1, -1 };

static const char* const k_env_names[3] = {
    "ESHKOL_PLATFORM_DISPLAY", "ESHKOL_PLATFORM_AUDIO", "ESHKOL_PLATFORM_MIDI"
};

/* Native backends in preference order, then the headless one. */
static const int k_auto_order[3][3] = {
    { ESHKOL_BACKEND_COCOA, ESHKOL_BACKEND_X11, ESHKOL_BACKEND_OFFSCREEN },
    { ESHKOL_BACKEND_COREAUDIO, ESHKOL_BACKEND_ALSA, ESHKOL_BACKEND_AUDIO_NULL },
    { ESHKOL_BACKEND_COREMIDI, ESHKOL_BACKEND_ALSA_SEQ, ESHKOL_BACKEND_LOOPBACK },
};

static int backend_from_name(int subsystem, const char* name) {
    if (!name || !*name || strcmp(name, "auto") == 0) return -1;
    static const int all[] = {
        ESHKOL_BACKEND_OFFSCREEN, ESHKOL_BACKEND_X11, ESHKOL_BACKEND_COCOA,
        ESHKOL_BACKEND_WAYLAND, ESHKOL_BACKEND_WIN32,
        ESHKOL_BACKEND_AUDIO_NULL, ESHKOL_BACKEND_ALSA, ESHKOL_BACKEND_COREAUDIO,
        ESHKOL_BACKEND_WASAPI,
        ESHKOL_BACKEND_LOOPBACK, ESHKOL_BACKEND_ALSA_SEQ, ESHKOL_BACKEND_COREMIDI,
        ESHKOL_BACKEND_WINMM };
    for (size_t i = 0; i < sizeof(all) / sizeof(all[0]); ++i) {
        if (subsystem_of(all[i]) == subsystem &&
            strcmp(eshkol_platform_backend_name(all[i]), name) == 0)
            return all[i];
    }
    return ESHKOL_BACKEND_NONE;  /* named but unknown: nothing is selected */
}

static int resolve_auto(int subsystem) {
    for (int i = 0; i < 3; ++i) {
        int b = k_auto_order[subsystem][i];
        if (eshkol_platform_backend_available(b)) return b;
    }
    return ESHKOL_BACKEND_NONE;
}

int eshkol_plat_selected(int subsystem) {
    if (subsystem < 0 || subsystem > 2) return ESHKOL_BACKEND_NONE;
    pthread_mutex_lock(&g_select_lock);
    int b = g_selected[subsystem];
    pthread_mutex_unlock(&g_select_lock);
    if (b >= 0) return b;
    int named = backend_from_name(subsystem, getenv(k_env_names[subsystem]));
    int resolved = named >= 0 ? named : resolve_auto(subsystem);
    pthread_mutex_lock(&g_select_lock);
    if (g_selected[subsystem] < 0) g_selected[subsystem] = resolved;
    b = g_selected[subsystem];
    pthread_mutex_unlock(&g_select_lock);
    return b;
}

int eshkol_platform_backend(int subsystem) {
    return eshkol_plat_selected(subsystem);
}

int eshkol_platform_select_backend(int subsystem, int backend) {
    if (subsystem < 0 || subsystem > 2) return ESHKOL_PLATFORM_EINVAL;
    if (backend != ESHKOL_BACKEND_NONE && subsystem_of(backend) != subsystem)
        return ESHKOL_PLATFORM_EINVAL;
    if (backend != ESHKOL_BACKEND_NONE && !eshkol_platform_backend_compiled(backend))
        return ESHKOL_PLATFORM_EUNAVAILABLE;
    pthread_mutex_lock(&g_select_lock);
    g_selected[subsystem] = backend;
    pthread_mutex_unlock(&g_select_lock);
    return ESHKOL_PLATFORM_OK;
}

int eshkol_platform_init(void) {
    for (int s = 0; s < 3; ++s) (void)eshkol_plat_selected(s);
    return ESHKOL_PLATFORM_OK;
}

/* ── Handle registry ───────────────────────────────────────────────── */

static pthread_mutex_t g_registry_lock = PTHREAD_MUTEX_INITIALIZER;
static eshkol_handle_registry_t* g_registry = NULL;

static eshkol_handle_registry_t* registry(void) {
    pthread_mutex_lock(&g_registry_lock);
    if (!g_registry) g_registry = eshkol_handle_registry_create();
    eshkol_handle_registry_t* r = g_registry;
    pthread_mutex_unlock(&g_registry_lock);
    return r;
}

static int close_platform_object(eshkol_handle_kind_t kind, uintptr_t native,
                                 void* user_data) {
    (void)user_data;
    switch (kind) {
        case ESHKOL_HANDLE_KIND_WINDOW:
            eshkol_window_destroy((eshkol_window_t*)native);
            return 0;
        case ESHKOL_HANDLE_KIND_AUDIO_DEVICE:
            eshkol_audio_close((eshkol_audio_device_t*)native);
            return 0;
        case ESHKOL_HANDLE_KIND_MIDI_PORT:
            eshkol_midi_close((eshkol_midi_port_t*)native);
            return 0;
        default:
            return 1;
    }
}

static int map_handle_status(int st) {
    switch (st) {
        case ESHKOL_HANDLE_OK:     return ESHKOL_PLATFORM_OK;
        case ESHKOL_HANDLE_EBUSY:  return ESHKOL_PLATFORM_EBUSY;
        case ESHKOL_HANDLE_ENOMEM: return ESHKOL_PLATFORM_ENOMEM;
        case ESHKOL_HANDLE_EINVAL: return ESHKOL_PLATFORM_EINVAL;
        case ESHKOL_HANDLE_ECLOSE: return ESHKOL_PLATFORM_EDEVICE;
        default:                   return ESHKOL_PLATFORM_ENOTFOUND;
    }
}

int64_t eshkol_plat_handle_adopt(uint32_t kind, void* object) {
    eshkol_handle_registry_t* r = registry();
    if (!r) return ESHKOL_PLATFORM_ENOMEM;
    eshkol_handle_id_t id = 0;
    int st = eshkol_handle_create(r, kind, (uintptr_t)object, close_platform_object,
                                  NULL, &id);
    if (st != ESHKOL_HANDLE_OK) return map_handle_status(st);
    /* Ids stay below 2^53 so they survive any numeric boxing exactly. */
    if (id > 9007199254740991ull) {
        eshkol_handle_close(r, id);
        return ESHKOL_PLATFORM_ENOMEM;
    }
    return (int64_t)id;
}

void* eshkol_plat_handle_borrow(int64_t id, uint32_t kind, uint64_t* lease, int* status) {
    eshkol_handle_registry_t* r = registry();
    if (!r || id <= 0) { if (status) *status = ESHKOL_PLATFORM_ENOTFOUND; return NULL; }
    uintptr_t native = 0;
    int st = eshkol_handle_borrow(r, (eshkol_handle_id_t)id, &native, lease);
    if (st != ESHKOL_HANDLE_OK) { if (status) *status = map_handle_status(st); return NULL; }
    /* Every platform object records its kind in its first word; a handle of
     * another kind is an unknown handle for this caller. */
    if (!native || *(const uint32_t*)native != kind) {
        eshkol_handle_release(r, (eshkol_handle_id_t)id, *lease);
        if (status) *status = ESHKOL_PLATFORM_ENOTFOUND;
        return NULL;
    }
    if (status) *status = ESHKOL_PLATFORM_OK;
    return (void*)native;
}

void eshkol_plat_handle_release(int64_t id, uint64_t lease) {
    eshkol_handle_registry_t* r = registry();
    if (r && id > 0) eshkol_handle_release(r, (eshkol_handle_id_t)id, lease);
}

int eshkol_plat_handle_close(int64_t id) {
    eshkol_handle_registry_t* r = registry();
    if (!r || id <= 0) return ESHKOL_PLATFORM_ENOTFOUND;
    return map_handle_status(eshkol_handle_close(r, (eshkol_handle_id_t)id));
}

void eshkol_platform_quit(void) {
    pthread_mutex_lock(&g_registry_lock);
    eshkol_handle_registry_t* r = g_registry;
    g_registry = NULL;
    pthread_mutex_unlock(&g_registry_lock);
    if (r) eshkol_handle_registry_destroy(r);
}

/* ── Eshkol bridge helpers ─────────────────────────────────────────── */

int32_t eshkol_plat_backend(int32_t subsystem) {
    return eshkol_plat_selected(subsystem);
}

const char* eshkol_plat_backend_name(int32_t backend) {
    return eshkol_platform_backend_name(backend);
}

int32_t eshkol_plat_select_backend(int32_t subsystem, const char* name) {
    if (subsystem < 0 || subsystem > 2) return ESHKOL_PLATFORM_EINVAL;
    int b = backend_from_name(subsystem, name);
    if (b < 0) {
        pthread_mutex_lock(&g_select_lock);
        g_selected[subsystem] = -1;
        pthread_mutex_unlock(&g_select_lock);
        return eshkol_plat_selected(subsystem) == ESHKOL_BACKEND_NONE
                   ? ESHKOL_PLATFORM_EUNAVAILABLE : ESHKOL_PLATFORM_OK;
    }
    if (b == ESHKOL_BACKEND_NONE) return ESHKOL_PLATFORM_ENOTFOUND;
    return eshkol_platform_select_backend(subsystem, b);
}
