/**
 * @file window.h
 * @brief Cross-platform windows with a CPU pixel surface and an event queue
 *        (ADR-0053; interface of docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md
 *        section 8.2).
 *
 * Backends: offscreen (always), X11 (Linux/BSD, libX11 loaded at run time),
 * Cocoa (macOS, AppKit loaded at run time). Every window owns an XRGB8888
 * surface; eshkol_window_present() copies it to the display. Each window has
 * its own bounded, thread-safe event queue: poll/wait pump the backend, then
 * pop. eshkol_window_push_event() may be called from any thread.
 *
 * Cocoa requires every window call to come from the process main thread.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_WINDOW_H
#define ESHKOL_PLATFORM_WINDOW_H

#include <stdbool.h>
#include <stdint.h>
#include <eshkol/platform/event.h>
#include <eshkol/platform/platform.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_window eshkol_window_t;

/** XRGB8888: each pixel is a native-endian uint32_t 0x00RRGGBB. */
#define ESHKOL_PIXEL_FORMAT_XRGB8888 1u

typedef struct eshkol_surface {
    uint32_t width;
    uint32_t height;
    uint32_t pitch;    /**< bytes per row */
    uint32_t format;   /**< ESHKOL_PIXEL_FORMAT_XRGB8888 */
    uint8_t* pixels;
} eshkol_surface_t;

#define ESHKOL_WINDOW_RESIZABLE  0x01u
#define ESHKOL_WINDOW_BORDERLESS 0x02u
#define ESHKOL_WINDOW_FULLSCREEN 0x04u
#define ESHKOL_WINDOW_HIDDEN     0x08u
#define ESHKOL_WINDOW_ALWAYS_TOP 0x10u

/** Largest accepted width/height. */
#define ESHKOL_WINDOW_MAX_DIM 16384

/* ── Lifecycle ─────────────────────────────────────────────────────── */
/** Create a window on the selected display backend; NULL on failure. */
eshkol_window_t* eshkol_window_create(const char* title, int width, int height,
                                      uint32_t flags);
/** Create a window on an explicit backend; returns a status code. */
int eshkol_window_create_ex(int backend, const char* title, int width,
                            int height, uint32_t flags, eshkol_window_t** out);
void eshkol_window_destroy(eshkol_window_t* window);

/* ── Properties ────────────────────────────────────────────────────── */
void eshkol_window_get_size(eshkol_window_t* window, int* w, int* h);
int  eshkol_window_set_size(eshkol_window_t* window, int w, int h);
void eshkol_window_get_position(eshkol_window_t* window, int* x, int* y);
int  eshkol_window_set_position(eshkol_window_t* window, int x, int y);
int  eshkol_window_set_title(eshkol_window_t* window, const char* title);
bool eshkol_window_should_close(eshkol_window_t* window);
void eshkol_window_request_close(eshkol_window_t* window);
uint32_t eshkol_window_id(eshkol_window_t* window);
int  eshkol_window_backend(eshkol_window_t* window);

/* ── Surface ───────────────────────────────────────────────────────── */
eshkol_surface_t* eshkol_window_get_surface(eshkol_window_t* window);
/** Copy the surface to the display. */
int  eshkol_window_present(eshkol_window_t* window);
/** Number of successful presents. */
uint64_t eshkol_window_present_count(eshkol_window_t* window);
/**
 * Read one pixel back from what the display backend last received
 * (X11: the server-side window image; Cocoa: the view's rendered backing;
 * offscreen: the presented frame). Verifies the present path end to end.
 */
int  eshkol_window_read_presented(eshkol_window_t* window, int x, int y,
                                  uint32_t* out_xrgb);

/* ── Events ────────────────────────────────────────────────────────── */
/** Pump the backend and pop one event; false when the queue is empty. */
bool eshkol_window_poll_event(eshkol_window_t* window, eshkol_event_t* event);
/** Block until an event arrives. */
void eshkol_window_wait_event(eshkol_window_t* window, eshkol_event_t* event);
/** Wait up to timeout_ms (negative = forever): 1 event, 0 timeout, <0 status. */
int  eshkol_window_wait_event_timeout(eshkol_window_t* window,
                                      eshkol_event_t* event, int timeout_ms);
/** Enqueue an event (thread-safe). window_id/timestamp are filled when 0. */
int  eshkol_window_push_event(eshkol_window_t* window, const eshkol_event_t* event);
/** Events dropped because the queue was full. */
uint64_t eshkol_window_dropped_events(eshkol_window_t* window);

/* ── Eshkol language bridge (generation-checked integer handles) ───── */
int64_t eshkol_plat_window_open(const char* title, int32_t width, int32_t height,
                                int32_t flags);
int32_t eshkol_plat_window_close(int64_t handle);
/** what: 0 width, 1 height, 2 x, 3 y, 4 should-close, 5 backend, 6 presents,
 *  7 window id, 8 dropped events. */
int64_t eshkol_plat_window_query(int64_t handle, int32_t what);
/** what: 0 size(a,b), 1 position(a,b), 2 request close. */
int32_t eshkol_plat_window_set(int64_t handle, int32_t what, int32_t a, int32_t b);
int32_t eshkol_plat_window_set_title(int64_t handle, const char* title);
int32_t eshkol_plat_window_fill_rect(int64_t handle, int32_t x, int32_t y,
                                     int32_t w, int32_t h, int64_t xrgb);
int64_t eshkol_plat_window_get_pixel(int64_t handle, int32_t x, int32_t y);
int32_t eshkol_plat_window_present(int64_t handle);
int64_t eshkol_plat_window_read_presented(int64_t handle, int32_t x, int32_t y);
/** Pump/pop into the handle's current-event slot. Returns the event type,
 *  ESHKOL_EVENT_NONE (-1) when none arrived in time, or < -1 on error
 *  (status - 1). timeout_ms: 0 poll, < 0 wait forever. */
int32_t eshkol_plat_window_next_event(int64_t handle, int32_t timeout_ms);
/** Integer field of the current event (see ESHKOL_EVENT_FIELD_*). */
int64_t eshkol_plat_window_event_field(int64_t handle, int32_t field);
/** Real field of the current event: 0 touch x, 1 touch y, 2 pressure. */
double  eshkol_plat_window_event_real(int64_t handle, int32_t field);
/** Push an event: key (a=keycode,b=modifiers,c=unicode), mouse (a=x,b=y,
 *  c=button), resize/move (a,b), user (a=code,b=data1,c=data2). */
int32_t eshkol_plat_window_push_event(int64_t handle, int32_t type,
                                      int64_t a, int64_t b, int64_t c);

enum {
    ESHKOL_EVENT_FIELD_TYPE = 0, ESHKOL_EVENT_FIELD_TIMESTAMP = 1,
    ESHKOL_EVENT_FIELD_WINDOW = 2, ESHKOL_EVENT_FIELD_KEYCODE = 3,
    ESHKOL_EVENT_FIELD_SCANCODE = 4, ESHKOL_EVENT_FIELD_MODIFIERS = 5,
    ESHKOL_EVENT_FIELD_UNICODE = 6, ESHKOL_EVENT_FIELD_X = 7,
    ESHKOL_EVENT_FIELD_Y = 8, ESHKOL_EVENT_FIELD_DX = 9,
    ESHKOL_EVENT_FIELD_DY = 10, ESHKOL_EVENT_FIELD_BUTTON = 11,
    ESHKOL_EVENT_FIELD_BUTTONS = 12, ESHKOL_EVENT_FIELD_SCROLL_X = 13,
    ESHKOL_EVENT_FIELD_SCROLL_Y = 14, ESHKOL_EVENT_FIELD_WIDTH = 15,
    ESHKOL_EVENT_FIELD_HEIGHT = 16, ESHKOL_EVENT_FIELD_CODE = 17,
    ESHKOL_EVENT_FIELD_DATA1 = 18, ESHKOL_EVENT_FIELD_DATA2 = 19,
    ESHKOL_EVENT_FIELD_FINGER = 20
};

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_WINDOW_H */
