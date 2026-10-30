/*
 * window_null.c — offscreen display backend (ADR-0053).
 *
 * A window is a surface plus a presented frame in memory. It has no OS input,
 * so events come from eshkol_window_push_event(); geometry requests complete
 * immediately and emit the same WINDOW_RESIZE / WINDOW_MOVE events a native
 * backend reports, so programs and tests observe identical sequences.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    uint32_t* presented;
    uint32_t pw, ph;
    char title[256];
} offscreen_t;

static int off_available(void) { return 1; }

static int off_create(eshkol_window_t* w, const char* title) {
    offscreen_t* o = (offscreen_t*)calloc(1, sizeof(*o));
    if (!o) return ESHKOL_PLATFORM_ENOMEM;
    strncpy(o->title, title, sizeof(o->title) - 1);
    w->impl = o;
    if (!(w->flags & ESHKOL_WINDOW_HIDDEN)) {
        eshkol_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = ESHKOL_EVENT_WINDOW_SHOW;
        eshkol_plat_window_emit(w, &ev);
    }
    return ESHKOL_PLATFORM_OK;
}

static void off_destroy(eshkol_window_t* w) {
    offscreen_t* o = (offscreen_t*)w->impl;
    if (!o) return;
    free(o->presented);
    free(o);
    w->impl = NULL;
}

static int off_set_size(eshkol_window_t* w, int width, int height) {
    int st = eshkol_plat_window_resize_surface(w, width, height);
    if (st != ESHKOL_PLATFORM_OK) return st;
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = ESHKOL_EVENT_WINDOW_RESIZE;
    ev.u.window.x = w->x;
    ev.u.window.y = w->y;
    ev.u.window.width = width;
    ev.u.window.height = height;
    eshkol_plat_window_emit(w, &ev);
    return ESHKOL_PLATFORM_OK;
}

static int off_set_position(eshkol_window_t* w, int x, int y) {
    w->x = x;
    w->y = y;
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = ESHKOL_EVENT_WINDOW_MOVE;
    ev.u.window.x = x;
    ev.u.window.y = y;
    ev.u.window.width = w->width;
    ev.u.window.height = w->height;
    eshkol_plat_window_emit(w, &ev);
    return ESHKOL_PLATFORM_OK;
}

static int off_set_title(eshkol_window_t* w, const char* title) {
    offscreen_t* o = (offscreen_t*)w->impl;
    strncpy(o->title, title, sizeof(o->title) - 1);
    o->title[sizeof(o->title) - 1] = 0;
    return ESHKOL_PLATFORM_OK;
}

static int off_present(eshkol_window_t* w) {
    offscreen_t* o = (offscreen_t*)w->impl;
    size_t n = (size_t)w->surface.width * w->surface.height;
    if (o->pw != w->surface.width || o->ph != w->surface.height) {
        uint32_t* p = (uint32_t*)realloc(o->presented, n * 4u);
        if (!p) return ESHKOL_PLATFORM_ENOMEM;
        o->presented = p;
        o->pw = w->surface.width;
        o->ph = w->surface.height;
    }
    for (uint32_t r = 0; r < w->surface.height; ++r)
        memcpy(o->presented + (size_t)r * o->pw,
               w->surface.pixels + (size_t)r * w->surface.pitch, (size_t)o->pw * 4u);
    return ESHKOL_PLATFORM_OK;
}

static int off_read_presented(eshkol_window_t* w, int x, int y, uint32_t* out) {
    offscreen_t* o = (offscreen_t*)w->impl;
    if (!o->presented || (uint32_t)x >= o->pw || (uint32_t)y >= o->ph)
        return ESHKOL_PLATFORM_ESTATE;
    *out = o->presented[(size_t)y * o->pw + (size_t)x] & 0x00FFFFFFu;
    return ESHKOL_PLATFORM_OK;
}

static void off_pump(eshkol_window_t* w) { (void)w; }

const eshkol_window_backend_ops_t eshkol_plat_window_offscreen_ops = {
    ESHKOL_BACKEND_OFFSCREEN,
    off_available, off_create, off_destroy, off_set_size, off_set_position,
    off_set_title, off_present, off_read_presented, off_pump,
    NULL /* wait on the event queue itself */
};
