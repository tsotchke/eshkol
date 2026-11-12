/*
 * window.c — backend-independent window object, surface and event plumbing,
 * plus the Eshkol language bridge for windows (ADR-0053).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static atomic_uint g_next_window_id = 1;

const eshkol_window_backend_ops_t* eshkol_plat_window_ops_for(int backend) {
    switch (backend) {
        case ESHKOL_BACKEND_OFFSCREEN: return &eshkol_plat_window_offscreen_ops;
#if defined(ESHKOL_PLATFORM_HAVE_X11)
        case ESHKOL_BACKEND_X11:       return &eshkol_plat_window_x11_ops;
#endif
#if defined(__APPLE__)
        case ESHKOL_BACKEND_COCOA:     return &eshkol_plat_window_cocoa_ops;
#endif
        default:                       return NULL;
    }
}

static int valid_dims(int w, int h) {
    return w > 0 && h > 0 && w <= ESHKOL_WINDOW_MAX_DIM && h <= ESHKOL_WINDOW_MAX_DIM;
}

int eshkol_plat_window_resize_surface(eshkol_window_t* w, int width, int height) {
    if (!valid_dims(width, height)) return ESHKOL_PLATFORM_EINVAL;
    if ((uint32_t)width == w->surface.width && (uint32_t)height == w->surface.height &&
        w->surface.pixels)
        return ESHKOL_PLATFORM_OK;
    uint32_t pitch = (uint32_t)width * 4u;
    uint8_t* px = (uint8_t*)calloc((size_t)height, pitch);
    if (!px) return ESHKOL_PLATFORM_ENOMEM;
    if (w->surface.pixels) {
        uint32_t rows = w->surface.height < (uint32_t)height ? w->surface.height : (uint32_t)height;
        uint32_t cols = w->surface.width < (uint32_t)width ? w->surface.width : (uint32_t)width;
        for (uint32_t r = 0; r < rows; ++r)
            memcpy(px + (size_t)r * pitch, w->surface.pixels + (size_t)r * w->surface.pitch,
                   (size_t)cols * 4u);
        free(w->surface.pixels);
    }
    w->surface.pixels = px;
    w->surface.width = (uint32_t)width;
    w->surface.height = (uint32_t)height;
    w->surface.pitch = pitch;
    w->surface.format = ESHKOL_PIXEL_FORMAT_XRGB8888;
    w->width = width;
    w->height = height;
    return ESHKOL_PLATFORM_OK;
}

void eshkol_plat_window_emit(eshkol_window_t* w, eshkol_event_t* ev) {
    if (!ev->window_id) ev->window_id = w->id;
    if (!ev->timestamp) ev->timestamp = eshkol_platform_time_us();
    if (ev->type == ESHKOL_EVENT_WINDOW_CLOSE) atomic_store(&w->should_close, 1);
    (void)eshkol_plat_queue_push(&w->queue, ev);
}

int eshkol_window_create_ex(int backend, const char* title, int width, int height,
                            uint32_t flags, eshkol_window_t** out) {
    if (!out) return ESHKOL_PLATFORM_EINVAL;
    *out = NULL;
    if (!valid_dims(width, height)) return ESHKOL_PLATFORM_EINVAL;
    const eshkol_window_backend_ops_t* ops = eshkol_plat_window_ops_for(backend);
    if (!ops || !ops->available()) return ESHKOL_PLATFORM_EUNAVAILABLE;
    eshkol_window_t* w = (eshkol_window_t*)calloc(1, sizeof(*w));
    if (!w) return ESHKOL_PLATFORM_ENOMEM;
    w->magic = ESHKOL_PLAT_KIND_WINDOW;
    w->ops = ops;
    w->flags = flags;
    w->id = atomic_fetch_add(&g_next_window_id, 1u);
    atomic_init(&w->should_close, 0);
    atomic_init(&w->presents, 0);
    w->current.type = ESHKOL_EVENT_NONE;
    int st = eshkol_plat_queue_init(&w->queue);
    if (st != ESHKOL_PLATFORM_OK) { free(w); return st; }
    st = eshkol_plat_window_resize_surface(w, width, height);
    if (st == ESHKOL_PLATFORM_OK) st = ops->create(w, title ? title : "Eshkol");
    if (st != ESHKOL_PLATFORM_OK) {
        eshkol_plat_queue_destroy(&w->queue);
        free(w->surface.pixels);
        free(w);
        return st;
    }
    *out = w;
    return ESHKOL_PLATFORM_OK;
}

eshkol_window_t* eshkol_window_create(const char* title, int width, int height,
                                      uint32_t flags) {
    eshkol_window_t* w = NULL;
    (void)eshkol_window_create_ex(eshkol_plat_selected(ESHKOL_PLATFORM_DISPLAY), title,
                                  width, height, flags, &w);
    return w;
}

void eshkol_window_destroy(eshkol_window_t* w) {
    if (!w) return;
    w->ops->destroy(w);
    eshkol_plat_queue_destroy(&w->queue);
    free(w->surface.pixels);
    w->magic = 0;
    free(w);
}

void eshkol_window_get_size(eshkol_window_t* w, int* width, int* height) {
    if (width) *width = w ? w->width : 0;
    if (height) *height = w ? w->height : 0;
}

int eshkol_window_set_size(eshkol_window_t* w, int width, int height) {
    if (!w) return ESHKOL_PLATFORM_EINVAL;
    if (!valid_dims(width, height)) return ESHKOL_PLATFORM_EINVAL;
    return w->ops->set_size(w, width, height);
}

void eshkol_window_get_position(eshkol_window_t* w, int* x, int* y) {
    if (x) *x = w ? w->x : 0;
    if (y) *y = w ? w->y : 0;
}

int eshkol_window_set_position(eshkol_window_t* w, int x, int y) {
    if (!w) return ESHKOL_PLATFORM_EINVAL;
    return w->ops->set_position(w, x, y);
}

int eshkol_window_set_title(eshkol_window_t* w, const char* title) {
    if (!w || !title) return ESHKOL_PLATFORM_EINVAL;
    return w->ops->set_title(w, title);
}

bool eshkol_window_should_close(eshkol_window_t* w) {
    return w && atomic_load(&w->should_close) != 0;
}

void eshkol_window_request_close(eshkol_window_t* w) {
    if (!w) return;
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = ESHKOL_EVENT_WINDOW_CLOSE;
    eshkol_plat_window_emit(w, &ev);
}

uint32_t eshkol_window_id(eshkol_window_t* w) { return w ? w->id : 0; }
int eshkol_window_backend(eshkol_window_t* w) { return w ? w->ops->backend : ESHKOL_BACKEND_NONE; }

eshkol_surface_t* eshkol_window_get_surface(eshkol_window_t* w) {
    return w ? &w->surface : NULL;
}

int eshkol_window_present(eshkol_window_t* w) {
    if (!w) return ESHKOL_PLATFORM_EINVAL;
    int st = w->ops->present(w);
    if (st == ESHKOL_PLATFORM_OK) atomic_fetch_add(&w->presents, 1u);
    return st;
}

uint64_t eshkol_window_present_count(eshkol_window_t* w) {
    return w ? (uint64_t)atomic_load(&w->presents) : 0;
}

int eshkol_window_read_presented(eshkol_window_t* w, int x, int y, uint32_t* out) {
    if (!w || !out || x < 0 || y < 0 || x >= w->width || y >= w->height)
        return ESHKOL_PLATFORM_EINVAL;
    if (atomic_load(&w->presents) == 0) return ESHKOL_PLATFORM_ESTATE;
    return w->ops->read_presented(w, x, y, out);
}

bool eshkol_window_poll_event(eshkol_window_t* w, eshkol_event_t* ev) {
    if (!w || !ev) return false;
    w->ops->pump(w);
    return eshkol_plat_queue_pop(&w->queue, ev);
}

int eshkol_window_wait_event_timeout(eshkol_window_t* w, eshkol_event_t* ev, int timeout_ms) {
    if (!w || !ev) return ESHKOL_PLATFORM_EINVAL;
    uint64_t deadline = timeout_ms < 0 ? UINT64_MAX
                                       : eshkol_plat_now_ns() + (uint64_t)timeout_ms * 1000000ull;
    for (;;) {
        w->ops->pump(w);
        if (eshkol_plat_queue_pop(&w->queue, ev)) return 1;
        uint64_t now = eshkol_plat_now_ns();
        if (now >= deadline) return 0;
        uint64_t left_ms = deadline == UINT64_MAX ? 10u : (deadline - now + 999999ull) / 1000000ull;
        int slice = left_ms > 10u ? 10 : (int)left_ms;
        if (w->ops->wait_os) {
            /* OS input and cross-thread pushes are both observed within a slice. */
            w->ops->wait_os(w, slice);
        } else {
            (void)eshkol_plat_queue_wait(&w->queue, slice);
        }
    }
}

void eshkol_window_wait_event(eshkol_window_t* w, eshkol_event_t* ev) {
    (void)eshkol_window_wait_event_timeout(w, ev, -1);
}

int eshkol_window_push_event(eshkol_window_t* w, const eshkol_event_t* ev) {
    if (!w || !ev) return ESHKOL_PLATFORM_EINVAL;
    eshkol_event_t copy = *ev;
    if (!copy.window_id) copy.window_id = w->id;
    if (!copy.timestamp) copy.timestamp = eshkol_platform_time_us();
    if (copy.type == ESHKOL_EVENT_WINDOW_CLOSE) atomic_store(&w->should_close, 1);
    return eshkol_plat_queue_push(&w->queue, &copy);
}

uint64_t eshkol_window_dropped_events(eshkol_window_t* w) {
    return w ? eshkol_plat_queue_dropped(&w->queue) : 0;
}

/* ── Eshkol language bridge ────────────────────────────────────────── */

#define WITH_WINDOW(handle, errval)                                           \
    uint64_t lease_ = 0; int st_ = 0;                                         \
    eshkol_window_t* w = (eshkol_window_t*)eshkol_plat_handle_borrow(         \
        (handle), ESHKOL_PLAT_KIND_WINDOW, &lease_, &st_);                    \
    if (!w) return st_ + (errval);
#define END_WINDOW(handle) eshkol_plat_handle_release((handle), lease_)

int64_t eshkol_plat_window_open(const char* title, int32_t width, int32_t height,
                                int32_t flags) {
    eshkol_window_t* w = NULL;
    int st = eshkol_window_create_ex(eshkol_plat_selected(ESHKOL_PLATFORM_DISPLAY),
                                     title, width, height, (uint32_t)flags, &w);
    if (st != ESHKOL_PLATFORM_OK) return st;
    int64_t id = eshkol_plat_handle_adopt(ESHKOL_PLAT_KIND_WINDOW, w);
    if (id < 0) eshkol_window_destroy(w);
    return id;
}

int32_t eshkol_plat_window_close(int64_t handle) {
    return eshkol_plat_handle_close(handle);
}

int64_t eshkol_plat_window_query(int64_t handle, int32_t what) {
    WITH_WINDOW(handle, 0)
    int64_t r;
    switch (what) {
        case 0: r = w->width; break;
        case 1: r = w->height; break;
        case 2: r = w->x; break;
        case 3: r = w->y; break;
        case 4: r = atomic_load(&w->should_close) ? 1 : 0; break;
        case 5: r = w->ops->backend; break;
        case 6: r = (int64_t)atomic_load(&w->presents); break;
        case 7: r = w->id; break;
        case 8: r = (int64_t)eshkol_plat_queue_dropped(&w->queue); break;
        default: r = ESHKOL_PLATFORM_EINVAL; break;
    }
    END_WINDOW(handle);
    return r;
}

int32_t eshkol_plat_window_set(int64_t handle, int32_t what, int32_t a, int32_t b) {
    WITH_WINDOW(handle, 0)
    int r;
    switch (what) {
        case 0: r = eshkol_window_set_size(w, a, b); break;
        case 1: r = eshkol_window_set_position(w, a, b); break;
        case 2: eshkol_window_request_close(w); r = ESHKOL_PLATFORM_OK; break;
        default: r = ESHKOL_PLATFORM_EINVAL; break;
    }
    END_WINDOW(handle);
    return r;
}

int32_t eshkol_plat_window_set_title(int64_t handle, const char* title) {
    WITH_WINDOW(handle, 0)
    int r = eshkol_window_set_title(w, title);
    END_WINDOW(handle);
    return r;
}

int32_t eshkol_plat_window_fill_rect(int64_t handle, int32_t x, int32_t y,
                                     int32_t rw, int32_t rh, int64_t xrgb) {
    WITH_WINDOW(handle, 0)
    int r = ESHKOL_PLATFORM_OK;
    if (rw < 0 || rh < 0) {
        r = ESHKOL_PLATFORM_EINVAL;
    } else {
        /* Clip to the surface. */
        int64_t x0 = x < 0 ? 0 : x, y0 = y < 0 ? 0 : y;
        int64_t x1 = (int64_t)x + rw, y1 = (int64_t)y + rh;
        if (x1 > w->width) x1 = w->width;
        if (y1 > w->height) y1 = w->height;
        uint32_t px = (uint32_t)(xrgb & 0x00FFFFFF);
        for (int64_t row = y0; row < y1; ++row) {
            uint32_t* line = (uint32_t*)(w->surface.pixels + (size_t)row * w->surface.pitch);
            for (int64_t col = x0; col < x1; ++col) line[col] = px;
        }
    }
    END_WINDOW(handle);
    return r;
}

int64_t eshkol_plat_window_get_pixel(int64_t handle, int32_t x, int32_t y) {
    WITH_WINDOW(handle, 0)
    int64_t r;
    if (x < 0 || y < 0 || x >= w->width || y >= w->height) {
        r = ESHKOL_PLATFORM_EINVAL;
    } else {
        r = ((uint32_t*)(w->surface.pixels + (size_t)y * w->surface.pitch))[x] & 0x00FFFFFFu;
    }
    END_WINDOW(handle);
    return r;
}

int32_t eshkol_plat_window_present(int64_t handle) {
    WITH_WINDOW(handle, 0)
    int r = eshkol_window_present(w);
    END_WINDOW(handle);
    return r;
}

int64_t eshkol_plat_window_read_presented(int64_t handle, int32_t x, int32_t y) {
    WITH_WINDOW(handle, 0)
    uint32_t px = 0;
    int st = eshkol_window_read_presented(w, x, y, &px);
    END_WINDOW(handle);
    return st == ESHKOL_PLATFORM_OK ? (int64_t)(px & 0x00FFFFFFu) : st;
}

int32_t eshkol_plat_window_next_event(int64_t handle, int32_t timeout_ms) {
    WITH_WINDOW(handle, -1)
    eshkol_event_t ev;
    int got = timeout_ms == 0 ? (eshkol_window_poll_event(w, &ev) ? 1 : 0)
                              : eshkol_window_wait_event_timeout(w, &ev, timeout_ms);
    int32_t r;
    if (got == 1) {
        w->current = ev;
        r = ev.type;
    } else {
        w->current.type = ESHKOL_EVENT_NONE;
        r = got == 0 ? ESHKOL_EVENT_NONE : got - 1;
    }
    END_WINDOW(handle);
    return r;
}

static int64_t event_field(const eshkol_event_t* e, int32_t field) {
    switch (field) {
        case ESHKOL_EVENT_FIELD_TYPE:      return e->type;
        case ESHKOL_EVENT_FIELD_TIMESTAMP: return (int64_t)e->timestamp;
        case ESHKOL_EVENT_FIELD_WINDOW:    return e->window_id;
        default: break;
    }
    switch (e->type) {
        case ESHKOL_EVENT_KEY_DOWN: case ESHKOL_EVENT_KEY_UP:
        case ESHKOL_EVENT_KEY_REPEAT: case ESHKOL_EVENT_TEXT_INPUT:
            switch (field) {
                case ESHKOL_EVENT_FIELD_KEYCODE:   return e->u.key.keycode;
                case ESHKOL_EVENT_FIELD_SCANCODE:  return e->u.key.scancode;
                case ESHKOL_EVENT_FIELD_MODIFIERS: return e->u.key.modifiers;
                case ESHKOL_EVENT_FIELD_UNICODE:   return e->u.key.unicode;
                default: return 0;
            }
        case ESHKOL_EVENT_MOUSE_MOVE: case ESHKOL_EVENT_MOUSE_DOWN:
        case ESHKOL_EVENT_MOUSE_UP: case ESHKOL_EVENT_MOUSE_SCROLL:
        case ESHKOL_EVENT_MOUSE_ENTER: case ESHKOL_EVENT_MOUSE_LEAVE:
            switch (field) {
                case ESHKOL_EVENT_FIELD_X:         return e->u.mouse.x;
                case ESHKOL_EVENT_FIELD_Y:         return e->u.mouse.y;
                case ESHKOL_EVENT_FIELD_DX:        return e->u.mouse.dx;
                case ESHKOL_EVENT_FIELD_DY:        return e->u.mouse.dy;
                case ESHKOL_EVENT_FIELD_BUTTON:    return e->u.mouse.button;
                case ESHKOL_EVENT_FIELD_BUTTONS:   return e->u.mouse.buttons;
                case ESHKOL_EVENT_FIELD_SCROLL_X:  return e->u.mouse.scroll_x;
                case ESHKOL_EVENT_FIELD_SCROLL_Y:  return e->u.mouse.scroll_y;
                case ESHKOL_EVENT_FIELD_MODIFIERS: return e->u.mouse.modifiers;
                default: return 0;
            }
        case ESHKOL_EVENT_TOUCH_BEGIN: case ESHKOL_EVENT_TOUCH_MOVE:
        case ESHKOL_EVENT_TOUCH_END: case ESHKOL_EVENT_TOUCH_CANCEL:
            return field == ESHKOL_EVENT_FIELD_FINGER ? (int64_t)e->u.touch.finger_id : 0;
        case ESHKOL_EVENT_WINDOW_RESIZE: case ESHKOL_EVENT_WINDOW_MOVE:
        case ESHKOL_EVENT_WINDOW_EXPOSE:
            switch (field) {
                case ESHKOL_EVENT_FIELD_X:      return e->u.window.x;
                case ESHKOL_EVENT_FIELD_Y:      return e->u.window.y;
                case ESHKOL_EVENT_FIELD_WIDTH:  return e->u.window.width;
                case ESHKOL_EVENT_FIELD_HEIGHT: return e->u.window.height;
                default: return 0;
            }
        case ESHKOL_EVENT_TIMER:
            return field == ESHKOL_EVENT_FIELD_CODE ? (int64_t)e->u.timer.timer_id
                 : field == ESHKOL_EVENT_FIELD_DATA1 ? (int64_t)e->u.timer.interval : 0;
        case ESHKOL_EVENT_USER:
            switch (field) {
                case ESHKOL_EVENT_FIELD_CODE:  return e->u.user.code;
                case ESHKOL_EVENT_FIELD_DATA1: return (int64_t)e->u.user.data1;
                case ESHKOL_EVENT_FIELD_DATA2: return (int64_t)e->u.user.data2;
                default: return 0;
            }
        default:
            return 0;
    }
}

int64_t eshkol_plat_window_event_field(int64_t handle, int32_t field) {
    WITH_WINDOW(handle, 0)
    int64_t r = event_field(&w->current, field);
    END_WINDOW(handle);
    return r;
}

double eshkol_plat_window_event_real(int64_t handle, int32_t field) {
    uint64_t lease_ = 0; int st_ = 0;
    eshkol_window_t* w = (eshkol_window_t*)eshkol_plat_handle_borrow(
        handle, ESHKOL_PLAT_KIND_WINDOW, &lease_, &st_);
    if (!w) return NAN;
    double r = 0.0;
    const eshkol_event_t* e = &w->current;
    if (e->type >= ESHKOL_EVENT_TOUCH_BEGIN && e->type <= ESHKOL_EVENT_TOUCH_CANCEL) {
        r = field == 0 ? e->u.touch.x : field == 1 ? e->u.touch.y
          : field == 2 ? e->u.touch.pressure : 0.0;
    }
    eshkol_plat_handle_release(handle, lease_);
    return r;
}

int32_t eshkol_plat_window_push_event(int64_t handle, int32_t type,
                                      int64_t a, int64_t b, int64_t c) {
    WITH_WINDOW(handle, 0)
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = type;
    int r = ESHKOL_PLATFORM_OK;
    switch (type) {
        case ESHKOL_EVENT_KEY_DOWN: case ESHKOL_EVENT_KEY_UP:
        case ESHKOL_EVENT_KEY_REPEAT: case ESHKOL_EVENT_TEXT_INPUT:
            ev.u.key.keycode = (uint32_t)a;
            ev.u.key.modifiers = (uint32_t)b;
            ev.u.key.unicode = (uint32_t)c;
            break;
        case ESHKOL_EVENT_MOUSE_MOVE: case ESHKOL_EVENT_MOUSE_DOWN:
        case ESHKOL_EVENT_MOUSE_UP: case ESHKOL_EVENT_MOUSE_ENTER:
        case ESHKOL_EVENT_MOUSE_LEAVE:
            ev.u.mouse.x = (int32_t)a;
            ev.u.mouse.y = (int32_t)b;
            ev.u.mouse.button = (uint32_t)c;
            if (c > 0 && c <= 5) ev.u.mouse.buttons = 1u << (c - 1);
            break;
        case ESHKOL_EVENT_MOUSE_SCROLL:
            ev.u.mouse.scroll_x = (int32_t)a;
            ev.u.mouse.scroll_y = (int32_t)b;
            break;
        case ESHKOL_EVENT_WINDOW_RESIZE: case ESHKOL_EVENT_WINDOW_MOVE:
            if (type == ESHKOL_EVENT_WINDOW_RESIZE) {
                ev.u.window.width = (int32_t)a; ev.u.window.height = (int32_t)b;
            } else {
                ev.u.window.x = (int32_t)a; ev.u.window.y = (int32_t)b;
            }
            break;
        case ESHKOL_EVENT_USER: case ESHKOL_EVENT_TIMER:
            ev.u.user.code = (uint32_t)a;
            ev.u.user.data1 = (uint64_t)b;
            ev.u.user.data2 = (uint64_t)c;
            if (type == ESHKOL_EVENT_TIMER) {
                ev.u.timer.timer_id = (uint32_t)a;
                ev.u.timer.interval = (uint64_t)b;
            }
            break;
        case ESHKOL_EVENT_WINDOW_CLOSE: case ESHKOL_EVENT_WINDOW_FOCUS:
        case ESHKOL_EVENT_WINDOW_BLUR: case ESHKOL_EVENT_WINDOW_SHOW:
        case ESHKOL_EVENT_WINDOW_HIDE: case ESHKOL_EVENT_WINDOW_EXPOSE:
        case ESHKOL_EVENT_QUIT:
            break;
        default:
            r = ESHKOL_PLATFORM_EINVAL;
            break;
    }
    if (r == ESHKOL_PLATFORM_OK) r = eshkol_window_push_event(w, &ev);
    END_WINDOW(handle);
    return r;
}
