/*
 * window_x11.c — X11 display backend (ADR-0053).
 *
 * Compiled where the Xlib headers exist; libX11 itself is bound at run time
 * (libX11.so.6), so the runtime archive carries no X11 link dependency and the
 * backend reports itself unavailable on hosts without the library or without
 * a reachable display ($DISPLAY).
 *
 * Each window owns its own display connection, so a window is pumped only by
 * the thread that owns it and no XInitThreads() is required. Presents use a
 * client-side XImage over the window surface (TrueColor 24/32-bit visuals).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_POSIX_C_SOURCE) && !defined(__APPLE__)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../platform_internal.h"

#if defined(ESHKOL_PLATFORM_HAVE_X11)

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>

/* ── run-time binding ──────────────────────────────────────────────── */

#define X11_FUNCS(F)                \
    F(XOpenDisplay)                 \
    F(XCloseDisplay)                \
    F(XDefaultScreen)               \
    F(XRootWindow)                  \
    F(XBlackPixel)                  \
    F(XDefaultVisual)               \
    F(XDefaultDepth)                \
    F(XCreateSimpleWindow)          \
    F(XDestroyWindow)               \
    F(XStoreName)                   \
    F(XSelectInput)                 \
    F(XMapWindow)                   \
    F(XInternAtom)                  \
    F(XSetWMProtocols)              \
    F(XResizeWindow)                \
    F(XMoveWindow)                  \
    F(XPending)                     \
    F(XEventsQueued)                \
    F(XPeekEvent)                   \
    F(XNextEvent)                   \
    F(XFlush)                       \
    F(XSync)                        \
    F(XCreateGC)                    \
    F(XFreeGC)                      \
    F(XCreateImage)                 \
    F(XPutImage)                    \
    F(XGetImage)                    \
    F(XLookupString)                \
    F(XLookupKeysym)                \
    F(XConnectionNumber)            \
    F(XSetWMNormalHints)            \
    F(XAllocSizeHints)              \
    F(XFree)

static struct {
#define DECL(name) __typeof__(&name) name;
    X11_FUNCS(DECL)
#undef DECL
} X;

static const char* const k_x11_symbols[] = {
#define NAME(name) #name,
    X11_FUNCS(NAME)
#undef NAME
    NULL
};

static pthread_once_t g_x11_once = PTHREAD_ONCE_INIT;
static int g_x11_loaded = 0;

static void x11_load(void) {
    static const char* const libs[] = { "libX11.so.6", "libX11.so", NULL };
    void* lib = eshkol_plat_dlopen_any(libs);
    g_x11_loaded = eshkol_plat_dlsym_all(lib, k_x11_symbols, (void**)&X);
}

static int x11_available(void) {
    pthread_once(&g_x11_once, x11_load);
    if (!g_x11_loaded) return 0;
    const char* d = getenv("DISPLAY");
    return d && *d;
}

/* ── per-window state ──────────────────────────────────────────────── */

typedef struct {
    Display* dpy;
    Window win;
    GC gc;
    Atom wm_delete;
    XImage* image;            /* wraps w->surface.pixels */
    uint32_t image_w, image_h;
    uint32_t buttons;         /* current button mask */
    unsigned char keys_down[32];
} x11_t;

static void x11_free_image(x11_t* x) {
    if (x->image) {
        x->image->data = NULL;  /* the surface owns the pixels */
        x->image->f.destroy_image(x->image);
        x->image = NULL;
    }
}

static int x11_create(eshkol_window_t* w, const char* title) {
    x11_t* x = (x11_t*)calloc(1, sizeof(*x));
    if (!x) return ESHKOL_PLATFORM_ENOMEM;
    x->dpy = X.XOpenDisplay(NULL);
    if (!x->dpy) { free(x); return ESHKOL_PLATFORM_EUNAVAILABLE; }
    int screen = X.XDefaultScreen(x->dpy);
    Visual* vis = X.XDefaultVisual(x->dpy, screen);
    int depth = X.XDefaultDepth(x->dpy, screen);
    if (!vis || vis->class != TrueColor || (depth != 24 && depth != 32) ||
        vis->red_mask != 0xFF0000ul || vis->green_mask != 0x00FF00ul ||
        vis->blue_mask != 0x0000FFul) {
        X.XCloseDisplay(x->dpy);
        free(x);
        return ESHKOL_PLATFORM_EUNAVAILABLE;
    }
    x->win = X.XCreateSimpleWindow(x->dpy, X.XRootWindow(x->dpy, screen), w->x, w->y,
                                   (unsigned)w->width, (unsigned)w->height, 0,
                                   X.XBlackPixel(x->dpy, screen),
                                   X.XBlackPixel(x->dpy, screen));
    X.XStoreName(x->dpy, x->win, title);
    X.XSelectInput(x->dpy, x->win,
                   KeyPressMask | KeyReleaseMask | ButtonPressMask | ButtonReleaseMask |
                   PointerMotionMask | EnterWindowMask | LeaveWindowMask |
                   FocusChangeMask | ExposureMask | StructureNotifyMask);
    x->wm_delete = X.XInternAtom(x->dpy, "WM_DELETE_WINDOW", False);
    X.XSetWMProtocols(x->dpy, x->win, &x->wm_delete, 1);
    if (!(w->flags & ESHKOL_WINDOW_RESIZABLE)) {
        XSizeHints* hints = X.XAllocSizeHints();
        if (hints) {
            hints->flags = PMinSize | PMaxSize;
            hints->min_width = hints->max_width = w->width;
            hints->min_height = hints->max_height = w->height;
            X.XSetWMNormalHints(x->dpy, x->win, hints);
            X.XFree(hints);
        }
    }
    x->gc = X.XCreateGC(x->dpy, x->win, 0, NULL);
    if (!(w->flags & ESHKOL_WINDOW_HIDDEN)) X.XMapWindow(x->dpy, x->win);
    X.XSync(x->dpy, False);
    w->impl = x;
    return ESHKOL_PLATFORM_OK;
}

static void x11_destroy(eshkol_window_t* w) {
    x11_t* x = (x11_t*)w->impl;
    if (!x) return;
    x11_free_image(x);
    if (x->gc) X.XFreeGC(x->dpy, x->gc);
    X.XDestroyWindow(x->dpy, x->win);
    X.XCloseDisplay(x->dpy);
    free(x);
    w->impl = NULL;
}

static int x11_set_size(eshkol_window_t* w, int width, int height) {
    x11_t* x = (x11_t*)w->impl;
    X.XResizeWindow(x->dpy, x->win, (unsigned)width, (unsigned)height);
    X.XFlush(x->dpy);
    return ESHKOL_PLATFORM_OK;  /* ConfigureNotify reports the new size */
}

static int x11_set_position(eshkol_window_t* w, int px, int py) {
    x11_t* x = (x11_t*)w->impl;
    X.XMoveWindow(x->dpy, x->win, px, py);
    X.XFlush(x->dpy);
    return ESHKOL_PLATFORM_OK;
}

static int x11_set_title(eshkol_window_t* w, const char* title) {
    x11_t* x = (x11_t*)w->impl;
    X.XStoreName(x->dpy, x->win, title);
    X.XFlush(x->dpy);
    return ESHKOL_PLATFORM_OK;
}

static int x11_present(eshkol_window_t* w) {
    x11_t* x = (x11_t*)w->impl;
    if (!x->image || x->image_w != w->surface.width || x->image_h != w->surface.height ||
        (uint8_t*)x->image->data != w->surface.pixels) {
        x11_free_image(x);
        int screen = X.XDefaultScreen(x->dpy);
        x->image = X.XCreateImage(x->dpy, X.XDefaultVisual(x->dpy, screen),
                                  (unsigned)X.XDefaultDepth(x->dpy, screen), ZPixmap, 0,
                                  (char*)w->surface.pixels, w->surface.width,
                                  w->surface.height, 32, (int)w->surface.pitch);
        if (!x->image) return ESHKOL_PLATFORM_EDEVICE;
        x->image_w = w->surface.width;
        x->image_h = w->surface.height;
    }
    X.XPutImage(x->dpy, x->win, x->gc, x->image, 0, 0, 0, 0, w->surface.width,
                w->surface.height);
    X.XSync(x->dpy, False);
    return ESHKOL_PLATFORM_OK;
}

static int x11_read_presented(eshkol_window_t* w, int px, int py, uint32_t* out) {
    x11_t* x = (x11_t*)w->impl;
    X.XSync(x->dpy, False);
    XImage* img = X.XGetImage(x->dpy, x->win, px, py, 1, 1, AllPlanes, ZPixmap);
    if (!img) return ESHKOL_PLATFORM_EDEVICE;
    *out = (uint32_t)img->f.get_pixel(img, 0, 0) & 0x00FFFFFFu;
    img->f.destroy_image(img);
    return ESHKOL_PLATFORM_OK;
}

/* ── event translation ─────────────────────────────────────────────── */

static uint32_t map_modifiers(unsigned int state) {
    uint32_t m = 0;
    if (state & ShiftMask) m |= ESHKOL_MOD_SHIFT;
    if (state & ControlMask) m |= ESHKOL_MOD_CTRL;
    if (state & Mod1Mask) m |= ESHKOL_MOD_ALT;
    if (state & Mod4Mask) m |= ESHKOL_MOD_META;
    if (state & LockMask) m |= ESHKOL_MOD_CAPSLOCK;
    if (state & Mod2Mask) m |= ESHKOL_MOD_NUMLOCK;
    return m;
}

static uint32_t map_keysym(KeySym ks) {
    if (ks >= XK_A && ks <= XK_Z) return (uint32_t)(ks - XK_A + 'a');
    if (ks >= 0x20 && ks <= 0x7E) return (uint32_t)ks;
    if (ks >= XK_F1 && ks <= XK_F12) return ESHKOL_KEY_F1 + (uint32_t)(ks - XK_F1);
    switch (ks) {
        case XK_BackSpace: return ESHKOL_KEY_BACKSPACE;
        case XK_Tab:       return ESHKOL_KEY_TAB;
        case XK_Return: case XK_KP_Enter: return ESHKOL_KEY_RETURN;
        case XK_Escape:    return ESHKOL_KEY_ESCAPE;
        case XK_Delete:    return ESHKOL_KEY_DELETE;
        case XK_Left:      return ESHKOL_KEY_LEFT;
        case XK_Right:     return ESHKOL_KEY_RIGHT;
        case XK_Up:        return ESHKOL_KEY_UP;
        case XK_Down:      return ESHKOL_KEY_DOWN;
        case XK_Home:      return ESHKOL_KEY_HOME;
        case XK_End:       return ESHKOL_KEY_END;
        case XK_Prior:     return ESHKOL_KEY_PAGE_UP;
        case XK_Next:      return ESHKOL_KEY_PAGE_DOWN;
        case XK_Insert:    return ESHKOL_KEY_INSERT;
        case XK_Shift_L: case XK_Shift_R:     return ESHKOL_KEY_SHIFT;
        case XK_Control_L: case XK_Control_R: return ESHKOL_KEY_CTRL;
        case XK_Alt_L: case XK_Alt_R:         return ESHKOL_KEY_ALT;
        case XK_Meta_L: case XK_Meta_R:
        case XK_Super_L: case XK_Super_R:     return ESHKOL_KEY_META;
        case XK_Caps_Lock: return ESHKOL_KEY_CAPSLOCK;
        default:           return ESHKOL_KEY_UNKNOWN;
    }
}

static void emit_key(eshkol_window_t* w, x11_t* x, XKeyEvent* ke, int type) {
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = type;
    ev.u.key.scancode = ke->keycode;
    ev.u.key.keycode = map_keysym(X.XLookupKeysym(ke, 0));
    ev.u.key.modifiers = map_modifiers(ke->state);
    char text[8] = {0};
    KeySym sym = 0;
    int n = X.XLookupString(ke, text, (int)sizeof(text), &sym, NULL);
    if (n == 1) ev.u.key.unicode = (unsigned char)text[0];
    eshkol_plat_window_emit(w, &ev);
    (void)x;
}

static void translate(eshkol_window_t* w, x11_t* x, XEvent* e) {
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    switch (e->type) {
        case KeyPress: {
            unsigned kc = e->xkey.keycode & 0xFFu;
            int down = (x->keys_down[kc >> 3] >> (kc & 7)) & 1;
            x->keys_down[kc >> 3] |= (unsigned char)(1u << (kc & 7));
            emit_key(w, x, &e->xkey, down ? ESHKOL_EVENT_KEY_REPEAT : ESHKOL_EVENT_KEY_DOWN);
            return;
        }
        case KeyRelease: {
            /* Auto-repeat arrives as Release+Press with the same time. */
            if (X.XEventsQueued(x->dpy, QueuedAfterReading)) {
                XEvent next;
                X.XPeekEvent(x->dpy, &next);
                if (next.type == KeyPress && next.xkey.time == e->xkey.time &&
                    next.xkey.keycode == e->xkey.keycode)
                    return;  /* the following KeyPress reports KEY_REPEAT */
            }
            unsigned kc = e->xkey.keycode & 0xFFu;
            x->keys_down[kc >> 3] &= (unsigned char)~(1u << (kc & 7));
            emit_key(w, x, &e->xkey, ESHKOL_EVENT_KEY_UP);
            return;
        }
        case ButtonPress:
        case ButtonRelease: {
            unsigned b = e->xbutton.button;
            ev.u.mouse.x = e->xbutton.x;
            ev.u.mouse.y = e->xbutton.y;
            ev.u.mouse.modifiers = map_modifiers(e->xbutton.state);
            if (b >= 4 && b <= 7) {
                if (e->type == ButtonRelease) return;
                ev.type = ESHKOL_EVENT_MOUSE_SCROLL;
                ev.u.mouse.scroll_y = b == 4 ? 1 : b == 5 ? -1 : 0;
                ev.u.mouse.scroll_x = b == 6 ? -1 : b == 7 ? 1 : 0;
            } else {
                uint32_t num = b == 1 ? ESHKOL_BUTTON_LEFT : b == 2 ? ESHKOL_BUTTON_MIDDLE
                             : b == 3 ? ESHKOL_BUTTON_RIGHT : b == 8 ? ESHKOL_BUTTON_X1
                             : b == 9 ? ESHKOL_BUTTON_X2 : 0;
                if (!num) return;
                if (e->type == ButtonPress) x->buttons |= 1u << (num - 1);
                else x->buttons &= ~(1u << (num - 1));
                ev.type = e->type == ButtonPress ? ESHKOL_EVENT_MOUSE_DOWN : ESHKOL_EVENT_MOUSE_UP;
                ev.u.mouse.button = num;
            }
            ev.u.mouse.buttons = x->buttons;
            break;
        }
        case MotionNotify:
            ev.type = ESHKOL_EVENT_MOUSE_MOVE;
            ev.u.mouse.x = e->xmotion.x;
            ev.u.mouse.y = e->xmotion.y;
            ev.u.mouse.buttons = x->buttons;
            ev.u.mouse.modifiers = map_modifiers(e->xmotion.state);
            break;
        case EnterNotify:
        case LeaveNotify:
            ev.type = e->type == EnterNotify ? ESHKOL_EVENT_MOUSE_ENTER : ESHKOL_EVENT_MOUSE_LEAVE;
            ev.u.mouse.x = e->xcrossing.x;
            ev.u.mouse.y = e->xcrossing.y;
            break;
        case FocusIn:  ev.type = ESHKOL_EVENT_WINDOW_FOCUS; break;
        case FocusOut: ev.type = ESHKOL_EVENT_WINDOW_BLUR; break;
        case MapNotify:   ev.type = ESHKOL_EVENT_WINDOW_SHOW; break;
        case UnmapNotify: ev.type = ESHKOL_EVENT_WINDOW_HIDE; break;
        case Expose:
            if (e->xexpose.count != 0) return;
            ev.type = ESHKOL_EVENT_WINDOW_EXPOSE;
            ev.u.window.width = w->width;
            ev.u.window.height = w->height;
            break;
        case ConfigureNotify: {
            int cw = e->xconfigure.width, ch = e->xconfigure.height;
            if (cw != w->width || ch != w->height) {
                if (eshkol_plat_window_resize_surface(w, cw, ch) == ESHKOL_PLATFORM_OK) {
                    eshkol_event_t rs;
                    memset(&rs, 0, sizeof(rs));
                    rs.type = ESHKOL_EVENT_WINDOW_RESIZE;
                    rs.u.window.x = e->xconfigure.x;
                    rs.u.window.y = e->xconfigure.y;
                    rs.u.window.width = cw;
                    rs.u.window.height = ch;
                    eshkol_plat_window_emit(w, &rs);
                }
            }
            if (e->xconfigure.x != w->x || e->xconfigure.y != w->y) {
                w->x = e->xconfigure.x;
                w->y = e->xconfigure.y;
                ev.type = ESHKOL_EVENT_WINDOW_MOVE;
                ev.u.window.x = w->x;
                ev.u.window.y = w->y;
                ev.u.window.width = w->width;
                ev.u.window.height = w->height;
                break;
            }
            return;
        }
        case ClientMessage:
            if ((Atom)e->xclient.data.l[0] != x->wm_delete) return;
            ev.type = ESHKOL_EVENT_WINDOW_CLOSE;
            break;
        default:
            return;
    }
    eshkol_plat_window_emit(w, &ev);
}

static void x11_pump(eshkol_window_t* w) {
    x11_t* x = (x11_t*)w->impl;
    while (X.XPending(x->dpy) > 0) {
        XEvent e;
        X.XNextEvent(x->dpy, &e);
        translate(w, x, &e);
    }
}

static void x11_wait_os(eshkol_window_t* w, int timeout_ms) {
    x11_t* x = (x11_t*)w->impl;
    if (X.XPending(x->dpy) > 0) return;
    struct pollfd pfd = { X.XConnectionNumber(x->dpy), POLLIN, 0 };
    (void)poll(&pfd, 1, timeout_ms);
}

const eshkol_window_backend_ops_t eshkol_plat_window_x11_ops = {
    ESHKOL_BACKEND_X11,
    x11_available, x11_create, x11_destroy, x11_set_size, x11_set_position,
    x11_set_title, x11_present, x11_read_presented, x11_pump, x11_wait_os
};

#endif /* ESHKOL_PLATFORM_HAVE_X11 */
