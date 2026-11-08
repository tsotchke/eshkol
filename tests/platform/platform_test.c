/*
 * platform_test.c — tests for the Eshkol platform layer (ADR-0053).
 *
 *   platform_test headless   offscreen windows + events, null audio clock,
 *                            MIDI loopback, the Eshkol bridge (always runs)
 *   platform_test x11        X11 backend against a real X server ($DISPLAY,
 *                            e.g. Xvfb); exit 77 when none is reachable
 *   platform_test alsa       ALSA PCM through alsa-lib's "null" device, and
 *                            the ALSA sequencer when /dev/snd/seq is usable;
 *                            exit 77 when libasound does not load
 *   platform_test macos      Cocoa (hidden window, synthetic AppKit events),
 *                            CoreAudio default output, CoreMIDI virtual-source
 *                            loopback; needs a logged-in window-server session
 *
 * Prints "PASS: platform <mode> N checks" on success.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "../../lib/platform/platform_internal.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_checks = 0, g_failures = 0;

#define CHECK(cond) do {                                                     \
    ++g_checks;                                                              \
    if (!(cond)) {                                                           \
        ++g_failures;                                                        \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #cond);     \
    }                                                                        \
} while (0)

#define CHECK_EQ(a, b) do {                                                  \
    long long va_ = (long long)(a), vb_ = (long long)(b);                    \
    ++g_checks;                                                              \
    if (va_ != vb_) {                                                        \
        ++g_failures;                                                        \
        fprintf(stderr, "FAIL: %s:%d: %s == %s (%lld vs %lld)\n", __FILE__,  \
                __LINE__, #a, #b, va_, vb_);                                 \
    }                                                                        \
} while (0)

static uint32_t px_at(eshkol_surface_t* s, int x, int y) {
    return ((uint32_t*)(s->pixels + (size_t)y * s->pitch))[x];
}

static void paint(eshkol_window_t* w) {
    eshkol_surface_t* s = eshkol_window_get_surface(w);
    for (uint32_t y = 0; y < s->height; ++y)
        for (uint32_t x = 0; x < s->width; ++x)
            ((uint32_t*)(s->pixels + (size_t)y * s->pitch))[x] =
                ((x * 5u) & 0xFFu) << 16 | ((y * 7u) & 0xFFu) << 8 | ((x ^ y) & 0xFFu);
}

static int drain_until(eshkol_window_t* w, int type, eshkol_event_t* out, int timeout_ms) {
    uint64_t end = eshkol_plat_now_ns() + (uint64_t)timeout_ms * 1000000ull;
    eshkol_event_t ev;
    while (eshkol_plat_now_ns() < end) {
        if (eshkol_window_wait_event_timeout(w, &ev, 20) == 1 && ev.type == type) {
            if (out) *out = ev;
            return 1;
        }
    }
    return 0;
}

/* ── shared window checks (every display backend) ──────────────────── */

static void window_contract(int backend) {
    eshkol_window_t* w = NULL;
    CHECK_EQ(eshkol_window_create_ex(backend, "eshkol platform test", 64, 48, 0, &w),
             ESHKOL_PLATFORM_OK);
    if (!w) return;
    CHECK_EQ(eshkol_window_backend(w), backend);
    eshkol_surface_t* s = eshkol_window_get_surface(w);
    CHECK(s && s->width == 64 && s->height == 48 && s->pitch == 256 &&
          s->format == ESHKOL_PIXEL_FORMAT_XRGB8888);
    paint(w);
    CHECK_EQ(eshkol_window_present(w), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_window_present_count(w), 1);
    int exact = 1;
    for (int y = 0; y < 48; y += 7)
        for (int x = 0; x < 64; x += 9) {
            uint32_t got = 0;
            if (eshkol_window_read_presented(w, x, y, &got) != ESHKOL_PLATFORM_OK ||
                got != (px_at(s, x, y) & 0xFFFFFFu))
                exact = 0;
        }
    CHECK(exact);

    /* Resize: the surface follows and a WINDOW_RESIZE event reports it. */
    CHECK_EQ(eshkol_window_set_size(w, 80, 60), ESHKOL_PLATFORM_OK);
    eshkol_event_t ev;
    CHECK(drain_until(w, ESHKOL_EVENT_WINDOW_RESIZE, &ev, 2000));
    CHECK_EQ(ev.u.window.width, 80);
    CHECK_EQ(ev.u.window.height, 60);
    int ww = 0, hh = 0;
    eshkol_window_get_size(w, &ww, &hh);
    CHECK(ww == 80 && hh == 60);
    CHECK(s->width == 80 && s->height == 60 && s->pitch == 320);
    CHECK_EQ(px_at(s, 10, 10) & 0xFFFFFFu, ((50u & 0xFF) << 16) | ((70u & 0xFF) << 8) | (10u ^ 10u));
    CHECK_EQ(eshkol_window_set_title(w, "renamed"), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_window_set_size(w, 0, 10), ESHKOL_PLATFORM_EINVAL);

    /* Pushed events arrive in order, stamped with the window and a time. */
    eshkol_event_t key;
    memset(&key, 0, sizeof(key));
    key.type = ESHKOL_EVENT_USER;
    key.u.user.code = 4242;
    CHECK_EQ(eshkol_window_push_event(w, &key), ESHKOL_PLATFORM_OK);
    CHECK(drain_until(w, ESHKOL_EVENT_USER, &ev, 1000));
    CHECK_EQ(ev.u.user.code, 4242);
    CHECK_EQ(ev.window_id, eshkol_window_id(w));
    CHECK(ev.timestamp > 0 && ev.timestamp <= eshkol_platform_time_us());

    eshkol_window_request_close(w);
    CHECK(eshkol_window_should_close(w));
    CHECK(drain_until(w, ESHKOL_EVENT_WINDOW_CLOSE, NULL, 1000));
    eshkol_window_destroy(w);
}

/* ── headless ──────────────────────────────────────────────────────── */

static void* pusher(void* arg) {
    eshkol_window_t* w = (eshkol_window_t*)arg;
    for (int i = 0; i < 200; ++i) {
        eshkol_event_t ev;
        memset(&ev, 0, sizeof(ev));
        ev.type = ESHKOL_EVENT_MOUSE_MOVE;
        ev.u.mouse.x = i;
        ev.u.mouse.y = 2 * i;
        eshkol_window_push_event(w, &ev);
    }
    return NULL;
}

static void headless_windows(void) {
    CHECK(eshkol_platform_backend_compiled(ESHKOL_BACKEND_OFFSCREEN));
    CHECK(eshkol_platform_backend_available(ESHKOL_BACKEND_OFFSCREEN));
    CHECK(!eshkol_platform_backend_compiled(ESHKOL_BACKEND_WAYLAND));
    CHECK(!eshkol_platform_backend_compiled(ESHKOL_BACKEND_WIN32));
    CHECK(strcmp(eshkol_platform_backend_name(ESHKOL_BACKEND_OFFSCREEN), "offscreen") == 0);
    CHECK_EQ(eshkol_platform_select_backend(ESHKOL_PLATFORM_DISPLAY, ESHKOL_BACKEND_OFFSCREEN),
             ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_platform_backend(ESHKOL_PLATFORM_DISPLAY), ESHKOL_BACKEND_OFFSCREEN);
    CHECK_EQ(eshkol_platform_select_backend(ESHKOL_PLATFORM_DISPLAY, ESHKOL_BACKEND_ALSA),
             ESHKOL_PLATFORM_EINVAL);
    eshkol_window_t* bad = NULL;
    CHECK_EQ(eshkol_window_create_ex(ESHKOL_BACKEND_WAYLAND, "x", 10, 10, 0, &bad),
             ESHKOL_PLATFORM_EUNAVAILABLE);
    CHECK_EQ(eshkol_window_create_ex(ESHKOL_BACKEND_OFFSCREEN, "x", 0, 10, 0, &bad),
             ESHKOL_PLATFORM_EINVAL);
    CHECK_EQ(eshkol_window_create_ex(ESHKOL_BACKEND_OFFSCREEN, "x", 10, 20000, 0, &bad),
             ESHKOL_PLATFORM_EINVAL);

    window_contract(ESHKOL_BACKEND_OFFSCREEN);

    /* SHOW on creation, MOVE on reposition, cross-thread pushes keep order. */
    eshkol_window_t* w = eshkol_window_create("headless", 32, 32, 0);
    CHECK(w != NULL);
    if (!w) return;
    eshkol_event_t ev;
    CHECK(eshkol_window_poll_event(w, &ev) && ev.type == ESHKOL_EVENT_WINDOW_SHOW);
    CHECK(!eshkol_window_poll_event(w, &ev));
    CHECK_EQ(eshkol_window_read_presented(w, 0, 0, &(uint32_t){0}), ESHKOL_PLATFORM_ESTATE);
    CHECK_EQ(eshkol_window_set_position(w, 11, 22), ESHKOL_PLATFORM_OK);
    CHECK(eshkol_window_poll_event(w, &ev) && ev.type == ESHKOL_EVENT_WINDOW_MOVE &&
          ev.u.window.x == 11 && ev.u.window.y == 22);
    pthread_t th;
    pthread_create(&th, NULL, pusher, w);
    int seen = 0, ordered = 1;
    while (seen < 200) {
        if (eshkol_window_wait_event_timeout(w, &ev, 1000) != 1) break;
        if (ev.type != ESHKOL_EVENT_MOUSE_MOVE || ev.u.mouse.x != seen ||
            ev.u.mouse.y != 2 * seen)
            ordered = 0;
        ++seen;
    }
    pthread_join(th, NULL);
    CHECK_EQ(seen, 200);
    CHECK(ordered);

    /* A wait with nothing queued returns 0 after roughly its timeout. */
    uint64_t t0 = eshkol_plat_now_ns();
    CHECK_EQ(eshkol_window_wait_event_timeout(w, &ev, 30), 0);
    uint64_t waited = (eshkol_plat_now_ns() - t0) / 1000000ull;
    CHECK(waited >= 25 && waited < 1000);

    /* A full queue drops new events and counts them. */
    for (unsigned i = 0; i < ESHKOL_PLAT_EVENT_QUEUE_CAP + 50; ++i) {
        eshkol_event_t u;
        memset(&u, 0, sizeof(u));
        u.type = ESHKOL_EVENT_USER;
        u.u.user.code = i;
        eshkol_window_push_event(w, &u);
    }
    CHECK_EQ(eshkol_window_dropped_events(w), 50);
    unsigned popped = 0;
    while (eshkol_window_poll_event(w, &ev)) ++popped;
    CHECK_EQ(popped, ESHKOL_PLAT_EVENT_QUEUE_CAP);
    eshkol_window_destroy(w);
}

/* Bridge: integer handles, kind checks, closed handles. */
static void headless_bridge(void) {
    int64_t h = eshkol_plat_window_open("bridge", 40, 30, 0);
    CHECK(h > 0);
    CHECK_EQ(eshkol_plat_window_query(h, 0), 40);
    CHECK_EQ(eshkol_plat_window_query(h, 1), 30);
    CHECK_EQ(eshkol_plat_window_query(h, 5), ESHKOL_BACKEND_OFFSCREEN);
    CHECK_EQ(eshkol_plat_window_next_event(h, 0), ESHKOL_EVENT_WINDOW_SHOW);
    CHECK_EQ(eshkol_plat_window_next_event(h, 0), ESHKOL_EVENT_NONE);
    CHECK_EQ(eshkol_plat_window_fill_rect(h, -5, -5, 15, 15, 0x123456), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_fill_rect(h, 35, 25, 100, 100, 0xABCDEF), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_get_pixel(h, 0, 0), 0x123456);
    CHECK_EQ(eshkol_plat_window_get_pixel(h, 9, 9), 0x123456);
    CHECK_EQ(eshkol_plat_window_get_pixel(h, 10, 10), 0);
    CHECK_EQ(eshkol_plat_window_get_pixel(h, 39, 29), 0xABCDEF);
    CHECK_EQ(eshkol_plat_window_get_pixel(h, 40, 0), ESHKOL_PLATFORM_EINVAL);
    CHECK_EQ(eshkol_plat_window_read_presented(h, 0, 0), ESHKOL_PLATFORM_ESTATE);
    CHECK_EQ(eshkol_plat_window_present(h), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_read_presented(h, 39, 29), 0xABCDEF);
    CHECK_EQ(eshkol_plat_window_query(h, 6), 1);
    CHECK_EQ(eshkol_plat_window_push_event(h, ESHKOL_EVENT_KEY_DOWN, 'q', ESHKOL_MOD_SHIFT, 'Q'),
             ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_push_event(h, ESHKOL_EVENT_MOUSE_DOWN, 7, 8, ESHKOL_BUTTON_RIGHT),
             ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_push_event(h, 99, 0, 0, 0), ESHKOL_PLATFORM_EINVAL);
    CHECK_EQ(eshkol_plat_window_next_event(h, 100), ESHKOL_EVENT_KEY_DOWN);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_KEYCODE), 'q');
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_MODIFIERS), ESHKOL_MOD_SHIFT);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_UNICODE), 'Q');
    CHECK_EQ(eshkol_plat_window_next_event(h, 100), ESHKOL_EVENT_MOUSE_DOWN);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_X), 7);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_Y), 8);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_BUTTON), ESHKOL_BUTTON_RIGHT);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_BUTTONS), 1u << 2);
    CHECK_EQ(eshkol_plat_window_set(h, 0, 50, 20), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_next_event(h, 100), ESHKOL_EVENT_WINDOW_RESIZE);
    CHECK_EQ(eshkol_plat_window_event_field(h, ESHKOL_EVENT_FIELD_WIDTH), 50);
    CHECK_EQ(eshkol_plat_window_set(h, 2, 0, 0), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_query(h, 4), 1);

    /* Kind checks: an audio handle is not a window and vice versa. */
    CHECK_EQ(eshkol_platform_select_backend(ESHKOL_PLATFORM_AUDIO, ESHKOL_BACKEND_AUDIO_NULL),
             ESHKOL_PLATFORM_OK);
    int64_t a = eshkol_plat_audio_open(ESHKOL_AUDIO_OUTPUT, 48000, 2, ESHKOL_AUDIO_FORMAT_F32,
                                       256, 0);
    CHECK(a > 0);
    CHECK_EQ(eshkol_plat_window_query(a, 0), ESHKOL_PLATFORM_ENOTFOUND);
    CHECK_EQ(eshkol_plat_audio_stat(h, 0), ESHKOL_PLATFORM_ENOTFOUND);
    CHECK_EQ(eshkol_plat_audio_close(a), ESHKOL_PLATFORM_OK);

    CHECK_EQ(eshkol_plat_window_close(h), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_window_close(h), ESHKOL_PLATFORM_ENOTFOUND);
    CHECK_EQ(eshkol_plat_window_query(h, 0), ESHKOL_PLATFORM_ENOTFOUND);
    CHECK_EQ(eshkol_plat_window_next_event(h, 0), ESHKOL_PLATFORM_ENOTFOUND - 1);
    CHECK(eshkol_plat_window_open("bad", -1, 5, 0) == ESHKOL_PLATFORM_EINVAL);
}

/* ── audio ─────────────────────────────────────────────────────────── */

typedef struct { atomic_uint calls; atomic_uint frames; atomic_uint bad; uint32_t expect; } cb_state_t;

static void counting_callback(void* ud, uint8_t* buf, uint32_t frames) {
    cb_state_t* s = (cb_state_t*)ud;
    atomic_fetch_add(&s->calls, 1u);
    atomic_fetch_add(&s->frames, frames);
    if (s->expect && frames != s->expect) atomic_fetch_add(&s->bad, 1u);
    memset(buf, 0, (size_t)frames * 8u);
}

static void audio_contract(int backend, const char* device, int timed, int with_input) {
    eshkol_audio_config_t cfg = { 48000, 2, ESHKOL_AUDIO_FORMAT_F32, 256 };

    /* Callback mode: the backend's thread calls the user callback once per period. */
    cb_state_t st;
    memset(&st, 0, sizeof(st));
    eshkol_audio_device_t* d = NULL;
    CHECK_EQ(eshkol_audio_open_ex(backend, ESHKOL_AUDIO_OUTPUT, device, &cfg, counting_callback,
                                  &st, 0, &d), ESHKOL_PLATFORM_OK);
    if (!d) return;
    eshkol_audio_config_t got;
    eshkol_audio_get_config(d, &got);
    st.expect = got.buffer_frames;
    CHECK_EQ(eshkol_audio_backend_of(d), backend);
    CHECK_EQ(eshkol_audio_start(d), ESHKOL_PLATFORM_OK);
    eshkol_plat_sleep_ms(200);
    CHECK_EQ(eshkol_audio_stop(d), ESHKOL_PLATFORM_OK);
    unsigned calls = atomic_load(&st.calls);
    eshkol_audio_stats_t s;
    eshkol_audio_get_stats(d, &s);
    CHECK(calls > 0);
    CHECK_EQ(s.callbacks, calls);
    CHECK_EQ(s.frames, atomic_load(&st.frames));
    CHECK_EQ(atomic_load(&st.bad), 0);
    CHECK(s.max_callback_ns >= s.last_callback_ns);
    if (timed) {
        CHECK(s.device_latency_frames > 0);   /* an unclocked PCM reports zero delay */
        /* 200 ms at 48 kHz / 256 frames is 37.5 periods. */
        CHECK(calls >= 25 && calls <= 45);
    }
    unsigned frozen = atomic_load(&st.calls);
    eshkol_plat_sleep_ms(30);
    CHECK_EQ(atomic_load(&st.calls), frozen);   /* stopped means stopped */
    eshkol_audio_close(d);

    /* Queue mode output: queued frames drain, then underruns are counted. */
    d = NULL;
    CHECK_EQ(eshkol_audio_open_ex(backend, ESHKOL_AUDIO_OUTPUT, device, &cfg, NULL, NULL, 4096, &d),
             ESHKOL_PLATFORM_OK);
    if (!d) return;
    float tone[2 * 1024];
    for (int i = 0; i < 1024; ++i) tone[2 * i] = tone[2 * i + 1] = (float)sin(i * 0.05);
    CHECK(eshkol_audio_queue(d, tone, 1024));
    CHECK(eshkol_audio_queue(d, tone, 1024));
    CHECK_EQ(eshkol_audio_queued_frames(d), 2048);
    CHECK(eshkol_audio_queue(d, tone, 1024));
    CHECK(eshkol_audio_queue(d, tone, 1024));
    CHECK(!eshkol_audio_queue(d, tone, 1));      /* ring of 4096 frames is full */
    CHECK_EQ(eshkol_audio_start(d), ESHKOL_PLATFORM_OK);
    eshkol_plat_sleep_ms(250);                   /* 4096 frames last ~85 ms */
    CHECK_EQ(eshkol_audio_pause(d), ESHKOL_PLATFORM_OK);
    eshkol_audio_get_stats(d, &s);
    CHECK_EQ(s.queued_frames, 0);
    CHECK(s.underruns > 0);
    CHECK_EQ(s.frames, 4096 + s.underrun_frames);
    eshkol_audio_close(d);

    if (!with_input) return;
    /* Queue mode input: an unread ring overruns and counts dropped frames. */
    d = NULL;
    CHECK_EQ(eshkol_audio_open_ex(backend, ESHKOL_AUDIO_INPUT, device, &cfg, NULL, NULL, 1024, &d),
             ESHKOL_PLATFORM_OK);
    if (!d) return;
    CHECK_EQ(eshkol_audio_start(d), ESHKOL_PLATFORM_OK);
    eshkol_plat_sleep_ms(150);
    CHECK_EQ(eshkol_audio_stop(d), ESHKOL_PLATFORM_OK);
    eshkol_audio_get_stats(d, &s);
    CHECK(s.overruns > 0);
    CHECK_EQ(s.frames, 1024 + s.overrun_frames);
    float cap[2 * 512];
    CHECK_EQ(eshkol_audio_dequeue(d, cap, 512), 512);
    int silent = 1;
    for (int i = 0; i < 1024; ++i) if (cap[i] != 0.0f) silent = 0;
    CHECK(silent);
    CHECK_EQ(eshkol_audio_queued_frames(d), 512);
    eshkol_audio_close(d);
}

static void headless_audio(void) {
    CHECK(eshkol_platform_backend_available(ESHKOL_BACKEND_AUDIO_NULL));
    CHECK_EQ(eshkol_audio_format_bytes(ESHKOL_AUDIO_FORMAT_S24), 3);
    eshkol_audio_config_t bad = { 48000, 0, ESHKOL_AUDIO_FORMAT_F32, 256 };
    eshkol_audio_device_t* d = NULL;
    CHECK_EQ(eshkol_audio_open_ex(ESHKOL_BACKEND_AUDIO_NULL, 0, NULL, &bad, NULL, NULL, 0, &d),
             ESHKOL_PLATFORM_EINVAL);
    CHECK_EQ(eshkol_audio_open_ex(ESHKOL_BACKEND_WASAPI, 0, NULL, &bad, NULL, NULL, 0, &d),
             ESHKOL_PLATFORM_EINVAL);
    audio_contract(ESHKOL_BACKEND_AUDIO_NULL, NULL, 1, 1);

    /* Bridge: per-sample staging becomes visible only at commit; every format
     * converts [-1, 1] and back within its quantum. */
    static const int formats[] = { ESHKOL_AUDIO_FORMAT_U8, ESHKOL_AUDIO_FORMAT_S16,
                                   ESHKOL_AUDIO_FORMAT_S24, ESHKOL_AUDIO_FORMAT_S32,
                                   ESHKOL_AUDIO_FORMAT_F32 };
    static const double quantum[] = { 1.0 / 127, 1.0 / 32767, 1.0 / 8388607,
                                      1.0 / 2147483647, 1e-7 };
    for (int f = 0; f < 5; ++f) {
        int64_t h = eshkol_plat_audio_open(ESHKOL_AUDIO_OUTPUT, 44100, 1, formats[f], 64, 256);
        CHECK(h > 0);
        const double vals[] = { 0.0, 0.5, -0.5, 1.0, -1.0, 0.123456 };
        for (int i = 0; i < 6; ++i) CHECK_EQ(eshkol_plat_audio_write(h, vals[i]), ESHKOL_PLATFORM_OK);
        CHECK_EQ(eshkol_plat_audio_stat(h, 11), 0);
        CHECK_EQ(eshkol_plat_audio_commit(h), 6);
        CHECK_EQ(eshkol_plat_audio_stat(h, 11), 6);
        uint64_t lease = 0; int stt = 0;
        eshkol_audio_device_t* dev = (eshkol_audio_device_t*)eshkol_plat_handle_borrow(
            h, ESHKOL_PLAT_KIND_AUDIO, &lease, &stt);
        CHECK(dev != NULL);
        if (dev) {
            uint32_t bps = eshkol_audio_format_bytes(formats[f]);
            int close_enough = 1;
            for (int i = 0; i < 6; ++i) {
                const uint8_t* p = dev->ring + (size_t)i * bps;
                double back;
                switch (formats[f]) {
                    case ESHKOL_AUDIO_FORMAT_U8: back = (p[0] - 128.0) / 127.0; break;
                    case ESHKOL_AUDIO_FORMAT_S16: { int16_t v; memcpy(&v, p, 2); back = v / 32767.0; break; }
                    case ESHKOL_AUDIO_FORMAT_S24: {
                        int32_t v = p[0] | (p[1] << 8) | (p[2] << 16);
                        if (v & 0x800000) v -= 0x1000000;
                        back = v / 8388607.0; break; }
                    case ESHKOL_AUDIO_FORMAT_S32: { int32_t v; memcpy(&v, p, 4); back = v / 2147483647.0; break; }
                    default: { float v; memcpy(&v, p, 4); back = v; break; }
                }
                if (fabs(back - vals[i]) > quantum[f]) close_enough = 0;
            }
            CHECK(close_enough);
            eshkol_plat_handle_release(h, lease);
        }
        CHECK_EQ(eshkol_plat_audio_read(h) != eshkol_plat_audio_read(h), 1);  /* NaN: output port */
        CHECK_EQ(eshkol_plat_audio_close(h), ESHKOL_PLATFORM_OK);
    }

    /* Bridge input: captured silence reads back as 0.0 sample by sample. */
    int64_t in = eshkol_plat_audio_open(ESHKOL_AUDIO_INPUT, 48000, 2, ESHKOL_AUDIO_FORMAT_S16, 128, 1024);
    CHECK(in > 0);
    CHECK_EQ(eshkol_plat_audio_control(in, 0), ESHKOL_PLATFORM_OK);
    eshkol_plat_sleep_ms(40);
    CHECK_EQ(eshkol_plat_audio_control(in, 1), ESHKOL_PLATFORM_OK);
    int avail = eshkol_plat_audio_available(in);
    CHECK(avail > 0);
    CHECK_EQ(eshkol_plat_audio_read(in), 0.0);
    CHECK_EQ(eshkol_plat_audio_read(in), 0.0);
    CHECK_EQ(eshkol_plat_audio_available(in), avail - 1);
    CHECK_EQ(eshkol_plat_audio_write(in, 0.5), ESHKOL_PLATFORM_ESTATE);
    CHECK(eshkol_plat_audio_stat(in, 0) > 0);
    CHECK_EQ(eshkol_plat_audio_stat(in, 14), ESHKOL_BACKEND_AUDIO_NULL);
    CHECK_EQ(eshkol_plat_audio_close(in), ESHKOL_PLATFORM_OK);
}

/* ── MIDI ──────────────────────────────────────────────────────────── */

static void midi_contract(int backend, const char* name) {
    eshkol_midi_port_t* out = NULL;
    eshkol_midi_port_t* in = NULL;
    CHECK_EQ(eshkol_midi_open_ex(backend, ESHKOL_MIDI_OUTPUT, name, &out), ESHKOL_PLATFORM_OK);
    if (!out) return;
    CHECK_EQ(eshkol_midi_open_ex(backend, ESHKOL_MIDI_INPUT, name, &in), ESHKOL_PLATFORM_OK);
    if (!in) { eshkol_midi_close(out); return; }
    CHECK_EQ(eshkol_platform_select_backend(ESHKOL_PLATFORM_MIDI, backend), ESHKOL_PLATFORM_OK);
    int found = 0;
    int n = eshkol_midi_source_count();
    for (int i = 0; i < n; ++i) {
        char buf[256];
        if (eshkol_midi_source_name(i, buf, sizeof buf) == ESHKOL_PLATFORM_OK && strcmp(buf, name) == 0)
            found = 1;
    }
    CHECK(found);

    const uint8_t note_on[3] = { 0x90, 60, 100 };
    const uint8_t cc[3] = { 0xB3, 7, 127 };
    const uint8_t pc[2] = { 0xC1, 5 };
    const uint8_t sysex[6] = { 0xF0, 0x7D, 0x01, 0x02, 0x03, 0xF7 };
    const uint8_t clock[1] = { 0xF8 };
    uint64_t t0 = eshkol_platform_time_us();
    CHECK_EQ(eshkol_midi_send(out, note_on, 3), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_midi_send(out, cc, 3), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_midi_send(out, pc, 2), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_midi_send(out, sysex, 6), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_midi_send(out, clock, 1), ESHKOL_PLATFORM_OK);
    const uint8_t* expect[5] = { note_on, cc, pc, sysex, clock };
    const int lens[5] = { 3, 3, 2, 6, 1 };
    int all = 1;
    for (int i = 0; i < 5; ++i) {
        uint8_t buf[64];
        uint64_t ts = 0;
        int got = eshkol_midi_receive_timeout(in, buf, sizeof buf, &ts, 2000);
        if (got != lens[i] || memcmp(buf, expect[i], (size_t)lens[i]) != 0 || ts < t0) {
            all = 0;
            fprintf(stderr, "  message %d: got %d bytes\n", i, got);
        }
    }
    CHECK(all);
    eshkol_midi_stats_t s;
    eshkol_midi_get_stats(out, &s);
    CHECK_EQ(s.sent, 5);
    eshkol_midi_get_stats(in, &s);
    CHECK_EQ(s.received, 5);
    eshkol_midi_close(in);
    eshkol_midi_close(out);
}

static void headless_midi(void) {
    CHECK_EQ(eshkol_midi_message_length(0x90), 3);
    CHECK_EQ(eshkol_midi_message_length(0xC0), 2);
    CHECK_EQ(eshkol_midi_message_length(0xF0), -1);
    CHECK_EQ(eshkol_midi_message_length(0xF8), 1);
    CHECK_EQ(eshkol_midi_message_length(0xF4), 0);
    CHECK_EQ(eshkol_midi_message_length(0x40), 0);

    midi_contract(ESHKOL_BACKEND_LOOPBACK, "eshkol-loopback-a");

    eshkol_midi_port_t* out = NULL;
    eshkol_midi_port_t* in = NULL;
    CHECK_EQ(eshkol_midi_open_ex(ESHKOL_BACKEND_LOOPBACK, ESHKOL_MIDI_INPUT, "no-such-source", &in),
             ESHKOL_PLATFORM_ENOTFOUND);
    CHECK_EQ(eshkol_midi_open_ex(ESHKOL_BACKEND_LOOPBACK, ESHKOL_MIDI_OUTPUT, "eshkol-src-b", &out),
             ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_midi_open_ex(ESHKOL_BACKEND_LOOPBACK, ESHKOL_MIDI_INPUT, "src-b", &in),
             ESHKOL_PLATFORM_OK);   /* substring match */
    if (!out || !in) return;

    /* Malformed messages are refused before they reach a backend. */
    const uint8_t short_note[2] = { 0x90, 60 };
    const uint8_t data_high[3] = { 0x90, 200, 1 };
    const uint8_t unterminated[3] = { 0xF0, 1, 2 };
    const uint8_t undefined[1] = { 0xF4 };
    CHECK_EQ(eshkol_midi_send(out, short_note, 2), ESHKOL_PLATFORM_EFORMAT);
    CHECK_EQ(eshkol_midi_send(out, data_high, 3), ESHKOL_PLATFORM_EFORMAT);
    CHECK_EQ(eshkol_midi_send(out, unterminated, 3), ESHKOL_PLATFORM_EFORMAT);
    CHECK_EQ(eshkol_midi_send(out, undefined, 1), ESHKOL_PLATFORM_EFORMAT);
    CHECK_EQ(eshkol_midi_send(in, short_note, 2), ESHKOL_PLATFORM_ESTATE);

    /* Stream parser: running status, interleaved real-time, split SysEx. */
    const uint8_t stream[] = { 0x90, 60, 100, 62, 101, 0xF8, 64, 102, 0xF0, 1, 2, 0xFE, 3, 0xF7,
                               0x45 /* orphan data byte: system common cleared status */ };
    eshkol_plat_midi_feed(in, stream, 14, 1);
    eshkol_plat_midi_feed(in, stream + 14, 1, 1);
    const uint8_t expect[][5] = { { 0x90, 60, 100 }, { 0x90, 62, 101 }, { 0xF8 }, { 0x90, 64, 102 },
                                  { 0xFE }, { 0xF0, 1, 2, 3, 0xF7 } };
    const int lens[] = { 3, 3, 1, 3, 1, 5 };
    int ok = 1;
    for (int i = 0; i < 6; ++i) {
        uint8_t buf[16];
        int got = eshkol_midi_receive(in, buf, sizeof buf, NULL);
        if (got != lens[i] || memcmp(buf, expect[i], (size_t)lens[i]) != 0) ok = 0;
    }
    CHECK(ok);
    CHECK_EQ(eshkol_midi_receive(in, (uint8_t[4]){0}, 4, NULL), 0);
    eshkol_midi_stats_t s;
    eshkol_midi_get_stats(in, &s);
    CHECK_EQ(s.malformed, 1);   /* 0x45 after F7: SysEx cleared running status */

    /* A too-small buffer keeps the message queued. */
    const uint8_t big[8] = { 0xF0, 1, 2, 3, 4, 5, 6, 0xF7 };
    CHECK_EQ(eshkol_midi_send(out, big, 8), ESHKOL_PLATFORM_OK);
    uint8_t small[4];
    CHECK_EQ(eshkol_midi_receive(in, small, 4, NULL), ESHKOL_PLATFORM_EFULL);
    uint8_t enough[8];
    CHECK_EQ(eshkol_midi_receive(in, enough, 8, NULL), 8);

    /* Ring overflow drops and counts whole messages; the ring wraps cleanly. */
    const uint8_t note[3] = { 0x80, 1, 2 };
    for (int i = 0; i < 6000; ++i) eshkol_midi_send(out, note, 3);
    eshkol_midi_get_stats(in, &s);
    CHECK(s.dropped > 0);
    uint64_t delivered = 0;
    uint8_t b3[3];
    while (eshkol_midi_receive(in, b3, 3, NULL) == 3) ++delivered;
    CHECK_EQ(delivered + s.dropped, 6000);
    for (int round = 0; round < 3; ++round) {
        for (int i = 0; i < 3000; ++i) eshkol_midi_send(out, note, 3);
        int cnt = 0;
        while (eshkol_midi_receive(in, b3, 3, NULL) == 3 && b3[0] == 0x80) ++cnt;
        CHECK_EQ(cnt, 3000);
    }

    /* Closing the source leaves the input open but silent. */
    eshkol_midi_close(out);
    CHECK_EQ(eshkol_midi_receive(in, b3, 3, NULL), 0);
    eshkol_midi_close(in);

    /* Bridge. */
    CHECK_EQ(eshkol_platform_select_backend(ESHKOL_PLATFORM_MIDI, ESHKOL_BACKEND_LOOPBACK),
             ESHKOL_PLATFORM_OK);
    int64_t ho = eshkol_plat_midi_open(ESHKOL_MIDI_OUTPUT, "bridge-out");
    int64_t hi = eshkol_plat_midi_open(ESHKOL_MIDI_INPUT, "bridge-out");
    CHECK(ho > 0 && hi > 0);
    CHECK_EQ(eshkol_plat_midi_source_count(), 1);
    CHECK(strcmp(eshkol_plat_midi_source_name(0), "bridge-out") == 0);
    CHECK(strcmp(eshkol_plat_midi_source_name(5), "") == 0);
    CHECK_EQ(eshkol_plat_midi_stage(ho, 0x92), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_stage(ho, 64), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_stage(ho, 90), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_stage(ho, 300), ESHKOL_PLATFORM_EINVAL);
    CHECK_EQ(eshkol_plat_midi_send_staged(ho), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_send_staged(ho), ESHKOL_PLATFORM_EFORMAT);  /* empty */
    CHECK_EQ(eshkol_plat_midi_receive(hi, 500), 3);
    CHECK_EQ(eshkol_plat_midi_byte(hi, 0), 0x92);
    CHECK_EQ(eshkol_plat_midi_byte(hi, 2), 90);
    CHECK_EQ(eshkol_plat_midi_byte(hi, 3), -1);
    CHECK(eshkol_plat_midi_timestamp(hi) > 0);
    CHECK_EQ(eshkol_plat_midi_stat(ho, 0), 1);
    CHECK_EQ(eshkol_plat_midi_stat(hi, 1), 1);
    CHECK_EQ(eshkol_plat_midi_stat(hi, 4), ESHKOL_BACKEND_LOOPBACK);
    CHECK_EQ(eshkol_plat_midi_receive(hi, 0), 0);
    CHECK_EQ(eshkol_plat_midi_close(hi), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_close(ho), ESHKOL_PLATFORM_OK);
    CHECK_EQ(eshkol_plat_midi_open(ESHKOL_MIDI_INPUT, ""), ESHKOL_PLATFORM_EINVAL);
}

/* ── native backends ───────────────────────────────────────────────── */

static int x11_mode(void) {
#if defined(ESHKOL_PLATFORM_HAVE_X11)
    if (!eshkol_platform_backend_available(ESHKOL_BACKEND_X11)) {
        printf("SKIP: X11 unavailable (libX11 or $DISPLAY)\n");
        return 77;
    }
    window_contract(ESHKOL_BACKEND_X11);

    /* Events synthesised by the X server reach the queue translated. */
    eshkol_window_t* w = NULL;
    CHECK_EQ(eshkol_window_create_ex(ESHKOL_BACKEND_X11, "x11 events", 50, 40, 0, &w),
             ESHKOL_PLATFORM_OK);
    if (!w) return 1;
    CHECK(drain_until(w, ESHKOL_EVENT_WINDOW_SHOW, NULL, 2000));
    CHECK(drain_until(w, ESHKOL_EVENT_WINDOW_EXPOSE, NULL, 2000));
    CHECK_EQ(eshkol_window_set_position(w, 13, 17), ESHKOL_PLATFORM_OK);
    eshkol_event_t ev;
    CHECK(drain_until(w, ESHKOL_EVENT_WINDOW_MOVE, &ev, 2000));
    eshkol_window_destroy(w);
    return 0;
#else
    printf("SKIP: built without Xlib headers\n");
    return 77;
#endif
}

static int alsa_mode(void) {
#if defined(__linux__)
    if (!eshkol_platform_backend_available(ESHKOL_BACKEND_ALSA)) {
        printf("SKIP: libasound unavailable\n");
        return 77;
    }
    /* alsa-lib's "null" PCM: the real ALSA open/params/write/read/delay path
     * without a sound card. It is not clocked, so period counts are untimed. */
    audio_contract(ESHKOL_BACKEND_ALSA, "null", 0, 1);
    if (eshkol_platform_backend_available(ESHKOL_BACKEND_ALSA_SEQ)) {
        midi_contract(ESHKOL_BACKEND_ALSA_SEQ, "eshkol-seq-test");
    } else {
        printf("NOTE: ALSA sequencer not accessible here; sequencer checks skipped\n");
    }
    return 0;
#else
    printf("SKIP: ALSA is Linux-only\n");
    return 77;
#endif
}

#if defined(__APPLE__)
int eshkol_test_cocoa_post(eshkol_window_t* w, int kind);

static int macos_mode(void) {
    int ran = 0;
    if (eshkol_platform_backend_available(ESHKOL_BACKEND_COCOA)) {
        ++ran;
        window_contract(ESHKOL_BACKEND_COCOA);
        eshkol_window_t* w = NULL;
        CHECK_EQ(eshkol_window_create_ex(ESHKOL_BACKEND_COCOA, "cocoa events", 60, 40,
                                         ESHKOL_WINDOW_HIDDEN, &w), ESHKOL_PLATFORM_OK);
        if (w) {
            eshkol_event_t ev;
            CHECK(eshkol_test_cocoa_post(w, 0));
            CHECK(drain_until(w, ESHKOL_EVENT_KEY_DOWN, &ev, 2000));
            CHECK_EQ(ev.u.key.keycode, 'a');
            CHECK_EQ(ev.u.key.unicode, 'A');
            CHECK(ev.u.key.modifiers & ESHKOL_MOD_SHIFT);
            CHECK(eshkol_test_cocoa_post(w, 1));
            CHECK(drain_until(w, ESHKOL_EVENT_MOUSE_DOWN, &ev, 2000));
            CHECK_EQ(ev.u.mouse.button, ESHKOL_BUTTON_LEFT);
            CHECK_EQ(ev.u.mouse.x, 10);
            CHECK_EQ(ev.u.mouse.y, 5);
            CHECK(eshkol_test_cocoa_post(w, 2));
            CHECK(drain_until(w, ESHKOL_EVENT_KEY_UP, &ev, 2000));
            CHECK_EQ(ev.u.key.keycode, ESHKOL_KEY_ESCAPE);
            eshkol_window_destroy(w);
        }
    } else {
        printf("NOTE: no window-server session; Cocoa checks skipped\n");
    }
    if (eshkol_platform_backend_available(ESHKOL_BACKEND_COREAUDIO)) {
        ++ran;
        /* Output only: capture needs the user's microphone consent, which an
         * unattended run cannot give (documented manual check). */
        audio_contract(ESHKOL_BACKEND_COREAUDIO, NULL, 1, 0);
    }
    if (eshkol_platform_backend_available(ESHKOL_BACKEND_COREMIDI)) {
        ++ran;
        midi_contract(ESHKOL_BACKEND_COREMIDI, "eshkol-coremidi-test");
    }
    return ran ? 0 : 77;
}
#endif

int main(int argc, char** argv) {
    const char* mode = argc > 1 ? argv[1] : "headless";
    eshkol_platform_init();
    int rc = 0;
    if (strcmp(mode, "headless") == 0) {
        headless_windows();
        headless_bridge();
        headless_audio();
        headless_midi();
    } else if (strcmp(mode, "x11") == 0) {
        rc = x11_mode();
    } else if (strcmp(mode, "alsa") == 0) {
        rc = alsa_mode();
#if defined(__APPLE__)
    } else if (strcmp(mode, "macos") == 0) {
        rc = macos_mode();
#endif
    } else {
        fprintf(stderr, "usage: platform_test headless|x11|alsa|macos\n");
        return 2;
    }
    if (rc == 77) return 77;
    eshkol_platform_quit();
    if (g_failures || rc) {
        printf("FAIL: platform %s %d of %d checks failed\n", mode, g_failures, g_checks);
        return 1;
    }
    printf("PASS: platform %s %d checks\n", mode, g_checks);
    return 0;
}
