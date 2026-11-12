/**
 * @file event.h
 * @brief Platform event record: keyboard, mouse, touch, window, timer and
 *        user events, with platform-independent key codes (ADR-0053).
 *
 * Event type values follow the event subtype numbering of
 * docs/future/MULTIMEDIA_SYSTEM_ARCHITECTURE.md section 2.2.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_PLATFORM_EVENT_H
#define ESHKOL_PLATFORM_EVENT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_EVENT_NONE          = -1,
    ESHKOL_EVENT_KEY_DOWN      = 0,
    ESHKOL_EVENT_KEY_UP        = 1,
    ESHKOL_EVENT_KEY_REPEAT    = 2,
    ESHKOL_EVENT_TEXT_INPUT    = 3,
    ESHKOL_EVENT_MOUSE_MOVE    = 4,
    ESHKOL_EVENT_MOUSE_DOWN    = 5,
    ESHKOL_EVENT_MOUSE_UP      = 6,
    ESHKOL_EVENT_MOUSE_SCROLL  = 7,
    ESHKOL_EVENT_MOUSE_ENTER   = 8,
    ESHKOL_EVENT_MOUSE_LEAVE   = 9,
    ESHKOL_EVENT_TOUCH_BEGIN   = 10,
    ESHKOL_EVENT_TOUCH_MOVE    = 11,
    ESHKOL_EVENT_TOUCH_END     = 12,
    ESHKOL_EVENT_TOUCH_CANCEL  = 13,
    ESHKOL_EVENT_WINDOW_RESIZE = 14,
    ESHKOL_EVENT_WINDOW_MOVE   = 15,
    ESHKOL_EVENT_WINDOW_CLOSE  = 16,
    ESHKOL_EVENT_WINDOW_FOCUS  = 17,
    ESHKOL_EVENT_WINDOW_BLUR   = 18,
    ESHKOL_EVENT_WINDOW_SHOW   = 19,
    ESHKOL_EVENT_WINDOW_HIDE   = 20,
    ESHKOL_EVENT_WINDOW_EXPOSE = 21,
    ESHKOL_EVENT_QUIT          = 22,
    ESHKOL_EVENT_TIMER         = 23,
    ESHKOL_EVENT_USER          = 31
} eshkol_event_type_t;

/* Modifier flags */
#define ESHKOL_MOD_SHIFT     0x0001u
#define ESHKOL_MOD_CTRL      0x0002u
#define ESHKOL_MOD_ALT       0x0004u
#define ESHKOL_MOD_META      0x0008u
#define ESHKOL_MOD_CAPSLOCK  0x0010u
#define ESHKOL_MOD_NUMLOCK   0x0020u

/* Mouse buttons: `button` holds one of these numbers, `buttons` a mask of
 * (1u << (number - 1)) bits. */
#define ESHKOL_BUTTON_LEFT   1u
#define ESHKOL_BUTTON_MIDDLE 2u
#define ESHKOL_BUTTON_RIGHT  3u
#define ESHKOL_BUTTON_X1     4u
#define ESHKOL_BUTTON_X2     5u

/* Key codes. Printable keys use the unshifted US-layout character code
 * ('a'..'z', '0'..'9', ' ', punctuation); control keys use the ASCII
 * control value; everything else lives at 0x1000 and above. */
#define ESHKOL_KEY_UNKNOWN    0x0000u
#define ESHKOL_KEY_BACKSPACE  0x0008u
#define ESHKOL_KEY_TAB        0x0009u
#define ESHKOL_KEY_RETURN     0x000Du
#define ESHKOL_KEY_ESCAPE     0x001Bu
#define ESHKOL_KEY_SPACE      0x0020u
#define ESHKOL_KEY_DELETE     0x007Fu
#define ESHKOL_KEY_LEFT       0x1001u
#define ESHKOL_KEY_RIGHT      0x1002u
#define ESHKOL_KEY_UP         0x1003u
#define ESHKOL_KEY_DOWN       0x1004u
#define ESHKOL_KEY_HOME       0x1005u
#define ESHKOL_KEY_END        0x1006u
#define ESHKOL_KEY_PAGE_UP    0x1007u
#define ESHKOL_KEY_PAGE_DOWN  0x1008u
#define ESHKOL_KEY_INSERT     0x1009u
#define ESHKOL_KEY_F1         0x1101u  /* F1..F12 = 0x1101..0x110C */
#define ESHKOL_KEY_SHIFT      0x1201u
#define ESHKOL_KEY_CTRL       0x1202u
#define ESHKOL_KEY_ALT        0x1203u
#define ESHKOL_KEY_META       0x1204u
#define ESHKOL_KEY_CAPSLOCK   0x1205u

typedef struct eshkol_event {
    int32_t  type;          /**< eshkol_event_type_t */
    uint32_t window_id;     /**< source window (0 when none) */
    uint64_t timestamp;     /**< eshkol_platform_time_us() at arrival */
    uint32_t device_id;     /**< source device (0 when none) */
    uint32_t reserved;
    union {
        struct { uint32_t keycode, scancode, modifiers, unicode; } key;
        struct {
            int32_t x, y, dx, dy;
            uint32_t button, buttons;
            int32_t scroll_x, scroll_y;
            uint32_t modifiers;
        } mouse;
        struct { uint32_t finger_id; float x, y, pressure; } touch;
        struct { int32_t x, y, width, height; } window;
        struct { uint32_t timer_id; uint64_t interval; } timer;
        struct { uint32_t code; uint64_t data1, data2; } user;
    } u;
} eshkol_event_t;

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_PLATFORM_EVENT_H */
