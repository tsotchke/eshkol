/*
 * cocoa_event_inject.m — posts synthetic AppKit events to an Eshkol window
 * so platform_test can verify Cocoa event translation (ADR-0053).
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#import <AppKit/AppKit.h>
#include "../../lib/platform/platform_internal.h"

/* Mirrors the private layout head of window_cocoa.m's per-window state. */
typedef struct { NSWindow* window; } cocoa_head_t;

int eshkol_test_cocoa_post(eshkol_window_t* w, int kind) {
    Class cEvent = NSClassFromString(@"NSEvent");
    Class cApp = NSClassFromString(@"NSApplication");
    if (!w || !w->impl || !cEvent || !cApp) return 0;
    NSWindow* win = ((cocoa_head_t*)w->impl)->window;
    NSEvent* e = nil;
    if (kind == 0) {          /* 'a' key down with shift */
        e = [cEvent keyEventWithType:NSEventTypeKeyDown location:NSMakePoint(0, 0)
                       modifierFlags:NSEventModifierFlagShift timestamp:0
                        windowNumber:[win windowNumber] context:nil characters:@"A"
         charactersIgnoringModifiers:@"A" isARepeat:NO keyCode:0x00];
    } else if (kind == 1) {   /* left button down at view (10, 5) from the top */
        NSPoint p = NSMakePoint(10, w->height - 5);
        e = [cEvent mouseEventWithType:NSEventTypeLeftMouseDown location:p modifierFlags:0
                             timestamp:0 windowNumber:[win windowNumber] context:nil
                           eventNumber:0 clickCount:1 pressure:1.0];
    } else {                  /* escape key up */
        e = [cEvent keyEventWithType:NSEventTypeKeyUp location:NSMakePoint(0, 0)
                       modifierFlags:0 timestamp:0 windowNumber:[win windowNumber]
                             context:nil characters:@"\x1b" charactersIgnoringModifiers:@"\x1b"
                           isARepeat:NO keyCode:0x35];
    }
    if (!e) return 0;
    [[cApp sharedApplication] postEvent:e atStart:NO];
    return 1;
}
