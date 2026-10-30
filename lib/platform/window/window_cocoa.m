/*
 * window_cocoa.m — Cocoa display backend (ADR-0053).
 *
 * AppKit is loaded at run time and every AppKit class is looked up by name;
 * the view and window-delegate classes are registered with the Objective-C
 * runtime on first use. The runtime archive therefore references no AppKit
 * symbol, and the macOS link line stays Foundation/CoreGraphics/objc only.
 *
 * All calls must come from the process main thread (an AppKit rule). Events
 * for every Eshkol window are translated while any window pumps, and each is
 * routed to the queue of the window it belongs to.
 *
 * Compiled as Objective-C with manual reference counting.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#if defined(__APPLE__)

#import <AppKit/AppKit.h>
#include <ApplicationServices/ApplicationServices.h>
#include <dlfcn.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <stdlib.h>
#include <string.h>

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_loaded = 0;
static Class clsApplication, clsWindow, clsView, clsDate, clsGraphicsContext, clsEvent;
static Class clsEshkolView, clsEshkolDelegate;
static int g_app_ready = 0;

typedef struct {
    NSWindow* window;
    NSView* view;
    id delegate;
    uint32_t* presented;
    uint32_t pw, ph;
    uint32_t buttons;
} cocoa_t;

static eshkol_window_t* owner_of(id obj) {
    if (!obj) return NULL;
    Ivar iv = class_getInstanceVariable(object_getClass(obj), "eshkolWindow");
    if (!iv) return NULL;
    return (eshkol_window_t*)(uintptr_t)object_getIvar(obj, iv);
}

static void set_owner(id obj, eshkol_window_t* w) {
    Ivar iv = class_getInstanceVariable(object_getClass(obj), "eshkolWindow");
    object_setIvar(obj, iv, (id)(uintptr_t)w);
}

/* ── runtime-registered view ───────────────────────────────────────── */

static void view_draw_rect(id self, SEL _cmd, NSRect dirty) {
    (void)_cmd; (void)dirty;
    eshkol_window_t* w = owner_of(self);
    if (!w || !w->impl) return;
    cocoa_t* c = (cocoa_t*)w->impl;
    if (!c->presented) return;
    CGContextRef ctx = [[clsGraphicsContext currentContext] CGContext];
    /* Tag pixels with the window's own colour space so presenting performs no
     * colour conversion: surface values are the values the display receives. */
    CGColorSpaceRef cs = NULL;
    NSColorSpace* wcs = [[self window] colorSpace];
    if (wcs && [wcs CGColorSpace]) cs = CGColorSpaceRetain([wcs CGColorSpace]);
    if (!cs) cs = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef dp = CGDataProviderCreateWithData(NULL, c->presented,
                                                        (size_t)c->pw * c->ph * 4u, NULL);
    CGImageRef img = CGImageCreate(c->pw, c->ph, 8, 32, (size_t)c->pw * 4u, cs,
                                   kCGImageAlphaNoneSkipFirst | kCGBitmapByteOrder32Little,
                                   dp, NULL, false, kCGRenderingIntentDefault);
    if (img) {
        CGContextSetInterpolationQuality(ctx, kCGInterpolationNone);
        CGContextDrawImage(ctx, CGRectMake(0, 0, c->pw, c->ph), img);
        CGImageRelease(img);
    }
    CGDataProviderRelease(dp);
    CGColorSpaceRelease(cs);
}

static BOOL view_yes(id self, SEL _cmd) { (void)self; (void)_cmd; return YES; }
static BOOL view_no(id self, SEL _cmd) { (void)self; (void)_cmd; return NO; }
/* Key events are translated in the pump; swallow them here so AppKit does
 * not walk the responder chain and beep. */
static void view_swallow(id self, SEL _cmd, id ev) { (void)self; (void)_cmd; (void)ev; }

/* ── runtime-registered window delegate ────────────────────────────── */

static void emit_simple(eshkol_window_t* w, int type) {
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    ev.type = type;
    ev.u.window.x = w->x;
    ev.u.window.y = w->y;
    ev.u.window.width = w->width;
    ev.u.window.height = w->height;
    eshkol_plat_window_emit(w, &ev);
}

static BOOL delegate_should_close(id self, SEL _cmd, id sender) {
    (void)_cmd; (void)sender;
    eshkol_window_t* w = owner_of(self);
    if (w) emit_simple(w, ESHKOL_EVENT_WINDOW_CLOSE);
    return NO;  /* the program decides when to destroy the window */
}

static void delegate_did_resize(id self, SEL _cmd, id note) {
    (void)_cmd; (void)note;
    eshkol_window_t* w = owner_of(self);
    if (!w || !w->impl) return;
    cocoa_t* c = (cocoa_t*)w->impl;
    NSRect r = [c->view bounds];
    int nw = (int)r.size.width, nh = (int)r.size.height;
    if ((nw != w->width || nh != w->height) &&
        eshkol_plat_window_resize_surface(w, nw, nh) == ESHKOL_PLATFORM_OK)
        emit_simple(w, ESHKOL_EVENT_WINDOW_RESIZE);
}

static void delegate_did_move(id self, SEL _cmd, id note) {
    (void)_cmd; (void)note;
    eshkol_window_t* w = owner_of(self);
    if (!w || !w->impl) return;
    NSRect f = [((cocoa_t*)w->impl)->window frame];
    w->x = (int)f.origin.x;
    w->y = (int)f.origin.y;
    emit_simple(w, ESHKOL_EVENT_WINDOW_MOVE);
}

static void delegate_focus(id self, SEL _cmd, id note) {
    (void)_cmd; (void)note;
    eshkol_window_t* w = owner_of(self);
    if (w) emit_simple(w, ESHKOL_EVENT_WINDOW_FOCUS);
}

static void delegate_blur(id self, SEL _cmd, id note) {
    (void)_cmd; (void)note;
    eshkol_window_t* w = owner_of(self);
    if (w) emit_simple(w, ESHKOL_EVENT_WINDOW_BLUR);
}

static void cocoa_load(void) {
    if (!dlopen("/System/Library/Frameworks/AppKit.framework/AppKit", RTLD_NOW | RTLD_GLOBAL))
        return;
    clsApplication = NSClassFromString(@"NSApplication");
    clsWindow = NSClassFromString(@"NSWindow");
    clsView = NSClassFromString(@"NSView");
    clsDate = NSClassFromString(@"NSDate");
    clsGraphicsContext = NSClassFromString(@"NSGraphicsContext");
    clsEvent = NSClassFromString(@"NSEvent");
    Class clsObject = NSClassFromString(@"NSObject");
    if (!clsApplication || !clsWindow || !clsView || !clsDate || !clsGraphicsContext || !clsEvent || !clsObject)
        return;

    clsEshkolView = objc_allocateClassPair(clsView, "EshkolPlatformView", 0);
    class_addIvar(clsEshkolView, "eshkolWindow", sizeof(void*), (uint8_t)__builtin_ctzl(sizeof(void*)), "^v");
    class_addMethod(clsEshkolView, @selector(drawRect:), (IMP)view_draw_rect, "v@:{CGRect={CGPoint=dd}{CGSize=dd}}");
    class_addMethod(clsEshkolView, @selector(acceptsFirstResponder), (IMP)view_yes, "c@:");
    class_addMethod(clsEshkolView, @selector(isFlipped), (IMP)view_no, "c@:");
    class_addMethod(clsEshkolView, @selector(keyDown:), (IMP)view_swallow, "v@:@");
    class_addMethod(clsEshkolView, @selector(keyUp:), (IMP)view_swallow, "v@:@");
    class_addMethod(clsEshkolView, @selector(flagsChanged:), (IMP)view_swallow, "v@:@");
    objc_registerClassPair(clsEshkolView);

    clsEshkolDelegate = objc_allocateClassPair(clsObject, "EshkolPlatformWindowDelegate", 0);
    class_addIvar(clsEshkolDelegate, "eshkolWindow", sizeof(void*), (uint8_t)__builtin_ctzl(sizeof(void*)), "^v");
    class_addMethod(clsEshkolDelegate, @selector(windowShouldClose:), (IMP)delegate_should_close, "c@:@");
    class_addMethod(clsEshkolDelegate, @selector(windowDidResize:), (IMP)delegate_did_resize, "v@:@");
    class_addMethod(clsEshkolDelegate, @selector(windowDidMove:), (IMP)delegate_did_move, "v@:@");
    class_addMethod(clsEshkolDelegate, @selector(windowDidBecomeKey:), (IMP)delegate_focus, "v@:@");
    class_addMethod(clsEshkolDelegate, @selector(windowDidResignKey:), (IMP)delegate_blur, "v@:@");
    objc_registerClassPair(clsEshkolDelegate);
    g_loaded = 1;
}

static int cocoa_available(void) {
    pthread_once(&g_once, cocoa_load);
    if (!g_loaded) return 0;
    /* A window server session must exist (not the case for daemons). */
    CFDictionaryRef session = CGSessionCopyCurrentDictionary();
    if (!session) return 0;
    CFRelease(session);
    return 1;
}

static void app_ready(int visible) {
    if (g_app_ready) return;
    NSApplication* app = [clsApplication sharedApplication];
    [app setActivationPolicy:visible ? NSApplicationActivationPolicyRegular
                                     : NSApplicationActivationPolicyAccessory];
    [app finishLaunching];
    g_app_ready = 1;
}

static int cocoa_create(eshkol_window_t* w, const char* title) {
    if (![NSThread isMainThread]) return ESHKOL_PLATFORM_ESTATE;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    app_ready(!(w->flags & ESHKOL_WINDOW_HIDDEN));
    cocoa_t* c = (cocoa_t*)calloc(1, sizeof(*c));
    if (!c) { [pool drain]; return ESHKOL_PLATFORM_ENOMEM; }
    NSWindowStyleMask style = (w->flags & ESHKOL_WINDOW_BORDERLESS)
        ? NSWindowStyleMaskBorderless
        : (NSWindowStyleMaskTitled | NSWindowStyleMaskClosable | NSWindowStyleMaskMiniaturizable);
    if (w->flags & ESHKOL_WINDOW_RESIZABLE) style |= NSWindowStyleMaskResizable;
    NSRect rect = NSMakeRect(w->x, w->y, w->width, w->height);
    c->window = [[clsWindow alloc] initWithContentRect:rect styleMask:style
                                             backing:NSBackingStoreBuffered defer:NO];
    if (!c->window) { free(c); [pool drain]; return ESHKOL_PLATFORM_EDEVICE; }
    [c->window setReleasedWhenClosed:NO];
    [c->window setTitle:[NSString stringWithUTF8String:title]];
    [c->window setAcceptsMouseMovedEvents:YES];
    c->view = [[clsEshkolView alloc] initWithFrame:NSMakeRect(0, 0, w->width, w->height)];
    set_owner(c->view, w);
    [c->window setContentView:c->view];
    c->delegate = [[clsEshkolDelegate alloc] init];
    set_owner(c->delegate, w);
    [c->window setDelegate:c->delegate];
    if (w->flags & ESHKOL_WINDOW_ALWAYS_TOP) [c->window setLevel:NSFloatingWindowLevel];
    w->impl = c;
    if (!(w->flags & ESHKOL_WINDOW_HIDDEN)) {
        [c->window makeKeyAndOrderFront:nil];
        [[clsApplication sharedApplication] activateIgnoringOtherApps:YES];
        emit_simple(w, ESHKOL_EVENT_WINDOW_SHOW);
    }
    [pool drain];
    return ESHKOL_PLATFORM_OK;
}

static void cocoa_destroy(eshkol_window_t* w) {
    cocoa_t* c = (cocoa_t*)w->impl;
    if (!c) return;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    [c->window setDelegate:nil];
    set_owner(c->view, NULL);
    set_owner(c->delegate, NULL);
    [c->window orderOut:nil];
    [c->window close];
    [c->window release];
    [c->view release];
    [c->delegate release];
    free(c->presented);
    free(c);
    w->impl = NULL;
    [pool drain];
}

static int cocoa_set_size(eshkol_window_t* w, int width, int height) {
    cocoa_t* c = (cocoa_t*)w->impl;
    [c->window setContentSize:NSMakeSize(width, height)];  /* windowDidResize: reports it */
    if (w->width != width || w->height != height) {
        /* Hidden windows may not deliver the notification synchronously. */
        if (eshkol_plat_window_resize_surface(w, width, height) == ESHKOL_PLATFORM_OK)
            emit_simple(w, ESHKOL_EVENT_WINDOW_RESIZE);
    }
    return ESHKOL_PLATFORM_OK;
}

static int cocoa_set_position(eshkol_window_t* w, int x, int y) {
    cocoa_t* c = (cocoa_t*)w->impl;
    [c->window setFrameOrigin:NSMakePoint(x, y)];
    if (w->x != x || w->y != y) {
        w->x = x;
        w->y = y;
        emit_simple(w, ESHKOL_EVENT_WINDOW_MOVE);
    }
    return ESHKOL_PLATFORM_OK;
}

static int cocoa_set_title(eshkol_window_t* w, const char* title) {
    cocoa_t* c = (cocoa_t*)w->impl;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    [c->window setTitle:[NSString stringWithUTF8String:title]];
    [pool drain];
    return ESHKOL_PLATFORM_OK;
}

static int cocoa_present(eshkol_window_t* w) {
    cocoa_t* c = (cocoa_t*)w->impl;
    size_t n = (size_t)w->surface.width * w->surface.height;
    if (c->pw != w->surface.width || c->ph != w->surface.height) {
        uint32_t* p = (uint32_t*)realloc(c->presented, n * 4u);
        if (!p) return ESHKOL_PLATFORM_ENOMEM;
        c->presented = p;
        c->pw = w->surface.width;
        c->ph = w->surface.height;
    }
    for (uint32_t r = 0; r < c->ph; ++r)
        memcpy(c->presented + (size_t)r * c->pw,
               w->surface.pixels + (size_t)r * w->surface.pitch, (size_t)c->pw * 4u);
    [c->view setNeedsDisplay:YES];
    [c->view displayIfNeeded];
    return ESHKOL_PLATFORM_OK;
}

static int cocoa_read_presented(eshkol_window_t* w, int x, int y, uint32_t* out) {
    cocoa_t* c = (cocoa_t*)w->impl;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSRect b = [c->view bounds];
    NSBitmapImageRep* rep = [c->view bitmapImageRepForCachingDisplayInRect:b];
    int st = ESHKOL_PLATFORM_EDEVICE;
    if (rep) {
        [c->view cacheDisplayInRect:b toBitmapImageRep:rep];
        /* Read the raw rendered sample (no NSColor conversion). */
        double sx = (double)[rep pixelsWide] / b.size.width;
        double sy = (double)[rep pixelsHigh] / b.size.height;
        NSInteger px = (NSInteger)(x * sx), py = (NSInteger)(y * sy);
        NSInteger spp = [rep samplesPerPixel];
        const unsigned char* data = [rep bitmapData];
        if (data && [rep bitsPerSample] == 8 && ![rep isPlanar] && spp >= 3 &&
            px < [rep pixelsWide] && py < [rep pixelsHigh]) {
            const unsigned char* p = data + py * [rep bytesPerRow] + px * ([rep bitsPerPixel] / 8);
            int first = ([rep bitmapFormat] & NSBitmapFormatAlphaFirst) && spp == 4 ? 1 : 0;
            *out = ((uint32_t)p[first] << 16) | ((uint32_t)p[first + 1] << 8) | p[first + 2];
            st = ESHKOL_PLATFORM_OK;
        }
    }
    [pool drain];
    return st;
}

/* ── event translation ─────────────────────────────────────────────── */

static uint32_t map_flags(NSEventModifierFlags f) {
    uint32_t m = 0;
    if (f & NSEventModifierFlagShift) m |= ESHKOL_MOD_SHIFT;
    if (f & NSEventModifierFlagControl) m |= ESHKOL_MOD_CTRL;
    if (f & NSEventModifierFlagOption) m |= ESHKOL_MOD_ALT;
    if (f & NSEventModifierFlagCommand) m |= ESHKOL_MOD_META;
    if (f & NSEventModifierFlagCapsLock) m |= ESHKOL_MOD_CAPSLOCK;
    return m;
}

static uint32_t map_keycode(unsigned short kc, NSString* chars) {
    switch (kc) {
        case 0x24: case 0x4C: return ESHKOL_KEY_RETURN;
        case 0x30: return ESHKOL_KEY_TAB;
        case 0x31: return ESHKOL_KEY_SPACE;
        case 0x33: return ESHKOL_KEY_BACKSPACE;
        case 0x35: return ESHKOL_KEY_ESCAPE;
        case 0x75: return ESHKOL_KEY_DELETE;
        case 0x7B: return ESHKOL_KEY_LEFT;
        case 0x7C: return ESHKOL_KEY_RIGHT;
        case 0x7D: return ESHKOL_KEY_DOWN;
        case 0x7E: return ESHKOL_KEY_UP;
        case 0x73: return ESHKOL_KEY_HOME;
        case 0x77: return ESHKOL_KEY_END;
        case 0x74: return ESHKOL_KEY_PAGE_UP;
        case 0x79: return ESHKOL_KEY_PAGE_DOWN;
        case 0x72: return ESHKOL_KEY_INSERT;
        case 0x7A: return ESHKOL_KEY_F1 + 0;  case 0x78: return ESHKOL_KEY_F1 + 1;
        case 0x63: return ESHKOL_KEY_F1 + 2;  case 0x76: return ESHKOL_KEY_F1 + 3;
        case 0x60: return ESHKOL_KEY_F1 + 4;  case 0x61: return ESHKOL_KEY_F1 + 5;
        case 0x62: return ESHKOL_KEY_F1 + 6;  case 0x64: return ESHKOL_KEY_F1 + 7;
        case 0x65: return ESHKOL_KEY_F1 + 8;  case 0x6D: return ESHKOL_KEY_F1 + 9;
        case 0x67: return ESHKOL_KEY_F1 + 10; case 0x6F: return ESHKOL_KEY_F1 + 11;
        case 0x38: case 0x3C: return ESHKOL_KEY_SHIFT;
        case 0x3B: case 0x3E: return ESHKOL_KEY_CTRL;
        case 0x3A: case 0x3D: return ESHKOL_KEY_ALT;
        case 0x37: case 0x36: return ESHKOL_KEY_META;
        case 0x39: return ESHKOL_KEY_CAPSLOCK;
        default: break;
    }
    if (chars && [chars length] > 0) {
        unichar ch = [chars characterAtIndex:0];
        if (ch >= 'A' && ch <= 'Z') return (uint32_t)(ch - 'A' + 'a');
        if (ch >= 0x20 && ch <= 0x7E) return ch;
    }
    return ESHKOL_KEY_UNKNOWN;
}

static int modifier_down(unsigned short kc, NSEventModifierFlags f) {
    switch (kc) {
        case 0x38: case 0x3C: return (f & NSEventModifierFlagShift) != 0;
        case 0x3B: case 0x3E: return (f & NSEventModifierFlagControl) != 0;
        case 0x3A: case 0x3D: return (f & NSEventModifierFlagOption) != 0;
        case 0x37: case 0x36: return (f & NSEventModifierFlagCommand) != 0;
        case 0x39: return (f & NSEventModifierFlagCapsLock) != 0;
        default: return 0;
    }
}

static void translate(NSEvent* e) {
    NSWindow* nw = [e window];
    eshkol_window_t* w = nw ? owner_of([nw delegate]) : NULL;
    if (!w || !w->impl) return;
    cocoa_t* c = (cocoa_t*)w->impl;
    eshkol_event_t ev;
    memset(&ev, 0, sizeof(ev));
    NSEventType t = [e type];
    switch (t) {
        case NSEventTypeKeyDown:
        case NSEventTypeKeyUp: {
            ev.type = t == NSEventTypeKeyUp ? ESHKOL_EVENT_KEY_UP
                    : [e isARepeat] ? ESHKOL_EVENT_KEY_REPEAT : ESHKOL_EVENT_KEY_DOWN;
            ev.u.key.scancode = [e keyCode];
            ev.u.key.keycode = map_keycode([e keyCode], [e charactersIgnoringModifiers]);
            ev.u.key.modifiers = map_flags([e modifierFlags]);
            NSString* chars = [e characters];
            if (chars && [chars length] > 0) {
                unichar ch = [chars characterAtIndex:0];
                if (ch < 0xF700 || ch > 0xF8FF) ev.u.key.unicode = ch;
            }
            break;
        }
        case NSEventTypeFlagsChanged:
            ev.type = modifier_down([e keyCode], [e modifierFlags]) ? ESHKOL_EVENT_KEY_DOWN
                                                                     : ESHKOL_EVENT_KEY_UP;
            ev.u.key.scancode = [e keyCode];
            ev.u.key.keycode = map_keycode([e keyCode], nil);
            ev.u.key.modifiers = map_flags([e modifierFlags]);
            break;
        case NSEventTypeLeftMouseDown: case NSEventTypeLeftMouseUp:
        case NSEventTypeRightMouseDown: case NSEventTypeRightMouseUp:
        case NSEventTypeOtherMouseDown: case NSEventTypeOtherMouseUp:
        case NSEventTypeMouseMoved: case NSEventTypeLeftMouseDragged:
        case NSEventTypeRightMouseDragged: case NSEventTypeOtherMouseDragged:
        case NSEventTypeScrollWheel:
        case NSEventTypeMouseEntered: case NSEventTypeMouseExited: {
            NSPoint p = [c->view convertPoint:[e locationInWindow] fromView:nil];
            ev.u.mouse.x = (int32_t)p.x;
            ev.u.mouse.y = (int32_t)(w->height - p.y);
            ev.u.mouse.modifiers = map_flags([e modifierFlags]);
            uint32_t num = 0;
            if (t == NSEventTypeLeftMouseDown || t == NSEventTypeLeftMouseUp) num = ESHKOL_BUTTON_LEFT;
            else if (t == NSEventTypeRightMouseDown || t == NSEventTypeRightMouseUp) num = ESHKOL_BUTTON_RIGHT;
            else if (t == NSEventTypeOtherMouseDown || t == NSEventTypeOtherMouseUp) {
                NSInteger b = [e buttonNumber];
                num = b == 2 ? ESHKOL_BUTTON_MIDDLE : b == 3 ? ESHKOL_BUTTON_X1
                    : b == 4 ? ESHKOL_BUTTON_X2 : 0;
            }
            if (t == NSEventTypeLeftMouseDown || t == NSEventTypeRightMouseDown ||
                t == NSEventTypeOtherMouseDown) {
                ev.type = ESHKOL_EVENT_MOUSE_DOWN;
                if (num) c->buttons |= 1u << (num - 1);
            } else if (t == NSEventTypeLeftMouseUp || t == NSEventTypeRightMouseUp ||
                       t == NSEventTypeOtherMouseUp) {
                ev.type = ESHKOL_EVENT_MOUSE_UP;
                if (num) c->buttons &= ~(1u << (num - 1));
            } else if (t == NSEventTypeScrollWheel) {
                ev.type = ESHKOL_EVENT_MOUSE_SCROLL;
                ev.u.mouse.scroll_x = (int32_t)lround([e scrollingDeltaX]);
                ev.u.mouse.scroll_y = (int32_t)lround([e scrollingDeltaY]);
            } else if (t == NSEventTypeMouseEntered) {
                ev.type = ESHKOL_EVENT_MOUSE_ENTER;
            } else if (t == NSEventTypeMouseExited) {
                ev.type = ESHKOL_EVENT_MOUSE_LEAVE;
            } else {
                ev.type = ESHKOL_EVENT_MOUSE_MOVE;
                ev.u.mouse.dx = (int32_t)lround([e deltaX]);
                ev.u.mouse.dy = (int32_t)lround([e deltaY]);
            }
            if (ev.type == ESHKOL_EVENT_MOUSE_DOWN || ev.type == ESHKOL_EVENT_MOUSE_UP) {
                if (!num) return;
                ev.u.mouse.button = num;
            }
            ev.u.mouse.buttons = c->buttons;
            break;
        }
        default:
            return;
    }
    eshkol_plat_window_emit(w, &ev);
}

static void cocoa_pump(eshkol_window_t* w) {
    (void)w;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSApplication* app = [clsApplication sharedApplication];
    for (;;) {
        NSEvent* e = [app nextEventMatchingMask:NSEventMaskAny
                                      untilDate:[clsDate distantPast]
                                         inMode:NSDefaultRunLoopMode
                                        dequeue:YES];
        if (!e) break;
        translate(e);
        [app sendEvent:e];
    }
    [app updateWindows];
    [pool drain];
}

static void cocoa_wait_os(eshkol_window_t* w, int timeout_ms) {
    (void)w;
    NSAutoreleasePool* pool = [[NSAutoreleasePool alloc] init];
    NSDate* until = timeout_ms < 0 ? [clsDate distantFuture]
                                   : [clsDate dateWithTimeIntervalSinceNow:timeout_ms / 1000.0];
    (void)[[clsApplication sharedApplication] nextEventMatchingMask:NSEventMaskAny
                                                        untilDate:until
                                                           inMode:NSDefaultRunLoopMode
                                                          dequeue:NO];
    [pool drain];
}

const eshkol_window_backend_ops_t eshkol_plat_window_cocoa_ops = {
    ESHKOL_BACKEND_COCOA,
    cocoa_available, cocoa_create, cocoa_destroy, cocoa_set_size, cocoa_set_position,
    cocoa_set_title, cocoa_present, cocoa_read_presented, cocoa_pump, cocoa_wait_os
};

#endif /* __APPLE__ */
