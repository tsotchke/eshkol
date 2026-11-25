/*
 * midi_coremidi.c — CoreMIDI backend (ADR-0053).
 *
 * CoreMIDI is bound at run time. Output ports are virtual sources published
 * to every MIDI client on the system (MIDISourceCreate + MIDIReceived); input
 * ports connect to the source whose display name matches, and CoreMIDI's
 * receive thread feeds the port's ring through the shared stream parser.
 *
 * The MIDI 1.0 packet-list entry points are used because they carry byte
 * streams unchanged on every macOS release that ships CoreMIDI.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#if defined(__APPLE__)

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

#include <CoreFoundation/CoreFoundation.h>
#include <CoreMIDI/CoreMIDI.h>
#include <stdlib.h>
#include <string.h>

static struct {
    __typeof__(&MIDIClientCreate) MIDIClientCreate;
    __typeof__(&MIDISourceCreate) MIDISourceCreate;
    __typeof__(&MIDIEndpointDispose) MIDIEndpointDispose;
    __typeof__(&MIDIInputPortCreate) MIDIInputPortCreate;
    __typeof__(&MIDIPortConnectSource) MIDIPortConnectSource;
    __typeof__(&MIDIPortDispose) MIDIPortDispose;
    __typeof__(&MIDIGetNumberOfSources) MIDIGetNumberOfSources;
    __typeof__(&MIDIGetSource) MIDIGetSource;
    __typeof__(&MIDIObjectGetStringProperty) MIDIObjectGetStringProperty;
    __typeof__(&MIDIReceived) MIDIReceived;
    __typeof__(&MIDIPacketListInit) MIDIPacketListInit;
    __typeof__(&MIDIPacketListAdd) MIDIPacketListAdd;
    const CFStringRef* kMIDIPropertyName;
} M;
static const char* const k_midi_symbols[] = {
    "MIDIClientCreate", "MIDISourceCreate", "MIDIEndpointDispose", "MIDIInputPortCreate",
    "MIDIPortConnectSource", "MIDIPortDispose", "MIDIGetNumberOfSources", "MIDIGetSource",
    "MIDIObjectGetStringProperty", "MIDIReceived", "MIDIPacketListInit", "MIDIPacketListAdd",
    "kMIDIPropertyName", NULL
};

static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_loaded = 0;
static MIDIClientRef g_client = 0;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

static void cm_load(void) {
    static const char* const libs[] = {
        "/System/Library/Frameworks/CoreMIDI.framework/CoreMIDI", NULL };
    g_loaded = eshkol_plat_dlsym_all(eshkol_plat_dlopen_any(libs), k_midi_symbols, (void**)&M);
}

static int cm_available(void) {
    pthread_once(&g_once, cm_load);
    return g_loaded;
}

static int ensure_client(void) {
    pthread_mutex_lock(&g_lock);
    if (!g_client &&
        M.MIDIClientCreate(CFSTR("eshkol"), NULL, NULL, &g_client) != noErr)
        g_client = 0;
    int ok = g_client != 0;
    pthread_mutex_unlock(&g_lock);
    return ok;
}

typedef struct {
    MIDIEndpointRef source;   /* output */
    MIDIPortRef port;         /* input */
} cm_port_t;

static int endpoint_name(MIDIEndpointRef ep, char* buf, uint32_t cap) {
    CFStringRef s = NULL;
    if (M.MIDIObjectGetStringProperty(ep, *M.kMIDIPropertyName, &s) != noErr || !s) return 0;
    Boolean ok = CFStringGetCString(s, buf, (CFIndex)cap, kCFStringEncodingUTF8);
    CFRelease(s);
    return ok ? 1 : 0;
}

static void read_proc(const MIDIPacketList* list, void* port_ref, void* src_ref) {
    (void)src_ref;
    eshkol_midi_port_t* p = (eshkol_midi_port_t*)port_ref;
    uint64_t ts = eshkol_platform_time_us();
    const MIDIPacket* pkt = &list->packet[0];
    for (UInt32 i = 0; i < list->numPackets; ++i) {
        eshkol_plat_midi_feed(p, pkt->data, pkt->length, ts);
        pkt = MIDIPacketNext(pkt);
    }
}

static int cm_open(eshkol_midi_port_t* p) {
    if (!ensure_client()) return ESHKOL_PLATFORM_EDEVICE;
    cm_port_t* c = (cm_port_t*)calloc(1, sizeof(*c));
    if (!c) return ESHKOL_PLATFORM_ENOMEM;
    CFStringRef name = CFStringCreateWithCString(NULL, p->name, kCFStringEncodingUTF8);
    if (!name) { free(c); return ESHKOL_PLATFORM_ENOMEM; }
    int st = ESHKOL_PLATFORM_OK;
    if (p->direction == ESHKOL_MIDI_OUTPUT) {
        if (M.MIDISourceCreate(g_client, name, &c->source) != noErr) st = ESHKOL_PLATFORM_EDEVICE;
    } else {
        MIDIEndpointRef found = 0, partial = 0;
        ItemCount n = M.MIDIGetNumberOfSources();
        char buf[256];
        for (ItemCount i = 0; i < n && !found; ++i) {
            MIDIEndpointRef ep = M.MIDIGetSource(i);
            if (!endpoint_name(ep, buf, sizeof(buf))) continue;
            if (strcmp(buf, p->name) == 0) found = ep;
            else if (!partial && strstr(buf, p->name)) partial = ep;
        }
        if (!found) found = partial;
        if (!found) {
            st = ESHKOL_PLATFORM_ENOTFOUND;
        } else if (M.MIDIInputPortCreate(g_client, name, read_proc, p, &c->port) != noErr ||
                   M.MIDIPortConnectSource(c->port, found, NULL) != noErr) {
            st = ESHKOL_PLATFORM_EDEVICE;
        }
    }
    CFRelease(name);
    if (st != ESHKOL_PLATFORM_OK) {
        if (c->port) M.MIDIPortDispose(c->port);
        free(c);
        return st;
    }
    p->impl = c;
    return ESHKOL_PLATFORM_OK;
}

static void cm_close(eshkol_midi_port_t* p) {
    cm_port_t* c = (cm_port_t*)p->impl;
    if (!c) return;
    /* MIDIPortDispose returns after in-flight read callbacks complete. */
    if (c->port) M.MIDIPortDispose(c->port);
    if (c->source) M.MIDIEndpointDispose(c->source);
    free(c);
    p->impl = NULL;
}

static int cm_send(eshkol_midi_port_t* p, const uint8_t* bytes, uint32_t len) {
    cm_port_t* c = (cm_port_t*)p->impl;
    Byte storage[ESHKOL_MIDI_MAX_MESSAGE + 256];
    MIDIPacketList* list = (MIDIPacketList*)storage;
    MIDIPacket* pkt = M.MIDIPacketListInit(list);
    pkt = M.MIDIPacketListAdd(list, sizeof(storage), pkt, 0, len, bytes);
    if (!pkt) return ESHKOL_PLATFORM_EFULL;
    return M.MIDIReceived(c->source, list) == noErr ? ESHKOL_PLATFORM_OK
                                                    : ESHKOL_PLATFORM_EDEVICE;
}

static int cm_source_count(void) {
    return (int)M.MIDIGetNumberOfSources();
}

static int cm_source_name(int index, char* buf, uint32_t cap) {
    if (index < 0 || (ItemCount)index >= M.MIDIGetNumberOfSources()) return ESHKOL_PLATFORM_ENOTFOUND;
    return endpoint_name(M.MIDIGetSource((ItemCount)index), buf, cap) ? ESHKOL_PLATFORM_OK
                                                                      : ESHKOL_PLATFORM_EDEVICE;
}

const eshkol_midi_backend_ops_t eshkol_plat_midi_coremidi_ops = {
    ESHKOL_BACKEND_COREMIDI, cm_available, cm_open, cm_close, cm_send,
    cm_source_count, cm_source_name
};

#pragma clang diagnostic pop

#endif /* __APPLE__ */
