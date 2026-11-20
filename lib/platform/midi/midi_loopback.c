/*
 * midi_loopback.c — in-process MIDI backend (ADR-0053).
 *
 * Output ports publish named sources; input ports subscribe to a source by
 * name. A send delivers the message to every subscriber's ring under one
 * process-wide lock, which keeps each ring single-producer. The headless
 * suites use this backend to exercise the same validation, ring and bridge
 * code as the OS backends.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#include <stdlib.h>
#include <string.h>

#define MAX_PORTS 256

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static eshkol_midi_port_t* g_outputs[MAX_PORTS];
static eshkol_midi_port_t* g_inputs[MAX_PORTS];

/* impl of an input port = the output port it is subscribed to (or NULL). */

static int lb_available(void) { return 1; }

static eshkol_midi_port_t* find_source(const char* name) {
    for (int i = 0; i < MAX_PORTS; ++i)
        if (g_outputs[i] && strcmp(g_outputs[i]->name, name) == 0) return g_outputs[i];
    for (int i = 0; i < MAX_PORTS; ++i)
        if (g_outputs[i] && strstr(g_outputs[i]->name, name)) return g_outputs[i];
    return NULL;
}

static int lb_open(eshkol_midi_port_t* p) {
    pthread_mutex_lock(&g_lock);
    int st = ESHKOL_PLATFORM_EFULL;
    eshkol_midi_port_t** table = p->direction == ESHKOL_MIDI_OUTPUT ? g_outputs : g_inputs;
    if (p->direction == ESHKOL_MIDI_INPUT) {
        eshkol_midi_port_t* src = find_source(p->name);
        if (!src) { pthread_mutex_unlock(&g_lock); return ESHKOL_PLATFORM_ENOTFOUND; }
        p->impl = src;
    }
    for (int i = 0; i < MAX_PORTS; ++i) {
        if (!table[i]) { table[i] = p; st = ESHKOL_PLATFORM_OK; break; }
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

static void lb_close(eshkol_midi_port_t* p) {
    pthread_mutex_lock(&g_lock);
    eshkol_midi_port_t** table = p->direction == ESHKOL_MIDI_OUTPUT ? g_outputs : g_inputs;
    for (int i = 0; i < MAX_PORTS; ++i) if (table[i] == p) table[i] = NULL;
    if (p->direction == ESHKOL_MIDI_OUTPUT) {
        for (int i = 0; i < MAX_PORTS; ++i)
            if (g_inputs[i] && g_inputs[i]->impl == p) g_inputs[i]->impl = NULL;
    }
    p->impl = NULL;
    pthread_mutex_unlock(&g_lock);
}

static int lb_send(eshkol_midi_port_t* p, const uint8_t* bytes, uint32_t len) {
    uint64_t ts = eshkol_platform_time_us();
    pthread_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_PORTS; ++i)
        if (g_inputs[i] && g_inputs[i]->impl == p)
            eshkol_plat_midi_feed(g_inputs[i], bytes, len, ts);
    pthread_mutex_unlock(&g_lock);
    return ESHKOL_PLATFORM_OK;
}

static int lb_source_count(void) {
    pthread_mutex_lock(&g_lock);
    int n = 0;
    for (int i = 0; i < MAX_PORTS; ++i) if (g_outputs[i]) ++n;
    pthread_mutex_unlock(&g_lock);
    return n;
}

static int lb_source_name(int index, char* buf, uint32_t cap) {
    pthread_mutex_lock(&g_lock);
    int n = 0, st = ESHKOL_PLATFORM_ENOTFOUND;
    for (int i = 0; i < MAX_PORTS; ++i) {
        if (!g_outputs[i]) continue;
        if (n++ == index) {
            strncpy(buf, g_outputs[i]->name, cap - 1);
            buf[cap - 1] = 0;
            st = ESHKOL_PLATFORM_OK;
            break;
        }
    }
    pthread_mutex_unlock(&g_lock);
    return st;
}

const eshkol_midi_backend_ops_t eshkol_plat_midi_loopback_ops = {
    ESHKOL_BACKEND_LOOPBACK, lb_available, lb_open, lb_close, lb_send,
    lb_source_count, lb_source_name
};
