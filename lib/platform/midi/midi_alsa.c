/*
 * midi_alsa.c — ALSA sequencer MIDI backend (ADR-0053).
 *
 * libasound.so.2 is bound at run time; the sequencer entry points are
 * declared here from alsa-lib's stable C ABI. Two sequencer clients are used:
 * "eshkol" (output ports, written by their owner threads) and "eshkol-in"
 * (input ports, drained by one reader thread). Each output port is a
 * subscribable source; an input port subscribes to the source whose port (or
 * client) name matches. Messages are converted with alsa-lib's MIDI event
 * coder in both directions.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#if !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "../platform_internal.h"

#if defined(__linux__)

#include <errno.h>
#include <poll.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* alsa-lib sequencer ABI (seq.h, seq_event.h, seq_midi_event.h). */
typedef struct _snd_seq snd_seq_t;
typedef struct _snd_midi_event snd_midi_event_t;
typedef struct _snd_seq_client_info snd_seq_client_info_t;
typedef struct _snd_seq_port_info snd_seq_port_info_t;
typedef struct {
    unsigned char type, flags, tag, queue;
    uint32_t time[2];
    unsigned char source_client, source_port;
    unsigned char dest_client, dest_port;
    uint32_t data[3];
} eshkol_snd_seq_event_t;

#if defined(__has_include)
#  if __has_include(<sound/asequencer.h>)
#    include <sound/asequencer.h>
_Static_assert(sizeof(struct snd_seq_event) == sizeof(eshkol_snd_seq_event_t),
               "sequencer event layout must match the kernel ABI");
_Static_assert(offsetof(struct snd_seq_event, dest) ==
               offsetof(eshkol_snd_seq_event_t, dest_client),
               "sequencer event address offset must match the kernel ABI");
#  endif
#endif

enum { SND_SEQ_OPEN_OUTPUT = 1, SND_SEQ_OPEN_INPUT = 2, SND_SEQ_NONBLOCK = 1 };
enum {
    SEQ_CAP_READ = 1u << 0, SEQ_CAP_WRITE = 1u << 1,
    SEQ_CAP_SUBS_READ = 1u << 5, SEQ_CAP_SUBS_WRITE = 1u << 6
};
enum { SEQ_TYPE_MIDI_GENERIC = 1u << 1, SEQ_TYPE_APPLICATION = 1u << 20 };
enum { SEQ_ADDRESS_UNKNOWN = 253, SEQ_ADDRESS_SUBSCRIBERS = 254, SEQ_QUEUE_DIRECT = 253 };

static struct {
    int (*snd_seq_open)(snd_seq_t**, const char*, int, int);
    int (*snd_seq_close)(snd_seq_t*);
    int (*snd_seq_set_client_name)(snd_seq_t*, const char*);
    int (*snd_seq_client_id)(snd_seq_t*);
    int (*snd_seq_create_simple_port)(snd_seq_t*, const char*, unsigned int, unsigned int);
    int (*snd_seq_delete_simple_port)(snd_seq_t*, int);
    int (*snd_seq_connect_from)(snd_seq_t*, int, int, int);
    int (*snd_seq_event_output_direct)(snd_seq_t*, eshkol_snd_seq_event_t*);
    int (*snd_seq_event_input)(snd_seq_t*, eshkol_snd_seq_event_t**);
    int (*snd_seq_poll_descriptors_count)(snd_seq_t*, short);
    int (*snd_seq_poll_descriptors)(snd_seq_t*, struct pollfd*, unsigned int, short);
    int (*snd_seq_client_info_malloc)(snd_seq_client_info_t**);
    void (*snd_seq_client_info_free)(snd_seq_client_info_t*);
    void (*snd_seq_client_info_set_client)(snd_seq_client_info_t*, int);
    int (*snd_seq_client_info_get_client)(const snd_seq_client_info_t*);
    const char* (*snd_seq_client_info_get_name)(snd_seq_client_info_t*);
    int (*snd_seq_query_next_client)(snd_seq_t*, snd_seq_client_info_t*);
    int (*snd_seq_port_info_malloc)(snd_seq_port_info_t**);
    void (*snd_seq_port_info_free)(snd_seq_port_info_t*);
    void (*snd_seq_port_info_set_client)(snd_seq_port_info_t*, int);
    void (*snd_seq_port_info_set_port)(snd_seq_port_info_t*, int);
    int (*snd_seq_port_info_get_port)(const snd_seq_port_info_t*);
    unsigned int (*snd_seq_port_info_get_capability)(const snd_seq_port_info_t*);
    const char* (*snd_seq_port_info_get_name)(const snd_seq_port_info_t*);
    int (*snd_seq_query_next_port)(snd_seq_t*, snd_seq_port_info_t*);
    int (*snd_midi_event_new)(size_t, snd_midi_event_t**);
    void (*snd_midi_event_free)(snd_midi_event_t*);
    long (*snd_midi_event_encode)(snd_midi_event_t*, const unsigned char*, long,
                                  eshkol_snd_seq_event_t*);
    long (*snd_midi_event_decode)(snd_midi_event_t*, unsigned char*, long,
                                  const eshkol_snd_seq_event_t*);
    void (*snd_midi_event_reset_encode)(snd_midi_event_t*);
    void (*snd_midi_event_reset_decode)(snd_midi_event_t*);
    void (*snd_midi_event_no_status)(snd_midi_event_t*, int);
} S;

static const char* const k_seq_symbols[] = {
    "snd_seq_open", "snd_seq_close", "snd_seq_set_client_name", "snd_seq_client_id",
    "snd_seq_create_simple_port", "snd_seq_delete_simple_port", "snd_seq_connect_from",
    "snd_seq_event_output_direct", "snd_seq_event_input", "snd_seq_poll_descriptors_count",
    "snd_seq_poll_descriptors", "snd_seq_client_info_malloc", "snd_seq_client_info_free",
    "snd_seq_client_info_set_client", "snd_seq_client_info_get_client",
    "snd_seq_client_info_get_name", "snd_seq_query_next_client", "snd_seq_port_info_malloc",
    "snd_seq_port_info_free", "snd_seq_port_info_set_client", "snd_seq_port_info_set_port",
    "snd_seq_port_info_get_port", "snd_seq_port_info_get_capability",
    "snd_seq_port_info_get_name", "snd_seq_query_next_port", "snd_midi_event_new",
    "snd_midi_event_free", "snd_midi_event_encode", "snd_midi_event_decode",
    "snd_midi_event_reset_encode", "snd_midi_event_reset_decode", "snd_midi_event_no_status",
    NULL
};

#define MAX_IN_PORTS 64

static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_once_t g_once = PTHREAD_ONCE_INIT;
static int g_loaded = 0;
static snd_seq_t* g_out = NULL;       /* output client */
static snd_seq_t* g_in = NULL;        /* input client */
static pthread_t g_reader;
static atomic_int g_reader_run;
static int g_reader_started = 0;
static snd_midi_event_t* g_decoder = NULL;
static struct { int port; eshkol_midi_port_t* p; } g_inputs[MAX_IN_PORTS];

typedef struct {
    int seq_port;
    snd_midi_event_t* encoder;
} seq_port_t;

static void seq_load(void) {
    static const char* const libs[] = { "libasound.so.2", "libasound.so", NULL };
    g_loaded = eshkol_plat_dlsym_all(eshkol_plat_dlopen_any(libs), k_seq_symbols, (void**)&S);
}

/* Available when the library loads and the sequencer device opens. */
static int seq_available(void) {
    pthread_once(&g_once, seq_load);
    if (!g_loaded) return 0;
    /* Probe permissions first so an inaccessible sequencer is reported as
     * unavailable without alsa-lib printing an open failure. */
    if (access("/dev/snd/seq", R_OK | W_OK) != 0) return 0;
    pthread_mutex_lock(&g_lock);
    if (!g_out) {
        if (S.snd_seq_open(&g_out, "default", SND_SEQ_OPEN_OUTPUT, 0) == 0)
            S.snd_seq_set_client_name(g_out, "eshkol");
        else
            g_out = NULL;
    }
    int ok = g_out != NULL;
    pthread_mutex_unlock(&g_lock);
    return ok;
}

static void* reader_thread(void* arg) {
    (void)arg;
    struct pollfd pfd[4];
    int n = S.snd_seq_poll_descriptors_count(g_in, POLLIN);
    if (n > 4) n = 4;
    S.snd_seq_poll_descriptors(g_in, pfd, (unsigned)n, POLLIN);
    unsigned char bytes[ESHKOL_MIDI_MAX_MESSAGE];
    while (atomic_load(&g_reader_run)) {
        if (poll(pfd, (nfds_t)n, 50) <= 0) continue;
        pthread_mutex_lock(&g_lock);
        for (;;) {
            eshkol_snd_seq_event_t* ev = NULL;
            int rc = S.snd_seq_event_input(g_in, &ev);
            if (rc < 0 || !ev) break;
            S.snd_midi_event_reset_decode(g_decoder);
            long len = S.snd_midi_event_decode(g_decoder, bytes, sizeof(bytes), ev);
            if (len > 0) {
                uint64_t ts = eshkol_platform_time_us();
                for (int i = 0; i < MAX_IN_PORTS; ++i)
                    if (g_inputs[i].p && g_inputs[i].port == ev->dest_port)
                        eshkol_plat_midi_feed(g_inputs[i].p, bytes, (uint32_t)len, ts);
            }
            if (rc == 0) break;   /* no more events buffered */
        }
        pthread_mutex_unlock(&g_lock);
    }
    return NULL;
}

/* Locate a readable, subscribable port by name; returns 0 on success. */
static int find_source(const char* name, int* client, int* port) {
    snd_seq_client_info_t* ci = NULL;
    snd_seq_port_info_t* pi = NULL;
    if (S.snd_seq_client_info_malloc(&ci) < 0) return -1;
    if (S.snd_seq_port_info_malloc(&pi) < 0) { S.snd_seq_client_info_free(ci); return -1; }
    int found = -1, partial_c = -1, partial_p = -1;
    S.snd_seq_client_info_set_client(ci, -1);
    while (found < 0 && S.snd_seq_query_next_client(g_out, ci) >= 0) {
        int c = S.snd_seq_client_info_get_client(ci);
        const char* cname = S.snd_seq_client_info_get_name(ci);
        S.snd_seq_port_info_set_client(pi, c);
        S.snd_seq_port_info_set_port(pi, -1);
        while (S.snd_seq_query_next_port(g_out, pi) >= 0) {
            unsigned caps = S.snd_seq_port_info_get_capability(pi);
            if ((caps & (SEQ_CAP_READ | SEQ_CAP_SUBS_READ)) != (SEQ_CAP_READ | SEQ_CAP_SUBS_READ))
                continue;
            const char* pname = S.snd_seq_port_info_get_name(pi);
            if (pname && strcmp(pname, name) == 0) {
                *client = c; *port = S.snd_seq_port_info_get_port(pi); found = 0; break;
            }
            if (partial_c < 0 && ((pname && strstr(pname, name)) || (cname && strstr(cname, name)))) {
                partial_c = c; partial_p = S.snd_seq_port_info_get_port(pi);
            }
        }
    }
    if (found < 0 && partial_c >= 0) { *client = partial_c; *port = partial_p; found = 0; }
    S.snd_seq_port_info_free(pi);
    S.snd_seq_client_info_free(ci);
    return found;
}

static int seq_open(eshkol_midi_port_t* p) {
    seq_port_t* sp = (seq_port_t*)calloc(1, sizeof(*sp));
    if (!sp) return ESHKOL_PLATFORM_ENOMEM;
    int st = ESHKOL_PLATFORM_OK;
    pthread_mutex_lock(&g_lock);
    if (p->direction == ESHKOL_MIDI_OUTPUT) {
        sp->seq_port = S.snd_seq_create_simple_port(g_out, p->name,
            SEQ_CAP_READ | SEQ_CAP_SUBS_READ, SEQ_TYPE_MIDI_GENERIC | SEQ_TYPE_APPLICATION);
        if (sp->seq_port < 0 || S.snd_midi_event_new(ESHKOL_MIDI_MAX_MESSAGE, &sp->encoder) < 0)
            st = ESHKOL_PLATFORM_EDEVICE;
    } else {
        int c = 0, port = 0, slot = -1;
        if (find_source(p->name, &c, &port) != 0) {
            st = ESHKOL_PLATFORM_ENOTFOUND;
        } else {
            for (int i = 0; i < MAX_IN_PORTS; ++i) if (!g_inputs[i].p) { slot = i; break; }
            if (slot < 0) st = ESHKOL_PLATFORM_EFULL;
        }
        if (st == ESHKOL_PLATFORM_OK && !g_in) {
            if (S.snd_seq_open(&g_in, "default", SND_SEQ_OPEN_INPUT, SND_SEQ_NONBLOCK) != 0 ||
                S.snd_midi_event_new(ESHKOL_MIDI_MAX_MESSAGE, &g_decoder) < 0) {
                g_in = NULL;
                st = ESHKOL_PLATFORM_EDEVICE;
            } else {
                S.snd_seq_set_client_name(g_in, "eshkol-in");
                S.snd_midi_event_no_status(g_decoder, 1);
            }
        }
        if (st == ESHKOL_PLATFORM_OK) {
            sp->seq_port = S.snd_seq_create_simple_port(g_in, p->name,
                SEQ_CAP_WRITE | SEQ_CAP_SUBS_WRITE, SEQ_TYPE_MIDI_GENERIC | SEQ_TYPE_APPLICATION);
            if (sp->seq_port < 0 || S.snd_seq_connect_from(g_in, sp->seq_port, c, port) < 0) {
                st = ESHKOL_PLATFORM_EDEVICE;
            } else {
                g_inputs[slot].port = sp->seq_port;
                g_inputs[slot].p = p;
                if (!g_reader_started) {
                    atomic_store(&g_reader_run, 1);
                    if (pthread_create(&g_reader, NULL, reader_thread, NULL) == 0)
                        g_reader_started = 1;
                    else
                        st = ESHKOL_PLATFORM_EDEVICE;
                }
            }
        }
    }
    pthread_mutex_unlock(&g_lock);
    if (st != ESHKOL_PLATFORM_OK) {
        if (sp->encoder) S.snd_midi_event_free(sp->encoder);
        free(sp);
        return st;
    }
    p->impl = sp;
    return ESHKOL_PLATFORM_OK;
}

static void seq_close(eshkol_midi_port_t* p) {
    seq_port_t* sp = (seq_port_t*)p->impl;
    if (!sp) return;
    pthread_mutex_lock(&g_lock);
    if (p->direction == ESHKOL_MIDI_OUTPUT) {
        S.snd_seq_delete_simple_port(g_out, sp->seq_port);
        S.snd_midi_event_free(sp->encoder);
    } else {
        for (int i = 0; i < MAX_IN_PORTS; ++i)
            if (g_inputs[i].p == p) g_inputs[i].p = NULL;
        S.snd_seq_delete_simple_port(g_in, sp->seq_port);
    }
    pthread_mutex_unlock(&g_lock);
    free(sp);
    p->impl = NULL;
}

static int seq_send(eshkol_midi_port_t* p, const uint8_t* bytes, uint32_t len) {
    seq_port_t* sp = (seq_port_t*)p->impl;
    eshkol_snd_seq_event_t ev;
    memset(&ev, 0, sizeof(ev));
    S.snd_midi_event_reset_encode(sp->encoder);
    long used = S.snd_midi_event_encode(sp->encoder, bytes, (long)len, &ev);
    if (used != (long)len) return ESHKOL_PLATFORM_EFORMAT;
    ev.source_port = (unsigned char)sp->seq_port;
    ev.dest_client = SEQ_ADDRESS_SUBSCRIBERS;
    ev.dest_port = SEQ_ADDRESS_UNKNOWN;
    ev.queue = SEQ_QUEUE_DIRECT;
    pthread_mutex_lock(&g_lock);
    int rc = S.snd_seq_event_output_direct(g_out, &ev);
    pthread_mutex_unlock(&g_lock);
    return rc < 0 ? ESHKOL_PLATFORM_EDEVICE : ESHKOL_PLATFORM_OK;
}

typedef struct { int index, seen; char* buf; uint32_t cap; } name_query_t;

static int seq_enumerate(name_query_t* q) {
    snd_seq_client_info_t* ci = NULL;
    snd_seq_port_info_t* pi = NULL;
    if (S.snd_seq_client_info_malloc(&ci) < 0) return ESHKOL_PLATFORM_ENOMEM;
    if (S.snd_seq_port_info_malloc(&pi) < 0) { S.snd_seq_client_info_free(ci); return ESHKOL_PLATFORM_ENOMEM; }
    int st = ESHKOL_PLATFORM_ENOTFOUND;
    pthread_mutex_lock(&g_lock);
    S.snd_seq_client_info_set_client(ci, -1);
    while (S.snd_seq_query_next_client(g_out, ci) >= 0) {
        S.snd_seq_port_info_set_client(pi, S.snd_seq_client_info_get_client(ci));
        S.snd_seq_port_info_set_port(pi, -1);
        while (S.snd_seq_query_next_port(g_out, pi) >= 0) {
            unsigned caps = S.snd_seq_port_info_get_capability(pi);
            if ((caps & (SEQ_CAP_READ | SEQ_CAP_SUBS_READ)) != (SEQ_CAP_READ | SEQ_CAP_SUBS_READ))
                continue;
            if (q->buf && q->seen == q->index) {
                const char* nm = S.snd_seq_port_info_get_name(pi);
                strncpy(q->buf, nm ? nm : "", q->cap - 1);
                q->buf[q->cap - 1] = 0;
                st = ESHKOL_PLATFORM_OK;
            }
            q->seen++;
        }
    }
    pthread_mutex_unlock(&g_lock);
    S.snd_seq_port_info_free(pi);
    S.snd_seq_client_info_free(ci);
    return q->buf ? st : ESHKOL_PLATFORM_OK;
}

static int seq_source_count(void) {
    name_query_t q = { -1, 0, NULL, 0 };
    int st = seq_enumerate(&q);
    return st == ESHKOL_PLATFORM_OK ? q.seen : st;
}

static int seq_source_name(int index, char* buf, uint32_t cap) {
    name_query_t q = { index, 0, buf, cap };
    return seq_enumerate(&q);
}

const eshkol_midi_backend_ops_t eshkol_plat_midi_alsa_ops = {
    ESHKOL_BACKEND_ALSA_SEQ, seq_available, seq_open, seq_close, seq_send,
    seq_source_count, seq_source_name
};

#endif /* __linux__ */
