/*
 * midi.c — backend-independent MIDI ports: message validation, the
 * running-status/SysEx stream parser, the lock-free inbound ring and the
 * Eshkol language bridge (ADR-0053).
 *
 * Inbound ring: one producer (the backend's delivery thread, or a sender
 * serialised by the loopback lock) and one consumer (the port owner). Records
 * are [u32 length][u64 timestamp][bytes, padded to 4]; a length of
 * 0xFFFFFFFF marks a wrap to the start of the ring.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "../platform_internal.h"

#include <stdlib.h>
#include <string.h>

#define RING_WRAP 0xFFFFFFFFu
#define RECORD_HEADER 12u

const eshkol_midi_backend_ops_t* eshkol_plat_midi_ops_for(int backend) {
    switch (backend) {
        case ESHKOL_BACKEND_LOOPBACK: return &eshkol_plat_midi_loopback_ops;
#if defined(__linux__)
        case ESHKOL_BACKEND_ALSA_SEQ: return &eshkol_plat_midi_alsa_ops;
#endif
#if defined(__APPLE__)
        case ESHKOL_BACKEND_COREMIDI: return &eshkol_plat_midi_coremidi_ops;
#endif
        default:                      return NULL;
    }
}

int eshkol_midi_message_length(uint8_t status) {
    if (status < 0x80) return 0;
    if (status < 0xC0) return 3;          /* note off/on, poly pressure, control change */
    if (status < 0xE0) return 2;          /* program change, channel pressure */
    if (status < 0xF0) return 3;          /* pitch bend */
    switch (status) {
        case 0xF0: return -1;             /* SysEx: variable, ends with F7 */
        case 0xF1: return 2;              /* MTC quarter frame */
        case 0xF2: return 3;              /* song position */
        case 0xF3: return 2;              /* song select */
        case 0xF6: return 1;              /* tune request */
        case 0xF8: case 0xFA: case 0xFB: case 0xFC: case 0xFE: case 0xFF:
            return 1;                     /* real-time */
        default:   return 0;              /* F4 F5 F7 F9 FD: undefined as a status */
    }
}

int eshkol_plat_midi_valid(const uint8_t* b, uint32_t len) {
    if (!b || len == 0 || len > ESHKOL_MIDI_MAX_MESSAGE) return 0;
    int expect = eshkol_midi_message_length(b[0]);
    if (expect == 0) return 0;
    if (expect < 0) {
        if (len < 2 || b[len - 1] != 0xF7) return 0;
        for (uint32_t i = 1; i + 1 < len; ++i) if (b[i] & 0x80) return 0;
        return 1;
    }
    if (len != (uint32_t)expect) return 0;
    for (uint32_t i = 1; i < len; ++i) if (b[i] & 0x80) return 0;
    return 1;
}

/* ── inbound ring ──────────────────────────────────────────────────── */

void eshkol_plat_midi_deliver(eshkol_midi_port_t* p, const uint8_t* msg, uint32_t len,
                              uint64_t ts) {
    if (!p->ring || len == 0 || len > ESHKOL_MIDI_MAX_MESSAGE) return;
    const uint64_t size = ESHKOL_PLAT_MIDI_RING_BYTES;
    uint32_t need = RECORD_HEADER + ((len + 3u) & ~3u);
    uint64_t w = atomic_load_explicit(&p->ring_write, memory_order_relaxed);
    uint64_t r = atomic_load_explicit(&p->ring_read, memory_order_acquire);
    uint32_t off = (uint32_t)(w % size);
    uint32_t tail = (uint32_t)(size - off);
    uint64_t total = need + (tail < need ? tail : 0u);
    if (size - (w - r) < total) {
        atomic_fetch_add_explicit(&p->dropped, 1u, memory_order_relaxed);
        return;
    }
    if (tail < need) {
        uint32_t wrap = RING_WRAP;
        memcpy(p->ring + off, &wrap, 4);   /* tail >= 4: records are 4-aligned */
        w += tail;
        off = 0;
    }
    memcpy(p->ring + off, &len, 4);
    memcpy(p->ring + off + 4, &ts, 8);
    memcpy(p->ring + off + RECORD_HEADER, msg, len);
    atomic_store_explicit(&p->ring_write, w + need, memory_order_release);
    atomic_fetch_add_explicit(&p->received, 1u, memory_order_relaxed);
}

void eshkol_plat_midi_feed(eshkol_midi_port_t* p, const uint8_t* bytes, uint32_t len,
                           uint64_t ts) {
    for (uint32_t i = 0; i < len; ++i) {
        uint8_t b = bytes[i];
        if (b >= 0xF8) {                         /* real-time: may interleave anything */
            if (eshkol_midi_message_length(b) == 1) eshkol_plat_midi_deliver(p, &b, 1, ts);
            continue;
        }
        if (b == 0xF0) {
            p->in_sysex = 1;
            p->parse_buf[0] = b;
            p->parse_len = 1;
            p->running_status = 0;
            continue;
        }
        if (p->in_sysex) {
            if (b == 0xF7) {
                if (p->parse_len < ESHKOL_MIDI_MAX_MESSAGE) {
                    p->parse_buf[p->parse_len++] = b;
                    eshkol_plat_midi_deliver(p, p->parse_buf, p->parse_len, ts);
                } else {
                    atomic_fetch_add_explicit(&p->malformed, 1u, memory_order_relaxed);
                }
                p->in_sysex = 0;
                p->parse_len = 0;
                continue;
            }
            if (b & 0x80) {                      /* status inside SysEx: abandon it */
                atomic_fetch_add_explicit(&p->malformed, 1u, memory_order_relaxed);
                p->in_sysex = 0;
                p->parse_len = 0;
            } else {
                if (p->parse_len < ESHKOL_MIDI_MAX_MESSAGE - 1) p->parse_buf[p->parse_len++] = b;
                else p->parse_len = ESHKOL_MIDI_MAX_MESSAGE;  /* overflow: dropped at F7 */
                continue;
            }
        }
        if (b & 0x80) {
            int expect = eshkol_midi_message_length(b);
            if (expect <= 0) {
                atomic_fetch_add_explicit(&p->malformed, 1u, memory_order_relaxed);
                p->running_status = 0;
                p->parse_len = 0;
                continue;
            }
            p->parse_buf[0] = b;
            p->parse_len = 1;
            p->parse_expected = (uint32_t)expect;
            p->running_status = b < 0xF0 ? b : 0;   /* system common clears running status */
        } else {
            if (p->parse_len == 0) {
                if (!p->running_status) {
                    atomic_fetch_add_explicit(&p->malformed, 1u, memory_order_relaxed);
                    continue;
                }
                p->parse_buf[0] = p->running_status;
                p->parse_len = 1;
                p->parse_expected = (uint32_t)eshkol_midi_message_length(p->running_status);
            }
            p->parse_buf[p->parse_len++] = b;
        }
        if (p->parse_len == p->parse_expected) {
            eshkol_plat_midi_deliver(p, p->parse_buf, p->parse_len, ts);
            p->parse_len = 0;
        }
    }
}

/* ── lifecycle ─────────────────────────────────────────────────────── */

int eshkol_midi_open_ex(int backend, int direction, const char* name,
                        eshkol_midi_port_t** out) {
    if (!out) return ESHKOL_PLATFORM_EINVAL;
    *out = NULL;
    if (!name || !*name || strlen(name) >= 128) return ESHKOL_PLATFORM_EINVAL;
    if (direction != ESHKOL_MIDI_OUTPUT && direction != ESHKOL_MIDI_INPUT)
        return ESHKOL_PLATFORM_EINVAL;
    const eshkol_midi_backend_ops_t* ops = eshkol_plat_midi_ops_for(backend);
    if (!ops || !ops->available()) return ESHKOL_PLATFORM_EUNAVAILABLE;
    eshkol_midi_port_t* p = (eshkol_midi_port_t*)calloc(1, sizeof(*p));
    if (!p) return ESHKOL_PLATFORM_ENOMEM;
    p->magic = ESHKOL_PLAT_KIND_MIDI;
    p->ops = ops;
    p->direction = direction;
    strcpy(p->name, name);
    atomic_init(&p->ring_write, 0);
    atomic_init(&p->ring_read, 0);
    if (direction == ESHKOL_MIDI_INPUT) {
        p->ring = (uint8_t*)calloc(1, ESHKOL_PLAT_MIDI_RING_BYTES);
        if (!p->ring) { free(p); return ESHKOL_PLATFORM_ENOMEM; }
    }
    int st = ops->open(p);
    if (st != ESHKOL_PLATFORM_OK) { free(p->ring); free(p); return st; }
    *out = p;
    return ESHKOL_PLATFORM_OK;
}

eshkol_midi_port_t* eshkol_midi_open_output(const char* name) {
    eshkol_midi_port_t* p = NULL;
    (void)eshkol_midi_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_MIDI), ESHKOL_MIDI_OUTPUT,
                              name, &p);
    return p;
}

eshkol_midi_port_t* eshkol_midi_open_input(const char* source_name) {
    eshkol_midi_port_t* p = NULL;
    (void)eshkol_midi_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_MIDI), ESHKOL_MIDI_INPUT,
                              source_name, &p);
    return p;
}

void eshkol_midi_close(eshkol_midi_port_t* p) {
    if (!p) return;
    p->ops->close(p);
    free(p->ring);
    p->magic = 0;
    free(p);
}

int eshkol_midi_send(eshkol_midi_port_t* p, const uint8_t* bytes, uint32_t len) {
    if (!p) return ESHKOL_PLATFORM_EINVAL;
    if (p->direction != ESHKOL_MIDI_OUTPUT) return ESHKOL_PLATFORM_ESTATE;
    if (!eshkol_plat_midi_valid(bytes, len)) return ESHKOL_PLATFORM_EFORMAT;
    int st = p->ops->send(p, bytes, len);
    if (st == ESHKOL_PLATFORM_OK) atomic_fetch_add_explicit(&p->sent, 1u, memory_order_relaxed);
    return st;
}

int eshkol_midi_receive(eshkol_midi_port_t* p, uint8_t* buf, uint32_t cap, uint64_t* ts) {
    if (!p || (!buf && cap)) return ESHKOL_PLATFORM_EINVAL;
    if (p->direction != ESHKOL_MIDI_INPUT) return ESHKOL_PLATFORM_ESTATE;
    const uint64_t size = ESHKOL_PLAT_MIDI_RING_BYTES;
    for (;;) {
        uint64_t r = atomic_load_explicit(&p->ring_read, memory_order_relaxed);
        uint64_t w = atomic_load_explicit(&p->ring_write, memory_order_acquire);
        if (r == w) return 0;
        uint32_t off = (uint32_t)(r % size);
        uint32_t len;
        memcpy(&len, p->ring + off, 4);
        if (len == RING_WRAP) {
            atomic_store_explicit(&p->ring_read, r + (size - off), memory_order_release);
            continue;
        }
        if (len > cap) return ESHKOL_PLATFORM_EFULL;
        if (ts) memcpy(ts, p->ring + off + 4, 8);
        memcpy(buf, p->ring + off + RECORD_HEADER, len);
        atomic_store_explicit(&p->ring_read, r + RECORD_HEADER + ((len + 3u) & ~3u),
                              memory_order_release);
        return (int)len;
    }
}

int eshkol_midi_receive_timeout(eshkol_midi_port_t* p, uint8_t* buf, uint32_t cap,
                                uint64_t* ts, int timeout_ms) {
    uint64_t deadline = timeout_ms < 0 ? UINT64_MAX
                                       : eshkol_plat_now_ns() + (uint64_t)timeout_ms * 1000000ull;
    for (;;) {
        int n = eshkol_midi_receive(p, buf, cap, ts);
        if (n != 0) return n;
        if (eshkol_plat_now_ns() >= deadline) return 0;
        eshkol_plat_sleep_ms(1);   /* the producer side never blocks or signals */
    }
}

int eshkol_midi_get_stats(eshkol_midi_port_t* p, eshkol_midi_stats_t* s) {
    if (!p || !s) return ESHKOL_PLATFORM_EINVAL;
    s->sent = atomic_load(&p->sent);
    s->received = atomic_load(&p->received);
    s->dropped = atomic_load(&p->dropped);
    s->malformed = atomic_load(&p->malformed);
    return ESHKOL_PLATFORM_OK;
}

int eshkol_midi_backend_of(eshkol_midi_port_t* p) {
    return p ? p->ops->backend : ESHKOL_BACKEND_NONE;
}

int eshkol_midi_source_count(void) {
    const eshkol_midi_backend_ops_t* ops =
        eshkol_plat_midi_ops_for(eshkol_plat_selected(ESHKOL_PLATFORM_MIDI));
    if (!ops || !ops->available()) return ESHKOL_PLATFORM_EUNAVAILABLE;
    return ops->source_count();
}

int eshkol_midi_source_name(int index, char* buf, uint32_t cap) {
    if (!buf || cap == 0) return ESHKOL_PLATFORM_EINVAL;
    const eshkol_midi_backend_ops_t* ops =
        eshkol_plat_midi_ops_for(eshkol_plat_selected(ESHKOL_PLATFORM_MIDI));
    if (!ops || !ops->available()) return ESHKOL_PLATFORM_EUNAVAILABLE;
    return ops->source_name(index, buf, cap);
}

/* ── Eshkol language bridge ────────────────────────────────────────── */

#define WITH_PORT(handle)                                                     \
    uint64_t lease_ = 0; int st_ = 0;                                         \
    eshkol_midi_port_t* p = (eshkol_midi_port_t*)eshkol_plat_handle_borrow(   \
        (handle), ESHKOL_PLAT_KIND_MIDI, &lease_, &st_);                      \
    if (!p) return st_;
#define END_PORT(handle) eshkol_plat_handle_release((handle), lease_)

int64_t eshkol_plat_midi_open(int32_t direction, const char* name) {
    eshkol_midi_port_t* p = NULL;
    int st = eshkol_midi_open_ex(eshkol_plat_selected(ESHKOL_PLATFORM_MIDI), direction,
                                 name, &p);
    if (st != ESHKOL_PLATFORM_OK) return st;
    p->current_len = 0;
    int64_t id = eshkol_plat_handle_adopt(ESHKOL_PLAT_KIND_MIDI, p);
    if (id < 0) eshkol_midi_close(p);
    return id;
}

int32_t eshkol_plat_midi_close(int64_t handle) {
    return eshkol_plat_handle_close(handle);
}

int32_t eshkol_plat_midi_stage(int64_t handle, int32_t byte) {
    WITH_PORT(handle)
    int r = ESHKOL_PLATFORM_OK;
    if (byte < 0 || byte > 255) r = ESHKOL_PLATFORM_EINVAL;
    else if (p->stage_len >= ESHKOL_MIDI_MAX_MESSAGE) r = ESHKOL_PLATFORM_EFULL;
    else p->stage[p->stage_len++] = (uint8_t)byte;
    END_PORT(handle);
    return r;
}

int32_t eshkol_plat_midi_send_staged(int64_t handle) {
    WITH_PORT(handle)
    int r = eshkol_midi_send(p, p->stage, p->stage_len);
    p->stage_len = 0;
    END_PORT(handle);
    return r;
}

int32_t eshkol_plat_midi_receive(int64_t handle, int32_t timeout_ms) {
    WITH_PORT(handle)
    uint64_t ts = 0;
    int n = timeout_ms == 0
                ? eshkol_midi_receive(p, p->current, sizeof(p->current), &ts)
                : eshkol_midi_receive_timeout(p, p->current, sizeof(p->current), &ts, timeout_ms);
    p->current_len = n > 0 ? n : 0;
    p->current_ts = n > 0 ? ts : 0;
    END_PORT(handle);
    return n;
}

int32_t eshkol_plat_midi_byte(int64_t handle, int32_t index) {
    WITH_PORT(handle)
    int32_t r = (index >= 0 && index < p->current_len) ? p->current[index] : -1;
    END_PORT(handle);
    return r;
}

int64_t eshkol_plat_midi_timestamp(int64_t handle) {
    WITH_PORT(handle)
    int64_t r = (int64_t)p->current_ts;
    END_PORT(handle);
    return r;
}

int64_t eshkol_plat_midi_stat(int64_t handle, int32_t field) {
    WITH_PORT(handle)
    eshkol_midi_stats_t s;
    eshkol_midi_get_stats(p, &s);
    int64_t r = field == 0 ? (int64_t)s.sent : field == 1 ? (int64_t)s.received
              : field == 2 ? (int64_t)s.dropped : field == 3 ? (int64_t)s.malformed
              : field == 4 ? p->ops->backend : ESHKOL_PLATFORM_EINVAL;
    END_PORT(handle);
    return r;
}

int32_t eshkol_plat_midi_source_count(void) {
    return eshkol_midi_source_count();
}

const char* eshkol_plat_midi_source_name(int32_t index) {
    static _Thread_local char name[256];
    if (eshkol_midi_source_name(index, name, sizeof(name)) != ESHKOL_PLATFORM_OK) name[0] = 0;
    return name;
}
