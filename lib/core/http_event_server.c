/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * http_event_server.c — Non-blocking, event-loop-backed concurrent HTTP/1.1
 * and RFC 6455 WebSocket server built directly on eshkol_event_loop_t.
 */

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#elif !defined(_WIN32)
#define _GNU_SOURCE
#endif

#include "eshkol/core/http_event_server.h"
#include "eshkol/core/event_loop.h"
#include "eshkol/core/http_syntax.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
typedef SOCKET esh_sock_t;
#define ESH_BAD_SOCK INVALID_SOCKET
static int esh_sock_init(void) {
    static int done = 0;
    if (!done) {
        WSADATA d;
        if (WSAStartup(MAKEWORD(2, 2), &d) == 0) done = 1;
    }
    return done;
}
static int esh_sock_close(esh_sock_t s) { return closesocket(s); }
static int esh_sock_nonblock(esh_sock_t s) {
    u_long on = 1;
    return ioctlsocket(s, FIONBIO, &on) == 0 ? 0 : -1;
}
static int esh_sock_wouldblock(void) {
    int e = WSAGetLastError();
    return e == WSAEWOULDBLOCK || e == WSAEINTR;
}
static uint64_t esh_now_ms(void) { return (uint64_t)GetTickCount64(); }
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>
typedef int esh_sock_t;
#define ESH_BAD_SOCK (-1)
static int esh_sock_init(void) { return 1; }
static int esh_sock_close(esh_sock_t s) { return close(s); }
static int esh_sock_nonblock(esh_sock_t s) {
    int fl = fcntl(s, F_GETFL, 0);
    if (fl < 0 || fcntl(s, F_SETFL, fl | O_NONBLOCK) < 0) return -1;
    (void)fcntl(s, F_SETFD, FD_CLOEXEC);
    return 0;
}
static int esh_sock_wouldblock(void) {
    return errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR;
}
static uint64_t esh_now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t)ts.tv_sec * 1000u + (uint64_t)ts.tv_nsec / 1000000u;
}
#endif

#define ESH_LISTEN_COOKIE ((uint64_t)0u)
#define ESH_MAX_CONNS_CAP 65000
#define ESH_DEFAULT_MAX_CONNS 16384
#define ESH_DEFAULT_MAX_HEADER (32u * 1024u)
#define ESH_DEFAULT_MAX_BODY (1024u * 1024u)

typedef enum {
    CONN_FREE = 0,
    CONN_HTTP = 1,
    CONN_WS   = 2
} conn_state_t;

typedef struct {
    esh_sock_t     fd;
    conn_state_t   state;
    uint32_t       generation;
    int            armed_events;
    int            close_after_write;
    int            free_next;
    uint64_t       last_active_ms;

    unsigned char* rbuf;
    size_t         rlen;
    size_t         rcap;

    unsigned char* wbuf;
    size_t         wpos;
    size_t         wlen;
    size_t         wcap;
} http_conn_t;

struct eshkol_http_event_server {
    eshkol_event_loop_t*           loop;
    esh_sock_t                     listen_fd;
    uint16_t                       port;
    int32_t                        max_conns;
    size_t                         max_header_bytes;
    size_t                         max_body_bytes;
    int32_t                        idle_timeout_ms;
    int                            allow_websocket;
    uint64_t                       last_sweep_ms;

    http_conn_t*                   conns;
    int                            free_head;

    eshkol_http_event_request_cb_t on_request;
    eshkol_http_event_ws_frame_cb_t on_ws_frame;
    eshkol_http_event_close_cb_t   on_close;
    void*                          user_data;

    eshkol_http_event_stats_t      stats;
};

static uint64_t make_conn_id(uint32_t slot, uint32_t generation) {
    return ((uint64_t)generation << 20) | (uint64_t)(slot + 1u);
}

static http_conn_t* resolve_conn(eshkol_http_event_server_t* srv, uint64_t conn_id, int* out_slot) {
    if (!srv || conn_id == 0) return NULL;
    uint32_t encoded = (uint32_t)(conn_id & 0xFFFFFu);
    uint32_t gen = (uint32_t)(conn_id >> 20);
    if (encoded == 0 || (int32_t)encoded > srv->max_conns || gen == 0) return NULL;
    int slot = (int)(encoded - 1u);
    http_conn_t* c = &srv->conns[slot];
    if (c->state == CONN_FREE || c->generation != gen) return NULL;
    if (out_slot) *out_slot = slot;
    return c;
}

/* ── RFC 6455 SHA-1 + Base64 + UTF-8 helpers (self-contained, zero deps) ── */

static uint32_t es_rol32(uint32_t v, unsigned n) { return (v << n) | (v >> (32u - n)); }

static void es_sha1(const unsigned char* data, size_t len, unsigned char out[20]) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    size_t padded = ((len + 9u + 63u) / 64u) * 64u;
    unsigned char blk[64];
    for (size_t base = 0; base < padded; base += 64u) {
        memset(blk, 0, sizeof(blk));
        for (size_t i = 0; i < 64u && base + i < len; ++i) blk[i] = data[base + i];
        if (len >= base && len < base + 64u) blk[len - base] = 0x80u;
        if (base + 64u == padded) {
            uint64_t bits = (uint64_t)len * 8u;
            for (int i = 0; i < 8; ++i) blk[56 + i] = (unsigned char)(bits >> (56 - 8 * i));
        }
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)blk[i*4] << 24) | ((uint32_t)blk[i*4+1] << 16) |
                   ((uint32_t)blk[i*4+2] << 8) | blk[i*4+3];
        for (int i = 16; i < 80; ++i)
            w[i] = es_rol32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20)      { f = (b & c) | ((~b) & d); k = 0x5A827999u; }
            else if (i < 40) { f = b ^ c ^ d;            k = 0x6ED9EBA1u; }
            else if (i < 60) { f = (b & c) | (b & d) | (c & d); k = 0x8F1BBCDCu; }
            else             { f = b ^ c ^ d;            k = 0xCA62C1D6u; }
            uint32_t t = es_rol32(a, 5) + f + e + k + w[i];
            e = d; d = c; c = es_rol32(b, 30); b = a; a = t;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e;
    }
    for (int i = 0; i < 5; ++i) {
        out[i*4]   = (unsigned char)(h[i] >> 24);
        out[i*4+1] = (unsigned char)(h[i] >> 16);
        out[i*4+2] = (unsigned char)(h[i] >> 8);
        out[i*4+3] = (unsigned char)(h[i]);
    }
}

static void es_base64_20(const unsigned char in[20], char out[29]) {
    static const char tab[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t i = 0, o = 0;
    while (i < 20) {
        uint32_t a = in[i++], b = i < 20 ? in[i++] : 0, c = i < 20 ? in[i++] : 0;
        out[o++] = tab[a >> 2];
        out[o++] = tab[((a & 3u) << 4) | (b >> 4)];
        out[o++] = (i - 1u < 20u) ? tab[((b & 15u) << 2) | (c >> 6)] : '=';
        out[o++] = (i < 20u) ? tab[c & 63u] : '=';
    }
    out[28] = '\0';
}

static int es_ascii_ieq_n(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        unsigned char ca = (unsigned char)a[i];
        unsigned char cb = (unsigned char)b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return 1;
}

static int es_header_contains_token(const char* val, size_t len, const char* tok) {
    size_t tlen = strlen(tok), pos = 0;
    while (pos < len) {
        while (pos < len && (val[pos] == ' ' || val[pos] == '\t' || val[pos] == ',')) ++pos;
        size_t end = pos;
        while (end < len && val[end] != ',') ++end;
        size_t trim = end;
        while (trim > pos && (val[trim - 1] == ' ' || val[trim - 1] == '\t')) --trim;
        if (trim - pos == tlen && es_ascii_ieq_n(val + pos, tok, tlen)) return 1;
        pos = (end < len) ? end + 1 : len;
    }
    return 0;
}

static int es_ws_key_valid(const char* key, size_t len) {
    if (len != 24 || key[22] != '=' || key[23] != '=') return 0;
    for (size_t i = 0; i < 22; ++i) {
        unsigned char c = (unsigned char)key[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '+' || c == '/')) return 0;
    }
    unsigned char last = (unsigned char)key[21];
    int v = (last >= 'A' && last <= 'Z') ? (last - 'A') :
            (last >= 'a' && last <= 'z') ? (last - 'a' + 26) :
            (last >= '0' && last <= '9') ? (last - '0' + 52) :
            (last == '+') ? 62 : 63;
    return (v & 15) == 0;
}

static int es_utf8_valid(const unsigned char* s, size_t n) {
    size_t i = 0;
    while (i < n) {
        unsigned char c = s[i++];
        if (c < 0x80u) continue;
        if (c >= 0xC2u && c <= 0xDFu) {
            if (i >= n || (s[i++] & 0xC0u) != 0x80u) return 0;
        } else if (c >= 0xE0u && c <= 0xEFu) {
            if (i + 1u >= n) return 0;
            unsigned char a = s[i++], b = s[i++];
            if ((a & 0xC0u) != 0x80u || (b & 0xC0u) != 0x80u ||
                (c == 0xE0u && a < 0xA0u) || (c == 0xEDu && a >= 0xA0u)) return 0;
        } else if (c >= 0xF0u && c <= 0xF4u) {
            if (i + 2u >= n) return 0;
            unsigned char a = s[i++], b = s[i++], d = s[i++];
            if ((a & 0xC0u) != 0x80u || (b & 0xC0u) != 0x80u || (d & 0xC0u) != 0x80u ||
                (c == 0xF0u && a < 0x90u) || (c == 0xF4u && a >= 0x90u)) return 0;
        } else {
            return 0;
        }
    }
    return 1;
}

static int es_ws_close_code_valid(unsigned code) {
    return (code >= 1000u && code <= 1014u && code != 1004u && code != 1005u && code != 1006u) ||
           (code >= 3000u && code <= 4999u);
}

/* ── Connection lifecycle and buffer management ──────────────────────────── */

static int conn_update_interest(eshkol_http_event_server_t* srv, int slot, int want) {
    http_conn_t* c = &srv->conns[slot];
    if (c->armed_events == want) return 0;
    uint64_t cid = make_conn_id((uint32_t)slot, c->generation);
    if (eshkol_event_loop_add(srv->loop, (int)c->fd, want, cid) != ESHKOL_EVENT_LOOP_OK)
        return -1;
    c->armed_events = want;
    return 0;
}

static void conn_close_slot(eshkol_http_event_server_t* srv, int slot, int notify) {
    http_conn_t* c = &srv->conns[slot];
    if (c->state == CONN_FREE) return;

    uint64_t cid = make_conn_id((uint32_t)slot, c->generation);
    conn_state_t prev_state = c->state;
    esh_sock_t fd = c->fd;

    if (fd != ESH_BAD_SOCK) {
        (void)eshkol_event_loop_remove(srv->loop, (int)fd);
        (void)esh_sock_close(fd);
    }
    free(c->rbuf);
    free(c->wbuf);

    c->fd = ESH_BAD_SOCK;
    c->state = CONN_FREE;
    c->armed_events = 0;
    c->close_after_write = 0;
    c->rbuf = NULL; c->rlen = 0; c->rcap = 0;
    c->wbuf = NULL; c->wpos = 0; c->wlen = 0; c->wcap = 0;
    c->free_next = srv->free_head;
    srv->free_head = slot;

    if (srv->stats.active_connections > 0) --srv->stats.active_connections;
    if (prev_state == CONN_WS && srv->stats.active_ws_connections > 0)
        --srv->stats.active_ws_connections;
    ++srv->stats.total_closed;

    if (notify && srv->on_close) srv->on_close(srv, cid, srv->user_data);
}

static void conn_reclaim_rbuf_if_empty(http_conn_t* c) {
    if (c->rlen == 0 && c->rbuf) {
        free(c->rbuf);
        c->rbuf = NULL;
        c->rcap = 0;
    }
}

static void conn_reclaim_wbuf_if_empty(http_conn_t* c) {
    if (c->wpos >= c->wlen && c->wbuf) {
        free(c->wbuf);
        c->wbuf = NULL;
        c->wpos = 0;
        c->wlen = 0;
        c->wcap = 0;
    }
}

static int conn_wbuf_append(http_conn_t* c, const void* data, size_t len) {
    if (len == 0) return 0;
    size_t live = c->wlen - c->wpos;
    if (c->wpos > 0 && live == 0) { c->wpos = 0; c->wlen = 0; }
    if (c->wlen + len > c->wcap) {
        if (c->wpos > 0 && live + len <= c->wcap) {
            memmove(c->wbuf, c->wbuf + c->wpos, live);
            c->wpos = 0;
            c->wlen = live;
        } else {
            size_t need = live + len;
            size_t new_cap = c->wcap ? c->wcap : 256u;
            while (new_cap < need) {
                if (new_cap > (SIZE_MAX / 2u)) { new_cap = need; break; }
                new_cap *= 2u;
            }
            unsigned char* nb = (unsigned char*)malloc(new_cap);
            if (!nb) return -1;
            if (live > 0) memcpy(nb, c->wbuf + c->wpos, live);
            free(c->wbuf);
            c->wbuf = nb;
            c->wpos = 0;
            c->wlen = live;
            c->wcap = new_cap;
        }
    }
    memcpy(c->wbuf + c->wlen, data, len);
    c->wlen += len;
    return 0;
}

static int conn_flush_write(eshkol_http_event_server_t* srv, int slot) {
    http_conn_t* c = &srv->conns[slot];
    while (c->wpos < c->wlen) {
        size_t rem = c->wlen - c->wpos;
#ifdef _WIN32
        int chunk = rem > (size_t)INT_MAX ? INT_MAX : (int)rem;
        int n = send(c->fd, (const char*)(c->wbuf + c->wpos), chunk, 0);
#else
        int flags = 0;
#ifdef MSG_NOSIGNAL
        flags = MSG_NOSIGNAL;
#endif
        ssize_t n = send(c->fd, c->wbuf + c->wpos, rem, flags);
#endif
        if (n > 0) {
            c->wpos += (size_t)n;
            c->last_active_ms = esh_now_ms();
            continue;
        }
        if (n < 0 && esh_sock_wouldblock()) {
            return conn_update_interest(srv, slot, ESHKOL_EVENT_READ | ESHKOL_EVENT_WRITE);
        }
        return -1;
    }
    conn_reclaim_wbuf_if_empty(c);
    if (c->close_after_write) {
        conn_close_slot(srv, slot, 1);
        return 0;
    }
    return conn_update_interest(srv, slot, ESHKOL_EVENT_READ);
}

static int conn_send_simple_error(eshkol_http_event_server_t* srv, int slot,
                                  int32_t status, const char* reason) {
    http_conn_t* c = &srv->conns[slot];
    char buf[256];
    int n = snprintf(buf, sizeof(buf),
        "HTTP/1.1 %d %s\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        status, reason);
    if (n <= 0 || (size_t)n >= sizeof(buf)) {
        conn_close_slot(srv, slot, 1);
        return -1;
    }
    c->close_after_write = 1;
    if (conn_wbuf_append(c, buf, (size_t)n) != 0 || conn_flush_write(srv, slot) != 0) {
        conn_close_slot(srv, slot, 1);
        return -1;
    }
    ++srv->stats.total_rejected;
    return 0;
}

/* ── RFC 6455 non-blocking frame processing ──────────────────────────────── */

static int conn_queue_ws_frame(eshkol_http_event_server_t* srv, int slot,
                               int opcode, const void* payload, size_t len) {
    http_conn_t* c = &srv->conns[slot];
    unsigned char hdr[10];
    size_t hlen = 0;
    hdr[hlen++] = (unsigned char)(0x80u | (unsigned)(opcode & 0x0Fu));
    if (len < 126u) {
        hdr[hlen++] = (unsigned char)len;
    } else if (len < 65536u) {
        hdr[hlen++] = 126u;
        hdr[hlen++] = (unsigned char)(len >> 8);
        hdr[hlen++] = (unsigned char)(len & 0xFFu);
    } else {
        hdr[hlen++] = 127u;
        uint64_t l64 = (uint64_t)len;
        for (int i = 7; i >= 0; --i) hdr[hlen++] = (unsigned char)((l64 >> (i * 8)) & 0xFFu);
    }
    if (conn_wbuf_append(c, hdr, hlen) != 0) return -1;
    if (len > 0 && conn_wbuf_append(c, payload, len) != 0) return -1;
    return conn_flush_write(srv, slot);
}

static void conn_process_ws_frames(eshkol_http_event_server_t* srv, int slot) {
    http_conn_t* c = &srv->conns[slot];
    uint32_t gen = c->generation;
    uint64_t cid = make_conn_id((uint32_t)slot, gen);

    while (c->state == CONN_WS && c->generation == gen && c->rlen >= 2u) {
        unsigned char b0 = c->rbuf[0], b1 = c->rbuf[1];
        int fin = (b0 & 0x80u) != 0;
        int rsv = (b0 & 0x70u) != 0;
        int opcode = (int)(b0 & 0x0Fu);
        int masked = (b1 & 0x80u) != 0;
        if (!fin || rsv || !masked) {
            conn_close_slot(srv, slot, 1);
            return;
        }
        if (opcode != 1 && opcode != 2 && opcode != 8 && opcode != 9 && opcode != 10) {
            conn_close_slot(srv, slot, 1);
            return;
        }
        uint64_t plen = (uint64_t)(b1 & 0x7Fu);
        size_t pos = 2u;
        if (plen == 126u) {
            if (c->rlen < 4u) break;
            plen = ((uint64_t)c->rbuf[2] << 8) | (uint64_t)c->rbuf[3];
            if (plen < 126u) { conn_close_slot(srv, slot, 1); return; }
            pos = 4u;
        } else if (plen == 127u) {
            if (c->rlen < 10u) break;
            if (c->rbuf[2] & 0x80u) { conn_close_slot(srv, slot, 1); return; }
            plen = 0;
            for (int i = 0; i < 8; ++i) plen = (plen << 8) | (uint64_t)c->rbuf[2 + i];
            if (plen < 65536u) { conn_close_slot(srv, slot, 1); return; }
            pos = 10u;
        }
        if ((opcode & 8) && plen > 125u) { conn_close_slot(srv, slot, 1); return; }
        if (plen > srv->max_body_bytes) { conn_close_slot(srv, slot, 1); return; }

        size_t frame_bytes = pos + 4u + (size_t)plen;
        if (c->rlen < frame_bytes) break;

        unsigned char mask[4];
        memcpy(mask, c->rbuf + pos, 4u);
        unsigned char* payload = c->rbuf + pos + 4u;
        for (size_t i = 0; i < (size_t)plen; ++i) payload[i] ^= mask[i & 3u];

        if (opcode == 1 && !es_utf8_valid(payload, (size_t)plen)) {
            conn_close_slot(srv, slot, 1);
            return;
        }
        if (opcode == 8) {
            if (plen == 1u) { conn_close_slot(srv, slot, 1); return; }
            if (plen >= 2u) {
                unsigned code = ((unsigned)payload[0] << 8) | (unsigned)payload[1];
                if (!es_ws_close_code_valid(code) || !es_utf8_valid(payload + 2u, (size_t)plen - 2u)) {
                    conn_close_slot(srv, slot, 1);
                    return;
                }
            }
            c->close_after_write = 1;
            ++srv->stats.total_ws_frames;
            (void)conn_queue_ws_frame(srv, slot, 8, payload, (size_t)plen);
            return;
        }

        /* Copy payload out before shifting rbuf if callback might re-enter or close. */
        unsigned char stack_copy[256];
        unsigned char* msg = stack_copy;
        int heap_copy = 0;
        if (plen > sizeof(stack_copy)) {
            msg = (unsigned char*)malloc((size_t)plen + 1u);
            if (!msg) { conn_close_slot(srv, slot, 1); return; }
            heap_copy = 1;
        }
        if (plen > 0) memcpy(msg, payload, (size_t)plen);
        if (plen < sizeof(stack_copy) || heap_copy) msg[plen] = '\0';

        size_t rem = c->rlen - frame_bytes;
        if (rem > 0) memmove(c->rbuf, c->rbuf + frame_bytes, rem);
        c->rlen = rem;
        conn_reclaim_rbuf_if_empty(c);
        ++srv->stats.total_ws_frames;

        if (opcode == 9) {
            int rc = conn_queue_ws_frame(srv, slot, 10, msg, (size_t)plen);
            if (heap_copy) free(msg);
            if (rc != 0) { conn_close_slot(srv, slot, 1); return; }
        } else if (opcode == 10) {
            if (heap_copy) free(msg);
        } else {
            if (srv->on_ws_frame) srv->on_ws_frame(srv, cid, opcode, msg, (size_t)plen, srv->user_data);
            if (heap_copy) free(msg);
        }
    }
}

/* ── RFC 9112 non-blocking HTTP/1.1 request parsing & dispatch ───────────── */

static void conn_process_http(eshkol_http_event_server_t* srv, int slot) {
    http_conn_t* c = &srv->conns[slot];
    uint32_t gen = c->generation;
    uint64_t cid = make_conn_id((uint32_t)slot, gen);

    while (c->state == CONN_HTTP && c->generation == gen && c->rlen > 0) {
        /* Locate end of header block ("\r\n\r\n"). */
        size_t head_end = 0;
        int found = 0;
        for (size_t i = 0; i + 3u < c->rlen; ++i) {
            if (c->rbuf[i] == '\r' && c->rbuf[i+1] == '\n' &&
                c->rbuf[i+2] == '\r' && c->rbuf[i+3] == '\n') {
                head_end = i;
                found = 1;
                break;
            }
        }
        if (!found) {
            if (c->rlen > srv->max_header_bytes) {
                (void)conn_send_simple_error(srv, slot, 431, "Request Header Fields Too Large");
            }
            return;
        }
        if (head_end + 4u > srv->max_header_bytes) {
            (void)conn_send_simple_error(srv, slot, 431, "Request Header Fields Too Large");
            return;
        }

        const char* base = (const char*)c->rbuf;
        const char* line_end = NULL;
        for (size_t i = 0; i + 1u < head_end + 2u; ++i) {
            if (base[i] == '\r' && base[i+1] == '\n') { line_end = base + i; break; }
        }
        if (!line_end) {
            (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
            return;
        }

        /* Parse request-line: METHOD SP TARGET SP HTTP/1.x */
        const char* sp1 = (const char*)memchr(base, ' ', (size_t)(line_end - base));
        if (!sp1 || sp1 == base) { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
        const char* sp2 = (const char*)memchr(sp1 + 1, ' ', (size_t)(line_end - (sp1 + 1)));
        if (!sp2 || sp2 == sp1 + 1) { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
        size_t method_len = (size_t)(sp1 - base);
        size_t target_len = (size_t)(sp2 - (sp1 + 1));
        size_t ver_len = (size_t)(line_end - (sp2 + 1));
        if (!eshkol_http_valid_token(base, method_len) || ver_len != 8u ||
            memcmp(sp2 + 1, "HTTP/1.", 7u) != 0 ||
            (sp2[8] != '0' && sp2[8] != '1')) {
            (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
            return;
        }
        int is_http11 = (sp2[8] == '1');
        for (size_t i = 0; i < target_len; ++i) {
            if (!eshkol_http_is_target_char((unsigned char)sp1[1 + i])) {
                (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                return;
            }
        }

        /* Parse and strictly validate all headers (RFC 9112). */
        int have_host = 0, have_cl = 0, have_te = 0;
        int conn_close = 0, conn_keep = 0, conn_upgrade = 0;
        int ws_upgrade = 0, ws_ver13 = 0, ws_has_key = 0;
        char ws_key[25];
        uint64_t content_length = 0;

        const char* cur = line_end + 2;
        const char* hdr_limit = base + head_end;
        while (cur < hdr_limit) {
            const char* e = cur;
            while (e < hdr_limit && !(e[0] == '\r' && e[1] == '\n')) ++e;
            const char* colon = (const char*)memchr(cur, ':', (size_t)(e - cur));
            if (!colon || colon == cur) {
                (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                return;
            }
            size_t nlen = (size_t)(colon - cur);
            if (!eshkol_http_valid_token(cur, nlen)) {
                (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                return;
            }
            const char* val = colon + 1;
            while (val < e && (*val == ' ' || *val == '\t')) ++val;
            const char* vend = e;
            while (vend > val && (vend[-1] == ' ' || vend[-1] == '\t')) --vend;
            size_t vlen = (size_t)(vend - val);
            for (size_t i = 0; i < vlen; ++i) {
                unsigned char ch = (unsigned char)val[i];
                if ((ch < 0x20u && ch != '\t') || ch == 0x7Fu) {
                    (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                    return;
                }
            }
            if (nlen == 4 && es_ascii_ieq_n(cur, "Host", 4)) {
                if (have_host++ || vlen == 0) { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
            } else if (nlen == 14 && es_ascii_ieq_n(cur, "Content-Length", 14)) {
                if (have_cl++ || vlen == 0) { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
                uint64_t parsed = 0;
                for (size_t i = 0; i < vlen; ++i) {
                    if (val[i] < '0' || val[i] > '9') { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
                    uint64_t d = (uint64_t)(val[i] - '0');
                    if (parsed > (UINT64_MAX - d) / 10u) { (void)conn_send_simple_error(srv, slot, 400, "Bad Request"); return; }
                    parsed = parsed * 10u + d;
                }
                content_length = parsed;
            } else if (nlen == 17 && es_ascii_ieq_n(cur, "Transfer-Encoding", 17)) {
                have_te = 1;
            } else if (nlen == 10 && es_ascii_ieq_n(cur, "Connection", 10)) {
                if (es_header_contains_token(val, vlen, "close")) conn_close = 1;
                if (es_header_contains_token(val, vlen, "keep-alive")) conn_keep = 1;
                if (es_header_contains_token(val, vlen, "upgrade")) conn_upgrade = 1;
            } else if (nlen == 7 && es_ascii_ieq_n(cur, "Upgrade", 7)) {
                if (vlen == 9 && es_ascii_ieq_n(val, "websocket", 9)) ws_upgrade = 1;
            } else if (nlen == 21 && es_ascii_ieq_n(cur, "Sec-WebSocket-Version", 21)) {
                if (vlen == 2 && memcmp(val, "13", 2) == 0) ws_ver13 = 1;
            } else if (nlen == 17 && es_ascii_ieq_n(cur, "Sec-WebSocket-Key", 17)) {
                if (ws_has_key++ || !es_ws_key_valid(val, vlen)) {
                    (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                    return;
                }
                memcpy(ws_key, val, 24u);
                ws_key[24] = '\0';
            }
            cur = (e + 2 <= hdr_limit) ? (e + 2) : hdr_limit;
        }

        if (have_te) {
            /* Refuse Transfer-Encoding (CL+TE smuggling defense & bounded framing). */
            (void)conn_send_simple_error(srv, slot, have_cl ? 400 : 501,
                                         have_cl ? "Bad Request" : "Not Implemented");
            return;
        }
        if (is_http11 && !have_host) {
            (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
            return;
        }
        if (content_length > srv->max_body_bytes) {
            (void)conn_send_simple_error(srv, slot, 413, "Payload Too Large");
            return;
        }

        size_t req_total = head_end + 4u + (size_t)content_length;
        if (c->rlen < req_total) return; /* Wait for remaining body bytes. */

        /* Check for RFC 6455 WebSocket upgrade request. */
        if (srv->allow_websocket && ws_upgrade && conn_upgrade) {
            if (!is_http11 || method_len != 3 || memcmp(base, "GET", 3) != 0 ||
                !ws_ver13 || ws_has_key != 1 || content_length != 0) {
                (void)conn_send_simple_error(srv, slot, 400, "Bad Request");
                return;
            }
            static const char guid[] = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
            unsigned char sha_in[60], digest[20];
            char accept_b64[29];
            memcpy(sha_in, ws_key, 24u);
            memcpy(sha_in + 24u, guid, sizeof(guid) - 1u);
            es_sha1(sha_in, 24u + sizeof(guid) - 1u, digest);
            es_base64_20(digest, accept_b64);

            char resp[256];
            int rlen = snprintf(resp, sizeof(resp),
                "HTTP/1.1 101 Switching Protocols\r\n"
                "Upgrade: websocket\r\n"
                "Connection: Upgrade\r\n"
                "Sec-WebSocket-Accept: %s\r\n\r\n", accept_b64);
            if (rlen <= 0 || (size_t)rlen >= sizeof(resp) ||
                conn_wbuf_append(c, resp, (size_t)rlen) != 0 ||
                conn_flush_write(srv, slot) != 0) {
                conn_close_slot(srv, slot, 1);
                return;
            }
            size_t extra = c->rlen - req_total;
            if (extra > 0) memmove(c->rbuf, c->rbuf + req_total, extra);
            c->rlen = extra;
            conn_reclaim_rbuf_if_empty(c);
            c->state = CONN_WS;
            ++srv->stats.total_ws_upgrades;
            ++srv->stats.active_ws_connections;
            if (c->rlen > 0) conn_process_ws_frames(srv, slot);
            return;
        }

        /* Normal HTTP/1.1 request: dispatch to handler. */
        int keep_alive = is_http11 ? !conn_close : (conn_keep && !conn_close);
        eshkol_http_event_request_t req;
        req.method = base;
        req.method_len = method_len;
        req.target = sp1 + 1;
        req.target_len = target_len;
        req.raw_headers = (line_end + 2 <= hdr_limit) ? (line_end + 2) : hdr_limit;
        req.raw_headers_len = (size_t)(hdr_limit - req.raw_headers);
        req.body = base + head_end + 4u;
        req.body_len = (size_t)content_length;
        req.keep_alive = keep_alive;

        /* Copy slice into temporary buffer only if pipelined bytes exist;
         * in the common non-pipelined case rbuf stays valid across callback
         * and is reclaimed immediately after! */
        size_t extra = c->rlen - req_total;
        ++srv->stats.total_http_requests;

        if (srv->on_request) {
            srv->on_request(srv, cid, &req, srv->user_data);
        } else {
            (void)eshkol_http_event_conn_respond(srv, cid, 200, "text/plain", "OK\n", 3u, keep_alive);
        }

        if (c->state != CONN_HTTP || c->generation != gen) return;
        if (extra > 0) {
            memmove(c->rbuf, c->rbuf + req_total, extra);
            c->rlen = extra;
        } else {
            c->rlen = 0;
            conn_reclaim_rbuf_if_empty(c);
        }
        if (c->close_after_write) return;
    }
}

static void conn_handle_read(eshkol_http_event_server_t* srv, int slot) {
    http_conn_t* c = &srv->conns[slot];
    uint32_t gen = c->generation;
    unsigned char tmp[4096];

    for (;;) {
        if (c->state == CONN_FREE || c->generation != gen) return;
#ifdef _WIN32
        int n = recv(c->fd, (char*)tmp, (int)sizeof(tmp), 0);
#else
        ssize_t n = recv(c->fd, tmp, sizeof(tmp), 0);
#endif
        if (n > 0) {
            size_t got = (size_t)n;
            size_t max_allowed = srv->max_header_bytes + srv->max_body_bytes + 64u;
            if (c->rlen + got > max_allowed) {
                if (c->state == CONN_HTTP)
                    (void)conn_send_simple_error(srv, slot, 413, "Payload Too Large");
                else
                    conn_close_slot(srv, slot, 1);
                return;
            }
            if (c->rlen + got > c->rcap) {
                size_t new_cap = c->rcap ? c->rcap : 512u;
                while (new_cap < c->rlen + got) {
                    if (new_cap > max_allowed / 2u) { new_cap = c->rlen + got; break; }
                    new_cap *= 2u;
                }
                unsigned char* nb = (unsigned char*)realloc(c->rbuf, new_cap);
                if (!nb) { conn_close_slot(srv, slot, 1); return; }
                c->rbuf = nb;
                c->rcap = new_cap;
            }
            memcpy(c->rbuf + c->rlen, tmp, got);
            c->rlen += got;
            c->last_active_ms = esh_now_ms();

            if (c->state == CONN_HTTP) conn_process_http(srv, slot);
            else if (c->state == CONN_WS) conn_process_ws_frames(srv, slot);

            if (c->state == CONN_FREE || c->generation != gen) return;
            if (got < sizeof(tmp)) break;
            continue;
        }
        if (n == 0) {
            conn_close_slot(srv, slot, 1);
            return;
        }
        if (esh_sock_wouldblock()) break;
        conn_close_slot(srv, slot, 1);
        return;
    }
}

static void server_accept_all(eshkol_http_event_server_t* srv) {
    uint64_t now = esh_now_ms();
    for (;;) {
        struct sockaddr_in addr;
#ifdef _WIN32
        int alen = (int)sizeof(addr);
#else
        socklen_t alen = (socklen_t)sizeof(addr);
#endif
        esh_sock_t cfd = accept(srv->listen_fd, (struct sockaddr*)&addr, &alen);
        if (cfd == ESH_BAD_SOCK) {
            if (esh_sock_wouldblock()) break;
            break;
        }
        if (srv->free_head < 0) {
            (void)esh_sock_close(cfd);
            ++srv->stats.total_rejected;
            continue;
        }
        if (esh_sock_nonblock(cfd) != 0) {
            (void)esh_sock_close(cfd);
            ++srv->stats.total_rejected;
            continue;
        }
#ifdef TCP_NODELAY
        { int one = 1; (void)setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, (const char*)&one, sizeof(one)); }
#endif
#ifdef SO_NOSIGPIPE
        { int one = 1; (void)setsockopt(cfd, SOL_SOCKET, SO_NOSIGPIPE, &one, sizeof(one)); }
#endif
        int slot = srv->free_head;
        http_conn_t* c = &srv->conns[slot];
        uint32_t next_gen = c->generation + 1u;
        if (next_gen == 0) next_gen = 1u;
        uint64_t cid = make_conn_id((uint32_t)slot, next_gen);

        if (eshkol_event_loop_add(srv->loop, (int)cfd, ESHKOL_EVENT_READ, cid) != ESHKOL_EVENT_LOOP_OK) {
            (void)esh_sock_close(cfd);
            ++srv->stats.total_rejected;
            continue;
        }
        srv->free_head = c->free_next;
        c->free_next = -1;
        c->fd = cfd;
        c->state = CONN_HTTP;
        c->generation = next_gen;
        c->armed_events = ESHKOL_EVENT_READ;
        c->close_after_write = 0;
        c->last_active_ms = now;
        c->rbuf = NULL; c->rlen = 0; c->rcap = 0;
        c->wbuf = NULL; c->wpos = 0; c->wlen = 0; c->wcap = 0;

        ++srv->stats.total_accepted;
        ++srv->stats.active_connections;
        if (srv->stats.active_connections > srv->stats.peak_connections)
            srv->stats.peak_connections = srv->stats.active_connections;
    }
}

static void server_sweep_idle(eshkol_http_event_server_t* srv) {
    if (srv->idle_timeout_ms <= 0) return;
    uint64_t now = esh_now_ms();
    if (now < srv->last_sweep_ms + 500u) return;
    srv->last_sweep_ms = now;
    uint64_t limit = (uint64_t)srv->idle_timeout_ms;
    for (int i = 0; i < srv->max_conns; ++i) {
        if (srv->conns[i].state != CONN_FREE &&
            now > srv->conns[i].last_active_ms &&
            now - srv->conns[i].last_active_ms >= limit) {
            conn_close_slot(srv, i, 1);
        }
    }
}

/* ── Public API ─────────────────────────────────────────────────────────── */

eshkol_http_event_server_t* eshkol_http_event_server_create(
    const eshkol_http_event_config_t* config,
    eshkol_http_event_request_cb_t on_request,
    eshkol_http_event_ws_frame_cb_t on_ws_frame,
    eshkol_http_event_close_cb_t on_close,
    void* user_data) {
    if (!esh_sock_init()) return NULL;

    int32_t port = config ? config->port : 0;
    int32_t max_conns = (config && config->max_connections > 0) ? config->max_connections : ESH_DEFAULT_MAX_CONNS;
    if (max_conns > ESH_MAX_CONNS_CAP) max_conns = ESH_MAX_CONNS_CAP;
    size_t max_hdr = (config && config->max_header_bytes > 0) ? config->max_header_bytes : ESH_DEFAULT_MAX_HEADER;
    size_t max_body = (config && config->max_body_bytes > 0) ? config->max_body_bytes : ESH_DEFAULT_MAX_BODY;

    esh_sock_t lfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (lfd == ESH_BAD_SOCK) return NULL;
    int one = 1;
    (void)setsockopt(lfd, SOL_SOCKET, SO_REUSEADDR, (const char*)&one, sizeof(one));
    if (esh_sock_nonblock(lfd) != 0) {
        (void)esh_sock_close(lfd);
        return NULL;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    addr.sin_port = htons((uint16_t)port);
    if (bind(lfd, (struct sockaddr*)&addr, sizeof(addr)) < 0 ||
        listen(lfd, 4096) < 0) {
        (void)esh_sock_close(lfd);
        return NULL;
    }
#ifdef _WIN32
    int alen = (int)sizeof(addr);
#else
    socklen_t alen = (socklen_t)sizeof(addr);
#endif
    if (getsockname(lfd, (struct sockaddr*)&addr, &alen) != 0) {
        (void)esh_sock_close(lfd);
        return NULL;
    }

    eshkol_event_loop_t* loop = eshkol_event_loop_create(max_conns + 2);
    if (!loop) {
        (void)esh_sock_close(lfd);
        return NULL;
    }
    if (eshkol_event_loop_add(loop, (int)lfd, ESHKOL_EVENT_READ, ESH_LISTEN_COOKIE) != ESHKOL_EVENT_LOOP_OK) {
        (void)eshkol_event_loop_close(loop);
        (void)esh_sock_close(lfd);
        return NULL;
    }

    eshkol_http_event_server_t* srv = (eshkol_http_event_server_t*)calloc(1, sizeof(*srv));
    http_conn_t* conns = (http_conn_t*)calloc((size_t)max_conns, sizeof(http_conn_t));
    if (!srv || !conns) {
        free(srv);
        free(conns);
        (void)eshkol_event_loop_close(loop);
        (void)esh_sock_close(lfd);
        return NULL;
    }
    for (int i = 0; i < max_conns; ++i) {
        conns[i].fd = ESH_BAD_SOCK;
        conns[i].state = CONN_FREE;
        conns[i].free_next = (i + 1 < max_conns) ? (i + 1) : -1;
    }
    srv->loop = loop;
    srv->listen_fd = lfd;
    srv->port = ntohs(addr.sin_port);
    srv->max_conns = max_conns;
    srv->max_header_bytes = max_hdr;
    srv->max_body_bytes = max_body;
    srv->idle_timeout_ms = config ? config->idle_timeout_ms : 0;
    srv->allow_websocket = config ? config->allow_websocket : 1;
    srv->last_sweep_ms = esh_now_ms();
    srv->conns = conns;
    srv->free_head = 0;
    srv->on_request = on_request;
    srv->on_ws_frame = on_ws_frame;
    srv->on_close = on_close;
    srv->user_data = user_data;
    return srv;
}

int32_t eshkol_http_event_server_port(const eshkol_http_event_server_t* server) {
    return server ? (int32_t)server->port : -1;
}

int eshkol_http_event_server_poll(eshkol_http_event_server_t* server, int timeout_ms) {
    if (!server) return -1;
    const eshkol_event_t* events = NULL;
    int n_events = 0;
    int rc = eshkol_event_loop_poll(server->loop, timeout_ms, &events, &n_events);
    if (rc != ESHKOL_EVENT_LOOP_OK) return -1;

    for (int i = 0; i < n_events; ++i) {
        if (events[i].user_data == ESH_LISTEN_COOKIE && events[i].fd == (int)server->listen_fd) {
            server_accept_all(server);
            continue;
        }
        int slot = -1;
        http_conn_t* c = resolve_conn(server, events[i].user_data, &slot);
        if (!c) continue;
        uint32_t gen = c->generation;

        if (events[i].events & ESHKOL_EVENT_WRITE) {
            if (conn_flush_write(server, slot) != 0) {
                conn_close_slot(server, slot, 1);
                continue;
            }
            if (c->state == CONN_FREE || c->generation != gen) continue;
        }
        if (events[i].events & ESHKOL_EVENT_READ) {
            conn_handle_read(server, slot);
            if (c->state == CONN_FREE || c->generation != gen) continue;
        }
        if ((events[i].events & ESHKOL_EVENT_ERROR) && !(events[i].events & ESHKOL_EVENT_READ)) {
            conn_close_slot(server, slot, 1);
        }
    }
    server_sweep_idle(server);
    return n_events;
}

static const char* status_reason(int32_t status) {
    switch (status) {
        case 200: return "OK";
        case 201: return "Created";
        case 204: return "No Content";
        case 301: return "Moved Permanently";
        case 302: return "Found";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 403: return "Forbidden";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 431: return "Request Header Fields Too Large";
        case 500: return "Internal Server Error";
        case 501: return "Not Implemented";
        default:  return "OK";
    }
}

int eshkol_http_event_conn_respond(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    int32_t status,
    const char* content_type,
    const void* body,
    size_t body_len,
    int keep_alive) {
    int slot = -1;
    http_conn_t* c = resolve_conn(server, conn_id, &slot);
    if (!c || c->state != CONN_HTTP) return -1;
    if (body_len > 0 && !body) return -1;

    const char* ctype = content_type ? content_type : "text/plain";
    if (!eshkol_http_valid_field_value(ctype, strlen(ctype))) return -1;

    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: %s\r\n\r\n",
        status, status_reason(status), ctype, body_len,
        keep_alive ? "keep-alive" : "close");
    if (hlen <= 0 || (size_t)hlen >= sizeof(header)) return -1;

    if (!keep_alive) c->close_after_write = 1;
    if (conn_wbuf_append(c, header, (size_t)hlen) != 0) {
        conn_close_slot(server, slot, 1);
        return -1;
    }
    if (body_len > 0 && conn_wbuf_append(c, body, body_len) != 0) {
        conn_close_slot(server, slot, 1);
        return -1;
    }
    if (conn_flush_write(server, slot) != 0) {
        conn_close_slot(server, slot, 1);
        return -1;
    }
    return 0;
}

int eshkol_http_event_ws_send(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    int opcode,
    const void* payload,
    size_t payload_len) {
    int slot = -1;
    http_conn_t* c = resolve_conn(server, conn_id, &slot);
    if (!c || c->state != CONN_WS) return -1;
    if (payload_len > 0 && !payload) return -1;
    if (opcode == 1 && payload_len > 0 && !es_utf8_valid((const unsigned char*)payload, payload_len))
        return -1;
    if (opcode == 8) c->close_after_write = 1;
    if (conn_queue_ws_frame(server, slot, opcode, payload, payload_len) != 0) {
        conn_close_slot(server, slot, 1);
        return -1;
    }
    return 0;
}

void eshkol_http_event_conn_close(eshkol_http_event_server_t* server, uint64_t conn_id) {
    int slot = -1;
    if (resolve_conn(server, conn_id, &slot)) conn_close_slot(server, slot, 1);
}

void eshkol_http_event_server_stats(
    const eshkol_http_event_server_t* server,
    eshkol_http_event_stats_t* out_stats) {
    if (!server || !out_stats) return;
    *out_stats = server->stats;
}

void eshkol_http_event_server_destroy(eshkol_http_event_server_t* server) {
    if (!server) return;
    for (int i = 0; i < server->max_conns; ++i) {
        if (server->conns[i].state != CONN_FREE) conn_close_slot(server, i, 0);
    }
    if (server->listen_fd != ESH_BAD_SOCK) {
        (void)eshkol_event_loop_remove(server->loop, (int)server->listen_fd);
        (void)esh_sock_close(server->listen_fd);
        server->listen_fd = ESH_BAD_SOCK;
    }
    if (server->loop) (void)eshkol_event_loop_close(server->loop);
    free(server->conns);
    free(server);
}
