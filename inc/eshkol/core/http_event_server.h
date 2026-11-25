/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * http_event_server.h — Non-blocking, event-loop-backed concurrent HTTP/1.1
 * and RFC 6455 WebSocket server built directly on eshkol_event_loop_t
 * (kqueue / epoll / IOCP).
 *
 * Designed for >= 10,000 simultaneous connections on a single event-loop
 * thread with O(1) connection dispatch, bounded per-connection buffers that
 * reclaim on keep-alive reset/close (flat RSS), strict RFC 9112 framing
 * validation, and zero leaked file descriptors.
 */
#ifndef ESHKOL_CORE_HTTP_EVENT_SERVER_H
#define ESHKOL_CORE_HTTP_EVENT_SERVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct eshkol_http_event_server eshkol_http_event_server_t;

typedef struct {
    const char* method;
    size_t      method_len;
    const char* target;
    size_t      target_len;
    const char* raw_headers;
    size_t      raw_headers_len;
    const char* body;
    size_t      body_len;
    int         keep_alive;
} eshkol_http_event_request_t;

typedef struct {
    int32_t  port;              /* 0 = ephemeral loopback port */
    int32_t  max_connections;   /* capacity up to 65000 (default 16384) */
    size_t   max_header_bytes;  /* per-request header cap (default 32 KiB) */
    size_t   max_body_bytes;    /* per-request body cap (default 1 MiB) */
    int32_t  idle_timeout_ms;   /* idle connection sweep timeout (0 = disabled) */
    int      allow_websocket;   /* 1 to accept RFC 6455 Upgrade: websocket */
} eshkol_http_event_config_t;

typedef struct {
    uint64_t total_accepted;
    uint64_t total_closed;
    uint64_t total_http_requests;
    uint64_t total_ws_upgrades;
    uint64_t total_ws_frames;
    uint64_t total_rejected;
    uint32_t active_connections;
    uint32_t peak_connections;
    uint32_t active_ws_connections;
} eshkol_http_event_stats_t;

/* Called when a complete HTTP/1.1 request is parsed on conn_id.
 * The handler should call eshkol_http_event_conn_respond(). */
typedef void (*eshkol_http_event_request_cb_t)(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    const eshkol_http_event_request_t* request,
    void* user_data);

/* Called when a complete unfragmented RFC 6455 WebSocket frame (text=1,
 * binary=2) arrives on an upgraded connection. Ping/pong and close control
 * frames are handled automatically by the server. */
typedef void (*eshkol_http_event_ws_frame_cb_t)(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    int opcode,
    const unsigned char* payload,
    size_t payload_len,
    void* user_data);

/* Optional lifecycle notification when a connection closes. */
typedef void (*eshkol_http_event_close_cb_t)(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    void* user_data);

eshkol_http_event_server_t* eshkol_http_event_server_create(
    const eshkol_http_event_config_t* config,
    eshkol_http_event_request_cb_t on_request,
    eshkol_http_event_ws_frame_cb_t on_ws_frame,
    eshkol_http_event_close_cb_t on_close,
    void* user_data);

/* Bound listening port (> 0), or -1 on error. */
int32_t eshkol_http_event_server_port(const eshkol_http_event_server_t* server);

/* Run one non-blocking event-loop poll cycle (accept, read, dispatch, write,
 * idle-timeout sweep). Returns number of events processed (>= 0), or -1 on error. */
int eshkol_http_event_server_poll(eshkol_http_event_server_t* server, int timeout_ms);

/* Queue an HTTP/1.1 response on conn_id. If keep_alive == 0, connection closes
 * cleanly once the write buffer drains. Returns 0 on success, -1 on error. */
int eshkol_http_event_conn_respond(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    int32_t status,
    const char* content_type,
    const void* body,
    size_t body_len,
    int keep_alive);

/* Queue an RFC 6455 server WebSocket frame (opcode 1=text, 2=binary, 8=close,
 * 9=ping, 10=pong) on an upgraded conn_id. Returns 0 on success, -1 on error. */
int eshkol_http_event_ws_send(
    eshkol_http_event_server_t* server,
    uint64_t conn_id,
    int opcode,
    const void* payload,
    size_t payload_len);

/* Close conn_id immediately and release its event-loop registration and buffers. */
void eshkol_http_event_conn_close(eshkol_http_event_server_t* server, uint64_t conn_id);

/* Copy live server counters into *out_stats. */
void eshkol_http_event_server_stats(
    const eshkol_http_event_server_t* server,
    eshkol_http_event_stats_t* out_stats);

/* Close all active connections, close the listening socket and event loop,
 * and release all memory with zero leaked file descriptors. */
void eshkol_http_event_server_destroy(eshkol_http_event_server_t* server);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_HTTP_EVENT_SERVER_H */
