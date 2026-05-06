/*******************************************************************************
 * HTTP Server + WebSocket for Eshkol Agent (B.4)
 *
 * Provides: minimal HTTP/1.1 server (for OAuth PKCE callback, MCP auth),
 * WebSocket client (for voice streaming, remote sessions, MCP WebSocket),
 * and Unix domain socket connect (for IDE IPC).
 *
 * HTTP server is single-connection, blocking — designed for short-lived
 * OAuth callbacks, not production web serving.
 *
 * Copyright (c) 2025 Eshkol Project — tsotchke
 ******************************************************************************/

#ifdef __APPLE__
#define _DARWIN_C_SOURCE
#else
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>

/*******************************************************************************
 * HTTP Server — Minimal, single-connection, for OAuth callbacks
 ******************************************************************************/

#define MAX_HTTP_SERVERS 4

typedef struct {
    int listen_fd;
    int client_fd;       /* Currently connected client (-1 if none) */
    uint16_t port;
} HttpServer;

static HttpServer* g_servers[MAX_HTTP_SERVERS] = {0};

/*
 * Create HTTP server and bind to port.
 * port: TCP port (0 = random available port)
 * Returns: handle (>= 1), -1 error
 */
int64_t eshkol_http_server_create(int32_t port) {
    int slot = -1;
    for (int i = 1; i < MAX_HTTP_SERVERS; i++) {
        if (!g_servers[i]) { slot = i; break; }
    }
    if (slot < 0) return -1;

    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  /* Bind to localhost only */
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }

    if (listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }

    /* Get actual port (if 0 was requested) */
    socklen_t addrlen = sizeof(addr);
    getsockname(fd, (struct sockaddr*)&addr, &addrlen);

    HttpServer* srv = (HttpServer*)calloc(1, sizeof(HttpServer));
    if (!srv) { close(fd); return -1; }
    srv->listen_fd = fd;
    srv->client_fd = -1;
    srv->port = ntohs(addr.sin_port);
    g_servers[slot] = srv;
    return (int64_t)slot;
}

/*
 * Get actual port number.
 */
int32_t eshkol_http_server_port(int64_t handle) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return -1;
    return (int32_t)g_servers[handle]->port;
}

/*
 * Accept one HTTP request. Blocks until a request arrives or timeout.
 *
 * Writes to buf: "METHOD PATH\n" followed by headers (one per line),
 * then "\n" then body.
 *
 * Returns: strlen written to buf, 0 on timeout, -1 error
 */
int32_t eshkol_http_server_accept(int64_t handle, char* buf, int32_t buf_size,
                                    int32_t timeout_ms) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return -1;
    HttpServer* srv = g_servers[handle];
    if (!buf || buf_size <= 0) return -1;

    /* Wait for connection */
    struct pollfd pfd = { .fd = srv->listen_fd, .events = POLLIN };
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0) return ret == 0 ? 0 : -1;

    struct sockaddr_in client_addr;
    socklen_t client_len = sizeof(client_addr);
    int client_fd = accept(srv->listen_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd < 0) return -1;
    srv->client_fd = client_fd;

    /* Set receive timeout */
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    /* Read request */
    int32_t total = 0;
    while (total < buf_size - 1) {
        ssize_t n = recv(client_fd, buf + total, (size_t)(buf_size - 1 - total), 0);
        if (n <= 0) break;
        total += (int32_t)n;
        /* Check for end of headers */
        if (total >= 4 && strstr(buf, "\r\n\r\n")) break;
    }
    buf[total] = '\0';

    return total;
}

/*
 * Send HTTP response on the current connection.
 */
void eshkol_http_server_respond(int64_t handle, int32_t status,
                                  const char* content_type, const char* body) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return;
    HttpServer* srv = g_servers[handle];
    if (srv->client_fd < 0) return;

    const char* reason = "OK";
    if (status == 301) reason = "Moved";
    else if (status == 400) reason = "Bad Request";
    else if (status == 404) reason = "Not Found";
    else if (status == 500) reason = "Internal Server Error";

    int body_len = body ? (int)strlen(body) : 0;
    char header[512];
    int hlen = snprintf(header, sizeof(header),
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n",
        status, reason,
        content_type ? content_type : "text/html",
        body_len);

    send(srv->client_fd, header, (size_t)hlen, 0);
    if (body && body_len > 0) {
        send(srv->client_fd, body, (size_t)body_len, 0);
    }
    close(srv->client_fd);
    srv->client_fd = -1;
}

/*
 * Close and free HTTP server.
 */
void eshkol_http_server_close(int64_t handle) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return;
    HttpServer* srv = g_servers[handle];
    if (srv->client_fd >= 0) close(srv->client_fd);
    close(srv->listen_fd);
    free(srv);
    g_servers[handle] = NULL;
}

/*******************************************************************************
 * Unix Domain Socket Connect (for IDE IPC: VS Code VSCODE_IPC_HOOK)
 ******************************************************************************/

int64_t eshkol_unix_socket_connect(const char* path) {
    if (!path) return -1;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    return (int64_t)fd;
}

/*******************************************************************************
 * WebSocket Client (RFC 6455)
 *
 * Minimal implementation for text/binary frames. Does NOT support
 * extensions (permessage-deflate) or fragmentation. Sufficient for
 * MCP WebSocket transport and voice streaming.
 *
 * For production use, consider linking libcurl 7.86+ or libwebsockets.
 * This implementation handles the common case.
 ******************************************************************************/

#define MAX_WS_HANDLES 8

typedef struct {
    int fd;
    int closed;
} WsConnection;

static WsConnection* g_ws[MAX_WS_HANDLES] = {0};

/* Minimal WebSocket: uses existing TCP socket after HTTP upgrade.
 * Full handshake requires base64(SHA1(key + magic)) which needs
 * either OpenSSL or CommonCrypto. For now, provide the fd-level
 * operations and let the Eshkol wrapper handle the upgrade via
 * the existing HTTP client. */

int64_t eshkol_ws_wrap_fd(int32_t fd) {
    for (int i = 1; i < MAX_WS_HANDLES; i++) {
        if (!g_ws[i]) {
            WsConnection* ws = (WsConnection*)calloc(1, sizeof(WsConnection));
            if (!ws) return -1;
            ws->fd = fd;
            ws->closed = 0;
            g_ws[i] = ws;
            return (int64_t)i;
        }
    }
    return -1;
}

/* Send WebSocket text frame (opcode 0x81, masked) */
int32_t eshkol_ws_send_text(int64_t handle, const char* data, int32_t len) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return -1;
    WsConnection* ws = g_ws[handle];
    if (ws->closed || !data || len <= 0) return -1;

    /* Frame header */
    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = 0x81;  /* FIN=1, opcode=text */

    /* Mask bit = 1 (client must mask), payload length */
    if (len < 126) {
        header[hlen++] = 0x80 | (unsigned char)len;
    } else if (len < 65536) {
        header[hlen++] = 0x80 | 126;
        header[hlen++] = (unsigned char)(len >> 8);
        header[hlen++] = (unsigned char)(len & 0xFF);
    } else {
        header[hlen++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) {
            header[hlen++] = (unsigned char)((len >> (i * 8)) & 0xFF);
        }
    }

    /* Masking key (all zeros for simplicity — still valid per RFC 6455) */
    unsigned char mask[4] = {0, 0, 0, 0};
    memcpy(header + hlen, mask, 4);
    hlen += 4;

    if (send(ws->fd, header, (size_t)hlen, 0) != hlen) return -1;
    /* Data is unmasked because mask key is all zeros */
    if (send(ws->fd, data, (size_t)len, 0) != len) return -1;
    return 0;
}

/* Send WebSocket binary frame (opcode 0x82) */
int32_t eshkol_ws_send_binary(int64_t handle, const char* data, int32_t len) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return -1;
    WsConnection* ws = g_ws[handle];
    if (ws->closed || !data || len <= 0) return -1;

    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = 0x82;  /* FIN=1, opcode=binary */
    if (len < 126) {
        header[hlen++] = 0x80 | (unsigned char)len;
    } else if (len < 65536) {
        header[hlen++] = 0x80 | 126;
        header[hlen++] = (unsigned char)(len >> 8);
        header[hlen++] = (unsigned char)(len & 0xFF);
    } else {
        header[hlen++] = 0x80 | 127;
        for (int i = 7; i >= 0; i--) {
            header[hlen++] = (unsigned char)((len >> (i * 8)) & 0xFF);
        }
    }
    unsigned char mask[4] = {0, 0, 0, 0};
    memcpy(header + hlen, mask, 4);
    hlen += 4;

    if (send(ws->fd, header, (size_t)hlen, 0) != hlen) return -1;
    if (send(ws->fd, data, (size_t)len, 0) != len) return -1;
    return 0;
}

/* Receive WebSocket frame */
int32_t eshkol_ws_receive(int64_t handle, char* buf, int32_t buf_size,
                            int32_t* frame_type, int32_t timeout_ms) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return -1;
    WsConnection* ws = g_ws[handle];
    if (ws->closed || !buf || buf_size <= 0 || !frame_type) return -1;

    /* Poll for data */
    struct pollfd pfd = { .fd = ws->fd, .events = POLLIN };
    int ret = poll(&pfd, 1, timeout_ms);
    if (ret <= 0) return ret == 0 ? 0 : -1;

    /* Read frame header (2 bytes minimum) */
    unsigned char hdr[2];
    if (recv(ws->fd, hdr, 2, MSG_WAITALL) != 2) { ws->closed = 1; return -1; }

    int opcode = hdr[0] & 0x0F;
    int masked = (hdr[1] & 0x80) != 0;
    uint64_t payload_len = hdr[1] & 0x7F;

    if (payload_len == 126) {
        unsigned char ext[2];
        if (recv(ws->fd, ext, 2, MSG_WAITALL) != 2) return -1;
        payload_len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (payload_len == 127) {
        unsigned char ext[8];
        if (recv(ws->fd, ext, 8, MSG_WAITALL) != 8) return -1;
        payload_len = 0;
        for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | ext[i];
    }

    /* Read mask key if present */
    unsigned char mask[4] = {0};
    if (masked) {
        if (recv(ws->fd, mask, 4, MSG_WAITALL) != 4) return -1;
    }

    /* Read payload */
    int32_t to_read = (int32_t)(payload_len < (uint64_t)buf_size ? payload_len : (uint64_t)(buf_size - 1));
    int32_t total = 0;
    while (total < to_read) {
        ssize_t n = recv(ws->fd, buf + total, (size_t)(to_read - total), 0);
        if (n <= 0) break;
        total += (int32_t)n;
    }

    /* Unmask if needed */
    if (masked) {
        for (int i = 0; i < total; i++) {
            buf[i] ^= (char)mask[i % 4];
        }
    }
    buf[total] = '\0';

    /* Map opcode to frame type */
    switch (opcode) {
        case 0x1: *frame_type = 1; break;  /* text */
        case 0x2: *frame_type = 2; break;  /* binary */
        case 0x8: *frame_type = 8; ws->closed = 1; break;  /* close */
        case 0x9: *frame_type = 9; break;  /* ping */
        case 0xA: *frame_type = 10; break; /* pong */
        default:  *frame_type = 0; break;
    }

    /* Auto-respond to ping with pong */
    if (opcode == 0x9) {
        unsigned char pong[2] = { 0x8A, 0x00 };  /* FIN=1, opcode=pong, length=0 */
        send(ws->fd, pong, 2, 0);
    }

    return total;
}

/* Close WebSocket connection */
void eshkol_ws_close(int64_t handle) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return;
    WsConnection* ws = g_ws[handle];
    if (!ws->closed) {
        /* Send close frame */
        unsigned char close_frame[6] = { 0x88, 0x80, 0, 0, 0, 0 };
        send(ws->fd, close_frame, 6, 0);
        ws->closed = 1;
    }
    close(ws->fd);
    free(ws);
    g_ws[handle] = NULL;
}
