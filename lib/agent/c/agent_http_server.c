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

#if defined(__APPLE__)
#define _DARWIN_C_SOURCE
#elif !defined(_WIN32)
#define _GNU_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#include <afunix.h>
#include <windows.h>
#include <bcrypt.h>
typedef SOCKET eshkol_socket_t;
typedef WSAPOLLFD eshkol_pollfd_t;
#define ESHKOL_INVALID_SOCKET INVALID_SOCKET
static INIT_ONCE g_winsock_once = INIT_ONCE_STATIC_INIT;
static int g_winsock_ok = 0;
static BOOL CALLBACK initialize_winsock(PINIT_ONCE once, PVOID param, PVOID* context) {
    (void)once; (void)param; (void)context;
    WSADATA data;
    g_winsock_ok = WSAStartup(MAKEWORD(2, 2), &data) == 0;
    return TRUE;
}
static int ensure_sockets(void) {
    InitOnceExecuteOnce(&g_winsock_once, initialize_winsock, NULL, NULL);
    return g_winsock_ok;
}
static int eshkol_socket_close(eshkol_socket_t socket_value) {
    return closesocket(socket_value);
}
static int eshkol_socket_poll(eshkol_pollfd_t* fds, unsigned long count, int timeout) {
    return WSAPoll(fds, count, timeout);
}
static int eshkol_socket_send(eshkol_socket_t socket_value, const void* data,
                              size_t length, int flags) {
    if (length > INT_MAX) return SOCKET_ERROR;
    return send(socket_value, (const char*)data, (int)length, flags);
}
static int eshkol_socket_recv(eshkol_socket_t socket_value, void* data,
                              size_t length, int flags) {
    if (length > INT_MAX) return SOCKET_ERROR;
    return recv(socket_value, (char*)data, (int)length, flags);
}
#else
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <poll.h>
typedef int eshkol_socket_t;
typedef struct pollfd eshkol_pollfd_t;
#define ESHKOL_INVALID_SOCKET (-1)
static int ensure_sockets(void) { return 1; }
static int eshkol_socket_close(eshkol_socket_t socket_value) { return close(socket_value); }
static int eshkol_socket_poll(eshkol_pollfd_t* fds, unsigned long count, int timeout) {
    return poll(fds, (nfds_t)count, timeout);
}
static int eshkol_socket_send(eshkol_socket_t socket_value, const void* data,
                              size_t length, int flags) {
    return (int)send(socket_value, data, length, flags);
}
static int eshkol_socket_recv(eshkol_socket_t socket_value, void* data,
                              size_t length, int flags) {
    return (int)recv(socket_value, data, length, flags);
}
#endif

/* Socket writes are allowed to complete partially on every supported OS.
 * Framed protocols cannot tolerate silent truncation, so HTTP and WebSocket
 * writers use one exact-length primitive. */
static int eshkol_socket_send_all(eshkol_socket_t socket_value,
                                  const void* data, size_t length) {
    const unsigned char* bytes = (const unsigned char*)data;
    size_t offset = 0;
    while (offset < length) {
        int sent = eshkol_socket_send(socket_value, bytes + offset,
                                      length - offset, 0);
        if (sent <= 0) return -1;
        offset += (size_t)sent;
    }
    return 0;
}

/*
 * Cryptographically strong random fill for the WebSocket client masking
 * key. RFC 6455 section 5.3 requires that the mask be unpredictable per frame
 * to prevent intermediate-cache poisoning. Prior versions of this file
 * shipped an all-zero mask, which is RFC-non-compliant.
 *
 * Source preference: arc4random_buf (macOS / BSDs / glibc >= 2.36) is
 * the best portable option. Fall back to /dev/urandom if the system
 * does not provide arc4random_buf.
 */
#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#  include <stdlib.h>  /* arc4random_buf */
#  define ESHKOL_WS_HAS_ARC4RANDOM 1
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 36))
#  include <stdlib.h>
#  define ESHKOL_WS_HAS_ARC4RANDOM 1
#else
#  define ESHKOL_WS_HAS_ARC4RANDOM 0
#endif

/**
 * @brief Fills a 4-byte WebSocket frame masking key with unpredictable bytes.
 *
 * RFC 6455 section 5.3 requires every client-to-server frame to be masked
 * with a key that is not predictable to an attacker. Prefers
 * arc4random_buf() where available, falls back to reading from
 * /dev/urandom, and as a last resort (both unavailable) uses a
 * time-seeded linear congruential generator that is not
 * cryptographically strong but avoids the RFC-violating all-zero mask.
 *
 * @param out Buffer of 4 bytes to receive the mask.
 */
static int eshkol_ws_random_mask(unsigned char out[4]) {
#ifdef _WIN32
    return BCryptGenRandom(NULL, out, 4, BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#elif ESHKOL_WS_HAS_ARC4RANDOM
    arc4random_buf(out, 4);
    return 1;
#else
    FILE* urnd = fopen("/dev/urandom", "rb");
    if (urnd) {
        size_t got = fread(out, 1, 4, urnd);
        fclose(urnd);
        if (got == 4) return 1;
    }
    return 0;
#endif
}

/*******************************************************************************
 * HTTP Server — Minimal, single-connection, for OAuth callbacks
 ******************************************************************************/

#define MAX_HTTP_SERVERS 4

typedef struct {
    eshkol_socket_t listen_fd;
    eshkol_socket_t client_fd;
    uint16_t port;
} HttpServer;

static HttpServer* g_servers[MAX_HTTP_SERVERS] = {0};

static int ascii_case_equal_n(const char* a, const char* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        unsigned char ca = (unsigned char)a[i];
        unsigned char cb = (unsigned char)b[i];
        if (ca >= 'A' && ca <= 'Z') ca = (unsigned char)(ca - 'A' + 'a');
        if (cb >= 'A' && cb <= 'Z') cb = (unsigned char)(cb - 'A' + 'a');
        if (ca != cb) return 0;
    }
    return 1;
}

static int64_t http_content_length(const char* headers, size_t header_bytes) {
    const char* cursor = headers;
    const char* end = headers + header_bytes;
    while (cursor < end) {
        const char* line_end = strstr(cursor, "\r\n");
        if (!line_end || line_end > end) line_end = end;
        static const char name[] = "content-length:";
        size_t line_len = (size_t)(line_end - cursor);
        if (line_len >= sizeof(name) - 1 &&
            ascii_case_equal_n(cursor, name, sizeof(name) - 1)) {
            const char* value = cursor + sizeof(name) - 1;
            while (value < line_end && (*value == ' ' || *value == '\t')) ++value;
            uint64_t parsed = 0;
            if (value == line_end) return -1;
            while (value < line_end) {
                if (*value < '0' || *value > '9') return -1;
                uint64_t digit = (uint64_t)(*value++ - '0');
                if (parsed > (UINT64_MAX - digit) / 10) return -1;
                parsed = parsed * 10 + digit;
            }
            return parsed <= INT64_MAX ? (int64_t)parsed : -1;
        }
        if (line_end == end) break;
        cursor = line_end + 2;
    }
    return 0;
}

/*
 * Create HTTP server and bind to port.
 * port: TCP port (0 = random available port)
 * Returns: handle (>= 1), -1 error
 */
/**
 * @brief Allocates a server slot, creates a loopback-only TCP listening socket, and binds/listens on @p port.
 *
 * Binds to 127.0.0.1 (INADDR_LOOPBACK) only, never all interfaces, since
 * this server exists for local OAuth PKCE callbacks and similar short-lived
 * local IPC, not for serving remote clients. If @p port is 0, the OS
 * assigns an ephemeral port, which can be retrieved with
 * eshkol_http_server_port().
 *
 * @param port TCP port to bind, or 0 to let the OS choose one.
 * @return Server handle (>= 1) on success, -1 on socket/bind/listen failure or if all server slots are in use.
 */
int64_t eshkol_http_server_create(int32_t port) {
    if (!ensure_sockets()) return -1;
    int slot = -1;
    for (int i = 1; i < MAX_HTTP_SERVERS; i++) {
        if (!g_servers[i]) { slot = i; break; }
    }
    if (slot < 0) return -1;

    eshkol_socket_t fd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (fd == ESHKOL_INVALID_SOCKET) return -1;

    int opt = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, (const char*)&opt, sizeof(opt));

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);  /* Bind to localhost only */
    addr.sin_port = htons((uint16_t)port);

    if (bind(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        eshkol_socket_close(fd);
        return -1;
    }

    if (listen(fd, 1) < 0) {
        eshkol_socket_close(fd);
        return -1;
    }

    /* Get actual port (if 0 was requested) */
#ifdef _WIN32
    int addrlen = sizeof(addr);
#else
    socklen_t addrlen = sizeof(addr);
#endif
    getsockname(fd, (struct sockaddr*)&addr, &addrlen);

    HttpServer* srv = (HttpServer*)calloc(1, sizeof(HttpServer));
    if (!srv) { eshkol_socket_close(fd); return -1; }
    srv->listen_fd = fd;
    srv->client_fd = ESHKOL_INVALID_SOCKET;
    srv->port = ntohs(addr.sin_port);
    g_servers[slot] = srv;
    return (int64_t)slot;
}

/*
 * Get actual port number.
 */
/**
 * @brief Returns the TCP port a server handle is actually bound to.
 *
 * Useful when eshkol_http_server_create() was called with port 0 and the
 * OS assigned an ephemeral port.
 *
 * @param handle Server handle from eshkol_http_server_create().
 * @return The bound port number, or -1 if @p handle is invalid.
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
/**
 * @brief Blocks up to @p timeout_ms waiting for one client connection, then reads its HTTP request into @p buf.
 *
 * Accepts a single connection on the listening socket, sets a 5-second
 * receive timeout on the accepted client socket, and reads until either
 * @p buf fills or the "\r\n\r\n" end-of-headers marker is seen (the request
 * body, if any, is not read separately here). The accepted client fd is
 * stashed on the server so a subsequent eshkol_http_server_respond() can
 * reply on the same connection.
 *
 * @param handle Server handle from eshkol_http_server_create().
 * @param buf Destination buffer for the raw request text (method/path, headers, and any body read so far).
 * @param buf_size Size of @p buf, including space for the terminating NUL.
 * @param timeout_ms How long to wait for an incoming connection before giving up.
 * @return Number of bytes written to @p buf on success, 0 on timeout, -1 on error.
 */
int32_t eshkol_http_server_accept(int64_t handle, char* buf, int32_t buf_size,
                                    int32_t timeout_ms) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return -1;
    HttpServer* srv = g_servers[handle];
    if (!buf || buf_size <= 0) return -1;

    /* Wait for connection */
    eshkol_pollfd_t pfd = { .fd = srv->listen_fd, .events = POLLIN };
    int ret = eshkol_socket_poll(&pfd, 1, timeout_ms);
    if (ret <= 0) return ret == 0 ? 0 : -1;

    struct sockaddr_in client_addr;
#ifdef _WIN32
    int client_len = sizeof(client_addr);
#else
    socklen_t client_len = sizeof(client_addr);
#endif
    eshkol_socket_t client_fd = accept(srv->listen_fd, (struct sockaddr*)&client_addr, &client_len);
    if (client_fd == ESHKOL_INVALID_SOCKET) return -1;
    srv->client_fd = client_fd;

    /* Set receive timeout */
#ifdef _WIN32
    DWORD receive_timeout = 5000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
               (const char*)&receive_timeout, sizeof(receive_timeout));
#else
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    /* Read the complete request headers and the declared body.  Stopping at
     * CRLFCRLF loses POST bodies whenever TCP splits the body into a later
     * packet, which is common with WinHTTP and under load. */
    int32_t total = 0;
    size_t required_total = 0;
    while (total < buf_size - 1) {
        int n = eshkol_socket_recv(client_fd, buf + total,
                                   (size_t)(buf_size - 1 - total), 0);
        if (n <= 0) break;
        total += (int32_t)n;
        buf[total] = '\0';
        if (required_total == 0 && total >= 4) {
            char* header_end = strstr(buf, "\r\n\r\n");
            if (header_end) {
                size_t header_bytes = (size_t)(header_end - buf) + 4;
                int64_t content_length = http_content_length(buf, header_bytes);
                if (content_length < 0 ||
                    (uint64_t)content_length > (uint64_t)(buf_size - 1) - header_bytes) {
                    eshkol_socket_close(client_fd);
                    srv->client_fd = ESHKOL_INVALID_SOCKET;
                    return -1;
                }
                required_total = header_bytes + (size_t)content_length;
            }
        }
        if (required_total > 0 && (size_t)total >= required_total) break;
    }
    buf[total] = '\0';

    return total;
}

/*
 * Send HTTP response on the current connection.
 */
/**
 * @brief Writes a complete HTTP/1.1 response (status line, Content-Type/Length headers, body) to the currently connected client and closes the connection.
 *
 * Maps a handful of common status codes (301, 400, 404, 500) to their
 * reason phrases, defaulting to "OK" otherwise. Always sends
 * "Connection: close" since this server is single-shot per client. Does
 * nothing if there is no client currently connected on @p handle.
 *
 * @param handle Server handle from eshkol_http_server_create().
 * @param status HTTP status code to send.
 * @param content_type Value for the Content-Type header, or "text/html" if NULL.
 * @param body Response body, or NULL/empty for no body.
 */
void eshkol_http_server_respond(int64_t handle, int32_t status,
                                  const char* content_type, const char* body) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return;
    HttpServer* srv = g_servers[handle];
    if (srv->client_fd == ESHKOL_INVALID_SOCKET) return;

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

    if (hlen <= 0 || (size_t)hlen >= sizeof(header) ||
        eshkol_socket_send_all(srv->client_fd, header, (size_t)hlen) != 0) {
        eshkol_socket_close(srv->client_fd);
        srv->client_fd = ESHKOL_INVALID_SOCKET;
        return;
    }
    if (body && body_len > 0) {
        (void)eshkol_socket_send_all(srv->client_fd, body, (size_t)body_len);
    }
    eshkol_socket_close(srv->client_fd);
    srv->client_fd = ESHKOL_INVALID_SOCKET;
}

/*
 * Close and free HTTP server.
 */
/**
 * @brief Closes any open client/listening sockets for @p handle, frees the server, and clears its slot.
 *
 * @param handle Server handle from eshkol_http_server_create().
 */
void eshkol_http_server_close(int64_t handle) {
    if (handle < 1 || handle >= MAX_HTTP_SERVERS || !g_servers[handle]) return;
    HttpServer* srv = g_servers[handle];
    if (srv->client_fd != ESHKOL_INVALID_SOCKET) eshkol_socket_close(srv->client_fd);
    eshkol_socket_close(srv->listen_fd);
    free(srv);
    g_servers[handle] = NULL;
}

/*******************************************************************************
 * Unix Domain Socket Connect (for IDE IPC: VS Code VSCODE_IPC_HOOK)
 ******************************************************************************/

/**
 * @brief Opens a Unix domain socket and connects it to @p path.
 *
 * Used for local IDE IPC (e.g. VS Code's VSCODE_IPC_HOOK socket). The
 * connected fd is returned directly rather than through a handle table,
 * since callers manage it like any other raw socket fd.
 *
 * @param path Filesystem path of the Unix domain socket to connect to.
 * @return Connected file descriptor (>= 0) on success, -1 on error or if @p path is NULL.
 */
int64_t eshkol_unix_socket_connect(const char* path) {
    if (!path || !ensure_sockets()) return -1;
    eshkol_socket_t fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd == ESHKOL_INVALID_SOCKET) return -1;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, path, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        eshkol_socket_close(fd);
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
    eshkol_socket_t fd;
    int closed;
} WsConnection;

static WsConnection* g_ws[MAX_WS_HANDLES] = {0};

/* Minimal WebSocket: uses existing TCP socket after HTTP upgrade.
 * Full handshake requires base64(SHA1(key + magic)) which needs
 * either OpenSSL or CommonCrypto. For now, provide the fd-level
 * operations and let the Eshkol wrapper handle the upgrade via
 * the existing HTTP client. */

/**
 * @brief Wraps an already-connected, already-upgraded TCP socket fd in a WebSocket handle for use with the eshkol_ws_* frame functions.
 *
 * Does not perform the HTTP Upgrade handshake itself — the caller is
 * expected to complete the WebSocket upgrade (e.g. via the Eshkol HTTP
 * client) before calling this, and pass in the resulting fd.
 *
 * @param fd Already-connected, already-upgraded socket file descriptor.
 * @return WebSocket handle (>= 1) on success, -1 if all handle slots are in use.
 */
int64_t eshkol_ws_wrap_fd(int64_t fd) {
    for (int i = 1; i < MAX_WS_HANDLES; i++) {
        if (!g_ws[i]) {
            WsConnection* ws = (WsConnection*)calloc(1, sizeof(WsConnection));
            if (!ws) return -1;
            ws->fd = (eshkol_socket_t)(uintptr_t)fd;
            ws->closed = 0;
            g_ws[i] = ws;
            return (int64_t)i;
        }
    }
    return -1;
}

/* Send WebSocket text frame (opcode 0x81, masked) */
/**
 * @brief Encodes and sends @p data as a single masked WebSocket text frame (opcode 0x1).
 *
 * Builds an RFC 6455 frame header with the appropriate extended-length
 * encoding for @p len (7-bit, 16-bit, or 64-bit length), generates a fresh
 * random mask per eshkol_ws_random_mask(), and masks the payload in
 * place — using a 4096-byte stack buffer directly when the payload fits,
 * otherwise masking and sending it in 4096-byte chunks to avoid a heap
 * allocation.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 * @param data Text payload bytes to send.
 * @param len Length of @p data in bytes.
 * @return 0 on success, -1 on error, invalid handle, closed connection, or a short send().
 */
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

    /* Masking key: cryptographically random per RFC 6455 section 5.3. */
    unsigned char mask[4];
    if (!eshkol_ws_random_mask(mask)) return -1;
    memcpy(header + hlen, mask, 4);
    hlen += 4;

    if (eshkol_socket_send_all(ws->fd, header, (size_t)hlen) != 0) return -1;
    /* Mask the payload in-place into a stack buffer when it fits, else
     * stream-mask in chunks. Avoids a heap allocation for small frames. */
    if (len > 0) {
        if (len <= 4096) {
            unsigned char buf[4096];
            for (int i = 0; i < len; i++) {
                buf[i] = (unsigned char)data[i] ^ mask[i & 3];
            }
            if (eshkol_socket_send_all(ws->fd, buf, (size_t)len) != 0) return -1;
        } else {
            unsigned char buf[4096];
            int sent = 0;
            while (sent < len) {
                int chunk = len - sent < 4096 ? len - sent : 4096;
                for (int i = 0; i < chunk; i++) {
                    buf[i] = (unsigned char)data[sent + i] ^ mask[(sent + i) & 3];
                }
                if (eshkol_socket_send_all(ws->fd, buf, (size_t)chunk) != 0) return -1;
                sent += chunk;
            }
        }
    }
    return 0;
}

/* Send WebSocket binary frame (opcode 0x82) */
/**
 * @brief Encodes and sends @p data as a single masked WebSocket binary frame (opcode 0x2).
 *
 * Identical framing/masking logic to eshkol_ws_send_text(), differing
 * only in the opcode byte.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 * @param data Binary payload bytes to send.
 * @param len Length of @p data in bytes.
 * @return 0 on success, -1 on error, invalid handle, closed connection, or a short send().
 */
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
    /* Masking key: cryptographically random per RFC 6455 section 5.3. */
    unsigned char mask[4];
    if (!eshkol_ws_random_mask(mask)) return -1;
    memcpy(header + hlen, mask, 4);
    hlen += 4;

    if (eshkol_socket_send_all(ws->fd, header, (size_t)hlen) != 0) return -1;
    if (len > 0) {
        if (len <= 4096) {
            unsigned char buf[4096];
            for (int i = 0; i < len; i++) {
                buf[i] = (unsigned char)data[i] ^ mask[i & 3];
            }
            if (eshkol_socket_send_all(ws->fd, buf, (size_t)len) != 0) return -1;
        } else {
            unsigned char buf[4096];
            int sent = 0;
            while (sent < len) {
                int chunk = len - sent < 4096 ? len - sent : 4096;
                for (int i = 0; i < chunk; i++) {
                    buf[i] = (unsigned char)data[sent + i] ^ mask[(sent + i) & 3];
                }
                if (eshkol_socket_send_all(ws->fd, buf, (size_t)chunk) != 0) return -1;
                sent += chunk;
            }
        }
    }
    return 0;
}

/* Receive WebSocket frame */
/**
 * @brief Waits up to @p timeout_ms for a WebSocket frame, decodes and unmasks it into @p buf, and reports its opcode via @p frame_type.
 *
 * Reads the 2-byte base header plus any RFC 6455 extended length field
 * (16-bit or 64-bit) and mask key, then reads and (if masked) unmasks the
 * payload, truncating to @p buf_size if the frame is larger than the
 * buffer. Close frames (opcode 0x8) mark the connection closed; ping
 * frames (opcode 0x9) are answered automatically with an empty pong
 * before returning.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 * @param buf Destination buffer for the (unmasked) payload.
 * @param buf_size Size of @p buf, including space for the terminating NUL.
 * @param frame_type Output: 1 text, 2 binary, 8 close, 9 ping, 10 pong, 0 unknown opcode.
 * @param timeout_ms How long to wait for a frame before returning 0.
 * @return Number of payload bytes written to @p buf, 0 on timeout, -1 on error.
 */
int32_t eshkol_ws_receive(int64_t handle, char* buf, int32_t buf_size,
                            int32_t* frame_type, int32_t timeout_ms) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return -1;
    WsConnection* ws = g_ws[handle];
    if (ws->closed || !buf || buf_size <= 0 || !frame_type) return -1;

    /* Poll for data */
    eshkol_pollfd_t pfd = { .fd = ws->fd, .events = POLLIN };
    int ret = eshkol_socket_poll(&pfd, 1, timeout_ms);
    if (ret <= 0) return ret == 0 ? 0 : -1;

    /* Read frame header (2 bytes minimum) */
    unsigned char hdr[2];
    if (eshkol_socket_recv(ws->fd, hdr, 2, MSG_WAITALL) != 2) { ws->closed = 1; return -1; }

    int opcode = hdr[0] & 0x0F;
    int masked = (hdr[1] & 0x80) != 0;
    uint64_t payload_len = hdr[1] & 0x7F;

    if (payload_len == 126) {
        unsigned char ext[2];
        if (eshkol_socket_recv(ws->fd, ext, 2, MSG_WAITALL) != 2) return -1;
        payload_len = ((uint64_t)ext[0] << 8) | ext[1];
    } else if (payload_len == 127) {
        unsigned char ext[8];
        if (eshkol_socket_recv(ws->fd, ext, 8, MSG_WAITALL) != 8) return -1;
        payload_len = 0;
        for (int i = 0; i < 8; i++) payload_len = (payload_len << 8) | ext[i];
    }

    /* Read mask key if present */
    unsigned char mask[4] = {0};
    if (masked) {
        if (eshkol_socket_recv(ws->fd, mask, 4, MSG_WAITALL) != 4) return -1;
    }

    /* Read payload */
    int32_t to_read = (int32_t)(payload_len < (uint64_t)buf_size ? payload_len : (uint64_t)(buf_size - 1));
    int32_t total = 0;
    while (total < to_read) {
        int n = eshkol_socket_recv(ws->fd, buf + total, (size_t)(to_read - total), 0);
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
        (void)eshkol_socket_send_all(ws->fd, pong, 2);
    }

    return total;
}

/* Close WebSocket connection */
/**
 * @brief Sends a WebSocket close frame (if not already closed), closes the underlying socket, and frees @p handle.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 */
void eshkol_ws_close(int64_t handle) {
    if (handle < 1 || handle >= MAX_WS_HANDLES || !g_ws[handle]) return;
    WsConnection* ws = g_ws[handle];
    if (!ws->closed) {
        /* Send close frame */
        unsigned char close_frame[6] = { 0x88, 0x80, 0, 0, 0, 0 };
        (void)eshkol_socket_send_all(ws->fd, close_frame, 6);
        ws->closed = 1;
    }
    eshkol_socket_close(ws->fd);
    free(ws);
    g_ws[handle] = NULL;
}
