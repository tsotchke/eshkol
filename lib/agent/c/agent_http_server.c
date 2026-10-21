/*******************************************************************************
 * HTTP Server + WebSocket for Eshkol Agent (B.4)
 *
 * Provides: minimal HTTP/1.1 server (for OAuth PKCE callback, MCP auth),
 * WebSocket server upgrade and client framing (for voice streaming,
 * remote sessions, MCP WebSocket),
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
#include "eshkol/agent_capabilities.h"

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

static uint64_t ws_now_ms(void);

static int eshkol_socket_nonblocking(eshkol_socket_t socket_value) {
#ifdef _WIN32
    u_long enabled=1;
    return ioctlsocket(socket_value,FIONBIO,&enabled)==0 ? 0 : -1;
#else
    int flags=fcntl(socket_value,F_GETFL,0);
    return flags<0 || fcntl(socket_value,F_SETFL,flags|O_NONBLOCK)<0 ? -1 : 0;
#endif
}

static int eshkol_socket_send_all_until(eshkol_socket_t socket_value,
                                        const void* data, size_t length,
                                        uint64_t deadline_ms) {
    const unsigned char* bytes=(const unsigned char*)data;
    size_t offset=0;
    while(offset<length) {
        uint64_t now=ws_now_ms();
        if(now==UINT64_MAX || now>=deadline_ms)return -1;
        eshkol_pollfd_t pfd={.fd=socket_value,.events=POLLOUT};
        int ready=eshkol_socket_poll(&pfd,1,(int)(deadline_ms-now));
        if(ready==0)return -1;
        if(ready<0) {
#ifdef _WIN32
            if(WSAGetLastError()==WSAEINTR)continue;
#else
            if(errno==EINTR)continue;
#endif
            return -1;
        }
        int sent=eshkol_socket_send(socket_value,bytes+offset,length-offset,
#ifdef MSG_NOSIGNAL
                                    MSG_NOSIGNAL
#else
                                    0
#endif
                                    );
        if(sent>0) { offset+=(size_t)sent; continue; }
#ifdef _WIN32
        int error=WSAGetLastError();
        if(error==WSAEINTR || error==WSAEWOULDBLOCK)continue;
#else
        if(errno==EINTR || errno==EAGAIN || errno==EWOULDBLOCK)continue;
#endif
        return -1;
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

/* RFC 6455 handshake helpers. Kept local so this optional agent target does
 * not acquire a new crypto-library or platform-link dependency. */
static uint32_t ws_rol32(uint32_t value, unsigned count) {
    return (value << count) | (value >> (32 - count));
}

static void ws_sha1(const unsigned char* data, size_t length,
                    unsigned char digest[20]) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu,
                     0x10325476u, 0xC3D2E1F0u};
    size_t padded = ((length + 9 + 63) / 64) * 64;
    unsigned char block[64];
    for (size_t base = 0; base < padded; base += 64) {
        memset(block, 0, sizeof(block));
        for (size_t i = 0; i < 64 && base + i < length; ++i)
            block[i] = data[base + i];
        if (length >= base && length < base + 64) block[length - base] = 0x80;
        if (base + 64 == padded) {
            uint64_t bits = (uint64_t)length * 8;
            for (int i = 0; i < 8; ++i)
                block[56 + i] = (unsigned char)(bits >> (56 - 8 * i));
        }
        uint32_t w[80];
        for (int i = 0; i < 16; ++i)
            w[i] = ((uint32_t)block[i*4] << 24) |
                   ((uint32_t)block[i*4+1] << 16) |
                   ((uint32_t)block[i*4+2] << 8) | block[i*4+3];
        for (int i = 16; i < 80; ++i)
            w[i] = ws_rol32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
        uint32_t a=h[0], b=h[1], c=h[2], d=h[3], e=h[4];
        for (int i = 0; i < 80; ++i) {
            uint32_t f, k;
            if (i < 20) { f=(b&c)|((~b)&d); k=0x5A827999u; }
            else if (i < 40) { f=b^c^d; k=0x6ED9EBA1u; }
            else if (i < 60) { f=(b&c)|(b&d)|(c&d); k=0x8F1BBCDCu; }
            else { f=b^c^d; k=0xCA62C1D6u; }
            uint32_t t=ws_rol32(a,5)+f+e+k+w[i];
            e=d; d=c; c=ws_rol32(b,30); b=a; a=t;
        }
        h[0]+=a; h[1]+=b; h[2]+=c; h[3]+=d; h[4]+=e;
    }
    for (int i = 0; i < 5; ++i) {
        digest[i*4]=(unsigned char)(h[i]>>24);
        digest[i*4+1]=(unsigned char)(h[i]>>16);
        digest[i*4+2]=(unsigned char)(h[i]>>8);
        digest[i*4+3]=(unsigned char)h[i];
    }
}

static void ws_base64_20(const unsigned char in[20], char out[29]) {
    static const char alphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    size_t i=0, o=0;
    while (i < 20) {
        uint32_t a=in[i++], b=i<20?in[i++]:0, c=i<20?in[i++]:0;
        out[o++]=alphabet[a>>2]; out[o++]=alphabet[((a&3)<<4)|(b>>4)];
        out[o++]=(i-1 < 20)?alphabet[((b&15)<<2)|(c>>6)]:'=';
        out[o++]=(i < 20)?alphabet[c&63]:'=';
    }
    out[28]='\0';
}

/*******************************************************************************
 * HTTP Server — Minimal, single-connection, for OAuth callbacks
 ******************************************************************************/

#define MAX_HTTP_SERVERS 4

typedef struct {
    eshkol_socket_t listen_fd;
    eshkol_socket_t client_fd;
    uint16_t port;
    unsigned char pending[32768];
    size_t pending_len;
} HttpServer;

static HttpServer* g_servers[MAX_HTTP_SERVERS] = {0};
/* Three low bits hold the table slot; generations stop before encoded IDs
 * exceed 2^53-1 and become inexact on the Eshkol numeric FFI surface. */
static uint64_t g_http_server_generation[MAX_HTTP_SERVERS] = {0};
#define MAX_PUBLIC_HANDLE_GENERATION ((UINT64_C(1) << 50) - 1)

static int http_server_slot(int64_t handle) {
    if (handle<=0) return -1;
    uint64_t id=(uint64_t)handle, slot=id&7u, generation=id>>3;
    if (slot==0 || slot>=MAX_HTTP_SERVERS || generation==0 ||
        generation!=g_http_server_generation[slot] || !g_servers[slot]) return -1;
    return (int)slot;
}

static int64_t http_server_public_handle(int slot) {
    if (slot<=0 || slot>=MAX_HTTP_SERVERS ||
        g_http_server_generation[slot]>=MAX_PUBLIC_HANDLE_GENERATION) return -1;
    ++g_http_server_generation[slot];
    return (int64_t)((g_http_server_generation[slot]<<3)|(uint64_t)slot);
}

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
        if (!g_servers[i] && g_http_server_generation[i]<MAX_PUBLIC_HANDLE_GENERATION) { slot = i; break; }
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
    int64_t public_handle=http_server_public_handle(slot);
    if (public_handle<0) { g_servers[slot]=NULL; eshkol_socket_close(fd); free(srv); return -1; }
    return public_handle;
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
    int slot=http_server_slot(handle);
    if (slot<0) return -1;
    return (int32_t)g_servers[slot]->port;
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
    int slot=http_server_slot(handle);
    if (slot<0) return -1;
    HttpServer* srv = g_servers[slot];
    if (srv->client_fd != ESHKOL_INVALID_SOCKET) return -1;
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
    srv->pending_len = 0;
#ifdef SO_SNDBUF
    { int send_buffer=65536; setsockopt(client_fd,SOL_SOCKET,SO_SNDBUF,
                                        (const char*)&send_buffer,sizeof(send_buffer)); }
#endif

    /* Set receive timeout */
#ifdef _WIN32
    DWORD receive_timeout = 5000;
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO,
               (const char*)&receive_timeout, sizeof(receive_timeout));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO,
               (const char*)&receive_timeout, sizeof(receive_timeout));
#ifdef SO_NOSIGPIPE
    { int no_sigpipe=1; setsockopt(client_fd,SOL_SOCKET,SO_NOSIGPIPE,&no_sigpipe,sizeof(no_sigpipe)); }
#endif
#else
    struct timeval tv = { .tv_sec = 5, .tv_usec = 0 };
    setsockopt(client_fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(client_fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#ifdef SO_NOSIGPIPE
    { int no_sigpipe=1; setsockopt(client_fd,SOL_SOCKET,SO_NOSIGPIPE,&no_sigpipe,sizeof(no_sigpipe)); }
#endif
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

    if (required_total > 0 && (size_t)total > required_total) {
        size_t extra=(size_t)total-required_total;
        if (extra > sizeof(srv->pending)) {
            eshkol_socket_close(client_fd);
            srv->client_fd=ESHKOL_INVALID_SOCKET;
            return -1;
        }
        memcpy(srv->pending,buf+required_total,extra);
        srv->pending_len=extra;
        total=(int32_t)required_total;
        buf[total]='\0';
    }

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
    int slot=http_server_slot(handle);
    if (slot<0) return;
    HttpServer* srv = g_servers[slot];
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
    int slot=http_server_slot(handle);
    if (slot<0) return;
    HttpServer* srv = g_servers[slot];
    if (srv->client_fd != ESHKOL_INVALID_SOCKET) eshkol_socket_close(srv->client_fd);
    eshkol_socket_close(srv->listen_fd);
    free(srv);
    g_servers[slot] = NULL;
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
 * WebSocket server and client framing (RFC 6455)
 *
 * Minimal implementation for text/binary frames, control-frame ping/pong,
 * and close. It rejects fragmentation and extensions.
 *
 * For production use, consider linking libcurl 7.86+ or libwebsockets.
 * This implementation handles the common case.
 ******************************************************************************/

#define MAX_WS_HANDLES 8
#define MAX_WS_PAYLOAD_BYTES (1024u * 1024u)
#define MAX_WS_WRITE_MS 2000

typedef struct {
    eshkol_socket_t fd;
    int closed;
    int server_side;
    unsigned char pending[32768];
    size_t pending_len;
    size_t pending_offset;
} WsConnection;

static WsConnection* g_ws[MAX_WS_HANDLES] = {0};
static uint64_t g_ws_generation[MAX_WS_HANDLES] = {0};

static int ws_slot_for_handle(int64_t handle) {
    if (handle<=0) return -1;
    uint64_t id=(uint64_t)handle, slot=id&7u, generation=id>>3;
    if (slot==0 || slot>=MAX_WS_HANDLES || generation==0 ||
        generation!=g_ws_generation[slot] || !g_ws[slot]) return -1;
    return (int)slot;
}

static int64_t ws_public_handle(int slot) {
    if (slot<=0 || slot>=MAX_WS_HANDLES ||
        g_ws_generation[slot]>=MAX_PUBLIC_HANDLE_GENERATION) return -1;
    ++g_ws_generation[slot];
    return (int64_t)((g_ws_generation[slot]<<3)|(uint64_t)slot);
}

static int ws_contains_token(const char* value, size_t length, const char* token) {
    size_t token_len = strlen(token), start = 0;
    while (start < length) {
        size_t end = start;
        while (start < length && (value[start] == ' ' || value[start] == '\t' || value[start] == ',')) ++start;
        end = start;
        while (end < length && value[end] != ',') ++end;
        while (end > start && (value[end-1] == ' ' || value[end-1] == '\t')) --end;
        if (end - start == token_len && ascii_case_equal_n(value + start, token, token_len)) return 1;
        start = end;
        while (start < length && value[start] != ',') ++start;
        if (start < length) ++start;
    }
    return 0;
}

static int ws_key_valid(const char* key, size_t len) {
    if (len != 24 || key[22] != '=' || key[23] != '=') return 0;
    for (size_t i = 0; i < 22; ++i) {
        unsigned char c=(unsigned char)key[i];
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
              (c >= '0' && c <= '9') || c == '+' || c == '/')) return 0;
    }
    /* 16 bytes encode to 22 significant base64 characters with the low
     * four bits of the final sextet zero. */
    unsigned char c=(unsigned char)key[21];
    int v = c >= 'A' && c <= 'Z' ? c-'A' : c >= 'a' && c <= 'z' ? c-'a'+26 :
            c >= '0' && c <= '9' ? c-'0'+52 : c == '+' ? 62 : 63;
    return (v & 15) == 0;
}

static int ws_utf8_valid(const unsigned char* s, size_t n) {
    size_t i=0;
    while(i<n) {
        unsigned char c=s[i++];
        if(c<0x80)continue;
        if(c>=0xc2 && c<=0xdf) {
            if(i>=n || (s[i++]&0xc0)!=0x80)return 0;
        } else if(c>=0xe0 && c<=0xef) {
            if(i+1>=n)return 0;
            unsigned char a=s[i++], b=s[i++];
            if((a&0xc0)!=0x80 || (b&0xc0)!=0x80 ||
               (c==0xe0 && a<0xa0) || (c==0xed && a>=0xa0))return 0;
        } else if(c>=0xf0 && c<=0xf4) {
            if(i+2>=n)return 0;
            unsigned char a=s[i++], b=s[i++], d=s[i++];
            if((a&0xc0)!=0x80 || (b&0xc0)!=0x80 || (d&0xc0)!=0x80 ||
               (c==0xf0 && a<0x90) || (c==0xf4 && a>=0x90))return 0;
        } else return 0;
    }
    return 1;
}

static int ws_close_code_valid(unsigned code) {
    return (code>=1000 && code<=1014 && code!=1004 && code!=1005 && code!=1006) ||
           (code>=3000 && code<=4999);
}

static int ws_parse_upgrade(const char* request, size_t length,
                            char key[25], size_t* header_length) {
    if (!request || length < 4 || length > 32768) return 0;
    const char* end = NULL;
    for (size_t i=0; i+3<length; ++i)
        if (!memcmp(request+i, "\r\n\r\n", 4)) { end=request+i; break; }
    if (!end) return 0;
    size_t head_len=(size_t)(end-request)+4;
    const char* line_end=NULL;
    for (size_t i=0; i+1<head_len; ++i)
        if (request[i]=='\r' && request[i+1]=='\n') { line_end=request+i; break; }
    if (!line_end || (size_t)(line_end-request) < 14 ||
        memcmp(request,"GET ",4) ||
        memcmp(line_end-9," HTTP/1.1",9)) return 0;
    const char* target=request+4;
    const char* target_end=line_end-9;
    if (target>=target_end) return 0;
    for (const char* p=target;p<target_end;++p)
        if ((unsigned char)*p<=0x20 || (unsigned char)*p>=0x7f) return 0;
    int have_host=0, have_upgrade=0, have_connection=0, have_version=0,
        have_key=0, have_content_length=0;
    const char* cursor=line_end+2;
    while (cursor < end) {
        const char* e=cursor;
        while (e < end && !(e[0]=='\r' && e[1]=='\n')) ++e;
        if (e > end) return 0;
        const char* colon=memchr(cursor,':',(size_t)(e-cursor));
        if (!colon || colon==cursor) return 0;
        for (const char* p=cursor;p<colon;++p) {
            unsigned char c=(unsigned char)*p;
            if (!((c>='A'&&c<='Z')||(c>='a'&&c<='z')||
                  (c>='0'&&c<='9')||(c!=0&&strchr("!#$%&'*+-.^_`|~",c)))) return 0;
        }
        const char* value=colon+1;
        while (value<e && (*value==' ' || *value=='\t')) ++value;
        const char* value_end=e;
        while (value_end>value && (value_end[-1]==' ' || value_end[-1]=='\t')) --value_end;
        for (const char* p=value;p<value_end;++p) {
            unsigned char c=(unsigned char)*p;
            if ((c<0x20 && c!='\t') || c==0x7f) return 0;
        }
        size_t name_len=(size_t)(colon-cursor), value_len=(size_t)(value_end-value);
        if (name_len==4 && ascii_case_equal_n(cursor,"Host",4)) {
            if (have_host++ || value_len==0) return 0;
        } else if (name_len==7 && ascii_case_equal_n(cursor,"Upgrade",7)) {
            if (have_upgrade++ || value_len!=9 || !ascii_case_equal_n(value,"websocket",9)) return 0;
        } else if (name_len==10 && ascii_case_equal_n(cursor,"Connection",10)) {
            if (have_connection++ || !ws_contains_token(value,value_len,"Upgrade")) return 0;
        } else if (name_len==21 && ascii_case_equal_n(cursor,"Sec-WebSocket-Version",21)) {
            if (have_version++ || value_len!=2 || memcmp(value,"13",2)) return 0;
        } else if (name_len==17 && ascii_case_equal_n(cursor,"Sec-WebSocket-Key",17)) {
            if (have_key++ || !ws_key_valid(value,value_len)) return 0;
            memcpy(key,value,24); key[24]='\0';
        } else if (name_len==14 && ascii_case_equal_n(cursor,"Content-Length",14)) {
            if (have_content_length++ || value_len!=1 || value[0]!='0') return 0;
        } else if (name_len==17 && ascii_case_equal_n(cursor,"Transfer-Encoding",17)) {
            return 0;
        }
        cursor=e+2;
    }
    if (header_length) *header_length=head_len;
    return have_host==1 && have_upgrade==1 && have_connection==1 &&
           have_version==1 && have_key==1;
}

static int ws_upgrade_accept(const char* key, char accept[29]) {
    static const char guid[]="258EAFA5-E914-47DA-95CA-C5AB0DC85B11";
    unsigned char input[60], digest[20];
    memcpy(input,key,24); memcpy(input+24,guid,sizeof(guid)-1);
    ws_sha1(input,24+sizeof(guid)-1,digest);
    ws_base64_20(digest,accept);
    return 1;
}

static uint64_t ws_now_ms(void) {
#ifdef _WIN32
    return (uint64_t)GetTickCount64();
#else
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now)!=0) return UINT64_MAX;
    return (uint64_t)now.tv_sec*1000 + (uint64_t)now.tv_nsec/1000000;
#endif
}

static int ws_read_exact(WsConnection* ws, unsigned char* out, size_t length,
                         uint64_t deadline) {
    size_t done=0;
    while (done<length && ws->pending_offset<ws->pending_len) {
        size_t available=ws->pending_len-ws->pending_offset;
        size_t take=length-done<available?length-done:available;
        memcpy(out+done,ws->pending+ws->pending_offset,take);
        ws->pending_offset+=take; done+=take;
    }
    if (ws->pending_offset==ws->pending_len) ws->pending_offset=ws->pending_len=0;
    while (done<length) {
        uint64_t now=ws_now_ms();
        if (now==UINT64_MAX || now>=deadline) return done ? -1 : -2;
        int remaining=(int)(deadline-now);
        eshkol_pollfd_t pfd={.fd=ws->fd,.events=POLLIN};
        int ready=eshkol_socket_poll(&pfd,1,remaining);
        if (ready<=0) return ready==0 && !done ? -2 : -1;
        int n=eshkol_socket_recv(ws->fd,out+done,length-done,0);
        if (n<=0) return -1;
        done+=(size_t)n;
    }
    return 0;
}

static int ws_send_control(WsConnection* ws, unsigned opcode,
                           const unsigned char* payload, size_t length) {
    unsigned char header[6], mask[4]={0};
    size_t hlen=0;
    header[hlen++]=(unsigned char)(0x80 | opcode);
    header[hlen++]=(unsigned char)(length | (ws->server_side?0:0x80));
    if (!ws->server_side) {
        if (!eshkol_ws_random_mask(mask)) return -1;
        memcpy(header+hlen,mask,4); hlen+=4;
    }
    if (ws->server_side) {
        uint64_t now=ws_now_ms();
        return now==UINT64_MAX ||
               eshkol_socket_send_all_until(ws->fd,header,hlen,now+MAX_WS_WRITE_MS)!=0 ||
               (length && eshkol_socket_send_all_until(ws->fd,payload,length,
                                                        now+MAX_WS_WRITE_MS)!=0) ? -1 : 0;
    }
    if (eshkol_socket_send_all(ws->fd,header,hlen)!=0) return -1;
    unsigned char masked[125];
    for (size_t i=0;i<length;++i) masked[i]=payload[i]^mask[i&3];
    return length?eshkol_socket_send_all(ws->fd,masked,length):0;
}

/* Minimal WebSocket framing. Server upgrades use the accepted HTTP
 * connection above; client wrappers continue to accept an already-upgraded
 * fd and use client masking rules. */

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
        if (!g_ws[i] && g_ws_generation[i]<MAX_PUBLIC_HANDLE_GENERATION) {
            WsConnection* ws = (WsConnection*)calloc(1, sizeof(WsConnection));
            if (!ws) return -1;
            ws->fd = (eshkol_socket_t)(uintptr_t)fd;
            ws->closed = 0;
            ws->server_side = 0;
            g_ws[i] = ws;
            int64_t public_handle=ws_public_handle(i);
            if (public_handle<0) { g_ws[i]=NULL; free(ws); return -1; }
            return public_handle;
        }
    }
    return -1;
}

/* Validates the exact request returned by http-server-accept, emits the RFC
 * 6455 response, and transfers the accepted socket to a WebSocket handle only
 * after the complete 101 response has been written. Bytes read beyond the
 * HTTP header are retained as the beginning of the first frame. */
int64_t eshkol_ws_server_upgrade(int64_t server_handle,
                                 const char* request, int32_t request_len) {
    int server_slot=http_server_slot(server_handle);
    if (server_slot<0 || request_len < 0) return -1;
    HttpServer* server=g_servers[server_slot];
    if (server->client_fd == ESHKOL_INVALID_SOCKET) return -1;
    char key[25], accept[29];
    size_t header_len=0;
    if (!ws_parse_upgrade(request,(size_t)request_len,key,&header_len)) {
        static const char bad[]="HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        if (eshkol_socket_nonblocking(server->client_fd)==0)
            (void)eshkol_socket_send_all_until(server->client_fd,bad,sizeof(bad)-1,
                                               ws_now_ms()+MAX_WS_WRITE_MS);
        eshkol_socket_close(server->client_fd);
        server->client_fd=ESHKOL_INVALID_SOCKET;
        return -1;
    }
    size_t request_pending_len=(size_t)request_len-header_len;
    size_t pending_len=server->pending_len+request_pending_len;
    if (pending_len > sizeof(((WsConnection*)0)->pending)) {
        static const char bad[]="HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
        if (eshkol_socket_nonblocking(server->client_fd)==0)
            (void)eshkol_socket_send_all_until(server->client_fd,bad,sizeof(bad)-1,
                                               ws_now_ms()+MAX_WS_WRITE_MS);
        eshkol_socket_close(server->client_fd);
        server->client_fd=ESHKOL_INVALID_SOCKET;
        return -1;
    }
    int slot=-1;
    for (int i=1;i<MAX_WS_HANDLES;++i)
        if (!g_ws[i] && g_ws_generation[i]<MAX_PUBLIC_HANDLE_GENERATION) { slot=i; break; }
    if (slot<0) return -1;
    WsConnection* ws=(WsConnection*)calloc(1,sizeof(*ws));
    if (!ws) return -1;
    ws_upgrade_accept(key,accept);
    char response[256];
    int response_len=snprintf(response,sizeof(response),
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\n"
        "Connection: Upgrade\r\nSec-WebSocket-Accept: %s\r\n\r\n",accept);
    if (response_len <= 0 || (size_t)response_len >= sizeof(response) ||
        eshkol_socket_nonblocking(server->client_fd)!=0 ||
        eshkol_socket_send_all_until(server->client_fd,response,(size_t)response_len,
                                     ws_now_ms()+MAX_WS_WRITE_MS)!=0) {
        free(ws);
        return -1;
    }
    ws->fd=server->client_fd;
    ws->server_side=1;
    if (server->pending_len) {
        memcpy(ws->pending,server->pending,server->pending_len);
        ws->pending_len=server->pending_len;
    }
    if (pending_len) {
        if (request_pending_len)
            memcpy(ws->pending+ws->pending_len,request+header_len,request_pending_len);
        ws->pending_len=pending_len;
    }
    g_ws[slot]=ws;
    int64_t public_handle=ws_public_handle(slot);
    if (public_handle<0) {
        g_ws[slot]=NULL;
        free(ws);
        return -1;
    }
    server->client_fd=ESHKOL_INVALID_SOCKET;
    server->pending_len=0;
    return public_handle;
}

/* Send WebSocket text frame (opcode 0x81, masked) */
/**
 * @brief Encodes and sends @p data as a single WebSocket text frame (opcode 0x1), masking only when this handle is a client.
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
    int slot=ws_slot_for_handle(handle);
    if (slot<0) return -1;
    WsConnection* ws = g_ws[slot];
    if (ws->closed || len < 0 || (uint32_t)len > MAX_WS_PAYLOAD_BYTES ||
        (len > 0 && !data) || (len>0 && !ws_utf8_valid((const unsigned char*)data,(size_t)len))) return -1;

    /* Frame header */
    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = 0x81;  /* FIN=1, opcode=text */

    /* Mask bit = 1 (client must mask), payload length */
    if (len < 126) {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | (unsigned char)len;
    } else if (len < 65536) {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | 126;
        header[hlen++] = (unsigned char)(len >> 8);
        header[hlen++] = (unsigned char)(len & 0xFF);
    } else {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | 127;
        uint64_t payload_len=(uint64_t)len;
        for (int i = 7; i >= 0; i--)
            header[hlen++] = (unsigned char)((payload_len >> (i * 8)) & 0xFF);
    }

    /* Masking key: cryptographically random per RFC 6455 section 5.3. */
    unsigned char mask[4] = {0};
    if (!ws->server_side) {
        if (!eshkol_ws_random_mask(mask)) return -1;
        memcpy(header + hlen, mask, 4);
        hlen += 4;
    }

    if (ws->server_side) {
        uint64_t now=ws_now_ms();
        if (now==UINT64_MAX || eshkol_socket_send_all_until(ws->fd,header,(size_t)hlen,now+MAX_WS_WRITE_MS)!=0 ||
            (len>0 && eshkol_socket_send_all_until(ws->fd,data,(size_t)len,now+MAX_WS_WRITE_MS)!=0)) {
            ws->closed=1; return -1;
        }
        return 0;
    }

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
 * @brief Encodes and sends @p data as a single WebSocket binary frame (opcode 0x2), masking only when this handle is a client.
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
    int slot=ws_slot_for_handle(handle);
    if (slot<0) return -1;
    WsConnection* ws = g_ws[slot];
    if (ws->closed || len < 0 || (uint32_t)len > MAX_WS_PAYLOAD_BYTES ||
        (len > 0 && !data)) return -1;

    unsigned char header[14];
    int hlen = 0;
    header[hlen++] = 0x82;  /* FIN=1, opcode=binary */
    if (len < 126) {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | (unsigned char)len;
    } else if (len < 65536) {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | 126;
        header[hlen++] = (unsigned char)(len >> 8);
        header[hlen++] = (unsigned char)(len & 0xFF);
    } else {
        header[hlen++] = (ws->server_side ? 0 : 0x80) | 127;
        uint64_t payload_len=(uint64_t)len;
        for (int i = 7; i >= 0; i--)
            header[hlen++] = (unsigned char)((payload_len >> (i * 8)) & 0xFF);
    }
    /* Masking key: cryptographically random per RFC 6455 section 5.3. */
    unsigned char mask[4] = {0};
    if (!ws->server_side) {
        if (!eshkol_ws_random_mask(mask)) return -1;
        memcpy(header + hlen, mask, 4);
        hlen += 4;
    }

    if (ws->server_side) {
        uint64_t now=ws_now_ms();
        if (now==UINT64_MAX || eshkol_socket_send_all_until(ws->fd,header,(size_t)hlen,now+MAX_WS_WRITE_MS)!=0 ||
            (len>0 && eshkol_socket_send_all_until(ws->fd,data,(size_t)len,now+MAX_WS_WRITE_MS)!=0)) {
            ws->closed=1; return -1;
        }
        return 0;
    }

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
 * Reads one complete unfragmented frame under a total @p timeout_ms deadline.
 * Server handles require masked input and client handles require unmasked
 * input. Oversized, truncated, malformed UTF-8 text, and unsupported frames
 * fail without reporting a partial payload. Ping payloads are echoed in pong.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 * @param buf Destination buffer for the (unmasked) payload.
 * @param buf_size Size of @p buf, including space for the terminating NUL.
 * @param frame_type Output: 1 text, 2 binary, 8 close, 9 ping, 10 pong, 0 unknown opcode.
 * @param timeout_ms How long to wait for a frame before returning 0.
 * @return Number of payload bytes written to @p buf (0 is a valid empty frame),
 *         0 on timeout only when @p frame_type is also 0, or -1 on error.
 */
int32_t eshkol_ws_receive(int64_t handle, char* buf, int32_t buf_size,
                            int32_t* frame_type, int32_t timeout_ms) {
    int slot=ws_slot_for_handle(handle);
    if (slot<0) return -1;
    WsConnection* ws = g_ws[slot];
    if (ws->closed || !buf || buf_size <= 0 || !frame_type) return -1;
    *frame_type=0;
    if (timeout_ms < 0) return -1;
    uint64_t now=ws_now_ms();
    if (now==UINT64_MAX || (uint64_t)timeout_ms>UINT64_MAX-now) { ws->closed=1; return -1; }
    uint64_t deadline=now+(uint64_t)timeout_ms;
    unsigned char hdr[2];
    int read_status=ws_read_exact(ws,hdr,sizeof(hdr),deadline);
    if (read_status==-2) return 0;
    if (read_status!=0) { ws->closed=1; return -1; }
    int fin=(hdr[0]&0x80)!=0, rsv=(hdr[0]&0x70)!=0;
    int opcode=hdr[0]&0x0f, masked=(hdr[1]&0x80)!=0;
    if (rsv || !fin || masked != ws->server_side) { ws->closed=1; return -1; }
    uint64_t payload_len=hdr[1]&0x7f;
    if (payload_len==126) {
        unsigned char ext[2];
        if (ws_read_exact(ws,ext,2,deadline)!=0) { ws->closed=1; return -1; }
        payload_len=((uint64_t)ext[0]<<8)|ext[1];
        if (payload_len<126) { ws->closed=1; return -1; }
    } else if (payload_len==127) {
        unsigned char ext[8];
        if (ws_read_exact(ws,ext,8,deadline)!=0 || (ext[0]&0x80)) { ws->closed=1; return -1; }
        payload_len=0;
        for (int i=0;i<8;++i) payload_len=(payload_len<<8)|ext[i];
        if (payload_len<65536) { ws->closed=1; return -1; }
    }
    if ((opcode&8) && payload_len>125) { ws->closed=1; return -1; }
    if (payload_len>MAX_WS_PAYLOAD_BYTES || payload_len>(uint64_t)(buf_size-1)) {
        ws->closed=1; return -1;
    }
    if (opcode!=1 && opcode!=2 && opcode!=8 && opcode!=9 && opcode!=10) {
        ws->closed=1; return -1;
    }
    unsigned char mask[4]={0};
    if (masked && ws_read_exact(ws,mask,4,deadline)!=0) { ws->closed=1; return -1; }
    size_t length=(size_t)payload_len;
    if (length && ws_read_exact(ws,(unsigned char*)buf,length,deadline)!=0) {
        ws->closed=1; return -1;
    }
    if (masked) for (size_t i=0;i<length;++i) buf[i]=(char)((unsigned char)buf[i]^mask[i&3]);
    buf[length]='\0';
    if ((opcode==1 && !ws_utf8_valid((unsigned char*)buf,length)) ||
        (opcode==8 && length>=2 &&
         (!ws_close_code_valid(((unsigned char)buf[0]<<8)|(unsigned char)buf[1]) ||
          !ws_utf8_valid((unsigned char*)buf+2,length-2)))) {
        ws->closed=1; return -1;
    }
    switch (opcode) {
        case 1: *frame_type=1; break;
        case 2: *frame_type=2; break;
        case 8:
            if (length==1) { ws->closed=1; return -1; }
            *frame_type=8;
            (void)ws_send_control(ws,8,(unsigned char*)buf,length);
            ws->closed=1;
            break;
        case 9: *frame_type=9; if (ws_send_control(ws,10,(unsigned char*)buf,length)!=0) { ws->closed=1; return -1; } break;
        case 10: *frame_type=10; break;
        default: ws->closed=1; return -1;
    }
    return (int32_t)length;
}

/* Close WebSocket connection */
/**
 * @brief Sends a WebSocket close frame (if not already closed), closes the underlying socket, and frees @p handle.
 *
 * @param handle WebSocket handle from eshkol_ws_wrap_fd().
 */
void eshkol_ws_close(int64_t handle) {
    int slot=ws_slot_for_handle(handle);
    if (slot<0) return;
    WsConnection* ws = g_ws[slot];
    if (!ws->closed) {
        (void)ws_send_control(ws,8,NULL,0);
        ws->closed = 1;
    }
    eshkol_socket_close(ws->fd);
    free(ws);
    g_ws[slot] = NULL;
}
