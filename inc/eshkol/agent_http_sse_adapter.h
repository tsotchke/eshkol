/**
 * @file agent_http_sse_adapter.h
 * @brief Accepted-HTTP-connection ownership adapter for server-side SSE.
 *
 * Transfers one accepted loopback HTTP/1.1 client connection from
 * agent_http_server.c onto an explicit-loop resource lease from
 * inc/eshkol/core/resource_lease.h. The HTTP server keeps its listener and
 * complete-response path for subsequent connections; the adapter owns
 * write-watch registration/removal on the caller's loop and single-close
 * teardown via the shared cleanup-ticket contract.
 */

#ifndef ESHKOL_AGENT_HTTP_SSE_ADAPTER_H
#define ESHKOL_AGENT_HTTP_SSE_ADAPTER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Maximum encoded SSE chunk size in bytes, including HTTP/1.1 chunk overhead. */
#define ESHKOL_SSE_MAX_CHUNK_BYTES ((size_t)(16u * 1024u))

/** Monotonic per-frame write budget in milliseconds (held across retries). */
#define ESHKOL_SSE_WRITE_BUDGET_MS 5000

/** Idle timeout between frames in milliseconds. */
#define ESHKOL_SSE_IDLE_TIMEOUT_MS 60000

/** Resource-lease socket tag reserved for adopted HTTP SSE connections. */
#define ESHKOL_HTTP_SSE_LEASE_TAG 14u
#define ESHKOL_HTTP_SSE_LEASE_MASK (UINT32_C(1) << ESHKOL_HTTP_SSE_LEASE_TAG)

/** Nonblocking send returned EAGAIN/EWOULDBLOCK or partial progress; write watch armed. */
#define ESHKOL_SSE_WOULD_BLOCK 1

typedef struct eshkol_http_sse_adapter eshkol_http_sse_adapter_t;

#ifdef ESHKOL_SSE_ADAPTER_TESTING
typedef struct {
    int64_t (*send_override)(int fd, const uint8_t* data, size_t len, void* ctx);
    int (*watch_remove_override)(int64_t loop_handle, int fd, void* ctx);
    void* ctx;
} eshkol_http_sse_adapter_hooks_t;
#endif

int eshkol_http_sse_adapter_adopt(int64_t server_handle,
                                  int64_t loop_handle,
                                  int64_t* out_lease_id,
                                  eshkol_http_sse_adapter_t** out_adapter);

int eshkol_http_sse_adapter_check_chunk_size(const uint8_t* chunk, size_t chunk_len);

int eshkol_http_sse_adapter_arm_write_watch(eshkol_http_sse_adapter_t* adapter,
                                            uint64_t cookie);
int eshkol_http_sse_adapter_disarm_write_watch(eshkol_http_sse_adapter_t* adapter);

int eshkol_http_sse_adapter_write_step(int64_t lease_id,
                                       eshkol_http_sse_adapter_t* adapter,
                                       const uint8_t* bytes,
                                       size_t len,
                                       size_t* offset_inout,
                                       int64_t deadline_ms,
                                       uint64_t watch_cookie,
                                       size_t* out_written);

int64_t eshkol_http_sse_adapter_request_close(int64_t lease_id);
int eshkol_http_sse_adapter_cancel(int64_t lease_id);
int eshkol_http_sse_adapter_fd(const eshkol_http_sse_adapter_t* adapter);
int eshkol_http_sse_adapter_watch_armed(const eshkol_http_sse_adapter_t* adapter);

#ifdef ESHKOL_SSE_ADAPTER_TESTING
void eshkol_http_sse_adapter_set_hooks(eshkol_http_sse_adapter_t* adapter,
                                       const eshkol_http_sse_adapter_hooks_t* hooks);
int eshkol_http_sse_adapter_close_count(const eshkol_http_sse_adapter_t* adapter);
#endif

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_AGENT_HTTP_SSE_ADAPTER_H */
