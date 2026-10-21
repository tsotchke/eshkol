#ifndef ESHKOL_CORE_NET_H
#define ESHKOL_CORE_NET_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/** Numeric IPv4/IPv6 sockets. Positive handles are never reused. Results:
 * -1 system failure, -2 timeout/would-block, -3 unsupported platform,
 * -4 invalid argument/handle. Close invalidates immediately; concurrent
 * operations hold references until completion. No DNS lookup is performed.
 * Kind 1 creates a listener, 2 connects a stream, 3 binds a datagram socket. */
int64_t eshkol_net_open(const char *address, int32_t port, int32_t kind, int32_t timeout_ms);
/** Accept a TCP connection within a bounded wait. */
int64_t eshkol_net_accept(int64_t handle, int32_t timeout_ms);
/** Read/write bytes; TCP read returning zero means EOF. UDP zero is a datagram. */
int32_t eshkol_net_read(int64_t handle, void *buffer, int32_t capacity, int32_t timeout_ms);
/** Send stream bytes, returning a potentially partial byte count. */
int32_t eshkol_net_write(int64_t handle, const void *buffer, int32_t length, int32_t timeout_ms);
/** Send one datagram to a numeric address. */
int32_t eshkol_net_sendto(int64_t handle, const char *address, int32_t port,
                        const void *buffer, int32_t length, int32_t timeout_ms);
/** Receive datagram; peer is rendered as numeric address followed by :port,
 * with brackets for IPv6. Oversized datagrams refuse rather than truncate. */
int32_t eshkol_net_recvfrom(int64_t handle, void *buffer, int32_t capacity,
                          char *peer, int32_t peer_capacity, int32_t timeout_ms);
/** Query local (peer=0) or remote (peer=1) endpoint, numeric address only. */
int32_t eshkol_net_address(int64_t handle, int32_t peer, char *buffer, int32_t capacity);
/** Query the local or peer port. */
int32_t eshkol_net_port(int64_t handle, int32_t peer);
/** Set user nonblocking mode; waits then return -2 without blocking. */
int32_t eshkol_net_nonblocking(int64_t handle, int32_t enabled);
/** Close once. Stale and double-closed handles return -4. */
int32_t eshkol_net_close(int64_t handle);
#ifdef __cplusplus
}
#endif
#endif
