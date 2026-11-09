#ifndef ESHKOL_CORE_NET_H
#define ESHKOL_CORE_NET_H
#include <stdint.h>
#include "eshkol/core/resource_lease.h"
#ifdef __cplusplus
extern "C" {
#endif
/** Numeric IPv4/IPv6 sockets owned through the shared resource lease layer
 * (eshkol/core/resource_lease.h): SOCKET-kind registry handles tagged
 * listener, stream or datagram. Results use the ESHKOL_IO_* outcome codes:
 * -1 system, -2 timeout/would-block, -3 unsupported platform, -4 invalid
 * argument or stale/wrong-kind handle, -5 busy (another operation holds the
 * handle), -6 cancelled (or closed during the operation), -7 peer reset,
 * -8 datagram truncated, -9 connection refused. Positive handles are exact and
 * never reused. Each operation waits on the calling thread's event loop under
 * one monotonic deadline. No DNS lookup is performed.
 * Kind 1 creates a listener, 2 connects a stream, 3 binds a datagram socket. */
int64_t eshkol_net_open(const char *address, int32_t port, int32_t kind, int32_t timeout_ms);
/** Accept a TCP connection within a bounded wait. */
int64_t eshkol_net_accept(int64_t handle, int32_t timeout_ms);
/** Read/write bytes; TCP read returning zero means EOF. At most 1 MiB per call. */
int32_t eshkol_net_read(int64_t handle, void *buffer, int32_t capacity, int32_t timeout_ms);
/** Send stream bytes, returning a potentially partial byte count. */
int32_t eshkol_net_write(int64_t handle, const void *buffer, int32_t length, int32_t timeout_ms);
/** Read into bytes [start, start + count) of an Eshkol bytevector; count >= 1.
 * Returns the byte count (zero = EOF). */
int32_t eshkol_net_read_bytes(int64_t handle, void *bytevector, int64_t start, int64_t count,
                              int32_t timeout_ms);
/** Write bytes [start, start + count) of a bytevector; returns a possibly partial count. */
int32_t eshkol_net_write_bytes(int64_t handle, const void *bytevector, int64_t start,
                               int64_t count, int32_t timeout_ms);
/** Write the whole range under one deadline; returns count or an outcome code. */
int64_t eshkol_net_write_all_bytes(int64_t handle, const void *bytevector, int64_t start,
                                   int64_t count, int32_t timeout_ms);
/** Send one datagram to a numeric address. */
int32_t eshkol_net_sendto(int64_t handle, const char *address, int32_t port,
                        const void *buffer, int32_t length, int32_t timeout_ms);
int32_t eshkol_net_sendto_bytes(int64_t handle, const char *address, int32_t port,
                                const void *bytevector, int64_t start, int64_t count,
                                int32_t timeout_ms);
/** Receive one datagram; peer is rendered as numeric address followed by :port,
 * with brackets for IPv6. A datagram larger than the capacity is consumed, its
 * prefix and sender are stored, and -8 (truncated) is returned. */
int32_t eshkol_net_recvfrom(int64_t handle, void *buffer, int32_t capacity,
                          char *peer, int32_t peer_capacity, int32_t timeout_ms);
int32_t eshkol_net_recvfrom_bytes(int64_t handle, void *bytevector, int64_t start,
                                  int64_t count, char *peer, int32_t peer_capacity,
                                  int32_t timeout_ms);
/** Query local (peer=0) or remote (peer=1) endpoint, numeric address only. */
int32_t eshkol_net_address(int64_t handle, int32_t peer, char *buffer, int32_t capacity);
/** Query the local or peer port. */
int32_t eshkol_net_port(int64_t handle, int32_t peer);
/** Set user nonblocking mode; waits then return -2 without blocking. */
int32_t eshkol_net_nonblocking(int64_t handle, int32_t enabled);
/** Shut down a stream's read (0), write (1) or both (2) directions. */
int32_t eshkol_net_shutdown(int64_t handle, int32_t how);
/** Cancel the operation waiting on handle, or the next one to wait. Any thread. */
int32_t eshkol_net_cancel(int64_t handle);
/** Close once: family 1 accepts TCP listeners and streams, 2 UDP sockets,
 * 0 any socket. A close during another thread's operation cancels it and
 * closes the descriptor when that operation ends. Stale, double-closed and
 * wrong-family handles return -4. */
int32_t eshkol_net_close_kind(int64_t handle, int32_t family);
int32_t eshkol_net_close(int64_t handle);
/** For owners layered on a stream (TLS): hold an exclusive borrow of a TCP
 * stream for the owner's lifetime, and move bytes through that borrow under a
 * caller deadline. While borrowed the handle reports busy to other callers;
 * closing it cancels the owner's operation and closes once the owner releases. */
int32_t eshkol_net_borrow_stream(int64_t handle, eshkol_resource_lease_t *lease);
int32_t eshkol_net_leased_io(const eshkol_resource_lease_t *lease, void *buffer, int32_t length,
                             int64_t deadline, int32_t writing);
#ifdef __cplusplus
}
#endif
#endif
