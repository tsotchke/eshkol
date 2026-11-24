/*
 * Native deterministic collectives over Eshkol's v1.4.0 socket layer.
 *
 * Uses eshkol_net_* TCP handles (backed by the shared resource lease registry
 * and event loop) with a framed little-endian binary wire protocol.
 * Every floating-point reduction folds rank contributions in strict ascending
 * rank order 0, 1, ..., world_size - 1 so all ranks receive bit-identical
 * IEEE-754 binary64 outputs regardless of network arrival order.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "eshkol/core/collectives.h"
#include "eshkol/core/net.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define ESHKOL_COLL_MAGIC      UINT32_C(0x45434F4C) /* "ECOL" */
#define ESHKOL_COLL_VERSION    UINT8_C(1)
#define ESHKOL_COLL_HDR_BYTES  24u
#define ESHKOL_COLL_IO_CHUNK   (256 * 1024)
#define ESHKOL_COLL_MAX_ELEMS  ((size_t)(64 * 1024 * 1024))

enum {
    COLL_MSG_HELLO          = 1,
    COLL_MSG_BARRIER        = 2,
    COLL_MSG_BROADCAST      = 3,
    COLL_MSG_ALLREDUCE      = 4,
    COLL_MSG_ALLGATHER      = 5,
    COLL_MSG_REDUCE_SCATTER = 6
};

struct eshkol_coll_comm {
    int32_t rank;
    int32_t world_size;
    int64_t listener_handle; /* rank 0 only */
    int64_t coord_stream;    /* rank > 0 only */
    int64_t *peer_streams;   /* rank 0 only: length world_size, index 0 unused */
};

static void put_u32_le(uint8_t *dst, uint32_t v) {
    dst[0] = (uint8_t)(v & 0xFFu);
    dst[1] = (uint8_t)((v >> 8) & 0xFFu);
    dst[2] = (uint8_t)((v >> 16) & 0xFFu);
    dst[3] = (uint8_t)((v >> 24) & 0xFFu);
}

static uint32_t get_u32_le(const uint8_t *src) {
    return (uint32_t)src[0] |
           ((uint32_t)src[1] << 8) |
           ((uint32_t)src[2] << 16) |
           ((uint32_t)src[3] << 24);
}

static void put_u64_le(uint8_t *dst, uint64_t v) {
    put_u32_le(dst, (uint32_t)(v & 0xFFFFFFFFu));
    put_u32_le(dst + 4, (uint32_t)(v >> 32));
}

static uint64_t get_u64_le(const uint8_t *src) {
    return (uint64_t)get_u32_le(src) | ((uint64_t)get_u32_le(src + 4) << 32);
}

static int32_t write_exact(int64_t handle, const void *buf, size_t len, int32_t timeout_ms) {
    const uint8_t *p = (const uint8_t *)buf;
    size_t sent = 0;
    while (sent < len) {
        size_t rem = len - sent;
        int32_t chunk = rem > ESHKOL_COLL_IO_CHUNK ? ESHKOL_COLL_IO_CHUNK : (int32_t)rem;
        int32_t n = eshkol_net_write(handle, p + sent, chunk, timeout_ms);
        if (n <= 0) return n < 0 ? n : ESHKOL_IO_RESET;
        sent += (size_t)n;
    }
    return 0;
}

static int32_t read_exact(int64_t handle, void *buf, size_t len, int32_t timeout_ms) {
    uint8_t *p = (uint8_t *)buf;
    size_t got = 0;
    while (got < len) {
        size_t rem = len - got;
        int32_t chunk = rem > ESHKOL_COLL_IO_CHUNK ? ESHKOL_COLL_IO_CHUNK : (int32_t)rem;
        int32_t n = eshkol_net_read(handle, p + got, chunk, timeout_ms);
        if (n <= 0) return n < 0 ? n : ESHKOL_IO_RESET;
        got += (size_t)n;
    }
    return 0;
}

static int32_t send_frame(int64_t handle, uint8_t msg_type, uint8_t op_tag,
                          int32_t rank, int32_t world_size,
                          const double *data, size_t count, int32_t timeout_ms) {
    uint8_t hdr[ESHKOL_COLL_HDR_BYTES];
    put_u32_le(hdr + 0, ESHKOL_COLL_MAGIC);
    hdr[4] = ESHKOL_COLL_VERSION;
    hdr[5] = msg_type;
    hdr[6] = op_tag;
    hdr[7] = 0;
    put_u32_le(hdr + 8, (uint32_t)rank);
    put_u32_le(hdr + 12, (uint32_t)world_size);
    put_u64_le(hdr + 16, (uint64_t)count);

    int32_t rc = write_exact(handle, hdr, sizeof(hdr), timeout_ms);
    if (rc != 0 || count == 0) return rc;

    /* Serialize IEEE-754 binary64 values explicitly as little-endian u64 bits. */
    uint8_t scratch[4096];
    size_t offset = 0;
    while (offset < count) {
        size_t batch = count - offset;
        if (batch > sizeof(scratch) / 8u) batch = sizeof(scratch) / 8u;
        for (size_t i = 0; i < batch; ++i) {
            uint64_t bits = 0;
            memcpy(&bits, &data[offset + i], sizeof(bits));
            put_u64_le(scratch + i * 8u, bits);
        }
        rc = write_exact(handle, scratch, batch * 8u, timeout_ms);
        if (rc != 0) return rc;
        offset += batch;
    }
    return 0;
}

static int32_t recv_frame(int64_t handle, uint8_t expected_msg, uint8_t *out_op,
                          int32_t *out_rank, int32_t expected_world,
                          double *data, size_t expected_count, int32_t timeout_ms) {
    uint8_t hdr[ESHKOL_COLL_HDR_BYTES];
    int32_t rc = read_exact(handle, hdr, sizeof(hdr), timeout_ms);
    if (rc != 0) return rc;

    uint32_t magic = get_u32_le(hdr + 0);
    uint8_t ver = hdr[4];
    uint8_t msg = hdr[5];
    uint8_t op = hdr[6];
    int32_t rank = (int32_t)get_u32_le(hdr + 8);
    int32_t world = (int32_t)get_u32_le(hdr + 12);
    uint64_t count = get_u64_le(hdr + 16);

    if (magic != ESHKOL_COLL_MAGIC || ver != ESHKOL_COLL_VERSION ||
        msg != expected_msg || world != expected_world ||
        count != (uint64_t)expected_count) {
        return ESHKOL_IO_INVALID;
    }
    if (out_op) *out_op = op;
    if (out_rank) *out_rank = rank;
    if (expected_count == 0) return 0;

    uint8_t scratch[4096];
    size_t offset = 0;
    while (offset < expected_count) {
        size_t batch = expected_count - offset;
        if (batch > sizeof(scratch) / 8u) batch = sizeof(scratch) / 8u;
        rc = read_exact(handle, scratch, batch * 8u, timeout_ms);
        if (rc != 0) return rc;
        for (size_t i = 0; i < batch; ++i) {
            uint64_t bits = get_u64_le(scratch + i * 8u);
            memcpy(&data[offset + i], &bits, sizeof(bits));
        }
        offset += batch;
    }
    return 0;
}

static double apply_reduce_op(double acc, double v, eshkol_coll_op_t op) {
    switch (op) {
        case ESHKOL_COLL_SUM:  return acc + v;
        case ESHKOL_COLL_PROD: return acc * v;
        case ESHKOL_COLL_MIN:
            if (isnan(acc)) return acc;
            if (isnan(v)) return v;
            return (v < acc) ? v : acc;
        case ESHKOL_COLL_MAX:
            if (isnan(acc)) return acc;
            if (isnan(v)) return v;
            return (v > acc) ? v : acc;
    }
    return acc;
}

int64_t eshkol_coll_listen_coord(const char *coord_host, int32_t coord_port,
                                 int32_t *out_bound_port) {
    if (!coord_host || coord_port < 0 || coord_port > 65535) return ESHKOL_IO_INVALID;
    int64_t listener = eshkol_net_open(coord_host, coord_port, 1, 0);
    if (listener < 0) return listener;
    if (out_bound_port) {
        int32_t p = eshkol_net_port(listener, 0);
        if (p <= 0) {
            eshkol_net_close_kind(listener, 1);
            return p < 0 ? p : ESHKOL_IO_SYSTEM;
        }
        *out_bound_port = p;
    }
    return listener;
}

eshkol_coll_comm_t *eshkol_coll_init(int32_t rank, int32_t world_size,
                                     const char *coord_host, int32_t coord_port,
                                     int64_t listener_handle, int32_t timeout_ms) {
    if (world_size <= 0 || rank < 0 || rank >= world_size || timeout_ms <= 0) {
        if (rank == 0 && listener_handle > 0) eshkol_net_close_kind(listener_handle, 1);
        return NULL;
    }
    eshkol_coll_comm_t *comm = (eshkol_coll_comm_t *)calloc(1, sizeof(*comm));
    if (!comm) {
        if (rank == 0 && listener_handle > 0) eshkol_net_close_kind(listener_handle, 1);
        return NULL;
    }
    comm->rank = rank;
    comm->world_size = world_size;
    comm->listener_handle = (rank == 0) ? listener_handle : 0;

    if (world_size == 1) return comm;

    if (rank == 0) {
        if (comm->listener_handle <= 0) {
            comm->listener_handle = eshkol_coll_listen_coord(coord_host, coord_port, NULL);
            if (comm->listener_handle < 0) {
                comm->listener_handle = 0;
                eshkol_coll_destroy(comm);
                return NULL;
            }
        }
        comm->peer_streams = (int64_t *)calloc((size_t)world_size, sizeof(int64_t));
        if (!comm->peer_streams) {
            eshkol_coll_destroy(comm);
            return NULL;
        }
        int32_t accepted = 0;
        while (accepted < world_size - 1) {
            int64_t stream = eshkol_net_accept(comm->listener_handle, timeout_ms);
            if (stream < 0) {
                eshkol_coll_destroy(comm);
                return NULL;
            }
            int32_t peer_rank = -1;
            if (recv_frame(stream, COLL_MSG_HELLO, NULL, &peer_rank, world_size,
                           NULL, 0, timeout_ms) != 0 ||
                peer_rank <= 0 || peer_rank >= world_size ||
                comm->peer_streams[peer_rank] != 0) {
                eshkol_net_close_kind(stream, 1);
                eshkol_coll_destroy(comm);
                return NULL;
            }
            comm->peer_streams[peer_rank] = stream;
            accepted++;
        }
        for (int32_t r = 1; r < world_size; ++r) {
            if (send_frame(comm->peer_streams[r], COLL_MSG_HELLO, 0, 0, world_size,
                           NULL, 0, timeout_ms) != 0) {
                eshkol_coll_destroy(comm);
                return NULL;
            }
        }
    } else {
        if (!coord_host || coord_port <= 0 || coord_port > 65535) {
            eshkol_coll_destroy(comm);
            return NULL;
        }
        comm->coord_stream = eshkol_net_open(coord_host, coord_port, 2, timeout_ms);
        if (comm->coord_stream < 0) {
            comm->coord_stream = 0;
            eshkol_coll_destroy(comm);
            return NULL;
        }
        if (send_frame(comm->coord_stream, COLL_MSG_HELLO, 0, rank, world_size,
                       NULL, 0, timeout_ms) != 0) {
            eshkol_coll_destroy(comm);
            return NULL;
        }
        int32_t ack_rank = -1;
        if (recv_frame(comm->coord_stream, COLL_MSG_HELLO, NULL, &ack_rank, world_size,
                       NULL, 0, timeout_ms) != 0 || ack_rank != 0) {
            eshkol_coll_destroy(comm);
            return NULL;
        }
    }
    return comm;
}

void eshkol_coll_destroy(eshkol_coll_comm_t *comm) {
    if (!comm) return;
    if (comm->peer_streams) {
        for (int32_t r = 1; r < comm->world_size; ++r) {
            if (comm->peer_streams[r] > 0) {
                eshkol_net_close_kind(comm->peer_streams[r], 1);
            }
        }
        free(comm->peer_streams);
    }
    if (comm->coord_stream > 0) {
        eshkol_net_close_kind(comm->coord_stream, 1);
    }
    if (comm->listener_handle > 0) {
        eshkol_net_close_kind(comm->listener_handle, 1);
    }
    free(comm);
}

int32_t eshkol_coll_rank(const eshkol_coll_comm_t *comm) {
    return comm ? comm->rank : -1;
}

int32_t eshkol_coll_world_size(const eshkol_coll_comm_t *comm) {
    return comm ? comm->world_size : 0;
}

int32_t eshkol_coll_barrier(eshkol_coll_comm_t *comm, int32_t timeout_ms) {
    if (!comm || timeout_ms <= 0) return ESHKOL_IO_INVALID;
    if (comm->world_size == 1) return 0;

    if (comm->rank == 0) {
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t from = -1;
            int32_t rc = recv_frame(comm->peer_streams[r], COLL_MSG_BARRIER, NULL,
                                    &from, comm->world_size, NULL, 0, timeout_ms);
            if (rc != 0 || from != r) return rc != 0 ? rc : ESHKOL_IO_INVALID;
        }
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t rc = send_frame(comm->peer_streams[r], COLL_MSG_BARRIER, 0,
                                    0, comm->world_size, NULL, 0, timeout_ms);
            if (rc != 0) return rc;
        }
    } else {
        int32_t rc = send_frame(comm->coord_stream, COLL_MSG_BARRIER, 0,
                                comm->rank, comm->world_size, NULL, 0, timeout_ms);
        if (rc != 0) return rc;
        int32_t from = -1;
        rc = recv_frame(comm->coord_stream, COLL_MSG_BARRIER, NULL,
                        &from, comm->world_size, NULL, 0, timeout_ms);
        if (rc != 0 || from != 0) return rc != 0 ? rc : ESHKOL_IO_INVALID;
    }
    return 0;
}

int32_t eshkol_coll_broadcast_f64(eshkol_coll_comm_t *comm, int32_t root,
                                  double *buffer, size_t count, int32_t timeout_ms) {
    if (!comm || root < 0 || root >= comm->world_size || timeout_ms <= 0 ||
        count > ESHKOL_COLL_MAX_ELEMS || (count > 0 && !buffer)) {
        return ESHKOL_IO_INVALID;
    }
    if (comm->world_size == 1) return 0;

    if (comm->rank == 0) {
        if (root != 0) {
            int32_t from = -1;
            int32_t rc = recv_frame(comm->peer_streams[root], COLL_MSG_BROADCAST, 0,
                                    &from, comm->world_size, buffer, count, timeout_ms);
            if (rc != 0 || from != root) return rc != 0 ? rc : ESHKOL_IO_INVALID;
        }
        for (int32_t r = 1; r < comm->world_size; ++r) {
            if (r == root) continue;
            int32_t rc = send_frame(comm->peer_streams[r], COLL_MSG_BROADCAST, 0,
                                    root, comm->world_size, buffer, count, timeout_ms);
            if (rc != 0) return rc;
        }
    } else if (comm->rank == root) {
        int32_t rc = send_frame(comm->coord_stream, COLL_MSG_BROADCAST, 0,
                                comm->rank, comm->world_size, buffer, count, timeout_ms);
        if (rc != 0) return rc;
    } else {
        int32_t from = -1;
        int32_t rc = recv_frame(comm->coord_stream, COLL_MSG_BROADCAST, NULL,
                                &from, comm->world_size, buffer, count, timeout_ms);
        if (rc != 0 || from != root) return rc != 0 ? rc : ESHKOL_IO_INVALID;
    }
    return 0;
}

int32_t eshkol_coll_allreduce_f64(eshkol_coll_comm_t *comm,
                                  const double *send_buf, double *recv_buf,
                                  size_t count, eshkol_coll_op_t op,
                                  int32_t timeout_ms) {
    if (!comm || timeout_ms <= 0 || count > ESHKOL_COLL_MAX_ELEMS ||
        (count > 0 && (!send_buf || !recv_buf)) ||
        (op < ESHKOL_COLL_SUM || op > ESHKOL_COLL_MAX)) {
        return ESHKOL_IO_INVALID;
    }
    if (comm->world_size == 1) {
        if (count > 0 && recv_buf != send_buf) {
            memcpy(recv_buf, send_buf, count * sizeof(double));
        }
        return 0;
    }

    if (comm->rank == 0) {
        double *scratch = count > 0 ? (double *)malloc(count * sizeof(double)) : NULL;
        if (count > 0 && !scratch) return ESHKOL_IO_SYSTEM;

        if (count > 0 && recv_buf != send_buf) {
            memcpy(recv_buf, send_buf, count * sizeof(double));
        }
        /* Strict ascending rank order 1..world_size-1 guarantees bit-identical
         * IEEE-754 fold independent of socket arrival order. */
        for (int32_t r = 1; r < comm->world_size; ++r) {
            uint8_t got_op = 0;
            int32_t from = -1;
            int32_t rc = recv_frame(comm->peer_streams[r], COLL_MSG_ALLREDUCE,
                                    &got_op, &from, comm->world_size,
                                    scratch, count, timeout_ms);
            if (rc != 0 || from != r || got_op != (uint8_t)op) {
                free(scratch);
                return rc != 0 ? rc : ESHKOL_IO_INVALID;
            }
            for (size_t i = 0; i < count; ++i) {
                recv_buf[i] = apply_reduce_op(recv_buf[i], scratch[i], op);
            }
        }
        free(scratch);
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t rc = send_frame(comm->peer_streams[r], COLL_MSG_ALLREDUCE,
                                    (uint8_t)op, 0, comm->world_size,
                                    recv_buf, count, timeout_ms);
            if (rc != 0) return rc;
        }
    } else {
        int32_t rc = send_frame(comm->coord_stream, COLL_MSG_ALLREDUCE,
                                (uint8_t)op, comm->rank, comm->world_size,
                                send_buf, count, timeout_ms);
        if (rc != 0) return rc;
        uint8_t got_op = 0;
        int32_t from = -1;
        rc = recv_frame(comm->coord_stream, COLL_MSG_ALLREDUCE,
                        &got_op, &from, comm->world_size,
                        recv_buf, count, timeout_ms);
        if (rc != 0 || from != 0 || got_op != (uint8_t)op) {
            return rc != 0 ? rc : ESHKOL_IO_INVALID;
        }
    }
    return 0;
}

int32_t eshkol_coll_allgather_f64(eshkol_coll_comm_t *comm,
                                  const double *send_buf, double *recv_buf,
                                  size_t send_count, int32_t timeout_ms) {
    if (!comm || timeout_ms <= 0 ||
        send_count > ESHKOL_COLL_MAX_ELEMS / (size_t)comm->world_size ||
        (send_count > 0 && (!send_buf || !recv_buf))) {
        return ESHKOL_IO_INVALID;
    }
    size_t total_count = send_count * (size_t)comm->world_size;
    if (comm->world_size == 1) {
        if (send_count > 0 && recv_buf != send_buf) {
            memcpy(recv_buf, send_buf, send_count * sizeof(double));
        }
        return 0;
    }

    if (comm->rank == 0) {
        if (send_count > 0) {
            memmove(recv_buf, send_buf, send_count * sizeof(double));
        }
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t from = -1;
            int32_t rc = recv_frame(comm->peer_streams[r], COLL_MSG_ALLGATHER, NULL,
                                    &from, comm->world_size,
                                    recv_buf + (size_t)r * send_count,
                                    send_count, timeout_ms);
            if (rc != 0 || from != r) return rc != 0 ? rc : ESHKOL_IO_INVALID;
        }
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t rc = send_frame(comm->peer_streams[r], COLL_MSG_ALLGATHER, 0,
                                    0, comm->world_size,
                                    recv_buf, total_count, timeout_ms);
            if (rc != 0) return rc;
        }
    } else {
        int32_t rc = send_frame(comm->coord_stream, COLL_MSG_ALLGATHER, 0,
                                comm->rank, comm->world_size,
                                send_buf, send_count, timeout_ms);
        if (rc != 0) return rc;
        int32_t from = -1;
        rc = recv_frame(comm->coord_stream, COLL_MSG_ALLGATHER, NULL,
                        &from, comm->world_size,
                        recv_buf, total_count, timeout_ms);
        if (rc != 0 || from != 0) return rc != 0 ? rc : ESHKOL_IO_INVALID;
    }
    return 0;
}

int32_t eshkol_coll_reduce_scatter_f64(eshkol_coll_comm_t *comm,
                                       const double *send_buf, double *recv_buf,
                                       size_t recv_count, eshkol_coll_op_t op,
                                       int32_t timeout_ms) {
    if (!comm || timeout_ms <= 0 ||
        recv_count > ESHKOL_COLL_MAX_ELEMS / (size_t)comm->world_size ||
        (recv_count > 0 && (!send_buf || !recv_buf)) ||
        (op < ESHKOL_COLL_SUM || op > ESHKOL_COLL_MAX)) {
        return ESHKOL_IO_INVALID;
    }
    size_t total_count = recv_count * (size_t)comm->world_size;
    if (comm->world_size == 1) {
        if (recv_count > 0 && recv_buf != send_buf) {
            memcpy(recv_buf, send_buf, recv_count * sizeof(double));
        }
        return 0;
    }

    if (comm->rank == 0) {
        double *accum = total_count > 0 ? (double *)malloc(total_count * sizeof(double)) : NULL;
        double *scratch = total_count > 0 ? (double *)malloc(total_count * sizeof(double)) : NULL;
        if (total_count > 0 && (!accum || !scratch)) {
            free(accum);
            free(scratch);
            return ESHKOL_IO_SYSTEM;
        }
        if (total_count > 0) memcpy(accum, send_buf, total_count * sizeof(double));
        for (int32_t r = 1; r < comm->world_size; ++r) {
            uint8_t got_op = 0;
            int32_t from = -1;
            int32_t rc = recv_frame(comm->peer_streams[r], COLL_MSG_REDUCE_SCATTER,
                                    &got_op, &from, comm->world_size,
                                    scratch, total_count, timeout_ms);
            if (rc != 0 || from != r || got_op != (uint8_t)op) {
                free(accum);
                free(scratch);
                return rc != 0 ? rc : ESHKOL_IO_INVALID;
            }
            for (size_t i = 0; i < total_count; ++i) {
                accum[i] = apply_reduce_op(accum[i], scratch[i], op);
            }
        }
        free(scratch);
        if (recv_count > 0) memcpy(recv_buf, accum, recv_count * sizeof(double));
        for (int32_t r = 1; r < comm->world_size; ++r) {
            int32_t rc = send_frame(comm->peer_streams[r], COLL_MSG_REDUCE_SCATTER,
                                    (uint8_t)op, 0, comm->world_size,
                                    accum + (size_t)r * recv_count,
                                    recv_count, timeout_ms);
            if (rc != 0) {
                free(accum);
                return rc;
            }
        }
        free(accum);
    } else {
        int32_t rc = send_frame(comm->coord_stream, COLL_MSG_REDUCE_SCATTER,
                                (uint8_t)op, comm->rank, comm->world_size,
                                send_buf, total_count, timeout_ms);
        if (rc != 0) return rc;
        uint8_t got_op = 0;
        int32_t from = -1;
        rc = recv_frame(comm->coord_stream, COLL_MSG_REDUCE_SCATTER,
                        &got_op, &from, comm->world_size,
                        recv_buf, recv_count, timeout_ms);
        if (rc != 0 || from != 0 || got_op != (uint8_t)op) {
            return rc != 0 ? rc : ESHKOL_IO_INVALID;
        }
    }
    return 0;
}
