/*
 * Native deterministic collectives over Eshkol's v1.4.0 socket layer.
 *
 * Implements Barrier, Broadcast, AllReduce (SUM, PROD, MIN, MAX), AllGather,
 * and ReduceScatter over numeric-address TCP sockets owned by eshkol_net_*.
 * All floating-point reductions evaluate in strict ascending rank order
 * (0, 1, ..., world_size - 1) so every rank observes bit-identical IEEE-754
 * binary64 outputs (0 ULP divergence) regardless of packet arrival order.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_CORE_COLLECTIVES_H
#define ESHKOL_CORE_COLLECTIVES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Reduction operator for collective AllReduce / ReduceScatter. */
typedef enum eshkol_coll_op {
    ESHKOL_COLL_SUM  = 0,
    ESHKOL_COLL_PROD = 1,
    ESHKOL_COLL_MIN  = 2,
    ESHKOL_COLL_MAX  = 3
} eshkol_coll_op_t;

/** Opaque communicator owning leased TCP handles for one rank. */
typedef struct eshkol_coll_comm eshkol_coll_comm_t;

/**
 * @brief Create a communicator for one rank in a world of @p world_size ranks.
 *
 * Rank 0 binds a TCP listener on @p coord_host : @p coord_port (pass port 0
 * for an ephemeral port, then query it with eshkol_coll_coord_port() or pass
 * a pre-bound listener port in @p out_port_cb / separate handshake).
 *
 * For single-call symmetric bootstrap where the coordinator port is already
 * chosen (or where rank 0 binds first via eshkol_coll_listen_coord()), call
 * eshkol_coll_init_with_listener() on rank 0 and eshkol_coll_init() on all
 * worker ranks.
 */
int64_t eshkol_coll_listen_coord(const char *coord_host, int32_t coord_port,
                                 int32_t *out_bound_port);

/**
 * @brief Initialize a communicator on rank 0 using a pre-bound listener from
 *        eshkol_coll_listen_coord(), or on rank > 0 by connecting to
 *        @p coord_host : @p coord_port.
 *
 * Takes ownership of @p listener_handle on rank 0 (and closes it on destroy
 * or error). On rank > 0, @p listener_handle is ignored (pass 0).
 *
 * @return Owned communicator on success, or NULL on error/timeout.
 */
eshkol_coll_comm_t *eshkol_coll_init(int32_t rank, int32_t world_size,
                                     const char *coord_host, int32_t coord_port,
                                     int64_t listener_handle, int32_t timeout_ms);

/** @brief Close all owned socket handles and free @p comm. Safe on NULL. */
void eshkol_coll_destroy(eshkol_coll_comm_t *comm);

/** @brief Return this communicator's rank in [0, world_size). */
int32_t eshkol_coll_rank(const eshkol_coll_comm_t *comm);

/** @brief Return this communicator's world size. */
int32_t eshkol_coll_world_size(const eshkol_coll_comm_t *comm);

/** @brief Synchronize all ranks in @p comm. Returns 0 on success, <0 on error. */
int32_t eshkol_coll_barrier(eshkol_coll_comm_t *comm, int32_t timeout_ms);

/**
 * @brief Broadcast @p count IEEE-754 doubles from @p root to all ranks.
 *        Bit-identical across all ranks (including -0.0 and subnormals).
 *        Returns 0 on success, <0 on error.
 */
int32_t eshkol_coll_broadcast_f64(eshkol_coll_comm_t *comm, int32_t root,
                                  double *buffer, size_t count, int32_t timeout_ms);

/**
 * @brief Deterministic AllReduce over @p count IEEE-754 doubles.
 *        Folds values in strict ascending rank order (0..world_size-1) so
 *        @p recv_buf is bit-identical on every rank. Returns 0 on success.
 */
int32_t eshkol_coll_allreduce_f64(eshkol_coll_comm_t *comm,
                                  const double *send_buf, double *recv_buf,
                                  size_t count, eshkol_coll_op_t op,
                                  int32_t timeout_ms);

/**
 * @brief Gather @p send_count IEEE-754 doubles from each rank into @p recv_buf
 *        (length world_size * send_count) in ascending rank order on every rank.
 *        Returns 0 on success, <0 on error.
 */
int32_t eshkol_coll_allgather_f64(eshkol_coll_comm_t *comm,
                                  const double *send_buf, double *recv_buf,
                                  size_t send_count, int32_t timeout_ms);

/**
 * @brief Deterministic ReduceScatter over @p recv_count * world_size doubles.
 *        Folds in strict ascending rank order and delivers slice
 *        [rank * recv_count, (rank + 1) * recv_count) to @p recv_buf.
 *        Returns 0 on success, <0 on error.
 */
int32_t eshkol_coll_reduce_scatter_f64(eshkol_coll_comm_t *comm,
                                       const double *send_buf, double *recv_buf,
                                       size_t recv_count, eshkol_coll_op_t op,
                                       int32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CORE_COLLECTIVES_H */
