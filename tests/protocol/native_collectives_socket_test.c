/*
 * Native deterministic collectives over sockets verification gate (Owner R7).
 *
 * Spawns W=4 concurrent rank threads communicating over real loopback TCP
 * sockets (eshkol_net_*) and verifies:
 *   1. Barrier synchronization across all 4 ranks
 *   2. Broadcast from non-zero root (root=2) preserving exact IEEE-754 bits
 *      (including -0.0, subnormals, 1e-300, 1e+300)
 *   3. Deterministic AllReduce (SUM, PROD, MIN, MAX) on non-associative
 *      IEEE-754 inputs (1e16, 1.0, -1e16, 0.125): 0 ULP bit-identical across
 *      all 4 ranks and matching exact ascending-rank fold
 *   4. AllGather: exact rank-ordered concatenation across all 4 ranks
 *   5. ReduceScatter: exact ascending-rank fold + disjoint slice scatter
 *   6. Exact zero leaked file descriptors after communicator destruction
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */
#include "eshkol/core/collectives.h"

#include <dirent.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* Standalone test policy stub: return 1 (capability granted) for loopback gate. */
int eshkol_capability_require(const char *capability) {
    (void)capability;
    return 1;
}

#define WORLD_SIZE 4
#define N_ELEMS    8

typedef struct {
    int32_t rank;
    int32_t port;
    int64_t listener_handle;
    int32_t status;
    double bcast_out[N_ELEMS];
    double allreduce_sum[N_ELEMS];
    double allreduce_prod[N_ELEMS];
    double allreduce_min[N_ELEMS];
    double allreduce_max[N_ELEMS];
    double allgather_out[WORLD_SIZE * 2];
    double reduce_scatter_out[2];
} rank_ctx_t;

static int count_open_fds(void) {
#if defined(__APPLE__)
    const char *dir_path = "/dev/fd";
#else
    const char *dir_path = "/proc/self/fd";
#endif
    DIR *d = opendir(dir_path);
    if (!d) return -1;
    int n = 0;
    struct dirent *ent;
    while ((ent = readdir(d)) != NULL) {
        if (ent->d_name[0] == '.') continue;
        n++;
    }
    closedir(d);
    return n - 1;
}

/* Non-associative IEEE-754 test vectors: (1e16 + 1.0) + (-1e16) != 1e16 + (1.0 + (-1e16)). */
static void make_rank_input(int32_t rank, double out[N_ELEMS]) {
    static const double table[WORLD_SIZE][N_ELEMS] = {
        { 1.0e16,  1.5,   -0.0,    2.0,    1.0e-16,  7.25, -3.5,  100.0 },
        { 1.0,     2.25,   0.0,   -3.0,    1.0,     -2.5,   4.0,   -0.25 },
        {-1.0e16, -0.5,    3.5,    0.5,   -1.0,      1.25,  0.125,  0.5  },
        { 0.125,   4.0,   -1.25,   4.0,    2.0e-16,  0.5,   9.0,    2.0  }
    };
    memcpy(out, table[rank], N_ELEMS * sizeof(double));
}

static void *rank_worker(void *arg) {
    rank_ctx_t *ctx = (rank_ctx_t *)arg;
    const int32_t timeout_ms = 5000;
    eshkol_coll_comm_t *comm = eshkol_coll_init(
        ctx->rank, WORLD_SIZE, "127.0.0.1", ctx->port,
        ctx->listener_handle, timeout_ms);
    if (!comm) {
        ctx->status = -10;
        return NULL;
    }

    if (eshkol_coll_barrier(comm, timeout_ms) != 0) {
        ctx->status = -11;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    /* Broadcast from root = 2 with tricky IEEE-754 edge values */
    memset(ctx->bcast_out, 0, sizeof(ctx->bcast_out));
    if (ctx->rank == 2) {
        ctx->bcast_out[0] = -0.0;
        ctx->bcast_out[1] = 1.0e-308;
        ctx->bcast_out[2] = 1.0e+300;
        ctx->bcast_out[3] = 3.14159265358979323846;
        ctx->bcast_out[4] = -2.71828182845904523536;
        ctx->bcast_out[5] = 4.9406564584124654e-324; /* smallest subnormal */
        ctx->bcast_out[6] = 0.0;
        ctx->bcast_out[7] = -1.0e-300;
    }
    if (eshkol_coll_broadcast_f64(comm, 2, ctx->bcast_out, N_ELEMS, timeout_ms) != 0) {
        ctx->status = -12;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    double send_buf[N_ELEMS];
    make_rank_input(ctx->rank, send_buf);

    if (eshkol_coll_allreduce_f64(comm, send_buf, ctx->allreduce_sum,
                                  N_ELEMS, ESHKOL_COLL_SUM, timeout_ms) != 0 ||
        eshkol_coll_allreduce_f64(comm, send_buf, ctx->allreduce_prod,
                                  N_ELEMS, ESHKOL_COLL_PROD, timeout_ms) != 0 ||
        eshkol_coll_allreduce_f64(comm, send_buf, ctx->allreduce_min,
                                  N_ELEMS, ESHKOL_COLL_MIN, timeout_ms) != 0 ||
        eshkol_coll_allreduce_f64(comm, send_buf, ctx->allreduce_max,
                                  N_ELEMS, ESHKOL_COLL_MAX, timeout_ms) != 0) {
        ctx->status = -13;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    double gather_send[2] = { (double)(ctx->rank * 10 + 1), (double)(ctx->rank * 10 + 2) };
    if (eshkol_coll_allgather_f64(comm, gather_send, ctx->allgather_out, 2, timeout_ms) != 0) {
        ctx->status = -14;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    if (eshkol_coll_reduce_scatter_f64(comm, send_buf, ctx->reduce_scatter_out,
                                       2, ESHKOL_COLL_SUM, timeout_ms) != 0) {
        ctx->status = -15;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    if (eshkol_coll_barrier(comm, timeout_ms) != 0) {
        ctx->status = -16;
        eshkol_coll_destroy(comm);
        return NULL;
    }

    eshkol_coll_destroy(comm);
    ctx->status = 0;
    return NULL;
}

static int bits_equal(const double *a, const double *b, size_t n) {
    return memcmp(a, b, n * sizeof(double)) == 0;
}

int main(void) {
    int fds_before = count_open_fds();
    int32_t port = 0;
    int64_t listener = eshkol_coll_listen_coord("127.0.0.1", 0, &port);
    if (listener <= 0 || port <= 0) {
        fprintf(stderr, "FAIL: listen_coord failed (listener=%lld port=%d)\n",
                (long long)listener, port);
        return 1;
    }

    rank_ctx_t ctx[WORLD_SIZE];
    pthread_t threads[WORLD_SIZE];
    memset(ctx, 0, sizeof(ctx));
    for (int32_t r = 0; r < WORLD_SIZE; ++r) {
        ctx[r].rank = r;
        ctx[r].port = port;
        ctx[r].listener_handle = (r == 0) ? listener : 0;
        ctx[r].status = -1;
        if (pthread_create(&threads[r], NULL, rank_worker, &ctx[r]) != 0) {
            fprintf(stderr, "FAIL: pthread_create rank %d\n", r);
            return 1;
        }
    }
    for (int32_t r = 0; r < WORLD_SIZE; ++r) {
        pthread_join(threads[r], NULL);
        if (ctx[r].status != 0) {
            fprintf(stderr, "FAIL: rank %d exited with status %d\n", r, ctx[r].status);
            return 1;
        }
    }

    /* Reference ascending-rank fold */
    double ref_sum[N_ELEMS], ref_prod[N_ELEMS], ref_min[N_ELEMS], ref_max[N_ELEMS];
    double inp[WORLD_SIZE][N_ELEMS];
    for (int32_t r = 0; r < WORLD_SIZE; ++r) make_rank_input(r, inp[r]);
    for (size_t i = 0; i < N_ELEMS; ++i) {
        double s = inp[0][i], p = inp[0][i], mn = inp[0][i], mx = inp[0][i];
        for (int32_t r = 1; r < WORLD_SIZE; ++r) {
            s += inp[r][i];
            p *= inp[r][i];
            if (inp[r][i] < mn) mn = inp[r][i];
            if (inp[r][i] > mx) mx = inp[r][i];
        }
        ref_sum[i] = s;
        ref_prod[i] = p;
        ref_min[i] = mn;
        ref_max[i] = mx;
    }

    for (int32_t r = 0; r < WORLD_SIZE; ++r) {
        if (!bits_equal(ctx[r].bcast_out, ctx[2].bcast_out, N_ELEMS)) {
            fprintf(stderr, "FAIL: broadcast bit mismatch on rank %d\n", r);
            return 1;
        }
        if (!bits_equal(ctx[r].allreduce_sum, ref_sum, N_ELEMS) ||
            !bits_equal(ctx[r].allreduce_prod, ref_prod, N_ELEMS) ||
            !bits_equal(ctx[r].allreduce_min, ref_min, N_ELEMS) ||
            !bits_equal(ctx[r].allreduce_max, ref_max, N_ELEMS)) {
            fprintf(stderr, "FAIL: allreduce bit mismatch on rank %d\n", r);
            return 1;
        }
        for (int32_t k = 0; k < WORLD_SIZE; ++k) {
            if (ctx[r].allgather_out[k * 2] != (double)(k * 10 + 1) ||
                ctx[r].allgather_out[k * 2 + 1] != (double)(k * 10 + 2)) {
                fprintf(stderr, "FAIL: allgather mismatch on rank %d slot %d\n", r, k);
                return 1;
            }
        }
        if (!bits_equal(ctx[r].reduce_scatter_out, ref_sum + r * 2, 2)) {
            fprintf(stderr, "FAIL: reduce_scatter bit mismatch on rank %d\n", r);
            return 1;
        }
    }

    int fds_after = count_open_fds();
    if (fds_before >= 0 && fds_after >= 0 && fds_after != fds_before) {
        fprintf(stderr, "FAIL: FD leak detected (before=%d after=%d)\n", fds_before, fds_after);
        return 1;
    }

    printf("PASS: native_collectives_socket_test (world=%d barrier/broadcast/allreduce[4ops]/allgather/reduce_scatter bit-identical 0-ULP fds=%d->%d)\n",
           WORLD_SIZE, fds_before, fds_after);
    return 0;
}
