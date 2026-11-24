/**
 * @file event_loop.c
 * @brief Portable half of the ESH-0011 event loop: handle registry, argument
 *        validation, registration bookkeeping, and result coalescing.
 *
 * Everything here is platform-independent. The five kernel calls live in the
 * one backend file CMake selects (event_loop_kqueue.c / event_loop_epoll.c /
 * event_loop_iocp.c / event_loop_stub.c) behind the contract in
 * event_loop_internal.h.
 *
 * Copyright (c) Eshkol Project — tsotchke. MIT License.
 */

#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#include "event_loop_internal.h"
#include "eshkol/core/handle.h"

extern eshkol_handle_registry_t *eshkol_resource_registry(void);
static _Thread_local unsigned char event_owner_token;
int eshkol_event_loop_is_owner(const eshkol_event_loop_t *loop) {
    return loop && loop->owner_thread == (uintptr_t)&event_owner_token;
}

#include <stdlib.h>
#include <errno.h>
#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#define EVENT_LOOP_HAS_WAKE_PIPE 1
#endif
#include <string.h>

#ifdef ESHKOL_RESOURCE_TESTING
static _Thread_local uint64_t test_add_calls, test_poll_calls;
uint64_t eshkol_event_loop_test_add_calls(void) { return test_add_calls; }
uint64_t eshkol_event_loop_test_poll_calls(void) { return test_poll_calls; }
#endif

/* ─────────────────────────────────────────────────────────────────────
 * Pointer API
 * ───────────────────────────────────────────────────────────────────── */

#define ESHKOL_FD_HASH_EMPTY (-1)
#define ESHKOL_FD_HASH_TOMB  (-2)

static int next_pow2_at_least(int n) {
    int cap = 16;
    while (cap < n && cap < (1 << 29)) cap <<= 1;
    return cap;
}

static uint32_t fd_hash_mix(int fd) {
    return (uint32_t)fd * 2654435761u;
}

static void fd_hash_reset_empty(eshkol_event_loop_t* loop) {
    for (int i = 0; i < loop->fd_hash_cap; ++i) {
        loop->fd_hash_keys[i] = ESHKOL_FD_HASH_EMPTY;
        loop->fd_hash_vals[i] = -1;
    }
    loop->fd_hash_tombstones = 0;
}

static void fd_hash_insert_raw(eshkol_event_loop_t* loop, int fd, int slot) {
    const uint32_t mask = (uint32_t)(loop->fd_hash_cap - 1);
    uint32_t pos = fd_hash_mix(fd) & mask;
    int first_tomb = -1;
    for (;;) {
        const int key = loop->fd_hash_keys[pos];
        if (key == fd) {
            loop->fd_hash_vals[pos] = slot;
            return;
        }
        if (key == ESHKOL_FD_HASH_TOMB) {
            if (first_tomb < 0) first_tomb = (int)pos;
        } else if (key == ESHKOL_FD_HASH_EMPTY) {
            const int target = (first_tomb >= 0) ? first_tomb : (int)pos;
            if (first_tomb >= 0 && loop->fd_hash_tombstones > 0)
                --loop->fd_hash_tombstones;
            loop->fd_hash_keys[target] = fd;
            loop->fd_hash_vals[target] = slot;
            return;
        }
        pos = (pos + 1u) & mask;
    }
}

static void fd_hash_rehash(eshkol_event_loop_t* loop) {
    fd_hash_reset_empty(loop);
    for (int i = 0; i < loop->max_registrations; ++i) {
        if (loop->regs[i].fd >= 0) fd_hash_insert_raw(loop, loop->regs[i].fd, i);
    }
}

static void fd_hash_insert(eshkol_event_loop_t* loop, int fd, int slot) {
    if ((loop->n_regs + loop->fd_hash_tombstones) * 4 >= loop->fd_hash_cap * 3)
        fd_hash_rehash(loop);
    fd_hash_insert_raw(loop, fd, slot);
}

static void fd_hash_erase(eshkol_event_loop_t* loop, int fd) {
    const uint32_t mask = (uint32_t)(loop->fd_hash_cap - 1);
    uint32_t pos = fd_hash_mix(fd) & mask;
    for (;;) {
        const int key = loop->fd_hash_keys[pos];
        if (key == ESHKOL_FD_HASH_EMPTY) return;
        if (key == fd) {
            loop->fd_hash_keys[pos] = ESHKOL_FD_HASH_TOMB;
            loop->fd_hash_vals[pos] = -1;
            ++loop->fd_hash_tombstones;
            return;
        }
        pos = (pos + 1u) & mask;
    }
}

static void event_loop_free_tables(eshkol_event_loop_t* loop) {
    free(loop->regs);
    free(loop->out);
    free(loop->fd_hash_keys);
    free(loop->fd_hash_vals);
    free(loop->reg_free_next);
    free(loop->reg_emit_gen);
    free(loop->reg_emit_idx);
}

eshkol_event_loop_t* eshkol_event_loop_create(int max_events) {
    if (max_events <= 0 || max_events > ESHKOL_EVENT_LOOP_MAX_EVENTS) return NULL;

    /* calloc, never the arena: lib/core/runtime_regions.cpp evacuates a region
     * by memcpy-ing objects to a new address, which would duplicate a struct
     * that owns a kernel descriptor. A malloc'd pointer is outside every
     * region arena, so region_index_owning() returns -1 and the evacuator
     * leaves it alone. See the memory-model note in event_loop.h. */
    eshkol_event_loop_t* loop =
        (eshkol_event_loop_t*)calloc(1, sizeof(eshkol_event_loop_t));
    if (!loop) return NULL;

    loop->owner_thread = (uintptr_t)&event_owner_token;
    loop->max_registrations = max_events;
    loop->max_events = max_events + 1;
    loop->wake_read = loop->wake_write = -1;
    loop->backend_fd = -1;
    loop->fd_hash_cap = next_pow2_at_least(max_events * 2 + 4);
    loop->regs = (eshkol_event_reg_t*)calloc((size_t)loop->max_events,
                                             sizeof(eshkol_event_reg_t));
    loop->out = (eshkol_event_t*)calloc((size_t)loop->max_events,
                                        sizeof(eshkol_event_t));
    loop->fd_hash_keys = (int*)malloc((size_t)loop->fd_hash_cap * sizeof(int));
    loop->fd_hash_vals = (int*)malloc((size_t)loop->fd_hash_cap * sizeof(int));
    loop->reg_free_next = (int*)malloc((size_t)max_events * sizeof(int));
    loop->reg_emit_gen = (uint32_t*)calloc((size_t)max_events, sizeof(uint32_t));
    loop->reg_emit_idx = (int*)calloc((size_t)max_events, sizeof(int));
    if (!loop->regs || !loop->out || !loop->fd_hash_keys || !loop->fd_hash_vals ||
        !loop->reg_free_next || !loop->reg_emit_gen || !loop->reg_emit_idx) {
        event_loop_free_tables(loop);
        free(loop);
        return NULL;
    }
    for (int i = 0; i < max_events; ++i) {
        loop->regs[i].fd = -1;
        loop->reg_free_next[i] = (i + 1 < max_events) ? (i + 1) : -1;
    }
    loop->regs[max_events].fd = -1;
    loop->reg_free_head = 0;
    fd_hash_reset_empty(loop);

    if (eshkol_event_backend_open(loop) != ESHKOL_EVENT_LOOP_OK) {
        event_loop_free_tables(loop);
        free(loop);
        return NULL;
    }
#if defined(EVENT_LOOP_HAS_WAKE_PIPE)
    int fds[2];
#if defined(__linux__)
    /* Atomic flags prevent a concurrent subprocess spawn from inheriting the pipe. */
    if (pipe2(fds, O_CLOEXEC | O_NONBLOCK) != 0) {
#else
    if (pipe(fds) != 0) {
#endif
        eshkol_event_backend_close(loop); event_loop_free_tables(loop); free(loop); return NULL;
    }
    loop->wake_read = fds[0]; loop->wake_write = fds[1];
#if !defined(__linux__)
    int read_flags = fcntl(fds[0], F_GETFL), write_flags = fcntl(fds[1], F_GETFL);
    if (read_flags < 0 || write_flags < 0 ||
        fcntl(fds[0], F_SETFL, read_flags | O_NONBLOCK) < 0 ||
        fcntl(fds[1], F_SETFL, write_flags | O_NONBLOCK) < 0 ||
        fcntl(fds[0], F_SETFD, FD_CLOEXEC) < 0 || fcntl(fds[1], F_SETFD, FD_CLOEXEC) < 0) {
        close(fds[0]); close(fds[1]); eshkol_event_backend_close(loop);
        event_loop_free_tables(loop); free(loop); return NULL;
    }
#endif
    if (eshkol_event_backend_add(loop, fds[0], ESHKOL_EVENT_READ, 0) != ESHKOL_EVENT_LOOP_OK) {
        close(fds[0]); close(fds[1]); eshkol_event_backend_close(loop);
        event_loop_free_tables(loop); free(loop); return NULL;
    }
#endif
    return loop;
}
int eshkol_event_loop_wake(eshkol_event_loop_t *loop) {
#if defined(EVENT_LOOP_HAS_WAKE_PIPE)
    if (!loop || loop->wake_write < 0) return ESHKOL_EVENT_LOOP_EINVAL;
    char byte = 1;
    ssize_t rc;
    do { rc = write(loop->wake_write, &byte, 1); } while (rc < 0 && errno == EINTR);
    return rc == 1 || (rc < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) ?
        ESHKOL_EVENT_LOOP_OK : ESHKOL_EVENT_LOOP_EOSERR;
#else
    (void)loop; return ESHKOL_EVENT_LOOP_ENOTSUP;
#endif
}
int eshkol_event_loop_set_maintenance(eshkol_event_loop_t *loop,
                                    void (*callback)(void *), void *data) {
    if (!loop || !eshkol_event_loop_is_owner(loop)) return ESHKOL_EVENT_LOOP_EBUSY;
    if (loop->maintenance && loop->maintenance != callback) return ESHKOL_EVENT_LOOP_EBUSY;
    loop->maintenance = callback; loop->maintenance_data = data;
    return ESHKOL_EVENT_LOOP_OK;
}

/** O(1) index of @p fd in the registration table, or -1. */
static int reg_find(const eshkol_event_loop_t* loop, int fd) {
    if (fd < 0 || loop->fd_hash_cap <= 0) return -1;
    const uint32_t mask = (uint32_t)(loop->fd_hash_cap - 1);
    uint32_t pos = fd_hash_mix(fd) & mask;
    for (;;) {
        const int key = loop->fd_hash_keys[pos];
        if (key == fd) return loop->fd_hash_vals[pos];
        if (key == ESHKOL_FD_HASH_EMPTY) return -1;
        pos = (pos + 1u) & mask;
    }
}

int eshkol_event_loop_registration_present(const eshkol_event_loop_t *loop, int fd) {
    return loop && reg_find(loop, fd) >= 0;
}

/** O(1) peek of a free registration slot, or -1 when the table is full. */
static int reg_free_slot(const eshkol_event_loop_t* loop) {
    return loop->reg_free_head;
}

uint64_t eshkol_event_lookup_user_data(const eshkol_event_loop_t* loop, int fd) {
    if (!loop) return 0;
    const int idx = reg_find(loop, fd);
    return idx < 0 ? 0u : loop->regs[idx].user_data;
}

int eshkol_event_loop_registration_matches(const eshkol_event_loop_t* loop,
    int fd, int events, uint64_t user_data) {
    if (!loop || fd < 0) return 0;
    const int idx = reg_find(loop, fd);
    return idx >= 0 && loop->regs[idx].events == events &&
           loop->regs[idx].user_data == user_data;
}

int eshkol_event_emit(eshkol_event_loop_t* loop, int* n_out, int fd, int events) {
    if (!loop || !n_out) return 0;

    /* O(1) coalescing via per-poll epoch stamps: kqueue delivers one record
     * per filter, epoll and Windows one per descriptor. */
    if (fd == loop->wake_read && fd >= 0) {
        if (loop->emit_epoch != 0 && loop->wake_emit_gen == loop->emit_epoch &&
            loop->wake_emit_idx >= 0 && loop->wake_emit_idx < *n_out &&
            loop->out[loop->wake_emit_idx].fd == fd) {
            loop->out[loop->wake_emit_idx].events |= events;
            return 1;
        }
        if (*n_out >= loop->max_events) return 0;
        loop->out[*n_out].fd = fd;
        loop->out[*n_out].events = events;
        loop->out[*n_out].user_data = 0u;
        loop->wake_emit_gen = loop->emit_epoch;
        loop->wake_emit_idx = *n_out;
        ++(*n_out);
        return 1;
    }

    const int slot = reg_find(loop, fd);
    if (slot >= 0) {
        if (loop->emit_epoch != 0 && loop->reg_emit_gen[slot] == loop->emit_epoch &&
            loop->reg_emit_idx[slot] >= 0 && loop->reg_emit_idx[slot] < *n_out &&
            loop->out[loop->reg_emit_idx[slot]].fd == fd) {
            loop->out[loop->reg_emit_idx[slot]].events |= events;
            return 1;
        }
        if (*n_out >= loop->max_events) return 0;
        loop->out[*n_out].fd = fd;
        loop->out[*n_out].events = events;
        loop->out[*n_out].user_data = loop->regs[slot].user_data;
        loop->reg_emit_gen[slot] = loop->emit_epoch;
        loop->reg_emit_idx[slot] = *n_out;
        ++(*n_out);
        return 1;
    }

    for (int i = 0; i < *n_out; ++i) {
        if (loop->out[i].fd == fd) {
            loop->out[i].events |= events;
            return 1;
        }
    }
    if (*n_out >= loop->max_events) return 0;

    loop->out[*n_out].fd = fd;
    loop->out[*n_out].events = events;
    loop->out[*n_out].user_data = 0u;
    ++(*n_out);
    return 1;
}

int eshkol_event_loop_add(eshkol_event_loop_t* loop, int fd, int events,
                          uint64_t user_data) {
    if (!loop || fd < 0 || fd == loop->wake_read) return ESHKOL_EVENT_LOOP_EINVAL;
    if (!eshkol_event_loop_is_owner(loop)) return ESHKOL_EVENT_LOOP_EBUSY;
#ifdef ESHKOL_RESOURCE_TESTING
    ++test_add_calls;
#endif

    /* ESHKOL_EVENT_ERROR is reported, never requested — asking for it alone is
     * a caller mistake, not a silent no-op registration. */
    events &= ESHKOL_EVENT_ALL;
    if (events == 0) return ESHKOL_EVENT_LOOP_EINVAL;

    int idx = reg_find(loop, fd);
    const int prev_events = (idx >= 0) ? loop->regs[idx].events : 0;
    if (idx < 0) {
        idx = reg_free_slot(loop);
        if (idx < 0) return ESHKOL_EVENT_LOOP_EFULL;
    }

    const int rc = eshkol_event_backend_add(loop, fd, events, prev_events);
    if (rc != ESHKOL_EVENT_LOOP_OK) return rc;

    if (prev_events == 0) {
        loop->reg_free_head = loop->reg_free_next[idx];
        loop->reg_free_next[idx] = -1;
        ++loop->n_regs;
        loop->regs[idx].fd = fd;
        fd_hash_insert(loop, fd, idx);
    }
    loop->regs[idx].events = events;
    loop->regs[idx].user_data = user_data;
    return ESHKOL_EVENT_LOOP_OK;
}

#ifdef ESHKOL_RESOURCE_TESTING
static int fail_remove_once;
void eshkol_event_loop_test_fail_remove_once(void) { fail_remove_once = 1; }
#endif
int eshkol_event_loop_remove(eshkol_event_loop_t* loop, int fd) {
    if (!loop || fd < 0 || fd == loop->wake_read) return ESHKOL_EVENT_LOOP_EINVAL;
    if (!eshkol_event_loop_is_owner(loop)) return ESHKOL_EVENT_LOOP_EBUSY;

    const int idx = reg_find(loop, fd);
    if (idx < 0) return ESHKOL_EVENT_LOOP_ENOENT;

    int rc;
#ifdef ESHKOL_RESOURCE_TESTING
    if (fail_remove_once) {
        fail_remove_once = 0; loop->last_os_error = EIO;
        rc = ESHKOL_EVENT_LOOP_EOSERR;
    } else
#endif
    rc = eshkol_event_backend_remove(loop, fd, loop->regs[idx].events);

    /* Keep the owner and watch on a genuine removal failure. Already-gone
     * descriptors are reported as success by the platform backend. */
    if (rc != ESHKOL_EVENT_LOOP_OK) return rc;
    fd_hash_erase(loop, fd);
    loop->regs[idx].fd = -1;
    loop->regs[idx].events = 0;
    loop->regs[idx].user_data = 0;
    loop->reg_emit_gen[idx] = 0;
    loop->reg_free_next[idx] = loop->reg_free_head;
    loop->reg_free_head = idx;
    if (loop->n_regs > 0) --loop->n_regs;
    return rc;
}

int eshkol_event_loop_poll(eshkol_event_loop_t* loop, int timeout_ms,
                           const eshkol_event_t** out_events, int* n_events) {
    if (!loop || !out_events || !n_events) return ESHKOL_EVENT_LOOP_EINVAL;
    if (!eshkol_event_loop_is_owner(loop)) return ESHKOL_EVENT_LOOP_EBUSY;
#ifdef ESHKOL_RESOURCE_TESTING
    ++test_poll_calls;
#endif

    *out_events = loop->out;
    *n_events = 0;
    if (loop->maintenance) loop->maintenance(loop->maintenance_data);

    /* A managed owner can still receive a cross-thread request with no public
     * watch armed. Terminal tickets no longer retain a loop, so a leftover
     * maintenance hook alone must not turn an empty unbounded poll into a wait.
     * Finite empty polls retain their timeout behavior for process reaping. */
    if (loop->n_regs == 0 && !loop->managed_owners && timeout_ms < 0)
        return ESHKOL_EVENT_LOOP_OK;

    if (++loop->emit_epoch == 0) {
        memset(loop->reg_emit_gen, 0,
               (size_t)loop->max_registrations * sizeof(uint32_t));
        loop->wake_emit_gen = 0;
        loop->emit_epoch = 1;
    }

    int n = 0;
    const int rc = eshkol_event_backend_poll(loop, timeout_ms, &n);
    if (rc != ESHKOL_EVENT_LOOP_OK) return rc;

    if (n < 0) n = 0;
    if (n > loop->max_events) n = loop->max_events;
#if defined(EVENT_LOOP_HAS_WAKE_PIPE)
    int public_count = 0;
    for (int i = 0; i < n; ++i) {
        if (loop->out[i].fd == loop->wake_read) {
            char bytes[64];
            while (read(loop->wake_read, bytes, sizeof(bytes)) > 0) {}
        } else loop->out[public_count++] = loop->out[i];
    }
    n = public_count;
#endif
    if (loop->maintenance) loop->maintenance(loop->maintenance_data);
    *n_events = n;
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_loop_close(eshkol_event_loop_t* loop) {
    if (!loop) return ESHKOL_EVENT_LOOP_EINVAL;
    if (!eshkol_event_loop_is_owner(loop)) return ESHKOL_EVENT_LOOP_EBUSY;
    if (loop->maintenance) loop->maintenance(loop->maintenance_data);
    if (loop->retain_count) return ESHKOL_EVENT_LOOP_EBUSY;
    eshkol_event_backend_close(loop);
#if defined(EVENT_LOOP_HAS_WAKE_PIPE)
    if (loop->wake_read >= 0) close(loop->wake_read);
    if (loop->wake_write >= 0) close(loop->wake_write);
#endif
    event_loop_free_tables(loop);
    free(loop);
    return ESHKOL_EVENT_LOOP_OK;
}

int eshkol_event_loop_count(const eshkol_event_loop_t* loop) {
    return loop ? loop->n_regs : -1;
}

int eshkol_event_loop_last_os_error(const eshkol_event_loop_t* loop) {
    return loop ? loop->last_os_error : 0;
}

const char* eshkol_event_loop_backend_name(void) {
    return eshkol_event_backend_name();
}

/* Public loop identities share the exact process-lifetime resource registry.
 * Kind checks prevent PROCESS/SOCKET integers from naming a loop. Ordinary
 * pointer and handle operations remain confined to the adopting thread. */

static int close_public_loop(eshkol_handle_kind_t kind, uintptr_t native, void *data) {
    (void)kind; (void)data;
    return eshkol_event_loop_close((eshkol_event_loop_t *)native);
}
int64_t eshkol_event_loop_open_handle(int max_events) {
    eshkol_event_loop_t *loop = eshkol_event_loop_create(max_events);
    if (!loop) return -1;
    eshkol_handle_id_t id;
    if (eshkol_handle_create_tagged(eshkol_resource_registry(), ESHKOL_HANDLE_KIND_EVENT_LOOP,
            0, (uintptr_t)loop, close_public_loop, NULL, &id)) {
        eshkol_event_loop_close(loop); return -1;
    }
    return (int64_t)id;
}
eshkol_event_loop_t *eshkol_event_loop_from_handle(int64_t handle) {
    if (handle <= 0) return NULL;
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    eshkol_handle_kind_t kind; uint32_t tag;
    if (eshkol_handle_get_kind(registry, (eshkol_handle_id_t)handle, &kind, &tag) ||
        kind != ESHKOL_HANDLE_KIND_EVENT_LOOP || tag != 0) return NULL;
    uintptr_t native; uint64_t lease;
    if (eshkol_handle_borrow(registry, (eshkol_handle_id_t)handle, &native, &lease)) return NULL;
    eshkol_event_loop_t *loop = (eshkol_event_loop_t *)native;
    int owner = eshkol_event_loop_is_owner(loop);
    (void)eshkol_handle_release(registry, (eshkol_handle_id_t)handle, lease);
    return owner ? loop : NULL;
}
int eshkol_event_loop_close_handle(int64_t handle) {
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(handle);
    if (!loop) {
        eshkol_handle_kind_t kind; uint32_t tag;
        if (handle > 0 && eshkol_handle_get_kind(eshkol_resource_registry(),
                (eshkol_handle_id_t)handle, &kind, &tag) == ESHKOL_HANDLE_OK &&
                kind == ESHKOL_HANDLE_KIND_EVENT_LOOP) return ESHKOL_EVENT_LOOP_EBUSY;
        return ESHKOL_EVENT_LOOP_EINVAL;
    }
    if (loop->maintenance) loop->maintenance(loop->maintenance_data);
    if (loop->retain_count) return ESHKOL_EVENT_LOOP_EBUSY;
    int rc = eshkol_handle_close(eshkol_resource_registry(), (eshkol_handle_id_t)handle);
    return rc ? ESHKOL_EVENT_LOOP_EINVAL : ESHKOL_EVENT_LOOP_OK;
}
int eshkol_event_loop_retain_handle(int64_t handle) {
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(handle);
    if (!loop) return ESHKOL_EVENT_LOOP_EINVAL;
    if (loop->retain_count == UINT32_MAX) return ESHKOL_EVENT_LOOP_EFULL;
    ++loop->retain_count;
    return ESHKOL_EVENT_LOOP_OK;
}
int eshkol_event_loop_release_handle(int64_t handle) {
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(handle);
    if (!loop || !loop->retain_count) return ESHKOL_EVENT_LOOP_EINVAL;
    --loop->retain_count;
    return ESHKOL_EVENT_LOOP_OK;
}
