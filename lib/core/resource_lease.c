/* Shared lease and outcome layer for hosted I/O resources.
 * See inc/eshkol/core/resource_lease.h for the contract. */
#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif
#include "eshkol/core/resource_lease.h"
#include "eshkol/core/event_loop.h"
#include <limits.h>
#include <stddef.h>
#include <string.h>

#define RESOURCE_LAST_ID INT64_C(9007199254740991)

#if defined(_WIN32)
#include <windows.h>
static INIT_ONCE registry_once = INIT_ONCE_STATIC_INIT;
static eshkol_handle_registry_t *process_registry;
static BOOL CALLBACK create_registry_once(PINIT_ONCE once, PVOID parameter, PVOID *context) {
    (void)once; (void)parameter; (void)context;
    process_registry = eshkol_handle_registry_create_exact();
    return TRUE;
}
eshkol_handle_registry_t *eshkol_resource_registry(void) {
    InitOnceExecuteOnce(&registry_once, create_registry_once, NULL, NULL);
    return process_registry;
}
int64_t eshkol_resource_now_ms(void) { return (int64_t)GetTickCount64(); }
#else
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
static pthread_once_t registry_once = PTHREAD_ONCE_INIT;
static eshkol_handle_registry_t *process_registry;
static void create_registry_once(void) { process_registry = eshkol_handle_registry_create_exact(); }
eshkol_handle_registry_t *eshkol_resource_registry(void) {
    /* Never destroyed: process exit releases descriptors, and destroying it
     * would race resources still owned by detached threads. */
    pthread_once(&registry_once, create_registry_once);
    return process_registry;
}
int64_t eshkol_resource_now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t) != 0) return -1;
    return (int64_t)t.tv_sec * 1000 + t.tv_nsec / 1000000;
}
#endif

int64_t eshkol_resource_deadline(int32_t timeout_ms) {
    if (timeout_ms < 0) return -1;
    int64_t now = eshkol_resource_now_ms();
    return now < 0 ? -1 : now + timeout_ms;
}
int32_t eshkol_resource_remaining(int64_t deadline) {
    if (deadline < 0) return -1;
    int64_t now = eshkol_resource_now_ms();
    if (now < 0) return -1;
    int64_t left = deadline - now;
    if (left <= 0) return 0;
    return left > INT_MAX ? INT_MAX : (int32_t)left;
}

int64_t eshkol_resource_adopt(eshkol_handle_kind_t kind, uint32_t tag, uintptr_t native,
                              eshkol_handle_close_fn close_fn, void *user_data) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    eshkol_handle_id_t id = 0;
    if (!registry || !close_fn ||
        eshkol_handle_create_tagged(registry, kind, tag, native, close_fn, user_data, &id) !=
            ESHKOL_HANDLE_OK)
        return ESHKOL_IO_SYSTEM;
    if (id > (eshkol_handle_id_t)RESOURCE_LAST_ID) {
        /* Unreachable with an exact registry; never hand out an inexact ID. */
        (void)eshkol_handle_close(registry, id);
        return ESHKOL_IO_SYSTEM;
    }
    return (int64_t)id;
}

static int resource_matches(eshkol_handle_registry_t *registry, int64_t id,
                            eshkol_handle_kind_t kind, uint32_t tag_mask, uint32_t *out_tag) {
    eshkol_handle_kind_t actual = 0;
    uint32_t tag = 0;
    if (id <= 0 || id > RESOURCE_LAST_ID ||
        eshkol_handle_get_kind(registry, (eshkol_handle_id_t)id, &actual, &tag) != ESHKOL_HANDLE_OK ||
        actual != kind || tag >= 32 || (tag_mask && !(tag_mask & (UINT32_C(1) << tag))))
        return 0;
    if (out_tag) *out_tag = tag;
    return 1;
}

int eshkol_resource_borrow(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask,
                           eshkol_resource_lease_t *out) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    uint32_t tag = 0;
    if (!registry || !out || !resource_matches(registry, id, kind, tag_mask, &tag))
        return ESHKOL_IO_INVALID;
    uintptr_t native = 0;
    uint64_t lease = 0;
    int status = eshkol_handle_borrow(registry, (eshkol_handle_id_t)id, &native, &lease);
    if (status == ESHKOL_HANDLE_EBUSY) return ESHKOL_IO_BUSY;
    if (status != ESHKOL_HANDLE_OK) return ESHKOL_IO_INVALID;
    out->id = id; out->lease = lease; out->native = native; out->tag = tag;
    return 0;
}

void eshkol_resource_release(eshkol_resource_lease_t *lease) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    if (!registry || !lease || !lease->lease) return;
    (void)eshkol_handle_release(registry, (eshkol_handle_id_t)lease->id, lease->lease);
    lease->lease = 0;
    lease->native = 0;
}

int eshkol_resource_close(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    if (!registry || !resource_matches(registry, id, kind, tag_mask, NULL))
        return ESHKOL_IO_INVALID;
    int status = eshkol_handle_close_after_release(registry, (eshkol_handle_id_t)id);
    /* ECLOSE still retires the identity: an uncertain OS close is never retried. */
    return status == ESHKOL_HANDLE_OK || status == ESHKOL_HANDLE_ECLOSE ? 0 : ESHKOL_IO_INVALID;
}

int eshkol_resource_cancel(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    if (!registry || !resource_matches(registry, id, kind, tag_mask, NULL))
        return ESHKOL_IO_INVALID;
    return eshkol_handle_interrupt(registry, (eshkol_handle_id_t)id) == ESHKOL_HANDLE_OK
               ? 0 : ESHKOL_IO_INVALID;
}

int eshkol_resource_bytes(const void *bytevector, int64_t start, int64_t count,
                          uint8_t **out_bytes) {
    if (!bytevector || !out_bytes || start < 0 || count < 0) return ESHKOL_IO_INVALID;
    int64_t length;
    memcpy(&length, bytevector, sizeof(length));
    if (length < 0 || start > length || count > length - start) return ESHKOL_IO_INVALID;
    *out_bytes = (uint8_t *)bytevector + sizeof(int64_t) + start;
    return 0;
}

#if defined(_WIN32) || defined(ESHKOL_VM_WASM) || defined(__EMSCRIPTEN__)
int eshkol_resource_wait(const eshkol_resource_lease_t *lease, int fd, int events,
                         int64_t deadline) {
    (void)lease; (void)fd; (void)events; (void)deadline;
    return ESHKOL_IO_UNSUPPORTED;
}
#else
/* One event loop plus one wake pipe per thread, created on first wait and
 * released at thread exit, so steady-state descriptor use is constant. */
typedef struct {
    eshkol_event_loop_t *loop;
    int wake_read;
    int wake_write;
} WaitContext;
static pthread_key_t wait_key;
static pthread_once_t wait_key_once = PTHREAD_ONCE_INIT;
static int wait_key_ready;
static void wait_context_free(void *value) {
    WaitContext *context = (WaitContext *)value;
    if (!context) return;
    if (context->loop) eshkol_event_loop_close(context->loop);
    if (context->wake_read >= 0) close(context->wake_read);
    if (context->wake_write >= 0) close(context->wake_write);
    free(context);
}
static void wait_key_create(void) { wait_key_ready = pthread_key_create(&wait_key, wait_context_free) == 0; }
static int set_pipe_flags(int fd) {
    int flags = fcntl(fd, F_GETFL);
    return flags >= 0 && fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0 &&
           fcntl(fd, F_SETFD, FD_CLOEXEC) == 0;
}
static WaitContext *wait_context(void) {
    pthread_once(&wait_key_once, wait_key_create);
    if (!wait_key_ready) return NULL;
    WaitContext *context = (WaitContext *)pthread_getspecific(wait_key);
    if (context) return context;
    context = (WaitContext *)calloc(1, sizeof(*context));
    if (!context) return NULL;
    int fds[2];
    context->wake_read = context->wake_write = -1;
    if (pipe(fds) != 0) { free(context); return NULL; }
    context->wake_read = fds[0]; context->wake_write = fds[1];
    context->loop = eshkol_event_loop_create(2);
    if (!set_pipe_flags(fds[0]) || !set_pipe_flags(fds[1]) || !context->loop ||
        eshkol_event_loop_add(context->loop, context->wake_read, ESHKOL_EVENT_READ, 0) !=
            ESHKOL_EVENT_LOOP_OK ||
        pthread_setspecific(wait_key, context) != 0) {
        wait_context_free(context);
        return NULL;
    }
    return context;
}
static void wake_waiter(void *value) {
    WaitContext *context = (WaitContext *)value;
    const char byte = 1;
    /* A full pipe already holds a pending wake. */
    ssize_t ignored = write(context->wake_write, &byte, 1);
    (void)ignored;
}
static void drain_wake(WaitContext *context) {
    char buffer[64];
    while (read(context->wake_read, buffer, sizeof(buffer)) > 0) {}
}
int eshkol_resource_wait(const eshkol_resource_lease_t *lease, int fd, int events,
                         int64_t deadline) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    if (!registry || !lease || !lease->lease || fd < 0 ||
        !(events & (ESHKOL_RESOURCE_READ | ESHKOL_RESOURCE_WRITE)))
        return ESHKOL_IO_INVALID;
    WaitContext *context = wait_context();
    if (!context) return ESHKOL_IO_SYSTEM;
    eshkol_handle_id_t id = (eshkol_handle_id_t)lease->id;
    drain_wake(context);
    int status = eshkol_handle_set_waker(registry, id, lease->lease, wake_waiter, context);
    if (status == ESHKOL_HANDLE_EINTERRUPTED) {
        (void)eshkol_handle_set_waker(registry, id, lease->lease, NULL, NULL);
        (void)eshkol_handle_take_interrupt(registry, id, lease->lease);
        return ESHKOL_IO_CANCELLED;
    }
    if (status != ESHKOL_HANDLE_OK) return ESHKOL_IO_SYSTEM;
    int result;
    int interest = events & (ESHKOL_EVENT_READ | ESHKOL_EVENT_WRITE);
    if (eshkol_event_loop_add(context->loop, fd, interest, (uint64_t)1) != ESHKOL_EVENT_LOOP_OK) {
        result = ESHKOL_IO_SYSTEM;
    } else {
        for (;;) {
            int32_t left = eshkol_resource_remaining(deadline);
            if (left < 0) { result = ESHKOL_IO_SYSTEM; break; }
            const eshkol_event_t *fired = NULL;
            int count = 0;
            if (eshkol_event_loop_poll(context->loop, left, &fired, &count) != ESHKOL_EVENT_LOOP_OK) {
                result = ESHKOL_IO_SYSTEM; break;
            }
            int ready = 0;
            for (int i = 0; i < count; ++i) {
                if (fired[i].fd == context->wake_read) drain_wake(context);
                else if (fired[i].fd == fd) ready = 1;
            }
            if (eshkol_handle_take_interrupt(registry, id, lease->lease) == 1) {
                result = ESHKOL_IO_CANCELLED; break;
            }
            /* Errors and hangups are reported as ready so the next system call
             * observes EOF, reset or the exact failure. */
            if (ready) { result = 0; break; }
            if (!left) { result = ESHKOL_IO_TIMEOUT; break; }
        }
        (void)eshkol_event_loop_remove(context->loop, fd);
    }
    (void)eshkol_handle_set_waker(registry, id, lease->lease, NULL, NULL);
    return result;
}
#endif
