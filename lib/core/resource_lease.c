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

static int managed_before_borrow(int64_t id);
static int managed_has_id(int64_t id);
static int managed_cancel(int64_t id);
static int managed_close_detached(int64_t id, eshkol_handle_kind_t kind, uint32_t mask);


#if defined(_WIN32)
#include <windows.h>
int64_t eshkol_resource_now_ms(void) { return (int64_t)GetTickCount64(); }
#else
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
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
    int admission = managed_before_borrow(id);
    if (admission) return admission;
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
    (void)eshkol_resource_pump(0);
}

int eshkol_resource_close(int64_t id, eshkol_handle_kind_t kind, uint32_t tag_mask) {
    if (kind == ESHKOL_HANDLE_KIND_EVENT_LOOP) {
        if (!resource_matches(eshkol_resource_registry(), id, kind, tag_mask, NULL))
            return ESHKOL_IO_INVALID;
        int rc = eshkol_event_loop_close_handle(id);
        return rc == ESHKOL_EVENT_LOOP_OK ? 0 :
               rc == ESHKOL_EVENT_LOOP_EBUSY ? ESHKOL_IO_BUSY : ESHKOL_IO_INVALID;
    }
    if (kind == ESHKOL_HANDLE_KIND_CLEANUP) {
        if (!resource_matches(eshkol_resource_registry(), id, kind, tag_mask, NULL))
            return ESHKOL_IO_INVALID;
        return eshkol_resource_cleanup_release(id);
    }
    if (managed_has_id(id)) return managed_close_detached(id, kind, tag_mask);
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
    if (managed_has_id(id)) return managed_cancel(id);
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
    if (managed_has_id(lease->id)) {
        int result = eshkol_resource_wait_on_loop(lease, fd, events, deadline);
        return result > 0 ? 0 : result;
    }
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

#if !defined(_WIN32) && !defined(ESHKOL_VM_WASM) && !defined(__EMSCRIPTEN__)
/* Explicit-loop resources share registry identities with PROCESS and network
 * resources. The registry callback only queues work: native teardown always
 * runs on the adopting loop thread, after the last borrow has ended. */
typedef struct ManagedResource {
    int64_t id, ticket, loop;
    eshkol_event_loop_t *loop_pointer;
    pthread_t owner;
    uintptr_t native;
    eshkol_resource_cleanup_fn cleanup;
    void *data;
    int sticky, cancelled, closing, ready, running, detached, native_done;
    int wait_fd, wait_events;
    eshkol_resource_cleanup_result_t result;
    struct ManagedResource *next;
} ManagedResource;
static pthread_mutex_t managed_lock = PTHREAD_MUTEX_INITIALIZER;
static ManagedResource *managed_resources;
#define CLEANUP_TAG 31u

static ManagedResource *managed_find(int64_t id) {
    for (ManagedResource *m = managed_resources; m; m = m->next)
        if (m->id == id) return m;
    return NULL;
}
static ManagedResource *ticket_find(int64_t ticket) {
    for (ManagedResource *m = managed_resources; m; m = m->next)
        if (m->ticket == ticket) return m;
    return NULL;
}
static int on_owner(const ManagedResource *m) { return pthread_equal(m->owner, pthread_self()); }
static void managed_wake(ManagedResource *m) {
    (void)eshkol_event_loop_wake(m->loop_pointer);
}
static int managed_has_id(int64_t id) {
    pthread_mutex_lock(&managed_lock);
    int found = managed_find(id) != NULL;
    pthread_mutex_unlock(&managed_lock);
    return found;
}
static int managed_before_borrow(int64_t id) {
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(id);
    int rc = !m ? 0 : m->closing ? ESHKOL_IO_INVALID :
             !on_owner(m) ? ESHKOL_IO_BUSY : m->cancelled ? ESHKOL_IO_CANCELLED : 0;
    if (m && rc == ESHKOL_IO_CANCELLED && !m->sticky) m->cancelled = 0;
    pthread_mutex_unlock(&managed_lock);
    return rc;
}
static int managed_cancel(int64_t id) {
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(id);
    int rc = !m || m->closing ? ESHKOL_IO_INVALID : 0;
    if (!rc) { m->cancelled = 1; managed_wake(m); }
    pthread_mutex_unlock(&managed_lock);
    return rc;
}
static int queued_close(eshkol_handle_kind_t kind, uintptr_t native, void *data) {
    (void)kind; (void)native;
    ManagedResource *m = data;
    pthread_mutex_lock(&managed_lock);
    m->ready = 1;
    managed_wake(m);
    pthread_mutex_unlock(&managed_lock);
    return 0;
}
static void managed_maintenance(void *data) {
    (void)eshkol_resource_pump((int64_t)(intptr_t)data);
}
#ifdef ESHKOL_RESOURCE_TESTING
static int fail_ticket_once;
void eshkol_resource_test_fail_ticket_once(void) { fail_ticket_once = 1; }
#endif
static int ticket_close(eshkol_handle_kind_t kind, uintptr_t native, void *data) {
    (void)kind; (void)native; (void)data;
    return 0;
}
int64_t eshkol_resource_adopt_on_loop(eshkol_handle_kind_t kind, uint32_t tag,
    uintptr_t native, eshkol_resource_cleanup_fn cleanup, void *data,
    int64_t loop, int sticky) {
    eshkol_event_loop_t *target = eshkol_event_loop_from_handle(loop);
    if (!cleanup || !target || !eshkol_event_loop_is_owner(target)) return ESHKOL_IO_INVALID;
    ManagedResource *m = calloc(1, sizeof(*m));
    if (!m) return ESHKOL_IO_SYSTEM;
    m->wait_fd = -1;
    if (eshkol_event_loop_retain_handle(loop) != 0) {
        free(m); return ESHKOL_IO_SYSTEM;
    }
    m->owner = pthread_self(); m->loop = loop; m->loop_pointer = target; m->native = native;
    m->cleanup = cleanup; m->data = data; m->sticky = sticky != 0;
    m->id = eshkol_resource_adopt(kind, tag, native, queued_close, m);
    if (m->id < 0) {
        (void)eshkol_event_loop_release_handle(loop);
        int64_t rc = m->id; free(m); return rc;
    }
    eshkol_handle_id_t ticket = 0;
    int ticket_status;
#ifdef ESHKOL_RESOURCE_TESTING
    if (fail_ticket_once) { fail_ticket_once = 0; ticket_status = ESHKOL_HANDLE_ENOMEM; }
    else
#endif
    ticket_status = eshkol_handle_create_tagged(eshkol_resource_registry(),
        ESHKOL_HANDLE_KIND_CLEANUP, CLEANUP_TAG, (uintptr_t)m, ticket_close, NULL, &ticket);
    if (ticket_status) {
        /* Neither identity has been published. The registry callback queues
         * only; returning this failure leaves native/data owned by the caller. */
        (void)eshkol_handle_close(eshkol_resource_registry(), (eshkol_handle_id_t)m->id);
        (void)eshkol_event_loop_release_handle(loop);
        free(m); return ESHKOL_IO_SYSTEM;
    }
    m->ticket = (int64_t)ticket;
    if (eshkol_event_loop_set_maintenance(target, managed_maintenance, (void *)(intptr_t)loop)) {
        (void)eshkol_handle_close(eshkol_resource_registry(), (eshkol_handle_id_t)m->id);
        (void)eshkol_handle_close(eshkol_resource_registry(), ticket);
        (void)eshkol_event_loop_release_handle(loop);
        free(m); return ESHKOL_IO_BUSY;
    }
    pthread_mutex_lock(&managed_lock);
    m->next = managed_resources; managed_resources = m;
    pthread_mutex_unlock(&managed_lock);
    return m->id;
}
int eshkol_resource_check(const eshkol_resource_lease_t *lease) {
    if (!lease || !lease->lease) return ESHKOL_IO_INVALID;
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(lease->id);
    int rc = !m ? 0 : !on_owner(m) ? ESHKOL_IO_BUSY :
             m->closing || m->cancelled ? ESHKOL_IO_CANCELLED : 0;
    if (m && rc == ESHKOL_IO_CANCELLED && !m->closing && !m->sticky) m->cancelled = 0;
    pthread_mutex_unlock(&managed_lock);
    return rc;
}
static int managed_unwatch(ManagedResource *m) {
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(m->loop);
    if (!loop) return ESHKOL_IO_SYSTEM;
    if (m->wait_fd >= 0) {
        int rc = eshkol_event_loop_remove(loop, m->wait_fd);
        if (rc != ESHKOL_EVENT_LOOP_OK && rc != ESHKOL_EVENT_LOOP_ENOENT) return ESHKOL_IO_SYSTEM;
        m->wait_fd = -1; m->wait_events = 0;
    }
    return 0;
}
int eshkol_resource_wait_on_loop(const eshkol_resource_lease_t *lease,
    int fd, int events, int64_t deadline) {
    if (!lease || !lease->lease || fd < 0 ||
        !(events & (ESHKOL_RESOURCE_READ | ESHKOL_RESOURCE_WRITE))) return ESHKOL_IO_INVALID;
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(lease->id);
    if (!m || !on_owner(m)) { pthread_mutex_unlock(&managed_lock); return m ? ESHKOL_IO_BUSY : ESHKOL_IO_INVALID; }
    int rc = m->closing || m->cancelled ? ESHKOL_IO_CANCELLED : 0;
    if (rc && !m->closing && !m->sticky) m->cancelled = 0;
    pthread_mutex_unlock(&managed_lock);
    if (rc) return rc;
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(m->loop);
    if (!loop || managed_unwatch(m)) return ESHKOL_IO_SYSTEM;
    if (eshkol_event_loop_registration_present(loop, fd)) return ESHKOL_IO_BUSY;
    rc = eshkol_resource_check(lease);
    if (rc) return rc;
    int interest = events & ESHKOL_EVENT_ALL;
    if (eshkol_event_loop_add(loop, fd, interest, (uint64_t)m->id)) {
        (void)managed_unwatch(m); return ESHKOL_IO_SYSTEM;
    }
    m->wait_fd = fd; m->wait_events = interest;
    for (;;) {
        rc = eshkol_resource_check(lease);
        if (rc) break;
        int32_t left = eshkol_resource_remaining(deadline);
        if (left < 0) { rc = ESHKOL_IO_SYSTEM; break; }
        const eshkol_event_t *fired; int count;
        if (eshkol_event_loop_poll(loop, left, &fired, &count)) { rc = ESHKOL_IO_SYSTEM; break; }
        int ready = 0;
        for (int i = 0; i < count; ++i) {
            if (fired[i].fd == fd && fired[i].user_data == (uint64_t)m->id)
                ready |= (fired[i].events & interest) | ((fired[i].events & ESHKOL_EVENT_ERROR) ? interest : 0);
        }
        rc = eshkol_resource_check(lease);
        if (rc) break;
        if (ready) { rc = ready; break; }
        if (!left || eshkol_resource_remaining(deadline) == 0) { rc = ESHKOL_IO_TIMEOUT; break; }
    }
    if (managed_unwatch(m)) rc = ESHKOL_IO_SYSTEM;
    if (rc == ESHKOL_IO_CANCELLED && !m->sticky) {
        pthread_mutex_lock(&managed_lock); m->cancelled = 0; pthread_mutex_unlock(&managed_lock);
    }
    return rc;
}
int64_t eshkol_resource_close_request(int64_t id, eshkol_handle_kind_t kind, uint32_t mask) {
    eshkol_handle_registry_t *registry = eshkol_resource_registry();
    if (!resource_matches(registry, id, kind, mask, NULL)) return ESHKOL_IO_INVALID;
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(id);
    if (!m || m->closing) { pthread_mutex_unlock(&managed_lock); return ESHKOL_IO_INVALID; }
    int64_t ticket = m->ticket;
    m->closing = 1;
    m->result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_PENDING, 0, 0, 0};
    int64_t loop = m->loop;
    managed_wake(m);
    pthread_mutex_unlock(&managed_lock);
    int status = eshkol_handle_close_after_release(registry, (eshkol_handle_id_t)id);
    if (status != ESHKOL_HANDLE_OK) return ESHKOL_IO_SYSTEM;
    (void)eshkol_resource_pump(loop);
    return (int64_t)ticket;
}
static int managed_close_detached(int64_t id, eshkol_handle_kind_t kind, uint32_t mask) {
    int64_t ticket = eshkol_resource_close_request(id, kind, mask);
    if (ticket < 0) return (int)ticket;
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = ticket_find(ticket);
    if (m) m->detached = 1;
    int complete = m && m->result.state == ESHKOL_CLEANUP_COMPLETE;
    pthread_mutex_unlock(&managed_lock);
    if (complete) (void)eshkol_resource_cleanup_release(ticket);
    return 0;
}
int eshkol_resource_pump(int64_t loop_handle) {
    for (;;) {
        pthread_mutex_lock(&managed_lock);
        ManagedResource *m = managed_resources;
        while (m && (!m->closing || !m->ready || m->running || !on_owner(m) ||
            (loop_handle && m->loop != loop_handle) || m->result.state != ESHKOL_CLEANUP_PENDING)) m = m->next;
        if (!m) { pthread_mutex_unlock(&managed_lock); return 0; }
        m->running = 1;
        pthread_mutex_unlock(&managed_lock);
        eshkol_resource_cleanup_result_t result = {ESHKOL_CLEANUP_COMPLETE, 0, 0, 0};
        int rc = managed_unwatch(m);
        if (rc) {
            eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(m->loop);
            result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_FAILED,
                ESHKOL_CLEANUP_WATCH, loop ? eshkol_event_loop_last_os_error(loop) : 0, 1};
        } else if (!m->native_done) {
            rc = m->cleanup(m->native, m->data, &result);
            if (!rc) result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_COMPLETE, 0, 0, 0};
            else if (result.state != ESHKOL_CLEANUP_FAILED)
                result.state = ESHKOL_CLEANUP_FAILED;
            if (!result.retryable) m->native_done = 1;
        }
        if (m->native_done) {
            int release = eshkol_event_loop_release_handle(m->loop);
            if (release) result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_FAILED,
                ESHKOL_CLEANUP_LOOP, 0, 1};
        }
        pthread_mutex_lock(&managed_lock);
        m->result = result; m->running = 0;
        int64_t ticket = m->ticket;
        int detach = m->detached && result.state == ESHKOL_CLEANUP_COMPLETE;
        pthread_mutex_unlock(&managed_lock);
        if (detach) (void)eshkol_resource_cleanup_release(ticket);
    }
}
int eshkol_resource_cleanup_result(int64_t ticket, eshkol_resource_cleanup_result_t *out) {
    if (!out || !resource_matches(eshkol_resource_registry(), ticket, ESHKOL_HANDLE_KIND_CLEANUP,
                                  UINT32_C(1) << CLEANUP_TAG, NULL)) return ESHKOL_IO_INVALID;
    (void)eshkol_resource_pump(0);
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = ticket_find(ticket);
    if (m) *out = m->result;
    pthread_mutex_unlock(&managed_lock);
    return m ? 0 : ESHKOL_IO_INVALID;
}
int eshkol_resource_cleanup_retry(int64_t ticket) {
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = ticket_find(ticket);
    int rc = !m ? ESHKOL_IO_INVALID : !on_owner(m) ? ESHKOL_IO_BUSY :
        m->result.state != ESHKOL_CLEANUP_FAILED || !m->result.retryable ? ESHKOL_IO_INVALID : 0;
    if (!rc) m->result = (eshkol_resource_cleanup_result_t){ESHKOL_CLEANUP_PENDING, 0, 0, 0};
    pthread_mutex_unlock(&managed_lock);
    if (!rc) (void)eshkol_resource_pump(0);
    return rc;
}
int eshkol_resource_cleanup_release(int64_t ticket) {
    pthread_mutex_lock(&managed_lock);
    ManagedResource **link = &managed_resources;
    while (*link && (*link)->ticket != ticket) link = &(*link)->next;
    ManagedResource *m = *link;
    if (!m || m->running || m->result.state == ESHKOL_CLEANUP_PENDING || m->result.retryable) {
        pthread_mutex_unlock(&managed_lock); return m ? ESHKOL_IO_BUSY : ESHKOL_IO_INVALID;
    }
    int rc = eshkol_handle_close(eshkol_resource_registry(), (eshkol_handle_id_t)ticket);
    if (!rc) *link = m->next;
    pthread_mutex_unlock(&managed_lock);
    if (!rc) free(m);
    return rc ? ESHKOL_IO_INVALID : 0;
}
int64_t eshkol_resource_cleanup_ticket(int64_t retired_id) {
    pthread_mutex_lock(&managed_lock);
    ManagedResource *m = managed_find(retired_id);
    int64_t ticket = m && m->closing && m->detached &&
        m->result.state == ESHKOL_CLEANUP_FAILED ? m->ticket : ESHKOL_IO_INVALID;
    if (ticket > 0) m->detached = 0;
    pthread_mutex_unlock(&managed_lock);
    return ticket;
}
int eshkol_resource_managed_registration_count(int64_t handle) {
    eshkol_event_loop_t *loop = eshkol_event_loop_from_handle(handle);
    if (!loop || !eshkol_event_loop_is_owner(loop)) return -1;
    int count = 0;
    pthread_mutex_lock(&managed_lock);
    for (ManagedResource *m = managed_resources; m; m = m->next) {
        if (m->loop != handle) continue;
        if (m->wait_fd >= 0) {
            if (!eshkol_event_loop_registration_matches(loop, m->wait_fd, m->wait_events, (uint64_t)m->id)) { count = -1; break; }
            ++count;
        }
    }
    pthread_mutex_unlock(&managed_lock);
    return count;
}
#else
static int managed_has_id(int64_t id) { (void)id; return 0; }
static int managed_before_borrow(int64_t id) { (void)id; return 0; }
static int managed_cancel(int64_t id) { (void)id; return ESHKOL_IO_UNSUPPORTED; }
static int managed_close_detached(int64_t id, eshkol_handle_kind_t kind, uint32_t mask) { (void)id; (void)kind; (void)mask; return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_resource_adopt_on_loop(eshkol_handle_kind_t kind, uint32_t tag, uintptr_t native, eshkol_resource_cleanup_fn cleanup, void *data, int64_t loop, int sticky) { (void)kind; (void)tag; (void)native; (void)cleanup; (void)data; (void)loop; (void)sticky; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_check(const eshkol_resource_lease_t *lease) { (void)lease; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_wait_on_loop(const eshkol_resource_lease_t *lease, int fd, int events, int64_t deadline) { (void)lease; (void)fd; (void)events; (void)deadline; return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_resource_close_request(int64_t id, eshkol_handle_kind_t kind, uint32_t mask) { (void)id; (void)kind; (void)mask; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_cleanup_result(int64_t ticket, eshkol_resource_cleanup_result_t *out) { (void)ticket; (void)out; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_cleanup_retry(int64_t ticket) { (void)ticket; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_cleanup_release(int64_t ticket) { (void)ticket; return ESHKOL_IO_UNSUPPORTED; }
int64_t eshkol_resource_cleanup_ticket(int64_t id) { (void)id; return ESHKOL_IO_UNSUPPORTED; }
int eshkol_resource_pump(int64_t loop) { (void)loop; return 0; }
int eshkol_resource_managed_registration_count(int64_t loop) { (void)loop; return 0; }
#endif
