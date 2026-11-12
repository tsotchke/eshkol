/**
 * @file handle.h
 * @brief Generation-tagged ownership handles for native resources.
 *
 * The registry is synchronized and stays outside movable arena storage.
 * Borrowing is exclusive (OPEN -> BORROWED -> OPEN). Close while borrowed
 * returns EBUSY. Close marks CLOSING under the lock, invokes the callback
 * without the lock, and becomes terminal CLOSED even if the callback fails;
 * uncertain OS closes must never be retried. Destroy rejects borrowed or
 * closing entries, otherwise closes each remaining open entry once outside
 * the lock. A create call transfers sole native-resource ownership only when
 * it returns OK; on any failure, ownership stays with the caller. IDs and
 * borrow leases are registry-local. Destroy returns ECLOSE if any callback
 * fails but still completes cleanup and frees the registry. The owner must
 * synchronize registry lifetime against external operations; close callbacks
 * may re-enter while destruction is in progress.
 *
 * Failure outputs are left untouched. A successful state query may report
 * CLOSED until its slot is reused; expired IDs after reuse are invalid. Lease
 * values are never recycled; exhaustion makes future borrows fail. Slots at
 * the maximum generation are retired instead of wrapping to an old ID.
 *
 * Interruption and close-after-release let one thread end another thread's
 * wait without invalidating its borrow: interrupt marks the identity and wakes
 * the borrower's registered waker; close-after-release additionally makes the
 * identity unusable at once and runs the close callback exactly once when the
 * active borrower releases. Exact registries keep every identity at or below
 * 2^53 - 1 so it survives binary64 numeric boundaries unchanged.
 */
#ifndef ESHKOL_CORE_HANDLE_H
#define ESHKOL_CORE_HANDLE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct eshkol_handle_registry eshkol_handle_registry_t;
typedef uint64_t eshkol_handle_id_t;
/* An open integer input at the C boundary: unknown kinds return EINVAL. */
typedef uint32_t eshkol_handle_kind_t;
enum {
    ESHKOL_HANDLE_KIND_INVALID = 0,
    ESHKOL_HANDLE_KIND_FILE = 1,
    ESHKOL_HANDLE_KIND_SOCKET = 2,
    ESHKOL_HANDLE_KIND_EVENT_LOOP = 3,
    /* Preserve published platform values; PROCESS follows the platform kinds. */
    ESHKOL_HANDLE_KIND_WINDOW = 4,
    ESHKOL_HANDLE_KIND_AUDIO_DEVICE = 5,
    ESHKOL_HANDLE_KIND_MIDI_PORT = 6,
    ESHKOL_HANDLE_KIND_PROCESS = 7
};

typedef enum {
    ESHKOL_HANDLE_STATE_INVALID = 0,
    ESHKOL_HANDLE_STATE_OPEN = 1,
    ESHKOL_HANDLE_STATE_BORROWED = 2,
    ESHKOL_HANDLE_STATE_CLOSING = 3,
    ESHKOL_HANDLE_STATE_CLOSED = 4
} eshkol_handle_state_t;

typedef enum {
    ESHKOL_HANDLE_OK = 0,
    ESHKOL_HANDLE_EINVAL = -1,
    ESHKOL_HANDLE_ENOMEM = -2,
    ESHKOL_HANDLE_ENOENT = -3,
    ESHKOL_HANDLE_ESTATE = -4,
    ESHKOL_HANDLE_EBUSY = -5,
    ESHKOL_HANDLE_ECLOSE = -6,
    ESHKOL_HANDLE_EEXHAUSTED = -7,
    ESHKOL_HANDLE_EINTERRUPTED = -8
} eshkol_handle_status_t;
/* Called under the registry lock by interrupt and close-after-release. It must
 * only signal (for example write one byte to a nonblocking pipe) and must not
 * call back into the registry. */
typedef void (*eshkol_handle_wake_fn)(void *context);
/* Zero means success; nonzero is reported as ECLOSE. */
typedef int (*eshkol_handle_close_fn)(eshkol_handle_kind_t kind,
                                      uintptr_t native_value, void *user_data);
/** @brief Create an empty hosted registry, or return NULL on allocation failure. */
eshkol_handle_registry_t *eshkol_handle_registry_create(void);
/**
 * @brief Create a registry whose identities never exceed 2^53 - 1.
 * Slots retire after 2^21 - 1 generations instead of after 2^32 - 1.
 */
eshkol_handle_registry_t *eshkol_handle_registry_create_exact(void);
/**
 * @brief Close all remaining owners and free a registry without allocating.
 * @return OK or ECLOSE after freeing; EBUSY leaves the registry alive when a
 * borrow or close is outstanding. NULL is EINVAL. The caller synchronizes
 * registry lifetime against external operations. Callbacks may re-enter.
 */
int eshkol_handle_registry_destroy(eshkol_handle_registry_t *registry);
/**
 * @brief Adopt sole ownership of a native resource with a required close callback.
 * @param native_value Opaque pointer or descriptor, valid for the supplied kind.
 * @param user_data Caller context kept alive until the close callback completes.
 * @param out_id Receives the registry-local identity only on success.
 * @return OK transfers ownership; every failure leaves ownership with the caller.
 */
int eshkol_handle_create(eshkol_handle_registry_t *registry,
    eshkol_handle_kind_t kind, uintptr_t native_value,
    eshkol_handle_close_fn close_fn, void *user_data, eshkol_handle_id_t *out_id);
/**
 * @brief Adopt a resource as eshkol_handle_create does, with an immutable
 * resource-defined subtype tag (for example a TCP listener versus a stream).
 */
int eshkol_handle_create_tagged(eshkol_handle_registry_t *registry,
    eshkol_handle_kind_t kind, uint32_t tag, uintptr_t native_value,
    eshkol_handle_close_fn close_fn, void *user_data, eshkol_handle_id_t *out_id);
/**
 * @brief Report the kind and tag of a live identity without borrowing it.
 * @return OK, or ENOENT for unknown, retired, closed or close-requested IDs.
 */
int eshkol_handle_get_kind(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, eshkol_handle_kind_t *out_kind, uint32_t *out_tag);
/**
 * @brief Exclusively borrow an open owner until its matching lease is released.
 * @param out_native_value Native resource, usable only during this borrow.
 * @param out_lease Unique lease required by release; keep outputs distinct.
 * @return OK, EBUSY for an active borrow/close, or an invalid/exhausted identity
 * error. Failure leaves both outputs untouched.
 */
int eshkol_handle_borrow(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uintptr_t *out_native_value, uint64_t *out_lease);
/**
 * @brief End the specific borrow identified by the owner and its lease.
 * @return OK or an error without ending another borrow. A stale or repeated
 * release cannot release a newer borrower. Stop using the native value first.
 */
int eshkol_handle_release(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uint64_t lease);
/**
 * @brief Attempt an open owner's close callback exactly once, outside the lock.
 * @return OK or ECLOSE leaves the identity terminally closed; EBUSY leaves an
 * outstanding borrower/closer intact. Invalid or previously closed IDs are ENOENT.
 */
int eshkol_handle_close(eshkol_handle_registry_t *registry, eshkol_handle_id_t id);
/**
 * @brief Close now when open; when borrowed, refuse new use at once, interrupt
 * the borrower, and run the close callback exactly once on its release.
 * @return OK (closed now or close scheduled), ECLOSE when an immediate close
 * callback failed, EBUSY while another close runs, ENOENT when retired.
 */
int eshkol_handle_close_after_release(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id);
/**
 * @brief Mark a live identity interrupted and wake its borrower, if any.
 * The mark persists until the next borrower consumes it.
 * @return OK, or ENOENT for unknown, retired or closed IDs.
 */
int eshkol_handle_interrupt(eshkol_handle_registry_t *registry, eshkol_handle_id_t id);
/**
 * @brief Install (or clear with NULL) the active borrower's waker.
 * @return OK, EINTERRUPTED when an interrupt is already pending (the waker is
 * still installed), or ESTATE when the lease is not the active borrow.
 */
int eshkol_handle_set_waker(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uint64_t lease, eshkol_handle_wake_fn wake_fn, void *context);
/**
 * @brief Consume a pending interrupt for the active borrow.
 * @return 1 when one was pending, 0 when none, or a negative status.
 */
int eshkol_handle_take_interrupt(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uint64_t lease);
/**
 * @brief Query state without borrowing or exposing the native resource.
 * @return OK for a known identity, including CLOSED before slot reuse. Invalid
 * or stale identities return ENOENT without changing the output.
 */
int eshkol_handle_get_state(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, eshkol_handle_state_t *out_state);
#ifdef __cplusplus
}
#endif
#endif
