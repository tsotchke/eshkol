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
 * CLOSED until its slot is reused; stale IDs after reuse are invalid. Lease
 * values are never recycled; exhaustion makes future borrows fail. Slots at
 * the maximum generation are retired instead of wrapping to an old ID.
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
    ESHKOL_HANDLE_KIND_EVENT_LOOP = 3
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
    ESHKOL_HANDLE_EEXHAUSTED = -7
} eshkol_handle_status_t;
/* Zero means success; nonzero is reported as ECLOSE. */
typedef int (*eshkol_handle_close_fn)(eshkol_handle_kind_t kind,
                                      uintptr_t native_value, void *user_data);
/** @brief Create an empty hosted registry, or return NULL on allocation failure. */
eshkol_handle_registry_t *eshkol_handle_registry_create(void);
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
