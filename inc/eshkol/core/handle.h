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
eshkol_handle_registry_t *eshkol_handle_registry_create(void);
/* If busy, leaves registry alive. Otherwise frees it after closing open entries. */
int eshkol_handle_registry_destroy(eshkol_handle_registry_t *registry);
/* close_fn is required. On success, the registry becomes the sole owner of
 * native_value and invokes close_fn exactly once. */
int eshkol_handle_create(eshkol_handle_registry_t *registry,
    eshkol_handle_kind_t kind, uintptr_t native_value,
    eshkol_handle_close_fn close_fn, void *user_data, eshkol_handle_id_t *out_id);
/* Outputs are written only on success. Borrow lease values are registry-local. */
int eshkol_handle_borrow(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uintptr_t *out_native_value, uint64_t *out_lease);
/* Release must present the lease returned by that specific borrow. */
int eshkol_handle_release(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, uint64_t lease);
int eshkol_handle_close(eshkol_handle_registry_t *registry, eshkol_handle_id_t id);
int eshkol_handle_get_state(eshkol_handle_registry_t *registry,
    eshkol_handle_id_t id, eshkol_handle_state_t *out_state);
#ifdef __cplusplus
}
#endif
#endif
