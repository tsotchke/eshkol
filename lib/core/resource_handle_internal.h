/* Private registry metadata for shared resource dispatch. Not an installed ABI. */
#ifndef ESHKOL_RESOURCE_HANDLE_INTERNAL_H
#define ESHKOL_RESOURCE_HANDLE_INTERNAL_H
#include "eshkol/core/handle.h"
#ifdef __cplusplus
extern "C" {
#endif
#define ESHKOL_HANDLE_FLAG_MANAGED UINT32_C(1)
int eshkol_handle_create_resource_tagged(eshkol_handle_registry_t *,
    eshkol_handle_kind_t, uint32_t tag, uint32_t flags, uintptr_t native,
    eshkol_handle_close_fn, void *, eshkol_handle_id_t *);
int eshkol_handle_resource_info(eshkol_handle_registry_t *, eshkol_handle_id_t,
    eshkol_handle_kind_t *, uint32_t *tag, uint32_t *flags);
int eshkol_handle_resource_flags(eshkol_handle_registry_t *, eshkol_handle_id_t, uint32_t *flags);
int eshkol_handle_release_resource(eshkol_handle_registry_t *, eshkol_handle_id_t,
    uint64_t lease, uint32_t *flags);
/* Transfer an internal cleanup ticket to detached success/failure recovery. */
int eshkol_resource_cleanup_detach(int64_t ticket);
#ifdef __cplusplus
}
#endif
#endif
