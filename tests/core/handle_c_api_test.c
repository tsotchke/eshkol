/* SPDX-License-Identifier: MIT */
#include <eshkol/core/handle.h>

#include <stdio.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int close_file(eshkol_handle_kind_t kind, uintptr_t value, void *data) {
    int *calls = (int *)data;
    ++*calls;
    if (kind != ESHKOL_HANDLE_KIND_FILE) return -1;
    return fclose((FILE *)value);
}

int main(void) {
    eshkol_handle_registry_t *registry = eshkol_handle_registry_create();
    eshkol_handle_id_t id = 0;
    uintptr_t native = 0;
    uint64_t lease = 0;
    int calls = 0;
    FILE *file = tmpfile();
    CHECK(registry && file);
    /* A foreign C caller may supply an unknown kind. Failed adoption leaves
     * the ID and the caller's ownership untouched. */
    CHECK(eshkol_handle_create(registry, 99, (uintptr_t)file, close_file,
        &calls, &id) == ESHKOL_HANDLE_EINVAL);
    CHECK(id == 0 && calls == 0);
    CHECK(eshkol_handle_create(registry, ESHKOL_HANDLE_KIND_FILE,
        (uintptr_t)file, close_file, &calls, &id) == ESHKOL_HANDLE_OK);
    CHECK(eshkol_handle_borrow(registry, id, &native, &lease) == ESHKOL_HANDLE_OK);
    CHECK(fputs("C ABI resource round trip\n", (FILE *)native) >= 0);
    CHECK(eshkol_handle_release(registry, id, lease) == ESHKOL_HANDLE_OK);
    CHECK(eshkol_handle_registry_destroy(registry) == ESHKOL_HANDLE_OK);
    CHECK(calls == 1);
    puts("PASS: handle_c_api_test");
    return 0;
}
