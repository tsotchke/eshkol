#ifndef ESHKOL_KEYCHAIN_NATIVE_H
#define ESHKOL_KEYCHAIN_NATIVE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Internal bounded ABI for lib/agent/keychain.esk. */
int64_t eshkol_keychain_size(const void *account, int64_t account_len);
int64_t eshkol_keychain_get_into(const void *account, int64_t account_len,
                                 void *output, int64_t output_capacity);
int eshkol_keychain_set(const void *account, int64_t account_len,
                        const void *value, int64_t value_len);
int eshkol_keychain_delete(const void *account, int64_t account_len);
int eshkol_keychain_has(const void *account, int64_t account_len);

#ifdef __cplusplus
}
#endif

#endif
