/* SPDX-License-Identifier: MIT */
#ifndef ESHKOL_UNIX_SOCKET_FIXTURE_SUPPORT_H
#define ESHKOL_UNIX_SOCKET_FIXTURE_SUPPORT_H
#include <stdint.h>
int32_t eshkol_unix_socket_test_fd_count(void);
int64_t eshkol_unix_socket_test_monotonic_ms(void);
int32_t eshkol_unix_socket_test_sleep_ms(int32_t ms);
int32_t eshkol_unix_socket_test_errno(int32_t which);
int32_t eshkol_unix_socket_test_pending_path(const char *path);
#endif
