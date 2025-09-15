/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_LOGGER_H
#define ESHKOL_LOGGER_H

#include <stdarg.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    ESHKOL_FATAL,
    ESHKOL_ERROR,
    ESHKOL_WARNING,
    ESHKOL_NOTICE,
    ESHKOL_INFO,
    ESHKOL_DEBUG
} eshkol_logger_t;

void eshkol_set_logger_level(eshkol_logger_t level);
void eshkol_printf(eshkol_logger_t level, const char *msg, ...);
void eshkol_stacktrace(eshkol_logger_t level);

#define eshkol_fatal(...) eshkol_printf(ESHKOL_FATAL, __VA_ARGS__)
#define eshkol_error(...) eshkol_printf(ESHKOL_ERROR, __VA_ARGS__)
#define eshkol_warn(...) eshkol_printf(ESHKOL_WARNING, __VA_ARGS__)
#define eshkol_notice(...) eshkol_printf(ESHKOL_NOTICE, __VA_ARGS__)
#define eshkol_info(...) eshkol_printf(ESHKOL_INFO, __VA_ARGS__)
#define eshkol_debug(...) eshkol_printf(ESHKOL_DEBUG, __VA_ARGS__)

#define eshkol_fatal_stacktrace() eshkol_stacktrace(ESHKOL_FATAL);
#define eshkol_error_stacktrace() eshkol_stacktrace(ESHKOL_ERROR);
#define eshkol_warn_stacktrace() eshkol_stacktrace(ESHKOL_WARNING);
#define eshkol_notice_stacktrace() eshkol_stacktrace(ESHKOL_NOTICE);
#define eshkol_info_stacktrace() eshkol_stacktrace(ESHKOL_INFO);
#define eshkol_debug_stacktrace() eshkol_stacktrace(ESHKOL_DEBUG);

#ifdef __cplusplus
};
#endif

#endif
