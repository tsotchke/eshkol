/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Hosted runtime diagnostic routing.
 */

#include <eshkol/core/runtime.h>
#include <eshkol/logger.h>

extern "C" void eshkol_runtime_default_diagnostic_sink(
    eshkol_runtime_diagnostic_level_t level,
    const char* message) {
    eshkol_logger_t logger_level = ESHKOL_DEBUG;
    switch (level) {
        case ESHKOL_RUNTIME_DIAGNOSTIC_DEBUG:
            logger_level = ESHKOL_DEBUG;
            break;
        case ESHKOL_RUNTIME_DIAGNOSTIC_INFO:
            logger_level = ESHKOL_INFO;
            break;
        case ESHKOL_RUNTIME_DIAGNOSTIC_WARNING:
            logger_level = ESHKOL_WARNING;
            break;
        case ESHKOL_RUNTIME_DIAGNOSTIC_ERROR:
            logger_level = ESHKOL_ERROR;
            break;
        default:
            logger_level = ESHKOL_ERROR;
            break;
    }

    eshkol_printf(logger_level, "%s",
                  message ? message : "Runtime diagnostic");
}
