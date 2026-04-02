/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Eshkol Logger
 *
 * Supports two output formats:
 * - TEXT: Human-readable colored output (default)
 * - JSON: Structured JSON for production/log aggregation
 *
 * JSON format example:
 * {"timestamp":"2025-01-06T10:30:00Z","level":"ERROR","msg":"Type mismatch","file":"foo.esk","line":42}
 */
#ifndef ESHKOL_LOGGER_H
#define ESHKOL_LOGGER_H

#include <stdarg.h>
#include <stddef.h>
#include <stdbool.h>

// Include config.h for eshkol_log_format_t (canonical definition)
#include <eshkol/core/config.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Log Levels
// ============================================================================

typedef enum {
    ESHKOL_FATAL,
    ESHKOL_ERROR,
    ESHKOL_WARNING,
    ESHKOL_NOTICE,
    ESHKOL_INFO,
    ESHKOL_DEBUG
} eshkol_logger_t;

// ============================================================================
// Log Formats
// ============================================================================

// eshkol_log_format_t is defined in config.h with values:
// - ESHKOL_LOG_FORMAT_TEXT: Human-readable text with colors
// - ESHKOL_LOG_FORMAT_JSON: Structured JSON (one object per line)

// Legacy aliases for backward compatibility
#define ESHKOL_LOG_TEXT ESHKOL_LOG_FORMAT_TEXT
#define ESHKOL_LOG_JSON ESHKOL_LOG_FORMAT_JSON

// ============================================================================
// Configuration
// ============================================================================

// Set minimum log level (messages below this level are suppressed)
void eshkol_set_logger_level(eshkol_logger_t level);

// Get current log level
eshkol_logger_t eshkol_get_logger_level(void);

// Set output format (TEXT or JSON)
void eshkol_set_log_format(eshkol_log_format_t format);

// Get current output format
eshkol_log_format_t eshkol_get_log_format(void);

// Set log output file (NULL = stderr)
// Returns: 0 on success, -1 on error
int eshkol_set_log_file(const char* path);

// Enable/disable colored output (TEXT format only)
void eshkol_set_color_output(bool enabled);

// Enable/disable timestamps in output
void eshkol_set_timestamps(bool enabled);

// ============================================================================
// Basic Logging
// ============================================================================

// Log a message at the specified level
void eshkol_printf(eshkol_logger_t level, const char *msg, ...);

// Log a stacktrace at the specified level
void eshkol_stacktrace(eshkol_logger_t level);

// ============================================================================
// Structured Logging (for JSON format)
// ============================================================================

// Log with source location context
void eshkol_log_with_location(eshkol_logger_t level,
                               const char* file,
                               int line,
                               const char* func,
                               const char* msg, ...);

// Log with key-value pairs (JSON format)
// Keys and values are alternating: key1, val1, key2, val2, ..., NULL
void eshkol_log_structured(eshkol_logger_t level,
                            const char* msg,
                            ...);  // key-value pairs, NULL terminated

// ============================================================================
// Convenience Macros
// ============================================================================

#define eshkol_fatal(...) eshkol_printf(ESHKOL_FATAL, __VA_ARGS__)
#define eshkol_error(...) eshkol_printf(ESHKOL_ERROR, __VA_ARGS__)
#define eshkol_warn(...) eshkol_printf(ESHKOL_WARNING, __VA_ARGS__)
#define eshkol_notice(...) eshkol_printf(ESHKOL_NOTICE, __VA_ARGS__)
#define eshkol_info(...) eshkol_printf(ESHKOL_INFO, __VA_ARGS__)
#define eshkol_debug(...) eshkol_printf(ESHKOL_DEBUG, __VA_ARGS__)

#define eshkol_fatal_stacktrace() eshkol_stacktrace(ESHKOL_FATAL)
#define eshkol_error_stacktrace() eshkol_stacktrace(ESHKOL_ERROR)
#define eshkol_warn_stacktrace() eshkol_stacktrace(ESHKOL_WARNING)
#define eshkol_notice_stacktrace() eshkol_stacktrace(ESHKOL_NOTICE)
#define eshkol_info_stacktrace() eshkol_stacktrace(ESHKOL_INFO)
#define eshkol_debug_stacktrace() eshkol_stacktrace(ESHKOL_DEBUG)

// Logging with automatic source location
#define ESHKOL_LOG(level, ...) \
    eshkol_log_with_location(level, __FILE__, __LINE__, __func__, __VA_ARGS__)

#define ESHKOL_FATAL(...) ESHKOL_LOG(ESHKOL_FATAL, __VA_ARGS__)
#define ESHKOL_ERROR(...) ESHKOL_LOG(ESHKOL_ERROR, __VA_ARGS__)
#define ESHKOL_WARN(...) ESHKOL_LOG(ESHKOL_WARNING, __VA_ARGS__)
#define ESHKOL_INFO(...) ESHKOL_LOG(ESHKOL_INFO, __VA_ARGS__)
#define ESHKOL_DEBUG(...) ESHKOL_LOG(ESHKOL_DEBUG, __VA_ARGS__)

// ============================================================================
// Utility Functions
// ============================================================================

// Get log level name as string
const char* eshkol_log_level_name(eshkol_logger_t level);

// Flush log output
void eshkol_log_flush(void);

// Close log file (if open)
void eshkol_log_close(void);

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_LOGGER_H
