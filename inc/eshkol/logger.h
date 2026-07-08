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

/**
 * @brief Log severity levels, ordered from most to least severe.
 *
 * A message is emitted only if its level is numerically less than or equal
 * to the current logger level (see eshkol_set_logger_level());  higher
 * numeric values are progressively more verbose and are suppressed first.
 */
typedef enum {
    ESHKOL_FATAL,   // Unrecoverable error; eshkol_printf/eshkol_log_* variants call exit(1) after logging at this level
    ESHKOL_ERROR,   // Recoverable error
    ESHKOL_WARNING, // Potential problem, execution continues
    ESHKOL_NOTICE,  // Default verbosity: noteworthy events under normal operation
    ESHKOL_INFO,    // Informational detail
    ESHKOL_DEBUG    // Verbose diagnostic detail
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

/**
 * @brief Sets the minimum log level; messages less severe (numerically
 *        greater) than this level are suppressed.
 *
 * @param level New maximum verbosity level (default is ESHKOL_NOTICE).
 * @note Thread-safe; stored in a lock-free atomic so that suppressed log
 *       calls can early-return without taking the logger's mutex.
 */
void eshkol_set_logger_level(eshkol_logger_t level);

/**
 * @brief Returns the currently configured minimum log level.
 * @return The current eshkol_logger_t level threshold.
 */
eshkol_logger_t eshkol_get_logger_level(void);

/**
 * @brief Sets the output format used by all subsequent log calls.
 * @param format ESHKOL_LOG_FORMAT_TEXT for human-readable colored output,
 *        or ESHKOL_LOG_FORMAT_JSON for structured one-object-per-line JSON.
 * @note Thread-safe; guarded by the logger's internal mutex.
 */
void eshkol_set_log_format(eshkol_log_format_t format);

/**
 * @brief Returns the currently configured output format.
 * @return The current eshkol_log_format_t value.
 */
eshkol_log_format_t eshkol_get_log_format(void);

/**
 * @brief Redirects log output to a file, or back to stderr.
 *
 * If a previously opened log file is owned by the logger, it is closed
 * before switching. The new file is opened in append mode.
 *
 * @param path Path to the log file, or NULL to reset output to stderr.
 * @return 0 on success; -1 if path is non-NULL and the file could not be
 *         opened.
 * @note Thread-safe; guarded by the logger's internal mutex.
 */
int eshkol_set_log_file(const char* path);

/**
 * @brief Enables or disables ANSI color codes in TEXT-format output.
 * @param enabled true to colorize level prefixes, false for plain text.
 * @note Has no effect on JSON-format output. Thread-safe.
 */
void eshkol_set_color_output(bool enabled);

/**
 * @brief Enables or disables timestamp output.
 * @param enabled true to include timestamps, false to omit them.
 * @note Thread-safe.
 */
void eshkol_set_timestamps(bool enabled);

// ============================================================================
// Basic Logging
// ============================================================================

/**
 * @brief Logs a printf-style formatted message at the given severity level.
 *
 * Suppressed (level more verbose than the current threshold) calls
 * early-return without formatting the message or taking any lock. If
 * level is ESHKOL_FATAL, the process calls exit(1) after the message is
 * emitted.
 *
 * @param level Severity level for this message.
 * @param msg printf-style format string.
 * @param ... Format arguments matching msg.
 * @note Thread-safe; the message body is formatted into a fixed 4096-byte
 *       buffer, so overly long messages are truncated.
 */
#ifdef __GNUC__
__attribute__((format(printf, 2, 3)))
#endif
void eshkol_printf(eshkol_logger_t level, const char *msg, ...);

/**
 * @brief Prints the current call stack at the given severity level.
 *
 * On suppressed levels this is a no-op. On macOS/Linux this uses
 * backtrace()/backtrace_symbols() with best-effort C++ demangling; on
 * Windows it uses dbghelp.dll (loaded lazily) to resolve symbol names and
 * source lines; on other platforms it falls back to logging a
 * "not available" message. If level is ESHKOL_FATAL, the process calls
 * exit(1) after printing.
 *
 * @param level Severity level to print the stacktrace at, and to gate
 *        suppression against the current log level threshold.
 */
void eshkol_stacktrace(eshkol_logger_t level);

// ============================================================================
// Structured Logging (for JSON format)
// ============================================================================

/**
 * @brief Logs a printf-style formatted message annotated with a source
 *        location (file, line, function).
 *
 * In JSON format, file/line/func populate the corresponding JSON fields
 * (omitted when file/func are NULL or line <= 0). In TEXT format, the
 * location is appended after the message as "[file:line]" when both file
 * is non-NULL and line > 0. Suppressed calls early-return without
 * formatting. If level is ESHKOL_FATAL, the process calls exit(1) after
 * logging. Typically invoked via the ESHKOL_LOG/ESHKOL_ERROR/etc. macros,
 * which supply __FILE__, __LINE__, and __func__ automatically.
 *
 * @param level Severity level for this message.
 * @param file Source file name, or NULL to omit.
 * @param line Source line number, or <= 0 to omit.
 * @param func Enclosing function name, or NULL to omit.
 * @param msg printf-style format string.
 * @param ... Format arguments matching msg.
 * @note Thread-safe.
 */
void eshkol_log_with_location(eshkol_logger_t level,
                               const char* file,
                               int line,
                               const char* func,
                               const char* msg, ...);

/**
 * @brief Logs a message together with a NULL-terminated list of key/value
 *        string pairs, rendered as structured JSON when JSON format is
 *        active.
 *
 * Arguments after msg must alternate as key1, val1, key2, val2, ... and be
 * terminated by a NULL key or NULL value. In TEXT format, the key/value
 * pairs are currently ignored and only the base message is printed.
 * Suppressed calls early-return without formatting. If level is
 * ESHKOL_FATAL, the process calls exit(1) after logging.
 *
 * @param level Severity level for this message.
 * @param msg Base log message (not a printf format string).
 * @param ... Alternating `const char*` key/value pairs, NULL-terminated.
 * @note Thread-safe.
 */
void eshkol_log_structured(eshkol_logger_t level,
                            const char* msg,
                            ...);  // key-value pairs, NULL terminated

// ============================================================================
// Convenience Macros
// ============================================================================

// Source-location-aware error reporting (clang/gcc diagnostic style)
// Prints: file:line:col: error: message
//         <source line>
//         ~~~~^~~~~
// If source_text is NULL, no caret line is printed.
// If file is NULL, uses "<unknown>".
/**
 * @brief Prints a clang/gcc-style diagnostic at ESHKOL_ERROR severity.
 *
 * Formats "file:line:col: error: message", followed by the offending
 * source line and a caret ('^') under the given column when source_text
 * is provided. This is independent of eshkol_set_logger_level()/JSON
 * format — it always writes directly to the current log output stream
 * using the diagnostic layout, not the standard TEXT/JSON encoders.
 *
 * @param file Source file name to report, or NULL to print "<unknown>".
 * @param line 1-based source line number, or 0 to omit line:col from the
 *        header and skip the source/caret lines.
 * @param column 1-based source column number, or 0 to omit the caret line.
 * @param source_text Full source text to extract the reported line from,
 *        or NULL to skip the source/caret lines entirely.
 * @param msg printf-style format string for the diagnostic message.
 * @param ... Format arguments matching msg.
 */
void eshkol_error_at(const char* file, unsigned line, unsigned column,
                     const char* source_text, const char* msg, ...);

/**
 * @brief Prints a clang/gcc-style diagnostic at ESHKOL_WARNING severity.
 *
 * Identical in behavior and formatting to eshkol_error_at(), except the
 * diagnostic is labeled "warning" instead of "error" and colored
 * accordingly.
 *
 * @param file Source file name to report, or NULL to print "<unknown>".
 * @param line 1-based source line number, or 0 to omit line:col from the
 *        header and skip the source/caret lines.
 * @param column 1-based source column number, or 0 to omit the caret line.
 * @param source_text Full source text to extract the reported line from,
 *        or NULL to skip the source/caret lines entirely.
 * @param msg printf-style format string for the diagnostic message.
 * @param ... Format arguments matching msg.
 */
void eshkol_warn_at(const char* file, unsigned line, unsigned column,
                    const char* source_text, const char* msg, ...);

/**
 * @brief Convenience macros wrapping eshkol_printf() for each severity
 *        level; each forwards its arguments as a printf-style format
 *        string and varargs (e.g. eshkol_error("failed: %d", code)).
 *        eshkol_fatal() terminates the process via exit(1) after logging.
 */
#define eshkol_fatal(...) eshkol_printf(ESHKOL_FATAL, __VA_ARGS__)
#define eshkol_error(...) eshkol_printf(ESHKOL_ERROR, __VA_ARGS__)
#define eshkol_warn(...) eshkol_printf(ESHKOL_WARNING, __VA_ARGS__)
#define eshkol_notice(...) eshkol_printf(ESHKOL_NOTICE, __VA_ARGS__)
#define eshkol_info(...) eshkol_printf(ESHKOL_INFO, __VA_ARGS__)
#define eshkol_debug(...) eshkol_printf(ESHKOL_DEBUG, __VA_ARGS__)

/**
 * @brief Convenience macros wrapping eshkol_stacktrace() for each severity
 *        level. eshkol_fatal_stacktrace() terminates the process via
 *        exit(1) after printing.
 */
#define eshkol_fatal_stacktrace() eshkol_stacktrace(ESHKOL_FATAL)
#define eshkol_error_stacktrace() eshkol_stacktrace(ESHKOL_ERROR)
#define eshkol_warn_stacktrace() eshkol_stacktrace(ESHKOL_WARNING)
#define eshkol_notice_stacktrace() eshkol_stacktrace(ESHKOL_NOTICE)
#define eshkol_info_stacktrace() eshkol_stacktrace(ESHKOL_INFO)
#define eshkol_debug_stacktrace() eshkol_stacktrace(ESHKOL_DEBUG)

/**
 * @brief Logs a printf-style message via eshkol_log_with_location(),
 *        automatically capturing the call site's __FILE__, __LINE__, and
 *        __func__.
 * @param level Severity level for this message (an eshkol_logger_t value).
 * @param ... printf-style format string followed by its format arguments.
 */
#define ESHKOL_LOG(level, ...) \
    eshkol_log_with_location(level, __FILE__, __LINE__, __func__, __VA_ARGS__)

/**
 * @brief Per-level shorthands for ESHKOL_LOG() that also capture source
 *        location automatically (e.g. ESHKOL_ERROR("bad value: %d", v)).
 *
 * @note These share their names with the corresponding eshkol_logger_t
 *       enumerators (ESHKOL_FATAL, ESHKOL_ERROR, ...). Because these are
 *       function-like macros, they only expand when followed by '(';
 *       used elsewhere, the identifier still refers to the enum constant.
 */
#define ESHKOL_FATAL(...) ESHKOL_LOG(ESHKOL_FATAL, __VA_ARGS__)
#define ESHKOL_ERROR(...) ESHKOL_LOG(ESHKOL_ERROR, __VA_ARGS__)
#define ESHKOL_WARN(...) ESHKOL_LOG(ESHKOL_WARNING, __VA_ARGS__)
#define ESHKOL_INFO(...) ESHKOL_LOG(ESHKOL_INFO, __VA_ARGS__)
#define ESHKOL_DEBUG(...) ESHKOL_LOG(ESHKOL_DEBUG, __VA_ARGS__)

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Returns the display name for a log level (e.g. "ERROR").
 * @param level Level to look up.
 * @return A statically-allocated, non-owned string naming the level, or
 *         "UNKNOWN" if level is out of range. Never returns NULL.
 */
const char* eshkol_log_level_name(eshkol_logger_t level);

/**
 * @brief Flushes the current log output stream (stderr or the configured
 *        log file).
 * @note Thread-safe; guarded by the logger's internal mutex.
 */
void eshkol_log_flush(void);

/**
 * @brief Closes the log file previously opened via eshkol_set_log_file(),
 *        if any, and reverts output to stderr.
 *
 * No-op if no file is currently open or output was never redirected from
 * stderr.
 * @note Thread-safe; guarded by the logger's internal mutex.
 */
void eshkol_log_close(void);

#ifdef __cplusplus
}
#endif

#endif // ESHKOL_LOGGER_H
