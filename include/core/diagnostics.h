/**
 * @file diagnostics.h
 * @brief Diagnostics and error reporting system for Eshkol
 * 
 * This file defines the diagnostics interface for Eshkol,
 * which provides error reporting and diagnostic messages.
 */

#ifndef ESHKOL_DIAGNOSTICS_H
#define ESHKOL_DIAGNOSTICS_H

#include "core/memory.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Source location
 * 
 * Represents a location in the source code.
 */
typedef struct {
    const char* file;   /**< Source file name */
    int line;           /**< Line number (1-based) */
    int column;         /**< Column number (1-based) */
    int length;         /**< Length of the token or expression */
} SourceLocation;

/**
 * @brief Create a source location
 * 
 * @param file Source file name
 * @param line Line number (1-based)
 * @param column Column number (1-based)
 * @param length Length of the token or expression
 * @return Source location
 */
SourceLocation source_location_create(const char* file, int line, int column, int length);

/**
 * @brief Diagnostic severity
 * 
 * Represents the severity of a diagnostic message.
 */
typedef enum {
    DIAGNOSTIC_INFO,     /**< Informational message */
    DIAGNOSTIC_WARNING,  /**< Warning message */
    DIAGNOSTIC_ERROR,    /**< Error message */
    DIAGNOSTIC_FATAL     /**< Fatal error message */
} DiagnosticSeverity;

/**
 * @brief Diagnostic message
 * 
 * Represents a diagnostic message with severity, location, and message text.
 */
typedef struct {
    DiagnosticSeverity severity;  /**< Severity of the diagnostic */
    SourceLocation location;      /**< Location of the diagnostic */
    const char* message;          /**< Message text */
    const char* code;             /**< Optional error code */
} Diagnostic;

/**
 * @brief Diagnostic context
 * 
 * Manages a collection of diagnostic messages.
 */
typedef struct DiagnosticContext DiagnosticContext;

/**
 * @brief Create a diagnostic context
 * 
 * @param arena Arena to allocate from
 * @return A new diagnostic context, or NULL on failure
 */
DiagnosticContext* diagnostic_context_create(Arena* arena);

/**
 * @brief Add a diagnostic message
 * 
 * @param context The diagnostic context
 * @param severity Severity of the diagnostic
 * @param location Location of the diagnostic
 * @param message Message text
 * @param code Optional error code (can be NULL)
 */
void diagnostic_context_add(
    DiagnosticContext* context,
    DiagnosticSeverity severity,
    SourceLocation location,
    const char* message,
    const char* code);

/**
 * @brief Get the number of diagnostic messages
 * 
 * @param context The diagnostic context
 * @return Number of diagnostic messages
 */
size_t diagnostic_context_get_count(DiagnosticContext* context);

/**
 * @brief Get a diagnostic message
 * 
 * @param context The diagnostic context
 * @param index Index of the diagnostic message
 * @return The diagnostic message, or NULL if index is out of bounds
 */
const Diagnostic* diagnostic_context_get(DiagnosticContext* context, size_t index);

/**
 * @brief Get the number of error and fatal messages
 * 
 * @param context The diagnostic context
 * @return Number of error and fatal messages
 */
size_t diagnostic_context_get_error_count(DiagnosticContext* context);

/**
 * @brief Check if there are any error or fatal messages
 * 
 * @param context The diagnostic context
 * @return true if there are any error or fatal messages, false otherwise
 */
bool diagnostic_context_has_errors(DiagnosticContext* context);

/**
 * @brief Print all diagnostic messages
 * 
 * @param context The diagnostic context
 */
void diagnostic_context_print(DiagnosticContext* context);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_DIAGNOSTICS_H */
