/**
 * @file diagnostics.c
 * @brief Implementation of the diagnostics system
 */

#include "core/diagnostics.h"
#include "core/memory.h"
#include "core/string_table.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>

/**
 * @brief Dynamic array of diagnostics
 */
typedef struct {
    Diagnostic* items;   /**< Array of diagnostics */
    size_t count;        /**< Number of diagnostics */
    size_t capacity;     /**< Capacity of the array */
} DiagnosticArray;

/**
 * @brief Diagnostic context structure
 */
struct DiagnosticContext {
    Arena* arena;                /**< Arena for allocations */
    DiagnosticArray diagnostics; /**< Array of diagnostics */
    size_t error_count;          /**< Number of error and fatal messages */
};

SourceLocation source_location_create(const char* file, int line, int column, int length) {
    SourceLocation location;
    location.file = file;
    location.line = line;
    location.column = column;
    location.length = length;
    return location;
}

DiagnosticContext* diagnostic_context_create(Arena* arena) {
    assert(arena != NULL);
    
    // Allocate context
    DiagnosticContext* context = arena_alloc(arena, sizeof(DiagnosticContext));
    if (!context) return NULL;
    
    // Initialize context
    context->arena = arena;
    context->diagnostics.items = NULL;
    context->diagnostics.count = 0;
    context->diagnostics.capacity = 0;
    context->error_count = 0;
    
    return context;
}

void diagnostic_context_add(
    DiagnosticContext* context,
    DiagnosticSeverity severity,
    SourceLocation location,
    const char* message,
    const char* code) {
    
    assert(context != NULL);
    assert(message != NULL);
    
    // Check if we need to resize the array
    if (context->diagnostics.count >= context->diagnostics.capacity) {
        size_t new_capacity = context->diagnostics.capacity == 0 ? 8 : context->diagnostics.capacity * 2;
        size_t new_size = new_capacity * sizeof(Diagnostic);
        
        // Allocate new array
        Diagnostic* new_items = arena_alloc(context->arena, new_size);
        if (!new_items) return; // Failed to allocate
        
        // Copy old items
        if (context->diagnostics.items) {
            memcpy(new_items, context->diagnostics.items, context->diagnostics.count * sizeof(Diagnostic));
        }
        
        // Update array
        context->diagnostics.items = new_items;
        context->diagnostics.capacity = new_capacity;
    }
    
    // Add diagnostic
    Diagnostic* diagnostic = &context->diagnostics.items[context->diagnostics.count++];
    diagnostic->severity = severity;
    diagnostic->location = location;
    diagnostic->message = message;
    diagnostic->code = code;
    
    // Update error count
    if (severity == DIAGNOSTIC_ERROR || severity == DIAGNOSTIC_FATAL) {
        context->error_count++;
    }
}

size_t diagnostic_context_get_count(DiagnosticContext* context) {
    assert(context != NULL);
    return context->diagnostics.count;
}

const Diagnostic* diagnostic_context_get(DiagnosticContext* context, size_t index) {
    assert(context != NULL);
    
    if (index >= context->diagnostics.count) {
        return NULL;
    }
    
    return &context->diagnostics.items[index];
}

size_t diagnostic_context_get_error_count(DiagnosticContext* context) {
    assert(context != NULL);
    return context->error_count;
}

bool diagnostic_context_has_errors(DiagnosticContext* context) {
    assert(context != NULL);
    return context->error_count > 0;
}

/**
 * @brief Get the severity name
 * 
 * @param severity Severity
 * @return Name of the severity
 */
static const char* get_severity_name(DiagnosticSeverity severity) {
    switch (severity) {
        case DIAGNOSTIC_INFO:    return "info";
        case DIAGNOSTIC_WARNING: return "warning";
        case DIAGNOSTIC_ERROR:   return "error";
        case DIAGNOSTIC_FATAL:   return "fatal";
        default:                 return "unknown";
    }
}

/**
 * @brief Get the severity color
 * 
 * @param severity Severity
 * @return ANSI color code for the severity
 */
static const char* get_severity_color(DiagnosticSeverity severity) {
    switch (severity) {
        case DIAGNOSTIC_INFO:    return "\033[36m"; // Cyan
        case DIAGNOSTIC_WARNING: return "\033[33m"; // Yellow
        case DIAGNOSTIC_ERROR:   return "\033[31m"; // Red
        case DIAGNOSTIC_FATAL:   return "\033[35m"; // Magenta
        default:                 return "\033[0m";  // Reset
    }
}

void diagnostic_context_print(DiagnosticContext* context) {
    assert(context != NULL);
    
    const char* reset_color = "\033[0m";
    
    // Print each diagnostic
    for (size_t i = 0; i < context->diagnostics.count; i++) {
        const Diagnostic* diagnostic = &context->diagnostics.items[i];
        
        // Get severity name and color
        const char* severity_name = get_severity_name(diagnostic->severity);
        const char* severity_color = get_severity_color(diagnostic->severity);
        
        // Print location
        if (diagnostic->location.file) {
            printf("%s:%d:%d: ", diagnostic->location.file, diagnostic->location.line, diagnostic->location.column);
        }
        
        // Print severity
        printf("%s%s%s", severity_color, severity_name, reset_color);
        
        // Print error code
        if (diagnostic->code) {
            printf("[%s]: ", diagnostic->code);
        } else {
            printf(": ");
        }
        
        // Print message
        printf("%s\n", diagnostic->message);
    }
}

void diagnostic_error(DiagnosticContext* context, int line, int column, const char* message) {
    assert(context != NULL);
    assert(message != NULL);
    
    // Create source location
    SourceLocation location = source_location_create(NULL, line, column, 0);
    
    // Add diagnostic
    diagnostic_context_add(context, DIAGNOSTIC_ERROR, location, message, NULL);
}
