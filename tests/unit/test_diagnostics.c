/**
 * @file test_diagnostics.c
 * @brief Unit tests for the diagnostics system
 */

#include "core/diagnostics.h"
#include "core/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Test diagnostic context creation
 */
static void test_diagnostic_context_create(void) {
    printf("Testing diagnostic context creation...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DiagnosticContext* context = diagnostic_context_create(arena);
    assert(context != NULL);
    assert(diagnostic_context_get_count(context) == 0);
    assert(diagnostic_context_get_error_count(context) == 0);
    assert(!diagnostic_context_has_errors(context));
    
    arena_destroy(arena);
    
    printf("PASS: diagnostic_context_create\n");
}

/**
 * @brief Test adding diagnostics
 */
static void test_diagnostic_context_add(void) {
    printf("Testing adding diagnostics...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DiagnosticContext* context = diagnostic_context_create(arena);
    assert(context != NULL);
    
    // Create a source location
    SourceLocation location = source_location_create("test.esk", 10, 5, 3);
    
    // Add an info diagnostic
    diagnostic_context_add(context, DIAGNOSTIC_INFO, location, "This is an info message", "I001");
    
    // Check that it was added
    assert(diagnostic_context_get_count(context) == 1);
    assert(diagnostic_context_get_error_count(context) == 0);
    assert(!diagnostic_context_has_errors(context));
    
    // Get the diagnostic
    const Diagnostic* diagnostic = diagnostic_context_get(context, 0);
    assert(diagnostic != NULL);
    assert(diagnostic->severity == DIAGNOSTIC_INFO);
    assert(diagnostic->location.file == location.file);
    assert(diagnostic->location.line == location.line);
    assert(diagnostic->location.column == location.column);
    assert(diagnostic->location.length == location.length);
    assert(strcmp(diagnostic->message, "This is an info message") == 0);
    assert(strcmp(diagnostic->code, "I001") == 0);
    
    // Add a warning diagnostic
    diagnostic_context_add(context, DIAGNOSTIC_WARNING, location, "This is a warning message", "W001");
    
    // Check that it was added
    assert(diagnostic_context_get_count(context) == 2);
    assert(diagnostic_context_get_error_count(context) == 0);
    assert(!diagnostic_context_has_errors(context));
    
    // Add an error diagnostic
    diagnostic_context_add(context, DIAGNOSTIC_ERROR, location, "This is an error message", "E001");
    
    // Check that it was added
    assert(diagnostic_context_get_count(context) == 3);
    assert(diagnostic_context_get_error_count(context) == 1);
    assert(diagnostic_context_has_errors(context));
    
    // Add a fatal diagnostic
    diagnostic_context_add(context, DIAGNOSTIC_FATAL, location, "This is a fatal message", "F001");
    
    // Check that it was added
    assert(diagnostic_context_get_count(context) == 4);
    assert(diagnostic_context_get_error_count(context) == 2);
    assert(diagnostic_context_has_errors(context));
    
    arena_destroy(arena);
    
    printf("PASS: diagnostic_context_add\n");
}

/**
 * @brief Test getting diagnostics
 */
static void test_diagnostic_context_get(void) {
    printf("Testing getting diagnostics...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DiagnosticContext* context = diagnostic_context_create(arena);
    assert(context != NULL);
    
    // Create a source location
    SourceLocation location = source_location_create("test.esk", 10, 5, 3);
    
    // Add some diagnostics
    diagnostic_context_add(context, DIAGNOSTIC_INFO, location, "Info message", "I001");
    diagnostic_context_add(context, DIAGNOSTIC_WARNING, location, "Warning message", "W001");
    diagnostic_context_add(context, DIAGNOSTIC_ERROR, location, "Error message", "E001");
    diagnostic_context_add(context, DIAGNOSTIC_FATAL, location, "Fatal message", "F001");
    
    // Check the count
    assert(diagnostic_context_get_count(context) == 4);
    
    // Get each diagnostic
    const Diagnostic* info = diagnostic_context_get(context, 0);
    const Diagnostic* warning = diagnostic_context_get(context, 1);
    const Diagnostic* error = diagnostic_context_get(context, 2);
    const Diagnostic* fatal = diagnostic_context_get(context, 3);
    
    // Check that they are not NULL
    assert(info != NULL);
    assert(warning != NULL);
    assert(error != NULL);
    assert(fatal != NULL);
    
    // Check the severities
    assert(info->severity == DIAGNOSTIC_INFO);
    assert(warning->severity == DIAGNOSTIC_WARNING);
    assert(error->severity == DIAGNOSTIC_ERROR);
    assert(fatal->severity == DIAGNOSTIC_FATAL);
    
    // Check the messages
    assert(strcmp(info->message, "Info message") == 0);
    assert(strcmp(warning->message, "Warning message") == 0);
    assert(strcmp(error->message, "Error message") == 0);
    assert(strcmp(fatal->message, "Fatal message") == 0);
    
    // Check the codes
    assert(strcmp(info->code, "I001") == 0);
    assert(strcmp(warning->code, "W001") == 0);
    assert(strcmp(error->code, "E001") == 0);
    assert(strcmp(fatal->code, "F001") == 0);
    
    // Check out of bounds
    assert(diagnostic_context_get(context, 4) == NULL);
    
    arena_destroy(arena);
    
    printf("PASS: diagnostic_context_get\n");
}

/**
 * @brief Test printing diagnostics
 */
static void test_diagnostic_context_print(void) {
    printf("Testing printing diagnostics...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DiagnosticContext* context = diagnostic_context_create(arena);
    assert(context != NULL);
    
    // Create a source location
    SourceLocation location = source_location_create("test.esk", 10, 5, 3);
    
    // Add some diagnostics
    diagnostic_context_add(context, DIAGNOSTIC_INFO, location, "This is an info message", "I001");
    diagnostic_context_add(context, DIAGNOSTIC_WARNING, location, "This is a warning message", "W001");
    diagnostic_context_add(context, DIAGNOSTIC_ERROR, location, "This is an error message", "E001");
    diagnostic_context_add(context, DIAGNOSTIC_FATAL, location, "This is a fatal message", "F001");
    
    // Print the diagnostics
    printf("Diagnostic output:\n");
    diagnostic_context_print(context);
    
    arena_destroy(arena);
    
    printf("PASS: diagnostic_context_print\n");
}

/**
 * @brief Test adding many diagnostics
 */
static void test_diagnostic_context_many(void) {
    printf("Testing adding many diagnostics...\n");
    
    Arena* arena = arena_create(1024);
    assert(arena != NULL);
    
    DiagnosticContext* context = diagnostic_context_create(arena);
    assert(context != NULL);
    
    // Create a source location
    SourceLocation location = source_location_create("test.esk", 10, 5, 3);
    
    // Add many diagnostics
    const int count = 100;
    for (int i = 0; i < count; i++) {
        char message[64];
        sprintf(message, "Diagnostic %d", i);
        
        DiagnosticSeverity severity = (DiagnosticSeverity)(i % 4);
        diagnostic_context_add(context, severity, location, message, NULL);
    }
    
    // Check the count
    assert(diagnostic_context_get_count(context) == count);
    
    // Check the error count
    int expected_error_count = count / 2; // DIAGNOSTIC_ERROR and DIAGNOSTIC_FATAL
    assert(diagnostic_context_get_error_count(context) == expected_error_count);
    
    // Check that we can get all diagnostics
    for (int i = 0; i < count; i++) {
        const Diagnostic* diagnostic = diagnostic_context_get(context, i);
        assert(diagnostic != NULL);
        
        char expected_message[64];
        sprintf(expected_message, "Diagnostic %d", i);
        assert(strcmp(diagnostic->message, expected_message) == 0);
    }
    
    arena_destroy(arena);
    
    printf("PASS: diagnostic_context_many\n");
}

/**
 * @brief Main function
 */
int main(void) {
    printf("Running diagnostics tests...\n");
    
    test_diagnostic_context_create();
    test_diagnostic_context_add();
    test_diagnostic_context_get();
    test_diagnostic_context_print();
    test_diagnostic_context_many();
    
    printf("All diagnostics tests passed!\n");
    return 0;
}
