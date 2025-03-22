/**
 * @file test_codegen.c
 * @brief Unit tests for the code generator
 */

#include "backend/codegen.h"
#include "frontend/ast/ast.h"
#include "core/memory.h"
#include "core/diagnostics.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * Test creating a code generator context
 */
static void test_codegen_context_create(void) {
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Create diagnostic context
    DiagnosticContext* diag = diagnostic_context_create(arena);
    assert(diag != NULL);
    
    // Create code generator context
    CodegenContext* codegen = codegen_context_create(arena, diag);
    assert(codegen != NULL);
    
    // Clean up
    arena_destroy(arena);
    
    printf("test_codegen_context_create: PASSED\n");
}

/**
 * Test generating C code for a simple AST
 */
static void test_codegen_generate_simple(void) {
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Create string table
    StringTable* strings = string_table_create(arena, 1024);
    assert(strings != NULL);
    
    // Create diagnostic context
    DiagnosticContext* diag = diagnostic_context_create(arena);
    assert(diag != NULL);
    
    // Create a simple AST (just a number literal)
    AstNode* ast = ast_create_number(arena, 42.0, 1, 1);
    assert(ast != NULL);
    
    // Create code generator context
    CodegenContext* codegen = codegen_context_create(arena, diag);
    assert(codegen != NULL);
    
    // Generate C code to a temporary file
    const char* temp_file = "test_codegen_output.c";
    bool result = codegen_generate(codegen, ast, temp_file);
    assert(result);
    
    // Clean up
    remove(temp_file);
    arena_destroy(arena);
    
    printf("test_codegen_generate_simple: PASSED\n");
}

/**
 * Main entry point
 */
int main(void) {
    test_codegen_context_create();
    test_codegen_generate_simple();
    
    printf("All tests passed!\n");
    return 0;
}
