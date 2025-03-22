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
 * Test generating C code for a function with typed parameters
 */
static void test_codegen_generate_typed_function(void) {
    // Create arena
    Arena* arena = arena_create(1024 * 1024);
    assert(arena != NULL);
    
    // Create string table
    StringTable* strings = string_table_create(arena, 1024);
    assert(strings != NULL);
    
    // Create diagnostic context
    DiagnosticContext* diag = diagnostic_context_create(arena);
    assert(diag != NULL);
    
    // Create function name
    StringId add_name = string_table_intern(strings, "add");
    AstNode* name = ast_create_identifier(arena, add_name, 1, 1);
    assert(name != NULL);
    
    // Create parameter types
    Type* int_type = type_integer_create(arena, INT_SIZE_32);
    assert(int_type != NULL);
    
    // Create parameters
    StringId a_name = string_table_intern(strings, "a");
    StringId b_name = string_table_intern(strings, "b");
    Parameter* param_a = parameter_create(arena, a_name, int_type, 1, 10);
    Parameter* param_b = parameter_create(arena, b_name, int_type, 1, 20);
    assert(param_a != NULL);
    assert(param_b != NULL);
    
    Parameter* params[2] = { param_a, param_b };
    
    // Create function body (a + b)
    AstNode* a_id = ast_create_identifier(arena, a_name, 2, 3);
    AstNode* b_id = ast_create_identifier(arena, b_name, 2, 7);
    StringId plus_name = string_table_intern(strings, "+");
    AstNode* plus = ast_create_identifier(arena, plus_name, 2, 5);
    AstNode* args[2] = { a_id, b_id };
    AstNode* body = ast_create_call(arena, plus, args, 2, 2, 3);
    assert(body != NULL);
    
    // Create function definition
    AstNode* func_def = ast_create_function_def(arena, name, params, 2, int_type, body, 1, 1);
    assert(func_def != NULL);
    
    // Create program with the function definition
    AstNode* program = ast_create_program(arena, &func_def, 1, 1, 1);
    assert(program != NULL);
    
    // Create code generator context
    CodegenContext* codegen = codegen_context_create(arena, diag);
    assert(codegen != NULL);
    
    // Generate C code to a temporary file
    const char* temp_file = "test_codegen_typed_function.c";
    bool result = codegen_generate(codegen, program, temp_file);
    assert(result);
    
    // Clean up
    remove(temp_file);
    arena_destroy(arena);
    
    printf("test_codegen_generate_typed_function: PASSED\n");
}

/**
 * Main entry point
 */
int main(void) {
    test_codegen_context_create();
    test_codegen_generate_simple();
    test_codegen_generate_typed_function();
    
    printf("All tests passed!\n");
    return 0;
}
