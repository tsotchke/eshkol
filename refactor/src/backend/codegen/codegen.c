/**
 * @file codegen.c
 * @brief Code generation implementation
 */

#include "backend/codegen.h"
#include "backend/codegen/context.h"
#include "frontend/type_inference/type_inference.h"
#include "core/memory.h"
#include "core/diagnostics.h"
#include "backend/codegen/program.h"
#include "backend/codegen/compiler.h"
#include "backend/codegen/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Create a code generator context
 */
CodegenContext* codegen_context_create(Arena* arena, DiagnosticContext* diagnostics, TypeInferenceContext* type_context) {
    assert(arena != NULL);
    assert(diagnostics != NULL);
    
    // Allocate context
    CodegenContext* context = arena_alloc(arena, sizeof(CodegenContext));
    if (!context) return NULL;
    
    // Initialize context using codegen_context_init
    if (!codegen_context_init(context, arena, type_context, NULL)) {
        return NULL;
    }
    
    // Set diagnostics (not set by codegen_context_init)
    context->diagnostics = diagnostics;
    
    return context;
}

/**
 * @brief Generate C code from an AST
 */
bool codegen_generate(CodegenContext* context, const AstNode* ast, const char* output_file) {
    assert(context != NULL);
    assert(ast != NULL);
    
    // Open output file
    FILE* output = NULL;
    if (output_file != NULL) {
        output = fopen(output_file, "w");
        if (!output) {
            diagnostic_error(codegen_context_get_diagnostics(context), "Failed to open output file: %s", output_file);
            return false;
        }
    } else {
        output = stdout;
    }
    
    // Set output file
    codegen_context_set_output(context, output);
    
    // Generate program
    bool success = codegen_generate_program(context, ast);
    
    // Close output file
    if (output_file != NULL) {
        FILE* output = codegen_context_get_output(context);
        if (output != NULL && output != stdout) {
            fclose(output);
            codegen_context_set_output(context, NULL);
        }
    }
    
    return success;
}

/**
 * @brief Compile and execute the generated code
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc) {
    assert(context != NULL);
    assert(c_file != NULL);
    
    // Get the base name of the C file (without extension)
    char* base_name = strdup(c_file);
    char* dot = strrchr(base_name, '.');
    if (dot) *dot = '\0';
    
    // Create the output file path
    char output_file[1024];
    snprintf(output_file, sizeof(output_file), "%s.out", base_name);
    
    // Compile the C file
    bool success = codegen_compile(context, c_file, output_file);
    if (!success) {
        free(base_name);
        return -1;
    }
    
    // Execute the compiled program
    int result = codegen_execute(context, output_file, args, argc);
    
    free(base_name);
    return result;
}
