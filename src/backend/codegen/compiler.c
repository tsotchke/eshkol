/**
 * @file compiler.c
 * @brief Top-level compilation functions implementation
 */

#include "backend/codegen/compiler.h"
#include "backend/codegen/context.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/program.h"
#include "core/memory.h"
#include "core/diagnostics.h"
#include "core/file_io.h"
#include "frontend/ast/ast.h"
#include "frontend/type_inference/type_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <unistd.h>

/**
 * @brief Generate C code from an AST
 */
bool codegen_generate(CodegenContext* context, const AstNode* ast, const char* output_file) {
    assert(context != NULL);
    assert(ast != NULL);
    
    // Open output file
    if (output_file != NULL) {
        FILE* output = fopen(output_file, "w");
        if (!output) {
            DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
            diagnostic_error(diagnostics, 0, 0, "Failed to open output file");
            return false;
        }
        codegen_context_set_output(context, output);
    } else {
        codegen_context_set_output(context, stdout);
    }
    
    // Generate code
    bool result = false;
    
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    if (ast->type == AST_PROGRAM) {
        diagnostic_debug(diagnostics, 0, 0, "Generating code for program");
        result = codegen_generate_program(context, ast);
        if (!result) {
            diagnostic_error(diagnostics, 0, 0, "Failed to generate code for program");
        }
    } else {
        diagnostic_debug(diagnostics, 0, 0, "Generating code for expression");
        result = codegen_generate_expression(context, ast);
        if (!result) {
            diagnostic_error(diagnostics, 0, 0, "Failed to generate code for expression");
        }
    }
    
    // Close output file
    FILE* output = codegen_context_get_output(context);
    if (output_file != NULL && output != NULL) {
        fclose(output);
    }
    
    codegen_context_set_output(context, NULL);
    
    return result;
}

/**
 * @brief Compile and execute the generated code
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc) {
    assert(context != NULL);
    assert(c_file != NULL);
    
    // Determine the include path based on the executable location
    char include_path[1024] = {0};
    
    // Get the path to the source file
    char source_path[1024] = {0};
    char *source_dir = NULL;
    
    // Get the absolute path of the source file
    if (c_file[0] == '/') {
        // Absolute path
        strncpy(source_path, c_file, sizeof(source_path) - 1);
    } else {
        // Relative path, prepend current directory
        char cwd[1024] = {0};
        if (getcwd(cwd, sizeof(cwd)) == NULL) {
            DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
            diagnostic_error(diagnostics, 0, 0, "Failed to get current working directory");
            return -1;
        }
        snprintf(source_path, sizeof(source_path), "%s/%s", cwd, c_file);
    }
    
    // Extract the directory from the source path
    source_dir = strdup(source_path);
    char *last_slash = strrchr(source_dir, '/');
    if (last_slash) {
        *last_slash = '\0';
    }
    
    // Construct the include path relative to the source directory
    snprintf(include_path, sizeof(include_path), "%s/../include", source_dir);
    
    // Create a temporary executable name
    char temp_executable[1024];
    snprintf(temp_executable, sizeof(temp_executable), "/tmp/eshkol_temp_%d", getpid());
    
    // Create command to compile the C file
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd), "gcc -I%s -o %s %s", include_path, temp_executable, c_file);
    
    // Free allocated memory
    free(source_dir);
    
    // Compile the C file
    int result = system(compile_cmd);
    if (result != 0) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, 0, 0, "Failed to compile generated C code");
        return -1;
    }
    
    // Create command to execute the compiled program
    char execute_cmd[1024];
    snprintf(execute_cmd, sizeof(execute_cmd), "%s", temp_executable);
    
    // Add arguments
    for (int i = 0; i < argc; i++) {
        strcat(execute_cmd, " ");
        strcat(execute_cmd, args[i]);
    }
    
    // Execute the compiled program
    result = system(execute_cmd);
    
    // Clean up the temporary executable
    unlink(temp_executable);
    
    return result;
}
