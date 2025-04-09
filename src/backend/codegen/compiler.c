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
    
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug: Print AST type
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "AST type: %d", ast->type);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    // Open output file
    if (output_file != NULL) {
        snprintf(debug_msg, sizeof(debug_msg), "Opening output file: %s", output_file);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        FILE* output = fopen(output_file, "w");
        if (!output) {
            snprintf(debug_msg, sizeof(debug_msg), "Failed to open output file: %s", output_file);
            diagnostic_error(diagnostics, 0, 0, debug_msg);
            return false;
        }
        codegen_context_set_output(context, output);
        diagnostic_debug(diagnostics, 0, 0, "Output file opened successfully");
    } else {
        diagnostic_debug(diagnostics, 0, 0, "Using stdout for output");
        codegen_context_set_output(context, stdout);
    }
    
    // Generate code
    bool result = false;
    
    if (ast->type == AST_PROGRAM) {
        snprintf(debug_msg, sizeof(debug_msg), "Generating code for program with %zu expressions", ast->as.program.expr_count);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        // Debug: Print program expressions
        for (size_t i = 0; i < ast->as.program.expr_count; i++) {
            snprintf(debug_msg, sizeof(debug_msg), "Program expression %zu: type %d", i, ast->as.program.exprs[i]->type);
            diagnostic_debug(diagnostics, 0, 0, debug_msg);
            
            // If it's a define, print more details
            if (ast->as.program.exprs[i]->type == AST_DEFINE) {
                AstNode* name = ast->as.program.exprs[i]->as.define.name;
                AstNode* value = ast->as.program.exprs[i]->as.define.value;
                
                if (name->type == AST_IDENTIFIER) {
                    snprintf(debug_msg, sizeof(debug_msg), "Define name: %s", name->as.identifier.name);
                    diagnostic_debug(diagnostics, 0, 0, debug_msg);
                }
                
                snprintf(debug_msg, sizeof(debug_msg), "Define value type: %d", value->type);
                diagnostic_debug(diagnostics, 0, 0, debug_msg);
            }
        }
        
        result = codegen_generate_program(context, ast);
        if (!result) {
            diagnostic_error(diagnostics, 0, 0, "Failed to generate code for program");
        }
    } else {
        snprintf(debug_msg, sizeof(debug_msg), "Generating code for expression of type %d", ast->type);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        result = codegen_generate_expression(context, ast);
        if (!result) {
            snprintf(debug_msg, sizeof(debug_msg), "Failed to generate code for expression of type %d", ast->type);
            diagnostic_error(diagnostics, 0, 0, debug_msg);
        }
    }
    
    // Close output file
    FILE* output = codegen_context_get_output(context);
    if (output_file != NULL && output != NULL) {
        diagnostic_debug(diagnostics, 0, 0, "Closing output file");
        fclose(output);
    }
    
    codegen_context_set_output(context, NULL);
    
    diagnostic_debug(diagnostics, 0, 0, result ? "Code generation succeeded" : "Code generation failed");
    return result;
}

/**
 * @brief Compile and execute the generated code
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc) {
    assert(context != NULL);
    assert(c_file != NULL);
    
    // Determine the include path based on the Eshkol directory
    char include_path[1024] = {0};
    
    // Get the path to the source file
    char source_path[1024] = {0};
    char *source_dir = NULL;
    char *eshkol_dir = NULL;
    
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
    
    // Determine the Eshkol directory by finding "eshkol" in the source path
    eshkol_dir = strstr(source_dir, "eshkol");
    
    if (eshkol_dir) {
        // If the source file is in the Eshkol directory or a subdirectory
        // Set include path relative to the Eshkol root
        *eshkol_dir = '\0'; // Truncate at "eshkol"
        snprintf(include_path, sizeof(include_path), "%seshkol/include", source_dir);
    } else {
        // If the source file is outside the Eshkol directory
        // Assume the Eshkol directory is a sibling of the current directory
        snprintf(include_path, sizeof(include_path), "%s/eshkol/include", source_dir);
    }
    
    // Create a temporary executable name
    char temp_executable[1024];
    snprintf(temp_executable, sizeof(temp_executable), "/tmp/eshkol_temp_%d", getpid());
    
    // Create command to compile the C file
    char compile_cmd[1024];
    snprintf(compile_cmd, sizeof(compile_cmd), "gcc -I%s -o %s %s %s", include_path, temp_executable, c_file, "../src/core/utils/closure.c");
    
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
