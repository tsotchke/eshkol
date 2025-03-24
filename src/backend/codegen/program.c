/**
 * @file program.c
 * @brief Program code generation implementation
 */

#include "backend/codegen/program.h"
#include "backend/codegen/context.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
#include "frontend/ast/ast.h"
#include "frontend/type_inference/type_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a program header
 */
static bool codegen_generate_program_header(CodegenContext* context) {
    assert(context != NULL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    assert(output != NULL);
    
    // Generate header
    fprintf(output, "#include <stdio.h>\n");
    fprintf(output, "#include <stdlib.h>\n");
    fprintf(output, "#include <stdbool.h>\n");
    fprintf(output, "#include <math.h>\n");
    fprintf(output, "#include \"core/vector.h\"\n");
    fprintf(output, "#include \"core/memory.h\"\n");
    fprintf(output, "#include \"core/autodiff.h\"\n\n");
    
    // Global arena variable
    fprintf(output, "// Global arena for memory allocations\n");
    fprintf(output, "Arena* arena = NULL;\n\n");
    
    // Define eshkol_value_t
    fprintf(output, "// Eshkol value type\n");
    fprintf(output, "typedef union {\n");
    fprintf(output, "    long integer;\n");
    fprintf(output, "    double floating;\n");
    fprintf(output, "    bool boolean;\n");
    fprintf(output, "    char character;\n");
    fprintf(output, "    char* string;\n");
    fprintf(output, "    void* pointer;\n");
    fprintf(output, "} eshkol_value_t;\n\n");
    
    return true;
}

/**
 * @brief Generate C code for function forward declarations
 */
static bool codegen_generate_forward_declarations(CodegenContext* context, const AstNode* program) {
    assert(context != NULL);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    assert(output != NULL);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Generate forward declarations
    fprintf(output, "// Forward declarations\n");
    
    for (size_t i = 0; i < program->as.program.expr_count; i++) {
        if (program->as.program.exprs[i]->type == AST_FUNCTION_DEF) {
            // Get function node
            AstNode* func_node = program->as.program.exprs[i];
            
            // Get return type
            const char* return_type = "void"; // Default to void
            
            // Try to get type from type inference context
            Type* resolved_type = type_inference_get_type(type_context, func_node);
            if (resolved_type != NULL) {
                if (resolved_type->kind == TYPE_FUNCTION) {
                    // Function type, use the return type
                    return_type = codegen_type_to_c_type(resolved_type->function.return_type);
                } else {
                    // Not a function type, use the resolved type directly
                    return_type = codegen_type_to_c_type(resolved_type);
                }
            } else if (func_node->as.function_def.return_type != NULL) {
                return_type = codegen_type_to_c_type(func_node->as.function_def.return_type);
            } else if (func_node->as.function_def.body != NULL && 
                      func_node->as.function_def.body->type_info != NULL) {
                return_type = codegen_type_to_c_type(func_node->as.function_def.body->type_info);
            }
            
            // Generate function declaration
            fprintf(output, "%s ", return_type);
            
            // Generate function name
            if (func_node->as.function_def.name->type == AST_IDENTIFIER) {
                fprintf(output, "%s", func_node->as.function_def.name->as.identifier.name);
            } else {
                fprintf(output, "_func_%zu_%zu", func_node->line, func_node->column);
            }
            
            // Generate parameter list
            fprintf(output, "(");
            
            for (size_t j = 0; j < func_node->as.function_def.param_count; j++) {
                if (j > 0) {
                    fprintf(output, ", ");
                }
                
                // Get parameter type
                const char* param_type = "float"; // Default to float for autodiff functions
                Parameter* param = func_node->as.function_def.params[j];
                AstNode* param_node = func_node->as.function_def.param_nodes[j];
                
                // Try to get type from type inference context
                Type* param_resolved_type = type_inference_get_type(type_context, param_node);
                if (param_resolved_type != NULL) {
                    param_type = codegen_type_to_c_type(param_resolved_type);
                } else if (param->type != NULL) {
                    param_type = codegen_type_to_c_type(param->type);
                }
                
                fprintf(output, "%s", param_type);
            }
            
            fprintf(output, ");\n");
        }
    }
    
    fprintf(output, "\n");
    
    return true;
}

/**
 * @brief Generate C code for a main function
 */
static bool codegen_generate_main_function(CodegenContext* context, const AstNode* program) {
    assert(context != NULL);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    assert(output != NULL);
    
    // Check if there's already a main function
    bool has_main = false;
    for (size_t i = 0; i < program->as.program.expr_count; i++) {
        if (program->as.program.exprs[i]->type == AST_FUNCTION_DEF) {
            AstNode* name = program->as.program.exprs[i]->as.function_def.name;
            if (name->type == AST_IDENTIFIER && strcmp(name->as.identifier.name, "main") == 0) {
                has_main = true;
                break;
            }
        }
    }
    
    // Generate main function if there isn't one already
    if (!has_main) {
        fprintf(output, "int main(int argc, char** argv) {\n");
        fprintf(output, "    // Initialize arena\n");
        fprintf(output, "    arena = arena_create(1024 * 1024);\n");
        fprintf(output, "    if (!arena) {\n");
        fprintf(output, "        fprintf(stderr, \"Failed to create memory arena\\n\");\n");
        fprintf(output, "        return 1;\n");
        fprintf(output, "    }\n\n");
        fprintf(output, "    printf(\"Hello from Eshkol!\\n\");\n\n");
        fprintf(output, "    // Clean up arena\n");
        fprintf(output, "    arena_destroy(arena);\n");
        fprintf(output, "    return 0;\n");
        fprintf(output, "}\n");
    }
    
    return true;
}

/**
 * @brief Generate C code for a program
 */
bool codegen_generate_program(CodegenContext* context, const AstNode* program) {
    assert(context != NULL);
    assert(program != NULL);
    assert(program->type == AST_PROGRAM);
    
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Generate program header
    diagnostic_debug(diagnostics, 0, 0, "Generating program header");
    if (!codegen_generate_program_header(context)) {
        diagnostic_error(diagnostics, 0, 0, "Failed to generate program header");
        return false;
    }
    
    // Generate forward declarations
    diagnostic_debug(diagnostics, 0, 0, "Generating forward declarations");
    if (!codegen_generate_forward_declarations(context, program)) {
        diagnostic_error(diagnostics, 0, 0, "Failed to generate forward declarations");
        return false;
    }
    
    // Generate expressions
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "Generating expressions (%zu total)", program->as.program.expr_count);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    for (size_t i = 0; i < program->as.program.expr_count; i++) {
        snprintf(debug_msg, sizeof(debug_msg), "Generating expression %zu of type %d", i, program->as.program.exprs[i]->type);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        if (!codegen_generate_expression(context, program->as.program.exprs[i])) {
            char error_msg[256];
            snprintf(error_msg, sizeof(error_msg), "Failed to generate expression %zu of type %d", i, program->as.program.exprs[i]->type);
            diagnostic_error(diagnostics, 0, 0, error_msg);
            return false;
        }
        
        // Add newline after each expression
        FILE* output = codegen_context_get_output(context);
        fprintf(output, "\n");
    }
    
    // Generate main function if needed
    diagnostic_debug(diagnostics, 0, 0, "Generating main function if needed");
    if (!codegen_generate_main_function(context, program)) {
        diagnostic_error(diagnostics, 0, 0, "Failed to generate main function");
        return false;
    }
    
    diagnostic_debug(diagnostics, 0, 0, "Program generation completed successfully");
    return true;
}
