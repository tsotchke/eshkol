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
    fprintf(output, "#include <string.h>\n\n");
    fprintf(output, "#include \"core/vector.h\"\n");
    fprintf(output, "#include \"core/memory.h\"\n");
    fprintf(output, "#include \"core/closure.h\"\n");
    fprintf(output, "#include \"core/autodiff.h\"\n\n");
    
    // Define minimal Arena implementation
    fprintf(output, "// Minimal Arena implementation\n");
    fprintf(output, "typedef struct Arena {\n");
    fprintf(output, "    void* memory;\n");
    fprintf(output, "    size_t size;\n");
    fprintf(output, "    size_t used;\n");
    fprintf(output, "} Arena;\n\n");
    
    fprintf(output, "Arena* arena_create(size_t size) {\n");
    fprintf(output, "    Arena* arena = malloc(sizeof(Arena));\n");
    fprintf(output, "    if (!arena) return NULL;\n");
    fprintf(output, "    arena->memory = malloc(size);\n");
    fprintf(output, "    if (!arena->memory) {\n");
    fprintf(output, "        free(arena);\n");
    fprintf(output, "        return NULL;\n");
    fprintf(output, "    }\n");
    fprintf(output, "    arena->size = size;\n");
    fprintf(output, "    arena->used = 0;\n");
    fprintf(output, "    return arena;\n");
    fprintf(output, "}\n\n");
    
    fprintf(output, "void arena_destroy(Arena* arena) {\n");
    fprintf(output, "    if (arena) {\n");
    fprintf(output, "        free(arena->memory);\n");
    fprintf(output, "        free(arena);\n");
    fprintf(output, "    }\n");
    fprintf(output, "}\n\n");
    
    fprintf(output, "void* arena_alloc(Arena* arena, size_t size) {\n");
    fprintf(output, "    if (!arena || arena->used + size > arena->size) return NULL;\n");
    fprintf(output, "    void* ptr = (char*)arena->memory + arena->used;\n");
    fprintf(output, "    arena->used += size;\n");
    fprintf(output, "    return ptr;\n");
    fprintf(output, "}\n\n");
    
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
                // Replace hyphens with underscores in function names
                char* function_name = strdup(func_node->as.function_def.name->as.identifier.name);
                if (function_name) {
                    for (char* p = function_name; *p; p++) {
                        if (*p == '-') {
                            *p = '_';
                        }
                    }
                    fprintf(output, "%s", function_name);
                    free(function_name);
                } else {
                    fprintf(output, "%s", func_node->as.function_def.name->as.identifier.name);
                }
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
            if (name->type == AST_IDENTIFIER) {
                // Check for "main" or "main-*" (which would be converted to main_*)
                const char* name_str = name->as.identifier.name;
                if (strcmp(name_str, "main") == 0 || strncmp(name_str, "main-", 5) == 0) {
                    has_main = true;
                    break;
                }
            }
        } else if (program->as.program.exprs[i]->type == AST_DEFINE) {
            AstNode* name = program->as.program.exprs[i]->as.define.name;
            if (name->type == AST_IDENTIFIER) {
                // Check for "main" or "main-*" (which would be converted to main_*)
                const char* name_str = name->as.identifier.name;
                if (strcmp(name_str, "main") == 0 || strncmp(name_str, "main-", 5) == 0) {
                    has_main = true;
                    break;
                }
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
        
        // Find all top-level expressions that are not function definitions
        // and add them to the main function
        for (size_t i = 0; i < program->as.program.expr_count; i++) {
            AstNode* expr = program->as.program.exprs[i];
            
            // Skip function definitions
            if (expr->type == AST_FUNCTION_DEF) {
                continue;
            }
            
            // Skip variable definitions with lambda values
            if (expr->type == AST_DEFINE && 
                expr->as.define.value && 
                expr->as.define.value->type == AST_LAMBDA) {
                continue;
            }
            
            // For display and other expressions, add them to the main function
            if (expr->type == AST_CALL) {
                AstNode* callee = expr->as.call.callee;
                if (callee && callee->type == AST_IDENTIFIER) {
                    const char* name = callee->as.identifier.name;
                    if (strcmp(name, "display") == 0 || strcmp(name, "newline") == 0) {
                        fprintf(output, "    ");
                        codegen_context_set_in_function(context, true);
                        codegen_generate_expression(context, expr);
                        codegen_context_set_in_function(context, false);
                        fprintf(output, ";\n");
                    }
                }
            }
        }
        
        fprintf(output, "\n    // Clean up arena\n");
        fprintf(output, "    arena_destroy(arena);\n");
        fprintf(output, "    return 0;\n");
        fprintf(output, "}\n");
    } else {
        // Add arena initialization code before the main function
        fprintf(output, "// Initialize arena before main\n");
        fprintf(output, "static void __attribute__((constructor)) init_arena() {\n");
        fprintf(output, "    arena = arena_create(1024 * 1024);\n");
        fprintf(output, "    if (!arena) {\n");
        fprintf(output, "        fprintf(stderr, \"Failed to create memory arena\\n\");\n");
        fprintf(output, "        exit(1);\n");
        fprintf(output, "    }\n");
        fprintf(output, "}\n\n");
        
        // Add arena cleanup code after the main function
        fprintf(output, "// Clean up arena after main\n");
        fprintf(output, "static void __attribute__((destructor)) cleanup_arena() {\n");
        fprintf(output, "    if (arena) {\n");
        fprintf(output, "        arena_destroy(arena);\n");
        fprintf(output, "        arena = NULL;\n");
        fprintf(output, "    }\n");
        fprintf(output, "}\n\n");
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
    
    // Debug: Print program structure
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "Program has %zu expressions", program->as.program.expr_count);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    for (size_t i = 0; i < program->as.program.expr_count; i++) {
        AstNode* expr = program->as.program.exprs[i];
        snprintf(debug_msg, sizeof(debug_msg), "Program expression %zu: type %d", i, expr->type);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        if (expr->type == AST_DEFINE) {
            AstNode* name = expr->as.define.name;
            AstNode* value = expr->as.define.value;
            
            if (name && name->type == AST_IDENTIFIER) {
                snprintf(debug_msg, sizeof(debug_msg), "Define name: %s", name->as.identifier.name);
                diagnostic_debug(diagnostics, 0, 0, debug_msg);
            } else {
                diagnostic_debug(diagnostics, 0, 0, "Define name is not an identifier or is NULL");
            }
            
            if (value) {
                snprintf(debug_msg, sizeof(debug_msg), "Define value type: %d", value->type);
                diagnostic_debug(diagnostics, 0, 0, debug_msg);
                
                if (value->type == AST_LAMBDA) {
                    snprintf(debug_msg, sizeof(debug_msg), "Lambda param count: %zu", value->as.lambda.param_count);
                    diagnostic_debug(diagnostics, 0, 0, debug_msg);
                    
                    if (value->as.lambda.body) {
                        snprintf(debug_msg, sizeof(debug_msg), "Lambda body type: %d", value->as.lambda.body->type);
                        diagnostic_debug(diagnostics, 0, 0, debug_msg);
                        
                        if (value->as.lambda.body->type == AST_BEGIN) {
                            snprintf(debug_msg, sizeof(debug_msg), "Begin has %zu expressions", 
                                    value->as.lambda.body->as.begin.expr_count);
                            diagnostic_debug(diagnostics, 0, 0, debug_msg);
                            
                            for (size_t j = 0; j < value->as.lambda.body->as.begin.expr_count; j++) {
                                AstNode* begin_expr = value->as.lambda.body->as.begin.exprs[j];
                                if (begin_expr) {
                                    snprintf(debug_msg, sizeof(debug_msg), "Begin expression %zu: type %d", j, begin_expr->type);
                                    diagnostic_debug(diagnostics, 0, 0, debug_msg);
                                    
                                    if (begin_expr->type == AST_CALL) {
                                        AstNode* callee = begin_expr->as.call.callee;
                                        if (callee && callee->type == AST_IDENTIFIER) {
                                            snprintf(debug_msg, sizeof(debug_msg), "Call function: %s", callee->as.identifier.name);
                                            diagnostic_debug(diagnostics, 0, 0, debug_msg);
                                        }
                                    }
                                } else {
                                    diagnostic_debug(diagnostics, 0, 0, "Begin expression is NULL");
                                }
                            }
                        }
                    } else {
                        diagnostic_debug(diagnostics, 0, 0, "Lambda body is NULL");
                    }
                }
            } else {
                diagnostic_debug(diagnostics, 0, 0, "Define value is NULL");
            }
        }
    }
    
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
    snprintf(debug_msg, sizeof(debug_msg), "Generating expressions (%zu total)", program->as.program.expr_count);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    for (size_t i = 0; i < program->as.program.expr_count; i++) {
        AstNode* expr = program->as.program.exprs[i];
        
        snprintf(debug_msg, sizeof(debug_msg), "Generating expression %zu of type %d", i, expr->type);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
        
        // Skip display and newline calls at the top level, as they'll be added to the main function
        if (expr->type == AST_CALL) {
            AstNode* callee = expr->as.call.callee;
            if (callee && callee->type == AST_IDENTIFIER) {
                const char* name = callee->as.identifier.name;
                if (strcmp(name, "display") == 0 || strcmp(name, "newline") == 0) {
                    snprintf(debug_msg, sizeof(debug_msg), "Skipping top-level %s call", name);
                    diagnostic_debug(diagnostics, 0, 0, debug_msg);
                    continue;
                }
            }
        }
        
        // Add more detailed debug info for specific node types
        if (expr->type == AST_DEFINE) {
            AstNode* name = expr->as.define.name;
            AstNode* value = expr->as.define.value;
            
            if (name && name->type == AST_IDENTIFIER) {
                snprintf(debug_msg, sizeof(debug_msg), "Define name: %s", name->as.identifier.name);
                diagnostic_debug(diagnostics, 0, 0, debug_msg);
            }
            
            if (value) {
                snprintf(debug_msg, sizeof(debug_msg), "Define value type: %d", value->type);
                diagnostic_debug(diagnostics, 0, 0, debug_msg);
                
                if (value->type == AST_LAMBDA) {
                    snprintf(debug_msg, sizeof(debug_msg), "Lambda param count: %zu", value->as.lambda.param_count);
                    diagnostic_debug(diagnostics, 0, 0, debug_msg);
                    
                    if (value->as.lambda.body) {
                        snprintf(debug_msg, sizeof(debug_msg), "Lambda body type: %d", value->as.lambda.body->type);
                        diagnostic_debug(diagnostics, 0, 0, debug_msg);
                    } else {
                        diagnostic_debug(diagnostics, 0, 0, "Lambda body is NULL");
                    }
                }
            }
        }
        
        // Try to generate the expression
        bool success = false;
        
        // Wrap the expression generation in a try-catch block
        FILE* output = codegen_context_get_output(context);
        if (output) {
            // Save the current position in the file
            long pos = ftell(output);
            
            // Try to generate the expression
            success = codegen_generate_expression(context, expr);
            
            if (!success) {
                // If generation failed, rewind the file to the saved position
                fseek(output, pos, SEEK_SET);
                
                snprintf(debug_msg, sizeof(debug_msg), "Failed to generate expression %zu of type %d", i, expr->type);
                diagnostic_error(diagnostics, 0, 0, debug_msg);
                
                // Print more detailed error information
                if (expr->type == AST_DEFINE) {
                    AstNode* name = expr->as.define.name;
                    if (name && name->type == AST_IDENTIFIER) {
                        snprintf(debug_msg, sizeof(debug_msg), "Failed to generate definition for '%s'", 
                                name->as.identifier.name);
                        diagnostic_error(diagnostics, 0, 0, debug_msg);
                    }
                }
                
                return false;
            }
            
            // Add newline after each expression
            fprintf(output, "\n");
        } else {
            diagnostic_error(diagnostics, 0, 0, "Output file is NULL");
            return false;
        }
        
        snprintf(debug_msg, sizeof(debug_msg), "Successfully generated expression %zu", i);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
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
