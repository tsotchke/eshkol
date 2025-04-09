/**
 * @file definitions.c
 * @brief Definition code generation implementation
 */

#include "backend/codegen/definitions.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
#include "backend/codegen/closures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a define expression
 */
bool codegen_generate_define(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_DEFINE);
    
    // Check if this is a function or variable definition
    if (node->as.define.value->type == AST_LAMBDA) {
        // Function definition
        return codegen_generate_function_def(context, node);
    } else {
        // Variable definition
        return codegen_generate_variable_def(context, node);
    }
}

/**
 * @brief Generate C code for a function definition
 */
bool codegen_generate_function_def(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_DEFINE || node->type == AST_FUNCTION_DEF);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "Generating function definition for node type %d", node->type);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    // Variables to hold function information
    const AstNode* name_node = NULL;
    const AstNode* body_node = NULL;
    size_t param_count = 0;
    Parameter** params = NULL;
    
    // Get function information based on node type
    if (node->type == AST_DEFINE) {
        // Define node with lambda value
        if (node->as.define.value->type != AST_LAMBDA) {
            diagnostic_error(diagnostics, node->line, node->column, "Expected lambda in function definition");
            return false;
        }
        
        name_node = node->as.define.name;
        body_node = node->as.define.value->as.lambda.body;
        param_count = node->as.define.value->as.lambda.param_count;
        params = node->as.define.value->as.lambda.params;
    } else if (node->type == AST_FUNCTION_DEF) {
        // Function definition node
        name_node = node->as.function_def.name;
        body_node = node->as.function_def.body;
        param_count = node->as.function_def.param_count;
        params = node->as.function_def.params;
    } else {
        diagnostic_error(diagnostics, node->line, node->column, "Invalid node type for function definition");
        return false;
    }
    
    // Debug message
    if (name_node && name_node->type == AST_IDENTIFIER) {
        snprintf(debug_msg, sizeof(debug_msg), "Function name: %s", name_node->as.identifier.name);
        diagnostic_debug(diagnostics, 0, 0, debug_msg);
    }
    
    // Get function type
    Type* func_type = type_inference_get_type(type_context, node);
    
    // Get return type
    Type* return_type = NULL;
    if (func_type && func_type->kind == TYPE_FUNCTION) {
        return_type = func_type->function.return_type;
    }
    
    // Generate return type
    if (return_type) {
        fprintf(output, "%s ", codegen_type_to_c_type(return_type));
    } else {
        fprintf(output, "int ");
    }
    
    // Generate function name
    if (name_node && name_node->type == AST_IDENTIFIER) {
        // Replace hyphens with underscores in function names
        char* function_name = strdup(name_node->as.identifier.name);
        if (function_name) {
            for (char* p = function_name; *p; p++) {
                if (*p == '-') {
                    *p = '_';
                }
            }
            fprintf(output, "%s(", function_name);
            free(function_name);
        } else {
            fprintf(output, "%s(", name_node->as.identifier.name);
        }
    } else {
        fprintf(output, "_func_%zu_%zu(", node->line, node->column);
    }
    
    // Generate parameters
    for (size_t i = 0; i < param_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        // Get parameter
        Parameter* param = params[i];
        
        // Get parameter type
        Type* param_type = NULL;
        if (func_type && func_type->kind == TYPE_FUNCTION && i < func_type->function.param_count) {
            param_type = func_type->function.params[i];
        }
        
        // Generate parameter type
        if (param_type) {
            fprintf(output, "%s ", codegen_type_to_c_type(param_type));
        } else {
            fprintf(output, "int ");
        }
        
        // Generate parameter name
        fprintf(output, "%s", param->name);
    }
    
    // Close parameter list
    fprintf(output, ") ");
    
    // Generate function body
    fprintf(output, "{\n");
    
    // Add a loop for tail call optimization
    fprintf(output, "    // Tail call optimization loop\n");
    fprintf(output, "    while (1) {\n");
    
    // Check if the body is a begin block (multiple statements)
    if (body_node && body_node->type == AST_BEGIN) {
        // Generate each statement in the begin block
        for (size_t i = 0; i < body_node->as.begin.expr_count - 1; i++) {
            fprintf(output, "        ");
            if (!codegen_generate_expression(context, body_node->as.begin.exprs[i])) {
                return false;
            }
            fprintf(output, ";\n");
        }
        
        // Generate the last statement with a return
        if (body_node->as.begin.expr_count > 0) {
            fprintf(output, "        return ");
            if (!codegen_generate_expression(context, body_node->as.begin.exprs[body_node->as.begin.expr_count - 1])) {
                return false;
            }
            fprintf(output, ";\n");
        }
    } else if (body_node) {
        // Single statement body
        fprintf(output, "        return ");
        if (!codegen_generate_expression(context, body_node)) {
            return false;
        }
        fprintf(output, ";\n");
    } else {
        // No body
        fprintf(output, "        return 0;\n");
    }
    
    // Close the loop
    fprintf(output, "    }\n");
    fprintf(output, "}");
    
    return true;
}

/**
 * @brief Generate C code for a variable definition
 */
bool codegen_generate_variable_def(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Check node type
    if (node->type != AST_DEFINE) {
        char debug_msg[256];
        snprintf(debug_msg, sizeof(debug_msg), "Expected AST_DEFINE node, got node type %d", node->type);
        diagnostic_error(diagnostics, node->line, node->column, debug_msg);
        return false;
    }
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Get variable type
    Type* var_type = type_inference_get_type(type_context, node->as.define.value);
    
    // Generate type
    if (var_type) {
        fprintf(output, "%s ", codegen_type_to_c_type(var_type));
    } else {
        fprintf(output, "int ");
    }
    
    // Generate variable name
    if (node->as.define.name->type == AST_IDENTIFIER) {
        // Replace hyphens with underscores in variable names
        char* variable_name = strdup(node->as.define.name->as.identifier.name);
        if (variable_name) {
            for (char* p = variable_name; *p; p++) {
                if (*p == '-') {
                    *p = '_';
                }
            }
            fprintf(output, "%s = ", variable_name);
            free(variable_name);
        } else {
            fprintf(output, "%s = ", node->as.define.name->as.identifier.name);
        }
    } else {
        fprintf(output, "_var_%zu_%zu = ", node->line, node->column);
    }
    
    // Generate value
    if (!codegen_generate_expression(context, node->as.define.value)) {
        return false;
    }
    
    return true;
}

/**
 * @brief Generate C code for a lambda expression
 */
bool codegen_generate_lambda(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LAMBDA);
    
    // Use the implementation from closures.c
    return codegen_generate_closure_constructor(context, node);
}

/**
 * @brief Generate C code for a set! expression
 */
bool codegen_generate_set(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_SET);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate variable name
    if (node->as.set.name->type == AST_IDENTIFIER) {
        // Replace hyphens with underscores in variable names
        char* variable_name = strdup(node->as.set.name->as.identifier.name);
        if (variable_name) {
            for (char* p = variable_name; *p; p++) {
                if (*p == '-') {
                    *p = '_';
                }
            }
            fprintf(output, "%s = ", variable_name);
            free(variable_name);
        } else {
            fprintf(output, "%s = ", node->as.set.name->as.identifier.name);
        }
    } else {
        fprintf(output, "_var_%zu_%zu = ", node->line, node->column);
    }
    
    // Generate value
    if (!codegen_generate_expression(context, node->as.set.value)) {
        return false;
    }
    
    return true;
}
