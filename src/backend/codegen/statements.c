/**
 * @file statements.c
 * @brief Statement code generation implementation
 */

#include "backend/codegen/statements.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/definitions.h"
#include "backend/codegen/program.h"
#include "backend/codegen/type_conversion.h"
#include "frontend/type_inference/type_inference.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a function definition
 */
bool codegen_generate_function_def(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    assert(output != NULL);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Determine if this is a define or function_def node
    const AstNode* func_node = node;
    const AstNode* name_node = NULL;
    const AstNode* body_node = NULL;
    const AstNode** param_nodes = NULL;
    size_t param_count = 0;
    
    if (node->type == AST_DEFINE) {
        // Define node with lambda value
        if (node->as.define.value->type != AST_LAMBDA) {
            DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
            diagnostic_error(diagnostics, node->line, node->column, "Expected lambda in function definition");
            return false;
        }
        
        name_node = node->as.define.name;
        func_node = node->as.define.value;
        body_node = func_node->as.lambda.body;
        // Cast Parameter** to AstNode** - this is safe because we're only reading
        param_nodes = (const AstNode**)func_node->as.lambda.params;
        param_count = func_node->as.lambda.param_count;
    } else if (node->type == AST_FUNCTION_DEF) {
        // Function definition node
        name_node = node->as.function_def.name;
        body_node = node->as.function_def.body;
        // Cast to const AstNode** to avoid discarding qualifiers warning
        param_nodes = (const AstNode**)node->as.function_def.param_nodes;
        param_count = node->as.function_def.param_count;
    } else {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Invalid node type for function definition");
        return false;
    }
    
    // Get return type
    const char* return_type = "int32_t"; // Default to int32_t
    
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
    } else if (node->type == AST_FUNCTION_DEF && node->as.function_def.return_type != NULL) {
        return_type = codegen_type_to_c_type(node->as.function_def.return_type);
    } else if (body_node != NULL && body_node->type_info != NULL) {
        return_type = codegen_type_to_c_type(body_node->type_info);
    }
    
    // Generate function declaration
    fprintf(output, "%s ", return_type);
    
    // Generate function name
    if (name_node->type == AST_IDENTIFIER) {
        fprintf(output, "%s", name_node->as.identifier.name);
    } else {
        fprintf(output, "_func_%zu_%zu", node->line, node->column);
    }
    
    // Generate parameter list
    fprintf(output, "(");
    
    for (size_t i = 0; i < param_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        // Get parameter type
        const char* param_type = "int32_t"; // Default to int32_t
        const AstNode* param_node = param_nodes[i];
        
        // Try to get type from type inference context
        Type* param_resolved_type = type_inference_get_type(type_context, param_node);
        if (param_resolved_type != NULL) {
            param_type = codegen_type_to_c_type(param_resolved_type);
        } else if (param_node->type_info != NULL) {
            param_type = codegen_type_to_c_type(param_node->type_info);
        } else if (node->type == AST_FUNCTION_DEF && 
                  i < node->as.function_def.param_count && 
                  node->as.function_def.params[i]->type != NULL) {
            param_type = codegen_type_to_c_type(node->as.function_def.params[i]->type);
        }
        
        fprintf(output, "%s ", param_type);
        
        // Generate parameter name
        if (param_node->type == AST_IDENTIFIER) {
            fprintf(output, "%s", param_node->as.identifier.name);
        } else {
            fprintf(output, "param_%zu", i);
        }
    }
    
    fprintf(output, ") {\n");
    
    // Set in function flag
    codegen_context_set_in_function(context, true);
    
    // Increment indent level
    codegen_context_increment_indent(context);
    
    // Generate function body
    if (body_node != NULL) {
        // Write indentation
        codegen_context_write_indent(context);
        
        // Generate return statement if needed
        if (strcmp(return_type, "void") != 0) {
            fprintf(output, "return ");
        }
        
        // Generate body expression
        if (!codegen_generate_expression(context, body_node)) {
            return false;
        }
        
        fprintf(output, ";\n");
    }
    
    // Decrement indent level
    codegen_context_decrement_indent(context);
    
    // Reset in function flag
    codegen_context_set_in_function(context, false);
    
    // Close function
    fprintf(output, "}\n\n");
    
    return true;
}

/**
 * @brief Generate C code for a variable definition
 */
bool codegen_generate_variable_def(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    assert(output != NULL);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Determine if this is a define or variable_def node
    const AstNode* name_node = NULL;
    const AstNode* value_node = NULL;
    
    if (node->type == AST_DEFINE) {
        // Define node with non-lambda value
        name_node = node->as.define.name;
        value_node = node->as.define.value;
    } else if (node->type == AST_VARIABLE_DEF) {
        // Variable definition node
        name_node = node->as.variable_def.name;
        value_node = node->as.variable_def.value;
    } else {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Invalid node type for variable definition");
        return false;
    }
    
    // Get variable type
    const char* var_type = "int32_t"; // Default to int32_t
    
    // Try to get type from type inference context
    Type* resolved_type = type_inference_get_type(type_context, value_node);
    if (resolved_type != NULL) {
        var_type = codegen_type_to_c_type(resolved_type);
    } else if (value_node != NULL && value_node->type_info != NULL) {
        var_type = codegen_type_to_c_type(value_node->type_info);
    }
    
    // Write indentation if in function
    if (codegen_context_in_function(context)) {
        codegen_context_write_indent(context);
    }
    
    // Generate variable declaration
    fprintf(output, "%s ", var_type);
    
    // Generate variable name
    if (name_node->type == AST_IDENTIFIER) {
        fprintf(output, "%s", name_node->as.identifier.name);
    } else {
        fprintf(output, "_var_%zu_%zu", node->line, node->column);
    }
    
    // Generate variable initialization if value is provided
    if (value_node != NULL) {
        fprintf(output, " = ");
        if (!codegen_generate_expression(context, value_node)) {
            return false;
        }
    }
    
    return true;
}

/**
 * @brief Generate C code for a statement
 */
bool codegen_generate_statement(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code based on node type
    switch (node->type) {
        case AST_FUNCTION_DEF:
            return codegen_generate_function_def(context, node);
        case AST_VARIABLE_DEF:
            return codegen_generate_variable_def(context, node);
        case AST_DEFINE:
            // For define expressions, we need to determine if it's a function or variable
            if (node->as.define.value->type == AST_LAMBDA) {
                // Function definition
                if (!codegen_generate_function_def(context, node)) {
                    return false;
                }
            } else {
                // Variable definition
                if (!codegen_generate_variable_def(context, node)) {
                    return false;
                }
            }
            fprintf(output, ";\n");
            return true;
        case AST_PROGRAM:
            return codegen_generate_program(context, node);
        default:
            // For other node types, generate an expression
            if (!codegen_generate_expression(context, node)) {
                return false;
            }
            fprintf(output, ";\n");
            return true;
    }
}
