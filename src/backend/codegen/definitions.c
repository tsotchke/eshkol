/**
 * @file definitions.c
 * @brief Definition code generation implementation
 */

#include "backend/codegen/definitions.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
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
    assert(node->type == AST_DEFINE);
    assert(node->as.define.value->type == AST_LAMBDA);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Get function type
    Type* func_type = type_inference_get_type(type_context, node->as.define.value);
    
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
    if (node->as.define.name->type == AST_IDENTIFIER) {
        fprintf(output, "%s(", node->as.define.name->as.identifier.name);
    } else {
        fprintf(output, "_func_%zu_%zu(", node->line, node->column);
    }
    
    // Get lambda node
    AstNode* lambda = node->as.define.value;
    
    // Generate parameters
    for (size_t i = 0; i < lambda->as.lambda.param_count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        // Get parameter
        Parameter* param = lambda->as.lambda.params[i];
        
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
    fprintf(output, "{ return ");
    if (!codegen_generate_expression(context, lambda->as.lambda.body)) {
        return false;
    }
    fprintf(output, "; }");
    
    return true;
}

/**
 * @brief Generate C code for a variable definition
 */
bool codegen_generate_variable_def(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_DEFINE);
    
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
        fprintf(output, "%s = ", node->as.define.name->as.identifier.name);
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
    
    // Lambda expressions are not directly supported in C
    // We need to generate a function pointer, but this is complex
    // For now, we'll just generate an error
    
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    diagnostic_error(diagnostics, node->line, node->column, "Lambda expressions are not supported in C");
    return false;
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
        fprintf(output, "%s = ", node->as.set.name->as.identifier.name);
    } else {
        fprintf(output, "_var_%zu_%zu = ", node->line, node->column);
    }
    
    // Generate value
    if (!codegen_generate_expression(context, node->as.set.value)) {
        return false;
    }
    
    return true;
}
