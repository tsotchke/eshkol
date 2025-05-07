/**
 * @file blocks.c
 * @brief Block code generation implementation
 */

#include "backend/codegen/blocks.h"
#include "backend/codegen/context.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a begin expression
 */
bool codegen_generate_begin(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_BEGIN);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.begin.expr_count; i++) {
        if (i > 0) {
            fprintf(output, "; ");
        }
        
        // Generate expression
        if (!codegen_generate_expression(context, node->as.begin.exprs[i])) {
            return false;
        }
    }
    
    // Close begin
    fprintf(output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a let expression
 */
bool codegen_generate_let(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LET);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate bindings
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        // Get binding
        AstNode* binding = node->as.let.bindings[i];
        
        // Get binding node (for type information)
        AstNode* binding_node = node->as.let.binding_nodes[i];
        
        // Get type context
        TypeInferenceContext* type_context = codegen_context_get_type_context(context);
        
        // Get variable type
        Type* var_type = type_inference_get_type(type_context, binding);
        
        // Generate type
        if (var_type) {
            fprintf(output, "%s ", codegen_type_to_c_type(var_type));
        } else {
            fprintf(output, "int ");
        }
        
        // Generate variable name
        if (binding_node->type == AST_IDENTIFIER) {
            fprintf(output, "%s = ", binding_node->as.identifier.name);
        } else {
            fprintf(output, "_let_var_%zu = ", i);
        }
        
        // Generate value
        if (!codegen_generate_expression(context, binding)) {
            return false;
        }
        
        fprintf(output, "; ");
    }
    
    // Generate body
    if (!codegen_generate_expression(context, node->as.let.body)) {
        return false;
    }
    
    // Close let
    fprintf(output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a letrec expression
 */
bool codegen_generate_letrec(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LETREC);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate forward declarations
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        // Get binding node (for type information)
        AstNode* binding_node = node->as.let.binding_nodes[i];
        
        // Get type context
        TypeInferenceContext* type_context = codegen_context_get_type_context(context);
        
        // Get variable type
        Type* var_type = type_inference_get_type(type_context, node->as.let.bindings[i]);
        
        // Generate type
        if (var_type) {
            fprintf(output, "%s ", codegen_type_to_c_type(var_type));
        } else {
            fprintf(output, "int ");
        }
        
        // Generate variable name
        if (binding_node->type == AST_IDENTIFIER) {
            fprintf(output, "%s; ", binding_node->as.identifier.name);
        } else {
            fprintf(output, "_letrec_var_%zu; ", i);
        }
    }
    
    // Generate bindings
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        // Get binding
        AstNode* binding = node->as.let.bindings[i];
        
        // Get binding node (for type information)
        AstNode* binding_node = node->as.let.binding_nodes[i];
        
        // Generate variable name
        if (binding_node->type == AST_IDENTIFIER) {
            fprintf(output, "%s = ", binding_node->as.identifier.name);
        } else {
            fprintf(output, "_letrec_var_%zu = ", i);
        }
        
        // Generate value
        if (!codegen_generate_expression(context, binding)) {
            return false;
        }
        
        fprintf(output, "; ");
    }
    
    // Generate body
    if (!codegen_generate_expression(context, node->as.let.body)) {
        return false;
    }
    
    // Close letrec
    fprintf(output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a let* expression
 */
bool codegen_generate_letstar(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LETSTAR);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate bindings
    for (size_t i = 0; i < node->as.let.binding_count; i++) {
        // Get binding
        AstNode* binding = node->as.let.bindings[i];
        
        // Get binding node (for type information)
        AstNode* binding_node = node->as.let.binding_nodes[i];
        
        // Get type context
        TypeInferenceContext* type_context = codegen_context_get_type_context(context);
        
        // Get variable type
        Type* var_type = type_inference_get_type(type_context, binding);
        
        // Generate type
        if (var_type) {
            fprintf(output, "%s ", codegen_type_to_c_type(var_type));
        } else {
            fprintf(output, "int ");
        }
        
        // Generate variable name
        if (binding_node->type == AST_IDENTIFIER) {
            fprintf(output, "%s = ", binding_node->as.identifier.name);
        } else {
            fprintf(output, "_letstar_var_%zu = ", i);
        }
        
        // Generate value
        if (!codegen_generate_expression(context, binding)) {
            return false;
        }
        
        fprintf(output, "; ");
    }
    
    // Generate body
    if (!codegen_generate_expression(context, node->as.let.body)) {
        return false;
    }
    
    // Close let*
    fprintf(output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a sequence of expressions
 */
bool codegen_generate_sequence(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_SEQUENCE);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.sequence.expr_count; i++) {
        if (i > 0) {
            fprintf(output, "; ");
        }
        
        // Generate expression
        if (!codegen_generate_expression(context, node->as.sequence.exprs[i])) {
            return false;
        }
    }
    
    // Close sequence
    fprintf(output, "; })");
    
    return true;
}

/**
 * @brief Generate C code for a do expression
 */
bool codegen_generate_do(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_DO);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
   
    // Definitions inside do blocks should be considered as being inside a function
    codegen_context_set_in_function(context, true);

    // Generate code
    fprintf(output, "({ ");
    
    // Generate variable declarations
    for (size_t i = 0; i < node->as.do_expr.binding_count; i++) {
        // Get binding
        AstNode* binding = node->as.do_expr.bindings[i];

       // Generate initial value
        if (!codegen_generate_expression(context, binding)) {
            return false;
        }
        
        fprintf(output, "; ");
    }
    
    // Generate loop
    fprintf(output, "while (!(");
    
    // Generate test
    if (!codegen_generate_expression(context, node->as.do_expr.test)) {
        return false;
    }
    
    fprintf(output, ")) { ");
    
    // Generate body
    for (size_t i = 0; i < node->as.do_expr.body_count; i++) {
        if (i > 0) {
            fprintf(output, "; ");
        }
        
        // Generate body expression
        if (!codegen_generate_expression(context, node->as.do_expr.body[i])) {
            return false;
        }
    }
    
    // Generate step
    fprintf(output, "; ");
    for (size_t j = 0; j < node->as.do_expr.binding_count; j++) {
        if (j > 0) {
            fprintf(output, ", ");
        }
        
        // Generate variable name - don't redeclare the type here
        AstNode* binding = node->as.do_expr.bindings[j];
        if(!codegen_generate_identifier(context, binding->as.define.name)) {
          return false;
    }
        fprintf(output, " = ");

        // Generate step value
        if (!codegen_generate_expression(context, node->as.do_expr.steps[j])) {
            return false;
        }
    }
    
    // Close loop
    fprintf(output, "; } ");
    
    // Generate result
    for (size_t i = 0; i < node->as.do_expr.result_count; i++) {
        if (i > 0) {
            fprintf(output, "; ");
        }
        
        // Generate result expression
        if (!codegen_generate_expression(context, node->as.do_expr.result[i])) {
            return false;
        }
    }
    
    // Close do
    fprintf(output, "; })");
    
    codegen_context_set_in_function(context, false);

    return true;
}
