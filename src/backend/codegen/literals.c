/**
 * @file literals.c
 * @brief Literal code generation implementation
 */

#include "backend/codegen/literals.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for a number literal
 */
bool codegen_generate_number_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_NUMBER);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "%g", node->as.number.value);
    
    return true;
}

/**
 * @brief Generate C code for a boolean literal
 */
bool codegen_generate_boolean_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_BOOLEAN);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "%s", node->as.boolean.value ? "true" : "false");
    
    return true;
}

/**
 * @brief Generate C code for a character literal
 */
bool codegen_generate_character_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_CHARACTER);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "'%c'", node->as.character.value);
    
    return true;
}

/**
 * @brief Generate C code for a string literal
 */
bool codegen_generate_string_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_STRING);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "\"%s\"", node->as.string.value);
    
    return true;
}

/**
 * @brief Generate C code for a vector literal
 */
bool codegen_generate_vector_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_VECTOR);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Get type context
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Get vector type
    Type* vector_type = type_inference_get_type(type_context, node);
    if (!vector_type) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Vector literal has no type");
        return false;
    }
    
    // Get element type
    Type* element_type = NULL;
    if (vector_type->kind == TYPE_VECTOR) {
        element_type = vector_type->vector.element_type;
    }
    
    if (!element_type) {
        DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
        diagnostic_error(diagnostics, node->line, node->column, "Vector type has no element type");
        return false;
    }
    
    // Get element type string
    const char* element_type_str = codegen_type_to_c_type(element_type);
    
    // Generate code
    fprintf(output, "vector_%s_create_from_array(arena, (", element_type_str);
    fprintf(output, "%s[]){", element_type_str);
    
    // Generate elements
    for (size_t i = 0; i < node->as.vector.count; i++) {
        if (i > 0) {
            fprintf(output, ", ");
        }
        
        // Generate element
        if (!codegen_generate_expression(context, node->as.vector.elements[i])) {
            return false;
        }
    }
    
    // Close vector
    fprintf(output, "}, %zu)", node->as.vector.count);
    
    return true;
}

/**
 * @brief Generate C code for a nil literal
 */
bool codegen_generate_nil_literal(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_LITERAL_NIL);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "NULL");
    
    return true;
}
