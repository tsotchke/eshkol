/**
 * @file expressions.c
 * @brief Expression code generation implementation
 */

#include "backend/codegen/expressions.h"
#include "backend/codegen/literals.h"
#include "backend/codegen/identifiers.h"
#include "backend/codegen/calls.h"
#include "backend/codegen/conditionals.h"
#include "backend/codegen/blocks.h"
#include "backend/codegen/definitions.h"
#include "backend/codegen/type_conversion.h"
#include "backend/codegen/closures.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for an expression
 */
bool codegen_generate_expression(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "Generating expression for node type %d", node->type);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    // Generate code based on node type
    switch (node->type) {
        case AST_LITERAL_NUMBER:
            return codegen_generate_number_literal(context, node);
        case AST_LITERAL_BOOLEAN:
            return codegen_generate_boolean_literal(context, node);
        case AST_LITERAL_CHARACTER:
            return codegen_generate_character_literal(context, node);
        case AST_LITERAL_STRING:
            return codegen_generate_string_literal(context, node);
        case AST_LITERAL_VECTOR:
            return codegen_generate_vector_literal(context, node);
        case AST_LITERAL_NIL:
            return codegen_generate_nil_literal(context, node);
        case AST_IDENTIFIER:
            return codegen_generate_identifier(context, node);
        case AST_CALL:
            return codegen_generate_call(context, node);
        case AST_IF:
            return codegen_generate_if(context, node);
        case AST_BEGIN:
            return codegen_generate_begin(context, node);
        case AST_LAMBDA:
            return codegen_generate_lambda(context, node);
        case AST_DEFINE:
            return codegen_generate_define(context, node);
        case AST_FUNCTION_DEF:
            return codegen_generate_function_def(context, node);
        case AST_VARIABLE_DEF:
            return codegen_generate_variable_def(context, node);
        case AST_LET:
            return codegen_generate_let(context, node);
        case AST_LETREC:
            return codegen_generate_letrec(context, node);
        case AST_LETSTAR:
            return codegen_generate_letstar(context, node);
        case AST_AND:
            return codegen_generate_and(context, node);
        case AST_OR:
            return codegen_generate_or(context, node);
        case AST_COND:
            return codegen_generate_cond(context, node);
        case AST_CASE:
            return codegen_generate_case(context, node);
        case AST_DO:
            return codegen_generate_do(context, node);
        case AST_SET:
            return codegen_generate_set(context, node);
        case AST_SEQUENCE:
            return codegen_generate_sequence(context, node);
        default: {
            // Unsupported node type
            DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
            char error_message[100];
            snprintf(error_message, sizeof(error_message), "Unsupported AST node type for code generation: %d", node->type);
            diagnostic_error(diagnostics, node->line, node->column, error_message);
            return false;
        }
    }
}

/**
 * @brief Generate C code for an expression that returns a string
 */
char* codegen_generate_expression_str(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    
    // Get diagnostics context
    DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
    
    // Debug message
    char debug_msg[256];
    snprintf(debug_msg, sizeof(debug_msg), "Generating expression string for node type %d", node->type);
    diagnostic_debug(diagnostics, 0, 0, debug_msg);
    
    // Generate code based on node type
    switch (node->type) {
        case AST_IF:
            return codegen_generate_if_expr(context, node);
        default: {
            // Unsupported node type
            DiagnosticContext* diagnostics = codegen_context_get_diagnostics(context);
            char error_message[100];
            snprintf(error_message, sizeof(error_message), "Unsupported AST node type for string code generation: %d", node->type);
            diagnostic_error(diagnostics, node->line, node->column, error_message);
            return NULL;
        }
    }
}
