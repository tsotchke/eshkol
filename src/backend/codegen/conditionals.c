/**
 * @file conditionals.c
 * @brief Conditional code generation implementation
 */

#include "backend/codegen/conditionals.h"
#include "backend/codegen/expressions.h"
#include "backend/codegen/type_conversion.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for an if expression
 */
bool codegen_generate_if(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_IF);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "((");
    
    // Generate condition
    if (!codegen_generate_expression(context, node->as.if_expr.condition)) {
        return false;
    }
    
    // Generate then branch
    fprintf(output, ") ? (");
    if (!codegen_generate_expression(context, node->as.if_expr.then_branch)) {
        return false;
    }
    
    // Generate else branch
    fprintf(output, ") : (");
    if (!codegen_generate_expression(context, node->as.if_expr.else_branch)) {
        return false;
    }
    
    // Close if
    fprintf(output, "))");
    
    return true;
}

/**
 * @brief Generate C code for a cond expression
 */
bool codegen_generate_cond(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_COND);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({");
    
    // Generate clauses
    for (size_t i = 0; i < node->as.cond.clause_count; i++) {
        AstNode* clause = node->as.cond.clauses[i];
        
        // Check if this is an else clause
        if (clause->type == AST_SEQUENCE && clause->as.sequence.expr_count == 2 &&
            clause->as.sequence.exprs[0]->type == AST_IDENTIFIER &&
            strcmp(clause->as.sequence.exprs[0]->as.identifier.name, "else") == 0) {
            
            // Generate else clause
            fprintf(output, " ");
            if (!codegen_generate_expression(context, clause->as.sequence.exprs[1])) {
                return false;
            }
            fprintf(output, ";");
        } else {
            // Generate if clause
            fprintf(output, " if (");
            
            // Generate condition
            if (!codegen_generate_expression(context, clause->as.sequence.exprs[0])) {
                return false;
            }
            
            // Generate body
            fprintf(output, ") { ");
            if (!codegen_generate_expression(context, clause->as.sequence.exprs[1])) {
                return false;
            }
            fprintf(output, "; }");
            
            // Add else if for next clause
            if (i < node->as.cond.clause_count - 1) {
                fprintf(output, " else");
            }
        }
    }
    
    // Close cond
    fprintf(output, " })");
    
    return true;
}

/**
 * @brief Generate C code for a case expression
 */
bool codegen_generate_case(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_CASE);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({ ");
    
    // Generate key
    const char* key_type = codegen_get_c_type(context, node->as.case_expr.key);
    fprintf(output, "%s _case_key = ", key_type);
    if (!codegen_generate_expression(context, node->as.case_expr.key)) {
        return false;
    }
    fprintf(output, "; ");
    
    // Generate clauses
    for (size_t i = 0; i < node->as.case_expr.clause_count; i++) {
        AstNode* clause = node->as.case_expr.clauses[i];
        
        // Check if this is an else clause
        if (clause->type == AST_SEQUENCE && clause->as.sequence.expr_count == 2 &&
            clause->as.sequence.exprs[0]->type == AST_IDENTIFIER &&
            strcmp(clause->as.sequence.exprs[0]->as.identifier.name, "else") == 0) {
            
            // Generate else clause
            fprintf(output, " else { ");
            if (!codegen_generate_expression(context, clause->as.sequence.exprs[1])) {
                return false;
            }
            fprintf(output, "; }");
        } else {
            // Generate if clause
            fprintf(output, " if (");
            
            // Generate condition (key == datum)
            if (clause->as.sequence.exprs[0]->type == AST_SEQUENCE) {
                // Multiple datums
                fprintf(output, "(");
                for (size_t j = 0; j < clause->as.sequence.exprs[0]->as.sequence.expr_count; j++) {
                    if (j > 0) {
                        fprintf(output, " || ");
                    }
                    fprintf(output, "_case_key == ");
                    if (!codegen_generate_expression(context, clause->as.sequence.exprs[0]->as.sequence.exprs[j])) {
                        return false;
                    }
                }
                fprintf(output, ")");
            } else {
                // Single datum
                fprintf(output, "_case_key == ");
                if (!codegen_generate_expression(context, clause->as.sequence.exprs[0])) {
                    return false;
                }
            }
            
            // Generate body
            fprintf(output, ") { ");
            if (!codegen_generate_expression(context, clause->as.sequence.exprs[1])) {
                return false;
            }
            fprintf(output, "; }");
            
            // Add else for next clause
            if (i < node->as.case_expr.clause_count - 1) {
                fprintf(output, " else");
            }
        }
    }
    
    // Close case
    fprintf(output, " })");
    
    return true;
}

/**
 * @brief Generate code for an if expression that returns a string
 */
char* codegen_generate_if_expr(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_IF);
    
    Arena* arena = codegen_context_get_arena(context);
    TypeInferenceContext* type_context = codegen_context_get_type_context(context);
    
    // Generate condition
    char* condition = codegen_generate_expression_str(context, node->as.if_expr.condition);
    if (!condition) return NULL;
    
    // Generate then branch
    char* then_branch = codegen_generate_expression_str(context, node->as.if_expr.then_branch);
    if (!then_branch) return NULL;
    
    // Generate else branch
    char* else_branch = NULL;
    if (node->as.if_expr.else_branch) {
        else_branch = codegen_generate_expression_str(context, node->as.if_expr.else_branch);
        if (!else_branch) return NULL;
    } else {
        else_branch = "0"; // Default to 0 for no else branch
    }
    
    // Get the types of the branches
    Type* then_type = type_inference_get_type(type_context, node->as.if_expr.then_branch);
    Type* else_type = node->as.if_expr.else_branch ? 
                     type_inference_get_type(type_context, node->as.if_expr.else_branch) :
                     type_void_create(arena);
    
    // Check if we have mixed types (e.g., number and string)
    if (then_type && else_type && 
        then_type->kind != else_type->kind &&
        then_type->kind != TYPE_VOID && else_type->kind != TYPE_VOID) {
        
        // Special case for mixed numeric types (integer and float)
        if ((then_type->kind == TYPE_INTEGER && else_type->kind == TYPE_FLOAT) ||
            (then_type->kind == TYPE_FLOAT && else_type->kind == TYPE_INTEGER)) {
            // Promote to float - this is handled automatically by the ternary operator
            char* result = arena_alloc(arena, strlen(condition) + strlen(then_branch) + strlen(else_branch) + 50);
            sprintf(result, "(%s ? %s : %s)", condition, then_branch, else_branch);
            return result;
        }
        
        // Special case for string and numeric types
        if ((then_type->kind == TYPE_STRING && (else_type->kind == TYPE_INTEGER || else_type->kind == TYPE_FLOAT)) ||
            (else_type->kind == TYPE_STRING && (then_type->kind == TYPE_INTEGER || then_type->kind == TYPE_FLOAT))) {
            
            // For scientific computing, we need to be careful with mixed types
            // We'll use void* as a generic return type
            
            // Convert both branches to void*
            char* then_converted = codegen_type_apply_conversion(context, then_branch, then_type, type_any_create(arena));
            char* else_converted = codegen_type_apply_conversion(context, else_branch, else_type, type_any_create(arena));
            
            // Generate the if expression with converted branches
            char* result = arena_alloc(arena, strlen(condition) + strlen(then_converted) + strlen(else_converted) + 50);
            sprintf(result, "(%s ? %s : %s)", condition, then_converted, else_converted);
            
            return result;
        }
    }
    
    // Generate the if expression
    char* result = arena_alloc(arena, strlen(condition) + strlen(then_branch) + strlen(else_branch) + 50);
    sprintf(result, "(%s ? %s : %s)", condition, then_branch, else_branch);
    
    return result;
}

/**
 * @brief Generate C code for an and expression
 */
bool codegen_generate_and(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_AND);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.logical.expr_count; i++) {
        if (i > 0) {
            fprintf(output, " && ");
        }
        
        // Generate expression
        fprintf(output, "(");
        if (!codegen_generate_expression(context, node->as.logical.exprs[i])) {
            return false;
        }
        fprintf(output, ")");
    }
    
    // Close and
    fprintf(output, ";})");
    
    return true;
}

/**
 * @brief Generate C code for an or expression
 */
bool codegen_generate_or(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_OR);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Generate code
    fprintf(output, "({");
    
    // Generate expressions
    for (size_t i = 0; i < node->as.logical.expr_count; i++) {
        if (i > 0) {
            fprintf(output, " || ");
        }
        
        // Generate expression
        fprintf(output, "(");
        if (!codegen_generate_expression(context, node->as.logical.exprs[i])) {
            return false;
        }
        fprintf(output, ")");
    }
    
    // Close or
    fprintf(output, ";})");
    
    return true;
}
