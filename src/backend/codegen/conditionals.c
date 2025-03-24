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
