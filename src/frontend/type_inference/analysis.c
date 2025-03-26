#include "frontend/type_inference/analysis.h"
#include "frontend/type_inference/context.h"
#include "core/type.h"
#include "core/type_conversion.h"
#include <stdio.h>
#include <string.h>

// Forward declarations
static void print_expr_types(TypeInferenceContext* context, AstNode* node, int indent);
static void print_type_inference_details(TypeInferenceContext* context, AstNode* node);

bool type_inference_analyze(TypeInferenceContext* context, AstNode* ast, const char* detail_level) {
    printf("Type Analysis Results:\n");
    printf("=====================\n\n");

    if (strcmp(detail_level, "basic") == 0) {
        // Basic analysis just shows inferred types for top-level definitions
        printf("Top-level Definitions:\n");
        for (size_t i = 0; i < ast->as.program.expr_count; i++) {
            AstNode* expr = ast->as.program.exprs[i];
            if (expr->type == AST_FUNCTION_DEF) {
                printf("Function %s: ", expr->as.function_def.name->as.identifier.name);
                if (expr->inferred_type) {
                    printf("%s\n", type_to_string(type_inference_get_arena(context), expr->inferred_type));
                } else {
                    printf("(type unknown)\n");
                }
            } else if (expr->type == AST_VARIABLE_DEF) {
                printf("Variable %s: ", expr->as.variable_def.name->as.identifier.name);
                if (expr->inferred_type) {
                    printf("%s\n", type_to_string(type_inference_get_arena(context), expr->inferred_type));
                } else {
                    printf("(type unknown)\n");
                }
            }
        }
    } else if (strcmp(detail_level, "detailed") == 0) {
        // Detailed analysis shows types for all expressions
        printf("All Expressions:\n");
        print_expr_types(context, ast, 0);
    } else if (strcmp(detail_level, "verbose") == 0) {
        // Verbose analysis shows types, constraints, and inference steps
        printf("Type Inference Details:\n");
        print_type_inference_details(context, ast);
    }

    printf("\nAnalysis complete.\n");
    return true;
}

static void print_expr_types(TypeInferenceContext* context, AstNode* node, int indent) {
    for (int i = 0; i < indent; i++) printf("  ");
    
    switch (node->type) {
        case AST_FUNCTION_DEF:
            printf("Function %s: %s\n", 
                node->as.function_def.name->as.identifier.name,
                node->inferred_type ? type_to_string(type_inference_get_arena(context), node->inferred_type) : "(type unknown)");
            print_expr_types(context, node->as.function_def.body, indent + 1);
            break;
            
        case AST_VARIABLE_DEF:
            printf("Variable %s: %s\n",
                node->as.variable_def.name->as.identifier.name,
                node->inferred_type ? type_to_string(type_inference_get_arena(context), node->inferred_type) : "(type unknown)");
            print_expr_types(context, node->as.variable_def.value, indent + 1);
            break;
            
        case AST_CALL:
            printf("Call: %s\n", 
                node->inferred_type ? type_to_string(type_inference_get_arena(context), node->inferred_type) : "(type unknown)");
            print_expr_types(context, node->as.call.callee, indent + 1);
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                print_expr_types(context, node->as.call.args[i], indent + 1);
            }
            break;
            
        // Add cases for other expression types...
            
        default:
            printf("%s: %s\n",
                ast_node_type_to_string(node->type),
                node->inferred_type ? type_to_string(type_inference_get_arena(context), node->inferred_type) : "(type unknown)");
    }
}

static void print_type_inference_details(TypeInferenceContext* context, AstNode* node) {
    printf("Node: %s\n", ast_node_type_to_string(node->type));
    printf("Inferred Type: %s\n", 
        node->inferred_type ? type_to_string(type_inference_get_arena(context), node->inferred_type) : "(type unknown)");
    printf("Explicit Type: %s\n",
        node->type_info ? type_to_string(type_inference_get_arena(context), node->type_info) : "(none)");
    printf("Constraints:\n");
    // Print type constraints...
    printf("\n");
    
    // Recursively print details for child nodes
    switch (node->type) {
        case AST_FUNCTION_DEF:
            print_type_inference_details(context, node->as.function_def.body);
            break;
        case AST_VARIABLE_DEF:
            print_type_inference_details(context, node->as.variable_def.value);
            break;
        case AST_CALL:
            print_type_inference_details(context, node->as.call.callee);
            for (size_t i = 0; i < node->as.call.arg_count; i++) {
                print_type_inference_details(context, node->as.call.args[i]);
            }
            break;
        // Add cases for other node types...
    }
}
