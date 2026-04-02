/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

#include <stdio.h>
#include <string.h>

static void print_indent(int indent) {
    for (int i = 0; i < indent; i++) {
        printf("  ");
    }
}

static const char* type_to_string(eshkol_type_t type) {
    switch (type) {
        case ESHKOL_INVALID: return "INVALID";
        case ESHKOL_UNTYPED: return "UNTYPED";
        case ESHKOL_UINT8: return "UINT8";
        case ESHKOL_UINT16: return "UINT16";
        case ESHKOL_UINT32: return "UINT32";
        case ESHKOL_UINT64: return "UINT64";
        case ESHKOL_INT8: return "INT8";
        case ESHKOL_INT16: return "INT16";
        case ESHKOL_INT32: return "INT32";
        case ESHKOL_INT64: return "INT64";
        case ESHKOL_DOUBLE: return "DOUBLE";
        case ESHKOL_STRING: return "STRING";
        case ESHKOL_FUNC: return "FUNC";
        case ESHKOL_VAR: return "VAR";
        case ESHKOL_OP: return "OP";
        case ESHKOL_BIGNUM_LITERAL: return "BIGNUM_LITERAL";
        case ESHKOL_SYMBOL: return "SYMBOL";
        default: return "UNKNOWN";
    }
}

static const char* op_to_string(eshkol_op_t op) {
    switch (op) {
        case ESHKOL_INVALID_OP: return "INVALID_OP";
        case ESHKOL_COMPOSE_OP: return "COMPOSE_OP";
        case ESHKOL_IF_OP: return "IF_OP";
        case ESHKOL_ADD_OP: return "ADD_OP";
        case ESHKOL_SUB_OP: return "SUB_OP";
        case ESHKOL_MUL_OP: return "MUL_OP";
        case ESHKOL_DIV_OP: return "DIV_OP";
        case ESHKOL_CALL_OP: return "CALL_OP";
        case ESHKOL_DEFINE_OP: return "DEFINE_OP";
        case ESHKOL_SEQUENCE_OP: return "SEQUENCE_OP";
        case ESHKOL_EXTERN_OP: return "EXTERN_OP";
        case ESHKOL_EXTERN_VAR_OP: return "EXTERN_VAR_OP";
        case ESHKOL_LAMBDA_OP: return "LAMBDA_OP";
        case ESHKOL_LET_OP: return "LET_OP";
        case ESHKOL_LET_STAR_OP: return "LET*_OP";
        case ESHKOL_LETREC_OP: return "LETREC_OP";
        case ESHKOL_LETREC_STAR_OP: return "LETREC*_OP";
        case ESHKOL_AND_OP: return "AND_OP";
        case ESHKOL_OR_OP: return "OR_OP";
        case ESHKOL_COND_OP: return "COND_OP";
        case ESHKOL_CASE_OP: return "CASE_OP";
        case ESHKOL_MATCH_OP: return "MATCH_OP";
        case ESHKOL_DO_OP: return "DO_OP";
        case ESHKOL_WHEN_OP: return "WHEN_OP";
        case ESHKOL_UNLESS_OP: return "UNLESS_OP";
        case ESHKOL_QUOTE_OP: return "QUOTE_OP";
        case ESHKOL_QUASIQUOTE_OP: return "QUASIQUOTE_OP";
        case ESHKOL_UNQUOTE_OP: return "UNQUOTE_OP";
        case ESHKOL_UNQUOTE_SPLICING_OP: return "UNQUOTE_SPLICING_OP";
        case ESHKOL_SET_OP: return "SET!_OP";
        case ESHKOL_DEFINE_TYPE_OP: return "DEFINE_TYPE_OP";
        case ESHKOL_IMPORT_OP: return "IMPORT_OP";
        case ESHKOL_REQUIRE_OP: return "REQUIRE_OP";
        case ESHKOL_PROVIDE_OP: return "PROVIDE_OP";
        case ESHKOL_WITH_REGION_OP: return "WITH_REGION_OP";
        case ESHKOL_OWNED_OP: return "OWNED_OP";
        case ESHKOL_MOVE_OP: return "MOVE_OP";
        case ESHKOL_BORROW_OP: return "BORROW_OP";
        case ESHKOL_SHARED_OP: return "SHARED_OP";
        case ESHKOL_WEAK_REF_OP: return "WEAK_REF_OP";
        case ESHKOL_TENSOR_OP: return "TENSOR_OP";
        case ESHKOL_DIFF_OP: return "DIFF_OP";
        case ESHKOL_DERIVATIVE_OP: return "DERIVATIVE_OP";
        case ESHKOL_GRADIENT_OP: return "GRADIENT_OP";
        case ESHKOL_JACOBIAN_OP: return "JACOBIAN_OP";
        case ESHKOL_HESSIAN_OP: return "HESSIAN_OP";
        case ESHKOL_DIVERGENCE_OP: return "DIVERGENCE_OP";
        case ESHKOL_CURL_OP: return "CURL_OP";
        case ESHKOL_LAPLACIAN_OP: return "LAPLACIAN_OP";
        case ESHKOL_DIRECTIONAL_DERIV_OP: return "DIRECTIONAL_DERIV_OP";
        case ESHKOL_TYPE_ANNOTATION_OP: return "TYPE_ANNOTATION_OP";
        case ESHKOL_FORALL_OP: return "FORALL_OP";
        case ESHKOL_GUARD_OP: return "GUARD_OP";
        case ESHKOL_RAISE_OP: return "RAISE_OP";
        case ESHKOL_LET_VALUES_OP: return "LET_VALUES_OP";
        case ESHKOL_LET_STAR_VALUES_OP: return "LET*_VALUES_OP";
        case ESHKOL_VALUES_OP: return "VALUES_OP";
        case ESHKOL_CALL_WITH_VALUES_OP: return "CALL_WITH_VALUES_OP";
        case ESHKOL_DEFINE_SYNTAX_OP: return "DEFINE_SYNTAX_OP";
        case ESHKOL_LET_SYNTAX_OP: return "LET_SYNTAX_OP";
        case ESHKOL_LETREC_SYNTAX_OP: return "LETREC_SYNTAX_OP";
        case ESHKOL_CALL_CC_OP: return "CALL_CC_OP";
        case ESHKOL_DYNAMIC_WIND_OP: return "DYNAMIC_WIND_OP";
        case ESHKOL_LOGIC_VAR_OP: return "LOGIC_VAR_OP";
        case ESHKOL_UNIFY_OP: return "UNIFY_OP";
        case ESHKOL_MAKE_SUBST_OP: return "MAKE_SUBST_OP";
        case ESHKOL_WALK_OP: return "WALK_OP";
        case ESHKOL_MAKE_FACT_OP: return "MAKE_FACT_OP";
        case ESHKOL_MAKE_KB_OP: return "MAKE_KB_OP";
        case ESHKOL_KB_ASSERT_OP: return "KB_ASSERT_OP";
        case ESHKOL_KB_QUERY_OP: return "KB_QUERY_OP";
        case ESHKOL_MAKE_FACTOR_GRAPH_OP: return "MAKE_FACTOR_GRAPH_OP";
        case ESHKOL_FG_ADD_FACTOR_OP: return "FG_ADD_FACTOR_OP";
        case ESHKOL_FG_INFER_OP: return "FG_INFER_OP";
        case ESHKOL_FREE_ENERGY_OP: return "FREE_ENERGY_OP";
        case ESHKOL_EXPECTED_FREE_ENERGY_OP: return "EXPECTED_FREE_ENERGY_OP";
        case ESHKOL_MAKE_WORKSPACE_OP: return "MAKE_WORKSPACE_OP";
        case ESHKOL_WS_REGISTER_OP: return "WS_REGISTER_OP";
        case ESHKOL_WS_STEP_OP: return "WS_STEP_OP";
        case ESHKOL_FG_UPDATE_CPT_OP: return "FG_UPDATE_CPT_OP";
        case ESHKOL_LOGIC_VAR_PRED_OP: return "LOGIC_VAR_PRED_OP";
        case ESHKOL_SUBSTITUTION_PRED_OP: return "SUBSTITUTION_PRED_OP";
        case ESHKOL_KB_PRED_OP: return "KB_PRED_OP";
        case ESHKOL_FACT_PRED_OP: return "FACT_PRED_OP";
        case ESHKOL_FACTOR_GRAPH_PRED_OP: return "FACTOR_GRAPH_PRED_OP";
        case ESHKOL_WORKSPACE_PRED_OP: return "WORKSPACE_PRED_OP";
        case ESHKOL_CASE_LAMBDA_OP: return "CASE_LAMBDA_OP";
        case ESHKOL_DEFINE_RECORD_TYPE_OP: return "DEFINE_RECORD_TYPE_OP";
        case ESHKOL_PARAMETERIZE_OP: return "PARAMETERIZE_OP";
        case ESHKOL_MAKE_PARAMETER_OP: return "MAKE_PARAMETER_OP";
        case ESHKOL_COND_EXPAND_OP: return "COND_EXPAND_OP";
        case ESHKOL_INCLUDE_OP: return "INCLUDE_OP";
        case ESHKOL_SYNTAX_ERROR_OP: return "SYNTAX_ERROR_OP";
        default: return "UNKNOWN_OP";
    }
}

static void print_operation(const eshkol_operations_t *op, int indent) {
    print_indent(indent);
    printf("Operation: %s\n", op_to_string(op->op));
    
    switch (op->op) {
        case ESHKOL_COMPOSE_OP:
            if (op->compose_op.func_a) {
                print_indent(indent + 1);
                printf("Function A:\n");
                eshkol_ast_pretty_print(op->compose_op.func_a, indent + 2);
            }
            if (op->compose_op.func_b) {
                print_indent(indent + 1);
                printf("Function B:\n");
                eshkol_ast_pretty_print(op->compose_op.func_b, indent + 2);
            }
            break;
            
        case ESHKOL_IF_OP:
            if (op->if_op.if_true) {
                print_indent(indent + 1);
                printf("If True:\n");
                print_operation(op->if_op.if_true, indent + 2);
            }
            if (op->if_op.if_false) {
                print_indent(indent + 1);
                printf("If False:\n");
                print_operation(op->if_op.if_false, indent + 2);
            }
            break;
            
        case ESHKOL_CALL_OP:
            // Check if this is an arithmetic operation and display as formula
            if (op->call_op.func && op->call_op.func->type == ESHKOL_VAR && op->call_op.func->variable.id) {
                const char* func_name = op->call_op.func->variable.id;
                
                if (strcmp(func_name, "+") == 0 || strcmp(func_name, "-") == 0 || 
                    strcmp(func_name, "*") == 0 || strcmp(func_name, "/") == 0) {
                    // Display as mathematical formula
                    print_indent(indent + 1);
                    printf("Arithmetic Formula: ");
                    
                    if (op->call_op.num_vars > 0) {
                        printf("(");
                        for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                            if (i > 0) printf(" %s ", func_name);
                            
                            // Print a simplified representation of the argument
                            eshkol_ast_t *arg = &op->call_op.variables[i];
                            if (arg->type == ESHKOL_INT64) {
                                printf("%lld", (long long)arg->int64_val);
                            } else if (arg->type == ESHKOL_DOUBLE) {
                                printf("%f", arg->double_val);
                            } else if (arg->type == ESHKOL_VAR && arg->variable.id) {
                                printf("%s", arg->variable.id);
                            } else if (arg->type == ESHKOL_OP) {
                                printf("(...)");  // Nested expression
                            } else {
                                printf("?");
                            }
                        }
                        printf(")\n");
                    } else {
                        printf("(%s)\n", func_name);
                    }
                    
                    // Also show detailed breakdown
                    print_indent(indent + 1);
                    printf("Function: %s\n", func_name);
                    if (op->call_op.variables && op->call_op.num_vars > 0) {
                        print_indent(indent + 1);
                        printf("Arguments (%llu):\n", (unsigned long long)op->call_op.num_vars);
                        for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                            print_indent(indent + 2);
                            printf("Arg[%llu]:\n", (unsigned long long)i);
                            eshkol_ast_pretty_print(&op->call_op.variables[i], indent + 3);
                        }
                    }
                } else {
                    // Regular function call display
                    print_indent(indent + 1);
                    printf("Function:\n");
                    eshkol_ast_pretty_print(op->call_op.func, indent + 2);
                    
                    if (op->call_op.variables && op->call_op.num_vars > 0) {
                        print_indent(indent + 1);
                        printf("Arguments (%llu):\n", (unsigned long long)op->call_op.num_vars);
                        for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                            print_indent(indent + 2);
                            printf("Arg[%llu]:\n", (unsigned long long)i);
                            eshkol_ast_pretty_print(&op->call_op.variables[i], indent + 3);
                        }
                    }
                }
            }
            break;
            
        case ESHKOL_DEFINE_OP:
            print_indent(indent + 1);
            printf("Name: %s\n", op->define_op.name ? op->define_op.name : "(null)");
            print_indent(indent + 1);
            printf("Is Function: %s\n", op->define_op.is_function ? "true" : "false");
            
            if (op->define_op.is_function && op->define_op.parameters && op->define_op.num_params > 0) {
                print_indent(indent + 1);
                printf("Parameters (%llu):\n", (unsigned long long)op->define_op.num_params);
                for (uint64_t i = 0; i < op->define_op.num_params; i++) {
                    print_indent(indent + 2);
                    printf("Param[%llu]:\n", (unsigned long long)i);
                    eshkol_ast_pretty_print(&op->define_op.parameters[i], indent + 3);
                }
            }
            
            if (op->define_op.value) {
                print_indent(indent + 1);
                printf("Value/Body:\n");
                eshkol_ast_pretty_print(op->define_op.value, indent + 2);
            }
            break;
            
        case ESHKOL_SEQUENCE_OP:
            print_indent(indent + 1);
            printf("Sequence with %llu expressions:\n", (unsigned long long)op->sequence_op.num_expressions);
            if (op->sequence_op.expressions && op->sequence_op.num_expressions > 0) {
                for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                    print_indent(indent + 2);
                    printf("Expression[%llu]:\n", (unsigned long long)i);
                    eshkol_ast_pretty_print(&op->sequence_op.expressions[i], indent + 3);
                }
            }
            break;
            
        case ESHKOL_ADD_OP:
        case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP:
        case ESHKOL_DIV_OP:
            print_indent(indent + 1);
            printf("(Arithmetic operation - arguments parsed separately)\n");
            break;
            
        default:
            // Print generic operation info with arguments if available
            if (op->call_op.variables && op->call_op.num_vars > 0) {
                print_indent(indent + 1);
                printf("Arguments (%llu):\n", (unsigned long long)op->call_op.num_vars);
                for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                    print_indent(indent + 2);
                    printf("Arg[%llu]:\n", (unsigned long long)i);
                    eshkol_ast_pretty_print(&op->call_op.variables[i], indent + 3);
                }
            }
            break;
    }
}

void eshkol_ast_pretty_print(const eshkol_ast_t *ast, int indent) {
    if (!ast) {
        print_indent(indent);
        printf("(null AST)\n");
        return;
    }
    
    print_indent(indent);
    printf("AST Node [%s]:\n", type_to_string(ast->type));
    
    switch (ast->type) {
        case ESHKOL_INVALID:
            print_indent(indent + 1);
            printf("(Invalid AST node)\n");
            break;
            
        case ESHKOL_UINT8:
            print_indent(indent + 1);
            printf("Value: %u\n", ast->uint8_val);
            break;
            
        case ESHKOL_UINT16:
            print_indent(indent + 1);
            printf("Value: %u\n", ast->uint16_val);
            break;
            
        case ESHKOL_UINT32:
            print_indent(indent + 1);
            printf("Value: %u\n", ast->uint32_val);
            break;
            
        case ESHKOL_UINT64:
            print_indent(indent + 1);
            printf("Value: %llu\n", (unsigned long long)ast->uint64_val);
            break;
            
        case ESHKOL_INT8:
            print_indent(indent + 1);
            printf("Value: %d\n", ast->int8_val);
            break;
            
        case ESHKOL_INT16:
            print_indent(indent + 1);
            printf("Value: %d\n", ast->int16_val);
            break;
            
        case ESHKOL_INT32:
            print_indent(indent + 1);
            printf("Value: %d\n", ast->int32_val);
            break;
            
        case ESHKOL_INT64:
            print_indent(indent + 1);
            printf("Value: %lld\n", (long long)ast->int64_val);
            break;
            
        case ESHKOL_DOUBLE:
            print_indent(indent + 1);
            printf("Value: %f\n", ast->double_val);
            break;
            
        case ESHKOL_STRING:
            print_indent(indent + 1);
            printf("Value: \"%s\" (size: %llu)\n",
                   ast->str_val.ptr ? ast->str_val.ptr : "(null)",
                   (unsigned long long)ast->str_val.size);
            break;

        case ESHKOL_BIGNUM_LITERAL:
            print_indent(indent + 1);
            printf("Value: %s (bignum literal)\n",
                   ast->str_val.ptr ? ast->str_val.ptr : "(null)");
            break;

        case ESHKOL_SYMBOL:
            print_indent(indent + 1);
            printf("Value: %s (symbol)\n",
                   ast->str_val.ptr ? ast->str_val.ptr : "(null)");
            break;

        case ESHKOL_VAR:
            print_indent(indent + 1);
            printf("ID: %s\n", ast->variable.id ? ast->variable.id : "(null)");
            if (ast->variable.data) {
                print_indent(indent + 1);
                printf("Data:\n");
                eshkol_ast_pretty_print(ast->variable.data, indent + 2);
            }
            break;
            
        case ESHKOL_FUNC:
            print_indent(indent + 1);
            printf("ID: %s\n", ast->eshkol_func.id ? ast->eshkol_func.id : "(null)");
            print_indent(indent + 1);
            printf("Is Lambda: %s\n", ast->eshkol_func.is_lambda ? "true" : "false");
            print_indent(indent + 1);
            printf("Num Variables: %llu\n", (unsigned long long)ast->eshkol_func.num_variables);
            
            if (ast->eshkol_func.variables && ast->eshkol_func.num_variables > 0) {
                print_indent(indent + 1);
                printf("Variables:\n");
                for (uint64_t i = 0; i < ast->eshkol_func.num_variables; i++) {
                    print_indent(indent + 2);
                    printf("Var[%llu]:\n", (unsigned long long)i);
                    eshkol_ast_pretty_print(&ast->eshkol_func.variables[i], indent + 3);
                }
            }
            
            if (ast->eshkol_func.func_commands) {
                print_indent(indent + 1);
                printf("Commands:\n");
                print_operation(ast->eshkol_func.func_commands, indent + 2);
            }
            break;
            
        case ESHKOL_OP:
            print_operation(&ast->operation, indent + 1);
            break;
            
        case ESHKOL_UNTYPED:
            print_indent(indent + 1);
            printf("Untyped data: %p\n", ast->untyped_data);
            break;
            
        default:
            print_indent(indent + 1);
            printf("(Unknown type)\n");
            break;
    }
}