/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#include <eshkol/eshkol.h>
#include <eshkol/logger.h>

void eshkol_ast_clean(eshkol_ast_t *ast)
{
    if (ast == nullptr) return;

    switch (ast->type) {
    case ESHKOL_STRING:
        if (ast->str_val.ptr != nullptr) delete [] ast->str_val.ptr;
        ast->str_val.ptr = nullptr;
        break;
    case ESHKOL_TENSOR:
        if (ast->tensor_val.elements != nullptr) {
            for (uint64_t i = 0; i < ast->tensor_val.total_elements; i++) {
                eshkol_ast_clean(&ast->tensor_val.elements[i]);
            }
            delete [] ast->tensor_val.elements;
            ast->tensor_val.elements = nullptr;
        }
        if (ast->tensor_val.dimensions != nullptr) {
            delete [] ast->tensor_val.dimensions;
            ast->tensor_val.dimensions = nullptr;
        }
        break;
    default:
        break;
    }

    ast->type = ESHKOL_INVALID;
}

// ===== SYMBOLIC DIFFERENTIATION AST HELPERS =====
// Memory management for symbolic AST nodes created during differentiation

eshkol_ast_t* eshkol_alloc_symbolic_ast() {
    eshkol_ast_t* node = (eshkol_ast_t*)malloc(sizeof(eshkol_ast_t));
    memset(node, 0, sizeof(eshkol_ast_t));
    return node;
}

// Helper: Create variable AST node
eshkol_ast_t* eshkol_make_var_ast(const char* name) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_VAR;
    ast->variable.id = strdup(name);
    ast->variable.data = nullptr;
    return ast;
}

// Helper: Create integer constant AST node
eshkol_ast_t* eshkol_make_int_ast(int64_t value) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_INT64;
    ast->int64_val = value;
    return ast;
}

// Helper: Create double constant AST node
eshkol_ast_t* eshkol_make_double_ast(double value) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_DOUBLE;
    ast->double_val = value;
    return ast;
}

// Helper: Create binary operation AST node (*, +, -, /)
eshkol_ast_t* eshkol_make_binary_op_ast(const char* op,
                                         eshkol_ast_t* left,
                                         eshkol_ast_t* right) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    
    // Create function AST for operator
    ast->operation.call_op.func = eshkol_make_var_ast(op);
    
    // Create arguments array
    ast->operation.call_op.variables =
        (eshkol_ast_t*)malloc(2 * sizeof(eshkol_ast_t));
    ast->operation.call_op.variables[0] = *left;
    ast->operation.call_op.variables[1] = *right;
    ast->operation.call_op.num_vars = 2;
    
    return ast;
}

// Helper: Create unary function call AST node (sin, cos, exp, log)
eshkol_ast_t* eshkol_make_unary_call_ast(const char* func, eshkol_ast_t* arg) {
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    
    ast->operation.call_op.func = eshkol_make_var_ast(func);
    ast->operation.call_op.variables =
        (eshkol_ast_t*)malloc(sizeof(eshkol_ast_t));
    ast->operation.call_op.variables[0] = *arg;
    ast->operation.call_op.num_vars = 1;
    
    return ast;
}

// Helper: Deep copy AST node
eshkol_ast_t* eshkol_copy_ast(const eshkol_ast_t* ast) {
    if (!ast) return nullptr;
    
    eshkol_ast_t* copy = eshkol_alloc_symbolic_ast();
    memcpy(copy, ast, sizeof(eshkol_ast_t));
    
    // Deep copy string fields if needed
    if (ast->type == ESHKOL_VAR && ast->variable.id) {
        copy->variable.id = strdup(ast->variable.id);
    }
    
    // Deep copy nested structures
    if (ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_CALL_OP) {
        if (ast->operation.call_op.func) {
            copy->operation.call_op.func = eshkol_copy_ast(ast->operation.call_op.func);
        }
        if (ast->operation.call_op.variables && ast->operation.call_op.num_vars > 0) {
            copy->operation.call_op.variables =
                (eshkol_ast_t*)malloc(ast->operation.call_op.num_vars * sizeof(eshkol_ast_t));
            for (uint64_t i = 0; i < ast->operation.call_op.num_vars; i++) {
                copy->operation.call_op.variables[i] = *eshkol_copy_ast(&ast->operation.call_op.variables[i]);
            }
        }
    }
    
    return copy;
}

// ===== HoTT TYPE EXPRESSION HELPERS =====
// Memory management and construction for HoTT type expressions

// Allocate and initialize a type expression with a given kind
static hott_type_expr_t* hott_alloc_type_expr(hott_type_kind_t kind) {
    hott_type_expr_t* type = (hott_type_expr_t*)malloc(sizeof(hott_type_expr_t));
    memset(type, 0, sizeof(hott_type_expr_t));
    type->kind = kind;
    return type;
}

// Primitive type constructors
hott_type_expr_t* hott_make_integer_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_INTEGER);
}

hott_type_expr_t* hott_make_real_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_REAL);
}

hott_type_expr_t* hott_make_boolean_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_BOOLEAN);
}

hott_type_expr_t* hott_make_string_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_STRING);
}

hott_type_expr_t* hott_make_char_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_CHAR);
}

hott_type_expr_t* hott_make_symbol_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_SYMBOL);
}

hott_type_expr_t* hott_make_null_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_NULL);
}

hott_type_expr_t* hott_make_any_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_ANY);
}

hott_type_expr_t* hott_make_nothing_type(void) {
    return hott_alloc_type_expr(HOTT_TYPE_NOTHING);
}

// Type variable constructor
hott_type_expr_t* hott_make_type_var(const char* name) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_VAR);
    type->var_name = strdup(name);
    return type;
}

// Arrow (function) type: (-> param1 param2 ... return)
hott_type_expr_t* hott_make_arrow_type(hott_type_expr_t** param_types,
                                        uint64_t num_params,
                                        hott_type_expr_t* return_type) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_ARROW);

    // Copy parameter types
    if (num_params > 0 && param_types) {
        type->arrow.param_types = (hott_type_expr_t**)malloc(num_params * sizeof(hott_type_expr_t*));
        for (uint64_t i = 0; i < num_params; i++) {
            type->arrow.param_types[i] = hott_copy_type_expr(param_types[i]);
        }
    } else {
        type->arrow.param_types = nullptr;
    }
    type->arrow.num_params = num_params;
    type->arrow.return_type = hott_copy_type_expr(return_type);

    return type;
}

// Container types
hott_type_expr_t* hott_make_list_type(hott_type_expr_t* element_type) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_LIST);
    type->container.element_type = hott_copy_type_expr(element_type);
    return type;
}

hott_type_expr_t* hott_make_vector_type(hott_type_expr_t* element_type) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_VECTOR);
    type->container.element_type = hott_copy_type_expr(element_type);
    return type;
}

hott_type_expr_t* hott_make_tensor_type(hott_type_expr_t* element_type) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_TENSOR);
    type->container.element_type = hott_copy_type_expr(element_type);
    return type;
}

// Pair and product types
hott_type_expr_t* hott_make_pair_type(hott_type_expr_t* left, hott_type_expr_t* right) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_PAIR);
    type->pair.left = hott_copy_type_expr(left);
    type->pair.right = hott_copy_type_expr(right);
    return type;
}

hott_type_expr_t* hott_make_product_type(hott_type_expr_t* left, hott_type_expr_t* right) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_PRODUCT);
    type->pair.left = hott_copy_type_expr(left);
    type->pair.right = hott_copy_type_expr(right);
    return type;
}

// Sum type
hott_type_expr_t* hott_make_sum_type(hott_type_expr_t* left, hott_type_expr_t* right) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_SUM);
    type->sum.left = hott_copy_type_expr(left);
    type->sum.right = hott_copy_type_expr(right);
    return type;
}

// Forall (polymorphic) type: (forall (a b ...) body-type)
hott_type_expr_t* hott_make_forall_type(char** type_vars,
                                         uint64_t num_vars,
                                         hott_type_expr_t* body) {
    hott_type_expr_t* type = hott_alloc_type_expr(HOTT_TYPE_FORALL);

    if (num_vars > 0 && type_vars) {
        type->forall.type_vars = (char**)malloc(num_vars * sizeof(char*));
        for (uint64_t i = 0; i < num_vars; i++) {
            type->forall.type_vars[i] = strdup(type_vars[i]);
        }
    } else {
        type->forall.type_vars = nullptr;
    }
    type->forall.num_vars = num_vars;
    type->forall.body = hott_copy_type_expr(body);

    return type;
}

// Deep copy a type expression
hott_type_expr_t* hott_copy_type_expr(const hott_type_expr_t* type) {
    if (!type) return nullptr;

    hott_type_expr_t* copy = hott_alloc_type_expr(type->kind);

    switch (type->kind) {
        case HOTT_TYPE_VAR:
            copy->var_name = type->var_name ? strdup(type->var_name) : nullptr;
            break;

        case HOTT_TYPE_ARROW:
            if (type->arrow.num_params > 0 && type->arrow.param_types) {
                copy->arrow.param_types = (hott_type_expr_t**)malloc(
                    type->arrow.num_params * sizeof(hott_type_expr_t*));
                for (uint64_t i = 0; i < type->arrow.num_params; i++) {
                    copy->arrow.param_types[i] = hott_copy_type_expr(type->arrow.param_types[i]);
                }
            }
            copy->arrow.num_params = type->arrow.num_params;
            copy->arrow.return_type = hott_copy_type_expr(type->arrow.return_type);
            break;

        case HOTT_TYPE_FORALL:
            if (type->forall.num_vars > 0 && type->forall.type_vars) {
                copy->forall.type_vars = (char**)malloc(type->forall.num_vars * sizeof(char*));
                for (uint64_t i = 0; i < type->forall.num_vars; i++) {
                    copy->forall.type_vars[i] = strdup(type->forall.type_vars[i]);
                }
            }
            copy->forall.num_vars = type->forall.num_vars;
            copy->forall.body = hott_copy_type_expr(type->forall.body);
            break;

        case HOTT_TYPE_LIST:
        case HOTT_TYPE_VECTOR:
            copy->container.element_type = hott_copy_type_expr(type->container.element_type);
            break;

        case HOTT_TYPE_PAIR:
        case HOTT_TYPE_PRODUCT:
            copy->pair.left = hott_copy_type_expr(type->pair.left);
            copy->pair.right = hott_copy_type_expr(type->pair.right);
            break;

        case HOTT_TYPE_SUM:
            copy->sum.left = hott_copy_type_expr(type->sum.left);
            copy->sum.right = hott_copy_type_expr(type->sum.right);
            break;

        case HOTT_TYPE_UNIVERSE:
            copy->universe.level = type->universe.level;
            break;

        default:
            // Primitive types have no additional data
            break;
    }

    return copy;
}

// Free a type expression and all its children
void hott_free_type_expr(hott_type_expr_t* type) {
    if (!type) return;

    switch (type->kind) {
        case HOTT_TYPE_VAR:
            if (type->var_name) free(type->var_name);
            break;

        case HOTT_TYPE_ARROW:
            if (type->arrow.param_types) {
                for (uint64_t i = 0; i < type->arrow.num_params; i++) {
                    hott_free_type_expr(type->arrow.param_types[i]);
                }
                free(type->arrow.param_types);
            }
            hott_free_type_expr(type->arrow.return_type);
            break;

        case HOTT_TYPE_FORALL:
            if (type->forall.type_vars) {
                for (uint64_t i = 0; i < type->forall.num_vars; i++) {
                    free(type->forall.type_vars[i]);
                }
                free(type->forall.type_vars);
            }
            hott_free_type_expr(type->forall.body);
            break;

        case HOTT_TYPE_LIST:
        case HOTT_TYPE_VECTOR:
            hott_free_type_expr(type->container.element_type);
            break;

        case HOTT_TYPE_PAIR:
        case HOTT_TYPE_PRODUCT:
            hott_free_type_expr(type->pair.left);
            hott_free_type_expr(type->pair.right);
            break;

        case HOTT_TYPE_SUM:
            hott_free_type_expr(type->sum.left);
            hott_free_type_expr(type->sum.right);
            break;

        default:
            break;
    }

    free(type);
}

// Convert type expression to string (for display/error messages)
// Returns a newly allocated string that must be freed by the caller
char* hott_type_to_string(const hott_type_expr_t* type) {
    if (!type) return strdup("null");

    char buffer[1024];
    buffer[0] = '\0';

    switch (type->kind) {
        case HOTT_TYPE_INVALID:
            return strdup("invalid");
        case HOTT_TYPE_INTEGER:
            return strdup("integer");
        case HOTT_TYPE_REAL:
            return strdup("real");
        case HOTT_TYPE_BOOLEAN:
            return strdup("boolean");
        case HOTT_TYPE_STRING:
            return strdup("string");
        case HOTT_TYPE_CHAR:
            return strdup("char");
        case HOTT_TYPE_SYMBOL:
            return strdup("symbol");
        case HOTT_TYPE_NULL:
            return strdup("null");
        case HOTT_TYPE_ANY:
            return strdup("any");
        case HOTT_TYPE_NOTHING:
            return strdup("nothing");

        case HOTT_TYPE_VAR:
            return strdup(type->var_name ? type->var_name : "?");

        case HOTT_TYPE_ARROW: {
            strcpy(buffer, "(-> ");
            for (uint64_t i = 0; i < type->arrow.num_params; i++) {
                char* param_str = hott_type_to_string(type->arrow.param_types[i]);
                strcat(buffer, param_str);
                strcat(buffer, " ");
                free(param_str);
            }
            char* ret_str = hott_type_to_string(type->arrow.return_type);
            strcat(buffer, ret_str);
            strcat(buffer, ")");
            free(ret_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_FORALL: {
            strcpy(buffer, "(forall (");
            for (uint64_t i = 0; i < type->forall.num_vars; i++) {
                if (i > 0) strcat(buffer, " ");
                strcat(buffer, type->forall.type_vars[i]);
            }
            strcat(buffer, ") ");
            char* body_str = hott_type_to_string(type->forall.body);
            strcat(buffer, body_str);
            strcat(buffer, ")");
            free(body_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_LIST: {
            strcpy(buffer, "(list ");
            char* elem_str = hott_type_to_string(type->container.element_type);
            strcat(buffer, elem_str);
            strcat(buffer, ")");
            free(elem_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_VECTOR: {
            strcpy(buffer, "(vector ");
            char* elem_str = hott_type_to_string(type->container.element_type);
            strcat(buffer, elem_str);
            strcat(buffer, ")");
            free(elem_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_PAIR: {
            strcpy(buffer, "(pair ");
            char* left_str = hott_type_to_string(type->pair.left);
            char* right_str = hott_type_to_string(type->pair.right);
            strcat(buffer, left_str);
            strcat(buffer, " ");
            strcat(buffer, right_str);
            strcat(buffer, ")");
            free(left_str);
            free(right_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_PRODUCT: {
            strcpy(buffer, "(* ");
            char* left_str = hott_type_to_string(type->pair.left);
            char* right_str = hott_type_to_string(type->pair.right);
            strcat(buffer, left_str);
            strcat(buffer, " ");
            strcat(buffer, right_str);
            strcat(buffer, ")");
            free(left_str);
            free(right_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_SUM: {
            strcpy(buffer, "(+ ");
            char* left_str = hott_type_to_string(type->sum.left);
            char* right_str = hott_type_to_string(type->sum.right);
            strcat(buffer, left_str);
            strcat(buffer, " ");
            strcat(buffer, right_str);
            strcat(buffer, ")");
            free(left_str);
            free(right_str);
            return strdup(buffer);
        }

        case HOTT_TYPE_UNIVERSE:
            snprintf(buffer, sizeof(buffer), "Type%u", type->universe.level);
            return strdup(buffer);

        case HOTT_TYPE_DEPENDENT:
            return strdup("(Pi ...)");

        case HOTT_TYPE_PATH:
            return strdup("(Path ...)");

        default:
            return strdup("unknown");
    }
}

// ===== REPL DISPLAY HELPER =====

// Helper: Wrap expression in (begin (display expr) (newline))
// Used by REPL to automatically display expression results
eshkol_ast_t* eshkol_wrap_with_display(eshkol_ast_t* expr) {
    // Create: (begin (display expr) (newline))
    eshkol_ast_t* wrapper = eshkol_alloc_symbolic_ast();
    wrapper->type = ESHKOL_OP;
    wrapper->operation.op = ESHKOL_CALL_OP;

    // Function: "begin"
    wrapper->operation.call_op.func = eshkol_make_var_ast("begin");
    wrapper->operation.call_op.num_vars = 2;
    wrapper->operation.call_op.variables =
        (eshkol_ast_t*)malloc(2 * sizeof(eshkol_ast_t));

    // Element 1: (display expr)
    eshkol_ast_t* display_call = eshkol_alloc_symbolic_ast();
    display_call->type = ESHKOL_OP;
    display_call->operation.op = ESHKOL_CALL_OP;
    display_call->operation.call_op.func = eshkol_make_var_ast("display");
    display_call->operation.call_op.num_vars = 1;
    display_call->operation.call_op.variables =
        (eshkol_ast_t*)malloc(sizeof(eshkol_ast_t));
    display_call->operation.call_op.variables[0] = *expr;

    // Element 2: (newline)
    eshkol_ast_t* newline_call = eshkol_alloc_symbolic_ast();
    newline_call->type = ESHKOL_OP;
    newline_call->operation.op = ESHKOL_CALL_OP;
    newline_call->operation.call_op.func = eshkol_make_var_ast("newline");
    newline_call->operation.call_op.num_vars = 0;
    newline_call->operation.call_op.variables = nullptr;

    wrapper->operation.call_op.variables[0] = *display_call;
    wrapper->operation.call_op.variables[1] = *newline_call;

    return wrapper;
}
