/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * Direct S-expression to AST Conversion
 *
 * High-performance direct conversion from runtime S-expressions (cons cells)
 * to AST nodes, avoiding the serialize→parse overhead. This enables efficient
 * runtime eval/compile for metaprogramming and self-modifying code.
 */

#include <eshkol/core/sexp_to_ast.h>
#include <eshkol/core/introspection.h>
#include <eshkol/logger.h>
#include "arena_memory.h"

#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>

namespace {

// ============================================================================
// Helper Functions
// ============================================================================

// Create a null value
inline eshkol_tagged_value_t make_null() {
    return ESHKOL_MAKE_NULL_VALUE();
}

// Check if value is null
inline bool is_null(eshkol_tagged_value_t value) {
    return value.type == ESHKOL_VALUE_NULL;
}

// Check if value is a pair/cons cell
inline bool is_pair(eshkol_tagged_value_t value) {
    return ESHKOL_IS_CONS_COMPAT(value);
}

// Check if value is a symbol
inline bool is_symbol(eshkol_tagged_value_t value) {
    return value.type == ESHKOL_VALUE_SYMBOL;
}

// Check if value is a string
inline bool is_string(eshkol_tagged_value_t value) {
    return ESHKOL_IS_STRING_COMPAT(value);
}

// Get raw pointer from tagged value
inline void* get_ptr(eshkol_tagged_value_t value) {
    return reinterpret_cast<void*>(value.data.ptr_val);
}

// Get CAR of a pair
inline eshkol_tagged_value_t pair_car(eshkol_tagged_value_t pair) {
    if (!is_pair(pair)) {
        return make_null();
    }
    arena_tagged_cons_cell_t* cell = static_cast<arena_tagged_cons_cell_t*>(get_ptr(pair));
    return cell ? cell->car : make_null();
}

// Get CDR of a pair
inline eshkol_tagged_value_t pair_cdr(eshkol_tagged_value_t pair) {
    if (!is_pair(pair)) {
        return make_null();
    }
    arena_tagged_cons_cell_t* cell = static_cast<arena_tagged_cons_cell_t*>(get_ptr(pair));
    return cell ? cell->cdr : make_null();
}

// Get symbol name from tagged value (returns pointer to interned string)
inline const char* get_symbol_name(eshkol_tagged_value_t value) {
    if (!is_symbol(value)) {
        return nullptr;
    }
    return static_cast<const char*>(get_ptr(value));
}

// Get string value from tagged value
inline const char* get_string_value(eshkol_tagged_value_t value) {
    if (!is_string(value)) {
        return nullptr;
    }
    return static_cast<const char*>(get_ptr(value));
}

// Count elements in a proper list
size_t list_length(eshkol_tagged_value_t list) {
    size_t count = 0;
    eshkol_tagged_value_t current = list;
    while (is_pair(current)) {
        count++;
        current = pair_cdr(current);
    }
    return count;
}

// Get nth element of a list (0-indexed)
eshkol_tagged_value_t list_ref(eshkol_tagged_value_t list, size_t index) {
    eshkol_tagged_value_t current = list;
    for (size_t i = 0; i < index && is_pair(current); i++) {
        current = pair_cdr(current);
    }
    return is_pair(current) ? pair_car(current) : make_null();
}

// Collect list elements into a vector
std::vector<eshkol_tagged_value_t> list_to_vector(eshkol_tagged_value_t list) {
    std::vector<eshkol_tagged_value_t> result;
    eshkol_tagged_value_t current = list;
    while (is_pair(current)) {
        result.push_back(pair_car(current));
        current = pair_cdr(current);
    }
    return result;
}

// Check if symbol matches a name
inline bool symbol_eq(eshkol_tagged_value_t value, const char* name) {
    const char* sym_name = get_symbol_name(value);
    return sym_name && strcmp(sym_name, name) == 0;
}

// Forward declaration for recursive conversion
eshkol_ast_t* convert_sexp(eshkol_tagged_value_t sexp);

// ============================================================================
// Special Form Handlers
// ============================================================================

// Convert lambda: (lambda (params...) body...)
eshkol_ast_t* convert_lambda(eshkol_tagged_value_t sexp) {
    // (lambda (params...) body)
    // Get params list (second element)
    eshkol_tagged_value_t params_sexp = list_ref(sexp, 1);
    // Get body (third element onwards)
    eshkol_tagged_value_t body_start = pair_cdr(pair_cdr(sexp));  // Skip lambda and params

    if (is_null(body_start) || !is_pair(body_start)) {
        eshkol_error("sexp_to_ast: lambda requires at least one body expression");
        return nullptr;
    }

    // Collect parameters
    std::vector<eshkol_tagged_value_t> param_sexps = list_to_vector(params_sexp);
    uint64_t num_params = param_sexps.size();

    // Check for variadic (rest parameter with dot notation)
    bool is_variadic = false;
    char* rest_param = nullptr;

    // Convert parameters to AST
    eshkol_ast_t* params = nullptr;
    if (num_params > 0) {
        params = static_cast<eshkol_ast_t*>(malloc(num_params * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < num_params; i++) {
            const char* param_name = get_symbol_name(param_sexps[i]);
            if (param_name) {
                params[i].type = ESHKOL_VAR;
                params[i].variable.id = strdup(param_name);
                params[i].variable.data = nullptr;
            } else {
                eshkol_error("sexp_to_ast: lambda parameter must be a symbol");
                free(params);
                return nullptr;
            }
        }
    }

    // Collect body expressions
    std::vector<eshkol_tagged_value_t> body_sexps = list_to_vector(body_start);

    // Convert body - if single expression, use directly; otherwise wrap in begin
    eshkol_ast_t* body = nullptr;
    if (body_sexps.size() == 1) {
        body = convert_sexp(body_sexps[0]);
    } else {
        // Create a sequence (begin) for multiple body expressions
        eshkol_ast_t* seq = eshkol_alloc_symbolic_ast();
        seq->type = ESHKOL_OP;
        seq->operation.op = ESHKOL_SEQUENCE_OP;
        seq->operation.sequence_op.num_expressions = body_sexps.size();
        seq->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
            malloc(body_sexps.size() * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < body_sexps.size(); i++) {
            eshkol_ast_t* expr = convert_sexp(body_sexps[i]);
            if (expr) {
                seq->operation.sequence_op.expressions[i] = *expr;
                free(expr);
            }
        }
        body = seq;
    }

    if (!body) {
        free(params);
        return nullptr;
    }

    // Create lambda AST
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_FUNC;
    ast->eshkol_func.id = nullptr;  // Anonymous lambda
    ast->eshkol_func.is_lambda = 1;
    ast->eshkol_func.func_commands = static_cast<eshkol_operations_t*>(
        malloc(sizeof(eshkol_operations_t)));
    ast->eshkol_func.func_commands->op = ESHKOL_LAMBDA_OP;
    ast->eshkol_func.func_commands->lambda_op.parameters = params;
    ast->eshkol_func.func_commands->lambda_op.num_params = num_params;
    ast->eshkol_func.func_commands->lambda_op.body = body;
    ast->eshkol_func.func_commands->lambda_op.captured_vars = nullptr;
    ast->eshkol_func.func_commands->lambda_op.num_captured = 0;
    ast->eshkol_func.func_commands->lambda_op.is_variadic = is_variadic;
    ast->eshkol_func.func_commands->lambda_op.rest_param = rest_param;
    ast->eshkol_func.func_commands->lambda_op.return_type = nullptr;
    ast->eshkol_func.func_commands->lambda_op.param_types = nullptr;
    ast->eshkol_func.variables = params;
    ast->eshkol_func.num_variables = num_params;
    ast->eshkol_func.size = 0;
    ast->eshkol_func.is_variadic = is_variadic;
    ast->eshkol_func.rest_param = rest_param;
    ast->eshkol_func.param_types = nullptr;
    ast->eshkol_func.return_type = nullptr;

    return ast;
}

// Convert define: (define name value) or (define (name params...) body...)
eshkol_ast_t* convert_define(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t second = list_ref(sexp, 1);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_DEFINE_OP;

    if (is_symbol(second)) {
        // Simple define: (define name value)
        const char* name = get_symbol_name(second);
        eshkol_tagged_value_t value_sexp = list_ref(sexp, 2);

        ast->operation.define_op.name = strdup(name);
        ast->operation.define_op.value = convert_sexp(value_sexp);
        ast->operation.define_op.is_function = 0;
        ast->operation.define_op.parameters = nullptr;
        ast->operation.define_op.num_params = 0;
        ast->operation.define_op.is_variadic = 0;
        ast->operation.define_op.rest_param = nullptr;
        ast->operation.define_op.is_external = 0;
        ast->operation.define_op.return_type = nullptr;
        ast->operation.define_op.param_types = nullptr;
    } else if (is_pair(second)) {
        // Function define: (define (name params...) body...)
        eshkol_tagged_value_t name_sexp = pair_car(second);
        const char* name = get_symbol_name(name_sexp);

        if (!name) {
            eshkol_error("sexp_to_ast: define function name must be a symbol");
            free(ast);
            return nullptr;
        }

        // Get parameters (rest of second element)
        std::vector<eshkol_tagged_value_t> param_sexps = list_to_vector(pair_cdr(second));
        uint64_t num_params = param_sexps.size();

        // Convert parameters
        eshkol_ast_t* params = nullptr;
        if (num_params > 0) {
            params = static_cast<eshkol_ast_t*>(malloc(num_params * sizeof(eshkol_ast_t)));
            for (size_t i = 0; i < num_params; i++) {
                const char* param_name = get_symbol_name(param_sexps[i]);
                if (param_name) {
                    params[i].type = ESHKOL_VAR;
                    params[i].variable.id = strdup(param_name);
                    params[i].variable.data = nullptr;
                }
            }
        }

        // Get body
        eshkol_tagged_value_t body_start = pair_cdr(pair_cdr(sexp));
        std::vector<eshkol_tagged_value_t> body_sexps = list_to_vector(body_start);

        eshkol_ast_t* body = nullptr;
        if (body_sexps.size() == 1) {
            body = convert_sexp(body_sexps[0]);
        } else if (body_sexps.size() > 1) {
            // Create sequence for multiple body expressions
            eshkol_ast_t* seq = eshkol_alloc_symbolic_ast();
            seq->type = ESHKOL_OP;
            seq->operation.op = ESHKOL_SEQUENCE_OP;
            seq->operation.sequence_op.num_expressions = body_sexps.size();
            seq->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
                malloc(body_sexps.size() * sizeof(eshkol_ast_t)));
            for (size_t i = 0; i < body_sexps.size(); i++) {
                eshkol_ast_t* expr = convert_sexp(body_sexps[i]);
                if (expr) {
                    seq->operation.sequence_op.expressions[i] = *expr;
                    free(expr);
                }
            }
            body = seq;
        }

        ast->operation.define_op.name = strdup(name);
        ast->operation.define_op.value = body;
        ast->operation.define_op.is_function = 1;
        ast->operation.define_op.parameters = params;
        ast->operation.define_op.num_params = num_params;
        ast->operation.define_op.is_variadic = 0;
        ast->operation.define_op.rest_param = nullptr;
        ast->operation.define_op.is_external = 0;
        ast->operation.define_op.return_type = nullptr;
        ast->operation.define_op.param_types = nullptr;
    } else {
        eshkol_error("sexp_to_ast: invalid define form");
        free(ast);
        return nullptr;
    }

    return ast;
}

// Convert if: (if test then else?)
eshkol_ast_t* convert_if(eshkol_tagged_value_t sexp) {
    size_t len = list_length(sexp);
    if (len < 3) {
        eshkol_error("sexp_to_ast: if requires at least test and then clause");
        return nullptr;
    }

    eshkol_ast_t* test_ast = convert_sexp(list_ref(sexp, 1));
    eshkol_ast_t* then_ast = convert_sexp(list_ref(sexp, 2));
    eshkol_ast_t* else_ast = (len >= 4) ? convert_sexp(list_ref(sexp, 3)) : nullptr;

    if (!test_ast || !then_ast) {
        return nullptr;
    }

    // Create if as a call representation
    // The parser creates if as CALL_OP with "if" as the function,
    // and the codegen handles this representation directly.
    // This matches the parser's output format.
    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    ast->operation.call_op.func = eshkol_make_var_ast("if");

    // Arguments: test, then, else (if present)
    int num_args = else_ast ? 3 : 2;
    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(num_args * sizeof(eshkol_ast_t)));
    ast->operation.call_op.variables[0] = *test_ast;
    ast->operation.call_op.variables[1] = *then_ast;
    if (else_ast) {
        ast->operation.call_op.variables[2] = *else_ast;
    }
    ast->operation.call_op.num_vars = num_args;

    free(test_ast);
    free(then_ast);
    if (else_ast) free(else_ast);

    return ast;
}

// Convert let: (let ((var val) ...) body...)
eshkol_ast_t* convert_let(eshkol_tagged_value_t sexp, eshkol_op_t let_type) {
    // (let bindings body...)
    // bindings = ((var1 val1) (var2 val2) ...)

    eshkol_tagged_value_t bindings_sexp = list_ref(sexp, 1);
    eshkol_tagged_value_t body_start = pair_cdr(pair_cdr(sexp));

    std::vector<eshkol_tagged_value_t> binding_pairs = list_to_vector(bindings_sexp);
    uint64_t num_bindings = binding_pairs.size();

    // Create bindings array - each binding is a pair (var, value)
    eshkol_ast_t* bindings = nullptr;
    if (num_bindings > 0) {
        // Each binding takes 2 AST nodes (var and value)
        bindings = static_cast<eshkol_ast_t*>(malloc(num_bindings * 2 * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < num_bindings; i++) {
            eshkol_tagged_value_t binding = binding_pairs[i];
            eshkol_tagged_value_t var = pair_car(binding);
            eshkol_tagged_value_t val = list_ref(binding, 1);

            const char* var_name = get_symbol_name(var);
            if (!var_name) {
                eshkol_error("sexp_to_ast: let binding variable must be a symbol");
                free(bindings);
                return nullptr;
            }

            // Variable
            bindings[i * 2].type = ESHKOL_VAR;
            bindings[i * 2].variable.id = strdup(var_name);
            bindings[i * 2].variable.data = nullptr;

            // Value
            eshkol_ast_t* val_ast = convert_sexp(val);
            if (val_ast) {
                bindings[i * 2 + 1] = *val_ast;
                free(val_ast);
            }
        }
    }

    // Convert body
    std::vector<eshkol_tagged_value_t> body_sexps = list_to_vector(body_start);
    eshkol_ast_t* body = nullptr;

    if (body_sexps.size() == 1) {
        body = convert_sexp(body_sexps[0]);
    } else if (body_sexps.size() > 1) {
        eshkol_ast_t* seq = eshkol_alloc_symbolic_ast();
        seq->type = ESHKOL_OP;
        seq->operation.op = ESHKOL_SEQUENCE_OP;
        seq->operation.sequence_op.num_expressions = body_sexps.size();
        seq->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
            malloc(body_sexps.size() * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < body_sexps.size(); i++) {
            eshkol_ast_t* expr = convert_sexp(body_sexps[i]);
            if (expr) {
                seq->operation.sequence_op.expressions[i] = *expr;
                free(expr);
            }
        }
        body = seq;
    }

    if (!body && body_sexps.empty()) {
        eshkol_error("sexp_to_ast: let requires at least one body expression");
        free(bindings);
        return nullptr;
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = let_type;
    ast->operation.let_op.bindings = bindings;
    ast->operation.let_op.num_bindings = num_bindings;
    ast->operation.let_op.body = body;
    ast->operation.let_op.name = nullptr;  // Not a named let
    ast->operation.let_op.binding_types = nullptr;

    return ast;
}

// Convert quote: (quote expr) or 'expr
eshkol_ast_t* convert_quote(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t quoted = list_ref(sexp, 1);

    // Create a QUOTE_OP with the quoted expression stored in call_op structure.
    // This matches the parser's representation where quote stores its
    // argument in call_op.variables[0] for the codegen to access.

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_QUOTE_OP;
    ast->operation.call_op.func = nullptr;

    // Convert the quoted expression to AST representation
    eshkol_ast_t* quoted_ast = convert_sexp(quoted);
    if (quoted_ast) {
        ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
            malloc(sizeof(eshkol_ast_t)));
        ast->operation.call_op.variables[0] = *quoted_ast;
        ast->operation.call_op.num_vars = 1;
        free(quoted_ast);
    } else {
        ast->operation.call_op.variables = nullptr;
        ast->operation.call_op.num_vars = 0;
    }

    return ast;
}

// Convert begin: (begin expr...)
eshkol_ast_t* convert_begin(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t exprs_start = pair_cdr(sexp);
    std::vector<eshkol_tagged_value_t> expr_sexps = list_to_vector(exprs_start);

    if (expr_sexps.empty()) {
        // Empty begin returns void
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_NULL;
        return ast;
    }

    if (expr_sexps.size() == 1) {
        return convert_sexp(expr_sexps[0]);
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_SEQUENCE_OP;
    ast->operation.sequence_op.num_expressions = expr_sexps.size();
    ast->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
        malloc(expr_sexps.size() * sizeof(eshkol_ast_t)));

    for (size_t i = 0; i < expr_sexps.size(); i++) {
        eshkol_ast_t* expr = convert_sexp(expr_sexps[i]);
        if (expr) {
            ast->operation.sequence_op.expressions[i] = *expr;
            free(expr);
        }
    }

    return ast;
}

// Convert set!: (set! var value)
eshkol_ast_t* convert_set(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t var_sexp = list_ref(sexp, 1);
    eshkol_tagged_value_t val_sexp = list_ref(sexp, 2);

    const char* var_name = get_symbol_name(var_sexp);
    if (!var_name) {
        eshkol_error("sexp_to_ast: set! variable must be a symbol");
        return nullptr;
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_SET_OP;
    ast->operation.set_op.name = strdup(var_name);
    ast->operation.set_op.value = convert_sexp(val_sexp);

    return ast;
}

// Convert and: (and expr1 expr2 ...)
// Short-circuit AND - returns first false value or last value
eshkol_ast_t* convert_and(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t exprs_start = pair_cdr(sexp);  // Skip 'and'
    std::vector<eshkol_tagged_value_t> expr_sexps = list_to_vector(exprs_start);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_AND_OP;
    ast->operation.sequence_op.num_expressions = expr_sexps.size();

    if (expr_sexps.empty()) {
        // (and) with no arguments returns #t
        ast->operation.sequence_op.expressions = nullptr;
    } else {
        ast->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
            malloc(expr_sexps.size() * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < expr_sexps.size(); i++) {
            eshkol_ast_t* expr = convert_sexp(expr_sexps[i]);
            if (expr) {
                ast->operation.sequence_op.expressions[i] = *expr;
                free(expr);
            }
        }
    }

    return ast;
}

// Convert or: (or expr1 expr2 ...)
// Short-circuit OR - returns first true value or last value
eshkol_ast_t* convert_or(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t exprs_start = pair_cdr(sexp);  // Skip 'or'
    std::vector<eshkol_tagged_value_t> expr_sexps = list_to_vector(exprs_start);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_OR_OP;
    ast->operation.sequence_op.num_expressions = expr_sexps.size();

    if (expr_sexps.empty()) {
        // (or) with no arguments returns #f
        ast->operation.sequence_op.expressions = nullptr;
    } else {
        ast->operation.sequence_op.expressions = static_cast<eshkol_ast_t*>(
            malloc(expr_sexps.size() * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < expr_sexps.size(); i++) {
            eshkol_ast_t* expr = convert_sexp(expr_sexps[i]);
            if (expr) {
                ast->operation.sequence_op.expressions[i] = *expr;
                free(expr);
            }
        }
    }

    return ast;
}

// Convert cond: (cond (test expr...) ... (else expr...))
eshkol_ast_t* convert_cond(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t clauses_start = pair_cdr(sexp);  // Skip 'cond'
    std::vector<eshkol_tagged_value_t> clause_sexps = list_to_vector(clauses_start);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_COND_OP;
    ast->operation.call_op.func = nullptr;  // cond doesn't have a function
    ast->operation.call_op.num_vars = clause_sexps.size();

    if (clause_sexps.empty()) {
        ast->operation.call_op.variables = nullptr;
    } else {
        ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
            malloc(clause_sexps.size() * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < clause_sexps.size(); i++) {
            // Each clause is a list (test expr...)
            eshkol_ast_t* clause_ast = convert_sexp(clause_sexps[i]);
            if (clause_ast) {
                ast->operation.call_op.variables[i] = *clause_ast;
                free(clause_ast);
            }
        }
    }

    return ast;
}

// Convert when: (when test expr...)
// One-armed if - executes body only when test is true
eshkol_ast_t* convert_when(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t rest = pair_cdr(sexp);  // Skip 'when'
    std::vector<eshkol_tagged_value_t> elements = list_to_vector(rest);

    if (elements.empty()) {
        eshkol_error("sexp_to_ast: when requires at least a test expression");
        return nullptr;
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_WHEN_OP;
    ast->operation.call_op.func = nullptr;
    ast->operation.call_op.num_vars = elements.size();

    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(elements.size() * sizeof(eshkol_ast_t)));
    for (size_t i = 0; i < elements.size(); i++) {
        eshkol_ast_t* elem = convert_sexp(elements[i]);
        if (elem) {
            ast->operation.call_op.variables[i] = *elem;
            free(elem);
        }
    }

    return ast;
}

// Convert unless: (unless test expr...)
// Negated when - executes body only when test is false
eshkol_ast_t* convert_unless(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t rest = pair_cdr(sexp);  // Skip 'unless'
    std::vector<eshkol_tagged_value_t> elements = list_to_vector(rest);

    if (elements.empty()) {
        eshkol_error("sexp_to_ast: unless requires at least a test expression");
        return nullptr;
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_UNLESS_OP;
    ast->operation.call_op.func = nullptr;
    ast->operation.call_op.num_vars = elements.size();

    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(elements.size() * sizeof(eshkol_ast_t)));
    for (size_t i = 0; i < elements.size(); i++) {
        eshkol_ast_t* elem = convert_sexp(elements[i]);
        if (elem) {
            ast->operation.call_op.variables[i] = *elem;
            free(elem);
        }
    }

    return ast;
}

// Convert case: (case key ((datum...) expr...) ... (else expr...))
eshkol_ast_t* convert_case(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t key_sexp = list_ref(sexp, 1);
    eshkol_tagged_value_t clauses_start = pair_cdr(pair_cdr(sexp));  // Skip 'case' and key
    std::vector<eshkol_tagged_value_t> clause_sexps = list_to_vector(clauses_start);

    eshkol_ast_t* key_ast = convert_sexp(key_sexp);
    if (!key_ast) {
        eshkol_error("sexp_to_ast: case requires a key expression");
        return nullptr;
    }

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CASE_OP;
    ast->operation.call_op.func = key_ast;
    ast->operation.call_op.num_vars = clause_sexps.size();

    if (clause_sexps.empty()) {
        ast->operation.call_op.variables = nullptr;
    } else {
        ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
            malloc(clause_sexps.size() * sizeof(eshkol_ast_t)));

        for (size_t i = 0; i < clause_sexps.size(); i++) {
            // Each clause is ((datum...) expr...) or (else expr...)
            eshkol_tagged_value_t clause = clause_sexps[i];
            eshkol_tagged_value_t first = pair_car(clause);

            eshkol_ast_t clause_ast;
            clause_ast.type = ESHKOL_CONS;

            // Check for else clause
            if (is_symbol(first) && symbol_eq(first, "else")) {
                // else clause - create marker
                eshkol_ast_t* else_marker = eshkol_make_var_ast("else");
                clause_ast.cons_cell.car = else_marker;

                // Rest is body expressions
                std::vector<eshkol_tagged_value_t> body_exprs = list_to_vector(pair_cdr(clause));
                eshkol_ast_t* body = eshkol_alloc_symbolic_ast();
                body->type = ESHKOL_OP;
                body->operation.op = ESHKOL_CALL_OP;
                body->operation.call_op.func = nullptr;
                body->operation.call_op.num_vars = body_exprs.size();
                if (body_exprs.size() > 0) {
                    body->operation.call_op.variables = static_cast<eshkol_ast_t*>(
                        malloc(body_exprs.size() * sizeof(eshkol_ast_t)));
                    for (size_t j = 0; j < body_exprs.size(); j++) {
                        eshkol_ast_t* expr = convert_sexp(body_exprs[j]);
                        if (expr) {
                            body->operation.call_op.variables[j] = *expr;
                            free(expr);
                        }
                    }
                } else {
                    body->operation.call_op.variables = nullptr;
                }
                clause_ast.cons_cell.cdr = body;
            } else {
                // Normal clause - first is datums list
                std::vector<eshkol_tagged_value_t> datums = list_to_vector(first);

                eshkol_ast_t* datums_ast = eshkol_alloc_symbolic_ast();
                datums_ast->type = ESHKOL_OP;
                datums_ast->operation.op = ESHKOL_CALL_OP;
                datums_ast->operation.call_op.func = nullptr;
                datums_ast->operation.call_op.num_vars = datums.size();
                if (datums.size() > 0) {
                    datums_ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
                        malloc(datums.size() * sizeof(eshkol_ast_t)));
                    for (size_t j = 0; j < datums.size(); j++) {
                        eshkol_ast_t* datum = convert_sexp(datums[j]);
                        if (datum) {
                            datums_ast->operation.call_op.variables[j] = *datum;
                            free(datum);
                        }
                    }
                } else {
                    datums_ast->operation.call_op.variables = nullptr;
                }
                clause_ast.cons_cell.car = datums_ast;

                // Body expressions
                std::vector<eshkol_tagged_value_t> body_exprs = list_to_vector(pair_cdr(clause));
                eshkol_ast_t* body = eshkol_alloc_symbolic_ast();
                body->type = ESHKOL_OP;
                body->operation.op = ESHKOL_CALL_OP;
                body->operation.call_op.func = nullptr;
                body->operation.call_op.num_vars = body_exprs.size();
                if (body_exprs.size() > 0) {
                    body->operation.call_op.variables = static_cast<eshkol_ast_t*>(
                        malloc(body_exprs.size() * sizeof(eshkol_ast_t)));
                    for (size_t j = 0; j < body_exprs.size(); j++) {
                        eshkol_ast_t* expr = convert_sexp(body_exprs[j]);
                        if (expr) {
                            body->operation.call_op.variables[j] = *expr;
                            free(expr);
                        }
                    }
                } else {
                    body->operation.call_op.variables = nullptr;
                }
                clause_ast.cons_cell.cdr = body;
            }

            ast->operation.call_op.variables[i] = clause_ast;
        }
    }

    return ast;
}

// Convert quasiquote: (quasiquote expr) or `expr
eshkol_ast_t* convert_quasiquote(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t quoted = list_ref(sexp, 1);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_QUASIQUOTE_OP;
    ast->operation.call_op.func = nullptr;
    ast->operation.call_op.num_vars = 1;
    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(sizeof(eshkol_ast_t)));

    // Convert the quasiquoted expression
    eshkol_ast_t* quoted_ast = convert_sexp(quoted);
    if (quoted_ast) {
        ast->operation.call_op.variables[0] = *quoted_ast;
        free(quoted_ast);
    }

    return ast;
}

// Convert unquote: (unquote expr) or ,expr
eshkol_ast_t* convert_unquote(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t expr = list_ref(sexp, 1);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_UNQUOTE_OP;
    ast->operation.call_op.func = nullptr;
    ast->operation.call_op.num_vars = 1;
    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(sizeof(eshkol_ast_t)));

    eshkol_ast_t* expr_ast = convert_sexp(expr);
    if (expr_ast) {
        ast->operation.call_op.variables[0] = *expr_ast;
        free(expr_ast);
    }

    return ast;
}

// Convert unquote-splicing: (unquote-splicing expr) or ,@expr
eshkol_ast_t* convert_unquote_splicing(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t expr = list_ref(sexp, 1);

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_UNQUOTE_SPLICING_OP;
    ast->operation.call_op.func = nullptr;
    ast->operation.call_op.num_vars = 1;
    ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
        malloc(sizeof(eshkol_ast_t)));

    eshkol_ast_t* expr_ast = convert_sexp(expr);
    if (expr_ast) {
        ast->operation.call_op.variables[0] = *expr_ast;
        free(expr_ast);
    }

    return ast;
}

// Convert function call: (func args...)
eshkol_ast_t* convert_call(eshkol_tagged_value_t sexp) {
    eshkol_tagged_value_t func_sexp = pair_car(sexp);
    eshkol_tagged_value_t args_start = pair_cdr(sexp);

    eshkol_ast_t* func_ast = convert_sexp(func_sexp);
    if (!func_ast) {
        return nullptr;
    }

    std::vector<eshkol_tagged_value_t> arg_sexps = list_to_vector(args_start);
    uint64_t num_args = arg_sexps.size();

    eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
    ast->type = ESHKOL_OP;
    ast->operation.op = ESHKOL_CALL_OP;
    ast->operation.call_op.func = func_ast;
    ast->operation.call_op.num_vars = num_args;

    if (num_args > 0) {
        ast->operation.call_op.variables = static_cast<eshkol_ast_t*>(
            malloc(num_args * sizeof(eshkol_ast_t)));
        for (size_t i = 0; i < num_args; i++) {
            eshkol_ast_t* arg_ast = convert_sexp(arg_sexps[i]);
            if (arg_ast) {
                ast->operation.call_op.variables[i] = *arg_ast;
                free(arg_ast);
            }
        }
    } else {
        ast->operation.call_op.variables = nullptr;
    }

    return ast;
}

// ============================================================================
// Main Conversion Function
// ============================================================================

eshkol_ast_t* convert_sexp(eshkol_tagged_value_t sexp) {
    // Handle atomic values

    // Null
    if (is_null(sexp)) {
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_NULL;
        return ast;
    }

    // Integer
    if (sexp.type == ESHKOL_VALUE_INT64) {
        return eshkol_make_int_ast(sexp.data.int_val);
    }

    // Double
    if (sexp.type == ESHKOL_VALUE_DOUBLE) {
        return eshkol_make_double_ast(sexp.data.double_val);
    }

    // Boolean
    if (sexp.type == ESHKOL_VALUE_BOOL) {
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_BOOL;
        ast->int64_val = sexp.data.int_val;  // 0 for false, 1 for true
        return ast;
    }

    // Symbol -> Variable reference
    if (is_symbol(sexp)) {
        const char* name = get_symbol_name(sexp);
        return eshkol_make_var_ast(name);
    }

    // String
    if (is_string(sexp)) {
        const char* str = get_string_value(sexp);
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_STRING;
        ast->str_val.ptr = strdup(str);
        ast->str_val.size = strlen(str);
        return ast;
    }

    // Pair/List -> Either special form or function call
    if (is_pair(sexp)) {
        eshkol_tagged_value_t head = pair_car(sexp);

        // Check for special forms
        if (is_symbol(head)) {
            const char* head_name = get_symbol_name(head);

            if (strcmp(head_name, "lambda") == 0) {
                return convert_lambda(sexp);
            }
            if (strcmp(head_name, "define") == 0) {
                return convert_define(sexp);
            }
            if (strcmp(head_name, "if") == 0) {
                return convert_if(sexp);
            }
            if (strcmp(head_name, "let") == 0) {
                return convert_let(sexp, ESHKOL_LET_OP);
            }
            if (strcmp(head_name, "let*") == 0) {
                return convert_let(sexp, ESHKOL_LET_STAR_OP);
            }
            if (strcmp(head_name, "letrec") == 0) {
                return convert_let(sexp, ESHKOL_LETREC_OP);
            }
            if (strcmp(head_name, "letrec*") == 0) {
                return convert_let(sexp, ESHKOL_LETREC_STAR_OP);
            }
            if (strcmp(head_name, "quote") == 0) {
                return convert_quote(sexp);
            }
            if (strcmp(head_name, "begin") == 0) {
                return convert_begin(sexp);
            }
            if (strcmp(head_name, "set!") == 0) {
                return convert_set(sexp);
            }
            if (strcmp(head_name, "and") == 0) {
                return convert_and(sexp);
            }
            if (strcmp(head_name, "or") == 0) {
                return convert_or(sexp);
            }
            if (strcmp(head_name, "cond") == 0) {
                return convert_cond(sexp);
            }
            if (strcmp(head_name, "case") == 0) {
                return convert_case(sexp);
            }
            if (strcmp(head_name, "when") == 0) {
                return convert_when(sexp);
            }
            if (strcmp(head_name, "unless") == 0) {
                return convert_unless(sexp);
            }
            if (strcmp(head_name, "quasiquote") == 0) {
                return convert_quasiquote(sexp);
            }
            if (strcmp(head_name, "unquote") == 0) {
                return convert_unquote(sexp);
            }
            if (strcmp(head_name, "unquote-splicing") == 0) {
                return convert_unquote_splicing(sexp);
            }
        }

        // Default: function call
        return convert_call(sexp);
    }

    // Character
    if (sexp.type == ESHKOL_VALUE_CHAR) {
        eshkol_ast_t* ast = eshkol_alloc_symbolic_ast();
        ast->type = ESHKOL_CHAR;
        ast->int64_val = sexp.data.int_val;
        return ast;
    }

    eshkol_error("sexp_to_ast: unhandled S-expression type: %d", sexp.type);
    return nullptr;
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

extern "C" {

eshkol_ast_t* eshkol_sexp_to_ast(eshkol_tagged_value_t sexp) {
    return convert_sexp(sexp);
}

void eshkol_free_sexp_ast(eshkol_ast_t* ast) {
    // Use the standard AST cleanup
    if (ast) {
        eshkol_ast_clean(ast);
        free(ast);
    }
}

bool eshkol_sexp_is_special_form(eshkol_tagged_value_t sexp) {
    if (!is_pair(sexp)) {
        return false;
    }

    eshkol_tagged_value_t head = pair_car(sexp);
    if (!is_symbol(head)) {
        return false;
    }

    const char* name = get_symbol_name(head);
    if (!name) {
        return false;
    }

    // Check against known special forms
    static const char* special_forms[] = {
        "lambda", "define", "if", "cond", "case", "match",
        "let", "let*", "letrec", "letrec*",
        "and", "or", "when", "unless",
        "begin", "quote", "quasiquote",
        "set!", "define-syntax",
        nullptr
    };

    for (const char** sf = special_forms; *sf; sf++) {
        if (strcmp(name, *sf) == 0) {
            return true;
        }
    }

    return false;
}

const char* eshkol_sexp_head_symbol(eshkol_tagged_value_t sexp) {
    if (!is_pair(sexp)) {
        return nullptr;
    }

    eshkol_tagged_value_t head = pair_car(sexp);
    return get_symbol_name(head);
}

size_t eshkol_sexp_list_length(eshkol_tagged_value_t sexp) {
    return list_length(sexp);
}

eshkol_tagged_value_t eshkol_sexp_list_ref(eshkol_tagged_value_t sexp, size_t index) {
    return list_ref(sexp, index);
}

// eshkol_compile_with_env is implemented in introspection.cpp
// because it requires access to the JIT infrastructure

} // extern "C"
