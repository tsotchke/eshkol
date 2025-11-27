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
