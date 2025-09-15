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
