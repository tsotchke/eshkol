/**
 * @file type_inference.h
 * @brief Type inference system for Eshkol
 */

#ifndef ESHKOL_TYPE_INFERENCE_H
#define ESHKOL_TYPE_INFERENCE_H

#include "core/memory.h"
#include "core/type.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

/**
 * @brief Type inference context
 */
typedef struct TypeInferenceContext TypeInferenceContext;

/**
 * @brief Create a type inference context
 * 
 * @param arena Arena for allocations
 * @return Type inference context
 */
TypeInferenceContext* type_inference_context_create(Arena* arena);

/**
 * @brief Infer types for an AST
 * 
 * @param context Type inference context
 * @param ast AST to infer types for
 * @return true if successful, false otherwise
 */
bool type_inference_infer(TypeInferenceContext* context, AstNode* ast);

/**
 * @brief Get the inferred type for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return Inferred type, or NULL if not inferred
 */
Type* type_inference_get_type(TypeInferenceContext* context, const AstNode* node);

/**
 * @brief Get the C type string for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return C type string, or NULL if not inferred
 */
const char* type_inference_get_c_type(TypeInferenceContext* context, const AstNode* node);

#endif /* ESHKOL_TYPE_INFERENCE_H */
