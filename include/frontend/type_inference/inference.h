/**
 * @file inference.h
 * @brief Type inference for AST nodes
 */

#ifndef ESHKOL_TYPE_INFERENCE_INFERENCE_H
#define ESHKOL_TYPE_INFERENCE_INFERENCE_H

#include "frontend/type_inference/context.h"
#include <stdbool.h>

/**
 * @brief Collect explicit types from the AST
 * 
 * This function collects explicit types from type declarations and inline type annotations.
 * 
 * @param context Type inference context
 * @param ast AST to collect types from
 * @return true if successful, false otherwise
 */
bool type_inference_collect_explicit_types(TypeInferenceContext* context, AstNode* ast);

/**
 * @brief Infer types for an AST
 * 
 * This function performs type inference on the AST, using explicit types where available.
 * 
 * @param context Type inference context
 * @param ast AST to infer types for
 * @return true if successful, false otherwise
 */
bool type_inference_infer(TypeInferenceContext* context, AstNode* ast);

/**
 * @brief Infer the type of a node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return Inferred type, or NULL if inference failed
 */
Type* type_inference_infer_node(TypeInferenceContext* context, AstNode* node);

/**
 * @brief Resolve the final type for an AST node
 * 
 * This function resolves the final type for an AST node, prioritizing explicit types over inferred types.
 * 
 * @param context Type inference context
 * @param node AST node
 * @return Resolved type, or NULL if not typed
 */
Type* type_inference_resolve_type(TypeInferenceContext* context, const AstNode* node);

#endif /* ESHKOL_TYPE_INFERENCE_INFERENCE_H */
