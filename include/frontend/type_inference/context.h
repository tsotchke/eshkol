/**
 * @file context.h
 * @brief Type inference context management
 */

#ifndef ESHKOL_TYPE_INFERENCE_CONTEXT_H
#define ESHKOL_TYPE_INFERENCE_CONTEXT_H

#include "core/memory.h"
#include "core/type.h"
#include "core/type_creation.h"
#include "core/type_comparison.h"
#include "core/type_conversion.h"
#include "frontend/ast/ast.h"
#include "core/diagnostics.h"
#include <stdbool.h>

/**
 * @brief Type inference context
 */
typedef struct TypeInferenceContext TypeInferenceContext;

/**
 * @brief Create a type inference context
 * 
 * @param arena Arena for allocations
 * @param diagnostics Diagnostic context for error reporting
 * @return Type inference context
 */
TypeInferenceContext* type_inference_context_create(Arena* arena, DiagnosticContext* diagnostics);

/**
 * @brief Add a node to the context with its inferred type
 * 
 * @param context Type inference context
 * @param node AST node
 * @param type Inferred type
 * @return true if successful, false otherwise
 */
bool type_inference_add_node(TypeInferenceContext* context, AstNode* node, Type* type);

/**
 * @brief Set an explicit type for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @param type Explicit type
 */
void type_inference_set_explicit_type(TypeInferenceContext* context, AstNode* node, Type* type);

/**
 * @brief Set a function signature
 * 
 * @param context Type inference context
 * @param function_name Function name
 * @param type Function type
 */
void type_inference_set_function_signature(TypeInferenceContext* context, StringId function_name, Type* type);

/**
 * @brief Get the inferred type for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return Inferred type, or NULL if not inferred
 */
Type* type_inference_get_type(TypeInferenceContext* context, const AstNode* node);

/**
 * @brief Get the explicit type for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return Explicit type, or NULL if not explicitly typed
 */
Type* type_inference_get_explicit_type(TypeInferenceContext* context, const AstNode* node);

/**
 * @brief Get the function signature for a function name
 * 
 * @param context Type inference context
 * @param function_name Function name
 * @return Function type, or NULL if not found
 */
Type* type_inference_get_function_signature(TypeInferenceContext* context, StringId function_name);

/**
 * @brief Get the arena from the context
 * 
 * @param context Type inference context
 * @return Arena
 */
Arena* type_inference_get_arena(TypeInferenceContext* context);

/**
 * @brief Get the diagnostics from the context
 * 
 * @param context Type inference context
 * @return Diagnostic context
 */
DiagnosticContext* type_inference_get_diagnostics(TypeInferenceContext* context);

#endif /* ESHKOL_TYPE_INFERENCE_CONTEXT_H */
