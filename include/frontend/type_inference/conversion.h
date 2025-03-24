/**
 * @file conversion.h
 * @brief Type conversion utilities
 */

#ifndef ESHKOL_TYPE_INFERENCE_CONVERSION_H
#define ESHKOL_TYPE_INFERENCE_CONVERSION_H

#include "core/type.h"
#include "frontend/type_inference/context.h"

/**
 * @brief Convert an Eshkol type to a C type string
 * 
 * @param type Eshkol type
 * @return C type string
 */
const char* type_to_c_type(Type* type);

/**
 * @brief Get the C type string for an AST node
 * 
 * @param context Type inference context
 * @param node AST node
 * @return C type string, or NULL if not typed
 */
const char* type_inference_get_c_type(TypeInferenceContext* context, const AstNode* node);

/**
 * @brief Check if a type can be implicitly converted to another type
 * 
 * @param from Source type
 * @param to Target type
 * @return true if conversion is possible, false otherwise
 */
bool type_can_convert(Type* from, Type* to);

/**
 * @brief Generate code for type conversion
 * 
 * @param arena Memory arena
 * @param from Source type
 * @param to Target type
 * @return Conversion code string
 */
const char* type_conversion_code(Arena* arena, Type* from, Type* to);

/**
 * @brief Apply type conversion to an expression
 * 
 * @param arena Memory arena
 * @param expr Expression string
 * @param from Source type
 * @param to Target type
 * @return Converted expression string
 */
char* type_apply_conversion(Arena* arena, const char* expr, Type* from, Type* to);

#endif /* ESHKOL_TYPE_INFERENCE_CONVERSION_H */
