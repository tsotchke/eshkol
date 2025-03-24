/**
 * @file type_conversion.h
 * @brief Type conversion for code generation
 */

#ifndef ESHKOL_CODEGEN_TYPE_CONVERSION_H
#define ESHKOL_CODEGEN_TYPE_CONVERSION_H

#include "backend/codegen/context.h"
#include "core/type.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert an Eshkol type to a C type string
 * 
 * @param type Eshkol type
 * @return C type string
 */
const char* codegen_type_to_c_type(Type* type);

/**
 * @brief Get the C type string for an AST node
 * 
 * @param context The code generator context
 * @param node AST node
 * @return C type string, or NULL if not typed
 */
const char* codegen_get_c_type(CodegenContext* context, const AstNode* node);

/**
 * @brief Check if a type can be implicitly converted to another type
 * 
 * @param from Source type
 * @param to Target type
 * @return true if conversion is possible, false otherwise
 */
bool codegen_type_can_convert(Type* from, Type* to);

/**
 * @brief Generate code for type conversion
 * 
 * @param context The code generator context
 * @param from Source type
 * @param to Target type
 * @return Conversion code string
 */
const char* codegen_type_conversion_code(CodegenContext* context, Type* from, Type* to);

/**
 * @brief Apply type conversion to an expression
 * 
 * @param context The code generator context
 * @param expr Expression string
 * @param from Source type
 * @param to Target type
 * @return Converted expression string
 */
char* codegen_type_apply_conversion(CodegenContext* context, const char* expr, Type* from, Type* to);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_TYPE_CONVERSION_H */
