/**
 * @file type_conversion.h
 * @brief Type conversion utilities for code generation
 */

#ifndef ESHKOL_CODEGEN_TYPE_CONVERSION_H
#define ESHKOL_CODEGEN_TYPE_CONVERSION_H

#include "frontend/type_inference/type_inference.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Convert an Eshkol type to a C type string
 * 
 * @param type The Eshkol type to convert
 * @return The corresponding C type string
 */
const char* codegen_type_to_c_type(Type* type);

/**
 * @brief Check if a type is a numeric type
 * 
 * @param type The type to check
 * @return true if the type is numeric, false otherwise
 */
bool codegen_type_is_numeric(Type* type);

/**
 * @brief Check if a type is a floating-point type
 * 
 * @param type The type to check
 * @return true if the type is floating-point, false otherwise
 */
bool codegen_type_is_float(Type* type);

/**
 * @brief Check if a type is an integer type
 * 
 * @param type The type to check
 * @return true if the type is integer, false otherwise
 */
bool codegen_type_is_integer(Type* type);

/**
 * @brief Check if a type is a boolean type
 * 
 * @param type The type to check
 * @return true if the type is boolean, false otherwise
 */
bool codegen_type_is_boolean(Type* type);

/**
 * @brief Check if a type is a string type
 * 
 * @param type The type to check
 * @return true if the type is string, false otherwise
 */
bool codegen_type_is_string(Type* type);

/**
 * @brief Check if a type is a vector type
 * 
 * @param type The type to check
 * @return true if the type is vector, false otherwise
 */
bool codegen_type_is_vector(Type* type);

/**
 * @brief Check if a type is a function type
 * 
 * @param type The type to check
 * @return true if the type is function, false otherwise
 */
bool codegen_type_is_function(Type* type);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_TYPE_CONVERSION_H */
