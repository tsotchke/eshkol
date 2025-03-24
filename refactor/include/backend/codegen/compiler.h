/**
 * @file compiler.h
 * @brief Compiler interface for code generation
 */

#ifndef ESHKOL_CODEGEN_COMPILER_H
#define ESHKOL_CODEGEN_COMPILER_H

#include "backend/codegen/context.h"
#include "frontend/ast/ast.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compile and execute the generated code
 * 
 * @param context The code generator context
 * @param c_file The C file to compile
 * @param args Arguments to pass to the compiled program
 * @param argc Number of arguments
 * @return The exit code of the compiled program, or -1 on failure
 */
int codegen_compile_and_execute(CodegenContext* context, const char* c_file, char** args, int argc);

/**
 * @brief Compile the generated code
 * 
 * @param context The code generator context
 * @param c_file The C file to compile
 * @param output_file The output file path
 * @return true if compilation succeeded, false otherwise
 */
bool codegen_compile(CodegenContext* context, const char* c_file, const char* output_file);

/**
 * @brief Execute the compiled code
 * 
 * @param context The code generator context
 * @param executable The executable file path
 * @param args Arguments to pass to the compiled program
 * @param argc Number of arguments
 * @return The exit code of the compiled program, or -1 on failure
 */
int codegen_execute(CodegenContext* context, const char* executable, char** args, int argc);

/**
 * @brief Get the include path for compilation
 * 
 * @param context The code generator context
 * @param c_file The C file to compile
 * @param include_path Buffer to store the include path
 * @param include_path_size Size of the include path buffer
 * @return true if the include path was successfully determined, false otherwise
 */
bool codegen_get_include_path(CodegenContext* context, const char* c_file, char* include_path, size_t include_path_size);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_COMPILER_H */
