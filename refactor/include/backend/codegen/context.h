/**
 * @file context.h
 * @brief Code generation context management
 */

#ifndef ESHKOL_CODEGEN_CONTEXT_H
#define ESHKOL_CODEGEN_CONTEXT_H

#include "core/memory.h"
#include "core/diagnostics.h"
#include "frontend/type_inference/type_inference.h"
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Code generator context
 * 
 * Manages the state for code generation.
 */
typedef struct CodegenContext {
    Arena* arena;                // Arena for allocations
    DiagnosticContext* diagnostics; // Diagnostic context for error reporting
    TypeInferenceContext* type_context; // Type inference context for type information
    FILE* output;                // Output file
    int indent_level;            // Current indentation level
    bool in_function;            // Whether we're currently in a function
    char* temp_dir;              // Temporary directory for compilation
} CodegenContext;

/**
 * @brief Create a code generator context
 * 
 * @param arena Arena for allocations
 * @param diagnostics Diagnostic context for error reporting
 * @param type_context Type inference context for type information
 * @return A new code generator context, or NULL on failure
 */
CodegenContext* codegen_context_create(Arena* arena, DiagnosticContext* diagnostics, TypeInferenceContext* type_context);

/**
 * @brief Initialize a code generator context
 * 
 * @param context The code generator context to initialize
 * @param arena Arena for allocations
 * @param type_context Type inference context for type information
 * @param output_file The output file path (can be NULL for stdout)
 * @return true if initialization succeeded, false otherwise
 */
bool codegen_context_init(CodegenContext* context, Arena* arena, TypeInferenceContext* type_context, const char* output_file);

/**
 * @brief Get the arena from the context
 * 
 * @param context The code generator context
 * @return The arena
 */
Arena* codegen_context_get_arena(CodegenContext* context);

/**
 * @brief Get the diagnostics from the context
 * 
 * @param context The code generator context
 * @return The diagnostic context
 */
DiagnosticContext* codegen_context_get_diagnostics(CodegenContext* context);

/**
 * @brief Get the type inference context from the context
 * 
 * @param context The code generator context
 * @return The type inference context
 */
TypeInferenceContext* codegen_context_get_type_context(CodegenContext* context);

/**
 * @brief Get the output file from the context
 * 
 * @param context The code generator context
 * @return The output file
 */
FILE* codegen_context_get_output(CodegenContext* context);

/**
 * @brief Set the output file for the context
 * 
 * @param context The code generator context
 * @param output The output file
 */
void codegen_context_set_output(CodegenContext* context, FILE* output);

/**
 * @brief Get the indent level from the context
 * 
 * @param context The code generator context
 * @return The indent level
 */
int codegen_context_get_indent_level(CodegenContext* context);

/**
 * @brief Set the indent level for the context
 * 
 * @param context The code generator context
 * @param indent_level The indent level
 */
void codegen_context_set_indent_level(CodegenContext* context, int indent_level);

/**
 * @brief Increment the indent level
 * 
 * @param context The code generator context
 */
void codegen_context_increment_indent(CodegenContext* context);

/**
 * @brief Decrement the indent level
 * 
 * @param context The code generator context
 */
void codegen_context_decrement_indent(CodegenContext* context);

/**
 * @brief Check if we're currently in a function
 * 
 * @param context The code generator context
 * @return true if in a function, false otherwise
 */
bool codegen_context_in_function(CodegenContext* context);

/**
 * @brief Set whether we're in a function
 * 
 * @param context The code generator context
 * @param in_function Whether we're in a function
 */
void codegen_context_set_in_function(CodegenContext* context, bool in_function);

/**
 * @brief Write indentation to the output
 * 
 * @param context The code generator context
 */
void codegen_context_write_indent(CodegenContext* context);

#ifdef __cplusplus
}
#endif

#endif /* ESHKOL_CODEGEN_CONTEXT_H */
