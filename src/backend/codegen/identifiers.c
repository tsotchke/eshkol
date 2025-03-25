/**
 * @file identifiers.c
 * @brief Identifier code generation implementation
 */

#include "backend/codegen/identifiers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/**
 * @brief Generate C code for an identifier
 */
bool codegen_generate_identifier(CodegenContext* context, const AstNode* node) {
    assert(context != NULL);
    assert(node != NULL);
    assert(node->type == AST_IDENTIFIER);
    
    // Get output file
    FILE* output = codegen_context_get_output(context);
    
    // Replace hyphens with underscores in identifiers
    char* identifier_name = strdup(node->as.identifier.name);
    if (identifier_name) {
        for (char* p = identifier_name; *p; p++) {
            if (*p == '-') {
                *p = '_';
            }
        }
        fprintf(output, "%s", identifier_name);
        free(identifier_name);
    } else {
        // Fallback if memory allocation fails
        fprintf(output, "%s", node->as.identifier.name);
    }
    
    return true;
}
