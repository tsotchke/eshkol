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
    
    // Generate code
    fprintf(output, "%s", node->as.identifier.name);
    
    return true;
}
