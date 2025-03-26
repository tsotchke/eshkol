#ifndef ESHKOL_TYPE_INFERENCE_ANALYSIS_H
#define ESHKOL_TYPE_INFERENCE_ANALYSIS_H

#include "frontend/type_inference/context.h"
#include "frontend/ast/ast.h"

/**
 * @brief Analyze type inference results
 * 
 * @param context Type inference context
 * @param ast AST to analyze
 * @param detail_level Detail level (basic, detailed, verbose)
 * @return true on success, false on failure
 */
bool type_inference_analyze(TypeInferenceContext* context, AstNode* ast, const char* detail_level);

#endif /* ESHKOL_TYPE_INFERENCE_ANALYSIS_H */
