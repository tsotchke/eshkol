/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MacroExpander implementation - Hygienic macro expansion for syntax-rules
 */

#include <eshkol/frontend/macro_expander.h>
#include <eshkol/logger.h>
#include <cstring>
#include <algorithm>

namespace eshkol {

MacroExpander::MacroExpander() {
}

MacroExpander::~MacroExpander() {
    // Macro definitions are owned by the AST, not by us
}

void MacroExpander::registerMacro(const eshkol_macro_def_t* macro) {
    if (macro && macro->name) {
        macros_[macro->name] = const_cast<eshkol_macro_def_t*>(macro);
    }
}

bool MacroExpander::isMacro(const std::string& name) const {
    return macros_.find(name) != macros_.end();
}

std::vector<eshkol_ast_t> MacroExpander::expandAll(const std::vector<eshkol_ast_t>& asts) {
    std::vector<eshkol_ast_t> result;

    // First pass: collect all macro definitions
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
            if (ast.operation.define_syntax_op.macro) {
                registerMacro(ast.operation.define_syntax_op.macro);
            }
        }
    }

    // Second pass: expand macros (skip define-syntax forms)
    for (const auto& ast : asts) {
        if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
            // Macro definitions don't produce runtime code
            continue;
        }
        result.push_back(expand(ast));
    }

    return result;
}

eshkol_ast_t MacroExpander::expand(const eshkol_ast_t& ast) {
    return expandNode(ast);
}

eshkol_ast_t MacroExpander::expandNode(const eshkol_ast_t& ast) {
    // Handle define-syntax: register and return null
    if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
        if (ast.operation.define_syntax_op.macro) {
            registerMacro(ast.operation.define_syntax_op.macro);
        }
        eshkol_ast_t null_ast;
        null_ast.type = ESHKOL_NULL;
        return null_ast;
    }

    // Check for macro call
    if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_CALL_OP) {
        const auto* call = &ast.operation.call_op;
        if (call->func && call->func->type == ESHKOL_VAR && call->func->variable.id) {
            std::string func_name = call->func->variable.id;
            if (isMacro(func_name)) {
                eshkol_ast_t expanded = tryExpandMacroCall(ast);
                // Recursively expand the result (macros can expand to other macros)
                return expandNode(expanded);
            }
        }
    }

    // Recursively expand sub-expressions
    eshkol_ast_t result = copyAst(ast);

    if (result.type == ESHKOL_OP) {
        auto* op = &result.operation;

        switch (op->op) {
            case ESHKOL_CALL_OP:
                if (op->call_op.func) {
                    eshkol_ast_t* new_func = new eshkol_ast_t;
                    *new_func = expandNode(*op->call_op.func);
                    op->call_op.func = new_func;
                }
                if (op->call_op.num_vars > 0 && op->call_op.variables) {
                    eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                        new_vars[i] = expandNode(op->call_op.variables[i]);
                    }
                    op->call_op.variables = new_vars;
                }
                break;

            case ESHKOL_SEQUENCE_OP:
                if (op->sequence_op.num_expressions > 0 && op->sequence_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->sequence_op.num_expressions];
                    for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                        new_exprs[i] = expandNode(op->sequence_op.expressions[i]);
                    }
                    op->sequence_op.expressions = new_exprs;
                }
                break;

            case ESHKOL_DEFINE_OP:
                if (op->define_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = expandNode(*op->define_op.value);
                    op->define_op.value = new_val;
                }
                break;

            case ESHKOL_LAMBDA_OP:
                if (op->lambda_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = expandNode(*op->lambda_op.body);
                    op->lambda_op.body = new_body;
                }
                break;

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
                // let_op uses bindings array, not values
                if (op->let_op.num_bindings > 0 && op->let_op.bindings) {
                    eshkol_ast_t* new_bindings = new eshkol_ast_t[op->let_op.num_bindings];
                    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                        new_bindings[i] = expandNode(op->let_op.bindings[i]);
                    }
                    op->let_op.bindings = new_bindings;
                }
                if (op->let_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = expandNode(*op->let_op.body);
                    op->let_op.body = new_body;
                }
                break;

            case ESHKOL_MATCH_OP:
                if (op->match_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = expandNode(*op->match_op.expr);
                    op->match_op.expr = new_expr;
                }
                if (op->match_op.num_clauses > 0 && op->match_op.clauses) {
                    for (uint64_t i = 0; i < op->match_op.num_clauses; i++) {
                        if (op->match_op.clauses[i].body) {
                            eshkol_ast_t* new_body = new eshkol_ast_t;
                            *new_body = expandNode(*op->match_op.clauses[i].body);
                            op->match_op.clauses[i].body = new_body;
                        }
                    }
                }
                break;

            default:
                // Other operations - leave as is
                break;
        }
    }

    return result;
}

eshkol_ast_t MacroExpander::tryExpandMacroCall(const eshkol_ast_t& call) {
    const auto* call_op = &call.operation.call_op;
    std::string macro_name = call_op->func->variable.id;

    auto it = macros_.find(macro_name);
    if (it == macros_.end()) {
        return call;
    }

    eshkol_macro_def_t* macro = it->second;

    // Build literals list for pattern matching
    std::vector<std::string> literals;
    for (uint64_t i = 0; i < macro->num_literals; i++) {
        if (macro->literals[i]) {
            literals.push_back(macro->literals[i]);
        }
    }

    // Try each rule in order
    for (uint64_t rule_idx = 0; rule_idx < macro->num_rules; rule_idx++) {
        eshkol_macro_rule_t* rule = &macro->rules[rule_idx];
        if (!rule->pattern || !rule->template_) continue;

        Bindings bindings;

        // Pattern is (macro-name arg1 arg2 ...)
        // Skip the first element (macro name), match remaining against arguments
        if (rule->pattern->type != MACRO_PAT_LIST) continue;

        auto* pat_list = &rule->pattern->list;

        // Pattern elements after macro name
        uint64_t pat_start = 1;  // Skip macro name

        // Try to match arguments against pattern elements
        bool matched = true;
        uint64_t arg_idx = 0;

        for (uint64_t pat_idx = pat_start; pat_idx < pat_list->num_elements && matched; pat_idx++) {
            eshkol_macro_pattern_t* elem = pat_list->elements[pat_idx];
            if (!elem) {
                matched = false;
                break;
            }

            if (elem->followed_by_ellipsis) {
                // Collect remaining arguments for this pattern variable
                std::string var_name = elem->identifier ? elem->identifier : "";
                Binding binding;
                binding.name = var_name;

                // Collect all remaining arguments
                while (arg_idx < call_op->num_vars) {
                    binding.values.push_back(call_op->variables[arg_idx]);
                    arg_idx++;
                }
                bindings[var_name] = binding;
            } else if (elem->type == MACRO_PAT_VARIABLE) {
                // Single variable - must have an argument
                if (arg_idx >= call_op->num_vars) {
                    matched = false;
                    break;
                }
                std::string var_name = elem->identifier ? elem->identifier : "";
                Binding binding;
                binding.name = var_name;
                binding.values.push_back(call_op->variables[arg_idx]);
                bindings[var_name] = binding;
                arg_idx++;
            } else if (elem->type == MACRO_PAT_LITERAL) {
                // Literal must match exactly
                if (arg_idx >= call_op->num_vars) {
                    matched = false;
                    break;
                }
                std::string expected = elem->identifier ? elem->identifier : "";
                std::string actual = getSymbolName(call_op->variables[arg_idx]);
                if (expected != actual) {
                    matched = false;
                    break;
                }
                arg_idx++;
            } else if (elem->type == MACRO_PAT_LIST) {
                // Nested list pattern
                if (arg_idx >= call_op->num_vars) {
                    matched = false;
                    break;
                }
                arg_idx++;
            }
        }

        // Check if all arguments were consumed
        bool has_ellipsis = false;
        if (pat_list->num_elements > pat_start) {
            eshkol_macro_pattern_t* last = pat_list->elements[pat_list->num_elements - 1];
            if (last && last->followed_by_ellipsis) {
                has_ellipsis = true;
            }
        }

        if (matched && (arg_idx == call_op->num_vars || has_ellipsis)) {
            // Rule matched! Instantiate the template
            return instantiateTemplate(rule->template_, bindings);
        }
    }

    // No rule matched
    eshkol_warn("Macro '%s' call did not match any pattern", macro_name.c_str());
    return call;
}

eshkol_ast_t MacroExpander::instantiateTemplate(const eshkol_macro_template_t* tmpl,
                                                  const Bindings& bindings) {
    if (!tmpl) {
        eshkol_ast_t null_ast;
        null_ast.type = ESHKOL_NULL;
        return null_ast;
    }

    switch (tmpl->type) {
        case MACRO_TPL_LITERAL:
            // The template is stored as a literal AST
            if (tmpl->literal) {
                return substituteBindings(*tmpl->literal, bindings);
            }
            break;

        case MACRO_TPL_VARIABLE: {
            // Look up variable in bindings
            if (!tmpl->variable_name) {
                eshkol_ast_t null_ast;
                null_ast.type = ESHKOL_NULL;
                return null_ast;
            }
            auto it = bindings.find(tmpl->variable_name);
            if (it != bindings.end() && !it->second.values.empty()) {
                return copyAst(it->second.values[0]);
            }
            // Variable not found - return as symbol reference
            eshkol_ast_t var_ast;
            var_ast.type = ESHKOL_VAR;
            var_ast.variable.id = strdup(tmpl->variable_name);
            return var_ast;
        }

        case MACRO_TPL_LIST: {
            // Build expression from list elements
            if (tmpl->list.num_elements == 0) {
                eshkol_ast_t null_ast;
                null_ast.type = ESHKOL_NULL;
                return null_ast;
            }

            std::vector<eshkol_ast_t> expanded;
            for (uint64_t i = 0; i < tmpl->list.num_elements; i++) {
                eshkol_ast_t elem = instantiateTemplate(tmpl->list.elements[i], bindings);
                expanded.push_back(elem);
            }

            // Build call expression
            if (expanded.size() > 0) {
                eshkol_ast_t result;
                result.type = ESHKOL_OP;
                result.operation.op = ESHKOL_CALL_OP;

                result.operation.call_op.func = new eshkol_ast_t;
                *result.operation.call_op.func = expanded[0];

                result.operation.call_op.num_vars = expanded.size() - 1;
                if (result.operation.call_op.num_vars > 0) {
                    result.operation.call_op.variables = new eshkol_ast_t[result.operation.call_op.num_vars];
                    for (size_t i = 1; i < expanded.size(); i++) {
                        result.operation.call_op.variables[i - 1] = expanded[i];
                    }
                } else {
                    result.operation.call_op.variables = nullptr;
                }

                return result;
            }
            break;
        }

        default:
            break;
    }

    eshkol_ast_t null_ast;
    null_ast.type = ESHKOL_NULL;
    return null_ast;
}

eshkol_ast_t MacroExpander::substituteBindings(const eshkol_ast_t& ast,
                                                 const Bindings& bindings) {
    // Check if this is a variable that should be substituted
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        std::string name = ast.variable.id;
        auto it = bindings.find(name);
        if (it != bindings.end() && !it->second.values.empty()) {
            return copyAst(it->second.values[0]);
        }
        return copyAst(ast);
    }

    // For operations, recursively substitute
    if (ast.type == ESHKOL_OP) {
        eshkol_ast_t result;
        result.type = ESHKOL_OP;
        result.operation = ast.operation;
        auto* op = &result.operation;

        switch (op->op) {
            case ESHKOL_CALL_OP:
                if (op->call_op.func) {
                    eshkol_ast_t* new_func = new eshkol_ast_t;
                    *new_func = substituteBindings(*op->call_op.func, bindings);
                    op->call_op.func = new_func;
                }
                if (op->call_op.num_vars > 0 && op->call_op.variables) {
                    eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                        new_vars[i] = substituteBindings(op->call_op.variables[i], bindings);
                    }
                    op->call_op.variables = new_vars;
                }
                break;

            case ESHKOL_SEQUENCE_OP:
                if (op->sequence_op.num_expressions > 0 && op->sequence_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->sequence_op.num_expressions];
                    for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                        new_exprs[i] = substituteBindings(op->sequence_op.expressions[i], bindings);
                    }
                    op->sequence_op.expressions = new_exprs;
                }
                break;

            case ESHKOL_DEFINE_OP:
                if (op->define_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = substituteBindings(*op->define_op.value, bindings);
                    op->define_op.value = new_val;
                }
                break;

            case ESHKOL_LAMBDA_OP:
                if (op->lambda_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = substituteBindings(*op->lambda_op.body, bindings);
                    op->lambda_op.body = new_body;
                }
                break;

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
                if (op->let_op.num_bindings > 0 && op->let_op.bindings) {
                    eshkol_ast_t* new_bindings = new eshkol_ast_t[op->let_op.num_bindings];
                    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                        new_bindings[i] = substituteBindings(op->let_op.bindings[i], bindings);
                    }
                    op->let_op.bindings = new_bindings;
                }
                if (op->let_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = substituteBindings(*op->let_op.body, bindings);
                    op->let_op.body = new_body;
                }
                break;

            default:
                break;
        }

        return result;
    }

    // For other AST types, just copy
    return copyAst(ast);
}

eshkol_ast_t MacroExpander::copyAst(const eshkol_ast_t& ast) {
    eshkol_ast_t result = ast;

    // Deep copy strings and nested pointers
    switch (ast.type) {
        case ESHKOL_STRING:
            if (ast.str_val.ptr) {
                result.str_val.ptr = strdup(ast.str_val.ptr);
                result.str_val.size = ast.str_val.size;
            }
            break;

        case ESHKOL_VAR:
            if (ast.variable.id) {
                result.variable.id = strdup(ast.variable.id);
            }
            break;

        case ESHKOL_OP:
            // Operations need deep copy of nested pointers
            // This is handled by the caller for specific operation types
            break;

        default:
            // Primitive types - shallow copy is sufficient
            break;
    }

    return result;
}

bool MacroExpander::isSymbol(const eshkol_ast_t& ast, const std::string& name) const {
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        return std::string(ast.variable.id) == name;
    }
    return false;
}

std::string MacroExpander::getSymbolName(const eshkol_ast_t& ast) const {
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        return std::string(ast.variable.id);
    }
    return "";
}

bool MacroExpander::isLiteral(const std::string& name,
                               const std::vector<std::string>& literals) const {
    return std::find(literals.begin(), literals.end(), name) != literals.end();
}

bool MacroExpander::matchPattern(const eshkol_macro_pattern_t* pattern,
                                  const eshkol_ast_t& ast,
                                  const std::vector<std::string>& literals,
                                  Bindings& bindings) {
    if (!pattern) return false;

    switch (pattern->type) {
        case MACRO_PAT_VARIABLE: {
            // Pattern variables match anything and bind the value
            std::string var_name = pattern->identifier ? pattern->identifier : "";
            Binding binding;
            binding.name = var_name;
            binding.values.push_back(ast);
            bindings[var_name] = binding;
            return true;
        }

        case MACRO_PAT_LITERAL: {
            // Literals must match exactly
            std::string lit_name = pattern->identifier ? pattern->identifier : "";
            return isSymbol(ast, lit_name);
        }

        case MACRO_PAT_LIST: {
            // List patterns match list-like ASTs (calls, sequences)
            if (ast.type != ESHKOL_OP) return false;

            const auto* op = &ast.operation;
            if (op->op != ESHKOL_CALL_OP) return false;

            // Match elements
            uint64_t ast_count = op->call_op.num_vars + 1;  // +1 for func
            uint64_t pat_count = pattern->list.num_elements;

            // Check for ellipsis
            bool has_ellipsis = false;
            for (uint64_t i = 0; i < pat_count; i++) {
                if (pattern->list.elements[i] && pattern->list.elements[i]->followed_by_ellipsis) {
                    has_ellipsis = true;
                    break;
                }
            }

            if (!has_ellipsis && ast_count != pat_count) return false;

            // Match each element
            for (uint64_t i = 0; i < pat_count && i < ast_count; i++) {
                eshkol_ast_t elem_ast;
                if (i == 0 && op->call_op.func) {
                    elem_ast = *op->call_op.func;
                } else if (i > 0 && op->call_op.variables) {
                    elem_ast = op->call_op.variables[i - 1];
                } else {
                    return false;
                }

                if (!matchPattern(pattern->list.elements[i], elem_ast, literals, bindings)) {
                    return false;
                }
            }

            return true;
        }

        default:
            return false;
    }
}

} // namespace eshkol
