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
    // Initialize with a global scope
    scope_stack_.emplace_back();
}

MacroExpander::~MacroExpander() {
    // Macro definitions are owned by the AST, not by us
}

void MacroExpander::pushScope() {
    scope_stack_.emplace_back();
}

void MacroExpander::popScope() {
    if (scope_stack_.size() > 1) {
        scope_stack_.pop_back();
    }
}

void MacroExpander::registerMacro(const eshkol_macro_def_t* macro) {
    if (macro && macro->name && !scope_stack_.empty()) {
        scope_stack_.back()[macro->name] = const_cast<eshkol_macro_def_t*>(macro);
    }
}

eshkol_macro_def_t* MacroExpander::lookupMacro(const std::string& name) const {
    // Search from innermost scope to outermost
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    return nullptr;
}

bool MacroExpander::isMacro(const std::string& name) const {
    return lookupMacro(name) != nullptr;
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
    // Use iterative re-expansion for macro calls to prevent unbounded recursion.
    // A macro expanding to another macro call is handled by looping, not recursing.
    // We track seen macro names per expansion chain to detect cycles.
    static thread_local int expansion_depth = 0;
    struct DepthGuard { DepthGuard() { ++expansion_depth; } ~DepthGuard() { --expansion_depth; } } depth_guard;
    if (expansion_depth > 1000) {
        eshkol_error("macro expansion depth limit exceeded (>1000)");
        return ast;
    }
    eshkol_ast_t current = ast;
    std::set<std::string> expansion_chain; // macros seen in this chain

    // Iterative macro re-expansion loop
    for (;;) {
        // Handle define-syntax: register and return null
        if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_DEFINE_SYNTAX_OP) {
            if (current.operation.define_syntax_op.macro) {
                registerMacro(current.operation.define_syntax_op.macro);
            }
            eshkol_ast_t null_ast;
            eshkol_ast_make_null(&null_ast);
            return null_ast;
        }

        // Handle let-syntax / letrec-syntax: push scope, register macros, expand body, pop scope
        if (current.type == ESHKOL_OP &&
            (current.operation.op == ESHKOL_LET_SYNTAX_OP || current.operation.op == ESHKOL_LETREC_SYNTAX_OP)) {
            const auto* ls = &current.operation.let_syntax_op;
            pushScope();
            for (uint64_t i = 0; i < ls->num_macros; i++) {
                if (ls->macros[i]) {
                    registerMacro(ls->macros[i]);
                }
            }
            eshkol_ast_t expanded_body = expandNode(*ls->body);
            popScope();
            return expanded_body;
        }

        // Check for macro call — if found, expand and LOOP (not recurse)
        if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_CALL_OP) {
            const auto* call = &current.operation.call_op;
            if (call->func && call->func->type == ESHKOL_VAR && call->func->variable.id) {
                std::string func_name = call->func->variable.id;
                if (isMacro(func_name)) {
                    // Cycle detection: if we've seen this macro in this chain, it's circular
                    if (expansion_chain.count(func_name)) {
                        eshkol_error("Circular macro expansion detected: '%s' expands back to itself", func_name.c_str());
                        return current;
                    }
                    expansion_chain.insert(func_name);
                    current = tryExpandMacroCall(current);
                    continue; // Re-expand iteratively
                }
            }
        }

        // Not a macro call — break out to do tree traversal
        break;
    }

    // Recursively expand sub-expressions (tree depth is bounded by input nesting)
    eshkol_ast_t result = copyAst(current);

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
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
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
                if (op->let_op.num_bindings > 0 && op->let_op.bindings) {
                    eshkol_ast_t* new_bindings = new eshkol_ast_t[op->let_op.num_bindings];
                    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                        // Bindings are CONS cells (var . val) — expand the value (cdr)
                        eshkol_ast_t binding = op->let_op.bindings[i];
                        if (binding.type == ESHKOL_CONS && binding.cons_cell.cdr) {
                            eshkol_ast_t* new_cdr = new eshkol_ast_t;
                            *new_cdr = expandNode(*binding.cons_cell.cdr);
                            binding.cons_cell.cdr = new_cdr;
                        }
                        new_bindings[i] = binding;
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

            // Ops that reuse call_op struct layout
            case ESHKOL_COND_OP:
            case ESHKOL_CASE_OP:
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
            case ESHKOL_DO_OP:
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

            case ESHKOL_SET_OP:
                if (op->set_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = expandNode(*op->set_op.value);
                    op->set_op.value = new_val;
                }
                break;

            case ESHKOL_GUARD_OP:
                if (op->guard_op.num_clauses > 0 && op->guard_op.clauses) {
                    eshkol_ast_t* new_clauses = new eshkol_ast_t[op->guard_op.num_clauses];
                    for (uint64_t i = 0; i < op->guard_op.num_clauses; i++) {
                        new_clauses[i] = expandNode(op->guard_op.clauses[i]);
                    }
                    op->guard_op.clauses = new_clauses;
                }
                if (op->guard_op.num_body_exprs > 0 && op->guard_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t[op->guard_op.num_body_exprs];
                    for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                        new_body[i] = expandNode(op->guard_op.body[i]);
                    }
                    op->guard_op.body = new_body;
                }
                break;

            case ESHKOL_RAISE_OP:
                if (op->raise_op.exception) {
                    eshkol_ast_t* new_exc = new eshkol_ast_t;
                    *new_exc = expandNode(*op->raise_op.exception);
                    op->raise_op.exception = new_exc;
                }
                break;

            case ESHKOL_VALUES_OP:
                if (op->values_op.num_values > 0 && op->values_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->values_op.num_values];
                    for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                        new_exprs[i] = expandNode(op->values_op.expressions[i]);
                    }
                    op->values_op.expressions = new_exprs;
                }
                break;

            case ESHKOL_CALL_CC_OP:
                if (op->call_cc_op.proc) {
                    eshkol_ast_t* new_proc = new eshkol_ast_t;
                    *new_proc = expandNode(*op->call_cc_op.proc);
                    op->call_cc_op.proc = new_proc;
                }
                break;

            case ESHKOL_DYNAMIC_WIND_OP:
                if (op->dynamic_wind_op.before) {
                    eshkol_ast_t* new_before = new eshkol_ast_t;
                    *new_before = expandNode(*op->dynamic_wind_op.before);
                    op->dynamic_wind_op.before = new_before;
                }
                if (op->dynamic_wind_op.thunk) {
                    eshkol_ast_t* new_thunk = new eshkol_ast_t;
                    *new_thunk = expandNode(*op->dynamic_wind_op.thunk);
                    op->dynamic_wind_op.thunk = new_thunk;
                }
                if (op->dynamic_wind_op.after) {
                    eshkol_ast_t* new_after = new eshkol_ast_t;
                    *new_after = expandNode(*op->dynamic_wind_op.after);
                    op->dynamic_wind_op.after = new_after;
                }
                break;

            default:
                break;
        }
    }

    return result;
}

eshkol_ast_t MacroExpander::tryExpandMacroCall(const eshkol_ast_t& call) {
    const auto* call_op = &call.operation.call_op;
    std::string macro_name = call_op->func->variable.id;

    eshkol_macro_def_t* macro = lookupMacro(macro_name);
    if (!macro) {
        return call;
    }

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
                // Nested list pattern - delegate to recursive matchPattern
                if (arg_idx >= call_op->num_vars) {
                    matched = false;
                    break;
                }
                if (!matchPattern(elem, call_op->variables[arg_idx], literals, bindings)) {
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

    // No rule matched — this is a syntax error, not a warning
    eshkol_error("syntax error: no matching pattern for macro '%s'", macro_name.c_str());
    return call;
}

eshkol_ast_t MacroExpander::instantiateTemplate(const eshkol_macro_template_t* tmpl,
                                                  const Bindings& bindings) {
    if (!tmpl) {
        eshkol_ast_t null_ast;
        eshkol_ast_make_null(&null_ast);
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
                eshkol_ast_make_null(&null_ast);
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
                eshkol_ast_make_null(&null_ast);
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
    eshkol_ast_make_null(&null_ast);
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

    // Recursively substitute in cons cells (used for let bindings: (var . val))
    if (ast.type == ESHKOL_CONS) {
        eshkol_ast_t result;
        result.type = ESHKOL_CONS;
        result.cons_cell.car = new eshkol_ast_t;
        *result.cons_cell.car = ast.cons_cell.car ?
            substituteBindings(*ast.cons_cell.car, bindings) : eshkol_ast_t{};
        result.cons_cell.cdr = new eshkol_ast_t;
        *result.cons_cell.cdr = ast.cons_cell.cdr ?
            substituteBindings(*ast.cons_cell.cdr, bindings) : eshkol_ast_t{};
        return result;
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
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
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

            case ESHKOL_MATCH_OP:
                if (op->match_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = substituteBindings(*op->match_op.expr, bindings);
                    op->match_op.expr = new_expr;
                }
                if (op->match_op.num_clauses > 0 && op->match_op.clauses) {
                    for (uint64_t i = 0; i < op->match_op.num_clauses; i++) {
                        if (op->match_op.clauses[i].body) {
                            eshkol_ast_t* new_body = new eshkol_ast_t;
                            *new_body = substituteBindings(*op->match_op.clauses[i].body, bindings);
                            op->match_op.clauses[i].body = new_body;
                        }
                    }
                }
                break;

            // Ops that reuse call_op struct layout
            case ESHKOL_COND_OP:
            case ESHKOL_CASE_OP:
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
            case ESHKOL_DO_OP:
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

            case ESHKOL_SET_OP:
                if (op->set_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = substituteBindings(*op->set_op.value, bindings);
                    op->set_op.value = new_val;
                }
                break;

            case ESHKOL_GUARD_OP:
                if (op->guard_op.num_clauses > 0 && op->guard_op.clauses) {
                    eshkol_ast_t* new_clauses = new eshkol_ast_t[op->guard_op.num_clauses];
                    for (uint64_t i = 0; i < op->guard_op.num_clauses; i++) {
                        new_clauses[i] = substituteBindings(op->guard_op.clauses[i], bindings);
                    }
                    op->guard_op.clauses = new_clauses;
                }
                if (op->guard_op.num_body_exprs > 0 && op->guard_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t[op->guard_op.num_body_exprs];
                    for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                        new_body[i] = substituteBindings(op->guard_op.body[i], bindings);
                    }
                    op->guard_op.body = new_body;
                }
                break;

            case ESHKOL_RAISE_OP:
                if (op->raise_op.exception) {
                    eshkol_ast_t* new_exc = new eshkol_ast_t;
                    *new_exc = substituteBindings(*op->raise_op.exception, bindings);
                    op->raise_op.exception = new_exc;
                }
                break;

            case ESHKOL_VALUES_OP:
                if (op->values_op.num_values > 0 && op->values_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->values_op.num_values];
                    for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                        new_exprs[i] = substituteBindings(op->values_op.expressions[i], bindings);
                    }
                    op->values_op.expressions = new_exprs;
                }
                break;

            case ESHKOL_CALL_CC_OP:
                if (op->call_cc_op.proc) {
                    eshkol_ast_t* new_proc = new eshkol_ast_t;
                    *new_proc = substituteBindings(*op->call_cc_op.proc, bindings);
                    op->call_cc_op.proc = new_proc;
                }
                break;

            case ESHKOL_DYNAMIC_WIND_OP:
                if (op->dynamic_wind_op.before) {
                    eshkol_ast_t* new_before = new eshkol_ast_t;
                    *new_before = substituteBindings(*op->dynamic_wind_op.before, bindings);
                    op->dynamic_wind_op.before = new_before;
                }
                if (op->dynamic_wind_op.thunk) {
                    eshkol_ast_t* new_thunk = new eshkol_ast_t;
                    *new_thunk = substituteBindings(*op->dynamic_wind_op.thunk, bindings);
                    op->dynamic_wind_op.thunk = new_thunk;
                }
                if (op->dynamic_wind_op.after) {
                    eshkol_ast_t* new_after = new eshkol_ast_t;
                    *new_after = substituteBindings(*op->dynamic_wind_op.after, bindings);
                    op->dynamic_wind_op.after = new_after;
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
            // R7RS: _ is a wildcard — match without binding
            if (var_name == "_") return true;
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
