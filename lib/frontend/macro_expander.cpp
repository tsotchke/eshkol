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

/**
 * @brief Constructs a macro expander with a single, empty global scope.
 */
MacroExpander::MacroExpander() {
    // Initialize with a global scope
    scope_stack_.emplace_back();
}

/**
 * @brief Destroys the expander. Registered macro definitions are owned by the
 * AST they came from, so nothing is freed here.
 */
MacroExpander::~MacroExpander() {
    // Macro definitions are owned by the AST, not by us
}

/**
 * @brief Pushes a new, empty macro scope onto the scope stack (used when
 * entering a `let-syntax`/`letrec-syntax` body).
 */
void MacroExpander::pushScope() {
    scope_stack_.emplace_back();
}

/**
 * @brief Pops the innermost macro scope, restoring the enclosing one.
 *
 * The global (outermost) scope is never popped, so this is a no-op once only
 * one scope remains on the stack.
 */
void MacroExpander::popScope() {
    if (scope_stack_.size() > 1) {
        scope_stack_.pop_back();
    }
}

/**
 * @brief Registers a `define-syntax` (or `let-syntax`/`letrec-syntax`) macro
 * definition under its name in the innermost scope.
 *
 * Silently does nothing if @p macro is null, has no name, or the scope stack
 * is empty.
 */
void MacroExpander::registerMacro(const eshkol_macro_def_t* macro) {
    if (macro && macro->name && !scope_stack_.empty()) {
        scope_stack_.back()[macro->name] = const_cast<eshkol_macro_def_t*>(macro);
    }
}

/**
 * @brief Looks up a macro by name, searching scopes from innermost to
 * outermost so that locally shadowing definitions win.
 *
 * @return The matching macro definition, or nullptr if @p name is not bound
 * to a macro in any active scope.
 */
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

/**
 * @brief Reports whether @p name currently resolves to a registered macro.
 */
bool MacroExpander::isMacro(const std::string& name) const {
    return lookupMacro(name) != nullptr;
}

/**
 * @brief Expands a top-level sequence of forms, applying two passes so
 * macros can be used before their textually-later `define-syntax` sibling
 * definitions are all visible (mirrors R7RS top-level macro scoping).
 *
 * The first pass registers every top-level `define-syntax` definition. The
 * second pass expands every non-`define-syntax` form (define-syntax forms
 * themselves produce no runtime code and are dropped from the result).
 *
 * @return The expanded forms, in original order, with all `define-syntax`
 * forms removed.
 */
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

/**
 * @brief Public entry point for expanding a single top-level or nested form.
 *
 * Thin forwarding wrapper around expandNode().
 */
eshkol_ast_t MacroExpander::expand(const eshkol_ast_t& ast) {
    return expandNode(ast);
}

/**
 * @brief Core recursive macro-expansion driver: repeatedly expands macro
 * calls at the current node, then descends into sub-expressions.
 *
 * A macro call is expanded iteratively (via a `for (;;)` loop) rather than by
 * recursive self-call, so a macro that expands into another macro call does
 * not grow the C++ call stack; a per-expansion-chain @c expansion_chain set
 * detects a macro expanding back into itself and reports a circular-expansion
 * error instead of looping forever. A thread-local @c expansion_depth guard
 * also caps total nested expandNode() recursion (from descending into child
 * forms) at 1000 to bound runaway expansion.
 *
 * Along the way this handles the three macro-introducing forms directly:
 * `define-syntax` is registered and erased (replaced with a null AST, since
 * it produces no runtime code); `let-syntax`/`letrec-syntax` push a scope,
 * register their local macros, expand the body, and pop the scope. Once no
 * more macro calls apply at this node, sub-expressions of every recognized
 * operation kind (calls, sequences, define/lambda/let-family bindings and
 * bodies, cond/case/when/unless/do, set!, guard, raise, values, call/cc,
 * dynamic-wind, and quasiquote/unquote/unquote-splicing operands) are
 * recursively expanded so nested macro uses anywhere in the tree are also
 * expanded. Quoted (`quote`) data is deliberately left un-expanded since it
 * is literal data, not code.
 *
 * @return The fully macro-expanded AST for this node and its subtree.
 */
eshkol_ast_t MacroExpander::expandNode(const eshkol_ast_t& ast) {
    // Use iterative re-expansion for macro calls to prevent unbounded recursion.
    // A macro expanding to another macro call is handled by looping, not recursing.
    // We track seen macro names per expansion chain to detect cycles.
    static thread_local int expansion_depth = 0;
    struct DepthGuard { DepthGuard() { ++expansion_depth; } ~DepthGuard() { --expansion_depth; } } depth_guard;
    if (expansion_depth > 1000) {
        // A deeply nested ordinary call is AST traversal, not macro
        // expansion. Reporting it as a macro-depth failure made a pure
        // arithmetic expression fail before LLVM codegen could measure its
        // actual complexity (ESH-0103). Keep the bound for real macro forms,
        // where it prevents runaway syntax expansion, but let non-macro
        // subtrees pass through unchanged. The caller's existing recursive
        // walk then returns the original deep subtree and codegen processes it
        // without inventing diagnostics for a macro-free program.
        bool is_macro_form = false;
        if (ast.type == ESHKOL_OP) {
            const auto op = ast.operation.op;
            if (op == ESHKOL_LET_SYNTAX_OP || op == ESHKOL_LETREC_SYNTAX_OP) {
                is_macro_form = true;
            } else if (op == ESHKOL_CALL_OP && ast.operation.call_op.func &&
                       ast.operation.call_op.func->type == ESHKOL_VAR &&
                       ast.operation.call_op.func->variable.id) {
                is_macro_form = isMacro(ast.operation.call_op.func->variable.id);
            }
        }
        if (is_macro_form) {
            eshkol_error("macro expansion depth limit exceeded (>1000)");
            return ast;
        }
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
            // Descend into quasiquote and unquote/unquote-splicing so macro
            // calls that a template introduced inside an unquote escape get
            // re-expanded (e.g. `(car `(,(+ (add1q 0) 1)))`). Note: QUOTE_OP is
            // deliberately NOT here — quoted forms are literal data and must not
            // be macro-expanded. Quasiquoted sub-lists are built as (list …)
            // calls with literal atoms, so real macro calls only ever appear in
            // unquote regions, which is exactly what we recurse through.
            case ESHKOL_QUASIQUOTE_OP:
            case ESHKOL_UNQUOTE_OP:
            case ESHKOL_UNQUOTE_SPLICING_OP:
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
            case ESHKOL_LETREC_STAR_OP:
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

            case ESHKOL_THE_OP:
                // Expand macros inside the wrapped expression of a (the T e)
                // ascription; the type expression carries no macro calls.
                if (op->the_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = expandNode(*op->the_op.expr);
                    op->the_op.expr = new_expr;
                }
                break;

            default:
                break;
        }
    }

    return result;
}

/**
 * @brief Attempts to expand one macro call by trying each `syntax-rules` rule
 * of the called macro in order until one matches.
 *
 * Looks up the macro named by @p call's callee, builds its literals list, and
 * for each rule tries matchPatternSeq() against the call's arguments
 * (skipping the macro-name pattern element itself). The first rule whose
 * pattern matches has its template instantiated via instantiateTemplate().
 *
 * @return The instantiated template AST on a matching rule; if the macro
 * name is not actually registered, or no rule's pattern matches (a syntax
 * error, reported via eshkol_error()), returns @p call unchanged.
 */
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

        // Match the macro-call arguments against the pattern elements
        // (skipping the macro-name element at index 0). This shares the
        // exact same sequence-matching engine used for nested list
        // sub-patterns, so ellipsis — including nested ellipsis like
        // ((r ...) ...) — is handled uniformly everywhere it can occur.
        bool matched = matchPatternSeq(pat_list->elements, /*pat_start=*/1, pat_list->num_elements,
                                        call_op->variables, call_op->num_vars, literals, bindings);

        if (matched) {
            // Rule matched! Instantiate the template
            return instantiateTemplate(rule->template_, bindings);
        }
    }

    // No rule matched — this is a syntax error, not a warning
    eshkol_error("syntax error: no matching pattern for macro '%s'", macro_name.c_str());
    return call;
}

/**
 * @brief Instantiates a matched macro rule's template into a concrete AST
 * node, dispatching on the template's shape.
 *
 * A @c MACRO_TPL_LITERAL template is literal AST data with pattern variables
 * substituted via substituteBindings(); a @c MACRO_TPL_VARIABLE template
 * looks up its name in @p bindings and copies the bound value (falling back
 * to a bare symbol reference if unbound); a @c MACRO_TPL_LIST template
 * recursively instantiates each element and rebuilds them as a call
 * expression (first element is the callee, the rest are arguments).
 *
 * @return The instantiated AST. Returns a null AST for a null @p tmpl, an
 * unbound/absent variable template, or an empty list template.
 */
eshkol_ast_t MacroExpander::instantiateTemplate(const eshkol_macro_template_t* tmpl,
                                                  const Bindings& bindings) {
    if (!tmpl) {
        eshkol_ast_t null_ast;
        eshkol_ast_make_null(&null_ast);
        return null_ast;
    }

    // Each instantiation starts from an empty rename scope: renames are local
    // to one expansion of one template. The counter is NOT reset, so names
    // stay unique across the whole run.
    active_renames_.clear();

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
            if (it != bindings.end() && matchTreeHasValue(it->second.tree)) {
                return copyAst(matchTreeFirstScalar(it->second.tree));
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

/**
 * @brief Reports whether @p ast is the literal ellipsis symbol `...` used to
 * mark repeated pattern/template elements in `syntax-rules`.
 */
bool MacroExpander::isEllipsisSymbol(const eshkol_ast_t& ast) const {
    return ast.type == ESHKOL_VAR && ast.variable.id &&
           std::string(ast.variable.id) == "...";
}

/**
 * @brief Reports whether a MatchTree holds at least one concrete matched
 * value.
 *
 * A depth-0 (scalar) tree always has a value; a repeated (depth >= 1) tree
 * has a value only if it matched at least one repetition.
 */
bool MacroExpander::matchTreeHasValue(const MatchTree& tree) {
    if (tree.depth == 0) return true;
    return !tree.elements.empty();
}

/**
 * @brief Descends into a MatchTree's first repetition at each level until
 * reaching a scalar (depth-0) leaf, and returns that matched AST.
 *
 * Used to obtain a representative concrete value for a pattern variable
 * without needing to know its full repetition structure.
 */
const eshkol_ast_t& MacroExpander::matchTreeFirstScalar(const MatchTree& tree) {
    if (tree.depth == 0) return tree.scalar;
    return matchTreeFirstScalar(tree.elements.front());
}

/**
 * @brief Produces the bindings map for one repetition of an ellipsis
 * expansion by "peeling" every repeated binding down one MatchTree level.
 *
 * For each entry in @p bindings: a non-repeated (depth-0) binding is shared
 * unchanged across every repetition; a repeated binding with at least one
 * matched element selects the sub-tree at @p index (clamped to the last
 * available element, matching how R7RS treats unequal-length repeated
 * bindings under one ellipsis); a repeated binding that matched zero
 * repetitions peels to an empty tree one level shallower.
 *
 * @return A new Bindings map holding the binding values applicable to
 * repetition @p index.
 */
MacroExpander::Bindings MacroExpander::peelBindingsAtIndex(const Bindings& bindings, size_t index) const {
    Bindings result;
    for (const auto& kv : bindings) {
        Binding nb;
        nb.name = kv.first;
        const MatchTree& tree = kv.second.tree;
        if (tree.depth == 0) {
            // Not repeated at this level — shared unchanged across every repetition.
            nb.tree = tree;
        } else if (!tree.elements.empty()) {
            size_t idx = std::min(index, tree.elements.size() - 1);
            nb.tree = tree.elements[idx];
        } else {
            // Matched zero repetitions — peel to an empty tree one level shallower.
            nb.tree.depth = tree.depth - 1;
        }
        result[kv.first] = nb;
    }
    return result;
}

/**
 * @brief Searches a template subtree for a pattern variable bound at
 * ellipsis depth >= 1, to use as the "driver" that determines how many
 * repetitions an adjacent `...` should expand to.
 *
 * Recurses through variable references, cons cells, and every recognized
 * operation kind's sub-expressions (mirroring the same traversal shape used
 * by expandNode()/substituteBindings()), stopping as soon as any qualifying
 * variable is found.
 *
 * @param binding_name Out-parameter set to the name of the first
 * depth->=1 pattern variable found; left unmodified if none is found.
 * @return true if a driver variable was found (and @p binding_name set),
 * false otherwise (e.g. the template element has no repeated pattern
 * variable at all — a misplaced-ellipsis error case handled by the caller).
 */
bool MacroExpander::findEllipsisDriver(const eshkol_ast_t& ast,
                                        const Bindings& bindings,
                                        std::string& binding_name) const {
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        std::string name = ast.variable.id;
        auto it = bindings.find(name);
        if (it != bindings.end() && it->second.tree.depth >= 1) {
            binding_name = name;
            return true;
        }
        return false;
    }

    if (ast.type == ESHKOL_CONS) {
        return (ast.cons_cell.car &&
                findEllipsisDriver(*ast.cons_cell.car, bindings, binding_name)) ||
               (ast.cons_cell.cdr &&
                findEllipsisDriver(*ast.cons_cell.cdr, bindings, binding_name));
    }

    if (ast.type != ESHKOL_OP) {
        return false;
    }

    const auto* op = &ast.operation;
    switch (op->op) {
        case ESHKOL_CALL_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_DO_OP:
        // Same call_op layout for the quote family — an ellipsis-repeated
        // template element may reference its driving pattern variable from
        // inside (quasi)quoted data (parallels the quote-family recursion in
        // substituteBindings).
        case ESHKOL_QUASIQUOTE_OP:
        case ESHKOL_UNQUOTE_OP:
        case ESHKOL_UNQUOTE_SPLICING_OP:
        case ESHKOL_QUOTE_OP:
            if (op->call_op.func &&
                findEllipsisDriver(*op->call_op.func, bindings, binding_name)) {
                return true;
            }
            for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                if (findEllipsisDriver(op->call_op.variables[i], bindings, binding_name)) {
                    return true;
                }
            }
            return false;

        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                if (findEllipsisDriver(op->sequence_op.expressions[i], bindings, binding_name)) {
                    return true;
                }
            }
            return false;

        case ESHKOL_DEFINE_OP:
            return op->define_op.value &&
                   findEllipsisDriver(*op->define_op.value, bindings, binding_name);

        case ESHKOL_LAMBDA_OP:
            return op->lambda_op.body &&
                   findEllipsisDriver(*op->lambda_op.body, bindings, binding_name);

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                if (findEllipsisDriver(op->let_op.bindings[i], bindings, binding_name)) {
                    return true;
                }
            }
            return op->let_op.body &&
                   findEllipsisDriver(*op->let_op.body, bindings, binding_name);

        case ESHKOL_MATCH_OP:
            if (op->match_op.expr &&
                findEllipsisDriver(*op->match_op.expr, bindings, binding_name)) {
                return true;
            }
            for (uint64_t i = 0; i < op->match_op.num_clauses; i++) {
                if (op->match_op.clauses[i].body &&
                    findEllipsisDriver(*op->match_op.clauses[i].body, bindings, binding_name)) {
                    return true;
                }
            }
            return false;

        case ESHKOL_SET_OP:
            return op->set_op.value &&
                   findEllipsisDriver(*op->set_op.value, bindings, binding_name);

        case ESHKOL_GUARD_OP:
            for (uint64_t i = 0; i < op->guard_op.num_clauses; i++) {
                if (findEllipsisDriver(op->guard_op.clauses[i], bindings, binding_name)) {
                    return true;
                }
            }
            for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                if (findEllipsisDriver(op->guard_op.body[i], bindings, binding_name)) {
                    return true;
                }
            }
            return false;

        case ESHKOL_RAISE_OP:
            return op->raise_op.exception &&
                   findEllipsisDriver(*op->raise_op.exception, bindings, binding_name);

        case ESHKOL_VALUES_OP:
            for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                if (findEllipsisDriver(op->values_op.expressions[i], bindings, binding_name)) {
                    return true;
                }
            }
            return false;

        case ESHKOL_CALL_CC_OP:
            return op->call_cc_op.proc &&
                   findEllipsisDriver(*op->call_cc_op.proc, bindings, binding_name);

        case ESHKOL_DYNAMIC_WIND_OP:
            return (op->dynamic_wind_op.before &&
                    findEllipsisDriver(*op->dynamic_wind_op.before, bindings, binding_name)) ||
                   (op->dynamic_wind_op.thunk &&
                    findEllipsisDriver(*op->dynamic_wind_op.thunk, bindings, binding_name)) ||
                   (op->dynamic_wind_op.after &&
                    findEllipsisDriver(*op->dynamic_wind_op.after, bindings, binding_name));

        default:
            return false;
    }
}

/**
 * @brief Expands a single template element followed by one or more
 * consecutive `...` markers into the list of AST nodes it produces.
 *
 * Finds the repetition count via findEllipsisDriver() (the number of matched
 * repetitions of the driving pattern variable), then for each repetition
 * index peels @p bindings down to that repetition's values
 * (peelBindingsAtIndex()) and substitutes @p ast against them. When
 * @p ellipsis_count is greater than 1 (R7RS nested-ellipsis flattening, e.g.
 * `row ... ...`), each repetition's result is itself recursively expanded one
 * ellipsis shallower and the results are concatenated (spliced) rather than
 * nested.
 *
 * @param ellipsis_count Number of consecutive `...` markers following
 * @p ast in the template.
 * @return The flattened list of expanded AST nodes, one per repetition (or
 * more, if further nested ellipsis splice additional elements in). If no
 * driver variable can be found, reports a misplaced-ellipsis error and
 * returns a single-element vector holding @p ast substituted as-is.
 */
std::vector<eshkol_ast_t> MacroExpander::expandEllipsisElementN(const eshkol_ast_t& ast,
                                                                  const Bindings& bindings,
                                                                  int ellipsis_count) {
    std::vector<eshkol_ast_t> expanded;

    std::string drive;
    if (!findEllipsisDriver(ast, bindings, drive)) {
        eshkol_error("misplaced ellipsis in macro template");
        expanded.push_back(substituteBindings(ast, bindings));
        return expanded;
    }

    size_t n = bindings.at(drive).tree.elements.size();
    expanded.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Bindings peeled = peelBindingsAtIndex(bindings, i);
        if (ellipsis_count <= 1) {
            // One ellipsis consumed and none left: substitute this repetition
            // as a single template result (may itself contain further,
            // independent ellipses — substituteBindings handles those).
            expanded.push_back(substituteBindings(ast, peeled));
        } else {
            // Another consecutive ellipsis follows in the template: this
            // repetition's result is itself flattened (spliced) rather than
            // nested, so recurse one ellipsis shallower and concatenate.
            std::vector<eshkol_ast_t> sub = expandEllipsisElementN(ast, peeled, ellipsis_count - 1);
            expanded.insert(expanded.end(), sub.begin(), sub.end());
        }
    }
    return expanded;
}

/**
 * @brief Substitutes template bindings across a flat array of template
 * elements, expanding and flattening any element followed by one or more
 * `...` markers.
 *
 * Walks @p items left to right; for each element it counts how many
 * consecutive ellipsis-symbol items follow it (R7RS nested ellipsis, e.g.
 * `row ... ...`, flattens one extra level per additional consecutive `...`),
 * and if at least one follows, delegates to expandEllipsisElementN() and
 * inlines its results. A bare `...` with no preceding element to repeat is a
 * misplaced-ellipsis error and is skipped. Elements with no following
 * ellipsis are substituted individually via substituteBindings().
 *
 * @return The fully substituted and ellipsis-flattened list of AST nodes.
 */
std::vector<eshkol_ast_t> MacroExpander::substituteBindingsInList(const eshkol_ast_t* items,
                                                                   uint64_t count,
                                                                   const Bindings& bindings) {
    std::vector<eshkol_ast_t> result;
    result.reserve(count);

    for (uint64_t i = 0; i < count; ) {
        // Count consecutive ellipsis markers following items[i] — R7RS
        // nested ellipsis ("row ... ...") flattens one extra level per
        // additional consecutive ellipsis.
        uint64_t j = i + 1;
        int ellipsis_count = 0;
        while (j < count && isEllipsisSymbol(items[j])) {
            ellipsis_count++;
            j++;
        }

        if (ellipsis_count > 0) {
            std::vector<eshkol_ast_t> expanded = expandEllipsisElementN(items[i], bindings, ellipsis_count);
            result.insert(result.end(), expanded.begin(), expanded.end());
            i = j;
            continue;
        }

        if (isEllipsisSymbol(items[i])) {
            eshkol_error("misplaced ellipsis in macro template");
            i++;
            continue;
        }

        result.push_back(substituteBindings(items[i], bindings));
        i++;
    }

    return result;
}

/**
 * @brief Recursively substitutes matched pattern-variable bindings into a
 * macro template subtree, producing the (non-ellipsis-flattened) expanded
 * AST for @p ast.
 *
 * A variable reference bound in @p bindings is replaced by a copy of its
 * matched value (via matchTreeFirstScalar()); an unbound variable is copied
 * as-is. Cons cells recurse into car/cdr. Operation nodes recurse into their
 * relevant sub-expressions per operation kind (calls, sequences,
 * define/lambda/let-family bindings and bodies, cond/case/when/unless/do,
 * set!, guard, raise, values, call/cc, dynamic-wind, and
 * quote/quasiquote/unquote/unquote-splicing operands — quoted data is
 * recursed into here, unlike expandNode(), because pattern variables inside
 * quoted template data must still be substituted per R7RS 4.3.2). List-typed
 * sub-expression arrays are substituted via substituteBindingsInList() so
 * ellipsis elements within them are expanded/flattened. Any other AST type is
 * simply deep-copied via copyAst().
 *
 * @return The substituted AST subtree.
 */
/**
 * @brief Allocate a fresh name for a template-introduced binder and bind it in
 * the active rename scope (R7RS 4.3.2 hygiene).
 *
 * The counter is monotonic across the whole expansion run, so two invocations
 * of the same macro receive distinct names — which is required, because their
 * bindings are distinct. The `.` in the generated name mirrors the VM
 * expander's format so the two engines are recognisably doing the same thing.
 */
std::string MacroExpander::freshName(const std::string& name) {
    std::string fresh = "_h" + std::to_string(rename_counter_++) + "." + name;
    active_renames_[name] = fresh;
    return fresh;
}

/**
 * @brief Reports whether a template identifier in a binder position may be
 * alpha-renamed.
 *
 * Pattern variables may not: they are replaced by caller code, which must keep
 * the caller's own names. `_` is not an identifier.
 */
bool MacroExpander::isRenameableBinder(const std::string& name, const Bindings& bindings) {
    if (name.empty() || name == "_") return false;
    return bindings.find(name) == bindings.end();
}

eshkol_ast_t MacroExpander::substituteBindings(const eshkol_ast_t& ast,
                                                 const Bindings& bindings) {
    // Check if this is a variable that should be substituted
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        std::string name = ast.variable.id;
        auto it = bindings.find(name);
        if (it != bindings.end() && matchTreeHasValue(it->second.tree)) {
            return copyAst(matchTreeFirstScalar(it->second.tree));
        }
        // Hygiene: a template identifier bound by the template itself carries
        // the fresh name allocated at its binder (R7RS 4.3.2). Anything else
        // is free in the template and is emitted verbatim.
        auto rn = active_renames_.find(name);
        if (!in_datum_ && rn != active_renames_.end()) {
            eshkol_ast_t result = copyAst(ast);
            // copyAst() strdup()s the id, so this must be free()/strdup() too.
            free(result.variable.id);
            result.variable.id = strdup(rn->second.c_str());
            return result;
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
            // quasiquote/unquote/unquote-splicing/quote store their operand(s)
            // in the same call_op layout (func=nullptr, variables[], num_vars).
            // R7RS §4.3.2: pattern variables occurring anywhere in a template —
            // including inside (quasi)quoted data and unquote escapes — must be
            // replaced by the matched input subforms. Recursing here is what
            // makes `(car `(,(+ x 1)))`-style macro templates substitute x
            // (previously these ops hit default: and copied the operand verbatim,
            // leaving x undefined and collapsing nested expansions).
            case ESHKOL_QUASIQUOTE_OP:
            case ESHKOL_UNQUOTE_OP:
            case ESHKOL_UNQUOTE_SPLICING_OP:
            case ESHKOL_QUOTE_OP: {
                // Hygiene: a symbol in quoted data is DATA, so it keeps its
                // literal name; unquote escapes back to identifier position.
                // Pattern-variable substitution is unaffected either way.
                const bool saved_datum = in_datum_;
                if (op->op == ESHKOL_QUOTE_OP || op->op == ESHKOL_QUASIQUOTE_OP) {
                    in_datum_ = true;
                } else if (op->op == ESHKOL_UNQUOTE_OP ||
                           op->op == ESHKOL_UNQUOTE_SPLICING_OP) {
                    in_datum_ = false;
                }
                if (op->call_op.func) {
                    eshkol_ast_t* new_func = new eshkol_ast_t;
                    *new_func = substituteBindings(*op->call_op.func, bindings);
                    op->call_op.func = new_func;
                }
                if (op->call_op.num_vars > 0 && op->call_op.variables) {
                    std::vector<eshkol_ast_t> new_vars_vec =
                        substituteBindingsInList(op->call_op.variables, op->call_op.num_vars, bindings);
                    op->call_op.num_vars = new_vars_vec.size();
                    if (op->call_op.num_vars > 0) {
                        eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                        for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                            new_vars[i] = new_vars_vec[i];
                        }
                        op->call_op.variables = new_vars;
                    } else {
                        op->call_op.variables = nullptr;
                    }
                }
                in_datum_ = saved_datum;
                break;
            }

            case ESHKOL_SEQUENCE_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
                if (op->sequence_op.num_expressions > 0 && op->sequence_op.expressions) {
                    std::vector<eshkol_ast_t> new_exprs_vec =
                        substituteBindingsInList(op->sequence_op.expressions,
                                                 op->sequence_op.num_expressions,
                                                 bindings);
                    op->sequence_op.num_expressions = new_exprs_vec.size();
                    if (op->sequence_op.num_expressions > 0) {
                        eshkol_ast_t* new_exprs = new eshkol_ast_t[op->sequence_op.num_expressions];
                        for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                            new_exprs[i] = new_exprs_vec[i];
                        }
                        op->sequence_op.expressions = new_exprs;
                    } else {
                        op->sequence_op.expressions = nullptr;
                    }
                }
                break;

            case ESHKOL_DEFINE_OP:
                if (op->define_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = substituteBindings(*op->define_op.value, bindings);
                    op->define_op.value = new_val;
                }
                break;

            case ESHKOL_LAMBDA_OP: {
                // Hygiene: a template-introduced parameter is fresh in the body.
                auto saved_renames = active_renames_;
                if (op->lambda_op.num_params > 0 && op->lambda_op.parameters) {
                    eshkol_ast_t* new_params = new eshkol_ast_t[op->lambda_op.num_params];
                    for (uint64_t i = 0; i < op->lambda_op.num_params; i++) {
                        const eshkol_ast_t& p = op->lambda_op.parameters[i];
                        if (p.type == ESHKOL_VAR && p.variable.id &&
                            isRenameableBinder(p.variable.id, bindings)) {
                            freshName(p.variable.id);
                        }
                        new_params[i] = substituteBindings(p, bindings);
                    }
                    op->lambda_op.parameters = new_params;
                }
                if (op->lambda_op.rest_param &&
                    isRenameableBinder(op->lambda_op.rest_param, bindings)) {
                    op->lambda_op.rest_param =
                        strdup(freshName(op->lambda_op.rest_param).c_str());
                }
                if (op->lambda_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = substituteBindings(*op->lambda_op.body, bindings);
                    op->lambda_op.body = new_body;
                }
                active_renames_ = saved_renames;
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP: {
                // Hygiene: each template-introduced binder gets a fresh name,
                // visible exactly where the binding construct makes it visible.
                //   let          inits see the OUTER scope
                //   let*         init i sees binders 0..i-1
                //   letrec(*)    every init sees every binder
                //   named let    the loop name is bound in the body
                const bool is_rec = (op->op == ESHKOL_LETREC_OP ||
                                     op->op == ESHKOL_LETREC_STAR_OP);
                const bool is_seq = (op->op == ESHKOL_LET_STAR_OP);
                auto saved_renames = active_renames_;

                if (is_rec && op->let_op.num_bindings > 0 && op->let_op.bindings) {
                    for (uint64_t i = 0; i < op->let_op.num_bindings; i++) {
                        const eshkol_ast_t& b = op->let_op.bindings[i];
                        if (b.type == ESHKOL_CONS && b.cons_cell.car &&
                            b.cons_cell.car->type == ESHKOL_VAR &&
                            b.cons_cell.car->variable.id &&
                            isRenameableBinder(b.cons_cell.car->variable.id, bindings)) {
                            freshName(b.cons_cell.car->variable.id);
                        }
                    }
                }
                if (op->let_op.name && isRenameableBinder(op->let_op.name, bindings)) {
                    op->let_op.name = strdup(freshName(op->let_op.name).c_str());
                }

                if (op->let_op.num_bindings > 0 && op->let_op.bindings) {
                    const uint64_t n = op->let_op.num_bindings;
                    eshkol_ast_t* new_bindings = new eshkol_ast_t[n];
                    std::vector<eshkol_ast_t> inits(n);
                    std::vector<bool> is_cons(n, false);

                    // Pass 1 — inits. For plain `let` every init is substituted
                    // while NO binder of this group is in scope (that is what
                    // makes `let` parallel); `let*` accumulates as it goes;
                    // `letrec` already has the whole group in scope.
                    for (uint64_t i = 0; i < n; i++) {
                        const eshkol_ast_t& b = op->let_op.bindings[i];
                        if (b.type != ESHKOL_CONS || !b.cons_cell.car) continue;
                        is_cons[i] = true;
                        inits[i] = b.cons_cell.cdr
                            ? substituteBindings(*b.cons_cell.cdr, bindings)
                            : eshkol_ast_t{};
                        if (is_seq && b.cons_cell.car->type == ESHKOL_VAR &&
                            b.cons_cell.car->variable.id &&
                            isRenameableBinder(b.cons_cell.car->variable.id, bindings)) {
                            freshName(b.cons_cell.car->variable.id);
                        }
                    }

                    // Pass 2 — binders. For plain `let` they all enter scope
                    // together, after every init has been substituted.
                    if (!is_rec && !is_seq) {
                        for (uint64_t i = 0; i < n; i++) {
                            if (!is_cons[i]) continue;
                            const eshkol_ast_t* car = op->let_op.bindings[i].cons_cell.car;
                            if (car->type == ESHKOL_VAR && car->variable.id &&
                                isRenameableBinder(car->variable.id, bindings)) {
                                freshName(car->variable.id);
                            }
                        }
                    }

                    for (uint64_t i = 0; i < n; i++) {
                        const eshkol_ast_t& b = op->let_op.bindings[i];
                        if (!is_cons[i]) {
                            new_bindings[i] = substituteBindings(b, bindings);
                            continue;
                        }
                        eshkol_ast_t nb;
                        nb.type = ESHKOL_CONS;
                        nb.cons_cell.car = new eshkol_ast_t;
                        *nb.cons_cell.car = substituteBindings(*b.cons_cell.car, bindings);
                        nb.cons_cell.cdr = new eshkol_ast_t;
                        *nb.cons_cell.cdr = inits[i];
                        new_bindings[i] = nb;
                    }
                    op->let_op.bindings = new_bindings;
                }
                if (op->let_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = substituteBindings(*op->let_op.body, bindings);
                    op->let_op.body = new_body;
                }
                active_renames_ = saved_renames;
                break;
            }

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
                    std::vector<eshkol_ast_t> new_vars_vec =
                        substituteBindingsInList(op->call_op.variables, op->call_op.num_vars, bindings);
                    op->call_op.num_vars = new_vars_vec.size();
                    if (op->call_op.num_vars > 0) {
                        eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                        for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                            new_vars[i] = new_vars_vec[i];
                        }
                        op->call_op.variables = new_vars;
                    } else {
                        op->call_op.variables = nullptr;
                    }
                }
                break;

            case ESHKOL_SET_OP:
                // The TARGET is a bare char*, not an AST node, so it was never
                // reached by the substitution walk. A template like
                //     ((_ a b) (let ((tmp a)) (set! a b) (set! b tmp)))
                // therefore emitted a literal `a`, and the classic swap! macro
                // failed to compile with "set!: undefined variable 'a'".
                // R7RS 4.3.2: a pattern variable is substituted wherever it
                // occurs in the template, assignment targets included.
                if (op->set_op.name) {
                    std::string target = op->set_op.name;
                    auto it = bindings.find(target);
                    if (it != bindings.end() && matchTreeHasValue(it->second.tree)) {
                        const eshkol_ast_t& v = matchTreeFirstScalar(it->second.tree);
                        if (v.type == ESHKOL_VAR && v.variable.id) {
                            op->set_op.name = strdup(v.variable.id);
                        }
                        // A non-identifier operand in a set! target position is
                        // a user error; leave the name alone so the compiler
                        // reports it rather than silently mutating something.
                    } else {
                        auto rn = active_renames_.find(target);
                        if (rn != active_renames_.end()) {
                            op->set_op.name = strdup(rn->second.c_str());
                        }
                    }
                }
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
                    std::vector<eshkol_ast_t> new_body_vec =
                        substituteBindingsInList(op->guard_op.body,
                                                 op->guard_op.num_body_exprs,
                                                 bindings);
                    op->guard_op.num_body_exprs = new_body_vec.size();
                    if (op->guard_op.num_body_exprs > 0) {
                        eshkol_ast_t* new_body = new eshkol_ast_t[op->guard_op.num_body_exprs];
                        for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                            new_body[i] = new_body_vec[i];
                        }
                        op->guard_op.body = new_body;
                    } else {
                        op->guard_op.body = nullptr;
                    }
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
                    std::vector<eshkol_ast_t> new_values_vec =
                        substituteBindingsInList(op->values_op.expressions,
                                                 op->values_op.num_values,
                                                 bindings);
                    op->values_op.num_values = new_values_vec.size();
                    if (op->values_op.num_values > 0) {
                        eshkol_ast_t* new_exprs = new eshkol_ast_t[op->values_op.num_values];
                        for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                            new_exprs[i] = new_values_vec[i];
                        }
                        op->values_op.expressions = new_exprs;
                    } else {
                        op->values_op.expressions = nullptr;
                    }
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

/**
 * @brief Shallow-copies an AST node, deep-copying any owned string data so
 * the copy does not alias the original's heap allocations.
 *
 * For @c ESHKOL_STRING and @c ESHKOL_VAR nodes, the string/identifier
 * pointer is duplicated via strdup(). @c ESHKOL_OP nodes are copied
 * shallowly here; deep-copying their nested operand pointers is the
 * responsibility of the caller (expandNode()/substituteBindings()), which
 * know which sub-pointers are relevant for each operation kind. All other
 * node types are plain (primitive) data and are copied as-is.
 *
 * @return The copied AST node.
 */
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

/**
 * @brief Reports whether @p ast is a variable reference whose identifier
 * equals @p name.
 */
bool MacroExpander::isSymbol(const eshkol_ast_t& ast, const std::string& name) const {
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        return std::string(ast.variable.id) == name;
    }
    return false;
}

/**
 * @brief Returns the identifier of a variable-reference AST node.
 *
 * @return The variable's name, or an empty string if @p ast is not a
 * variable reference (or has a null identifier).
 */
std::string MacroExpander::getSymbolName(const eshkol_ast_t& ast) const {
    if (ast.type == ESHKOL_VAR && ast.variable.id) {
        return std::string(ast.variable.id);
    }
    return "";
}

/**
 * @brief Reports whether @p name is one of a macro's declared
 * `syntax-rules` literal identifiers (which must match exactly rather than
 * bind as a pattern variable).
 */
bool MacroExpander::isLiteral(const std::string& name,
                               const std::vector<std::string>& literals) const {
    return std::find(literals.begin(), literals.end(), name) != literals.end();
}

/**
 * @brief Matches a single `syntax-rules` pattern node against an input AST,
 * recording any pattern-variable bindings on success.
 *
 * Dispatches on pattern kind: @c MACRO_PAT_VARIABLE binds the matched AST
 * under the pattern's identifier (the wildcard `_` matches anything without
 * binding); @c MACRO_PAT_LITERAL requires @p ast to be a symbol reference
 * exactly equal to the pattern's literal identifier; @c MACRO_PAT_LIST
 * requires @p ast to be a call-shaped form and delegates element-by-element
 * matching (including ellipsis handling) to matchPatternSeq().
 *
 * @param bindings Accumulates matched pattern-variable bindings; only
 * modified on a successful match (partial matches from a failed sub-pattern
 * may still leave entries in @p bindings, since callers discard the whole
 * attempt on failure).
 * @return true if @p pattern matches @p ast, false otherwise (including a
 * null @p pattern).
 */
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
            binding.tree.depth = 0;
            binding.tree.scalar = ast;
            bindings[var_name] = binding;
            return true;
        }

        case MACRO_PAT_LITERAL: {
            // Literals must match exactly
            std::string lit_name = pattern->identifier ? pattern->identifier : "";
            return isSymbol(ast, lit_name);
        }

        case MACRO_PAT_LIST: {
            // List patterns match list-like ASTs (calls, sequences), which are
            // represented generically as [func, variables...]. Delegate to the
            // shared sequence matcher so ellipsis (including nested ellipsis,
            // e.g. a sub-pattern like (row ...) itself followed by ...) is
            // handled identically to a top-level macro-call pattern.
            if (ast.type != ESHKOL_OP) return false;

            const auto* op = &ast.operation;
            if (op->op != ESHKOL_CALL_OP || !op->call_op.func) return false;

            std::vector<eshkol_ast_t> args;
            args.reserve(op->call_op.num_vars + 1);
            args.push_back(*op->call_op.func);
            for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                args.push_back(op->call_op.variables[i]);
            }

            return matchPatternSeq(pattern->list.elements, 0, pattern->list.num_elements,
                                    args.data(), args.size(), literals, bindings);
        }

        default:
            return false;
    }
}

/**
 * @brief Matches a sequence of `syntax-rules` pattern elements (starting at
 * @p pat_start) against a sequence of input argument ASTs, handling ellipsis
 * repetition.
 *
 * Shared by both top-level macro-call matching (tryExpandMacroCall(), which
 * skips the macro-name element) and nested list sub-pattern matching
 * (matchPattern()'s @c MACRO_PAT_LIST case), so nested ellipsis (e.g.
 * `((r ...) ...)`) is handled uniformly wherever it occurs. A pattern element
 * marked @c followed_by_ellipsis greedily consumes every remaining argument
 * as one repetition each of that element's sub-pattern; the per-repetition
 * bindings are then merged into a single depth+1 MatchTree per pattern
 * variable the element can bind (computed via collectPatternVarDepths()), so
 * even a zero-repetition match binds those variables to an empty sequence
 * rather than leaving them unbound. An element with no ellipsis consumes
 * exactly one argument.
 *
 * @return true only if every pattern element matched and the entire argument
 * sequence was consumed exactly (no leftover arguments or unmatched
 * patterns); false otherwise.
 */
bool MacroExpander::matchPatternSeq(eshkol_macro_pattern_t* const* elements,
                                     uint64_t pat_start, uint64_t pat_count,
                                     const eshkol_ast_t* args, uint64_t arg_count,
                                     const std::vector<std::string>& literals,
                                     Bindings& bindings) {
    uint64_t arg_idx = 0;

    for (uint64_t pat_idx = pat_start; pat_idx < pat_count; pat_idx++) {
        eshkol_macro_pattern_t* elem = elements[pat_idx];
        if (!elem) return false;

        if (elem->followed_by_ellipsis) {
            // Greedily consume every remaining argument as one repetition of
            // `elem`'s (possibly itself nested-ellipsis) sub-pattern.
            uint64_t reps = arg_count - arg_idx;
            std::vector<Bindings> sub_results(reps);
            for (uint64_t r = 0; r < reps; r++) {
                Bindings sub;
                if (!matchPattern(elem, args[arg_idx + r], literals, sub)) {
                    return false;
                }
                sub_results[r] = std::move(sub);
            }
            arg_idx += reps;

            // Merge the per-repetition bindings into one depth+1 MatchTree
            // per variable that `elem` can bind — computed structurally so
            // that a zero-repetition match still binds every such variable
            // to an empty sequence (R7RS 4.3.2) instead of leaving it
            // unbound (which previously surfaced as "Undefined variable").
            std::map<std::string, int> local_depths;
            collectPatternVarDepths(elem, literals, 0, local_depths);
            for (const auto& ld : local_depths) {
                const std::string& name = ld.first;
                int child_depth = ld.second;

                Binding b;
                b.name = name;
                b.tree.depth = child_depth + 1;
                b.tree.elements.reserve(reps);
                for (uint64_t r = 0; r < reps; r++) {
                    auto found = sub_results[r].find(name);
                    if (found != sub_results[r].end()) {
                        b.tree.elements.push_back(found->second.tree);
                    } else {
                        MatchTree empty;
                        empty.depth = child_depth;
                        b.tree.elements.push_back(empty);
                    }
                }
                bindings[name] = b;
            }
            continue;
        }

        // No ellipsis on this pattern element: consume exactly one argument.
        if (arg_idx >= arg_count) return false;
        if (!matchPattern(elem, args[arg_idx], literals, bindings)) return false;
        arg_idx++;
    }

    return arg_idx == arg_count;
}

/**
 * @brief Recursively walks a `syntax-rules` sub-pattern, recording each
 * pattern variable's ellipsis-nesting depth (number of enclosing `...`
 * repetitions) into @p out.
 *
 * @p depth is the nesting depth accumulated so far by the caller; each
 * @c MACRO_PAT_LIST child adds one more level if that child itself is
 * marked @c followed_by_ellipsis. Used by matchPatternSeq() to know, for
 * every variable bindable within an ellipsis-repeated element, what depth of
 * MatchTree to build even when a given repetition doesn't happen to bind it.
 *
 * @param out Out-parameter map of pattern-variable name to depth; entries
 * are added, never cleared, so callers should pass a fresh map per
 * ellipsis element.
 */
void MacroExpander::collectPatternVarDepths(const eshkol_macro_pattern_t* pattern,
                                             const std::vector<std::string>& literals,
                                             int depth,
                                             std::map<std::string, int>& out) const {
    if (!pattern) return;
    (void)literals;

    switch (pattern->type) {
        case MACRO_PAT_VARIABLE: {
            std::string name = pattern->identifier ? pattern->identifier : "";
            if (!name.empty() && name != "_") {
                out[name] = depth;
            }
            break;
        }

        case MACRO_PAT_LITERAL:
            break;

        case MACRO_PAT_LIST:
            for (uint64_t i = 0; i < pattern->list.num_elements; i++) {
                eshkol_macro_pattern_t* child = pattern->list.elements[i];
                if (!child) continue;
                int child_depth = depth + (child->followed_by_ellipsis ? 1 : 0);
                collectPatternVarDepths(child, literals, child_depth, out);
            }
            break;

        default:
            break;
    }
}

} // namespace eshkol
