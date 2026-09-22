/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MacroExpander implementation - Hygienic macro expansion for syntax-rules
 */

#include <eshkol/core/ast_routing.h>
#include <eshkol/frontend/ast_strings.h>
#include <eshkol/frontend/macro_expander.h>
#include <eshkol/frontend/syntax_color.h>
#include <eshkol/frontend/syntax_datum.h>
#include <eshkol/frontend/syntax_rules.h>
#include <eshkol/logger.h>
#include <cstring>
#include <algorithm>
#include <functional>

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
        MacroBinding binding;
        binding.macro = const_cast<eshkol_macro_def_t*>(macro);
        binding.value_env = value_renames_;
        binding.macro_env.reserve(scope_stack_.size());
        for (const auto& scope : scope_stack_) {
            std::map<std::string, eshkol_macro_def_t*> snapshot;
            for (const auto& item : scope) snapshot[item.first] = item.second.macro;
            binding.macro_env.push_back(std::move(snapshot));
        }
        binding.macro_env.back()[macro->name] = const_cast<eshkol_macro_def_t*>(macro);
        definition_bindings_[macro] = binding;
        scope_stack_.back()[macro->name] = std::move(binding);
    }
}

void MacroExpander::registerMacroWithEnv(
    const eshkol_macro_def_t* macro,
    const std::vector<std::map<std::string, eshkol_macro_def_t*>>& env) {
    if (!macro || !macro->name || scope_stack_.empty()) return;
    MacroBinding binding;
    binding.macro = const_cast<eshkol_macro_def_t*>(macro);
    binding.value_env = value_renames_;
    binding.macro_env = env;
    definition_bindings_[macro] = binding;
    scope_stack_.back()[macro->name] = std::move(binding);
}

/**
 * @brief Looks up a macro by name, searching scopes from innermost to
 * outermost so that locally shadowing definitions win.
 *
 * @return The matching macro definition, or nullptr if @p name is not bound
 * to a macro in any active scope.
 */
eshkol_macro_def_t* MacroExpander::lookupMacro(const std::string& name) const {
    auto alias = macro_aliases_.find(name);
    if (alias != macro_aliases_.end())
        return const_cast<eshkol_macro_def_t*>(alias->second);
    if (value_renames_.count(name)) return nullptr;
    // Search from innermost scope to outermost
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second.macro;
        }
    }
    return nullptr;
}

const MacroExpander::MacroBinding* MacroExpander::lookupBinding(const std::string& name) const {
    auto alias = macro_aliases_.find(name);
    if (alias != macro_aliases_.end()) {
        auto binding = definition_bindings_.find(alias->second);
        return binding == definition_bindings_.end() ? nullptr : &binding->second;
    }
    for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) return &found->second;
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

    // Global definitions share one region, including forward references.
    std::map<std::string, eshkol_macro_def_t*> global_macros;
    for (const auto& item : scope_stack_.front()) global_macros[item.first] = item.second.macro;
    for (auto& item : scope_stack_.front()) {
        item.second.macro_env = {global_macros};
        definition_bindings_[item.second.macro] = item.second;
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
    if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_QUASIQUOTE_OP)
        return expandQuasiquoted(ast, 0);
    // Use iterative re-expansion for macro calls to prevent unbounded recursion.
    // A macro expanding to another macro call is handled by looping, not recursing.
    // We track seen macro names per expansion chain to detect cycles.
    static thread_local int expansion_depth = 0;
    struct DepthGuard { DepthGuard() { ++expansion_depth; } ~DepthGuard() { --expansion_depth; } } depth_guard;
    // Nodes the expander creates for this form (as opposed to copies of
    // template nodes, which keep the template's own location) are born with
    // the location of the form being expanded.
    EshkolAstBirthLocationScope birth_location(ast.line, ast.column);
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
    // A transformer may legitimately rewrite a use into another use of
    // itself (continuation-passing macros do so once per element); only a
    // chain that never reaches a non-macro form is an error.
    static constexpr unsigned kMaxExpansionSteps = 100000;
    unsigned expansion_steps = 0;

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
            const auto outer_values = value_renames_;
            std::vector<std::map<std::string, eshkol_macro_def_t*>> outer_macro_env;
            for (const auto& scope : scope_stack_) {
                std::map<std::string, eshkol_macro_def_t*> snapshot;
                for (const auto& item : scope) snapshot[item.first] = item.second.macro;
                outer_macro_env.push_back(std::move(snapshot));
            }
            pushScope();
            for (uint64_t i = 0; i < ls->num_macros; i++) {
                if (ls->macros[i]) {
                    if (current.operation.op == ESHKOL_LET_SYNTAX_OP)
                        registerMacroWithEnv(ls->macros[i], outer_macro_env);
                    else
                        registerMacro(ls->macros[i]);
                }
            }
            for (uint64_t i = 0; i < ls->num_macros; ++i)
                if (ls->macros[i] && ls->macros[i]->name)
                    value_renames_.erase(ls->macros[i]->name);
            if (current.operation.op == ESHKOL_LETREC_SYNTAX_OP) {
                auto recursive_env = outer_macro_env;
                std::map<std::string, eshkol_macro_def_t*> group;
                for (const auto& item : scope_stack_.back()) group[item.first] = item.second.macro;
                recursive_env.push_back(std::move(group));
                for (auto& item : scope_stack_.back()) {
                    item.second.macro_env = recursive_env;
                    item.second.value_env = value_renames_;
                    definition_bindings_[item.second.macro] = item.second;
                }
            }
            eshkol_ast_t expanded_body = expandNode(*ls->body);
            popScope();
            value_renames_ = outer_values;
            return expanded_body;
        }

        // Check for macro call — if found, expand and LOOP (not recurse)
        if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_CALL_OP) {
            const auto* call = &current.operation.call_op;
            if (call->func && call->func->type == ESHKOL_VAR && call->func->variable.id) {
                std::string func_name = call->func->variable.id;
                if (isMacro(func_name)) {
                    if (++expansion_steps > kMaxExpansionSteps) {
                        eshkol_error("macro expansion of '%s' did not terminate after %u steps",
                                     func_name.c_str(), kMaxExpansionSteps);
                        return current;
                    }
                    eshkol_ast_t expanded = tryExpandMacroCall(current);
                    if (expanded.node_id == current.node_id && expanded.type == current.type &&
                        expanded.type == ESHKOL_OP && expanded.operation.op == ESHKOL_CALL_OP &&
                        expanded.operation.call_op.func == current.operation.call_op.func)
                        return current;          // no rule matched; already reported
                    current = expanded;
                    continue; // Re-expand iteratively
                }
                // The parser read this use as macro syntax, but here the
                // keyword is shadowed by a value binding (or was never
                // bound): it is an ordinary call.
                if (eshkol::syntax_use_unparsed(current.node_id))
                    current = reparseAsCall(current);
            }

        }

        // Not a macro call — break out to do tree traversal
        break;
    }

    // Recursively expand sub-expressions (tree depth is bounded by input nesting)
    if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_QUASIQUOTE_OP)
        return expandQuasiquoted(current, 0);
    if (current.type == ESHKOL_VAR && current.variable.id) {
        const std::string resolved = resolveValue(current.variable.id);
        if (resolved != current.variable.id) {
            eshkol_ast_t renamed = copyAst(current);
            renamed.variable.id = eshkol_ast_string_copy(resolved);
            return renamed;
        }
    }
    // CONS nodes are used by the parser for structural operands (notably the
    // binding/test clauses of `do` and let bindings).  They are still AST
    // trees: skipping them leaves identifiers introduced inside those
    // operands unrenamed, producing backend-only "undefined variable"
    // failures.
    if (current.type == ESHKOL_CONS) {
        eshkol_ast_t result = current;
        result.cons_cell.car = current.cons_cell.car
            ? new eshkol_ast_t(expandNode(*current.cons_cell.car)) : nullptr;
        result.cons_cell.cdr = current.cons_cell.cdr
            ? new eshkol_ast_t(expandNode(*current.cons_cell.cdr)) : nullptr;
        return result;
    }
    eshkol_ast_t result = copyAst(current);

    if (result.type == ESHKOL_OP) {
        auto* op = &result.operation;

        {
            enum class AstRoute {
                Call, Sequence, Define, Lambda, Let, Match,
                Cond, Set, Guard, Raise, Values, CallCc,
                DynamicWind, The, Compose, Tensor, Diff, Derivative, Gradient,
                Jacobian, Hessian, Divergence, Curl, Laplacian, DirectionalDeriv,
                Taylor, WithRegion, Owned, Move, Shared, WeakRef, Borrow,
                CallWithValues, LetValues, CaseLambda, Parameterize, CallPayload, Leaf
            };
            auto expand_ptr = [&](eshkol_ast_t*& child) {
                if (child) child = new eshkol_ast_t(expandNode(*child));
            };
            auto expand_array = [&](eshkol_ast_t*& items, uint64_t count) {
                if (!items) return;
                auto* fresh = new eshkol_ast_t[count];
                for (uint64_t i = 0; i < count; ++i) fresh[i] = expandNode(items[i]);
                items = fresh;
            };
            switch (eshkol::routeAstOperation(op->op,
                eshkol::AstRouteGroup<AstRoute::Call,
                    ESHKOL_CALL_OP, ESHKOL_IF_OP, ESHKOL_QUASIQUOTE_OP, ESHKOL_UNQUOTE_OP, ESHKOL_UNQUOTE_SPLICING_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Sequence,
                    ESHKOL_SEQUENCE_OP, ESHKOL_AND_OP, ESHKOL_OR_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Define, ESHKOL_DEFINE_OP>{},
                eshkol::AstRouteGroup<AstRoute::Lambda, ESHKOL_LAMBDA_OP>{},
                eshkol::AstRouteGroup<AstRoute::Let,
                    ESHKOL_LET_OP, ESHKOL_LET_STAR_OP, ESHKOL_LETREC_OP, ESHKOL_LETREC_STAR_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Match, ESHKOL_MATCH_OP>{},
                eshkol::AstRouteGroup<AstRoute::Cond,
                    ESHKOL_COND_OP, ESHKOL_CASE_OP, ESHKOL_WHEN_OP, ESHKOL_UNLESS_OP,
                    ESHKOL_DO_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Set, ESHKOL_SET_OP>{},
                eshkol::AstRouteGroup<AstRoute::Guard, ESHKOL_GUARD_OP>{},
                eshkol::AstRouteGroup<AstRoute::Raise, ESHKOL_RAISE_OP>{},
                eshkol::AstRouteGroup<AstRoute::Values, ESHKOL_VALUES_OP>{},
                eshkol::AstRouteGroup<AstRoute::CallCc, ESHKOL_CALL_CC_OP>{},
                eshkol::AstRouteGroup<AstRoute::DynamicWind, ESHKOL_DYNAMIC_WIND_OP>{},
                eshkol::AstRouteGroup<AstRoute::The, ESHKOL_THE_OP>{},
                eshkol::AstRouteGroup<AstRoute::Compose, ESHKOL_COMPOSE_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Tensor, ESHKOL_TENSOR_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Diff, ESHKOL_DIFF_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Derivative, ESHKOL_DERIVATIVE_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Gradient, ESHKOL_GRADIENT_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Jacobian, ESHKOL_JACOBIAN_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Hessian, ESHKOL_HESSIAN_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Divergence, ESHKOL_DIVERGENCE_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Curl, ESHKOL_CURL_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Laplacian, ESHKOL_LAPLACIAN_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::DirectionalDeriv, ESHKOL_DIRECTIONAL_DERIV_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Taylor, ESHKOL_TAYLOR_OP, ESHKOL_DERIVATIVE_N_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::WithRegion, ESHKOL_WITH_REGION_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Owned, ESHKOL_OWNED_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Move, ESHKOL_MOVE_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Shared, ESHKOL_SHARED_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::WeakRef, ESHKOL_WEAK_REF_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Borrow, ESHKOL_BORROW_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::CallWithValues, ESHKOL_CALL_WITH_VALUES_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::LetValues, ESHKOL_LET_VALUES_OP,
                    ESHKOL_LET_STAR_VALUES_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::CaseLambda, ESHKOL_CASE_LAMBDA_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Parameterize, ESHKOL_PARAMETERIZE_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::CallPayload, ESHKOL_UNIFY_OP, ESHKOL_MAKE_SUBST_OP,
                    ESHKOL_WALK_OP, ESHKOL_MAKE_FACT_OP, ESHKOL_MAKE_KB_OP, ESHKOL_KB_ASSERT_OP,
                    ESHKOL_KB_QUERY_OP, ESHKOL_MAKE_FACTOR_GRAPH_OP, ESHKOL_FG_ADD_FACTOR_OP,
                    ESHKOL_FG_INFER_OP, ESHKOL_FREE_ENERGY_OP, ESHKOL_EXPECTED_FREE_ENERGY_OP,
                    ESHKOL_MAKE_WORKSPACE_OP, ESHKOL_WS_REGISTER_OP, ESHKOL_WS_STEP_OP,
                    ESHKOL_FG_UPDATE_CPT_OP, ESHKOL_FG_OBSERVE_OP, ESHKOL_LOGIC_VAR_PRED_OP,
                    ESHKOL_SUBSTITUTION_PRED_OP, ESHKOL_KB_PRED_OP, ESHKOL_FACT_PRED_OP,
                    ESHKOL_FACTOR_GRAPH_PRED_OP, ESHKOL_WORKSPACE_PRED_OP, ESHKOL_KB_QUERY_PREFIX_OP,
                    ESHKOL_DNC_MAKE_OP, ESHKOL_DNC_CONTENT_ADDR_OP, ESHKOL_DNC_LOC_ADDR_OP,
                    ESHKOL_DNC_READ_OP, ESHKOL_DNC_WRITE_OP, ESHKOL_DNC_ALLOC_WEIGHTS_OP,
                    ESHKOL_DNC_READ_GRAD_OP, ESHKOL_DNC_PRED_OP, ESHKOL_SDNC_PROGRAM_OP,
                    ESHKOL_SDNC_RUN_OP, ESHKOL_SDNC_WEIGHT_GRAD_OP, ESHKOL_SDNC_PARAMS_OP,
                    ESHKOL_SDNC_SET_PARAMS_OP, ESHKOL_SDNC_IMPROVE_OP, ESHKOL_SDNC_PRED_OP
                >{},
                eshkol::AstRouteGroup<AstRoute::Leaf, ESHKOL_INVALID_OP, ESHKOL_ADD_OP,
                    ESHKOL_SUB_OP, ESHKOL_MUL_OP, ESHKOL_DIV_OP, ESHKOL_EXTERN_OP,
                    ESHKOL_EXTERN_VAR_OP, ESHKOL_QUOTE_OP, ESHKOL_DEFINE_TYPE_OP, ESHKOL_IMPORT_OP,
                    ESHKOL_REQUIRE_OP, ESHKOL_PROVIDE_OP, ESHKOL_TYPE_ANNOTATION_OP,
                    ESHKOL_FORALL_OP, ESHKOL_DEFINE_SYNTAX_OP, ESHKOL_LET_SYNTAX_OP,
                    ESHKOL_LETREC_SYNTAX_OP, ESHKOL_LOGIC_VAR_OP, ESHKOL_DEFINE_RECORD_TYPE_OP,
                    ESHKOL_MAKE_PARAMETER_OP, ESHKOL_COND_EXPAND_OP, ESHKOL_INCLUDE_OP,
                    ESHKOL_SYNTAX_ERROR_OP
                >{}
            )) {
            case AstRoute::Call:
            // Descend into quasiquote and unquote/unquote-splicing so macro
            // calls that a template introduced inside an unquote escape get
            // re-expanded (e.g. `(car `(,(+ (add1q 0) 1)))`). Note: QUOTE_OP is
            // deliberately NOT here — quoted forms are literal data and must not
            // be macro-expanded. Quasiquoted sub-lists are built as (list …)
            // calls with literal atoms, so real macro calls only ever appear in
            // unquote regions, which is exactly what we recurse through.



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

            case AstRoute::Sequence:
                if (op->sequence_op.num_expressions > 0 && op->sequence_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->sequence_op.num_expressions];
                    for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                        new_exprs[i] = expandNode(op->sequence_op.expressions[i]);
                    }
                    op->sequence_op.expressions = new_exprs;
                }
                break;

            case AstRoute::Define: {
                // A definition the parser did not lower into a body's
                // letrec* is a top-level definition: a colored name that a
                // template introduced defines the uncolored name
                // (ADR-0026), as definitions from templates always did.
                if (op->define_op.name) {
                    auto bound = value_renames_.find(op->define_op.name);
                    if (bound != value_renames_.end())
                        op->define_op.name = eshkol_ast_string_copy(bound->second);
                    else if (eshkol_syntax_is_colored(op->define_op.name))
                        op->define_op.name = eshkol_ast_string_copy(std::string(
                            op->define_op.name, eshkol_syntax_base_length(op->define_op.name)));
                }
                // (define (f x ...) body): the formals bind in the body. A
                // formal a template introduced gets a fresh name; a source
                // formal keeps its name and shadows any enclosing binding --
                // value or macro keyword -- of its spelling.
                const auto saved = value_renames_;
                if (op->define_op.is_function) {
                    auto bind_formal = [&](char* name) -> char* {
                        if (!name) return name;
                        if (eshkol_syntax_is_colored(name)) {
                            const std::string fresh = freshValueName(name);
                            value_renames_[name] = fresh;
                            return eshkol_ast_string_copy(fresh);
                        }
                        value_renames_[name] = name;   // shadows macros and outer bindings
                        return name;
                    };
                    if (op->define_op.num_params > 0 && op->define_op.parameters) {
                        auto* parameters = new eshkol_ast_t[op->define_op.num_params];
                        for (uint64_t i = 0; i < op->define_op.num_params; ++i) {
                            parameters[i] = copyAst(op->define_op.parameters[i]);
                            if (parameters[i].type == ESHKOL_VAR && parameters[i].variable.id)
                                parameters[i].variable.id = bind_formal(parameters[i].variable.id);
                        }
                        op->define_op.parameters = parameters;
                    }
                    if (op->define_op.rest_param)
                        op->define_op.rest_param = bind_formal(op->define_op.rest_param);
                }
                if (op->define_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = expandNode(*op->define_op.value);
                    op->define_op.value = new_val;
                }
                value_renames_ = saved;
                break;
            }

            case AstRoute::Lambda: {
                const auto saved = value_renames_;
                pushScope();
                const auto count = op->lambda_op.num_params;
                auto* parameters = count ? new eshkol_ast_t[count] : nullptr;
                for (uint64_t i = 0; i < count; ++i) {
                    parameters[i] = copyAst(op->lambda_op.parameters[i]);
                    if (parameters[i].type == ESHKOL_VAR && parameters[i].variable.id) {
                        const std::string name = parameters[i].variable.id;
                        const std::string fresh = freshValueName(name);
                        value_renames_[name] = fresh;
                        parameters[i].variable.id = eshkol_ast_string_copy(fresh);
                    }
                }
                op->lambda_op.parameters = parameters;
                if (op->lambda_op.rest_param) {
                    const std::string name = op->lambda_op.rest_param;
                    const std::string fresh = freshValueName(name);
                    value_renames_[name] = fresh;
                    op->lambda_op.rest_param = eshkol_ast_string_copy(fresh);
                }
                if (op->lambda_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = expandNode(*op->lambda_op.body);
                    op->lambda_op.body = new_body;
                }
                popScope();
                value_renames_ = saved;
                break;
            }

            case AstRoute::Let: {
                const auto saved = value_renames_;
                pushScope();
                auto body_env = saved;
                if (op->let_op.name) {
                    const std::string old_name = op->let_op.name;
                    const std::string fresh = freshValueName(old_name);
                    body_env[old_name] = fresh;
                    op->let_op.name = eshkol_ast_string_copy(fresh);
                }
                const auto count = op->let_op.num_bindings;
                auto* bindings = count ? new eshkol_ast_t[count] : nullptr;
                std::vector<std::pair<std::string, std::string>> names(count);
                for (uint64_t i = 0; i < count; ++i) {
                    bindings[i] = op->let_op.bindings[i];
                    auto& binding = bindings[i];
                    if (binding.type == ESHKOL_CONS && binding.cons_cell.car &&
                        binding.cons_cell.car->type == ESHKOL_VAR && binding.cons_cell.car->variable.id) {
                        names[i].first = binding.cons_cell.car->variable.id;
                        names[i].second = freshValueName(names[i].first);
                        auto* variable = new eshkol_ast_t(copyAst(*binding.cons_cell.car));
                        variable->variable.id = eshkol_ast_string_copy(names[i].second);
                        binding.cons_cell.car = variable;
                        body_env[names[i].first] = names[i].second;
                    }
                }
                const bool recursive = op->op == ESHKOL_LETREC_OP || op->op == ESHKOL_LETREC_STAR_OP;
                auto sequential_env = saved;
                for (uint64_t i = 0; i < count; ++i) {
                    value_renames_ = recursive ? body_env :
                        (op->op == ESHKOL_LET_STAR_OP ? sequential_env : saved);
                    auto& binding = bindings[i];
                    if (binding.type == ESHKOL_CONS && binding.cons_cell.cdr)
                        binding.cons_cell.cdr = new eshkol_ast_t(expandNode(*binding.cons_cell.cdr));
                    if (!names[i].first.empty()) sequential_env[names[i].first] = names[i].second;
                }
                op->let_op.bindings = bindings;
                value_renames_ = body_env;
                if (op->let_op.body)
                    op->let_op.body = new eshkol_ast_t(expandNode(*op->let_op.body));
                popScope();
                value_renames_ = saved;
                break;
            }

            case AstRoute::Match:
                if (op->match_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = expandNode(*op->match_op.expr);
                    op->match_op.expr = new_expr;
                }
                if (op->match_op.num_clauses > 0 && op->match_op.clauses) {
                    const auto saved = value_renames_;
                    auto* clauses = new eshkol_match_clause_t[op->match_op.num_clauses];
                    for (uint64_t i = 0; i < op->match_op.num_clauses; i++) {
                        value_renames_ = saved;
                        clauses[i] = op->match_op.clauses[i];
                        std::map<std::string, std::string> pattern_names;
                        auto bind_name = [&](const char* name) {
                            auto& fresh = pattern_names[name];
                            if (fresh.empty()) fresh = freshValueName(name);
                            value_renames_[name] = fresh;
                            return eshkol_ast_string_copy(fresh);
                        };
                        std::function<eshkol_pattern_t*(const eshkol_pattern_t*)> rename_pattern;
                        rename_pattern = [&](const eshkol_pattern_t* pattern) -> eshkol_pattern_t* {
                            if (!pattern) return nullptr;
                            auto* renamed = new eshkol_pattern_t(*pattern);
                            if (pattern->type == PATTERN_VARIABLE && pattern->variable.name)
                                renamed->variable.name = bind_name(pattern->variable.name);
                            else if (pattern->type == PATTERN_CONS) {
                                renamed->cons.car_pattern = rename_pattern(pattern->cons.car_pattern);
                                renamed->cons.cdr_pattern = rename_pattern(pattern->cons.cdr_pattern);
                            } else if (pattern->type == PATTERN_LIST || pattern->type == PATTERN_OR) {
                                const auto count = pattern->type == PATTERN_LIST ? pattern->list.num_patterns : pattern->or_pat.num_patterns;
                                auto** original = pattern->type == PATTERN_LIST ? pattern->list.patterns : pattern->or_pat.patterns;
                                auto** children = count ? new eshkol_pattern_t*[count] : nullptr;
                                for (uint64_t j = 0; j < count; ++j) children[j] = rename_pattern(original[j]);
                                if (pattern->type == PATTERN_LIST) renamed->list.patterns = children;
                                else renamed->or_pat.patterns = children;
                            } else if (pattern->type == PATTERN_PREDICATE) {
                                const auto bindings = value_renames_;
                                value_renames_ = saved;
                                if (pattern->predicate.predicate)
                                    renamed->predicate.predicate = new eshkol_ast_t(expandNode(*pattern->predicate.predicate));
                                value_renames_ = bindings;
                                if (pattern->predicate.binding_name)
                                    renamed->predicate.binding_name = bind_name(pattern->predicate.binding_name);
                            }
                            return renamed;
                        };
                        clauses[i].pattern = rename_pattern(clauses[i].pattern);
                        if (clauses[i].body) clauses[i].body = new eshkol_ast_t(expandNode(*clauses[i].body));
                    }
                    op->match_op.clauses = clauses;
                    value_renames_ = saved;
                }
                break;

            // Ops that reuse call_op struct layout
            case AstRoute::Cond:
                if (op->op == ESHKOL_CASE_OP) {
                    // A clause is (datums . body): the datums are quoted data
                    // and `else` is the parser's clause marker, not a
                    // reference, so only the key and the bodies are code.
                    if (op->call_op.func)
                        op->call_op.func = new eshkol_ast_t(expandNode(*op->call_op.func));
                    if (op->call_op.num_vars > 0 && op->call_op.variables) {
                        auto* clauses = new eshkol_ast_t[op->call_op.num_vars];
                        for (uint64_t i = 0; i < op->call_op.num_vars; ++i) {
                            clauses[i] = op->call_op.variables[i];
                            if (clauses[i].type == ESHKOL_CONS && clauses[i].cons_cell.cdr)
                                clauses[i].cons_cell.cdr =
                                    new eshkol_ast_t(expandNode(*clauses[i].cons_cell.cdr));
                            else if (clauses[i].type != ESHKOL_CONS)
                                clauses[i] = expandNode(clauses[i]);
                        }
                        op->call_op.variables = clauses;
                    }
                    break;
                }
                if (op->op == ESHKOL_DO_OP && op->call_op.func &&
                    op->call_op.func->type == ESHKOL_CONS &&
                    op->call_op.func->cons_cell.car &&
                    op->call_op.func->cons_cell.car->type == ESHKOL_OP &&
                    op->call_op.func->cons_cell.car->operation.op == ESHKOL_CALL_OP) {
                    // `do` stores bindings and its test clause in a CONS
                    // scaffold.  Binding initializers see the outer scope;
                    // steps, test/results, and body see all loop variables.
                    const auto saved = value_renames_;
                    pushScope();
                    auto* main = op->call_op.func;
                    auto* binding_list = main->cons_cell.car;
                    const uint64_t n = binding_list->operation.call_op.num_vars;
                    auto* new_bindings = n ? new eshkol_ast_t[n] : nullptr;
                    std::vector<std::pair<std::string, std::string>> names(n);
                    for (uint64_t i = 0; i < n; ++i) {
                        const auto& old_binding = binding_list->operation.call_op.variables[i];
                        new_bindings[i] = old_binding;
                        if (old_binding.type == ESHKOL_CONS && old_binding.cons_cell.cdr) {
                            new_bindings[i].cons_cell.cdr =
                                new eshkol_ast_t(*old_binding.cons_cell.cdr);
                        }
                        if (old_binding.type != ESHKOL_CONS || !old_binding.cons_cell.car ||
                            old_binding.cons_cell.car->type != ESHKOL_VAR ||
                            !old_binding.cons_cell.car->variable.id) continue;
                        names[i].first = old_binding.cons_cell.car->variable.id;
                        names[i].second = freshValueName(names[i].first);
                        auto* var = new eshkol_ast_t(copyAst(*old_binding.cons_cell.car));
                        var->variable.id = eshkol_ast_string_copy(names[i].second);
                        new_bindings[i].cons_cell.car = var;
                        value_renames_[names[i].first] = names[i].second;
                    }
                    for (uint64_t i = 0; i < n; ++i) {
                        auto& b = new_bindings[i];
                        if (b.type != ESHKOL_CONS || !b.cons_cell.cdr ||
                            b.cons_cell.cdr->type != ESHKOL_CONS) continue;
                        auto* init_step = b.cons_cell.cdr;
                        const auto& original = binding_list->operation.call_op.variables[i];
                        value_renames_ = saved;
                        init_step->cons_cell.car = original.cons_cell.cdr &&
                            original.cons_cell.cdr->type == ESHKOL_CONS &&
                            original.cons_cell.cdr->cons_cell.car
                            ? new eshkol_ast_t(expandNode(*original.cons_cell.cdr->cons_cell.car))
                            : nullptr;
                        // Steps execute in the loop-variable environment.
                        for (const auto& item : names)
                            if (!item.first.empty()) value_renames_[item.first] = item.second;
                        if (original.cons_cell.cdr && original.cons_cell.cdr->type == ESHKOL_CONS &&
                            original.cons_cell.cdr->cons_cell.cdr)
                            init_step->cons_cell.cdr = new eshkol_ast_t(
                                expandNode(*original.cons_cell.cdr->cons_cell.cdr));
                    }
                    value_renames_ = saved;
                    for (const auto& item : names) if (!item.first.empty()) value_renames_[item.first] = item.second;
                    binding_list->operation.call_op.variables = new_bindings;
                    if (main->cons_cell.cdr && main->cons_cell.cdr->type == ESHKOL_CONS) {
                        auto* test_clause = main->cons_cell.cdr;
                        if (test_clause->cons_cell.car)
                            test_clause->cons_cell.car = new eshkol_ast_t(expandNode(*test_clause->cons_cell.car));
                        if (test_clause->cons_cell.cdr && test_clause->cons_cell.cdr->type == ESHKOL_OP) {
                            auto* results = test_clause->cons_cell.cdr;
                            for (uint64_t i = 0; i < results->operation.call_op.num_vars; ++i)
                                results->operation.call_op.variables[i] = expandNode(results->operation.call_op.variables[i]);
                        }
                    }
                    for (uint64_t i = 0; i < op->call_op.num_vars; ++i)
                        op->call_op.variables[i] = expandNode(op->call_op.variables[i]);
                    popScope();
                    value_renames_ = saved;
                    break;
                }
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

            case AstRoute::Set:
                if (op->set_op.name) {
                    const std::string target = resolveValue(op->set_op.name);
                    if (target != op->set_op.name)
                        op->set_op.name = eshkol_ast_string_copy(target);
                }
                if (op->set_op.value) {
                    eshkol_ast_t* new_val = new eshkol_ast_t;
                    *new_val = expandNode(*op->set_op.value);
                    op->set_op.value = new_val;
                }
                break;

            case AstRoute::Guard: {
                const auto saved = value_renames_;
                if (op->guard_op.var_name) {
                    const std::string name = op->guard_op.var_name;
                    const std::string fresh = freshValueName(name);
                    value_renames_[name] = fresh;
                    op->guard_op.var_name = eshkol_ast_string_copy(fresh);
                }
                if (op->guard_op.num_clauses > 0 && op->guard_op.clauses) {
                    eshkol_ast_t* new_clauses = new eshkol_ast_t[op->guard_op.num_clauses];
                    for (uint64_t i = 0; i < op->guard_op.num_clauses; i++) {
                        new_clauses[i] = expandNode(op->guard_op.clauses[i]);
                    }
                    op->guard_op.clauses = new_clauses;
                }
                value_renames_ = saved;
                if (op->guard_op.num_body_exprs > 0 && op->guard_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t[op->guard_op.num_body_exprs];
                    for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                        new_body[i] = expandNode(op->guard_op.body[i]);
                    }
                    op->guard_op.body = new_body;
                }
                break;
            }

            case AstRoute::Raise:
                if (op->raise_op.exception) {
                    eshkol_ast_t* new_exc = new eshkol_ast_t;
                    *new_exc = expandNode(*op->raise_op.exception);
                    op->raise_op.exception = new_exc;
                }
                break;

            case AstRoute::Values:
                if (op->values_op.num_values > 0 && op->values_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->values_op.num_values];
                    for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                        new_exprs[i] = expandNode(op->values_op.expressions[i]);
                    }
                    op->values_op.expressions = new_exprs;
                }
                break;

            case AstRoute::CallCc:
                if (op->call_cc_op.proc) {
                    eshkol_ast_t* new_proc = new eshkol_ast_t;
                    *new_proc = expandNode(*op->call_cc_op.proc);
                    op->call_cc_op.proc = new_proc;
                }
                break;

            case AstRoute::DynamicWind:
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

            case AstRoute::The:
                // Expand macros inside the wrapped expression of a (the T e)
                // ascription; the type expression carries no macro calls.
                if (op->the_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = expandNode(*op->the_op.expr);
                    op->the_op.expr = new_expr;
                }
                break;

            case AstRoute::Compose:
                expand_ptr(op->compose_op.func_a); expand_ptr(op->compose_op.func_b); break;
            case AstRoute::Tensor:
                expand_array(op->tensor_op.elements, op->tensor_op.total_elements); break;
            case AstRoute::Diff:
                if (op->diff_op.variable) {
                    const std::string variable = resolveValue(op->diff_op.variable);
                    if (variable != op->diff_op.variable)
                        op->diff_op.variable = eshkol_ast_string_copy(variable);
                }
                expand_ptr(op->diff_op.expression); break;
            case AstRoute::Derivative:
                expand_ptr(op->derivative_op.function); expand_ptr(op->derivative_op.point); break;
            case AstRoute::Gradient:
                expand_ptr(op->gradient_op.function); expand_ptr(op->gradient_op.point); break;
            case AstRoute::Jacobian:
                expand_ptr(op->jacobian_op.function); expand_ptr(op->jacobian_op.point); break;
            case AstRoute::Hessian:
                expand_ptr(op->hessian_op.function); expand_ptr(op->hessian_op.point); break;
            case AstRoute::Divergence:
                expand_ptr(op->divergence_op.function); expand_ptr(op->divergence_op.point); break;
            case AstRoute::Curl:
                expand_ptr(op->curl_op.function); expand_ptr(op->curl_op.point); break;
            case AstRoute::Laplacian:
                expand_ptr(op->laplacian_op.function); expand_ptr(op->laplacian_op.point); break;
            case AstRoute::DirectionalDeriv:
                expand_ptr(op->directional_deriv_op.function);
                expand_ptr(op->directional_deriv_op.point);
                expand_ptr(op->directional_deriv_op.direction); break;
            case AstRoute::Taylor:
                expand_ptr(op->taylor_op.function); expand_ptr(op->taylor_op.point);
                expand_ptr(op->taylor_op.order); break;
            case AstRoute::WithRegion:
                expand_array(op->with_region_op.body, op->with_region_op.num_body_exprs); break;
            case AstRoute::Owned: expand_ptr(op->owned_op.value); break;
            case AstRoute::Move: expand_ptr(op->move_op.value); break;
            case AstRoute::Shared: expand_ptr(op->shared_op.value); break;
            case AstRoute::WeakRef: expand_ptr(op->weak_ref_op.value); break;
            case AstRoute::Borrow:
                expand_ptr(op->borrow_op.value);
                expand_array(op->borrow_op.body, op->borrow_op.num_body_exprs); break;
            case AstRoute::CallWithValues:
                expand_ptr(op->call_with_values_op.producer);
                expand_ptr(op->call_with_values_op.consumer); break;
            case AstRoute::LetValues: {
                const auto saved = value_renames_;
                auto body_env = saved;
                const bool sequential = op->op == ESHKOL_LET_STAR_VALUES_OP;
                auto* producers = op->let_values_op.num_bindings ?
                    new eshkol_ast_t[op->let_values_op.num_bindings] : nullptr;
                auto*** variables = op->let_values_op.num_bindings ?
                    new char**[op->let_values_op.num_bindings] : nullptr;
                for (uint64_t i = 0; i < op->let_values_op.num_bindings; ++i) {
                    value_renames_ = sequential ? body_env : saved;
                    producers[i] = expandNode(op->let_values_op.producers[i]);
                    variables[i] = new char*[op->let_values_op.binding_var_counts[i]];
                    for (uint64_t j = 0; j < op->let_values_op.binding_var_counts[i]; ++j) {
                        const char* old = op->let_values_op.binding_vars[i][j];
                        const std::string fresh = freshValueName(old);
                        variables[i][j] = eshkol_ast_string_copy(fresh);
                        body_env[old] = fresh;
                    }
                }
                op->let_values_op.producers = producers;
                op->let_values_op.binding_vars = variables;
                value_renames_ = body_env;
                op->let_values_op.body = op->let_values_op.body ?
                    new eshkol_ast_t(expandNode(*op->let_values_op.body)) : nullptr;
                value_renames_ = saved;
                break;
            }
            case AstRoute::CaseLambda:
                expand_array(op->case_lambda_op.clauses, op->case_lambda_op.num_clauses); break;
            case AstRoute::Parameterize:
                expand_array(op->parameterize_op.params, op->parameterize_op.num_bindings);
                expand_array(op->parameterize_op.values, op->parameterize_op.num_bindings);
                expand_ptr(op->parameterize_op.body); break;
            case AstRoute::CallPayload:
                // Neuro-symbolic, DNC and SDNC operations carry their operands
                // in the generic call_op payload despite their distinct tags.
                expand_array(op->call_op.variables, op->call_op.num_vars); break;
            case AstRoute::Leaf:
                // No macro-expandable operand: literal data (quote), syntax
                // definitions (already registered), declarations and
                // directives, and forms whose operands are not expressions.
                break;
        }
        }
    }

    return result;
}

/** A fresh unique spelling for a binder written @p name (colors dropped). */
std::string MacroExpander::freshValueName(const std::string& name) {
    return "_v" + std::to_string(rename_counter_++) + "." +
           name.substr(0, eshkol_syntax_base_length(name.c_str()));
}

/**
 * @brief What a value reference spelled @p name denotes at this point.
 *
 * A binder in scope wins. A colored identifier no binder of its expansion
 * binds is free in its template and denotes what its spelling denoted where
 * the macro was defined (ADR-0026).
 */
std::string MacroExpander::resolveValue(const std::string& name) const {
    auto bound = value_renames_.find(name);
    if (bound != value_renames_.end()) return bound->second;
    if (eshkol_syntax_is_colored(name.c_str())) return resolveFree(name);
    return name;
}

std::string MacroExpander::resolveFree(const std::string& name) const {
    size_t prefix_length = 0;
    const unsigned color = eshkol_syntax_last_color(name.c_str(), &prefix_length);
    const std::string spelled(name, 0, prefix_length);
    auto producer = color_macro_.find(color);
    if (producer != color_macro_.end()) {
        auto definition = definition_bindings_.find(producer->second);
        if (definition != definition_bindings_.end()) {
            auto local = definition->second.value_env.find(spelled);
            if (local != definition->second.value_env.end()) return local->second;
        }
    }
    // Not bound where the macro was defined either: peel the next color
    // (a macro-defining macro), or it is the top-level binding.
    if (eshkol_syntax_is_colored(spelled.c_str())) return resolveFree(spelled);
    return spelled;
}

std::string MacroExpander::keywordAlias(const MacroBinding* binding, const std::string& name) {
    if (!binding) return {};
    if (binding->value_env.count(name)) return {};     // a value binding shadows it
    for (auto scope = binding->macro_env.rbegin(); scope != binding->macro_env.rend(); ++scope) {
        auto found = scope->find(name);
        if (found == scope->end()) continue;
        auto& alias = macro_alias_names_[found->second];
        if (alias.empty()) {
            alias = "__eshkol_macro_binding_" + std::to_string(rename_counter_++);
            macro_aliases_[alias] = found->second;
        }
        return alias;
    }
    return {};
}

std::set<std::string> MacroExpander::visibleMacroNames() const {
    std::set<std::string> names;
    for (const auto& scope : scope_stack_)
        for (const auto& item : scope) names.insert(item.first);
    for (const auto& alias : macro_aliases_) names.insert(alias.first);
    return names;
}

/**
 * @brief Expands one macro use (ADR-0026).
 *
 * The use's reader syntax (recorded by the parser) is matched against the
 * transformer's rules; the first matching template is instantiated with a
 * fresh color and parsed as ordinary Eshkol syntax. Macro keywords the
 * template names are resolved in the definition environment and emitted as
 * aliases the parser recognises.
 *
 * @return The parsed expansion, or @p call unchanged when no rule matches or
 * the template is malformed (both reported as errors).
 */
eshkol_ast_t MacroExpander::tryExpandMacroCall(const eshkol_ast_t& call) {
    const std::string macro_name = call.operation.call_op.func->variable.id;
    eshkol_macro_def_t* macro = lookupMacro(macro_name);
    if (!macro) return call;
    auto transformer = eshkol::syntax_macro(macro);
    auto use = eshkol::syntax_use(call.node_id);
    if (!transformer || !use) {
        eshkol_error("macro '%s' is used where its syntax was not recorded",
                     macro_name.c_str());
        return call;
    }

    const auto definition = definition_bindings_.find(macro);
    const MacroBinding* binding =
        definition == definition_bindings_.end() ? nullptr : &definition->second;
    const unsigned color = ++color_counter_;
    color_macro_[color] = macro;

    SyntaxDatum expansion;
    std::string error;
    const auto outcome = eshkol::syntax_rules_apply(
        *transformer, *use, color,
        [&](const std::string& name) { return keywordAlias(binding, name); },
        expansion, error);
    if (outcome == eshkol::SyntaxRulesOutcome::NoMatch) {
        eshkol_error("syntax error: no matching pattern for macro '%s'", macro_name.c_str());
        return call;
    }
    if (outcome == eshkol::SyntaxRulesOutcome::Error) {
        eshkol_error("macro '%s': %s", macro_name.c_str(), error.c_str());
        return call;
    }
    eshkol_ast_t parsed = eshkol::parse_syntax_datum(expansion, visibleMacroNames());
    if (parsed.type == ESHKOL_INVALID) {
        eshkol_error("macro '%s' expanded to syntax that does not parse", macro_name.c_str());
        return call;
    }
    return parsed;
}

eshkol_ast_t MacroExpander::reparseAsCall(const eshkol_ast_t& call) {
    auto use = eshkol::syntax_use(call.node_id);
    if (!use) return call;
    std::set<std::string> names = visibleMacroNames();
    names.erase(call.operation.call_op.func->variable.id);
    eshkol_ast_t parsed = eshkol::parse_syntax_datum(*use, names);
    return parsed.type == ESHKOL_INVALID ? call : parsed;
}

eshkol_ast_t MacroExpander::expandQuasiquoted(const eshkol_ast_t& ast, unsigned depth) {
    eshkol_ast_t result = copyAst(ast);
    if (ast.type == ESHKOL_CONS) {
        if (ast.cons_cell.car) result.cons_cell.car = new eshkol_ast_t(expandQuasiquoted(*ast.cons_cell.car, depth));
        if (ast.cons_cell.cdr) result.cons_cell.cdr = new eshkol_ast_t(expandQuasiquoted(*ast.cons_cell.cdr, depth));
    } else if (ast.type == ESHKOL_OP) {
        auto* op = &result.operation;
        const bool escape = op->op == ESHKOL_UNQUOTE_OP || op->op == ESHKOL_UNQUOTE_SPLICING_OP;
        if (op->op == ESHKOL_TENSOR_OP) {
            auto* elements = new eshkol_ast_t[op->tensor_op.total_elements];
            for (uint64_t i = 0; i < op->tensor_op.total_elements; ++i)
                elements[i] = expandQuasiquoted(op->tensor_op.elements[i], depth);
            op->tensor_op.elements = elements;
        } else if (op->op == ESHKOL_CALL_OP || op->op == ESHKOL_QUOTE_OP ||
                   op->op == ESHKOL_QUASIQUOTE_OP || escape) {
            const unsigned child_depth = op->op == ESHKOL_QUASIQUOTE_OP ? depth + 1 :
                (escape && depth ? depth - 1 : depth);
            auto* arguments = op->call_op.num_vars ? new eshkol_ast_t[op->call_op.num_vars] : nullptr;
            for (uint64_t i = 0; i < op->call_op.num_vars; ++i)
                arguments[i] = escape && depth == 1 ? expandNode(op->call_op.variables[i]) :
                    expandQuasiquoted(op->call_op.variables[i], child_depth);
            op->call_op.variables = arguments;
            if (op->call_op.func) op->call_op.func = new eshkol_ast_t(copyAst(*op->call_op.func));
        }
    }
    return result;
}

eshkol_ast_t MacroExpander::copyAst(const eshkol_ast_t& ast) {
    eshkol_ast_t result = ast;

    // Deep copy strings and nested pointers
    switch (ast.type) {
        case ESHKOL_STRING:
            if (ast.str_val.ptr) {
                // strndup terminates the copy even when a producer's size
                // excludes the NUL.
                result.str_val.ptr = eshkol_ast_strndup(ast.str_val.ptr, ast.str_val.size);
                result.str_val.size = ast.str_val.size;
            }
            break;

        case ESHKOL_VAR:
            if (ast.variable.id) {
                result.variable.id = eshkol_ast_strdup(ast.variable.id);
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

} // namespace eshkol
