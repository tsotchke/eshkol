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
#include <eshkol/frontend/shadowable_ops.h>
#include <eshkol/frontend/syntax_color.h>
#include <eshkol/frontend/syntax_datum.h>
#include <eshkol/frontend/syntax_rules.h>
#include <eshkol/logger.h>
#include <cstring>
#include <algorithm>
#include <functional>

namespace eshkol {

namespace {
// Enclosing expansions that expanded a template-introduced macro use (see
// expandNodeTask).
thread_local unsigned macro_nesting_depth = 0;
constexpr unsigned kMaxMacroNestingDepth = 1000;
}  // namespace

void MacroExpander::RenameEnv::bind(const std::string& name, const std::string& renamed) {
    auto found = map_.find(name);
    if (found == map_.end()) {
        log_.push_back({name, false, {}});
        map_.emplace(name, renamed);
    } else {
        log_.push_back({name, true, found->second});
        found->second = renamed;
    }
}

void MacroExpander::RenameEnv::erase(const std::string& name) {
    auto found = map_.find(name);
    if (found == map_.end()) return;
    log_.push_back({name, true, found->second});
    map_.erase(found);
}

void MacroExpander::RenameEnv::restore(size_t mark) {
    while (log_.size() > mark) {
        Undo& undo = log_.back();
        if (undo.had_binding) map_[undo.name] = std::move(undo.previous);
        else map_.erase(undo.name);
        log_.pop_back();
    }
}

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
        binding.value_env = value_renames_.bindings();
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
    binding.value_env = value_renames_.bindings();
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
        result.push_back(expandToplevelForm(ast));
    }

    return result;
}

eshkol_ast_t MacroExpander::expandToplevelForm(const eshkol_ast_t& ast) {
    toplevel_form_ = true;
    return expandNode(ast);
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
 * @brief Synchronous entry to expandNodeTask(): runs the expansion of @p ast
 * and every form under it on the explicit continuation stack, so the native
 * stack stays flat however deeply the source nests.
 */
eshkol_ast_t MacroExpander::expandNode(const eshkol_ast_t& ast) {
    return expandNodeTask(ast).run();
}

/** Expands the form @p child points to (if any) in place. */
ContinuationTask<bool> MacroExpander::expandChildTask(eshkol_ast_t*& child) {
    if (child) child = new eshkol_ast_t(co_await expandNodeTask(*child));
    co_return true;
}

/** Expands each of the @p count forms of @p items (if any) into a new array. */
ContinuationTask<bool> MacroExpander::expandArrayTask(eshkol_ast_t*& items, uint64_t count) {
    if (!items) co_return true;
    auto* fresh = new eshkol_ast_t[count];
    for (uint64_t i = 0; i < count; ++i) fresh[i] = co_await expandNodeTask(items[i]);
    items = fresh;
    co_return true;
}

/**
 * @brief Core macro-expansion driver: repeatedly expands macro calls at the
 * current node, then descends into sub-expressions. It is a ContinuationTask:
 * each descent suspends on the explicit continuation stack instead of
 * recursing on the native stack.
 *
 * A macro call is expanded iteratively (via a `for (;;)` loop) rather than by
 * recursive self-call, so a macro that expands into another macro call does
 * not grow the C++ call stack; a per-expansion-chain @c expansion_chain set
 * detects a macro expanding back into itself and reports a circular-expansion
 * error instead of looping forever. A macro whose expansion places a further
 * use of a macro inside a sub-form (rather than at the head) grows the tree
 * one level per step; the thread-local @c macro_nesting_depth counts only the
 * enclosing expansions of macro uses that an earlier template introduced, and
 * caps that chain at @c kMaxMacroNestingDepth. Source structure (calls, lets,
 * lambdas, and macro uses written in the source) does not count toward it: its
 * depth is bounded by the source text, and every level of it is traversed so
 * that each binder and each of its references receive the same fresh name
 * however deeply the program nests.
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
ContinuationTask<eshkol_ast_t> MacroExpander::expandNodeTask(eshkol_ast_t ast) {
    // Whether this node is a top-level form; nothing nested inherits it.
    const bool toplevel = toplevel_form_;
    toplevel_form_ = false;
    if (ast.type == ESHKOL_OP && ast.operation.op == ESHKOL_QUASIQUOTE_OP)
        co_return co_await expandQuasiquotedTask(ast, 0);
    // Use iterative re-expansion for macro calls to prevent unbounded recursion.
    // A macro expanding to another macro call is handled by looping, not recursing.
    // We track seen macro names per expansion chain to detect cycles.
    //
    // Nesting introduced by macro expansion is the only depth that the source
    // text does not bound, so it is the only depth that is limited: the
    // counter rises once for each enclosing expansion that expanded a macro
    // use introduced by an earlier expansion's template.
    struct MacroNesting {
        bool entered = false;
        void enter() { if (!entered) { entered = true; ++macro_nesting_depth; } }
        ~MacroNesting() { if (entered) --macro_nesting_depth; }
    } macro_nesting;
    // Nodes the expander creates for this form (as opposed to copies of
    // template nodes, which keep the template's own location) are born with
    // the location of the form being expanded.
    EshkolAstBirthLocationScope birth_location(ast.line, ast.column);
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
            co_return null_ast;
        }

        // Handle let-syntax / letrec-syntax: push scope, register macros, expand body, pop scope
        if (current.type == ESHKOL_OP &&
            (current.operation.op == ESHKOL_LET_SYNTAX_OP || current.operation.op == ESHKOL_LETREC_SYNTAX_OP)) {
            const auto* ls = &current.operation.let_syntax_op;
            const size_t outer_values = value_renames_.mark();
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
                    item.second.value_env = value_renames_.bindings();
                    definition_bindings_[item.second.macro] = item.second;
                }
            }
            eshkol_ast_t expanded_body = co_await expandNodeTask(*ls->body);
            popScope();
            value_renames_.restore(outer_values);
            co_return expanded_body;
        }

        // Check for macro call — if found, expand and LOOP (not recurse)
        if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_CALL_OP) {
            const auto* call = &current.operation.call_op;
            if (call->func && call->func->type == ESHKOL_VAR && call->func->variable.id) {
                std::string func_name = call->func->variable.id;
                if (isMacro(func_name)) {
                    // A keyword an expansion's template introduced carries
                    // that expansion's color (syntax_color.h) or is the alias
                    // keywordAlias() gave it; a keyword written in the source
                    // is neither. Only the former nests beyond what the source
                    // text bounds.
                    if (eshkol_syntax_is_colored(func_name.c_str()) ||
                        macro_aliases_.count(func_name)) {
                        if (!macro_nesting.entered && macro_nesting_depth >= kMaxMacroNestingDepth) {
                            eshkol_error("macro expansion depth limit exceeded (>%u)",
                                         kMaxMacroNestingDepth);
                            co_return current;
                        }
                        macro_nesting.enter();
                    }
                    if (++expansion_steps > kMaxExpansionSteps) {
                        eshkol_error("macro expansion of '%s' did not terminate after %u steps",
                                     func_name.c_str(), kMaxExpansionSteps);
                        co_return current;
                    }
                    eshkol_ast_t expanded;
                    if (toplevel) {
                        eshkol::ToplevelFormParseScope toplevel_parse;
                        expanded = tryExpandMacroCall(current);
                    } else {
                        expanded = tryExpandMacroCall(current);
                    }
                    if (expanded.node_id == current.node_id && expanded.type == current.type &&
                        expanded.type == ESHKOL_OP && expanded.operation.op == ESHKOL_CALL_OP &&
                        expanded.operation.call_op.func == current.operation.call_op.func)
                        co_return current;          // no rule matched; already reported
                    current = expanded;
                    continue; // Re-expand iteratively
                }
                // The parser read this use as macro syntax, but here the
                // keyword is shadowed by a value binding (or was never
                // bound): it is an ordinary call.
                if (eshkol::syntax_use_unparsed(current.node_id)) {
                    if (toplevel) {
                        eshkol::ToplevelFormParseScope toplevel_parse;
                        current = reparseAsCall(current);
                    } else {
                        current = reparseAsCall(current);
                    }
                }
            }

        }

        // Not a macro call — break out to do tree traversal
        break;
    }

    // A top-level begin (a sequence) and a top-level with-region pass their
    // top-level status to their forms (R7RS 5.1), so a macro use among them
    // expands in top-level mode too.
    if (toplevel && current.type == ESHKOL_OP &&
        (current.operation.op == ESHKOL_SEQUENCE_OP ||
         current.operation.op == ESHKOL_WITH_REGION_OP)) {
        eshkol_ast_t result = copyAst(current);
        const bool is_sequence = current.operation.op == ESHKOL_SEQUENCE_OP;
        const uint64_t n = is_sequence ? current.operation.sequence_op.num_expressions
                                       : current.operation.with_region_op.num_body_exprs;
        const eshkol_ast_t* items = is_sequence ? current.operation.sequence_op.expressions
                                                : current.operation.with_region_op.body;
        eshkol_ast_t* fresh = n ? new eshkol_ast_t[n] : nullptr;
        for (uint64_t i = 0; i < n; ++i) {
            toplevel_form_ = true;
            fresh[i] = co_await expandNodeTask(items[i]);
        }
        if (is_sequence) result.operation.sequence_op.expressions = fresh;
        else result.operation.with_region_op.body = fresh;
        co_return result;
    }

    // Recursively expand sub-expressions (tree depth is bounded by input nesting)
    if (current.type == ESHKOL_OP && current.operation.op == ESHKOL_QUASIQUOTE_OP)
        co_return co_await expandQuasiquotedTask(current, 0);
    if (current.type == ESHKOL_VAR && current.variable.id) {
        const std::string resolved = resolveValue(current.variable.id);
        if (resolved != current.variable.id) {
            eshkol_ast_t renamed = copyAst(current);
            renamed.variable.id = eshkol_ast_string_copy(resolved);
            co_return renamed;
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
            ? new eshkol_ast_t(co_await expandNodeTask(*current.cons_cell.car)) : nullptr;
        result.cons_cell.cdr = current.cons_cell.cdr
            ? new eshkol_ast_t(co_await expandNodeTask(*current.cons_cell.cdr)) : nullptr;
        co_return result;
    }
    eshkol_ast_t result = copyAst(current);

    if (result.type == ESHKOL_OP) {
        auto* op = &result.operation;

        // A builtin the parser lowered to its own node, used where a local
        // binder of the same name is in scope, is a call of that binder.
        // Scope is known here and nowhere later: once the binder carries its
        // fresh spelling, no downstream pass can see that it shadows the
        // builtin (shadowable_ops.h). The operands already sit in the call
        // payload; the head becomes the variable, which the Call route below
        // resolves like any other reference.
        {
            const auto& shadowable = eshkol::userShadowableBuiltinOps();
            auto builtin = shadowable.find(op->op);
            if (builtin != shadowable.end() &&
                value_renames_.find(builtin->second) != value_renames_.end()) {
                auto* head = new eshkol_ast_t{};
                head->type = ESHKOL_VAR;
                head->variable.id = eshkol_ast_string_copy(builtin->second);
                head->line = result.line;
                head->column = result.column;
                op->op = ESHKOL_CALL_OP;
                op->call_op.func = head;
            }
        }

        {
            enum class AstRoute {
                Call, Sequence, Define, Lambda, Let, Match,
                Cond, Set, Guard, Raise, Values, CallCc,
                DynamicWind, The, Compose, Tensor, Diff, Derivative, Gradient,
                Jacobian, Hessian, Divergence, Curl, Laplacian, DirectionalDeriv,
                Taylor, WithRegion, Owned, Move, Shared, WeakRef, Borrow,
                CallWithValues, LetValues, CaseLambda, Parameterize, CallPayload, Leaf
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
                    *new_func = co_await expandNodeTask(*op->call_op.func);
                    op->call_op.func = new_func;
                }
                if (op->call_op.num_vars > 0 && op->call_op.variables) {
                    eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                        new_vars[i] = co_await expandNodeTask(op->call_op.variables[i]);
                    }
                    op->call_op.variables = new_vars;
                }
                break;

            case AstRoute::Sequence:
                if (op->sequence_op.num_expressions > 0 && op->sequence_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->sequence_op.num_expressions];
                    for (uint64_t i = 0; i < op->sequence_op.num_expressions; i++) {
                        new_exprs[i] = co_await expandNodeTask(op->sequence_op.expressions[i]);
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
                const size_t saved = value_renames_.mark();
                if (op->define_op.is_function) {
                    auto bind_formal = [&](char* name) -> char* {
                        if (!name) return name;
                        if (eshkol_syntax_is_colored(name)) {
                            const std::string fresh = freshValueName(name);
                            value_renames_.bind(name, fresh);
                            return eshkol_ast_string_copy(fresh);
                        }
                        value_renames_.bind(name, name);   // shadows macros and outer bindings
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
                    *new_val = co_await expandNodeTask(*op->define_op.value);
                    op->define_op.value = new_val;
                }
                value_renames_.restore(saved);
                break;
            }

            case AstRoute::Lambda: {
                const size_t saved = value_renames_.mark();
                pushScope();
                const auto count = op->lambda_op.num_params;
                auto* parameters = count ? new eshkol_ast_t[count] : nullptr;
                for (uint64_t i = 0; i < count; ++i) {
                    parameters[i] = copyAst(op->lambda_op.parameters[i]);
                    if (parameters[i].type == ESHKOL_VAR && parameters[i].variable.id) {
                        const std::string name = parameters[i].variable.id;
                        const std::string fresh = freshValueName(name);
                        value_renames_.bind(name, fresh);
                        parameters[i].variable.id = eshkol_ast_string_copy(fresh);
                    }
                }
                op->lambda_op.parameters = parameters;
                if (op->lambda_op.rest_param) {
                    const std::string name = op->lambda_op.rest_param;
                    const std::string fresh = freshValueName(name);
                    value_renames_.bind(name, fresh);
                    op->lambda_op.rest_param = eshkol_ast_string_copy(fresh);
                }
                if (op->lambda_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t;
                    *new_body = co_await expandNodeTask(*op->lambda_op.body);
                    op->lambda_op.body = new_body;
                }
                popScope();
                value_renames_.restore(saved);
                break;
            }

            case AstRoute::Let: {
                // Initializers see the enclosing scope (let), the bindings
                // before them (let*), or every binding (letrec, letrec*); the
                // body sees every binding and a named let's own name.
                const size_t saved = value_renames_.mark();
                pushScope();
                std::string let_name, let_fresh;
                if (op->let_op.name) {
                    let_name = op->let_op.name;
                    let_fresh = freshValueName(let_name);
                    op->let_op.name = eshkol_ast_string_copy(let_fresh);
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
                    }
                }
                auto bind_body_scope = [&]() {
                    if (!let_name.empty()) value_renames_.bind(let_name, let_fresh);
                    for (const auto& name : names)
                        if (!name.first.empty()) value_renames_.bind(name.first, name.second);
                };
                const bool recursive = op->op == ESHKOL_LETREC_OP || op->op == ESHKOL_LETREC_STAR_OP;
                const bool sequential = op->op == ESHKOL_LET_STAR_OP;
                if (recursive) bind_body_scope();
                for (uint64_t i = 0; i < count; ++i) {
                    auto& binding = bindings[i];
                    if (binding.type == ESHKOL_CONS && binding.cons_cell.cdr)
                        binding.cons_cell.cdr = new eshkol_ast_t(co_await expandNodeTask(*binding.cons_cell.cdr));
                    if (sequential && !names[i].first.empty())
                        value_renames_.bind(names[i].first, names[i].second);
                }
                op->let_op.bindings = bindings;
                if (!recursive) {
                    value_renames_.restore(saved);
                    bind_body_scope();
                }
                if (op->let_op.body)
                    op->let_op.body = new eshkol_ast_t(co_await expandNodeTask(*op->let_op.body));
                popScope();
                value_renames_.restore(saved);
                break;
            }

            case AstRoute::Match:
                if (op->match_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = co_await expandNodeTask(*op->match_op.expr);
                    op->match_op.expr = new_expr;
                }
                if (op->match_op.num_clauses > 0 && op->match_op.clauses) {
                    const size_t saved = value_renames_.mark();
                    auto* clauses = new eshkol_match_clause_t[op->match_op.num_clauses];
                    for (uint64_t i = 0; i < op->match_op.num_clauses; i++) {
                        value_renames_.restore(saved);
                        clauses[i] = op->match_op.clauses[i];
                        std::map<std::string, std::string> pattern_names;
                        auto bind_name = [&](const char* name) {
                            auto& fresh = pattern_names[name];
                            if (fresh.empty()) fresh = freshValueName(name);
                            value_renames_.bind(name, fresh);
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
                                // The predicate sees the scope around the match, not
                                // the clause's pattern variables; those are bound
                                // again afterwards.
                                value_renames_.restore(saved);
                                if (pattern->predicate.predicate)
                                    renamed->predicate.predicate = new eshkol_ast_t(expandNode(*pattern->predicate.predicate));
                                for (const auto& bound : pattern_names)
                                    value_renames_.bind(bound.first, bound.second);
                                if (pattern->predicate.binding_name)
                                    renamed->predicate.binding_name = bind_name(pattern->predicate.binding_name);
                            }
                            return renamed;
                        };
                        clauses[i].pattern = rename_pattern(clauses[i].pattern);
                        if (clauses[i].body) clauses[i].body = new eshkol_ast_t(co_await expandNodeTask(*clauses[i].body));
                    }
                    op->match_op.clauses = clauses;
                    value_renames_.restore(saved);
                }
                break;

            // Ops that reuse call_op struct layout
            case AstRoute::Cond:
                if (op->op == ESHKOL_CASE_OP) {
                    // A clause is (datums . body): the datums are quoted data
                    // and `else` is the parser's clause marker, not a
                    // reference, so only the key and the bodies are code.
                    if (op->call_op.func)
                        op->call_op.func = new eshkol_ast_t(co_await expandNodeTask(*op->call_op.func));
                    if (op->call_op.num_vars > 0 && op->call_op.variables) {
                        auto* clauses = new eshkol_ast_t[op->call_op.num_vars];
                        for (uint64_t i = 0; i < op->call_op.num_vars; ++i) {
                            clauses[i] = op->call_op.variables[i];
                            if (clauses[i].type == ESHKOL_CONS && clauses[i].cons_cell.cdr)
                                clauses[i].cons_cell.cdr =
                                    new eshkol_ast_t(co_await expandNodeTask(*clauses[i].cons_cell.cdr));
                            else if (clauses[i].type != ESHKOL_CONS)
                                clauses[i] = co_await expandNodeTask(clauses[i]);
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
                    const size_t saved = value_renames_.mark();
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
                        value_renames_.bind(names[i].first, names[i].second);
                    }
                    for (uint64_t i = 0; i < n; ++i) {
                        auto& b = new_bindings[i];
                        if (b.type != ESHKOL_CONS || !b.cons_cell.cdr ||
                            b.cons_cell.cdr->type != ESHKOL_CONS) continue;
                        auto* init_step = b.cons_cell.cdr;
                        const auto& original = binding_list->operation.call_op.variables[i];
                        value_renames_.restore(saved);
                        init_step->cons_cell.car = original.cons_cell.cdr &&
                            original.cons_cell.cdr->type == ESHKOL_CONS &&
                            original.cons_cell.cdr->cons_cell.car
                            ? new eshkol_ast_t(co_await expandNodeTask(*original.cons_cell.cdr->cons_cell.car))
                            : nullptr;
                        // Steps execute in the loop-variable environment.
                        for (const auto& item : names)
                            if (!item.first.empty()) value_renames_.bind(item.first, item.second);
                        if (original.cons_cell.cdr && original.cons_cell.cdr->type == ESHKOL_CONS &&
                            original.cons_cell.cdr->cons_cell.cdr)
                            init_step->cons_cell.cdr = new eshkol_ast_t(
                                co_await expandNodeTask(*original.cons_cell.cdr->cons_cell.cdr));
                    }
                    value_renames_.restore(saved);
                    for (const auto& item : names) if (!item.first.empty()) value_renames_.bind(item.first, item.second);
                    binding_list->operation.call_op.variables = new_bindings;
                    if (main->cons_cell.cdr && main->cons_cell.cdr->type == ESHKOL_CONS) {
                        auto* test_clause = main->cons_cell.cdr;
                        if (test_clause->cons_cell.car)
                            test_clause->cons_cell.car = new eshkol_ast_t(co_await expandNodeTask(*test_clause->cons_cell.car));
                        if (test_clause->cons_cell.cdr && test_clause->cons_cell.cdr->type == ESHKOL_OP) {
                            auto* results = test_clause->cons_cell.cdr;
                            for (uint64_t i = 0; i < results->operation.call_op.num_vars; ++i)
                                results->operation.call_op.variables[i] = co_await expandNodeTask(results->operation.call_op.variables[i]);
                        }
                    }
                    for (uint64_t i = 0; i < op->call_op.num_vars; ++i)
                        op->call_op.variables[i] = co_await expandNodeTask(op->call_op.variables[i]);
                    popScope();
                    value_renames_.restore(saved);
                    break;
                }
                if (op->call_op.func) {
                    eshkol_ast_t* new_func = new eshkol_ast_t;
                    *new_func = co_await expandNodeTask(*op->call_op.func);
                    op->call_op.func = new_func;
                }
                if (op->call_op.num_vars > 0 && op->call_op.variables) {
                    eshkol_ast_t* new_vars = new eshkol_ast_t[op->call_op.num_vars];
                    for (uint64_t i = 0; i < op->call_op.num_vars; i++) {
                        new_vars[i] = co_await expandNodeTask(op->call_op.variables[i]);
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
                    *new_val = co_await expandNodeTask(*op->set_op.value);
                    op->set_op.value = new_val;
                }
                break;

            case AstRoute::Guard: {
                const size_t saved = value_renames_.mark();
                if (op->guard_op.var_name) {
                    const std::string name = op->guard_op.var_name;
                    const std::string fresh = freshValueName(name);
                    value_renames_.bind(name, fresh);
                    op->guard_op.var_name = eshkol_ast_string_copy(fresh);
                }
                if (op->guard_op.num_clauses > 0 && op->guard_op.clauses) {
                    eshkol_ast_t* new_clauses = new eshkol_ast_t[op->guard_op.num_clauses];
                    for (uint64_t i = 0; i < op->guard_op.num_clauses; i++) {
                        new_clauses[i] = co_await expandNodeTask(op->guard_op.clauses[i]);
                    }
                    op->guard_op.clauses = new_clauses;
                }
                value_renames_.restore(saved);
                if (op->guard_op.num_body_exprs > 0 && op->guard_op.body) {
                    eshkol_ast_t* new_body = new eshkol_ast_t[op->guard_op.num_body_exprs];
                    for (uint64_t i = 0; i < op->guard_op.num_body_exprs; i++) {
                        new_body[i] = co_await expandNodeTask(op->guard_op.body[i]);
                    }
                    op->guard_op.body = new_body;
                }
                break;
            }

            case AstRoute::Raise:
                if (op->raise_op.exception) {
                    eshkol_ast_t* new_exc = new eshkol_ast_t;
                    *new_exc = co_await expandNodeTask(*op->raise_op.exception);
                    op->raise_op.exception = new_exc;
                }
                break;

            case AstRoute::Values:
                if (op->values_op.num_values > 0 && op->values_op.expressions) {
                    eshkol_ast_t* new_exprs = new eshkol_ast_t[op->values_op.num_values];
                    for (uint64_t i = 0; i < op->values_op.num_values; i++) {
                        new_exprs[i] = co_await expandNodeTask(op->values_op.expressions[i]);
                    }
                    op->values_op.expressions = new_exprs;
                }
                break;

            case AstRoute::CallCc:
                if (op->call_cc_op.proc) {
                    eshkol_ast_t* new_proc = new eshkol_ast_t;
                    *new_proc = co_await expandNodeTask(*op->call_cc_op.proc);
                    op->call_cc_op.proc = new_proc;
                }
                break;

            case AstRoute::DynamicWind:
                if (op->dynamic_wind_op.before) {
                    eshkol_ast_t* new_before = new eshkol_ast_t;
                    *new_before = co_await expandNodeTask(*op->dynamic_wind_op.before);
                    op->dynamic_wind_op.before = new_before;
                }
                if (op->dynamic_wind_op.thunk) {
                    eshkol_ast_t* new_thunk = new eshkol_ast_t;
                    *new_thunk = co_await expandNodeTask(*op->dynamic_wind_op.thunk);
                    op->dynamic_wind_op.thunk = new_thunk;
                }
                if (op->dynamic_wind_op.after) {
                    eshkol_ast_t* new_after = new eshkol_ast_t;
                    *new_after = co_await expandNodeTask(*op->dynamic_wind_op.after);
                    op->dynamic_wind_op.after = new_after;
                }
                break;

            case AstRoute::The:
                // Expand macros inside the wrapped expression of a (the T e)
                // ascription; the type expression carries no macro calls.
                if (op->the_op.expr) {
                    eshkol_ast_t* new_expr = new eshkol_ast_t;
                    *new_expr = co_await expandNodeTask(*op->the_op.expr);
                    op->the_op.expr = new_expr;
                }
                break;

            case AstRoute::Compose:
                co_await expandChildTask(op->compose_op.func_a); co_await expandChildTask(op->compose_op.func_b); break;
            case AstRoute::Tensor:
                co_await expandArrayTask(op->tensor_op.elements, op->tensor_op.total_elements); break;
            case AstRoute::Diff:
                if (op->diff_op.variable) {
                    const std::string variable = resolveValue(op->diff_op.variable);
                    if (variable != op->diff_op.variable)
                        op->diff_op.variable = eshkol_ast_string_copy(variable);
                }
                co_await expandChildTask(op->diff_op.expression); break;
            case AstRoute::Derivative:
                co_await expandChildTask(op->derivative_op.function); co_await expandChildTask(op->derivative_op.point); break;
            case AstRoute::Gradient:
                co_await expandChildTask(op->gradient_op.function); co_await expandChildTask(op->gradient_op.point); break;
            case AstRoute::Jacobian:
                co_await expandChildTask(op->jacobian_op.function); co_await expandChildTask(op->jacobian_op.point); break;
            case AstRoute::Hessian:
                co_await expandChildTask(op->hessian_op.function); co_await expandChildTask(op->hessian_op.point); break;
            case AstRoute::Divergence:
                co_await expandChildTask(op->divergence_op.function); co_await expandChildTask(op->divergence_op.point); break;
            case AstRoute::Curl:
                co_await expandChildTask(op->curl_op.function); co_await expandChildTask(op->curl_op.point); break;
            case AstRoute::Laplacian:
                co_await expandChildTask(op->laplacian_op.function); co_await expandChildTask(op->laplacian_op.point); break;
            case AstRoute::DirectionalDeriv:
                co_await expandChildTask(op->directional_deriv_op.function);
                co_await expandChildTask(op->directional_deriv_op.point);
                co_await expandChildTask(op->directional_deriv_op.direction); break;
            case AstRoute::Taylor:
                co_await expandChildTask(op->taylor_op.function); co_await expandChildTask(op->taylor_op.point);
                co_await expandChildTask(op->taylor_op.order); break;
            case AstRoute::WithRegion:
                co_await expandArrayTask(op->with_region_op.body, op->with_region_op.num_body_exprs); break;
            case AstRoute::Owned: co_await expandChildTask(op->owned_op.value); break;
            case AstRoute::Move: co_await expandChildTask(op->move_op.value); break;
            case AstRoute::Shared: co_await expandChildTask(op->shared_op.value); break;
            case AstRoute::WeakRef: co_await expandChildTask(op->weak_ref_op.value); break;
            case AstRoute::Borrow:
                co_await expandChildTask(op->borrow_op.value);
                co_await expandArrayTask(op->borrow_op.body, op->borrow_op.num_body_exprs); break;
            case AstRoute::CallWithValues:
                co_await expandChildTask(op->call_with_values_op.producer);
                co_await expandChildTask(op->call_with_values_op.consumer); break;
            case AstRoute::LetValues: {
                // Producers see the enclosing scope (let-values) or the
                // formals before them (let*-values); the body sees all.
                const size_t saved = value_renames_.mark();
                const bool sequential = op->op == ESHKOL_LET_STAR_VALUES_OP;
                std::vector<std::pair<std::string, std::string>> formals;
                auto* producers = op->let_values_op.num_bindings ?
                    new eshkol_ast_t[op->let_values_op.num_bindings] : nullptr;
                auto*** variables = op->let_values_op.num_bindings ?
                    new char**[op->let_values_op.num_bindings] : nullptr;
                for (uint64_t i = 0; i < op->let_values_op.num_bindings; ++i) {
                    producers[i] = co_await expandNodeTask(op->let_values_op.producers[i]);
                    variables[i] = new char*[op->let_values_op.binding_var_counts[i]];
                    for (uint64_t j = 0; j < op->let_values_op.binding_var_counts[i]; ++j) {
                        const char* old = op->let_values_op.binding_vars[i][j];
                        const std::string fresh = freshValueName(old);
                        variables[i][j] = eshkol_ast_string_copy(fresh);
                        if (sequential) value_renames_.bind(old, fresh);
                        else formals.emplace_back(old, fresh);
                    }
                }
                op->let_values_op.producers = producers;
                op->let_values_op.binding_vars = variables;
                for (const auto& formal : formals) value_renames_.bind(formal.first, formal.second);
                op->let_values_op.body = op->let_values_op.body ?
                    new eshkol_ast_t(co_await expandNodeTask(*op->let_values_op.body)) : nullptr;
                value_renames_.restore(saved);
                break;
            }
            case AstRoute::CaseLambda:
                co_await expandArrayTask(op->case_lambda_op.clauses, op->case_lambda_op.num_clauses); break;
            case AstRoute::Parameterize:
                co_await expandArrayTask(op->parameterize_op.params, op->parameterize_op.num_bindings);
                co_await expandArrayTask(op->parameterize_op.values, op->parameterize_op.num_bindings);
                co_await expandChildTask(op->parameterize_op.body); break;
            case AstRoute::CallPayload:
                // Neuro-symbolic, DNC and SDNC operations carry their operands
                // in the generic call_op payload despite their distinct tags.
                co_await expandArrayTask(op->call_op.variables, op->call_op.num_vars); break;
            case AstRoute::Leaf:
                // No macro-expandable operand: literal data (quote), syntax
                // definitions (already registered), declarations and
                // directives, and forms whose operands are not expressions.
                break;
        }
        }
    }

    co_return result;
}

/** A fresh unique spelling for a binder written @p name (colors dropped).
 *  The format is owned by syntax_color.h, which also recovers the source
 *  spelling for everything a user reads. */
std::string MacroExpander::freshValueName(const std::string& name) {
    return ESHKOL_SYNTAX_FRESH_PREFIX + std::to_string(rename_counter_++) +
           ESHKOL_SYNTAX_FRESH_SEPARATOR +
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

ContinuationTask<eshkol_ast_t> MacroExpander::expandQuasiquotedTask(eshkol_ast_t ast, unsigned depth) {
    eshkol_ast_t result = copyAst(ast);
    if (ast.type == ESHKOL_CONS) {
        if (ast.cons_cell.car) result.cons_cell.car = new eshkol_ast_t(co_await expandQuasiquotedTask(*ast.cons_cell.car, depth));
        if (ast.cons_cell.cdr) result.cons_cell.cdr = new eshkol_ast_t(co_await expandQuasiquotedTask(*ast.cons_cell.cdr, depth));
    } else if (ast.type == ESHKOL_OP) {
        auto* op = &result.operation;
        const bool escape = op->op == ESHKOL_UNQUOTE_OP || op->op == ESHKOL_UNQUOTE_SPLICING_OP;
        if (op->op == ESHKOL_TENSOR_OP) {
            auto* elements = new eshkol_ast_t[op->tensor_op.total_elements];
            for (uint64_t i = 0; i < op->tensor_op.total_elements; ++i)
                elements[i] = co_await expandQuasiquotedTask(op->tensor_op.elements[i], depth);
            op->tensor_op.elements = elements;
        } else if (op->op == ESHKOL_CALL_OP || op->op == ESHKOL_QUOTE_OP ||
                   op->op == ESHKOL_QUASIQUOTE_OP || escape) {
            const unsigned child_depth = op->op == ESHKOL_QUASIQUOTE_OP ? depth + 1 :
                (escape && depth ? depth - 1 : depth);
            auto* arguments = op->call_op.num_vars ? new eshkol_ast_t[op->call_op.num_vars] : nullptr;
            for (uint64_t i = 0; i < op->call_op.num_vars; ++i)
                arguments[i] = escape && depth == 1
                    ? co_await expandNodeTask(op->call_op.variables[i])
                    : co_await expandQuasiquotedTask(op->call_op.variables[i], child_depth);
            op->call_op.variables = arguments;
            if (op->call_op.func) op->call_op.func = new eshkol_ast_t(copyAst(*op->call_op.func));
        }
    }
    co_return result;
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
