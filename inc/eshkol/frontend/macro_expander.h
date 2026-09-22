/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MacroExpander - Hygienic macro expansion for Eshkol (ADR-0026)
 *
 * This module handles:
 * - Registering macro definitions (define-syntax, let-syntax, letrec-syntax)
 *   with their definition environments
 * - Recognizing macro uses and expanding them: the use's reader syntax
 *   (syntax_datum.h) is rewritten by the one syntax-rules engine
 *   (syntax_rules_core.h) and the expansion is parsed again
 * - Resolving every value identifier: binders are renamed apart, and an
 *   identifier a template introduced (syntax_color.h) resolves in its
 *   macro's definition environment
 */
#ifndef ESHKOL_FRONTEND_MACRO_EXPANDER_H
#define ESHKOL_FRONTEND_MACRO_EXPANDER_H

#include <eshkol/eshkol.h>
#include <string>
#include <map>
#include <set>
#include <vector>

namespace eshkol {

/**
 * MacroExpander handles syntax-rules macro expansion.
 *
 * Usage:
 * 1. Create a MacroExpander
 * 2. Call expand() on the top-level AST
 * 3. Use the returned AST (with macros expanded) for codegen
 *
 * The expander is recursive - expanded macros may themselves use macros.
 */
class MacroExpander {
public:
    /**
     * Construct an empty macro expander with no macros registered yet.
     * A global scope is pushed onto the scope stack so top-level
     * define-syntax forms have somewhere to register.
     */
    MacroExpander();

    /**
     * Destroy the macro expander.
     *
     * Registered eshkol_macro_def_t entries are owned by the AST they came
     * from, not by the expander, so no macro definitions are freed here.
     */
    ~MacroExpander();

    /**
     * Expand all macros in the given AST.
     * Modifies the AST in-place and returns it.
     *
     * @param ast The AST to expand
     * @return The expanded AST
     */
    eshkol_ast_t expand(const eshkol_ast_t& ast);

    /**
     * Expand macros in a vector of ASTs (e.g., top-level program).
     *
     * @param asts Vector of ASTs
     * @return Vector of expanded ASTs
     */
    std::vector<eshkol_ast_t> expandAll(const std::vector<eshkol_ast_t>& asts);

    /**
     * Check if a name is a defined macro.
     *
     * @param name Name to check
     * @return true if name is a macro
     */
    bool isMacro(const std::string& name) const;

    /**
     * Get the number of macros defined.
     */
    size_t macroCount() const {
        size_t count = 0;
        for (const auto& scope : scope_stack_) count += scope.size();
        return count;
    }

private:
    // Macro definition table - scope stack for hygiene
    // scope_stack_[0] is global scope, higher indices are nested let-syntax scopes
    struct MacroBinding {
        eshkol_macro_def_t* macro = nullptr;
        // Definition-site lexical value names.  The expander alpha-renames
        // binders as it walks the tree; retaining this snapshot makes a free
        // identifier in a template refer to the definition environment.
        std::map<std::string, std::string> value_env;
        std::vector<std::map<std::string, eshkol_macro_def_t*>> macro_env;
    };
    std::vector<std::map<std::string, MacroBinding>> scope_stack_;
    std::map<const eshkol_macro_def_t*, MacroBinding> definition_bindings_;
    std::map<std::string, const eshkol_macro_def_t*> macro_aliases_;
    std::map<const eshkol_macro_def_t*, std::string> macro_alias_names_;
    // Lexical value bindings in scope: source (possibly colored) name ->
    // the unique name the binder was renamed to.
    std::map<std::string, std::string> value_renames_;
    uint64_t rename_counter_ = 0;

    // ── Hygiene (ADR-0026) ──────────────────────────────────────────────
    // Each expansion colors the identifiers its template introduces
    // (syntax_color.h). color_macro_ remembers which transformer produced a
    // color, so a colored identifier that no binder of the expansion binds
    // is resolved in THAT transformer's definition environment.
    unsigned color_counter_ = 0;
    std::map<unsigned, const eshkol_macro_def_t*> color_macro_;

    // Push/pop macro scopes for let-syntax/letrec-syntax
    void pushScope();
    void popScope();

    // Look up a macro in the scope stack (inner scopes shadow outer)
    eshkol_macro_def_t* lookupMacro(const std::string& name) const;
    const MacroBinding* lookupBinding(const std::string& name) const;
    eshkol_ast_t expandQuasiquoted(const eshkol_ast_t& ast, unsigned depth);

    /** A fresh unique spelling for a binder written @p name. */
    std::string freshValueName(const std::string& name);
    /** What a value reference spelled @p name denotes here. */
    std::string resolveValue(const std::string& name) const;
    /** A colored identifier no binder of its expansion binds (ADR-0026). */
    std::string resolveFree(const std::string& name) const;
    /** Spelling that denotes, at a use site, the macro keyword a template
     *  identifier denotes in @p binding's definition environment. */
    std::string keywordAlias(const MacroBinding* binding, const std::string& name);
    /** Macro keywords the parser must recognise inside an expansion. */
    std::set<std::string> visibleMacroNames() const;

    /**
     * Register a macro definition.
     */
    void registerMacro(const eshkol_macro_def_t* macro);
    void registerMacroWithEnv(const eshkol_macro_def_t* macro,
                              const std::vector<std::map<std::string, eshkol_macro_def_t*>>& env);

    /**
     * Expand a single AST node.
     */
    eshkol_ast_t expandNode(const eshkol_ast_t& ast);

    /**
     * Expand one macro use: match its reader syntax against the transformer,
     * instantiate the template, and parse the expansion. Returns @p call
     * unchanged (after reporting) when no rule matches.
     */
    eshkol_ast_t tryExpandMacroCall(const eshkol_ast_t& call);

    /** Re-parse a use the parser recorded as a macro use whose keyword is
     *  shadowed here by a lexical value binding: it is an ordinary call. */
    eshkol_ast_t reparseAsCall(const eshkol_ast_t& call);

    /**
     * Deep copy an AST node.
     */
    eshkol_ast_t copyAst(const eshkol_ast_t& ast);
};

} // namespace eshkol

#endif // ESHKOL_FRONTEND_MACRO_EXPANDER_H
