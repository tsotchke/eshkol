/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 * MacroExpander - Hygienic macro expansion for Eshkol
 *
 * This module handles:
 * - Collecting macro definitions from AST
 * - Recognizing macro calls
 * - Pattern matching against syntax-rules patterns
 * - Template instantiation with captured bindings
 * - Ellipsis (...) repetition handling
 */
#ifndef ESHKOL_FRONTEND_MACRO_EXPANDER_H
#define ESHKOL_FRONTEND_MACRO_EXPANDER_H

#include <eshkol/eshkol.h>
#include <string>
#include <map>
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
    MacroExpander();
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
    size_t macroCount() const { return macros_.size(); }

private:
    // Macro definition table
    std::map<std::string, eshkol_macro_def_t*> macros_;

    // Bindings captured during pattern matching
    // Maps pattern variable names to captured AST nodes
    struct Binding {
        std::string name;
        std::vector<eshkol_ast_t> values;  // Multiple values for ellipsis
    };
    using Bindings = std::map<std::string, Binding>;

    /**
     * Register a macro definition.
     */
    void registerMacro(const eshkol_macro_def_t* macro);

    /**
     * Expand a single AST node.
     */
    eshkol_ast_t expandNode(const eshkol_ast_t& ast);

    /**
     * Try to expand a call as a macro.
     * Returns expanded AST if successful, or the original if not a macro call.
     */
    eshkol_ast_t tryExpandMacroCall(const eshkol_ast_t& call);

    /**
     * Match a pattern against an AST.
     * Returns true if match succeeds, populates bindings.
     */
    bool matchPattern(const eshkol_macro_pattern_t* pattern,
                      const eshkol_ast_t& ast,
                      const std::vector<std::string>& literals,
                      Bindings& bindings);

    /**
     * Instantiate a template with bindings.
     */
    eshkol_ast_t instantiateTemplate(const eshkol_macro_template_t* tmpl,
                                      const Bindings& bindings);

    /**
     * Check if an identifier is a literal in the current macro.
     */
    bool isLiteral(const std::string& name,
                   const std::vector<std::string>& literals) const;

    /**
     * Deep copy an AST node.
     */
    eshkol_ast_t copyAst(const eshkol_ast_t& ast);

    /**
     * Check if an AST is a symbol with a given name.
     */
    bool isSymbol(const eshkol_ast_t& ast, const std::string& name) const;

    /**
     * Get symbol name from an AST (or empty string if not a symbol).
     */
    std::string getSymbolName(const eshkol_ast_t& ast) const;

    /**
     * Substitute bindings in an AST (for template instantiation).
     */
    eshkol_ast_t substituteBindings(const eshkol_ast_t& ast, const Bindings& bindings);
};

} // namespace eshkol

#endif // ESHKOL_FRONTEND_MACRO_EXPANDER_H
