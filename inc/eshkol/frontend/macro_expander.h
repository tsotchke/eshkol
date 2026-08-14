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
    std::vector<std::map<std::string, eshkol_macro_def_t*>> scope_stack_;

    // Push/pop macro scopes for let-syntax/letrec-syntax
    void pushScope();
    void popScope();

    // Look up a macro in the scope stack (inner scopes shadow outer)
    eshkol_macro_def_t* lookupMacro(const std::string& name) const;

    // A matched value at some ellipsis-nesting depth (R7RS 4.3.2).
    //   depth == 0 : a single AST node (scalar match, no ellipsis).
    //   depth == N : a list of MatchTree, each at depth N-1 (one entry per
    //                repetition matched by the Nth-from-outermost ellipsis).
    // This lets a pattern variable matched under nested ellipses, e.g.
    // ((r ...) ...), carry its full nested shape so templates can either
    // preserve the nesting (single ellipsis per level) or flatten it
    // (consecutive "... ..." in the template, one flatten per extra ellipsis).
    struct MatchTree {
        int depth = 0;
        eshkol_ast_t scalar{};
        std::vector<MatchTree> elements;
    };

    // True if `tree` denotes at least one concrete AST value (a depth-0
    // scalar, or a non-empty nested list bottoming out in one). False for a
    // tree produced by matching an ellipsis pattern zero times.
    static bool matchTreeHasValue(const MatchTree& tree);

    // The leftmost scalar AST reachable from `tree` (descends elements[0]
    // repeatedly until depth 0). Only meaningful when matchTreeHasValue()
    // is true; used when a template references a pattern variable directly
    // (without enough trailing ellipses to fully unwrap its matched depth).
    static const eshkol_ast_t& matchTreeFirstScalar(const MatchTree& tree);

    // Bindings captured during pattern matching
    // Maps pattern variable names to their captured (possibly nested) match tree
    struct Binding {
        std::string name;
        MatchTree tree;
    };
    using Bindings = std::map<std::string, Binding>;

    // ── Hygiene: alpha-renaming of template-introduced binders (R7RS 4.3.2)
    //
    // A template may bind identifiers of its own, as in
    //     ((_ e) (let ((tmp 100)) (+ tmp e)))
    // R7RS requires that `tmp` be FRESH: it may not capture a `tmp` the caller
    // passes in through `e`, and the caller's `tmp` may not capture it.  Without
    // renaming, (hyg tmp) expands to (let ((tmp 100)) (+ tmp tmp)) and yields
    // 200 where R7RS requires 107 — filed as SW-30.
    //
    // `active_renames_` maps a template-introduced binder to its fresh name and
    // is scoped by save/restore around each binding form, so a name is renamed
    // only inside the extent that introduced it.  A template identifier that is
    // FREE (referring to the macro-definition environment) is left alone.
    //
    // Pattern variables are never renamed: they are replaced by caller code,
    // which must keep its own names.  The rename is decided at the binder,
    // before its value is substituted, precisely so caller code is never
    // touched.
    //
    // NOT covered: referential transparency for a template's free identifiers.
    // eshkol_macro_def_t carries no definition environment, so a free template
    // identifier is still resolved at the USE site.  This closes the capture
    // half of hygiene only — see the SW-30 ledger entry.
    std::map<std::string, std::string> active_renames_;
    uint64_t rename_counter_ = 0;

    // True while substituting inside QUOTED DATA, where symbols are data
    // rather than identifiers and must keep their literal names. Without this
    // a template's own gensym leaks into the program's OUTPUT: the template
    // (let ((tmp 5)) (list 'tmp tmp)) would print (_h0.tmp 5). Set by quote
    // and quasiquote, cleared again by unquote / unquote-splicing.
    bool in_datum_ = false;

    // Allocate a fresh name for `name` and bind it in `active_renames_`.
    // The counter is monotonic across the whole expansion, so two invocations
    // of the same macro get distinct names — required, since their bindings
    // are distinct.
    std::string freshName(const std::string& name);

    // True if `name` is a template identifier eligible for renaming: not a
    // pattern variable (those carry caller code) and not the `_` wildcard.
    static bool isRenameableBinder(const std::string& name, const Bindings& bindings);

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
     * Match a sequence of pattern elements (a macro call's argument list, or
     * the elements of a nested list pattern) against a flat array of ASTs.
     * Shared by top-level macro-call matching and nested MACRO_PAT_LIST
     * matching so both support ellipsis (including nested ellipsis) the
     * same way. At most one element in [pat_start, pat_count) may be
     * followed_by_ellipsis, and (per current grammar) it must be the last
     * pattern element — it greedily consumes all remaining args.
     */
    bool matchPatternSeq(eshkol_macro_pattern_t* const* elements,
                         uint64_t pat_start, uint64_t pat_count,
                         const eshkol_ast_t* args, uint64_t arg_count,
                         const std::vector<std::string>& literals,
                         Bindings& bindings);

    /**
     * Structurally collect the pattern variables bound within `pattern`
     * (a single pattern element, possibly a nested list) and the ellipsis
     * depth at which each is bound *relative to* `pattern` itself (i.e. not
     * counting any ellipsis that wraps `pattern` from the outside). Used to
     * know which variables to bind (and at what nested-empty shape) when an
     * ellipsis pattern matches zero repetitions, and to know the variable
     * set produced by one repetition when merging matched repetitions.
     */
    void collectPatternVarDepths(const eshkol_macro_pattern_t* pattern,
                                 const std::vector<std::string>& literals,
                                 int depth,
                                 std::map<std::string, int>& out) const;

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

    /**
     * Check if an AST node is the syntax-rules ellipsis marker.
     */
    bool isEllipsisSymbol(const eshkol_ast_t& ast) const;

    /**
     * "Peel" one ellipsis level off every depth>=1 binding in `bindings`,
     * selecting the sub-tree for repetition `index` (depth-0 bindings are
     * shared unchanged across every repetition). The result is the binding
     * environment to use when instantiating a repeated template element for
     * one specific repetition.
     */
    Bindings peelBindingsAtIndex(const Bindings& bindings, size_t index) const;

    /**
     * Find a pattern variable referenced within `ast` whose bound MatchTree
     * has depth >= 1 (i.e. it was matched under at least one ellipsis) —
     * this is the variable that "drives" the repetition count when `ast` is
     * followed by ellipsis in a template. Depth-0 (non-repeated) bindings
     * referenced alongside it are ignored/shared, not used to drive count.
     */
    bool findEllipsisDriver(const eshkol_ast_t& ast,
                            const Bindings& bindings,
                            std::string& binding_name) const;

    /**
     * Expand a template element followed by `ellipsis_count` consecutive
     * ellipses (R7RS 4.3.2 nested ellipsis, e.g. "row ... ..."). For
     * ellipsis_count == 1 this is the ordinary single-ellipsis repetition;
     * for ellipsis_count > 1 each extra ellipsis flattens (splices) one more
     * nesting level into the enclosing list instead of nesting sub-lists.
     */
    std::vector<eshkol_ast_t> expandEllipsisElementN(const eshkol_ast_t& ast,
                                                      const Bindings& bindings,
                                                      int ellipsis_count);

    /**
     * Substitute a sequence/call argument list, expanding element ellipses.
     */
    std::vector<eshkol_ast_t> substituteBindingsInList(const eshkol_ast_t* items,
                                                        uint64_t count,
                                                        const Bindings& bindings);
};

} // namespace eshkol

#endif // ESHKOL_FRONTEND_MACRO_EXPANDER_H
