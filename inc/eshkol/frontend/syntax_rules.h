/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_FRONTEND_SYNTAX_RULES_H
#define ESHKOL_FRONTEND_SYNTAX_RULES_H

/**
 * @file syntax_rules.h
 * @brief R7RS 4.3.2 `syntax-rules` over reader syntax (ADR-0026).
 *
 * Pure datum rewriting: match a macro use against a transformer's patterns
 * and instantiate the first matching template. Every identifier the template
 * introduces is colored with the expansion's color (syntax_color.h); pattern
 * variables are replaced by the caller's syntax unchanged. The caller of this
 * module decides what a colored identifier means (the native expander
 * resolves it in the macro's definition environment).
 */

#include <eshkol/frontend/syntax_datum.h>

#include <functional>
#include <string>

namespace eshkol {

/**
 * Maps a template identifier (as spelled in the template) to the spelling
 * that denotes the same macro keyword at the use site, or returns an empty
 * string when the identifier does not denote a macro keyword where the
 * macro was defined. Keywords are resolved, not colored, because a macro
 * keyword is expanded before any binding form could give a color meaning.
 */
using SyntaxKeywordResolver = std::function<std::string(const std::string&)>;

/** Outcome of applying a transformer to a use. */
enum class SyntaxRulesOutcome { Expanded, NoMatch, Error };

/**
 * Apply @p transformer to @p use (a list datum whose head is the keyword).
 * On Expanded, @p expansion holds the instantiated template; on Error,
 * @p error describes a malformed template instantiation.
 */
SyntaxRulesOutcome syntax_rules_apply(const MacroSyntax& transformer,
                                      const SyntaxDatum& use,
                                      unsigned color,
                                      const SyntaxKeywordResolver& keyword,
                                      SyntaxDatum& expansion,
                                      std::string& error);

} // namespace eshkol

#endif // ESHKOL_FRONTEND_SYNTAX_RULES_H
