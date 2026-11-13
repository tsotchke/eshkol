/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 */
#ifndef ESHKOL_TYPES_EFFECT_CHECK_H
#define ESHKOL_TYPES_EFFECT_CHECK_H

/**
 * @file effect_check.h
 * @brief Effect rows: inference and checking (ADR-0045, the effect axis `ε`
 *        of ADR-0004).
 *
 * Every top-level procedure gets a *latent effect row*: what calling it may
 * do. A row is a set of labels plus open tails.
 *
 *   labels  IO, State, Exception, Control, and every effect declared with
 *           `define-effect` (Store, Choice, Async, a program's own, ...).
 *   tails   `param:i` stands for the latent effects of the procedure passed
 *           as the i-th argument (effect polymorphism: `map` performs what its
 *           function performs), minus the labels a handler in between
 *           discharges; `?` stands for a call to a procedure the analysis
 *           cannot see. An open tail is never Pure (ADR-0004 invariant 2).
 *
 * Rules (Plotkin-Pretnar effect typing over the lowered forms):
 *
 *   - calling an operation of effect E adds E;
 *   - `handle` (deep) discharges E from its body when it has a clause for
 *     every operation of E; its clauses' and return clause's rows are added;
 *     a resumption call adds nothing (its effects are the handled body's);
 *   - `guard` with a catch-all clause discharges Exception from its body;
 *     `raise`/`error` add Exception; `call/cc` adds Control; `set!` of a
 *     variable bound outside the enclosing procedure adds State;
 *   - builtins and library procedures come from the signature registry
 *     (effect_check.cpp): a table of effectful and higher-order builtins,
 *     and the naming rules of the standard library (a name ending in `!`
 *     mutates; the I/O families display/write/read-/open-/... perform IO);
 *   - calling a top-level procedure adds its latent row with each `param:i`
 *     replaced by the latent row of the actual argument.
 *
 * `(declare-effects f (L ...))` bounds f's latent row: the inferred row must
 * be contained in it. `()` or `Pure` is the empty closed row; a dotted tail
 * `(L ... . e)` makes the declaration effect-polymorphic in f's procedure
 * arguments. A violation is reported through TypeChecker::reportTypeIssue,
 * so it is an error under --strict-types and a warning in gradual mode.
 *
 * ESHKOL_DUMP_EFFECTS=1 prints every inferred latent row to stderr.
 */

#include <eshkol/eshkol.h>

#include <cstddef>

namespace eshkol::hott { class TypeChecker; }

namespace eshkol::effects {

/**
 * Infer the latent effect rows of every top-level procedure in @p asts and
 * check them against the program's effect declarations. Diagnostics go
 * through @p checker's enforcement point. Returns the number of violations.
 */
size_t checkProgram(const eshkol_ast_t* asts, size_t count, eshkol::hott::TypeChecker& checker);

} // namespace eshkol::effects

#endif // ESHKOL_TYPES_EFFECT_CHECK_H
