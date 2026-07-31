/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef ESHKOL_FRONTEND_LIBRARY_REGISTRY_H
#define ESHKOL_FRONTEND_LIBRARY_REGISTRY_H

/**
 * @file library_registry.h
 * @brief Compile-time registry of the R7RS libraries a compilation unit defines itself.
 *
 * R7RS-small 5.6.1 says a `define-library` form "defines a new library" and
 * that "the library ... may be imported" by the forms that follow it, so a
 * library written in the very unit being compiled has to be importable from
 * that unit without ever touching the filesystem. Before this registry, the
 * frontend lowered `define-library` to a bare sequence and threw the library
 * name away (see ADR 0006), leaving `import` with nothing but
 * `resolve_module_path()` — a filesystem search that can only ever find a
 * library that lives in some *other* file.
 *
 * The registry restores the missing half of the resolution order:
 *
 *   1. libraries established earlier in this compilation unit (this file), then
 *   2. precompiled stdlib modules, then
 *   3. the filesystem search path.
 *
 * Nothing here inspects file names, so "same file" is never special-cased as a
 * string; what makes a library resolvable is that its defining form has already
 * been *processed*, which is also what makes the ordering rule fall out for
 * free. An `import` that precedes its `define-library` sees an empty registry
 * and fails, and planUnit() lets the driver turn that failure into a diagnostic
 * that names the offending forward reference instead of a bare "not found".
 *
 * Population is driven by the marker node that
 * `parse_define_library_form()` appends to every lowered library: an
 * `ESHKOL_PROVIDE_OP` whose `provide_op.library_name` is non-null. A plain
 * `(provide ...)` leaves that field null and registers nothing, so the older
 * `provide`/`require` module style is untouched.
 *
 * State is process-global because a compilation unit is process-global in both
 * lanes: `eshkol-run` compiles one program per process, and the JIT/REPL
 * context is one session. reset() exists for embedders and tests that drive
 * more than one unit through a single process.
 */

#include <eshkol/eshkol.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace eshkol {
namespace library_registry {

/**
 * @brief Forgets every library recorded for the current compilation unit.
 *
 * Only needed by embedders/tests that compile more than one unit in a single
 * process; the one-shot `eshkol-run` driver never has to call it.
 */
void reset();

/**
 * @brief Records the library established by a `define-library` marker node.
 *
 * @param provide_ast An `ESHKOL_PROVIDE_OP` node. Nodes whose
 *   `provide_op.library_name` is null (a plain `(provide ...)`) are ignored.
 *   The node's export names become the library's importable surface, which is
 *   what serves a later `(import (prefix (lib) p:))`.
 */
void define(const eshkol_ast_t& provide_ast);

/**
 * @brief Recursively records every library marker inside one top-level form.
 *
 * Descends through `ESHKOL_SEQUENCE_OP` because a lowered `define-library` is
 * exactly such a sequence, and the driver lanes differ in whether they flatten
 * top-level sequences before walking them.
 */
void defineFromForm(const eshkol_ast_t& form);

/**
 * @brief Reports whether @p name was established earlier in this compilation unit.
 */
bool defined(const std::string& name);

/**
 * @brief Returns the export surface recorded for @p name, or null if @p name is
 *        not a library of this compilation unit.
 */
const std::set<std::string>* exports(const std::string& name);

/**
 * @brief Pre-scans a compilation unit so forward references can be diagnosed.
 *
 * Records the source line of every `define-library` in @p unit that has not
 * been processed yet. A library that is later define()d drops out of the plan,
 * so whatever remains when an import fails is precisely the set of libraries
 * that exist in this unit but are written *below* the import.
 */
void planUnit(const std::vector<eshkol_ast_t>& unit);

/**
 * @brief Reports whether @p name is defined later in this compilation unit.
 *
 * @param line Optional out-parameter receiving the `define-library` form's line.
 * @return true when @p name is an as-yet-unprocessed library of this unit —
 *   i.e. the import that just failed is an illegal forward reference rather
 *   than a genuinely missing module.
 */
bool plannedLater(const std::string& name, uint32_t* line);

}  // namespace library_registry
}  // namespace eshkol

#endif  // ESHKOL_FRONTEND_LIBRARY_REGISTRY_H
