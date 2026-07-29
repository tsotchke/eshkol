/*
 * Copyright (C) tsotchke
 *
 * SPDX-License-Identifier: MIT
 *
 */
/**
 * @file library_registry.cpp
 * @brief Storage for the compile-time R7RS library registry (see library_registry.h).
 */

#include "library_registry.h"

#include <map>

namespace eshkol {
namespace library_registry {

namespace {

/** Libraries whose `define-library` form has already been processed. */
std::map<std::string, std::set<std::string>>& definedLibraries() {
    static std::map<std::string, std::set<std::string>> libraries;
    return libraries;
}

/** Libraries this unit still has ahead of it, by source line (see planUnit()). */
std::map<std::string, uint32_t>& plannedLibraries() {
    static std::map<std::string, uint32_t> planned;
    return planned;
}

/** @return the library name a marker node carries, or null for a plain `(provide ...)`. */
const char* markerLibraryName(const eshkol_ast_t& ast) {
    if (ast.type != ESHKOL_OP || ast.operation.op != ESHKOL_PROVIDE_OP) {
        return nullptr;
    }
    return ast.operation.provide_op.library_name;
}

/** Applies @p visit to @p form and, recursively, to the members of a sequence. */
template <typename Visitor>
void walkForm(const eshkol_ast_t& form, const Visitor& visit) {
    if (form.type != ESHKOL_OP) {
        return;
    }
    if (form.operation.op == ESHKOL_SEQUENCE_OP) {
        for (uint64_t i = 0; i < form.operation.sequence_op.num_expressions; i++) {
            walkForm(form.operation.sequence_op.expressions[i], visit);
        }
        return;
    }
    visit(form);
}

}  // namespace

void reset() {
    definedLibraries().clear();
    plannedLibraries().clear();
}

void define(const eshkol_ast_t& provide_ast) {
    const char* name = markerLibraryName(provide_ast);
    if (!name) {
        return;
    }

    std::set<std::string> exported;
    const auto& provide = provide_ast.operation.provide_op;
    for (uint64_t i = 0; i < provide.num_exports; i++) {
        if (provide.export_names[i]) {
            exported.insert(provide.export_names[i]);
        }
    }

    // A repeated `define-library` for the same name re-establishes it; the last
    // definition processed is the one subsequent imports see, which matches the
    // top-level `define` rule the lowering already relies on.
    definedLibraries()[name] = std::move(exported);
    plannedLibraries().erase(name);
}

void defineFromForm(const eshkol_ast_t& form) {
    walkForm(form, [](const eshkol_ast_t& node) { define(node); });
}

bool defined(const std::string& name) {
    return definedLibraries().count(name) > 0;
}

const std::set<std::string>* exports(const std::string& name) {
    auto it = definedLibraries().find(name);
    return it == definedLibraries().end() ? nullptr : &it->second;
}

void planUnit(const std::vector<eshkol_ast_t>& unit) {
    for (const auto& form : unit) {
        walkForm(form, [](const eshkol_ast_t& node) {
            const char* name = markerLibraryName(node);
            if (!name || defined(name)) {
                return;
            }
            plannedLibraries().emplace(name, node.line);
        });
    }
}

bool plannedLater(const std::string& name, uint32_t* line) {
    auto it = plannedLibraries().find(name);
    if (it == plannedLibraries().end()) {
        return false;
    }
    if (line) {
        *line = it->second;
    }
    return true;
}

}  // namespace library_registry
}  // namespace eshkol
