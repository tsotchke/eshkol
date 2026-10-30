/**
 * @file structural_checker.h
 * @brief The structural layer of the one type checker (ADR-0046).
 *
 * Checks the structurally typed fragment of a program: extensible records
 * with row polymorphism, arbitrary-rank polymorphism (bidirectional,
 * predicative; Peyton Jones, Vytiniotis, Weirich and Shields, "Practical
 * type inference for arbitrary-rank types", JFP 2007) and binary session
 * types over channel endpoints. It is driven by TypeChecker::synthesize() for
 * every top-level form, reports through TypeChecker::reportStructuralViolation(),
 * and enforces endpoint linearity through the TypeChecker's own linear
 * judgment (analyzeLinearUses / reportLinearVerdict in strict mode), so the
 * native engine and the VM bridge draw one verdict.
 *
 * Which forms it judges: a top-level definition whose annotations use a
 * structural type former (`record`, `chan`, a protocol, `-o`, `forall`, a
 * user type constructor), every form that mentions an operation of a
 * required structural library (core.records, core.sessions, core.threads),
 * and, transitively, every definition and expression that mentions one of
 * those definitions. Everything else stays with the gradual TypeId layer;
 * a value crossing from it is `any` (Dyn), and a linear value may not cross
 * into it.
 */
#ifndef ESHKOL_TYPES_STRUCTURAL_CHECKER_H
#define ESHKOL_TYPES_STRUCTURAL_CHECKER_H

#include "eshkol/eshkol.h"
#include "eshkol/types/structural_types.h"

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace eshkol::hott {

class TypeChecker;
class TypeEnvironment;

class StructuralChecker {
public:
    StructuralChecker(TypeChecker& tc, TypeEnvironment& env);
    ~StructuralChecker();

    /** See TypeChecker::beginProgram(). */
    void beginProgram(const eshkol_ast_t* asts, size_t num_asts);

    /** Judge one top-level form (after the TypeId layer has seen it). */
    void checkTopLevel(eshkol_ast_t* form);

    /** Printed type scheme of a checked top-level name, if it has one. */
    std::optional<std::string> schemeOf(const std::string& name);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace eshkol::hott

#endif  // ESHKOL_TYPES_STRUCTURAL_CHECKER_H
