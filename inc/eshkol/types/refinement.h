/**
 * @file refinement.h
 * @brief Refinement-type verification (ADR-0044): the entailment judgment
 *        Phi |= phi of ADR-0004, discharged by the built-in decision
 *        procedure (eshkol/types/smt.h) or, optionally, by z3.
 *
 * A refinement type `(Refine T phi)` (or `(Refine T x phi)` with an explicit
 * binder) denotes the values of T that satisfy phi; its runtime
 * representation is T's. The checker walks the program after ordinary type
 * checking, collects facts Phi from function contracts, branch conditions,
 * let bindings and arithmetic, and generates one verification condition per
 * obligation:
 *
 *   - an argument passed to a parameter with a refinement type,
 *   - the value a function with a refined result type returns,
 *   - a value bound by an annotated `let` or ascribed with `(the ...)`,
 *   - inside functions that carry refinement annotations: vector, string,
 *     list and tensor index bounds, sizes passed to make-vector/make-string,
 *     and exact division, quotient, remainder and modulo by zero.
 *
 * Each condition Phi => phi is valid iff Phi /\ ~phi is unsatisfiable.
 * Verdicts and their severity:
 *
 *   proved        - nothing is reported;
 *   contradicted  - Phi => ~phi: every execution reaching the point violates
 *                   the refinement; an error in every mode but --unsafe;
 *   not proved    - a counterexample exists; an error inside annotated
 *                   functions, a gradual-boundary type issue (warning, or an
 *                   error under --strict-types) in unannotated code;
 *   undecided     - the solver answered Unknown (step budget, nonlinear
 *                   arithmetic); a type issue, never success.
 *
 * Programs that contain no refinement type are not walked at all.
 */
#ifndef ESHKOL_TYPES_REFINEMENT_H
#define ESHKOL_TYPES_REFINEMENT_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "eshkol/eshkol.h"

namespace eshkol::hott {

class TypeChecker;

struct RefinementOptions {
    bool strict_types = false;
    bool unsafe_mode = false;
    /** "internal" (default), "z3", or "cross" (both; disagreement is an error). */
    std::string solver = "internal";
    uint64_t budget = 2000000;

    /** Defaults, overridden by ESHKOL_REFINEMENT_SOLVER and ESHKOL_REFINEMENT_BUDGET. */
    static RefinementOptions fromEnvironment(bool strict_types, bool unsafe_mode);
};

struct RefinementReport {
    bool ran = false;
    size_t obligations = 0;
    size_t proved = 0;
    size_t errors = 0;     // fatal findings (contradicted, not proved in annotated code, malformed)
    size_t issues = 0;     // gradual-boundary and undecided findings, reported as type issues
};

/** True when any type expression in the program is (or names) a refinement type. */
bool programHasRefinements(const eshkol_ast_t* asts, size_t num_asts);

/**
 * Verify every refinement obligation of the program. Errors are printed as
 * `[ERROR] Refinement error: ...`; gradual-boundary and undecided findings go
 * through TypeChecker::reportTypeIssue so --strict-types governs them.
 */
RefinementReport checkProgramRefinements(TypeChecker& checker, const eshkol_ast_t* asts,
                                         size_t num_asts, const RefinementOptions& options);

}  // namespace eshkol::hott

#endif  // ESHKOL_TYPES_REFINEMENT_H
