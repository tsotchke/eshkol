/**
 * @file hole_synthesis.h
 * @brief Type-directed completion of `??` synthesis holes (ADR-0052).
 *
 * A program may leave `??` where an expression belongs:
 *
 *     (define (area (w : integer) (h : integer)) : integer ??)
 *
 * The type checker reports each hole with the type its position expects and
 * with its scope still live (HoleObserver). HoleSynthesizer then enumerates
 * terms bottom-up by size from the bindings in scope at the hole and a typed
 * table of builtin operators, composing only arguments of the parameter types
 * each operator declares; ranks the candidates of the expected type by how
 * many distinct local bindings they use, then by size; and keeps those that
 * the checker itself accepts at the expected type (TypeChecker::checkSpeculative,
 * the checking direction Γ ⊢ e ⇐ τ). fill() replaces each hole with its best
 * candidate and prints the candidates as a diagnostic; a hole with no
 * well-typed completion within the size bound is an error.
 */
#ifndef ESHKOL_TYPES_HOLE_SYNTHESIS_H
#define ESHKOL_TYPES_HOLE_SYNTHESIS_H

#include <eshkol/eshkol.h>
#include <eshkol/types/type_checker.h>

#include <string>
#include <vector>

namespace eshkol::hott {

class HoleSynthesizer : public HoleObserver {
public:
    struct Options {
        size_t max_size = 5;           ///< largest term size enumerated (nodes)
        size_t max_terms = 40000;      ///< bound on the number of terms built per hole
        size_t max_reported = 6;       ///< candidates listed in the diagnostic
        size_t max_validations = 400;  ///< candidates offered to the checker per hole
    };

    struct Report {
        eshkol_ast_t* node = nullptr;
        std::string owner;               ///< enclosing top-level definition, or ""
        TypeId expected = BuiltinTypes::Value;
        std::string expected_name;
        std::vector<std::string> candidates;  ///< best first, each accepted by the checker
        size_t enumerated = 0;           ///< terms built for this hole
    };

    HoleSynthesizer();
    explicit HoleSynthesizer(Options options);

    /** Name the top-level definition being checked; it is not offered as a
     *  component of its own holes (a self-call is never a completion). */
    void beginDefinition(const std::string& owner);

    void onHole(TypeChecker& checker, eshkol_ast_t* hole, TypeId expected) override;

    /** Holes reported since the last fill(). */
    bool pending() const { return !reports_.empty(); }
    const std::vector<Report>& reports() const { return reports_; }

    /**
     * Replace every reported hole by its best candidate and print one
     * diagnostic per hole. Returns the number of holes that had no candidate
     * (each reported as an error). @p file and @p source_text locate the
     * diagnostics (either may be null). Clears the reports.
     */
    size_t fill(const char* file = nullptr, const char* source_text = nullptr);

    /** Candidate list for a type, as program text (for tests and tools). */
    static std::string describe(const Report& report);

private:
    Options options_;
    std::string owner_;
    std::vector<Report> reports_;
};

}  // namespace eshkol::hott

#endif
