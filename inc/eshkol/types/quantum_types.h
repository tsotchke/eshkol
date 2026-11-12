/**
 * @file quantum_types.h
 * @brief Quantum register types for the HoTT checker (ADR-0048).
 *
 * The quantum specialisation of ADR-0004 on the checker that exists today:
 *
 *  - `qreg<e>`   a register of e qubits. Linear (TYPE_FLAG_LINEAR): it is
 *                used exactly once, so the no-cloning and no-discard rules of
 *                the linear `Qubit` type apply to whole registers unchanged.
 *  - `qidx<e>`   the address of one qubit of a `qreg<e>`: a natural number
 *                below e. Two `qidx` arguments of one application must name
 *                different qubits (a gate on qubit i "twice" would copy it).
 *  - `qregion`   the capability of one quantum region. `qalloc` needs it, so
 *                allocation is lexically scoped to the region that owns the
 *                simulator state; a region body may not return quantum data.
 *  - `qregion-body` the parameter type of `call-with-quantum-region`: a
 *                procedure of one `qregion` whose result is classical.
 *
 * Indices are linear expressions over naturals, `c + a1*v1 + ... + ak*vk`
 * (IndexExpr). Variables are bound by unification at applications, or, when
 * a variable is spelled like a parameter of integer type, by the compile-time
 * value of that argument (`(qalloc qr 3) : qreg<3>`). An index the checker
 * cannot know (a run-time size) is the bare family `qreg`, which is still
 * linear and is compatible with every dimension: dimension checking is
 * gradual, linearity is not.
 *
 * Every finding is reported through the checker's linear-violation channel,
 * so it is fatal in every mode but `--unsafe`, on the native and the bytecode
 * engines alike (the VM runs the same checker through linear_check_bridge).
 */
#ifndef ESHKOL_TYPES_QUANTUM_TYPES_H
#define ESHKOL_TYPES_QUANTUM_TYPES_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "eshkol/eshkol.h"
#include "eshkol/types/hott_types.h"

namespace eshkol::hott::quantum {

/** A linear index expression c + sum a_v * v over named naturals. */
struct IndexExpr {
    bool known = true;                       // false: unknown until run time
    int64_t constant = 0;
    std::map<std::string, int64_t> terms;    // variable -> nonzero coefficient

    static IndexExpr nat(int64_t n);
    static IndexExpr var(const std::string& v);
    static IndexExpr unknown();
    /** Parse "3", "n", "n+1", "m+n-2", "2*n", "?"; nullopt on a syntax error. */
    static std::optional<IndexExpr> parse(const std::string& text);

    bool isConstant() const { return known && terms.empty(); }
    std::optional<int64_t> value() const;
    IndexExpr plus(const IndexExpr& other) const;
    IndexExpr scaled(int64_t k) const;
    /** Simultaneous substitution of bound variables. */
    IndexExpr substitute(const std::map<std::string, IndexExpr>& binding) const;
    /** Variables not bound in @p binding. */
    std::vector<std::string> freeVariables(const std::map<std::string, IndexExpr>& binding) const;
    /** Canonical spelling ("n+1", "2*m-n", "3", "?"); equal expressions spell equally. */
    std::string toString() const;
    bool sameAs(const IndexExpr& other) const { return toString() == other.toString(); }
};

enum class Family : uint8_t { QReg, QIdx };

/** The family TypeId (QReg or QIdx). */
TypeId familyBase(Family f);
const char* familyName(Family f);

/** Parse a spelled indexed type `qreg<e>` / `qidx<e>` (case-insensitive family). */
std::optional<std::pair<Family, IndexExpr>> parseIndexedSpelling(const std::string& name,
                                                                 std::string* error = nullptr);

/** The TypeId of family<index>; an unknown index is the bare family. */
TypeId indexedType(TypeEnvironment& env, Family f, const IndexExpr& index);

/** The family and index of a TypeId; the bare family has an unknown index. */
std::optional<std::pair<Family, IndexExpr>> indexedTypeOf(const TypeEnvironment& env, TypeId t);

/** qreg (any dimension), Qubit or qregion: data that must stay inside its region. */
bool isQuantumResource(const TypeEnvironment& env, TypeId t);

// ---------------------------------------------------------------------------
// Signatures
// ---------------------------------------------------------------------------

/** One type in a quantum signature, keeping what TypeIds alone forget. */
struct SpecType {
    enum class Kind { Plain, Indexed, Arrow, Var };
    Kind kind = Kind::Plain;
    TypeId plain = BuiltinTypes::Value;          // Plain
    Family family = Family::QReg;                // Indexed
    IndexExpr index;                             // Indexed
    std::vector<SpecType> params;                // Arrow
    std::shared_ptr<SpecType> result;            // Arrow
    std::string var;                             // Var
    std::string toString(const TypeEnvironment& env) const;
};

struct Signature {
    std::string name;
    std::vector<std::pair<std::string, SpecType>> params;
    SpecType result;
    bool variadic = false;
    /** True when any part of the signature is a quantum family or qregion type. */
    bool quantum = false;
    std::string key(const TypeEnvironment& env) const;
};

/**
 * Convert a parsed annotation to a SpecType. @p resolve resolves a plain
 * annotation to its TypeId (the checker's resolveType); type variables are
 * names that are neither builtin type spellings nor aliases (@p is_alias).
 */
SpecType specFromAnnotation(TypeEnvironment& env, const hott_type_expr_t* annotation,
                            const std::function<TypeId(const hott_type_expr_t*)>& resolve,
                            const std::function<bool(const std::string&)>& is_alias);

/** The TypeId a SpecType erases to (type variables erase to Value). */
TypeId eraseSpec(TypeEnvironment& env, const SpecType& spec);

/** Outcome of checking one application against a quantum signature. */
struct ApplicationOutcome {
    TypeId result = BuiltinTypes::Value;
    /** Per argument: the type the generic argument check should compare against. */
    std::vector<TypeId> expected;
};

using ViolationReporter = std::function<void(const std::string&, const eshkol_ast_t*)>;

/**
 * The registry of quantum signatures known to one checker: the primitives of
 * core.quantum, bound at construction, and every definition whose
 * annotations mention a quantum type. A signature is found through the
 * function TypeId its binding carries.
 */
class Registry {
public:
    /** Bind every primitive signature through @p bind (name, function TypeId). */
    void bindPrimitives(TypeEnvironment& env,
                        const std::function<void(const std::string&, TypeId)>& bind);
    /** Register @p sig and return the function TypeId that identifies it. */
    TypeId registerSignature(TypeEnvironment& env, const Signature& sig);
    const Signature* signatureOf(TypeId fn) const;

    /**
     * Check one application. @p args are the argument ASTs, @p arg_types the
     * synthesized argument types; findings go to @p report (fatal channel).
     */
    ApplicationOutcome checkApplication(TypeEnvironment& env, const Signature& sig,
                                        const eshkol_ast_t* args, size_t nargs,
                                        const std::vector<TypeId>& arg_types,
                                        const eshkol_ast_t* call_site,
                                        const ViolationReporter& report) const;

private:
    std::map<uint16_t, Signature> by_type_;
    std::map<std::string, uint16_t> by_key_;
};

/**
 * Compare a declared result (annotation) with a synthesized body type: a
 * dimension mismatch between two known indices of the same family is
 * reported. Returns true when a finding was reported.
 */
bool checkDeclaredResult(const TypeEnvironment& env, TypeId declared, TypeId actual,
                         const std::string& what, const eshkol_ast_t* site,
                         const ViolationReporter& report);

/** The compile-time natural value of an argument expression, if it has one. */
std::optional<int64_t> literalNatural(const eshkol_ast_t* expr);

/**
 * Turn the spelled type form `(qreg e)` / `(qidx e)` read by the parser into
 * the canonical spelling `qreg<e>`; @p index_sexpr is the index written as an
 * S-expression string ("3", "n", "(+ n 1)", "(* 2 n)").
 */
std::optional<std::string> canonicalIndexedSpelling(const std::string& family,
                                                    const std::string& index_sexpr);

namespace detail {
/** An S-expression of the tiny reader used for primitive signatures and indices. */
struct Sx {
    bool list = false;
    std::string atom;
    std::vector<Sx> items;
};
bool readSx(const std::string& s, size_t& i, Sx& out);
}  // namespace detail

}  // namespace eshkol::hott::quantum

#endif  // ESHKOL_TYPES_QUANTUM_TYPES_H
