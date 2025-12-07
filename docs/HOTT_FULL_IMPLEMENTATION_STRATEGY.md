# HoTT Full Implementation Strategy

**Document Version:** 1.0
**Date:** 2025-12-07
**Status:** Planning

---

## Executive Summary

This document outlines a detailed implementation strategy to bring Eshkol's HoTT (Homotopy Type Theory) type system from its current foundational state to a complete, production-ready dependent type system. The implementation is organized into 7 phases with clear dependencies, deliverables, and testing criteria.

---

## Current State Assessment

### Implemented (Foundation)
- Type infrastructure (TypeId, TypeNode, TypeEnvironment)
- Universe levels (U0, U1, U2, UOmega)
- Bidirectional type checking (synthesis/checking)
- CTValue for compile-time values
- DependentType with dimension indices
- PiType structure (no substitution)
- SigmaType structure (no eliminators)
- DimensionChecker for bounds verification
- LinearContext, BorrowChecker, UnsafeContext

### Missing (Core HoTT)
- Path types with refl and J eliminator
- Type-level substitution for Π-types
- Σ-type eliminators (fst, snd, sigma-elim)
- Inductive type definitions
- Pattern matching with dependent refinement
- Termination checking
- Univalence axiom
- Higher Inductive Types

---

## Implementation Phases

### Phase 7: Type-Level Computation Engine
**Priority:** CRITICAL
**Estimated Effort:** 3-4 weeks
**Dependencies:** None (builds on existing infrastructure)

This phase adds the computational backbone needed for all dependent type features.

#### 7.1 Type Expression AST

**File:** `inc/eshkol/types/type_expr.h` (new)

```cpp
namespace eshkol::hott {

/**
 * Type-level expressions that can be evaluated/normalized.
 * Separate from hott_type_expr_t (parser) - this is the internal representation.
 */
class TypeExpr {
public:
    enum class Kind {
        // Atoms
        Var,        // Type variable reference
        Const,      // Constant type (Int64, String, etc.)

        // Constructors
        Pi,         // Π(x:A).B - dependent function
        Sigma,      // Σ(x:A).B - dependent pair
        Id,         // Id_A(a,b) - identity/path type

        // Type-level application
        App,        // F A - apply type family to argument

        // Type-level lambda (for type families)
        TLam,       // λ(x:K).T - type-level abstraction

        // Inductive types
        Ind,        // Reference to inductive definition

        // Universe
        Type,       // Type_i (universe level i)
    };

    Kind kind;

    // For Var
    std::string var_name;
    size_t debruijn_index;  // De Bruijn index for bound variables

    // For Const
    TypeId const_type;

    // For Pi, Sigma
    std::string binder_name;
    std::unique_ptr<TypeExpr> domain;
    std::unique_ptr<TypeExpr> codomain;  // May reference binder

    // For Id
    std::unique_ptr<TypeExpr> id_type;   // A in Id_A(a,b)
    std::unique_ptr<TypeExpr> id_left;   // a
    std::unique_ptr<TypeExpr> id_right;  // b

    // For App
    std::unique_ptr<TypeExpr> app_func;
    std::unique_ptr<TypeExpr> app_arg;

    // For TLam
    std::unique_ptr<TypeExpr> tlam_body;

    // For Type
    Universe universe_level;

    // Source location for errors
    int line = 0, column = 0;
};

} // namespace eshkol::hott
```

#### 7.2 Type Substitution

**File:** `lib/types/type_subst.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Substitution: replace variable x with expression e in type T.
 * [e/x]T
 *
 * Must handle:
 * - Capture-avoiding substitution
 * - De Bruijn indices
 * - Normalization after substitution
 */
class TypeSubstitution {
public:
    // Single substitution
    static TypeExpr substitute(
        const TypeExpr& type,
        const std::string& var,
        const TypeExpr& replacement
    );

    // Multiple simultaneous substitutions
    static TypeExpr substituteMany(
        const TypeExpr& type,
        const std::vector<std::pair<std::string, TypeExpr>>& substs
    );

    // Shift de Bruijn indices (for going under binders)
    static TypeExpr shift(const TypeExpr& type, int amount, int cutoff = 0);

private:
    // Rename bound variable to avoid capture
    static std::string freshName(const std::string& base,
                                  const std::set<std::string>& avoid);
};

/**
 * Type normalization - reduce type expressions to normal form.
 */
class TypeNormalizer {
public:
    explicit TypeNormalizer(TypeEnvironment& env);

    // Normalize to weak head normal form (enough for type checking)
    TypeExpr whnf(const TypeExpr& type);

    // Normalize completely (for display, comparison)
    TypeExpr normalize(const TypeExpr& type);

    // Check if two types are definitionally equal
    bool definitionallyEqual(const TypeExpr& a, const TypeExpr& b);

private:
    TypeEnvironment& env_;

    // Unfold type-level definitions
    std::optional<TypeExpr> unfold(const TypeExpr& type);

    // Beta reduction for type-level lambdas
    TypeExpr betaReduce(const TypeExpr& lam, const TypeExpr& arg);
};

} // namespace eshkol::hott
```

#### 7.3 Value Expressions (for dependent types)

**File:** `inc/eshkol/types/value_expr.h` (new)

```cpp
namespace eshkol::hott {

/**
 * Compile-time value expressions.
 * Extended from CTValue to support symbolic computation.
 */
class ValueExpr {
public:
    enum class Kind {
        // Literals
        Nat,        // Natural number literal
        Int,        // Integer literal
        Bool,       // Boolean literal

        // Variables
        Var,        // Reference to value variable

        // Arithmetic
        Add, Sub, Mul, Div, Mod,

        // Comparison
        Eq, Lt, Le, Gt, Ge,

        // Conditionals
        If,         // if c then t else e

        // Function application (for computing with functions)
        App,

        // Constructors (for inductives)
        Ctor,       // Constructor application

        // Unknown (runtime value)
        Unknown,
    };

    Kind kind;

    // ... fields for each kind

    // Evaluate to CTValue if possible
    std::optional<CTValue> tryEval() const;

    // Symbolic comparison
    CTValue::CompareResult compare(const ValueExpr& other) const;
};

} // namespace eshkol::hott
```

#### 7.4 Integration Points

**Modify:** `lib/types/type_checker.cpp`

```cpp
// In TypeChecker class:

TypeExpr resolveToTypeExpr(const hott_type_expr_t* parser_type);

TypeCheckResult synthesizeWithTypeExpr(const eshkol_ast_t* expr);

// When checking function application:
TypeCheckResult synthesizeApplication(const eshkol_ast_t* expr) {
    // Get function type
    auto func_result = synthesize(call.function);

    // Extract Pi type
    if (auto pi = asPiType(func_result.type_expr)) {
        // Check argument against domain
        auto arg_result = check(call.argument, pi->domain);

        // CRITICAL: Substitute argument into codomain
        TypeExpr result_type = TypeSubstitution::substitute(
            pi->codomain,
            pi->binder_name,
            arg_result.value_expr  // The actual argument value
        );

        // Normalize result type
        result_type = normalizer_.whnf(result_type);

        return TypeCheckResult::ok(result_type);
    }
}
```

#### 7.5 Deliverables
- [ ] TypeExpr AST with all node types
- [ ] TypeSubstitution with capture avoidance
- [ ] TypeNormalizer with beta reduction
- [ ] ValueExpr for symbolic values
- [ ] Integration with existing type checker
- [ ] Unit tests for substitution edge cases

---

### Phase 8: Path Types (Identity Types)
**Priority:** CRITICAL
**Estimated Effort:** 2-3 weeks
**Dependencies:** Phase 7

Path types are the foundation of HoTT - they represent proofs of equality.

#### 8.1 Path Type Representation

**Add to:** `inc/eshkol/types/hott_types.h`

```cpp
namespace BuiltinTypes {
    // Path/Identity type constructor
    // Id : (A : Type) -> A -> A -> Type
    inline constexpr TypeId Id{210, Universe::U2, TYPE_FLAG_PROOF};
}

/**
 * Path type: Id_A(a, b) represents proofs that a equals b in type A.
 *
 * In HoTT, paths have computational content:
 * - refl : (a : A) -> Id_A(a, a)
 * - Paths can be composed, inverted, mapped over
 */
struct PathType {
    TypeExpr base_type;   // A
    ValueExpr left;       // a
    ValueExpr right;      // b

    // Check if this is reflexivity (left == right)
    bool isRefl() const;

    // Check if endpoints are definitionally equal
    bool endpointsEqual(TypeNormalizer& norm) const;
};
```

#### 8.2 Path Constructors and Eliminators

**File:** `lib/types/path_types.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Path type operations.
 */
class PathOps {
public:
    explicit PathOps(TypeEnvironment& env, TypeNormalizer& norm);

    // === Constructors ===

    /**
     * refl : (A : Type) -> (a : A) -> Id_A(a, a)
     * The canonical proof that anything equals itself.
     */
    TypeCheckResult checkRefl(
        const TypeExpr& type_A,
        const ValueExpr& value_a
    );

    // === Eliminators ===

    /**
     * J eliminator (path induction):
     *
     * J : (A : Type) ->
     *     (a : A) ->
     *     (C : (x : A) -> Id_A(a, x) -> Type) ->  -- motive
     *     (d : C a refl) ->                        -- base case
     *     (x : A) ->
     *     (p : Id_A(a, x)) ->
     *     C x p
     *
     * This is THE fundamental principle for reasoning about equality.
     */
    TypeCheckResult checkJ(
        const TypeExpr& type_A,
        const ValueExpr& value_a,
        const TypeExpr& motive_C,   // C : (x:A) -> Id(a,x) -> Type
        const ValueExpr& base_d,    // d : C a refl
        const ValueExpr& value_x,
        const ValueExpr& path_p     // p : Id(a,x)
    );

    /**
     * Compute J when path is refl:
     * J A a C d a refl ≡ d
     */
    std::optional<ValueExpr> computeJ(
        const ValueExpr& base_d,
        const ValueExpr& path_p
    );

    // === Derived Operations ===

    /**
     * transport : (B : A -> Type) -> Id_A(a, b) -> B(a) -> B(b)
     * Move a value along a path.
     */
    TypeCheckResult checkTransport(
        const TypeExpr& family_B,
        const PathType& path,
        const ValueExpr& value
    );

    /**
     * ap : (f : A -> B) -> Id_A(a, b) -> Id_B(f a, f b)
     * Apply function to both sides of equality.
     */
    TypeCheckResult checkAp(
        const ValueExpr& func_f,
        const PathType& path
    );

    /**
     * concat : Id_A(a, b) -> Id_A(b, c) -> Id_A(a, c)
     * Compose paths.
     */
    TypeCheckResult checkConcat(
        const PathType& path_ab,
        const PathType& path_bc
    );

    /**
     * inv : Id_A(a, b) -> Id_A(b, a)
     * Invert a path.
     */
    TypeCheckResult checkInv(const PathType& path);

private:
    TypeEnvironment& env_;
    TypeNormalizer& norm_;
};

} // namespace eshkol::hott
```

#### 8.3 Parser Extensions for Paths

**Modify:** `lib/frontend/parser.cpp`

```cpp
// Add new AST node types
enum EshkolAstType {
    // ... existing types ...
    ESHKOL_REFL,        // (refl e) - reflexivity proof
    ESHKOL_J,           // (J A a C d x p) - path induction
    ESHKOL_TRANSPORT,   // (transport B p x) - transport along path
    ESHKOL_AP,          // (ap f p) - apply function to path
    ESHKOL_PATH_CONCAT, // (concat p q) - path composition
    ESHKOL_PATH_INV,    // (inv p) - path inversion
};

// Add type expression for identity type
// (= a b) or (Id A a b)
case HOTT_TYPE_ID:
    // Parse Id_A(a, b)
    break;
```

#### 8.4 Syntax Examples

```scheme
;; Reflexivity
(define (refl-example (x : Int)) : (= x x)
  (refl x))

;; Symmetry from J
(define (sym {A : Type} {x y : A} (p : (= x y))) : (= y x)
  (J A x
     (lambda (z : A) (q : (= x z)) (= z x))  ;; motive
     (refl x)                                  ;; base case: (= x x)
     y p))

;; Transitivity from J
(define (trans {A : Type} {x y z : A}
               (p : (= x y)) (q : (= y z))) : (= x z)
  (J A y
     (lambda (w : A) (r : (= y w)) (= x w))
     p    ;; when r is refl, we need (= x y), which is p
     z q))

;; Congruence (ap)
(define (cong {A B : Type} (f : (-> A B)) {x y : A} (p : (= x y)))
    : (= (f x) (f y))
  (ap f p))
```

#### 8.5 Deliverables
- [ ] PathType representation
- [ ] refl constructor with type checking
- [ ] J eliminator with computation rule
- [ ] transport, ap, concat, inv derived operations
- [ ] Parser support for path syntax
- [ ] Tests for path algebra laws

---

### Phase 9: Inductive Types
**Priority:** CRITICAL
**Estimated Effort:** 4-5 weeks
**Dependencies:** Phases 7, 8

Inductive types are the workhorse of dependent type theory.

#### 9.1 Inductive Type Declarations

**File:** `inc/eshkol/types/inductive.h` (new)

```cpp
namespace eshkol::hott {

/**
 * Constructor of an inductive type.
 */
struct InductiveConstructor {
    std::string name;
    std::vector<std::pair<std::string, TypeExpr>> params;  // Constructor parameters
    std::vector<TypeExpr> recursive_args;  // Which args are recursive
    TypeExpr return_type;  // Must be the inductive type (with indices)

    // For indexed families: indices as expressions
    std::vector<ValueExpr> indices;
};

/**
 * Inductive type definition.
 *
 * Examples:
 *   data Nat : Type where
 *     zero : Nat
 *     succ : Nat -> Nat
 *
 *   data Vec (A : Type) : Nat -> Type where
 *     nil  : Vec A 0
 *     cons : (n : Nat) -> A -> Vec A n -> Vec A (succ n)
 */
struct InductiveType {
    std::string name;

    // Parameters (uniform across all constructors)
    std::vector<std::pair<std::string, TypeExpr>> parameters;

    // Indices (can vary between constructors)
    std::vector<std::pair<std::string, TypeExpr>> indices;

    // Target universe
    Universe target_universe;

    // Constructors
    std::vector<InductiveConstructor> constructors;

    // Generated type ID
    TypeId type_id;

    // Is this a recursive type?
    bool is_recursive;

    // Is this strictly positive? (required for consistency)
    bool is_strictly_positive;
};

/**
 * Inductive type registry.
 */
class InductiveRegistry {
public:
    // Register a new inductive type
    TypeCheckResult registerInductive(const InductiveType& ind);

    // Lookup inductive by name
    const InductiveType* lookup(const std::string& name) const;

    // Lookup constructor
    const InductiveConstructor* lookupConstructor(const std::string& name) const;

    // Get the inductive type for a constructor
    const InductiveType* getInductiveForConstructor(const std::string& ctor) const;

    // Check strict positivity
    bool checkStrictPositivity(const InductiveType& ind);

private:
    std::map<std::string, InductiveType> inductives_;
    std::map<std::string, std::string> ctor_to_ind_;  // ctor name -> ind name
};

} // namespace eshkol::hott
```

#### 9.2 Eliminator Generation

**File:** `lib/types/inductive_elim.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Generate elimination principle for an inductive type.
 *
 * For Nat:
 *   Nat-elim : (P : Nat -> Type) ->
 *              P zero ->
 *              ((n : Nat) -> P n -> P (succ n)) ->
 *              (n : Nat) -> P n
 *
 * For Vec:
 *   Vec-elim : (A : Type) ->
 *              (P : (n : Nat) -> Vec A n -> Type) ->
 *              P 0 nil ->
 *              ((n : Nat) -> (x : A) -> (xs : Vec A n) ->
 *               P n xs -> P (succ n) (cons n x xs)) ->
 *              (n : Nat) -> (v : Vec A n) -> P n v
 */
class EliminatorGenerator {
public:
    explicit EliminatorGenerator(TypeEnvironment& env);

    // Generate the eliminator type for an inductive
    TypeExpr generateEliminatorType(const InductiveType& ind);

    // Generate computation rules
    // e.g., Nat-elim P z s zero ≡ z
    //       Nat-elim P z s (succ n) ≡ s n (Nat-elim P z s n)
    std::vector<ComputationRule> generateComputationRules(const InductiveType& ind);

private:
    TypeEnvironment& env_;

    // Build the motive type: (indices...) -> Ind params indices -> Type
    TypeExpr buildMotiveType(const InductiveType& ind);

    // Build method type for a constructor
    TypeExpr buildMethodType(const InductiveType& ind,
                             const InductiveConstructor& ctor,
                             const std::string& motive_name);
};

/**
 * Computation rule: when eliminator meets constructor.
 */
struct ComputationRule {
    std::string eliminator;
    std::string constructor;

    // LHS pattern: elim P methods... (ctor args...)
    // RHS: method args... (recursive-calls...)

    ValueExpr reduce(const std::vector<ValueExpr>& elim_args) const;
};

} // namespace eshkol::hott
```

#### 9.3 Parser Extensions for Inductives

**Modify:** `lib/frontend/parser.cpp`

```scheme
;; Syntax for defining inductives
(define-inductive Nat : Type
  [zero : Nat]
  [succ : (-> Nat Nat)])

;; Indexed family
(define-inductive (Vec A) : (-> Nat Type)
  [nil : (Vec A 0)]
  [cons : (forall (n : Nat) (-> A (Vec A n) (Vec A (succ n))))])

;; Identity type (could be built-in but definable)
(define-inductive (Id {A : Type} (a : A)) : (-> A Type)
  [refl : (Id a a)])
```

#### 9.4 Built-in Inductives

```cpp
// Register core inductive types at startup
void registerBuiltinInductives(InductiveRegistry& reg) {
    // Natural numbers
    InductiveType nat;
    nat.name = "Nat";
    nat.target_universe = Universe::U0;
    nat.constructors = {
        {"zero", {}, {}, nat_type},
        {"succ", {{"n", nat_type}}, {nat_type}, nat_type}
    };
    reg.registerInductive(nat);

    // Booleans
    InductiveType bool_ind;
    bool_ind.name = "Bool";
    bool_ind.target_universe = Universe::U0;
    bool_ind.constructors = {
        {"true", {}, {}, bool_type},
        {"false", {}, {}, bool_type}
    };
    reg.registerInductive(bool_ind);

    // Lists
    InductiveType list;
    list.name = "List";
    list.parameters = {{"A", type_type}};
    list.constructors = {
        {"nil", {}, {}, list_A},
        {"cons", {{"x", A}, {"xs", list_A}}, {list_A}, list_A}
    };
    reg.registerInductive(list);

    // Vectors (indexed)
    InductiveType vec;
    vec.name = "Vec";
    vec.parameters = {{"A", type_type}};
    vec.indices = {{"n", nat_type}};
    vec.constructors = {
        {"vnil", {}, {}, vec_A_0, {zero}},
        {"vcons", {{"n", nat}, {"x", A}, {"xs", vec_A_n}},
                  {vec_A_n}, vec_A_succ_n, {succ_n}}
    };
    reg.registerInductive(vec);
}
```

#### 9.5 Deliverables
- [ ] InductiveType, InductiveConstructor structures
- [ ] InductiveRegistry with positivity checking
- [ ] EliminatorGenerator for automatic eliminator synthesis
- [ ] ComputationRule for reduction
- [ ] Parser syntax for define-inductive
- [ ] Built-in Nat, Bool, List, Vec, Maybe
- [ ] Tests for induction principles

---

### Phase 10: Dependent Pattern Matching
**Priority:** HIGH
**Estimated Effort:** 3-4 weeks
**Dependencies:** Phase 9

Pattern matching is the ergonomic interface to inductive elimination.

#### 10.1 Pattern AST

**File:** `inc/eshkol/types/pattern.h` (new)

```cpp
namespace eshkol::hott {

/**
 * Pattern for matching against inductive values.
 */
struct Pattern {
    enum class Kind {
        Var,        // x - binds value
        Wildcard,   // _ - ignores value
        Ctor,       // (Ctor p1 p2 ...) - constructor pattern
        Literal,    // 0, "hello", etc.
        As,         // x @ p - bind and match
        Inaccessible, // .e - forced by other patterns
    };

    Kind kind;

    // For Var
    std::string var_name;

    // For Ctor
    std::string ctor_name;
    std::vector<Pattern> ctor_args;

    // For Literal
    CTValue literal_value;

    // For As
    std::string as_name;
    std::unique_ptr<Pattern> as_pattern;

    // For Inaccessible
    std::unique_ptr<ValueExpr> inaccessible_expr;

    // Source location
    int line, column;
};

/**
 * Match clause: pattern -> body
 */
struct MatchClause {
    std::vector<Pattern> patterns;  // One per scrutinee
    std::unique_ptr<eshkol_ast_t> body;

    // Guards (optional)
    std::vector<std::unique_ptr<eshkol_ast_t>> guards;
};

/**
 * Match expression.
 */
struct MatchExpr {
    std::vector<std::unique_ptr<eshkol_ast_t>> scrutinees;
    std::vector<MatchClause> clauses;

    // Return type (may depend on scrutinees)
    std::optional<TypeExpr> return_type;
};

} // namespace eshkol::hott
```

#### 10.2 Pattern Elaboration

**File:** `lib/types/pattern_elab.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Elaborate pattern matching to eliminator applications.
 */
class PatternElaborator {
public:
    PatternElaborator(TypeChecker& checker, InductiveRegistry& inductives);

    /**
     * Elaborate a match expression.
     *
     * (match n
     *   [zero body1]
     *   [(succ m) body2])
     *
     * becomes:
     *
     * (Nat-elim (lambda (n) ReturnType)
     *           body1
     *           (lambda (m) (lambda (ih) body2))
     *           n)
     */
    TypeCheckResult elaborate(const MatchExpr& match);

    /**
     * Check coverage: all constructors are handled.
     */
    TypeCheckResult checkCoverage(
        const InductiveType& ind,
        const std::vector<MatchClause>& clauses
    );

    /**
     * Check for redundant clauses.
     */
    std::vector<size_t> findRedundantClauses(
        const std::vector<MatchClause>& clauses
    );

private:
    TypeChecker& checker_;
    InductiveRegistry& inductives_;

    // Build the motive from return type annotation
    TypeExpr buildMotive(
        const std::vector<Pattern>& patterns,
        const TypeExpr& return_type
    );

    // Elaborate a single clause to a method
    ValueExpr elaborateClause(
        const MatchClause& clause,
        const InductiveConstructor& ctor
    );

    // Handle dependent pattern matching:
    // When matching (cons n x xs), we learn that the index is (succ n)
    void refineContext(
        const Pattern& pattern,
        const InductiveConstructor& ctor,
        Context& ctx
    );
};

} // namespace eshkol::hott
```

#### 10.3 Syntax Examples

```scheme
;; Simple pattern matching
(define (is-zero (n : Nat)) : Bool
  (match n
    [zero #t]
    [(succ _) #f]))

;; With binding
(define (pred (n : Nat)) : Nat
  (match n
    [zero zero]
    [(succ m) m]))

;; Dependent pattern matching - type changes based on pattern
(define (vec-head {A : Type} {n : Nat} (v : (Vec A (succ n)))) : A
  (match v
    [(vcons _ x _) x]))  ;; Pattern forces n = succ m for some m

;; Multiple scrutinees
(define (zip {A B : Type} {n : Nat}
             (xs : (Vec A n)) (ys : (Vec B n))) : (Vec (* A B) n)
  (match xs ys
    [vnil vnil vnil]
    [(vcons n x xs') (vcons _ y ys')
     (vcons n (cons x y) (zip xs' ys'))]))

;; With clause for auxiliary computations
(define (filter {A : Type} (p : (-> A Bool)) (xs : (List A)))
    : (Sigma (ys : (List A)) (<= (length ys) (length xs)))
  (match xs
    [nil (sigma nil refl)]
    [(cons x xs')
     (with [(ys pf) := (filter p xs')]
       (if (p x)
           (sigma (cons x ys) (le-succ pf))
           (sigma ys (le-step pf))))]))
```

#### 10.4 Deliverables
- [ ] Pattern AST structures
- [ ] PatternElaborator to eliminators
- [ ] Coverage checking algorithm
- [ ] Redundancy detection
- [ ] Dependent refinement (learning from patterns)
- [ ] With-clause support
- [ ] Parser syntax for match expressions
- [ ] Tests for various matching scenarios

---

### Phase 11: Termination Checking
**Priority:** HIGH
**Estimated Effort:** 3-4 weeks
**Dependencies:** Phases 9, 10

Ensures all recursive definitions terminate.

#### 11.1 Termination Checker

**File:** `lib/types/termination.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Termination checking for recursive functions.
 */
class TerminationChecker {
public:
    explicit TerminationChecker(TypeChecker& checker);

    /**
     * Check that a recursive function terminates.
     *
     * Strategy:
     * 1. Try structural recursion (arguments get smaller)
     * 2. If that fails, look for well-founded recursion
     * 3. If that fails, require explicit termination proof
     */
    TypeCheckResult checkTermination(
        const std::string& func_name,
        const std::vector<std::string>& param_names,
        const eshkol_ast_t* body
    );

private:
    TypeChecker& checker_;

    // === Structural Recursion ===

    /**
     * Check if all recursive calls are on structurally smaller arguments.
     *
     * "Structurally smaller" means:
     * - Direct subterm (car, cdr, tail, etc.)
     * - Pattern match binding (in (succ n), n is smaller than (succ n))
     */
    bool checkStructuralRecursion(
        const std::string& func_name,
        size_t decreasing_arg,  // Which argument decreases
        const eshkol_ast_t* body
    );

    // Track which variables are structurally smaller than others
    struct SizeRelation {
        std::string smaller;
        std::string larger;
    };

    std::vector<SizeRelation> collectSizeRelations(
        const std::vector<MatchClause>& clauses
    );

    // === Well-Founded Recursion ===

    /**
     * Check well-founded recursion with explicit measure.
     *
     * User provides:
     * - A measure function: args -> Nat (or other well-ordered type)
     * - Proof that measure decreases on each recursive call
     */
    bool checkWellFoundedRecursion(
        const std::string& func_name,
        const ValueExpr& measure,
        const eshkol_ast_t* body
    );

    // === Call Graph Analysis ===

    /**
     * For mutual recursion, build call graph and find size-change termination.
     */
    bool checkMutualTermination(
        const std::vector<std::string>& func_names,
        const std::vector<eshkol_ast_t*>& bodies
    );
};

/**
 * Size-change termination analysis.
 * Based on "The Size-Change Principle for Program Termination"
 * (Lee, Jones, Ben-Amram 2001)
 */
class SizeChangeAnalysis {
public:
    enum class Change { Decreasing, NonIncreasing, Unknown };

    struct SizeChangeGraph {
        std::string from_func;
        std::string to_func;
        std::vector<std::vector<Change>> matrix;  // [from_param][to_param]
    };

    // Build size-change graphs from call sites
    std::vector<SizeChangeGraph> buildGraphs(
        const std::map<std::string, eshkol_ast_t*>& functions
    );

    // Check if all infinite call sequences have a decreasing parameter
    bool checkTermination(const std::vector<SizeChangeGraph>& graphs);
};

} // namespace eshkol::hott
```

#### 11.2 Syntax for Termination Hints

```scheme
;; Structural recursion (automatic)
(define (factorial (n : Nat)) : Nat
  (match n
    [zero 1]
    [(succ m) (* n (factorial m))]))  ;; m < n structurally

;; Explicit decreasing argument
(define (ackermann (m : Nat) (n : Nat)) : Nat
  (terminating (lexicographic m n)  ;; (m,n) decreases lexicographically
    (match m n
      [zero _ (succ n)]
      [(succ m') zero (ackermann m' 1)]
      [(succ m') (succ n') (ackermann m' (ackermann m n'))])))

;; Well-founded recursion with measure
(define (merge-sort {A : Type} (cmp : (-> A A Bool)) (xs : (List A))) : (List A)
  (terminating (measure (length xs))
    (match xs
      [nil nil]
      [(cons x nil) (cons x nil)]
      [_ (let [(halves (split xs))]
           (merge cmp
                  (merge-sort cmp (fst halves))
                  (merge-sort cmp (snd halves))))])))
```

#### 11.3 Deliverables
- [ ] TerminationChecker with structural recursion
- [ ] SizeChangeAnalysis for complex cases
- [ ] Well-founded recursion support
- [ ] Mutual recursion handling
- [ ] Integration with function definitions
- [ ] Clear error messages for non-termination

---

### Phase 12: Σ-Type Completeness
**Priority:** HIGH
**Estimated Effort:** 1-2 weeks
**Dependencies:** Phases 7, 9

Complete the Σ-type (dependent pair) implementation.

#### 12.1 Σ-Type Operations

**Modify:** `lib/types/type_checker.cpp`

```cpp
// === Σ-Type Checking ===

/**
 * Check dependent pair construction.
 *
 * (sigma a b) : Σ(x:A).B(x)
 * where a : A and b : B[a/x]
 */
TypeCheckResult TypeChecker::checkSigmaIntro(
    const ValueExpr& first,
    const ValueExpr& second,
    const SigmaType& expected
) {
    // Check first component
    auto first_result = check(first, expected.witness_type);
    if (!first_result.success) return first_result;

    // Substitute first into body type
    TypeExpr body_type = TypeSubstitution::substitute(
        expected.body_type,
        expected.witness_name,
        first
    );

    // Check second component against substituted type
    auto second_result = check(second, body_type);
    if (!second_result.success) return second_result;

    return TypeCheckResult::ok(sigmaTypeToTypeExpr(expected));
}

/**
 * Check first projection.
 *
 * (fst p) : A  where p : Σ(x:A).B(x)
 */
TypeCheckResult TypeChecker::checkFst(const ValueExpr& pair) {
    auto pair_type = synthesize(pair);
    if (!pair_type.success) return pair_type;

    auto sigma = asSigmaType(pair_type.type_expr);
    if (!sigma) {
        return TypeCheckResult::error("fst expects Σ-type, got " +
                                       typeToString(pair_type.type_expr));
    }

    return TypeCheckResult::ok(sigma->witness_type);
}

/**
 * Check second projection - this is dependent!
 *
 * (snd p) : B[fst p / x]  where p : Σ(x:A).B(x)
 */
TypeCheckResult TypeChecker::checkSnd(const ValueExpr& pair) {
    auto pair_type = synthesize(pair);
    if (!pair_type.success) return pair_type;

    auto sigma = asSigmaType(pair_type.type_expr);
    if (!sigma) {
        return TypeCheckResult::error("snd expects Σ-type, got " +
                                       typeToString(pair_type.type_expr));
    }

    // The type of snd depends on the actual first component
    ValueExpr fst_value = ValueExpr::makeFst(pair);
    TypeExpr result_type = TypeSubstitution::substitute(
        sigma->body_type,
        sigma->witness_name,
        fst_value
    );

    return TypeCheckResult::ok(result_type);
}

/**
 * Σ-type elimination (pattern matching).
 *
 * (sigma-elim p (lambda (x y) body))
 * where p : Σ(x:A).B(x)
 *       x : A, y : B(x) in body
 */
TypeCheckResult TypeChecker::checkSigmaElim(
    const ValueExpr& pair,
    const eshkol_ast_t* handler
) {
    // ... elaborate to pattern match
}
```

#### 12.2 Syntax

```scheme
;; Σ-type annotation
(: pair-type (Sigma (n : Nat) (Vec Int n)))

;; Construction
(define my-pair : (Sigma (n : Nat) (Vec Int n))
  (sigma 3 (vector 1 2 3)))

;; Projections
(define the-length : Nat (fst my-pair))
(define the-vector : (Vec Int 3) (snd my-pair))  ;; Type depends on fst

;; Pattern matching on Σ
(define (process-pair (p : (Sigma (n : Nat) (Vec Int n)))) : Int
  (match p
    [(sigma n v) (vec-sum v)]))

;; Existential packaging
(define (make-sized-vec (xs : (List Int))) : (Sigma (n : Nat) (Vec Int n))
  (sigma (length xs) (list-to-vec xs)))
```

---

### Phase 13: Univalence and HITs (Advanced)
**Priority:** MEDIUM
**Estimated Effort:** 4-6 weeks
**Dependencies:** Phases 7, 8, 9

The crowning features of HoTT.

#### 13.1 Equivalences

**File:** `lib/types/equivalence.cpp` (new)

```cpp
namespace eshkol::hott {

/**
 * Type equivalence: A ≃ B
 *
 * An equivalence consists of:
 * - f : A -> B
 * - g : B -> A
 * - η : ∀a. g(f(a)) = a
 * - ε : ∀b. f(g(b)) = b
 * - τ : ∀a. ap f (η a) = ε (f a)  (coherence)
 */
struct Equivalence {
    TypeExpr type_A;
    TypeExpr type_B;
    ValueExpr forward;   // f
    ValueExpr backward;  // g
    ValueExpr eta;       // η (section proof)
    ValueExpr epsilon;   // ε (retraction proof)
    // τ is derivable from the others in HoTT
};

class EquivalenceChecker {
public:
    // Check that a value has equivalence type
    TypeCheckResult checkEquivalence(
        const ValueExpr& equiv,
        const TypeExpr& A,
        const TypeExpr& B
    );

    // Build identity equivalence
    Equivalence idEquiv(const TypeExpr& A);

    // Compose equivalences
    Equivalence composeEquiv(const Equivalence& e1, const Equivalence& e2);

    // Inverse equivalence
    Equivalence invertEquiv(const Equivalence& e);
};

} // namespace eshkol::hott
```

#### 13.2 Univalence Axiom

```cpp
/**
 * Univalence: (A ≃ B) ≃ (A = B)
 *
 * This is either:
 * - Postulated as an axiom
 * - Computed using cubical type theory (advanced)
 */
class Univalence {
public:
    /**
     * ua : (A ≃ B) -> (A = B)
     * Convert equivalence to path.
     */
    PathType ua(const Equivalence& equiv);

    /**
     * ua⁻¹ : (A = B) -> (A ≃ B)
     * Convert path to equivalence (via transport).
     */
    Equivalence uaInv(const PathType& path);

    /**
     * Computation: transport along ua is the forward map.
     * transport (λX.X) (ua e) a ≡ e.forward a
     */
    ValueExpr transportUa(const Equivalence& e, const ValueExpr& a);
};
```

#### 13.3 Higher Inductive Types

```cpp
/**
 * Higher Inductive Type definition.
 * Like inductive types but with path constructors.
 */
struct HigherInductiveType {
    std::string name;

    // Point constructors (like regular inductive)
    std::vector<InductiveConstructor> point_ctors;

    // Path constructors
    struct PathConstructor {
        std::string name;
        std::vector<std::pair<std::string, TypeExpr>> params;
        ValueExpr endpoint_left;
        ValueExpr endpoint_right;
        // For 2-paths, etc: dimension
    };
    std::vector<PathConstructor> path_ctors;

    // Truncation level (-2 = contr, -1 = prop, 0 = set, etc.)
    int truncation_level = -3;  // -3 means no truncation
};

// Example: Circle
// data S¹ : Type where
//   base : S¹
//   loop : base = base
```

#### 13.4 Deliverables
- [ ] Equivalence type and operations
- [ ] Univalence axiom (postulated)
- [ ] HIT definition syntax
- [ ] HIT eliminator generation
- [ ] Propositional truncation
- [ ] Quotient types

---

## Implementation Timeline

```
Phase 7 (Type Computation)     ████████████████  Weeks 1-4
Phase 8 (Path Types)           ████████████      Weeks 3-6
Phase 9 (Inductives)           ████████████████████  Weeks 5-10
Phase 10 (Pattern Matching)    ████████████████  Weeks 9-13
Phase 11 (Termination)         ████████████████  Weeks 11-15
Phase 12 (Σ-Types)             ████████          Weeks 14-16
Phase 13 (Univalence/HITs)     ████████████████████  Weeks 16-22
                               ─────────────────────────────────
                               1   4   8   12  16  20  24 weeks
```

---

## Testing Strategy

### Unit Tests (per phase)
- Type substitution correctness
- Normalization termination
- Path algebra laws
- Inductive eliminator types
- Pattern coverage
- Termination detection

### Integration Tests
- Full programs using dependent types
- Libraries: vectors, matrices, verified arithmetic
- Proofs: basic lemmas about Nat, List

### Property-Based Tests
- Random type generation
- Substitution properties: [a/x][b/y]T = [a/x](T[b/y]) when...
- Normalization: normalize(normalize(T)) = normalize(T)

### Regression Tests
- Each bug fix gets a test case
- Performance benchmarks

---

## Risk Assessment

| Risk | Mitigation |
|------|------------|
| Substitution bugs causing unsoundness | Extensive unit tests, formal spec |
| Termination checker too restrictive | Multiple strategies, escape hatches |
| Poor error messages | Invest in error infrastructure early |
| Performance issues | Profile early, lazy normalization |
| Feature interactions | Careful phase ordering |

---

## Success Criteria

The HoTT implementation is complete when:

1. **Core Features Work**
   - Can define Nat, Vec, List as inductives
   - Can prove properties about them
   - Path types with J eliminator function correctly

2. **Practical Usability**
   - Pattern matching is ergonomic
   - Error messages are helpful
   - Common patterns are not too verbose

3. **Theoretical Soundness**
   - No way to prove False
   - Termination checker catches infinite loops
   - Type safety: well-typed programs don't crash

4. **Performance**
   - Type checking is fast enough for interactive use
   - Generated code is efficient

---

## References

- [The HoTT Book](https://homotopytypetheory.org/book/)
- [Agda Documentation](https://agda.readthedocs.io/)
- [Coq Reference Manual](https://coq.inria.fr/refman/)
- [Size-Change Termination Paper](https://doi.org/10.1145/360204.360210)
- [Elaboration Zoo](https://github.com/AndrasKovacs/elaboration-zoo)
