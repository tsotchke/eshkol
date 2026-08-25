/**
 * @file type_checker.h
 * @brief Bidirectional type checker for Eshkol's HoTT type system
 *
 * Implements a bidirectional type checking algorithm that alternates between:
 * - Synthesis mode (Γ ⊢ e ⇒ τ): Infer type from expression structure (bottom-up)
 * - Checking mode (Γ ⊢ e ⇐ τ): Verify expression has expected type (top-down)
 *
 * This approach provides good type inference while keeping error messages precise.
 */

#ifndef ESHKOL_TYPE_CHECKER_H
#define ESHKOL_TYPE_CHECKER_H

#include "eshkol/types/hott_types.h"
#include "eshkol/types/dependent.h"
#include "eshkol/eshkol.h"
#include <vector>
#include <map>
#include <set>
#include <optional>
#include <string>

namespace eshkol::hott {

/**
 * Result of a type checking operation
 */
struct TypeCheckResult {
    bool success;                  // True if the check/synthesis succeeded
    TypeId inferred_type;          // Resulting type (valid when success is true)
    std::string error_message;     // Diagnostic message (populated when success is false)

    // Source location for error reporting
    int line = 0;
    int column = 0;

    // Additional context for better error messages
    std::string context;           // e.g., "in function 'foo'"
    std::string expected_type;     // For type mismatches
    std::string actual_type;       // For type mismatches
    std::string hint;              // Suggested fix

    // Factory methods
    /** Construct a successful result carrying the given inferred/checked type. */
    static TypeCheckResult ok(TypeId type) {
        return {true, type, "", 0, 0, "", "", "", ""};
    }

    /** Construct a failed result with a plain error message and optional source location. */
    static TypeCheckResult error(const std::string& msg, int line = 0, int col = 0) {
        return {false, TypeId{}, msg, line, col, "", "", "", ""};
    }

    // Enhanced error factory with context
    static TypeCheckResult typeMismatch(const std::string& expected, const std::string& actual,
                                        const std::string& context = "") {
        TypeCheckResult result;
        result.success = false;
        result.expected_type = expected;
        result.actual_type = actual;
        result.context = context;
        result.error_message = "Type mismatch: expected '" + expected + "', got '" + actual + "'";
        if (!context.empty()) {
            result.error_message += " " + context;
        }
        return result;
    }

    // Error with hint
    static TypeCheckResult errorWithHint(const std::string& msg, const std::string& hint,
                                         int line = 0, int col = 0) {
        TypeCheckResult result = error(msg, line, col);
        result.hint = hint;
        return result;
    }

    // Format error for display
    std::string formatError() const {
        std::string result = error_message;
        if (!hint.empty()) {
            result += "\n  Hint: " + hint;
        }
        return result;
    }
};

/**
 * Type checking context - tracks variable bindings in scope
 */
class Context {
public:
    /** Construct a context with a single (global) empty scope. */
    Context();

    // Scope management
    /** Push a new (empty) variable scope. */
    void pushScope();
    /** Pop the innermost variable scope, discarding its bindings. */
    void popScope();

    // Variable bindings
    /** Bind @p name to @p type in the innermost scope. */
    void bind(const std::string& name, TypeId type);
    /** Look up @p name, searching from the innermost scope outward. Returns nullopt if unbound. */
    std::optional<TypeId> lookup(const std::string& name) const;

    // Type aliases (from define-type)
    /** Register a (possibly parameterized) type alias introduced by `define-type`. */
    void defineTypeAlias(const std::string& name, hott_type_expr_t* type_expr,
                        const std::vector<std::string>& params = {});
    /** Look up a type alias by name. Returns nullopt if not defined. */
    std::optional<hott_type_expr_t*> lookupTypeAlias(const std::string& name) const;

    // Parameterized type alias support
    /** True if the named type alias has formal type parameters. */
    bool hasTypeAliasParams(const std::string& name) const;
    /** Get the formal parameter names for a parameterized type alias. */
    const std::vector<std::string>& getTypeAliasParams(const std::string& name) const;

    /**
     * Instantiate a parameterized type alias with concrete type arguments
     * @param name The type alias name
     * @param type_args The concrete type arguments to substitute
     * @return A new type expression with parameters substituted, or nullptr if not found
     */
    hott_type_expr_t* instantiateTypeAlias(const std::string& name,
                                           const std::vector<hott_type_expr_t*>& type_args) const;

    // Linear type binding (Phase 6)
    //
    // Enforcement scope — what "use exactly once" actually covers today (#320).
    // The AUTHORITATIVE judgment is no longer the incidental usage counting
    // below: it is TypeChecker::analyzeLinearUses(), a dedicated worst-case-path
    // walk of the binder's body (see its documentation for the decidable
    // fragment and for why a body outside that fragment yields no verdict at
    // all rather than a wrong one). The counters here remain for the use-site
    // bookkeeping that checkLinearVariable() and the narrowing paths rely on.
    //  * Enforced (fatal): `define`/`lambda` PARAMETERS and `let`-family
    //    BINDINGS typed TYPE_FLAG_LINEAR, when the body lies inside the
    //    decidable fragment.
    //  * Static, not dynamic: a use is counted where a linear variable NAME
    //    appears. A closure that captures a linear variable counts as ONE static
    //    use regardless of how many times the closure is later invoked (0 or many
    //    dynamic duplications are invisible to this check).
    /** Bind @p name as a linear variable of the given type (must be used exactly once). */
    void bindLinear(const std::string& name, TypeId type);
    /** Record a use of the linear variable @p name, incrementing its usage count. */
    void useLinear(const std::string& name);
    /** Mark the linear variable @p name as consumed (fully used). */
    void consumeLinear(const std::string& name);
    /** True if @p name was bound via bindLinear(). */
    bool isLinear(const std::string& name) const;
    /** True if the linear variable @p name has been used at least once. */
    bool isLinearUsed(const std::string& name) const;

    // Linear type verification
    /** Get names of linear variables in scope that were never used. */
    std::vector<std::string> getUnusedLinear() const;
    /** Get names of linear variables in scope that were used more than once. */
    std::vector<std::string> getOverusedLinear() const;
    /** True if all linear variables in scope were used exactly once. */
    bool checkLinearConstraints() const;

    // Branch-exclusive linear accounting (#320 item 2). Mutually-exclusive
    // branches (if/cond) must charge the MAX linear use across them, not the sum
    // — e.g. (if b (X q) (Z q)) consumes the linear q exactly once at runtime.
    /** Capture the current per-variable linear usage counts. */
    std::map<std::string, int> snapshotLinearUsage() const { return linear_usage_count_; }
    /** Restore linear usage counts to a previous snapshot (rewind a branch). */
    void restoreLinearUsage(const std::map<std::string, int>& snap) { linear_usage_count_ = snap; }
    /** Merge another branch's usage counts in by taking the per-variable max. */
    void mergeMaxLinearUsage(const std::map<std::string, int>& other) {
        for (const auto& kv : other) {
            auto it = linear_usage_count_.find(kv.first);
            if (it == linear_usage_count_.end() || kv.second > it->second) {
                linear_usage_count_[kv.first] = kv.second;
            }
        }
    }

private:
    // Stack of scopes, each scope is a map from name to type
    std::vector<std::map<std::string, TypeId>> scopes_;

    // Type aliases defined by define-type
    std::map<std::string, hott_type_expr_t*> type_aliases_;

    // Type alias parameters for parameterized types (e.g., (define-type (MyList a) (list a)))
    std::map<std::string, std::vector<std::string>> type_alias_params_;

    // Empty vector for when no params exist (returned by reference)
    static inline const std::vector<std::string> empty_params_ = {};

    // Linear type tracking
    std::set<std::string> linear_vars_;
    std::map<std::string, int> linear_usage_count_;  // 0=unused, 1=used, 2+=overused

    // Per-scope record of names bound linear (via bindLinear) in each active
    // scope, parallel to scopes_. popScope() uses it to remove exactly this
    // scope's linear tracking — restoring any shadowed outer binding — so a
    // linear variable and its over/under-use diagnostics do not leak into
    // sibling defines/lambdas (#320). had_prior/prior_count carry the shadowed
    // outer usage count to restore when this scope exits.
    struct LinearScopeEntry { std::string name; bool had_prior; int prior_count; };
    std::vector<std::vector<LinearScopeEntry>> linear_scope_saves_;
};

/**
 * Linear type usage tracker for quantum no-cloning enforcement
 */
class LinearContext {
public:
    /** Usage state of a tracked linear variable. */
    enum class Usage {
        Unused,       // Declared but never used
        UsedOnce,     // Used exactly once (satisfies linearity)
        UsedMultiple  // Used more than once (violates linearity)
    };

    /** Declare @p name as a tracked linear variable (initially Unused). */
    void declareLinear(const std::string& name);
    /** Record a use of @p name, advancing its usage state. */
    void use(const std::string& name);
    void consume(const std::string& name);  // Mark as fully consumed (must use exactly once)

    /** True if every declared linear variable has been used exactly once. */
    bool checkAllUsedOnce() const;
    /** Get names of declared linear variables that were never used. */
    std::vector<std::string> getUnused() const;
    /** Get names of declared linear variables that were used more than once. */
    std::vector<std::string> getOverused() const;

    // Query usage
    /** Get the current usage state of @p name. */
    Usage getUsage(const std::string& name) const;
    /** True if @p name is a declared linear variable. */
    bool isLinear(const std::string& name) const;

private:
    std::map<std::string, Usage> usage_;
    std::set<std::string> linear_vars_;
};

/**
 * Borrow state for ownership tracking
 */
enum class BorrowState {
    Owned,          // Value is owned, can be moved or borrowed
    Moved,          // Value has been moved, cannot be used
    BorrowedShared, // Value is borrowed immutably (multiple readers allowed)
    BorrowedMut,    // Value is borrowed mutably (exclusive access)
    Dropped         // Value has been explicitly dropped
};

/**
 * Borrow checker for ownership and borrowing semantics
 *
 * Tracks:
 * - Ownership of values (owned, moved, dropped)
 * - Active borrows (shared or mutable)
 * - Borrow lifetimes (scopes)
 *
 * Rules enforced:
 * 1. A value can be moved only once
 * 2. While borrowed, a value cannot be moved
 * 3. Mutable borrows are exclusive (no other borrows allowed)
 * 4. Shared borrows allow multiple readers
 * 5. Borrows must not outlive the borrowed value
 */
class BorrowChecker {
public:
    /** Per-variable ownership/borrow bookkeeping. */
    struct BorrowInfo {
        BorrowState state;
        size_t borrow_count;        // Number of active shared borrows
        bool has_mutable_borrow;    // Is there an active mutable borrow?
        size_t scope_depth;         // Scope where value was defined

        BorrowInfo()
            : state(BorrowState::Owned)
            , borrow_count(0)
            , has_mutable_borrow(false)
            , scope_depth(0) {}
    };

    /** A single ownership/borrow-checking violation. */
    struct BorrowError {
        /** Category of ownership/borrow violation. */
        enum class Kind {
            UseAfterMove,             // Value used after being moved
            UseAfterDrop,             // Value used after being dropped
            MoveWhileBorrowed,        // Attempted to move a value with active borrows
            DoubleMutableBorrow,      // Attempted a second mutable borrow while one is active
            MutableBorrowWhileShared, // Attempted a mutable borrow while shared borrows are active
            BorrowOutlivesValue       // Borrow's scope outlived the borrowed value
        };
        Kind kind;              // Violation category
        std::string variable;   // Name of the offending variable
        std::string message;    // Human-readable diagnostic
    };

    /** Construct a checker starting at scope depth 0 with no tracked values. */
    BorrowChecker() : current_scope_(0) {}

    // Scope management
    /** Enter a new nested scope. */
    void pushScope();
    /** Exit the current scope, returning to its parent. */
    void popScope();
    /** Get the current scope nesting depth. */
    size_t currentScope() const { return current_scope_; }

    // Ownership operations
    /** Declare @p name as a newly owned value in the current scope. */
    void declareOwned(const std::string& name);
    bool move(const std::string& name);          // Returns false if move not allowed
    bool drop(const std::string& name);          // Explicitly drop a value

    // Borrowing operations
    bool borrowShared(const std::string& name);  // Immutable borrow
    bool borrowMut(const std::string& name);     // Mutable borrow
    void returnBorrow(const std::string& name);  // End a borrow

    // State queries
    /** Get the current BorrowState of @p name. */
    BorrowState getState(const std::string& name) const;
    /** True if @p name can currently be used (not moved-from or dropped). */
    bool canUse(const std::string& name) const;
    /** True if @p name can currently be moved (owned, no active borrows). */
    bool canMove(const std::string& name) const;
    /** True if @p name can currently be borrowed immutably. */
    bool canBorrowShared(const std::string& name) const;
    /** True if @p name can currently be borrowed mutably (no other active borrows). */
    bool canBorrowMut(const std::string& name) const;

    // Error access
    /** True if any borrow-checking violations have been recorded. */
    bool hasErrors() const { return !errors_.empty(); }
    /** Get all recorded borrow-checking violations. */
    const std::vector<BorrowError>& errors() const { return errors_; }
    /** Discard all recorded violations. */
    void clearErrors() { errors_.clear(); }

private:
    std::map<std::string, BorrowInfo> values_;
    std::vector<BorrowError> errors_;
    size_t current_scope_;

    void addError(BorrowError::Kind kind, const std::string& var, const std::string& msg);
};

/**
 * Unsafe context for escape hatches
 *
 * In unsafe blocks, the following restrictions are relaxed:
 * - Linear types can be duplicated (no-cloning bypassed)
 * - Borrows can be used after the original is moved
 * - Linear variables don't need to be consumed
 *
 * This is needed for:
 * - FFI interop with C libraries
 * - Low-level memory manipulation
 * - Performance-critical code that needs to bypass safety checks
 * - Implementing safe abstractions on top of unsafe primitives
 */
class UnsafeContext {
public:
    /** Construct a context outside any unsafe block (depth 0). */
    UnsafeContext() : unsafe_depth_(0) {}

    // Enter/exit unsafe blocks
    /** Enter an unsafe block, incrementing the nesting depth. */
    void enterUnsafe() { unsafe_depth_++; }
    /** Exit an unsafe block, decrementing the nesting depth (floored at 0). */
    void exitUnsafe() { if (unsafe_depth_ > 0) unsafe_depth_--; }

    // Check if currently in unsafe context
    bool isUnsafe() const { return unsafe_depth_ > 0; }

    // Get nesting depth (for nested unsafe blocks)
    size_t depth() const { return unsafe_depth_; }

    // RAII helper for scoped unsafe blocks
    class ScopedUnsafe {
    public:
        /** Enter an unsafe block on @p ctx for the lifetime of this object. */
        explicit ScopedUnsafe(UnsafeContext& ctx) : ctx_(ctx) { ctx_.enterUnsafe(); }
        /** Exit the unsafe block entered by the constructor. */
        ~ScopedUnsafe() { ctx_.exitUnsafe(); }
        ScopedUnsafe(const ScopedUnsafe&) = delete;
        ScopedUnsafe& operator=(const ScopedUnsafe&) = delete;
    private:
        UnsafeContext& ctx_;
    };

private:
    size_t unsafe_depth_;
};

/**
 * Main type checker class
 *
 * Usage:
 *   TypeEnvironment env;
 *   TypeChecker checker(env);
 *
 *   // Check a function
 *   auto result = checker.synthesize(ast);
 *   if (!result.success) {
 *       reportError(result);
 *   }
 */
class TypeChecker {
public:
    /**
     * Construct a type checker over the given type environment.
     * @param env Type environment providing type registration/lookup/subtyping
     * @param strict_types If true, type issues call eshkol_error() and abort compilation;
     *                     if false (default), issues call eshkol_warn() and compilation continues
     * @param unsafe_mode If true, starts with type/linear/borrow issues silently ignored
     */
    explicit TypeChecker(TypeEnvironment& env, bool strict_types = false, bool unsafe_mode = false);

    // === Bidirectional Type Checking ===

    /**
     * Synthesize: Infer type from expression (bottom-up)
     * Γ ⊢ e ⇒ τ
     * Stores the inferred type in expr->inferred_hott_type
     */
    TypeCheckResult synthesize(eshkol_ast_t* expr);

    /**
     * Check: Verify expression has expected type (top-down)
     * Γ ⊢ e ⇐ τ
     * Stores the checked type in expr->inferred_hott_type
     */
    TypeCheckResult check(eshkol_ast_t* expr, TypeId expected);

    // === Error Management ===

    /** True if any type errors have been recorded during synthesis/checking. */
    bool hasErrors() const { return !errors_.empty(); }
    /** Get all recorded type errors. */
    const std::vector<TypeCheckResult>& errors() const { return errors_; }
    /** Discard all recorded type errors. */
    void clearErrors() { errors_.clear(); }

    // === Context Access ===

    /** Get the mutable variable-binding context. */
    Context& context() { return ctx_; }
    /** Get the variable-binding context. */
    const Context& context() const { return ctx_; }

    // === Type Resolution ===

    /**
     * Resolve a HoTT type expression to a TypeId
     */
    TypeId resolveType(const hott_type_expr_t* type_expr);

    /**
     * @brief Whether ascribing @p ascribed to an expression synthesized as
     *        @p actual is believable, i.e. not a provable contradiction.
     *
     * `(the <type> expr)` is a *trusted* assertion: the checker adopts
     * `<type>` without re-deriving it, and codegen emits `expr` verbatim
     * (COMPLETE_LANGUAGE_SPECIFICATION.md 3.6.6 — the IR stays
     * byte-identical, there is no runtime tag check). Trust is not the same
     * as silence, though: when the inner expression's type IS statically
     * known and shares no branch of the type graph with the ascribed type,
     * the ascription cannot be satisfied by any value and is a mistake worth
     * reporting rather than propagating.
     *
     * Deliberately permissive — it only rejects the provably impossible:
     * - widening (`actual <: ascribed`) is fine;
     * - narrowing (`ascribed <: actual`) is fine and is the whole point of
     *   the form (recovering a precise type from a dynamic container);
     * - anything involving the dynamic root `Value`, or an unresolved type,
     *   is trusted;
     * - numeric-tower to numeric-tower is always allowed, because this type
     *   graph keeps `Integer`/`Rational`/`Real`/`Complex` flat as siblings
     *   under `Number`, so cross-tower ascriptions like `(the real 1)` have
     *   no subtyping relation in either direction yet are entirely
     *   legitimate.
     *
     * @param actual   Type synthesized for the inner expression.
     * @param ascribed Type named by the ascription.
     * @return false only when the two types are both concrete and disjoint.
     */
    bool ascriptionIsBelievable(TypeId actual, TypeId ascribed) const;

    // === Dimension Checking (Phase 5.3) ===

    /**
     * Check that a vector/tensor access is within bounds
     * @param index_type Type of the index expression
     * @param bound_type Type containing the dimension bound
     * @param context Description for error messages
     */
    TypeCheckResult checkVectorBounds(const CTValue& index, const CTValue& bound,
                                      const std::string& context = "");

    /**
     * Check that two vectors have compatible dimensions for dot product
     */
    TypeCheckResult checkDotProductDimensions(const DependentType& left,
                                              const DependentType& right);

    /**
     * Check matrix multiplication dimension compatibility
     */
    TypeCheckResult checkMatrixMultiplyDimensions(const DependentType& left,
                                                  const DependentType& right);

    /**
     * Extract dimension information from a vector/tensor type
     */
    std::optional<CTValue> extractDimension(TypeId type, size_t dim_index = 0) const;

    // === Linear Type Checking (Phase 6.2) ===

    /**
     * Check if a type is linear (must be used exactly once)
     */
    bool isLinearType(TypeId type) const;

    /**
     * Check linear variable usage in a function body
     * Reports errors for unused or overused linear variables
     */
    void checkLinearUsage();

    /**
     * Verify a linear variable is being used correctly
     * Returns error if variable was already used
     */
    TypeCheckResult checkLinearVariable(const std::string& name);

    /**
     * Check linear bindings at let scope end
     * Verifies all linear bindings were consumed exactly once
     */
    TypeCheckResult checkLinearLet(const std::vector<std::string>& bindings);

    // === Unsafe Context (Phase 6.4) ===

    /**
     * Enter an unsafe block (escape hatch)
     * Linear and borrow restrictions are relaxed
     */
    void enterUnsafe();

    /**
     * Exit an unsafe block
     */
    void exitUnsafe();

    /**
     * Check if currently in unsafe context
     */
    bool isUnsafe() const;

    // === Borrow Checking (Phase 6.3) ===

    /**
     * Get the borrow checker for ownership tracking
     */
    /** Get the mutable borrow checker used for ownership/borrow tracking. */
    BorrowChecker& borrowChecker() { return borrow_; }
    /** Get the borrow checker used for ownership/borrow tracking. */
    const BorrowChecker& borrowChecker() const { return borrow_; }

    // === Unified Enforcement ===

    /**
     * Unified enforcement point: respects strict_types/unsafe_mode config.
     * In unsafe mode: silently ignores.
     * In strict mode: calls eshkol_error() (compilation stops).
     * In gradual mode (default): calls eshkol_warn() (continues).
     */
    void reportTypeIssue(const std::string& msg, const eshkol_ast_t* node = nullptr);

    /**
     * Linearity enforcement point: a no-cloning / no-discard violation.
     *
     * Unlike reportTypeIssue(), this is FATAL in every mode except `--unsafe`.
     * Gradual typing is a statement about how much the checker can INFER; it is
     * not a licence to compile a program the checker has PROVEN ill-formed.
     * `Qubit` is documented — in the language guide, the specification, the
     * v1.3.4 release notes and the public announcement — as making the
     * no-cloning theorem a compile-time type error, and a diagnostic that lets
     * the compile exit 0 and write a runnable binary is not a type error. Only
     * `--unsafe`, whose whole documented purpose is "linear types can be
     * duplicated (no-cloning bypassed)", still suppresses it.
     */
    void reportLinearViolation(const std::string& msg, const eshkol_ast_t* node = nullptr);

    /** True if any linearity violation was reported (fatal regardless of --strict-types). */
    bool hasLinearityViolations() const { return linearity_violations_ > 0; }
    /** Number of linearity violations reported. */
    size_t linearityViolations() const { return linearity_violations_; }

    /**
     * @brief Verdict of the static linear-use analysis over one binder body.
     *
     * `decidable` is the load-bearing field: it is false when the body contains
     * a form the analysis cannot account for, in which case `uses` is
     * meaningless and NO linearity verdict may be drawn from it.
     */
    struct LinearUseAnalysis {
        bool decidable = true;      // false: body left the decidable fragment
        int  uses = 0;              // worst-case uses along any single path
        const char* blocker = "";   // name of the form that ended decidability
    };

    /**
     * @brief Count worst-case uses of the linear binding @p name within @p body.
     *
     * See the implementation for the decidable fragment and the accounting
     * rules (max across mutually-exclusive `if` branches, sum across sequenced
     * and short-circuit forms, alias propagation through `let`, shadowing).
     */
    LinearUseAnalysis analyzeLinearUses(const eshkol_ast_t* body,
                                        const std::string& name) const;

    /**
     * @brief Enforce use-exactly-once for each name in @p linear_names over
     *        @p body, reporting violations at @p site.
     *
     * Fatal (reportLinearViolation) when analyzeLinearUses() reaches a verdict;
     * a warning naming the undecidable form otherwise, so the boundary of what
     * is enforced is visible in the build log rather than only in the docs.
     */
    void enforceLinearBindings(const std::vector<std::string>& linear_names,
                               const eshkol_ast_t* body,
                               const eshkol_ast_t* site);

private:
    bool strict_types_ = false;
    bool unsafe_mode_ = false;
    size_t linearity_violations_ = 0;
    TypeEnvironment& env_;
    Context ctx_;
    BorrowChecker borrow_;
    UnsafeContext unsafe_;
    std::vector<TypeCheckResult> errors_;

    // === Synthesis Helpers ===

    TypeCheckResult synthesizeLiteral(eshkol_ast_t* expr);
    TypeCheckResult synthesizeVariable(eshkol_ast_t* expr);
    TypeCheckResult synthesizeOperation(eshkol_ast_t* expr);
    TypeCheckResult synthesizeLambda(eshkol_ast_t* expr);
    TypeCheckResult synthesizeApplication(eshkol_ast_t* expr);
    TypeCheckResult synthesizeDefine(eshkol_ast_t* expr);
    TypeCheckResult synthesizeLet(eshkol_ast_t* expr);
    TypeCheckResult synthesizeIf(eshkol_ast_t* expr);

    // === Checking Helpers ===

    TypeCheckResult checkLambda(eshkol_ast_t* expr, TypeId expected);

    // === Type Operations ===

    /**
     * Check if a type is a function type and extract domain/codomain
     */
    bool isFunctionType(TypeId type, TypeId& domain, TypeId& codomain) const;

    /**
     * Unify two types (for type inference)
     */
    bool unify(TypeId a, TypeId b);

    // === Error Reporting ===

    void addError(const std::string& msg, int line = 0, int col = 0);
    void addTypeMismatch(TypeId expected, TypeId actual, int line = 0, int col = 0);
};

/**
 * Type check a complete program (list of top-level definitions)
 *
 * @param env The type environment
 * @param asts Array of top-level ASTs
 * @param num_asts Number of ASTs
 * @param strict If true, abort on first error; if false, collect all errors
 * @return Vector of type check results (errors)
 */
std::vector<TypeCheckResult> typeCheckProgram(
    TypeEnvironment& env,
    eshkol_ast_t* asts,
    size_t num_asts,
    bool strict = false);

} // namespace eshkol::hott

#endif // ESHKOL_TYPE_CHECKER_H
