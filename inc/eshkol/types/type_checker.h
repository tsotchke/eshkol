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
    bool success;
    TypeId inferred_type;
    std::string error_message;

    // Source location for error reporting
    int line = 0;
    int column = 0;

    // Factory methods
    static TypeCheckResult ok(TypeId type) {
        return {true, type, ""};
    }

    static TypeCheckResult error(const std::string& msg, int line = 0, int col = 0) {
        return {false, TypeId{}, msg, line, col};
    }
};

/**
 * Type checking context - tracks variable bindings in scope
 */
class Context {
public:
    Context();

    // Scope management
    void pushScope();
    void popScope();

    // Variable bindings
    void bind(const std::string& name, TypeId type);
    std::optional<TypeId> lookup(const std::string& name) const;

    // Type aliases (from define-type)
    void defineTypeAlias(const std::string& name, hott_type_expr_t* type_expr,
                        const std::vector<std::string>& params = {});
    std::optional<hott_type_expr_t*> lookupTypeAlias(const std::string& name) const;

    // Linear type binding (Phase 6)
    void bindLinear(const std::string& name, TypeId type);
    void useLinear(const std::string& name);
    void consumeLinear(const std::string& name);
    bool isLinear(const std::string& name) const;
    bool isLinearUsed(const std::string& name) const;

    // Linear type verification
    std::vector<std::string> getUnusedLinear() const;
    std::vector<std::string> getOverusedLinear() const;
    bool checkLinearConstraints() const;

private:
    // Stack of scopes, each scope is a map from name to type
    std::vector<std::map<std::string, TypeId>> scopes_;

    // Type aliases defined by define-type
    std::map<std::string, hott_type_expr_t*> type_aliases_;

    // Linear type tracking
    std::set<std::string> linear_vars_;
    std::map<std::string, int> linear_usage_count_;  // 0=unused, 1=used, 2+=overused
};

/**
 * Linear type usage tracker for quantum no-cloning enforcement
 */
class LinearContext {
public:
    enum class Usage { Unused, UsedOnce, UsedMultiple };

    void declareLinear(const std::string& name);
    void use(const std::string& name);
    void consume(const std::string& name);  // Mark as fully consumed (must use exactly once)

    bool checkAllUsedOnce() const;
    std::vector<std::string> getUnused() const;
    std::vector<std::string> getOverused() const;

    // Query usage
    Usage getUsage(const std::string& name) const;
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

    struct BorrowError {
        enum class Kind {
            UseAfterMove,
            UseAfterDrop,
            MoveWhileBorrowed,
            DoubleMutableBorrow,
            MutableBorrowWhileShared,
            BorrowOutlivesValue
        };
        Kind kind;
        std::string variable;
        std::string message;
    };

    BorrowChecker() : current_scope_(0) {}

    // Scope management
    void pushScope();
    void popScope();
    size_t currentScope() const { return current_scope_; }

    // Ownership operations
    void declareOwned(const std::string& name);
    bool move(const std::string& name);          // Returns false if move not allowed
    bool drop(const std::string& name);          // Explicitly drop a value

    // Borrowing operations
    bool borrowShared(const std::string& name);  // Immutable borrow
    bool borrowMut(const std::string& name);     // Mutable borrow
    void returnBorrow(const std::string& name);  // End a borrow

    // State queries
    BorrowState getState(const std::string& name) const;
    bool canUse(const std::string& name) const;
    bool canMove(const std::string& name) const;
    bool canBorrowShared(const std::string& name) const;
    bool canBorrowMut(const std::string& name) const;

    // Error access
    bool hasErrors() const { return !errors_.empty(); }
    const std::vector<BorrowError>& errors() const { return errors_; }
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
    UnsafeContext() : unsafe_depth_(0) {}

    // Enter/exit unsafe blocks
    void enterUnsafe() { unsafe_depth_++; }
    void exitUnsafe() { if (unsafe_depth_ > 0) unsafe_depth_--; }

    // Check if currently in unsafe context
    bool isUnsafe() const { return unsafe_depth_ > 0; }

    // Get nesting depth (for nested unsafe blocks)
    size_t depth() const { return unsafe_depth_; }

    // RAII helper for scoped unsafe blocks
    class ScopedUnsafe {
    public:
        explicit ScopedUnsafe(UnsafeContext& ctx) : ctx_(ctx) { ctx_.enterUnsafe(); }
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
    explicit TypeChecker(TypeEnvironment& env);

    // === Bidirectional Type Checking ===

    /**
     * Synthesize: Infer type from expression (bottom-up)
     * Γ ⊢ e ⇒ τ
     */
    TypeCheckResult synthesize(const eshkol_ast_t* expr);

    /**
     * Check: Verify expression has expected type (top-down)
     * Γ ⊢ e ⇐ τ
     */
    TypeCheckResult check(const eshkol_ast_t* expr, TypeId expected);

    // === Error Management ===

    bool hasErrors() const { return !errors_.empty(); }
    const std::vector<TypeCheckResult>& errors() const { return errors_; }
    void clearErrors() { errors_.clear(); }

    // === Context Access ===

    Context& context() { return ctx_; }
    const Context& context() const { return ctx_; }

    // === Type Resolution ===

    /**
     * Resolve a HoTT type expression to a TypeId
     */
    TypeId resolveType(const hott_type_expr_t* type_expr);

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
    BorrowChecker& borrowChecker() { return borrow_; }
    const BorrowChecker& borrowChecker() const { return borrow_; }

private:
    TypeEnvironment& env_;
    Context ctx_;
    BorrowChecker borrow_;
    UnsafeContext unsafe_;
    std::vector<TypeCheckResult> errors_;

    // === Synthesis Helpers ===

    TypeCheckResult synthesizeLiteral(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeVariable(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeOperation(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeLambda(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeApplication(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeDefine(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeLet(const eshkol_ast_t* expr);
    TypeCheckResult synthesizeIf(const eshkol_ast_t* expr);

    // === Checking Helpers ===

    TypeCheckResult checkLambda(const eshkol_ast_t* expr, TypeId expected);

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
    const eshkol_ast_t* asts,
    size_t num_asts,
    bool strict = false);

} // namespace eshkol::hott

#endif // ESHKOL_TYPE_CHECKER_H
