/**
 * @file type_checker.cpp
 * @brief Implementation of bidirectional type checking for Eshkol's HoTT type system
 */

#include "eshkol/types/type_checker.h"
#include "../../lib/core/arena_memory.h"
#include <cstdio>
#include <sstream>
#include <cstring>
#include <cstdlib>

namespace eshkol::hott {

// ============================================================================
// Context Implementation
// ============================================================================

/**
 * @brief Construct a type-checking context, seeding it with a single global scope.
 */
Context::Context() {
    // Start with global scope
    scopes_.push_back({});
}

/**
 * @brief Push a new, empty variable-binding scope onto the scope stack.
 */
void Context::pushScope() {
    scopes_.push_back({});
}

/**
 * @brief Pop the innermost variable-binding scope, discarding its bindings.
 *
 * The outermost (global) scope is never popped, so the scope stack always
 * has at least one entry.
 */
void Context::popScope() {
    if (scopes_.size() > 1) {
        scopes_.pop_back();
    }
}

/**
 * @brief Bind @p name to @p type in the innermost (current) scope.
 *
 * If @p name is already bound in the innermost scope, the previous binding
 * is shadowed/overwritten.
 */
void Context::bind(const std::string& name, TypeId type) {
    if (!scopes_.empty()) {
        scopes_.back()[name] = type;
    }
}

/**
 * @brief Look up the type bound to @p name, searching from the innermost
 * scope outward to the global scope.
 * @return The bound TypeId, or std::nullopt if @p name is not bound in any
 * scope.
 */
std::optional<TypeId> Context::lookup(const std::string& name) const {
    // Search from innermost to outermost scope
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
        auto found = it->find(name);
        if (found != it->end()) {
            return found->second;
        }
    }
    return std::nullopt;
}

/**
 * @brief Register a type alias introduced by a `define-type` form.
 *
 * Records @p type_expr under @p name so later references to @p name resolve
 * to it. If @p params is non-empty, @p name is also recorded as a
 * parameterized (generic) alias, e.g. `(define-type (MyList a) (list a))`.
 */
void Context::defineTypeAlias(const std::string& name, hott_type_expr_t* type_expr,
                              const std::vector<std::string>& params) {
    type_aliases_[name] = type_expr;
    // Store parameters for parameterized type aliases (e.g., (define-type (MyList a) (list a)))
    if (!params.empty()) {
        type_alias_params_[name] = params;
    }
}

/**
 * @brief True if the type alias @p name was declared with formal type parameters.
 */
bool Context::hasTypeAliasParams(const std::string& name) const {
    return type_alias_params_.find(name) != type_alias_params_.end();
}

/**
 * @brief Get the formal type parameter names for the parameterized type alias @p name.
 * @return The declared parameter names, or a reference to an empty vector if
 * @p name has no parameters (or is not a known alias).
 */
const std::vector<std::string>& Context::getTypeAliasParams(const std::string& name) const {
    auto it = type_alias_params_.find(name);
    if (it != type_alias_params_.end()) {
        return it->second;
    }
    return empty_params_;
}

/**
 * @brief Look up the type expression registered for the type alias @p name.
 * @return The alias's type expression, or std::nullopt if @p name is not a
 * known alias.
 */
std::optional<hott_type_expr_t*> Context::lookupTypeAlias(const std::string& name) const {
    auto found = type_aliases_.find(name);
    if (found != type_aliases_.end()) {
        return found->second;
    }
    return std::nullopt;
}

/**
 * @brief Allocate a zero-initialized hott_type_expr_t of the given @p kind from the global arena.
 * @return The new type expression with `kind` set (and all other fields
 * zeroed), or nullptr if arena allocation fails.
 */
static hott_type_expr_t* allocTypeExpr(hott_type_kind_t kind) {
    void* mem = arena_allocate_zeroed(get_global_arena(), sizeof(hott_type_expr_t));
    hott_type_expr_t* type = (hott_type_expr_t*)mem;
    if (type) type->kind = kind;
    return type;
}

/**
 * @brief Recursively substitute type variables in @p type_expr per @p substitutions.
 *
 * Deep-copies @p type_expr, replacing any HOTT_TYPE_VAR node whose name is a
 * key of @p substitutions with a copy of the corresponding replacement type
 * (falling back to copying the variable itself if unmapped). For
 * HOTT_TYPE_FORALL, the variables bound by the forall are removed from the
 * substitution map before recursing into its body so bound variables are not
 * accidentally captured/replaced. Used to instantiate parameterized type
 * aliases (see Context::instantiateTypeAlias).
 * @return A freshly allocated type expression with substitutions applied, or
 * nullptr if @p type_expr is nullptr.
 */
static hott_type_expr_t* substituteTypeVars(
    const hott_type_expr_t* type_expr,
    const std::map<std::string, hott_type_expr_t*>& substitutions) {
    if (!type_expr) return nullptr;

    // Check if this is a type variable that needs substitution
    if (type_expr->kind == HOTT_TYPE_VAR && type_expr->var_name) {
        auto it = substitutions.find(type_expr->var_name);
        if (it != substitutions.end()) {
            // Return a copy of the substituted type
            return hott_copy_type_expr(it->second);
        }
        // Not in substitutions, return a copy of the original
        return hott_copy_type_expr(type_expr);
    }

    // Create a new type expression with substituted children
    hott_type_expr_t* result = allocTypeExpr(type_expr->kind);

    switch (type_expr->kind) {
        case HOTT_TYPE_VAR:
            // Already handled above, but needed for completeness
            result->var_name = type_expr->var_name ? strdup(type_expr->var_name) : nullptr;
            break;

        case HOTT_TYPE_ARROW:
            if (type_expr->arrow.num_params > 0 && type_expr->arrow.param_types) {
                result->arrow.param_types = (hott_type_expr_t**)arena_allocate(get_global_arena(),
                    type_expr->arrow.num_params * sizeof(hott_type_expr_t*));
                for (uint64_t i = 0; i < type_expr->arrow.num_params; i++) {
                    result->arrow.param_types[i] = substituteTypeVars(
                        type_expr->arrow.param_types[i], substitutions);
                }
            }
            result->arrow.num_params = type_expr->arrow.num_params;
            result->arrow.return_type = substituteTypeVars(
                type_expr->arrow.return_type, substitutions);
            break;

        case HOTT_TYPE_FORALL:
            // For forall types, we need to be careful not to substitute bound variables
            if (type_expr->forall.num_vars > 0 && type_expr->forall.type_vars) {
                result->forall.type_vars = (char**)arena_allocate(get_global_arena(),
                    type_expr->forall.num_vars * sizeof(char*));
                // Create a modified substitution map excluding bound variables
                std::map<std::string, hott_type_expr_t*> inner_subst = substitutions;
                for (uint64_t i = 0; i < type_expr->forall.num_vars; i++) {
                    result->forall.type_vars[i] = strdup(type_expr->forall.type_vars[i]);
                    inner_subst.erase(type_expr->forall.type_vars[i]);
                }
                result->forall.num_vars = type_expr->forall.num_vars;
                result->forall.body = substituteTypeVars(type_expr->forall.body, inner_subst);
            } else {
                result->forall.num_vars = 0;
                result->forall.type_vars = nullptr;
                result->forall.body = substituteTypeVars(type_expr->forall.body, substitutions);
            }
            break;

        case HOTT_TYPE_LIST:
        case HOTT_TYPE_VECTOR:
        case HOTT_TYPE_TENSOR:
        case HOTT_TYPE_POINTER:
            result->container.element_type = substituteTypeVars(
                type_expr->container.element_type, substitutions);
            break;

        case HOTT_TYPE_PAIR:
        case HOTT_TYPE_PRODUCT:
            result->pair.left = substituteTypeVars(type_expr->pair.left, substitutions);
            result->pair.right = substituteTypeVars(type_expr->pair.right, substitutions);
            break;

        case HOTT_TYPE_SUM:
            result->sum.left = substituteTypeVars(type_expr->sum.left, substitutions);
            result->sum.right = substituteTypeVars(type_expr->sum.right, substitutions);
            break;

        case HOTT_TYPE_UNIVERSE:
            result->universe.level = type_expr->universe.level;
            break;

        default:
            // Primitive types have no additional data to substitute
            break;
    }

    return result;
}

/**
 * @brief Instantiate the parameterized type alias @p name with concrete type arguments.
 *
 * Looks up @p name's registered type expression and formal parameters, binds
 * each formal parameter to the corresponding entry of @p type_args (pairing
 * up to `min(params.size(), type_args.size())`), and returns the result of
 * substituting those bindings into the alias body via substituteTypeVars().
 * If @p name has no declared parameters, a plain copy of the alias's type
 * expression is returned.
 * @return The instantiated type expression, or nullptr if @p name is not a
 * known type alias.
 */
hott_type_expr_t* Context::instantiateTypeAlias(
    const std::string& name,
    const std::vector<hott_type_expr_t*>& type_args) const {

    // Look up the type alias
    auto alias_it = type_aliases_.find(name);
    if (alias_it == type_aliases_.end()) {
        return nullptr;  // Alias not found
    }

    // Look up the type parameters
    auto params_it = type_alias_params_.find(name);
    if (params_it == type_alias_params_.end()) {
        // No parameters - just return a copy of the alias
        return hott_copy_type_expr(alias_it->second);
    }

    const std::vector<std::string>& params = params_it->second;

    // Build substitution map
    std::map<std::string, hott_type_expr_t*> substitutions;
    size_t num_to_substitute = std::min(params.size(), type_args.size());
    for (size_t i = 0; i < num_to_substitute; i++) {
        substitutions[params[i]] = type_args[i];
    }

    // Perform substitution
    return substituteTypeVars(alias_it->second, substitutions);
}

// ============================================================================
// LinearContext Implementation
// ============================================================================

/**
 * @brief Declare @p name as a tracked linear variable, starting in the Unused state.
 */
void LinearContext::declareLinear(const std::string& name) {
    linear_vars_.insert(name);
    usage_[name] = Usage::Unused;
}

/**
 * @brief Record a use of @p name, advancing its usage state.
 *
 * No-op if @p name is not a declared linear variable. Otherwise transitions
 * Unused -> UsedOnce on the first use, and UsedOnce/UsedMultiple ->
 * UsedMultiple on any subsequent use, so a linearity violation (used more
 * than once) is detectable later via getOverused()/checkAllUsedOnce().
 */
void LinearContext::use(const std::string& name) {
    if (linear_vars_.count(name) == 0) return;

    auto& u = usage_[name];
    if (u == Usage::Unused) {
        u = Usage::UsedOnce;
    } else {
        u = Usage::UsedMultiple;
    }
}

/**
 * @brief Check that every declared linear variable was used exactly once.
 * @return True if all tracked variables are in the UsedOnce state; false if
 * any is Unused or UsedMultiple.
 */
bool LinearContext::checkAllUsedOnce() const {
    for (const auto& name : linear_vars_) {
        auto it = usage_.find(name);
        if (it == usage_.end() || it->second != Usage::UsedOnce) {
            return false;
        }
    }
    return true;
}

/**
 * @brief Get the names of declared linear variables that were never used.
 * @return Names currently in the Unused usage state.
 */
std::vector<std::string> LinearContext::getUnused() const {
    std::vector<std::string> result;
    for (const auto& name : linear_vars_) {
        auto it = usage_.find(name);
        if (it == usage_.end() || it->second == Usage::Unused) {
            result.push_back(name);
        }
    }
    return result;
}

/**
 * @brief Get the names of declared linear variables that were used more than once.
 * @return Names currently in the UsedMultiple usage state.
 */
std::vector<std::string> LinearContext::getOverused() const {
    std::vector<std::string> result;
    for (const auto& name : linear_vars_) {
        auto it = usage_.find(name);
        if (it != usage_.end() && it->second == Usage::UsedMultiple) {
            result.push_back(name);
        }
    }
    return result;
}

/**
 * @brief Mark @p name as fully consumed (must be used exactly once).
 *
 * Semantically delegates to use(); provided as a clearer-named API entry
 * point for call sites that consume a linear variable outright.
 */
void LinearContext::consume(const std::string& name) {
    // Mark as used exactly once (same as use() semantically, but clearer for API)
    use(name);
}

/**
 * @brief Get the current usage state of @p name.
 * @return The tracked Usage value, or Usage::Unused if @p name has no
 * recorded usage (including if it is not a declared linear variable).
 */
LinearContext::Usage LinearContext::getUsage(const std::string& name) const {
    auto it = usage_.find(name);
    if (it == usage_.end()) {
        return Usage::Unused;
    }
    return it->second;
}

/**
 * @brief True if @p name was declared via declareLinear().
 */
bool LinearContext::isLinear(const std::string& name) const {
    return linear_vars_.count(name) > 0;
}

// ============================================================================
// BorrowChecker Implementation (Phase 6.3)
// ============================================================================

/**
 * @brief Enter a new nested scope, incrementing the current scope depth.
 */
void BorrowChecker::pushScope() {
    current_scope_++;
}

/**
 * @brief Exit the current scope, checking that no borrows outlive it.
 *
 * For every tracked value whose BorrowInfo::scope_depth equals the scope
 * being popped and which still has an active shared or mutable borrow,
 * records a BorrowError::Kind::BorrowOutlivesValue error before decrementing
 * the scope depth. No-op if already at the outermost scope (depth 0).
 */
void BorrowChecker::popScope() {
    if (current_scope_ > 0) {
        // Check for borrows that outlive their scope
        for (auto& [name, info] : values_) {
            if (info.scope_depth == current_scope_ &&
                (info.borrow_count > 0 || info.has_mutable_borrow)) {
                addError(BorrowError::Kind::BorrowOutlivesValue, name,
                        "Borrow of '" + name + "' outlives its scope");
            }
        }
        current_scope_--;
    }
}

/**
 * @brief Declare @p name as a newly owned value in the current scope.
 *
 * Initializes (or resets) its BorrowInfo to BorrowState::Owned, no active
 * borrows, and BorrowInfo::scope_depth set to the current scope.
 */
void BorrowChecker::declareOwned(const std::string& name) {
    BorrowInfo info;
    info.state = BorrowState::Owned;
    info.scope_depth = current_scope_;
    values_[name] = info;
}

/**
 * @brief Attempt to move the value bound to @p name.
 *
 * Fails (recording the corresponding BorrowError) if @p name is unknown,
 * already moved (UseAfterMove), already dropped (UseAfterDrop), or has any
 * active shared/mutable borrow (MoveWhileBorrowed). On success, transitions
 * @p name's state to BorrowState::Moved.
 * @return True if the move succeeded; false otherwise.
 */
bool BorrowChecker::move(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return false;
    }

    auto& info = it->second;

    // Cannot move if already moved or dropped
    if (info.state == BorrowState::Moved) {
        addError(BorrowError::Kind::UseAfterMove, name,
                "Cannot move '" + name + "': already moved");
        return false;
    }
    if (info.state == BorrowState::Dropped) {
        addError(BorrowError::Kind::UseAfterDrop, name,
                "Cannot move '" + name + "': already dropped");
        return false;
    }

    // Cannot move if borrowed
    if (info.borrow_count > 0 || info.has_mutable_borrow) {
        addError(BorrowError::Kind::MoveWhileBorrowed, name,
                "Cannot move '" + name + "': currently borrowed");
        return false;
    }

    info.state = BorrowState::Moved;
    return true;
}

/**
 * @brief Explicitly drop the value bound to @p name.
 *
 * Fails silently (returns false, no error recorded) if @p name is unknown or
 * already moved/dropped. Fails with a MoveWhileBorrowed error if @p name has
 * any active shared/mutable borrow. On success, transitions @p name's state
 * to BorrowState::Dropped.
 * @return True if the drop succeeded; false otherwise.
 */
bool BorrowChecker::drop(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return false;
    }

    auto& info = it->second;

    if (info.state == BorrowState::Moved || info.state == BorrowState::Dropped) {
        return false;  // Already gone
    }

    if (info.borrow_count > 0 || info.has_mutable_borrow) {
        addError(BorrowError::Kind::MoveWhileBorrowed, name,
                "Cannot drop '" + name + "': currently borrowed");
        return false;
    }

    info.state = BorrowState::Dropped;
    return true;
}

/**
 * @brief Attempt to take an immutable (shared) borrow of @p name.
 *
 * Fails if @p name is unknown, already moved (UseAfterMove), already dropped
 * (UseAfterDrop), or currently has an active mutable borrow
 * (MutableBorrowWhileShared). On success, increments the shared borrow
 * count and, if the value was Owned, transitions it to BorrowedShared.
 * @return True if the borrow succeeded; false otherwise.
 */
bool BorrowChecker::borrowShared(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return false;
    }

    auto& info = it->second;

    if (info.state == BorrowState::Moved) {
        addError(BorrowError::Kind::UseAfterMove, name,
                "Cannot borrow '" + name + "': already moved");
        return false;
    }
    if (info.state == BorrowState::Dropped) {
        addError(BorrowError::Kind::UseAfterDrop, name,
                "Cannot borrow '" + name + "': already dropped");
        return false;
    }

    // Cannot create shared borrow if mutable borrow exists
    if (info.has_mutable_borrow) {
        addError(BorrowError::Kind::MutableBorrowWhileShared, name,
                "Cannot create shared borrow of '" + name + "': mutable borrow active");
        return false;
    }

    info.borrow_count++;
    if (info.state == BorrowState::Owned) {
        info.state = BorrowState::BorrowedShared;
    }
    return true;
}

/**
 * @brief Attempt to take an exclusive (mutable) borrow of @p name.
 *
 * Fails if @p name is unknown, already moved (UseAfterMove), already dropped
 * (UseAfterDrop), has any active shared borrows (MutableBorrowWhileShared),
 * or already has an active mutable borrow (DoubleMutableBorrow). On success,
 * sets BorrowInfo::has_mutable_borrow and transitions the state to
 * BorrowState::BorrowedMut.
 * @return True if the borrow succeeded; false otherwise.
 */
bool BorrowChecker::borrowMut(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return false;
    }

    auto& info = it->second;

    if (info.state == BorrowState::Moved) {
        addError(BorrowError::Kind::UseAfterMove, name,
                "Cannot borrow '" + name + "': already moved");
        return false;
    }
    if (info.state == BorrowState::Dropped) {
        addError(BorrowError::Kind::UseAfterDrop, name,
                "Cannot borrow '" + name + "': already dropped");
        return false;
    }

    // Cannot create mutable borrow if any borrows exist
    if (info.borrow_count > 0) {
        addError(BorrowError::Kind::MutableBorrowWhileShared, name,
                "Cannot create mutable borrow of '" + name + "': shared borrows active");
        return false;
    }
    if (info.has_mutable_borrow) {
        addError(BorrowError::Kind::DoubleMutableBorrow, name,
                "Cannot create mutable borrow of '" + name + "': already mutably borrowed");
        return false;
    }

    info.has_mutable_borrow = true;
    info.state = BorrowState::BorrowedMut;
    return true;
}

/**
 * @brief End one active borrow of @p name (mutable borrow takes priority over shared).
 *
 * No-op if @p name is unknown. Releases the mutable borrow if one is active,
 * otherwise decrements the shared borrow count if positive. If no borrows
 * remain afterward, restores the state from BorrowedShared/BorrowedMut back
 * to BorrowState::Owned.
 */
void BorrowChecker::returnBorrow(const std::string& name) {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return;
    }

    auto& info = it->second;

    if (info.has_mutable_borrow) {
        info.has_mutable_borrow = false;
    } else if (info.borrow_count > 0) {
        info.borrow_count--;
    }

    // Restore to Owned if no borrows remain
    if (!info.has_mutable_borrow && info.borrow_count == 0) {
        if (info.state == BorrowState::BorrowedShared ||
            info.state == BorrowState::BorrowedMut) {
            info.state = BorrowState::Owned;
        }
    }
}

/**
 * @brief Get the current BorrowState of @p name.
 * @return The tracked state, or BorrowState::Owned if @p name is not tracked
 * (treated as an untracked value that is assumed owned).
 */
BorrowState BorrowChecker::getState(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return BorrowState::Owned;  // Unknown = assume owned
    }
    return it->second.state;
}

/**
 * @brief True if @p name can currently be used, i.e. it has not been moved-from or dropped.
 *
 * Untracked names are assumed usable.
 */
bool BorrowChecker::canUse(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return true;  // Unknown = assume usable
    }
    return it->second.state != BorrowState::Moved &&
           it->second.state != BorrowState::Dropped;
}

/**
 * @brief True if @p name can currently be moved: owned, with no active shared or mutable borrows.
 *
 * Untracked names are assumed movable.
 */
bool BorrowChecker::canMove(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return true;
    }
    const auto& info = it->second;
    return info.state == BorrowState::Owned &&
           info.borrow_count == 0 &&
           !info.has_mutable_borrow;
}

/**
 * @brief True if @p name can currently be borrowed immutably: owned or already
 * shared-borrowed, and not mutably borrowed.
 *
 * Untracked names are assumed borrowable.
 */
bool BorrowChecker::canBorrowShared(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return true;
    }
    const auto& info = it->second;
    return (info.state == BorrowState::Owned ||
            info.state == BorrowState::BorrowedShared) &&
           !info.has_mutable_borrow;
}

/**
 * @brief True if @p name can currently be borrowed mutably: owned, with no
 * active shared or mutable borrows.
 *
 * Untracked names are assumed borrowable.
 */
bool BorrowChecker::canBorrowMut(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return true;
    }
    const auto& info = it->second;
    return info.state == BorrowState::Owned &&
           info.borrow_count == 0 &&
           !info.has_mutable_borrow;
}

/**
 * @brief Record a borrow-checking violation of the given @p kind for variable @p var.
 */
void BorrowChecker::addError(BorrowError::Kind kind, const std::string& var,
                              const std::string& msg) {
    errors_.push_back({kind, var, msg});
}

// ============================================================================
// Context Linear Methods (Phase 6)
// ============================================================================

/**
 * @brief Bind @p name to @p type in the innermost scope and mark it as a
 * linear variable (must be used exactly once).
 *
 * Combines an ordinary Context::bind() with registering @p name in the
 * linear-variable tracking set with a fresh usage count of 0.
 */
void Context::bindLinear(const std::string& name, TypeId type) {
    // Bind the variable normally
    bind(name, type);
    // Mark it as linear
    linear_vars_.insert(name);
    linear_usage_count_[name] = 0;  // Unused initially
}

/**
 * @brief Record a use of the linear variable @p name, incrementing its usage count.
 *
 * No-op if @p name was not bound via bindLinear(). A count greater than 1
 * indicates a linearity violation, surfaced via getOverusedLinear().
 */
void Context::useLinear(const std::string& name) {
    if (linear_vars_.count(name) > 0) {
        linear_usage_count_[name]++;
    }
}

/**
 * @brief Mark the linear variable @p name as consumed.
 *
 * Currently identical to useLinear(); kept as a distinctly named entry point
 * for call sites that consume a linear binding outright (e.g. at let scope
 * exit).
 */
void Context::consumeLinear(const std::string& name) {
    // Same as useLinear for tracking
    useLinear(name);
}

/**
 * @brief True if @p name was bound via bindLinear().
 */
bool Context::isLinear(const std::string& name) const {
    return linear_vars_.count(name) > 0;
}

/**
 * @brief True if the linear variable @p name has been used at least once.
 */
bool Context::isLinearUsed(const std::string& name) const {
    auto it = linear_usage_count_.find(name);
    return it != linear_usage_count_.end() && it->second > 0;
}

/**
 * @brief Get the names of linear variables in scope that were never used.
 * @return Names whose usage count is zero (or has no recorded entry).
 */
std::vector<std::string> Context::getUnusedLinear() const {
    std::vector<std::string> result;
    for (const auto& name : linear_vars_) {
        auto it = linear_usage_count_.find(name);
        if (it == linear_usage_count_.end() || it->second == 0) {
            result.push_back(name);
        }
    }
    return result;
}

/**
 * @brief Get the names of linear variables in scope that were used more than once.
 * @return Names whose usage count exceeds 1.
 */
std::vector<std::string> Context::getOverusedLinear() const {
    std::vector<std::string> result;
    for (const auto& name : linear_vars_) {
        auto it = linear_usage_count_.find(name);
        if (it != linear_usage_count_.end() && it->second > 1) {
            result.push_back(name);
        }
    }
    return result;
}

/**
 * @brief True if every linear variable currently in scope was used exactly once.
 */
bool Context::checkLinearConstraints() const {
    for (const auto& name : linear_vars_) {
        auto it = linear_usage_count_.find(name);
        if (it == linear_usage_count_.end() || it->second != 1) {
            return false;  // Not used exactly once
        }
    }
    return true;
}

// ============================================================================
// TypeChecker Implementation
// ============================================================================

/**
 * @brief Construct a type checker over @p env with the given strictness/safety mode.
 * @param env Type environment providing type registration/lookup/subtyping.
 * @param strict_types If true, reported type issues call eshkol_error() and
 * abort compilation; if false, they call eshkol_warn() and compilation
 * continues (gradual typing).
 * @param unsafe_mode If true, the checker starts with type/linear/borrow
 * issues silently ignored.
 */
TypeChecker::TypeChecker(TypeEnvironment& env, bool strict_types, bool unsafe_mode)
    : strict_types_(strict_types), unsafe_mode_(unsafe_mode), env_(env) {}

/**
 * @brief Store a successful result's inferred type into @p expr->inferred_hott_type, then return the result unchanged.
 *
 * Common tail call used by the synthesize()/check() helpers so every
 * synthesis/checking rule records its result type on the AST node it typed.
 * Failed results pass through without touching @p expr.
 */
static TypeCheckResult storeAndReturn(eshkol_ast_t* expr, TypeCheckResult result) {
    if (result.success && expr) {
        expr->inferred_hott_type = result.inferred_type.pack();
    }
    return result;
}

/**
 * @brief Build a TypeCheckResult::error() using @p expr's source line/column, if available.
 *
 * Falls back to an error with no location (line/column 0) if @p expr is null.
 */
static TypeCheckResult errorAt(const eshkol_ast_t* expr, const std::string& msg) {
    if (expr) {
        return TypeCheckResult::error(msg, expr->line, expr->column);
    }
    return TypeCheckResult::error(msg);
}

/**
 * @brief Bind @p name in @p ctx to a synthetic function type of the given @p arity.
 *
 * Used to seed the context with the signatures of well-known stdlib
 * procedures (see bindKnownRequireExports()) whose parameter/return types
 * are approximated as BuiltinTypes::Value, so calls to them type-check
 * without requiring full stdlib type declarations.
 */
static void bindKnownProcedure(Context& ctx,
                               TypeEnvironment& env,
                               const char* name,
                               size_t arity,
                               bool is_variadic = false) {
    std::vector<TypeId> params(arity, BuiltinTypes::Value);
    ctx.bind(name, env.makeFunctionType(params, BuiltinTypes::Value, is_variadic));
}

/**
 * @brief True if a `require` operation @p op lists @p module_name among its imported modules.
 */
static bool requireNamesModule(const eshkol_operation& op, const char* module_name) {
    for (uint64_t i = 0; i < op.require_op.num_modules; ++i) {
        const char* candidate = op.require_op.module_names[i];
        if (candidate && std::strcmp(candidate, module_name) == 0) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Bind the known exported procedures of a `require`d module into @p ctx.
 *
 * Given a `require` operation @p op, checks whether it imports the "stdlib"
 * or "core.list.higher_order" module and, if so, binds the well-known
 * higher-order list procedures (map1/map2/map3, fold variants, for-each,
 * any, every) via bindKnownProcedure() so subsequent calls to them
 * type-check even though the stdlib's own type declarations are not loaded
 * during isolated type checking.
 */
static void bindKnownRequireExports(Context& ctx,
                                    TypeEnvironment& env,
                                    const eshkol_operation& op) {
    const bool imports_stdlib = requireNamesModule(op, "stdlib");
    const bool imports_higher_order =
        imports_stdlib || requireNamesModule(op, "core.list.higher_order");

    if (imports_higher_order) {
        bindKnownProcedure(ctx, env, "map1", 2);
        bindKnownProcedure(ctx, env, "map2", 3);
        bindKnownProcedure(ctx, env, "map3", 4);
        bindKnownProcedure(ctx, env, "fold", 3);
        bindKnownProcedure(ctx, env, "fold-left", 3);
        bindKnownProcedure(ctx, env, "foldl", 3);
        bindKnownProcedure(ctx, env, "fold-right", 3);
        bindKnownProcedure(ctx, env, "foldr", 3);
        bindKnownProcedure(ctx, env, "for-each", 2);
        bindKnownProcedure(ctx, env, "any", 2);
        bindKnownProcedure(ctx, env, "every", 2);
    }
}

/**
 * @brief Synthesis-mode entry point: infer the type of @p expr bottom-up (Γ ⊢ e ⇒ τ).
 *
 * Dispatches on @p expr's AST node kind to the appropriate synthesize*
 * helper (literals/symbols to synthesizeLiteral(), variable references to
 * synthesizeVariable(), operation forms to synthesizeOperation(), lambdas to
 * synthesizeLambda(), and cons cells trivially to BuiltinTypes::List). On
 * success, stores the inferred type into @p expr->inferred_hott_type via
 * storeAndReturn().
 * @return A successful TypeCheckResult carrying the inferred type, or a
 * failing result (e.g. null expression, or an AST node kind with no
 * synthesis rule).
 */
TypeCheckResult TypeChecker::synthesize(eshkol_ast_t* expr) {
    if (!expr) {
        return TypeCheckResult::error("Null expression");
    }

    TypeCheckResult result;
    switch (expr->type) {
        case ESHKOL_INT64:
        case ESHKOL_DOUBLE:
        case ESHKOL_STRING:
        case ESHKOL_BOOL:
        case ESHKOL_NULL:
        case ESHKOL_CHAR:
        case ESHKOL_BIGNUM_LITERAL:
        case ESHKOL_SYMBOL:
            result = synthesizeLiteral(expr);
            break;

        case ESHKOL_VAR:
            result = synthesizeVariable(expr);
            break;

        case ESHKOL_OP:
            result = synthesizeOperation(expr);
            break;

        case ESHKOL_FUNC:
            result = synthesizeLambda(expr);
            break;

        case ESHKOL_CONS:
            // Cons cell - synthesize as list
            result = TypeCheckResult::ok(BuiltinTypes::List);
            break;

        default:
            return TypeCheckResult::error("Cannot synthesize type for this expression");
    }

    return storeAndReturn(expr, result);
}

/**
 * @brief Checking-mode entry point: verify @p expr has the expected type, top-down (Γ ⊢ e ⇐ τ).
 *
 * Lambdas get a dedicated checking rule (checkLambda()) so their parameter
 * types can be inferred from @p expected rather than requiring annotations.
 * All other expression kinds fall back to synthesizing @p expr's type and
 * verifying it is a subtype of @p expected via TypeEnvironment::isSubtype().
 * @param expr The expression to check.
 * @param expected The type @p expr is expected to have.
 * @return TypeCheckResult::ok(expected) on success; a type-mismatch error
 * (also recorded via addTypeMismatch()) if synthesis succeeds but the
 * inferred type is not a subtype of @p expected; or the failing result from
 * synthesis if that fails first.
 */
TypeCheckResult TypeChecker::check(eshkol_ast_t* expr, TypeId expected) {
    if (!expr) {
        return TypeCheckResult::error("Null expression");
    }

    // Special case: lambda can infer param types from expected
    if (expr->type == ESHKOL_OP &&
        expr->operation.op == ESHKOL_LAMBDA_OP) {
        return checkLambda(expr, expected);
    }

    // General case: synthesize then check subtyping
    auto result = synthesize(expr);
    if (!result.success) {
        return result;
    }

    if (env_.isSubtype(result.inferred_type, expected)) {
        return TypeCheckResult::ok(expected);
    }

    addTypeMismatch(expected, result.inferred_type, expr->line, expr->column);
    return errorAt(expr,
        "Type mismatch: expected " + env_.getTypeName(expected) +
        ", got " + env_.getTypeName(result.inferred_type));
}

/**
 * @brief Synthesis rule for self-evaluating literals and bare symbols.
 *
 * Maps each literal AST node kind to its builtin type: ESHKOL_INT64 ->
 * Int64, ESHKOL_BIGNUM_LITERAL -> BigInt, ESHKOL_DOUBLE -> Float64,
 * ESHKOL_STRING -> String, ESHKOL_BOOL -> Boolean, ESHKOL_NULL -> Null,
 * ESHKOL_CHAR -> Char, and ESHKOL_SYMBOL -> Symbol.
 * @return TypeCheckResult::ok() with the corresponding builtin type, or an
 * error if @p expr->type is not one of the recognized literal kinds.
 */
TypeCheckResult TypeChecker::synthesizeLiteral(eshkol_ast_t* expr) {
    switch (expr->type) {
        case ESHKOL_INT64:
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        case ESHKOL_BIGNUM_LITERAL:
            return TypeCheckResult::ok(BuiltinTypes::BigInt);

        case ESHKOL_DOUBLE:
            return TypeCheckResult::ok(BuiltinTypes::Float64);

        case ESHKOL_STRING:
            return TypeCheckResult::ok(BuiltinTypes::String);

        case ESHKOL_BOOL:
            return TypeCheckResult::ok(BuiltinTypes::Boolean);

        case ESHKOL_NULL:
            return TypeCheckResult::ok(BuiltinTypes::Null);

        case ESHKOL_CHAR:
            return TypeCheckResult::ok(BuiltinTypes::Char);

        case ESHKOL_SYMBOL:
            return TypeCheckResult::ok(BuiltinTypes::Symbol);

        default:
            return TypeCheckResult::error("Unknown literal type");
    }
}

/**
 * @brief Synthesis rule for variable references (Γ ⊢ x ⇒ τ).
 *
 * Looks up the variable's name in the context. If unbound, special-cases a
 * few builtins used as first-class values: the arithmetic operators
 * (+, -, *, /) synthesize a variadic-looking (Int64, Int64) -> Int64
 * function type, and the I/O procedures (display, write, newline)
 * synthesize a (Value) -> Null function type, matching how codegen wraps
 * them as unary closures. Any other unbound name is an error. If the
 * variable is bound and tracked as linear (Context::isLinear()), records a
 * use via Context::useLinear() and reports a type issue if it was already
 * used (linearity violation).
 * @return The variable's bound type (or the synthesized builtin function
 * type), or an error if the name is unbound and not a recognized builtin.
 */
TypeCheckResult TypeChecker::synthesizeVariable(eshkol_ast_t* expr) {
    if (!expr->variable.id) {
        return errorAt(expr, "Variable has no name");
    }

    std::string name = expr->variable.id;
    auto type = ctx_.lookup(name);
    if (!type) {
        // Check if it's a builtin arithmetic operator (used as first-class value)
        // These are valid but not in the type context when used as values
        if (name == "+" || name == "-" || name == "*" || name == "/") {
            // Return a function type (variadic, returns number)
            // For now, return a simple (Number, Number) -> Number
            // Using Int64 as the base numeric type
            return TypeCheckResult::ok(
                env_.makeFunctionType({BuiltinTypes::Int64, BuiltinTypes::Int64}, BuiltinTypes::Int64));
        }
        // Quirk 11: I/O builtins as first-class values (display, write,
        // newline). Codegen wraps each as a unary closure in codegenVariable;
        // the type checker just needs to agree they're callable.
        if (name == "display" || name == "write" || name == "newline") {
            return TypeCheckResult::ok(
                env_.makeFunctionType({BuiltinTypes::Value}, BuiltinTypes::Null));
        }
        return errorAt(expr, "Unbound variable: " + name);
    }

    // Track linear variable usage
    if (ctx_.isLinear(name)) {
        if (ctx_.isLinearUsed(name)) {
            reportTypeIssue("linear variable '" + name + "' used more than once", expr);
        }
        ctx_.useLinear(name);
    }

    return TypeCheckResult::ok(*type);
}

/**
 * @brief Synthesis rule for ESHKOL_OP nodes: dispatches on the operation kind
 * (expr->operation.op) to the appropriate typing rule.
 *
 * Delegates the structurally interesting forms to dedicated helpers:
 * `define` to synthesizeDefine(), `let`/`let*`/`letrec`/`letrec*` to
 * synthesizeLet(), `if` to synthesizeIf(), `lambda`/`case-lambda` to
 * synthesizeLambda(), and function calls to synthesizeApplication().
 * `define-type` is handled inline: it registers the named (possibly
 * parameterized) type alias into the context via Context::defineTypeAlias()
 * and synthesizes BuiltinTypes::Null. `require` binds the known exported
 * procedures of imported stdlib modules via bindKnownRequireExports() and
 * also synthesizes Null. Arithmetic (+, -, *, /) synthesizes its first two
 * operands and returns their arithmetic-promoted type via
 * TypeEnvironment::promoteForArithmetic(); `begin`/sequence returns the type
 * of its last sub-expression (or Null if empty). All remaining operation
 * kinds (tensors, booleans, predicates, control flow, continuations,
 * multiple values, quoting, logic/consciousness/DNC constructors, memory
 * management, extern declarations, macro/syntax forms, and the catch-all
 * default) synthesize a fixed builtin type — mostly BuiltinTypes::Value for
 * forms whose precise result type is branch- or runtime-dependent, Boolean
 * for predicates and and/or, and Function for calculus operators
 * (diff/gradient/jacobian/etc).
 * @return A successful TypeCheckResult per the rule above; this function has
 * no failure path of its own (unsuccessful results only propagate from
 * delegated helpers or nested synthesize() calls).
 */
TypeCheckResult TypeChecker::synthesizeOperation(eshkol_ast_t* expr) {
    switch (expr->operation.op) {
        case ESHKOL_DEFINE_OP:
            return synthesizeDefine(expr);

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            return synthesizeLet(expr);

        case ESHKOL_IF_OP:
            return synthesizeIf(expr);

        case ESHKOL_LAMBDA_OP:
        case ESHKOL_CASE_LAMBDA_OP:
            return synthesizeLambda(expr);

        case ESHKOL_CALL_OP:
            return synthesizeApplication(expr);

        case ESHKOL_DEFINE_TYPE_OP: {
            // Type definition - register alias and return null
            if (expr->operation.define_type_op.name &&
                expr->operation.define_type_op.type_expr) {
                // Extract type parameters if present
                std::vector<std::string> type_params;
                for (uint64_t i = 0; i < expr->operation.define_type_op.num_type_params; i++) {
                    if (expr->operation.define_type_op.type_params[i]) {
                        type_params.push_back(expr->operation.define_type_op.type_params[i]);
                    }
                }
                ctx_.defineTypeAlias(
                    expr->operation.define_type_op.name,
                    expr->operation.define_type_op.type_expr,
                    type_params);
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        // Arithmetic operations
        case ESHKOL_ADD_OP:
        case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP:
        case ESHKOL_DIV_OP: {
            if (expr->operation.call_op.num_vars < 2) {
                return errorAt(expr, "Arithmetic requires at least 2 operands");
            }
            auto left = synthesize(&expr->operation.call_op.variables[0]);
            auto right = synthesize(&expr->operation.call_op.variables[1]);
            if (!left.success) return left;
            if (!right.success) return right;

            // Promote types
            TypeId result_type = env_.promoteForArithmetic(
                left.inferred_type, right.inferred_type);
            return TypeCheckResult::ok(result_type);
        }

        // Tensor operations
        case ESHKOL_TENSOR_OP:
            return TypeCheckResult::ok(BuiltinTypes::Vector);

        // Sequence returns last expression type
        case ESHKOL_SEQUENCE_OP:
            if (expr->operation.sequence_op.num_expressions > 0) {
                return synthesize(&expr->operation.sequence_op.expressions[expr->operation.sequence_op.num_expressions - 1]);
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);

        // Boolean operations
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            return TypeCheckResult::ok(BuiltinTypes::Boolean);

        // Predicate operations — always return Boolean
        case ESHKOL_LOGIC_VAR_PRED_OP:
        case ESHKOL_SUBSTITUTION_PRED_OP:
        case ESHKOL_KB_PRED_OP:
        case ESHKOL_FACT_PRED_OP:
        case ESHKOL_FACTOR_GRAPH_PRED_OP:
        case ESHKOL_WORKSPACE_PRED_OP:
            return TypeCheckResult::ok(BuiltinTypes::Boolean);

        // Side-effect operations — return Null (void)
        case ESHKOL_SET_OP:
        case ESHKOL_IMPORT_OP:
        case ESHKOL_PROVIDE_OP:
        case ESHKOL_KB_ASSERT_OP:
        case ESHKOL_FG_ADD_FACTOR_OP:
        case ESHKOL_WS_REGISTER_OP:
        case ESHKOL_DEFINE_SYNTAX_OP:
        case ESHKOL_DEFINE_RECORD_TYPE_OP:
        case ESHKOL_TYPE_ANNOTATION_OP:
            return TypeCheckResult::ok(BuiltinTypes::Null);

        case ESHKOL_REQUIRE_OP:
            bindKnownRequireExports(ctx_, env_, expr->operation);
            return TypeCheckResult::ok(BuiltinTypes::Null);

        // Calculus operations — return Function (closures over differentiated functions)
        case ESHKOL_DIFF_OP:
        case ESHKOL_DERIVATIVE_OP:
        case ESHKOL_GRADIENT_OP:
        case ESHKOL_JACOBIAN_OP:
        case ESHKOL_HESSIAN_OP:
        case ESHKOL_DIVERGENCE_OP:
        case ESHKOL_CURL_OP:
        case ESHKOL_LAPLACIAN_OP:
        case ESHKOL_DIRECTIONAL_DERIV_OP:
            return TypeCheckResult::ok(BuiltinTypes::Function);

        // Free energy operations — return scalar Float64
        case ESHKOL_FREE_ENERGY_OP:
        case ESHKOL_EXPECTED_FREE_ENERGY_OP:
            return TypeCheckResult::ok(BuiltinTypes::Float64);

        // Control flow — return Value (branch-dependent)
        case ESHKOL_COND_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_MATCH_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_DO_OP:
        case ESHKOL_GUARD_OP:
        case ESHKOL_RAISE_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Continuation/dynamic wind — return Value
        case ESHKOL_CALL_CC_OP:
        case ESHKOL_DYNAMIC_WIND_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Multiple values — return Value
        case ESHKOL_VALUES_OP:
        case ESHKOL_CALL_WITH_VALUES_OP:
        case ESHKOL_LET_VALUES_OP:
        case ESHKOL_LET_STAR_VALUES_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Quoting — return Value (any datum)
        case ESHKOL_QUOTE_OP:
        case ESHKOL_QUASIQUOTE_OP:
        case ESHKOL_UNQUOTE_OP:
        case ESHKOL_UNQUOTE_SPLICING_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Logic/consciousness constructors — return Value (opaque heap objects)
        case ESHKOL_LOGIC_VAR_OP:
        case ESHKOL_UNIFY_OP:
        case ESHKOL_WALK_OP:
        case ESHKOL_MAKE_SUBST_OP:
        case ESHKOL_MAKE_FACT_OP:
        case ESHKOL_MAKE_KB_OP:
        case ESHKOL_KB_QUERY_OP:
        case ESHKOL_KB_QUERY_PREFIX_OP:
        case ESHKOL_MAKE_FACTOR_GRAPH_OP:
        case ESHKOL_FG_INFER_OP:
        case ESHKOL_FG_UPDATE_CPT_OP:
        case ESHKOL_MAKE_WORKSPACE_OP:
        case ESHKOL_WS_STEP_OP:
        case ESHKOL_DNC_MAKE_OP:
        case ESHKOL_DNC_CONTENT_ADDR_OP:
        case ESHKOL_DNC_LOC_ADDR_OP:
        case ESHKOL_DNC_READ_OP:
        case ESHKOL_DNC_WRITE_OP:
        case ESHKOL_DNC_ALLOC_WEIGHTS_OP:
        case ESHKOL_DNC_READ_GRAD_OP:
        case ESHKOL_DNC_PRED_OP:
        case ESHKOL_SDNC_PROGRAM_OP:
        case ESHKOL_SDNC_RUN_OP:
        case ESHKOL_SDNC_WEIGHT_GRAD_OP:
        case ESHKOL_SDNC_PARAMS_OP:
        case ESHKOL_SDNC_SET_PARAMS_OP:
        case ESHKOL_SDNC_IMPROVE_OP:
        case ESHKOL_SDNC_PRED_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Memory management — return Value (identity-like wrappers)
        case ESHKOL_WITH_REGION_OP:
        case ESHKOL_OWNED_OP:
        case ESHKOL_MOVE_OP:
        case ESHKOL_BORROW_OP:
        case ESHKOL_SHARED_OP:
        case ESHKOL_WEAK_REF_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // External declarations — return Value
        case ESHKOL_EXTERN_OP:
        case ESHKOL_EXTERN_VAR_OP:
        case ESHKOL_COMPOSE_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        // Macro/syntax forms — return Value (should be expanded before reaching here)
        case ESHKOL_LET_SYNTAX_OP:
        case ESHKOL_LETREC_SYNTAX_OP:
        case ESHKOL_PARAMETERIZE_OP:
        case ESHKOL_MAKE_PARAMETER_OP:
        case ESHKOL_COND_EXPAND_OP:
        case ESHKOL_INCLUDE_OP:
        case ESHKOL_FORALL_OP:
        case ESHKOL_SYNTAX_ERROR_OP:
            return TypeCheckResult::ok(BuiltinTypes::Value);

        case ESHKOL_INVALID_OP:
        default:
            return TypeCheckResult::ok(BuiltinTypes::Value);
    }
}

/**
 * @brief Synthesis rule for lambda/case-lambda expressions (Γ ⊢ (lambda ...) ⇒ τ).
 *
 * Pushes a fresh scope and binds each parameter: its type comes from an
 * explicit annotation (resolveType()) if present, otherwise defaults to
 * BuiltinTypes::Value; parameters whose type carries TYPE_FLAG_LINEAR are
 * bound via Context::bindLinear() so single-use is enforced. The body is
 * then synthesized in this scope. Before popping the scope, verifies linear
 * parameter usage via Context::checkLinearConstraints(), reporting a type
 * issue for every unused or overused linear parameter. If the lambda has a
 * return type annotation, checks the synthesized body type is a subtype of
 * it (failing otherwise) and uses the annotated type for the result;
 * otherwise uses the body's inferred type directly.
 * @return A function type (via TypeEnvironment::makeFunctionType()) from the
 * parameter types to the return type, respecting lambda.is_variadic; or an
 * error if the body fails to synthesize or violates the return annotation.
 */
TypeCheckResult TypeChecker::synthesizeLambda(eshkol_ast_t* expr) {
    const auto& lambda = expr->operation.lambda_op;

    ctx_.pushScope();

    // Collect parameter types and bind parameters in context
    std::vector<TypeId> param_types;
    for (size_t i = 0; i < lambda.num_params; i++) {
        std::string param_name = lambda.parameters[i].variable.id;

        // Get type from annotation if available
        TypeId param_type = BuiltinTypes::Value;  // Default
        if (lambda.param_types && lambda.param_types[i]) {
            param_type = resolveType(lambda.param_types[i]);
        }

        param_types.push_back(param_type);

        // Register linear parameters for use-once checking
        if (param_type.flags & TYPE_FLAG_LINEAR) {
            ctx_.bindLinear(param_name, param_type);
        } else {
            ctx_.bind(param_name, param_type);
        }
    }

    // Synthesize body type
    auto body_result = synthesize(lambda.body);

    // Check linear variable constraints before popping scope
    if (!ctx_.checkLinearConstraints()) {
        auto unused = ctx_.getUnusedLinear();
        for (const auto& name : unused) {
            reportTypeIssue("linear variable '" + name + "' was not consumed", expr);
        }
        auto overused = ctx_.getOverusedLinear();
        for (const auto& name : overused) {
            reportTypeIssue("linear variable '" + name + "' was consumed more than once", expr);
        }
    }

    ctx_.popScope();

    if (!body_result.success) {
        return body_result;
    }

    // Determine return type
    TypeId return_type = body_result.inferred_type;

    // If return type annotation exists, check against it
    if (lambda.return_type) {
        TypeId expected_return = resolveType(lambda.return_type);
        if (!env_.isSubtype(body_result.inferred_type, expected_return)) {
            return TypeCheckResult::error(
                "Lambda body type " + env_.getTypeName(body_result.inferred_type) +
                " doesn't match return annotation " + env_.getTypeName(expected_return));
        }
        return_type = expected_return;  // Use annotated type for precision
    }

    // Create proper function type encoding param types and return type
    if (param_types.empty()) {
        // Nullary function
        return TypeCheckResult::ok(env_.makeFunctionType({}, return_type,
                                                         lambda.is_variadic));
    }

    return TypeCheckResult::ok(env_.makeFunctionType(param_types, return_type,
                                                      lambda.is_variadic));
}

/**
 * @brief Synthesis rule for function application, ESHKOL_CALL_OP (Γ ⊢ (f a...) ⇒ τ).
 *
 * This is the largest and most detail-heavy synthesis rule in the checker:
 * it implements per-builtin typing for essentially all of Eshkol's
 * primitive procedures, then falls back to generic function-type lookup for
 * user-defined calls and inline lambdas.
 *
 * Structure:
 * - If there is no callee expression, synthesizes BuiltinTypes::Value.
 * - Determines whether the callee is a named builtin (a bare ESHKOL_VAR),
 *   and computes should_skip_builtin_designator_arg(), a local predicate
 *   identifying positional arguments that are compile-time designators
 *   rather than ordinary values (e.g. the ordering/width argument of
 *   atomic-, volatile-, target-intrinsic builtins) so those slots are not
 *   mistakenly type-checked as values.
 * - Pre-synthesizes every argument (skipping designator slots, which are
 *   given BuiltinTypes::Value) so nested sub-expressions are always
 *   type-checked even when the callee's own return type doesn't depend on
 *   argument types (e.g. `(display (vector-ref v -1))` still checks the
 *   vector-ref).
 * - If the callee is a recognized builtin name, dispatches through a long
 *   chain of name comparisons covering: arithmetic (+,-,*,/) with backward
 *   inference that narrows Value-typed variable operands to Number;
 *   comparison and numeric predicates (returns Boolean, also narrowing
 *   operands to Number); polymorphic type predicates (equal?, pair?,
 *   number?, ...); list operations with parametric Pair<A,B> tracking for
 *   car/cdr/cons (including the R7RS proper-vs-improper list narrowing rule
 *   for cons); string operations (with Value->String narrowing); numeric
 *   library functions (abs, sqrt, trig, floor/ceiling/round, expt, modulo,
 *   etc.); low-level FFI/memory intrinsics (volatile-load/store,
 *   atomic-load/store/exchange/compare-exchange/fetch-*, target-intrinsic,
 *   compiler-fence, memory-fence, addr-of, null-ptr, ptr->usize,
 *   usize->ptr, ptr-add); vector operations (vector-ref/length/set!/copy/
 *   append, list<->vector conversions); port and I/O predicates/procedures;
 *   promises; bytevectors; tensor/model save-load; rational and complex
 *   number constructors/accessors; DSP operations (fft, filters, window
 *   functions, convolution); numerical optimization (gradient-descent,
 *   adam, l-bfgs, conjugate-gradient, line-search, tensor-dot/norm/svd);
 *   display/newline/write, set!, begin, textual/binary I/O; and the
 *   not/and/or and `if`-as-procedure forms, with `if` computing the least
 *   common supertype of its branches via TypeEnvironment::leastCommonSupertype().
 * - If the callee is not (or is no longer, having fallen through) a
 *   recognized builtin, resolves its function type by looking it up in the
 *   context (named callee) or synthesizing it (inline lambda callee). If
 *   that resolves to a function type, checks argument count against the
 *   PiType (skipped for variadic callees) and checks each argument's
 *   synthesized type against the parameter type (allowing Value on either
 *   side, and using leastCommonSupertype() for compatibility), reporting a
 *   type issue via reportTypeIssue() for any mismatch, then returns the
 *   function's return type.
 * @return A TypeCheckResult per the builtin/generic rule above.
 * BuiltinTypes::Value is used as the fallback whenever the callee's
 * signature or the call itself cannot be determined precisely.
 */
TypeCheckResult TypeChecker::synthesizeApplication(eshkol_ast_t* expr) {
    const auto& call = expr->operation.call_op;

    // The function being called is in call.func, arguments are in call.variables
    const eshkol_ast_t* func_expr = call.func;

    // If we have no function expression, return Value
    if (!func_expr) {
        return TypeCheckResult::ok(BuiltinTypes::Value);
    }

    std::string builtin_name;
    const bool has_builtin_name = func_expr->type == ESHKOL_VAR && func_expr->variable.id;
    if (has_builtin_name) {
        builtin_name = func_expr->variable.id;
    }

    const auto should_skip_builtin_designator_arg = [&](size_t index) {
        if (!has_builtin_name) {
            return false;
        }

        if (builtin_name == "compiler-fence" || builtin_name == "memory-fence" ||
            builtin_name == "volatile-load" || builtin_name == "volatile-store!") {
            return index == 0;
        }

        if (builtin_name == "atomic-load") {
            return index == 0 || index == 2;
        }

        if (builtin_name == "atomic-store!" || builtin_name == "atomic-exchange!" ||
            builtin_name == "atomic-fetch-add!" || builtin_name == "atomic-fetch-sub!" ||
            builtin_name == "atomic-fetch-and!" || builtin_name == "atomic-fetch-or!" ||
            builtin_name == "atomic-fetch-xor!") {
            return index == 0 || index == 3;
        }

        if (builtin_name == "atomic-compare-exchange!") {
            return index == 0 || index == 4 || index == 5;
        }

        if (builtin_name == "target-intrinsic") {
            return index == 0 || (index >= 2 && (index % 2) == 0);
        }

        return false;
    };

    // Pre-synthesize all arguments to ensure nested expressions are type-checked
    // (e.g., (display (vector-ref v -1)) must check the vector-ref even though
    // display itself doesn't need arg types for its return type)
    std::vector<TypeCheckResult> arg_types;
    for (size_t i = 0; i < call.num_vars; ++i) {
        if (should_skip_builtin_designator_arg(i)) {
            arg_types.push_back(TypeCheckResult::ok(BuiltinTypes::Value));
            continue;
        }
        arg_types.push_back(synthesize(const_cast<eshkol_ast_t*>(&call.variables[i])));
    }

    // Check for builtin operators first
    if (func_expr->type == ESHKOL_VAR && func_expr->variable.id) {
        std::string func_name = func_expr->variable.id;

        // Arithmetic operators: use pre-synthesized operand types and promote
        if (func_name == "+" || func_name == "-" || func_name == "*" || func_name == "/") {
            // Narrow Value-typed variable arguments to Number (backward type inference)
            for (size_t i = 0; i < call.num_vars; ++i) {
                if (call.variables[i].type == ESHKOL_VAR && call.variables[i].variable.id) {
                    auto cur_type = ctx_.lookup(call.variables[i].variable.id);
                    if (cur_type && *cur_type == BuiltinTypes::Value) {
                        ctx_.bind(call.variables[i].variable.id, BuiltinTypes::Number);
                    }
                }
            }
            if (call.num_vars >= 2 && arg_types[0].success && arg_types[1].success) {
                TypeId result_type = env_.promoteForArithmetic(
                    arg_types[0].inferred_type, arg_types[1].inferred_type);
                return TypeCheckResult::ok(result_type);
            }
            return TypeCheckResult::ok(BuiltinTypes::Number);
        }

        // Comparison operators return Boolean
        if (func_name == "<" || func_name == ">" || func_name == "<=" ||
            func_name == ">=" || func_name == "=" || func_name == "!=" ||
            func_name == "zero?" || func_name == "positive?" || func_name == "negative?" ||
            func_name == "odd?" || func_name == "even?") {
            // Narrow numeric comparison arguments to Number
            for (size_t i = 0; i < call.num_vars; ++i) {
                if (call.variables[i].type == ESHKOL_VAR && call.variables[i].variable.id) {
                    auto cur_type = ctx_.lookup(call.variables[i].variable.id);
                    if (cur_type && *cur_type == BuiltinTypes::Value) {
                        ctx_.bind(call.variables[i].variable.id, BuiltinTypes::Number);
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        // Polymorphic predicates (work on any type, no narrowing)
        if (func_name == "equal?" || func_name == "eq?" || func_name == "eqv?" ||
            func_name == "null?" || func_name == "pair?" || func_name == "list?" ||
            func_name == "number?" || func_name == "string?" || func_name == "symbol?" ||
            func_name == "boolean?" || func_name == "vector?" || func_name == "procedure?" ||
            func_name == "integer?" || func_name == "real?" || func_name == "exact?" ||
            func_name == "inexact?" || func_name == "complex?" || func_name == "bignum?" ||
            func_name == "exact-integer?" || func_name == "char?" || func_name == "port?" ||
            func_name == "input-port?" || func_name == "output-port?" ||
            func_name == "eof-object?" || func_name == "bytevector?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // List operations — with parametric pair type tracking
        if (func_name == "car" || func_name == "first") {
            // If argument is a tracked Pair<A,B>, return A
            if (call.num_vars >= 1 && arg_types[0].success) {
                auto elems = env_.getPairElementTypes(arg_types[0].inferred_type);
                if (elems) {
                    return TypeCheckResult::ok(elems->first);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "cdr" || func_name == "rest") {
            // If argument is a tracked Pair<A,B>, return B (precise element type)
            if (call.num_vars >= 1 && arg_types[0].success) {
                auto elems = env_.getPairElementTypes(arg_types[0].inferred_type);
                if (elems) {
                    return TypeCheckResult::ok(elems->second);
                }
                // cdr of a known List → List (proper list tail)
                TypeId arg_type = arg_types[0].inferred_type;
                if (arg_type == BuiltinTypes::List) {
                    return TypeCheckResult::ok(BuiltinTypes::List);
                }
                // cdr of bare Pair or Value → Value (could be anything)
                if (arg_type == BuiltinTypes::Value || arg_type == BuiltinTypes::Pair) {
                    return TypeCheckResult::ok(BuiltinTypes::Value);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::List);
        }
        if (func_name == "cons") {
            // R7RS: a list is either '() or (cons X list). When the cdr is
            // itself a List (or '()), (cons car cdr) is a proper List —
            // not Pair<A, List>. Otherwise it's a (potentially improper)
            // pair with distinct head/tail types.
            //
            // Quirk 1 (2026-04-24): previously always returned Pair<A, B>,
            // so the idiomatic accumulator pattern
            //   (let loop ((xs ...) (acc (list)))
            //     (loop (cdr xs) (cons (... ...) acc)))
            // produced a spurious "expected List, got Pair<List, List>"
            // warning on every run when the loop's `acc` parameter was
            // inferred as List but the recursive call passed a
            // (cons _ acc) — which the old rule typed as Pair<_, List>.
            if (call.num_vars >= 2 && arg_types[0].success && arg_types[1].success) {
                TypeId cdr_type = arg_types[1].inferred_type;
                // Narrow to List when the cdr is a List / Null.
                if (cdr_type == BuiltinTypes::List ||
                    cdr_type == BuiltinTypes::Null) {
                    return TypeCheckResult::ok(BuiltinTypes::List);
                }
                TypeId pair_type = env_.makePairType(
                    arg_types[0].inferred_type, cdr_type);
                return TypeCheckResult::ok(pair_type);
            }
            return TypeCheckResult::ok(BuiltinTypes::Pair);
        }
        if (func_name == "list" || func_name == "append" ||
            func_name == "reverse" || func_name == "map" || func_name == "filter" ||
            func_name == "reduce" || func_name == "for-each" ||
            func_name == "list-copy" || func_name == "list-set!") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }
        if (func_name == "length") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }

        // String operations — narrow Value-typed string arguments
        if (func_name == "string-append" || func_name == "substring" ||
            func_name == "string-upcase" || func_name == "string-downcase" ||
            func_name == "number->string" || func_name == "symbol->string") {
            // Narrow string arguments (not number->string which takes a number)
            if (func_name != "number->string" && func_name != "symbol->string") {
                for (size_t i = 0; i < call.num_vars; ++i) {
                    if (call.variables[i].type == ESHKOL_VAR && call.variables[i].variable.id) {
                        auto cur_type = ctx_.lookup(call.variables[i].variable.id);
                        if (cur_type && *cur_type == BuiltinTypes::Value) {
                            ctx_.bind(call.variables[i].variable.id, BuiltinTypes::String);
                        }
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::String);
        }
        if (func_name == "string-length" || func_name == "string-byte-length" || func_name == "string->number") {
            // Narrow first arg to String
            if (call.num_vars >= 1 && call.variables[0].type == ESHKOL_VAR && call.variables[0].variable.id) {
                auto cur_type = ctx_.lookup(call.variables[0].variable.id);
                if (cur_type && *cur_type == BuiltinTypes::Value) {
                    ctx_.bind(call.variables[0].variable.id, BuiltinTypes::String);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "string=?" || func_name == "string<?" || func_name == "string>?") {
            // Narrow both args to String
            for (size_t i = 0; i < call.num_vars; ++i) {
                if (call.variables[i].type == ESHKOL_VAR && call.variables[i].variable.id) {
                    auto cur_type = ctx_.lookup(call.variables[i].variable.id);
                    if (cur_type && *cur_type == BuiltinTypes::Value) {
                        ctx_.bind(call.variables[i].variable.id, BuiltinTypes::String);
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // Math functions
        if (func_name == "abs" || func_name == "sqrt" || func_name == "exp" ||
            func_name == "log" || func_name == "sin" || func_name == "cos" ||
            func_name == "tan" || func_name == "asin" || func_name == "acos" ||
            func_name == "atan" || func_name == "floor" || func_name == "ceiling" ||
            func_name == "round" || func_name == "truncate" || func_name == "min" ||
            func_name == "max" || func_name == "expt" || func_name == "modulo" ||
            func_name == "remainder" || func_name == "quotient") {
            // Use pre-synthesized argument to determine if result is int or float
            if (call.num_vars >= 1 && arg_types[0].success) {
                // Most math functions return Float64, some preserve int
                if (func_name == "abs" || func_name == "min" || func_name == "max" ||
                    func_name == "modulo" || func_name == "remainder" || func_name == "quotient") {
                    return TypeCheckResult::ok(arg_types[0].inferred_type);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Float64);
        }

        // Low-level pointer conversions
        auto resolve_low_level_type_designator =
            [&](const eshkol_ast_t& type_arg, const char* builtin,
                const char* role, bool allow_null) -> std::optional<TypeId> {
            if (type_arg.type != ESHKOL_VAR || !type_arg.variable.id) {
                reportTypeIssue(std::string(builtin) + " expects a " + role + " type name",
                                &type_arg);
                return std::nullopt;
            }

            auto type_id = env_.lookupType(type_arg.variable.id);
            if (!type_id) {
                reportTypeIssue(std::string(builtin) + " received unknown " + role + " type '" +
                                    std::string(type_arg.variable.id) + "'",
                                &type_arg);
                return std::nullopt;
            }

            switch (type_id->id) {
                case BuiltinTypes::Int8.id:
                case BuiltinTypes::Int16.id:
                case BuiltinTypes::Int32.id:
                case BuiltinTypes::Int64.id:
                case BuiltinTypes::ISize.id:
                case BuiltinTypes::UInt8.id:
                case BuiltinTypes::UInt16.id:
                case BuiltinTypes::UInt32.id:
                case BuiltinTypes::UInt64.id:
                case BuiltinTypes::USize.id:
                case BuiltinTypes::Pointer.id:
                    return *type_id;
                case BuiltinTypes::Null.id:
                    if (allow_null) {
                        return *type_id;
                    }
                    [[fallthrough]];
                default:
                    reportTypeIssue(std::string(builtin) + " does not support " + role +
                                        " type '" + std::string(type_arg.variable.id) + "'",
                                    &type_arg);
                    return std::nullopt;
            }
        };

        auto resolve_fence_ordering =
            [&](const eshkol_ast_t& ordering_arg, const char* builtin) -> bool {
            if (ordering_arg.type != ESHKOL_VAR || !ordering_arg.variable.id) {
                reportTypeIssue(std::string(builtin) +
                                    " expects a fence ordering as its first argument",
                                &ordering_arg);
                return false;
            }

            const std::string ordering_name = ordering_arg.variable.id;
            if (ordering_name == "acquire" ||
                ordering_name == "release" ||
                ordering_name == "acq-rel" ||
                ordering_name == "seq-cst") {
                return true;
            }

            reportTypeIssue(std::string(builtin) + " does not support fence ordering '" +
                                ordering_name + "'",
                            &ordering_arg);
            return false;
        };

        auto resolve_atomic_ordering =
            [&](const eshkol_ast_t& ordering_arg, const char* builtin,
                bool for_store) -> bool {
            if (ordering_arg.type != ESHKOL_VAR || !ordering_arg.variable.id) {
                reportTypeIssue(std::string(builtin) +
                                    " expects a memory ordering designator",
                                &ordering_arg);
                return false;
            }

            const std::string ordering_name = ordering_arg.variable.id;
            if (ordering_name == "relaxed" || ordering_name == "seq-cst") {
                return true;
            }
            if (!for_store && ordering_name == "acquire") {
                return true;
            }
            if (for_store && ordering_name == "release") {
                return true;
            }

            reportTypeIssue(std::string(builtin) + " does not support memory ordering '" +
                                ordering_name + "' for " +
                                (for_store ? "atomic stores" : "atomic loads"),
                            &ordering_arg);
            return false;
        };

        auto resolve_atomic_rmw_ordering =
            [&](const eshkol_ast_t& ordering_arg, const char* builtin) -> bool {
            if (ordering_arg.type != ESHKOL_VAR || !ordering_arg.variable.id) {
                reportTypeIssue(std::string(builtin) +
                                    " expects a memory ordering designator",
                                &ordering_arg);
                return false;
            }

            const std::string ordering_name = ordering_arg.variable.id;
            if (ordering_name == "relaxed" ||
                ordering_name == "acquire" ||
                ordering_name == "release" ||
                ordering_name == "acq-rel" ||
                ordering_name == "seq-cst") {
                return true;
            }

            reportTypeIssue(std::string(builtin) + " does not support memory ordering '" +
                                ordering_name + "' for atomic read-modify-write operations",
                            &ordering_arg);
            return false;
        };

        auto atomic_cmpxchg_failure_ordering_allowed =
            [](const std::string& success_name,
               const std::string& failure_name) -> bool {
            if (failure_name == "release" || failure_name == "acq-rel") {
                return false;
            }
            if (success_name == "relaxed") {
                return failure_name == "relaxed";
            }
            if (success_name == "acquire") {
                return failure_name == "relaxed" || failure_name == "acquire";
            }
            if (success_name == "release") {
                return failure_name == "relaxed";
            }
            if (success_name == "acq-rel") {
                return failure_name == "relaxed" || failure_name == "acquire";
            }
            if (success_name == "seq-cst") {
                return failure_name == "relaxed" || failure_name == "acquire" ||
                       failure_name == "seq-cst";
            }
            return false;
        };

        auto resolve_atomic_cmpxchg_failure_ordering =
            [&](const eshkol_ast_t& success_arg, const eshkol_ast_t& failure_arg,
                const char* builtin) -> bool {
            if (failure_arg.type != ESHKOL_VAR || !failure_arg.variable.id) {
                reportTypeIssue(std::string(builtin) +
                                    " expects a failure memory ordering designator",
                                &failure_arg);
                return false;
            }

            const std::string failure_name = failure_arg.variable.id;
            if (failure_name != "relaxed" && failure_name != "acquire" &&
                failure_name != "seq-cst") {
                reportTypeIssue(std::string(builtin) +
                                    " failure ordering must be relaxed, acquire, or seq-cst",
                                &failure_arg);
                return false;
            }

            if (success_arg.type != ESHKOL_VAR || !success_arg.variable.id) {
                return true;
            }

            const std::string success_name = success_arg.variable.id;
            if (success_name != "relaxed" && success_name != "acquire" &&
                success_name != "release" && success_name != "acq-rel" &&
                success_name != "seq-cst") {
                return true;
            }

            if (!atomic_cmpxchg_failure_ordering_allowed(success_name, failure_name)) {
                reportTypeIssue(std::string(builtin) +
                                    " failure ordering cannot be stronger than success ordering",
                                &failure_arg);
                return false;
            }

            return true;
        };

        if (func_name == "volatile-load") {
            auto load_type = call.num_vars >= 1
                                 ? resolve_low_level_type_designator(
                                       call.variables[0], "volatile-load", "machine", false)
                                 : std::nullopt;

            if (call.num_vars != 2) {
                reportTypeIssue("volatile-load expects exactly 2 arguments", expr);
            } else if (arg_types[1].success) {
                TypeId pointer_type = arg_types[1].inferred_type;
                if (pointer_type == BuiltinTypes::Value &&
                    call.variables[1].type == ESHKOL_VAR &&
                    call.variables[1].variable.id) {
                    ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                } else if (pointer_type != BuiltinTypes::Pointer) {
                    reportTypeIssue("volatile-load expects a Ptr address operand",
                                    &call.variables[1]);
                }
            }

            return TypeCheckResult::ok(load_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "volatile-store!") {
            auto store_type = call.num_vars >= 1
                                  ? resolve_low_level_type_designator(
                                        call.variables[0], "volatile-store!", "machine", false)
                                  : std::nullopt;

            if (call.num_vars != 3) {
                reportTypeIssue("volatile-store! expects exactly 3 arguments", expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("volatile-store! expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }

                if (store_type && arg_types[2].success) {
                    TypeId value_type = arg_types[2].inferred_type;
                    if (*store_type == BuiltinTypes::Pointer) {
                        if (value_type == BuiltinTypes::Value &&
                            call.variables[2].type == ESHKOL_VAR &&
                            call.variables[2].variable.id) {
                            ctx_.bind(call.variables[2].variable.id, BuiltinTypes::Pointer);
                        } else if (value_type != BuiltinTypes::Pointer) {
                            reportTypeIssue("volatile-store! expects a Ptr value for pointer stores",
                                            &call.variables[2]);
                        }
                    } else if (value_type == BuiltinTypes::Value &&
                               call.variables[2].type == ESHKOL_VAR &&
                               call.variables[2].variable.id) {
                        ctx_.bind(call.variables[2].variable.id, *store_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            "volatile-store! expects an integer value compatible with the machine type",
                            &call.variables[2]);
                    }
                }
            }

            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        if (func_name == "atomic-load") {
            auto load_type = call.num_vars >= 1
                                 ? resolve_low_level_type_designator(
                                       call.variables[0], "atomic-load", "machine", false)
                                 : std::nullopt;

            if (call.num_vars != 3) {
                reportTypeIssue("atomic-load expects exactly 3 arguments", expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("atomic-load expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }
                resolve_atomic_ordering(call.variables[2], "atomic-load", false);
            }

            return TypeCheckResult::ok(load_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "atomic-store!") {
            auto store_type = call.num_vars >= 1
                                  ? resolve_low_level_type_designator(
                                        call.variables[0], "atomic-store!", "machine", false)
                                  : std::nullopt;

            if (call.num_vars != 4) {
                reportTypeIssue("atomic-store! expects exactly 4 arguments", expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("atomic-store! expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }

                if (store_type && arg_types[2].success) {
                    TypeId value_type = arg_types[2].inferred_type;
                    if (*store_type == BuiltinTypes::Pointer) {
                        if (value_type == BuiltinTypes::Value &&
                            call.variables[2].type == ESHKOL_VAR &&
                            call.variables[2].variable.id) {
                            ctx_.bind(call.variables[2].variable.id, BuiltinTypes::Pointer);
                        } else if (value_type != BuiltinTypes::Pointer) {
                            reportTypeIssue("atomic-store! expects a Ptr value for pointer stores",
                                            &call.variables[2]);
                        }
                    } else if (value_type == BuiltinTypes::Value &&
                               call.variables[2].type == ESHKOL_VAR &&
                               call.variables[2].variable.id) {
                        ctx_.bind(call.variables[2].variable.id, *store_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            "atomic-store! expects an integer value compatible with the machine type",
                            &call.variables[2]);
                    }
                }

                resolve_atomic_ordering(call.variables[3], "atomic-store!", true);
            }

            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        if (func_name == "atomic-exchange!") {
            auto exchange_type = call.num_vars >= 1
                                     ? resolve_low_level_type_designator(
                                           call.variables[0], "atomic-exchange!", "machine", false)
                                     : std::nullopt;

            if (call.num_vars != 4) {
                reportTypeIssue("atomic-exchange! expects exactly 4 arguments", expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("atomic-exchange! expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }

                if (exchange_type && arg_types[2].success) {
                    TypeId value_type = arg_types[2].inferred_type;
                    if (*exchange_type == BuiltinTypes::Pointer) {
                        if (value_type == BuiltinTypes::Value &&
                            call.variables[2].type == ESHKOL_VAR &&
                            call.variables[2].variable.id) {
                            ctx_.bind(call.variables[2].variable.id, BuiltinTypes::Pointer);
                        } else if (value_type != BuiltinTypes::Pointer) {
                            reportTypeIssue("atomic-exchange! expects a Ptr value for pointer exchanges",
                                            &call.variables[2]);
                        }
                    } else if (value_type == BuiltinTypes::Value &&
                               call.variables[2].type == ESHKOL_VAR &&
                               call.variables[2].variable.id) {
                        ctx_.bind(call.variables[2].variable.id, *exchange_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            "atomic-exchange! expects an integer value compatible with the machine type",
                            &call.variables[2]);
                    }
                }

                resolve_atomic_rmw_ordering(call.variables[3], "atomic-exchange!");
            }

            return TypeCheckResult::ok(exchange_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "atomic-compare-exchange!") {
            auto cas_type = call.num_vars >= 1
                                ? resolve_low_level_type_designator(
                                      call.variables[0], "atomic-compare-exchange!",
                                      "machine", false)
                                : std::nullopt;

            if (call.num_vars != 6) {
                reportTypeIssue("atomic-compare-exchange! expects exactly 6 arguments",
                                expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("atomic-compare-exchange! expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }

                auto check_cas_value_operand = [&](size_t index, const char* role) {
                    if (!cas_type || !arg_types[index].success) {
                        return;
                    }

                    TypeId value_type = arg_types[index].inferred_type;
                    if (*cas_type == BuiltinTypes::Pointer) {
                        if (value_type == BuiltinTypes::Value &&
                            call.variables[index].type == ESHKOL_VAR &&
                            call.variables[index].variable.id) {
                            ctx_.bind(call.variables[index].variable.id,
                                      BuiltinTypes::Pointer);
                        } else if (value_type != BuiltinTypes::Pointer) {
                            reportTypeIssue(
                                std::string("atomic-compare-exchange! expects a Ptr ") +
                                    role + " value for pointer compare-exchanges",
                                &call.variables[index]);
                        }
                    } else if (value_type == BuiltinTypes::Value &&
                               call.variables[index].type == ESHKOL_VAR &&
                               call.variables[index].variable.id) {
                        ctx_.bind(call.variables[index].variable.id, *cas_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            std::string("atomic-compare-exchange! expects an integer ") +
                                role + " value compatible with the machine type",
                            &call.variables[index]);
                    }
                };

                check_cas_value_operand(2, "expected");
                check_cas_value_operand(3, "desired");

                resolve_atomic_rmw_ordering(call.variables[4],
                                            "atomic-compare-exchange!");
                resolve_atomic_cmpxchg_failure_ordering(
                    call.variables[4], call.variables[5], "atomic-compare-exchange!");
            }

            return TypeCheckResult::ok(cas_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "atomic-fetch-add!" || func_name == "atomic-fetch-sub!" ||
            func_name == "atomic-fetch-and!" || func_name == "atomic-fetch-or!" ||
            func_name == "atomic-fetch-xor!") {
            const char* builtin = func_name.c_str();
            auto rmw_type = call.num_vars >= 1
                                ? resolve_low_level_type_designator(
                                      call.variables[0], builtin, "machine", false)
                                : std::nullopt;

            if (rmw_type && *rmw_type == BuiltinTypes::Pointer) {
                reportTypeIssue(func_name + " requires an integer machine type",
                                &call.variables[0]);
            }

            if (call.num_vars != 4) {
                reportTypeIssue(func_name + " expects exactly 4 arguments", expr);
            } else {
                if (arg_types[1].success) {
                    TypeId pointer_type = arg_types[1].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue(func_name + " expects a Ptr address operand",
                                        &call.variables[1]);
                    }
                }

                if (rmw_type && *rmw_type != BuiltinTypes::Pointer &&
                    arg_types[2].success) {
                    TypeId value_type = arg_types[2].inferred_type;
                    if (value_type == BuiltinTypes::Value &&
                        call.variables[2].type == ESHKOL_VAR &&
                        call.variables[2].variable.id) {
                        ctx_.bind(call.variables[2].variable.id, *rmw_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            func_name + " expects an integer value compatible with the machine type",
                            &call.variables[2]);
                    }
                }

                resolve_atomic_rmw_ordering(call.variables[3], builtin);
            }

            return TypeCheckResult::ok(rmw_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "target-intrinsic") {
            auto return_type = call.num_vars >= 1
                                   ? resolve_low_level_type_designator(
                                         call.variables[0], "target-intrinsic", "return", true)
                                   : std::nullopt;

            if (call.num_vars < 2) {
                reportTypeIssue(
                    "target-intrinsic expects a return type, intrinsic name, and zero or more typed arguments",
                    expr);
            } else {
                if (call.variables[1].type != ESHKOL_STRING || !call.variables[1].str_val.ptr) {
                    reportTypeIssue(
                        "target-intrinsic expects an LLVM intrinsic name string as its second argument",
                        &call.variables[1]);
                }

                if ((call.num_vars % 2) != 0) {
                    reportTypeIssue(
                        "target-intrinsic expects argument type/value pairs after the intrinsic name",
                        expr);
                }

                for (size_t i = 2; i + 1 < call.num_vars; i += 2) {
                    auto arg_type = resolve_low_level_type_designator(
                        call.variables[i], "target-intrinsic", "argument", false);
                    if (!arg_type || !arg_types[i + 1].success) {
                        continue;
                    }

                    TypeId value_type = arg_types[i + 1].inferred_type;
                    if (*arg_type == BuiltinTypes::Pointer) {
                        if (value_type == BuiltinTypes::Value &&
                            call.variables[i + 1].type == ESHKOL_VAR &&
                            call.variables[i + 1].variable.id) {
                            ctx_.bind(call.variables[i + 1].variable.id, BuiltinTypes::Pointer);
                        } else if (value_type != BuiltinTypes::Pointer) {
                            reportTypeIssue(
                                "target-intrinsic expects a Ptr value for pointer-typed arguments",
                                &call.variables[i + 1]);
                        }
                    } else if (value_type == BuiltinTypes::Value &&
                               call.variables[i + 1].type == ESHKOL_VAR &&
                               call.variables[i + 1].variable.id) {
                        ctx_.bind(call.variables[i + 1].variable.id, *arg_type);
                    } else if (!env_.isSubtype(value_type, BuiltinTypes::Integer)) {
                        reportTypeIssue(
                            "target-intrinsic expects integer-compatible values for machine integer arguments",
                            &call.variables[i + 1]);
                    }
                }
            }

            return TypeCheckResult::ok(return_type.value_or(BuiltinTypes::Value));
        }

        if (func_name == "compiler-fence") {
            if (call.num_vars != 1) {
                reportTypeIssue("compiler-fence expects exactly 1 ordering argument", expr);
            } else {
                resolve_fence_ordering(call.variables[0], "compiler-fence");
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        if (func_name == "memory-fence") {
            if (call.num_vars != 1) {
                reportTypeIssue("memory-fence expects exactly 1 ordering argument", expr);
            } else {
                resolve_fence_ordering(call.variables[0], "memory-fence");
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        if (func_name == "addr-of") {
            if (call.num_vars != 1) {
                reportTypeIssue("addr-of expects exactly 1 argument", expr);
            } else if (call.variables[0].type != ESHKOL_VAR ||
                       !call.variables[0].variable.id) {
                reportTypeIssue("addr-of requires a variable reference", &call.variables[0]);
            }
            return TypeCheckResult::ok(BuiltinTypes::Pointer);
        }
        if (func_name == "null-ptr") {
            return TypeCheckResult::ok(BuiltinTypes::Pointer);
        }
        if (func_name == "ptr->usize") {
            if (call.num_vars >= 1 && arg_types[0].success) {
                TypeId arg_type = arg_types[0].inferred_type;
                if (arg_type == BuiltinTypes::Value &&
                    call.variables[0].type == ESHKOL_VAR &&
                    call.variables[0].variable.id) {
                    ctx_.bind(call.variables[0].variable.id, BuiltinTypes::Pointer);
                } else if (arg_type != BuiltinTypes::Pointer) {
                    reportTypeIssue("ptr->usize expects a Ptr argument", &call.variables[0]);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::USize);
        }
        if (func_name == "usize->ptr") {
            if (call.num_vars >= 1 && arg_types[0].success) {
                TypeId arg_type = arg_types[0].inferred_type;
                if (arg_type == BuiltinTypes::Value &&
                    call.variables[0].type == ESHKOL_VAR &&
                    call.variables[0].variable.id) {
                    ctx_.bind(call.variables[0].variable.id, BuiltinTypes::USize);
                } else if (!env_.isSubtype(arg_type, BuiltinTypes::Integer)) {
                    reportTypeIssue("usize->ptr expects an integer address value", &call.variables[0]);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Pointer);
        }
        if (func_name == "ptr-add") {
            if (call.num_vars != 2) {
                reportTypeIssue("ptr-add expects exactly 2 arguments", expr);
            } else {
                if (arg_types[0].success) {
                    TypeId pointer_type = arg_types[0].inferred_type;
                    if (pointer_type == BuiltinTypes::Value &&
                        call.variables[0].type == ESHKOL_VAR &&
                        call.variables[0].variable.id) {
                        ctx_.bind(call.variables[0].variable.id, BuiltinTypes::Pointer);
                    } else if (pointer_type != BuiltinTypes::Pointer) {
                        reportTypeIssue("ptr-add expects a Ptr base operand",
                                        &call.variables[0]);
                    }
                }

                if (arg_types[1].success) {
                    TypeId offset_type = arg_types[1].inferred_type;
                    if (offset_type == BuiltinTypes::Value &&
                        call.variables[1].type == ESHKOL_VAR &&
                        call.variables[1].variable.id) {
                        ctx_.bind(call.variables[1].variable.id, BuiltinTypes::ISize);
                    } else if (!env_.isSubtype(offset_type, BuiltinTypes::Integer)) {
                        reportTypeIssue("ptr-add expects an integer byte offset",
                                        &call.variables[1]);
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Pointer);
        }

        // Vector operations
        if (func_name == "vector" || func_name == "make-vector" || func_name == "vector-copy") {
            return TypeCheckResult::ok(BuiltinTypes::Vector);
        }
        if (func_name == "vector-ref") {
            // Dimension checking: if index is a literal, validate non-negative
            if (call.num_vars >= 2) {
                const eshkol_ast_t& idx_arg = call.variables[1];
                if (idx_arg.type == ESHKOL_INT64 && idx_arg.int64_val < 0) {
                    reportTypeIssue("vector-ref: negative index " +
                        std::to_string(idx_arg.int64_val), &idx_arg);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "vector-length") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "vector-set!") {
            // Dimension checking: validate index
            if (call.num_vars >= 2) {
                const eshkol_ast_t& idx_arg = call.variables[1];
                if (idx_arg.type == ESHKOL_INT64 && idx_arg.int64_val < 0) {
                    reportTypeIssue("vector-set!: negative index " +
                        std::to_string(idx_arg.int64_val), &idx_arg);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }
        if (func_name == "vector-copy!" || func_name == "vector-fill!") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }
        if (func_name == "vector-append" || func_name == "list->vector") {
            return TypeCheckResult::ok(BuiltinTypes::Vector);
        }
        if (func_name == "vector->list") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }

        // Port operations
        if (func_name == "current-input-port" || func_name == "current-output-port" ||
            func_name == "current-error-port") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "input-port?" || func_name == "output-port?" ||
            func_name == "port?" || func_name == "char-ready?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        if (func_name == "read-char" || func_name == "peek-char") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "make-promise" || func_name == "%make-lazy-promise" ||
            func_name == "%make-lazy-promise-force" || func_name == "force") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "promise?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        // Bytevector operations
        if (func_name == "make-bytevector" || func_name == "bytevector" ||
            func_name == "bytevector-copy" || func_name == "bytevector-append" ||
            func_name == "string->utf8") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "bytevector-length" || func_name == "bytevector-u8-ref") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "bytevector-u8-set!" || func_name == "bytevector-copy!") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }
        if (func_name == "bytevector?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        if (func_name == "utf8->string") {
            return TypeCheckResult::ok(BuiltinTypes::String);
        }
        if (func_name == "tensor-save" || func_name == "model-save") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        if (func_name == "tensor-load") {
            return TypeCheckResult::ok(BuiltinTypes::Tensor);
        }
        if (func_name == "model-load") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }

        // Rational operations
        if (func_name == "rational?" || func_name == "exact-rational?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }
        if (func_name == "numerator" || func_name == "denominator") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "make-rational" || func_name == "rationalize") {
            return TypeCheckResult::ok(BuiltinTypes::Number);
        }
        if (func_name == "make-complex" || func_name == "make-rectangular" || func_name == "make-polar") {
            return TypeCheckResult::ok(BuiltinTypes::Complex);
        }
        if (func_name == "real-part" || func_name == "imag-part" ||
            func_name == "magnitude" || func_name == "angle") {
            return TypeCheckResult::ok(BuiltinTypes::Float64);
        }

        // Signal Processing Filters (stdlib: signal.filters)
        if (func_name == "fft" || func_name == "ifft" ||
            func_name == "hamming-window" || func_name == "hann-window" ||
            func_name == "blackman-window" || func_name == "kaiser-window" ||
            func_name == "apply-window" || func_name == "convolve" ||
            func_name == "fast-convolve" || func_name == "fir-filter" ||
            func_name == "iir-filter") {
            return TypeCheckResult::ok(BuiltinTypes::Vector);
        }
        if (func_name == "butterworth-lowpass" || func_name == "butterworth-highpass" ||
            func_name == "butterworth-bandpass" || func_name == "frequency-response") {
            return TypeCheckResult::ok(BuiltinTypes::Pair);
        }

        // Optimization Algorithms (stdlib: ml.optimization)
        if (func_name == "gradient-descent" || func_name == "adam" ||
            func_name == "l-bfgs" || func_name == "conjugate-gradient") {
            return TypeCheckResult::ok(BuiltinTypes::Vector);
        }
        if (func_name == "line-search" || func_name == "tensor-dot" ||
            func_name == "tensor-norm") {
            return TypeCheckResult::ok(BuiltinTypes::Float64);
        }
        if (func_name == "tensor-svd") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }

        // I/O operations
        if (func_name == "display" || func_name == "newline" || func_name == "write") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        // Mutation operations — borrow check the target
        if (func_name == "set!") {
            if (call.num_vars >= 1 && call.variables[0].type == ESHKOL_VAR &&
                call.variables[0].variable.id) {
                std::string target = call.variables[0].variable.id;
                // Check borrow rules: can't mutate a shared-borrowed value
                if (!borrow_.canBorrowMut(target)) {
                    auto state = borrow_.getState(target);
                    if (state == BorrowState::BorrowedShared) {
                        reportTypeIssue("cannot set! '" + target +
                            "': value is borrowed immutably", &call.variables[0]);
                    } else if (state == BorrowState::Moved) {
                        reportTypeIssue("cannot set! '" + target +
                            "': value has been moved", &call.variables[0]);
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        // Begin expression - returns type of last expression (already pre-synthesized)
        if (func_name == "begin") {
            if (call.num_vars == 0) {
                return TypeCheckResult::ok(BuiltinTypes::Null);
            }
            return arg_types[call.num_vars - 1];
        }
        if (func_name == "read" || func_name == "read-line" || func_name == "read-string") {
            return TypeCheckResult::ok(BuiltinTypes::String);
        }

        // Binary I/O
        if (func_name == "open-binary-input-file" || func_name == "open-binary-output-file") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "read-u8" || func_name == "read-bytevector" || func_name == "read-bytevector!") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "write-u8" || func_name == "write-bytevector") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }
        if (func_name == "u8-ready?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // Boolean operations
        if (func_name == "not" || func_name == "and" || func_name == "or") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // If expression - use pre-synthesized branch types and compute LCS
        if (func_name == "if") {
            // if has: condition (0), then-branch (1), else-branch (2)
            if (call.num_vars >= 2) {
                if (!arg_types[1].success) return arg_types[1];

                if (call.num_vars >= 3) {
                    if (!arg_types[2].success) return arg_types[2];

                    TypeId then_t = arg_types[1].inferred_type;
                    TypeId else_t = arg_types[2].inferred_type;

                    // If one branch is Value (unknown/top), prefer the other branch's type
                    // This handles loops/recursion where one branch is a recursive call
                    if (then_t == BuiltinTypes::Value && else_t != BuiltinTypes::Value) {
                        return TypeCheckResult::ok(else_t);
                    }
                    if (else_t == BuiltinTypes::Value && then_t != BuiltinTypes::Value) {
                        return TypeCheckResult::ok(then_t);
                    }

                    // Compute LCS of branches
                    auto lcs = env_.leastCommonSupertype(then_t, else_t);
                    if (lcs) {
                        return TypeCheckResult::ok(*lcs);
                    }
                }
                return arg_types[1];
            }
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
    }

    // Try to determine the function type from context
    TypeId func_type = BuiltinTypes::Function;

    if (func_expr->type == ESHKOL_VAR && func_expr->variable.id) {
        // Look up the function in context
        auto lookup = ctx_.lookup(func_expr->variable.id);
        if (lookup) {
            func_type = *lookup;
        }
    } else if (func_expr->type == ESHKOL_OP && func_expr->operation.op == ESHKOL_LAMBDA_OP) {
        // Inline lambda - synthesize its type
        auto lambda_result = synthesize(const_cast<eshkol_ast_t*>(func_expr));
        if (lambda_result.success) {
            func_type = lambda_result.inferred_type;
        }
    }

    // If we have a proper function type, return its return type
    if (env_.isFunctionType(func_type)) {
        TypeId return_type = env_.getFunctionReturnType(func_type);

        // Check argument count and types if we have function type info
        const PiType* pi = env_.getFunctionType(func_type);
        if (pi) {
            // Arity check — skip entirely when the callee is variadic: a
            // rest-arg accepts any number of trailing positional args, so
            // `(f a b)` on `(define (f x . rest) …)` is well-formed.
            if (!pi->is_variadic &&
                pi->params.size() > 0 && call.num_vars > pi->params.size()) {
                std::string msg = "function '" +
                    std::string(func_expr->type == ESHKOL_VAR ? func_expr->variable.id : "<lambda>") +
                    "' expects " + std::to_string(pi->params.size()) +
                    " arguments, got " + std::to_string(call.num_vars);
                reportTypeIssue(msg, expr);
            }

            // Argument type checking (use pre-synthesized arg_types)
            size_t check_count = std::min(pi->params.size(), static_cast<size_t>(call.num_vars));
            for (size_t i = 0; i < check_count; ++i) {
                if (arg_types[i].success && pi->params[i].type.id != 0) {
                    TypeId expected = pi->params[i].type;
                    TypeId actual = arg_types[i].inferred_type;
                    // Skip check if either is Value (unknown/top type) or same type
                    if (expected != BuiltinTypes::Value && actual != BuiltinTypes::Value &&
                        expected != actual) {
                        // Check if types are compatible via LCS
                        auto lcs = env_.leastCommonSupertype(expected, actual);
                        if (!lcs || *lcs != expected) {
                            std::string msg = "argument " + std::to_string(i + 1) + " of '" +
                                std::string(func_expr->type == ESHKOL_VAR ? func_expr->variable.id : "<lambda>") +
                                "': expected " + env_.getTypeName(expected) +
                                ", got " + env_.getTypeName(actual);
                            reportTypeIssue(msg, &call.variables[i]);
                        }
                    }
                }
            }
        }

        return TypeCheckResult::ok(return_type);
    }

    // Fallback: unknown function type
    return TypeCheckResult::ok(BuiltinTypes::Value);
}

/**
 * @brief Synthesis rule for `define` forms, both variable and function shapes.
 *
 * External declarations (`def.is_external`, e.g. `extern` FFI bindings) are
 * bound directly from their (optional) parameter/return type annotations
 * without checking a body, since none exists.
 *
 * For a genuine function define `(define (name params...) body)`: computes
 * parameter types from annotations (defaulting to BuiltinTypes::Value), then
 * pre-binds `name` in the context to a function type built from those
 * parameter types and the declared (or Value) return type *before*
 * synthesizing the body — this makes recursive calls to `name` resolve
 * during body checking. The body is then synthesized in a fresh scope with
 * parameters bound. Because arithmetic/comparison rules in
 * synthesizeApplication() can narrow a Value-typed parameter to a more
 * precise type (e.g. Number) as a side effect of checking the body, the
 * parameter types are re-read from the context immediately after body
 * synthesis but before the scope is popped, and this narrowed parameter
 * list is what the function's final type is built from. If a return type
 * annotation exists, the body's inferred type must be a subtype of it
 * (error otherwise); the final inferred function type (narrowed params +
 * resolved return) replaces the pre-binding.
 *
 * For a variable define `(define name value)`: simply synthesizes @p value's
 * type and binds `name` to it.
 * @return TypeCheckResult::ok(BuiltinTypes::Null) on success (define itself
 * has no value in this type system); an error if unnamed, if body synthesis
 * fails for a variable define, or if a function's body type violates its
 * return annotation.
 */
TypeCheckResult TypeChecker::synthesizeDefine(eshkol_ast_t* expr) {
    const auto& def = expr->operation.define_op;

    if (!def.name) {
        return errorAt(expr, "Define without name");
    }

    if (def.is_external) {
        if (def.is_function) {
            std::vector<TypeId> param_types;
            for (size_t i = 0; i < def.num_params; i++) {
                if (def.parameters && def.parameters[i].type == ESHKOL_VAR &&
                    def.parameters[i].variable.id) {
                    TypeId param_type = BuiltinTypes::Value;
                    if (def.param_types && def.param_types[i]) {
                        param_type = resolveType(def.param_types[i]);
                    }
                    param_types.push_back(param_type);
                }
            }

            TypeId return_type = def.return_type
                ? resolveType(def.return_type)
                : BuiltinTypes::Value;
            TypeId func_type = env_.makeFunctionType(param_types, return_type,
                                                      def.is_variadic);
            ctx_.bind(def.name, func_type);
        } else {
            ctx_.bind(def.name, BuiltinTypes::Value);
        }
        return TypeCheckResult::ok(BuiltinTypes::Null);
    }

    TypeCheckResult value_type = TypeCheckResult::ok(BuiltinTypes::Value);

    if (def.is_function) {
        // Function define: (define (name params...) body)
        // Handle like a lambda - bind parameters and check body

        // First, collect parameter types (needed for recursive binding)
        std::vector<TypeId> param_types;
        for (size_t i = 0; i < def.num_params; i++) {
            if (def.parameters && def.parameters[i].type == ESHKOL_VAR &&
                def.parameters[i].variable.id) {
                TypeId param_type = BuiltinTypes::Value;
                if (def.param_types && def.param_types[i]) {
                    param_type = resolveType(def.param_types[i]);
                }
                param_types.push_back(param_type);
            }
        }

        // For recursive functions: if we have a return type annotation,
        // pre-bind the function name with its declared type so recursive calls resolve
        TypeId declared_return = BuiltinTypes::Value;
        if (def.return_type) {
            declared_return = resolveType(def.return_type);
        }
        TypeId func_type = env_.makeFunctionType(param_types, declared_return,
                                                  def.is_variadic);
        ctx_.bind(def.name, func_type);

        // Now push scope and bind parameters
        ctx_.pushScope();
        for (size_t i = 0; i < def.num_params; i++) {
            if (def.parameters && def.parameters[i].type == ESHKOL_VAR &&
                def.parameters[i].variable.id) {
                ctx_.bind(def.parameters[i].variable.id, param_types[i]);
            }
        }

        // Type check the body if present
        if (def.value) {
            value_type = synthesize(def.value);
        }

        // Re-read parameter types BEFORE popping scope — body synthesis may have
        // narrowed them via backward type inference (e.g., used in arithmetic → Number)
        std::vector<TypeId> narrowed_param_types;
        for (size_t i = 0; i < def.num_params; i++) {
            if (def.parameters && def.parameters[i].type == ESHKOL_VAR &&
                def.parameters[i].variable.id) {
                auto narrowed = ctx_.lookup(def.parameters[i].variable.id);
                narrowed_param_types.push_back(narrowed ? *narrowed : param_types[i]);
            }
        }

        ctx_.popScope();

        // Determine return type
        TypeId return_type = value_type.success ? value_type.inferred_type : BuiltinTypes::Value;
        if (def.return_type) {
            TypeId annotated_return = resolveType(def.return_type);
            // Check that body type is compatible with annotation
            if (value_type.success && !env_.isSubtype(value_type.inferred_type, annotated_return)) {
                return errorAt(expr,
                    "Function '" + std::string(def.name) + "' body type " +
                    env_.getTypeName(value_type.inferred_type) +
                    " doesn't match return annotation " + env_.getTypeName(annotated_return));
            }
            return_type = annotated_return;
        }

        // Create function type with narrowed param types and inferred return type
        TypeId final_func_type = env_.makeFunctionType(narrowed_param_types, return_type,
                                                        def.is_variadic);
        value_type = TypeCheckResult::ok(final_func_type);

        // Re-bind function name with the full inferred type (overrides pre-binding)
        ctx_.bind(def.name, final_func_type);
    } else {
        // Variable define: (define name value)
        if (def.value) {
            value_type = synthesize(def.value);
            if (!value_type.success) {
                return value_type;
            }
        }
    }

    // Bind in context (only for non-functions - functions were pre-bound for recursion)
    if (!def.is_function) {
        ctx_.bind(def.name, value_type.inferred_type);
    }

    return TypeCheckResult::ok(BuiltinTypes::Null);
}

/**
 * @brief Synthesis rule for `let`/`let*`/`letrec`/`letrec*` forms (all share this rule).
 *
 * Pushes a fresh scope, then for each binding determines its type from an
 * explicit annotation (resolveType()) or, failing that, by synthesizing the
 * bound expression, and binds the name in the (already-pushed) scope — so
 * later bindings and the body can see earlier ones, matching letrec-style
 * visibility for all four let variants. For a named let (`(let loop
 * ((i 0)) body)`), pre-binds `loop` as a recursive function from the
 * binding types to BuiltinTypes::Value so recursive calls resolve during
 * body synthesis, then re-binds it with the body's actual inferred return
 * type afterward (best-effort, since this happens after the body was
 * already checked against the placeholder type). The body is synthesized
 * and its result becomes the let's type; the scope is popped before
 * returning.
 * @return The body's TypeCheckResult (success or failure).
 */
TypeCheckResult TypeChecker::synthesizeLet(eshkol_ast_t* expr) {
    const auto& let = expr->operation.let_op;

    ctx_.pushScope();

    // Collect binding types
    std::vector<TypeId> binding_types;
    std::vector<std::string> binding_names;

    // Process each binding
    for (size_t i = 0; i < let.num_bindings; i++) {
        const auto& binding = let.bindings[i];
        if (binding.type != ESHKOL_CONS) continue;

        // Get variable name
        if (!binding.cons_cell.car) continue;
        if (binding.cons_cell.car->type != ESHKOL_VAR) continue;
        char* var_id = binding.cons_cell.car->variable.id;
        if (!var_id) continue;
        std::string name = var_id;

        // Get binding type from annotation or infer from value
        TypeId binding_type = BuiltinTypes::Value;
        if (let.binding_types && let.binding_types[i]) {
            binding_type = resolveType(let.binding_types[i]);
        } else if (binding.cons_cell.cdr) {
            auto inferred = synthesize(binding.cons_cell.cdr);
            if (inferred.success) {
                binding_type = inferred.inferred_type;
            }
        }

        binding_names.push_back(name);
        binding_types.push_back(binding_type);
        ctx_.bind(name, binding_type);
    }

    // Handle named let (loop): pre-bind the loop name as a recursive function
    // Named let: (let loop ((i 0)) body) - loop is a function that takes i and returns body type
    if (let.name) {
        // For named let, we need to figure out the return type
        // Pre-bind with Value return type, then update after body synthesis
        TypeId loop_type = env_.makeFunctionType(binding_types, BuiltinTypes::Value);
        ctx_.bind(let.name, loop_type);
    }

    // Synthesize body type
    auto body_result = synthesize(let.body);

    // For named let, update the loop function's return type based on body
    if (let.name && body_result.success) {
        // Re-bind with the actual return type (though this is after the fact)
        TypeId loop_type = env_.makeFunctionType(binding_types, body_result.inferred_type);
        ctx_.bind(let.name, loop_type);
    }

    ctx_.popScope();

    return body_result;
}

/**
 * @brief Synthesis rule for `if` expressions: type is the least common
 * supertype (LCS) of the branch types.
 *
 * Requires at least a condition and a then-branch (error otherwise); the
 * condition itself is not type-checked here. Synthesizes the then-branch
 * type, and if an else-branch is present, synthesizes it too. As a
 * special case for named-let recursive loops (where a recursive call
 * branch often still carries the placeholder BuiltinTypes::Value while the
 * base-case branch has a concrete type), if exactly one branch is Value,
 * the other (concrete) branch's type is preferred outright rather than
 * computing an LCS. Otherwise the result is
 * TypeEnvironment::leastCommonSupertype() of the two branch types. With no
 * else-branch, the then-branch's type is returned directly.
 * @return The then-branch's TypeCheckResult if there is no else-branch or
 * synthesis of a branch fails; otherwise the LCS (or preferred concrete
 * branch type) of both branches.
 */
TypeCheckResult TypeChecker::synthesizeIf(eshkol_ast_t* expr) {
    // if has: condition, then-branch, else-branch
    // For now, just synthesize branches and take LCS
    if (expr->operation.call_op.num_vars < 2) {
        return errorAt(expr, "if requires condition and then-branch");
    }

    auto then_type = synthesize(&expr->operation.call_op.variables[1]);
    if (!then_type.success) return then_type;

    if (expr->operation.call_op.num_vars >= 3) {
        auto else_type = synthesize(&expr->operation.call_op.variables[2]);
        if (!else_type.success) return else_type;

        TypeId then_t = then_type.inferred_type;
        TypeId else_t = else_type.inferred_type;

        // For named let loops: if one branch is Value (recursive call),
        // prefer the concrete type from the base case branch
        if (then_t == BuiltinTypes::Value && else_t != BuiltinTypes::Value) {
            return TypeCheckResult::ok(else_t);
        }
        if (else_t == BuiltinTypes::Value && then_t != BuiltinTypes::Value) {
            return TypeCheckResult::ok(then_t);
        }

        // Compute LCS of branches
        auto lcs = env_.leastCommonSupertype(then_t, else_t);
        if (lcs) {
            return TypeCheckResult::ok(*lcs);
        }
    }

    return then_type;
}

/**
 * @brief Checking rule for lambda expressions (Γ ⊢ (lambda ...) ⇐ τ).
 *
 * Currently a placeholder: rather than propagating @p expected's domain
 * types down into unannotated parameters, it simply defers entirely to the
 * synthesis rule synthesizeLambda(). The @p expected parameter is unused.
 * @return Whatever synthesizeLambda() returns for @p expr.
 */
TypeCheckResult TypeChecker::checkLambda(eshkol_ast_t* expr, TypeId expected) {
    // If expected is a function type, use its domain for param types
    // For now, just synthesize
    return synthesizeLambda(expr);
}

/**
 * @brief Check whether @p type is a function type, and if so extract its domain/codomain.
 *
 * Currently simplified: only recognizes the single builtin
 * BuiltinTypes::Function marker and never populates @p domain/@p codomain
 * (both output parameters are unused placeholders for a future
 * parameterized function-type representation).
 * @return True if @p type.id equals BuiltinTypes::Function.id.
 */
bool TypeChecker::isFunctionType(TypeId type, TypeId& domain, TypeId& codomain) const {
    // Check if type is Function or a parameterized function type
    // Simplified for now
    return type.id == BuiltinTypes::Function.id;
}

/**
 * @brief Simple type unification: true if @p a and @p b are equal or either
 * is a subtype of the other.
 *
 * This is not full bidirectional unification with substitution/metavariable
 * solving — it is a compatibility check used where the checker needs a
 * best-effort answer to "can these two types be reconciled."
 */
bool TypeChecker::unify(TypeId a, TypeId b) {
    // Simple unification - just check equality or subtyping
    return a == b || env_.isSubtype(a, b) || env_.isSubtype(b, a);
}

/**
 * @brief Resolve a parsed HoTT type expression (as written in source, e.g.
 * a type annotation) to a concrete TypeId in the type environment.
 */
/**
 * @brief Resolve a parsed hott_type_expr_t (a type annotation as written in
 * source) to a concrete runtime TypeId.
 *
 * Maps each HOTT_TYPE_* primitive kind to its corresponding BuiltinTypes
 * entry (Integer->Int64, Real->Float64, String, Boolean, Char, Null, Symbol,
 * Any->Value). HOTT_TYPE_VAR is resolved by first checking whether its name
 * is a registered type alias (Context::lookupTypeAlias(), recursing into the
 * alias's expansion) and otherwise a builtin type name
 * (TypeEnvironment::lookupType()), falling back to Value if neither matches.
 * HOTT_TYPE_ARROW recursively resolves its parameter and return types and
 * builds a proper function type via TypeEnvironment::makeFunctionType().
 * List/Vector/Tensor/Pointer/Pair map to their respective builtin container
 * types; Product and Sum types are both represented at runtime as Pair
 * (Sum specifically as a tagged `(tag . value)` cons cell, since HoTT
 * polymorphism is erased and tagged values carry their own type tag).
 * HOTT_TYPE_FORALL resolves to its body's type (the quantifier itself has no
 * runtime representation — instantiation/substitution happens at compile
 * time via substituteTypeVars()). Any other/unrecognized kind, or a null @p
 * type_expr, resolves to BuiltinTypes::Value.
 */
TypeId TypeChecker::resolveType(const hott_type_expr_t* type_expr) {
    if (!type_expr) {
        return BuiltinTypes::Value;
    }

    switch (type_expr->kind) {
        // Primitive types - each has its own kind
        case HOTT_TYPE_INTEGER:
            return BuiltinTypes::Int64;

        case HOTT_TYPE_REAL:
            return BuiltinTypes::Float64;

        case HOTT_TYPE_STRING:
            return BuiltinTypes::String;

        case HOTT_TYPE_BOOLEAN:
            return BuiltinTypes::Boolean;

        case HOTT_TYPE_CHAR:
            return BuiltinTypes::Char;

        case HOTT_TYPE_NULL:
            return BuiltinTypes::Null;

        case HOTT_TYPE_SYMBOL:
            return BuiltinTypes::Symbol;

        case HOTT_TYPE_ANY:
            return BuiltinTypes::Value;

        // Type variable - check for alias
        case HOTT_TYPE_VAR: {
            if (type_expr->var_name) {
                auto alias = ctx_.lookupTypeAlias(type_expr->var_name);
                if (alias) {
                    return resolveType(*alias);
                }
                auto builtin = env_.lookupType(type_expr->var_name);
                if (builtin) {
                    return *builtin;
                }
            }
            return BuiltinTypes::Value;
        }

        case HOTT_TYPE_ARROW: {
            // Resolve arrow type to a proper function type with param/return types
            std::vector<TypeId> param_types;
            for (uint64_t i = 0; i < type_expr->arrow.num_params; i++) {
                if (type_expr->arrow.param_types && type_expr->arrow.param_types[i]) {
                    param_types.push_back(resolveType(type_expr->arrow.param_types[i]));
                } else {
                    param_types.push_back(BuiltinTypes::Value);
                }
            }
            TypeId return_type = BuiltinTypes::Value;
            if (type_expr->arrow.return_type) {
                return_type = resolveType(type_expr->arrow.return_type);
            }
            return env_.makeFunctionType(param_types, return_type);
        }

        case HOTT_TYPE_LIST:
            return BuiltinTypes::List;

        case HOTT_TYPE_VECTOR:
            return BuiltinTypes::Vector;

        case HOTT_TYPE_TENSOR:
            return BuiltinTypes::Tensor;

        case HOTT_TYPE_POINTER:
            return BuiltinTypes::Pointer;

        case HOTT_TYPE_PAIR:
            return BuiltinTypes::Pair;

        case HOTT_TYPE_PRODUCT:
            return BuiltinTypes::Pair;  // Product types represented as pairs at runtime

        case HOTT_TYPE_SUM:
            return BuiltinTypes::Pair;  // Sum types represented as tagged cons pairs: (tag . value)

        case HOTT_TYPE_FORALL:
            // Polymorphic types are erased at runtime — tagged values handle polymorphism
            // naturally since every value carries its own type tag. The type checker
            // performs substitution/instantiation at compile time (see substituteTypeVars).
            if (type_expr->forall.body) {
                return resolveType(type_expr->forall.body);
            }
            return BuiltinTypes::Value;

        default:
            return BuiltinTypes::Value;
    }
}

/**
 * @brief Append a plain error TypeCheckResult (@p msg at @p line:@p col) to the recorded error list.
 */
void TypeChecker::addError(const std::string& msg, int line, int col) {
    errors_.push_back(TypeCheckResult::error(msg, line, col));
}

/**
 * @brief Record a formatted "type mismatch: expected X, got Y" error at @p line:@p col.
 */
void TypeChecker::addTypeMismatch(TypeId expected, TypeId actual, int line, int col) {
    std::ostringstream ss;
    ss << "Type mismatch: expected " << env_.getTypeName(expected)
       << ", got " << env_.getTypeName(actual);
    addError(ss.str(), line, col);
}

/**
 * @brief Unified type-issue reporting entry point, respecting unsafe/strict mode.
 *
 * In unsafe mode, does nothing (issues are silently ignored). Otherwise
 * appends @p node's line/column (when available) to @p msg and prints it to
 * stderr — as `[ERROR] Type error: ...` if strict_types_ is set, or
 * `[WARN] Type warning: ...` otherwise (gradual typing: warn and continue) —
 * and always records the unformatted @p msg via addError() so it also
 * appears in errors().
 */
void TypeChecker::reportTypeIssue(const std::string& msg, const eshkol_ast_t* node) {
    if (unsafe_mode_) return;

    std::string loc_msg = msg;
    if (node && node->line > 0) {
        loc_msg += " (line " + std::to_string(node->line);
        if (node->column > 0) {
            loc_msg += ":" + std::to_string(node->column);
        }
        loc_msg += ")";
    }

    if (strict_types_) {
        fprintf(stderr, "[ERROR] Type error: %s\n", loc_msg.c_str());
    } else {
        fprintf(stderr, "[WARN] Type warning: %s\n", loc_msg.c_str());
    }
    addError(msg, node ? node->line : 0, node ? node->column : 0);
}

// ============================================================================
// Dimension Checking (Phase 5.3)
// ============================================================================

/**
 * @brief Check that a vector/tensor access @p index is within the compile-time bound @p bound.
 *
 * Delegates to DimensionChecker::checkBounds(); on failure, both records the
 * error via addError() and returns it.
 * @param context Description used in the diagnostic message on failure.
 * @return TypeCheckResult::ok(BuiltinTypes::Boolean) if in bounds, otherwise
 * a failing result carrying DimensionChecker's error message.
 */
TypeCheckResult TypeChecker::checkVectorBounds(const CTValue& index, const CTValue& bound,
                                               const std::string& context) {
    auto result = DimensionChecker::checkBounds(index, bound, context);
    if (result.valid) {
        return TypeCheckResult::ok(BuiltinTypes::Boolean);
    }
    addError(result.error_message);
    return TypeCheckResult::error(result.error_message);
}

/**
 * @brief Check that @p left and @p right have compatible dimensions for a dot product.
 *
 * Delegates to DimensionChecker::checkDotProductDimensions(); on failure,
 * both records the error via addError() and returns it.
 * @return TypeCheckResult::ok(BuiltinTypes::Float64) (a dot product is a
 * scalar) if the dimensions are compatible, otherwise a failing result
 * carrying DimensionChecker's error message.
 */
TypeCheckResult TypeChecker::checkDotProductDimensions(const DependentType& left,
                                                       const DependentType& right) {
    auto result = DimensionChecker::checkDotProductDimensions(left, right, "dot product");
    if (result.valid) {
        // Dot product returns scalar
        return TypeCheckResult::ok(BuiltinTypes::Float64);
    }
    addError(result.error_message);
    return TypeCheckResult::error(result.error_message);
}

/**
 * @brief Check that @p left and @p right have compatible dimensions for matrix multiplication.
 *
 * Delegates to DimensionChecker::checkMatMulDimensions(); on failure, both
 * records the error via addError() and returns it. The result dimensions
 * for a valid (m x n) * (n x p) -> (m x p) multiply are not yet tracked
 * precisely — only the generic Tensor type is returned.
 * @return TypeCheckResult::ok(BuiltinTypes::Tensor) if the dimensions are
 * compatible, otherwise a failing result carrying DimensionChecker's error
 * message.
 */
TypeCheckResult TypeChecker::checkMatrixMultiplyDimensions(const DependentType& left,
                                                           const DependentType& right) {
    auto result = DimensionChecker::checkMatMulDimensions(left, right, "matrix multiply");
    if (result.valid) {
        // Matrix multiply result dimensions: (m x n) * (n x p) -> (m x p)
        // For now, just return Tensor as the result type
        return TypeCheckResult::ok(BuiltinTypes::Tensor);
    }
    addError(result.error_message);
    return TypeCheckResult::error(result.error_message);
}

/**
 * @brief Extract the compile-time size of dimension @p dim_index of a vector/tensor @p type.
 *
 * First checks the type environment's per-@p type dimension cache
 * (TypeEnvironment::getDimensionInfo()); if that misses and @p type is
 * exactly BuiltinTypes::Tensor or BuiltinTypes::Vector, falls back to
 * checking the cache keyed by that base builtin type (dimensions recorded
 * generically rather than per concrete instantiation).
 * @return The dimension's CTValue if it is known and statically a natural
 * number (CTValue::isNat()); std::nullopt if unknown or symbolic.
 */
std::optional<CTValue> TypeChecker::extractDimension(TypeId type, size_t dim_index) const {
    // Look up dimension info from the type environment's cache
    auto dim_info = env_.getDimensionInfo(type);
    if (dim_info && dim_index < dim_info->size()) {
        const auto& dim = (*dim_info)[dim_index];
        if (dim.isNat()) {
            return CTValue::makeNat(dim.nat_value);
        }
    }

    // Check if this is a tensor type with known dimensions
    if (type == BuiltinTypes::Tensor) {
        // Check dimension cache again (may have been stored with base type)
        auto tensor_dims = env_.getDimensionInfo(BuiltinTypes::Tensor);
        if (tensor_dims && dim_index < tensor_dims->size()) {
            const auto& dim = (*tensor_dims)[dim_index];
            if (dim.isNat()) {
                return CTValue::makeNat(dim.nat_value);
            }
        }
        return std::nullopt;
    }

    // Check if this is a vector type
    if (type == BuiltinTypes::Vector) {
        // Check dimension cache with base type
        auto vec_dims = env_.getDimensionInfo(BuiltinTypes::Vector);
        if (vec_dims && dim_index < vec_dims->size()) {
            const auto& dim = (*vec_dims)[dim_index];
            if (dim.isNat()) {
                return CTValue::makeNat(dim.nat_value);
            }
        }
        return std::nullopt;
    }

    return std::nullopt;
}

// ============================================================================
// Linear Type Checking (Phase 6.2)
// ============================================================================

/**
 * @brief True if @p type is linear (must be used exactly once): the no-cloning constraint.
 *
 * Checks both @p type's own TYPE_FLAG_LINEAR bit and, as a fallback, the
 * flags on the type's registered TypeNode in the environment
 * (TypeEnvironment::getTypeNode()), in case the flag was set at
 * registration time rather than on this particular TypeId instance.
 */
bool TypeChecker::isLinearType(TypeId type) const {
    // Check if the type has the LINEAR flag set
    if (type.flags & TYPE_FLAG_LINEAR) {
        return true;
    }

    // Also check the TypeNode in the environment
    const TypeNode* node = env_.getTypeNode(type);
    if (node && (node->id.flags & TYPE_FLAG_LINEAR)) {
        return true;
    }

    return false;
}

/**
 * @brief Check linear variable usage across the current context, recording
 * an error for every violation found.
 *
 * Reports (via addError()) one error per linear variable in
 * Context::getUnusedLinear() (never used) and one per variable in
 * Context::getOverusedLinear() (used more than once). Does not itself
 * return a result — callers inspect errors() afterward.
 */
void TypeChecker::checkLinearUsage() {
    // Check for unused linear variables
    auto unused = ctx_.getUnusedLinear();
    for (const auto& name : unused) {
        addError("Linear variable '" + name + "' was never used (must be used exactly once)");
    }

    // Check for overused linear variables
    auto overused = ctx_.getOverusedLinear();
    for (const auto& name : overused) {
        addError("Linear variable '" + name + "' was used more than once (must be used exactly once)");
    }
}

/**
 * @brief Verify a single use-site of variable @p name, enforcing linear no-cloning.
 *
 * If @p name is not tracked as linear, this is just a normal variable
 * lookup. If it is linear, checks whether it was already used
 * (Context::isLinearUsed()): in isUnsafe() mode, a repeat use is tolerated
 * (still recorded via Context::useLinear() for tracking) and its type
 * returned; otherwise a repeat use is an error. On a fresh (first) use,
 * marks it used and returns its type.
 * @return The variable's type on success; an error if @p name is unbound,
 * or if it is linear and already used outside unsafe mode.
 */
TypeCheckResult TypeChecker::checkLinearVariable(const std::string& name) {
    // Check if this is a linear variable
    if (!ctx_.isLinear(name)) {
        // Not a linear variable, no special checking needed
        auto type = ctx_.lookup(name);
        if (type) {
            return TypeCheckResult::ok(*type);
        }
        return TypeCheckResult::error("Undefined variable: " + name);
    }

    // It's a linear variable - check if it was already used
    if (ctx_.isLinearUsed(name)) {
        // In unsafe context, allow multiple uses
        if (isUnsafe()) {
            // Mark as used anyway for tracking
            ctx_.useLinear(name);
            auto type = ctx_.lookup(name);
            if (type) {
                return TypeCheckResult::ok(*type);
            }
        }
        return TypeCheckResult::error("Linear variable '" + name + "' was already used (cannot use more than once)");
    }

    // Mark as used
    ctx_.useLinear(name);

    // Return the type
    auto type = ctx_.lookup(name);
    if (type) {
        return TypeCheckResult::ok(*type);
    }
    return TypeCheckResult::error("Undefined linear variable: " + name);
}

/**
 * @brief Check that all linear @p bindings introduced by a let scope were consumed exactly once.
 *
 * Intended to be called at the end of a let scope's lifetime. In isUnsafe()
 * mode, all checks are skipped and success is returned unconditionally.
 * Otherwise, filters @p bindings down to those tracked as linear
 * (Context::isLinear()) and collects any that were never used
 * (Context::isLinearUsed() false); overuse is not re-checked here since
 * checkLinearVariable() already reports it at the use site. If any bindings
 * are unused, records and returns a single combined error listing all of
 * their names.
 * @return TypeCheckResult::ok(BuiltinTypes::Null) if unsafe or all linear
 * bindings were used; otherwise an error naming the unconsumed bindings.
 */
TypeCheckResult TypeChecker::checkLinearLet(const std::vector<std::string>& bindings) {
    // Check that all linear bindings were consumed exactly once
    // This is called at the end of a let scope

    // In unsafe context, skip these checks
    if (isUnsafe()) {
        return TypeCheckResult::ok(BuiltinTypes::Null);
    }

    std::vector<std::string> unused;
    std::vector<std::string> overused;

    for (const auto& name : bindings) {
        if (!ctx_.isLinear(name)) {
            continue;  // Only check linear variables
        }

        if (!ctx_.isLinearUsed(name)) {
            unused.push_back(name);
        }
        // Note: overuse is checked in checkLinearVariable
    }

    // Report unused linear variables
    if (!unused.empty()) {
        std::ostringstream ss;
        ss << "Linear variable(s) not consumed: ";
        bool first = true;
        for (const auto& name : unused) {
            if (!first) ss << ", ";
            ss << "'" << name << "'";
            first = false;
        }
        ss << " (linear types must be used exactly once)";
        addError(ss.str());
        return TypeCheckResult::error(ss.str());
    }

    return TypeCheckResult::ok(BuiltinTypes::Null);
}

// ============================================================================
// Unsafe Context (Phase 6.4)
// ============================================================================

/**
 * @brief Enter an unsafe block: relax linear/borrow restrictions for its duration.
 *
 * Delegates to UnsafeContext::enterUnsafe(), incrementing the nesting depth.
 */
void TypeChecker::enterUnsafe() {
    unsafe_.enterUnsafe();
}

/**
 * @brief Exit the innermost unsafe block, restoring the previous safety level.
 *
 * Delegates to UnsafeContext::exitUnsafe(), decrementing the nesting depth.
 */
void TypeChecker::exitUnsafe() {
    unsafe_.exitUnsafe();
}

/**
 * @brief True if the checker is currently inside one or more nested unsafe blocks.
 */
bool TypeChecker::isUnsafe() const {
    return unsafe_.isUnsafe();
}

// ============================================================================
// Program-Level Type Checking
// ============================================================================

/**
 * @brief Type check a complete program: synthesize the type of every
 * top-level AST in @p asts.
 *
 * Constructs a fresh TypeChecker over @p env (using default strict/unsafe
 * settings) and calls synthesize() on each of the @p num_asts top-level
 * definitions in order. If @p strict is true, returns immediately with the
 * accumulated errors as soon as any top-level synthesis fails; if false,
 * continues checking all remaining top-level forms regardless of earlier
 * failures so all errors can be reported together.
 * @param env The type environment.
 * @param asts Array of top-level ASTs.
 * @param num_asts Number of ASTs.
 * @param strict If true, stop and return on the first error; if false,
 * collect all errors across the whole program.
 * @return The type checker's accumulated errors (empty if the program is
 * well-typed).
 */
std::vector<TypeCheckResult> typeCheckProgram(
    TypeEnvironment& env,
    eshkol_ast_t* asts,
    size_t num_asts,
    bool strict) {

    TypeChecker checker(env);

    for (size_t i = 0; i < num_asts; i++) {
        auto result = checker.synthesize(&asts[i]);
        if (!result.success) {
            if (strict) {
                return checker.errors();
            }
        }
    }

    return checker.errors();
}

} // namespace eshkol::hott
