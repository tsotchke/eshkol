/**
 * @file type_checker.cpp
 * @brief Implementation of bidirectional type checking for Eshkol's HoTT type system
 */

#include "eshkol/types/type_checker.h"
#include <sstream>
#include <cstring>
#include <cstdlib>

namespace eshkol::hott {

// ============================================================================
// Context Implementation
// ============================================================================

Context::Context() {
    // Start with global scope
    scopes_.push_back({});
}

void Context::pushScope() {
    scopes_.push_back({});
}

void Context::popScope() {
    if (scopes_.size() > 1) {
        scopes_.pop_back();
    }
}

void Context::bind(const std::string& name, TypeId type) {
    if (!scopes_.empty()) {
        scopes_.back()[name] = type;
    }
}

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

void Context::defineTypeAlias(const std::string& name, hott_type_expr_t* type_expr,
                              const std::vector<std::string>& params) {
    type_aliases_[name] = type_expr;
    // Store parameters for parameterized type aliases (e.g., (define-type (MyList a) (list a)))
    if (!params.empty()) {
        type_alias_params_[name] = params;
    }
}

bool Context::hasTypeAliasParams(const std::string& name) const {
    return type_alias_params_.find(name) != type_alias_params_.end();
}

const std::vector<std::string>& Context::getTypeAliasParams(const std::string& name) const {
    auto it = type_alias_params_.find(name);
    if (it != type_alias_params_.end()) {
        return it->second;
    }
    return empty_params_;
}

std::optional<hott_type_expr_t*> Context::lookupTypeAlias(const std::string& name) const {
    auto found = type_aliases_.find(name);
    if (found != type_aliases_.end()) {
        return found->second;
    }
    return std::nullopt;
}

// Helper function to allocate a type expression
static hott_type_expr_t* allocTypeExpr(hott_type_kind_t kind) {
    hott_type_expr_t* type = (hott_type_expr_t*)malloc(sizeof(hott_type_expr_t));
    if (type) {
        memset(type, 0, sizeof(hott_type_expr_t));
        type->kind = kind;
    }
    return type;
}

// Helper function to substitute type variables in a type expression
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
                result->arrow.param_types = (hott_type_expr_t**)malloc(
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
                result->forall.type_vars = (char**)malloc(
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

void LinearContext::declareLinear(const std::string& name) {
    linear_vars_.insert(name);
    usage_[name] = Usage::Unused;
}

void LinearContext::use(const std::string& name) {
    if (linear_vars_.count(name) == 0) return;

    auto& u = usage_[name];
    if (u == Usage::Unused) {
        u = Usage::UsedOnce;
    } else {
        u = Usage::UsedMultiple;
    }
}

bool LinearContext::checkAllUsedOnce() const {
    for (const auto& name : linear_vars_) {
        auto it = usage_.find(name);
        if (it == usage_.end() || it->second != Usage::UsedOnce) {
            return false;
        }
    }
    return true;
}

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

void LinearContext::consume(const std::string& name) {
    // Mark as used exactly once (same as use() semantically, but clearer for API)
    use(name);
}

LinearContext::Usage LinearContext::getUsage(const std::string& name) const {
    auto it = usage_.find(name);
    if (it == usage_.end()) {
        return Usage::Unused;
    }
    return it->second;
}

bool LinearContext::isLinear(const std::string& name) const {
    return linear_vars_.count(name) > 0;
}

// ============================================================================
// BorrowChecker Implementation (Phase 6.3)
// ============================================================================

void BorrowChecker::pushScope() {
    current_scope_++;
}

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

void BorrowChecker::declareOwned(const std::string& name) {
    BorrowInfo info;
    info.state = BorrowState::Owned;
    info.scope_depth = current_scope_;
    values_[name] = info;
}

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

BorrowState BorrowChecker::getState(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return BorrowState::Owned;  // Unknown = assume owned
    }
    return it->second.state;
}

bool BorrowChecker::canUse(const std::string& name) const {
    auto it = values_.find(name);
    if (it == values_.end()) {
        return true;  // Unknown = assume usable
    }
    return it->second.state != BorrowState::Moved &&
           it->second.state != BorrowState::Dropped;
}

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

void BorrowChecker::addError(BorrowError::Kind kind, const std::string& var,
                              const std::string& msg) {
    errors_.push_back({kind, var, msg});
}

// ============================================================================
// Context Linear Methods (Phase 6)
// ============================================================================

void Context::bindLinear(const std::string& name, TypeId type) {
    // Bind the variable normally
    bind(name, type);
    // Mark it as linear
    linear_vars_.insert(name);
    linear_usage_count_[name] = 0;  // Unused initially
}

void Context::useLinear(const std::string& name) {
    if (linear_vars_.count(name) > 0) {
        linear_usage_count_[name]++;
    }
}

void Context::consumeLinear(const std::string& name) {
    // Same as useLinear for tracking
    useLinear(name);
}

bool Context::isLinear(const std::string& name) const {
    return linear_vars_.count(name) > 0;
}

bool Context::isLinearUsed(const std::string& name) const {
    auto it = linear_usage_count_.find(name);
    return it != linear_usage_count_.end() && it->second > 0;
}

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

TypeChecker::TypeChecker(TypeEnvironment& env) : env_(env) {}

// Helper to store inferred type in AST and return result
static TypeCheckResult storeAndReturn(eshkol_ast_t* expr, TypeCheckResult result) {
    if (result.success && expr) {
        expr->inferred_hott_type = result.inferred_type.pack();
    }
    return result;
}

// Helper to create error with source location from AST node
static TypeCheckResult errorAt(const eshkol_ast_t* expr, const std::string& msg) {
    if (expr) {
        return TypeCheckResult::error(msg, expr->line, expr->column);
    }
    return TypeCheckResult::error(msg);
}

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

TypeCheckResult TypeChecker::synthesizeLiteral(eshkol_ast_t* expr) {
    switch (expr->type) {
        case ESHKOL_INT64:
            return TypeCheckResult::ok(BuiltinTypes::Int64);

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

        default:
            return TypeCheckResult::error("Unknown literal type");
    }
}

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
        return errorAt(expr, "Unbound variable: " + name);
    }

    return TypeCheckResult::ok(*type);
}

TypeCheckResult TypeChecker::synthesizeOperation(eshkol_ast_t* expr) {
    switch (expr->operation.op) {
        case ESHKOL_DEFINE_OP:
            return synthesizeDefine(expr);

        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
            return synthesizeLet(expr);

        case ESHKOL_IF_OP:
            return synthesizeIf(expr);

        case ESHKOL_LAMBDA_OP:
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

        default:
            // Unknown or complex operation - return Value (top type)
            return TypeCheckResult::ok(BuiltinTypes::Value);
    }
}

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
        ctx_.bind(param_name, param_type);
    }

    // Synthesize body type
    auto body_result = synthesize(lambda.body);

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
        return TypeCheckResult::ok(env_.makeFunctionType({}, return_type));
    }

    return TypeCheckResult::ok(env_.makeFunctionType(param_types, return_type));
}

TypeCheckResult TypeChecker::synthesizeApplication(eshkol_ast_t* expr) {
    const auto& call = expr->operation.call_op;

    // The function being called is in call.func, arguments are in call.variables
    const eshkol_ast_t* func_expr = call.func;

    // If we have no function expression, return Value
    if (!func_expr) {
        return TypeCheckResult::ok(BuiltinTypes::Value);
    }

    // Check for builtin operators first
    if (func_expr->type == ESHKOL_VAR && func_expr->variable.id) {
        std::string func_name = func_expr->variable.id;

        // Arithmetic operators: synthesize operand types and promote
        if (func_name == "+" || func_name == "-" || func_name == "*" || func_name == "/") {
            if (call.num_vars >= 2) {  // Need at least 2 operands
                auto left = synthesize(const_cast<eshkol_ast_t*>(&call.variables[0]));
                auto right = synthesize(const_cast<eshkol_ast_t*>(&call.variables[1]));
                if (left.success && right.success) {
                    TypeId result_type = env_.promoteForArithmetic(
                        left.inferred_type, right.inferred_type);
                    return TypeCheckResult::ok(result_type);
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Number);
        }

        // Comparison operators return Boolean
        if (func_name == "<" || func_name == ">" || func_name == "<=" ||
            func_name == ">=" || func_name == "=" || func_name == "equal?" ||
            func_name == "eq?" || func_name == "eqv?" || func_name == "!=" ||
            func_name == "zero?" || func_name == "positive?" || func_name == "negative?" ||
            func_name == "odd?" || func_name == "even?" || func_name == "null?" ||
            func_name == "pair?" || func_name == "list?" || func_name == "number?" ||
            func_name == "string?" || func_name == "symbol?" || func_name == "boolean?" ||
            func_name == "vector?" || func_name == "procedure?") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // List operations
        if (func_name == "car" || func_name == "first") {
            return TypeCheckResult::ok(BuiltinTypes::Value);  // Element type unknown
        }
        if (func_name == "cdr" || func_name == "rest") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }
        if (func_name == "cons") {
            return TypeCheckResult::ok(BuiltinTypes::Pair);
        }
        if (func_name == "list" || func_name == "append" ||
            func_name == "reverse" || func_name == "map" || func_name == "filter") {
            return TypeCheckResult::ok(BuiltinTypes::List);
        }
        if (func_name == "length") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }

        // String operations
        if (func_name == "string-append" || func_name == "substring" ||
            func_name == "string-upcase" || func_name == "string-downcase" ||
            func_name == "number->string" || func_name == "symbol->string") {
            return TypeCheckResult::ok(BuiltinTypes::String);
        }
        if (func_name == "string-length" || func_name == "string->number") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "string=?" || func_name == "string<?" || func_name == "string>?") {
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
            // Synthesize argument to determine if result is int or float
            if (call.num_vars >= 1) {
                auto arg = synthesize(const_cast<eshkol_ast_t*>(&call.variables[0]));
                if (arg.success) {
                    // Most math functions return Float64, some preserve int
                    if (func_name == "abs" || func_name == "min" || func_name == "max" ||
                        func_name == "modulo" || func_name == "remainder" || func_name == "quotient") {
                        return TypeCheckResult::ok(arg.inferred_type);
                    }
                }
            }
            return TypeCheckResult::ok(BuiltinTypes::Float64);
        }

        // Vector operations
        if (func_name == "vector" || func_name == "make-vector" || func_name == "vector-copy") {
            return TypeCheckResult::ok(BuiltinTypes::Vector);
        }
        if (func_name == "vector-ref") {
            return TypeCheckResult::ok(BuiltinTypes::Value);
        }
        if (func_name == "vector-length") {
            return TypeCheckResult::ok(BuiltinTypes::Int64);
        }
        if (func_name == "vector-set!") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        // I/O operations
        if (func_name == "display" || func_name == "newline" || func_name == "write" ||
            func_name == "set!") {
            return TypeCheckResult::ok(BuiltinTypes::Null);
        }

        // Begin expression - returns type of last expression
        if (func_name == "begin") {
            if (call.num_vars == 0) {
                return TypeCheckResult::ok(BuiltinTypes::Null);
            }
            // Synthesize all expressions, return type of last one
            for (size_t i = 0; i < call.num_vars - 1; ++i) {
                synthesize(const_cast<eshkol_ast_t*>(&call.variables[i]));
            }
            return synthesize(const_cast<eshkol_ast_t*>(&call.variables[call.num_vars - 1]));
        }
        if (func_name == "read" || func_name == "read-line") {
            return TypeCheckResult::ok(BuiltinTypes::String);
        }

        // Boolean operations
        if (func_name == "not" || func_name == "and" || func_name == "or") {
            return TypeCheckResult::ok(BuiltinTypes::Boolean);
        }

        // If expression - synthesize branch types and compute LCS
        if (func_name == "if") {
            // if has: condition (0), then-branch (1), else-branch (2)
            if (call.num_vars >= 2) {
                auto then_type = synthesize(const_cast<eshkol_ast_t*>(&call.variables[1]));
                if (!then_type.success) return then_type;

                if (call.num_vars >= 3) {
                    auto else_type = synthesize(const_cast<eshkol_ast_t*>(&call.variables[2]));
                    if (!else_type.success) return else_type;

                    TypeId then_t = then_type.inferred_type;
                    TypeId else_t = else_type.inferred_type;

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
                return then_type;
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

        // Check argument count if we have function type info
        const PiType* pi = env_.getFunctionType(func_type);
        if (pi && pi->params.size() != call.num_vars) {
            // Arity mismatch - not necessarily an error for curried functions
            // For now, just return the return type
        }

        return TypeCheckResult::ok(return_type);
    }

    // Fallback: unknown function type
    return TypeCheckResult::ok(BuiltinTypes::Value);
}

TypeCheckResult TypeChecker::synthesizeDefine(eshkol_ast_t* expr) {
    const auto& def = expr->operation.define_op;

    if (!def.name) {
        return errorAt(expr, "Define without name");
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
        TypeId func_type = env_.makeFunctionType(param_types, declared_return);
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

        // Create proper function type with param and return types
        value_type = TypeCheckResult::ok(env_.makeFunctionType(param_types, return_type));
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

TypeCheckResult TypeChecker::checkLambda(eshkol_ast_t* expr, TypeId expected) {
    // If expected is a function type, use its domain for param types
    // For now, just synthesize
    return synthesizeLambda(expr);
}

bool TypeChecker::isFunctionType(TypeId type, TypeId& domain, TypeId& codomain) const {
    // Check if type is Function or a parameterized function type
    // Simplified for now
    return type.id == BuiltinTypes::Function.id;
}

bool TypeChecker::unify(TypeId a, TypeId b) {
    // Simple unification - just check equality or subtyping
    return a == b || env_.isSubtype(a, b) || env_.isSubtype(b, a);
}

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

        case HOTT_TYPE_PAIR:
            return BuiltinTypes::Pair;

        case HOTT_TYPE_PRODUCT:
            return BuiltinTypes::Pair;  // Represent as pair for now

        case HOTT_TYPE_SUM:
            return BuiltinTypes::Value;  // Sum types need more work

        case HOTT_TYPE_FORALL:
            // Polymorphic - return Value for now
            return BuiltinTypes::Value;

        default:
            return BuiltinTypes::Value;
    }
}

void TypeChecker::addError(const std::string& msg, int line, int col) {
    errors_.push_back(TypeCheckResult::error(msg, line, col));
}

void TypeChecker::addTypeMismatch(TypeId expected, TypeId actual, int line, int col) {
    std::ostringstream ss;
    ss << "Type mismatch: expected " << env_.getTypeName(expected)
       << ", got " << env_.getTypeName(actual);
    addError(ss.str(), line, col);
}

// ============================================================================
// Dimension Checking (Phase 5.3)
// ============================================================================

TypeCheckResult TypeChecker::checkVectorBounds(const CTValue& index, const CTValue& bound,
                                               const std::string& context) {
    auto result = DimensionChecker::checkBounds(index, bound, context);
    if (result.valid) {
        return TypeCheckResult::ok(BuiltinTypes::Boolean);
    }
    addError(result.error_message);
    return TypeCheckResult::error(result.error_message);
}

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

void TypeChecker::enterUnsafe() {
    unsafe_.enterUnsafe();
}

void TypeChecker::exitUnsafe() {
    unsafe_.exitUnsafe();
}

bool TypeChecker::isUnsafe() const {
    return unsafe_.isUnsafe();
}

// ============================================================================
// Program-Level Type Checking
// ============================================================================

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
