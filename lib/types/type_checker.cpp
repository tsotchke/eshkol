/**
 * @file type_checker.cpp
 * @brief Implementation of bidirectional type checking for Eshkol's HoTT type system
 */

#include "eshkol/types/type_checker.h"
#include <sstream>

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
    // TODO: Store params for parameterized type aliases
}

std::optional<hott_type_expr_t*> Context::lookupTypeAlias(const std::string& name) const {
    auto found = type_aliases_.find(name);
    if (found != type_aliases_.end()) {
        return found->second;
    }
    return std::nullopt;
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

TypeCheckResult TypeChecker::synthesize(const eshkol_ast_t* expr) {
    if (!expr) {
        return TypeCheckResult::error("Null expression");
    }

    switch (expr->type) {
        case ESHKOL_INT64:
        case ESHKOL_DOUBLE:
        case ESHKOL_STRING:
        case ESHKOL_BOOL:
        case ESHKOL_NULL:
        case ESHKOL_CHAR:
            return synthesizeLiteral(expr);

        case ESHKOL_VAR:
            return synthesizeVariable(expr);

        case ESHKOL_OP:
            return synthesizeOperation(expr);

        case ESHKOL_FUNC:
            return synthesizeLambda(expr);

        case ESHKOL_CONS:
            // Cons cell - synthesize as list
            return TypeCheckResult::ok(BuiltinTypes::List);

        default:
            return TypeCheckResult::error("Cannot synthesize type for this expression");
    }
}

TypeCheckResult TypeChecker::check(const eshkol_ast_t* expr, TypeId expected) {
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

    addTypeMismatch(expected, result.inferred_type);
    return TypeCheckResult::error(
        "Type mismatch: expected " + env_.getTypeName(expected) +
        ", got " + env_.getTypeName(result.inferred_type));
}

TypeCheckResult TypeChecker::synthesizeLiteral(const eshkol_ast_t* expr) {
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

TypeCheckResult TypeChecker::synthesizeVariable(const eshkol_ast_t* expr) {
    if (!expr->variable.id) {
        return TypeCheckResult::error("Variable has no name");
    }

    std::string name = expr->variable.id;
    auto type = ctx_.lookup(name);
    if (!type) {
        return TypeCheckResult::error("Unbound variable: " + name);
    }

    return TypeCheckResult::ok(*type);
}

TypeCheckResult TypeChecker::synthesizeOperation(const eshkol_ast_t* expr) {
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

        case ESHKOL_DEFINE_TYPE_OP:
            // Type definition - register alias and return null
            if (expr->operation.define_type_op.name &&
                expr->operation.define_type_op.type_expr) {
                ctx_.defineTypeAlias(
                    expr->operation.define_type_op.name,
                    expr->operation.define_type_op.type_expr);
            }
            return TypeCheckResult::ok(BuiltinTypes::Null);

        // Arithmetic operations
        case ESHKOL_ADD_OP:
        case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP:
        case ESHKOL_DIV_OP: {
            if (expr->operation.call_op.num_vars < 2) {
                return TypeCheckResult::error("Arithmetic requires at least 2 operands");
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

TypeCheckResult TypeChecker::synthesizeLambda(const eshkol_ast_t* expr) {
    const auto& lambda = expr->operation.lambda_op;

    ctx_.pushScope();

    // Bind parameters in context
    for (size_t i = 0; i < lambda.num_params; i++) {
        std::string param_name = lambda.parameters[i].variable.id;

        // Get type from annotation if available
        TypeId param_type = BuiltinTypes::Value;  // Default
        if (lambda.param_types && lambda.param_types[i]) {
            param_type = resolveType(lambda.param_types[i]);
        }

        ctx_.bind(param_name, param_type);
    }

    // Synthesize body type
    auto body_result = synthesize(lambda.body);

    ctx_.popScope();

    if (!body_result.success) {
        return body_result;
    }

    // If return type annotation exists, check against it
    if (lambda.return_type) {
        TypeId expected_return = resolveType(lambda.return_type);
        if (!env_.isSubtype(body_result.inferred_type, expected_return)) {
            return TypeCheckResult::error(
                "Lambda body type " + env_.getTypeName(body_result.inferred_type) +
                " doesn't match return annotation " + env_.getTypeName(expected_return));
        }
    }

    // Return function type
    // For now, just return Function; full function type would encode params
    return TypeCheckResult::ok(BuiltinTypes::Function);
}

TypeCheckResult TypeChecker::synthesizeApplication(const eshkol_ast_t* expr) {
    // For now, just return Value since we don't have full function type info
    // A complete implementation would check the function type and return its codomain
    return TypeCheckResult::ok(BuiltinTypes::Value);
}

TypeCheckResult TypeChecker::synthesizeDefine(const eshkol_ast_t* expr) {
    const auto& def = expr->operation.define_op;

    if (!def.name) {
        return TypeCheckResult::error("Define without name");
    }

    TypeCheckResult value_type = TypeCheckResult::ok(BuiltinTypes::Value);

    if (def.is_function) {
        // Function define: (define (name params...) body)
        // Handle like a lambda - bind parameters and check body
        ctx_.pushScope();

        // Bind parameters in scope
        for (size_t i = 0; i < def.num_params; i++) {
            if (def.parameters && def.parameters[i].type == ESHKOL_VAR &&
                def.parameters[i].variable.id) {
                TypeId param_type = BuiltinTypes::Value;
                if (def.param_types && def.param_types[i]) {
                    param_type = resolveType(def.param_types[i]);
                }
                ctx_.bind(def.parameters[i].variable.id, param_type);
            }
        }

        // Type check the body if present
        if (def.value) {
            value_type = synthesize(def.value);
        }

        ctx_.popScope();

        // Create function type
        TypeId return_type = value_type.success ? value_type.inferred_type : BuiltinTypes::Value;
        if (def.return_type) {
            return_type = resolveType(def.return_type);
        }
        value_type = TypeCheckResult::ok(BuiltinTypes::Function);
    } else {
        // Variable define: (define name value)
        if (def.value) {
            value_type = synthesize(def.value);
            if (!value_type.success) {
                return value_type;
            }
        }
    }

    // Bind in context
    ctx_.bind(def.name, value_type.inferred_type);

    return TypeCheckResult::ok(BuiltinTypes::Null);
}

TypeCheckResult TypeChecker::synthesizeLet(const eshkol_ast_t* expr) {
    const auto& let = expr->operation.let_op;

    ctx_.pushScope();

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

        ctx_.bind(name, binding_type);
    }

    // Synthesize body type
    auto body_result = synthesize(let.body);

    ctx_.popScope();

    return body_result;
}

TypeCheckResult TypeChecker::synthesizeIf(const eshkol_ast_t* expr) {
    // if has: condition, then-branch, else-branch
    // For now, just synthesize branches and take LCS
    if (expr->operation.call_op.num_vars < 2) {
        return TypeCheckResult::error("if requires condition and then-branch");
    }

    auto then_type = synthesize(&expr->operation.call_op.variables[1]);
    if (!then_type.success) return then_type;

    if (expr->operation.call_op.num_vars >= 3) {
        auto else_type = synthesize(&expr->operation.call_op.variables[2]);
        if (!else_type.success) return else_type;

        // Compute LCS of branches
        auto lcs = env_.leastCommonSupertype(then_type.inferred_type, else_type.inferred_type);
        if (lcs) {
            return TypeCheckResult::ok(*lcs);
        }
    }

    return then_type;
}

TypeCheckResult TypeChecker::checkLambda(const eshkol_ast_t* expr, TypeId expected) {
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

        case HOTT_TYPE_ARROW:
            return BuiltinTypes::Function;

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
    // For now, we don't have runtime dimension tracking in TypeId
    // This would require looking up dependent type info
    // Return nullopt to indicate dimension is unknown

    // TODO: When we have full dependent type tracking, look up the type's
    // dimension parameters here

    // Check if this is a tensor type with known dimensions
    if (type == BuiltinTypes::Tensor) {
        // Tensor dimensions are runtime-determined currently
        return std::nullopt;
    }

    // Check if this is a vector type
    if (type == BuiltinTypes::Vector) {
        // Vector dimensions are runtime-determined currently
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
    const eshkol_ast_t* asts,
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
