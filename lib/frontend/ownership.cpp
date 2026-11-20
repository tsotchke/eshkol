#include <eshkol/frontend/ownership.h>
#include <eshkol/frontend/syntax_color.h>
#include <eshkol/types/type_checker.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <sstream>
#include <utility>

namespace eshkol {
namespace {
namespace sr = strict_resources;

// ── value flow helpers for alias and escape checking ────────────────────
// Every expression a node evaluates or closes over, following the payload of
// each operator the macro expander traverses. Leaf operators have none.
template <class F>
void forEachChildExpr(const eshkol_ast_t* e, F&& f) {
    if (!e) return;
    if (e->type == ESHKOL_CONS) {
        f(e->cons_cell.car);
        f(e->cons_cell.cdr);
        return;
    }
    if (e->type != ESHKOL_OP) return;
    const auto& op = e->operation;
    auto each = [&](const eshkol_ast_t* nodes, uint64_t count) {
        if (nodes)
            for (uint64_t i = 0; i < count; ++i) f(&nodes[i]);
    };
    switch (op.op) {
        case ESHKOL_SEQUENCE_OP:
        case ESHKOL_AND_OP:
        case ESHKOL_OR_OP:
            each(op.sequence_op.expressions, op.sequence_op.num_expressions);
            return;
        case ESHKOL_DEFINE_OP:
            f(op.define_op.value);
            return;
        case ESHKOL_LAMBDA_OP:
            f(op.lambda_op.body);
            return;
        case ESHKOL_LET_OP:
        case ESHKOL_LET_STAR_OP:
        case ESHKOL_LETREC_OP:
        case ESHKOL_LETREC_STAR_OP:
            for (uint64_t i = 0; op.let_op.bindings && i < op.let_op.num_bindings; ++i) {
                const eshkol_ast_t& binding = op.let_op.bindings[i];
                if (binding.type == ESHKOL_CONS) f(binding.cons_cell.cdr);
            }
            f(op.let_op.body);
            return;
        case ESHKOL_MATCH_OP:
            f(op.match_op.expr);
            for (uint64_t i = 0; op.match_op.clauses && i < op.match_op.num_clauses; ++i) {
                f(op.match_op.clauses[i].guard);
                f(op.match_op.clauses[i].body);
            }
            return;
        case ESHKOL_SET_OP:
            f(op.set_op.value);
            return;
        case ESHKOL_GUARD_OP:
            each(op.guard_op.clauses, op.guard_op.num_clauses);
            each(op.guard_op.body, op.guard_op.num_body_exprs);
            return;
        case ESHKOL_RAISE_OP:
            f(op.raise_op.exception);
            return;
        case ESHKOL_VALUES_OP:
            each(op.values_op.expressions, op.values_op.num_values);
            return;
        case ESHKOL_CALL_CC_OP:
            f(op.call_cc_op.proc);
            return;
        case ESHKOL_DYNAMIC_WIND_OP:
            f(op.dynamic_wind_op.before);
            f(op.dynamic_wind_op.thunk);
            f(op.dynamic_wind_op.after);
            return;
        case ESHKOL_THE_OP:
            f(op.the_op.expr);
            return;
        case ESHKOL_COMPOSE_OP:
            f(op.compose_op.func_a);
            f(op.compose_op.func_b);
            return;
        case ESHKOL_TENSOR_OP:
            each(op.tensor_op.elements, op.tensor_op.total_elements);
            return;
        case ESHKOL_DIFF_OP:
            f(op.diff_op.expression);
            return;
        case ESHKOL_DERIVATIVE_OP:
            f(op.derivative_op.function);
            f(op.derivative_op.point);
            return;
        case ESHKOL_GRADIENT_OP:
        case ESHKOL_JACOBIAN_OP:
        case ESHKOL_HESSIAN_OP:
        case ESHKOL_DIVERGENCE_OP:
        case ESHKOL_CURL_OP:
        case ESHKOL_LAPLACIAN_OP:
            f(op.gradient_op.function);
            f(op.gradient_op.point);
            return;
        case ESHKOL_DIRECTIONAL_DERIV_OP:
            f(op.directional_deriv_op.function);
            f(op.directional_deriv_op.point);
            f(op.directional_deriv_op.direction);
            return;
        case ESHKOL_TAYLOR_OP:
        case ESHKOL_DERIVATIVE_N_OP:
            f(op.taylor_op.function);
            f(op.taylor_op.point);
            f(op.taylor_op.order);
            return;
        case ESHKOL_WITH_REGION_OP:
            each(op.with_region_op.body, op.with_region_op.num_body_exprs);
            return;
        case ESHKOL_OWNED_OP:
            f(op.owned_op.value);
            return;
        case ESHKOL_MOVE_OP:
            f(op.move_op.value);
            return;
        case ESHKOL_SHARED_OP:
            f(op.shared_op.value);
            return;
        case ESHKOL_WEAK_REF_OP:
            f(op.weak_ref_op.value);
            return;
        case ESHKOL_BORROW_OP:
            f(op.borrow_op.value);
            each(op.borrow_op.body, op.borrow_op.num_body_exprs);
            return;
        case ESHKOL_CALL_WITH_VALUES_OP:
            f(op.call_with_values_op.producer);
            f(op.call_with_values_op.consumer);
            return;
        case ESHKOL_LET_VALUES_OP:
        case ESHKOL_LET_STAR_VALUES_OP:
            each(op.let_values_op.producers, op.let_values_op.num_bindings);
            f(op.let_values_op.body);
            return;
        case ESHKOL_CASE_LAMBDA_OP:
            each(op.case_lambda_op.clauses, op.case_lambda_op.num_clauses);
            return;
        case ESHKOL_PARAMETERIZE_OP:
            each(op.parameterize_op.params, op.parameterize_op.num_bindings);
            each(op.parameterize_op.values, op.parameterize_op.num_bindings);
            f(op.parameterize_op.body);
            return;
        case ESHKOL_CALL_OP:
        case ESHKOL_IF_OP:
        case ESHKOL_ADD_OP:
        case ESHKOL_SUB_OP:
        case ESHKOL_MUL_OP:
        case ESHKOL_DIV_OP:
        case ESHKOL_QUASIQUOTE_OP:
        case ESHKOL_UNQUOTE_OP:
        case ESHKOL_UNQUOTE_SPLICING_OP:
        case ESHKOL_COND_OP:
        case ESHKOL_CASE_OP:
        case ESHKOL_WHEN_OP:
        case ESHKOL_UNLESS_OP:
        case ESHKOL_DO_OP:
        case ESHKOL_UNIFY_OP: case ESHKOL_MAKE_SUBST_OP: case ESHKOL_WALK_OP:
        case ESHKOL_MAKE_FACT_OP: case ESHKOL_MAKE_KB_OP: case ESHKOL_KB_ASSERT_OP:
        case ESHKOL_KB_QUERY_OP: case ESHKOL_MAKE_FACTOR_GRAPH_OP: case ESHKOL_FG_ADD_FACTOR_OP:
        case ESHKOL_FG_INFER_OP: case ESHKOL_FREE_ENERGY_OP: case ESHKOL_EXPECTED_FREE_ENERGY_OP:
        case ESHKOL_MAKE_WORKSPACE_OP: case ESHKOL_WS_REGISTER_OP: case ESHKOL_WS_STEP_OP:
        case ESHKOL_FG_UPDATE_CPT_OP: case ESHKOL_FG_OBSERVE_OP: case ESHKOL_LOGIC_VAR_PRED_OP:
        case ESHKOL_SUBSTITUTION_PRED_OP: case ESHKOL_KB_PRED_OP: case ESHKOL_FACT_PRED_OP:
        case ESHKOL_FACTOR_GRAPH_PRED_OP: case ESHKOL_WORKSPACE_PRED_OP: case ESHKOL_KB_QUERY_PREFIX_OP:
        case ESHKOL_DNC_MAKE_OP: case ESHKOL_DNC_CONTENT_ADDR_OP: case ESHKOL_DNC_LOC_ADDR_OP:
        case ESHKOL_DNC_READ_OP: case ESHKOL_DNC_WRITE_OP: case ESHKOL_DNC_ALLOC_WEIGHTS_OP:
        case ESHKOL_DNC_READ_GRAD_OP: case ESHKOL_DNC_PRED_OP: case ESHKOL_SDNC_PROGRAM_OP:
        case ESHKOL_SDNC_RUN_OP: case ESHKOL_SDNC_WEIGHT_GRAD_OP: case ESHKOL_SDNC_PARAMS_OP:
        case ESHKOL_SDNC_SET_PARAMS_OP: case ESHKOL_SDNC_IMPROVE_OP: case ESHKOL_SDNC_PRED_OP:
            f(op.call_op.func);
            each(op.call_op.variables, op.call_op.num_vars);
            return;
        default:
            return;
    }
}

bool containsOwnershipForm(const eshkol_ast_t* e, unsigned depth = 0) {
    if (!e || depth > 4096) return e != nullptr;
    if (e->type == ESHKOL_OP && (e->operation.op == ESHKOL_OWNED_OP ||
                                 e->operation.op == ESHKOL_MOVE_OP ||
                                 e->operation.op == ESHKOL_BORROW_OP))
        return true;
    bool found = false;
    forEachChildExpr(e, [&](const eshkol_ast_t* child) {
        found = found || containsOwnershipForm(child, depth + 1);
    });
    return found;
}

// Builtins whose result holds no reference to an argument: numbers,
// booleans, characters, fresh strings and symbols, and the unspecified value.
bool isScalarResultBuiltin(const std::string& name) {
    if (!name.empty() && name.back() == '?') return true;
    static const std::set<std::string> names = {
        "+", "-", "*", "/", "=", "<", "<=", ">", ">=", "not", "quotient", "remainder",
        "modulo", "abs", "min", "max", "expt", "exp", "log", "sqrt", "sin", "cos", "tan",
        "asin", "acos", "atan", "floor", "ceiling", "round", "truncate", "square", "gcd",
        "lcm", "numerator", "denominator", "exact->inexact", "inexact->exact", "exact",
        "inexact", "number->string", "string->number", "length", "vector-length",
        "string-length", "string-ref", "char->integer", "integer->char", "symbol->string",
        "string->symbol", "string-append", "substring", "string-copy", "list->string",
        "string-upcase", "string-downcase", "display", "write", "newline", "write-string",
        "write-char", "print", "println", "void", "type-name", "bitwise-and", "bitwise-or",
        "bitwise-xor", "bitwise-not", "arithmetic-shift", "hash", "string-hash", "eq-hash"
    };
    return names.count(name) != 0;
}

// Builtins that may call a procedure argument with their other arguments.
bool callsProcedureArguments(const std::string& name) {
    static const char* const parts[] = {
        "map", "for-each", "fold", "reduce", "filter", "sort", "apply", "find", "any",
        "every", "count", "partition", "remove", "call-with", "call/cc", "with-",
        "dynamic-wind", "walk", "update", "spawn", "thread", "parallel", "future", "force",
        "promise", "tabulate", "iterate", "compose", "memo", "assoc", "member", "delete"
    };
    for (const char* part : parts)
        if (name.find(part) != std::string::npos) return true;
    return false;
}

// Builtins that keep an argument after they return. A Scheme mutator keeps
// every argument after the object it mutates; the others keep all of them.
enum class Keeps { NONE, AFTER_FIRST, ALL };
Keeps builtinKeeps(const std::string& name) {
    static const char* const parts[] = {
        "spawn", "thread", "register", "schedule", "async", "future", "send", "enqueue",
        "publish", "atexit", "set-", "parameter", "promise", "delay", "channel"
    };
    for (const char* part : parts)
        if (name.find(part) != std::string::npos) return Keeps::ALL;
    if (!name.empty() && name.back() == '!') return Keeps::AFTER_FIRST;
    return Keeps::NONE;
}

// What a procedure does with its arguments beyond computing its result.
struct ProcSummary {
    std::set<size_t> keeps;                       // parameter slots stored past the call
    std::map<size_t, std::set<size_t>> passes;    // procedure slot -> slots it receives
    size_t num_params = 0;
    bool rest = false;

    bool operator==(const ProcSummary& other) const {
        return keeps == other.keeps && passes == other.passes;
    }
    bool keepsNothing() const { return keeps.empty() && passes.empty(); }
    size_t slot(size_t argument) const {
        if (argument < num_params) return argument;
        return rest ? num_params : SIZE_MAX;
    }
};

class OwnershipAnalyzer {
public:
    using ResourceKind = sr::Kind;

    // Typestate of one tracked handle binding (docs/design/adr/0056-linear-handle-typestate.md).
    enum class ResourceState : uint8_t {
        NONE,
        ERROR_OR_LIVE,    // acquisition result not yet refined
        ERROR_ONLY,       // refined to the failure value: owns nothing
        LIVE,             // open
        CLOSED,           // consumed by its close operation
        CLOSED_OR_ERROR,  // closed on every path that acquired it
        MAYBE_LIVE,       // open on some paths only: an error at every use and at scope exit
        MOVED             // ownership transferred to another owner or returned
    };

    enum class CallableEffect : uint8_t {
        NONE,
        CONTRACT,          // a handle contract from inc/eshkol/strict_resource_contracts.h
        USER_FUNCTION,     // analyzed at each call site against the caller's handle states
        CONTINUATION,      // an escape-only continuation bound by call/cc
        LOOP,              // a named-let loop: tail calls are back edges
        TERMINAL_CONTROL,  // exit / error
        UNKNOWN_RESOURCE_EFFECT
    };

    // Ownership state for a variable
    using State = OwnershipState;
    struct VariableInfo {
        State state = State::UNOWNED;
        std::string defined_at;  // Location info for error messages
        bool is_owned_binding = false;   // Was bound with (owned ...)
        uint64_t binding_id = 0;
        ResourceKind resource_kind = ResourceKind::NONE;
        ResourceState resource_state = ResourceState::NONE;
        CallableEffect callable_effect = CallableEffect::NONE;
        std::string resource_alias;
        ResourceState moved_from_state = ResourceState::NONE;
        bool is_defined = false;
        const sr::Contract* contract = nullptr;
        bool is_proven_callable = false;
        bool is_verified_stdlib_primitive = false;
        const eshkol_ast_t* function_node = nullptr;  // DEFINE (function form) or LAMBDA node
        uint64_t function_scope_id = 0;                // scope the function closes over
        uint64_t frame_id = 0;                         // CONTINUATION / LOOP frame
        bool lent = false;                             // lent by a borrow scope that closes it
        bool is_exit = false;                          // TERMINAL_CONTROL that ends the process
        std::string unknown_reason;
        // Alias facts, recorded while the unit uses owned, move or borrow.
        std::set<uint64_t> reaches;                   // bindings its value may refer to
        const eshkol_ast_t* proc_node = nullptr;      // function definition or lambda it holds
        std::shared_ptr<const ProcSummary> summary;   // procedure accepted by an earlier input
        bool loop_proc = false;                       // a named-let loop procedure
        bool data = false;                            // holds a value that is never a procedure
        bool external = false;                        // declared here, defined by a linked object

        VariableInfo() = default;
        VariableInfo(State s, std::string at, bool owned, uint64_t id)
            : state(s), defined_at(std::move(at)), is_owned_binding(owned), binding_id(id) {}
    };

    // Ownership scope (lexical block)
    struct Scope {
        std::map<std::string, VariableInfo> variables;
        std::string name;  // For debugging
        uint64_t id = 0;
    };

    OwnershipAnalyzer(const ResourceContracts* contracts, bool follow_branches)
        : contracts_(contracts), follow_branches_(follow_branches), has_errors_(false) {}

    bool needsBranchAnalysis() const { return needs_branch_analysis_; }

    // Run analysis on all ASTs
    bool analyze(const std::vector<eshkol_ast_t>& asts, OwnershipBindings* globals) {
        has_errors_ = false;
        strict_resources_ = contracts_ != nullptr;
        errors_.clear();
        reported_.clear();
        scope_stack_.clear();
        borrowed_counts_.clear();
        branch_frames_.clear();
        accepted_factory_nodes_.clear();
        accepted_move_nodes_.clear();
        allowed_resource_use_nodes_.clear();
        parallel_binding_info_.clear();
        call_position_nodes_.clear();
        proc_arg_nodes_.clear();
        alias_value_nodes_.clear();
        named_lambda_nodes_.clear();
        inline_lambdas_.clear();
        inline_pre_states_.clear();
        pending_procs_.clear();
        let_init_nodes_.clear();
        closed_frames_.clear();
        pending_bind_.clear();
        last_call_result_.clear();
        frames_.clear();
        saved_snapshots_.clear();
        summary_memo_.clear();
        in_progress_.clear();
        memoizable_stack_.clear();
        classification_memo_.clear();
        classification_in_progress_.clear();
        call_chain_.clear();
        fn_ = nullptr;
        dead_ = false;
        handler_depth_ = 0;
        callable_epoch_ = 0;
        capture_scan_truncated_ = false;
        next_binding_id_ = 1;
        next_scope_id_ = 1;
        next_frame_id_ = 1;

        if (strict_resources_) {
            for (const auto& module : contracts_->untrusted_modules)
                reportResourceError(module +
                    " source does not match the pinned stock SDK contract; resource effects are unsupported");
        }

        // Push global scope
        pushScope("global");
        if (globals) loadSession(*globals);
        prepareAliasTracking(asts, globals);
        prepareStrictBindings(asts);

        // Required module ASTs are also executed by the compiler/JIT module
        // loader. Check their top-level initializers under the same effect
        // policy as user forms; DEFINE_OP already suppresses traversal into
        // function bodies, which are declarations rather than initializers.
        for (size_t i = 0; i < asts.size(); ++i) {
            analyzeAST(&asts[i], true);
            if (needs_branch_analysis_) return false;
        }

        if (strict_resources_) {
            checkResourceScopeExit();
        }

        // Check global scope exit
        checkScopeExit();
        if (globals && !has_errors_) saveSession(*globals);
        popScope();

        return !has_errors_;
    }

    // Get error messages
    const std::vector<std::string>& getErrors() const { return errors_; }
private:
    const ResourceContracts* contracts_;
    bool follow_branches_;
    bool needs_branch_analysis_ = false;
    const sr::Contract* trustedResourceContract(const eshkol_ast_t* ast) {
        if (!contracts_ || !ast || ast->type != ESHKOL_OP || ast->operation.op != ESHKOL_DEFINE_OP ||
            !ast->operation.define_op.name) return nullptr;
        auto it = contracts_->definitions.find(ast->operation.define_op.name);
        return it == contracts_->definitions.end() ? nullptr : it->second;
    }

    bool hasVerifiedStdlibScalarPrimitiveName(const char* name) {
        return contracts_ && name && contracts_->scalar_primitives.count(name) != 0;
    }

    bool isVerifiedStdlibScalarPrimitive(const eshkol_ast_t* ast) {
        if (!ast || ast->type != ESHKOL_OP) return false;
        const char* name = nullptr;
        if (ast->operation.op == ESHKOL_DEFINE_OP)
            name = ast->operation.define_op.name;
        else if (ast->operation.op == ESHKOL_EXTERN_OP)
            name = ast->operation.extern_op.name;
        return hasVerifiedStdlibScalarPrimitiveName(name);
    }

    const sr::Contract* verifiedStdlibContract(const eshkol_ast_t* ast) {
        if (!contracts_ || !ast || ast->type != ESHKOL_OP) return nullptr;
        const char* name = nullptr;
        if (ast->operation.op == ESHKOL_DEFINE_OP)
            name = ast->operation.define_op.name;
        else if (ast->operation.op == ESHKOL_EXTERN_OP)
            name = ast->operation.extern_op.name;
        auto it = name ? contracts_->stdlib_contracts.find(name) : contracts_->stdlib_contracts.end();
        return it == contracts_->stdlib_contracts.end() ? nullptr : it->second;
    }

    std::vector<Scope> scope_stack_;
    std::vector<std::string> errors_;
    std::set<std::string> reported_;
    bool has_errors_;
    bool strict_resources_ = false;

    struct WorkItem {
        enum class Kind { AST, LET_BINDING, LET_SEQUENTIAL_INSTALL,
                          LET_PARALLEL_INIT, LET_PARALLEL_INSTALL,
                          EXIT_SCOPE, UNBORROW, IF_AFTER_CONDITION,
                          IF_AFTER_TRUE, IF_AFTER_FALSE, COND_CLAUSE,
                          LOOP_END, LOOP_BACKEDGE, DO_INSTALL, DO_AFTER_TEST,
                          DO_AFTER_ITERATION, DO_EXIT, GUARD_AFTER_BODY,
                          GUARD_JOIN, WIND_ENTER, WIND_AFTER_BODY,
                          WIND_AFTER_RAISES, WIND_FORWARD, CALLCC_JOIN,
                          INLINE_END, CALL_AFTER, USER_CALL_APPLY,
                          TERMINAL_AFTER, RAISE_AFTER, POP_FRAME };
        Kind kind = Kind::AST;
        const eshkol_ast_t* ast = nullptr;
        const eshkol_operations_t* op = nullptr;
        uint64_t index = 0;
        std::string name;
        uint64_t binding_id = 0;
        uint64_t branch_id = 0;
        bool resource_effects_enabled = true;
        const eshkol_ast_t* clauses = nullptr;
        uint64_t count = 0;
        bool flag = false;

        static WorkItem astNode(const eshkol_ast_t* node,
                                bool resource_effects_enabled = true) {
            WorkItem item;
            item.ast = node;
            item.resource_effects_enabled = resource_effects_enabled;
            return item;
        }
        static WorkItem continuation(Kind kind, const eshkol_operations_t* op,
                                     uint64_t index = 0,
                                     bool resource_effects_enabled = true) {
            WorkItem item;
            item.kind = kind;
            item.op = op;
            item.index = index;
            item.resource_effects_enabled = resource_effects_enabled;
            return item;
        }
        static WorkItem unborrow(uint64_t binding_id) {
            WorkItem item;
            item.kind = Kind::UNBORROW;
            item.binding_id = binding_id;
            return item;
        }
        static WorkItem branchContinuation(Kind kind, const eshkol_ast_t* node,
                                           uint64_t branch_id,
                                           bool resource_effects_enabled = true) {
            WorkItem item;
            item.kind = kind;
            item.ast = node;
            item.branch_id = branch_id;
            item.resource_effects_enabled = resource_effects_enabled;
            return item;
        }
        static WorkItem marker(Kind kind, const eshkol_ast_t* node, uint64_t id,
                               bool resource_effects_enabled = true) {
            WorkItem item;
            item.kind = kind;
            item.ast = node;
            item.branch_id = id;
            item.resource_effects_enabled = resource_effects_enabled;
            return item;
        }
        static WorkItem condClause(const eshkol_ast_t* clauses, uint64_t count,
                                   uint64_t index, bool is_guard,
                                   bool resource_effects_enabled = true) {
            WorkItem item;
            item.kind = Kind::COND_CLAUSE;
            item.clauses = clauses;
            item.count = count;
            item.index = index;
            item.flag = is_guard;
            item.resource_effects_enabled = resource_effects_enabled;
            return item;
        }
    };

    // Analysis state at one program point: every lexical binding's typestate,
    // the active loans, and whether the point is reachable.
    struct Snapshot {
        std::vector<Scope> scopes;
        std::map<uint64_t, size_t> borrows;
        bool dead = false;
    };

    struct BranchFrame {
        const eshkol_ast_t* condition = nullptr;
        bool resource_effects_enabled = true;
        std::vector<WorkItem> true_items;
        std::vector<WorkItem> false_items;
        Snapshot base;
        Snapshot true_state;
    };

    // Control frames that collect the states of non-local exits.
    enum class FrameKind { GUARD, WIND, CALLCC, LOOP, DO };
    struct Frame {
        FrameKind kind = FrameKind::GUARD;
        uint64_t id = 0;
        size_t depth = 0;            // scopes [0, depth) survive the exit
        bool any = false;            // GUARD/WIND: raise points; CALLCC: escapes
        Snapshot exits;
        bool raised = false;                  // WIND: an exception passes through
        std::set<uint64_t> escape_targets;    // WIND: continuations that escape through it
        // LOOP / DO
        std::string loop_name;
        uint64_t loop_binding_id = 0;
        std::map<uint64_t, ResourceState> entry_states;
        std::map<uint64_t, size_t> entry_borrows;
        std::set<const eshkol_ast_t*> tail_calls;
    };

    // Return facts for the function body being analyzed at a call site.
    struct FunctionContext {
        std::string name;
        std::set<const eshkol_ast_t*> return_nodes;
        std::set<uint64_t> param_ids;
        size_t function_depth = 0;
        bool returned_handle = false;
        bool returned_value = false;
        bool return_conflict = false;
        bool may_exit = false;          // some path ends the process
        bool may_raise_unhandled = false;  // some path raises with no handler in the chain
        ResourceKind return_kind = ResourceKind::NONE;
        ResourceState return_state = ResourceState::NONE;
    };

    struct ParamPost {
        size_t index = 0;
        ResourceState resource_state = ResourceState::NONE;
        ResourceState moved_from_state = ResourceState::NONE;
        State state = State::UNOWNED;
    };

    // Effect of one call on the caller's handles, computed by analyzing the
    // callee body against the caller's handle states (context-sensitive).
    struct Summary {
        bool dead = false;
        std::vector<std::pair<uint64_t, VariableInfo>> env_post;
        std::vector<ParamPost> param_post;
        bool returns_handle = false;
        ResourceKind return_kind = ResourceKind::NONE;
        ResourceState return_state = ResourceState::NONE;
        bool may_exit = false;
        bool may_raise_unhandled = false;
    };

    struct ArgHandle {
        size_t index = 0;
        uint64_t caller_id = 0;  // 0: a handle lent by a borrow scope
        ResourceKind kind = ResourceKind::NONE;
        ResourceState state = ResourceState::NONE;
        bool borrowed = false;
        bool lent = false;
        std::string name;
    };

    struct PendingCall {
        VariableInfo function;
        std::string name;
        std::vector<ArgHandle> handles;
        std::set<size_t> pure_callables;
        const sr::Contract* lender = nullptr;  // borrow scope applying a named procedure
        sr::ProcMode mode = sr::ProcMode::ONCE;
    };

    struct InlineLambda {
        sr::ProcMode mode = sr::ProcMode::ONCE;
        ResourceKind lend_kind = ResourceKind::NONE;
        std::string lender;
    };

    struct InProgress {
        size_t depth = 0;
        Summary assumption;
        bool used = false;
    };

    enum class FnClass { PURE, RELEVANT, UNKNOWN };
    struct FnClassResult {
        FnClass cls = FnClass::PURE;
        std::string reason;
    };

    struct FunctionShape {
        const eshkol_ast_t* params = nullptr;
        uint64_t num_params = 0;
        const eshkol_ast_t* body = nullptr;
        const char* rest = nullptr;
    };

    struct SavedContext {
        std::vector<Scope> scopes;
        std::map<uint64_t, size_t> borrows;
        bool dead = false;
        std::vector<Frame> frames;
        int handler_depth = 0;
        FunctionContext* fn = nullptr;
        bool effects = true;
        std::vector<size_t> markers;
    };

    // ── diagnostics ─────────────────────────────────────────────────────
    void reportResourceError(const std::string& message) {
        // Unreachable code cannot misuse a handle: a path that already left
        // through exit, raise, a loop back edge, or a continuation is dead.
        if (dead_) return;
        std::string context;
        if (!call_chain_.empty()) {
            context = "in '" + call_chain_.back() + "'";
            if (call_chain_.size() > 1) context += " (called from '" + call_chain_[call_chain_.size() - 2] + "')";
            context += ": ";
        }
        reportError("Strict resource error: " + context + message);
    }

    void truncateErrors(size_t mark) {
        while (errors_.size() > mark) {
            reported_.erase(errors_.back());
            errors_.pop_back();
        }
        has_errors_ = !errors_.empty();
    }

    static std::string kindText(ResourceKind kind) { return sr::kindName(kind); }

    static std::string acceptedText(uint32_t mask) {
        std::string out;
        for (unsigned k = 1; k < static_cast<unsigned>(ResourceKind::COUNT); ++k) {
            if (!(mask & (1u << k))) continue;
            if (!out.empty()) out += " or ";
            out += sr::kindName(static_cast<ResourceKind>(k));
        }
        return out.empty() ? "no handle" : out;
    }

    static std::string stateText(ResourceState state) {
        switch (state) {
            case ResourceState::ERROR_OR_LIVE: return "not yet checked for failure";
            case ResourceState::ERROR_ONLY: return "a failed acquisition";
            case ResourceState::LIVE: return "open";
            case ResourceState::CLOSED: return "closed";
            case ResourceState::CLOSED_OR_ERROR: return "closed";
            case ResourceState::MAYBE_LIVE: return "open on some paths only";
            case ResourceState::MOVED: return "moved";
            default: return "untracked";
        }
    }

    static ResourceState initialFactoryState(ResourceKind kind) {
        return sr::failureOf(kind) == sr::Failure::RAISES ? ResourceState::LIVE
                                                          : ResourceState::ERROR_OR_LIVE;
    }

    // ── callee resolution ───────────────────────────────────────────────
    struct Callee {
        CallableEffect effect = CallableEffect::NONE;
        const sr::Contract* contract = nullptr;
        VariableInfo* binding = nullptr;
        std::string name;
    };

    Callee resolveCallee(const eshkol_ast_t* func) {
        Callee callee;
        callee.name = getVarName(func);
        if (callee.name.empty()) return callee;
        callee.binding = lookupVariable(callee.name);
        if (callee.binding) {
            if (callee.binding->is_verified_stdlib_primitive) return callee;
            callee.effect = callee.binding->callable_effect;
            callee.contract = callee.binding->contract;
            return callee;
        }
        unresolvedCallable(callee.name, &callee.effect, &callee.contract);
        return callee;
    }

    void unresolvedCallable(const std::string& name, CallableEffect* effect,
                            const sr::Contract** contract) const {
        *contract = nullptr;
        if (name == "exit" || name == "error") {
            *effect = CallableEffect::TERMINAL_CONTROL;
            return;
        }
        if (const sr::Contract* builtin = sr::findBuiltinContract(name.c_str())) {
            *effect = CallableEffect::CONTRACT;
            *contract = builtin;
            return;
        }
        *effect = isKnownResourceSafeBuiltin(name) ? CallableEffect::NONE
                                                   : CallableEffect::UNKNOWN_RESOURCE_EFFECT;
    }

    CallableEffect callableEffectOf(const eshkol_ast_t* expression) {
        return resolveCallee(expression).effect;
    }

    const sr::Contract* contractOf(const eshkol_ast_t* expression) {
        Callee callee = resolveCallee(expression);
        return callee.effect == CallableEffect::CONTRACT ? callee.contract : nullptr;
    }

    // Kind a call returns when it is a handle factory contract.
    ResourceKind factoryKindOfCall(const eshkol_ast_t* call_node) {
        if (!call_node || call_node->type != ESHKOL_OP ||
            call_node->operation.op != ESHKOL_CALL_OP) return ResourceKind::NONE;
        const sr::Contract* contract = contractOf(call_node->operation.call_op.func);
        return contract && contract->role == sr::Role::FACTORY ? contract->produces
                                                                : ResourceKind::NONE;
    }

    bool isUserFunctionCall(const eshkol_ast_t* call_node) {
        if (!call_node || call_node->type != ESHKOL_OP ||
            call_node->operation.op != ESHKOL_CALL_OP) return false;
        return callableEffectOf(call_node->operation.call_op.func) == CallableEffect::USER_FUNCTION;
    }

    static bool contractHasHandleEffect(const sr::Contract* contract) {
        if (!contract) return false;
        if (contract->role == sr::Role::SAFE) return false;
        if (contract->role == sr::Role::OBSERVE) {
            for (const auto& arg : contract->handle_args)
                if (arg.index >= 0 && !arg.optional) return true;
            return false;
        }
        return true;
    }

    // ── conditionals ────────────────────────────────────────────────────
    bool isIfForm(const eshkol_ast_t* node) const {
        if (!node || node->type != ESHKOL_OP) return false;
        const bool special_if = node->operation.op == ESHKOL_IF_OP;
        if (node->operation.op != ESHKOL_CALL_OP && !special_if) return false;
        const auto& call = node->operation.call_op;
        const bool has_if_callee = call.func && call.func->type == ESHKOL_VAR &&
            call.func->variable.id && std::strcmp(call.func->variable.id, "if") == 0;
        return (call.num_vars == 3 || call.num_vars == 2) && call.variables &&
               (special_if || has_if_callee);
    }

    static bool isTrueLiteral(const eshkol_ast_t* node) {
        return node && node->type == ESHKOL_BOOL && node->int64_val != 0;
    }

    bool isLiveResourceState(ResourceState state) const {
        return state == ResourceState::LIVE ||
               state == ResourceState::ERROR_OR_LIVE ||
               state == ResourceState::MAYBE_LIVE;
    }

    bool hasTrackedResourceBinding() const {
        for (const auto& scope : scope_stack_)
            for (const auto& entry : scope.variables)
                if (entry.second.resource_kind != ResourceKind::NONE &&
                    isLiveResourceState(entry.second.resource_state))
                    return true;
        return false;
    }

    void checkResourceScopeExit() {
        if (scope_stack_.empty() || dead_) return;
        const auto& scope = currentScope();
        for (const auto& entry : scope.variables) {
            const auto& name = entry.first;
            const auto& info = entry.second;
            if (info.lent) continue;  // the lending scope closes it
            if (info.resource_kind != ResourceKind::NONE &&
                isLiveResourceState(info.resource_state)) {
                reportResourceError("resource '" + name +
                    "' may still be live at scope exit; close it on every successful path (" +
                    kindText(info.resource_kind) + " is " + stateText(info.resource_state) + ")");
            }
        }
    }

    ResourceState joinResourceState(ResourceState left, ResourceState right) const {
        if (left == right) return left;
        if (left == ResourceState::MOVED && right == ResourceState::MOVED)
            return ResourceState::MOVED;
        if (left == ResourceState::MOVED || right == ResourceState::MOVED) {
            const ResourceState other = left == ResourceState::MOVED ? right : left;
            if (other == ResourceState::ERROR_ONLY ||
                other == ResourceState::CLOSED_OR_ERROR)
                return ResourceState::CLOSED_OR_ERROR;
            if (other == ResourceState::CLOSED) return ResourceState::CLOSED;
        }
        if (left == ResourceState::NONE || right == ResourceState::NONE)
            return ResourceState::MAYBE_LIVE;
        const bool left_live = isLiveResourceState(left);
        const bool right_live = isLiveResourceState(right);
        if (left_live || right_live) {
            if ((left == ResourceState::ERROR_OR_LIVE && right == ResourceState::ERROR_ONLY) ||
                (right == ResourceState::ERROR_OR_LIVE && left == ResourceState::ERROR_ONLY))
                return ResourceState::ERROR_OR_LIVE;
            return ResourceState::MAYBE_LIVE;
        }
        if ((left == ResourceState::ERROR_ONLY && right == ResourceState::CLOSED) ||
            (right == ResourceState::ERROR_ONLY && left == ResourceState::CLOSED) ||
            left == ResourceState::CLOSED_OR_ERROR || right == ResourceState::CLOSED_OR_ERROR)
            return ResourceState::CLOSED_OR_ERROR;
        return ResourceState::MAYBE_LIVE;
    }

    State joinOwnershipState(State left, State right) const {
        if (left == right) return left;
        if (left == State::MOVED || right == State::MOVED ||
            left == State::MAYBE_MOVED || right == State::MAYBE_MOVED)
            return State::MAYBE_MOVED;
        return State::UNOWNED;
    }

    // ── snapshots and joins ─────────────────────────────────────────────
    Snapshot snapshot() const { return Snapshot{scope_stack_, borrowed_counts_, dead_}; }

    void restore(const Snapshot& state) {
        scope_stack_ = state.scopes;
        borrowed_counts_ = state.borrows;
        dead_ = state.dead;
    }

    void joinVariable(VariableInfo& joined, const VariableInfo& a, const VariableInfo& b) {
        joined.state = joinOwnershipState(a.state, b.state);
        joined.resource_state = joinResourceState(a.resource_state, b.resource_state);
        if (a.resource_kind == b.resource_kind) {
            joined.resource_kind = a.resource_kind;
        } else if (a.resource_kind != ResourceKind::NONE || b.resource_kind != ResourceKind::NONE) {
            joined.resource_kind = a.resource_kind != ResourceKind::NONE
                ? a.resource_kind : b.resource_kind;
            joined.resource_state = ResourceState::MAYBE_LIVE;
        }
        if (a.resource_state == ResourceState::MOVED && b.resource_state == ResourceState::MOVED)
            joined.moved_from_state = joinResourceState(a.moved_from_state, b.moved_from_state);
        const bool same_callable = a.callable_effect == b.callable_effect &&
            a.contract == b.contract && a.function_node == b.function_node;
        if (!same_callable) {
            joined.callable_effect = (a.callable_effect == CallableEffect::NONE &&
                                      b.callable_effect == CallableEffect::NONE)
                ? CallableEffect::NONE : CallableEffect::UNKNOWN_RESOURCE_EFFECT;
            joined.contract = nullptr;
            joined.function_node = nullptr;
            joined.unknown_reason = "it is bound to different procedures on different paths";
        }
        joined.is_proven_callable = a.is_proven_callable && b.is_proven_callable;
        joined.lent = a.lent && b.lent;
        joined.reaches = a.reaches;
        joined.reaches.insert(b.reaches.begin(), b.reaches.end());
        joined.proc_node = a.proc_node == b.proc_node ? a.proc_node : nullptr;
        joined.summary = a.summary == b.summary ? a.summary : nullptr;
        joined.loop_proc = a.loop_proc && b.loop_proc;
        joined.data = a.data && b.data;
    }

    Snapshot joinSnapshots(const Snapshot& a, const Snapshot& b) {
        if (a.dead && b.dead) return a;
        if (a.dead) return b;
        if (b.dead) return a;
        if (a.scopes.size() != b.scopes.size()) {
            reportResourceError("branch changes lexical scope shape in the checked fragment");
            return a;
        }
        Snapshot joined = b;
        for (size_t i = 0; i < b.scopes.size(); ++i) {
            const auto& left = a.scopes[i].variables;
            auto& out = joined.scopes[i].variables;
            for (auto& pair : out) {
                auto it = left.find(pair.first);
                if (it == left.end()) {
                    if (pair.second.resource_kind != ResourceKind::NONE &&
                        isLiveResourceState(pair.second.resource_state))
                        reportResourceError("resource binding '" + pair.first +
                            "' is created in only one branch; this flow is unsupported");
                    continue;
                }
                if (it->second.binding_id != pair.second.binding_id) {
                    reportResourceError("binding '" + pair.first +
                        "' is rebound differently across branches in the checked fragment");
                    continue;
                }
                VariableInfo merged = pair.second;
                joinVariable(merged, it->second, pair.second);
                pair.second = std::move(merged);
            }
            for (const auto& pair : left) {
                if (out.count(pair.first)) continue;
                if (pair.second.resource_kind != ResourceKind::NONE &&
                    isLiveResourceState(pair.second.resource_state))
                    reportResourceError("resource binding '" + pair.first +
                        "' is created in only one branch; this flow is unsupported");
            }
        }
        if (a.borrows != b.borrows)
            reportResourceError("borrow state does not balance across both branches");
        return joined;
    }

    // Drop scopes a non-local exit leaves. A handle still open in a dropped
    // scope can never be closed: report it.
    void truncateForExit(Snapshot& state, size_t depth, const std::string& reason) {
        if (state.scopes.size() <= depth) return;
        for (size_t i = depth; i < state.scopes.size(); ++i) {
            for (const auto& entry : state.scopes[i].variables) {
                const auto& info = entry.second;
                if (!state.dead && !info.lent && info.resource_kind != ResourceKind::NONE &&
                    isLiveResourceState(info.resource_state)) {
                    reportResourceError("handle '" + entry.first + "' (" + kindText(info.resource_kind) +
                        ") is still open when " + reason +
                        " leaves its scope; close it first, or acquire it with a call-with-* "
                        "borrow scope or a dynamic-wind cleanup");
                }
                state.borrows.erase(info.binding_id);
            }
        }
        state.scopes.resize(depth);
    }

    void addExit(Frame& frame, Snapshot state, const std::string& reason) {
        if (state.dead) return;
        truncateForExit(state, frame.depth, reason);
        if (!frame.any) {
            frame.exits = std::move(state);
            frame.any = true;
        } else {
            frame.exits = joinSnapshots(frame.exits, state);
        }
    }

    Frame* findFrame(uint64_t id) {
        for (auto it = frames_.rbegin(); it != frames_.rend(); ++it)
            if (it->id == id) return &*it;
        return nullptr;
    }

    bool hasHandlerFrame() const {
        for (const auto& frame : frames_)
            if (frame.kind == FrameKind::GUARD || frame.kind == FrameKind::WIND) return true;
        return false;
    }

    // An operation that may raise. Inside a guard or dynamic-wind extent the
    // handler observes the state at this point; inside a callee whose caller
    // holds a handler, a handle the callee opened would be lost.
    void recordRaisePoint(const std::string& reason) {
        if (dead_) return;
        for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
            if (it->kind == FrameKind::GUARD || it->kind == FrameKind::WIND) {
                if (it->kind == FrameKind::WIND) it->raised = true;
                addExit(*it, snapshot(), reason);
                return;
            }
        }
        if (handler_depth_ > 0 && fn_) {
            Snapshot state = snapshot();
            truncateForExit(state, fn_->function_depth + 1, reason);
            checkFunctionLocalsOnExit(reason);
        }
    }

    void checkFunctionLocalsOnExit(const std::string& reason) {
        if (!fn_ || dead_ || scope_stack_.size() <= fn_->function_depth) return;
        const auto& scope = scope_stack_[fn_->function_depth];
        for (const auto& entry : scope.variables) {
            const auto& info = entry.second;
            if (fn_->param_ids.count(info.binding_id) || info.lent) continue;
            if (info.resource_kind != ResourceKind::NONE && isLiveResourceState(info.resource_state))
                reportResourceError("handle '" + entry.first + "' (" + kindText(info.resource_kind) +
                    ") is still open when " + reason + " leaves function '" + fn_->name +
                    "'; close it first or use a call-with-* or dynamic-wind cleanup");
        }
    }

    // An explicit raise/error. With no handler in scope it ends the program,
    // so every handle must already be closed.
    void explicitRaise(const std::string& what) {
        if (dead_) return;
        if (hasHandlerFrame() || handler_depth_ > 0) {
            recordRaisePoint("the exception raised by " + what);
        } else {
            if (fn_) fn_->may_raise_unhandled = true;
            for (const auto& scope : scope_stack_)
                for (const auto& entry : scope.variables)
                    if (entry.second.resource_kind != ResourceKind::NONE &&
                        isLiveResourceState(entry.second.resource_state))
                        reportResourceError("terminal control transfer '" + what +
                            "' occurs while resource '" + entry.first + "' may still be live");
        }
        dead_ = true;
    }

    // ── refinement of fallible acquisitions ─────────────────────────────
    bool conditionRefinement(const eshkol_ast_t* condition,
                             uint64_t* binding_id, bool* true_is_error) {
        if (!condition) return false;
        if (condition->type == ESHKOL_VAR) {
            VariableInfo* info = lookupVariable(getVarName(condition));
            if (!info || info->resource_kind == ResourceKind::NONE ||
                sr::failureOf(info->resource_kind) != sr::Failure::FALSE_VALUE)
                return false;
            *binding_id = info->binding_id;
            *true_is_error = false;  // a #f result is the failure
            return true;
        }
        if (condition->type != ESHKOL_OP) return false;
        const eshkol_operations_t* op = &condition->operation;
        if (op->op == ESHKOL_CALL_OP && op->call_op.num_vars == 1 &&
            op->call_op.func && op->call_op.func->type == ESHKOL_VAR &&
            op->call_op.func->variable.id &&
            std::strcmp(op->call_op.func->variable.id, "not") == 0) {
            if (lookupVariable("not")) return false;  // a lexical replacement is not the primitive
            bool nested_true_is_error = false;
            if (conditionRefinement(&op->call_op.variables[0], binding_id,
                                    &nested_true_is_error)) {
                *true_is_error = !nested_true_is_error;
                return true;
            }
            return false;
        }
        if (op->op != ESHKOL_CALL_OP || op->call_op.num_vars != 1) return false;
        const sr::Contract* predicate = contractOf(op->call_op.func);
        if (!predicate || predicate->role != sr::Role::ERROR_PREDICATE) return false;
        VariableInfo* info = lookupVariable(getVarName(&op->call_op.variables[0]));
        if (!info || info->resource_kind == ResourceKind::NONE) return false;
        if (sr::failureOf(info->resource_kind) != predicate->family) return false;
        *binding_id = info->binding_id;
        *true_is_error = true;
        return true;
    }

    void refineBranchCondition(const eshkol_ast_t* condition, bool truth) {
        uint64_t binding_id = 0;
        bool true_is_error = false;
        if (!conditionRefinement(condition, &binding_id, &true_is_error)) return;
        std::string name;
        VariableInfo* info = findBindingById(binding_id, &name);
        if (!info) return;
        const bool error_path = truth == true_is_error;
        if (info->resource_state == ResourceState::ERROR_OR_LIVE)
            info->resource_state = error_path ? ResourceState::ERROR_ONLY : ResourceState::LIVE;
        else if (info->resource_state == ResourceState::MAYBE_LIVE)
            reportResourceError(std::string(sr::failureRefiner(sr::failureOf(info->resource_kind))) +
                " cannot refine resource '" + name +
                "' after incompatible branch states");
    }

    // ── static data and builtin sets ────────────────────────────────────
    bool isStaticDataLiteral(const eshkol_ast_t* value) const {
        if (!value) return false;
        switch (value->type) {
            case ESHKOL_UINT8:
            case ESHKOL_UINT16:
            case ESHKOL_UINT32:
            case ESHKOL_UINT64:
            case ESHKOL_INT8:
            case ESHKOL_INT16:
            case ESHKOL_INT32:
            case ESHKOL_INT64:
            case ESHKOL_DOUBLE:
            case ESHKOL_STRING:
            case ESHKOL_NULL:
            case ESHKOL_CHAR:
            case ESHKOL_BOOL:
            case ESHKOL_BIGNUM_LITERAL:
            case ESHKOL_SYMBOL:
                return true;
            default:
                return false;
        }
    }

    bool isPureResourceDataBuiltin(const std::string& name) const {
        static const std::set<std::string> names = {
            "+", "-", "*", "/", "=", "<", "<=", ">", ">=", "boolean=?",
            "eq?", "eqv?", "equal?", "not", "integer?", "exact?", "inexact?",
            "number?", "real?", "string?", "symbol?", "vector?", "pair?", "null?",
            "procedure?", "vector-length", "vector-ref", "car", "cdr", "caar", "cadr",
            "caddr", "cadddr", "cddr", "cdddr", "cddddr", "cons", "list", "append",
            "string-length", "string-ref", "substring", "void"
        };
        return names.count(name) != 0;
    }

    struct PreparedBinding {
        CallableEffect effect = CallableEffect::NONE;
        const sr::Contract* contract = nullptr;
        const eshkol_ast_t* function_node = nullptr;
        bool proven = false;
        bool is_exit = false;
        ResourceKind future_kind = ResourceKind::NONE;
        std::string unknown_reason;
    };

    bool isStaticDataExpression(
        const eshkol_ast_t* value,
        const std::map<std::string, PreparedBinding>* known = nullptr,
        unsigned depth = 0) {
        if (!value || depth > 128) return false;
        if (isStaticDataLiteral(value)) return true;
        if (value->type == ESHKOL_VAR) {
            const std::string name = getVarName(value);
            if (known) {
                auto entry = known->find(name);
                return entry != known->end() && entry->second.effect == CallableEffect::NONE &&
                       !entry->second.proven;
            }
            VariableInfo* info = lookupVariable(name);
            return info && info->resource_kind == ResourceKind::NONE &&
                   info->callable_effect == CallableEffect::NONE &&
                   !info->is_proven_callable;
        }
        if (value->type == ESHKOL_CONS) {
            const std::string head = getVarName(value->cons_cell.car);
            if (!head.empty()) {
                if (!isPureResourceDataBuiltin(head)) return false;
                return isStaticDataExpression(value->cons_cell.cdr, known, depth + 1);
            }
            return isStaticDataExpression(value->cons_cell.car, known, depth + 1) &&
                   isStaticDataExpression(value->cons_cell.cdr, known, depth + 1);
        }
        if (value->type != ESHKOL_OP) return false;
        const auto& op = value->operation;
        if (op.op == ESHKOL_QUOTE_OP) return true;
        if (op.op == ESHKOL_CALL_OP) {
            const std::string head = getVarName(op.call_op.func);
            if (!isPureResourceDataBuiltin(head)) return false;
            if (known ? known->count(head) != 0 : lookupVariable(head) != nullptr) return false;
            for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                if (!isStaticDataExpression(&op.call_op.variables[i], known, depth + 1)) return false;
            return true;
        }
        if (op.op == ESHKOL_SEQUENCE_OP) {
            for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                if (!isStaticDataExpression(&op.sequence_op.expressions[i], known, depth + 1)) return false;
            return op.sequence_op.num_expressions > 0;
        }
        return false;
    }

    // Closed set of compiler primitives whose resource effect is known to be
    // scalar/data output. Every other unresolved compiler builtin is rejected
    // under --strict-resources because it may create a hidden native handle.
    // Builtins with a handle contract are listed in strict_resource_contracts.h.
    bool isKnownResourceSafeBuiltin(const std::string& name) const {
        static const std::set<std::string> names = {
            "+", "-", "*", "/", "=", "<", "<=", ">", ">=", "boolean=?",
            "eq?", "eqv?", "equal?", "not", "and", "or", "integer?", "exact?",
            "inexact?", "number?", "real?", "string?", "symbol?", "vector?",
            "pair?", "null?", "procedure?", "vector-length", "vector-ref", "car",
            "cdr", "caar", "cadr", "caddr", "cadddr", "cddr", "cdddr", "cddddr",
            "cons", "list", "append", "string-length", "string-ref", "substring",
            "display", "write", "newline", "exit", "error", "void",
            "inexact->exact", "make-rational"
        };
        return names.count(name) != 0;
    }

    bool isRawResourceBuiltin(const std::string& name) const {
        static const std::set<std::string> names = {
            "unix-socket-connect", "socket-send", "socket-recv", "socket-close",
            "websocket-connect", "websocket-send", "websocket-receive",
            "websocket-close", "fd-close"
        };
        return names.count(name) != 0;
    }

    // ── top-level pre-binding ───────────────────────────────────────────
    PreparedBinding prepareFromName(const std::string& source,
                                    const std::map<std::string, PreparedBinding>& known) const {
        PreparedBinding out;
        auto it = known.find(source);
        if (it != known.end()) {
            out = it->second;
            out.future_kind = ResourceKind::NONE;
            return out;
        }
        unresolvedCallable(source, &out.effect, &out.contract);
        out.is_exit = source == "exit";
        if (out.effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
            out.unknown_reason = "builtin '" + source + "' has no strict resource effect summary";
            out.proven = false;
        } else {
            out.proven = out.effect != CallableEffect::NONE || isKnownResourceSafeBuiltin(source);
        }
        return out;
    }

    void prepareStrictBindings(const std::vector<eshkol_ast_t>& asts) {
        if (!strict_resources_) return;
        std::map<std::string, PreparedBinding> prepared;

        for (const auto& ast : asts) {
            if (ast.type != ESHKOL_OP) continue;
            const auto& op = ast.operation;
            if (op.op == ESHKOL_DEFINE_OP && op.define_op.name) {
                const std::string name = op.define_op.name;
                PreparedBinding binding;
                const eshkol_ast_t* value = op.define_op.value;
                if (const sr::Contract* contract = trustedResourceContract(&ast)) {
                    binding.effect = CallableEffect::CONTRACT;
                    binding.contract = contract;
                    binding.proven = true;
                } else if (const sr::Contract* stdlib_contract = verifiedStdlibContract(&ast)) {
                    binding.effect = CallableEffect::CONTRACT;
                    binding.contract = stdlib_contract;
                    binding.proven = true;
                } else if (isVerifiedStdlibScalarPrimitive(&ast)) {
                    binding.proven = true;
                } else if (op.define_op.is_external) {
                    if (op.define_op.is_function) {
                        binding.effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                        binding.unknown_reason = "'" + name + "' is an external definition";
                        binding.proven = true;
                    } else if (!isStaticDataExpression(value, &prepared)) {
                        binding.effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                        binding.unknown_reason = "'" + name + "' has an opaque external initializer";
                    }
                } else if (op.define_op.is_function) {
                    binding.effect = CallableEffect::USER_FUNCTION;
                    binding.function_node = &ast;
                    binding.proven = true;
                } else if (value && value->type == ESHKOL_OP &&
                           value->operation.op == ESHKOL_LAMBDA_OP) {
                    binding.effect = CallableEffect::USER_FUNCTION;
                    binding.function_node = value;
                    binding.proven = true;
                } else if (value && value->type == ESHKOL_VAR) {
                    binding = prepareFromName(getVarName(value), prepared);
                } else if (value && value->type == ESHKOL_OP &&
                           value->operation.op == ESHKOL_CALL_OP) {
                    const std::string callee = getVarName(value->operation.call_op.func);
                    PreparedBinding source = prepareFromName(callee, prepared);
                    if (source.effect == CallableEffect::CONTRACT && source.contract &&
                        source.contract->role == sr::Role::FACTORY) {
                        binding.future_kind = source.contract->produces;
                    } else if (source.effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
                        binding.effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                        binding.unknown_reason = "'" + name + "' is the result of '" + callee +
                            "', whose effect is not proved";
                    }
                } else if (!isStaticDataExpression(value, &prepared)) {
                    binding.effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                    binding.unknown_reason = "'" + name + "' has a computed value";
                }
                prepared[name] = binding;
            } else if (op.op == ESHKOL_EXTERN_OP && op.extern_op.name) {
                PreparedBinding binding;
                if (const sr::Contract* stdlib_contract = verifiedStdlibContract(&ast)) {
                    binding.effect = CallableEffect::CONTRACT;
                    binding.contract = stdlib_contract;
                    binding.proven = true;
                } else if (isVerifiedStdlibScalarPrimitive(&ast)) {
                    binding.proven = true;
                } else {
                    binding.effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                    binding.unknown_reason = "'" + std::string(op.extern_op.name) + "' is a foreign function";
                    binding.proven = true;
                }
                prepared[op.extern_op.name] = binding;
            }
        }

        // Give top-level names stable lexical identities before any body is
        // checked. This lets a closure resolve a later global declaration by
        // binding ID, while local shadowing still wins through normal scopes.
        for (const auto& ast : asts) {
            if (ast.type != ESHKOL_OP) continue;
            const auto& op = ast.operation;
            std::string name;
            if (op.op == ESHKOL_DEFINE_OP && op.define_op.name) {
                name = op.define_op.name;
            } else if (op.op == ESHKOL_EXTERN_OP && op.extern_op.name) {
                name = op.extern_op.name;
            }
            if (name.empty() || currentScope().variables.count(name)) continue;
            VariableInfo info{State::UNOWNED, name, false, next_binding_id_++};
            auto it = prepared.find(name);
            if (it != prepared.end()) {
                info.callable_effect = it->second.effect;
                info.contract = it->second.contract;
                info.function_node = it->second.function_node;
                info.function_scope_id = currentScope().id;
                info.is_proven_callable = it->second.proven;
                info.is_exit = it->second.is_exit;
                info.unknown_reason = it->second.unknown_reason;
                if (it->second.future_kind != ResourceKind::NONE) {
                    info.resource_kind = it->second.future_kind;
                    info.resource_state = initialFactoryState(it->second.future_kind);
                }
            }
            info.is_verified_stdlib_primitive =
                (op.op == ESHKOL_DEFINE_OP || op.op == ESHKOL_EXTERN_OP) &&
                isVerifiedStdlibScalarPrimitive(&ast);
            currentScope().variables.emplace(name, std::move(info));
        }
    }

    // ── expression scans ────────────────────────────────────────────────
    bool expressionContainsResource(const eshkol_ast_t* expression) {
        std::vector<const eshkol_ast_t*> todo;
        if (expression) todo.push_back(expression);
        while (!todo.empty()) {
            const eshkol_ast_t* node = todo.back();
            todo.pop_back();
            if (!node) continue;
            if (node->type == ESHKOL_VAR) {
                VariableInfo* info = lookupVariable(getVarName(node));
                if (info && info->resource_kind != ResourceKind::NONE) return true;
                continue;
            }
            if (node->type == ESHKOL_CONS) {
                todo.push_back(node->cons_cell.car);
                todo.push_back(node->cons_cell.cdr);
                continue;
            }
            if (node->type != ESHKOL_OP) continue;
            const auto& op = node->operation;
            switch (op.op) {
                case ESHKOL_CALL_OP:
                case ESHKOL_IF_OP:
                case ESHKOL_COND_OP:
                    for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                        todo.push_back(&op.call_op.variables[i]);
                    break;
                case ESHKOL_AND_OP:
                case ESHKOL_OR_OP:
                case ESHKOL_SEQUENCE_OP:
                    for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                        todo.push_back(&op.sequence_op.expressions[i]);
                    break;
                case ESHKOL_LAMBDA_OP:
                    if (op.lambda_op.body) todo.push_back(op.lambda_op.body);
                    break;
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    for (uint64_t i = 0; i < op.let_op.num_bindings; ++i)
                        todo.push_back(&op.let_op.bindings[i]);
                    todo.push_back(op.let_op.body);
                    break;
                case ESHKOL_SET_OP:
                    todo.push_back(op.set_op.value);
                    break;
                case ESHKOL_BORROW_OP:
                    todo.push_back(op.borrow_op.value);
                    for (uint64_t i = 0; i < op.borrow_op.num_body_exprs; ++i)
                        todo.push_back(&op.borrow_op.body[i]);
                    break;
                case ESHKOL_MOVE_OP:
                    todo.push_back(op.move_op.value);
                    break;
                case ESHKOL_OWNED_OP:
                    todo.push_back(op.owned_op.value);
                    break;
                default:
                    break;
            }
        }
        return false;
    }

    bool expressionProducesResourceValue(const eshkol_ast_t* expression) {
        if (!expression) return false;
        if (expression->type == ESHKOL_VAR) {
            VariableInfo* info = lookupVariable(getVarName(expression));
            return info && info->resource_kind != ResourceKind::NONE;
        }
        if (expression->type != ESHKOL_OP) return false;
        const auto& op = expression->operation;
        if ((op.op == ESHKOL_IF_OP || isIfForm(expression)) && op.call_op.num_vars >= 2)
            return expressionProducesResourceValue(&op.call_op.variables[1]) ||
                   (op.call_op.num_vars == 3 &&
                    expressionProducesResourceValue(&op.call_op.variables[2]));
        if (op.op == ESHKOL_CALL_OP) {
            if (factoryKindOfCall(expression) != ResourceKind::NONE) return true;
            auto result = last_call_result_.find(expression);
            return result != last_call_result_.end() && result->second.first != ResourceKind::NONE;
        }
        if (op.op == ESHKOL_MOVE_OP)
            return expressionProducesResourceValue(op.move_op.value);
        if (op.op == ESHKOL_OWNED_OP)
            return expressionProducesResourceValue(op.owned_op.value);
        if (op.op == ESHKOL_SEQUENCE_OP && op.sequence_op.num_expressions)
            return expressionProducesResourceValue(
                &op.sequence_op.expressions[op.sequence_op.num_expressions - 1]);
        return false;
    }

    bool expressionContainsCallableContract(const eshkol_ast_t* expression) {
        std::vector<const eshkol_ast_t*> todo;
        if (expression) todo.push_back(expression);
        while (!todo.empty()) {
            const eshkol_ast_t* node = todo.back();
            todo.pop_back();
            if (!node) continue;
            if (node->type == ESHKOL_VAR) {
                const std::string name = getVarName(node);
                VariableInfo* info = lookupVariable(name);
                if (info && info->is_proven_callable &&
                    info->callable_effect != CallableEffect::NONE)
                    return true;
                continue;
            }
            if (node->type == ESHKOL_CONS) {
                todo.push_back(node->cons_cell.car);
                todo.push_back(node->cons_cell.cdr);
                continue;
            }
            if (node->type == ESHKOL_OP && node->operation.op == ESHKOL_LAMBDA_OP)
                return true;
            if (node->type == ESHKOL_OP && node->operation.op == ESHKOL_CALL_OP) {
                const auto& call = node->operation.call_op;
                for (uint64_t i = 0; i < call.num_vars; ++i)
                    todo.push_back(&call.variables[i]);
            }
        }
        return false;
    }

    bool expressionHasResourceTransition(const eshkol_ast_t* expression) {
        std::vector<const eshkol_ast_t*> todo;
        if (expression) todo.push_back(expression);
        while (!todo.empty()) {
            const eshkol_ast_t* node = todo.back();
            todo.pop_back();
            if (!node) continue;
            if (node->type == ESHKOL_CONS) {
                todo.push_back(node->cons_cell.car);
                todo.push_back(node->cons_cell.cdr);
                continue;
            }
            if (node->type != ESHKOL_OP) continue;
            const auto& op = node->operation;
            if (op.op == ESHKOL_CALL_OP) {
                Callee callee = resolveCallee(op.call_op.func);
                if (callee.effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT ||
                    callee.effect == CallableEffect::USER_FUNCTION ||
                    callee.effect == CallableEffect::CONTINUATION ||
                    callee.effect == CallableEffect::LOOP)
                    return true;
                if (callee.effect == CallableEffect::CONTRACT && callee.contract &&
                    (callee.contract->role == sr::Role::FACTORY ||
                     callee.contract->role == sr::Role::CLOSE ||
                     callee.contract->role == sr::Role::LEND))
                    return true;
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    todo.push_back(&op.call_op.variables[i]);
            } else if (op.op == ESHKOL_IF_OP || op.op == ESHKOL_COND_OP) {
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    todo.push_back(&op.call_op.variables[i]);
            } else if (op.op == ESHKOL_AND_OP || op.op == ESHKOL_OR_OP ||
                       op.op == ESHKOL_SEQUENCE_OP) {
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    todo.push_back(&op.sequence_op.expressions[i]);
            } else if (op.op == ESHKOL_MOVE_OP) {
                if (expressionContainsResource(op.move_op.value)) return true;
            } else if (op.op == ESHKOL_SET_OP) {
                VariableInfo* target = lookupVariable(op.set_op.name ? op.set_op.name : "");
                if (target && (target->resource_kind != ResourceKind::NONE ||
                               target->callable_effect != CallableEffect::NONE)) return true;
            } else if (op.op == ESHKOL_RAISE_OP || op.op == ESHKOL_CALL_CC_OP ||
                       op.op == ESHKOL_DYNAMIC_WIND_OP || op.op == ESHKOL_GUARD_OP) {
                return true;
            }
        }
        return false;
    }

    // ── binding initializers ────────────────────────────────────────────
    void bindResourceInitializer(VariableInfo& target, const eshkol_ast_t* value,
                                 const std::string& target_name, uint64_t env_scope_id) {
        target.resource_kind = ResourceKind::NONE;
        target.resource_state = ResourceState::NONE;
        target.moved_from_state = ResourceState::NONE;
        target.callable_effect = CallableEffect::NONE;
        target.contract = nullptr;
        target.function_node = nullptr;
        target.function_scope_id = 0;
        target.is_proven_callable = false;
        target.is_exit = false;
        target.unknown_reason.clear();
        if (value && value->type == ESHKOL_VAR) {
            const std::string source_name = getVarName(value);
            VariableInfo* source = lookupVariable(source_name);
            if (source && source->resource_kind == ResourceKind::NONE) {
                target.callable_effect = source->callable_effect;
                target.contract = source->contract;
                target.function_node = source->function_node;
                target.function_scope_id = source->function_scope_id;
                target.frame_id = source->frame_id;
                target.is_proven_callable = source->is_proven_callable;
                target.is_exit = source->is_exit;
                target.unknown_reason = source->unknown_reason;
                if (source->callable_effect != CallableEffect::NONE || source->is_proven_callable)
                    alias_value_nodes_.insert(value);
                if (source->callable_effect == CallableEffect::CONTINUATION ||
                    source->callable_effect == CallableEffect::LOOP)
                    alias_value_nodes_.erase(value);  // aliasing an escape is an escape
                return;
            }
            if (!source) {
                unresolvedCallable(source_name, &target.callable_effect, &target.contract);
                target.is_exit = source_name == "exit";
                if (target.callable_effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
                    target.unknown_reason = "builtin '" + source_name +
                        "' has no strict resource effect summary";
                } else {
                    target.is_proven_callable = target.callable_effect != CallableEffect::NONE ||
                                                isKnownResourceSafeBuiltin(source_name);
                }
                return;
            }
        } else if (value && value->type == ESHKOL_OP &&
                   value->operation.op == ESHKOL_LAMBDA_OP) {
            target.is_proven_callable = true;
            target.callable_effect = CallableEffect::USER_FUNCTION;
            target.function_node = value;
            target.function_scope_id = env_scope_id;
            named_lambda_nodes_.insert(value);
            return;
        }

        ResourceKind produced_kind = factoryKindOfCall(value);
        if (produced_kind != ResourceKind::NONE) {
            accepted_factory_nodes_.insert(value);
            target.resource_kind = produced_kind;
            target.resource_state = initialFactoryState(produced_kind);
            target.resource_alias = target_name;
            return;
        }
        if (value && value->type == ESHKOL_OP && value->operation.op == ESHKOL_CALL_OP) {
            auto result = last_call_result_.find(value);
            if (result != last_call_result_.end() && result->second.first != ResourceKind::NONE) {
                target.resource_kind = result->second.first;
                target.resource_state = result->second.second;
                target.resource_alias = target_name;
                return;
            }
            if (isUserFunctionCall(value)) {
                accepted_factory_nodes_.insert(value);
                return;
            }
            Callee callee = resolveCallee(value->operation.call_op.func);
            if (callee.effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
                target.callable_effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                target.unknown_reason = "'" + target_name + "' is the result of '" + callee.name +
                    "', whose effect is not proved";
            }
        }

        if (value && value->type == ESHKOL_OP && value->operation.op == ESHKOL_MOVE_OP) {
            const std::string source_name = getVarName(value->operation.move_op.value);
            VariableInfo* source = lookupVariable(source_name);
            if (source && source->resource_kind != ResourceKind::NONE) {
                accepted_move_nodes_.insert(value);
                target.resource_kind = source->resource_kind;
                target.resource_state = source->resource_state == ResourceState::MOVED
                    ? source->moved_from_state : source->resource_state;
                target.resource_alias = target_name;
                return;
            }
        }

        const std::string source_name = getVarName(value);
        VariableInfo* source = lookupVariable(source_name);
        if (source && source->resource_kind != ResourceKind::NONE) {
            reportResourceError("copy of resource '" + source_name +
                "' into '" + target_name + "' is unsupported; use move or borrow");
        } else if (expressionProducesResourceValue(value)) {
            reportResourceError("resource value escapes through initializer for '" +
                target_name + "'");
        }
        if (value && value->type != ESHKOL_VAR && expressionContainsCallableContract(value) &&
            !(value->type == ESHKOL_OP && value->operation.op == ESHKOL_CALL_OP))
            reportResourceError("resource contract callable escapes through initializer for '" +
                target_name + "'");
    }

    void acceptDirectFactoryInitializer(const eshkol_ast_t* value) {
        if (!value || value->type != ESHKOL_OP ||
            value->operation.op != ESHKOL_CALL_OP) return;
        if (factoryKindOfCall(value) != ResourceKind::NONE || isUserFunctionCall(value))
            accepted_factory_nodes_.insert(value);
    }

    // ── closure captures ────────────────────────────────────────────────
    void collectCapturedResourceIds(const eshkol_ast_t* node,
                                    std::vector<std::set<std::string>>& lexical,
                                    std::set<uint64_t>& captures,
                                    unsigned depth = 0) {
        if (!node) return;
        if (depth > 256) {
            capture_scan_truncated_ = true;
            return;
        }
        if (node->type == ESHKOL_VAR) {
            const std::string name = getVarName(node);
            for (auto it = lexical.rbegin(); it != lexical.rend(); ++it)
                if (it->count(name)) return;
            VariableInfo* info = lookupVariable(name);
            if (info && (info->resource_kind != ResourceKind::NONE ||
                         info->callable_effect != CallableEffect::NONE))
                captures.insert(info->binding_id);
            if (!info) {
                // An unshadowed builtin with a handle contract (open-input-file,
                // make-event-loop, ...) is captured like a contract binding.
                const sr::Contract* builtin = sr::findBuiltinContract(name.c_str());
                if (builtin && contractHasHandleEffect(builtin)) captured_builtin_contracts_.insert(name);
            }
            return;
        }
        if (node->type == ESHKOL_CONS) {
            collectCapturedResourceIds(node->cons_cell.car, lexical, captures, depth + 1);
            collectCapturedResourceIds(node->cons_cell.cdr, lexical, captures, depth + 1);
            return;
        }
        if (node->type != ESHKOL_OP) return;
        const auto& op = node->operation;
        switch (op.op) {
            case ESHKOL_LAMBDA_OP: {
                lexical.emplace_back();
                for (uint64_t i = 0; i < op.lambda_op.num_params; ++i)
                    lexical.back().insert(getVarName(&op.lambda_op.parameters[i]));
                if (op.lambda_op.rest_param) lexical.back().insert(op.lambda_op.rest_param);
                collectCapturedResourceIds(op.lambda_op.body, lexical, captures, depth + 1);
                lexical.pop_back();
                break;
            }
            case ESHKOL_DEFINE_OP: {
                if (op.define_op.name && !lexical.empty()) lexical.back().insert(op.define_op.name);
                if (op.define_op.is_function) {
                    lexical.emplace_back();
                    for (uint64_t i = 0; i < op.define_op.num_params; ++i)
                        lexical.back().insert(getVarName(&op.define_op.parameters[i]));
                    if (op.define_op.rest_param) lexical.back().insert(op.define_op.rest_param);
                    collectCapturedResourceIds(op.define_op.value, lexical, captures, depth + 1);
                    lexical.pop_back();
                } else {
                    collectCapturedResourceIds(op.define_op.value, lexical, captures, depth + 1);
                }
                break;
            }
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP: {
                lexical.emplace_back();
                if (op.op == ESHKOL_LETREC_OP || op.op == ESHKOL_LETREC_STAR_OP) {
                    for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                        const auto* binding = &op.let_op.bindings[i];
                        if (binding->type == ESHKOL_CONS)
                            lexical.back().insert(getVarName(binding->cons_cell.car));
                    }
                }
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const auto* binding = &op.let_op.bindings[i];
                    if (binding->type != ESHKOL_CONS) continue;
                    collectCapturedResourceIds(binding->cons_cell.cdr, lexical, captures, depth + 1);
                    if (op.op == ESHKOL_LET_STAR_OP)
                        lexical.back().insert(getVarName(binding->cons_cell.car));
                }
                if (op.op == ESHKOL_LET_OP) {
                    for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                        const auto* binding = &op.let_op.bindings[i];
                        if (binding->type == ESHKOL_CONS)
                            lexical.back().insert(getVarName(binding->cons_cell.car));
                    }
                    if (op.let_op.name) lexical.back().insert(op.let_op.name);
                }
                collectCapturedResourceIds(op.let_op.body, lexical, captures, depth + 1);
                lexical.pop_back();
                break;
            }
            case ESHKOL_CALL_OP:
            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
                if (op.op == ESHKOL_CALL_OP && op.call_op.func)
                    collectCapturedResourceIds(op.call_op.func, lexical, captures, depth + 1);
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectCapturedResourceIds(&op.call_op.variables[i], lexical, captures, depth + 1);
                break;
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
            case ESHKOL_SEQUENCE_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    collectCapturedResourceIds(&op.sequence_op.expressions[i], lexical, captures, depth + 1);
                break;
            case ESHKOL_SET_OP:
                if (op.set_op.name) {
                    bool local = false;
                    for (auto it = lexical.rbegin(); it != lexical.rend() && !local; ++it)
                        local = it->count(op.set_op.name) != 0;
                    VariableInfo* info = local ? nullptr : lookupVariable(op.set_op.name);
                    if (info) captures.insert(info->binding_id);
                }
                collectCapturedResourceIds(op.set_op.value, lexical, captures, depth + 1);
                break;
            case ESHKOL_BORROW_OP:
                collectCapturedResourceIds(op.borrow_op.value, lexical, captures, depth + 1);
                for (uint64_t i = 0; i < op.borrow_op.num_body_exprs; ++i)
                    collectCapturedResourceIds(&op.borrow_op.body[i], lexical, captures, depth + 1);
                break;
            case ESHKOL_MOVE_OP:
                collectCapturedResourceIds(op.move_op.value, lexical, captures, depth + 1);
                break;
            case ESHKOL_OWNED_OP:
                collectCapturedResourceIds(op.owned_op.value, lexical, captures, depth + 1);
                break;
            case ESHKOL_RAISE_OP:
                collectCapturedResourceIds(op.raise_op.exception, lexical, captures, depth + 1);
                break;
            case ESHKOL_CALL_CC_OP:
                collectCapturedResourceIds(op.call_cc_op.proc, lexical, captures, depth + 1);
                break;
            case ESHKOL_DYNAMIC_WIND_OP:
                collectCapturedResourceIds(op.dynamic_wind_op.before, lexical, captures, depth + 1);
                collectCapturedResourceIds(op.dynamic_wind_op.thunk, lexical, captures, depth + 1);
                collectCapturedResourceIds(op.dynamic_wind_op.after, lexical, captures, depth + 1);
                break;
            case ESHKOL_GUARD_OP:
                for (uint64_t i = 0; i < op.guard_op.num_body_exprs; ++i)
                    collectCapturedResourceIds(&op.guard_op.body[i], lexical, captures, depth + 1);
                lexical.emplace_back();
                if (op.guard_op.var_name) lexical.back().insert(op.guard_op.var_name);
                for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i)
                    collectCapturedResourceIds(&op.guard_op.clauses[i], lexical, captures, depth + 1);
                lexical.pop_back();
                break;
            case ESHKOL_DO_OP: {
                // func = (bindings . (test . results)); variables = body
                std::vector<const eshkol_ast_t*> bindings;
                const eshkol_ast_t* test = nullptr;
                const eshkol_ast_t* results = nullptr;
                decomposeDo(op, &bindings, &test, &results);
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                        binding->cons_cell.cdr->type == ESHKOL_CONS)
                        collectCapturedResourceIds(binding->cons_cell.cdr->cons_cell.car, lexical, captures, depth + 1);
                lexical.emplace_back();
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS)
                        lexical.back().insert(getVarName(binding->cons_cell.car));
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                        binding->cons_cell.cdr->type == ESHKOL_CONS)
                        collectCapturedResourceIds(binding->cons_cell.cdr->cons_cell.cdr, lexical, captures, depth + 1);
                collectCapturedResourceIds(test, lexical, captures, depth + 1);
                collectCapturedResourceIds(results, lexical, captures, depth + 1);
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectCapturedResourceIds(&op.call_op.variables[i], lexical, captures, depth + 1);
                lexical.pop_back();
                break;
            }
            case ESHKOL_THE_OP:
                collectCapturedResourceIds(op.the_op.expr, lexical, captures, depth + 1);
                break;
            default:
                break;
        }
    }

    // A closure value whose call sites the checker cannot see may not hold a
    // handle, a loan, a continuation, or a handle contract.
    void checkLambdaResourceCapture(const eshkol_ast_t* lambda,
                                    bool reject_callable_contract_captures = true) {
        if (!lambda || lambda->type != ESHKOL_OP ||
            lambda->operation.op != ESHKOL_LAMBDA_OP) return;
        std::vector<std::set<std::string>> lexical(1);
        for (uint64_t i = 0; i < lambda->operation.lambda_op.num_params; ++i)
            lexical.back().insert(getVarName(&lambda->operation.lambda_op.parameters[i]));
        if (lambda->operation.lambda_op.rest_param)
            lexical.back().insert(lambda->operation.lambda_op.rest_param);
        std::set<uint64_t> captures;
        capture_scan_truncated_ = false;
        captured_builtin_contracts_.clear();
        collectCapturedResourceIds(lambda->operation.lambda_op.body, lexical, captures);
        if (capture_scan_truncated_)
            reportResourceError("closure capture analysis exceeded the checked depth; closure effects are unsupported");
        if (reject_callable_contract_captures)
            for (const auto& name : captured_builtin_contracts_)
                reportResourceError("resource contract callable '" + name +
                    "' is captured by a closure; closure effects are unsupported");
        for (uint64_t id : captures) {
            std::string name;
            VariableInfo* info = findBindingById(id, &name);
            if (!info) continue;
            if (info->resource_kind != ResourceKind::NONE) {
                reportResourceError(isBorrowed(id) || info->lent
                    ? "borrowed resource '" + name + "' escapes through a closure"
                    : "resource '" + name + "' is captured by a closure whose calls the checker "
                      "cannot see; closure effects are unsupported (call it directly, bind it to a "
                      "name, or pass it to a borrow scope)");
            } else if (info->callable_effect == CallableEffect::CONTINUATION) {
                if (hasTrackedResourceBinding())
                    reportResourceError("continuation '" + name + "' is captured by a closure while "
                        "handles are open; the checker cannot follow a re-entry");
            } else if (info->callable_effect == CallableEffect::LOOP) {
                if (hasTrackedResourceBinding())
                    reportResourceError("loop '" + name + "' is captured by a closure while handles "
                        "are open; the checker follows loops only through tail calls");
            } else if (reject_callable_contract_captures &&
                       info->callable_effect != CallableEffect::NONE &&
                       !(info->callable_effect == CallableEffect::USER_FUNCTION &&
                         classifyBinding(*info).cls == FnClass::PURE)) {
                reportResourceError("resource contract callable '" + name +
                    "' is captured by a closure; closure effects are unsupported");
            }
        }
    }

    // ── do-loop and function shapes ─────────────────────────────────────
    static void decomposeDo(const eshkol_operations_t& op,
                            std::vector<const eshkol_ast_t*>* bindings,
                            const eshkol_ast_t** test, const eshkol_ast_t** results) {
        const eshkol_ast_t* main = op.call_op.func;
        if (!main || main->type != ESHKOL_CONS) return;
        const eshkol_ast_t* list = main->cons_cell.car;
        if (list && list->type == ESHKOL_OP && list->operation.op == ESHKOL_CALL_OP)
            for (uint64_t i = 0; i < list->operation.call_op.num_vars; ++i)
                bindings->push_back(&list->operation.call_op.variables[i]);
        const eshkol_ast_t* clause = main->cons_cell.cdr;
        if (clause && clause->type == ESHKOL_CONS) {
            *test = clause->cons_cell.car;
            *results = clause->cons_cell.cdr;
        }
    }

    static FunctionShape functionShape(const eshkol_ast_t* node) {
        FunctionShape shape;
        if (!node || node->type != ESHKOL_OP) return shape;
        if (node->operation.op == ESHKOL_DEFINE_OP && node->operation.define_op.is_function) {
            shape.params = node->operation.define_op.parameters;
            shape.num_params = node->operation.define_op.num_params;
            shape.body = node->operation.define_op.value;
            shape.rest = node->operation.define_op.rest_param;
        } else if (node->operation.op == ESHKOL_LAMBDA_OP) {
            shape.params = node->operation.lambda_op.parameters;
            shape.num_params = node->operation.lambda_op.num_params;
            shape.body = node->operation.lambda_op.body;
            shape.rest = node->operation.lambda_op.rest_param;
        }
        return shape;
    }

    // Syntactic tail positions of a body. Calls found here are the only loop
    // back edges, and handle values found here are the function's results.
    void collectTailNodes(const eshkol_ast_t* root, std::set<const eshkol_ast_t*>& out,
                          bool calls_only) {
        std::vector<const eshkol_ast_t*> pending{root};
        while (!pending.empty()) {
            const auto* node = pending.back();
            pending.pop_back();
            if (!node) continue;
            if (node->type != ESHKOL_OP) {
                if (!calls_only) out.insert(node);
                continue;
            }
            const auto& op = node->operation;
            if (isIfForm(node)) {
                pending.push_back(&op.call_op.variables[1]);
                if (op.call_op.num_vars == 3) pending.push_back(&op.call_op.variables[2]);
                continue;
            }
            switch (op.op) {
                case ESHKOL_CALL_OP:
                    out.insert(node);
                    break;
                case ESHKOL_AND_OP:
                case ESHKOL_OR_OP:
                    if (!calls_only) {
                        for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                            pending.push_back(&op.sequence_op.expressions[i]);
                        break;
                    }
                    [[fallthrough]];
                case ESHKOL_SEQUENCE_OP:
                    if (op.sequence_op.num_expressions)
                        pending.push_back(&op.sequence_op.expressions[op.sequence_op.num_expressions - 1]);
                    break;
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    if (op.op != ESHKOL_LET_OP || !op.let_op.name)
                        pending.push_back(op.let_op.body);
                    break;
                case ESHKOL_WHEN_OP:
                case ESHKOL_UNLESS_OP:
                    if (op.call_op.num_vars > 1)
                        pending.push_back(&op.call_op.variables[op.call_op.num_vars - 1]);
                    break;
                case ESHKOL_COND_OP:
                    for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                        const auto& clause = op.call_op.variables[i];
                        if (clause.type == ESHKOL_OP && clause.operation.op == ESHKOL_CALL_OP &&
                            clause.operation.call_op.num_vars)
                            pending.push_back(&clause.operation.call_op.variables[clause.operation.call_op.num_vars - 1]);
                    }
                    break;
                case ESHKOL_THE_OP:
                    if (!calls_only) pending.push_back(op.the_op.expr);
                    break;
                case ESHKOL_BORROW_OP:
                    if (!calls_only && op.borrow_op.num_body_exprs)
                        pending.push_back(&op.borrow_op.body[op.borrow_op.num_body_exprs - 1]);
                    break;
                default:
                    if (!calls_only) out.insert(node);
                    break;
            }
        }
    }

    // ── function classification (no handle arguments) ───────────────────
    // A function that names no handle, handle contract, loan, or
    // handle-relevant function cannot change a handle: it needs no
    // flow analysis at its call sites. A function that calls an operation
    // with no proved effect is rejected with that operation as the reason.
    FnClassResult classifyBinding(const VariableInfo& fn) {
        FnClassResult result;
        if (fn.callable_effect != CallableEffect::USER_FUNCTION || !fn.function_node) {
            if (fn.callable_effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
                result.cls = FnClass::UNKNOWN;
                result.reason = fn.unknown_reason;
            } else if (fn.callable_effect != CallableEffect::NONE) {
                result.cls = (fn.callable_effect == CallableEffect::CONTRACT &&
                              !contractHasHandleEffect(fn.contract) && fn.contract &&
                              fn.contract->proc_args[0].index < 0)
                    ? FnClass::PURE : FnClass::RELEVANT;
            }
            return result;
        }
        auto memo = classification_memo_.find(fn.function_node);
        if (memo != classification_memo_.end()) return memo->second;
        if (classification_in_progress_.count(fn.function_node)) return result;  // PURE assumption
        classification_in_progress_.insert(fn.function_node);
        int env_depth = findScopeIndexById(fn.function_scope_id);
        if (env_depth < 0) env_depth = 0;
        FunctionShape shape = functionShape(fn.function_node);
        std::vector<std::set<std::string>> lexical(1);
        for (uint64_t i = 0; i < shape.num_params; ++i)
            lexical.back().insert(getVarName(&shape.params[i]));
        if (shape.rest) lexical.back().insert(shape.rest);
        scanFunctionBody(shape.body, lexical, static_cast<size_t>(env_depth), result, false, 0);
        classification_in_progress_.erase(fn.function_node);
        classification_memo_[fn.function_node] = result;
        return result;
    }

    void classifyName(const std::string& name, bool call_position,
                      const std::vector<std::set<std::string>>& lexical, size_t env_depth,
                      FnClassResult& result) {
        if (result.cls == FnClass::UNKNOWN) return;
        for (auto it = lexical.rbegin(); it != lexical.rend(); ++it)
            if (it->count(name)) return;
        static const std::set<std::string> keywords = {
            "if", "else", "=>", "begin", "quote", "lambda", "define", "set!", "let",
            "let*", "letrec", "cond", "and", "or", "when", "unless", "do", "case"};
        if (keywords.count(name)) return;
        VariableInfo* info = lookupVariableUpTo(name, env_depth);
        if (!info) {
            CallableEffect effect;
            const sr::Contract* contract;
            unresolvedCallable(name, &effect, &contract);
            if (effect == CallableEffect::TERMINAL_CONTROL) {
                // exit/error end the path: the caller's open handles matter.
                result.cls = FnClass::RELEVANT;
            } else if (effect == CallableEffect::UNKNOWN_RESOURCE_EFFECT) {
                result.cls = FnClass::UNKNOWN;
                result.reason = "builtin '" + name + "' has no strict resource effect summary";
            } else if (effect == CallableEffect::CONTRACT && contractHasHandleEffect(contract)) {
                result.cls = FnClass::RELEVANT;
            }
            return;
        }
        if (info->is_verified_stdlib_primitive) return;
        if (info->resource_kind != ResourceKind::NONE) {
            result.cls = FnClass::RELEVANT;
            return;
        }
        switch (info->callable_effect) {
            case CallableEffect::NONE:
                return;
            case CallableEffect::TERMINAL_CONTROL:
                result.cls = FnClass::RELEVANT;
                return;
            case CallableEffect::CONTRACT:
                if (contractHasHandleEffect(info->contract) || !call_position)
                    result.cls = FnClass::RELEVANT;
                return;
            case CallableEffect::USER_FUNCTION: {
                FnClassResult nested = classifyBinding(*info);
                if (nested.cls == FnClass::UNKNOWN) {
                    result = nested;
                    if (result.reason.empty()) result.reason = "'" + name + "' has no proved effect";
                } else if (nested.cls == FnClass::RELEVANT) {
                    result.cls = FnClass::RELEVANT;
                }
                return;
            }
            case CallableEffect::CONTINUATION:
            case CallableEffect::LOOP:
                result.cls = FnClass::RELEVANT;
                return;
            case CallableEffect::UNKNOWN_RESOURCE_EFFECT:
                if (call_position) {
                    result.cls = FnClass::UNKNOWN;
                    result.reason = info->unknown_reason.empty()
                        ? "'" + name + "' has no proved resource effect" : info->unknown_reason;
                }
                return;
        }
    }

    void scanFunctionBody(const eshkol_ast_t* node, std::vector<std::set<std::string>>& lexical,
                          size_t env_depth, FnClassResult& result, bool call_position,
                          unsigned depth) {
        if (!node || result.cls == FnClass::UNKNOWN) return;
        if (depth > 256) {
            result.cls = FnClass::UNKNOWN;
            result.reason = "the function body is nested deeper than the checker follows";
            return;
        }
        auto scan = [&](const eshkol_ast_t* child, bool call = false) {
            scanFunctionBody(child, lexical, env_depth, result, call, depth + 1);
        };
        if (node->type == ESHKOL_VAR) {
            classifyName(getVarName(node), call_position, lexical, env_depth, result);
            return;
        }
        if (node->type == ESHKOL_CONS) {
            scan(node->cons_cell.car);
            scan(node->cons_cell.cdr);
            return;
        }
        if (node->type != ESHKOL_OP) return;
        const auto& op = node->operation;
        switch (op.op) {
            case ESHKOL_CALL_OP:
                scan(op.call_op.func, true);
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) scan(&op.call_op.variables[i]);
                return;
            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) scan(&op.call_op.variables[i]);
                return;
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
            case ESHKOL_SEQUENCE_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    scan(&op.sequence_op.expressions[i]);
                return;
            case ESHKOL_LAMBDA_OP:
                lexical.emplace_back();
                for (uint64_t i = 0; i < op.lambda_op.num_params; ++i)
                    lexical.back().insert(getVarName(&op.lambda_op.parameters[i]));
                if (op.lambda_op.rest_param) lexical.back().insert(op.lambda_op.rest_param);
                scan(op.lambda_op.body);
                lexical.pop_back();
                return;
            case ESHKOL_DEFINE_OP:
                if (op.define_op.name) lexical.back().insert(op.define_op.name);
                if (op.define_op.is_external) {
                    result.cls = FnClass::UNKNOWN;
                    result.reason = "it declares an external definition";
                    return;
                }
                if (op.define_op.is_function) {
                    lexical.emplace_back();
                    for (uint64_t i = 0; i < op.define_op.num_params; ++i)
                        lexical.back().insert(getVarName(&op.define_op.parameters[i]));
                    if (op.define_op.rest_param) lexical.back().insert(op.define_op.rest_param);
                    scan(op.define_op.value);
                    lexical.pop_back();
                } else {
                    scan(op.define_op.value);
                }
                return;
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP:
                lexical.emplace_back();
                if (op.op != ESHKOL_LET_OP && op.op != ESHKOL_LET_STAR_OP)
                    for (uint64_t i = 0; i < op.let_op.num_bindings; ++i)
                        if (op.let_op.bindings[i].type == ESHKOL_CONS)
                            lexical.back().insert(getVarName(op.let_op.bindings[i].cons_cell.car));
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const auto& binding = op.let_op.bindings[i];
                    if (binding.type != ESHKOL_CONS) continue;
                    scan(binding.cons_cell.cdr);
                    if (op.op == ESHKOL_LET_STAR_OP)
                        lexical.back().insert(getVarName(binding.cons_cell.car));
                }
                if (op.op == ESHKOL_LET_OP) {
                    for (uint64_t i = 0; i < op.let_op.num_bindings; ++i)
                        if (op.let_op.bindings[i].type == ESHKOL_CONS)
                            lexical.back().insert(getVarName(op.let_op.bindings[i].cons_cell.car));
                    if (op.let_op.name) lexical.back().insert(op.let_op.name);
                }
                scan(op.let_op.body);
                lexical.pop_back();
                return;
            case ESHKOL_SET_OP:
                if (op.set_op.name) {
                    bool local = false;
                    for (auto it = lexical.rbegin(); it != lexical.rend() && !local; ++it)
                        local = it->count(op.set_op.name) != 0;
                    if (!local) {
                        VariableInfo* info = lookupVariableUpTo(op.set_op.name, env_depth);
                        if (info && (info->resource_kind != ResourceKind::NONE ||
                                     info->callable_effect != CallableEffect::NONE ||
                                     info->is_proven_callable))
                            result.cls = FnClass::RELEVANT;
                    }
                }
                scan(op.set_op.value);
                return;
            case ESHKOL_BORROW_OP:
                scan(op.borrow_op.value);
                for (uint64_t i = 0; i < op.borrow_op.num_body_exprs; ++i) scan(&op.borrow_op.body[i]);
                return;
            case ESHKOL_MOVE_OP: scan(op.move_op.value); return;
            case ESHKOL_OWNED_OP: scan(op.owned_op.value); return;
            case ESHKOL_SHARED_OP: scan(op.shared_op.value); return;
            case ESHKOL_WEAK_REF_OP: scan(op.weak_ref_op.value); return;
            case ESHKOL_THE_OP: scan(op.the_op.expr); return;
            case ESHKOL_WITH_REGION_OP:
                for (uint64_t i = 0; i < op.with_region_op.num_body_exprs; ++i)
                    scan(&op.with_region_op.body[i]);
                return;
            case ESHKOL_RAISE_OP:
                if (result.cls == FnClass::PURE) result.cls = FnClass::RELEVANT;
                scan(op.raise_op.exception);
                return;
            case ESHKOL_CALL_CC_OP: scan(op.call_cc_op.proc); return;
            case ESHKOL_DYNAMIC_WIND_OP:
                scan(op.dynamic_wind_op.before);
                scan(op.dynamic_wind_op.thunk);
                scan(op.dynamic_wind_op.after);
                return;
            case ESHKOL_GUARD_OP:
                for (uint64_t i = 0; i < op.guard_op.num_body_exprs; ++i) scan(&op.guard_op.body[i]);
                lexical.emplace_back();
                if (op.guard_op.var_name) lexical.back().insert(op.guard_op.var_name);
                for (uint64_t i = 0; i < op.guard_op.num_clauses; ++i) scan(&op.guard_op.clauses[i]);
                lexical.pop_back();
                return;
            case ESHKOL_DO_OP: {
                std::vector<const eshkol_ast_t*> bindings;
                const eshkol_ast_t* test = nullptr;
                const eshkol_ast_t* results = nullptr;
                decomposeDo(op, &bindings, &test, &results);
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                        binding->cons_cell.cdr->type == ESHKOL_CONS)
                        scan(binding->cons_cell.cdr->cons_cell.car);
                lexical.emplace_back();
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS)
                        lexical.back().insert(getVarName(binding->cons_cell.car));
                for (const auto* binding : bindings)
                    if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                        binding->cons_cell.cdr->type == ESHKOL_CONS)
                        scan(binding->cons_cell.cdr->cons_cell.cdr);
                scan(test);
                if (results && results->type == ESHKOL_OP && results->operation.op == ESHKOL_CALL_OP)
                    for (uint64_t i = 0; i < results->operation.call_op.num_vars; ++i)
                        scan(&results->operation.call_op.variables[i]);
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) scan(&op.call_op.variables[i]);
                lexical.pop_back();
                return;
            }
            case ESHKOL_QUOTE_OP:
            case ESHKOL_DEFINE_TYPE_OP:
            case ESHKOL_IMPORT_OP:
            case ESHKOL_REQUIRE_OP:
            case ESHKOL_PROVIDE_OP:
            case ESHKOL_DEFINE_SYNTAX_OP:
            case ESHKOL_TYPE_ANNOTATION_OP:
                return;
            default:
                result.cls = FnClass::UNKNOWN;
                result.reason = "operator form " + std::to_string(static_cast<int>(op.op)) +
                    " has no strict resource traversal summary";
                return;
        }
    }

    // ── handle arguments ────────────────────────────────────────────────
    ResourceKind useHandleArg(const eshkol_ast_t* argument, const sr::HandleArg& spec,
                              const std::string& operation, bool allow_error_union = false) {
        const std::string name = getVarName(argument);
        VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
        if (!info || info->resource_kind == ResourceKind::NONE) {
            if (spec.optional) return ResourceKind::NONE;  // a memory or current port
            if (name.empty()) {
                reportResourceError(operation + " requires a direct tracked resource owner variable");
            } else {
                reportResourceError(operation + " received an untracked value '" + name +
                    "'; raw or container-derived resource IDs are unsupported");
            }
            return ResourceKind::NONE;
        }
        allowed_resource_use_nodes_.insert(argument);
        if (info->resource_state == ResourceState::CLOSED ||
            info->resource_state == ResourceState::CLOSED_OR_ERROR ||
            info->resource_state == ResourceState::MOVED ||
            info->resource_state == ResourceState::MAYBE_LIVE) {
            reportResourceError("use of closed or moved resource '" + name + "' in " + operation +
                " (use after close: the " + kindText(info->resource_kind) + " is " +
                stateText(info->resource_state) + ")");
            return info->resource_kind;
        }
        if (!allow_error_union && info->resource_state == ResourceState::ERROR_ONLY) {
            reportResourceError("resource operation " + operation +
                " is used on a failed acquisition '" + name + "'");
            return info->resource_kind;
        }
        if (!allow_error_union && info->resource_state == ResourceState::ERROR_OR_LIVE) {
            reportResourceError("resource '" + name + "' must be refined with " +
                sr::failureRefiner(sr::failureOf(info->resource_kind)) + " before " + operation);
            return info->resource_kind;
        }
        if (!(spec.accepts & sr::bit(info->resource_kind))) {
            reportResourceError("resource kind mismatch: " + operation + " expects " +
                acceptedText(spec.accepts) + ", got '" + name + "' (" +
                kindText(info->resource_kind) + ")");
        }
        return info->resource_kind;
    }

    /** @brief Move the owner of @p argument out, for an operation that takes
     *  ownership of the handle (sr::HandleArg::consumed).
     *
     *  The produced value — a TLS session over the stream it was handshaked
     *  on, say — is the single close site from then on, so the caller's owner
     *  becomes moved-from and a later use or close of it is reported. A
     *  borrowed owner cannot be given away at all. */
    void moveHandleOwner(const eshkol_ast_t* argument, const std::string& operation) {
        const std::string name = getVarName(argument);
        VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
        if (!info || info->resource_kind == ResourceKind::NONE) return;
        if (info->lent || isBorrowed(info->binding_id)) {
            reportResourceError("cannot pass borrowed resource '" + name + "' to " + operation +
                ", which takes ownership of it");
            return;
        }
        if (info->resource_state == ResourceState::LIVE) {
            info->moved_from_state = ResourceState::LIVE;
            info->resource_state = ResourceState::MOVED;
            info->state = State::MOVED;
        }
    }

    void closeHandle(const eshkol_ast_t* argument, const sr::Contract& contract,
                     const std::string& operation) {
        const std::string name = getVarName(argument);
        VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
        if (!info || info->resource_kind == ResourceKind::NONE) {
            reportResourceError(operation +
                " requires a tracked resource owner; raw or forged integer IDs are unsupported");
            return;
        }
        allowed_resource_use_nodes_.insert(argument);
        if (!(contract.handle_args[0].accepts & sr::bit(info->resource_kind))) {
            reportResourceError("resource kind mismatch: " + operation + " expects " +
                acceptedText(contract.handle_args[0].accepts) + ", got '" + name + "' (" +
                kindText(info->resource_kind) + ")");
            return;
        }
        if (info->lent) {
            reportResourceError("cannot close borrowed resource '" + name +
                "' inside its borrow scope; the borrow scope closes it");
            return;
        }
        if (isBorrowed(info->binding_id)) {
            reportResourceError("cannot consume borrowed resource '" + name +
                "' (close while borrowed)");
            return;
        }
        if (info->resource_state == ResourceState::CLOSED ||
            info->resource_state == ResourceState::CLOSED_OR_ERROR) {
            reportResourceError("duplicate close of resource '" + name + "' (double close of a " +
                kindText(info->resource_kind) + ")");
            return;
        }
        if (info->resource_state == ResourceState::MOVED ||
            info->resource_state == ResourceState::MAYBE_LIVE) {
            reportResourceError("duplicate close or use after close of resource '" + name + "' (it is " +
                stateText(info->resource_state) + ")");
            return;
        }
        if (info->resource_state == ResourceState::ERROR_ONLY)
            return;  // failed acquisition owns no runtime handle
        info->resource_state = info->resource_state == ResourceState::ERROR_OR_LIVE
            ? ResourceState::CLOSED_OR_ERROR : ResourceState::CLOSED;
        info->state = State::MOVED;
    }

    // ── procedure arguments of higher-order contracts ───────────────────
    void setupProcArgs(const eshkol_ast_t* node, const sr::Contract& contract,
                       ResourceKind lend_kind, std::vector<WorkItem>& work) {
        const auto& call = node->operation.call_op;
        for (const auto& spec : contract.proc_args) {
            if (spec.index < 0 || static_cast<uint64_t>(spec.index) >= call.num_vars) continue;
            const eshkol_ast_t* arg = &call.variables[spec.index];
            const bool lends_here = contract.role == sr::Role::LEND &&
                                    spec.index == contract.proc_args[0].index;
            proc_arg_nodes_.insert(arg);
            if (arg->type == ESHKOL_OP && arg->operation.op == ESHKOL_LAMBDA_OP) {
                InlineLambda inline_info;
                inline_info.mode = spec.mode;
                inline_info.lend_kind = lends_here ? lend_kind : ResourceKind::NONE;
                inline_info.lender = contract.name;
                inline_lambdas_[arg] = inline_info;
                continue;
            }
            const std::string name = getVarName(arg);
            VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
            if (info && info->callable_effect == CallableEffect::USER_FUNCTION) {
                if (spec.mode == sr::ProcMode::THREAD) {
                    if (classifyBinding(*info).cls != FnClass::PURE)
                        reportResourceError("thread body '" + name + "' uses handles or handle "
                            "operations; handles cannot cross a thread boundary in the checked fragment");
                    continue;
                }
                PendingCall pending;
                pending.function = *info;
                pending.name = name;
                pending.mode = spec.mode;
                if (lends_here) {
                    ArgHandle lent;
                    lent.index = 0;
                    lent.kind = lend_kind;
                    lent.state = ResourceState::LIVE;
                    lent.lent = true;
                    lent.name = "the handle lent by " + std::string(contract.name);
                    pending.handles.push_back(lent);
                    pending.lender = &contract;
                }
                const uint64_t id = next_frame_id_++;
                pending_procs_[id] = std::move(pending);
                WorkItem apply = WorkItem::marker(WorkItem::Kind::USER_CALL_APPLY, node, id);
                apply.flag = true;  // a procedure argument, not the call itself
                work.push_back(apply);
                continue;
            }
            if (info && info->callable_effect == CallableEffect::CONTRACT && info->contract) {
                if (info->contract->role == sr::Role::CLOSE && lends_here)
                    reportResourceError(std::string(contract.name) + " closes the handle it lends; passing " +
                        name + " would close it twice");
                else if (info->contract->role != sr::Role::SAFE &&
                         info->contract->role != sr::Role::OBSERVE)
                    reportResourceError("resource contract callable escapes through '" +
                        std::string(contract.name) + "'");
                continue;
            }
            if (!info && !name.empty()) {
                CallableEffect effect;
                const sr::Contract* builtin;
                unresolvedCallable(name, &effect, &builtin);
                if (effect == CallableEffect::CONTRACT && builtin) {
                    if (builtin->role == sr::Role::CLOSE && lends_here)
                        reportResourceError(std::string(contract.name) + " closes the handle it lends; passing " +
                            name + " would close it twice");
                    else if (builtin->role != sr::Role::SAFE && builtin->role != sr::Role::OBSERVE)
                        reportResourceError("resource contract callable escapes through '" +
                            std::string(contract.name) + "'");
                    continue;
                }
                if (effect == CallableEffect::NONE) continue;
            }
            if (lends_here || hasTrackedResourceBinding()) {
                reportResourceError("the procedure passed to " + std::string(contract.name) +
                    " is computed; the checker cannot see its effect on open handles "
                    "(pass a lambda or a named function)");
            }
        }
    }

    // ── user function calls ─────────────────────────────────────────────
    void handleUserCall(const eshkol_ast_t* node, const Callee& callee,
                        std::vector<WorkItem>& work) {
        const auto& call = node->operation.call_op;
        PendingCall pending;
        pending.function = *callee.binding;
        pending.name = callee.name;
        std::set<uint64_t> seen;
        for (uint64_t i = 0; i < call.num_vars; ++i) {
            const eshkol_ast_t* arg = &call.variables[i];
            const std::string name = getVarName(arg);
            VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
            if (info && info->resource_kind != ResourceKind::NONE) {
                allowed_resource_use_nodes_.insert(arg);
                if (!seen.insert(info->binding_id).second) {
                    reportResourceError("handle '" + name + "' is passed twice to '" + callee.name +
                        "'; two parameters cannot own one handle");
                    continue;
                }
                ArgHandle handle;
                handle.index = static_cast<size_t>(i);
                handle.caller_id = info->binding_id;
                handle.name = name;
                pending.handles.push_back(handle);
                continue;
            }
            if (info && info->is_proven_callable && info->callable_effect != CallableEffect::NONE) {
                FnClassResult cls = classifyBinding(*info);
                if (cls.cls != FnClass::PURE) {
                    reportResourceError("procedure '" + name + "' is passed as a value to '" + callee.name +
                        "' but uses handles or handle operations; the checker follows handle effects "
                        "only through direct calls and borrow scopes");
                } else {
                    pending.pure_callables.insert(static_cast<size_t>(i));
                    proc_arg_nodes_.insert(arg);
                }
                continue;
            }
            if (arg->type == ESHKOL_OP && arg->operation.op == ESHKOL_LAMBDA_OP) {
                if (!named_lambda_nodes_.count(arg)) {
                    // A closure handed to a user function is a value the callee may keep.
                    named_lambda_nodes_.insert(arg);
                    checkLambdaResourceCapture(arg);
                }
                continue;
            }
            if (expressionProducesResourceValue(arg) && factoryKindOfCall(arg) == ResourceKind::NONE)
                reportResourceError("resource value escapes through call to '" + callee.name + "'");
        }
        const bool flow = !pending.handles.empty() || classifyBinding(*callee.binding).cls == FnClass::RELEVANT;
        if (!flow) {
            FnClassResult cls = classifyBinding(*callee.binding);
            if (cls.cls == FnClass::UNKNOWN)
                reportResourceError("call to '" + callee.name + "' is unsupported in strict mode: " +
                    (cls.reason.empty() ? "its effect is not proved" : cls.reason));
            if (fn_ && fn_->return_nodes.count(node)) fn_->returned_value = true;
            return;
        }
        const uint64_t id = next_frame_id_++;
        pending_procs_[id] = std::move(pending);
        work.push_back(WorkItem::marker(WorkItem::Kind::USER_CALL_APPLY, node, id));
    }

    std::string summaryKey(const VariableInfo& fn, int env_depth,
                           const std::vector<ArgHandle>& handles,
                           const std::set<size_t>& pure_callables) const {
        std::string key = std::to_string(reinterpret_cast<uintptr_t>(fn.function_node)) + "|" +
                          std::to_string(env_depth) + "|" + std::to_string(callable_epoch_) + "|" +
                          ((handler_depth_ > 0 || hasHandlerFrame()) ? "h" : "-") + "|";
        for (int i = 0; i <= env_depth && i < static_cast<int>(scope_stack_.size()); ++i) {
            for (const auto& entry : scope_stack_[i].variables) {
                const auto& v = entry.second;
                if (v.resource_kind == ResourceKind::NONE) continue;
                auto borrow = borrowed_counts_.find(v.binding_id);
                key += std::to_string(v.binding_id) + ":" +
                       std::to_string(static_cast<int>(v.resource_state)) + ":" +
                       std::to_string(borrow == borrowed_counts_.end() ? 0 : borrow->second) +
                       (v.lent ? "l" : "") + ",";
            }
        }
        key += "|";
        for (const auto& h : handles)
            key += std::to_string(h.index) + ":" + std::to_string(static_cast<int>(h.kind)) + ":" +
                   std::to_string(static_cast<int>(h.state)) + (h.borrowed ? "b" : "") +
                   (h.lent ? "l" : "") + ",";
        key += "|";
        for (size_t index : pure_callables) key += std::to_string(index) + ",";
        return key;
    }

    static bool sameMutable(const VariableInfo& a, const VariableInfo& b) {
        return a.resource_kind == b.resource_kind && a.resource_state == b.resource_state &&
               a.moved_from_state == b.moved_from_state && a.state == b.state &&
               a.callable_effect == b.callable_effect && a.contract == b.contract &&
               a.function_node == b.function_node && a.is_proven_callable == b.is_proven_callable &&
               a.lent == b.lent && a.is_defined == b.is_defined;
    }

    static bool sameSummary(const Summary& a, const Summary& b) {
        if (a.dead != b.dead || a.returns_handle != b.returns_handle ||
            a.may_exit != b.may_exit || a.may_raise_unhandled != b.may_raise_unhandled ||
            a.return_kind != b.return_kind || a.return_state != b.return_state ||
            a.env_post.size() != b.env_post.size() || a.param_post.size() != b.param_post.size())
            return false;
        for (size_t i = 0; i < a.env_post.size(); ++i)
            if (a.env_post[i].first != b.env_post[i].first ||
                !sameMutable(a.env_post[i].second, b.env_post[i].second)) return false;
        for (size_t i = 0; i < a.param_post.size(); ++i)
            if (a.param_post[i].index != b.param_post[i].index ||
                a.param_post[i].resource_state != b.param_post[i].resource_state ||
                a.param_post[i].state != b.param_post[i].state) return false;
        return true;
    }

    static void copyMutable(VariableInfo& target, const VariableInfo& source) {
        target.resource_kind = source.resource_kind;
        target.resource_state = source.resource_state;
        target.moved_from_state = source.moved_from_state;
        target.state = source.state;
        target.callable_effect = source.callable_effect;
        target.contract = source.contract;
        target.function_node = source.function_node;
        target.function_scope_id = source.function_scope_id;
        target.is_proven_callable = source.is_proven_callable;
        target.lent = source.lent;
        target.is_defined = source.is_defined;
        target.unknown_reason = source.unknown_reason;
    }

    SavedContext saveContext() const {
        SavedContext saved;
        saved.scopes = scope_stack_;
        saved.borrows = borrowed_counts_;
        saved.dead = dead_;
        saved.frames = frames_;
        saved.handler_depth = handler_depth_;
        saved.fn = fn_;
        saved.effects = active_resource_effects_enabled_;
        saved.markers = body_markers_;
        return saved;
    }

    void restoreContext(SavedContext& saved) {
        scope_stack_ = std::move(saved.scopes);
        borrowed_counts_ = std::move(saved.borrows);
        dead_ = saved.dead;
        frames_ = std::move(saved.frames);
        handler_depth_ = saved.handler_depth;
        fn_ = saved.fn;
        active_resource_effects_enabled_ = saved.effects;
        body_markers_ = std::move(saved.markers);
    }

    // Analyze one call of a user function against the caller's handle
    // states. Recursive calls reach a fixed point starting from "does not
    // return" (a tail call is a back edge, so only base cases contribute).
    Summary analyzeFunctionCall(const VariableInfo& fn, const std::string& name,
                                const std::vector<ArgHandle>& handles,
                                const std::set<size_t>& pure_callables) {
        int env_depth = findScopeIndexById(fn.function_scope_id);
        const bool escaped = env_depth < 0;
        if (escaped) env_depth = 0;
        const std::string key = summaryKey(fn, env_depth, handles, pure_callables);
        auto memo = summary_memo_.find(key);
        if (memo != summary_memo_.end()) return memo->second;
        auto running = in_progress_.find(key);
        if (running != in_progress_.end()) {
            running->second.used = true;
            for (size_t j = running->second.depth + 1; j < memoizable_stack_.size(); ++j)
                memoizable_stack_[j] = false;
            return running->second.assumption;
        }
        if (call_chain_.size() > 64) {
            reportResourceError("call chain through '" + name + "' is deeper than the checker follows");
            return Summary{};
        }
        const size_t depth = memoizable_stack_.size();
        memoizable_stack_.push_back(true);
        InProgress progress;
        progress.depth = depth;
        progress.assumption.dead = true;
        in_progress_[key] = progress;
        const size_t error_mark = errors_.size();
        Summary result;
        bool converged = false;
        for (int iteration = 0; iteration < 6; ++iteration) {
            in_progress_[key].used = false;
            truncateErrors(error_mark);
            Summary current = runFunctionBody(fn, name, env_depth, escaped, handles, pure_callables);
            if (!in_progress_[key].used || sameSummary(current, in_progress_[key].assumption)) {
                result = std::move(current);
                converged = true;
                break;
            }
            in_progress_[key].assumption = current;
            result = std::move(current);
        }
        in_progress_.erase(key);
        const bool memoizable = memoizable_stack_.back();
        memoizable_stack_.pop_back();
        if (!converged)
            reportResourceError("recursive function '" + name +
                "' does not reach a stable handle state; the checker cannot summarize it");
        if (memoizable && converged) summary_memo_[key] = result;
        return result;
    }

    Summary runFunctionBody(const VariableInfo& fn, const std::string& name, int env_depth,
                            bool escaped, const std::vector<ArgHandle>& handles,
                            const std::set<size_t>& pure_callables) {
        SavedContext saved = saveContext();
        const bool caller_has_handler = saved.handler_depth > 0 || hasHandlerFrame();
        std::vector<Scope> env(saved.scopes.begin(),
                               saved.scopes.begin() + std::min<size_t>(saved.scopes.size(),
                                                                       static_cast<size_t>(env_depth) + 1));
        scope_stack_ = std::move(env);
        body_markers_.clear();
        // While the call runs, a handle passed as an argument is owned by its
        // parameter; the caller's binding is not a second owner the callee
        // may use or must close.
        for (const auto& handle : handles) {
            if (handle.caller_id == 0) continue;
            if (VariableInfo* outer = findBindingById(handle.caller_id)) {
                outer->moved_from_state = outer->resource_state;
                outer->resource_state = ResourceState::MOVED;
            }
        }
        borrowed_counts_ = saved.borrows;
        dead_ = false;
        frames_.clear();
        handler_depth_ = caller_has_handler ? 1 : 0;
        FunctionShape shape = functionShape(fn.function_node);
        FunctionContext context;
        context.name = name;
        collectTailNodes(shape.body, context.return_nodes, false);
        for (const auto* tail : context.return_nodes)
            if (tail->type == ESHKOL_OP && tail->operation.op == ESHKOL_CALL_OP)
                accepted_factory_nodes_.insert(tail);
        fn_ = &context;
        call_chain_.push_back(name);
        pushScope("function " + name);
        context.function_depth = scope_stack_.size() - 1;
        std::map<size_t, std::string> param_names;
        for (uint64_t i = 0; i < shape.num_params; ++i) {
            const std::string param = getVarName(&shape.params[i]);
            if (param.empty()) continue;
            VariableInfo info{State::UNOWNED, param, false, next_binding_id_++};
            info.is_defined = true;
            if (pure_callables.count(static_cast<size_t>(i))) info.is_proven_callable = true;
            for (const auto& handle : handles) {
                if (handle.index != i) continue;
                info.resource_kind = handle.kind;
                info.resource_state = handle.state;
                info.resource_alias = param;
                info.lent = handle.lent;
                if (handle.borrowed || handle.lent) ++borrowed_counts_[info.binding_id];
                context.param_ids.insert(info.binding_id);
                param_names[handle.index] = param;
            }
            currentScope().variables[param] = std::move(info);
        }
        if (shape.rest) currentScope().variables[shape.rest] =
            VariableInfo{State::UNOWNED, shape.rest, false, next_binding_id_++};
        (void)escaped;
        analyzeAST(shape.body, true);

        Summary summary;
        summary.dead = dead_;
        if (!dead_) {
            for (const auto& entry : currentScope().variables) {
                const auto& info = entry.second;
                if (context.param_ids.count(info.binding_id) || info.lent) continue;
                if (info.resource_kind != ResourceKind::NONE && isLiveResourceState(info.resource_state))
                    reportResourceError("handle '" + entry.first + "' (" + kindText(info.resource_kind) +
                        ") may still be open when function '" + name +
                        "' returns; close it on every path or return it");
            }
            if (context.returned_handle && context.returned_value)
                reportResourceError("function '" + name + "' returns an owned handle on some paths "
                    "and another value on others; a caller cannot know whether to close the result");
            if (context.return_conflict)
                reportResourceError("function '" + name + "' returns handles of different kinds");
        }
        for (const auto& entry : param_names) {
            auto it = currentScope().variables.find(entry.second);
            if (it == currentScope().variables.end()) continue;
            ParamPost post;
            post.index = entry.first;
            post.resource_state = it->second.resource_state;
            post.moved_from_state = it->second.moved_from_state;
            post.state = it->second.state;
            summary.param_post.push_back(post);
            auto borrow = borrowed_counts_.find(it->second.binding_id);
            if (borrow != borrowed_counts_.end()) borrowed_counts_.erase(borrow);
        }
        summary.returns_handle = context.returned_handle && !context.returned_value;
        summary.may_exit = context.may_exit;
        summary.may_raise_unhandled = context.may_raise_unhandled;
        summary.return_kind = context.return_kind;
        summary.return_state = context.return_state;
        popScope();
        for (size_t i = 0; i < scope_stack_.size() && i < saved.scopes.size(); ++i) {
            for (const auto& entry : scope_stack_[i].variables) {
                auto before = saved.scopes[i].variables.find(entry.first);
                if (before != saved.scopes[i].variables.end() &&
                    before->second.binding_id == entry.second.binding_id &&
                    sameMutable(before->second, entry.second))
                    continue;
                if (before == saved.scopes[i].variables.end() ||
                    before->second.binding_id != entry.second.binding_id)
                    continue;  // a binding the callee introduced at an outer level
                summary.env_post.emplace_back(entry.second.binding_id, entry.second);
            }
        }
        call_chain_.pop_back();
        restoreContext(saved);
        return summary;
    }

    // A callee that may end the process, or raise with no handler anywhere
    // in the chain, does so while the caller's own handles are open.
    void checkCallerHandlesAtProgramEnd(const PendingCall& pending, const Summary& summary) {
        if (dead_ || !(summary.may_exit || summary.may_raise_unhandled)) return;
        if (fn_) {
            fn_->may_exit = fn_->may_exit || summary.may_exit;
            fn_->may_raise_unhandled = fn_->may_raise_unhandled || summary.may_raise_unhandled;
        }
        // The callee already checked its parameters and the scopes it closes
        // over; only the caller's own deeper scopes are left.
        std::set<uint64_t> passed;
        for (const auto& handle : pending.handles) passed.insert(handle.caller_id);
        const int env_depth = std::max(0, findScopeIndexById(pending.function.function_scope_id));
        for (size_t i = static_cast<size_t>(env_depth) + 1; i < scope_stack_.size(); ++i)
            for (const auto& entry : scope_stack_[i].variables) {
                const auto& info = entry.second;
                if (info.resource_kind == ResourceKind::NONE || info.lent || passed.count(info.binding_id) ||
                    !isLiveResourceState(info.resource_state))
                    continue;
                reportResourceError("call to '" + pending.name + "' may " +
                    (summary.may_exit ? std::string("end the process") : std::string("raise with no handler")) +
                    " while resource '" + entry.first + "' may still be live");
            }
    }

    void applySummary(const eshkol_ast_t* node, const PendingCall& pending, const Summary& summary) {
        checkCallerHandlesAtProgramEnd(pending, summary);
        for (const auto& entry : summary.env_post) {
            VariableInfo* target = findBindingById(entry.first);
            if (target) copyMutable(*target, entry.second);
        }
        for (const auto& post : summary.param_post) {
            for (const auto& handle : pending.handles) {
                if (handle.index != post.index) continue;
                if (handle.caller_id == 0) {
                    if (post.resource_state != ResourceState::LIVE && !summary.dead)
                        reportResourceError("'" + pending.name + "' does not leave " + handle.name +
                            " open; a borrow scope closes its handle itself");
                    continue;
                }
                VariableInfo* target = findBindingById(handle.caller_id);
                if (!target) continue;
                target->resource_state = post.resource_state;
                target->moved_from_state = post.moved_from_state;
                target->state = post.state;
            }
        }
        if (pending.mode == sr::ProcMode::MANY) {
            bool changes = false;
            for (const auto& entry : summary.env_post)
                if (entry.second.resource_kind != ResourceKind::NONE) changes = true;
            if (changes)
                reportResourceError("procedure '" + pending.name + "' runs zero or more times but "
                    "changes the state of a handle; it must leave every handle as it found it");
        }
        if (!node) {
            if (summary.returns_handle)
                reportResourceError("procedure '" + pending.name + "' returns an owned handle to a "
                    "higher-order operation that does not track it");
            if (summary.dead) dead_ = true;
            return;
        }
        if (summary.returns_handle) {
            last_call_result_[node] = {summary.return_kind, summary.return_state};
            auto bind = pending_bind_.find(node);
            if (bind != pending_bind_.end()) {
                VariableInfo* target = findBindingById(bind->second);
                if (target) {
                    target->resource_kind = summary.return_kind;
                    target->resource_state = summary.return_state;
                    target->resource_alias = pending.name;
                }
            }
            if (fn_ && fn_->return_nodes.count(node)) {
                recordReturnedHandle(summary.return_kind, summary.return_state);
            } else if (bind == pending_bind_.end() && !let_init_nodes_.count(node)) {
                reportResourceError("factory result from '" + pending.name +
                    "' must be bound directly to one lexical owner");
            }
        } else if (fn_ && fn_->return_nodes.count(node)) {
            fn_->returned_value = true;
        }
        if (summary.dead) dead_ = true;
    }

    void recordReturnedHandle(ResourceKind kind, ResourceState state) {
        if (!fn_ || dead_) return;
        if (fn_->returned_handle && fn_->return_kind != kind) fn_->return_conflict = true;
        if (fn_->returned_handle)
            fn_->return_state = joinResourceState(fn_->return_state, state);
        else
            fn_->return_state = state;
        fn_->returned_handle = true;
        fn_->return_kind = kind;
    }

    void applyPendingUserCall(const WorkItem& item) {
        auto it = pending_procs_.find(item.branch_id);
        if (it == pending_procs_.end()) return;
        PendingCall pending = std::move(it->second);
        pending_procs_.erase(it);
        if (dead_) return;
        // Argument handles are read after the arguments were evaluated.
        for (auto& handle : pending.handles) {
            if (handle.caller_id == 0) continue;
            std::string name;
            VariableInfo* info = findBindingById(handle.caller_id, &name);
            if (!info) continue;
            handle.kind = info->resource_kind;
            handle.state = info->resource_state;
            handle.borrowed = isBorrowed(info->binding_id) || info->lent;
            handle.lent = false;
            if (info->resource_state == ResourceState::CLOSED ||
                info->resource_state == ResourceState::CLOSED_OR_ERROR ||
                info->resource_state == ResourceState::MOVED ||
                info->resource_state == ResourceState::MAYBE_LIVE) {
                reportResourceError("use of closed or moved resource '" + name + "' in call to '" +
                    pending.name + "' (use after close: the " + kindText(info->resource_kind) +
                    " is " + stateText(info->resource_state) + ")");
                return;
            }
        }
        Summary summary = analyzeFunctionCall(pending.function, pending.name, pending.handles,
                                              pending.pure_callables);
        if (item.flag) {
            // a procedure argument of a borrow scope or higher-order builtin
            applySummary(nullptr, pending, summary);
            return;
        }
        applySummary(item.ast, pending, summary);
        recordRaisePoint("an exception raised inside '" + pending.name + "'");
    }

    // ── contract calls ──────────────────────────────────────────────────
    void handleContractCall(const eshkol_ast_t* node, const sr::Contract& contract,
                            const std::string& function_name, std::vector<WorkItem>& work) {
        const auto& call = node->operation.call_op;
        auto arg = [&](int8_t index) -> const eshkol_ast_t* {
            return index >= 0 && static_cast<uint64_t>(index) < call.num_vars
                ? &call.variables[index] : nullptr;
        };
        if (contract.may_raise) recordRaisePoint("an exception raised by " + function_name);
        switch (contract.role) {
            case sr::Role::FACTORY: {
                const bool returned = fn_ && fn_->return_nodes.count(node);
                if (!accepted_factory_nodes_.count(node) && !returned) {
                    reportResourceError("factory result from '" + function_name +
                        "' must be bound directly to one lexical owner");
                }
                for (const auto& spec : contract.handle_args)
                    if (const eshkol_ast_t* handle = arg(spec.index)) {
                        useHandleArg(handle, spec, function_name);
                        if (spec.consumed) moveHandleOwner(handle, function_name);
                    }
                if (contract.proc_args[0].index >= 0)
                    setupProcArgs(node, contract, ResourceKind::NONE, work);
                if (returned) recordReturnedHandle(contract.produces, initialFactoryState(contract.produces));
                return;
            }
            case sr::Role::CLOSE:
                if (const eshkol_ast_t* handle = arg(contract.handle_args[0].index))
                    closeHandle(handle, contract, function_name);
                break;
            case sr::Role::OBSERVE:
                for (const auto& spec : contract.handle_args)
                    if (const eshkol_ast_t* handle = arg(spec.index))
                        useHandleArg(handle, spec, function_name);
                if (contract.proc_args[0].index >= 0)
                    setupProcArgs(node, contract, ResourceKind::NONE, work);
                break;
            case sr::Role::ERROR_PREDICATE:
            case sr::Role::ERROR_OBSERVE: {
                const eshkol_ast_t* value = arg(0);
                VariableInfo* info = value ? lookupVariable(getVarName(value)) : nullptr;
                if (info && info->resource_kind != ResourceKind::NONE) {
                    allowed_resource_use_nodes_.insert(value);
                    if (sr::failureOf(info->resource_kind) != contract.family)
                        reportResourceError(function_name + " cannot refine '" + getVarName(value) +
                            "', a " + kindText(info->resource_kind) + "; its failures are refined by " +
                            sr::failureRefiner(sr::failureOf(info->resource_kind)));
                    else if (contract.role == sr::Role::ERROR_OBSERVE &&
                             info->resource_state != ResourceState::ERROR_ONLY)
                        reportResourceError(function_name + " requires the error branch of an acquisition");
                }
                break;
            }
            case sr::Role::LEND: {
                ResourceKind lent = contract.produces;
                // A handle the scope only reads stays with its owner; one the
                // scope takes over moves, because the scope's own close is from
                // then on the single close site for it.
                for (const auto& spec : contract.handle_args) {
                    if (contract.consumes_handle && spec.index == contract.handle_args[0].index)
                        continue;
                    if (const eshkol_ast_t* handle = arg(spec.index)) {
                        useHandleArg(handle, spec, function_name);
                        if (spec.consumed) moveHandleOwner(handle, function_name);
                    }
                }
                if (contract.consumes_handle) {
                    const eshkol_ast_t* handle = arg(contract.handle_args[0].index);
                    const std::string name = getVarName(handle);
                    VariableInfo* info = name.empty() ? nullptr : lookupVariable(name);
                    lent = handle ? useHandleArg(handle, contract.handle_args[0], function_name)
                                  : ResourceKind::NONE;
                    if (info && info->resource_kind != ResourceKind::NONE) {
                        if (info->lent || isBorrowed(info->binding_id)) {
                            reportResourceError("cannot pass borrowed resource '" + name + "' to " +
                                function_name + ", which closes it");
                        } else if (info->resource_state == ResourceState::LIVE) {
                            info->moved_from_state = ResourceState::LIVE;
                            info->resource_state = ResourceState::MOVED;
                            info->state = State::MOVED;
                            WorkItem after = WorkItem::marker(WorkItem::Kind::CALL_AFTER, node,
                                                              info->binding_id);
                            after.flag = true;  // close the consumed owner after the loan
                            work.push_back(after);
                        }
                    }
                }
                if (contract.proc_args[0].index >= 0)
                    setupProcArgs(node, contract, lent, work);
                if (fn_ && fn_->return_nodes.count(node)) fn_->returned_value = true;
                return;
            }
            case sr::Role::SAFE:
                if (contract.proc_args[0].index >= 0)
                    setupProcArgs(node, contract, ResourceKind::NONE, work);
                break;
        }
        if (fn_ && fn_->return_nodes.count(node)) fn_->returned_value = true;
    }

    void checkPlainCallArguments(const eshkol_ast_t* node, const std::string& function_name) {
        const auto& call = node->operation.call_op;
        for (uint64_t i = 0; i < call.num_vars; ++i) {
            const eshkol_ast_t* arg = &call.variables[i];
            // Handle arguments the contract accepted, and procedure
            // arguments it runs inline, are not escapes.
            if (proc_arg_nodes_.count(arg) || allowed_resource_use_nodes_.count(arg)) continue;
            if (expressionProducesResourceValue(arg)) {
                reportResourceError("resource value escapes through unsupported call '" +
                    function_name + "'");
                return;
            }
            if (expressionContainsCallableContract(arg)) {
                reportResourceError("resource contract callable escapes through '" +
                    function_name + "'");
                return;
            }
        }
    }

    void handleContinuationCall(const eshkol_ast_t* node, const Callee& callee) {
        Frame* target = findFrame(callee.binding->frame_id);
        if (!target) {
            if (hasTrackedResourceBinding())
                reportResourceError("continuation '" + callee.name + "' is invoked outside the dynamic "
                    "extent the checker follows while handles are open");
            dead_ = true;
            return;
        }
        routeEscape(snapshot(), callee.binding->frame_id, "continuation '" + callee.name + "'");
        dead_ = true;
        (void)node;
    }

    // Send an escape state to its call/cc frame, through any dynamic-wind
    // frames in between: their after thunks run first.
    void routeEscape(Snapshot state, uint64_t target_id, const std::string& reason) {
        for (auto it = frames_.rbegin(); it != frames_.rend(); ++it) {
            if (it->id == target_id) {
                addExit(*it, std::move(state), reason);
                return;
            }
            if (it->kind == FrameKind::WIND) {
                it->escape_targets.insert(target_id);
                addExit(*it, std::move(state), reason);
                return;
            }
        }
    }

    void handleLoopCall(const eshkol_ast_t* node, const Callee& callee,
                        std::vector<WorkItem>& work) {
        Frame* frame = findFrame(callee.binding->frame_id);
        const bool tail = frame && frame->tail_calls.count(node) && &frames_.back() == frame;
        if (tail) {
            work.push_back(WorkItem::marker(WorkItem::Kind::LOOP_BACKEDGE, node, frame->id));
            return;
        }
        if ((frame && !frame->entry_states.empty()) || hasTrackedResourceBinding())
            reportResourceError("call to loop '" + callee.name + "' is not in tail position; the "
                "checker follows loops through tail calls only while handles are in scope");
    }

    void checkBackEdge(const Frame& frame, const std::string& what) {
        if (dead_) return;
        for (const auto& entry : frame.entry_states) {
            std::string name;
            VariableInfo* info = findBindingById(entry.first, &name);
            if (!info) continue;
            if (info->resource_state != entry.second)
                reportResourceError(what + " leaves handle '" + name + "' " +
                    stateText(info->resource_state) + " on its back edge, but it was " +
                    stateText(entry.second) + " when the loop started; every iteration must leave "
                    "a handle as it found it");
        }
        for (size_t i = frame.depth; i < scope_stack_.size(); ++i) {
            for (const auto& entry : scope_stack_[i].variables) {
                const auto& info = entry.second;
                if (frame.entry_states.count(info.binding_id) || info.lent) continue;
                if (info.resource_kind != ResourceKind::NONE && isLiveResourceState(info.resource_state))
                    reportResourceError("handle '" + entry.first + "' acquired in an iteration of " +
                        what + " is still open when the next iteration starts");
            }
        }
        if (borrowed_counts_ != frame.entry_borrows)
            reportResourceError(what + " starts its next iteration inside a borrow scope");
    }

    Frame makeLoopFrame(FrameKind kind, const std::string& name, uint64_t binding_id) {
        Frame frame;
        frame.kind = kind;
        frame.id = next_frame_id_++;
        // The loop's own scope is checked on the back edge: a handle defined
        // directly in the body must be closed before the next iteration.
        frame.depth = scope_stack_.empty() ? 0 : scope_stack_.size() - 1;
        frame.loop_name = name;
        frame.loop_binding_id = binding_id;
        for (const auto& scope : scope_stack_)
            for (const auto& entry : scope.variables)
                if (entry.second.resource_kind != ResourceKind::NONE)
                    frame.entry_states[entry.second.binding_id] = entry.second.resource_state;
        frame.entry_borrows = borrowed_counts_;
        return frame;
    }

    void processResourceCall(const eshkol_ast_t* node, std::vector<WorkItem>& work) {
        if (!strict_resources_ || !active_resource_effects_enabled_ || !node ||
            node->type != ESHKOL_OP || node->operation.op != ESHKOL_CALL_OP) return;
        const auto& call = node->operation.call_op;
        if (call.func && call.func->type == ESHKOL_VAR) call_position_nodes_.insert(call.func);
        if (call.func && call.func->type == ESHKOL_OP &&
            call.func->operation.op == ESHKOL_LAMBDA_OP) {
            // ((lambda (x ...) body) arg ...): the body runs once, here.
            for (uint64_t i = 0; i < call.num_vars; ++i)
                if (expressionProducesResourceValue(&call.variables[i]))
                    reportResourceError("pass a handle to a named procedure or bind it with let; an "
                        "immediately applied lambda does not take handle arguments");
            InlineLambda inline_info;
            inline_info.mode = sr::ProcMode::ONCE;
            inline_lambdas_[call.func] = inline_info;
            return;
        }
        Callee callee = resolveCallee(call.func);
        const std::string& function_name = callee.name;
        if (function_name == "not" && !callee.binding) {
            uint64_t binding_id = 0;
            bool true_is_error = false;
            if (conditionRefinement(node, &binding_id, &true_is_error)) {
                const auto* value = &call.variables[0];
                if (value->type == ESHKOL_VAR) allowed_resource_use_nodes_.insert(value);
                return;
            }
        }
        if (function_name.empty()) {
            if (hasTrackedResourceBinding())
                reportResourceError("call through a computed procedure while handles are open; the "
                    "checker cannot see its effect");
            else if (call.func && expressionContainsCallableContract(call.func))
                reportResourceError("call through a computed procedure that may carry a handle contract");
            recordRaisePoint("an exception raised by a computed call");
            return;
        }
        switch (callee.effect) {
            case CallableEffect::CONTRACT:
                if (callee.contract) handleContractCall(node, *callee.contract, function_name, work);
                checkPlainCallArguments(node, function_name);
                break;
            case CallableEffect::USER_FUNCTION:
                recordRaisePoint("an exception raised inside '" + function_name + "'");
                handleUserCall(node, callee, work);
                break;
            case CallableEffect::CONTINUATION:
                checkPlainCallArguments(node, function_name);
                handleContinuationCall(node, callee);
                break;
            case CallableEffect::LOOP:
                handleLoopCall(node, callee, work);
                break;
            case CallableEffect::TERMINAL_CONTROL: {
                WorkItem after = WorkItem::marker(WorkItem::Kind::TERMINAL_AFTER, node, 0);
                after.name = function_name;
                after.flag = callee.binding ? callee.binding->is_exit : function_name == "exit";
                work.push_back(after);
                break;
            }
            case CallableEffect::UNKNOWN_RESOURCE_EFFECT:
                reportResourceError("untrusted FFI or raw resource operation '" + function_name +
                    "' is unsupported in strict mode" +
                    (callee.binding && !callee.binding->unknown_reason.empty()
                         ? " (" + callee.binding->unknown_reason + ")" : std::string()));
                break;
            case CallableEffect::NONE:
                if (callee.binding && !callee.binding->is_proven_callable &&
                    !callee.binding->is_verified_stdlib_primitive) {
                    reportResourceError("callable value '" + function_name +
                        "' has no proved resource-effect provenance");
                    break;
                }
                if (!callee.binding && isRawResourceBuiltin(function_name)) {
                    reportResourceError("untrusted FFI or raw resource operation '" + function_name +
                        "' is unsupported in strict mode");
                    break;
                }
                if (!callee.binding && !isKnownResourceSafeBuiltin(function_name)) {
                    reportResourceError("builtin '" + function_name +
                        "' has no strict resource effect summary and is unsupported");
                    break;
                }
                checkPlainCallArguments(node, function_name);
                recordRaisePoint("an exception raised by " + function_name);
                if (fn_ && fn_->return_nodes.count(node)) fn_->returned_value = true;
                break;
        }
        if (!dead_ && (hasHandlerFrame() || handler_depth_ > 0) &&
            callee.effect == CallableEffect::CONTRACT && callee.contract && callee.contract->may_raise) {
            WorkItem after = WorkItem::marker(WorkItem::Kind::CALL_AFTER, node, 0);
            after.name = function_name;
            work.push_back(after);
        }
    }

    // ── scopes and lookup ───────────────────────────────────────────────
    void pushScope(const std::string& name = "") {
        Scope scope;
        scope.name = name;
        scope.id = next_scope_id_++;
        scope_stack_.push_back(std::move(scope));
    }

    void popScope() {
        if (!scope_stack_.empty()) {
            if (alias_tracking_)
                for (const auto& [name, info] : scope_stack_.back().variables)
                    scoped_out_[info.binding_id] = {name, info.state, info.is_owned_binding, info.reaches};
            while (!body_markers_.empty() && body_markers_.back() + 1 >= scope_stack_.size())
                body_markers_.pop_back();
            scope_stack_.pop_back();
        }
    }

    Scope& currentScope() {
        return scope_stack_.back();
    }

    // Look up variable in all scopes
    VariableInfo* lookupVariable(const std::string& name) {
        for (auto it = scope_stack_.rbegin(); it != scope_stack_.rend(); ++it) {
            auto var_it = it->variables.find(name);
            if (var_it != it->variables.end()) {
                return &var_it->second;
            }
        }
        return nullptr;
    }

    VariableInfo* lookupVariableUpTo(const std::string& name, size_t depth) {
        const size_t top = std::min(depth + 1, scope_stack_.size());
        for (size_t i = top; i > 0; --i) {
            auto var_it = scope_stack_[i - 1].variables.find(name);
            if (var_it != scope_stack_[i - 1].variables.end()) return &var_it->second;
        }
        return nullptr;
    }

    VariableInfo* findBindingById(uint64_t id, std::string* name = nullptr) {
        for (auto scope = scope_stack_.rbegin(); scope != scope_stack_.rend(); ++scope)
            for (auto& entry : scope->variables)
                if (entry.second.binding_id == id) {
                    if (name) *name = entry.first;
                    return &entry.second;
                }
        return nullptr;
    }

    int findScopeIndexById(uint64_t id) const {
        for (size_t i = scope_stack_.size(); i > 0; --i)
            if (scope_stack_[i - 1].id == id) return static_cast<int>(i - 1);
        return -1;
    }

    uint64_t next_binding_id_ = 1;
    uint64_t next_scope_id_ = 1;
    uint64_t next_frame_id_ = 1;
    std::map<uint64_t, size_t> borrowed_counts_;
    uint64_t next_branch_id_ = 1;
    std::map<uint64_t, BranchFrame> branch_frames_;
    std::set<const eshkol_ast_t*> accepted_factory_nodes_;
    std::set<const eshkol_ast_t*> accepted_move_nodes_;
    struct LoanResult {
        uint64_t owner;
        std::set<const eshkol_ast_t*> expressions;
    };
    std::vector<LoanResult> loan_results_;
    std::set<const eshkol_ast_t*> allowed_resource_use_nodes_;
    std::map<const eshkol_ast_t*, VariableInfo> parallel_binding_info_;
    std::set<const eshkol_ast_t*> call_position_nodes_;
    std::set<const eshkol_ast_t*> proc_arg_nodes_;
    std::set<const eshkol_ast_t*> alias_value_nodes_;
    std::set<const eshkol_ast_t*> named_lambda_nodes_;
    std::set<const eshkol_ast_t*> let_init_nodes_;
    std::map<const eshkol_ast_t*, InlineLambda> inline_lambdas_;
    std::vector<std::map<uint64_t, ResourceState>> inline_pre_states_;
    std::map<uint64_t, PendingCall> pending_procs_;
    std::map<const eshkol_ast_t*, uint64_t> pending_bind_;
    std::map<const eshkol_ast_t*, std::pair<ResourceKind, ResourceState>> last_call_result_;
    std::vector<Frame> frames_;
    std::map<uint64_t, Frame> closed_frames_;
    std::map<uint64_t, Snapshot> saved_snapshots_;
    std::map<std::string, Summary> summary_memo_;
    std::map<std::string, InProgress> in_progress_;
    std::vector<bool> memoizable_stack_;
    std::map<const eshkol_ast_t*, FnClassResult> classification_memo_;
    std::set<const eshkol_ast_t*> classification_in_progress_;
    std::vector<std::string> call_chain_;
    FunctionContext* fn_ = nullptr;
    bool dead_ = false;
    int handler_depth_ = 0;
    uint64_t callable_epoch_ = 0;
    bool capture_scan_truncated_ = false;
    std::set<std::string> captured_builtin_contracts_;
    bool active_resource_effects_enabled_ = true;
    WorkItem astChild(const eshkol_ast_t* node) const {
        return WorkItem::astNode(node, active_resource_effects_enabled_);
    }
    static hott::BorrowState borrowState(const VariableInfo& binding) {
        if (binding.state == State::MOVED || binding.state == State::MAYBE_MOVED)
            return hott::BorrowState::Moved;
        return hott::BorrowState::Owned;
    }

    bool isBorrowed(uint64_t binding_id) const {
        auto it = borrowed_counts_.find(binding_id);
        return it != borrowed_counts_.end() && it->second > 0;
    }

    void reportError(const std::string& msg) {
        auto display = eshkol_syntax_source_text(msg);
        if (!reported_.insert(display).second) return;
        errors_.push_back(std::move(display));
        has_errors_ = true;
    }

    // Check scope exit for unconsumed owned values
    void checkScopeExit() {
        if (scope_stack_.empty()) return;

        const auto& scope = currentScope();

        // Global scope owned values don't require consumption (they live until program exit)
        if (scope.name == "global") return;

        for (const auto& [name, info] : scope.variables) {
            if ((info.state == State::OWNED || info.state == State::MAYBE_MOVED) &&
                info.is_owned_binding) {
                reportError("Owned value '" + name + "' not consumed before scope exit" +
                           (info.defined_at.empty() ? "" : " (defined at " + info.defined_at + ")"));
            }
        }
    }

    // Get variable name from AST
    std::string getVarName(const eshkol_ast_t* ast) const {
        if (ast && ast->type == ESHKOL_VAR && ast->variable.id) {
            return ast->variable.id;
        }
        return "";
    }

    // ── branching ───────────────────────────────────────────────────────
    void startBranch(const eshkol_ast_t* condition, std::vector<WorkItem> true_items,
                     std::vector<WorkItem> false_items, bool effects,
                     std::vector<WorkItem>& work) {
        const uint64_t branch_id = next_branch_id_++;
        BranchFrame frame;
        frame.condition = condition;
        frame.resource_effects_enabled = effects;
        frame.true_items = std::move(true_items);
        frame.false_items = std::move(false_items);
        branch_frames_[branch_id] = std::move(frame);
        uint64_t binding_id = 0;
        bool true_is_error = false;
        if (effects && condition && condition->type == ESHKOL_VAR &&
            conditionRefinement(condition, &binding_id, &true_is_error))
            allowed_resource_use_nodes_.insert(condition);
        work.push_back(WorkItem::branchContinuation(WorkItem::Kind::IF_AFTER_CONDITION, nullptr,
                                                    branch_id, effects));
        if (condition) work.push_back(WorkItem::astNode(condition, effects));
    }

    static void pushItems(std::vector<WorkItem>& work, const std::vector<WorkItem>& items) {
        for (size_t i = items.size(); i > 0; --i) work.push_back(items[i - 1]);
    }

    std::vector<WorkItem> astItems(const eshkol_ast_t* nodes, uint64_t count, bool effects,
                                   uint64_t first = 0) const {
        std::vector<WorkItem> items;
        for (uint64_t i = first; i < count; ++i) items.push_back(WorkItem::astNode(&nodes[i], effects));
        return items;
    }

    // Bind a named-let or do loop and start its body.
    void bindLoopName(const std::string& name, uint64_t frame_id) {
        VariableInfo info{State::UNOWNED, name, false, next_binding_id_++};
        info.callable_effect = CallableEffect::LOOP;
        info.frame_id = frame_id;
        info.is_proven_callable = true;
        info.is_defined = true;
        currentScope().variables[name] = std::move(info);
    }

    void analyzeAST(const eshkol_ast_t* ast, bool resource_effects_enabled = true) {
        if (!ast) return;

        // Heap-backed records preserve post-child actions without recursive calls.
        std::vector<WorkItem> work;
        work.push_back(WorkItem::astNode(ast, strict_resources_ && resource_effects_enabled));

        while (!work.empty()) {
            WorkItem item = std::move(work.back());
            work.pop_back();
            active_resource_effects_enabled_ = item.resource_effects_enabled;

            switch (item.kind) {
                case WorkItem::Kind::AST:
                    break;
                case WorkItem::Kind::EXIT_SCOPE:
                    checkScopeExit();
                    if (strict_resources_ && item.resource_effects_enabled)
                        checkResourceScopeExit();
                    popScope();
                    continue;
                case WorkItem::Kind::UNBORROW: {
                    loan_results_.pop_back();
                    auto borrow = borrowed_counts_.find(item.binding_id);
                    if (borrow != borrowed_counts_.end() && --borrow->second == 0)
                        borrowed_counts_.erase(borrow);
                    continue;
                }
                case WorkItem::Kind::IF_AFTER_CONDITION: {
                    auto frame_it = branch_frames_.find(item.branch_id);
                    if (frame_it == branch_frames_.end()) continue;
                    auto& frame = frame_it->second;
                    frame.base = snapshot();
                    if (item.resource_effects_enabled) refineBranchCondition(frame.condition, true);
                    work.push_back(WorkItem::branchContinuation(
                        WorkItem::Kind::IF_AFTER_TRUE, nullptr, item.branch_id,
                        item.resource_effects_enabled));
                    pushItems(work, frame.true_items);
                    continue;
                }
                case WorkItem::Kind::IF_AFTER_TRUE: {
                    auto frame_it = branch_frames_.find(item.branch_id);
                    if (frame_it == branch_frames_.end()) continue;
                    auto& frame = frame_it->second;
                    frame.true_state = snapshot();
                    restore(frame.base);
                    if (item.resource_effects_enabled) refineBranchCondition(frame.condition, false);
                    work.push_back(WorkItem::branchContinuation(
                        WorkItem::Kind::IF_AFTER_FALSE, nullptr, item.branch_id,
                        item.resource_effects_enabled));
                    pushItems(work, frame.false_items);
                    continue;
                }
                case WorkItem::Kind::IF_AFTER_FALSE: {
                    auto frame_it = branch_frames_.find(item.branch_id);
                    if (frame_it != branch_frames_.end()) {
                        Snapshot joined = joinSnapshots(frame_it->second.true_state, snapshot());
                        restore(joined);
                        branch_frames_.erase(frame_it);
                    }
                    continue;
                }
                case WorkItem::Kind::COND_CLAUSE: {
                    if (item.index >= item.count) {
                        if (item.flag) {
                            // no guard clause matched: the exception is raised again
                            recordRaisePoint("the exception re-raised by guard");
                            dead_ = true;
                        }
                        continue;
                    }
                    const eshkol_ast_t* clause = &item.clauses[item.index];
                    if (clause->type != ESHKOL_OP || clause->operation.op != ESHKOL_CALL_OP) {
                        reportResourceError("malformed cond clause in the checked fragment");
                        continue;
                    }
                    const auto& parts = clause->operation.call_op;
                    const std::string head = getVarName(parts.func);
                    const bool is_else = head == "else" || isTrueLiteral(parts.func);
                    if (is_else) {
                        pushItems(work, astItems(parts.variables, parts.num_vars, item.resource_effects_enabled));
                        continue;
                    }
                    if (parts.num_vars == 2 && getVarName(&parts.variables[0]) == "=>") {
                        reportResourceError("cond => clauses are not followed by the strict resource "
                            "checker; bind the test value with let and use if");
                        continue;
                    }
                    std::vector<WorkItem> rest;
                    rest.push_back(WorkItem::condClause(item.clauses, item.count, item.index + 1,
                                                        item.flag, item.resource_effects_enabled));
                    startBranch(parts.func,
                                astItems(parts.variables, parts.num_vars, item.resource_effects_enabled),
                                std::move(rest), item.resource_effects_enabled, work);
                    continue;
                }
                case WorkItem::Kind::LOOP_END: {
                    if (!frames_.empty() && frames_.back().id == item.branch_id) frames_.pop_back();
                    continue;
                }
                case WorkItem::Kind::LOOP_BACKEDGE: {
                    Frame* frame = findFrame(item.branch_id);
                    if (frame) checkBackEdge(*frame, "loop '" + frame->loop_name + "'");
                    dead_ = true;
                    continue;
                }
                case WorkItem::Kind::DO_INSTALL: {
                    const eshkol_operations_t* op = item.op;
                    std::vector<const eshkol_ast_t*> bindings;
                    const eshkol_ast_t* test = nullptr;
                    const eshkol_ast_t* results = nullptr;
                    decomposeDo(*op, &bindings, &test, &results);
                    std::vector<VariableInfo> infos;
                    for (const auto* binding : bindings) {
                        VariableInfo info{State::UNOWNED, "", false, 0};
                        if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                            binding->cons_cell.cdr->type == ESHKOL_CONS)
                            bindResourceInitializer(info, binding->cons_cell.cdr->cons_cell.car,
                                                    getVarName(binding->cons_cell.car), currentScope().id);
                        infos.push_back(info);
                    }
                    pushScope("do");
                    for (size_t i = 0; i < bindings.size(); ++i) {
                        const std::string name = getVarName(bindings[i]->cons_cell.car);
                        if (name.empty()) continue;
                        VariableInfo info = infos[i];
                        info.binding_id = next_binding_id_++;
                        info.is_defined = true;
                        currentScope().variables[name] = std::move(info);
                    }
                    Frame frame = makeLoopFrame(FrameKind::DO, "do", 0);
                    const uint64_t id = frame.id;
                    frames_.push_back(std::move(frame));
                    work.push_back(WorkItem::marker(WorkItem::Kind::DO_AFTER_TEST, nullptr, id, true));
                    work.back().op = op;
                    if (test) work.push_back(WorkItem::astNode(test, true));
                    continue;
                }
                case WorkItem::Kind::DO_AFTER_TEST: {
                    const eshkol_operations_t* op = item.op;
                    std::vector<const eshkol_ast_t*> bindings;
                    const eshkol_ast_t* test = nullptr;
                    const eshkol_ast_t* results = nullptr;
                    decomposeDo(*op, &bindings, &test, &results);
                    saved_snapshots_[item.branch_id] = snapshot();
                    WorkItem iteration = WorkItem::marker(WorkItem::Kind::DO_AFTER_ITERATION, nullptr,
                                                          item.branch_id, true);
                    iteration.op = op;
                    work.push_back(iteration);
                    for (size_t i = bindings.size(); i > 0; --i) {
                        const auto* binding = bindings[i - 1];
                        if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                            binding->cons_cell.cdr->type == ESHKOL_CONS) {
                            const eshkol_ast_t* step = binding->cons_cell.cdr->cons_cell.cdr;
                            if (step && step->type == ESHKOL_VAR &&
                                getVarName(step) == getVarName(binding->cons_cell.car))
                                continue;  // no step
                            VariableInfo* var = lookupVariable(getVarName(binding->cons_cell.car));
                            if (var && expressionProducesResourceValue(step))
                                reportResourceError("do variable '" + getVarName(binding->cons_cell.car) +
                                    "' cannot carry a handle between iterations");
                            work.push_back(WorkItem::astNode(step, true));
                        }
                    }
                    pushItems(work, astItems(op->call_op.variables, op->call_op.num_vars, true));
                    continue;
                }
                case WorkItem::Kind::DO_AFTER_ITERATION: {
                    Frame* frame = findFrame(item.branch_id);
                    if (frame) checkBackEdge(*frame, "do loop");
                    restore(saved_snapshots_[item.branch_id]);
                    saved_snapshots_.erase(item.branch_id);
                    const eshkol_operations_t* op = item.op;
                    std::vector<const eshkol_ast_t*> bindings;
                    const eshkol_ast_t* test = nullptr;
                    const eshkol_ast_t* results = nullptr;
                    decomposeDo(*op, &bindings, &test, &results);
                    work.push_back(WorkItem::marker(WorkItem::Kind::DO_EXIT, nullptr, item.branch_id, true));
                    if (results && results->type == ESHKOL_OP && results->operation.op == ESHKOL_CALL_OP)
                        pushItems(work, astItems(results->operation.call_op.variables,
                                                 results->operation.call_op.num_vars, true));
                    continue;
                }
                case WorkItem::Kind::DO_EXIT:
                    if (!frames_.empty() && frames_.back().id == item.branch_id) frames_.pop_back();
                    checkScopeExit();
                    checkResourceScopeExit();
                    popScope();
                    continue;
                case WorkItem::Kind::GUARD_AFTER_BODY: {
                    const eshkol_operations_t* op = item.op;
                    if (frames_.empty() || frames_.back().id != item.branch_id) continue;
                    Frame frame = std::move(frames_.back());
                    frames_.pop_back();
                    if (!frame.any) continue;  // nothing in the body can raise
                    saved_snapshots_[item.branch_id] = snapshot();
                    restore(frame.exits);
                    pushScope("guard handler");
                    if (op->guard_op.var_name)
                        currentScope().variables[op->guard_op.var_name] =
                            VariableInfo{State::UNOWNED, op->guard_op.var_name, false, next_binding_id_++};
                    work.push_back(WorkItem::marker(WorkItem::Kind::GUARD_JOIN, nullptr, item.branch_id, true));
                    work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, op, 0, true));
                    work.push_back(WorkItem::condClause(op->guard_op.clauses, op->guard_op.num_clauses,
                                                        0, true, true));
                    continue;
                }
                case WorkItem::Kind::GUARD_JOIN: {
                    Snapshot normal = saved_snapshots_[item.branch_id];
                    saved_snapshots_.erase(item.branch_id);
                    restore(joinSnapshots(normal, snapshot()));
                    continue;
                }
                case WorkItem::Kind::WIND_ENTER: {
                    Frame frame;
                    frame.kind = FrameKind::WIND;
                    frame.id = item.branch_id;
                    frame.depth = scope_stack_.size();
                    frames_.push_back(std::move(frame));
                    continue;
                }
                case WorkItem::Kind::WIND_AFTER_BODY: {
                    if (!frames_.empty() && frames_.back().id == item.branch_id) {
                        closed_frames_[item.branch_id] = std::move(frames_.back());
                        frames_.pop_back();
                    }
                    continue;
                }
                case WorkItem::Kind::WIND_AFTER_RAISES: {
                    auto closed = closed_frames_.find(item.branch_id);
                    if (closed == closed_frames_.end() || !closed->second.any) {
                        closed_frames_.erase(item.branch_id);
                        continue;
                    }
                    // Run the after thunk again from the state of every
                    // exception or escape that leaves the body.
                    saved_snapshots_[item.branch_id] = snapshot();
                    restore(closed->second.exits);
                    work.push_back(WorkItem::marker(WorkItem::Kind::WIND_FORWARD, nullptr, item.branch_id, true));
                    work.push_back(WorkItem::astNode(item.ast, true));
                    continue;
                }
                case WorkItem::Kind::WIND_FORWARD: {
                    Frame frame = std::move(closed_frames_[item.branch_id]);
                    closed_frames_.erase(item.branch_id);
                    Snapshot after_exit = snapshot();
                    restore(saved_snapshots_[item.branch_id]);
                    saved_snapshots_.erase(item.branch_id);
                    if (after_exit.dead) continue;
                    if (frame.raised) {
                        Snapshot current = snapshot();
                        restore(after_exit);
                        recordRaisePoint("an exception leaving dynamic-wind");
                        restore(current);
                    }
                    for (uint64_t target : frame.escape_targets)
                        routeEscape(after_exit, target, "a continuation escape through dynamic-wind");
                    continue;
                }
                case WorkItem::Kind::CALLCC_JOIN: {
                    if (frames_.empty() || frames_.back().id != item.branch_id) continue;
                    Frame frame = std::move(frames_.back());
                    frames_.pop_back();
                    if (frame.any) restore(joinSnapshots(snapshot(), frame.exits));
                    continue;
                }
                case WorkItem::Kind::INLINE_END: {
                    auto info_it = inline_lambdas_.find(item.ast);
                    const InlineLambda info = info_it == inline_lambdas_.end() ? InlineLambda{} : info_it->second;
                    if (item.binding_id) {
                        auto borrow = borrowed_counts_.find(item.binding_id);
                        if (borrow != borrowed_counts_.end() && --borrow->second == 0)
                            borrowed_counts_.erase(borrow);
                    }
                    if ((info.mode == sr::ProcMode::MANY || info.mode == sr::ProcMode::THREAD) &&
                        !inline_pre_states_.empty()) {
                        const auto before = inline_pre_states_.back();
                        inline_pre_states_.pop_back();
                        for (const auto& entry : before) {
                            std::string name;
                            VariableInfo* current = findBindingById(entry.first, &name);
                            if (current && current->resource_state != entry.second)
                                reportResourceError("a procedure passed to " + info.lender +
                                    " runs zero or more times but changes handle '" + name + "' from " +
                                    stateText(entry.second) + " to " + stateText(current->resource_state) +
                                    "; it must leave every handle as it found it");
                        }
                    }
                    checkScopeExit();
                    checkResourceScopeExit();
                    popScope();
                    continue;
                }
                case WorkItem::Kind::CALL_AFTER: {
                    if (item.flag) {
                        VariableInfo* owner = findBindingById(item.branch_id);
                        if (owner) {
                            owner->resource_state = ResourceState::CLOSED;
                            owner->state = State::MOVED;
                        }
                    } else {
                        recordRaisePoint("an exception raised by " + item.name);
                    }
                    continue;
                }
                case WorkItem::Kind::USER_CALL_APPLY:
                    applyPendingUserCall(item);
                    continue;
                case WorkItem::Kind::TERMINAL_AFTER:
                    if (item.flag) {
                        if (fn_ && !dead_) fn_->may_exit = true;
                        if (!dead_)
                            for (const auto& scope : scope_stack_)
                                for (const auto& entry : scope.variables)
                                    if (entry.second.resource_kind != ResourceKind::NONE &&
                                        isLiveResourceState(entry.second.resource_state))
                                        reportResourceError("terminal control transfer '" + item.name +
                                            "' occurs while resource '" + entry.first + "' may still be live");
                        dead_ = true;
                    } else {
                        explicitRaise("'" + item.name + "'");
                    }
                    continue;
                case WorkItem::Kind::RAISE_AFTER:
                    explicitRaise("raise");
                    continue;
                case WorkItem::Kind::POP_FRAME:
                    if (!frames_.empty() && frames_.back().id == item.branch_id) frames_.pop_back();
                    continue;
                case WorkItem::Kind::LET_PARALLEL_INIT: {
                    if (item.index == item.op->let_op.num_bindings) {
                        work.push_back(WorkItem::continuation(WorkItem::Kind::LET_PARALLEL_INSTALL, item.op, 0, item.resource_effects_enabled));
                        continue;
                    }
                    const eshkol_ast_t* binding = &item.op->let_op.bindings[item.index];
                    work.push_back(WorkItem::continuation(WorkItem::Kind::LET_PARALLEL_INIT, item.op, item.index + 1, item.resource_effects_enabled));
                    if (binding->type == ESHKOL_CONS) {
                        checkCopy(binding->cons_cell.cdr);
                        if (strict_resources_ && item.resource_effects_enabled) {
                            acceptDirectFactoryInitializer(binding->cons_cell.cdr);
                            let_init_nodes_.insert(binding->cons_cell.cdr);
                            std::string name = getVarName(binding->cons_cell.car);
                            VariableInfo inferred{State::UNOWNED, "", false, 0};
                            const uint64_t env_scope = scope_stack_.size() > 1
                                ? scope_stack_[scope_stack_.size() - 2].id : currentScope().id;
                            bindResourceInitializer(inferred, binding->cons_cell.cdr, name, env_scope);
                            parallel_binding_info_[binding] = std::move(inferred);
                        }
                        work.push_back(astChild(binding->cons_cell.cdr));
                    }
                    continue;
                }
                case WorkItem::Kind::LET_SEQUENTIAL_INSTALL: {
                    const eshkol_ast_t* binding = &item.op->let_op.bindings[item.index];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (!var_name.empty()) {
                            bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                            VariableInfo info{is_owned ? State::OWNED : State::UNOWNED,
                                              "", is_owned, next_binding_id_++};
                            if (strict_resources_ && item.resource_effects_enabled)
                                bindResourceInitializer(info, binding->cons_cell.cdr, var_name,
                                                        currentScope().id);
                            bindValueFacts(info, binding->cons_cell.cdr);
                            info.is_defined = true;
                            currentScope().variables[var_name] = std::move(info);
                        }
                    }
                    continue;
                }
                case WorkItem::Kind::LET_PARALLEL_INSTALL: {
                    for (uint64_t i = 0; i < item.op->let_op.num_bindings; ++i) {
                        const eshkol_ast_t* binding = &item.op->let_op.bindings[i];
                        if (binding->type != ESHKOL_CONS) continue;
                        std::string var_name = getVarName(binding->cons_cell.car);
                        if (var_name.empty()) continue;
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        VariableInfo info{is_owned ? State::OWNED : State::UNOWNED,
                                          "", is_owned, next_binding_id_++};
                        if (strict_resources_ && item.resource_effects_enabled) {
                            auto cached = parallel_binding_info_.find(binding);
                            if (cached != parallel_binding_info_.end()) {
                                const uint64_t binding_id = info.binding_id;
                                info = cached->second;
                                info.binding_id = binding_id;
                                info.state = is_owned ? State::OWNED : State::UNOWNED;
                                info.is_owned_binding = is_owned;
                                parallel_binding_info_.erase(cached);
                            }
                            auto result = last_call_result_.find(binding->cons_cell.cdr);
                            if (result != last_call_result_.end() && result->second.first != ResourceKind::NONE) {
                                info.resource_kind = result->second.first;
                                info.resource_state = result->second.second;
                                info.resource_alias = var_name;
                            }
                        }
                        bindValueFacts(info, binding->cons_cell.cdr);
                        info.is_defined = true;
                        currentScope().variables[var_name] = std::move(info);
                    }
                    work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, item.op, 0, item.resource_effects_enabled));
                    if (strict_resources_ && item.resource_effects_enabled && item.op->let_op.name) {
                        // Named let: the loop name is bound inside the body
                        // and tail calls to it are back edges.
                        Frame frame = makeLoopFrame(FrameKind::LOOP, item.op->let_op.name, 0);
                        collectTailNodes(item.op->let_op.body, frame.tail_calls, true);
                        bindLoopName(item.op->let_op.name, frame.id);
                        frame.loop_binding_id = currentScope().variables[item.op->let_op.name].binding_id;
                        const uint64_t id = frame.id;
                        frames_.push_back(std::move(frame));
                        work.push_back(WorkItem::marker(WorkItem::Kind::LOOP_END, nullptr, id, true));
                    }
                    bindLoopFacts(item.op);
                    work.push_back(astChild(item.op->let_op.body));
                    continue;
                }
                case WorkItem::Kind::LET_BINDING: {
                    if (item.index == item.op->let_op.num_bindings) {
                        work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, item.op, 0, item.resource_effects_enabled));
                        work.push_back(astChild(item.op->let_op.body));
                        continue;
                    }
                    const eshkol_ast_t* binding = &item.op->let_op.bindings[item.index];
                    work.push_back(WorkItem::continuation(
                        WorkItem::Kind::LET_BINDING, item.op, item.index + 1, item.resource_effects_enabled));
                    if (binding->type == ESHKOL_CONS) {
                        checkCopy(binding->cons_cell.cdr);
                        if (strict_resources_ && item.resource_effects_enabled) {
                            acceptDirectFactoryInitializer(binding->cons_cell.cdr);
                            let_init_nodes_.insert(binding->cons_cell.cdr);
                        }
                        work.push_back(WorkItem::continuation(
                            WorkItem::Kind::LET_SEQUENTIAL_INSTALL, item.op, item.index, item.resource_effects_enabled));
                        work.push_back(astChild(binding->cons_cell.cdr));
                    }
                    continue;
                }
            }

            if (!item.ast) continue;
            const eshkol_ast_t* node = item.ast;
            const bool checked = strict_resources_ && item.resource_effects_enabled;
            if (checked && fn_ && fn_->return_nodes.count(node) && !dead_ &&
                node->type != ESHKOL_VAR &&
                !(node->type == ESHKOL_OP && node->operation.op == ESHKOL_CALL_OP))
                fn_->returned_value = true;
            if (!follow_branches_ && node->type == ESHKOL_OP &&
                (node->operation.op == ESHKOL_OWNED_OP || node->operation.op == ESHKOL_MOVE_OP ||
                 node->operation.op == ESHKOL_BORROW_OP)) {
                needs_branch_analysis_ = true;
                return;
            }
            if (follow_branches_ && isIfForm(node)) {
                const auto& call = node->operation.call_op;
                std::vector<WorkItem> false_items;
                if (call.num_vars == 3) false_items.push_back(astChild(&call.variables[2]));
                startBranch(&call.variables[0], {astChild(&call.variables[1])},
                            std::move(false_items), item.resource_effects_enabled, work);
                continue;
            }
            if (checked)
                processResourceCall(node, work);
            switch (node->type) {
                case ESHKOL_VAR: {
                    std::string name = getVarName(node);
                    if (!name.empty()) {
                        VariableInfo* info = lookupVariable(name);
                        if (info && (info->state == State::MOVED ||
                                     info->state == State::MAYBE_MOVED) &&
                            !(strict_resources_ && allowed_resource_use_nodes_.count(node)) &&
                            !(checked && info->resource_kind != ResourceKind::NONE)) {
                            reportError("Use of moved value '" + name + "'");
                        }
                        if (info) {
                            for (const auto& loan : loan_results_) {
                                if (loan.owner == info->binding_id && loan.expressions.count(node))
                                    reportError("Borrowed value '" + name + "' escapes its borrow scope");
                            }
                            checkAliasUse(name, *info);
                        }
                        if (checked && info) checkVariableUse(node, name, *info);
                    }
                    break;
                }

                case ESHKOL_CONS:
                    work.push_back(astChild(node->cons_cell.cdr));
                    work.push_back(astChild(node->cons_cell.car));
                    break;

                case ESHKOL_OP:
                    analyzeOperation(node, item, work);
                    break;

                default:
                    break;
            }
        }
    }

    void checkVariableUse(const eshkol_ast_t* node, const std::string& name, VariableInfo& info) {
        if (info.resource_kind != ResourceKind::NONE) {
            if (allowed_resource_use_nodes_.count(node)) return;
            if (fn_ && fn_->return_nodes.count(node)) {
                if (fn_->param_ids.count(info.binding_id)) {
                    reportResourceError("function '" + fn_->name + "' returns its handle parameter '" +
                        name + "'; return a handle the function acquires, or close it");
                } else if (info.lent || isBorrowed(info.binding_id)) {
                    reportResourceError("borrowed resource '" + name + "' escapes its borrow scope "
                        "through a return value");
                } else if (info.resource_state == ResourceState::CLOSED ||
                           info.resource_state == ResourceState::CLOSED_OR_ERROR ||
                           info.resource_state == ResourceState::MOVED ||
                           info.resource_state == ResourceState::MAYBE_LIVE) {
                    reportResourceError("function '" + fn_->name + "' returns resource '" + name +
                        "', which is " + stateText(info.resource_state));
                } else {
                    recordReturnedHandle(info.resource_kind, info.resource_state);
                    info.moved_from_state = info.resource_state;
                    info.resource_state = ResourceState::MOVED;
                }
                return;
            }
            if (info.lent || isBorrowed(info.binding_id)) {
                reportResourceError("borrowed resource '" + name + "' escapes its borrow scope; a "
                    "borrowed handle may only be passed to operations that use it");
                return;
            }
            reportResourceError("resource value '" + name +
                "' is used outside a recognized owner, observer, close, move, or borrow operation");
            return;
        }
        if (fn_ && fn_->return_nodes.count(node) && !dead_) fn_->returned_value = true;
        const bool called = call_position_nodes_.count(node) != 0;
        if (called || proc_arg_nodes_.count(node) || alias_value_nodes_.count(node)) return;
        if (info.callable_effect == CallableEffect::CONTINUATION) {
            if (hasTrackedResourceBinding())
                reportResourceError("continuation '" + name + "' escapes as a value while handles are "
                    "open; the checker cannot follow a re-entry into their scope");
        } else if (info.callable_effect == CallableEffect::LOOP) {
            if (hasTrackedResourceBinding())
                reportResourceError("loop '" + name + "' escapes as a value while handles are open; "
                    "the checker follows loops only through tail calls");
        } else if (info.callable_effect == CallableEffect::USER_FUNCTION && info.function_node) {
            FnClassResult cls = classifyBinding(info);
            if (cls.cls == FnClass::RELEVANT)
                reportResourceError("function '" + name + "' uses handles or handle operations and "
                    "escapes as a value; the checker cannot see where it is called");
        }
    }

    void defineFunctionInfo(VariableInfo& info, const eshkol_ast_t* node, const std::string& name) {
        if (const sr::Contract* contract = trustedResourceContract(node)) {
            info.callable_effect = CallableEffect::CONTRACT;
            info.contract = contract;
            info.is_proven_callable = true;
            return;
        }
        (void)name;
        info.callable_effect = CallableEffect::USER_FUNCTION;
        info.function_node = node;
        info.function_scope_id = currentScope().id;
        info.is_proven_callable = true;
    }

    // An inline lambda runs here: exactly once (a borrow scope, dynamic-wind,
    // an immediately applied lambda) or zero or more times (for-each, a
    // handler). A lent parameter is a loan the lambda may not close or keep.
    void startInlineLambda(const eshkol_ast_t* node, const InlineLambda& info,
                           std::vector<WorkItem>& work) {
        const auto& lambda = node->operation.lambda_op;
        if (info.mode == sr::ProcMode::THREAD) {
            std::vector<std::set<std::string>> lexical(1);
            for (uint64_t i = 0; i < lambda.num_params; ++i)
                lexical.back().insert(getVarName(&lambda.parameters[i]));
            std::set<uint64_t> captures;
            collectCapturedResourceIds(lambda.body, lexical, captures);
            for (uint64_t id : captures) {
                std::string name;
                VariableInfo* captured = findBindingById(id, &name);
                if (captured && captured->resource_kind != ResourceKind::NONE)
                    reportResourceError("handle '" + name + "' is captured by a thread body; handles "
                        "cannot cross a thread boundary in the checked fragment");
            }
        }
        std::map<uint64_t, ResourceState> before;
        if (info.mode == sr::ProcMode::MANY || info.mode == sr::ProcMode::THREAD) {
            for (const auto& scope : scope_stack_)
                for (const auto& entry : scope.variables)
                    if (entry.second.resource_kind != ResourceKind::NONE)
                        before[entry.second.binding_id] = entry.second.resource_state;
            inline_pre_states_.push_back(std::move(before));
        }
        pushScope("inline lambda");
        uint64_t lent_id = 0;
        for (uint64_t i = 0; i < lambda.num_params; ++i) {
            const std::string param = getVarName(&lambda.parameters[i]);
            if (param.empty()) continue;
            VariableInfo var{State::UNOWNED, param, false, next_binding_id_++};
            var.is_defined = true;
            if (i == 0 && info.lend_kind != ResourceKind::NONE) {
                var.resource_kind = info.lend_kind;
                var.resource_state = ResourceState::LIVE;
                var.resource_alias = param;
                var.lent = true;
                lent_id = var.binding_id;
                ++borrowed_counts_[lent_id];
            }
            currentScope().variables[param] = std::move(var);
        }
        if (lambda.rest_param)
            currentScope().variables[lambda.rest_param] =
                VariableInfo{State::UNOWNED, lambda.rest_param, false, next_binding_id_++};
        WorkItem end = WorkItem::marker(WorkItem::Kind::INLINE_END, node, 0, true);
        end.binding_id = lent_id;
        work.push_back(end);
        work.push_back(WorkItem::astNode(lambda.body, true));
    }

    void analyzeOperation(const eshkol_ast_t* node, const WorkItem& item,
                          std::vector<WorkItem>& work) {
        const eshkol_operations_t* op = &node->operation;
        const bool checked = strict_resources_ && item.resource_effects_enabled;
        switch (op->op) {
            case ESHKOL_DEFINE_OP: {
                std::string name = op->define_op.name ? op->define_op.name : "";
                if (name.empty()) break;
                if (!op->define_op.is_function) checkCopy(op->define_op.value);
                bool is_owned = !op->define_op.is_function && transfersOwnership(op->define_op.value);
                VariableInfo info{is_owned ? State::OWNED : State::UNOWNED,
                                  name, is_owned, next_binding_id_++};
                auto existing = currentScope().variables.find(name);
                VariableInfo previous;
                bool had_previous = false;
                if (existing != currentScope().variables.end()) {
                    if (checked && existing->second.is_defined &&
                        existing->second.resource_kind != ResourceKind::NONE &&
                        isLiveResourceState(existing->second.resource_state)) {
                        reportResourceError("resource '" + name +
                            "' is overwritten before it is closed");
                    }
                    info.binding_id = existing->second.binding_id;
                    previous = existing->second;
                    had_previous = true;
                } else if (scope_stack_.size() == 1) {
                    auto predeclared = predeclared_.find(name);
                    if (predeclared != predeclared_.end()) info.binding_id = predeclared->second;
                }
                if (op->define_op.is_function) bindFunctionFacts(info, node);
                else bindValueFacts(info, op->define_op.value);
                if (checked) {
                    info.is_defined = true;
                    if (const sr::Contract* stdlib_contract = verifiedStdlibContract(node)) {
                        info.callable_effect = CallableEffect::CONTRACT;
                        info.contract = stdlib_contract;
                        info.is_proven_callable = true;
                    } else if (op->define_op.is_external) {
                        if (isVerifiedStdlibScalarPrimitive(node)) {
                            info.is_proven_callable = true;
                            info.is_verified_stdlib_primitive = true;
                        } else if (op->define_op.is_function) {
                            info.callable_effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                            info.unknown_reason = "'" + name + "' is an external definition";
                            info.is_proven_callable = true;
                        } else if (!isStaticDataExpression(op->define_op.value)) {
                            info.callable_effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                            reportResourceError("opaque external initializer for '" +
                                name + "' is unsupported in strict mode");
                        }
                    } else if (op->define_op.is_function) {
                        defineFunctionInfo(info, node, name);
                    } else {
                        bindResourceInitializer(info, op->define_op.value, name, currentScope().id);
                        if (op->define_op.value && op->define_op.value->type == ESHKOL_OP &&
                            op->define_op.value->operation.op == ESHKOL_CALL_OP &&
                            isUserFunctionCall(op->define_op.value))
                            pending_bind_[op->define_op.value] = info.binding_id;
                    }
                    if (had_previous && (previous.callable_effect != info.callable_effect ||
                                         previous.contract != info.contract ||
                                         previous.function_node != info.function_node))
                        ++callable_epoch_;
                }
                currentScope().variables[name] = std::move(info);
                if (checked)
                    acceptDirectFactoryInitializer(op->define_op.value);
                if (op->define_op.is_function) {
                    // The body is checked at each call site with the caller's
                    // handle states; here only the ownership forms are walked.
                    pushScope("function");
                    enterBody();
                    for (uint64_t i = 0; i < op->define_op.num_params; ++i) {
                        const std::string param_name = getVarName(&op->define_op.parameters[i]);
                        if (!param_name.empty()) {
                            currentScope().variables[param_name] = {
                                State::UNOWNED, param_name, false, next_binding_id_++
                            };
                        }
                    }
                    if (alias_tracking_ && op->define_op.rest_param)
                        currentScope().variables[op->define_op.rest_param] = {
                            State::UNOWNED, op->define_op.rest_param, false, next_binding_id_++};
                    work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, op, 0, false));
                    work.push_back(WorkItem::astNode(op->define_op.value, false));
                } else {
                    work.push_back(astChild(op->define_op.value));
                }
                break;
            }

            case ESHKOL_OWNED_OP:
                work.push_back(astChild(op->owned_op.value));
                break;

            case ESHKOL_MOVE_OP: {
                std::string var_name = getVarName(op->move_op.value);
                if (alias_tracking_ && boundOutsideCurrentBody(var_name))
                    reportError("Cannot move '" + var_name + "' inside a procedure or loop body that may run "
                                "more than once; it is bound outside that body");
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info) {
                        auto issue = hott::BorrowChecker::moveViolation(borrowState(*info),
                                                                        isBorrowed(info->binding_id));
                        if (!issue) {
                            info->state = State::MOVED;
                        } else if (*issue == hott::BorrowChecker::BorrowError::Kind::MoveWhileBorrowed) {
                            reportError("Cannot move '" + var_name + "' while it is borrowed");
                        } else if (info->state == State::MAYBE_MOVED) {
                            reportError("Move of maybe-moved value '" + var_name + "'");
                        } else if (!(checked && info->resource_kind != ResourceKind::NONE)) {
                            reportError("Double move of '" + var_name + "' - value already moved");
                        }
                        if (checked && info->resource_kind != ResourceKind::NONE) {
                            if (!accepted_move_nodes_.count(node)) {
                                reportResourceError("resource move from '" + var_name +
                                    "' must initialize a new tracked owner");
                            } else if (info->resource_state == ResourceState::CLOSED ||
                                info->resource_state == ResourceState::CLOSED_OR_ERROR ||
                                info->resource_state == ResourceState::MOVED) {
                                reportResourceError("cannot move closed or already moved resource '" + var_name + "'");
                            } else if (isBorrowed(info->binding_id) || info->lent) {
                                reportResourceError("cannot move borrowed resource '" + var_name + "'");
                            } else {
                                info->moved_from_state = info->resource_state;
                                info->resource_state = ResourceState::MOVED;
                                allowed_resource_use_nodes_.insert(op->move_op.value);
                            }
                        }
                    }
                } else {
                    work.push_back(astChild(op->move_op.value));
                }
                break;
            }

            case ESHKOL_BORROW_OP: {
                std::string var_name = getVarName(op->borrow_op.value);
                uint64_t borrowed_binding_id = 0;
                if (!var_name.empty()) {
                    VariableInfo* info = lookupVariable(var_name);
                    if (info && hott::BorrowChecker::sharedBorrowViolation(borrowState(*info), false) &&
                        !(checked && info->resource_kind != ResourceKind::NONE)) {
                        reportError("Cannot borrow moved value '" + var_name + "'");
                    } else if (info) {
                        if (checked && info->resource_kind != ResourceKind::NONE) {
                            if (info->resource_state != ResourceState::LIVE)
                                reportResourceError("resource '" + var_name +
                                    "' must be on a successful acquisition path before borrow (it is " +
                                    stateText(info->resource_state) + ")");
                            allowed_resource_use_nodes_.insert(op->borrow_op.value);
                        }
                        borrowed_binding_id = info->binding_id;
                        ++borrowed_counts_[borrowed_binding_id];
                    }
                }
                if (borrowed_binding_id) {
                    work.push_back(WorkItem::unborrow(borrowed_binding_id));
                    loan_results_.push_back({borrowed_binding_id, {}});
                    if (op->borrow_op.num_body_exprs)
                        collectTailNodes(&op->borrow_op.body[op->borrow_op.num_body_exprs - 1],
                                         loan_results_.back().expressions, false);
                }
                for (uint64_t i = op->borrow_op.num_body_exprs; i > 0; --i)
                    work.push_back(astChild(&op->borrow_op.body[i - 1]));
                if (var_name.empty()) work.push_back(astChild(op->borrow_op.value));
                break;
            }

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
                pushScope("let");
                work.push_back(WorkItem::continuation(
                    op->op == ESHKOL_LET_OP ? WorkItem::Kind::LET_PARALLEL_INIT
                                           : WorkItem::Kind::LET_BINDING, op, 0, item.resource_effects_enabled));
                break;

            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP: {
                pushScope("letrec");
                for (uint64_t i = 0; i < op->let_op.num_bindings; ++i) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type == ESHKOL_CONS) {
                        std::string var_name = getVarName(binding->cons_cell.car);
                        bool is_owned = transfersOwnership(binding->cons_cell.cdr);
                        if (!var_name.empty()) {
                            currentScope().variables[var_name] = {
                                is_owned ? State::OWNED : State::UNOWNED, "", is_owned, next_binding_id_++
                            };
                            currentScope().variables[var_name].is_defined = true;
                        }
                    }
                }
                for (uint64_t i = 0; record_procs_ && i < op->let_op.num_bindings; ++i) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i];
                    if (binding->type != ESHKOL_CONS) continue;
                    auto bound = currentScope().variables.find(getVarName(binding->cons_cell.car));
                    if (bound != currentScope().variables.end())
                        bindValueFacts(bound->second, binding->cons_cell.cdr);
                }
                if (checked) {
                    for (uint64_t i = 0; i < op->let_op.num_bindings; ++i) {
                        const eshkol_ast_t* binding = &op->let_op.bindings[i];
                        if (binding->type != ESHKOL_CONS) continue;
                        const std::string var_name = getVarName(binding->cons_cell.car);
                        VariableInfo* info = lookupVariable(var_name);
                        if (!info) continue;
                        bindResourceInitializer(*info, binding->cons_cell.cdr, var_name, currentScope().id);
                        acceptDirectFactoryInitializer(binding->cons_cell.cdr);
                        if (isUserFunctionCall(binding->cons_cell.cdr))
                            pending_bind_[binding->cons_cell.cdr] = info->binding_id;
                    }
                }
                work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, op, 0, item.resource_effects_enabled));
                work.push_back(astChild(op->let_op.body));
                for (uint64_t i = op->let_op.num_bindings; i > 0; --i) {
                    const eshkol_ast_t* binding = &op->let_op.bindings[i - 1];
                    if (binding->type == ESHKOL_CONS) {
                        work.push_back(astChild(binding->cons_cell.cdr));
                    }
                }
                break;
            }

            case ESHKOL_LAMBDA_OP: {
                auto inline_it = checked ? inline_lambdas_.find(node) : inline_lambdas_.end();
                if (inline_it != inline_lambdas_.end()) {
                    startInlineLambda(node, inline_it->second, work);
                    break;
                }
                if (checked && !named_lambda_nodes_.count(node))
                    checkLambdaResourceCapture(node);
                if (checked && fn_ && fn_->return_nodes.count(node) && !dead_)
                    fn_->returned_value = true;
                if (alias_tracking_) {
                    for (const auto& loan : loan_results_) {
                        if (!loan.expressions.count(node)) continue;
                        Reach captured;
                        mentionsOf(node, captured);
                        std::string borrowed;
                        if (captured.count(loan.owner) && findBindingById(loan.owner, &borrowed))
                            reportError("Borrowed value '" + borrowed + "' escapes its borrow scope");
                    }
                }
                pushScope("lambda");
                ParamReach params;
                if (alias_tracking_) {
                    auto known = lambda_param_reach_.find(node);
                    if (known != lambda_param_reach_.end()) params = known->second;
                    if (!immediate_lambdas_.count(node)) enterBody();
                }
                for (uint64_t i = 0; i < op->lambda_op.num_params; ++i) {
                    std::string param_name = getVarName(&op->lambda_op.parameters[i]);
                    if (!param_name.empty()) {
                        VariableInfo param{State::UNOWNED, "", false, next_binding_id_++};
                        param.reaches = i < params.positional.size() ? params.positional[i] : params.all;
                        currentScope().variables[param_name] = std::move(param);
                    }
                }
                if (alias_tracking_ && op->lambda_op.rest_param) {
                    VariableInfo rest{State::UNOWNED, "", false, next_binding_id_++};
                    rest.reaches = params.all;
                    for (size_t i = op->lambda_op.num_params; i < params.positional.size(); ++i)
                        rest.reaches.insert(params.positional[i].begin(), params.positional[i].end());
                    currentScope().variables[op->lambda_op.rest_param] = std::move(rest);
                }
                work.push_back(WorkItem::continuation(
                    WorkItem::Kind::EXIT_SCOPE, op, 0,
                    strict_resources_ ? false : item.resource_effects_enabled));
                work.push_back(WorkItem::astNode(op->lambda_op.body,
                    strict_resources_ ? false : item.resource_effects_enabled));
                break;
            }

            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
                if (checked && expressionHasResourceTransition(node))
                    reportResourceError("resource transitions under and/or are unsupported in the checked fragment");
                for (uint64_t i = op->sequence_op.num_expressions; i > 0; --i)
                    work.push_back(astChild(&op->sequence_op.expressions[i - 1]));
                break;

            case ESHKOL_COND_OP:
                if (checked) {
                    work.push_back(WorkItem::condClause(op->call_op.variables, op->call_op.num_vars,
                                                        0, false, true));
                    break;
                }
                for (uint64_t i = op->call_op.num_vars; i > 0; --i) {
                    clause_nodes_.insert(&op->call_op.variables[i - 1]);
                    work.push_back(astChild(&op->call_op.variables[i - 1]));
                }
                break;

            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
                if (!follow_branches_) {
                    pushItems(work, astItems(op->call_op.variables, op->call_op.num_vars, false));
                } else if (op->call_op.num_vars >= 1) {
                    auto body = astItems(op->call_op.variables, op->call_op.num_vars, checked, 1);
                    if (op->op == ESHKOL_WHEN_OP)
                        startBranch(&op->call_op.variables[0], std::move(body), {}, checked, work);
                    else
                        startBranch(&op->call_op.variables[0], {}, std::move(body), checked, work);
                    break;
                }
                break;

            case ESHKOL_CALL_OP: {
                if (alias_tracking_ && !clause_nodes_.count(node)) checkCallFlow(node);
                const bool inline_callee = checked && op->call_op.func &&
                    inline_lambdas_.count(op->call_op.func) != 0;
                if (inline_callee) {
                    // Arguments are evaluated before the body runs.
                    work.push_back(astChild(op->call_op.func));
                    for (uint64_t i = op->call_op.num_vars; i > 0; --i)
                        work.push_back(astChild(&op->call_op.variables[i - 1]));
                    break;
                }
                // An applied lambda runs after its arguments are evaluated.
                if (isLambdaNode(op->call_op.func)) work.push_back(astChild(op->call_op.func));
                for (uint64_t i = op->call_op.num_vars; i > 0; --i) {
                    work.push_back(astChild(&op->call_op.variables[i - 1]));
                }
                if (!isLambdaNode(op->call_op.func)) work.push_back(astChild(op->call_op.func));
                break;
            }

            case ESHKOL_ADD_OP:
            case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP:
            case ESHKOL_DIV_OP:
            case ESHKOL_IF_OP:
                for (uint64_t i = op->call_op.num_vars; i > 0; --i)
                    work.push_back(astChild(&op->call_op.variables[i - 1]));
                break;

            case ESHKOL_SEQUENCE_OP:
                for (uint64_t i = op->sequence_op.num_expressions; i > 0; --i) {
                    work.push_back(astChild(&op->sequence_op.expressions[i - 1]));
                }
                break;

            case ESHKOL_WITH_REGION_OP:
                pushScope("region");
                work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, op, 0, item.resource_effects_enabled));
                for (uint64_t i = op->with_region_op.num_body_exprs; i > 0; --i) {
                    work.push_back(astChild(&op->with_region_op.body[i - 1]));
                }
                break;

            case ESHKOL_SHARED_OP:
                work.push_back(astChild(op->shared_op.value));
                break;

            case ESHKOL_WEAK_REF_OP:
                work.push_back(astChild(op->weak_ref_op.value));
                break;

            case ESHKOL_THE_OP:
                work.push_back(astChild(op->the_op.expr));
                break;

            case ESHKOL_SET_OP:
                checkCopy(op->set_op.value);
                if (alias_tracking_ && op->set_op.name) {
                    Reach assigned;
                    reachOf(op->set_op.value, assigned);
                    const auto owners = reachedOwners(assigned);
                    if (!owners.empty())
                        reportError("Assignment to '" + std::string(op->set_op.name) +
                                    "' would keep a reference to owned value '" + owners.front().name +
                                    "'; transfer it with move");
                    if (VariableInfo* target = lookupVariable(op->set_op.name)) {
                        target->reaches.insert(assigned.begin(), assigned.end());
                        target->proc_node = nullptr;
                        target->summary = nullptr;
                        target->loop_proc = false;
                        target->external = false;
                        target->data = target->data && neverProcedure(op->set_op.value);
                    }
                }
                if (op->set_op.name) {
                    auto* binding = lookupVariable(op->set_op.name);
                    if (binding && isBorrowed(binding->binding_id))
                        reportError("Cannot assign to borrowed value '" + std::string(op->set_op.name) + "'");
                }
                if (checked && op->set_op.name) {
                    VariableInfo* target = lookupVariable(op->set_op.name);
                    if (target) {
                        if (target->resource_kind != ResourceKind::NONE) {
                            reportResourceError("set! cannot overwrite resource owner '" +
                                std::string(op->set_op.name) + "'");
                            target->resource_state = ResourceState::MAYBE_LIVE;
                        }
                        if (target->callable_effect != CallableEffect::NONE ||
                            target->is_proven_callable ||
                            callableEffectOf(op->set_op.value) != CallableEffect::NONE) {
                            target->callable_effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                            target->contract = nullptr;
                            target->function_node = nullptr;
                            target->is_proven_callable = false;
                            target->unknown_reason = "'" + std::string(op->set_op.name) +
                                "' is reassigned with set!";
                            ++callable_epoch_;
                        }
                    }
                    if (expressionProducesResourceValue(op->set_op.value))
                        reportResourceError("resource value escapes through set! of '" +
                            std::string(op->set_op.name) + "'");
                }
                work.push_back(astChild(op->set_op.value));
                break;

            case ESHKOL_EXTERN_OP: {
                const std::string name = op->extern_op.name ? op->extern_op.name : "";
                if (!name.empty()) {
                    VariableInfo info{State::UNOWNED, name, false, next_binding_id_++};
                    auto existing = currentScope().variables.find(name);
                    if (existing != currentScope().variables.end())
                        info.binding_id = existing->second.binding_id;
                    if (const sr::Contract* stdlib_contract = verifiedStdlibContract(node)) {
                        info.callable_effect = CallableEffect::CONTRACT;
                        info.contract = stdlib_contract;
                        info.is_proven_callable = true;
                    } else if (isVerifiedStdlibScalarPrimitive(node)) {
                        info.is_proven_callable = true;
                        info.is_verified_stdlib_primitive = true;
                    } else {
                        info.callable_effect = CallableEffect::UNKNOWN_RESOURCE_EFFECT;
                        info.unknown_reason = "'" + name + "' is a foreign function";
                        info.is_proven_callable = true;
                    }
                    info.is_defined = true;
                    currentScope().variables[name] = std::move(info);
                }
                break;
            }

            case ESHKOL_EXTERN_VAR_OP:
                if (checked)
                    reportResourceError("untrusted extern variable is unsupported in the checked fragment");
                break;

            case ESHKOL_DO_OP: {
                if (!checked) {
                    refuseUnfollowed(node);
                    break;
                }
                std::vector<const eshkol_ast_t*> bindings;
                const eshkol_ast_t* test = nullptr;
                const eshkol_ast_t* results = nullptr;
                decomposeDo(*op, &bindings, &test, &results);
                work.push_back(WorkItem::continuation(WorkItem::Kind::DO_INSTALL, op, 0, true));
                for (size_t i = bindings.size(); i > 0; --i) {
                    const auto* binding = bindings[i - 1];
                    if (binding->type == ESHKOL_CONS && binding->cons_cell.cdr &&
                        binding->cons_cell.cdr->type == ESHKOL_CONS) {
                        acceptDirectFactoryInitializer(binding->cons_cell.cdr->cons_cell.car);
                        work.push_back(WorkItem::astNode(binding->cons_cell.cdr->cons_cell.car, true));
                    }
                }
                break;
            }

            case ESHKOL_GUARD_OP: {
                if (!checked) {
                    refuseUnfollowed(node);
                    break;
                }
                Frame frame;
                frame.kind = FrameKind::GUARD;
                frame.id = next_frame_id_++;
                frame.depth = scope_stack_.size();
                const uint64_t id = frame.id;
                frames_.push_back(std::move(frame));
                WorkItem after = WorkItem::continuation(WorkItem::Kind::GUARD_AFTER_BODY, op, 0, true);
                after.branch_id = id;
                work.push_back(after);
                pushItems(work, astItems(op->guard_op.body, op->guard_op.num_body_exprs, true));
                break;
            }

            case ESHKOL_RAISE_OP:
                if (!checked) {
                    refuseUnfollowed(node);
                    break;
                }
                if (alias_tracking_) {
                    const std::string raised = ownerReachedBy(op->raise_op.exception);
                    if (!raised.empty())
                        reportError("Raised value refers to owned value '" + raised +
                                    "'; a handler the checker cannot see may keep it");
                }
                work.push_back(WorkItem::marker(WorkItem::Kind::RAISE_AFTER, node, 0, true));
                work.push_back(astChild(op->raise_op.exception));
                break;

            case ESHKOL_CALL_CC_OP: {
                const eshkol_ast_t* proc = op->call_cc_op.proc;
                if (!checked) {
                    refuseUnfollowed(node);
                    break;
                }
                if (!proc || proc->type != ESHKOL_OP || proc->operation.op != ESHKOL_LAMBDA_OP ||
                    proc->operation.lambda_op.num_params < 1) {
                    if (hasTrackedResourceBinding())
                        reportResourceError("call/cc with a computed procedure while handles are open; "
                            "the checker cannot see where the continuation goes");
                    work.push_back(astChild(proc));
                    break;
                }
                Frame frame;
                frame.kind = FrameKind::CALLCC;
                frame.id = next_frame_id_++;
                frame.depth = scope_stack_.size();
                const uint64_t id = frame.id;
                frames_.push_back(std::move(frame));
                work.push_back(WorkItem::marker(WorkItem::Kind::CALLCC_JOIN, nullptr, id, true));
                work.push_back(WorkItem::continuation(WorkItem::Kind::EXIT_SCOPE, op, 0, true));
                pushScope("call/cc");
                const auto& lambda = proc->operation.lambda_op;
                const std::string k = getVarName(&lambda.parameters[0]);
                VariableInfo continuation{State::UNOWNED, k, false, next_binding_id_++};
                continuation.callable_effect = CallableEffect::CONTINUATION;
                continuation.frame_id = id;
                continuation.is_proven_callable = true;
                continuation.is_defined = true;
                if (!k.empty()) currentScope().variables[k] = std::move(continuation);
                work.push_back(WorkItem::astNode(lambda.body, true));
                break;
            }

            case ESHKOL_DYNAMIC_WIND_OP: {
                if (!checked) {
                    refuseUnfollowed(node);
                    break;
                }
                const auto& wind = op->dynamic_wind_op;
                auto thunk = [](const eshkol_ast_t* p) {
                    return p && p->type == ESHKOL_OP && p->operation.op == ESHKOL_LAMBDA_OP &&
                           p->operation.lambda_op.num_params == 0;
                };
                if (!thunk(wind.before) || !thunk(wind.thunk) || !thunk(wind.after)) {
                    if (hasTrackedResourceBinding())
                        reportResourceError("dynamic-wind with computed thunks while handles are open; "
                            "the checker cannot see the cleanup");
                    work.push_back(astChild(wind.after));
                    work.push_back(astChild(wind.thunk));
                    work.push_back(astChild(wind.before));
                    break;
                }
                InlineLambda once;
                once.mode = sr::ProcMode::ONCE;
                inline_lambdas_[wind.before] = once;
                inline_lambdas_[wind.thunk] = once;
                inline_lambdas_[wind.after] = once;
                const uint64_t id = next_frame_id_++;
                work.push_back(WorkItem::marker(WorkItem::Kind::WIND_AFTER_RAISES, wind.after, id, true));
                work.push_back(WorkItem::astNode(wind.after, true));
                work.push_back(WorkItem::marker(WorkItem::Kind::WIND_AFTER_BODY, nullptr, id, true));
                work.push_back(WorkItem::astNode(wind.thunk, true));
                work.push_back(WorkItem::marker(WorkItem::Kind::WIND_ENTER, nullptr, id, true));
                work.push_back(WorkItem::astNode(wind.before, true));
                break;
            }

            case ESHKOL_VALUES_OP:
            case ESHKOL_CALL_WITH_VALUES_OP:
                refuseUnfollowed(node);
                if (checked)
                    reportResourceError("multiple-value resource flow is unsupported in the checked fragment");
                break;

            case ESHKOL_QUASIQUOTE_OP:
            case ESHKOL_UNQUOTE_OP:
            case ESHKOL_UNQUOTE_SPLICING_OP:
                refuseUnfollowed(node);
                if (checked)
                    reportResourceError("quasiquote/unquote evaluation is unsupported in the checked fragment");
                break;

            case ESHKOL_CASE_OP:
            case ESHKOL_MATCH_OP:
                refuseUnfollowed(node);
                if (checked)
                    reportResourceError("case and match are not followed by the strict resource checker; "
                        "use cond or if for resource flow");
                break;

            case ESHKOL_DEFINE_TYPE_OP:
            case ESHKOL_IMPORT_OP:
            case ESHKOL_REQUIRE_OP:
            case ESHKOL_PROVIDE_OP:
            case ESHKOL_DEFINE_SYNTAX_OP:
            case ESHKOL_QUOTE_OP:
            case ESHKOL_TYPE_ANNOTATION_OP:
                break;
            default:
                refuseUnfollowed(node);
                if (checked)
                    reportResourceError("operator form " +
                        std::to_string(static_cast<int>(op->op)) +
                        " has no strict resource traversal summary");
                break;
        }
    }

    // ── alias and escape checking ───────────────────────────────────────
    // A unit that uses owned, move or borrow records, for each binding, the
    // bindings its value may refer to. Reading a binding that refers to a
    // moved owner is refused. A place the checker cannot follow -- an
    // assignment, an argument a procedure keeps, a procedure the checker
    // cannot resolve, an operator it does not model -- refuses any value
    // that refers to an owner.
    using Reach = std::set<uint64_t>;
    struct ScopedOut {
        std::string name;
        State state = State::UNOWNED;
        bool owner = false;
        Reach reaches;
    };
    struct ParamReach {
        std::vector<Reach> positional;
        Reach all;
    };
    struct BindingView {
        std::string name;
        State state = State::UNOWNED;
        bool owner = false;
        const Reach* reaches = nullptr;
    };
    bool alias_tracking_ = false;
    bool record_procs_ = false;                   // keep procedure facts for a session
    std::map<uint64_t, ScopedOut> scoped_out_;
    std::map<std::string, uint64_t> predeclared_;
    std::map<uint64_t, std::string> predeclared_names_;
    std::set<uint64_t> future_owners_;
    std::map<std::string, const eshkol_ast_t*> global_procs_;
    std::set<std::string> global_names_;
    std::set<std::string> external_names_;
    std::vector<size_t> body_markers_;
    std::vector<std::set<std::string>> shadows_;  // binder names inside a value being measured
    std::set<const eshkol_ast_t*> immediate_lambdas_;
    std::set<const eshkol_ast_t*> clause_nodes_;
    std::map<const eshkol_ast_t*, ParamReach> lambda_param_reach_;
    std::map<const eshkol_ast_t*, ProcSummary> proc_table_;
    std::set<const eshkol_ast_t*> proc_final_;

    static bool isLambdaNode(const eshkol_ast_t* e) {
        return e && e->type == ESHKOL_OP && e->operation.op == ESHKOL_LAMBDA_OP;
    }

    // Top-level names, the procedures they are bound to once, and owners
    // defined later in the unit, so a procedure can refer to them earlier.
    void prepareAliasTracking(const std::vector<eshkol_ast_t>& asts, const OwnershipBindings* globals) {
        alias_tracking_ = false;
        bool active = false;
        if (globals)
            for (const auto& entry : *globals)
                active = active || entry.second.owner || !entry.second.reaches.empty();
        for (size_t i = 0; !active && i < asts.size(); ++i) active = containsOwnershipForm(&asts[i]);
        if (!globals && (!active || !follow_branches_)) return;
        record_procs_ = true;
        if (globals)
            for (const auto& entry : *globals)
                if (!entry.second.scoped_out) global_names_.insert(entry.first);
        std::map<std::string, int> definitions;
        std::set<std::string> assigned;
        std::vector<const eshkol_ast_t*> pending;
        for (const auto& ast : asts) pending.push_back(&ast);
        while (!pending.empty()) {
            const eshkol_ast_t* form = pending.back();
            pending.pop_back();
            if (!form || form->type != ESHKOL_OP) continue;
            if (form->operation.op == ESHKOL_SEQUENCE_OP) {
                for (uint64_t i = 0; i < form->operation.sequence_op.num_expressions; ++i)
                    pending.push_back(&form->operation.sequence_op.expressions[i]);
                continue;
            }
            if (form->operation.op != ESHKOL_DEFINE_OP || !form->operation.define_op.name) continue;
            const auto& define = form->operation.define_op;
            const std::string name = define.name;
            ++definitions[name];
            global_names_.insert(name);
            if (define.is_external) external_names_.insert(name);
            else if (define.is_function) global_procs_[name] = form;
            else if (isLambdaNode(define.value)) global_procs_[name] = define.value;
            if (!currentScope().variables.count(name) && !predeclared_.count(name)) {
                const uint64_t id = next_binding_id_++;
                predeclared_[name] = id;
                predeclared_names_[id] = name;
            }
            auto pre = predeclared_.find(name);
            if (pre != predeclared_.end() && !define.is_function && transfersOwnership(define.value))
                future_owners_.insert(pre->second);
        }
        std::vector<const eshkol_ast_t*> walk;
        for (const auto& ast : asts) walk.push_back(&ast);
        while (!walk.empty()) {
            const eshkol_ast_t* node = walk.back();
            walk.pop_back();
            if (!node) continue;
            if (node->type == ESHKOL_OP && node->operation.op == ESHKOL_SET_OP && node->operation.set_op.name)
                assigned.insert(node->operation.set_op.name);
            forEachChildExpr(node, [&](const eshkol_ast_t* child) { walk.push_back(child); });
        }
        for (const auto& [name, count] : definitions)
            if (count > 1 || assigned.count(name)) global_procs_.erase(name);
        for (const auto& name : assigned)
            if (VariableInfo* session = lookupVariable(name)) session->summary = nullptr;
        alias_tracking_ = active && follow_branches_;
    }

    bool shadowed(const std::string& name) const {
        for (const auto& names : shadows_)
            if (names.count(name)) return true;
        return false;
    }

    uint64_t resolveName(const std::string& name) {
        if (name.empty() || shadowed(name)) return 0;
        if (VariableInfo* info = lookupVariable(name)) return info->binding_id;
        auto pre = predeclared_.find(name);
        return pre == predeclared_.end() ? 0 : pre->second;
    }

    // A name no binding of the unit, the session or an enclosing scope
    // defines, or an external declaration: a runtime builtin.
    bool isBuiltinName(const std::string& name) {
        if (name.empty() || shadowed(name)) return false;
        if (external_names_.count(name)) return true;
        return !resolveName(name) && !global_names_.count(name);
    }

    bool viewBinding(uint64_t id, BindingView& view) {
        static const Reach none;
        std::string name;
        if (VariableInfo* info = findBindingById(id, &name)) {
            view = {name, info->state, info->is_owned_binding || future_owners_.count(id) != 0,
                    &info->reaches};
            return true;
        }
        auto out = scoped_out_.find(id);
        if (out != scoped_out_.end()) {
            view = {out->second.name, out->second.state, out->second.owner, &out->second.reaches};
            return true;
        }
        auto pre = predeclared_names_.find(id);
        if (pre != predeclared_names_.end()) {
            view = {pre->second, State::UNOWNED, future_owners_.count(id) != 0, &none};
            return true;
        }
        return false;
    }

    // Owners a value may refer to through any chain of bindings.
    std::vector<BindingView> reachedOwners(const Reach& seeds, uint64_t self = 0) {
        std::vector<BindingView> owners;
        std::set<uint64_t> seen;
        std::vector<uint64_t> pending(seeds.begin(), seeds.end());
        while (!pending.empty()) {
            const uint64_t id = pending.back();
            pending.pop_back();
            if (!seen.insert(id).second) continue;
            BindingView view;
            if (!viewBinding(id, view)) continue;
            if (view.owner && id != self) owners.push_back(view);
            for (uint64_t next : *view.reaches) pending.push_back(next);
        }
        return owners;
    }

    void allVisibleOwners(Reach& out) const {
        for (const auto& scope : scope_stack_)
            for (const auto& entry : scope.variables)
                if (entry.second.is_owned_binding) out.insert(entry.second.binding_id);
        for (const auto& entry : scoped_out_)
            if (entry.second.owner) out.insert(entry.first);
        out.insert(future_owners_.begin(), future_owners_.end());
    }

    // Visit the children of a binding form with its bound names shadowing
    // the enclosing scopes, so a parameter or local never stands for an
    // outer binding of the same spelling. Returns false for other forms.
    template <class Visit>
    bool visitBinder(const eshkol_ast_t* e, Visit&& visit) {
        if (!e || e->type != ESHKOL_OP) return false;
        const auto& op = e->operation;
        auto shadow = [&](const std::vector<std::string>& names, const eshkol_ast_t* body) {
            shadows_.emplace_back(names.begin(), names.end());
            visit(body);
            shadows_.pop_back();
        };
        auto params = [&](const eshkol_ast_t* list, uint64_t count, const char* rest) {
            std::vector<std::string> names;
            for (uint64_t i = 0; list && i < count; ++i) names.push_back(getVarName(&list[i]));
            if (rest) names.push_back(rest);
            return names;
        };
        if (op.op == ESHKOL_LAMBDA_OP) {
            shadow(params(op.lambda_op.parameters, op.lambda_op.num_params, op.lambda_op.rest_param),
                   op.lambda_op.body);
            return true;
        }
        if (op.op == ESHKOL_DEFINE_OP && op.define_op.is_function) {
            shadow(params(op.define_op.parameters, op.define_op.num_params, op.define_op.rest_param),
                   op.define_op.value);
            return true;
        }
        if (op.op == ESHKOL_LET_OP || op.op == ESHKOL_LET_STAR_OP || op.op == ESHKOL_LETREC_OP ||
            op.op == ESHKOL_LETREC_STAR_OP) {
            std::vector<std::string> names;
            if (op.let_op.name) names.push_back(op.let_op.name);
            for (uint64_t i = 0; op.let_op.bindings && i < op.let_op.num_bindings; ++i) {
                const eshkol_ast_t& binding = op.let_op.bindings[i];
                if (binding.type != ESHKOL_CONS) continue;
                names.push_back(getVarName(binding.cons_cell.car));
                visit(binding.cons_cell.cdr);
            }
            shadow(names, op.let_op.body);
            return true;
        }
        return false;
    }

    // Bindings an expression's value may refer to: every binding it names,
    // except the operands of a builtin that returns a fresh scalar, string
    // or symbol. A procedure value refers to everything its body names.
    void reachOf(const eshkol_ast_t* e, Reach& out, unsigned depth = 0) {
        if (!e) return;
        if (depth > 2048) {
            allVisibleOwners(out);
            return;
        }
        if (e->type == ESHKOL_VAR) {
            if (const uint64_t id = resolveName(getVarName(e))) out.insert(id);
            return;
        }
        if (e->type == ESHKOL_OP) {
            const auto& op = e->operation;
            switch (op.op) {
                case ESHKOL_QUOTE_OP:
                case ESHKOL_SET_OP:
                case ESHKOL_ADD_OP:
                case ESHKOL_SUB_OP:
                case ESHKOL_MUL_OP:
                case ESHKOL_DIV_OP:
                    return;
                case ESHKOL_LAMBDA_OP:
                    mentionsOf(e, out, depth + 1);
                    return;
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    visitBinder(e, [&](const eshkol_ast_t* child) { reachOf(child, out, depth + 1); });
                    return;
                case ESHKOL_MOVE_OP:
                    if (op.move_op.value && op.move_op.value->type == ESHKOL_VAR) {
                        const std::string source = getVarName(op.move_op.value);
                        if (VariableInfo* moved = shadowed(source) ? nullptr : lookupVariable(source))
                            out.insert(moved->reaches.begin(), moved->reaches.end());
                        return;
                    }
                    break;
                case ESHKOL_CALL_OP:
                    if (!isIfForm(e) && op.call_op.func && op.call_op.func->type == ESHKOL_VAR) {
                        const std::string head = getVarName(op.call_op.func);
                        if (isScalarResultBuiltin(head) && isBuiltinName(head)) return;
                    }
                    break;
                default:
                    break;
            }
        }
        forEachChildExpr(e, [&](const eshkol_ast_t* child) { reachOf(child, out, depth + 1); });
    }

    // Every binding an expression names, wherever it names it.
    void mentionsOf(const eshkol_ast_t* e, Reach& out, unsigned depth = 0) {
        if (!e) return;
        if (depth > 2048) {
            allVisibleOwners(out);
            return;
        }
        if (e->type == ESHKOL_VAR) {
            if (const uint64_t id = resolveName(getVarName(e))) out.insert(id);
            return;
        }
        if (e->type == ESHKOL_OP && e->operation.op == ESHKOL_QUOTE_OP) return;
        auto visit = [&](const eshkol_ast_t* child) { mentionsOf(child, out, depth + 1); };
        if (visitBinder(e, visit)) return;
        forEachChildExpr(e, visit);
    }

    // A value that is never a procedure: a datum, a number, a fresh string
    // or a structure built by a builtin constructor.
    bool neverProcedure(const eshkol_ast_t* e) {
        if (!e) return true;
        if (e->type == ESHKOL_VAR) {
            const std::string name = getVarName(e);
            VariableInfo* info = shadowed(name) ? nullptr : lookupVariable(name);
            return info && info->data;
        }
        if (e->type != ESHKOL_OP) return true;
        const auto& op = e->operation;
        switch (op.op) {
            case ESHKOL_QUOTE_OP:
            case ESHKOL_ADD_OP:
            case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP:
            case ESHKOL_DIV_OP:
            case ESHKOL_TENSOR_OP:
                return true;
            case ESHKOL_OWNED_OP:
                return neverProcedure(op.owned_op.value);
            case ESHKOL_MOVE_OP:
                return neverProcedure(op.move_op.value);
            case ESHKOL_SHARED_OP:
                return neverProcedure(op.shared_op.value);
            case ESHKOL_THE_OP:
                return neverProcedure(op.the_op.expr);
            case ESHKOL_CALL_OP: {
                static const std::set<std::string> constructors = {
                    "cons", "list", "vector", "make-vector", "make-string", "string", "list-copy",
                    "vector-copy", "append", "reverse", "make-list", "iota", "make-hash-table",
                    "make-tensor", "zeros", "ones", "list->vector", "vector->list", "string->list"
                };
                if (isIfForm(e) || !op.call_op.func || op.call_op.func->type != ESHKOL_VAR) return false;
                const std::string head = getVarName(op.call_op.func);
                return isBuiltinName(head) && (isScalarResultBuiltin(head) || constructors.count(head));
            }
            default:
                return false;
        }
    }

    void bindValueFacts(VariableInfo& info, const eshkol_ast_t* value) {
        if (!record_procs_) return;
        info.reaches.clear();
        if (alias_tracking_) reachOf(value, info.reaches);
        info.reaches.erase(info.binding_id);
        info.proc_node = isLambdaNode(value) ? value : nullptr;
        info.summary = nullptr;
        info.loop_proc = false;
        info.external = false;
        info.data = neverProcedure(value);
        if (value && value->type == ESHKOL_VAR) {
            if (VariableInfo* source = lookupVariable(getVarName(value))) {
                info.proc_node = source->proc_node;
                info.summary = source->summary;
                info.external = source->external;
            } else if (isBuiltinName(getVarName(value))) {
                info.external = true;
            }
        }
    }

    void bindFunctionFacts(VariableInfo& info, const eshkol_ast_t* define_node) {
        if (!record_procs_) return;
        const auto& define = define_node->operation.define_op;
        info.reaches.clear();
        if (alias_tracking_) mentionsOf(define_node, info.reaches);
        info.reaches.erase(info.binding_id);
        info.proc_node = define.is_external ? nullptr : define_node;
        info.external = define.is_external != 0;
        info.summary = nullptr;
        info.loop_proc = false;
        info.data = false;
    }

    // Procedure bodies may run more than once. A move inside one may only
    // take an owner bound inside it.
    void enterBody() {
        if (alias_tracking_ && !scope_stack_.empty()) body_markers_.push_back(scope_stack_.size() - 1);
    }

    bool boundOutsideCurrentBody(const std::string& name) const {
        if (body_markers_.empty()) return false;
        for (size_t i = scope_stack_.size(); i > 0; --i)
            if (scope_stack_[i - 1].variables.count(name)) return i - 1 < body_markers_.back();
        return predeclared_.count(name) != 0;
    }

    void checkAliasUse(const std::string& name, const VariableInfo& info) {
        if (!alias_tracking_ || info.reaches.empty()) return;
        for (const auto& owner : reachedOwners(info.reaches, info.binding_id)) {
            if (owner.state == State::MOVED)
                reportError("Use of '" + name + "', which refers to moved value '" + owner.name + "'");
            else if (owner.state == State::MAYBE_MOVED)
                reportError("Use of '" + name + "', which may refer to moved value '" + owner.name + "'");
        }
    }

    std::string ownerReachedBy(const eshkol_ast_t* e) {
        Reach reach;
        reachOf(e, reach);
        const auto owners = reachedOwners(reach);
        return owners.empty() ? std::string() : owners.front().name;
    }

    bool touchesOwnership(const eshkol_ast_t* e, std::string& name, unsigned depth = 0) {
        if (!e) return false;
        if (depth > 2048) return true;
        if (e->type == ESHKOL_VAR) {
            const uint64_t id = resolveName(getVarName(e));
            if (!id) return false;
            const auto owners = reachedOwners(Reach{id});
            if (owners.empty()) return false;
            name = owners.front().name;
            return true;
        }
        if (e->type == ESHKOL_OP) {
            const auto op = e->operation.op;
            if (op == ESHKOL_QUOTE_OP) return false;
            if (op == ESHKOL_OWNED_OP || op == ESHKOL_MOVE_OP || op == ESHKOL_BORROW_OP) return true;
        }
        bool found = false;
        forEachChildExpr(e, [&](const eshkol_ast_t* child) {
            if (!found) found = touchesOwnership(child, name, depth + 1);
        });
        return found;
    }

    static const char* unfollowedFormName(eshkol_op_t op) {
        switch (op) {
            case ESHKOL_DO_OP: return "do loops";
            case ESHKOL_GUARD_OP: return "guard";
            case ESHKOL_RAISE_OP: return "raise";
            case ESHKOL_CALL_CC_OP: return "call/cc";
            case ESHKOL_DYNAMIC_WIND_OP: return "dynamic-wind";
            case ESHKOL_VALUES_OP: return "values";
            case ESHKOL_CALL_WITH_VALUES_OP: return "call-with-values";
            case ESHKOL_LET_VALUES_OP:
            case ESHKOL_LET_STAR_VALUES_OP: return "let-values";
            case ESHKOL_QUASIQUOTE_OP:
            case ESHKOL_UNQUOTE_OP:
            case ESHKOL_UNQUOTE_SPLICING_OP: return "quasiquote";
            case ESHKOL_CASE_OP: return "case";
            case ESHKOL_MATCH_OP: return "match";
            case ESHKOL_CASE_LAMBDA_OP: return "case-lambda";
            case ESHKOL_PARAMETERIZE_OP: return "parameterize";
            default: return "this operator";
        }
    }

    // A form the checker does not model may not involve an owner.
    void refuseUnfollowed(const eshkol_ast_t* node) {
        if (!alias_tracking_) return;
        std::string name;
        if (!touchesOwnership(node, name)) return;
        const std::string form = unfollowedFormName(node->operation.op);
        if (name.empty())
            reportError("The ownership checker does not follow " + form +
                        " that contains owned, move or borrow; bind the value with let and use if");
        else
            reportError("The ownership checker does not follow " + form + " that refers to owned value '" +
                        name + "'; bind the value with let and use if");
    }

    // ── procedure summaries ─────────────────────────────────────────────
    // What a procedure does with its parameters, flow-insensitively: which
    // it may keep past the call and which procedure parameters it calls with
    // which others. Mutually recursive procedures reach a fixed point from
    // "keeps nothing".
    class ProcScan {
    public:
        ProcScan(OwnershipAnalyzer& analyzer, ProcSummary& out, std::vector<const eshkol_ast_t*>& deps)
            : a_(analyzer), out_(out), deps_(deps) {}

        void run(const eshkol_ast_t* proc) {
            const FunctionShape shape = functionShape(proc);
            out_.num_params = shape.num_params;
            out_.rest = shape.rest != nullptr;
            for (uint64_t i = 0; i < shape.num_params; ++i)
                env_.push_back({a_.getVarName(&shape.params[i]), Tokens{static_cast<size_t>(i)}});
            if (shape.rest) env_.push_back({shape.rest, Tokens{static_cast<size_t>(shape.num_params)}});
            scan(shape.body);
        }

    private:
        using Tokens = std::set<size_t>;
        struct Entry {
            std::string name;
            Tokens tokens;
            const eshkol_ast_t* proc = nullptr;
            bool loop = false;
            size_t loop_vars = 0;
            size_t loop_count = 0;
        };
        struct LambdaFlow {
            Tokens captured;
            Tokens result;
        };
        enum class Head { LOOP, PROC, PARAM, BUILTIN, UNKNOWN };

        OwnershipAnalyzer& a_;
        ProcSummary& out_;
        std::vector<const eshkol_ast_t*>& deps_;
        std::vector<Entry> env_;
        std::vector<std::pair<size_t, Tokens>> frames_;
        unsigned depth_ = 0;

        static void add(Tokens& to, const Tokens& from) { to.insert(from.begin(), from.end()); }
        Tokens all() const {
            Tokens tokens;
            for (size_t i = 0; i <= out_.num_params; ++i) tokens.insert(i);
            return tokens;
        }
        void keep(const Tokens& tokens) { add(out_.keeps, tokens); }
        int find(const std::string& name) const {
            for (size_t i = env_.size(); i > 0; --i)
                if (env_[i - 1].name == name) return static_cast<int>(i - 1);
            return -1;
        }
        Tokens lookup(const std::string& name) {
            const int i = find(name);
            if (i < 0) return {};
            for (auto& frame : frames_)
                if (static_cast<size_t>(i) < frame.first) add(frame.second, env_[i].tokens);
            return env_[i].tokens;
        }
        const ProcSummary* globalSummary(const std::string& name) {
            auto proc = a_.global_procs_.find(name);
            if (proc != a_.global_procs_.end()) return &a_.summaryForScan(proc->second, deps_);
            if (a_.external_names_.count(name) || a_.scope_stack_.empty()) return nullptr;
            auto session = a_.scope_stack_.front().variables.find(name);
            if (session != a_.scope_stack_.front().variables.end() && session->second.summary)
                return session->second.summary.get();
            return nullptr;
        }
        bool isBuiltin(const std::string& name) const {
            return a_.external_names_.count(name) || !a_.global_names_.count(name);
        }

        Tokens mentions(const eshkol_ast_t* e, unsigned depth = 0) {
            Tokens tokens;
            if (!e || depth > 1024) return depth > 1024 ? all() : tokens;
            if (e->type == ESHKOL_VAR) return lookup(a_.getVarName(e));
            if (e->type == ESHKOL_OP && e->operation.op == ESHKOL_QUOTE_OP) return tokens;
            forEachChildExpr(e, [&](const eshkol_ast_t* child) { add(tokens, mentions(child, depth + 1)); });
            return tokens;
        }

        LambdaFlow lambdaScan(const eshkol_ast_t* node, const std::vector<Tokens>* positional,
                              const Tokens* every) {
            const FunctionShape shape = functionShape(node);
            const size_t mark = env_.size();
            frames_.push_back({mark, {}});
            for (uint64_t i = 0; i < shape.num_params; ++i) {
                Tokens tokens;
                if (positional && i < positional->size()) tokens = (*positional)[i];
                else if (every) tokens = *every;
                env_.push_back({a_.getVarName(&shape.params[i]), tokens});
            }
            if (shape.rest) {
                Tokens tokens;
                if (every) tokens = *every;
                if (positional)
                    for (size_t i = shape.num_params; i < positional->size(); ++i) add(tokens, (*positional)[i]);
                env_.push_back({shape.rest, tokens});
            }
            LambdaFlow flow;
            flow.result = scan(shape.body);
            flow.captured = std::move(frames_.back().second);
            frames_.pop_back();
            env_.resize(mark);
            for (auto& frame : frames_)
                if (frame.first <= mark) add(frame.second, flow.captured);
            return flow;
        }

        Tokens scanLet(const eshkol_ast_t* e) {
            const auto& let = e->operation.let_op;
            const eshkol_op_t kind = e->operation.op;
            const size_t mark = env_.size();
            auto name = [&](uint64_t i) {
                const eshkol_ast_t& binding = let.bindings[i];
                return binding.type == ESHKOL_CONS ? a_.getVarName(binding.cons_cell.car) : std::string();
            };
            auto init = [&](uint64_t i) -> const eshkol_ast_t* {
                const eshkol_ast_t& binding = let.bindings[i];
                return binding.type == ESHKOL_CONS ? binding.cons_cell.cdr : nullptr;
            };
            const uint64_t count = let.bindings ? let.num_bindings : 0;
            if (kind == ESHKOL_LET_OP && let.name) {
                std::vector<Tokens> inits;
                for (uint64_t i = 0; i < count; ++i) inits.push_back(scan(init(i)));
                const size_t base = env_.size();
                for (uint64_t i = 0; i < count; ++i) env_.push_back({name(i), inits[i]});
                Entry loop{let.name, {}};
                loop.loop = true;
                loop.loop_vars = base;
                loop.loop_count = count;
                env_.push_back(loop);
                Tokens result;
                for (int round = 0; round < 8; ++round) {
                    std::vector<Tokens> before;
                    for (uint64_t i = 0; i < count; ++i) before.push_back(env_[base + i].tokens);
                    result = scan(let.body);
                    bool stable = true;
                    for (uint64_t i = 0; i < count; ++i) stable = stable && before[i] == env_[base + i].tokens;
                    if (stable) break;
                    if (round == 7) {
                        result = all();
                        keep(result);
                    }
                }
                env_.resize(mark);
                return result;
            }
            if (kind == ESHKOL_LET_OP) {
                std::vector<Entry> entries;
                for (uint64_t i = 0; i < count; ++i) {
                    Entry entry{name(i), scan(init(i))};
                    if (isLambdaNode(init(i))) entry.proc = init(i);
                    entries.push_back(std::move(entry));
                }
                for (auto& entry : entries) env_.push_back(std::move(entry));
            } else if (kind == ESHKOL_LET_STAR_OP) {
                for (uint64_t i = 0; i < count; ++i) {
                    Entry entry{name(i), scan(init(i))};
                    if (isLambdaNode(init(i))) entry.proc = init(i);
                    env_.push_back(std::move(entry));
                }
            } else {
                for (uint64_t i = 0; i < count; ++i) {
                    Entry entry{name(i), {}};
                    if (isLambdaNode(init(i))) entry.proc = init(i);
                    env_.push_back(std::move(entry));
                }
                for (int round = 0; round < 3; ++round)
                    for (uint64_t i = 0; i < count; ++i) {
                        Tokens tokens = scan(init(i));
                        add(env_[mark + i].tokens, tokens);
                    }
            }
            Tokens result = scan(let.body);
            env_.resize(mark);
            return result;
        }

        // A procedure argument that may be called with @p values.
        void receive(const eshkol_ast_t* arg, const Tokens& values) {
            if (values.empty() || !arg || isLambdaNode(arg)) return;
            if (arg->type != ESHKOL_VAR) {
                if (arg->type == ESHKOL_OP && arg->operation.op != ESHKOL_QUOTE_OP) keep(values);
                return;
            }
            const std::string name = a_.getVarName(arg);
            const int i = find(name);
            if (i >= 0) {
                if (env_[i].proc && !env_[i].loop) {
                    if (!a_.summaryForScan(env_[i].proc, deps_).keepsNothing()) keep(values);
                } else if (!env_[i].tokens.empty() && !env_[i].loop) {
                    for (size_t p : Tokens(env_[i].tokens)) add(out_.passes[p], values);
                } else {
                    keep(values);
                }
                return;
            }
            if (const ProcSummary* summary = globalSummary(name)) {
                if (!summary->keepsNothing()) keep(values);
                return;
            }
            if (isBuiltin(name)) {
                if (builtinKeeps(name) != Keeps::NONE) keep(values);
                return;
            }
            keep(values);
        }

        Tokens scanCall(const eshkol_ast_t* e) {
            const auto& call = e->operation.call_op;
            const eshkol_ast_t* head = call.func;
            const size_t n = call.variables ? call.num_vars : 0;
            std::vector<Tokens> args(n);
            for (size_t j = 0; j < n; ++j)
                if (!isLambdaNode(&call.variables[j])) args[j] = scan(&call.variables[j]);
            auto others = [&](size_t k) {
                Tokens tokens;
                for (size_t j = 0; j < n; ++j)
                    if (j != k) add(tokens, args[j]);
                return tokens;
            };
            if (isLambdaNode(head)) {
                for (size_t j = 0; j < n; ++j)
                    if (isLambdaNode(&call.variables[j])) {
                        const Tokens every = others(j);
                        args[j] = lambdaScan(&call.variables[j], nullptr, &every).captured;
                    }
                LambdaFlow flow = lambdaScan(head, &args, nullptr);
                add(flow.result, flow.captured);
                return flow.result;
            }
            Head kind = Head::UNKNOWN;
            const ProcSummary* summary = nullptr;
            Tokens head_tokens;
            std::string head_name;
            int head_index = -1;
            if (head && head->type == ESHKOL_VAR) {
                head_name = a_.getVarName(head);
                head_index = find(head_name);
                head_tokens = lookup(head_name);
                if (head_index >= 0) {
                    const Entry& entry = env_[head_index];
                    if (entry.loop) kind = Head::LOOP;
                    else if (entry.proc) {
                        kind = Head::PROC;
                        summary = &a_.summaryForScan(entry.proc, deps_);
                    } else if (!entry.tokens.empty()) kind = Head::PARAM;
                } else if ((summary = globalSummary(head_name))) {
                    kind = Head::PROC;
                } else if (isBuiltin(head_name)) {
                    kind = Head::BUILTIN;
                }
            } else {
                head_tokens = scan(head);
            }
            std::vector<Tokens> receives(n);
            if (kind == Head::PROC) {
                for (const auto& [slot, sources] : summary->passes)
                    for (size_t k = 0; k < n; ++k) {
                        if (summary->slot(k) != slot) continue;
                        for (size_t j = 0; j < n; ++j)
                            if (sources.count(summary->slot(j))) add(receives[k], args[j]);
                    }
            } else if (kind != Head::LOOP &&
                       (kind != Head::BUILTIN || callsProcedureArguments(head_name))) {
                for (size_t k = 0; k < n; ++k) receives[k] = others(k);
            }
            for (size_t j = 0; j < n; ++j)
                if (isLambdaNode(&call.variables[j]))
                    args[j] = lambdaScan(&call.variables[j], nullptr, &receives[j]).captured;
            switch (kind) {
                case Head::LOOP: {
                    const size_t base = env_[head_index].loop_vars;
                    const size_t count = std::min(n, env_[head_index].loop_count);
                    for (size_t j = 0; j < count; ++j) add(env_[base + j].tokens, args[j]);
                    break;
                }
                case Head::PARAM: {
                    Tokens values;
                    for (const auto& tokens : args) add(values, tokens);
                    for (size_t p : Tokens(env_[head_index].tokens)) add(out_.passes[p], values);
                    break;
                }
                case Head::PROC:
                    for (size_t j = 0; j < n; ++j)
                        if (summary->keeps.count(summary->slot(j))) keep(args[j]);
                    for (size_t k = 0; k < n; ++k) receive(&call.variables[k], receives[k]);
                    break;
                case Head::BUILTIN: {
                    const Keeps keeps = builtinKeeps(head_name);
                    for (size_t j = 0; j < n; ++j)
                        if (keeps == Keeps::ALL || (keeps == Keeps::AFTER_FIRST && j > 0)) keep(args[j]);
                    if (callsProcedureArguments(head_name))
                        for (size_t k = 0; k < n; ++k) receive(&call.variables[k], others(k));
                    break;
                }
                case Head::UNKNOWN:
                    for (const auto& tokens : args) keep(tokens);
                    break;
            }
            if (kind == Head::BUILTIN && isScalarResultBuiltin(head_name)) return {};
            Tokens result = head_tokens;
            for (const auto& tokens : args) add(result, tokens);
            return result;
        }

        Tokens scan(const eshkol_ast_t* e) {
            if (!e) return {};
            if (depth_ > 1024) {
                Tokens tokens = all();
                keep(tokens);
                return tokens;
            }
            struct Depth {
                unsigned& value;
                explicit Depth(unsigned& d) : value(d) { ++value; }
                ~Depth() { --value; }
            } guard(depth_);
            if (e->type == ESHKOL_VAR) return lookup(a_.getVarName(e));
            if (e->type == ESHKOL_CONS) {
                Tokens tokens = scan(e->cons_cell.car);
                add(tokens, scan(e->cons_cell.cdr));
                return tokens;
            }
            if (e->type != ESHKOL_OP) return {};
            const auto& op = e->operation;
            if (a_.isIfForm(e)) {
                Tokens tokens;
                scan(&op.call_op.variables[0]);
                for (uint64_t i = 1; i < op.call_op.num_vars; ++i) add(tokens, scan(&op.call_op.variables[i]));
                return tokens;
            }
            auto last = [&](const eshkol_ast_t* nodes, uint64_t count) {
                Tokens tokens;
                for (uint64_t i = 0; nodes && i < count; ++i) tokens = scan(&nodes[i]);
                return tokens;
            };
            switch (op.op) {
                case ESHKOL_QUOTE_OP:
                case ESHKOL_DEFINE_TYPE_OP:
                case ESHKOL_IMPORT_OP:
                case ESHKOL_REQUIRE_OP:
                case ESHKOL_PROVIDE_OP:
                case ESHKOL_DEFINE_SYNTAX_OP:
                case ESHKOL_TYPE_ANNOTATION_OP:
                case ESHKOL_EXTERN_OP:
                case ESHKOL_EXTERN_VAR_OP:
                    return {};
                case ESHKOL_ADD_OP:
                case ESHKOL_SUB_OP:
                case ESHKOL_MUL_OP:
                case ESHKOL_DIV_OP:
                    for (uint64_t i = 0; op.call_op.variables && i < op.call_op.num_vars; ++i)
                        scan(&op.call_op.variables[i]);
                    return {};
                case ESHKOL_SEQUENCE_OP:
                    return last(op.sequence_op.expressions, op.sequence_op.num_expressions);
                case ESHKOL_WITH_REGION_OP:
                    return last(op.with_region_op.body, op.with_region_op.num_body_exprs);
                case ESHKOL_AND_OP:
                case ESHKOL_OR_OP: {
                    Tokens tokens;
                    for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                        add(tokens, scan(&op.sequence_op.expressions[i]));
                    return tokens;
                }
                case ESHKOL_WHEN_OP:
                case ESHKOL_UNLESS_OP:
                    return last(op.call_op.variables, op.call_op.num_vars);
                case ESHKOL_COND_OP: {
                    Tokens tokens;
                    for (uint64_t i = 0; op.call_op.variables && i < op.call_op.num_vars; ++i) {
                        const eshkol_ast_t* clause = &op.call_op.variables[i];
                        if (clause->type == ESHKOL_OP && clause->operation.op == ESHKOL_CALL_OP) {
                            add(tokens, scan(clause->operation.call_op.func));
                            add(tokens, last(clause->operation.call_op.variables,
                                             clause->operation.call_op.num_vars));
                        } else {
                            add(tokens, scan(clause));
                        }
                    }
                    return tokens;
                }
                case ESHKOL_DEFINE_OP: {
                    const auto& define = op.define_op;
                    Entry entry{define.name ? define.name : "", {}};
                    if (define.is_function) {
                        entry.tokens = lambdaScan(e, nullptr, nullptr).captured;
                        if (!define.is_external) entry.proc = e;
                    } else {
                        entry.tokens = scan(define.value);
                        if (isLambdaNode(define.value)) entry.proc = define.value;
                    }
                    env_.push_back(std::move(entry));
                    return {};
                }
                case ESHKOL_SET_OP: {
                    const Tokens tokens = scan(op.set_op.value);
                    keep(tokens);
                    if (op.set_op.name) {
                        const int i = find(op.set_op.name);
                        if (i >= 0) {
                            add(env_[i].tokens, tokens);
                            env_[i].proc = nullptr;
                        }
                    }
                    return {};
                }
                case ESHKOL_LAMBDA_OP:
                    return lambdaScan(e, nullptr, nullptr).captured;
                case ESHKOL_LET_OP:
                case ESHKOL_LET_STAR_OP:
                case ESHKOL_LETREC_OP:
                case ESHKOL_LETREC_STAR_OP:
                    return scanLet(e);
                case ESHKOL_OWNED_OP:
                    return scan(op.owned_op.value);
                case ESHKOL_MOVE_OP:
                    return scan(op.move_op.value);
                case ESHKOL_SHARED_OP:
                    return scan(op.shared_op.value);
                case ESHKOL_WEAK_REF_OP:
                    return scan(op.weak_ref_op.value);
                case ESHKOL_THE_OP:
                    return scan(op.the_op.expr);
                case ESHKOL_BORROW_OP:
                    scan(op.borrow_op.value);
                    return last(op.borrow_op.body, op.borrow_op.num_body_exprs);
                case ESHKOL_RAISE_OP:
                    keep(scan(op.raise_op.exception));
                    return {};
                case ESHKOL_CALL_OP:
                    return scanCall(e);
                default: {
                    Tokens tokens = mentions(e);
                    keep(tokens);
                    return tokens;
                }
            }
        }
    };

    // Summary of a procedure being scanned: final, or the current
    // assumption of its fixed point, which the caller then iterates.
    const ProcSummary& summaryForScan(const eshkol_ast_t* proc, std::vector<const eshkol_ast_t*>& deps) {
        if (!proc_final_.count(proc)) deps.push_back(proc);
        return proc_table_[proc];
    }

    const ProcSummary& summaryOf(const eshkol_ast_t* proc) {
        if (proc_final_.count(proc)) return proc_table_[proc];
        std::vector<const eshkol_ast_t*> group{proc};
        std::set<const eshkol_ast_t*> members{proc};
        proc_table_[proc];
        bool changed = true;
        for (int round = 0; changed && round < 64; ++round) {
            changed = false;
            for (size_t i = 0; i < group.size(); ++i) {
                std::vector<const eshkol_ast_t*> deps;
                ProcSummary next;
                ProcScan(*this, next, deps).run(group[i]);
                for (const auto* dep : deps)
                    if (members.insert(dep).second) {
                        group.push_back(dep);
                        changed = true;
                    }
                ProcSummary& current = proc_table_[group[i]];
                ProcSummary merged = current;
                merged.num_params = next.num_params;
                merged.rest = next.rest;
                merged.keeps.insert(next.keeps.begin(), next.keeps.end());
                for (const auto& [slot, sources] : next.passes)
                    merged.passes[slot].insert(sources.begin(), sources.end());
                if (!(merged == current) || merged.num_params != current.num_params ||
                    merged.rest != current.rest) {
                    current = std::move(merged);
                    changed = true;
                }
            }
        }
        if (changed)
            for (const auto* member : group) {
                ProcSummary& summary = proc_table_[member];
                for (size_t i = 0; i <= summary.num_params; ++i) summary.keeps.insert(i);
            }
        for (const auto* member : group) proc_final_.insert(member);
        return proc_table_[proc];
    }

    const ProcSummary* summaryOfBinding(const VariableInfo& info) {
        if (info.summary) return info.summary.get();
        if (info.proc_node) return &summaryOf(info.proc_node);
        return nullptr;
    }

    const ProcSummary* globalSummaryByName(const std::string& name) {
        auto proc = global_procs_.find(name);
        return proc == global_procs_.end() ? nullptr : &summaryOf(proc->second);
    }

    // ── calls ───────────────────────────────────────────────────────────
    // Arguments that refer to owners may only reach procedures that do not
    // keep them. Lambdas written as arguments receive what the callee may
    // pass them, and their bodies are checked with those parameters.
    void checkCallFlow(const eshkol_ast_t* node) {
        const auto& call = node->operation.call_op;
        const eshkol_ast_t* head = call.func;
        const size_t n = call.variables ? call.num_vars : 0;
        std::vector<Reach> reach(n);
        std::vector<std::string> owner(n);
        bool any = false;
        for (size_t j = 0; j < n; ++j) {
            reachOf(&call.variables[j], reach[j]);
            const auto owners = reachedOwners(reach[j]);
            if (!owners.empty()) {
                owner[j] = owners.front().name;
                any = true;
            }
        }
        if (isLambdaNode(head)) {
            immediate_lambdas_.insert(head);
            lambda_param_reach_[head].positional = reach;
            return;
        }
        Reach head_reach;
        reachOf(head, head_reach);
        auto others = [&](size_t k) {
            Reach result = head_reach;
            for (size_t j = 0; j < n; ++j)
                if (j != k) result.insert(reach[j].begin(), reach[j].end());
            return result;
        };
        for (size_t k = 0; k < n; ++k)
            if (isLambdaNode(&call.variables[k])) lambda_param_reach_[&call.variables[k]].all = others(k);
        if (!any) return;
        auto refuseKept = [&](const std::string& value, const std::string& callee) {
            reportError("'" + callee + "' may keep a reference to owned value '" + value +
                        "' after the call; transfer it with move");
        };
        auto refuseUnknown = [&](const std::string& value) {
            reportError("Owned value '" + value +
                        "' is passed to a procedure the ownership checker cannot resolve");
        };
        auto receive = [&](size_t k, const std::string& value) {
            const eshkol_ast_t* arg = &call.variables[k];
            if (value.empty() || isLambdaNode(arg)) return;
            if (arg->type != ESHKOL_VAR) {
                if (!neverProcedure(arg)) refuseUnknown(value);
                return;
            }
            const std::string name = getVarName(arg);
            if (isBuiltinName(name)) {
                if (builtinKeeps(name) != Keeps::NONE) refuseKept(value, name);
                return;
            }
            VariableInfo* info = lookupVariable(name);
            if (info && info->data) return;
            if (info && info->external) {
                if (builtinKeeps(name) != Keeps::NONE) refuseKept(value, name);
                return;
            }
            const ProcSummary* summary = info ? summaryOfBinding(*info) : globalSummaryByName(name);
            if (!summary) refuseUnknown(value);
            else if (!summary->keepsNothing()) refuseKept(value, name);
        };
        auto firstOther = [&](size_t k, const std::set<size_t>* sources, const ProcSummary* summary) {
            for (size_t j = 0; j < n; ++j) {
                if (j == k || owner[j].empty()) continue;
                if (!sources || sources->count(summary->slot(j))) return owner[j];
            }
            return std::string();
        };
        if (head && head->type == ESHKOL_VAR) {
            const std::string head_name = getVarName(head);
            VariableInfo* info = lookupVariable(head_name);
            if ((!info && isBuiltinName(head_name)) || (info && info->external)) {
                if (head_name == "eval") {
                    reportError("eval may name owned value '" + firstOther(n, nullptr, nullptr) +
                                "' outside the ownership checker");
                    return;
                }
                const Keeps keeps = builtinKeeps(head_name);
                for (size_t j = 0; j < n; ++j)
                    if (!owner[j].empty() && (keeps == Keeps::ALL || (keeps == Keeps::AFTER_FIRST && j > 0)))
                        refuseKept(owner[j], head_name);
                if (callsProcedureArguments(head_name))
                    for (size_t k = 0; k < n; ++k) receive(k, firstOther(k, nullptr, nullptr));
                return;
            }
            if (info && info->loop_proc) return;  // its variables record every argument
            const ProcSummary* summary = info ? summaryOfBinding(*info) : globalSummaryByName(head_name);
            if (summary) {
                for (size_t j = 0; j < n; ++j)
                    if (!owner[j].empty() && summary->keeps.count(summary->slot(j)))
                        refuseKept(owner[j], head_name);
                for (const auto& [slot, sources] : summary->passes)
                    for (size_t k = 0; k < n; ++k)
                        if (summary->slot(k) == slot) receive(k, firstOther(k, &sources, summary));
                return;
            }
        }
        for (size_t j = 0; j < n; ++j)
            if (!owner[j].empty()) refuseUnknown(owner[j]);
    }

    // Named-let variables also take every argument of a call to the loop.
    void bindLoopFacts(const eshkol_operations_t* op) {
        if (!alias_tracking_ || !op->let_op.name) return;
        const std::string loop = op->let_op.name;
        std::vector<VariableInfo*> vars;
        for (uint64_t i = 0; i < op->let_op.num_bindings; ++i) {
            const eshkol_ast_t& binding = op->let_op.bindings[i];
            const std::string name = binding.type == ESHKOL_CONS ? getVarName(binding.cons_cell.car) : "";
            auto found = currentScope().variables.find(name);
            vars.push_back(name.empty() || found == currentScope().variables.end() ? nullptr : &found->second);
        }
        auto existing = currentScope().variables.find(loop);
        if (existing == currentScope().variables.end()) {
            VariableInfo info{State::UNOWNED, loop, false, next_binding_id_++};
            info.is_defined = true;
            existing = currentScope().variables.emplace(loop, std::move(info)).first;
        }
        existing->second.loop_proc = true;
        std::vector<const eshkol_ast_t*> walk{op->let_op.body};
        while (!walk.empty()) {
            const eshkol_ast_t* node = walk.back();
            walk.pop_back();
            if (!node) continue;
            if (node->type == ESHKOL_OP && node->operation.op == ESHKOL_CALL_OP &&
                getVarName(node->operation.call_op.func) == loop) {
                const auto& call = node->operation.call_op;
                for (uint64_t i = 0; call.variables && i < call.num_vars && i < vars.size(); ++i)
                    if (vars[i]) reachOf(&call.variables[i], vars[i]->reaches);
            }
            forEachChildExpr(node, [&](const eshkol_ast_t* child) { walk.push_back(child); });
        }
        Reach body;
        mentionsOf(op->let_op.body, body);
        body.erase(existing->second.binding_id);
        existing->second.reaches = std::move(body);
        enterBody();
    }

    // ── interactive sessions ────────────────────────────────────────────
    static std::string scopedOutKey(const std::string& name, uint64_t id) {
        return std::string("\x1f") + name + "\x1f" + std::to_string(id);
    }

    static std::string scopedOutName(const std::string& key) {
        if (key.empty() || key[0] != '\x1f') return key;
        const size_t end = key.find('\x1f', 1);
        return key.substr(1, end == std::string::npos ? std::string::npos : end - 1);
    }

    void loadSession(const OwnershipBindings& globals) {
        std::map<std::string, uint64_t> ids;
        for (const auto& entry : globals) ids[entry.first] = next_binding_id_++;
        for (const auto& [name, binding] : globals) {
            Reach reaches;
            for (const auto& target : binding.reaches) {
                auto id = ids.find(target);
                if (id != ids.end()) reaches.insert(id->second);
            }
            if (binding.scoped_out) {
                scoped_out_[ids[name]] = {scopedOutName(name), binding.state, binding.owner, reaches};
                continue;
            }
            VariableInfo info{binding.state, name, binding.owner, ids[name]};
            info.reaches = std::move(reaches);
            info.data = binding.data;
            info.is_defined = true;
            if (binding.procedure) {
                auto summary = std::make_shared<ProcSummary>();
                summary->keeps = binding.keeps;
                summary->passes = binding.passes;
                summary->num_params = binding.num_params;
                summary->rest = binding.rest;
                info.summary = std::move(summary);
            }
            currentScope().variables[name] = std::move(info);
        }
    }

    void saveSession(OwnershipBindings& globals) {
        OwnershipBindings saved;
        std::map<uint64_t, std::string> keys;
        for (const auto& [name, info] : currentScope().variables) keys[info.binding_id] = name;
        std::vector<uint64_t> pending;
        for (const auto& entry : currentScope().variables)
            pending.insert(pending.end(), entry.second.reaches.begin(), entry.second.reaches.end());
        while (!pending.empty()) {
            const uint64_t id = pending.back();
            pending.pop_back();
            if (keys.count(id)) continue;
            auto out = scoped_out_.find(id);
            if (out == scoped_out_.end()) continue;
            keys[id] = scopedOutKey(out->second.name, id);
            pending.insert(pending.end(), out->second.reaches.begin(), out->second.reaches.end());
        }
        auto names = [&](const Reach& reaches) {
            std::set<std::string> result;
            for (uint64_t id : reaches) {
                auto key = keys.find(id);
                if (key != keys.end()) result.insert(key->second);
            }
            return result;
        };
        for (const auto& [name, info] : currentScope().variables) {
            OwnershipBinding binding;
            binding.state = info.state;
            binding.owner = info.is_owned_binding;
            binding.reaches = names(info.reaches);
            binding.data = info.data;
            if ((info.proc_node || info.summary) && !info.external) {
                const ProcSummary& summary = info.summary ? *info.summary : summaryOf(info.proc_node);
                binding.procedure = true;
                binding.keeps = summary.keeps;
                binding.passes = summary.passes;
                binding.num_params = summary.num_params;
                binding.rest = summary.rest;
            }
            saved[name] = std::move(binding);
        }
        for (const auto& [id, key] : keys) {
            auto out = scoped_out_.find(id);
            if (out == scoped_out_.end() || saved.count(key)) continue;
            OwnershipBinding binding;
            binding.state = out->second.state;
            binding.owner = out->second.owner;
            binding.reaches = names(out->second.reaches);
            binding.scoped_out = true;
            saved[key] = std::move(binding);
        }
        globals.swap(saved);
    }

    void checkCopy(const eshkol_ast_t* value) {
        if (!value || value->type != ESHKOL_VAR || !value->variable.id) return;
        const auto* source = lookupVariable(value->variable.id);
        if (source && (source->state == State::OWNED || isBorrowed(source->binding_id)))
            reportError("Cannot copy owned or borrowed value '" + std::string(value->variable.id) +
                        "' into another binding; transfer ownership with move");
    }

    // Check if expression is (owned ...)
    bool isOwnedExpr(const eshkol_ast_t* ast) {
        return ast && ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_OWNED_OP;
    }

    // Check if expression is (move ...)
    bool isMoveExpr(const eshkol_ast_t* ast) {
        return ast && ast->type == ESHKOL_OP && ast->operation.op == ESHKOL_MOVE_OP;
    }

    // Check if expression transfers ownership (owned or move)
    bool transfersOwnership(const eshkol_ast_t* ast) {
        return isOwnedExpr(ast) || isMoveExpr(ast);
    }
};

}

static thread_local OwnershipBindings* interactive_bindings = nullptr;

OwnershipTransaction::OwnershipTransaction(OwnershipBindings& session)
    : session_(session), pending_(session), previous_(interactive_bindings) {
    interactive_bindings = &pending_;
}

OwnershipTransaction::~OwnershipTransaction() {
    interactive_bindings = previous_;
}

void OwnershipTransaction::commit() { session_.swap(pending_); }

OwnershipBindings* OwnershipTransaction::current() { return interactive_bindings; }

std::vector<std::string> checkOwnership(const std::vector<eshkol_ast_t>& forms,
                                        const ResourceContracts* strict,
                                        OwnershipBindings* globals) {
    // Most compilation units contain only ordinary bindings. Walk them once
    // without allocating control-flow snapshots. Encountering an ownership
    // form restarts the walk before any provisional globals are published.
    const bool live_ownership = globals && std::any_of(globals->begin(), globals->end(),
        [](const auto& binding) {
            return binding.second.state != OwnershipState::UNOWNED || binding.second.owner ||
                   !binding.second.reaches.empty();
        });
    OwnershipAnalyzer analyzer(strict, strict || live_ownership);
    analyzer.analyze(forms, globals);
    if (!analyzer.needsBranchAnalysis()) return analyzer.getErrors();
    OwnershipAnalyzer flow(strict, true);
    flow.analyze(forms, globals);
    return flow.getErrors();
}

}
