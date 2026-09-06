/*
 * lib/backend/xla/region_formation.cpp — the region-formation pass.
 *
 * See inc/eshkol/backend/xla/region_formation.h for what this is for. This
 * file is the walk itself, and its two load-bearing properties are:
 *
 *   1. A NODE THAT CANNOT BE PLACED IS A BREAK, NOT A SKIP. Every path out of
 *      eligible() that is not "yes" produces a GraphBreak with a reason. The
 *      default arm of every switch over the AST is a break naming the form,
 *      so a special form nobody has considered stops a region rather than
 *      silently joining one.
 *
 *   2. THE WALK CONTINUES THROUGH A BREAK. A host builtin's arguments, a
 *      differentiated function's body, a print's operand: each can hold a
 *      region of its own, and "maximal" means nothing if the pass stops at the
 *      first thing it cannot place.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include "eshkol/backend/xla/region_formation.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"

namespace eshkol {
namespace xla {

namespace {

/** @brief One row of the generated classification table. */
struct RegionBuiltinLabel {
    const char* name;
    BuiltinLabel label;
};

/** @brief One row of the generated core-op table: a builtin the device
 *         lowering owns directly, and the arities at which it means that op.
 *         `arity_mask` bit N is set when arity N is meant; `arity1_kind` is
 *         the op at arity one, which differs from `kind` for the reduce/
 *         elementwise pairs (tensor-max is Maximum at two operands and
 *         ReduceMax at one). */
struct RegionCoreOp {
    const char* name;
    DeviceOpKind kind;
    uint32_t arity_mask;
    DeviceOpKind arity1_kind;
};

#include "region_tables.inc"

constexpr size_t kBuiltinLabelCount = sizeof(kBuiltinLabels) / sizeof(kBuiltinLabels[0]);
// kLoweredBuiltins (the composition rows) is generated but deliberately not
// consulted here; see hasLowering(). Naming it keeps the compiler from
// warning about an unused table while leaving the reason in one place.
constexpr size_t kLoweredCount = sizeof(kLoweredBuiltins) / sizeof(kLoweredBuiltins[0]);
static_assert(kLoweredCount > 0, "the composition table should not be empty");
constexpr size_t kCoreOpCount = sizeof(kCoreOps) / sizeof(kCoreOps[0]);

/** @brief The label the classification table gives @p name. */
BuiltinLabel labelOf(const char* name) {
    if (!name) return BuiltinLabel::Unclassified;
    size_t lo = 0, hi = kBuiltinLabelCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = std::strcmp(kBuiltinLabels[mid].name, name);
        if (c == 0) return kBuiltinLabels[mid].label;
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return BuiltinLabel::Unclassified;
}

/** @brief Does a StableHLO lowering exist for @p name at @p arity?
 *
 *  Only `core_ops` counts, not the composition rows above it in
 *  device_lowering_table.yaml. Those compositions are real and are measured
 *  by tests/xla/builtin_parity_test, but they are expressed as SSA steps in
 *  YAML and the region emitter has no interpreter for them: it emits a
 *  DeviceOpKind per node. Admitting a builtin into a region that the emitter
 *  would then refuse would move the failure from "reported graph break" to
 *  "region that cannot be compiled", which is the wrong end of this stage's
 *  contract. A composition joins the region-eligible set when it gains a
 *  core_ops row, i.e. when a DeviceOpKind can express it.
 */
bool hasLowering(const char* name, uint64_t arity) {
    if (!name) return false;
    size_t lo = 0, hi = kCoreOpCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = std::strcmp(kCoreOps[mid].name, name);
        if (c == 0) {
            if (arity >= 32) return false;
            return (kCoreOps[mid].arity_mask & (1u << arity)) != 0;
        }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

/** @brief The DeviceOpKind @p name means at @p arity, or false when it has
 *         none. Arity separates the reduce/elementwise pairs: `tensor-max` of
 *         two operands is a Maximum, of one it is a ReduceMax. */
bool coreOpKind(const char* name, uint64_t arity, DeviceOpKind* out) {
    if (!name) return false;
    size_t lo = 0, hi = kCoreOpCount;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = std::strcmp(kCoreOps[mid].name, name);
        if (c == 0) {
            if (arity >= 32 || !(kCoreOps[mid].arity_mask & (1u << arity))) return false;
            *out = (arity == 1) ? kCoreOps[mid].arity1_kind : kCoreOps[mid].kind;
            return true;
        }
        if (c < 0) lo = mid + 1; else hi = mid;
    }
    return false;
}

const char* labelName(BuiltinLabel label) {
    switch (label) {
        case BuiltinLabel::Device: return "device";
        case BuiltinLabel::Host: return "host";
        case BuiltinLabel::HostWithDeviceInner: return "host-with-device-inner";
        case BuiltinLabel::Unclassified: return "unclassified";
    }
    return "unclassified";
}

/** @brief The name a `let` binding binds.
 *
 *  A binding node is a CONS whose car is the name and whose cdr is the value
 *  — the layout llvm_codegen.cpp reads (see its let handling). Reading it as
 *  anything else does not crash: it silently sees no bindings, which makes a
 *  `let` look like a region with no inputs and hides every break inside a
 *  binding's value. That is exactly what happened before this was checked
 *  against the code that already knew.
 */
const char* bindingName(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS && b->cons_cell.car &&
        b->cons_cell.car->type == ESHKOL_VAR)
        return b->cons_cell.car->variable.id;
    if (b->type == ESHKOL_VAR) return b->variable.id;
    return nullptr;
}

/** @brief The expression a `let` binding binds it to. */
const eshkol_ast_t* bindingValue(const eshkol_ast_t* b) {
    if (!b) return nullptr;
    if (b->type == ESHKOL_CONS) return b->cons_cell.cdr;
    if (b->type == ESHKOL_VAR) return b->variable.data;
    return b;
}

/** @brief The variable name a call's callee position holds, or nullptr. */
const char* calleeName(const eshkol_ast_t* node) {
    if (!node) return nullptr;
    if (node->type == ESHKOL_VAR) return node->variable.id;
    return nullptr;
}

/** @brief Is this call node a conditional written as a call to `if`? */
bool isIfCall(const eshkol_operations_t& op) {
    if (op.op != ESHKOL_CALL_OP) return false;
    const eshkol_ast_t* f = op.call_op.func;
    return f && f->type == ESHKOL_VAR && f->variable.id &&
           std::strcmp(f->variable.id, "if") == 0 && op.call_op.num_vars == 3;
}

/** @brief Contract text for each break reason, quoted from
 *         docs/design/ESHKOL_S_FRAGMENT.md rather than paraphrased, so a
 *         report says why in the contract's own words. */
std::string contractText(BreakReason reason, const std::string& detail) {
    switch (reason) {
        case BreakReason::HostBuiltin:
            return "labelled host in builtin_classification.yaml: host-only by "
                   "nature (no device value representation, non-local or "
                   "side-effecting control, or object lifecycle/metadata)";
        case BreakReason::HostWithDeviceInner:
            return "labelled host-with-device-inner: the builtin itself is host, "
                   "and its argument procedure's body is walked for regions of "
                   "its own";
        case BreakReason::UnclassifiedBuiltin:
            return "absent from builtin_classification.yaml; an unclassified "
                   "builtin is a gap, not an implicit host";
        case BreakReason::NoLowering:
            return "labelled device but no StableHLO lowering exists for it at "
                   "this arity; the label is a claim the builtin CAN be "
                   "represented, not that the emitter can emit it";
        case BreakReason::NonAdmittedConstruct:
            return "condition 3 (structured control flow) admits only `if` with "
                   "both arms in the fragment and a tail-recursive loop over "
                   "fragment-typed state; " + detail;
        case BreakReason::HostValueDomain:
            return "condition 1 (value domain) excludes cons cells and lists, "
                   "strings and characters, symbols, ports, closures and "
                   "continuations unconditionally";
        case BreakReason::UnknownShape:
            return "condition 2 (shape discipline) requires a static shape, or "
                   "a dimension StableHLO's bounded dynamism can express";
        case BreakReason::RecursiveCall:
            return "a recursive call cannot be inlined into a region and is not "
                   "the tail-recursive loop over fragment-typed state that "
                   "condition 3 admits";
        case BreakReason::HostFunction:
            return "a top-level function of this module whose own body leaves "
                   "the fragment, so a call to it cannot be inlined into a "
                   "region; its body's own breaks are reported where it is "
                   "defined";
        case BreakReason::UnknownFunction:
            return "neither a registered builtin nor a top-level definition in "
                   "this module, so nothing is known about what it computes";
    }
    return "";
}

} // namespace

bool regionCoreOpKind(const char* name, uint64_t arity, DeviceOpKind* out) {
    return coreOpKind(name, arity, out);
}

const char* breakReasonName(BreakReason reason) {
    switch (reason) {
        case BreakReason::HostBuiltin: return "host-builtin";
        case BreakReason::HostWithDeviceInner: return "host-with-device-inner";
        case BreakReason::UnclassifiedBuiltin: return "unclassified-builtin";
        case BreakReason::NoLowering: return "no-lowering";
        case BreakReason::NonAdmittedConstruct: return "non-admitted-construct";
        case BreakReason::HostValueDomain: return "host-value-domain";
        case BreakReason::UnknownShape: return "unknown-shape";
        case BreakReason::RecursiveCall: return "recursive-call";
        case BreakReason::HostFunction: return "host-function";
        case BreakReason::UnknownFunction: return "unknown-function";
    }
    return "unknown";
}

std::string RegionShape::text() const {
    if (!known) return "?";
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < dims.size(); ++i) {
        if (i) oss << ",";
        oss << dims[i];
    }
    oss << "]";
    return oss.str();
}

RegionFormationOptions regionOptionsFromEnvironment() {
    RegionFormationOptions o;
    const char* trace = std::getenv("ESHKOL_XLA_TRACE_REGIONS");
    o.trace = trace && trace[0] && std::strcmp(trace, "0") != 0;
    const char* report = std::getenv("ESHKOL_XLA_REGION_REPORT");
    if (report && report[0]) o.report_path = report;
    const char* on = std::getenv("ESHKOL_XLA_REGIONS");
    o.enabled = on && on[0] && std::strcmp(on, "0") != 0;
    return o;
}

// ─────────────────────────────────────────────────────────────────────────
// The pass
// ─────────────────────────────────────────────────────────────────────────

class RegionFormation::Impl {
public:
    explicit Impl(RegionFormationOptions options) : options_(std::move(options)) {}

    RegionFormationOptions options_;
    ModuleReport report_;

    /** Top-level function bodies, for the interprocedural decision. */
    std::unordered_map<std::string, const eshkol_ast_t*> bodies_;
    /** Parameter names of each top-level function, in order. */
    std::unordered_map<std::string, std::vector<std::string>> params_;
    /** The same functions, in the form a region emitter consumes. */
    std::map<std::string, RegionFunction> functions_;
    /** Shapes of the module's top-level value definitions.
     *
     *  WHY THIS EXISTS. Nearly every corpus program binds its data at top
     *  level — `(define X #(0.25 0.5 0.75 1.0))` — and a region that takes X
     *  as an input reported its shape as unknown, which is not true: the
     *  frontend wrote [4] into that node. An unknown shape is not a harmless
     *  approximation here, because a region keyed by an unknown signature
     *  cannot share an executable with the next entry that has the same
     *  actual shape. */
    std::map<std::string, RegionShape> module_shapes_;

    /** Memoised eligibility of a top-level function's whole body. */
    std::unordered_map<std::string, int> function_eligible_;   // -1 unknown, 0 no, 1 yes

    /** The unit being analysed. */
    UnitReport* unit_ = nullptr;
    /** Depth of enclosing differentiation operators. */
    int gradient_depth_ = 0;
    /** Bindings whose value is known to be host-domain (a list, a string...). */
    std::vector<std::set<std::string>> host_bindings_;
    /** Bindings whose static shape is known. */
    std::vector<std::map<std::string, RegionShape>> shape_bindings_;

    // ── declaration ──────────────────────────────────────────────────────

    void declare(const eshkol_ast_t* form) {
        if (!form || form->type != ESHKOL_OP) return;
        if (form->operation.op != ESHKOL_DEFINE_OP) return;
        const auto& d = form->operation.define_op;
        if (!d.name) return;
        if (!d.is_function) {
            if (d.value) module_shapes_[d.name] = shapeOf(d.value);
            return;
        }
        if (!d.value) return;
        bodies_[d.name] = d.value;
        std::vector<std::string> names;
        for (uint64_t i = 0; i < d.num_params; ++i) {
            const eshkol_ast_t* p = &d.parameters[i];
            if (p->type == ESHKOL_VAR && p->variable.id) names.push_back(p->variable.id);
            else names.push_back("");
        }
        params_[d.name] = names;
        RegionFunction fn;
        fn.params = std::move(names);
        fn.body = d.value;
        functions_[d.name] = std::move(fn);
        function_eligible_[d.name] = -1;
    }

    /** @brief Is a call to top-level @p name a device node?
     *
     *  True only when the function's whole body is eligible. This is the
     *  interprocedural half of the pass and the reason a qLLM forward pass
     *  written as a dozen small `define`s can be one region rather than a
     *  dozen breaks. Recursion answers false: `active` holds the functions
     *  currently being decided, and re-entering one is the cycle.
     */
    bool functionEligible(const std::string& name, std::set<std::string>& active) {
        auto memo = function_eligible_.find(name);
        if (memo == function_eligible_.end()) return false;   // not a top-level define
        if (memo->second >= 0) return memo->second == 1;
        if (active.count(name)) return false;                 // recursive
        active.insert(name);
        // Decide the body with the function's parameters in scope as
        // unknown-shape device leaves, and with reporting suppressed: this is
        // a question, not a walk, and its breaks belong to whoever calls it.
        pushScope();
        for (const std::string& p : params_[name]) {
            if (!p.empty()) shape_bindings_.back()[p] = RegionShape{};
        }
        std::vector<GraphBreak> discard;
        bool ok = eligible(bodies_[name], &discard, active);
        popScope();
        active.erase(name);
        function_eligible_[name] = ok ? 1 : 0;
        return ok;
    }

    // ── scopes ───────────────────────────────────────────────────────────

    void pushScope() {
        host_bindings_.emplace_back();
        shape_bindings_.emplace_back();
    }
    void popScope() {
        host_bindings_.pop_back();
        shape_bindings_.pop_back();
    }
    bool isHostBinding(const char* name) const {
        if (!name) return false;
        for (auto it = host_bindings_.rbegin(); it != host_bindings_.rend(); ++it)
            if (it->count(name)) return true;
        return false;
    }
    RegionShape shapeOfBinding(const char* name) const {
        if (name) {
            for (auto it = shape_bindings_.rbegin(); it != shape_bindings_.rend(); ++it) {
                auto f = it->find(name);
                if (f != it->end()) return f->second;
            }
            auto m = module_shapes_.find(name);
            if (m != module_shapes_.end()) return m->second;
        }
        return RegionShape{};
    }

    // ── eligibility ──────────────────────────────────────────────────────

    void addBreak(std::vector<GraphBreak>* out, const eshkol_ast_t* node,
                  const std::string& construct, BreakReason reason,
                  const std::string& builtin, BuiltinLabel label,
                  const std::string& detail = "") {
        if (!out) return;
        GraphBreak b;
        b.line = node ? node->line : 0;
        b.column = node ? node->column : 0;
        b.construct = construct;
        b.reason = reason;
        b.reason_text = contractText(reason, detail);
        b.builtin = builtin;
        b.node = node;
        if (!builtin.empty()) b.label = labelName(label);
        out->push_back(std::move(b));
    }

    /** @brief The name of an AST form, for a break's `construct` field. */
    static const char* formName(eshkol_op_t op) {
        switch (op) {
            case ESHKOL_SET_OP: return "set!";
            case ESHKOL_COND_OP: return "cond";
            case ESHKOL_CASE_OP: return "case";
            case ESHKOL_WHEN_OP: return "when";
            case ESHKOL_UNLESS_OP: return "unless";
            case ESHKOL_AND_OP: return "and";
            case ESHKOL_OR_OP: return "or";
            case ESHKOL_DO_OP: return "do";
            case ESHKOL_MATCH_OP: return "match";
            case ESHKOL_CALL_CC_OP: return "call/cc";
            case ESHKOL_DYNAMIC_WIND_OP: return "dynamic-wind";
            case ESHKOL_GUARD_OP: return "guard";
            case ESHKOL_RAISE_OP: return "raise";
            case ESHKOL_QUOTE_OP: return "quote";
            case ESHKOL_QUASIQUOTE_OP: return "quasiquote";
            case ESHKOL_LAMBDA_OP: return "lambda";
            case ESHKOL_LETREC_OP: return "letrec";
            case ESHKOL_LETREC_STAR_OP: return "letrec*";
            case ESHKOL_GRADIENT_OP: return "gradient";
            case ESHKOL_DERIVATIVE_OP: return "derivative";
            case ESHKOL_JACOBIAN_OP: return "jacobian";
            case ESHKOL_HESSIAN_OP: return "hessian";
            case ESHKOL_TAYLOR_OP: return "taylor";
            case ESHKOL_WITH_REGION_OP: return "with-region";
            case ESHKOL_VALUES_OP: return "values";
            case ESHKOL_LET_VALUES_OP: return "let-values";
            case ESHKOL_DEFINE_OP: return "define";
            case ESHKOL_IMPORT_OP: return "import";
            case ESHKOL_REQUIRE_OP: return "require";
            case ESHKOL_PROVIDE_OP: return "provide";
            case ESHKOL_DEFINE_SYNTAX_OP: return "define-syntax";
            case ESHKOL_SEQUENCE_OP: return "begin";
            default: return "unsupported-form";
        }
    }

    /** @brief Is this a differentiation operator whose operand's regions must
     *         be emitted forward-and-backward together? */
    static bool isDifferentiation(eshkol_op_t op) {
        switch (op) {
            case ESHKOL_GRADIENT_OP:
            case ESHKOL_DERIVATIVE_OP:
            case ESHKOL_JACOBIAN_OP:
            case ESHKOL_HESSIAN_OP:
            case ESHKOL_DIVERGENCE_OP:
            case ESHKOL_CURL_OP:
            case ESHKOL_LAPLACIAN_OP:
            case ESHKOL_DIRECTIONAL_DERIV_OP:
            case ESHKOL_TAYLOR_OP:
            case ESHKOL_DERIVATIVE_N_OP:
            case ESHKOL_DIFF_OP:
                return true;
            default:
                return false;
        }
    }

    /**
     * @brief Is @p node device-eligible, with every break it contains
     *        appended to @p breaks?
     *
     * A node is eligible when every one of its children is; the walk therefore
     * returns false as soon as it records a break, but it records the break
     * FIRST, and it keeps walking siblings, because a report that stopped at
     * the first break would name one break out of five.
     */
    bool eligible(const eshkol_ast_t* node, std::vector<GraphBreak>* breaks,
                  std::set<std::string>& active) {
        if (!node) return false;

        switch (node->type) {
            case ESHKOL_DOUBLE:
            case ESHKOL_INT8: case ESHKOL_INT16: case ESHKOL_INT32: case ESHKOL_INT64:
            case ESHKOL_UINT8: case ESHKOL_UINT16: case ESHKOL_UINT32: case ESHKOL_UINT64:
            case ESHKOL_BOOL:
                return true;                    // a fixed-width scalar literal
            case ESHKOL_TENSOR:
                return true;                    // a tensor literal, shape and all
            case ESHKOL_STRING:
            case ESHKOL_CHAR:
            case ESHKOL_SYMBOL:
            case ESHKOL_BIGNUM_LITERAL:
            case ESHKOL_CONS:
            case ESHKOL_NULL:
                addBreak(breaks, node, node->type == ESHKOL_STRING ? "string-literal"
                          : node->type == ESHKOL_CHAR ? "char-literal"
                          : node->type == ESHKOL_SYMBOL ? "symbol-literal"
                          : node->type == ESHKOL_BIGNUM_LITERAL ? "bignum-literal"
                          : "list-literal",
                         BreakReason::HostValueDomain, "", BuiltinLabel::Unclassified);
                return false;
            case ESHKOL_VAR: {
                if (isHostBinding(node->variable.id)) {
                    addBreak(breaks, node, node->variable.id ? node->variable.id : "variable",
                             BreakReason::HostValueDomain, "", BuiltinLabel::Unclassified);
                    return false;
                }
                return true;                    // a leaf: an input to the region
            }
            case ESHKOL_OP:
                break;
            default:
                addBreak(breaks, node, "unsupported-value", BreakReason::HostValueDomain,
                         "", BuiltinLabel::Unclassified);
                return false;
        }

        const eshkol_operations_t& op = node->operation;
        switch (op.op) {
            // The arithmetic operators the parser gives their own tag. They
            // are `+`/`-`/`*`/`/` by another name, and are lowered as such.
            case ESHKOL_ADD_OP:
            case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP:
            case ESHKOL_DIV_OP:
            case ESHKOL_TENSOR_OP:
            case ESHKOL_CALL_OP:
            case ESHKOL_IF_OP:
                return eligibleCall(node, breaks, active);

            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                // A `let` is binding, not control flow: condition 3 does not
                // exclude it, and every binding it introduces is an SSA name
                // inside the region. A NAMED let is a loop, and the contract
                // admits only a tail-recursive loop over fragment-typed state,
                // which this pass does not yet prove; so it breaks and says so.
                if (op.let_op.name) {
                    addBreak(breaks, node, "named-let", BreakReason::NonAdmittedConstruct,
                             "", BuiltinLabel::Unclassified,
                             "a named let is a loop whose state this pass does not yet "
                             "prove fragment-typed");
                    return false;
                }
                bool ok = true;
                pushScope();
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const eshkol_ast_t* b = &op.let_op.bindings[i];
                    const char* name = bindingName(b);
                    const eshkol_ast_t* value = bindingValue(b);
                    bool this_ok = eligible(value, breaks, active);
                    if (!this_ok) ok = false;
                    if (name) {
                        shape_bindings_.back()[name] = shapeOf(value);
                        if (!this_ok) host_bindings_.back().insert(name);
                    }
                }
                if (!eligible(op.let_op.body, breaks, active)) ok = false;
                popScope();
                return ok;
            }

            case ESHKOL_SEQUENCE_OP: {
                // (begin a b) is eligible only when every expression is: a
                // begin exists to sequence effects, and an effect is exactly
                // what condition 4 excludes.
                bool ok = op.sequence_op.num_expressions > 0;
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    if (!eligible(&op.sequence_op.expressions[i], breaks, active)) ok = false;
                return ok;
            }

            case ESHKOL_QUOTE_OP:
            case ESHKOL_QUASIQUOTE_OP:
                // A quoted datum is a value question, not a control-flow one:
                // what `quote` yields is a symbol or a list, and condition 1
                // excludes both. Reporting it as a non-admitted construct
                // would send a reader to condition 3, which has nothing to
                // say about it.
                addBreak(breaks, node, formName(op.op), BreakReason::HostValueDomain,
                         "", BuiltinLabel::Unclassified);
                return false;

            default: {
                // A special form whose keyword is ALSO a classified builtin
                // (`gradient`, `jacobian`, `values`, ...) reports the label
                // the classification table gives it, because that table is
                // where the decision was actually made. Saying "not admitted
                // by condition 3" about `gradient` would be true of the form
                // and silent about the reason, which is that the AD tape is a
                // host value.
                const char* form = formName(op.op);
                BuiltinLabel label = labelOf(form);
                if (label == BuiltinLabel::Host || label == BuiltinLabel::HostWithDeviceInner) {
                    addBreak(breaks, node, form,
                             label == BuiltinLabel::Host ? BreakReason::HostBuiltin
                                                         : BreakReason::HostWithDeviceInner,
                             form, label);
                    return false;
                }
                addBreak(breaks, node, form, BreakReason::NonAdmittedConstruct,
                         "", BuiltinLabel::Unclassified,
                         std::string("`") + form + "` is not one of them");
                return false;
            }
        }
    }

    /** @brief Eligibility of a call-shaped node (call, if, cond, and the
     *         arithmetic operator tags, which share call_op's layout). */
    bool eligibleCall(const eshkol_ast_t* node, std::vector<GraphBreak>* breaks,
                      std::set<std::string>& active) {
        const eshkol_operations_t& op = node->operation;

        if (op.op == ESHKOL_TENSOR_OP) {
            // A tensor literal built from expressions. Eligible when every
            // element is; its shape is written in the node.
            bool ok = true;
            for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i)
                if (!eligible(&op.tensor_op.elements[i], breaks, active)) ok = false;
            return ok;
        }

        const uint64_t argc = op.call_op.num_vars;
        bool args_ok = true;
        for (uint64_t i = 0; i < argc; ++i)
            if (!eligible(&op.call_op.variables[i], breaks, active)) args_ok = false;

        if (op.op == ESHKOL_IF_OP || isIfCall(op)) {
            // Condition 3 admits `if` with both arms in the fragment, and it
            // lowers to stablehlo.case. The REGION EMITTER does not emit
            // stablehlo.case yet, so admitting a conditional here would form a
            // region that cannot be compiled — the failure would move from a
            // reported graph break to a compile error inside the emitter,
            // which is the wrong end of this stage's contract. Same rule as
            // any other unlowered op, and it becomes eligible the day
            // region_execution.cpp emits the case.
            //
            // isIfCall() is here because a conditional does not always reach
            // this pass tagged ESHKOL_IF_OP: forms the parser rewrites into a
            // conditional arrive as an ordinary call whose callee is the name
            // `if`.
            (void)args_ok;
            addBreak(breaks, node, "if", BreakReason::NoLowering, "if",
                     BuiltinLabel::Device);
            return false;
        }
        if (op.op == ESHKOL_ADD_OP || op.op == ESHKOL_SUB_OP ||
            op.op == ESHKOL_MUL_OP || op.op == ESHKOL_DIV_OP) {
            const char* name = op.op == ESHKOL_ADD_OP ? "+"
                             : op.op == ESHKOL_SUB_OP ? "-"
                             : op.op == ESHKOL_MUL_OP ? "*" : "/";
            if (!hasLowering(name, argc)) {
                addBreak(breaks, node, name, BreakReason::NoLowering, name,
                         BuiltinLabel::Device);
                return false;
            }
            return args_ok;
        }

        const char* callee = calleeName(op.call_op.func);
        if (!callee) {
            // A computed callee: the value in the operator position is a
            // closure, which condition 1 excludes as a value.
            addBreak(breaks, node, "computed-callee", BreakReason::HostValueDomain,
                     "", BuiltinLabel::Unclassified);
            return false;
        }

        // A call into a top-level definition of this module.
        if (bodies_.count(callee)) {
            if (!functionEligible(callee, active)) {
                addBreak(breaks, node, callee,
                         active.count(callee) ? BreakReason::RecursiveCall
                                              : BreakReason::HostFunction,
                         "", BuiltinLabel::Unclassified);
                return false;
            }
            return args_ok;
        }

        const BuiltinLabel label = labelOf(callee);
        switch (label) {
            case BuiltinLabel::Device:
                if (!hasLowering(callee, argc)) {
                    addBreak(breaks, node, callee, BreakReason::NoLowering, callee, label);
                    return false;
                }
                return args_ok;
            case BuiltinLabel::Host:
                addBreak(breaks, node, callee, BreakReason::HostBuiltin, callee, label);
                return false;
            case BuiltinLabel::HostWithDeviceInner:
                addBreak(breaks, node, callee, BreakReason::HostWithDeviceInner, callee, label);
                return false;
            case BuiltinLabel::Unclassified:
                addBreak(breaks, node, callee, BreakReason::UnknownFunction, callee, label);
                return false;
        }
        return false;
    }

    // ── shapes ───────────────────────────────────────────────────────────

    /**
     * @brief The static shape of @p node, where the frontend knows it.
     *
     * Only two things settle a shape without running the program: a tensor
     * literal writes its dimensions into the node, and a variable bound to
     * something whose shape is known inherits it. Elementwise ops preserve
     * their operand's shape; a matmul's is the two operands' outer dims.
     * Everything else is honestly unknown, and an unknown shape is REPORTED
     * as unknown rather than guessed: the region is then keyed by the shape
     * signature it is first entered with, which is the shape-specialisation
     * cache the executable cache already implements.
     */
    RegionShape shapeOf(const eshkol_ast_t* node) {
        RegionShape unknown;
        if (!node) return unknown;
        if (node->type == ESHKOL_TENSOR) {
            RegionShape s;
            s.known = true;
            for (uint64_t i = 0; i < node->tensor_val.num_dimensions; ++i)
                s.dims.push_back(static_cast<int64_t>(node->tensor_val.dimensions[i]));
            return s;
        }
        if (node->type == ESHKOL_DOUBLE || node->type == ESHKOL_INT64 ||
            node->type == ESHKOL_INT32 || node->type == ESHKOL_BOOL) {
            RegionShape s; s.known = true; return s;   // rank 0
        }
        if (node->type == ESHKOL_VAR) return shapeOfBinding(node->variable.id);
        if (node->type != ESHKOL_OP) return unknown;

        const eshkol_operations_t& op = node->operation;
        if (op.op == ESHKOL_TENSOR_OP) {
            RegionShape s;
            s.known = true;
            for (uint64_t i = 0; i < op.tensor_op.num_dimensions; ++i)
                s.dims.push_back(static_cast<int64_t>(op.tensor_op.dimensions[i]));
            return s;
        }
        if (op.op == ESHKOL_ADD_OP || op.op == ESHKOL_SUB_OP ||
            op.op == ESHKOL_MUL_OP || op.op == ESHKOL_DIV_OP) {
            // Elementwise: the wider operand's shape, which for equal shapes
            // is that shape and for a broadcast is the non-scalar one.
            RegionShape best;
            for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                RegionShape s = shapeOf(&op.call_op.variables[i]);
                if (s.known && s.dims.size() >= best.dims.size()) best = s;
            }
            return best;
        }
        if (op.op != ESHKOL_CALL_OP) return unknown;
        const char* callee = calleeName(op.call_op.func);
        if (!callee) return unknown;

        const uint64_t argc = op.call_op.num_vars;
        if (std::strcmp(callee, "tensor-matmul") == 0 ||
            std::strcmp(callee, "matmul") == 0 ||
            std::strcmp(callee, "tensor-dot") == 0) {
            if (argc != 2) return unknown;
            RegionShape a = shapeOf(&op.call_op.variables[0]);
            RegionShape b = shapeOf(&op.call_op.variables[1]);
            if (!a.known || !b.known || a.dims.size() != 2 || b.dims.size() != 2)
                return unknown;
            RegionShape s; s.known = true;
            s.dims = {a.dims[0], b.dims[1]};
            return s;
        }
        if (std::strcmp(callee, "tensor-transpose") == 0 ||
            std::strcmp(callee, "transpose") == 0) {
            if (argc < 1) return unknown;
            RegionShape a = shapeOf(&op.call_op.variables[0]);
            if (!a.known || a.dims.size() != 2) return unknown;
            RegionShape s; s.known = true;
            s.dims = {a.dims[1], a.dims[0]};
            return s;
        }
        if (std::strcmp(callee, "tensor-sum") == 0 ||
            std::strcmp(callee, "tensor-mean") == 0 ||
            std::strcmp(callee, "tensor-reduce-all") == 0) {
            if (argc != 1) return unknown;
            RegionShape a = shapeOf(&op.call_op.variables[0]);
            if (!a.known) return unknown;
            RegionShape s; s.known = true; return s;    // reduced to a scalar
        }
        // Every other lowered builtin this pass admits is shape-preserving in
        // its first operand: the elementwise unary and binary set, the
        // activations, and the compositions in device_lowering_table.yaml.
        if (argc >= 1 && hasLowering(callee, argc)) {
            RegionShape best;
            for (uint64_t i = 0; i < argc; ++i) {
                RegionShape s = shapeOf(&op.call_op.variables[i]);
                if (s.known && s.dims.size() >= best.dims.size()) best = s;
            }
            return best;
        }
        return unknown;
    }

    // ── region collection ────────────────────────────────────────────────

    /** @brief The device operations inside an eligible subtree, in evaluation
     *         order (operands before the operator that consumes them). */
    void collectOps(const eshkol_ast_t* node, std::vector<std::string>* ops,
                    std::set<std::string>& seen_functions) {
        if (!node) return;
        if (node->type != ESHKOL_OP) return;
        const eshkol_operations_t& op = node->operation;
        switch (op.op) {
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP: case ESHKOL_DIV_OP: {
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectOps(&op.call_op.variables[i], ops, seen_functions);
                ops->push_back(op.op == ESHKOL_ADD_OP ? "+"
                             : op.op == ESHKOL_SUB_OP ? "-"
                             : op.op == ESHKOL_MUL_OP ? "*" : "/");
                return;
            }
            case ESHKOL_IF_OP: {
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectOps(&op.call_op.variables[i], ops, seen_functions);
                ops->push_back("if");
                return;
            }
            case ESHKOL_TENSOR_OP:
                for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i)
                    collectOps(&op.tensor_op.elements[i], ops, seen_functions);
                return;
            case ESHKOL_CALL_OP: {
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectOps(&op.call_op.variables[i], ops, seen_functions);
                if (isIfCall(op)) { ops->push_back("if"); return; }
                const char* callee = calleeName(op.call_op.func);
                if (!callee) return;
                auto body = bodies_.find(callee);
                if (body != bodies_.end()) {
                    // A call into an eligible top-level function contributes
                    // the ops of its BODY, because that is what the region
                    // actually contains once the call is inlined into it.
                    if (seen_functions.insert(callee).second) {
                        collectOps(body->second, ops, seen_functions);
                        seen_functions.erase(callee);
                    }
                    return;
                }
                ops->push_back(callee);
                return;
            }
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i)
                    collectOps(bindingValue(&op.let_op.bindings[i]), ops, seen_functions);
                collectOps(op.let_op.body, ops, seen_functions);
                return;
            }
            case ESHKOL_SEQUENCE_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    collectOps(&op.sequence_op.expressions[i], ops, seen_functions);
                return;
            default:
                return;
        }
    }

    /** @brief The free variables of an eligible subtree: the values the region
     *         takes from the host. */
    void collectInputs(const eshkol_ast_t* node, std::vector<std::string>* bound,
                       std::vector<RegionInput>* inputs) {
        if (!node) return;
        if (node->type == ESHKOL_VAR) {
            const char* id = node->variable.id;
            if (!id) return;
            if (std::find(bound->begin(), bound->end(), id) != bound->end()) return;
            for (const RegionInput& in : *inputs) if (in.name == id) return;
            RegionInput in;
            in.name = id;
            in.shape = shapeOfBinding(id);
            inputs->push_back(std::move(in));
            return;
        }
        if (node->type != ESHKOL_OP) return;
        const eshkol_operations_t& op = node->operation;
        switch (op.op) {
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP: case ESHKOL_DIV_OP:
            case ESHKOL_IF_OP:
            case ESHKOL_CALL_OP:
                // The callee position is a name, not a value: a call to `tanh`
                // does not make `tanh` an input.
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i)
                    collectInputs(&op.call_op.variables[i], bound, inputs);
                return;
            case ESHKOL_TENSOR_OP:
                for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i)
                    collectInputs(&op.tensor_op.elements[i], bound, inputs);
                return;
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP: {
                size_t depth = bound->size();
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const eshkol_ast_t* b = &op.let_op.bindings[i];
                    collectInputs(bindingValue(b), bound, inputs);
                    if (const char* n = bindingName(b)) bound->push_back(n);
                }
                collectInputs(op.let_op.body, bound, inputs);
                bound->resize(depth);
                return;
            }
            case ESHKOL_SEQUENCE_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    collectInputs(&op.sequence_op.expressions[i], bound, inputs);
                return;
            default:
                return;
        }
    }

    /** @brief Record @p node as a region. Returns false when the subtree holds
     *         no device operation at all — a bare variable or literal is
     *         eligible but is not worth a host-to-device transfer, and calling
     *         it a region would inflate every count in the report. */
    bool formRegion(const eshkol_ast_t* node) {
        std::vector<std::string> ops;
        std::set<std::string> seen;
        collectOps(node, &ops, seen);
        if (ops.empty()) return false;

        Region r;
        r.id = static_cast<int>(unit_->regions.size());
        r.line = node->line;
        r.column = node->column;
        r.ops = std::move(ops);
        std::vector<std::string> bound;
        collectInputs(node, &bound, &r.inputs);
        r.result = shapeOf(node);
        r.inside_gradient = gradient_depth_ > 0;
        r.root = node;

        std::ostringstream sig;
        sig << (unit_->unit.empty() ? "<toplevel>" : unit_->unit) << "#" << r.id << ":";
        for (size_t i = 0; i < r.inputs.size(); ++i) {
            if (i) sig << ",";
            sig << r.inputs[i].shape.text();
        }
        sig << "->" << r.result.text();
        r.shape_signature = sig.str();

        report_.total_device_ops += static_cast<int>(r.ops.size());
        unit_->regions.push_back(std::move(r));
        report_.total_regions++;
        return true;
    }

    // ── the outlining walk ───────────────────────────────────────────────

    /**
     * @brief Outline @p node.
     *
     * If the node is eligible it is a REGION ROOT and the walk does not
     * descend: that is what makes the region maximal, and it is why no
     * eligible node can be left on the host — an eligible node's ancestors
     * are walked before it, and the first eligible one on the path becomes
     * the root.
     *
     * If it is not, its breaks are recorded and the walk continues into its
     * children, so that a print's argument, a map's procedure body, or a
     * differentiated function's body can each hold regions of their own.
     */
    void outline(const eshkol_ast_t* node) {
        if (!node) return;

        std::vector<GraphBreak> breaks;
        std::set<std::string> active;
        if (eligible(node, &breaks, active)) {
            if (formRegion(node)) return;
            // Eligible but with no device operation in it (a bare variable, a
            // literal). Nothing to outline and nothing to report.
            return;
        }
        // RULING: a LEAF is data, not a construct, so it is never a graph
        // break of its own. A string literal makes `(display "x")` ineligible
        // — that is what stops the region — but the break to report is the
        // display, and reporting the literal beside it would name the same
        // boundary twice and put a "break" on every argument of every host
        // call in the program. A leaf can still be the REASON a break was
        // recorded at the construct above it; that is recorded there.
        if (node->type != ESHKOL_OP) return;
        for (GraphBreak& b : breaks) {
            // Only the breaks AT this node belong here; the breaks of a child
            // are recorded when the walk reaches that child. Recording the
            // whole set here would report every break once per ancestor.
            if (b.node == node) {
                unit_->breaks.push_back(std::move(b));
                report_.total_breaks++;
            }
        }
        descend(node);
    }

    /** @brief Walk into a node's children, outlining each. */
    void descend(const eshkol_ast_t* node) {
        if (!node || node->type != ESHKOL_OP) return;
        const eshkol_operations_t& op = node->operation;

        if (isDifferentiation(op.op)) {
            // The tape is host, but the function it differentiates is not
            // necessarily: a region found in here must be emitted forward and
            // backward in one module (the S3 VJP machinery), never executed
            // forward-only and left out of the tape.
            gradient_depth_++;
            switch (op.op) {
                case ESHKOL_GRADIENT_OP:
                    outline(op.gradient_op.function);
                    outline(op.gradient_op.point);
                    break;
                case ESHKOL_DERIVATIVE_OP:
                    outline(op.derivative_op.function);
                    outline(op.derivative_op.point);
                    break;
                case ESHKOL_JACOBIAN_OP:
                case ESHKOL_HESSIAN_OP:
                    outline(op.gradient_op.function);
                    outline(op.gradient_op.point);
                    break;
                case ESHKOL_TAYLOR_OP:
                case ESHKOL_DERIVATIVE_N_OP:
                    outline(op.taylor_op.function);
                    outline(op.taylor_op.point);
                    break;
                case ESHKOL_DIFF_OP:
                    outline(op.diff_op.expression);
                    break;
                default:
                    break;
            }
            gradient_depth_--;
            return;
        }

        switch (op.op) {
            case ESHKOL_ADD_OP: case ESHKOL_SUB_OP:
            case ESHKOL_MUL_OP: case ESHKOL_DIV_OP:
            case ESHKOL_IF_OP:
            case ESHKOL_COND_OP:
            case ESHKOL_CALL_OP: {
                // A `host-with-device-inner` builtin NAMES the inner
                // evaluation that is still eligible: "the argument
                // procedure's body, when that body is itself in Eshkol-S".
                // So a lambda in that position is not a break — it is the
                // sanctioned boundary — and what gets outlined is its body.
                // Under any other callee a lambda is a closure in value
                // position and breaks as one.
                const char* callee = calleeName(op.call_op.func);
                const bool inner_ok =
                    callee && labelOf(callee) == BuiltinLabel::HostWithDeviceInner;
                for (uint64_t i = 0; i < op.call_op.num_vars; ++i) {
                    const eshkol_ast_t* arg = &op.call_op.variables[i];
                    if (inner_ok && arg->type == ESHKOL_OP &&
                        arg->operation.op == ESHKOL_LAMBDA_OP) {
                        pushScope();
                        for (uint64_t k = 0; k < arg->operation.lambda_op.num_params; ++k) {
                            const eshkol_ast_t* prm = &arg->operation.lambda_op.parameters[k];
                            if (prm->type == ESHKOL_VAR && prm->variable.id)
                                shape_bindings_.back()[prm->variable.id] = RegionShape{};
                        }
                        outline(arg->operation.lambda_op.body);
                        popScope();
                        continue;
                    }
                    outline(arg);
                }
                return;
            }
            case ESHKOL_TENSOR_OP:
                for (uint64_t i = 0; i < op.tensor_op.total_elements; ++i)
                    outline(&op.tensor_op.elements[i]);
                return;
            case ESHKOL_SEQUENCE_OP:
            case ESHKOL_AND_OP:
            case ESHKOL_OR_OP:
            case ESHKOL_WHEN_OP:
            case ESHKOL_UNLESS_OP:
                for (uint64_t i = 0; i < op.sequence_op.num_expressions; ++i)
                    outline(&op.sequence_op.expressions[i]);
                return;
            case ESHKOL_LET_OP:
            case ESHKOL_LET_STAR_OP:
            case ESHKOL_LETREC_OP:
            case ESHKOL_LETREC_STAR_OP: {
                pushScope();
                for (uint64_t i = 0; i < op.let_op.num_bindings; ++i) {
                    const eshkol_ast_t* b = &op.let_op.bindings[i];
                    const char* name = bindingName(b);
                    const eshkol_ast_t* value = bindingValue(b);
                    outline(value);
                    if (name) {
                        shape_bindings_.back()[name] = shapeOf(value);
                        std::vector<GraphBreak> discard;
                        std::set<std::string> active;
                        if (!eligible(value, &discard, active))
                            host_bindings_.back().insert(name);
                    }
                }
                outline(op.let_op.body);
                popScope();
                return;
            }
            case ESHKOL_LAMBDA_OP:
                pushScope();
                for (uint64_t i = 0; i < op.lambda_op.num_params; ++i) {
                    const eshkol_ast_t* p = &op.lambda_op.parameters[i];
                    if (p->type == ESHKOL_VAR && p->variable.id)
                        shape_bindings_.back()[p->variable.id] = RegionShape{};
                }
                outline(op.lambda_op.body);
                popScope();
                return;
            case ESHKOL_DEFINE_OP:
                outline(op.define_op.value);
                return;
            case ESHKOL_SET_OP:
                outline(op.set_op.value);
                return;
            case ESHKOL_WITH_REGION_OP:
                for (uint64_t i = 0; i < op.with_region_op.num_body_exprs; ++i)
                    outline(&op.with_region_op.body[i]);
                return;
            default:
                // A form whose children this pass cannot reach. The break has
                // already been reported by outline(); saying so here as well
                // would double-count it, and descending into memory this pass
                // does not understand the layout of would be worse than not
                // descending.
                return;
        }
    }

    // ── entry ────────────────────────────────────────────────────────────

    void analyze(const eshkol_ast_t* form) {
        UnitReport unit;
        unit.unit = "<toplevel>";
        if (form && form->type == ESHKOL_OP &&
            form->operation.op == ESHKOL_DEFINE_OP &&
            form->operation.define_op.name) {
            unit.unit = form->operation.define_op.name;
        }
        report_.units.push_back(std::move(unit));
        unit_ = &report_.units.back();

        pushScope();
        if (form && form->type == ESHKOL_OP &&
            form->operation.op == ESHKOL_DEFINE_OP) {
            const auto& d = form->operation.define_op;
            for (uint64_t i = 0; i < d.num_params; ++i) {
                const eshkol_ast_t* p = &d.parameters[i];
                if (p->type == ESHKOL_VAR && p->variable.id)
                    shape_bindings_.back()[p->variable.id] = RegionShape{};
            }
            outline(d.value);
        } else {
            outline(form);
        }
        popScope();
        unit_ = nullptr;
    }
};

// ─────────────────────────────────────────────────────────────────────────
// Public surface
// ─────────────────────────────────────────────────────────────────────────

RegionFormation::RegionFormation(RegionFormationOptions options)
    : impl_(new Impl(std::move(options))) {}

RegionFormation::~RegionFormation() { delete impl_; }

void RegionFormation::setProgram(const std::string& path) {
    impl_->report_.program = path;
}

void RegionFormation::declare(const eshkol_ast_t* form) { impl_->declare(form); }

void RegionFormation::analyze(const eshkol_ast_t* form) { impl_->analyze(form); }

const ModuleReport& RegionFormation::report() const { return impl_->report_; }

const std::map<std::string, RegionFunction>& RegionFormation::functions() const {
    return impl_->functions_;
}

namespace {

/** @brief JSON string escaping. Only the characters JSON requires. */
std::string jsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

void writeShape(std::ostringstream& o, const RegionShape& s) {
    if (!s.known) { o << "\"?\""; return; }
    o << "[";
    for (size_t i = 0; i < s.dims.size(); ++i) {
        if (i) o << ",";
        o << s.dims[i];
    }
    o << "]";
}

} // namespace

std::string RegionFormation::toJson() const {
    const ModuleReport& r = impl_->report_;
    std::ostringstream o;
    o << "{\n";
    o << "  \"version\": 1,\n";
    o << "  \"program\": \"" << jsonEscape(r.program) << "\",\n";
    o << "  \"units\": [\n";
    for (size_t u = 0; u < r.units.size(); ++u) {
        const UnitReport& unit = r.units[u];
        o << "    {\n";
        o << "      \"unit\": \"" << jsonEscape(unit.unit) << "\",\n";
        o << "      \"regions\": [\n";
        for (size_t i = 0; i < unit.regions.size(); ++i) {
            const Region& reg = unit.regions[i];
            o << "        {\"id\": " << reg.id
              << ", \"line\": " << reg.line
              << ", \"inside_gradient\": " << (reg.inside_gradient ? "true" : "false")
              << ", \"ops\": [";
            for (size_t k = 0; k < reg.ops.size(); ++k) {
                if (k) o << ", ";
                o << "\"" << jsonEscape(reg.ops[k]) << "\"";
            }
            o << "], \"inputs\": [";
            for (size_t k = 0; k < reg.inputs.size(); ++k) {
                if (k) o << ", ";
                o << "{\"name\": \"" << jsonEscape(reg.inputs[k].name) << "\", \"shape\": ";
                writeShape(o, reg.inputs[k].shape);
                o << "}";
            }
            o << "], \"result_shape\": ";
            writeShape(o, reg.result);
            o << ", \"shape_signature\": \"" << jsonEscape(reg.shape_signature) << "\"}";
            if (i + 1 < unit.regions.size()) o << ",";
            o << "\n";
        }
        o << "      ],\n";
        o << "      \"breaks\": [\n";
        for (size_t i = 0; i < unit.breaks.size(); ++i) {
            const GraphBreak& b = unit.breaks[i];
            o << "        {\"line\": " << b.line
              << ", \"construct\": \"" << jsonEscape(b.construct) << "\""
              << ", \"reason\": \"" << breakReasonName(b.reason) << "\"";
            if (!b.builtin.empty()) {
                o << ", \"builtin\": \"" << jsonEscape(b.builtin) << "\""
                  << ", \"label\": \"" << jsonEscape(b.label) << "\"";
            }
            o << ", \"reason_text\": \"" << jsonEscape(b.reason_text) << "\"}";
            if (i + 1 < unit.breaks.size()) o << ",";
            o << "\n";
        }
        o << "      ]\n";
        o << "    }";
        if (u + 1 < r.units.size()) o << ",";
        o << "\n";
    }
    o << "  ],\n";
    o << "  \"totals\": {\"regions\": " << r.total_regions
      << ", \"device_ops\": " << r.total_device_ops
      << ", \"breaks\": " << r.total_breaks << "}\n";
    o << "}\n";
    return o.str();
}

void RegionFormation::writeTrace(std::ostream& out) const {
    const ModuleReport& r = impl_->report_;
    for (const UnitReport& unit : r.units) {
        for (const Region& reg : unit.regions) {
            out << "region " << reg.id << ": " << reg.ops.size()
                << " ops on device (";
            for (size_t i = 0; i < reg.ops.size(); ++i) {
                if (i) out << ", ";
                out << reg.ops[i];
            }
            out << ") in " << unit.unit << " at line " << reg.line
                << ", result " << reg.result.text();
            if (reg.inside_gradient) out << ", inside a differentiated expression";
            out << "\n";
        }
        for (const GraphBreak& b : unit.breaks) {
            out << "break at line " << b.line << ": (" << b.construct << " ...) is "
                << (b.builtin.empty() ? breakReasonName(b.reason) : b.label.c_str());
            if (!b.builtin.empty()) out << " (label: " << b.label << ")";
            out << " -- " << b.reason_text << "\n";
        }
    }
    out << "regions: " << r.total_regions << ", device ops: " << r.total_device_ops
        << ", breaks: " << r.total_breaks << "\n";
}

bool RegionFormation::writeReportFile(std::string* error) const {
    if (impl_->options_.report_path.empty()) return true;
    std::ofstream out(impl_->options_.report_path.c_str());
    if (!out) {
        if (error) *error = "cannot open " + impl_->options_.report_path + " for writing";
        return false;
    }
    out << toJson();
    if (!out) {
        if (error) *error = "write to " + impl_->options_.report_path + " failed";
        return false;
    }
    return true;
}

} // namespace xla
} // namespace eshkol
