/*
 * Per-builtin device/host parity, driven by the lowering table.
 *
 * WHAT THIS IS FOR.
 *
 * tests/xla/op_parity_test covers the ops that have an eshkol_xla_* C runtime
 * entry point — about forty builtins. The other two hundred device-classified
 * builtins have their host implementation compiled into the language, with no
 * C function a harness can call, so there was no reference to diff a lowering
 * against and no honest way to grade one.
 *
 * This closes that. scripts/gen_builtin_parity_plan.py runs the REAL builtin
 * through eshkol-run with the device switch off and records what it printed;
 * this program builds the StableHLO graph that
 * lib/backend/xla/device_lowering_table.yaml says the builtin lowers to,
 * executes it through PJRT, and compares.
 *
 * THE TWO SIDES ARE DIFFERENT CODE PATHS, WHICH IS THE ENTIRE POINT. The
 * reference is the language's own compiled implementation; the device value is
 * a StableHLO graph on an accelerator. Nothing computes both. A reference
 * obtained by evaluating the lowering in double precision here would agree
 * with the device however wrong the lowering was, and every row would be
 * worthless — which is the exact failure the honesty contract at the top of
 * scripts/run_xla_gate.sh exists to prevent.
 *
 * The scalar-kind rows go further: the host applies the builtin to each input
 * separately, N independent scalar calls, while the device evaluates one
 * rank-1 graph over all N at once. The two do not even agree on the shape of
 * the work.
 *
 * TWO ANTI-VACUITY GUARDS, both of which can fail:
 *   - the shared comparator control in tests/xla/parity_compare.h, run before
 *     any device is required;
 *   - a degenerate-reference check. A row whose reference is the same value
 *     for every input proves nothing: it would pass for any device graph that
 *     happened to be constant. This was not hypothetical — the first `%` row
 *     written for this table had inputs whose remainder was 1.0 twelve times
 *     over, and would have passed for a device computing a - 2b. Such a row is
 *     reported DEGENERATE and does not count as covered.
 *
 * Exit: 0 every row agreed; 1 a row disagreed or a control failed; 77 no PJRT
 * device was reachable.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "eshkol/backend/xla/device_lowering.h"
#include "eshkol/backend/xla/stablehlo_emitter.h"
#include "eshkol/backend/xla/xla_runtime.h"
#include "eshkol/backend/xla/xla_types.h"

#include "parity_compare.h"

using eshkol::xla::BinaryOp;
using eshkol::xla::BufferDescriptor;
using eshkol::xla::BufferElementType;
using eshkol::xla::DeviceExecutor;
using eshkol::xla::ElementType;
using eshkol::xla::ExecutionResult;
using eshkol::xla::getDefaultRuntime;
using eshkol::xla::registerStableHLODeviceExecutor;
using eshkol::xla::StableHLOEmitter;
using eshkol::xla::UnaryOp;
using eshkol::xla::XLARuntime;

using eshkol_parity::Comparison;
using eshkol_parity::compareArrays;
using eshkol_parity::ToleranceClass;

namespace {

struct Step {
    std::string op;        // "unary" | "binary" | "const" | "reduce"
    std::string kind;      // enumerator name, for unary/binary/reduce
    std::string out;
    std::string in0, in1;  // operands; in0 doubles as `like` for const
    double value = 0.0;    // const only
    std::vector<int64_t> axes;  // reduce axes, or broadcast result shape
    std::vector<int64_t> dims;  // broadcast dimension map
    std::string dir;            // compare direction
    int64_t dim = 0;            // iota dimension
    std::string to;             // convert target element type
};

struct Entry {
    std::string name;
    ToleranceClass tolerance_class = ToleranceClass::Arithmetic;
    int n = 0;
    // A reduction collapses its operand, so the result is a rank-0 scalar and
    // the reference is one number rather than n. The plan says which.
    bool scalar_result = false;
    // The element type the graph is built over. Integer builtins cannot be
    // evaluated in f32: bitwise-and of two floats is not a question StableHLO
    // will answer, and rounding an integer through f32 would lose exactness
    // above 2^24 while looking fine below it.
    bool integer_graph = false;
    // Result length, when it is not the input length. A generator like arange
    // has no operands at all, so n cannot stand in for it.
    int result_len = -1;
    // Set only for a builtin whose answer IS a constant — `ones`, `zeros`.
    // The degeneracy guard asks whether the reference varies, and for these
    // the constancy is the semantics rather than a badly chosen input set.
    // The exemption is per-entry and the table must say why, so it cannot be
    // used to quiet a row that is degenerate by accident; the row is also
    // printed as the weaker thing it is.
    bool constant_by_definition = false;
    std::vector<std::vector<double>> inputs;
    std::vector<double> reference;
    std::string noref;     // non-empty when no reference could be obtained
    std::vector<Step> steps;
    std::string result;
};

bool parseUnary(const std::string& s, UnaryOp* out) {
    static const std::map<std::string, UnaryOp> m = {
        {"Abs", UnaryOp::Abs}, {"Negate", UnaryOp::Negate}, {"Sqrt", UnaryOp::Sqrt},
        {"Rsqrt", UnaryOp::Rsqrt}, {"Cbrt", UnaryOp::Cbrt}, {"Exp", UnaryOp::Exp},
        {"Expm1", UnaryOp::Expm1}, {"Log", UnaryOp::Log}, {"Log1p", UnaryOp::Log1p},
        {"Logistic", UnaryOp::Logistic}, {"Sin", UnaryOp::Sin}, {"Cos", UnaryOp::Cos},
        {"Tan", UnaryOp::Tan}, {"Tanh", UnaryOp::Tanh}, {"Floor", UnaryOp::Floor},
        {"Ceil", UnaryOp::Ceil}, {"RoundNearestAfz", UnaryOp::RoundNearestAfz},
        {"RoundNearestEven", UnaryOp::RoundNearestEven}, {"Sign", UnaryOp::Sign},
        {"IsFinite", UnaryOp::IsFinite},
        {"Not", UnaryOp::Not}, {"PopulationCount", UnaryOp::PopulationCount},
        {"CountLeadingZeros", UnaryOp::CountLeadingZeros},
    };
    auto it = m.find(s);
    if (it == m.end()) return false;
    *out = it->second;
    return true;
}

bool parseReduce(const std::string& s, eshkol::xla::StableHLOOp* out) {
    if (s == "Sum")  { *out = eshkol::xla::StableHLOOp::REDUCE_SUM;  return true; }
    if (s == "Max")  { *out = eshkol::xla::StableHLOOp::REDUCE_MAX;  return true; }
    if (s == "Min")  { *out = eshkol::xla::StableHLOOp::REDUCE_MIN;  return true; }
    if (s == "Prod") { *out = eshkol::xla::StableHLOOp::REDUCE_PROD; return true; }
    return false;
}

bool parseBinary(const std::string& s, BinaryOp* out) {
    static const std::map<std::string, BinaryOp> m = {
        {"Add", BinaryOp::Add}, {"Subtract", BinaryOp::Subtract},
        {"Multiply", BinaryOp::Multiply}, {"Divide", BinaryOp::Divide},
        {"Power", BinaryOp::Power}, {"Remainder", BinaryOp::Remainder},
        {"Maximum", BinaryOp::Maximum}, {"Minimum", BinaryOp::Minimum},
        {"Atan2", BinaryOp::Atan2},
        {"And", BinaryOp::And}, {"Or", BinaryOp::Or}, {"Xor", BinaryOp::Xor},
        {"ShiftLeft", BinaryOp::ShiftLeft},
        {"ShiftRightLogical", BinaryOp::ShiftRightLogical},
        {"ShiftRightArithmetic", BinaryOp::ShiftRightArithmetic},
    };
    auto it = m.find(s);
    if (it == m.end()) return false;
    *out = it->second;
    return true;
}

/** @brief Read the plan the generator wrote. Line-oriented on purpose. */
bool loadPlan(const std::string& path, std::vector<Entry>* out, std::string* error) {
    std::ifstream in(path);
    if (!in) { *error = "cannot open plan file " + path; return false; }
    std::string line;
    Entry cur;
    bool open = false;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "BUILTIN") {
            cur = Entry();
            ls >> cur.name;
            open = true;
        } else if (!open) {
            continue;
        } else if (tag == "CLASS") {
            std::string c; ls >> c;
            cur.tolerance_class = (c == "transcendental") ? ToleranceClass::Transcendental
                                                          : ToleranceClass::Arithmetic;
        } else if (tag == "N") {
            ls >> cur.n;
        } else if (tag == "ETYPE") {
            std::string t; ls >> t;
            cur.integer_graph = (t == "s64");
        } else if (tag == "CONSTREF") {
            cur.constant_by_definition = true;
        } else if (tag == "RLEN") {
            ls >> cur.result_len;
        } else if (tag == "RSHAPE") {
            std::string r; ls >> r;
            cur.scalar_result = (r == "scalar");
        } else if (tag == "IN") {
            int idx = 0; ls >> idx;
            std::vector<double> vs; double v;
            while (ls >> v) vs.push_back(v);
            if (static_cast<int>(cur.inputs.size()) <= idx) cur.inputs.resize(idx + 1);
            cur.inputs[idx] = vs;
        } else if (tag == "REF") {
            double v;
            while (ls >> v) cur.reference.push_back(v);
        } else if (tag == "NOREF") {
            std::string rest;
            std::getline(ls, rest);
            cur.noref = rest.empty() ? "no reference" : rest.substr(1);
        } else if (tag == "OP") {
            Step st;
            ls >> st.op;
            if (st.op == "unary") {
                ls >> st.kind >> st.out >> st.in0;
            } else if (st.op == "binary") {
                ls >> st.kind >> st.out >> st.in0 >> st.in1;
            } else if (st.op == "const") {
                ls >> st.out >> st.value >> st.in0;
            } else if (st.op == "reduce") {
                ls >> st.kind >> st.out >> st.in0;
                int64_t ax;
                while (ls >> ax) st.axes.push_back(ax);
            } else if (st.op == "iota") {
                ls >> st.out >> st.dim;
                int64_t d;
                while (ls >> d) st.axes.push_back(d);
            } else if (st.op == "compare") {
                ls >> st.dir >> st.out >> st.in0 >> st.in1;
            } else if (st.op == "convert") {
                ls >> st.to >> st.out >> st.in0;
            } else if (st.op == "broadcast") {
                // "out in <shape...> | <dims...>" — the bar keeps the two
                // integer lists apart without needing a real parser.
                ls >> st.out >> st.in0;
                std::string tok;
                bool after_bar = false;
                while (ls >> tok) {
                    if (tok == "|") { after_bar = true; continue; }
                    const int64_t v = std::strtoll(tok.c_str(), nullptr, 10);
                    if (after_bar) st.dims.push_back(v); else st.axes.push_back(v);
                }
            } else {
                *error = "unknown lowering step '" + st.op + "' in " + cur.name;
                return false;
            }
            cur.steps.push_back(st);
        } else if (tag == "RESULT") {
            ls >> cur.result;
        } else if (tag == "END") {
            out->push_back(cur);
            open = false;
        }
    }
    return true;
}

/**
 * @brief True when every reference value is identical — see the header.
 *
 * A SCALAR-RESULT ROW IS EXEMPT, and the caller passes scalar_result to say
 * so. The guard asks whether the reference VARIES across the inputs, and a
 * reduction produces one number by construction, so there is no variation to
 * find and "constant" would be true of every correct row. Such a row's
 * discriminating power is different in kind: the single value is a specific
 * number that a wrong graph does not reproduce — mse-loss over these inputs
 * is one particular double, not any double.
 */
bool referenceIsDegenerate(const std::vector<double>& ref, bool scalar_result) {
    if (scalar_result) return false;
    if (ref.size() < 2) return true;
    for (size_t i = 1; i < ref.size(); ++i) {
        if (ref[i] != ref[0]) return false;
    }
    return true;
}

ElementType g_elem = ElementType::F32;
std::string g_dtype = "f32";

/**
 * @brief Build, compile and run one entry's graph, returning host f64 values.
 *
 * Every operand is a rank-1 tensor of N elements: the table's lowerings are
 * elementwise, so the shape carries no information and one dimension keeps the
 * plan format from having to describe shapes it does not need yet.
 */
bool runOnDevice(const Entry& e, std::vector<double>* out, std::string* error) {
    StableHLOEmitter emitter;
    if (!emitter.isAvailable()) { *error = "StableHLO emitter unavailable"; return false; }

    const std::vector<int64_t> shape = {static_cast<int64_t>(e.n)};
    // An integer graph is built over i64 whatever the device float type is:
    // the exactness is the point, and Eshkol's integers are 64-bit.
    const ElementType elem = e.integer_graph ? ElementType::I64 : g_elem;
    std::vector<StableHLOEmitter::ParamSpec> params;
    for (size_t i = 0; i < e.inputs.size(); ++i) {
        params.push_back(StableHLOEmitter::ParamSpec{shape, elem});
    }
    std::vector<void*> args = emitter.beginFunction("main", params);
    if (args.size() != params.size()) { *error = "beginFunction failed"; return false; }

    // `elem` is in scope for the step loop below: iota names its own element
    // type, and it must be the graph's, not a second opinion about it.
    std::map<std::string, void*> env;
    for (size_t i = 0; i < args.size(); ++i) env["a" + std::to_string(i)] = args[i];

    for (const Step& st : e.steps) {
        auto lookup = [&](const std::string& n, void** v) {
            auto it = env.find(n);
            if (it == env.end()) return false;
            *v = it->second;
            return true;
        };
        void* result = nullptr;
        if (st.op == "unary") {
            void* x = nullptr;
            if (!lookup(st.in0, &x)) { *error = "undefined value " + st.in0; return false; }
            UnaryOp op;
            if (!parseUnary(st.kind, &op)) { *error = "unknown unary op " + st.kind; return false; }
            result = emitter.emitUnary(op, x);
        } else if (st.op == "binary") {
            void* a = nullptr; void* b = nullptr;
            if (!lookup(st.in0, &a) || !lookup(st.in1, &b)) {
                *error = "undefined operand in " + st.op + " " + st.kind;
                return false;
            }
            BinaryOp op;
            if (!parseBinary(st.kind, &op)) { *error = "unknown binary op " + st.kind; return false; }
            result = emitter.emitBinary(op, a, b);
        } else if (st.op == "const") {
            void* like = nullptr;
            if (!lookup(st.in0, &like)) { *error = "undefined value " + st.in0; return false; }
            result = emitter.emitConstantLike(like, st.value);
        } else if (st.op == "iota") {
            result = emitter.emitIota(st.axes, st.dim, elem);
        } else if (st.op == "compare") {
            void* a = nullptr; void* b2 = nullptr;
            if (!lookup(st.in0, &a) || !lookup(st.in1, &b2)) {
                *error = "undefined operand in compare"; return false;
            }
            eshkol::xla::ComparisonDirection dir;
            if      (st.dir == "EQ") dir = eshkol::xla::ComparisonDirection::EQ;
            else if (st.dir == "NE") dir = eshkol::xla::ComparisonDirection::NE;
            else if (st.dir == "LT") dir = eshkol::xla::ComparisonDirection::LT;
            else if (st.dir == "LE") dir = eshkol::xla::ComparisonDirection::LE;
            else if (st.dir == "GT") dir = eshkol::xla::ComparisonDirection::GT;
            else if (st.dir == "GE") dir = eshkol::xla::ComparisonDirection::GE;
            else { *error = "unknown compare direction " + st.dir; return false; }
            result = emitter.emitCompare(a, b2, dir);
        } else if (st.op == "convert") {
            void* x = nullptr;
            if (!lookup(st.in0, &x)) { *error = "undefined value " + st.in0; return false; }
            ElementType target;
            if      (st.to == "f32")  target = ElementType::F32;
            else if (st.to == "f64")  target = ElementType::F64;
            else if (st.to == "s64")  target = ElementType::I64;
            else if (st.to == "s32")  target = ElementType::I32;
            else if (st.to == "bool") target = ElementType::BOOL;
            else { *error = "unknown convert target " + st.to; return false; }
            result = emitter.emitConvert(x, target);
        } else if (st.op == "broadcast") {
            void* x = nullptr;
            if (!lookup(st.in0, &x)) { *error = "undefined value " + st.in0; return false; }
            result = emitter.emitBroadcastInDim(x, st.axes, st.dims);
        } else if (st.op == "reduce") {
            void* x = nullptr;
            if (!lookup(st.in0, &x)) { *error = "undefined value " + st.in0; return false; }
            eshkol::xla::StableHLOOp op;
            if (!parseReduce(st.kind, &op)) { *error = "unknown reduce kind " + st.kind; return false; }
            result = emitter.emitReduce(x, st.axes, op);
        }
        if (!result) { *error = "emit failed for step producing " + st.out; return false; }
        env[st.out] = result;
    }

    auto it = env.find(e.result);
    if (it == env.end()) { *error = "result " + e.result + " was never produced"; return false; }
    if (!emitter.endFunction({it->second})) { *error = "endFunction failed"; return false; }

    const std::string module_text = emitter.serializeToString();
    if (module_text.empty()) { *error = "empty module"; return false; }

    XLARuntime& rt = getDefaultRuntime();
    void* exe = rt.compileStableHLO(module_text, error);
    if (!exe) return false;

    const bool narrow = (!e.integer_graph && g_elem != ElementType::F64);
    std::vector<std::vector<float>> staged(narrow ? e.inputs.size() : 0);
    std::vector<std::vector<int64_t>> staged_i(e.integer_graph ? e.inputs.size() : 0);
    std::vector<BufferDescriptor> inputs(e.inputs.size());
    for (size_t i = 0; i < e.inputs.size(); ++i) {
        inputs[i].shape = shape;
        inputs[i].on_device = false;
        if (e.integer_graph) {
            staged_i[i].resize(static_cast<size_t>(e.n));
            for (int j = 0; j < e.n; ++j) {
                staged_i[i][j] = static_cast<int64_t>(llround(e.inputs[i][j]));
            }
            inputs[i].data = staged_i[i].data();
            inputs[i].element_size = sizeof(int64_t);
            inputs[i].elem = BufferElementType::S64;
        } else if (narrow) {
            staged[i].resize(static_cast<size_t>(e.n));
            for (int j = 0; j < e.n; ++j) staged[i][j] = static_cast<float>(e.inputs[i][j]);
            inputs[i].data = staged[i].data();
            inputs[i].element_size = sizeof(float);
            inputs[i].elem = BufferElementType::F32;
        } else {
            inputs[i].data = const_cast<double*>(e.inputs[i].data());
            inputs[i].element_size = sizeof(double);
            inputs[i].elem = BufferElementType::F64;
        }
    }

    const int result_n = e.scalar_result ? 1
                       : (e.result_len > 0 ? e.result_len : e.n);
    const std::vector<int64_t> result_shape =
        e.scalar_result ? std::vector<int64_t>{}
                        : std::vector<int64_t>{static_cast<int64_t>(result_n)};
    out->assign(static_cast<size_t>(result_n), 0.0);
    std::vector<float> result_staged;
    std::vector<int64_t> result_staged_i;
    std::vector<BufferDescriptor> outputs(1);
    outputs[0].shape = result_shape;
    outputs[0].on_device = false;
    if (e.integer_graph) {
        result_staged_i.resize(static_cast<size_t>(result_n));
        outputs[0].data = result_staged_i.data();
        outputs[0].element_size = sizeof(int64_t);
        outputs[0].elem = BufferElementType::S64;
    } else if (narrow) {
        result_staged.resize(static_cast<size_t>(result_n));
        outputs[0].data = result_staged.data();
        outputs[0].element_size = sizeof(float);
        outputs[0].elem = BufferElementType::F32;
    } else {
        outputs[0].data = out->data();
        outputs[0].element_size = sizeof(double);
        outputs[0].elem = BufferElementType::F64;
    }

    ExecutionResult exec = rt.execute(exe, inputs, outputs);
    rt.releaseExecutable(exe);
    if (!exec.success) { *error = exec.error_message; return false; }
    if (e.integer_graph) {
        for (int j = 0; j < result_n; ++j) (*out)[j] = static_cast<double>(result_staged_i[j]);
    } else if (narrow) {
        for (int j = 0; j < result_n; ++j) (*out)[j] = static_cast<double>(result_staged[j]);
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string plan_path =
        argc > 1 ? argv[1] : ".scratch/xla_gate/builtin_parity_plan.txt";

    std::cout << "=========================================" << std::endl;
    std::cout << "  Per-Builtin Parity (device vs host)" << std::endl;
    std::cout << "=========================================" << std::endl;

    ::setenv("ESHKOL_XLA_PJRT", "1", 1);

    if (!eshkol_parity::test_comparator_rejects_a_perturbed_result()) {
        std::cerr << "The comparator control failed; no row below would mean anything."
                  << std::endl;
        return 1;
    }

    std::vector<Entry> entries;
    std::string error;
    if (!loadPlan(plan_path, &entries, &error)) {
        std::cerr << "FAIL: " << error << std::endl;
        std::cerr << "Run scripts/gen_builtin_parity_plan.py first." << std::endl;
        return 1;
    }
    if (entries.empty()) {
        std::cerr << "FAIL: the plan contained no builtins" << std::endl;
        return 1;
    }

    DeviceExecutor* executor = registerStableHLODeviceExecutor();
    std::string why;
    if (!executor || !executor->available(&why)) {
        std::cout << "SKIP: no PJRT device available: " << why << std::endl;
        return 77;
    }

    g_dtype = executor->dtypeName();
    g_elem = (g_dtype == "f64") ? ElementType::F64 : ElementType::F32;
    // One source for the bounds, shared with op_parity_test: two harnesses
    // that computed them separately would drift the first time one was revised.
    eshkol_parity::setTolerancesForDtype(g_dtype);
    const double tol_arith = eshkol_parity::toleranceFor(ToleranceClass::Arithmetic);
    const double tol_trans = eshkol_parity::toleranceFor(ToleranceClass::Transcendental);

    std::cout << "Device: " << executor->description() << std::endl;
    std::cout << "Plan: " << plan_path << " (" << entries.size() << " builtins)" << std::endl;
    std::cout << "Tolerance (" << g_dtype << ", absolute or relative, whichever is looser; "
                 "docs/design/ESHKOL_S_FRAGMENT.md): arithmetic=" << tol_arith
              << " transcendental=" << tol_trans << std::endl;

    std::printf("\n%-16s %-6s %-15s %-12s %-12s %-9s %s\n",
                "builtin", "dtype", "class", "max abs err", "max rel err", "tol", "result");
    std::printf("%-16s %-6s %-15s %-12s %-12s %-9s %s\n",
                "----------------", "------", "---------------", "------------",
                "------------", "---------", "------");

    int passed = 0, failed = 0;
    std::vector<std::string> covered, uncovered;

    for (const Entry& e : entries) {
        const double tol = (e.tolerance_class == ToleranceClass::Transcendental)
                               ? tol_trans : tol_arith;
        const char* cls = (e.tolerance_class == ToleranceClass::Transcendental)
                               ? "transcendental" : "arithmetic";

        if (!e.noref.empty()) {
            std::printf("%-16s %-6s %-15s %-12s %-12s %-9s NO REFERENCE (%s)\n",
                        e.name.c_str(), g_dtype.c_str(), cls, "-", "-", "-", e.noref.c_str());
            uncovered.push_back(e.name);
            failed++;
            continue;
        }
        if (referenceIsDegenerate(e.reference,
                                  e.scalar_result || e.constant_by_definition)) {
            std::printf("%-16s %-6s %-15s %-12s %-12s %-9s DEGENERATE (reference is "
                        "constant across every input; the row could not fail)\n",
                        e.name.c_str(), g_dtype.c_str(), cls, "-", "-", "-");
            uncovered.push_back(e.name);
            failed++;
            continue;
        }

        std::vector<double> device;
        std::string run_error;
        if (!runOnDevice(e, &device, &run_error)) {
            std::printf("%-16s %-6s %-15s %-12s %-12s %-9s FAIL (device: %s)\n",
                        e.name.c_str(), g_dtype.c_str(), cls, "-", "-", "-",
                        run_error.c_str());
            uncovered.push_back(e.name);
            failed++;
            continue;
        }

        Comparison cmp = compareArrays(device, e.reference, tol);
        std::printf("%-16s %-6s %-15s %-12.3e %-12.3e %-9.1e %s\n",
                    e.name.c_str(), g_dtype.c_str(), cls, cmp.max_abs, cmp.max_rel, tol,
                    cmp.agreed ? (e.constant_by_definition
                                      ? "PASS (constant by definition)" : "PASS")
                               : "FAIL");
        if (cmp.agreed) {
            passed++;
            covered.push_back(e.name);
        } else {
            failed++;
            uncovered.push_back(e.name);
            const int i = cmp.worst_index;
            if (i >= 0) {
                std::printf("       first disagreement at index %d: device=%.17g host=%.17g\n",
                            i, device[static_cast<size_t>(i)],
                            e.reference[static_cast<size_t>(i)]);
            }
        }
    }

    std::cout << std::endl;
    std::cout << "Rows passed: " << passed << std::endl;
    std::cout << "Rows failed: " << failed << std::endl;
    std::cout << "SUMMARY: rows_passed=" << passed << " rows_failed=" << failed
              << " dtype=" << g_dtype
              << " tol_arithmetic=" << tol_arith
              << " tol_transcendental=" << tol_trans << std::endl;
    std::cout << "COVERED_BUILTINS:";
    for (const std::string& b : covered) std::cout << " " << b;
    std::cout << std::endl;
    std::cout << "UNCOVERED_BUILTINS:";
    for (const std::string& b : uncovered) std::cout << " " << b;
    std::cout << std::endl;

    return failed == 0 ? 0 : 1;
}
