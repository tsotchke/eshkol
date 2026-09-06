/*
 * tests/xla/region_formation_test.cpp — grades the region-formation pass
 * against the corpus in tests/xla/regions/.
 *
 * WHAT THIS MEASURES, AND WHY IT IS NOT A SNAPSHOT TEST.
 *
 * Every corpus program has a sibling <name>.expected.json stating, from the
 * Eshkol-S contract, how many regions the program should have, which device
 * operations belong to each, and which graph breaks should be reported with
 * which construct. Those files were written from the contract, NOT from this
 * pass's output. That is the whole point: a file regenerated from the
 * implementation grades nothing, because the implementation always agrees
 * with itself. A disagreement here is a finding on one side or the other, and
 * which side is wrong is a judgement someone has to make.
 *
 * Two criteria are graded, and they are separate because they fail
 * separately:
 *
 *   outlines_maximal  — the regions the pass formed are the ones the contract
 *                       says are there: same count, same ops, same order.
 *                       Nothing eligible was left on the host, and nothing
 *                       ineligible was swept into a region.
 *   breaks_reported   — every break the contract predicts is reported, with
 *                       the construct that caused it, and no break is
 *                       reported that the contract does not predict. A break
 *                       nobody reports is the silent fallback this whole
 *                       stage exists to make impossible.
 *
 * Copyright (C) tsotchke
 * SPDX-License-Identifier: MIT
 */

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <dirent.h>

#include "eshkol/eshkol.h"
#include "eshkol/backend/xla/region_formation.h"
#include "eshkol/backend/xla/region_execution.h"
#include "eshkol/backend/xla/xla_runtime.h"

#include "parity_compare.h"
// arena_memory.h before region_host_reference.h: the latter reads
// eshkol_tensor_t out of the pointers the *_host entry points return, and
// that struct is defined here.
#include "../../lib/core/arena_memory.h"
#include "region_host_reference.h"

// The host implementations, under the names they carry since the device split
// in lib/backend/xla/xla_runtime.cpp. These are the reference the device's
// region result is compared against, and they are a genuinely different code
// path: BLAS and SIMD on the host against one fused StableHLO module on the
// device. Evaluating the same formula twice in double precision would agree
// however wrong both were.
extern "C" {
void* eshkol_xla_matmul_host(void* arena, const double* a, const double* b,
                             const int64_t* a_shape, const int64_t* b_shape,
                             int64_t a_rank, int64_t b_rank);
void* eshkol_xla_elementwise_host(void* arena, const double* a, const double* b,
                                  int64_t total, const uint64_t* shape, int64_t rank,
                                  int64_t b_total, const uint64_t* b_shape, int64_t b_rank,
                                  int64_t op_code);
int eshkol_xla_elementwise_is_binary(int64_t op_code);
void* eshkol_xla_reduce_host(void* arena, const double* data, int64_t total,
                             const uint64_t* shape, int64_t rank, int64_t axis,
                             int64_t op_code);
void* eshkol_xla_transpose_host(void* arena, const double* data, const uint64_t* shape,
                                int64_t rank, const int64_t* perm);
}

using namespace eshkol::xla;

#include "region_host_reference.inc"

namespace {

// ─────────────────────────────────────────────────────────────────────────
// A minimal reader for the expected-regions files.
//
// Deliberately not a general JSON parser: it reads exactly the schema the
// expected files use, and rejects anything else loudly. A lenient reader here
// would let a typo in an expectation ("regons") silently expect nothing,
// which reads as a pass.
// ─────────────────────────────────────────────────────────────────────────

struct ExpectedRegion {
    std::vector<std::string> ops;
    int inputs = -1;                 ///< -1 when the file does not state it
    bool inside_gradient = false;
};

struct ExpectedBreak {
    std::string construct;
    std::string reason;
    std::string builtin;
};

struct Expectation {
    std::string program;
    std::string intent;
    std::vector<ExpectedRegion> regions;
    std::vector<ExpectedBreak> breaks;
    std::vector<std::string> rulings;
};

class JsonReader {
public:
    explicit JsonReader(const std::string& text) : s_(text) {}

    bool error() const { return !error_.empty(); }
    const std::string& errorText() const { return error_; }

    void skipSpace() {
        while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
    }
    bool expect(char c) {
        skipSpace();
        if (i_ < s_.size() && s_[i_] == c) { ++i_; return true; }
        fail(std::string("expected '") + c + "'");
        return false;
    }
    bool peek(char c) {
        skipSpace();
        return i_ < s_.size() && s_[i_] == c;
    }
    std::string readString() {
        skipSpace();
        if (i_ >= s_.size() || s_[i_] != '"') { fail("expected a string"); return ""; }
        ++i_;
        std::string out;
        while (i_ < s_.size() && s_[i_] != '"') {
            if (s_[i_] == '\\' && i_ + 1 < s_.size()) {
                ++i_;
                char c = s_[i_++];
                switch (c) {
                    case 'n': out += '\n'; break;
                    case 't': out += '\t'; break;
                    case 'r': out += '\r'; break;
                    default: out += c; break;
                }
            } else {
                out += s_[i_++];
            }
        }
        if (i_ >= s_.size()) { fail("unterminated string"); return ""; }
        ++i_;
        return out;
    }
    long readNumber() {
        skipSpace();
        size_t start = i_;
        if (i_ < s_.size() && (s_[i_] == '-' || s_[i_] == '+')) ++i_;
        while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) ++i_;
        if (start == i_) { fail("expected a number"); return 0; }
        return std::strtol(s_.substr(start, i_ - start).c_str(), nullptr, 10);
    }
    bool readBool() {
        skipSpace();
        if (s_.compare(i_, 4, "true") == 0) { i_ += 4; return true; }
        if (s_.compare(i_, 5, "false") == 0) { i_ += 5; return false; }
        fail("expected true or false");
        return false;
    }
    void fail(const std::string& why) {
        if (error_.empty()) {
            std::ostringstream o;
            o << why << " at offset " << i_;
            error_ = o.str();
        }
    }

private:
    const std::string& s_;
    size_t i_ = 0;
    std::string error_;
};

bool readExpectation(const std::string& path, Expectation* out, std::string* error) {
    std::ifstream in(path.c_str());
    if (!in) { *error = "cannot open " + path; return false; }
    std::ostringstream buf;
    buf << in.rdbuf();
    std::string text = buf.str();

    JsonReader r(text);
    if (!r.expect('{')) { *error = path + ": " + r.errorText(); return false; }
    bool first = true;
    while (!r.peek('}')) {
        if (!first && !r.expect(',')) break;
        first = false;
        std::string key = r.readString();
        if (r.error()) break;
        if (!r.expect(':')) break;
        if (key == "program") {
            out->program = r.readString();
        } else if (key == "intent") {
            out->intent = r.readString();
        } else if (key == "rulings") {
            // Which rulings in tests/xla/regions/README.md this file was
            // revised under. Read and ignored for grading; it exists so that
            // a changed expectation carries the reason it changed, next to
            // the expectation itself.
            if (!r.expect('[')) break;
            bool fr = true;
            while (!r.peek(']')) {
                if (!fr && !r.expect(',')) break;
                fr = false;
                out->rulings.push_back(r.readString());
                if (r.error()) break;
            }
            if (!r.expect(']')) break;
        } else if (key == "regions") {
            if (!r.expect('[')) break;
            bool f2 = true;
            while (!r.peek(']')) {
                if (!f2 && !r.expect(',')) break;
                f2 = false;
                ExpectedRegion reg;
                if (!r.expect('{')) break;
                bool f3 = true;
                while (!r.peek('}')) {
                    if (!f3 && !r.expect(',')) break;
                    f3 = false;
                    std::string k = r.readString();
                    if (!r.expect(':')) break;
                    if (k == "ops") {
                        if (!r.expect('[')) break;
                        bool f4 = true;
                        while (!r.peek(']')) {
                            if (!f4 && !r.expect(',')) break;
                            f4 = false;
                            reg.ops.push_back(r.readString());
                            if (r.error()) break;
                        }
                        if (!r.expect(']')) break;
                    } else if (k == "inputs") {
                        reg.inputs = static_cast<int>(r.readNumber());
                    } else if (k == "inside_gradient") {
                        reg.inside_gradient = r.readBool();
                    } else {
                        *error = path + ": unknown region key \"" + k + "\"";
                        return false;
                    }
                    if (r.error()) break;
                }
                if (!r.expect('}')) break;
                out->regions.push_back(std::move(reg));
            }
            if (!r.expect(']')) break;
        } else if (key == "breaks") {
            if (!r.expect('[')) break;
            bool f2 = true;
            while (!r.peek(']')) {
                if (!f2 && !r.expect(',')) break;
                f2 = false;
                ExpectedBreak b;
                if (!r.expect('{')) break;
                bool f3 = true;
                while (!r.peek('}')) {
                    if (!f3 && !r.expect(',')) break;
                    f3 = false;
                    std::string k = r.readString();
                    if (!r.expect(':')) break;
                    if (k == "construct") b.construct = r.readString();
                    else if (k == "reason") b.reason = r.readString();
                    else if (k == "builtin") b.builtin = r.readString();
                    else { *error = path + ": unknown break key \"" + k + "\""; return false; }
                    if (r.error()) break;
                }
                if (!r.expect('}')) break;
                out->breaks.push_back(std::move(b));
            }
            if (!r.expect(']')) break;
        } else {
            *error = path + ": unknown key \"" + key + "\"";
            return false;
        }
        if (r.error()) break;
    }
    if (r.error()) { *error = path + ": " + r.errorText(); return false; }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────
// Running the pass over one program
// ─────────────────────────────────────────────────────────────────────────

/** @brief Parse every top-level form of @p path.
 *
 *  The forms are held in a deque rather than a vector because the pass keeps
 *  pointers into them (a region names the subtree it replaces) and a vector
 *  moves its elements when it grows.
 */
bool parseProgram(const std::string& path, std::vector<eshkol_ast_t*>* forms,
                  std::string* error) {
    std::ifstream in(path.c_str());
    if (!in) { *error = "cannot open " + path; return false; }
    eshkol_reset_parse_line_counter();
    eshkol_reset_parse_errors();
    eshkol_set_parse_source_context(path.c_str());
    for (;;) {
        eshkol_ast_t* node = new eshkol_ast_t();
        *node = eshkol_parse_next_ast(in);
        if (node->type == ESHKOL_INVALID) { delete node; break; }
        forms->push_back(node);
        if (forms->size() > 4096) { *error = path + ": more than 4096 top-level forms"; return false; }
    }
    if (eshkol_parse_had_error()) { *error = path + ": parse error"; return false; }
    if (forms->empty()) { *error = path + ": no top-level forms"; return false; }
    return true;
}

struct Mismatch {
    std::string program;
    std::string detail;
};

/** @brief Flatten a module report into one region list and one break list, in
 *         source order, which is the shape the expectation states. */
void flatten(const ModuleReport& r, std::vector<const Region*>* regions,
             std::vector<const GraphBreak*>* breaks) {
    for (const UnitReport& u : r.units) {
        for (const Region& reg : u.regions) regions->push_back(&reg);
        for (const GraphBreak& b : u.breaks) breaks->push_back(&b);
    }
}

std::string joinOps(const std::vector<std::string>& ops) {
    std::string out;
    for (size_t i = 0; i < ops.size(); ++i) {
        if (i) out += ", ";
        out += ops[i];
    }
    return out;
}


// ─────────────────────────────────────────────────────────────────────────
// result_parity
//
// For each region the pass formed whose input shapes are all statically
// known, the region is executed TWICE over the same deterministic inputs:
// once as one fused StableHLO module on the PJRT device, once through the
// host runtime's own entry points. The two are compared with
// parity_compare.h's tolerance classes.
//
// This is not yet the whole-program comparison the criterion asks for: an
// outlined region is not bound back into the program, so there is no "run the
// program with regions on". What it does measure is the part that can be
// wrong numerically — the fused module against the host — over every region
// of the corpus.
// ─────────────────────────────────────────────────────────────────────────

struct ParityRow {
    std::string program;
    std::string unit;
    int region_id = 0;
    size_t ops = 0;
    std::string result_shape;
    bool ran = false;
    bool passed = false;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    std::string note;
};

/** @brief Deterministic inputs, one distinct series per operand position, so
 *         that swapping two operands changes the answer. Values are multiples
 *         of 0.125 and stay inside the domain of sqrt, log and atanh. */
std::vector<double> inputsFor(size_t index, int64_t n) {
    const double base = 0.25 + 0.125 * static_cast<double>(index);
    return eshkol_parity::makeData(n, base, 0.125);
}

bool measureRegionParity(const Region& region,
                         const std::map<std::string, RegionFunction>& functions,
                         arena_t* arena, ParityRow* row) {
    std::vector<std::vector<int64_t>> shapes;
    for (const RegionInput& in : region.inputs) {
        if (!in.shape.known) { row->note = "input shape not static"; return false; }
        shapes.push_back(in.shape.dims);
    }
    if (shapes.empty()) { row->note = "region takes no inputs"; return false; }

    std::vector<std::vector<double>> data;
    for (size_t i = 0; i < shapes.size(); ++i)
        data.push_back(inputsFor(i, eshkol_parity::numElements(shapes[i])));

    // Host first: if the host reference cannot be produced there is nothing to
    // compare against, and running the device anyway would print a number with
    // no verdict attached to it.
    eshkol_region_host::Evaluator host(arena, functions);
    for (size_t i = 0; i < shapes.size(); ++i) {
        eshkol_region_host::HostVal v;
        v.shape = shapes[i];
        v.data = data[i];
        v.ok = true;
        host.bind(region.inputs[i].name, v);
    }
    std::string host_error;
    eshkol_region_host::HostVal expected = host.eval(region.root, nullptr, &host_error);
    if (!expected.ok) { row->note = "host reference: " + host_error; return false; }

    RegionExecutor exec(functions);
    std::vector<RegionOperand> operands;
    for (size_t i = 0; i < shapes.size(); ++i)
        operands.push_back(RegionOperand{shapes[i], data[i].data()});

    std::vector<double> device(expected.data.size(), 0.0);
    std::vector<int64_t> device_shape;
    std::string error;
    if (!exec.execute(region, operands, device.data(), &device_shape, &error)) {
        row->note = "device: " + error;
        return false;
    }
    if (device_shape != expected.shape) {
        row->note = "device returned " + eshkol_parity::shapeText(device_shape) +
                    " and the host returned " + eshkol_parity::shapeText(expected.shape);
        row->ran = true;
        return false;
    }

    // Every corpus region contains at least one approximated elementary
    // function or is pure arithmetic; the class is taken as transcendental
    // when any op in it is one, because a chain is no more accurate than its
    // loosest link.
    eshkol_parity::ToleranceClass cls = eshkol_parity::ToleranceClass::Arithmetic;
    for (const std::string& op : region.ops) {
        if (op == "tanh" || op == "exp" || op == "log" || op == "sin" || op == "cos" ||
            op == "sqrt" || op == "tensor-sqrt" || op == "sigmoid" || op == "atanh" ||
            op == "tensor-exp" || op == "tensor-log" || op == "softmax") {
            cls = eshkol_parity::ToleranceClass::Transcendental;
            break;
        }
    }
    eshkol_parity::Comparison cmp =
        eshkol_parity::compareArrays(device, expected.data, eshkol_parity::toleranceFor(cls));

    row->ran = true;
    row->passed = cmp.agreed;
    row->worst_abs = cmp.max_abs;
    row->worst_rel = cmp.max_rel;
    row->result_shape = eshkol_parity::shapeText(device_shape);
    row->ops = region.ops.size();
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string dir = "tests/xla/regions";
    std::string report_dir;
    bool measure_parity = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--corpus" && i + 1 < argc) dir = argv[++i];
        else if (a == "--report-dir" && i + 1 < argc) report_dir = argv[++i];
        else if (a == "--parity") measure_parity = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--corpus DIR] [--report-dir DIR] [--parity]\n",
                         argv[0]);
            return 2;
        }
    }

    std::vector<std::string> programs;
    {
        DIR* d = opendir(dir.c_str());
        if (!d) {
            std::fprintf(stderr, "cannot open corpus directory %s\n", dir.c_str());
            return 2;
        }
        while (struct dirent* e = readdir(d)) {
            std::string name = e->d_name;
            if (name.size() > 4 && name.compare(name.size() - 4, 4, ".esk") == 0)
                programs.push_back(dir + "/" + name);
        }
        closedir(d);
        std::sort(programs.begin(), programs.end());
    }
    if (programs.empty()) {
        std::fprintf(stderr, "no *.esk programs in %s\n", dir.c_str());
        return 2;
    }

    arena_t* g_arena = nullptr;
    if (measure_parity) {
        // Ask for device execution before anything latches the answer:
        // deviceExecutionRequested() reads this variable exactly once, on
        // first use. Set here rather than left to the caller so the harness
        // measures what it says it measures however it is invoked.
        ::setenv("ESHKOL_XLA_PJRT", "1", 1);

        // The comparator control, before any row: a comparator that cannot
        // reject a perturbed result would make every PASS below meaningless.
        if (!eshkol_parity::test_comparator_rejects_a_perturbed_result()) {
            std::fprintf(stderr, "the comparator control failed; no parity row would mean anything\n");
            return 1;
        }
        g_arena = arena_create(4 * 1024 * 1024);
        if (!g_arena) { std::fprintf(stderr, "arena_create failed\n"); return 2; }
        // Installing the executor is what makes a device reachable at all; a
        // run without it would report every region as "no device" rather than
        // silently computing on the host.
        registerStableHLODeviceExecutor();
        std::string why;
        DeviceExecutor* d = deviceExecutor();
        if (!d || !d->available(&why)) {
            std::printf("result_parity: NOT MEASURED (no device: %s)\n",
                        d ? why.c_str() : "no executor installed");
            measure_parity = false;
        } else {
            eshkol_parity::setTolerancesForDtype(d->dtypeName());
            std::printf("device: %s, computing in %s\n\n",
                        d->description().c_str(), d->dtypeName().c_str());
        }
    }

    std::printf("Region formation over %zu corpus programs in %s\n\n",
                programs.size(), dir.c_str());
    std::printf("%-34s %8s %8s %8s %8s  %s\n",
                "program", "regions", "expect", "breaks", "expect", "verdict");
    std::printf("%s\n", std::string(96, '-').c_str());

    std::vector<ParityRow> parity_rows;
    std::vector<Mismatch> outline_failures;
    std::vector<Mismatch> break_failures;
    std::set<std::string> outline_bad_programs;
    std::set<std::string> break_bad_programs;
    int programs_ok = 0;

    for (const std::string& program : programs) {
        std::string base = program.substr(0, program.size() - 4);
        std::string expected_path = base + ".expected.json";

        Expectation expect;
        std::string error;
        if (!readExpectation(expected_path, &expect, &error)) {
            std::printf("%-34s %8s %8s %8s %8s  EXPECTATION UNREADABLE\n",
                        program.c_str(), "-", "-", "-", "-");
            outline_failures.push_back({program, error});
            break_failures.push_back({program, error});
            continue;
        }

        std::vector<eshkol_ast_t*> forms;
        if (!parseProgram(program, &forms, &error)) {
            std::printf("%-34s %8s %8s %8s %8s  PARSE FAILED\n",
                        program.c_str(), "-", "-", "-", "-");
            outline_failures.push_back({program, error});
            break_failures.push_back({program, error});
            continue;
        }

        // The environment is the source of the trace and report settings, so
        // that a corpus run and a compiler run are configured the same way;
        // --report-dir only overrides where the JSON goes.
        RegionFormationOptions options = regionOptionsFromEnvironment();
        if (!report_dir.empty()) {
            std::string leaf = base.substr(base.find_last_of('/') + 1);
            options.report_path = report_dir + "/" + leaf + ".report.json";
        }
        RegionFormation pass(options);
        pass.setProgram(program);
        for (eshkol_ast_t* f : forms) pass.declare(f);
        for (eshkol_ast_t* f : forms) pass.analyze(f);
        if (!options.report_path.empty()) {
            std::string werr;
            if (!pass.writeReportFile(&werr))
                std::fprintf(stderr, "  (report not written: %s)\n", werr.c_str());
        }

        if (options.trace) pass.writeTrace(std::cerr);

        std::vector<const Region*> regions;
        std::vector<const GraphBreak*> breaks;
        flatten(pass.report(), &regions, &breaks);

        // ── outlines_maximal ──
        std::vector<std::string> outline_problems;
        if (regions.size() != expect.regions.size()) {
            std::ostringstream o;
            o << "formed " << regions.size() << " regions, contract says "
              << expect.regions.size();
            outline_problems.push_back(o.str());
        }
        for (size_t i = 0; i < std::min(regions.size(), expect.regions.size()); ++i) {
            if (regions[i]->ops != expect.regions[i].ops) {
                std::ostringstream o;
                o << "region " << i << " ops [" << joinOps(regions[i]->ops)
                  << "], contract says [" << joinOps(expect.regions[i].ops) << "]";
                outline_problems.push_back(o.str());
            }
            if (expect.regions[i].inputs >= 0 &&
                static_cast<int>(regions[i]->inputs.size()) != expect.regions[i].inputs) {
                std::ostringstream o;
                o << "region " << i << " takes " << regions[i]->inputs.size()
                  << " inputs, contract says " << expect.regions[i].inputs;
                outline_problems.push_back(o.str());
            }
            if (regions[i]->inside_gradient != expect.regions[i].inside_gradient) {
                std::ostringstream o;
                o << "region " << i << " inside_gradient="
                  << (regions[i]->inside_gradient ? "true" : "false")
                  << ", contract says "
                  << (expect.regions[i].inside_gradient ? "true" : "false");
                outline_problems.push_back(o.str());
            }
        }

        // ── breaks_reported ── matched as multisets on
        // (construct, reason, builtin): a break is identified by what it is,
        // not by where it is, so that adding a blank line to a corpus program
        // does not fail the gate.
        std::multiset<std::string> got, want;
        for (const GraphBreak* b : breaks)
            got.insert(b->construct + "|" + breakReasonName(b->reason) + "|" + b->builtin);
        for (const ExpectedBreak& b : expect.breaks)
            want.insert(b.construct + "|" + b.reason + "|" + b.builtin);

        std::vector<std::string> break_problems;
        {
            std::multiset<std::string> missing, extra;
            std::set_difference(want.begin(), want.end(), got.begin(), got.end(),
                                std::inserter(missing, missing.begin()));
            std::set_difference(got.begin(), got.end(), want.begin(), want.end(),
                                std::inserter(extra, extra.begin()));
            for (const std::string& m : missing)
                break_problems.push_back("break the contract predicts was not reported: " + m);
            for (const std::string& e : extra)
                break_problems.push_back("break reported that the contract does not predict: " + e);
        }

        const bool ok = outline_problems.empty() && break_problems.empty();
        std::printf("%-34s %8zu %8zu %8zu %8zu  %s\n",
                    program.c_str(), regions.size(), expect.regions.size(),
                    breaks.size(), expect.breaks.size(), ok ? "PASS" : "FAIL");
        if (ok) programs_ok++;
        for (const std::string& p : outline_problems) {
            std::printf("      outline: %s\n", p.c_str());
            outline_failures.push_back({program, p});
            outline_bad_programs.insert(program);
        }
        for (const std::string& p : break_problems) {
            std::printf("      break:   %s\n", p.c_str());
            break_failures.push_back({program, p});
            break_bad_programs.insert(program);
        }

        if (measure_parity) {
            for (const Region* reg : regions) {
                ParityRow row;
                row.program = program;
                row.region_id = reg->id;
                row.ops = reg->ops.size();
                if (!measureRegionParity(*reg, pass.functions(), g_arena, &row))
                    row.passed = false;
                parity_rows.push_back(row);
            }
        }

        // The forms are freed after parity, because a region points into them.
        for (eshkol_ast_t* f : forms) delete f;
    }

    std::printf("\n");
    std::printf("outlines_maximal: %s (%zu of %zu programs agree with the contract;"
                " %zu disagreements)\n",
                outline_failures.empty() ? "PASS" : "FAIL",
                programs.size() - outline_bad_programs.size(), programs.size(),
                outline_failures.size());
    std::printf("breaks_reported:  %s (%zu of %zu programs agree with the contract;"
                " %zu disagreements)\n",
                break_failures.empty() ? "PASS" : "FAIL",
                programs.size() - break_bad_programs.size(), programs.size(),
                break_failures.size());
    std::printf("programs fully agreeing: %d of %zu\n", programs_ok, programs.size());

    if (!parity_rows.empty()) {
        std::printf("\n%-34s %6s %5s %10s %11s %11s  %s\n",
                    "program", "region", "ops", "shape", "max abs", "max rel", "verdict");
        std::printf("%s\n", std::string(104, '-').c_str());
        size_t measured = 0, agreed = 0, skipped = 0;
        for (const ParityRow& r : parity_rows) {
            if (!r.ran) {
                std::printf("%-34s %6d %5zu %10s %11s %11s  SKIP (%s)\n",
                            r.program.c_str(), r.region_id, r.ops, "-", "-", "-",
                            r.note.c_str());
                skipped++;
                continue;
            }
            measured++;
            if (r.passed) agreed++;
            std::printf("%-34s %6d %5zu %10s %11.3e %11.3e  %s%s%s\n",
                        r.program.c_str(), r.region_id, r.ops, r.result_shape.c_str(),
                        r.worst_abs, r.worst_rel, r.passed ? "PASS" : "FAIL",
                        r.note.empty() ? "" : " ", r.note.c_str());
        }
        std::printf("\nregion_device_host_parity: %s (%zu of %zu regions measured agree;"
                    " %zu not measured)\n",
                    (measured > 0 && agreed == measured) ? "PASS" : "FAIL",
                    agreed, measured, skipped);
        if (measured == 0)
            std::printf("region_device_host_parity: FAIL (nothing was measured)\n");
    }

    return (outline_failures.empty() && break_failures.empty()) ? 0 : 1;
}
