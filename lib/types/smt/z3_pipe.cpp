/**
 * @file z3_pipe.cpp
 * @brief Optional z3 backend for refinement verification conditions
 *        (ADR-0044): the query as SMT-LIB2 over a pipe to `z3 -in -smt2`.
 *
 * Compiled in when the build option ESHKOL_REFINEMENT_Z3 is ON (POSIX hosts).
 * The internal procedure stays the default; z3 is selected with
 * ESHKOL_REFINEMENT_SOLVER=z3, or run beside it with =cross, which reports any
 * query on which the two disagree. Resource use is bounded with z3's
 * deterministic `rlimit`, so a query answers the same way on every run.
 * A Sat answer carries the values z3 reports for the query's variables.
 */
#include "eshkol/types/smt.h"

#include <cstdlib>
#include <set>
#include <sstream>

#if defined(ESHKOL_REFINEMENT_Z3) && !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace eshkol::smt {

bool z3BackendCompiled() {
#if defined(ESHKOL_REFINEMENT_Z3) && !defined(_WIN32)
    return true;
#else
    return false;
#endif
}

#if defined(ESHKOL_REFINEMENT_Z3) && !defined(_WIN32)

namespace {

#ifndef ESHKOL_Z3_DEFAULT_PATH
#define ESHKOL_Z3_DEFAULT_PATH "z3"
#endif

// Run @p program with @p input on stdin; false when it cannot be started.
bool runPipe(const std::string& program, const std::string& input, std::string& output) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) != 0) return false;
    if (pipe(out_pipe) != 0) {
        close(in_pipe[0]);
        close(in_pipe[1]);
        return false;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, in_pipe[0], 0);
    posix_spawn_file_actions_adddup2(&fa, out_pipe[1], 1);
    posix_spawn_file_actions_addclose(&fa, in_pipe[1]);
    posix_spawn_file_actions_addclose(&fa, out_pipe[0]);
    std::string a0 = program, a1 = "-in", a2 = "-smt2";
    char* argv[] = {a0.data(), a1.data(), a2.data(), nullptr};
    pid_t pid = 0;
    int rc = program.find('/') != std::string::npos
                 ? posix_spawn(&pid, program.c_str(), &fa, nullptr, argv, environ)
                 : posix_spawnp(&pid, program.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(in_pipe[0]);
    close(out_pipe[1]);
    if (rc != 0) {
        close(in_pipe[1]);
        close(out_pipe[0]);
        return false;
    }
    // Queries are small; write everything, then read to EOF.
    void (*old)(int) = signal(SIGPIPE, SIG_IGN);
    size_t off = 0;
    while (off < input.size()) {
        ssize_t n = write(in_pipe[1], input.data() + off, input.size() - off);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            break;
        }
        off += static_cast<size_t>(n);
    }
    close(in_pipe[1]);
    char buf[4096];
    for (;;) {
        ssize_t n = read(out_pipe[0], buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        output.append(buf, static_cast<size_t>(n));
    }
    close(out_pipe[0]);
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    signal(SIGPIPE, old);
    return true;
}

// Parse one value printed by (get-value ...): 3, (- 3), 1.5, (/ 1.0 2.0), (- (/ 1 2)), true.
std::optional<Rational> parseNumber(const std::string& s, size_t& i);

void skipWs(const std::string& s, size_t& i) {
    while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\t' || s[i] == '\r')) ++i;
}

std::optional<Rational> parseAtomNumber(const std::string& tok) {
    size_t dot = tok.find('.');
    if (dot == std::string::npos) return Rational::parse(tok);
    std::string whole = tok.substr(0, dot), frac = tok.substr(dot + 1);
    auto w = Integer::parse(whole.empty() ? "0" : whole);
    if (!w) return std::nullopt;
    Integer scale(1);
    for (size_t k = 0; k < frac.size(); ++k) scale *= Integer(10);
    auto f = Integer::parse(frac.empty() ? "0" : frac);
    if (!f) return std::nullopt;
    return Rational(*w) + Rational(*f, scale);
}

std::optional<Rational> parseNumber(const std::string& s, size_t& i) {
    skipWs(s, i);
    if (i >= s.size()) return std::nullopt;
    if (s[i] != '(') {
        size_t j = i;
        while (j < s.size() && s[j] != ' ' && s[j] != ')' && s[j] != '\n') ++j;
        std::string tok = s.substr(i, j - i);
        i = j;
        return parseAtomNumber(tok);
    }
    ++i;
    skipWs(s, i);
    char op = s[i];
    ++i;
    auto a = parseNumber(s, i);
    if (!a) return std::nullopt;
    skipWs(s, i);
    std::optional<Rational> result;
    if (op == '-' && i < s.size() && s[i] == ')') {
        result = -*a;
    } else if (op == '/') {
        auto b = parseNumber(s, i);
        if (!b || b->isZero()) return std::nullopt;
        result = *a / *b;
    } else if (op == '-') {
        auto b = parseNumber(s, i);
        if (!b) return std::nullopt;
        result = *a - *b;
    } else {
        return std::nullopt;
    }
    skipWs(s, i);
    if (i < s.size() && s[i] == ')') ++i;
    return result;
}

void collectVars(Term t, std::set<Term>& seen, std::vector<Term>& vars) {
    if (!seen.insert(t).second) return;
    if (t->kind == Kind::Var) vars.push_back(t);
    for (Term a : t->args) collectVars(a, seen, vars);
}

class Z3Solver final : public Solver {
public:
    Z3Solver(std::string path, uint64_t rlimit) : path_(std::move(path)), rlimit_(rlimit) {}
    std::string name() const override { return "z3"; }

    Outcome check(TermManager& tm, const std::vector<Term>& assertions) override {
        Outcome out;
        std::set<Term> seen;
        std::vector<Term> vars;
        for (Term a : assertions) collectVars(a, seen, vars);
        std::ostringstream script;
        script << "(set-option :produce-models true)\n(set-option :rlimit " << rlimit_ << ")\n";
        script << toSmtLib2Script(tm, assertions, true);
        std::vector<Term> numeric_or_bool;
        for (Term v : vars) {
            if (v->sort != Sort::Obj) numeric_or_bool.push_back(v);
        }
        if (!numeric_or_bool.empty()) {
            script << "(get-value (";
            for (Term v : numeric_or_bool) script << " " << toSmtLib2(v);
            script << "))\n";
        }
        std::string output;
        if (!runPipe(path_, script.str(), output)) {
            out.reason = "z3 could not be started (" + path_ + ")";
            return out;
        }
        size_t nl = output.find('\n');
        std::string first = output.substr(0, nl);
        while (!first.empty() && (first.back() == '\r' || first.back() == ' ')) first.pop_back();
        if (first == "unsat") {
            out.result = Result::Unsat;
            return out;
        }
        if (first != "sat") {
            out.reason = "z3 answered '" + first + "'";
            return out;
        }
        out.result = Result::Sat;
        // ((x 3) (b true) ...)
        size_t i = nl == std::string::npos ? output.size() : nl + 1;
        for (Term v : numeric_or_bool) {
            std::string sym = toSmtLib2(v);
            size_t at = output.find("(" + sym + " ", i);
            if (at == std::string::npos) continue;
            size_t j = at + sym.size() + 2;
            if (v->sort == Sort::Bool) {
                out.model.booleans[v->name] = output.compare(j, 4, "true") == 0;
            } else if (auto r = parseNumber(output, j)) {
                out.model.numbers[v->name] = *r;
            }
        }
        return out;
    }

private:
    std::string path_;
    uint64_t rlimit_;
};

}  // namespace

std::unique_ptr<Solver> makeZ3Solver(const std::string& z3_path, uint64_t rlimit) {
    std::string path = z3_path;
    if (path.empty()) {
        const char* env = std::getenv("ESHKOL_Z3");
        path = (env && *env) ? env : ESHKOL_Z3_DEFAULT_PATH;
    }
    return std::make_unique<Z3Solver>(path, rlimit);
}

#else

std::unique_ptr<Solver> makeZ3Solver(const std::string&, uint64_t) { return nullptr; }

#endif

}  // namespace eshkol::smt
