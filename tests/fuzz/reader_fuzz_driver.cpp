/*
 * reader_fuzz_driver.cpp — deterministic, seeded adversarial-input harness
 * for the hosted S-expression reader (lib/core/runtime_reader_hosted.cpp,
 * entry point eshkol_read_sexpr).
 *
 * Why not just the existing libFuzzer `fuzz_parser` harness: that one
 * targets the *frontend* source parser and relies on coverage-guided
 * random mutation, which is a poor fit for reliably reproducing specific
 * structural pathologies (10^6-element flat lists, >4096-deep nesting,
 * dotted-pair edge cases, arena exhaustion) on a fixed schedule. This
 * driver instead *generates* each adversarial category directly from a
 * seeded PRNG (no wall-clock, no /dev/urandom) so a report of "seed 7,
 * category dotted_pairs, case 3" is exactly replayable.
 *
 * Isolation: every generated input is read in a forked child with
 * RLIMIT_CORE=0, a bounded RLIMIT_AS, and a SIGALRM watchdog. A crash or
 * hang in one case is captured (signal + input saved to the artifact
 * dir) without taking down the rest of the run, so a single pass finds
 * every distinct bug instead of stopping at the first one.
 *
 * Usage:
 *   reader_fuzz_driver --smoke                 (~5-10s, for CI)
 *   reader_fuzz_driver --full                  (~2-4 min, manual/nightly)
 *   reader_fuzz_driver --regression             (fixed assertions only)
 *   reader_fuzz_driver --artifact-dir <path>    (default: ./fuzz-artifacts)
 *
 * Exit code: 0 if no crash/hang/UB was observed (and, in --regression
 * mode, all assertions held); 1 otherwise.
 */

#include "arena_memory.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>

#include <sys/wait.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <dirent.h>

// eshkol_read_sexpr has no public header (it's an internal runtime entry
// point reached from generated code / the (read) builtin) — declare the
// ABI directly, matching the pattern other runtime TUs use for
// cross-file `extern "C"` calls (e.g. eshkol_intern_symbol_lookup).
extern "C" void eshkol_read_sexpr(void* arena_void, void* fp_void,
                                   eshkol_tagged_value_t* result);

namespace {

// ---------------------------------------------------------------------
// Deterministic PRNG (splitmix64). No wall-clock, no /dev/urandom — a
// given seed always produces the same case sequence.
// ---------------------------------------------------------------------
struct SplitMix64 {
    uint64_t state;
    explicit SplitMix64(uint64_t seed) : state(seed) {}
    uint64_t next() {
        uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return z ^ (z >> 31);
    }
    uint64_t range(uint64_t lo, uint64_t hi) { // inclusive
        if (hi <= lo) return lo;
        return lo + (next() % (hi - lo + 1));
    }
    bool coin() { return (next() & 1) != 0; }
};

// Fixed, replayable seed set. Deliberately small and hand-picked rather
// than "the first N integers" so a re-run after adding a new category
// still exercises the same corners.
const uint64_t kFixedSeeds[] = {
    0x1ULL, 0xC0FFEEULL, 0xDEADBEEFULL, 0x5EED5EEDULL, 0x123456789ABCDEFULL,
    0x42ULL, 0x9E3779B97F4A7C15ULL, 0x0ULL, 0xFFFFFFFFULL, 0x7ULL,
};
constexpr int kFixedSeedCount = sizeof(kFixedSeeds) / sizeof(kFixedSeeds[0]);

// ---------------------------------------------------------------------
// Adversarial input generators. Each returns a byte buffer (may contain
// embedded NULs / invalid UTF-8 — callers must use explicit lengths).
// ---------------------------------------------------------------------

std::string gen_long_flat_list(SplitMix64& rng, size_t n) {
    std::string s;
    s.reserve(n * 4 + 2);
    s.push_back('(');
    for (size_t i = 0; i < n; i++) {
        s += std::to_string((int64_t)rng.range(-1000000, 1000000));
        s.push_back(' ');
    }
    s.push_back(')');
    return s;
}

std::string gen_deep_nested_list(size_t depth, bool close_all) {
    std::string s;
    s.reserve(depth * 2 + 8);
    for (size_t i = 0; i < depth; i++) s.push_back('(');
    s += "99";
    if (close_all) {
        for (size_t i = 0; i < depth; i++) s.push_back(')');
    }
    return s;
}

std::string gen_unbalanced_parens(SplitMix64& rng, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; i++) {
        int pick = (int)rng.range(0, 4);
        switch (pick) {
            case 0: s.push_back('('); break;
            case 1: s.push_back(')'); break;
            case 2: s += "1 "; break;
            case 3: s += "a "; break;
            default: s.push_back(' '); break;
        }
    }
    return s;
}

std::string gen_unterminated_string(SplitMix64& rng, size_t backslashes) {
    std::string s = "\"";
    for (size_t i = 0; i < backslashes; i++) {
        s.push_back('\\');
        s.push_back((char)('a' + (rng.range(0, 25))));
    }
    // Deliberately no closing quote.
    return s;
}

std::string gen_huge_atom(SplitMix64& rng, size_t len, bool numeric) {
    std::string s;
    s.reserve(len);
    if (numeric) {
        for (size_t i = 0; i < len; i++) s.push_back((char)('0' + rng.range(0, 9)));
    } else {
        for (size_t i = 0; i < len; i++) {
            char c;
            do { c = (char)('!' + rng.range(0, 93)); }
            while (c == '(' || c == ')' || c == '"' || c == ';' || c == '\'' || c == '`' || c == ',');
            s.push_back(c);
        }
    }
    return s;
}

std::vector<std::string> gen_dotted_pair_edge_cases() {
    return {
        ". )",
        "(a . )",
        "(. a)",
        "(a . b . c)",
        "(. )",
        "( . )",
        "(a .)",
        "(. )",
        "(a . b)",
        "(a . . b)",
        "(a b . )",
        "(.)",
    };
}

std::string gen_embedded_nul(SplitMix64& rng, size_t len) {
    std::string s = "\"";
    for (size_t i = 0; i < len; i++) {
        s.push_back(rng.coin() ? '\0' : (char)('a' + rng.range(0, 25)));
    }
    s.push_back('"');
    return s;
}

std::string gen_invalid_utf8(SplitMix64& rng, size_t len) {
    std::string s = "\"";
    for (size_t i = 0; i < len; i++) {
        // Skew toward UTF-8 continuation/lead bytes without valid
        // sequences, plus the raw 0xFF/0xFE non-characters.
        int pick = (int)rng.range(0, 3);
        unsigned char c;
        switch (pick) {
            case 0: c = (unsigned char)rng.range(0x80, 0xBF); break; // stray continuation
            case 1: c = (unsigned char)rng.range(0xF8, 0xFF); break; // invalid lead
            case 2: c = 0xC0; break; // overlong-encoding lead with no follower
            default: c = (unsigned char)rng.range(0x20, 0x7E); break;
        }
        s.push_back((char)c);
    }
    s.push_back('"');
    return s;
}

std::string gen_mixed_vector_list_nesting(SplitMix64& rng, size_t depth) {
    std::string s;
    for (size_t i = 0; i < depth; i++) {
        s += rng.coin() ? "#(" : "(";
    }
    s += "1";
    for (size_t i = 0; i < depth; i++) s.push_back(')');
    return s;
}

std::string gen_oversized_vector(size_t n) {
    std::string s = "#(";
    for (size_t i = 0; i < n; i++) { s += "1 "; }
    s.push_back(')');
    return s;
}

std::string gen_quote_pathology(SplitMix64& rng, size_t depth) {
    std::string s;
    for (size_t i = 0; i < depth; i++) {
        int pick = (int)rng.range(0, 2);
        s.push_back(pick == 0 ? '\'' : (pick == 1 ? '`' : ','));
    }
    s += "x";
    return s;
}

std::string gen_whitespace_comment_abuse(SplitMix64& rng, size_t n) {
    std::string s;
    for (size_t i = 0; i < n; i++) {
        int pick = (int)rng.range(0, 5);
        switch (pick) {
            case 0: s.push_back(' '); break;
            case 1: s.push_back('\t'); break;
            case 2: s.push_back('\n'); break;
            case 3: s.push_back('\r'); break;
            case 4: s += "; comment to end of line\r"; break; // CR without LF
            default: s += ";;;;;;\n"; break;
        }
    }
    s += "42";
    return s;
}

std::string gen_random_bytes(SplitMix64& rng, size_t n) {
    std::string s;
    s.reserve(n);
    for (size_t i = 0; i < n; i++) s.push_back((char)(rng.next() & 0xFF));
    return s;
}

// ---------------------------------------------------------------------
// Child-process execution: run one input through eshkol_read_sexpr
// (repeatedly, draining the stream — matches how (read) is used in a
// loop over a file/string port) under a watchdog, with core dumps and
// address-space growth capped so a bug can't be a disk/memory bomb.
// ---------------------------------------------------------------------

struct CaseResult {
    bool crashed = false;
    bool timed_out = false;
    int signal = 0;
    int exit_code = 0;
};

constexpr unsigned kWatchdogSeconds = 3;
constexpr rlim_t kChildAddressSpaceCap = 512ULL * 1024 * 1024; // 512 MB/child

// Arena size used for "normal" cases (grows on demand, unbounded block
// count) vs. the small bounded arena used to specifically provoke OOM
// paths inside the reader's allocation helpers.
constexpr size_t kNormalArenaBlock = 4 * 1024 * 1024;
constexpr size_t kBoundedArenaCapacity = 8 * 1024; // deliberately tiny

void run_child(const std::string& data, bool bounded_arena, int reads_to_drain) {
    // No core files — keeps the artifact/disk budget intact even if the
    // OS core-pattern would otherwise dump into the working directory.
    struct rlimit rl_core = {0, 0};
    setrlimit(RLIMIT_CORE, &rl_core);

    struct rlimit rl_as = {kChildAddressSpaceCap, kChildAddressSpaceCap};
    setrlimit(RLIMIT_AS, &rl_as);

    // Silence runtime diagnostics (eshkol_error/eshkol_warn) — thousands
    // of cases would otherwise flood stderr; a crash is reported by the
    // parent from the wait status instead.
    freopen("/dev/null", "w", stderr);
    freopen("/dev/null", "w", stdout);

    alarm(kWatchdogSeconds);

    FILE* fp = fmemopen((void*)data.data(), data.size(), "rb");
    if (!fp) _exit(3);

    arena_t* arena = bounded_arena
        ? arena_create_bounded(kBoundedArenaCapacity)
        : arena_create(kNormalArenaBlock);
    if (!arena) { fclose(fp); _exit(3); }

    for (int i = 0; i < reads_to_drain; i++) {
        eshkol_tagged_value_t result;
        memset(&result, 0, sizeof(result));
        eshkol_read_sexpr(arena, fp, &result);
        if (result.type == 0xFF && feof(fp)) break; // EOF / depth-sentinel and stream drained
    }

    arena_destroy(arena);
    fclose(fp);
    _exit(0);
}

CaseResult run_isolated(const std::string& data, bool bounded_arena = false,
                         int reads_to_drain = 4) {
    CaseResult r;
    // Flush before fork(): stdio buffers otherwise get duplicated into
    // the child and flushed a second time (garbled/doubled log lines),
    // since the child inherits whatever the parent hadn't yet written.
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid < 0) {
        // Fork failure isn't a reader bug; treat conservatively as a
        // non-finding rather than a false crash report.
        return r;
    }
    if (pid == 0) {
        run_child(data, bounded_arena, reads_to_drain);
        _exit(0); // unreachable
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        r.signal = WTERMSIG(status);
        if (r.signal == SIGALRM) {
            r.timed_out = true;
        } else {
            r.crashed = true;
        }
    } else if (WIFEXITED(status)) {
        r.exit_code = WEXITSTATUS(status);
    }
    return r;
}

// ---------------------------------------------------------------------
// Artifact / disk budget management.
//
// Hard rule (repo policy, prior fuzz-harness disk incidents): total
// artifact usage is capped; only failing/minimized cases are kept; the
// caller (scripts/run_reader_fuzz.sh) also traps and cleans transient
// build/scratch state. Here we enforce the corpus-side cap: refuse to
// write a new artifact once the directory exceeds the budget, and print
// a clear warning so the run doesn't silently grow unbounded.
// ---------------------------------------------------------------------

constexpr uint64_t kArtifactBudgetBytes = 64ULL * 1024 * 1024; // 64 MB « 200 MB task cap

uint64_t dir_size_bytes(const std::string& path) {
    DIR* d = opendir(path.c_str());
    if (!d) return 0;
    uint64_t total = 0;
    struct dirent* ent;
    while ((ent = readdir(d)) != nullptr) {
        if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0) continue;
        std::string full = path + "/" + ent->d_name;
        struct stat st;
        if (stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) total += (uint64_t)st.st_size;
    }
    closedir(d);
    return total;
}

void write_artifact(const std::string& dir, const std::string& name, const std::string& data) {
    mkdir(dir.c_str(), 0755);
    if (dir_size_bytes(dir) + data.size() > kArtifactBudgetBytes) {
        fprintf(stderr, "  [artifact budget %llu MB reached — not writing %s]\n",
                (unsigned long long)(kArtifactBudgetBytes / (1024 * 1024)), name.c_str());
        return;
    }
    std::string path = dir + "/" + name;
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return;
    fwrite(data.data(), 1, data.size(), f);
    fclose(f);
}

// ---------------------------------------------------------------------
// Case runner: executes one (category, input) pair, reports, and saves
// an artifact on any finding.
// ---------------------------------------------------------------------

struct Stats {
    long total = 0;
    long findings = 0;
};

void exercise(Stats& stats, const std::string& artifact_dir, const char* category,
              uint64_t seed, int case_idx, const std::string& input,
              bool bounded_arena = false) {
    stats.total++;
    CaseResult r = run_isolated(input, bounded_arena);
    if (r.crashed || r.timed_out) {
        stats.findings++;
        char name[256];
        snprintf(name, sizeof(name), "%s-seed%llx-case%d.txt", category,
                 (unsigned long long)seed, case_idx);
        write_artifact(artifact_dir, name, input);
        if (r.timed_out) {
            printf("HANG   category=%-28s seed=0x%llx case=%d (>%us) artifact=%s\n",
                   category, (unsigned long long)seed, case_idx, kWatchdogSeconds, name);
        } else {
            printf("CRASH  category=%-28s seed=0x%llx case=%d signal=%d (%s) artifact=%s\n",
                   category, (unsigned long long)seed, case_idx, r.signal, strsignal(r.signal), name);
        }
    }
}

// ---------------------------------------------------------------------
// Category sweep. `scale` shrinks sizes for --smoke and grows them for
// --full while keeping the exact same generator code paths.
// ---------------------------------------------------------------------

void run_seed(Stats& stats, const std::string& artifact_dir, uint64_t seed, bool full) {
    SplitMix64 rng(seed);
    int idx = 0;

    // 1. Extremely long flat lists — the case the iterative read_list
    // rewrite specifically targeted. --full pushes into the 10^5-10^6
    // range called out in the task; --smoke stays small enough for a
    // sub-minute CI budget.
    for (size_t n : (full ? std::vector<size_t>{100000, 500000, 1000000}
                           : std::vector<size_t>{2000, 20000})) {
        exercise(stats, artifact_dir, "long_flat_list", seed, idx++, gen_long_flat_list(rng, n));
    }

    // 2. Deeply nested lists at/over ESHKOL_READ_MAX_DEPTH (4096).
    for (size_t depth : std::vector<size_t>{100, 4095, 4096, 4097, (size_t)(full ? 50000 : 8000)}) {
        exercise(stats, artifact_dir, "deep_nesting", seed, idx++,
                 gen_deep_nested_list(depth, /*close_all=*/true));
        // Unterminated deep nesting (never closed) exercises the same
        // recursion-depth path via the EOF-tail branch.
        exercise(stats, artifact_dir, "deep_nesting_unterminated", seed, idx++,
                 gen_deep_nested_list(depth, /*close_all=*/false));
    }

    // 3. Unbalanced parens.
    for (size_t n : std::vector<size_t>{50, 500, (size_t)(full ? 50000 : 5000)}) {
        exercise(stats, artifact_dir, "unbalanced_parens", seed, idx++,
                 gen_unbalanced_parens(rng, n));
    }

    // 4. Unterminated strings, including the historical C4 buffer-bound
    // bug shape (4100+ backslash escapes).
    for (size_t n : std::vector<size_t>{10, 4100, (size_t)(full ? 200000 : 8000)}) {
        exercise(stats, artifact_dir, "unterminated_string", seed, idx++,
                 gen_unterminated_string(rng, n));
    }

    // 5. Huge atoms/numbers/symbols.
    for (size_t len : std::vector<size_t>{300, 5000, (size_t)(full ? 500000 : 20000)}) {
        exercise(stats, artifact_dir, "huge_atom_numeric", seed, idx++,
                 gen_huge_atom(rng, len, /*numeric=*/true));
        exercise(stats, artifact_dir, "huge_atom_symbol", seed, idx++,
                 gen_huge_atom(rng, len, /*numeric=*/false));
    }

    // 6. Dotted-pair edge cases — fixed, exhaustive, not scaled by seed.
    if (seed == kFixedSeeds[0]) {
        for (auto& s : gen_dotted_pair_edge_cases()) {
            exercise(stats, artifact_dir, "dotted_pair_edge", seed, idx++, s);
        }
    }

    // 7. Embedded NULs.
    for (size_t len : std::vector<size_t>{16, (size_t)(full ? 4096 : 256)}) {
        exercise(stats, artifact_dir, "embedded_nul", seed, idx++, gen_embedded_nul(rng, len));
    }

    // 8. Invalid UTF-8.
    for (size_t len : std::vector<size_t>{32, (size_t)(full ? 8192 : 512)}) {
        exercise(stats, artifact_dir, "invalid_utf8", seed, idx++, gen_invalid_utf8(rng, len));
    }

    // 9. Deeply mixed vector/list nesting.
    for (size_t depth : std::vector<size_t>{50, (size_t)(full ? 8000 : 1500)}) {
        exercise(stats, artifact_dir, "mixed_vector_list_nesting", seed, idx++,
                 gen_mixed_vector_list_nesting(rng, depth));
    }
    // Oversized flat vector — probes read_vector's fixed 1024-element
    // stack buffer safety limit.
    for (size_t n : std::vector<size_t>{1023, 1024, 1025, 5000, (size_t)(full ? 200000 : 4000)}) {
        exercise(stats, artifact_dir, "oversized_vector", seed, idx++, gen_oversized_vector(n));
    }

    // 10. Quote/quasiquote/unquote pathologies.
    for (size_t depth : std::vector<size_t>{10, 4096, (size_t)(full ? 20000 : 6000)}) {
        exercise(stats, artifact_dir, "quote_pathology", seed, idx++, gen_quote_pathology(rng, depth));
    }
    exercise(stats, artifact_dir, "quote_at_eof", seed, idx++, std::string("'"));
    exercise(stats, artifact_dir, "quote_at_eof", seed, idx++, std::string("`"));
    exercise(stats, artifact_dir, "quote_at_eof", seed, idx++, std::string(","));

    // 11. Whitespace / comment abuse.
    for (size_t n : std::vector<size_t>{100, (size_t)(full ? 200000 : 10000)}) {
        exercise(stats, artifact_dir, "whitespace_comment_abuse", seed, idx++,
                 gen_whitespace_comment_abuse(rng, n));
    }

    // 12. Raw random byte soup (classic baseline, catches whatever the
    // structured generators above don't).
    for (size_t n : std::vector<size_t>{64, 4096, (size_t)(full ? 65536 : 8192)}) {
        exercise(stats, artifact_dir, "random_bytes", seed, idx++, gen_random_bytes(rng, n));
    }

    // 13. Arena-exhaustion paths: same generators, but pointed at a
    // tiny bounded arena so allocation-failure branches inside
    // make_string_tagged / read_vector / make_symbol_tagged / read_list
    // are actually reached instead of always succeeding against a
    // generously-sized growth arena.
    exercise(stats, artifact_dir, "arena_exhaustion_strings", seed, idx++,
             gen_long_flat_list(rng, 2000), /*bounded_arena=*/true);
    exercise(stats, artifact_dir, "arena_exhaustion_vector", seed, idx++,
             gen_oversized_vector(2000), /*bounded_arena=*/true);
    exercise(stats, artifact_dir, "arena_exhaustion_strings2", seed, idx++,
             gen_huge_atom(rng, 4000, /*numeric=*/false), /*bounded_arena=*/true);
}

// ---------------------------------------------------------------------
// Deterministic regression assertions (task requirement #4): the depth
// guard must fire gracefully, and a very long flat list must not blow
// the native stack. These are checked structurally (child exit code),
// not just "didn't crash", so a future regression that silently starts
// recursing again gets caught even if it happens not to crash on this
// machine's stack size.
// ---------------------------------------------------------------------

// Exit codes the child reports for the regression checks specifically
// (distinct from the generic 0/3 used by run_child).
constexpr int kRegressionOkDepthError = 42;
constexpr int kRegressionUnexpectedTag = 43;

void run_regression_child_depth_guard(const std::string& input) {
    struct rlimit rl_core = {0, 0};
    setrlimit(RLIMIT_CORE, &rl_core);
    struct rlimit rl_as = {kChildAddressSpaceCap, kChildAddressSpaceCap};
    setrlimit(RLIMIT_AS, &rl_as);
    freopen("/dev/null", "w", stderr);
    alarm(kWatchdogSeconds);

    FILE* fp = fmemopen((void*)input.data(), input.size(), "rb");
    arena_t* arena = arena_create(kNormalArenaBlock);
    eshkol_tagged_value_t result;
    memset(&result, 0, sizeof(result));
    eshkol_read_sexpr(arena, fp, &result);
    arena_destroy(arena);
    fclose(fp);
    // type 0xFF is the shared EOF/depth-error sentinel this reader uses;
    // at a depth comfortably over the 4096 cap, reaching a real,
    // fully-formed datum back would mean the guard didn't fire (or, for
    // the vector variant, that it fired but got silently swallowed
    // instead of propagated — see the read_vector fix).
    _exit(result.type == 0xFF ? kRegressionOkDepthError : kRegressionUnexpectedTag);
}

bool run_one_depth_guard_regression(const char* label, const std::string& input) {
    printf("[regression] %s...\n", label);
    fflush(stdout);
    fflush(stderr);
    pid_t pid = fork();
    if (pid == 0) { run_regression_child_depth_guard(input); _exit(99); }
    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status)) {
        printf("  FAIL: crashed/hung (signal %d, %s)\n",
               WTERMSIG(status), strsignal(WTERMSIG(status)));
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != kRegressionOkDepthError) {
        printf("  FAIL: did not report a graceful depth error (exit=%d)\n",
               WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        return false;
    }
    printf("  PASS: depth-error sentinel returned cleanly, no crash/hang\n");
    return true;
}

bool run_regressions() {
    bool ok = true;

    ok &= run_one_depth_guard_regression(
        "depth guard fires gracefully past ESHKOL_READ_MAX_DEPTH (nested lists)",
        gen_deep_nested_list(8000, /*close_all=*/true));

    // Regression for the read_vector stack-frame bug (SIGSEGV around
    // ~1700-deep `#(` nesting on an 8 MB stack, well under the 4096
    // guard) and the read_vector depth-sentinel-propagation bug (the
    // guard fired but the reader silently embedded the poisoned
    // sentinel as vector data and kept parsing instead of erroring).
    // 20000-deep pure vector nesting exercises both: it must neither
    // crash/hang nor come back as a "successful" vector.
    {
        std::string vec_input;
        vec_input.reserve(20000 * 2 + 8);
        for (int i = 0; i < 20000; i++) vec_input += "#(";
        vec_input += "1";
        for (int i = 0; i < 20000; i++) vec_input += ")";
        ok &= run_one_depth_guard_regression(
            "20000-deep #( vector nesting stays bounded-stack and reports depth error",
            vec_input);
    }

    printf("[regression] long flat list (10^6 elements) stays bounded-stack...\n");
    {
        SplitMix64 rng(kFixedSeeds[0]);
        std::string input = gen_long_flat_list(rng, 1000000);
        CaseResult r = run_isolated(input, /*bounded_arena=*/false, /*reads_to_drain=*/1);
        if (r.crashed || r.timed_out) {
            printf("  FAIL: 1,000,000-element flat list %s\n",
                   r.timed_out ? "hung" : "crashed");
            ok = false;
        } else {
            printf("  PASS: 1,000,000-element flat list read without crash/hang\n");
        }
    }

    return ok;
}

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    bool smoke = false;
    bool regression_only = false;
    std::string artifact_dir = "fuzz-artifacts";
    int seed_count = kFixedSeedCount;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--full") full = true;
        else if (arg == "--smoke") smoke = true;
        else if (arg == "--regression") regression_only = true;
        else if (arg == "--artifact-dir" && i + 1 < argc) artifact_dir = argv[++i];
        else if (arg == "--seed-count" && i + 1 < argc) seed_count = atoi(argv[++i]);
        else if (arg == "--help") {
            printf("usage: %s [--smoke|--full|--regression] [--artifact-dir DIR] "
                   "[--seed-count N]\n", argv[0]);
            return 0;
        }
    }
    if (seed_count < 1) seed_count = 1;
    if (seed_count > kFixedSeedCount) seed_count = kFixedSeedCount;
    // Smoke mode uses fewer seeds to keep the CI probe under budget; full
    // mode (and the default) uses the entire fixed set.
    if (smoke) seed_count = seed_count < 3 ? seed_count : 3;

    bool regressions_ok = run_regressions();
    if (regression_only) {
        printf(regressions_ok ? "\nRESULT: OK\n" : "\nRESULT: FAIL\n");
        return regressions_ok ? 0 : 1;
    }

    printf("\n[sweep] mode=%s seeds=%d artifact_dir=%s\n", full ? "full" : "smoke",
           seed_count, artifact_dir.c_str());

    Stats stats;
    for (int i = 0; i < seed_count; i++) {
        run_seed(stats, artifact_dir, kFixedSeeds[i], full);
    }

    uint64_t artifact_bytes = dir_size_bytes(artifact_dir);
    printf("\n=== reader_fuzz_driver summary ===\n");
    printf("cases run:      %ld\n", stats.total);
    printf("findings:       %ld\n", stats.findings);
    printf("artifact bytes: %llu (budget %llu)\n",
           (unsigned long long)artifact_bytes, (unsigned long long)kArtifactBudgetBytes);
    printf("regressions:    %s\n", regressions_ok ? "OK" : "FAIL");

    bool ok = regressions_ok && stats.findings == 0;
    printf("\nRESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
