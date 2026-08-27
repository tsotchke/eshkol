# Contributing to Eshkol

Thank you for your interest in contributing to Eshkol! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Contributing to Eshkol](#contributing-to-eshkol)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
    - [Development Environment Setup](#development-environment-setup)
    - [Building the Project](#building-the-project)
    - [Running Tests](#running-tests)
  - [How to Contribute](#how-to-contribute)
    - [Reporting Issues](#reporting-issues)
    - [Suggesting Enhancements](#suggesting-enhancements)
    - [Pull Requests](#pull-requests)
    - [Branch Protection and Required Checks](#branch-protection-and-required-checks)
  - [Development Guidelines](#development-guidelines)
    - [Coding Standards](#coding-standards)
      - [C Code](#c-code)
      - [TypeScript Code (MCP Tools)](#typescript-code-mcp-tools)
    - [Documentation](#documentation)
    - [Testing](#testing)
  - [Project Structure](#project-structure)
  - [Communication](#communication)
  - [Priority Areas for Contribution (v1.4+)](#priority-areas-for-contribution-v14)
    - [Immediate Priorities (v1.4-connection - July 2026)](#immediate-priorities-v14-connection---july-2026)
    - [Near-Term (v1.5-intelligence - August 2026)](#near-term-v15-intelligence---august-2026)
    - [Ongoing](#ongoing)
  - [Recognition](#recognition)

## Code of Conduct

We expect all contributors to adhere to our Code of Conduct. Please be respectful and considerate of others when participating in our community.

## Getting Started

### Development Environment Setup

To set up your development environment for Eshkol, you'll need:

1. **C/C++ Compiler**
   - A C17 + C++20 compiler — the standards `CMakeLists.txt` enforces
     (`CMAKE_C_STANDARD 17`, `CMAKE_CXX_STANDARD 20`)
   - GCC 11+ or Clang 14+ — the toolchain the CI matrix actually builds with
     (the `ubuntu-22.04` / `ubuntu-22.04-arm` runners' defaults). Older
     compilers are untested and not supported
   - On macOS: the default AppleClang (macOS 14 / macOS 15 runners)
   - On Linux: `sudo apt install build-essential`
   - On Windows: Visual Studio 2022 with Desktop development for C++ and
     LLVM 21 ClangCL

2. **CMake**
   - Version 3.14 or higher
   - On macOS: `brew install cmake`
   - On Linux: `sudo apt install cmake`
   - On Windows: Download from [cmake.org](https://cmake.org/download/)

3. **LLVM**
   - Version 21 required for lite/native builds
   - On macOS: `brew install llvm@21`
   - On Linux: install `llvm-21` and `llvm-21-dev` from `apt.llvm.org`
   - On Windows: install the official LLVM 21 SDK and point `LLVM_DIR` at its `lib/cmake/llvm` directory
   - Ensure `llvm-config` is in your PATH on macOS/Linux, or set `LLVM_DIR` on native Windows

4. **Native image I/O codecs**
   - On macOS: ImageIO/CoreGraphics are system frameworks; no extra codec package is required
   - On Linux: `sudo apt install pkg-config libpng-dev libjpeg-dev libwebp-dev`
   - On Windows: the native backend uses GDI+

5. **Git**
   - On macOS: `brew install git`
   - On Linux: `sudo apt install git`
   - On Windows: Download from [git-scm.com](https://git-scm.com/download/win)

### Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/tsotchke/eshkol.git
   cd eshkol
   ```

2. Create a build directory and build the project:
   ```bash
   cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
   cmake --build build --parallel
   ```

### Running Tests

Run the full test suite:

```bash
# From the project root
bash scripts/run_all_tests.sh
```

Or run specific test categories:

```bash
# Individual test suites
bash scripts/run_bignum_tests.sh
bash scripts/run_rational_tests.sh
bash scripts/run_parallel_tests.sh
bash scripts/run_gpu_tests.sh
bash scripts/run_signal_tests.sh
bash scripts/run_macros_tests.sh
# ... see scripts/ directory for all test runners
```

### Testing the Bytecode VM

The bytecode VM can be built and tested independently:

```bash
# Build
gcc -O2 -std=c11 -w lib/backend/eshkol_vm.c -o test_vm -lm -lpthread

# Run all 50 built-in tests
ESHKOL_VM_NO_DISASM=1 ./test_vm

# Run a single Eshkol program through the VM
./test_vm program.esk
```

### Building the Website

The website is written in Eshkol and compiled to WebAssembly:

```bash
# Compile the website
./build/eshkol-run --wasm site/src/main.esk -o site/static/eshkol-site.wasm

# Rebuild the browser REPL VM (site/static/eshkol-vm.{js,wasm})
# CI pins emsdk 4.0.22 (.github/workflows/ci.yml); use the same version locally
# or the bundle can diverge from the checked-in artifact.
emcc -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='EshkolVM' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap"]' \
  -s ERROR_ON_UNDEFINED_SYMBOLS=0 \
  -DESHKOL_VM_WASM -DESHKOL_VM_NO_DISASM \
  -I inc -I lib/backend lib/backend/vm_wasm_repl.c \
  -o site/static/eshkol-vm.js -lm

# Serve locally
cd site/static && python3 -m http.server 8888
```

The REPL VM bundle is a **checked-in artifact** and neither `scripts/build-site.sh`
nor the Pages deploy regenerates it, so it must be rebuilt by hand whenever
`lib/backend/vm_wasm_repl.c`, `lib/backend/eshkol_vm.c`, or the prelude cache
changes — otherwise the browser REPL silently keeps running an older VM. Two
flags are not optional: `-I inc` (the VM includes `eshkol/backend/vm_limits.h`)
and `ERROR_ON_UNDEFINED_SYMBOLS=0`, which leaves the native leaf runtime deps
that are not part of the VM WASM (`eshkol_qrng_uint64`, `eshkol_qrng_double`,
`eshkol_linear_solve`) as aborting stubs so a program calling them fails cleanly.

After rebuilding, check the bundle in node before committing it — the same
`repl_eval` entry point the site uses:

```bash
node -e '
const f=require("./site/static/eshkol-vm.js");
f({print:t=>console.log(t)}).then(m=>{
  const ev=m.cwrap("repl_eval","string",["string"]);
  ev("(display (sqrt 2.0))");                                   // 1.4142135623730951
  ev("(define (v p) (+ (* (vref p 0) (vref p 0)) (* (vref p 1) (vref p 1))))");
  ev("(display (gradient v (vector 3.0 4.0)))");                // #(6 8)
});'
```

Then update the VM WASM size statistic in `site/src/main.esk` (the `s2`
`"...KB"` cell) to match the new artifact — `scripts/verify_site_release.py`
fails if the published number drifts by more than 1KB.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion for improvement:

1. Check the [GitHub Issues](https://github.com/tsotchke/eshkol/issues) to see if the issue has already been reported.
2. If not, create a new issue with a clear title and description.
3. Include steps to reproduce the issue, expected behavior, and actual behavior.
4. If possible, include code examples, logs, or screenshots.

### Suggesting Enhancements

We welcome suggestions for new features or improvements:

1. Check the [ROADMAP.md](ROADMAP.md) to see if your suggestion is already planned.
2. Check existing issues to avoid duplicates.
3. Create a new issue with the label "enhancement".
4. Clearly describe the feature and its benefits.
5. If possible, outline how the feature might be implemented.

### Pull Requests

We follow a standard GitHub flow for contributions:

1. Fork the repository.
2. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name` or `git checkout -b fix/issue-description`.
3. Make your changes, following our [coding standards](#coding-standards).
4. Add or update tests as necessary.
5. Update documentation to reflect your changes.
6. Commit your changes with clear, descriptive commit messages.
7. Push your branch to your fork: `git push origin your-branch-name`.
8. Submit a pull request to the `master` branch of the Eshkol repository.
9. Respond to any feedback or questions during the review process.

### Branch Protection and Required Checks

`master` is branch-protected. The protection rules themselves are applied in the
repository settings (Settings → Branches → branch protection rule for `master`);
this section is the authoritative reference for *what* that configuration
requires, checked directly against
`gh api repos/tsotchke/eshkol/branches/master/protection` on 2026-08-26. It is
the enforcement half of the closed-loop assurance architecture
(`docs/design/adr/0010-closed-loop-assurance.md`, closing AR1 gap A5): before this
was in place, `master` had zero required checks and every lane was advisory.

**Required status checks** (16 contexts; a PR cannot merge until all of these are
green):

- `guard` — enforces the commit-identity allowlist on every PR (workflow
  `identity-guard.yml`, job id `guard`).
- `assurance-gates`, `surface-manifest`, `wasm-execute-diff`.
- `linux-x64-asan-ubsan` — the memory-safety (ASan + UBSan) lane; required so
  address/UB regressions cannot merge behind an advisory lane.
- `linux-x64-xla`, `linux-arm64-xla`, `windows-arm64-xla`, `macos-arm64-xla`,
  `macos-x64-xla`.
- `linux-x64-cuda`, `linux-arm64-cuda`, `windows-x64-cuda`.
- `windows-arm64-lite`, `macos-arm64-lite`, `macos-x64-lite`.

**Other rule-set settings**, matching the live config exactly rather than an
aspirational description of it:

- `strict: false` — a branch does **not** have to be up to date with `master`
  before merging.
- `enforce_admins: true` — the rules apply to repository administrators too;
  there is no bypass.
- `required_linear_history: true`.
- No pull-request review count is required (there is no
  `required_pull_request_reviews` block on the rule), so there is nothing to
  "dismiss" on new commits — that behavior does not exist on this repository.
- Force pushes and branch deletion are both disabled on `master`.
- The repository is squash-only: `allow_squash_merge: true`,
  `allow_merge_commit: false`, `allow_rebase_merge: false`. Every PR lands on
  `master` as a single squash commit.

**Advisory checks** (run on every PR and must be *reviewed* before merge, but do
not mechanically block it):

- `linux-x64-lite`, `linux-arm64-lite` — demoted out of the required set; both
  lite lanes hit chronic hosted-runner reclaim (exit 143 / `BlobNotFound`)
  independent of code changes, so they cannot gate the default merge.
- `quantum-macos` — opt-in Moonlab quantum lane (`continue-on-error`, networked).
- `mesh-gate-advisory` — self-hosted mesh CI lane (see
  `docs/platform/SELF_HOSTED_RUNNERS.md`); skips rather than blocks when no
  self-hosted runner is online.
- `bench-smoke`.

Note that the XLA and CUDA lanes are **required**, not advisory, on this
repository — an earlier draft of this section described them as optional; the
live rule set does not.

Advisory does not mean ignorable: a reviewer checks **all** lanes, including the
advisory ones, before merging, and dedupes stale check entries by taking the latest
run per lane name. Re-run the failed jobs once before treating a lite-lane red as a
real regression.

**Release-blocking readiness.** Publishing a release is additionally gated by the
`release-readiness-gate` job in `.github/workflows/release.yml`, which regenerates
the oracle traces at the tagged SHA and runs `icc architecture-verify` +
`icc readiness --target v1.3-evolve`. `publish-release` depends on it, so **no
release asset is published unless readiness is ready/100** at the cut SHA. The gate
requires ICC to be provisioned on the release runner via the `ICC_BIN` repository
variable (a path to the ICC binary; optionally `ICC_REPO` for the registered index
name, default `eshkol_lang`). If ICC is unavailable on a real tag push the gate
emits a loud error and blocks the release — it never fail-opens to a green publish.
A non-publishing `workflow_dispatch` dry-run treats the same conditions as advisory
warnings, since it ships nothing.

## Development Guidelines

### Coding Standards

We follow these coding standards for consistency:

#### C Code

- Use 4 spaces for indentation (no tabs).
- Follow the [Linux kernel coding style](https://www.kernel.org/doc/html/latest/process/coding-style.html) with some modifications:
  - Use camelCase for function names and variables.
  - Use PascalCase for struct and enum names.
  - Use ALL_CAPS for macros and constants.
- Prefix all public functions with `eshkol_`.
- Keep lines under 100 characters when possible.
- Add comments for complex logic or non-obvious behavior.
- Use descriptive variable and function names.

#### TypeScript Code (MCP Tools)

- Use 2 spaces for indentation.
- Follow the [TypeScript Style Guide](https://google.github.io/styleguide/tsguide.html).
- Use camelCase for variables and functions.
- Use PascalCase for classes, interfaces, and type aliases.
- Use descriptive names and add JSDoc comments.

### Documentation

Good documentation is crucial for the project:

- Update relevant documentation when making changes.
- Document all public APIs with clear descriptions, parameters, and return values.
- Add examples for complex features.
- Keep the README and other high-level documentation up to date.
- Use Markdown for all documentation files.

#### API Reference (docs/api/)

`docs/api/` is a generated browsable reference for the public C/C++ headers
under `inc/eshkol/**/*.h`, harvested from their Doxygen `/** @brief ... */`
comment blocks by `scripts/gen_api_docs.py`. It is not hand-edited.

When you add or change a Doxygen comment on a public header symbol,
regenerate the reference and commit the result alongside your change:

```sh
make api-docs          # regenerate docs/api/
make api-docs-check    # verify docs/api/ has no drift (used before a PR)
```

The generator is documentation-only — it never modifies files under `inc/`
— and its output is deterministic (sorted, stable across re-runs), so a
regeneration with no underlying comment changes produces an empty diff. It
is not run automatically in CI; regenerate locally when you touch a header
comment.

### Testing

We strive for good test coverage:

- Add tests for new features.
- Update tests when modifying existing features.
- Ensure all tests pass before submitting a pull request.
- Follow the existing test patterns in the codebase.

## Project Structure

Understanding the project structure will help you contribute effectively:

```
eshkol/
├── build/                  # Build output (generated)
├── docs/                   # Documentation
│   ├── architecture/       # Architecture overview
│   ├── breakdown/          # Deep-dive technical docs (20 files)
│   ├── components/         # Component documentation
│   ├── development/        # Development workflow
│   └── vision/             # Vision and design history
├── exe/                    # Compiler and REPL executables
│   ├── eshkol-run.cpp      # AOT compiler
│   └── eshkol-repl.cpp     # JIT REPL
├── inc/eshkol/             # Public header files
│   ├── backend/            # Code generation headers (21 modules)
│   ├── core/               # Runtime headers (logic, inference, workspace)
│   ├── frontend/           # Parser, macro expander headers
│   └── types/              # Type system headers
├── lib/                    # Implementation source code
│   ├── backend/            # LLVM codegen (21 modules)
│   │   ├── gpu/            # GPU backends (Metal, CUDA)
│   │   └── xla/            # XLA/StableHLO backend
│   ├── core/               # Runtime (arena, AST, logic, inference, workspace)
│   ├── frontend/           # Parser, macro expander
│   ├── math/               # Math stdlib (special functions, ODE, constants)
│   ├── signal/             # Signal processing (FFT, filters)
│   ├── ml/                 # Machine learning stdlib
│   ├── random/             # Random number generators
│   ├── web/                # Web/WASM platform
│   ├── repl/               # JIT compiler
│   └── types/              # Type checker, HoTT types
├── tests/                  # Test suite (45 suites by feature)
│   ├── autodiff/           # AD tests (3 modes)
│   ├── bignum/             # Arbitrary-precision integer tests
│   ├── complex/            # Complex number tests
│   ├── features/           # Core language features
│   ├── logic/              # Consciousness engine tests
│   ├── numeric/            # Numeric regression tests
│   ├── parallel/           # Parallel primitives tests
│   ├── signal/             # Signal processing tests
│   └── ...                 # 26 more test directories
├── tools/                  # Developer tools
│   ├── lsp/                # Language Server Protocol
│   ├── pkg/                # Package manager (eshkol-pkg)
│   └── vscode-eshkol/      # VSCode extension
├── scripts/                # Build and test scripts
├── CMakeLists.txt          # Main build configuration
├── CONTRIBUTING.md         # This file
├── LICENSE                 # MIT license
├── README.md               # Project overview
└── ROADMAP.md              # Development roadmap
```

## Communication

- **GitHub Issues**: For bug reports, feature requests, and specific technical discussions.
- **GitHub Discussions**: For general questions, ideas, and community discussions.
- **Pull Requests**: For code contributions and code reviews.

## Priority Areas for Contribution (v1.4+)

v1.0-foundation, v1.1-accelerate, v1.2-scale, and v1.3.0-evolve (arbitrary-order
AD, full R7RS conformance, closure/TCO/memory hardening) are **complete**, and
v1.3.1 through v1.3.4-evolve have landed further robustness for long-running,
resident programs, an opt-in differentiable quantum stack, and a
consumer-hardening correctness wave (automatic per-iteration reclamation,
race-free `parallel-map`, exact gradients through every callable form, R7RS
exactness contagion on both engines). We welcome contributions for upcoming
releases:

### Immediate Priorities (v1.4-connection)
1. **TCP/UDP Sockets**: Linear resource types with guaranteed close
2. **TLS/SSL**: Via system libraries
3. **HTTP Client**: Built on sockets + TLS
4. **Linear Types for Handles**: `open -> borrowed -> closed` with compile-time tracking
5. **Debugger**: REPL step-through with breakpoints + variable inspection

Already delivered ahead of this list: the portable event loop (kqueue / epoll /
IOCP) shipped in v1.3.4-evolve, `eshkol-doc` shipped in v1.3.2-evolve, and the
linear-type machinery landed in v1.3.4-evolve as the linear `Qubit` type —
extending it to handles is what remains.

### Near-Term (v1.5-intelligence - August 2026)
1. **Neural-Symbolic Search**: Differentiable logic programs (building on v1.1 consciousness engine)
2. **Symbol Embeddings & Soft Unification**: Differentiable similarity over the knowledge base
3. **LSTM/GRU Cells**: Standard recurrent neural architectures
4. **Multi-GPU Support**: Distribute work across multiple GPUs

### Ongoing
1. **Documentation**: Tutorials, examples, case studies
2. **Testing**: Expanded test coverage, benchmarking
3. **Standard Library**: Additional modules and utilities
4. **Bug Fixes**: Report and fix any issues found

See [ROADMAP.md](ROADMAP.md) for complete development plans.

## Recognition

We value all contributions and will recognize contributors in our release notes and on the project website. Significant contributors may be invited to join the core team.

Thank you for contributing to Eshkol!
