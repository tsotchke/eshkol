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
  - [Development Guidelines](#development-guidelines)
    - [Coding Standards](#coding-standards)
      - [C Code](#c-code)
      - [TypeScript Code (MCP Tools)](#typescript-code-mcp-tools)
    - [Documentation](#documentation)
    - [Testing](#testing)
  - [Project Structure](#project-structure)
  - [Communication](#communication)
  - [Priority Areas for Contribution (v1.2+)](#priority-areas-for-contribution-v12)
    - [Immediate Priorities (v1.2-scale - Q2 2026)](#immediate-priorities-v12-scale---q2-2026)
    - [Near-Term (v1.5-intelligence - Q3 2026)](#near-term-v15-intelligence---q3-2026)
    - [Ongoing](#ongoing)
  - [Recognition](#recognition)

## Code of Conduct

We expect all contributors to adhere to our Code of Conduct. Please be respectful and considerate of others when participating in our community.

## Getting Started

### Development Environment Setup

To set up your development environment for Eshkol, you'll need:

1. **C/C++ Compiler**
   - GCC 9.0+ or Clang 10.0+
   - On macOS: `brew install gcc` or use the default Clang
   - On Linux: `sudo apt install build-essential`
   - On Windows: Visual Studio 2022 with Desktop development for C++ and Clang tools

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

4. **Git**
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

# Rebuild the browser REPL VM
emcc -O2 -s WASM=1 -s MODULARIZE=1 -s EXPORT_NAME='EshkolVM' \
  -DESHKOL_VM_WASM -DESHKOL_VM_NO_DISASM \
  -I lib/backend lib/backend/vm_wasm_repl.c \
  -o site/static/eshkol-vm.js -lm

# Serve locally
cd site/static && python3 -m http.server 8888
```

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
├── tests/                  # Test suite (35 suites by feature)
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

## Priority Areas for Contribution (v1.2+)

v1.0-foundation and v1.1-accelerate are **complete**. We welcome contributions for upcoming releases:

### Immediate Priorities (v1.2-scale - Q2 2026)
1. **Distributed Computing**: Multi-node gradient synchronization
2. **Model Deployment**: ONNX/TFLite/CoreML export
3. **Vulkan Compute**: Cross-platform GPU backend (Metal and CUDA already shipped in v1.1)
4. **Visual Debugger**: Source-level debugging UI
5. **Python Bindings**: Interoperability with Python ML ecosystem

### Near-Term (v1.5-intelligence - Q3 2026)
1. **Neural-Symbolic Search**: Differentiable logic programs (building on v1.1 consciousness engine)
2. **Multi-GPU Support**: Distribute work across multiple GPUs
3. **Profile-Guided Optimization**: Runtime profile-based optimization
4. **Full R7RS Library System**: `define-library` / `import` with renaming

### Ongoing
1. **Documentation**: Tutorials, examples, case studies
2. **Testing**: Expanded test coverage, benchmarking
3. **Standard Library**: Additional modules and utilities
4. **Bug Fixes**: Report and fix any issues found

See [ROADMAP.md](ROADMAP.md) for complete development plans.

## Recognition

We value all contributions and will recognize contributors in our release notes and on the project website. Significant contributors may be invited to join the core team.

Thank you for contributing to Eshkol!
