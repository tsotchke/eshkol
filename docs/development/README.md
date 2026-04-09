# Eshkol Development Documentation

This directory indexes development workflow documentation.

## Building Eshkol

### Prerequisites

- LLVM 21 (required for lite/native builds)
- CMake 3.14+ (required)
- C++20 compiler: GCC 10+, Clang 13+, or MSVC 2022+ (required)

### Build from Source

```bash
# Clone repository
git clone https://github.com/tsotchke/eshkol.git
cd eshkol

# Configure
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build (parallel)
cmake --build build -j$(nproc)

# Test executables
build/eshkol-run   # AOT compiler
build/eshkol-repl  # JIT REPL
```

See [Getting Started](../breakdown/GETTING_STARTED.md) for detailed installation instructions.

## Running Tests

Eshkol includes 35 test suites with 438 test files:

```bash
# Run all tests
./scripts/run_all_tests.sh

# Run specific test suite
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
./scripts/run_ml_tests.sh
```

### Test Categories

- **autodiff/** - Automatic differentiation (3 modes, vector calculus)
- **lists/** - List operations, higher-order functions, closures
- **neural/** - Neural network operations
- **ml/** - Machine learning primitives (tensor operations)
- **types/** - Type system (HoTT, gradual typing)
- **features/** - Language features (exception handling, macros, etc.)
- **memory/** - OALR memory management (linear types, borrowing)
- **modules/** - Module system
- **stdlib/** - Standard library functions
- **json/** - JSON parsing
- **system/** - System operations (display, hash tables)
- **migration/** - Pointer consolidation tests
- **consciousness/** - Logic programming, factor graphs, workspace
- **signal/** - FFT, filters, window functions
- **parallel/** - Parallel primitives, futures
- **gpu/** - GPU dispatch, Metal/CUDA tests
- **bignum/** - Arbitrary-precision integer tests
- **rational/** - Exact rational arithmetic tests
- **continuations/** - call/cc, dynamic-wind, guard/raise
- **bytevector/** - R7RS bytevector operations
- **io/** - File I/O, port operations
- **benchmark/** - Performance benchmarks

## Contributing

### Code Structure

```
eshkol/
├── exe/              # Compiler and REPL executables
├── inc/eshkol/       # Public headers
│   ├── backend/      # LLVM codegen headers (21 modules)
│   ├── frontend/     # Parser, macro expander headers
│   └── types/        # Type system headers
├── lib/              # Implementation
│   ├── backend/      # LLVM codegen (19 .cpp files)
│   │   ├── gpu/      # GPU backends (Metal, CUDA)
│   │   └── xla/      # XLA/StableHLO backend
│   ├── core/         # Runtime (arena, AST, display, logic, inference, workspace)
│   ├── frontend/     # Parser, macro expander
│   ├── math/         # Math stdlib (special functions, ODE, constants)
│   ├── signal/       # Signal processing (FFT, filters)
│   ├── ml/           # Machine learning (optimization, activations)
│   ├── random/       # Random number generators
│   ├── web/          # Web/WASM platform (DOM API)
│   ├── tensor/       # Tensor utilities
│   ├── repl/         # JIT compiler
│   └── types/        # Type checker, HoTT types
├── tests/            # Test suite (35 suites, 438 files)
├── tools/            # Developer tools
│   ├── lsp/          # Language Server Protocol
│   ├── pkg/          # Package manager (eshkol-pkg)
│   └── vscode-eshkol/ # VSCode extension
└── scripts/          # Build and test scripts
```

### Development Workflow

1. Make changes to source files
2. Rebuild: `cmake --build build`
3. Run relevant tests: `./scripts/run_*_tests.sh`
4. Verify no regressions in other test suites
5. Update documentation if APIs changed

### Adding New Features

When implementing new language features:

1. Update AST structures in [`inc/eshkol/eshkol.h`](../../inc/eshkol/eshkol.h)
2. Extend parser in [`lib/frontend/parser.cpp`](../../lib/frontend/parser.cpp)
3. Add type checking in [`lib/types/type_checker.cpp`](../../lib/types/type_checker.cpp)
4. Implement LLVM codegen (create new module or extend existing)
5. Add comprehensive tests
6. Document in relevant docs/breakdown/ files

## Project Status

**Current**: v1.1.13-accelerate (production-ready)

See [Roadmap](../breakdown/ROADMAP.md) for planned features.

## See Also

- [Compilation Guide](../breakdown/COMPILATION_GUIDE.md) - Build process, debugging
- [Compiler Architecture](../breakdown/COMPILER_ARCHITECTURE.md) - Implementation details
- [Feature Matrix](../FEATURE_MATRIX.md) - Implementation status
