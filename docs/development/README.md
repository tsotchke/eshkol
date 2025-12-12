# Eshkol Development Documentation

This directory indexes development workflow documentation.

## Building Eshkol

### Prerequisites

- LLVM 14+ (required)
- CMake 3.16+ (required)
- C++17 compiler: GCC 9+, Clang 10+, or MSVC 2019+ (required)

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

Eshkol includes comprehensive test suites across 12 categories:

```bash
# Run specific test suite
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
./scripts/run_neural_tests.sh
./scripts/run_tensor_tests.sh

# Run all tests
for script in scripts/run_*_tests.sh; do
    echo "Running $script..."
    $script || echo "FAILED: $script"
done
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

## Contributing

### Code Structure

```
eshkol/
├── exe/              # Compiler and REPL executables
├── inc/eshkol/       # Public headers
│   ├── backend/      # LLVM codegen headers (19 modules)
│   ├── frontend/     # Parser, macro expander headers
│   └── types/        # Type system headers
├── lib/              # Implementation
│   ├── backend/      # LLVM codegen (19 .cpp files)
│   ├── core/         # Runtime (arena, AST, display)
│   ├── frontend/     # Parser, macro expander
│   ├── repl/         # JIT compiler
│   └── types/        # Type checker, HoTT types
├── tests/            # Test suite (12 categories)
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

**Current**: v1.0.0-foundation (production-ready)

See [Roadmap](../breakdown/ROADMAP.md) for planned features.

## See Also

- [Compilation Guide](../breakdown/COMPILATION_GUIDE.md) - Build process, debugging
- [Compiler Architecture](../breakdown/COMPILER_ARCHITECTURE.md) - Implementation details
- [Feature Matrix](../FEATURE_MATRIX.md) - Implementation status
