# Test Coverage - Eshkol v1.0-Architecture

**Test Infrastructure**: 12 test categories, 300+ test files, comprehensive coverage across all major systems.

## Test Execution

```bash
# Run all test suites
for script in scripts/run_*_tests.sh; do
    echo "=== Running $script ==="
    $script || echo "FAILED"
done

# Run specific suite
./scripts/run_autodiff_tests.sh
./scripts/run_list_tests.sh
./scripts/run_neural_tests.sh
```

---

## Test Categories

### 1. Automatic Differentiation (tests/autodiff/)

**Coverage**: 3 AD modes, vector calculus, nested gradients

**Test count**: 50+ files

**Key tests:**
- `validation_01_type_detection.esk` - AD type system integration
- `validation_02_graph_construction.esk` - Computational graph correctness
- `validation_03_backward_pass.esk` - Backpropagation algorithm
- `validation_04_tensor_integration.esk` - Tensor + AD interaction
- `phase3_real_ad_test.esk` - Reverse-mode AD
- `phase4_vector_calculus_test.esk` - Gradient, jacobian, hessian
- `test_gradient_composition.esk` - Higher-order gradients
- `test_jacobian_display.esk` - Jacobian matrix display

**Run**: `./scripts/run_autodiff_tests.sh`

### 2. List Operations (tests/lists/)

**Coverage**: Cons cells, higher-order functions, closures, Phase 3B tagged cons

**Test count**: 100+ files

**Key tests:**
- `comprehensive_list_test.esk` - All list operations
- `phase3_polymorphic_completion_test.esk` - Polymorphic list functions
- `map_comprehensive_test.esk` - Map with various input types
- `lambda_sexpr_homoiconic_test.esk` - Homoiconicity
- `test_production_advanced.esk` - Production patterns

**Run**: `./scripts/run_list_tests.sh`

### 3. Neural Networks (tests/neural/)

**Coverage**: Layers, activation functions, training loops

**Test count**: 7 files

**Key tests:**
- `nn_working.esk` - Complete neural network
- `nn_training.esk` - Backpropagation training
- `nn_computation.esk` - Forward pass
- `nn_minimal.esk` - Minimal network

**Run**: `./scripts/run_neural_tests.sh`

### 4. Machine Learning Primitives (tests/ml/)

**Coverage**: Tensor operations, linear algebra, matrix operations

**Test count**: 5 files

**Key tests:**
- `impressive_demo.esk` - Comprehensive ML demo
- `matmul_test.esk` - Matrix multiplication
- `tensor_dot_test.esk` - Dot products
- `transpose_test.esk` - Matrix transpose

**Run**: `./scripts/run_ml_tests.sh`

### 5. Type System (tests/types/)

**Coverage**: HoTT types, gradual typing, type inference

**Test count**: 3 files

**Key tests:**
- `hott_comprehensive_test.esk` - HoTT type system
- `mixed_types_stress_test.esk` - Mixed type operations
- `hott_types_test.cpp` - C++ type system tests

**Run**: `./scripts/run_types_tests.sh`

### 6. Language Features (tests/features/)

**Coverage**: Exception handling, pattern matching, macros, stress tests

**Test count**: 8 files

**Key tests:**
- `ultimate_math_stress.esk` - Extreme mathematical operations
- `extreme_stress_test.esk` - Language feature stress
- `exception_test.esk` - Exception handling (`guard`, `raise`)
- `pattern_matching_test.esk` - Pattern match
- `hott_return_types.esk` - HoTT type annotations

**Run**: `./scripts/run_features_tests.sh`

### 7. Memory Management (tests/memory/)

**Coverage**: OALR, linear types, borrowing, arena allocation

**Test count**: 6 files

**Key tests:**
- `double_move.esk` - Detects double-move errors
- `use_after_move.esk` - Detects use-after-move
- `move_while_borrowed.esk` - Borrow checker
- `valid_ownership.esk` - Correct linear type usage
- `region_test.esk` - Arena regions

**Run**: `./scripts/run_memory_tests.sh`

### 8. Module System (tests/modules/)

**Coverage**: `require`, `provide`, visibility, circular dependencies

**Test count**: 5 files

**Key tests:**
- `module_test.esk` - Basic module loading
- `visibility_test.esk` - Symbol visibility
- `cycle_test.esk` - Circular dependency detection
- `stdlib_test.esk` - Standard library modules

**Run**: `./scripts/run_modules_tests.sh`

### 9. Standard Library (tests/stdlib/)

**Coverage**: String ops, CSV, list utilities

**Test count**: 6 files

**Key tests:**
- `csv_comprehensive_test.esk` - CSV parsing
- `strings_closure_test.esk` - String operations with closures
- `query_only_test.esk` - List query functions

**Run**: `./scripts/run_stdlib_tests.sh`

### 10. JSON (tests/json/)

**Coverage**: JSON parsing, serialization, file I/O

**Test count**: 3 files

**Key tests:**
- `json_test.esk` - JSON parsing
- `json_file_io_test.esk` - JSON file operations
- `json_stress_test.esk` - Large JSON documents

**Run**: `./scripts/run_json_tests.sh`

### 11. System Operations (tests/system/)

**Coverage**: Hash tables, display, I/O

**Test count**: 8 files

**Key tests:**
- `hash_test.esk` - Hash table operations
- `display_int_test.esk` - Display system
- `system_test.esk` - System primitives

**Run**: `./scripts/run_system_tests.sh`

### 12. Macros (tests/macros/)

**Coverage**: `define-syntax`, `syntax-rules`, hygiene

**Test count**: 3 files

**Key tests:**
- `basic_macro_test.esk` - Macro expansion
- `minimal_macro_test.esk` - Simple macros

**Run**: (No dedicated script - run via features tests)

---

## Test Statistics

**Total test files**: 300+

**Coverage areas**:
- ✅ Core language (lists, closures, macros)
- ✅ Type system (HoTT, gradual typing)
- ✅ Memory management (OALR, linear types)
- ✅ Automatic differentiation (3 modes, vector calculus)
- ✅ Tensors and linear algebra
- ✅ Neural networks
- ✅ Standard library
- ✅ Module system
- ✅ Exception handling
- ✅ JSON parsing
- ✅ Hash tables
- ✅ File I/O

**Test philosophy**: Comprehensive coverage ensures production readiness. Every major feature has test coverage.

---

## See Also

- [Master Architecture](ESHKOL_V1_ARCHITECTURE.md) - What's being tested
- [Feature Matrix](FEATURE_MATRIX.md) - Implementation status
- [Development Guide](docs/development/README.md) - Running tests