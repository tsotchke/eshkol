# Eshkol Master Development Plan
**Version 1.0 | November 2025**
**24-Month Roadmap to Production Neuro-Symbolic Computing Platform**

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Vision & Differentiation](#vision--differentiation)
3. [Current State](#current-state)
4. [Phase 1: Foundation & Core Language (Months 1-6)](#phase-1-foundation--core-language-months-1-6)
5. [Phase 2: Native Scientific Computing (Months 7-12)](#phase-2-native-scientific-computing-months-7-12)
6. [Phase 3: Symbolic & Neural DSL (Months 13-18)](#phase-3-symbolic--neural-dsl-months-13-18)
7. [Phase 4: Formal Verification (Months 19-24)](#phase-4-formal-verification-months-19-24)
8. [Team & Resources](#team--resources)
9. [Success Metrics](#success-metrics)
10. [Risk Mitigation](#risk-mitigation)

---

## Executive Summary

### The Vision

Eshkol is not just another Scheme implementation. It is a groundbreaking tri-language ecosystem combining:

- **Eshkol (Scheme-based)**: High-performance scientific computing with LLVM backend
- **Haskell Integration**: Advanced type system and functional purity
- **Lean Integration**: Dependent types, formal verification, HoTT foundations

**Unique Value Proposition**: The world's first formally verified neural network framework with proof-carrying AI systems.

### Timeline

- **Months 1-6**: Foundation (homoiconic metaprogramming, I/O, modules)
- **Months 7-12**: Native scientific computing (tensors, autodiff, GPU)
- **Months 13-18**: Neuro-symbolic AI (neural DSL + symbolic reasoning)
- **Months 19-24**: Formal verification (HoTT, Lean integration, eprover)

### Key Differentiators

1. **Only formally verified neural network framework** in existence
2. **Native homoiconic ML language** (not Python with Lisp syntax)
3. **Proof-carrying AI systems** (mathematical guarantees)
4. **Tri-language ecosystem** (Eshkol-Haskell-Lean)
5. **HoTT foundations** for AI (cutting-edge type theory)

---

## Vision & Differentiation

### Why No PyTorch?

We are building a **native Eshkol ecosystem** with:
- Native tensor operations (LLVM-based, not bindings)
- Custom autodiff system (already partially implemented)
- GPU acceleration via LLVM (CUDA/OpenCL/Vulkan)
- Formally verified implementations (impossible with Python FFI)

### Research Contributions

- Verified neural network compilation
- HoTT-based AI systems
- Neuro-symbolic integration with formal proofs
- LLVM-based GPU code generation for functional languages

### Target Users

- AI safety researchers (need provable guarantees)
- High-assurance systems (aerospace, medical, finance)
- Scientific computing (performance + correctness)
- Academic research (type theory + ML intersection)

---

## Current State

### Repository Status

**Working Directory**: `/Users/tyr/Desktop/eshkol` (current LLVM implementation)
**Original Docs**: `/Users/tyr/Desktop/tscheme/docs` (vision documents)

### What Exists ✅

- LLVM backend (functional, in production use)
- Arena memory management (90% complete)
- Mixed-type list operations (just completed)
- Basic Scheme (~65% R7RS compliance)
- Type system foundation (gradual typing)
- Autodiff (partial, has bugs)

### What's Working 🟡

- Higher-order functions (need rewrite for type system)
- Autodiff (exists but unreliable - SCH-006, 007, 008)
- Vector operations (70% complete)
- SIMD optimization (60% complete)

### Critical Gaps ❌

- No eval, macros, quasiquotation
- No module system, REPL
- No file I/O, serialization
- No native tensor operations
- No pattern matching
- No GPU support
- No formal verification

### Known Issues to Fix

- **SCH-006**: Type inference for autodiff incomplete
- **SCH-007**: Vector return types not handled
- **SCH-008**: Type conflicts in generated code
- **SCH-011**: Lambda capture analysis incomplete
- **Memory issues**: Some edge cases in list operations

---

## Phase 1: Foundation & Core Language (Months 1-6)

**Goal**: Production-ready base Eshkol with homoiconic metaprogramming

### Timeline: 120 sessions (2-4 hours each)

---

### Month 1: Stabilization (Sessions 1-20)

#### Week 1: Mixed-Type Completion (Sessions 1-10)

**Session 1-2: Commit & Build**
- [ ] **Files**: `lib/backend/llvm_codegen.cpp`, `lib/core/arena_memory.cpp`
- [ ] Commit unstaged mixed-type changes
- [ ] Add `tests/phase_2a_group_a_test.esk` to git
- [ ] Rebuild: `cmake --build build`
- [ ] Test: `./build/eshkol-run tests/phase_2a_group_a_test.esk`
- [ ] Create `docs/BUILD_STATUS.md` tracking current state

**Session 3-4: Analysis**
- [ ] **Files**: `lib/backend/llvm_codegen.cpp` (lines 5182-5801)
- [ ] Read all higher-order function implementations
- [ ] Document which use old struct access (`CreateStructGEP`)
- [ ] Create `docs/HIGHER_ORDER_REWRITE_PLAN.md` with:
  - Complete function list needing updates
  - Required changes per function
  - Test cases for each
  - Priority order: map → filter → fold → for-each

**Session 5-6: Rewrite map (single-list)**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenMapSingleList` (line ~5182)
- [ ] Replace line 5213 `CreateStructGEP` with `extractCarAsTaggedValue`
- [ ] Replace cdr access with `arena_tagged_cons_get_ptr`
- [ ] Update PHI nodes to use `tagged_value_type`
- [ ] **Test**: `(map (lambda (x) (* x 2)) (list 1 2.5 3))`
- [ ] Expected output: `(2 5.0 6)`

**Session 7-8: Rewrite map (multi-list)**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenMapMultiList` (line ~5276)
- [ ] Extract cars using `extractCarAsTaggedValue` for each list
- [ ] Update cdr iteration for all input lists
- [ ] Handle lists of different lengths
- [ ] **Test**: `(map + (list 1 2 3) (list 4.5 5.5 6.5))`
- [ ] Expected output: `(5.5 7.5 9.5)`

**Session 9-10: Rewrite filter**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenFilter` (line ~5396)
- [ ] Extract elements using `extractCarAsTaggedValue`
- [ ] Update predicate application for tagged values
- [ ] Update result list building
- [ ] **Test**: `(filter (lambda (x) (> x 5)) (list 1 8.5 3 9))`
- [ ] Expected output: `(8.5 9)`

#### Week 2: More Higher-Order Functions (Sessions 11-20)

**Session 11-12: Rewrite fold**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenFold` (line ~5513)
- [ ] Update accumulator handling for tagged values
- [ ] Fix loop iteration with tagged extraction
- [ ] **Test**: `(fold + 0 (list 1 2.5 3))`
- [ ] Expected output: `6.5`

**Session 13-14: Implement fold-right**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenFoldRight` (currently stub at line 5807)
- [ ] Implement right-to-left folding
- [ ] Use recursive approach or reverse list
- [ ] **Test**: `(fold-right cons '() (list 1 2 3))`
- [ ] Expected output: `(1 2 3)`

**Session 15-16: Rewrite for-each**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Function**: `codegenForEachSingleList` (line ~5759)
- [ ] Replace struct access (lines 5784-5789)
- [ ] Use `extractCarAsTaggedValue` for element extraction
- [ ] Update cdr navigation
- [ ] **Test**: `(for-each display (list 1 2.5 3))`
- [ ] Should print: `12.53` (no spaces in basic display)

**Session 17-18: Update member/assoc family**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] **Functions**: `codegenMember`, `codegenAssoc` (~line 5813+)
- [ ] Update all three: member/memq/memv
- [ ] Update all three: assoc/assq/assv
- [ ] Use tagged value extraction
- [ ] **Test**:
  ```scheme
  (member 2.5 (list 1 2.5 3))     ; => (2.5 3)
  (assoc 'b (list (list 'a 1) (list 'b 2.5)))  ; => (b 2.5)
  ```

**Session 19-20: Update utility functions**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] Update `find`, `partition`, `split-at`
- [ ] Update `remove`, `remq`, `remv`
- [ ] Update `take` if needed
- [ ] Create comprehensive test file: `tests/phase_2a_group_b_test.esk`

---

### Month 2: Autodiff Fixes & Examples (Sessions 21-40)

#### Week 3: Fix Autodiff Type Bugs (Sessions 21-30)

**Session 21-22: Investigate SCH-006**
- [ ] **Files**: `lib/backend/llvm_codegen.cpp`, `lib/frontend/type_inference/`
- [ ] Review issue: Type inference for autodiff incomplete
- [ ] Locate autodiff type inference code
- [ ] Create minimal test cases reproducing bug
- [ ] Document findings in `docs/AUTODIFF_TYPE_BUGS.md`

**Session 23-24: Fix SCH-007 (Vector Returns)**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] Review issue: Vector return types not handled
- [ ] Find vector return code generation
- [ ] Implement proper LLVM vector type returns
- [ ] **Test**: `(gradient (lambda (v) (dot v v)) #(1 2 3))`
- [ ] Expected: Return vector `#(2 4 6)`

**Session 25-26: Fix SCH-008 (Type Conflicts)**
- [ ] **File**: `lib/backend/llvm_codegen.cpp`
- [ ] Review issue: Type conflicts in generated code
- [ ] Identify conflicting type declarations in LLVM IR
- [ ] Implement type unification where needed
- [ ] Test compilation of complex autodiff programs

**Session 27-28: Test Autodiff Thoroughly**
- [ ] Create `tests/autodiff_comprehensive.esk`
- [ ] Test forward mode: `(d/dx (lambda (x) (* x x)) 5)`
- [ ] Test reverse mode: `(gradient f v)`
- [ ] Test composition: `(d/dx (compose f g) x)`
- [ ] Test vector functions
- [ ] Document any remaining issues

**Session 29-30: Autodiff Performance**
- [ ] Benchmark autodiff operations
- [ ] Optimize tape-based reverse mode
- [ ] Implement checkpointing for memory
- [ ] Document performance characteristics

#### Week 4: Examples & Documentation (Sessions 31-40)

**Session 31-32: Identify Core Examples**
- [ ] **Directory**: `examples/`
- [ ] Review all 100 example files
- [ ] Identify 30 most important examples
- [ ] Categorize by topic:
  - Basic syntax (5 examples)
  - List operations (5 examples)
  - Higher-order functions (5 examples)
  - Numerical computing (5 examples)
  - Autodiff (5 examples)
  - Advanced features (5 examples)

**Session 33-34: Update Syntax (Part 1)**
- [ ] **Files**: `examples/*.esk` (first 15)
- [ ] Update for current `main` syntax
- [ ] Update for current `define` syntax
- [ ] Test each example runs correctly
- [ ] Create `docs/SYNTAX_MIGRATION.md` documenting changes:
  ```scheme
  ; OLD SYNTAX
  (define main (lambda () ...))

  ; NEW SYNTAX
  (main ...)
  ```

**Session 35-36: Update Syntax (Part 2)**
- [ ] **Files**: `examples/*.esk` (remaining 15)
- [ ] Continue fixing examples
- [ ] Add comments explaining key concepts
- [ ] Ensure output is clear and educational

**Session 37-38: Create New Examples**
- [ ] Create `examples/mixed_types_demo.esk`
- [ ] Create `examples/higher_order_demo.esk`
- [ ] Create `examples/autodiff_tutorial.esk`
- [ ] Create `examples/vector_operations.esk`
- [ ] All should be well-commented and educational

**Session 39-40: Documentation Updates**
- [ ] Update `README.md` with current status
- [ ] Remove claims about unimplemented features
- [ ] Add "Quick Start" section
- [ ] Add "Examples" section with links
- [ ] Update feature matrix showing what works

---

### Month 3: Infrastructure (Sessions 41-60)

#### Week 5: CI/CD & Packaging (Sessions 41-50)

**Session 41-42: GitHub Actions CI**
- [ ] **File**: `.github/workflows/ci.yml` (new)
- [ ] Create workflow for Ubuntu 22.04:
  ```yaml
  name: CI
  on: [push, pull_request]
  jobs:
    build-linux:
      runs-on: ubuntu-latest
      steps:
        - uses: actions/checkout@v3
        - name: Install LLVM
          run: sudo apt-get install llvm-14-dev clang-14
        - name: Build
          run: |
            mkdir build
            cd build
            cmake ..
            make -j$(nproc)
        - name: Test
          run: |
            cd build
            ./eshkol-run ../tests/phase_2a_group_a_test.esk
  ```

**Session 43-44: macOS CI**
- [ ] **File**: `.github/workflows/ci.yml`
- [ ] Add macOS workflow:
  ```yaml
    build-macos:
      runs-on: macos-latest
      steps:
        - uses: actions/checkout@v3
        - name: Install LLVM
          run: brew install llvm
        - name: Build
          run: |
            export LLVM_DIR=$(brew --prefix llvm)
            mkdir build && cd build
            cmake -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm ..
            make -j$(sysctl -n hw.ncpu)
  ```

**Session 45-46: Add Install Targets**
- [ ] **File**: `CMakeLists.txt`
- [ ] Add install commands:
  ```cmake
  # Install executable
  install(TARGETS eshkol-run
          RUNTIME DESTINATION bin
          COMPONENT runtime)

  # Install library
  install(TARGETS eshkol-static
          ARCHIVE DESTINATION lib
          COMPONENT development)

  # Install headers
  install(DIRECTORY inc/eshkol
          DESTINATION include
          COMPONENT development
          FILES_MATCHING PATTERN "*.h")

  # Install examples
  install(DIRECTORY examples/
          DESTINATION share/eshkol/examples
          COMPONENT examples)
  ```
- [ ] Test: `sudo make install`
- [ ] Verify installation to `/usr/local`

**Session 47-48: CPack for Debian**
- [ ] **File**: `CMakeLists.txt`
- [ ] Add CPack configuration:
  ```cmake
  include(CPack)
  set(CPACK_PACKAGE_NAME "eshkol")
  set(CPACK_PACKAGE_VERSION "1.0.0")
  set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
      "Formally verified neuro-symbolic computing platform")
  set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Eshkol Team")
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "llvm-14")
  set(CPACK_GENERATOR "DEB")
  ```
- [ ] Build package: `cpack`
- [ ] Test installation: `sudo dpkg -i eshkol_1.0.0_amd64.deb`

**Session 49-50: Docker Build System**
- [ ] **File**: `docker/ubuntu/release/Dockerfile` (new)
  ```dockerfile
  FROM ubuntu:22.04

  RUN apt-get update && apt-get install -y \
      build-essential \
      cmake \
      llvm-14-dev \
      clang-14

  WORKDIR /app
  COPY . .

  RUN mkdir build && cd build && \
      cmake .. && \
      make -j$(nproc) && \
      cpack

  CMD ["/bin/bash"]
  ```
- [ ] Test: `docker build -t eshkol-build -f docker/ubuntu/release/Dockerfile .`
- [ ] Extract .deb: `docker cp container:/app/build/*.deb .`

#### Week 6: Homebrew & Testing (Sessions 51-60)

**Session 51-52: Homebrew Formula**
- [ ] Create new repo: `homebrew-eshkol`
- [ ] **File**: `Formula/eshkol.rb`
  ```ruby
  class Eshkol < Formula
    desc "Formally verified neuro-symbolic computing platform"
    homepage "https://github.com/openSVM/eshkol"
    url "https://github.com/openSVM/eshkol/archive/v1.0.0.tar.gz"
    sha256 "..." # Will be computed on release

    depends_on "cmake" => :build
    depends_on "llvm"

    def install
      system "cmake", ".", *std_cmake_args
      system "make"
      system "make", "install"
    end

    test do
      (testpath/"test.esk").write "(display (+ 2 3))"
      assert_equal "5", shell_output("#{bin}/eshkol-run test.esk").strip
    end
  end
  ```
- [ ] Test local install: `brew install --build-from-source ./Formula/eshkol.rb`

**Session 53-54: Integration Tests**
- [ ] **Directory**: `tests/integration/` (new)
- [ ] Create `tests/integration/file_io_test.esk`
- [ ] Create `tests/integration/module_test.esk` (for later)
- [ ] Create `tests/integration/complex_computation.esk`
- [ ] Run all integration tests in CI

**Session 55-56: Memory Testing**
- [ ] Install Valgrind: `sudo apt-get install valgrind`
- [ ] Run on all tests:
  ```bash
  for test in tests/*.esk; do
    valgrind --leak-check=full --error-exitcode=1 \
      build/eshkol-run "$test"
  done
  ```
- [ ] Fix any memory leaks found
- [ ] Document memory testing in `docs/TESTING.md`

**Session 57-58: Documentation Cleanup**
- [ ] **Files**: `docs/aidocs/*.md`
- [ ] Review `GETTING_STARTED.md` - remove debugger references (lines 253-272)
- [ ] Review `COMPILATION_GUIDE.md` - remove profiler references
- [ ] Update with current reality
- [ ] Create `SECURITY.md` for vulnerability reporting

**Session 59-60: v1.0 Release Prep**
- [ ] Create `CHANGELOG.md`
- [ ] Create `RELEASE_NOTES_v1.0.md`
- [ ] Tag release: `git tag -a v1.0-foundation -m "v1.0 Foundation Release"`
- [ ] Create GitHub release with binaries
- [ ] Update website/docs with v1.0 info

---

### Month 4: Eval & Apply (Sessions 61-80)

#### Week 7-8: Runtime Code Execution (Sessions 61-80)

**Session 61-62: Design Eval System**
- [ ] **File**: `docs/EVAL_DESIGN.md` (new)
- [ ] Document eval execution model:
  - How eval interprets expressions
  - Environment handling
  - Interaction with compiled code
  - Security considerations
- [ ] Design API:
  ```scheme
  (eval expr)                    ; Use current environment
  (eval expr env)                ; Use specified environment
  (interaction-environment)      ; Get REPL environment
  ```

**Session 63-64: Environment Data Structure**
- [ ] **Files**: `lib/core/environment.h` (new), `lib/core/environment.c` (new)
  ```c
  typedef struct environment {
      struct environment* parent;
      hash_table_t* bindings;      // symbol -> value mapping
      bool is_mutable;             // can bindings change?
  } environment_t;

  environment_t* environment_create(environment_t* parent);
  void environment_destroy(environment_t* env);
  void environment_bind(environment_t* env, const char* name,
                       eshkol_value_t* value);
  eshkol_value_t* environment_lookup(environment_t* env,
                                     const char* name);
  bool environment_set(environment_t* env, const char* name,
                      eshkol_value_t* value);
  ```
- [ ] Implement environment chaining (lexical scope)
- [ ] Test environment operations

**Session 65-66: Eval for Literals**
- [ ] **File**: `lib/runtime/eval.c` (new)
  ```c
  eshkol_value_t* eshkol_eval(eshkol_ast_t* expr, environment_t* env) {
      switch (expr->type) {
          case AST_INTEGER:
              return value_create_integer(expr->data.integer.value);

          case AST_FLOAT:
              return value_create_float(expr->data.float_val.value);

          case AST_BOOLEAN:
              return value_create_boolean(expr->data.boolean.value);

          case AST_STRING:
              return value_create_string(expr->data.string.value);

          // More cases to come...
      }
  }
  ```
- [ ] Test: `(eval 42)` → 42
- [ ] Test: `(eval 3.14)` → 3.14
- [ ] Test: `(eval #t)` → #t

**Session 67-68: Eval for Variables**
- [ ] Implement variable lookup in eval
  ```c
  case AST_VARIABLE: {
      const char* name = expr->data.variable.name;
      eshkol_value_t* value = environment_lookup(env, name);
      if (!value) {
          eshkol_error("Unbound variable: %s", name);
          return NULL;
      }
      return value;
  }
  ```
- [ ] Test:
  ```scheme
  (define x 10)
  (eval 'x)  ; => 10
  ```

**Session 69-70: Eval for Quote**
- [ ] Implement quote handling
  ```c
  case AST_QUOTE:
      return ast_to_value(expr->data.quote.datum);
  ```
- [ ] Implement `ast_to_value` conversion
- [ ] Test: `(eval '(1 2 3))` → `(1 2 3)`

**Session 71-72: Eval for If**
- [ ] Implement conditional evaluation
  ```c
  case AST_IF: {
      eshkol_value_t* test = eshkol_eval(expr->data.if_expr.test, env);
      if (value_is_true(test)) {
          return eshkol_eval(expr->data.if_expr.consequent, env);
      } else {
          return eshkol_eval(expr->data.if_expr.alternate, env);
      }
  }
  ```
- [ ] Test: `(eval '(if #t 1 2))` → 1

**Session 73-74: Eval for Function Calls**
- [ ] Implement function application in eval
  ```c
  case AST_CALL: {
      eshkol_value_t* func = eshkol_eval(expr->data.call.callee, env);
      size_t argc = expr->data.call.arg_count;
      eshkol_value_t** args = malloc(argc * sizeof(eshkol_value_t*));

      for (size_t i = 0; i < argc; i++) {
          args[i] = eshkol_eval(expr->data.call.args[i], env);
      }

      eshkol_value_t* result = apply_function(func, args, argc, env);
      free(args);
      return result;
  }
  ```

**Session 75-76: Implement Apply**
- [ ] **File**: `lib/runtime/apply.c` (new)
  ```c
  eshkol_value_t* apply_function(eshkol_value_t* func,
                                 eshkol_value_t** args,
                                 size_t argc,
                                 environment_t* env) {
      if (func->type == VALUE_BUILTIN) {
          // Call builtin function
          return func->data.builtin.function(args, argc);
      } else if (func->type == VALUE_LAMBDA) {
          // Create new environment with parameters bound to args
          environment_t* call_env = environment_create(func->data.lambda.closure);

          for (size_t i = 0; i < argc; i++) {
              environment_bind(call_env,
                             func->data.lambda.params[i],
                             args[i]);
          }

          // Evaluate body in new environment
          eshkol_value_t* result = eshkol_eval(func->data.lambda.body, call_env);
          environment_destroy(call_env);
          return result;
      }
  }
  ```

**Session 77-78: Eval for Lambda**
- [ ] Implement lambda creation in eval
  ```c
  case AST_LAMBDA: {
      eshkol_value_t* lambda = value_create_lambda();
      lambda->data.lambda.params = copy_params(expr->data.lambda.params);
      lambda->data.lambda.param_count = expr->data.lambda.param_count;
      lambda->data.lambda.body = expr->data.lambda.body;
      lambda->data.lambda.closure = env;  // Capture environment
      return lambda;
  }
  ```
- [ ] Test:
  ```scheme
  (define f (eval '(lambda (x) (* x 2))))
  (f 5)  ; => 10
  ```

**Session 79-80: Integration & Testing**
- [ ] Create comprehensive eval test suite
- [ ] Test eval with complex expressions
- [ ] Test interaction between eval and compiled code
- [ ] Benchmark eval performance
- [ ] Document eval system in user guide

---

### Month 5: Macros (Sessions 81-100)

#### Week 9-10: Macro System (Sessions 81-100)

**Session 81-82: Macro Design**
- [ ] **File**: `docs/MACRO_SYSTEM_DESIGN.md` (new)
- [ ] Design syntax-rules pattern language:
  - Patterns: literals, pattern variables, ellipsis (...)
  - Templates: substitution, ellipsis expansion
  - Hygienic renaming strategy
- [ ] Document API:
  ```scheme
  (define-syntax name
    (syntax-rules (literals ...)
      ((pattern1) template1)
      ((pattern2) template2)))
  ```

**Session 83-84: Pattern Parser**
- [ ] **File**: `lib/frontend/parser/parser_macros.c` (new)
- [ ] Parse `define-syntax` forms
- [ ] Parse `syntax-rules` patterns
- [ ] Create macro AST node types:
  ```c
  typedef struct macro_pattern {
      enum { PAT_LITERAL, PAT_VARIABLE, PAT_ELLIPSIS, PAT_LIST } type;
      union {
          char* literal;
          char* variable;
          struct macro_pattern** subpatterns;
      } data;
  } macro_pattern_t;
  ```

**Session 85-86: Pattern Matching**
- [ ] **File**: `lib/frontend/macro_expander.c` (new)
- [ ] Implement pattern matching:
  ```c
  typedef struct match_result {
      bool matches;
      hash_table_t* bindings;  // variable -> value
  } match_result_t;

  match_result_t* match_pattern(macro_pattern_t* pattern,
                                eshkol_ast_t* expr,
                                char** literals,
                                size_t literal_count);
  ```
- [ ] Handle literal matching
- [ ] Handle pattern variables
- [ ] Handle ellipsis (...) matching

**Session 87-88: Template Expansion**
- [ ] Implement template substitution:
  ```c
  eshkol_ast_t* expand_template(macro_template_t* template,
                               hash_table_t* bindings);
  ```
- [ ] Handle simple substitution
- [ ] Handle ellipsis expansion
- [ ] Handle nested patterns

**Session 89-90: Hygienic Renaming**
- [ ] Implement hygiene system:
  ```c
  typedef struct rename_context {
      hash_table_t* renames;  // original -> renamed
      size_t gensym_counter;
  } rename_context_t;

  char* generate_unique_name(rename_context_t* ctx, const char* base);
  void rename_variables(eshkol_ast_t* expr, rename_context_t* ctx);
  ```
- [ ] Prevent variable capture
- [ ] Preserve intentional captures

**Session 91-92: Macro Expansion Pass**
- [ ] Integrate macros into compilation pipeline
- [ ] Add macro expansion after parsing, before type checking
- [ ] Expand macros recursively
- [ ] Handle macro-generating macros

**Session 93-94: Standard Macros**
- [ ] **File**: `stdlib/macros.esk` (new)
- [ ] Implement `let`:
  ```scheme
  (define-syntax let
    (syntax-rules ()
      ((let ((var val) ...) body ...)
       ((lambda (var ...) body ...) val ...))))
  ```
- [ ] Implement `let*`, `letrec`
- [ ] Implement `and`, `or`
- [ ] Implement `cond`, `case`

**Session 95-96: Advanced Macros**
- [ ] Implement `when`, `unless`
- [ ] Implement `do` loops
- [ ] Implement `define-record-type`
- [ ] Test all standard macros

**Session 97-98: Macro Testing**
- [ ] Create `tests/macros_test.esk`
- [ ] Test hygiene: variable capture prevention
- [ ] Test ellipsis patterns
- [ ] Test macro composition
- [ ] Document macro system in user guide

**Session 99-100: v1.1 Release**
- [ ] Tag v1.1-metaprogramming
- [ ] Create release notes
- [ ] Update documentation
- [ ] Announce release

---

### Month 6: I/O & Modules (Sessions 101-120)

#### Week 11-12: File I/O & Serialization (Sessions 101-110)

**Session 101-102: File I/O Design**
- [ ] **File**: `docs/IO_SYSTEM_DESIGN.md` (new)
- [ ] Design file I/O API:
  ```scheme
  (open-input-file path)         ; => input-port
  (open-output-file path)        ; => output-port
  (close-port port)
  (read port)                    ; read one expression
  (write obj port)               ; write one expression
  (read-file path)               ; read entire file as string
  (write-file path content)      ; write string to file
  ```

**Session 103-104: Implement File Primitives**
- [ ] **Files**: `lib/runtime/file_io.h` (new), `lib/runtime/file_io.c` (new)
  ```c
  typedef struct file_port {
      FILE* handle;
      char* path;
      bool is_input;
      bool is_open;
      size_t line_number;
  } file_port_t;

  file_port_t* eshkol_open_input_file(const char* path);
  file_port_t* eshkol_open_output_file(const char* path);
  void eshkol_close_port(file_port_t* port);
  char* eshkol_read_file(const char* path);
  bool eshkol_write_file(const char* path, const char* content);
  ```

**Session 105-106: JSON Serialization**
- [ ] **Files**: `lib/runtime/json.h` (new), `lib/runtime/json.c` (new)
- [ ] Implement JSON encoder:
  ```c
  char* eshkol_to_json(eshkol_value_t* value);
  eshkol_value_t* eshkol_from_json(const char* json_string);
  ```
- [ ] Map Eshkol types to JSON:
  - Integer → number
  - Float → number
  - String → string
  - List → array
  - Pairs → objects with "car" and "cdr" keys

**Session 107-108: Binary Tensor Format**
- [ ] **File**: `lib/runtime/tensor_io.c` (new)
- [ ] Design binary format:
  ```
  [Header: 16 bytes]
    - Magic number: 0xE5C0DA7A (4 bytes)
    - Version: 1 (4 bytes)
    - Rank: (4 bytes)
    - Element type: (4 bytes)
  [Shape: rank * 4 bytes]
  [Data: num_elements * element_size bytes]
  ```
- [ ] Implement save/load for tensors

**Session 109-110: I/O Testing**
- [ ] Create `tests/file_io_test.esk`
- [ ] Test reading and writing files
- [ ] Test JSON round-trip
- [ ] Test binary tensor I/O

#### Week 13-14: Module System (Sessions 111-120)

**Session 111-112: Module Design**
- [ ] **File**: `docs/MODULE_SYSTEM_DESIGN.md` (new)
- [ ] Design module syntax:
  ```scheme
  (module name
    (export symbol1 symbol2 ...)
    (import (module1) (module2 prefix:))

    ; Module body
    (define ...)
    ...)
  ```

**Session 113-114: Module Parser**
- [ ] **File**: `lib/frontend/parser/parser_modules.c` (new)
- [ ] Parse module declarations
- [ ] Parse export lists
- [ ] Parse import statements

**Session 115-116: Module System**
- [ ] **File**: `lib/frontend/module_system.c` (new)
  ```c
  typedef struct module {
      char* name;
      hash_table_t* exports;      // public bindings
      hash_table_t* private;      // private bindings
      module_t** dependencies;
  } module_t;

  module_t* module_create(const char* name);
  void module_export(module_t* mod, const char* name,
                    eshkol_value_t* value);
  eshkol_value_t* module_import(module_t* target,
                               module_t* source,
                               const char* name);
  ```

**Session 117-118: REPL Implementation**
- [ ] **File**: `exe/eshkol-repl.cpp` (new)
- [ ] Implement read-eval-print loop
- [ ] Add line editing (readline library)
- [ ] Add history
- [ ] Test interactive development

**Session 119-120: v1.2 Release**
- [ ] Complete integration testing
- [ ] Tag v1.2-infrastructure
- [ ] Create release notes
- [ ] Update documentation
- [ ] Release binaries

---

## Phase 2: Native Scientific Computing (Months 7-12)

**Goal**: High-performance tensor operations with autodiff and GPU support

### Month 7-8: Tensor Foundation (Sessions 121-160)

**Key Deliverables**:
- Native tensor type in LLVM IR
- Multi-dimensional array operations
- SIMD vectorization (60% → 100%)
- Memory-efficient tensor storage
- Broadcasting support
- Lazy evaluation for tensors

**Session Breakdown** (detailed breakdown available on request):
- Sessions 121-130: Tensor type design & LLVM IR
- Sessions 131-140: Basic tensor operations (add, mul, matmul)
- Sessions 141-150: SIMD optimization
- Sessions 151-160: Broadcasting & lazy evaluation

---

### Month 9-10: Complete Autodiff (Sessions 161-200)

**Key Deliverables**:
- Forward mode autodiff (complete)
- Reverse mode autodiff (complete)
- Jacobian and Hessian computation
- Gradient computation for compositions
- Efficient tape-based reverse mode
- Memory-efficient checkpointing

**Session Breakdown**:
- Sessions 161-170: Forward mode completion
- Sessions 171-180: Reverse mode implementation
- Sessions 181-190: Jacobian/Hessian
- Sessions 191-200: Optimization & testing

---

### Month 11-12: GPU Acceleration (Sessions 201-240)

**Key Deliverables**:
- LLVM PTX backend for NVIDIA GPUs
- CUDA kernel generation from LLVM IR
- GPU memory management
- Data transfer optimization
- Automatic CPU/GPU dispatch
- OpenCL support for broader compatibility
- **v1.4-gpu release**

**Session Breakdown**:
- Sessions 201-210: LLVM PTX backend setup
- Sessions 211-220: CUDA kernel generation
- Sessions 221-230: GPU memory management
- Sessions 231-240: OpenCL & optimization

---

## Phase 3: Symbolic & Neural DSL (Months 13-18)

**Goal**: Neuro-symbolic integration with native neural networks

### Month 13-14: Symbolic Reasoning (Sessions 241-280)

**Key Deliverables**:
- Unification engine (miniKanren-style)
- Logic variables and constraints
- Backtracking search
- Pattern matching integration
- Rule-based systems
- Forward/backward chaining
- **v1.5-symbolic release**

---

### Month 15-16: Neural Network DSL (Sessions 281-320)

**Key Deliverables**:
- Layer abstractions:
  - Dense (fully connected)
  - Conv2D (convolutional)
  - MaxPool, AvgPool
  - Activation functions (ReLU, sigmoid, tanh, softmax)
- Training loop infrastructure:
  - Optimizers (SGD, Adam, RMSprop)
  - Loss functions (MSE, cross-entropy)
  - Batch processing
  - Learning rate scheduling
- Model save/load (binary format)
- **Example**: MNIST classifier in pure Eshkol

---

### Month 17-18: Neuro-Symbolic Integration (Sessions 321-360)

**Key Deliverables**:
- Differentiable logic primitives
- Neural-symbolic hybrid architectures
- Explanation generation from models
- Integration with pattern matching
- Complete neural network library
- **v2.0-neurosymbolic release**

---

## Phase 4: Formal Verification (Months 19-24)

**Goal**: HoTT foundations, formal verification, enterprise readiness

### Month 19-20: HoTT & Lean Integration (Sessions 361-400)

**Key Deliverables**:
- Dependent type support (basic)
- HoTT (Homotopy Type Theory) foundations
- Type universe hierarchy
- Proof irrelevance
- Integration with Lean (proof transport)
- Shared IR with verification information
- Proof-carrying code infrastructure
- **v2.1-verified release**

---

### Month 21-22: Eshkol Prover (Sessions 401-440)

**Key Deliverables**:
- eprover (Eshkol Prover) tool
- Automated theorem proving for Eshkol programs
- Property-based testing framework
- Program synthesis from specifications
- Verified compilation
- Correctness proofs for optimizations
- Verified neural network primitives
- **v2.2-prover release**

---

### Month 23-24: Production Hardening (Sessions 441-480)

**Key Deliverables**:
- Performance optimization (JIT, advanced SIMD)
- Security audit
- Comprehensive error handling
- Enterprise logging and monitoring
- Complete documentation:
  - API reference
  - Tutorials
  - Case studies
- Academic paper draft
- **v2.5-production release**

---

## Team & Resources

### Core Team (5.5 FTE)

1. **Language Engineers (2 FTE)**
   - Core implementation
   - LLVM backend development
   - Type system implementation

2. **ML/Tensor Engineer (1 FTE)**
   - Native tensor operations
   - GPU acceleration
   - Autodiff system

3. **Type Theory Researcher (1 FTE)**
   - HoTT foundations
   - Formal verification
   - Lean integration

4. **Systems Engineer (1 FTE)**
   - Performance optimization
   - Concurrency
   - Production hardening

5. **DevOps Engineer (0.5 FTE)**
   - CI/CD
   - Packaging
   - Infrastructure

6. **Technical Writer (0.5 FTE)**
   - Documentation
   - Tutorials
   - User guides

### Infrastructure Requirements

- **CI/CD**: GitHub Actions
- **Testing**: GPU test machines (NVIDIA, AMD)
- **Verification**: Lean installation
- **Packaging**: Homebrew tap, apt repository
- **Documentation**: ReadTheDocs or similar

---

## Success Metrics

### Technical Milestones

| Milestone | Date | Key Metric |
|-----------|------|------------|
| v1.0-foundation | Month 2 | Builds on Linux/macOS, 90% test coverage |
| v1.1-metaprogramming | Month 4 | Self-modifying code works |
| v1.2-infrastructure | Month 6 | Can build modular programs |
| v1.3-tensor | Month 10 | Train simple model natively |
| v1.4-gpu | Month 12 | 10x speedup on GPU vs CPU |
| v1.5-symbolic | Month 14 | Solve constraint problems |
| v2.0-neurosymbolic | Month 18 | MNIST in pure Eshkol |
| v2.1-verified | Month 20 | Prove simple theorems |
| v2.2-prover | Month 22 | Verify neural net properties |
| v2.5-production | Month 24 | Production deployment ready |

### Community Metrics

- **Month 6**: 100+ GitHub stars
- **Month 12**: 500+ GitHub stars, 5+ external contributors
- **Month 18**: 1000+ GitHub stars, first production deployment
- **Month 24**: 2000+ GitHub stars, 10+ production deployments, 1+ academic paper

### Performance Metrics

- **Autodiff overhead**: < 3x vs hand-written derivatives
- **GPU speedup**: 10-100x vs CPU for neural networks
- **Compilation time**: < 10s for 10K LOC program
- **Memory efficiency**: < 2x overhead vs C

---

## Risk Mitigation

### Technical Risks

1. **GPU code generation complexity**
   - Mitigation: Start with LLVM PTX (proven), use LLVM's infrastructure
   - Fallback: CPU-only for v1.0-v1.3, GPU in v1.4+

2. **Formal verification overhead**
   - Mitigation: Make verification optional, gradual adoption
   - Focus: Verify critical components first (neural networks)

3. **Performance vs verification tradeoff**
   - Mitigation: Optimize verified code paths separately
   - Strategy: Provide both verified and unverified versions

4. **Lean integration complexity**
   - Mitigation: Start with proof transport, not full integration
   - Delay: Full Lean integration can be post-v2.5 if needed

### Timeline Risks

1. **Scope creep**
   - Mitigation: Strict phase discipline
   - Defer: Non-critical features to post-v2.5
   - Review: Monthly progress reviews

2. **Team size/availability**
   - Mitigation: Can scale up with funding if needed
   - Flexibility: Some roles can be contractors

3. **Technical unknowns**
   - Mitigation: Budget 20% time for research/prototyping
   - Strategy: Fail fast, pivot if needed

### Market/Adoption Risks

1. **Competing with established tools**
   - Differentiation: Only formally verified ML framework
   - Niche: Target high-assurance systems first

2. **Learning curve**
   - Mitigation: Excellent documentation
   - Strategy: Tutorials, examples, community support

3. **Ecosystem maturity**
   - Mitigation: Native Eshkol libraries, not Python bindings
   - Timeline: Gradual growth over 2 years

---

## How to Use This Plan

### Daily Development

1. **Pick session** from current month
2. **Review objectives** and files
3. **Implement** changes
4. **Test** thoroughly
5. **Commit** with session number in message
6. **Update** progress in project tracker

### Weekly Reviews

- Complete 5-6 sessions per week
- Review week's progress
- Adjust timeline if needed
- Update stakeholders

### Monthly Milestones

- Review milestone achievements
- Plan next month's priorities
- Adjust resource allocation
- Publish progress report

### Session Commit Format

```
Session XXX: Brief description

Detailed description of changes

Implements: #issue-number
Part of: Phase N, Month M
Session: XXX/480

Test: description of tests
```

---

## Next Steps

### Immediate Actions (Week 1)

1. **Setup**:
   - Clone both repos
   - Review current state
   - Install dependencies (LLVM, build tools)

2. **Planning**:
   - Create GitHub project board
   - Import this plan as issues
   - Assign team members to phases

3. **Infrastructure**:
   - Set up CI/CD pipeline
   - Configure test infrastructure
   - Set up documentation system

4. **Begin Development**:
   - Start Session 1: Commit mixed-type work
   - Begin Session 2: Build testing

### First Month Goals

- Complete Sessions 1-20
- Release v1.0-foundation
- Have CI/CD running
- Begin eval implementation

---

## Appendices

### A. File Structure

```
eshkol/
├── build/                  # Build artifacts
├── docs/                   # Documentation
│   ├── MASTER_DEVELOPMENT_PLAN.md
│   ├── BUILD_STATUS.md
│   ├── HIGHER_ORDER_REWRITE_PLAN.md
│   ├── AUTODIFF_TYPE_BUGS.md
│   ├── EVAL_DESIGN.md
│   ├── MACRO_SYSTEM_DESIGN.md
│   ├── MODULE_SYSTEM_DESIGN.md
│   └── ...
├── examples/               # Example programs
├── inc/eshkol/            # Public headers
├── lib/                    # Implementation
│   ├── backend/           # LLVM codegen
│   ├── core/              # Core runtime
│   ├── frontend/          # Parser, type checker
│   └── runtime/           # Runtime support
├── stdlib/                 # Standard library
├── tests/                  # Test suite
└── CMakeLists.txt
```

### B. Reference Documents

From original tscheme repository:
- `ESHKOL_DEVELOPMENT_BIBLE.md` - Overall vision
- `ESHKOL_HASKELL_LEAN_ROADMAP.md` - Tri-language integration
- `ESHKOL_LLVM_BACKEND_AND_BEYOND.md` - LLVM implementation guide
- `ESHKOL_COMPLETE_LANGUAGE_SPECIFICATION.md` - Language spec (2421 lines)

### C. Key Technologies

- **LLVM**: 14+ for code generation
- **CMake**: 3.14+ for build system
- **Git**: Version control
- **GitHub Actions**: CI/CD
- **Docker**: Packaging
- **Lean**: Formal verification (Phase 4)

---

**Document Status**: Living document, updated as development progresses
**Last Updated**: November 2025
**Next Review**: End of Month 1 (after Session 20)
