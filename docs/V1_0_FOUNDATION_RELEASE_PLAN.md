# Eshkol v1.0-Foundation Release Plan

**Target Release**: v1.0-foundation  
**Timeline**: 60 sessions (~3 months)  
**Status**: Planning Complete, Implementation Starting  
**Created**: 2025-11-13

---

## Executive Summary

This plan details the roadmap to achieve **v1.0-foundation** stable release for Eshkol, focusing on:
1. **Stabilizing mixed-type list operations** (completing tagged value migration)
2. **Fixing critical autodiff bugs** (SCH-006, SCH-007, SCH-008)
3. **Establishing CI/CD infrastructure** (GitHub Actions, packaging)
4. **Comprehensive documentation** (examples, API reference)

**Timeline**: Sessions 1-60 of the 480-session master plan  
**Scope**: Foundation layer WITHOUT metaprogramming (eval, macros, modules deferred to v1.1+)

---

## Current State Analysis

### What's Working ✅
- **LLVM Backend**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - 7050 lines, functional
- **Arena Memory**: [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp) - Complete with tagged cons cells
- **Tagged Value System**: Full implementation in [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h)
  - 24-byte tagged cons cells with type preservation
  - Type-safe helpers: `arena_tagged_cons_get_int64/double/ptr`
  - Exactness tracking (Scheme compliance)
- **Basic List Operations**: `car`, `cdr`, `cons`, `list`, compound operations (caar-cddddr)
- **Build System**: CMake with LLVM 14+ support
- **Test Suite**: 40+ tests covering core functionality

### Critical Gaps ❌
**Higher-Order Functions**: Currently use legacy `CreateStructGEP` direct struct access instead of tagged value helpers. Identified locations:

| Function | File | Lines | CreateStructGEP Count | Status |
|----------|------|-------|----------------------|--------|
| codegenMapSingleList | llvm_codegen.cpp | 5182-5273 | 3 calls | 🔄 Needs migration |
| codegenMapMultiList | llvm_codegen.cpp | 5276-5393 | 3 calls | 🔄 Needs migration |
| codegenFilter | llvm_codegen.cpp | 5396-5510 | 3 calls | 🔄 Needs migration |
| codegenFold | llvm_codegen.cpp | 5513-5585 | 2 calls | 🔄 Needs migration |
| codegenForEachSingleList | llvm_codegen.cpp | 5759-5804 | 2 calls | 🔄 Needs migration |
| codegenMember | llvm_codegen.cpp | ~5653-5723 | 2 calls | 🔄 Needs migration |
| codegenAssoc | llvm_codegen.cpp | ~5813-5923 | 3 calls | 🔄 Needs migration |
| codegenTake | llvm_codegen.cpp | ~5979-6079 | 3 calls | 🔄 Needs migration |
| codegenFind | llvm_codegen.cpp | ~6138-6240 | 2 calls | 🔄 Needs migration |
| codegenPartition | llvm_codegen.cpp | ~6242-6389 | 6 calls | 🔄 Needs migration |
| codegenSplitAt | llvm_codegen.cpp | ~6392-6496 | 3 calls | 🔄 Needs migration |
| codegenRemove | llvm_codegen.cpp | ~6499-6604 | 3 calls | 🔄 Needs migration |
| codegenLast | llvm_codegen.cpp | ~6607-6684 | 2 calls | 🔄 Needs migration |
| codegenLastPair | llvm_codegen.cpp | ~6687-6751 | 1 call | 🔄 Needs migration |
| codegenFoldRight | llvm_codegen.cpp | 5807-5810 | 0 (stub) | ⏳ Not implemented |

**Known Issues**:
- SCH-006: Type inference for autodiff incomplete
- SCH-007: Vector return types not handled
- SCH-008: Type conflicts in generated code
- No CI/CD infrastructure
- ~100 examples need syntax updates

---

## Month 1: Stabilization (Sessions 1-20)

### Week 1: Mixed-Type Completion (Sessions 1-10)

#### **Session 1-2: Commit & Build Verification**
**Files**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp), [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp), [`tests/phase_2a_group_a_test.esk`](tests/phase_2a_group_a_test.esk)

**Tasks**:
1. Check git status for unstaged changes
2. Commit mixed-type tagged value system with proper message
3. Add [`tests/phase_2a_group_a_test.esk`](tests/phase_2a_group_a_test.esk) to git
4. Clean rebuild: `cmake --build build --clean-first`
5. Test: `./build/eshkol-run tests/phase_2a_group_a_test.esk`
6. Create [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md) tracking current state

**Expected Output**: All Phase 2A Group A tests pass (length, list-ref, list-tail, drop, last, last-pair)

**Commit Format**:
```
Session 001-002: Commit mixed-type system and verify build

- Committed tagged value system implementation
- Added Phase 2A Group A test to repository
- Verified clean build on current system
- All Group A traversal tests passing

Part of: Phase 1, Month 1
Session: 001-002/480
Test: phase_2a_group_a_test.esk passes with expected output
```

#### **Session 3-4: Analysis & Documentation**
**Files**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) (lines 5040-6751)

**Tasks**:
1. Read all 15 higher-order function implementations
2. For each function, document:
   - Current `CreateStructGEP` usage (line numbers)
   - Required migration to tagged helpers
   - Test case needed
3. Create [`docs/HIGHER_ORDER_REWRITE_PLAN.md`](docs/HIGHER_ORDER_REWRITE_PLAN.md) with:
   - Complete function list
   - Migration priority: map → filter → fold → for-each → utilities
   - Per-function change specification
   - Test coverage plan

**Deliverable**: Complete migration roadmap document

#### **Session 5-6: Migrate map (single-list)**
**File**: [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)  
**Function**: `codegenMapSingleList` (lines 5182-5273)

**Changes Required**:
```cpp
// Line 5222: BEFORE (direct struct access)
Value* input_car_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 0);
Value* input_element = builder->CreateLoad(Type::getInt64Ty(*context), input_car_ptr);

// AFTER (tagged value extraction)
Value* input_element_tagged = extractCarAsTaggedValue(current_val);
Value* input_element = unpackInt64FromTaggedValue(input_element_tagged);

// Line 5258: BEFORE (direct cdr access)
Value* input_cdr_ptr = builder->CreateStructGEP(arena_cons_type, input_cons_ptr, 1);
Value* input_cdr = builder->CreateLoad(Type::getInt64Ty(*context), input_cdr_ptr);

// AFTER (use tagged helper)
Value* is_cdr = ConstantInt::get(Type::getInt1Ty(*context), 1);
Value* input_cdr = builder->CreateCall(arena_tagged_cons_get_ptr_func, {input_cons_ptr, is_cdr});
```

**Test**: `(map (lambda (x) (* x 2)) (list 1 2.5 3))` → Expected: `(2 5.0 6)`

**LLVM Best Practices**:
- Use `getelementptr` inbounds for bounds checking
- Proper alignment specifications
- PHI nodes for control flow merges
- Type consistency across basic blocks

#### **Session 7-8: Migrate map (multi-list)**
**Function**: `codegenMapMultiList` (lines 5276-5393)

**Changes**: Similar pattern to single-list, but for synchronized multi-list traversal:
- Line 5332: Extract car from each list using tagged helpers
- Line 5377-5378: Extract cdr from each list using `arena_tagged_cons_get_ptr`

**Test**: `(map + (list 1 2 3) (list 4.5 5.5 6.5))` → Expected: `(5.5 7.5 9.5)`

#### **Session 9-10: Migrate filter**
**Function**: `codegenFilter` (lines 5396-5510)

**Changes**:
- Line 5457: Use `extractCarAsTaggedValue` instead of direct access
- Line 5495: Use tagged helper for cdr access

**Test**: `(filter (lambda (x) (> x 5)) (list 1 8.5 3 9))` → Expected: `(8.5 9)`

### Week 2: Higher-Order Functions (Sessions 11-20)

#### **Session 11-12: Migrate fold**
**Function**: `codegenFold` (lines 5513-5585)

**Changes**:
- Line 5565: Use tagged extraction for car
- Line 5576: Use tagged helper for cdr

**Test**: `(fold + 0 (list 1 2.5 3))` → Expected: `6.5`

#### **Session 13-14: Implement fold-right**
**Function**: `codegenFoldRight` (line 5807 - currently stub)

**Implementation Strategy**:
```cpp
Value* codegenFoldRight(const eshkol_operations_t* op) {
    // Right-to-left folding: (fold-right proc init list)
    // Build accumulator by recursing/reversing list first
    // Then fold from right: (proc elem acc)
    
    // Option 1: Recursive approach (cleaner)
    // Option 2: Reverse list then left fold (easier with current tools)
}
```

**Test**: `(fold-right cons '() (list 1 2 3))` → Expected: `(1 2 3)`

#### **Session 15-16: Migrate for-each**
**Function**: `codegenForEachSingleList` (lines 5759-5804)

**Changes**:
- Line 5788: Use tagged extraction
- Line 5795: Use tagged helper for cdr

**Test**: `(for-each display (list 1 2.5 3))` → Output: `12.53`

#### **Session 17-18: Update member/assoc family**
**Functions**: `codegenMember`, `codegenAssoc` (6 functions total)

**Changes**: All three variants (member/memq/memv, assoc/assq/assv) need:
- Tagged value extraction for comparisons
- Proper cdr navigation

**Tests**:
```scheme
(member 2.5 (list 1 2.5 3))               ; => (2.5 3)
(assoc 'b (list (list 'a 1) (list 'b 2.5)))  ; => (b 2.5)
```

#### **Session 19-20: Update utility functions**
**Functions**: `find`, `partition`, `split-at`, `remove`, `take`, `last`, `last-pair`

**Tasks**:
- Migrate all to use tagged value system
- Create [`tests/phase_2a_group_b_test.esk`](tests/phase_2a_group_b_test.esk)
- Test comprehensive coverage

---

## Month 2: Autodiff Fixes & Examples (Sessions 21-40)

### Week 3: Autodiff Type Bugs (Sessions 21-30)

#### **Session 21-22: Investigate SCH-006**
**Issue**: Type inference for autodiff incomplete

**Tasks**:
1. Locate autodiff type inference code in [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp)
2. Create minimal test cases reproducing the bug
3. Document findings in [`docs/AUTODIFF_TYPE_BUGS.md`](docs/AUTODIFF_TYPE_BUGS.md)

#### **Session 23-24: Fix SCH-007 (Vector Returns)**
**Issue**: Vector return types not handled

**Test**: `(gradient (lambda (v) (dot v v)) #(1 2 3))` → Expected: vector `#(2 4 6)`

**Changes**: Implement proper LLVM vector type returns in autodiff codegen

#### **Session 25-26: Fix SCH-008 (Type Conflicts)**
**Issue**: Type conflicts in generated LLVM IR

**Tasks**:
1. Identify conflicting type declarations
2. Implement type unification
3. Test complex autodiff programs

#### **Session 27-30: Comprehensive Autodiff Testing**
**Deliverable**: [`tests/autodiff_comprehensive.esk`](tests/autodiff_comprehensive.esk)

**Test Coverage**:
- Forward mode: `(d/dx (lambda (x) (* x x)) 5)`
- Reverse mode: `(gradient f v)`
- Composition: `(d/dx (compose f g) x)`
- Vector functions
- Performance benchmarks

### Week 4: Examples & Documentation (Sessions 31-40)

#### **Session 31-32: Example Categorization**
**Directory**: [`examples/`](examples/)

**Tasks**:
1. Review all ~100 example files
2. Identify 30 most important examples
3. Categorize:
   - Basic syntax (5 examples)
   - List operations (5 examples)
   - Higher-order functions (5 examples)
   - Numerical computing (5 examples)
   - Autodiff (5 examples)
   - Advanced features (5 examples)

#### **Session 33-36: Update Example Syntax**
**Files**: [`examples/*.esk`](examples/) (30 core examples)

**Migration Pattern**:
```scheme
# OLD SYNTAX (non-standard)
(define main (lambda () ...))

# NEW SYNTAX (current)
(main ...)
```

**Deliverable**: [`docs/SYNTAX_MIGRATION.md`](docs/SYNTAX_MIGRATION.md)

#### **Session 37-38: Create New Examples**
**New Files**:
- `examples/mixed_types_demo.esk` - Showcase tagged value system
- `examples/higher_order_demo.esk` - map, filter, fold examples
- `examples/autodiff_tutorial.esk` - Educational autodiff guide
- `examples/vector_operations.esk` - Vector calculus showcase

All examples must be:
- Well-commented
- Educational
- Working with current build
- Demonstrating best practices

#### **Session 39-40: Documentation Updates**
**Files**: [`README.md`](README.md), [`docs/aidocs/GETTING_STARTED.md`](docs/aidocs/GETTING_STARTED.md)

**Changes**:
- Remove claims about unimplemented features (debugger, profiler)
- Add "Quick Start" section
- Add "Examples" section with links
- Update feature matrix showing actual status
- Accurate performance claims

---

## Month 3: Infrastructure (Sessions 41-60)

### Week 5: CI/CD Setup (Sessions 41-50)

#### **Session 41-42: GitHub Actions CI (Ubuntu)**
**File**: `.github/workflows/ci.yml` (new)

**Configuration**:
```yaml
name: Eshkol CI
on: [push, pull_request]
jobs:
  build-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v3
      - name: Install LLVM
        run: |
          sudo apt-get update
          sudo apt-get install -y llvm-14-dev clang-14
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
          ./eshkol-run ../tests/mixed_type_lists_basic_test.esk
```

#### **Session 43-44: GitHub Actions CI (macOS)**
**Addition to**: `.github/workflows/ci.yml`

**Configuration**:
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

#### **Session 45-46: CMake Install Targets**
**File**: [`CMakeLists.txt`](CMakeLists.txt)

**Additions**:
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

**Test**: `sudo make install` and verify installation to `/usr/local`

#### **Session 47-48: CPack for Debian**
**File**: [`CMakeLists.txt`](CMakeLists.txt)

**Configuration**:
```cmake
include(CPack)
set(CPACK_PACKAGE_NAME "eshkol")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "High-performance Scheme with scientific computing and autodiff")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Eshkol Team")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "llvm-14")
set(CPACK_GENERATOR "DEB")
```

**Test**: `cpack` and `sudo dpkg -i eshkol_1.0.0_amd64.deb`

#### **Session 49-50: Docker Build System**
**File**: `docker/ubuntu/release/Dockerfile` (new)

**Configuration**:
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

### Week 6: Testing & Release Prep (Sessions 51-60)

#### **Session 51-52: Homebrew Formula**
**Repository**: `homebrew-eshkol` (new)  
**File**: `Formula/eshkol.rb`

```ruby
class Eshkol < Formula
  desc "High-performance Scheme with scientific computing"
  homepage "https://github.com/openSVM/eshkol"
  url "https://github.com/openSVM/eshkol/archive/v1.0.0.tar.gz"
  sha256 "TBD"

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

#### **Session 53-54: Integration Tests**
**Directory**: `tests/integration/` (new)

**Files**:
- `tests/integration/mixed_type_comprehensive.esk`
- `tests/integration/higher_order_comprehensive.esk`
- `tests/integration/complex_computation.esk`

#### **Session 55-56: Memory Testing**
**Tools**: Valgrind

**Script**:
```bash
for test in tests/*.esk; do
  valgrind --leak-check=full --error-exitcode=1 \
    build/eshkol-run "$test" || echo "LEAK in $test"
done
```

**Deliverable**: Memory leak-free codebase, document findings in [`docs/TESTING.md`](docs/TESTING.md)

#### **Session 57-58: Documentation Cleanup**
**Files**:
- [`docs/aidocs/GETTING_STARTED.md`](docs/aidocs/GETTING_STARTED.md) - Remove debugger references (lines 253-272)
- [`docs/aidocs/COMPILATION_GUIDE.md`](docs/aidocs/COMPILATION_GUIDE.md) - Remove profiler references
- Create `SECURITY.md` for vulnerability reporting

#### **Session 59-60: v1.0 Release Preparation**
**Tasks**:
1. Create `CHANGELOG.md`
2. Create `RELEASE_NOTES_v1.0.md`
3. Tag release: `git tag -a v1.0-foundation -m "v1.0 Foundation Release"`
4. Build release artifacts:
   - Debian package (`.deb`)
   - macOS Homebrew formula
   - Source tarball
5. Create GitHub release with binaries
6. Update website/docs

---

## Technical Implementation Details

### Tagged Value System Architecture

**Data Structures**:
```c
// 24-byte tagged cons cell (lib/core/arena_memory.h:80-86)
typedef struct arena_tagged_cons_cell {
    uint8_t car_type;              // Type tag for car (4 bits base + 4 bits flags)
    uint8_t cdr_type;              // Type tag for cdr
    uint16_t flags;                // Reserved (immutability, etc.)
    eshkol_tagged_data_t car_data; // 8-byte union (int64/double/ptr)
    eshkol_tagged_data_t cdr_data; // 8-byte union
} arena_tagged_cons_cell_t;
```

**Type Tags** (inc/eshkol/eshkol.h:41-48):
- `ESHKOL_VALUE_NULL = 0`
- `ESHKOL_VALUE_INT64 = 1` 
- `ESHKOL_VALUE_DOUBLE = 2`
- `ESHKOL_VALUE_CONS_PTR = 3`

**Flags**:
- `ESHKOL_VALUE_EXACT_FLAG = 0x10` (for Scheme exactness)
- `ESHKOL_VALUE_INEXACT_FLAG = 0x20`

### LLVM IR Helper Functions

**Already Implemented** (lib/backend/llvm_codegen.cpp:1168-1244):
```cpp
// Extract car as tagged value (handles type detection)
Value* extractCarAsTaggedValue(Value* cons_ptr_int);

// Extract cdr as tagged value  
Value* extractCdrAsTaggedValue(Value* cons_ptr_int);

// Pack/unpack specific types
Value* packInt64ToTaggedValue(Value* int64_val, bool is_exact);
Value* packDoubleToTaggedValue(Value* double_val);
Value* unpackInt64FromTaggedValue(Value* tagged_val);
Value* unpackDoubleFromTaggedValue(Value* tagged_val);
```

**C Helper Functions** (called from LLVM IR):
```cpp
// Declared in lib/backend/llvm_codegen.cpp:518-682
arena_tagged_cons_get_int64_func    // int64_t (cell*, bool is_cdr)
arena_tagged_cons_get_double_func   // double (cell*, bool is_cdr)
arena_tagged_cons_get_ptr_func      // uint64_t (cell*, bool is_cdr)
arena_tagged_cons_get_type_func     // uint8_t (cell*, bool is_cdr)
arena_tagged_cons_set_int64_func    // void (cell*, bool, int64, type)
arena_tagged_cons_set_double_func   // void (cell*, bool, double, type)
arena_tagged_cons_set_ptr_func      // void (cell*, bool, uint64, type)
arena_tagged_cons_set_null_func     // void (cell*, bool)
```

### Migration Pattern Template

For any function using direct struct access:

**BEFORE**:
```cpp
StructType* arena_cons_type = StructType::get(Type::getInt64Ty(*context), Type::getInt64Ty(*context));
Value* cons_ptr = builder->CreateIntToPtr(list_ptr, builder->getPtrTy());
Value* car_ptr = builder->CreateStructGEP(arena_cons_type, cons_ptr, 0);  // ❌ DEPRECATED
Value* element = builder->CreateLoad(Type::getInt64Ty(*context), car_ptr);
```

**AFTER**:
```cpp
// Use tagged extraction helper
Value* element_tagged = extractCarAsTaggedValue(list_ptr);
Value* element = unpackInt64FromTaggedValue(element_tagged);  // Or unpackDouble if needed

// For type-aware processing:
Value* cons_ptr = builder->CreateIntToPtr(list_ptr, builder->getPtrTy());
Value* is_car = ConstantInt::get(Type::getInt1Ty(*context), 0);
Value* car_type = builder->CreateCall(arena_tagged_cons_get_type_func, {cons_ptr, is_car});

// Branch based on type
Value* base_type = builder->CreateAnd(car_type, ConstantInt::get(Type::getInt8Ty(*context), 0x0F));
Value* is_double = builder->CreateICmpEQ(base_type, ConstantInt::get(Type::getInt8Ty(*context), ESHKOL_VALUE_DOUBLE));
```

---

## Success Metrics

### Technical Milestones

| Milestone | Criteria | Status |
|-----------|----------|--------|
| Phase 2A Complete | All list operations use tagged values | ⏳ In Progress |
| Autodiff Fixed | SCH-006/007/008 resolved | ⏳ Planned |
| Examples Updated | 30 examples working, documented | ⏳ Planned |
| CI/CD Running | Linux + macOS builds green | ⏳ Planned |
| Packaging Complete | .deb + Homebrew available | ⏳ Planned |
| Memory Clean | Valgrind reports no leaks | ⏳ Planned |

### Quality Gates

**Must Pass Before Release**:
1. ✅ [`tests/phase_2a_group_a_test.esk`](tests/phase_2a_group_a_test.esk) - Group A traversal
2. ⏳ `tests/phase_2a_group_b_test.esk` - Group B utilities (to be created)
3. ⏳ `tests/autodiff_comprehensive.esk` - Full autodiff validation
4. ⏳ `tests/integration/` - All integration tests pass
5. ⏳ Valgrind clean on entire test suite
6. ⏳ CI builds pass on Ubuntu 22.04 and macOS

### Performance Targets
- Autodiff overhead: < 3x vs hand-written derivatives
- Compilation time: < 10s for 10K LOC
- Memory efficiency: < 2x overhead vs C

---

## Risk Mitigation

### Technical Risks

**Risk 1: Higher-Order Function Migration Breaks Existing Tests**
- **Mitigation**: Test after each function migration
- **Rollback**: Git tags at each session
- **Validation**: Comprehensive test suite

**Risk 2: LLVM Version Compatibility**
- **Mitigation**: Document LLVM 14+ requirement clearly
- **Testing**: CI tests on multiple LLVM versions
- **Fallback**: Provide Docker images with correct LLVM

**Risk 3: Memory Management Issues**
- **Mitigation**: Valgrind testing after each change
- **Tools**: Arena statistics tracking
- **Validation**: Stress tests with large lists

### Timeline Risks

**Risk 1: Scope Creep**
- **Mitigation**: Strict adherence to session plan
- **Defer**: Non-critical features to v1.1+
- **Review**: After each 10 sessions

**Risk 2: Autodiff Complexity**
- **Mitigation**: Allocate extra time (10 sessions)
- **Strategy**: Fix critical bugs first, defer optimizations
- **Fallback**: Document known limitations clearly

---

## Session Workflow

### Per-Session Process
1. **Review Session Objectives** from this plan
2. **Switch to Code Mode** for implementation
3. **Make Changes** following LLVM best practices
4. **Test Immediately** with session-specific test
5. **Commit with Session Format**:
   ```
   Session XXX: Brief description
   
   Detailed changes
   
   Part of: Phase 1, Month M
   Session: XXX/480
   Test: test file and expected results
   ```
6. **Update Status** in [`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md)

### Weekly Review (Every 5 Sessions)
- Progress assessment
- Timeline adjustments
- Stakeholder update
- Risk review

### Monthly Milestones
- Month 1 Complete: All higher-order functions migrated
- Month 2 Complete: Autodiff bugs fixed, examples updated
- Month 3 Complete: CI/CD running, v1.0 released

---

## Reference Information

### Key Files to Modify
- [`lib/backend/llvm_codegen.cpp`](lib/backend/llvm_codegen.cpp) - Primary development (7050 lines)
- [`CMakeLists.txt`](CMakeLists.txt) - Build and packaging
- [`README.md`](README.md) - User-facing documentation
- [`examples/*.esk`](examples/) - Example programs

### Key Files for Reference
- [`inc/eshkol/eshkol.h`](inc/eshkol/eshkol.h) - Core type definitions
- [`lib/core/arena_memory.h`](lib/core/arena_memory.h) - Arena API
- [`lib/core/arena_memory.cpp`](lib/core/arena_memory.cpp) - Arena implementation
- [`docs/MASTER_DEVELOPMENT_PLAN.md`](docs/MASTER_DEVELOPMENT_PLAN.md) - Overall roadmap

### LLVM Resources
- **Language Reference**: https://llvm.org/docs/LangRef.html
- **Programmer's Manual**: https://llvm.org/docs/ProgrammersManual.html
- **IR Builder**: https://llvm.org/doxygen/classllvm_1_1IRBuilder.html
- **GetElementPtr**: https://llvm.org/docs/GetElementPtr.html

### Testing Resources
- Test directory: [`tests/`](tests/)
- Example directory: [`examples/`](examples/)
- Phase 2A tests: Validation of tagged value system
- Integration tests: End-to-end validation

---

## Next Actions

### Immediate (Today)
1. ✅ Review this plan for completeness
2. ⏳ Begin Session 1-2: Commit and build verification
3. ⏳ Run [`tests/phase_2a_group_a_test.esk`](tests/phase_2a_group_a_test.esk) to establish baseline

### This Week
- Complete Sessions 1-10 (Week 1 of Month 1)
- Migrate map, filter functions
- Create [`docs/HIGHER_ORDER_REWRITE_PLAN.md`](docs/HIGHER_ORDER_REWRITE_PLAN.md)

### This Month
- Complete all 20 sessions of Month 1
- All higher-order functions using tagged values
- Baseline established for Month 2

---

## Document Status

- **Status**: Active Development Plan
- **Last Updated**: 2025-11-13
- **Next Review**: After Session 20 (end of Month 1)
- **Owner**: Eshkol Development Team
- **Version**: 1.0

---

## Appendix A: Function Migration Checklist

- [ ] codegenMapSingleList (Session 5-6)
- [ ] codegenMapMultiList (Session 7-8)
- [ ] codegenFilter (Session 9-10)
- [ ] codegenFold (Session 11-12)
- [ ] codegenFoldRight (Session 13-14) - NEW IMPLEMENTATION
- [ ] codegenForEachSingleList (Session 15-16)
- [ ] codegenMember/memq/memv (Session 17)
- [ ] codegenAssoc/assq/assv (Session 17-18)
- [ ] codegenTake (Session 19)
- [ ] codegenDrop (Session 19)
- [ ] codegenFind (Session 19)
- [ ] codegenPartition (Session 20)
- [ ] codegenSplitAt (Session 20)
- [ ] codegenRemove/remq/remv (Session 20)
- [ ] codegenLast (Session 20)
- [ ] codegenLastPair (Session 20)

## Appendix B: Test Coverage Plan

### Core Tests (Must Pass)
- ✅ [`tests/phase_2a_group_a_test.esk`](tests/phase_2a_group_a_test.esk) - Traversal functions
- ⏳ `tests/phase_2a_group_b_test.esk` - Higher-order functions (to create)
- ⏳ `tests/autodiff_comprehensive.esk` - Autodiff validation (to create)
- ⏳ `tests/integration/` - End-to-end scenarios (to create)

### Example Validation
- 30 core examples must run successfully
- Each example must have expected output documented
- Examples must demonstrate best practices

### Memory Validation
- No leaks in any test
- Arena statistics normal ranges
- No invalid memory access (AddressSanitizer clean)

---

**END OF PLAN**