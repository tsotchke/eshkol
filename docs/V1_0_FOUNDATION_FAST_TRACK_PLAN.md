# Eshkol v1.0-Foundation Fast Track Plan

**Date**: November 27, 2025  
**Status**: 🎉 **AHEAD OF SCHEDULE** - Autodiff Complete!  
**Remaining Sessions**: 30 (instead of 40)  
**Target Completion**: Q1 2026

---

## 🎉 Executive Summary

**EXCELLENT NEWS**: We're **10 sessions ahead of schedule!**

### ✅ Already Complete
- ✅ **v1.0-architecture** (Sessions 1-20): 67/67 list operation tests passing
- ✅ **Autodiff System** (originally Sessions 21-30): **43/43 tests passing - PRODUCTION READY**
  - All symbolic differentiation working
  - Forward-mode AD complete
  - Reverse-mode AD complete
  - Full vector calculus suite (jacobian, divergence, curl, laplacian, hessian)
  - All mathematical operations verified correct

### 📋 Remaining Work: 33 Tasks
**Focus**: Infrastructure, Examples, Documentation, and Release

---

## Current Test Status

### Core Language Tests
- **Total**: 67 tests
- **Pass Rate**: 100% (67/67 passing)
- **Categories**: List operations, mixed types, higher-order functions

### Autodiff Tests  
- **Total**: 43 tests
- **Pass Rate**: 100% (43/43 passing)
- **Coverage**: All autodiff operations mathematically verified

### Total Test Coverage
- **Combined**: 110 tests, 100% passing
- **Zero implementation bugs found**
- **Production-ready foundation**

---

## Remaining Work Breakdown

### Phase 1: Examples & Documentation (13 tasks)

#### Examples Work (8 tasks)
1. **Audit all examples** - Review ~100 existing example files
   - Categorize as: working/broken/deprecated
   - Identify best candidates for v1.0
   - Document current status of each

2. **Select 30 best examples** - Based on:
   - Educational value
   - Current functionality (works without changes)
   - Best practices demonstration
   - Coverage of core features

3. **Create directory structure**
   ```
   examples/
   ├── 01-basics/        (5 examples)
   ├── 02-list-ops/      (5 examples)
   ├── 03-higher-order/  (5 examples)
   ├── 04-numerical/     (5 examples)
   ├── 05-autodiff/      (5 examples)
   └── 06-advanced/      (5 examples)
   ```

4. **Update examples batch 1** (15 files)
   - Modernize syntax (remove old `main` wrapper pattern)
   - Add comprehensive comments
   - Include expected output
   - Test each example

5. **Update examples batch 2** (15 files)
   - Same process as batch 1
   - Special attention to autodiff examples
   - Verify all work correctly

6. **Create showcase example 1**: `showcase/mixed_types_demo.esk`
   - Publication-quality demonstration
   - Comprehensive documentation

7. **Create showcase example 2**: `showcase/higher_order_demo.esk`
   - Map, filter, fold comprehensive guide

8. **Create showcase examples 3-4**: Autodiff & vector calculus
   - `showcase/autodiff_tutorial.esk`
   - `showcase/vector_operations.esk`

#### Documentation Work (5 tasks)
9. **Update README.md**
   - ✅ Remove debugger claims (doesn't exist)
   - ✅ Remove profiler claims (doesn't exist)
   - ✅ Accurate feature matrix
   - ✅ Link to examples and quick start

10. **Update GETTING_STARTED.md**
    - Remove lines 253-272 (debugger references)
    - Remove profiler references
    - Update installation instructions
    - Add troubleshooting section

11. **Create QUICK_START.md** - 5-minute introduction
12. **Create EXAMPLES_GUIDE.md** - Guide with learning path
13. **Create TROUBLESHOOTING.md** - Common issues and solutions

---

### Phase 2: CI/CD Infrastructure (10 tasks)

#### GitHub Actions (2 tasks)
14. **Create .github/workflows/ci.yml** - Ubuntu CI
    - Run all 110 tests automatically
    - Fail build if any test fails
    - Upload test results as artifacts

15. **Add macOS job to ci.yml** - Cross-platform validation
    - Handle Homebrew LLVM installation
    - Same test coverage as Ubuntu

#### Packaging (4 tasks)
16. **Add CPack to CMakeLists.txt** - Debian package generation
    ```cmake
    set(CPACK_PACKAGE_NAME "eshkol")
    set(CPACK_PACKAGE_VERSION "1.0.0")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "llvm-14")
    set(CPACK_GENERATOR "DEB")
    ```

17. **Add install targets to CMakeLists.txt**
    - Install executable to /usr/local/bin
    - Install examples to /usr/local/share/eshkol
    - Install documentation

18. **Create cmake/cmake_uninstall.cmake.in** - Uninstall support

19. **Test installation**
    - `sudo cmake --install build`
    - Verify files in /usr/local
    - Test uninstall

#### Docker (3 tasks)
20. **Create docker/ubuntu/release/Dockerfile**
21. **Create scripts/docker_build.sh** - Build automation
22. **Create scripts/docker_test.sh** - Clean container testing

#### Homebrew (1 task)
23. **Create homebrew-eshkol repository**
    - Formula/eshkol.rb for macOS distribution
    - Test local installation

---

### Phase 3: Testing & Release (10 tasks)

#### Integration Testing (2 tasks)
24. **Create tests/integration/ directory** with 4 tests:
    - `mixed_type_comprehensive.esk` - All mixed-type operations
    - `higher_order_comprehensive.esk` - All higher-order functions
    - `autodiff_integration.esk` - Autodiff in complex scenarios
    - `complex_computation.esk` - Real-world computation

25. **Verify all integration tests pass** - 100% pass rate required

#### Validation Testing (3 tasks)
26. **Create scripts/memory_test.sh** - Valgrind testing
    - Zero leaks required
    - Test all 110+ tests with Valgrind

27. **Create scripts/performance_test.sh** - Benchmark autodiff overhead
    - Verify < 3x target
    - Document actual performance

28. **Create docs/TESTING.md** - Testing guide
    - Document all test categories
    - How to run tests
    - CI/CD information

#### Release Documentation (5 tasks)
29. **Create SECURITY.md** - Vulnerability reporting policy
30. **Create CODE_OF_CONDUCT.md** - Community guidelines  
31. **Create CHANGELOG.md** - Version history for v1.0-foundation
32. **Create RELEASE_NOTES_v1.0.md** - Comprehensive release documentation

33. **Release v1.0-foundation**
    - Git tag: `v1.0-foundation`
    - Build artifacts (.deb, source tarball, checksums)
    - Create GitHub release
    - Upload binaries
    - Publish announcement

---

## Execution Strategy

### Fast Track Approach (30 sessions)

**Week 1-2: Examples & Documentation** (13 tasks)
- Days 1-3: Audit and select examples (tasks 1-2)
- Days 4-6: Organize and update batch 1 (tasks 3-6)
- Days 7-9: Update batch 2 and create showcases (tasks 7-8)
- Days 10-12: Documentation updates (tasks 9-13)

**Week 3-4: Infrastructure** (10 tasks)
- Days 13-15: CI/CD setup (tasks 14-15)
- Days 16-18: Packaging (tasks 16-19)
- Days 19-21: Docker (tasks 20-22)
- Day 22: Homebrew (task 23)

**Week 5-6: Testing & Release** (10 tasks)
- Days 23-24: Integration tests (tasks 24-25)
- Days 25-27: Validation testing (tasks 26-28)
- Days 28-29: Release docs (tasks 29-32)
- Day 30: RELEASE (task 33)

### Parallelization Opportunities

Tasks that can be done in parallel:
- Examples work (tasks 1-8) + Documentation (tasks 9-13)
- Docker (tasks 20-22) + Homebrew (task 23)
- Final docs (tasks 29-32) can start while testing (tasks 24-28) is running

**Potential compression**: 30 sessions → 25 sessions with aggressive parallelization

---

## Critical Path Items

### Must Complete Before Release
1. ✅ Autodiff working (DONE - 43/43 tests passing)
2. ✅ List operations working (DONE - 67/67 tests passing)
3. ⏳ 30 working, documented examples
4. ⏳ CI/CD running on Ubuntu + macOS
5. ⏳ .deb package building and installing correctly
6. ⏳ All integration tests passing
7. ⏳ Memory clean (Valgrind verified)
8. ⏳ Performance targets met (< 3x autodiff overhead)

### Nice to Have (Can defer to v1.0.1)
- Docker build system
- Homebrew formula
- Full 30 examples (minimum 25 acceptable)

---

## Key Differences from Original Plan

### Sessions 21-30: SKIPPED ✅
**Original**: Autodiff bug fixes (SCH-006, SCH-007, SCH-008)  
**Reality**: All autodiff already working perfectly - no fixes needed!

**Impact**: 10 sessions saved, can be reallocated or compress timeline

### Updated Focus
Original Month 2 (Sessions 21-40):
- ❌ Sessions 21-30: Autodiff fixes (NOT NEEDED)
- ✅ Sessions 31-40: Examples & docs (STILL NEEDED)

New Month 2 (Sessions 21-30):
- ✅ Sessions 21-30: Examples & docs (advanced from 31-40)

New Month 3 (Sessions 31-50):
- ✅ Sessions 31-50: Infrastructure + testing + release (was 41-60)

**Timeline Compression**: 60 sessions → 50 sessions (or less with parallelization)

---

## Documentation Accuracy Issues Found

### README.md Issues
- ❌ Claims debugger exists (lines need updating)
- ❌ Claims profiler exists (lines need updating)
- ❌ Package manager installation instructions (not yet available)
- ⏳ Feature matrix needs accuracy update

### GETTING_STARTED.md Issues  
- ❌ Lines 253-272: Debugger section (REMOVE)
- ❌ Lines 285-299: Profiler section (REMOVE)
- ❌ File I/O examples (not implemented yet)
- ❌ Editor integration instructions (extensions don't exist)

### What Needs Removal
All references to:
- Debugger (`eshkol debug`, breakpoints, etc.)
- Profiler (`eshkol profile`, etc.)
- File I/O (`read-file`, `write-file`, etc.)
- String operations not implemented
- Package manager installs (until packages exist)

---

## Revised v1.0-Foundation Scope

### ✅ Core Features (Implemented & Tested)
- Mixed-type lists (int64 + double)
- 17 polymorphic higher-order functions
- Complete autodiff suite
- Arena-based memory management
- LLVM backend
- Type-safe operations

### 🎯 v1.0-Foundation Adds
- 30 curated, working examples
- Accurate documentation
- CI/CD automation
- Debian packaging
- Installation support
- Integration tests
- Memory validation
- Performance verification

### ❌ Explicitly NOT in v1.0-Foundation
- Debugger (planned for future)
- Profiler (planned for future)
- File I/O (planned for v1.2)
- String operations (partial only)
- REPL (planned for v1.2)
- Module system (planned for v1.2)
- Macros (planned for v1.1)

---

## Success Criteria (Updated)

### Technical Criteria
| Criterion | Target | Current Status |
|-----------|--------|----------------|
| Core tests | 100% pass | ✅ 67/67 passing |
| Autodiff tests | 100% pass | ✅ 43/43 passing |
| Integration tests | 4 passing | ⏳ To create |
| Memory | Zero leaks | ⏳ To verify |
| Performance | < 3x overhead | ⏳ To benchmark |
| Examples | 30 working | ⏳ To curate |
| CI/CD | Ubuntu + macOS | ⏳ To create |
| Packaging | .deb working | ⏳ To create |

### Documentation Criteria
| Criterion | Target | Current Status |
|-----------|--------|----------------|
| Accuracy | 100% | ⏳ Needs update |
| No false claims | Zero | ⏳ Debugger/profiler to remove |
| Examples tested | All 30 | ⏳ To verify |
| Quick start works | 5 min success | ⏳ To create |

---

## Risk Assessment (Updated)

### Low Risk ✅
- ✅ Autodiff working (was HIGH risk, now DONE)
- Examples curation (clear task, well-defined)
- CI/CD setup (well-documented, proven patterns)
- Documentation updates (straightforward edits)

### Medium Risk ⚠️
- Integration testing (may find unexpected issues)
- Memory testing (may find leaks requiring fixes)
- Performance testing (may not meet < 3x target)

### Mitigation Strategies
- **Integration issues**: Allocate buffer time (2-3 extra sessions)
- **Memory leaks**: Fix immediately if found (non-negotiable)
- **Performance**: Document actual overhead if > 3x, don't block release

---

## Timeline Estimate

### Conservative (6 weeks)
- Week 1-2: Examples & docs (13 tasks, 2 weeks @ 6-7 tasks/week)
- Week 3-4: Infrastructure (10 tasks, 2 weeks @ 5 tasks/week)
- Week 5-6: Testing & release (10 tasks, 2 weeks @ 5 tasks/week)
- **Total**: 6 weeks, 30 sessions

### Aggressive (4 weeks)
- Week 1: Examples audit + batch 1 (7 tasks with parallel work)
- Week 2: Examples batch 2 + docs (6 tasks with parallel work)
- Week 3: Infrastructure (10 tasks, aggressive pace)
- Week 4: Testing + release (10 tasks, compressed)
- **Total**: 4 weeks, 25-28 sessions

### Recommended (5 weeks)
- Week 1: Examples audit & selection (4 tasks)
- Week 2: Examples updates & docs (9 tasks, parallel work)
- Week 3: CI/CD & packaging (6 tasks)
- Week 4: Docker, Homebrew, testing (8 tasks)
- Week 5: Integration, validation, release (6 tasks)
- **Total**: 5 weeks, 30 sessions with buffer

---

## Next Immediate Actions

### This Week (Priority 1)
1. **Start example audit** - Review all ~100 examples
2. **Select 30 best** - Categorize and document
3. **Begin batch 1 updates** - First 15 examples
4. **Update README.md** - Remove false claims

### Next Week (Priority 2)
5. **Finish batch 2 updates** - Remaining 15 examples
6. **Create showcases** - 4 publication-quality examples
7. **Complete documentation** - QUICK_START, EXAMPLES_GUIDE, TROUBLESHOOTING
8. **Begin CI/CD** - Start GitHub Actions setup

---

## Detailed Task Breakdown

### Examples Work (Tasks 1-8)

**Task 1: Audit Examples**
- Review all files in `examples/` directory
- Test each one to see if it compiles/runs
- Categorize: ✅ Working / ⚠️ Needs fixes / ❌ Broken/deprecated
- Document in `docs/EXAMPLE_AUDIT.md`

**Task 2: Select 30 Best**
- Choose based on educational value and working status
- Ensure coverage: basics (5), lists (5), higher-order (5), numerical (5), autodiff (5), advanced (5)
- Document selection in `docs/EXAMPLE_CATALOG.md`

**Task 3: Create Directory Structure**
```bash
mkdir -p examples/01-basics
mkdir -p examples/02-list-ops
mkdir -p examples/03-higher-order
mkdir -p examples/04-numerical
mkdir -p examples/05-autodiff
mkdir -p examples/06-advanced
mkdir -p examples/showcase
```

**Tasks 4-5: Update Examples (Batch 1 & 2)**
For each example:
- Update syntax: Remove `(define main (lambda () ...))` pattern
- Add header comments explaining purpose
- Include expected output in comments
- Test to verify it works
- Move to appropriate category directory

**Tasks 6-8: Create Showcases**
- `showcase/mixed_types_demo.esk` - Tagged value system demonstration
- `showcase/higher_order_demo.esk` - Map/filter/fold comprehensive
- `showcase/autodiff_tutorial.esk` - Educational autodiff guide
- `showcase/vector_operations.esk` - Vector calculus showcase

---

### Documentation Work (Tasks 9-13)

**Task 9: Update README.md**

Changes needed:
```markdown
# REMOVE these sections:
- Debugger references
- Profiler references
- "Production-ready" claims (change to "Early adopter release")

# UPDATE these sections:
- Feature matrix to show actual status
- Installation (remove package managers until available)
- Examples section (link to new organized examples)

# ADD these sections:
- Quick Start link
- Known Limitations section
- Link to example catalog
```

**Task 10: Update GETTING_STARTED.md**

Remove:
- Lines 253-272: Debugger section
- Lines 285-299: Profiler section
- File I/O examples (not implemented)
- Editor integration (extensions don't exist)

Add:
- Troubleshooting section
- What works now section
- Known limitations section

**Tasks 11-13: Create New Docs**
- `docs/QUICK_START.md` - 5-minute tutorial
- `docs/EXAMPLES_GUIDE.md` - Comprehensive example guide
- `docs/TROUBLESHOOTING.md` - Common issues

---

### Infrastructure Work (Tasks 14-23)

**Task 14: GitHub Actions CI (Ubuntu)**

File: `.github/workflows/ci.yml`

```yaml
name: Eshkol CI

on:
  push:
    branches: [ main, develop, fix/autodiff ]
  pull_request:
    branches: [ main, develop ]

jobs:
  build-ubuntu:
    runs-on: ubuntu-22.04
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Install LLVM
        run: |
          sudo apt-get update
          sudo apt-get install -y llvm-14-dev clang-14 cmake
      
      - name: Build
        run: |
          mkdir build && cd build
          cmake ..
          make -j$(nproc)
      
      - name: Run Core Tests
        run: bash scripts/run_all_tests.sh
      
      - name: Run Autodiff Tests
        run: bash scripts/run_autodiff_tests.sh
      
      - name: Verify 100% Pass Rate
        run: |
          # Expect 110 total tests passing
          if [ -f test_results.txt ]; then
            pass_count=$(grep -c "PASS" test_results.txt || echo 0)
            if [ "$pass_count" -ne 110 ]; then
              echo "Expected 110 tests, got $pass_count"
              exit 1
            fi
          fi
      
      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results-ubuntu
          path: |
            test_outputs/
            autodiff_test_outputs/
```

**Task 15: Add macOS Job**

Same workflow file, add:
```yaml
  build-macos:
    runs-on: macos-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install LLVM
        run: brew install llvm cmake
      - name: Build
        env:
          LLVM_DIR: /opt/homebrew/opt/llvm
        run: |
          mkdir build && cd build
          cmake -DLLVM_DIR=$LLVM_DIR/lib/cmake/llvm ..
          make -j$(sysctl -n hw.ncpu)
      - name: Test
        run: |
          bash scripts/run_all_tests.sh
          bash scripts/run_autodiff_tests.sh
```

**Tasks 16-19: CMake Packaging**

Add to `CMakeLists.txt`:
- Install targets for executable, examples, docs
- CPack configuration for Debian
- Uninstall target
- Test installation process

**Tasks 20-22: Docker**

Create reproducible build environment:
- Dockerfile for Ubuntu 22.04
- Scripts to build in Docker
- Scripts to test package in clean container

**Task 23: Homebrew**

Create new repo `homebrew-eshkol` with:
- `Formula/eshkol.rb`
- Installation instructions
- Local testing

---

### Testing & Release Work (Tasks 24-33)

**Tasks 24-25: Integration Tests**

Create 4 comprehensive tests:
1. Mixed-type operations - All list operations with mixed types
2. Higher-order functions - Map/filter/fold with complex scenarios
3. Autodiff integration - Autodiff used with higher-order functions
4. Complex computation - Realistic scientific computing task

**Tasks 26-28: Validation**

- Memory testing with Valgrind (zero leaks required)
- Performance benchmarking (autodiff overhead)
- Testing documentation

**Tasks 29-32: Release Documentation**

Create professional release package:
- SECURITY.md for vulnerability reporting
- CODE_OF_CONDUCT.md for community
- CHANGELOG.md with v1.0-foundation entry
- RELEASE_NOTES_v1.0.md with highlights

**Task 33: Release**

Execute release process:
```bash
# Tag release
git tag -a v1.0-foundation -m "v1.0-foundation: Production-ready release"

# Build artifacts
cd build
cpack  # Creates eshkol_1.0.0_amd64.deb
cd ..
git archive --format=tar.gz --prefix=eshkol-1.0.0/ v1.0-foundation > eshkol-1.0.0.tar.gz

# Checksums
sha256sum build/eshkol_1.0.0_amd64.deb > eshkol_1.0.0_amd64.deb.sha256
sha256sum eshkol-1.0.0.tar.gz > eshkol-1.0.0.tar.gz.sha256

# Push
git push origin v1.0-foundation

# Create GitHub release
# Upload: .deb, tarball, checksums
# Include: RELEASE_NOTES_v1.0.md content
```

---

## Quality Gates

### Before Starting Phase 2 (Infrastructure)
- ✅ All 30 examples selected and working
- ✅ All examples tested manually
- ✅ Documentation accurate (no false claims)
- ✅ Example catalog complete

### Before Starting Phase 3 (Testing & Release)
- ✅ CI/CD running on Ubuntu + macOS
- ✅ All 110 tests passing in CI
- ✅ .deb package builds successfully
- ✅ Install/uninstall works correctly

### Before Release
- ✅ All 4 integration tests passing
- ✅ Valgrind reports zero leaks
- ✅ Autodiff overhead < 3x (or documented)
- ✅ All documentation accurate
- ✅ Release notes complete
- ✅ GitHub release ready

---

## Success Metrics

### v1.0-Foundation Release Success
| Metric | Target | How to Measure |
|--------|--------|----------------|
| Test pass rate | 100% | 110/110 tests |
| Memory safety | Zero leaks | Valgrind clean |
| Performance | < 3x autodiff | Benchmarks |
| Examples | 30 working | Manual verification |
| Documentation | 100% accurate | Review + test |
| CI/CD | Green builds | GitHub Actions |
| Installation | Works cleanly | Test on clean Ubuntu |

### Post-Release (Week 1)
- 50+ GitHub stars
- 10+ downloads
- Zero critical bugs
- Positive feedback

---

## Appendix: File Checklist

### Files to Create
- [ ] `.github/workflows/ci.yml`
- [ ] `cmake/cmake_uninstall.cmake.in`
- [ ] `docker/ubuntu/release/Dockerfile`
- [ ] `scripts/docker_build.sh`
- [ ] `scripts/docker_test.sh`
- [ ] `scripts/memory_test.sh`
- [ ] `scripts/performance_test.sh`
- [ ] `tests/integration/mixed_type_comprehensive.esk`
- [ ] `tests/integration/higher_order_comprehensive.esk`
- [ ] `tests/integration/autodiff_integration.esk`
- [ ] `tests/integration/complex_computation.esk`
- [ ] `docs/QUICK_START.md`
- [ ] `docs/EXAMPLES_GUIDE.md`
- [ ] `docs/TROUBLESHOOTING.md`
- [ ] `docs/TESTING.md`
- [ ] `docs/EXAMPLE_AUDIT.md`
- [ ] `docs/EXAMPLE_CATALOG.md`
- [ ] `SECURITY.md`
- [ ] `CODE_OF_CONDUCT.md`
- [ ] `CHANGELOG.md`
- [ ] `RELEASE_NOTES_v1.0.md`

### Files to Update
- [ ] `README.md` - Remove false claims, accurate feature matrix
- [ ] `docs/aidocs/GETTING_STARTED.md` - Remove debugger/profiler
- [ ] `CMakeLists.txt` - Add install targets and CPack
- [ ] 30 example files - Modernize syntax and add docs

### Examples to Reorganize
- Move 30 selected examples to new structure
- Create README.md in each category
- Add showcase/ directory with 4 examples

---

## Command Reference

### Building and Testing
```bash
# Build
cmake -B build && cmake --build build

# Run all tests
bash scripts/run_all_tests.sh              # Core: 67 tests
bash scripts/run_autodiff_tests.sh         # Autodiff: 43 tests
bash scripts/run_tests_with_output.sh      # With output capture

# Memory testing
bash scripts/memory_test.sh                # Valgrind (to create)

# Performance testing
bash scripts/performance_test.sh           # Benchmarks (to create)
```

### Installation Testing
```bash
# Install
sudo cmake --install build

# Verify
which eshkol-run
ls /usr/local/share/eshkol/examples

# Test
eshkol-run /usr/local/share/eshkol/examples/01-basics/hello.esk

# Uninstall
sudo cmake --build build --target uninstall
```

### Packaging
```bash
# Build .deb
cd build
cpack

# Test package
sudo dpkg -i eshkol_1.0.0_amd64.deb
eshkol-run /usr/share/eshkol/examples/01-basics/hello.esk
sudo dpkg -r eshkol
```

---

## Conclusion

We're in an **excellent position** for v1.0-foundation release:

✅ **Ahead of schedule**: 10 sessions saved from autodiff being complete  
✅ **Strong foundation**: 110 tests, 100% passing  
✅ **Clear path**: 33 well-defined tasks remaining  
✅ **Low risk**: Most remaining work is infrastructure, not implementation

**Recommended next step**: Begin example audit and curation (Tasks 1-2), which can inform all subsequent work.

---

**Document Status**: Active Fast Track Plan  
**Created**: November 27, 2025  
**Timeline**: 30 sessions (4-6 weeks)  
**Next Milestone**: Examples complete (Tasks 1-13)