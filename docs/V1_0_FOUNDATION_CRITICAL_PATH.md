# Eshkol v1.0-Foundation Critical Path & Dependencies

**Created**: November 17, 2025  
**Purpose**: Track critical dependencies and execution path for v1.0-foundation  
**Timeline**: Sessions 21-60 (6-8 weeks)

---

## Critical Path Overview

### Definition
The **critical path** is the sequence of dependent tasks that determines the minimum time needed to complete v1.0-foundation. Any delay in critical path items delays the entire release.

### Critical Path Length
- **Total Sessions**: 40 (Sessions 21-60)
- **Critical Path Sessions**: 30 (can be compressed with parallelization)
- **Minimum Timeline**: 5 weeks (with parallel execution)
- **Expected Timeline**: 6-8 weeks (accounting for testing and integration)

---

## Critical Path Items (MUST Complete in Order)

### 🔴 **Tier 1: Blocking Items** (Cannot parallelize)

#### CP-1: Autodiff Investigation (Sessions 21-22)
- **Duration**: 2 sessions
- **Dependencies**: None (can start immediately)
- **Blocks**: All subsequent autodiff work
- **Risk**: Medium (investigation may reveal more complexity)
- **Deliverable**: `docs/AUTODIFF_TYPE_ANALYSIS.md`

**Why Critical**: Must understand issues before fixing them.

---

#### CP-2: SCH-006 Fix (Sessions 23-24)
- **Duration**: 2 sessions
- **Dependencies**: CP-1 complete
- **Blocks**: SCH-007, SCH-008 (type system changes affect other bugs)
- **Risk**: High (complex type inference)
- **Deliverable**: Type inference working for all autodiff modes

**Why Critical**: Foundation for other autodiff fixes. If type inference is broken, vector returns and type conflicts can't be properly tested.

---

#### CP-3: SCH-007 Fix (Sessions 25-26)
- **Duration**: 2 sessions
- **Dependencies**: CP-2 complete
- **Blocks**: Comprehensive testing (need vector returns to test gradients)
- **Risk**: Medium (LLVM vector types are well-documented)
- **Deliverable**: Vector returns working correctly

**Why Critical**: Required for gradient function to work, which is a headline feature.

---

#### CP-4: SCH-008 Fix (Sessions 27-28)
- **Duration**: 2 sessions
- **Dependencies**: CP-2, CP-3 complete
- **Blocks**: Autodiff testing (can't test if code won't compile)
- **Risk**: Medium (pattern is understood, implementation is straightforward)
- **Deliverable**: Clean LLVM IR generation

**Why Critical**: Blocks testing of complex autodiff programs. Without this, we can't validate the complete system.

---

#### CP-5: Comprehensive Autodiff Testing (Sessions 29-30)
- **Duration**: 2 sessions
- **Dependencies**: CP-2, CP-3, CP-4 all complete
- **Blocks**: Example creation (examples need working autodiff)
- **Risk**: Low (testing phase, not implementation)
- **Deliverable**: `tests/autodiff_comprehensive.esk` passing

**Why Critical**: Must verify all fixes before moving to examples. This is the quality gate for Month 2.

---

#### CP-6: CI/CD Setup - Ubuntu (Sessions 41-42)
- **Duration**: 2 sessions
- **Dependencies**: Autodiff fixes complete (tests must pass in CI)
- **Blocks**: macOS CI, all subsequent infrastructure
- **Risk**: Low (GitHub Actions is well-documented)
- **Deliverable**: `.github/workflows/ci.yml` working on Ubuntu

**Why Critical**: Foundation for all automated testing. Without this, we can't guarantee quality on every commit.

---

#### CP-7: CI/CD Setup - macOS (Sessions 43-44)
- **Duration**: 2 sessions
- **Dependencies**: CP-6 complete
- **Blocks**: Install targets (need CI to verify)
- **Risk**: Low (similar to Ubuntu setup)
- **Deliverable**: CI working on both platforms

**Why Critical**: Cross-platform validation. Can't release without macOS support.

---

#### CP-8: CMake Install Targets (Sessions 45-46)
- **Duration**: 2 sessions
- **Dependencies**: CP-7 complete
- **Blocks**: Packaging (install targets needed for packages)
- **Risk**: Low (CMake install is straightforward)
- **Deliverable**: `sudo make install` working

**Why Critical**: Required for packaging. Both .deb and Homebrew need proper install targets.

---

#### CP-9: CPack Debian Packaging (Sessions 47-48)
- **Duration**: 2 sessions
- **Dependencies**: CP-8 complete
- **Blocks**: Integration testing (need packages to test installation)
- **Risk**: Low (CPack is well-documented)
- **Deliverable**: Working .deb package

**Why Critical**: Primary distribution method for Linux users.

---

#### CP-10: Integration Testing (Sessions 53-54)
- **Duration**: 2 sessions
- **Dependencies**: All infrastructure complete (need packages to test)
- **Blocks**: Memory testing (integration tests must pass first)
- **Risk**: Low (tests are straightforward)
- **Deliverable**: 4 integration tests passing

**Why Critical**: Final validation before release. Must verify end-to-end workflows.

---

#### CP-11: Memory & Performance Testing (Sessions 55-56)
- **Duration**: 2 sessions
- **Dependencies**: CP-10 complete
- **Blocks**: Release (must be memory-clean and performant)
- **Risk**: Medium (may find issues requiring fixes)
- **Deliverable**: Valgrind clean, < 3x autodiff overhead verified

**Why Critical**: Quality gate for release. Can't ship with memory leaks or poor performance.

---

#### CP-12: v1.0-foundation Release (Sessions 59-60)
- **Duration**: 2 sessions
- **Dependencies**: ALL previous items complete
- **Blocks**: Nothing (this is the goal)
- **Risk**: Low (just release mechanics)
- **Deliverable**: PUBLIC RELEASE ✅

**Why Critical**: The ultimate deliverable.

---

## Parallel Path Items (Can Overlap)

### 🟡 **Tier 2: Parallelizable Items** (Can overlap with critical path)

#### PP-1: Example Curation (Sessions 31-32)
- **Duration**: 2 sessions
- **Dependencies**: CP-5 complete (need working autodiff for autodiff examples)
- **Can Overlap With**: None (sequentially before PP-2)
- **Risk**: Low

---

#### PP-2: Example Updates Batch 1 (Sessions 33-34)
- **Duration**: 2 sessions
- **Dependencies**: PP-1 complete
- **Can Overlap With**: CP-6, CP-7 (while CI is being set up)
- **Risk**: Low

---

#### PP-3: Example Updates Batch 2 (Sessions 35-36)
- **Duration**: 2 sessions
- **Dependencies**: PP-2 complete
- **Can Overlap With**: CP-7, CP-8 (while CI and install targets being set up)
- **Risk**: Low

---

#### PP-4: Showcase Examples (Sessions 37-38)
- **Duration**: 2 sessions
- **Dependencies**: PP-3 complete
- **Can Overlap With**: CP-8, CP-9 (while packaging is being set up)
- **Risk**: Low

---

#### PP-5: Documentation Pass (Sessions 39-40)
- **Duration**: 2 sessions
- **Dependencies**: PP-4 complete
- **Can Overlap With**: CP-9 (packaging)
- **Risk**: Low

---

#### PP-6: Docker Build (Sessions 49-50)
- **Duration**: 2 sessions
- **Dependencies**: CP-9 complete (need .deb package)
- **Can Overlap With**: CP-10 (integration testing)
- **Risk**: Low

---

#### PP-7: Homebrew Formula (Sessions 51-52)
- **Duration**: 2 sessions
- **Dependencies**: CP-8 complete (need install targets)
- **Can Overlap With**: CP-10 (integration testing)
- **Risk**: Low

---

#### PP-8: Final Documentation (Sessions 57-58)
- **Duration**: 2 sessions
- **Dependencies**: CP-11 complete (need final numbers for docs)
- **Can Overlap With**: Release preparation
- **Risk**: Low

---

## Dependency Matrix

### Forward Dependencies (What Each Item Blocks)

| Item | Blocks | Why |
|------|--------|-----|
| CP-1: Investigation | CP-2, CP-3, CP-4 | Can't fix without understanding |
| CP-2: SCH-006 | CP-3, CP-4, CP-5 | Type inference affects everything |
| CP-3: SCH-007 | CP-5 | Can't test gradients without vector returns |
| CP-4: SCH-008 | CP-5 | Can't test if code won't compile |
| CP-5: Autodiff Test | PP-1, CP-6 | Examples need working autodiff, CI needs passing tests |
| CP-6: CI Ubuntu | CP-7, CP-8 | macOS CI needs Ubuntu CI pattern, install needs CI |
| CP-7: CI macOS | CP-8 | Install needs both platforms verified |
| CP-8: Install | CP-9, PP-7 | Packaging needs install targets |
| CP-9: Debian | CP-10, PP-6 | Integration needs packages, Docker needs .deb |
| CP-10: Integration | CP-11 | Memory testing needs integration tests passing |
| CP-11: Memory/Perf | CP-12 | Can't release without clean memory and good performance |

### Backward Dependencies (What Each Item Requires)

| Item | Requires | Why |
|------|----------|-----|
| PP-1: Example Curation | CP-5 | Need working autodiff to select autodiff examples |
| PP-2: Examples Batch 1 | PP-1 | Need curated list |
| PP-3: Examples Batch 2 | PP-2 | Sequential batches |
| PP-4: Showcase | PP-3 | Build on updated examples |
| PP-5: Documentation | PP-4, CP-5 | Need final examples and autodiff status |
| PP-6: Docker | CP-9 | Needs .deb package |
| PP-7: Homebrew | CP-8 | Needs install targets |
| PP-8: Final Docs | CP-11 | Needs final performance numbers |

---

## Parallelization Opportunities

### Week-by-Week Parallel Execution

#### **Week 3: Autodiff Fixes** (Sessions 21-30)
```
SERIAL (Critical Path):
├─ Session 21-22: Investigation [CP-1]
├─ Session 23-24: SCH-006 Fix [CP-2]
├─ Session 25-26: SCH-007 Fix [CP-3]
├─ Session 27-28: SCH-008 Fix [CP-4]
└─ Session 29-30: Autodiff Testing [CP-5]

PARALLEL: None this week (all critical path)

Timeline: 10 sessions (cannot compress)
```

#### **Week 4: Examples & Documentation** (Sessions 31-40)
```
SERIAL:
├─ Session 31-32: Example Curation [PP-1]
├─ Session 33-34: Examples Batch 1 [PP-2]
├─ Session 35-36: Examples Batch 2 [PP-3]
└─ Session 37-38: Showcase [PP-4]

OVERLAP WITH WEEK 5:
└─ Session 39-40: Documentation [PP-5] ← Can start while CI setup begins

Timeline: 10 sessions (can compress to 8 if overlap with Week 5)
```

#### **Week 5: CI/CD & Packaging** (Sessions 41-50)
```
SERIAL (Critical Path):
├─ Session 41-42: CI Ubuntu [CP-6]
├─ Session 43-44: CI macOS [CP-7]
├─ Session 45-46: Install [CP-8]
└─ Session 47-48: Debian [CP-9]

PARALLEL (Can overlap):
├─ Session 49-50: Docker [PP-6] ← Can start after CP-9
└─ Session 39-40: Docs [PP-5] ← Can overlap with Session 41-42

Timeline: 10 sessions (can compress to 8-9 with parallelization)
```

#### **Week 6: Release Prep** (Sessions 51-60)
```
PARALLEL START:
├─ Session 51-52: Homebrew [PP-7] ← Needs CP-8
└─ Session 51-52: Docker [PP-6] ← If not done in Week 5

SERIAL (Critical Path):
├─ Session 53-54: Integration Testing [CP-10]
├─ Session 55-56: Memory/Performance [CP-11]
└─ Session 59-60: Release [CP-12]

PARALLEL:
└─ Session 57-58: Final Docs [PP-8] ← Can overlap with Session 59-60 prep

Timeline: 10 sessions (can compress to 8 with parallelization)
```

### Optimal Parallelization Strategy

```
Week 3: 10 sessions (no compression possible)
Week 4: 8 sessions (2 sessions saved by overlapping with Week 5)
Week 5: 8 sessions (2 sessions saved by parallelizing Docker)
Week 6: 8 sessions (2 sessions saved by parallelizing docs)

TOTAL: 34 sessions minimum (vs 40 sequential)

TIMELINE: 
- Conservative: 8 weeks (1 week = 5 sessions)
- Aggressive: 5 weeks (1 week = 6-7 sessions with parallelization)
- Recommended: 6-7 weeks (balanced pace with quality)
```

---

## Resource Allocation

### Single Developer Timeline

If working solo:
- **Sequential Execution**: 40 sessions
- **5 sessions/week**: 8 weeks
- **6 sessions/week**: 6.7 weeks
- **Can't parallelize** without additional resources

### Two Developer Timeline

If two developers available:
- **One on critical path**: Autodiff → CI/CD → Release
- **One on parallel path**: Examples → Documentation → Testing
- **Timeline**: ~5 weeks (significant compression)

**Recommended split**:
```
Developer A (Critical Path):
├─ Sessions 21-30: Autodiff fixes (Week 3)
├─ Sessions 41-50: CI/CD setup (Week 5)
└─ Sessions 59-60: Release (Week 6)

Developer B (Parallel Path):
├─ Sessions 31-40: Examples + Docs (Week 4)
├─ Sessions 51-56: Integration + Memory testing (Week 6)
└─ Sessions 57-58: Final docs (Week 6)

COORDINATION POINTS:
- End of Week 3: Handoff (Autodiff → Examples)
- End of Week 4: Handoff (Examples → CI)
- End of Week 5: Sync (CI + Examples both ready)
```

---

## Task Dependency Graph

### Session-Level Dependencies

```
Sessions 21-22 (CP-1) ──────────────┐
                                     ↓
Sessions 23-24 (CP-2) ──────────────┼─────────┐
                                     ↓         ↓
Sessions 25-26 (CP-3) ──────────────┤         ↓
                                     ↓         ↓
Sessions 27-28 (CP-4) ──────────────┤         ↓
                                     ↓         ↓
Sessions 29-30 (CP-5) ──────────────┼─────────┼─────────┐
                                     ↓         ↓         ↓
Sessions 31-32 (PP-1) ──────────────┘         ↓         ↓
         ↓                                     ↓         ↓
Sessions 33-34 (PP-2) ────────────────────────┤         ↓
         ↓                                     ↓         ↓
Sessions 35-36 (PP-3) ────────────────────────┤         ↓
         ↓                                     ↓         ↓
Sessions 37-38 (PP-4) ────────────────────────┤         ↓
         ↓                                     ↓         ↓
Sessions 39-40 (PP-5) ────────────────────────┤         ↓
                                               ↓         ↓
Sessions 41-42 (CP-6) ─────────────────────────┘        ↓
         ↓                                               ↓
Sessions 43-44 (CP-7) ──────────────────────────────────┤
         ↓                                               ↓
Sessions 45-46 (CP-8) ──────────────────────────────────┼─────────┐
         ↓                                               ↓         ↓
Sessions 47-48 (CP-9) ──────────────────────────────────┤         ↓
         ↓                                               ↓         ↓
Sessions 49-50 (PP-6) ──────────────────────────────────┘         ↓
                                                                   ↓
Sessions 51-52 (PP-7) ────────────────────────────────────────────┤
                                                                   ↓
Sessions 53-54 (CP-10) ───────────────────────────────────────────┤
         ↓                                                         ↓
Sessions 55-56 (CP-11) ───────────────────────────────────────────┤
         ↓                                                         ↓
Sessions 57-58 (PP-8) ────────────────────────────────────────────┘
         ↓
Sessions 59-60 (CP-12) [RELEASE]
```

**Legend**:
- CP = Critical Path
- PP = Parallel Path
- ─── = Dependency link
- ┐└├┤┼┘ = Dependency merge points

---

## Blockers and Risks

### High-Risk Blockers

#### Blocker 1: Autodiff Type Inference Complexity
- **Item**: CP-2 (Sessions 23-24)
- **Risk**: May take longer than 2 sessions
- **Impact**: Delays all subsequent work
- **Mitigation**: 
  - Allocate buffer time (Sessions 27-30 can absorb overflow)
  - Daily progress checkpoints
  - Can pivot to simpler approach if needed
- **Contingency**: Document limitations if can't fully fix, release with known issues

#### Blocker 2: CI/CD Setup Issues
- **Item**: CP-6, CP-7 (Sessions 41-44)
- **Risk**: Platform-specific issues may arise
- **Impact**: Delays packaging and release
- **Mitigation**:
  - Use proven GitHub Actions templates
  - Test locally before committing
  - Have both Ubuntu and macOS environments available
- **Contingency**: Start with Ubuntu only, add macOS in v1.0.1

#### Blocker 3: Memory Leaks Found
- **Item**: CP-11 (Sessions 55-56)
- **Risk**: Valgrind may find leaks requiring fixes
- **Impact**: Delays release
- **Mitigation**:
  - Run Valgrind throughout development (not just at end)
  - Fix issues as they arise
  - Have memory debugging expertise available
- **Contingency**: Document known leaks if can't fix, release with warnings

### Medium-Risk Items

#### Risk 1: Example Update Delays
- **Items**: PP-2, PP-3 (Sessions 33-36)
- **Risk**: Some examples may be broken beyond repair
- **Impact**: May not achieve 30 examples
- **Mitigation**: Select 40 candidates, aim for 30, accept 25 minimum
- **Contingency**: Release with 25 examples, add more in v1.0.1

#### Risk 2: Performance Target Not Met
- **Item**: CP-11 (Sessions 55-56)
- **Risk**: Autodiff overhead may exceed 3x
- **Impact**: Performance claims need adjustment
- **Mitigation**: 
  - Benchmark throughout autodiff development
  - Optimize during Sessions 29-30 if needed
  - Have profiling tools ready
- **Contingency**: Document actual overhead, adjust claims

### Low-Risk Items

All infrastructure setup (Docker, Homebrew) is low-risk as it's well-documented and doesn't block core functionality.

---

## Progress Tracking

### Daily Tracking Template

```markdown
## Session [N] Progress - [Date]

**Objective**: [Session goal from execution plan]

**Started**: [Time]
**Completed**: [Time]
**Duration**: [Hours]

### What Was Done
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

### Issues Encountered
- Issue 1: [Description] → [Resolution]
- Issue 2: [Description] → [Status]

### Tests Run
- Test 1: [Pass/Fail]
- Test 2: [Pass/Fail]

### Commits Made
- Commit SHA: [hash]
- Message: [message]

### Blockers
- [ ] Blocker 1: [Description]
- [ ] Blocker 2: [Description]

### Next Session
**Objective**: [Next session goal]
**Dependencies**: [What must be complete]
**Estimated Duration**: [Hours]

---

**Status**: [Complete ✅ / In Progress ⏳ / Blocked ❌]
```

### Weekly Tracking Template

```markdown
## Week [N] Summary - [Date Range]

**Sessions Completed**: [X/5]
**Critical Path Progress**: [X% complete]

### Achievements
- Achievement 1
- Achievement 2

### Metrics
- Tests passing: [X/66]
- CI status: [Green/Red]
- Memory: [Clean/Issues]

### Risks Materialized
- Risk 1: [Description] → [Impact]
- Mitigation 1: [What we did]

### Next Week Plan
- Focus: [Area]
- Critical items: [List]
- Expected completion: [Items]

### Timeline Adjustment
- Original estimate: [X sessions]
- Actual: [Y sessions]
- Variance: [+/- Z sessions]
- Reason: [Why]

---

**On Track**: [Yes ✅ / No ❌ / At Risk ⚠️]
```

---

## Critical Path Timeline

### Gantt Chart (Session Level)

```
Week 3 (Sessions 21-30): AUTODIFF FIXES
S21 ██ Investigation
S22 ██ Investigation  
S23 ██ SCH-006 Fix
S24 ██ SCH-006 Fix
S25 ██ SCH-007 Fix
S26 ██ SCH-007 Fix
S27 ██ SCH-008 Fix
S28 ██ SCH-008 Fix
S29 ██ Testing
S30 ██ Testing

Week 4 (Sessions 31-40): EXAMPLES & DOCS
S31 ██ Curation
S32 ██ Curation
S33 ██ Batch 1
S34 ██ Batch 1
S35 ██ Batch 2
S36 ██ Batch 2
S37 ██ Showcase
S38 ██ Showcase
S39 ██ Docs          ← Can overlap with Week 5 start
S40 ██ Docs

Week 5 (Sessions 41-50): CI/CD
S41 ██ CI Ubuntu     ← Can start while S39-40 finishing
S42 ██ CI Ubuntu
S43 ██ CI macOS
S44 ██ CI macOS
S45 ██ Install
S46 ██ Install
S47 ██ Debian
S48 ██ Debian
S49 ██ Docker        ← Parallel with Integration prep
S50 ██ Docker

Week 6 (Sessions 51-60): RELEASE
S51 ██ Homebrew      ← Parallel with Integration
S52 ██ Homebrew
S53 ██ Integration
S54 ██ Integration
S55 ██ Memory/Perf
S56 ██ Memory/Perf
S57 ██ Final Docs    ← Parallel with Release prep
S58 ██ Final Docs
S59 ██ RELEASE
S60 ██ RELEASE

Legend:
██ Critical Path (blocking)
██ Parallel Path (can overlap)
```

---

## Risk-Adjusted Timeline

### Conservative Estimate (No Issues)
- **Week 3**: 10 sessions (autodiff)
- **Week 4**: 8 sessions (examples, overlap)
- **Week 5**: 8 sessions (CI/CD, parallel)
- **Week 6**: 8 sessions (release, parallel)
- **Total**: 34 sessions = **~6-7 weeks**

### Realistic Estimate (Minor Issues)
- **Week 3**: 12 sessions (+20% buffer for autodiff complexity)
- **Week 4**: 10 sessions (+25% buffer for example issues)
- **Week 5**: 9 sessions (+12% buffer for CI issues)
- **Week 6**: 9 sessions (+12% buffer for testing issues)
- **Total**: 40 sessions = **8 weeks**

### Pessimistic Estimate (Major Issues)
- **Week 3**: 15 sessions (+50% buffer)
- **Week 4**: 10 sessions
- **Week 5**: 10 sessions
- **Week 6**: 10 sessions
- **Total**: 45 sessions = **9 weeks**

**Recommended Planning**: Use realistic estimate (8 weeks) with built-in buffer.

---

## Milestone Checkpoints

### Checkpoint 1: End of Week 3 (Session 30)
**Date**: ~2 weeks from start  
**Critical Question**: Is autodiff fully working?

**Validation**:
- ✅ SCH-006 resolved (type inference)
- ✅ SCH-007 resolved (vector returns)
- ✅ SCH-008 resolved (type conflicts)
- ✅ `tests/autodiff_comprehensive.esk` passing
- ✅ Performance < 3x verified

**GO/NO-GO Decision**:
- **GO**: Proceed to examples and infrastructure
- **NO-GO**: Extend autodiff work, push back timeline

---

### Checkpoint 2: End of Week 4 (Session 40)
**Date**: ~4 weeks from start  
**Critical Question**: Are examples and docs ready?

**Validation**:
- ✅ 30 examples selected and categorized
- ✅ All examples using current syntax
- ✅ Showcase examples created
- ✅ Documentation accurate (no false claims)
- ✅ Ready for CI integration

**GO/NO-GO Decision**:
- **GO**: Proceed to CI/CD setup
- **NO-GO**: Finish examples, may reduce from 30 to 25

---

### Checkpoint 3: End of Week 5 (Session 50)
**Date**: ~5 weeks from start  
**Critical Question**: Is infrastructure ready?

**Validation**:
- ✅ CI working on Ubuntu + macOS
- ✅ Install targets functional
- ✅ .deb package builds
- ✅ Docker build system working
- ✅ All tests passing in CI

**GO/NO-GO Decision**:
- **GO**: Proceed to release prep
- **NO-GO**: Fix infrastructure, may drop Docker/Homebrew from v1.0

---

### Checkpoint 4: End of Week 6 (Session 60)
**Date**: ~6 weeks from start  
**Critical Question**: Ready to release?

**Validation**:
- ✅ Integration tests passing
- ✅ Memory clean (Valgrind)
- ✅ Performance targets met
- ✅ All packages built
- ✅ Documentation finalized
- ✅ All quality gates passed

**GO/NO-GO Decision**:
- **GO**: PUBLIC RELEASE v1.0-foundation ✅
- **NO-GO**: Fix critical issues, release when ready

---

## Contingency Plans

### If Autodiff Takes Too Long (> Session 30)

**Scenario**: Type inference fix is more complex than expected

**Options**:
1. **Extend Timeline**: Add 1-2 weeks, push release date
2. **Reduce Scope**: Fix only SCH-008 (blocker), document SCH-006/007 as limitations
3. **Parallel Work**: Start examples while one person continues autodiff

**Recommended**: Option 1 (extend timeline) if < 5 sessions over, Option 2 if > 5 sessions over

---

### If CI/CD Has Issues (Sessions 41-50)

**Scenario**: GitHub Actions setup more complex than expected

**Options**:
1. **Start with Ubuntu Only**: Defer macOS to v1.0.1
2. **Use Docker CI**: Run tests in Docker instead of native runners
3. **Extend Timeline**: Add 1 week for CI setup

**Recommended**: Option 1 (Ubuntu only initially)

---

### If Examples Don't Reach 30

**Scenario**: Can't get 30 examples working by Session 40

**Options**:
1. **Accept 25 Examples**: Still good coverage
2. **Extend Examples Work**: Add 2 sessions (Sessions 40-42)
3. **Post-Release Examples**: Release with 25, add more in v1.0.1

**Recommended**: Option 1 (25 is acceptable)

---

### If Memory Issues Found (Session 55-56)

**Scenario**: Valgrind finds memory leaks

**Options**:
1. **Fix Immediately**: Add sessions to fix (may delay release)
2. **Document Known Leaks**: Release with known issues documented
3. **Defer to v1.0.1**: Release as-is, fix in patch

**Recommended**: Option 1 if fixable in < 3 sessions, Option 2 otherwise. Never Option 3 (memory leaks are critical).

---

## Daily Execution Checklist

### Every Session Checklist

**Before Starting**:
- [ ] Review session objectives from execution plan
- [ ] Check dependencies are complete
- [ ] Ensure environment is set up (LLVM, build, etc.)
- [ ] Have previous session's results available

**During Session**:
- [ ] Follow execution plan steps
- [ ] Test changes immediately
- [ ] Document issues as they arise
- [ ] Keep notes for session report

**After Session**:
- [ ] All tests passing
- [ ] Changes committed with session tag
- [ ] `docs/BUILD_STATUS.md` updated
- [ ] Blockers documented if any
- [ ] Next session prepared

---

## Decision Points

### When to Pivot

**Pivot Trigger 1**: Autodiff taking > 15 sessions
- **Action**: Reduce scope to SCH-008 only
- **Impact**: Document SCH-006/007 as known limitations
- **Timeline**: Saves 5+ sessions

**Pivot Trigger 2**: Examples taking > 12 sessions
- **Action**: Reduce to 25 examples
- **Impact**: Minimal (still good coverage)
- **Timeline**: Saves 2 sessions

**Pivot Trigger 3**: CI/CD blocked by platform issues
- **Action**: Ubuntu only for v1.0, macOS in v1.0.1
- **Impact**: Reduces initial platform support
- **Timeline**: Saves 2-4 sessions

### When to Extend Timeline

**Extend If**:
- Autodiff fixes taking 12-15 sessions (reasonable complexity)
- Memory issues found that can be fixed in < 3 sessions
- Performance optimization needed (< 2 sessions)

**Don't Extend If**:
- Examples taking too long (can reduce scope)
- Documentation not perfect (can iterate post-release)
- Nice-to-have features (can defer to v1.0.1)

---

## Success Criteria by Phase

### Month 2 Success (Session 40)
- ✅ All autodiff bugs fixed (SCH-006/007/008)
- ✅ Comprehensive autodiff test suite passing
- ✅ 30 examples updated and working
- ✅ Documentation accurate
- ✅ Ready for infrastructure phase

### Month 3 Success (Session 60)
- ✅ CI/CD working on Ubuntu + macOS
- ✅ .deb package builds and installs
- ✅ Homebrew formula ready
- ✅ Integration tests passing
- ✅ Memory clean (Valgrind)
- ✅ Performance targets met
- ✅ Ready for public release

### Release Success (Post-Session 60)
- ✅ v1.0-foundation released on GitHub
- ✅ Packages available for download
- ✅ Documentation published
- ✅ Community notified
- ✅ No critical bugs in first week
- ✅ Positive user feedback

---

## Communication Plan

### Internal Communication (Development Team)

**Daily** (if multi-person team):
- Standup: What was done, what's next, blockers
- Update critical path status
- Share findings/learnings

**Weekly**:
- Review progress against plan
- Check milestones
- Adjust timeline if needed
- Update stakeholders

**Monthly**:
- Comprehensive review
- Metrics analysis
- Planning adjustments
- Public progress report

### External Communication (Community)

**After Session 30** (Autodiff Complete):
- Blog post: "Autodiff System Complete"
- Social media: Progress update
- GitHub discussion: Technical details

**After Session 40** (Month 2 Complete):
- Progress report: "v1.0-foundation: 2 Months In"
- Preview release: Examples and capabilities
- Call for feedback on examples

**After Session 50** (Infrastructure Ready):
- Blog post: "CI/CD and Packaging Complete"
- Beta testing call (for integration tests)

**Session 60** (Release):
- Full release announcement
- Blog post series
- Social media campaign
- Community forums

---

## Appendix: Session Dependencies Reference

### Quick Lookup Table

| Session | Item | Type | Dependencies | Blocks | Risk |
|---------|------|------|--------------|--------|------|
| 21-22 | Investigation | CP | None | 23-60 | Med |
| 23-24 | SCH-006 | CP | 21-22 | 25-60 | High |
| 25-26 | SCH-007 | CP | 23-24 | 29-60 | Med |
| 27-28 | SCH-008 | CP | 23-24 | 29-60 | Med |
| 29-30 | Autodiff Test | CP | 23-28 | 31-60 | Low |
| 31-32 | Curation | PP | 29-30 | 33-40 | Low |
| 33-34 | Examples 1 | PP | 31-32 | 35-40 | Low |
| 35-36 | Examples 2 | PP | 33-34 | 37-40 | Low |
| 37-38 | Showcase | PP | 35-36 | 39-40 | Low |
| 39-40 | Docs | PP | 37-38 | 41+ | Low |
| 41-42 | CI Ubuntu | CP | 29-30 | 43-60 | Low |
| 43-44 | CI macOS | CP | 41-42 | 45-60 | Low |
| 45-46 | Install | CP | 43-44 | 47-60 | Low |
| 47-48 | Debian | CP | 45-46 | 49-60 | Low |
| 49-50 | Docker | PP | 47-48 | None | Low |
| 51-52 | Homebrew | PP | 45-46 | None | Low |
| 53-54 | Integration | CP | 47-52 | 55-60 | Low |
| 55-56 | Memory/Perf | CP | 53-54 | 59-60 | Med |
| 57-58 | Final Docs | PP | 55-56 | None | Low |
| 59-60 | RELEASE | CP | ALL | None | Low |

**Legend**:
- CP = Critical Path (blocking)
- PP = Parallel Path (can overlap)
- Risk: High/Med/Low

---

## Critical Path Optimization

### Strategies to Compress Timeline

**Strategy 1: Aggressive Parallelization**
- Run examples work while autodiff testing happens
- Start CI setup while final examples are being updated
- Begin integration tests while docs are being finalized
- **Potential Savings**: 4-6 sessions

**Strategy 2: Two-Track Development**
- Track A: Critical path (autodiff → CI → release)
- Track B: Parallel path (examples → docs → testing)
- Requires 2 developers
- **Potential Savings**: 10-12 sessions

**Strategy 3: Reduced Scope**
- 25 examples instead of 30
- Ubuntu CI only (macOS in v1.0.1)
- Skip Docker (can add later)
- **Potential Savings**: 4-6 sessions
- **Impact**: Reduced initial value

**Recommended**: Strategy 1 (parallelization) if solo developer, Strategy 2 if team available.

### What NOT to Skip

**Never Skip**:
- ✅ Autodiff bug fixes (core value proposition)
- ✅ CI setup (quality requirement)
- ✅ Memory testing (critical for stability)
- ✅ Integration tests (validation requirement)
- ✅ Debian packaging (primary Linux distribution)

**Can Defer to v1.0.1**:
- Homebrew formula (nice-to-have for macOS)
- Docker images (convenience, not critical)
- 5-10 examples (25 is acceptable minimum)
- Final documentation polish (can iterate)

---

## Summary: Critical Path Highlights

### The 3 Critical Weeks

**Week 3: Autodiff** (Sessions 21-30)
- **Duration**: 10 sessions (cannot compress)
- **Risk**: HIGH
- **Impact**: Blocks everything else
- **Mitigation**: Buffer time built in

**Week 5: CI/CD** (Sessions 41-50)  
- **Duration**: 8-10 sessions (some parallelization possible)
- **Risk**: MEDIUM
- **Impact**: Required for release
- **Mitigation**: Use proven patterns

**Week 6: Release** (Sessions 51-60)
- **Duration**: 8-10 sessions (some parallelization possible)
- **Risk**: MEDIUM (depends on testing findings)
- **Impact**: The release itself
- **Mitigation**: Comprehensive testing throughout

### Key Dependencies
1. **Autodiff → Examples** (Examples need working autodiff)
2. **Examples → CI** (CI needs examples to test)
3. **CI → Packaging** (Packaging needs CI verification)
4. **Packaging → Integration** (Integration tests need packages)
5. **Integration → Release** (Must pass before release)

### Resource Requirements
- **Minimum**: 1 developer, 8 weeks
- **Optimal**: 2 developers, 5-6 weeks
- **Infrastructure**: LLVM 14+, Ubuntu 22.04, macOS (for testing)

---

**Document Status**: Critical path documented  
**Created**: November 17, 2025  
**Purpose**: Track dependencies and manage timeline  
**Updates**: Review after each checkpoint (Sessions 30, 40, 50)

**Related Documents**:
- [`V1_0_FOUNDATION_EXECUTION_PLAN.md`](V1_0_FOUNDATION_EXECUTION_PLAN.md) - Detailed steps
- [`V1_0_FOUNDATION_ARCHITECTURE_DIAGRAMS.md`](V1_0_FOUNDATION_ARCHITECTURE_DIAGRAMS.md) - Visual architecture
- [`BUILD_STATUS.md`](BUILD_STATUS.md) - Current build status

---

**END OF CRITICAL PATH DOCUMENT**