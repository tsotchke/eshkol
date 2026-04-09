# CI/CD Pipelines

**Status:** Production (v1.1.13)
**Applies to:** Eshkol compiler v1.1-accelerate and later

---

## Overview

Eshkol uses three GitHub Actions workflows for continuous integration, release publishing, and website deployment. All workflows are defined in `.github/workflows/`.

| Workflow | File | Purpose |
|----------|------|---------|
| CI | `ci.yml` | Build and test on every push and pull request |
| Release | `release.yml` | Build artifacts and publish on version tags |
| Deploy Site | `pages.yml` | Deploy project website to GitHub Pages |

---

## CI Workflow (`ci.yml`)

### Triggers

- **Push** to `master`, `develop`, or any `feat/**` branch
- **Pull request** targeting `master`

### Build Matrix

The CI runs as two matrix jobs: one Unix matrix and one Windows matrix.

#### Unix matrix

| Job | Runner | Variant |
|-----|--------|---------|
| `linux-x64-lite` | `ubuntu-22.04` | Lite |
| `linux-arm64-lite` | `ubuntu-22.04-arm` | Lite |
| `linux-x64-xla` | `ubuntu-22.04` | XLA |
| `linux-arm64-xla` | `ubuntu-22.04-arm` | XLA |
| `linux-x64-cuda` | `ubuntu-22.04` | CUDA |
| `linux-arm64-cuda` | `ubuntu-22.04-arm` | CUDA |
| `macos-arm64-lite` | `macos-14` | Lite |
| `macos-x64-lite` | `macos-15-intel` | Lite |
| `macos-arm64-xla` | `macos-14` | XLA |
| `macos-x64-xla` | `macos-15-intel` | XLA |

#### Windows matrix

| Job | Runner | Variant |
|-----|--------|---------|
| `windows-x64-lite` | `windows-latest` | Lite |
| `windows-arm64-lite` | `windows-11-arm` | Lite |
| `windows-x64-xla` | `windows-latest` | XLA |
| `windows-arm64-xla` | `windows-11-arm` | XLA |
| `windows-x64-cuda` | `windows-latest` | CUDA |
| `windows-arm64-cuda` | `windows-11-arm` | CUDA |

### Steps (per job)

1. **Checkout** the repository
2. **Install dependencies:**
   - Linux: LLVM 21 from apt.llvm.org, CMake, Ninja, readline, pkg-config
   - macOS: LLVM 21, CMake, Ninja, readline via Homebrew
   - Windows: Visual Studio 2022 + ClangCL, official LLVM 21 SDK archive, native CMake
3. **Configure** with CMake (Ninja on Linux/macOS, Visual Studio generator on Windows)
4. **Build** in parallel
5. **Run tests** via the lane-appropriate suite:
   - Lite: `run_all_tests.sh` or `run_all_tests.ps1`
   - XLA: `run_xla_tests.sh` or `run_all_tests.ps1 -Mode xla`
   - CUDA: `run_gpu_tests.sh` or `run_all_tests.ps1 -Mode gpu`
6. **Upload artifacts** for lite lanes only: `eshkol-run`, `eshkol-repl`, `stdlib.o`

### Artifacts

Release builds upload platform-specific artifacts:

| Artifact Name | Contents |
|---------------|----------|
| `eshkol-linux-x64` | Linux x86-64 binaries |
| `eshkol-linux-arm64` | Linux ARM64 binaries |
| `eshkol-macos-arm64` | macOS Apple Silicon binaries |
| `eshkol-macos-x64` | macOS Intel binaries |
| `eshkol-windows-x64` | Windows x86-64 binaries |
| `eshkol-windows-arm64` | Windows ARM64 binaries |

---

## Release Workflow (`release.yml`)

### Trigger

Push of an annotated or lightweight tag matching `v*` (for example `v1.1.13` or `v1.2.0-rc.1`).

### How to Trigger It

For a real release on the canonical repository:

```bash
git checkout master
git pull --ff-only origin master
git tag -a v1.1.13 -m "Eshkol v1.1.13"
git push origin v1.1.13
```

For a dry run on a fork, push the tag to the fork remote instead:

```bash
git tag -a v1.1.13-rc.1 -m "Eshkol v1.1.13-rc.1"
git push fork v1.1.13-rc.1
```

The workflow runs against the exact commit the tag points at. It is not manually dispatched and it does not wait for branch pushes. If you need to rerun it after changing `release.yml`, delete the tag locally and remotely, recreate it on the fixed commit, and push it again.

```bash
git tag -d v1.1.13-rc.1
git push --delete fork v1.1.13-rc.1
git tag -a v1.1.13-rc.1 -m "Eshkol v1.1.13-rc.1"
git push fork v1.1.13-rc.1
```

Tags containing `alpha`, `beta`, or `rc` are published as GitHub pre-releases automatically.

### Jobs

The release workflow consists of three logical stages:

#### 1. `unix-release-matrix`

Builds, tests, packages, and uploads release archives for:

- `linux-x64-lite`
- `linux-arm64-lite`
- `linux-x64-xla`
- `linux-arm64-xla`
- `linux-x64-cuda`
- `linux-arm64-cuda`
- `macos-arm64-lite`
- `macos-x64-lite`
- `macos-arm64-xla`
- `macos-x64-xla`

Each Unix lane produces `eshkol-<tag>-<lane>.tar.gz`.

#### 2. `windows-release-matrix`

Builds, tests, packages, and uploads release archives for:

- `windows-x64-lite`
- `windows-arm64-lite`
- `windows-x64-xla`
- `windows-arm64-xla`
- `windows-x64-cuda`
- `windows-arm64-cuda`

Each Windows lane produces `eshkol-<tag>-<lane>.zip`.

#### 3. `publish-release`

Downloads all packaged artifacts, generates `SHA256SUMS.txt`, and publishes a GitHub Release with auto-generated release notes.

### Release Artifacts Summary

| File Pattern | Platform | Format |
|-------------|----------|--------|
| `eshkol-<tag>-linux-*.tar.gz` | Linux x64/ARM64 | tar.gz |
| `eshkol-<tag>-macos-*.tar.gz` | macOS x64/ARM64 | tar.gz |
| `eshkol-<tag>-windows-*.zip` | Windows x64/ARM64 | zip |
| `SHA256SUMS.txt` | All artifacts | text |

Each packaged artifact contains:

- `bin/` — `eshkol-run`, `eshkol-repl`, plus Windows runtime DLLs when present
- `lib/stdlib.o`
- `lib/eshkol/stdlib.o`
- `share/eshkol/stdlib.esk`
- `share/eshkol/math.esk` when present
- `share/eshkol/lib/**/*.esk` for the standard library and module tree
- `README.md`, `LICENSE`, `CHANGELOG.md` when present

### Operational Notes

- The release workflow rebuilds and retests artifacts on the tag; it does not download artifacts from the CI workflow.
- Windows release lanes reuse the same per-architecture LLVM SDK cache keys as CI.
- A failed lane prevents `publish-release` from creating or updating the GitHub Release.
- Because the trigger is tag-based, changing `release.yml` alone does nothing until a matching tag is pushed.

---

## Deploy Site Workflow (`pages.yml`)

### Trigger

- **Push** to `master` when files under `site/static/` change
- **Manual dispatch** via GitHub UI

### What It Does

Deploys the contents of `site/static/` to GitHub Pages. This is a simple static site deployment with no build step -- the site content is pre-built and committed to the repository.

### Configuration

- Uses GitHub Pages environment with `id-token: write` permission
- Concurrency group `"pages"` prevents concurrent deployments
- Does not cancel in-progress deployments (`cancel-in-progress: false`)

---

## Dependencies

All CI and release jobs share the same dependency set:

| Dependency | Version | Purpose |
|------------|---------|---------|
| LLVM | 21 | Compiler backend and JIT |
| CMake | 3.14+ | Build system |
| Ninja | latest | Build generator |
| readline | latest | REPL line editing |
| pkg-config | latest | Library discovery |
| OpenBLAS | latest | Linear algebra (Linux Docker builds) |

On macOS, LLVM 21 is installed via Homebrew (`brew install llvm@21`). On Linux, LLVM 21 is installed from the official LLVM apt repository at `apt.llvm.org`. On Windows, the CI downloads the official LLVM 21 SDK archive and points `LLVM_DIR` at its CMake package.

---

## Adding a New Workflow

When adding a new CI job:

1. Place the workflow in `.github/workflows/`
2. Use `ubuntu-22.04` or `macos-14` runners for consistency
3. Follow the existing LLVM 21 installation pattern and prefer explicit `LLVM_CONFIG_EXECUTABLE` or `LLVM_DIR`
4. Run tests via `./scripts/run_all_tests.sh`
5. Upload artifacts with `actions/upload-artifact@v4`

---

## GitLab CI (Parallel Pipeline)

The project runs a parallel CI pipeline on GitLab CI in addition to GitHub Actions, providing additional platform coverage and build matrix flexibility.

**File:** `.gitlab-ci.yml`

### Build Matrix

| Platform | Architecture | Variant |
|----------|-------------|---------|
| Linux (Ubuntu) | x86-64 | lite |
| Linux (Ubuntu) | x86-64 | XLA |
| Linux (Ubuntu) | x86-64 | CUDA |
| Linux (Ubuntu) | ARM64 | lite |
| Linux (Ubuntu) | ARM64 | XLA |
| macOS | ARM64 (Apple Silicon) | lite |
| macOS | ARM64 (Apple Silicon) | XLA |
| Windows | x86-64 | lite (VS 2022 + ClangCL) |

### Toolchain

All GitLab CI runners use **LLVM 21** via `cmake/LLVMToolchain.cmake` — the same toolchain contract as local builds and GitHub Actions. Linux runners install from the official LLVM apt repository (`llvm-toolchain-jammy-21`). macOS runners use Homebrew `llvm@21`. Windows uses the LLVM 21 SDK with Visual Studio 2022.

### Pipeline Stages

1. **build** — CMake configure + build (Ninja)
2. **test** — `scripts/run_all_tests.sh`
3. **package** — artifact collection (release matrix only)

---

## See Also

- [Docker](DOCKER.md) -- Docker images used by CUDA and XLA release builds
- [Getting Started](GETTING_STARTED.md) -- Build instructions
- [Developer Tools](DEVELOPER_TOOLS.md) -- Debugging and testing
- [Package Manager](PACKAGE_MANAGER.md) -- Homebrew formula details
