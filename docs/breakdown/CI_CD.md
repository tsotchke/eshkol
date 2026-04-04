# CI/CD Pipelines

**Status:** Production (v1.1.11)
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

The CI runs across four platform configurations:

| Job | Runner | Architecture | Build Types |
|-----|--------|-------------|-------------|
| `build-linux` | `ubuntu-22.04` | x86-64 | Debug, Release |
| `build-linux-arm64` | `ubuntu-22.04-arm` | ARM64 | Release |
| `build-macos-arm` | `macos-14` | Apple Silicon (ARM64) | Release |
| `build-macos-intel` | `macos-15-large` | Intel (x86-64) | Release |

The Linux x86-64 job tests both Debug and Release configurations. All other platforms test Release only.

### Steps (per job)

1. **Checkout** the repository
2. **Install dependencies:**
   - Linux: LLVM 17 from apt.llvm.org, CMake, Ninja, readline, pkg-config
   - macOS: LLVM 17, CMake, Ninja, readline via Homebrew
3. **Configure** with CMake and Ninja generator
4. **Build** in parallel
5. **Run all tests** via `./scripts/run_all_tests.sh`
6. **Upload artifacts** (Release builds only): `eshkol-run`, `eshkol-repl`, `stdlib.o`

### Artifacts

Release builds upload platform-specific artifacts:

| Artifact Name | Contents |
|---------------|----------|
| `eshkol-linux-x64` | Linux x86-64 binaries |
| `eshkol-linux-arm64` | Linux ARM64 binaries |
| `eshkol-macos-arm64` | macOS Apple Silicon binaries |
| `eshkol-macos-x64` | macOS Intel binaries |

---

## Release Workflow (`release.yml`)

### Trigger

Push of a tag matching `v*` (e.g., `v1.1.11`, `v1.2.0-beta.1`).

### Jobs

The release workflow consists of seven jobs:

#### 1. create-release

Creates a GitHub Release with installation instructions and feature highlights. Tags containing `alpha`, `beta`, or `rc` are marked as pre-releases.

#### 2. build-and-upload

Builds on four platforms and uploads tarballs:

| Platform | Artifact |
|----------|----------|
| Linux x86-64 | `eshkol-<tag>-linux-x64.tar.gz` |
| Linux ARM64 | `eshkol-<tag>-linux-arm64.tar.gz` |
| macOS Apple Silicon | `eshkol-<tag>-macos-arm64.tar.gz` |
| macOS Intel | `eshkol-<tag>-macos-x64.tar.gz` |

Each tarball contains:
- `bin/` -- `eshkol-run`, `eshkol-repl`
- `lib/` -- `stdlib.o`, `libeshkol-static.a`
- `share/eshkol/` -- standard library source files (`core/`, `math/`, `signal/`, `ml/`, `random/`, `web/`, `tensor/`)
- `README.md`, `LICENSE`

#### 3. build-debian-package (x86-64)

Builds a `.deb` package for Debian/Ubuntu x86-64 systems using `scripts/build-deb.sh`.

#### 4. build-debian-package-arm64

Builds a `.deb` package for Debian/Ubuntu ARM64 systems.

#### 5. build-xla

Builds the XLA-enabled variant using `docker/xla/Dockerfile`. Produces `eshkol-<tag>-linux-x64-xla.tar.gz` with StableHLO/MLIR tensor operation support.

#### 6. build-cuda

Builds the CUDA-enabled variant using `docker/cuda/Dockerfile`. Produces `eshkol-<tag>-linux-x64-cuda.tar.gz` with native float64 GPU acceleration via CUDA.

#### 7. update-homebrew

Updates the Homebrew formula in the `tsotchke/homebrew-eshkol` tap repository. Only runs for stable releases (skips `alpha`, `beta`, `rc` tags). Computes the SHA256 of the source tarball and pushes an updated `Formula/eshkol.rb`.

### Release Artifacts Summary

| File | Platform | Special Features |
|------|----------|-----------------|
| `eshkol-<tag>-linux-x64.tar.gz` | Linux x86-64 | Standard build |
| `eshkol-<tag>-linux-arm64.tar.gz` | Linux ARM64 | Standard build |
| `eshkol-<tag>-macos-arm64.tar.gz` | macOS ARM64 | Metal SF64 GPU support |
| `eshkol-<tag>-macos-x64.tar.gz` | macOS Intel | Standard build |
| `eshkol_<ver>_amd64.deb` | Debian/Ubuntu x64 | System package |
| `eshkol_<ver>_arm64.deb` | Debian/Ubuntu ARM64 | System package |
| `eshkol-<tag>-linux-x64-xla.tar.gz` | Linux x86-64 | XLA/MLIR tensor ops |
| `eshkol-<tag>-linux-x64-cuda.tar.gz` | Linux x86-64 | CUDA GPU acceleration |

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
| LLVM | 17 | Compiler backend and JIT |
| CMake | 3.14+ | Build system |
| Ninja | latest | Build generator |
| readline | latest | REPL line editing |
| pkg-config | latest | Library discovery |
| OpenBLAS | latest | Linear algebra (Linux Docker builds) |

On macOS, LLVM 17 is installed via Homebrew (`brew install llvm@17`). On Linux, LLVM 17 is installed from the official LLVM apt repository at `apt.llvm.org`.

---

## Adding a New Workflow

When adding a new CI job:

1. Place the workflow in `.github/workflows/`
2. Use `ubuntu-22.04` or `macos-14` runners for consistency
3. Follow the existing LLVM 17 installation pattern
4. Run tests via `./scripts/run_all_tests.sh`
5. Upload artifacts with `actions/upload-artifact@v4`

---

## See Also

- [Docker](DOCKER.md) -- Docker images used by CUDA and XLA release builds
- [Getting Started](GETTING_STARTED.md) -- Build instructions
- [Developer Tools](DEVELOPER_TOOLS.md) -- Debugging and testing
- [Package Manager](PACKAGE_MANAGER.md) -- Homebrew formula details
