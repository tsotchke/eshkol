# Eshkol v1.0.0-foundation Release Guide

**Repository:** github.com/tsotchke/eshkol
**Website:** eshkol.ai
**Release Type:** First Official Full Release
**Date:** December 2025

---

## Table of Contents

1. [Pre-Release Checklist](#1-pre-release-checklist)
2. [CI/CD Pipeline Setup](#2-cicd-pipeline-setup)
3. [Homebrew Distribution](#3-homebrew-distribution)
4. [Debian/APT Distribution](#4-debianaptdistribution)
5. [Release Process](#5-release-process)
6. [Post-Release](#6-post-release)

---

## 1. Pre-Release Checklist

### 1.1 Code Status

```
[x] All tests passing (295/295 = 100%)
[x] Build compiles cleanly
[x] Type system complete (HoTT phases 1-6)
[ ] Remove backup files
[ ] Update version numbers
[ ] Create CHANGELOG.md
[ ] Create RELEASE_NOTES.md
[ ] Clean up false documentation
```

### 1.2 File Cleanup

Remove these before release:

```bash
rm -f lib/backend/llvm_codegen.cpp.backup
rm -f lib/backend/llvm_codegen.cpp.bak
rm -f lib/backend/llvm_codegen.cpp.bak2
rm -f lib/backend/llvm_codegen.cpp.tmp
```

Add to `.gitignore`:
```
*.bak
*.backup
*.tmp
a.out
*.o
build/
```

### 1.3 Version Update

Current version location: `CMakeLists.txt` line 8

```cmake
set(ESHKOL_VERSION "1.0.0")
```

Also add to `inc/eshkol/eshkol.h`:
```c
#define ESHKOL_VERSION_MAJOR 1
#define ESHKOL_VERSION_MINOR 0
#define ESHKOL_VERSION_PATCH 0
#define ESHKOL_VERSION_STRING "1.0.0-foundation"
```

### 1.4 Required New Files

Create these files:

**CHANGELOG.md**
```markdown
# Changelog

## [1.0.0-foundation] - 2025-12-XX

### Added
- HoTT-based type system with universe levels (U0, U1, U2)
- Bidirectional type checking
- Type annotations: `(: name type)`, `(param : type)`
- Polymorphic types with `forall`
- Automatic differentiation (forward and reverse mode)
- Vector calculus: gradient, jacobian, hessian, divergence, curl, laplacian
- Arena-based memory management (OALR)
- Tail call optimization
- Module system with require/provide
- Interactive REPL with JIT
- Pre-compiled standard library

### Platforms
- macOS 12+ (x86_64, arm64)
- Ubuntu 22.04+ (x86_64)
- Debian 12+ (x86_64)
```

**SECURITY.md**
```markdown
# Security Policy

## Reporting Vulnerabilities

Email: security@eshkol.ai

Do NOT open public issues for security vulnerabilities.

## Response Timeline
- Initial response: 48 hours
- Status update: 7 days
```

---

## 2. CI/CD Pipeline Setup

### 2.1 GitHub Actions Structure

Create `.github/workflows/` directory:

```
.github/
├── workflows/
│   ├── ci.yml           # Build and test on every push
│   ├── release.yml      # Triggered on version tags
│   └── nightly.yml      # Optional: nightly builds
└── ISSUE_TEMPLATE/
    ├── bug_report.md
    └── feature_request.md
```

### 2.2 Main CI Workflow

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [master, develop, 'feat/**']
  pull_request:
    branches: [master]

jobs:
  build-linux:
    runs-on: ubuntu-22.04
    strategy:
      matrix:
        build_type: [Debug, Release]

    steps:
      - uses: actions/checkout@v4

      - name: Install Dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y \
            cmake ninja-build \
            llvm-17-dev libllvm17 \
            libreadline-dev

      - name: Configure
        run: |
          cmake -B build -G Ninja \
            -DCMAKE_BUILD_TYPE=${{ matrix.build_type }}

      - name: Build
        run: cmake --build build --parallel

      - name: Test
        run: |
          ./scripts/run_types_tests.sh
          ./scripts/run_cpp_type_tests.sh
          ./scripts/run_list_tests.sh
          ./scripts/run_autodiff_tests.sh
          ./scripts/run_memory_tests.sh
          ./scripts/run_modules_tests.sh
          ./scripts/run_stdlib_tests.sh
          ./scripts/run_features_tests.sh
          ./scripts/run_ml_tests.sh
          ./scripts/run_neural_tests.sh

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        if: matrix.build_type == 'Release'
        with:
          name: eshkol-linux-x64
          path: |
            build/eshkol-run
            build/eshkol-repl
            build/stdlib.o

  build-macos:
    runs-on: macos-14  # M1
    steps:
      - uses: actions/checkout@v4

      - name: Install Dependencies
        run: brew install llvm@17 cmake ninja readline

      - name: Build
        run: |
          export PATH="/opt/homebrew/opt/llvm@17/bin:$PATH"
          cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
          cmake --build build --parallel

      - name: Test
        run: |
          ./scripts/run_types_tests.sh
          ./scripts/run_list_tests.sh
          ./scripts/run_autodiff_tests.sh

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: eshkol-macos-arm64
          path: |
            build/eshkol-run
            build/eshkol-repl
            build/stdlib.o

  build-macos-intel:
    runs-on: macos-13  # Intel
    steps:
      - uses: actions/checkout@v4

      - name: Install Dependencies
        run: brew install llvm@17 cmake ninja readline

      - name: Build
        run: |
          export PATH="/usr/local/opt/llvm@17/bin:$PATH"
          cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
          cmake --build build --parallel

      - name: Upload Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: eshkol-macos-x64
          path: |
            build/eshkol-run
            build/eshkol-repl
            build/stdlib.o
```

### 2.3 Release Workflow

Create `.github/workflows/release.yml`:

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

permissions:
  contents: write

jobs:
  create-release:
    runs-on: ubuntu-22.04
    outputs:
      upload_url: ${{ steps.create.outputs.upload_url }}
    steps:
      - uses: actions/checkout@v4

      - name: Create Release
        id: create
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref_name }}
          release_name: Eshkol ${{ github.ref_name }}
          body: |
            See CHANGELOG.md for details.

            ## Quick Install

            **macOS:**
            ```bash
            brew tap tsotchke/eshkol
            brew install eshkol
            ```

            **Linux (Debian/Ubuntu):**
            ```bash
            curl -fsSL https://eshkol.ai/install.sh | bash
            ```
          draft: false
          prerelease: false

  build-and-upload:
    needs: create-release
    strategy:
      matrix:
        include:
          - os: ubuntu-22.04
            name: linux-x64
          - os: macos-14
            name: macos-arm64
          - os: macos-13
            name: macos-x64

    runs-on: ${{ matrix.os }}

    steps:
      - uses: actions/checkout@v4

      - name: Install Dependencies (Linux)
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install -y cmake ninja-build llvm-17-dev libreadline-dev

      - name: Install Dependencies (macOS)
        if: runner.os == 'macOS'
        run: brew install llvm@17 cmake ninja readline

      - name: Build
        run: |
          if [[ "$RUNNER_OS" == "macOS" ]]; then
            export PATH="/opt/homebrew/opt/llvm@17/bin:/usr/local/opt/llvm@17/bin:$PATH"
          fi
          cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
          cmake --build build --parallel

      - name: Package
        run: |
          mkdir -p pkg/bin pkg/lib pkg/share/eshkol
          cp build/eshkol-run pkg/bin/
          cp build/eshkol-repl pkg/bin/
          cp build/stdlib.o pkg/lib/
          cp lib/stdlib.esk pkg/share/eshkol/
          cp -r lib/core pkg/share/eshkol/
          cp README.md LICENSE pkg/
          tar -czvf eshkol-${{ github.ref_name }}-${{ matrix.name }}.tar.gz -C pkg .

      - name: Upload Release Asset
        uses: actions/upload-release-asset@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          upload_url: ${{ needs.create-release.outputs.upload_url }}
          asset_path: eshkol-${{ github.ref_name }}-${{ matrix.name }}.tar.gz
          asset_name: eshkol-${{ github.ref_name }}-${{ matrix.name }}.tar.gz
          asset_content_type: application/gzip
```

---

## 3. Homebrew Distribution

### 3.1 Create Homebrew Tap Repository

Create new repo: `github.com/tsotchke/homebrew-eshkol`

Structure:
```
homebrew-eshkol/
├── Formula/
│   └── eshkol.rb
├── README.md
└── LICENSE
```

### 3.2 Homebrew Formula

Create `Formula/eshkol.rb`:

```ruby
class Eshkol < Formula
  desc "Functional programming language with HoTT types and autodiff"
  homepage "https://eshkol.ai"
  url "https://github.com/tsotchke/eshkol/archive/v1.0.0-foundation.tar.gz"
  sha256 "REPLACE_WITH_ACTUAL_SHA256_AFTER_TAGGING"
  license "MIT"
  head "https://github.com/tsotchke/eshkol.git", branch: "master"

  depends_on "cmake" => :build
  depends_on "ninja" => :build
  depends_on "llvm@17"
  depends_on "readline"

  def install
    ENV["PATH"] = "#{Formula["llvm@17"].opt_bin}:#{ENV["PATH"]}"

    system "cmake", "-B", "build", "-G", "Ninja",
           "-DCMAKE_BUILD_TYPE=Release",
           *std_cmake_args
    system "cmake", "--build", "build"

    bin.install "build/eshkol-run"
    bin.install "build/eshkol-repl"
    lib.install "build/stdlib.o"
    (share/"eshkol").install "lib/stdlib.esk"
    (share/"eshkol/core").install Dir["lib/core/*"]
  end

  test do
    (testpath/"test.esk").write('(display "Hello")')
    system "#{bin}/eshkol-run", "test.esk", "-L#{lib}"
    assert_predicate testpath/"a.out", :exist?
  end
end
```

### 3.3 Installation for Users

After setup, users install via:

```bash
brew tap tsotchke/eshkol
brew install eshkol
```

### 3.4 Updating the Formula

After creating a release tag, update the formula:

```bash
# Get SHA256 of release tarball
curl -sL https://github.com/tsotchke/eshkol/archive/v1.0.0-foundation.tar.gz | shasum -a 256

# Update formula with new SHA256
# Push to homebrew-eshkol repo
```

---

## 4. Debian/APT Distribution

### 4.1 Self-Hosted APT Repository on eshkol.ai

Host APT packages on eshkol.ai using Railway or similar.

Directory structure for the repo:
```
apt/
├── pool/
│   └── main/
│       └── e/
│           └── eshkol/
│               └── eshkol_1.0.0-1_amd64.deb
├── dists/
│   └── stable/
│       ├── main/
│       │   └── binary-amd64/
│       │       └── Packages.gz
│       ├── Release
│       ├── Release.gpg
│       └── InRelease
└── eshkol.gpg  # Public key
```

### 4.2 Create Debian Package

Create `scripts/build-deb.sh`:

```bash
#!/bin/bash
set -e

VERSION="${1:-1.0.0}"
ARCH="amd64"
PKG_NAME="eshkol"
PKG_DIR="${PKG_NAME}_${VERSION}-1_${ARCH}"

echo "Building Debian package for $PKG_NAME $VERSION"

# Build first
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Create package structure
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/lib/eshkol"
mkdir -p "$PKG_DIR/usr/share/eshkol"

# Copy files
cp build/eshkol-run "$PKG_DIR/usr/bin/"
cp build/eshkol-repl "$PKG_DIR/usr/bin/"
cp build/stdlib.o "$PKG_DIR/usr/lib/eshkol/"
cp lib/stdlib.esk "$PKG_DIR/usr/share/eshkol/"
cp -r lib/core "$PKG_DIR/usr/share/eshkol/"

# Create control file
cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: eshkol
Version: ${VERSION}-1
Section: devel
Priority: optional
Architecture: ${ARCH}
Depends: libllvm17, libreadline8
Maintainer: Eshkol Team <team@eshkol.ai>
Description: Functional programming language with HoTT types
 Eshkol is a functional programming language featuring:
  - Homotopy Type Theory based type system
  - Automatic differentiation for ML
  - Arena-based memory management
  - Scheme-compatible syntax
Homepage: https://eshkol.ai
EOF

# Build package
dpkg-deb --build "$PKG_DIR"

echo "Package built: ${PKG_DIR}.deb"
```

### 4.3 GPG Key Setup

Generate a GPG key for package signing:

```bash
# Generate key (use email: packages@eshkol.ai)
gpg --full-generate-key

# Export public key
gpg --armor --export packages@eshkol.ai > eshkol.gpg

# Export for APT
gpg --armor --export packages@eshkol.ai | sudo tee /usr/share/keyrings/eshkol.gpg
```

### 4.4 Create APT Repository

Create `scripts/update-apt-repo.sh`:

```bash
#!/bin/bash
set -e

REPO_DIR="apt-repo"
DEB_FILE="$1"

if [ -z "$DEB_FILE" ]; then
    echo "Usage: $0 <deb-file>"
    exit 1
fi

# Create structure
mkdir -p "$REPO_DIR/pool/main/e/eshkol"
mkdir -p "$REPO_DIR/dists/stable/main/binary-amd64"

# Copy package
cp "$DEB_FILE" "$REPO_DIR/pool/main/e/eshkol/"

# Generate Packages file
cd "$REPO_DIR"
dpkg-scanpackages pool/ /dev/null | gzip -9c > dists/stable/main/binary-amd64/Packages.gz
dpkg-scanpackages pool/ /dev/null > dists/stable/main/binary-amd64/Packages

# Create Release file
cat > dists/stable/Release << EOF
Origin: Eshkol
Label: Eshkol
Suite: stable
Codename: stable
Version: 1.0
Architectures: amd64
Components: main
Description: Eshkol Programming Language
Date: $(date -R)
EOF

# Generate checksums
cd dists/stable
apt-ftparchive release . >> Release

# Sign
gpg --armor --detach-sign -o Release.gpg Release
gpg --armor --clearsign -o InRelease Release

echo "APT repository updated"
```

### 4.5 Installation for Users

Create `install.sh` to host at eshkol.ai:

```bash
#!/bin/bash
set -e

echo "Installing Eshkol..."

# Add GPG key
curl -fsSL https://eshkol.ai/apt/eshkol.gpg | sudo gpg --dearmor -o /usr/share/keyrings/eshkol.gpg

# Add repository
echo "deb [signed-by=/usr/share/keyrings/eshkol.gpg] https://eshkol.ai/apt stable main" | \
  sudo tee /etc/apt/sources.list.d/eshkol.list

# Install
sudo apt-get update
sudo apt-get install -y eshkol

echo "Eshkol installed successfully!"
echo "Run 'eshkol-repl' to start the REPL"
```

Users install via:
```bash
curl -fsSL https://eshkol.ai/install.sh | bash
```

---

## 5. Release Process

### 5.1 Pre-Release Steps

```bash
# 1. Ensure clean state
git status  # Should be clean

# 2. Run all tests
./scripts/run_types_tests.sh
./scripts/run_cpp_type_tests.sh
./scripts/run_list_tests.sh
./scripts/run_autodiff_tests.sh
./scripts/run_memory_tests.sh
./scripts/run_modules_tests.sh
./scripts/run_stdlib_tests.sh
./scripts/run_features_tests.sh
./scripts/run_ml_tests.sh
./scripts/run_neural_tests.sh

# 3. Update version in CMakeLists.txt
sed -i '' 's/ESHKOL_VERSION "[^"]*"/ESHKOL_VERSION "1.0.0"/' CMakeLists.txt

# 4. Update CHANGELOG.md

# 5. Commit
git add -A
git commit -m "chore: prepare v1.0.0-foundation release"

# 6. Create tag
git tag -a v1.0.0-foundation -m "Eshkol v1.0.0-foundation - First official release"

# 7. Push
git push origin master
git push origin v1.0.0-foundation
```

### 5.2 After GitHub Actions Complete

1. **Verify release assets** on GitHub Releases page
2. **Update Homebrew formula** with new SHA256
3. **Build and upload Debian package**
4. **Update eshkol.ai website**

### 5.3 Announcement

Draft announcement for:
- GitHub Releases (automatic from workflow)
- Twitter/X
- Hacker News (if appropriate)
- Reddit r/programming, r/ProgrammingLanguages

---

## 6. Post-Release

### 6.1 Verify Installations Work

```bash
# Test Homebrew
brew tap tsotchke/eshkol
brew install eshkol
eshkol-repl --version

# Test APT
curl -fsSL https://eshkol.ai/install.sh | bash
eshkol-repl --version

# Test from source
git clone https://github.com/tsotchke/eshkol.git
cd eshkol
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/eshkol-run --version
```

### 6.2 Monitor

- GitHub Issues for bug reports
- Build status in CI
- Download counts

### 6.3 Hotfix Process

If critical bugs found:

```bash
# Fix on master
git checkout master
# ... make fix ...
git commit -m "fix: critical bug description"

# Tag patch release
git tag -a v1.0.1-foundation -m "Hotfix: description"
git push origin master v1.0.1-foundation

# Update Homebrew formula
# Rebuild Debian package
```

---

## Appendix A: LLVM Version Requirements

Based on codebase analysis, LLVM 17 is the tested version. The code uses:
- Standard LLVM C++ API (IRBuilder, Module, Function)
- ORC JIT for REPL
- No version-specific features detected

**Minimum:** LLVM 15 (untested)
**Recommended:** LLVM 17
**Maximum tested:** LLVM 17

To support LLVM 18+, test and update if needed.

---

## Appendix B: Documentation Cleanup

Remove or update these files before release:

| File | Action |
|------|--------|
| docs/aidocs/GETTING_STARTED.md | Remove package manager claims, debugger section |
| docs/aidocs/EBPF_GUIDE.md | Remove (not implemented) |
| README.md | Update feature status table |
| docs/scheme_compatibility/KNOWN_ISSUES.md | Update to current status |

---

## Appendix C: Quick Reference

### Repository URLs

| Resource | URL |
|----------|-----|
| Main repo | github.com/tsotchke/eshkol |
| Homebrew tap | github.com/tsotchke/homebrew-eshkol |
| Website | eshkol.ai |
| APT repo | eshkol.ai/apt |
| Install script | eshkol.ai/install.sh |

### Key Commands

```bash
# Build
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# Test all
./scripts/run_types_tests.sh && \
./scripts/run_list_tests.sh && \
./scripts/run_autodiff_tests.sh

# Create release
git tag -a v1.0.0-foundation -m "Release message"
git push origin v1.0.0-foundation
```

---

*Document version: 1.0*
*Last updated: December 7, 2025*
