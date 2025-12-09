# Eshkol Release Guide

Step-by-step instructions for releasing Eshkol v1.0.0-foundation.

---

## Pre-Release Checklist

### 1. Verify All Tests Pass

```bash
# Run all test suites
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
```

All tests should pass before proceeding.

### 2. Verify Build Works

```bash
# Clean build
rm -rf build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel

# Verify executables exist
ls -la build/eshkol-run build/eshkol-repl build/stdlib.o
```

### 3. Clean Up Repository

```bash
# Remove backup files
find . -name "*.bak" -delete
find . -name "*.backup" -delete
find . -name "*.tmp" -delete

# Check for any uncommitted changes
git status
```

### 4. Update CHANGELOG.md

Edit `CHANGELOG.md` and set the release date:

```markdown
## [1.0.0-foundation] - 2025-12-09
```

---

## Creating the Release

### 1. Commit Final Changes

```bash
git add -A
git commit -m "chore: prepare v1.0.0-foundation release"
```

### 2. Create Git Tag

```bash
git tag -a v1.0.0-foundation -m "Eshkol v1.0.0-foundation - First official release"
```

### 3. Push to GitHub

```bash
git push origin master
git push origin v1.0.0-foundation
```

This triggers the GitHub Actions release workflow which will:
- Build for Linux x64, macOS ARM64, and macOS x64
- Create release tarballs
- Build Debian package
- Create GitHub Release with assets

---

## Post-Release: Homebrew Setup

### 1. Create Homebrew Tap Repository

Create a new GitHub repository: `tsotchke/homebrew-eshkol`

### 2. Get Release Tarball SHA256

```bash
curl -sL https://github.com/tsotchke/eshkol/archive/v1.0.0-foundation.tar.gz | shasum -a 256
```

### 3. Copy and Update Formula

Copy `packaging/homebrew/eshkol.rb` to the tap repository at `Formula/eshkol.rb`.

Replace `REPLACE_WITH_ACTUAL_SHA256` with the actual SHA256 from step 2.

### 4. Push Formula

```bash
cd homebrew-eshkol
git add Formula/eshkol.rb
git commit -m "Add eshkol 1.0.0-foundation"
git push
```

### 5. Test Installation

```bash
brew tap tsotchke/eshkol
brew install eshkol
eshkol-repl --version
```

---

## Post-Release: Debian/APT Repository (Optional)

If hosting an APT repository on eshkol.ai:

### 1. Build Debian Package Locally

```bash
# Using Docker
./scripts/make-docker.sh build 1.0.0 release

# Package will be in build/debian/release/
```

### 2. Set Up GPG Key (One-Time)

```bash
# Generate key for package signing
gpg --full-generate-key
# Use email: packages@eshkol.ai

# Export public key
gpg --armor --export packages@eshkol.ai > eshkol.gpg
```

### 3. Create APT Repository Structure

```bash
mkdir -p apt-repo/pool/main/e/eshkol
mkdir -p apt-repo/dists/stable/main/binary-amd64

# Copy package
cp eshkol_1.0.0-1_amd64.deb apt-repo/pool/main/e/eshkol/

# Generate package index
cd apt-repo
dpkg-scanpackages pool/ /dev/null | gzip -9c > dists/stable/main/binary-amd64/Packages.gz
```

### 4. Sign Repository

```bash
cd apt-repo/dists/stable
gpg --armor --detach-sign -o Release.gpg Release
gpg --armor --clearsign -o InRelease Release
```

### 5. Upload to eshkol.ai

Upload the `apt-repo/` directory to your web server.

---

## Verification Checklist

After release, verify:

- [ ] GitHub Release page shows all assets (3 tarballs + .deb)
- [ ] `brew install tsotchke/eshkol/eshkol` works on macOS
- [ ] Download links in release notes work
- [ ] Version shows correctly: `eshkol-repl` should display `1.0.0-foundation`

---

## Hotfix Process

If critical bugs are found after release:

```bash
# Fix the bug on master
git checkout master
# ... make fix ...
git commit -m "fix: description of fix"

# Create patch release
git tag -a v1.0.1-foundation -m "Hotfix: description"
git push origin master v1.0.1-foundation

# Update Homebrew formula with new SHA256
# Rebuild Debian package
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `.github/workflows/ci.yml` | CI build/test on every push |
| `.github/workflows/release.yml` | Automated release on tags |
| `docker/debian/release/Dockerfile` | Debian package build |
| `docker/ubuntu/release/Dockerfile` | Ubuntu package build |
| `packaging/homebrew/eshkol.rb` | Homebrew formula template |
| `cmake/Packing.cmake` | CPack configuration for .deb |
| `scripts/make-docker.sh` | Docker build orchestration |
| `CHANGELOG.md` | Version history |
| `SECURITY.md` | Security policy |

---

## Quick Commands Reference

```bash
# Run all tests
for script in scripts/run_*_tests.sh; do $script || exit 1; done

# Build release
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Create and push release
git tag -a v1.0.0-foundation -m "Release message" && git push origin v1.0.0-foundation

# Get tarball SHA256
curl -sL https://github.com/tsotchke/eshkol/archive/v1.0.0-foundation.tar.gz | shasum -a 256

# Build Debian package via Docker
./scripts/make-docker.sh build 1.0.0 release
```
