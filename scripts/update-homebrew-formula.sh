#!/bin/bash
# update-homebrew-formula.sh
#
# Updates packaging/homebrew/eshkol.rb (and optionally the tsotchke/homebrew-eshkol
# tap repository) to point at a new release tag.  Computes the sha256 of the
# GitHub source-archive tarball for the given tag.
#
# This script exists because the auto-Homebrew-update step was removed from
# .github/workflows/release.yml during the v1.1.13 release-workflow rewrite.
# Run this manually after each tagged release.
#
# Usage:
#   scripts/update-homebrew-formula.sh <tag> [--push-tap]
#
# Examples:
#   scripts/update-homebrew-formula.sh v1.1.13-accelerate
#   scripts/update-homebrew-formula.sh v1.1.14 --push-tap
#
# When --push-tap is given the script also commits and pushes the formula to
# the tsotchke/homebrew-eshkol tap repository (requires push access).

set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 <tag> [--push-tap]" >&2
    exit 1
fi

TAG="$1"
PUSH_TAP="${2:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FORMULA_PATH="$REPO_ROOT/packaging/homebrew/eshkol.rb"
TARBALL_URL="https://github.com/tsotchke/eshkol/archive/refs/tags/${TAG}.tar.gz"

if [ ! -f "$FORMULA_PATH" ]; then
    echo "error: formula not found at $FORMULA_PATH" >&2
    exit 1
fi

# Wait for GitHub to publish the source tarball.  GitHub generates this lazily,
# so the first request after a tag push may 404 — retry up to 10× with backoff.
echo "==> Fetching $TARBALL_URL ..."
TMPDIR_TGZ="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TGZ"' EXIT
TARBALL="$TMPDIR_TGZ/eshkol-${TAG}.tar.gz"

attempt=0
max_attempts=10
while [ $attempt -lt $max_attempts ]; do
    if curl --fail --silent --location --output "$TARBALL" "$TARBALL_URL"; then
        break
    fi
    attempt=$((attempt + 1))
    echo "    attempt $attempt/$max_attempts failed, retrying in 5s..." >&2
    sleep 5
done

if [ ! -s "$TARBALL" ]; then
    echo "error: failed to download $TARBALL_URL after $max_attempts attempts" >&2
    exit 1
fi

SHA256="$(shasum -a 256 "$TARBALL" | awk '{print $1}')"
echo "==> sha256: $SHA256"

# In-place update of the formula's url and sha256 lines.
python3 - "$FORMULA_PATH" "$TAG" "$SHA256" <<'PY'
import re
import sys
path, tag, sha = sys.argv[1:]
with open(path) as f:
    src = f.read()
src = re.sub(
    r'url "https://github\.com/tsotchke/eshkol/archive/(refs/tags/)?[^"]+"',
    f'url "https://github.com/tsotchke/eshkol/archive/refs/tags/{tag}.tar.gz"',
    src,
)
src = re.sub(r'sha256 "[^"]*"', f'sha256 "{sha}"', src)
with open(path, "w") as f:
    f.write(src)
print(f"==> updated {path}")
PY

# Optional: also push to the tap repository
if [ "$PUSH_TAP" = "--push-tap" ]; then
    TAP_DIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR_TGZ" "$TAP_DIR"' EXIT
    echo "==> Cloning tsotchke/homebrew-eshkol into $TAP_DIR"
    git clone --quiet --depth 1 git@github.com:tsotchke/homebrew-eshkol.git "$TAP_DIR/tap"
    cp "$FORMULA_PATH" "$TAP_DIR/tap/Formula/eshkol.rb"
    cd "$TAP_DIR/tap"
    if git diff --quiet; then
        echo "==> tap formula already up to date, nothing to push"
    else
        git add Formula/eshkol.rb
        git commit -m "eshkol: update to ${TAG}"
        git push origin HEAD
        echo "==> pushed updated formula to tsotchke/homebrew-eshkol"
    fi
fi

echo "==> done"
