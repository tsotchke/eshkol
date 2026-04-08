#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAKE_ROOT="${SCRIPT_DIR}/fake-llvm-root"

case "${1:-}" in
  --version)
    echo "21.1.8"
    ;;
  --cxxflags)
    echo "-I${FAKE_ROOT}/include -std=c++20 -DFAKE_LLVM=1"
    ;;
  --ldflags)
    echo "-L${FAKE_ROOT}/lib"
    ;;
  --libs)
    echo "-lLLVMCore -lLLVMSupport"
    ;;
  --system-libs)
    echo "-lpthread -ldl -lm"
    ;;
  --includedir)
    echo "${FAKE_ROOT}/include"
    ;;
  *)
    echo "unsupported arg: $*" >&2
    exit 1
    ;;
esac
