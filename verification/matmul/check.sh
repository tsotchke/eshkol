#!/bin/sh
set -eu
cd "$(dirname "$0")"
compiler=$(command -v eshkol-run)
runtime=$(dirname "$(dirname "$(realpath "$compiler")")")/lib/libeshkol-runtime.a
node replay.cjs
python3 shape-writers.py --compiler "$compiler"
python3 guard-replay.py --compiler "$compiler" --runtime "$runtime"
python3 publication-replay.py --runtime "$runtime"
python3 eligibility-replay.py --compiler "$compiler"
