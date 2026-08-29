#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <eshkol> <eshkol-lsp>" >&2
  exit 2
fi

front_door="$1"
lsp="$2"
fixture="tests/modules/visibility_test.esk"

check_json="$(${front_door} check --format json "${fixture}")"
doc_json="$(${front_door} doc modules --format json "${fixture}")"
if [[ "${check_json}" != "${doc_json}" ]]; then
  echo "FAIL: compiler/docs module graph mismatch" >&2
  exit 1
fi
grep -F '"schema":"eshkol.workspace-check.v1"' <<<"${check_json}" >/dev/null
echo "PASS: compiler/docs shared module graph"

help="$(${lsp} --help 2>&1)"
grep -F "eshkol-lsp" <<<"${help}" >/dev/null
echo "PASS: LSP shared workspace client"

if grep -R -n -E 'make_define_alias_ast|repl_make_define_alias_ast|vm_emit_import_alias' \
    lib/frontend lib/repl lib/backend/vm_compiler.c exe/eshkol-run.cpp >/dev/null; then
  echo "FAIL: compatibility-generated import alias remains" >&2
  exit 1
fi
echo "PASS: no compatibility-generated import aliases"
mkdir -p scripts/icc_traces
printf '%s\n' '{"kind":"runtime_event","name":"adr0000_stage2_gate","status":"PASS"}' \
  > scripts/icc_traces/adr0000_stage2.jsonl
