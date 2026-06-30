#!/bin/bash
# ast_dump_let_ops_test.sh — AST printer must not read the wrong operation
# union fields for let-family forms.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="$ROOT/${BUILD_DIR:-build}/eshkol-run"

if [ ! -x "$RUN" ]; then
    echo "SKIP: $RUN not built"
    exit 0
fi

WORK=$(mktemp -d -t eshkol_ast_dump_let.XXXXXX)
trap 'rm -rf "$WORK"' EXIT

PASS=0
FAIL=0

check_debug_ast() {
    local name="$1"
    local expr="$2"
    local expected="$3"
    local log="$WORK/${name}.log"

    if ESHKOL_PATH="$ROOT/lib" "$RUN" --no-stdlib -e "$expr" --debug >"$log" 2>&1; then
        if grep -qF "$expected" "$log" && ! grep -qF "fatal signal" "$log"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            echo "FAIL: $name"
            sed -n '1,120p' "$log" | sed 's/^/  /'
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: $name exited nonzero"
        sed -n '1,120p' "$log" | sed 's/^/  /'
    fi
}

check_dump_ast_file() {
    local src="$WORK/numeric_import.esk"
    local log="$WORK/dump_ast_file.log"
    cat > "$src" <<'ESK'
(require core.numeric_extras)
(display "ok")
(newline)
ESK

    if (cd "$WORK" && ESHKOL_PATH="$ROOT/lib" "$RUN" --dump-ast "$src" >"$log" 2>&1); then
        if grep -qF "Operation: DEFINE_OP" "$log" && ! grep -qF "fatal signal" "$log"; then
            PASS=$((PASS + 1))
        else
            FAIL=$((FAIL + 1))
            echo "FAIL: dump-ast numeric import output"
            sed -n '1,160p' "$log" | sed 's/^/  /'
        fi
    else
        FAIL=$((FAIL + 1))
        echo "FAIL: dump-ast numeric import exited nonzero"
        sed -n '1,160p' "$log" | sed 's/^/  /'
    fi
}

check_debug_ast "plain-let" "(let ((x 1)) x)" "Operation: LET_OP"
check_debug_ast "named-let" "(let loop ((x 1)) x)" "Name: loop"
check_debug_ast "let-star" "(let* ((x 1) (y x)) y)" "Operation: LET*_OP"
check_debug_ast "letrec-star" "(letrec* ((x 1)) x)" "Operation: LETREC*_OP"
check_dump_ast_file

echo "ast-dump-let-ops: $PASS pass, $FAIL fail"
exit $FAIL
