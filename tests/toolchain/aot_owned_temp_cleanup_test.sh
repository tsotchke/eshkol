#!/usr/bin/env bash
set -euo pipefail

ESHKOL_RUN="${1:?usage: aot_owned_temp_cleanup_test.sh /path/to/eshkol-run}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/eshkol_named"

cat > "$WORK/program.esk" <<'EOF'
(define (main) 0)
EOF
cat > "$WORK/stdlib.c" <<'EOF'
void preserved_stdlib_object(void) {}
EOF
cat > "$WORK/helpers.c" <<'EOF'
void preserved_helper_object(void) {}
EOF
cc -c "$WORK/stdlib.c" -o "$WORK/eshkol_named/stdlib.o"
cc -c "$WORK/helpers.c" -o "$WORK/eshkol_named/my_eshkol_helpers.o"

out="$WORK/success"
"$ESHKOL_RUN" -n -o "$out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/success.log" 2>&1 || {
        cat "$WORK/success.log" >&2
        echo "FAIL: successful link fixture did not link" >&2
        exit 1
    }
test -x "$out" || { echo "FAIL: linked executable missing" >&2; exit 1; }
test -f "$WORK/eshkol_named/stdlib.o" || { echo "FAIL: cleanup deleted caller stdlib.o" >&2; exit 1; }
test -f "$WORK/eshkol_named/my_eshkol_helpers.o" || { echo "FAIL: cleanup deleted caller helper object" >&2; exit 1; }
test ! -e "$out.tmp.o" || { echo "FAIL: successful link left its owned .tmp.o" >&2; exit 1; }

cat > "$WORK/failing.c" <<'EOF'
extern void deliberately_missing_link_symbol(void);
void preserved_helper_object(void) { deliberately_missing_link_symbol(); }
EOF
cc -c "$WORK/failing.c" -o "$WORK/eshkol_named/my_eshkol_helpers.o"
failed_out="$WORK/failure"
if "$ESHKOL_RUN" -n -o "$failed_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/failure.log" 2>&1; then
    cat "$WORK/failure.log" >&2
    echo "FAIL: intentionally unresolved caller object unexpectedly linked" >&2
    exit 1
fi
test -f "$WORK/eshkol_named/stdlib.o" || { echo "FAIL: failed-link cleanup deleted caller stdlib.o" >&2; exit 1; }
test -f "$WORK/eshkol_named/my_eshkol_helpers.o" || { echo "FAIL: failed-link cleanup deleted caller helper object" >&2; exit 1; }
test ! -e "$failed_out.tmp.o" || { echo "FAIL: failed link left its owned .tmp.o" >&2; exit 1; }

# A similarly named, pre-existing caller object is never ours to overwrite.
existing_out="$WORK/existing"
printf 'caller-owned scratch-name contents\n' > "$existing_out.tmp.o"
cp "$existing_out.tmp.o" "$WORK/preexisting.expected"
if "$ESHKOL_RUN" -n -o "$existing_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/existing.log" 2>&1; then
    echo "FAIL: compiler overwrote an existing caller-owned temporary-name file" >&2
    exit 1
fi
cmp "$existing_out.tmp.o" "$WORK/preexisting.expected" || {
    echo "FAIL: compiler changed or removed a pre-existing caller object" >&2
    exit 1
}

echo "PASS: aot_owned_temp_cleanup_test"
