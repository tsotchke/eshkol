#!/usr/bin/env bash
set -euo pipefail

ESHKOL_RUN="${1:?usage: aot_owned_temp_cleanup_test.sh /path/to/eshkol-run /path/to/stdlib.o}"
# The packaged stdlib object of the build under test. The driver admits an
# object named stdlib.o only when it carries this build's identity, so the
# caller-held copy below is a real packaged object, not a stand-in.
PACKAGED_STDLIB="${2:?usage: aot_owned_temp_cleanup_test.sh /path/to/eshkol-run /path/to/stdlib.o}"
test -f "$PACKAGED_STDLIB" || { echo "FAIL: packaged stdlib object not found: $PACKAGED_STDLIB" >&2; exit 1; }
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/eshkol_named"

# The driver's scratch object for <output> is <output>.<pid>.<n>.tmp.o.
scratch_objects() { compgen -G "$1.*.tmp.o" || true; }

cat > "$WORK/program.esk" <<'EOF'
(define (main) 0)
EOF
cat > "$WORK/helpers.c" <<'EOF'
void preserved_helper_object(void) {}
EOF
cp "$PACKAGED_STDLIB" "$WORK/eshkol_named/stdlib.o"
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
cmp -s "$PACKAGED_STDLIB" "$WORK/eshkol_named/stdlib.o" || { echo "FAIL: cleanup deleted or changed caller stdlib.o" >&2; exit 1; }
test -f "$WORK/eshkol_named/my_eshkol_helpers.o" || { echo "FAIL: cleanup deleted caller helper object" >&2; exit 1; }
test -z "$(scratch_objects "$out")" || { echo "FAIL: successful link left its scratch object" >&2; exit 1; }

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
cmp -s "$PACKAGED_STDLIB" "$WORK/eshkol_named/stdlib.o" || { echo "FAIL: failed-link cleanup deleted or changed caller stdlib.o" >&2; exit 1; }
test -f "$WORK/eshkol_named/my_eshkol_helpers.o" || { echo "FAIL: failed-link cleanup deleted caller helper object" >&2; exit 1; }
test -z "$(scratch_objects "$failed_out")" || { echo "FAIL: failed link left its scratch object" >&2; exit 1; }

# A caller's file beside the output is never the driver's, whatever its name:
# the link goes through and the file keeps its bytes.
cc -c "$WORK/helpers.c" -o "$WORK/eshkol_named/my_eshkol_helpers.o"
existing_out="$WORK/existing"
printf 'caller-owned scratch-name contents\n' > "$existing_out.tmp.o"
cp "$existing_out.tmp.o" "$WORK/preexisting.expected"
"$ESHKOL_RUN" -n -o "$existing_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/existing.log" 2>&1 || {
        cat "$WORK/existing.log" >&2
        echo "FAIL: a caller's file beside the output held up the link" >&2
        exit 1
    }
test -x "$existing_out" || { echo "FAIL: linked executable missing beside a caller's file" >&2; exit 1; }
cmp "$existing_out.tmp.o" "$WORK/preexisting.expected" || {
    echo "FAIL: compiler changed or removed a pre-existing caller object" >&2
    exit 1
}
test -z "$(scratch_objects "$existing_out")" || { echo "FAIL: link beside a caller's file left its scratch object" >&2; exit 1; }

# A compile stopped mid-run cannot clean up after itself. The scratch object
# it leaves carries its process id, so the next compile of the same output
# goes through, and reclaims the leftover because that process is gone.
stopped_out="$WORK/stopped"
abandoned=""
for attempt in $(seq 1 25); do
    "$ESHKOL_RUN" -n -o "$stopped_out" "$WORK/program.esk" \
        "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
        >"$WORK/stopped.log" 2>&1 &
    compiler=$!
    while kill -0 "$compiler" 2>/dev/null; do
        if [ -n "$(scratch_objects "$stopped_out")" ]; then
            kill -KILL "$compiler" 2>/dev/null || true
            break
        fi
    done
    wait "$compiler" 2>/dev/null || true
    abandoned="$(scratch_objects "$stopped_out")"
    [ -n "$abandoned" ] && break
done
test -n "$abandoned" || { echo "FAIL: could not stop a compile while it held its scratch object" >&2; exit 1; }
# The stopped driver's linker child, if it had started, still names the
# scratch object on its command line; let it end before reusing the output.
for _ in $(seq 1 300); do
    pgrep -f -- "$(basename "$abandoned")" >/dev/null 2>&1 || break
    sleep 0.2
done
rm -f "$stopped_out"
"$ESHKOL_RUN" -n -o "$stopped_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/rerun.log" 2>&1 || {
        cat "$WORK/rerun.log" >&2
        echo "FAIL: compile after a stopped compile did not go through" >&2
        exit 1
    }
test -x "$stopped_out" || { echo "FAIL: compile after a stopped compile produced no executable" >&2; exit 1; }
"$stopped_out" || { echo "FAIL: executable built after a stopped compile did not run" >&2; exit 1; }
test -z "$(scratch_objects "$stopped_out")" || {
    echo "FAIL: scratch object of a stopped compile was not reclaimed: $(scratch_objects "$stopped_out")" >&2
    exit 1
}

echo "PASS: aot_owned_temp_cleanup_test"
