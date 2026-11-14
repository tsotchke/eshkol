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
        >"$WORK/stopped-compile.log" 2>&1 &
    compiler=$!
    while kill -0 "$compiler" 2>/dev/null; do
        if [ -n "$(scratch_objects "$stopped_out")" ]; then
            kill -KILL "$compiler" 2>/dev/null || true
            break
        fi
        sleep 0.05
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
leftovers="$(compgen -G "$stopped_out.*" || true)"
test -z "$leftovers" || {
    echo "FAIL: files of a stopped compile were not reclaimed: $leftovers" >&2
    exit 1
}

# Compiles of one output at the same moment. The output path has one owner at
# a time: whichever compile holds it finishes and publishes a whole executable
# in one step; another waits for it (30 s unless
# ESHKOL_OUTPUT_CLAIM_WAIT_SECONDS says otherwise) and then takes its turn, or
# is told the output is owned and stops. The executable on disk is always one
# complete link, and nothing of any compile is left beside it.
shared_out="$WORK/shared"
mkdir -p "$WORK/reference"
"$ESHKOL_RUN" -n -o "$WORK/reference/shared" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/reference.log" 2>&1 || { cat "$WORK/reference.log" >&2; echo "FAIL: reference link" >&2; exit 1; }
reference_size="$(wc -c < "$WORK/reference/shared")"
refused_rounds=0
for round in 1 2 3 4; do
    # Rounds 1 and 2 do not wait, so an overlap ends in a refusal; rounds 3
    # and 4 use the default bound, so every compile gets its turn.
    if [ "$round" -le 2 ]; then wait_bound=0; else wait_bound=""; fi
    pids=()
    for who in a b c; do
        ESHKOL_OUTPUT_CLAIM_WAIT_SECONDS="${wait_bound:-30}" \
        "$ESHKOL_RUN" -n -o "$shared_out" "$WORK/program.esk" \
            "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
            >"$WORK/shared-$who.log" 2>&1 &
        pids+=("$!")
    done
    succeeded=0
    refused=0
    index=0
    for who in a b c; do
        if wait "${pids[$index]}"; then
            succeeded=$((succeeded + 1))
        elif grep -q "Another compile is producing" "$WORK/shared-$who.log"; then
            refused=$((refused + 1))
        else
            cat "$WORK/shared-$who.log" >&2
            echo "FAIL: a concurrent compile ended for a reason other than the output being owned" >&2
            exit 1
        fi
        index=$((index + 1))
    done
    [ "$succeeded" -ge 1 ] || { echo "FAIL: no concurrent compile produced the output" >&2; exit 1; }
    if [ -z "$wait_bound" ] && [ "$succeeded" -ne 3 ]; then
        echo "FAIL: with the default wait, $succeeded of 3 compiles of one output went through" >&2
        exit 1
    fi
    [ "$refused" -gt 0 ] && refused_rounds=$((refused_rounds + 1))
    test -x "$shared_out" || { echo "FAIL: concurrent compiles left no executable" >&2; exit 1; }
    [ "$(wc -c < "$shared_out")" = "$reference_size" ] || {
        echo "FAIL: executable from concurrent compiles is not one complete link" >&2; exit 1; }
    "$shared_out" || { echo "FAIL: executable from concurrent compiles did not run" >&2; exit 1; }
    leftovers="$(compgen -G "$shared_out.*" || true)"
    test -z "$leftovers" || { echo "FAIL: concurrent compiles left files beside the output: $leftovers" >&2; exit 1; }
done
echo "no-wait rounds in which a compile was told the output is owned: $refused_rounds of 2"

# A second compile started while the first holds the output waits and then
# succeeds: both exit 0 and the output is a complete link.
ESHKOL_OUTPUT_CLAIM_WAIT_SECONDS=120 "$ESHKOL_RUN" -n -o "$shared_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/first.log" 2>&1 &
first=$!
for _ in $(seq 1 400); do
    [ -e "$shared_out.eshkol-link.lock" ] && break
    kill -0 "$first" 2>/dev/null || break
    sleep 0.05
done
ESHKOL_OUTPUT_CLAIM_WAIT_SECONDS=120 "$ESHKOL_RUN" -n -o "$shared_out" "$WORK/program.esk" \
    "$WORK/eshkol_named/stdlib.o" "$WORK/eshkol_named/my_eshkol_helpers.o" \
    >"$WORK/second.log" 2>&1 || { cat "$WORK/second.log" >&2; echo "FAIL: second compile did not succeed after the first" >&2; exit 1; }
wait "$first" || { cat "$WORK/first.log" >&2; echo "FAIL: first compile did not succeed" >&2; exit 1; }
[ "$(wc -c < "$shared_out")" = "$reference_size" ] || { echo "FAIL: output after two sequential owners is not one complete link" >&2; exit 1; }
"$shared_out" || { echo "FAIL: output after two sequential owners did not run" >&2; exit 1; }
leftovers="$(compgen -G "$shared_out.*" || true)"
test -z "$leftovers" || { echo "FAIL: sequential owners left files beside the output: $leftovers" >&2; exit 1; }

echo "PASS: aot_owned_temp_cleanup_test"
