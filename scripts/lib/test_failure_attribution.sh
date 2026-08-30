#!/usr/bin/env bash
# Extract failure records only when the output names the failing test.
#
# A suite normally prints its test name and result on one line. A child that
# crashes can split that line because the shell prints the crash before the
# runner prints its exit status. That is the only case where the immediately
# preceding test header may be reused. Ordinary assertion text is never
# attributed to a previous test; named failure-list entries are used instead.

eshkol_extract_failure_records() {
    local suite_name="$1"
    local clean_file="$2"
    local split_crash_pattern='^[[:space:]]*RUNTIME FAIL[[:space:]]+\(exit[[:space:]]+[0-9]+\)[[:space:]]*$'
    local line line_esk fail_type test_prefix suffix candidate
    local pending_test=""
    local failure_list_type=""
    local seen_identities="|"

    emit_record() {
        local identity="$1"
        local kind="$2"
        local identity_key

        [ -n "$identity" ] || return
        identity_key="${identity//[^A-Za-z0-9_.\/: -]/_}"
        case "$seen_identities" in
            *"|$identity_key|"*) return ;;
        esac
        seen_identities="${seen_identities}${identity_key}|"
        printf '%s\t%s\t%s\n' "$suite_name" "$identity" "$kind"
    }

    failure_type() {
        local text="$1"
        if printf '%s\n' "$text" | grep -qE 'COMPILE FAIL(ED)?'; then
            printf 'COMPILE FAIL'
        elif printf '%s\n' "$text" | grep -qE 'RUNTIME FAIL'; then
            printf 'RUNTIME FAIL'
        elif printf '%s\n' "$text" | grep -qE 'RUNTIME ERROR'; then
            printf 'RUNTIME ERROR'
        elif printf '%s\n' "$text" | grep -qE 'ASSERTION FAIL'; then
            printf 'ASSERTION FAIL'
        elif printf '%s\n' "$text" | grep -qE 'TESTS FAILED'; then
            printf 'TESTS FAILED'
        elif printf '%s\n' "$text" | grep -qE 'SEGFAULT'; then
            printf 'SEGFAULT'
        elif printf '%s\n' "$text" | grep -qE '(^|[^[:alpha:]])FAIL([^[:alpha:]]|$)'; then
            printf 'FAIL'
        elif printf '%s\n' "$text" | grep -qE '^[[:space:]]*FAILED([^[:alpha:]]|$)'; then
            printf 'FAIL'
        fi
    }

    while IFS= read -r line; do
        # The summary headings are contracts: their list entries name the
        # tests that the suite counted as failed.
        case "$line" in
            *"Compile Failures"*|*"Compile failures"*)
                failure_list_type='COMPILE FAIL'; pending_test=""; continue ;;
            *"Runtime Failures"*|*"Runtime failures"*)
                failure_list_type='RUNTIME FAIL'; pending_test=""; continue ;;
            *"Runtime Errors"*|*"Runtime errors"*|*"runtime errors"*)
                failure_list_type='RUNTIME ERROR'; pending_test=""; continue ;;
            *"Tests with FAIL markers"*)
                failure_list_type='ASSERTION FAIL'; pending_test=""; continue ;;
            *"Failed Tests:"*|*"Failed tests:"*)
                failure_list_type='FAIL'; pending_test=""; continue ;;
        esac

        if [ -n "$failure_list_type" ]; then
            if [ -z "$line" ]; then
                failure_list_type=""
                continue
            fi
            candidate=$(printf '%s\n' "$line" | sed -nE 's/^[[:space:]]*[-*][[:space:]]+(.+)$/\1/p')
            if [ -n "$candidate" ]; then
                line_esk=$(printf '%s\n' "$candidate" | grep -oE '[A-Za-z0-9_./-]+\.esk(::[A-Za-z0-9_.-]+)?' | head -1)
                if [ -n "$line_esk" ]; then
                    emit_record "$line_esk" "$failure_list_type"
                else
                    # Some legacy suites list stem names rather than paths;
                    # remove only the runner's parenthesised reason suffix.
                    candidate=$(printf '%s\n' "$candidate" | sed -E 's/[[:space:]]+\([^)]*\)$//')
                    emit_record "$candidate" "$failure_list_type"
                fi
                continue
            fi
            # A non-list line ends the named-failure section. It must not be
            # interpreted using the preceding list entry or test header.
            failure_list_type=""
        fi

        # A split crash is the one safe use of a preceding test header. The
        # header must be the immediately preceding line; a bare FAIL is not.
        if printf '%s\n' "$line" | grep -qE "$split_crash_pattern"; then
            if [ -n "$pending_test" ]; then
                emit_record "$pending_test" 'RUNTIME FAIL'
            fi
            pending_test=""
            continue
        fi

        line_esk=$(printf '%s\n' "$line" | grep -oE '[A-Za-z0-9_./-]+\.esk(::[A-Za-z0-9_.-]+)?' | head -1)
        fail_type=$(failure_type "$line")

        # A result line containing a path is unambiguous only when its status
        # occurs after that path. Diagnostics may mention both a source file
        # and the word FAIL, so do not treat those as result records.
        if [ -n "$line_esk" ] && [ -n "$fail_type" ]; then
            suffix="${line#*"$line_esk"}"
            if printf '%s\n' "$suffix" | grep -qE 'COMPILE FAIL|RUNTIME FAIL|RUNTIME ERROR|ASSERTION FAIL|TESTS FAILED|SEGFAULT|(^|[^[:alpha:]])FAIL([^[:alpha:]]|$)' ||
               printf '%s\n' "$line" | grep -qE '^[[:space:]]*FAILED([^[:alpha:]]|$)'; then
                emit_record "$line_esk" "$fail_type"
                pending_test=""
                continue
            fi
        fi

        # `Testing ...` and legacy indented runners also have explicit names
        # without a .esk suffix. Only those result-shaped lines qualify. A
        # colon in a bare indented prefix is assertion prose (for example
        # `case: FAIL`), not a test name.
        if [ -n "$fail_type" ] && printf '%s\n' "$line" | grep -qE '^[[:space:]]*(Testing:?|\[[0-9]+/[0-9]+\]|[A-Za-z0-9_./-]+[[:space:]])'; then
            test_prefix=$(printf '%s\n' "$line" | sed -nE 's/^[[:space:]]*(Testing:?[[:space:]]+|\[[0-9]+\/[0-9]+\][[:space:]]+)?(.+)[[:space:]]+(COMPILE FAIL(ED)?|RUNTIME FAIL|RUNTIME ERROR|ASSERTION FAIL|TESTS FAILED|SEGFAULT|FAIL)([[:space:]:]|$).*/\2/p')
            test_prefix=$(printf '%s' "$test_prefix" | sed -E 's/[[:space:]]+$//')
            if [ -n "$test_prefix" ] && ! printf '%s\n' "$test_prefix" | grep -q ':'; then
                emit_record "$test_prefix" "$fail_type"
                pending_test=""
                continue
            fi
        fi

        # Capture only a header with no verdict for the immediately following
        # split-crash line. Any other intervening line invalidates the carry.
        if [ -z "$fail_type" ] && [ -n "$line_esk" ]; then
            pending_test="$line_esk"
        elif [ -z "$fail_type" ] && printf '%s\n' "$line" | grep -qE '^[[:space:]]*Testing:?[[:space:]]+'; then
            pending_test=$(printf '%s\n' "$line" | sed -E 's/^[[:space:]]*Testing:?[[:space:]]+(.+)[[:space:]]*$/\1/')
        else
            pending_test=""
        fi
    done < "$clean_file"
}
