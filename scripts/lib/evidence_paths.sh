#!/usr/bin/env bash
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#
# One rule for every evidence path a gate accepts from its environment.
#
# A gate that reads TRACE_DIR (or any other evidence location) from the
# environment must make it absolute before first use. Evidence producers change
# directory as a matter of course: `ctest --test-dir build --output-junit P`
# resolves a relative P inside build/, a harness that cd's into a scratch tree
# resolves it there, and so on. A relative evidence path therefore names a
# different file for the tool that writes the evidence and for the recorder
# that reads it, and a run in which every required test passed is graded as if
# none had executed.
#
# The meaning of a relative value is fixed here, once: it is relative to the
# repository root, which is what every workflow and wrapper that passes
# `scripts/icc_traces` intends. The path does not have to exist yet and
# symlinks are not resolved.
#
# Bash 3.2 compatible (the macOS system shell).

eshkol_abs_path() { # path base-dir -> absolute path on stdout
    local path="${1-}" base="${2-}"
    if [ -z "$path" ]; then
        echo "eshkol_abs_path: empty path" >&2
        return 2
    fi
    case "$path" in
        /*) printf '%s\n' "$path"; return 0 ;;
    esac
    case "$base" in
        /*) ;;
        *) echo "eshkol_abs_path: base directory must be absolute: '$base'" >&2; return 2 ;;
    esac
    path="${path#./}"
    printf '%s/%s\n' "${base%/}" "$path"
}

eshkol_evidence_abs_var() { # VARIABLE-NAME base-dir : rewrite the variable in place
    local name="${1-}" base="${2-}" value
    case "$name" in
        ''|*[!A-Za-z0-9_]*|[0-9]*)
            echo "eshkol_evidence_abs_var: invalid variable name: '$name'" >&2
            return 2 ;;
    esac
    value="${!name-}"
    value="$(eshkol_abs_path "$value" "$base")" || return $?
    printf -v "$name" '%s' "$value"
}
