#!/usr/bin/env bash
# Copyright (C) tsotchke
# SPDX-License-Identifier: MIT
#
# One rule for every script that writes a generated file (a here-doc fixture,
# a synthesized .esk program, a rewritten ledger) at a path named by a shell
# variable.
#
# A here-doc piped straight into `cat`, redirected at a target path held in
# a variable, is a write-in-place: the moment the shell opens that path for
# writing, any prior content at it is gone, and if the process is
# killed mid-heredoc (disk full, SIGKILL under memory pressure, a bad
# substitution inside the here-doc body that changes it out from under the
# writer) whatever partial bytes made it to disk are what the next reader
# sees at $TARGET. A test fixture, a benchmark input, or a synthesized gate
# program is exactly the kind of file a *different* step reads right back —
# so a half-written file there is not an inert leftover, it is silently
# wrong input to whatever runs next.
#
# The fix is the standard validated temp-file install: write to a private
# temp file that sits beside the target (so the two are on the same
# filesystem and `mv` is atomic), refuse to install anything empty or
# missing, then move it into place in one step. A reader either sees the
# old file or the complete new one, never a partial write.
#
# Bash 3.2 compatible (the macOS system shell).

# eshkol_checked_target VAR -> nothing; fails loudly if the named variable is
# unset, empty, or not an absolute path. Call this before treating the
# variable's value as an install target.
eshkol_checked_target() { # variable-name
    local name="${1-}" value
    case "$name" in
        ''|*[!A-Za-z0-9_]*|[0-9]*)
            echo "eshkol_checked_target: invalid variable name: '$name'" >&2
            return 2 ;;
    esac
    value="${!name-}"
    if [ -z "$value" ]; then
        echo "eshkol_checked_target: $name is empty" >&2
        return 2
    fi
    case "$value" in
        /*) ;;
        *)
            echo "eshkol_checked_target: $name must be an absolute path: '$value'" >&2
            return 2 ;;
    esac
}

# eshkol_install_tmp TARGET -> private temp file path on stdout
#
# Creates it beside TARGET (same directory, so the later install() is an
# atomic same-filesystem rename, not a cross-filesystem copy) with a name no
# other invocation can collide with. Use it as the destination of whatever
# here-doc or redirection used to write TARGET directly, then hand both
# paths to eshkol_install_checked to move the finished, validated content
# into place -- see any caller in scripts/ or tests/ for the three-line
# shape (claim the temp path, write it, install it).
eshkol_install_tmp() { # target-path
    local target="${1-}" dir tmp
    if [ -z "$target" ]; then
        echo "eshkol_install_tmp: empty target path" >&2
        return 2
    fi
    case "$target" in
        /*) ;;
        *)
            echo "eshkol_install_tmp: target must be an absolute path: '$target'" >&2
            return 2 ;;
    esac
    dir="$(dirname -- "$target")"
    if [ ! -d "$dir" ]; then
        echo "eshkol_install_tmp: target directory does not exist: $dir" >&2
        return 2
    fi
    tmp="$(mktemp "$target.XXXXXX" 2>/dev/null)" || {
        echo "eshkol_install_tmp: mktemp failed for $target" >&2
        return 2
    }
    printf '%s\n' "$tmp"
}

# eshkol_install_checked TMP TARGET -> validated atomic install
#
# Refuses to install an empty or missing temp file (the signature of a
# heredoc that never ran, or one whose body evaluated to nothing) and
# refuses a relative TARGET. On success TARGET holds exactly what TMP held
# and TMP no longer exists; on failure TMP is removed and TARGET is
# untouched.
eshkol_install_checked() { # tmp-path target-path
    local tmp="${1-}" target="${2-}"
    if [ -z "$tmp" ] || [ -z "$target" ]; then
        echo "eshkol_install_checked: usage: eshkol_install_checked TMP TARGET" >&2
        return 2
    fi
    case "$target" in
        /*) ;;
        *)
            echo "eshkol_install_checked: target must be an absolute path: '$target'" >&2
            rm -f -- "$tmp" 2>/dev/null
            return 2 ;;
    esac
    if [ ! -s "$tmp" ]; then
        echo "eshkol_install_checked: refusing to install empty/missing file: $tmp" >&2
        rm -f -- "$tmp" 2>/dev/null
        return 2
    fi
    if ! mv -f -- "$tmp" "$target" 2>/dev/null; then
        echo "eshkol_install_checked: install failed: $tmp -> $target" >&2
        rm -f -- "$tmp" 2>/dev/null
        return 2
    fi
}

# eshkol_require_output_file_path PATH -> nothing; fails loudly unless PATH
# is a non-empty absolute path and not a symlink.
#
# Call this immediately before a redirection writes to a variable path, so
# the write target is confirmed rather than assumed. This is the same
# contract tests/toolchain/sanitizer_build_dir_contract_test.sh established
# for its fake `cmake` output log; this is the shared version so every
# redirect site can use it instead of re-deriving the same three checks.
eshkol_require_output_file_path() { # output-path
    local output_path="${1-}"
    if [ -z "$output_path" ]; then
        echo "eshkol_require_output_file_path: output path is required" >&2
        return 2
    fi
    case "$output_path" in
        /*) ;;
        *)
            echo "eshkol_require_output_file_path: output path must be absolute: $output_path" >&2
            return 2 ;;
    esac
    if [ -L "$output_path" ]; then
        echo "eshkol_require_output_file_path: refusing symlinked output path: $output_path" >&2
        return 2
    fi
}

# eshkol_checked_rm PATH [PATH...] -> validated removal
#
# Refuses to run at all unless every argument is a non-empty absolute path
# and not "/" itself. Unlike an unguarded removal of a bare variable path,
# this turns a variable that was unexpectedly cleared, or unexpectedly
# rebound to something outside its expected scratch tree, into a loud
# failure instead of a silent no-op (empty argument) or a catastrophic
# removal (a variable that ended up holding "/" or a directory the caller
# did not intend).
# eshkol_resolve_trusted_command NAME -> resolved absolute path on stdout
#
# A bare `command -v "$X" >/dev/null 2>&1` probe is a PATH lookup whose
# result the caller trusts implicitly: whichever `$X` a caller's PATH
# happens to resolve first is what runs next, with no explicit statement
# that resolving it that way was an acceptable decision. This is that
# decision made explicit, and made a little stronger than the bare probe:
# resolve through PATH, then require the result to be a regular,
# executable file before handing it back, so a same-named directory or a
# non-executable match earlier on PATH cannot be mistaken for the real
# tool. Fails (prints nothing, returns 1) on an empty name or an
# unresolvable / non-executable / non-regular match.
eshkol_resolve_trusted_command() { # command-name
    local name="${1-}" resolved
    [ -n "$name" ] || return 1
    resolved="$(command -v -- "$name" 2>/dev/null)" || return 1
    [ -n "$resolved" ] && [ -f "$resolved" ] && [ -x "$resolved" ] || return 1
    printf '%s\n' "$resolved"
}

# eshkol_command_available NAME -> boolean only (no path on stdout), for the
# common case of a caller that just needs to branch on presence and will
# invoke the bare name afterward -- the shell's own PATH lookup resolves it
# identically at call time, so nothing is lost by not capturing the path
# here. Built on eshkol_resolve_trusted_command so both entry points apply
# the same trust decision.
eshkol_command_available() { # command-name
    eshkol_resolve_trusted_command "${1-}" >/dev/null
}

eshkol_checked_rm() { # path...
    local p
    if [ "$#" -eq 0 ]; then
        echo "eshkol_checked_rm: no paths given" >&2
        return 2
    fi
    for p in "$@"; do
        if [ -z "$p" ]; then
            echo "eshkol_checked_rm: refusing empty path argument" >&2
            return 2
        fi
        case "$p" in
            /) echo "eshkol_checked_rm: refusing to remove /" >&2; return 2 ;;
            /*) ;;
            *)
                echo "eshkol_checked_rm: refusing relative path: '$p'" >&2
                return 2 ;;
        esac
    done
    rm -f -- "$@"
}
