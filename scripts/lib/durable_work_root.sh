#!/usr/bin/env bash
# Optional deterministic evidence roots for release gates.
#
# A caller that sets ESHKOL_DURABLE_WORK_ROOT opts out of ephemeral scratch
# space.  Each gate must claim a unique direct child through
# eshkol_durable_prepare_dir; claiming an existing child is an error so a
# release run cannot accidentally consume stale evidence.  The caller owns the
# root itself and may use one root for a complete battery.

eshkol_durable_enabled() {
    [ -n "${ESHKOL_DURABLE_WORK_ROOT:-}" ]
}

eshkol_durable_validate_root() {
    local root="${ESHKOL_DURABLE_WORK_ROOT:-}" parent
    case "$root" in
        /*) ;;
        *) echo "ESHKOL_DURABLE_WORK_ROOT must be an absolute path" >&2; return 2 ;;
    esac
    if [ "$root" = / ] || [ -L "$root" ]; then
        echo "ESHKOL_DURABLE_WORK_ROOT must name a non-symlink directory other than /" >&2
        return 2
    fi
    if [ -e "$root" ] && [ ! -d "$root" ]; then
        echo "ESHKOL_DURABLE_WORK_ROOT is not a directory: $root" >&2
        return 2
    fi
    if [ ! -e "$root" ]; then
        parent="$(dirname "$root")"
        if [ ! -d "$parent" ] || [ -L "$parent" ]; then
            echo "ESHKOL_DURABLE_WORK_ROOT parent must be an existing, non-symlink directory: $parent" >&2
            return 2
        fi
        mkdir "$root" || return $?
    fi
}

eshkol_durable_prepare_dir() { # gate-name
    local gate="$1" target
    eshkol_durable_validate_root || return $?
    case "$gate" in
        *[!A-Za-z0-9._-]*|'') echo "invalid durable gate name: $gate" >&2; return 2 ;;
    esac
    target="${ESHKOL_DURABLE_WORK_ROOT}/${gate}"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "durable evidence target already exists: $target" >&2
        return 2
    fi
    mkdir "$target" || return $?
    printf '%s\n' "$target"
}

eshkol_durable_file() { # claimed-directory basename
    local dir="$1" name="$2" target
    case "$name" in
        *[!A-Za-z0-9._-]*|'') echo "invalid durable evidence file name: $name" >&2; return 2 ;;
    esac
    target="$dir/$name"
    if [ -e "$target" ] || [ -L "$target" ]; then
        echo "durable evidence file already exists: $target" >&2
        return 2
    fi
    printf '%s\n' "$target"
}
