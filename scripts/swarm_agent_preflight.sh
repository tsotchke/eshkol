#!/usr/bin/env sh
# Guardrail for tsotchke-chan / multi-agent Eshkol work.
#
# Run this before an agent starts a task, especially from a portfolio mirror or
# per-task worktree. It prevents the two failure modes that lose work:
# stale mirrors and destructive git operations before untracked work is pinned.

set -eu

usage() {
    cat <<'USAGE'
Usage: scripts/swarm_agent_preflight.sh [--task ESH-NNNN] [--sync] [--snapshot]

Checks:
  - running inside a git checkout with .swarm task metadata available
  - known tsotchke portfolio/run mirrors are synced to origin/master
  - optional snapshot commit pins current dirty/untracked state

Options:
  --task ESH-NNNN  Verify the task spec exists.
  --sync           In tsotchke mirrors, fast-forward to origin/master if clean.
  --snapshot       Commit all current tracked/untracked changes before checks.

Use --snapshot before any destructive git operation. Do not rely on git stash
for preservation when ignored or untracked files are part of the task.
USAGE
}

task_id=""
do_sync=0
do_snapshot=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --task)
            [ "$#" -ge 2 ] || { usage >&2; exit 2; }
            task_id="$2"
            shift 2
            ;;
        --sync)
            do_sync=1
            shift
            ;;
        --snapshot)
            do_snapshot=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
done

icc_tool=${ICC_TOOL:-"$HOME/Desktop/infinite_context_coder/bin/icc"}
icc_repo=${ICC_REPO:-eshkol_lang}
if [ -x "$icc_tool" ] && [ "${ICC_PREFLIGHT_LOCAL_ONLY:-0}" != "1" ]; then
    icc_args="agent-preflight --repo $icc_repo --require-swarm"
    if [ -n "$task_id" ]; then
        icc_args="$icc_args --task-id $task_id --require-swarm-task"
    fi
    if [ "$do_sync" -eq 1 ]; then
        icc_args="$icc_args --fetch --sync-mirror"
    fi
    if [ "$do_snapshot" -eq 1 ]; then
        icc_args="$icc_args --snapshot"
    fi
    # shellcheck disable=SC2086
    exec "$icc_tool" $icc_args
fi

repo_root=$(/usr/bin/git rev-parse --show-toplevel 2>/dev/null) || {
    echo "FAIL: not inside a git checkout" >&2
    exit 1
}
cd "$repo_root"

if [ -n "$task_id" ] && [ ! -f ".swarm/tasks/${task_id}.json" ]; then
    echo "FAIL: missing .swarm/tasks/${task_id}.json in $repo_root" >&2
    echo "      This checkout is stale or not the Eshkol swarm tree." >&2
    exit 1
fi

if [ "$do_snapshot" -eq 1 ]; then
    if [ -n "$(/usr/bin/git status --porcelain=v1)" ]; then
        /usr/bin/git add -A
        ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        label=${task_id:-manual}
        /usr/bin/git commit -m "Agent safety snapshot before ${label} ${ts}"
        echo "OK: snapshot commit created before continuing"
    else
        echo "OK: no local changes to snapshot"
    fi
fi

case "$repo_root" in
    */.tsotchke/portfolio/*|*/.tsotchke/runs/*)
        origin_url=$(/usr/bin/git config --get remote.origin.url || true)
        if [ -z "$origin_url" ]; then
            echo "FAIL: tsotchke mirror has no origin remote; cannot prove freshness" >&2
            exit 1
        fi

        if [ "$do_sync" -eq 1 ]; then
            if [ -n "$(/usr/bin/git status --porcelain=v1)" ]; then
                echo "FAIL: mirror is dirty; snapshot or commit before --sync" >&2
                exit 1
            fi
            /usr/bin/git fetch origin master
            /usr/bin/git merge --ff-only origin/master
        else
            /usr/bin/git fetch origin master
        fi

        head=$(/usr/bin/git rev-parse HEAD)
        origin_head=$(/usr/bin/git rev-parse origin/master)
        if [ "$head" != "$origin_head" ]; then
            echo "FAIL: tsotchke mirror is not at origin/master" >&2
            echo "      HEAD=$head" >&2
            echo "      origin/master=$origin_head" >&2
            echo "      Re-run with --sync if the tree is clean." >&2
            exit 1
        fi
        ;;
esac

if [ ! -d .swarm ] || [ ! -f .swarm/README.md ]; then
    echo "FAIL: .swarm metadata missing; this checkout cannot run v1.3 swarm tasks" >&2
    exit 1
fi

if [ -d deps/stb ]; then
    echo "FAIL: deps/stb exists; v1.3 native image I/O must use platform/system codecs" >&2
    exit 1
fi

echo "OK: swarm preflight passed for ${task_id:-repo}"
