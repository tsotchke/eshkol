#!/usr/bin/env bash
# Single, reusable definition of this repo's CI network-retry policy.
#
# Motivation: several required lanes (linux-x64-cuda, linux-arm64-cuda,
# linux-arm64-lite among them) install packages from apt.llvm.org and the
# NVIDIA CUDA apt repository. Those repos have transient outages that have
# nothing to do with this repo's code: `apt-get install` fails with exit
# code 100 (APT's generic "an error occurred" code) when a mirror is
# momentarily unreachable, on a build that would succeed a minute later.
# Before this script existed, most of those calls ran once with no retry at
# all (the raw `apt-get install -y ...` in "Install Dependencies (Linux)"),
# and the two call sites that DID retry (`apt-get update` in the two
# near-identical "Add LLVM Repository (Linux)" steps) each hand-rolled a
# DIFFERENT retry loop, so the policy existed in two places and could drift.
#
# This script is that policy's one definition: every network-dependent
# setup command in ci.yml (apt-get update/install, the apt.llvm.org and
# CUDA keyring fetches) is called through this wrapper instead of pasting a
# retry loop at each site. A command that fails on every attempt still
# fails the step and the job — this absorbs an intermittent repository
# blip, it does not turn a real failure into a pass, and it never uses
# continue-on-error.
#
# Usage
#   scripts/ci_retry.sh [--attempts N] [--base-delay SECONDS] -- <command...>
#
#   --attempts     Maximum attempts (default 3).
#   --base-delay   Delay in seconds before the 2nd attempt; doubles after
#                  every subsequent failure (exponential backoff). Default 10.
#
# The command is executed via its own argv (never re-parsed by a shell), so
# ordinary commands and their arguments need no extra quoting. A command
# that itself streams through a pipe (e.g. `wget ... | sudo tee ...`) is
# NOT supported directly — write the fetch to a file and consume the file
# in a separate, non-retried step, because a shell pipeline's exit status
# is not this script's to observe without `bash -c` re-quoting.
#
# Exit status: the last attempt's exit status (0 only if some attempt
# succeeded).
#
# scripts/ci_retry.sh --self-test proves the wrapper both retries and
# eventually gives up: a command that fails twice then succeeds must
# succeed within 3 attempts, and a command that always fails must exit
# nonzero after exactly N attempts, no more and no fewer.
set -uo pipefail

self_test() {
  local workdir counter status all_ok
  workdir="$(mktemp -d)"
  trap 'rm -rf "$workdir"' RETURN
  all_ok=0

  echo "ci_retry.sh self-test:"

  # Case 1: fails on attempts 1 and 2, succeeds on attempt 3 -> must PASS.
  counter="$workdir/flaky_counter"
  printf '0' > "$counter"
  cat > "$workdir/flaky.sh" <<'EOS'
#!/usr/bin/env bash
n=$(cat "$1")
n=$((n + 1))
printf '%s' "$n" > "$1"
[ "$n" -ge 3 ]
EOS
  chmod +x "$workdir/flaky.sh"
  if "$0" --attempts 5 --base-delay 0 -- "$workdir/flaky.sh" "$counter"; then
    got="$(cat "$counter")"
    if [ "$got" = "3" ]; then
      echo "  [OK] flaky command (fails x2, succeeds on 3rd) succeeds within 5 attempts, ran exactly 3 times"
    else
      echo "  [GATE IS BROKEN] flaky command succeeded but ran $got times, expected exactly 3"
      all_ok=1
    fi
  else
    echo "  [GATE IS BROKEN] flaky command (fails x2, succeeds on 3rd) did not succeed within 5 attempts"
    all_ok=1
  fi

  # Case 2: always fails -> must exit nonzero after EXACTLY --attempts tries.
  counter="$workdir/always_fail_counter"
  printf '0' > "$counter"
  cat > "$workdir/always_fail.sh" <<'EOS'
#!/usr/bin/env bash
n=$(cat "$1")
n=$((n + 1))
printf '%s' "$n" > "$1"
exit 7
EOS
  chmod +x "$workdir/always_fail.sh"
  if "$0" --attempts 3 --base-delay 0 -- "$workdir/always_fail.sh" "$counter"; then
    echo "  [GATE IS BROKEN] always-failing command reported success"
    all_ok=1
  else
    status=$?
    got="$(cat "$counter")"
    if [ "$status" = "7" ] && [ "$got" = "3" ]; then
      echo "  [OK] always-failing command exits with its own status (7) after exactly 3 attempts"
    else
      echo "  [GATE IS BROKEN] always-failing command: exit=$status (want 7), attempts=$got (want 3)"
      all_ok=1
    fi
  fi

  if [ "$all_ok" -eq 0 ]; then
    echo "self-test: PASS — retries a flaky command to success and gives up on a hard failure after exactly N attempts"
  else
    echo "self-test: FAIL — the retry wrapper did not behave as specified" >&2
  fi
  return "$all_ok"
}

if [ "${1:-}" = "--self-test" ]; then
  self_test
  exit $?
fi

attempts=3
base_delay=10

while [ $# -gt 0 ]; do
  case "$1" in
    --attempts)
      attempts="$2"
      shift 2
      ;;
    --base-delay)
      base_delay="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "ci_retry.sh: unexpected argument before '--': $1" >&2
      exit 64
      ;;
  esac
done

if [ $# -eq 0 ]; then
  echo "ci_retry.sh: no command given (usage: ci_retry.sh [--attempts N] [--base-delay SECONDS] -- <command...>)" >&2
  exit 64
fi

case "$attempts" in
  ''|*[!0-9]*) echo "ci_retry.sh: --attempts must be a positive integer, got '$attempts'" >&2; exit 64 ;;
esac
case "$base_delay" in
  ''|*[!0-9]*) echo "ci_retry.sh: --base-delay must be a non-negative integer, got '$base_delay'" >&2; exit 64 ;;
esac
if [ "$attempts" -lt 1 ]; then
  echo "ci_retry.sh: --attempts must be >= 1, got '$attempts'" >&2
  exit 64
fi

delay="$base_delay"
attempt=1
while :; do
  # Deliberately NOT `if "$@"; then exit 0; fi; status=$?` — when the
  # condition of an `if` with no `else` is false, bash/POSIX define the
  # compound's own exit status as ZERO regardless of the condition's exit
  # status, so a `status=$?` read after `fi` would always see 0 and every
  # failure would be misreported as exit 0. Capturing status immediately
  # after the command itself, before any other command runs, is required.
  "$@"
  status=$?
  if [ "$status" -eq 0 ]; then
    exit 0
  fi
  if [ "$attempt" -ge "$attempts" ]; then
    echo "ci_retry.sh: '$*' failed on attempt $attempt/$attempts (exit $status) — giving up" >&2
    exit "$status"
  fi
  echo "ci_retry.sh: '$*' failed on attempt $attempt/$attempts (exit $status) — retrying in ${delay}s" >&2
  sleep "$delay"
  delay=$((delay * 2))
  attempt=$((attempt + 1))
done
