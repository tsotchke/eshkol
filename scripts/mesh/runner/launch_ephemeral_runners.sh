#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

usage() {
    echo "usage: $0 N" >&2
    exit 64
}

is_safe_name() {
    [[ "$1" =~ ^[A-Za-z0-9._-]+$ ]]
}

(( $# == 1 )) || usage
slots="$1"
[[ "$slots" =~ ^[1-9][0-9]*$ ]] || usage

: "${ESHKOL_RUNNER_TOKEN_FILE:?ESHKOL_RUNNER_TOKEN_FILE must name the host-only token file}"
: "${ESHKOL_RUNNER_REPOSITORY:?ESHKOL_RUNNER_REPOSITORY must be owner/repository}"
: "${ESHKOL_RUNNER_URL:?ESHKOL_RUNNER_URL must be the repository web URL}"
: "${ESHKOL_RUNNER_API_URL:?ESHKOL_RUNNER_API_URL must be the GitHub API base URL}"
: "${ESHKOL_FETCHCONTENT_CACHE:?ESHKOL_FETCHCONTENT_CACHE must name the shared FetchContent cache directory}"
: "${ESHKOL_RUNNER_ENV_FILE:?ESHKOL_RUNNER_ENV_FILE must name the host-only runner env file}"

image="${ESHKOL_RUNNER_IMAGE:-eshkol-ci-runner:local}"
prefix="${ESHKOL_RUNNER_PREFIX:-eshkol-mesh}"
cpus="${ESHKOL_RUNNER_CPUS:-16}"
memory="${ESHKOL_RUNNER_MEMORY:-32g}"
pids_limit="${ESHKOL_RUNNER_PIDS_LIMIT:-4096}"
log_file="${ESHKOL_RUNNER_LOG_FILE:-./eshkol-mesh-runners.log}"

is_safe_name "$prefix" || { echo "error: ESHKOL_RUNNER_PREFIX is unsafe" >&2; exit 64; }
[[ -f "$ESHKOL_RUNNER_TOKEN_FILE" && -r "$ESHKOL_RUNNER_TOKEN_FILE" ]] || {
    echo "error: token file is not a readable regular file" >&2
    exit 1
}
[[ -f "$ESHKOL_FETCHCONTENT_CACHE" ]] && {
    echo "error: FetchContent cache must be a directory" >&2
    exit 1
}
[[ -d "$ESHKOL_FETCHCONTENT_CACHE" ]] || { echo "error: FetchContent cache directory is missing" >&2; exit 1; }
[[ -f "$ESHKOL_RUNNER_ENV_FILE" ]] || { echo "error: runner env file is missing" >&2; exit 1; }

token_mode="$(stat -c '%a' "$ESHKOL_RUNNER_TOKEN_FILE" 2>/dev/null || stat -f '%Lp' "$ESHKOL_RUNNER_TOKEN_FILE")"
[[ "$token_mode" == 600 ]] || {
    echo "error: ESHKOL_RUNNER_TOKEN_FILE must have mode 600" >&2
    exit 1
}

env_mode="$(stat -c '%a' "$ESHKOL_RUNNER_ENV_FILE" 2>/dev/null || stat -f '%Lp' "$ESHKOL_RUNNER_ENV_FILE")"
[[ "$env_mode" == 600 ]] || {
    echo "error: ESHKOL_RUNNER_ENV_FILE must have mode 600" >&2
    exit 1
}

[[ "$ESHKOL_RUNNER_REPOSITORY" =~ ^[^/[:space:]]+/[^/[:space:]]+$ ]] || {
    echo "error: ESHKOL_RUNNER_REPOSITORY must be owner/repository" >&2
    exit 64
}
[[ "$ESHKOL_RUNNER_URL" == https://* ]] || {
    echo "error: ESHKOL_RUNNER_URL must use https" >&2
    exit 64
}
[[ "$ESHKOL_RUNNER_API_URL" == https://* ]] || {
    echo "error: ESHKOL_RUNNER_API_URL must use https" >&2
    exit 64
}

validate_runner_env() {
    local line key found=0
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        [[ "$line" == *=* ]] || {
            echo "error: runner env file contains a line without KEY=VALUE" >&2
            return 1
        }
        key="${line%%=*}"
        [[ "$key" == FETCHCONTENT_BASE_DIR ]] || {
            echo "error: runner env file may contain only FETCHCONTENT_BASE_DIR" >&2
            return 1
        }
        [[ "$line" == FETCHCONTENT_BASE_DIR=/deps ]] || {
            echo "error: runner env file must set FETCHCONTENT_BASE_DIR=/deps" >&2
            return 1
        }
        found=1
    done < "$ESHKOL_RUNNER_ENV_FILE"
    (( found == 1 )) || {
        echo "error: runner env file must set FETCHCONTENT_BASE_DIR=/deps" >&2
        return 1
    }
}

validate_runner_env || exit 1

mkdir -p "$(dirname "$log_file")"
touch "$log_file"
chmod 600 "$log_file"

log() {
    printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" >> "$log_file"
}

redact_stream() {
    local secret="$1" line
    while IFS= read -r line; do
        line="${line//"$secret"/[REDACTED]}"
        printf '%s\n' "$line" >> "$log_file"
    done
}

mint_token() {
    local api_token response
    api_token="$(< "$ESHKOL_RUNNER_TOKEN_FILE")"
    [[ -n "$api_token" && "$api_token" != *$'\n'* ]] || {
        echo "error: token file is empty or contains a newline" >&2
        return 1
    }
    response="$(curl --fail --silent --show-error --location \
        --connect-timeout 20 --max-time 60 --retry 3 --retry-all-errors \
        --header 'Accept: application/vnd.github+json' \
        --header 'X-GitHub-Api-Version: 2022-11-28' \
        --header "Authorization: Bearer $api_token" \
        --request POST \
        "${ESHKOL_RUNNER_API_URL%/}/repos/${ESHKOL_RUNNER_REPOSITORY}/actions/runners/registration-token")"
    jq -er '.token | select(type == "string" and length > 0)' <<< "$response"
}

ensure_no_stale_container() {
    local container_name="$1"
    if docker container inspect "$container_name" >/dev/null 2>&1; then
        echo "error: Docker container $container_name already exists; refusing to remove an unknown container" >&2
        exit 1
    fi
}

run_slot() {
    local slot="$1" container_name volume_name registration_token rc
    container_name="${prefix}-${slot}"
    volume_name="${prefix}-work-${slot}-$$"
    ensure_no_stale_container "$container_name"

    while :; do
        registration_token="$(mint_token)" || {
            log "slot=$slot token mint failed; retrying in 15s"
            sleep 15
            continue
        }
        docker volume create "$volume_name" >/dev/null

        log "slot=$slot starting ephemeral container"
        set +e
        docker run --rm --init \
            --cpus "$cpus" --memory "$memory" --pids-limit "$pids_limit" \
            --network bridge \
            --read-only --security-opt no-new-privileges:true --cap-drop ALL \
            --tmpfs /tmp:rw,nosuid,nodev \
            --tmpfs /run:rw,nosuid,nodev,mode=0755 \
            --tmpfs /home/runner:rw,nosuid,nodev,uid=10001,gid=10001,mode=0700 \
            --tmpfs /runner:rw,nosuid,nodev,exec,uid=10001,gid=10001,mode=0755 \
            --mount "type=volume,source=$volume_name,target=/work" \
            --mount "type=bind,source=$ESHKOL_FETCHCONTENT_CACHE,target=/deps,readonly" \
            --env-file "$ESHKOL_RUNNER_ENV_FILE" \
            --env ESHKOL_RUNNER_CONTAINER=1 \
            --env ESHKOL_RUNNER_EPHEMERAL=1 \
            --workdir /runner \
            --name "$container_name" \
            "$image" --url "$ESHKOL_RUNNER_URL" --token "$registration_token" --name "$container_name" 2>&1 \
            | redact_stream "$registration_token"
        rc="${PIPESTATUS[0]}"
        set -e
        docker volume rm "$volume_name" >/dev/null 2>&1 || true
        log "slot=$slot container exit=$rc; requesting a fresh registration token"
        sleep 2
    done
}

worker_pids=()
container_names=()
volume_names=()
cleanup() {
    local pid container volume
    trap - TERM INT EXIT
    for container in "${container_names[@]}"; do
        docker stop --time 10 "$container" >/dev/null 2>&1 || true
    done
    for pid in "${worker_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait || true
    for volume in "${volume_names[@]}"; do
        docker volume rm "$volume" >/dev/null 2>&1 || true
    done
}
trap cleanup TERM INT EXIT

for ((slot = 1; slot <= slots; slot++)); do
    container_names+=("${prefix}-${slot}")
    volume_names+=("${prefix}-work-${slot}-$$")
    run_slot "$slot" &
    worker_pids+=("$!")
done

cat >&2 <<'EOF'
Launcher is running. The token is read only on the host and redacted from the log.
For restart-on-login, install the supplied user unit and run exactly:
  systemctl --user daemon-reload
  systemctl --user enable --now eshkol-mesh-runner.service
EOF

wait
