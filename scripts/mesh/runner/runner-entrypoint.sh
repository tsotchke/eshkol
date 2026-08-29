#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    echo "usage: runner-entrypoint --url URL --token TOKEN --name NAME" >&2
    exit 64
}

runner_url=""
registration_token=""
runner_name=""
while (($#)); do
    case "$1" in
        --url)
            (($# >= 2)) || usage
            runner_url="$2"
            shift 2
            ;;
        --token)
            (($# >= 2)) || usage
            registration_token="$2"
            shift 2
            ;;
        --name)
            (($# >= 2)) || usage
            runner_name="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

[[ -n "$runner_url" && -n "$registration_token" && -n "$runner_name" ]] || usage
[[ "$runner_name" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "error: runner name contains unsupported characters" >&2
    exit 64
}

if [[ ! -x /runner/run.sh ]]; then
    cp --recursive /runner-image/. /runner/
fi

cd /runner
./config.sh \
    --url "$runner_url" \
    --token "$registration_token" \
    --ephemeral \
    --unattended \
    --replace \
    --name "$runner_name" \
    --labels eshkol,linux-mesh \
    --work /work

exec ./run.sh
