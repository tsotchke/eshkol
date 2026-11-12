#!/usr/bin/env bash
# Compatibility entry point; the Python driver owns deadlines and peer cleanup.
set -euo pipefail
if [ "$#" -ne 7 ]; then
    echo "usage: $0 runner mode scenario work-root source peer fixture-library" >&2
    exit 2
fi
exec python3 "$(dirname "$0")/run_unix_socket_stream_pending.py" \
    --runner "$1" --mode "$2" --scenario "$3" --work-root "$4" \
    --source "$5" --peer "$6" --fixture-library "$7"
