#!/bin/sh
set -eu
cd "$(dirname "$0")"
python3 ../heat-p0/replay.py --intervals 20 40
python3 native_access.py
