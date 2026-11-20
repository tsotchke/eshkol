#!/bin/sh
# Exercise a live client that reaches the peer after the first ten seconds.
set -eu
sleep 11
exec "${TLS_TEST_RUNNER:?}" "$@"
