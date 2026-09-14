#!/bin/sh
# Run "$@" under a raised, finite native stack limit.
#
# The parser deliberately refuses deeply nested input before it can exhaust the
# caller's stack. A Debug compiler needs more frames to parse the shipped
# aggregate stdlib than a normal login-shell stack supplies on Linux, so the
# stdlib build rule runs eshkol-run through this launcher with the same 512 MiB
# every other Eshkol test runner asks for (scripts/run_tco_tests.sh is the
# canonical site), clamped to the hard limit where that is lower.
#
# It must be a large *finite* request, never `unlimited`: an infinite
# RLIMIT_STACK puts Linux into the legacy bottom-up mmap layout, which maps
# shared libraries straight through AddressSanitizer's fixed shadow range, and
# every sanitized eshkol-run then aborts before it compiles anything ("Shadow
# memory range interleaves with an existing memory mapping").
stack_kib=524288
hard_kib=$(ulimit -Hs 2>/dev/null || echo unlimited)
case "$hard_kib" in
  ''|*[!0-9]*) ;;
  *) [ "$hard_kib" -lt "$stack_kib" ] && stack_kib="$hard_kib" ;;
esac
ulimit -s "$stack_kib" 2>/dev/null || true
exec "$@"
