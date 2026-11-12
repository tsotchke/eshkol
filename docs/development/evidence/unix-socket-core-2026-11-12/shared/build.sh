#!/usr/bin/env bash
set -euo pipefail
cd /home/gabe/Documents/ChatGPT/eshkol-sockets
for mode in normal sanitized; do
  flags=()
  if [[ $mode == sanitized ]]; then flags=(-fsanitize=address,undefined -fno-omit-frame-pointer); fi
  clang -std=c11 -D_GNU_SOURCE -DESHKOL_RESOURCE_TESTING -Wall -Wextra -Werror -Iinc -Ilib/core -pthread "${flags[@]}" -c tests/core/resource_explicit_loop_test.c -o /tmp/eshkol-explicit-loop/test.o
  for part in resource_lease event_loop event_loop_epoll; do
    clang -std=c11 -D_GNU_SOURCE -DESHKOL_RESOURCE_TESTING -Wall -Wextra -Werror -Iinc -Ilib/core -pthread "${flags[@]}" -c lib/core/$part.c -o /tmp/eshkol-explicit-loop/$part.o
  done
  clang++ -std=c++17 -Wall -Wextra -Werror -Iinc -pthread "${flags[@]}" -c lib/core/runtime_handle.cpp -o /tmp/eshkol-explicit-loop/runtime_handle.o
  clang++ -pthread "${flags[@]}" /tmp/eshkol-explicit-loop/test.o /tmp/eshkol-explicit-loop/resource_lease.o /tmp/eshkol-explicit-loop/event_loop.o /tmp/eshkol-explicit-loop/event_loop_epoll.o /tmp/eshkol-explicit-loop/runtime_handle.o -o /tmp/eshkol-explicit-loop/test-$mode
  ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 /tmp/eshkol-explicit-loop/test-$mode | tee /tmp/eshkol-explicit-loop/$mode.log
done
