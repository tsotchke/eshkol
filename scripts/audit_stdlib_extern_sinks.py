#!/usr/bin/env python3
"""Classify every `(extern ...)` declaration in the Eshkol module tree.

A module shipped with Eshkol is exempt from the `ffi` capability, because its
declarations are how the standard library reaches the runtime entry points it is
built on and each of those carries its own gate. That reasoning holds only for
the declarations that name an Eshkol runtime entry point. One that names a host
symbol directly reaches the host with whatever gate that symbol happens to
carry, which for a libc function is none — so those are the ones to look at.

Reports every declaration grouped by what its real symbol is:

  runtime   an eshkol_* / qllm_* entry point, gated where it is implemented
  libc      a host symbol, reached directly
  unknown   neither, e.g. a symbol provided by a bundled library

and, for the host symbols, which sink class they touch, so the table can be
checked against the gates that exist.

    scripts/audit_stdlib_extern_sinks.py            # the table
    scripts/audit_stdlib_extern_sinks.py --check    # fail if a host sink is
                                                    # declared without a gate
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIB = ROOT / "lib"

EXTERN = re.compile(r"\(extern\s+(.*?)\)", re.S)

# Host symbols that reach a sink, by the capability that governs the class.
# Only the ones that take a NAME are listed: naming a file, a host or a program
# is what creates access, and that is what a capability governs. An operation on
# a descriptor the program already holds — read, write, fsync, flock, close —
# cannot reach anything the gated open did not already permit, so it is not a
# sink; DESCRIPTOR_ONLY below records that reading rather than leaving it
# implicit.
HOST_SINKS = {
    "file-read": {"open", "openat", "fopen", "freopen", "stat",
                  "lstat", "access", "readlink", "opendir", "realpath"},
    "file-write": {"rename", "renameat", "unlink", "unlinkat",
                   "remove", "mkdir", "mkdirat", "rmdir", "truncate",
                   "chmod", "link", "symlink", "mkstemp", "mkdtemp"},
    "subprocess": {"fork", "vfork", "execv", "execve", "execvp", "execl", "execlp",
                   "posix_spawn", "posix_spawnp", "wait", "waitpid", "kill",
                   "CreateProcessA", "CreateProcessW"},
    "shell": {"system", "popen"},
    "network": {"socket", "connect", "bind", "listen", "accept", "send", "recv",
                "sendto", "recvfrom", "getaddrinfo"},
    "ffi": {"dlopen", "dlsym", "dlclose", "LoadLibraryA", "LoadLibraryW",
            "GetProcAddress"},
}
SINK_OF = {symbol: capability
           for capability, symbols in HOST_SINKS.items()
           for symbol in symbols}

# Operations on a descriptor the program already holds. The gated open decided
# what it may reach; these cannot widen it.
DESCRIPTOR_ONLY = {"read", "pread", "write", "pwrite", "fread", "fwrite",
                   "fsync", "fdatasync", "fstat", "ftruncate", "fchmod",
                   "fileno", "flock", "fcntl", "lseek", "readdir", "mmap",
                   "munmap", "close", "fclose", "fflush", "dup", "dup2"}

# Host symbols that touch nothing a policy governs.
BENIGN = {"malloc", "free", "calloc", "realloc", "memcpy", "memmove", "memset",
          "strlen", "strcmp", "strncmp", "abs", "labs", "llabs", "exit",
          "_exit", "strtol", "strtod", "snprintf",
          "usleep", "sleep", "time", "clock_gettime", "getpid"} | DESCRIPTOR_ONLY


def declarations():
    for path in sorted(LIB.rglob("*.esk")):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in EXTERN.finditer(text):
            tokens = match.group(1).split()
            if ":real" in tokens:
                real = tokens[tokens.index(":real") + 1]
            elif len(tokens) >= 2:
                real = tokens[1]
            else:
                continue
            name = tokens[1] if len(tokens) >= 2 else real
            yield path.relative_to(ROOT), name, real


def classify(real: str) -> str:
    if real.startswith("eshkol_") or real.startswith("qllm_"):
        return "runtime"
    if real in BENIGN or real in SINK_OF:
        return "libc"
    return "unknown"


def module_asks_for_a_capability(path: Path) -> bool:
    """Whether the declaring module checks a capability itself.

    A module may name a host symbol that takes a path when it asks for the
    capability at the call site — the memory store opens a directory to fsync
    it, which libc only offers through `open`. That is the same check the
    public builtin for the class makes, just written where the call is.
    """
    text = (ROOT / path).read_text(encoding="utf-8", errors="replace")
    return "capability-require!" in text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    groups = defaultdict(list)
    sinks = []
    for path, name, real in declarations():
        kind = classify(real)
        groups[kind].append((path, name, real))
        if real in SINK_OF and not module_asks_for_a_capability(path):
            sinks.append((path, name, real, SINK_OF[real]))

    total = sum(len(v) for v in groups.values())
    print(f"extern declarations in the module tree: {total}")
    for kind in ("runtime", "libc", "unknown"):
        print(f"  {kind:8} {len(groups[kind])}")

    print()
    print("host symbols that take a name, in a module that does not check a "
          "capability itself:")
    if not sinks:
        print("  (none)")
    for path, name, real, capability in sorted(sinks):
        print(f"  {capability:11} {real:16} {name:24} {path}")

    if groups["unknown"]:
        print()
        print("host symbols this audit does not classify (check them by hand):")
        for path, name, real in sorted(groups["unknown"]):
            print(f"  {real:24} {name:24} {path}")

    if args.check and sinks:
        print()
        print("FAIL: a module names a host sink symbol directly and checks no "
              "capability for it; route it through the runtime's gated wrapper "
              "or ask for the capability at the call site")
        return 1
    if args.check:
        print()
        print("PASS: every host sink a module names is either the runtime's "
              "gated wrapper or checked where it is called")
    return 0


if __name__ == "__main__":
    sys.exit(main())
