#!/usr/bin/env python3
"""Compare configured source inputs with an independently supplied compiler digest."""
import argparse
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

COLLECTOR = Path("cmake/StdlibIdentityInputs.cmake")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def stable_bytes(path):
    """Do not follow a final symlink or block on a nonregular input."""
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode), "not a regular file: " + str(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    with os.fdopen(fd, "rb") as stream:
        opened = os.fstat(stream.fileno())
        need(stat.S_ISREG(opened.st_mode), "opened nonregular input: " + str(path))
        value = stream.read()
        after = os.fstat(stream.fileno())
    key = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    need(key(before) == key(opened) == key(after) == key(path.lstat()),
         "input changed during read: " + str(path))
    return value, key(after)


def scopes(module):
    text = module.decode("utf-8")
    result = []
    for name in ("patterns", "files"):
        blocks = re.findall(r"set\(_stdlib_identity_" + name + r"\s+([^)]*)\)", text)
        need(len(blocks) == 1, "unsupported collector declaration: " + name)
        body = blocks[0]
        items = re.findall(r'"([^"\n]+)"', body)
        need(items and not re.sub(r'"[^"\n]+"|\s+', "", body),
             "unsupported collector syntax: " + name)
        need(len(items) == len(set(items)), "duplicate collector scope")
        for item in items:
            need(re.fullmatch(r"[A-Za-z0-9_./*-]+", item) is not None and
                 not item.startswith("/") and all(p not in ("", ".", "..") for p in item.split("/")),
                 "unsafe collector scope")
            if name == "patterns":
                need(item.endswith(("/*", "/*.esk")) and "*" not in item.rsplit("/", 1)[0],
                     "unsupported recursive pattern")
            else:
                need("*" not in item, "unsupported literal input")
        result.append(items)
    return result


def checked_path(root, relative, optional=False):
    path = root
    for part in Path(relative).parts:
        path = path / part
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if optional:
                return None
            raise
        need(not stat.S_ISLNK(mode), "symlink in source input: " + str(path))
    return path


def namespace(root, patterns, files):
    paths = set(files)
    for pattern in patterns:
        directory, mask = pattern.rsplit("/", 1)
        base = checked_path(root, directory, optional=True)
        if base is None:
            continue
        need(base.is_dir(), "source scope is not a directory: " + str(base))
        def walk_error(error):
            raise error
        for current, dirs, names in os.walk(base, followlinks=False, onerror=walk_error):
            for name in dirs + names:
                path = Path(current) / name
                mode = path.lstat().st_mode
                # Refuse even nonmatching links/special entries inside an input scope.
                need(stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                     "symlink or special source entry: " + str(path))
            for name in names:
                # CMake GLOB is case-insensitive on macOS and Windows.
                candidate = name.lower() if sys.platform in ("darwin", "win32") else name
                glob_mask = mask.lower() if sys.platform in ("darwin", "win32") else mask
                if fnmatch.fnmatchcase(candidate, glob_mask):
                    paths.add((Path(current) / name).relative_to(root).as_posix())
    for relative in paths:
        need(not any(c in relative for c in ("\n", "\r", ";", "\\")),
             "unsupported input path encoding")
        path = checked_path(root, relative)
        need(stat.S_ISREG(path.lstat().st_mode), "nonregular source input: " + str(path))
    return sorted(paths)


def verify(source_root, material_path, expected):
    need(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
         "expected fingerprint must be 64 lowercase hex characters")
    root = source_root.resolve(strict=True)
    material, material_stat = stable_bytes(material_path)
    need(hashlib.sha256(material).hexdigest() == expected,
         "material digest differs from independently supplied fingerprint")
    # Only the pinned toolchain suffix comes from material. Discover source paths
    # independently, so a forged/partial manifest cannot omit an actual input.
    lines = material.splitlines(keepends=True)
    need(lines and re.fullmatch(rb"toolchain:c=[^;\r\n]+;cxx=[^;\r\n]+;llvm=[^;\r\n]+\n", lines[-1]),
         "invalid toolchain suffix")
    trusted_module = Path(__file__).resolve().parents[1] / COLLECTOR
    trusted, _ = stable_bytes(trusted_module)
    module, _ = stable_bytes(checked_path(root, COLLECTOR))
    need(module == trusted, "source collector differs from this checker's reviewed collector")
    patterns, files = scopes(trusted)
    paths = namespace(root, patterns, files)
    records, stamps = [], {}
    for relative in paths:
        path = checked_path(root, relative)
        need(path.resolve() != material_path.resolve(), "material file is inside source input scope")
        value, stamps[relative] = stable_bytes(path)
        records.append(relative + ":" + hashlib.sha256(value).hexdigest() + "\n")
    rebuilt = "".join(records).encode("utf-8") + lines[-1]
    need(namespace(root, patterns, files) == paths, "source namespace changed during verification")
    for relative, stamp in stamps.items():
        s = checked_path(root, relative).lstat()
        need((s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns) == stamp,
             "source changed during verification: " + relative)
    material_after, stat_after = stable_bytes(material_path)
    need(material == material_after and material_stat == stat_after, "material changed during verification")
    need(rebuilt == material, "configured source namespace or bytes differ from material")
    return {"status": "PASS", "source_fingerprint": expected, "input_count": len(paths),
            "toolchain": lines[-1].decode("utf-8").rstrip("\n"),
            "scope": "configured stdlib identity inputs; no release-selection or runtime claim"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--material", type=Path, required=True)
    parser.add_argument("--expected-source-fingerprint", required=True)
    args = parser.parse_args()
    try:
        result = verify(args.source_root, args.material, args.expected_source_fingerprint)
    except (OSError, ValueError, UnicodeError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
