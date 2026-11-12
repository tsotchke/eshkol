#!/usr/bin/env python3
"""Admit an explicitly pinned compiler/source/stdlib/runtime cohort."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from check_stdlib_source_identity import checked_path, need, stable_bytes
from check_stdlib_source_identity import verify as verify_source


def exact_fields(value, keys, label):
    need(isinstance(value, dict) and set(value) == set(keys),
         "invalid fields in " + label)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, "duplicate manifest key: " + key)
        result[key] = value
    return result


def identity_fields(text):
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        need(separator and key and value and key not in result,
             "invalid or duplicate compiler identity field")
        result[key] = value
    return result


def run(argv):
    result = subprocess.run(list(map(str, argv)), capture_output=True,
                            timeout=30, env=dict(os.environ, GIT_OPTIONAL_LOCKS="0", LC_ALL="C"))
    need(result.returncode == 0,
         "command refused: " + str(argv[0]) + ": " + result.stderr.decode("utf-8", "replace")[-1200:])
    return result.stdout.decode("utf-8"), result.stderr


def revision(root):
    top, _ = run(["git", "-C", root, "rev-parse", "--show-toplevel"])
    need(Path(top.strip()).resolve() == root,
         "source root is not its own Git checkout")
    head, _ = run(["git", "-C", root, "rev-parse", "--verify", "HEAD"])
    return head.strip()


def verify(manifest, expected_manifest_sha256, source_root, artifact_root, selected=None):
    if selected is not None:
        exact_fields(selected, ["compiler", "stdlib", "runtime"], "effective consumer paths")
    need(digest(expected_manifest_sha256), "manifest pin must be 64 lowercase hex characters")
    blob, manifest_stamp = stable_bytes(manifest)
    need(hashlib.sha256(blob).hexdigest() == expected_manifest_sha256,
         "manifest differs from independently supplied SHA-256 pin")
    data = json.loads(blob, object_pairs_hook=unique_object)
    exact_fields(data, ["schema", "source_revision", "identity", "artifacts"], "manifest")
    need(data["schema"] == "eshkol.consumer-toolchain/v1", "unsupported manifest schema")
    need(isinstance(data["source_revision"], str) and
         re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", data["source_revision"]),
         "source revision must be a full lowercase Git object ID")
    identity = data["identity"]
    exact_fields(identity, ["release", "abi", "source", "symbol"], "identity")
    need(all(isinstance(v, str) and v and not any(c in v for c in "\r\n")
             for v in identity.values()), "invalid identity value")
    need(digest(identity["source"]), "invalid source fingerprint")
    normalize = lambda s: re.sub(r"[^A-Za-z0-9_]", "_", s)
    need(identity["symbol"] == "eshkol_stdlib_identity_release_" + normalize(identity["release"]) +
         "_abi_" + normalize(identity["abi"]) + "_source_" + identity["source"],
         "identity fields disagree with symbol")
    artifacts = data["artifacts"]
    exact_fields(artifacts, ["compiler", "stdlib", "runtime", "source_material"], "artifacts")
    root = artifact_root.resolve(strict=True)
    source = source_root.resolve(strict=True)
    need(root.is_dir() and source.is_dir(), "source and artifact roots must be directories")
    paths, stamps, names = {}, {}, set()
    for kind, row in artifacts.items():
        exact_fields(row, ["path", "sha256"], kind)
        relative = row["path"]
        need(isinstance(relative, str) and relative and "\\" not in relative and
             not relative.startswith("/") and
             all(p not in ("", ".", "..") for p in relative.split("/")),
             "unsafe artifact path: " + kind)
        need(relative not in names and digest(row["sha256"]), "duplicate path or invalid artifact digest")
        names.add(relative)
        path = checked_path(root, relative)
        value, stamps[kind] = stable_bytes(path)
        need(hashlib.sha256(value).hexdigest() == row["sha256"], "artifact digest mismatch: " + kind)
        paths[kind] = path
    need(os.access(paths["compiler"], os.X_OK), "compiler is not executable")
    if selected is not None:
        for kind, path in selected.items():
            need(path.absolute() == paths[kind], "effective consumer path mismatch: " + kind)
    need(revision(source) == data["source_revision"], "source revision mismatch")
    source_report = verify_source(source, paths["source_material"], identity["source"])
    expected = dict(schema="eshkol.stdlib-identity/v1", **identity)
    output, diagnostics = run([paths["compiler"], "--stdlib-identity"])
    need(not diagnostics and identity_fields(output) == expected, "compiled identity mismatch")
    output, diagnostics = run([paths["compiler"], "--check-stdlib-identity", paths["stdlib"]])
    need(not diagnostics and identity_fields(output) == dict(expected, status="PASS"),
         "stdlib identity mismatch")
    # Repeat source verification after executing the pinned metadata commands.
    # This detects drift; it does not provide filesystem snapshot isolation.
    verify_source(source, paths["source_material"], identity["source"])
    need(revision(source) == data["source_revision"], "source revision changed during admission")
    for kind, path in paths.items():
        value, stamp = stable_bytes(path)
        need(stamp == stamps[kind] and hashlib.sha256(value).hexdigest() == artifacts[kind]["sha256"],
             "artifact changed during admission: " + kind)
    after, stamp = stable_bytes(manifest)
    need(after == blob and stamp == manifest_stamp, "manifest changed during admission")
    return {"status": "PASS_CONSUMER_TOOLCHAIN_ADMISSION", "manifest_sha256": expected_manifest_sha256,
            "consumer_bound": selected is not None,
            "consumer_paths": {kind: str(path.absolute()) for kind,path in selected.items()} if selected else None,
            "source_revision": data["source_revision"], "identity": identity,
            "artifacts": artifacts, "source_inputs": source_report["input_count"],
            "scope": "pinned cohort and metadata admission; not runtime or release qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    for kind in ("compiler", "stdlib", "runtime"):
        parser.add_argument("--" + kind + "-path", type=Path)
    args = parser.parse_args()
    try:
        selected = {kind: getattr(args, kind + "_path") for kind in ("compiler", "stdlib", "runtime")
                    if getattr(args, kind + "_path") is not None}
        result = verify(args.manifest, args.expected_manifest_sha256, args.source_root,
                        args.artifact_root, selected or None)
    except (OSError, ValueError, UnicodeError, subprocess.SubprocessError) as error:
        print(json.dumps({"status": "FAIL", "reason": str(error)}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
