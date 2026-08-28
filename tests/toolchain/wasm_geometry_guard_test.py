#!/usr/bin/env python3
"""Check the WASM ABI geometry contract and its mismatch direction."""

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_wasm_imports.py"
spec = importlib.util.spec_from_file_location("check_wasm_imports", SCRIPT)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load {SCRIPT}")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


def main() -> int:
    for path in checker.JS_FILES:
        failures = checker.validate_wasm_abi_geometry(path.read_text())
        if failures:
            raise AssertionError(f"{path}: {failures}")

    fields = "\n".join(
        f"    {name}: {value},"
        for name, value in checker.WASM_ABI_GEOMETRY.items()
    )
    valid = f"const wasmAbiGeometry = Object.freeze({{\n{fields}\n}});"
    wrong = valid.replace("objectHeaderSize: 8", "objectHeaderSize: 9", 1)
    failures = checker.validate_wasm_abi_geometry(wrong)
    if not any("objectHeaderSize" in failure for failure in failures):
        raise AssertionError("wrong object header size did not fire the guard")

    print("wasm_geometry_guard_test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
