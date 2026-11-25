"""Emit the installed source manifest and the module closure shipped in stdlib.o."""

import hashlib
import json
from pathlib import Path
import sys

from gen_language_surface import extract_stdlib_surface


def main():
    root = Path(__file__).resolve().parents[1]
    closure = {row["path"] for row in extract_stdlib_surface(str(root))["modules"]}
    rows = []
    for source in sorted((root / "lib").rglob("*.esk")):
        path = source.relative_to(root / "lib").as_posix()
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        compiled = "true" if "lib/" + path in closure else "false"
        rows.append(f'    {{{json.dumps(path)}, "{digest}", {compiled}}},\n')
    sys.stdout.writelines(sorted(rows))


if __name__ == "__main__":
    main()
