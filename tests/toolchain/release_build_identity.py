"""Compare configured public identity, packaging and the built compiler."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / "scripts"))
from release_target import project_identity

runner = Path(sys.argv[1])
build = Path(sys.argv[2])
tag, _ = project_identity()
version = tag[1:]
components = version.split("-", 1)[0].split(".")
header = (build / "generated/eshkol/version.h").read_text().splitlines()
for key, value in zip(("MAJOR", "MINOR", "PATCH", "STRING"), [*components, f'"{version}"']):
    if f"#define ESHKOL_VERSION_{key} {value}" not in header:
        raise SystemExit(f"public header must report {key}={value}")
if (build / "release-tag").read_text().strip() != tag:
    raise SystemExit("package release tag differs from the project identity")
if f"Version: {version}" not in (build / "eshkol.pc").read_text().splitlines():
    raise SystemExit("pkg-config version differs from the project identity")
if f'set(CPACK_PACKAGE_VERSION "{version}")' not in (build / "CPackConfig.cmake").read_text().splitlines():
    raise SystemExit("CPack version differs from the project identity")
result = subprocess.run([runner, "--version"], text=True, capture_output=True, check=True)
if result.stdout.strip() != f"Eshkol Compiler {tag}":
    raise SystemExit(f"compiler reported {result.stdout!r}, expected {tag}")
print(f"PASS {tag}")
