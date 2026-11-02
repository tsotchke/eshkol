"""Check the exact verdict contract of the required curvature VM fixtures."""
import argparse
from pathlib import Path
import re


# Counts include the success summary in fixtures that emit one.
CONTRACTS = {
    "signed_curvature_surface": (37, "prefix"),
    "geometric_riemannian_surface_regression": (166, "prefix"),
    "geometric_adversarial_surface_regression": (11, "prefix"),
    "frechet_mean_surface_regression": (14, "suffix"),
    "riemannian_adam_state_regression": (19, "prefix"),
}
FAILURE = re.compile(
    r"^\s*(?:FAIL|FAILED|FAILURE|ERROR)\b|:\s*(?:FAIL|FAILED|FAILURE)\s*$"
    r"|unhandled native call|fatal signal",
    re.IGNORECASE | re.MULTILINE,
)
PASS = {
    "prefix": re.compile(r"^PASS: .+$"),
    "suffix": re.compile(r"^.+: PASS$"),
}


def validate(fixture, text):
    expected, style = CONTRACTS[fixture]
    if FAILURE.search(text):
        raise ValueError(f"{fixture}: failure diagnostic in VM output")
    verdicts = [line for line in text.splitlines() if PASS[style].fullmatch(line)]
    if len(verdicts) != expected:
        raise ValueError(
            f"{fixture}: expected {expected} {style} PASS verdicts, "
            f"found {len(verdicts)}"
        )
    other = "suffix" if style == "prefix" else "prefix"
    if any(PASS[other].fullmatch(line) for line in text.splitlines()):
        raise ValueError(f"{fixture}: unexpected verdict style")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture", choices=CONTRACTS)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        validate(args.fixture, args.output.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"PASS: exact VM verdict inventory for {args.fixture}")


if __name__ == "__main__":
    main()
