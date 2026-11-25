# Render the timeout ceiling into the diagnostic buffer

A quiet timeout records the last limit error and returns without printing. On the upstream reference, formatting the ceiling still asks libc to convert an unsigned long long. A process that has registered a GNU printf handler for that conversion sees the request even in quiet mode. The new timeout branch writes the digits, the `ms` suffix and the terminator directly into its 64-byte local buffer.

## Reproduce

Use a Git checkout containing upstream commit `eb224260b4814baf3ad260744c8dcdd05fb84898`. The replay reads the reference translation unit from that commit and the candidate from your working tree. It compiles both against the same headers and links both against the same Eshkol runtime archive.

The tested platform was x86-64 GNU/Linux, glibc 2.44 and GCC 16.2.1. You'll need Python 3.9+, Git, Lean 4.35.0, a GNU C compiler with the glibc printf registration headers, a C++17 compiler, and linkable PNG, JPEG, WebP and OpenSSL crypto libraries. Unpack Eshkol's v1.3.6-evolve Linux x64 lite release into `ESHKOL_RELEASE`. Its archive SHA-256 is `a1c467edf5ee519ab36d3f1a24df2eb2cba80211e303fec5be25a49f223d5f3b`. The runner checks the digest of the extracted static runtime archive before linking. [Public release archive](https://github.com/tsotchke/eshkol/releases/download/v1.3.6-evolve/eshkol-v1.3.6-evolve-linux-x64-lite.tar.gz)

```sh
python3 verification/timeout-decimal/replay.py --release "$ESHKOL_RELEASE"
```

Pass `--lean /path/to/lean` if needed. The command prints a JSON receipt on success. The [recorded receipt](receipt.json) includes the reference and candidate source digests, both ELF digests, compiler and libc versions, axiom reports and case counts.

## What the experiment checks

Each case starts a fresh process. A direct formatting call first renders the largest unsigned 64-bit integer as `18446744073709551615ms` and confirms whether the handler is active. The probe then arranges for any subsequent entry into that unsigned conversion handler to exit with a distinct status. It calls the public `eshkol_limit_enforce` entry point.

At a 30,000 ms ceiling the reference reaches the handler in quiet, warning and fatal modes. The candidate must match the reference's ordinary stdout, stderr and status in all three policies with either registry. The 61 ceilings include zero, the largest uint64, and values immediately around all 19 in-range positive powers of ten from 10 through 10¹⁹. On advisory return, the probe also checks the recorded limit error. Fifteen further comparisons cover heap-soft, heap-hard, stack, tensor and string limits across the three policies.

## What Lean establishes

For every unsigned 64-bit input, `Verificatum.TimeoutDecimal.quotient` and `digit` establish the division-by-ten identity and the value of one ASCII digit. `formatterExtent` establishes one to twenty digit writes in offsets 41 through 60. The suffix occupies offsets 61 through 63. The runner checks the three theorem types against independently written targets and audits their axioms. The arithmetic uses `propext` and `Quot.sound`, and the extent theorem also uses `Classical.choice`.

The production comparison covers the automatic formatting of the timeout ceiling. Later warning logging, fatal I/O, and the effects of other conversions are separate. The theorem concerns the modeled arithmetic and indices. The native comparison exercises this build and corpus. A general proof connecting the compiled instructions to the model remains open.
