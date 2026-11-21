# Authorization R1 · Consume the ABI fingerprint ticket

The Eshkol v1.3.6-evolve Linux x64 lite CLI reports six ABI fields, including `runtime_header_size=8`. R2 checked a constant fold at the header-size call in `main`. The released CLI and the folded CLI print identical fields. This gives the consumption test two different executable identities with the same output. [R2 selection and evidence origins](r2.json)

## Run the operation

Use Linux x86-64 with glibc, Python 3.10+, the pinned Eshkol x64 lite installation and its LLVM 21 dependencies. The kernel must support executable sealed memfds and procfs. The command takes a reviewed release as its reference, reconstructs the five-byte fold and checks the complete proposed ELF against R2's pinned selection. An optional `--candidate FILE` submits independently produced bytes through the same gate.

```sh
python3 -B verification/consumption/sink.py --reference "$(command -v eshkol-run)"
```

If LLVM is installed outside the default system location, supply its library directory with `--llvm-dir`. The consumer builds a fresh environment from the release runtime-dependency directory, the configured LLVM directory and the C locale.

## Reproduce the handoff

Install Lean 4.35.0 on PATH. The replay requires permission to inspect its own child's executable, mappings, arguments, environment, working directory and pipe wait channel through procfs.

```sh
python3 -B verification/consumption/replay.py --reference "$(command -v eshkol-run)" --output consumption.json
```

The caller submits bytes for assessment. The consuming method takes only the ticket. It identifies the expected folded image itself and invokes the package 12 gate with those bytes. It writes the returned bytes into a kernel-sealed snapshot and executes from that descriptor with fixed arguments. The owner serializes authorization, policy renewal and dispatch through child completion. The action and loader manifest participate in the scope digest. The loader baseline is acquired from the operator's reference installation.

| Control | Observable result |
|---|---|
| Replace the input pathname after release and immediately before process creation | The pathname contains the reference. The child's executable has the sealed descriptor's device and inode, and the folded image's SHA-256. A write to the sealed descriptor returns EPERM. |
| Restore generation 0 on disk after the live owner advanced to 1 | Both a generation-0 ticket and a generation-1 ticket presented during rollback fail. Fresh assessment reports a stale policy. Restoring the current record permits a fresh assessment. |
| Present the valid ABI ticket to the heat execution sink or another instance of the ABI sink | Both reject at lookup. The issuing sink subsequently consumes the same ticket successfully. |

The replay fills the child's stdout pipe before launch so it can inspect this short-lived process while the CLI is writing its response. It invokes the normal process-creation path, reads the kernel's executable identity and the actual argument and environment block, then drains the padding and captures the CLI output. It checks the observed command, working directory, environment and executable against the issuance record. It also compares file-backed executable mappings with the bound library closure. The receipt carries digests in place of installation paths. [Replay](replay.py)

## Translator and trust boundary

The R0 kernel and ticket interface retain their original definitions. The subset translator now matches the entire parsed module against four explicit AST shapes. Two comparison operators and two Boolean connectives account for those four combinations. Comments, whitespace and the module docstring carry no decision semantics. Defaults, decorators, extra operands, changed return expressions, ignored type directives and other unrecognized syntax fail translation. Fourteen valid-Python controls exercise the rejection path. The two existing semantic mutations still reach Lean and fail the corresponding theorems. [Translator](../authorization/extract_kernel.py) · [Kernel refinement](../authorization/Refinement.lean)

R1 consumes the independently pinned R2 selection. Its new evidence is a native observation of one operation. The Lean proofs cover the translated decision and byte-release functions. Python wrapper correctness, faithful translation, Linux descriptor execution, procfs and the dynamic loader remain trust assumptions. The library directories and system libraries must be operator-controlled and stable during launch. Mapping observations describe the observed execution. The monotonic generation lives in one running owner. A restarted owner creates a fresh gate and ticket table. Durable anti-rollback across a restored process or machine snapshot requires an external monotonic authority. Eshkol's ordinary compiler entry points continue to have their existing behavior. [Consumer](sink.py)
