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

## Execution context

The ABI consumer binds a reviewed dependency contract into every ticket. The reusable binding holds the invocation, immutable executable snapshot, file identities and launch phases. The ABI consumer still owns the approved operation, expected output and publication decision. [Dependency contract](abi-context.json) · [Binding](binding.py)

| Class | ABI fingerprint scope |
|---|---|
| Fixed | The ELF, arguments, environment, working directory, descriptor roles, owner policy and six output fields. The startup closure binds each selected SONAME to file content. At publication the sink rereads the policy checkpoint and selected file paths. |
| Permitted to vary | PIDs, descriptor and inode numbers, addresses, scheduling and pipe delivery within the timeout. Identity comparisons within a launch still use the actual sealed inode and pipe endpoints. |
| Outside this contract | Lifetime admission of later dynamic loads, additional file reads, shared memory, external services, additional descriptors opened by the child and kernel behavior. |

The inherited descriptor contract gives the child null stdin, separate stdout and stderr pipes, and the sealed executable descriptor. The replay marks a separate parent file inheritable and verifies that the child lacks that file. An attempted policy change requiring lifetime admission of dynamically loaded code fails the reviewed contract check. A consumer requiring that property needs an enforcement point throughout execution. A startup loader listing and a sample of executable mappings establish the narrower observations recorded here.

For the post-start control, the replay gives one process a private copy of the release libraries. While the child is blocked writing its result, the replay confirms it has mapped the approved PNG library and replaces the selected pathname with an inert data file. The child exits successfully with all six expected fields. The publication check detects the changed digest and withholds its output. Fresh admission also declines the changed closure. The result is a checkpoint guarantee. Transient changes restored between observations can escape detection, and effects already performed by the child persist.

## One-use failure semantics

The issuer spends a recognized ticket when it removes the ticket from its private table, before live-scope revalidation or any attempt to start the authorized operation. A foreign ticket fails lookup at that sink. For a successfully released ticket the consumer records its last confirmed phase.

```text
bytes released → launch requested → child handle received → child exit observed → result published
```

An exception ends the attempt. Once the issuer has spent the ticket, every phase is terminal for retry with that ticket. “Launch requested” records entry to process creation. A caller receiving an error there may have lost confirmation of a successful run. Acknowledgment in this local sink is a returned child handle. A distributed acknowledgment protocol would need its own evidence.

| Control | Authorized workload starts observed | Same-ticket outcome |
|---|---:|---|
| Fail snapshot preparation after byte release | 0 | Refused |
| Fail the launch call before child creation | 0 | Refused |
| Suppress launch confirmation after the real CLI completed | 1 | Refused |
| Race two callers with the same ticket | 1 | One publication and one refusal |

The loader's dependency probes are separate from the workload-start count. Reassessment can issue a new ticket. Where repeated effects matter, the caller must reconcile an ambiguous earlier outcome before requesting another execution. These guarantees concern one live issuer; they do not establish exactly-once execution across restarts.

## Translator and trust boundary

The R0 kernel and ticket interface retain their original definitions. The subset translator now matches the entire parsed module against four explicit AST shapes. Two comparison operators and two Boolean connectives account for those four combinations. Comments, whitespace and the module docstring carry no decision semantics. Defaults, decorators, extra operands, changed return expressions, ignored type directives and other unrecognized syntax fail translation. Fourteen valid-Python controls exercise the rejection path. The two existing semantic mutations still reach Lean and fail the corresponding theorems. [Translator](../authorization/extract_kernel.py) · [Kernel refinement](../authorization/Refinement.lean)

R1 consumes the independently pinned R2 selection. Its new evidence is a native observation of one operation. The Lean proofs cover the translated decision and byte-release functions. Python wrapper correctness, faithful translation, Linux descriptor execution, procfs and the dynamic loader remain trust assumptions. The library directories and system libraries must be operator-controlled and stable during launch. Mapping observations describe the observed execution. The monotonic generation lives in one running owner. A restarted owner creates a fresh gate and ticket table. Durable anti-rollback across a restored process or machine snapshot requires an external monotonic authority. Eshkol's ordinary compiler entry points continue to have their existing behavior. [Consumer](sink.py)
