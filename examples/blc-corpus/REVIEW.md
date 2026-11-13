# Reviewing contributed BLC programs

`.blc` is executable lambda code represented as ASCII bits. `.blc8` is the
same code packed into bytes. The calculus has no host file/network primitive;
that fact does not protect a vulnerable parser or evaluator, or prevent a
program from exhausting resources. Treat both forms as untrusted programs.

Run static admission, which never reduces a term:

```sh
python3 examples/blc-corpus/review.py check
python3 examples/blc-corpus/review_self_test.py
```

Review the protocol and fixtures, the generated `review/*.txt` node tables,
and changes to the validator/runner themselves. Tables use explicit lambda,
application and 1-based bound-variable nodes. They re-encode to the complete
original bits; there are no ignored trailing bits or hidden byte payloads.
The gate bounds syntax, file size and nesting, rejects open/truncated code,
symlinks, noncanonical text and incorrect packed padding, and checks the
committed readable views. It never rewrites views during checking.

Authors regenerate views and seals explicitly with `review.py generate`.
Hashes establish identity, not trust: an author can change both data and its
manifest. Syntax admission does not prove termination, protocol correctness
or benevolent intent. Independent fixtures sample behavior; they do not
prove a program implements its advertised function on every input.

For execution of untrusted submissions, use an ephemeral, unprivileged Linux
runner with no secrets or publishing credentials. Run the review tool from a
reviewed revision. CI must not execute contributor-modified infrastructure
with privileged pull-request triggers or secrets. Validator changes require
review just as compiler changes do.

The execution wrapper fails closed without bubblewrap and working user,
network and PID namespaces. It mounts only system tools, explicit runtime
directories, the corpus and the Eshkol library read-only; temporary compiler
outputs and the report have their own writable directory. It clears the host
environment and exposes no home directory or agent socket. On Python 3.10,
only the trusted `tomli` package is mounted separately if needed. The existing
runner retains per-process CPU, address-space and wall bounds, caps captured
stdout/stderr at 8 MiB combined and bounds regular output-file size. Use
immutable source/runtime directories during admission and execution.

```sh
python3 examples/blc-corpus/review.py run \
  --binary /opt/eshkol/bin/eshkol-run --runtime-root /opt/eshkol \
  --report /tmp/blc-report.toml -- \
  --backend eshkol-aot --fixture vs-empty --timeout 30 --memory-mib 2048
```

Use a fixture ID from `fixtures.toml`. Additional `--runtime-root` arguments
may expose narrowly scoped dependency directories. They are trusted runtime
inputs; never supply a home, workspace or filesystem root. There is no
unsandboxed fallback. The normal `scripts/run_blc_bytes_gate.sh --corpus` path uses this wrapper.
`ESHKOL_SANDBOX_RUNTIME_ROOT` and colon-separated
`ESHKOL_SANDBOX_DEPENDENCIES` expose explicitly supplied runtime directories.
Exported logs/reports reject symlinks and nonregular files.
The old `run.py run` command is a low-level trusted-code
runner and does not establish a network/filesystem security boundary.
Namespaces and resource bounds reduce exposure; they are not a proof against
kernel vulnerabilities or every possible denial of service. A fresh VM or
hardened disposable container remains appropriate for arbitrary contributors.
