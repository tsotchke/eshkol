# Consumer toolchain admission preparation

`scripts/check_consumer_toolchain.py` checks one explicitly selected compiler,
source, standard library and runtime cohort before a consumer uses it.
`cmake/ConsumerToolchainAdmission.cmake` exposes the same check to consumer builds.
No cohort is selected by these tools. Existing agent version/revision requirements
are unchanged; this module is not wired into the agent's default build.

The release owner must supply the SHA-256 of an independently reviewed manifest.
Computing a hash from an untrusted manifest at the point of admission does not
establish trust. The checker and its reviewed source collector must themselves
come from a trusted checkout; the artifact directory supplies no executable
verification scripts. A selected compiler is executed only after its byte hash
matches the independently pinned manifest.

The exact manifest shape is:

```json
{
  "schema": "eshkol.consumer-toolchain/v1",
  "source_revision": "full lowercase Git object ID",
  "identity": {
    "release": "exact raw release string",
    "abi": "exact object ABI string",
    "source": "64 lowercase hex source-material fingerprint",
    "symbol": "exact identity symbol derived from release, ABI and fingerprint"
  },
  "artifacts": {
    "compiler": {"path": "bin/eshkol-run", "sha256": "64 lowercase hex"},
    "stdlib": {"path": "lib/stdlib.o", "sha256": "64 lowercase hex"},
    "runtime": {"path": "lib/libeshkol-static.a", "sha256": "64 lowercase hex"},
    "source_material": {"path": "stdlib-source-material.txt", "sha256": "64 lowercase hex"}
  }
}
```

All artifact paths are distinct, relative to the supplied artifact root, with no
parent traversal or symlink components. Inputs must be regular files. Unknown
fields, duplicate JSON keys, missing pins, absent Git information, and identity
mismatches cause refusal. Source archives without their own Git checkout are
currently refused rather than borrowing the enclosing checkout's identity.

Source revision and configured source material are verified independently of
artifact paths. The source-material check includes ignored files within the
reviewed collector's scopes; Git-clean status alone is insufficient. The exact
compiler metadata must match the manifest, and its stdlib check must succeed on
the pinned object. The runtime archive is bound by its independently pinned byte
hash; the tool does not invent a runtime-archive identity stamp.

```cmake
include("${TRUSTED_ESHKOL_TOOLS}/cmake/ConsumerToolchainAdmission.cmake")
eshkol_require_consumer_toolchain(
    MANIFEST "${SELECTED_MANIFEST}"
    MANIFEST_SHA256 "${INDEPENDENT_MANIFEST_SHA256}"
    SOURCE_ROOT "${SELECTED_SOURCE_ROOT}"
    ARTIFACT_ROOT "${SELECTED_ARTIFACT_ROOT}")
file(WRITE "${CMAKE_BINARY_DIR}/consumer-admission.json"
    "${ESHKOL_CONSUMER_ADMISSION_JSON}")
```

Run the check before consuming the cohort. Source and artifacts must remain
quiescent: repeated checks detect changes during admission but provide neither
filesystem snapshot isolation nor protection against changes after return.
This does not qualify binary behavior, FFI/worker compatibility, dependency
closure, installed consumers, or a release. Existing native/runtime acceptance
and owner selection remain required. Adding the CMake module changes configured
source inputs; previous compiler qualification cannot be promoted to this branch.

Build-free controls retain their logs and synthetic inputs:

```sh
python3 tests/toolchain/consumer_toolchain_admission_test.py --work-root /path/to/proof
```

These controls use Python compiler metadata fixtures with real Git/CMake checks.
They do not compile or execute an Eshkol product, and are not registered as a
native compiler acceptance test.
