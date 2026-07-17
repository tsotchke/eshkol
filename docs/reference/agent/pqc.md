# `agent.pqc` — ML-KEM (FIPS 203) Post-Quantum KEM

ML-KEM (FIPS 203, the standardized Kyber) key encapsulation over Moonlab's
implementation, at the 512/768/1024 security levels. Keys, ciphertexts, and
the shared secret are ordinary R7RS **bytevectors** sized exactly to the
FIPS 203 parameter sets — this module deliberately exposes only serializable
values, no opaque handles.

```scheme
(require agent.pqc)
```

Source: `lib/agent/pqc.esk`. C shim: `lib/agent/c/agent_pqc.c` (symbols
`eshkol_mlkem_*`).

## Availability — opt-in build flag

Like [`agent.quantum`](quantum.md), this module shares Moonlab's opt-in link
target and is only usable when Eshkol is built with
`-DESHKOL_QUANTUM_ENABLED=ON`. On other builds the externs do not resolve
(honest unavailability, not a silent classical stand-in).

## Randomness and determinism

Production key generation and encapsulation always use Moonlab's
**Bell-verified QRNG** wrappers. Moonlab's deterministic entry points are
intentionally not part of this Scheme surface; the C shim uses them only in
its fixed-seed NIST KAT helper exercised by the test suite
(`tests/quantum/pqc_mlkem_test.esk`), which checks the published pq-crystals
count=0 SHA3-256 fingerprints for every artifact (pk/sk/ct/K).

## Error contract

- `level` must be `512`, `768`, or `1024` — anything else raises a catchable
  Eshkol error.
- Every key/ciphertext argument is validated for bytevector type **and** exact
  FIPS 203 length before the C call; wrong type or size raises.
- A failing Moonlab call raises a catchable error carrying Moonlab's own
  message (via `eshkol_mlkem_last_error`).
- As required by FIPS 203, `mlkem-decaps` on a correctly sized but tampered
  ciphertext returns an **implicit-rejection** secret rather than raising —
  no validity oracle is exposed.

All three KEM operations are capability-gated on `ffi`
(see [capabilities](capabilities.md)).

## KEM operations

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `mlkem-keygen` | `(mlkem-keygen level)` | `#(public-key secret-key)` — two bytevectors sized by the parameter set; QRNG-seeded |
| `mlkem-encaps` | `(mlkem-encaps public-key)` | `#(ciphertext shared-secret)` — QRNG-seeded; the parameter set is inferred from the public key's exact FIPS 203 length |
| `mlkem-decaps` | `(mlkem-decaps secret-key ciphertext)` | the 32-byte shared-secret bytevector; parameter set inferred from the secret-key length |

## Size accessors

| Name | Signature | 512 | 768 | 1024 |
|------|-----------|-----|-----|------|
| `mlkem-public-key-bytes` | `(mlkem-public-key-bytes level)` | 800 | 1184 | 1568 |
| `mlkem-secret-key-bytes` | `(mlkem-secret-key-bytes level)` | 1632 | 2400 | 3168 |
| `mlkem-ciphertext-bytes` | `(mlkem-ciphertext-bytes level)` | 768 | 1088 | 1568 |
| `mlkem-shared-secret-bytes` | constant (not a procedure) | 32 | 32 | 32 |

`mlkem-shared-secret-bytes` is a plain provided value (`32`); the other three
are procedures of `level` and raise on an invalid level.

## Round-trip example

The pattern verified at all three levels by
`tests/quantum/pqc_mlkem_test.esk` (sizes, round-trip equality, and
distinctness of independently generated keypairs/encapsulations):

```scheme
(require agent.pqc)

(let* ((keypair       (mlkem-keygen 768))
       (public-key    (vector-ref keypair 0))
       (secret-key    (vector-ref keypair 1))
       (encapsulation (mlkem-encaps public-key))
       (ciphertext    (vector-ref encapsulation 0))
       (shared-secret (vector-ref encapsulation 1))
       (recovered     (mlkem-decaps secret-key ciphertext)))
  ;; shared-secret and recovered are equal 32-byte bytevectors
  (display (bytevector-length public-key)) (newline)     ; 1184
  (display (bytevector-length ciphertext)) (newline)     ; 1088
  (display (bytevector-length recovered)) (newline))     ; 32
```

## See also

- [`agent.quantum`](quantum.md) — the state-vector/VQE surface sharing the
  Moonlab link target, and the `quantum-random` builtin family that seeds
  these operations.
- [`agent.crypto`](crypto.md) — classical hashing/randomness/base64url.
