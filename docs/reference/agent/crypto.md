# `agent.crypto` — Hashing, Randomness, and Base64URL

Cryptographic primitives and encodings used by the agent surface (JWT signing,
content-addressing, token generation). Backed by C runtime symbols
(`eshkol_sha256`, `eshkol_hmac_sha256`, `eshkol_random_bytes`,
`eshkol_random_hex`) declared through the [FFI `extern`](ffi.md) mechanism.

```scheme
(require agent.crypto)
```

Source: `lib/agent/crypto.esk`.

## Provided procedures

| Procedure | Signature | Returns |
|-----------|-----------|---------|
| `sha256` | `(sha256 data)` | 64-char lowercase hex digest string |
| `hmac-sha256` | `(hmac-sha256 key data)` | hex digest string |
| `random-bytes` | `(random-bytes len)` | string of `len` random bytes |
| `random-hex` | `(random-hex hex-len)` | hex string of `hex-len` hex chars |
| `uuid-v4` | `(uuid-v4)` | RFC-4122 v4 UUID string |
| `base64url-encode` | `(base64url-encode data)` | base64url string, no padding |
| `base64url-decode` | `(base64url-decode url-str)` | decoded byte string |

## Unpredictable values come from here, not from `(random)`

Eshkol has two generators and they are for different things.

- `(random)` and `(set-random-seed! n)` are a **seeded, reproducible** stream
  (a drand48-compatible LCG). Seeded with the same value they produce the same
  sequence, which is what makes a simulation, a sampler or a shuffle
  replayable. Left unseeded the runtime seeds it from the clock on first use,
  so an unseeded run differs from the next — but a clock is a small, guessable
  space, and that is not what the generator is for. Use `(random)` for
  simulation and replay. Do not use it for anything whose worth rests on being
  unguessable.
- `random-bytes`, `random-hex` and `uuid-v4` draw from the **operating
  system's cryptographic source** — `SecRandomCopyBytes` on macOS,
  `BCryptGenRandom` on Windows, `RAND_bytes` where OpenSSL is linked, and
  `/dev/urandom` otherwise. There is no fallback: when that source cannot be
  read they report failure rather than return a value a clock could have
  produced. Use these for session identifiers, tokens, nonces, salts and keys.

The runtime follows the same rule internally. The `Sec-WebSocket-Key` and the
per-frame masking key of the [WebSocket client](http.md), and the session
identifiers of the
[compilation server](../runtime/compilation-server.md), all come from the
cryptographic source, on the native engine and on the bytecode VM alike.

`data` / `key` are byte strings (Eshkol strings are length-tagged byte buffers,
so binary payloads are fine). `base64url-encode` takes a **string** (bytes-as-string),
not a byte list — it delegates to `base64-encode-string`, strips `=` padding, and
maps `+/` to `-_`. `base64url-decode` re-pads and reverses the mapping.

## Verified examples

```scheme
(require agent.crypto)
(display (sha256 "hello")) (newline)
;; => 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824

(display (base64url-encode "hi?")) (newline)
;; => aGk_
```

```
$ eshkol-run -r sha.esk
2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
aGk_
```

Both `-r` (JIT) and AOT builds resolve the crypto C symbols; because other
faculties (e.g. the [memory faculty](memory-faculty.md)) call `sha256`
internally, the transitive agent-FFI link scan pulls `eshkol_sha256` into AOT
binaries automatically (see [FFI & AOT linking](ffi.md)).

## Notes

- `sha256` returns the hex string, not raw bytes. For raw digest bytes call the
  underlying `sha256-raw` (`:real eshkol_sha256`) directly.
- `base64url` is padding-free (JWT-style). Round-tripping through
  `base64url-decode` yields the original byte string.
