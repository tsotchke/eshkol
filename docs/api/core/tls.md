# `core.tls`

`core.tls` provides optional native OpenSSL client and server connections over
`core.net`'s opaque TCP handles. Positive context, listener, and session IDs
are exact integers no greater than `9007199254740991`; IDs are never reused.
C failures use `-1` for system or TLS failure, `-2` for an operation deadline,
`-3` when native TLS is unavailable, and `-4` for invalid input or a stale ID.
Scheme returns a `tls-error` vector for negative results. Context creation,
connection, accept, read, and write require the `network` capability. Explicit
CA, certificate, and private-key paths also require `file-read`. Capability
and port queries and cleanup remain usable after capabilities are revoked.

`(tls-client-context #f)` uses system default trust paths. Passing a CA file
replaces that trust source. Client contexts always verify the certificate
chain and peer identity; there is no unverified-client option. Server
contexts load a PEM certificate chain and matching unencrypted PEM private key
and fail closed when a file is missing or the key does not match. Encrypted
keys return a TLS error; this API does not request passwords from the terminal.
Native sessions use TLS 1.2 or newer.

On macOS, when neither `SSL_CERT_FILE` nor `SSL_CERT_DIR` is set,
`(tls-client-context #f)` loads the system PEM bundle at `/etc/ssl/cert.pem`
before consulting OpenSSL's build-time default paths. `ESHKOL_OPENSSL_CERT_FILE`
provides an explicit Eshkol-specific override; the standard OpenSSL variables
take precedence when present.

`(tls-listen context address port)` binds a numeric IPv4/IPv6 address; port
zero selects an ephemeral port, which `tls-listener-port` returns.
`(tls-accept listener timeout-ms)` accepts and completes the server handshake
within one total deadline. `(tls-connect context address port server-name
timeout-ms)` connects to a numeric address and verifies the peer against a
DNS name or IP address. Pass `#f` for `server-name` to verify the numeric
address itself. The TCP connection and TLS handshake share that deadline.

`tls-read` returns up to the requested number of plaintext string bytes; an
empty string indicates authenticated TLS `close_notify`. `tls-write` returns
the number of bytes sent. Reads and writes use one monotonic deadline each,
including any retries for TLS records. Binary strings, including embedded NUL
and multibyte UTF-8, are preserved. A timed-out or failed write poisons the
session because part of a TLS record may already have been sent; close it after
a write error. Closing a context blocks new connections
while existing listeners or sessions retain the native context until their
references are released. Closing a session or listener invalidates its ID
immediately and is safe while an operation is in flight.

`tls-available?` reports whether OpenSSL 1.1.1 or newer was linked for a
supported native build. The public module is qualified for native JIT and AOT
on POSIX. Windows, hosted VM, and WebAssembly module use is unsupported and
unqualified. C builds without the native TLS backend return the explicit
unavailable error; there is never a plaintext fallback.
