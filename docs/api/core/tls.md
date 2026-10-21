# `core.tls`

`core.tls` currently exposes optional native OpenSSL client and server context
construction. A positive context identifier is an exact integer no greater
than `9007199254740991`; identifiers are never reused. C failures use `-1` for
system or TLS setup failure, `-3` when native TLS is unavailable, and `-4` for
invalid input or a stale identifier. Scheme returns a `tls-error` vector for
negative results.

`(tls-client-context #f)` loads the platform default trust paths. Passing a CA
file replaces that trust source. The client context always requires peer
certificate verification. `(tls-server-context certificate-file private-key-file)`
loads a PEM certificate chain and private key and fails if they do not match.
`tls-context-close` closes an identifier once; repeated close returns `-4`.

The public context boundary does not yet provide TLS listeners, sessions,
handshakes, or encrypted reads and writes. `tls-available?` reports whether
OpenSSL context construction is linked for the current native build. Windows,
VM, and WebAssembly TLS are currently unavailable; no plaintext fallback is
provided.
