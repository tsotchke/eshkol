# Third-Party Notices

Eshkol is licensed under the MIT License. Release packages also contain
compiled portions of the projects below. Exact upstream license texts are
shipped in the package `licenses/` directory and are required by the release
artifact verifier.

| Component | Pinned source | License |
|---|---|---|
| PCRE2 | `f454e231fe5006dd7ff8f4693fd2b8eb94333429` | BSD-3-Clause WITH PCRE2-exception; bundled SLJIT is BSD-2-Clause |
| SQLite amalgamation 3.53.3 | SHA3-256 `d45c688a8cb23f68611a894a756a12d7eb6ab6e9e2468ca70adbeab3808b5ab9` | Public domain |
| zlib 1.3.1 | `51b7f2abdade71cd9bb0e7a373ef2610ec6f9daf` | Zlib |
| Tree-sitter 0.26.8 | `cd5b087cd9f45ca6d93ab1954f6b7c8534f324d2` | MIT; bundled Unicode data carries the Unicode/ICU notices shipped separately |
| tree-sitter-javascript | `58404d8cf191d69f2674a8fd507bd5776f46cb11` | MIT |
| tree-sitter-typescript and TSX | `75b3874edb2dc714fb1fd77a32013d0f8699989f` | MIT |
| tree-sitter-python | `26855eabccb19c6abf499fbc5b8dc7cc9ab8bc64` | MIT |
| tree-sitter-rust | `77a3747266f4d621d0757825e6b11edcbf991ca5` | MIT |
| tree-sitter-go | `2346a3ab1bb3857b48b29d779a1ef9799a248cd7` | MIT |
| tree-sitter-c | `b780e47fc780ddc8da13afa35a3f4ed5c157823d` | MIT |
| tree-sitter-cpp | `8b5b49eb196bec7040441bee33b2c9a4838d6967` | MIT |
| tree-sitter-java | `e10607b45ff745f5f876bfa3e94fbcc6b44bdc11` | MIT |
| tree-sitter-ruby | `ad907a69da0c8a4f7a943a7fe012712208da6dee` | MIT |
| tree-sitter-bash | `a06c2e4415e9bc0346c6b86d401879ffb44058f7` | MIT |
| Yoga 3.2.1 | `042f5013152eb81c1552dec945b88f7b95ca350f` | MIT |
| curl (Linux packages only) | `a05f34973e6c4bb629d018f7cb51487be1c904d8` | curl license |

Linux packages link against the platform OpenSSL libraries at final AOT link
time; they do not redistribute OpenSSL binaries. Windows system libraries and
Apple frameworks are likewise supplied by their operating-system SDKs and are
not redistributed by Eshkol.

SQLite has been dedicated to the public domain by its authors. The staged
SQLite notice records the public-domain declaration and the canonical upstream
license page: <https://www.sqlite.org/copyright.html>.
