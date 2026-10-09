---
name: Bug Report
about: Report a bug or unexpected behavior
title: '[BUG] '
labels: bug
assignees: ''
---

## Description
A clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:

1. Create file with content:
```scheme
; Your Eshkol code here
```

2. Run command:
```bash
eshkol-run yourfile.esk -o yourfile && ./yourfile
# or JIT-compile and run in memory, without writing an artifact:
eshkol-run -r yourfile.esk
```

3. See error

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Output
```
Paste any error messages here
```

## Environment
- OS: [e.g., macOS 14.0, Ubuntu 22.04]
- Architecture: [e.g., arm64, x86_64]
- Eshkol Version: [output of `eshkol-run --version`, e.g., Eshkol Compiler v1.3.6-evolve]
- LLVM Version: [output of `llvm-config --version`; Eshkol builds against LLVM 21, e.g., 21.1.8]
- Build capabilities: [output of `eshkol-run --features`, for GPU, BLAS or platform-specific reports]
- Installation Method: [e.g., Homebrew, .deb package, release archive, source build]

## Additional Context
Add any other context about the problem here.
