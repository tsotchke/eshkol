# Eshkol

Eshkol is a high-performance Scheme-like language that compiles to C, designed specifically for scientific computing applications.

## Overview

Eshkol (tscheme) combines the elegant syntax of Scheme with the performance of C. Key features include:

- Scheme/LISP syntax with optional static type annotations
- Direct compilation to efficient C code
- High-performance memory management
- Scientific computing optimizations (SIMD, parallel processing)
- Full C interoperability
- MIT Scheme compatibility

## Project Status

This project is in early development. The initial implementation focuses on creating a minimal but functional compiler with high-performance design principles.

## Building

```bash
mkdir -p build
cd build
cmake ..
make
```

## Usage

```bash
# Compile and run
./eshkol input.esk

# Compile to C
./eshkol input.esk output.c
```

## File Extensions

- `.esk` - Eshkol source files
- `.eskh` - Eshkol header files
- `.eskir` - Intermediate representation
- `.eskc` - Generated C code
- `.esklib` - Compiled library
- `.eskmod` - Module file
- `.eskproj` - Project configuration
- `.eskpkg` - Package definition

## Examples

See the `examples/` directory for sample programs.

## License

MIT
