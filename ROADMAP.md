# Eshkol Development Roadmap

This document outlines the planned development path for Eshkol, including estimated timelines and feature priorities.

## Version Timeline

### Version 0.1.0-alpha (Early Developer Preview) - Current Release

**Focus**: Core language features, basic Scheme compatibility, and scientific computing foundations.

**Key Components**:
- Basic Scheme syntax and core special forms
- Function composition with JIT compilation (partial)
- Optional type annotations and basic type inference
- Vector operations and automatic differentiation
- MCP tools for code analysis

### Version 0.2.0-alpha (Function Composition Update) - Target: Q2 2025

**Focus**: Improving function composition and type system integration.

**Key Components**:
- Fixed function composition issues (SCH-019)
- Complete JIT compilation for N-ary composition
- Improved type system integration
- Enhanced MCP tools
- Performance improvements

### Version 0.3.0-alpha (Type System Update) - Target: Q3 2025

**Focus**: Enhanced type system and Scheme compatibility.

**Key Components**:
- Enhanced type inference
- Better error reporting
- Type predicates implementation
- Equality predicates implementation
- List processing functions

### Version 0.4.0-alpha (Performance Update) - Target: Q4 2025

**Focus**: Performance optimizations and memory management.

**Key Components**:
- Reference counting implementation
- Optimized closure system
- Enhanced SIMD support
- Memory usage optimizations
- Performance benchmarking suite

### Version 0.5.0-beta (Tail Call Update) - Target: Q1 2026

**Focus**: Tail call optimization and recursion support.

**Key Components**:
- Tail call optimization
- Improved recursion support
- Continuations support (partial)
- Performance enhancements
- Expanded standard library

### Version 0.6.0-beta (Continuations Update) - Target: Q2 2026

**Focus**: Continuations and advanced control flow.

**Key Components**:
- Full continuations support
- Advanced control flow features
- Enhanced error handling
- Expanded standard library
- Improved debugging tools

### Version 1.0.0 (Full Release) - Target: Q3-Q4 2026

**Focus**: Production-ready release with complete feature set.

**Key Components**:
- Complete Scheme compatibility
- Full numeric tower
- Hygienic macro system
- Comprehensive standard library
- Production-ready performance
- Complete documentation

## Development Priorities

### Near-Term Priorities (1-3 Months)

1. **Function Composition Improvements**
   - Fix remaining issues with function composition (SCH-019)
   - Complete JIT compilation for N-ary composition
   - Implement comprehensive test suite
   - Optimize performance for complex compositions

2. **Type System Integration**
   - Resolve type inference issues for autodiff functions (SCH-006)
   - Fix vector return type handling (SCH-007)
   - Address type conflicts in generated C code (SCH-008)
   - Improve mutual recursion handling in type inference (SCH-015)

3. **Core Scheme Compatibility**
   - Implement basic type predicates (SCH-014)
   - Add equality predicates (SCH-016)
   - Implement list processing functions (SCH-018)
   - Add higher-order functions (SCH-017)

### Medium-Term Goals (3-6 Months)

1. **Advanced Type System**
   - Complete type system integration with Scheme
   - Implement full type inference for complex patterns
   - Add comprehensive type error reporting
   - Optimize type checking performance

2. **Performance Optimization**
   - Implement reference counting for proper memory management
   - Optimize closure environment representation
   - Improve variable lookup in closures
   - Enhance SIMD optimizations

3. **MCP Tools Enhancement**
   - Improve type analysis tools
   - Enhance binding flow visualization
   - Develop better code generation comparison tools
   - Create performance profiling visualization

### Long-Term Features (6-12 Months)

1. **Tail Call Optimization (SCH-002)**
   - Design and implement proper tail call optimization
   - Integrate with existing recursion patterns
   - Optimize for performance
   - Develop comprehensive test suite

2. **Continuations Support (SCH-005)**
   - Design continuation implementation
   - Implement call/cc and related functions
   - Integrate with existing control flow
   - Test with complex scenarios

3. **Full Numeric Tower (SCH-003)**
   - Implement rational numbers
   - Add complex number support
   - Ensure proper type integration
   - Optimize numeric operations

4. **Hygienic Macro System (SCH-004)**
   - Design macro expansion system
   - Implement hygienic macro support
   - Add syntax-rules or similar
   - Create comprehensive test suite

## Component Roadmaps

### Core Language Features

| Feature | Current Status | Target Version | Priority |
|---------|----------------|----------------|----------|
| Basic Scheme Syntax | 90% Complete | 0.1.0-alpha | Completed |
| Core Special Forms | 85% Complete | 0.1.0-alpha | Completed |
| Lambda Expressions | 80% Complete | 0.1.0-alpha | Completed |
| Lexical Scoping | 80% Complete | 0.1.0-alpha | Completed |
| Function Composition | 75% Complete | 0.2.0-alpha | High |
| Tail Call Optimization | Planned | 0.5.0-beta | Medium |
| Continuations | Planned | 0.6.0-beta | Medium |
| Hygienic Macros | Planned | 1.0.0 | Low |

### Type System

| Feature | Current Status | Target Version | Priority |
|---------|----------------|----------------|----------|
| Optional Type Annotations | 80% Complete | 0.1.0-alpha | Completed |
| Basic Type Inference | 70% Complete | 0.1.0-alpha | Completed |
| Type Checking | 60% Complete | 0.2.0-alpha | High |
| Autodiff Type Integration | 50% Complete | 0.2.0-alpha | High |
| Vector Type Handling | 50% Complete | 0.2.0-alpha | High |
| Type Error Reporting | 40% Complete | 0.3.0-alpha | Medium |
| Polymorphic Types | Planned | 0.3.0-alpha | Medium |
| Type Classes/Traits | Planned | 0.4.0-alpha | Low |

### Scientific Computing

| Feature | Current Status | Target Version | Priority |
|---------|----------------|----------------|----------|
| Vector Operations | 80% Complete | 0.1.0-alpha | Completed |
| Forward-mode Autodiff | 80% Complete | 0.1.0-alpha | Completed |
| Reverse-mode Autodiff | 70% Complete | 0.1.0-alpha | Completed |
| SIMD Optimization | 60% Complete | 0.2.0-alpha | High |
| Matrix Operations | 50% Complete | 0.2.0-alpha | High |
| Gradient-based Optimization | 40% Complete | 0.3.0-alpha | Medium |
| GPU Acceleration | Planned | 0.6.0-beta | Low |
| Distributed Computing | Planned | Post-1.0 | Low |

### Memory Management

| Feature | Current Status | Target Version | Priority |
|---------|----------------|----------------|----------|
| Arena Allocation | 90% Complete | 0.1.0-alpha | Completed |
| Memory Tracking | 80% Complete | 0.1.0-alpha | Completed |
| Object Pools | 70% Complete | 0.1.0-alpha | Completed |
| Reference Counting | Planned | 0.4.0-alpha | High |
| Memory Optimization | Planned | 0.4.0-alpha | High |
| Garbage Collection (Optional) | Planned | Post-1.0 | Low |

### Scheme Compatibility

| Feature | Current Status | Target Version | Priority |
|---------|----------------|----------------|----------|
| Core Data Types | 70% Complete | 0.1.0-alpha | Completed |
| Basic Operations | 65% Complete | 0.1.0-alpha | Completed |
| Type Predicates | Planned | 0.2.0-alpha | High |
| Equality Predicates | Planned | 0.2.0-alpha | High |
| List Processing | Planned | 0.3.0-alpha | High |
| Higher-order Functions | Planned | 0.3.0-alpha | Medium |
| I/O Functions | Planned | 0.4.0-alpha | Medium |
| Full R5RS Compatibility | Planned | 1.0.0 | Low |
| R7RS-small Compatibility | Planned | Post-1.0 | Low |

## How to Contribute

We welcome contributions to help us achieve these roadmap goals! Here are some ways you can help:

1. **Fix Known Issues**: Check the [KNOWN_ISSUES.md](docs/scheme_compatibility/KNOWN_ISSUES.md) file for current limitations and bugs that need fixing.

2. **Implement Planned Features**: Pick a feature from the roadmap that interests you and implement it.

3. **Improve Documentation**: Help us improve our documentation to make Eshkol more accessible to new users.

4. **Write Tests**: Help us improve our test coverage to ensure Eshkol is robust and reliable.

5. **Performance Optimization**: Help us identify and fix performance bottlenecks.

See our [CONTRIBUTING.md](CONTRIBUTING.md) for more details on how to contribute.

## Roadmap Updates

This roadmap is a living document and will be updated regularly as development progresses. Major updates to the roadmap will be announced in the release notes and on the project's GitHub Discussions.

Last updated: April 2, 2025
