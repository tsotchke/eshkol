# Eshkol Documentation

This directory contains comprehensive documentation for the Eshkol programming language.

## Overview

Eshkol is a high-performance Scheme-like language that compiles to C, designed specifically for scientific computing and AI applications. It combines the elegance and expressiveness of Scheme with the performance of C and modern language features.

## Documentation Sections

### [Core Language](aidocs/README.md)

Documentation covering core language features and implementation details:

- [Memory Management](aidocs/MEMORY_MANAGEMENT.md): Arena-based memory management in Eshkol
- [Closures and Globals](aidocs/CLOSURES_AND_GLOBALS.md): Closure implementation and global variable handling
- [Function Composition](aidocs/FUNCTION_COMPOSITION.md): Implementation of function composition
- [Compilation Guide](aidocs/COMPILATION_GUIDE.md): Guide to compiling Eshkol programs
- [Compiler Architecture](aidocs/COMPILER_ARCHITECTURE.md): Architecture of the Eshkol compiler

### [Vision](vision/README.md)

Documentation outlining the vision, goals, and technical details of the Eshkol programming language:

- [Purpose and Vision](vision/PURPOSE_AND_VISION.md): The core purpose and long-term vision for Eshkol
- [Differentiation Analysis](vision/DIFFERENTIATION_ANALYSIS.md): How Eshkol compares to other languages
- [AI Focus](vision/AI_FOCUS.md): Eshkol's unique advantages for AI development
- [Scientific Computing](vision/SCIENTIFIC_COMPUTING.md): Eshkol's scientific computing capabilities
- [Technical White Paper](vision/TECHNICAL_WHITE_PAPER.md): Technical details of Eshkol's implementation
- [Future Roadmap](vision/FUTURE_ROADMAP.md): The planned evolution of Eshkol

### [Scheme Compatibility](scheme_compatibility/README.md)

Documentation related to Eshkol's compatibility with the Scheme programming language:

- [Master Tracking](scheme_compatibility/MASTER_TRACKING.md): Overall implementation status and progress tracking
- [Implementation Plan](scheme_compatibility/IMPLEMENTATION_PLAN.md): Phased approach to implementing Scheme compatibility
- [Specification](scheme_compatibility/SPECIFICATION.md): Definitive reference for Scheme compatibility in Eshkol
- [Registry](scheme_compatibility/REGISTRY.md): Tracks implementation status of all Scheme functions
- [Dependencies](scheme_compatibility/DEPENDENCIES.md): Maps dependencies between Scheme features
- [Evolution](scheme_compatibility/EVOLUTION.md): Future plans for Scheme compatibility
- [Known Issues](scheme_compatibility/KNOWN_ISSUES.md): Tracks known issues and limitations
- [MCP Tools for Scheme](scheme_compatibility/MCP_TOOLS_FOR_SCHEME.md): Documentation for MCP tools for Scheme compatibility analysis

### [Type System](type_system/README.md)

Documentation related to Eshkol's type system:

- [Type System Overview](type_system/TYPE_SYSTEM.md): Comprehensive overview of Eshkol's type system
- [Influences](type_system/INFLUENCES.md): Languages and systems that influenced Eshkol's type system
- [Scientific Computing and AI](type_system/SCIENTIFIC_COMPUTING_AND_AI.md): How the type system enables scientific computing and AI
- [Scheme Compatibility](type_system/SCHEME_COMPATIBILITY.md): How Eshkol maintains Scheme compatibility while adding types
- [Automatic Differentiation](type_system/AUTODIFF.md): The synergy between the type system and automatic differentiation
- [MCP Tools for Autodiff](type_system/MCP_TOOLS_FOR_AUTODIFF.md): Documentation for MCP tools for autodiff analysis

### [Tutorials](tutorials/README.md)

Step-by-step guides and practical examples for learning and using Eshkol:

- [Type System Tutorial](tutorials/TYPE_SYSTEM_TUTORIAL.md): A practical guide to using Eshkol's type system

### [Reference](reference/README.md)

Comprehensive and detailed information about Eshkol's syntax, semantics, and features:

- [Type System Reference](reference/TYPE_SYSTEM_REFERENCE.md): A comprehensive reference for all type-related syntax and semantics

### [Planning](planning/README.md)

Documentation related to the planning and design of Eshkol:

- [Initial Design Conversation](planning/initial_design_conversation.md): The initial design discussion for Eshkol

### Architecture

Documentation related to the architecture of Eshkol (coming soon).

### Components

Documentation related to the components of Eshkol (coming soon).

### Development

Documentation related to the development of Eshkol (coming soon).

## MCP Tools

Eshkol provides several MCP tools for analyzing and debugging code:

- **Type Analysis Tools**
  - [analyze-types](type_system/MCP_TOOLS_FOR_AUTODIFF.md#analyze-types): Analyzes type inference and type checking for Eshkol files
  - [analyze-codegen](type_system/MCP_TOOLS_FOR_AUTODIFF.md#analyze-codegen): Identifies code generation issues

- **Binding and Lambda Analysis Tools**
  - [analyze-bindings](scheme_compatibility/MCP_TOOLS_FOR_SCHEME.md#analyze-bindings): Analyzes variable bindings in Scheme code
  - [analyze-lambda-captures](type_system/MCP_TOOLS_FOR_AUTODIFF.md#analyze-lambda-captures): Analyzes closure environments and variable captures
  - [analyze-binding-lifetime](type_system/MCP_TOOLS_FOR_AUTODIFF.md#analyze-binding-lifetime): Tracks binding creation and destruction
  - [analyze-binding-access](type_system/MCP_TOOLS_FOR_AUTODIFF.md#analyze-binding-access): Examines how bindings are used

- **Visualization Tools**
  - [visualize-ast](type_system/MCP_TOOLS_FOR_AUTODIFF.md#visualize-ast): Generates AST visualizations
  - [visualize-closure-memory](type_system/MCP_TOOLS_FOR_AUTODIFF.md#visualize-closure-memory): Shows closure memory representation
  - [visualize-binding-flow](type_system/MCP_TOOLS_FOR_AUTODIFF.md#visualize-binding-flow): Tracks binding values through transformation stages

- **Recursion Analysis Tools**
  - [analyze-mutual-recursion](scheme_compatibility/MCP_TOOLS_FOR_SCHEME.md#analyze-mutual-recursion): Analyzes mutual recursion patterns
  - [analyze-scheme-recursion](scheme_compatibility/MCP_TOOLS_FOR_SCHEME.md#analyze-scheme-recursion): Analyzes recursion in Scheme code
  - [analyze-tscheme-recursion](scheme_compatibility/MCP_TOOLS_FOR_SCHEME.md#analyze-tscheme-recursion): Uses improved TScheme parser for recursion analysis

## Key Features

- **Homoiconicity with Performance**: Eshkol preserves the homoiconicity of Lisp/Scheme (code as data) while delivering C-level performance.
- **Gradual Typing**: Eshkol embraces optional static typing, allowing developers to start with dynamic typing for rapid prototyping and gradually add type annotations for performance and safety.
- **Scientific Computing as a First-Class Citizen**: Eshkol integrates vector operations, automatic differentiation, and SIMD optimization directly into the language core.
- **Memory Efficiency**: Eshkol's arena-based memory management system provides deterministic performance without the unpredictable pauses of garbage collection.
- **Seamless C Interoperability**: Eshkol compiles directly to C, enabling seamless integration with the vast ecosystem of C libraries and tools.
- **Scheme Compatibility**: Eshkol maintains compatibility with standard Scheme (R5RS and R7RS), allowing developers to leverage existing Scheme code and knowledge.

## Implementation Phases

Eshkol's development is being implemented in phases:

### Phase 1: Core Language Implementation (2025)
- Core language features
- Scheme compatibility
- Basic tooling
- Documentation

### Phase 2: Scientific Computing Extensions (2025-2026)
- Vector and matrix operations
- Automatic differentiation
- Numerical computing
- Scientific libraries
- Interoperability

### Phase 3: AI-Specific Features (2026-2027)
- Neural network support
- Neuro-symbolic AI
- Reinforcement learning
- AI development tools
- AI libraries

### Phase 4: Ecosystem Development (2027-2028)
- Package ecosystem
- Advanced tooling
- Community infrastructure
- Learning resources
- Integration with other ecosystems

### Phase 5: Advanced Features and Optimizations (2028+)
- Advanced language features
- Performance optimizations
- Research areas
- New domains

## Getting Involved

We welcome contributions from researchers, engineers, and enthusiasts who share our vision for Eshkol. See the [Future Roadmap](vision/FUTURE_ROADMAP.md) document for information on how to get involved.
