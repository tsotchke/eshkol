# Scheme Compatibility Documentation

This directory contains documentation related to Eshkol's compatibility with the Scheme programming language.

## Overview

Eshkol aims to be compatible with both R5RS and R7RS-small standards, allowing developers to leverage existing Scheme code and knowledge. This compatibility is being implemented in phases, with each phase focusing on a specific set of features.

## Documents

### [Master Tracking](MASTER_TRACKING.md)
Central index for all Scheme compatibility documentation, including overall implementation status and progress tracking.

### [Implementation Plan](IMPLEMENTATION_PLAN.md)
Detailed phased approach to implementing Scheme compatibility, including priority levels for different features and implementation strategy.

### [Specification](SPECIFICATION.md)
Definitive reference for Scheme compatibility in Eshkol, including standards followed and intentional deviations.

### [Registry](REGISTRY.md)
Tracks implementation status of all Scheme functions, syntax, and features, including detailed compliance information.

### [Dependencies](DEPENDENCIES.md)
Maps dependencies between Scheme features, including function-level dependencies and implementation order considerations.

### [Evolution](EVOLUTION.md)
Future plans for Scheme compatibility beyond basic implementation, including performance optimizations and extensions.

### [Known Issues](KNOWN_ISSUES.md)
Tracks known issues, limitations, and compatibility notes, including workarounds and planned improvements.

### [MCP Tools for Scheme](MCP_TOOLS_FOR_SCHEME.md)
Documentation for MCP tools available for analyzing Scheme compatibility, including tools for analyzing recursion, bindings, and lambda captures.

## Function Status

The `function_status` directory contains detailed status information for each function group:

- [Pairs and Lists](function_status/pairs_and_lists.md): Status of core list operations

## Phase Tracking

The `phase_tracking` directory contains detailed tracking information for each implementation phase:

- [Phase 1](phase_tracking/phase1.md): Core Data Types and Fundamental Operations

## Templates

The `templates` directory contains templates for documenting and implementing Scheme functions:

- [Function Documentation Template](templates/function_documentation_template.md): Template for documenting Scheme functions
- [Function Implementation Template](templates/function_implementation_template.c): Template for implementing Scheme functions in C
- [Function Test Template](templates/function_test_template.c): Template for testing Scheme functions

## Implementation Phases

Eshkol's Scheme compatibility is being implemented in phases:

### Phase 1: Core Data Types and Fundamental Operations
- Pairs and Lists
- Type Predicates
- Equality Predicates
- Basic Arithmetic
- Extended Pair Operations

### Phase 2: List Processing and Control Flow
- List Processing Functions
- Control Flow Constructs
- Boolean Operations
- Quoting and Evaluation
- Basic I/O

### Phase 3: Higher-Order Functions and Data Structures
- Higher-Order Functions
- Vectors
- Strings
- Characters
- Symbols

### Phase 4: I/O and System Interface
- File I/O
- Port Operations
- String Ports
- System Interface
- Error Handling

### Phase 5: Advanced Features
- Macros
- Continuations
- Dynamic Binding
- Eval and Apply
- Libraries and Modules

## MCP Tools for Scheme Analysis

Eshkol provides several MCP tools for analyzing Scheme code:

- **[analyze-scheme-recursion](MCP_TOOLS_FOR_SCHEME.md#analyze-scheme-recursion)**: Analyzes mutual recursion and lambda captures in Scheme code using AST-based parsing
- **[analyze-tscheme-recursion](MCP_TOOLS_FOR_SCHEME.md#analyze-tscheme-recursion)**: Uses improved TScheme parser for recursion analysis
- **[analyze-bindings](MCP_TOOLS_FOR_SCHEME.md#analyze-bindings)**: Analyzes variable bindings in Scheme code
- **[analyze-binding-access](MCP_TOOLS_FOR_SCHEME.md#analyze-binding-access)**: Examines how bindings are used in Scheme code

These tools can help identify issues with Scheme code, mutual recursion, and other aspects of Scheme compatibility. See the [MCP Tools for Scheme](MCP_TOOLS_FOR_SCHEME.md) document for more information.

## Test Coverage

See the [Test Coverage](test_coverage.md) document for information on test coverage for all implemented functions.

## Getting Involved

We welcome contributions to Eshkol's Scheme compatibility implementation. If you're interested in contributing, please see the [Future Roadmap](../vision/FUTURE_ROADMAP.md) document for information on how to get involved.
