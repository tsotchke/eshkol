# Scheme Compatibility Documentation

This directory contains documentation related to Eshkol's compatibility with the Scheme programming language.

## Quick Links

- [Progress Dashboard](PROGRESS_DASHBOARD.md): High-level overview of Scheme compatibility progress
- [Known Issues](KNOWN_ISSUES.md): Current issues and limitations
- [Implementation Plan](IMPLEMENTATION_PLAN.md): Phased approach to implementation
- [Master Tracking](MASTER_TRACKING.md): Central index for all documentation

## Overview

Eshkol aims to be compatible with both R5RS and R7RS-small standards, allowing developers to leverage existing Scheme code and knowledge. This compatibility is being implemented in phases, with each phase focusing on a specific set of features.

## Documents

### Core Documentation

- [Progress Dashboard](PROGRESS_DASHBOARD.md): High-level overview of Scheme compatibility progress
- [Master Tracking](MASTER_TRACKING.md): Central index for all Scheme compatibility documentation
- [Implementation Plan](IMPLEMENTATION_PLAN.md): Detailed phased approach to implementing Scheme compatibility
- [Specification](SPECIFICATION.md): Definitive reference for Scheme compatibility in Eshkol
- [Registry](REGISTRY.md): Tracks implementation status of all Scheme functions, syntax, and features
- [Dependencies](DEPENDENCIES.md): Maps dependencies between Scheme features
- [Evolution](EVOLUTION.md): Future plans for Scheme compatibility beyond basic implementation
- [Known Issues](KNOWN_ISSUES.md): Tracks known issues, limitations, and compatibility notes
- [MCP Tools for Scheme](MCP_TOOLS_FOR_SCHEME.md): Documentation for MCP tools available for analyzing Scheme compatibility

### Implementation Roadmaps

- [Type Predicates Roadmap](roadmaps/type_predicates_roadmap.md): Implementation plan for type predicates
- [Equality Predicates Roadmap](roadmaps/equality_predicates_roadmap.md): Implementation plan for equality predicates
- [List Processing Roadmap](roadmaps/list_processing_roadmap.md): Implementation plan for list processing functions
- [Higher-Order Functions Roadmap](roadmaps/higher_order_functions_roadmap.md): Implementation plan for higher-order functions

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
- [Type Predicate Documentation Template](templates/type_predicate_documentation_template.md): Template for documenting type predicates

## Example Files

The `examples` directory contains example files demonstrating the use of Scheme features in Eshkol:

- [Type Predicates](../../examples/type_predicates.esk): Demonstrates the use of type predicates
- [Equality Predicates](../../examples/equality_predicates.esk): Demonstrates the use of equality predicates
- [List Operations](../../examples/list_operations.esk): Demonstrates the use of list operations
- [Higher-Order Functions](../../examples/higher_order_functions.esk): Demonstrates the use of higher-order functions
- [Function Composition](../../examples/function_composition.esk): Demonstrates function composition
- [Mutual Recursion](../../examples/mutual_recursion.esk): Demonstrates mutual recursion

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
- **[analyze-binding-lifetime](MCP_TOOLS_FOR_SCHEME.md#analyze-binding-lifetime)**: Analyzes when bindings are created and destroyed
- **[analyze-lambda-captures](MCP_TOOLS_FOR_SCHEME.md#analyze-lambda-captures)**: Analyzes free/bound variables and closure environments
- **[visualize-closure-memory](MCP_TOOLS_FOR_SCHEME.md#visualize-closure-memory)**: Visualizes how closures are represented in memory
- **[visualize-binding-flow](MCP_TOOLS_FOR_SCHEME.md#visualize-binding-flow)**: Tracks binding values through transformation stages

These tools can help identify issues with Scheme code, mutual recursion, and other aspects of Scheme compatibility. See the [MCP Tools for Scheme](MCP_TOOLS_FOR_SCHEME.md) document for more information.

## Test Coverage

See the [Test Coverage](test_coverage.md) document for information on test coverage for all implemented functions.

## Getting Involved

We welcome contributions to Eshkol's Scheme compatibility implementation. If you're interested in contributing, please see the [Future Roadmap](../vision/FUTURE_ROADMAP.md) document for information on how to get involved.
