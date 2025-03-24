# Eshkol Documentation

This directory contains comprehensive documentation for the Eshkol programming language.

## Overview

Eshkol is a high-performance Scheme-like language that compiles to C, designed specifically for scientific computing and AI applications. It combines the elegance and expressiveness of Scheme with the performance of C and modern language features.

## Documentation Sections

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

### [Planning](planning/README.md)

Documentation related to the planning and design of Eshkol:

- [Initial Design Conversation](planning/initial_design_conversation.md): The initial design discussion for Eshkol

### Architecture

Documentation related to the architecture of Eshkol (coming soon).

### Components

Documentation related to the components of Eshkol (coming soon).

### Development

Documentation related to the development of Eshkol (coming soon).

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
