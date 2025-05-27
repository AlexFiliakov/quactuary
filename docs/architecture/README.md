# quActuary Architecture Documentation

This directory contains comprehensive architectural diagrams and documentation for the quActuary quantum-powered actuarial modeling framework.

## Overview

quActuary is a sophisticated Python package that combines classical and quantum computing approaches for actuarial calculations. The architecture is designed for:

- **Performance**: Adaptive optimization based on problem size and available resources
- **Scalability**: From small in-memory calculations to massive streaming simulations
- **Flexibility**: Seamless switching between quantum and classical backends
- **Extensibility**: Plugin architecture for distributions and computation strategies

## Documentation Structure

### 📊 High-Level Architecture
- **[Context Diagram](./context_diagram.md)**: System-wide view showing external integrations and major components
- **[Module Overview](./module_overview.md)**: Detailed module relationships and dependencies

### 🔧 Detailed Class Diagrams
- **[Core Classes](./class_diagrams/core_classes.md)**: Backend, Pricing, and Business Object classes
- **[Data Models](./class_diagrams/data_models.md)**: Distribution hierarchy and data structures
- **[Service Layer](./class_diagrams/service_layer.md)**: MCP server, CLI tools, and API interfaces

## Quick Navigation

### By Component

#### Backend System
- Quantum/Classical backend abstraction → [Core Classes](./class_diagrams/core_classes.md#backend-system-classes)
- Backend switching mechanism → [Context Diagram](./context_diagram.md#key-data-flows)

#### Distributions
- Distribution class hierarchy → [Data Models](./class_diagrams/data_models.md#distribution-hierarchy)
- Compound distributions → [Data Models](./class_diagrams/data_models.md#compound-and-extended-distributions)
- Distribution factory pattern → [Data Models](./class_diagrams/data_models.md#distribution-factory-pattern)

#### Pricing Engine
- Pricing strategy pattern → [Core Classes](./class_diagrams/core_classes.md#pricing-system-classes)
- Risk calculations → [Service Layer](./class_diagrams/service_layer.md#api-service-layer)

#### Performance Infrastructure
- Parallel processing → [Core Classes](./class_diagrams/core_classes.md#performance-infrastructure-classes)
- Memory management → [Module Overview](./module_overview.md#performance-module-interactions)
- Optimization selection → [Context Diagram](./context_diagram.md#performance-optimization-flow)

#### External Interfaces
- MCP Server → [Service Layer](./class_diagrams/service_layer.md#mcp-server-architecture)
- CLI Tools → [Service Layer](./class_diagrams/service_layer.md#cli-tools-architecture)
- Python API → [Service Layer](./class_diagrams/service_layer.md#api-service-layer)

### By Use Case

#### Setting Up a Portfolio
1. Business objects structure → [Core Classes](./class_diagrams/core_classes.md#business-object-classes)
2. Portfolio builder pattern → [Service Layer](./class_diagrams/service_layer.md#api-service-layer)
3. Data validation → [Data Models](./class_diagrams/data_models.md#type-system-and-validation)

#### Running Simulations
1. Optimization selection → [Context Diagram](./context_diagram.md#performance-optimization-flow)
2. Simulation execution → [Module Overview](./module_overview.md#performance-module-interactions)
3. Result processing → [Data Models](./class_diagrams/data_models.md#data-transfer-objects)

#### Integrating with External Systems
1. MCP protocol → [Context Diagram](./context_diagram.md#external-integrations)
2. Tool implementation → [Service Layer](./class_diagrams/service_layer.md#mcp-server-architecture)
3. Error handling → [Service Layer](./class_diagrams/service_layer.md#error-handling-and-validation)

## Key Architectural Patterns

### Design Patterns Used
- **Strategy Pattern**: Pricing calculations with pluggable strategies
- **Factory Pattern**: Distribution creation with validation
- **Facade Pattern**: Simplified backend interface
- **Builder Pattern**: Incremental portfolio construction
- **Template Method**: Base distribution algorithm structure

### Performance Strategies
1. **Adaptive Optimization**: Dynamic strategy selection based on problem characteristics
2. **Streaming Algorithms**: Handle datasets larger than available memory
3. **Parallel Processing**: Multi-core utilization with work stealing
4. **JIT Compilation**: Numba acceleration for critical paths
5. **Quantum Acceleration**: Optional quantum backend for suitable problems

## Architecture Principles

### 1. Separation of Concerns
- Clear boundaries between business logic, computation, and infrastructure
- Modular design enabling independent component evolution

### 2. Performance First
- Optimization at every layer from algorithm selection to memory management
- Benchmarking and profiling built into development workflow

### 3. Fail-Safe Design
- Graceful degradation from parallel to serial execution
- Comprehensive error handling with recovery strategies

### 4. Extensibility
- Plugin architecture for new distributions
- Strategy pattern for alternative implementations
- Configuration-driven behavior

## Getting Started

For developers new to the codebase:

1. Start with the [Context Diagram](./context_diagram.md) for system overview
2. Review [Module Overview](./module_overview.md) for code organization
3. Deep dive into specific components via class diagrams
4. Refer to [CLAUDE.md](/mnt/c/Users/alexf/OneDrive/Documents/Projects/quActuary/CLAUDE.md) for development guidelines

## Maintenance

These diagrams are generated using Mermaid and should be updated when:
- Adding new modules or significant classes
- Changing module dependencies
- Modifying external interfaces
- Implementing new patterns or strategies

See [mermaid.md](../../.claude/commands/project_management/mermaid.md) for diagram maintenance guidelines.