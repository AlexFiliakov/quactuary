# High-Level System Context Diagram

This diagram shows the overall architecture of the quActuary quantum-powered actuarial modeling framework.

```mermaid
flowchart TB
    %% External Systems
    User[/"Users<br/>(Actuaries, Data Scientists)"/]
    Claude[/"Claude Code<br/>(AI Assistant)"/]
    QiskitCloud[/"IBM Quantum Cloud<br/>(Optional)"/]
    
    %% Main System
    System[["quActuary System<br/>Quantum-Powered Actuarial Tools"]]
    
    %% Core Components
    subgraph Core["Core Components"]
        Backend["Backend Manager<br/>(Quantum/Classical Switch)"]
        Pricing["Pricing Engine<br/>(Risk Calculations)"]
        Distributions["Distribution Library<br/>(Probability Models)"]
        Business["Business Objects<br/>(Portfolio/Policy)"]
    end
    
    %% Performance Layer
    subgraph Performance["Performance Infrastructure"]
        Parallel["Parallel Processing<br/>(Multi-core/Threading)"]
        Memory["Memory Management<br/>(Streaming/Batching)"]
        Optimizer["Optimization Selector<br/>(Strategy Selection)"]
        JIT["JIT Compilation<br/>(Numba)"]
    end
    
    %% External Interfaces
    subgraph Interfaces["External Interfaces"]
        MCP["MCP Server<br/>(Model Context Protocol)"]
        API["Python API<br/>(Public Interface)"]
        CLI["CLI Tools<br/>(Performance/Benchmarks)"]
    end
    
    %% Data Flow
    User -->|"Python Code"| API
    User -->|"Commands"| CLI
    Claude -->|"MCP Protocol"| MCP
    
    API --> System
    CLI --> System
    MCP --> System
    
    System --> Core
    Core --> Performance
    
    Backend -->|"Quantum Circuits"| QiskitCloud
    
    %% Internal Core Connections
    Pricing --> Backend
    Pricing --> Distributions
    Pricing --> Business
    Distributions --> Backend
    
    %% Performance Connections
    Parallel --> Memory
    Memory --> Optimizer
    Optimizer --> JIT
    Performance --> Backend
    
    %% Styling
    classDef external fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef system fill:#fff3e0,stroke:#e65100,stroke-width:3px
    classDef core fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef perf fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px
    classDef interface fill:#fce4ec,stroke:#880e4f,stroke-width:2px
    
    class User,Claude,QiskitCloud external
    class System system
    class Backend,Pricing,Distributions,Business core
    class Parallel,Memory,Optimizer,JIT perf
    class MCP,API,CLI interface
```

## System Overview

The quActuary system is designed as a modular, high-performance actuarial modeling framework with quantum computing capabilities. The architecture follows these key principles:

### 1. **Layered Architecture**
- **Interface Layer**: Multiple entry points (Python API, CLI, MCP Server)
- **Core Layer**: Business logic and computational models
- **Performance Layer**: Optimization and resource management
- **Backend Layer**: Abstracted quantum/classical computation

### 2. **Key Data Flows**

#### User Interaction Flow
```mermaid
flowchart LR
    A[User Request] --> B{Interface Type}
    B -->|API Call| C[Python API]
    B -->|Command| D[CLI Tool]
    B -->|AI Assistant| E[MCP Server]
    
    C --> F[Core System]
    D --> F
    E --> F
    
    F --> G[Select Backend]
    G -->|Classical| H[NumPy/SciPy]
    G -->|Quantum| I[Qiskit]
    
    H --> J[Results]
    I --> J
```

#### Performance Optimization Flow
```mermaid
flowchart TD
    A[Portfolio Analysis] --> B[Optimization Selector]
    B --> C{Problem Size}
    
    C -->|Small<br/><1GB| D[In-Memory<br/>Vectorized]
    C -->|Medium<br/>1-10GB| E[Batched<br/>Parallel]
    C -->|Large<br/>>10GB| F[Streaming<br/>Online Stats]
    
    D --> G[Execute]
    E --> G
    F --> G
    
    G --> H{Memory OK?}
    H -->|Yes| I[Continue]
    H -->|No| J[Adaptive<br/>Adjustment]
    J --> G
```

### 3. **External Integrations**

- **IBM Quantum Cloud**: Optional quantum backend for hardware execution
- **Claude Code**: AI-powered development assistance via MCP
- **Scientific Python Stack**: NumPy, SciPy, Pandas for computation
- **Qiskit**: Quantum computing framework (v1.4.2)

### 4. **Key Architectural Decisions**

1. **Backend Abstraction**: Seamless switching between quantum and classical computation
2. **Strategy Pattern**: Flexible pricing strategies with runtime selection
3. **Resource Adaptation**: Dynamic optimization based on available resources
4. **Streaming Architecture**: Handle problems larger than available memory
5. **MCP Integration**: Modern AI assistant integration for development

This architecture enables quActuary to scale from simple actuarial calculations to complex portfolio simulations while leveraging quantum acceleration when beneficial.