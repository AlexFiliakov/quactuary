# Core Classes - Detailed Class Diagrams

This document provides detailed class diagrams for the core components of the quActuary system.

## Backend System Classes

```mermaid
classDiagram
    class Backend {
        <<interface>>
        +name: str
        +is_quantum: bool
        +execute(circuit) Result
        +optimize(objective) OptimizationResult
    }
    
    class ClassicalBackend {
        +name: str = "classical"
        +is_quantum: bool = False
    }
    
    class BackendManager {
        -_backend: Backend
        -_original_backend: Backend
        +__init__(backend: Backend)
        +execute(circuit) Result
        +optimize(objective) OptimizationResult
        +is_quantum() bool
        +backend_name() str
    }
    
    class QuantumBackend {
        -_backend: qiskit.Backend
        +name: str
        +is_quantum: bool = True
        +execute(circuit) Result
        +optimize(objective) OptimizationResult
    }
    
    Backend <|-- ClassicalBackend
    Backend <|-- QuantumBackend
    Backend <-- BackendManager : wraps
    
    class BackendContext {
        -_manager: BackendManager
        -_previous_backend: Backend
        +__enter__()
        +__exit__()
    }
    
    BackendManager <-- BackendContext : uses
```

## Pricing System Classes

```mermaid
classDiagram
    class PricingModel {
        -strategy: PricingStrategy
        +__init__(strategy: PricingStrategy)
        +calculate_premium(portfolio, n_simulations) PremiumResult
        +calculate_risk_measures(losses) dict
        +set_strategy(strategy: PricingStrategy)
    }
    
    class PricingStrategy {
        <<abstract>>
        +simulate_losses(portfolio, n_simulations) ndarray
        +calculate_statistics(losses) dict
    }
    
    class ClassicalPricingStrategy {
        -backend: BackendManager
        +simulate_losses(portfolio, n_simulations) ndarray
        +calculate_statistics(losses) dict
        -_simulate_inforce(inforce, n_simulations) ndarray
    }
    
    class QuantumPricingStrategy {
        -backend: BackendManager
        +simulate_losses(portfolio, n_simulations) ndarray
        +calculate_statistics(losses) dict
        -_quantum_amplitude_estimation() float
    }
    
    class PremiumResult {
        +pure_premium: float
        +risk_load: float
        +total_premium: float
        +confidence_interval: tuple
        +risk_measures: dict
    }
    
    PricingStrategy <|-- ClassicalPricingStrategy
    PricingStrategy <|-- QuantumPricingStrategy
    PricingModel --> PricingStrategy : uses
    PricingModel ..> PremiumResult : creates
```

## Business Object Classes

```mermaid
classDiagram
    class PolicyTerms {
        <<frozen dataclass>>
        +limit: float
        +attachment: float
        +coinsurance: float
        +corridor_start: float
        +corridor_end: float
        +validate() bool
        +apply_to_loss(loss: float) float
    }
    
    class Inforce {
        +exposure_count: float
        +avg_exposure_value: float
        +policy_terms: PolicyTerms
        +frequency_dist: FrequencyDistribution
        +severity_dist: SeverityDistribution
        +compound_dist: CompoundDistribution
        +total_exposure_value() float
        +simulate_losses(n_simulations) ndarray
    }
    
    class Portfolio {
        -_inforces: List[Inforce]
        +add_inforce(inforce: Inforce)
        +remove_inforce(index: int)
        +get_inforces() List[Inforce]
        +total_exposure() float
        +simulate_losses(n_simulations) ndarray
    }
    
    class PolicyResult {
        +ground_up_loss: float
        +capped_loss: float
        +after_corridor: float
        +final_loss: float
        +retained_loss: float
    }
    
    PolicyTerms <-- Inforce : has
    Inforce <-- Portfolio : contains
    PolicyTerms ..> PolicyResult : produces
```

## Distribution Base Classes

```mermaid
classDiagram
    class Distribution {
        <<abstract>>
        #_params: dict
        +sample(size: int) ndarray
        +pdf(x: ndarray) ndarray
        +cdf(x: ndarray) ndarray
        +mean() float
        +variance() float
        +get_params() dict
        +set_params(**kwargs)
        #_validate_params()
    }
    
    class FrequencyDistribution {
        <<abstract>>
        +pmf(k: int) float
        +sample(size: int) ndarray
    }
    
    class SeverityDistribution {
        <<abstract>>
        +ppf(q: ndarray) ndarray
        +sample(size: int) ndarray
    }
    
    class ParametricDistribution {
        <<abstract>>
        #_param_bounds: dict
        +fit(data: ndarray) dict
        +log_likelihood(data: ndarray) float
        #_check_param_bounds()
    }
    
    Distribution <|-- FrequencyDistribution
    Distribution <|-- SeverityDistribution
    Distribution <|-- ParametricDistribution
    ParametricDistribution <|-- FrequencyDistribution
    ParametricDistribution <|-- SeverityDistribution
```

## Performance Infrastructure Classes

```mermaid
classDiagram
    class ParallelConfig {
        <<dataclass>>
        +n_workers: int
        +chunk_size: int
        +backend: str
        +timeout: float
        +error_handling: str
        +progress_bar: bool
        +memory_limit: float
        +validate() bool
    }
    
    class ParallelSimulator {
        -config: ParallelConfig
        -memory_manager: MemoryManager
        +__init__(config: ParallelConfig)
        +simulate(func, tasks, **kwargs) List
        -_run_parallel() List
        -_run_serial() List
        -_handle_error(error) Any
    }
    
    class MemoryManager {
        -threshold_mb: float
        -safety_factor: float
        +__init__(threshold_mb, safety_factor)
        +get_available_memory() float
        +calculate_batch_size(item_size, n_items) int
        +monitor_memory() MemoryStatus
        +should_gc() bool
    }
    
    class OptimizationSelector {
        -memory_threshold: float
        -parallel_threshold: int
        +analyze_portfolio(portfolio) OptimizationProfile
        +select_optimizations(profile) OptimizationConfig
        -_estimate_memory_usage() float
        -_select_parallel_config() ParallelConfig
    }
    
    class OptimizationProfile {
        <<dataclass>>
        +portfolio_size: int
        +total_simulations: int
        +memory_estimate_mb: float
        +available_memory_mb: float
        +cpu_count: int
        +has_gpu: bool
    }
    
    class OptimizationConfig {
        <<dataclass>>
        +use_vectorization: bool
        +use_jit: bool
        +use_qmc: bool
        +parallel_config: ParallelConfig
        +memory_optimization: str
        +batch_size: int
    }
    
    ParallelSimulator --> ParallelConfig : uses
    ParallelSimulator --> MemoryManager : uses
    OptimizationSelector --> OptimizationProfile : creates
    OptimizationSelector --> OptimizationConfig : creates
    OptimizationConfig --> ParallelConfig : contains
```

## Class Interaction Sequence

```mermaid
sequenceDiagram
    participant User
    participant PricingModel
    participant Strategy as PricingStrategy
    participant Backend as BackendManager
    participant Portfolio
    participant Optimizer as OptimizationSelector
    participant Parallel as ParallelSimulator
    
    User->>PricingModel: calculate_premium(portfolio, n_sims)
    PricingModel->>Strategy: simulate_losses(portfolio, n_sims)
    
    Strategy->>Optimizer: analyze_portfolio(portfolio)
    Optimizer->>Optimizer: select_optimizations()
    Optimizer-->>Strategy: OptimizationConfig
    
    Strategy->>Backend: check is_quantum()
    
    alt Quantum Backend
        Strategy->>Backend: execute(quantum_circuit)
    else Classical Backend
        Strategy->>Parallel: simulate(loss_function, tasks)
        Parallel->>Portfolio: simulate_losses()
        Portfolio->>Portfolio: aggregate_inforce_losses()
    end
    
    Strategy-->>PricingModel: loss_array
    PricingModel->>PricingModel: calculate_risk_measures()
    PricingModel-->>User: PremiumResult
```

## Design Patterns Used

### 1. Strategy Pattern (Pricing)
- `PricingModel` uses `PricingStrategy` interface
- Allows runtime switching between Classical/Quantum strategies

### 2. Facade Pattern (Backend)
- `BackendManager` provides unified interface to different backends
- Hides complexity of quantum/classical differences

### 3. Builder Pattern (Portfolio)
- `Portfolio` built incrementally with `add_inforce()`
- Complex objects constructed step-by-step

### 4. Template Method (Distributions)
- Base `Distribution` class defines algorithm structure
- Subclasses implement specific distribution logic

### 5. Singleton Pattern (Backend)
- Global backend instance via `get_backend()`
- Ensures consistent backend across application