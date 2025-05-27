# Data Models and Distribution Classes

This document details the data structures and distribution models used throughout the quActuary system.

## Distribution Hierarchy

```mermaid
classDiagram
    class Distribution {
        <<abstract>>
        #_params: dict
        +sample(size: int) ndarray
        +mean() float
        +variance() float
        +get_params() dict
    }
    
    %% Frequency Distributions
    class FrequencyDistribution {
        <<abstract>>
        +pmf(k: int) float
    }
    
    class Poisson {
        -lambda_: float
        +pmf(k: int) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Binomial {
        -n: int
        -p: float
        +pmf(k: int) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class NegativeBinomial {
        -r: float
        -p: float
        +pmf(k: int) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Geometric {
        -p: float
        +pmf(k: int) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    %% Severity Distributions
    class SeverityDistribution {
        <<abstract>>
        +pdf(x: float) float
        +cdf(x: float) float
        +ppf(q: float) float
    }
    
    class Lognormal {
        -mu: float
        -sigma: float
        +pdf(x: float) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Gamma {
        -alpha: float
        -beta: float
        +pdf(x: float) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Exponential {
        -lambda_: float
        +pdf(x: float) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Pareto {
        -alpha: float
        -x_m: float
        +pdf(x: float) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class Weibull {
        -k: float
        -lambda_: float
        +pdf(x: float) float
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    %% Inheritance
    Distribution <|-- FrequencyDistribution
    Distribution <|-- SeverityDistribution
    
    FrequencyDistribution <|-- Poisson
    FrequencyDistribution <|-- Binomial
    FrequencyDistribution <|-- NegativeBinomial
    FrequencyDistribution <|-- Geometric
    
    SeverityDistribution <|-- Lognormal
    SeverityDistribution <|-- Gamma
    SeverityDistribution <|-- Exponential
    SeverityDistribution <|-- Pareto
    SeverityDistribution <|-- Weibull
```

## Compound and Extended Distributions

```mermaid
classDiagram
    class CompoundDistribution {
        -frequency: FrequencyDistribution
        -severity: SeverityDistribution
        -random_state: Optional[RandomState]
        +__init__(frequency, severity, random_state)
        +aggregate_loss(n_simulations: int) ndarray
        +mean() float
        +variance() float
        +sample(size: int) ndarray
    }
    
    class MixedPoissonDistribution {
        <<abstract>>
        -base_params: dict
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class PoissonGamma {
        -alpha: float
        -beta: float
        -exposure: float
        +__init__(alpha, beta, exposure)
        +sample(size: int) ndarray
        +pmf(k: int) float
        +mean() float
        +variance() float
    }
    
    class PoissonInverseGaussian {
        -mu: float
        -phi: float
        -exposure: float
        +__init__(mu, phi, exposure)
        +sample(size: int) ndarray
        +mean() float
        +variance() float
    }
    
    class ZeroInflatedDistribution {
        -base_distribution: Distribution
        -p_zero: float
        -em_tolerance: float
        -em_max_iter: int
        +__init__(base_dist, p_zero)
        +sample(size: int) ndarray
        +fit_em(data: ndarray) dict
        +mean() float
        +variance() float
    }
    
    class EdgeworthDistribution {
        -base_mean: float
        -base_variance: float
        -base_skewness: float
        -base_kurtosis: float
        -correction_terms: dict
        +__init__(mean, var, skew, kurt)
        +sample(size: int) ndarray
        +pdf(x: ndarray) ndarray
        +apply_correction(x: ndarray) ndarray
    }
    
    class QMCDistributionWrapper {
        -base_distribution: Distribution
        -qmc_engine: str
        -scramble: bool
        -seed: Optional[int]
        +__init__(base_dist, qmc_engine)
        +sample(size: int) ndarray
        +sample_qmc(size: int) ndarray
        +effective_sample_size() int
    }
    
    %% Relationships
    FrequencyDistribution <-- CompoundDistribution : uses
    SeverityDistribution <-- CompoundDistribution : uses
    
    Distribution <|-- MixedPoissonDistribution
    MixedPoissonDistribution <|-- PoissonGamma
    MixedPoissonDistribution <|-- PoissonInverseGaussian
    
    Distribution <-- ZeroInflatedDistribution : wraps
    Distribution <-- QMCDistributionWrapper : wraps
    Distribution <|-- EdgeworthDistribution
```

## Data Transfer Objects

```mermaid
classDiagram
    class SimulationResult {
        <<dataclass>>
        +losses: ndarray
        +computation_time: float
        +n_simulations: int
        +convergence_metrics: dict
        +memory_usage_mb: float
        +is_valid() bool
    }
    
    class ConvergenceMetrics {
        <<dataclass>>
        +mean_history: List[float]
        +std_history: List[float]
        +quantile_history: Dict[float, List[float]]
        +convergence_achieved: bool
        +iterations_to_converge: int
    }
    
    class RiskMeasures {
        <<dataclass>>
        +mean: float
        +std: float
        +var: Dict[float, float]
        +tvar: Dict[float, float]
        +expected_shortfall: float
        +maximum_loss: float
        +probability_ruin: float
    }
    
    class PortfolioMetrics {
        <<dataclass>>
        +total_exposure: float
        +average_frequency: float
        +average_severity: float
        +diversification_ratio: float
        +concentration_risk: float
        +largest_exposure: float
    }
    
    class OptimizationMetrics {
        <<dataclass>>
        +speedup_factor: float
        +memory_efficiency: float
        +accuracy_ratio: float
        +optimization_methods: List[str]
        +hardware_utilization: dict
    }
    
    SimulationResult --> ConvergenceMetrics : contains
    SimulationResult --> RiskMeasures : produces
    Portfolio ..> PortfolioMetrics : generates
    OptimizationSelector ..> OptimizationMetrics : tracks
```

## Distribution Factory Pattern

```mermaid
flowchart TD
    subgraph Factory["Distribution Factory"]
        CreateFreq["create_frequency_distribution()"]
        CreateSev["create_severity_distribution()"]
        CreateComp["create_compound_distribution()"]
        CreateExt["create_extended_distribution()"]
    end
    
    subgraph Input["Input Parameters"]
        FreqType["type: 'poisson'<br/>params: {lambda: 5}"]
        SevType["type: 'lognormal'<br/>params: {mu: 10, sigma: 2}"]
        CompType["frequency + severity"]
        ExtType["type: 'zero_inflated'<br/>base_dist + p_zero"]
    end
    
    subgraph Output["Created Distributions"]
        Freq["Poisson(lambda=5)"]
        Sev["Lognormal(mu=10, sigma=2)"]
        Comp["CompoundDistribution"]
        Ext["ZeroInflatedDistribution"]
    end
    
    FreqType --> CreateFreq --> Freq
    SevType --> CreateSev --> Sev
    CompType --> CreateComp --> Comp
    ExtType --> CreateExt --> Ext
    
    %% Validation
    CreateFreq --> Validate{Validate<br/>Parameters}
    CreateSev --> Validate
    CreateComp --> Validate
    CreateExt --> Validate
    
    Validate -->|Invalid| Error[Raise<br/>ValueError]
```

## State Management

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    
    Uninitialized --> Configured: set_params()
    
    Configured --> Sampling: sample()
    Configured --> Fitting: fit()
    
    Sampling --> Configured: complete
    
    Fitting --> Optimizing: EM/MLE
    Optimizing --> Converged: tolerance met
    Optimizing --> Failed: max_iter reached
    
    Converged --> Configured: update params
    Failed --> Configured: keep original
    
    Configured --> Modified: set_params()
    Modified --> Configured: validate
```

## Data Flow Through System

```mermaid
flowchart TB
    subgraph Input["Input Data"]
        Historical["Historical<br/>Loss Data"]
        Params["Distribution<br/>Parameters"]
        Portfolio["Portfolio<br/>Structure"]
    end
    
    subgraph Processing["Processing Layer"]
        Fitting["Parameter<br/>Fitting"]
        Validation["Data<br/>Validation"]
        Transform["Data<br/>Transformation"]
    end
    
    subgraph Models["Model Layer"]
        Freq["Frequency<br/>Model"]
        Sev["Severity<br/>Model"]
        Compound["Compound<br/>Model"]
    end
    
    subgraph Simulation["Simulation"]
        Generate["Generate<br/>Samples"]
        Aggregate["Aggregate<br/>Losses"]
        Calculate["Calculate<br/>Metrics"]
    end
    
    subgraph Output["Output"]
        Results["Simulation<br/>Results"]
        Metrics["Risk<br/>Measures"]
        Report["Analysis<br/>Report"]
    end
    
    Historical --> Fitting --> Freq
    Historical --> Fitting --> Sev
    Params --> Validation --> Freq
    Params --> Validation --> Sev
    Portfolio --> Transform --> Compound
    
    Freq --> Compound
    Sev --> Compound
    
    Compound --> Generate
    Generate --> Aggregate
    Aggregate --> Calculate
    
    Calculate --> Results
    Calculate --> Metrics
    Results --> Report
    Metrics --> Report
```

## Type System and Validation

```mermaid
classDiagram
    class Validator {
        <<utility>>
        +validate_positive(value: float, name: str)
        +validate_probability(value: float, name: str)
        +validate_integer(value: int, name: str)
        +validate_array(arr: ndarray, name: str)
        +validate_distribution_params(dist_type: str, params: dict)
    }
    
    class TypedParameter {
        <<dataclass>>
        +name: str
        +value: Any
        +param_type: Type
        +constraints: List[Constraint]
        +validate() bool
    }
    
    class Constraint {
        <<abstract>>
        +check(value: Any) bool
        +error_message: str
    }
    
    class RangeConstraint {
        -min_value: float
        -max_value: float
        +check(value: float) bool
    }
    
    class PositiveConstraint {
        +check(value: float) bool
    }
    
    class ProbabilityConstraint {
        +check(value: float) bool
    }
    
    Constraint <|-- RangeConstraint
    Constraint <|-- PositiveConstraint
    Constraint <|-- ProbabilityConstraint
    
    TypedParameter --> Constraint : uses
    Validator --> TypedParameter : validates
    Distribution --> Validator : uses
```