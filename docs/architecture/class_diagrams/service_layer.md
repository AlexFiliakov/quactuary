# Service Layer Architecture

This document details the service layer components, including the MCP server, CLI tools, and API interfaces.

## MCP Server Architecture

```mermaid
classDiagram
    class QuactuaryMCP {
        -name: str = "quactuary-mcp"
        -pricing_model: PricingModel
        -portfolio_builder: PortfolioBuilder
        +__init__()
        +calculate_premium(params) PremiumResult
        +fit_distribution(params) FitResult
        +analyze_portfolio(params) AnalysisResult
        +run_diagnostics(params) DiagnosticsResult
    }
    
    class MCPTool {
        <<interface>>
        +name: str
        +description: str
        +parameters: dict
        +execute(params: dict) dict
    }
    
    class PricingTool {
        +name: str = "calculate_premium"
        +execute(params) dict
        -_validate_params(params)
        -_build_portfolio(params)
        -_run_simulation(portfolio, config)
    }
    
    class DistributionTool {
        +name: str = "fit_distribution"
        +execute(params) dict
        -_fit_frequency(data, dist_type)
        -_fit_severity(data, dist_type)
        -_calculate_goodness_of_fit(data, distribution)
    }
    
    class PortfolioTool {
        +name: str = "analyze_portfolio"
        +execute(params) dict
        -_calculate_metrics(portfolio)
        -_assess_risk_concentration(portfolio)
        -_generate_recommendations(metrics)
    }
    
    class DiagnosticsTool {
        +name: str = "run_diagnostics"
        +execute(params) dict
        -_test_convergence(params)
        -_measure_efficiency(params)
        -_validate_accuracy(params)
    }
    
    MCPTool <|-- PricingTool
    MCPTool <|-- DistributionTool
    MCPTool <|-- PortfolioTool
    MCPTool <|-- DiagnosticsTool
    
    QuactuaryMCP --> PricingTool : registers
    QuactuaryMCP --> DistributionTool : registers
    QuactuaryMCP --> PortfolioTool : registers
    QuactuaryMCP --> DiagnosticsTool : registers
```

## CLI Tools Architecture

```mermaid
flowchart TB
    subgraph CLI["CLI Interface"]
        Main["run_dev.py<br/>Main Entry Point"]
        
        subgraph Commands["Commands"]
            Install["install<br/>Setup Environment"]
            Test["test<br/>Run Tests"]
            Lint["lint<br/>Code Quality"]
            Format["format<br/>Auto-format"]
            Docs["docs<br/>Build Docs"]
            Profile["profile<br/>Performance"]
            Benchmark["benchmark<br/>Run Benchmarks"]
        end
        
        subgraph Parsers["Argument Parsers"]
            TestParser["Test Options<br/>--coverage<br/>--profile<br/>--markers"]
            LintParser["Lint Options<br/>--fix<br/>--check"]
            DocsParser["Docs Options<br/>--serve<br/>--clean"]
        end
    end
    
    subgraph Execution["Execution Layer"]
        Runner["Command Runner"]
        Logger["Progress Logger"]
        ErrorHandler["Error Handler"]
    end
    
    Main --> Commands
    Commands --> Parsers
    Parsers --> Runner
    Runner --> Logger
    Runner --> ErrorHandler
    
    %% External Tools
    Runner -->|subprocess| Pytest["pytest"]
    Runner -->|subprocess| Ruff["ruff"]
    Runner -->|subprocess| Black["black"]
    Runner -->|subprocess| Sphinx["sphinx-build"]
```

## API Service Layer

```mermaid
classDiagram
    class QuactuaryAPI {
        <<facade>>
        +create_portfolio(config) Portfolio
        +calculate_premium(portfolio, params) PremiumResult
        +fit_distribution(data, dist_type) Distribution
        +run_simulation(portfolio, n_sims) SimulationResult
        +analyze_risk(results) RiskMeasures
    }
    
    class PortfolioBuilder {
        -portfolio: Portfolio
        +add_line_of_business(params) PortfolioBuilder
        +with_frequency(dist_type, params) PortfolioBuilder
        +with_severity(dist_type, params) PortfolioBuilder
        +with_policy_terms(terms) PortfolioBuilder
        +build() Portfolio
    }
    
    class SimulationService {
        -optimizer: OptimizationSelector
        -executor: SimulationExecutor
        +configure(config: SimulationConfig)
        +run(portfolio, n_sims) SimulationResult
        +run_batch(portfolios, n_sims) List[SimulationResult]
    }
    
    class RiskAnalysisService {
        +calculate_var(losses, confidence) float
        +calculate_tvar(losses, confidence) float
        +calculate_economic_capital(losses, params) float
        +stress_test(portfolio, scenarios) StressTestResult
    }
    
    class ReportingService {
        -template_engine: TemplateEngine
        +generate_premium_report(result) Report
        +generate_risk_report(metrics) Report
        +export_results(results, format) bytes
    }
    
    QuactuaryAPI --> PortfolioBuilder : uses
    QuactuaryAPI --> SimulationService : uses
    QuactuaryAPI --> RiskAnalysisService : uses
    QuactuaryAPI --> ReportingService : uses
```

## Service Integration Patterns

```mermaid
sequenceDiagram
    participant Client
    participant API as QuactuaryAPI
    participant Builder as PortfolioBuilder
    participant Sim as SimulationService
    participant Risk as RiskAnalysisService
    participant Report as ReportingService
    
    Client->>API: calculate_premium(config)
    
    API->>Builder: create portfolio
    Builder->>Builder: configure components
    Builder-->>API: portfolio
    
    API->>Sim: run(portfolio, n_sims)
    Sim->>Sim: optimize configuration
    Sim->>Sim: execute simulation
    Sim-->>API: simulation_result
    
    API->>Risk: analyze_risk(results)
    Risk->>Risk: calculate measures
    Risk-->>API: risk_measures
    
    API->>Report: generate_report(results)
    Report-->>API: report
    
    API-->>Client: PremiumResult
```

## Error Handling and Validation

```mermaid
flowchart TD
    subgraph Validation["Input Validation Layer"]
        Schema["Schema<br/>Validation"]
        Business["Business<br/>Rules"]
        Constraints["Parameter<br/>Constraints"]
    end
    
    subgraph Errors["Error Types"]
        ValidationErr["ValidationError<br/>Invalid Input"]
        ConfigErr["ConfigurationError<br/>Bad Config"]
        ComputeErr["ComputationError<br/>Calc Failed"]
        ResourceErr["ResourceError<br/>Memory/CPU"]
    end
    
    subgraph Handling["Error Handling"]
        Catch["Exception<br/>Catcher"]
        Log["Error<br/>Logger"]
        Transform["Error<br/>Transformer"]
        Response["Error<br/>Response"]
    end
    
    Request[Client Request] --> Validation
    
    Validation -->|Invalid| ValidationErr
    Validation -->|Valid| Process[Process Request]
    
    Process -->|Config Issue| ConfigErr
    Process -->|Calc Issue| ComputeErr
    Process -->|Resource Issue| ResourceErr
    Process -->|Success| Result[Success Result]
    
    ValidationErr --> Catch
    ConfigErr --> Catch
    ComputeErr --> Catch
    ResourceErr --> Catch
    
    Catch --> Log
    Log --> Transform
    Transform --> Response
    
    Response --> Client[Client Response]
    Result --> Client
```

## Service Configuration

```mermaid
classDiagram
    class ServiceConfig {
        <<dataclass>>
        +api_version: str
        +max_portfolio_size: int
        +max_simulations: int
        +default_confidence_levels: List[float]
        +enable_caching: bool
        +cache_ttl_seconds: int
    }
    
    class SimulationConfig {
        <<dataclass>>
        +n_simulations: int
        +random_seed: Optional[int]
        +convergence_tolerance: float
        +max_iterations: int
        +enable_qmc: bool
        +parallel_config: ParallelConfig
    }
    
    class MCPConfig {
        <<dataclass>>
        +server_name: str
        +host: str
        +port: int
        +enable_auth: bool
        +max_request_size: int
        +timeout_seconds: int
    }
    
    class CLIConfig {
        <<dataclass>>
        +verbose: bool
        +color_output: bool
        +progress_bars: bool
        +log_level: str
        +output_format: str
    }
    
    ServiceConfig --> SimulationConfig : contains
    ServiceConfig --> MCPConfig : contains
    ServiceConfig --> CLIConfig : contains
```

## Service Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initializing: Service Start
    
    Initializing --> Loading: Load Config
    Loading --> Validating: Validate Setup
    
    Validating --> Ready: Valid
    Validating --> Failed: Invalid
    
    Ready --> Processing: Handle Request
    Processing --> Ready: Complete
    Processing --> Error: Exception
    
    Error --> Ready: Recovered
    Error --> Failed: Fatal
    
    Ready --> Stopping: Shutdown Signal
    Stopping --> Cleanup: Release Resources
    Cleanup --> [*]: Terminated
    
    Failed --> [*]: Exit
```

## Performance Monitoring

```mermaid
flowchart LR
    subgraph Metrics["Service Metrics"]
        Latency["Request<br/>Latency"]
        Throughput["Request<br/>Throughput"]
        Errors["Error<br/>Rate"]
        Resources["Resource<br/>Usage"]
    end
    
    subgraph Collectors["Metric Collectors"]
        Timer["Timing<br/>Decorator"]
        Counter["Request<br/>Counter"]
        Gauge["Resource<br/>Gauge"]
        Histogram["Distribution<br/>Histogram"]
    end
    
    subgraph Storage["Metric Storage"]
        Memory["In-Memory<br/>Store"]
        File["File<br/>Backend"]
        Database["Database<br/>Backend"]
    end
    
    subgraph Reporting["Reporting"]
        Console["Console<br/>Output"]
        JSON["JSON<br/>Export"]
        Dashboard["Web<br/>Dashboard"]
    end
    
    Latency --> Timer --> Memory
    Throughput --> Counter --> Memory
    Errors --> Counter --> Memory
    Resources --> Gauge --> Memory
    
    Memory --> File
    Memory --> Database
    
    Memory --> Console
    File --> JSON
    Database --> Dashboard
```