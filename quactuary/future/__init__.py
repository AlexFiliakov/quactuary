"""
Future features and experimental actuarial modeling capabilities.

This package contains next-generation actuarial modeling features that are under
active development or considered experimental. These modules represent the future
direction of the quActuary framework, incorporating cutting-edge techniques from
actuarial science, machine learning, and quantitative finance.

Package Contents:
    dependence: Advanced dependence modeling and copula implementations
    life: Life insurance and annuity modeling capabilities
    machine_learning: ML-enhanced actuarial modeling and prediction
    portfolio_optimization: Modern portfolio theory applied to insurance
    reserving: Advanced loss reserving and claims development techniques

Development Status:
    These modules are in various stages of development:
    - Some are research prototypes exploring new methodologies
    - Others are stable but not yet fully integrated into the main API
    - All are subject to API changes in future releases
    - Documentation and examples may be incomplete

Target Capabilities:
    Advanced Risk Modeling:
        - Multi-dimensional copula modeling for dependent risks
        - Extreme value theory for tail risk assessment
        - Dynamic risk factor modeling with time-varying parameters
        - Spatial and temporal correlation modeling
        
    Machine Learning Integration:
        - Neural network-based frequency and severity modeling
        - Ensemble methods for improved prediction accuracy
        - Automated feature engineering for insurance data
        - Real-time model updating and calibration
        
    Modern Portfolio Techniques:
        - Risk-based capital optimization for insurance portfolios
        - Dynamic hedging strategies for insurance liabilities
        - Capital allocation and performance attribution
        - Solvency II and other regulatory capital modeling
        
    Advanced Reserving:
        - Stochastic loss development models
        - Bootstrap and Bayesian approaches to uncertainty quantification
        - Claims inflation and emergence pattern modeling
        - Integration with external economic and market data

Usage Philosophy:
    Future modules follow these principles:
    1. Backward compatibility with existing quActuary APIs
    2. Integration with classical actuarial methods
    3. Performance optimization for large-scale applications
    4. Comprehensive uncertainty quantification
    5. Production-ready reliability and testing

Examples:
    Exploring future capabilities:
        >>> import quactuary.future as qf
        >>> 
        >>> # Check what experimental features are available
        >>> print(qf.__all__)
        >>> 
        >>> # Use experimental dependence modeling
        >>> from quactuary.future.dependence import CopulaModel
        >>> copula = CopulaModel(family='gaussian')
        >>> 
        >>> # Try ML-enhanced modeling
        >>> from quactuary.future.machine_learning import NeuralSeverityModel
        >>> ml_model = NeuralSeverityModel()
        
    Integration with main package:
        >>> from quactuary import PricingModel
        >>> from quactuary.future.portfolio_optimization import RiskBudgeting
        >>> 
        >>> # Combine traditional and future approaches
        >>> pricing = PricingModel(portfolio)
        >>> optimizer = RiskBudgeting(pricing.results)
        >>> optimized_portfolio = optimizer.optimize()

Development Guidelines:
    If you're contributing to future modules:
    1. Follow the established quActuary coding standards
    2. Include comprehensive docstrings and examples
    3. Provide extensive unit tests
    4. Consider backward compatibility implications
    5. Document any external dependencies clearly
    6. Include performance benchmarks where relevant

API Stability:
    - Modules in this package may have breaking changes between versions
    - Check release notes for migration guidance
    - Pin specific versions for production use
    - Provide feedback on experimental features via GitHub issues

Migration Path:
    As future modules mature, they will be moved to the main package:
    1. API stabilization and comprehensive testing
    2. Documentation completion and example development
    3. Performance optimization and benchmarking
    4. Integration testing with existing components
    5. Graduation to the main quactuary namespace

Notes:
    - Some modules may require additional dependencies
    - Performance characteristics may not be fully optimized
    - Breaking changes may occur without deprecation warnings
    - Not all features may be available on all platforms
    - Consider these modules as preview/beta functionality

See Also:
    - Main quactuary package: For stable, production-ready features
    - Documentation: For detailed guides on experimental features
    - GitHub Issues: For reporting bugs or requesting features
    - Contributing Guide: For development guidelines and standards
"""

# Future modules that are ready for preview
__all__ = [
    'dependence',
    'life', 
    'machine_learning',
    'portfolio_optimization',
    'reserving'
]

# Version information for future package
__version__ = "0.1.0-alpha"
__status__ = "experimental"

# Import experimental modules with graceful failure
import warnings

def _import_with_warning(module_name):
    """Import future module with experimental warning."""
    try:
        return __import__(f'quactuary.future.{module_name}', fromlist=[module_name])
    except ImportError as e:
        warnings.warn(
            f"Future module '{module_name}' could not be imported. "
            f"This may be due to missing dependencies or incomplete implementation. "
            f"Error: {e}",
            ImportWarning
        )
        return None

# Lazy imports for future modules
def __getattr__(name):
    """Lazy loading of future modules with warnings."""
    if name in __all__:
        return _import_with_warning(name)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Provide information about experimental status
def get_experimental_features():
    """
    Get information about experimental features and their status.
    
    Returns:
        dict: Dictionary with feature names and their development status.
        
    Examples:
        >>> import quactuary.future as qf
        >>> features = qf.get_experimental_features()
        >>> for feature, status in features.items():
        ...     print(f"{feature}: {status}")
    """
    return {
        'dependence': 'Alpha - Basic copula modeling implemented',
        'life': 'Planning - Life insurance modeling in design phase',
        'machine_learning': 'Alpha - Neural network models available',
        'portfolio_optimization': 'Beta - Risk budgeting and optimization tools',
        'reserving': 'Alpha - Stochastic reserving methods'
    }

def experimental_warning():
    """Display warning about experimental nature of future modules."""
    warnings.warn(
        "You are using experimental features from quactuary.future. "
        "These features are under active development and may change "
        "significantly between versions. Use with caution in production "
        "environments and consider pinning specific versions.",
        UserWarning,
        stacklevel=2
    )