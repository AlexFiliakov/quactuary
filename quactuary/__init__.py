"""
quactuary: Quantum-accelerated actuarial modeling framework.

quactuary is a comprehensive Python package for actuarial calculations that seamlessly
integrates classical and quantum computing approaches. It provides tools for insurance
pricing, risk analysis, and portfolio optimization with optional quantum acceleration
via Qiskit.

Key Features:
    - **Unified Interface**: Single API for both classical and quantum calculations
    - **Distribution Library**: Comprehensive set of frequency and severity distributions
    - **Portfolio Modeling**: Flexible framework for insurance portfolio analysis
    - **Quantum Acceleration**: Optional quantum algorithms for suitable problems
    - **Performance Optimization**: JIT compilation and parallelization support
    - **Extensible Design**: Easy to add new distributions and algorithms

Main Components:
    - `pricing`: Portfolio pricing and risk measure calculations
    - `distributions`: Frequency, severity, and compound distributions
    - `book`: Insurance portfolio and policy term modeling
    - `backend`: Quantum/classical backend management
    - `classical`: Traditional Monte Carlo implementations
    - `quantum`: Quantum algorithm implementations (experimental)

Quick Start:
    Basic portfolio pricing:
        >>> import quactuary as qa
        >>> from quactuary.book import Portfolio
        >>> from quactuary.pricing import PricingModel
        >>> 
        >>> # Create portfolio from your data
        >>> portfolio = Portfolio(policies_dataframe)
        >>> 
        >>> # Initialize pricing model
        >>> model = PricingModel(portfolio)
        >>> 
        >>> # Calculate risk measures
        >>> result = model.simulate(n_sims=10000)
        >>> print(f"Expected loss: ${result.estimates['mean']:,.2f}")
        >>> print(f"95% VaR: ${result.estimates['VaR']:,.2f}")

    Using distributions:
        >>> from quactuary.distributions import Poisson, Lognormal, CompoundDistribution
        >>> 
        >>> # Define frequency and severity
        >>> frequency = Poisson(lambda_=100)
        >>> severity = Lognormal(shape=1.5, scale=np.exp(7))
        >>> 
        >>> # Create compound distribution
        >>> compound = CompoundDistribution.create(frequency, severity)
        >>> print(f"Expected aggregate loss: ${compound.mean():,.2f}")

    Switching to quantum backend:
        >>> import quactuary as qa
        >>> 
        >>> # Switch to quantum simulator
        >>> qa.set_backend('quantum', provider='AerSimulator')
        >>> 
        >>> # All subsequent calculations use quantum algorithms
        >>> result = model.simulate(n_sims=1000)

Installation:
    Basic installation:
        pip install quactuary
        
    With quantum support:
        pip install quactuary[quantum]
        
    Development installation:
        pip install -e .[dev]

Documentation:
    Full documentation available at: https://docs.quactuary.com
    
    Key sections:
    - User Guide: Step-by-step tutorials
    - API Reference: Detailed function documentation  
    - Theory: Mathematical foundations
    - Examples: Jupyter notebooks

Version:
    Current version accessible via `quactuary.__version__`

License:
    BSD 3-Clause License - See LICENSE file for details

Author:
    Alex Filiakov - https://alexfiliakov.com
    GitHub: https://github.com/AlexFiliakov

Notes:
    - Quantum features are experimental and under active development
    - Classical methods are production-ready and well-tested
    - To contribute to this project, see https://docs.quactuary.com/development for development guidelines
"""

try:
    from ._version import version as __version__
except ImportError:
    try:
        # use installed distribution metadata
        import importlib.metadata as _im
    except ImportError:
        import importlib_metadata as _im  # python <3.8 fallback
    __version__ = _im.version("quactuary")

from .backend import get_backend, set_backend
from .book import Inforce, PolicyTerms, Portfolio
from .pricing import PricingModel

# Make key classes available at package level
__all__ = [
    '__version__',
    'get_backend',
    'set_backend', 
    'PricingModel',
    'Portfolio',
    'Inforce',
    'PolicyTerms',
]
