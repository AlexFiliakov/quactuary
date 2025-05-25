"""
Pricing strategy implementations using the strategy pattern.

This module implements the strategy pattern for actuarial pricing calculations,
providing a clean separation between different computational approaches (classical,
quantum, JIT-optimized) while maintaining a consistent interface. This architecture
replaces the previous multiple inheritance design with composition, resulting in
more maintainable and testable code.

Key Components:
    - PricingStrategy: Abstract base class defining the strategy interface
    - ClassicalPricingStrategy: Traditional Monte Carlo implementation
    - QuantumPricingStrategy: Quantum-accelerated calculations (future)
    - Helper functions for strategy selection based on backend type

Design Benefits:
    - Separation of concerns: Each strategy encapsulates its approach
    - Extensibility: New strategies can be added without modifying existing code
    - Testability: Strategies can be tested independently
    - Runtime flexibility: Strategies can be swapped dynamically

Examples:
    Using classical strategy directly:
        >>> from quactuary.pricing_strategies import ClassicalPricingStrategy
        >>> from quactuary.book import Portfolio
        >>> 
        >>> portfolio = Portfolio(policies_df)
        >>> strategy = ClassicalPricingStrategy(use_jit=True)
        >>> result = strategy.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     n_sims=10000
        ... )

    Strategy selection based on backend:
        >>> from quactuary.backend import get_backend
        >>> from quactuary.pricing_strategies import get_strategy_for_backend
        >>> 
        >>> backend = get_backend()  # Current global backend
        >>> strategy = get_strategy_for_backend(backend)
        >>> # Automatically returns appropriate strategy

    Integration with PricingModel:
        >>> from quactuary.pricing import PricingModel
        >>> 
        >>> # PricingModel uses strategies internally
        >>> model = PricingModel(portfolio, strategy=ClassicalPricingStrategy())
        >>> # Or let it auto-select based on backend
        >>> model = PricingModel(portfolio)  # Uses default strategy

Notes:
    - Classical strategy supports both standard and JIT-optimized execution
    - Quantum strategy is a placeholder for future implementation
    - Strategies should be stateless when possible for thread safety
    - The pattern allows for easy addition of new computational approaches
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from qiskit.providers import Backend, BackendV1, BackendV2

from quactuary.backend import BackendManager, ClassicalBackend, get_backend
from quactuary.book import Portfolio
from quactuary.datatypes import PricingResult


class PricingStrategy(ABC):
    """
    Abstract base class for pricing strategies.
    
    This class defines the interface that all pricing strategies must implement,
    ensuring consistent behavior across different computational approaches. Each
    concrete strategy encapsulates a specific method for calculating portfolio
    risk measures (classical Monte Carlo, quantum algorithms, etc.).
    
    The strategy pattern allows the PricingModel to delegate calculations to
    different implementations without knowing the specific details, enabling
    runtime flexibility and clean separation of concerns.
    
    Subclasses must implement:
        - calculate_portfolio_statistics: Main calculation method
        
    Design principles:
        - Strategies should be stateless when possible
        - All strategies return PricingResult objects
        - Parameters should be validated by the strategy
        - Strategies can have their own initialization parameters
        
    Examples:
        Creating a custom strategy:
            >>> class CustomStrategy(PricingStrategy):
            ...     def calculate_portfolio_statistics(self, portfolio, **kwargs):
            ...         # Custom implementation here
            ...         return PricingResult(...)
    """
    
    @abstractmethod
    def calculate_portfolio_statistics(
        self,
        portfolio: Portfolio,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        **kwargs
    ) -> PricingResult:
        """
        Calculate portfolio statistics using this strategy.
        
        Args:
            portfolio: The portfolio to analyze
            mean: Whether to calculate mean loss
            variance: Whether to calculate variance
            value_at_risk: Whether to calculate VaR
            tail_value_at_risk: Whether to calculate TVaR
            tail_alpha: Alpha level for tail risk measures
            n_sims: Number of simulations (for simulation-based strategies)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            PricingResult containing the calculated statistics
        """
        pass


class ClassicalPricingStrategy(PricingStrategy):
    """
    Classical pricing strategy using Monte Carlo simulation.
    
    This strategy implements traditional actuarial calculations using Monte Carlo
    simulation methods. It supports both standard Python implementations and
    JIT-compiled versions for improved performance on large portfolios.
    
    The classical approach:
    - Generates random samples from loss distributions
    - Computes empirical statistics from the samples
    - Provides consistent, well-understood results
    - Scales linearly with number of simulations
    
    Features:
        - Standard Monte Carlo simulation
        - Optional JIT compilation via Numba
        - Support for all standard risk measures
        - Integration with quasi-random sequences
        
    Attributes:
        use_jit (bool): Whether to use JIT-compiled kernels for performance.
            
    Examples:
        Standard usage:
            >>> strategy = ClassicalPricingStrategy(use_jit=False)
            >>> result = strategy.calculate_portfolio_statistics(
            ...     portfolio, n_sims=10000
            ... )
            
        With JIT optimization:
            >>> strategy = ClassicalPricingStrategy(use_jit=True)
            >>> # Faster for large portfolios and many simulations
            >>> result = strategy.calculate_portfolio_statistics(
            ...     portfolio, n_sims=100000
            ... )
            
    Performance notes:
        - JIT compilation has overhead on first call
        - Benefits increase with portfolio size and simulation count
        - Standard implementation better for small/quick calculations
        - JIT version can be 10-100x faster for large problems
    """
    
    def __init__(self, use_jit: bool = True):
        """
        Initialize the classical pricing strategy.
        
        Args:
            use_jit (bool): Whether to use JIT compilation for performance.
                Default is True. Set to False for better debugging or when
                working with small portfolios where compilation overhead
                isn't worth the speedup.
                
        Examples:
            >>> # For production with large portfolios
            >>> strategy = ClassicalPricingStrategy(use_jit=True)
            >>> 
            >>> # For debugging or small calculations  
            >>> strategy = ClassicalPricingStrategy(use_jit=False)
        """
        self.use_jit = use_jit
    
    def calculate_portfolio_statistics(
        self,
        portfolio: Portfolio,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        **kwargs
    ) -> PricingResult:
        """
        Calculate portfolio statistics using classical methods.
        
        Uses Monte Carlo simulation and classical statistical methods
        to compute portfolio risk measures.
        """
        if self.use_jit:
            # Use JIT-optimized implementation
            from quactuary.classical_jit import ClassicalJITPricingModel
            
            jit_model = ClassicalJITPricingModel()
            return jit_model.calculate_portfolio_statistics(
                portfolio=portfolio,
                mean=mean,
                variance=variance,
                value_at_risk=value_at_risk,
                tail_value_at_risk=tail_value_at_risk,
                tail_alpha=tail_alpha,
                n_sims=n_sims,
                use_jit=True,
                **kwargs
            )
        else:
            # Import the original implementation from classical module
            from quactuary.classical import ClassicalPricingModel
            
            # Create a temporary instance to delegate to existing logic
            classical_model = ClassicalPricingModel()
            
            # Delegate to the existing implementation
            return classical_model.calculate_portfolio_statistics(
                portfolio=portfolio,
                mean=mean,
                variance=variance,
                value_at_risk=value_at_risk,
                tail_value_at_risk=tail_value_at_risk,
                tail_alpha=tail_alpha,
                n_sims=n_sims,
                **kwargs
            )


class QuantumPricingStrategy(PricingStrategy):
    """
    Quantum pricing strategy (future implementation).
    
    This strategy will implement quantum-accelerated actuarial calculations
    using quantum computing algorithms via Qiskit.
    
    Currently provides a clean interface stub that raises NotImplementedError
    until quantum implementation is complete.
    """
    
    def __init__(self):
        """Initialize the quantum pricing strategy."""
        pass
    
    def calculate_portfolio_statistics(
        self,
        portfolio: Portfolio,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        **kwargs
    ) -> PricingResult:
        """
        Calculate portfolio statistics using quantum methods.
        
        Currently raises NotImplementedError as quantum implementation
        is planned for future development.
        """
        raise NotImplementedError(
            "Quantum pricing strategy is not yet implemented. "
            "Use ClassicalPricingStrategy for current functionality."
        )


def get_strategy_for_backend(backend: Optional[BackendManager] = None) -> PricingStrategy:
    """
    Get the appropriate pricing strategy for the given backend.
    
    Args:
        backend: The backend manager to determine strategy for.
                If None, uses the current global backend.
                
    Returns:
        PricingStrategy instance appropriate for the backend type.
    """
    if backend is None:
        cur_backend = get_backend().backend
    else:
        cur_backend = backend.backend
    
    if isinstance(cur_backend, ClassicalBackend):
        return ClassicalPricingStrategy()
    elif isinstance(cur_backend, (Backend, BackendV1, BackendV2)):
        return QuantumPricingStrategy()
    else:
        raise ValueError(
            f"Unsupported backend type: {type(cur_backend)}. "
            "Must be a Qiskit or classical backend."
        )