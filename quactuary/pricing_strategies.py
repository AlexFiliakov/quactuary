"""
Pricing strategy implementations for the strategy pattern architecture.

This module provides the strategy pattern implementation to replace multiple inheritance
in the PricingModel class, enabling cleaner separation of concerns and easier testing.
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
    
    Defines the interface that all pricing strategies must implement.
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
    
    This strategy implements classical actuarial calculations using
    traditional Monte Carlo methods and statistical approaches.
    """
    
    def __init__(self, use_jit: bool = True):
        """
        Initialize the classical pricing strategy.
        
        Args:
            use_jit: Whether to use JIT compilation for performance (default: True)
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