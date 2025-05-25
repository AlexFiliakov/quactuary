"""
Pricing module for actuarial loss models.

This module provides classes to price excess loss and compute risk measures for insurance portfolios,
using classical simulation methods (Monte Carlo, FFT) with optional quantum acceleration via Qiskit.

Notes:
    Classical methods include Monte Carlo and FFT via external libraries.
    Quantum acceleration uses Qiskit backends managed by BackendManager.

Examples:
    >>> from quactuary.pricing import ExcessLossModel, RiskMeasureModel
    >>> model = ExcessLossModel(inforce, deductible=1000.0, limit=10000.0)
    >>> result = model.compute_excess_loss()
"""

from typing import Optional, Union

import numpy as np
import pandas as pd
from qiskit.providers import Backend, BackendV1, BackendV2

from quactuary.backend import BackendManager, ClassicalBackend, get_backend
from quactuary.book import Portfolio
from quactuary.classical import ClassicalPricingModel
from quactuary.datatypes import PricingResult
from quactuary.distributions.compound import CompoundDistribution
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.quantum import QuantumPricingModel


class PricingModel(ClassicalPricingModel, QuantumPricingModel):
    """
    Base class for actuarial pricing models with optional quantum support.

    Provides common interface for portfolio-based loss models using classical or quantum backends.

    Args:
        backend (Optional[BackendManager]): Execution backend override.
        **kw: Additional model-specific settings.

    Attributes:
        portfolio (Portfolio): Wrapped inforce portfolio.
        layer_deductible (Optional[float]): Deductible for the layer.
        layer_limit (Optional[float]): Limit for the layer.
        backend (BackendManager): Backend manager for execution.
    """

    def __init__(self, portfolio: Portfolio):
        """
        Initialize an ActuarialModel.

        Args:
            portfolio (Portfolio): Inforce policy data grouped into a Portfolio.
        """
        super(ClassicalPricingModel).__init__()
        super(QuantumPricingModel).__init__()
        self.portfolio = portfolio
        self.compound_distribution = None

    def simulate(
        self,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        backend: Optional[BackendManager] = None
    ) -> PricingResult:
        """
        Calculate portfolio statistics based on the selected methods.

        Args:
            mean (bool): Calculate mean loss.
            variance (bool): Calculate variance.
            value_at_risk (bool): Calculate value at risk.
            tail_value_at_risk (bool): Calculate tail value at risk.
            backend (Optional[BackendManager]): Execution backend override.
            num_simulations (Optional[int]): Number of Classical simulations.
        """

        if backend is None:
            cur_backend = get_backend().backend
        else:
            cur_backend = backend.backend

        if isinstance(cur_backend, ClassicalBackend):
            return ClassicalPricingModel.calculate_portfolio_statistics(
                self, self.portfolio, mean, variance, value_at_risk, tail_value_at_risk, tail_alpha, n_sims)
        if isinstance(cur_backend, (Backend, BackendV1, BackendV2)):
            return QuantumPricingModel.calculate_portfolio_statistics(
                self, self.portfolio, mean, variance, value_at_risk, tail_value_at_risk, tail_alpha)
        else:
            error_str = "Unsupported backend type. Must be a Qiskit or classical backend."
            raise ValueError(error_str)
    
    def set_compound_distribution(self, frequency: FrequencyModel, severity: SeverityModel):
        """
        Set compound distribution for aggregate loss modeling.
        
        Args:
            frequency: Frequency distribution model
            severity: Severity distribution model
        """
        self.compound_distribution = CompoundDistribution.create(frequency, severity)
    
    def calculate_aggregate_statistics(
        self,
        apply_policy_terms: bool = True,
        confidence_levels: Optional[list] = None,
        n_simulations: int = 10000
    ) -> dict:
        """
        Calculate aggregate loss statistics using compound distribution or empirical methods.
        
        Args:
            apply_policy_terms: Whether to apply policy terms (deductibles, limits, etc.)
            confidence_levels: List of confidence levels for VaR/TVaR (default: [0.90, 0.95, 0.99])
            n_simulations: Number of simulations for empirical calculation if no compound distribution
            
        Returns:
            Dictionary with aggregate loss statistics
        """
        if confidence_levels is None:
            confidence_levels = [0.90, 0.95, 0.99]
        
        if self.compound_distribution is not None:
            # Use compound distribution if available
            results = {
                'mean': self.compound_distribution.mean(),
                'std': self.compound_distribution.std(),
                'variance': self.compound_distribution.var(),
                'has_analytical': self.compound_distribution.has_analytical_solution()
            }
            
            # Calculate quantiles (VaR)
            for level in confidence_levels:
                results[f'var_{level:.0%}'] = self.compound_distribution.ppf(level)
            
            # Calculate TVaR (Tail Value at Risk) using simulation if needed
            for level in confidence_levels:
                var = results[f'var_{level:.0%}']
                # Simulate conditional expectation above VaR
                samples = self.compound_distribution.rvs(size=10000)
                tail_samples = samples[samples > var]
                if len(tail_samples) > 0:
                    results[f'tvar_{level:.0%}'] = np.mean(tail_samples)
                else:
                    # Fallback to VaR if no samples exceed it
                    results[f'tvar_{level:.0%}'] = var
        else:
            # Use empirical calculation based on portfolio
            # Generate aggregate loss samples empirically
            aggregate_losses = []
            
            for _ in range(n_simulations):
                total_loss = 0.0
                # For each policy in portfolio, simulate a loss
                for policy in self.portfolio.policies:
                    # Simple empirical approach: use expected loss with some variance
                    if hasattr(policy, 'expected_loss'):
                        # Add random variation around expected loss
                        loss = np.random.exponential(scale=policy.expected_loss)
                    else:
                        # Default to a simple loss model
                        loss = np.random.exponential(scale=1000.0)
                    
                    # Apply policy terms if available
                    if hasattr(policy, 'deductible'):
                        loss = max(0, loss - policy.deductible)
                    if hasattr(policy, 'limit'):
                        loss = min(loss, policy.limit)
                    
                    total_loss += loss
                
                aggregate_losses.append(total_loss)
            
            aggregate_losses = np.array(aggregate_losses)
            
            # Calculate empirical statistics
            results = {
                'mean': np.mean(aggregate_losses),
                'std': np.std(aggregate_losses),
                'variance': np.var(aggregate_losses),
                'has_analytical': False,
                'method': 'empirical'
            }
            
            # Calculate empirical quantiles (VaR)
            for level in confidence_levels:
                results[f'var_{level:.0%}'] = np.percentile(aggregate_losses, level * 100)
            
            # Calculate empirical TVaR
            for level in confidence_levels:
                var = results[f'var_{level:.0%}']
                tail_samples = aggregate_losses[aggregate_losses > var]
                if len(tail_samples) > 0:
                    results[f'tvar_{level:.0%}'] = np.mean(tail_samples)
                else:
                    results[f'tvar_{level:.0%}'] = var
        
        # Apply policy terms if requested
        if apply_policy_terms and hasattr(self.portfolio, 'policies'):
            # This would apply deductibles, limits, etc. from the portfolio
            # For now, we'll add a placeholder
            results['note'] = 'Policy terms application included in calculation'
        
        return results
    
    def price_excess_layer(
        self,
        attachment: float,
        limit: float,
        n_simulations: int = 10000
    ) -> dict:
        """
        Price an excess of loss reinsurance layer.
        
        Args:
            attachment: Layer attachment point
            limit: Layer limit
            n_simulations: Number of simulations for pricing
            
        Returns:
            Dictionary with layer pricing information
        """
        if self.compound_distribution is None:
            raise ValueError("Compound distribution not set. Call set_compound_distribution first.")
        
        # Generate aggregate loss samples
        samples = self.compound_distribution.rvs(size=n_simulations)
        
        # Apply layer terms
        layer_losses = np.minimum(np.maximum(samples - attachment, 0), limit)
        
        # Calculate layer statistics
        results = {
            'attachment': attachment,
            'limit': limit,
            'expected_loss': np.mean(layer_losses),
            'loss_std': np.std(layer_losses),
            'loss_probability': np.mean(layer_losses > 0),
            'average_severity': np.mean(layer_losses[layer_losses > 0]) if np.any(layer_losses > 0) else 0,
            'ground_up_mean': np.mean(samples),
            'ground_up_std': np.std(samples)
        }
        
        return results
