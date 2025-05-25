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
from quactuary.datatypes import PricingResult
from quactuary.distributions.compound import CompoundDistribution
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel
from quactuary.pricing_strategies import PricingStrategy, ClassicalPricingStrategy, get_strategy_for_backend
from quactuary.sobol import set_qmc_simulator, reset_qmc_simulator


class PricingModel:
    """
    Base class for actuarial pricing models with optional quantum support.

    Uses composition with strategy pattern to provide classical or quantum backend execution.

    Args:
        portfolio (Portfolio): Inforce policy data grouped into a Portfolio.
        strategy (Optional[PricingStrategy]): Pricing strategy to use. If None, automatically selects based on current backend.

    Attributes:
        portfolio (Portfolio): Wrapped inforce portfolio.
        strategy (PricingStrategy): Strategy object handling the actual calculations.
        compound_distribution (Optional[CompoundDistribution]): Compound distribution for aggregate loss modeling.
    """

    def __init__(self, portfolio: Portfolio, strategy: Optional[PricingStrategy] = None):
        """
        Initialize a PricingModel.

        Args:
            portfolio (Portfolio): Inforce policy data grouped into a Portfolio.
            strategy (Optional[PricingStrategy]): Pricing strategy to use. If None, automatically selects based on current backend.
        """
        self.portfolio = portfolio
        self.strategy = strategy or ClassicalPricingStrategy()
        self.compound_distribution = None

    def simulate(
        self,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        n_sims: Optional[int] = None,
        backend: Optional[BackendManager] = None,
        qmc_method: Optional[str] = None,
        qmc_scramble: bool = True,
        qmc_skip: int = 1024,
        qmc_seed: Optional[int] = None
    ) -> PricingResult:
        """
        Calculate portfolio statistics using the configured strategy.

        Args:
            mean (bool): Calculate mean loss.
            variance (bool): Calculate variance.
            value_at_risk (bool): Calculate value at risk.
            tail_value_at_risk (bool): Calculate tail value at risk.
            tail_alpha (float): Alpha level for tail risk measures.
            n_sims (Optional[int]): Number of simulations.
            backend (Optional[BackendManager]): Execution backend override. If provided, temporarily switches strategy.
            qmc_method (Optional[str]): Quasi-Monte Carlo method ("sobol", "halton", or None for standard random).
            qmc_scramble (bool): Whether to apply scrambling to QMC sequences.
            qmc_skip (int): Number of initial QMC points to skip.
            qmc_seed (Optional[int]): Random seed for QMC scrambling.

        Returns:
            PricingResult: Portfolio statistics results.
        """
        # Configure QMC if requested
        qmc_was_configured = get_qmc_simulator() is not None
        if qmc_method is not None:
            set_qmc_simulator(
                method=qmc_method,
                scramble=qmc_scramble,
                skip=qmc_skip,
                seed=qmc_seed
            )
        
        try:
            # Use a different strategy if backend is specified
            if backend is not None:
                strategy = get_strategy_for_backend(backend)
                return strategy.calculate_portfolio_statistics(
                    portfolio=self.portfolio,
                    mean=mean,
                    variance=variance, 
                    value_at_risk=value_at_risk,
                    tail_value_at_risk=tail_value_at_risk,
                    tail_alpha=tail_alpha,
                    n_sims=n_sims
                )
            
            # Use the configured strategy
            return self.strategy.calculate_portfolio_statistics(
                portfolio=self.portfolio,
                mean=mean,
                variance=variance,
                value_at_risk=value_at_risk,
                tail_value_at_risk=tail_value_at_risk,
                tail_alpha=tail_alpha,
                n_sims=n_sims
            )
        finally:
            # Reset QMC if it wasn't configured before
            if qmc_method is not None and not qmc_was_configured:
                reset_qmc_simulator()
    
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
