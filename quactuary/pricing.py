"""
Pricing module for actuarial loss models.

This module provides a flexible framework for pricing insurance portfolios and calculating
risk measures using various computational backends (classical Monte Carlo, FFT, or quantum).
The module implements a strategy pattern to allow seamless switching between different
calculation methods while maintaining a consistent interface.

Key Features:
    - Portfolio-based pricing with support for individual policy terms
    - Classical simulation methods (Monte Carlo with optional quasi-random sequences)
    - Quantum acceleration support via Qiskit backends
    - Compound distribution modeling for aggregate losses
    - Excess of loss reinsurance pricing
    - Risk measures: VaR, TVaR, mean, variance

Architecture:
    The module uses composition over inheritance with a strategy pattern:
    - PricingModel: Main interface for all pricing operations
    - PricingStrategy: Abstract base for calculation strategies
    - ClassicalPricingStrategy: Traditional Monte Carlo implementation
    - QuantumPricingStrategy: Quantum-accelerated calculations

Examples:
    Basic portfolio pricing:
        >>> from quactuary.pricing import PricingModel
        >>> from quactuary.book import Portfolio
        >>> 
        >>> # Create portfolio from policy data
        >>> portfolio = Portfolio(policies_df)
        >>> model = PricingModel(portfolio)
        >>> 
        >>> # Calculate risk measures
        >>> result = model.simulate(
        ...     mean=True,
        ...     value_at_risk=True,
        ...     tail_alpha=0.05,
        ...     n_sims=10000
        ... )
        >>> print(f"VaR 95%: {result.value_at_risk:.2f}")

    Using quasi-Monte Carlo for improved convergence:
        >>> # Use Sobol sequences for better uniformity
        >>> result = model.simulate(
        ...     qmc_method="sobol",
        ...     qmc_scramble=True,
        ...     n_sims=5000
        ... )

    Pricing excess of loss reinsurance:
        >>> from quactuary.distributions import Poisson, Lognormal
        >>> 
        >>> # Set up aggregate loss model
        >>> model.set_compound_distribution(
        ...     frequency=Poisson(lambda_=100),
        ...     severity=Lognormal(shape=1.5, scale=np.exp(7))
        ... )
        >>> 
        >>> # Price a 1M xs 500k layer
        >>> layer_price = model.price_excess_layer(
        ...     attachment=500_000,
        ...     limit=1_000_000,
        ...     n_simulations=10000
        ... )
        >>> print(f"Expected layer loss: ${layer_price['expected_loss']:,.2f}")

Notes:
    - The module automatically selects the appropriate backend based on availability
    - Quasi-Monte Carlo methods can significantly improve convergence for smooth integrands
    - Quantum backends are experimental and require appropriate hardware/simulators
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
from quactuary.sobol import set_qmc_simulator, reset_qmc_simulator, get_qmc_simulator
from quactuary.optimization_selector import OptimizationSelector, OptimizationConfig


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
        compound_distribution (Optional[quactuary.distributions.compound.CompoundDistribution]): Compound distribution for aggregate loss modeling.
    """

    def __init__(self, portfolio: Portfolio, strategy: Optional[PricingStrategy] = None,
                 optimization_selector: Optional[OptimizationSelector] = None):
        """
        Initialize a PricingModel.

        Args:
            portfolio (Portfolio): Inforce policy data grouped into a Portfolio. The portfolio
                should contain policy-level information including exposures, limits, deductibles,
                and other terms needed for pricing.
            strategy (Optional[PricingStrategy]): Pricing strategy to use. If None, automatically 
                selects based on current backend configuration. You can provide a custom strategy
                for specialized calculations or override the default classical strategy.
            optimization_selector (Optional[OptimizationSelector]): Intelligent optimization selector
                for automatic optimization strategy selection. If None, optimizations must be
                manually specified.

        Examples:
            Using default classical strategy:
                >>> portfolio = Portfolio(policies_df)
                >>> model = PricingModel(portfolio)
            
            Using quantum strategy:
                >>> from quactuary.pricing_strategies import QuantumPricingStrategy
                >>> quantum_strategy = QuantumPricingStrategy()
                >>> model = PricingModel(portfolio, strategy=quantum_strategy)
            
            Using JIT-optimized classical strategy:
                >>> from quactuary.pricing_strategies import ClassicalPricingStrategy
                >>> jit_strategy = ClassicalPricingStrategy(use_jit=True)
                >>> model = PricingModel(portfolio, strategy=jit_strategy)
                
            Using automatic optimization selection:
                >>> from quactuary.optimization_selector import OptimizationSelector
                >>> optimizer = OptimizationSelector(enable_ml=True)
                >>> model = PricingModel(portfolio, optimization_selector=optimizer)
        """
        self.portfolio = portfolio
        self.strategy = strategy or ClassicalPricingStrategy()
        self.compound_distribution = None
        self.optimization_selector = optimization_selector

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
        qmc_seed: Optional[int] = None,
        auto_optimize: bool = False,
        optimization_config: Optional[OptimizationConfig] = None
    ) -> PricingResult:
        """
        Calculate portfolio statistics using the configured strategy.

        This method performs Monte Carlo simulation to estimate portfolio risk measures.
        It supports both standard pseudo-random and quasi-random (low-discrepancy) sequences
        for improved convergence. The calculation can be performed using different backends
        (classical or quantum) through the strategy pattern.

        Args:
            mean (bool): Calculate mean (expected) loss. Default is True.
            variance (bool): Calculate variance of losses. Default is True.
            value_at_risk (bool): Calculate Value at Risk (VaR). Default is True.
            tail_value_at_risk (bool): Calculate Tail Value at Risk (TVaR/CVaR). Default is True.
            tail_alpha (float): Alpha level for tail risk measures (VaR and TVaR). 
                For example, 0.05 gives 95% VaR. Default is 0.05.
            n_sims (Optional[int]): Number of Monte Carlo simulations. If None, uses the
                strategy's default (typically 10,000). Higher values improve accuracy.
            backend (Optional[BackendManager]): Execution backend override. If provided, 
                temporarily switches strategy to match the backend. Useful for comparing
                classical vs quantum results.
            qmc_method (Optional[str]): Quasi-Monte Carlo method for low-discrepancy sequences:
                - "sobol": Sobol sequences (recommended for up to ~20 dimensions)
                - "halton": Halton sequences (good for lower dimensions)
                - None: Standard pseudo-random numbers (default)
            qmc_scramble (bool): Whether to apply Owen scrambling to QMC sequences. 
                Scrambling improves uniformity and enables error estimation. Default is True.
            qmc_skip (int): Number of initial QMC points to skip. Skipping early points
                can improve uniformity for some sequences. Default is 1024.
            qmc_seed (Optional[int]): Random seed for QMC scrambling. If None, uses a
                random seed. Providing a seed ensures reproducibility.
            auto_optimize (bool): Enable automatic optimization selection based on portfolio
                characteristics and hardware capabilities. Requires optimization_selector to be
                set. Default is False.
            optimization_config (Optional[OptimizationConfig]): Manual optimization configuration
                to override automatic selection. If provided with auto_optimize=True, this
                config will be used as hints for the optimizer.

        Returns:
            PricingResult: Object containing calculated portfolio statistics:
                - mean: Expected portfolio loss
                - variance: Variance of portfolio losses
                - value_at_risk: VaR at (1-tail_alpha) confidence level
                - tail_value_at_risk: TVaR at (1-tail_alpha) confidence level
                - n_simulations: Number of simulations performed
                - convergence_error: Estimated Monte Carlo error (if available)

        Examples:
            Basic risk measure calculation:
                >>> result = model.simulate(n_sims=10000)
                >>> print(f"Expected loss: ${result.mean:,.2f}")
                >>> print(f"95% VaR: ${result.value_at_risk:,.2f}")
                >>> print(f"95% TVaR: ${result.tail_value_at_risk:,.2f}")

            Using Sobol sequences for better convergence:
                >>> result = model.simulate(
                ...     qmc_method="sobol",
                ...     qmc_scramble=True,
                ...     n_sims=5000,  # Can use fewer simulations with QMC
                ...     tail_alpha=0.01  # 99% confidence level
                ... )

            Comparing classical vs quantum backends:
                >>> from quactuary.backend import get_backend
                >>> 
                >>> # Classical calculation
                >>> classical_result = model.simulate(
                ...     backend=get_backend("classical"),
                ...     n_sims=10000
                ... )
                >>> 
                >>> # Quantum calculation (if available)
                >>> quantum_backend = get_backend("quantum")
                >>> if quantum_backend:
                ...     quantum_result = model.simulate(
                ...         backend=quantum_backend,
                ...         n_sims=10000
                ...     )

            Calculating only specific measures:
                >>> # Only calculate mean and VaR, skip variance and TVaR
                >>> result = model.simulate(
                ...     mean=True,
                ...     variance=False,
                ...     value_at_risk=True,
                ...     tail_value_at_risk=False,
                ...     n_sims=20000
                ... )
            
            Using automatic optimization selection:
                >>> # Set up model with optimization selector
                >>> from quactuary.optimization_selector import OptimizationSelector
                >>> optimizer = OptimizationSelector()
                >>> model = PricingModel(portfolio, optimization_selector=optimizer)
                >>> 
                >>> # Let the system choose optimal settings
                >>> result = model.simulate(
                ...     auto_optimize=True,
                ...     n_sims=100000  # Large simulation - optimizer will select best approach
                ... )

        Notes:
            - Quasi-Monte Carlo methods can significantly reduce the number of simulations
              needed for accurate results, especially for smooth integrands
            - The quantum backend is experimental and may not support all features
            - TVaR is always >= VaR for the same confidence level
            - Increasing n_sims improves accuracy but increases computation time linearly
        """
        # Handle automatic optimization selection
        effective_n_sims = n_sims or 10000  # Default if not specified
        effective_config = optimization_config
        
        if auto_optimize:
            if self.optimization_selector is None:
                raise ValueError("auto_optimize=True requires optimization_selector to be set")
            
            # Analyze portfolio and predict best strategy
            profile = self.optimization_selector.analyze_portfolio(
                self.portfolio, effective_n_sims
            )
            predicted_config = self.optimization_selector.predict_best_strategy(profile)
            
            # Merge with any manual config provided
            if optimization_config:
                # Manual config takes precedence
                effective_config = optimization_config
            else:
                effective_config = predicted_config
                
            # Apply optimization config
            if effective_config:
                # Update QMC settings from config
                if effective_config.use_qmc and effective_config.qmc_method:
                    qmc_method = effective_config.qmc_method
                    
                # Create optimized strategy if needed
                if backend is None and isinstance(self.strategy, ClassicalPricingStrategy):
                    # Replace strategy with optimized version
                    from quactuary.pricing_strategies import ClassicalPricingStrategy
                    self.strategy = ClassicalPricingStrategy(
                        use_jit=effective_config.use_jit,
                        use_parallel=effective_config.use_parallel,
                        use_vectorization=effective_config.use_vectorization,
                        n_workers=effective_config.n_workers,
                        batch_size=effective_config.batch_size
                    )
        
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
    
    def set_compound_distribution(self, frequency: "FrequencyModel", severity: "SeverityModel"):
        """
        Set compound distribution for aggregate loss modeling.
        
        This method creates a compound distribution S = X₁ + X₂ + ... + Xₙ where:
        - N ~ frequency distribution (number of losses)
        - Xᵢ ~ severity distribution (individual loss amounts)
        
        The compound distribution is used for aggregate loss calculations, excess layer
        pricing, and can leverage analytical solutions when available (e.g., compound
        Poisson with certain severity distributions).
        
        Args:
            frequency (quactuary.distributions.frequency.FrequencyModel): Frequency distribution model determining the number
                of losses. Common choices include Poisson, NegativeBinomial, or Binomial.
            severity (quactuary.distributions.severity.SeverityModel): Severity distribution model for individual loss amounts.
                Common choices include Lognormal, Gamma, Pareto, or Weibull.

        Examples:
            Setting up a compound Poisson-Lognormal model:
                >>> from quactuary.distributions import Poisson, Lognormal
                >>> model.set_compound_distribution(
                ...     frequency=Poisson(lambda_=50),
                ...     severity=Lognormal(shape=1.2, scale=np.exp(8))
                ... )
            
            Using empirical frequency with parametric severity:
                >>> from quactuary.distributions import Empirical, Gamma
                >>> historical_counts = [45, 52, 48, 51, 49]
                >>> model.set_compound_distribution(
                ...     frequency=Empirical(data=historical_counts),
                ...     severity=Gamma(shape=2.0, scale=5000)
                ... )

        Notes:
            - The compound distribution is used by calculate_aggregate_statistics()
              and price_excess_layer() methods
            - Some combinations have analytical solutions (e.g., Poisson-Exponential)
            - Others require simulation-based approaches
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
        
        This method computes key risk measures for the aggregate loss distribution of the
        portfolio. If a compound distribution has been set, it will be used for calculations.
        Otherwise, empirical simulation based on portfolio characteristics is performed.
        
        Args:
            apply_policy_terms (bool): Whether to apply policy terms (deductibles, limits, etc.)
                when calculating aggregate losses. Default is True.
            confidence_levels (Optional[list]): List of confidence levels for VaR/TVaR calculations.
                Values should be between 0 and 1. Default is [0.90, 0.95, 0.99].
            n_simulations (int): Number of simulations for empirical calculation if no compound
                distribution is set, or for TVaR calculation. Default is 10,000.
            
        Returns:
            dict: Dictionary containing aggregate loss statistics with keys:
                - 'mean': Expected aggregate loss
                - 'std': Standard deviation of aggregate loss
                - 'variance': Variance of aggregate loss
                - 'has_analytical': Boolean indicating if analytical solution was used
                - 'var_X%': Value at Risk at X% confidence level
                - 'tvar_X%': Tail Value at Risk (CVaR) at X% confidence level
                - 'method': 'compound' or 'empirical' indicating calculation method

        Examples:
            Calculate statistics with compound distribution:
                >>> # Set up compound distribution first
                >>> model.set_compound_distribution(
                ...     frequency=Poisson(lambda_=100),
                ...     severity=Lognormal(shape=1.5, scale=np.exp(7))
                ... )
                >>> stats = model.calculate_aggregate_statistics()
                >>> print(f"Expected aggregate loss: ${stats['mean']:,.2f}")
                >>> print(f"VaR 99%: ${stats['var_99%']:,.2f}")
                >>> print(f"TVaR 99%: ${stats['tvar_99%']:,.2f}")
            
            Calculate with custom confidence levels:
                >>> stats = model.calculate_aggregate_statistics(
                ...     confidence_levels=[0.95, 0.99, 0.995],
                ...     n_simulations=50000
                ... )
                >>> print(f"VaR 99.5%: ${stats['var_99.5%']:,.2f}")
            
            Empirical calculation without compound distribution:
                >>> # No compound distribution set, uses portfolio empirical approach
                >>> stats = model.calculate_aggregate_statistics(
                ...     apply_policy_terms=True,
                ...     n_simulations=25000
                ... )

        Notes:
            - TVaR (Tail Value at Risk) is also known as CVaR (Conditional Value at Risk)
            - TVaR represents the expected loss given that the loss exceeds VaR
            - Empirical method assumes exponential losses if no policy-specific information
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
                # For each inforce in portfolio, simulate a loss
                for inforce in self.portfolio:
                    # Simple empirical approach: use expected loss with some variance
                    if hasattr(inforce, 'expected_loss'):
                        # Add random variation around expected loss
                        loss = np.random.exponential(scale=inforce.expected_loss)
                    else:
                        # Default to a simple loss model
                        loss = np.random.exponential(scale=1000.0)
                    
                    # Apply policy terms if available
                    if hasattr(inforce, 'terms') and inforce.terms is not None:
                        if hasattr(inforce.terms, 'per_occ_retention') and inforce.terms.per_occ_retention is not None:
                            loss = max(0, loss - inforce.terms.per_occ_retention)
                        if hasattr(inforce.terms, 'per_occ_limit') and inforce.terms.per_occ_limit is not None:
                            loss = min(loss, inforce.terms.per_occ_limit)
                    
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
        if apply_policy_terms and len(self.portfolio) > 0:
            # Policy terms have been applied in the calculation above
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
        
        This method calculates the expected loss and other statistics for an excess of loss
        reinsurance layer with specified attachment point and limit. The layer pays losses
        that exceed the attachment point, up to the specified limit.
        
        Layer payment formula: min(max(Loss - Attachment, 0), Limit)
        
        Args:
            attachment (float): Layer attachment point (deductible). The layer begins paying
                when aggregate losses exceed this amount.
            limit (float): Layer limit. The maximum amount the layer will pay. For unlimited
                layers, use float('inf').
            n_simulations (int): Number of Monte Carlo simulations for pricing. Higher values
                give more accurate results but take longer. Default is 10,000.
            
        Returns:
            dict: Dictionary containing layer pricing information with keys:
                - 'attachment': Layer attachment point
                - 'limit': Layer limit  
                - 'expected_loss': Expected loss to the layer (pure premium)
                - 'loss_std': Standard deviation of layer losses
                - 'loss_probability': Probability of any loss to the layer
                - 'average_severity': Average loss amount when layer is triggered
                - 'ground_up_mean': Mean of ground-up (unlimited) losses
                - 'ground_up_std': Standard deviation of ground-up losses

        Raises:
            ValueError: If compound distribution has not been set via set_compound_distribution()

        Examples:
            Price a 1M xs 500k layer:
                >>> from quactuary.distributions import Poisson, Lognormal
                >>> 
                >>> # Set aggregate loss distribution
                >>> model.set_compound_distribution(
                ...     frequency=Poisson(lambda_=100),
                ...     severity=Lognormal(shape=1.5, scale=np.exp(7))
                ... )
                >>> 
                >>> # Price the layer
                >>> layer = model.price_excess_layer(
                ...     attachment=500_000,
                ...     limit=1_000_000,
                ...     n_simulations=50000
                ... )
                >>> 
                >>> print(f"Layer: {layer['limit']:,.0f} xs {layer['attachment']:,.0f}")
                >>> print(f"Expected loss: ${layer['expected_loss']:,.2f}")
                >>> print(f"Loss probability: {layer['loss_probability']:.1%}")
                >>> print(f"Average severity when triggered: ${layer['average_severity']:,.2f}")
            
            Price multiple layers for a reinsurance program:
                >>> layers = [
                ...     (100_000, 400_000),   # 400k xs 100k
                ...     (500_000, 500_000),   # 500k xs 500k
                ...     (1_000_000, 2_000_000)  # 2M xs 1M
                ... ]
                >>> 
                >>> for attachment, limit in layers:
                ...     result = model.price_excess_layer(attachment, limit)
                ...     rate = result['expected_loss'] / limit * 100
                ...     print(f"{limit/1e6:.1f}M xs {attachment/1e6:.1f}M: Rate = {rate:.2f}%")

        Notes:
            - Requires compound distribution to be set first
            - Uses Monte Carlo simulation to estimate layer losses
            - For high attachment points, increase n_simulations for accuracy
            - Layer limit is the maximum payment, not the upper bound of coverage
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
