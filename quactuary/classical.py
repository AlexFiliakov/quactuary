"""
Classical actuarial integration module.

This module provides classical Monte Carlo simulation capabilities for actuarial calculations.
It implements standard simulation-based methods for computing portfolio risk measures including
mean, variance, Value at Risk (VaR), and Tail Value at Risk (TVaR).

The module serves as the foundation for classical actuarial computations within the quactuary
framework, offering a consistent interface for risk measure calculations that can be used
standalone or as part of the larger pricing framework.

Key Features:
    - Monte Carlo simulation for portfolio loss distributions
    - Standard risk measure calculations (mean, variance, VaR, TVaR)
    - Integration with Portfolio objects for policy-level detail
    - Support for both single-value and distributional results

Examples:
    Direct usage with portfolio:
        >>> from quactuary.classical import ClassicalPricingModel
        >>> from quactuary.book import Portfolio
        >>> 
        >>> # Create portfolio and model
        >>> portfolio = Portfolio(policies_df)
        >>> model = ClassicalPricingModel()
        >>> 
        >>> # Calculate risk measures
        >>> result = model.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     mean=True,
        ...     value_at_risk=True,
        ...     tail_alpha=0.05,
        ...     n_sims=10000
        ... )
        >>> print(f"Expected loss: ${result.estimates['mean']:,.2f}")
        >>> print(f"95% VaR: ${result.estimates['VaR']:,.2f}")

    Calculating specific risk measures:
        >>> # Only calculate TVaR at 99% confidence
        >>> result = model.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     mean=False,
        ...     variance=False,
        ...     value_at_risk=False,
        ...     tail_value_at_risk=True,
        ...     tail_alpha=0.01,  # 99% confidence
        ...     n_sims=25000
        ... )
        >>> print(f"99% TVaR: ${result.estimates['TVaR']:,.2f}")

Notes:
    - This module implements the classical (non-quantum) calculation methods
    - For quantum-accelerated calculations, see the quantum module
    - TVaR is also known as Conditional Value at Risk (CVaR) or Expected Shortfall
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quactuary.book import Portfolio
from quactuary.datatypes import PricingResult


class ClassicalPricingModel():
    """
    Classical Monte Carlo simulation model for actuarial pricing.

    This class provides standard Monte Carlo simulation capabilities for calculating
    portfolio risk measures. It serves as the classical (non-quantum) implementation
    of pricing calculations within the quactuary framework.

    The model integrates with Portfolio objects to simulate loss distributions and
    compute various risk metrics including expected loss, variance, Value at Risk (VaR),
    and Tail Value at Risk (TVaR).

    Attributes:
        None currently - this is a stateless calculation engine.

    Examples:
        Basic usage:
            >>> from quactuary.book import Portfolio
            >>> portfolio = Portfolio(policies_df)
            >>> model = ClassicalPricingModel()
            >>> result = model.calculate_portfolio_statistics(
            ...     portfolio=portfolio,
            ...     n_sims=10000
            ... )

        Integration with pricing framework:
            >>> from quactuary.pricing import PricingModel
            >>> from quactuary.pricing_strategies import ClassicalPricingStrategy
            >>> 
            >>> # ClassicalPricingModel is used internally by ClassicalPricingStrategy
            >>> strategy = ClassicalPricingStrategy()
            >>> pricing_model = PricingModel(portfolio, strategy=strategy)

    Notes:
        - This class can be used directly or through the PricingModel interface
        - For large portfolios, consider using the JIT-optimized version for better performance
        - The class is designed to be stateless for thread safety
    """

    def __init__(self):
        """
        Initialize the Classical Pricing Model.

        This is a lightweight initialization as the model is stateless and performs
        all calculations on-demand based on the provided portfolio.

        Examples:
            >>> model = ClassicalPricingModel()
            >>> # Model is ready to use for calculations
        """
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
        *args,
        **kwargs
    ) -> PricingResult:
        """
        Calculate portfolio risk statistics using Classical Monte Carlo simulation.

        This method performs Monte Carlo simulation on the provided portfolio to estimate
        various risk measures. It generates loss scenarios and computes statistics based
        on the empirical distribution of simulated losses.

        Args:
            portfolio (Portfolio): The portfolio object containing policy information
                and loss distributions. Must have a simulate() method.
            mean (bool): Whether to calculate the mean (expected) loss. Default is True.
            variance (bool): Whether to calculate the variance of losses. Default is True.
            value_at_risk (bool): Whether to calculate Value at Risk (VaR). Default is True.
            tail_value_at_risk (bool): Whether to calculate Tail Value at Risk (TVaR/CVaR).
                Default is True.
            tail_alpha (float): The tail probability for VaR and TVaR calculations.
                For example, 0.05 corresponds to 95% VaR/TVaR. Default is 0.05.
            n_sims (Optional[int]): Number of Monte Carlo simulations to perform.
                If None, defaults to 1. Higher values provide more accurate estimates.
            *args: Additional positional arguments (for compatibility).
            **kwargs: Additional keyword arguments (for compatibility).

        Returns:
            PricingResult: Object containing the calculated statistics with the following structure:
                - estimates (dict): Dictionary of calculated risk measures:
                    - 'mean': Expected portfolio loss (if requested)
                    - 'variance': Variance of portfolio losses (if requested)
                    - 'VaR': Value at Risk at (1-tail_alpha) confidence (if requested)
                    - 'TVaR': Tail Value at Risk at (1-tail_alpha) confidence (if requested)
                - intervals (dict): Confidence intervals (currently empty)
                - samples (pd.Series): The simulated loss values
                - metadata (dict): Additional information:
                    - 'n_sims': Number of simulations performed
                    - 'run_date': Timestamp of calculation
                    - 'tail_alpha': Alpha level used for tail measures (if applicable)

        Examples:
            Calculate all risk measures:
                >>> portfolio = Portfolio(policies_df)
                >>> model = ClassicalPricingModel()
                >>> result = model.calculate_portfolio_statistics(
                ...     portfolio=portfolio,
                ...     n_sims=10000
                ... )
                >>> print(f"Mean: ${result.estimates['mean']:,.2f}")
                >>> print(f"Std Dev: ${np.sqrt(result.estimates['variance']):,.2f}")
                >>> print(f"95% VaR: ${result.estimates['VaR']:,.2f}")
                >>> print(f"95% TVaR: ${result.estimates['TVaR']:,.2f}")

            Calculate only tail risk measures at 99% confidence:
                >>> result = model.calculate_portfolio_statistics(
                ...     portfolio=portfolio,
                ...     mean=False,
                ...     variance=False,
                ...     tail_alpha=0.01,  # 99% confidence
                ...     n_sims=50000
                ... )
                >>> print(f"99% VaR: ${result.estimates['VaR']:,.2f}")
                >>> print(f"99% TVaR: ${result.estimates['TVaR']:,.2f}")

            Access the simulated samples for further analysis:
                >>> result = model.calculate_portfolio_statistics(portfolio, n_sims=1000)
                >>> samples = result.samples
                >>> # Plot histogram of losses
                >>> samples.hist(bins=50)

        Notes:
            - TVaR is the expected loss given that the loss exceeds VaR
            - For single-valued simulations (n_sims=1), variance is set to 0
            - The method handles both Series and scalar simulation results
            - Computational cost scales linearly with n_sims
        """
        if n_sims is None:
            n_sims = 1

        simulations = portfolio.simulate(n_sims=n_sims)

        result = PricingResult(
            estimates={},
            intervals={},
            samples=(simulations if isinstance(simulations, pd.Series)
                     else pd.Series(simulations)),
            metadata={
                "n_sims": n_sims,
                "run_date": datetime.now(),
            }
        )

        if mean:
            if isinstance(simulations, pd.Series):
                # Simulation is a pandas Series
                mean_result = np.mean(simulations)
            else:
                # Simulation is a single value
                mean_result = simulations
            result.estimates['mean'] = mean_result
        if variance:
            if isinstance(simulations, pd.Series):
                # Simulation is a pandas Series
                variance_result = np.var(simulations)
            else:
                # Simulation is a single value
                variance_result = 0.0
            result.estimates['variance'] = variance_result
        if tail_value_at_risk:
            if isinstance(simulations, pd.Series):
                # Simulation is a pandas Series
                VaR_result = np.percentile(simulations, 100 * (1 - tail_alpha))
                TVaR_result = np.mean(simulations[simulations >= VaR_result])
            else:
                # Simulation is a single value
                VaR_result = 0.0
                TVaR_result = 0.0
            if value_at_risk:
                result.estimates['VaR'] = VaR_result
            result.estimates['TVaR'] = TVaR_result
        elif value_at_risk:
            # Avoid computing twice if TVaR is also requested
            if isinstance(simulations, pd.Series):
                # Simulation is a pandas Series
                VaR_result = np.percentile(simulations, 100 * (1 - tail_alpha))
            else:
                # Simulation is a single value
                VaR_result = 0.0
            result.estimates['VaR'] = VaR_result

        if value_at_risk or tail_value_at_risk:
            result.metadata['tail_alpha'] = tail_alpha

        return result
