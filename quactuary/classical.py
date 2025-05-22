"""
Classical actuarial integration module.

Centralizes classical support across actuarial models, providing consistent
algorithm execution and result handling.

Examples:
    >>> from quactuary.classical import ClassicalModelMixin, ClassicalResult
    >>> class MyModel(ClassicalModelMixin):
    ...     pass
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from quactuary.book import Portfolio
from quactuary.datatypes import PricingResult


class ClassicalPricingModel ():
    """
    Mixin providing Classical Monte Carlo simulation capabilities.

    Include this mixin in model classes to enable classical algorithms.
    """

    def __init__(self):
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
        Compute mean loss for the portfolio using Classical Monte Carlo.

        Returns:
            Expected loss result.
            Simulated loss values for the portfolio.
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
