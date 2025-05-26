"""
Data types and result containers for the quactuary package.

This module defines standardized data structures for representing results from
actuarial calculations. These types provide a consistent interface for passing
data between different components of the framework and presenting results to users.

Key Components:
    - PricingResult: Container for portfolio pricing and risk measure results
    - Additional result types may be added for specific calculations

Design Principles:
    - Use dataclasses for clear, self-documenting structures
    - Include both point estimates and uncertainty measures
    - Store metadata for reproducibility and auditing
    - Support both single values and distributional results

Examples:
    Creating a pricing result:
        >>> from quactuary.datatypes import PricingResult
        >>> import pandas as pd
        >>> 
        >>> result = PricingResult(
        ...     estimates={'mean': 50000, 'var_95': 75000},
        ...     intervals={'mean': (49000, 51000)},
        ...     samples=pd.Series([48000, 52000, 49500, ...]),
        ...     metadata={'n_sims': 10000, 'method': 'monte_carlo'}
        ... )

    Accessing results:
        >>> print(f"Expected loss: ${result.estimates['mean']:,.2f}")
        >>> print(f"95% VaR: ${result.estimates['var_95']:,.2f}")
        >>> print(f"Simulations performed: {result.metadata['n_sims']}")

Notes:
    - Results are designed to be self-contained and informative
    - Metadata should include all parameters needed to reproduce results
    - Samples can be used for additional analysis or visualization
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PricingResult:
    """
    Container for results from actuarial pricing and risk calculations.

    This dataclass provides a standardized format for returning results from
    various pricing models and strategies. It includes point estimates, confidence
    intervals, raw simulation data, and metadata for full transparency and
    reproducibility.

    The structure supports both simple scalar results and complex distributional
    outputs, making it suitable for a wide range of actuarial applications from
    basic expected value calculations to sophisticated risk measure estimations.

    Attributes:
        estimates (dict): Point estimates for calculated statistics. Common keys include:
            - 'mean': Expected loss or value
            - 'variance': Variance of the loss distribution
            - 'VaR': Value at Risk at specified confidence level
            - 'TVaR': Tail Value at Risk (Conditional VaR)
            - Additional custom statistics as needed
            
        intervals (dict): Confidence intervals for the estimates. Keys match those
            in estimates, values are tuples (lower, upper). May be empty if
            intervals are not calculated or not applicable.
            
        samples (Optional[pd.Series]): Raw simulation samples used to derive the
            estimates. Useful for additional analysis, visualization, or validation.
            None if samples are not retained (e.g., analytical calculations).
            
        metadata (dict): Additional information about the calculation. Common keys:
            - 'n_sims': Number of simulations performed
            - 'run_date': Timestamp of calculation
            - 'method': Calculation method used ('monte_carlo', 'analytical', etc.)
            - 'tail_alpha': Alpha level for tail risk measures
            - 'backend': Computational backend used
            - Additional parameters for reproducibility

    Examples:
        Typical Monte Carlo result:
            >>> result = PricingResult(
            ...     estimates={
            ...         'mean': 125000.0,
            ...         'variance': 1e9,
            ...         'VaR': 180000.0,
            ...         'TVaR': 210000.0
            ...     },
            ...     intervals={
            ...         'mean': (124000.0, 126000.0),
            ...         'VaR': (178000.0, 182000.0)
            ...     },
            ...     samples=pd.Series(simulation_data),
            ...     metadata={
            ...         'n_sims': 100000,
            ...         'tail_alpha': 0.05,
            ...         'method': 'monte_carlo',
            ...         'run_date': datetime.now()
            ...     }
            ... )

        Analytical calculation result:
            >>> result = PricingResult(
            ...     estimates={'mean': 50000.0, 'variance': 2.5e8},
            ...     intervals={},  # No sampling uncertainty
            ...     samples=None,  # No simulations performed
            ...     metadata={'method': 'analytical', 'distribution': 'compound_poisson'}
            ... )

        Accessing results:
            >>> expected_loss = result.estimates['mean']
            >>> var_95 = result.estimates.get('VaR', None)
            >>> if result.samples is not None:
            ...     empirical_percentile = np.percentile(result.samples, 95)

    Notes:
        - All monetary values should be in consistent units (typically base currency)
        - Estimates dict should never be empty; at minimum include mean or a primary metric
        - Intervals may be empty for deterministic or analytical calculations
        - Samples storage is optional and may impact memory for large simulations
        - Metadata should be comprehensive enough to reproduce the calculation
    """
    estimates: dict
    intervals: dict
    samples: Optional[pd.Series]
    metadata: dict
    
    @property
    def mean(self) -> float:
        """Get mean estimate for backward compatibility."""
        return self.estimates.get('mean', 0.0)
    
    @property
    def variance(self) -> float:
        """Get variance estimate for backward compatibility."""
        return self.estimates.get('variance', 0.0)
    
    @property
    def value_at_risk(self) -> float:
        """Get VaR estimate for backward compatibility."""
        return self.estimates.get('VaR', self.estimates.get('value_at_risk', 0.0))
    
    @property
    def tail_value_at_risk(self) -> float:
        """Get TVaR estimate for backward compatibility."""
        return self.estimates.get('TVaR', self.estimates.get('tail_value_at_risk', 0.0))
