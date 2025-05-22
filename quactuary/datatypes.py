"""
This module defines the datatypes used for outputs in the Quactuary project.
"""
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class PricingResult():
    """
    Container for results from pricing computations.

    Models can convert these results into user-facing formats (e.g., numbers or DataFrames).

    Attributes:
        estimates (pd.Series[float]): Point estimates for various statistics.
        intervals (pd.Series[tuple[float, float]]): Confidence intervals for estimates.
        samples (Optional[pd.Series[pd.Series[float]]]): Simulated samples.
        metadata (dict): Additional run details.
    """
    estimates: dict
    intervals: dict
    samples: Optional[pd.Series]
    metadata: dict
