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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only imported for type‐checking, avoids circular import at runtime
    from quactuary.pricing import PricingResult


class ClassicalPricingModel ():
    """
    Mixin providing Classical Monte Carlo simulation capabilities.

    Include this mixin in model classes to enable classical algorithms.
    """

    def __init__(self):
        pass

    def calculate_portfolio_statistics(
        self,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.95
    ) -> PricingResult:
        """
        Compute mean loss for the portfolio using Classical Monte Carlo.

        Returns:
            Expected loss result.
            Simulated loss values for the portfolio.
        """
        error_message = "TODO: Implement Classical Monte Carlo simulation and pull statistics."
        raise NotImplementedError(error_message)
