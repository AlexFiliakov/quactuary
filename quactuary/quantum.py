"""
Quantum actuarial integration module.

Centralizes quantum computing support across actuarial models, providing consistent
circuit construction, execution, and result handling.

Examples:
    >>> from quactuary.quantum import QuantumModelMixin, QuantumResult
    >>> class MyModel(QuantumModelMixin):
    ...     pass
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # only imported for type‐checking, avoids circular import at runtime
    from quactuary.pricing import PricingResult


class QuantumPricingModel ():
    """
    Mixin providing quantum circuit creation and execution capabilities.

    Include this mixin in model classes to enable quantum algorithms via Qiskit.
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
        result = []
        if mean:
            mean_result = self.mean_loss()
            result.append(mean_result)
        if variance:
            variance_result = self.variance()
            result.append(variance_result)
        if value_at_risk:
            VaR_result = self.value_at_risk(alpha=tail_alpha)
            result.append(VaR_result)
        if tail_value_at_risk:
            TVaR_result = self.tail_value_at_risk(alpha=tail_alpha)
            result.append(TVaR_result)

        error_message = "TODO: Implement portfolio statistics reporting."
        raise NotImplementedError(error_message)
        from quactuary.pricing import PricingResult
        return PricingResult()

    # These are probably going to be separate circuits until we see how to combine them.
    def mean_loss(self):
        """
        Builds the quantum circuit to compute mean loss for the portfolio using quantum circuits.

        Returns:
            Expected loss result.
        """
        error_message = "TODO: Implement mean loss quantum circuit."
        raise NotImplementedError(error_message)
        return 0.0

    def variance(self):
        """
        Builds the quantum circuit to compute variance for the portfolio using quantum circuits.

        Returns:
            Expected variance result.
        """
        error_message = "TODO: Implement loss variance quantum circuit."
        raise NotImplementedError(error_message)
        return 0.0

    def value_at_risk(self, alpha: float = 0.95):
        """
        Builds the quantum circuit to compute the value at risk (VaR) for the portfolio using quantum circuits.

        Args:
            alpha (float): Confidence level for VaR calculation.

        Returns:
            Expected value at risk result.
        """
        error_message = "TODO: Implement VaR loss quantum circuit."
        raise NotImplementedError(error_message)
        return 0.0

    def tail_value_at_risk(self, alpha: float = 0.95):
        """
        Builds the quantum circuit to compute the tail value at risk (TVaR) for the portfolio using quantum circuits.

        Args:
            alpha (float): Confidence level for VaR calculation.

        Returns:
            Expected tail value at risk result.
        """
        error_message = "TODO: Implement TVaR loss quantum circuit."
        raise NotImplementedError(error_message)
        return 0.0
