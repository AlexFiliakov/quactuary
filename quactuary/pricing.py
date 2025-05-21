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

from typing import Optional

from qiskit.providers import Backend, BackendV1

from quactuary.backend import BackendManager, ClassicalBackend, set_backend
from quactuary.book import Portfolio
from quactuary.classical import ClassicalPricingModel
from quactuary.quantum import QuantumPricingModel


class PricingResult():
    """
    Container for results from pricing computations.

    Models can convert these results into user-facing formats (e.g., numbers or DataFrames).

    Attributes:
        estimates (dict[str, float]): Point estimates for various statistics.
        intervals (dict[str, tuple[float, float]]): Confidence intervals for estimates.
        samples (Optional[np.ndarray]): Raw samples obtained from execution.
        metadata (dict): Additional run details.
    """
    @property
    def estimates(self):
        raise NotImplementedError(
            "TODO: Implement the pricing result dataclass.")


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

    def calculate_portfolio_statistics(
            self,
            mean: bool = True,
            variance: bool = True,
            value_at_risk: bool = True,
            tail_value_at_risk: bool = True,
            tail_alpha: float = 0.95,
            backend: Optional[BackendManager] = None) -> PricingResult:
        """
        Calculate portfolio statistics based on the selected methods.

        Args:
            mean (bool): Calculate mean loss.
            variance (bool): Calculate variance.
            value_at_risk (bool): Calculate value at risk.
            tail_value_at_risk (bool): Calculate tail value at risk.
            backend (Optional[BackendManager]): Execution backend override.
        """

        if backend is None:
            if self.portfolio.backend is None:
                set_backend()
            cur_backend = self.portfolio.backend.backend
        else:
            cur_backend = backend.backend

        if isinstance(cur_backend, ClassicalBackend):
            return ClassicalPricingModel.calculate_portfolio_statistics(
                self, mean, variance, value_at_risk, tail_value_at_risk, tail_alpha)
        if isinstance(cur_backend, Backend) or \
                isinstance(cur_backend, BackendV1):
            return QuantumPricingModel.calculate_portfolio_statistics(
                self, mean, variance, value_at_risk, tail_value_at_risk, tail_alpha)
        else:
            error_str = "Unsupported backend type. Must be a Qiskit or classical backend."
            raise ValueError(error_str)
