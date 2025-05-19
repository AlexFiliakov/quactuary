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

from quactuary import get_backend
from quactuary.book import Inforce, Portfolio
from quactuary.quantum import QuantumModelMixin


class ActuarialModel(QuantumModelMixin):
    """
    Base class for actuarial pricing models with optional quantum support.

    Provides common interface for portfolio-based loss models using classical or quantum backends.

    Args:
        inforce (Inforce or Portfolio): Policy inforce data.
        deductible (Optional[float]): Layer deductible.
        limit (Optional[float]): Layer limit.
        backend (Optional[BackendManager]): Execution backend override.
        **kw: Additional model-specific settings.

    Attributes:
        portfolio (Portfolio): Wrapped inforce portfolio.
        layer_deductible (Optional[float]): Deductible for the layer.
        layer_limit (Optional[float]): Limit for the layer.
        backend (BackendManager): Backend manager for execution.
    """
    def __init__(self, book, deductible=None, limit=None, backend=None, **kw):
        """
        Initialize an ActuarialModel.

        Args:
            inforce (Inforce or Portfolio): Policy inforce data.
            deductible (Optional[float]): Layer deductible.
            limit (Optional[float]): Layer limit.
            backend (Optional[BackendManager]): Execution backend.
            **kw: Additional settings (ignored).

        Raises:
            TypeError: If inforce is not an Inforce or Portfolio.
        """
        if isinstance(book, Inforce):
            self.portfolio = Portfolio([book])
        elif isinstance(book, Portfolio):
            self.portfolio = book
        else:
            raise TypeError("Need Inforce or Portfolio")

        # optional layer terms that sit above policy terms
        self.layer_deductible = deductible
        self.layer_limit = limit

        # Use the global backend manager
        if backend is not None:
            self.backend = backend
        else:
            self.backend = get_backend()


class ExcessLossModel(ActuarialModel):
    """
    Pricing model for aggregate excess loss.

    Computes the loss exceeding a deductible up to a layer limit for an insurance portfolio.

    Args:
        inforce (Inforce or Portfolio): Portfolio data.
        deductible (Optional[float]): Layer deductible.
        limit (Optional[float]): Layer limit.
        **kw: Additional settings.

    Examples:
        >>> model = ExcessLossModel(inforce, deductible=1000, limit=10000)
        >>> losses = model.compute_excess_loss()
    """
    def __init__(self, inforce, deductible=None, limit=None, **kw):
        super().__init__(inforce, deductible, limit, **kw)

    def compute_excess_loss(self, backend=None):
        """
        Compute aggregate excess loss for the portfolio.

        Args:
            backend (Optional[BackendManager]): Execution backend override.

        Returns:
            Any: Excess loss result (e.g., numeric array or DataFrame).

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("compute_excess_loss is not implemented.")


class RiskMeasureModel(ActuarialModel):
    """
    Pricing model for risk measures (VaR, TVaR).

    Computes quantile-based risk metrics for insurance portfolios, with optional quantum acceleration.

    Args:
        inforce (Inforce or Portfolio): Portfolio data.
        deductible (Optional[float]): Layer deductible.
        limit (Optional[float]): Layer limit.
        **kw: Additional settings.

    Examples:
        >>> rm = RiskMeasureModel(inforce)
        >>> var = rm.value_at_risk(alpha=0.99)
    """
    def __init__(self, inforce, deductible=None, limit=None, **kw):
        super().__init__(inforce, deductible, limit, **kw)

    def value_at_risk(self, alpha=0.95, backend=None):
        """
        Compute the Value at Risk (VaR) at the given confidence level.

        Args:
            alpha (float, optional): Confidence level (0 < alpha < 1). Defaults to 0.95.
            backend (Optional[BackendManager]): Execution backend override.

        Returns:
            float: Estimated VaR value.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("value_at_risk is not implemented.")

    def tail_value_at_risk(self, alpha=0.95, backend=None):
        """
        Compute the Tail Value at Risk (TVaR) at the given confidence level.

        Args:
            alpha (float, optional): Confidence level (0 < alpha < 1). Defaults to 0.95.
            backend (Optional[BackendManager]): Execution backend override.

        Returns:
            float: Estimated TVaR value.

        Raises:
            NotImplementedError: Method not yet implemented.
        """
        raise NotImplementedError("tail_value_at_risk is not implemented.")
