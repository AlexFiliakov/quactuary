"""
Pricing module for Excess Loss Model

Notes:
-------
The focus is on using classical methods (Monte Carlo, FFT) by calling external libs, with an option to accelerate via quantum algorithms.
"""

from quactuary.entities import Inforce, Portfolio
from quactuary.quantum import QuantumModelMixin

from quactuary import get_backend


class ActuarialModel(QuantumModelMixin):
    """
    Base class for actuarial models in the quActuary framework.
    This class serves as a foundation for building various actuarial models
    that can be used for pricing, risk assessment, and other actuarial tasks.
    It provides a common interface and shared functionality for all models.

    Parameters:
    ----------
    - `inforce`: An Inforce object or a Portfolio object.
    - `deductible`: Optional deductible for the layer.
    - `limit`: Optional limit for the layer.
    - `backend`: Optional backend for quantum simulations.
    - `**kw`: Additional keyword arguments for model configuration.

    Attributes:
    ----------
    - `portfolio`: The portfolio of inforce policies.
    - `layer_deductible`: Optional deductible for the layer.
    - `layer_limit`: Optional limit for the layer.
    - `backend`: The backend manager for quantum simulations.
    """
    def __init__(self, inforce, deductible=None, limit=None, backend=None, **kw):
        """
        Initialize the Quantum Actuarial Model.
        Parameters:
        ----------
        - `inforce`: An Inforce object or a Portfolio object.
        - `deductible`: Optional deductible for the layer.
        - `limit`: Optional limit for the layer.
        """
        if isinstance(inforce, Inforce):
            self.portfolio = Portfolio([inforce])
        elif isinstance(inforce, Portfolio):
            self.portfolio = inforce
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
    Excess Loss Model for pricing.
    This model is designed to compute excess loss for a portfolio.

    Parameters:
    ----------
    - `inforce`: An Inforce object or a Portfolio object.
    - `deductible`: Optional deductible for the layer.
    - `limit`: Optional limit for the layer.
    """

    def __init__(self, inforce, deductible=None, limit=None, **kw):
        super().__init__(inforce, deductible, limit, **kw)

    def compute_excess_loss(self, backend=None):
        """
        Compute the excess loss for the portfolio using quantum simulation.

        Parameters:
        ----------
        - `backend`: Optional backend for overriding the BackendManager settings.
        """
        pass


class RiskMeasureModel(ActuarialModel):
    """
    Risk Measure Model for pricing.
    This model is designed to compute risk measures such as Value at Risk (VaR)
    and Tail Value at Risk (TVaR) using quantum simulation.

    It extends the QuantumActuarialModel class to provide specific implementations for these risk measures.

    Parameters:
    ----------
    - `inforce`: An Inforce object or a Portfolio object.
    - `deductible`: Optional deductible for the layer.
    - `limit`: Optional limit for the layer.
    """

    def __init__(self, inforce, deductible=None, limit=None, **kw):
        super().__init__(inforce, deductible, limit, **kw)

    def value_at_risk(self, alpha=0.95, backend=None):
        """
        Compute the Value at Risk (VaR) for the portfolio using quantum simulation.

        Parameters:
        ----------
        - `alpha`: Confidence level for VaR (default is 0.95).
        - `backend`: Optional backend for overriding the BackendManager settings.
        """
        pass

    def tail_value_at_risk(self, alpha=0.95, backend=None):
        """
        Compute the Tail Value at Risk (TVaR) for the portfolio using quantum simulation.

        Parameters:
        ----------
        - `alpha`: Confidence level for TVaR (default is 0.95).
        - `backend`: Optional backend for overriding the BackendManager settings.
        """
        pass
