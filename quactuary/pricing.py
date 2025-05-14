"""
Pricing module for Excess Loss Model

Notes:
-------
The focus is on using classical methods (Monte Carlo, FFT) by calling external libs, with an option to accelerate via quantum algorithms.
"""

from quactuary.quantum import QuantumActuarialModel


class ExcessLossModel(QuantumActuarialModel):
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

    def compute_excess_loss(self):
        """
        Compute the excess loss for the portfolio using quantum simulation.
        """
        pass


class RiskMeasureModel(QuantumActuarialModel):
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

    def value_at_risk(self, alpha=0.95):
        """
        Compute the Value at Risk (VaR) for the portfolio using quantum simulation.
        """
        pass

    def tail_value_at_risk(self, alpha=0.95):
        """
        Compute the Tail Value at Risk (TVaR) for the portfolio using quantum simulation.
        """
        pass
