"""
Pricing module for Excess Loss Model

Notes:
-------
The focus is on using classical methods (Monte Carlo, FFT) by calling external libs, with an option to accelerate via quantum algorithms.
"""

from quactuary.core.quantum import QuantumActuarialModel


class ExcessLossModel(QuantumActuarialModel):
    def __init__(self, obj, deductible=None, limit=None, **kw):
        super().__init__(obj, deductible, limit, **kw)

    def compute_excess_loss(self):
        """
        Compute the excess loss for the portfolio using quantum simulation.
        """
        pass


class VaRModel(QuantumActuarialModel):
    def __init__(self, obj, deductible=None, limit=None, **kw):
        super().__init__(obj, deductible, limit, **kw)

    def value_at_risk(self):
        """
        Compute the Value at Risk (VaR) for the portfolio using quantum simulation.
        """
        pass


class TVaRModel(QuantumActuarialModel):
    def __init__(self, obj, deductible=None, limit=None, **kw):
        super().__init__(obj, deductible, limit, **kw)

    def tail_value_at_risk(self):
        """
        Compute the Tail Value at Risk (TVaR) for the portfolio using quantum simulation.
        """
        pass
