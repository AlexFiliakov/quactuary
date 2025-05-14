"""
Quantum Actuarial functionality.
"""

from qiskit import QuantumCircuit
from quactuary.entities import Inforce, Portfolio


class QuantumActuarialModel():
    def __init__(self, inforce, deductible=None, limit=None, **kw):
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

    def build_circuit(self):
        qc = QuantumCircuit()
        # call each bucket’s state‑prep routine and ripple‑add payouts
        for bucket in self.portfolio:
            qc.compose(bucket._quantum_state(), inplace=True)
        # apply layer deductible/limit truncation oracle if needed …
        return qc
