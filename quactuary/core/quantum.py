"""
Quantum Actuarial functionality.
"""

from qiskit import QuantumCircuit
from quactuary.entities import Inforce, Portfolio


class QuantumActuarialModel():
    def __init__(self, obj, deductible=None, limit=None, **kw):
        if isinstance(obj, Inforce):
            self.portfolio = Portfolio([obj])
        elif isinstance(obj, Portfolio):
            self.portfolio = obj
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
