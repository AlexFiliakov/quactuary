from qiskit import QuantumCircuit
from quactuary.entities import Inforce, Portfolio
from quactuary.pricing.quantum_actuarial_model import QuantumActuarialModel


class ExcessLossModel(QuantumActuarialModel):
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
        super().__init__(**kw)

    def build_circuit(self):
        qc = QuantumCircuit()
        # call each bucket’s state‑prep routine and ripple‑add payouts
        for bucket in self.portfolio:
            qc.compose(bucket._quantum_state(), inplace=True)
        # apply layer deductible/limit truncation oracle if needed …
        return qc

    def compute_excess_loss(self, confidence=0.95, classical_samples=None):
        """
        Compute the excess loss for the portfolio using quantum simulation.
        This method uses the quantum circuit built in `build_circuit` to
        simulate the excess loss.

        To debug, pass `classical_samples` to run a classical simulation
        instead of a quantum simulation. This is useful for testing and
        validating the quantum simulation results.

        Parameters
        ----------
        confidence : float
            The confidence level for the excess loss calculation.
            Default is 0.95 (95% confidence).
        classical_samples : int
            The number of classical samples to use for simulation.
            Default is None to run quantum backend.
        """
        pass
