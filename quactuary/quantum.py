"""
Quantum Actuarial functionality.

Purpose:
Centralize the quantum computing integration so that each model can invoke quantum algorithms consistently.
"""


from dataclasses import dataclass

from qiskit import QuantumCircuit
from quactuary.entities import Inforce, Portfolio

from quactuary import get_backend


@dataclass
class QuantumResult():
    """
    Hold results like confidence intervals of quantum estimates, which models can then convert to user-facing outputs (e.g. a number or DataFrame).
    """
    pass


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

        # Use the global backend manager
        self.backend = get_backend()

    def build_circuit(self):
        qc = QuantumCircuit()
        # call each bucket's state‑prep routine and ripple‑add payouts
        for bucket in self.portfolio:
            qc.compose(bucket._quantum_state(), inplace=True)
        # apply layer deductible/limit truncation oracle if needed …
        return qc

    def run(self):
        """
        Build and run the quantum circuit using the configured backend.

        Returns:
        -------
        - Result from running the circuit
        """
        circuit = self.build_circuit()
        return self.backend.run(circuit)


class QuantumModelMixin ():
    """
    Mixin class for quantum models.

    This mixin is designed to be used with other classes to provide quantum functionality.
    """
    pass


def __init__(self):
    """
    Initialize the Quantum Model Mixin.
    """
    pass


def _run_amplitude_estimation(circuit, confidence):
    """
    Run amplitude estimation on the given quantum circuit.

    Parameters:
    ----------
    - `circuit`: The quantum circuit to run.
    - `confidence`: The confidence level for the estimation.

    Returns:
    -------
    - Estimated value from the amplitude estimation.
    """
    # Placeholder for actual amplitude estimation logic
    pass
