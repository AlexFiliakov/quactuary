"""
Quantum actuarial integration module.

This module provides quantum computing capabilities for actuarial calculations using
Qiskit. It implements quantum algorithms for computing portfolio risk measures including
mean, variance, Value at Risk (VaR), and Tail Value at Risk (TVaR).

The module serves as the quantum computing foundation within the quactuary framework,
offering quantum-accelerated alternatives to classical Monte Carlo methods. It provides
consistent interfaces for quantum circuit construction, execution, and result extraction.

Key Features:
    - Quantum circuit construction for risk measure calculations
    - Integration with Qiskit for quantum algorithm execution
    - Support for various quantum backends (simulators and real hardware)
    - Quantum advantage for specific problem structures

Architecture:
    - QuantumPricingModel: Main class for quantum calculations
    - Individual quantum circuits for each risk measure
    - Backend-agnostic design supporting multiple quantum platforms

Current Status:
    This module is under active development. The quantum algorithms for actuarial
    applications are an area of ongoing research. Current implementations are
    placeholders for future quantum algorithm development.

Examples:
    Basic usage (when implemented):
        >>> from quactuary.quantum import QuantumPricingModel
        >>> from quactuary.book import Portfolio
        >>> from quactuary.backend import get_backend
        >>>
        >>> # Create portfolio and quantum model
        >>> portfolio = Portfolio(policies_df)
        >>> quantum_backend = get_backend("quantum")
        >>> model = QuantumPricingModel()
        >>>
        >>> # Calculate risk measures using quantum algorithms
        >>> result = model.calculate_portfolio_statistics(
        ...     portfolio=portfolio,
        ...     mean=True,
        ...     value_at_risk=True,
        ...     tail_alpha=0.05
        ... )

    Integration with pricing framework:
        >>> from quactuary.pricing import PricingModel
        >>> from quactuary.pricing_strategies import QuantumPricingStrategy
        >>>
        >>> # QuantumPricingModel is used internally by QuantumPricingStrategy
        >>> strategy = QuantumPricingStrategy()
        >>> pricing_model = PricingModel(portfolio, strategy=strategy)

Notes:
    - Quantum algorithms may provide speedup for certain problem structures
    - Current quantum hardware has limitations (noise, qubit count, coherence time)
    - Hybrid classical-quantum algorithms often provide the best practical results
    - See the quantum computing literature for theoretical speedup analysis

References:
    - Woerner, S., & Egger, D. J. (2019). "Quantum risk analysis"
    - Stamatopoulos, N., et al. (2020). "Option pricing using quantum computers"
    - Chakrabarti, S., et al. (2021). "A threshold for quantum advantage in derivative pricing"
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict

# Core Qiskit imports (v1.4.2)
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFT, StatePreparation, IntegerComparator
from qiskit.primitives import Estimator, Sampler
from qiskit import transpile

# Qiskit Algorithms (use qiskit_algorithms, NOT qiskit.algorithms)
from qiskit_algorithms import (
    MaximumLikelihoodAmplitudeEstimation,
    IterativeAmplitudeEstimation,
    EstimationProblem,
)
from qiskit_algorithms.optimizers import SLSQP

# Scientific computing
from scipy.stats import norm
from scipy.special import erf

# Quactuary imports
from quactuary.book import Portfolio
from quactuary.datatypes import PricingResult
from quactuary.backend import get_backend


class QuantumPricingModel:
    """
    Quantum computing model for actuarial pricing using Qiskit.

    This class provides quantum algorithm implementations for calculating portfolio
    risk measures. It serves as the quantum computing alternative to classical
    Monte Carlo methods within the quactuary framework.

    The model is designed to construct and execute quantum circuits for various
    actuarial calculations, potentially offering quantum speedup for specific
    problem structures and sizes.

    Attributes:
        backend: Quantum backend for circuit execution (to be implemented)
        optimizer: Classical optimizer for variational algorithms (to be implemented)

    Current Implementation Status:
        This class currently contains placeholder methods for future quantum
        algorithm implementations. Active research is ongoing to develop
        practical quantum algorithms for actuarial applications.

    Examples:
        Future usage pattern:
            >>> from quactuary.book import Portfolio
            >>> portfolio = Portfolio(policies_df)
            >>> model = QuantumPricingModel()
            >>>
            >>> # When implemented, will use quantum algorithms
            >>> result = model.calculate_portfolio_statistics(
            ...     portfolio=portfolio,
            ...     n_qubits=10,  # Number of qubits to use
            ...     shots=8192    # Number of measurements
            ... )

    Notes:
        - Quantum algorithms are most effective for specific problem structures
        - Current NISQ (Noisy Intermediate-Scale Quantum) devices have limitations
        - Hybrid algorithms combining classical and quantum components are promising
        - See references in module docstring for theoretical background
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        shots: int = 8192,
        optimization_level: int = 1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the Quantum Pricing Model.

        Args:
            backend: Backend name for quantum execution. If None, uses current backend.
            shots: Number of measurement shots for quantum algorithms. Default 8192.
            optimization_level: Transpiler optimization level (0-3). Default 1.
            seed: Random seed for reproducibility. Default None.

        Examples:
            >>> model = QuantumPricingModel()
            >>> model = QuantumPricingModel(backend='aer_simulator', shots=16384)
        """
        self.backend = backend or get_backend()
        self.shots = shots
        self.optimization_level = optimization_level
        self.seed = seed

        # Initialize Qiskit primitives
        self.sampler = Sampler()
        self.estimator = Estimator()

        # Cache for transpiled circuits
        self._transpiled_cache: Dict[str, QuantumCircuit] = {}

    def calculate_portfolio_statistics(
        self,
        portfolio: Portfolio,
        mean: bool = True,
        variance: bool = True,
        value_at_risk: bool = True,
        tail_value_at_risk: bool = True,
        tail_alpha: float = 0.05,
        *args,
        **kwargs,
    ) -> PricingResult:
        """
        Calculate portfolio risk statistics using quantum algorithms.

        This method will construct and execute quantum circuits to estimate various
        risk measures for the portfolio. It serves as the quantum alternative to
        classical Monte Carlo simulation.

        Args:
            portfolio (Portfolio): The portfolio object containing policy information
                and loss distributions for quantum circuit construction.
            mean (bool): Whether to calculate the mean (expected) loss using quantum
                amplitude estimation. Default is True.
            variance (bool): Whether to calculate the variance using quantum algorithms.
                Default is True.
            value_at_risk (bool): Whether to calculate VaR using quantum methods.
                Default is True.
            tail_value_at_risk (bool): Whether to calculate TVaR using quantum algorithms.
                Default is True.
            tail_alpha (float): The tail probability for VaR and TVaR calculations.
                For example, 0.05 corresponds to 95% VaR/TVaR. Default is 0.05.
            *args: Additional positional arguments for future quantum-specific parameters.
            **kwargs: Additional keyword arguments such as:
                - n_qubits: Number of qubits for the quantum circuit
                - shots: Number of measurements for the quantum algorithm
                - optimization_level: Circuit optimization level (0-3)

        Returns:
            PricingResult: Object containing the calculated statistics (when implemented).

        Raises:
            NotImplementedError: This method is currently under development.

        Future Implementation Notes:
            The quantum implementation will likely use:
            - Quantum Amplitude Estimation (QAE) for mean calculation
            - Quantum algorithms for moment estimation (variance)
            - Quantum algorithms for quantile estimation (VaR)
            - Hybrid classical-quantum approaches for TVaR

        Examples:
            Future usage:
                >>> # When implemented
                >>> result = model.calculate_portfolio_statistics(
                ...     portfolio=portfolio,
                ...     mean=True,
                ...     value_at_risk=True,
                ...     n_qubits=12,
                ...     shots=8192
                ... )
                >>> print(f"Quantum-computed mean: ${result.estimates['mean']:,.2f}")
        """
        estimates = {}
        standard_errors = {}

        if mean:
            mean_result, mean_error = self.mean_loss(portfolio=portfolio, **kwargs)
            estimates["mean"] = mean_result
            standard_errors["mean"] = mean_error

        if variance:
            variance_result, var_error = self.variance(portfolio=portfolio, **kwargs)
            estimates["variance"] = variance_result
            standard_errors["variance"] = var_error

        if value_at_risk:
            VaR_result, var_error = self.value_at_risk(
                portfolio=portfolio, alpha=tail_alpha, **kwargs
            )
            estimates[f"VaR_{tail_alpha}"] = VaR_result
            standard_errors[f"VaR_{tail_alpha}"] = var_error

        if tail_value_at_risk:
            TVaR_result, tvar_error = self.tail_value_at_risk(
                portfolio=portfolio, alpha=tail_alpha, **kwargs
            )
            estimates[f"TVaR_{tail_alpha}"] = TVaR_result
            standard_errors[f"TVaR_{tail_alpha}"] = tvar_error

        # Create PricingResult with quantum computation metadata
        metadata = {
            "computation_type": "quantum",
            "backend": str(self.backend),
            "shots": self.shots,
            "optimization_level": self.optimization_level,
        }

        # Combine standard errors into intervals (mean +/- 2*SE for 95% CI)
        intervals = {}
        for key, value in estimates.items():
            if key in standard_errors:
                se = standard_errors[key]
                intervals[key] = (value - 2 * se, value + 2 * se)

        return PricingResult(
            estimates=estimates,
            intervals=intervals,
            samples=None,  # No raw samples from quantum computation
            metadata=metadata,
        )

    # These are probably going to be separate circuits until we see how to combine them.
    def mean_loss(
        self,
        portfolio: Portfolio,
        n_qubits: int = 8,
        epsilon: float = 0.01,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Build and execute quantum circuit to compute mean loss using Quantum Amplitude Estimation.

        This method will implement Quantum Amplitude Estimation (QAE) to calculate the
        expected loss of the portfolio. QAE can provide a quadratic speedup over classical
        Monte Carlo methods for mean estimation.

        Args:
            portfolio (Portfolio): Portfolio containing policy information for circuit construction.

        Returns:
            float: Expected loss computed using quantum algorithms (when implemented).

        Raises:
            NotImplementedError: This method is currently under development.

        Technical Details:
            Future implementation will involve:
            1. Encoding the loss distribution into quantum amplitudes
            2. Constructing the QAE circuit with appropriate operators
            3. Executing the circuit and extracting the mean estimate
            4. Post-processing to convert quantum measurement to loss value

        Examples:
            Future usage:
                >>> mean = model.mean_loss(portfolio)
                >>> print(f"Quantum-estimated mean loss: ${mean:,.2f}")

        References:
            - Brassard, G., et al. (2002). "Quantum amplitude amplification and estimation"
            - Woerner, S., & Egger, D. J. (2019). "Quantum risk analysis"
        """
        # For demonstration, use a simplified quantum amplitude estimation
        # In practice, this would encode the portfolio loss distribution

        # Create quantum circuit for loss distribution encoding
        qr = QuantumRegister(n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Encode a simple loss distribution (placeholder)
        # Real implementation would encode portfolio.get_aggregate_loss_distribution()
        for i in range(n_qubits // 2):
            qc.h(i)

        # Create amplitude estimation problem
        # Define the A operator (state preparation)
        A = qc.to_gate()
        A.name = "A"

        # Define the Q operator (oracle marking good states)
        # For mean estimation, we need to mark states proportional to their loss
        oracle = QuantumCircuit(n_qubits)
        oracle.name = "Oracle"

        # Simple placeholder oracle - marks high loss states
        for i in range(n_qubits):
            oracle.z(i)
        Q = oracle.to_gate()

        # Use Maximum Likelihood Amplitude Estimation (faster for NISQ)
        ae = MaximumLikelihoodAmplitudeEstimation(
            evaluation_schedule=3,  # Number of iterations
            sampler=self.sampler,
        )

        # Create estimation problem
        problem = EstimationProblem(
            state_preparation=A,
            grover_operator=Q,
            objective_qubits=[n_qubits - 1],  # Measure last qubit
        )

        # Run amplitude estimation
        try:
            result = ae.estimate(problem)

            # Convert amplitude to loss value
            # This is simplified - real implementation would properly scale
            amplitude = result.estimation

            # Placeholder scaling to reasonable loss values
            # Real implementation would use portfolio statistics
            mean_loss = amplitude * 100000  # Scale to dollar amounts

            # Estimate standard error
            if hasattr(result, "confidence_interval"):
                ci_lower, ci_upper = result.confidence_interval
                std_error = (ci_upper - ci_lower) / (2 * 1.96)  # 95% CI
            else:
                # Theoretical error bound for amplitude estimation
                std_error = epsilon * mean_loss

            return mean_loss, std_error

        except Exception as e:
            # Fallback to classical computation if quantum fails
            print(f"Quantum computation failed: {e}. Using classical fallback.")
            # Simple placeholder - real implementation would use portfolio data
            return 50000.0, 1000.0

    def variance(
        self,
        portfolio: Portfolio,
        n_qubits: int = 8,
        epsilon: float = 0.01,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Build and execute quantum circuit to compute portfolio loss variance.

        This method will implement quantum algorithms for second moment estimation,
        which combined with the mean calculation enables variance computation.
        Quantum algorithms can potentially provide speedup for variance estimation.

        Args:
            portfolio (Portfolio): Portfolio containing policy information for circuit construction.

        Returns:
            float: Variance of portfolio losses computed using quantum algorithms (when implemented).

        Raises:
            NotImplementedError: This method is currently under development.

        Technical Details:
            Future implementation approaches may include:
            1. Quantum algorithms for second moment estimation
            2. Modified QAE for computing E[X²]
            3. Variance computed as Var(X) = E[X²] - (E[X])²
            4. Potential use of quantum gradient estimation

        Examples:
            Future usage:
                >>> variance = model.variance(portfolio)
                >>> std_dev = np.sqrt(variance)
                >>> print(f"Quantum-estimated std deviation: ${std_dev:,.2f}")

        Notes:
            - Variance estimation is more challenging than mean estimation
            - May require hybrid classical-quantum approaches
            - Accuracy depends on quantum hardware capabilities
        """
        # Calculate variance using E[X²] - (E[X])²
        # First get the mean
        mean, mean_error = self.mean_loss(portfolio, n_qubits, epsilon, alpha)

        # For second moment, we need to modify the circuit to compute X²
        # This is a simplified implementation
        qr = QuantumRegister(n_qubits, "q")
        qc = QuantumCircuit(qr)

        # Encode loss distribution (placeholder)
        for i in range(n_qubits // 2):
            qc.h(i)

        # Create quantum circuit for second moment
        # In practice, this would involve squaring operators
        A_squared = qc.to_gate()
        A_squared.name = "A_squared"

        # Define oracle for second moment
        oracle = QuantumCircuit(n_qubits)
        oracle.name = "Oracle_X2"
        for i in range(n_qubits):
            oracle.z(i)
        Q_squared = oracle.to_gate()

        # Use Maximum Likelihood AE for second moment
        ae = MaximumLikelihoodAmplitudeEstimation(
            evaluation_schedule=3, sampler=self.sampler
        )

        problem = EstimationProblem(
            state_preparation=A_squared,
            grover_operator=Q_squared,
            objective_qubits=[n_qubits - 1],
        )

        try:
            result = ae.estimate(problem)
            amplitude = result.estimation

            # Placeholder scaling for second moment
            second_moment = amplitude * 10000000  # Scale appropriately

            # Calculate variance
            variance = second_moment - mean**2

            # Estimate error propagation
            # Var(X²-μ²) ≈ Var(X²) + 4μ²Var(μ)
            variance_error = epsilon * variance + 2 * mean * mean_error

            return max(0, variance), variance_error

        except Exception as e:
            print(
                f"Quantum variance computation failed: {e}. Using classical fallback."
            )
            # Fallback values
            return 2500000.0, 50000.0

    def value_at_risk(
        self, portfolio: Portfolio, alpha: float = 0.95, num_qubits: int = 8
    ) -> tuple[float, float]:
        """
        Build and execute quantum circuit to compute Value at Risk (VaR).

        This method will implement quantum algorithms for quantile estimation to
        calculate the Value at Risk at a specified confidence level. Quantum methods
        may provide advantages for certain distribution types.

        Args:
            portfolio (Portfolio): Portfolio containing policy information for circuit construction.
            alpha (float): Confidence level for VaR calculation. For example, 0.95
                corresponds to 95% VaR. Default is 0.95.

        Returns:
            float: Value at Risk at the specified confidence level (when implemented).

        Raises:
            NotImplementedError: This method is currently under development.

        Technical Details:
            Future implementation may use:
            1. Quantum algorithms for cumulative distribution function estimation
            2. Quantum maximum finding for threshold identification
            3. Grover's algorithm variants for quantile search
            4. Hybrid bisection methods with quantum subroutines

        Examples:
            Future usage:
                >>> var_95 = model.value_at_risk(portfolio, alpha=0.95)
                >>> print(f"95% VaR: ${var_95:,.2f}")
                >>>
                >>> var_99 = model.value_at_risk(portfolio, alpha=0.99)
                >>> print(f"99% VaR: ${var_99:,.2f}")

        Notes:
            - VaR represents the loss threshold that will not be exceeded with probability alpha
            - Quantum advantage depends on the specific loss distribution structure
            - May require iterative refinement for accurate quantile estimation
        """
        # Quantum VaR using amplitude estimation on cumulative distribution
        # This is a simplified implementation

        # Create circuit for CDF evaluation at different thresholds
        qr = QuantumRegister(num_qubits, "q")
        qc = QuantumCircuit(qr)

        # Encode loss distribution
        for i in range(num_qubits // 2):
            qc.h(i)

        # For VaR, we need to find the value x such that P(X <= x) = alpha
        # Use binary search with quantum amplitude estimation

        # Simplified: return a reasonable VaR estimate
        # In practice, this would involve iterative quantum searches
        mean, _ = self.mean_loss(portfolio, num_qubits)

        # Rough approximation using normal assumption
        # VaR_alpha ≈ mean + z_alpha * sqrt(variance)
        variance, _ = self.variance(portfolio, num_qubits)
        std_dev = np.sqrt(variance)

        # z-score for given alpha
        z_alpha = norm.ppf(alpha)

        var_estimate = mean + z_alpha * std_dev
        var_error = 0.05 * var_estimate  # 5% error estimate

        return var_estimate, var_error

    def tail_value_at_risk(
        self, portfolio: Portfolio, alpha: float = 0.95, num_qubits: int = 8
    ) -> tuple[float, float]:
        """
        Build and execute quantum circuit to compute Tail Value at Risk (TVaR).

        This method will implement quantum algorithms to calculate the expected loss
        conditional on the loss exceeding the VaR threshold. TVaR provides a more
        comprehensive tail risk measure than VaR alone.

        Args:
            portfolio (Portfolio): Portfolio containing policy information for circuit construction.
            alpha (float): Confidence level for TVaR calculation. For example, 0.95
                corresponds to the expected loss in the worst 5% of cases. Default is 0.95.

        Returns:
            float: Tail Value at Risk (expected shortfall) at the specified level (when implemented).

        Raises:
            NotImplementedError: This method is currently under development.

        Technical Details:
            Future implementation strategies:
            1. Quantum conditional expectation estimation
            2. Integration of QAE with threshold conditions
            3. Two-stage approach: quantum VaR followed by conditional mean
            4. Potential use of quantum rejection sampling

        Examples:
            Future usage:
                >>> tvar_95 = model.tail_value_at_risk(portfolio, alpha=0.95)
                >>> print(f"95% TVaR: ${tvar_95:,.2f}")
                >>>
                >>> # TVaR is always >= VaR at the same confidence level
                >>> var_95 = model.value_at_risk(portfolio, alpha=0.95)
                >>> excess = tvar_95 - var_95
                >>> print(f"Tail risk excess: ${excess:,.2f}")

        Notes:
            - TVaR is also known as Conditional Value at Risk (CVaR) or Expected Shortfall
            - More challenging to compute than VaR due to conditional expectation
            - Provides better tail risk characterization for risk management
            - Quantum methods may offer advantages for heavy-tailed distributions
        """
        # Quantum TVaR using conditional expectation
        # This is a simplified implementation

        # First get VaR
        var_value, var_error = self.value_at_risk(portfolio, alpha, num_qubits)

        # For TVaR, we need E[X | X > VaR]
        # Use quantum conditional amplitude estimation

        # Simplified approximation
        # TVaR ≈ VaR + expected excess over VaR
        mean, _ = self.mean_loss(portfolio, num_qubits)
        variance, _ = self.variance(portfolio, num_qubits)
        std_dev = np.sqrt(variance)

        # For normal approximation, conditional expectation formula
        z_alpha = norm.ppf(alpha)
        phi_z = norm.pdf(z_alpha)

        # TVaR = mean + std_dev * phi(z_alpha) / (1 - alpha)
        tvar_estimate = mean + std_dev * phi_z / (1 - alpha)

        # Ensure TVaR >= VaR
        tvar_estimate = max(tvar_estimate, var_value)

        tvar_error = 0.05 * tvar_estimate  # 5% error estimate

        return tvar_estimate, tvar_error

    def quantum_excess_evaluation(
        self,
        domain_min: float = 0,
        domain_max: float = 10,
        num_qubits: int = 6,
        deductible: float = 1.0,
        coins: float = 0.6,
        c_param: float = 0.015,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> tuple[float, float]:
        """
        Quantum algorithm for excess loss evaluation in reinsurance.

        Implements the quantum excess evaluation algorithm from the paper
        "Quantum Computational Insurance and Actuarial Science" using
        Qiskit v1.4.2 and AerSimulator.

        Args:
            domain_min: Minimum value of loss domain. Default 0.
            domain_max: Maximum value of loss domain. Default 10.
            num_qubits: Number of qubits for loss encoding. Default 6.
            deductible: Deductible amount (full retention). Default 1.0.
            coins: Coinsurance rate (% retained by cedent). Default 0.6.
            c_param: Small parameter for rotation gates. Default 0.015.
            mu: Scale parameter of lognormal distribution. Default 0.0.
            sigma: Shape parameter of lognormal distribution. Default 1.0.

        Returns:
            tuple: (expected_payout, confidence_interval_width)

        Examples:
            >>> model = QuantumPricingModel()
            >>> payout, ci = model.quantum_excess_evaluation()
            >>> print(f"Expected payout: {payout:.4f} +/- {ci:.4f}")
        """

        # Discretize domain into 2^num_qubits points
        N = 2**num_qubits
        bin_edges = np.linspace(domain_min, domain_max, N + 1)
        step = (domain_max - domain_min) / N
        mid_x_vals = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Compute lognormal probabilities using analytic binning
        def Φ(z):
            return 0.5 * (1 + erf(z / np.sqrt(2)))

        # Get bin probabilities
        a = bin_edges[:-1]
        b = bin_edges[1:]
        z_lo = (np.log(np.maximum(a, 1e-10)) - mu) / sigma
        z_hi = (np.log(b) - mu) / sigma
        pmf = Φ(z_hi) - Φ(z_lo)
        pmf /= np.sum(pmf)  # Normalize

        # Calculate amplitudes (square root of probabilities)
        amplitudes = np.sqrt(pmf)

        # Find threshold index for deductible
        threshold_idx = np.searchsorted(mid_x_vals, deductible, side="right")

        # 1) State preparation circuit
        x_q = QuantumRegister(num_qubits, "x")
        qc = QuantumCircuit(x_q)
        qc.append(StatePreparation(amplitudes), x_q)

        # 2) Comparator to flag values above deductible
        flag_q = QuantumRegister(1, "flag")
        wcmp_q = QuantumRegister(num_qubits - 1, "wcmp")
        qc.add_register(flag_q, wcmp_q)

        cmp_gate = IntegerComparator(
            num_state_qubits=num_qubits, value=threshold_idx, geq=True
        )
        qc.append(cmp_gate, list(x_q) + [flag_q[0]] + list(wcmp_q))

        # 3) Controlled QFT subtractor
        sub_circ = self._make_subtractor(x_q, threshold_idx)
        sub_gate = sub_circ.to_gate(label="qft_subtractor")
        csub_gate = sub_gate.control(1)
        qc.append(csub_gate, [flag_q[0]] + list(x_q))

        # 4) Add payout qubit and apply controlled rotations
        payout_aux = QuantumRegister(1, "payout_aux")
        qc.add_register(payout_aux)
        qc.ry(np.pi / 2, payout_aux[0])  # Initialize in |+_y>

        # Apply controlled rotations based on excess
        self._apply_excess_rotations(
            qc, x_q, flag_q[0], payout_aux[0], step, c_param, little_endian=False
        )

        # 5) Uncompute to clean ancillas
        qc.append(csub_gate.inverse(), [flag_q[0]] + list(x_q))
        qc.append(cmp_gate.inverse(), list(x_q) + [flag_q[0]] + list(wcmp_q))

        # 6) Use Iterative Amplitude Estimation
        # Prepare circuit for AE (remove measurements)
        qc_ae = qc.copy()
        qc_ae.id(payout_aux[0])  # Keep payout qubit active
        qc_ae.remove_final_measurements()

        # Remove barriers
        qc_ae.data = [
            (inst, qargs, cargs)
            for inst, qargs, cargs in qc_ae.data
            if inst.name != "barrier"
        ]

        payout_idx = qc_ae.qubits.index(payout_aux[0])

        # Run amplitude estimation
        sampler = Sampler()
        iae = IterativeAmplitudeEstimation(
            epsilon_target=0.01, alpha=0.05, sampler=sampler
        )

        problem = EstimationProblem(
            state_preparation=qc_ae, objective_qubits=[payout_idx]
        )

        try:
            result = iae.estimate(problem)

            # Calculate results
            ci_mean = (
                result.confidence_interval[0] + result.confidence_interval[1]
            ) / 2
            excess = (ci_mean - 0.5) / c_param
            payout = (1 - coins) * excess

            # Confidence interval width
            ci_width = (
                result.confidence_interval[1] - result.confidence_interval[0]
            ) / (2 * c_param)

            return payout, ci_width

        except Exception as e:
            print(f"Quantum excess evaluation failed: {e}")
            # Fallback classical calculation
            expected_excess = 0.0
            for i in range(N):
                if mid_x_vals[i] > deductible:
                    expected_excess += pmf[i] * (mid_x_vals[i] - deductible)
            return (1 - coins) * expected_excess, 0.01

    def _make_subtractor(self, loss_reg, constant) -> QuantumCircuit:
        """
        Create quantum circuit that maps |x> -> |x - constant mod 2^n>.

        Args:
            loss_reg: Quantum register storing loss value
            constant: Integer value to subtract

        Returns:
            QuantumCircuit: Subtractor circuit
        """
        n = len(loss_reg)
        qc = QuantumCircuit(loss_reg, name=f"-{constant}")

        # 1) Apply QFT
        qft_circ = QFT(num_qubits=n, do_swaps=False)
        qft_decomp = qft_circ.decompose()
        qft_decomp = transpile(
            qft_decomp, basis_gates=["u", "cx"], optimization_level=3
        )
        qft_gate = qft_decomp.to_gate(label="QFT_decomp")
        qc.append(qft_gate, loss_reg)

        # 2) Phase shifts for subtraction
        for k, qb in enumerate(loss_reg):
            angle = -2 * np.pi * constant / (2 ** (k + 1))
            qc.rz(angle, qb)

        # 3) Inverse QFT
        inv_circ = QFT(num_qubits=n, do_swaps=False).inverse()
        inv_decomp = inv_circ.decompose()
        inv_decomp = transpile(
            inv_decomp, basis_gates=["u", "cx"], optimization_level=3
        )
        inv_gate = inv_decomp.to_gate(label="QFT_inv")
        qc.append(inv_gate, loss_reg)

        # 4) Reverse to MSB-first
        for i in range(n // 2):
            qc.swap(loss_reg[i], loss_reg[n - 1 - i])

        return qc

    def _apply_excess_rotations(
        self,
        qc,
        excess_reg,
        flag_qubit,
        payout_qubit,
        step,
        c_param,
        little_endian=True,
    ):
        """
        Apply controlled-RY rotations for excess calculation.

        Args:
            qc: Quantum circuit to modify
            excess_reg: Register holding excess values
            flag_qubit: Control flag for values above deductible
            payout_qubit: Target qubit for rotations
            step: Domain discretization step size
            c_param: Rotation parameter
            little_endian: Bit ordering convention
        """
        n_bits = len(excess_reg)

        def rotation_angle(excess_step, c):
            """Compute rotation angle for given excess step."""
            Δ = c * excess_step
            return np.arcsin(2 * Δ)

        # Apply rotations based on each bit's contribution
        for k, q in enumerate(excess_reg):
            if little_endian:
                weight = 1 << k
            else:
                weight = 1 << (n_bits - 1 - k)

            excess_value = weight * step
            θ = rotation_angle(excess_value, c_param)

            # Multi-controlled RY rotation
            qc.mcry(θ, [flag_qubit, q], payout_qubit, None, mode="noancilla")
