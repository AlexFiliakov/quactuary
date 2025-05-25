import sys
from datetime import date
from unittest.mock import MagicMock

import pytest

# Mock qiskit to avoid import errors
mock_qiskit = MagicMock()
mock_qiskit.__version__ = "1.4.2"
sys.modules['qiskit'] = mock_qiskit
sys.modules['qiskit.providers'] = MagicMock()

from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity
from quactuary.quantum import QuantumPricingModel

test_sparse_policy = PolicyTerms(
    effective_date=date(2027, 1, 1),
    expiration_date=date(2028, 1, 1),
    exposure_amount=5_000_000,
    retention_type="SIR",
    per_occ_retention=40_000,
    coverage="cm",
    notes="Sparse str test"
)

test_freq = DeterministicFrequency(3)  # type: ignore[attr-defined]
test_sev = ConstantSeverity(100)

test_inforce = Inforce(
    n_policies=5,
    terms=test_sparse_policy,
    frequency=test_freq,
    severity=test_sev,
    name="Test Inforce"
)

test_portfolio = Portfolio([test_inforce])


def test_quantum_pricing_model_mixin():
    qpmm = QuantumPricingModel()
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(test_portfolio)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            portfolio=test_portfolio,
            mean=False,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=False,
            tail_alpha=0.95)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            portfolio=test_portfolio,
            mean=True,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=False)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            portfolio=test_portfolio,
            mean=False,
            variance=True,
            value_at_risk=False,
            tail_value_at_risk=False)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            portfolio=test_portfolio,
            mean=False,
            variance=False,
            value_at_risk=True,
            tail_value_at_risk=False,
            tail_alpha=0.80)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(
            portfolio=test_portfolio,
            mean=False,
            variance=False,
            value_at_risk=False,
            tail_value_at_risk=True,
            tail_alpha=0.80)
    with pytest.raises(NotImplementedError):
        qpmm.calculate_portfolio_statistics(portfolio=test_portfolio)
    with pytest.raises(NotImplementedError):
        qpmm.mean_loss(test_portfolio)
    with pytest.raises(NotImplementedError):
        qpmm.variance(test_portfolio)
    with pytest.raises(NotImplementedError):
        qpmm.value_at_risk(test_portfolio, 0.95)
    with pytest.raises(NotImplementedError):
        qpmm.tail_value_at_risk(test_portfolio, 0.95)
