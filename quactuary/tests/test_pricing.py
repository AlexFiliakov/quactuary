from datetime import date

import pytest

from quactuary.backend import BackendManager, ClassicalBackend
from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity
from quactuary.pricing import PricingModel, PricingResult

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


def test_pricing_result():
    with pytest.raises(NotImplementedError):
        result = PricingResult()
        result.estimates


def test_pricing_model():
    pm = PricingModel(test_portfolio)
    with pytest.raises(NotImplementedError):
        pm.calculate_portfolio_statistics()
    new_manager = BackendManager(ClassicalBackend())
    with pytest.raises(NotImplementedError):
        pm.calculate_portfolio_statistics(backend=new_manager)
