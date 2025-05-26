from datetime import date

import pandas as pd
import pytest

from quactuary.book import Inforce, PolicyTerms, Portfolio
from quactuary.classical import ClassicalPricingModel
from quactuary.datatypes import PricingResult
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity

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


def test_classical_pricing_model():
    cpm = ClassicalPricingModel()

    # Test single simulation
    sample = cpm.calculate_portfolio_statistics(test_portfolio)
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 1
    assert sample.metadata["run_date"] is not None
    assert sample.metadata["tail_alpha"] == 0.05
    assert sample.estimates["mean"] == pytest.approx(100.0 * 3 * 5)
    assert sample.estimates["variance"] == pytest.approx(0.0)
    assert sample.estimates["VaR"] == pytest.approx(0.0)
    assert sample.estimates["TVaR"] == pytest.approx(0.0)

    # Test multiple simulations
    samples = cpm.calculate_portfolio_statistics(test_portfolio, n_sims=100)
    assert isinstance(samples, PricingResult)
    assert samples.metadata["n_sims"] == 100
    assert samples.metadata["run_date"] is not None
    assert samples.metadata["tail_alpha"] == 0.05
    assert samples.estimates["mean"] == pytest.approx(100.0 * 3 * 5)
    assert samples.estimates["variance"] == pytest.approx(0.0)
    assert samples.estimates["VaR"] == pytest.approx(100.0 * 3 * 5)
    assert samples.estimates["TVaR"] == pytest.approx(100.0 * 3 * 5)

    # Test mean only
    sample = cpm.calculate_portfolio_statistics(
        test_portfolio,
        mean=True,
        variance=False,
        value_at_risk=False,
        tail_value_at_risk=False
    )
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 1
    assert sample.metadata["run_date"] is not None
    assert sample.estimates["mean"] == pytest.approx(100.0 * 3 * 5)
    assert "variance" not in sample.estimates
    assert "tail_alpha" not in sample.metadata
    assert "VaR" not in sample.estimates
    assert "TVaR" not in sample.estimates

    # Test variance only
    sample = cpm.calculate_portfolio_statistics(
        test_portfolio,
        mean=False,
        variance=True,
        value_at_risk=False,
        tail_value_at_risk=False
    )
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 1
    assert sample.metadata["run_date"] is not None
    assert sample.estimates["variance"] == pytest.approx(0.0)
    assert "mean" not in sample.estimates
    assert "tail_alpha" not in sample.metadata
    assert "VaR" not in sample.estimates
    assert "TVaR" not in sample.estimates

    # Test VaR only (single)
    sample = cpm.calculate_portfolio_statistics(
        test_portfolio,
        mean=False,
        variance=False,
        value_at_risk=True,
        tail_value_at_risk=False
    )
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 1
    assert sample.metadata["run_date"] is not None
    assert sample.metadata["tail_alpha"] == 0.05
    assert sample.estimates["VaR"] == pytest.approx(0.0)
    assert "mean" not in sample.estimates
    assert "variance" not in sample.estimates
    assert "TVaR" not in sample.estimates

    # Test VaR only (multiple)
    sample = cpm.calculate_portfolio_statistics(
        test_portfolio,
        mean=False,
        variance=False,
        value_at_risk=True,
        tail_value_at_risk=False,
        n_sims=10
    )
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 10
    assert sample.metadata["run_date"] is not None
    assert sample.metadata["tail_alpha"] == 0.05
    assert sample.estimates["VaR"] == pytest.approx(100.0 * 3 * 5)
    assert "mean" not in sample.estimates
    assert "variance" not in sample.estimates
    assert "TVaR" not in sample.estimates

    # Test TVaR only
    sample = cpm.calculate_portfolio_statistics(
        test_portfolio,
        mean=False,
        variance=False,
        value_at_risk=False,
        tail_value_at_risk=True
    )
    assert isinstance(sample, PricingResult)
    assert sample.metadata["n_sims"] == 1
    assert sample.metadata["run_date"] is not None
    assert sample.metadata["tail_alpha"] == 0.05
    assert sample.estimates["TVaR"] == pytest.approx(0.0)
    assert "mean" not in sample.estimates
    assert "variance" not in sample.estimates
    assert "VaR" not in sample.estimates
