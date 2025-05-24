import locale
from datetime import date

import numpy as np
import pandas as pd
import pytest

import quactuary as qa
import quactuary.book as book
from quactuary.book import LOB, Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


test_policy = PolicyTerms(
    effective_date=date(2026, 1, 1),
    expiration_date=date(2027, 1, 1),
    lob=LOB.WC,
    exposure_base=book.PAYROLL,  # type: ignore[attr-defined]
    exposure_amount=100_000_000,
    retention_type="deductible",
    per_occ_retention=500_000,
    agg_retention=1_000_000,
    corridor_retention=200_000,
    coinsurance=0.2,
    per_occ_limit=3_000_000,
    agg_limit=6_000_000,
    attachment=1_500_000,
    coverage="occ",
    notes="Comprehensive str test"
)

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
    terms=test_policy,
    frequency=test_freq,
    severity=test_sev,
    name="Test Inforce"
)


def test_policy_terms_str():
    expected_str = """Effective Date: 2026-01-01
Expiration Date: 2027-01-01
LoB: Workers' Compensation
Exposure Base: ExposureBase(name='Payroll', unit='USD', description='Total payroll amount.')
Exposure Amount: 100,000,000
Retention Type: deductible
Per-Occurrence Retention: 500,000
Aggregate Retention: 1,000,000
Corridor Retention: 200,000
Coinsurance: 20.00%
Per Occurrence Limit: 3,000,000
Aggregate Limit: 6,000,000
Attachment: 1,500,000
Coverage: occ
Notes: Comprehensive str test
"""
    assert str(test_policy) == expected_str

    expected_sparse_str = """Effective Date: 2027-01-01
Expiration Date: 2028-01-01
Exposure Amount: 5,000,000
Retention Type: SIR
Per-Occurrence Retention: 40,000
Coverage: cm
Notes: Sparse str test
"""
    assert str(test_sparse_policy) == expected_sparse_str


def test_inforce_str():
    expected_str = """Bucket: Test Inforce
- Number of Policies: 5
- Frequency Model: DeterministicFrequency(value=3)
- Severity Model: ConstantSeverity(value=100.0)
- Policy Terms:
  - Effective Date: 2026-01-01
  - Expiration Date: 2027-01-01
  - LoB: Workers' Compensation
  - Exposure Base: ExposureBase(name='Payroll', unit='USD', description='Total payroll amount.')
  - Exposure Amount: 100,000,000
  - Retention Type: deductible
  - Per-Occurrence Retention: 500,000
  - Aggregate Retention: 1,000,000
  - Corridor Retention: 200,000
  - Coinsurance: 20.00%
  - Per Occurrence Limit: 3,000,000
  - Aggregate Limit: 6,000,000
  - Attachment: 1,500,000
  - Coverage: occ
  - Notes: Comprehensive str test
"""
    assert str(test_inforce) == expected_str


def test_portfolio_from_single_inforce():
    test_portfolio = Portfolio(test_inforce)
    assert len(test_portfolio) == test_inforce.n_policies


def test_portfolio_str():
    test_portfolio = test_inforce + test_inforce

    expected_str = """Bucket: Test Inforce
- Number of Policies: 5
- Frequency Model: DeterministicFrequency(value=3)
- Severity Model: ConstantSeverity(value=100.0)
- Policy Terms:
  - Effective Date: 2026-01-01
  - Expiration Date: 2027-01-01
  - LoB: Workers' Compensation
  - Exposure Base: ExposureBase(name='Payroll', unit='USD', description='Total payroll amount.')
  - Exposure Amount: 100,000,000
  - Retention Type: deductible
  - Per-Occurrence Retention: 500,000
  - Aggregate Retention: 1,000,000
  - Corridor Retention: 200,000
  - Coinsurance: 20.00%
  - Per Occurrence Limit: 3,000,000
  - Aggregate Limit: 6,000,000
  - Attachment: 1,500,000
  - Coverage: occ
  - Notes: Comprehensive str test
"""
    assert str(test_portfolio) == expected_str + expected_str


def test_inforce_simulate():
    assert len(test_inforce) == 5
    # Test single random variable (one sample for inforce experience)
    sample = test_inforce.simulate()
    assert isinstance(sample, float)
    expected_result = 100.0 * 3 * 5
    assert sample == pytest.approx(expected_result)

    samples = test_inforce.simulate(7)
    assert isinstance(samples, pd.Series)
    assert len(samples) == 7
    expected_results = pd.Series([expected_result]).repeat(7).reset_index(drop=True)
    pd.testing.assert_series_equal(samples, expected_results)


def test_portfolio_simulate():
    # Validate implementation of __add__ and __iadd__ methods
    test_portfolio = Portfolio([test_inforce])
    assert len(test_portfolio) == 5
    test_portfolio = test_portfolio + test_portfolio
    assert len(test_portfolio) == 10
    test_portfolio += test_inforce
    assert len(test_portfolio) == 15
    test_portfolio += test_portfolio
    assert len(test_portfolio) == 30
    test_portfolio = test_portfolio + test_inforce
    assert len(test_portfolio) == 35

    # Validate unimplemented operations
    with pytest.raises(NotImplementedError):
        result = test_portfolio + 5
    with pytest.raises(NotImplementedError):
        test_portfolio += 5

    test_portfolio = test_inforce + test_inforce
    assert len(test_portfolio) == 10
    assert test_portfolio.policies == 10

    sample = test_portfolio.simulate()
    assert isinstance(sample, float)
    expected_result = 100.0 * 3 * 5 * 2
    assert sample == pytest.approx(expected_result)

    samples = test_portfolio.simulate(7)
    assert isinstance(samples, pd.Series)
    assert len(samples) == 7
    expected_results = pd.Series([expected_result]).repeat(7).reset_index(drop=True)
    pd.testing.assert_series_equal(samples, expected_results)
