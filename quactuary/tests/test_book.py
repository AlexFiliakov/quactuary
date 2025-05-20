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


def test_inforce_rvs():
    assert len(test_inforce) == 5
    # Test single random variable (one sample for inforce experience)
    sample = test_inforce.rvs()
    expected_series = pd.Series([100.0, 100.0, 100.0])
    for s in sample:
        pd.testing.assert_series_equal(s, expected_series)

    samples = test_inforce.rvs(7)
    assert len(samples) == 7
    for sample_instance in samples:
        assert len(sample_instance) == 5
        for s in sample_instance:
            pd.testing.assert_series_equal(s, expected_series)


def test_portfolio_rvs():
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
        test_portfolio + 5
    with pytest.raises(NotImplementedError):
        test_portfolio += 5

    test_portfolio = test_inforce + test_inforce
    assert len(test_portfolio) == 10
    assert test_portfolio.total_policies() == 10

    sample = test_portfolio.rvs()
    expected_series = pd.Series([100.0, 100.0, 100.0])
    assert len(sample) == 2
    for bucket in sample:
        assert len(bucket) == 5
        for pol in bucket:
            pd.testing.assert_series_equal(pol, expected_series)

    samples = test_portfolio.rvs(7)
    assert len(samples) == 7
    for buckets in samples:
        assert len(buckets) == 2
        for bucket in buckets:
            assert len(bucket) == 5
            for pol in bucket:
                pd.testing.assert_series_equal(pol, expected_series)
