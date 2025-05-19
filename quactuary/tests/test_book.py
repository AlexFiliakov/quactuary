import locale

import numpy as np
import pandas as pd
import pytest

import quactuary as qa
from quactuary.book import LOB, Inforce, PolicyTerms, Portfolio
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')


def test_policy_terms_str():
    test_policy = PolicyTerms(
        effective_date='2026-01-01',  # type: ignore[attr-defined]
        expiration_date='2027-01-01',  # type: ignore[attr-defined]
        lob=LOB.WC,
        exposure_base=qa.book.PAYROLL,  # type: ignore[attr-defined]
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


def test_inforce_str():
    test_policy = PolicyTerms(
        effective_date='2026-01-01',  # type: ignore[attr-defined]
        expiration_date='2027-01-01',  # type: ignore[attr-defined]
        lob=LOB.WC,
        exposure_base=qa.book.PAYROLL,  # type: ignore[attr-defined]
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

    test_freq = DeterministicFrequency(3)  # type: ignore[attr-defined]
    test_sev = ConstantSeverity(100)

    test_inforce = Inforce(
        n_policies=5,
        terms=test_policy,
        frequency=test_freq,
        severity=test_sev,
        name="Test Inforce"
    )

    expected_str = """Bucket: Test Inforce
- Number of Policies: 5
- Frequency Model: DeterministicFrequency(value=3)
- Severity Model: ConstantSeverity(value=100)
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
    test_policy = PolicyTerms(
        effective_date='2026-01-01',  # type: ignore[attr-defined]
        expiration_date='2027-01-01',  # type: ignore[attr-defined]
        lob=LOB.WC,
        exposure_base=qa.book.PAYROLL,  # type: ignore[attr-defined]
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

    test_freq = DeterministicFrequency(3)  # type: ignore[attr-defined]
    test_sev = ConstantSeverity(100)

    test_inforce = Inforce(
        n_policies=5,
        terms=test_policy,
        frequency=test_freq,
        severity=test_sev,
        name="Test Inforce"
    )

    test_portfolio = test_inforce + test_inforce

    expected_str = """Bucket: Test Inforce
- Number of Policies: 5
- Frequency Model: DeterministicFrequency(value=3)
- Severity Model: ConstantSeverity(value=100)
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


@pytest.mark.skip(reason="Need to implement this test.")
def test_inforce():
    test_policy = PolicyTerms(
        effective_date='2026-01-01',  # type: ignore[attr-defined]
        expiration_date='2027-01-01',  # type: ignore[attr-defined]
        lob=LOB.WC,
        exposure_base=qa.book.PAYROLL,  # type: ignore[attr-defined]
        exposure_amount=100_000_000,
        retention_type="deductible",
        per_occ_retention=500_000,
        coverage="occ"
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

    # Test single random variable (one sample for inforce experience)
    sample = test_inforce.rvs()
    expected_series = pd.Series([100.0, 100.0, 100.0])
    expected_experience = (
        expected_series,
        expected_series,
        expected_series,
        expected_series,
        expected_series)
    pd.testing.assert_series_equal(
        sample,  # type: ignore[attr-defined]
        expected_experience)  # type: ignore[attr-defined]

    samples = test_inforce.rvs(3)
    expected_multiple = (expected_experience,
                         expected_experience,
                         expected_experience)
    for sample, expected in zip(samples, expected_multiple):
        pd.testing.assert_series_equal(
            sample,  # type: ignore[attr-defined]
            expected)  # type: ignore[attr-defined]
