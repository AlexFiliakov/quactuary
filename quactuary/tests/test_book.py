import numpy as np
import pandas as pd
import pytest

import quactuary as qa
from quactuary.book import LOB, Inforce, PolicyTerms
from quactuary.distributions.frequency import DeterministicFrequency
from quactuary.distributions.severity import ConstantSeverity


@pytest.mark.skip(reason="Need to implement this test.")
def test_inforce():
    test_policy = PolicyTerms(
        effective_date='2026-01-01',
        expiration_date='2027-01-01',
        lob=LOB.WC,
        exposure_base=qa.book.PAYROLL,
        exposure_amount=100_000_000,
        retention_type="deductible",
        per_occ_retention=500_000,
        coverage="occ"
    )

    test_freq = DeterministicFrequency(3)
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
    pd.testing.assert_series_equal(sample, expected_experience)

    samples = test_inforce.rvs(3)
    expected_multiple = (expected_experience,
                         expected_experience,
                         expected_experience)
    for sample, expected in zip(samples, expected_multiple):
        pd.testing.assert_series_equal(sample, expected)
