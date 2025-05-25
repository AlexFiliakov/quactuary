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

try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    # Fallback to C locale if en_US.UTF-8 is not available
    locale.setlocale(locale.LC_ALL, 'C')


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
    expected_results = pd.Series(
        [expected_result]).repeat(7).reset_index(drop=True)
    pd.testing.assert_series_equal(samples, expected_results)


def test_policy_validation():
    """Test policy parameter validation in __post_init__."""
    # Test invalid date order
    with pytest.raises(ValueError, match="Effective date must be before expiration date"):
        PolicyTerms(
            effective_date=date(2024, 12, 31),
            expiration_date=date(2024, 1, 1)
        )

    # Test invalid coinsurance
    with pytest.raises(ValueError, match="Coinsurance must be between 0.0 and 1.0"):
        PolicyTerms(
            effective_date=date(2024, 1, 1),
            expiration_date=date(2024, 12, 31),
            coinsurance=1.5
        )

    # Test invalid retention type
    with pytest.raises(ValueError, match="Retention type must be 'deductible' or 'SIR'"):
        PolicyTerms(
            effective_date=date(2024, 1, 1),
            expiration_date=date(2024, 12, 31),
            retention_type="invalid"
        )

    # Test negative values
    with pytest.raises(ValueError, match="per_occ_retention cannot be negative"):
        PolicyTerms(
            effective_date=date(2024, 1, 1),
            expiration_date=date(2024, 12, 31),
            per_occ_retention=-1000
        )


def test_is_policy_active():
    """Test policy active date checking."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31)
    )

    # Test dates within policy period
    assert terms.is_policy_active(date(2024, 1, 1)) is True  # Effective date
    assert terms.is_policy_active(date(2024, 6, 15)) is True  # Mid-year
    assert terms.is_policy_active(
        date(2024, 12, 30)) is True  # Day before expiration

    # Test dates outside policy period
    assert terms.is_policy_active(
        date(2023, 12, 31)) is False  # Before effective
    assert terms.is_policy_active(
        date(2024, 12, 31)) is False  # Expiration date
    assert terms.is_policy_active(
        date(2025, 1, 1)) is False  # After expiration


def test_apply_retention():
    """Test retention application logic."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        retention_type="deductible",
        per_occ_retention=1000.0,
        agg_retention=5000.0
    )

    # Test simple per-occurrence retention
    net_loss, retained = terms.apply_retention(2500.0)
    assert net_loss == 1500.0
    assert retained == 1000.0

    # Test loss below retention
    net_loss, retained = terms.apply_retention(800.0)
    assert net_loss == 0.0
    assert retained == 800.0

    # Test array input
    losses = np.array([500.0, 1500.0, 3000.0])
    net_losses, retained_amounts = terms.apply_retention(losses)
    np.testing.assert_array_equal(net_losses, [0.0, 500.0, 2000.0])
    np.testing.assert_array_equal(retained_amounts, [500.0, 1000.0, 1000.0])

    # Test aggregate retention
    net_loss, retained = terms.apply_retention(
        2000.0, aggregate_losses_so_far=4500.0)
    # Only 500 left in aggregate retention
    assert retained < 1000.0  # Less than per-occurrence due to aggregate limit


def test_apply_limits():
    """Test limit application logic."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        per_occ_limit=10000.0,
        agg_limit=25000.0
    )

    # Test per-occurrence limit
    limited, excess = terms.apply_limits(15000.0)
    assert limited == 10000.0
    assert excess == 5000.0

    # Test within limits
    limited, excess = terms.apply_limits(8000.0)
    assert limited == 8000.0
    assert excess == 0.0

    # Test aggregate limit
    limited, excess = terms.apply_limits(
        10000.0, aggregate_paid_so_far=20000.0)
    assert limited == 5000.0  # Only 5000 left in aggregate
    assert excess == 5000.0

    # Test array input with aggregate limit
    losses = np.array([8000.0, 9000.0, 10000.0])
    limited_losses, excess_amounts = terms.apply_limits(
        losses, aggregate_paid_so_far=10000.0)
    # Aggregate limit is 25000, already paid 10000, so 15000 remaining
    # First loss: 8000 (total: 18000)
    # Second loss: 7000 partial (total: 25000, limit reached)
    # Third loss: 0
    assert limited_losses[0] == 8000.0
    assert limited_losses[1] == 7000.0
    assert limited_losses[2] == 0.0


def test_apply_coinsurance():
    """Test coinsurance application."""
    # Test with 20% coinsurance (insured pays 20%, insurer pays 80%)
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        coinsurance=0.2
    )

    assert terms.apply_coinsurance(10000.0) == 8000.0

    # Test with no coinsurance (insurer pays all)
    terms_no_coins = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        coinsurance=None
    )
    assert terms_no_coins.apply_coinsurance(10000.0) == 10000.0

    # Test array input
    losses = np.array([1000.0, 2000.0, 3000.0])
    insurer_shares = terms.apply_coinsurance(losses)
    np.testing.assert_array_equal(insurer_shares, [800.0, 1600.0, 2400.0])


def test_apply_xol_attachment():
    """Test excess-of-loss attachment application."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        attachment=100000.0
    )

    # Test loss above attachment
    assert terms.apply_xol_attachment(150000.0) == 50000.0

    # Test loss below attachment
    assert terms.apply_xol_attachment(80000.0) == 0.0

    # Test loss exactly at attachment
    assert terms.apply_xol_attachment(100000.0) == 0.0

    # Test array input
    losses = np.array([50000.0, 100000.0, 150000.0, 200000.0])
    excess_losses = terms.apply_xol_attachment(losses)
    np.testing.assert_array_equal(excess_losses, [0.0, 0.0, 50000.0, 100000.0])


def test_apply_policy_terms_comprehensive():
    """Test complete policy term application."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        retention_type="deductible",
        per_occ_retention=1000.0,
        per_occ_limit=10000.0,
        corridor_retention=500.0,
        coinsurance=0.2  # 20% coinsurance
    )

    # Test standard loss
    result = terms.apply_policy_terms(15000.0)

    # Ground up: 15000
    # After retention: 15000 - 1000 = 14000
    # After limit: min(14000, 10000) = 10000
    # After corridor: 10000 - 500 = 9500
    # After coinsurance: 9500 * 0.8 = 7600

    assert result.ground_up_loss == 15000.0
    assert result.retained_amount == 1000.0
    assert result.loss_after_retention == 14000.0
    assert result.limited_loss == 10000.0
    assert result.amount_over_limit == 4000.0
    assert result.corridor_retained == 500.0
    assert result.insurer_share == 7600.0
    assert result.insured_share == 1000.0 + 500.0 + \
        1900.0  # retention + corridor + coinsurance


def test_apply_policy_terms_xol():
    """Test policy terms with XoL attachment."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        attachment=100000.0,
        per_occ_limit=50000.0,
        retention_type="SIR",
        per_occ_retention=5000.0
    )

    # Test loss above attachment
    result = terms.apply_policy_terms(150000.0)

    # Ground up: 150000
    # After attachment: 150000 - 100000 = 50000
    # After SIR: 50000 - 5000 = 45000
    # After limit: min(45000, 50000) = 45000

    assert result.ground_up_loss == 150000.0
    assert result.excess_attachment == 100000.0
    assert result.loss_after_attachment == 50000.0
    assert result.retained_amount == 5000.0
    assert result.loss_after_retention == 45000.0
    assert result.limited_loss == 45000.0
    assert result.insurer_share == 45000.0


def test_exposure_methods():
    """Test exposure-related methods."""
    terms = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        exposure_base=book.PAYROLL,
        exposure_amount=1000000.0,
        lob=LOB.WC
    )

    # Test get_exposure_info
    info = terms.get_exposure_info()
    assert info['exposure_base_name'] == 'Payroll'
    assert info['exposure_base_unit'] == 'USD'
    assert info['exposure_amount'] == 1000000.0
    assert info['lob'] == LOB.WC
    assert info['lob_name'] == "Workers' Compensation"

    # Test calculate_rate_per_unit
    rate = terms.calculate_rate_per_unit(5000.0)
    assert rate == 5.0  # $5 per $1000 of payroll

    # Test with non-monetary exposure
    terms_vehicles = PolicyTerms(
        effective_date=date(2024, 1, 1),
        expiration_date=date(2024, 12, 31),
        exposure_base=book.VEHICLES,
        exposure_amount=100.0
    )
    rate = terms_vehicles.calculate_rate_per_unit(5000.0)
    assert rate == 50.0  # $50 per vehicle


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
    expected_results = pd.Series(
        [expected_result]).repeat(7).reset_index(drop=True)
    pd.testing.assert_series_equal(samples, expected_results)
