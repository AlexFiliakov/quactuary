"""
Insurance entity models for policy terms, inforce buckets, and portfolios.

This module defines the core business objects for representing insurance policy
terms (`PolicyTerms`), aggregation buckets (`Inforce`), and collections of buckets (`Portfolio`).

Examples:
    >>> from quactuary.entities import PolicyTerms, Inforce, Portfolio
    >>> terms = PolicyTerms(per_occ_deductible=100.0, coinsurance=0.8)
    >>> bucket = Inforce(n_policies=50, freq=freq_model, sev=sev_model, terms=terms)
    >>> portfolio = Portfolio([bucket])
"""

from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


@dataclass(frozen=True, slots=True)
class ExposureBase:
    """
    Base class for exposure definitions.

    Attributes:
        name (str): Name of the exposure base.
        description (str): Description of the exposure base.
    """
    name: str
    unit: str
    description: str


# Predefined exposure bases
PAYROLL = ExposureBase(name="Payroll", unit="USD",
                       description="Total payroll amount.")
SALES = ExposureBase(name="Sales", unit="USD",
                     description="Total sales amount.")
SQUARE_FOOTAGE = ExposureBase(
    name="Square Footage", unit="sq ft", description="Total square footage.")
VEHICLES = ExposureBase(name="Vehicles", unit="count",
                        description="Total number of vehicles.")
REPLACEMENT_VALUE = ExposureBase(
    name="Replacement Value", unit="USD", description="Total replacement value.")
NON_INFLATIONARY_UNIT = ExposureBase(
    name="Non-Inflationary Unit", unit="unit", description="Non-inflationary unit of measure.")


class LOB(str, Enum):
    """
    Enumeration of lines of business (LOB) for insurance policies.
    """
    GLPL = "General and Product Liability"
    GL = "General Liability"
    PL = "Product Liability"
    WC = "Workers' Compensation"
    CAuto = "Auto Liability"
    EL = "Employers' Liability"
    D_O = "Directors and Officers Liability"
    E_O = "Errors and Omissions Liability"
    CPP = "Commercial Property Package"
    Cyber = "Cyber Liability"
    EPLI = "Employment Practices Liability"
    Umbrella = "Umbrella Liability"
    PAuto = "Personal Auto"
    PProperty = "Personal Property"
    PPL = "Personal Property Liability"
    PGL = "Personal General Liability"
    PWC = "Personal Watercraft"
    PWCPL = "Personal Watercraft Liability"


@dataclass(frozen=True, slots=True)
class PolicyTerms:
    """
    Terms defining insurance policy layer details.

    Attributes:
        per_occ_deductible (float): Deductible per occurrence.
        coinsurance (float): Insurer share proportion (1.0 = 100%).
        per_occ_limit (Optional[float]): Per-occurrence limit.
        agg_limit (Optional[float]): Aggregate limit.
        attachment (float): Attachment point for excess-of-loss.
        coverage (str): Coverage type (e.g., "OCC", "CLAIMS-MADE").
    """
    effective_date:     date  # Effective date of the policy
    expiration_date:    date  # Expiration date of the policy
    lob:                Optional[LOB] = None  # Line of business (optional)
    # Exposure base (Payroll, Sales, Square Footage, Vehicles, Replacement Value etc.)
    exposure_base:      Optional[ExposureBase] = None
    # Exposure amount (e.g., limit, sum insured)
    exposure_amount:    float = 0.0
    retention_type:     str = "deductible"  # deductible / SIR
    per_occ_retention:  float = 0.0  # retention per occurrence
    agg_retention:      Optional[float] = None  # aggregate retention
    corridor_retention: Optional[float] = None  # corridor retention
    # 0 → 0% insured's share, 1.0 → 100% insured's share
    coinsurance:        Optional[float] = None
    # per-occurrence limit (None if unlimited)
    per_occ_limit:      Optional[float] = None
    agg_limit:          Optional[float] = None  # aggregate limit (Optional)
    attachment:         float = 0.0  # for XoL layers
    coverage:           str = "occ"  # occ (occurrence) / cm (claims-made)
    notes:              str = ""  # Additional notes (ad hoc)
    # ...additional fields defined as needed...
    # TODO: reinstatements: Reinstatement terms (optional).
    # TODO: corridors: Corridor terms (optional).
    # TODO: premium: Premium amount (optional).
    # TODO: commission: Commission percentage (optional).
    # TODO: taxes: Taxes applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: exclusions: Exclusions applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: sublimits: Sublimits applicable to the policy (optional).


# TODO: Add a Reinsurance policy dataclass.

@dataclass(slots=True)
class Inforce:
    """
    Bucket of policies sharing frequency and severity characteristics.

    Args:
        n_policies (int): Number of policies in this bucket.
        freq (FrequencyModel): Claim count distribution.
        sev (SeverityModel): Claim severity distribution.
        terms (PolicyTerms): Policy terms and layer definitions.
        name (str, optional): Bucket label. Defaults to "Unnamed Bucket".

    Methods:
        classical_sample(n_sims: int) -> np.ndarray: Monte Carlo aggregate loss sample.
    """
    n_policies:       int
    terms:            PolicyTerms
    frequency:             FrequencyModel
    severity:              SeverityModel
    name:             str = "Unnamed Bucket"

    def __len__(self):
        """
        Get the number of policies in this bucket.

        Returns:
            int: Number of policies in the bucket.

        Examples:
            >>> len(bucket)
        """
        return self.n_policies


class Portfolio(list):
    """
    Portfolio of in-force buckets.

    Extends `list` to include portfolio-level operations.

    Examples:
        >>> p1 = Portfolio([bucket1])
        >>> p2 = Portfolio([bucket2])
        >>> combined = p1 + p2
    """

    def __init__(self, buckets: list[Inforce]):
        """
        Initialize the portfolio with a list of in-force buckets.

        Args:
            buckets (list[Inforce]): List of `Inforce` objects.

        Examples:
            >>> portfolio = Portfolio([gl_inforce, wc_inforce])
        """
        super().__init__(buckets)

    def __add__(self, other):
        """
        Merge this portfolio with another.

        Args:
            other (Portfolio): Another portfolio to merge.

        Returns:
            Portfolio: New portfolio containing buckets from both.

        Examples:
            >>> combined = portfolio1 + portfolio2
            >>> len(combined)
        """
        return Portfolio(list(self) + list(other))

    def __len__(self):
        """
        Get the number of buckets in the portfolio.

        Returns:
            int: Number of `Inforce` objects in the portfolio.

        Examples:
            >>> len(portfolio)
        """
        return self.total_policies()

    def total_policies(self) -> int:
        """
        Compute the total number of policies in the portfolio.

        Returns:
            int: Sum of `n_policies` across all buckets.

        Examples:
            >>> portfolio.total_policies()
        """
        # Sum n_policies from each Inforce in the portfolio
        return sum(bucket.n_policies for bucket in self)
