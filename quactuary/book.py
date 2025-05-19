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

import locale
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel

locale.setlocale(locale.LC_ALL, '')


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

    def __str__(self):
        return self.value


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
    attachment:         Optional[float] = None  # for XoL layers
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

    def __str__(self) -> str:
        output = f"Effective Date: {self.effective_date}\n" + \
            f"Expiration Date: {self.expiration_date}\n"
        if self.lob:
            output += f"LoB: {self.lob}\n"
        if self.exposure_base:
            output += f"Exposure Base: {self.exposure_base}\n"
        output += f"Exposure Amount: {self.exposure_amount:n}\n"
        output += f"Retention Type: {self.retention_type}\n"
        output += f"Per-Occurrence Retention: {self.per_occ_retention:n}\n"
        if self.agg_retention:
            output += f"Aggregate Retention: {self.agg_retention:n}\n"
        if self.corridor_retention:
            output += f"Corridor Retention: {self.corridor_retention:n}\n"
        if self.coinsurance:
            output += f"Coinsurance: {self.coinsurance:.2%}\n"
        if self.per_occ_limit:
            output += f"Per Occurrence Limit: {self.per_occ_limit:n}\n"
        if self.agg_limit:
            output += f"Aggregate Limit: {self.agg_limit:n}\n"
        if self.attachment:
            output += f"Attachment: {self.attachment:n}\n"
        output += f"Coverage: {self.coverage}\n"
        output += f"Notes: {self.notes}\n"
        return output

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
    frequency:        FrequencyModel
    severity:         SeverityModel
    name:             str = "Unnamed Bucket"

    def __add__(self, other):
        """
        Merge this bucket with another into a Portfolio.

        Args:
            other (Inforce): Another in-force bucket to merge.

        Returns:
            Inforce: New in-force bucket containing policies from both.

        Examples:
            >>> combined = bucket1 + bucket2
            >>> len(combined)
        """
        return Portfolio([self, other])

    def __len__(self):
        """
        Get the number of policies in this bucket.

        Returns:
            int: Number of policies in the bucket.

        Examples:
            >>> len(bucket)
        """
        return self.n_policies

    def __str__(self) -> str:
        output = f"Bucket: {self.name}\n" + \
            f"- Number of Policies: {self.n_policies:n}\n" + \
            f"- Frequency Model: {str(self.frequency)}\n" + \
            f"- Severity Model: {str(self.severity)}\n" + \
            f"- Policy Terms:\n"
        for pol_detail in str(self.terms).splitlines():
            output += f"  - {pol_detail}\n"
        return output

    def rvs(self, n_sims: int = 1) -> tuple:
        """
        Generate random variates from frequency and severity, returning aggregate losses.

        Args:
            n_sims (int): Number of simulations to run.

        Returns:
            tuple: Tuple containing frequency and severity samples.

        Examples:
            >>> bucket.rvs(n_sims=1_000)
        """
        if n_sims == 1:
            # 1 simulation: one tuple of severity values, one per policy
            freq_samples = [self.frequency.rvs()
                            for _ in range(self.n_policies)]
            return tuple(self.severity.rvs(count)  # type: ignore[attr-defined]
                         for count in freq_samples)
        else:
            # Multiple simulations: a tuple of tuples
            all_sims: list[tuple] = []
            for _ in range(n_sims):
                freq_samples = [self.frequency.rvs()
                                for _ in range(self.n_policies)]
                sev_samples = tuple(self.severity.rvs(count)  # type: ignore[attr-defined]
                                    for count in freq_samples)
                all_sims.append(sev_samples)
            return tuple(all_sims)


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
        Merge this portfolio with another or add an in-force bucket.

        Args:
            other (Portfolio or Inforce): Another portfolio or an in-force bucket to merge.

        Returns:
            Portfolio: New portfolio containing buckets from both.

        Examples:
            >>> combined = portfolio1 + portfolio2
            >>> combined = portfolio + inforce_bucket
            >>> len(combined)
        """
        if isinstance(other, Portfolio):
            return Portfolio(list(self) + list(other))
        elif hasattr(other, 'n_policies'):
            return Portfolio(list(self) + [other])
        else:
            return NotImplemented

    def __len__(self):
        """
        Get the number of buckets in the portfolio.

        Returns:
            int: Number of `Inforce` objects in the portfolio.

        Examples:
            >>> len(portfolio)
        """
        return self.total_policies()

    def __str__(self) -> str:
        output = ""
        for bucket in self:
            output += str(bucket)
        return output

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

    def rvs(self, n_sims: int = 1) -> tuple:
        """
        Generate random variates from all buckets in the portfolio.

        Args:
            n_sims (int): Number of simulations to run.

        Returns:
            tuple: Tuple containing frequency and severity samples for each bucket.

        Examples:
            >>> portfolio.rvs(n_sims=1_000)
        """
        bucket_sims = [bucket.rvs(n_sims) for bucket in self]
        if n_sims == 1:
            # 1 simulation: one tuple of severity values, one per policy
            return tuple(bucket_sims)
        else:
            # Multiple simulations: a tuple of tuples
            grouped_by_sim = []
            for i in range(n_sims):
                grouped_by_sim.append(
                    tuple(bucket[i] for bucket in bucket_sims))
            return tuple(grouped_by_sim)
