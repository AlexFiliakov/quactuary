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
from typing import Optional

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


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
    per_occ_deductible: float = 0.0
    coinsurance:        float = 1.00          # 1 → 100% insurer share
    per_occ_limit:      Optional[float] = None
    agg_limit:          Optional[float] = None
    attachment:         float = 0.0         # for XoL layers
    coverage:           str = "OCC"       # OCC / CLAIMS-MADE / etc.
    # ...additional fields defined as needed...
    # TODO: policy_dates: Dates of the policy (optional).
    # TODO: reinstatements: Reinstatement terms (optional).
    # TODO: corridors: Corridor terms (optional).
    # TODO: lob: Line of business (optional).
    # TODO: name: Name of the policy (optional).
    # TODO: policy_number: Policy number (optional).
    # TODO: insurer: Insurer name (optional).
    # TODO: broker: Broker name (optional).
    # TODO: underwriter: Underwriter name (optional).
    # TODO: notes: Additional notes (optional).
    # TODO: status: Status of the policy (e.g., "active", "inactive").
    # TODO: effective_date: Effective date of the policy (optional).
    # TODO: expiration_date: Expiration date of the policy (optional).
    # TODO: premium: Premium amount (optional).
    # TODO: commission: Commission percentage (optional).
    # TODO: taxes: Taxes applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: limits: Limits applicable to the policy (optional).
    # TODO: deductibles: Deductibles applicable to the policy (optional).
    # TODO: exclusions: Exclusions applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).
    # TODO: sublimits: Sublimits applicable to the policy (optional).
    # TODO: endorsements: Endorsements applicable to the policy (optional).


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
    freq:             FrequencyModel
    sev:              SeverityModel
    terms:            PolicyTerms
    name:             str = "Unnamed Bucket"

    # --- helper methods ---
    def classical_sample(self, n_sims: int = 100_000):
        """
        Generate Monte Carlo aggregate loss samples for the bucket.

        Args:
            n_sims (int, optional): Number of simulation runs. Defaults to 100000.

        Returns:
            np.ndarray: Aggregate loss per simulation.

        Examples:
            >>> samples = bucket.classical_sample(n_sims=50000)
        """
        # Placeholder: implement sampling logic here
        raise NotImplementedError("classical_sample is not yet implemented.")


class Portfolio(list):
    """
    Portfolio of in-force buckets.

    Extends `list` to include portfolio-level operations.

    Examples:
        >>> p1 = Portfolio([bucket1])
        >>> p2 = Portfolio([bucket2])
        >>> combined = p1 + p2
    """

    def total_policies(self) -> int:
        """
        Compute the total number of policies in the portfolio.

        Returns:
            int: Sum of `n_policies` across all buckets.

        Examples:
            >>> portfolio.total_policies()
            1500
        """
        # Sum n_policies from each Inforce in the portfolio
        return sum(bucket.n_policies for bucket in self)

    def __add__(self, other):
        """
        Merge this portfolio with another.

        Args:
            other (Portfolio): Another portfolio to merge.

        Returns:
            Portfolio: New portfolio containing buckets from both.

        Examples:
            >>> combined = p1 + p2
        """
        return Portfolio(list(self) + list(other))
