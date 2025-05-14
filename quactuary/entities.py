"""
Objects representing insurance policy terms, policies and portfolios.
"""

from dataclasses import dataclass
from typing import Optional

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


@dataclass(frozen=True, slots=True)
class PolicyTerms:
    """
    Represents the terms of a policy, including limits, deductibles,
    coinsurance, and coverage type.

    Attributes:
    ----------
    - `per_occ_deductible`: Deductible amount per occurrence.
    - `coinsurance`: Coinsurance percentage (1.00 = 100%).
    - `per_occ_limit`: Limit amount per occurrence (optional).
    - `agg_limit`: Aggregate limit amount (optional).
    - TODO: add corridor deductibles.
    - `attachment`: Attachment point for excess of loss layers.
    - `coverage`: Type of coverage (e.g., "OCC", "CLAIMS‑MADE").
    - `policy_dates`: Dates of the policy (optional).
    - `reinstatements`: Reinstatement terms (optional).
    - `corridors`: Corridor terms (optional).
    - `lob`: Line of business (optional).
    - `name`: Name of the policy (optional).
    - `policy_number`: Policy number (optional).
    - `insurer`: Insurer name (optional).
    - `broker`: Broker name (optional).
    - `underwriter`: Underwriter name (optional).
    - `notes`: Additional notes (optional).
    - `status`: Status of the policy (e.g., "active", "inactive").
    - `effective_date`: Effective date of the policy (optional).
    - `expiration_date`: Expiration date of the policy (optional).
    - `premium`: Premium amount (optional).
    - `commission`: Commission percentage (optional).
    - `taxes`: Taxes applicable to the policy (optional).
    - `endorsements`: Endorsements applicable to the policy (optional).
    - `limits`: Limits applicable to the policy (optional).
    - `deductibles`: Deductibles applicable to the policy (optional).
    - `exclusions`: Exclusions applicable to the policy (optional).
    - `endorsements`: Endorsements applicable to the policy (optional).
    - `sublimits`: Sublimits applicable to the policy (optional).
    - `endorsements`: Endorsements applicable to the policy (optional).
    """
    per_occ_deductible: float = 0.0
    coinsurance:        float = 1.00          # 1 → 100% insurer share
    per_occ_limit:      Optional[float] = None
    agg_limit:          Optional[float] = None
    attachment:         float = 0.0         # for XoL layers
    coverage:           str = "OCC"       # OCC / CLAIMS‑MADE / etc.
    # TODO: policy dates, reinstatements, etc.
    # TODO: corridors
    # TODO: LoB


@dataclass(slots=True)
class Inforce:
    """
    Represents a single in-force policy or a bucket of policies
    with the same frequency and severity models.

    Attributes:
    ----------
    - `n_policies`: Number of policies in the bucket.
    - `freq`: FrequencyModel instance representing the frequency of claims.
    - `sev`: SeverityModel instance representing the severity of claims.
    - `terms`: PolicyTerms instance representing the terms of the policy.
    - `name`: Name of the bucket (optional).

    Methods:
    -------
    - `classical_sample`: Generates a Monte-Carlo aggregate loss sample for the policy.    
    """
    n_policies:       int
    freq:             FrequencyModel
    sev:              SeverityModel
    terms:            PolicyTerms
    name:             str = "Unnamed Bucket"

    # --- helper methods ---
    def classical_sample(self, n_sims: int = 100_000):
        """Monte-Carlo aggregate loss sample - handy for QA."""
        # sample frequency first, then severities, apply policy terms…


class Portfolio(list):
    """
    A collection of Inforce objects, representing a portfolio of policies.
    This class is a subclass of list, so it can be used like a list of Inforce objects.

    It also provides additional methods for portfolio management.
    Attributes:
    ----------
    - `total_policies`: Returns the total number of policies in the portfolio.
    - `__add__`: Combines two portfolios into one.

    Usage:
    --------
    `portfolio = qa.Portfolio([wc_inforce, gl_inforce])`

    Notes:
    -------
    Portfolio implements methods to aggregate all segments’ loss distributions
    (using the distributions module or calling out to the `aggregate` library for convolution).
    """

    def total_policies(self) -> int:
        pass

    def __add__(self, other):
        """
        Combine two portfolios into one.
        """
        pass
