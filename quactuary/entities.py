from dataclasses import dataclass
from typing import Optional

from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


@dataclass(frozen=True, slots=True)
class PolicyTerms:
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
