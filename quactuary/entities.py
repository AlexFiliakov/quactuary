from dataclasses import dataclass

from quactuary.core.policy import PolicyTerms
from quactuary.distributions.frequency import FrequencyModel
from quactuary.distributions.severity import SeverityModel


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
    """

    def total_policies(self) -> int: ...
    def __add__(self, other): ...      # combine portfolios
